"""
noc_lora.py — the NoC COUNT of train_set_transformer.py: the frozen backbone plus a count-only
LoRA r8 (models/set_transformer.py :: enable_noc_lora), fit in a COMBINATION-DISJOINT K-fold over the
real test set, so every profile is counted by a LoRA that never saw its donor combination.

The ID pass never reads the LoRA and no backbone weight receives gradient, so ID cannot move; the
trainer's [ID GUARD] checks that numerically. Chosen 2026-09-11 over a full clone and two adapter
placements by a rule fixed before that run (5 seeds, deepNoC protocol: macro F1 0.9505 +/- 0.003);
the alternatives that lost are in doc/failed_experiments.md. Split out of train_set_transformer.py
unchanged.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score


class InvariantCountHead(nn.Module):
    """Count from `gate` WITHOUT donor identity.

    gate[i] is donor i (slot i is initialised from donor i's genotype), so the model's own
    `noc_head = Linear(45, 5)` can learn "columns 17 and 29 are lit -> 4 people", i.e. memorise the
    donor combination instead of counting. Sorting the 45 values descending destroys that axis: what
    survives is how many slots are lit and how strongly, which is what counting needs. Sum and the
    number of slots >= 0.5 are appended because they are what the zero-shot readout already uses.
    The encoder still receives gradient through gate -> slots -> H, so this removes the identity
    channel, not the fine-tuning.
    """

    MAC_DIM = 19          # _mac_feats_torch: per-locus allele counts at 4 RFU cuts + 3 height stats

    def __init__(self, n_noc: int = 5, topm: int = 12, d_hidden: int = 32, n_feat: int = None,
                 use_probs: bool = True, use_mac: bool = False):
        super().__init__()
        self.topm = topm
        self.use_probs = bool(use_probs)
        self.use_mac = bool(use_mac)
        topm2 = topm + 2
        d_in = topm2 * (2 if self.use_probs else 1) + (self.MAC_DIM if self.use_mac else 0)
        # A FIXED affine scaler calibrated once on the fold's training rows, NOT LayerNorm across the
        # feature axis: LayerNorm removes each sample's scale, and the scale (how much total gate mass
        # there is) IS the count signal. The pre-LoRA NocCountHead made this same point.
        self.register_buffer("feat_mean", torch.zeros(d_in))
        self.register_buffer("feat_std", torch.ones(d_in))
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, n_noc))

    def _sorted(self, v: torch.Tensor) -> torch.Tensor:
        s = torch.sort(v, dim=1, descending=True).values[:, :self.topm]        # drops the donor axis
        return torch.cat([s, v.sum(1, keepdim=True), (v >= 0.5).to(v.dtype).sum(1, keepdim=True)], 1)

    def feats(self, gate: torch.Tensor, probs: torch.Tensor,
              mac: torch.Tensor | None = None) -> torch.Tensor:
        """Identity-blind views of the profile. Sorting removes WHO from each, but NOT where the numbers
        came from, and that distinction is the whole problem:

          `gate`  = AdaSlot slot existence. Independent of the ID posterior.
          `probs` = sigmoid(logits_cls), the ID posterior. The STRONGER feature by far (gate alone
                    ~0.49 macro F1 on fold 1, the probability profile ~0.88) — which is exactly why
                    lora_inv collapses to .198 on rows holding a donor OUTSIDE the panel, where the ID
                    cannot be right, while the pure-gate gate_cal holds .630 (muc 22). Sorting removed
                    the donor axis; it did not remove the dependence on ID VALUES.
          `mac`   = _mac_feats_torch: per-locus allele counts at RFU 0/50/150/500 + log-height stats.
                    Physical peak evidence (PACE, Marciano & Adelman 2017; deepNoC 2024). Carries NO ID
                    and NO donor identity, so it is the one channel that cannot fail when the ID fails.

        use_probs=False drops the ID posterior entirely; use_mac=True adds the physical channel. Both
        live in the LoRA, so the backbone stays frozen and fp(logits_cls) is byte-identical either way.
        """
        f = [self._sorted(gate)]
        if self.use_probs:
            f.append(self._sorted(probs))
        if self.use_mac:
            assert mac is not None, "use_mac=True needs the MAC features"
            f.append(mac)
        return torch.cat(f, dim=1)

    @torch.no_grad()
    def calibrate(self, X: torch.Tensor) -> None:
        self.feat_mean.copy_(X.mean(0)); self.feat_std.copy_(X.std(0).clamp(min=1e-6))

    def forward(self, gate: torch.Tensor, probs: torch.Tensor,
                mac: torch.Tensor | None = None) -> torch.Tensor:
        return self.net((self.feats(gate, probs, mac) - self.feat_mean) / self.feat_std)


class _NocLoraView(nn.Module):
    """Only noc_lora.* trains; a call runs the NoC pass alone.

    head=None keeps the model's own noc_head (a Linear over the 45 gate values IN DONOR ORDER, which
    is what lets a fine-tuned counter memorise combinations). Passing an InvariantCountHead instead
    reads the SORTED gate and sorted probability profile of the adapted pass, so the count cannot see
    which donors are lit — the same change that took the clone from 0.8031 to 0.8665 on fold 1.
    """

    def __init__(self, m, head=None):
        super().__init__()
        self.m = m
        for n, p in m.named_parameters():
            p.requires_grad = n.startswith("noc_lora.")
        self.invariant = head is not None
        self.noc_head = head if self.invariant else m.noc_lora.head

    def lora_params(self):
        return [p for n, p in self.m.named_parameters() if n.startswith(("noc_lora.A.", "noc_lora.B."))]

    def forward(self, tokens, mask):
        if not self.invariant:
            return self.m(tokens, mask, noc_only=True)["logits_noc"]
        o = self.m.noc_pass(tokens, mask)
        # _mac_feats_torch is already @torch.no_grad(): a FIXED physical feature, no gradient, no ID.
        mac = self.m._mac_feats_torch(tokens, mask) if getattr(self.noc_head, "use_mac", False) else None
        return self.noc_head(o["gate"], torch.sigmoid(o["logits_cls"]), mac)


@torch.no_grad()
def _lora_predict(net, T, M, rows, dev):
    out = np.zeros((len(rows), 5))
    for i in range(0, len(rows), 256):
        sl = rows[i:i + 256]
        out[i:i + 256] = net(torch.from_numpy(T[sl]).to(dev),
                             torch.from_numpy(M[sl]).to(dev)).softmax(1).cpu().numpy()
    return out


def fold_record(fi, groups, tr, te, y, kt, tr_loss, tr_rec, tr_f1):
    """Train-pool vs held-out numbers of one count fold. Held-out F1 is over the NOC classes present."""
    hf = float(f1_score(y[te], kt, average="macro", labels=sorted(set(y[te].tolist())), zero_division=0))
    rec = {"fold": fi, "n_train": int(len(tr)), "combos_train": int(len(np.unique(groups[tr]))),
           "n_heldout": int(len(te)), "combos_heldout": int(len(np.unique(groups[te]))),
           "train_loss": round(tr_loss, 4), "train_f1": round(tr_f1, 4),
           "train_recall": [None if r != r else round(r, 4) for r in tr_rec],
           "heldout_acc": round(float((kt == y[te]).mean()), 4), "heldout_f1": round(hf, 4),
           "gap": round(tr_f1 - hf, 4)}
    print(f"  [GAP] fold {fi}: train-pool macroF1 {tr_f1:.3f} (loss {tr_loss:.3f}) | held-out macroF1 {hf:.3f} "
          f"on NOC {sorted(set(y[te].tolist()))} | gap {tr_f1 - hf:+.3f}", flush=True)
    return rec


@torch.no_grad()
def _train_pool_report(net, T, M, Y, w, dev):
    """TRAIN rows only: loss, recall per NOC, macro F1. Held-out rows are never scored during training."""
    Y = np.asarray(Y)
    lg = []
    for i in range(0, len(T), 256):
        lg.append(net(torch.from_numpy(T[i:i + 256]).to(dev),
                      torch.from_numpy(M[i:i + 256]).to(dev)).float().cpu())
    lg = torch.cat(lg)
    ls = float(F.cross_entropy(lg, torch.from_numpy(Y - 1).long(), weight=w.float().cpu()))
    pred = lg.argmax(1).numpy() + 1
    rec = [float((pred[Y == k] == k).mean()) if (Y == k).any() else float("nan") for k in range(1, 6)]
    return ls, rec, float(f1_score(Y, pred, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0))


def _lora_fit(net, dl, w, ep, mon_at, T, M, Y, name, dev, head_lr=5e-5, skip_warm=False, val=None):
    """10 epochs count head only (lr 3e-4), then `ep` epochs: LoRA lr 5e-4, head `head_lr`, cosine decay,
    clip 1.0, class-weighted CE (recipe fixed 2026-09-11 before the arms ran).

    skip_warm: the head was already warmed up elsewhere (the invariant head, on cached features).
    val=(T, M, Y): rows of donor combinations held out of THIS fit. After every epoch the macro F1 on
    them is measured and the best state is kept; the returned net is that state. It stops the fit at
    the point where learning more about the training combinations stops helping on unseen ones —
    i.e. where memorising starts. No test row is ever involved."""
    if not skip_warm:
        opt = torch.optim.AdamW(net.noc_head.parameters(), lr=3e-4, weight_decay=1e-4)
        for _ in range(10):
            net.train()
            for tb, mb, nb in dl:
                ls = F.cross_entropy(net(tb.to(dev), mb.to(dev)), (nb - 1).to(dev), weight=w)
                opt.zero_grad(); ls.backward(); opt.step()
    trainable = [p for p in net.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": list(net.noc_head.parameters()), "lr": head_lr},
                             {"params": net.lora_params(), "lr": 5e-4}], weight_decay=1e-4)
    best = (-1.0, 0, None)

    def val_f1():
        net.eval()
        Tv, Mv, Yv = val
        pv = _lora_predict(net, Tv, Mv, np.arange(len(Tv)), dev).argmax(1) + 1
        return float(f1_score(Yv, pv, average="macro", labels=sorted(set(Yv.tolist())), zero_division=0))

    if val is not None:              # epoch 0 = the state before any LoRA step
        best = (val_f1(), 0, copy.deepcopy(net.state_dict()))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ep, eta_min=1e-6)
    last = 0.0
    for e in range(ep):
        net.train()
        tot, nbat = 0.0, 0
        for tb, mb, nb in dl:
            ls = F.cross_entropy(net(tb.to(dev), mb.to(dev)), (nb - 1).to(dev), weight=w)
            opt.zero_grad(); ls.backward()
            nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
            tot += ls.item(); nbat += 1
        sch.step()
        last = tot / max(nbat, 1)
        vf = None
        if val is not None:
            vf = val_f1()
            if vf > best[0]:
                best = (vf, e + 1, copy.deepcopy(net.state_dict()))
        if (e + 1) in mon_at:
            net.eval()
            lsr, rec, mf1 = _train_pool_report(net, T, M, Y, w, dev)
            print(f"       [TRAIN {name:<8}] ep {e + 1:>4}/{ep:<4} batch loss {last:.3f} | TRAIN pool: loss "
                  f"{lsr:.3f} macroF1 {mf1:.3f} recall NOC1..5 " + " ".join(f"{r:.2f}" for r in rec)
                  + (f" | inner-val macroF1 {vf:.3f}" if vf is not None else ""))
    if val is not None:
        net.load_state_dict(best[2])
        print(f"       [TRAIN {name:<8}] inner-val (unseen combinations) best macroF1 {best[0]:.3f} at epoch "
              f"{best[1]}/{ep} -> that state is kept", flush=True)
    net.eval()
    return last


def budget_rows(tr, y, groups, budget, seed, budget_seed, fi, K, tag):
    """The label budget: at most budget[k-1] rows of NOC k, drawn at random from the fit side `tr` only. Every
    count arm draws through here with the same seed, so arms compared at one budget fit on the SAME rows."""
    br = np.random.default_rng(int(seed) * 1000 + int(budget_seed) * 100 + fi)
    n_before = len(tr)
    tr = np.sort(np.concatenate([br.choice(tr[y[tr] == k], min(int(budget[k - 1]), int((y[tr] == k).sum())),
                                           replace=False) for k in range(1, 6)]))
    print(f"  [{tag}] fold {fi}/{K}: label budget {list(budget)} (draw {budget_seed}) -> fit on {len(tr)} "
          f"of {n_before} rows, NOC " + str({int(v): int((y[tr] == v).sum()) for v in range(1, 6)})
          + f", {len(np.unique(groups[tr]))} combos", flush=True)
    return tr


def noc_lora_kfold(model, tok, msk, noc, groups, K, steps, rank, seed, dev, open_tok=None, open_msk=None,
                   invariant=False, head_warm_ep=200, scope="all", epochs=None, head_lr=5e-5,
                   inner_val=False, fold_log=None, splits=None, use_probs=True, use_mac=False,
                   budget=None, budget_seed=0):
    """Each fold attaches a fresh LoRA to a frozen copy of the backbone, trains it on the other folds'
    rows and scores its held-out rows, so every row is counted by a model that never saw its donor
    combination. Open rows belong to no fold and get the mean of the K posteriors.
    Returns (k, P, P_open).
    fold_log: optional list; one dict per fold is appended (train-pool vs held-out macro F1, the
    memorisation gap), so the report can show where a count stops generalising.
    budget: optional 5 ints, the most labelled rows per NOC1..5 a fit may use. Rows are drawn at random from
    THAT fold's training side only; the held-out side is untouched. deepNoC fine-tuned on 34/87/79/93/78.
"""
    import copy
    from sklearn.model_selection import StratifiedGroupKFold
    y = np.clip(np.asarray(noc), 1, 5).astype(np.int64)
    folds = splits if splits is not None else list(
        StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed).split(tok, y - 1, groups=groups))
    K = len(folds)
    P = np.zeros((len(tok), 5))
    Po = np.zeros((len(open_tok), 5)) if open_tok is not None else None
    for fi, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        shared = set(groups[tr].tolist()) & set(groups[te].tolist())
        if shared and splits is None:
            raise SystemExit(f"[NOC LORA] fold {fi}: {len(shared)} combinations on both sides")
        if shared:      # only a caller-supplied split may be leaky, and it says so in the log
            print(f"  [NOC LORA] fold {fi}/{K}: {len(shared)} combinations on BOTH sides "
                  f"(caller-supplied split)", flush=True)
        tv = None
        if inner_val:   # hold out ~1/5 of THIS fold's training combinations to stop before memorising
            fit_i, val_i = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=int(seed) + fi)
                                .split(tr, y[tr] - 1, groups=groups[tr]))
            tr_val = tr[val_i]; tr = tr[fit_i]
            if set(groups[tr].tolist()) & set(groups[tr_val].tolist()):
                raise SystemExit(f"[NOC LORA] fold {fi}: inner val shares combinations with the fit rows")
            tv = (tok[tr_val], msk[tr_val], y[tr_val])
            print(f"  [NOC LORA] fold {fi}/{K}: inner val = {len(tr_val)} rows / "
                  f"{len(np.unique(groups[tr_val]))} combinations held out of the fit, NOC "
                  + str({int(v): int((y[tr_val] == v).sum()) for v in range(1, 6)}), flush=True)
        if budget is not None:      # label budget: the same number of real profiles deepNoC fine-tuned on
            tr = budget_rows(tr, y, groups, budget, seed, budget_seed, fi, K, "NOC LORA")
        w = 1.0 / np.sqrt(np.clip(np.bincount(y[tr], minlength=6)[1:], 1, None))
        w = torch.tensor(np.clip(w / w.mean(), 0.6, 1.6), dtype=torch.float32, device=dev)
        nbat = max(1, int(np.ceil(len(tr) / 64)))
        ep = int(epochs) if epochs else max(1, int(round(steps / nbat)))
        mon = sorted({1, max(1, round(0.5 * ep)), ep})
        g = torch.Generator().manual_seed(int(seed) * 100 + fi)
        dl = DataLoader(torch.utils.data.TensorDataset(
            torch.from_numpy(tok[tr]), torch.from_numpy(msk[tr]), torch.from_numpy(y[tr])),
            batch_size=64, shuffle=True, generator=g)
        m = copy.deepcopy(model).to(dev).enable_noc_lora(rank, scope=scope)
        head = None
        if invariant:
            head = InvariantCountHead(use_probs=use_probs, use_mac=use_mac).to(dev)
            with torch.no_grad():       # B starts at 0, so the adapted pass IS the backbone pass here
                _Xw = []
                for i in range(0, len(tr), 256):
                    _tb = torch.from_numpy(tok[tr][i:i + 256]).to(dev)
                    _mb = torch.from_numpy(msk[tr][i:i + 256]).to(dev)
                    _oi = m.noc_pass(_tb, _mb)
                    _Xw.append(head.feats(_oi["gate"], torch.sigmoid(_oi["logits_cls"]),
                                          m._mac_feats_torch(_tb, _mb) if use_mac else None))
                Xw = torch.cat(_Xw)
            head.calibrate(Xw)
            Xn = (Xw - head.feat_mean) / head.feat_std
            Yw = torch.from_numpy(y[tr] - 1).to(dev)
            optw = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
            for _ in range(head_warm_ep):      # cheap: the features are cached, the encoder is idle
                head.train()
                perm = torch.randperm(len(Xn), device=dev)
                for i in range(0, len(Xn), 64):
                    ix = perm[i:i + 64]
                    ls = F.cross_entropy(head.net(Xn[ix]), Yw[ix], weight=w)
                    optw.zero_grad(); ls.backward(); optw.step()
            with torch.no_grad():
                _acc = float((head.net(Xn).argmax(1).cpu().numpy() + 1 == y[tr]).mean())
            print(f"  [NOC LORA] fold {fi}/{K}: invariant head warm-up on cached features, "
                  f"{head_warm_ep} ep -> train acc {_acc:.4f}", flush=True)
        view = _NocLoraView(m, head)
        ntr = sum(p.numel() for p in view.parameters() if p.requires_grad)
        print(f"  [NOC LORA] fold {fi}/{K}: train {len(tr)} rows / {len(np.unique(groups[tr]))} combos | "
              f"held-out {len(te)} rows / {len(np.unique(groups[te]))} combos, NOC "
              + str({int(v): int((y[te] == v).sum()) for v in range(1, 6)})
              + f" | trainable {ntr:,} | {ep} epochs x {nbat} batches ~ {ep * nbat} steps", flush=True)
        _lora_fit(view, dl, w, ep, mon, tok[tr], msk[tr], y[tr], f"lora f{fi}", dev,
                  head_lr=head_lr, skip_warm=invariant, val=tv)
        P[te] = _lora_predict(view, tok, msk, te, dev)
        if Po is not None:
            Po += _lora_predict(view, open_tok, open_msk, np.arange(len(open_tok)), dev)
        kt = P[te].argmax(1) + 1
        if fold_log is not None:
            fold_log.append(fold_record(fi, groups, tr, te, y, kt, *_train_pool_report(view, tok[tr], msk[tr], y[tr], w, dev)))
        print(f"  [NOC LORA] fold {fi}/{K}: held-out acc {float((kt == y[te]).mean()):.4f} | recall NOC1..5 "
              + " ".join(f"{float((kt[y[te] == v] == v).mean()) if (y[te] == v).any() else float('nan'):.2f}"
                         for v in range(1, 6)) + f" | {time.time() - t0:.0f}s", flush=True)
        del m, view
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return (P.argmax(1) + 1).astype(int), P, (Po / K if Po is not None else None)




# ── Count heads fit on FROZEN features: no LoRA, no backbone pass after the first ─────────────────────
# With a few hundred labels a smaller adaptation memorises less. All of them read features computed once from
# the frozen backbone (eval mode), so the ID pass is untouched by construction and a fit takes seconds.
#   card_cal   logits_card / T + b                                  6 params   (recalibrates the zero-shot head)
#   card_ft    copy of noc_head, from its own weights               230        (re-weights the 45 gate values)
#   head_only  InvariantCountHead: sorted gate + sorted ID probs    ~1.1k      (the LoRA arm's warm-up, no LoRA)
# and three that NEVER read the ID posterior, so a contributor outside the panel cannot pull the count with it:
#   head_free  InvariantCountHead: sorted gate only                 ~0.6k
#   head_mac   InvariantCountHead: sorted gate + 19 MAC physical    ~1.3k
#   head_unf   MLP on the reject pool (encoder over ALL peaks, no feas_filter) + MAC   ~4.8k
#              the only input that still sees the alleles of someone outside the panel
HEAD_ARMS = ("card_cal", "card_ft", "head_only", "head_free", "head_mac", "head_unf")
_INV_FEATS = {"head_only": dict(use_probs=True, use_mac=False), "head_free": dict(use_probs=False, use_mac=False),
              "head_mac": dict(use_probs=False, use_mac=True)}


@torch.no_grad()
def frozen_count_feats(model, tok, msk, dev, bs=256):
    """Per row of the frozen backbone: gate (N,45), sigmoid(logits_cls) (N,45), logits_card (N,5),
    MAC physical features (N,19) and the unfiltered reject pool z_rej (N,d)."""
    model.eval()
    G, Pr, C, MC, Z = [], [], [], [], []
    for i in range(0, len(tok), bs):
        t = torch.from_numpy(tok[i:i + bs]).to(dev); m = torch.from_numpy(msk[i:i + bs]).to(dev)
        o = model(t, m)
        G.append(o["gate"].float().cpu()); Pr.append(torch.sigmoid(o["logits_cls"]).float().cpu())
        C.append(o["logits_card"].float().cpu()); MC.append(model._mac_feats_torch(t, m).float().cpu())
        Z.append(model._reject_pool(t, m).float().cpu())
    return torch.cat(G), torch.cat(Pr), torch.cat(C), torch.cat(MC), torch.cat(Z)


class _CardCal(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(1)); self.b = nn.Parameter(torch.zeros(5))

    def forward(self, g, pr, c, mac, z):
        return c / torch.exp(self.log_t) + self.b


class _CardFt(nn.Module):
    def __init__(self, noc_head):
        super().__init__()
        self.lin = copy.deepcopy(noc_head).float().cpu()

    def forward(self, g, pr, c, mac, z):
        return self.lin(g)


class _Inv(nn.Module):
    """InvariantCountHead over sorted gate (+ sorted ID probs) (+ MAC); feature scaler fit on the fit rows."""
    def __init__(self, use_probs, use_mac):
        super().__init__()
        self.h = InvariantCountHead(use_probs=use_probs, use_mac=use_mac)

    def raw(self, g, pr, c, mac, z):
        return self.h.feats(g, pr, mac if self.h.use_mac else None)

    def forward(self, g, pr, c, mac, z):
        return self.h.net((self.raw(g, pr, c, mac, z) - self.h.feat_mean) / self.h.feat_std)


class _Unf(nn.Module):
    """MLP on [z_rej, MAC]: the encoder's view of EVERY peak (no panel filter) plus physical peak counts."""
    def __init__(self, d_z, d_hidden=32):
        super().__init__()
        d_in = d_z + InvariantCountHead.MAC_DIM
        self.register_buffer("feat_mean", torch.zeros(d_in)); self.register_buffer("feat_std", torch.ones(d_in))
        self.net = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(), nn.Linear(d_hidden, 5))

    def raw(self, g, pr, c, mac, z):
        return torch.cat([z, mac], 1)

    def forward(self, g, pr, c, mac, z):
        return self.net((self.raw(g, pr, c, mac, z) - self.feat_mean) / self.feat_std)


def head_kfold(model, tok, msk, noc, groups, K, seed, dev, kind, open_tok=None, open_msk=None, fold_log=None,
               splits=None, budget=None, budget_seed=0):
    """Same folds, same budget draws and same class weights as noc_lora_kfold, but the fit is a small head on
    frozen features (HEAD_ARMS). Returns (k, P, P_open) like noc_lora_kfold.
    Recipes, fixed before any run (2026-10-01): card_cal / card_ft full-batch AdamW, 300 steps, lr 0.05 / 1e-3;
    every other head = the LoRA arm's warm-up exactly (feature scaler on the fit rows, 200 epochs, batch 64,
    lr 3e-4, wd 1e-4)."""
    from sklearn.model_selection import StratifiedGroupKFold
    assert kind in HEAD_ARMS, kind
    y = np.clip(np.asarray(noc), 1, 5).astype(np.int64)
    folds = splits if splits is not None else list(
        StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed).split(tok, y - 1, groups=groups))
    K = len(folds)
    FT = frozen_count_feats(model, tok, msk, dev)
    Fo = frozen_count_feats(model, open_tok, open_msk, dev) if open_tok is not None else None
    P = np.zeros((len(tok), 5))
    Po = np.zeros((len(open_tok), 5)) if open_tok is not None else None
    sub = lambda X, r: tuple(x[torch.from_numpy(np.asarray(r))] for x in X)
    for fi, (tr, te) in enumerate(folds, 1):
        t0 = time.time()
        if splits is None and set(groups[tr].tolist()) & set(groups[te].tolist()):
            raise SystemExit(f"[HEAD {kind}] fold {fi}: combinations on both sides")
        if budget is not None:
            tr = budget_rows(tr, y, groups, budget, seed, budget_seed, fi, K, f"HEAD {kind}")
        w = 1.0 / np.sqrt(np.clip(np.bincount(y[tr], minlength=6)[1:], 1, None))
        w = torch.tensor(np.clip(w / w.mean(), 0.6, 1.6), dtype=torch.float32)
        torch.manual_seed(int(seed) * 100 + fi)
        net = (_CardCal() if kind == "card_cal" else
               _CardFt(model.cls_decoder_module.noc_head) if kind == "card_ft" else
               _Unf(FT[4].shape[1]) if kind == "head_unf" else _Inv(**_INV_FEATS[kind]))
        Xtr = sub(FT, tr); Ytr = torch.from_numpy(y[tr] - 1)
        if kind in ("card_cal", "card_ft"):
            opt = torch.optim.AdamW(net.parameters(), lr=0.05 if kind == "card_cal" else 1e-3, weight_decay=0.0)
            for _ in range(300):
                ls = F.cross_entropy(net(*Xtr), Ytr, weight=w)
                opt.zero_grad(); ls.backward(); opt.step()
        else:
            with torch.no_grad():
                R = net.raw(*Xtr)
                tgt = net.h if kind != "head_unf" else net
                tgt.feat_mean.copy_(R.mean(0)); tgt.feat_std.copy_(R.std(0).clamp(min=1e-6))
            opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
            for _ in range(200):
                net.train()
                perm = torch.randperm(len(tr))
                for i in range(0, len(tr), 64):
                    ix = perm[i:i + 64]
                    ls = F.cross_entropy(net(*(x[ix] for x in Xtr)), Ytr[ix], weight=w)
                    opt.zero_grad(); ls.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            lg_tr = net(*Xtr)
            P[te] = torch.softmax(net(*sub(FT, te)), 1).numpy()
            if Po is not None:
                Po += torch.softmax(net(*Fo), 1).numpy()
        pred_tr = lg_tr.argmax(1).numpy() + 1
        tr_rec = [float((pred_tr[y[tr] == k] == k).mean()) if (y[tr] == k).any() else float("nan") for k in range(1, 6)]
        tr_f1 = float(f1_score(y[tr], pred_tr, average="macro", labels=[1, 2, 3, 4, 5], zero_division=0))
        kt = P[te].argmax(1) + 1
        if fold_log is not None:
            fold_log.append(fold_record(fi, groups, tr, te, y, kt, float(F.cross_entropy(lg_tr, Ytr, weight=w)),
                                        tr_rec, tr_f1))
        extra = (f" | T {float(torch.exp(net.log_t)):.3f} b " + " ".join(f"{v:+.2f}" for v in net.b.tolist())
                 if kind == "card_cal" else "")
        print(f"  [HEAD {kind}] fold {fi}/{K}: fit {len(tr)} rows | held-out acc {float((kt == y[te]).mean()):.4f}"
              f"{extra} | {time.time() - t0:.1f}s", flush=True)
    return (P.argmax(1) + 1).astype(int), P, (Po / K if Po is not None else None)
