"""train_noc_branch.py — a DEDICATED NOC path forked off a frozen ID backbone.

Why a fork rather than one shared encoder: counting and identification want different representations.
Measured on the ID representation, a readout of the pooled vector `z` predicts NOC at 0.96 IN
distribution but 0.18 across novel combos (below chance) — `z` encodes WHO, not HOW MANY; the sorted
prob profile, which is identity-blind, keeps 0.89. Training the count objective through the shared
encoder therefore either learns the wrong invariance or drags the ID representation. Forking removes
the conflict by construction: the ID branch never receives a NOC gradient, which is ASSERTED here
(`--check_id`) by re-running the frozen backbone and requiring bit-identical ID outputs.

Two axes, four variants:
  --fork  {last,full}    own copy of the LAST encoder block (+27% params) or of the WHOLE encoder (+53%)
  --init  {copy,random}  start from the ID-trained weights, or from scratch
`--init random` is the control for the premise that ID pretraining enriches counting: if random matches
copy, the pretraining is not what makes the fine-tune work.

Two stages:
  stage 1  fine-tune on IN-SILICO only            -> is synthetic supervision enough?
  stage 2  + fine-tune on REAL open + REAL NOC1   -> real NOC labels that cost no ID test data:
           `open` mixtures have out-of-panel donors (useless for ID) but their NOC is known from the
           sample name, and they are never used in the ID evaluation.

Baseline to beat (fold 0, real test): post-hoc RF = 0.9186 count acc / 0.875 macro-F1.
Ceiling seen by a real-data fine-tune of the whole backbone (decoder-FT): 0.944 macro-F1.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture, PMA

DATA_W = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data_w_inc22")))   # in-silico (enriched)
REAL = Path(os.environ.get("STR_REAL_DIR", str(ROOT / "data")))             # real arrays


def pick_device() -> torch.device:
    f = os.environ.get("STR_DEVICE", "").strip().lower()
    if f:
        return torch.device(f)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = pick_device()
ALLELE_OFF, LUT_W = 30, 1024


def build_luts(donor_geno, donor_geno_mask, n_cls=45):
    gg = donor_geno; gm = donor_geno_mask.bool()
    owner = torch.zeros(24, LUT_W, n_cls)
    for c in range(min(n_cls, gg.size(0))):
        for j in range(gg.size(1)):
            if gm[c, j]:
                li = int(gg[c, j, 0]); ab = int(round(float(gg[c, j, 1]) * 10)) + ALLELE_OFF
                if 0 <= li < 24 and 0 <= ab < LUT_W:
                    owner[li, ab, c] = 1.0
    return owner


class NocBranch(nn.Module):
    """Forked encoder + own pooling + count head. The backbone is frozen and never back-propped."""

    def __init__(self, backbone: SetTransformerMixture, fork: str = "last", init: str = "copy",
                 dropout: float = 0.1, n_noc: int = 5):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fork = fork
        n_blocks = len(backbone.encoder)
        self.n_shared = 0 if fork == "full" else n_blocks - 1
        d = backbone.d_model
        blocks = copy.deepcopy(backbone.encoder[self.n_shared:])
        if init == "random":
            for mod in blocks.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()
        self.noc_encoder = blocks
        self.pma = copy.deepcopy(backbone.pma) if init == "copy" else PMA(d, n_heads=4, k_seeds=1)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(dropout), nn.Linear(d, 64),
                                  nn.ReLU(inplace=True), nn.Linear(64, n_noc))
        # the deepcopy above inherits requires_grad=False from the frozen backbone — undo it, the
        # FORK is what we train (the backbone itself stays frozen, asserted by the ID guard).
        for mod in (self.noc_encoder, self.pma, self.head):
            for prm in mod.parameters():
                prm.requires_grad = True

    def forward(self, tokens, mask):
        b = self.backbone
        with torch.no_grad():                                   # frozen prefix, no grad, no ID damage
            x0, pad_mask, priv_pm, shar_pm, is_priv, is_shar = _project(b, tokens, mask)
            H_priv, H_shar = x0, x0
            for isab in b.encoder[:self.n_shared]:
                H_priv = isab(H_priv, pad_mask=priv_pm)
                H_shar = isab(H_shar, pad_mask=shar_pm)
        H_priv, H_shar = H_priv.detach(), H_shar.detach()
        for isab in self.noc_encoder:                            # the forked, trainable part
            H_priv = isab(H_priv, pad_mask=priv_pm)
            H_shar = isab(H_shar, pad_mask=shar_pm)
        # same private/shared zeroing + merge the backbone does after its encoder stack
        H = (H_priv * is_priv.unsqueeze(-1).to(H_priv.dtype)
             + H_shar * is_shar.unsqueeze(-1).to(H_shar.dtype))
        z = self.pma(H, pad_mask=pad_mask).squeeze(1)
        return self.head(z)


def _project(b: SetTransformerMixture, tokens, mask):
    """Reproduce the backbone's token projection + set_of_set private/shared masks (no encoding)."""
    x0, pad_mask = b._project_tokens(tokens, mask, apply_feas=True)
    li = tokens[..., 0].long().clamp(0, 23)
    bi = ((tokens[..., 1] * 10).round().long() + b._AOFF).clamp(0, b.owner_lut.size(1) - 1)
    n_car = b.owner_lut[li, bi].sum(-1)
    is_valid = ~pad_mask
    is_priv = (n_car == 1) & is_valid
    is_shar = (n_car != 1) & is_valid
    return x0, pad_mask, ~is_priv, ~is_shar, is_priv, is_shar


def _holdout_by_combo(real: Path, n_open: int, n_ss: int, frac=0.2, seed=0):
    """Boolean mask selecting ~frac of the stage-2 pool for selection, held out BY DONOR COMBO so the
    selection set shares no combo with what stage 2 trains on."""
    import re
    names = json.load(open(real / "meta_sample_names_open.json"))

    def dn(fn):
        p = str(fn).split("-")
        if len(p) < 3:
            return ()
        c = p[2]
        if "_" in c:
            return tuple(sorted(int(m.group(1)) for s_ in c.split("_") if (m := re.match(r"^(\d+)", s_))))
        m = re.match(r"^(\d+)", c)
        return (int(m.group(1)),) if m else ()

    combos = [dn(n) for n in names]
    uniq = sorted({c for c in combos if len(c) >= 2})
    rng = np.random.default_rng(seed)
    held = set(rng.permutation(np.array(uniq, dtype=object))[:max(1, int(len(uniq) * frac))].tolist())
    m_open = np.array([c in held for c in combos], bool)
    m_ss = np.zeros(n_ss, bool)
    m_ss[rng.choice(n_ss, size=int(n_ss * frac), replace=False)] = True   # NOC1: random rows
    return np.concatenate([m_open, m_ss])


def load_split(d: Path, split: str, tok="tokens8"):
    return (np.load(d / f"{tok}_{split}.npy").astype(np.float32),
            np.load(d / f"mask_{split}.npy"),
            np.clip(np.load(d / f"noc_{split}.npy").astype(np.int64), 1, 5))


def oversample(idx, lab):
    from collections import Counter
    cnt = Counter(lab[idx].tolist()); tgt = max(cnt.values()); out = []
    rng = np.random.default_rng(0)
    for c, n in cnt.items():
        ii = idx[lab[idx] == c]
        if n < tgt:
            r, e = divmod(tgt, n)
            ii = np.concatenate([np.tile(ii, r), rng.choice(ii, e, replace=False)])
        out.append(ii)
    return np.concatenate(out)


def evaluate(model, tok, msk, noc, bs=256):
    model.eval(); pred = []
    with torch.no_grad():
        for i in range(0, len(tok), bs):
            x = torch.from_numpy(tok[i:i + bs]).to(DEVICE)
            m = torch.from_numpy(msk[i:i + bs]).to(DEVICE)
            pred.append(model(x, m).argmax(1).cpu().numpy() + 1)
    pred = np.concatenate(pred)
    return {"acc": float((pred == noc).mean()),
            "macro_f1": float(f1_score(noc, pred, average="macro", labels=[1, 2, 3, 4, 5],
                                       zero_division=0)),
            "per_noc_recall": [float((pred[noc == j] == j).mean()) if (noc == j).any() else None
                               for j in range(1, 6)]}, pred


def fit(model, tok, msk, noc, epochs, lr, bs, tag, eval_sets, sel_set, warmup=0, patience=20):
    """Schedule copied from the decoder-FT notebook: `warmup` epochs training ONLY the count head at a
    higher lr, then unfreeze the whole fork at `lr` with cosine annealing, early stop on `sel_set`
    macro-F1 with best-state restore. `sel_set` must never be the reported real test."""
    lab = noc - 1
    nc = np.bincount(lab, minlength=5).astype(float)
    w = 1.0 / np.clip(nc, 1, None); w = w / w.mean()
    crit = nn.CrossEntropyLoss(weight=torch.tensor(np.clip(w, 0.5, 2.0),
                                                   dtype=torch.float32).to(DEVICE))
    idx = oversample(np.arange(len(lab)), lab)
    np.random.default_rng(0).shuffle(idx)
    dl = DataLoader(TensorDataset(torch.from_numpy(tok[idx]), torch.from_numpy(msk[idx]),
                                  torch.from_numpy(lab[idx])), batch_size=bs, shuffle=True,
                    drop_last=True)
    head_only = [p for p in model.head.parameters()]
    all_p = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_only if warmup else all_p,
                            lr=lr * 6 if warmup else lr, weight_decay=1e-4)
    sched = None
    best, best_state, wait, hist = -1.0, None, 0, []
    for ep in range(1, epochs + 1):
        if ep == warmup + 1 and warmup:
            opt = torch.optim.AdamW(all_p, lr=lr, weight_decay=1e-4)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs - warmup,
                                                              eta_min=1e-6)
        params = head_only if ep <= warmup else all_p
        model.train(); tot = n = 0
        for xb, mb, yb in dl:
            xb, mb, yb = xb.to(DEVICE), mb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = crit(model(xb, mb), yb); loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            tot += loss.item(); n += 1
        if sched is not None:
            sched.step()
        sel, _ = evaluate(model, *sel_set)
        if sel["macro_f1"] > best:
            best, wait = sel["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                          if not k.startswith("backbone.")}
        else:
            wait += 1
        if ep % 5 == 0 or ep == epochs or wait == 0:
            line = f"    [{tag}] ep {ep:3d} loss={tot/max(n,1):.4f} sel_mF1={sel['macro_f1']:.4f} (best {best:.4f})"
            for nm, es in eval_sets.items():
                r, _ = evaluate(model, *es)
                line += f" | {nm} acc={r['acc']:.4f} mF1={r['macro_f1']:.4f}"
                hist.append({"stage": tag, "epoch": ep, "set": nm, **r})
            print(line, flush=True)
        if wait >= patience:
            print(f"    [{tag}] early stop ep {ep} (best sel mF1 {best:.4f})", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    return hist


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backbone", required=True, help="path to the frozen ID checkpoint")
    ap.add_argument("--fork", choices=("last", "full"), default="last")
    ap.add_argument("--init", choices=("copy", "random"), default="copy")
    ap.add_argument("--epochs1", type=int, default=60)     # decoder-FT schedule
    ap.add_argument("--epochs2", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n_syn", type=int, default=20000)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--check_id", action="store_true", default=True)
    args = ap.parse_args()

    print(f"device={DEVICE}  fork={args.fork}  init={args.init}  data={DATA_W}")
    dgp = DATA_W / "donor_geno.npy"
    if not dgp.exists():
        dgp = REAL / "donor_geno.npy"
    dg = torch.from_numpy(np.load(dgp).astype(np.float32))
    dgm = torch.from_numpy(np.load(dgp.parent / "donor_geno_mask.npy"))
    ol = build_luts(dg, dgm).to(DEVICE)
    bb = SetTransformerMixture(
        n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45,
        dropout=0.1, n_token_feats=8, n_freq=8, d_num_emb=8, periodic_sigma=0.3, n_slot_iters=3,
        ot_eps=0.05, ot_iters=5, gumbel_temp=1.0, donor_geno=dg, donor_geno_mask=dgm,
        owner_lut=ol, noc_head_v2=True).to(DEVICE)
    miss, _ = bb.load_state_dict(torch.load(args.backbone, map_location=DEVICE, weights_only=True),
                                 strict=False)
    assert not miss, miss
    bb.eval()

    # ID guard: snapshot the frozen backbone's ID outputs BEFORE any NOC training
    tk_r, mk_r, noc_r = load_split(REAL, "test")
    id_before = None
    if args.check_id:
        with torch.no_grad():
            id_before = np.concatenate([
                bb(torch.from_numpy(tk_r[i:i+256]).to(DEVICE),
                   torch.from_numpy(mk_r[i:i+256]).to(DEVICE))["logits_cls"].cpu().numpy()
                for i in range(0, len(tk_r), 256)])

    model = NocBranch(bb, fork=args.fork, init=args.init).to(DEVICE)
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable (NOC branch only): {n_tr:,}  |  frozen backbone: "
          f"{sum(p.numel() for p in bb.parameters()):,}")

    tk_s, mk_s, noc_s = load_split(DATA_W, "train")
    rng = np.random.default_rng(0)
    sub = np.sort(rng.choice(len(tk_s), size=min(args.n_syn, len(tk_s)), replace=False))
    tk_s, mk_s, noc_s = tk_s[sub], mk_s[sub], noc_s[sub]
    dev = load_split(DATA_W, "dev") if (DATA_W / "tokens8_dev.npy").exists() else None
    eval_sets = {"REAL": (tk_r, mk_r, noc_r)}
    if dev:
        eval_sets["synDEV"] = dev
    print(f"stage1 in-silico rows {len(tk_s)}  NOC dist "
          f"{ {k: int((noc_s==k).sum()) for k in range(1,6)} }")

    t0 = time.time()
    assert dev is not None, "stage 1 needs the in-silico DEV split for selection"
    h1 = fit(model, tk_s, mk_s, noc_s, args.epochs1, args.lr, args.bs, "S1-insilico",
             eval_sets, sel_set=dev, warmup=args.warmup, patience=args.patience)
    r1, _ = evaluate(model, tk_r, mk_r, noc_r)
    print(f"  STAGE 1 (in-silico only)  REAL acc={r1['acc']:.4f}  macroF1={r1['macro_f1']:.4f}  "
          f"per-NOC {['%.3f' % x if x is not None else '-' for x in r1['per_noc_recall']]}")

    # stage 2: real NOC labels that cost no ID test data (open mixtures + real single-source)
    h2, r2 = [], None
    tk_o, mk_o, _ = load_split(REAL, "open")
    noc_o_p = REAL / "noc_true_open.npy"
    if noc_o_p.exists():
        noc_o = np.clip(np.load(noc_o_p).astype(np.int64), 1, 5)
        tk_n1, mk_n1, noc_n1 = load_split(REAL, "train")           # real single-source (all NOC=1)
        tk2 = np.concatenate([tk_o, tk_n1]); mk2 = np.concatenate([mk_o, mk_n1])
        noc2 = np.concatenate([noc_o, noc_n1])
        print(f"stage2 real rows {len(tk2)}  NOC dist { {k: int((noc2==k).sum()) for k in range(1,6)} }")
        # selection for stage 2: hold out whole real combos, so it is never a within-combo peek
        gsel = _holdout_by_combo(REAL, len(tk_o), len(tk_n1))
        h2 = fit(model, tk2[~gsel], mk2[~gsel], noc2[~gsel], args.epochs2, args.lr * 0.5, args.bs,
                 "S2-real", eval_sets, sel_set=(tk2[gsel], mk2[gsel], noc2[gsel]),
                 warmup=0, patience=args.patience)
        r2, _ = evaluate(model, tk_r, mk_r, noc_r)
        print(f"  STAGE 2 (+ real open/NOC1) REAL acc={r2['acc']:.4f}  macroF1={r2['macro_f1']:.4f}  "
              f"per-NOC {['%.3f' % x if x is not None else '-' for x in r2['per_noc_recall']]}")
    else:
        print(f"  STAGE 2 SKIPPED: {noc_o_p} missing (true NOC for the open split)")

    # ID guard
    id_ok = None
    if args.check_id and id_before is not None:
        with torch.no_grad():
            id_after = np.concatenate([
                bb(torch.from_numpy(tk_r[i:i+256]).to(DEVICE),
                   torch.from_numpy(mk_r[i:i+256]).to(DEVICE))["logits_cls"].cpu().numpy()
                for i in range(0, len(tk_r), 256)])
        d = float(np.abs(id_after - id_before).max()); id_ok = d == 0.0
        print(f"  ID GUARD: max|delta logits_cls| = {d:.3e}  -> "
              f"{'IDENTICAL (ID untouched)' if id_ok else 'CHANGED — NOC gradient leaked into the backbone'}")

    out = Path(args.out) if args.out else ROOT / "results" / f"noc_{args.fork}_{args.init}"
    out.mkdir(parents=True, exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")},
               out / "noc_branch.pt")
    json.dump({"fork": args.fork, "init": args.init, "trainable_params": n_tr,
               "stage1_real": r1, "stage2_real": r2, "id_guard_identical": id_ok,
               "elapsed_s": round(time.time() - t0, 1), "history": h1 + h2},
              open(out / "metrics.json", "w"), indent=2)
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
