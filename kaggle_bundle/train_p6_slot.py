"""
train_p6_slot.py — P6 (SEPARATE CODE; does not modify existing files).

Per-CONTRIBUTOR SLOT set-prediction unifying ID + NOC (DETR Carion 2020 / Deep Set Prediction Networks
Zhang 2019 / DeepSetNet Rezatofighi 2017 = project C8). K learnable slot queries cross-attend the encoder
H; each slot predicts ONE donor (0..44) OR ∅ (=45). ID = donors of non-∅ slots; NOC = #non-∅ slots
(intrinsic, combo-invariant "count by binding contributors one-by-one"). Trained with a permutation-
invariant SET LOSS via Hungarian matching (scipy linear_sum_assignment).

Reuses (imports, no edits): harness from train_set_transformer.py; encoder/aux backbone + MAB from
models.set_transformer. RISK (honest, C8): set-prediction on a CLOSED 45-set is harder/less stable than the
45-query multi-label baseline — this is the research arm; judge ID set-EM + NOC acc per NOC on dev vs train.

Usage: python train_p6_slot.py [--seed 42] [--out_subdir inc4_p6_slot] [--epochs N]
Smoke:  P6_SMOKE=1 python train_p6_slot.py
"""
import os, sys, json, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from train_set_transformer import ClosedSetDataset, set_seed, per_noc_em, DEVICE, DATA_DIR
from models.set_transformer import SetTransformerMixture, MAB
from torch.utils.data import DataLoader

N_DONOR = 45
NULL = 45          # ∅ class index
K_SLOTS = 8        # > max NOC (5), DETR-style slack


def base_kwargs(d_model=128, n_heads=4, n_isab=2, m_ind=32, dropout=0.1):
    return dict(n_loci=24, d_locus=16, d_model=d_model, n_heads=n_heads, n_isab=n_isab,
                m_inducing=m_ind, n_classes=45, n_noc=6, dropout=dropout, cls_decoder="per_donor",
                decoder_source="encoded", n_token_feats=8, encoder="isab++", dec_layers=2,
                num_embed="periodic", n_freq=8, d_num_emb=8, periodic_sigma=0.3, aux_heads=True)


class P6Model(nn.Module):
    def __init__(self, d_model=128, n_heads=4, dropout=0.1, k_slots=K_SLOTS, **bk):
        super().__init__()
        self.base = SetTransformerMixture(**base_kwargs(d_model=d_model, n_heads=n_heads, dropout=dropout, **bk))
        self.slots = nn.Parameter(torch.empty(1, k_slots, d_model)); nn.init.xavier_uniform_(self.slots)
        self.slot_mab = MAB(d_model, n_heads, dropout)
        self.slot_cls = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                                      nn.Linear(d_model, N_DONOR + 1))   # 45 donors + ∅
        self.k = k_slots

    def forward(self, tokens, mask):
        x0, H, pad = self.base._encode_set(tokens, mask)
        z = self.base.pma(H, pad_mask=pad).squeeze(1)
        S = self.slots.expand(H.size(0), -1, -1)
        Sd = self.slot_mab(S, H, key_padding_mask=pad)           # (B,K,d)
        out = {"logits_slot": self.slot_cls(Sd)}                 # (B,K,46)
        if getattr(self.base, "aux_heads", False):
            out["logits_attr"] = self.base.attr_head(H); out["phi"] = F.softplus(self.base.phi_head(z))
        return out


def set_loss(logits_slot, y, null_coef: float = 1.0, diversity: float = 0.0):
    """logits_slot (B,K,46), y (B,45) multi-hot. Hungarian-matched set CE. ∅-padding to K targets.

    Inc6 C slot-fix (null_coef<1, diversity>0):
      • null_coef = DETR's eos_coef (Carion et al. 2020, §A.3): down-weight the ∅ ("no-object")
        target in the CE. K=8 slots but ~85% of samples are NOC1 → 7/8 targets are ∅; unweighted,
        the ∅ class dominates and every slot collapses to ∅ (under-counting = the C8 collapse).
      • diversity = anti-duplicate term: penalise pairwise agreement of slots' donor distributions
        so distinct contributors bind to distinct slots (the other half of the binding problem).
    Defaults (1.0, 0.0) reproduce the original inc4_p6_slot exactly."""
    B, K, C = logits_slot.shape
    tot = 0.0
    logp = F.log_softmax(logits_slot, dim=-1)
    w = torch.ones(C, device=logits_slot.device)
    w[NULL] = null_coef
    for b in range(B):
        donors = torch.where(y[b] > 0.5)[0].tolist()
        tgt = donors + [NULL] * (K - len(donors))               # pad to K with ∅
        tgt = torch.tensor(tgt[:K], device=logits_slot.device)
        cost = -logp[b][:, tgt]                                  # (K,K) = -logprob(slot i -> target j)
        r, c = linear_sum_assignment(cost.detach().cpu().numpy())
        assigned = torch.empty(K, dtype=torch.long, device=logits_slot.device)
        assigned[torch.as_tensor(r, device=logits_slot.device)] = tgt[torch.as_tensor(c, device=logits_slot.device)]
        tot = tot + F.nll_loss(logp[b], assigned, weight=w)
    loss = tot / B
    if diversity > 0.0 and K > 1:
        p = logp.exp()[:, :, :N_DONOR]                           # (B,K,45) donor mass per slot
        G = torch.bmm(p, p.transpose(1, 2))                     # (B,K,K) slot-slot agreement
        off = G.sum((1, 2)) - torch.diagonal(G, dim1=1, dim2=2).sum(1)   # off-diagonal (duplicates)
        loss = loss + diversity * (off / (K * (K - 1))).mean()
    return loss


@torch.no_grad()
def predict_sets(model, tok, msk):
    """-> y_pred (N,45) multi-hot, noc_pred (N,) = #distinct non-∅ donors."""
    yp = np.zeros((len(tok), N_DONOR), int); kp = np.zeros(len(tok), int)
    for i in range(0, len(tok), 256):
        o = model(torch.from_numpy(tok[i:i+256]).to(DEVICE), torch.from_numpy(msk[i:i+256]).to(DEVICE))
        cls = o["logits_slot"].argmax(-1).cpu().numpy()         # (b,K)
        for j in range(cls.shape[0]):
            donors = set(int(c) for c in cls[j] if c < N_DONOR)
            for d in donors:
                yp[i + j, d] = 1
            kp[i + j] = max(1, len(donors))
    return yp, kp


def _smoke():
    torch.manual_seed(0)
    m = P6Model(d_model=32, n_heads=2, n_isab=1, m_ind=8).to(DEVICE)
    tok = torch.rand(6, 30, 8, device=DEVICE); msk = torch.ones(6, 30, dtype=torch.bool, device=DEVICE)
    out = m(tok, msk)
    y = torch.zeros(6, 45);  # give 1..5 donors each
    for b in range(6):
        y[b, torch.randperm(45)[:b % 5 + 1]] = 1
    loss = set_loss(out["logits_slot"], y.to(DEVICE)); loss.backward()
    yp, kp = predict_sets(m, tok.cpu().numpy(), msk.cpu().numpy())
    assert out["logits_slot"].shape == (6, K_SLOTS, 46) and yp.shape == (6, 45)
    print(f"P6 SMOKE OK  loss={loss.item():.3f}  slot{tuple(out['logits_slot'].shape)}  noc_pred={kp.tolist()}")


def setem_per_noc(yp, y, noc):
    nocc = np.clip(noc, 1, 5)
    return {int(k): float((yp[nocc == k] == y[nocc == k]).all(1).mean()) for k in range(1, 6) if (nocc == k).any()}

def noc_acc_per_noc(kp, noc):
    nocc = np.clip(noc, 1, 5)
    return {int(k): float((np.clip(kp[nocc == k], 1, 5) == k).mean()) for k in range(1, 6) if (nocc == k).any()}


def train_p6(seed, out_subdir, epochs, null_coef=1.0, diversity=0.0):
    set_seed(seed); tp = "tokens8"
    if null_coef != 1.0 or diversity != 0.0:
        print(f"Inc6 C slot-fix ON: DETR eos_coef(null_coef)={null_coef}  slot_diversity={diversity}")
    tr = ClosedSetDataset("train", tp); te = ClosedSetDataset("test", tp)
    dev = ClosedSetDataset("dev", tp) if (DATA_DIR / f"{tp}_dev.npy").exists() else te
    trL = DataLoader(tr, batch_size=256, shuffle=True, num_workers=0)
    model = P6Model().to(DEVICE)
    _tk = tr.tokens.numpy(); _mk = tr.mask.numpy().astype(bool); _num = _tk[:, :, 1:8][_mk]
    model.base.feat_mean.copy_(torch.tensor(_num.mean(0), dtype=torch.float32, device=DEVICE))
    model.base.feat_std.copy_(torch.tensor(_num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=6e-4, weight_decay=1e-4)
    dev_arr = (dev.tokens.numpy(), dev.mask.numpy(), dev.noc.numpy(), dev.y.numpy())
    best, best_state = -1, None
    EP = epochs or 150
    for ep in range(1, EP + 1):
        model.train()
        for tok, msk, y, noc, attr, phi in trL:
            tok, msk, y = tok.to(DEVICE), msk.to(DEVICE), y.float().to(DEVICE)
            o = model(tok, msk)
            loss = set_loss(o["logits_slot"], y, null_coef=null_coef, diversity=diversity)
            if "logits_attr" in o:
                loss = loss + 0.3 * F.cross_entropy(o["logits_attr"].reshape(-1, o["logits_attr"].size(-1)),
                                                    attr.to(DEVICE).reshape(-1), ignore_index=-1)
                loss = loss + 0.1 * F.mse_loss(o["phi"], phi.to(DEVICE))
            opt.zero_grad(); loss.backward(); opt.step()
        yp, kp = predict_sets(model, dev_arr[0], dev_arr[1])
        dev_em = float((yp == dev_arr[3]).all(1).mean())
        if dev_em > best:
            best, best_state = dev_em, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 20 == 0:
            print(f"  ep{ep} dev_setEM={dev_em:.4f} best={best:.4f}")
    model.load_state_dict(best_state)

    out_dir = Path("results") / out_subdir; out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "best_model.pt")
    splits = {"dev": dev_arr,
              "train": (tr.tokens.numpy(), tr.mask.numpy(), tr.noc.numpy(), tr.y.numpy()),
              "test": (te.tokens.numpy(), te.mask.numpy(), te.noc.numpy(), te.y.numpy())}
    gen = {}
    for nm, (tk, mk, nc, yy) in splits.items():
        if nm == "train":
            idx = np.random.default_rng(1).choice(len(tk), size=min(4000, len(tk)), replace=False)
            tk, mk, nc, yy = tk[idx], mk[idx], nc[idx], yy[idx]
        yp, kp = predict_sets(model, tk, mk)
        gen[nm] = {"id_setEM": setem_per_noc(yp, yy, nc), "noc_acc": noc_acc_per_noc(kp, nc)}

    metrics = {"model": "p6_slot", "config": {**base_kwargs(), "seed": seed, "out_subdir": out_subdir,
               "k_slots": K_SLOTS}, "test_setEM_overall": gen["test"]["id_setEM"],
               "generalization": gen}
    json.dump(metrics, open(out_dir / "metrics.json", "w"), indent=2)
    print("\n== P6 (slot set-prediction): ID set-EM & NOC acc per NOC ==")
    for sp in ("train", "dev", "test"):
        print(f"  {sp:>5}: setEM N5={gen[sp]['id_setEM'].get(5)}  nocAcc N5={gen[sp]['noc_acc'].get(5)}")
    print(f"wrote {out_dir/'metrics.json'}")


if __name__ == "__main__":
    if os.environ.get("P6_SMOKE"):
        _smoke(); sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default="inc4_p6_slot")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--slot_fix", action="store_true",
                    help="Inc6 C: DETR eos_coef + slot-diversity to break the ∅-collapse (sets "
                         "null_coef=0.1, slot_diversity=0.05 unless overridden)")
    ap.add_argument("--null_coef", type=float, default=None, help="DETR eos_coef for the ∅ class")
    ap.add_argument("--slot_diversity", type=float, default=None, help="anti-duplicate slot penalty")
    a = ap.parse_args()
    nc = a.null_coef if a.null_coef is not None else (0.1 if a.slot_fix else 1.0)
    dv = a.slot_diversity if a.slot_diversity is not None else (0.05 if a.slot_fix else 0.0)
    train_p6(a.seed, a.out_subdir, a.epochs, null_coef=nc, diversity=dv)
