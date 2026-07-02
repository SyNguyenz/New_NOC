"""
train_noc_marker.py — Replicate the "other code" approach on OUR no-leak split.

Marker-level (N, 24, 8) ALIGNED per-locus representation (NOT per-allele token bag) +
Set Transformer over the 24 loci, NOC-only loss. Tests the hypothesis: is the
per-allele token architecture what handicaps ST? If marker-ST reaches ~0.8 / generalizes
to our NOVEL combos, the representation was the culprit (not ST itself).

8 combo-invariant features per locus (NO allele identity -> can't memorize combos):
  [n_alleles_raw, n>50RFU, n>150RFU, n>300RFU, max_logh, mean_logh, sum_logh, std_logh]

Compares: marker-ST NOC vs XGB-tabular (0.918) vs embedding noc-head, + downstream EM.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_LOCI = 24


def build_markers(split: str) -> np.ndarray:
    """(N, 24, 8) combo-invariant per-locus features."""
    tok = np.load(DATA / f"tokens_{split}.npy")
    msk = np.load(DATA / f"mask_{split}.npy")
    h_rfu = np.expm1(tok[:, :, 2])
    N = len(tok)
    out = np.zeros((N, N_LOCI, 8), dtype=np.float32)
    for i in range(N):
        v = msk[i]
        loci = tok[i, :, 0][v].astype(int)
        rfu = h_rfu[i][v]
        logh = tok[i, :, 2][v]
        for L in range(N_LOCI):
            sel = loci == L
            if not sel.any():
                continue
            r = rfu[sel]; lh = logh[sel]
            out[i, L] = [sel.sum(), (r > 50).sum(), (r > 150).sum(), (r > 300).sum(),
                         lh.max(), lh.mean(), lh.sum(), lh.std()]
    return out


class MarkerNOCSetTransformer(nn.Module):
    """ISAB Set Transformer over 24 aligned loci -> NOC logits."""
    def __init__(self, d_in=8, d_model=64, n_heads=4, n_isab=2, m=16, n_noc=5, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.locus_emb = nn.Parameter(torch.zeros(1, N_LOCI, d_model))
        nn.init.normal_(self.locus_emb, std=0.02)
        from models.set_transformer import ISAB, PMA
        self.enc = nn.ModuleList([ISAB(d_model, n_heads, m, dropout) for _ in range(n_isab)])
        self.pma = PMA(d_model, n_heads, k_seeds=1, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_noc))

    def forward(self, x):
        h = self.proj(x) + self.locus_emb           # (B, 24, d)
        for isab in self.enc:
            h = isab(h, pad_mask=None)
        z = self.pma(h, pad_mask=None).squeeze(1)
        return self.head(z)


def per_noc(pred, true):
    out = [(pred == true).mean(), (np.abs(pred - true) <= 1).mean()]
    for k in range(1, 6):
        m = true == k
        out.append((pred[m] == k).mean() if m.sum() else float("nan"))
    return out


def downstream_em(k_arr):
    import sys; sys.path.insert(0, str(ROOT))
    from models.set_transformer import SetTransformerMixture
    m = SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
                              m_inducing=32, n_classes=45, n_noc=6, dropout=0.1,
                              cls_decoder="hybrid", n_flat=590, decouple_reject=True).to(DEV)
    m.load_state_dict(torch.load(ROOT/"results"/"set_transformer_hybrid_asl_decoup"/"best_model.pt", weights_only=True)); m.eval()
    t = torch.from_numpy(np.load(DATA/"tokens_test.npy")); msk = torch.from_numpy(np.load(DATA/"mask_test.npy"))
    xf = torch.from_numpy(np.load(DATA/"Xflat_test.npy").astype(np.float32)); P = []
    with torch.no_grad():
        for i in range(0, len(t), 256):
            P.append(torch.sigmoid(m(t[i:i+256].to(DEV), msk[i:i+256].to(DEV), xf[i:i+256].to(DEV))["logits_cls"]).cpu().numpy())
    P = np.concatenate(P); yt = np.load(DATA/"y_test_set.npy"); noc = np.load(DATA/"noc_test.npy")
    yp = np.zeros_like(P, dtype=int)
    for i in range(len(P)):
        k = int(max(1, min(5, round(k_arr[i])))); yp[i, np.argsort(P[i])[::-1][:k]] = 1
    em = (yt == yp).all(1)
    return [em.mean()] + [em[noc == j].mean() if (noc == j).sum() else float("nan") for j in range(1, 6)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=25)
    args = ap.parse_args()

    print("Building marker features..."); t0 = time.time()
    Xtr, Xva, Xte = build_markers("train"), build_markers("val"), build_markers("test")
    ntr = np.load(DATA/"noc_train.npy"); nva = np.load(DATA/"noc_val.npy"); nte = np.load(DATA/"noc_test.npy")
    print(f"  {Xtr.shape} in {time.time()-t0:.0f}s")
    # standardize per-feature
    mu, sd = Xtr.reshape(-1, 8).mean(0), Xtr.reshape(-1, 8).std(0) + 1e-6
    Xtr = (Xtr - mu)/sd; Xva = (Xva - mu)/sd; Xte = (Xte - mu)/sd

    def ld(X, n): return DataLoader(TensorDataset(torch.from_numpy(X), torch.from_numpy(n-1).long()),
                                    batch_size=128, shuffle=(n is ntr))
    tr_l, va_l = ld(Xtr, ntr), ld(Xva, nva)

    counts = np.bincount(ntr-1, minlength=5).astype(float)
    w = np.where(counts > 0, 1.0/counts, 0.0); w = w/w[counts > 0].mean()
    ce = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32).to(DEV))
    print(f"NOC class weights: {[f'{x:.2f}' for x in w]}")

    model = MarkerNOCSetTransformer().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.25, patience=8, min_lr=1e-5)
    best_f1, best_state, pc = -1, None, 0
    for ep in range(1, args.epochs+1):
        model.train()
        for x, y in tr_l:
            x, y = x.to(DEV), y.to(DEV)
            loss = ce(model(x), y); opt.zero_grad(); loss.backward(); opt.step()
        model.eval(); pr, tutu = [], []
        with torch.no_grad():
            for x, y in va_l:
                pr.append(model(x.to(DEV)).argmax(1).cpu().numpy()); tutu.append(y.numpy())
        vf1 = f1_score(np.concatenate(tutu), np.concatenate(pr), average="macro", zero_division=0)
        sch.step(vf1)
        if ep % 20 == 0 or ep == 1: print(f"  Ep {ep:3d} val_noc_f1={vf1:.4f}")
        if vf1 > best_f1: best_f1, best_state, pc = vf1, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= args.patience: print(f"  Early stop ep {ep} (best F1={best_f1:.4f})"); break
    model.load_state_dict({k: v.to(DEV) for k, v in best_state.items()})

    model.eval()
    with torch.no_grad():
        k_marker = model(torch.from_numpy(Xte).to(DEV)).argmax(1).cpu().numpy() + 1
    mf1 = f1_score(nte, k_marker, average="macro", zero_division=0)
    print("\n=== marker-ST NOC (test, no-leak novel combos) ===")
    print(f"  Macro F1: {mf1:.4f}")
    print(f"  {'':<14}{'acc':>6}{'w1':>6}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    print(f"  {'marker-ST':<14}" + "".join(f"{x:>7.3f}" if i > 1 else f"{x:>6.3f}" for i, x in enumerate(per_noc(k_marker, nte))))

    print("\n=== Downstream EM (marker-ST NOC + hybrid ranking) ===")
    print(f"  {'k-source':<16}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    print(f"  {'oracle':<16}" + "".join(f"{x:>7.3f}" for x in downstream_em(nte.astype(float))))
    print(f"  {'marker-ST NOC':<16}" + "".join(f"{x:>7.3f}" for x in downstream_em(k_marker.astype(float))))
    print("\n  (ref: XGB tabular NOC downstream 0.892, acc 0.918)")


if __name__ == "__main__":
    main()
