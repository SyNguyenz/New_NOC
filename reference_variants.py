"""
reference_variants.py — Test 4 ways to build the per-donor reference template G_d that
pgNOC deconvolves against (h ~ sum_d phi_d * G_d). Measured by standalone pgNOC per-NOC
count accuracy on the IN-SILICO dev (large, hard, labeled). Maps results back to theory.

  v0 global      : mean(rfu/rfu.sum() over all 590 bins)         [current; keeps noise+height, global norm]
  a  consensus   : consensus alleles only (>=50% of donor NOC1), DISCRETE indicator, global-norm [no height, cleaned]
  c  cons+height : consensus alleles only, KEEP mean relative height, global-norm  [hybrid: cleaned + height]
  b  per-locus   : consensus alleles, height normalized WITHIN each locus (removes locus-efficiency bias)
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pgnoc

D = ROOT / "data_insilico_w"; DEV = "cuda" if torch.cuda.is_available() else "cpu"
META = json.load(open(ROOT / "data/meta_set.json"))
LOCI = META["loci"]; L2I = {l: i for i, l in enumerate(LOCI)}
BIN_LOCUS = np.array([L2I[c.rsplit("_", 1)[0]] for c in META["flat_cols"]])
AT_REF = 50.0   # RFU: an allele is "present" in a NOC1 profile if rfu > AT_REF


def build_refs_variant(kind):
    X = np.expm1(np.load(D / "Xflat_train.npy").astype(np.float64))
    y = np.load(D / "y_train_set.npy"); n = np.load(D / "noc_train.npy").astype(int)
    ss = n == 1; G = np.zeros((45, X.shape[1]))
    for d in range(45):
        idx = np.where(ss & (y[:, d] == 1))[0]
        if not len(idx):
            continue
        R = X[idx]                                            # (n_d, 590) rfu
        if kind == "global":
            rel = R / (R.sum(1, keepdims=True) + 1e-12); g = rel.mean(0)
        else:
            cons = (R > AT_REF).mean(0) >= 0.5                # consensus allele mask
            if kind == "consensus":
                g = cons.astype(float)
            elif kind == "cons_height":
                g = R.mean(0) * cons                          # mean rfu, keep only consensus bins
            elif kind == "perlocus":
                mh = R.mean(0) * cons                         # consensus mean heights
                g = np.zeros_like(mh)
                for L in range(24):                           # normalize within each locus
                    m = BIN_LOCUS == L; s = mh[m].sum()
                    if s > 0: g[m] = mh[m] / s
            g = g / (g.sum() + 1e-12)                         # global-normalize to sum 1
        G[d] = g
    return G


def main():
    t0 = time.time()
    m = pgnoc.SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
        m_inducing=32, n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid",
        n_flat=590, decouple_reject=True).to(DEV)
    m.load_state_dict(torch.load(ROOT / "results/hybrid_50k_weight/best_model.pt", weights_only=True)); m.eval()

    noc = np.clip(np.load(D / "noc_train.npy").astype(int), 1, 5)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(noc == k)[0], size=min(1200, (noc == k).sum()), replace=False)
                          for k in range(1, 6)]); rng.shuffle(idx)
    t = np.load(D / "tokens_train.npy")[idx]; mk = np.load(D / "mask_train.npy")[idx]
    xf = np.load(D / "Xflat_train.npy")[idx].astype(np.float32)
    H = np.expm1(np.load(D / "Xflat_train.npy")[idx].astype(np.float64)); nsub = noc[idx]
    with torch.no_grad():
        P = np.concatenate([torch.sigmoid(m(torch.from_numpy(t[i:i+512]).to(DEV),
            torch.from_numpy(mk[i:i+512]).to(DEV), torch.from_numpy(xf[i:i+512]).to(DEV))["logits_cls"]).cpu().numpy()
            for i in range(0, len(t), 512)])
    nfit = len(idx) * 3 // 5; fit, dv = np.arange(nfit), np.arange(nfit, len(idx))
    print(f"setup {time.time()-t0:.0f}s; dev subset {len(idx)}")

    print(f"\n  {'reference':<14}{'devAcc':>8}{'N1':>6}{'N2':>6}{'N3':>6}{'N4':>6}{'N5':>6}")
    for kind in ["global", "consensus", "cons_height", "perlocus"]:
        G = build_refs_variant(kind)
        C = np.zeros((len(idx), pgnoc.KMAX))
        for i in range(len(idx)):
            _, C[i] = pgnoc.estimate_noc(H[i], P[i], G, alpha=0.5)
        pen, _ = pgnoc.tune_penalty(C[fit], nsub[fit])
        ar = np.arange(1, pgnoc.KMAX + 1); k = (C[dv] + pen * ar).argmin(1) + 1
        td = nsub[dv]; pn = [(k[td == j] == j).mean() for j in range(1, 6)]
        print(f"  {kind:<14}{(k==td).mean():>8.4f}" + "".join(f"{x:>6.2f}" for x in pn))


if __name__ == "__main__":
    main()
