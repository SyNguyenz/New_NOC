"""
optimize_pgnoc.py — Tune pgNOC knobs on a held-out IN-SILICO dev (the correct dev:
large, hard, labeled; real val is saturated). Sweeps the gamma variance exponent alpha
(w=1/hrel^alpha; higher = more minor-peak emphasis = better high-NOC) and candidate
pool size. Penalty for k* is tuned on the FIT split; per-NOC count accuracy reported on
the DEV split. No real test touched.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import pgnoc

D = ROOT / "data_insilico_w"; DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    t0 = time.time()
    m = pgnoc.SetTransformerMixture(n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2,
        m_inducing=32, n_classes=45, n_noc=6, dropout=0.1, cls_decoder="hybrid",
        n_flat=590, decouple_reject=True).to(DEV)
    m.load_state_dict(torch.load(ROOT / "results/hybrid_50k_weight/best_model.pt", weights_only=True)); m.eval()
    G = pgnoc.build_refs()

    noc = np.clip(np.load(D / "noc_train.npy").astype(int), 1, 5)
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(np.where(noc == k)[0], size=min(1200, (noc == k).sum()), replace=False)
                          for k in range(1, 6)])
    rng.shuffle(idx)
    t = np.load(D / "tokens_train.npy")[idx]; mk = np.load(D / "mask_train.npy")[idx]
    xf = np.load(D / "Xflat_train.npy")[idx].astype(np.float32)
    H = np.expm1(np.load(D / "Xflat_train.npy")[idx].astype(np.float64)); nsub = noc[idx]
    with torch.no_grad():
        P = np.concatenate([torch.sigmoid(m(torch.from_numpy(t[i:i+512]).to(DEV),
            torch.from_numpy(mk[i:i+512]).to(DEV), torch.from_numpy(xf[i:i+512]).to(DEV))["logits_cls"]).cpu().numpy()
            for i in range(0, len(t), 512)])
    nfit = len(idx) * 3 // 5
    fit, dev = np.arange(nfit), np.arange(nfit, len(idx))
    print(f"setup {time.time()-t0:.0f}s; subset {len(idx)} (fit {len(fit)}, dev {len(dev)})")

    def cost_all(alpha, pool):
        C = np.zeros((len(idx), pgnoc.KMAX))
        for i in range(len(idx)):
            _, C[i] = pgnoc.estimate_noc(H[i], P[i], G, pool=pool, alpha=alpha)
        return C

    def acc_dev(C):
        pen, _ = pgnoc.tune_penalty(C[fit], nsub[fit])
        ar = np.arange(1, pgnoc.KMAX + 1); k = (C[dev] + pen * ar).argmin(1) + 1
        td = nsub[dev]
        return (k == td).mean(), [(k[td == j] == j).mean() for j in range(1, 6)], pen

    print(f"\n  {'alpha':>5}{'pool':>5}  {'devAcc':>7}{'N1':>6}{'N2':>6}{'N3':>6}{'N4':>6}{'N5':>6}{'pen':>7}")
    best = None
    for alpha in [0.25, 0.5, 0.75, 1.0]:
        for pool in [8, 12]:
            t1 = time.time(); C = cost_all(alpha, pool); a, pn, pen = acc_dev(C)
            print(f"  {alpha:>5}{pool:>5}  {a:>7.4f}" + "".join(f"{x:>6.2f}" for x in pn) +
                  f"{pen:>7.3f}  ({time.time()-t1:.0f}s)")
            if best is None or a > best[0]: best = (a, alpha, pool)
    print(f"\nBEST on in-silico dev: alpha={best[1]} pool={best[2]} (devAcc={best[0]:.4f})")
    print("current default = alpha 0.5 / pool 8")


if __name__ == "__main__":
    main()
