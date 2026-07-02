"""
probe_noc_structure.py — manipulation check for Rank-N-Contrast (F19).

A null downstream result (card_noc_acc unchanged) is only interpretable if the contrast actually
shaped the projection. This probes z_noc_proj_test.npy (dumped at test) for ORDINAL structure,
independent of the NOC head:
  1. Spearman(pairwise feature distance, |NOCi - NOCj|) — RNC success => strongly POSITIVE
     (features ordered along the 1..5 axis: far in NOC => far in feature space).
  2. Linear-probe NOC from z_noc_proj (logistic regression, 5-fold) — accuracy vs the NOC prior.
If both are near chance, the contrast did NOT work (manipulation failed) => the downstream null is
uninterpretable; fix tau / weight before concluding anything about RNC.

Usage:
  python probe_noc_structure.py --results inc2_2c_pe3_shared_seed42 --data data_w
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results subdir (must contain z_noc_proj_test.npy)")
    ap.add_argument("--data", default="data_w", help="data dir with noc_test.npy")
    ap.add_argument("--results_root", default="results")
    args = ap.parse_args()
    rdir = Path(args.results_root) / args.results
    zp = rdir / "z_noc_proj_test.npy"
    if not zp.exists():
        print(f"NO z_noc_proj_test.npy in {rdir} — arm has no RNC, or older run. Nothing to probe.")
        return
    Z = np.load(zp).astype(np.float64)                       # (N, d_proj)
    noc = np.load(Path(args.data) / "noc_test.npy").astype(int)[: len(Z)]
    print(f"probe {args.results}: Z={Z.shape}  NOC dist={dict(zip(*np.unique(noc, return_counts=True)))}")

    # L2-normalize (matches the contrast space)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    # 1. Spearman(pairwise feature distance, label distance) on a random pair sample
    rng = np.random.default_rng(0)
    N = len(Zn); m = min(20000, N * (N - 1) // 2)
    i = rng.integers(0, N, m); j = rng.integers(0, N, m)
    ok = i != j; i, j = i[ok], j[ok]
    fd = np.linalg.norm(Zn[i] - Zn[j], axis=1)               # feature distance
    ld = np.abs(noc[i] - noc[j]).astype(float)               # label distance
    try:
        from scipy.stats import spearmanr
        rho, p = spearmanr(fd, ld)
    except Exception:
        rho = np.corrcoef(np.argsort(np.argsort(fd)), np.argsort(np.argsort(ld)))[0, 1]; p = float("nan")
    print(f"  [1] Spearman(feature-dist, |dNOC|) = {rho:+.3f} (p={p:.1e})  "
          f"-> {'ORDINAL structure present' if rho > 0.15 else 'NEAR ZERO => contrast did not shape projection'}")

    # 2. Linear probe: logistic regression NOC from z_noc_proj, 5-fold accuracy vs majority prior
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        acc = cross_val_score(LogisticRegression(max_iter=1000),
                              Z, noc, cv=5, scoring="accuracy").mean()
        prior = np.bincount(noc).max() / len(noc)
        print(f"  [2] linear-probe NOC acc = {acc:.3f}  (majority prior {prior:.3f})  "
              f"-> {'projection encodes NOC' if acc > prior + 0.05 else 'no better than prior'}")
    except Exception as e:
        print(f"  [2] linear probe skipped: {e}")

    print("  VERDICT: manipulation " +
          ("SUCCEEDED — downstream null is interpretable." if rho > 0.15
           else "FAILED — contrast inert; downstream null is NOT evidence against RNC (fix tau/weight)."))


if __name__ == "__main__":
    main()
