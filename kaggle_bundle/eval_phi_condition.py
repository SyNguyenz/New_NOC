"""
eval_phi_condition.py — Increment 2 evaluation grounded in design_increment2 §12.

Two evaluations on REAL test (which, per the PROVEDIt naming convention, DOES carry phi and
DNA condition — recovered by extract_phi_condition.py):

  A. CONDITION-STRATIFIED EM  (Alfonse 2018 PROVEDIt was built across 144 conditions for exactly
     this; Green & Mortera height-information ceiling + Tvedebrink/ISFG dropout-vs-template predict
     degraded/inhibited strata lose ID info). EM per (condition) and (condition x NOC), Wilson CIs.
     Uses results/<run>/y_test_pred.npy + y_test_true.npy (no model needed).

  B. phi (Mx) ESTIMATION  (Zhu 2026 feature iii; EuroForMix/STRmix validate estimated-vs-known Mx;
     HSU abundance RMSE). Needs results/<run>/phi_pred_test.npy (dumped by train_set_transformer.py
     when --aux_heads) + phi_test.npy (real NOMINAL ratio). Because nominal != realized proportion
     (degradation + per-locus amp efficiency), the PRIMARY metric is SCALE-ROBUST Spearman rank-corr
     + major/minor ordering; MAE is secondary.

  C. ERROR vs TEMPLATE/Q  (Tvedebrink/ISFG: dropout rises as template/peak-height falls). Mean
     template_ng & Q for EM-correct vs EM-wrong, if meta_template_/meta_qindex_ present.

Usage:
  python eval_phi_condition.py --results inc2_2b_privsup
  python eval_phi_condition.py --results inc2_2b_privsup --data data_insilico_w
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
COND_CATS = ["untreated", "dnase", "fragmentase", "sonication", "uv", "humic", "unknown"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% CI for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rho = Pearson on ranks (no scipy dependency)."""
    if len(a) < 2:
        return np.nan
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    return float((ra * rb).sum() / den) if den > 0 else np.nan


def load(p: Path):
    return np.load(p) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="results subdir (e.g. inc2_2b_privsup)")
    ap.add_argument("--data", default=None, help="data dir (default: data_insilico_w if present else data)")
    args = ap.parse_args()

    rdir = ROOT / "results" / args.results
    ddir = Path(args.data) if args.data else (
        ROOT / "data_insilico_w" if (ROOT / "data_insilico_w" / "noc_test.npy").exists()
        else ROOT / "data")
    print(f"results: {rdir}\ndata:    {ddir}\n")

    y_pred = load(rdir / "y_test_pred.npy")
    y_true = load(rdir / "y_test_true.npy")
    noc    = load(ddir / "noc_test.npy")
    cond   = load(ddir / "condition_test.npy")
    if y_pred is None or y_true is None:
        raise SystemExit("missing y_test_pred/y_test_true — run training first")
    noc = noc.astype(int) if noc is not None else y_true.sum(1).astype(int)
    em = (y_pred == y_true).all(1).astype(int)
    N = len(em)
    print(f"N test = {N} | overall EM = {em.mean():.3f}\n")

    # ── A. condition-stratified EM ────────────────────────────────────────────
    if cond is not None:
        print("== A. EM by DNA condition (Wilson 95% CI) " + "=" * 28)
        print(f"  {'condition':<12}{'n':>5}{'EM':>8}{'95% CI':>16}")
        for c in range(len(COND_CATS)):
            m = cond == c
            if not m.any():
                continue
            k, n = int(em[m].sum()), int(m.sum())
            lo, hi = wilson(k, n)
            print(f"  {COND_CATS[c]:<12}{n:>5}{em[m].mean():>8.3f}   [{lo:.3f}, {hi:.3f}]")
        print("\n== A2. EM by (condition x NOC) " + "=" * 39)
        hdr = "  {:<12}".format("condition") + "".join(f"NOC{k:>4}" for k in [1, 2, 3, 4, 5])
        print(hdr)
        for c in range(len(COND_CATS)):
            mc = cond == c
            if not mc.any():
                continue
            row = f"  {COND_CATS[c]:<12}"
            for k in [1, 2, 3, 4, 5]:
                m = mc & (noc == k)
                row += f"{(em[m].mean() if m.any() else np.nan):>7.2f}" if m.any() else f"{'-':>7}"
            print(row)
        print()
    else:
        print("(A skipped: condition_test.npy not found — run extract_phi_condition.py)\n")

    # ── B. phi (Mx) estimation ────────────────────────────────────────────────
    phi_pred = load(rdir / "phi_pred_test.npy")
    phi_true = load(ddir / "phi_test.npy")
    if phi_pred is not None and phi_true is not None:
        print("== B. phi (Mx) estimation vs nominal ratio " + "=" * 27)
        print("  (primary = Spearman + major ID; MAE secondary; nominal!=realized, see §12)")
        rows = {k: {"sp": [], "maj": [], "mae": []} for k in [2, 3, 4, 5]}
        for i in range(N):
            present = np.where(phi_true[i] > 0)[0]
            k = len(present)
            if k < 2:
                continue
            pt = phi_true[i, present]
            pp = phi_pred[i, present]
            pp_n = pp / pp.sum() if pp.sum() > 0 else pp
            rows[k]["sp"].append(spearman(pp, pt))
            rows[k]["maj"].append(int(present[pp.argmax()] == present[pt.argmax()]))
            rows[k]["mae"].append(float(np.abs(pp_n - pt).mean()))
        print(f"  {'NOC':<5}{'n':>5}{'Spearman':>10}{'major-ID':>10}{'MAE':>8}")
        allsp, allmaj, allmae, tot = [], [], [], 0
        for k in [2, 3, 4, 5]:
            sp = rows[k]["sp"]
            if not sp:
                continue
            n = len(sp); tot += n
            allsp += sp; allmaj += rows[k]["maj"]; allmae += rows[k]["mae"]
            print(f"  {k:<5}{n:>5}{np.nanmean(sp):>10.3f}{np.mean(rows[k]['maj']):>10.3f}"
                  f"{np.mean(rows[k]['mae']):>8.3f}")
        if tot:
            print(f"  {'all':<5}{tot:>5}{np.nanmean(allsp):>10.3f}{np.mean(allmaj):>10.3f}"
                  f"{np.mean(allmae):>8.3f}")
        print()
    else:
        miss = "phi_pred_test.npy" if phi_pred is None else "phi_test.npy"
        print(f"(B skipped: {miss} not found — train with --aux_heads / run extract_phi_condition.py)\n")

    # ── D. per-peak allele->donor attribution on real (does the grouping transfer?) ──
    attr_pred = load(rdir / "attr_pred_test.npy")
    attr_true = load(ddir / "attr_test.npy")
    if attr_pred is not None and attr_true is not None and attr_pred.shape == attr_true.shape:
        print("== D. per-peak allele->donor attribution accuracy (real) " + "=" * 13)
        print("  (over labelled peaks attr>=0, from reference genotypes; build_real_attr.py)")
        lab = attr_true >= 0
        correct = (attr_pred == attr_true) & lab
        nocrow = np.repeat(noc[:, None], attr_true.shape[1], axis=1)
        print(f"  {'NOC':<5}{'labelled':>10}{'acc':>8}")
        for k in [1, 2, 3, 4, 5]:
            m = lab & (nocrow == k)
            if m.any():
                print(f"  {k:<5}{int(m.sum()):>10}{correct[m].sum() / m.sum():>8.3f}")
        if lab.any():
            print(f"  {'all':<5}{int(lab.sum()):>10}{correct.sum() / lab.sum():>8.3f}")
        print()
    else:
        miss = "attr_pred_test.npy" if attr_pred is None else "attr_test.npy"
        print(f"(D skipped: {miss} missing/misaligned — --aux_heads + build_real_attr.py)\n")

    # ── C. error vs template / Q (stratified WITHIN NOC to remove the NOC confound) ──
    # NOTE: template_ng RISES with NOC (more contributors => more total template), and failures
    # concentrate at high NOC, so the pooled correct-vs-wrong comparison is CONFOUNDED by NOC and
    # can show the OPPOSITE sign of the dropout effect. The dropout hypothesis (Tvedebrink/ISFG:
    # dropout rises as template/peak-height falls) must be tested WITHIN each NOC stratum: there,
    # dropout-driven failure => wrong < correct. The direction is reported empirically, not asserted.
    tpl = load(ddir / "meta_template_test.npy")
    q   = load(ddir / "meta_qindex_test.npy")
    if tpl is not None or q is not None:
        print("== C. template_ng / Q : EM-correct vs EM-wrong (within-NOC; 'all' row is NOC-confounded) "
              + "=" * 3)
        print("  dropout-driven failure (Tvedebrink/ISFG) => 'wrong<corr' WITHIN a NOC stratum")
        ok = em == 1
        for nm, arr in [("template_ng", tpl), ("Q", q)]:
            if arr is None:
                continue
            if len(arr) != N:                                 # misaligned (e.g. old extract_metadata) -> skip
                print(f"  {nm:<12} SKIP: length {len(arr)} != N {N} (re-run extract_phi_condition.py)")
                continue
            a = arr.astype(float)
            fin = np.isfinite(a)
            print(f"  {nm}")
            print(f"    {'NOC':<5}{'n_ok':>6}{'n_wrong':>8}{'med_correct':>13}{'med_wrong':>11}{'dir':>12}")
            strata = [("all", fin)] + [(str(k), fin & (noc == k)) for k in [1, 2, 3, 4, 5]]
            for label, sub in strata:
                mok, mw = sub & ok, sub & ~ok
                if not (mok.any() and mw.any()):              # need both groups to compare
                    continue
                mc, mwd = float(np.median(a[mok])), float(np.median(a[mw]))
                d = "wrong<corr" if mwd < mc else ("wrong>corr" if mwd > mc else "~equal")
                tag = "  (confounded)" if label == "all" else ""
                print(f"    {label:<5}{int(mok.sum()):>6}{int(mw.sum()):>8}"
                      f"{mc:>13.3f}{mwd:>11.3f}{d:>12}{tag}")
        print()
    else:
        print("(C skipped: meta_template_/meta_qindex_ not found)\n")


if __name__ == "__main__":
    main()
