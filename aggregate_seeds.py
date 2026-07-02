"""
aggregate_seeds.py — collapse per-seed runs of each arm into mean ± 95% CI.

Why: small NOC strata (n=44-48) swing ~±10pp run-to-run from training stochasticity alone
(weight init + shuffle + dropout). The SAME inc2_2b_privsup config gave NOC2 oracle 0.864 vs
0.977 on two runs. A single run therefore cannot separate a real per-arm effect from noise.
This reads results/<arm>_s<seed>/metrics.json across seeds and reports, per arm:
  - overall test EM and oracle EM  (mean ± 95% CI, min..max)
  - per-NOC ORACLE (the ranking ceiling — the load-bearing metric, esp. NOC2)
  - per-NOC joint-card EM
CI = t-based (small n); falls back to ±std if scipy missing or n<2. A per-arm delta is only
credible if the arms' CIs do not overlap (stated explicitly, no p-hacking).

Run dirs are named <arm>_seed<N> (suffix "_seed" — NOT "_s", which would clash with the sigma
dirs like inc2_2b_pe_s05). Aggregates over all seeds of a given arm.

Usage:
  python aggregate_seeds.py --results results --arms inc2_2b_privsup,inc2_2b_pe_s3,...
  (omit --arms to auto-discover every base arm that has >=1 <arm>_seed<N> dir)
"""
from __future__ import annotations
import argparse, glob, json, re
from pathlib import Path
import numpy as np

NOCS = ["1", "2", "3", "4", "5"]


def _ci_halfwidth(vals: list[float]) -> float:
    """95% CI half-width about the mean. t-based for small n; ±std if n<2 or no scipy."""
    a = np.asarray([v for v in vals if v is not None], float)
    n = len(a)
    if n < 2:
        return float("nan")
    sd = a.std(ddof=1)
    se = sd / np.sqrt(n)
    try:
        from scipy.stats import t
        return float(t.ppf(0.975, n - 1) * se)
    except Exception:
        return float(1.96 * se)


def _fmt(vals: list[float]) -> str:
    a = [v for v in vals if v is not None]
    if not a:
        return "   n/a"
    m = float(np.mean(a))
    if len(a) < 2:
        return f"{m:.3f} (n=1)"
    hw = _ci_halfwidth(a)
    return f"{m:.3f}±{hw:.3f} [{min(a):.3f},{max(a):.3f}]"


def discover_arms(results: Path) -> list[str]:
    arms = set()
    for d in results.glob("*_seed*"):
        if d.is_dir() and (d / "metrics.json").exists():
            m = re.match(r"(.+)_seed\d+$", d.name)
            if m:
                arms.add(m.group(1))
    return sorted(arms)


def load_arm(results: Path, arm: str) -> list[dict]:
    out = []
    for mf in sorted(glob.glob(str(results / f"{arm}_seed*" / "metrics.json"))):
        try:
            out.append(json.load(open(mf, encoding="utf-8-sig")))  # tolerate BOM
        except Exception as e:
            print(f"  ! skip {mf}: {e}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--arms", default="", help="comma list of base arm names; empty = auto-discover")
    args = ap.parse_args()
    results = Path(args.results)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()] or discover_arms(results)
    if not arms:
        print(f"no *_s<seed> runs under {results}"); return

    print(f"\n=== SEED AGGREGATION (mean ± 95% CI [min,max]) — {results} ===")
    rows = {}
    for arm in arms:
        runs = load_arm(results, arm)
        if not runs:
            print(f"\n{arm}: no runs found"); continue
        seeds = [r.get("config", {}).get("seed") for r in runs]
        em = [r.get("test", {}).get("exact_match") for r in runs]
        orc = [r.get("oracle_em") for r in runs]
        pno = {k: [(r.get("per_noc_oracle") or {}).get(k) for r in runs] for k in NOCS}
        pnem = {k: [(r.get("per_noc_joint_card") or {}).get(k) for r in runs] for k in NOCS}
        rows[arm] = {"em": em, "orc": orc, "pno": pno}
        print(f"\n{arm}  (n_seeds={len(runs)}, seeds={seeds})")
        print(f"  test EM    : {_fmt(em)}")
        print(f"  oracle EM  : {_fmt(orc)}")
        print(f"  per-NOC ORACLE : " + " | ".join(f"N{k} {_fmt(pno[k])}" for k in NOCS))
        print(f"  per-NOC EM     : " + " | ".join(f"N{k} {_fmt(pnem[k])}" for k in NOCS))

    # Pairwise comparison vs the first arm. At small n the 95% t-CI is very wide and can call a
    # real effect "not separable", so we ALSO report rank-based COMPLETE DOMINANCE (min of one arm
    # > max of the other across seeds) — a nonparametric signal; 3v3 complete separation is the
    # strongest result possible at n=3 (exact one-sided p=1/C(6,3)=0.05).
    def _dominance(a: list[float], b: list[float]) -> str:
        a = [v for v in a if v is not None]; b = [v for v in b if v is not None]
        if len(a) < 2 or len(b) < 2:
            return "n<2"
        if min(a) > max(b):
            return f"DOMINATES (min {min(a):.3f} > base max {max(b):.3f})"
        if max(a) < min(b):
            return f"DOMINATED (max {max(a):.3f} < base min {min(b):.3f})"
        return "overlapping ranges -> not separable at this n"

    if len(rows) >= 2:
        base = arms[0]
        print(f"\n--- vs baseline '{base}' (Δmean + rank-based complete-dominance across seeds) ---")
        for metric in ("orc", "em"):
            label = "oracle EM" if metric == "orc" else "test EM"
            b = [v for v in rows[base][metric] if v is not None]
            bm = np.mean(b) if b else float("nan")
            print(f"  [{label}]")
            for arm in arms[1:]:
                if arm not in rows:
                    continue
                a = [v for v in rows[arm][metric] if v is not None]
                if not a:
                    continue
                print(f"    {arm:<28} Δ={np.mean(a) - bm:+.3f}  {_dominance(a, b)}")
    print()


if __name__ == "__main__":
    main()
