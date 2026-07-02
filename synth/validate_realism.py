"""
synth/validate_realism.py  —  Task 2.4

So sanh phan bo synthetic vs 20 hon hop THAT (multi-person PROVEDIt).
Bat buoc vi identification can fidelity allele-level, khong chi dem NOC.

Metrics so sanh:
  1. Tong peak height / profile (sum log1p height)
  2. So allele / profile
  3. Heterozygote balance (min/max allele height ratio) trung binh
  4. Mx uoc luong (ratio donor lon nhat / tong height)
  5. #peaks / locus trung binh

Ket qua:
  - KS-test (p-value) moi metric
  - Histogram chong PNG
  - results/set_transformer/realism_validation.json + .png

Usage:
  python synth/validate_realism.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT  = Path(__file__).resolve().parents[1]
DATA  = ROOT / "data"
SYNTH = DATA / "synth"
OUT   = ROOT / "results" / "set_transformer"

# ── Load helpers ──────────────────────────────────────────────────────────────

def load_meta():
    with open(DATA / "meta_set.json") as f:
        return json.load(f)

def load_names(split):
    with open(DATA / f"meta_sample_names_{split}.json") as f:
        return json.load(f)

# ── Feature extraction per sample ─────────────────────────────────────────────

def extract_features(tokens, mask, noc_arr, min_h_log=np.log1p(50)):
    """
    tokens: (N, 160, 3) — locus_idx, allele, log1p_height
    mask:   (N, 160)    — True = valid token
    Returns df with per-sample metrics.
    """
    N = len(tokens)
    records = []
    for i in range(N):
        valid  = mask[i]
        toks   = tokens[i]
        n_valid = valid.sum()
        if n_valid == 0:
            continue

        heights = toks[:n_valid, 2]          # log1p heights
        h_real  = heights[heights >= min_h_log]
        if len(h_real) == 0:
            continue

        total_h   = float(h_real.sum())
        n_alleles = int(len(h_real))

        # Heterozygote balance: per-locus, take min/(min+max) for loci with 2 peaks
        locus_heights: dict[int, list] = {}
        for j in range(n_valid):
            if toks[j, 2] < min_h_log:
                continue
            li = int(round(toks[j, 0]))
            locus_heights.setdefault(li, []).append(float(toks[j, 2]))

        balances = []
        for li, hs in locus_heights.items():
            if len(hs) >= 2:
                s = sorted(hs)
                balances.append(s[0] / s[-1])

        het_balance = float(np.mean(balances)) if balances else 1.0
        peaks_per_locus = n_valid / max(len(locus_heights), 1)

        records.append({
            "total_log_height": total_h,
            "n_alleles": n_alleles,
            "het_balance": het_balance,
            "peaks_per_locus": peaks_per_locus,
            "noc": int(noc_arr[i]),
        })

    return pd.DataFrame(records)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Load real multi-person samples ────────────────────────────────────────
    print("Loading real multi-person samples ...")
    real_rows = []
    for split in ("train", "val", "test"):
        tok  = np.load(DATA / f"tokens_{split}.npy")
        msk  = np.load(DATA / f"mask_{split}.npy")
        nocs = np.load(DATA / f"noc_{split}.npy")
        mul  = nocs >= 2
        if mul.sum() == 0:
            continue
        df = extract_features(tok[mul], msk[mul], nocs[mul])
        real_rows.append(df)

    df_real = pd.concat(real_rows, ignore_index=True)
    print(f"  Real multi-person samples: {len(df_real)}")

    # ── Load synthetic samples ─────────────────────────────────────────────────
    synth_tok  = DATA / "tokens_synth_train.npy"
    synth_msk  = DATA / "mask_synth_train.npy"
    synth_noc  = DATA / "noc_synth_train.npy"

    if not synth_tok.exists():
        print("ERROR: synthetic train arrays not found. Run generate_dataset.py first.")
        return

    print("Loading synthetic samples ...")
    tok  = np.load(synth_tok)
    msk  = np.load(synth_msk)
    nocs = np.load(synth_noc)
    df_synth = extract_features(tok, msk, nocs)
    print(f"  Synthetic samples: {len(df_synth)}")

    # ── KS tests ──────────────────────────────────────────────────────────────
    metrics = ["total_log_height", "n_alleles", "het_balance", "peaks_per_locus"]
    ks_results = {}
    print("\n=== KS tests (synthetic vs real) ===")
    print(f"{'Metric':<20} {'KS stat':>8} {'p-value':>10} {'result':>8}")
    print("-" * 50)
    for m in metrics:
        real_vals  = df_real[m].dropna().values
        synth_vals = df_synth[m].dropna().values
        if len(real_vals) == 0 or len(synth_vals) == 0:
            continue
        ks, p = stats.ks_2samp(real_vals, synth_vals)
        ok = "OK" if p > 0.05 else "WARN"
        print(f"  {m:<20} {ks:>8.4f} {p:>10.4f}  [{ok}]")
        ks_results[m] = {"ks_stat": float(ks), "p_value": float(p)}

    # ── Plots ─────────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        metric_labels = {
            "total_log_height":  "Total log1p(height) / profile",
            "n_alleles":         "Number of alleles / profile",
            "het_balance":       "Heterozygote balance (mean)",
            "peaks_per_locus":   "Peaks per locus",
        }

        for ax, m in zip(axes, metrics):
            rv = df_real[m].dropna().values
            sv = df_synth[m].dropna().values
            ax.hist(rv, bins=30, alpha=0.5, color="#4C72B0",
                    density=True, label=f"Real (n={len(rv)})")
            ax.hist(sv, bins=30, alpha=0.5, color="#DD8452",
                    density=True, label=f"Synthetic (n={len(sv)})")
            ks = ks_results.get(m, {}).get("ks_stat", float("nan"))
            p  = ks_results.get(m, {}).get("p_value", float("nan"))
            ax.set_title(f"{metric_labels[m]}\nKS={ks:.3f}  p={p:.3f}",
                         fontsize=10)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.suptitle("Realism Validation: Synthetic vs Real mixtures (GF29cycles)",
                     fontsize=12, y=1.01)
        plt.tight_layout()
        OUT.mkdir(parents=True, exist_ok=True)
        plot_path = OUT / "realism_validation.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved -> {plot_path}")
    except ImportError:
        print("\nmatplotlib not available — skipping plots")

    # ── Summary statistics ────────────────────────────────────────────────────
    print("\n=== Descriptive statistics ===")
    for m in metrics:
        rv = df_real[m].dropna().values
        sv = df_synth[m].dropna().values
        print(f"  {m}:")
        print(f"    Real:  mean={rv.mean():.3f} std={rv.std():.3f}"
              f" range=[{rv.min():.3f},{rv.max():.3f}]")
        print(f"    Synth: mean={sv.mean():.3f} std={sv.std():.3f}"
              f" range=[{sv.min():.3f},{sv.max():.3f}]")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    report = {
        "n_real":  len(df_real),
        "n_synth": len(df_synth),
        "ks_tests": ks_results,
        "real_stats": {
            m: {
                "mean": float(df_real[m].mean()),
                "std":  float(df_real[m].std()),
                "min":  float(df_real[m].min()),
                "max":  float(df_real[m].max()),
            } for m in metrics
        },
        "synth_stats": {
            m: {
                "mean": float(df_synth[m].mean()),
                "std":  float(df_synth[m].std()),
                "min":  float(df_synth[m].min()),
                "max":  float(df_synth[m].max()),
            } for m in metrics
        },
    }
    json_path = OUT / "realism_validation.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report -> {json_path}")

    # Pass/warn
    n_warn = sum(1 for r in ks_results.values() if r["p_value"] <= 0.05)
    if n_warn == 0:
        print("\nPASS: All KS tests p > 0.05 -- synthetic distribution matches real.")
    else:
        print(f"\nWARN: {n_warn}/{len(metrics)} metrics show distribution mismatch (p<=0.05).")
        print("  Consider adjusting template/degradation range in simulate_mixtures.R")


if __name__ == "__main__":
    main()
