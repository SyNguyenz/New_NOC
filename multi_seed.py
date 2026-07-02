"""
Multi-seed training for confidence intervals.

Trains the Set Transformer (full configuration) with 5 different random seeds,
keeping the data split fixed (seed=42 in prepare_data_set.py).
Reports mean ± std for all key metrics.

Seeds vary PyTorch model initialization and training batch shuffling.
Data split is fixed to ensure fair comparison.

Usage:
  python multi_seed.py               # seeds 0,1,2,3,4  (5 runs)
  python multi_seed.py --seeds 0 1 2 # custom seeds
  python multi_seed.py --epochs 60   # epoch override
  python multi_seed.py --quick       # 3 seeds, 40 epochs for smoke test

Output: results/multi_seed/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from ablation import train_variant

OUT_DIR = ROOT / "results" / "multi_seed"


def set_seeds(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",   nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--quick",   action="store_true",
                        help="3 seeds, 40 epochs (smoke test)")
    args = parser.parse_args()

    if args.quick:
        args.seeds = [0, 1, 2]
        if args.epochs is None:
            args.epochs = 40

    cfg_path = ROOT / "configs" / "set_transformer.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    for seed in args.seeds:
        print(f"\n{'='*60}")
        print(f"  SEED {seed}  ({args.seeds.index(seed)+1}/{len(args.seeds)})")
        print("="*60)
        set_seeds(seed)
        r = train_variant("full", cfg.copy(), out_name=f"seed_{seed}")
        r["seed"] = seed
        all_results.append(r)

    # Aggregate statistics
    metric_keys = ["macro_f1", "micro_f1", "exact_match", "hamming",
                   "precision", "recall"]
    stats = {}
    for k in metric_keys:
        vals = [r["test"].get(k) for r in all_results if r["test"].get(k) is not None]
        if vals:
            stats[k] = {
                "mean":   float(np.mean(vals)),
                "std":    float(np.std(vals, ddof=1)),
                "min":    float(np.min(vals)),
                "max":    float(np.max(vals)),
                "values": [float(v) for v in vals],
            }

    auroc_vals = [r.get("reject_auroc") for r in all_results
                  if r.get("reject_auroc") is not None]
    if auroc_vals:
        stats["reject_auroc"] = {
            "mean": float(np.mean(auroc_vals)),
            "std":  float(np.std(auroc_vals, ddof=1)),
            "values": [float(v) for v in auroc_vals],
        }

    print(f"\n{'='*70}")
    print(f"  MULTI-SEED SUMMARY  ({len(args.seeds)} seeds: {args.seeds})")
    print("="*70)
    for k, v in stats.items():
        print(f"  {k:<15}  {v['mean']:.4f} ± {v['std']:.4f}"
              f"  [{v.get('min', v['values'][0]):.4f}, {v.get('max', v['values'][-1]):.4f}]")
    print("="*70)

    summary = {
        "seeds": args.seeds,
        "n_seeds": len(args.seeds),
        "epochs_per_run": cfg["epochs"],
        "stats": stats,
        "all_runs": all_results,
    }
    out_path = OUT_DIR / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
