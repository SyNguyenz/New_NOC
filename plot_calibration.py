"""
Plot calibration reliability diagram from analyze_st.py output.
Reads results/set_transformer/analysis.json (no GPU needed).

Outputs:
  results/set_transformer/calibration_reliability.png
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IN   = ROOT / "results" / "set_transformer" / "analysis.json"
OUT  = ROOT / "results" / "set_transformer" / "calibration_reliability.png"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    with open(IN) as f:
        data = json.load(f)

    cal = data["calibration"]
    n_bins = cal["n_bins"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    sections = [
        ("reject_head",    "Reject head", axes[0]),
        ("donor_head_top1", "Donor head (top-1 confidence)", axes[1]),
    ]

    for key, title, ax in sections:
        bins = cal[key]["bins"]
        ece  = cal[key]["ece"]

        # Filter out empty bins
        confs = [b["conf"] for b in bins if b["conf"] is not None]
        accs  = [b["acc"]  for b in bins if b["conf"] is not None]
        ns    = [b["n"]    for b in bins if b["conf"] is not None]

        # Perfect calibration line
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect calibration")

        # Scatter with point size proportional to bin population
        max_n = max(ns) if ns else 1
        sizes = [max(30, 200 * n / max_n) for n in ns]
        sc = ax.scatter(confs, accs, s=sizes, color="#4C72B0", alpha=0.85,
                        zorder=5, label="Bin accuracy")

        # Bar showing gap from perfect
        bar_w = 1.0 / n_bins * 0.8
        for conf, acc in zip(confs, accs):
            ax.bar(conf, acc, width=bar_w, alpha=0.25, color="#4C72B0",
                   align="center", bottom=0)
            # Gap from perfect
            gap_color = "#C44E52" if acc < conf else "#55A868"
            ax.bar(conf, conf - acc, width=bar_w * 0.5, alpha=0.5,
                   color=gap_color, align="center", bottom=min(acc, conf))

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Confidence (predicted probability)", fontsize=11)
        ax.set_ylabel("Accuracy (fraction correct)", fontsize=11)
        ax.set_title(f"{title}\nECE = {ece:.4f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    plt.suptitle("Calibration reliability diagrams — Set Transformer",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
