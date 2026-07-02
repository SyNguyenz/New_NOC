"""
Stratified evaluation — break Exact Match and Macro F1 down by forensic
conditions encoded in PROVEDIt filenames (template DNA, Q-index, injection time).

Reads predictions saved by training scripts plus metadata from extract_metadata.py.
Reports stratified tables for the Set Transformer (default), or any results dir
via --results_dir.

Outputs: prints tables; saves results/{model}/stratified.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


# ── Binning ───────────────────────────────────────────────────────────────

TEMPLATE_BINS = [(0.0, 0.05),  (0.05, 0.15),  (0.15, 0.5),  (0.5, np.inf)]
TEMPLATE_LABELS = ["<=0.05 ng (very low)", "0.05-0.15 ng (low)",
                   "0.15-0.50 ng (med)",  ">0.50 ng (high)"]

QINDEX_BINS = [(0.0, 1.0), (1.0, 10.0), (10.0, 100.0), (100.0, np.inf)]
QINDEX_LABELS = ["Q<1 (severe)", "Q 1-10 (degraded)",
                 "Q 10-100 (moderate)", "Q>100 (clean)"]


def bin_index(values: np.ndarray, bins: list[tuple[float, float]]) -> np.ndarray:
    """Return bin index for each value; -1 if NaN or out of range."""
    out = np.full(len(values), -1, dtype=np.int32)
    for i, (lo, hi) in enumerate(bins):
        mask = (values >= lo) & (values < hi) & np.isfinite(values)
        out[mask] = i
    return out


# ── Reporting ─────────────────────────────────────────────────────────────

def report_strat(y_true: np.ndarray, y_pred: np.ndarray, idx: np.ndarray,
                 labels: list[str], title: str) -> list[dict]:
    print(f"\n-- {title} " + "-" * (60 - len(title)))
    print(f"  {'Stratum':<28} {'n':>6} {'ExactMatch':>11} {'MacroF1':>9}")
    print("  " + "-" * 56)
    rows = []
    for k, name in enumerate(labels):
        mask = idx == k
        n = int(mask.sum())
        if n == 0:
            rows.append({"stratum": name, "n": 0,
                         "exact_match": None, "macro_f1": None})
            continue
        em = float(np.all(y_true[mask] == y_pred[mask], axis=1).mean())
        mf = float(f1_score(y_true[mask], y_pred[mask],
                            average="macro", zero_division=0))
        rows.append({"stratum": name, "n": n, "exact_match": em, "macro_f1": mf})
        print(f"  {name:<28} {n:>6} {em:>11.4f} {mf:>9.4f}")
    # Out-of-range / NaN bin
    unbinned = (idx == -1).sum()
    if unbinned > 0:
        print(f"  (unbinned/NaN: {unbinned})")
    return rows


def report_strat_noc(y_true: np.ndarray, y_pred: np.ndarray,
                     strat_idx: np.ndarray, noc: np.ndarray,
                     strat_labels: list[str], title: str) -> list[dict]:
    """2D stratification: strat × NOC, report ExactMatch."""
    print(f"\n-- {title} (Exact Match) " + "-" * (60 - len(title)))
    nocs = sorted(np.unique(noc[noc > 0]).tolist())
    hdr  = "  " + f"{'Stratum':<28}" + "".join(f"{f'NOC={n}':>8}" for n in nocs) + f"{'Total':>8}"
    print(hdr); print("  " + "-" * (28 + 8 * (len(nocs) + 1)))
    rows = []
    for k, name in enumerate(strat_labels):
        m_s = strat_idx == k
        if m_s.sum() == 0: continue
        per_noc = {}
        cells = []
        for n in nocs:
            m = m_s & (noc == n)
            if m.sum() == 0:
                cells.append(f"{'-':>8}"); per_noc[int(n)] = None
            else:
                em = float(np.all(y_true[m] == y_pred[m], axis=1).mean())
                cells.append(f"{em:>8.3f}"); per_noc[int(n)] = em
        total = m_s.sum()
        em_all = float(np.all(y_true[m_s] == y_pred[m_s], axis=1).mean())
        print(f"  {name:<28}" + "".join(cells) + f"{em_all:>8.3f} (n={total})")
        rows.append({"stratum": name, "n": int(total),
                     "per_noc": per_noc, "exact_match_all": em_all})
    return rows


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir",
                        default="results/set_transformer",
                        help="Folder containing y_test_true.npy + y_test_pred.npy")
    parser.add_argument("--split", default="test",
                        help="Data split (test / val / train)")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    y_true = np.load(rd / "y_test_true.npy")
    y_pred = np.load(rd / "y_test_pred.npy")
    print(f"Loaded {y_true.shape[0]} samples from {rd}")
    assert y_true.shape == y_pred.shape, "shape mismatch"

    tpl = np.load(DATA_DIR / f"meta_template_{args.split}.npy")
    qix = np.load(DATA_DIR / f"meta_qindex_{args.split}.npy")
    inj = np.load(DATA_DIR / f"meta_injection_{args.split}.npy")
    noc = np.load(DATA_DIR / f"noc_{args.split}.npy")

    assert len(tpl) == len(y_true), \
        f"metadata length {len(tpl)} != predictions {len(y_true)}"

    # 1D stratification
    tpl_idx = bin_index(tpl, TEMPLATE_BINS)
    qix_idx = bin_index(qix, QINDEX_BINS)
    inj_unique = sorted(set(int(v) for v in inj.tolist() if v > 0))
    inj_idx = np.full(len(inj), -1, dtype=np.int32)
    inj_labels = []
    for k, v in enumerate(inj_unique):
        inj_idx[inj == v] = k
        inj_labels.append(f"{v} sec")

    print("\n" + "=" * 64)
    print("  STRATIFIED EVALUATION")
    print("=" * 64)

    out = {}
    out["by_template"] = report_strat(y_true, y_pred, tpl_idx,
                                      TEMPLATE_LABELS, "By template DNA (ng)")
    out["by_qindex"]   = report_strat(y_true, y_pred, qix_idx,
                                      QINDEX_LABELS,   "By PROVEDIt Q-index")
    out["by_injection"] = report_strat(y_true, y_pred, inj_idx,
                                       inj_labels,     "By injection time")

    # 2D: template × NOC
    out["template_x_noc"] = report_strat_noc(
        y_true, y_pred, tpl_idx, noc, TEMPLATE_LABELS,
        "By template × NOC",
    )

    # Save
    out_path = rd / "stratified.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
