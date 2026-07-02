"""
Evaluation script — load any y_true/y_pred .npy pair and print full metrics.

Usage:
    python evaluate.py results/baseline_xgb/
    python evaluate.py --true data/y_test.npy --pred results/baseline_xgb/y_test_pred.npy
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    classification_report,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


def load_pair(results_dir: Path = None, true_path: Path = None, pred_path: Path = None):
    if results_dir is not None:
        true_path = results_dir / "y_test_true.npy"
        pred_path = results_dir / "y_test_pred.npy"
    y_true = np.load(true_path)
    y_pred = np.load(pred_path)
    return y_true, y_pred


def full_report(y_true: np.ndarray, y_pred: np.ndarray, meta_path: Path = None):
    # Load donor labels if meta available
    donor_labels = [str(i) for i in range(45)]
    if meta_path and meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        known = meta.get("known_donors", [])
        donor_labels = [f"D{d:02d}" for d in known]

    print("=" * 60)
    print("MULTI-LABEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Samples : {y_true.shape[0]}")
    print(f"  Classes : {y_true.shape[1]}")
    print()

    # Aggregate metrics
    print("-- Aggregate metrics " + "-" * 39)
    print(f"  Macro F1        : {f1_score(y_true, y_pred, average='macro',  zero_division=0):.4f}")
    print(f"  Micro F1        : {f1_score(y_true, y_pred, average='micro',  zero_division=0):.4f}")
    print(f"  Weighted F1     : {f1_score(y_true, y_pred, average='weighted', zero_division=0):.4f}")
    print(f"  Hamming Loss    : {hamming_loss(y_true, y_pred):.4f}")
    print(f"  Exact Match Acc : {np.all(y_true == y_pred, axis=1).mean():.4f}")
    print(f"  Macro Precision : {precision_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Macro Recall    : {recall_score(y_true, y_pred, average='macro', zero_division=0):.4f}")

    # NOC-stratified exact match
    noc_true = y_true.sum(axis=1).astype(int)
    exact = np.all(y_true == y_pred, axis=1)
    print()
    print("-- Exact match by NOC " + "-" * 38)
    for noc in sorted(np.unique(noc_true)):
        mask = noc_true == noc
        em = exact[mask].mean()
        print(f"  NOC={noc}: {em:.3f}  ({mask.sum()} samples)")

    # Per-class report
    print()
    print("-- Per-class F1 (sorted ascending) " + "-" * 25)
    per_f1  = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_pre = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_rec = recall_score(y_true, y_pred, average=None, zero_division=0)
    support = y_true.sum(axis=0).astype(int)

    order = np.argsort(per_f1)
    print(f"  {'Donor':<8} {'F1':>6} {'Pre':>6} {'Rec':>6} {'Support':>8}")
    print(f"  {'-'*40}")
    for i in order:
        print(f"  {donor_labels[i]:<8} {per_f1[i]:>6.3f} {per_pre[i]:>6.3f} {per_rec[i]:>6.3f} {support[i]:>8}")

    print()
    print(f"  Zero-F1 classes: {int((per_f1 == 0).sum())} / {len(per_f1)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", nargs="?", type=Path, default=None)
    parser.add_argument("--true", type=Path, default=None)
    parser.add_argument("--pred", type=Path, default=None)
    args = parser.parse_args()

    if args.results_dir:
        y_true, y_pred = load_pair(results_dir=args.results_dir)
        meta_path = Path("data/meta.json")
    elif args.true and args.pred:
        y_true, y_pred = load_pair(true_path=args.true, pred_path=args.pred)
        meta_path = Path("data/meta.json")
    else:
        parser.error("Provide either a results_dir or both --true and --pred")

    full_report(y_true, y_pred, meta_path)


if __name__ == "__main__":
    main()
