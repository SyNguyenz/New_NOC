"""
Week 1 baseline — XGBoost multi-label classifier (one-vs-rest, 45 classes).

Usage:
    python models/baseline_xgb.py [--config configs/xgb_baseline.json]

Outputs predictions to results/baseline_xgb/ and prints metrics table.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results" / "baseline_xgb"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": 5,   # classes are sparse; boost positive class
    "tree_method": "hist",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}


def load_data():
    X_train = np.load(DATA_DIR / "X_train.npy")
    X_val   = np.load(DATA_DIR / "X_val.npy")
    X_test  = np.load(DATA_DIR / "X_test.npy")
    y_train = np.load(DATA_DIR / "y_train.npy")
    y_val   = np.load(DATA_DIR / "y_val.npy")
    y_test  = np.load(DATA_DIR / "y_test.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import (
        f1_score, hamming_loss, precision_score, recall_score
    )
    results = {
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        "micro_f1":    float(f1_score(y_true, y_pred, average="micro",  zero_division=0)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }

    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    results["per_class_f1"] = per_class_f1.tolist()

    return results


def print_metrics(split: str, m: dict):
    print(f"\n-- {split.upper()} " + "-" * 30)
    print(f"  Macro F1        : {m['macro_f1']:.4f}")
    print(f"  Micro F1        : {m['micro_f1']:.4f}")
    print(f"  Hamming Loss    : {m['hamming_loss']:.4f}")
    print(f"  Exact Match     : {m['exact_match']:.4f}")
    print(f"  Macro Precision : {m['macro_precision']:.4f}")
    print(f"  Macro Recall    : {m['macro_recall']:.4f}")

    pcf = np.array(m["per_class_f1"])
    print(f"  Per-class F1    : mean={pcf.mean():.3f}  "
          f"min={pcf.min():.3f}  max={pcf.max():.3f}  "
          f"zero={int((pcf == 0).sum())}/45 classes")


def main(params: dict):
    print("Loading data …")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_val  : {X_val.shape}    y_val  : {y_val.shape}")
    print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

    # Combine train+val for final model fitting after hyperparam selection
    X_tv = np.concatenate([X_train, X_val])
    y_tv = np.concatenate([y_train, y_val])

    base = XGBClassifier(**params)
    model = MultiOutputClassifier(base, n_jobs=-1)

    print(f"\nTraining on train split ({len(X_train)} samples) …")
    t0 = time.time()
    model.fit(X_train, y_train)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Evaluate on val
    y_val_pred = model.predict(X_val)
    val_metrics = compute_metrics(y_val, y_val_pred)
    print_metrics("val", val_metrics)

    # Retrain on train+val, evaluate on test
    print(f"\nRetraining on train+val ({len(X_tv)} samples) for final test eval …")
    t0 = time.time()
    model.fit(X_tv, y_tv)
    print(f"  Done in {time.time()-t0:.1f}s")

    y_test_pred = model.predict(X_test)
    test_metrics = compute_metrics(y_test, y_test_pred)
    print_metrics("test", test_metrics)

    # ── Save ───────────────────────────────────────────────────────────────
    import joblib
    np.save(RESULTS_DIR / "y_test_pred.npy", y_test_pred)
    np.save(RESULTS_DIR / "y_test_true.npy", y_test)
    joblib.dump(model, RESULTS_DIR / "model.pkl")

    results = {
        "params": params,
        "val":  val_metrics,
        "test": test_metrics,
    }
    with open(RESULTS_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults + model saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config overriding default XGB params")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    if args.config:
        with open(args.config) as f:
            params.update(json.load(f))

    main(params)
