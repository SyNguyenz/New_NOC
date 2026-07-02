"""
Train and evaluate sklearn baselines for 45-class multi-label contributor ID.

Models (planv2 Tuần 2):
  lr  — Per-donor logistic regression (MultiOutputClassifier)
  knn — kNN multi-label (native multi-output)
  xgb — XGBoost per-donor (MultiOutputClassifier)

Data: data/Xflat_{split}.npy + data/y_{split}_set.npy (GF29cycles, 590-dim)

Usage:
  python train_baselines.py --model lr
  python train_baselines.py --model knn
  python train_baselines.py --model xgb
  python train_baselines.py --model all
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, hamming_loss, precision_score, recall_score,
)
import joblib

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

import os
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
RESULT_SUFFIX = os.environ.get("STR_RESULT_SUFFIX", "")


# ── Data ───────────────────────────────────────────────────────────────────

def load_data():
    X_tr = np.load(DATA_DIR / "Xflat_train.npy")
    X_va = np.load(DATA_DIR / "Xflat_val.npy")
    X_te = np.load(DATA_DIR / "Xflat_test.npy")
    y_tr = np.load(DATA_DIR / "y_train_set.npy")
    y_va = np.load(DATA_DIR / "y_val_set.npy")
    y_te = np.load(DATA_DIR / "y_test_set.npy")
    noc_tr = np.load(DATA_DIR / "noc_train.npy")
    noc_te = np.load(DATA_DIR / "noc_test.npy")

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_va = scaler.transform(X_va)
    X_te = scaler.transform(X_te)

    meta = json.load(open(DATA_DIR / "meta_set.json"))
    donor_labels = [f"D{d:02d}" for d in meta["known_donors"]]

    return X_tr, X_va, X_te, y_tr, y_va, y_te, noc_tr, noc_te, scaler, donor_labels


# ── Evaluation ─────────────────────────────────────────────────────────────

def full_report(y_true, y_pred, noc_true, donor_labels, title="TEST"):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"  Samples: {y_true.shape[0]}  Classes: {y_true.shape[1]}")
    print("="*60)

    mf1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    mif1 = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    hl   = hamming_loss(y_true, y_pred)
    em   = np.all(y_true == y_pred, axis=1).mean()
    mp   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mr   = recall_score(y_true, y_pred, average="macro",    zero_division=0)

    jac  = (
        (y_true & y_pred.astype(bool)).sum(1) /
        (y_true | y_pred.astype(bool)).sum(1).clip(min=1)
    ).mean()

    print(f"  Macro F1     : {mf1:.4f}")
    print(f"  Micro F1     : {mif1:.4f}")
    print(f"  Hamming Loss : {hl:.4f}")
    print(f"  Exact Match  : {em:.4f}")
    print(f"  Macro Pre    : {mp:.4f}")
    print(f"  Macro Rec    : {mr:.4f}")
    print(f"  Jaccard (avg): {jac:.4f}")

    print("\n  -- Exact match by NOC " + "-"*36)
    exact = np.all(y_true == y_pred, axis=1)
    per_noc = {}
    for noc in sorted(np.unique(noc_true)):
        mask = noc_true == noc
        em_n = float(exact[mask].mean()) if mask.sum() else float("nan")
        f1_n = float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0)) if mask.sum() else float("nan")
        per_noc[int(noc)] = {"em": round(em_n, 4), "n": int(mask.sum())}
        print(f"    NOC={noc}: EM={em_n:.3f}  MacroF1={f1_n:.3f}  (n={mask.sum()})")

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    print(f"\n  Zero-F1 classes: {int((per_f1 == 0).sum())} / {len(per_f1)}")
    print("="*60)

    return {
        "macro_f1":    float(mf1),
        "micro_f1":    float(mif1),
        "hamming":     float(hl),
        "exact_match": float(em),
        "precision":   float(mp),
        "recall":      float(mr),
        "jaccard":     float(jac),
        "per_noc":     per_noc,
        "zero_f1_classes": int((per_f1 == 0).sum()),
    }


# ── Models ─────────────────────────────────────────────────────────────────

def build_lr():
    return MultiOutputClassifier(
        LogisticRegression(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs",
        ),
        n_jobs=-1,
    )


def build_knn():
    # KNeighborsClassifier natively supports multi-output
    return KNeighborsClassifier(
        n_neighbors=11,
        metric="cosine",
        algorithm="brute",
        n_jobs=-1,
    )


def build_xgb():
    if not HAS_XGB:
        raise ImportError("xgboost not installed")
    return MultiOutputClassifier(
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=5,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False,
            n_jobs=2,
        ),
        n_jobs=-1,
    )


BUILDERS = {"lr": build_lr, "knn": build_knn, "xgb": build_xgb}


# ── Main ───────────────────────────────────────────────────────────────────

def run(model_name: str):
    print(f"\n{'#'*60}")
    print(f"  Baseline: {model_name.upper()}")
    print(f"{'#'*60}")

    results_dir = ROOT / "results" / f"baseline_{model_name}{RESULT_SUFFIX}"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_tr, X_va, X_te, y_tr, y_va, y_te, noc_tr, noc_te, scaler, donor_labels = load_data()
    print(f"Train: {X_tr.shape}  Val: {X_va.shape}  Test: {X_te.shape}")

    model = BUILDERS[model_name]()

    # Fit on train only; val used only for final metric comparison
    print(f"Fitting {model_name} ...")
    t0 = time.time()
    model.fit(X_tr, y_tr.astype(int))
    print(f"  Done in {time.time()-t0:.1f}s")

    # Val
    y_va_pred = model.predict(X_va).astype(np.float32)
    va_metrics = full_report(
        y_va.astype(int), y_va_pred.astype(int),
        np.load(DATA_DIR / "noc_val.npy"),
        donor_labels, title=f"{model_name.upper()} — VAL"
    )

    # Test
    y_te_pred = model.predict(X_te).astype(np.float32)
    te_metrics = full_report(
        y_te.astype(int), y_te_pred.astype(int),
        noc_te, donor_labels, title=f"{model_name.upper()} — TEST"
    )

    # Save predictions + probabilities (probs needed for top-k decoding compare)
    np.save(results_dir / "y_test_pred.npy", y_te_pred)
    np.save(results_dir / "y_test_true.npy", y_te)
    try:
        proba = model.predict_proba(X_te)  # list of (N,2) per donor
        probs = np.stack([p[:, 1] for p in proba], axis=1).astype(np.float32)
        np.save(results_dir / "probs_test.npy", probs)
    except Exception as e:
        print(f"  (predict_proba unavailable: {e})")
    joblib.dump(model,  results_dir / "model.pkl")
    joblib.dump(scaler, results_dir / "scaler.pkl")

    out = {
        "model":    model_name,
        "val":      {k: v for k, v in va_metrics.items() if k != "per_noc"},
        "per_noc":  te_metrics.get("per_noc", {}),
        "test":     {k: v for k, v in te_metrics.items() if k != "per_noc"},
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved -> {results_dir}")
    return te_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", required=True,
        choices=["lr", "knn", "xgb", "all"],
    )
    args = parser.parse_args()

    targets = list(BUILDERS.keys()) if args.model == "all" else [args.model]
    summary = {}
    for m in targets:
        summary[m] = run(m)

    if len(targets) > 1:
        print("\n" + "="*60)
        print("  SUMMARY — Test Macro F1")
        print("="*60)
        for m, r in summary.items():
            print(f"  {m.upper():<6}: {r['macro_f1']:.4f}")
