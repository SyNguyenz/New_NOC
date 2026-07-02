"""
train_tabpfn.py — TabPFN v2 binary-relevance multi-label classifier.

Strategy: one TabPFNClassifier per donor (45 total, Binary Relevance).
Uses Xflat_{split}.npy (N, 590) features — same input as LR/XGB baselines.

Data note: TabPFN v2 supports up to ~10k samples / 500 features.
We have 6105 x 590 — marginally over the 500-feature soft limit.
ignore_pretraining_limits=True bypasses the check.

Usage:
  python train_tabpfn.py
  python train_tabpfn.py --no_noleak   # use leaky split in data_leaky/
  python train_tabpfn.py --n_est 16    # more estimators (slower, usually better)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Load TABPFN_TOKEN from .env if not already in environment
_env_file = Path(__file__).resolve().parent / ".env"
if _env_file.exists() and not os.environ.get("TABPFN_TOKEN"):
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line.startswith("TABPFN_TOKEN=") and not _line.startswith("#"):
            os.environ["TABPFN_TOKEN"] = _line.split("=", 1)[1].strip()
            break

import numpy as np
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parent


def load_split(data_dir: Path, split: str):
    X   = np.load(data_dir / f"Xflat_{split}.npy").astype(np.float32)
    y   = np.load(data_dir / f"y_{split}_set.npy").astype(np.float32)
    noc = np.load(data_dir / f"noc_{split}.npy").astype(np.int32)
    return X, y, noc


def full_report(y_true, y_pred, noc_true, title):
    mf1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    mif1 = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    hl   = hamming_loss(y_true, y_pred)
    em   = np.all(y_true == y_pred, axis=1).mean()
    mp   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mr   = recall_score(y_true, y_pred, average="macro",    zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)
    print(f"  Macro F1     : {mf1:.4f}")
    print(f"  Micro F1     : {mif1:.4f}")
    print(f"  Hamming Loss : {hl:.4f}")
    print(f"  Exact Match  : {em:.4f}")
    print(f"  Macro Pre    : {mp:.4f}")
    print(f"  Macro Rec    : {mr:.4f}")

    print("\n  -- Exact match by NOC " + "-"*36)
    exact = np.all(y_true == y_pred, axis=1)
    per_noc = {}
    for noc in sorted(np.unique(noc_true)):
        m = noc_true == noc
        em_n = float(exact[m].mean()) if m.sum() else float("nan")
        f1_n = float(f1_score(y_true[m], y_pred[m], average="macro", zero_division=0)) if m.sum() else float("nan")
        per_noc[int(noc)] = {"em": round(em_n, 4), "n": int(m.sum())}
        print(f"    NOC={noc}: EM={em_n:.3f}  MacroF1={f1_n:.3f}  (n={m.sum()})")
    print("="*60)

    return {
        "macro_f1":    round(float(mf1),  4),
        "micro_f1":    round(float(mif1), 4),
        "hamming":     round(float(hl),   6),
        "exact_match": round(float(em),   4),
        "precision":   round(float(mp),   4),
        "recall":      round(float(mr),   4),
        "per_noc":     per_noc,
    }


def train(args):
    data_dir = ROOT / ("data_leaky" if args.no_noleak else "data")
    out_dir  = ROOT / "results" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    X_tr, y_tr, noc_tr = load_split(data_dir, "train")
    X_va, y_va, noc_va = load_split(data_dir, "val")
    X_te, y_te, noc_te = load_split(data_dir, "test")

    # Combine train+val for final fit (same as XGB baseline approach)
    X_tv = np.concatenate([X_tr, X_va], axis=0)
    y_tv = np.concatenate([y_tr, y_va], axis=0)

    n_donors = y_tr.shape[1]
    print(f"TabPFN binary-relevance: {n_donors} donors")
    print(f"Train={len(X_tr)}  Val={len(X_va)}  Test={len(X_te)}  Features={X_tr.shape[1]}")
    print(f"n_estimators={args.n_est}  device=auto  ignore_limits=True")
    print(f"Data dir: {data_dir}")

    # ── Threshold search on val (fit on train only) ──────────────────────────
    print("\nFitting on train (for val threshold search)...")
    t0 = time.time()
    val_probs = np.zeros((len(X_va), n_donors), dtype=np.float32)
    clf_proto = dict(
        n_estimators=args.n_est,
        ignore_pretraining_limits=True,
        random_state=42,
        show_progress_bar=False,
    )
    for d in range(n_donors):
        labels = y_tr[:, d].astype(int)
        if labels.sum() == 0:
            val_probs[:, d] = 0.0
            continue
        clf = TabPFNClassifier(**clf_proto)
        clf.fit(X_tr, labels)
        p = clf.predict_proba(X_va)
        val_probs[:, d] = p[:, 1] if p.shape[1] > 1 else p[:, 0]
        if (d + 1) % 9 == 0:
            print(f"  donor {d+1}/{n_donors}  elapsed={time.time()-t0:.0f}s")

    # Threshold search on val
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.1, 0.9, 0.05):
        yp = (val_probs >= t).astype(int)
        f = f1_score(y_va, yp, average="macro", zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, float(t)
    print(f"\nBest val threshold: {best_t:.2f}  (macro F1={best_f1:.4f})")
    val_metrics = full_report(y_va, (val_probs >= best_t).astype(int), noc_va, "TabPFN — VAL")

    # ── Final fit on train+val, evaluate on test ─────────────────────────────
    print("\nFitting on train+val for final test evaluation...")
    test_probs = np.zeros((len(X_te), n_donors), dtype=np.float32)
    t1 = time.time()
    for d in range(n_donors):
        labels = y_tv[:, d].astype(int)
        if labels.sum() == 0:
            test_probs[:, d] = 0.0
            continue
        clf = TabPFNClassifier(**clf_proto)
        clf.fit(X_tv, labels)
        p = clf.predict_proba(X_te)
        test_probs[:, d] = p[:, 1] if p.shape[1] > 1 else p[:, 0]
        if (d + 1) % 9 == 0:
            print(f"  donor {d+1}/{n_donors}  elapsed={time.time()-t1:.0f}s")

    y_pred = (test_probs >= best_t).astype(int)
    test_metrics = full_report(y_te, y_pred, noc_te, "TabPFN — TEST")

    # ── Save ─────────────────────────────────────────────────────────────────
    np.save(out_dir / "probs_test.npy", test_probs)
    np.save(out_dir / "y_test_pred.npy", y_pred)
    np.save(out_dir / "y_test_true.npy", y_te)

    result = {
        "model": "tabpfn",
        "n_estimators": args.n_est,
        "data_dir": str(data_dir),
        "best_threshold": best_t,
        "val": val_metrics,
        "test": test_metrics,
        "total_time_s": round(time.time() - t0, 1),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_dir}")
    print(f"Total time: {result['total_time_s']:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_est",      type=int, default=8,
                    help="TabPFN n_estimators (default 8)")
    ap.add_argument("--no_noleak",  action="store_true",
                    help="Use data_leaky/ instead of data/ (leaky split)")
    ap.add_argument("--out_subdir", type=str, default="tabpfn",
                    help="results subdir name")
    args = ap.parse_args()
    train(args)
