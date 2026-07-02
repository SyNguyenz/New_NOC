"""
Week 1 — Data preparation for 45-class multi-label classifier.

Reads combined_preprocessed_data.csv (from NOC_DNA pipeline),
parses donor IDs from filenames, builds 45-dim binary labels,
filters to closed-set, and saves train/val/test splits.

Donor IDs 1-50 from PROVEDIt RD14-0003.
45 known donors + 5 hold-out unknowns (seed=42).
"""

import re
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).resolve().parent
PREPROCESSED_CSV = OUT_DIR / "combined_preprocessed_data.csv"

RANDOM_SEED = 42

# ── Donor split ────────────────────────────────────────────────────────────
ALL_DONORS = list(range(1, 51))          # 50 donors, IDs 1-50

rng = np.random.default_rng(RANDOM_SEED)
_shuffled = rng.permutation(ALL_DONORS).tolist()
UNKNOWN_DONORS = sorted(_shuffled[:5])   # 5 hold-out unknowns
KNOWN_DONORS   = sorted(_shuffled[5:])  # 45 known


def parse_donors(filename: str) -> list[int]:
    """Extract contributor donor IDs from a PROVEDIt sample filename.

    Naming convention (after the study-ID segment RD14-0003):
      - 1-person : 01d3a      → [1]
      - multi    : 31_32_33   → [31, 32, 33]
    """
    parts = filename.split("-")
    if len(parts) < 3:
        return []
    contrib_part = parts[2]  # e.g. "31_32" or "01d3a"

    if "_" in contrib_part:
        donors = []
        for seg in contrib_part.split("_"):
            m = re.match(r"^(\d+)", seg)
            if m:
                donors.append(int(m.group(1)))
        return donors
    else:
        m = re.match(r"^(\d+)", contrib_part)
        return [int(m.group(1))] if m else []


def build_label_vector(donor_ids: list[int]) -> np.ndarray:
    """Build a 45-dim binary vector. Index i = 1 iff KNOWN_DONORS[i] ∈ donor_ids."""
    vec = np.zeros(45, dtype=np.float32)
    for d in donor_ids:
        if d in KNOWN_DONORS:
            vec[KNOWN_DONORS.index(d)] = 1.0
    return vec


def main():
    print(f"Loading {PREPROCESSED_CSV} …")
    df = pd.read_csv(PREPROCESSED_CSV, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Parse donors
    df["_donors"] = df["Sample File"].apply(parse_donors)

    # Filter out samples with no parsed donor (shouldn't happen)
    df = df[df["_donors"].map(len) > 0].copy()

    # Closed-set flag: all contributors must be in KNOWN_DONORS
    df["_is_closed"] = df["_donors"].apply(
        lambda ds: all(d in KNOWN_DONORS for d in ds)
    )
    closed_df  = df[df["_is_closed"]].copy()
    open_df    = df[~df["_is_closed"]].copy()
    print(f"  Closed-set samples : {len(closed_df)}")
    print(f"  Open-set  samples  : {len(open_df)}")

    # Build label matrix for closed-set
    label_matrix = np.stack(closed_df["_donors"].apply(build_label_vector).values)
    print(f"  Label matrix shape : {label_matrix.shape}")
    print(f"  Labels per sample  : mean={label_matrix.sum(1).mean():.2f}  "
          f"max={int(label_matrix.sum(1).max())}")

    # Feature columns = everything except Sample File, target_noc, helper cols
    drop_cols = {"Sample File", "target_noc", "_donors", "_is_closed"}
    feature_cols = [c for c in closed_df.columns if c not in drop_cols]
    X = closed_df[feature_cols].fillna(0).values.astype(np.float32)
    print(f"  Feature matrix     : {X.shape}")

    # ── Train / Val / Test split (70 / 15 / 15) ────────────────────────────
    idx = np.arange(len(X))
    idx_train, idx_tmp = train_test_split(idx, test_size=0.30, random_state=RANDOM_SEED)
    idx_val, idx_test  = train_test_split(idx_tmp, test_size=0.50, random_state=RANDOM_SEED)

    splits = {"train": idx_train, "val": idx_val, "test": idx_test}
    for name, ix in splits.items():
        print(f"  {name:5s}: {len(ix):4d} samples")

    # ── Save ───────────────────────────────────────────────────────────────
    np.save(OUT_DIR / "X_train.npy", X[idx_train])
    np.save(OUT_DIR / "X_val.npy",   X[idx_val])
    np.save(OUT_DIR / "X_test.npy",  X[idx_test])
    np.save(OUT_DIR / "y_train.npy", label_matrix[idx_train])
    np.save(OUT_DIR / "y_val.npy",   label_matrix[idx_val])
    np.save(OUT_DIR / "y_test.npy",  label_matrix[idx_test])

    # Save open-set for later integration with OOD module
    X_open = open_df[feature_cols].fillna(0).values.astype(np.float32)
    y_open = np.stack(open_df["_donors"].apply(build_label_vector).values)
    np.save(OUT_DIR / "X_open.npy", X_open)
    np.save(OUT_DIR / "y_open.npy", y_open)
    print(f"  open-set: {len(X_open):4d} samples")

    # Save metadata
    meta = {
        "feature_cols": feature_cols,
        "known_donors": KNOWN_DONORS,
        "unknown_donors": UNKNOWN_DONORS,
        "random_seed": RANDOM_SEED,
        "n_features": len(feature_cols),
        "n_classes": 45,
    }
    with open(OUT_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\nDone. Files saved to:", OUT_DIR)
    print("Known donors :", KNOWN_DONORS)
    print("Unknown donors:", UNKNOWN_DONORS)


if __name__ == "__main__":
    main()
