"""
Week 1 — Set-structured feature extraction for Set Transformer.

Reads raw GF29cycles Filtered CSVs. For each mixture sample produces:
  - Set of tokens (locus_idx, allele_float, log1p_height) — variable length
  - Flat feature vector (one entry per LOCUS_ALLELE bin) — same semantics as
    combined_preprocessed_data.csv but derived directly from peak heights

Outputs (under data/):
  tokens_{split}.npy   (N, MAX_SEQ, 3) float32  — padded, 0 = padding token
  mask_{split}.npy     (N, MAX_SEQ)    bool      — True = valid token
  Xflat_{split}.npy    (N, F)          float32   — flat log1p-height features
  y_{split}.npy        (N, 45)         float32   — multi-label targets
  noc_{split}.npy      (N,)            int32      — number of contributors
  meta_set.json                                   — loci list, allele bins, MAX_SEQ, splits

Kit: 3500_GF29cycles (GlobalFiler, 24 loci).
Donor split: seed=42, same logic as prepare_data.py.
Sample split: 70/15/15 random (random_state=42), same seed.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_FILTERED = ROOT / "data_raw" / "PROVEDIt_1-5-Person CSVs Filtered"
KIT_PATTERN = str(RAW_FILTERED / "*GF29cycles" / "**" / "*.csv")

MAX_SEQ = 160
RANDOM_SEED = 42

# ── Donor split (identical to prepare_data.py) ─────────────────────────────
ALL_DONORS = list(range(1, 51))
_rng = np.random.default_rng(RANDOM_SEED)
_shuffled = _rng.permutation(ALL_DONORS).tolist()
UNKNOWN_DONORS = sorted(_shuffled[:5])
KNOWN_DONORS = sorted(_shuffled[5:])
KNOWN_SET = set(KNOWN_DONORS)


# ── Allele value encoding ──────────────────────────────────────────────────
def allele_to_float(allele: str) -> float | None:
    """Return float representation of an allele string, or None if invalid."""
    a = str(allele).strip()
    if a in ("", "nan", "OL"):
        return None
    if a == "X":
        return -2.0
    if a == "Y":
        return -1.0
    try:
        return float(a)
    except ValueError:
        return None


# ── Donor parsing (identical logic to prepare_data.py) ────────────────────
def parse_donors(filename: str) -> list[int]:
    parts = filename.split("-")
    if len(parts) < 3:
        return []
    contrib_part = parts[2]
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
    vec = np.zeros(45, dtype=np.float32)
    for d in donor_ids:
        if d in KNOWN_SET:
            vec[KNOWN_DONORS.index(d)] = 1.0
    return vec


# ── Main ───────────────────────────────────────────────────────────────────
def main():

    print("Scanning GF29cycles CSVs …")
    csv_files = glob.glob(KIT_PATTERN, recursive=True)
    print(f"  Found {len(csv_files)} CSV files")

    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    raw = pd.concat(dfs, ignore_index=True)
    print(f"  Total locus rows: {len(raw)}")

    allele_cols = [c for c in raw.columns if c.startswith("Allele ")]
    height_cols = [c.replace("Allele", "Height") for c in allele_cols]

    # Build canonical locus list (sorted, reproducible)
    loci = sorted(raw["Marker"].dropna().unique().tolist())
    locus_to_idx = {loc: i for i, loc in enumerate(loci)}
    print(f"  Loci ({len(loci)}): {loci}")

    # First pass: collect all unique LOCUS_ALLELE bins for flat feature matrix
    print("First pass: collecting allele bins …")
    allele_bins: dict[str, set] = {loc: set() for loc in loci}
    for _, row in raw.iterrows():
        locus = row["Marker"]
        if locus not in locus_to_idx:
            continue
        for ac in allele_cols:
            av = allele_to_float(row[ac])
            if av is not None:
                allele_bins[locus].add(av)

    # Sort bins per locus; build flat feature column list
    locus_bin_lists: dict[str, list[float]] = {
        loc: sorted(bins) for loc, bins in allele_bins.items()
    }
    flat_cols: list[str] = []
    flat_col_index: dict[tuple[str, float], int] = {}
    for loc in loci:
        for av in locus_bin_lists[loc]:
            allele_str = str(int(av)) if av == int(av) else str(av)
            if av == -2.0:
                allele_str = "X"
            elif av == -1.0:
                allele_str = "Y"
            col_name = f"{loc}_{allele_str}"
            flat_col_index[(loc, av)] = len(flat_cols)
            flat_cols.append(col_name)
    n_flat = len(flat_cols)
    print(f"  Flat feature dims: {n_flat}")

    # Second pass: build per-sample token sets and flat vectors
    print("Second pass: building sample features …")
    sample_tokens: dict[str, list[tuple[float, float, float]]] = {}
    sample_flats: dict[str, np.ndarray] = {}
    sample_donors: dict[str, list[int]] = {}

    for sf, grp in raw.groupby("Sample File"):
        donors = parse_donors(str(sf))
        if not donors:
            continue
        tokens: list[tuple[float, float, float]] = []
        flat = np.zeros(n_flat, dtype=np.float32)
        for _, row in grp.iterrows():
            locus = row["Marker"]
            if locus not in locus_to_idx:
                continue
            locus_idx = float(locus_to_idx[locus])
            for ac, hc in zip(allele_cols, height_cols):
                av = allele_to_float(row[ac])
                if av is None:
                    continue
                h = row.get(hc, None)
                if pd.isna(h):
                    continue
                try:
                    h_val = float(h)
                except (ValueError, TypeError):
                    continue
                log_h = float(np.log1p(h_val))
                tokens.append((locus_idx, av, log_h))
                col_idx = flat_col_index.get((locus, av))
                if col_idx is not None:
                    flat[col_idx] = max(flat[col_idx], log_h)
        if not tokens:
            continue
        sample_tokens[sf] = tokens
        sample_flats[sf] = flat
        sample_donors[sf] = donors

    sample_files = sorted(sample_tokens.keys())
    print(f"  Valid samples: {len(sample_files)}")

    # Build arrays
    labels = np.stack([build_label_vector(sample_donors[sf]) for sf in sample_files])
    nocs = labels.sum(axis=1).astype(np.int32)
    has_unknown = np.array(
        [any(d not in KNOWN_SET for d in sample_donors[sf]) for sf in sample_files],
        dtype=bool,
    )
    is_closed = ~has_unknown

    # Pad tokens
    tokens_arr = np.zeros((len(sample_files), MAX_SEQ, 3), dtype=np.float32)
    mask_arr = np.zeros((len(sample_files), MAX_SEQ), dtype=bool)
    for i, sf in enumerate(sample_files):
        toks = sample_tokens[sf]
        n = min(len(toks), MAX_SEQ)
        tokens_arr[i, :n, :] = np.array(toks[:n], dtype=np.float32)
        mask_arr[i, :n] = True

    flat_arr = np.stack([sample_flats[sf] for sf in sample_files])

    # ── Closed-set vs open-set ────────────────────────────────────────────────
    closed_idx = np.where(is_closed)[0]
    open_idx   = np.where(~is_closed)[0]
    print(f"  Closed-set: {len(closed_idx)}  Open-set: {len(open_idx)}")

    # ── Group-aware split (Phase-1 fix) ───────────────────────────────────────
    #
    # Rationale:
    #   Single-source (NOC=1): stratify by donor so every donor appears in
    #     train + val + test (reference database recognition, no leak by design).
    #   Multi-person (NOC>=2): assign ENTIRE combos to a split — no replicate
    #     of the same donor-combo can span train and test. This is the fix for
    #     the 20-mixture memorization issue.
    #
    # Combo assignments (seed=42 shuffle within each NOC, last=test, 2nd-last=val):
    #   NOC=2 (5 combos): 3 train / 1 val / 1 test
    #   NOC=3 (6 combos): 4 train / 1 val / 1 test
    #   NOC=4 (4 combos): 2 train / 1 val / 1 test
    #   NOC=5 (5 combos): 3 train / 1 val / 1 test

    rng_split = np.random.default_rng(RANDOM_SEED)

    # --- Multi-person: group by combo -----------------------------------------
    # Map each closed-set sample to its donor combo
    sample_combo = {}
    for i in closed_idx:
        sf = sample_files[i]
        donors_i = sample_donors[sf]
        noc_i = int(nocs[i])
        if noc_i >= 2:
            sample_combo[i] = tuple(sorted(donors_i))

    # Find all unique combos per NOC and shuffle deterministically
    combos_by_noc: dict[int, list] = {}
    for i, combo in sample_combo.items():
        noc_i = len(combo)
        combos_by_noc.setdefault(noc_i, set()).add(combo)
    combos_by_noc = {noc: sorted(combos) for noc, combos in combos_by_noc.items()}

    combo_split: dict[tuple, str] = {}   # combo -> "train"/"val"/"test"
    split_policy_combos: dict[str, dict] = {"train": {}, "val": {}, "test": {}}

    for noc_i, combos in sorted(combos_by_noc.items()):
        shuffled = [tuple(c) for c in rng_split.permutation(combos)]
        # last -> test, second-to-last -> val, rest -> train
        test_combo = shuffled[-1]
        val_combo  = shuffled[-2]
        train_combos_noc = shuffled[:-2]
        combo_split[test_combo] = "test"
        combo_split[val_combo]  = "val"
        for c in train_combos_noc:
            combo_split[c] = "train"
        split_policy_combos["test"][f"NOC{noc_i}"] = [[int(d) for d in test_combo]]
        split_policy_combos["val"][f"NOC{noc_i}"]  = [[int(d) for d in val_combo]]
        split_policy_combos["train"][f"NOC{noc_i}"] = [[int(d) for d in c] for c in train_combos_noc]
        print(f"  NOC={noc_i} combos: train={[[int(d) for d in c] for c in train_combos_noc]}"
              f"  val={[int(d) for d in val_combo]}  test={[int(d) for d in test_combo]}")

    # Collect multi-person indices per split
    multi_train = [i for i, c in sample_combo.items() if combo_split[c] == "train"]
    multi_val   = [i for i, c in sample_combo.items() if combo_split[c] == "val"]
    multi_test  = [i for i, c in sample_combo.items() if combo_split[c] == "test"]

    # --- Single-source: stratify by donor -------------------------------------
    single_idx = np.array([i for i in closed_idx if int(nocs[i]) == 1])
    donor_labels = np.array([
        KNOWN_DONORS.index(sample_donors[sample_files[i]][0])
        for i in single_idx
    ])

    ss_train, ss_tmp, _, ss_lbl_tmp = train_test_split(
        single_idx, donor_labels,
        test_size=0.30, random_state=RANDOM_SEED, stratify=donor_labels
    )
    ss_val, ss_test = train_test_split(
        ss_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=ss_lbl_tmp
    )

    # --- Combine --------------------------------------------------------------
    idx_train = np.concatenate([ss_train, multi_train])
    idx_val   = np.concatenate([ss_val,   multi_val])
    idx_test  = np.concatenate([ss_test,  multi_test])

    splits = {"train": idx_train, "val": idx_val, "test": idx_test, "open": open_idx}
    for name, ix in splits.items():
        n_ss  = int((nocs[ix] == 1).sum())
        n_mix = int((nocs[ix] >= 2).sum())
        print(f"  {name:5s}: {len(ix):5d} samples  (NOC=1: {n_ss}, NOC>=2: {n_mix})")

    # ── Save arrays ───────────────────────────────────────────────────────────
    print("Saving …")
    for split, ix in splits.items():
        np.save(DATA_DIR / f"tokens_{split}.npy",  tokens_arr[ix])
        np.save(DATA_DIR / f"mask_{split}.npy",    mask_arr[ix])
        np.save(DATA_DIR / f"Xflat_{split}.npy",   flat_arr[ix])
        np.save(DATA_DIR / f"y_{split}_set.npy",   labels[ix])
        np.save(DATA_DIR / f"noc_{split}.npy",     nocs[ix])

    # Save sample names for audit / metadata scripts
    for split, ix in splits.items():
        names = [sample_files[i] for i in ix]
        with open(DATA_DIR / f"meta_sample_names_{split}.json", "w") as f:
            json.dump(names, f)

    meta = {
        "kit": "3500_GF29cycles",
        "loci": loci,
        "locus_to_idx": locus_to_idx,
        "locus_bin_lists": {
            loc: [float(v) for v in bins]
            for loc, bins in locus_bin_lists.items()
        },
        "flat_cols": flat_cols,
        "n_flat": n_flat,
        "max_seq": MAX_SEQ,
        "known_donors": KNOWN_DONORS,
        "unknown_donors": UNKNOWN_DONORS,
        "random_seed": RANDOM_SEED,
        "n_classes": 45,
        "split_sizes": {k: len(v) for k, v in splits.items()},
        "split_policy": {
            "description": (
                "Single-source (NOC=1): stratified by donor (all 45 donors in every split). "
                "Multi-person (NOC>=2): group-aware by donor-combo — entire combo "
                "assigned to one split, no combo spans train and test."
            ),
            "multi_person_combos": split_policy_combos,
        },
    }
    with open(DATA_DIR / "meta_set.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Files saved to: {DATA_DIR}")
    print(f"Token shape example (train): {tokens_arr[idx_train].shape}")
    print(f"Flat shape example   (train): {flat_arr[idx_train].shape}")
    print(f"Known donors  : {KNOWN_DONORS}")
    print(f"Unknown donors: {UNKNOWN_DONORS}")


if __name__ == "__main__":
    main()
