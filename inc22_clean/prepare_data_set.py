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
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths (script lives at inc22_clean/; data/ is OUTPUT only, never shipped in the code bundle) ──
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_FILTERED = ROOT / "data_raw" / "PROVEDIt_1-5-Person CSVs Filtered"
KIT_PATTERN = str(RAW_FILTERED / "*GF29cycles" / "**" / "*.csv")

MAX_SEQ = 160
RANDOM_SEED = 42

# ── Donor split — 10-fold, every donor is unknown exactly once ─────────────
# The seed-42 permutation is cut into 10 blocks of 5; block STR_FOLD is the hold-out unknown set.
# STR_FOLD=0 (the default) is the original split [6, 21, 26, 40, 50] — byte-identical behaviour.
ALL_DONORS = list(range(1, 51))
N_FOLDS = 10
FOLD = int(os.environ.get("STR_FOLD", "0"))
if not 0 <= FOLD < N_FOLDS:
    raise SystemExit(f"STR_FOLD must be in [0, {N_FOLDS - 1}], got {FOLD}")
_rng = np.random.default_rng(RANDOM_SEED)
_shuffled = _rng.permutation(ALL_DONORS).tolist()
UNKNOWN_DONORS = sorted(_shuffled[5 * FOLD: 5 * FOLD + 5])
KNOWN_DONORS = sorted(set(ALL_DONORS) - set(UNKNOWN_DONORS))
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
    DATA_DIR.mkdir(parents=True, exist_ok=True)   # data/ is output-only, absent in a fresh checkout

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

    # ── Split policy ──────────────────────────────────────────────────────────
    #
    #   Single-source (NOC=1): stratify by donor so every donor appears in
    #     train + val + test (reference database recognition, no leak by design).
    #   Multi-person (NOC>=2): EVERY closed combo goes to TEST.
    #
    # Why all of them: the network never trains on a real mixture — make_insilico.build_train keeps
    # the real NOC=1 rows and DISCARDS the real mixtures, training on in-silico only. So parking real
    # combos in `train` protected nothing; it just shrank the evaluation (fold 0 reached 233 of 1333
    # real mixture samples). Instead every real combo is excluded from in-silico generation, and the
    # post-hoc decode stage (phi-rerank alpha + RF count) is fit LEAVE-ONE-COMBO-OUT at eval time so
    # each combo is scored by a decode that never saw it. combo_id_{split}.npy carries the grouping
    # (-1 = single-source). val is single-source only — it supplies the k=1 rows for the RF count.

    # --- Multi-person: every closed combo -> test -------------------------------
    sample_combo = {}
    for i in closed_idx:
        sf = sample_files[i]
        donors_i = sample_donors[sf]
        noc_i = int(nocs[i])
        if noc_i >= 2:
            sample_combo[i] = tuple(sorted(donors_i))

    combos_by_noc: dict[int, list] = {}
    for i, combo in sample_combo.items():
        combos_by_noc.setdefault(len(combo), set()).add(combo)
    combos_by_noc = {noc: sorted(combos) for noc, combos in combos_by_noc.items()}

    all_combos = sorted({c for cs in combos_by_noc.values() for c in cs})
    combo_to_id = {c: j for j, c in enumerate(all_combos)}
    split_policy_combos: dict[str, dict] = {"train": {}, "val": {}, "test": {}}
    for noc_i, combos in sorted(combos_by_noc.items()):
        split_policy_combos["test"][f"NOC{noc_i}"] = [[int(d) for d in c] for c in combos]
        n_samp = sum(1 for c in sample_combo.values() if len(c) == noc_i)
        print(f"  NOC={noc_i}: {len(combos)} combos -> test ({n_samp} samples)")

    multi_test = sorted(sample_combo)

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
    idx_train = ss_train                                   # single-source only
    idx_val   = ss_val                                     # single-source only (k=1 rows for RF count)
    idx_test  = np.concatenate([ss_test, np.array(multi_test, dtype=ss_test.dtype)])

    # combo grouping for leave-one-combo-out decode fitting (-1 = single-source)
    combo_id_all = np.full(len(sample_files), -1, dtype=np.int32)
    for i, c in sample_combo.items():
        combo_id_all[i] = combo_to_id[c]

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
        np.save(DATA_DIR / f"combo_id_{split}.npy", combo_id_all[ix])

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
        "combo_list": [[int(d) for d in c] for c in all_combos],
        "split_policy": {
            "description": (
                "Single-source (NOC=1): stratified by donor (all 45 donors in every split). "
                "Multi-person (NOC>=2): EVERY closed donor-combo goes to test — the network trains "
                "on in-silico mixtures only, so no real combo is spent on train/val. All of them are "
                "excluded from in-silico generation, and the post-hoc decode (phi-rerank alpha + RF "
                "count) is fit leave-one-combo-out via combo_id_test.npy. val is single-source only."
            ),
            "multi_person_combos": split_policy_combos,
        },
    }
    with open(DATA_DIR / "meta_set.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Fold provenance in a SEPARATE file so meta_set.json stays byte-identical to the
    # pre-fold-patch output for STR_FOLD=0 (the already-published inc22 run).
    with open(DATA_DIR / "fold_info.json", "w") as f:
        json.dump({"fold": FOLD, "n_folds": N_FOLDS, "random_seed": RANDOM_SEED,
                   "unknown_donors": UNKNOWN_DONORS, "known_donors": KNOWN_DONORS}, f, indent=2)

    print(f"\nDone. Files saved to: {DATA_DIR}")
    print(f"FOLD          : {FOLD} / {N_FOLDS}")
    print(f"Token shape example (train): {tokens_arr[idx_train].shape}")
    print(f"Flat shape example   (train): {flat_arr[idx_train].shape}")
    print(f"Known donors  : {KNOWN_DONORS}")
    print(f"Unknown donors: {UNKNOWN_DONORS}")


if __name__ == "__main__":
    main()
