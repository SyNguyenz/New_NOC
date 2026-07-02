"""
synth/generate_dataset.py  —  Task 2.3

Orchestrate synthetic mixture generation:
  1. Define combo space (train combos — novel, not in real-train)
  2. Write spec.json for simulate_mixtures.R
  3. Call R via subprocess
  4. Convert R output CSV -> tokens/mask/Xflat/y_set/noc .npy arrays
     (same format + allele bins as prepare_data_set.py)

Output:
  data/synth/tokens_synth_{train,val}.npy
  data/synth/mask_synth_{train,val}.npy
  data/synth/Xflat_synth_{train,val}.npy
  data/synth/y_set_synth_{train,val}.npy
  data/synth/noc_synth_{train,val}.npy

Usage:
  python synth/generate_dataset.py [--n-per-combo 500] [--val-frac 0.15]
  python synth/generate_dataset.py --quick   # 50/combo for smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
SYNTH  = DATA / "synth"
SYNTH.mkdir(parents=True, exist_ok=True)

R_EXE     = r"C:\Program Files\R\R-4.6.0\bin\Rscript.exe"
R_SCRIPT  = ROOT / "synth" / "simulate_mixtures.R"

MAX_SEQ = 160


# ── Load metadata ─────────────────────────────────────────────────────────────

def load_meta():
    with open(DATA / "meta_set.json") as f:
        return json.load(f)

def load_names(split):
    with open(DATA / f"meta_sample_names_{split}.json") as f:
        return json.load(f)


# ── Build combo space ─────────────────────────────────────────────────────────

def get_real_train_combos():
    """Combos that appear in the REAL multi-person train split."""
    import re
    def parse_donors(fn):
        parts = fn.split("-")
        if len(parts) < 3:
            return []
        c = parts[2]
        if "_" in c:
            out = []
            for seg in c.split("_"):
                m = re.match(r"^(\d+)", seg)
                if m:
                    out.append(int(m.group(1)))
            return out
        m = re.match(r"^(\d+)", c)
        return [int(m.group(1))] if m else []

    train_names = load_names("train")
    combos = set()
    for fn in train_names:
        d = parse_donors(fn)
        if len(d) >= 2:
            combos.add(tuple(sorted(d)))
    return combos


def build_synthetic_combos(meta, n_per_noc_per_combo: int, val_frac: float,
                            include_novel_only: bool = True):
    """
    Tao combo space cho synthetic.
    - NOC 2-5: lay tat ca C(45, noc) to hop -> filter ra nhung combo:
        a) CHUA co trong real-train combos (novel), OR
        b) Deu duoc dung (neu include_novel_only=False)
    - Giu toi da MAX_COMBOS_PER_NOC de tranh qua lon
    - Chia train/val theo val_frac
    """
    known = meta["known_donors"]  # 45 donors

    real_train_combos = get_real_train_combos()
    print(f"Real train combos: {len(real_train_combos)}")

    MAX_COMBOS_PER_NOC = {2: 50, 3: 50, 4: 40, 5: 30}

    train_specs = []
    val_specs   = []

    for noc in range(2, 6):
        all_combos = list(combinations(known, noc))
        rng = np.random.default_rng(42 + noc)
        rng.shuffle(all_combos)

        if include_novel_only:
            novel = [c for c in all_combos
                     if tuple(sorted(c)) not in real_train_combos]
        else:
            novel = list(all_combos)

        # Cap
        max_c = MAX_COMBOS_PER_NOC.get(noc, 30)
        novel = novel[:max_c]

        n_val   = max(1, int(len(novel) * val_frac))
        n_train = len(novel) - n_val

        for c in novel[:n_train]:
            train_specs.append({
                "combo": list(c),
                "n_mixtures": n_per_noc_per_combo,
                "seed_offset": hash(c) % 10000
            })
        for c in novel[n_train:]:
            val_specs.append({
                "combo": list(c),
                "n_mixtures": max(1, n_per_noc_per_combo // 4),
                "seed_offset": hash(c) % 10000 + 50000
            })

        print(f"  NOC={noc}: {len(novel)} novel combos -> "
              f"{n_train} train / {n_val} val combos")

    print(f"Total train combos: {len(train_specs)}, "
          f"val combos: {len(val_specs)}")
    return train_specs, val_specs


# ── Run R simulator ──────────────────────────────────────────────────────────

def run_r_simulation(specs, out_csv: Path, seed: int = 42) -> bool:
    spec_path = SYNTH / "_current_spec.json"
    with open(spec_path, "w") as f:
        json.dump(specs, f)

    geno_path = SYNTH / "donor_genotypes.csv"
    cmd = [R_EXE, str(R_SCRIPT),
           "--spec", str(spec_path),
           "--out",  str(out_csv),
           "--geno", str(geno_path),
           "--seed", str(seed)]

    print(f"Running R: {len(specs)} combo specs -> {out_csv.name}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))

    if result.stdout:
        print(result.stdout)
    if result.returncode != 0:
        print("R STDERR:", result.stderr[-2000:])
        return False
    return True


# ── Convert R CSV -> numpy arrays ─────────────────────────────────────────────

def allele_to_float(a: str) -> float | None:
    a = str(a).strip()
    if a in ("", "nan", "OL", "None"):
        return None
    if a.upper() == "X":
        return -2.0
    if a.upper() == "Y":
        return -1.0
    try:
        return float(a)
    except ValueError:
        return None


def csv_to_arrays(csv_path: Path, meta: dict) -> dict | None:
    """Convert simulate_mixtures.R output CSV to token/flat arrays."""
    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    print(f"  CSV rows: {len(df)}, columns: {list(df.columns)}")

    # Expected columns from R output:
    # SampleName, Locus, Allele, Height, Size, donor_ids, noc, ratios, sample_id
    # Rename if needed
    col_map = {}
    for c in df.columns:
        cl = c.lower().replace(" ", "").replace("_", "").replace(".", "")
        if cl in ("samplename", "sampname", "sample"):
            col_map[c] = "SampleName"
        elif cl in ("locus", "marker"):        # R outputs 'Marker'
            col_map[c] = "Locus"
        elif cl == "allele":
            col_map[c] = "Allele"
        elif cl == "height":
            col_map[c] = "Height"
    df = df.rename(columns=col_map)

    # Use only rows where HeightAtOrAboveDetectionThreshold is TRUE (if present)
    if "HeightAtOrAboveDetectionThreshold" in df.columns:
        df = df[df["HeightAtOrAboveDetectionThreshold"].astype(str).str.upper() == "TRUE"].copy()

    # Check required columns
    required = {"SampleName", "Locus", "Allele", "Height"}
    if not required.issubset(df.columns):
        print(f"ERROR: missing columns {required - set(df.columns)}. Got: {list(df.columns)}")
        return None

    locus_to_idx   = meta["locus_to_idx"]
    flat_cols_list = meta["flat_cols"]
    known_donors   = meta["known_donors"]
    n_flat         = meta["n_flat"]

    # Build flat column index: (locus, allele_str) -> col_idx
    flat_col_index: dict[tuple, int] = {}
    for idx, col in enumerate(flat_cols_list):
        # col format: "LOCUS_ALLELE"
        parts = col.rsplit("_", 1)
        if len(parts) == 2:
            flat_col_index[(parts[0], parts[1])] = idx

    # Also build from meta locus_bin_lists for lookup
    locus_bin_lists = meta["locus_bin_lists"]
    locus_allele_to_flat: dict[tuple, int] = {}
    for loc, bins in locus_bin_lists.items():
        for b in bins:
            b_str = str(int(b)) if b == int(b) else str(b)
            if b == -2.0:
                b_str = "X"
            elif b == -1.0:
                b_str = "Y"
            col_name = f"{loc}_{b_str}"
            if col_name in flat_cols_list:
                locus_allele_to_flat[(loc, b)] = flat_cols_list.index(col_name)

    def build_label(donor_ids_str: str) -> np.ndarray:
        vec = np.zeros(45, dtype=np.float32)
        for d in donor_ids_str.split(";"):
            try:
                di = int(d)
                if di in known_donors:
                    vec[known_donors.index(di)] = 1.0
            except ValueError:
                pass
        return vec

    # Group by sample_id
    if "sample_id" in df.columns:
        group_col = "sample_id"
    else:
        group_col = "SampleName"

    grouped = df.groupby(group_col)
    sample_ids = list(grouped.groups.keys())
    N = len(sample_ids)

    tokens_arr = np.zeros((N, MAX_SEQ, 3), dtype=np.float32)
    mask_arr   = np.zeros((N, MAX_SEQ), dtype=bool)
    flat_arr   = np.zeros((N, n_flat),  dtype=np.float32)
    label_arr  = np.zeros((N, 45),      dtype=np.float32)
    noc_arr    = np.zeros(N,            dtype=np.int32)

    n_skipped = 0
    for i, sid in enumerate(sample_ids):
        grp = grouped.get_group(sid)

        # Get metadata from first row
        if "donor_ids" in grp.columns:
            label_arr[i] = build_label(str(grp["donor_ids"].iloc[0]))
        if "noc" in grp.columns:
            noc_arr[i] = int(grp["noc"].iloc[0])
        else:
            noc_arr[i] = int(label_arr[i].sum())

        tokens = []
        for _, row in grp.iterrows():
            locus  = str(row["Locus"]).strip()
            if locus not in locus_to_idx:
                continue
            allele_str = str(row["Allele"]).strip()
            av = allele_to_float(allele_str)
            if av is None:
                continue
            try:
                h_val = float(row["Height"])
            except (ValueError, TypeError):
                continue
            if h_val <= 0:
                continue

            locus_idx = float(locus_to_idx[locus])
            log_h     = float(np.log1p(h_val))
            tokens.append((locus_idx, av, log_h))

            # Flat features: find nearest bin
            bins_for_locus = locus_bin_lists.get(locus, [])
            if bins_for_locus:
                b_arr    = np.array(bins_for_locus)
                nearest  = float(b_arr[np.argmin(np.abs(b_arr - av))])
                col_idx  = locus_allele_to_flat.get((locus, nearest))
                if col_idx is not None:
                    flat_arr[i, col_idx] = max(flat_arr[i, col_idx], log_h)

        if not tokens:
            n_skipped += 1
            continue

        n = min(len(tokens), MAX_SEQ)
        tokens_arr[i, :n, :] = np.array(tokens[:n], dtype=np.float32)
        mask_arr[i, :n]      = True

    if n_skipped:
        print(f"  Skipped {n_skipped} empty samples")

    return {
        "tokens": tokens_arr,
        "mask":   mask_arr,
        "Xflat":  flat_arr,
        "y_set":  label_arr,
        "noc":    noc_arr,
        "sample_ids": sample_ids,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-combo", type=int, default=300,
                        help="Mixtures per (combo) in train (default 300)")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Fraction of combos for val (default 0.15)")
    parser.add_argument("--quick", action="store_true",
                        help="Smoke test: 20 mixtures/combo, max 5 combos/NOC")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.quick:
        args.n_per_combo = 20

    meta = load_meta()
    print("=== Building synthetic combo space ===")
    train_specs, val_specs = build_synthetic_combos(
        meta, args.n_per_combo, args.val_frac,
        include_novel_only=True
    )

    total_synth = (sum(s["n_mixtures"] for s in train_specs) +
                   sum(s["n_mixtures"] for s in val_specs))
    print(f"\nExpected synthetic mixtures: ~{total_synth:,}")

    # ── Simulate train ────────────────────────────────────────────────────────
    print("\n=== Simulating TRAIN synthetic mixtures ===")
    train_csv = SYNTH / "raw_synth_train.csv"
    ok = run_r_simulation(train_specs, train_csv, seed=args.seed)
    if not ok:
        print("ERROR: R simulation failed for train")
        sys.exit(1)

    print("\n=== Converting TRAIN CSV -> numpy arrays ===")
    train_arrays = csv_to_arrays(train_csv, meta)
    if train_arrays is None:
        sys.exit(1)

    # ── Simulate val ──────────────────────────────────────────────────────────
    print("\n=== Simulating VAL synthetic mixtures ===")
    val_csv = SYNTH / "raw_synth_val.csv"
    ok = run_r_simulation(val_specs, val_csv, seed=args.seed + 1000)
    if not ok:
        print("ERROR: R simulation failed for val")
        sys.exit(1)

    print("\n=== Converting VAL CSV -> numpy arrays ===")
    val_arrays = csv_to_arrays(val_csv, meta)
    if val_arrays is None:
        sys.exit(1)

    # ── Save ──────────────────────────────────────────────────────────────────
    print("\n=== Saving arrays ===")
    for split, arrays in [("synth_train", train_arrays), ("synth_val", val_arrays)]:
        for key in ("tokens", "mask", "Xflat", "y_set", "noc"):
            p = DATA / f"{key}_{split}.npy"
            np.save(p, arrays[key])
            print(f"  {p.name}: {arrays[key].shape}")
        with open(DATA / f"meta_sample_names_{split}.json", "w") as f:
            json.dump(arrays["sample_ids"], f)

    print("\n=== Summary ===")
    print(f"Synthetic train: {len(train_arrays['sample_ids'])} mixtures")
    print(f"Synthetic val  : {len(val_arrays['sample_ids'])} mixtures")
    for split, arrays in [("train", train_arrays), ("val", val_arrays)]:
        nocs = arrays["noc"]
        for noc in range(2, 6):
            n = (nocs == noc).sum()
            print(f"  {split} NOC={noc}: {n}")

    print("\nDone.")


if __name__ == "__main__":
    main()
