"""
extract_size.py — recover per-peak SIZE (bp) that prepare_data_set.py dropped.

The raw GeneMapper CSVs export (Allele N, Size N, Height N) triplets — prepare_data_set.py kept
only Allele+Height. Size(bp) is the missing PG sufficient statistic ({allele, height, size};
design_increment2 §8) and the substrate for the DEGRADATION relationship (height decays with
fragment size; §1/§13). It is FREE — already in the same CSVs, no instrument re-prep ("Increment 4
needs re-prep" was wrong).

This builds a SIDECAR aligned to the EXISTING tokens (does NOT regenerate the core arrays): for each
sample (from meta_sample_names_{split}.json, aligned to the arrays) it joins the CSV peaks by
(locus, allele) and writes the size onto each token position.

Output (data/):  size_{split}.npy  (N, 160) float32   (0.0 on pad / unmatched)

Usage:  python extract_size.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
RAW = ROOT / "data_raw" / "PROVEDIt_1-5-Person CSVs Filtered"
KIT_PATTERN = str(RAW / "*GF29cycles" / "**" / "*.csv")
META = json.load(open(DATA / "meta_set.json"))
LOCUS_TO_IDX = META["locus_to_idx"]
MAX_SEQ = META["max_seq"]


def allele_key(v) -> str:
    s = str(v).strip()
    if s in ("", "nan", "OL", "None"):
        return ""
    if s in ("X", "-2.0", "-2"):
        return "-2.0"
    if s in ("Y", "-1.0", "-1"):
        return "-1.0"
    try:
        return f"{round(float(s), 1):.1f}"
    except ValueError:
        return ""


def build_size_lookup() -> dict[str, dict[tuple, float]]:
    """sample_file -> {(locus_idx, allele_key): size_bp}."""
    files = glob.glob(KIT_PATTERN, recursive=True)
    raw = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    allele_cols = [c for c in raw.columns if c.startswith("Allele ")]
    size_cols = [c.replace("Allele", "Size") for c in allele_cols]
    look: dict[str, dict[tuple, float]] = defaultdict(dict)
    for _, row in raw.iterrows():
        loc = row.get("Marker")
        if loc not in LOCUS_TO_IDX:
            continue
        li = LOCUS_TO_IDX[loc]
        for ac, sc in zip(allele_cols, size_cols):
            ak = allele_key(row.get(ac))
            if not ak or sc not in row:
                continue
            sz = row.get(sc)
            try:
                szf = float(sz)
            except (ValueError, TypeError):
                continue
            if np.isfinite(szf):
                look[str(row["Sample File"])][(li, ak)] = szf
    return look


def main():
    print(f"data: {DATA}\nbuilding size lookup from raw CSVs ...")
    look = build_size_lookup()
    print(f"  samples with size: {len(look)}")
    for split in ["train", "val", "test", "open", "dev"]:
        tp = DATA / f"tokens_{split}.npy"
        npath = DATA / f"meta_sample_names_{split}.json"
        if not tp.exists() or not npath.exists():
            continue
        tok = np.load(tp); mask = np.load(DATA / f"mask_{split}.npy").astype(bool)
        names = json.load(open(npath))
        N = len(tok)
        if len(names) != N:
            print(f"  {split:5s} SKIP: names {len(names)} != tokens {N} (names misaligned)")
            continue
        size = np.zeros((N, MAX_SEQ), np.float32)
        n_valid = n_hit = 0
        for i in range(N):
            smap = look.get(str(names[i]), {})
            for j in range(MAX_SEQ):
                if not mask[i, j]:
                    continue
                n_valid += 1
                key = (int(round(float(tok[i, j, 0]))), allele_key(tok[i, j, 1]))
                if key in smap:
                    size[i, j] = smap[key]; n_hit += 1
        np.save(DATA / f"size_{split}.npy", size)
        cov = n_hit / max(n_valid, 1)
        rng = size[mask & (size > 0)]
        print(f"  {split:5s} N={N:5d}  valid={n_valid:7d}  size-matched={cov:.3f}"
              f"  bp[min={rng.min():.0f} mean={rng.mean():.0f} max={rng.max():.0f}]  -> size_{split}.npy")
    print("Done. Per-peak size(bp) saved to data/.")


if __name__ == "__main__":
    main()
