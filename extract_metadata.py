"""
Re-derive sample-level metadata for stratification analysis.

PROVEDIt encodes per-sample conditions in the "Sample File" column:
  A02_RD14-0003-15d2U60-0.25GF-Q4.5_01.15sec.hid
       └─proj─┘ └donor┘└temp┘└Q ┘ └inj─┘
       └──contrib_part─┘└tmpl┘└qual┘└replicate.injection_time┘

This script re-scans GF29cycles CSVs, recovers the same sorted sample list
that prepare_data_set.py used, then extracts:
  - template_ng         (float, ng of input DNA)
  - q_index             (float, PROVEDIt quality index)
  - injection_sec       (int, capillary injection time in seconds)

Outputs (under data/):
  meta_template_{split}.npy   float32 (N,)   NaN if not parseable
  meta_qindex_{split}.npy     float32 (N,)
  meta_injection_{split}.npy  int32   (N,)   0 if not parseable
  meta_sample_names_{split}.json   list[str]  for debug

Splits use the same RANDOM_SEED=42 logic as prepare_data_set.py.
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_FILTERED = ROOT / "data_raw" / "PROVEDIt_1-5-Person CSVs Filtered"
KIT_PATTERN = str(RAW_FILTERED / "*GF29cycles" / "**" / "*.csv")

RANDOM_SEED = 42
ALL_DONORS = list(range(1, 51))
_rng = np.random.default_rng(RANDOM_SEED)
_shuffled = _rng.permutation(ALL_DONORS).tolist()
UNKNOWN_DONORS = sorted(_shuffled[:5])
KNOWN_DONORS = sorted(_shuffled[5:])
KNOWN_SET = set(KNOWN_DONORS)


# ── Filename parsing (mirror prepare_data_set.py) ─────────────────────────

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
    m = re.match(r"^(\d+)", contrib_part)
    return [int(m.group(1))] if m else []


# ── Metadata regex patterns ───────────────────────────────────────────────

RE_TEMPLATE  = re.compile(r"-([\d\.]+)GF\b")
RE_TEMPLATE_GENERIC = re.compile(r"-([\d\.]+)(?:GF|IP|PP)\b")
RE_QINDEX    = re.compile(r"-Q([\d\.]+)")
RE_INJECTION = re.compile(r"_(\d+)\.?(\d+)?sec(?:\.hid)?\b")


def parse_template_ng(sample_file: str) -> float:
    """Extract template DNA mass in ng. Format: -0.25GF, -0.0625IP, etc."""
    m = RE_TEMPLATE.search(sample_file) or RE_TEMPLATE_GENERIC.search(sample_file)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("nan")


def parse_q_index(sample_file: str) -> float:
    """Extract PROVEDIt quality index. Format: -Q4.5, -Q14.4"""
    m = RE_QINDEX.search(sample_file)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return float("nan")


def parse_injection_sec(sample_file: str) -> int:
    """
    Extract capillary injection time in seconds.
    Format: _01.15sec.hid (replicate 01, 15 sec) or _15sec.hid (15 sec).
    The integer right before 'sec' is the injection time.
    """
    # Pattern: any digits immediately preceding 'sec'
    m = re.search(r"(\d+)sec", sample_file)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            pass
    return 0


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Scanning GF29cycles CSVs …")
    csv_files = glob.glob(KIT_PATTERN, recursive=True)
    print(f"  Found {len(csv_files)} CSV files")

    dfs = [pd.read_csv(f, low_memory=False, usecols=["Sample File"])
           for f in csv_files]
    raw = pd.concat(dfs, ignore_index=True)
    sample_names_all = raw["Sample File"].dropna().unique().tolist()
    print(f"  Unique sample names: {len(sample_names_all)}")

    # Filter to samples whose contributors are derivable (same as prepare_data_set)
    valid_samples = []
    has_unknown = []
    for sf in sample_names_all:
        donors = parse_donors(str(sf))
        if not donors:
            continue
        valid_samples.append(sf)
        has_unknown.append(any(d not in KNOWN_SET for d in donors))
    print(f"  Valid samples: {len(valid_samples)}")

    # Sort to match prepare_data_set.py exactly (line 181: sorted(sample_tokens.keys()))
    order = np.argsort(valid_samples)
    sample_files = [valid_samples[i] for i in order]
    has_unknown_arr = np.array([has_unknown[i] for i in order], dtype=bool)
    is_closed = ~has_unknown_arr

    closed_idx = np.where(is_closed)[0]
    open_idx   = np.where(~is_closed)[0]
    print(f"  Closed-set: {len(closed_idx)}  Open-set: {len(open_idx)}")

    # Reproduce 70/15/15 split with same seed
    idx_train, idx_tmp = train_test_split(
        closed_idx, test_size=0.30, random_state=RANDOM_SEED
    )
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, random_state=RANDOM_SEED
    )
    splits = {"train": idx_train, "val": idx_val, "test": idx_test, "open": open_idx}

    # Sanity check against existing data
    existing = {
        s: int(np.load(DATA_DIR / f"tokens_{s}.npy", mmap_mode="r").shape[0])
        for s in ("train", "val", "test", "open")
    }
    print("\nSize check (existing tokens_*.npy vs reproduced):")
    for s, ix in splits.items():
        ok = "OK" if len(ix) == existing[s] else "MISMATCH"
        print(f"  {s:5s} reproduced={len(ix):5d}  existing={existing[s]:5d}  [{ok}]")

    # Extract metadata + save per split
    print("\nExtracting metadata …")
    for split, ix in splits.items():
        names = [sample_files[i] for i in ix]
        tpl = np.array([parse_template_ng(n)   for n in names], dtype=np.float32)
        q   = np.array([parse_q_index(n)       for n in names], dtype=np.float32)
        inj = np.array([parse_injection_sec(n) for n in names], dtype=np.int32)
        np.save(DATA_DIR / f"meta_template_{split}.npy",  tpl)
        np.save(DATA_DIR / f"meta_qindex_{split}.npy",    q)
        np.save(DATA_DIR / f"meta_injection_{split}.npy", inj)
        with open(DATA_DIR / f"meta_sample_names_{split}.json", "w") as f:
            json.dump(names, f)

        ok_tpl = int(np.isfinite(tpl).sum())
        ok_q   = int(np.isfinite(q).sum())
        ok_inj = int((inj > 0).sum())
        print(f"  {split:5s} n={len(ix):5d}  template_ok={ok_tpl:5d}"
              f"  q_ok={ok_q:5d}  inj_ok={ok_inj:5d}")
        if len(ix) > 0:
            print(f"         tpl range=[{np.nanmin(tpl):.3f}, {np.nanmax(tpl):.3f}]"
                  f"  q range=[{np.nanmin(q):.2f}, {np.nanmax(q):.2f}]"
                  f"  inj uniq={sorted(set(inj.tolist()))}")

    print("\nDone. Stratification arrays saved to data/.")


if __name__ == "__main__":
    main()
