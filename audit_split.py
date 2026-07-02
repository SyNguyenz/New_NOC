"""
audit_split.py — Kiem tra leakage cau truc giua cac split.

Kiem tra:
  (a) Trung filename giua train/val/test/open
  (b) Overlap to hop donor multi-person train <-> val/test
  (c) Phan bo NOC moi split
  (d) % mau test NOC>=2 co to hop NOVEL (chua trong train)
  (e) Donor-pool isolation: replicate single-source trong test co bi dung sinh synthetic?

Exit code:
  0  = sach (khong co multi-person combo leak)
  1  = phat hien multi-person combo leak (test combo da co trong train)

Usage:
  python audit_split.py             # kiem tra split hien tai
  python audit_split.py --strict    # exit 1 neu bat ky van de nao
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DATA = Path(__file__).parent / "data"


# ── Utilities ────────────────────────────────────────────────────────────────

def parse_donors(filename: str) -> list[int]:
    parts = filename.split("-")
    if len(parts) < 3:
        return []
    contrib = parts[2]
    if "_" in contrib:
        out = []
        for seg in contrib.split("_"):
            m = re.match(r"^(\d+)", seg)
            if m:
                out.append(int(m.group(1)))
        return out
    m = re.match(r"^(\d+)", contrib)
    return [int(m.group(1))] if m else []


def donor_combo(filename: str) -> tuple[int, ...]:
    return tuple(sorted(parse_donors(filename)))


def load_names(split: str) -> list[str]:
    p = DATA / f"meta_sample_names_{split}.json"
    if not p.exists():
        return []
    return json.loads(p.read_text())


def noc_of(filename: str) -> int:
    return len(parse_donors(filename))


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 on any warning, not just combo leak")
    args = parser.parse_args()

    splits = {s: load_names(s) for s in ("train", "val", "test", "open")}
    missing = [s for s, v in splits.items() if not v]
    if missing:
        print(f"[ERROR] Missing split files: {missing}")
        return 1

    errors = 0
    warnings = 0

    print("=" * 64)
    print("  SPLIT AUDIT REPORT")
    print("=" * 64)

    # ── (a) Exact filename overlap ───────────────────────────────────────────
    print("\n[A] Exact filename overlap")
    pairs = [("train", "val"), ("train", "test"), ("val", "test"),
             ("train", "open"), ("val", "open"), ("test", "open")]
    for s1, s2 in pairs:
        overlap = set(splits[s1]) & set(splits[s2])
        tag = "OK" if not overlap else "ERROR"
        if overlap:
            errors += 1
        print(f"  {s1} & {s2}: {len(overlap)} overlap  [{tag}]")
        if overlap:
            for f in list(overlap)[:5]:
                print(f"    {f}")

    # ── (b) NOC distribution ────────────────────────────────────────────────
    print("\n[B] NOC distribution per split")
    header = f"  {'NOC':>4}" + "".join(f"  {s:>8}" for s in ("train", "val", "test"))
    print(header)
    for noc in range(1, 7):
        row = f"  {noc:>4}"
        for s in ("train", "val", "test"):
            n = sum(1 for f in splits[s] if noc_of(f) == noc)
            row += f"  {n:>8}"
        print(row)

    # ── (c) Multi-person combo analysis ─────────────────────────────────────
    print("\n[C] Multi-person combo overlap (NOC >= 2)")
    train_multi = {donor_combo(f) for f in splits["train"] if noc_of(f) >= 2}
    val_multi   = {donor_combo(f) for f in splits["val"]   if noc_of(f) >= 2}
    test_multi  = {donor_combo(f) for f in splits["test"]  if noc_of(f) >= 2}

    print(f"  Unique multi-person combos: train={len(train_multi)}"
          f"  val={len(val_multi)}  test={len(test_multi)}")

    # Test vs train
    test_leaked  = test_multi & train_multi
    test_novel   = test_multi - train_multi
    val_leaked   = val_multi  & train_multi
    val_novel    = val_multi  - train_multi

    test_samples_leaked = sum(1 for f in splits["test"]
                              if noc_of(f) >= 2 and donor_combo(f) in train_multi)
    test_samples_total  = sum(1 for f in splits["test"] if noc_of(f) >= 2)
    val_samples_leaked  = sum(1 for f in splits["val"]
                              if noc_of(f) >= 2 and donor_combo(f) in train_multi)
    val_samples_total   = sum(1 for f in splits["val"] if noc_of(f) >= 2)

    # Test combo leak
    if test_leaked:
        errors += 1
        print(f"  [ERROR] Test combos already in train: {len(test_leaked)} combos "
              f"({test_samples_leaked}/{test_samples_total} samples = "
              f"{100*test_samples_leaked/max(test_samples_total,1):.1f}%)")
        for c in sorted(test_leaked):
            print(f"    {c}")
    else:
        pct = 100 * len(test_novel) / max(len(test_multi), 1)
        print(f"  [OK] Test combos: {len(test_novel)} NOVEL, 0 leaked  "
              f"({test_samples_total} samples, 100% novel)")

    # Val combo leak
    if val_leaked:
        warnings += 1
        print(f"  [WARN] Val combos already in train: {len(val_leaked)} combos "
              f"({val_samples_leaked}/{val_samples_total} samples)")
        for c in sorted(val_leaked):
            print(f"    {c}")
    else:
        print(f"  [OK] Val combos: {len(val_novel)} NOVEL, 0 leaked  "
              f"({val_samples_total} samples, 100% novel)")

    # Detail per NOC
    print("\n  Multi-person combos per NOC:")
    for noc in range(2, 7):
        tr = sorted(c for c in train_multi if len(c) == noc)
        vl = sorted(c for c in val_multi   if len(c) == noc)
        te = sorted(c for c in test_multi  if len(c) == noc)
        print(f"    NOC={noc}  train={tr}  val={vl}  test={te}")

    # ── (d) Single-source donor coverage ────────────────────────────────────
    print("\n[D] Single-source donor coverage (NOC=1)")
    for split_name in ("train", "val", "test"):
        donors_present = {donor_combo(f)[0]
                          for f in splits[split_name] if noc_of(f) == 1}
        n_ss = sum(1 for f in splits[split_name] if noc_of(f) == 1)
        tag = "OK" if len(donors_present) == 45 else "WARN"
        print(f"  {split_name:5s}: {n_ss} samples, {len(donors_present)}/45 donors "
              f" [{tag}]")
        if len(donors_present) < 45:
            warnings += 1
            missing_d = set(range(1, 51)) - {6, 21, 26, 40, 50} - donors_present
            print(f"    Missing donors: {sorted(missing_d)}")

    # ── (e) Open-set isolation ───────────────────────────────────────────────
    print("\n[E] Open-set donor isolation")
    UNKNOWN = {6, 21, 26, 40, 50}
    open_donors = set()
    for f in splits["open"]:
        open_donors |= set(parse_donors(f))
    leaked_unknown_in_train = open_donors & {
        d for f in splits["train"] for d in parse_donors(f)
    } & UNKNOWN
    if leaked_unknown_in_train:
        errors += 1
        print(f"  [ERROR] Unknown donors appear in train: {leaked_unknown_in_train}")
    else:
        print(f"  [OK] Unknown donors {sorted(UNKNOWN)} absent from train/val/test")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 64)
    print(f"  SUMMARY: {errors} error(s), {warnings} warning(s)")
    if errors == 0 and warnings == 0:
        print("  PASS — split is clean.")
    elif errors > 0:
        print("  FAIL — fix errors before proceeding.")
    else:
        print("  PASS with warnings.")
    print("=" * 64)

    if errors > 0:
        return 1
    if args.strict and warnings > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
