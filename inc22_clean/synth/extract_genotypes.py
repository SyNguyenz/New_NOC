"""
synth/extract_genotypes.py  —  Task 2.1

Trich genotype consensus cho 45 known donor tu single-source (NOC=1)
mau thuoc TRAIN-SPLIT.

Thuat toan consensus:
  - Moi donor, moi locus: thu thap tat ca (allele, height) tu moi replicate
    trong train.
  - Allele duoc coi la "that" neu xuat hien >= MIN_REP_FRAC replicate VA
    height trung binh >= MIN_HEIGHT_RFU.
  - Giu toi da 2 allele co count cao nhat → {allele1, allele2}.
  - Homozygote: neu chi co 1 allele → allele1 = allele2.

QC bat buoc:
  - Kiem tra moi allele trong 20 hon hop THAT (multi-person train+test) nam
    trong union genotype cac donor thanh phan.
  - Log ti le khop (coverage) va bat ki allele missing.

Output:
  data/synth/donor_genotypes.csv  —  45 x 24 loci
    cols: donor_id, locus, allele1, allele2

Usage:
  python synth/extract_genotypes.py
  python synth/extract_genotypes.py --qc-only   # chi chay QC tren file da co
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1]
DATA   = ROOT / "data"
OUT    = DATA / "synth"
OUT.mkdir(parents=True, exist_ok=True)

# Nguong consensus
MIN_REP_FRAC = 0.30   # allele phai co mat trong >= 30% replicate
MIN_HEIGHT   = 50.0   # log1p(height) >= log1p(50) ~ 3.93 → height >= 50 RFU

# ── Load metadata ─────────────────────────────────────────────────────────────

def load_meta():
    with open(DATA / "meta_set.json") as f:
        return json.load(f)

def load_names(split):
    with open(DATA / f"meta_sample_names_{split}.json") as f:
        return json.load(f)

def parse_donors(filename: str) -> list[int]:
    parts = filename.split("-")
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


# ── Core extraction ──────────────────────────────────────────────────────────

def extract_genotypes():
    meta        = load_meta()
    loci        = meta["loci"]               # 24 loci, sorted
    locus_to_idx = meta["locus_to_idx"]
    known_donors = meta["known_donors"]      # 45 donors

    train_names = load_names("train")
    tokens_train = np.load(DATA / "tokens_train.npy")   # (N, 160, 3)
    mask_train   = np.load(DATA / "mask_train.npy")     # (N, 160)
    noc_train    = np.load(DATA / "noc_train.npy")      # (N,)
    y_train      = np.load(DATA / "y_train_set.npy")    # (N, 45)

    # Filter to NOC=1 only
    ss_mask = noc_train == 1
    ss_idx  = np.where(ss_mask)[0]
    print(f"Single-source train samples: {len(ss_idx)}")

    # For each single-source sample, identify donor (unique label in y)
    # allele_data[donor_id][locus_name] = list of (allele_val, log1p_height)
    allele_data: dict[int, dict[str, list]] = {d: defaultdict(list) for d in known_donors}

    idx_to_locus = {v: k for k, v in locus_to_idx.items()}

    for i in ss_idx:
        # Find donor from y label
        y_row = y_train[i]
        pos = np.where(y_row > 0.5)[0]
        if len(pos) != 1:
            continue
        donor_id = known_donors[int(pos[0])]

        tokens = tokens_train[i]  # (160, 3)
        valid  = mask_train[i]    # (160,)
        for j in range(160):
            if not valid[j]:
                break
            locus_idx = int(round(tokens[j, 0]))
            allele    = float(tokens[j, 1])
            log_h     = float(tokens[j, 2])
            if log_h < np.log1p(MIN_HEIGHT):
                continue
            locus_name = idx_to_locus.get(locus_idx)
            if locus_name is None or locus_name == "AMEL":
                continue
            allele_data[donor_id][locus_name].append((allele, log_h))

    # Consensus per donor per locus
    records = []
    missing_loci = []

    for donor_id in known_donors:
        donor_replicates = sum(
            1 for i in ss_idx
            if known_donors[int(np.where(y_train[i] > 0.5)[0][0])] == donor_id
        )
        min_reps = max(1, int(MIN_REP_FRAC * donor_replicates))

        for locus in loci:
            if locus == "AMEL":
                # AMEL: synthetic mixtures typically skip or hardcode
                records.append({
                    "donor_id": donor_id, "locus": locus,
                    "allele1": "X", "allele2": "X"
                })
                continue

            peaks = allele_data[donor_id].get(locus, [])
            if not peaks:
                missing_loci.append((donor_id, locus))
                records.append({
                    "donor_id": donor_id, "locus": locus,
                    "allele1": None, "allele2": None
                })
                continue

            # Count occurrences of each allele value (round to 1 decimal)
            count: dict[str, int] = defaultdict(int)
            height_sum: dict[str, float] = defaultdict(float)
            for allele, log_h in peaks:
                key = str(round(allele, 1))
                count[key] += 1
                height_sum[key] += np.expm1(log_h)

            # Filter by min replicate count
            valid_alleles = {k: count[k] for k in count if count[k] >= min_reps}

            if not valid_alleles:
                # Relax: take top-2 by count regardless
                sorted_alleles = sorted(count, key=lambda k: count[k], reverse=True)
                valid_alleles = {k: count[k] for k in sorted_alleles[:2]}

            # Top-2 by count (break ties by height)
            top2 = sorted(
                valid_alleles.keys(),
                key=lambda k: (valid_alleles[k], height_sum.get(k, 0)),
                reverse=True
            )[:2]

            # Numeric sort for consistent ordering
            top2_num = sorted(top2, key=lambda x: float(x))

            a1 = top2_num[0]
            a2 = top2_num[1] if len(top2_num) > 1 else top2_num[0]

            records.append({
                "donor_id": donor_id, "locus": locus,
                "allele1": a1, "allele2": a2
            })

    df = pd.DataFrame(records)
    print(f"\nGenotype table: {len(df)} rows ({len(known_donors)} donors x {len(loci)} loci)")
    if missing_loci:
        print(f"WARNING: {len(missing_loci)} (donor,locus) pairs had no data:")
        for d, l in missing_loci[:10]:
            print(f"  donor={d} locus={l}")

    out_path = OUT / "donor_genotypes.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved -> {out_path}")
    return df


# ── QC: check real mixtures ───────────────────────────────────────────────────

def qc_against_real(df: pd.DataFrame):
    """
    Kiem tra: moi allele trong 20 hon hop that (multi-person)
    phai nam trong union genotype cac donor thanh phan.
    """
    meta         = load_meta()
    loci         = meta["loci"]
    locus_to_idx = meta["locus_to_idx"]
    idx_to_locus = {v: k for k, v in locus_to_idx.items()}
    known_donors = meta["known_donors"]

    # Build genotype lookup: donor_id -> locus -> {allele_strings}
    geno: dict[int, dict[str, set]] = defaultdict(lambda: defaultdict(set))
    for _, row in df.iterrows():
        if pd.isna(row["allele1"]):
            continue
        donor_id = int(row["donor_id"])
        locus    = row["locus"]
        geno[donor_id][locus].add(str(row["allele1"]))
        geno[donor_id][locus].add(str(row["allele2"]))

    # Collect all multi-person samples from ALL splits
    all_missing = []
    total_checked = 0

    for split in ("train", "val", "test"):
        names  = load_names(split)
        tokens = np.load(DATA / f"tokens_{split}.npy")
        mask   = np.load(DATA / f"mask_{split}.npy")
        nocs   = np.load(DATA / f"noc_{split}.npy")

        for i, sf in enumerate(names):
            if int(nocs[i]) < 2:
                continue
            donors_in_mix = [d for d in parse_donors(sf) if d in set(known_donors)]
            if not donors_in_mix:
                continue

            union_alleles: dict[str, set] = defaultdict(set)
            for d in donors_in_mix:
                for loc, als in geno[d].items():
                    union_alleles[loc] |= als

            # Check observed alleles in mixture
            toks  = tokens[i]
            valid = mask[i]
            for j in range(160):
                if not valid[j]:
                    break
                locus_idx = int(round(toks[j, 0]))
                allele    = round(float(toks[j, 1]), 1)
                log_h     = float(toks[j, 2])
                if log_h < np.log1p(MIN_HEIGHT):
                    continue
                locus_name = idx_to_locus.get(locus_idx)
                if locus_name is None or locus_name == "AMEL":
                    continue
                total_checked += 1
                expected = union_alleles.get(locus_name, set())
                if str(allele) not in expected:
                    all_missing.append({
                        "sample": sf, "locus": locus_name,
                        "allele": allele, "donors": donors_in_mix
                    })

    coverage = 1.0 - len(all_missing) / max(total_checked, 1)
    print(f"\n=== QC: Real mixture allele coverage ===")
    print(f"Total allele observations checked: {total_checked}")
    print(f"Missing from consensus genotype  : {len(all_missing)}")
    print(f"Coverage: {coverage*100:.2f}%")

    if all_missing:
        # Show first 10 missing
        print("\nSample missing alleles (first 10):")
        for m in all_missing[:10]:
            print(f"  {m['sample'][:60]}  locus={m['locus']}  "
                  f"allele={m['allele']}  donors={m['donors']}")

    # Save QC report
    qc_path = OUT / "genotype_qc.json"
    with open(qc_path, "w") as f:
        json.dump({
            "total_checked": total_checked,
            "missing": len(all_missing),
            "coverage": round(coverage, 6),
            "missing_samples": all_missing[:50]
        }, f, indent=2)
    print(f"QC report -> {qc_path}")
    return coverage


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qc-only", action="store_true")
    args = parser.parse_args()

    out_path = OUT / "donor_genotypes.csv"

    if args.qc_only:
        if not out_path.exists():
            print("ERROR: donor_genotypes.csv not found. Run without --qc-only first.")
            return
        df = pd.read_csv(out_path)
    else:
        print("=== Extracting consensus genotypes ===")
        df = extract_genotypes()

    print("\n=== Running QC against real mixtures ===")
    coverage = qc_against_real(df)

    if coverage < 0.90:
        print("\nWARNING: Coverage < 90% — genotype extraction may need tuning.")
    else:
        print(f"\nOK: Coverage {coverage*100:.1f}% — genotypes are reliable.")


if __name__ == "__main__":
    main()
