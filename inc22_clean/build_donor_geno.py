"""
build_donor_geno.py — precompute the reference-genotype query tokens for Lever A (--geno_query),
so training (incl. Kaggle, where the raw xlsx is absent) just loads a .npy from the data bundle.

For each of the 45 KNOWN donors, emit a padded token set of their reference alleles:
  donor_geno.npy       (C, G, 11) float32   col0=locus_idx, col1=allele_float, col2..10=0 (neutral)
  donor_geno_mask.npy  (C, G)     bool       True = a real reference allele
Genotype source = RAW data_raw/.../*GF29cycles/*RD14-0003 GF Known Genotypes.xlsx (authoritative).

Usage:  python build_donor_geno.py
"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
META = json.load(open(DATA / "meta_set.json"))
KNOWN = META["known_donors"]; COL = {d: i for i, d in enumerate(KNOWN)}
LOCUS_TO_IDX = META["locus_to_idx"]
NFEAT = 11   # widest token (tok11); the model slices [1:n_token_feats] and neutralises cols>=1 anyway


def allele_to_float(s: str):
    s = str(s).strip()
    if s in ("", "nan", "NaN", "None"):
        return None
    if s == "X": return -2.0
    if s == "Y": return -1.0
    try:
        return round(float(s), 1)
    except ValueError:
        return None


def main():
    f = glob.glob("data_raw/**/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx", recursive=True)[0]
    df = pd.read_excel(f, sheet_name=0)
    loci_cols = [c for c in df.columns if c in LOCUS_TO_IDX]

    per_donor: dict[int, list[tuple[float, float]]] = {}
    for _, r in df.iterrows():
        try:
            d = int(r["Sample ID"])
        except (ValueError, TypeError):
            continue
        if d not in COL:
            continue
        toks = []
        for loc in loci_cols:
            cell = r[loc]
            if pd.isna(cell):
                continue
            for a in str(cell).split(","):
                av = allele_to_float(a)
                if av is not None:
                    toks.append((float(LOCUS_TO_IDX[loc]), av))
        per_donor[d] = toks

    C = len(KNOWN)
    G = max(len(per_donor.get(KNOWN[c], [])) for c in range(C))
    geno = np.zeros((C, G, NFEAT), dtype=np.float32)
    mask = np.zeros((C, G), dtype=bool)
    for c in range(C):
        toks = per_donor.get(KNOWN[c], [])
        for j, (li, av) in enumerate(toks):
            geno[c, j, 0] = li; geno[c, j, 1] = av
            mask[c, j] = True
    np.save(DATA / "donor_geno.npy", geno)
    np.save(DATA / "donor_geno_mask.npy", mask)
    cov = mask.sum(1)
    print(f"donor_geno: C={C} G={G}  alleles/donor min={cov.min()} mean={cov.mean():.1f} max={cov.max()}")
    print(f"wrote {DATA/'donor_geno.npy'} (+ mask)")


if __name__ == "__main__":
    main()
