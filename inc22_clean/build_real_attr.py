"""
build_real_attr.py — Increment 2 / option 2: per-peak allele->donor attribution labels for the
REAL splits (val/test/train), reconstructed from contributor REFERENCE GENOTYPES.

The PROVEDIt real mixtures DO carry provenance after all (design_increment2 §5/§12): each observed
(locus, allele) peak is attributable to the contributor(s) whose reference single-source genotype
contains that allele. So 2b's allele->donor supervision is no longer purely "privileged" — it can be
EVALUATED on real (does the attention's grouping transfer?). Source of references: the consensus
genotype table data/synth/donor_genotypes.csv (synth/extract_genotypes.py; QC coverage ~91%).

Per peak (locus L, allele a) of a mixture with known contributors C:
  candidates = { c in C : a in genotype(c, L) }
    |cand|==1 -> that donor's column
    |cand|>1  -> SHARED allele -> the contributor with the largest NOMINAL phi (phi_{split}.npy);
                 tiebreak = lowest column
    |cand|==0 -> drop-in / stutter / genotype gap -> -1 (ignore_index; matches in-silico pad=-1)
Padding peaks -> -1. Single-source: every true-allele peak -> that donor; artifacts -> -1.

Output (data/):  attr_{split}.npy  (N, 160) int16   (-1 = pad/unattributable)

Usage:  python build_real_attr.py
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
META = json.load(open(DATA / "meta_set.json"))
KNOWN = META["known_donors"]
COL = {d: i for i, d in enumerate(KNOWN)}
LOCUS_TO_IDX = META["locus_to_idx"]
MAX_SEQ = META["max_seq"]


def allele_key(v) -> str:
    """Canonical string for an allele value/encoding (matches token float encoding)."""
    s = str(v).strip()
    if s in ("", "nan", "None"):
        return ""
    if s in ("X", "-2.0", "-2"):
        return "-2.0"
    if s in ("Y", "-1.0", "-1"):
        return "-1.0"
    try:
        return f"{round(float(s), 1):.1f}"
    except ValueError:
        return ""


def load_genotypes() -> dict[int, dict[int, set]]:
    """geno[col][locus_idx] = {allele_key, ...} for known donors."""
    df = pd.read_csv(DATA / "synth" / "donor_genotypes.csv")
    geno: dict[int, dict[int, set]] = defaultdict(lambda: defaultdict(set))
    for _, r in df.iterrows():
        d = int(r["donor_id"])
        if d not in COL or r["locus"] not in LOCUS_TO_IDX:
            continue
        li = LOCUS_TO_IDX[r["locus"]]
        for a in (r["allele1"], r["allele2"]):
            k = allele_key(a)
            if k:
                geno[COL[d]][li].add(k)
    return geno


def main():
    geno = load_genotypes()
    print(f"data: {DATA} | genotypes for {len(geno)} donors")
    for split in ["train", "val", "test"]:
        tp = DATA / f"tokens_{split}.npy"
        if not tp.exists():
            continue
        tok = np.load(tp); mask = np.load(DATA / f"mask_{split}.npy").astype(bool)
        y = np.load(DATA / f"y_{split}_set.npy")
        pf = DATA / f"phi_{split}.npy"
        phi = np.load(pf) if pf.exists() else None
        N = len(tok)
        attr = np.full((N, MAX_SEQ), -1, dtype=np.int16)
        n_valid = n_attr = n_shared = 0
        for i in range(N):
            present = np.where(y[i] > 0)[0]
            if len(present) == 0:
                continue
            gset = {c: geno.get(c, {}) for c in present}
            for j in range(MAX_SEQ):
                if not mask[i, j]:
                    continue
                n_valid += 1
                li = int(round(float(tok[i, j, 0])))
                ak = allele_key(tok[i, j, 1])
                cand = [c for c in present if ak in gset[c].get(li, ())]
                if not cand:
                    continue
                if len(cand) == 1:
                    attr[i, j] = cand[0]
                else:
                    n_shared += 1
                    if phi is not None:
                        cand = sorted(cand, key=lambda c: (-float(phi[i, c]), c))
                    attr[i, j] = cand[0]
                n_attr += 1
        np.save(DATA / f"attr_{split}.npy", attr)
        cov = n_attr / max(n_valid, 1)
        print(f"  {split:5s} N={N:5d}  valid_peaks={n_valid:7d}  attributed={n_attr:7d} "
              f"({cov:.3f})  shared-allele(phi-resolved)={n_shared:6d}  -> attr_{split}.npy")
    print("Done. Real allele->donor attribution saved to data/.")


if __name__ == "__main__":
    main()
