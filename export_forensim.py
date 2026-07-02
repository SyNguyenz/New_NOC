"""Export allele-frequency table + per-sample observed alleles for forensim's
likestim() (validated MLE NOC, Haned 2011). Frequencies estimated from real
known-donor single-source (NOC=1) profiles. Autosomal STRs only (drop AMEL/Y)."""
import json, csv
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent
D = ROOT / "data_insilico_w"
OUT = ROOT / "forensim_io"; OUT.mkdir(exist_ok=True)
META = json.load(open(ROOT / "data/meta_set.json"))
FC = META["flat_cols"]
DROP = {"AMEL", "DYS391", "Yindel"}                       # sex / Y markers, not autosomal NOC
col_loc, col_all = [], []
for c in FC:
    loc, al = c.rsplit("_", 1); col_loc.append(loc); col_all.append(al)
col_loc = np.array(col_loc); col_all = np.array(col_all)
keep = np.array([l not in DROP for l in col_loc])
loci = [l for l in META["loci"] if l not in DROP]

import sys
AT = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0   # analytical threshold (RFU)
print(f"analytical threshold AT = {AT} RFU")

def present(Xflat):                                       # allele present if RFU > AT
    return np.expm1(Xflat) > AT                            # Xflat = log1p(RFU)

# --- allele frequencies from real NOC=1 known-donor profiles ---
Xtr = np.load(D / "Xflat_train.npy"); ntr = np.load(D / "noc_train.npy").astype(int)
ss = present(Xtr[ntr == 1])                               # (n_ss, 590) bool
cnt = ss.sum(0).astype(float)                             # presence count per bin
with open(OUT / "freq.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["locus", "allele", "freq"])
    for loc in loci:
        m = col_loc == loc; c = cnt[m]; al = col_all[m]
        s = c.sum()
        if s == 0:
            continue
        for a, v in zip(al, c / s):                       # normalize per locus -> sum 1
            if v > 0:
                w.writerow([loc, a, f"{v:.6g}"])

# --- per-sample observed alleles (test) ---
def dump_alleles(split):
    X = np.load(D / f"Xflat_{split}.npy"); noc = np.load(D / f"noc_{split}.npy").astype(int)
    rfu = np.expm1(X); pres = present(X)
    with open(OUT / f"mix_{split}.csv", "w", newline="") as f, \
         open(OUT / f"mixh_{split}.csv", "w", newline="") as fh:
        w = csv.writer(f); w.writerow(["sample", "locus", "allele"])
        wh = csv.writer(fh); wh.writerow(["sample", "locus", "allele", "height"])
        for i in range(len(X)):
            for j in np.where(pres[i] & keep)[0]:
                w.writerow([i, col_loc[j], col_all[j]])
                wh.writerow([i, col_loc[j], col_all[j], int(round(rfu[i, j]))])
    np.save(OUT / f"noc_{split}.npy", noc)
    with open(OUT / f"noc_{split}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["noc"])
        for v in noc:
            w.writerow([int(v)])
    return len(X), noc

n, noc = dump_alleles("test")
import collections
print(f"freq.csv + mix_test.csv written ({n} samples, {len(loci)} autosomal loci)")
print("test NOC dist:", dict(sorted(collections.Counter(np.clip(noc,1,5)).items())))
