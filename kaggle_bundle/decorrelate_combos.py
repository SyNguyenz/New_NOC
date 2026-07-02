"""
decorrelate_combos.py — P4 DATA lever (combo-generalization, F29).

Rebalances the in-silico TRAIN so donor participation is FLATTER → weakens the "bind-combo" shortcut
(Geirhos 2020 shortcut learning; Keysers 2020 atom-coverage). It does NOT (and cannot) cover the combo
space (C(45,5)≈1.2M); it removes memorizable redundancy + flattens the per-donor marginal so the per-donor
rule becomes the easier fit. Operates on <data_dir>/*_train arrays IN PLACE (must run BEFORE make_dev_split).

This is MARGINAL decorrelation on EXISTING samples. Fuller decorrelation = regenerate via make_insilico with
a combo-balanced / pairwise-decorrelated sampler (documented, not done here).

Method: weight each sample by the geometric-mean INVERSE frequency of its present donors (rare-donor samples
up-weighted), resample with replacement, then CAP duplicates (avoid memorizing up-weighted samples).

Usage: python decorrelate_combos.py <data_dir> [seed=0] [dup_cap=3]
"""
import sys, collections
from pathlib import Path
import numpy as np

D = Path(sys.argv[1])
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
CAP = int(sys.argv[3]) if len(sys.argv) > 3 else 3
rng = np.random.default_rng(SEED)

y = np.load(D / "y_train_set.npy")
N = len(y)
freq = y.sum(0).clip(min=1)                       # per-donor frequency
loginv = -np.log(freq / freq.sum())               # rare donor -> high
w = np.exp((y * loginv).sum(1) / y.sum(1).clip(min=1))   # geo-mean inverse-freq of present donors
p = w / w.sum()

draw = rng.choice(N, size=N, replace=True, p=p)
cnt = collections.Counter(); keep = []
for i in draw:
    if cnt[i] < CAP:
        keep.append(int(i)); cnt[i] += 1
keep = np.array(sorted(keep))

def cv(arr):
    f = arr.sum(0); return float(f.std() / f.mean())

print(f"decorrelate: train {N} -> {len(keep)}  | donor-freq CV {cv(y):.3f} -> {cv(y[keep]):.3f} (lower=flatter)")

# rewrite ALL aligned *_train arrays by the same index selection
names = {"tokens": "", "mask": "", "Xflat": "", "y": "_set", "noc": ""}
for extra in ["attr", "phi", "size"]:
    if (D / f"{extra}_train.npy").exists():
        names[extra] = ""
for n, suf in names.items():
    f = D / f"{n}_train{suf}.npy"
    if f.exists():
        np.save(f, np.load(f)[keep])
print(f"wrote decorrelated *_train in {D}")
