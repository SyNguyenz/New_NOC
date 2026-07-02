"""
make_dev_split.py — carve a COMBO-DISJOINT, balanced DEV split out of the in-silico TRAIN
for checkpoint selection. No generation/re-upload: just re-partitions the existing in-silico.

Why: selecting on the REAL val is wrong — val is 82% NOC1 (different distribution from the
in-silico-balanced train; oracle-EM saturates -> undertrained checkpoints). Standard ML: the
selection set should match the TRAIN distribution. We also want it NOVEL-COMBO (held out by
contributor-combo, not just by sample) so it detects combo-overfitting — matching the real
test, which is novel combos. Real val is KEPT (for the two-stage decoder's stage1 prior + a
domain-matched secondary check).

Multi-person (NOC>=2): hold out a fraction of UNIQUE combos per NOC stratum -> their samples
form dev; the rest stay in train (so dev combos are unseen in train).
NOC1: hold out a fraction of SAMPLES (donors stay in train via remaining samples -> references intact).

Mutates the data dir in place: shrinks tokens/mask/Xflat/y/noc _train and writes *_dev.
Usage:  python make_dev_split.py <data_dir> [combo_frac=0.15] [noc1_frac=0.06] [seed=0]
"""
import sys
from pathlib import Path
import numpy as np

D = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data_insilico_w")
COMBO_FRAC = float(sys.argv[2]) if len(sys.argv) > 2 else 0.15
NOC1_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.06
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
rng = np.random.default_rng(SEED)

names = ["tokens", "mask", "y", "noc"]
suff = {"tokens": "", "mask": "", "y": "_set", "noc": ""}
arr = {n: np.load(D / f"{n}_train{suff[n]}.npy") for n in names}
y = arr["y"]; noc = np.clip(arr["noc"].astype(int), 1, 5); N = len(noc)
# Optional labels: Xflat + privileged (attr/phi/size + Inc-LUPI mu/var/dropin/beta). Only carry arrays whose
# length MATCHES the train — a STALE array left in a re-used/in-place dir at a different size is SKIPPED,
# not crashed on (e.g. an old Xflat_train when gen_lupi regenerated the train in-place on Kaggle).
for extra in ["Xflat", "attr", "phi", "size", "mu", "var", "dropin", "beta"]:
    p = D / f"{extra}_train.npy"
    if p.exists():
        a = np.load(p)
        if len(a) == N:
            names.append(extra); suff[extra] = ""; arr[extra] = a
        else:
            print(f"  WARN: skipping stale {extra}_train (len {len(a)} != train {N})")

dev_mask = np.zeros(N, bool)
# multi-person: hold out whole combos per NOC stratum
for k in [2, 3, 4, 5]:
    idx = np.where(noc == k)[0]
    combos = {}
    for i in idx:
        c = tuple(np.where(y[i] == 1)[0].tolist())
        combos.setdefault(c, []).append(i)
    uniq = list(combos)
    rng.shuffle(uniq)
    n_dev = max(1, int(round(len(uniq) * COMBO_FRAC)))
    for c in uniq[:n_dev]:
        dev_mask[combos[c]] = True
    print(f"  NOC{k}: {len(uniq)} unique combos -> {n_dev} held to dev "
          f"({dev_mask[idx].sum()} samples)")
# NOC1: hold out random samples (donors remain in train)
idx1 = np.where(noc == 1)[0]
n1 = int(round(len(idx1) * NOC1_FRAC))
dev_mask[rng.choice(idx1, size=n1, replace=False)] = True
print(f"  NOC1: {len(idx1)} samples -> {n1} held to dev")

tr = ~dev_mask
print(f"\ntrain {N} -> train {tr.sum()} + dev {dev_mask.sum()}")
import collections
print("dev NOC dist:", dict(sorted(collections.Counter(noc[dev_mask]).items())))

for n in names:
    np.save(D / f"{n}_dev{suff[n]}.npy", arr[n][dev_mask])
    np.save(D / f"{n}_train{suff[n]}.npy", arr[n][tr])
print(f"\nwrote *_dev and shrunk *_train in {D}")
