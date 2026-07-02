"""
check_data_identity.py — SHA256 the in-silico arrays so we can prove whether two dataset
versions (e.g. the OLD vs the rerun's data_insilico_w) are byte-identical.

Why: the same inc2_2b_privsup config gave NOC2 oracle 0.864 vs 0.977 on two runs. Two causes
are possible: (a) the DATA changed between runs ("richer data"), or (b) pure training
stochasticity (now removed by the train-script seed). This isolates (a): make_dev_split (seed 0)
and features/enrich are deterministic, so if the SOURCE arrays below hash-match, every downstream
train/dev/tok8/9/11 array matches too → any remaining result swing is training seed, not data.

Usage:
  python check_data_identity.py <data_dir>                 # print hashes
  python check_data_identity.py <data_dir_A> <data_dir_B>  # compare, report IDENTICAL/DIFFER
"""
from __future__ import annotations
import hashlib, sys
from pathlib import Path
import numpy as np

# canonical source arrays that determine everything downstream (split + enrich are deterministic)
KEYS = [
    "tokens_train", "mask_train", "y_train_set", "noc_train", "Xflat_train",
    "tokens_val", "mask_val", "y_val_set", "noc_val",
    "tokens_test", "mask_test", "y_test_set", "noc_test",
    "tokens_open", "mask_open",
    # Increment-2 privileged labels (if present)
    "attr_train", "phi_train", "size_train",
]


def h(p: Path) -> str:
    # hash the raw array bytes (C-contiguous, dtype+shape included) — robust to load order
    a = np.load(p)
    m = hashlib.sha256()
    m.update(str(a.dtype).encode()); m.update(str(a.shape).encode())
    m.update(np.ascontiguousarray(a).tobytes())
    return m.hexdigest()


def hashes(d: Path) -> dict[str, str]:
    out = {}
    for k in KEYS:
        p = d / f"{k}.npy"
        out[k] = h(p) if p.exists() else None
    return out


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    A = Path(sys.argv[1]); ha = hashes(A)
    if len(sys.argv) == 2:
        print(f"\n== {A} ==")
        for k, v in ha.items():
            print(f"  {k:<16} {v[:16] if v else '(absent)'}")
        return
    B = Path(sys.argv[2]); hb = hashes(B)
    print(f"\n== compare A={A}  vs  B={B} ==")
    n_diff = n_same = n_skip = 0
    for k in KEYS:
        va, vb = ha[k], hb[k]
        if va is None and vb is None:
            continue
        if va is None or vb is None:
            print(f"  {k:<16} ONLY-IN-{'A' if vb is None else 'B'}"); n_skip += 1
        elif va == vb:
            n_same += 1
        else:
            print(f"  {k:<16} DIFFER  A={va[:12]}  B={vb[:12]}"); n_diff += 1
    print(f"\n  {n_same} identical, {n_diff} differ, {n_skip} present-in-one")
    print("  => " + ("DATA IDENTICAL: any result swing is TRAINING SEED, not data."
                     if n_diff == 0 and n_skip == 0 else
                     "DATA DIFFERS: the rerun's data changed — confound to control before comparing arms."))


if __name__ == "__main__":
    main()
