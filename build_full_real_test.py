"""
build_full_real_test.py — REPLACE the data_insilico_w 'test' split with the FULL real test
covering ALL raw GF29 known-donor combos (5/6/4/5 per NOC), with NO leak:
  - multi-donor (NOC>=2): ALL real closed samples from data/ {train,val,test} (1333; 0 leak)
  - NOC1: only HELD-OUT real (data/ val+test NOC1 = 2249; data/train NOC1 is IN synth-train -> excluded)
Total 3582. Pools every per-sample array (tokens/mask/Xflat/size/attr/condition/.../y/noc),
re-enriches -> tokens8/9 (+ tokens11 via size), backs up the old 1-combo test first.

Usage: python build_full_real_test.py
"""
import json, shutil, os
from pathlib import Path
import numpy as np

from features.enrich import enrich_tokens, add_size_fields

REAL = Path("data"); OUT = Path("data_insilico_w")
BK = OUT / "_backup_test_1combo"
SPLITS = ["train", "val", "test"]
# per-sample .npy arrays (suffix _{sp}); 'y' handled separately (y_{sp}_set)
NPY = ["tokens", "mask", "Xflat", "size", "attr", "condition", "condition_level",
       "meta_qindex", "meta_template", "phi", "noc"]
JSONL = ["condition", "meta_sample_names"]   # per-sample json lists: {name}_{sp}.json

# ---- keep masks per split (no-leak) ----
keep = {}
for sp in SPLITS:
    y = np.load(REAL / f"y_{sp}_set.npy").astype(int)
    n = np.clip(np.load(REAL / f"noc_{sp}.npy").astype(int), 1, 5)
    closed = (y.sum(1) == n)
    k = closed & (n >= 2)                       # all multi (0 leak)
    if sp in ("val", "test"):
        k = k | (closed & (n == 1))             # held-out NOC1 only
    keep[sp] = k
    print(f"  {sp}: keep {k.sum()}  (NOC1 {int((k&(n==1)).sum())}, multi {int((k&(n>=2)).sum())})")

# ---- which arrays exist for ALL splits (else skip to preserve alignment) ----
def all_exist(suffix_fmt):
    return all((REAL / suffix_fmt(sp)).exists() for sp in SPLITS)

pool_npy = [a for a in NPY if all_exist(lambda sp, a=a: f"{a}_{sp}.npy")]
skipped = [a for a in NPY if a not in pool_npy]
pool_json = [a for a in JSONL if all_exist(lambda sp, a=a: f"{a}_{sp}.json")]
print(f"\npool npy : {pool_npy}\nskip npy : {skipped}\npool json: {pool_json}")

# ---- pool ----
def cat(name):
    return np.concatenate([np.load(REAL / f"{name}_{sp}.npy", allow_pickle=True)[keep[sp]] for sp in SPLITS])

arrs = {a: cat(a) for a in pool_npy}
y = np.concatenate([np.load(REAL / f"y_{sp}_set.npy")[keep[sp]] for sp in SPLITS])
N = len(y)
assert all(len(v) == N for v in arrs.values()), {a: len(v) for a, v in arrs.items()}

jsons = {}
for a in pool_json:
    out = []
    for sp in SPLITS:
        L = json.load(open(REAL / f"{a}_{sp}.json"))
        out += [L[i] for i in np.where(keep[sp])[0]]
    assert len(out) == N
    jsons[a] = out

# ---- enrich ----
tok = arrs["tokens"].astype(np.float32); mask = arrs["mask"].astype(bool)
en9 = enrich_tokens(tok, mask)
en8 = en9[:, :, :8]
en11 = add_size_fields(en9, mask, arrs["size"]) if "size" in arrs else None

# ---- backup old test ----
BK.mkdir(exist_ok=True)
for f in os.listdir(OUT):
    if f == "_backup_test_1combo":
        continue
    if "_test." in f or f in ("y_test_set.npy",) or f.startswith(("tokens8_test", "tokens9_test", "tokens11_test")):
        shutil.copy2(OUT / f, BK / f)
print(f"\nbacked up old test files -> {BK}")

# ---- write new test ----
for a, v in arrs.items():
    np.save(OUT / f"{a}_test.npy", v)
np.save(OUT / "y_test_set.npy", y.astype(np.float32))
np.save(OUT / "tokens9_test.npy", en9)
np.save(OUT / "tokens8_test.npy", en8)
if en11 is not None:
    np.save(OUT / "tokens11_test.npy", en11)
for a, L in jsons.items():
    json.dump(L, open(OUT / f"{a}_test.json", "w"))

# ---- verify + manifest ----
noc = np.clip(arrs["noc"].astype(int), 1, 5)
known = json.load(open(OUT / "meta_set.json"))["known_donors"]
combos = {k: set() for k in range(2, 6)}
for i in np.where(noc >= 2)[0]:
    combos[noc[i]].add(frozenset(np.where(y[i].astype(int) == 1)[0].tolist()))
percomb = {k: len(v) for k, v in combos.items()}
print(f"\nNEW test: N={N}  per-NOC={{1:{int((noc==1).sum())},2:{int((noc==2).sum())},3:{int((noc==3).sum())},4:{int((noc==4).sum())},5:{int((noc==5).sum())}}}")
print(f"combos/NOC = {percomb}  (raw known-only = 5/6/4/5)")
manifest = {
    "description": "FULL real test covering all raw GF29 known-donor combos, no leak. "
                   "Built by build_full_real_test.py from data/ (held-out NOC1 + all multi).",
    "n": int(N), "per_noc": {int(k): int((noc == k).sum()) for k in range(1, 6)},
    "combos_per_noc": {int(k): int(v) for k, v in percomb.items()},
    "skipped_arrays": skipped, "old_test_backup": str(BK),
}
json.dump(manifest, open(OUT / "FULL_TEST_MANIFEST.json", "w"), indent=2)
print(f"wrote manifest -> {OUT/'FULL_TEST_MANIFEST.json'}")
