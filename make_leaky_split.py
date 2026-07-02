"""
make_leaky_split.py — Create a LEAKY (random sample-level) split from the SAME
PROVEDIt data, for the leaky-vs-noleak analysis. No new data: just pools the
existing closed-set arrays (train+val+test) and re-splits 70/15/15 randomly.

Output: data_leaky/  (same filenames as data/, model scripts read via STR_DATA_DIR)
"""
import numpy as np, json, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data"; OUT=ROOT/"data_leaky"
OUT.mkdir(exist_ok=True)
SEED=42

keys=["tokens","mask","Xflat","y","noc"]
def fname(k,split): return f"{k}_{split}.npy" if k!="y" else f"y_{split}_set.npy"

# Pool closed-set (train+val+test = all closed-set samples)
pool={k:[] for k in keys}; names=[]
for split in ("train","val","test"):
    for k in keys:
        pool[k].append(np.load(DATA/fname(k,split)))
    names += json.loads((DATA/f"meta_sample_names_{split}.json").read_text())
pooled={k:np.concatenate(pool[k]) for k in keys}
names=np.array(names)
N=len(pooled["noc"]); print(f"Pooled closed-set: {N}")

# Random leaky split (sample-level, ignores combo grouping)
idx=np.arange(N)
itr,itmp=train_test_split(idx,test_size=0.30,random_state=SEED)
iva,ite=train_test_split(itmp,test_size=0.50,random_state=SEED)
splits={"train":itr,"val":iva,"test":ite}

for split,ix in splits.items():
    for k in keys:
        np.save(OUT/fname(k,split), pooled[k][ix])
    (OUT/f"meta_sample_names_{split}.json").write_text(json.dumps(names[ix].tolist()))
    print(f"  {split}: {len(ix)}")

# open-set: copy as-is (not used in this analysis but scripts may expect it)
for k in keys:
    src=DATA/fname(k,"open")
    if src.exists(): shutil.copy(src, OUT/fname(k,"open"))
shutil.copy(DATA/"meta_set.json", OUT/"meta_set.json")
for split in ("train","val","test","open"):
    src=DATA/f"meta_sample_names_{split}.json"
# copy open names
if (DATA/"meta_sample_names_open.json").exists():
    shutil.copy(DATA/"meta_sample_names_open.json", OUT/"meta_sample_names_open.json")
print(f"Saved leaky split -> {OUT}")
