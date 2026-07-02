"""
measure_real_seen_vs_novel.py — split the REAL test by whether each sample's contributor
combo was SEEN in the model's actual training combos, then report oracle EM per NOC x
seen/novel. Isolates domain-gap (real vs synthetic) from combinatorial generalization:
if real SEEN-combo oracle is high across NOC, the barrier is novel-combo, not domain.

Training combos are reproduced by replaying make_dev_split's carve (combo_frac=0.15,
noc1_frac=0.06, seed=0 defaults) so "seen" = combos in the post-carve TRAIN portion only.

Usage: python measure_real_seen_vs_novel.py [run_dir] [data_dir]
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

import eval_crossfolder as ecf
from eval_cross_inc3 import build_model_inc3
from train_set_transformer import topk_decode, DEVICE

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/inc4_p1_stack_seed42")
data = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")

cfg = json.load(open(run / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8)
state = torch.load(run / "best_model.pt", map_location=DEVICE, weights_only=True)
model = build_model_inc3(cfg, state)
model.load_state_dict(state)
model.eval()
print(f"run={run}  data={data}  n_tok={n_tok}  device={DEVICE}")

# ---- real test ----
tok = np.load(data / f"tokens{n_tok}_test.npy").astype(np.float32)
mask = np.load(data / "mask_test.npy")
y = np.load(data / "y_test_set.npy").astype(int)
noc = np.clip(np.load(data / "noc_test.npy").astype(int), 1, 5)

# ---- reproduce the train/dev carve to get the model's ACTUAL training combos ----
ytr = np.load(data / "y_train_set.npy").astype(int)
ntr = np.clip(np.load(data / "noc_train.npy").astype(int), 1, 5)
rng = np.random.default_rng(0)
dev = np.zeros(len(ntr), bool)
for k in [2, 3, 4, 5]:
    idx = np.where(ntr == k)[0]
    combos = {}
    for i in idx:
        c = tuple(np.where(ytr[i] == 1)[0].tolist())
        combos.setdefault(c, []).append(i)
    uniq = list(combos)
    rng.shuffle(uniq)
    ndev = max(1, int(round(len(uniq) * 0.15)))
    for c in uniq[:ndev]:
        dev[combos[c]] = True
idx1 = np.where(ntr == 1)[0]
dev[rng.choice(idx1, size=int(round(len(idx1) * 0.06)), replace=False)] = True
trmask = ~dev
train_combos = set(frozenset(np.where(ytr[i] == 1)[0].tolist()) for i in np.where(trmask)[0])
print(f"train combos (post-carve): {len(train_combos)}  (carved {dev.sum()} to dev)")

# ---- forward + oracle decode (top-k at TRUE noc) ----
P, C, R = ecf.forward_all(model, tok, mask)
pred = topk_decode(P, noc)
em = (pred == y).all(1)
seen = np.array([frozenset(np.where(r == 1)[0].tolist()) in train_combos for r in y])

print("\nREAL TEST — ORACLE EM, split by combo-seen-in-training")
print(f"{'NOC':>4} {'n_seen':>7} {'EM_seen':>8} | {'n_novel':>8} {'EM_novel':>9}")
def m(x): return float(x.mean()) if len(x) else float('nan')
for k in [1, 2, 3, 4, 5]:
    mk = noc == k
    s, nv = mk & seen, mk & (~seen)
    print(f"{k:>4} {s.sum():>7} {m(em[s]):>8.3f} | {nv.sum():>8} {m(em[nv]):>9.3f}")
print(f"{'all':>4} {seen.sum():>7} {m(em[seen]):>8.3f} | {(~seen).sum():>8} {m(em[~seen]):>9.3f}")
print(f"\noverall oracle EM = {em.mean():.4f}  (seen frac = {seen.mean():.3f})")
