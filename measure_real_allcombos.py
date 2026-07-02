"""
measure_real_allcombos.py — evaluate the model on a REAL test set covering ALL raw GF29
known-donor combos (not just the 1-combo/NOC in data_insilico_w test). Pools the real
closed-set multi-donor samples from data/ (the real prepare_data_set splits, which together
hold every combo: train 3/4/2/3 + val 1 + test 1 per NOC = 5/6/4/5). Enriches 3->8 fields,
runs oracle decode (top-k at TRUE noc), reports EM per NOC and per combo, flags seen/novel
vs the synthetic training combos the model actually trained on (data_insilico_w).

Usage: python measure_real_allcombos.py [run_dir]
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

import eval_crossfolder as ecf
from eval_cross_inc3 import build_model_inc3
from train_set_transformer import topk_decode, DEVICE
from features.enrich import enrich_tokens

run = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/inc4_p1_stack_seed42")
REAL = Path("data")            # real prepare_data_set splits (all combos)
SYNTH = Path("data_insilico_w")  # what the model trained on

known = json.load(open(SYNTH / "meta_set.json"))["known_donors"]
def i2d(fs):  # class-idx frozenset -> donor IDs
    return tuple(sorted(known[i] for i in fs))

# ---- model ----
cfg = json.load(open(run / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8)
state = torch.load(run / "best_model.pt", map_location=DEVICE, weights_only=True)
model = build_model_inc3(cfg, state); model.load_state_dict(state); model.eval()

# ---- synthetic training combos (what model has SEEN) ----
ytr = np.load(SYNTH / "y_train_set.npy").astype(int)
ntr = np.clip(np.load(SYNTH / "noc_train.npy").astype(int), 1, 5)
seen_combos = {k: set() for k in range(1, 6)}
for i in range(len(ytr)):
    seen_combos[ntr[i]].add(frozenset(np.where(ytr[i] == 1)[0].tolist()))

# ---- pool ALL real closed multi-donor samples from data/ train+val+test ----
toks, masks, ys, nocs = [], [], [], []
for sp in ["train", "val", "test"]:
    t = np.load(REAL / f"tokens_{sp}.npy").astype(np.float32)
    m = np.load(REAL / f"mask_{sp}.npy")
    y = np.load(REAL / f"y_{sp}_set.npy").astype(int)
    n = np.clip(np.load(REAL / f"noc_{sp}.npy").astype(int), 1, 5)
    keep = (n >= 2) & (y.sum(1) == n)          # closed-set, complete, multi-donor
    toks.append(t[keep]); masks.append(m[keep]); ys.append(y[keep]); nocs.append(n[keep])
tok = np.concatenate(toks); mask = np.concatenate(masks)
y = np.concatenate(ys); noc = np.concatenate(nocs)
assert tok.shape[-1] == 3, tok.shape
tok8 = enrich_tokens(tok, mask)[:, :, :n_tok]   # 3 -> enriched, take n_tok slice
print(f"pooled real multi-donor (all combos): n={len(y)}  "
      f"per-NOC={[int((noc==k).sum()) for k in range(2,6)]}")

# ---- forward + oracle ----
P, C, R = ecf.forward_all(model, tok8, mask)
pred = topk_decode(P, noc)
em = (pred == y).all(1)
combo = np.array([frozenset(np.where(r == 1)[0].tolist()) for r in y], dtype=object)
seen = np.array([combo[i] in seen_combos[noc[i]] for i in range(len(y))])

print("\n=== REAL — oracle EM over ALL raw combos, per NOC ===")
print(f"{'NOC':>4} {'n':>5} {'EM_all':>7} | {'n_seen':>6} {'EM_seen':>8} | {'n_novel':>7} {'EM_novel':>9}")
def mn(x): return float(x.mean()) if len(x) else float('nan')
for k in [2, 3, 4, 5]:
    mk = noc == k; s = mk & seen; nv = mk & (~seen)
    print(f"{k:>4} {mk.sum():>5} {mn(em[mk]):>7.3f} | {s.sum():>6} {mn(em[s]):>8.3f} | {nv.sum():>7} {mn(em[nv]):>9.3f}")
allm = noc >= 2
print(f"{'2-5':>4} {allm.sum():>5} {mn(em[allm]):>7.3f} | {seen.sum():>6} {mn(em[seen]):>8.3f} | {(~seen).sum():>7} {mn(em[~seen]):>9.3f}")

print("\n=== per-combo breakdown (donor IDs) ===")
print(f"{'NOC':>4} {'combo (donor IDs)':>22} {'n':>4} {'oracleEM':>9} {'seen?':>6}")
for k in [2, 3, 4, 5]:
    cs = {}
    for i in np.where(noc == k)[0]:
        cs.setdefault(combo[i], []).append(i)
    for c, idxs in sorted(cs.items(), key=lambda kv: -len(kv[1])):
        idxs = np.array(idxs)
        flag = "SEEN" if c in seen_combos[k] else "novel"
        print(f"{k:>4} {str(i2d(c)):>22} {len(idxs):>4} {em[idxs].mean():>9.3f} {flag:>6}")
