"""
probe_score_rebalance.py — EVAL-ONLY no-train test of two user ideas for the low-phi N5 wall:
  (1) "rebalance the scoring for weak minor donors"  -> decode-time per-donor logit re-normalization.
  (2) "predict phi too and use it"                   -> minorw ALREADY has an aux phi head; can predicted
                                                        phi re-rank the missed minor above its decoy?

Decisive feasibility read = RANK of the missed true minor in the prob ordering:
  * near-miss (rank just below k, e.g. 6-8 at NOC5)  => score margin issue -> rebalancing CAN recover (GO).
  * deep-miss (rank 15-40)                            => evidence is crushed, not a threshold -> NO-GO.

Then it actually RUNS decode-time re-rankers and reports DEV N5/N4 oracle vs baseline top-k:
  base | per-donor z-norm | phi-boost | phi-alone | oracle-upper(true-set in top-k?)

Usage: python probe_score_rebalance.py [run] [data]
"""
import sys, json
from pathlib import Path
import numpy as np, torch

RUN  = Path("results") / (sys.argv[1] if len(sys.argv) > 1 else "inc6_minorw_seed42")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8); tp = f"tokens{n_tok}"
tok = np.load(DATA / f"{tp}_train.npy").astype(np.float32)
msk = np.load(DATA / "mask_train.npy"); ymat = np.load(DATA / "y_train_set.npy").astype(np.float32)
noc = np.load(DATA / "noc_train.npy").astype(int); phi = np.load(DATA / "phi_train.npy").astype(np.float32)

def dev_mask_seed0(y, noc, combo_frac=0.15, noc1_frac=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc); m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx: combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * combo_frac)))]: m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * noc1_frac)), replace=False)] = True
    return m
dmask = dev_mask_seed0(ymat, noc)

model = SetTransformerMixture(
    n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
    n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
    cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
    num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
    periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=cfg.get("aux_heads",False),
    d_proj=cfg.get("d_proj",64), sparse_attn=cfg.get("sparse_attn",False),
).to(DEVICE)
sd = torch.load(RUN / "best_model.pt", map_location=DEVICE)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
model.load_state_dict(sd, strict=False); model.eval()

@torch.no_grad()
def fwd(idx, bs=256):
    L, PH = [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i+bs]
        o = model(torch.from_numpy(tok[b]).to(DEVICE), torch.from_numpy(msk[b]).to(DEVICE))
        L.append(o["logits_cls"].cpu().numpy())
        PH.append(o["phi"].cpu().numpy() if "phi" in o else np.zeros((len(b), 45), np.float32))
    return np.concatenate(L), np.concatenate(PH)

tr_idx = np.where(~dmask)[0]; dv_idx = np.where(dmask)[0]
rng = np.random.default_rng(1); tr_s = rng.choice(tr_idx, min(8000, len(tr_idx)), replace=False)
Ltr, _ = fwd(tr_s)
Ldv, PHdv = fwd(dv_idx)
nocdv = np.clip(noc[dv_idx], 1, 5); Ydv = ymat[dv_idx]
mu_d = Ltr.mean(0); sd_d = Ltr.std(0) + 1e-6                       # per-donor logit stats (train)

def oracle(scores, noc_keep):
    out = {}
    for k in noc_keep:
        sel = np.where(nocdv == k)[0]
        em = []
        for j in sel:
            top = np.argsort(scores[j])[::-1][:k]; pred = np.zeros(45); pred[top] = 1
            em.append((pred == Ydv[j]).all())
        out[k] = float(np.mean(em)) if em else float("nan")
    return out

P = 1/(1+np.exp(-Ldv))
scorers = {
    "base (sigmoid logit)":      P,
    "per-donor z-norm logit":    (Ldv - mu_d) / sd_d,
    "phi-boost (P * (1+phi))":   P * (1 + PHdv),
    "P + 0.5*phi":               P + 0.5 * PHdv / (PHdv.max()+1e-6),
    "phi-head alone":            PHdv,
}
print(f"run={RUN.name}\n")
print("== DEV oracle by decode re-ranker (no retrain) ==")
print(f"  {'scorer':26s}  N3     N4     N5")
for name, S in scorers.items():
    o = oracle(S, [3,4,5])
    print(f"  {name:26s}  {o[3]:.3f}  {o[4]:.3f}  {o[5]:.3f}")

# ── rank of MISSED true minors (NOC5), and phi-head discrimination ────────────────────────
print("\n== NOC5: where does the MISSED true minor rank? (base ordering) ==")
ranks_missed, phi_true_missed, phi_decoy = [], [], []
sel5 = np.where(nocdv == 5)[0]
for j in sel5:
    order = np.argsort(P[j])[::-1]; rankpos = {d: r for r, d in enumerate(order)}  # 0-based
    top5 = set(order[:5].tolist())
    true = set(np.where(Ydv[j] == 1)[0].tolist())
    for d in true:
        if d not in top5:                                  # a missed true donor
            ranks_missed.append(rankpos[d] + 1)            # 1-based rank
            phi_true_missed.append(PHdv[j, d])
    decoys = top5 - true
    for d in decoys: phi_decoy.append(PHdv[j, d])
ranks_missed = np.array(ranks_missed)
if len(ranks_missed):
    print(f"  n missed true donors = {len(ranks_missed)}")
    for lo, hi in [(6,6),(7,8),(9,12),(13,20),(21,45)]:
        m = (ranks_missed>=lo)&(ranks_missed<=hi)
        print(f"    rank {lo:>2}-{hi:<2}: {100*m.mean():5.1f}%")
    print(f"  median missed rank = {np.median(ranks_missed):.0f}  (k=5 cut; rank 6 = just-missed)")
    print(f"  near-miss (rank 6-8) share = {100*((ranks_missed>=6)&(ranks_missed<=8)).mean():.1f}%")
print("\n== phi-head discrimination (idea 2): does predicted phi 'see' the missed minor? ==")
print(f"  mean predicted-phi  missed-true-minor = {np.mean(phi_true_missed):.3f}  |  decoy-in-top5 = {np.mean(phi_decoy):.3f}")
print("  (if missed-true >> decoy => phi head carries rescue signal -> GO; if <= => phi can't rescue)")

print("\n== CEILING of a competition-aware re-rank: NOC5 sample fully recoverable within top-M ==")
print("  (frac of NOC5 samples where ALL 5 true donors rank <= M; M=5 is current oracle)")
for M in [5, 6, 7, 8, 10, 12]:
    ok = []
    for j in sel5:
        order = np.argsort(P[j])[::-1][:M]
        ok.append(set(np.where(Ydv[j]==1)[0]).issubset(set(order.tolist())))
    print(f"    top-{M:<2}: {np.mean(ok):.3f}")
print("  gap top5->top8 = headroom a per-sample (matching/peeling) decode could reclaim; flat => deep-miss dominates.")
