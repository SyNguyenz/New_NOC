"""
probe_root_levers.py — EVAL-ONLY no-train feasibility screen (Inc6-style GO/NO-GO) for ROOT levers
of the low-phi N5 combo-generalization wall. Adjudicates the user's two objections:
  (1) slot attention already tried+collapsed -> skip; (2) peeling = symptom not root -> TEST it.

Three frozen-checkpoint probes:

 P1/P2  ADDITIVE-COMPOSITION & MEMORIZATION-LOCUS.
   Fit a closed-form ridge per-donor readout on SUMMED frozen token features (additive, permutation-
   invariant — the kernel-theory "conjunction-wise additive" model) at two depths:
     x0 = PRE-ISAB projected tokens (locally identifiable alleles, NOT context-mixed)
     H  = POST-ISAB encoded tokens  (globally context-mixed -> combo entanglement hypothesis)
   Fit on TRAIN-fit combos, eval per-NOC ORACLE on DEV (held-out combos).
     * additive-x0 DEV oracle HIGH (>= model's per-donor decoder)  => additive representation GENERALIZES
       to novel combos -> ISAB+attention decoder is what overfits combos -> REPRESENTATION lever is GO,
       and peeling (decode-only) is insufficient.
     * additive-x0 ~ additive-H ~ low                              => info not additively separable ->
       representation lever NO-GO.

 P3  AUGMENTATION/COVERAGE FEASIBILITY (re-test F29).
   Per DEV donor instance, bin recall by combo-proximity to train (n train combos containing that donor;
   min Jaccard distance of its combo to any train combo). Flat recall => coverage/augmentation NO-GO.

Usage: python probe_root_levers.py <run_dir> [data_dir]
"""
import sys, json
from pathlib import Path
import numpy as np, torch

RUN  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/inc6_minorw_seed42")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8); tok_prefix = f"tokens{n_tok}" if n_tok > 3 else "tokens"
print(f"run={RUN}  data={DATA}  device={DEVICE}  n_tok={n_tok}")

tok = np.load(DATA / f"{tok_prefix}_train.npy").astype(np.float32)
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
tr_idx = np.where(~dmask)[0]; dv_idx = np.where(dmask)[0]

# ── frozen model ─────────────────────────────────────────────────────────────────────────
model = SetTransformerMixture(
    n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
    n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
    n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
    cls_decoder=cfg.get("cls_decoder", "pooled"), decoder_source=cfg.get("decoder_source", "encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder", "isab"), dec_layers=cfg.get("dec_layers", 2),
    num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
    periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
    d_proj=cfg.get("d_proj", 64), sparse_attn=cfg.get("sparse_attn", False),
).to(DEVICE)
sd = torch.load(RUN / "best_model.pt", map_location=DEVICE)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
model.load_state_dict(sd, strict=False); model.eval()

@torch.no_grad()
def feats(idx, bs=256):
    """Return summed-over-valid-token features at x0 (pre-ISAB) and H (post-ISAB), + decoder logits."""
    FX, FH, LOG = [], [], []
    for i in range(0, len(idx), bs):
        b = idx[i:i+bs]
        t = torch.from_numpy(tok[b]).to(DEVICE); m = torch.from_numpy(msk[b]).to(DEVICE)
        x0, H, pad = model._encode_set(t, m)
        v = (~pad).unsqueeze(-1).float()
        FX.append((x0 * v).sum(1).cpu().numpy()); FH.append((H * v).sum(1).cpu().numpy())
        out = model(t, m); LOG.append(out["logits_cls"].cpu().numpy())
    return np.concatenate(FX), np.concatenate(FH), np.concatenate(LOG)

# subsample train-fit for the ridge fit
rng = np.random.default_rng(1)
tr_fit = rng.choice(tr_idx, size=min(15000, len(tr_idx)), replace=False)
print(f"fit on {len(tr_fit)} train-fit, eval {len(dv_idx)} dev")
FXtr, FHtr, _ = feats(tr_fit); FXdv, FHdv, LOGdv = feats(dv_idx)
Ytr = ymat[tr_fit]; Ydv = ymat[dv_idx]; nocdv = np.clip(noc[dv_idx], 1, 5)

def ridge_fit(F, Y, lam=10.0):
    mu = F.mean(0); sd_ = F.std(0) + 1e-6; Fs = (F - mu) / sd_
    Fs = np.concatenate([Fs, np.ones((len(Fs), 1))], 1)
    A = Fs.T @ Fs + lam * np.eye(Fs.shape[1]); B = Fs.T @ Y
    W = np.linalg.solve(A, B)
    return (mu, sd_, W)

def ridge_pred(F, P):
    mu, sd_, W = P; Fs = (F - mu) / sd_
    Fs = np.concatenate([Fs, np.ones((len(Fs), 1))], 1)
    return Fs @ W

def oracle_per_noc(scores, Y, nocc):
    out = {}
    for k in range(1, 6):
        sel = nocc == k
        if not sel.any(): continue
        em = []
        for j in np.where(sel)[0]:
            top = np.argsort(scores[j])[::-1][:k]; pred = np.zeros(scores.shape[1]); pred[top] = 1
            em.append((pred == Y[j]).all())
        out[k] = float(np.mean(em))
    return out

Px0 = ridge_fit(FXtr, Ytr); PH = ridge_fit(FHtr, Ytr)
o_x0 = oracle_per_noc(ridge_pred(FXdv, Px0), Ydv, nocdv)
o_H  = oracle_per_noc(ridge_pred(FHdv, PH),  Ydv, nocdv)
o_mdl = oracle_per_noc(1 / (1 + np.exp(-LOGdv)), Ydv, nocdv)
# train-fit oracle of the additive probes (fit quality / memorization read)
o_x0_tr = oracle_per_noc(ridge_pred(FXtr, Px0), Ytr, np.clip(noc[tr_fit],1,5))
o_H_tr  = oracle_per_noc(ridge_pred(FHtr, PH),  Ytr, np.clip(noc[tr_fit],1,5))

print("\n================ P1/P2  ADDITIVE READOUT — DEV oracle (held-out combos) ================")
print(f"  {'NOC':>3} | additive-x0(preISAB) | additive-H(postISAB) | model per-donor decoder")
for k in range(1, 6):
    print(f"  {k:>3} |   dev {o_x0.get(k,float('nan')):.3f}  (tr {o_x0_tr.get(k,float('nan')):.3f}) "
          f"|   dev {o_H.get(k,float('nan')):.3f}  (tr {o_H_tr.get(k,float('nan')):.3f}) "
          f"|   dev {o_mdl.get(k,float('nan')):.3f}")
print("  READ: additive-x0 DEV >= model  => additive/pre-ISAB representation generalizes (REPR lever GO; peeling insufficient).")
print("        additive-x0 ~ additive-H ~ low => not additively separable (REPR lever NO-GO).")

# ── P3  augmentation/coverage feasibility ─────────────────────────────────────────────────
def combos_of(idx):
    return [frozenset(np.where(ymat[i] == 1)[0].tolist()) for i in idx]
tr_combos = combos_of(tr_idx)
from collections import Counter
donor_train_combocount = Counter()
for c in set(tr_combos):
    for d in c: donor_train_combocount[d] += 1
tr_combo_set = list(set(tr_combos))

# per dev donor instance: recall vs (a) #train-combos containing donor, (b) min Jaccard dist of combo to train
Pdv = 1 / (1 + np.exp(-LOGdv))
recs = []  # (noc, donor_train_combos, minjac, recalled, phi)
for j, gi in enumerate(dv_idx):
    k = int(nocdv[j]); C = frozenset(np.where(ymat[gi] == 1)[0].tolist())
    top = set(np.argsort(Pdv[j])[::-1][:k].tolist())
    # min Jaccard distance of C to any train combo of same size (sample a few hundred for speed)
    same = [c for c in tr_combo_set if len(c) == k]
    if same:
        mj = min(1 - len(C & c) / len(C | c) for c in same[:2000])
    else:
        mj = 1.0
    for d in C:
        recs.append((k, donor_train_combocount[d], mj, int(d in top), float(phi[gi, d])))
A = np.array(recs, float)
print("\n================ P3  AUGMENTATION/COVERAGE FEASIBILITY (NOC5, low-phi<.15) ================")
sub = A[(A[:, 0] == 5) & (A[:, 4] < 0.15)]
print(f"  NOC5 low-phi donor instances: {len(sub)}")
print("  by #train-combos containing the donor (more = better covered):")
for lo, hi in [(0, 50), (50, 150), (150, 400), (400, 1e9)]:
    m = (sub[:, 1] >= lo) & (sub[:, 1] < hi)
    if m.sum(): print(f"    [{lo:>4}-{hi if hi<1e9 else 'inf':>4}) n={int(m.sum()):>5}  recall={sub[m,3].mean():.3f}")
print("  by min Jaccard dist of its combo to nearest train combo (smaller = closer to a seen combo):")
for lo, hi in [(0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]:
    m = (sub[:, 2] >= lo) & (sub[:, 2] < hi)
    if m.sum(): print(f"    [{lo:.1f}-{hi:.1f}) n={int(m.sum()):>5}  recall={sub[m,3].mean():.3f}")
print("  READ: recall rising with coverage/closeness => augmentation GO. Flat => NO-GO (confirms F29 per-donor-systematic).")
