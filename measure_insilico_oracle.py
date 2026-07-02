"""
measure_insilico_oracle.py — EVAL-ONLY (no retrain). The COMBO-GENERALIZATION judge.

Reports per-NOC ORACLE EM and deployed-COUNT accuracy on THREE sets for a saved checkpoint:
  IN-SILICO TRAIN-fit (seen)  ·  IN-SILICO DEV (held-out combo-disjoint, zero domain shift)  ·  REAL test.
The decisive read (F29): if TRAIN-fit ~1.0 but DEV low, the model overfits training COMBOS (combinatorial
generalization gap), not capacity, not domain gap. This is the metric every generalization arm must be judged
on — NOT aggregate EM (which is ~83% NOC1 and masks the high-NOC combo gap; the F23 trap).

DEV split: if `<tok_prefix>_dev.npy` exists (the runner carves it via make_dev_split) it is loaded directly;
else it is reconstructed IN MEMORY from the FULL train (seed=0, identical to make_dev_split) — non-destructive.

Writes a "generalization" block into the run's metrics.json (per-NOC oracle/count on dev/train/test) so the
result is captured for download, and prints a readout + no-regression guard (dev N1/2/3 oracle).

Usage:
  python measure_insilico_oracle.py <run_dir> [data_dir]
  env FORCE_SRC=raw  → probe: force decoder_source at inference (train/infer mismatch; diagnostic only)
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

RUN = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/inc2_2b_pe_s3_seed42")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from models.set_transformer import SetTransformerMixture

cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8)
tok_prefix = f"tokens{n_tok}" if n_tok > 3 else "tokens"
print(f"run={RUN}  data={DATA}  device={DEVICE}  n_tok={n_tok}")


# ── load any split (tokens/mask/y/noc) ──────────────────────────────────────────────────
def load_split(split):
    tok = np.load(DATA / f"{tok_prefix}_{split}.npy").astype(np.float32)
    msk = np.load(DATA / f"mask_{split}.npy")
    y   = np.load(DATA / f"y_{split}_set.npy")
    noc = np.load(DATA / f"noc_{split}.npy").astype(int)
    return tok, msk, y, noc

def subset(arrs, m):
    return tuple(a[m] for a in arrs)

# ── DEV: prefer the carved split; else reconstruct seed=0 like make_dev_split ────────────
def dev_mask_seed0(y, noc, combo_frac=0.15, noc1_frac=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc)
    m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx:
            combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * combo_frac)))]:
            m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * noc1_frac)), replace=False)] = True
    return m

if (DATA / f"{tok_prefix}_dev.npy").exists():
    DEV = load_split("dev")
    TRAINFIT_FULL = load_split("train")        # already shrunk by make_dev_split — fine for a fit read
    print(f"DEV = carved split ({len(DEV[0])} samples)")
else:
    full = load_split("train")
    dmask = dev_mask_seed0(full[2], full[3])
    DEV = subset(full, dmask); TRAINFIT_FULL = subset(full, ~dmask)
    print(f"DEV = reconstructed seed=0 ({dmask.sum()} samples of {len(full[0])} train)")

# TRAIN-fit subsample (capacity read)
rng = np.random.default_rng(1)
sidx = rng.choice(len(TRAINFIT_FULL[0]), size=min(4000, len(TRAINFIT_FULL[0])), replace=False)
TRAINFIT = subset(TRAINFIT_FULL, sidx)
TEST = load_split("test")

# ── build model + load checkpoint ───────────────────────────────────────────────────────
dg = dgm = None
if cfg.get("geno_query", False) or cfg.get("ref_match", False):
    gp = DATA / "donor_geno.npy"
    if not gp.exists():
        gp = Path("data") / "donor_geno.npy"
    dg = torch.from_numpy(np.load(gp).astype(np.float32))
    dgm = torch.from_numpy(np.load(gp.parent / "donor_geno_mask.npy"))

model = SetTransformerMixture(
    n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
    n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
    n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
    cls_decoder=cfg.get("cls_decoder", "pooled"), decoder_source=cfg.get("decoder_source", "encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder", "isab"), dec_layers=cfg.get("dec_layers", 2),
    num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
    periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
    noc_contrast=cfg.get("noc_contrast", False),
    noc_detach=(cfg.get("noc_contrast_mode", "shared") == "detach"),
    d_proj=cfg.get("d_proj", 64), sparse_attn=cfg.get("sparse_attn", False),
    geno_query=cfg.get("geno_query", False), donor_geno=dg, donor_geno_mask=dgm,
    donor_contrast=cfg.get("donor_contrast", False),
    noc_ord_head=cfg.get("noc_ord_head", False), noc_ord_detach=cfg.get("noc_ord_detach", False),
    noc_ord_replace=cfg.get("noc_ord_replace", False),
    vib=cfg.get("vib", False),          # Inc6 2f: MUST match training or the per-donor latent path is wrong
    mass_pool=cfg.get("mass_pool", False),   # Inc7: MUST match training or the encoder block is wrong
    attn_sink=int(cfg.get("attn_sink", 0)),  # Inc9 B4: MUST match training (changes the decoder attention)
    donor_recon=cfg.get("donor_recon", False),  # Inc9 B1: harmless at infer (aux head unused for logits)
    ref_match=cfg.get("ref_match", False),       # Inc10: MUST match training (adds the per-donor match logit)
    ref_match_learn=cfg.get("ref_match_learn", False),
    nc_attn=cfg.get("nc_attn", "none"),          # Inc11: MUST match training (changes the encoder MAB attention)
    nc_learnable_bias=cfg.get("nc_learnable_bias", False),
).to(DEVICE)
sd = torch.load(RUN / "best_model.pt", map_location=DEVICE)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
model.load_state_dict(sd, strict=False)
_fs = os.environ.get("FORCE_SRC")
if _fs:
    print(f"*** PROBE: forcing decoder_source -> {_fs} (train/infer mismatch) ***"); model.decoder_source = _fs
model.eval()


@torch.no_grad()
def forward_all(tok, msk):
    P, K = [], []
    for i in range(0, len(tok), 256):
        o = model(torch.from_numpy(tok[i:i+256]).to(DEVICE), torch.from_numpy(msk[i:i+256]).to(DEVICE))
        P.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        K.append(o["logits_card"].argmax(1).cpu().numpy() + 1)   # deployed count k=1..5
    return np.concatenate(P), np.concatenate(K)

def measure(name, arrs):
    tok, msk, y, noc = arrs
    P, kp = forward_all(tok, msk); nocc = np.clip(noc.astype(int), 1, 5)
    orc, cnt = {}, {}
    for k in range(1, 6):
        sel = nocc == k
        if not sel.any():
            continue
        em = np.zeros(sel.sum()); yy = y[sel]; PP = P[sel]
        for j in range(sel.sum()):
            top = np.argsort(PP[j])[::-1][:k]; pred = np.zeros(P.shape[1], int); pred[top] = 1
            em[j] = (pred == yy[j]).all()
        orc[k] = float(em.mean()); cnt[k] = float((kp[sel] == k).mean())
    print(f"\n== {name} (n={len(P)}) ==   NOC: " +
          "  ".join(f"{k}[orc{orc[k]:.2f}/cnt{cnt[k]:.2f}]" for k in sorted(orc)))
    return orc, cnt

o_dev, c_dev     = measure("IN-SILICO DEV (combo-disjoint)", DEV)
o_tr,  c_tr      = measure("IN-SILICO TRAIN-fit", TRAINFIT)
o_te,  c_te      = measure("REAL TEST", TEST)

# ── readout + write into metrics.json ───────────────────────────────────────────────────
def g(d, k): return round(d.get(k, float("nan")), 4)
print("\n================ COMBO-GENERALIZATION READOUT ================")
print(f"N5 oracle:  train {g(o_tr,5)}  -> DEV {g(o_dev,5)}  | real {g(o_te,5)}")
print(f"N4 oracle:  train {g(o_tr,4)}  -> DEV {g(o_dev,4)}  | real {g(o_te,4)}")
print(f"N5 count :  train {g(c_tr,5)}  -> DEV {g(c_dev,5)}  | real {g(c_te,5)}")
print(f"guard dev oracle N1/N2/N3: {g(o_dev,1)}/{g(o_dev,2)}/{g(o_dev,3)}  (must stay high)")
print("train>>DEV => combo-overfit (the wall). DEV up vs base => the lever helped generalization.")

mf = RUN / "metrics.json"
M = json.load(open(mf))
M["generalization"] = {
    "dev_oracle": {k: g(o_dev, k) for k in o_dev}, "dev_count": {k: g(c_dev, k) for k in c_dev},
    "train_oracle": {k: g(o_tr, k) for k in o_tr}, "train_count": {k: g(c_tr, k) for k in c_tr},
    "test_oracle": {k: g(o_te, k) for k in o_te}, "test_count": {k: g(c_te, k) for k in c_te},
}
json.dump(M, open(mf, "w"), indent=2)
print(f"wrote generalization block -> {mf}")
