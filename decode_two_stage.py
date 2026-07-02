"""
decode_two_stage.py — POST-HOC decoder upgrade on an existing checkpoint (no retraining).

Why: the inc2 runs decoded with joint-card / post-hoc only (metrics.json: two_stage=None).
The decoded↔oracle gap is ~100% COUNT error (C3); two-stage+pgNOC is the project's best
count decoder (C4: post-hoc 0.940 -> two-stage 0.950 -> +pgNOC 0.954). 2b lifted the oracle
the most, so a stronger decoder should convert more of that ranking into EM — especially NOC4/5.

This loads results/<run>/best_model.pt, forwards train/val/test for per-donor probs, then runs:
  oracle | joint-card | post-hoc | two-stage(prob+MAC) | two-stage+pgNOC
all on the SAME probabilities. Reuses the decode helpers from train_set_transformer.py.

Run: STR_DATA_DIR=data_insilico_w python decode_two_stage.py inc2_2b_privsup
"""
import os, sys, json
os.environ.setdefault("STR_DATA_DIR", "data_insilico_w")

from pathlib import Path
import numpy as np
import torch

import train_set_transformer as T
from models.set_transformer import SetTransformerMixture

RUN     = sys.argv[1] if len(sys.argv) > 1 else "inc2_2b_privsup"
N_FIT_PG = int(sys.argv[2]) if len(sys.argv) > 2 else 12000   # stratified train subsample for pgNOC NNLS
DEVICE  = T.DEVICE
rdir    = T.ROOT / "results" / RUN
cfg     = json.load(open(rdir / "metrics.json"))["config"]
n_tok   = cfg.get("n_token_feats", 3)
tok_pfx = f"tokens{n_tok}" if n_tok > 3 else "tokens"
print(f"run={RUN}  data={T.DATA_DIR}  n_token_feats={n_tok}  aux_heads={cfg.get('aux_heads', False)}")

# ── build model from cfg + load checkpoint (feat_mean/std are buffers in the ckpt) ──
model = SetTransformerMixture(
    n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
    n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
    n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
    cls_decoder=cfg.get("cls_decoder", "pooled"), decoder_source=cfg.get("decoder_source", "encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder", "isab"), dec_layers=cfg.get("dec_layers", 2),
    num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
    periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
).to(DEVICE)
model.load_state_dict(torch.load(rdir / "best_model.pt", map_location=DEVICE, weights_only=True))
model.eval()


def forward(split, want_card=False):
    tok = np.load(T.DATA_DIR / f"{tok_pfx}_{split}.npy").astype(np.float32)
    mk  = np.load(T.DATA_DIR / f"mask_{split}.npy")
    P, C = [], []
    with torch.no_grad():
        for i in range(0, len(tok), 256):
            t = torch.from_numpy(tok[i:i + 256]).to(DEVICE)
            m = torch.from_numpy(mk[i:i + 256]).to(DEVICE)
            out = model(t, m)
            P.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
            if want_card:
                C.append(out["logits_card"].cpu().numpy())
    P = np.concatenate(P)
    return (P, np.concatenate(C), tok, mk) if want_card else (P, tok, mk)


print("forwarding train/val/test ...")
P_tr, tok_tr, mk_tr           = forward("train")
P_va, tok_va, mk_va           = forward("val")
P_te, card_te, tok_te, mk_te  = forward("test", want_card=True)

noc_tr = np.load(T.DATA_DIR / "noc_train.npy"); noc_va = np.load(T.DATA_DIR / "noc_val.npy")
noc_te = np.load(T.DATA_DIR / "noc_test.npy")
y_tr   = np.load(T.DATA_DIR / "y_train_set.npy"); y_va = np.load(T.DATA_DIR / "y_val_set.npy")
y_te   = np.load(rdir / "y_test_true.npy")           # aligned multi-hot truth saved at train time
Xf_tr  = np.load(T.DATA_DIR / "Xflat_train.npy"); Xf_va = np.load(T.DATA_DIR / "Xflat_val.npy")
Xf_te  = np.load(T.DATA_DIR / "Xflat_test.npy")

# ── existing decoders (free) ──
k_card = card_te.argmax(1) + 1
k_post = T.posthoc_cardinality(P_va, y_va, P_te)
rows = {
    "oracle":     T.per_noc_em(y_te, T.topk_decode(P_te, noc_te), noc_te),
    "joint-card": T.per_noc_em(y_te, T.topk_decode(P_te, k_card), noc_te),
    "post-hoc":   T.per_noc_em(y_te, T.topk_decode(P_te, k_post), noc_te),
}

# ── two-stage (prob-profile + MAC), fit on full train ──
print("two-stage (prob+MAC) ...")
fT = T.card_features(P_tr, tok_tr, mk_tr)
fV = T.card_features(P_va, tok_va, mk_va)
fE = T.card_features(P_te, tok_te, mk_te)
k_two = T.two_stage_cardinality(fT, noc_tr, fV, noc_va, fE)
rows["two-stage"] = T.per_noc_em(y_te, T.topk_decode(P_te, k_two), noc_te)

# ── two-stage + pgNOC (global-ref deconvolution cost curve), train subsampled for NNLS budget ──
print(f"pgNOC: building refs + cost features (train subsample={N_FIT_PG}) ...")
rng = np.random.default_rng(42)
ncl = np.clip(noc_tr, 1, 5)
sub = np.concatenate([
    rng.choice(np.where(ncl == j)[0],
               size=min((ncl == j).sum(), N_FIT_PG // 5), replace=False)
    for j in range(1, 6)
])
G = T.build_pgnoc_refs(Xf_tr, y_tr, noc_tr)                      # refs from ALL single-source train
pg_tr = T.pgnoc_cost_features(Xf_tr[sub], P_tr[sub], G)
pg_va = T.pgnoc_cost_features(Xf_va,      P_va,      G)
pg_te = T.pgnoc_cost_features(Xf_te,      P_te,      G)
fT2 = np.hstack([T.card_features(P_tr[sub], tok_tr[sub], mk_tr[sub]), pg_tr])
fV2 = np.hstack([fV, pg_va])
fE2 = np.hstack([fE, pg_te])
k_pg = T.two_stage_cardinality(fT2, noc_tr[sub], fV2, noc_va, fE2,
                               stage2_reg=dict(max_depth=4, min_child_weight=10))
rows["two-stage+pgNOC"] = T.per_noc_em(y_te, T.topk_decode(P_te, k_pg), noc_te)

# ── report ──
print(f"\n{'='*64}\n  {RUN} — decoder comparison (post-hoc on the SAME checkpoint)\n{'='*64}")
print(f"  {'decode':<18}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
for nm in ["oracle", "joint-card", "post-hoc", "two-stage", "two-stage+pgNOC"]:
    print(f"  {nm:<18}" + "".join(f"{x:>7.3f}" for x in rows[nm]))

best = max(["joint-card", "post-hoc", "two-stage", "two-stage+pgNOC"], key=lambda n: rows[n][0])
print(f"\n  best deployable = {best}  (EM {rows[best][0]:.4f})  vs oracle {rows['oracle'][0]:.4f}")
out = {"run": RUN, "n_fit_pg": int(len(sub)),
       "decode": {nm: [round(float(x), 4) for x in r] for nm, r in rows.items()},
       "k_noc_acc": {nm: round(float((np.clip(k, 1, 5) == noc_te).mean()), 4)
                     for nm, k in [("joint-card", k_card), ("post-hoc", k_post),
                                   ("two-stage", k_two), ("two-stage+pgNOC", k_pg)]}}
json.dump(out, open(rdir / "decode_two_stage.json", "w"), indent=2)
print(f"  saved -> {rdir / 'decode_two_stage.json'}")
