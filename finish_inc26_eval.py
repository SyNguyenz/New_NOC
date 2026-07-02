"""
finish_inc26_eval.py — run the phi-rerank-onward eval for an ALREADY-TRAINED inc26_hybrid_aslot run
whose training finished (best_model.pt saved) but whose eval crashed at `import phi_rerank` (module
not uploaded).  Reproduces the EXACT eval tail of train_set_transformer.py (lines ~1786-1929) for the
inc26 config:  infer -> phi_rerank (deconv -> tune alpha on val -> rerank) -> count_on_rerank
(posthoc_cardinality_rank) -> decode -> metrics.json + y_test_pred.

inc26 = inc22 architecture (aslot / isab++ nc_mab0 / periodic / aux / set_of_set / feas_filter /
noc_head_v2) trained on REPLICATE-POOLED data (R=3), with phi_rerank + count_on_rerank at decode.
So the eval runs on the rep3-pooled val/test (tokens8_{split}_rep3), exactly as the original would.

Reuses the ORIGINAL code (models/set_transformer.py, phi_rerank.py, train_set_transformer helpers)
so this is byte-faithful to what the inc26 run would have produced — nothing is reimplemented.

Usage:
    STR_DATA_DIR=<rep3 data dir> python finish_inc26_eval.py \
        --run results/inc26_hybrid_aslot_seed42 --rep 3
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
ALLELE_OFF = 30; LUT_W = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="results/inc26_hybrid_aslot_seed42")
ap.add_argument("--rep", type=int, default=3)
ap.add_argument("--data", default=os.environ.get("STR_DATA_DIR", "data_w_inc26"))
args = ap.parse_args()
os.environ["STR_DATA_DIR"] = args.data                  # MUST be set before importing train_set_transformer

# Original code (no reimplementation):
from models.set_transformer import SetTransformerMixture
from models.ordinal import corn_probs
import phi_rerank as pr
from train_set_transformer import (ClosedSetDataset, OpenSetDataset, posthoc_cardinality_rank,
                                    posthoc_cardinality, topk_decode, per_noc_em, full_report)
from torch.utils.data import DataLoader

RUN = Path(args.run); DATA = Path(args.data); REP = args.rep
print(f"run={RUN}  data={DATA}  rep={REP}  device={DEVICE}")

# ── reference genotypes + owner_lut (built exactly as the trainer does) ──
gp = DATA / "donor_geno.npy"
if not gp.exists():
    gp = ROOT / "data" / "donor_geno.npy"
donor_geno = torch.from_numpy(np.load(gp).astype(np.float32))
donor_geno_mask = torch.from_numpy(np.load(gp.parent / "donor_geno_mask.npy"))
owner_lut = torch.zeros(24, LUT_W, 45)
gm = donor_geno_mask.bool()
for c in range(min(45, donor_geno.size(0))):
    for j in range(donor_geno.size(1)):
        if gm[c, j]:
            li = int(donor_geno[c, j, 0]); ab = int(round(float(donor_geno[c, j, 1]) * 10)) + ALLELE_OFF
            if 0 <= li < 24 and 0 <= ab < LUT_W:
                owner_lut[li, ab, c] = 1.0
owner_lut = owner_lut.to(DEVICE)

# ── model with the inc26 (= inc22) architecture; buffers (incl owner_lut/feat_*) restored by the ckpt ──
model = SetTransformerMixture(
    n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45, n_noc=6,
    dropout=0.1, cls_decoder="aslot", n_token_feats=8, encoder="isab++",
    num_embed="periodic", periodic_sigma=0.3, aux_heads=True,
    nc_attn="mab0", feas_filter=True, set_of_set=True, soft_geno_attr=False,
    donor_geno=donor_geno, donor_geno_mask=donor_geno_mask, owner_lut=owner_lut,
    n_slot_iters=3, ot_eps=0.05, ot_iters=5, noc_head_v2=True,
).to(DEVICE)
sd = torch.load(RUN / "best_model.pt", weights_only=True, map_location=DEVICE)
missing, unexpected = model.load_state_dict(sd, strict=False)
print(f"loaded checkpoint (missing={len(missing)} unexpected={len(unexpected)})")
model.eval()

# ── rep3-pooled splits (val/test) + open (single-profile) ──
test_ds = ClosedSetDataset("test", "tokens8", REP)
val_ds  = ClosedSetDataset("val",  "tokens8", REP)
open_ds = OpenSetDataset("tokens8")
test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False)


@torch.no_grad()
def infer(loader):
    P, L, V2 = [], [], []
    for tokens, mask, *_ in loader:
        out = model(tokens.to(DEVICE), mask.to(DEVICE))
        L.append(out["logits_cls"].cpu().numpy())
        P.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
        if "logits_count_v2" in out:
            V2.append(out["logits_count_v2"].cpu().numpy())
    return (np.concatenate(P), np.concatenate(L),
            np.concatenate(V2) if V2 else None)


P_te, L_te, v2_te = infer(test_loader)
P_va, L_va, _     = infer(val_loader)
y_te_true = test_ds.y.numpy(); noc_te = test_ds.noc.numpy()
y_va = val_ds.y.numpy(); noc_va = val_ds.noc.numpy()

# ── phi_rerank: independent EM deconvolution on the rep3-POOLED tokens (the math channel sees the
#    replicate-combined evidence too); alpha tuned on val (C6-clean). ──
dg = donor_geno.cpu().numpy(); dgm = donor_geno_mask.cpu().numpy()
PHv = pr.deconv_phi(val_ds.tokens.numpy(),  val_ds.mask.numpy(),  dg, dgm)
PHt = pr.deconv_phi(test_ds.tokens.numpy(), test_ds.mask.numpy(), dg, dgm)
alpha = pr.tune_alpha(L_va, PHv, y_va, noc_va)
rank_te = pr.rerank_scores(L_te, PHt, alpha)
rank_va = pr.rerank_scores(L_va, PHv, alpha)
print(f"phi_rerank ON: val-tuned alpha={alpha}")

# ── count AFTER rerank (count_on_rerank): RF on prob-profile + reranked-score profile ──
k_post = posthoc_cardinality_rank(P_va, rank_va, y_va, P_te, rank_te)
em_post = per_noc_em(y_te_true, topk_decode(rank_te, k_post), noc_te)
oracle  = per_noc_em(y_te_true, topk_decode(rank_te, noc_te), noc_te)

# diagnostics: joint slot-gate card head + the noc_head_v2 CORN head (reported, NOT the headline)
em_v2 = k_v2 = None
if v2_te is not None:
    k_v2 = corn_probs(torch.from_numpy(v2_te), 5).numpy().argmax(1) + 1
    em_v2 = per_noc_em(y_te_true, topk_decode(rank_te, k_v2), noc_te)

y_te_pred = topk_decode(rank_te, k_post)
te_metrics = full_report(y_te_true, y_te_pred, noc_te, "inc26_hybrid_aslot — TEST (post_hoc on reranked)")
print(f"  {'decode':<12}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
rows = [("oracle", oracle), ("post-hoc", em_post)] + ([("noc_v2", em_v2)] if em_v2 else [])
for nm, r in rows:
    print(f"  {nm:<12}" + "".join(f"{x:>7.3f}" for x in r))
count_acc = float((np.clip(k_post, 1, 5) == np.clip(noc_te, 1, 5)).mean())
print(f"  post-hoc count accuracy: {count_acc:.4f}")

# ── reject AUROC (closed test vs open) ──
from sklearn.metrics import roc_auc_score
scores, labels = [], []
with torch.no_grad():
    for tokens, mask, *_ in test_loader:
        scores.append(torch.sigmoid(model(tokens.to(DEVICE), mask.to(DEVICE))["logit_reject"]).cpu().numpy())
        labels.append(np.zeros(len(tokens)))
    for o_tok, o_mask in DataLoader(open_ds, batch_size=256, shuffle=False):
        scores.append(torch.sigmoid(model(o_tok.to(DEVICE), o_mask.to(DEVICE))["logit_reject"]).cpu().numpy())
        labels.append(np.ones(len(o_tok)))
try:
    auroc = float(roc_auc_score(np.concatenate(labels), np.concatenate(scores).ravel()))
except Exception:
    auroc = None

# ── save ──
np.save(RUN / "y_test_pred.npy", y_te_pred)
np.save(RUN / "y_test_true.npy", y_te_true)


def _pn(lst):
    return {str(j): (None if np.isnan(lst[j]) else round(float(lst[j]), 4)) for j in range(1, 6)}


cfg = {
    "model": "set_transformer", "arm": "inc26_hybrid_aslot", "replicates": REP,
    "cls_decoder": "aslot", "encoder": "isab++", "nc_attn": "mab0", "num_embed": "periodic",
    "periodic_sigma": 0.3, "n_token_feats": 8, "aux_heads": True, "set_of_set": True,
    "feas_filter": True, "soft_attr_label": True, "noc_head_v2": True,
    "phi_rerank": True, "count_on_rerank": True,
    "n_slot_iters": 3, "ot_eps": 0.05, "ot_iters": 5, "mask_peaks": 0.15, "epochs": 150,
}
out_dict = {
    "model": "set_transformer", "config": cfg, "decode": "post_hoc (phi-rerank + count_on_rerank)",
    "phi_rerank_alpha": float(alpha),
    "em_post_hoc": round(float(em_post[0]), 4),
    "em_noc_v2": (round(float(em_v2[0]), 4) if em_v2 is not None else None),
    "oracle_em": round(float(oracle[0]), 4),
    "count_acc": round(count_acc, 4),
    "reject_auroc": auroc,
    "per_noc": te_metrics.get("per_noc", {}),
    "per_noc_oracle": _pn(oracle),
    "per_noc_post_hoc": _pn(em_post),
    "per_noc_noc_v2": (_pn(em_v2) if em_v2 is not None else None),
    "test": {k: v for k, v in te_metrics.items() if k != "per_noc"},
}
with open(RUN / "metrics.json", "w") as f:
    json.dump(out_dict, f, indent=2)
print(f"\nSaved metrics.json + y_test_pred/y_test_true -> {RUN}")
