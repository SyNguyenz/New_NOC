"""
measure_real_allcombos_multi.py — run the ALL-RAW-COMBO real oracle eval across many arms
(inc3 + inc4, seed 42). Real pool = all closed multi-donor samples from data/ (every raw
known combo: 5/6/4/5 per NOC). Oracle = top-k at TRUE noc. p6 (slot) skipped (incompatible
decode); p5 uses its P5Model wrapper.

Usage: python measure_real_allcombos_multi.py
"""
import json
from pathlib import Path
import numpy as np
import torch

from train_set_transformer import topk_decode, DEVICE
from models.set_transformer import SetTransformerMixture
from features.enrich import enrich_tokens

REAL = Path("data"); SYNTH = Path("data_insilico_w"); RES = Path("results")
ARMS = [
    "inc3_repA_genoq", "inc3_repB_donorcon", "inc3_repC_additive",
    "inc3_nocV1_ordrnc", "inc3_nocV2_ordrnc_pcgrad", "inc3_nocV3_ordreplace",
    "inc4_p1_stack", "inc4_p2_local", "inc4_p3_irm", "inc4_p4_decorr",
    "inc4_p5_noc_intrinsic",
]  # inc4_p6_slot skipped: slot/Hungarian decode has no top-k oracle

def build_any(cfg, state):
    kw = dict(
        n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
        n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
        cls_decoder=cfg.get("cls_decoder", "pooled"), decoder_source=cfg.get("decoder_source", "encoded"),
        n_token_feats=cfg.get("n_token_feats", 8), encoder=cfg.get("encoder", "isab"),
        dec_layers=cfg.get("dec_layers", 2), num_embed=cfg.get("num_embed", "raw"),
        n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
        noc_contrast=cfg.get("noc_contrast", False),
        noc_detach=(cfg.get("noc_contrast_mode", "shared") == "detach"),
        d_proj=cfg.get("d_proj", 64), sparse_attn=cfg.get("sparse_attn", False),
        geno_query=bool(cfg.get("geno_query")), donor_contrast=cfg.get("donor_contrast", False),
        noc_ord_head=cfg.get("noc_ord_head", False), noc_ord_detach=cfg.get("noc_ord_detach", False),
        noc_ord_replace=cfg.get("noc_ord_replace", False),
    )
    if cfg.get("geno_query"):
        kw["donor_geno"] = torch.zeros_like(state["donor_geno"]).float()
        kw["donor_geno_mask"] = torch.zeros_like(state["donor_geno_mask"]).bool()
    return SetTransformerMixture(**kw).to(DEVICE)

def load_model(arm):
    ck = RES / f"{arm}_seed42"
    cfg = json.load(open(ck / "metrics.json"))["config"]
    state = torch.load(ck / "best_model.pt", map_location=DEVICE, weights_only=True)
    if cfg.get("model") == "p5_noc_intrinsic" or arm == "inc4_p5_noc_intrinsic":
        from train_p5_noc_intrinsic import P5Model
        m = P5Model().to(DEVICE)
    else:
        m = build_any(cfg, state)
    m.load_state_dict(state)
    m.eval()
    return m, cfg.get("n_token_feats", 8)

def cls_probs(model, tok, mask, bs=256):
    P = []
    with torch.no_grad():
        for i in range(0, len(tok), bs):
            out = model(torch.from_numpy(tok[i:i+bs]).to(DEVICE),
                        torch.from_numpy(mask[i:i+bs]).to(DEVICE))
            P.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
    return np.concatenate(P)

# ---- build the all-combo real pool ONCE (3-feat tokens; enrich per-arm n_tok) ----
toks, masks, ys, nocs = [], [], [], []
for sp in ["train", "val", "test"]:
    t = np.load(REAL / f"tokens_{sp}.npy").astype(np.float32); m = np.load(REAL / f"mask_{sp}.npy")
    y = np.load(REAL / f"y_{sp}_set.npy").astype(int); n = np.clip(np.load(REAL / f"noc_{sp}.npy").astype(int), 1, 5)
    keep = (n >= 2) & (y.sum(1) == n)
    toks.append(t[keep]); masks.append(m[keep]); ys.append(y[keep]); nocs.append(n[keep])
tok3 = np.concatenate(toks); mask = np.concatenate(masks); y = np.concatenate(ys); noc = np.concatenate(nocs)
en = enrich_tokens(tok3, mask)
print(f"real all-combo pool: n={len(y)}  per-NOC={[int((noc==k).sum()) for k in range(2,6)]}\n")

# old 1-combo test oracle (from metrics.json generalization.test_oracle) for reference
def old_test_oracle(arm):
    g = json.load(open(RES / f"{arm}_seed42" / "metrics.json")).get("generalization", {}).get("test_oracle", {})
    return g

hdr = f"{'arm':<28}{'N2':>7}{'N3':>7}{'N4':>7}{'N5':>7}{'2-5':>7}   (old 1-combo: N2/N3/N4/N5)"
print(hdr); print("-" * len(hdr))
rows = {}
for arm in ARMS:
    try:
        model, n_tok = load_model(arm)
        P = cls_probs(model, en[:, :, :n_tok], mask)
        pred = topk_decode(P, noc); em = (pred == y).all(1)
        per = {k: float(em[noc == k].mean()) for k in [2, 3, 4, 5]}
        allm = float(em[noc >= 2].mean())
        rows[arm] = per | {"all": allm}
        o = old_test_oracle(arm)
        ostr = "/".join(o.get(str(k), "-") if isinstance(o.get(str(k)), str) else f"{o.get(str(k),0):.2f}" for k in [2,3,4,5]) if o else "n/a"
        print(f"{arm:<28}{per[2]:>7.3f}{per[3]:>7.3f}{per[4]:>7.3f}{per[5]:>7.3f}{allm:>7.3f}   ({ostr})")
    except Exception as e:
        print(f"{arm:<28} ERROR: {type(e).__name__}: {e}")
