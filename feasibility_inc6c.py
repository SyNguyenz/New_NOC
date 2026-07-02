"""
feasibility_inc6c.py — the CEILING gate + DeepSIC feasibility (no-train, local checkpoints).

(A) UNION-ORACLE: across the independently-trained per-donor arms, is each dev-N5 (and N4) true set in
    the top-k of ANY arm? union >> best-single => different models get different combos => a single
    better model has headroom (MODEL-limited) => architectures like DeepSIC worth it. union ~= best
    => every model fails the SAME combos => INFO-limited ceiling => >0.9 likely unreachable here.

(B) ORACLE-PEEL CEILING (= DeepSIC headroom): for dev-N5, remove the OTHER 4 true donors' NNLS-fitted
    reference contributions, re-rank the residual by reference cosine; if neural-BURIED donors recover
    top-5, perfect iterative cancellation (what a trained DeepSIC approximates) breaks the wall.

Usage: python feasibility_inc6c.py
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import nnls

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
ROOT = Path(__file__).resolve().parent
DATA = Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pgnoc as PG
from models.set_transformer import SetTransformerMixture

ARMS = ["inc6_maskp_seed42", "inc6_andmask_seed42", "inc6_sam_seed42", "inc6_maskpre_seed42",
        "inc6_meta_seed42", "inc6_fomaml_seed42", "inc5_res_rand1_seed42", "inc6_vib_seed42"]


def dev_mask_seed0(y, noc, cf=0.15, nf=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc); m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx:
            combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * cf)))]:
            m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * nf)), replace=False)] = True
    return m


# dev split (reconstruct seed0 from train; carries Xflat for the peel)
y = np.load(DATA / "y_train_set.npy").astype(int); noc = np.clip(np.load(DATA / "noc_train.npy").astype(int), 1, 5)
tok8 = np.load(DATA / "tokens8_train.npy").astype(np.float32); msk = np.load(DATA / "mask_train.npy").astype(bool)
Xf = np.load(DATA / "Xflat_train.npy").astype(np.float64)
dm = dev_mask_seed0(y, noc)
di = np.where(dm)[0]
print(f"dev reconstructed: {len(di)} samples  (N4={ (noc[di]==4).sum() }, N5={ (noc[di]==5).sum() })\n")


def build(cfg):
    return SetTransformerMixture(
        n_loci=24, d_locus=16, d_model=128, n_heads=4, n_isab=2, m_inducing=32, n_classes=45, n_noc=6,
        dropout=cfg.get("dropout", 0.1), cls_decoder=cfg.get("cls_decoder", "per_donor"),
        decoder_source=cfg.get("decoder_source", "encoded"), n_token_feats=cfg.get("n_token_feats", 8),
        encoder=cfg.get("encoder", "isab++"), dec_layers=cfg.get("dec_layers", 2),
        num_embed=cfg.get("num_embed", "periodic"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 0.3), aux_heads=cfg.get("aux_heads", False),
        sparse_attn=cfg.get("sparse_attn", False), vib=cfg.get("vib", False)).to(DEVICE)


@torch.no_grad()
def arm_topk_correct(run, k_arr, idx):
    cfgp = ROOT / "results" / run / "metrics.json"
    cfg = json.load(open(cfgp))["config"]
    m = build(cfg)
    sd = torch.load(ROOT / "results" / run / "best_model.pt", map_location=DEVICE, weights_only=False)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    m.load_state_dict(sd, strict=False); m.eval()
    correct = np.zeros(len(idx), bool)
    for b in range(0, len(idx), 256):
        bi = idx[b:b+256]
        P = torch.sigmoid(m(torch.from_numpy(tok8[bi]).to(DEVICE),
                            torch.from_numpy(msk[bi]).to(DEVICE))["logits_cls"]).cpu().numpy()
        for j, gi in enumerate(bi):
            k = k_arr[gi]; top = np.argsort(P[j])[::-1][:k]
            pr = np.zeros(45, int); pr[top] = 1
            correct[b + j] = (pr == y[gi]).all()
    del m; torch.cuda.empty_cache()
    return correct


# ── (A) UNION-ORACLE ─────────────────────────────────────────────────────────────────────
print("== (A) UNION-ORACLE across arms (model-limited vs info-limited) ==")
for K in (4, 5):
    idx = di[noc[di] == K]
    per_arm = {}
    for run in ARMS:
        try:
            per_arm[run] = arm_topk_correct(run, noc, idx)
        except Exception as e:
            print(f"  (skip {run}: {e})")
    M = np.stack(list(per_arm.values()))            # (n_arms, n_samples) bool
    single = {r: c.mean() for r, c in per_arm.items()}
    best_single = max(single.values())
    union = M.any(0).mean()
    nofix = (~M.any(0)).mean()
    print(f"\n  NOC{K} (n={len(idx)}):")
    for r, v in sorted(single.items(), key=lambda x: -x[1]):
        print(f"    {r:<26} oracle={v:.3f}")
    print(f"    --> BEST-SINGLE={best_single:.3f}   UNION(any arm)={union:.3f}   none-get-it={nofix:.3f}")
    print(f"    headroom (union - best) = {union - best_single:+.3f}")


# ── (B) ORACLE-PEEL CEILING (DeepSIC headroom) on dev-N5 ─────────────────────────────────
print("\n== (B) ORACLE-PEEL CEILING on dev-N5 (DeepSIC: perfect cancellation -> recover buried donor) ==")
G = PG.build_refs(); Gn = G / (np.linalg.norm(G, axis=1, keepdims=True) + 1e-9)
neural = arm_topk_correct  # reuse not needed; get neural ranks from best arm (maskp)
cfg = json.load(open(ROOT / "results" / "inc6_maskp_seed42" / "metrics.json"))["config"]
mm = build(cfg); sd = torch.load(ROOT / "results" / "inc6_maskp_seed42" / "best_model.pt", map_location=DEVICE, weights_only=False)
mm.load_state_dict(sd, strict=False); mm.eval()
idx5 = di[noc[di] == 5]
buckets = {"neural<=5": [], "neural 6-15": [], "neural>=16": []}
with torch.no_grad():
    for b in range(0, len(idx5), 256):
        bi = idx5[b:b+256]
        Pb = torch.sigmoid(mm(torch.from_numpy(tok8[bi]).to(DEVICE),
                              torch.from_numpy(msk[bi]).to(DEVICE))["logits_cls"]).cpu().numpy()
        for j, gi in enumerate(bi):
            S = list(np.where(y[gi] == 1)[0]); hrel = Xf[gi] / (Xf[gi].sum() + 1e-12)
            order = list(np.argsort(Pb[j])[::-1])
            for d in S:
                others = [e for e in S if e != d]
                A = G[others].T; phi, _ = nnls(A, hrel); resid = np.clip(hrel - A @ phi, 0, None)
                pr = 45 if resid.sum() < 1e-9 else list(np.argsort(Gn @ (resid / (np.linalg.norm(resid) + 1e-9)))[::-1]).index(d) + 1
                nr = order.index(d) + 1
                buckets["neural<=5" if nr <= 5 else ("neural 6-15" if nr <= 15 else "neural>=16")].append(pr)
del mm; torch.cuda.empty_cache()
print(f"  {'neural-rank bucket':<16}{'n':>6}{'peel-top1%':>12}{'peel-top5%':>12}")
for k, v in buckets.items():
    v = np.array(v)
    if len(v):
        print(f"  {k:<16}{len(v):>6}{(v == 1).mean():>12.2f}{(v <= 5).mean():>12.2f}")
print("\n  buried (neural>=16) peel-top5 HIGH => perfect cancellation recovers them => DeepSIC has headroom.")
