"""
probe_peel_prod.py — Iterative residual-donor peeling on production inc6_maskp.

Key points vs the small-scale probe (ab_residual_donor.py):
  - No training: loads pre-trained checkpoint directly
  - Production size: d_model=128, tokens8 (8-feature enriched tokens)
  - Stopping criterion B uses the REAL trained reject head (logit_reject, AUROC ~0.99)
  - v2 update_feas_mask: after identifying k*, private peaks of k* are removed from
    the mask -> encoder sees a true residual (not zombie 0-height tokens)

"trừ khỏi mix và cả encoder": subtraction reduces height at shared peaks; mask update
removes private peaks -> encoder only attends to remaining-contributor evidence.
Reject head then evaluates the clean residual and fires when all real donors are gone.

Run:
  STR_DATA_DIR=data_insilico_w CHECKPOINT=results/inc6_maskp_seed42/best_model.pt \\
      python probe_peel_prod.py
"""

from __future__ import annotations
import os, sys, json
import numpy as np
import torch
from pathlib import Path

ROOT     = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data_insilico_w")))
CKPT     = Path(os.environ.get("CHECKPOINT", str(ROOT / "results/inc6_maskp_seed42/best_model.pt")))
DEV      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture

# ── Load checkpoint + model ────────────────────────────────────────────────
print(f"Loading checkpoint: {CKPT}", flush=True)
# checkpoint is a plain state_dict; config lives in metrics.json
state_dict = torch.load(str(CKPT), map_location=DEV, weights_only=False)
metrics_path = CKPT.parent / "metrics.json"
with open(metrics_path) as f:
    meta = json.load(f)
cfg = meta["config"]
print(f"  d_model={cfg['d_model']}  sparse={cfg['sparse_attn']}  "
      f"aux={cfg['aux_heads']}  encoder={cfg['encoder']}", flush=True)

model = SetTransformerMixture(
    n_loci         = cfg["n_loci"],
    d_locus        = cfg.get("d_locus", 16),
    d_model        = cfg["d_model"],
    n_heads        = cfg["n_heads"],
    n_isab         = cfg["n_isab"],
    m_inducing     = cfg["m_inducing"],
    n_classes      = cfg["n_classes"],
    n_noc          = cfg["n_noc"],
    dropout        = cfg["dropout"],
    aux_heads      = cfg["aux_heads"],
    cls_decoder    = cfg["cls_decoder"],
    dec_aggr       = "sparsemax",
    dec_layers     = 2,
    sparse_attn    = cfg["sparse_attn"],
    num_embed      = cfg.get("num_embed", "raw"),
    periodic_sigma = cfg.get("periodic_sigma", 1.0),
    encoder        = cfg.get("encoder", "isab"),
    n_token_feats  = cfg.get("n_token_feats", 3),
).to(DEV)
model.load_state_dict(state_dict)
model.eval()
print("Model loaded.", flush=True)

C = cfg["n_classes"]   # 45

# ── Load tokens8 test / val ────────────────────────────────────────────────
print("Loading data ...", flush=True)
Xt   = np.load(DATA_DIR / "tokens8_test.npy").astype(np.float32)
Mt   = np.load(DATA_DIR / "mask_test.npy")
Yt   = np.load(DATA_DIR / "y_test_set.npy").astype(np.float32)
NOCt = np.load(DATA_DIR / "noc_test.npy").astype(np.int64)

Xv   = np.load(DATA_DIR / "tokens8_val.npy").astype(np.float32)
Mv   = np.load(DATA_DIR / "mask_val.npy")
Yv   = np.load(DATA_DIR / "y_val_set.npy").astype(np.float32)
NOCv = np.load(DATA_DIR / "noc_val.npy").astype(np.int64)
PHIv = np.load(DATA_DIR / "phi_val.npy").astype(np.float32)

print(f"Test={len(Xt)}  Val={len(Xv)}  C={C}", flush=True)

# ── Genotype CN LUT (for update_feas_mask and oracle subtraction) ──────────
geno_path = None
for cand in (DATA_DIR / "donor_geno.npy", ROOT / "data" / "donor_geno.npy",
             ROOT / "kaggle_bundle" / "donor_geno.npy"):
    if cand.exists(): geno_path = cand; break
assert geno_path is not None, "donor_geno.npy not found"

g  = np.load(geno_path).astype(np.float32)
gm = np.load(geno_path.parent / "donor_geno_mask.npy")

LUT_W = 1024
CN = np.zeros((24, LUT_W, C), np.float32)
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c, j]:
            l_ = int(g[c, j, 0]); b_ = int(round(g[c, j, 1] * 10))
            if 0 <= l_ < 24 and 0 <= b_ < LUT_W:
                CN[l_, b_, c] += 1.0
print(f"CN LUT {CN.shape} max={int(CN.max())}", flush=True)

# ── Carrier lookup: (locus, allele_bin) -> [donor_idx, ...] ──────────────
carr: dict[tuple, list] = {}
for c_ in range(C):
    for j in range(g.shape[1]):
        if gm[c_, j]:
            l_ = int(g[c_, j, 0]); b_ = int(round(g[c_, j, 1] * 10))
            if 0 <= l_ < 24 and 0 <= b_ < LUT_W:
                carr.setdefault((l_, b_), []).append(c_)

NITER_EM = 5

def compute_priv_score(x_np: np.ndarray, m_np: np.ndarray) -> np.ndarray:
    """Private allele evidence per donor [B, C].
    For each visible peak: if only 1 carrier in the 45-panel, add raw height to that donor.
    Decoys have no private alleles in the mixture → score ≈ 0.
    True donors present private alleles → score > 0 (scales with NOC rank).
    """
    B = len(x_np)
    scores = np.zeros((B, C), np.float32)
    for i in range(B):
        for pk in np.where(m_np[i])[0]:
            l_ = int(round(float(x_np[i, pk, 0])))
            b_ = int(round(float(x_np[i, pk, 1]) * 10))
            if not (0 <= l_ < 24 and 0 <= b_ < LUT_W): continue
            cs = carr.get((l_, b_), [])
            if len(cs) == 1:
                scores[i, cs[0]] += max(0.0, np.expm1(float(x_np[i, pk, 2])))
    return scores

def compute_em_phi(x1: np.ndarray, m1: np.ndarray,
                   excluded_set: set | None = None) -> np.ndarray:
    """EM uniform phi for a SINGLE sample (x1 shape [1,N,F]).
    Excluded donors are suppressed. Returns phi [C]."""
    excl = set(excluded_set or [])
    peaks_li, peaks_bi, peaks_h = [], [], []
    for k in np.where(m1[0])[0]:
        l_ = int(round(float(x1[0, k, 0])))
        b_ = int(round(float(x1[0, k, 1]) * 10))
        if not (0 <= l_ < 24 and 0 <= b_ < LUT_W): continue
        if not carr.get((l_, b_)): continue
        peaks_li.append(l_); peaks_bi.append(b_)
        peaks_h.append(max(0.0, np.expm1(float(x1[0, k, 2]))))

    if not peaks_li:
        return np.zeros(C, np.float32)

    h = np.array(peaks_h, np.float64)
    nP = len(peaks_li)
    S = np.full((nP, C + 1), -1e9)
    for r, (l_, b_) in enumerate(zip(peaks_li, peaks_bi)):
        for c in carr.get((l_, b_), []):
            if c not in excl:
                S[r, c] = 0.0
        S[r, C] = -2.0

    phi = np.ones(C + 1, np.float64) / (C + 1)
    for _ in range(NITER_EM):
        z = S + np.log(phi + 1e-9)
        z -= z.max(1, keepdims=True)
        A = np.exp(z); A /= A.sum(1, keepdims=True)
        w   = (A[:, :C] * h[:, None]).sum(0)
        bg  = (A[:, C] * h).sum()
        tot = w.sum() + bg
        phi = np.concatenate([w, [bg]]) / max(tot, 1e-9)

    result = phi[:C].astype(np.float32)
    for c in excl: result[c] = 0.0
    return result

def compute_em_phi_cn(x1: np.ndarray, m1: np.ndarray,
                      excluded_set: set | None = None) -> np.ndarray:
    """CN-weighted EM phi (EuroForMix-style E-step).
    S[r,c] = log(CN[l,b,c]) instead of binary 0 → homozygous donors (CN=2) get 2× weight.
    This is closer to P(h | φ, CN) ∝ φ_k * CN_k as in EuroForMix Eq. 1.
    """
    excl = set(excluded_set or [])
    peaks_li, peaks_bi, peaks_h = [], [], []
    for k in np.where(m1[0])[0]:
        l_ = int(round(float(x1[0, k, 0])))
        b_ = int(round(float(x1[0, k, 1]) * 10))
        if not (0 <= l_ < 24 and 0 <= b_ < LUT_W): continue
        if not carr.get((l_, b_)): continue
        peaks_li.append(l_); peaks_bi.append(b_)
        peaks_h.append(max(0.0, np.expm1(float(x1[0, k, 2]))))

    if not peaks_li:
        return np.zeros(C, np.float32)

    h = np.array(peaks_h, np.float64)
    nP = len(peaks_li)
    S = np.full((nP, C + 1), -1e9)
    for r, (l_, b_) in enumerate(zip(peaks_li, peaks_bi)):
        for c in carr.get((l_, b_), []):
            if c not in excl:
                cn_val = float(CN[l_, b_, c])
                S[r, c] = np.log(max(cn_val, 1.0))  # log(1)=0 het, log(2)≈0.693 hom
        S[r, C] = -2.0

    phi = np.ones(C + 1, np.float64) / (C + 1)
    for _ in range(NITER_EM):
        z = S + np.log(phi + 1e-9)
        z -= z.max(1, keepdims=True)
        A = np.exp(z); A /= A.sum(1, keepdims=True)
        w   = (A[:, :C] * h[:, None]).sum(0)
        bg  = (A[:, C] * h).sum()
        tot = w.sum() + bg
        phi = np.concatenate([w, [bg]]) / max(tot, 1e-9)

    result = phi[:C].astype(np.float32)
    for c in excl: result[c] = 0.0
    return result


def em_subtract(x_np: np.ndarray, m_np: np.ndarray,
                phi_em: np.ndarray, k: int, excluded_set: set) -> np.ndarray:
    """Subtract donor k: frac = phi_k*CN_k / sum_{remaining incl k}."""
    x2     = x_np.copy()
    h      = np.expm1(x_np[:, :, 2])
    li     = x_np[:, :, 0].astype(int).clip(0, 23)
    bi     = np.round(x_np[:, :, 1] * 10).astype(int).clip(0, LUT_W - 1)
    rem    = _rem_mask_incl_k(excluded_set, k)
    cn_all  = CN[li, bi, :]                       # (1, N, C)
    phi_b   = phi_em[np.newaxis, :]               # (1, C)
    phi_rem = phi_b * rem[np.newaxis, :]
    total   = (phi_rem[:, np.newaxis, :] * cn_all).sum(-1).clip(min=1e-9)
    frac    = (phi_b[:, k:k+1] * cn_all[:, :, k] / total).clip(0.0, 1.0)
    x2[:, :, 2] = np.log1p((h * (1.0 - frac)).clip(min=0.0))
    return x2

# ── Genotype-consistency mask on attr logits ───────────────────────────────
def apply_geno_mask(AL_np, x_np, m_np):
    B = len(x_np)
    for i in range(B):
        for k in np.where(m_np[i])[0]:
            l_ = int(round(float(x_np[i, k, 0])))
            b_ = int(round(float(x_np[i, k, 1]) * 10))
            if not (0 <= l_ < 24 and 0 <= b_ < LUT_W): continue
            cn = CN[l_, b_, :]
            n_car = int((cn > 0).sum())
            if n_car == 0:
                AL_np[i, k, :C] = -1e9
            elif n_car == 1:
                AL_np[i, k, :C] = -1e9
                AL_np[i, k, int(np.where(cn > 0)[0][0])] = 1e9
            else:
                AL_np[i, k, :C][cn == 0] = -1e9

# ── Feasibility pre-filter (F38): remove peaks with no carrier in 45-panel ──
def feas_prefilter(m_np: np.ndarray, x_np: np.ndarray) -> np.ndarray:
    """Drop peaks whose allele is carried by NO donor in the panel.
    F38: 16% of peaks, 0% true contributor loss."""
    m_new = m_np.copy()
    for i in range(len(x_np)):
        for pk_idx in np.where(m_np[i])[0]:
            l_ = int(round(float(x_np[i, pk_idx, 0])))
            b_ = int(round(float(x_np[i, pk_idx, 1]) * 10))
            if not (0 <= l_ < 24 and 0 <= b_ < LUT_W):
                m_new[i, pk_idx] = False; continue
            if CN[l_, b_, :].sum() < 1e-9:
                m_new[i, pk_idx] = False
    return m_new

# ── v2 feas-mask update (remove private peaks of identified donors) ─────────
def update_feas_mask(m_cur, x_np_orig, excluded_list):
    """
    Mask peaks with no remaining carrier after donor exclusion.
    This is the "cả encoder" part: encoder can no longer attend to k*'s private peaks.
    """
    m_new = m_cur.copy()
    for i in range(len(x_np_orig)):
        excl = excluded_list[i]
        if not excl: continue
        for pk_idx in np.where(m_cur[i])[0]:
            l_ = int(round(float(x_np_orig[i, pk_idx, 0])))
            b_ = int(round(float(x_np_orig[i, pk_idx, 1]) * 10))
            if not (0 <= l_ < 24 and 0 <= b_ < LUT_W): continue
            cn_rem = CN[l_, b_, :].copy()
            for d in excl:
                cn_rem[d] = 0.0
            if cn_rem.sum() < 1e-9:
                m_new[i, pk_idx] = False
    return m_new

# ── Inference: cls + attr + reject head ───────────────────────────────────
@torch.no_grad()
def get_probs_attr_rej(x_np, m_np, excluded_list=None):
    """
    Batched forward. Excluded donors are zeroed post-hoc in both cls_p and
    attr_logits — equivalent to treating them as "unknown / removed from class set."
    Combined with feas_mask update (private peaks removed from encoder input),
    this gives full "exclude from mix AND from class context" effect.
    """
    model.eval()
    xb = torch.from_numpy(x_np).to(DEV)
    mb = torch.from_numpy(m_np.astype(np.bool_)).to(DEV)
    out = model(xb, mb)
    cls_p = torch.sigmoid(out["logits_cls"]).cpu().numpy()
    rej_p = torch.sigmoid(out["logit_reject"]).cpu().numpy().ravel()
    la    = out["logits_attr"].cpu().numpy()

    # Zero excluded donors in cls AND attr — removes them from class set
    if excluded_list is not None:
        for i, excl in enumerate(excluded_list):
            for d in excl:
                cls_p[i, d] = 0.0
                la[i, :, d] = -1e9   # collapses to background class in softmax

    apply_geno_mask(la, x_np, m_np)
    attr_p = np.exp(la - la.max(-1, keepdims=True))
    attr_p /= attr_p.sum(-1, keepdims=True).clip(min=1e-9)
    return cls_p, attr_p, rej_p

# ── Subtraction ────────────────────────────────────────────────────────────
def subtract_donor(x_np, m_np, attr_p, k):
    x2   = x_np.copy()
    h    = np.expm1(x_np[:, :, 2])
    frac = attr_p[:, :, k].clip(0.0, 1.0)
    x2[:, :, 2] = np.log1p((h * (1.0 - frac)).clip(min=0.0))
    return x2

def _rem_mask_incl_k(excluded_set: set, k: int) -> np.ndarray:
    """Boolean mask [C]: True for donors still in mixture BEFORE removing k.
    Denominator must include k so frac = phi_k*CN_k / sum_{remaining incl k}."""
    rem = np.ones(C, bool)
    for d in excluded_set:
        if d != k:
            rem[d] = False
    return rem

def oracle_subtract(x_np, m_np, phi_b, k, excluded_set):
    """Oracle subtraction: frac = phi_k*CN_k / sum_{remaining incl k}."""
    x2      = x_np.copy()
    h       = np.expm1(x_np[:, :, 2])
    li      = x_np[:, :, 0].astype(int).clip(0, 23)
    bi      = np.round(x_np[:, :, 1] * 10).astype(int).clip(0, LUT_W - 1)
    rem     = _rem_mask_incl_k(excluded_set, k)
    cn_all  = CN[li, bi, :]
    phi_rem = phi_b * rem[np.newaxis, :]
    total   = (phi_rem[:, np.newaxis, :] * cn_all).sum(-1).clip(min=1e-9)
    frac    = (phi_b[:, k:k+1] * cn_all[:, :, k] / total).clip(0.0, 1.0)
    x2[:, :, 2] = np.log1p((h * (1.0 - frac)).clip(min=0.0))
    return x2

# ── Generic peeling pipeline ───────────────────────────────────────────────
def _zscore(arr, idxs):
    """z-score arr over given indices; return full-length array."""
    sub = arr[idxs]; mu = sub.mean(); sd = sub.std()
    return (arr - mu) / (sd + 1e-9)

def _zscore_arr(arr):
    """z-score full array."""
    mu = arr.mean(); sd = arr.std()
    return (arr - mu) / (sd + 1e-9)

def run_pipeline(x_np, m_np, y_np, phi_np=None, use_oracle=False, use_em=False,
                 phi_alpha: float = 0.0, max_steps=6, update_mask=True):
    """
    Returns per-sample lists:
      per_recalls[i][s] - exact match after s+1 picks
      per_confs[i][s]   - max cls_prob at step s
      per_rej[i][s]     - reject_prob BEFORE step s

    phi_alpha > 0: at each step combine z(cls_p) + phi_alpha * z(log(phi_em))
                   for picking (phi rerank per step); use_em must be True.
    """
    B        = len(x_np)
    x_cur    = x_np.copy()
    m_cur    = feas_prefilter(m_np, x_np)   # drop no-carrier noise peaks (F38)
    excluded = [set() for _ in range(B)]
    per_recalls = [[] for _ in range(B)]
    per_confs   = [[] for _ in range(B)]
    per_rej     = [[] for _ in range(B)]

    for step in range(max_steps):
        cls_p, attr_p, rej_p = get_probs_attr_rej(x_cur, m_cur, excluded)

        # pre-compute EM phi for all samples when needed
        step_phi_em = [None] * B
        if use_em:
            for i in range(B):
                step_phi_em[i] = compute_em_phi(x_cur[i:i+1], m_cur[i:i+1], excluded[i])

        step_ks = []
        for i in range(B):
            per_rej[i].append(float(rej_p[i]))

            avail = [c for c in range(C) if c not in excluded[i]]
            if not avail:
                per_confs[i].append(0.0)
                per_recalls[i].append(per_recalls[i][-1] if per_recalls[i] else 0.0)
                step_ks.append(-1); continue

            if use_em and phi_alpha > 0 and step_phi_em[i] is not None:
                avail_arr = np.array(avail)
                phi_sc  = np.log(step_phi_em[i] + 1e-6)
                score   = _zscore(cls_p[i], avail_arr) + phi_alpha * _zscore(phi_sc, avail_arr)
                k = int(avail_arr[np.argmax(score[avail_arr])])
            else:
                k = int(max(avail, key=lambda c: cls_p[i, c]))

            per_confs[i].append(float(cls_p[i, k]))
            excluded[i].add(k); step_ks.append(k)

            id_set   = set(excluded[i])
            true_set = set(np.where(y_np[i] > 0.5)[0])
            per_recalls[i].append(1.0 if true_set.issubset(id_set) else 0.0)

        if update_mask:
            m_cur = update_feas_mask(m_cur, x_np, excluded)

        x_next = x_cur.copy()
        for i in range(B):
            k = step_ks[i]
            if k < 0: continue
            if use_oracle and phi_np is not None:
                x_next[i:i+1] = oracle_subtract(x_cur[i:i+1], m_cur[i:i+1],
                                                  phi_np[i:i+1], k, excluded[i])
            elif use_em:
                phi_em = step_phi_em[i] if step_phi_em[i] is not None \
                         else compute_em_phi(x_cur[i:i+1], m_cur[i:i+1], excluded[i])
                x_next[i:i+1] = em_subtract(x_cur[i:i+1], m_cur[i:i+1],
                                              phi_em, k, excluded[i])
            else:
                x_next[i:i+1] = subtract_donor(x_cur[i:i+1], m_cur[i:i+1],
                                                 attr_p[i:i+1], k)
        x_cur = x_next

    return per_recalls, per_confs, per_rej


def agg(per_vals, steps):
    return [float(np.mean([per_vals[i][s] for i in range(len(per_vals))]))
            for s in range(steps)]

@torch.no_grad()
def direct_topk_exact(x_np, m_np, y_np, k=5):
    """Exact match rate: fraction of samples where ALL k true donors are in top-k.
    Matches per_noc_oracle in metrics.json."""
    xb = torch.from_numpy(x_np).to(DEV)
    mb = torch.from_numpy(m_np.astype(np.bool_)).to(DEV)
    cls_p = torch.sigmoid(model(xb, mb)["logits_cls"]).cpu().numpy()
    topk  = np.argsort(-cls_p, axis=1)[:, :k]
    hits  = [int(len(set(topk[i]) & set(np.where(y_np[i] > 0.5)[0])) == k)
             for i in range(len(x_np))]
    return float(np.mean(hits))

# ── NOC estimation ─────────────────────────────────────────────────────────
def est_noc_conf(per_confs, threshold):
    """Criterion A: stop when conf < threshold."""
    noc = np.ones(len(per_confs), int)
    for i, cs in enumerate(per_confs):
        cnt = 0
        for c in cs:
            if c >= threshold: cnt += 1
            else: break
        noc[i] = max(1, cnt)
    return noc.clip(1, 5)

def est_noc_priv(priv_mat: np.ndarray, threshold: float) -> np.ndarray:
    """NOC from priv_score: count donors with priv_score > threshold.
    Decoys have priv_score=0 (no private alleles in mixture) → not counted."""
    return (priv_mat > threshold).sum(axis=1).clip(1, 5).astype(int)

def est_noc_rej(per_rej, threshold):
    """
    Criterion B: reject_prob = 'complexity / contributors remaining'.
    HIGH reject -> complex (more contributors present) -> CONTINUE.
    LOW reject  -> simple/empty residual -> all found -> STOP.
    NOC = number of steps where reject_prob > threshold.
    """
    noc = np.ones(len(per_rej), int)
    for i, rs in enumerate(per_rej):
        cnt = 0
        for r in rs:
            if r > threshold: cnt += 1  # more contributors -> continue
            else: break                 # all found -> stop
        noc[i] = max(1, cnt)
    return noc.clip(1, 5)

# ── Run experiments ────────────────────────────────────────────────────────
MAX_STEPS = 6
n5t  = NOCt == 5; n5v = NOCv == 5
Xt5, Mt5, Yt5            = Xt[n5t], Mt[n5t], Yt[n5t]
Xv5, Mv5, Yv5, PHIv5     = Xv[n5v], Mv[n5v], Yv[n5v], PHIv[n5v]

print(f"N5 test={n5t.sum()}  N5 val={n5v.sum()}", flush=True)

bsl_t = direct_topk_exact(Xt5, Mt5, Yt5, k=5)
bsl_v = direct_topk_exact(Xv5, Mv5, Yv5, k=5)

print("Test N5 model peel ...", flush=True)
t_r, t_c, t_rj = run_pipeline(Xt5, Mt5, Yt5, update_mask=True, max_steps=MAX_STEPS)

print("Val N5 model peel ...", flush=True)
v_r, v_c, v_rj = run_pipeline(Xv5, Mv5, Yv5, update_mask=True, max_steps=MAX_STEPS)

print("Val N5 oracle peel ...", flush=True)
vo_r, vo_c, vo_rj = run_pipeline(Xv5, Mv5, Yv5, PHIv5,
                                   use_oracle=True, update_mask=True, max_steps=MAX_STEPS)

print("Test N5 EM-uniform peel ...", flush=True)
te_r, te_c, te_rj = run_pipeline(Xt5, Mt5, Yt5, use_em=True,
                                   update_mask=True, max_steps=MAX_STEPS)

print("Val N5 EM-uniform peel ...", flush=True)
ve_r, ve_c, ve_rj = run_pipeline(Xv5, Mv5, Yv5, use_em=True,
                                   update_mask=True, max_steps=MAX_STEPS)

# ── phi rerank standalone (single-pass, no subtraction) ───────────────────
def phi_rerank_exact(x_np, m_np, y_np, k=5, alpha=0.5, priv_alpha=0.0, phi_fn=None):
    """Single-pass: z(cls_logit) + alpha*z(log(phi)) + priv_alpha*z(priv_score).
    phi_fn: callable(x1, m1) -> phi[C]. Defaults to compute_em_phi (uniform compat).
    """
    if phi_fn is None: phi_fn = compute_em_phi
    xb = torch.from_numpy(x_np).to(DEV)
    mb = torch.from_numpy(m_np.astype(np.bool_)).to(DEV)
    with torch.no_grad():
        cls_l = model(xb, mb)["logits_cls"].cpu().numpy()
    m_feas = feas_prefilter(m_np, x_np)
    phi_em = np.stack([phi_fn(x_np[i:i+1], m_feas[i:i+1]) for i in range(len(x_np))])
    priv   = compute_priv_score(x_np, m_feas) if priv_alpha != 0 else None
    hits = []
    for i in range(len(x_np)):
        true_set = set(np.where(y_np[i] > 0.5)[0])
        score = _zscore_arr(cls_l[i]) + alpha * _zscore_arr(np.log(phi_em[i] + 1e-6))
        if priv_alpha != 0 and priv is not None:
            score = score + priv_alpha * _zscore_arr(priv[i])
        topk  = set(np.argsort(-score)[:k])
        hits.append(1.0 if true_set.issubset(topk) else 0.0)
    return float(np.mean(hits))

# alpha sweep on val, apply best to test
ALPHAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
print("phi-rerank alpha sweep on val N5 ...", flush=True)
pr_val  = {a: phi_rerank_exact(Xv5, Mv5, Yv5, k=5, alpha=a) for a in ALPHAS}
best_pr_alpha = max(ALPHAS, key=lambda a: pr_val[a])
pr_test = {a: phi_rerank_exact(Xt5, Mt5, Yt5, k=5, alpha=a) for a in ALPHAS}

# priv_alpha sweep with phi alpha fixed at best_pr_alpha
PRIV_ALPHAS = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
print(f"priv_alpha sweep (phi_alpha fixed={best_pr_alpha}) on val N5 ...", flush=True)
prv_val  = {p: phi_rerank_exact(Xv5, Mv5, Yv5, k=5, alpha=best_pr_alpha, priv_alpha=p)
            for p in PRIV_ALPHAS}
best_prv_alpha = max(PRIV_ALPHAS, key=lambda p: prv_val[p])
prv_test = {p: phi_rerank_exact(Xt5, Mt5, Yt5, k=5, alpha=best_pr_alpha, priv_alpha=p)
            for p in PRIV_ALPHAS}

# ── CN-weighted EM phi comparison (uniform vs CN-weighted) ────────────────
print("\n--- EM phi variant comparison: uniform vs CN-weighted ---", flush=True)
# decoyAUC: for N5 missed cases, does CN phi rank true donor above decoy better?
from sklearn.metrics import roc_auc_score

def _phi_decoy_auc(x_np, m_np, y_np, phi_fn):
    m_feas = feas_prefilter(m_np, x_np)
    phis = np.stack([phi_fn(x_np[i:i+1], m_feas[i:i+1]) for i in range(len(x_np))])
    scores, labels = [], []
    xb = torch.from_numpy(x_np).to(DEV)
    mb = torch.from_numpy(m_np.astype(np.bool_)).to(DEV)
    with torch.no_grad():
        cls_l = model(xb, mb)["logits_cls"].cpu().numpy()
    for i in range(len(x_np)):
        true_set = set(np.where(y_np[i] > 0.5)[0])
        pred_top5 = set(np.argsort(-cls_l[i])[:5])
        missed = true_set - pred_top5  # true donors the cls misses
        if not missed: continue
        for c in range(len(y_np[i])):
            scores.append(float(phis[i, c]))
            labels.append(1 if c in missed else 0)
    if sum(labels) == 0: return float('nan')
    return roc_auc_score(labels, scores)

print("  Computing decoyAUC (N5 val) ...", flush=True)
auc_unif = _phi_decoy_auc(Xv5, Mv5, Yv5, compute_em_phi)
auc_cn   = _phi_decoy_auc(Xv5, Mv5, Yv5, compute_em_phi_cn)
print(f"  uniform EM phi decoyAUC (val N5): {auc_unif:.4f}")
print(f"  CN-wtd  EM phi decoyAUC (val N5): {auc_cn:.4f}")

# N5 oracle: val alpha sweep for CN phi
print("  CN phi alpha sweep on val N5 ...", flush=True)
cn_val  = {a: phi_rerank_exact(Xv5, Mv5, Yv5, k=5, alpha=a, phi_fn=compute_em_phi_cn) for a in ALPHAS}
best_cn_alpha = max(ALPHAS, key=lambda a: cn_val[a])
cn_test = {a: phi_rerank_exact(Xt5, Mt5, Yt5, k=5, alpha=a, phi_fn=compute_em_phi_cn) for a in ALPHAS}
print(f"  uniform phi best alpha={best_pr_alpha}: val={pr_val[best_pr_alpha]:.4f}  test={pr_test[best_pr_alpha]:.4f}")
print(f"  CN-wtd  phi best alpha={best_cn_alpha}: val={cn_val[best_cn_alpha]:.4f}  test={cn_test[best_cn_alpha]:.4f}")

# ── combined: EM peeling + phi rerank picking (alpha sweep) ───────────────
COMB_ALPHAS = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5]
print("Combined EM-peel + phi-rerank picking alpha sweep on val N5 ...", flush=True)
comb_val5 = {}
for a in COMB_ALPHAS:
    r, c, _ = run_pipeline(Xv5, Mv5, Yv5, use_em=True, phi_alpha=a,
                            update_mask=True, max_steps=MAX_STEPS)
    comb_val5[a] = agg(r, MAX_STEPS)
    print(f"  alpha={a:.2f}  step5={comb_val5[a][4]:.4f}  step6={comb_val5[a][5]:.4f}", flush=True)

best_comb_alpha = max(COMB_ALPHAS, key=lambda a: comb_val5[a][4])
print(f"Combined test N5 (alpha={best_comb_alpha}) ...", flush=True)
cr_t, cc_t, _ = run_pipeline(Xt5, Mt5, Yt5, use_em=True, phi_alpha=best_comb_alpha,
                               update_mask=True, max_steps=MAX_STEPS)

# ── Results ────────────────────────────────────────────────────────────────
W  = 78
S_ = list(range(1, MAX_STEPS + 1))
hdr = "step: " + "  ".join(f"  {s}" for s in S_)

print("\n" + "=" * W)
print(f"  inc6_maskp peeling probe  (reject AUROC={meta.get('reject_auroc',0.9918):.4f})")
print("=" * W)

print(f"\n--- Real test N5 (n={n5t.sum()}) ---")
print(f"  baseline exact_match (per_noc_oracle)  : {bsl_t:.4f}")
print(f"  phi rerank a={best_pr_alpha} (single-pass)        : {pr_test[best_pr_alpha]:.4f}"
      f"  (val-best; val={pr_val[best_pr_alpha]:.4f})")
print("  " + hdr)
print("  model peel    : " + "  ".join(f"{r:.4f}" for r in agg(t_r, MAX_STEPS)))
print("  EM-unif peel  : " + "  ".join(f"{r:.4f}" for r in agg(te_r, MAX_STEPS)))
print(f"  combined(a={best_comb_alpha}) : " + "  ".join(f"{r:.4f}" for r in agg(cr_t, MAX_STEPS)))
print("  max-conf(mdl) : " + "  ".join(f"{c:.3f}" for c in agg(t_c, MAX_STEPS)))
print("  max-conf(em)  : " + "  ".join(f"{c:.3f}" for c in agg(te_c, MAX_STEPS)))

print(f"\n--- Val N5 / in-silico (n={n5v.sum()}) ---")
print(f"  baseline exact_match (per_noc_oracle)  : {bsl_v:.4f}")
print("  " + hdr)
print("  model peel    : " + "  ".join(f"{r:.4f}" for r in agg(v_r, MAX_STEPS)))
print("  EM-unif peel  : " + "  ".join(f"{r:.4f}" for r in agg(ve_r, MAX_STEPS)))
print("  oracle peel   : " + "  ".join(f"{r:.4f}" for r in agg(vo_r, MAX_STEPS)))
print("  max-conf(mdl) : " + "  ".join(f"{c:.3f}" for c in agg(v_c, MAX_STEPS)))
print("  max-conf(em)  : " + "  ".join(f"{c:.3f}" for c in agg(ve_c, MAX_STEPS)))

gap = agg(vo_r, MAX_STEPS)[4] - agg(v_r, MAX_STEPS)[4]
em_gap = agg(vo_r, MAX_STEPS)[4] - agg(ve_r, MAX_STEPS)[4]
print(f"  oracle gap at step 5 : model {gap:+.4f}  EM {em_gap:+.4f}")

print(f"\n--- phi rerank alpha sweep (single-pass, val then test) ---")
print("  alpha: " + "  ".join(f"{a:>5}" for a in ALPHAS))
print("  val:   " + "  ".join(f"{pr_val[a]:.4f}" for a in ALPHAS))
print("  test:  " + "  ".join(f"{pr_test[a]:.4f}" for a in ALPHAS))
print(f"  >> best val alpha={best_pr_alpha}  test={pr_test[best_pr_alpha]:.4f}")

print(f"\n--- priv_alpha sweep (phi_alpha={best_pr_alpha} fixed, val then test) ---")
print("  priv_a: " + "  ".join(f"{p:>5}" for p in PRIV_ALPHAS))
print("  val:    " + "  ".join(f"{prv_val[p]:.4f}" for p in PRIV_ALPHAS))
print("  test:   " + "  ".join(f"{prv_test[p]:.4f}" for p in PRIV_ALPHAS))
print(f"  >> best val priv_alpha={best_prv_alpha}  test={prv_test[best_prv_alpha]:.4f}"
      f"  (vs phi-only test={pr_test[best_pr_alpha]:.4f})")

print(f"\n--- Combined EM-peel+phi-rerank picking alpha sweep (val N5) ---")
print("  alpha: " + "  ".join(f"{a:>5}" for a in COMB_ALPHAS))
print("  step5: " + "  ".join(f"{comb_val5[a][4]:.4f}" for a in COMB_ALPHAS))
print("  step6: " + "  ".join(f"{comb_val5[a][5]:.4f}" for a in COMB_ALPHAS))
print(f"  >> best val alpha={best_comb_alpha}"
      f"  test step5={agg(cr_t, MAX_STEPS)[4]:.4f}"
      f"  step6={agg(cr_t, MAX_STEPS)[5]:.4f}")

# ── NOC estimation sweep on val N5 ────────────────────────────────────────
print(f"\n--- NOC estimation  (val N5, true NOC=5, n={n5v.sum()}) ---")

conf_ths = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
rej_ths  = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80]

print("\n  Criterion A model-conf (conf < threshold -> stop):")
print(f"  {'thresh':>7}  {'NOC=5%':>8}  {'mean':>6}  {'<5%':>6}  {'>5%':>5}")
for th in conf_ths:
    n = est_noc_conf(v_c, th)
    print(f"  {th:>7.2f}  {(n==5).mean():>8.3f}  {n.mean():>6.2f}  {(n<5).mean():>6.3f}  {(n>5).mean():>5.3f}")

print("\n  Criterion A EM-conf (conf < threshold -> stop):")
print(f"  {'thresh':>7}  {'NOC=5%':>8}  {'mean':>6}  {'<5%':>6}  {'>5%':>5}")
for th in conf_ths:
    n = est_noc_conf(ve_c, th)
    print(f"  {th:>7.2f}  {(n==5).mean():>8.3f}  {n.mean():>6.2f}  {(n<5).mean():>6.3f}  {(n>5).mean():>5.3f}")

print("\n  Criterion B (reject_prob > threshold -> OOD -> stop):")
print(f"  {'thresh':>7}  {'NOC=5%':>8}  {'mean':>6}  {'<5%':>6}  {'>5%':>5}")
for th in rej_ths:
    n = est_noc_rej(v_rj, th)
    print(f"  {th:>7.2f}  {(n==5).mean():>8.3f}  {n.mean():>6.2f}  {(n<5).mean():>6.3f}  {(n>5).mean():>5.3f}")

# ── NOC estimation sweep on REAL TEST N5 ──────────────────────────────────
print(f"\n--- NOC estimation  (real test N5, true NOC=5, n={n5t.sum()}) ---")

print("\n  Criterion A model-conf (conf < threshold -> stop):")
print(f"  {'thresh':>7}  {'NOC=5%':>8}  {'mean':>6}")
for th in conf_ths:
    n = est_noc_conf(t_c, th)
    print(f"  {th:>7.2f}  {(n==5).mean():>8.3f}  {n.mean():>6.2f}")

print("\n  Criterion A EM-conf (conf < threshold -> stop):")
print(f"  {'thresh':>7}  {'NOC=5%':>8}  {'mean':>6}")
for th in conf_ths:
    n = est_noc_conf(te_c, th)
    print(f"  {th:>7.2f}  {(n==5).mean():>8.3f}  {n.mean():>6.2f}")

print("\n  Criterion B (reject_prob > threshold -> OOD -> stop):")
print(f"  {'thresh':>7}  {'NOC=5%':>8}  {'mean':>6}")
for th in rej_ths:
    n = est_noc_rej(t_rj, th)
    print(f"  {th:>7.2f}  {(n==5).mean():>8.3f}  {n.mean():>6.2f}")

# ── All-NOC EM NOC estimation (real test) ─────────────────────────────────
print(f"\n--- All-NOC NOC estimation: EM Criterion A vs card_head (real test) ---")
card_noc_acc = meta.get("card_noc_acc", None)
print(f"  card_head NOC accuracy: {card_noc_acc}")

# Run EM pipeline on full test set (all NOC) with a reasonable max_steps=6
print("  Running EM pipeline on all NOC test samples ...", flush=True)
all_em_r, all_em_c, _ = run_pipeline(Xt, Mt, Yt, use_em=True,
                                       update_mask=True, max_steps=6)

noc_th = 0.10  # Criterion A threshold (100% on val N5)

print(f"\n  Criterion A EM (threshold={noc_th}):")
print(f"  {'NOC':>4}  {'n':>5}  {'acc%':>8}  {'card_head':>10}")
all_noc_correct = 0; all_n = 0
for noc_k in [1, 2, 3, 4, 5]:
    mask  = NOCt == noc_k
    if mask.sum() == 0: continue
    idx   = np.where(mask)[0]
    per_c = [all_em_c[i] for i in idx]
    noc_est = est_noc_conf(per_c, noc_th)
    acc = float((noc_est == noc_k).mean())
    all_noc_correct += (noc_est == noc_k).sum()
    all_n += mask.sum()
    ch = meta.get("per_noc", {}).get(str(noc_k), "?")
    print(f"  {noc_k:>4}  {mask.sum():>5}  {acc:>8.4f}  {ch!s:>10}")

overall_em_noc_acc = all_noc_correct / max(1, all_n)
print(f"  Overall EM NOC acc: {overall_em_noc_acc:.4f}  vs card_head: {card_noc_acc}")

# Threshold sweep: find best for all-NOC overall accuracy
print(f"\n  Threshold sweep (overall NOC acc across N1-N5):")
print(f"  {'thresh':>7}  {'overall':>9}  {'N1':>6}  {'N2':>6}  {'N3':>6}  {'N4':>6}  {'N5':>6}")
for th in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    row_vals = []; tot_c = 0; tot_n = 0
    for noc_k in [1, 2, 3, 4, 5]:
        mask = NOCt == noc_k
        if mask.sum() == 0: row_vals.append(float("nan")); continue
        idx = np.where(mask)[0]
        per_c = [all_em_c[i] for i in idx]
        noc_est = est_noc_conf(per_c, th)
        acc = float((noc_est == noc_k).mean())
        row_vals.append(acc)
        tot_c += (noc_est == noc_k).sum(); tot_n += mask.sum()
    ov = tot_c / max(1, tot_n)
    print(f"  {th:>7.2f}  {ov:>9.4f}  " + "  ".join(f"{v:>6.4f}" for v in row_vals))

# ── NOC estimation via priv_score (no peeling needed) ────────────────────────
print(f"\n--- NOC estimation via priv_score (real test, ALL NOC) ---")
print("  Hypothesis: true donors have priv_score>0, decoys have priv_score=0")
m_feas_t = feas_prefilter(Mt, Xt)
priv_t = compute_priv_score(Xt, m_feas_t)  # [B, C]

PRIV_THRESHOLDS = [0.0, 10.0, 30.0, 50.0, 100.0, 200.0, 500.0]
print(f"\n  {'thresh':>8}  {'overall':>9}  {'N1':>6}  {'N2':>6}  {'N3':>6}  {'N4':>6}  {'N5':>6}")
for th in PRIV_THRESHOLDS:
    row = []; tot_c = 0; tot_n = 0
    for noc_k in [1, 2, 3, 4, 5]:
        mask = NOCt == noc_k
        if mask.sum() == 0: row.append(float("nan")); continue
        noc_est = est_noc_priv(priv_t[mask], th)
        acc = float((noc_est == noc_k).mean())
        row.append(acc); tot_c += (noc_est == noc_k).sum(); tot_n += mask.sum()
    ov = tot_c / max(1, tot_n)
    print(f"  {th:>8.0f}  {ov:>9.4f}  " + "  ".join(f"{v:>6.4f}" for v in row))
print(f"  card_head: {card_noc_acc}")

print(f"\n--- priv_score stats (real test N5, true vs decoy) ---")
n5_mask = NOCt == 5
priv_n5 = priv_t[n5_mask]   # [N5, C]
y_n5    = Yt[n5_mask]
true_priv  = [priv_n5[i, j] for i in range(len(priv_n5))
              for j in range(C) if y_n5[i, j] > 0.5]
decoy_priv = [priv_n5[i, j] for i in range(len(priv_n5))
              for j in range(C) if y_n5[i, j] < 0.5]
print(f"  true donors  n={len(true_priv)}  mean={np.mean(true_priv):.1f}"
      f"  median={np.median(true_priv):.1f}  zero%={(np.array(true_priv)==0).mean():.3f}")
print(f"  decoys       n={len(decoy_priv)}  mean={np.mean(decoy_priv):.1f}"
      f"  median={np.median(decoy_priv):.1f}  zero%={(np.array(decoy_priv)==0).mean():.3f}")

# ── All-NOC peeling exact-match (real test, model peel) ────────────────────
print(f"\n--- All-NOC exact-match vs baseline (real test, model peel) ---")
print(f"  {'NOC':>4}  {'n':>5}  {'baseline':>9}  {'step1':>6}  {'step2':>6}  {'step3':>6}  {'step4':>6}  {'step5':>6}")
for noc_k in [1, 2, 3, 4, 5]:
    mask = NOCt == noc_k
    if mask.sum() == 0: continue
    Xk, Mk, Yk = Xt[mask], Mt[mask], Yt[mask]
    bk = direct_topk_exact(Xk, Mk, Yk, k=noc_k)
    rk, _, _ = run_pipeline(Xk, Mk, Yk, update_mask=True, max_steps=5)
    steps_ = agg(rk, 5)
    row = f"  {noc_k:>4}  {mask.sum():>5}  {bk:>9.4f}  " + "  ".join(f"{steps_[s]:>6.4f}" for s in range(5))
    print(row)

print("\n" + "=" * W)
print("Interpretation:")
print(f"  inc6_maskp baseline N5 oracle: {meta.get('per_noc_oracle', {}).get('5', '?')}")
print(f"  Peeling real test N5 step-5 recall: {agg(t_r,MAX_STEPS)[4]:.4f} ({agg(t_r,MAX_STEPS)[4]-bsl_t:+.4f} vs baseline)")
print(f"  Oracle gap (val N5, step 5): {gap:+.4f} -- upper bound from better attr training")
print("  reject_prob rising = OOD = all donors identified; look for threshold above.")
print("=" * W)
