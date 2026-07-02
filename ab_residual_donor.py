"""
ab_residual_donor.py  v2 - iterative residual-donor subtraction with feas-mask update.

v2 change: after identifying donor k*, zero out peaks with no remaining carrier
(owner_lut private peaks become zombie 0-height tokens in v1 -> now masked out).
This makes oracle subtraction work correctly and closes the distribution shift.

Two NOC stopping criteria:
  A (conf):  stop when max_remaining_cls_prob < threshold
  B (OOD):   stop when normalized_entropy(remaining cls soft) > threshold

Comparison: v1 (no mask update) vs v2 (mask update) recall on real test N5 and val N5.
"""

from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture

SEED = 42
TRAIN_N = 12_000
EPOCHS = 30
BATCH = 128
LR = 3e-4
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(s):
    import random; random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

set_seed(SEED)

# ── Load data ──────────────────────────────────────────────────────────────
print("Loading data ...", flush=True)
X    = np.load(DATA_DIR / "tokens_train.npy").astype(np.float32)
M    = np.load(DATA_DIR / "mask_train.npy")
Y    = np.load(DATA_DIR / "y_train_set.npy").astype(np.float32)
NOC  = np.load(DATA_DIR / "noc_train.npy").astype(np.int64)
ATTR = np.load(DATA_DIR / "attr_train.npy").astype(np.int64)
PHI  = np.load(DATA_DIR / "phi_train.npy").astype(np.float32)

Xt   = np.load(DATA_DIR / "tokens_test.npy").astype(np.float32)
Mt   = np.load(DATA_DIR / "mask_test.npy")
Yt   = np.load(DATA_DIR / "y_test_set.npy").astype(np.float32)
NOCt = np.load(DATA_DIR / "noc_test.npy").astype(np.int64)

C = Y.shape[1]
print(f"Donors={C}  Train={len(X)}  Test={len(Xt)}", flush=True)

rng = np.random.default_rng(SEED)
idx = rng.choice(len(X), min(TRAIN_N, len(X)), replace=False)
X, M, Y, NOC, ATTR, PHI = X[idx], M[idx], Y[idx], NOC[idx], ATTR[idx], PHI[idx]
TRAIN_N = len(X)

n5m = NOCt == 5
Xt5, Mt5, Yt5 = Xt[n5m], Mt[n5m], Yt[n5m]
print(f"N5 test: {n5m.sum()}", flush=True)

# ── Genotype LUT ───────────────────────────────────────────────────────────
geno_path = None
for cand in (DATA_DIR / "donor_geno.npy", ROOT / "data" / "donor_geno.npy", ROOT / "donor_geno.npy"):
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
CN_t = torch.from_numpy(CN).to(DEV)
print(f"CN LUT: {CN.shape}  max={int(CN.max())}", flush=True)

# ── Soft labels + geno mask ────────────────────────────────────────────────
def build_soft_labels(xb, mb, phib):
    """EuroForMix phi*CN soft attr labels: (B, N, C+1)."""
    li = xb[..., 0].long().clamp(0, 23)
    bi = (xb[..., 1] * 10).round().long().clamp(0, LUT_W - 1)
    cn = CN_t[li, bi]                         # (B, N, C)
    phi_cn = phib.unsqueeze(1) * cn
    total  = phi_cn.sum(-1, keepdim=True)
    feas   = (total.squeeze(-1) > 1e-9) & mb.bool()
    norm   = phi_cn / total.clamp(min=1e-9)
    soft_y = torch.zeros(*xb.shape[:2], C + 1, device=DEV)
    soft_y[..., :C] = norm * feas.unsqueeze(-1).float()
    soft_y[..., C]  = (~feas & mb.bool()).float()
    return soft_y

def apply_geno_mask(AL_np, x_np, m_np):
    """Genotype-consistency mask on attr logits (B, N, C+1), in-place."""
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

# ── Dataset + DataLoader ───────────────────────────────────────────────────
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

def make_loader(xn, mn, yn, nocn, atn, phin, batch=BATCH, shuffle=True):
    ds = TensorDataset(
        torch.from_numpy(xn), torch.from_numpy(mn.astype(np.bool_)),
        torch.from_numpy(yn), torch.from_numpy(nocn),
        torch.from_numpy(atn), torch.from_numpy(phin),
    )
    g_ = torch.Generator(); g_.manual_seed(SEED)
    return DataLoader(ds, batch_size=batch,
                      sampler=RandomSampler(ds, generator=g_) if shuffle else None,
                      shuffle=False)

train_loader = make_loader(X, M, Y, NOC, ATTR, PHI)

# ── Model ──────────────────────────────────────────────────────────────────
def make_model():
    return SetTransformerMixture(
        n_loci=24, d_locus=8, d_model=64, n_heads=4, n_isab=2,
        m_inducing=16, n_classes=C, n_noc=6, dropout=0.1,
        aux_heads=True, cls_decoder="per_donor", dec_aggr="sparsemax",
        dec_layers=2, sparse_attn=True,
    ).to(DEV)

# ── Training (soft CE on attr head) ────────────────────────────────────────
print("\n=== Training (soft CE) ===", flush=True)
model = make_model()
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
log_var_attr = torch.zeros((), device=DEV, requires_grad=True)
log_var_phi  = torch.zeros((), device=DEV, requires_grad=True)
opt.add_param_group({"params": [log_var_attr, log_var_phi], "weight_decay": 0.0})
bce = nn.BCEWithLogitsLoss()

for ep in range(1, EPOCHS + 1):
    model.train(); tot = 0.0; nb = 0
    for xb, mb, yb, nocb, atb, phib in train_loader:
        xb, mb  = xb.to(DEV), mb.to(DEV)
        yb, atb, phib = yb.to(DEV), atb.to(DEV), phib.to(DEV)
        out  = model(xb, mb)
        loss = bce(out["logits_cls"], yb)
        if "logits_attr" in out:
            la      = out["logits_attr"]
            soft_y  = build_soft_labels(xb, mb, phib)
            log_p   = F.log_softmax(la, dim=-1)
            raw_l   = -(soft_y * log_p).sum(-1)
            vm      = mb.float()
            l_attr  = (raw_l * vm).sum() / vm.sum().clamp(min=1)
            l_phi   = F.l1_loss(out["phi"], phib)
            loss    = loss + (torch.exp(-log_var_attr) * l_attr + log_var_attr
                              + torch.exp(-log_var_phi)  * l_phi  + log_var_phi)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        tot += loss.item(); nb += 1
    if ep % 10 == 0:
        print(f"  ep {ep}/{EPOCHS}  loss={tot/nb:.3f}", flush=True)

# ── Inference helpers ──────────────────────────────────────────────────────

@torch.no_grad()
def get_probs_attr_v2(x_np, m_np, excluded_list=None):
    """
    Returns:
      cls_p     (B, C)       - sigmoid cls probs (excluded donors zeroed)
      attr_p    (B, N, C+1)  - softmax attr probs after geno-mask
      ood_score (B,)         - normalized entropy of remaining cls distribution
                               0 = confident (in-dist); 1 = uniform (OOD)
    """
    model.eval()
    xb = torch.from_numpy(x_np).to(DEV)
    mb = torch.from_numpy(m_np.astype(np.bool_)).to(DEV)
    out = model(xb, mb)
    cls_p = torch.sigmoid(out["logits_cls"]).cpu().numpy()
    la    = out["logits_attr"].cpu().numpy()

    if excluded_list is not None:
        for i, excl in enumerate(excluded_list):
            for d in excl:
                cls_p[i, d] = 0.0
                la[i, :, d] = -1e9

    apply_geno_mask(la, x_np, m_np)
    attr_p = np.exp(la - la.max(-1, keepdims=True))
    attr_p /= attr_p.sum(-1, keepdims=True).clip(min=1e-9)

    # Normalized entropy of remaining cls soft-distribution
    eps = 1e-9
    B = len(x_np)
    ood_scores = np.ones(B, np.float32)
    for i in range(B):
        excl   = excluded_list[i] if excluded_list else set()
        avail  = [c for c in range(C) if c not in excl]
        if len(avail) < 2:
            continue
        p_rem  = cls_p[i, avail].clip(0.0)
        p_sum  = p_rem.sum()
        if p_sum < eps:
            ood_scores[i] = 1.0
            continue
        p_soft = p_rem / p_sum
        H      = -(p_soft * np.log(p_soft + eps)).sum()
        H_max  = np.log(len(avail))
        ood_scores[i] = float(H / H_max) if H_max > eps else 1.0

    return cls_p, attr_p, ood_scores


def subtract_donor(x_np, m_np, attr_p, donor_k):
    """Remove donor k contribution via model attr (1 - attr_frac)."""
    x2 = x_np.copy()
    h  = np.expm1(x_np[:, :, 2])
    frac = attr_p[:, :, donor_k].clip(0.0, 1.0)
    x2[:, :, 2] = np.log1p((h * (1.0 - frac)).clip(min=0.0))
    return x2


def oracle_subtract_v2(x_np, m_np, phi_b, donor_k, excluded_set):
    """
    Oracle subtraction: fraction = phi_k * CN_k / sum_{remaining}(phi_c * CN_c).
    Uses remaining donors (excluded_set) in denominator to avoid double-counting.
    """
    x2  = x_np.copy()
    h   = np.expm1(x_np[:, :, 2])
    li  = x_np[:, :, 0].astype(int).clip(0, 23)
    bi  = np.round(x_np[:, :, 1] * 10).astype(int).clip(0, LUT_W - 1)

    # Remaining donors = not in excluded_set AND not the current donor_k
    rem_mask = np.ones(C, bool)
    for d in excluded_set:
        rem_mask[d] = False
    # donor_k is in excluded_set after this call (we include it in denominator now)
    # denominator = remaining donors before k is removed (k is still "present")

    cn_all  = CN[li, bi, :]                                   # (B, N, C)
    phi_rem = phi_b * rem_mask[np.newaxis, :]                  # (B, C)
    phi_cn  = phi_rem[:, np.newaxis, :] * cn_all               # (B, N, C)
    total   = phi_cn.sum(-1).clip(min=1e-9)                    # (B, N)
    cn_k    = cn_all[:, :, donor_k]                            # (B, N)
    frac    = (phi_b[:, donor_k:donor_k+1] * cn_k / total).clip(0.0, 1.0)
    x2[:, :, 2] = np.log1p((h * (1.0 - frac)).clip(min=0.0))
    return x2


def update_feas_mask(m_cur, x_np_orig, excluded_list):
    """
    After identifying donors in excluded_list, mask peaks with no remaining carrier.
    Private peaks of identified donors had their height subtracted -> now 0-height
    zombie tokens that confuse the model. This removes them from the mask.
    """
    m_new = m_cur.copy()
    for i in range(len(x_np_orig)):
        excl = excluded_list[i]
        if not excl:
            continue
        for pk_idx in np.where(m_cur[i])[0]:
            l_ = int(round(float(x_np_orig[i, pk_idx, 0])))
            b_ = int(round(float(x_np_orig[i, pk_idx, 1]) * 10))
            if not (0 <= l_ < 24 and 0 <= b_ < LUT_W):
                continue
            cn_rem = CN[l_, b_, :].copy()
            for d in excl:
                cn_rem[d] = 0.0
            if cn_rem.sum() < 1e-9:
                m_new[i, pk_idx] = False
    return m_new

# ── Iterative pipeline ─────────────────────────────────────────────────────

def run_pipeline(x_np, m_np, y_np, phi_np=None, use_oracle=False,
                 max_steps=6, update_mask=True):
    """
    Generic iterative subtraction pipeline.

    update_mask=False -> v1 (no feas-mask update, zombie peaks persist)
    update_mask=True  -> v2 (feas-mask updated each step)

    Returns per-sample:
      per_recalls[i][s] = recall after picking s+1 donors (0-indexed step s)
      per_confs[i][s]   = max_cls_prob at step s (confidence of the pick)
      per_ood[i][s]     = normalized entropy BEFORE step s (0=confident, 1=uniform/OOD)
    """
    B        = len(x_np)
    x_cur    = x_np.copy()
    m_cur    = m_np.copy()
    excluded = [set() for _ in range(B)]

    per_recalls = [[] for _ in range(B)]
    per_confs   = [[] for _ in range(B)]
    per_ood     = [[] for _ in range(B)]
    last_ks     = []

    for step in range(max_steps):
        cls_p, attr_p, ood_sc = get_probs_attr_v2(x_cur, m_cur, excluded)
        step_ks = []

        for i in range(B):
            per_ood[i].append(float(ood_sc[i]))

            avail = [c for c in range(C) if c not in excluded[i]]
            if not avail:
                per_confs[i].append(0.0)
                prev = per_recalls[i][-1] if per_recalls[i] else 0.0
                per_recalls[i].append(prev)
                step_ks.append(-1)
                continue

            k    = int(max(avail, key=lambda c: cls_p[i, c]))
            conf = float(cls_p[i, k])
            per_confs[i].append(conf)
            excluded[i].add(k)
            step_ks.append(k)

            id_set   = set(excluded[i])
            true_set = set(np.where(y_np[i] > 0.5)[0])
            per_recalls[i].append(len(id_set & true_set) / max(1, len(true_set)))

        # Update feas mask BEFORE subtraction (removes zombie peaks)
        if update_mask:
            m_cur = update_feas_mask(m_cur, x_np, excluded)

        # Subtract identified donor from x_cur
        x_next = x_cur.copy()
        for i in range(B):
            k = step_ks[i]
            if k < 0: continue
            if use_oracle and phi_np is not None:
                x_next[i:i+1] = oracle_subtract_v2(
                    x_cur[i:i+1], m_cur[i:i+1], phi_np[i:i+1], k, excluded[i])
            else:
                x_next[i:i+1] = subtract_donor(
                    x_cur[i:i+1], m_cur[i:i+1], attr_p[i:i+1], k)
        x_cur = x_next

    return per_recalls, per_confs, per_ood


def agg_step(per_vals, steps):
    return [float(np.mean([per_vals[i][s] for i in range(len(per_vals))]))
            for s in range(steps)]


# ── Baseline: direct top-5 recall ─────────────────────────────────────────

@torch.no_grad()
def direct_top5_recall(x_np, m_np, y_np):
    model.eval()
    xb = torch.from_numpy(x_np).to(DEV)
    mb = torch.from_numpy(m_np.astype(np.bool_)).to(DEV)
    cls_p = torch.sigmoid(model(xb, mb)["logits_cls"]).cpu().numpy()
    top5  = np.argsort(-cls_p, axis=1)[:, :5]
    recs  = [len(set(top5[i]) & set(np.where(y_np[i] > 0.5)[0])) / 5
             for i in range(len(x_np))]
    return float(np.mean(recs))

# ── NOC estimation helpers ─────────────────────────────────────────────────

def est_noc_conf(per_confs, threshold):
    """
    Criterion A: NOC = number of sequential steps with conf >= threshold.
    Stop at first step where conf drops below threshold.
    """
    noc_est = np.ones(len(per_confs), int)
    for i, confs in enumerate(per_confs):
        count = 0
        for c in confs:
            if c >= threshold:
                count += 1
            else:
                break
        noc_est[i] = max(1, count)
    return noc_est.clip(1, 5)


def est_noc_ood(per_ood, threshold):
    """
    Criterion B: NOC = number of steps before OOD entropy fires (> threshold).
    Per-step OOD is computed BEFORE picking -> NOC = steps where ood < threshold.
    """
    noc_est = np.ones(len(per_ood), int)
    for i, oods in enumerate(per_ood):
        count = 0
        for h in oods:
            if h < threshold:  # still in-distribution
                count += 1
            else:
                break          # OOD detected -> stop
        noc_est[i] = max(1, count)
    return noc_est.clip(1, 5)

# ── Val N5 (in-silico, has GT phi for oracle) ─────────────────────────────

Xv   = np.load(DATA_DIR / "tokens_val.npy").astype(np.float32)
Mv   = np.load(DATA_DIR / "mask_val.npy")
Yv   = np.load(DATA_DIR / "y_val_set.npy").astype(np.float32)
NOCv = np.load(DATA_DIR / "noc_val.npy").astype(np.int64)
PHIv = np.load(DATA_DIR / "phi_val.npy").astype(np.float32)

n5v = NOCv == 5
Xv5, Mv5, Yv5, PHIv5 = Xv[n5v], Mv[n5v], Yv[n5v], PHIv[n5v]
print(f"N5 val (in-silico): {n5v.sum()}", flush=True)

# ── Run all experiments ────────────────────────────────────────────────────
print("\n=== Inference ===", flush=True)
MAX_STEPS = 6

# Baselines
bsl_test = direct_top5_recall(Xt5, Mt5, Yt5)
bsl_val  = direct_top5_recall(Xv5, Mv5, Yv5)

# Real test N5: v1 and v2 model-attr
print("Real test N5 - v1 (no mask update) ...", flush=True)
t_r1, t_c1, t_o1 = run_pipeline(Xt5, Mt5, Yt5, update_mask=False, max_steps=MAX_STEPS)

print("Real test N5 - v2 (mask update) ...", flush=True)
t_r2, t_c2, t_o2 = run_pipeline(Xt5, Mt5, Yt5, update_mask=True,  max_steps=MAX_STEPS)

# Val N5: v1 model, v2 model, v2 oracle
print("Val N5 - v1 model ...", flush=True)
v_r1, v_c1, v_o1 = run_pipeline(Xv5, Mv5, Yv5, update_mask=False, max_steps=MAX_STEPS)

print("Val N5 - v2 model ...", flush=True)
v_r2, v_c2, v_o2 = run_pipeline(Xv5, Mv5, Yv5, update_mask=True,  max_steps=MAX_STEPS)

print("Val N5 - v2 oracle ...", flush=True)
v_r2o, v_c2o, v_o2o = run_pipeline(Xv5, Mv5, Yv5, PHIv5,
                                     use_oracle=True, update_mask=True, max_steps=MAX_STEPS)

# ── Print tables ───────────────────────────────────────────────────────────
W = 78
print("\n" + "=" * W)
print(f"  Residual Subtraction v2  (train={TRAIN_N} ep={EPOCHS} seed={SEED})")
print("=" * W)

steps = list(range(1, MAX_STEPS + 1))
hdr = "step:  " + "  ".join(f"{s}" for s in steps)

print(f"\n--- Real test N5 (n={n5m.sum()}) ---")
print(f"  baseline top-5 recall : {bsl_test:.3f}")
print("  " + hdr)
print("  v1 model recall : " + "  ".join(f"{r:.3f}" for r in agg_step(t_r1, MAX_STEPS)))
print("  v2 model recall : " + "  ".join(f"{r:.3f}" for r in agg_step(t_r2, MAX_STEPS)))
print("  v1 max-conf     : " + "  ".join(f"{c:.3f}" for c in agg_step(t_c1, MAX_STEPS)))
print("  v2 max-conf     : " + "  ".join(f"{c:.3f}" for c in agg_step(t_c2, MAX_STEPS)))
print("  v1 ood-entropy  : " + "  ".join(f"{h:.3f}" for h in agg_step(t_o1, MAX_STEPS)))
print("  v2 ood-entropy  : " + "  ".join(f"{h:.3f}" for h in agg_step(t_o2, MAX_STEPS)))

print(f"\n--- Val N5 / in-silico (n={n5v.sum()}) ---")
print(f"  baseline top-5 recall : {bsl_val:.3f}")
print("  " + hdr)
print("  v1 model recall : " + "  ".join(f"{r:.3f}" for r in agg_step(v_r1, MAX_STEPS)))
print("  v2 model recall : " + "  ".join(f"{r:.3f}" for r in agg_step(v_r2, MAX_STEPS)))
print("  v2 oracle recall: " + "  ".join(f"{r:.3f}" for r in agg_step(v_r2o, MAX_STEPS)))
print("  v1 max-conf     : " + "  ".join(f"{c:.3f}" for c in agg_step(v_c1, MAX_STEPS)))
print("  v2 max-conf     : " + "  ".join(f"{c:.3f}" for c in agg_step(v_c2, MAX_STEPS)))
print("  v1 ood-entropy  : " + "  ".join(f"{h:.3f}" for h in agg_step(v_o1, MAX_STEPS)))
print("  v2 ood-entropy  : " + "  ".join(f"{h:.3f}" for h in agg_step(v_o2, MAX_STEPS)))

# ── NOC estimation sweep: val N5 (true NOC=5) ─────────────────────────────
print(f"\n--- NOC estimation (val N5, true NOC=5, n={n5v.sum()}) ---")

# Criterion A: confidence threshold sweep
conf_ths = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
print("\n  Criterion A (conf): stop when max_cls_prob < threshold")
print(f"  {'thresh':>7}  {'NOC=5 %':>8}  {'mean est':>9}  {'<5 %':>6}  {'>5 %':>5}")
print("  " + "-" * 42)
for th in conf_ths:
    n_est = est_noc_conf(v_c2, th)
    acc5  = float((n_est == 5).mean())
    mean_ = float(n_est.mean())
    lt5   = float((n_est < 5).mean())
    gt5   = float((n_est > 5).mean())
    print(f"  {th:>7.2f}  {acc5:>8.3f}  {mean_:>9.2f}  {lt5:>6.3f}  {gt5:>5.3f}")

# Criterion B: OOD entropy threshold sweep
ood_ths = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
print("\n  Criterion B (OOD entropy): stop when normalized_H > threshold")
print(f"  {'thresh':>7}  {'NOC=5 %':>8}  {'mean est':>9}  {'<5 %':>6}  {'>5 %':>5}")
print("  " + "-" * 42)
for th in ood_ths:
    n_est = est_noc_ood(v_o2, th)
    acc5  = float((n_est == 5).mean())
    mean_ = float(n_est.mean())
    lt5   = float((n_est < 5).mean())
    gt5   = float((n_est > 5).mean())
    print(f"  {th:>7.2f}  {acc5:>8.3f}  {mean_:>9.2f}  {lt5:>6.3f}  {gt5:>5.3f}")

# ── Same sweep on real test N5 (true NOC=5) ───────────────────────────────
print(f"\n--- NOC estimation (real test N5, true NOC=5, n={n5m.sum()}) ---")
print("\n  Criterion A (conf): ")
print(f"  {'thresh':>7}  {'NOC=5 %':>8}  {'mean est':>9}")
print("  " + "-" * 28)
for th in conf_ths:
    n_est = est_noc_conf(t_c2, th)
    print(f"  {th:>7.2f}  {float((n_est==5).mean()):>8.3f}  {float(n_est.mean()):>9.2f}")

print("\n  Criterion B (OOD entropy):")
print(f"  {'thresh':>7}  {'NOC=5 %':>8}  {'mean est':>9}")
print("  " + "-" * 28)
for th in ood_ths:
    n_est = est_noc_ood(t_o2, th)
    print(f"  {th:>7.2f}  {float((n_est==5).mean()):>8.3f}  {float(n_est.mean()):>9.2f}")

# ── Summary ────────────────────────────────────────────────────────────────
print("\n" + "=" * W)
print("Key comparisons at step 5:")
d_t = agg_step(t_r2, MAX_STEPS)[4] - agg_step(t_r1, MAX_STEPS)[4]
d_v = agg_step(v_r2, MAX_STEPS)[4] - agg_step(v_r1, MAX_STEPS)[4]
d_o = agg_step(v_r2o, MAX_STEPS)[4] - agg_step(v_r2, MAX_STEPS)[4]
print(f"  v2 vs v1 (mask update) real test N5 : {d_t:+.3f}")
print(f"  v2 vs v1 (mask update) val N5       : {d_v:+.3f}")
print(f"  oracle vs model (v2) val N5          : {d_o:+.3f}  (+ = oracle better; should be + in v2)")
print(f"  v2 baseline vs v2 step5 real test   : {agg_step(t_r2,MAX_STEPS)[4]-bsl_test:+.3f}")
print(f"  v2 baseline vs v2 step5 val         : {agg_step(v_r2,MAX_STEPS)[4]-bsl_val:+.3f}")
print("=" * W)
print("If v2 oracle > v2 model: subtraction quality is the bottleneck (train better attr).")
print("If v2 model > baseline:  peeling helps even with imperfect attr.")
print("NOC: look for threshold where NOC=5% is highest on real test.")
