"""
Train Set Transformer for 45-class multi-label contributor identification.

Data:
  tokens_{split}.npy  (N, 160, 3)  — set-structured allele tokens
  mask_{split}.npy    (N, 160)     — True = valid token
  y_{split}_set.npy   (N, 45)      — multi-label targets
  noc_{split}.npy     (N,)         — number of contributors (0-5)
  tokens_open.npy, mask_open.npy   — open-set for reject head training

Multi-task loss (planv2):
  L = BCE(cls, closed-set only) + α·BCE(reject, all) + β·CE(noc, closed-set only)

Reject head training strategy:
  - Closed-set samples in batch → reject_label = 0
  - Open-set samples appended per batch → reject_label = 1
  Each training step sees (batch_size) closed + (open_ratio * batch_size) open samples.

Usage:
  python train_set_transformer.py
  python train_set_transformer.py --config configs/set_transformer.json
  python train_set_transformer.py --n_isab 3 --epochs 80
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.metrics import f1_score, roc_auc_score

import sys

import os
ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from models.ordinal import corn_loss, corn_probs, supcon_loss   # Inc3 levers (verified primitives)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> torch.Generator:
    """Make a run reproducible: seed python/numpy/torch (+cuda) and return a seeded
    Generator for the train DataLoader's shuffle. Without this, weight init + batch
    order + dropout differ every run -> ~±10pp swings on small NOC strata (n=44-48),
    which swamps per-arm effects. Fix the seed for comparability; sweep seeds for CIs."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ── Cardinality-aware decoding (Cortes/Mohri 2024) ──────────────────────────

def cardinality_target(probs: torch.Tensor, y: torch.Tensor,
                       lam: float = 0.02, n_card: int = 5) -> torch.Tensor:
    """EM-optimal NOC target per sample: argmin_k [set_mismatch(top-k, y) + lam*k].
    Returns target in {0..n_card-1} (= k-1). Vectorized, no_grad."""
    with torch.no_grad():
        B, C = probs.shape
        order = probs.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        ar = torch.arange(C, device=probs.device).expand(B, C)
        rank.scatter_(1, order, ar)                       # rank[i,c] = position (0=top)
        K = y.sum(1).clamp(min=1)
        costs = []
        for k in range(1, n_card + 1):
            topk = (rank < k).float()
            miss = (y * (1 - topk)).sum(1) / K
            extra = ((1 - y) * topk).sum(1) / k
            costs.append(miss + extra + lam * k)
        return torch.stack(costs, 1).argmin(1)            # (B,) in {0..n_card-1}


def topk_decode(probs: np.ndarray, k_arr: np.ndarray) -> np.ndarray:
    """Binary multi-label prediction: top-k_arr[i] donors per sample."""
    yp = np.zeros_like(probs, dtype=int)
    for i in range(len(probs)):
        k = int(max(1, min(5, round(k_arr[i]))))
        yp[i, np.argsort(probs[i])[::-1][:k]] = 1
    return yp


def _card_feats(P):
    s = np.sort(P, 1)[:, ::-1][:, :8]
    return np.concatenate([s, P.sum(1, keepdims=True), (P >= 0.5).sum(1, keepdims=True)], 1)


def posthoc_cardinality(P_val, y_val, P_test, lam: float = 0.02):
    """Robust post-hoc cardinality: RandomForest on sorted-prob profile, fit on val.
    Handles imbalance naturally (reads final clean probs)."""
    from sklearn.ensemble import RandomForestClassifier
    C = P_val.shape[1]
    tk = np.ones(len(P_val), int)
    for i in range(len(P_val)):
        K = max(int(y_val[i].sum()), 1); best, bc = 1, 9e9
        for k in range(1, 6):
            yp = np.zeros(C); yp[np.argsort(P_val[i])[::-1][:k]] = 1
            c = (y_val[i]*(1-yp)).sum()/K + ((1-y_val[i])*yp).sum()/k + lam*k
            if c < bc: bc, best = c, k
        tk[i] = best
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(_card_feats(P_val), tk)
    return rf.predict(_card_feats(P_test))


def _rank_feats(S):
    """Count features for a RANKING SCORE (not a probability): sorted top scores + consecutive gaps —
    the 'elbow' between true donors and decoys, which the rerank sharpens."""
    s = np.sort(S, 1)[:, ::-1][:, :12]
    return np.concatenate([s, -np.diff(s, axis=1)], 1)            # top-12 + 11 elbow gaps


def posthoc_cardinality_rank(P_val, S_val, y_val, P_test, S_test, lam: float = 0.02):
    """Count fit on BOTH the prob-profile (P; good for low NOC) AND the reranked-score profile (S;
    sharper true-vs-decoy elbow at high NOC). Cost-derived target on the reranked ranking — the thing
    actually decoded. NOTE: on single-profile data this lifts N5 toward oracle but TRADES N3/N4
    (the count is trade-bound, C1); intended for replicate data where the new info may break the trade."""
    from sklearn.ensemble import RandomForestClassifier
    C = S_val.shape[1]; tk = np.ones(len(S_val), int)
    for i in range(len(S_val)):
        K = max(int(y_val[i].sum()), 1); best, bc = 1, 9e9
        for k in range(1, 6):
            yp = np.zeros(C); yp[np.argsort(S_val[i])[::-1][:k]] = 1
            c = (y_val[i] * (1 - yp)).sum() / K + ((1 - y_val[i]) * yp).sum() / k + lam * k
            if c < bc: bc, best = c, k
        tk[i] = best
    Fv = np.concatenate([_card_feats(P_val), _rank_feats(S_val)], 1)
    Ft = np.concatenate([_card_feats(P_test), _rank_feats(S_test)], 1)
    return RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(Fv, tk).predict(Ft)


def per_noc_em(y_true, y_pred, noc):
    e = np.all(y_true == y_pred, axis=1)
    return [e.mean()] + [e[noc == j].mean() if (noc == j).sum() else float("nan") for j in range(1, 6)]


# ── Combo-invariant MAC/height count features + two-stage cardinality ────────
_MAC_THRESHOLDS = [0, 50, 100, 150, 250, 500]


def mac_feats(tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """46 combo-invariant per-locus allele-count + height features (PACE/deepNoC
    style; literature: MAC/TAC are the physical NOC signal). tokens=(N,160,3)
    [locus_idx, allele, log1p_height], mask=(N,160) bool."""
    h = np.expm1(tokens[:, :, 2]); out = []
    for i in range(len(tokens)):
        v0 = mask[i]; loci_all = tokens[i, :, 0].astype(int); row = []
        for thr in _MAC_THRESHOLDS:
            v = v0 & (h[i] > thr); loci = loci_all[v]
            if len(loci):
                c = np.bincount(loci, minlength=24); nz = c[c > 0]
                row += [c.max(), len(loci), int((c >= 2).sum()),
                        int((c >= 3).sum()), int((c >= 4).sum()), float(nz.mean())]
            else:
                row += [0, 0, 0, 0, 0, 0.0]
        hv = h[i][v0]
        if len(hv):
            row += list(np.percentile(hv, [10, 25, 50, 75, 90]))
            row += [hv.mean(), hv.std(), hv.max(), float(np.std(np.log1p(hv)))]
        else:
            row += [0] * 9
        row += [int(np.unique(loci_all[v0]).size)]
        out.append(row)
    return np.array(out, dtype=np.float32)


def card_features(P: np.ndarray, tokens: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Combined cardinality features = donor-prob profile (which donors) +
    MAC/height (how many) — the two complement: probs disambiguate identity,
    MAC supplies the physical count signal probs saturate on at high NOC."""
    return np.hstack([_card_feats(P), mac_feats(tokens, mask)])


def rank_n_contrast(z: torch.Tensor, labels: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    """Rank-N-Contrast (Zha et al. NeurIPS 2023, arXiv 2210.01189) — ORDINAL contrast for NOC.
    Orders the projected features so feature-distance matches label-distance ranking: for anchor i
    and positive j, the negatives are every sample k AT LEAST as far in label space as j
    (|y_i−y_k| ≥ |y_i−y_j|). Pulls NOC1..5 onto a monotone 1→5 manifold (a NOC1–NOC5 pair pushed
    farther than NOC4–NOC5), unlike unordered SupCon. Features L2-normalized; sim = −euclidean/τ.
    Operates on the DECOUPLED projection (model.proj_noc) so the count geometry never touches the
    ID readout (§4b-C valve 2/3). CAVEAT: features are L2-normalised so euclidean ∈ [0,2] → tau must
    be SMALL (≈0.1–0.5); tau=2 squashes sim into [-1,0] and the loss cannot descend (F19 — manipulation
    check failed at tau=2). Verify with probe_noc_structure.py before reading any downstream result."""
    z = torch.nn.functional.normalize(z, dim=1)
    sim = -torch.cdist(z, z) / tau                                   # (B,B) higher = closer
    yd = (labels.view(-1, 1) - labels.view(1, -1)).abs().float()     # (B,B) label distance
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    # neg[i,j,k] = (|y_i−y_k| ≥ |y_i−y_j|) and k ≠ i  → the denominator set for (anchor i, positive j)
    neg = (yd.unsqueeze(1) >= yd.unsqueeze(2)) & (~eye).unsqueeze(1)
    masked = sim.unsqueeze(1).masked_fill(~neg, float("-inf"))       # (B,B,B)
    denom = torch.logsumexp(masked, dim=2)                           # (B,B): logsumexp_k over negatives
    logprob = sim - denom                                            # (B,B) for (i, positive j)
    return -(logprob[~eye]).mean()                                   # mean over i, j≠i


def pcgrad_backward(main_loss, aux_loss, params):
    """PCGrad-style gradient surgery (Yu et al. 2020, arXiv 2001.06782) — §4b-C valve 4. NOTE: this
    is the ASYMMETRIC 'protect-main' variant, NOT vanilla (symmetric) PCGrad: we project ONLY the
    auxiliary (Rank-N-Contrast) gradient, never the main one — deliberate, since the ID task must
    not sacrifice its gradient to help the count contrast. If g_aux conflicts with g_main
    (cosine < 0), remove g_aux's component along g_main, then set .grad = g_main + g_aux_proj.
    Two grad passes; params where only one loss has a gradient project as a no-op (so the shared
    encoder is the only place surgery bites). Writes p.grad directly (call before optimizer.step)."""
    params = [p for p in params if p.requires_grad]
    g_main = torch.autograd.grad(main_loss, params, retain_graph=True, allow_unused=True)
    g_aux  = torch.autograd.grad(aux_loss,  params, retain_graph=False, allow_unused=True)
    flat = lambda gs: torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1)
                                 for g, p in zip(gs, params)])
    fm, fa = flat(g_main), flat(g_aux)
    dot = torch.dot(fm, fa)
    if dot < 0:
        fa = fa - (dot / (torch.dot(fm, fm) + 1e-12)) * fm          # remove conflicting component
    total = fm + fa
    idx = 0
    for p in params:
        n = p.numel()
        p.grad = total[idx:idx + n].view_as(p).detach().clone()
        idx += n


def and_mask_backward(env_losses, params, base_grad=None, tau: float = 0.0):
    """ILC AND-mask (Parascandolo et al. 2020, "Learning explanations that are hard to vary",
    arXiv 2009.00329) — Increment 6 root lever. Keep ONLY the gradient components on which the
    environments AGREE in sign; zero the rest. Here environments = {low-NOC, high-NOC} so the cls
    features that survive are the ones whose update direction helps BOTH easy (low-NOC, generalizing)
    and hard (high-NOC, combo-memorizing) strata — attacking the combo-overfit mechanism without
    IRM's penalty (which broke the aux F1, F23/F19). UNLIKE IRM this never adds a penalty term to the
    loss → aux/φ/attr heads are untouched.

    env_losses : list of per-environment scalar cls losses (≥1; if <2 we fall back to their mean).
    params     : model params over which to mask (shared encoder + cls head).
    base_grad  : optional dict{p: tensor} of already-computed grads from the OTHER losses (reject/noc/
                 aux); the masked cls grad is ADDED on top. If None, p.grad is set from the mask only.
    tau        : agreement threshold in [0,1]; component passes if |Σ sign| >= tau·E (0 = strict
                 consensus i.e. all same sign for E=2). Writes p.grad directly; call before step()."""
    params = [p for p in params if p.requires_grad]
    if len(env_losses) < 2:                                   # degenerate batch (one env) → plain grad
        env_losses = [sum(env_losses) / max(1, len(env_losses))]
    grads = [torch.autograd.grad(L, params, retain_graph=True, allow_unused=True) for L in env_losses]
    E = len(env_losses)
    for i, p in enumerate(params):
        gs = torch.stack([(g[i] if g[i] is not None else torch.zeros_like(p)) for g in grads])  # (E,*)
        mean_g = gs.mean(0)
        if E >= 2:
            agree = gs.sign().sum(0).abs() >= max(1.0, tau * E)   # consensus mask
            masked = mean_g * agree.to(mean_g.dtype)
        else:
            masked = mean_g
        if base_grad is not None and p in base_grad and base_grad[p] is not None:
            p.grad = (base_grad[p] + masked).detach().clone()
        else:
            p.grad = masked.detach().clone()


def vicreg_donor_regs(H: torch.Tensor, attr: torch.Tensor, n_classes: int = 45,
                      gamma: float = 1.0, eps: float = 1e-4):
    """Increment 8 (F32) — VICReg (Bardes, Ponce & LeCun, ICLR 2022, arXiv 2105.04906) applied VERBATIM
    to the per-(sample,donor) MEAN of the encoded set H. Z = the batch of donor reps, shape (K, d).
      variance  v(Z) = (1/d) Σ_j max(0, γ − std(z_j))           [anti-collapse → PRESERVES identity,
                       std(z_j)=√(Var(z_j)+ε), γ=1                 N1-SAFE: no per-sample subtraction]
      covariance c(Z) = (1/d) Σ_{i≠j} C(Z)_{ij}²                 [decorrelate dims → removes the redundant
                       C(Z)=(1/(K−1)) Zc^T Zc                       combo-shared carrier = the 'de-smooth']
      invariance s     = mean over donors of E‖z − z̄_donor‖²     [pull the SAME donor's rep together across
                       (multi-view generalization: the 'views'      the combos it appears in = combo-invariance;
                        are a donor's reps in different combos)      N1 trivially consistent]
    Paper coefficients λ(inv)=25, μ(var)=25, ν(cov)=1 are applied by the caller. attr=-1 (real splits)
    are skipped. Returns (var, cov, inv) scalar tensors (0 when degenerate)."""
    B, N, d = H.shape
    dev = H.device
    valid = attr >= 0
    zero = H.sum() * 0.0
    if valid.sum() < 2:
        return zero, zero, zero
    samp = torch.arange(B, device=dev).view(B, 1).expand(B, N)
    key = samp * n_classes + attr.clamp(min=0)
    vH, vkey = H[valid], key[valid]
    uk, inv = torch.unique(vkey, return_inverse=True)           # (K,), (P,)
    sumH = torch.zeros(len(uk), d, device=dev).index_add_(0, inv, vH)
    cnt  = torch.zeros(len(uk), device=dev).index_add_(0, inv, torch.ones(len(vkey), device=dev))
    Z = sumH / cnt.clamp(min=1).unsqueeze(1)                    # (K, d) donor reps
    K = Z.size(0)
    # variance (hinge on per-dim std)
    std = torch.sqrt(Z.var(dim=0, unbiased=False) + eps)        # (d,)
    var = torch.relu(gamma - std).mean()
    # covariance (off-diagonal of the centered cov matrix, squared, /d)
    if K > 2:
        Zc = Z - Z.mean(dim=0, keepdim=True)
        cov = (Zc.T @ Zc) / (K - 1)                             # (d,d)
        off = cov - torch.diag(torch.diagonal(cov))
        covariance = off.pow(2).sum() / d
    else:
        covariance = zero
    # invariance: same-donor reps across the batch's combos pulled to their donor mean
    don_of = uk % n_classes
    inv_terms = []
    for dd in don_of.unique():
        m = don_of == dd
        if m.sum() >= 2:
            r = Z[m]
            inv_terms.append(((r - r.mean(0, keepdim=True)) ** 2).sum(1).mean())
    invariance = torch.stack(inv_terms).mean() if inv_terms else zero
    return var, covariance, invariance


def irm_penalty(logits: torch.Tensor, y: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    """IRMv1 penalty (Arjovsky et al. 2019, arXiv 1907.02893) — §7 shortcut suppression. Treat NOC
    strata as ENVIRONMENTS; penalise per-environment sensitivity of the loss to a dummy scale w=1.0
    on the logits → drives features whose optimal classifier is INVARIANT across environments,
    suppressing the spurious combo/NOC co-occurrence shortcut and keeping the causal allele→donor
    signal. penalty = Σ_e ||∇_w BCE(w·logits_e, y_e)||² ."""
    scale = torch.ones((), device=logits.device, requires_grad=True)
    pen = logits.new_zeros(())
    for e in torch.unique(env):
        m = (env == e)
        if m.sum() == 0:
            continue
        loss_e = F.binary_cross_entropy_with_logits(logits[m] * scale, y[m].float())
        g = torch.autograd.grad(loss_e, scale, create_graph=True)[0]
        pen = pen + g.pow(2)
    return pen


def build_pgnoc_refs(xflat_train: np.ndarray, y_train: np.ndarray, noc_train: np.ndarray) -> np.ndarray:
    """Per-donor reference template G_d = mean GLOBAL-relative-RFU over donor d's
    single-source (NOC=1) profiles. 'global' ref chosen over consensus/per-locus:
    sharper (PG-style) refs win IN-DISTRIBUTION but OVERFIT in-silico -> worse real
    transfer (reference_variants.py); the crude global ref is domain-robust."""
    rfu = np.expm1(xflat_train.astype(np.float64)); ss = noc_train == 1
    n_cls = y_train.shape[1]; G = np.zeros((n_cls, rfu.shape[1]))
    for d in range(n_cls):
        idx = np.where(ss & (y_train[:, d] == 1))[0]
        if len(idx):
            rel = rfu[idx] / (rfu[idx].sum(1, keepdims=True) + 1e-12)
            G[d] = rel.mean(0)
    return G


def pgnoc_cost_features(xflat: np.ndarray, probs: np.ndarray, G: np.ndarray,
                        pool: int = 8, alpha: float = 0.5, kmax: int = 6) -> np.ndarray:
    """Continuous-model (NOCIt/EuroForMix-style) deconvolution cost curve per sample,
    used as EXTRA features for the count classifier. Greedy gamma-weighted NNLS fit of
    observed relative heights against top-`pool` candidate donor references; cost_k =
    weighted residual after k donors. Features = [cost_1..kmax, marginal drops]."""
    from scipy.optimize import nnls
    H = np.expm1(xflat.astype(np.float64)); N = len(H); feats = np.zeros((N, 2 * kmax - 1))
    for i in range(N):
        h = H[i]; hrel = h / (h.sum() + 1e-12)
        w = 1.0 / np.power(hrel + 1e-4, alpha); b = hrel * w
        cand = list(np.argsort(probs[i])[::-1][:pool]); chosen, cost = [], []
        for k in range(kmax):
            best_c, best_d = np.inf, None
            for d in cand:
                A = (G[chosen + [d]].T) * w[:, None]
                phi, _ = nnls(A, b); c = float(np.linalg.norm(b - A @ phi))
                if c < best_c: best_c, best_d = c, d
            chosen.append(best_d); cand.remove(best_d); cost.append(best_c)
        cost = np.array(cost)
        feats[i] = np.concatenate([cost, cost[:-1] - cost[1:]])
    return feats.astype(np.float32)


def two_stage_cardinality(feat_tr, noc_tr, feat_va, noc_va, feat_te, stage2_reg=None):
    """Two-stage NOC decoder that resolves the prior-vs-boundary tension:
      stage1 NOC1-vs-multi  — fit on real val (correct ~85%-NOC1 prior, easy task)
      stage2 count 2..5     — fit on combo-diverse in-silico train (multi-rich,
                              learns the k=4/5 boundary the tiny real val lacks)
    Inputs are card_features() [prob-profile + MAC] optionally hstacked with
    pgnoc_cost_features(). stage2_reg overrides stage2 XGB kwargs (deployable uses
    max_depth=4, min_child_weight=10 to avoid overfitting the pgNOC features).
    Returns k in 1..5. (two-stage prob+MAC = 0.950; +pgNOC global ref = 0.954.)"""
    import xgboost as xgb
    noc_tr = np.clip(noc_tr, 1, 5); noc_va = np.clip(noc_va, 1, 5)
    s1 = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42)
    s1.fit(feat_va, (noc_va >= 2).astype(int))
    multi = s1.predict(feat_te).astype(bool)
    mb = noc_tr >= 2
    s2kw = dict(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, eval_metric="mlogloss", random_state=42)
    if stage2_reg: s2kw.update(stage2_reg)
    s2 = xgb.XGBClassifier(**s2kw)
    s2.fit(feat_tr[mb], noc_tr[mb] - 2)            # labels 2..5 -> 0..3
    k = np.ones(len(feat_te), int)
    if multi.any():
        k[multi] = s2.predict(feat_te[multi]) + 2
    return k


# ── Datasets ───────────────────────────────────────────────────────────────

class ClosedSetDataset(Dataset):
    """Closed-set samples. reject_label = 0 for all."""

    def __init__(self, split: str, tok_prefix: str = "tokens", rep: int = 1):
        # Replicate-augmented (richer data): load the pooled _rep{R} peak arrays; y/noc/phi are
        # per-mixture (unchanged) so they load WITHOUT the suffix. rep<=1 -> exact prior behaviour.
        rs = f"_rep{rep}" if rep and rep > 1 else ""
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"{tok_prefix}_{split}{rs}.npy").astype(np.float32))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / f"mask_{split}{rs}.npy"))
        self.y      = torch.from_numpy(np.load(DATA_DIR / f"y_{split}_set.npy"))
        self.noc    = torch.from_numpy(np.load(DATA_DIR / f"noc_{split}.npy").astype(np.int64))
        # Increment 2 §5 — privileged labels (in-silico only): per-peak donor attribution + phi.
        # Sentinels (attr=-1 -> ignore_index, phi=0) when absent (real splits) so the aux loss
        # naturally contributes nothing on data without provenance.
        N, S = self.tokens.shape[0], self.tokens.shape[1]
        ap = DATA_DIR / f"attr_{split}{rs}.npy"; pp = DATA_DIR / f"phi_{split}.npy"
        if ap.exists() and pp.exists():
            self.attr = torch.from_numpy(np.load(ap).astype(np.int64))
            self.phi  = torch.from_numpy(np.load(pp).astype(np.float32))
        else:
            self.attr = torch.full((N, S), -1, dtype=torch.int64)
            self.phi  = torch.zeros((N, 45), dtype=torch.float32)
        # Inc-LUPI privileged PHYSICAL labels (synthetic-only). Sentinels when absent (real val/test):
        # beta=-1 (masks the row in the loss), mu/var/dropin=0. return_lupi gates the extra __getitem__ fields.
        self.return_lupi = False
        mp = DATA_DIR / f"mu_{split}{rs}.npy"
        if mp.exists():
            self.mu     = torch.from_numpy(np.load(mp).astype(np.float32))
            self.var    = torch.from_numpy(np.load(DATA_DIR / f"var_{split}{rs}.npy").astype(np.float32))
            self.dropin = torch.from_numpy(np.load(DATA_DIR / f"dropin_{split}{rs}.npy").astype(np.float32))
            self.beta   = torch.from_numpy(np.load(DATA_DIR / f"beta_{split}{rs}.npy").astype(np.float32))
        else:
            self.mu     = torch.zeros((N, S), dtype=torch.float32)
            self.var    = torch.zeros((N, S), dtype=torch.float32)
            self.dropin = torch.zeros((N, S), dtype=torch.float32)
            self.beta   = torch.full((N,), -1.0, dtype=torch.float32)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        base = (self.tokens[i], self.mask[i], self.y[i], self.noc[i], self.attr[i], self.phi[i])
        if self.return_lupi:
            base = base + (self.mu[i], self.var[i], self.dropin[i], self.beta[i])
        return base


class OpenSetDataset(Dataset):
    """Open-set samples (has unknown contributor). reject_label = 1."""

    def __init__(self, tok_prefix: str = "tokens"):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"{tok_prefix}_open.npy").astype(np.float32))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / "mask_open.npy"))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i]


# ── Loss helpers ───────────────────────────────────────────────────────────

def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = y.sum(0).clip(min=1)
    neg = (1 - y).sum(0).clip(min=1)
    return torch.tensor(neg / pos, dtype=torch.float32)


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label (Ben-Baruch 2020; used by Query2Label).
    Down-weights easy negatives (gamma_neg) and clips low-prob negatives,
    focusing learning on hard positives — better calibrated than BCE+pos_weight
    for heavily imbalanced multi-label (here ~3% positive rate).
    """
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, logits, targets, weight=None):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        # Asymmetric focusing
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss *= (1 - pt) ** gamma
        if weight is not None:                 # Inc6 minor-weight: per-(sample,donor) cost
            loss = loss * weight
        return -loss.mean()


# ── Increment 9 (F33) decoder-under-read losses: stop ASL(gamma_neg=4) from driving faint positives
#    to ~0 (probe_decoder_mech: true-minor score .84->.14). Each is a published long-tail multi-label loss.
class LDAMLoss(nn.Module):
    """A1 — Label-Distribution-Aware Margin (Cao 2019) + Deferred Re-Weighting (Cui 2019), multi-label form.
    Per-class margin m_c ∝ n_c^-1/4 is SUBTRACTED from the positive logit so rare/faint positives must
    clear a larger functional margin; class-balanced reweighting switches on after `drw_epoch`."""
    def __init__(self, cls_num, max_m=0.5, drw_epoch=20, beta=0.999):
        super().__init__()
        nc = np.clip(np.asarray(cls_num, float), 1, None)
        m = 1.0 / np.sqrt(np.sqrt(nc)); m = m * (max_m / m.max())
        eff = 1.0 - np.power(beta, nc); w = (1 - beta) / eff; w = w / w.mean()
        self.register_buffer("m", torch.tensor(m, dtype=torch.float32))
        self.register_buffer("cb_w", torch.tensor(w, dtype=torch.float32))
        self.drw_epoch = int(drw_epoch); self.epoch = 0
    def set_epoch(self, e): self.epoch = int(e)
    def forward(self, logits, targets, weight=None):
        z = logits - self.m.unsqueeze(0) * targets        # margin only on the positive state
        loss = F.binary_cross_entropy_with_logits(z, targets, reduction="none")
        if self.epoch >= self.drw_epoch:
            loss = loss * self.cb_w.unsqueeze(0)
        if weight is not None: loss = loss * weight
        return loss.mean()


class DistributionBalancedLoss(nn.Module):
    """A2 — Distribution-Balanced Loss (Wu 2020, ECCV): rebalance rare-positive weight + Negative-Tolerant
    Regularization (per-class bias shifts negative logits down + a lam<1 scale on the negative term) so the
    abundant negatives stop over-suppressing rare positives. Contained per-class approximation."""
    def __init__(self, cls_num, n_train, mu=0.3, lam=0.1, kappa=0.05, alpha=0.1):
        super().__init__()
        f = np.clip(np.asarray(cls_num, float), 1, None) / max(1, int(n_train))   # P(y=1) per class
        rw = alpha + 1.0 / (1.0 + np.exp(-mu * (1.0 / np.clip(f, 1e-6, 1) - 1.0))) # up-weight rare
        self.register_buffer("rw", torch.tensor(rw / rw.mean(), dtype=torch.float32))
        self.register_buffer("nt_bias", torch.tensor(kappa * np.log(1.0 / np.clip(f, 1e-6, 1) - 1.0 + 1e-6),
                                                      dtype=torch.float32))
        self.lam = float(lam)
    def forward(self, logits, targets, weight=None):
        z = logits + self.nt_bias.unsqueeze(0)            # negative-tolerant logit bias
        l_pos = F.binary_cross_entropy_with_logits(z, torch.ones_like(targets), reduction="none")
        l_neg = F.binary_cross_entropy_with_logits(z, torch.zeros_like(targets), reduction="none") * self.lam
        loss = (targets * l_pos + (1 - targets) * l_neg) * self.rw.unsqueeze(0)
        if weight is not None: loss = loss * weight
        return loss.mean()


class BalancedAsymmetricLoss(AsymmetricLoss):
    """A3 — Balanced Asymmetric Loss: ASL (keeps negative down-weight/clip) + small POSITIVE focusing and
    per-class positive up-weighting so rare/uncertain positives get gradient instead of being suppressed."""
    def __init__(self, cls_num, gamma_neg=4.0, gamma_pos=1.0, clip=0.05):
        super().__init__(gamma_neg=gamma_neg, gamma_pos=gamma_pos, clip=clip)
        w = 1.0 / np.sqrt(np.clip(np.asarray(cls_num, float), 1, None)); w = w / w.mean()
        self.register_buffer("pos_w", torch.tensor(w, dtype=torch.float32))
    def forward(self, logits, targets, weight=None):
        pw = self.pos_w.unsqueeze(0) * targets + (1 - targets)
        w = pw if weight is None else pw * weight
        return super().forward(logits, targets, weight=w)


def build_cls_loss(name, y_train_np, cfg):
    """Increment-9 decoder loss selector. name in {asl,ldam,dbloss,bal}; falls back to caller for asl/bce."""
    cls_num = y_train_np.sum(0); n_train = len(y_train_np)
    if name == "ldam":
        return LDAMLoss(cls_num, drw_epoch=int(cfg.get("ldam_drw_epoch", 20)))
    if name == "dbloss":
        return DistributionBalancedLoss(cls_num, n_train)
    if name == "bal":
        return BalancedAsymmetricLoss(cls_num, gamma_neg=cfg.get("asl_gamma_neg", 4.0),
                                      gamma_pos=cfg.get("bal_gamma_pos", 1.0), clip=cfg.get("asl_clip", 0.05))
    return None


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_closed(model, loader, threshold=0.5, phi_inject_fn=None):
    """Returns (macro_f1, y_true, y_pred, noc_true)."""
    model.eval()
    all_true, all_pred, all_noc = [], [], []
    for tokens, mask, y, noc, *rest in loader:
        em = phi_inject_fn(rest).to(DEVICE) if phi_inject_fn else None
        out = model(tokens.to(DEVICE), mask.to(DEVICE), em_phi=em)
        probs = torch.sigmoid(out["logits_cls"]).cpu().numpy()
        all_pred.append((probs >= threshold).astype(np.float32))
        all_true.append(y.numpy())
        all_noc.append(noc.numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    noc_all = np.concatenate(all_noc)
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return mf1, y_true, y_pred, noc_all


@torch.no_grad()
def evaluate_oracle_em(model, loader, phi_inject_fn=None):
    """Oracle top-k Exact Match (k = true NOC) on val — selection metric.
    More stable than macro-F1 over 45 sparse classes on novel combos, and matches
    the headline eval. Used for scheduler/early-stop/best-model selection."""
    model.eval()
    all_probs, all_true, all_noc = [], [], []
    for tokens, mask, y, noc, *rest in loader:
        em = phi_inject_fn(rest).to(DEVICE) if phi_inject_fn else None
        out = model(tokens.to(DEVICE), mask.to(DEVICE), em_phi=em)
        all_probs.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
        all_true.append(y.numpy()); all_noc.append(noc.numpy())
    probs = np.concatenate(all_probs); y_true = np.concatenate(all_true); noc = np.concatenate(all_noc)
    yp = np.zeros_like(probs, dtype=int); rec = np.zeros(len(probs))
    for i in range(len(probs)):
        k = int(max(1, min(5, noc[i])))
        top = np.argsort(probs[i])[::-1][:k]; yp[i, top] = 1
        rec[i] = y_true[i][top].sum() / k                  # oracle recall@k
    em = (y_true == yp).all(1); nocc = np.clip(noc, 1, 5)
    strata = [rec[nocc == j].mean() for j in range(1, 6) if (nocc == j).any()]
    return float(em.mean()), float(np.mean(strata))        # (overall EM, macro-over-NOC recall@k)


@torch.no_grad()
def evaluate_per_noc_oracle(model, loader, phi_inject_fn=None):
    """Per-NOC oracle EM (full set correct at top-true-k). EARLY_ABORT guard/trajectory signal only —
    matches measure_insilico_oracle's definition so the in-train check agrees with the headline judge."""
    model.eval()
    P, Y, N = [], [], []
    for tokens, mask, y, noc, *rest in loader:
        em = phi_inject_fn(rest).to(DEVICE) if phi_inject_fn else None
        P.append(torch.sigmoid(model(tokens.to(DEVICE), mask.to(DEVICE), em_phi=em)["logits_cls"]).cpu().numpy())
        Y.append(y.numpy()); N.append(noc.numpy())
    P = np.concatenate(P); Y = np.concatenate(Y); N = np.clip(np.concatenate(N), 1, 5)
    out = {}
    for j in range(1, 6):
        m = np.where(N == j)[0]
        if not len(m):
            continue
        ems = []
        for i in m:
            top = np.argsort(P[i])[::-1][:j]; pred = np.zeros(P.shape[1], int); pred[top] = 1
            ems.append(bool((pred == Y[i]).all()))
        out[j] = float(np.mean(ems))
    return out


def full_report(y_true, y_pred, noc_true, title):
    from sklearn.metrics import hamming_loss, precision_score, recall_score
    mf1  = f1_score(y_true, y_pred, average="macro",  zero_division=0)
    mif1 = f1_score(y_true, y_pred, average="micro",  zero_division=0)
    hl   = hamming_loss(y_true, y_pred)
    em   = np.all(y_true == y_pred, axis=1).mean()
    mp   = precision_score(y_true, y_pred, average="macro", zero_division=0)
    mr   = recall_score(y_true, y_pred, average="macro",    zero_division=0)
    jac  = (
        (y_true.astype(bool) & y_pred.astype(bool)).sum(1) /
        (y_true.astype(bool) | y_pred.astype(bool)).sum(1).clip(min=1)
    ).mean()

    print(f"\n{'='*60}")
    print(f"  {title}")
    print("="*60)
    print(f"  Macro F1     : {mf1:.4f}")
    print(f"  Micro F1     : {mif1:.4f}")
    print(f"  Hamming Loss : {hl:.4f}")
    print(f"  Exact Match  : {em:.4f}")
    print(f"  Macro Pre    : {mp:.4f}")
    print(f"  Macro Rec    : {mr:.4f}")
    print(f"  Jaccard (avg): {jac:.4f}")

    print("\n  -- Exact match by NOC " + "-"*36)
    exact = np.all(y_true == y_pred, axis=1)
    per_noc = {}
    for noc in sorted(np.unique(noc_true)):
        m = noc_true == noc
        em_n = float(exact[m].mean()) if m.sum() else float("nan")
        f1_n = float(f1_score(y_true[m], y_pred[m], average="macro", zero_division=0)) if m.sum() else float("nan")
        per_noc[int(noc)] = {"em": round(em_n, 4), "n": int(m.sum())}
        print(f"    NOC={noc}: EM={em_n:.3f}  MacroF1={f1_n:.3f}  (n={m.sum()})")

    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    print(f"\n  Zero-F1 classes: {int((per_f1 == 0).sum())} / {len(per_f1)}")
    print("="*60)

    return {
        "macro_f1":    float(mf1),
        "micro_f1":    float(mif1),
        "hamming":     float(hl),
        "exact_match": float(em),
        "precision":   float(mp),
        "recall":      float(mr),
        "jaccard":     float(jac),
        "per_noc":     per_noc,
        "zero_f1_classes": int((per_f1 == 0).sum()),
    }


# ── Training ───────────────────────────────────────────────────────────────

def make_harder(tokens, mask, attr, y, phi, factor):
    """Lever C (Inc13) on-the-fly difficulty step: scale the FAINTEST present minor's peak heights
    by `factor`<1 (per sample) -> a fainter version of the SAME mixture (same donor set/label). The
    three HEIGHT-DERIVED token features are recomputed exactly (log_h col2, Hb col3, glob_rel col7);
    SR/rank (col4/5) left stale = tolerable counterfactual noise (NCM). Used to teach difficulty-
    invariance "use the easy version to read the hard one", curriculum-annealed."""
    B, N, _ = tokens.shape
    t = tokens.clone()
    h = torch.expm1(t[:, :, 2])                                          # (B,N) RFU
    mk = mask.bool().float()
    present = (y > 0.5)
    phi_masked = phi.masked_fill(~present, float("inf"))
    c = phi_masked.argmin(1)                                            # faintest present donor
    sel = (attr == c.unsqueeze(1)) & mask.bool()                       # its peaks
    h = h * torch.where(sel, torch.full_like(h, factor), torch.ones_like(h))
    hmax = (h * mk).amax(1, keepdim=True).clamp(min=1e-6)
    glob = (h / hmax) * mk                                             # glob_rel
    loc = t[:, :, 0].long().clamp(0, 23)
    locsum = torch.zeros(B, 24, device=tokens.device)
    locsum.scatter_add_(1, loc, h * mk)
    hb = (h / locsum.gather(1, loc).clamp(min=1e-6)) * mk             # Hb
    t[:, :, 2] = torch.log1p(h) * mk
    t[:, :, 3] = hb
    t[:, :, 7] = glob
    return t


def subtract_height(tokens, mask, mult):
    """Lever A-v2 (Inc14) additive-SUBTRACTION counterfactual, grounded in the EuroForMix continuous
    model: a peak height is the SUM of each contributor's gamma deposit (t·Mx·copy). To 'remove' a
    contributor c we SUBTRACT its deposit — multiply each peak's height by `mult`=(1-frac_c) where
    frac_c = c's share of that peak. A c-PRIVATE allele -> mult~0 (drops out); a SHARED allele -> only
    partially reduced (the kept donor's deposit REMAINS). This is the forensically-correct counterfactual
    that A-v1 violated by DELETING whole peaks (which destroyed 41% of kept donors' shared evidence).
    Recompute the three height-derived features exactly (log_h col2, Hb col3, glob_rel col7)."""
    B, N, _ = tokens.shape
    t = tokens.clone()
    mk = mask.bool().float()
    h = torch.expm1(t[:, :, 2]) * mult                                  # subtracted RFU
    hmax = (h * mk).amax(1, keepdim=True).clamp(min=1e-6)
    glob = (h / hmax) * mk
    loc = t[:, :, 0].long().clamp(0, 23)
    locsum = torch.zeros(B, 24, device=tokens.device)
    locsum.scatter_add_(1, loc, h * mk)
    hb = (h / locsum.gather(1, loc).clamp(min=1e-6)) * mk
    t[:, :, 2] = torch.log1p(h) * mk
    t[:, :, 3] = hb
    t[:, :, 7] = glob
    return t


def _mmd2_rbf(a, b):
    """Unbiased-ish RBF MMD^2 between two sample sets a,b (n_a,d)/(n_b,d), median-heuristic bandwidth.
    Distributional (NOT pointwise) — the CIP/Veitch operationalization that avoids the cosine->1 collapse."""
    import torch as _t
    cat = _t.cat([a, b], 0)
    with _t.no_grad():
        d2 = _t.cdist(cat, cat).pow(2)
        med = d2[d2 > 0].median()
        sigma2 = (med * 0.5).clamp(min=1e-3)
    def K(x, y):
        return _t.exp(-_t.cdist(x, y).pow(2) / (2 * sigma2))
    return K(a, a).mean() + K(b, b).mean() - 2 * K(a, b).mean()


def cond_slot_mmd(id_main, id_cf, keep, min_n=4):
    """Lever A-v2 invariance penalty: conditional MMD on the per-donor IDENTITY sub-rep, CONDITIONED on
    donor identity (grouped by donor slot k = CIP's 'condition on the label'). For each kept donor slot,
    the distribution of its identity embeddings must be invariant to which co-donor was subtracted. Per-slot
    grouping (not a global match) keeps it conditional; matching DISTRIBUTIONS (not pairs) + the cls task
    anchoring last_reps prevents the trivial-collapse failure A-v1 hit."""
    import torch as _t
    total = id_main.new_zeros(()); nk = 0
    for k in range(id_main.size(1)):
        m = keep[:, k]
        if int(m.sum()) < min_n:
            continue
        total = total + _mmd2_rbf(id_main[m, k], id_cf[m, k]); nk += 1
    return total / max(nk, 1)


def train(cfg: dict):
    subdir = cfg.get("out_subdir", "set_transformer")
    results_dir = ROOT / "results" / subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    cfg["seed"] = seed                      # persist into saved metrics.json config
    gen = set_seed(seed)
    print(f"seed = {seed} (reproducible: weight init + shuffle + dropout)")

    # Datasets
    n_tok = cfg.get("n_token_feats", 3)
    tok_prefix = f"tokens{n_tok}" if n_tok > 3 else "tokens"
    rep = int(cfg.get("replicates", 1))   # richer-data lever: load pooled _rep{R} peak arrays (rep<=1 = unchanged)
    if rep > 1:
        print(f"REPLICATES on: pooling R={rep} amplifications per mixture (peak arrays suffixed _rep{rep}; open-set single-profile)")
    train_ds = ClosedSetDataset("train", tok_prefix, rep)
    train_ds.return_lupi = bool(cfg.get("lupi_phys", False))   # Inc-LUPI: yield physical labels in train batches
    val_ds   = ClosedSetDataset("val", tok_prefix, rep)
    test_ds  = ClosedSetDataset("test", tok_prefix, rep)
    open_ds  = OpenSetDataset(tok_prefix)

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    val_loader  = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    # SELECTION set = combo-disjoint balanced in-silico DEV if present, else real val.
    _dev_sfx = f"_rep{rep}" if rep > 1 else ""
    if (DATA_DIR / f"{tok_prefix}_dev{_dev_sfx}.npy").exists():
        sel_loader = DataLoader(ClosedSetDataset("dev", tok_prefix, rep), batch_size=256, shuffle=False)
        print(f"selection set = in-silico DEV ({len(sel_loader.dataset)} samples)")
    else:
        sel_loader = val_loader; print("selection set = real val (no dev split found)")

    # Open-set infinite sampler for reject head (seeded for reproducibility)
    open_iter = iter(DataLoader(
        open_ds,
        sampler=RandomSampler(open_ds, replacement=True,
                              num_samples=len(train_ds) * cfg["epochs"], generator=gen),
        batch_size=max(1, int(cfg["batch_size"] * cfg.get("open_ratio", 0.25))),
        num_workers=0,
    ))

    # Inc3 A / Inc10 / Inc14: reference-genotype tokens (precomputed by build_donor_geno.py, shipped in bundle).
    # Inc14 B-v2/A-v2 also need genotypes — to build the peak->donor OWNERSHIP lookup (co-membership target
    # for the multi-label head, and the additive-subtraction fractions for the counterfactual).
    need_geno = (cfg.get("geno_query", False) or cfg.get("ref_match", False)
                 or cfg.get("ml_attr", False) or cfg.get("add_invar", False) or cfg.get("recon", False)
                 or cfg.get("add_recon", False) or cfg.get("soft_geno_attr", False)
                 or cfg.get("feas_filter", False) or cfg.get("soft_attr_label", False)
                 or cfg.get("set_of_set", False) or cfg.get("em_phi_feature", False)
                 or cfg.get("noise_gate", False)
                 or cfg.get("cls_decoder", "pooled") == "aslot")
    donor_geno = donor_geno_mask = None
    owner_lut = None
    cn_lut = None          # copy-number LUT for soft attr labels (accumulated: het->1, homo->2)
    ALLELE_OFF = 30                                  # round(allele*10)+OFF -> non-negative bin index
    if need_geno:
        gp = DATA_DIR / "donor_geno.npy"
        for cand in (DATA_DIR / "donor_geno.npy", ROOT / "data" / "donor_geno.npy", ROOT / "donor_geno.npy"):
            if cand.exists():
                gp = cand; break
        donor_geno = torch.from_numpy(np.load(gp).astype(np.float32))
        donor_geno_mask = torch.from_numpy(np.load(gp.parent / "donor_geno_mask.npy"))
        print(f"reference genotypes loaded {tuple(donor_geno.shape)} from {gp.parent} "
              f"(geno_query={cfg.get('geno_query', False)} ref_match={cfg.get('ref_match', False)} "
              f"ml_attr={cfg.get('ml_attr', False)} add_invar={cfg.get('add_invar', False)})")
        if cfg.get("ml_attr", False) or cfg.get("add_invar", False) or cfg.get("recon", False) or cfg.get("add_recon", False) or cfg.get("soft_geno_attr", False) or cfg.get("feas_filter", False) or cfg.get("soft_attr_label", False) or cfg.get("set_of_set", False) or cfg.get("em_phi_feature", False) or cfg.get("noise_gate", False):
            # owner_lut[locus, allele_bin, donor] = 1 if donor's reference genotype carries that allele.
            n_cls = int(cfg.get("n_classes", 45)); LUT_W = 1024
            owner_lut = torch.zeros(24, LUT_W, n_cls)
            gg = donor_geno; gm = donor_geno_mask.bool()
            for c in range(min(n_cls, gg.size(0))):
                for j in range(gg.size(1)):
                    if gm[c, j]:
                        li = int(gg[c, j, 0]); ab = int(round(float(gg[c, j, 1]) * 10)) + ALLELE_OFF
                        if 0 <= li < 24 and 0 <= ab < LUT_W:
                            owner_lut[li, ab, c] = 1.0
            owner_lut = owner_lut.to(DEVICE)
            print(f"Inc14 owner_lut built {tuple(owner_lut.shape)} "
                  f"(mean owners/allele-bin among present = {owner_lut.sum(-1)[owner_lut.sum(-1)>0].mean():.2f})")
            # Separate copy-number LUT for soft attr labels: same loop but accumulate (het->1, homo->2).
            if cfg.get("soft_attr_label", False):
                cn_lut = torch.zeros(24, LUT_W, n_cls)
                for c in range(min(n_cls, gg.size(0))):
                    for j in range(gg.size(1)):
                        if gm[c, j]:
                            li = int(gg[c, j, 0]); ab = int(round(float(gg[c, j, 1]) * 10)) + ALLELE_OFF
                            if 0 <= li < 24 and 0 <= ab < LUT_W:
                                cn_lut[li, ab, c] += 1.0   # accumulate: assigns 2 for homozygous
                cn_lut = cn_lut.to(DEVICE)
                print(f"soft_attr_label cn_lut built {tuple(cn_lut.shape)} (max CN = {int(cn_lut.max())})")

    # Model
    model = SetTransformerMixture(
        n_loci     = cfg.get("n_loci", 24),
        d_locus    = cfg.get("d_locus", 16),
        d_model    = cfg.get("d_model", 128),
        n_heads    = cfg.get("n_heads", 4),
        n_isab     = cfg.get("n_isab", 2),
        m_inducing = cfg.get("m_inducing", 32),
        n_classes  = cfg.get("n_classes", 45),
        n_noc      = cfg.get("n_noc", 6),
        dropout    = cfg.get("dropout", 0.1),
        cls_decoder = cfg.get("cls_decoder", "pooled"),
        decoder_source = cfg.get("decoder_source", "encoded"),
        n_token_feats = n_tok,
        encoder = cfg.get("encoder", "isab"),
        dec_layers = cfg.get("dec_layers", 2),
        dec_aggr = cfg.get("dec_aggr", "sparsemax"),
        num_embed = cfg.get("num_embed", "raw"),
        n_freq = cfg.get("n_freq", 8),
        d_num_emb = cfg.get("d_num_emb", 8),
        periodic_sigma = cfg.get("periodic_sigma", 1.0),
        aux_heads = cfg.get("aux_heads", False),
        lupi_phys = cfg.get("lupi_phys", False),        # Inc-LUPI: privileged physical heads
        ml_attr = cfg.get("ml_attr", False),            # Inc14 B-v2: multi-label co-membership head
        noc_contrast = cfg.get("noc_contrast", False),
        noc_detach = (cfg.get("noc_contrast_mode", "shared") == "detach"),
        d_proj = cfg.get("d_proj", 64),
        sparse_attn = cfg.get("sparse_attn", False),
        geno_query = cfg.get("geno_query", False),
        donor_geno = donor_geno,
        donor_geno_mask = donor_geno_mask,
        donor_contrast = cfg.get("donor_contrast", False),
        noc_ord_head = cfg.get("noc_ord_head", False),
        noc_ord_detach = cfg.get("noc_ord_detach", False),
        noc_ord_replace = cfg.get("noc_ord_replace", False),
        vib = cfg.get("vib", False),
        mass_pool = cfg.get("mass_pool", False),
        vicreg = cfg.get("vicreg", False),
        vicreg_inv = cfg.get("vicreg_inv", False),
        attn_sink = int(cfg.get("attn_sink", 0)),       # Inc9 B4
        donor_recon = cfg.get("donor_recon", False),    # Inc9 B1
        query_denoise = cfg.get("query_denoise", False), # Inc9 A4
        qdn_noise = cfg.get("qdn_noise", 1.0),          # Inc9 A4
        ref_match = cfg.get("ref_match", False),         # Inc10
        ref_match_learn = cfg.get("ref_match_learn", False),  # Inc10 V2
        nc_attn = cfg.get("nc_attn", "none"),            # Inc11 (F35b): non-competitive sigmoid encoder attn
        nc_learnable_bias = cfg.get("nc_learnable_bias", False),
        phi_inject = cfg.get("phi_inject", False),         # Inc18 V1: inject phi into per-donor queries
        soft_geno_attr = cfg.get("soft_geno_attr", False), # Inc18 V2: private-hard / shared-soft attr at inference
        feas_filter = cfg.get("feas_filter", False),       # Inc18 V3: filter infeasible peaks before encoder
        set_of_set  = cfg.get("set_of_set", False),        # SoS: split private/shared sets before encoder
        owner_lut = owner_lut,                             # (24, LUT_W, C) or None
        # Adaptive slot (CoSA+GSANet+MESH+AdaSlot; cls_decoder="aslot")
        n_slot_iters = int(cfg.get("n_slot_iters", 3)),
        ot_eps       = float(cfg.get("ot_eps", 0.05)),
        ot_iters     = int(cfg.get("ot_iters", 5)),
        gumbel_temp  = float(cfg.get("gumbel_temp", 1.0)),
        gate_mass    = bool(cfg.get("gate_mass", False)),     # §3.2.1 mass-aware AdaSlot gate
        phi_gated    = bool(cfg.get("phi_gated", False)),     # §3.4 presence-gated phi + β-NLL
        noc_head_v2  = bool(cfg.get("noc_head_v2", False)),   # paper-proof CORN count head (mass+MAC+prob)
        em_phi_feature = bool(cfg.get("em_phi_feature", False)),  # internalize EM Mx deconvolution (LOP cls + Hill noc)
        noise_gate = bool(cfg.get("noise_gate", False)),          # SOFT supervised per-peak stutter/drop-in reliability gate
        noise_s_floor = float(cfg.get("noise_s_floor", 0.05)),
        # EBM-SPEN joint (cls_decoder="spen")
        n_inf_steps        = int(cfg.get("n_inf_steps", 10)),
        inf_lr             = float(cfg.get("inf_lr", 0.5)),
        spen_global_hidden = int(cfg.get("spen_global_hidden", 128)),
    ).to(DEVICE)
    if cfg.get("ref_match", False):
        print(f"Inc10 ref_match ON  (learn_weights={cfg.get('ref_match_learn', False)}) "
              f"— discriminability-weighted reference-allele matching added to per-donor logit")
    if n_tok > 3:   # enriched tokens: set per-feature standardization from train valid peaks
        _tk = train_ds.tokens.numpy(); _mk = train_ds.mask.numpy().astype(bool)
        _num = _tk[:, :, 1:n_tok][_mk]
        model.feat_mean.copy_(torch.tensor(_num.mean(0), dtype=torch.float32, device=DEVICE))
        model.feat_std.copy_(torch.tensor(_num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
        print(f"enriched tokens: n_token_feats={n_tok} | num_embed={cfg.get('num_embed','raw')}")
    print(f"cls_decoder: {cfg.get('cls_decoder', 'pooled')} | "
          f"decoder_source: {cfg.get('decoder_source', 'encoded')} | "
          f"loss: {cfg.get('loss', 'bce')}")

    # Whole-pipeline fine-tune probe: warm-start encoder+decoder from a prior checkpoint, then continue
    # training with the new objective (so the encoder ADAPTS to it — a fair test, unlike a frozen-rep probe).
    ws = cfg.get("warm_start")
    if ws:
        wsp = Path(ws) if Path(ws).is_absolute() else (ROOT / ws)
        sd = torch.load(wsp, weights_only=True, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"warm-start from {wsp} (missing={len(missing)} new params, unexpected={len(unexpected)})")

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device : {DEVICE}")
    print(f"Params : {n_params:,}")
    print(f"Train  : {len(train_ds)} closed  |  Open: {len(open_ds)}")

    # Loss
    y_train_np = np.load(DATA_DIR / "y_train_set.npy")
    pos_weight = compute_pos_weight(y_train_np)
    # Cap pos_weight to avoid extreme recall bias (empirical neg/pos ~32x for rare
    # donors causes over-prediction). Default cap=10; config can override.
    pw_cap = cfg.get("pos_weight_cap", 10.0)
    pos_weight = pos_weight.clamp(max=pw_cap).to(DEVICE)
    cls_loss_name = cfg.get("cls_loss")           # Inc9 A1/A2/A3 decoder-under-read losses (overrides --loss)
    _inc9_loss = build_cls_loss(cls_loss_name, y_train_np, cfg) if cls_loss_name else None
    if _inc9_loss is not None:
        bce_cls = _inc9_loss.to(DEVICE)
        print(f"cls loss: {cls_loss_name} (Inc9 long-tail decoder loss)")
    elif cfg.get("loss", "bce") == "asl":
        bce_cls = AsymmetricLoss(
            gamma_neg=cfg.get("asl_gamma_neg", 4.0),
            gamma_pos=cfg.get("asl_gamma_pos", 0.0),
            clip=cfg.get("asl_clip", 0.05),
        )
        print(f"cls loss: AsymmetricLoss(gamma_neg={cfg.get('asl_gamma_neg',4.0)}, "
              f"gamma_pos={cfg.get('asl_gamma_pos',0.0)}, clip={cfg.get('asl_clip',0.05)})")
    else:
        print(f"pos_weight: capped at {pw_cap}, range [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")
        bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_rej    = nn.BCEWithLogitsLoss()

    alpha = cfg.get("alpha_reject", 0.5)
    beta  = cfg.get("beta_card",    0.3)        # weight for cardinality head
    use_noc_v2 = bool(cfg.get("noc_head_v2", False))     # paper-proof CORN count head (mass+MAC+prob)
    noc_v2_w   = float(cfg.get("noc_v2_weight", beta))   # weight of its CORN-on-true-NOC loss
    if use_noc_v2:
        print(f"noc_head_v2 ON: CORN ordinal count on TRUE noc (w={noc_v2_w}); test decode compares vs joint/post-hoc")
    # §3.2 LEARNED gate-count (design_4head_decomposition.md): calibrate the AdaSlot existence gate directly
    # to the true NOC so count = Σgate is *learned* WITH gradient to the encoder — the priority fix vs the
    # DETACHED post-hoc noc_head_v2/RF crutch. aslot-only (needs the gate). Default off → path unchanged.
    use_gate_count = bool(cfg.get("gate_count", False)) and cfg.get("cls_decoder") == "aslot"
    gate_count_w   = float(cfg.get("gate_count_weight", beta))   # Σgate≈NOC smooth-L1 weight
    if cfg.get("gate_count", False) and not use_gate_count:
        print("[WARN] --gate_count needs --cls_decoder aslot (the AdaSlot gate) -> ignored")
    if use_gate_count:
        print(f"gate_count ON (design 4-head sec3.2): differentiable sum(gate)~=NOC consistency "
              f"(smooth-L1 w={gate_count_w}); count=sum(gate) LEARNED (gradient->encoder); "
              f"decode adds gate_sum vs joint/post-hoc/noc_v2")
    # Per-NOC val-oracle TRAJECTORY logging (--log_per_noc): writes hrow['val_per_noc_oracle'] each epoch
    # so per-NOC convergence SPEED is inspectable (does N5 converge slower / peak LATER than N1-N4, and do
    # extra heads widen that gap?). Diagnostic only: one extra val pass; selection metric untouched.
    # Default off -> history schema + per-epoch timing unchanged (bit-identical training).
    use_log_per_noc = bool(cfg.get("log_per_noc", False))
    if use_log_per_noc:
        print("log_per_noc ON: per-NOC val oracle appended to history each epoch (per-NOC convergence trajectory)")
    # §3.2.1 mass-aware gate (built into the model via gate_mass); §3.2.4 anneal gumbel_temp; §3.4 phi_gated.
    use_gate_mass = bool(cfg.get("gate_mass", False)) and cfg.get("cls_decoder") == "aslot"
    if use_gate_mass:
        print("gate_mass ON (sec3.2.1): grad-enabled slot-mass fed into the AdaSlot existence gate (ReZero, no-op start)")
    # §3.2.4 anneal the Gumbel temperature from gumbel_temp -> gate_temp_final over training (sharper gate
    # for counting late; Concrete/Gumbel-Softmax). Off (None) -> temp fixed at gumbel_temp (bit-identical).
    gate_temp_init  = float(cfg.get("gumbel_temp", 1.0))
    gate_temp_final = cfg.get("gate_temp_final", None)
    use_temp_anneal = (gate_temp_final is not None) and cfg.get("cls_decoder") == "aslot"
    if use_temp_anneal:
        gate_temp_final = float(gate_temp_final)
        print(f"gate_temp anneal ON (sec3.2.4): gumbel_temp {gate_temp_init} -> {gate_temp_final} linearly over training")
    # §3.4 presence-gated phi + β-NLL: replace L1-on-sparse with a heteroscedastic β-NLL masked to present
    # donors (needs aux_heads + the phi_logvar_head built when phi_gated). Off -> L1 phi path unchanged.
    use_phi_gated = bool(cfg.get("phi_gated", False)) and getattr(model, "phi_gated", False)
    phi_gated_w   = float(cfg.get("phi_gated_weight", 0.3))   # FIXED weight (NOT Kendall — β-NLL can go negative)
    if use_phi_gated:
        print(f"phi_gated ON (sec3.4): presence-gated phi + beta-NLL (masked to present donors, fixed w={phi_gated_w}, "
              f"NOT Kendall since beta-NLL can be negative) replaces L1-on-sparse")
    card_lam = cfg.get("card_lambda", 0.02)     # cardinality cost weight
    # Cardinality class weights (train ~85% NOC=1 -> plain CE collapses to k=1)
    _nc = np.bincount(np.clip(np.load(DATA_DIR/"noc_train.npy"),1,5)-1, minlength=5).astype(float)
    _w = 1.0/np.clip(_nc,1,None); _w = _w/_w.mean()
    card_w = torch.tensor(np.clip(_w, 0.5, 2.0), dtype=torch.float32).to(DEVICE)   # bounded both ways
    print(f"card class weights: {[round(x,2) for x in card_w.tolist()]}")

    # Increment 2 §5 — Kendall (2018) homoscedastic-uncertainty weighting for the privileged
    # aux losses: minimise exp(-s)·L + s so each task's weight self-tunes (hand-tuning infeasible).
    use_aux = bool(cfg.get("aux_heads", False)) and getattr(model, "aux_heads", False)
    log_var_attr = torch.zeros((), device=DEVICE, requires_grad=use_aux)
    log_var_phi  = torch.zeros((), device=DEVICE, requires_grad=use_aux)
    # Inc-LUPI physical heads — gates + Kendall log-variances (one per used loss).
    use_lupi     = bool(cfg.get("lupi_phys", False)) and getattr(model, "lupi_phys", False)
    use_l_degr   = use_lupi and bool(cfg.get("lupi_degr", False))
    use_l_mu     = use_lupi and bool(cfg.get("lupi_mu", False))
    use_l_var    = use_lupi and bool(cfg.get("lupi_var", False))
    use_l_dropin = use_lupi and bool(cfg.get("lupi_dropin", False))
    LUPI_DEG_MAX = 0.004    # = make_insilico DEG_BETA_MAX (β scaling to [0,1])
    log_var_degr   = torch.zeros((), device=DEVICE, requires_grad=use_l_degr)
    log_var_mu     = torch.zeros((), device=DEVICE, requires_grad=use_l_mu)
    log_var_lvar   = torch.zeros((), device=DEVICE, requires_grad=use_l_var)
    log_var_dropin = torch.zeros((), device=DEVICE, requires_grad=use_l_dropin)
    # Increment 2 §4b — decoupled ordinal NOC contrast (Rank-N-Contrast), Kendall-weighted too.
    use_rnc = bool(cfg.get("noc_contrast", False)) and getattr(model, "noc_contrast", False)
    rnc_tau = cfg.get("rnc_tau", 2.0)
    # §4b-C valve mode for the NOC contrast: "shared" (grad reaches encoder, original 2c),
    # "detach" (pma_noc pools H.detach() — set on the model), "pcgrad" (grad reaches encoder but
    # is projected off the main-task gradient). detach/pcgrad are the ID-protection valves.
    rnc_mode = cfg.get("noc_contrast_mode", "shared")
    # rnc_fixed_weight: if set, BYPASS Kendall for the contrast (fixed weight * loss_rnc). Kendall
    # silently down-weights a hard/non-descending task -> can disable RNC; a fixed weight isolates
    # whether the contrast itself works (F19 manipulation-check fix). None = Kendall (original).
    rnc_fixed_w = cfg.get("rnc_fixed_weight", None)
    # Inc3 B: supervised-contrastive peak grouping by donor (fixed weight, avoids F19 Kendall-inert trap).
    use_donorcon = bool(cfg.get("donor_contrast", False)) and getattr(model, "donor_contrast", False)
    donorcon_w = cfg.get("donor_contrast_weight", 0.1)
    donorcon_tau = cfg.get("donor_contrast_tau", 0.1)
    # Inc3 V1-3: CORN ordinal count head on its own encoder pool; ensemble (V1/V2) or replace (V3) at decode.
    use_ord = bool(cfg.get("noc_ord_head", False)) and getattr(model, "noc_ord_head", False)
    ord_w = cfg.get("noc_ord_weight", beta)
    ord_replace = bool(cfg.get("noc_ord_replace", False))
    # log_var_rnc trainable only when RNC is on AND Kendall (not fixed-weight) is used.
    log_var_rnc = torch.zeros((), device=DEVICE, requires_grad=(use_rnc and rnc_fixed_w is None))
    # IRM (§7, Arjovsky 2019): NOC-stratified invariance penalty on the cls logits.
    # irm_anneal (epochs): linear warmup of the penalty weight 0 -> irm_lambda. REQUIRED — a high
    # fixed penalty from scratch stops features from forming (Arjovsky §5.1; Gulrajani & Lopez-Paz
    # 2020 / DomainBed). Without warmup, large lambda breaks training.
    use_irm = bool(cfg.get("irm", False))
    irm_lambda = cfg.get("irm_lambda", 1.0)
    irm_anneal = cfg.get("irm_anneal", 10)
    # Increment 8 (F32) — VICReg (arXiv 2105.04906) VERBATIM on the per-donor pooled encoded reps.
    # N1-safe by construction (the variance term PRESERVES per-dim info instead of subtracting a per-
    # sample mean, which destroyed N1: 1.0->0.02). V1 vicreg = variance+covariance (anti-collapse decorr
    # = de-smooth the combo carrier). V2 + invariance = pull the SAME donor's rep together across combos.
    # Paper coefficients: var mu=25, cov nu=1, inv lambda=25; overall scaled by vicreg_w (aux magnitude).
    use_vicreg     = bool(cfg.get("vicreg", False)) and getattr(model, "return_encoded", False)
    use_vicreg_inv = bool(cfg.get("vicreg_inv", False)) and use_vicreg
    vicreg_w       = cfg.get("vicreg_weight", 0.04)            # overall aux scale (paper ratios kept inside)
    VIC_MU, VIC_NU, VIC_LAM = 25.0, 1.0, 25.0                  # var, cov, inv (paper defaults)
    if use_vicreg: print(f"Inc8 VICReg ON  var+cov{' +inv' if use_vicreg_inv else ''}  w={vicreg_w} (mu/nu/lam=25/1/25)")
    use_donor_recon = bool(cfg.get("donor_recon", False)) and getattr(model, "donor_recon", False)  # Inc9 B1
    donor_recon_w   = cfg.get("donor_recon_weight", 0.1)
    if use_donor_recon: print(f"Inc9 B1 donor-recon ON  w={donor_recon_w}")
    if int(cfg.get("attn_sink", 0)) > 0: print(f"Inc9 B4 attn-sink ON  n_sink={int(cfg.get('attn_sink',0))}")
    use_qdn  = bool(cfg.get("query_denoise", False))                 # Inc9 A4 (needs geno_query for the anchor)
    qdn_w    = cfg.get("qdn_weight", 0.5)
    use_ifm  = bool(cfg.get("ifm", False)) and cfg.get("cls_decoder") == "per_donor"  # Inc9 B2
    ifm_eps  = cfg.get("ifm_eps", 0.05); ifm_w = cfg.get("ifm_weight", 0.5)
    if use_qdn: print(f"Inc9 A4 query-denoise ON  w={qdn_w} noise={cfg.get('qdn_noise',1.0)}")
    if use_ifm: print(f"Inc9 B2 IFM ON  eps={ifm_eps} w={ifm_w}")
    # Increment 6 (root anti-memorization levers): D-mask = input-peak dropout; D-andmask = ILC
    # AND-mask on the cls grad across low/high-NOC environments. Both attack combo-memorization
    # WITHOUT a penalty term, so the aux/phi/attr heads (which IRM broke, F23) stay intact.
    mask_peaks_p = float(cfg.get("mask_peaks", 0.0) or 0.0)
    mask_min     = int(cfg.get("mask_peaks_min", 8))
    use_and_mask = bool(cfg.get("and_mask", False))
    and_tau      = float(cfg.get("and_mask_thresh", 0.0) or 0.0)
    use_sam      = bool(cfg.get("sam", False))
    sam_rho      = float(cfg.get("sam_rho", 0.05) or 0.05)
    use_vib      = bool(cfg.get("vib", False))
    vib_w        = float(cfg.get("vib_weight", 1e-3) or 1e-3)
    mpre_epochs  = int(cfg.get("masked_pretrain_epochs", 0) or 0)
    mpre_ratio   = float(cfg.get("mask_pretrain_ratio", 0.3) or 0.3)
    use_minorw   = bool(cfg.get("minor_weight", False))
    minor_phi_ref = float(cfg.get("minor_phi_ref", 0.2) or 0.2)
    minor_cap    = float(cfg.get("minor_cap", 4.0) or 4.0)
    # Lever B (Increment 13) — Generalized Distillation / LUPI (Lopez-Paz et al. ICLR 2016):
    # distill the COMBO-INVARIANT per-peak attribution readout (attr_head soft-vote = teacher)
    # INTO the per-donor decoder (student) so the decoder internalises the combo-invariant
    # ranking instead of overfitting donor combos. Teacher detached (attr_head keeps its own CE);
    # weight RAMPED from 0 so the early (untrained) attr_head can't mislead. Needs aux_heads.
    use_distill  = bool(cfg.get("distill_attr", False)) and bool(cfg.get("aux_heads", False))
    distill_w    = float(cfg.get("distill_weight", 0.5) or 0.5)
    distill_ramp = int(cfg.get("distill_ramp", 20) or 20)
    if use_distill:
        print(f"Inc13 Lever B distill ON: attr_head->decoder (LUPI), w={distill_w} ramp={distill_ramp}ep")
    # Lever A (Increment 13) — Counterfactual combo-invariance (Veitch et al. NeurIPS 2021; Noisy
    # Counterfactual Matching 2025). On-the-fly LEAVE-ONE-DONOR-OUT intervention: drop all peaks of one
    # present donor (via attr) and require the OTHER present donors' logits to be UNCHANGED. Teaches that
    # a donor's presence is independent of co-occurring donors (attacks combo-memorization). NCM-style:
    # the consistency hits ONLY the unchanged-donor direction (narrow, not a blunt global penalty like IRM).
    use_cf  = bool(cfg.get("cf_invariance", False)) and bool(cfg.get("aux_heads", False))
    cf_w    = float(cfg.get("cf_weight", 1.0) or 1.0)
    cf_ramp = int(cfg.get("cf_ramp", 10) or 10)
    if use_cf:
        print(f"Inc13 Lever A cf-invariance ON (representation-level): leave-one-donor-out rep-match, w={cf_w} ramp={cf_ramp}ep")
    # Lever C (Increment 13) — DIFFICULTY-LADDER ("use easy to teach hard", FixMatch/FlexMatch-style
    # weak->strong consistency + curriculum, but with TRUE labels at every rung so it can't compound
    # error like SSL pseudo-labels). On-the-fly: a fainter version of each mixture (scale the faintest
    # minor's heights, curriculum-annealed harder over epochs); true-label loss on it + consistency to
    # the full-mixture prediction (easy teaches hard). Needs attr (faintest-minor peaks) + phi.
    use_diff   = bool(cfg.get("diff_ladder", False)) and bool(cfg.get("aux_heads", False))
    diff_w     = float(cfg.get("diff_weight", 0.5) or 0.5)
    diff_minf  = float(cfg.get("diff_min_factor", 0.5) or 0.5)   # keep the minor IDENTIFIABLE (valid label)
    diff_ramp  = int(cfg.get("diff_ramp", 30) or 30)
    if use_diff:
        print(f"Inc13 Lever C difficulty-ladder ON (feature-level, FitNets/AT): faint-minor harden w={diff_w} "
              f"factor 0.7->{diff_minf} over {diff_ramp}ep (per-donor rep cosine-match, no output-on-hard)")

    # Inc14 B-v2 — multi-label genotype-attribution head (CPI co-membership) + MLD distillation.
    # Forensic ground: peak height is additive over contributors; an allele is 'included' for EVERY
    # present donor carrying it (75% shared). attr_head softmax can't learn this (probe: teacher ceiling
    # .82 vs multi-label readout .96). ml_attr = sigmoid head learns it; ml_distill = per-class binary KD
    # (Multi-Label KD 'MLD', Yang 2023) distils that richer teacher into the per-donor decoder.
    use_mlattr     = bool(cfg.get("ml_attr", False)) and owner_lut is not None
    ml_attr_w      = float(cfg.get("ml_attr_weight", 1.0) or 1.0)
    use_ml_distill = bool(cfg.get("ml_distill", False)) and use_mlattr
    ml_distill_w   = float(cfg.get("ml_distill_weight", 0.5) or 0.5)
    ml_distill_ramp = int(cfg.get("ml_distill_ramp", 20) or 20)
    ml_pos_weight  = torch.tensor(float(cfg.get("ml_pos_weight", 20.0)), device=DEVICE)
    if use_mlattr:
        print(f"Inc14 B-v2 ml_attr ON: per-peak multi-label genotype co-membership head (BCE w={ml_attr_w}, "
              f"pos_weight={float(ml_pos_weight)}) | MLD distill={use_ml_distill} (w={ml_distill_w} ramp={ml_distill_ramp})")
    # Inc14 A-v2 — additive-subtraction counterfactual + conditional MMD on the IDENTITY sub-rep.
    use_addinv   = bool(cfg.get("add_invar", False)) and owner_lut is not None
    addinv_w     = float(cfg.get("add_invar_weight", 1.0) or 1.0)
    addinv_ramp  = int(cfg.get("add_invar_ramp", 10) or 10)
    id_dim       = int(cfg.get("id_dim", cfg.get("d_model", 128) // 2))   # identity = first half of last_reps
    if use_addinv:
        print(f"Inc14 A-v2 add_invar ON: additive-subtraction counterfactual + per-slot conditional MMD on "
              f"identity sub-rep[:{id_dim}] (w={addinv_w} ramp={addinv_ramp}ep) — forensic-correct, anti-collapse")

    def gather_owner(tok):
        """(B,N,n_classes) — does each known donor's genotype carry this peak's (locus,allele)."""
        loc = tok[:, :, 0].long().clamp(0, 23)
        ab = (torch.round(tok[:, :, 1] * 10).long() + ALLELE_OFF).clamp(0, owner_lut.size(1) - 1)
        return owner_lut[loc, ab]                              # advanced-index -> (B,N,n_classes)

    # Inc17 — noise-spectrum: height-stratified filter augmentation. Drop peaks below RFU level F (clean<->noisy),
    # keeping >= mask_min so NOC stays identifiable. The real faint band is ~91% artifact -> high F = cleaner.
    use_noise_spec = bool(cfg.get("noise_spectrum", False))
    ns_levels = [float(x) for x in str(cfg.get("noise_spectrum_levels", "0,15,30,45")).split(",")]
    use_xnoise = bool(cfg.get("xnoise_consistency", False))
    xnoise_w = float(cfg.get("xnoise_weight", 1.0) or 1.0)
    xnoise_ramp = int(cfg.get("xnoise_ramp", 15) or 15)
    xnoise_clean = float(cfg.get("xnoise_clean", 30.0) or 30.0)
    if use_noise_spec:
        print(f"Inc17 NOISE-SPECTRUM ON: random filter level/batch from {ns_levels} RFU (min keep {mask_min})")
    if use_xnoise:
        print(f"Inc17 X-NOISE CONSISTENCY ON: noisy->clean(F={xnoise_clean}) stop-grad anchor, w={xnoise_w} ramp={xnoise_ramp}ep")
    def filter_by_height(tk, mk, F):
        if F <= 0: return mk
        mb = mk.bool(); rfu = torch.expm1(tk[:, :, 2])
        keep = mb & (rfu >= F)
        enough = keep.sum(1, keepdim=True) >= mask_min        # only filter where >= mask_min peaks survive
        return torch.where(enough, keep, mb).to(mk.dtype)

    # Inc15 — "LEARN THE PROCESS" forward-reconstruction (analysis-by-synthesis): the donors+proportions
    # the model PREDICTS must additively explain the OBSERVED peak heights (EuroForMix forward model:
    # H(allele) = T * sum_d present_d * Mx_d * copy_d(allele)). A wrongly-included decoy then creates a
    # reconstruction error -> the WHOLE pipeline (encoder+decoder) is pulled toward reconstruction-consistent,
    # combo-invariant-by-construction identification. Probed whole-pipeline (fine-tune), not bolted-on.
    use_recon = bool(cfg.get("recon", False)) and owner_lut is not None
    recon_w   = float(cfg.get("recon_weight", 0.3) or 0.3)
    recon_ramp = int(cfg.get("recon_ramp", 0) or 0)
    if use_recon:
        print(f"Inc15 RECON ON (learn-the-process forward reconstruction): predicted donors+phi must explain "
              f"observed heights, w={recon_w} ramp={recon_ramp}ep")

    # Inc16 — additive GRID recon over PRESENT+ABSENT (validated synth recipe; the absent positions are the
    # damning-absence signal Inc15 lacks). hhat[locus,allele] = exp(logG)*sum_d sigma(logit_d)*phi_d*dosage_d.
    use_add_recon = bool(cfg.get("add_recon", False)) and owner_lut is not None
    add_recon_w   = float(cfg.get("add_recon_weight", 0.1) or 0.1)
    add_recon_logG = torch.zeros(1, device=DEVICE, requires_grad=True) if use_add_recon else None
    add_recon_possible = (owner_lut.sum(-1) > 0) if use_add_recon else None    # (24,LUT_W) panel-possible positions
    if use_add_recon:
        print(f"Inc16 ADD_RECON ON: additive grid log-recon over present+absent (w={add_recon_w}); "
              f"predict 0 at damning-absent positions -> penalizes decoy inclusion. possible bins={int(add_recon_possible.sum())}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    kendall = ([log_var_attr, log_var_phi] if use_aux else []) \
        + ([log_var_rnc] if (use_rnc and rnc_fixed_w is None) else []) \
        + ([log_var_degr] if use_l_degr else []) + ([log_var_mu] if use_l_mu else []) \
        + ([log_var_lvar] if use_l_var else []) + ([log_var_dropin] if use_l_dropin else [])
    if kendall:
        optimizer.add_param_group({"params": kendall, "weight_decay": 0.0})
    if use_add_recon:
        optimizer.add_param_group({"params": [add_recon_logG], "weight_decay": 0.0})   # learnable template scale
    # Params over which PCGrad does its surgery (model + Kendall log-vars). Built once.
    pcgrad_params = list(model.parameters()) + kendall
    if use_aux:
        print(f"aux heads ON: per-peak allele->donor attribution + phi regression (Kendall-weighted)")
    if use_rnc:
        wstr = f"fixed_weight={rnc_fixed_w}" if rnc_fixed_w is not None else "Kendall-weighted"
        print(f"NOC contrast ON: Rank-N-Contrast (tau={rnc_tau}, mode={rnc_mode}, decoupled pool+projection, {wstr})")
    _cdec = cfg.get("cls_decoder", "pooled")
    if cfg.get("sparse_attn", False):
        if _cdec == "per_donor":
            print("sparse attention ON: sparsemax per-donor decoder (§7)")
        else:
            print(f"  [WARN] --sparse_attn IGNORED with cls_decoder={_cdec} "
                  f"(only the per_donor decoder consumes it; no sparse attention is active).")
    if cfg.get("soft_geno_attr", False) and _cdec in ("aslot", "spen"):
        print(f"  [WARN] --soft_geno_attr IGNORED with cls_decoder={_cdec} "
              f"(inference attr-masking runs only in the standard forward path, not _forward_{_cdec}).")
    if use_irm:
        print(f"IRM ON: NOC-environment invariance penalty (lambda={irm_lambda}, warmup={irm_anneal}ep, §7)")
    if mask_peaks_p > 0.0:
        print(f"Inc6 D-mask ON: input-peak dropout p={mask_peaks_p} (min keep {mask_min}) — anti combo-memorization")
    if use_and_mask:
        print(f"Inc6 D-andmask ON: ILC AND-mask on cls grad across low/high-NOC envs (tau={and_tau})")
    if use_minorw:
        print(f"Inc6 minor-weight ON: positive cost = clip({minor_phi_ref}/mixture_frac, 1, {minor_cap}) "
              f"— rescue low-phi minor contributors (verified N5 root cause)")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )

    # Inc6 3e: self-supervised MASKED-FEATURE PRE-TRAIN of the encoder (SAINT/MAE-style). Randomly
    # mask a fraction of valid peaks' continuous features and reconstruct them from the ISAB-encoded
    # H — forces per-peak/per-locus structure the encoder must infer from CONTEXT (not a memorized
    # combo template), warming the encoder toward combo-invariant features before supervised ID.
    if mpre_epochs > 0 and n_tok > 3:
        n_num = n_tok - 1
        recon_head = nn.Sequential(nn.Linear(model.d_model, model.d_model), nn.ReLU(inplace=True),
                                   nn.Linear(model.d_model, n_num)).to(DEVICE)
        pre_opt = torch.optim.AdamW(list(model.parameters()) + list(recon_head.parameters()),
                                    lr=cfg.get("lr", 3e-4), weight_decay=cfg.get("weight_decay", 1e-4))
        print(f"Inc6 3e masked-pretrain: {mpre_epochs} epochs, mask ratio {mpre_ratio} (reconstruct {n_num} feats)")
        model.train()
        for pe in range(1, mpre_epochs + 1):
            tot = 0.0; nb = 0
            for tokens, mask, *_ in train_loader:
                tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)
                mb = mask.bool()
                cont_true = (tokens[:, :, 1:n_tok] - model.feat_mean) / model.feat_std   # standardized target
                mpos = (torch.rand_like(mb, dtype=torch.float) < mpre_ratio) & mb        # peaks to mask
                tok_m = tokens.clone()
                tok_m[:, :, 1:n_tok] = torch.where(mpos.unsqueeze(-1), model.feat_mean.expand_as(tok_m[:, :, 1:n_tok]),
                                                   tok_m[:, :, 1:n_tok])
                _x0, H, _pad = model._encode_set(tok_m, mask)
                pred = recon_head(H)                                                     # (B,N,n_num) standardized
                sel = mpos.unsqueeze(-1).expand_as(pred)
                if sel.any():
                    loss_pre = F.mse_loss(pred[sel], cont_true[sel])
                    pre_opt.zero_grad(); loss_pre.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0); pre_opt.step()
                    tot += loss_pre.item(); nb += 1
            if pe % max(1, mpre_epochs // 5) == 0:
                print(f"  pretrain ep{pe}/{mpre_epochs}  recon_mse={tot/max(nb,1):.4f}")
        del recon_head, pre_opt; torch.cuda.empty_cache() if DEVICE.type == "cuda" else None

    best_f1, best_epoch, patience_count = 0.0, 0, 0
    patience = cfg.get("patience", 15)
    epochs   = cfg.get("epochs", 100)
    history  = []
    # EARLY_ABORT (fail-fast lever screening): default OFF -> path byte-identical. When on, a periodic
    # DEV per-NOC oracle check kills clearly-failing runs so the runner moves to the next lever.
    early_abort = bool(cfg.get("early_abort"))
    ea_base_n5  = float(cfg.get("early_abort_base_n5", 0.59))   # base inc2_2d_sparse DEV N5 oracle
    best_dev_n5, ea_strikes, abort_info = 0.0, 0, None
    if early_abort:
        print(f"EARLY_ABORT ON: guard N1/N2>=.95 (ep>=15), no-diverge, DEV N5 oracle >= {ea_base_n5:.2f}-0.05 (ep>=50, 2 strikes)")

    print(f"\nTraining up to {epochs} epochs (patience={patience}) ...")
    t0 = time.time()
    _ldam = bce_cls if isinstance(bce_cls, LDAMLoss) else None   # Inc9 A1: DRW needs the epoch

    # Inc18 V1: phi injection setup — privileged GT phi at train time; EM phi at inference via probe
    use_phi_inject = cfg.get("phi_inject", False)
    # phi_inject_fn extracts GT phi (index 1 in rest=(attr,phi)) from DataLoader batch
    _phi_fn = (lambda rest: rest[1]) if use_phi_inject else None
    # Inc18 V4 (ab_soft_label_compare result, F38+): EuroForMix phi*CN soft labels for attr head.
    # Replaces hard CE (which NaN-crashes on stutter). Needs aux_heads + cn_lut (donor_geno).
    use_soft_attr_label = bool(cfg.get("soft_attr_label", False)) and cn_lut is not None
    if use_soft_attr_label:
        print("soft_attr_label ON: EuroForMix phi*CN soft labels for attr head (no stutter NaN, wins decoyAUC+N5)")

    for epoch in range(1, epochs + 1):
        model.train()
        if _ldam is not None: _ldam.set_epoch(epoch)     # Inc9 A1: deferred re-weighting schedule
        if use_temp_anneal:                              # §3.2.4: linearly sharpen the AdaSlot gate over training
            frac = (epoch - 1) / max(1, epochs - 1)
            model.cls_decoder_module.gumbel_temp = gate_temp_init + (gate_temp_final - gate_temp_init) * frac
        epoch_loss = epoch_cls = epoch_rej = epoch_noc_l = 0.0
        epoch_attr = epoch_phi = epoch_rnc = epoch_irm = 0.0
        epoch_gate_count = 0.0                            # §3.2 Σgate≈NOC consistency

        for _batch in train_loader:
            tokens, mask, y, noc, attr, phi = _batch[:6]
            mu_t, var_t, dropin_t, beta_t = (_batch[6:10] if len(_batch) > 6 else (None, None, None, None))
            tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)
            if mu_t is not None:
                mu_t = mu_t.to(DEVICE); var_t = var_t.to(DEVICE); dropin_t = dropin_t.to(DEVICE); beta_t = beta_t.to(DEVICE)
            em_phi_in = phi.to(DEVICE) if use_phi_inject else None  # GT phi as privileged train signal
            y, noc = y.to(DEVICE), noc.to(DEVICE)
            attr = attr.to(DEVICE)            # per-peak donor (Inc13 levers A/C use it on-the-fly)
            phi  = phi.to(DEVICE)             # mixture proportions (Inc13 lever C picks the faintest minor)

            # Inc6 D-mask: randomly drop a fraction of VALID input peaks (observation dropout). The
            # label y (donor SET) is unchanged → the decoder must infer donors from a random peak
            # subset rather than template-matching a memorized combo. Keep >= mask_min peaks/sample so
            # NOC stays identifiable (never blank a profile). Train only; eval mask untouched.
            mask_orig = mask                                       # Inc17: unfiltered mask kept for the clean consistency view
            if use_noise_spec:                                     # Inc17: random filter level per batch (multi-noise training)
                Fm = ns_levels[int(torch.randint(len(ns_levels), (1,)).item())]
                mask = filter_by_height(tokens, mask_orig, Fm)
            elif mask_peaks_p > 0.0:
                mb = mask.bool()
                drop = (torch.rand_like(mb, dtype=torch.float) < mask_peaks_p) & mb
                if owner_lut is not None:
                    # Q5: never drop a minor donor's PRIVATE (single-carrier) peaks — the exact
                    # identifying evidence set_of_set isolates to protect. Drop only shared/redundant
                    # peaks, where the anti-memorization benefit lives anyway.
                    n_car = gather_owner(tokens).sum(-1)            # (B,N) #panel carriers per peak
                    drop = drop & (n_car != 1)
                kept = mb & ~drop
                enough = kept.sum(1, keepdim=True) >= mask_min      # (B,1) — only apply where safe
                mask = torch.where(enough, kept, mb).to(mask.dtype)

            # Closed-set forward
            out = model(tokens, mask, em_phi=em_phi_in)
            # Inc13 lever C (feature-level): snapshot the per-donor reps from THIS main forward now,
            # before any auxiliary forward (lever A's cf / lever C's hard) overwrites last_reps.
            reps_main = getattr(model.cls_decoder_module, "last_reps", None)
            if reps_main is not None:
                reps_main = reps_main.detach()
            # Inc6 minor-weight: up-weight low-mixture-fraction (minor) positives, capped so the
            # phi<0.05 noise floor is not over-emphasised (gamma_pos would chase it — see probe).
            if use_minorw and isinstance(bce_cls, AsymmetricLoss):
                phi_d = phi.to(DEVICE)
                tot = phi_d.sum(1, keepdim=True)
                frac = phi_d / tot.clamp(min=1e-6)
                ww = (minor_phi_ref / frac.clamp(min=1e-3)).clamp(1.0, minor_cap)
                wmat = torch.where((y > 0.5) & (tot > 0), ww, torch.ones_like(y))
                loss_cls = bce_cls(out["logits_cls"], y, weight=wmat)
            else:
                loss_cls = bce_cls(out["logits_cls"], y)
            # SPEN auxiliary: BCE on initial local logits (local head trains independently)
            if "logits_cls_0" in out:
                spen_aux_w = float(cfg.get("spen_aux_w", 1.0))
                loss_cls = loss_cls + spen_aux_w * bce_cls(out["logits_cls_0"], y)
            # Cardinality-aware NOC: cost-sensitive CE toward EM-optimal k
            card_tgt = cardinality_target(torch.sigmoid(out["logits_cls"]).detach(), y, card_lam)
            loss_noc = F.cross_entropy(out["logits_card"], card_tgt, weight=card_w)

            # Reject: closed → label 0
            rej_closed = out["logit_reject"]
            rej_label  = torch.zeros(len(tokens), 1, device=DEVICE)

            # Open-set batch for reject head
            try:
                open_batch = next(open_iter)
            except StopIteration:
                open_iter = iter(DataLoader(
                    open_ds, batch_size=max(1, int(cfg["batch_size"] * cfg.get("open_ratio", 0.25))),
                    sampler=RandomSampler(open_ds, replacement=True, num_samples=len(open_ds) * 10,
                                          generator=gen),
                    num_workers=0,
                ))
                open_batch = next(open_iter)

            o_tok, o_mask = open_batch[0].to(DEVICE), open_batch[1].to(DEVICE)
            rej_open  = model(o_tok, o_mask)["logit_reject"]
            rej_label_open = torch.ones(len(o_tok), 1, device=DEVICE)

            all_rej    = torch.cat([rej_closed, rej_open], dim=0)
            all_rej_lbl = torch.cat([rej_label, rej_label_open], dim=0)
            loss_rej   = bce_rej(all_rej, all_rej_lbl)

            loss = loss_cls + alpha * loss_rej + beta * loss_noc

            # Paper-proof NOC head (noc_head_v2): CORN rank-consistent ordinal regression on the TRUE
            # number of contributors (Cao 2020 / Shi 2021) — independent of the decode-aligned card_tgt;
            # trains the differentiable count head reading mass-preserving slot-mass + MAC + prob-profile.
            if use_noc_v2 and "logits_count_v2" in out:
                loss = loss + noc_v2_w * corn_loss(out["logits_count_v2"], noc.clamp(1, 5), 5)

            # §3.2 LEARNED gate-count (--gate_count): calibrate the AdaSlot existence gate directly to the
            # true NOC so count = Σgate is a *learned* quantity. Unlike noc_head_v2 (reads DETACHED features
            # → a post-hoc reader capped by ID quality), this consistency loss keeps gradient through
            # gate_logit → slots → MESH → encoder, so H learns count-relevant peak-mass features
            # (AdaSlot CVPR2024; design §3.2.2/§3.2.5). Robust smooth-L1 on the sum of gate PROBABILITIES
            # (noise-free sigmoid — the "sum of gate probabilities = true count" regularizer) vs the true NOC.
            if use_gate_count and "gate_logit" in out:
                gate_sum = torch.sigmoid(out["gate_logit"]).sum(-1)           # (B,) Σ P(slot active)
                loss_gate_count = F.smooth_l1_loss(gate_sum, noc.clamp(1, 5).float())
                loss = loss + gate_count_w * loss_gate_count
                epoch_gate_count += loss_gate_count.item()

            # SOFT per-peak noise/stutter gate (noise_gate): supervised by the PROVEN real/noise label —
            # peak's (locus,allele) carried by >=1 TRUE contributor (Balding-Buckleton drop-in distinction).
            # The gate gets its OWN calibrated P(real) target so it can't co-adapt through the cls loss.
            if cfg.get("noise_gate", False) and "noise_gate_logit" in out:
                real = ((gather_owner(tokens) * y.unsqueeze(1)).sum(-1) > 0).float()   # (B,N) 1=real allele
                mvalid = mask.bool()
                loss_ng = F.binary_cross_entropy_with_logits(
                    out["noise_gate_logit"][mvalid], real[mvalid])
                loss = loss + cfg.get("noise_gate_w", 0.3) * loss_ng

            # Inc6 2f: VIB KL on the per-donor latents (read off the decoder module after forward)
            if use_vib and getattr(model.cls_decoder_module, "kl", None) is not None:
                loss = loss + vib_w * model.cls_decoder_module.kl

            # Inc3 V1-3: CORN rank-consistent ordinal count loss on the encoder count pool (the pool
            # RNC also shapes) — the count head finally reads an RNC-shaped representation (fixes F21).
            loss_ord_v = 0.0
            if use_ord and "logits_card_ord" in out:
                loss_ord = corn_loss(out["logits_card_ord"], noc.clamp(1, 5), 5)
                loss = loss + ord_w * loss_ord
                loss_ord_v = loss_ord.item()

            # IRM (§7): NOC-environment invariance penalty on the cls logits, with a linear penalty
            # WARMUP (0 -> irm_lambda over irm_anneal epochs) so features can form first.
            loss_irm_v = 0.0
            if use_irm:
                loss_irm = irm_penalty(out["logits_cls"], y, noc)
                lam_eff = irm_lambda * min(1.0, epoch / max(1, irm_anneal))
                loss = loss + lam_eff * loss_irm
                loss_irm_v = loss_irm.item()

            # Privileged auxiliary supervision (in-silico only): per-peak allele→donor
            # attribution (CE, pad/none ignored) + phi abundance regression (L1), each
            # Kendall-weighted. Guarded so batches without provenance (all attr=-1) skip.
            # soft_attr_label: EuroForMix phi*CN soft CE instead of hard CE (no NaN on stutter).
            loss_attr_v = loss_phi_v = 0.0
            if use_aux:
                la = out["logits_attr"]; B_, S_, C_ = la.shape
                if (attr >= 0).any():
                    if use_soft_attr_label:
                        li_ = tokens[..., 0].long().clamp(0, 23)
                        bi_ = (tokens[..., 1] * 10).round().long() + ALLELE_OFF
                        bi_ = bi_.clamp(0, cn_lut.size(1) - 1)
                        cn_ = cn_lut[li_, bi_]                      # (B, N, C)
                        phi_cn_ = phi.unsqueeze(1) * cn_            # (B, N, C)
                        tot_ = phi_cn_.sum(-1, keepdim=True)        # (B, N, 1)
                        # attr==-1 peaks (artifact/drop-in, or provenance-less real samples) carry no
                        # true source: route them to background, not a coincidentally-feasible donor
                        # (which would be a false attribution the hard-CE path excluded via ignore_index).
                        feas_ = (tot_.squeeze(-1) > 1e-9) & mask.bool() & (attr >= 0)
                        norm_ = phi_cn_ / tot_.clamp(min=1e-9)
                        bg_ = (~feas_ & mask.bool()).unsqueeze(-1).float()
                        soft_y_ = torch.cat([norm_ * feas_.unsqueeze(-1).float(), bg_], dim=-1)  # (B, N, C+1)
                        log_p_ = F.log_softmax(la, dim=-1)           # (B, N, C+1)
                        raw_l_ = -(soft_y_ * log_p_).sum(-1)         # (B, N)
                        vm_ = mask.bool().float()
                        loss_attr = (raw_l_ * vm_).sum() / vm_.sum().clamp(min=1)
                    else:
                        loss_attr = F.cross_entropy(la.reshape(B_ * S_, C_), attr.reshape(B_ * S_),
                                                    ignore_index=-1)
                    if use_phi_gated and "phi_logvar" in out and (y > 0.5).any():
                        # §3.4 presence-gated β-NLL (Seitzer 2022): supervise phi ONLY on PRESENT donors
                        # (y>0.5) so the ~88%-zero absent slots can't collapse it to 0; heteroscedastic with
                        # a stop-grad β=0.5 weight (avoids the variance-collapse pitfall).
                        pres = (y > 0.5)
                        lv = out["phi_logvar"].clamp(-8.0, 8.0)
                        nll = 0.5 * torch.exp(-lv) * (out["phi"] - phi) ** 2 + 0.5 * lv
                        w_b = torch.exp(lv).detach().clamp(min=1e-6) ** 0.5
                        loss_phi = (nll * w_b)[pres].mean()
                    else:
                        loss_phi  = F.l1_loss(out["phi"], phi)
                    # attr (CE, bounded >=0) stays Kendall-weighted. The §3.4 β-NLL phi can go NEGATIVE →
                    # a Kendall wrapper would detonate (exp(−log_var)·(neg loss)→−∞ = the §4/§6b pitfall), so
                    # phi_gated uses a FIXED weight; the legacy L1 phi (>=0) keeps its Kendall wrapper.
                    loss = loss + torch.exp(-log_var_attr) * loss_attr + log_var_attr
                    if use_phi_gated and "phi_logvar" in out:
                        loss = loss + phi_gated_w * loss_phi
                    else:
                        loss = loss + torch.exp(-log_var_phi) * loss_phi + log_var_phi
                    loss_attr_v = loss_attr.item(); loss_phi_v = loss_phi.item()

            # Inc-LUPI: privileged PHYSICAL supervision (synthetic rows only; β=-1 sentinel masks real N1).
            #   degradation β (scaled L1) · clean mu denoise + variance as Gaussian β-NLL (Seitzer 2022,
            #   β=0.5 stop-grad weight → avoids the variance-collapse pitfall) · per-peak drop-in (BCE).
            #   All Kendall-weighted; per-peak losses masked to valid peaks of labelled rows.
            if use_lupi and beta_t is not None and "degr" in out:
                syn = (beta_t >= 0)                                  # synthetic rows carry physical labels
                vm  = mask.bool() & syn.unsqueeze(1)                 # valid peaks of labelled rows
                if use_l_degr and syn.any():
                    bt = (beta_t / LUPI_DEG_MAX).clamp(0.0, 1.0)
                    loss_degr = F.l1_loss(out["degr"][syn], bt[syn])
                    loss = loss + torch.exp(-log_var_degr) * loss_degr + log_var_degr
                if (use_l_mu or use_l_var) and vm.any():
                    mu_tgt = torch.log1p(mu_t.clamp(min=0.0))        # clean log1p-height target (0 at phantom peaks)
                    mean = out["mu_mean"]; logv = out["mu_logvar"].clamp(-8.0, 8.0)
                    if use_l_mu:                                     # β-NLL (β=0.5): weight = stop-grad(σ²)^β
                        nll = 0.5 * torch.exp(-logv) * (mean - mu_tgt) ** 2 + 0.5 * logv
                        w_bnll = torch.exp(logv).detach().clamp(min=1e-6) ** 0.5
                        loss_mu = (nll * w_bnll)[vm].mean()
                        loss = loss + torch.exp(-log_var_mu) * loss_mu + log_var_mu
                    if use_l_var:                                   # tie predicted log-var → TRUE delta-method log-space var
                        tgt_lv = torch.log(var_t.clamp(min=1e-6)) - 2.0 * mu_tgt   # Var(log1p h) ≈ Var(h)/(mu+1)²
                        loss_lvar = F.l1_loss(logv[vm], tgt_lv[vm])
                        loss = loss + torch.exp(-log_var_lvar) * loss_lvar + log_var_lvar
                if use_l_dropin and vm.any():                       # per-peak drop-in (phantom); rare → pos_weight
                    loss_di = F.binary_cross_entropy_with_logits(
                        out["dropin_logit"][vm], dropin_t[vm], pos_weight=torch.tensor(15.0, device=DEVICE))
                    loss = loss + torch.exp(-log_var_dropin) * loss_di + log_var_dropin

            # Lever B (Inc13, generalized distillation): teacher = combo-invariant attr_head
            # soft-vote (max over the SAME valid peaks the model saw), detached; student = the
            # per-donor decoder logits. Soft-target BCE pulls the decoder's per-donor ranking
            # toward the combo-invariant readout. Ramped from 0 (early attr_head is untrained).
            loss_distill_v = 0.0
            if use_distill and "logits_attr" in out:
                with torch.no_grad():
                    ap = torch.softmax(out["logits_attr"][:, :, :45], dim=-1)       # (B,N,45)
                    ap = ap.masked_fill(~mask.bool().unsqueeze(-1), 0.0)
                    teacher_p = ap.max(dim=1).values.clamp(1e-4, 1 - 1e-4)          # (B,45) combo-invariant
                w_eff = distill_w * min(1.0, epoch / max(1, distill_ramp))
                loss_distill = F.binary_cross_entropy_with_logits(out["logits_cls"], teacher_p)
                loss = loss + w_eff * loss_distill
                loss_distill_v = loss_distill.item()

            # Lever A (Inc13): leave-one-donor-out counterfactual consistency. For each sample with >=2
            # present donors, drop one present donor's peaks (attr==c) and require the OTHER present
            # donors' decoder logits to match the full-mixture prediction (detached target). Narrow:
            # loss only on the kept present donors. One extra forward/batch.
            loss_cf_v = 0.0
            if use_cf and reps_main is not None and (attr >= 0).any():
                Bc = tokens.size(0)
                present = (y > 0.5)
                valid_s = present.sum(1) >= 2                          # need >=2 donors to leave one out
                if valid_s.any():
                    probs = present.float().clone(); probs[~valid_s] = 1.0   # avoid all-zero rows
                    c_i = torch.multinomial(probs, 1).squeeze(1)             # (B,) donor to remove
                    remove = (attr == c_i.unsqueeze(1)) & mask.bool()        # its peaks
                    mask_cf = mask.bool() & ~remove
                    # never empty a sample (attention over an all-masked set -> NaN): revert those rows
                    empty = mask_cf.sum(1) == 0
                    mask_cf = torch.where(empty.unsqueeze(1), mask.bool(), mask_cf).to(mask.dtype)
                    _ = model(tokens, mask_cf)                               # cf forward (overwrites last_reps)
                    reps_cf = model.cls_decoder_module.last_reps             # (B,45,d)
                    keep = present.clone()
                    keep[torch.arange(Bc, device=DEVICE), c_i] = False       # the OTHER present donors
                    keep = keep & valid_s.unsqueeze(1)
                    # REPRESENTATION-level invariance (Veitch 2021 / FitNets): removing a co-donor must
                    # leave the OTHER donors' per-donor REPRESENTATIONS unchanged (cosine to the main reps,
                    # detached). Output-level consistency collapsed to a trivial invariance (train N5 0.10);
                    # the rep is still used for classification so rep-invariance can't go uninformative.
                    cos = F.cosine_similarity(reps_cf, reps_main, dim=-1)     # (B,45)
                    loss_cf = ((1.0 - cos) * keep.float()).sum() / keep.float().sum().clamp(min=1.0)
                    w_cf = cf_w * min(1.0, epoch / max(1, cf_ramp))
                    loss = loss + w_cf * loss_cf
                    loss_cf_v = loss_cf.item()

            # Lever C (Inc13) — FEATURE-LEVEL difficulty teaching (FitNets / Attention Transfer): make a
            # fainter version of the mixture (curriculum factor 0.7 -> diff_minf, floored so the minor stays
            # IDENTIFIABLE = valid) and teach the hard forward to FORM THE SAME per-donor representation as
            # the easy one (cosine, present donors). This teaches the METHOD ("extract this donor's evidence
            # even when faint / where to look"), NOT the result — output-label-on-hard taught the model to
            # over-call (just say yes), collapsing precision. No output loss on the hard forward. Scale-free
            # (cosine in [0,2]) so it can't dominate the cls loss.
            loss_diff_v = 0.0
            if use_diff and reps_main is not None and (attr >= 0).any():
                present = (y > 0.5)
                if present.any():
                    prog = min(1.0, epoch / max(1, diff_ramp))
                    factor = 0.7 - (0.7 - diff_minf) * prog
                    t_hard = make_harder(tokens, mask, attr, y, phi, factor)
                    _ = model(t_hard, mask)                                   # forward (overwrites last_reps)
                    reps_hard = model.cls_decoder_module.last_reps            # (B,45,d)
                    cos = F.cosine_similarity(reps_hard, reps_main, dim=-1)   # (B,45) vs easy reps (teacher)
                    loss_diff = ((1.0 - cos) * present.float()).sum() / present.float().sum().clamp(min=1.0)
                    loss = loss + diff_w * loss_diff
                    loss_diff_v = loss_diff.item()

            # Inc14 B-v2 — multi-label genotype co-membership head (CPI) + MLD distillation.
            loss_mlattr_v = loss_mld_v = 0.0
            if use_mlattr and "logits_mlattr" in out:
                owner = gather_owner(tokens)                              # (B,N,45) genotype ownership
                ml_tgt = owner * (y > 0.5).float().unsqueeze(1)           # present donors who own the allele
                vpk = mask.bool().unsqueeze(-1).float()                   # (B,N,1) valid peaks
                lm = out["logits_mlattr"]
                bce = F.binary_cross_entropy_with_logits(lm, ml_tgt, reduction="none",
                                                         pos_weight=ml_pos_weight)
                loss_mlattr = (bce * vpk).sum() / vpk.sum().clamp(min=1.0) / lm.size(-1)
                loss = loss + ml_attr_w * loss_mlattr
                loss_mlattr_v = loss_mlattr.item()
                # MLD distillation (per-class binary KD): teacher = multi-label readout (max over peaks),
                # detached; student = the per-donor decoder logits. Ramped (early head untrained).
                if use_ml_distill:
                    with torch.no_grad():
                        ap = torch.sigmoid(lm).masked_fill(~mask.bool().unsqueeze(-1), 0.0)
                        teacher_p = ap.max(dim=1).values.clamp(1e-4, 1 - 1e-4)   # (B,45) co-membership presence
                    w_eff = ml_distill_w * min(1.0, epoch / max(1, ml_distill_ramp))
                    loss_mld = F.binary_cross_entropy_with_logits(out["logits_cls"], teacher_p)
                    loss = loss + w_eff * loss_mld
                    loss_mld_v = loss_mld.item()

            # Inc14 A-v2 — additive-SUBTRACTION counterfactual + conditional MMD on the identity sub-rep.
            loss_addinv_v = 0.0
            if use_addinv and reps_main is not None and (attr >= 0).any():
                Ba = tokens.size(0); present = (y > 0.5)
                valid_s = present.sum(1) >= 2
                if valid_s.any():
                    owner = gather_owner(tokens)                          # (B,N,45)
                    w = owner * phi.clamp(min=0).unsqueeze(1) * present.float().unsqueeze(1)  # Mx*own*present
                    probs = present.float().clone(); probs[~valid_s] = 1.0
                    c_i = torch.multinomial(probs, 1).squeeze(1)          # (B,) donor to subtract (diversified)
                    wsum = w.sum(-1).clamp(min=1e-6)                      # (B,N) total deposit per peak
                    wc = w.gather(-1, c_i.view(-1, 1, 1).expand(-1, w.size(1), 1)).squeeze(-1)  # c's deposit
                    mult = (1.0 - (wc / wsum)).clamp(0.0, 1.0)            # additive subtraction multiplier
                    t_sub = subtract_height(tokens, mask, mult)          # peaks REMAIN (shared kept, private->0)
                    _ = model(t_sub, mask)                               # cf forward (overwrites last_reps)
                    reps_cf = model.cls_decoder_module.last_reps
                    keep = present.clone()
                    keep[torch.arange(Ba, device=DEVICE), c_i] = False    # the OTHER present donors
                    keep = keep & valid_s.unsqueeze(1)
                    # identity sub-rep = first id_dim channels (task uses full rep -> can't collapse; the
                    # free remainder absorbs the legitimate proportion change). reps_main is detached anchor.
                    loss_addinv = cond_slot_mmd(reps_main[:, :, :id_dim], reps_cf[:, :, :id_dim], keep)
                    w_ai = addinv_w * min(1.0, epoch / max(1, addinv_ramp))
                    loss = loss + w_ai * loss_addinv
                    loss_addinv_v = float(loss_addinv)

            # Inc15 — forward RECONSTRUCTION ("learn the process"): predicted donors (sigmoid logits_cls)
            # gated by predicted phi must additively reproduce the observed peak heights. Scale-free
            # (per-sample sum-normalized), restricted to peaks explainable by >=1 known donor (artifacts
            # excluded). Differentiable through logits_cls + phi -> shapes encoder+decoder jointly.
            loss_recon_v = 0.0
            if use_recon and "phi" in out:
                owner = gather_owner(tokens)                              # (B,N,45) copy number per peak
                w = torch.sigmoid(out["logits_cls"]) * out["phi"]        # (B,45) presence-gated proportion
                pred_h = (owner * w.unsqueeze(1)).sum(-1)                # (B,N) additive predicted height
                obs_h = torch.expm1(tokens[:, :, 2])                     # (B,N) observed RFU
                expl = (owner.sum(-1) > 0) & mask.bool()                 # peaks a known donor can explain
                em = expl.float()
                pn = pred_h / (pred_h * em).sum(1, keepdim=True).clamp(min=1e-6)   # remove template scale T
                on = obs_h / (obs_h * em).sum(1, keepdim=True).clamp(min=1e-6)
                w_rc = recon_w * min(1.0, epoch / max(1, recon_ramp)) if recon_ramp > 0 else recon_w
                loss_recon = (F.l1_loss(torch.log1p(1e3 * pn), torch.log1p(1e3 * on), reduction="none")
                              * em).sum() / em.sum().clamp(min=1.0)
                loss = loss + w_rc * loss_recon
                loss_recon_v = loss_recon.item()

            # Inc16 — ADDITIVE GRID recon over PRESENT+ABSENT (validated synth recipe): the predicted
            # sigma(logit)*phi*dosage must reproduce observed heights AND be ~0 where a predicted donor's
            # expected allele is ABSENT. The absent term is the damning-absence signal that penalizes decoys.
            if use_add_recon and "phi" in out:
                w_ar = torch.sigmoid(out["logits_cls"]) * out["phi"]               # (B,45)
                hhat = torch.exp(add_recon_logG) * torch.einsum('lwd,bd->blw', owner_lut, w_ar)  # (B,24,LUT_W)
                B_ar = tokens.shape[0]; mk_ar = mask.bool()
                loc_ar = tokens[:, :, 0].long().clamp(0, 23)
                ab_ar = (torch.round(tokens[:, :, 1] * 10).long() + ALLELE_OFF).clamp(0, owner_lut.size(1) - 1)
                obs_ar = torch.expm1(tokens[:, :, 2])
                obs_grid = torch.zeros_like(hhat)
                bidx_ar = torch.arange(B_ar, device=DEVICE).unsqueeze(1).expand(-1, tokens.shape[1])
                obs_grid.index_put_((bidx_ar[mk_ar], loc_ar[mk_ar], ab_ar[mk_ar]), obs_ar[mk_ar], accumulate=True)
                rl_ar = (torch.log1p(hhat) - torch.log1p(obs_grid)) ** 2           # log-scale (down-weights majors)
                poss_ar = add_recon_possible.unsqueeze(0).float()                  # (1,24,LUT_W) present+damning-absent
                loss_add_recon = (rl_ar * poss_ar).sum() / (poss_ar.sum() * B_ar).clamp(min=1.0)
                loss = loss + add_recon_w * loss_add_recon

            # Inc8 (F32): VICReg (var+cov [+inv]) on per-donor pooled encoded reps (N1-safe). attr=labels.
            loss_vic_v = 0.0
            if use_vicreg and "encoded" in out:
                v_var, v_cov, v_inv = vicreg_donor_regs(out["encoded"], attr.to(DEVICE), n_classes=45)
                vic = VIC_MU * v_var + VIC_NU * v_cov
                if use_vicreg_inv:
                    vic = vic + VIC_LAM * v_inv
                loss = loss + vicreg_w * vic; loss_vic_v = float(vic)

            # Inc9 B1 (source reconstruction, PIT-spirit): each present donor's rep must reconstruct the
            # pooled encoding of ITS OWN peaks -> the rep can't be absorbed into a dominant donor.
            if use_donor_recon and "donor_reps" in out:
                attr_d = attr.to(DEVICE); H_enc = out["encoded"].detach()
                B_, N_, d_ = H_enc.shape
                valid = (attr_d >= 0); idx = attr_d.clamp(min=0).unsqueeze(-1)   # (B,N,1)
                tgt = torch.zeros(B_, 45, d_, device=DEVICE)
                cnt = torch.zeros(B_, 45, 1, device=DEVICE)
                tgt.scatter_add_(1, idx.expand(-1, -1, d_), H_enc * valid.unsqueeze(-1))
                cnt.scatter_add_(1, idx, valid.unsqueeze(-1).float())
                has = cnt.squeeze(-1) > 0
                tgt = tgt / cnt.clamp(min=1.0)
                m = has.unsqueeze(-1).float()
                denom = m.sum().clamp(min=1.0) * d_
                loss_recon = (((out["donor_reps"] - tgt) ** 2) * m).sum() / denom
                loss = loss + donor_recon_w * loss_recon

            # Inc9 A4 (DN-DETR query denoising): supervise the noised-anchor pass to predict the true set.
            if use_qdn and out.get("logits_cls_dn") is not None:
                loss = loss + qdn_w * bce_cls(out["logits_cls_dn"], y)

            # Inc9 B2 (IFM, Robinson 2021): adversarially perturb the per-donor reps in the gradient
            # direction (feature the model leans on) and require correct prediction -> stops shortcut/
            # feature suppression (the dominant-donor shortcut). Re-scores last_reps, no decoder re-run.
            if use_ifm and getattr(model.cls_decoder_module, "last_reps", None) is not None:
                D = model.cls_decoder_module.last_reps
                g = torch.autograd.grad(loss_cls, D, retain_graph=True)[0]
                D_adv = D + ifm_eps * g.sign()
                loss = loss + ifm_w * bce_cls(model.cls_decoder_module.decode_score(D_adv), y)

            # Inc3 B: supervised-contrastive peak grouping by SOURCE DONOR (Khosla 2020). Pulls a
            # contributor's peaks together / pushes other donors' apart -> explicit deconvolution
            # grouping (doc2 §2). Subsample valid peaks to keep the O(M^2) loss cheap.
            loss_dc_v = 0.0
            if use_donorcon and "z_peak" in out:
                attr_d = attr.to(DEVICE)
                zc = out["z_peak"]; Bz, Sz, dz = zc.shape
                zf = zc.reshape(Bz * Sz, dz); af = attr_d.reshape(Bz * Sz)
                idx = torch.where(af >= 0)[0]
                if idx.numel() > 1024:
                    idx = idx[torch.randperm(idx.numel(), device=idx.device)[:1024]]
                if idx.numel() > 1:
                    loss_dc = supcon_loss(zf[idx], af[idx], tau=donorcon_tau)
                    loss = loss + donorcon_w * loss_dc
                    loss_dc_v = loss_dc.item()

            # Decoupled ordinal NOC contrast (§4b-A) on the discarded projection head. Kept SEPARATE
            # from the main loss so the §4b-C valve can control how its gradient meets the encoder.
            loss_rnc_v = 0.0
            rnc_term = None
            if use_rnc and "z_noc_proj" in out:
                loss_rnc = rank_n_contrast(out["z_noc_proj"], noc.float(), tau=rnc_tau)
                if rnc_fixed_w is not None:                       # fixed weight: bypass Kendall (F19)
                    rnc_term = rnc_fixed_w * loss_rnc
                else:                                            # Kendall homoscedastic weighting
                    rnc_term = torch.exp(-log_var_rnc) * loss_rnc + log_var_rnc
                loss_rnc_v = loss_rnc.item()

            # Inc17 — cross-level consistency (placed LAST, after every internal-state-reading aux block, so the
            # clean-view forward doesn't clobber last_reps/kl): clean view (higher filter, less artifact) is
            # supervised to the truth AND anchors the noisy view (stop-grad) -> discounts artifact-supported decoys.
            if use_xnoise:
                mask_clean = filter_by_height(tokens, mask_orig, xnoise_clean)
                out_clean = model(tokens, mask_clean)
                loss = loss + bce_cls(out_clean["logits_cls"], y)             # supervised clean view (true label)
                p_n = torch.sigmoid(out["logits_cls"]); p_c = torch.sigmoid(out_clean["logits_cls"]).detach()
                w_xn = xnoise_w * min(1.0, epoch / max(1, xnoise_ramp))       # ramp-up (anti-collapse, Pi-model)
                loss = loss + w_xn * F.mse_loss(p_n, p_c)

            optimizer.zero_grad()
            if use_and_mask:
                # Inc6 D-andmask: AND-mask ONLY the cls gradient across {low-NOC, high-NOC} envs; the
                # other losses (reject/noc/aux) backprop normally and are added underneath. Snapshot
                # the non-cls grad first, then overwrite p.grad = non-cls + AND-masked cls.
                if rnc_term is not None:
                    loss = loss + rnc_term
                other = loss - loss_cls                      # everything except the raw cls term
                other.backward(retain_graph=True)
                # include Kendall log-vars so their non-cls grad survives the second zero_grad (cls
                # doesn't depend on them → AND-mask leaves their grad = the snapshot).
                am_params = [p for p in pcgrad_params if p.requires_grad]
                base_grad = {p: (p.grad.detach().clone() if p.grad is not None else None) for p in am_params}
                nl = noc.clamp(1, 5)
                env_losses = []
                for em in ((nl <= 3), (nl >= 4)):
                    if em.any():
                        env_losses.append(bce_cls(out["logits_cls"][em], y[em]))
                optimizer.zero_grad()
                and_mask_backward(env_losses, am_params, base_grad=base_grad, tau=and_tau)
            elif rnc_term is not None and rnc_mode == "pcgrad":
                # valve 4: project the contrast gradient off the main-task gradient (no plain sum).
                pcgrad_backward(loss, rnc_term, pcgrad_params)
            else:
                # "shared" (grad reaches encoder) or "detach" (H already detached in the model) →
                # plain Kendall-weighted sum.
                if rnc_term is not None:
                    loss = loss + rnc_term
                loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = len(tokens)
            epoch_loss  += loss.item() * bs
            epoch_cls   += loss_cls.item() * bs
            epoch_rej   += loss_rej.item() * bs
            epoch_noc_l += loss_noc.item() * bs
            epoch_attr  += loss_attr_v * bs
            epoch_phi   += loss_phi_v * bs
            epoch_rnc   += loss_rnc_v * bs
            epoch_irm   += loss_irm_v * bs

        n = len(train_ds)
        epoch_loss /= n

        # Validation — select on oracle-EM (stable) but also track macro-F1
        val_f1, _, _, _ = evaluate_closed(model, val_loader, phi_inject_fn=None)
        val_em, val_macrec = evaluate_oracle_em(model, sel_loader, phi_inject_fn=None)   # GT phi NOT passed at val (would leak labels)
        val_sel = val_macrec                   # SELECTION = macro-over-NOC oracle Recall@k (non-saturating, graded)
        scheduler.step(val_sel)

        lr_now = optimizer.param_groups[0]["lr"]
        hrow = {
            "epoch": epoch,
            "loss": round(epoch_loss, 4),
            "val_macro_f1": round(val_f1, 4),
            "val_oracle_em": round(val_em, 4),
            "val_macro_recall": round(val_macrec, 4),
        }
        if use_rnc:   # F19 manipulation-check visibility: raw contrast loss + effective weight
            hrow["rnc_loss"] = round(epoch_rnc / n, 4)
            hrow["rnc_weight"] = round(float(rnc_fixed_w) if rnc_fixed_w is not None
                                       else float(torch.exp(-log_var_rnc).item()), 4)
        if use_irm:
            hrow["irm_loss"] = round(epoch_irm / n, 6)
        if use_log_per_noc:   # per-NOC val oracle trajectory (tests the per-NOC convergence-speed hypothesis)
            _pn = evaluate_per_noc_oracle(model, sel_loader, phi_inject_fn=None)
            hrow["val_per_noc_oracle"] = {str(k): round(v, 4) for k, v in _pn.items()}
        history.append(hrow)

        # ── EARLY-ABORT check (every 10 ep; only when enabled) ────────────────────────────────
        if early_abort and epoch % 10 == 0:
            pn = evaluate_per_noc_oracle(model, sel_loader, phi_inject_fn=None)
            best_dev_n5 = max(best_dev_n5, pn.get(5, 0.0))
            hrow["dev_oracle"] = {str(k): round(v, 4) for k, v in pn.items()}
            reason = None
            if epoch >= 15 and (pn.get(1, 1.0) < 0.95 or pn.get(2, 1.0) < 0.95):
                reason = f"guard_collapse (DEV oracle N1={pn.get(1, float('nan')):.2f} N2={pn.get(2, float('nan')):.2f})"
            elif epoch >= 15 and val_macrec < 0.50:
                reason = f"diverged (val_macrec={val_macrec:.2f})"
            elif epoch >= 50:
                if best_dev_n5 < ea_base_n5 - 0.05:
                    ea_strikes += 1
                    if ea_strikes >= 2:
                        reason = (f"n5_below_base (best DEV N5 oracle {best_dev_n5:.3f} < "
                                  f"{ea_base_n5:.2f}-0.05 for 2 checks)")
                else:
                    ea_strikes = 0
            if reason:
                abort_info = {"aborted": True, "reason": reason, "epoch": epoch,
                              "best_dev_n5": round(best_dev_n5, 4),
                              "dev_per_noc_oracle": {str(k): round(v, 4) for k, v in pn.items()}}
                print(f"  >>> EARLY-ABORT at ep {epoch}: {reason} -> stop, next lever")
                break

        if epoch % 10 == 0 or epoch == 1:
            aux_str = (f" attr={epoch_attr/n:.3f} phi={epoch_phi/n:.3f}") if use_aux else ""
            if use_rnc:
                _w = float(rnc_fixed_w) if rnc_fixed_w is not None else float(torch.exp(-log_var_rnc).item())
                aux_str += f" rnc={epoch_rnc/n:.3f}(w={_w:.2f})"
            aux_str += (f" irm={epoch_irm/n:.4f}") if use_irm else ""
            aux_str += (f" gcount={epoch_gate_count/n:.3f}") if use_gate_count else ""
            print(
                f"  Ep {epoch:3d} | loss={epoch_loss:.4f} "
                f"(cls={epoch_cls/n:.3f} rej={epoch_rej/n:.3f} noc={epoch_noc_l/n:.3f}{aux_str}) "
                f"| val_macrec={val_macrec:.4f} val_em={val_em:.4f} val_f1={val_f1:.4f} | lr={lr_now:.1e}"
            )

        if val_sel > best_f1:
            best_f1, best_epoch, patience_count = val_sel, epoch, 0
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stop at ep {epoch}  (best ep {best_epoch}, val macro-recall={best_f1:.4f})")
                break

    print(f"Training done in {time.time()-t0:.1f}s")

    # ── Test evaluation: cardinality-head k* decode (deployable) + oracle ───
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.eval()
    probs_list, card_list, yt_list, noc_list, phi_list, attr_list = [], [], [], [], [], []
    noc_proj_list = []; ord_list = []; v2_list = []; logit_list = []; gate_list = []
    with torch.no_grad():
        for tokens, mask, y, noc, *_ in test_loader:
            out = model(tokens.to(DEVICE), mask.to(DEVICE))
            probs_list.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
            logit_list.append(out["logits_cls"].cpu().numpy())          # raw logits for the phi-rerank LOP
            card_list.append(out["logits_card"].cpu().numpy())
            yt_list.append(y.numpy()); noc_list.append(noc.numpy())
            if "phi" in out:                                  # Inc2 aux: dump phi for real-φ eval
                phi_list.append(out["phi"].cpu().numpy())
            if "logits_attr" in out:                          # Inc2 aux: dump per-peak attribution
                attr_list.append(out["logits_attr"].argmax(-1).cpu().numpy().astype(np.int16))
            if "z_noc_proj" in out:                           # F19: dump for the ordinal-structure probe
                noc_proj_list.append(out["z_noc_proj"].cpu().numpy())
            if "logits_card_ord" in out:                      # Inc3 V1-3: CORN ordinal count logits
                ord_list.append(out["logits_card_ord"].cpu().numpy())
            if "logits_count_v2" in out:                      # paper-proof CORN count head (B,4)
                v2_list.append(out["logits_count_v2"].cpu().numpy())
            if use_gate_count and "gate_logit" in out:        # §3.2 learned gate-count: Σ P(slot active)
                gate_list.append(torch.sigmoid(out["gate_logit"]).sum(-1).cpu().numpy())
    P_te = np.concatenate(probs_list); card_te = np.concatenate(card_list)
    y_te_true = np.concatenate(yt_list); noc_te = np.concatenate(noc_list)
    L_te = np.concatenate(logit_list)                                   # raw test logits (phi-rerank)

    # Val probs for post-hoc cardinality (+ raw logits for the phi-rerank alpha tuning)
    vp_list, vy_list, vlog_list = [], [], []
    with torch.no_grad():
        for tokens, mask, y, *_ in val_loader:
            _lc = model(tokens.to(DEVICE), mask.to(DEVICE))["logits_cls"].cpu().numpy()
            vlog_list.append(_lc); vp_list.append(1.0 / (1.0 + np.exp(-_lc)))
            vy_list.append(y.numpy())
    P_va = np.concatenate(vp_list); y_va = np.concatenate(vy_list); L_va = np.concatenate(vlog_list)

    # joint card-head estimate; Inc3 V1-3: blend/replace with the CORN ordinal count if present
    if ord_list:
        ord_te = np.concatenate(ord_list)
        ord_p = corn_probs(torch.from_numpy(ord_te), 5).numpy()         # (N,5) P(NOC=k)
        card_p = np.exp(card_te - card_te.max(1, keepdims=True))
        card_p = card_p / card_p.sum(1, keepdims=True)
        comb = ord_p if ord_replace else 0.5 * (card_p + ord_p)         # V3 replace | V1/V2 ensemble
        k_card = comb.argmax(1) + 1
    else:
        k_card = card_te.argmax(1) + 1
    # DEPLOYABLE phi-rerank (phi_rerank.py; flag-gated, default off → rank_te = P_te, decode unchanged). Reranks
    # the per-donor RANKING via a logarithmic opinion pool with an INDEPENDENT EM mixture-proportion
    # deconvolution (alpha tuned on val, C6-clean). On replicate data (--replicates) the pooled tokens give
    # deconv_phi a replicate-COMBINED phi (EFMrep-style pooling) — the math channel sees the richer data too.
    rank_te = P_te; rank_va = P_va
    if cfg.get("phi_rerank", False) and donor_geno is not None:
        import phi_rerank as _pr
        _dg = donor_geno.cpu().numpy(); _dgm = donor_geno_mask.cpu().numpy()
        _PHv = _pr.deconv_phi(val_ds.tokens.numpy(), val_ds.mask.numpy(), _dg, _dgm)
        _PHt = _pr.deconv_phi(test_ds.tokens.numpy(), test_ds.mask.numpy(), _dg, _dgm)
        _alpha = _pr.tune_alpha(L_va, _PHv, y_va, val_ds.noc.numpy())
        rank_te = _pr.rerank_scores(L_te, _PHt, _alpha)
        rank_va = _pr.rerank_scores(L_va, _PHv, _alpha)
        print(f"phi_rerank ON: val-tuned alpha={_alpha} — LOP deconvolution rerank applied to test ranking")
    elif cfg.get("phi_rerank", False):
        print("[WARN] --phi_rerank requested but donor_geno is None -> skipped (ranking unchanged)")
    # COUNT after rerank: fit the post-hoc count on the RERANKED score profile (count the thing you
    # threshold) when enabled + reranking; else the legacy count on the raw prob-profile.
    if cfg.get("count_on_rerank", False) and cfg.get("phi_rerank", False) and donor_geno is not None:
        k_post = posthoc_cardinality_rank(P_va, rank_va, y_va, P_te, rank_te)
        print("count_on_rerank ON: post-hoc count fit on the RERANKED score + prob-profile (consistent with the decode)")
    else:
        k_post = posthoc_cardinality(P_va, y_va, P_te)       # legacy: count on raw probs
    em_card = per_noc_em(y_te_true, topk_decode(rank_te, k_card), noc_te)
    em_post = per_noc_em(y_te_true, topk_decode(rank_te, k_post), noc_te)
    oracle = per_noc_em(y_te_true, topk_decode(rank_te, noc_te), noc_te)
    # paper-proof learned CORN count head (noc_head_v2): decode k via rank-consistent corn_probs
    em_v2 = None; k_v2 = None
    if v2_list:
        v2_p = corn_probs(torch.from_numpy(np.concatenate(v2_list)), 5).numpy()   # (N,5) P(NOC=k)
        k_v2 = v2_p.argmax(1) + 1
        em_v2 = per_noc_em(y_te_true, topk_decode(rank_te, k_v2), noc_te)
    # §3.2 learned gate-count decode: k = round(Σ P(slot active)), clamped to [1,5]. Uses the SAME rank_te
    # as every other decoder (differs ONLY in the count k), so this isolates the gate-count quality.
    em_gs = None; k_gs = None
    if gate_list:
        gate_sum_te = np.concatenate(gate_list)
        k_gs = np.clip(np.rint(gate_sum_te), 1, 5).astype(int)
        em_gs = per_noc_em(y_te_true, topk_decode(rank_te, k_gs), noc_te)
    # DEPLOYABLE decoder is PRE-DECLARED post_hoc (the C4/C5-validated robust count) — NOT whichever scores
    # best on TEST. The old `max(decoders, key=test_em)` was selection-on-test / HARKing: optimistically biased
    # (Cawley & Talbot, JMLR 2010) and against the project's own C6/C7 rule (select on in-silico DEV, eval test
    # ONCE). joint_card / noc_v2 stay computed + reported below as DIAGNOSTICS; adopting one over post_hoc must
    # be decided on the in-silico dev oracle (measure_insilico_oracle.py) + seeds, never on this test number.
    decoders = [("joint_card", em_card, k_card), ("post_hoc", em_post, k_post)]
    if em_v2 is not None:
        decoders.append(("noc_v2", em_v2, k_v2))
    if em_gs is not None:
        decoders.append(("gate_sum", em_gs, k_gs))
    _diag = max(decoders, key=lambda c: c[1][0])
    if _diag[0] != "post_hoc":
        print(f"  [diagnostic only] {_diag[0]} scores higher on TEST (EM {_diag[1][0]:.4f} vs post_hoc "
              f"{em_post[0]:.4f}) — VALIDATE on in-silico dev + seeds before adopting; NOT selected as headline.")
    decode_name, _best_em, k_use = "post_hoc", em_post, k_post
    use_post = True
    y_te_pred = topk_decode(rank_te, k_use)
    te_metrics = full_report(y_te_true, y_te_pred, noc_te,
                             f"SET TRANSFORMER — TEST ({decode_name} card decode)")
    print(f"  {'decode':<14}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    _rows = [("oracle", oracle), ("joint-card", em_card), ("post-hoc", em_post)]
    if em_v2 is not None: _rows.append(("noc_v2", em_v2))
    if em_gs is not None: _rows.append(("gate_sum", em_gs))
    for nm, r in _rows:
        print(f"  {nm:<14}" + "".join(f"{x:>7.3f}" for x in r))
    oracle_em = oracle[0]; card_noc_acc = float((np.clip(k_use, 1, 5) == noc_te).mean())
    best_thresh = None

    # ── Open-set reject AUROC ─────────────────────────────────────────────
    print("\n-- Reject head AUROC " + "-"*39)
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        # Closed-set test → label 0
        for tokens, mask, *_ in test_loader:
            rej = torch.sigmoid(model(tokens.to(DEVICE), mask.to(DEVICE))["logit_reject"])
            scores.append(rej.cpu().numpy())
            labels.append(np.zeros(len(tokens)))
        # Open-set → label 1
        open_loader = DataLoader(open_ds, batch_size=256, shuffle=False)
        for o_tok, o_mask in open_loader:
            rej = torch.sigmoid(model(o_tok.to(DEVICE), o_mask.to(DEVICE))["logit_reject"])
            scores.append(rej.cpu().numpy())
            labels.append(np.ones(len(o_tok)))

    scores = np.concatenate(scores).ravel()
    labels = np.concatenate(labels)
    try:
        auroc = roc_auc_score(labels, scores)
        print(f"  Reject AUROC (closed vs open): {auroc:.4f}")
    except Exception as e:
        auroc = None
        print(f"  AUROC error: {e}")

    # ── Save ──────────────────────────────────────────────────────────────
    np.save(results_dir / "y_test_pred.npy", y_te_pred)
    np.save(results_dir / "y_test_true.npy", y_te_true)
    if phi_list:                                              # Inc2: phi predictions on real test
        np.save(results_dir / "phi_pred_test.npy", np.concatenate(phi_list))
    if attr_list:                                             # Inc2: per-peak attribution predictions
        np.save(results_dir / "attr_pred_test.npy", np.concatenate(attr_list))
    if noc_proj_list:                                         # F19: z_noc_proj for ordinal-structure probe
        np.save(results_dir / "z_noc_proj_test.npy", np.concatenate(noc_proj_list))

    # per-NOC breakdown for each decode (list = [overall, NOC1..NOC5]); NaN-safe dicts so the
    # seed-aggregator can report per-NOC oracle/EM with CIs (NOC2 etc. — the load-bearing strata).
    def _per_noc_dict(lst):
        return {str(j): (None if np.isnan(lst[j]) else round(float(lst[j]), 4)) for j in range(1, 6)}

    out_dict = {
        "model":             "set_transformer",
        "config":            cfg,
        "best_val_macro_recall": round(best_f1, 4),
        "best_epoch":        best_epoch,
        "decode":            decode_name,
        "em_joint_card":     round(float(em_card[0]), 4),
        "em_post_hoc":       round(float(em_post[0]), 4),
        "em_noc_v2":         (round(float(em_v2[0]), 4) if em_v2 is not None else None),
        "em_gate_sum":       (round(float(em_gs[0]), 4) if em_gs is not None else None),
        "gate_count_acc":    (round(float((np.clip(k_gs, 1, 5) == noc_te).mean()), 4) if k_gs is not None else None),
        "oracle_em":         round(float(oracle_em), 4),
        "card_noc_acc":      round(card_noc_acc, 4),
        "reject_auroc":      float(auroc) if auroc is not None else None,
        "per_noc":           te_metrics.get("per_noc", {}),
        "per_noc_oracle":    _per_noc_dict(oracle),
        "per_noc_joint_card": _per_noc_dict(em_card),
        "per_noc_post_hoc":  _per_noc_dict(em_post),
        "per_noc_noc_v2":    (_per_noc_dict(em_v2) if em_v2 is not None else None),
        "per_noc_gate_sum":  (_per_noc_dict(em_gs) if em_gs is not None else None),
        "history":           history,
        "test":              {k: v for k, v in te_metrics.items() if k != "per_noc"},
        "early_abort":       abort_info,   # None unless a fail-fast rule fired
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(out_dict, f, indent=2)

    print(f"\nSaved -> {results_dir}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--n_isab",     type=int,   default=None)
    parser.add_argument("--d_model",    type=int,   default=None)
    parser.add_argument("--m_inducing", type=int,   default=None)
    parser.add_argument("--epochs",   type=int,   default=None)
    parser.add_argument("--lr",       type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--cls_decoder", type=str, default=None,
                        choices=["pooled", "per_donor", "additive", "dsmil", "sos", "aslot", "spen"])
    parser.add_argument("--dec_aggr", type=str, default=None,
                        choices=["sparsemax", "entmax15", "entmax_temp"],
                        help="Inc12: per_donor decoder key-aggregation (softer than sparsemax reads diffuse faint evidence)")
    parser.add_argument("--decoder_source", type=str, default=None,
                        choices=["encoded", "raw", "local"])
    parser.add_argument("--loss", type=str, default=None,
                        choices=["bce", "asl"])
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--n_token_feats", type=int, default=None,
                        help="3=baseline token; 8=enriched relational token (needs tokens8_*.npy)")
    parser.add_argument("--encoder", type=str, default=None, choices=["isab", "isab++"],
                        help="isab (LayerNorm) | isab++ (SetNorm + clean-path, Set Transformer++)")
    parser.add_argument("--dec_layers", type=int, default=None, help="per_donor decoder layers (Q2L=2)")
    parser.add_argument("--num_embed", type=str, default=None, choices=["raw", "periodic"],
                        help="raw=scalar→shared Linear | periodic=per-feature PLR embedding (Gorishniy 2022)")
    parser.add_argument("--n_freq", type=int, default=None, help="periodic embedding frequencies per feature")
    parser.add_argument("--d_num_emb", type=int, default=None, help="periodic embedding dim per feature")
    parser.add_argument("--periodic_sigma", type=float, default=None, help="periodic frequency init std")
    parser.add_argument("--aux_heads", action="store_true",
                        help="Inc2 §5: privileged aux supervision (allele→donor attr + phi, Kendall-weighted)")
    parser.add_argument("--lupi_phys", action="store_true",
                        help="Inc-LUPI: privileged PHYSICAL heads (degradation β, clean mu+var Gaussian β-NLL, drop-in)")
    parser.add_argument("--lupi_degr", action="store_true", help="LUPI: train degradation-β regression head")
    parser.add_argument("--lupi_mu", action="store_true", help="LUPI: train clean-mu denoise head (β-NLL, Seitzer 2022)")
    parser.add_argument("--lupi_var", action="store_true", help="LUPI: supervise the mu-head variance to true var")
    parser.add_argument("--lupi_dropin", action="store_true", help="LUPI: train per-peak drop-in (phantom) head")
    parser.add_argument("--noc_contrast", action="store_true",
                        help="Inc2 §4b: decoupled ordinal NOC contrast (Rank-N-Contrast on separate pool+projection)")
    parser.add_argument("--rnc_tau", type=float, default=None, help="Rank-N-Contrast temperature (default 2.0)")
    parser.add_argument("--noc_contrast_mode", type=str, default=None, choices=["shared", "detach", "pcgrad"],
                        help="§4b-C valve: shared (grad->encoder, orig 2c) | detach (pma_noc pools H.detach) | "
                             "pcgrad (project contrast grad off main-task grad)")
    parser.add_argument("--rnc_fixed_weight", type=float, default=None,
                        help="F19 fix: fixed RNC weight (bypass Kendall auto-down-weighting)")
    parser.add_argument("--sparse_attn", action="store_true",
                        help="Inc2d §7: sparsemax per-donor decoder (zero attention to spurious peaks)")
    parser.add_argument("--irm", action="store_true",
                        help="Inc2d §7: IRM NOC-environment invariance penalty (suppress combo shortcut)")
    parser.add_argument("--irm_lambda", type=float, default=None, help="IRM penalty weight (default 1.0)")
    parser.add_argument("--irm_anneal", type=int, default=None,
                        help="IRM penalty warmup epochs 0->lambda (default 10; required for stability)")
    parser.add_argument("--d_proj", type=int, default=None, help="contrastive projection dim (default 64)")
    # Inc3 levers (each = one published method; see reports/design_increment3_levers.md)
    parser.add_argument("--geno_query", action="store_true",
                        help="Inc3 A: reference-genotype-conditioned donor queries (DAB-DETR/Q2L + LUPI)")
    parser.add_argument("--ref_match", action="store_true",
                        help="Inc10: add a discriminability-weighted reference-allele match score to each per-donor "
                             "logit (AFIA/explicit-matching; credits decisive panel-rare alleles). Needs donor_geno.")
    parser.add_argument("--ref_match_learn", action="store_true",
                        help="Inc10 V2: LEARN the per-(donor,allele) discriminability weight (init 1/panel-rarity); "
                             "default (V1) keeps it fixed at panel-rarity (= the verified ensemble, internalised).")
    parser.add_argument("--donor_contrast", action="store_true",
                        help="Inc3 B: supervised-contrastive peak grouping by source donor (Khosla 2020)")
    parser.add_argument("--donor_contrast_weight", type=float, default=None, help="SupCon weight (default 0.1)")
    parser.add_argument("--noc_ord_head", action="store_true",
                        help="Inc3 V1-3: CORN ordinal count head on its own encoder pool (Shi/Cao/Raschka 2023)")
    parser.add_argument("--noc_ord_detach", action="store_true",
                        help="Inc3 V1: pma_card pools H.detach() (count pool decoupled from ID)")
    parser.add_argument("--noc_ord_replace", action="store_true",
                        help="Inc3 V3: count = CORN-on-encoder only (drop ID-derived card head at decode)")
    parser.add_argument("--noc_ord_weight", type=float, default=None, help="CORN count loss weight (default beta)")
    # ── Increment 6 — root anti-memorization levers (combo-overfit wall, F29/F30) ──
    parser.add_argument("--mask_peaks", type=float, default=None,
                        help="D: per-batch input-peak dropout prob during TRAIN (anti combo-memorization). "
                             "0=off. ~0.15 typical. Decoder can't template-match a memorized peak set.")
    parser.add_argument("--mask_peaks_min", type=int, default=None,
                        help="min valid peaks kept per sample when --mask_peaks (default 8; keeps NOC identifiable)")
    parser.add_argument("--and_mask", action="store_true",
                        help="D: ILC AND-mask (Parascandolo 2020) on the cls-loss gradient across "
                             "low-NOC vs high-NOC environments — keep only feature directions that help BOTH.")
    parser.add_argument("--and_mask_thresh", type=float, default=None,
                        help="AND-mask sign-agreement threshold tau in [0,1] (default 0.0 = pure agreement; "
                             "fraction of env consensus required to pass a gradient component)")
    parser.add_argument("--sam", action="store_true",
                        help="4a: Sharpness-Aware Minimization (Foret 2021) — two-step, seek flat minima")
    parser.add_argument("--sam_rho", type=float, default=None, help="SAM neighbourhood size rho (default 0.05)")
    parser.add_argument("--vib", action="store_true",
                        help="2f: variational information bottleneck on per-donor latents (disentangle)")
    parser.add_argument("--vib_weight", type=float, default=None, help="VIB KL weight (default 1e-3)")
    parser.add_argument("--masked_pretrain_epochs", type=int, default=None,
                        help="3e: N epochs of self-supervised masked-feature reconstruction BEFORE supervised training")
    parser.add_argument("--mask_pretrain_ratio", type=float, default=None,
                        help="fraction of valid peaks masked during masked pretrain (default 0.3)")
    parser.add_argument("--minor_weight", action="store_true",
                        help="Inc6 minor-sensitivity: cost-weight POSITIVE donors by capped inverse mixture-"
                             "fraction clip(phi_ref/frac,1,cap) — rescues low-phi minor contributors (verified "
                             "root cause of the N5 wall) WITHOUT chasing the phi<0.05 noise floor (the cap)")
    parser.add_argument("--minor_phi_ref", type=float, default=None, help="reference mixture-fraction (default 0.2)")
    parser.add_argument("--minor_cap", type=float, default=None, help="max positive up-weight (default 4.0)")
    parser.add_argument("--mass_pool", action="store_true",
                        help="Inc7 (F31): mass-preserving inducing-point compression (scaled weighted-SUM, "
                             "Fischer&Gartner 2024 arXiv 2407.04170) so the many tall MAJOR peaks stop washing "
                             "the few faint MINOR peaks in the encoded H — the root of the N5 oracle wall. "
                             "Requires --encoder isab++.")
    # ── Increment 8 — VICReg (arXiv 2105.04906) on per-donor reps (F32; verbatim, N1-safe) ──
    parser.add_argument("--vicreg", action="store_true",
                        help="Inc8 V1: VICReg variance+covariance on per-donor pooled reps (anti-collapse "
                             "decorrelation = de-smooth the combo carrier; Bardes 2022). N1-safe (variance term).")
    parser.add_argument("--vicreg_inv", action="store_true",
                        help="Inc8 V2: + VICReg invariance term (pull the SAME donor's rep together across the "
                             "combos it appears in — the multi-view = same donor in different combos). Needs --vicreg.")
    parser.add_argument("--vicreg_weight", type=float, default=None,
                        help="overall VICReg aux scale (default 0.04; paper ratios var/cov/inv=25/1/25 kept inside)")
    parser.add_argument("--out_subdir", type=str, default=None,
                        help="results subdir name (default set_transformer)")
    parser.add_argument("--seed", type=int, default=None,
                        help="reproducibility seed (default 42); sweep for CIs on small NOC strata")
    parser.add_argument("--gamma_neg", type=float, default=None,
                        help="ASL negative focusing gamma (default 4.0). Lower (1-2) = stop over-suppressing "
                             "rare/faint positives (decoder under-read lever; cf. long-tail multi-label loss).")
    parser.add_argument("--cls_loss", type=str, default=None, choices=["asl", "ldam", "dbloss", "bal"],
                        help="Inc9 decoder under-read lever: ldam=LDAM-DRW (A1), dbloss=Distribution-Balanced (A2), "
                             "bal=Balanced-Asymmetric (A3). Overrides --loss for the cls head.")
    parser.add_argument("--ldam_drw_epoch", type=int, default=None, help="LDAM deferred-reweight start epoch (default 20)")
    parser.add_argument("--bal_gamma_pos", type=float, default=None, help="BAL positive focusing gamma (default 1.0)")
    parser.add_argument("--attn_sink", type=int, default=None,
                        help="Inc9 B4: # null sink keys in the per-donor decoder (softmax-1 / quiet attention; "
                             "1 = literal softmax-1). Stops dominant peaks saturating the donor-query attention.")
    parser.add_argument("--donor_recon", action="store_true",
                        help="Inc9 B1: per-donor source-reconstruction aux (each rep reconstructs its own peaks' "
                             "pooled encoding) = anti-absorption / PIT-spirit on the encoder.")
    parser.add_argument("--donor_recon_weight", type=float, default=None, help="B1 recon loss weight (default 0.1)")
    parser.add_argument("--query_denoise", action="store_true",
                        help="Inc9 A4: DN-DETR query denoising — extra train pass with NOISED genotype-anchored "
                             "queries supervised to the true set (needs --geno_query for the anchor).")
    parser.add_argument("--qdn_noise", type=float, default=None, help="A4 query noise stddev (default 1.0)")
    parser.add_argument("--qdn_weight", type=float, default=None, help="A4 denoise loss weight (default 0.5)")
    parser.add_argument("--ifm", action="store_true",
                        help="Inc9 B2: Implicit Feature Modification — adversarially perturb per-donor reps and "
                             "require correct prediction (stops the dominant-donor shortcut / feature suppression).")
    parser.add_argument("--ifm_eps", type=float, default=None, help="B2 adversarial step size (default 0.05)")
    parser.add_argument("--ifm_weight", type=float, default=None, help="B2 IFM loss weight (default 0.5)")
    parser.add_argument("--nc_attn", type=str, default=None, choices=["none", "mab0", "both"],
                        help="Inc11 (F35b): NON-COMPETITIVE sigmoid encoder attention (Ramapuram et al. 2024, "
                             "arXiv 2409.04431). 'mab0' = sigmoid in the inducing-pool step only (the localized "
                             "lever: stops tall majors washing faint minors via softmax-over-peaks); 'both' = "
                             "mab0+mab1. Requires --encoder isab++.")
    parser.add_argument("--nc_learnable_bias", action="store_true",
                        help="Inc11: add a learnable per-block offset to the sigmoid-attention −log(n) bias.")
    parser.add_argument("--early_abort", action="store_true",
                        help="fail-fast lever screening: periodic DEV per-NOC oracle check kills a clearly-"
                             "failing run early (guard-collapse / diverge / N5-below-base) so the runner moves on.")
    parser.add_argument("--early_abort_base_n5", type=float, default=None,
                        help="base DEV N5 oracle to beat for the trajectory rule (default 0.59 = inc2_2d_sparse).")
    # Increment 13 — Lever B (attr->decoder distillation, LUPI). Needs --aux_heads.
    parser.add_argument("--distill_attr", action="store_true",
                        help="Lever B: distill combo-invariant attr_head soft-vote into the per-donor decoder")
    parser.add_argument("--distill_weight", type=float, default=None, help="Lever B distill loss weight (default 0.5)")
    parser.add_argument("--distill_ramp", type=int, default=None, help="Lever B ramp epochs 0->weight (default 20)")
    # Increment 13 — Lever A (counterfactual leave-one-donor-out combo-invariance). Needs --aux_heads.
    parser.add_argument("--cf_invariance", action="store_true",
                        help="Lever A: leave-one-donor-out counterfactual consistency (combo-invariance)")
    parser.add_argument("--cf_weight", type=float, default=None, help="Lever A consistency weight (default 1.0)")
    parser.add_argument("--cf_ramp", type=int, default=None, help="Lever A ramp epochs 0->weight (default 10)")
    # Increment 13 — Lever C (difficulty-ladder, easy-teaches-hard). Needs --aux_heads.
    parser.add_argument("--diff_ladder", action="store_true",
                        help="Lever C: faint-minor difficulty-ladder (true-label + easy->hard consistency, curriculum)")
    parser.add_argument("--diff_weight", type=float, default=None, help="Lever C weight (default 0.5)")
    parser.add_argument("--diff_min_factor", type=float, default=None, help="Lever C hardest height-scale factor (default 0.5, keeps minor identifiable)")
    parser.add_argument("--diff_ramp", type=int, default=None, help="Lever C curriculum epochs 0.7->min_factor (default 30)")
    # Increment 14 — B-v2 (multi-label genotype co-membership head + MLD distillation). Needs donor_geno.
    parser.add_argument("--ml_attr", action="store_true",
                        help="Inc14 B-v2: per-peak multi-label genotype co-membership head (CPI), shapes H")
    parser.add_argument("--ml_attr_weight", type=float, default=None, help="B-v2 multi-label BCE aux weight (default 1.0)")
    parser.add_argument("--ml_distill", action="store_true",
                        help="Inc14 B-v2: distil the multi-label teacher into the decoder (per-class binary KD / MLD)")
    parser.add_argument("--ml_distill_weight", type=float, default=None, help="B-v2 MLD distill weight (default 0.5)")
    parser.add_argument("--ml_distill_ramp", type=int, default=None, help="B-v2 MLD ramp epochs (default 20)")
    parser.add_argument("--ml_pos_weight", type=float, default=None, help="B-v2 BCE positive weight for sparse co-membership (default 20)")
    # Increment 14 — A-v2 (additive-subtraction counterfactual + conditional MMD on identity sub-rep). Needs donor_geno.
    parser.add_argument("--add_invar", action="store_true",
                        help="Inc14 A-v2: additive-subtraction counterfactual invariance (forensic-correct, anti-collapse)")
    parser.add_argument("--add_invar_weight", type=float, default=None, help="A-v2 conditional-MMD weight (default 1.0)")
    parser.add_argument("--add_invar_ramp", type=int, default=None, help="A-v2 ramp epochs (default 10)")
    parser.add_argument("--id_dim", type=int, default=None, help="A-v2 identity sub-rep dim (default d_model/2)")
    # Increment 15 — "learn the process" forward reconstruction (analysis-by-synthesis). Needs donor_geno.
    parser.add_argument("--recon", action="store_true",
                        help="Inc15: forward-reconstruction — predicted donors+phi must explain observed heights")
    parser.add_argument("--recon_weight", type=float, default=None, help="Inc15 reconstruction loss weight (default 0.3)")
    parser.add_argument("--recon_ramp", type=int, default=None, help="Inc15 reconstruction ramp epochs (default 0=immediate)")
    # Increment 16 — ADDITIVE GRID recon (Wiedemer compositional-gen condition; validated in synth_additive_ceiling.py).
    parser.add_argument("--add_recon", action="store_true",
                        help="additive grid log-recon over PRESENT+ABSENT allele positions (the validated recipe; "
                             "Inc15 recon only scores PRESENT peaks): sigma(logit)*phi*dosage must match observed "
                             "heights AND predict ~0 where a predicted donor's expected allele is ABSENT (damning-"
                             "absence penalizes decoy inclusion). Shapes encoder+decoder. Needs donor_geno + aux_heads.")
    parser.add_argument("--add_recon_weight", type=float, default=None, help="Inc16 additive grid recon weight (default 0.1)")
    # Increment 17 — NOISE-SPECTRUM (diffusion-classifier multi-noise + Pi/FixMatch consistency). The real faint
    # band is ~91% ARTIFACT; true alleles are noise-ROBUST, decoy fodder is noise-SENSITIVE -> learn invariance.
    parser.add_argument("--noise_spectrum", action="store_true",
                        help="Inc17: height-stratified multi-noise aug — each batch at a random filter level F "
                             "(drop peaks < F RFU). Teaches donor-set invariance across the artifact-noise spectrum.")
    parser.add_argument("--noise_spectrum_levels", type=str, default=None, help="comma RFU filter levels (default '0,15,30,45')")
    parser.add_argument("--xnoise_consistency", action="store_true",
                        help="Inc17: cross-level consistency — push noisy-view prediction toward CLEAN-view "
                             "(stop-grad anchor, ramp-up); discounts artifact-supported decoys (Pi-model/FixMatch).")
    parser.add_argument("--xnoise_weight", type=float, default=None, help="Inc17 consistency weight (default 1.0)")
    parser.add_argument("--xnoise_ramp", type=int, default=None, help="Inc17 consistency ramp epochs (default 15)")
    parser.add_argument("--xnoise_clean", type=float, default=None, help="Inc17 anchor clean filter level RFU (default 30)")
    parser.add_argument("--warm_start", type=str, default=None,
                        help="path to a results/<run>/best_model.pt to warm-start (whole-pipeline fine-tune probe)")
    # Increment 18 V1 — phi injection into per-donor decoder queries (FiLM-style).
    # GT phi used as privileged signal at train time; EM-uniform phi injected at inference (probe_phi_rerank.py).
    parser.add_argument("--phi_inject", action="store_true",
                        help="Inc18 V1: inject phi into per-donor decoder queries (privileged GT phi train, EM phi infer)")
    # Increment 18 V2 — genotype-constrained soft attr head: mask non-carrier logits to -inf at INFERENCE
    # so attr logits only score genotype-compatible donors per peak. Training uses standard unmasked CE
    # (handles stutter-peak attr labels that would be masked to -inf during training otherwise).
    parser.add_argument("--soft_geno_attr", action="store_true",
                        help="Inc18 V2: private peaks hard-assign, shared peaks soft-mask at inference; needs --aux_heads + donor_geno")
    parser.add_argument("--feas_filter", action="store_true",
                        help="Inc18 V3: filter infeasible peaks (0 carriers in reference panel) before encoder — train+inference")
    parser.add_argument("--soft_attr_label", action="store_true",
                        help="Inc18 V4: EuroForMix phi*CN soft labels for attr CE (no NaN on stutter; "
                             "proven +0.048 decoyAUC +0.156 N5 model oracle at small scale). Needs --aux_heads + donor_geno.")
    parser.add_argument("--set_of_set", action="store_true",
                        help="SoS: split mixture peaks into private (n_carriers==1) and shared (n_carriers>1) sets "
                             "BEFORE the ISAB encoder; encode each set independently (same weights, lite), then merge. "
                             "Prevents explaining-away of minor-donor private allele evidence. Needs donor_geno.")
    # ── Adaptive Slot Attention (CoSA+GSANet+MESH+AdaSlot; use --cls_decoder aslot) ──
    parser.add_argument("--n_slot_iters", type=int, default=None,
                        help="aslot: MESH slot-attention iterations (default 3)")
    parser.add_argument("--ot_eps", type=float, default=None,
                        help="aslot: Sinkhorn OT regularization ε — lower=sharper (default 0.05)")
    parser.add_argument("--ot_iters", type=int, default=None,
                        help="aslot: Sinkhorn normalization steps per MESH iter (default 5)")
    parser.add_argument("--gumbel_temp", type=float, default=None,
                        help="aslot: AdaSlot concrete-Bernoulli temperature (default 1.0)")
    parser.add_argument("--noc_head_v2", action="store_true",
                        help="paper-proof learned NOC head: CORN ordinal regression on TRUE noc, reading "
                             "mass-preserving slot-mass + MAC physical features + prob-profile (default off).")
    parser.add_argument("--noc_v2_weight", type=float, default=None,
                        help="noc_head_v2 CORN-on-true-NOC loss weight (default = beta_card)")
    parser.add_argument("--gate_count", action="store_true",
                        help="sec3.2 LEARNED gate-count: differentiable sum(gate)~=NOC consistency on the AdaSlot "
                             "existence gate (gradient to encoder, unlike the DETACHED noc_head_v2). count=sum(gate) "
                             "becomes learned, not post-hoc; decode compares vs joint/post-hoc/noc_v2. aslot only.")
    parser.add_argument("--gate_count_weight", type=float, default=None,
                        help="gate_count sum(gate)~=NOC smooth-L1 loss weight (default = beta = NOC loss weight)")
    parser.add_argument("--log_per_noc", action="store_true",
                        help="log per-NOC val oracle EM to history each epoch (val_per_noc_oracle) — inspect "
                             "per-NOC convergence speed (does N5 peak later than N1-N4?). Diagnostic; default off.")
    parser.add_argument("--gate_mass", action="store_true",
                        help="sec3.2.1 mass-aware gate: feed grad-enabled slot-mass into the AdaSlot existence "
                             "gate (ReZero zero-init, no-op start). aslot only; pairs with --gate_count.")
    parser.add_argument("--gate_temp_final", type=float, default=None,
                        help="sec3.2.4: anneal gumbel_temp linearly from --gumbel_temp to this value over training "
                             "(sharper gate for counting late). Default None = fixed temp.")
    parser.add_argument("--phi_gated", action="store_true",
                        help="sec3.4 presence-gated phi + beta-NLL: supervise phi only on present donors with a "
                             "heteroscedastic loss (adds phi_logvar_head). Fixes L1-on-sparse collapse. Needs --aux_heads.")
    parser.add_argument("--phi_gated_weight", type=float, default=None,
                        help="sec3.4 fixed weight for the beta-NLL phi loss (default 0.3; NOT Kendall since beta-NLL can be negative)")
    # ── EBM-SPEN Joint (--cls_decoder spen) ────────────────────────────────────────────────
    parser.add_argument("--n_inf_steps", type=int, default=None,
                        help="spen: projected GD steps at inference (default 10)")
    parser.add_argument("--inf_lr", type=float, default=None,
                        help="spen: gradient descent step size in y-space (default 0.5)")
    parser.add_argument("--spen_global_hidden", type=int, default=None,
                        help="spen: hidden dim of global energy MLP (default 128)")
    parser.add_argument("--spen_aux_w", type=float, default=None,
                        help="spen: weight for auxiliary BCE on initial local logits (default 1.0)")
    parser.add_argument("--phi_rerank", action="store_true",
                        help="DEPLOYABLE post-hoc rerank of the cls ranking by an INDEPENDENT EM mixture-proportion "
                             "deconvolution (phi_rerank.py; logarithmic opinion pool, alpha tuned on val). Needs "
                             "donor_geno. Default off = decode unchanged. Verified +7pp N5 oracle on inc22_fixed.")
    parser.add_argument("--em_phi_feature", action="store_true",
                        help="INTERNALIZE the EM mixture-proportion (Mx) deconvolution into the model: FiLM-lite "
                             "log-phi -> cls logit (learned logarithmic opinion pool) + Hill effective-count -> "
                             "noc_head_v2. Deterministic/deployable; needs donor_geno. Default off = no-op (zero-init).")
    parser.add_argument("--noise_gate", action="store_true",
                        help="SOFT in-model per-peak noise/stutter reliability gate s_p=P(real allele) from PROVEN "
                             "structural features (Tvedebrink height + n±1 stutter ratios; Brookes/Bright), supervised "
                             "by the carried-by-a-true-donor label (Balding-Buckleton drop-in). Soft-floored down-weight "
                             "of phantom peaks in the aslot decoder. Default off = decoder unchanged.")
    parser.add_argument("--noise_gate_w", type=float, default=None,
                        help="supervised BCE weight for --noise_gate (default 0.3)")
    parser.add_argument("--replicates", type=int, default=None,
                        help="RICHER DATA: pool R replicate amplifications per mixture (loads _rep{R} peak "
                             "arrays from make_replicates.py; EFMrep). Default 1 = single profile (unchanged).")
    parser.add_argument("--count_on_rerank", action="store_true",
                        help="Fit the post-hoc count on the RERANKED score profile (count the thing you "
                             "threshold) instead of the raw prob-profile. Needs --phi_rerank. Default off = legacy.")
    args = parser.parse_args()

    cfg_path = args.config or str(ROOT / "configs" / "set_transformer.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    # CLI overrides
    for k in ("n_isab", "d_model", "m_inducing", "epochs", "lr", "batch_size", "cls_decoder",
              "decoder_source", "loss", "n_token_feats", "encoder", "dec_layers",
              "num_embed", "n_freq", "d_num_emb", "periodic_sigma", "rnc_tau",
              "noc_contrast_mode", "rnc_fixed_weight", "irm_lambda", "irm_anneal", "d_proj", "seed",
              "donor_contrast_weight", "noc_ord_weight",
              "mask_peaks", "mask_peaks_min", "and_mask_thresh",
              "sam_rho", "vib_weight", "masked_pretrain_epochs", "mask_pretrain_ratio",
              "minor_phi_ref", "minor_cap", "vicreg_weight", "early_abort_base_n5",
              "cls_loss", "ldam_drw_epoch", "bal_gamma_pos", "attn_sink", "donor_recon_weight",
              "qdn_noise", "qdn_weight", "ifm_eps", "ifm_weight", "nc_attn", "dec_aggr",
              "distill_weight", "distill_ramp", "cf_weight", "cf_ramp",
              "diff_weight", "diff_min_factor", "diff_ramp",
              "ml_attr_weight", "ml_distill_weight", "ml_distill_ramp", "ml_pos_weight",
              "add_invar_weight", "add_invar_ramp", "id_dim",
              "recon_weight", "recon_ramp", "add_recon_weight",
              "noise_spectrum_levels", "xnoise_weight", "xnoise_ramp", "xnoise_clean", "warm_start",
              "n_slot_iters", "ot_eps", "ot_iters", "gumbel_temp", "noc_v2_weight", "gate_count_weight", "gate_temp_final", "phi_gated_weight",   # aslot
              "n_inf_steps", "inf_lr", "spen_global_hidden", "spen_aux_w"):  # spen
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    if args.donor_recon:
        cfg["donor_recon"] = True
    if args.query_denoise:
        cfg["query_denoise"] = True
    if args.ifm:
        cfg["ifm"] = True
    if args.gamma_neg is not None:
        cfg["asl_gamma_neg"] = args.gamma_neg
    if args.early_abort:
        cfg["early_abort"] = True
    for flag in ("geno_query", "donor_contrast", "noc_ord_head", "noc_ord_detach", "noc_ord_replace",
                 "and_mask", "sam", "vib", "minor_weight", "mass_pool",
                 "vicreg", "vicreg_inv", "ref_match", "ref_match_learn", "nc_learnable_bias",
                 "distill_attr", "cf_invariance", "diff_ladder",
                 "ml_attr", "ml_distill", "add_invar", "recon", "add_recon",
                 "noise_spectrum", "xnoise_consistency",
                 "phi_inject", "soft_geno_attr", "feas_filter", "soft_attr_label",
                 "set_of_set", "noc_head_v2", "gate_count", "log_per_noc", "gate_mass", "phi_gated",
                 "phi_rerank", "em_phi_feature",
                 "noise_gate", "count_on_rerank"):  # Inc18 / SoS / NOC head / gate-count / per-noc log / mass-gate / phi-gated / phi-rerank / em-phi / noise gate / count-on-rerank
        if getattr(args, flag):
            cfg[flag] = True
    if args.noise_gate_w is not None:
        cfg["noise_gate_w"] = args.noise_gate_w
    if args.replicates is not None:
        cfg["replicates"] = args.replicates
    if args.pos_weight is not None:
        cfg["pos_weight_cls"] = args.pos_weight
    if args.aux_heads:
        cfg["aux_heads"] = True
    for _lf in ("lupi_phys", "lupi_degr", "lupi_mu", "lupi_var", "lupi_dropin"):
        if getattr(args, _lf, False):
            cfg[_lf] = True
    if args.noc_contrast:
        cfg["noc_contrast"] = True
    if args.sparse_attn:
        cfg["sparse_attn"] = True
    if args.irm:
        cfg["irm"] = True
    if args.out_subdir is not None:
        cfg["out_subdir"] = args.out_subdir

    train(cfg)
