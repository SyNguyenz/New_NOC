"""
train_set_transformer.py — standalone trainer for the `inc22_fixed_aslot` arm, one archive fold per run.

ONE backbone, ONE forward pass, TWO outputs (models/set_transformer.py, AdaptiveSlotDecoder):
    logits_cls  = cls_raw + gate_logit   -> WHICH donors (ID)
    logits_card = noc_head(gate)         -> HOW MANY donors (NoC)

Phase 1 (this file) trains that backbone on in-silico mixtures + real single-source profiles:
  * model  = SetTransformerMixture (CoSA+GSANet+MESH+AdaSlot, isab++/nc_mab0, periodic embed, aux heads,
             set_of_set, feas_filter)
  * loss   = ASL(gamma_neg=4) on logits_cls
             + alpha_reject * BCE(reject, closed-0 / open-1)
             + beta_card    * CE(logits_card, EM-optimal-k target, class-weighted)
             + w_gate       * SmoothL1(sum gate, NOC)        (the gate counts live slots)
             + beta_card    * CORN(noc_head_v2, true NOC)    (trained, NEVER decoded — see below)
             + Kendall( soft_attr_label CE + L1 phi )        (aux_heads, soft_attr_label)
  * train aug = mask_peaks 0.15 (drop shared peaks only; never a minor's private/single-carrier peak)
  * selection = macro-over-NOC oracle Recall@k on the in-silico DEV split. No early stopping;
                best_model.pt = best DEV epoch, last_model.pt = final epoch.
                best_model.fold.json records WHICH archive fold the backbone was trained on, so a later
                --init_from refuses a backbone whose in-silico train set holds this fold's test combos.

DECODE — the ID ranking and the count are separate, and the count has several readouts:
  1. ID: phi_rerank, an independent EM mixture-proportion deconvolution pooled with the per-donor
     logits (alpha fit leave-one-combination-out). Reranks the RANKING only, never the count.
     Also reported with alpha fit on in-silico DEV alone, which is the label-free number.
  2. COUNT (--count, --compare). Two need NO real mixture label:
       zero_shot  logits_card straight off the frozen backbone
       gate       round(sum gate)
       gate_cal   sum(gate) -> k, calibrated on in-silico DEV
     and two fit on the real test set under a combination-disjoint K-fold, so every profile is counted
     by a module that never saw its donor combination:
       lora/lora_inv    frozen backbone + count-only LoRA (enable_noc_lora)
     `_inv` = read through an identity-blind head on the SORTED gate/probabilities. The fine-tuned-clone
     arm and the post-hoc RandomForest were both closed; their code is in archive/2026-10-01_truoc_clean/.
     The ID pass never reads the LoRA and no backbone weight receives gradient ([ID GUARD] checks it).
  3. --protocol: `strict` = the combination-disjoint K-fold above (the headline number).
     `deepnoc` = deepNoC's own split verbatim (paper s2.6, every second profile, 50/50, combinations
     SHARED on purpose) so our number can sit beside the published one. The difference between the two
     is the combination-reuse premium. `both` reports both.
  4. ID set = top-k of the reranked ranking at k from the count; oracle = top-true-k (the ceiling).

noc_head_v2 (CORN ordinal head) IS trained but its output is NEVER decoded: it collapses N3/N4
(~0.21/0.17). It stays because the published run trained with it and its gradient enters the global
clip_grad_norm_ denominator (|g_corn| ~ |g_rest|, 90% of steps clipped) — dropping it is not
training-neutral. Every count that was tried and dropped is in doc/failed_experiments.md.

Run (STR_DATA_DIR points at the enriched, dev-split data dir; kaggle_run_increment1.py prepares it):
    STR_DATA_DIR=<data_dir> python train_set_transformer.py --seed 42 --out_subdir fold0
    STR_DATA_DIR=<data_dir> python train_set_transformer.py --seed 42 --out_subdir fold1 --init_from <best_model.pt>
Writes results/<out_subdir>_seed<seed>/{best_model.pt, best_model.fold.json, metrics.json, report.json,
y_test_pred.npy, y_test_true.npy} and reports/fold<F>_seed<seed>.json (pooled by report_folds.py).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from models.ordinal import corn_loss, corn_probs
from noc_lora import noc_lora_kfold, head_kfold, HEAD_ARMS   # the COUNT: LoRA, or a small head, on a frozen backbone
from fold_report import (decode_k, arrfp, count_diagnostics, f1_macro, fold_stamp,   # provenance + report
                         open_labels, params_fp, score_count_arm, score_open_split, write_report)
import phi_rerank as pr

# Data dir (the orchestrator sets STR_DATA_DIR to the enriched, dev-split data_w).
DATA_DIR = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data_insilico_w")))


def _pick_device() -> torch.device:
    """CUDA (Kaggle/Colab) -> MPS (Apple Silicon) -> CPU. Override with STR_DEVICE=cpu|mps|cuda.

    CUDA stays first so an existing GPU run is unchanged; MPS lets the same code train on a Mac.
    """
    forced = os.environ.get("STR_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _pick_device()
ALLELE_OFF = 30
LUT_W = 1024
# Every count readout this trainer knows. ONE list: --count and --compare both read it, so adding an arm
# is a single edit. kaggle_run_increment1.py must NOT restrict these (the trainer owns the list).
# The four LoRA arms differ ONLY in which identity-blind features the count head reads. All of them
# keep the backbone frozen and leave the ID pass untouched ([ID GUARD] params_unchanged + fp unchanged),
# which is the whole reason the count lives in a LoRA and not in Phase 1.
#   lora        donor-indexed noc_head copy      (can memorise WHICH donors)
#   lora_inv    sorted gate + sorted ID probs    (shipped; still leans on ID VALUES -> .198 off-panel)
#   lora_mac    the above + 19 MAC physical feats (adds the one channel that survives when ID fails)
#   lora_free   sorted gate + MAC, NO ID probs   (fully independent of the ID posterior)
LORA_FEATS = {"lora_inv":  dict(use_probs=True,  use_mac=False),
              "lora_mac":  dict(use_probs=True,  use_mac=True),
              "lora_free": dict(use_probs=False, use_mac=True)}
# card_cal / card_ft / head_only (noc_lora.HEAD_ARMS): no LoRA at all, a 6 / 230 / ~1.1k parameter head fit on
# features of the frozen backbone. Meant for a deepNoC-sized label budget (--lora_budget), where less to fit is
# less to memorise.
COUNT_ARMS = ["lora", "lora_inv", "lora_mac", "lora_free", "gate_cal", "zero_shot", "gate", "corn", *HEAD_ARMS]

# ── Fixed inc22 configuration (the `configs/set_transformer.json` base + the inc22 flags) ──
CFG = {
    "n_loci": 24, "d_locus": 16, "d_model": 128, "n_heads": 4, "n_isab": 2, "m_inducing": 32,
    "n_classes": 45, "dropout": 0.1,
    "lr": 6e-4, "weight_decay": 1e-4, "batch_size": 256, "epochs": 150,   # no early stopping
    "alpha_reject": 0.5, "beta_card": 0.3, "card_lambda": 0.02, "open_ratio": 0.25,
    "n_token_feats": 8, "num_embed": "periodic", "periodic_sigma": 0.3, "n_freq": 8, "d_num_emb": 8,
    "encoder": "isab++", "nc_attn": "mab0", "cls_decoder": "aslot",
    "aux_heads": True, "set_of_set": True, "feas_filter": True, "soft_attr_label": True,
    "mask_peaks": 0.15, "mask_peaks_min": 8,
    "n_slot_iters": 3, "ot_eps": 0.05, "ot_iters": 5, "gumbel_temp": 1.0,
    "loss": "asl", "asl_gamma_neg": 4.0, "asl_gamma_pos": 0.0, "asl_clip": 0.05,
    # decode: phi-rerank the RANKING, count = NoC LoRA (combination-disjoint K-fold).
    "phi_rerank": True,
    # CORN ordinal count head on DETACHED features. Trained, never decoded (see module docstring).
    "noc_head_v2": True,          # its CORN loss is weighted by beta_card, as in the original
    # Make the AdaSlot gate mean what AdaSlot says it means: the number of live slots IS the count.
    # `gate` is NOT detached and feeds logits_cls through gate_logit, so this term reaches the encoder —
    # that is the point, and the ID oracle in the report is the guard. This weight decides which count
    # readout actually learns: ours puts the signal in logits_card (beta_card), the synznguyen branch
    # puts it here. Numbers for both: doc/failed_experiments.md muc 23.
    "w_gate": 0.05,            # weight on SmoothL1(sum gate, NOC)
    # COUNT. Shipped 2026-09-24: lora_inv scope=decoder — one frozen backbone serving both tasks with
    # ~26k trainable parameters instead of a 1.5M copy. Every arm, every budget and every number measured
    # so far: doc/failed_experiments.md muc 10-13 (budget) and muc 22-23 (readouts). Do NOT restate
    # results here; a stale number in a comment has already been quoted as if it were current.
    "count": "lora_inv", "compare": [], "count_decode": "expect", "count_kfold": 5,
    # PROTOCOL: "strict" = the K-fold combination-disjoint count above (the headline). "deepnoc" adds
    # deepNoC's own fine-tune split, verbatim (paper s2.6: "every second profile was used to train ...
    # and the remaining ... were used to test"): 50/50, NO disjointness of any kind, so both halves hold
    # the same donor combinations. It exists only so our number can sit beside the published one on the
    # same footing; the difference between the two is the combination-reuse premium. "both" runs both.
    "protocol": "both",
    "lora_steps": 3000, "lora_rank": 8, "lora_scope": "decoder",   # rank 16 lost to rank 8 (muc 13)
    # LoRA recipe meant to COUNT, not memorise. Budget is bounded by the DATA, not the model: ~13-18 real
    # mixture combinations per fit, so extra epochs buy memorisation. lora_epochs=0 restores the step
    # budget. Sweep and memorisation gaps: muc 11-13.
    "lora_epochs": 5, "lora_head_lr": 3e-4,
    # LABEL BUDGET. None = every labelled row of the fit side. deepNoC fine-tuned on 371 real profiles,
    # 34/87/79/93/78 for NOC1..5 (Table 2 column sums subtracted from 68/175/158/186/156); a fit here otherwise
    # sees ~1900-2060. A budget draws that many rows per NOC from the fit side only, under both protocols.
    "lora_budget": None, "lora_budget_seed": 0,
}


def set_seed(seed: int) -> torch.Generator:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if DEVICE.type == "mps":
        torch.mps.manual_seed(seed)
    g = torch.Generator(); g.manual_seed(seed)
    return g


# ── Loss / decode helpers (verbatim from train_set_transformer.py) ──────────
class AsymmetricLoss(nn.Module):
    """Asymmetric Loss (Ben-Baruch 2020): down-weight easy negatives, focus hard positives."""
    def __init__(self, gamma_neg=4.0, gamma_pos=0.0, clip=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg; self.gamma_pos = gamma_pos; self.clip = clip; self.eps = eps

    def forward(self, logits, targets, weight=None):
        xs_pos = torch.sigmoid(logits)
        xs_neg = 1.0 - xs_pos
        if self.clip and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)
        los_pos = targets * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - targets) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg
        pt = xs_pos * targets + xs_neg * (1 - targets)
        gamma = self.gamma_pos * targets + self.gamma_neg * (1 - targets)
        loss *= (1 - pt) ** gamma
        if weight is not None:
            loss = loss * weight
        return -loss.mean()


def cardinality_target(probs, y, lam=0.02, n_card=5):
    """EM-optimal NOC target per sample: argmin_k [set_mismatch(top-k, y) + lam*k] -> {0..n_card-1}."""
    with torch.no_grad():
        B, C = probs.shape
        order = probs.argsort(dim=1, descending=True)
        rank = torch.empty_like(order)
        ar = torch.arange(C, device=probs.device).expand(B, C)
        rank.scatter_(1, order, ar)
        K = y.sum(1).clamp(min=1)
        costs = []
        for k in range(1, n_card + 1):
            topk = (rank < k).float()
            miss = (y * (1 - topk)).sum(1) / K
            extra = ((1 - y) * topk).sum(1) / k
            costs.append(miss + extra + lam * k)
        return torch.stack(costs, 1).argmin(1)


def topk_decode(probs, k_arr):
    yp = np.zeros_like(probs, dtype=int)
    for i in range(len(probs)):
        k = int(max(1, min(5, round(k_arr[i]))))
        yp[i, np.argsort(probs[i])[::-1][:k]] = 1
    return yp


def _pn_(lst):
    """per-NOC dict from a per_noc_em list (index 0 = overall)."""
    return {str(j): (None if np.isnan(lst[j]) else round(float(lst[j]), 4)) for j in range(1, 6)}


def per_noc_em(y_true, y_pred, noc):
    e = np.all(y_true == y_pred, axis=1)
    return [e.mean()] + [e[noc == j].mean() if (noc == j).sum() else float("nan") for j in range(1, 6)]


def loco_decode(L_te, y_te, noc_te, PHt, combo_id):
    """LEAVE-ONE-COMBO-OUT fit of the phi-rerank alpha.

    Every real donor-combo lives in test now (the network trains on in-silico only, so no real combo
    has to be spent as a fitting set). Each combo is reranked with an alpha fit on the OTHER combos
    only — group-disjoint exactly like the old held-out val, but every real mixture becomes
    reportable instead of ~15% of them. The single-source test rows form one extra group, fit on all
    mixtures.

    Returns (rank_te, alphas_per_group, alpha_full). alpha_full is fit on ALL mixtures — that is the
    artifact you would ship; the per-group alphas exist only to keep the estimate honest.
    """
    mix = combo_id >= 0
    groups = [np.where(combo_id == c)[0] for c in np.unique(combo_id[mix])]
    ss = np.where(~mix)[0]
    if len(ss):
        groups.append(ss)
    alpha_full = float(pr.tune_alpha(L_te[mix], PHt[mix], y_te[mix], noc_te[mix])) if mix.any() else 0.0

    rank_te = np.zeros(L_te.shape, dtype=np.float64)
    alphas = []
    for g in groups:
        held = np.zeros(len(L_te), bool); held[g] = True
        fit = mix & ~held                       # other combos only — a group never fits on itself
        assert not fit[g].any(), "leave-one-combo-out violated"
        a = float(pr.tune_alpha(L_te[fit], PHt[fit], y_te[fit], noc_te[fit])) if fit.any() else alpha_full
        alphas.append(a)
        rank_te[g] = pr.rerank_scores(L_te[g], PHt[g], a)
    return rank_te, alphas, alpha_full


# ── Datasets ─────────────────────────────────────────────────────────────────
class ClosedSetDataset(Dataset):
    def __init__(self, split: str):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"tokens8_{split}.npy").astype(np.float32))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / f"mask_{split}.npy"))
        self.y      = torch.from_numpy(np.load(DATA_DIR / f"y_{split}_set.npy"))
        self.noc    = torch.from_numpy(np.load(DATA_DIR / f"noc_{split}.npy").astype(np.int64))
        N, S = self.tokens.shape[0], self.tokens.shape[1]
        ap = DATA_DIR / f"attr_{split}.npy"; pp = DATA_DIR / f"phi_{split}.npy"
        if ap.exists() and pp.exists():
            self.attr = torch.from_numpy(np.load(ap).astype(np.int64))
            self.phi  = torch.from_numpy(np.load(pp).astype(np.float32))
        else:                                                  # real splits without provenance -> sentinels
            self.attr = torch.full((N, S), -1, dtype=torch.int64)
            self.phi  = torch.zeros((N, 45), dtype=torch.float32)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i], self.y[i], self.noc[i], self.attr[i], self.phi[i]


class OpenSetDataset(Dataset):
    def __init__(self):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / "tokens8_open.npy").astype(np.float32))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / "mask_open.npy"))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i]


# ── Selection-metric eval (verbatim) ─────────────────────────────────────────
@torch.no_grad()
def evaluate_closed(model, loader, threshold=0.5):
    model.eval(); all_true, all_pred = [], []
    for tokens, mask, y, *_ in loader:
        probs = torch.sigmoid(model(tokens.to(DEVICE), mask.to(DEVICE))["logits_cls"]).cpu().numpy()
        all_pred.append((probs >= threshold).astype(np.float32)); all_true.append(y.numpy())
    return float(f1_score(np.concatenate(all_true), np.concatenate(all_pred),
                          average="macro", zero_division=0))


@torch.no_grad()
def evaluate_oracle_em(model, loader):
    """Oracle top-k EM (k=true NOC) -> (overall EM, macro-over-NOC recall@k). Selection metric."""
    model.eval(); all_probs, all_true, all_noc = [], [], []
    for tokens, mask, y, noc, *_ in loader:
        all_probs.append(torch.sigmoid(model(tokens.to(DEVICE), mask.to(DEVICE))["logits_cls"]).cpu().numpy())
        all_true.append(y.numpy()); all_noc.append(noc.numpy())
    probs = np.concatenate(all_probs); y_true = np.concatenate(all_true); noc = np.concatenate(all_noc)
    yp = np.zeros_like(probs, dtype=int); rec = np.zeros(len(probs))
    for i in range(len(probs)):
        k = int(max(1, min(5, noc[i]))); top = np.argsort(probs[i])[::-1][:k]
        yp[i, top] = 1; rec[i] = y_true[i][top].sum() / k
    em = (y_true == yp).all(1); nocc = np.clip(noc, 1, 5)
    strata = [rec[nocc == j].mean() for j in range(1, 6) if (nocc == j).any()]
    return float(em.mean()), float(np.mean(strata))


def per_noc_oracle_on_scores(S, Y, noc):
    """Per-NOC oracle EM (full set correct at top-true-k) on a RANKING score S."""
    N = np.clip(noc, 1, 5); out = {}
    for j in range(1, 6):
        m = np.where(N == j)[0]
        if not len(m):
            continue
        ems = []
        for i in m:
            top = np.argsort(S[i])[::-1][:j]; pred = np.zeros(S.shape[1], int); pred[top] = 1
            ems.append(bool((pred == Y[i]).all()))
        out[str(j)] = round(float(np.mean(ems)), 4)
    return out


# ── owner_lut / cn_lut (carrier + copy-number LUTs from reference genotypes) ──
def build_luts(donor_geno, donor_geno_mask, n_cls):
    gg = donor_geno; gm = donor_geno_mask.bool()
    owner = torch.zeros(24, LUT_W, n_cls)
    cn = torch.zeros(24, LUT_W, n_cls)
    for c in range(min(n_cls, gg.size(0))):
        for j in range(gg.size(1)):
            if gm[c, j]:
                li = int(gg[c, j, 0]); ab = int(round(float(gg[c, j, 1]) * 10)) + ALLELE_OFF
                if 0 <= li < 24 and 0 <= ab < LUT_W:
                    owner[li, ab, c] = 1.0
                    cn[li, ab, c] += 1.0                       # accumulate: 2 for homozygous
    return owner, cn


def run_phase1(model, epochs, train_loader, val_loader, sel_loader, open_iter, open_ds,
                optimizer, scheduler, bce_cls, bce_rej, card_w, card_lam, alpha, beta, noc_v2_w,
                w_gate, log_var_attr, log_var_phi, mask_peaks_p, mask_min, gather_owner, cn_lut,
                cfg, gen, results_dir, cur_stamp, t0):
    """The Phase 1 epoch loop. Writes best_model.pt (best in-silico DEV epoch) + its fold stamp and
    last_model.pt (final epoch); returns (best_sel, best_epoch, history).

    Extracted from train() unchanged — the body below is byte-for-byte the loop that used to sit
    inline, so a run before and after the extraction produces identical numbers (verified on the
    CPU smoke fold)."""
    best_sel, best_epoch, history = 0.0, 0, []
    train_ds = train_loader.dataset
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_cls = epoch_rej = epoch_noc_l = epoch_attr = epoch_phi = 0.0
        for tokens, mask, y, noc, attr, phi in train_loader:
            tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)
            y, noc = y.to(DEVICE), noc.to(DEVICE); attr = attr.to(DEVICE); phi = phi.to(DEVICE)

            # mask_peaks: drop a fraction of valid SHARED peaks (never a minor's private peak),
            # keep >= mask_min so NOC stays identifiable.  Train only.
            if mask_peaks_p > 0.0:
                mb = mask.bool()
                drop = (torch.rand_like(mb, dtype=torch.float) < mask_peaks_p) & mb
                n_car = gather_owner(tokens).sum(-1)
                drop = drop & (n_car != 1)
                kept = mb & ~drop
                enough = kept.sum(1, keepdim=True) >= mask_min
                mask = torch.where(enough, kept, mb).to(mask.dtype)

            out = model(tokens, mask)
            loss_cls = bce_cls(out["logits_cls"], y)
            card_tgt = cardinality_target(torch.sigmoid(out["logits_cls"]).detach(), y, card_lam)
            loss_noc = F.cross_entropy(out["logits_card"], card_tgt, weight=card_w)

            # reject: closed -> 0, open -> 1
            try:
                open_batch = next(open_iter)
            except StopIteration:
                open_iter = iter(DataLoader(
                    open_ds, batch_size=max(1, int(cfg["batch_size"] * cfg["open_ratio"])),
                    sampler=RandomSampler(open_ds, replacement=True, num_samples=len(open_ds) * 10,
                                          generator=gen), num_workers=0))
                open_batch = next(open_iter)
            o_tok, o_mask = open_batch[0].to(DEVICE), open_batch[1].to(DEVICE)
            rej_open = model(o_tok, o_mask)["logit_reject"]
            all_rej = torch.cat([out["logit_reject"], rej_open], dim=0)
            all_rej_lbl = torch.cat([torch.zeros(len(tokens), 1, device=DEVICE),
                                     torch.ones(len(o_tok), 1, device=DEVICE)], dim=0)
            loss_rej = bce_rej(all_rej, all_rej_lbl)

            loss = loss_cls + alpha * loss_rej + beta * loss_noc

            # CORN count head on TRUE noc (noc_head_v2). Inputs are detached, so this adds no
            # encoder gradient — but it DOES add gradient mass to clip_grad_norm_, exactly as in
            # the published arm. Decode ignores logits_count_v2.
            if cfg["noc_head_v2"] and "logits_count_v2" in out:
                loss = loss + noc_v2_w * corn_loss(out["logits_count_v2"], noc.clamp(1, 5), 5)

            # tier B: the number of live AdaSlot slots IS the count. Unlike every other count path
            # here, `gate` is NOT detached — this term reaches the encoder through gate_logit.
            if w_gate > 0:
                loss = loss + w_gate * F.smooth_l1_loss(
                    out["gate"].sum(1), noc.clamp(1, 5).to(out["gate"].dtype))

            # privileged aux: soft_attr_label CE (EuroForMix phi*CN soft labels) + L1 phi, Kendall-weighted
            loss_attr_v = loss_phi_v = 0.0
            if (attr >= 0).any():
                la = out["logits_attr"]
                li_ = tokens[..., 0].long().clamp(0, 23)
                bi_ = (tokens[..., 1] * 10).round().long() + ALLELE_OFF
                bi_ = bi_.clamp(0, cn_lut.size(1) - 1)
                cn_ = cn_lut[li_, bi_]                                    # (B, N, C)
                phi_cn_ = phi.unsqueeze(1) * cn_
                tot_ = phi_cn_.sum(-1, keepdim=True)
                feas_ = (tot_.squeeze(-1) > 1e-9) & mask.bool() & (attr >= 0)
                norm_ = phi_cn_ / tot_.clamp(min=1e-9)
                bg_ = (~feas_ & mask.bool()).unsqueeze(-1).float()
                soft_y_ = torch.cat([norm_ * feas_.unsqueeze(-1).float(), bg_], dim=-1)
                log_p_ = F.log_softmax(la, dim=-1)
                raw_l_ = -(soft_y_ * log_p_).sum(-1)
                vm_ = mask.bool().float()
                loss_attr = (raw_l_ * vm_).sum() / vm_.sum().clamp(min=1)
                loss_phi = F.l1_loss(out["phi"], phi)
                loss = loss + (torch.exp(-log_var_attr) * loss_attr + log_var_attr
                               + torch.exp(-log_var_phi) * loss_phi + log_var_phi)
                loss_attr_v = loss_attr.item(); loss_phi_v = loss_phi.item()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            bs = len(tokens)
            epoch_loss += loss.item() * bs; epoch_cls += loss_cls.item() * bs
            epoch_rej += loss_rej.item() * bs; epoch_noc_l += loss_noc.item() * bs
            epoch_attr += loss_attr_v * bs; epoch_phi += loss_phi_v * bs

        n = len(train_ds); epoch_loss /= n
        val_f1 = evaluate_closed(model, val_loader)
        val_em, val_macrec = evaluate_oracle_em(model, sel_loader)
        sel_score = val_macrec
        scheduler.step(sel_score)
        history.append({"epoch": epoch, "loss": round(epoch_loss, 4), "val_macro_f1": round(val_f1, 4),
                        "val_oracle_em": round(val_em, 4), "val_macro_recall": round(val_macrec, 4)})
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | loss={epoch_loss:.4f} (cls={epoch_cls/n:.3f} rej={epoch_rej/n:.3f} "
                  f"noc={epoch_noc_l/n:.3f} attr={epoch_attr/n:.3f} phi={epoch_phi/n:.3f}) "
                  f"| DEV macrec={val_macrec:.4f} em={val_em:.4f} f1={val_f1:.4f} "
                  + f"| lr={optimizer.param_groups[0]['lr']:.1e}  ({time.time()-t0:.0f}s)", flush=True)

        # NO early stopping: the selection metric saturates and a patience counter stops on noise.
        # Always run the full `epochs`. `best_model.pt` = best in-silico DEV epoch, `last_model.pt` =
        # final epoch, so both can be evaluated.
        if sel_score > best_sel:
            best_sel, best_epoch = sel_score, epoch
            torch.save(model.state_dict(), results_dir / "best_model.pt")
            # Sidecar, not inside the checkpoint (loaded with weights_only=True): WHICH archive cut this
            # backbone was trained on, so a later --init_from can refuse a mismatch.
            if cur_stamp is not None:
                st = dict(cur_stamp, withhold_donors=[], reject_open_rows=None)
                cfg["_bb_src"] = "Phase 1 trained in this run"; cfg["_bb_stamp"] = dict(st)
                json.dump(st, open(results_dir / "best_model.fold.json", "w"), indent=1)
        torch.save(model.state_dict(), results_dir / "last_model.pt")
    print(f"Training done in {time.time()-t0:.1f}s")
    return best_sel, best_epoch, history


def train(seed: int, out_subdir: str, init_from: str | None = None):
    cfg = dict(CFG)
    cfg["seed"] = seed; cfg["out_subdir"] = out_subdir
    results_dir = ROOT / "results" / f"{out_subdir}_seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    gen = set_seed(seed)
    _srcs = [ROOT / f for f in ("train_set_transformer.py", "noc_lora.py",
                                "fold_report.py", "models/set_transformer.py")]   # CODE covers every file
    cfg["_code_hash"] = hashlib.md5(b"".join(f.read_bytes() for f in _srcs)).hexdigest()[:10]
    print(f"CODE {cfg['_code_hash']}  seed={seed}  data={DATA_DIR}  device={DEVICE}")
    _cd = (f"{cfg['count']} r{cfg['lora_rank']} scope={cfg['lora_scope']} x "
           + (f"{cfg['lora_epochs']} ep" if cfg['count'] == 'lora_inv' and cfg['lora_epochs']
              else f"{cfg['lora_steps']} steps"))
    print(f"epochs={cfg['epochs']}  w_gate={cfg['w_gate']}  beta_card={cfg['beta_card']}  "
          f"noc_head_v2={cfg['noc_head_v2']}  count={_cd}, {cfg['count_kfold']}-fold combination-disjoint"
          + (f"  LABEL BUDGET {cfg['lora_budget']} draw {cfg['lora_budget_seed']}" if cfg['lora_budget'] else ""))
    if cfg["lora_budget"] is not None and len(cfg["lora_budget"]) != 5:
        raise SystemExit(f"--lora_budget needs 5 values (NOC1..5), got {cfg['lora_budget']}")

    train_ds = ClosedSetDataset("train"); val_ds = ClosedSetDataset("val")
    test_ds = ClosedSetDataset("test"); open_ds = OpenSetDataset()
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    if not (DATA_DIR / "tokens8_dev.npy").exists():
        raise SystemExit("tokens8_dev.npy missing: run make_dev_split.py + features/enrich.py "
                         "(kaggle_run_increment1.py does)")
    sel_loader = DataLoader(ClosedSetDataset("dev"), batch_size=256, shuffle=False)
    print(f"selection set = in-silico DEV ({len(sel_loader.dataset)} samples)")

    open_iter = iter(DataLoader(
        open_ds, sampler=RandomSampler(open_ds, replacement=True,
                                       num_samples=len(train_ds) * cfg["epochs"], generator=gen),
        batch_size=max(1, int(cfg["batch_size"] * cfg["open_ratio"])), num_workers=0))

    # reference genotypes + carrier/copy-number LUTs
    gp = DATA_DIR / "donor_geno.npy"
    if not gp.exists():
        gp = ROOT / "data" / "donor_geno.npy"
    donor_geno = torch.from_numpy(np.load(gp).astype(np.float32))
    donor_geno_mask = torch.from_numpy(np.load(gp.parent / "donor_geno_mask.npy"))
    owner_lut, cn_lut = build_luts(donor_geno, donor_geno_mask, cfg["n_classes"])
    owner_lut = owner_lut.to(DEVICE); cn_lut = cn_lut.to(DEVICE)
    print(f"reference genotypes {tuple(donor_geno.shape)} from {gp.parent} | owner_lut {tuple(owner_lut.shape)}")

    model = SetTransformerMixture(
        n_loci=cfg["n_loci"], d_locus=cfg["d_locus"], d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"], n_classes=cfg["n_classes"], dropout=cfg["dropout"],
        n_token_feats=cfg["n_token_feats"], n_freq=cfg["n_freq"], d_num_emb=cfg["d_num_emb"],
        periodic_sigma=cfg["periodic_sigma"], n_slot_iters=cfg["n_slot_iters"], ot_eps=cfg["ot_eps"],
        ot_iters=cfg["ot_iters"], gumbel_temp=cfg["gumbel_temp"],
        donor_geno=donor_geno, donor_geno_mask=donor_geno_mask, owner_lut=owner_lut,
        noc_head_v2=cfg["noc_head_v2"],
    ).to(DEVICE)
    # per-feature standardization from train valid peaks (enriched tokens)
    _tk = train_ds.tokens.numpy(); _mk = train_ds.mask.numpy().astype(bool)
    _num = _tk[:, :, 1:cfg["n_token_feats"]][_mk]
    model.feat_mean.copy_(torch.tensor(_num.mean(0), dtype=torch.float32, device=DEVICE))
    model.feat_std.copy_(torch.tensor(_num.std(0) + 1e-6, dtype=torch.float32, device=DEVICE))
    print(f"params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Loss objects + weights ────────────────────────────────────────────────
    bce_cls = AsymmetricLoss(gamma_neg=cfg["asl_gamma_neg"], gamma_pos=cfg["asl_gamma_pos"], clip=cfg["asl_clip"])
    bce_rej = nn.BCEWithLogitsLoss()
    alpha = cfg["alpha_reject"]; beta = cfg["beta_card"]; card_lam = cfg["card_lambda"]
    w_gate = float(cfg["w_gate"])
    noc_v2_w = beta                       # the CORN head carries beta_card's weight, as in the original
    _nc = np.bincount(np.clip(np.load(DATA_DIR / "noc_train.npy"), 1, 5) - 1, minlength=5).astype(float)
    _w = 1.0 / np.clip(_nc, 1, None); _w = _w / _w.mean()
    card_w = torch.tensor(np.clip(_w, 0.5, 2.0), dtype=torch.float32).to(DEVICE)
    # Kendall homoscedastic-uncertainty weights for the privileged aux losses
    log_var_attr = torch.zeros((), device=DEVICE, requires_grad=True)
    log_var_phi  = torch.zeros((), device=DEVICE, requires_grad=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer.add_param_group({"params": [log_var_attr, log_var_phi], "weight_decay": 0.0})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=5, min_lr=1e-6)

    mask_peaks_p = cfg["mask_peaks"]; mask_min = cfg["mask_peaks_min"]

    def gather_owner(tok):
        loc = tok[:, :, 0].long().clamp(0, 23)
        ab = (torch.round(tok[:, :, 1] * 10).long() + ALLELE_OFF).clamp(0, owner_lut.size(1) - 1)
        return owner_lut[loc, ab]

    epochs = cfg["epochs"]        # --init_from sets this to 0, which makes run_phase1 a no-op
    cur_stamp = fold_stamp(DATA_DIR)
    if cur_stamp is not None:
        print(f"  [FOLD] data dir: unknown_donors={cur_stamp['unknown_donors']} n_combo={cur_stamp['n_combo']} "
              f"combo_md5={cur_stamp['combo_md5']} test_n={cur_stamp['split_sizes'].get('test')}")
    if init_from:                      # reuse THIS fold's trained Phase 1 backbone
        src = Path(init_from)
        if src.is_dir():
            src = src / "best_model.pt"
        sc = src.with_suffix(".fold.json")
        st = json.load(open(sc)) if sc.exists() else None
        print(f"\nPhase 1 SKIPPED — backbone loaded from {src}")
        cfg["_bb_src"] = str(src); cfg["_bb_stamp"] = st
        if st is None:
            print("  [FOLD] backbone carries NO fold stamp (best_model.fold.json missing next to it) -> which "
                  "archive cut it was trained on CANNOT be verified. Valid only if it is this fold's backbone.")
        elif cur_stamp is not None and st.get("combo_md5") != cur_stamp["combo_md5"]:
            raise SystemExit(f"  [FOLD] MISMATCH: backbone trained on unknown_donors={st.get('unknown_donors')} "
                             f"(combo_md5 {st.get('combo_md5')}), data dir is {cur_stamp['unknown_donors']} "
                             f"(combo_md5 {cur_stamp['combo_md5']}). Its in-silico train set holds combinations "
                             f"of this fold's test set. Use this fold's backbone or train Phase 1.")
        else:
            print(f"  [FOLD] backbone stamp matches the data dir (combo_md5 {st.get('combo_md5')})  (clean)")
        model.load_state_dict(torch.load(src, map_location=DEVICE, weights_only=True), strict=False)
        torch.save(model.state_dict(), results_dir / "best_model.pt")
        if st is not None:
            json.dump(st, open(results_dir / "best_model.fold.json", "w"), indent=1)
        epochs = 0
    else:
        print(f"\nTraining {epochs} epochs (no early stopping) ...")
    t0 = time.time()

    best_sel, best_epoch, history = run_phase1(
        model, epochs, train_loader, val_loader, sel_loader, open_iter, open_ds, optimizer,
        scheduler, bce_cls, bce_rej, card_w, card_lam, alpha, beta, noc_v2_w, w_gate,
        log_var_attr, log_var_phi, mask_peaks_p, mask_min, gather_owner, cn_lut, cfg, gen,
        results_dir, cur_stamp, t0)

    # ── Integrity: no real test combination may exist in the in-silico train set ──────────
    cid_p = DATA_DIR / "combo_id_test.npy"
    if not cid_p.exists():
        raise SystemExit("combo_id_test.npy missing: the NoC folds and the phi-rerank alpha are grouped by "
                         "donor combination and cannot run without it")
    _y = np.load(DATA_DIR / "y_test_set.npy"); _noc = np.load(DATA_DIR / "noc_test.npy").astype(np.int64)
    _ytr = np.load(DATA_DIR / "y_train_set.npy"); _ntr = np.load(DATA_DIR / "noc_train.npy").astype(int)
    _c_tr = {tuple(np.where(r)[0].tolist()) for r in _ytr[_ntr >= 2]}
    _c_te = {tuple(np.where(r)[0].tolist()) for r in _y[_noc >= 2]}
    _hit = _c_tr & _c_te
    cfg["_split_audit_shared"] = len(_hit)
    print(f"  [SPLIT AUDIT] in-silico train combos={len(_c_tr)}  real test combos={len(_c_te)}"
          f"  SHARED={len(_hit)}" + ("  <<< PHASE 1 SAW THE EVAL COMBOS" if _hit else "  (clean)"))

    # open split: profiles holding at least one donor outside the 45-donor panel
    _op_tok = _op_msk = _op_true = _op_panel = None
    _lab, _why = open_labels(DATA_DIR, len(open_ds), ROOT / "open_labels")
    if _lab is None:
        print(f"  open-set NoC skipped: {_why}")
    else:
        _op_tok = open_ds.tokens.numpy(); _op_msk = open_ds.mask.numpy()
        _op_true = np.clip(_lab[0], 1, 5); _op_panel = _lab[1]
        _uk = np.clip(_op_true - _op_panel, 0, 5)
        print(f"  open-set NoC labels <- {_why}: {len(_op_true)} profiles | true NOC "
              + str({int(v): int((_op_true == v).sum()) for v in range(1, 6)})
              + " | contributors outside the panel " + str({int(v): int((_uk == v).sum()) for v in np.unique(_uk)}))

    # ── Test eval: phi-rerank the ranking, count = NoC LoRA ─────────────────────
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.eval()

    @torch.no_grad()
    def infer(ds):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        L, Y, NOC, S, ZS, CN = [], [], [], [], [], []
        for tokens, mask, y, noc, *_ in loader:
            o = model(tokens.to(DEVICE), mask.to(DEVICE))
            L.append(o["logits_cls"].cpu().numpy()); S.append(o["gate"].sum(1).cpu().numpy())
            ZS.append(o["logits_card"].softmax(1).cpu().numpy())      # zero-shot count, Phase 1 only
            if "logits_count_v2" in o:                                # CORN: true-NOC target, no ID in it
                CN.append(corn_probs(o["logits_count_v2"], 5).cpu().numpy())
            Y.append(y.numpy()); NOC.append(noc.numpy())
        L = np.concatenate(L)
        return (L, 1.0 / (1.0 + np.exp(-L)), np.concatenate(Y), np.concatenate(NOC),
                np.concatenate(S), np.concatenate(ZS),
                np.concatenate(CN) if CN else None)

    eval_label = f"full test ({len(test_ds)} samples)"
    print(f"\n=== Evaluating on {eval_label} ===")
    L_te, P_te, y_te_true, noc_te, S_te, ZS_te, CN_te = infer(test_ds)

    dg = donor_geno.cpu().numpy(); dgm = donor_geno_mask.cpu().numpy()
    PHt = pr.deconv_phi(test_ds.tokens.numpy(), test_ds.mask.numpy(), dg, dgm)
    _param_fp_before = params_fp(model)

    # Decode = phi-rerank the RANKING, alpha fit LEAVE-ONE-COMBO-OUT.
    combo_id = np.load(cid_p).astype(int)
    rank_te, alphas, alpha_pr = loco_decode(L_te, y_te_true, noc_te, PHt, combo_id)
    n_groups = len(alphas)
    decode_desc = f"phi_rerank(ranking), LOCO alpha + {cfg['count']} count (combination-disjoint K-fold)"
    print(f"phi_rerank: LOCO over {n_groups} groups, alpha/group={alphas}, shipped alpha={alpha_pr}")
    print(f"  [ID GUARD] fp(logits_cls)={arrfp(L_te)}  fp(rank_te)={arrfp(rank_te)}  "
          f"params_unchanged={params_fp(model) == _param_fp_before}")

    # tier B readout: the gate sum, rounded. Always reported — it costs nothing and it is the honest
    # check on whether the gate actually learned to count in this run.
    k_gate = np.clip(np.rint(S_te), 1, 5).astype(int)
    mix_ = noc_te >= 2
    _kt5 = np.clip(noc_te, 1, 5)
    corr_gate = float(np.corrcoef(S_te, noc_te)[0, 1])     # computed once, printed and stored below
    print(f"  count round(sum gate): acc {float((k_gate == _kt5).mean()):.4f}  "
          f"acc_mix {float((k_gate[mix_] == noc_te[mix_]).mean()):.4f}  "
          f"macroF1_mix {f1_score(_kt5[mix_], k_gate[mix_], average='macro', labels=[2,3,4,5], zero_division=0):.4f}  "
          f"corr(sum gate, NOC) {corr_gate:+.4f}")

    # ZERO-SHOT: the count the frozen Phase 1 backbone already gives (logits_card = noc_head(gate)),
    # with no real mixture label anywhere. deepNoC reports the same milestone before fine-tuning.
    k_zero = (ZS_te.argmax(1) + 1).astype(int)
    print(f"  count zero-shot logits_card (Phase 1 only, no real mixture label, decode=argmax): "
          f"acc {float((k_zero == _kt5).mean()):.4f}  macroF1 {f1_macro(_kt5, k_zero):.4f}")

    # COUNT on the real test set, K folds that are DISJOINT BY DONOR COMBINATION: every profile is
    # counted by a module that never saw its combination.
    _K = int(cfg["count_kfold"])
    _graw = [tuple(np.flatnonzero(r).tolist()) for r in y_te_true]    # group = donor combination
    _gid = {g_: i_ for i_, g_ in enumerate(sorted(set(_graw)))}
    _grp = np.array([_gid[g_] for g_ in _graw])
    _tk, _mk_ = np.ascontiguousarray(test_ds.tokens.numpy()), np.ascontiguousarray(test_ds.mask.numpy())
    @torch.no_grad()
    def _gate_sum(tk, mk):
        """sum(gate) cho bat ky split nao — dai luong dem khong tham so cua model."""
        out = []
        for i in range(0, len(tk), 256):
            o = model(torch.from_numpy(tk[i:i + 256]).to(DEVICE), torch.from_numpy(mk[i:i + 256]).to(DEVICE))
            out.append(o["gate"].sum(1).cpu().numpy())
        return np.concatenate(out)

    _S_open = _gate_sum(_op_tok, _op_msk) if _op_tok is not None else None   # one pass, both gate arms

    # Every count readout that gets reported, with the decode it was read with — the SAME readout under
    # two decodes is two different numbers (zero_shot: .9102 argmax vs .9014 expect on fold 0), so the
    # decode belongs in the label. Baseline rows are free and always present; arm rows come from --count
    # and --compare. Keys of count_macro_f1_all_sources stay bare for the Kaggle cells that read them.
    _cands = [("gate", k_gate), ("zero_shot", k_zero)]
    _cand_dec = {"gate": "round", "zero_shot": "argmax"}
    p_count = None; count_info = None; k_post = None; _p_open = None; _arms = {}
    _run = [cfg["count"]] + [c for c in dict.fromkeys(cfg.get("compare") or []) if c != cfg["count"]]
    for _nm in _run:
        set_seed(seed)                       # the count depends on the seed only, not on what ran before
        _t0 = time.time()
        _fl = []                             # per count fold: train-pool vs held-out F1 (memorisation gap)
        if _nm == "gate":
            # round(sum gate) NHU MOT ARM DAY DU. sum(gate) = so slot AdaSlot dang song, duoc day bang
            # w_gate*SmoothL1(sum gate, NOC) tren in-silico — khong nhan hon hop that nao. Truoc day chi in
            # lam moc; lam arm thi no duoc cham bang dung khung nay: confusion, precision, F1 tung lop, cot
            # open. Posterior la one-hot tai k = round(sum gate), dung nghia "lam tron", khong bia do tin
            # cay — nen bang nguong tin cay cua arm nay khong co y nghia.
            print("\n  === COUNT = round(sum gate) (so slot song, KHONG dung nhan hon hop that nao) ===")
            _k0 = np.clip(np.rint(S_te), 1, 5).astype(int)
            _P = np.zeros((len(noc_te), 5)); _P[np.arange(len(_k0)), _k0 - 1] = 1.0
            _Po = None
            if _S_open is not None:
                _ko0 = np.clip(np.rint(_S_open), 1, 5).astype(int)
                _Po = np.zeros((len(_ko0), 5)); _Po[np.arange(len(_ko0)), _ko0 - 1] = 1.0
            _info = {"source": "round(sum gate)", "real_mixture_labels_used": 0,
                     "corr_sumgate_noc": round(corr_gate, 4),
                     "n_combos": len(_gid)}
        elif _nm == "zero_shot":
            # THE COUNT WITH NO REAL MIXTURE ANNOTATION AT ALL. logits_card is Phase 1's own count head,
            # trained only on in-silico rows, so nothing about a real mixture label enters it. It is already
            # printed as a free baseline; running it as an ARM gives it the same report as every other count
            # — ID EM@k at its k, the open column, selective classification, per-NOC. On the team's
            # `--conditioned` build it reached macro F1 .9102 against .9238 for the fine-tuned LoRA, so the
            # whole fine-tune is worth +.014 there; on the old build the same arm was .4256-.6588, which is
            # why this was only a baseline until now.
            print(f"\n  === ZERO-SHOT COUNT (logits_card cua Phase 1, KHONG dung nhan hon hop that nao) ===")
            _P = ZS_te.copy()
            _Po = None
            if _op_tok is not None:
                _zs = []
                with torch.no_grad():
                    for i in range(0, len(_op_tok), 256):
                        o = model(torch.from_numpy(_op_tok[i:i + 256]).to(DEVICE),
                                  torch.from_numpy(_op_msk[i:i + 256]).to(DEVICE))
                        _zs.append(o["logits_card"].softmax(1).cpu().numpy())
                _Po = np.concatenate(_zs)
            _info = {"source": "logits_card (Phase 1 only)", "real_mixture_labels_used": 0,
                     "n_combos": len(_gid)}
        elif _nm == "corn":
            # THE COUNT THAT NEVER READS THE ID. noc_head_v2 is the CORN ordinal head: it is trained in
            # EVERY run on `noc.clamp(1,5)` — the TRUE count, not the ID-derived target that logits_card
            # gets — from DETACHED, identity-blind inputs (sorted probability profile + the 19 MAC/PACE
            # physical features deepNoC itself uses + sorted slot_mass). corn_probs() has existed in
            # models/ordinal.py the whole time and nothing ever called it, so this head has trained for
            # 150 epochs every run and been thrown away. It was dropped for collapsing N3/N4 (~.21/.17),
            # but that was measured BEFORE the --conditioned data, and the MAC features lean on peak
            # HEIGHT, which is exactly what the old generator got wrong (24 RFU vs 9 RFU real).
            assert CN_te is not None, "corn arm needs noc_head_v2=True"
            print("\n  === CORN COUNT (noc_head_v2, nhan = NOC THAT, dac trung bat bien danh tinh, "
                  "KHONG dung nhan hon hop that nao) ===")
            _P = CN_te.copy()
            _Po = None
            if _op_tok is not None:
                _cn = []
                with torch.no_grad():
                    for i in range(0, len(_op_tok), 256):
                        o = model(torch.from_numpy(_op_tok[i:i + 256]).to(DEVICE),
                                  torch.from_numpy(_op_msk[i:i + 256]).to(DEVICE))
                        _cn.append(corn_probs(o["logits_count_v2"], 5).cpu().numpy())
                _Po = np.concatenate(_cn)
            _info = {"source": "noc_head_v2 CORN (true-NOC target, identity-blind feats)",
                     "real_mixture_labels_used": 0, "n_combos": len(_gid)}
        elif _nm == "gate_cal":
            # THE COUNT WITHOUT ANY REAL MIXTURE LABEL. sum(gate) is the number of live AdaSlot slots, and it
            # is well ORDERED against NOC (corr .90-.98 measured) but mis-SCALED: it over-calls NOC3/4 and
            # under-calls NOC2/5, so round() throws most of that ordering away. Fit the one scalar -> k
            # mapping on the in-silico DEV split, which is combination-disjoint from the real test and holds
            # no real label. No real mixture annotation is touched, so one fit serves both protocols.
            from sklearn.linear_model import LogisticRegression
            _dev = ClosedSetDataset("dev")
            _Sd = _gate_sum(_dev.tokens.numpy(), _dev.mask.numpy().astype(bool))
            _yd = np.clip(_dev.noc.numpy(), 1, 5)
            _clf = LogisticRegression(max_iter=2000).fit(np.c_[_Sd, _Sd ** 2], _yd)
            print(f"\n  === GATE CALIBRATED COUNT (sum(gate) -> k, hieu chinh tren in-silico DEV "
                  f"{len(_Sd)} hang, KHONG dung nhan hon hop that) ===")
            print(f"  corr(sum gate, NOC) tren DEV {float(np.corrcoef(_Sd, _yd)[0, 1]):+.4f} | "
                  f"acc tren DEV {float((_clf.predict(np.c_[_Sd, _Sd ** 2]) == _yd).mean()):.4f}", flush=True)
            _P = np.zeros((len(noc_te), 5))
            _P[:, _clf.classes_.astype(int) - 1] = _clf.predict_proba(np.c_[S_te, S_te ** 2])
            _Po = None
            if _S_open is not None:
                _Po = np.zeros((len(_S_open), 5))
                _Po[:, _clf.classes_.astype(int) - 1] = _clf.predict_proba(
                    np.c_[_S_open, _S_open ** 2])
            _info = {"source": "sum(gate) calibrated on in-silico DEV", "n_dev": int(len(_Sd)),
                     "real_mixture_labels_used": 0, "n_combos": len(_gid)}
        elif _nm in HEAD_ARMS:
            print(f"\n  === HEAD {_nm} on FROZEN features (no LoRA) — {_K}-fold, combination-disjoint, "
                  f"{len(_gid)} combos over {len(noc_te)} rows ===")
            _k, _P, _Po = head_kfold(model, _tk, _mk_, noc_te, _grp, _K, seed, DEVICE, _nm, open_tok=_op_tok,
                                     open_msk=_op_msk, fold_log=_fl, budget=cfg["lora_budget"],
                                     budget_seed=cfg["lora_budget_seed"])
            _info = {"kfold": _K, "head": _nm, "lora": False, "n_combos": len(_gid),
                     "label_budget": cfg["lora_budget"], "label_budget_draw": cfg["lora_budget_seed"]}
        else:
            _inv = _nm in LORA_FEATS
            _lf = LORA_FEATS.get(_nm, {})
            _hdesc = ("noc_head copy, donor-indexed" if not _inv else
                      "SORTED " + " + ".join(["gate"] + (["ID probs"] if _lf["use_probs"] else [])
                                             + (["MAC physical"] if _lf["use_mac"] else []))
                      + ", identity-blind")
            print(f"\n  === NoC LoRA COUNT (backbone frozen + count-only LoRA r{cfg['lora_rank']}, head = "
                  + _hdesc
                  + f") — {_K}-fold, combination-disjoint, {len(_gid)} combos over {len(noc_te)} rows, "
                  + (f"{cfg['lora_epochs']} epochs, head lr {cfg['lora_head_lr']:.0e}"
                     if _inv and cfg['lora_epochs']
                     else f"{cfg['lora_steps']} steps per fit") + f", scope={cfg['lora_scope']} ===")
            _k, _P, _Po = noc_lora_kfold(model, _tk, _mk_, noc_te, _grp, _K, int(cfg["lora_steps"]),
                                         int(cfg["lora_rank"]), seed, DEVICE, open_tok=_op_tok,
                                         open_msk=_op_msk, invariant=_inv, scope=cfg["lora_scope"],
                                         epochs=(cfg["lora_epochs"] or None) if _inv else None,
                                         head_lr=cfg["lora_head_lr"] if _inv else 5e-5, fold_log=_fl,
                                         budget=cfg["lora_budget"], budget_seed=cfg["lora_budget_seed"], **_lf)
            _info = {"kfold": _K, "steps_per_fit": int(cfg["lora_steps"]), "rank": int(cfg["lora_rank"]),
                     "n_combos": len(_gid), "head": _hdesc,
                     "uses_id_posterior": bool(_lf.get("use_probs", True)) if _inv else True,
                     "scope": cfg["lora_scope"], "epochs": cfg["lora_epochs"] if _inv else None,
                     "head_lr": cfg["lora_head_lr"] if _inv else 5e-5,
                     "label_budget": cfg["lora_budget"], "label_budget_draw": cfg["lora_budget_seed"]}
        _alt = "expect" if cfg["count_decode"] == "argmax" else "argmax"
        _k = score_count_arm(_nm, _P, _info, _kt5, cfg["count_decode"],
                             folds=_fl, seconds=time.time() - _t0)
        _f5 = np.array(_info["per_noc_f1"])
        if cfg["protocol"] in ("deepnoc", "both") and _nm not in ("gate_cal", "zero_shot", "gate", "corn"):  # khong fit tren test
            _ev = np.arange(len(noc_te)); _dsp = [(_ev[::2], _ev[1::2])]      # fit even, score odd
            _te2 = _dsp[0][1]
            print(f"\n  === PROTOCOL deepnoc for {_nm}: every-second-profile 50/50, NO disjointness "
                  f"(deepNoC s2.6) — fit {len(_dsp[0][0])} rows, score {len(_te2)} rows, "
                  f"combinations SHARED on purpose ===")
            set_seed(seed)
            if _nm in HEAD_ARMS:
                _, _Pd, _ = head_kfold(model, _tk, _mk_, noc_te, _grp, 1, seed, DEVICE, _nm, splits=_dsp,
                                       budget=cfg["lora_budget"], budget_seed=cfg["lora_budget_seed"])
            else:
                _, _Pd, _ = noc_lora_kfold(model, _tk, _mk_, noc_te, _grp, 1, int(cfg["lora_steps"]),
                                           int(cfg["lora_rank"]), seed, DEVICE, invariant=_inv,
                                           scope=cfg["lora_scope"],
                                           epochs=(cfg["lora_epochs"] or None) if _inv else None,
                                           head_lr=cfg["lora_head_lr"] if _inv else 5e-5, splits=_dsp,
                                           budget=cfg["lora_budget"], budget_seed=cfg["lora_budget_seed"],
                                           **_lf)   # same feature set as the strict run, or the two differ
            _kd = decode_k(_Pd[_te2], cfg["count_decode"])
            _kd_alt = decode_k(_Pd[_te2], _alt)
            _y2 = _kt5[_te2]
            _f5d = f1_score(_y2, _kd, average=None, labels=[1, 2, 3, 4, 5], zero_division=0)
            _info["deepnoc"] = {"n_fit": int(len(_dsp[0][0])), "n_eval": int(len(_te2)),
                                "macro_f1": round(float(_f5d.mean()), 4),
                                "accuracy": round(float((_kd == _y2).mean()), 4),
                                "per_noc_f1": [round(float(v), 4) for v in _f5d],
                                "macro_f1_" + _alt + "_decode": round(f1_macro(_y2, _kd_alt), 4)}
            print(f"  [DEEPNOC] {_nm}: macroF1 {_f5d.mean():.4f} ({_alt} {f1_macro(_y2, _kd_alt):.4f}) | "
                  f"acc {float((_kd == _y2).mean()):.4f} | per-NOC F1 "
                  + " ".join(f"{v:.3f}" for v in _f5d)
                  + f" | strict {_info['macro_f1']:.4f} -> premium {_f5d.mean() - _info['macro_f1']:+.4f}",
                  flush=True)
        print(f"  count {_nm} [K-fold pooled]: acc {float((_k == _kt5).mean()):.4f}  "
              f"acc_mix {float((_k[mix_] == noc_te[mix_]).mean()):.4f}  macroF1 {_f5.mean():.4f}  "
              f"per-NOC F1 " + " ".join(f"{v:.3f}" for v in _f5) + f"  ({_info['seconds']:.0f}s)")
        if _Po is not None and _op_true is not None:
            score_open_split(_nm, _Po, _info, _op_true, _op_panel, cfg["count_decode"], _K)
        _arms[_nm] = _info
        _cands.append((f"noc_{_nm}", _k))
        _cand_dec[f"noc_{_nm}"] = cfg["count_decode"]
        if _nm == cfg["count"]:                       # the shipped one
            k_post, p_count, count_info, _p_open = _k, _P, _info, _Po
    assert k_post is not None, f"count '{cfg['count']}' produced nothing"
    k_src = f"noc_{cfg['count']}"
    print(f"  {'per-NOC recall':<22}{'macroF1':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    for nm, kk in _cands:
        _r5 = [float((kk[_kt5 == v] == v).mean()) if (_kt5 == v).any() else float("nan") for v in range(1, 6)]
        print(f"  {nm + '/' + _cand_dec[nm]:<22}{f1_macro(_kt5, kk):>8.3f}"
              + "".join(f"{x:>7.3f}" for x in _r5))

    sel_curve = count_diagnostics(DATA_DIR, noc_te, _kt5, mix_, _cands, p_count, k_src)

    y_te_pred = topk_decode(rank_te, k_post)
    em_post = per_noc_em(y_te_true, y_te_pred, noc_te)
    oracle = per_noc_em(y_te_true, topk_decode(rank_te, noc_te), noc_te)
    count_acc = float((np.clip(k_post, 1, 5) == _kt5).mean())

    print(f"\n  {'decode':<14}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    for nm, r in (("oracle", oracle), (f"k={k_src}", em_post)):
        print(f"  {nm:<14}" + "".join(f"{x:>7.3f}" for x in r))
    print(f"  count accuracy (k={k_src}): {count_acc:.4f}")
    # recall is the column comparable to published NoC tables; deepNoC's per-NOC row is one-vs-rest
    _kp = np.clip(k_post, 1, 5)
    _rec = [float((_kp[_kt5 == v] == v).mean()) if (_kt5 == v).any() else None for v in range(1, 6)]
    _ovr = [float(((_kp == v) == (_kt5 == v)).mean()) for v in range(1, 6)]
    _mf1 = f1_macro(_kt5, _kp)
    print(f"  {'count metric':<14}{'macroF1':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    print(f"  {'recall':<14}{_mf1:>8.3f}" + "".join(f"{'   n/a' if x is None else f'{x:>7.3f}'}" for x in _rec))
    print(f"  {'(one-vs-rest)':<14}{'':>8}" + "".join(f"{x:>7.3f}" for x in _ovr))

    # DEV per-NOC oracle on the reranked ranking (combo-generalization judge)
    dev_ds = ClosedSetDataset("dev")
    L_dev, P_dev, y_dev, noc_dev, _, _, _ = infer(dev_ds)
    PHd = pr.deconv_phi(dev_ds.tokens.numpy(), dev_ds.mask.numpy(), dg, dgm)
    rank_dev = pr.rerank_scores(L_dev, PHd, alpha_pr)
    dev_oracle = per_noc_oracle_on_scores(rank_dev, y_dev, noc_dev)
    print(f"  DEV per-NOC oracle (reranked): {dev_oracle}")

    # ALPHA WITHOUT A REAL LABEL. The shipped alpha above is tuned leave-one-combo-out on the REAL test
    # annotations, which is honest but not label-free: a new lab with no annotated mixtures cannot do it.
    # Tune the same alpha on the in-silico DEV mixtures instead (no real label anywhere) and re-score the
    # test with it, so the report carries the number a label-free deployment would actually get.
    _dmix = noc_dev >= 2
    alpha_dev = float(pr.tune_alpha(L_dev[_dmix], PHd[_dmix], y_dev[_dmix], noc_dev[_dmix])) if _dmix.any() else 0.3
    _em_dev_alpha = per_noc_em(y_te_true, topk_decode(pr.rerank_scores(L_te, PHt, alpha_dev), k_post), noc_te)
    _or_dev_alpha = per_noc_em(y_te_true, topk_decode(pr.rerank_scores(L_te, PHt, alpha_dev), noc_te), noc_te)
    print(f"  [ALPHA KHONG NHAN THAT] alpha tu in-silico DEV = {alpha_dev} (alpha LOCO tren nhan that = "
          f"{alpha_pr}) -> EM@k {_em_dev_alpha[0]:.4f} (LOCO: {em_post[0]:.4f}, chenh "
          f"{_em_dev_alpha[0] - em_post[0]:+.4f}) | oracle EM {_or_dev_alpha[0]:.4f} (LOCO: {oracle[0]:.4f})")

    # reject AUROC (the reject head is part of the inc22 objective)
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

    np.save(results_dir / "y_test_pred.npy", y_te_pred)
    np.save(results_dir / "y_test_true.npy", y_te_true)
    np.save(results_dir / "count_proba.npy", p_count)

    out_dict = {
        "model": "set_transformer", "config": cfg,
        "best_selection_score": round(best_sel, 4), "best_epoch": best_epoch,
        "decode": decode_desc,
        "eval_n": len(test_ds),
        "phi_rerank_alpha": float(alpha_pr),
        "phi_rerank_alpha_per_group": alphas,
        "n_decode_groups": n_groups,
        "n_test_mixtures": int((noc_te >= 2).sum()),
        "em_at_pred_k": round(float(em_post[0]), 4),
        "count_source": k_src,
        "oracle_em": round(float(oracle[0]), 4),
        "count_acc": round(count_acc, 4),
        "count_macro_f1": round(_mf1, 4),
        "count_recall_per_noc": {str(v): (None if _rec[v - 1] is None else round(_rec[v - 1], 4)) for v in range(1, 6)},
        "count_one_vs_rest_per_noc": {str(v): round(_ovr[v - 1], 4) for v in range(1, 6)},
        "count_info": count_info,
        "alpha_dev_no_real_label": {"alpha": alpha_dev, "em_at_pred_k": round(float(_em_dev_alpha[0]), 4),
                                    "oracle_em": round(float(_or_dev_alpha[0]), 4)},
        "count_macro_f1_all_sources": {nm: round(f1_macro(_kt5, kk), 4) for nm, kk in _cands},
        # which decode produced each entry above: the same readout under two decodes is two numbers
        "count_decode_per_source": dict(_cand_dec),
        "count_arms": _arms,                 # every count run (shipped + --compare), with per-fold gap
        "selective_classification": sel_curve,
        "count_acc_gate": round(float((k_gate == _kt5).mean()), 4),
        "count_macro_f1_gate_mix": round(float(f1_score(
            _kt5[mix_], k_gate[mix_], average="macro", labels=[2, 3, 4, 5],
            zero_division=0)), 4),
        "corr_sumgate_noc": round(corr_gate, 4),
        # per-row sum(gate). corr is .95 while round(sum gate) only scores .72, so the ordering is there
        # and the FIXED half-integer thresholds are what lose it. Storing the raw array lets the exact
        # ceiling of ANY monotone threshold rule be computed later with no training at all.
        "sum_gate_per_row": [round(float(v), 4) for v in S_te],
        "noc_true_per_row": [int(v) for v in noc_te],
        "reject_auroc": auroc,
        "per_noc_oracle": _pn_(oracle),
        "per_noc_at_pred_k": _pn_(em_post),
        "dev_per_noc_oracle": dev_oracle,
        "history": history,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved -> {results_dir}")

    write_report(results_dir, ROOT, DATA_DIR, cfg, seed, eval_label, model, noc_te, k_post, k_gate,
                 y_te_true, y_te_pred, topk_decode(rank_te, noc_te), rank_te, L_te, combo_id,
                 len(open_ds), oracle, em_post, auroc, alpha_pr, count_info, k_src, _mf1,
                 _param_fp_before)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the clean inc22_fixed_aslot pipeline (one archive fold).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default="inc22_fixed_aslot")
    ap.add_argument("--init_from", type=str, default=None,
                    help="best_model.pt (or its dir) of THIS fold's Phase 1; skips Phase 1")
    ap.add_argument("--epochs", type=int, default=CFG["epochs"], help="Phase 1 epochs (150; lower only for smoke tests)")
    ap.add_argument("--count", choices=COUNT_ARMS, default=CFG["count"],
                    help="lora_inv (default) = count-only LoRA read through an identity-blind head on "
                         "the SORTED gate; lora = the same LoRA read through the donor-indexed noc_head; "
                         "zero_shot / gate / gate_cal need no real mixture label at all")
    ap.add_argument("--w_gate", type=float, default=CFG["w_gate"],
                    help="trong so SmoothL1(sum gate, NOC). 0.05 = mac dinh; nhanh synznguyen dat het tin "
                         "hieu dem vao day nen gate cua ho tot hon han gate cua ta (mix F1 .8632 vs .6937)")
    ap.add_argument("--beta_card", type=float, default=CFG["beta_card"],
                    help="trong so CE(logits_card, k toi uu EM). 0.3 = mac dinh; nhanh kia khong co so hang nay")
    ap.add_argument("--protocol", choices=["strict", "deepnoc", "both"], default=CFG["protocol"],
                    help="strict = combination-disjoint K-fold; deepnoc = every-second-profile 50/50 "
                         "(deepNoC s2.6, combinations shared); both = report both")
    ap.add_argument("--count_decode", choices=["argmax", "expect"], default=CFG["count_decode"],
                    help="argmax = top class; expect = round the expected value (keeps the NoC order)")
    ap.add_argument("--compare", nargs="*", default=[], choices=COUNT_ARMS,
                    help="extra counts run on the same backbone and reported beside --count, never shipped")
    ap.add_argument("--count_kfold", type=int, default=CFG["count_kfold"], help="K of the combination-disjoint folds")
    ap.add_argument("--lora_steps", type=int, default=CFG["lora_steps"], help="gradient steps per NoC LoRA fit")
    ap.add_argument("--lora_rank", type=int, default=CFG["lora_rank"])
    ap.add_argument("--lora_epochs", type=int, default=CFG["lora_epochs"],
                    help="lora_inv budget in epochs; 0 -> use --lora_steps")
    ap.add_argument("--lora_head_lr", type=float, default=CFG["lora_head_lr"])
    ap.add_argument("--lora_budget", type=str, default=None,
                    help="most labelled rows per NOC1..5 per LoRA fit, e.g. 34,87,79,93,78 (deepNoC's fine-tune set)")
    ap.add_argument("--lora_budget_seed", type=int, default=0, help="which random draw of the budget rows")
    ap.add_argument("--lora_scope", choices=["all", "encoder", "encoder_attn", "decoder"],
                    default=CFG["lora_scope"], help="which matrices the count LoRA may change")
    args = ap.parse_args()
    CFG.update(epochs=args.epochs, count=args.count, compare=args.compare, count_kfold=args.count_kfold,
               count_decode=args.count_decode, protocol=args.protocol,
               w_gate=args.w_gate, beta_card=args.beta_card,
               lora_steps=args.lora_steps, lora_rank=args.lora_rank, lora_scope=args.lora_scope,
               lora_epochs=args.lora_epochs, lora_head_lr=args.lora_head_lr,
               lora_budget=[int(v) for v in args.lora_budget.split(",")] if args.lora_budget else None,
               lora_budget_seed=args.lora_budget_seed)
    train(args.seed, args.out_subdir, args.init_from)
