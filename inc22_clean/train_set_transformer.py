"""
train_set_transformer.py — CLEAN, STANDALONE trainer for the `inc22_fixed_aslot` arm ONLY.

Faithful extraction of the single inc22 code path from train_set_transformer.py:
  * model  = SetTransformerMixture (CoSA+GSANet+MESH+AdaSlot, isab++/nc_mab0, periodic embed, aux heads,
             set_of_set, feas_filter). See models/set_transformer.py.
  * loss   = ASL(gamma_neg=4) on logits_cls
             + 0.5 * BCE(reject, closed-0 / open-1)
             + 0.3 * CE(logits_card, EM-optimal-k target, class-weighted)
             + Kendall( soft_attr_label CE  +  L1 phi )         (aux_heads, soft_attr_label)
  * train aug = mask_peaks 0.15 (drop shared peaks only; never a minor's private/single-carrier peak)
  * selection = macro-over-NOC oracle Recall@k on the in-silico DEV split (else real val)

DECODE:
  1. phi_rerank: independent EM mixture-proportion deconvolution -> logarithmic-opinion-pool rerank
     of the per-donor logits (alpha tuned on val).  Reranks the RANKING (which donors) only.
  2. COUNT = post-hoc RandomForest on the PROB profile (posthoc_cardinality, the original inc22 count).
     NOT count-on-rerank: counting on the LOP-reranked score ('lop count') trades N3/N4 down for N5
     (measured ~-13pp N3, -7pp N4) — rejected.  The learned CORN count head (noc_head_v2) is likewise
     EXCLUDED (it collapses N3/N4 ~0.21/0.17).
  3. decode top-k_post of the reranked ranking.  oracle = top-true-k of the reranked ranking (ceiling).

Everything inc22 does NOT use was removed (no replicates / em_phi / noise_gate / ref_match /
soft_geno_attr / sparse_attn / minor_weight / irm / vicreg / rnc / distill / CORN / the flag zoo).

Run (STR_DATA_DIR points at the enriched, dev-split data dir; donor_geno.npy must be present there
or under ./data):
    STR_DATA_DIR=<data_dir> python train_set_transformer.py --seed 42 --out_subdir inc22_fixed_aslot
Writes results/<out_subdir>_seed<seed>/{best_model.pt, metrics.json, y_test_pred.npy, y_test_true.npy}.
"""
from __future__ import annotations

import argparse
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
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
import phi_rerank as pr

# Data dir (the orchestrator sets STR_DATA_DIR to the enriched, dev-split data_w).
DATA_DIR = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data_insilico_w")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALLELE_OFF = 30
LUT_W = 1024

# ── Fixed inc22 configuration (the `configs/set_transformer.json` base + the inc22 flags) ──
CFG = {
    "n_loci": 24, "d_locus": 16, "d_model": 128, "n_heads": 4, "n_isab": 2, "m_inducing": 32,
    "n_classes": 45, "dropout": 0.1,
    "lr": 6e-4, "weight_decay": 1e-4, "batch_size": 256, "epochs": 150, "patience": 25,
    "alpha_reject": 0.5, "beta_card": 0.3, "card_lambda": 0.02, "open_ratio": 0.25,
    "n_token_feats": 8, "num_embed": "periodic", "periodic_sigma": 0.3, "n_freq": 8, "d_num_emb": 8,
    "encoder": "isab++", "nc_attn": "mab0", "cls_decoder": "aslot",
    "aux_heads": True, "set_of_set": True, "feas_filter": True, "soft_attr_label": True,
    "mask_peaks": 0.15, "mask_peaks_min": 8,
    "n_slot_iters": 3, "ot_eps": 0.05, "ot_iters": 5, "gumbel_temp": 1.0,
    "loss": "asl", "asl_gamma_neg": 4.0, "asl_gamma_pos": 0.0, "asl_clip": 0.05,
    # decode: phi-rerank the RANKING, count = post-hoc RF on the PROB profile (CORN excluded — it
    # collapses N3/N4; count-on-rerank trades N3/N4 for N5).
    "phi_rerank": True, "count_on_probs": True,
}


def set_seed(seed: int) -> torch.Generator:
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


def _card_feats(P):
    s = np.sort(P, 1)[:, ::-1][:, :8]
    return np.concatenate([s, P.sum(1, keepdims=True), (P >= 0.5).sum(1, keepdims=True)], 1)


def posthoc_cardinality(P_val, y_val, P_test, lam=0.02):
    """ORIGINAL inc22 count: post-hoc RandomForest on the sorted PROB profile, fit on val with the
    EM-optimal-k cost target.  Count on PROBS — NOT on the phi-reranked score: counting on the rerank
    ('lop count') trades N3/N4 down for N5 (measured ~-13pp N3, -7pp N4), which violates 'no trading'."""
    C = P_val.shape[1]; tk = np.ones(len(P_val), int)
    for i in range(len(P_val)):
        K = max(int(y_val[i].sum()), 1); best, bc = 1, 9e9
        for k in range(1, 6):
            yp = np.zeros(C); yp[np.argsort(P_val[i])[::-1][:k]] = 1
            c = (y_val[i] * (1 - yp)).sum() / K + ((1 - y_val[i]) * yp).sum() / k + lam * k
            if c < bc: bc, best = c, k
        tk[i] = best
    return RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42).fit(
        _card_feats(P_val), tk).predict(_card_feats(P_test))


def per_noc_em(y_true, y_pred, noc):
    e = np.all(y_true == y_pred, axis=1)
    return [e.mean()] + [e[noc == j].mean() if (noc == j).sum() else float("nan") for j in range(1, 6)]


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


def train(seed: int, out_subdir: str):
    cfg = dict(CFG); cfg["seed"] = seed; cfg["out_subdir"] = out_subdir
    results_dir = ROOT / "results" / f"{out_subdir}_seed{seed}"
    results_dir.mkdir(parents=True, exist_ok=True)
    gen = set_seed(seed)
    print(f"seed={seed}  data={DATA_DIR}  device={DEVICE}")

    train_ds = ClosedSetDataset("train"); val_ds = ClosedSetDataset("val")
    test_ds = ClosedSetDataset("test"); open_ds = OpenSetDataset()
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    if (DATA_DIR / "tokens8_dev.npy").exists():
        sel_loader = DataLoader(ClosedSetDataset("dev"), batch_size=256, shuffle=False)
        print(f"selection set = in-silico DEV ({len(sel_loader.dataset)} samples)")
    else:
        sel_loader = val_loader; print("selection set = real val (no dev split found)")
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

    best_sel, best_epoch, patience_count = 0.0, 0, 0
    patience = cfg["patience"]; epochs = cfg["epochs"]; history = []
    print(f"\nTraining up to {epochs} epochs (patience={patience}) ...")
    t0 = time.time()

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
        scheduler.step(val_macrec)
        history.append({"epoch": epoch, "loss": round(epoch_loss, 4), "val_macro_f1": round(val_f1, 4),
                        "val_oracle_em": round(val_em, 4), "val_macro_recall": round(val_macrec, 4)})
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | loss={epoch_loss:.4f} (cls={epoch_cls/n:.3f} rej={epoch_rej/n:.3f} "
                  f"noc={epoch_noc_l/n:.3f} attr={epoch_attr/n:.3f} phi={epoch_phi/n:.3f}) "
                  f"| val_macrec={val_macrec:.4f} val_em={val_em:.4f} val_f1={val_f1:.4f} "
                  f"| lr={optimizer.param_groups[0]['lr']:.1e}")

        if val_macrec > best_sel:
            best_sel, best_epoch, patience_count = val_macrec, epoch, 0
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stop at ep {epoch} (best ep {best_epoch}, val macro-recall={best_sel:.4f})")
                break
    print(f"Training done in {time.time()-t0:.1f}s")

    # ── Test eval: phi-rerank -> post-hoc RF count on the reranked score ────────
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.eval()

    @torch.no_grad()
    def infer(ds):
        loader = DataLoader(ds, batch_size=256, shuffle=False)
        L, Y, NOC = [], [], []
        for tokens, mask, y, noc, *_ in loader:
            L.append(model(tokens.to(DEVICE), mask.to(DEVICE))["logits_cls"].cpu().numpy())
            Y.append(y.numpy()); NOC.append(noc.numpy())
        L = np.concatenate(L)
        return L, 1.0 / (1.0 + np.exp(-L)), np.concatenate(Y), np.concatenate(NOC)

    L_te, P_te, y_te_true, noc_te = infer(test_ds)
    L_va, P_va, y_va, noc_va = infer(val_ds)

    dg = donor_geno.cpu().numpy(); dgm = donor_geno_mask.cpu().numpy()
    PHv = pr.deconv_phi(val_ds.tokens.numpy(), val_ds.mask.numpy(), dg, dgm)
    PHt = pr.deconv_phi(test_ds.tokens.numpy(), test_ds.mask.numpy(), dg, dgm)
    alpha_pr = pr.tune_alpha(L_va, PHv, y_va, noc_va)
    rank_te = pr.rerank_scores(L_te, PHt, alpha_pr)        # phi-rerank the RANKING (which donors / decode order)
    print(f"phi_rerank: val-tuned alpha={alpha_pr}")

    # COUNT = post-hoc RF on the PROB profile (original inc22; count-on-rerank trades N3/N4 -> excluded)
    k_post = posthoc_cardinality(P_va, y_va, P_te)
    y_te_pred = topk_decode(rank_te, k_post)
    em_post = per_noc_em(y_te_true, y_te_pred, noc_te)
    oracle = per_noc_em(y_te_true, topk_decode(rank_te, noc_te), noc_te)
    count_acc = float((np.clip(k_post, 1, 5) == np.clip(noc_te, 1, 5)).mean())

    print(f"  {'decode':<12}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    for nm, r in (("oracle", oracle), ("post-hoc", em_post)):
        print(f"  {nm:<12}" + "".join(f"{x:>7.3f}" for x in r))
    print(f"  post-hoc count accuracy: {count_acc:.4f}")

    # DEV per-NOC oracle on the reranked ranking (combo-generalization judge)
    dev_oracle = None
    if (DATA_DIR / "tokens8_dev.npy").exists():
        dev_ds = ClosedSetDataset("dev")
        L_dev, P_dev, y_dev, noc_dev = infer(dev_ds)
        PHd = pr.deconv_phi(dev_ds.tokens.numpy(), dev_ds.mask.numpy(), dg, dgm)
        rank_dev = pr.rerank_scores(L_dev, PHd, alpha_pr)
        dev_oracle = per_noc_oracle_on_scores(rank_dev, y_dev, noc_dev)
        print(f"  DEV per-NOC oracle (reranked): {dev_oracle}")

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

    def _pn(lst):
        return {str(j): (None if np.isnan(lst[j]) else round(float(lst[j]), 4)) for j in range(1, 6)}

    out_dict = {
        "model": "set_transformer", "config": cfg,
        "best_val_macro_recall": round(best_sel, 4), "best_epoch": best_epoch,
        "decode": "phi_rerank(ranking) + posthoc_rf_count(probs)",
        "phi_rerank_alpha": float(alpha_pr),
        "em_post_hoc": round(float(em_post[0]), 4),
        "oracle_em": round(float(oracle[0]), 4),
        "count_acc": round(count_acc, 4),
        "reject_auroc": auroc,
        "per_noc_oracle": _pn(oracle),
        "per_noc_post_hoc": _pn(em_post),
        "dev_per_noc_oracle": dev_oracle,
        "history": history,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved -> {results_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the clean inc22_fixed_aslot pipeline.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_subdir", type=str, default="inc22_fixed_aslot")
    args = ap.parse_args()
    train(args.seed, args.out_subdir)
