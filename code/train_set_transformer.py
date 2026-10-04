"""
train_set_transformer.py — trainer + test-time decode for the set-transformer contributor model.

  * model  = SetTransformerMixture (CoSA+GSANet+MESH+AdaSlot, isab++/nc_mab0, periodic embed, aux heads,
             set_of_set, feas_filter). See models/set_transformer.py.
  * loss   = ASL(gamma_neg=4) on logits_cls
             + 0.05 * SmoothL1(sum gate, NOC)                   (the live AdaSlot slots ARE the count)
             + 0.3 * CORN(noc_head_v2)                          (detached inputs; trained, never decoded)
             + Kendall( soft_attr_label CE  +  L1 phi )         (aux_heads, soft_attr_label)
  * train aug = mask_peaks 0.15 on shared peaks (private peaks are never masked)
  * selection = macro-over-NOC oracle Recall@k on the val split. No early stopping;
                best_model.pt = best DEV epoch, last_model.pt = final epoch.

Training sees the CLOSED set only (panel donors); the open split (samples with a donor outside the panel) is test.

DECODE (decode_layer.py, every parameter chosen on val, no real mixture label is read):
  penalty + bonus on the gate -> k = round(sum corrected gate); order = set-conditional greedy on the
  phi-reranked logits; the k-th pick is dropped when it explains almost nothing new; open score from how well the
  decoded set fits the panel, flagged above the 95th percentile of val.

Run (STR_DATA_DIR = an enriched data dir with gen_law.npy):
    STR_DATA_DIR=<data_dir> python train_set_transformer.py --seed 42 --out_subdir <name>
Fine-tune:  STR_INIT_FROM=<ckpt> STR_EPOCHS=3 STR_LR=1e-4 ...      Evaluate only:  STR_EVAL_ONLY=1 STR_INIT_FROM=<ckpt>
Writes results/<out_subdir>_seed<seed>/{best_model.pt, last_model.pt, metrics.json, y_test_pred.npy, y_test_true.npy}.
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
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from models.ordinal import corn_loss
import decode_layer as DL
import gen_law as gl

# Data dir (the orchestrator sets STR_DATA_DIR to the enriched data_w).
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

# ── Fixed configuration ──
CFG = {
    "n_loci": 24, "d_locus": 16, "d_model": 128, "n_heads": 4, "n_isab": 2, "m_inducing": 32,
    "n_classes": 45, "dropout": 0.1,
    "lr": 6e-4, "weight_decay": 1e-4, "batch_size": 256, "epochs": 150,   # no early stopping
    "n_token_feats": 8, "num_embed": "periodic", "periodic_sigma": 0.3, "n_freq": 8, "d_num_emb": 8,
    "encoder": "isab++", "nc_attn": "mab0", "cls_decoder": "aslot",
    "aux_heads": True, "set_of_set": True, "feas_filter": True, "soft_attr_label": True,
    "mask_peaks": 0.15, "mask_peaks_min": 8,
    "n_slot_iters": 3, "ot_eps": 0.05, "ot_iters": 5, "gumbel_temp": 1.0,
    "loss": "asl", "asl_gamma_neg": 4.0, "asl_gamma_pos": 0.0, "asl_clip": 0.05,
    # CORN ordinal count head on DETACHED features. Decode never reads it, but the published run
    # trained with it and its gradient enters the global clip_grad_norm_ denominator; dropping it
    # measurably hurt (EM .9162 vs .9260, count error 5.37% vs 3.74%).
    "noc_head_v2": True, "noc_v2_weight": 0.3,
    # tier B: the number of live AdaSlot slots IS the count. Fold 0 — corr(sum gate, NOC) flips
    # -0.966 -> +0.980, median sum(gate) lands on 1/2/3/4/5, and the tier-A readout on that backbone
    # reaches count .9414 (vs .9361) with ID oracle unharmed. gate is NOT detached and feeds
    # logits_cls via gate_logit, so this reaches the encoder — that is the point.
    "w_gate": 0.05,            # weight on SmoothL1(sum gate, NOC)
}
# Warm start + schedule overrides, for finetuning an existing checkpoint on a new data mix without
# forking the loss. STR_INIT_FROM loads weights AFTER feat_mean/feat_std are recomputed, so the
# checkpoint's own normalisation is what survives — a finetune must keep the statistics it was
# trained under. Unset = train from scratch, i.e. the shipped behaviour is unchanged.
INIT_FROM = os.environ.get("STR_INIT_FROM", "").strip()
# STR_EVAL_ONLY=1: no training - load STR_INIT_FROM and run the test-time decode/eval exactly as after training.
EVAL_ONLY = os.environ.get("STR_EVAL_ONLY", "0") == "1"
CFG["epochs"] = int(os.environ.get("STR_EPOCHS", CFG["epochs"]))
CFG["lr"] = float(os.environ.get("STR_LR", CFG["lr"]))
# STR_BATCH: 256 is the trained recipe; a 4 GB card needs 128 (256 spills out of VRAM at MAX_SEQ 224).
CFG["batch_size"] = int(os.environ.get("STR_BATCH", CFG["batch_size"]))
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


def _pn_(lst):
    """per-NOC dict from a per_noc_em list (index 0 = overall)."""
    return {str(j): (None if np.isnan(lst[j]) else round(float(lst[j]), 4)) for j in range(1, 6)}


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

        # the generator's TRUE per-peak split, when the set carries it (in-silico rows only)
        fp = DATA_DIR / f"attrfrac_{split}.npy"; cp = DATA_DIR / f"attrcol_{split}.npy"
        if fp.exists() and cp.exists():
            self.frac = torch.from_numpy(np.load(fp).astype(np.float32))
            self.fcol = torch.from_numpy(np.load(cp).astype(np.int64))
        else:
            self.frac = torch.zeros((N, S, 5)); self.fcol = torch.full((N, 5), -1, dtype=torch.int64)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return (self.tokens[i], self.mask[i], self.y[i], self.noc[i], self.attr[i], self.phi[i],
                self.frac[i], self.fcol[i])


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
    print(f"w_gate={cfg['w_gate']}  noc_head_v2={cfg['noc_head_v2']}  mask_peaks={cfg['mask_peaks']}")

    train_ds = ClosedSetDataset("train"); val_ds = ClosedSetDataset("val")
    test_ds = ClosedSetDataset("test"); open_ds = OpenSetDataset()
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=gen,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)
    sel_loader = val_loader
    print(f"selection set = val ({len(val_ds)} samples: real NOC1 + combo-disjoint in-silico mixtures)")

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
    if INIT_FROM:
        model.load_state_dict(torch.load(INIT_FROM, map_location=DEVICE, weights_only=True), strict=False)
        print(f"warm start <- {INIT_FROM}")

    # ── Loss objects + weights ────────────────────────────────────────────────
    bce_cls = AsymmetricLoss(gamma_neg=cfg["asl_gamma_neg"], gamma_pos=cfg["asl_gamma_pos"], clip=cfg["asl_clip"])
    w_gate = float(cfg["w_gate"])
    noc_v2_w = float(cfg["noc_v2_weight"])
    # Kendall homoscedastic-uncertainty weights for the privileged aux losses
    log_var_attr = torch.zeros((), device=DEVICE, requires_grad=True)
    log_var_phi  = torch.zeros((), device=DEVICE, requires_grad=True)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    optimizer.add_param_group({"params": [log_var_attr, log_var_phi], "weight_decay": 0.0})
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5,
                                                           patience=5, min_lr=1e-6)

    mask_peaks_p = cfg["mask_peaks"]; mask_min = cfg["mask_peaks_min"]

    def gather_owner(tok):
        loc = tok[:, :, 0].long().clamp(0, 23)
        ab = (torch.round(tok[:, :, 1] * 10).long() + ALLELE_OFF).clamp(0, owner_lut.size(1) - 1)
        return owner_lut[loc, ab]

    best_sel, best_epoch = 0.0, 0
    epochs = cfg["epochs"]; history = []
    if EVAL_ONLY:                                                  # see EVAL_ONLY
        assert INIT_FROM, "STR_EVAL_ONLY needs STR_INIT_FROM"
        epochs = 0
        torch.save(model.state_dict(), results_dir / "best_model.pt")
        torch.save(model.state_dict(), results_dir / "last_model.pt")
    print(f"\nTraining {epochs} epochs (no early stopping) ...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = epoch_cls = epoch_attr = epoch_phi = 0.0
        for tokens, mask, y, noc, attr, phi, frac, fcol in train_loader:
            tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)
            y, noc = y.to(DEVICE), noc.to(DEVICE); attr = attr.to(DEVICE); phi = phi.to(DEVICE)
            frac = frac.to(DEVICE); fcol = fcol.to(DEVICE)

            # mask_peaks: drop a fraction of the SHARED peaks, keeping >= mask_min so NOC stays identifiable.
            # A private peak (carried by exactly one donor) is never dropped: it is what identifies that donor.
            if mask_peaks_p > 0.0:
                mb = mask.bool()
                r = torch.rand_like(mb, dtype=torch.float)
                priv = gather_owner(tokens).sum(-1) == 1
                drop = mb & ~priv & (r < mask_peaks_p)
                kept = mb & ~drop
                enough = kept.sum(1, keepdim=True) >= mask_min
                mask = torch.where(enough, kept, mb).to(mask.dtype)

            out = model(tokens, mask)
            loss_cls = bce_cls(out["logits_cls"], y)

            loss = loss_cls

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
                # WHERE THE GENERATOR KNOWS, USE WHAT IT KNOWS. The line above rebuilds the split from phi x copy
                # number, which is a model of the split, not the split: it cannot tell that a faint contributor put
                # 12% into this peak and nothing into that one. The generator computed the real thing and it is now
                # kept (attrfrac_*), so those rows are supervised on it and only the real single sources fall back.
                have_ = (fcol >= 0).any(1)
                if have_.any():
                    tgt = torch.zeros_like(soft_y_)
                    ix = fcol.clamp(min=0).unsqueeze(1).expand(-1, frac.size(1), -1)
                    tgt.scatter_(2, ix, frac * (fcol >= 0).unsqueeze(1).float())
                    tgt[..., -1] = (1.0 - tgt[..., :-1].sum(-1)).clamp(min=0.0)   # the rest is artefact/noise
                    soft_y_ = torch.where(have_.view(-1, 1, 1), tgt, soft_y_)
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
            epoch_attr += loss_attr_v * bs; epoch_phi += loss_phi_v * bs

        n = len(train_ds); epoch_loss /= n
        val_f1 = evaluate_closed(model, val_loader)
        val_em, val_macrec = evaluate_oracle_em(model, sel_loader)
        sel_score = val_macrec
        scheduler.step(sel_score)
        history.append({"epoch": epoch, "loss": round(epoch_loss, 4), "val_macro_f1": round(val_f1, 4),
                        "val_oracle_em": round(val_em, 4), "val_macro_recall": round(val_macrec, 4)})
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | loss={epoch_loss:.4f} (cls={epoch_cls/n:.3f} "
                  f"attr={epoch_attr/n:.3f} phi={epoch_phi/n:.3f}) "
                  f"| DEV macrec={val_macrec:.4f} em={val_em:.4f} f1={val_f1:.4f} "
                  + f"| lr={optimizer.param_groups[0]['lr']:.1e}")

        # NO early stopping: the selection metric saturates and a patience counter stops on noise.
        # Always run the full `epochs`. `best_model.pt` = best val epoch, `last_model.pt` =
        # final epoch, so both can be evaluated.
        if sel_score > best_sel:
            best_sel, best_epoch = sel_score, epoch
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        torch.save(model.state_dict(), results_dir / "last_model.pt")
    print(f"Training done in {time.time()-t0:.1f}s")

    # ── Test eval: the decode layer (decode_layer.py), every parameter chosen on val ─────────
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.eval()
    dg = donor_geno.cpu().numpy(); dgm = donor_geno_mask.cpu().numpy().astype(bool)
    law = gl.load_law(DATA_DIR / gl.LAW_FILE)

    def split(ds):
        return DL.Split(model, ds.tokens.numpy(), ds.mask.numpy(), ds.y.numpy(), ds.noc.numpy(), dg, dgm, law, DEVICE)
    dev = split(val_ds); te = split(test_ds)
    dcfg = DL.fit(dev)
    y_te_pred, k_te = DL.apply(te, dcfg)
    y_te_true, noc_te = te.y, te.noc
    n_cls = y_te_true.shape[1]

    em_post = per_noc_em(y_te_true, y_te_pred, noc_te)
    oracle = per_noc_em(y_te_true, DL.to_sets(te.picks, noc_te, n_cls), noc_te)
    k_gate = np.clip(np.rint(te.G.sum(1)), 1, 5).astype(int)

    def count_per_noc(k):
        return [float((k == noc_te).mean())] + [float((k[noc_te == j] == j).mean()) if (noc_te == j).any()
                                                else float("nan") for j in range(1, 6)]
    noc_per = count_per_noc(k_te); gate_per = count_per_noc(k_gate)
    mix_ = noc_te >= 2

    def macro(lst):
        return float(np.nanmean(lst[2:6]))
    print(f"  {'':<26}{'overall':>8}{'NOC1':>7}{'NOC2':>7}{'NOC3':>7}{'NOC4':>7}{'NOC5':>7}")
    for nm, r in (("ID @ true k (greedy order)", oracle), ("ID @ decoded k", em_post),
                  ("NOC acc (decoded)", noc_per), ("NOC acc (round sum gate)", gate_per)):
        print(f"  {nm:<26}" + "".join(f"{x:>7.3f}" for x in r))
    print(f"  macro NOC2-5: ID {macro(em_post):.4f}  count {macro(noc_per):.4f}   "
          f"under {int(((k_te < noc_te) & mix_).sum())} / over {int(((k_te > noc_te) & mix_).sum())}")

    # ── full precision / recall / F1 (per class, macro = unweighted mean over classes, micro = pooled) ──
    rep_count = DL.multiclass_report(noc_te, np.clip(k_te, 1, 5), [1, 2, 3, 4, 5])
    rep_id = {"all": DL.multilabel_report(y_te_true, y_te_pred), "mixtures": DL.multilabel_report(y_te_true[mix_], y_te_pred[mix_])}

    # ── open set: samples with a donor outside the panel, never trained on. Closed test = 0, open = 1. ──
    dcfg = DL.fit_open(dev, dcfg)
    sc_te, fl_te = DL.open_score(te, dcfg, k_te)
    op = DL.Split(model, open_ds.tokens.numpy(), open_ds.mask.numpy(), None, None, dg, dgm, law, DEVICE)
    _, k_op = DL.apply(op, dcfg)
    sc_op, fl_op = DL.open_score(op, dcfg, k_op)
    np.save(results_dir / "open_score_test.npy", sc_te); np.save(results_dir / "open_score_open.npy", sc_op)
    y_open = np.r_[np.zeros(len(sc_te), int), np.ones(len(sc_op), int)]
    rep_open = DL.multiclass_report(y_open, np.r_[fl_te, fl_op].astype(int), [0, 1])
    rep_open["auroc"] = float(roc_auc_score(y_open, np.r_[sc_te, sc_op]))
    rep_open["open_flagged"] = float(fl_op.mean()); rep_open["closed_flagged"] = float(fl_te.mean())

    def show(title, rep, names):
        print(f"\n  {title}")
        print(f"    {'class':<10}{'precision':>10}{'recall':>9}{'f1':>8}{'support':>9}")
        for c, d in rep["per_class"].items():
            print(f"    {names.get(c, c):<10}{d['precision']:>10.3f}{d['recall']:>9.3f}{d['f1']:>8.3f}{d['support']:>9d}")
        for avg in ("macro", "micro"):
            d = rep[avg]; print(f"    {avg:<10}{d['precision']:>10.3f}{d['recall']:>9.3f}{d['f1']:>8.3f}")
    show("COUNT (NOC, closed test)", rep_count, {str(j): f"NOC{j}" for j in range(1, 6)})
    print("\n  ID (donor-level, closed test)      precision  recall      f1")
    for part, r in rep_id.items():
        for avg in ("micro", "macro"):
            d = r[avg]; print(f"    {part:<9}{avg:<6}{'':<15}{d['precision']:>9.3f}{d['recall']:>8.3f}{d['f1']:>8.3f}")
    show(f"OPEN SET (closed test {len(sc_te)} vs open {len(sc_op)}; AUROC {rep_open['auroc']:.3f})", rep_open,
         {"0": "closed", "1": "open"})
    print(f"    flagged: open {rep_open['open_flagged']:.3f} | closed test {rep_open['closed_flagged']:.3f}")

    np.save(results_dir / "y_test_pred.npy", y_te_pred)
    np.save(results_dir / "y_test_true.npy", y_te_true)

    def _pn(lst):
        return _pn_(lst)

    out_dict = {
        "model": "set_transformer", "config": cfg,
        "best_selection_score": round(best_sel, 4), "best_epoch": best_epoch,
        "decode": dcfg,
        "n_test_mixtures": int(mix_.sum()),
        "macro_id_mix": round(macro(em_post), 4), "macro_count_mix": round(macro(noc_per), 4),
        "em_at_pred_k": round(float(em_post[0]), 4), "per_noc_at_pred_k": _pn(em_post),
        "count_acc": round(float(noc_per[0]), 4), "per_noc_count": _pn(noc_per),
        "count_acc_gate": round(float(gate_per[0]), 4), "per_noc_count_gate": _pn(gate_per),
        "oracle_em": round(float(oracle[0]), 4), "per_noc_oracle": _pn(oracle),
        "count_report": rep_count, "id_report": rep_id, "open_report": rep_open,
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
