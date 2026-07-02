"""
train_synth.py  —  Phase 3.1 + 3.2

Train Set Transformer on SYNTHETIC mixtures + REAL single-source references,
then evaluate ZERO-SHOT on the REAL no-leak test (novel combos).

Training data:
  - Synthetic multi-person mixtures (data/tokens_synth_train.npy ...)  NOC 2-5
  - Real single-source train (NOC=1 from real train split) — reference DB
Validation (early stopping):
  - Synthetic val (data/*_synth_val.npy) + real single-source val
Test (zero-shot, the headline experiment):
  - REAL test split (data/tokens_test.npy ...) — contains NOVEL combos
Open-set (reject head):
  - Real open set (data/tokens_open.npy)

Compares against:
  1. Real-leaky (old)   — results stored separately
  2. Real-only no-leak  — results/set_transformer/metrics.json
  3. THIS (synth->real) — results/set_transformer_synth/metrics.json

Usage:
  python train_synth.py
  python train_synth.py --epochs 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler, ConcatDataset
from sklearn.metrics import f1_score, roc_auc_score

import sys

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture
from train_set_transformer import (
    compute_pos_weight, evaluate_closed, full_report, OpenSetDataset,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Datasets ───────────────────────────────────────────────────────────────

class ArrayDataset(Dataset):
    """Generic dataset from explicit arrays."""
    def __init__(self, tokens, mask, y, noc):
        self.tokens = torch.from_numpy(tokens)
        self.mask   = torch.from_numpy(mask)
        self.y      = torch.from_numpy(y)
        self.noc    = torch.from_numpy(noc.astype(np.int64))

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        return self.tokens[i], self.mask[i], self.y[i], self.noc[i]


def load_arrays(prefix: str):
    """Load tokens/mask/y/noc for a given split prefix."""
    tokens = np.load(DATA_DIR / f"tokens_{prefix}.npy")
    mask   = np.load(DATA_DIR / f"mask_{prefix}.npy")
    # y file naming differs: real uses y_{split}_set.npy, synth uses y_set_{split}.npy
    y_real  = DATA_DIR / f"y_{prefix}_set.npy"
    y_synth = DATA_DIR / f"y_set_{prefix}.npy"
    if y_real.exists():
        y = np.load(y_real)
    elif y_synth.exists():
        y = np.load(y_synth)
    else:
        raise FileNotFoundError(f"No y file for {prefix}")
    noc = np.load(DATA_DIR / f"noc_{prefix}.npy")
    return tokens, mask, y, noc


def filter_single_source(tokens, mask, y, noc):
    """Keep only NOC==1 samples (single-source references)."""
    m = noc == 1
    return tokens[m], mask[m], y[m], noc[m]


def build_training_data():
    """
    Train = synthetic multi-person + real single-source train.
    Val   = synthetic val + real single-source val.
    """
    # Synthetic multi-person
    s_tok, s_msk, s_y, s_noc = load_arrays("synth_train")
    sv_tok, sv_msk, sv_y, sv_noc = load_arrays("synth_val")

    # Real single-source (reference DB)
    rt_tok, rt_msk, rt_y, rt_noc = load_arrays("train")
    rt_tok, rt_msk, rt_y, rt_noc = filter_single_source(rt_tok, rt_msk, rt_y, rt_noc)
    rv_tok, rv_msk, rv_y, rv_noc = load_arrays("val")
    rv_tok, rv_msk, rv_y, rv_noc = filter_single_source(rv_tok, rv_msk, rv_y, rv_noc)

    print(f"  Synthetic train mixtures : {len(s_tok)}")
    print(f"  Real single-source train : {len(rt_tok)}")
    print(f"  Synthetic val mixtures   : {len(sv_tok)}")
    print(f"  Real single-source val   : {len(rv_tok)}")

    train_ds = ConcatDataset([
        ArrayDataset(s_tok, s_msk, s_y, s_noc),
        ArrayDataset(rt_tok, rt_msk, rt_y, rt_noc),
    ])
    val_ds = ConcatDataset([
        ArrayDataset(sv_tok, sv_msk, sv_y, sv_noc),
        ArrayDataset(rv_tok, rv_msk, rv_y, rv_noc),
    ])

    # Combined training labels for pos_weight
    y_train_combined = np.concatenate([s_y, rt_y], axis=0)
    return train_ds, val_ds, y_train_combined


# ── Training ───────────────────────────────────────────────────────────────

def train(cfg: dict):
    results_dir = ROOT / "results" / "set_transformer_synth"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building Phase-3 training data (synth + real single-source) ===")
    train_ds, val_ds, y_train_combined = build_training_data()

    # Real test (zero-shot eval target — novel combos)
    rte_tok, rte_msk, rte_y, rte_noc = load_arrays("test")
    test_ds = ArrayDataset(rte_tok, rte_msk, rte_y, rte_noc)

    open_ds = OpenSetDataset()

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    open_bs = max(1, int(cfg["batch_size"] * cfg.get("open_ratio", 0.25)))
    def make_open_iter():
        return iter(DataLoader(
            open_ds, batch_size=open_bs,
            sampler=RandomSampler(open_ds, replacement=True, num_samples=len(open_ds) * 10),
            num_workers=0,
        ))
    open_iter = make_open_iter()

    # Model
    model = SetTransformerMixture(
        n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16),
        d_model=cfg.get("d_model", 128), n_heads=cfg.get("n_heads", 4),
        n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
        n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6),
        dropout=cfg.get("dropout", 0.1),
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nDevice : {DEVICE}")
    print(f"Params : {n_params:,}")
    print(f"Train  : {len(train_ds)} (synth+SS)  Val: {len(val_ds)}  Test(real): {len(test_ds)}")

    pos_weight = compute_pos_weight(y_train_combined).to(DEVICE)
    bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_rej = nn.BCEWithLogitsLoss()
    ce_noc  = nn.CrossEntropyLoss()

    alpha = cfg.get("alpha_reject", 0.5)
    beta  = cfg.get("beta_noc", 0.3)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get("lr", 3e-4),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    best_f1, best_epoch, patience_count = 0.0, 0, 0
    patience = cfg.get("patience", 15)
    epochs   = cfg.get("epochs", 60)
    history  = []

    print(f"\nTraining up to {epochs} epochs (patience={patience}) ...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for tokens, mask, y, noc in train_loader:
            tokens, mask = tokens.to(DEVICE), mask.to(DEVICE)
            y, noc = y.to(DEVICE), noc.to(DEVICE)

            out = model(tokens, mask)
            loss_cls = bce_cls(out["logits_cls"], y)
            loss_noc = ce_noc(out["logits_noc"], noc)

            rej_closed = out["logit_reject"]
            rej_label  = torch.zeros(len(tokens), 1, device=DEVICE)

            try:
                open_batch = next(open_iter)
            except StopIteration:
                open_iter = make_open_iter()
                open_batch = next(open_iter)
            o_tok, o_mask = open_batch[0].to(DEVICE), open_batch[1].to(DEVICE)
            rej_open = model(o_tok, o_mask)["logit_reject"]
            rej_label_open = torch.ones(len(o_tok), 1, device=DEVICE)

            all_rej     = torch.cat([rej_closed, rej_open], dim=0)
            all_rej_lbl = torch.cat([rej_label, rej_label_open], dim=0)
            loss_rej    = bce_rej(all_rej, all_rej_lbl)

            loss = loss_cls + alpha * loss_rej + beta * loss_noc

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(tokens)

        epoch_loss /= len(train_ds)
        val_f1, _, _, _ = evaluate_closed(model, val_loader)
        scheduler.step(val_f1)
        history.append({"epoch": epoch, "loss": round(epoch_loss, 4),
                        "val_macro_f1": round(val_f1, 4)})

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | loss={epoch_loss:.4f} | val_f1={val_f1:.4f} "
                  f"| lr={optimizer.param_groups[0]['lr']:.1e}")

        if val_f1 > best_f1:
            best_f1, best_epoch, patience_count = val_f1, epoch, 0
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stop at ep {epoch} (best ep {best_epoch}, val F1={best_f1:.4f})")
                break

    print(f"Training done in {time.time()-t0:.1f}s")

    # ── Threshold search on val ────────────────────────────────────────────
    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for tokens, mask, y, _ in val_loader:
            out = model(tokens.to(DEVICE), mask.to(DEVICE))
            all_probs.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
            all_true.append(y.numpy())
    y_va_probs = np.concatenate(all_probs)
    y_va_true  = np.concatenate(all_true)

    best_thresh, best_thresh_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.85, 0.05):
        f1 = float(f1_score(y_va_true, (y_va_probs >= t).astype(int),
                            average="macro", zero_division=0))
        if f1 > best_thresh_f1:
            best_thresh_f1, best_thresh = f1, float(t)
    print(f"\nBest threshold (val): {best_thresh:.2f} (macro F1={best_thresh_f1:.4f})")

    # ── ZERO-SHOT test on REAL ─────────────────────────────────────────────
    test_f1, y_te_true, y_te_pred, noc_te = evaluate_closed(model, test_loader, best_thresh)
    te_metrics = full_report(y_te_true, y_te_pred, noc_te,
                             "SYNTH->REAL ZERO-SHOT — REAL TEST (novel combos)")

    # ── Reject AUROC ───────────────────────────────────────────────────────
    print("\n-- Reject head AUROC " + "-"*39)
    scores, labels = [], []
    with torch.no_grad():
        for tokens, mask, _, _ in test_loader:
            rej = torch.sigmoid(model(tokens.to(DEVICE), mask.to(DEVICE))["logit_reject"])
            scores.append(rej.cpu().numpy()); labels.append(np.zeros(len(tokens)))
        for o_tok, o_mask in DataLoader(open_ds, batch_size=256, shuffle=False):
            rej = torch.sigmoid(model(o_tok.to(DEVICE), o_mask.to(DEVICE))["logit_reject"])
            scores.append(rej.cpu().numpy()); labels.append(np.ones(len(o_tok)))
    scores = np.concatenate(scores).ravel()
    labels = np.concatenate(labels)
    try:
        auroc = roc_auc_score(labels, scores)
        print(f"  Reject AUROC (closed vs open): {auroc:.4f}")
    except Exception as e:
        auroc = None
        print(f"  AUROC error: {e}")

    # ── Save ───────────────────────────────────────────────────────────────
    np.save(results_dir / "y_test_pred.npy", y_te_pred)
    np.save(results_dir / "y_test_true.npy", y_te_true)
    np.save(results_dir / "noc_test.npy", noc_te)

    out_dict = {
        "model": "set_transformer_synth",
        "training": "synthetic multi-person + real single-source",
        "test_set": "real no-leak test (novel combos) — ZERO-SHOT",
        "config": cfg,
        "best_val_macro_f1": round(best_f1, 4),
        "best_epoch": best_epoch,
        "best_threshold": best_thresh,
        "reject_auroc": float(auroc) if auroc is not None else None,
        "per_noc": te_metrics.get("per_noc", {}),
        "history": history,
        "test": {k: v for k, v in te_metrics.items() if k != "per_noc"},
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(out_dict, f, indent=2)
    print(f"\nSaved -> {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    args = parser.parse_args()

    cfg_path = args.config or str(ROOT / "configs" / "set_transformer.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    for k in ("epochs", "lr", "batch_size"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v
    # Default to 60 epochs for synth (larger dataset)
    if args.epochs is None:
        cfg["epochs"] = 60

    train(cfg)
