"""
Clean NOC-only Set Transformer training on all PROVEDIt Filtered data.

Run after prepare_noc_filtered.py generates data/noc_filtered/.

Pure NOC classification: loss = CrossEntropyLoss(weight=noc_weights)(logits_noc, noc)
No contributor ID loss, no reject loss — eliminates gradient conflict.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent
# DATA_DIR and RESULTS_DIR are set dynamically in main() based on --data-dir
DATA_DIR: Path = ROOT / "data" / "noc_filtered"   # default, overridden at runtime
RESULTS_DIR: Path = ROOT / "results" / "noc_set_transformer"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Default hyperparameters ────────────────────────────────────────────────

DEFAULT_CFG = {
    "d_locus": 16,
    "d_model": 64,
    "n_heads": 4,
    "n_isab": 2,
    "m_inducing": 32,
    "n_classes": 1,    # unused (cls head not trained), kept minimal
    "dropout": 0.2,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "epochs": 150,
    "patience": 25,
    "out_subdir": "noc_set_transformer",
}


# ── Dataset loading ────────────────────────────────────────────────────────

def load_split(split: str) -> TensorDataset:
    tokens = torch.from_numpy(np.load(DATA_DIR / f"tokens_{split}.npy"))
    mask   = torch.from_numpy(np.load(DATA_DIR / f"mask_{split}.npy"))
    noc    = torch.from_numpy(np.load(DATA_DIR / f"noc_{split}.npy")).long()
    return TensorDataset(tokens, mask, noc)


# ── Evaluation ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred = [], []
    for tokens, mask, noc in loader:
        out = model(tokens.to(DEVICE), mask.to(DEVICE))
        pred = out["logits_noc"].argmax(dim=1).cpu().numpy()
        all_true.append(noc.numpy())
        all_pred.append(pred)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return mf1, y_true, y_pred


# ── Training ───────────────────────────────────────────────────────────────

def train(cfg: dict) -> None:
    from set_transformer import SetTransformerMixture

    with open(DATA_DIR / "vocab.json") as f:
        vocab = json.load(f)

    n_loci = vocab["n_loci"]   # 28 unified markers
    n_noc  = vocab["n_noc"]    # 5 (0-indexed NOC labels)

    # Class-weighted CE for NOC imbalance
    noc_train = np.load(DATA_DIR / "noc_train.npy")
    counts = np.bincount(noc_train, minlength=n_noc).astype(float)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights[counts > 0].mean()
    print(f"NOC class weights: {[f'{w:.3f}' for w in weights]}")
    noc_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    ce_noc = nn.CrossEntropyLoss(weight=noc_weights)

    # Datasets
    ds_train = load_split("train")
    ds_val   = load_split("val")
    ds_test  = load_split("test")
    train_loader = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,  num_workers=0)
    val_loader   = DataLoader(ds_val,   batch_size=cfg["batch_size"], shuffle=False, num_workers=0)
    test_loader  = DataLoader(ds_test,  batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    # Model
    model = SetTransformerMixture(
        n_loci=n_loci,
        d_locus=cfg["d_locus"],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_isab=cfg["n_isab"],
        m_inducing=cfg["m_inducing"],
        n_classes=cfg["n_classes"],
        n_noc=n_noc,
        dropout=cfg["dropout"],
        cls_decoder="pooled",
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Device : {DEVICE}")
    print(f"Params : {n_params:,}")
    print(f"Train  : {len(ds_train)}  Val: {len(ds_val)}  Test: {len(ds_test)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.25, patience=8, min_lr=1e-5
    )

    best_val_f1 = -1.0
    best_ep = 0
    best_state: dict = {}
    patience_counter = 0
    t0 = time.time()

    print(f"\nTraining up to {cfg['epochs']} epochs (patience={cfg['patience']}) …")

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for tokens, mask, noc in train_loader:
            tokens, mask, noc = tokens.to(DEVICE), mask.to(DEVICE), noc.to(DEVICE)
            out = model(tokens, mask)
            loss = ce_noc(out["logits_noc"], noc)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(noc)

        epoch_loss /= len(ds_train)
        val_f1, _, _ = evaluate(model, val_loader)
        scheduler.step(val_f1)

        if ep % 5 == 0 or ep == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Ep {ep:3d} | loss={epoch_loss:.4f} | val_noc_f1={val_f1:.4f} | lr={lr_now:.1e}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_ep = ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"  Early stop at ep {ep}  (best ep {best_ep}, val F1={best_val_f1:.4f})")
                break

    elapsed = time.time() - t0
    print(f"Training done in {elapsed:.1f}s\n")

    # Restore best checkpoint
    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})

    # Test evaluation
    test_f1, y_true, y_pred = evaluate(model, test_loader)
    noc_names = [f"NOC={i+1}" for i in range(n_noc)]
    print("=" * 60)
    print("  NOC SET TRANSFORMER — TEST RESULTS")
    print("=" * 60)
    print(f"  Macro F1: {test_f1:.4f}")
    print()
    print(classification_report(
        y_true, y_pred, target_names=noc_names,
        labels=list(range(n_noc)), zero_division=0
    ))

    # Baseline comparison
    print("  Baselines (NOC Macro F1 on GF29 filtered data):")
    print("    XGB:      0.8189")
    print("    CNN:      0.8594")
    print("    TabPFN:   0.9122")
    print(f"    This:     {test_f1:.4f}")

    # Save
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, RESULTS_DIR / "best_model.pt")
    results = {
        "test_noc_macro_f1": test_f1,
        "best_val_f1": best_val_f1,
        "best_epoch": best_ep,
        "n_params": n_params,
        "cfg": cfg,
        "vocab_n_loci": n_loci,
    }
    with open(RESULTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {RESULTS_DIR}")


def main() -> None:
    global DATA_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to data dir containing tokens/mask/noc npy + vocab.json "
                             "(default: data/noc_filtered). Use data/noc_gf29 for GF29-only run.")
    args = parser.parse_args()

    if args.data_dir:
        DATA_DIR   = Path(args.data_dir)
        subdir     = DATA_DIR.name          # e.g. "noc_gf29"
        RESULTS_DIR = ROOT / "results" / subdir

    cfg = dict(DEFAULT_CFG)
    if args.config:
        with open(args.config) as f:
            cfg.update(json.load(f))
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.patience is not None:
        cfg["patience"] = args.patience

    train(cfg)


if __name__ == "__main__":
    main()
