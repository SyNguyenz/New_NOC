"""
Training script for NOC ablation experiments (E1+E2+E3 and beyond).

Supports:
  --model isab      NOCTransformerISAB  (Set Transformer + PE + marker-level)  [E1+E2+E3]
  --model std       NOCTransformerSTD   (Standard Transformer + PE)             [bonus]
  --data-dir PATH   directory with marker_{split}.npy + noc_{split}.npy + vocab.json

Usage:
  python3 train_noc_v2.py --model isab --data-dir data/marker_gf29
  python3 train_noc_v2.py --model std  --data-dir data/marker_gf29
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

ROOT    = Path(__file__).resolve().parent
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_CFG = {
    "d_model":    64,
    "n_heads":    4,
    "n_isab":     2,       # for ISAB model
    "n_layers":   3,       # for STD model
    "m_inducing": 32,
    "pma_seeds":  1,
    "dropout":    0.2,
    "pool":       "mean",  # for STD model
    "lr":         3e-4,
    "weight_decay": 1e-4,
    "batch_size": 128,
    "epochs":     150,
    "patience":   25,
}


def load_split(data_dir: Path, split: str) -> TensorDataset:
    markers = torch.from_numpy(np.load(data_dir / f"marker_{split}.npy"))  # (N, 24, 8)
    noc     = torch.from_numpy(np.load(data_dir / f"noc_{split}.npy")).long()
    return TensorDataset(markers, noc)


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_true, all_pred = [], []
    for x, noc in loader:
        logits = model(x.to(DEVICE))
        all_true.append(noc.numpy())
        all_pred.append(logits.argmax(1).cpu().numpy())
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0)), y_true, y_pred


def build_model(arch: str, vocab: dict, cfg: dict) -> nn.Module:
    from noc_transformer import NOCTransformerISAB, NOCTransformerSTD
    n_markers = vocab["n_loci"]
    d_feat    = vocab["n_feat"]
    n_noc     = vocab["n_noc"]

    if arch == "isab":
        return NOCTransformerISAB(
            n_markers=n_markers, d_feat=d_feat,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
            n_noc=n_noc, dropout=cfg["dropout"], pma_seeds=cfg["pma_seeds"],
        )
    elif arch == "std":
        return NOCTransformerSTD(
            n_markers=n_markers, d_feat=d_feat,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"], n_noc=n_noc,
            dropout=cfg["dropout"], pool=cfg["pool"],
        )
    else:
        raise ValueError(f"Unknown arch: {arch}")


def train(arch: str, data_dir: Path, cfg: dict) -> None:
    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)
    n_noc = vocab["n_noc"]

    # Class-weighted CE
    noc_train = np.load(data_dir / "noc_train.npy")
    counts  = np.bincount(noc_train, minlength=n_noc).astype(float)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights[counts > 0].mean()
    print(f"NOC class weights: {[f'{w:.3f}' for w in weights]}")
    ce = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))

    ds_train = load_split(data_dir, "train")
    ds_val   = load_split(data_dir, "val")
    ds_test  = load_split(data_dir, "test")
    kw = dict(num_workers=0, pin_memory=False)
    train_loader = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(ds_val,   batch_size=cfg["batch_size"], shuffle=False, **kw)
    test_loader  = DataLoader(ds_test,  batch_size=cfg["batch_size"], shuffle=False, **kw)

    model = build_model(arch, vocab, cfg).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model : {arch.upper()}  ({n_params:,} params)")
    print(f"Device: {DEVICE}")
    print(f"Data  : train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.25, patience=8, min_lr=1e-5
    )

    best_val, best_ep, best_state = -1.0, 0, {}
    patience_left = cfg["patience"]
    t0 = time.time()

    print(f"\nTraining up to {cfg['epochs']} epochs (patience={cfg['patience']}) …")

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        ep_loss = 0.0
        for x, noc in train_loader:
            x, noc = x.to(DEVICE), noc.to(DEVICE)
            loss = ce(model(x), noc)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(noc)

        ep_loss /= len(ds_train)
        val_f1, _, _ = evaluate(model, val_loader)
        scheduler.step(val_f1)

        if ep % 5 == 0 or ep == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Ep {ep:3d} | loss={ep_loss:.4f} | val_f1={val_f1:.4f} | lr={lr_now:.1e}")

        if val_f1 > best_val:
            best_val, best_ep = val_f1, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f"  Early stop ep {ep}  (best ep {best_ep}, val F1={best_val:.4f})")
                break

    print(f"Training done in {time.time()-t0:.1f}s\n")

    model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    test_f1, y_true, y_pred = evaluate(model, test_loader)
    noc_names = [f"NOC={i+1}" for i in range(n_noc)]

    print("=" * 60)
    print(f"  [{arch.upper()}] NOC TRANSFORMER — TEST RESULTS")
    print("=" * 60)
    print(f"  Macro F1 : {test_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=noc_names,
                                 labels=list(range(n_noc)), zero_division=0))

    print("  Baselines (GF29, NOC Macro F1):")
    print("    E0  Set Transformer (peak-level, 3 heads): 0.4318")
    print("    V3  Forensic Transformer (marker-level):   0.7848")
    print("    CNN:                                       0.8594")
    print("    XGB:                                       0.8189")
    print("    TabPFN:                                    0.9122")
    print(f"    This [{arch.upper()}]:                            {test_f1:.4f}")

    # Save
    out_dir = ROOT / "results" / f"noc_v2_{arch}"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "best_model.pt")
    json.dump({
        "arch": arch, "test_noc_macro_f1": test_f1,
        "best_val_f1": best_val, "best_epoch": best_ep,
        "n_params": n_params, "cfg": cfg,
    }, open(out_dir / "results.json", "w"), indent=2)
    print(f"\nSaved → {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    default="isab", choices=["isab", "std"])
    parser.add_argument("--data-dir", default="data/marker_gf29")
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--d-model",  type=int, default=None)
    args = parser.parse_args()

    cfg = dict(DEFAULT_CFG)
    if args.epochs   is not None: cfg["epochs"]   = args.epochs
    if args.patience is not None: cfg["patience"] = args.patience
    if args.d_model  is not None: cfg["d_model"]  = args.d_model

    data_dir = ROOT / args.data_dir
    train(args.model, data_dir, cfg)


if __name__ == "__main__":
    main()
