"""
Shared training loop cho MLP và CNN.

Usage:
    python train.py --model mlp --config configs/mlp.json
    python train.py --model cnn  --config configs/cnn.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import joblib

ROOT = Path(__file__).resolve().parent
DATA_DIR  = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
RESULT_SUFFIX = os.environ.get("STR_RESULT_SUFFIX", "")
sys.path.insert(0, str(ROOT))

from models.mlp import MLP
from models.cnn import LociCNN
from features.loci_matrix import LociMatrix

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────────────────

def load_splits():
    X_train = np.load(DATA_DIR / "Xflat_train.npy")
    X_val   = np.load(DATA_DIR / "Xflat_val.npy")
    X_test  = np.load(DATA_DIR / "Xflat_test.npy")
    y_train = np.load(DATA_DIR / "y_train_set.npy")
    y_val   = np.load(DATA_DIR / "y_val_set.npy")
    y_test  = np.load(DATA_DIR / "y_test_set.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_pos_weight(y_train: np.ndarray) -> torch.Tensor:
    pos = y_train.sum(axis=0).clip(min=1)
    neg = (1 - y_train).sum(axis=0).clip(min=1)
    return torch.tensor(neg / pos, dtype=torch.float32)


def make_loaders(X_train, X_val, y_train, y_val, batch_size: int):
    def to_loader(X, y, shuffle):
        ds = TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=False)
    return to_loader(X_train, y_train, True), to_loader(X_val, y_val, False)


def val_macro_f1(model, loader, threshold=0.5) -> float:
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            logits = model(X_b.to(DEVICE))
            preds  = (torch.sigmoid(logits) >= threshold).cpu().numpy()
            all_true.append(y_b.numpy())
            all_pred.append(preds)
    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def print_metrics(y_true, y_pred):
    from sklearn.metrics import hamming_loss, f1_score
    print(f"  Macro F1     : {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")
    print(f"  Micro F1     : {f1_score(y_true, y_pred, average='micro', zero_division=0):.4f}")
    print(f"  Hamming Loss : {hamming_loss(y_true, y_pred):.4f}")
    print(f"  Exact Match  : {np.all(y_true == y_pred, axis=1).mean():.4f}")
    pcf = f1_score(y_true, y_pred, average=None, zero_division=0)
    print(f"  Per-class F1 : mean={pcf.mean():.3f}  min={pcf.min():.3f}  "
          f"max={pcf.max():.3f}  zero={int((pcf==0).sum())}/45")


# ── Training loop ──────────────────────────────────────────────────────────

def train(model_name: str, cfg: dict):
    results_dir = ROOT / "results" / f"{model_name}{RESULT_SUFFIX}"
    results_dir.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    # Scale features (data is already log1p; StandardScaler normalises per-feature)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    joblib.dump(scaler, results_dir / "scaler.pkl")

    # CNN: transform flat features → (n_loci, max_alleles)
    lm = None
    if model_name == "cnn":
        meta = json.load(open(DATA_DIR / "meta_set.json"))
        lm = LociMatrix(meta.get("flat_cols", meta.get("feature_cols")))
        lm.save(results_dir / "loci_matrix.json")
        X_train = lm.transform(X_train)
        X_val   = lm.transform(X_val)
        X_test  = lm.transform(X_test)
        print(f"CNN input shape: {X_train.shape}")  # (N, 24, 33)

    train_loader, val_loader = make_loaders(
        X_train, X_val, y_train, y_val, cfg["batch_size"]
    )

    # Build model
    if model_name == "mlp":
        model = MLP(
            in_features=X_train.shape[1],
            hidden_dims=cfg.get("hidden_dims", [512, 256]),
            n_classes=45,
            dropout=cfg.get("dropout", 0.3),
        )
    else:
        n_loci, max_alleles = X_train.shape[1], X_train.shape[2]
        model = LociCNN(
            n_loci=n_loci,
            max_alleles=max_alleles,
            channels=cfg.get("channels", [64, 128]),
            kernel_size=cfg.get("kernel_size", 3),
            n_classes=45,
            dropout=cfg.get("dropout", 0.3),
        )

    model = model.to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model_name}  |  params: {n_params:,}  |  device: {DEVICE}")

    pos_weight = compute_pos_weight(y_train).to(DEVICE)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer  = torch.optim.Adam(model.parameters(), lr=cfg.get("lr", 1e-3),
                                  weight_decay=cfg.get("weight_decay", 1e-4))
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_f1, best_epoch, patience_count = 0.0, 0, 0
    patience = cfg.get("patience", 15)
    epochs   = cfg.get("epochs", 150)
    history  = []

    print(f"\nTraining for up to {epochs} epochs (patience={patience}) ...")
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_b)

        epoch_loss /= len(X_train)
        f1 = val_macro_f1(model, val_loader)
        scheduler.step(f1)
        history.append({"epoch": epoch, "loss": round(epoch_loss, 4), "val_macro_f1": round(f1, 4)})

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d} | loss={epoch_loss:.4f} | val_macro_f1={f1:.4f} | lr={lr_now:.2e}")

        if f1 > best_f1:
            best_f1, best_epoch, patience_count = f1, epoch, 0
            torch.save(model.state_dict(), results_dir / "best_model.pt")
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"  Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    print(f"Training done in {time.time()-t_start:.1f}s | best val macro F1: {best_f1:.4f}")

    # ── Final evaluation on test ──────────────────────────────────────────
    from sklearn.metrics import hamming_loss, precision_score, recall_score

    model.load_state_dict(torch.load(results_dir / "best_model.pt", weights_only=True))
    model.eval()

    # Threshold search on val
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    with torch.no_grad():
        val_probs = torch.sigmoid(model(X_val_t.to(DEVICE))).cpu().numpy()

    best_thresh, best_thresh_f1 = 0.5, 0.0
    for t in np.arange(0.2, 0.85, 0.05):
        f1_t = float(f1_score(y_val, (val_probs >= t).astype(int),
                              average="macro", zero_division=0))
        if f1_t > best_thresh_f1:
            best_thresh_f1, best_thresh = f1_t, float(t)
    print(f"Best threshold (val): {best_thresh:.2f}  (macro F1={best_thresh_f1:.4f})")

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test_t.to(DEVICE))
        test_probs = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        y_test_pred = (test_probs >= best_thresh).astype(np.float32)
    np.save(results_dir / "probs_test.npy", test_probs)

    print("\n-- TEST " + "-" * 30)
    print_metrics(y_test, y_test_pred)

    # Per-NOC exact match
    noc_test = np.load(DATA_DIR / "noc_test.npy").astype(int)
    exact = np.all(y_test == y_test_pred, axis=1)
    per_noc = {}
    for noc in sorted(np.unique(noc_test)):
        m = noc_test == noc
        per_noc[int(noc)] = {"em": round(float(exact[m].mean()), 4), "n": int(m.sum())}
    print("  Per-NOC EM:", {k: f"{v['em']:.3f}(n={v['n']})" for k, v in per_noc.items()})

    # Save
    np.save(results_dir / "y_test_pred.npy", y_test_pred)
    np.save(results_dir / "y_test_true.npy", y_test)

    jac = (
        (y_test.astype(bool) & y_test_pred.astype(bool)).sum(1) /
        (y_test.astype(bool) | y_test_pred.astype(bool)).sum(1).clip(min=1)
    ).mean()

    results = {
        "model": model_name,
        "config": cfg,
        "best_val_macro_f1": round(best_f1, 4),
        "best_epoch": best_epoch,
        "best_threshold": round(best_thresh, 2),
        "history": history,
        "test": {
            "macro_f1":  float(f1_score(y_test, y_test_pred, average="macro",  zero_division=0)),
            "micro_f1":  float(f1_score(y_test, y_test_pred, average="micro",  zero_division=0)),
            "hamming":   float(hamming_loss(y_test, y_test_pred)),
            "exact_match": float(exact.mean()),
            "precision": float(precision_score(y_test, y_test_pred, average="macro", zero_division=0)),
            "recall":    float(recall_score(y_test, y_test_pred,    average="macro", zero_division=0)),
            "jaccard":   float(jac),
        },
        "per_noc": per_noc,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved to {results_dir}")


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["mlp", "cnn"])
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    cfg_path = args.config or str(ROOT / "configs" / f"{args.model}.json")
    with open(cfg_path) as f:
        cfg = json.load(f)

    train(args.model, cfg)
