"""
Ensemble of 5 Set Transformer models (different seeds) for NOC prediction.

Usage:
  python3 train_ensemble.py --data-dir data/marker_filtered
  python3 train_ensemble.py --data-dir data/marker_gf29

Trains N_SEEDS models sequentially, averages softmax probabilities → final prediction.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parent

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_SEEDS = 5

DEFAULT_CFG = {
    "d_model":      64,
    "n_heads":      4,
    "n_isab":       2,
    "m_inducing":   32,
    "pma_seeds":    1,
    "dropout":      0.2,
    "lr":           3e-4,
    "weight_decay": 1e-4,
    "batch_size":   128,
    "epochs":       150,
    "patience":     25,
}


# ── Reproducibility ────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS doesn't expose manual_seed separately; torch.manual_seed covers it


# ── Data ───────────────────────────────────────────────────────────────────

def load_split(data_dir: Path, split: str) -> TensorDataset:
    markers = torch.from_numpy(np.load(data_dir / f"marker_{split}.npy"))
    noc     = torch.from_numpy(np.load(data_dir / f"noc_{split}.npy")).long()
    return TensorDataset(markers, noc)


# ── Train one model ────────────────────────────────────────────────────────

def train_one(
    seed: int,
    vocab: dict,
    cfg: dict,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    ce: nn.CrossEntropyLoss,
) -> dict:
    """Train one model, return best state_dict."""
    from noc_transformer import NOCTransformerISAB

    set_seed(seed)
    model = NOCTransformerISAB(
        n_markers=vocab["n_loci"], d_feat=vocab["n_feat"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
        n_noc=vocab["n_noc"], dropout=cfg["dropout"],
        pma_seeds=cfg["pma_seeds"],
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.25, patience=8, min_lr=1e-5
    )

    best_val, best_ep, best_state, patience_left = -1.0, 0, {}, cfg["patience"]

    for ep in range(1, cfg["epochs"] + 1):
        model.train()
        for x, noc in train_loader:
            x, noc = x.to(DEVICE), noc.to(DEVICE)
            loss = ce(model(x), noc)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for x, noc in val_loader:
                preds.append(model(x.to(DEVICE)).argmax(1).cpu().numpy())
                trues.append(noc.numpy())
        val_f1 = float(f1_score(
            np.concatenate(trues), np.concatenate(preds),
            average="macro", zero_division=0
        ))
        sched.step(val_f1)

        if val_f1 > best_val:
            best_val, best_ep = val_f1, ep
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_left = cfg["patience"]
        else:
            patience_left -= 1
            if patience_left == 0:
                break

    print(f"    seed={seed}  best_ep={best_ep:3d}  val_f1={best_val:.4f}")
    return best_state


# ── Get test logits from one model ─────────────────────────────────────────

@torch.no_grad()
def get_logits(state_dict: dict, vocab: dict, cfg: dict, loader: DataLoader) -> np.ndarray:
    from noc_transformer import NOCTransformerISAB
    model = NOCTransformerISAB(
        n_markers=vocab["n_loci"], d_feat=vocab["n_feat"],
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
        n_noc=vocab["n_noc"], dropout=cfg["dropout"],
        pma_seeds=cfg["pma_seeds"],
    ).to(DEVICE)
    model.load_state_dict({k: v.to(DEVICE) for k, v in state_dict.items()})
    model.eval()
    all_logits = []
    for x, _ in loader:
        all_logits.append(model(x.to(DEVICE)).cpu().numpy())
    return np.concatenate(all_logits, axis=0)   # (N, n_noc)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/marker_filtered")
    parser.add_argument("--n-seeds",  type=int, default=N_SEEDS)
    parser.add_argument("--epochs",   type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    data_dir = ROOT / args.data_dir
    cfg      = dict(DEFAULT_CFG)
    if args.epochs   is not None: cfg["epochs"]   = args.epochs
    if args.patience is not None: cfg["patience"] = args.patience

    with open(data_dir / "vocab.json") as f:
        vocab = json.load(f)
    n_noc = vocab["n_noc"]

    # Class weights
    noc_train = np.load(data_dir / "noc_train.npy")
    counts  = np.bincount(noc_train, minlength=n_noc).astype(float)
    weights = np.where(counts > 0, 1.0 / counts, 0.0)
    weights = weights / weights[counts > 0].mean()
    ce = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    )

    kw = dict(num_workers=0, pin_memory=False)
    ds_train = load_split(data_dir, "train")
    ds_val   = load_split(data_dir, "val")
    ds_test  = load_split(data_dir, "test")
    train_loader = DataLoader(ds_train, batch_size=cfg["batch_size"], shuffle=True,  **kw)
    val_loader   = DataLoader(ds_val,   batch_size=cfg["batch_size"], shuffle=False, **kw)
    test_loader  = DataLoader(ds_test,  batch_size=cfg["batch_size"], shuffle=False, **kw)

    # Ground-truth labels for test
    y_true = np.load(data_dir / "noc_test.npy")

    n_params = sum(
        p.numel() for p in __import__("noc_transformer").NOCTransformerISAB(
            n_markers=vocab["n_loci"], d_feat=vocab["n_feat"],
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
            n_noc=n_noc, dropout=cfg["dropout"],
        ).parameters() if p.requires_grad
    )

    print(f"Device  : {DEVICE}")
    print(f"Params  : {n_params:,} × {args.n_seeds} seeds")
    print(f"Data    : train={len(ds_train)}  val={len(ds_val)}  test={len(ds_test)}")
    print(f"NOC wts : {[f'{w:.3f}' for w in weights]}")
    print(f"\nTraining {args.n_seeds} models …")

    t0 = time.time()
    all_logits   = []
    single_f1s   = []

    for seed in range(args.n_seeds):
        print(f"\n  ── Seed {seed} ──────────────────────────────")
        state = train_one(seed, vocab, cfg, train_loader, val_loader, ce)

        logits = get_logits(state, vocab, cfg, test_loader)
        all_logits.append(logits)

        # Single-model test F1
        y_pred_single = logits.argmax(axis=1)
        f1_single = float(f1_score(y_true, y_pred_single, average="macro", zero_division=0))
        single_f1s.append(f1_single)
        print(f"    test_f1={f1_single:.4f}")

        # Running ensemble
        ensemble_probs = np.mean([np.exp(l) / np.exp(l).sum(axis=1, keepdims=True)
                                   for l in all_logits], axis=0)
        ensemble_pred = ensemble_probs.argmax(axis=1)
        ens_f1 = float(f1_score(y_true, ensemble_pred, average="macro", zero_division=0))
        print(f"    ensemble_f1 ({seed+1} models)={ens_f1:.4f}")

    total_time = time.time() - t0
    print(f"\nTotal training time: {total_time:.0f}s")

    # Final ensemble
    ensemble_probs = np.mean([
        np.exp(l) / np.exp(l).sum(axis=1, keepdims=True)
        for l in all_logits
    ], axis=0)
    y_pred_ens = ensemble_probs.argmax(axis=1)
    final_f1   = float(f1_score(y_true, y_pred_ens, average="macro", zero_division=0))

    noc_names = [f"NOC={i+1}" for i in range(n_noc)]
    print("\n" + "=" * 60)
    print("  ENSEMBLE (5×) — FINAL TEST RESULTS")
    print("=" * 60)
    print(f"  Single-model F1s : {[f'{f:.4f}' for f in single_f1s]}")
    print(f"  Single-model mean: {np.mean(single_f1s):.4f} ± {np.std(single_f1s):.4f}")
    print(f"  Ensemble Macro F1: {final_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred_ens, target_names=noc_names,
                                  labels=list(range(n_noc)), zero_division=0))
    print("  Baselines:")
    print("    TabPFN  : 0.9122  ← target")
    print("    CNN     : 0.8594")
    print("    XGB     : 0.8189")
    print("    V3 Transformer: 0.7848")
    print(f"    This (ensemble): {final_f1:.4f}")

    # Save
    out_dir = ROOT / "results" / f"ensemble_{data_dir.name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "ensemble_probs.npy", ensemble_probs)
    json.dump({
        "ensemble_macro_f1": final_f1,
        "single_f1s": single_f1s,
        "single_mean": float(np.mean(single_f1s)),
        "single_std":  float(np.std(single_f1s)),
        "n_seeds": args.n_seeds,
        "device": str(DEVICE),
        "cfg": cfg,
    }, open(out_dir / "results.json", "w"), indent=2)
    print(f"\nSaved → {out_dir}")


if __name__ == "__main__":
    main()
