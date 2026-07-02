"""
Open-set scoring comparison — Set Transformer.

Loads best ST checkpoint and computes 5 open-set scoring functions on
test (closed, label=0) and open-set (label=1) data without retraining.

Methods:
  1. reject  — sigmoid(reject_head)         (trained explicitly for this task)
  2. msp     — 1 - max sigmoid(cls_logits)  (Hendrycks & Gimpel 2017, multi-label adapt)
  3. energy  — -logsumexp(cls_logits)       (Liu et al. 2020)
  4. maha    — min Mahalanobis dist to per-donor centroid in z_mix
              (Lee et al. 2018, shared covariance)
  5. openmax — max Weibull CDF of distance to class mean activation
              (Bendale & Boult 2016, simplified for multi-label)

Output: results/open_set_scoring.json — AUROC × 5 methods.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import weibull_min
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT  = ROOT / "results" / "set_transformer" / "best_model.pt"
OUT   = ROOT / "results" / "open_set_scoring.json"


class ClosedSet(Dataset):
    def __init__(self, split: str):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"tokens_{split}.npy"))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / f"mask_{split}.npy"))
        self.y      = torch.from_numpy(np.load(DATA_DIR / f"y_{split}_set.npy"))
    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i], self.mask[i], self.y[i]


class OpenSet(Dataset):
    def __init__(self):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / "tokens_open.npy"))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / "mask_open.npy"))
    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i], self.mask[i]


# ── Feature extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract(model, loader, has_y: bool):
    """Run model.encode + cls/reject heads. Returns z, logits, reject, y (or None)."""
    Zs, Ls, Rs, Ys = [], [], [], []
    for batch in loader:
        if has_y:
            tokens, mask, y = batch
            Ys.append(y.numpy())
        else:
            tokens, mask = batch
        z = model.encode(tokens.to(DEVICE), mask.to(DEVICE))
        Zs.append(z.cpu().numpy())
        Ls.append(model.cls_head(z).cpu().numpy())
        Rs.append(model.reject_head(z).cpu().numpy())
    Z = np.concatenate(Zs); L = np.concatenate(Ls); R = np.concatenate(Rs)
    Y = np.concatenate(Ys) if has_y else None
    return Z, L, R, Y


# ── Scoring functions (higher = more likely open/unknown) ─────────────────

def score_reject(R):
    """Sigmoid of reject logit."""
    return 1.0 / (1.0 + np.exp(-R.ravel()))


def score_msp(L):
    """1 - max sigmoid(cls_logits). Multi-label adaptation: low max conf → likely unknown."""
    p = 1.0 / (1.0 + np.exp(-L))
    return 1.0 - p.max(axis=1)


def score_energy(L, T: float = 1.0):
    """Energy = -T * logsumexp(L / T). Higher = lower total activation = more unknown."""
    m = L.max(axis=1, keepdims=True)
    lse = m.squeeze(1) + T * np.log(np.exp((L - m) / T).sum(axis=1))
    return -lse


def fit_mahalanobis(Z_tr, Y_tr, ridge: float = 1e-3):
    """Per-class mean + shared inverse covariance. Y_tr is multi-label (N, C)."""
    n_classes = Y_tr.shape[1]
    d = Z_tr.shape[1]
    centroids = np.zeros((n_classes, d), dtype=np.float64)
    global_mean = Z_tr.mean(axis=0)
    for c in range(n_classes):
        mask = Y_tr[:, c] > 0.5
        centroids[c] = Z_tr[mask].mean(axis=0) if mask.sum() > 0 else global_mean
    cov = np.cov(Z_tr.T) + ridge * np.eye(d)
    return centroids, np.linalg.inv(cov)


def score_mahalanobis(Z, centroids, inv_cov):
    """Min Mahalanobis distance over all 45 donor centroids."""
    out = np.empty(Z.shape[0], dtype=np.float64)
    for i in range(Z.shape[0]):
        diffs = Z[i] - centroids                         # (C, d)
        dists = np.einsum('ij,jk,ik->i', diffs, inv_cov, diffs)
        out[i] = dists.min()
    return out


def fit_openmax(Z_tr, Y_tr, tail_size: int = 20):
    """Per-class mean activation + Weibull fit on top-`tail_size` distances."""
    n_classes = Y_tr.shape[1]
    fits = []
    for c in range(n_classes):
        mask = Y_tr[:, c] > 0.5
        if mask.sum() < tail_size:
            fits.append(None); continue
        zs = Z_tr[mask]
        mu = zs.mean(axis=0)
        dists = np.linalg.norm(zs - mu, axis=1)
        tail = np.sort(dists)[-tail_size:]
        try:
            shape, loc, scale = weibull_min.fit(tail, floc=0)
            fits.append((mu, shape, scale))
        except Exception:
            fits.append(None)
    return fits


def score_openmax(Z, fits):
    """
    For each sample, find the closest donor centroid; return Weibull CDF of that
    distance against the closest class's tail. Closed samples → small dist → low CDF;
    open samples → even closest dist large → high CDF.
    """
    out = np.zeros(Z.shape[0], dtype=np.float64)
    valid = [(c, f) for c, f in enumerate(fits) if f is not None]
    for i in range(Z.shape[0]):
        best_dist = np.inf
        best_params = None
        for _, (mu, shape, scale) in valid:
            d = float(np.linalg.norm(Z[i] - mu))
            if d < best_dist:
                best_dist = d
                best_params = (shape, scale)
        if best_params is None:
            out[i] = 0.0
        else:
            shape, scale = best_params
            out[i] = float(weibull_min.cdf(best_dist, shape, loc=0, scale=scale))
    return out


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="all",
                        choices=["reject", "msp", "energy", "maha", "openmax", "all"])
    parser.add_argument("--ckpt", default=str(CKPT))
    parser.add_argument("--config", default=str(ROOT / "configs" / "set_transformer.json"))
    parser.add_argument("--tail", type=int, default=20, help="Openmax Weibull tail size")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    print(f"Device: {DEVICE}")
    model = SetTransformerMixture(
        n_loci=cfg["n_loci"], d_locus=cfg["d_locus"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
        n_classes=cfg["n_classes"], n_noc=cfg["n_noc"], dropout=cfg["dropout"],
    ).to(DEVICE)
    state = torch.load(args.ckpt, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    # --- Extract features ---
    bsz = 256
    train_loader = DataLoader(ClosedSet("train"), batch_size=bsz, shuffle=False)
    test_loader  = DataLoader(ClosedSet("test"),  batch_size=bsz, shuffle=False)
    open_loader  = DataLoader(OpenSet(),          batch_size=bsz, shuffle=False)

    t0 = time.time()
    Z_tr, _, _, Y_tr = extract(model, train_loader, has_y=True)
    Z_te, L_te, R_te, _ = extract(model, test_loader,  has_y=True)
    Z_op, L_op, R_op, _ = extract(model, open_loader,  has_y=False)
    print(f"Features extracted in {time.time()-t0:.1f}s")
    print(f"  train: Z {Z_tr.shape}, Y {Y_tr.shape}")
    print(f"  test:  Z {Z_te.shape}")
    print(f"  open:  Z {Z_op.shape}")

    methods = ["reject", "msp", "energy", "maha", "openmax"] if args.method == "all" else [args.method]
    results = {}

    labels = np.concatenate([np.zeros(len(Z_te)), np.ones(len(Z_op))])

    if "reject" in methods:
        s = np.concatenate([score_reject(R_te), score_reject(R_op)])
        results["reject"] = float(roc_auc_score(labels, s))

    if "msp" in methods:
        s = np.concatenate([score_msp(L_te), score_msp(L_op)])
        results["msp"] = float(roc_auc_score(labels, s))

    if "energy" in methods:
        s = np.concatenate([score_energy(L_te), score_energy(L_op)])
        results["energy"] = float(roc_auc_score(labels, s))

    if "maha" in methods:
        print("Fitting Mahalanobis (per-donor centroids + shared cov) ...")
        cents, inv_cov = fit_mahalanobis(Z_tr, Y_tr)
        s = np.concatenate([score_mahalanobis(Z_te, cents, inv_cov),
                            score_mahalanobis(Z_op, cents, inv_cov)])
        results["maha"] = float(roc_auc_score(labels, s))

    if "openmax" in methods:
        print(f"Fitting Openmax (Weibull tail={args.tail}) ...")
        fits = fit_openmax(Z_tr, Y_tr, tail_size=args.tail)
        n_valid = sum(1 for f in fits if f is not None)
        print(f"  valid Weibull fits: {n_valid}/{len(fits)}")
        s = np.concatenate([score_openmax(Z_te, fits), score_openmax(Z_op, fits)])
        results["openmax"] = float(roc_auc_score(labels, s))

    # --- Report ---
    print(f"\n{'='*48}")
    print(f"  Open-set AUROC ({len(Z_te)} closed vs {len(Z_op)} open)")
    print("="*48)
    for m in ["reject", "msp", "energy", "maha", "openmax"]:
        if m in results:
            print(f"  {m:<10} AUROC = {results[m]:.4f}")
    print("="*48)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump({
            "n_closed_test": int(len(Z_te)),
            "n_open":        int(len(Z_op)),
            "checkpoint":    str(args.ckpt),
            "auroc":         results,
        }, f, indent=2)
    print(f"\nSaved -> {OUT}")


if __name__ == "__main__":
    main()
