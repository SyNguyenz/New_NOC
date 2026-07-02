"""
Ablation study — Set Transformer components.

Variants (--variant):
  full       ISAB+PMA encoder, all 3 heads  [reference]
  deep_sets  Deep Sets (mean pool), all 3 heads
  mlp_enc    Flat 590-dim MLP encoder, cls+noc only (no reject)
  no_noc     Set Transformer, beta_noc=0
  no_reject  Set Transformer, alpha_reject=0 (skip open-set training)
  all        Run all 5 variants sequentially then print summary

Usage:
  python ablation.py --variant full
  python ablation.py --variant all
  python ablation.py --variant all --epochs 60
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from sklearn.metrics import f1_score, roc_auc_score

import sys

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VARIANTS = ["full", "deep_sets", "mlp_enc", "no_noc", "no_reject"]

# Architecture sweep (vary one hyperparameter at a time, others at default)
SWEEP_CONFIGS = {
    "m_inducing": [
        {"m_inducing": 8,  "label": "sweep_m8"},
        {"m_inducing": 16, "label": "sweep_m16"},
        {"m_inducing": 32, "label": "sweep_m32"},
        {"m_inducing": 64, "label": "sweep_m64"},
    ],
    "n_isab": [
        {"n_isab": 1, "label": "sweep_isab1"},
        {"n_isab": 2, "label": "sweep_isab2"},
        {"n_isab": 3, "label": "sweep_isab3"},
    ],
    "n_heads": [
        {"n_heads": 2, "label": "sweep_h2"},
        {"n_heads": 4, "label": "sweep_h4"},
        {"n_heads": 8, "label": "sweep_h8"},
    ],
}


# ── Ablation model variants ────────────────────────────────────────────────

class DeepSetsMixture(nn.Module):
    """
    Deep Sets (Zaheer et al. NeurIPS 2017): rho(mean_i phi(x_i)).
    Same input encoding and task heads as SetTransformerMixture.
    """

    def __init__(
        self,
        n_loci: int = 24,
        d_locus: int = 16,
        d_model: int = 128,
        n_classes: int = 45,
        n_noc: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_loci = n_loci
        self.locus_embed = nn.Embedding(n_loci + 1, d_locus, padding_idx=n_loci)
        d_in = d_locus + 2
        self.phi = nn.Sequential(
            nn.Linear(d_in, d_model),  nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
        )
        self.rho = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.cls_head    = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.reject_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.noc_head    = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_noc))

    def encode(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        locus_idx  = tokens[:, :, 0].long().clamp(0, self.n_loci - 1)
        continuous = tokens[:, :, 1:]
        locus_emb  = self.locus_embed(locus_idx)
        x = torch.cat([locus_emb, continuous], dim=-1)   # (B, N, d_in)
        x = self.phi(x)                                   # (B, N, d_model)
        x = x * mask.unsqueeze(-1)                        # zero out padding
        count = mask.float().sum(1, keepdim=True).clamp(1)
        z = x.sum(1) / count                              # mean pool  (B, d_model)
        return self.rho(z)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        z = self.encode(tokens, mask)
        return {
            "logits_cls":   self.cls_head(z),
            "logit_reject": self.reject_head(z),
            "logits_noc":   self.noc_head(z),
        }


class MLPMixture(nn.Module):
    """
    Flat MLP encoder: 590-dim Xflat -> hidden -> d_model.
    No reject head (flat features have no set structure for open-set training).
    """

    def __init__(
        self,
        d_in: int = 590,
        d_model: int = 128,
        n_classes: int = 45,
        n_noc: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, 512),    nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(512, 256),     nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(256, d_model), nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        self.noc_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_noc))

    def forward(self, x_flat: torch.Tensor) -> dict:
        z = self.encoder(x_flat)
        return {
            "logits_cls":   self.cls_head(z),
            "logit_reject": None,
            "logits_noc":   self.noc_head(z),
        }


# ── Datasets ───────────────────────────────────────────────────────────────

class ClosedSetDataset(Dataset):
    def __init__(self, split: str):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"tokens_{split}.npy"))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / f"mask_{split}.npy"))
        self.y      = torch.from_numpy(np.load(DATA_DIR / f"y_{split}_set.npy"))
        self.noc    = torch.from_numpy(np.load(DATA_DIR / f"noc_{split}.npy").astype(np.int64))

    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i], self.mask[i], self.y[i], self.noc[i]


class FlatClosedSetDataset(Dataset):
    """For MLP encoder: log1p flat features, pre-scaled by StandardScaler."""

    def __init__(self, split: str, scaler=None):
        X = np.load(DATA_DIR / f"Xflat_{split}.npy")
        if scaler is not None:
            X = scaler.transform(X).astype(np.float32)
        self.X   = torch.from_numpy(X)
        self.y   = torch.from_numpy(np.load(DATA_DIR / f"y_{split}_set.npy"))
        self.noc = torch.from_numpy(np.load(DATA_DIR / f"noc_{split}.npy").astype(np.int64))

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i], self.noc[i]


class OpenSetDataset(Dataset):
    def __init__(self):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / "tokens_open.npy"))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / "mask_open.npy"))

    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i], self.mask[i]


# ── Helpers ────────────────────────────────────────────────────────────────

def compute_pos_weight(y: np.ndarray) -> torch.Tensor:
    pos = y.sum(0).clip(min=1)
    neg = (1 - y).sum(0).clip(min=1)
    return torch.tensor(neg / pos, dtype=torch.float32)


@torch.no_grad()
def evaluate_cls(model, loader, threshold: float = 0.5, flat: bool = False):
    """Returns (macro_f1, y_true, y_pred, noc_true)."""
    model.eval()
    all_true, all_pred, all_noc = [], [], []
    for batch in loader:
        if flat:
            x, y, noc = batch
            out = model(x.to(DEVICE))
        else:
            tokens, mask, y, noc = batch
            out = model(tokens.to(DEVICE), mask.to(DEVICE))
        probs = torch.sigmoid(out["logits_cls"]).cpu().numpy()
        all_pred.append((probs >= threshold).astype(np.float32))
        all_true.append(y.numpy())
        all_noc.append(noc.numpy())
    y_true  = np.concatenate(all_true)
    y_pred  = np.concatenate(all_pred)
    noc_all = np.concatenate(all_noc)
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return mf1, y_true, y_pred, noc_all


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import hamming_loss, precision_score, recall_score
    return {
        "macro_f1":  float(f1_score(y_true, y_pred, average="macro",  zero_division=0)),
        "micro_f1":  float(f1_score(y_true, y_pred, average="micro",  zero_division=0)),
        "hamming":   float(hamming_loss(y_true, y_pred)),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred,    average="macro", zero_division=0)),
    }


# ── Training ───────────────────────────────────────────────────────────────

def train_variant(variant: str, cfg: dict, out_name: str | None = None) -> dict:
    name = out_name if out_name else variant
    out_dir = ROOT / "results" / f"ablation_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    use_flat    = (variant == "mlp_enc")
    use_reject  = (variant not in ("no_reject", "mlp_enc"))

    # --- Data ---
    if use_flat:
        from sklearn.preprocessing import StandardScaler
        import joblib
        X_tr = np.load(DATA_DIR / "Xflat_train.npy")
        scaler = StandardScaler().fit(X_tr)
        joblib.dump(scaler, out_dir / "scaler.pkl")
        train_ds = FlatClosedSetDataset("train", scaler)
        val_ds   = FlatClosedSetDataset("val",   scaler)
        test_ds  = FlatClosedSetDataset("test",  scaler)
    else:
        train_ds = ClosedSetDataset("train")
        val_ds   = ClosedSetDataset("val")
        test_ds  = ClosedSetDataset("test")

    open_ds   = OpenSetDataset() if use_reject else None
    open_iter = None

    bsz = cfg["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bsz, shuffle=True,
                              num_workers=0, pin_memory=(DEVICE.type == "cuda"))
    val_loader   = DataLoader(val_ds,  batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=0)

    if use_reject:
        open_iter = iter(DataLoader(
            open_ds,
            sampler=RandomSampler(open_ds, replacement=True,
                                  num_samples=len(train_ds) * cfg["epochs"]),
            batch_size=max(1, int(bsz * cfg.get("open_ratio", 0.25))),
            num_workers=0,
        ))

    # --- Model ---
    d_model  = cfg.get("d_model",  128)
    n_classes = cfg.get("n_classes", 45)
    n_noc    = cfg.get("n_noc",    6)
    dropout  = cfg.get("dropout",  0.1)

    if variant in ("full", "no_noc", "no_reject"):
        model = SetTransformerMixture(
            n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16),
            d_model=d_model, n_heads=cfg.get("n_heads", 4),
            n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
            n_classes=n_classes, n_noc=n_noc, dropout=dropout,
        ).to(DEVICE)
    elif variant == "deep_sets":
        model = DeepSetsMixture(
            n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16),
            d_model=d_model, n_classes=n_classes, n_noc=n_noc, dropout=dropout,
        ).to(DEVICE)
    elif variant == "mlp_enc":
        model = MLPMixture(
            d_in=590, d_model=d_model, n_classes=n_classes,
            n_noc=n_noc, dropout=dropout,
        ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    alpha = cfg.get("alpha_reject", 0.5) if use_reject else 0.0
    beta  = cfg.get("beta_noc",     0.3) if variant != "no_noc" else 0.0

    print(f"\n{'='*62}")
    print(f"  Variant  : {variant.upper()}")
    print(f"  Params   : {n_params:,}  |  Device: {DEVICE}")
    print(f"  alpha_rej: {alpha}  beta_noc: {beta}")
    print(f"  Data     : {'flat 590-dim' if use_flat else 'set tokens (B, N, 3)'}")
    print("="*62)

    # --- Loss ---
    y_train_np = np.load(DATA_DIR / "y_train_set.npy")
    pos_weight = compute_pos_weight(y_train_np).to(DEVICE)
    bce_cls = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    bce_rej = nn.BCEWithLogitsLoss()
    ce_noc  = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("lr", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
    )

    best_f1, best_epoch, patience_cnt = 0.0, 0, 0
    patience = cfg.get("patience", 15)
    epochs   = cfg.get("epochs", 100)
    history  = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        n_seen  = 0

        for batch in train_loader:
            if use_flat:
                x, y, noc = batch
                x = x.to(DEVICE); y = y.to(DEVICE); noc = noc.to(DEVICE)
                out = model(x)
            else:
                tokens, mask, y, noc = batch
                tokens = tokens.to(DEVICE); mask = mask.to(DEVICE)
                y = y.to(DEVICE); noc = noc.to(DEVICE)
                out = model(tokens, mask)

            loss = bce_cls(out["logits_cls"], y)
            if beta > 0:
                loss = loss + beta * ce_noc(out["logits_noc"], noc)

            if use_reject:
                rej_closed  = out["logit_reject"]
                rej_lbl     = torch.zeros(len(y), 1, device=DEVICE)
                try:
                    o_tok, o_mask = next(open_iter)
                except StopIteration:
                    open_iter = iter(DataLoader(
                        open_ds,
                        sampler=RandomSampler(open_ds, replacement=True,
                                              num_samples=len(open_ds) * 10),
                        batch_size=max(1, int(bsz * cfg.get("open_ratio", 0.25))),
                        num_workers=0,
                    ))
                    o_tok, o_mask = next(open_iter)
                rej_open     = model(o_tok.to(DEVICE), o_mask.to(DEVICE))["logit_reject"]
                rej_open_lbl = torch.ones(len(o_tok), 1, device=DEVICE)
                loss = loss + alpha * bce_rej(
                    torch.cat([rej_closed, rej_open]),
                    torch.cat([rej_lbl,    rej_open_lbl]),
                )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(y)
            n_seen  += len(y)

        ep_loss /= n_seen
        val_f1, _, _, _ = evaluate_cls(model, val_loader, flat=use_flat)
        scheduler.step(val_f1)
        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "loss": round(ep_loss, 4), "val_f1": round(val_f1, 4)})

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Ep {epoch:3d} | loss={ep_loss:.4f} | val_f1={val_f1:.4f} | lr={lr_now:.1e}")

        if val_f1 > best_f1:
            best_f1, best_epoch, patience_cnt = val_f1, epoch, 0
            torch.save(model.state_dict(), out_dir / "best_model.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stop ep {epoch}  (best {best_epoch}, val_f1={best_f1:.4f})")
                break

    print(f"  Trained {epoch} epochs in {time.time()-t0:.1f}s")

    # --- Threshold search on val ---
    model.load_state_dict(torch.load(out_dir / "best_model.pt", weights_only=True))
    model.eval()
    all_probs, all_true_v = [], []
    with torch.no_grad():
        for batch in val_loader:
            if use_flat:
                x, y, _ = batch
                out = model(x.to(DEVICE))
            else:
                tokens, mask, y, _ = batch
                out = model(tokens.to(DEVICE), mask.to(DEVICE))
            all_probs.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
            all_true_v.append(y.numpy())
    va_probs = np.concatenate(all_probs)
    va_true  = np.concatenate(all_true_v)

    best_t, best_tf1 = 0.5, 0.0
    for t in np.arange(0.2, 0.85, 0.05):
        f1 = float(f1_score(va_true, (va_probs >= t).astype(int),
                            average="macro", zero_division=0))
        if f1 > best_tf1:
            best_tf1, best_t = f1, float(t)
    print(f"  Best threshold: {best_t:.2f}  (val macro F1={best_tf1:.4f})")

    # --- Test evaluation ---
    _, y_te_true, y_te_pred, noc_te = evaluate_cls(model, test_loader, best_t, flat=use_flat)
    te_m = compute_metrics(y_te_true, y_te_pred)

    print(f"\n  TEST  MacroF1={te_m['macro_f1']:.4f}  MicroF1={te_m['micro_f1']:.4f}"
          f"  EM={te_m['exact_match']:.4f}  Hamming={te_m['hamming']:.4f}")

    # Per-NOC exact match
    exact   = np.all(y_te_true == y_te_pred, axis=1)
    per_noc = {}
    for noc in sorted(np.unique(noc_te)):
        m = noc_te == noc
        per_noc[int(noc)] = {"em": round(float(exact[m].mean()), 4), "n": int(m.sum())}
    noc_str = "  ".join(f"NOC{k}:{v['em']:.3f}(n={v['n']})" for k, v in per_noc.items())
    print(f"  Per-NOC: {noc_str}")

    # --- Reject AUROC ---
    auroc = None
    if use_reject:
        open_loader = DataLoader(open_ds, batch_size=256, shuffle=False)
        scores, lbls = [], []
        with torch.no_grad():
            for tokens, mask, _, _ in test_loader:
                s = torch.sigmoid(model(tokens.to(DEVICE), mask.to(DEVICE))["logit_reject"])
                scores.append(s.cpu().numpy())
                lbls.append(np.zeros(len(tokens)))
            for o_tok, o_mask in open_loader:
                s = torch.sigmoid(model(o_tok.to(DEVICE), o_mask.to(DEVICE))["logit_reject"])
                scores.append(s.cpu().numpy())
                lbls.append(np.ones(len(o_tok)))
        try:
            auroc = float(roc_auc_score(
                np.concatenate(lbls),
                np.concatenate(scores).ravel(),
            ))
            print(f"  Reject AUROC (closed vs open): {auroc:.4f}")
        except Exception as e:
            print(f"  AUROC error: {e}")

    # --- Save ---
    np.save(out_dir / "y_test_pred.npy", y_te_pred)
    np.save(out_dir / "y_test_true.npy", y_te_true)
    result = {
        "variant":        variant,
        "n_params":       n_params,
        "best_val_f1":    round(best_f1, 4),
        "best_epoch":     best_epoch,
        "best_threshold": round(best_t, 2),
        "reject_auroc":   auroc,
        "test":           te_m,
        "per_noc":        per_noc,
        "history":        history,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"  Saved -> {out_dir}\n")
    return result


# ── Architecture sweep ─────────────────────────────────────────────────────

def run_sweep(axis: str, cfg: dict) -> list[dict]:
    """Train 'full' variant over one hyperparameter axis, report summary table."""
    sweep_cfgs = SWEEP_CONFIGS.get(axis)
    if sweep_cfgs is None:
        raise ValueError(f"Unknown sweep axis: {axis}. Choose: {list(SWEEP_CONFIGS)}")
    print(f"\n{'='*64}")
    print(f"  SWEEP: {axis}  ({len(sweep_cfgs)} configs)")
    print("="*64)
    all_results = []
    for sc in sweep_cfgs:
        label = sc.pop("label")
        run_cfg = cfg.copy()
        run_cfg.update(sc)
        r = train_variant("full", run_cfg, out_name=label)
        r["sweep_label"] = label
        r["sweep_overrides"] = sc
        all_results.append(r)
        sc["label"] = label  # restore for future calls

    print(f"\n{'='*72}")
    print(f"  SWEEP SUMMARY: {axis}")
    print("="*72)
    hdr = f"  {'Config':<20} {'MacroF1':>8} {'MicroF1':>8} {'ExactMatch':>11} {'Hamming':>8}"
    print(hdr); print("  " + "-"*56)
    for r in all_results:
        t = r["test"]
        print(f"  {r['sweep_label']:<20} {t['macro_f1']:>8.4f} {t['micro_f1']:>8.4f}"
              f" {t['exact_match']:>11.4f} {t['hamming']:>8.4f}")
    print("="*72)

    (ROOT / "results").mkdir(exist_ok=True)
    out_path = ROOT / "results" / f"sweep_{axis}.json"
    import json as _json
    with open(out_path, "w") as f:
        _json.dump(all_results, f, indent=2)
    print(f"Saved -> {out_path}\n")
    return all_results


# ── Summary table ──────────────────────────────────────────────────────────

def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*72}")
    print("  ABLATION SUMMARY  (test set, 1,325 samples, GF29cycles)")
    print("="*72)
    hdr = f"  {'Variant':<12} {'MacroF1':>8} {'MicroF1':>8} {'ExactMatch':>11} {'Hamming':>8} {'RejectAUC':>10}"
    print(hdr)
    print("  " + "-"*68)
    for r in results:
        t   = r["test"]
        auc = f"{r['reject_auroc']:.4f}" if r["reject_auroc"] is not None else "    N/A"
        print(
            f"  {r['variant']:<12} {t['macro_f1']:>8.4f} {t['micro_f1']:>8.4f}"
            f" {t['exact_match']:>11.4f} {t['hamming']:>8.4f} {auc:>10}"
        )
    print("="*72)


# ── Entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",    default="full",
                        choices=VARIANTS + ["all"])
    parser.add_argument("--sweep",      default=None,
                        choices=list(SWEEP_CONFIGS) + ["all"],
                        help="Architecture sweep axis (m_inducing / n_isab / n_heads / all)")
    parser.add_argument("--epochs",     type=int,   default=None)
    parser.add_argument("--batch_size", type=int,   default=None)
    parser.add_argument("--lr",         type=float, default=None)
    args = parser.parse_args()

    cfg_path = ROOT / "configs" / "set_transformer.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    for k in ("epochs", "batch_size", "lr"):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    if args.sweep:
        axes = list(SWEEP_CONFIGS) if args.sweep == "all" else [args.sweep]
        for ax in axes:
            run_sweep(ax, cfg.copy())
    elif args.variant == "all":
        all_results = []
        for v in VARIANTS:
            r = train_variant(v, cfg.copy())
            all_results.append(r)
        print_summary(all_results)
    else:
        train_variant(args.variant, cfg.copy())
