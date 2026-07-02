"""
Post-hoc analysis of the Set Transformer checkpoint.

Loads best ST model, runs on the test set, and reports:
  - NOC head accuracy per class + overall (compare with deepNoC 90% on PROVEDIt)
  - Calibration: ECE + reliability bins for reject head and aggregated donor heads
  - Saves raw probabilities for downstream use

Output: results/set_transformer/analysis.json + probs_test.npy + probs_reject.npy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "set_transformer"
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClosedSet(Dataset):
    def __init__(self, split: str):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / f"tokens_{split}.npy"))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / f"mask_{split}.npy"))
        self.y      = torch.from_numpy(np.load(DATA_DIR / f"y_{split}_set.npy"))
        self.noc    = torch.from_numpy(
            np.load(DATA_DIR / f"noc_{split}.npy").astype(np.int64))

    def __len__(self): return len(self.tokens)
    def __getitem__(self, i):
        return self.tokens[i], self.mask[i], self.y[i], self.noc[i]


class OpenSet(Dataset):
    def __init__(self):
        self.tokens = torch.from_numpy(np.load(DATA_DIR / "tokens_open.npy"))
        self.mask   = torch.from_numpy(np.load(DATA_DIR / "mask_open.npy"))
    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i], self.mask[i]


@torch.no_grad()
def run_closed(model, loader):
    probs_cls, probs_rej, logits_noc, y_all, noc_all = [], [], [], [], []
    for tokens, mask, y, noc in loader:
        out = model(tokens.to(DEVICE), mask.to(DEVICE))
        probs_cls.append(torch.sigmoid(out["logits_cls"]).cpu().numpy())
        probs_rej.append(torch.sigmoid(out["logit_reject"]).cpu().numpy())
        logits_noc.append(out["logits_noc"].cpu().numpy())
        y_all.append(y.numpy()); noc_all.append(noc.numpy())
    return (np.concatenate(probs_cls), np.concatenate(probs_rej).ravel(),
            np.concatenate(logits_noc), np.concatenate(y_all),
            np.concatenate(noc_all))


@torch.no_grad()
def run_open(model, loader):
    probs_rej = []
    for tokens, mask in loader:
        out = model(tokens.to(DEVICE), mask.to(DEVICE))
        probs_rej.append(torch.sigmoid(out["logit_reject"]).cpu().numpy())
    return np.concatenate(probs_rej).ravel()


# ── Calibration ───────────────────────────────────────────────────────────

def reliability_bins(confidence: np.ndarray, correct: np.ndarray, n_bins: int = 10):
    """Returns list of (lo, hi, acc, conf, n) per bin and ECE."""
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    ece = 0.0
    N = len(confidence)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        n = int(mask.sum())
        if n == 0:
            out.append({"lo": float(lo), "hi": float(hi),
                        "acc": None, "conf": None, "n": 0})
            continue
        acc  = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        out.append({"lo": float(lo), "hi": float(hi),
                    "acc": acc, "conf": conf, "n": n})
        ece += (n / N) * abs(acc - conf)
    return out, float(ece)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default=str(OUT_DIR / "best_model.pt"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "set_transformer.json"))
    parser.add_argument("--thresh", type=float, default=0.80,
                        help="Sigmoid threshold for donor head (from val search)")
    parser.add_argument("--bins",   type=int,   default=10)
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

    test_loader = DataLoader(ClosedSet("test"), batch_size=256, shuffle=False)
    open_loader = DataLoader(OpenSet(),         batch_size=256, shuffle=False)

    probs_cls, probs_rej_te, logits_noc, y_true, noc_true = run_closed(model, test_loader)
    probs_rej_op = run_open(model, open_loader)
    print(f"Inference done. test={probs_cls.shape}, open={probs_rej_op.shape}")

    # ── NOC accuracy ──────────────────────────────────────────────────────
    noc_pred = logits_noc.argmax(axis=1)
    noc_acc_overall = float((noc_pred == noc_true).mean())
    noc_per = {}
    for k in sorted(np.unique(noc_true)):
        m = noc_true == k
        noc_per[int(k)] = {
            "acc": float((noc_pred[m] == noc_true[m]).mean()),
            "n":   int(m.sum()),
        }

    # Confusion matrix on NOC
    classes = sorted(np.unique(np.concatenate([noc_true, noc_pred])).tolist())
    cm = confusion_matrix(noc_true, noc_pred, labels=classes)
    print("\n-- NOC head accuracy --")
    print(f"  Overall: {noc_acc_overall:.4f}")
    for k, v in noc_per.items():
        print(f"  NOC={k}: {v['acc']:.4f}  (n={v['n']})")
    print(f"  Compare deepNoC (Taylor 2024) on PROVEDIt 1–5: 0.90 overall")

    # ── Calibration ───────────────────────────────────────────────────────
    # Reject head: bin by sigmoid score, "correct" = predicted-open matches true-open
    rej_score = np.concatenate([probs_rej_te, probs_rej_op])
    rej_label = np.concatenate([np.zeros(len(probs_rej_te)),
                                np.ones(len(probs_rej_op))])
    # Define "prediction" as score>0.5 → 1; "correct" = (pred==label)
    # For reliability, treat sigmoid as P(open), correct = (label==1)
    rej_bins, rej_ece = reliability_bins(rej_score, rej_label, n_bins=args.bins)
    print(f"\n-- Calibration: reject head --")
    print(f"  ECE = {rej_ece:.4f}")

    # Donor head: per-prediction reliability — only count positive predictions where
    # confidence is high; for each (sample, donor) pair compute (prob, true_label).
    # To keep it tractable, focus on the strongest donor per sample (max probability).
    max_conf = probs_cls.max(axis=1)
    max_idx  = probs_cls.argmax(axis=1)
    # "correct" = the max-prob donor is actually in the true label set
    max_correct = y_true[np.arange(len(y_true)), max_idx]
    donor_bins, donor_ece = reliability_bins(max_conf, max_correct.astype(float),
                                             n_bins=args.bins)
    print(f"-- Calibration: donor head (top-1 per sample) --")
    print(f"  ECE = {donor_ece:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(OUT_DIR / "probs_test.npy",   probs_cls)
    np.save(OUT_DIR / "probs_reject_test.npy", probs_rej_te)
    np.save(OUT_DIR / "probs_reject_open.npy", probs_rej_op)
    np.save(OUT_DIR / "logits_noc_test.npy", logits_noc)

    result = {
        "threshold_used": args.thresh,
        "noc_head": {
            "overall_accuracy": noc_acc_overall,
            "per_noc":          noc_per,
            "confusion_classes": classes,
            "confusion_matrix": cm.tolist(),
        },
        "calibration": {
            "n_bins": args.bins,
            "reject_head": {"ece": rej_ece, "bins": rej_bins},
            "donor_head_top1": {"ece": donor_ece, "bins": donor_bins},
        },
    }
    out_path = OUT_DIR / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
