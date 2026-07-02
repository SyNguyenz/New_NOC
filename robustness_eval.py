"""
Robustness evaluation: model trained on Filtered data evaluated on UnFiltered data.

Loads the best ST checkpoint, builds features from UnFiltered GF29cycles CSVs
(using same allele bins from meta_set.json to keep feature space consistent),
then evaluates closed-set identification and open-set rejection.

This tests domain shift robustness — unfiltered data includes low-quality peaks
that the analytical threshold would have removed in training data.

Output: results/set_transformer/robustness.json
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_UNFILTERED = (ROOT / "data_raw" / "PROVEDIt_1-5-Person CSVs UnFiltered"
                  / "PROVEDIt_1-5-Person CSVs UnFiltered_3500_GF29cycles")
sys.path.insert(0, str(ROOT))
from models.set_transformer import SetTransformerMixture

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT = ROOT / "results" / "set_transformer" / "best_model.pt"
MAX_SEQ = 160

ALL_DONORS = list(range(1, 51))
_rng = np.random.default_rng(42)
_shuffled = _rng.permutation(ALL_DONORS).tolist()
UNKNOWN_DONORS = sorted(_shuffled[:5])
KNOWN_DONORS = sorted(_shuffled[5:])
KNOWN_SET = set(KNOWN_DONORS)


def allele_to_float(allele: str):
    a = str(allele).strip()
    if a in ("", "nan", "OL"): return None
    if a == "X": return -2.0
    if a == "Y": return -1.0
    try: return float(a)
    except ValueError: return None


def parse_donors(filename: str) -> list[int]:
    parts = filename.split("-")
    if len(parts) < 3: return []
    contrib_part = parts[2]
    if "_" in contrib_part:
        donors = []
        for seg in contrib_part.split("_"):
            m = re.match(r"^(\d+)", seg)
            if m: donors.append(int(m.group(1)))
        return donors
    m = re.match(r"^(\d+)", contrib_part)
    return [int(m.group(1))] if m else []


def build_features_from_unfiltered(meta: dict, return_names: bool = False):
    """
    Scan UnFiltered CSVs and extract features using the allele bins defined in meta_set.json.
    Returns (tokens_arr, mask_arr, y_arr, has_unknown_arr).
    """
    loci = meta["loci"]
    locus_to_idx = {l: i for i, l in enumerate(loci)}
    flat_col_index = {}
    for loc in loci:
        for av in meta["locus_bin_lists"][loc]:
            flat_col_index[(loc, float(av))] = len(flat_col_index)

    pattern = str(RAW_UNFILTERED / "**" / "*.csv")
    csv_files = glob.glob(pattern, recursive=True)
    print(f"  Found {len(csv_files)} UnFiltered GF29cycles CSVs")

    allele_cols = None
    sample_tokens = {}
    sample_donors = {}

    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path, low_memory=False)
        except Exception as e:
            print(f"  Skipping {csv_path}: {e}")
            continue
        if allele_cols is None:
            allele_cols = [c for c in df.columns if c.startswith("Allele ")]
        for sf, grp in df.groupby("Sample File"):
            donors = parse_donors(str(sf))
            if not donors: continue
            tokens = []
            for _, row in grp.iterrows():
                locus = row.get("Marker")
                if locus not in locus_to_idx: continue
                li = float(locus_to_idx[locus])
                for ac in allele_cols:
                    hc = ac.replace("Allele", "Height")
                    av = allele_to_float(row.get(ac))
                    if av is None: continue
                    h = row.get(hc)
                    if pd.isna(h): continue
                    try: h_val = float(h)
                    except (ValueError, TypeError): continue
                    tokens.append((li, av, float(np.log1p(h_val))))
            if not tokens: continue
            sample_tokens[sf] = tokens
            sample_donors[sf] = donors

    sample_files = sorted(sample_tokens.keys())
    print(f"  Valid samples: {len(sample_files)}")

    # Build arrays
    N = len(sample_files)
    tokens_arr = np.zeros((N, MAX_SEQ, 3), dtype=np.float32)
    mask_arr = np.zeros((N, MAX_SEQ), dtype=bool)
    y_arr = np.zeros((N, 45), dtype=np.float32)
    has_unknown = np.zeros(N, dtype=bool)

    for i, sf in enumerate(sample_files):
        toks = sample_tokens[sf]
        n = min(len(toks), MAX_SEQ)
        tokens_arr[i, :n] = np.array(toks[:n], dtype=np.float32)
        mask_arr[i, :n] = True
        donors = sample_donors[sf]
        has_unknown[i] = any(d not in KNOWN_SET for d in donors)
        for d in donors:
            if d in KNOWN_SET:
                y_arr[i, KNOWN_DONORS.index(d)] = 1.0

    if return_names:
        return tokens_arr, mask_arr, y_arr, has_unknown, sample_files
    return tokens_arr, mask_arr, y_arr, has_unknown


class UnfilteredSet(Dataset):
    def __init__(self, tokens, mask, y):
        self.tokens = torch.from_numpy(tokens)
        self.mask   = torch.from_numpy(mask)
        self.y      = torch.from_numpy(y)

    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i], self.mask[i], self.y[i]


@torch.no_grad()
def evaluate(model, loader, threshold: float = 0.80):
    all_true, all_pred, all_rej = [], [], []
    for tokens, mask, y in loader:
        out = model(tokens.to(DEVICE), mask.to(DEVICE))
        p = torch.sigmoid(out["logits_cls"]).cpu().numpy()
        r = torch.sigmoid(out["logit_reject"]).cpu().numpy().ravel()
        all_pred.append((p >= threshold).astype(np.float32))
        all_true.append(y.numpy())
        all_rej.append(r)
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_rej)


def metrics(y_true, y_pred):
    return {
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1":    float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(np.all(y_true == y_pred, axis=1).mean()),
        "hamming":     float(hamming_loss(y_true, y_pred)),
    }


def main():
    with open(DATA_DIR / "meta_set.json") as f:
        meta = json.load(f)
    with open(ROOT / "configs" / "set_transformer.json") as f:
        cfg = json.load(f)

    # Load test split sample names (from extract_metadata.py) to avoid leakage
    test_names_path = DATA_DIR / "meta_sample_names_test.json"
    open_names_path = DATA_DIR / "meta_sample_names_open.json"
    test_sample_set = None
    open_sample_set = None
    if test_names_path.exists():
        with open(test_names_path) as f:
            test_sample_set = set(json.load(f))
        print(f"Test split names loaded: {len(test_sample_set)} samples")
    if open_names_path.exists():
        with open(open_names_path) as f:
            open_sample_set = set(json.load(f))
        print(f"Open split names loaded: {len(open_sample_set)} samples")

    print(f"Device: {DEVICE}")
    model = SetTransformerMixture(
        n_loci=cfg["n_loci"], d_locus=cfg["d_locus"], d_model=cfg["d_model"],
        n_heads=cfg["n_heads"], n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
        n_classes=cfg["n_classes"], n_noc=cfg["n_noc"], dropout=cfg["dropout"],
    ).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded checkpoint: {CKPT}")

    print("\nExtracting UnFiltered features...")
    tokens_arr, mask_arr, y_arr, has_unknown, sample_names = \
        build_features_from_unfiltered(meta, return_names=True)

    # Restrict to test-split samples if names are available (avoids leakage)
    if test_sample_set is not None:
        test_mask = np.array([n in test_sample_set for n in sample_names], dtype=bool)
        open_mask_unf = np.array([n in (open_sample_set or set()) for n in sample_names], dtype=bool)
        closed_mask = test_mask
    else:
        closed_mask = ~has_unknown
        open_mask_unf = has_unknown

    # For open-set samples: use all open-split names from unfiltered
    open_mask = open_mask_unf if open_sample_set else has_unknown
    print(f"  Closed-set (test split matched): {closed_mask.sum()}  Open-set: {open_mask.sum()}")

    # Evaluate closed-set
    if closed_mask.sum() > 0:
        closed_ds = UnfilteredSet(tokens_arr[closed_mask], mask_arr[closed_mask], y_arr[closed_mask])
        loader = DataLoader(closed_ds, batch_size=256, shuffle=False)
        y_true_uf, y_pred_uf, rej_closed = evaluate(model, loader, threshold=0.80)
        m_uf = metrics(y_true_uf, y_pred_uf)
    else:
        m_uf = {}; rej_closed = np.array([])

    # Open-set AUROC
    auroc = None
    if open_mask.sum() > 0 and len(rej_closed) > 0:
        open_ds = UnfilteredSet(tokens_arr[open_mask], mask_arr[open_mask],
                                y_arr[open_mask])
        open_loader = DataLoader(open_ds, batch_size=256, shuffle=False)
        _, _, rej_open = evaluate(model, open_loader, threshold=0.80)
        labels = np.concatenate([np.zeros(len(rej_closed)), np.ones(len(rej_open))])
        scores = np.concatenate([rej_closed, rej_open])
        try:
            auroc = float(roc_auc_score(labels, scores))
        except Exception as e:
            print(f"  AUROC error: {e}")

    # Reference: filtered test metrics
    with open(ROOT / "results" / "set_transformer" / "metrics.json") as f:
        filt_metrics = json.load(f)

    print("\n" + "=" * 60)
    print("  ROBUSTNESS EVALUATION: Filtered -> UnFiltered (GF29cycles)")
    print("=" * 60)
    print(f"  {'Metric':<15} {'Filtered (test)':>17} {'UnFiltered':>12}")
    print("  " + "-" * 46)
    for k in ["macro_f1", "micro_f1", "exact_match", "hamming"]:
        fv = filt_metrics.get("test", filt_metrics).get(k, float("nan"))
        uv = m_uf.get(k, float("nan"))
        print(f"  {k:<15} {fv:>17.4f} {uv:>12.4f}")
    if auroc is not None:
        print(f"  {'reject_auroc':<15} {filt_metrics.get('reject_auroc', float('nan')):>17.4f} {auroc:>12.4f}")
    print("=" * 60)

    result = {
        "domain": "unfiltered_gf29cycles",
        "n_closed":   int(closed_mask.sum()),
        "n_open":     int(open_mask.sum()),
        "threshold":  0.80,
        "metrics_unfiltered": m_uf,
        "reject_auroc_unfiltered": auroc,
        "metrics_filtered_ref": filt_metrics.get("test", filt_metrics),
    }
    out_path = ROOT / "results" / "set_transformer" / "robustness.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
