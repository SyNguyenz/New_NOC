"""
Attention weight visualization for Set Transformer.

Hooks into the PMA (Pooling by Multihead Attention) layer to capture
which allele tokens receive highest weight when building the mixture
embedding z_mix. Aggregates weights by locus to show which STR markers
are most informative for contributor identification.

Outputs:
  results/set_transformer/attn_locus_importance.json
  results/set_transformer/attn_locus_importance.png
  results/set_transformer/attn_locus_per_noc.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
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
        self.noc    = torch.from_numpy(
            np.load(DATA_DIR / f"noc_{split}.npy").astype(np.int64))

    def __len__(self): return len(self.tokens)
    def __getitem__(self, i): return self.tokens[i], self.mask[i], self.noc[i]


# ── Attention hooking ─────────────────────────────────────────────────────

def attach_pma_hook(model: SetTransformerMixture):
    """
    Monkey-patch PMA's mab.attn.forward to capture attention weights.
    Returns a dict; 'pma' key is populated after each forward pass.
    """
    captured = {"pma": None}
    orig_forward = model.pma.mab.attn.forward

    def hooked_forward(query, key, value, key_padding_mask=None, **kwargs):
        out, weights = orig_forward(
            query, key, value,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,  # average over heads
        )
        captured["pma"] = weights.detach().cpu()  # (B, k, N)
        return out, None  # MAB expects (attn_out, _)

    model.pma.mab.attn.forward = hooked_forward
    return captured


@torch.no_grad()
def collect_attention(model, loader, captured):
    """Run inference, collect (attn_weights, token_locus_idx, mask, noc) per batch."""
    all_attn = []   # list of (B, N) arrays
    all_loci = []   # list of (B, N) int arrays — locus index per token
    all_mask = []   # list of (B, N) bool arrays
    all_noc  = []

    for tokens, mask, noc in loader:
        tokens_d = tokens.to(DEVICE); mask_d = mask.to(DEVICE)
        model.encode(tokens_d, mask_d)

        attn = captured["pma"]          # (B, 1, N) or (B, N) depending on heads
        if attn.dim() == 3:
            attn = attn[:, 0, :]        # (B, N)

        locus_idx = tokens[:, :, 0].long().numpy()  # (B, N)
        all_attn.append(attn.numpy())
        all_loci.append(locus_idx)
        all_mask.append(mask.numpy())
        all_noc.append(noc.numpy())

    attn_all = np.concatenate(all_attn)   # (N_samples, MAX_SEQ)
    loci_all = np.concatenate(all_loci)
    mask_all = np.concatenate(all_mask)
    noc_all  = np.concatenate(all_noc)
    return attn_all, loci_all, mask_all, noc_all


# ── Aggregation ──────────────────────────────────────────────────────────

def locus_importance(attn, loci, mask, n_loci: int = 24):
    """
    For each sample, sum attention weights for each locus.
    Then average over samples → (n_loci,) mean importance.
    """
    N, S = attn.shape
    locus_attn = np.zeros((N, n_loci), dtype=np.float32)
    for i in range(N):
        valid = mask[i]  # (S,) bool
        if valid.sum() == 0:
            continue
        a = attn[i] * valid.astype(np.float32)  # zero out padding
        # Normalize per sample so each sums to 1 over valid tokens
        total = a.sum()
        if total > 0:
            a = a / total
        for t in range(S):
            if valid[t]:
                l = int(loci[i, t])
                if 0 <= l < n_loci:
                    locus_attn[i, l] += a[t]
    return locus_attn  # (N, n_loci)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_locus_importance(mean_imp, locus_names, out_path, title="Locus importance"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    order = np.argsort(-mean_imp)
    labels = [locus_names[i] for i in order]
    values = mean_imp[order]

    fig, ax = plt.subplots(figsize=(12, 4))
    bars = ax.bar(range(len(labels)), values, color="#4C72B0", alpha=0.85)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("Mean attention weight (normalized)")
    ax.set_title(title)
    ax.set_xlim(-0.5, len(labels) - 0.5)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"  Saved -> {out_path}")


def plot_per_noc(locus_attn, noc_all, locus_names, out_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    nocs = sorted(np.unique(noc_all[noc_all > 0]).tolist())
    n_loci = locus_attn.shape[1]
    fig, axes = plt.subplots(1, len(nocs), figsize=(4 * len(nocs), 3.5), sharey=True)
    if len(nocs) == 1:
        axes = [axes]

    all_means = []
    for ax, noc in zip(axes, nocs):
        mask = noc_all == noc
        mean_imp = locus_attn[mask].mean(axis=0)
        all_means.append(mean_imp)
        order = np.argsort(-mean_imp)
        ax.bar(range(n_loci), mean_imp[order], color="#C44E52", alpha=0.75)
        ax.set_xticks(range(n_loci))
        ax.set_xticklabels([locus_names[i] for i in order],
                           rotation=70, ha="right", fontsize=7)
        ax.set_title(f"NOC={noc} (n={mask.sum()})", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("Mean attention weight")

    plt.suptitle("PMA attention by number of contributors", y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved -> {out_path}")

    return all_means


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",   default=str(OUT_DIR / "best_model.pt"))
    parser.add_argument("--config", default=str(ROOT / "configs" / "set_transformer.json"))
    parser.add_argument("--split",  default="test")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    with open(DATA_DIR / "meta_set.json") as f:
        meta = json.load(f)
    locus_names = meta["loci"]   # list of 24 locus names
    n_loci = len(locus_names)

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

    captured = attach_pma_hook(model)

    loader = DataLoader(ClosedSet(args.split), batch_size=256, shuffle=False)
    attn_all, loci_all, mask_all, noc_all = collect_attention(model, loader, captured)
    print(f"Collected attention for {len(attn_all)} samples")

    locus_attn = locus_importance(attn_all, loci_all, mask_all, n_loci)
    mean_imp   = locus_attn.mean(axis=0)  # (n_loci,)

    # Rank loci
    order = np.argsort(-mean_imp)
    print("\n-- Locus importance (top-10, PMA attention) --")
    for rank, i in enumerate(order[:10], 1):
        print(f"  {rank:2d}. {locus_names[i]:<15} {mean_imp[i]:.4f}")

    # Save JSON
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "split": args.split,
        "loci": locus_names,
        "mean_importance": mean_imp.tolist(),
        "ranked_loci": [locus_names[i] for i in order],
        "ranked_importance": [float(mean_imp[i]) for i in order],
    }
    # Per-NOC
    nocs_unique = sorted(np.unique(noc_all[noc_all > 0]).tolist())
    per_noc = {}
    for noc in nocs_unique:
        m = noc_all == noc
        mi = locus_attn[m].mean(axis=0)
        per_noc[str(noc)] = {
            "n": int(m.sum()),
            "mean_importance": mi.tolist(),
            "top3_loci": [locus_names[i] for i in np.argsort(-mi)[:3]],
        }
    result["per_noc"] = per_noc

    out_json = OUT_DIR / "attn_locus_importance.json"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved -> {out_json}")

    # Plots
    plot_locus_importance(
        mean_imp, locus_names,
        OUT_DIR / "attn_locus_importance.png",
        title="PMA attention: mean locus importance (all NOC)",
    )
    plot_per_noc(
        locus_attn, noc_all, locus_names,
        OUT_DIR / "attn_locus_per_noc.png",
    )


if __name__ == "__main__":
    main()
