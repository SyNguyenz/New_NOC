"""models/ordinal.py — CORN rank-consistent ordinal regression (Shi, Cao, Raschka 2023,
arXiv 2111.08851; coral-pytorch). Ported VERBATIM from the original trainer so the `noc_head_v2` CORN
count head trains with the exact loss the published inc22 arm used.

For levels 1..K the head emits K-1 logits; task k models P(y > k | y >= k). Rank-consistent at
inference: P(y>k) = prod_{j<=k} sigmoid(logit_j) is monotonically non-increasing.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


# ── CORN ordinal regression (Shi/Cao/Raschka 2023) ──────────────────────────
def corn_loss(logits: torch.Tensor, levels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """logits (B, n_classes-1); levels (B,) integer rank in 1..n_classes. Conditional-prob CORN loss."""
    device = logits.device
    y = levels.long() - 1                                  # -> 0..K-1
    losses = []
    for k in range(n_classes - 1):
        # task k (threshold between rank k+1 and k+2): conditional on y >= k  => rank >= k+1
        mask = y >= k
        if mask.sum() == 0:
            continue
        lk = logits[mask, k]
        tk = (y[mask] > k).float()                        # 1 if rank exceeds k+1
        losses.append(F.binary_cross_entropy_with_logits(lk, tk, reduction="mean"))
    if not losses:
        return torch.zeros((), device=device)
    return torch.stack(losses).mean()


def corn_probs(logits: torch.Tensor, n_classes: int) -> torch.Tensor:
    """logits (B, n_classes-1) -> per-class probability (B, n_classes) over ranks 1..n_classes.
    P(y>k) = prod_{j<=k} sigmoid(logit_j); P(y=k) = P(y>=k) - P(y>=k+1)."""
    cond = torch.sigmoid(logits)                          # (B, K-1) = P(y>k | y>=k)
    cum = torch.cumprod(cond, dim=1)                      # (B, K-1) = P(y > k) for k=1..K-1
    B = logits.size(0)
    ge = torch.cat([torch.ones(B, 1, device=logits.device), cum], dim=1)  # P(y>=k), k=1..K
    ge_next = torch.cat([cum, torch.zeros(B, 1, device=logits.device)], dim=1)  # P(y>=k+1)
    p = (ge - ge_next).clamp_min(0.0)                     # P(y=k)
    return p / p.sum(1, keepdim=True).clamp_min(1e-8)


