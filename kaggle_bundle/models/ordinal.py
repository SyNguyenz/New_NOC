"""
models/ordinal.py — verified loss primitives for Increment-3 levers. Each implements a
PUBLISHED method exactly (no self-invented mechanism):

  corn_loss / corn_probs : CORN rank-consistent ordinal regression via conditional
      probabilities (Shi, Cao, Raschka 2023, arXiv 2111.08851; coral-pytorch). For levels
      1..K the head outputs K-1 logits; task k models P(y > k | y >= k). Rank-consistent at
      inference because P(y>k) = prod_{j<=k} sigmoid(logit_j) is monotonically non-increasing.

  supcon_loss : Supervised Contrastive Learning (Khosla et al. 2020, arXiv 2004.11362).
      L = mean_i  -1/|P(i)| sum_{p in P(i)} log exp(z_i.z_p/tau) / sum_{a!=i} exp(z_i.z_a/tau)
      with P(i) = samples sharing i's label. Features L2-normalised.

Run `python -m models.ordinal` for a self-check.
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


# ── Supervised contrastive loss (Khosla 2020) ───────────────────────────────
def supcon_loss(z: torch.Tensor, labels: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    """z (M, d) features (will be L2-normalised); labels (M,). Items with the SAME label are
    positives. Ignores anchors with no same-label positive in the batch."""
    if z.size(0) < 2:
        return torch.zeros((), device=z.device)
    z = F.normalize(z, dim=1)
    sim = (z @ z.t()) / tau                                # (M, M)
    M = z.size(0)
    self_mask = torch.eye(M, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))       # exclude self from denominator
    logprob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    logprob = logprob.masked_fill(self_mask, 0.0)         # diagonal is -inf; zero it (pos excludes it anyway)
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask   # (M, M) positives
    pos_count = pos.sum(1)
    valid = pos_count > 0
    if valid.sum() == 0:
        return torch.zeros((), device=z.device)
    # mean over positives per anchor, then mean over valid anchors
    per_anchor = (logprob * pos).sum(1)[valid] / pos_count[valid].clamp_min(1)
    return -per_anchor.mean()


if __name__ == "__main__":
    torch.manual_seed(0)
    # CORN: a head that perfectly orders should give low loss + argmax-correct probs
    B, K = 200, 5
    levels = torch.randint(1, K + 1, (B,))
    # construct logits that strongly encode the true rank (monotone)
    logit = torch.zeros(B, K - 1)
    for i in range(B):
        for k in range(K - 1):
            logit[i, k] = 4.0 if (levels[i] - 1) > k else -4.0
    l = corn_loss(logit, levels, K)
    p = corn_probs(logit, K)
    acc = (p.argmax(1) + 1 == levels).float().mean()
    print(f"CORN  loss={l.item():.4f}  argmax-acc={acc.item():.3f}  (expect low loss, acc~1.0)")
    assert l.item() < 0.05 and acc.item() > 0.99, "CORN self-check FAILED"
    # rank-consistency: P(y>=k) monotone non-increasing in k
    ge = torch.cumprod(torch.sigmoid(logit), 1)
    assert (ge[:, 1:] <= ge[:, :-1] + 1e-6).all(), "CORN not rank-consistent"

    # SupCon: two well-separated clusters -> low loss; random -> higher
    d = 16
    lab = torch.cat([torch.zeros(50), torch.ones(50)]).long()
    z_good = torch.cat([torch.randn(50, d) * 0.1 + 5, torch.randn(50, d) * 0.1 - 5])
    z_rand = torch.randn(100, d)
    lg = supcon_loss(z_good, lab, tau=0.1); lr = supcon_loss(z_rand, lab, tau=0.1)
    print(f"SupCon clustered={lg.item():.4f}  random={lr.item():.4f}  (expect clustered << random)")
    assert lg.item() < lr.item(), "SupCon self-check FAILED"
    print("ordinal.py self-checks PASSED")
