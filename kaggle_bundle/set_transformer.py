"""
Set Transformer for 45-class multi-label contributor identification.

Architecture (planv2):
  tokens (B, N, 3) → InputProj → ISAB × n_isab → PMA_1 → z_mix (B, d)
  z_mix → cls_head   → 45 logits  (BCE, closed-set identification)
  z_mix → reject_head → 1 logit   (BCE, has-unknown contributor)
  z_mix → noc_head   → 6 logits   (CE,  number of contributors 0-5)

Token format per allele observation:
  dim 0: locus index    (int cast to long for embedding)
  dim 1: allele value   (float: X→-2, Y→-1, numeric allele)
  dim 2: log1p(height)  (float)

Reference: Lee et al. "Set Transformer" ICML 2019.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Per-feature numerical embedding ────────────────────────────────────────

class PeriodicNumEmbedding(nn.Module):
    """Per-feature periodic numerical embedding — Gorishniy 2022 ("On Embeddings
    for Numerical Features", arXiv 2203.05556), the PLR / Periodic-Linear-ReLU
    variant.

    Each standardized scalar feature x_j → v_j = [sin(2π c_j x_j), cos(2π c_j x_j)]
    with learnable PER-FEATURE frequencies c_j ∈ R^{n_freq} (init N(0, σ)), then a
    PER-FEATURE Linear + ReLU → d_emb. This replaces concatenating raw standardized
    scalars into one shared Linear, which is the rotation-invariance failure mode
    that makes plain NNs lose to trees on tabular lookup targets (Grinsztajn 2022,
    §3 of design_representation_redesign). Each feature gets its own embedding so
    allele / log_h / Hb / SR / rank / n-peaks / global-rel are no longer entangled
    on a single linear axis.
    """

    def __init__(self, n_num: int, d_emb: int = 8, n_freq: int = 8, sigma: float = 1.0):
        super().__init__()
        self.n_num = n_num
        self.d_emb = d_emb
        self.n_freq = n_freq
        # per-feature frequency coefficients c: (n_num, n_freq)
        self.coeffs = nn.Parameter(torch.randn(n_num, n_freq) * sigma)
        # per-feature linear: weight (n_num, 2*n_freq, d_emb), bias (n_num, d_emb)
        self.weight = nn.Parameter(torch.empty(n_num, 2 * n_freq, d_emb))
        self.bias = nn.Parameter(torch.zeros(n_num, d_emb))
        nn.init.trunc_normal_(self.weight, std=1.0 / math.sqrt(2 * n_freq))
        self.act = nn.ReLU(inplace=True)

    @property
    def out_dim(self) -> int:
        return self.n_num * self.d_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, n_num) standardized scalars
        v = (2 * math.pi) * x.unsqueeze(-1) * self.coeffs        # (B, N, n_num, n_freq)
        per = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)    # (B, N, n_num, 2*n_freq)
        emb = torch.einsum("bnfk,fkd->bnfd", per, self.weight) + self.bias  # (B, N, n_num, d_emb)
        emb = self.act(emb)
        return emb.flatten(-2)                                   # (B, N, n_num*d_emb)


# ── Sparse attention (Increment 2d §7) ─────────────────────────────────────

def sparsemax(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Sparsemax (Martins & Astudillo 2016, arXiv 1602.02068): Euclidean projection of z onto the
    probability simplex. Unlike softmax it can assign EXACTLY ZERO mass → a donor query attends to
    only its donor's alleles and zeros the rest (doc2 §7 'remove attention to spurious peaks').
    Padding is handled by the caller (logits set very negative → mass 0)."""
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    K = z.size(dim)
    rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = K
    rng = rng.view(shape)
    cumsum = z_sorted.cumsum(dim)
    support = (1.0 + rng * z_sorted) > cumsum            # support condition uses cumsum
    k = support.to(z.dtype).sum(dim, keepdim=True).clamp(min=1.0)
    tau = (cumsum.gather(dim, (k.long() - 1)) - 1.0) / k  # threshold uses cumsum - 1
    return torch.clamp(z - tau, min=0.0)


def entmax15(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """1.5-entmax (Peters, Niculae & Martins 2019, arXiv 1905.05702): the α=1.5 member of the α-entmax
    family. Softer than sparsemax (α=2) — keeps MORE keys with small mass while still able to assign
    exact zeros (interpolates softmax α→1 ↔ sparsemax α=2). Exact closed form (deep-spin forward), autograd
    handles the backward. Inc12: the per-donor decoder's sparsemax (hard) DROPS a faint donor's DIFFUSE
    evidence (the +.10 under-read measured vs the soft-vote ceiling); 1.5-entmax reads it back without
    going fully dense (dense-everywhere hurts the loud donors)."""
    max_val, _ = z.max(dim=dim, keepdim=True)
    z = (z - max_val) / 2                                # /2 = the (α-1) factor for α=1.5
    z_srt, _ = torch.sort(z, descending=True, dim=dim)
    K = z.size(dim)
    rng = torch.arange(1, K + 1, device=z.device, dtype=z.dtype)
    shape = [1] * z.dim(); shape[dim] = K
    rho = rng.view(shape)
    mean = z_srt.cumsum(dim) / rho
    mean_sq = (z_srt ** 2).cumsum(dim) / rho
    ss = rho * (mean_sq - mean ** 2)
    delta = (1 - ss) / rho
    tau = mean - torch.sqrt(torch.clamp(delta, min=0))
    support = (tau <= z_srt).sum(dim=dim, keepdim=True).clamp(min=1)
    tau_star = tau.gather(dim, support - 1)
    return torch.clamp(z - tau_star, min=0) ** 2


class SparseMAB(nn.Module):
    """Multihead cross-attention with a SPARSE/ADAPTIVE map instead of softmax, same residual/FFN/norm
    skeleton as MAB (drop-in for the per-donor decoder). Queries X attend to keys/values Y.
      aggr="sparsemax"   : α=2 Euclidean-simplex projection (doc2 §7, the original).
      aggr="entmax15"    : α=1.5 (softer — reads diffuse faint-donor evidence; Inc12).
      aggr="entmax_temp" : 1.5-entmax with a LEARNABLE per-head temperature (ASEntmax, 2508.17821) — each
                           head interpolates sparse(loud donor)↔dense(faint donor); init T=1 = entmax15."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, aggr: str = "sparsemax"):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.dh = d_model // n_heads; self.aggr = aggr
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(inplace=True),
                                nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        # ASEntmax: learnable per-head log-inverse-temperature; scores*exp(log_inv_temp); init 0 => T=1
        self.log_inv_temp = nn.Parameter(torch.zeros(n_heads)) if aggr == "entmax_temp" else None

    def _map(self, scores):
        if self.aggr == "sparsemax":
            return sparsemax(scores, dim=-1)
        if self.aggr == "entmax_temp":
            scores = scores * torch.exp(self.log_inv_temp).view(1, -1, 1, 1)  # per-head temperature
        return entmax15(scores, dim=-1)

    def forward(self, X, Y, key_padding_mask=None):
        B, Nq, _ = X.shape; Nk = Y.size(1)
        q = self.q(X).view(B, Nq, self.h, self.dh).transpose(1, 2)   # (B,h,Nq,dh)
        k = self.k(Y).view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = self.v(Y).view(B, Nk, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)      # (B,h,Nq,Nk)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e4)
        attn = self._map(scores)                                     # sparse/adaptive over keys
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, Nq, -1)
        H = self.norm1(X + self.o(out))
        return self.norm2(H + self.ff(H))


# ── Attention building blocks ──────────────────────────────────────────────

class MAB(nn.Module):
    """
    Multi-head Attention Block.
      H   = LayerNorm(X + MultiheadAttn(query=X, key=Y, value=Y))
      out = LayerNorm(H + FFN(H))
    key_padding_mask: (B, S_y) bool, True = ignore (PyTorch convention).
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out, _ = self.attn(X, Y, Y, key_padding_mask=key_padding_mask)
        attn_out = attn_out.nan_to_num(0.0)  # safe fallback: all-masked keys → zero output
        H = self.norm1(X + attn_out)
        return self.norm2(H + self.ff(H))


class ISAB(nn.Module):
    """
    Induced Set Attention Block (O(nm) complexity).
      H   = MAB(I_m, X)   — m learnable inducing points attend to input
      out = MAB(X, H)     — input attends back to compressed representation
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        m_inducing: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, m_inducing, d_model))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(d_model, n_heads, dropout)  # I attends to X
        self.mab1 = MAB(d_model, n_heads, dropout)  # X attends to H

    def forward(
        self, X: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # X: (B, N, d)  pad_mask: (B, N) True=ignore
        B = X.size(0)
        I = self.I.expand(B, -1, -1)                    # (B, m, d)
        H = self.mab0(I, X, key_padding_mask=pad_mask)  # (B, m, d)
        return self.mab1(X, H)                          # (B, N, d)


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.
      out = MAB(S_k, rFF(X))   — k seed vectors pool the set
    Returns (B, k, d); use k=1 for a single mixture embedding.

    A learned null key/value is appended to the pooled set so the attention key set is
    NEVER empty. When every real element is padding-masked (e.g. an all-out-of-panel
    sample after feas_filter), the seeds attend to the null token instead of softmax-ing
    over an all-masked row. nn.MultiheadAttention's all-masked row yields NaN, and MAB's
    nan_to_num only patches the forward value — in backward 0 × (NaN Jacobian) = NaN still
    corrupts parameters. The null key has >=1 valid key always, removing the NaN in BOTH
    passes. The null is a distinct learned constant, not a real peak, so it never
    reintroduces infeasible-allele evidence into the pooled representation.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        k_seeds: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, k_seeds, d_model))
        nn.init.xavier_uniform_(self.S)
        self.rff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True))
        self.mab = MAB(d_model, n_heads, dropout)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, d_model))   # never-masked pooling sink

    def forward(
        self, X: torch.Tensor, pad_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B = X.size(0)
        S = self.S.expand(B, -1, -1)                          # (B, k, d)
        Y = torch.cat([self.rff(X), self.null_kv.expand(B, -1, -1)], dim=1)  # (B, N+1, d)
        if pad_mask is not None:
            # append a never-masked column for the null key → an all-masked real set still
            # has >=1 valid key (prevents the all-masked NaN in forward AND backward).
            null_col = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
            pad_mask = torch.cat([pad_mask, null_col], dim=1)  # (B, N+1)
        return self.mab(S, Y, key_padding_mask=pad_mask)       # (B, k, d)


class SetNorm(nn.Module):
    """Set Norm (Zhang et al. ICML 2022, arXiv 2206.11925): standardize each set with a
    SINGLE per-set scalar mean/var over ALL valid elements × features, then per-feature affine.
    Unlike LayerNorm (per-element over features — removes element-specific scale, hurting sets),
    SetNorm removes only the global set mean/var, preserving element scale. Masked: stats over
    valid elements only."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)); self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        d = x.size(-1)
        if pad_mask is not None:
            v = (~pad_mask).unsqueeze(-1).to(x.dtype)            # (B,N,1) 1=valid
            n = v.sum(dim=(1, 2), keepdim=True).clamp(min=1) * d
            mu = (x * v).sum(dim=(1, 2), keepdim=True) / n
            var = (((x - mu) ** 2) * v).sum(dim=(1, 2), keepdim=True) / n
        else:
            mu = x.mean(dim=(1, 2), keepdim=True)
            var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gamma + self.beta


class MABpp(nn.Module):
    """Clean-path, pre-norm SetNorm Multihead Attention Block (Set Transformer++).
      H   = X + Attn(SetNorm(X)?, SetNorm(Y), Y)     (clean residual: X added unmodified)
      out = H + FFN(SetNorm(H))                       (clean residual, pre-norm)
    norm_q=False for inducing-point queries (paper: inducing points are NOT normalized)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, norm_q: bool = True):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(inplace=True),
                                nn.Linear(4 * d_model, d_model))
        self.norm_kv = SetNorm(d_model); self.norm_h = SetNorm(d_model)
        self.norm_q = SetNorm(d_model) if norm_q else None

    def forward(self, X, Y, q_mask=None, kv_mask=None):
        Xn = self.norm_q(X, q_mask) if self.norm_q is not None else X
        # BERT-style float additive mask (Devlin et al. 2019 §3.1): bool True → float -1e4 added to
        # attention logits before softmax.  When ALL keys are masked the logits are uniformly -1e4,
        # so softmax → 1/N instead of softmax(-inf,…) = NaN.  The NaN is prevented in BOTH forward
        # and backward (unlike nan_to_num which only patches the value; NaN Jacobian × 0 = NaN in
        # IEEE 754 and corrupts parameter gradients on the next backward pass).
        float_kv = kv_mask.float() * -1e4 if (kv_mask is not None and kv_mask.dtype == torch.bool) else kv_mask
        a, _ = self.attn(Xn, self.norm_kv(Y, kv_mask), Y, key_padding_mask=float_kv)
        H = X + a                                        # clean residual
        return H + self.ff(self.norm_h(H, q_mask))       # clean residual + pre-norm FFN


class SigmoidMABpp(nn.Module):
    """MABpp with NON-COMPETITIVE SIGMOID cross-attention (Ramapuram et al., Apple 2024, arXiv 2409.04431).

        Attn(X,Y) = σ(QKᵀ/√d_h + b) · V ,   b = −log(n_keys)   (per-sample, the paper's best-practice bias
                                                                that recovers a finite output limit as the
                                                                key count grows), NO row-normalization.
    Drop-in for ISAB++'s mab0: standard MAB pools the N peaks into each inducing point by a softmax-over-PEAKS
    weighted MEAN, so the tall MAJOR peaks monopolize the normalized mass and a faint MINOR peak is washed
    (F35/F35b: relaxing the mab0 competition de-absorbs the faint tail; mab1 relaxation does NOT). Sigmoid
    gives each (inducing, peak) pair an INDEPENDENT [0,1] gate — a faint peak can be admitted to an inducing
    point regardless of how tall the majors are. Same clean-path pre-norm SetNorm skeleton as MABpp."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0,
                 norm_q: bool = True, learnable_bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.dh = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model); self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model); self.wo = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(inplace=True),
                                nn.Linear(4 * d_model, d_model))
        self.norm_kv = SetNorm(d_model); self.norm_h = SetNorm(d_model)
        self.norm_q = SetNorm(d_model) if norm_q else None
        self.drop = nn.Dropout(dropout)
        # learnable per-block bias offset (added to −log n); paper allows fixed or learnable b
        self.bias = nn.Parameter(torch.zeros(1)) if learnable_bias else None

    def forward(self, X, Y, q_mask=None, kv_mask=None):
        Xn = self.norm_q(X, q_mask) if self.norm_q is not None else X
        Yn = self.norm_kv(Y, kv_mask)
        B, Lq, d = Xn.shape; Lk = Yn.size(1)
        q = self.wq(Xn).view(B, Lq, self.h, self.dh).transpose(1, 2)        # (B,h,Lq,dh)
        k = self.wk(Yn).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.wv(Y ).view(B, Lk, self.h, self.dh).transpose(1, 2)        # value from UNNORMED Y (= MABpp)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)             # (B,h,Lq,Lk)
        if kv_mask is not None:                                            # b = −log(#valid keys) per sample
            nval = (~kv_mask).sum(1).clamp(min=1).to(scores.dtype)         # (B,)
            b = -torch.log(nval).view(B, 1, 1, 1)
        else:
            b = -math.log(max(Lk, 1))
        if self.bias is not None:
            b = b + self.bias
        A = torch.sigmoid(scores + b)                                      # non-competitive gates, NO normalization
        if kv_mask is not None:
            A = A.masked_fill(kv_mask[:, None, None, :], 0.0)              # padded keys contribute 0
        a = (self.drop(A) @ v).transpose(1, 2).reshape(B, Lq, d)
        a = self.wo(a)
        H = X + a                                                          # clean residual
        return H + self.ff(self.norm_h(H, q_mask))                        # clean residual + pre-norm FFN


class ISABpp(nn.Module):
    """ISAB++ (Set Transformer++): ISAB with SetNorm + clean-path residuals → trainable when deep.

    nc_mab0 / nc_mab1 (Increment 11, F35b): swap the softmax MAB for non-competitive SigmoidMABpp at that
    step. The N5 wall mechanism is the mab0 softmax-over-peaks competition (tall majors wash faint minors);
    the probe localized the lever to mab0 only (mab1 relaxation hurt)."""
    def __init__(self, d_model: int, n_heads: int, m_inducing: int, dropout: float = 0.0,
                 nc_mab0: bool = False, nc_mab1: bool = False, nc_learnable_bias: bool = False):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, m_inducing, d_model)); nn.init.xavier_uniform_(self.I)
        self.mab0 = (SigmoidMABpp(d_model, n_heads, dropout, norm_q=False, learnable_bias=nc_learnable_bias)
                     if nc_mab0 else MABpp(d_model, n_heads, dropout, norm_q=False))   # inducing attend X
        self.mab1 = (SigmoidMABpp(d_model, n_heads, dropout, norm_q=True, learnable_bias=nc_learnable_bias)
                     if nc_mab1 else MABpp(d_model, n_heads, dropout, norm_q=True))     # X attend H

    def forward(self, X, pad_mask=None):
        I = self.I.expand(X.size(0), -1, -1)
        H = self.mab0(I, X, q_mask=None, kv_mask=pad_mask)           # (B,m,d)
        return self.mab1(X, H, q_mask=pad_mask, kv_mask=None)        # (B,N,d)


# ── Mass-preserving inducing-point compression (Increment 7 — F31 height/mass-dominance lever) ────

class MassMABpp(nn.Module):
    """Mass-PRESERVING cross-attention for the inducing-point compression step.

    Grounded in Fischer & Gärtner 2024 ("Attention Normalization Impacts Cardinality Generalization in
    Slot Attention", arXiv 2407.04170 — scaled weighted SUM vs weighted MEAN) + AttSets (Yang 2018).

    DIAGNOSIS (F31, 3-seed): standard MAB pools the N peaks into each inducing point by a softmax-over-
    PEAKS WEIGHTED MEAN. The many tall MAJOR-contributor peaks dominate that normalized mass, so the few
    faint MINOR-contributor peaks are averaged away → the encoded token H for a minor's private allele is
    washed (attr .07; recovers to .81 only when the major peaks are removed). This is the root of the N5
    oracle wall.

    FIX: each PEAK is softmax-assigned over the INDUCING POINTS (competitive slot assignment — softmax on
    the query/inducing axis, NOT the key/peak axis), then each inducing point aggregates its assigned peaks
    by a SCALED WEIGHTED SUM (NOT divided by the per-slot assignment mass). A distinctive minor peak can
    then claim its own inducing point and its signal is PRESERVED instead of washed by the major majority.
    Same clean-path pre-norm SetNorm skeleton as MABpp, so it is a drop-in for ISAB++'s mab0.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads; self.dh = d_model // n_heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model); self.o = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(inplace=True),
                                nn.Linear(4 * d_model, d_model))
        self.norm_kv = SetNorm(d_model); self.norm_h = SetNorm(d_model)  # inducing queries unnormed (paper)
        self.scale = nn.Parameter(torch.ones(1))         # learned scale for the cardinality-robust SUM
        self.drop = nn.Dropout(dropout)

    def forward(self, Q, KV, kv_mask=None):
        # Q: (B, M, d) inducing points (queries) ; KV: (B, N, d) peaks ; kv_mask: (B, N) True = pad
        B, M, _ = Q.shape; N = KV.size(1)
        kvn = self.norm_kv(KV, kv_mask)
        q = self.q(Q).view(B, M, self.h, self.dh).transpose(1, 2)        # (B,h,M,dh)
        k = self.k(kvn).view(B, N, self.h, self.dh).transpose(1, 2)      # (B,h,N,dh)
        v = self.v(KV).view(B, N, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)          # (B,h,M,N)
        if kv_mask is not None:
            scores = scores.masked_fill(kv_mask[:, None, None, :], -1e4)
        A = torch.softmax(scores, dim=2)                                 # softmax over INDUCING (M) per peak
        if kv_mask is not None:
            A = A.masked_fill(kv_mask[:, None, None, :], 0.0)            # padded peaks contribute 0 mass
        # scaled weighted SUM over peaks (N): preserve assignment mass, normalize by sqrt(N_valid) for
        # set-size stability, then a learned scale lets the encoder calibrate the magnitude.
        if kv_mask is not None:
            nval = (~kv_mask).sum(1).clamp(min=1).to(v.dtype)           # (B,)
        else:
            nval = torch.full((B,), float(N), device=Q.device, dtype=v.dtype)
        denom = nval.sqrt().view(B, 1, 1, 1)
        agg = (self.drop(A) @ v) * (self.scale / denom)                 # (B,h,M,dh)
        out = agg.transpose(1, 2).reshape(B, M, -1)
        H = Q + self.o(out)                                             # clean residual
        return H + self.ff(self.norm_h(H))


class MassISABpp(nn.Module):
    """ISAB++ whose inducing-point COMPRESSION uses mass-preserving MassMABpp (anti major-washing,
    Increment 7); the X-attends-back step stays standard MABpp so per-peak outputs keep full context."""
    def __init__(self, d_model: int, n_heads: int, m_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, m_inducing, d_model)); nn.init.xavier_uniform_(self.I)
        self.mab0 = MassMABpp(d_model, n_heads, dropout)            # mass-preserving inducing compression
        self.mab1 = MABpp(d_model, n_heads, dropout, norm_q=True)   # X attends back (standard)

    def forward(self, X, pad_mask=None):
        I = self.I.expand(X.size(0), -1, -1)
        H = self.mab0(I, X, kv_mask=pad_mask)                       # (B,m,d) mass-preserving
        return self.mab1(X, H, q_mask=pad_mask, kv_mask=None)       # (B,N,d)


class LocalTokenEncoder(nn.Module):
    """
    Position-wise token encoder — NO cross-token attention.

    Each allele token is transformed independently (like a per-allele MLP),
    so a token's representation never mixes with other tokens in the set.
    This preserves per-allele identity (the property that lets independent
    per-donor classifiers — e.g. LR on explicit allele bins — generalize to
    novel donor combos) while still adding learnable depth, unlike raw tokens.

    Contrast with ISAB, which globally context-mixes every token against the
    whole set (strong, but entangles co-occurring donors → interpolates novel
    combos toward seen ones).
    """

    def __init__(self, d_model: int, n_blocks: int = 2, dropout: float = 0.0):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.ModuleDict({
                "ff": nn.Sequential(
                    nn.Linear(d_model, 4 * d_model),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout),
                    nn.Linear(4 * d_model, d_model),
                ),
                "norm": nn.LayerNorm(d_model),
            }))

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, N, d) — purely position-wise, pad_mask unused (no cross-token op)
        for blk in self.blocks:
            x = blk["norm"](x + blk["ff"](x))
        return x


class DSMILDecoder(nn.Module):
    """Dual-Stream MIL decoder (Li et al., CVPR 2021, arXiv 2011.08939), multi-donor adaptation.
    Inc12: directly fixes the loud-vs-faint aggregation split that the single sparsemax decoder gets wrong —
      stream 1 (instance/MAX): a per-peak donor classifier; the CRITICAL peak (argmax) handles the LOUD donor.
      stream 2 (bag/ATTENTION): every peak attends (softmax over peaks) to donor c's critical-peak embedding,
                                weighted-sum → bag rep → bag score; this DENSE aggregation catches the faint
                                donor's DIFFUSE evidence.
      logit_c = 0.5 * (max_peak instance_score_c + bag_score_c).
    Reads the encoded set H (decoder_source); independent per donor (no donor<->donor attention) => combo-safe."""

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.inst = nn.Linear(d_model, n_classes)             # per-peak instance scores (B,N,C)
        self.q = nn.Linear(d_model, d_model)                  # shared query/key transform (DSMIL: critical vs others)
        self.v = nn.Linear(d_model, d_model)
        self.bag_w = nn.Parameter(torch.empty(n_classes, d_model)); nn.init.xavier_uniform_(self.bag_w)
        self.bag_b = nn.Parameter(torch.zeros(n_classes))
        self.drop = nn.Dropout(dropout)
        self.scale = d_model ** -0.5

    def forward(self, H: torch.Tensor, pad_mask: torch.Tensor | None = None,
                geno_query: torch.Tensor | None = None) -> torch.Tensor:
        B, N, d = H.shape; C = self.inst.out_features
        S = self.inst(H)                                      # (B,N,C) instance scores
        if pad_mask is not None:
            S = S.masked_fill(pad_mask[:, :, None], -1e4)
        max_inst, crit_idx = S.max(dim=1)                    # (B,C) critical-peak score + index per donor
        crit = torch.gather(H, 1, crit_idx.unsqueeze(-1).expand(-1, -1, d))   # (B,C,d) critical embeddings
        Qc = self.q(crit)                                    # (B,C,d)
        Kp = self.q(H)                                       # (B,N,d) same transform
        att = (Qc @ Kp.transpose(1, 2)) * self.scale         # (B,C,N) each donor vs all peaks
        if pad_mask is not None:
            att = att.masked_fill(pad_mask[:, None, :], -1e4)
        att = torch.softmax(att, dim=-1)                     # DENSE over peaks → catches diffuse evidence
        bag = self.drop(att) @ self.v(H)                     # (B,C,d) bag rep per donor
        bag_score = torch.einsum("bnd,nd->bn", bag, self.bag_w) + self.bag_b   # (B,C)
        return 0.5 * (max_inst + bag_score)


class AdditiveDonorDecoder(nn.Module):
    """
    Additive per-donor readout — NO softmax competition over tokens.

    Each token casts an independent additive 'vote' for every donor; votes are
    SUMMED over valid tokens (not softmax-attended). This mirrors how linear /
    tree models (LR, XGBoost) score donors: every diagnostic allele contributes
    additively to its donor's logit, regardless of what other alleles are present.

    Motivation: softmax cross-attention (PerDonorDecoder) normalizes attention
    weights across tokens, so a MINORITY contributor whose few unique alleles are
    overshadowed by a major contributor's many alleles gets suppressed (observed:
    donor 48 in test combo (46,47,48) → prob 0.0-0.13 across all attention variants,
    while LR keeps 48 and reaches NOC3 EM 0.750). Additive readout removes that
    competition: 48's alleles add up independently.

      evidence = Linear(H)             (B, N, 45)   per-token, per-donor vote
      logit_d  = bias_d + Σ_i m_i · evidence[:, i, d]   (sum over valid tokens)
    """

    def __init__(self, d_model: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.vote = nn.Linear(d_model, n_classes)
        self.bias = nn.Parameter(torch.zeros(n_classes))
        # learnable log-scale to temper sum magnitude (set-size robustness)
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, H: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        evidence = self.vote(self.dropout(H))            # (B, N, 45)
        if pad_mask is not None:
            valid = (~pad_mask).unsqueeze(-1).to(evidence.dtype)  # (B, N, 1)
            evidence = evidence * valid
        logits = evidence.sum(dim=1) * torch.exp(self.log_scale) + self.bias  # (B, 45)
        return logits


class PerDonorDecoder(nn.Module):
    """
    Per-donor cross-attention decoder for compositional contributor ID.

    Each of `n_classes` donors has its own learnable query vector. The query
    attends INDEPENDENTLY to the encoded token set, then is scored by its own
    per-donor weight. This gives the per-donor inductive bias of independent
    binary classifiers (which generalize to novel combos) while retaining
    attention over the allele set.

      Q  = donor_queries           (B, 45, d)
      D  = MAB(Q, H)               (B, 45, d)   each donor attends to set H
      logit_i = <D_i, w_i> + b_i   (B, 45)      independent per-donor scoring
    """

    def __init__(self, d_model: int, n_heads: int, n_classes: int, dropout: float = 0.0,
                 n_layers: int = 2, sparse: bool = False, vib: bool = False, n_sink: int = 0,
                 dec_aggr: str = "sparsemax", phi_inject: bool = False):
        super().__init__()
        # Inc18 V1: FiLM-style phi injection — project log(phi_c) scalar into per-donor query
        # (zero-init => identical to base at start; privileged GT phi at train, EM phi at inference)
        self.phi_inject = phi_inject
        if phi_inject:
            self.phi_proj = nn.Linear(1, d_model)
            nn.init.zeros_(self.phi_proj.weight)
            nn.init.zeros_(self.phi_proj.bias)
        self.donor_queries = nn.Parameter(torch.empty(1, n_classes, d_model))
        nn.init.xavier_uniform_(self.donor_queries)
        # Inc9 B4 (softmax-1 / "quiet attention"): n_sink learnable null KEYS appended to the set so a
        # donor query can attend to NOTHING instead of being forced to normalize mass onto dominant peaks
        # (zero-init = literal softmax-1 at start). Lets faint minors keep their own peak's attention.
        self.n_sink = int(n_sink)
        if self.n_sink > 0:
            self.sink = nn.Parameter(torch.zeros(1, self.n_sink, d_model))
        # Inc6 2f: variational information bottleneck (Alemi 2017) on EACH donor's latent D_i —
        # reparameterize D_i ~ N(mu_i, sigma_i) + KL(.||N(0,I)) so the per-donor code keeps only the
        # information needed to decide donor i and DROPS combo-specific nuisance (the binding/combo-
        # entanglement the F29 wall is made of). Independent per donor => "disentangled donor latents".
        self.vib = vib
        if vib:
            self.to_mu = nn.Linear(d_model, d_model)
            self.to_logvar = nn.Linear(d_model, d_model)
        self.kl = None
        # L stacked cross-attention layers (Query2Label uses L=2). NO self-attention among
        # donor queries — deliberate: donor<->donor attention would model combo co-occurrence
        # (memorization on novel combos); ML-Decoder also drops query self-attention.
        # sparse=True (doc2 §7): sparsemax cross-attention → each donor attends to ONLY its alleles.
        # Inc12: dec_aggr in {sparsemax, entmax15, entmax_temp} softens the (hard) sparsemax so a faint
        # donor's DIFFUSE evidence is not dropped (the +.10 under-read vs the soft-vote ceiling).
        if sparse:
            self.layers = nn.ModuleList(
                [SparseMAB(d_model, n_heads, dropout, aggr=dec_aggr) for _ in range(n_layers)])
        else:
            self.layers = nn.ModuleList([MAB(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.score_w = nn.Parameter(torch.empty(n_classes, d_model))
        nn.init.xavier_uniform_(self.score_w)
        self.score_b = nn.Parameter(torch.zeros(n_classes))

    def forward(self, H: torch.Tensor, pad_mask: torch.Tensor | None = None,
                geno_query: torch.Tensor | None = None,
                phi_query: torch.Tensor | None = None) -> torch.Tensor:
        q = self.donor_queries
        if geno_query is not None:                             # Inc3 A: anchor query on reference genotype
            # (45,d) reference anchor (broadcast over batch) OR (B,45,d) per-sample anchor (Inc17 MLC prototype)
            q = q + (geno_query if geno_query.dim() == 3 else geno_query.unsqueeze(0))
        if self.n_sink > 0:                                    # Inc9 B4: append null sink keys (softmax-1)
            H = torch.cat([H, self.sink.expand(H.size(0), -1, -1)], dim=1)
            if pad_mask is not None:
                pad_mask = torch.cat(
                    [pad_mask, torch.zeros(H.size(0), self.n_sink, dtype=torch.bool, device=H.device)], dim=1)
        D = q.expand(H.size(0), -1, -1)                        # (B, 45, d)
        if phi_query is not None and self.phi_inject:          # Inc18 V1: inject log(phi) into donor queries
            log_phi = torch.log(phi_query.clamp(min=1e-6)).unsqueeze(-1)  # (B, 45, 1)
            D = D + self.phi_proj(log_phi)                     # (B, 45, d); zero-init → base at start
        for mab in self.layers:                                # queries cross-attend the set, L times
            D = mab(D, H, key_padding_mask=pad_mask)
        D = self.dropout(D)
        self.last_reps = D                                     # Inc9 B1/B2: expose per-donor reps
        if self.vib:                                           # Inc6 2f: bottleneck each donor latent
            mu = self.to_mu(D); logvar = self.to_logvar(D).clamp(-8, 8)
            if self.training:
                D = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            else:
                D = mu
            # KL(N(mu,sigma)||N(0,I)) averaged over batch & donors (read by the train loop)
            self.kl = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()
        return self.decode_score(D)                            # (B, 45)

    def decode_score(self, D: torch.Tensor) -> torch.Tensor:
        """Independent per-donor scoring of reps D (B,45,d). Exposed so Inc9 B2 (IFM) can re-score an
        adversarially-perturbed copy of last_reps without re-running the cross-attention."""
        return torch.einsum("bnd,nd->bn", D, self.score_w) + self.score_b


class CardinalityHead(nn.Module):
    """
    Cardinality-aware NOC estimator (Cortes/Mohri 2024) — replaces the broken
    embedding noc_head. Reads the SORTED donor-probability PROFILE (not z_mix),
    so it dodges the pooling-can't-count bottleneck and is combo-invariant
    (the *shape* of the prob distribution — how many high probs — generalizes
    val->test where the z_mix noc-head's bias did not).

      feat = [sorted_top_m(probs), sum(probs), #(probs>=0.5)]
      logits_card = MLP(feat)  -> scores over NOC k=1..n_card
    Trained cost-sensitive to pick the k minimizing top-k set-mismatch + lambda*k.
    Inference: k* = argmax(logits_card)+1, decode top-k* of donor probs.
    """
    def __init__(self, n_classes: int = 45, top_m: int = 8, n_card: int = 5, dropout: float = 0.0):
        super().__init__()
        self.top_m = top_m
        self.net = nn.Sequential(
            nn.Linear(top_m + 2, 64), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(64, n_card),
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        # probs: (B, n_classes) sigmoid probabilities (detached upstream)
        s = torch.sort(probs, dim=1, descending=True).values[:, :self.top_m]
        feat = torch.cat([s, probs.sum(1, keepdim=True),
                          (probs >= 0.5).float().sum(1, keepdim=True)], dim=1)
        return self.net(feat)                              # (B, n_card)


# ── Set-of-Set decoder (Inc19) ──────────────────────────────────────────────

class SetOfSetDecoder(nn.Module):
    """
    Two-level set-of-set decoder.

    Level-1 (private): each donor attends ONLY to peaks where that donor is the
                       SOLE carrier in the 45-panel (n_car==1). Unambiguous signal.
    Level-2 (shared):  each donor attends to peaks shared among multiple carriers
                       (n_car>1), with query enriched by the complement of other
                       donors' private evidence — accounting for what others explain.

    ctx_proj is zero-initialised: at init ctx=0 → level-2 reduces to unconstrained
    cross-attention (graceful fallback before ctx_proj learns anything).
    Requires owner_lut to be passed from the parent SetTransformerMixture.
    """

    def __init__(self, d_model: int, n_heads: int, n_classes: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_h = d_model // n_heads
        self.n_classes = n_classes

        self.donor_queries = nn.Parameter(torch.empty(1, n_classes, d_model))
        nn.init.xavier_uniform_(self.donor_queries)

        # Level-1: private peaks
        self.W_q_priv = nn.Linear(d_model, d_model, bias=False)
        self.W_k_priv = nn.Linear(d_model, d_model, bias=False)
        self.W_v_priv = nn.Linear(d_model, d_model, bias=False)
        self.W_o_priv = nn.Linear(d_model, d_model)
        self.norm_priv = nn.LayerNorm(d_model)

        # Context: compress other donors' private reps → shared query delta
        self.ctx_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.ctx_proj.weight)
        nn.init.zeros_(self.ctx_proj.bias)

        # Level-2: shared peaks
        self.W_q_shar = nn.Linear(d_model, d_model, bias=False)
        self.W_k_shar = nn.Linear(d_model, d_model, bias=False)
        self.W_v_shar = nn.Linear(d_model, d_model, bias=False)
        self.W_o_shar = nn.Linear(d_model, d_model)
        self.norm_shar = nn.LayerNorm(d_model)

        # Per-donor scoring (identical interface to PerDonorDecoder)
        self.score_w = nn.Parameter(torch.empty(n_classes, d_model))
        nn.init.xavier_uniform_(self.score_w)
        self.score_b = nn.Parameter(torch.zeros(n_classes))
        self.dropout = nn.Dropout(dropout)
        self.last_reps: torch.Tensor | None = None

    def _sdpa(self, Wq, Wk, Wv, Wo, q, Haug, bias):
        """Per-donor masked SDPA.
        q    : (B, C, d)      Haug : (B, N+1, d)
        bias : (B, C, N+1) additive (0=attend, -1e9=block) → (B, C, d)
        """
        B, C, d = q.shape
        N1 = Haug.size(1)
        n_h, d_h = self.n_heads, self.d_h
        Q = Wq(q).reshape(B, C, n_h, d_h).permute(0, 2, 1, 3)        # (B,n_h,C,d_h)
        K = Wk(Haug).reshape(B, N1, n_h, d_h).permute(0, 2, 1, 3)    # (B,n_h,N+1,d_h)
        V = Wv(Haug).reshape(B, N1, n_h, d_h).permute(0, 2, 1, 3)    # (B,n_h,N+1,d_h)
        mask = bias.unsqueeze(1).expand(B, n_h, C, N1)                # (B,n_h,C,N+1)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask.to(Q.dtype))
        return Wo(out.permute(0, 2, 1, 3).reshape(B, C, d))           # (B,C,d)

    def forward(self, H: torch.Tensor, pad_mask: torch.Tensor | None = None,
                owner_lut: torch.Tensor | None = None,
                tokens: torch.Tensor | None = None, aoff: int = 30) -> torch.Tensor:
        B, N, d = H.shape
        C = self.n_classes

        # ── Carrier masks ────────────────────────────────────────────────────
        if owner_lut is not None and tokens is not None:
            li = tokens[..., 0].long().clamp(0, 23)
            bi = (tokens[..., 1] * 10).round().long() + aoff
            bi = bi.clamp(0, owner_lut.size(1) - 1)
            cmask = owner_lut[li, bi]                        # (B, N, C)
            n_car = cmask.sum(-1)                            # (B, N)
        else:
            cmask = torch.ones(B, N, C, device=H.device)
            n_car = torch.full((B, N), float(C), device=H.device)

        is_carrier = cmask.permute(0, 2, 1).bool()          # (B, C, N)
        is_pad = (pad_mask.unsqueeze(1).expand(B, C, N)
                  if pad_mask is not None
                  else torch.zeros(B, C, N, dtype=torch.bool, device=H.device))
        nc = n_car.unsqueeze(1)                              # (B, 1, N)
        priv_valid = is_carrier & (nc == 1) & ~is_pad       # (B, C, N)
        shar_valid = is_carrier & (nc  > 1) & ~is_pad       # (B, C, N)

        # ── Null key (zero vector) avoids all-inf softmax for donors with 0 peaks ──
        null_k = torch.zeros(B, 1, d, device=H.device)
        Haug   = torch.cat([H, null_k], dim=1)              # (B, N+1, d)
        null_col = torch.ones(B, C, 1, dtype=torch.bool, device=H.device)

        INF = 1e9
        def _bias(valid: torch.Tensor) -> torch.Tensor:     # (B,C,N) → (B,C,N+1) float
            v_aug = torch.cat([valid, null_col], dim=2)
            return torch.where(v_aug,
                               H.new_zeros(1),
                               H.new_full((1,), -INF))

        priv_bias = _bias(priv_valid)                        # (B, C, N+1)
        shar_bias = _bias(shar_valid)                        # (B, C, N+1)

        # ── Level-1: private peaks ────────────────────────────────────────────
        q = self.donor_queries.expand(B, -1, -1)            # (B, C, d)
        priv_delta = self._sdpa(self.W_q_priv, self.W_k_priv, self.W_v_priv,
                                self.W_o_priv, q, Haug, priv_bias)
        priv_rep = self.norm_priv(q + priv_delta)           # (B, C, d)

        # ── Context: complement of other donors' private evidence ─────────────
        # priv_mean - priv_rep/C ≈ mean of OTHER donors' private reps (residual complement)
        priv_mean = priv_rep.mean(dim=1, keepdim=True)      # (B, 1, d)
        ctx = self.ctx_proj(priv_mean.expand(B, C, d) - priv_rep / C)  # (B, C, d)

        # ── Level-2: shared peaks with context-informed query ──────────────────
        q_shar = q + ctx                                     # (B, C, d)
        shar_delta = self._sdpa(self.W_q_shar, self.W_k_shar, self.W_v_shar,
                                self.W_o_shar, q_shar, Haug, shar_bias)
        shar_rep = self.norm_shar(q_shar + shar_delta)      # (B, C, d)

        # ── Combine + score ───────────────────────────────────────────────────
        D = self.dropout(priv_rep + shar_rep)
        self.last_reps = D.detach()
        return torch.einsum("bcd,cd->bc", D, self.score_w) + self.score_b  # (B, C)


# ── Sinkhorn OT (MESH ICML2023) ────────────────────────────────────────────

def sinkhorn_log(
    affinity: torch.Tensor,   # (B, K, N)  dot-product affinities (higher = more compatible)
    pad_mask: torch.Tensor,   # (B, N)     True = ignore (padding)
    eps: float = 0.05,        # regularization: lower → sharper / more OT-like
    n_iters: int = 5,         # Sinkhorn iterations
) -> torch.Tensor:
    """
    Log-domain Sinkhorn normalisation: replaces competitive softmax in slot attention
    with a doubly-normalised (approx. doubly-stochastic) assignment.
    Each shared peak distributes its evidence fractionally across multiple slots
    instead of being winner-take-all → addresses explaining-away.
    Returns A (B, K, N): each slot's distribution over peaks (rows sum ≈ 1 after softmax).
    """
    la = affinity / eps
    la = la.masked_fill(pad_mask.unsqueeze(1), -1e9)
    for _ in range(n_iters):
        la = la - torch.logsumexp(la, dim=1, keepdim=True)          # column-norm (over K slots per peak)
        la = la.masked_fill(pad_mask.unsqueeze(1), -1e9)             # re-mask after col-step
        la = la - torch.logsumexp(la, dim=2, keepdim=True)          # row-norm (over N peaks per slot)
        la = la.masked_fill(pad_mask.unsqueeze(1), -1e9)             # re-mask after row-step
    return torch.exp(la)                                             # (B, K, N); row-norm done by loop


# ── Adaptive Slot Attention decoder ────────────────────────────────────────
# Papers:  CoSA   (ICLR2024)   — genotype-conditioned slot initialisation
#          GSANet (CVPR2024)   — attr-soft guided init refinement
#          MESH   (ICML2023)   — Sinkhorn OT replaces competitive softmax in loop
#          AdaSlot(CVPR2024)   — Gumbel-Sigmoid gate for adaptive NOC

class AdaptiveSlotDecoder(nn.Module):
    """
    Iterative slot decoder for DNA mixture contributor identification.

    Slots = panel donors (K = n_classes = 45).
    Flow:
      1. CoSA:    slots ← _encode_geno() broadcast to batch  (external-signal init)
      2. GSANet:  slots += gated(attr_weighted_aggregation(H))  (attr-guided refinement)
      3. MESH:    n_iters × [Sinkhorn(Q·K^T/√d) → weighted-V → GRU → FFN]
      4. AdaSlot: gate = Gumbel-Sigmoid(gate_head(slots))  (soft NOC, concrete Bernoulli)
      5. Output:  cls_logit = cls_head(slots); card from gate profile
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int = 45,
        n_noc: int = 5,             # NOC output classes (1..5)
        n_iters: int = 3,           # MESH slot-attention iterations
        ot_eps: float = 0.05,       # Sinkhorn ε (lower = sharper assignment)
        ot_iters: int = 5,          # Sinkhorn normalization steps per iter
        gumbel_temp: float = 1.0,   # AdaSlot concrete-Bernoulli temperature
        dropout: float = 0.1,
        gate_mass: bool = False,    # §3.2.1: feed grad-enabled slot-mass into the existence gate
    ):
        super().__init__()
        self.n_iters = n_iters
        self.ot_eps = ot_eps
        self.ot_iters = ot_iters
        self.gumbel_temp = gumbel_temp
        self.scale = math.sqrt(d_model)

        # ── MESH: slot attention projections (shared across iters) ─────────
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Slot update: GRU + FFN (Locatello SA style)
        self.gru = nn.GRUCell(d_model, d_model)
        self.slot_norm1 = nn.LayerNorm(d_model)
        self.slot_norm2 = nn.LayerNorm(d_model)
        self.slot_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        # ── GSANet: attr-guided init refinement ────────────────────────────
        # gsanet_proj maps the attr-weighted peak aggregate to slot space;
        # gsanet_gate is a per-slot learned gate on how much to trust this signal.
        self.gsanet_proj = nn.Linear(d_model, d_model)
        self.gsanet_gate = nn.Linear(d_model, 1)

        # ── AdaSlot (CVPR2024): separate gate_head for slot existence ─────────
        # gate_head: "does this slot have an active contributor?" (binary existence)
        # cls_head:  "if active, which donor?" (identity content)
        # In logit space: logits_cls = cls_raw + gate_logit  → inactive slots (gate→-∞)
        # produce very negative cls logits (suppressed from prediction).
        # Count CE flows to gate_head without detach → gate_head learns slot existence.
        # Both heads get gradient from the multi-label CLS LOSS (ASL when --loss asl, else BCE) applied to
        # logits_cls = cls_raw + gate_logit. NOTE: the trainer variable is named `bce_cls` but holds an
        # AsymmetricLoss under --loss asl — the objective here is ASL(γ_neg=4), not plain BCE.
        self.gate_head = nn.Linear(d_model, 1)                              # AdaSlot slot-existence
        self.cls_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))  # content

        # §3.2.1 MASS-AWARE gate (gate_mass; default off → no-op): feed a grad-enabled per-slot mass into
        # the existence gate so "a slot is a real contributor iff it explains real peak mass" (the SUM
        # counting provably needs — DeepSets/GIN). ZERO-INIT (ReZero, Bachlechner 2020) → exact no-op at
        # start (bit-identical warm-start), trains in only if the mass signal helps the gate/count.
        self.use_gate_mass = bool(gate_mass)
        if self.use_gate_mass:
            self.mass_gate = nn.Linear(1, 1)
            nn.init.zeros_(self.mass_gate.weight); nn.init.zeros_(self.mass_gate.bias)

        # ── NOC head from gate profile ──────────────────────────────────────
        # Takes the K-dim gate vector (soft NOC indicator) → n_noc logits.
        # No .detach() on gate → count CE trains gate_head (AdaSlot design).
        self.noc_head = nn.Linear(n_classes, n_noc)

    def forward(
        self,
        H: torch.Tensor,                            # (B, N, d) encoded peaks
        pad_mask: torch.Tensor,                     # (B, N) True = padding
        geno_slots: torch.Tensor,                   # (B, K, d) CoSA init
        attr_logits: torch.Tensor | None = None,   # (B, N, K[+1]) attr logits incl. optional background col (GSANet)
        peak_w: torch.Tensor | None = None,         # (B, N) soft reliability ∈[floor,1]; down-weights phantom peaks
    ) -> dict:
        B, N, d = H.shape
        K = geno_slots.shape[1]

        # ── 1. CoSA: genotype-conditioned slot init ─────────────────────────
        slots = geno_slots.clone()                  # (B, K, d)

        # ── 2. GSANet: attr-weighted peak aggregate → refine init ───────────
        if attr_logits is not None:
            # Softmax over ALL attr classes INCLUDING the background/none column (when present), then
            # keep only the K donor weights. A peak the attr head reads as background sends its mass to
            # the dropped background column, so artifact/stutter peaks no longer force themselves onto a
            # donor slot (which injected false evidence into the slot init). Degrades to the plain
            # donor-softmax when attr_logits already has exactly K columns.
            attr_w = torch.softmax(attr_logits, dim=-1)                # (B, N, K[+1])
            attr_w = attr_w[..., :K]                                   # (B, N, K) donors; background dropped
            attr_w = attr_w * (~pad_mask).float().unsqueeze(-1)        # zero padding
            if peak_w is not None:
                attr_w = attr_w * peak_w.unsqueeze(-1)                 # soft: phantom peaks contribute less to slot init
            attr_agg = attr_w.transpose(1, 2) @ H                     # (B, K, d)
            attr_agg = self.gsanet_proj(attr_agg)                     # (B, K, d)
            g = torch.sigmoid(self.gsanet_gate(attr_agg))             # (B, K, 1)
            slots = slots + g * attr_agg                               # gated residual

        # ── 3. MESH: iterative Sinkhorn OT routing ──────────────────────────
        K_h = self.k_proj(H)                        # (B, N, d) — fixed across iters
        V_h = self.v_proj(H)                        # (B, N, d)
        if peak_w is not None:
            V_h = V_h * peak_w.unsqueeze(-1)        # soft: a phantom peak contributes little VALUE to any slot
                                                    # (it can still be attended via affinity, but carries ~no content)

        for _ in range(self.n_iters):
            Q = self.q_proj(self.slot_norm1(slots)) # (B, K, d)
            affinity = (Q @ K_h.transpose(1, 2)) / self.scale         # (B, K, N)

            # Sinkhorn OT: fractional peak→slot assignment (MESH)
            A = sinkhorn_log(affinity, pad_mask,
                             eps=self.ot_eps, n_iters=self.ot_iters)   # (B, K, N)

            updates = A @ V_h                                           # (B, K, d)

            # GRU state update (Locatello-style SA)
            slots = self.gru(
                updates.reshape(B * K, d),
                slots.reshape(B * K, d),
            ).reshape(B, K, d)

            slots = slots + self.slot_ffn(self.slot_norm2(slots))      # FFN residual

        # Mass-preserving count signal (Fischer & Gärtner 2024, arXiv 2407.04170; GIN/DeepSets): the MESH
        # row-normalization (weighted mean) ERASES per-slot assignment mass, which a multiset counter
        # provably needs (sum is injective on multisets; mean/max are not). Recover it as the expected
        # #peaks assigned to each slot = column-softmax over slots, summed over peaks. Detached (read by
        # the count head only; never perturbs the slots).
        with torch.no_grad():
            aff_f = (self.q_proj(self.slot_norm1(slots)) @ K_h.transpose(1, 2)) / self.scale  # (B,K,N)
            aff_f = aff_f.masked_fill(pad_mask.unsqueeze(1), -1e9)
            p2s = torch.softmax(aff_f, dim=1)                          # peak→slot over K slots
            pw = (~pad_mask).unsqueeze(1).to(p2s.dtype)
            if peak_w is not None:
                pw = pw * peak_w.unsqueeze(1).to(p2s.dtype)            # phantom peaks don't inflate the count signal
            slot_mass = (p2s * pw).sum(-1)                             # (B, K)

        # ── 4. AdaSlot (CVPR2024): separate gate_head, concrete Bernoulli ──────
        gate_logit = self.gate_head(slots).squeeze(-1)                 # (B, K) slot existence
        if self.use_gate_mass:
            # §3.2.1: recompute the per-slot mass WITH gradient (the block above is detached for the count
            # head) and add a ReZero-gated log1p-mass term → the gate sees "does this slot explain real
            # peak mass?". Zero-init mass_gate → no-op at start; gradient flows gate→slots→MESH→encoder.
            aff_g = (self.q_proj(self.slot_norm1(slots)) @ K_h.transpose(1, 2)) / self.scale  # (B,K,N)
            aff_g = aff_g.masked_fill(pad_mask.unsqueeze(1), -1e9)
            p2s_g = torch.softmax(aff_g, dim=1)
            pw_g = (~pad_mask).unsqueeze(1).to(p2s_g.dtype)
            if peak_w is not None:
                pw_g = pw_g * peak_w.unsqueeze(1).to(p2s_g.dtype)
            mass_g = (p2s_g * pw_g).sum(-1)                             # (B, K) grad-enabled slot mass
            gate_logit = gate_logit + self.mass_gate(torch.log1p(mass_g).unsqueeze(-1)).squeeze(-1)
        if self.training:
            # Binary-Concrete / Gumbel-Sigmoid (Maddison et al., ICLR 2017): the additive noise must be
            # LOGISTIC, L = log(u) - log(1-u) (= the difference of two Gumbels), which is zero-mean and
            # symmetric. A single Gumbel(0,1) has mean +0.5772 and is right-skewed → it biases EVERY gate
            # toward "active", inflating the slot-existence signal the NOC head reads (over-counts donors).
            u = torch.rand_like(gate_logit).clamp(1e-7, 1 - 1e-7)
            logistic = torch.log(u) - torch.log1p(-u)
            gate = torch.sigmoid((gate_logit + logistic) / self.gumbel_temp)
        else:
            gate = torch.sigmoid(gate_logit)

        # ── 5. Cls logit: content + existence in logit space ─────────────────
        # logits_cls = cls_head(slots) + gate_logit:
        #   • inactive slot (gate_logit→−∞): cls logit suppressed → donor excluded ✓
        #   • active slot (gate_logit→0): cls_head dominates → donor identity ✓
        # Both cls_head AND gate_head receive gradient from the cls loss (ASL when --loss asl) via the additive path.
        cls_raw    = self.cls_head(slots).squeeze(-1)                  # (B, K) identity
        logits_cls = cls_raw + gate_logit                              # (B, K) combined

        # NOC head: count CE trains gate_head (no .detach() — AdaSlot design).
        # gate_head thus receives gradient from BOTH the cls loss (ASL/BCE, via logits_cls) AND count CE.
        logits_card = self.noc_head(gate)                              # (B, n_noc)

        return {
            "logits_cls":  logits_cls,
            "logits_card": logits_card,
            "gate":        gate,
            "gate_logit":  gate_logit,
            "attn":        A,                                          # (B, K, N) for inspection
            "slot_mass":   slot_mass,                                  # (B, K) mass-preserving count signal
        }


# ── EBM-SPEN Joint decoder (Belanger & McCallum 2016 / Tu & Gimpel 2018) ────

class SPENDecoder(nn.Module):
    """
    Joint Energy-Based Structured Prediction Energy Network.

    Energy (all terms differentiable w.r.t. y ∈ [0,1]^K):
      E(x, y) = -local_logits · y                 (local: dot unary, from encoder)
               + global_net(y)                     (global: set-level MLP, captures cardinality/co-occurance)
               + joint_sc( x * tanh(y_proj(y)) )  (joint: x-y cross term, conditions on evidence)

    Inference: T steps projected gradient descent in y-space
      y^(t+1) = clamp( y^(t) - lr * ∇_y E(x, y^(t)), 0, 1 )
      Warm start: y^(0) = σ(local_logits)

    Training (Tu & Gimpel 2018 end-to-end, full BPTT):
      - ALL T steps with create_graph=True → gradient flows through the full GD trajectory
      - each step: g_t = ∂E/∂y_t (create_graph), y_{t+1} = clamp(y_t - lr·g_t, 0, 1)
      - x_feat and local_logits are detached before entry (energy params only, not encoder)
      - global + joint nets zero-init → pure local model at initialization
    """

    def __init__(
        self,
        d_model: int,
        n_classes: int = 45,
        n_noc: int = 5,
        n_inf_steps: int = 10,
        inf_lr: float = 0.5,
        d_global: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        K = n_classes
        self.n_inf_steps = n_inf_steps
        self.inf_lr      = inf_lr
        self.n_classes   = n_classes

        # ── Global energy: set-level structural MLP ──────────────────────────
        self.global_net = nn.Sequential(
            nn.Linear(K, d_global), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_global, d_global // 2), nn.GELU(),
            nn.Linear(d_global // 2, 1),
        )
        # ── Joint energy: x-representation gated by y ────────────────────────
        self.y_proj   = nn.Linear(K, d_model, bias=False)
        self.joint_ln = nn.LayerNorm(d_model)
        self.joint_sc = nn.Linear(d_model, 1, bias=False)

        # Zero-init the OUTPUT layers only → energy = 0 at init, gradients still flow.
        # Intermediate layers use default init so hidden activations are non-zero.
        nn.init.zeros_(self.global_net[-1].weight)
        nn.init.zeros_(self.global_net[-1].bias)
        nn.init.zeros_(self.joint_sc.weight)

    def _energy(
        self,
        x_d:     torch.Tensor,   # (B, d)  pooled encoder — DETACHED
        lg_d:    torch.Tensor,   # (B, K)  local logits  — DETACHED
        y:       torch.Tensor,   # (B, K)  ∈ [0,1]; may require_grad
    ) -> torch.Tensor:           # (B,)   scalar energy per sample
        E_loc   = -(lg_d * y).sum(-1)                              # local unary
        E_glob  = self.global_net(y).squeeze(-1)                   # set-level
        y_e     = torch.tanh(self.y_proj(y))                       # (B, d)
        E_joint = self.joint_sc(self.joint_ln(x_d * y_e)).squeeze(-1)  # x-y cross
        return E_loc + E_glob + E_joint

    @torch.no_grad()
    def _infer_detached(
        self,
        x_d:  torch.Tensor,
        lg_d: torch.Tensor,
        n:    int,
        y_init: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """T steps projected GD without graph (eval + warm-up)."""
        y = y_init if y_init is not None else torch.sigmoid(lg_d)
        for _ in range(n):
            with torch.enable_grad():
                y_v = y.requires_grad_(True)
                g = torch.autograd.grad(self._energy(x_d, lg_d, y_v).sum(), y_v)[0]
            y = (y - self.inf_lr * g.detach()).clamp(0, 1)
        return y

    def forward(
        self,
        x_feat:       torch.Tensor,   # (B, d) pooled encoder
        local_logits: torch.Tensor,   # (B, K) initial logits from local head
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (y_T, y_T_in_graph):
          y_T           – final (B, K) soft predictions, detached (for cardinality head)
          y_T_in_graph  – full-BPTT output in computational graph (for BCE loss)
        """
        x_d  = x_feat.detach()
        lg_d = local_logits.detach()

        if self.training and self.n_inf_steps > 0:
            # Full BPTT (Tu & Gimpel 2018): all T steps with create_graph=True
            # Gradient flows from loss → y_T → g_{T-1} → y_{T-1} → ... → energy params
            y = torch.sigmoid(lg_d).detach().requires_grad_(True)
            for _ in range(self.n_inf_steps):
                g = torch.autograd.grad(
                    self._energy(x_d, lg_d, y).sum(), y, create_graph=True
                )[0]
                y = (y - self.inf_lr * g).clamp(0, 1)
            y_T_graph = y
            y_T       = y_T_graph.detach()
        else:
            y_T       = self._infer_detached(x_d, lg_d, self.n_inf_steps)
            y_T_graph = y_T

        return y_T, y_T_graph


# ── Full model ─────────────────────────────────────────────────────────────

class SetTransformerMixture(nn.Module):
    """
    Set Transformer for STR mixture contributor identification.

    Args:
        n_loci:     number of distinct loci (default 24 for GlobalFiler)
        d_locus:    locus embedding dimension
        d_model:    transformer hidden dimension
        n_heads:    attention heads (must divide d_model)
        n_isab:     number of ISAB encoder blocks (planv2: 2-3)
        m_inducing: inducing points per ISAB (planv2 doesn't specify; 32 default)
        n_classes:  closed-set donor count (45)
        n_noc:      number of NOC classes (6: counts 0-5)
        dropout:    dropout rate
    """

    def __init__(
        self,
        n_loci: int = 24,
        d_locus: int = 16,
        d_model: int = 128,
        n_heads: int = 4,
        n_isab: int = 2,
        m_inducing: int = 32,
        n_classes: int = 45,
        n_noc: int = 5,
        dropout: float = 0.1,
        cls_decoder: str = "pooled",   # "pooled" | "per_donor" | "additive" | "hybrid" | "dsmil"
        decoder_source: str = "encoded",  # "encoded" (ISAB H) | "raw" (pre-ISAB tokens) | "local" (position-wise enc)
        n_flat: int = 590,             # dim of aligned Xflat features (for hybrid)
        decouple_reject: bool = False, # separate PMA pool for reject head
        n_token_feats: int = 3,        # token fields: 3=(locus,allele,log_h); 8=enriched (Increment 1)
        encoder: str = "isab",         # "isab" (Set Transformer, LayerNorm) | "isab++" (SetNorm + clean-path)
        dec_layers: int = 2,           # per_donor decoder cross-attention layers (Query2Label L=2)
        dec_aggr: str = "sparsemax",   # Inc12: per_donor decoder key-aggregation "sparsemax"|"entmax15"|"entmax_temp"
        num_embed: str = "raw",        # "raw" (scalar→shared Linear) | "periodic" (per-feature PLR, Gorishniy 2022)
        n_freq: int = 8,               # periodic embedding: frequencies per feature
        d_num_emb: int = 8,            # periodic embedding: output dim per feature
        periodic_sigma: float = 1.0,   # periodic embedding: frequency init std N(0, σ)
        aux_heads: bool = False,       # Inc2 §5: privileged auxiliary heads (allele→donor attr + phi)
        ml_attr: bool = False,         # Inc14 B-v2: per-peak MULTI-LABEL (sigmoid) genotype co-membership head
                                       # (CPI/inclusion: a peak is owned by EVERY present donor whose genotype
                                       # has its allele — softmax attr_head structurally can't learn this; the
                                       # 75% shared alleles. Teacher for MLD distillation; ceiling probe N5 .96)
        noc_contrast: bool = False,    # Inc2 §4b-A/C: decoupled ordinal NOC contrast (Rank-N-Contrast)
        noc_detach: bool = False,      # §4b-C valve 2/3: pma_noc pools H.detach() (contrast shapes ONLY
                                       # its private discarded pool/proj, never the shared encoder)
        d_proj: int = 64,              # contrastive projection dim (discarded at inference)
        sparse_attn: bool = False,     # Inc2d §7: sparsemax per-donor decoder (zero attention to spurious peaks)
        geno_query: bool = False,      # Inc3 A: condition donor queries on reference genotype (DAB-DETR/Q2L)
        donor_geno: torch.Tensor | None = None,       # (n_classes, G, n_token_feats) reference-allele tokens
        donor_geno_mask: torch.Tensor | None = None,  # (n_classes, G) bool — valid reference alleles
        donor_contrast: bool = False,  # Inc3 B: supervised-contrastive peak grouping by donor (SupCon)
        noc_ord_head: bool = False,    # Inc3 V1-3: CORN ordinal count head on its own encoder pool
        noc_ord_detach: bool = False,  # V1: pma_card pools H.detach() (count pool decoupled from ID)
        noc_ord_replace: bool = False, # V3: count = CORN-on-encoder only (drop ID-derived card head)
        vib: bool = False,             # Inc6 2f: variational information bottleneck on per-donor latents
        mass_pool: bool = False,       # Inc7 (F31): mass-preserving inducing compression vs major-washing
        vicreg: bool = False,          # Inc8 V1 (F32): VICReg variance+covariance on per-donor reps (de-smooth)
        vicreg_inv: bool = False,      # Inc8 V2 (F32): + VICReg invariance (same donor across combos)
        attn_sink: int = 0,            # Inc9 B4: null sink keys in the per-donor decoder (softmax-1 / quiet attn)
        donor_recon: bool = False,     # Inc9 B1: per-donor source-reconstruction aux (anti-absorption, PIT-spirit)
        query_denoise: bool = False,   # Inc9 A4: DN-DETR — extra pass with NOISED geno-anchored queries, denoise to GT
        qdn_noise: float = 1.0,        # A4: stddev of the query noise (in geno-embedding space)
        ref_match: bool = False,       # Inc10: discriminability-weighted reference-allele match added to per-donor logit
        ref_match_learn: bool = False, # Inc10 V2: LEARN the per-(donor,allele) discriminability weight (AFIA); else fixed 1/panel-rarity
        nc_attn: str = "none",         # Inc11 (F35b): non-competitive sigmoid encoder attention "none"|"mab0"|"both" (Ramapuram 2024)
        nc_learnable_bias: bool = False,  # Inc11: add a learnable offset to the sigmoid-attention −log(n) bias
        sic_iters: int = 1,            # Inc16: deep-unfolded Successive Interference Cancellation iterations
        phi_inject: bool = False,      # Inc18 V1: inject EM-uniform phi (or GT phi) into per-donor decoder queries
        soft_geno_attr: bool = False,  # Inc18 V2: attr logits — private peaks hard-assigned, shared soft-masked at inference
        feas_filter: bool = False,     # Inc18 V3: remove infeasible peaks (0 carriers) before encoder (train+infer)
        set_of_set: bool = False,      # SoS: split peaks into private (n_car==1) / shared (n_car>1) BEFORE encoder;
        # ── Adaptive Slot (CoSA+GSANet+MESH+AdaSlot; cls_decoder="aslot") ──
        n_slot_iters: int = 3,         # MESH slot attention iterations
        ot_eps: float = 0.05,          # Sinkhorn ε (lower = sharper OT assignment)
        ot_iters: int = 5,             # Sinkhorn normalization steps per iter
        gumbel_temp: float = 1.0,      # AdaSlot concrete-Bernoulli temperature
        gate_mass: bool = False,       # §3.2.1: feed grad-enabled slot-mass into the AdaSlot existence gate (off = no-op)
        phi_gated: bool = False,       # §3.4: presence-gated phi + β-NLL (adds phi_logvar_head; loss masked to present donors)
        noc_head_v2: bool = False,     # paper-proof CORN count head (slot-mass + MAC + prob-profile); off = no-op
        lupi_phys: bool = False,       # Inc-LUPI: privileged PHYSICAL heads (degradation β; clean mu+var Gaussian β-NLL; drop-in) — F10 transfer lever
        em_phi_feature: bool = False,  # internalize the INDEPENDENT EM mixture-proportion (Mx) deconvolution:
                                       #   FiLM-lite log-phi -> cls logit (LOP) + Hill effective-count -> noc head
                                       # two separate encoder passes then merge — fixes explaining-away at the source
        noise_gate: bool = False,      # in-model SOFT per-peak reliability gate s_p=P(real allele) from PROVEN
                                       #   structural features (Tvedebrink height + stutter n±1 ratios; Brookes/Bright)
                                       #   supervised by the carried-by-a-true-donor label (Balding-Buckleton drop-in);
                                       #   soft-floored down-weight of phantom (stutter/drop-in) peaks in the aslot decoder
        noise_s_floor: float = 0.05,   # floor so the gate is SOFT (never a hard 0 mask) — Lipschitz-robust per design rule
        # ── EBM-SPEN Joint (cls_decoder="spen") ──────────────────────────────
        n_inf_steps: int = 10,         # projected GD steps during inference
        inf_lr: float = 0.5,           # GD step size in y-space
        spen_global_hidden: int = 128, # hidden dim of global energy MLP
        owner_lut: torch.Tensor | None = None,  # (24, LUT_W, C) carrier lookup, built from donor_geno in trainer
    ):
        super().__init__()
        self.phi_inject = phi_inject
        self.soft_geno_attr = soft_geno_attr
        self.feas_filter = feas_filter
        self.set_of_set = set_of_set
        if (soft_geno_attr or feas_filter or set_of_set or cls_decoder == "sos") and owner_lut is not None:
            self.register_buffer("owner_lut", owner_lut.float())   # (24, LUT_W, C)
            self._AOFF = 30                                         # allele bin offset (matches trainer's ALLELE_OFF)
        else:
            self.owner_lut = None
        self.vib = vib
        self.mass_pool = mass_pool
        self.donor_recon = bool(donor_recon)
        self.query_denoise = bool(query_denoise)
        self.qdn_noise = float(qdn_noise)
        # Inc8/Inc9: VICReg + donor-recon operate on the encoded token set H in the train loop; forward
        # just exposes H (and B1 the per-donor reps via decoder.last_reps). No inference-path change.
        self.return_encoded = bool(vicreg or vicreg_inv or donor_recon)

        self.n_loci = n_loci
        self.d_model = d_model
        self.n_classes = n_classes
        self.cls_decoder = cls_decoder
        self.decoder_source = decoder_source
        self.n_flat = n_flat
        self.decouple_reject = decouple_reject
        self.n_token_feats = n_token_feats

        # Locus index (int) → embedding; remaining fields are continuous (allele, log_h, +relational).
        n_num = n_token_feats - 1
        self.locus_embed = nn.Embedding(n_loci + 1, d_locus, padding_idx=n_loci)
        # Per-feature standardization (identity by default = backward compatible; the
        # training script sets these from train-split valid-peak statistics for enriched tokens).
        self.register_buffer("feat_mean", torch.zeros(n_num))
        self.register_buffer("feat_std", torch.ones(n_num))
        # Numerical embedding: "raw" feeds standardized scalars straight into the shared
        # input_proj (original behaviour); "periodic" gives EACH feature its own PLR
        # embedding first (Gorishniy 2022) — fixes the rotation-invariance bottleneck that
        # made the enriched 8-field token underperform under "raw" (Increment 1 finding).
        self.num_embed_kind = num_embed
        if num_embed == "periodic":
            self.num_embed = PeriodicNumEmbedding(n_num, d_num_emb, n_freq, periodic_sigma)
            proj_in = d_locus + self.num_embed.out_dim
        else:
            self.num_embed = None
            proj_in = d_locus + n_num
        self.input_proj = nn.Linear(proj_in, d_model)

        # Set Transformer encoder: ISAB (LayerNorm) or ISAB++ (SetNorm + clean-path, trainable deep).
        # Inc7: mass_pool swaps ISAB++ for MassISABpp (mass-preserving inducing compression) — requires ++.
        self.encoder_kind = encoder
        self.nc_attn = nc_attn
        nc0 = nc_attn in ("mab0", "both")     # Inc11: which MAB step gets non-competitive sigmoid attention
        nc1 = nc_attn == "both"
        if encoder == "isab++":
            if mass_pool:
                assert nc_attn == "none", "nc_attn and mass_pool are mutually exclusive levers"
                self.encoder = nn.ModuleList(
                    [MassISABpp(d_model, n_heads, m_inducing, dropout) for _ in range(n_isab)])
            else:
                self.encoder = nn.ModuleList(
                    [ISABpp(d_model, n_heads, m_inducing, dropout,
                            nc_mab0=nc0, nc_mab1=nc1, nc_learnable_bias=nc_learnable_bias)
                     for _ in range(n_isab)])
        else:
            assert not mass_pool, "mass_pool requires encoder=isab++ (SetNorm clean-path skeleton)"
            assert nc_attn == "none", "nc_attn requires encoder=isab++"
            self.encoder = nn.ModuleList(
                [ISAB(d_model, n_heads, m_inducing, dropout) for _ in range(n_isab)])

        # Pooling → z_mix (used by reject + noc heads, and pooled cls)
        self.pma = PMA(d_model, n_heads, k_seeds=1, dropout=dropout)
        # Optional separate pool for reject head — decouples open-set detection from
        # the donor-ID gradient (which, sharing z_mix via set_adjust, pulls z off the
        # reject task: reject AUROC 0.993 flat-only -> 0.945 hybrid).
        # Dedicated reject pool also when feas_filter is on: the reject head must pool an UNFILTERED
        # encode so out-of-panel (unknown-contributor) alleles reach it, and that pool must be separate
        # from the shared pma so the reject gradient never touches the cls pool. See _reject_pool/_reject_z.
        if decouple_reject or feas_filter:
            self.pma_reject = PMA(d_model, n_heads, k_seeds=1, dropout=dropout)

        # Classification head
        if cls_decoder == "per_donor":
            # Per-donor cross-attention decoder (compositional, generalizes to novel combos)
            self.cls_decoder_module = PerDonorDecoder(d_model, n_heads, n_classes, dropout,
                                                      n_layers=dec_layers, sparse=sparse_attn, vib=vib,
                                                      n_sink=attn_sink, dec_aggr=dec_aggr,
                                                      phi_inject=phi_inject)  # Inc18 V1
            if self.donor_recon:      # Inc9 B1: map each donor rep back to its own peaks' pooled encoding
                self.donor_recon_head = nn.Linear(d_model, d_model)
            # "local": position-wise token encoder feeding the per-donor decoder,
            # bypassing ISAB context-mixing (tests the entanglement hypothesis).
            if decoder_source == "local":
                self.local_encoder = LocalTokenEncoder(d_model, n_blocks=2, dropout=dropout)
        elif cls_decoder == "additive":
            # Additive per-donor readout (LR-like, no softmax competition over tokens)
            self.cls_decoder_module = AdditiveDonorDecoder(d_model, n_classes, dropout)
            if decoder_source == "local":
                self.local_encoder = LocalTokenEncoder(d_model, n_blocks=2, dropout=dropout)
        elif cls_decoder == "dsmil":
            # Inc12: Dual-Stream MIL (max instance + dense bag) — loud donor via max, faint via dense bag.
            self.cls_decoder_module = DSMILDecoder(d_model, n_classes, dropout)
            if decoder_source == "local":
                self.local_encoder = LocalTokenEncoder(d_model, n_blocks=2, dropout=dropout)
        elif cls_decoder == "sos":
            # Inc19: two-level set-of-set decoder (private alleles L1 + shared alleles L2 w/ context).
            self.cls_decoder_module = SetOfSetDecoder(d_model, n_heads, n_classes, dropout)
        elif cls_decoder == "aslot":
            # Inc20+: CoSA + GSANet + MESH + AdaSlot (see AdaptiveSlotDecoder above).
            # Requires donor_geno for CoSA slot init; aux_heads for GSANet (optional but recommended).
            assert donor_geno is not None, "cls_decoder='aslot' needs donor_geno for CoSA slot init"
            self.cls_decoder_module = AdaptiveSlotDecoder(
                d_model, n_classes=n_classes, n_noc=5,
                n_iters=n_slot_iters, ot_eps=ot_eps, ot_iters=ot_iters,
                gumbel_temp=gumbel_temp, dropout=dropout, gate_mass=gate_mass,
            )
            # Register donor_geno buffers + geno_proj if not already done by geno_query branch.
            # _encode_geno() uses these for the CoSA slot initialization.
            if not geno_query:
                self.register_buffer("donor_geno", donor_geno.float())
                self.register_buffer("donor_geno_mask", donor_geno_mask.bool())
                self.geno_proj = nn.Linear(d_model, d_model)
                nn.init.zeros_(self.geno_proj.weight); nn.init.zeros_(self.geno_proj.bias)
        elif cls_decoder == "spen":
            # Inc21: Joint EBM-SPEN structured prediction.
            # Local head: pooled z → 45-dim initial logits (returned as logits_cls_0).
            # SPENDecoder refines via T-step projected GD minimising
            #   E(x,y) = -local·y + global_net(y) + joint_sc(x * tanh(y_proj(y))).
            # Training: (T-1) detached warm-up + 1 step create_graph (Tu & Gimpel 2018).
            self.spen_local_head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(d_model, n_classes)
            )
            self.cls_decoder_module = SPENDecoder(
                d_model, n_classes=n_classes, n_noc=5,
                n_inf_steps=n_inf_steps, inf_lr=inf_lr,
                d_global=spen_global_hidden, dropout=dropout,
            )
        elif cls_decoder == "hybrid":
            # Two-stream: aligned Xflat base (preserves minority donors, LR-like) +
            # bounded set-attention adjustment (interaction/joint structure, ST-like).
            #   logits = flat_head(Xflat) + set_scale * tanh(set_adjust(z_mix))
            # tanh bounds the set correction so it can nudge but NOT annihilate the
            # flat base — protecting faint minority contributors (e.g. donor 48).
            self.flat_norm = nn.LayerNorm(n_flat)
            self.flat_head = nn.Sequential(
                nn.Linear(n_flat, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(256, n_classes),
            )
            self.set_adjust = nn.Linear(d_model, n_classes)
            self.set_scale = nn.Parameter(torch.tensor(0.1))
        else:
            # Original: joint linear head on pooled z_mix
            self.cls_head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(d_model, n_classes),
            )

        self.reject_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Cardinality-aware NOC head (replaces old embedding noc_head). Reads the
        # sorted donor-prob profile -> NOC k=1..5 scores.
        self.cardinality_head = CardinalityHead(n_classes, top_m=8, n_card=5, dropout=dropout)

        # Paper-proof learned NOC head (gated by noc_head_v2; default off → no-op). CORN ordinal regression
        # (Cao et al. 2020 arXiv 1901.07884; Shi et al. 2021 arXiv 2111.08851) over NOC 1..5, reading a
        # multiset-COUNT input: sorted prob-profile + MAC/physical features (PACE, Marciano & Adelman 2017;
        # deepNoC 2024) + mass-preserving slot-mass (Fischer & Gärtner 2024 / GIN/DeepSets — sum, not the
        # row-normalized mean that erases count). Output 4 = K-1 rank-consistent thresholds.
        self.noc_head_v2 = bool(noc_head_v2)
        if self.noc_head_v2:
            self._ocv2_topm = 8
            _ocin = (self._ocv2_topm + 2) + 19 + self._ocv2_topm + (5 if em_phi_feature else 0)  # prob(10)+MAC(19)+mass(8)[+Hill(5)]
            self.ord_count_head = nn.Sequential(
                nn.Linear(_ocin, 64), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, 4))

        # em_phi_feature: internalize the INDEPENDENT EM mixture-proportion deconvolution (phi_rerank.py is
        # the post-hoc floor). cls: a per-donor function of standardized log-phi added to the logit = a
        # learned logarithmic OPINION POOL / product-of-experts (Genest & Zidek 1986; Hinton 2002), valid
        # because phi is orthogonal to logits_cls (measured corr ~0.73). Delivered FiLM-lite (Perez 2018).
        # Last layer ZERO-init (ReZero, Bachlechner 2020) -> exact no-op at start, trains in only if it helps.
        self.em_phi_feature = bool(em_phi_feature)
        if self.em_phi_feature:
            self.emphi_head = nn.Sequential(nn.Linear(1, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
            nn.init.zeros_(self.emphi_head[-1].weight); nn.init.zeros_(self.emphi_head[-1].bias)

        # ── SOFT per-peak NOISE/STUTTER reliability gate (noise_gate; default off → no-op) ───────────
        # s_p = P(peak is a real allele) ∈ [floor,1] from PROVEN structural features:
        #   log1p height (Tvedebrink peak-quality), back/forward stutter ratio to the n±1 parent and
        #   parent-present flags (stutter = n-1 repeat at SR≈5-15% of a taller parent; Brookes 2012,
        #   Bright/STRmix), and within-locus relative height. Supervised (trainer) by the proven label
        #   "peak's (locus,allele) carried by ≥1 TRUE contributor" (Balding-Buckleton real-vs-drop-in),
        #   so the gate learns a calibrated reliability instead of co-adapting through the cls loss.
        # Bias of the last layer init HIGH (+4) → s_p≈1 at start (graceful near-no-op); the supervised
        # loss then pushes phantoms down. Applied as a soft, floored down-weight in the aslot decoder.
        self.noise_gate = bool(noise_gate)
        self.noise_s_floor = float(noise_s_floor)
        self._NG_F = 6                                          # [log1p h, BSR, has_back, FSR, has_fwd, rel_h]
        if self.noise_gate:
            self.noise_gate_head = nn.Sequential(
                nn.Linear(self._NG_F, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))
            # tiny (NOT zero) last-layer weight + high bias → s_p≈0.98 near-no-op start AND gradient still
            # flows to the first layer (zeroing the last weight would block it — the gate is supervised, so
            # it needs grad flow, not an exact no-op like the em_phi additive term).
            nn.init.normal_(self.noise_gate_head[-1].weight, std=0.01)
            nn.init.constant_(self.noise_gate_head[-1].bias, 4.0)   # sigmoid(4)≈0.98 → near-no-op start

        # Increment 2 §5 — PRIVILEGED auxiliary supervision (in-silico only; LUPI / generalized
        # distillation). attr_head: per-peak donor attribution (n_classes + 1 "none/background")
        # on the context-mixed encoded peaks H -> teaches the encoder allele→donor grouping
        # directly. phi_head: mixture-proportion (abundance) regression on the pooled z (ANC via
        # softplus). Both are HEADS-ONLY at train time; discarded at inference (real data has no
        # provenance) so they cost nothing on the test path.
        self.aux_heads = aux_heads
        if aux_heads:
            self.attr_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes + 1))
            self.phi_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        # §3.4 presence-gated phi + β-NLL (phi_gated; needs aux_heads). Adds a per-donor log-variance head so
        # phi is trained with a heteroscedastic β-NLL (Seitzer 2022) on the PRESENT donors only (loss masked
        # by y>0) — fixes the L1-on-88%-zero collapse (which drove phi→0 everywhere). Off → phi_head path
        # unchanged (bit-identical).
        self.phi_gated = bool(phi_gated) and bool(aux_heads)
        if self.phi_gated:
            self.phi_logvar_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))
        # Inc-LUPI privileged PHYSICAL heads (train-only; discarded at inference like aux_heads).
        self.lupi_phys = lupi_phys
        if lupi_phys:
            self.degr_head   = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))   # per-mixture degradation β (scaled [0,1])
            self.muvar_head  = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 2))   # per-peak (mu_mean in log1p, log-variance) — Gaussian β-NLL
            self.dropin_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))   # per-peak drop-in (phantom) logit

        # Inc14 B-v2 — per-peak MULTI-LABEL genotype co-membership head (sigmoid, n_classes independent
        # binaries). Unlike attr_head's softmax (forced 1-donor/peak — forensically wrong, CPI never
        # assigns), this head can credit a SHARED allele to every present donor who carries it (the 75%
        # masked evidence the faint minor needs). Inference-inert (aux only); read in the train loop as
        # the MLD distillation teacher and as a representation-shaping BCE aux.
        self.ml_attr = ml_attr
        if ml_attr:
            self.ml_attr_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

        # Inc16 — deep-unfolded SUCCESSIVE INTERFERENCE CANCELLATION (DeepSIC, Shlezinger/Eldar 2022 +
        # deep unfolding). Q weight-TIED iterations: encode residual peaks -> identify donors -> forward-
        # reconstruct the confident donors' expected contribution (genotype copy x Mx) -> SUBTRACT from the
        # RAW peak heights (signal level, NOT latent — the encoder entangles) -> re-encode the residual.
        # Removing dominant donors exposes the faint minor and collapses a decoy's borrowed evidence.
        # ACCUMULATE by max over iterations (a donor is present if confident in ANY iteration, so subtracted
        # majors aren't lost). sic_alpha zero-init => no subtraction at init => reduces EXACTLY to the base
        # one-shot decoder (warm-startable). Needs aux_heads (phi for reconstruction) + donor_geno.
        self.sic_iters = int(sic_iters)
        if self.sic_iters > 1:
            assert aux_heads, "sic_iters needs aux_heads (phi head for reconstruction)"
            assert donor_geno is not None, "sic_iters needs donor_geno (copy numbers for reconstruction)"
            assert cls_decoder == "per_donor", "sic_iters implemented for the per_donor decoder"
            self.sic_alpha = nn.Parameter(torch.zeros(self.sic_iters - 1))   # zero-init -> reduce to base
            self.sic_off = 30; lutw = 1024
            lut = torch.zeros(n_loci, lutw, n_classes)
            gg = donor_geno; gmk = (donor_geno_mask.bool() if donor_geno_mask is not None
                                    else torch.ones(gg.shape[:2], dtype=torch.bool))
            for c in range(min(n_classes, gg.size(0))):
                byloc = {}
                for j in range(gg.size(1)):
                    if gmk[c, j]:
                        li = int(gg[c, j, 0]); al = int(round(float(gg[c, j, 1]) * 10))
                        byloc.setdefault(li, []).append(al)
                for li, als in byloc.items():
                    u = set(als); cp = 2.0 if len(u) == 1 else 1.0
                    for a in u:
                        ab = a + self.sic_off
                        if 0 <= li < n_loci and 0 <= ab < lutw:
                            lut[li, ab, c] = cp
            self.register_buffer("sic_owner_lut", lut)

        # Increment 2 §4b — DECOUPLED ordinal NOC contrast (Rank-N-Contrast, Zha 2023).
        # ID-PROTECTION VALVES (§4b-C): the contrastive lives on its OWN pooled vector
        # (pma_noc — same decoupling precedent as pma_reject) + a projection head proj_noc
        # that is DISCARDED at inference (SimCLR: representation-before-projection). So the
        # NOC-ordinal geometry never sits on the ID readout (z / per-donor decoder) → shaping
        # the representation by count cannot collapse donor identity (the negative-transfer
        # failure mode §4b-C warns about). z_noc_proj is emitted only for the training loss.
        self.noc_contrast = noc_contrast
        self.noc_detach = noc_detach
        # Inc3 NOC-ordinal (V1-3): CORN count head reads its OWN encoder pool (pma_card) that RNC
        # actually shapes — unlike the base card head which reads the detached ID profile (F21).
        self.noc_ord_head = noc_ord_head
        self.noc_ord_detach = noc_ord_detach
        self.noc_ord_replace = noc_ord_replace
        if noc_ord_head:
            self.pma_card = PMA(d_model, n_heads, k_seeds=1, dropout=dropout)
            self.corn_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 4))  # K-1=4 thresholds (NOC1..5)
        # RNC projection: built when contrast is on. With noc_ord_head it pools pma_card (the count
        # pool) so the contrast shapes the representation the count head READS; else its own pma_noc.
        if noc_contrast:
            self.proj_noc = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_proj))
            if not noc_ord_head:
                self.pma_noc = PMA(d_model, n_heads, k_seeds=1, dropout=dropout)

        # Inc3 A — reference-genotype-conditioned donor queries (DAB-DETR/Q2L + LUPI).
        self.geno_query = geno_query
        if geno_query:
            assert donor_geno is not None, "geno_query needs donor_geno tokens"
            self.register_buffer("donor_geno", donor_geno.float())            # (C, G, n_token_feats)
            self.register_buffer("donor_geno_mask", donor_geno_mask.bool())   # (C, G)
            self.geno_proj = nn.Linear(d_model, d_model)
            nn.init.zeros_(self.geno_proj.weight); nn.init.zeros_(self.geno_proj.bias)  # start == base

        # Inc10 — discriminability-weighted reference-allele MATCHING (AFIA / Less-is-More + explicit
        # matching). probe7/8 (no-train): adding a rarity-weighted reference-match score to the per-donor
        # logit lifts N5 set-EM .72->.83 by crediting the DECISIVE (panel-rare) alleles the holistic
        # decoder under-uses (verified mechanism). geno_query (soft query-anchor) did NOT realise this;
        # here it is an EXPLICIT additive match score. V1 = fixed 1/panel_count weight (= the verified
        # ensemble, internalised, learnable scale only); V2 (ref_match_learn) = LEARN per-allele weight
        # (init panel-rarity) = AFIA. No encoder/decoder change; deterministic match (no grad through it).
        self.ref_match = bool(ref_match)
        if self.ref_match:
            assert donor_geno is not None, "ref_match needs donor_geno tokens"
            # Register the geno buffers ONLY if no other branch already did (geno_query at
            # __init__ above, or the aslot/CoSA decoder branch which registers them for its slot
            # init). Guarding on hasattr (not just `not geno_query`) prevents a double-register
            # KeyError when --ref_match is combined with --cls_decoder aslot (or any future geno user).
            if not hasattr(self, "donor_geno"):
                self.register_buffer("donor_geno", donor_geno.float())
                self.register_buffer("donor_geno_mask", donor_geno_mask.bool())
            g = donor_geno.float(); gm = donor_geno_mask.bool()                 # (C,G,*), (C,G)
            # integer key per reference allele: locus*1000 + round(allele*10)+30  (alleles ~ -2..51.2)
            self._ref_kmax = 24000
            rk = (g[:, :, 0].long().clamp(0, 23) * 1000
                  + (torch.round(g[:, :, 1] * 10).long() + 30).clamp(0, 999))
            rk = torch.where(gm, rk, torch.full_like(rk, self._ref_kmax - 1))    # invalid -> sentinel slot
            # DEDUP per donor: donor_geno stores homozygous loci as 2 identical rows; keep each unique
            # (locus,allele) key ONCE so the match counts a present allele once (= the validated probe).
            valid = torch.zeros_like(gm)
            cnt = defaultdict(int)
            for c in range(g.size(0)):
                seen = set()
                for j in range(g.size(1)):
                    if gm[c, j]:
                        k = int(rk[c, j])
                        if k not in seen:
                            seen.add(k); valid[c, j] = True; cnt[k] += 1   # count donor once per key
            self.register_buffer("ref_key", rk)                                 # (C,G) long
            self.register_buffer("ref_valid", valid.float())                    # (C,G) dedup'd
            w0 = torch.zeros_like(self.ref_valid)
            for c in range(g.size(0)):
                for j in range(g.size(1)):
                    if valid[c, j]:
                        w0[c, j] = 1.0 / max(cnt[int(rk[c, j])], 1)
            if ref_match_learn:
                self.ref_w = nn.Parameter(w0)                                   # AFIA: learn discriminability
            else:
                self.register_buffer("ref_w", w0)                              # V1: fixed panel-rarity
            # init SMALL (not 1.0): the donor-standardised match adds up to ~√C ≈ 6.5 logits at scale 1.0,
            # which can dominate the slot/per-donor signal early in training. Start at 0.1 (≤~0.65-logit nudge)
            # and let the gradient raise it only as it reduces loss — zero/small-init of an additive residual
            # branch is a stability-preserving init (ReZero, Bachlechner 2020; Fixup, Zhang 2019). The match
            # signal is still exposed from step 1 (scale>0); training recovers the right weight.
            self.match_scale = nn.Parameter(torch.tensor(0.1))

        # Inc3 B — supervised-contrastive peak grouping by source donor (SupCon, decoupled projection).
        self.donor_contrast = donor_contrast
        if donor_contrast:
            self.proj_peak = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
                nn.Linear(d_model, d_proj))

    def encode(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Backward-compatible: returns pooled z_mix (B, d_model)."""
        _x0, H, pad_mask = self._encode_set(tokens, mask)
        z = self.pma(H, pad_mask=pad_mask)   # (B, 1, d_model)
        return z.squeeze(1)                  # (B, d_model)

    def _project_tokens(
        self, tokens: torch.Tensor, mask: torch.Tensor, apply_feas: bool = True
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Token → projected embeddings x0 (B, N, d) + pad_mask (pre-ISAB).
        apply_feas=False skips feas_filter (used by the reject pool, which must see out-of-panel peaks)."""
        locus_idx = tokens[:, :, 0].long().clamp(0, self.n_loci - 1)
        continuous = tokens[:, :, 1:self.n_token_feats]
        continuous = (continuous - self.feat_mean) / self.feat_std   # standardize (identity if not set)
        locus_emb = self.locus_embed(locus_idx)
        num = self.num_embed(continuous) if self.num_embed is not None else continuous
        x = torch.cat([locus_emb, num], dim=-1)
        x = self.input_proj(x)
        x = x * mask.unsqueeze(-1)          # zero padding
        pad_mask = ~mask                    # True = ignore
        if self.feas_filter and self.owner_lut is not None and apply_feas:
            # Inc18 V3: remove peaks not carried by any reference donor — pure noise.
            # Applied at BOTH train and inference for train/test consistency (no covariate shift).
            bi = (tokens[..., 1] * 10).round().long() + self._AOFF
            bi = bi.clamp(0, self.owner_lut.size(1) - 1)
            feas = self.owner_lut[locus_idx, bi].sum(-1) > 0  # (B, N) True = at least 1 carrier
            x = x * feas.unsqueeze(-1).to(x.dtype)            # zero infeasible embeddings
            pad_mask = pad_mask | ~feas                        # infeasible → ignored by attention
        return x, pad_mask

    def _encode_geno(self) -> torch.Tensor:
        """Inc3 A: encode each donor's reference-allele set into a query anchor (C, d_model),
        through the SAME locus_embed + input_proj as peaks (so it lives in the peak space).
        Non-allele continuous features are neutralised to 0 (= mean post-standardization)."""
        g = self.donor_geno                                  # (C, G, n_token_feats)
        locus_idx = g[:, :, 0].long().clamp(0, self.n_loci - 1)
        cont = (g[:, :, 1:self.n_token_feats] - self.feat_mean) / self.feat_std
        cont = cont.clone(); cont[:, :, 1:] = 0.0            # keep allele (col 0), neutralise the rest
        num = self.num_embed(cont) if self.num_embed is not None else cont
        x = self.input_proj(torch.cat([self.locus_embed(locus_idx), num], dim=-1))
        m = self.donor_geno_mask.unsqueeze(-1).float()
        emb = (x * m).sum(1) / m.sum(1).clamp_min(1.0)       # masked mean over a donor's alleles (C, d)
        return self.geno_proj(emb)                            # zero-init at start -> == base

    def _ref_match_score(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Inc10: per-donor discriminability-weighted reference-allele match (B, C).
        For each donor c: sum over its reference alleles PRESENT in the observed peaks of ref_w[c,*]
        (= 1/panel_count at init). Matching is discrete (no grad); gradient flows to ref_w (V2) and the
        global match_scale only. Standardized across donors per sample for stable additive scale."""
        B = tokens.size(0); K = self._ref_kmax
        ok = (tokens[:, :, 0].long().clamp(0, 23) * 1000
              + (torch.round(tokens[:, :, 1] * 10).long() + 30).clamp(0, 999))
        ok = torch.where(mask.bool(), ok, torch.full_like(ok, K - 1)).clamp(0, K - 1)
        table = torch.zeros(B, K, device=tokens.device)
        table.scatter_(1, ok, torch.ones_like(ok, dtype=table.dtype))           # presence per observed key
        present = table[:, self.ref_key.reshape(-1)].reshape(B, *self.ref_key.shape)  # (B,C,G)
        match = (present * (self.ref_w * self.ref_valid).unsqueeze(0)).sum(-1)   # (B,C)
        match = (match - match.mean(1, keepdim=True)) / (match.std(1, keepdim=True) + 1e-6)
        return self.match_scale * match

    def _encode_set(
        self, tokens: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns (x0, H, pad_mask):
            x0: (B, N, d)  pre-ISAB projected tokens (locally identifiable alleles)
            H:  (B, N, d)  ISAB-encoded tokens (globally context-mixed)

        Set-of-Set mode (--set_of_set): peaks are split into private (n_carriers==1) and
        shared (n_carriers>1) sets BEFORE encoding.  Each set passes through the SAME ISAB
        encoder independently (lite: shared weights, no extra params), then the outputs are
        merged element-wise.  This prevents shared peaks from drowning minor-donor private
        peaks through competitive attention (explaining-away).
        """
        x0, pad_mask = self._project_tokens(tokens, mask)

        if self.set_of_set and self.owner_lut is not None:
            # Carrier count per peak from panel LUT
            li = tokens[..., 0].long().clamp(0, 23)
            bi = (tokens[..., 1] * 10).round().long() + self._AOFF
            bi = bi.clamp(0, self.owner_lut.size(1) - 1)
            n_car = self.owner_lut[li, bi].sum(-1)   # (B, N)  #panel donors carrying this allele

            # Private = exactly 1 carrier; Shared = everything else (n_car>1 OR n_car==0).
            # n_car==0 peaks (allele not in panel LUT) are routed through the shared encoder
            # rather than being excluded: in OOD settings a kit-shifted real allele can appear
            # as n_car==0 due to bin rounding, and excluding it entirely would blind the decoder.
            # If feas_filter is also enabled it will have already removed true noise peaks from
            # pad_mask before we get here; SoS itself stays OOD-safe.
            is_valid = ~pad_mask                      # (B, N)
            is_priv  = (n_car == 1) & is_valid
            is_shar  = (n_car != 1) & is_valid        # n_car>1 (shared) + n_car==0 (unknown/OOD)

            # BERT-style -1e4 float mask in MABpp.forward prevents NaN when all keys are masked
            # (samples with zero private or zero shared peaks); no special guard needed here.
            priv_pm = ~is_priv   # True = masked from private encoder
            shar_pm = ~is_shar   # True = masked from shared encoder

            # --- Private encoder pass: only private peaks attend to each other ---
            H_priv = x0
            for isab in self.encoder:
                H_priv = isab(H_priv, pad_mask=priv_pm)
            H_priv = H_priv * is_priv.unsqueeze(-1).to(H_priv.dtype)   # zero non-private positions

            # --- Shared encoder pass: only shared peaks attend to each other ---
            H_shar = x0
            for isab in self.encoder:
                H_shar = isab(H_shar, pad_mask=shar_pm)
            H_shar = H_shar * is_shar.unsqueeze(-1).to(H_shar.dtype)   # zero non-shared positions

            # Merge: is_priv & is_shar are disjoint → addition = clean selection
            # Peaks with n_car==0 (infeasible) contribute 0; feas_filter (if on) already added
            # them to pad_mask so decoder won't attend to them anyway.
            H = H_priv + H_shar
            return x0, H, pad_mask

        # Standard single-encoder path
        H = x0
        for isab in self.encoder:
            H = isab(H, pad_mask=pad_mask)
        return x0, H, pad_mask

    def _reject_pool(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Reject-head pool from an UNFILTERED, plain single-pass encode, DETACHED before pooling.
        feas_filter strips out-of-panel peaks before the encoder — exactly the unknown-contributor
        evidence the open-set reject head needs (F1). This re-encodes WITHOUT feas_filter (same encoder
        weights, no set_of_set split), and `.detach()` + the dedicated pma_reject guarantee the reject
        gradient never reaches the encoder or the shared cls pool, so the whole cls path stays
        gradient-identical to the baseline (no encoder contamination)."""
        x0u, pad_u = self._project_tokens(tokens, mask, apply_feas=False)
        Hu = x0u
        for isab in self.encoder:
            Hu = isab(Hu, pad_mask=pad_u)
        return self.pma_reject(Hu.detach(), pad_mask=pad_u).squeeze(1)

    def _reject_z(self, z: torch.Tensor, H: torch.Tensor, pad_mask: torch.Tensor,
                  tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Pick the reject pool: unfiltered+detached when feas_filter removed peaks reject needs >
        dedicated pma_reject (decouple_reject) > shared z."""
        if self.feas_filter and getattr(self, "pma_reject", None) is not None:
            return self._reject_pool(tokens, mask)
        if self.decouple_reject:
            return self.pma_reject(H, pad_mask=pad_mask).squeeze(1)
        return z

    def _mac_feats_torch(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Torch MAC/physical count features (PACE, Marciano & Adelman 2017; deepNoC 2024): per-locus
        allele counts at several RFU thresholds + global height stats — the PHYSICAL number-of-contributors
        signal (max allele count) the washed slot gate lacks. Fixed input feature (no grad). Returns (B,19)."""
        with torch.no_grad():
            h = torch.expm1(tokens[:, :, 2])                          # (B,N) RFU
            loc = tokens[:, :, 0].long().clamp(0, self.n_loci - 1)
            mb = mask.bool(); B = tokens.size(0); feats = []
            for T in (0.0, 50.0, 150.0, 500.0):
                v = (mb & (h > T)).to(h.dtype)                        # (B,N)
                cnt = torch.zeros(B, self.n_loci, device=tokens.device).scatter_add_(1, loc, v)
                feats += [cnt.amax(1, keepdim=True),                  # MAC at this threshold
                          (cnt >= 2).sum(1, keepdim=True).to(h.dtype),
                          (cnt >= 3).sum(1, keepdim=True).to(h.dtype),
                          v.sum(1, keepdim=True)]                     # #peaks > T
            mf = mb.to(h.dtype); denom = mf.sum(1, keepdim=True).clamp(min=1)
            lg = torch.log1p(h.clamp(min=0)) * mf
            lmean = lg.sum(1, keepdim=True) / denom
            lvar = (((lg - lmean) ** 2) * mf).sum(1, keepdim=True) / denom
            feats += [lmean, torch.sqrt(lvar + 1e-6), lg.amax(1, keepdim=True)]
            return torch.cat(feats, dim=1)                            # (B, 19)

    def _stutter_feats(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """PROVEN per-peak structural features for the soft noise gate (B, N, 6). Stutter is an n-1
        repeat artifact at ~5-15% of a TALLER parent's height (Brookes 2012; Bright/STRmix); drop-in/
        dropout-adjacent peaks are low (Tvedebrink height peak-quality). Features:
          [log1p height, back-stutter ratio (h / parent@a+1), has-back-parent,
           forward-stutter ratio (h / parent@a-1), has-fwd-parent, within-locus relative height].
        These give the gate the STRUCTURE phi lacks: distinguish 'low because faint minor' (no taller
        n±1 parent) from 'low because stutter' (taller parent at SR<~0.15). Deterministic, no grad."""
        with torch.no_grad():
            h = torch.expm1(tokens[:, :, 2]).clamp(min=0)                 # (B,N) RFU
            a = tokens[:, :, 1]; loc = tokens[:, :, 0]; mb = mask.bool()
            same = (loc[:, :, None] == loc[:, None, :]) & mb[:, None, :]   # (B,N,N) q valid, same locus as p
            hq = h[:, None, :]
            back = same & (torch.abs(a[:, None, :] - (a[:, :, None] + 1.0)) < 0.05)   # q is p's n+1 parent
            fwd  = same & (torch.abs(a[:, None, :] - (a[:, :, None] - 1.0)) < 0.05)   # q is p's n-1 parent
            has_back = back.any(-1).float(); has_fwd = fwd.any(-1).float()
            BSR = h / ((back.float() * hq).amax(-1) + 1e-6) * has_back
            FSR = h / ((fwd.float()  * hq).amax(-1) + 1e-6) * has_fwd
            rel_h = h / (same.float() * hq).amax(-1).clamp(min=1e-6)       # height vs tallest at locus
            feats = torch.stack([torch.log1p(h), BSR.clamp(0, 2), has_back,
                                 FSR.clamp(0, 2), has_fwd, rel_h.clamp(0, 1)], dim=-1)  # (B,N,6)
            return feats * mb.float().unsqueeze(-1)                        # zero padding

    def _ord_count_logits(self, logits_cls: torch.Tensor, tokens: torch.Tensor,
                          mask: torch.Tensor, slot_mass: torch.Tensor | None = None,
                          emphi_hill: torch.Tensor | None = None) -> torch.Tensor:
        """Paper-proof count head (noc_head_v2): CORN ordinal logits (B,4 = K-1) over NOC 1..5 from a
        multiset-COUNT input = sorted prob-profile + MAC physical features + mass-preserving slot-mass.
        All inputs DETACHED (the count head does not perturb the ranking/slots)."""
        m = self._ocv2_topm
        probs = torch.sigmoid(logits_cls).detach()
        s = torch.sort(probs, dim=1, descending=True).values[:, :m]
        pp = torch.cat([s, probs.sum(1, keepdim=True),
                        (probs >= 0.5).to(probs.dtype).sum(1, keepdim=True)], dim=1)   # (B, m+2)
        mac = self._mac_feats_torch(tokens, mask)                                       # (B, 19)
        if slot_mass is not None:
            sm = torch.sort(slot_mass.detach(), dim=1, descending=True).values[:, :m]   # (B, m)
        else:
            sm = probs.new_zeros(probs.size(0), m)
        parts = [pp, mac, sm]
        if emphi_hill is not None:
            parts.append(emphi_hill.detach())                                           # (B, 5) Hill effective-count
        return self.ord_count_head(torch.cat(parts, dim=1))                             # (B, 4)

    def _em_deconv_phi(self, tokens: torch.Tensor, mask: torch.Tensor, n_iters: int = 10) -> torch.Tensor:
        """INDEPENDENT EM mixture-proportion (Mx) deconvolution (EuroForMix-class, Bleka 2016), fully in
        torch from tokens + owner_lut — NO neural signal, so it stays ORTHOGONAL to logits_cls (the property
        the logarithmic-opinion-pool needs; measured corr ~0.73). Uniform compat among genotype-carriers +
        background sink, height-weighted EM. Returns (B, C). Detached — a fixed input feature, no grad."""
        with torch.no_grad():
            B, N, _ = tokens.shape; C = self.owner_lut.size(2)
            loc = tokens[:, :, 0].long().clamp(0, self.n_loci - 1)
            ab = (tokens[:, :, 1] * 10).round().long() + self._AOFF
            ab = ab.clamp(0, self.owner_lut.size(1) - 1)
            carr = self.owner_lut[loc, ab]                                # (B, N, C) carrier membership {0,1}
            valid = mask.bool() & (carr.sum(-1) > 0)                      # peaks with >=1 panel carrier
            h = (torch.expm1(tokens[:, :, 2]).clamp(min=0) * valid).to(tokens.dtype)     # (B, N) RFU
            NEG = -1e9
            S = torch.full((B, N, C + 1), NEG, device=tokens.device, dtype=tokens.dtype)
            S[:, :, :C] = torch.where(carr > 0, torch.zeros_like(carr), torch.full_like(carr, NEG))
            S[:, :, C] = -2.0                                             # background sink (finite -> no NaN row)
            phi = torch.full((B, C + 1), 1.0 / (C + 1), device=tokens.device, dtype=tokens.dtype)
            for _ in range(n_iters):
                z = S + torch.log(phi + 1e-9).unsqueeze(1)               # (B, N, C+1)
                z = z - z.amax(-1, keepdim=True)
                A = torch.softmax(z, dim=-1)
                w = (A[:, :, :C] * h.unsqueeze(-1)).sum(1)               # (B, C)
                bg = (A[:, :, C] * h).sum(1)                             # (B,)
                tot = (w.sum(1, keepdim=True) + bg.unsqueeze(1)).clamp(min=1e-9)
                phi = torch.cat([w, bg.unsqueeze(1)], 1) / tot
            return phi[:, :C]                                            # (B, C)

    def _hill_feats(self, phi: torch.Tensor) -> torch.Tensor:
        """Effective-number-of-contributors features from the Mx vector (Hill 1973; Jost 2006): richness at
        thresholds, exp(Shannon entropy) and inverse Simpson — a proven proportion->count transform. (B, 5)."""
        p = phi / phi.sum(1, keepdim=True).clamp(min=1e-9)
        rich = torch.stack([(phi > t).to(phi.dtype).sum(1) for t in (0.02, 0.05, 0.10)], 1)  # (B,3)
        hill1 = torch.exp(-(p * torch.log(p + 1e-9)).sum(1)).unsqueeze(1)                     # exp(Shannon)
        hill2 = (1.0 / (p * p).sum(1).clamp(min=1e-9)).unsqueeze(1)                           # inverse Simpson
        return torch.cat([rich, hill1, hill2], 1)                                            # (B, 5)

    def _apply_emphi(self, logits_cls: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor):
        """If em_phi_feature: add the internalized LOP term (FiLM-lite on standardized log-phi) to the cls
        logit and return the Hill count features for the NOC head. Returns (logits_cls, emphi_hill | None).
        No-op (and (.., None)) when the flag is off or owner_lut is absent."""
        if not self.em_phi_feature or self.owner_lut is None:
            return logits_cls, None
        phi = self._em_deconv_phi(tokens, mask)                          # (B, C) detached
        lp = torch.log(phi + 1e-6)
        lp = (lp - lp.mean(1, keepdim=True)) / (lp.std(1, keepdim=True) + 1e-6)   # per-sample z (LOP scale)
        logits_cls = logits_cls + self.emphi_head(lp.unsqueeze(-1)).squeeze(-1)   # (B, C)
        return logits_cls, self._hill_feats(phi)

    # ── Inc16 deep-unfolded SIC helpers ──────────────────────────────────────
    def _sic_set_heights(self, tokens: torch.Tensor, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Rebuild the height-derived token features (log_h col2, Hb col3, glob_rel col7) from heights h
        (the current residual). col2==base when h==observed -> iteration 0 reproduces the original input."""
        nl = self.n_loci; t = tokens.clone(); mk = mask.bool().float()
        hmax = (h * mk).amax(1, keepdim=True).clamp(min=1e-6)
        glob = (h / hmax) * mk
        loc = t[:, :, 0].long().clamp(0, nl - 1)
        locsum = torch.zeros(t.size(0), nl, device=h.device)
        locsum.scatter_add_(1, loc, h * mk)
        hb = (h / locsum.gather(1, loc).clamp(min=1e-6)) * mk
        t[:, :, 2] = torch.log1p(h) * mk
        if t.size(2) > 3: t[:, :, 3] = hb
        if t.size(2) > 7: t[:, :, 7] = glob
        return t

    def _sic_gather_owner(self, tokens: torch.Tensor) -> torch.Tensor:
        loc = tokens[:, :, 0].long().clamp(0, self.n_loci - 1)
        ab = (torch.round(tokens[:, :, 1] * 10).long() + self.sic_off).clamp(0, self.sic_owner_lut.size(1) - 1)
        return self.sic_owner_lut[loc, ab]                       # (B, N, n_classes) copy number

    def _forward_sic(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        obs_h = torch.expm1(tokens[:, :, 2]) * mask              # observed peak heights (0 at pad)
        r = obs_h.clone(); per = []; H0 = z0 = pad0 = None
        for q in range(self.sic_iters):
            tk = self._sic_set_heights(tokens, r, mask)
            _x0, H, pad = self._encode_set(tk, mask)
            z = self.pma(H, pad_mask=pad).squeeze(1)
            gq = self._encode_geno() if self.geno_query else None
            lg = self.cls_decoder_module(H, pad_mask=pad, geno_query=gq)
            per.append(lg)
            if q == 0:
                H0, z0, pad0 = H, z, pad
            if q < self.sic_iters - 1:                           # reconstruct & subtract for the next iter
                phi = torch.nn.functional.softplus(self.phi_head(z))
                w = torch.sigmoid(lg) * phi                      # (B,45) presence-gated proportion
                pred = (self._sic_gather_owner(tk) * w.unsqueeze(1)).sum(-1)   # (B,N) reconstructed (relative)
                mk = mask.float()
                T = (obs_h * mk).sum(1, keepdim=True) / (pred * mk).sum(1, keepdim=True).clamp(min=1e-6)
                r = torch.relu(obs_h - self.sic_alpha[q] * T * pred)   # alpha = fraction of reconstruction to cancel
        logits_cls = torch.stack(per, 0).max(0).values          # accumulate: present if confident in ANY iter
        self._sic_per = per
        z_rej = self._reject_z(z0, H0, pad0, tokens, mask)
        out = {
            "logits_cls": logits_cls,
            "logit_reject": self.reject_head(z_rej),
            "logits_card": self.cardinality_head(torch.sigmoid(logits_cls).detach()),
        }
        if self.aux_heads:                                       # aux from the full mixture (iteration 0)
            out["logits_attr"] = self.attr_head(H0)
            out["phi"] = torch.nn.functional.softplus(self.phi_head(z0))
        return out

    def _forward_aslot(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        Forward for cls_decoder="aslot" (CoSA + GSANet + MESH + AdaSlot).
        Self-contained: returns the complete output dict directly.

        Flow:
          encoder → _encode_geno CoSA slot init → (GSANet attr refinement if aux_heads) →
          AdaptiveSlotDecoder (MESH OT loop + AdaSlot gate) → reject head
        """
        x0, H, pad_mask = self._encode_set(tokens, mask)
        B = tokens.size(0)

        # CoSA: broadcast genotype-conditioned slot init to batch
        geno_e    = self._encode_geno()                          # (K, d)
        geno_slots = geno_e.unsqueeze(0).expand(B, -1, -1)      # (B, K, d)

        # GSANet: attr-weighted peak aggregate → refine slot init.
        attr_raw = None
        if self.aux_heads:
            attr_raw = self.attr_head(H)                         # (B, N, n_classes+1) incl background

        # SOFT per-peak noise/stutter reliability gate (default off → peak_w None → decoder unchanged).
        peak_w = ng_logit = None
        if self.noise_gate:
            ng_logit = self.noise_gate_head(self._stutter_feats(tokens, mask)).squeeze(-1)   # (B,N)
            peak_w = (self.noise_s_floor + (1.0 - self.noise_s_floor) * torch.sigmoid(ng_logit)) * mask.float()

        # Pass the FULL attr logits (incl. background col) so GSANet softmaxes over K+1 then drops the
        # background mass inside AdaptiveSlotDecoder — artifact/stutter peaks aren't forced onto a donor slot.
        slot_out = self.cls_decoder_module(
            H, pad_mask, geno_slots, attr_logits=attr_raw, peak_w=peak_w)

        # Reject head: feas_filter on → pool an unfiltered, detached encode so out-of-panel
        # (unknown-contributor) alleles reach reject; else shared z (or dedicated pma_reject).
        z = self.pma(H, pad_mask=pad_mask).squeeze(1)            # (B, d)
        z_rej = self._reject_z(z, H, pad_mask, tokens, mask)

        # Inc10 — discriminability-weighted reference-allele match, added to the per-donor logit
        # (IDF / Robertson–Spärck Jones rarity weighting; theorem-optimal under term-independence).
        # ENCODER-BYPASS: _ref_match_score reads the RAW tokens, NOT H, so a faint minor washed out of
        # H can still be credited by its decisive panel-rare alleles. Identical additive term, scale,
        # and standardisation as the per_donor main-forward path (set_transformer.py:1926); the aslot
        # decoder returns early so it must be applied here. No self.training guard → train/infer parity.
        logits_cls = slot_out["logits_cls"]                      # (B, K)
        if self.ref_match:
            logits_cls = logits_cls + self._ref_match_score(tokens, mask)
        logits_cls, emphi_hill = self._apply_emphi(logits_cls, tokens, mask)   # em_phi_feature (no-op if off)

        out = {
            "logits_cls":  logits_cls,                           # (B, K)  (incl. ref_match / em_phi if on)
            "logit_reject": self.reject_head(z_rej),             # (B, 1)
            "logits_card": slot_out["logits_card"],              # (B, 5)  slot-gate count (unchanged)
            # §3.2 learned gate-count: surface the AdaSlot existence gate so the trainer can add the
            # differentiable Σgate≈NOC consistency loss (gradient → gate_head → slots → MESH → encoder)
            # and decode k=round(Σgate). BIT-IDENTICAL: these tensors already exist; adding dict keys
            # changes no computation and they are read only when --gate_count is on.
            "gate":        slot_out["gate"],                     # (B,K) concrete sample (train) / sigmoid (eval)
            "gate_logit":  slot_out["gate_logit"],               # (B,K) pre-noise existence logit
            "slot_mass":   slot_out["slot_mass"],                # (B,K) mass-preserving count signal (detached)
        }
        if self.noc_head_v2:                                     # paper-proof CORN count head → (B, 4)
            # feed the ref-matched/em-phi logits so the count head reads the SAME ranking as the oracle/decode
            out["logits_count_v2"] = self._ord_count_logits(
                logits_cls, tokens, mask, slot_out.get("slot_mass"), emphi_hill)
        if self.aux_heads:
            out["logits_attr"] = attr_raw                        # (B, N, K+1)
            out["phi"] = torch.nn.functional.softplus(
                self.phi_head(z))                                 # (B, K)
            if self.phi_gated:                                   # §3.4: per-donor log-variance for β-NLL
                out["phi_logvar"] = self.phi_logvar_head(z)      # (B, K)
        if self.lupi_phys:                                       # Inc-LUPI physical privileged outputs
            out["degr"]         = self.degr_head(z).squeeze(-1)  # (B,)   scaled degradation β
            _mv = self.muvar_head(H)                             # (B, N, 2)
            out["mu_mean"]      = _mv[..., 0]                    # (B, N) clean log1p-height
            out["mu_logvar"]    = _mv[..., 1]                    # (B, N) log-variance
            out["dropin_logit"] = self.dropin_head(H).squeeze(-1)  # (B, N) drop-in phantom logit
        if self.noise_gate:
            out["noise_gate_logit"] = ng_logit                   # (B, N) — supervised by real/noise label in trainer
        return out

    def _forward_spen(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        Forward for cls_decoder="spen" (EBM-SPEN joint decoder).

        Flow:
          encoder → z (PMA pool) → local_head → logits_0
          SPENDecoder T-step projected GD → y_T
          cardinality_head(y_T) → logits_card
          Returns logits_cls (from y_T), logits_cls_0 (initial), logits_card, logit_reject.
        """
        _x0, H, pad_mask = self._encode_set(tokens, mask)
        z = self.pma(H, pad_mask=pad_mask).squeeze(1)           # (B, d)
        z_rej = self._reject_z(z, H, pad_mask, tokens, mask)

        # Local head: initial logits (auxiliary BCE target, also warm-start for GD)
        local_logits = self.spen_local_head(z)                  # (B, K)

        # SPENDecoder: T-step inference in y-space
        y_T, y_T_graph = self.cls_decoder_module(z, local_logits)

        # logit transform of y_T for BCE loss (stable)
        eps    = 1e-6
        lg_T   = (torch.log(y_T_graph.clamp(eps, 1 - eps))
                  - torch.log((1 - y_T_graph).clamp(eps, 1 - eps)))  # (B, K)

        # Cardinality from refined soft predictions (CardinalityHead expects probs)
        logits_card = self.cardinality_head(y_T.detach())       # (B, n_card)

        out = {
            "logits_cls":   lg_T,             # SPEN-refined (in graph during train)
            "logits_cls_0": local_logits,     # initial local head — auxiliary BCE
            "logit_reject": self.reject_head(z_rej),
            "logits_card":  logits_card,
        }
        if self.aux_heads:
            out["logits_attr"] = self.attr_head(H)              # (B, N, K+1)
            out["phi"]         = torch.nn.functional.softplus(self.phi_head(z))
        return out

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        xflat: torch.Tensor | None = None,
        em_phi: torch.Tensor | None = None,   # Inc18 V1: (B, C) phi to inject into donor queries
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            logits_cls:    (B, 45)  — raw logits for BCE/ASL loss
            logit_reject:  (B, 1)   — raw logit for has-unknown BCE
            logits_noc:    (B, 6)   — raw logits for NOC cross-entropy

        xflat: (B, n_flat) aligned Xflat features — REQUIRED for cls_decoder="hybrid".
        """
        if self.sic_iters > 1:                               # Inc16 deep-unfolded SIC (reduces to base at init)
            return self._forward_sic(tokens, mask)
        if self.cls_decoder == "aslot":                      # adaptive slot (CoSA+GSANet+MESH+AdaSlot)
            return self._forward_aslot(tokens, mask)
        if self.cls_decoder == "spen":                       # EBM-SPEN joint energy-based decoder
            return self._forward_spen(tokens, mask)
        x0, H, pad_mask = self._encode_set(tokens, mask)
        z = self.pma(H, pad_mask=pad_mask).squeeze(1)        # (B, d_model)

        if self.cls_decoder in ("per_donor", "additive", "dsmil", "sos"):
            # decoder_source="raw": attend to pre-ISAB tokens (less entangled,
            # each donor query picks its own alleles like a per-donor classifier);
            # "encoded": attend to ISAB H (more context, more entanglement);
            # "local": position-wise encoded tokens (depth without cross-token mixing).
            if self.decoder_source == "raw":
                src = x0
            elif self.decoder_source == "local":
                src = self.local_encoder(x0, pad_mask=pad_mask)
            else:
                src = H
            if self.cls_decoder == "per_donor":
                gq = self._encode_geno() if self.geno_query else None      # Inc3 A
                logits_cls = self.cls_decoder_module(src, pad_mask=pad_mask, geno_query=gq,
                                                     phi_query=em_phi if self.phi_inject else None)  # Inc18 V1
                # Inc9 A4 (DN-DETR): extra train-time pass with NOISED genotype-anchored queries; the
                # decoder learns to recover the right identity from a perturbed anchor (stable supervision
                # for faint donors). Inference-inert. Needs geno_query for the anchor.
                if self.query_denoise and self.training and gq is not None:
                    gq_n = gq + torch.randn_like(gq) * self.qdn_noise
                    self._logits_dn = self.cls_decoder_module(src, pad_mask=pad_mask, geno_query=gq_n)
                else:
                    self._logits_dn = None
            elif self.cls_decoder == "sos":
                logits_cls = self.cls_decoder_module(src, pad_mask=pad_mask,
                                                     owner_lut=self.owner_lut,
                                                     tokens=tokens, aoff=self._AOFF)
            else:
                logits_cls = self.cls_decoder_module(src, pad_mask=pad_mask)
        elif self.cls_decoder == "hybrid":
            if xflat is None:
                raise ValueError("cls_decoder='hybrid' requires xflat input")
            flat_logits = self.flat_head(self.flat_norm(xflat))      # aligned base
            set_logits  = torch.tanh(self.set_adjust(z))             # bounded adjust
            logits_cls  = flat_logits + self.set_scale * set_logits
        else:
            logits_cls = self.cls_head(z)

        if self.ref_match:                                   # Inc10: add discriminability-weighted ref match
            logits_cls = logits_cls + self._ref_match_score(tokens, mask)
        logits_cls, _emphi_hill = self._apply_emphi(logits_cls, tokens, mask)   # em_phi_feature (no-op if off)

        # Reject head: feas_filter on → unfiltered detached pool (see _reject_z); else decoupled/shared.
        z_rej = self._reject_z(z, H, pad_mask, tokens, mask)

        # Cardinality head reads the DETACHED donor-prob profile (so its gradient
        # does not perturb the ranking; ranking is trained by the cls loss alone).
        logits_card = self.cardinality_head(torch.sigmoid(logits_cls).detach())

        out = {
            "logits_cls": logits_cls,
            "logit_reject": self.reject_head(z_rej),
            "logits_card": logits_card,            # (B,5) NOC k=1..5 (replaces logits_noc)
        }
        if self.noc_head_v2:                        # paper-proof CORN count head → (B,4); no slot-mass here
            out["logits_count_v2"] = self._ord_count_logits(logits_cls, tokens, mask, None, _emphi_hill)
        if self.return_encoded:                    # Inc8: expose H for the decorr/consistency reg
            out["encoded"] = H
        if self.donor_recon and getattr(self.cls_decoder_module, "last_reps", None) is not None:
            out["donor_reps"] = self.donor_recon_head(self.cls_decoder_module.last_reps)  # (B,45,d)
        if getattr(self, "_logits_dn", None) is not None:        # Inc9 A4: noised-query denoise logits
            out["logits_cls_dn"] = self._logits_dn
        if self.aux_heads:
            attr_logit = self.attr_head(H)                    # (B, N, n_classes+1) per-peak donor
            if self.soft_geno_attr and self.owner_lut is not None and not self.training:
                # Inc18 V2: genotype-constrained attr logits (INFERENCE only — training keeps unmasked CE
                # to handle stutter-peak label noise: stutter attr label = parent donor ≠ carrier of stutter allele).
                # Private peaks (1 carrier): HARD assign — unambiguous, no valid alternative.
                # Shared peaks (>1 carriers): SOFT mask — non-carriers → -inf, carriers compete normally.
                li = tokens[..., 0].long().clamp(0, 23)
                bi = (tokens[..., 1] * 10).round().long() + self._AOFF
                bi = bi.clamp(0, self.owner_lut.size(1) - 1)
                cmask = self.owner_lut[li, bi]                # (B, N, C) 1=carrier
                n_car = cmask.sum(-1, keepdim=True)           # (B, N, 1) #carriers per peak
                bg_z = torch.zeros(*cmask.shape[:-1], 1, dtype=torch.bool, device=cmask.device)
                carr_full = torch.cat([cmask.bool(), bg_z], dim=-1)   # (B, N, C+1)
                # Private: set carrier → +inf, all others → -inf
                is_priv = (n_car == 1)
                priv_exp = is_priv.expand_as(carr_full)
                attr_logit = torch.where(carr_full & priv_exp,
                                         torch.full_like(attr_logit, float("inf")), attr_logit)
                attr_logit = torch.where(~carr_full & priv_exp,
                                         torch.full_like(attr_logit, float("-inf")), attr_logit)
                # Shared: non-carriers → -inf (carriers keep learned logit)
                is_shar = (n_car > 1)
                bg_o = torch.ones(*cmask.shape[:-1], 1, device=cmask.device)
                full_mask = torch.cat([cmask, bg_o], dim=-1).bool()   # (B, N, C+1) 1=eligible
                shar_exp = is_shar.expand_as(full_mask)
                attr_logit = torch.where(~full_mask & shar_exp,
                                         torch.full_like(attr_logit, float("-inf")), attr_logit)
            out["logits_attr"] = attr_logit
            out["phi"] = torch.nn.functional.softplus(self.phi_head(z))  # (B, n_classes) ANC abundance
        if self.ml_attr:                                      # Inc14 B-v2: per-peak multi-label co-membership
            out["logits_mlattr"] = self.ml_attr_head(H)       # (B, N, n_classes) sigmoid logits
        if self.donor_contrast:                               # Inc3 B: per-peak proj for SupCon-by-donor
            out["z_peak"] = self.proj_peak(H)                 # (B, N, d_proj)
        if self.noc_ord_head:                                 # Inc3 V1-3: CORN count on its own encoder pool
            Hc = H.detach() if self.noc_ord_detach else H
            z_card = self.pma_card(Hc, pad_mask=pad_mask).squeeze(1)
            out["logits_card_ord"] = self.corn_head(z_card)   # (B, 4) CORN conditional logits
            if self.noc_contrast:                             # RNC shapes the pool the count head READS
                out["z_noc_proj"] = self.proj_noc(z_card)
        elif self.noc_contrast:
            # Own pool + projection (decoupled from z) — training-only signal for Rank-N-Contrast.
            # noc_detach (§4b-C valve 2/3): pool a DETACHED H so the contrast gradient cannot reach
            # the shared encoder — it then shapes only pma_noc/proj_noc (discarded at inference).
            H_noc = H.detach() if self.noc_detach else H
            z_noc = self.pma_noc(H_noc, pad_mask=pad_mask).squeeze(1)
            out["z_noc_proj"] = self.proj_noc(z_noc)          # (B, d_proj)
        return out
