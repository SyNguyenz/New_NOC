"""
set_transformer.py — CLEAN, STANDALONE model for the `inc22_fixed_aslot` arm ONLY.

This is a faithful, verbatim EXTRACTION of the single code path that
`models/set_transformer.py` takes for the inc22_fixed_aslot configuration:

    cls_decoder = aslot        (CoSA + GSANet + MESH + AdaSlot)
    encoder     = isab++       with nc_attn = mab0  (SigmoidMABpp in mab0, MABpp in mab1)
    num_embed   = periodic     (Gorishniy 2022 PLR), periodic_sigma = 0.3
    n_token_feats = 8
    aux_heads   = True         (attr_head + phi_head; GSANet uses attr logits)
    set_of_set  = True         (private/shared split BEFORE the encoder)
    feas_filter = True         (drop 0-carrier peaks before the encoder; dedicated reject pool)
    n_slot_iters=3, ot_eps=0.05, ot_iters=5

The COUNT (NOC) that is DECODED is the post-hoc RandomForest on the prob profile
(posthoc_cardinality) — the CORN/noc_head_v2 output is never decoded (it collapses N3/N4,
measured ~0.21/0.17). The CORN head itself IS present and trained (`noc_head_v2=True`), because
the published arm trained with it: its inputs are detached so it cannot change the ranking, but
its gradient DOES enter the global clip_grad_norm_ denominator, so removing it silently enlarges
every other parameter's effective step.

Everything the inc22 path NEVER touches was removed (per_donor/additive/dsmil/sos/spen
decoders, geno_query, ref_match, em_phi_feature, noise_gate, soft_geno_attr, sparse_attn,
vib, mass_pool, vicreg, donor_recon, query_denoise, noc_contrast/noc_ord, sic, the unused
`cardinality_head`, the `hybrid`/`pooled` heads, the decoder_source raw/local branches, …).

BIT-IDENTICAL guarantee (verify_inc22.py): the parameter NAMES and the forward computation of the
RANKING heads (logits_cls / logits_card / logits_attr / phi / logit_reject) match the original
`SetTransformerMixture` for the inc22 config, so the published checkpoint
results/inc22_fixed_aslot_seed42/best_model.pt loads (strict=False — the checkpoint's extra keys are
only the unused `cardinality_head.*`) and produces byte-identical outputs/loss/grad — including
`logits_count_v2` and the `slot_mass` it reads.

Classes/functions copied verbatim from models/set_transformer.py:
  PeriodicNumEmbedding, SetNorm, MABpp, SigmoidMABpp, ISABpp, PMA, sinkhorn_log,
  AdaptiveSlotDecoder.  SetTransformerMixture below is the inc22-only slice of SetTransformerMixture.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Per-feature numerical embedding (Gorishniy 2022, PLR) ───────────────────
class PeriodicNumEmbedding(nn.Module):
    """Each standardized scalar x_j -> [sin(2pi c_j x_j), cos(2pi c_j x_j)] with learnable
    per-feature frequencies c_j ~ N(0, sigma), then a per-feature Linear + ReLU -> d_emb.
    Fixes the rotation-invariance bottleneck of feeding raw scalars into one shared Linear."""

    def __init__(self, n_num: int, d_emb: int = 8, n_freq: int = 8, sigma: float = 1.0):
        super().__init__()
        self.n_num = n_num
        self.d_emb = d_emb
        self.n_freq = n_freq
        self.coeffs = nn.Parameter(torch.randn(n_num, n_freq) * sigma)
        self.weight = nn.Parameter(torch.empty(n_num, 2 * n_freq, d_emb))
        self.bias = nn.Parameter(torch.zeros(n_num, d_emb))
        nn.init.trunc_normal_(self.weight, std=1.0 / math.sqrt(2 * n_freq))
        self.act = nn.ReLU(inplace=True)

    @property
    def out_dim(self) -> int:
        return self.n_num * self.d_emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = (2 * math.pi) * x.unsqueeze(-1) * self.coeffs        # (B, N, n_num, n_freq)
        per = torch.cat([torch.sin(v), torch.cos(v)], dim=-1)    # (B, N, n_num, 2*n_freq)
        emb = torch.einsum("bnfk,fkd->bnfd", per, self.weight) + self.bias
        emb = self.act(emb)
        return emb.flatten(-2)                                   # (B, N, n_num*d_emb)


# ── SetNorm (Zhang 2022) ────────────────────────────────────────────────────
class SetNorm(nn.Module):
    """Standardize each set with a SINGLE per-set scalar mean/var over all valid elements x
    features, then per-feature affine.  Masked: stats over valid elements only."""
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model)); self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        d = x.size(-1)
        if pad_mask is not None:
            v = (~pad_mask).unsqueeze(-1).to(x.dtype)
            n = v.sum(dim=(1, 2), keepdim=True).clamp(min=1) * d
            mu = (x * v).sum(dim=(1, 2), keepdim=True) / n
            var = (((x - mu) ** 2) * v).sum(dim=(1, 2), keepdim=True) / n
        else:
            mu = x.mean(dim=(1, 2), keepdim=True)
            var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(var + self.eps) * self.gamma + self.beta


# ── Clean-path pre-norm SetNorm attention blocks (Set Transformer++) ────────
class MABpp(nn.Module):
    """Clean-path, pre-norm SetNorm Multihead Attention Block.
      H   = X + Attn(SetNorm(X)?, SetNorm(Y), Y)        (clean residual)
      out = H + FFN(SetNorm(H))                          (pre-norm)
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
        # BERT-style float additive mask: bool True -> -1e4 added before softmax, so an all-masked
        # key row -> uniform 1/N instead of NaN (prevented in BOTH forward and backward).
        float_kv = kv_mask.float() * -1e4 if (kv_mask is not None and kv_mask.dtype == torch.bool) else kv_mask
        a, _ = self.attn(Xn, self.norm_kv(Y, kv_mask), Y, key_padding_mask=float_kv)
        H = X + a
        return H + self.ff(self.norm_h(H, q_mask))


class SigmoidMABpp(nn.Module):
    """MABpp with NON-COMPETITIVE SIGMOID cross-attention (Ramapuram et al. 2024, arXiv 2409.04431):
        Attn(X,Y) = sigma(QK^T/sqrt(d_h) + b) V,  b = -log(n_keys),  NO row-normalization.
    Drop-in for ISAB++'s mab0 so a faint MINOR peak is admitted to an inducing point regardless of
    how tall the MAJOR peaks are (de-absorbs the faint tail; F35b)."""
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
        self.bias = nn.Parameter(torch.zeros(1)) if learnable_bias else None

    def forward(self, X, Y, q_mask=None, kv_mask=None):
        Xn = self.norm_q(X, q_mask) if self.norm_q is not None else X
        Yn = self.norm_kv(Y, kv_mask)
        B, Lq, d = Xn.shape; Lk = Yn.size(1)
        q = self.wq(Xn).view(B, Lq, self.h, self.dh).transpose(1, 2)
        k = self.wk(Yn).view(B, Lk, self.h, self.dh).transpose(1, 2)
        v = self.wv(Y ).view(B, Lk, self.h, self.dh).transpose(1, 2)       # value from UNNORMED Y (= MABpp)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if kv_mask is not None:
            nval = (~kv_mask).sum(1).clamp(min=1).to(scores.dtype)
            b = -torch.log(nval).view(B, 1, 1, 1)
        else:
            b = -math.log(max(Lk, 1))
        if self.bias is not None:
            b = b + self.bias
        A = torch.sigmoid(scores + b)                                      # non-competitive gates
        if kv_mask is not None:
            A = A.masked_fill(kv_mask[:, None, None, :], 0.0)
        a = (self.drop(A) @ v).transpose(1, 2).reshape(B, Lq, d)
        a = self.wo(a)
        H = X + a
        return H + self.ff(self.norm_h(H, q_mask))


class ISABpp(nn.Module):
    """ISAB++ (SetNorm + clean-path residuals).  inc22 = nc_attn 'mab0': mab0 is the non-competitive
    SigmoidMABpp (de-absorb faint minors), mab1 the standard softmax MABpp."""
    def __init__(self, d_model: int, n_heads: int, m_inducing: int, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.empty(1, m_inducing, d_model)); nn.init.xavier_uniform_(self.I)
        self.mab0 = SigmoidMABpp(d_model, n_heads, dropout, norm_q=False, learnable_bias=False)  # inducing attend X
        self.mab1 = MABpp(d_model, n_heads, dropout, norm_q=True)                                  # X attend H

    def forward(self, X, pad_mask=None):
        I = self.I.expand(X.size(0), -1, -1)
        H = self.mab0(I, X, q_mask=None, kv_mask=pad_mask)           # (B,m,d)
        return self.mab1(X, H, q_mask=pad_mask, kv_mask=None)        # (B,N,d)


class PMA(nn.Module):
    """Pooling by Multihead Attention with a learned never-masked null key/value, so an all-masked
    set (e.g. fully out-of-panel after feas_filter) pools onto the null instead of NaN-ing."""
    def __init__(self, d_model: int, n_heads: int, k_seeds: int = 1, dropout: float = 0.0):
        super().__init__()
        self.S = nn.Parameter(torch.empty(1, k_seeds, d_model)); nn.init.xavier_uniform_(self.S)
        self.rff = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(inplace=True))
        self.mab = MAB_softmax(d_model, n_heads, dropout)
        self.null_kv = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, X: torch.Tensor, pad_mask: torch.Tensor | None = None) -> torch.Tensor:
        B = X.size(0)
        S = self.S.expand(B, -1, -1)
        Y = torch.cat([self.rff(X), self.null_kv.expand(B, -1, -1)], dim=1)
        if pad_mask is not None:
            null_col = torch.zeros(B, 1, dtype=torch.bool, device=pad_mask.device)
            pad_mask = torch.cat([pad_mask, null_col], dim=1)
        return self.mab(S, Y, key_padding_mask=pad_mask)


class MAB_softmax(nn.Module):
    """Plain LayerNorm MAB used only inside PMA (matches `MAB` in set_transformer.py; the PMA pool's
    `mab` submodule). Kept under this name purely for clarity — its parameter names (attn/ff/norm1/
    norm2) match the original so the checkpoint's `pma.mab.*` / `pma_reject.mab.*` keys load."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.ReLU(inplace=True),
                                nn.Linear(4 * d_model, d_model))
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, X, Y, key_padding_mask=None):
        attn_out, _ = self.attn(X, Y, Y, key_padding_mask=key_padding_mask)
        attn_out = attn_out.nan_to_num(0.0)
        H = self.norm1(X + attn_out)
        return self.norm2(H + self.ff(H))


# ── Sinkhorn OT (MESH, ICML2023) ────────────────────────────────────────────
def sinkhorn_log(affinity: torch.Tensor, pad_mask: torch.Tensor,
                 eps: float = 0.05, n_iters: int = 5) -> torch.Tensor:
    """Log-domain Sinkhorn: doubly-normalised (approx doubly-stochastic) peak->slot assignment so a
    shared peak splits fractionally across slots instead of winner-take-all (anti explaining-away)."""
    la = affinity / eps
    la = la.masked_fill(pad_mask.unsqueeze(1), -1e9)
    for _ in range(n_iters):
        la = la - torch.logsumexp(la, dim=1, keepdim=True)          # column-norm (over K slots per peak)
        la = la.masked_fill(pad_mask.unsqueeze(1), -1e9)
        la = la - torch.logsumexp(la, dim=2, keepdim=True)          # row-norm (over N peaks per slot)
        la = la.masked_fill(pad_mask.unsqueeze(1), -1e9)
    return torch.exp(la)


# ── Adaptive Slot decoder (CoSA + GSANet + MESH + AdaSlot) ──────────────────
class AdaptiveSlotDecoder(nn.Module):
    """Slots = panel donors (K=45).  CoSA geno init -> GSANet attr-guided refine -> MESH Sinkhorn-OT
    slot-attention loop -> AdaSlot Gumbel-Sigmoid existence gate.  logits_cls = cls_head(slots) +
    gate_logit; logits_card = noc_head(gate).  (The CORN slot-mass count signal is excluded — see header.)"""

    def __init__(self, d_model: int, n_classes: int = 45, n_noc: int = 5,
                 n_iters: int = 3, ot_eps: float = 0.05, ot_iters: int = 5,
                 gumbel_temp: float = 1.0, dropout: float = 0.1):
        super().__init__()
        self.n_iters = n_iters
        self.ot_eps = ot_eps
        self.ot_iters = ot_iters
        self.gumbel_temp = gumbel_temp
        self.scale = math.sqrt(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        self.gru = nn.GRUCell(d_model, d_model)
        self.slot_norm1 = nn.LayerNorm(d_model)
        self.slot_norm2 = nn.LayerNorm(d_model)
        self.slot_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
        )

        self.gsanet_proj = nn.Linear(d_model, d_model)
        self.gsanet_gate = nn.Linear(d_model, 1)

        self.gate_head = nn.Linear(d_model, 1)                              # AdaSlot slot-existence
        self.cls_head  = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))  # content
        self.noc_head = nn.Linear(n_classes, n_noc)

    def forward(self, H: torch.Tensor, pad_mask: torch.Tensor, geno_slots: torch.Tensor,
                attr_logits: torch.Tensor | None = None) -> dict:
        B, N, d = H.shape
        K = geno_slots.shape[1]

        # 1. CoSA: genotype-conditioned slot init
        slots = geno_slots.clone()

        # 2. GSANet: attr-weighted peak aggregate -> refine slot init
        if attr_logits is not None:
            attr_w = torch.softmax(attr_logits, dim=-1)                # (B, N, K[+1])
            attr_w = attr_w[..., :K]                                   # drop background col
            attr_w = attr_w * (~pad_mask).float().unsqueeze(-1)
            attr_agg = attr_w.transpose(1, 2) @ H                     # (B, K, d)
            attr_agg = self.gsanet_proj(attr_agg)
            g = torch.sigmoid(self.gsanet_gate(attr_agg))
            slots = slots + g * attr_agg

        # 3. MESH: iterative Sinkhorn-OT routing
        K_h = self.k_proj(H)
        V_h = self.v_proj(H)
        for _ in range(self.n_iters):
            Q = self.q_proj(self.slot_norm1(slots))
            affinity = (Q @ K_h.transpose(1, 2)) / self.scale          # (B, K, N)
            A = sinkhorn_log(affinity, pad_mask, eps=self.ot_eps, n_iters=self.ot_iters)
            updates = A @ V_h
            slots = self.gru(updates.reshape(B * K, d), slots.reshape(B * K, d)).reshape(B, K, d)
            slots = slots + self.slot_ffn(self.slot_norm2(slots))

        # Mass-preserving count signal (Fischer & Gärtner 2024, arXiv 2407.04170; GIN/DeepSets): the MESH
        # row-normalization (weighted mean) ERASES per-slot assignment mass, which a multiset counter
        # provably needs (sum is injective on multisets; mean/max are not). Recover it as the expected
        # #peaks assigned to each slot = column-softmax over slots, summed over peaks. Detached (read by
        # the CORN count head only; never perturbs the slots).  [verbatim: original decoder]
        with torch.no_grad():
            aff_f = (self.q_proj(self.slot_norm1(slots)) @ K_h.transpose(1, 2)) / self.scale  # (B,K,N)
            aff_f = aff_f.masked_fill(pad_mask.unsqueeze(1), -1e9)
            p2s = torch.softmax(aff_f, dim=1)                          # peak->slot over K slots
            pw = (~pad_mask).unsqueeze(1).to(p2s.dtype)
            slot_mass = (p2s * pw).sum(-1)                             # (B, K)

        # 4. AdaSlot: separate gate_head, concrete Bernoulli (Logistic noise = diff of two Gumbels)
        gate_logit = self.gate_head(slots).squeeze(-1)                 # (B, K)
        if self.training:
            u = torch.rand_like(gate_logit).clamp(1e-7, 1 - 1e-7)
            logistic = torch.log(u) - torch.log1p(-u)
            gate = torch.sigmoid((gate_logit + logistic) / self.gumbel_temp)
        else:
            gate = torch.sigmoid(gate_logit)

        # 5. logits_cls = content + existence in logit space
        cls_raw    = self.cls_head(slots).squeeze(-1)
        logits_cls = cls_raw + gate_logit
        logits_card = self.noc_head(gate)                              # (B, n_noc)

        return {
            "logits_cls":  logits_cls,
            "logits_card": logits_card,
            "gate":        gate,
            "gate_logit":  gate_logit,
            "slot_mass":   slot_mass,          # (B,K) detached; CORN count-head input only
            "attn":        A,
        }


# ── Full inc22 model (the inc22-only slice of SetTransformerMixture) ────────
class SetTransformerMixture(nn.Module):
    """`inc22_fixed_aslot` model.  Fixed architecture (no flag zoo).  Parameter names match the
    original SetTransformerMixture for the inc22 config so the published checkpoint loads (strict=False:
    the checkpoint keys absent here are the UNUSED `cardinality_head.*` and the EXCLUDED CORN count head
    `ord_count_head.*`).  CORN (noc_head_v2) is excluded by design — it collapses N3/N4 (measured); the
    count is post-hoc RandomForest on the prob profile."""

    _AOFF = 30   # round(allele*10)+OFF -> allele-bin index (matches the trainer's ALLELE_OFF)

    def __init__(self, n_loci: int = 24, d_locus: int = 16, d_model: int = 128, n_heads: int = 4,
                 n_isab: int = 2, m_inducing: int = 32, n_classes: int = 45, dropout: float = 0.1,
                 n_token_feats: int = 8, n_freq: int = 8, d_num_emb: int = 8, periodic_sigma: float = 0.3,
                 n_slot_iters: int = 3, ot_eps: float = 0.05, ot_iters: int = 5, gumbel_temp: float = 1.0,
                 donor_geno: torch.Tensor | None = None, donor_geno_mask: torch.Tensor | None = None,
                 owner_lut: torch.Tensor | None = None, noc_head_v2: bool = True):
        super().__init__()
        assert donor_geno is not None, "aslot needs donor_geno for CoSA slot init"
        assert owner_lut is not None, "inc22 (set_of_set/feas_filter) needs owner_lut"
        self.n_loci = n_loci
        self.d_model = d_model
        self.n_classes = n_classes
        self.n_token_feats = n_token_feats

        # owner_lut carrier LUT (set_of_set / feas_filter / count features) — registered buffer.
        self.register_buffer("owner_lut", owner_lut.float())           # (24, LUT_W, C)

        n_num = n_token_feats - 1
        self.locus_embed = nn.Embedding(n_loci + 1, d_locus, padding_idx=n_loci)
        self.register_buffer("feat_mean", torch.zeros(n_num))
        self.register_buffer("feat_std", torch.ones(n_num))
        self.num_embed = PeriodicNumEmbedding(n_num, d_num_emb, n_freq, periodic_sigma)
        proj_in = d_locus + self.num_embed.out_dim
        self.input_proj = nn.Linear(proj_in, d_model)

        # ISAB++ encoder, nc_attn = mab0
        self.encoder = nn.ModuleList(
            [ISABpp(d_model, n_heads, m_inducing, dropout) for _ in range(n_isab)])

        # pooling: shared pool (z) + dedicated reject pool (feas_filter keeps reject seeing out-of-panel)
        self.pma = PMA(d_model, n_heads, k_seeds=1, dropout=dropout)
        self.pma_reject = PMA(d_model, n_heads, k_seeds=1, dropout=dropout)

        # aslot decoder + CoSA genotype buffers/projection (registered exactly as in the original)
        self.cls_decoder_module = AdaptiveSlotDecoder(
            d_model, n_classes=n_classes, n_noc=5,
            n_iters=n_slot_iters, ot_eps=ot_eps, ot_iters=ot_iters,
            gumbel_temp=gumbel_temp, dropout=dropout,
        )
        self.register_buffer("donor_geno", donor_geno.float())
        self.register_buffer("donor_geno_mask", donor_geno_mask.bool())
        self.geno_proj = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.geno_proj.weight); nn.init.zeros_(self.geno_proj.bias)

        self.reject_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, 1))

        # privileged aux heads (in-silico only): per-peak attribution + phi abundance
        self.attr_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes + 1))
        self.phi_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(d_model, n_classes))

        # noc_head_v2 (the published inc22 arm's --noc_head_v2): CORN ordinal count head on a
        # multiset-COUNT input = sorted prob profile + MAC physical features + mass-preserving
        # slot_mass. ALL INPUTS DETACHED — it never perturbs the ranking, and decode never reads it.
        # It is kept because the published run TRAINED with it: its gradient enters the global
        # clip_grad_norm_ denominator (measured |g_corn| ~ |g_rest|, 90% of steps clipped), so dropping
        # it silently enlarges every other parameter's effective step.
        self.noc_head_v2 = noc_head_v2
        self._ocv2_topm = 8
        if noc_head_v2:
            _ocin = (self._ocv2_topm + 2) + 19 + self._ocv2_topm      # prob(10) + MAC(19) + mass(8)
            self.ord_count_head = nn.Sequential(
                nn.Linear(_ocin, 64), nn.ReLU(inplace=True),
                nn.Dropout(dropout), nn.Linear(64, 4))

    # ── token projection (with feas_filter) ──────────────────────────────────
    def _project_tokens(self, tokens, mask, apply_feas: bool = True):
        locus_idx = tokens[:, :, 0].long().clamp(0, self.n_loci - 1)
        continuous = tokens[:, :, 1:self.n_token_feats]
        continuous = (continuous - self.feat_mean) / self.feat_std
        locus_emb = self.locus_embed(locus_idx)
        num = self.num_embed(continuous)
        x = self.input_proj(torch.cat([locus_emb, num], dim=-1))
        x = x * mask.unsqueeze(-1)
        pad_mask = ~mask
        if apply_feas:                                                # feas_filter: drop 0-carrier peaks
            bi = (tokens[..., 1] * 10).round().long() + self._AOFF
            bi = bi.clamp(0, self.owner_lut.size(1) - 1)
            feas = self.owner_lut[locus_idx, bi].sum(-1) > 0
            x = x * feas.unsqueeze(-1).to(x.dtype)
            pad_mask = pad_mask | ~feas
        return x, pad_mask

    def _encode_geno(self) -> torch.Tensor:
        """CoSA: encode each donor's reference-allele set into a query anchor (C, d), through the SAME
        locus_embed + input_proj as peaks.  Non-allele continuous features neutralised to 0."""
        g = self.donor_geno
        locus_idx = g[:, :, 0].long().clamp(0, self.n_loci - 1)
        cont = (g[:, :, 1:self.n_token_feats] - self.feat_mean) / self.feat_std
        cont = cont.clone(); cont[:, :, 1:] = 0.0
        num = self.num_embed(cont)
        x = self.input_proj(torch.cat([self.locus_embed(locus_idx), num], dim=-1))
        m = self.donor_geno_mask.unsqueeze(-1).float()
        emb = (x * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.geno_proj(emb)                                    # zero-init -> 0 at start

    def _encode_set(self, tokens, mask):
        """Returns (x0, H, pad_mask).  set_of_set: split peaks into private (n_car==1) / shared
        (n_car!=1) BEFORE the encoder; each set passes the SAME ISAB++ independently, then merge."""
        x0, pad_mask = self._project_tokens(tokens, mask)
        li = tokens[..., 0].long().clamp(0, 23)
        bi = (tokens[..., 1] * 10).round().long() + self._AOFF
        bi = bi.clamp(0, self.owner_lut.size(1) - 1)
        n_car = self.owner_lut[li, bi].sum(-1)
        is_valid = ~pad_mask
        is_priv = (n_car == 1) & is_valid
        is_shar = (n_car != 1) & is_valid                             # n_car>1 (shared) + n_car==0 (OOD/unknown)
        priv_pm = ~is_priv
        shar_pm = ~is_shar
        H_priv = x0
        for isab in self.encoder:
            H_priv = isab(H_priv, pad_mask=priv_pm)
        H_priv = H_priv * is_priv.unsqueeze(-1).to(H_priv.dtype)
        H_shar = x0
        for isab in self.encoder:
            H_shar = isab(H_shar, pad_mask=shar_pm)
        H_shar = H_shar * is_shar.unsqueeze(-1).to(H_shar.dtype)
        H = H_priv + H_shar                                           # disjoint -> clean selection
        return x0, H, pad_mask

    def _reject_pool(self, tokens, mask):
        """Reject pool from an UNFILTERED, plain single-pass encode, DETACHED before pooling — so the
        open-set (out-of-panel) evidence feas_filter strips still reaches reject, and the reject
        gradient never touches the encoder or the cls pool."""
        x0u, pad_u = self._project_tokens(tokens, mask, apply_feas=False)
        Hu = x0u
        for isab in self.encoder:
            Hu = isab(Hu, pad_mask=pad_u)
        return self.pma_reject(Hu.detach(), pad_mask=pad_u).squeeze(1)

    def _mac_feats_torch(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """MAC/physical count features (PACE, Marciano & Adelman 2017; deepNoC 2024): per-locus allele
        counts at several RFU thresholds + global height stats. Fixed input feature (no grad). (B,19).
        [verbatim: original SetTransformerMixture._mac_feats_torch]"""
        with torch.no_grad():
            h = torch.expm1(tokens[:, :, 2])                           # (B,N) RFU
            loc = tokens[:, :, 0].long().clamp(0, self.n_loci - 1)
            mb = mask.bool(); B = tokens.size(0); feats = []
            for T in (0.0, 50.0, 150.0, 500.0):
                v = (mb & (h > T)).to(h.dtype)
                cnt = torch.zeros(B, self.n_loci, device=tokens.device).scatter_add_(1, loc, v)
                feats += [cnt.amax(1, keepdim=True),
                          (cnt >= 2).sum(1, keepdim=True).to(h.dtype),
                          (cnt >= 3).sum(1, keepdim=True).to(h.dtype),
                          v.sum(1, keepdim=True)]
            mf = mb.to(h.dtype); denom = mf.sum(1, keepdim=True).clamp(min=1)
            lg = torch.log1p(h.clamp(min=0)) * mf
            lmean = lg.sum(1, keepdim=True) / denom
            lvar = (((lg - lmean) ** 2) * mf).sum(1, keepdim=True) / denom
            feats += [lmean, torch.sqrt(lvar + 1e-6), lg.amax(1, keepdim=True)]
            return torch.cat(feats, dim=1)                             # (B, 19)

    def _ord_count_logits(self, logits_cls: torch.Tensor, tokens: torch.Tensor,
                          mask: torch.Tensor, slot_mass: torch.Tensor | None = None) -> torch.Tensor:
        """CORN ordinal logits (B,4 = K-1) over NOC 1..5 from a multiset-COUNT input = sorted prob
        profile + MAC features + mass-preserving slot_mass. ALL inputs DETACHED (the count head does
        not perturb the ranking/slots).  [verbatim: original _ord_count_logits, em_phi_feature off]"""
        m = self._ocv2_topm
        probs = torch.sigmoid(logits_cls).detach()
        s = torch.sort(probs, dim=1, descending=True).values[:, :m]
        pp = torch.cat([s, probs.sum(1, keepdim=True),
                        (probs >= 0.5).to(probs.dtype).sum(1, keepdim=True)], dim=1)   # (B, m+2)
        mac = self._mac_feats_torch(tokens, mask)                                      # (B, 19)
        if slot_mass is not None:
            sm = torch.sort(slot_mass.detach(), dim=1, descending=True).values[:, :m]   # (B, m)
        else:
            sm = probs.new_zeros(probs.size(0), m)
        return self.ord_count_head(torch.cat([pp, mac, sm], dim=1))                    # (B, 4)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """inc22 forward = the _forward_aslot path of SetTransformerMixture (CORN/noc_head_v2 excluded)."""
        x0, H, pad_mask = self._encode_set(tokens, mask)
        B = tokens.size(0)

        geno_e = self._encode_geno()                                  # (K, d)
        geno_slots = geno_e.unsqueeze(0).expand(B, -1, -1)            # (B, K, d)

        attr_raw = self.attr_head(H)                                  # (B, N, K+1) incl background (GSANet)
        slot_out = self.cls_decoder_module(H, pad_mask, geno_slots, attr_logits=attr_raw)

        z = self.pma(H, pad_mask=pad_mask).squeeze(1)
        z_rej = self._reject_pool(tokens, mask)                       # feas_filter: unfiltered + detached

        out = {
            "logits_cls":   slot_out["logits_cls"],
            "logit_reject": self.reject_head(z_rej),
            "logits_card":  slot_out["logits_card"],
            "logits_attr":  attr_raw,
            "phi":          F.softplus(self.phi_head(z)),
        }
        if self.noc_head_v2:
            out["logits_count_v2"] = self._ord_count_logits(
                slot_out["logits_cls"], tokens, mask, slot_out.get("slot_mass"))
        # permutation-invariant slot aggregates, surfaced for a count head. Both are detached in the
        # decoder, so exposing them cannot alter the ID path.
        out["gate"] = slot_out["gate"]
        out["slot_mass"] = slot_out["slot_mass"]
        return out
