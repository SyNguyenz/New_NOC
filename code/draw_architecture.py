"""
draw_architecture.py — PAPER-STYLE top-down schematics of the `inc22_fixed_aslot` model
(SetTransformerMixture: ISAB++ encoder + CoSA/GSANet/MESH/AdaSlot decoder).

Style = like a paper figure: every box holds ONLY a block name (Linear, SetNorm, FFN,
GRU, Attention, Sinkhorn, ...). The *operation* of each block is DRAWN OUT with
operator nodes (+ add, x matmul, * gate, c concat), residual arrows and tensor shapes
on the edges — not written as sentences inside the boxes.

Tool: pure matplotlib (no Graphviz). Each figure → PNG + SVG.

    fig1_overview.{png,svg}        end-to-end graph
    fig2_embedding.{png,svg}       token projection (Embedding + Periodic-PLR + feas_filter)
    fig3_encoder.{png,svg}         set_of_set split + 2x ISAB++ + merge
    fig4_isab_block.{png,svg}      ISAB++ wiring + MAB internals (Sigmoid mab0 / Softmax mab1)
    fig5_decoder.{png,svg}         AdaptiveSlot: CoSA -> GSANet -> MESH -> AdaSlot -> heads
    fig6_mesh_iter.{png,svg}       one MESH Sinkhorn-OT slot-attention iteration
    fig7_decode_posthoc.{png,svg}  phi-rerank (EM + LOP) + RandomForest count + top-k

Run:  python inc22_clean/draw_architecture.py      Output dir: inc22_clean/diagrams/
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

# ── Palette ──────────────────────────────────────────────────────────────────
EDGE = "#37474F"
OP   = "#BBDEFB"   # generic operation block
CORE = "#FFB74D"   # signature op (SigmoidAttn / Sinkhorn / AdaSlot gate)
ENC  = "#C8E6C9"   # encoder-stage block
DEC  = "#FFE0B2"   # decoder-stage block
DATA = "#FAFAFA"   # data tensor node
OPC  = "#FFF176"   # operator circle
LW = 1.4
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diagrams")


# ── Primitives (all use CENTER-based coords: b = (cx, cy, w, h)) ─────────────
def blk(ax, cx, cy, name, w=0.15, h=0.05, fc=OP, fs=9.5, ls="-", lw=LW, bold=True):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.010", ec=EDGE, fc=fc, ls=ls, lw=lw, zorder=3))
    ax.text(cx, cy, name, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)
    return (cx, cy, w, h)


def dat(ax, cx, cy, name, w=0.16, h=0.046, fc=DATA, fs=8.6):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.022", ec="#90A4AE", fc=fc, lw=1.0, zorder=3))
    ax.text(cx, cy, name, ha="center", va="center", fontsize=fs, style="italic",
            color="#263238", zorder=4)
    return (cx, cy, w, h)


def op(ax, cx, cy, sym, r=0.0135, fc=OPC, fs=12):
    ax.add_patch(Circle((cx, cy), r, ec=EDGE, fc=fc, lw=1.3, zorder=5))
    ax.text(cx, cy, sym, ha="center", va="center", fontsize=fs, zorder=6)
    return (cx, cy, 2 * r, 2 * r)


def stack(ax, cx, cy, name, w=0.16, h=0.05, fc=ENC, fs=9.5, n="x2"):
    """A named block drawn as a small stack (shadow boxes) to denote repetition."""
    for k in (2, 1):
        d = 0.006 * k
        ax.add_patch(FancyBboxPatch((cx - w / 2 + d, cy - h / 2 - d), w, h,
            boxstyle="round,pad=0.004,rounding_size=0.010", ec=EDGE, fc=fc, lw=1.0, zorder=2, alpha=0.55))
    b = blk(ax, cx, cy, name, w, h, fc, fs)
    ax.text(cx + w / 2 - 0.012, cy + h / 2 - 0.012, n, ha="right", va="top",
            fontsize=7.6, fontweight="bold", color="#B71C1C", zorder=6)
    return b


def PT(b): cx, cy, w, h = b; return (cx, cy + h / 2)
def PB(b): cx, cy, w, h = b; return (cx, cy - h / 2)
def PL(b): cx, cy, w, h = b; return (cx - w / 2, cy)
def PR(b): cx, cy, w, h = b; return (cx + w / 2, cy)


def e(ax, p0, p1, shp="", color=EDGE, ls="-", rad=0.0, fs=7.4, dx=0.0, dy=0.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13, lw=LW,
        color=color, linestyle=ls, connectionstyle=f"arc3,rad={rad}", zorder=1, shrinkA=1.5, shrinkB=1.5))
    if shp:
        mx, my = (p0[0] + p1[0]) / 2 + dx, (p0[1] + p1[1]) / 2 + dy
        ax.text(mx, my, shp, ha="center", va="center", fontsize=fs, color="#546E7A",
                style="italic", zorder=4,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.9))


def via(ax, *pts, color=EDGE, ls="-", shp="", fs=7.4, dx=0.0, dy=0.0):
    """Multi-segment connector: straight lines through waypoints, arrowhead on last."""
    for i in range(len(pts) - 2):
        ax.add_patch(FancyArrowPatch(pts[i], pts[i + 1], arrowstyle="-", lw=LW, color=color,
            linestyle=ls, zorder=1, shrinkA=1.5, shrinkB=1.5))
    e(ax, pts[-2], pts[-1], shp=shp, color=color, ls=ls, fs=fs, dx=dx, dy=dy)


def ref(ax, b, txt):
    cx, cy, w, h = b
    ax.text(cx + w / 2 + 0.012, cy, txt, ha="left", va="center", fontsize=7.6,
            color="#1565C0", style="italic", zorder=5)


def new_ax(w_in, h_in, title, subtitle=""):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.992, title, ha="center", va="top", fontsize=15, fontweight="bold")
    if subtitle:
        ax.text(0.5, 0.957, subtitle, ha="center", va="top", fontsize=8.8, color="#455A64")
    return fig, ax


def save(fig, name):
    fig.savefig(os.path.join(OUTDIR, f"{name}.png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(OUTDIR, f"{name}.svg"), bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .svg")


def opkey(ax, x=0.012, y=0.20):
    items = [("+  add", OPC), ("x  matmul", OPC), ("*  gate (elem.)", OPC), ("c  concat", OPC)]
    ax.text(x, y + 0.035, "operators", fontsize=7.8, fontweight="bold", color="#37474F")
    for i, (t, _) in enumerate(items):
        ax.text(x, y - i * 0.024, t, fontsize=7.4, color="#37474F")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 1 — OVERVIEW (end-to-end graph)
# ═════════════════════════════════════════════════════════════════════════════
def fig1_overview():
    fig, ax = new_ax(8.6, 12.8, "Fig 1 — Architecture overview",
        "inc22_fixed_aslot  •  d=128, 4 heads, 2x ISAB++, K=45 slots")
    x = 0.40
    d_in = dat(ax, x, 0.905, "tokens, mask", w=0.22, h=0.05)
    b_emb = blk(ax, x, 0.826, "Embedding", w=0.24, h=0.052, fc=OP); ref(ax, b_emb, "Fig 2")
    b_enc = blk(ax, x, 0.730, "Encoder",   w=0.24, h=0.052, fc=ENC); ref(ax, b_enc, "Fig 3, 4")
    b_cosa = blk(ax, 0.80, 0.730, "CoSA", w=0.16, h=0.052, fc="#FFF59D")
    b_dec = blk(ax, x, 0.600, "Decoder",   w=0.24, h=0.052, fc=DEC); ref(ax, b_dec, "Fig 5–5b, 6")
    # heads
    b_h = blk(ax, 0.79, 0.480, "Aux heads", w=0.18, h=0.05, fc="#F8BBD0")
    d_out = dat(ax, x, 0.470, "model outputs", w=0.30, h=0.05)
    b_dec2 = blk(ax, x, 0.350, "Post-hoc decode", w=0.28, h=0.052, fc="#CFD8DC"); ref(ax, b_dec2, "Fig 7")
    d_pred = dat(ax, x, 0.255, "y (set) , k (NOC)", w=0.26, h=0.05)

    e(ax, PB(d_in), PT(b_emb), "B x N x 8")
    e(ax, PB(b_emb), PT(b_enc), "B x N x d")
    e(ax, PB(b_enc), PT(b_dec), "H : B x N x d")
    e(ax, PB(b_cosa), (0.80, 0.628), color="#F9A825")
    via(ax, (0.80, 0.628), (0.80, 0.600), PR(b_dec), color="#F9A825", shp="geno_slots", dy=0.012)
    # encoder H -> heads (routed down a clear column at x=0.64 so it does NOT cross CoSA)
    via(ax, PR(b_enc), (0.64, 0.730), (0.64, 0.480), PL(b_h), color="#AD1457", ls="--", shp="H", dy=0.012, dx=-0.02)
    e(ax, PB(b_dec), PT(d_out))
    via(ax, PB(b_h), (0.79, 0.470), PR(d_out), color="#AD1457", ls="--")
    e(ax, PB(d_out), PT(b_dec2))
    e(ax, PB(b_dec2), PT(d_pred))

    # output legend (what the dict holds)
    ax.text(0.40, 0.175, "outputs = logits_cls (B,45) · logits_card (B,5) · logit_reject (B,1) · logits_attr (B,N,46) · phi (B,45)",
            ha="center", fontsize=7.8, color="#455A64")
    save(fig, "fig1_overview")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 2 — EMBEDDING
# ═════════════════════════════════════════════════════════════════════════════
def fig2_embedding():
    fig, ax = new_ax(11.5, 8.2, "Fig 2 — Token embedding",
        "_project_tokens : each peak token (8 fields) -> d=128 vector; feas_filter masks infeasible peaks")
    d_tok = dat(ax, 0.50, 0.910, "tokens  (B, N, 8)", w=0.24, h=0.05)
    d_loc = dat(ax, 0.16, 0.790, "locus_idx", w=0.14, h=0.044)
    d_num = dat(ax, 0.60, 0.790, "fields 1..7", w=0.16, h=0.044)
    b_emb = blk(ax, 0.16, 0.690, "Embedding", w=0.16, h=0.05, fc=OP)
    b_std = blk(ax, 0.60, 0.690, "Standardize", w=0.16, h=0.05, fc=OP)
    b_plr = blk(ax, 0.60, 0.600, "Periodic-PLR", w=0.16, h=0.05, fc=CORE)
    o_cat = op(ax, 0.38, 0.500, "c")
    b_lin = blk(ax, 0.38, 0.420, "Linear", w=0.16, h=0.05, fc=OP)
    o_mul = op(ax, 0.38, 0.340, "*")
    d_mask = dat(ax, 0.10, 0.340, "mask", w=0.10, h=0.04)
    b_feas = blk(ax, 0.38, 0.255, "feas_filter", w=0.17, h=0.05, fc=CORE)
    d_own = dat(ax, 0.10, 0.255, "owner_lut", w=0.12, h=0.04)
    o_mul2 = op(ax, 0.38, 0.175, "*")
    d_out = dat(ax, 0.38, 0.095, "x (B,N,d), pad_mask", w=0.26, h=0.05)

    via(ax, PB(d_tok), (0.16, 0.855), PT(d_loc))
    via(ax, PB(d_tok), (0.60, 0.855), PT(d_num))
    e(ax, PB(d_loc), PT(b_emb))
    e(ax, PB(d_num), PT(b_std))
    e(ax, PB(b_std), PT(b_plr))
    via(ax, PB(b_emb), (0.16, 0.500), PL(o_cat), shp="16", dx=-0.02)
    via(ax, PB(b_plr), (0.60, 0.500), PR(o_cat), shp="56", dx=0.02)
    e(ax, PB(o_cat), PT(b_lin))
    e(ax, PB(b_lin), PT(o_mul))
    e(ax, PR(d_mask), PL(o_mul))
    e(ax, PB(o_mul), PT(b_feas))
    e(ax, PR(d_own), PL(b_feas))
    e(ax, PB(b_feas), PT(o_mul2), "gate")
    e(ax, PB(o_mul2), PT(d_out))

    # PLR "operation drawn out" callout (right side)
    ax.add_patch(FancyBboxPatch((0.80, 0.34), 0.185, 0.40, boxstyle="round,pad=0.006",
        ec="#90A4AE", fc="#FFFDE7", ls="--", lw=1.0, zorder=1))
    ax.text(0.8925, 0.715, "Periodic-PLR (per feature)", ha="center", fontsize=8.4, fontweight="bold")
    p1 = blk(ax, 0.8925, 0.650, "x 2pi c", w=0.10, h=0.04, fc=OP, fs=8.5)
    p2 = blk(ax, 0.8925, 0.560, "sin , cos", w=0.12, h=0.04, fc=OP, fs=8.5)
    p3 = blk(ax, 0.8925, 0.470, "Linear", w=0.10, h=0.04, fc=OP, fs=8.5)
    p4 = blk(ax, 0.8925, 0.385, "ReLU", w=0.10, h=0.04, fc=OP, fs=8.5)
    e(ax, PB(p1), PT(p2)); e(ax, PB(p2), PT(p3)); e(ax, PB(p3), PT(p4))
    ax.text(0.8925, 0.345, "c learnable ~ N(0, 0.3^2)", ha="center", fontsize=7.0, color="#546E7A")
    e(ax, PR(b_plr), (0.80, 0.600), color="#9E9E9E", ls=":")
    opkey(ax, 0.012, 0.20)
    save(fig, "fig2_embedding")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 3 — ENCODER
# ═════════════════════════════════════════════════════════════════════════════
def fig3_encoder():
    fig, ax = new_ax(11, 8.4, "Fig 3 — Encoder (set_of_set + 2x ISAB++)",
        "Private / shared peaks share the SAME ISAB++ weights but use separate masks, then are summed")
    d_x = dat(ax, 0.50, 0.910, "x (B,N,d), pad_mask", w=0.26, h=0.05)
    b_rt = blk(ax, 0.50, 0.810, "route by n_car", w=0.22, h=0.05, fc="#E1BEE7")
    # private branch
    s1 = stack(ax, 0.24, 0.680, "ISAB++", w=0.17, h=0.052, fc=ENC, n="x2")
    o1 = op(ax, 0.24, 0.580, "*")
    d_p = dat(ax, 0.07, 0.580, "1[private]", w=0.12, h=0.04)
    # shared branch
    s2 = stack(ax, 0.66, 0.680, "ISAB++", w=0.17, h=0.052, fc=ENC, n="x2")
    o2 = op(ax, 0.66, 0.580, "*")
    d_s = dat(ax, 0.84, 0.580, "1[shared]", w=0.12, h=0.04)
    o_add = op(ax, 0.45, 0.480, "+")
    d_H = dat(ax, 0.45, 0.390, "H (B,N,d)", w=0.18, h=0.05)
    ax.text(0.45, 0.330, "→ Decoder (Fig 5) & aux heads", ha="center", fontsize=8.4, color="#1565C0", style="italic")

    e(ax, PB(d_x), PT(b_rt))
    via(ax, PB(b_rt), (0.24, 0.760), PT(s1), shp="n_car==1", dy=0.012, dx=-0.03)
    via(ax, PB(b_rt), (0.66, 0.760), PT(s2), shp="n_car!=1", dy=0.012, dx=0.03)
    e(ax, PB(s1), PT(o1)); e(ax, PR(d_p), PL(o1))
    e(ax, PB(s2), PT(o2)); e(ax, PL(d_s), PR(o2))
    via(ax, PB(o1), (0.24, 0.480), PL(o_add))
    via(ax, PB(o2), (0.66, 0.480), PR(o_add))
    e(ax, PB(o_add), PT(d_H))

    # reject branch (parallel, dashed)
    ax.add_patch(FancyBboxPatch((0.18, 0.075), 0.64, 0.165, boxstyle="round,pad=0.006",
        ec="#AD1457", fc="#FCE4EC", ls="--", lw=1.0, zorder=0))
    ax.text(0.50, 0.222, "reject branch (parallel, _reject_pool)", ha="center", fontsize=8.6,
            fontweight="bold", color="#AD1457")
    rd = dat(ax, 0.235, 0.150, "x (no feas)", w=0.14, h=0.044)
    r1 = stack(ax, 0.42, 0.150, "ISAB++", w=0.13, h=0.046, fc=ENC, n="x2", fs=8.6)
    r2 = blk(ax, 0.585, 0.150, "detach", w=0.10, h=0.046, fc="#ECEFF1", fs=8.6)
    r3 = blk(ax, 0.73, 0.150, "PMA", w=0.10, h=0.046, fc="#F8BBD0", fs=8.6)
    e(ax, PR(rd), PL(r1)); e(ax, PR(r1), PL(r2)); e(ax, PR(r2), PL(r3))
    ax.text(0.73, 0.105, "logit_reject", ha="center", fontsize=7.6, style="italic", color="#AD1457")
    opkey(ax, 0.86, 0.74)
    save(fig, "fig3_encoder")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 4 — ISAB++ block + MAB internals
# ═════════════════════════════════════════════════════════════════════════════
def fig4_isab_block():
    fig, ax = new_ax(12, 8.6, "Fig 4 — ISAB++ block  &  MAB internals",
        "ISAB++ = SigmoidMAB (mab0, non-competitive) then MAB (mab1, softmax). Bottom: the shared MAB operation, drawn out.")
    # ── top wiring (ISAB++) ──
    ax.text(0.5, 0.905, "ISAB++ wiring", ha="center", fontsize=9.2, fontweight="bold", color="#2E7D32")
    d_I = dat(ax, 0.10, 0.840, "I (M=32, d)", w=0.15, h=0.046)
    d_X = dat(ax, 0.10, 0.745, "X (B,N,d)", w=0.15, h=0.046)
    b_m0 = blk(ax, 0.42, 0.800, "SigmoidMAB  (mab0)", w=0.26, h=0.06, fc=CORE, fs=9.5)
    b_m1 = blk(ax, 0.80, 0.800, "MAB  (mab1)", w=0.20, h=0.06, fc=ENC, fs=9.5)
    d_out = dat(ax, 0.80, 0.700, "out (B,N,d)", w=0.16, h=0.046)
    e(ax, PR(d_I), (0.29, 0.815), shp="query");
    e(ax, PR(d_X), (0.29, 0.785), shp="k,v", dy=-0.004)
    e(ax, PR(b_m0), PL(b_m1), "H_ind")
    via(ax, PB(d_X), (0.10, 0.700), (0.66, 0.700), (0.70, 0.770), color="#1B5E20", ls="--", shp="X query", dy=0.012)
    e(ax, PB(b_m1), PT(d_out))
    ax.plot([0.03, 0.97], [0.66, 0.66], color="#B0BEC5", lw=1.0, ls=":")

    # ── bottom: MAB internals (operation drawn out) ──
    ax.text(0.5, 0.625, "MAB(Xq, Y) internals  —  Attention = Sigmoid (mab0) / Softmax (mab1)",
            ha="center", fontsize=9.2, fontweight="bold", color="#1565C0")
    d_q = dat(ax, 0.20, 0.560, "Xq (query)", w=0.15, h=0.044)
    d_y = dat(ax, 0.62, 0.560, "Y (key/value)", w=0.16, h=0.044)
    n_q = blk(ax, 0.20, 0.470, "SetNorm", w=0.14, h=0.046, fc=OP, fs=8.8)
    n_k = blk(ax, 0.62, 0.470, "SetNorm", w=0.14, h=0.046, fc=OP, fs=8.8)
    b_at = blk(ax, 0.41, 0.370, "Attention", w=0.20, h=0.058, fc=CORE, fs=9.5)
    o_a1 = op(ax, 0.20, 0.275, "+")
    n_h = blk(ax, 0.20, 0.185, "SetNorm", w=0.14, h=0.046, fc=OP, fs=8.8)
    b_ff = blk(ax, 0.20, 0.100, "FFN", w=0.12, h=0.046, fc=OP, fs=8.8)
    o_a2 = op(ax, 0.20, 0.030, "+")
    d_o = dat(ax, 0.45, 0.030, "out", w=0.10, h=0.044)

    e(ax, PB(d_q), PT(n_q))
    e(ax, PB(d_y), PT(n_k))
    via(ax, PB(n_q), (0.20, 0.420), (0.33, 0.420), PT(b_at), shp="Q", dy=0.01)
    via(ax, PB(n_k), (0.62, 0.420), (0.49, 0.420), PT(b_at), shp="K,V", dy=0.01)
    # value from UN-normed Y (paper detail) - thin extra edge
    via(ax, PR(d_y), (0.74, 0.560), (0.74, 0.395), (0.51, 0.385), color="#8D6E63", ls=":", shp="V (raw Y)", dy=0.012)
    via(ax, PB(b_at), (0.41, 0.275), PR(o_a1), shp="a")
    via(ax, PB(d_q), (0.06, 0.560), (0.06, 0.275), PL(o_a1), color="#1B5E20", ls="--", shp="Xq", dy=0.0, dx=-0.005)
    e(ax, PB(o_a1), PT(n_h), "H")
    e(ax, PB(n_h), PT(b_ff))
    e(ax, PB(b_ff), PT(o_a2))
    via(ax, PL(o_a1), (0.10, 0.275), (0.10, 0.030), PL(o_a2), color="#1B5E20", ls="--", shp="H", dx=-0.004)
    e(ax, PR(o_a2), PL(d_o))
    opkey(ax, 0.86, 0.30)
    save(fig, "fig4_isab_block")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 5 — DECODER (AdaptiveSlot)
# ═════════════════════════════════════════════════════════════════════════════
def fig5_decoder():
    """Decoder OVERVIEW — named stages only; details in Fig 5a / 5b / 6."""
    fig, ax = new_ax(9, 11, "Fig 5 — Decoder overview (AdaptiveSlot)",
        "Slots = K=45 panel donors. Stages: CoSA init -> GSANet -> MESH (x3) -> AdaSlot -> heads")
    xc = 0.46
    d_H = dat(ax, 0.11, 0.905, "H", w=0.10, h=0.046)
    d_g = dat(ax, xc, 0.905, "geno_slots (B,K,d)", w=0.28, h=0.05)
    d_a = dat(ax, 0.86, 0.905, "attr_logits", w=0.18, h=0.046)
    b_init = blk(ax, xc, 0.790, "CoSA init", w=0.28, h=0.056, fc=DEC)
    b_gsa = blk(ax, xc, 0.660, "GSANet refine", w=0.28, h=0.056, fc=DEC); ref(ax, b_gsa, "Fig 5a")
    b_mesh = stack(ax, xc, 0.520, "MESH (Sinkhorn-OT)", w=0.30, h=0.062, fc=CORE, n="x3"); ref(ax, b_mesh, "Fig 6")
    b_ada = blk(ax, xc, 0.380, "AdaSlot gate", w=0.28, h=0.056, fc=DEC); ref(ax, b_ada, "Fig 5b")
    b_head = blk(ax, xc, 0.250, "output heads (cls_head / noc_head)", w=0.40, h=0.056, fc="#D1C4E9", fs=8.8)
    d_cls = dat(ax, 0.31, 0.130, "logits_cls (B,45)", w=0.24, h=0.05)
    d_noc = dat(ax, 0.66, 0.130, "logits_card (B,5)", w=0.22, h=0.05)

    e(ax, PB(d_g), PT(b_init), "S")
    e(ax, PB(b_init), PT(b_gsa), "S")
    e(ax, PB(b_gsa), PT(b_mesh), "S")
    e(ax, PB(b_mesh), PT(b_ada), "S")
    e(ax, PB(b_ada), PT(b_head), "gate, gate_logit")
    via(ax, PB(b_head), (xc, 0.205), (0.31, 0.205), PT(d_cls))
    via(ax, PB(b_head), (xc, 0.205), (0.66, 0.205), PT(d_noc))
    # side inputs (dashed)
    via(ax, PB(d_H), (0.11, 0.660), PL(b_gsa), color="#1B5E20", ls="--", shp="H", dx=-0.005)
    via(ax, (0.11, 0.660), (0.11, 0.520), PL(b_mesh), color="#1B5E20", ls="--", shp="H", dx=-0.005)
    via(ax, PB(d_a), (0.86, 0.715), (0.57, 0.715), (0.57, 0.688), color="#AD1457", ls="--", shp="attr", dy=0.012)
    save(fig, "fig5_decoder")


def fig5a_gsanet():
    """GSANet detail (operation drawn out)."""
    fig, ax = new_ax(11, 7.6, "Fig 5a — GSANet (attribution-guided slot refine)",
        "Aggregate peaks into each slot using attribution weights, gate the aggregate, add to slots")
    d_attr = dat(ax, 0.15, 0.860, "attr_logits (B,N,K+1)", w=0.24, h=0.05)
    d_H = dat(ax, 0.52, 0.860, "H (B,N,d)", w=0.18, h=0.05)
    d_S = dat(ax, 0.86, 0.860, "slots S (B,K,d)", w=0.20, h=0.05)
    b_sm = blk(ax, 0.15, 0.730, "softmax", w=0.13, h=0.05, fc=OP)
    d_al = dat(ax, 0.15, 0.620, "alpha[:K]  (drop bg)", w=0.22, h=0.046)
    o_mm = op(ax, 0.36, 0.500, "x")
    b_pr = blk(ax, 0.36, 0.380, "Linear (Wproj)", w=0.18, h=0.05, fc=OP)
    d_agg = dat(ax, 0.36, 0.275, "agg (B,K,d)", w=0.16, h=0.046)
    b_g = blk(ax, 0.62, 0.380, "Linear -> σ", w=0.16, h=0.05, fc=OP)
    o_ml = op(ax, 0.50, 0.175, "*")
    o_add = op(ax, 0.76, 0.175, "+")
    d_out = dat(ax, 0.76, 0.065, "S (refined)", w=0.18, h=0.05, fc=DEC)

    e(ax, PB(d_attr), PT(b_sm)); e(ax, PB(b_sm), PT(d_al))
    via(ax, PB(d_al), (0.15, 0.500), PL(o_mm), shp="alpha")
    via(ax, PB(d_H), (0.52, 0.500), PR(o_mm), shp="H")
    e(ax, PB(o_mm), PT(b_pr)); e(ax, PB(b_pr), PT(d_agg))
    via(ax, PB(d_agg), (0.36, 0.175), PL(o_ml), shp="agg", dx=-0.02)
    via(ax, PR(d_agg), (0.62, 0.275), PT(b_g), color="#9E9E9E", shp="agg")
    via(ax, PB(b_g), (0.62, 0.175), PR(o_ml), shp="g")
    e(ax, PR(o_ml), PL(o_add), "g*agg")
    via(ax, PB(d_S), (0.86, 0.175), PR(o_add), color="#6A1B9A", shp="S")
    e(ax, PB(o_add), PT(d_out))
    opkey(ax, 0.90, 0.74)
    save(fig, "fig5a_gsanet")


def fig5b_adaslot():
    """AdaSlot existence gate + output heads (operation drawn out)."""
    fig, ax = new_ax(11, 7.8, "Fig 5b — AdaSlot gate + output heads",
        "Existence gate via Gumbel-Sigmoid; logits_cls = cls_head(S) + gate_logit; logits_card = noc_head(gate)")
    d_S = dat(ax, 0.50, 0.890, "slots S (B,K,d)", w=0.22, h=0.05, fc=DEC)
    # gate column (left)
    b_lin = blk(ax, 0.26, 0.760, "Linear (gate_head)", w=0.20, h=0.05, fc=OP, fs=8.6)
    d_gl = dat(ax, 0.26, 0.650, "gate_logit (B,K)", w=0.20, h=0.046)
    o_add = op(ax, 0.26, 0.530, "+")
    d_noise = dat(ax, 0.07, 0.530, "Logistic noise", w=0.15, h=0.046)
    b_sig = blk(ax, 0.26, 0.410, "σ", w=0.09, h=0.05, fc=CORE, fs=11)
    d_gate = dat(ax, 0.26, 0.300, "gate (B,K)", w=0.15, h=0.046)
    b_noc = blk(ax, 0.26, 0.180, "noc_head", w=0.16, h=0.05, fc="#D1C4E9")
    d_card = dat(ax, 0.26, 0.070, "logits_card (B,5)", w=0.20, h=0.05)
    # cls column (right)
    b_cls = blk(ax, 0.68, 0.600, "cls_head", w=0.16, h=0.05, fc="#D1C4E9")
    o_cls = op(ax, 0.68, 0.430, "+")
    d_cls = dat(ax, 0.68, 0.300, "logits_cls (B,45)", w=0.22, h=0.05)

    via(ax, PB(d_S), (0.26, 0.830), PT(b_lin))
    via(ax, PB(d_S), (0.68, 0.830), (0.68, 0.700), PT(b_cls))
    e(ax, PB(b_lin), PT(d_gl)); e(ax, PB(d_gl), PT(o_add))
    e(ax, PR(d_noise), PL(o_add)); ax.text(0.07, 0.498, "train only", ha="center", fontsize=6.8, color="#90A4AE")
    e(ax, PB(o_add), PT(b_sig)); e(ax, PB(b_sig), PT(d_gate))
    e(ax, PB(d_gate), PT(b_noc), "gate"); e(ax, PB(b_noc), PT(d_card))
    e(ax, PB(b_cls), PT(o_cls))
    via(ax, PR(d_gl), (0.47, 0.650), (0.47, 0.430), PL(o_cls), color="#9E9E9E", shp="gate_logit", dy=0.012)
    e(ax, PB(o_cls), PT(d_cls))
    opkey(ax, 0.88, 0.74)
    save(fig, "fig5b_adaslot")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 6 — MESH single iteration
# ═════════════════════════════════════════════════════════════════════════════
def fig6_mesh_iter():
    """MESH slot-attention iteration — decluttered: hidden-state on the left, loop on the right,
    Sinkhorn shown as a block (its internals are Fig 6a)."""
    fig, ax = new_ax(11, 8.8, "Fig 6 — MESH: one Sinkhorn-OT iteration (x3)",
        "Slot attention with a doubly-normalized assignment so a shared peak splits across slots (anti explaining-away)")
    d_S = dat(ax, 0.36, 0.910, "slots  S (B,K,d)", w=0.22, h=0.05, fc=DEC)
    d_H = dat(ax, 0.80, 0.910, "H (B,N,d)", w=0.16, h=0.05)
    n_ln = blk(ax, 0.24, 0.795, "LayerNorm", w=0.15, h=0.05, fc=OP, fs=8.8)
    b_wq = blk(ax, 0.24, 0.690, "Linear Wq", w=0.15, h=0.05, fc=OP, fs=8.8)
    b_wk = blk(ax, 0.64, 0.795, "Linear Wk", w=0.15, h=0.05, fc=OP, fs=8.8)
    b_wv = blk(ax, 0.86, 0.795, "Linear Wv", w=0.15, h=0.05, fc=OP, fs=8.8)
    o_dot = op(ax, 0.44, 0.585, "x")
    b_sink = blk(ax, 0.44, 0.475, "Sinkhorn", w=0.18, h=0.058, fc=CORE, fs=10); ref(ax, b_sink, "Fig 6a")
    o_av = op(ax, 0.44, 0.370, "x")
    b_gru = blk(ax, 0.44, 0.265, "GRU", w=0.14, h=0.052, fc=DEC, fs=9.5)
    o_res = op(ax, 0.44, 0.160, "+")
    b_ff = blk(ax, 0.66, 0.160, "FFN ∘ LN", w=0.16, h=0.05, fc=OP, fs=8.8)
    d_out = dat(ax, 0.44, 0.055, "S (updated)", w=0.18, h=0.05, fc=DEC)

    via(ax, PB(d_S), (0.24, 0.850), PT(n_ln))
    e(ax, PB(n_ln), PT(b_wq))
    via(ax, PB(d_H), (0.64, 0.850), PT(b_wk))
    via(ax, PB(d_H), (0.86, 0.850), PT(b_wv))
    via(ax, PB(b_wq), (0.24, 0.585), PL(o_dot), shp="Q")
    via(ax, PB(b_wk), (0.64, 0.585), PR(o_dot), shp="K")
    e(ax, PB(o_dot), PT(b_sink), "affinity /sqrt d")
    e(ax, PB(b_sink), PT(o_av), "A")
    via(ax, PB(b_wv), (0.86, 0.370), PR(o_av), shp="V")
    e(ax, PB(o_av), PT(b_gru), "A·V")
    # hidden-state S -> GRU (left side, clear column at x=0.09)
    via(ax, PL(d_S), (0.09, 0.910), (0.09, 0.265), PL(b_gru), color="#6A1B9A", ls="--", shp="S (hidden)", dx=-0.005, dy=0.10)
    e(ax, PB(b_gru), PT(o_res))
    via(ax, PR(b_gru), (0.66, 0.265), PT(b_ff), color="#9E9E9E")
    e(ax, PL(b_ff), PR(o_res))
    e(ax, PB(o_res), PT(d_out))
    # loop-back (right side, clear column at x=0.95)
    via(ax, PR(d_out), (0.95, 0.055), (0.95, 0.910), PR(d_S), color="#6A1B9A")
    ax.text(0.965, 0.50, "loop x3", rotation=90, ha="center", va="center", fontsize=8.2,
            style="italic", color="#6A1B9A")
    opkey(ax, 0.04, 0.12)
    save(fig, "fig6_mesh_iter")


def fig6a_sinkhorn():
    """Sinkhorn (log-domain) detail — the doubly-normalized assignment, drawn out."""
    fig, ax = new_ax(9, 7.0, "Fig 6a — Sinkhorn (log-domain, doubly-normalized)",
        "Make the peak->slot assignment approx doubly-stochastic so a shared peak splits across slots")
    d_in = dat(ax, 0.50, 0.880, "affinity (B,K,N)", w=0.28, h=0.052)
    b_eps = blk(ax, 0.50, 0.760, "/ eps   (eps=0.05)", w=0.24, h=0.052, fc=OP)
    # iteration box
    ax.add_patch(FancyBboxPatch((0.18, 0.330), 0.64, 0.320, boxstyle="round,pad=0.008",
        ec="#90A4AE", fc="#FFF8E1", ls="--", lw=1.1, zorder=0))
    ax.text(0.74, 0.610, "x5", ha="center", fontsize=11, fontweight="bold", color="#B71C1C")
    b_col = blk(ax, 0.50, 0.560, "col-norm  =  L - logsumexp over K slots", w=0.56, h=0.058, fc=OP, fs=8.8)
    b_row = blk(ax, 0.50, 0.420, "row-norm  =  L - logsumexp over N peaks", w=0.56, h=0.058, fc=OP, fs=8.8)
    b_exp = blk(ax, 0.50, 0.230, "exp", w=0.14, h=0.052, fc=OP)
    d_A = dat(ax, 0.50, 0.110, "A in [0,1]  (B,K,N)", w=0.26, h=0.052)

    e(ax, PB(d_in), PT(b_eps), "B x K x N")
    e(ax, PB(b_eps), PT(b_col), "L")
    e(ax, PB(b_col), PT(b_row))
    e(ax, PB(b_row), PT(b_exp))
    e(ax, PB(b_exp), PT(d_A))
    ax.text(0.50, 0.300, "column-norm: per peak over the K slots   •   row-norm: per slot over the N peaks",
            ha="center", fontsize=7.6, style="italic", color="#546E7A")
    save(fig, "fig6a_sinkhorn")


# ═════════════════════════════════════════════════════════════════════════════
# FIG 7 — DECODE post-hoc
# ═════════════════════════════════════════════════════════════════════════════
def fig7_decode_posthoc():
    fig, ax = new_ax(12, 8.4, "Fig 7 — Post-hoc decode (inference)",
        "Two independent branches: phi-rerank fixes WHICH donors (ranking); RandomForest decides HOW MANY (count)")
    d_lg = dat(ax, 0.50, 0.910, "logits_cls (B,45)", w=0.22, h=0.05)

    # Branch A — ranking (left)
    ax.text(0.20, 0.860, "A · phi-rerank  (which donors)", ha="center", fontsize=9.5, fontweight="bold", color="#37474F")
    d_pk = dat(ax, 0.12, 0.795, "peaks + genotypes", w=0.20, h=0.046)
    b_em = stack(ax, 0.12, 0.700, "EM deconv", w=0.18, h=0.05, fc=CORE, n="x10", fs=9)
    d_phi = dat(ax, 0.12, 0.610, "phi (Mx)", w=0.13, h=0.044)
    b_lp = blk(ax, 0.12, 0.520, "log", w=0.09, h=0.044, fc=OP, fs=8.6)
    z1 = blk(ax, 0.12, 0.430, "z-norm", w=0.12, h=0.044, fc=OP, fs=8.6)
    z2 = blk(ax, 0.34, 0.430, "z-norm", w=0.12, h=0.044, fc=OP, fs=8.6)
    o_lop = op(ax, 0.23, 0.340, "+")
    d_sc = dat(ax, 0.23, 0.255, "score", w=0.12, h=0.046)
    e(ax, PB(d_pk), PT(b_em)); e(ax, PB(b_em), PT(d_phi))
    e(ax, PB(d_phi), PT(b_lp)); e(ax, PB(b_lp), PT(z1))
    via(ax, PL(d_lg), (0.34, 0.885), (0.34, 0.475), color="#9E9E9E", shp="logit", dy=0.0)
    via(ax, PB(z2), (0.34, 0.340), PR(o_lop))
    via(ax, PB(z1), (0.12, 0.340), PL(o_lop), shp="a·z(log phi)", dy=0.012, dx=-0.03)
    e(ax, PB(o_lop), PT(d_sc))
    ax.text(0.135, 0.300, "score = z(logit) + a·z(log phi)", ha="left", fontsize=7.2, style="italic", color="#546E7A")

    # Branch B — count (right)
    ax.text(0.80, 0.860, "B · count  (how many)", ha="center", fontsize=9.5, fontweight="bold", color="#37474F")
    b_sig = blk(ax, 0.80, 0.795, "σ", w=0.08, h=0.046, fc=OP, fs=10)
    d_P = dat(ax, 0.80, 0.705, "P", w=0.08, h=0.044)
    b_feat = blk(ax, 0.80, 0.610, "feature extract", w=0.21, h=0.05, fc=OP, fs=8.8)
    b_rf = blk(ax, 0.80, 0.505, "RandomForest", w=0.21, h=0.052, fc=CORE, fs=9.5)
    d_k = dat(ax, 0.80, 0.410, "k (NOC)", w=0.12, h=0.046)
    via(ax, PR(d_lg), (0.80, 0.910), PT(b_sig))
    e(ax, PB(b_sig), PT(d_P))
    e(ax, PB(d_P), PT(b_feat), "P")
    e(ax, PB(b_feat), PT(b_rf)); e(ax, PB(b_rf), PT(d_k))
    ax.text(0.985, 0.360, "fit on val (EM-optimal-k);", ha="right", fontsize=7.0, color="#90A4AE")
    ax.text(0.985, 0.343, "count on PROBS, not on the rerank", ha="right", fontsize=7.0, color="#90A4AE")

    # combine
    b_topk = blk(ax, 0.50, 0.245, "top-k", w=0.12, h=0.052, fc="#CFD8DC", fs=9.5)
    d_pred = dat(ax, 0.50, 0.150, "y (set)  ,  k (NOC)", w=0.24, h=0.052)
    via(ax, PB(d_sc), (0.23, 0.205), PL(b_topk), color=EDGE, shp="score", dy=0.012)
    via(ax, PL(d_k), (0.62, 0.410), (0.62, 0.300), PR(b_topk), color=EDGE, shp="k", dx=-0.02)
    e(ax, PB(b_topk), PT(d_pred))
    opkey(ax, 0.012, 0.22)
    save(fig, "fig7_decode_posthoc")


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    print(f"Drawing diagrams -> {OUTDIR}")
    fig1_overview()
    fig2_embedding()
    fig3_encoder()
    fig4_isab_block()
    fig5_decoder()
    fig5a_gsanet()
    fig5b_adaslot()
    fig6_mesh_iter()
    fig6a_sinkhorn()
    fig7_decode_posthoc()
    print("Done.")


if __name__ == "__main__":
    main()
