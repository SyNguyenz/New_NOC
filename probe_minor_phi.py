"""
probe_minor_phi.py — EVAL-ONLY. The clean test for: "is low-NOC oracle caused by low-phi minor
donors being unrankable (phi-limited), OR by combinatorial generalization (combo-overfit)?"

Donor-level recall: for each TRUE donor d in a sample, is class d inside the model's top-NOC
predictions? Stratify recall by that donor's realized phi (mixture fraction), separately for
TRAIN-fit combos (seen) vs DEV combos (held-out, combo-disjoint, zero domain shift).

Decisive read:
  * If low-phi donors are recalled on TRAIN but NOT on DEV  -> combo-generalization, NOT phi.
  * If recall tracks phi identically on TRAIN and DEV        -> genuinely phi/info-limited.

DEV reconstructed seed=0 == make_dev_split (same as measure_insilico_oracle.py).

Usage: python probe_minor_phi.py <run_dir> [data_dir]
"""
import sys, json
from pathlib import Path
import numpy as np, torch

RUN  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/inc6_minorw_seed42")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8); tok_prefix = f"tokens{n_tok}" if n_tok > 3 else "tokens"
print(f"run={RUN}  data={DATA}  device={DEVICE}  n_tok={n_tok}")

# ── load FULL train + reconstruct dev mask seed=0 (== make_dev_split) ─────────────────────
tok = np.load(DATA / f"{tok_prefix}_train.npy").astype(np.float32)
msk = np.load(DATA / "mask_train.npy")
y   = np.load(DATA / "y_train_set.npy")
noc = np.load(DATA / "noc_train.npy").astype(int)
phi = np.load(DATA / "phi_train.npy").astype(np.float32)

def dev_mask_seed0(y, noc, combo_frac=0.15, noc1_frac=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc)
    m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx:
            combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * combo_frac)))]:
            m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * noc1_frac)), replace=False)] = True
    return m

dmask = dev_mask_seed0(y, noc)
print(f"reconstructed dev seed=0: {dmask.sum()} dev / {(~dmask).sum()} train-fit")

# ── model ────────────────────────────────────────────────────────────────────────────────
model = SetTransformerMixture(
    n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
    n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
    n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
    cls_decoder=cfg.get("cls_decoder", "pooled"), decoder_source=cfg.get("decoder_source", "encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder", "isab"), dec_layers=cfg.get("dec_layers", 2),
    num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
    periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
    d_proj=cfg.get("d_proj", 64), sparse_attn=cfg.get("sparse_attn", False),
).to(DEVICE)
sd = torch.load(RUN / "best_model.pt", map_location=DEVICE)
sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
model.load_state_dict(sd, strict=False); model.eval()

@torch.no_grad()
def probs(tok, msk):
    out = []
    for i in range(0, len(tok), 256):
        o = model(torch.from_numpy(tok[i:i+256]).to(DEVICE), torch.from_numpy(msk[i:i+256]).to(DEVICE))
        out.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
    return np.concatenate(out)

# ── donor-level records: (phi, recalled, noc, set) ───────────────────────────────────────
def collect(sel_mask, label, cap=12000):
    idx = np.where(sel_mask)[0]
    if len(idx) > cap:
        idx = np.random.default_rng(1).choice(idx, cap, replace=False)
    P = probs(tok[idx], msk[idx])
    nocc = np.clip(noc[idx], 1, 5)
    recs = []
    for j, gi in enumerate(idx):
        k = int(nocc[j]); top = set(np.argsort(P[j])[::-1][:k].tolist())
        for d in np.where(y[gi] == 1)[0]:
            recs.append((float(phi[gi, d]), int(d in top), k))
    a = np.array(recs, dtype=float)
    print(f"[{label}] {len(idx)} samples, {len(a)} donor instances")
    return a

print("\nforwarding (train-fit + dev) ...")
A_tr  = collect(~dmask, "TRAIN-fit")
A_dev = collect(dmask,  "DEV")

# ── recall vs phi bins, TRAIN vs DEV, overall + NOC4/NOC5 ────────────────────────────────
bins = [0.0, 0.10, 0.15, 0.20, 0.30, 1.01]
blab = ["<.10", ".10-.15", ".15-.20", ".20-.30", ">.30"]
def table(noc_filter=None):
    print(f"\n  phi-bin     TRAIN n   recall   |   DEV n   recall   | gap(tr-dev)")
    for b in range(len(bins) - 1):
        def cell(A):
            m = (A[:, 0] >= bins[b]) & (A[:, 0] < bins[b+1])
            if noc_filter is not None: m &= np.isin(A[:, 2], noc_filter)
            n = int(m.sum()); r = float(A[m, 1].mean()) if n else float("nan")
            return n, r
        ntr, rtr = cell(A_tr); ndv, rdv = cell(A_dev)
        gap = (rtr - rdv) if (ntr and ndv) else float("nan")
        print(f"  {blab[b]:9s}  {ntr:7d}   {rtr:6.3f}   |  {ndv:6d}   {rdv:6.3f}   |  {gap:+.3f}")

print("\n================ DONOR RECALL vs PHI — ALL NOC ================"); table()
print("\n================ DONOR RECALL vs PHI — NOC4 only ============="); table([4])
print("\n================ DONOR RECALL vs PHI — NOC5 only ============="); table([5])
print("\nread: low-phi bin recall high on TRAIN but low on DEV  => combo-generalization wall, not phi.")
