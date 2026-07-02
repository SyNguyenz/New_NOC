"""
probe_encoder_inc9.py — EVAL-ONLY encoder dissection for the Inc9 M2 encoder arms.

Goal: the M2 levers (B1 donor_recon, B4 attn_sink) were meant to make the ENCODER better
(de-smooth / anti-absorption per F33). This probes each part of the encoder FLOW on a saved
checkpoint and contrasts SYNTHETIC (combo-disjoint dev) vs REAL test, per NOC, so we can tell:
  - did the encoder actually de-smooth?  (H token over-smoothing)
  - can faint minors keep their private peaks readable?  (attr top-1 on the faintest donor's peaks)
  - are per-donor identities separated or collapsed?  (last_reps cosine among present donors)
  - is the final ranking healthy?  (present-vs-absent prob margin)
  - (B4) is the sink actually used?  (attention mass dumped on the null sink key)

For B1 the decisive read is SYNTH-vs-REAL: if the encoder geometry is fine on synthetic but
degrades on real, the recon aux overfit the encoder to the synthetic domain (encoder fault);
if geometry looks the same yet the margin/ranking collapses, the fault is downstream.

Usage:  python probe_encoder_inc9.py [run_dir ...]      (defaults: base, B1, B4)
"""
import os, sys, json, types, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from models.set_transformer import SetTransformerMixture, SparseMAB

DATA = Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CAP = 400          # max samples per NOC per split (geometry probe, keeps 4GB GPU happy)
RUNS = sys.argv[1:] or [
    "results/inc2_2d_sparse_seed42",   # base (same config, no B-lever)
    "results/inc9_b1_recon_seed42",    # B1 donor-recon
    "results/inc9_b4_sink_seed42",     # B4 attn-sink
]
print(f"device={DEVICE}  cap/NOC={CAP}")


# ── data ────────────────────────────────────────────────────────────────────────────────
def load(split, prefix="tokens8"):
    tok = np.load(DATA / f"{prefix}_{split}.npy").astype(np.float32)
    msk = np.load(DATA / f"mask_{split}.npy")
    y   = np.load(DATA / f"y_{split}_set.npy")
    noc = np.load(DATA / f"noc_{split}.npy").astype(int)
    attr = np.load(DATA / f"attr_{split}.npy")
    phi  = np.load(DATA / f"phi_{split}.npy").astype(np.float32)
    return dict(tok=tok, msk=msk, y=y, noc=noc, attr=attr, phi=phi)


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


def cap_by_noc(d, cap=CAP, seed=1):
    rng = np.random.default_rng(seed); keep = []
    for k in range(1, 6):
        idx = np.where(np.clip(d["noc"], 1, 5) == k)[0]
        if len(idx) > cap:
            idx = rng.choice(idx, cap, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep)
    return {kk: vv[keep] for kk, vv in d.items()}


full = load("train")
dm = dev_mask_seed0(full["y"], full["noc"])
SYNTH = cap_by_noc({k: v[dm] for k, v in full.items()})
REAL = cap_by_noc(load("test"))
del full
ATTR_BG = int(np.load(DATA / "attr_train.npy").max())   # background/none code (== n_classes)
print(f"SYNTH dev n={len(SYNTH['noc'])}  REAL test n={len(REAL['noc'])}  attr background code={ATTR_BG}")


# ── per-run probe ─────────────────────────────────────────────────────────────────────────
def build(cfg):
    n_tok = cfg.get("n_token_feats", 8)
    m = SetTransformerMixture(
        n_loci=cfg.get("n_loci", 24), d_locus=cfg.get("d_locus", 16), d_model=cfg.get("d_model", 128),
        n_heads=cfg.get("n_heads", 4), n_isab=cfg.get("n_isab", 2), m_inducing=cfg.get("m_inducing", 32),
        n_classes=cfg.get("n_classes", 45), n_noc=cfg.get("n_noc", 6), dropout=cfg.get("dropout", 0.1),
        cls_decoder=cfg.get("cls_decoder", "per_donor"), decoder_source=cfg.get("decoder_source", "encoded"),
        n_token_feats=n_tok, encoder=cfg.get("encoder", "isab"), dec_layers=cfg.get("dec_layers", 2),
        num_embed=cfg.get("num_embed", "raw"), n_freq=cfg.get("n_freq", 8), d_num_emb=cfg.get("d_num_emb", 8),
        periodic_sigma=cfg.get("periodic_sigma", 1.0), aux_heads=cfg.get("aux_heads", False),
        sparse_attn=cfg.get("sparse_attn", False),
        attn_sink=int(cfg.get("attn_sink", 0) or 0),
        donor_recon=cfg.get("donor_recon", False),
    ).to(DEVICE)
    return m, n_tok


def patch_sink_capture(model):
    """Monkeypatch the decoder's SparseMAB layers to stash their sparsemax attention (B,h,Nq,Nk)."""
    from models.set_transformer import sparsemax
    layers = [l for l in model.cls_decoder_module.layers if isinstance(l, SparseMAB)]
    def fwd(self, X, Y, key_padding_mask=None):
        B, Nq, _ = X.shape; Nk = Y.size(1)
        q = self.q(X).view(B, Nq, self.h, self.dh).transpose(1, 2)
        k = self.k(Y).view(B, Nk, self.h, self.dh).transpose(1, 2)
        v = self.v(Y).view(B, Nk, self.h, self.dh).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], -1e4)
        attn = sparsemax(scores, dim=-1)
        self._last_attn = attn.detach()
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, Nq, -1)
        H = self.norm1(X + self.o(out))
        return self.norm2(H + self.ff(H))
    for l in layers:
        l.forward = types.MethodType(fwd, l)
    return layers


def mean_pairwise_cos(X, valid):
    """mean off-diagonal cosine among `valid` rows of X (n,d). high => over-smoothed/collapsed."""
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return np.nan
    V = X[idx]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    C = V @ V.T
    n = len(idx)
    return float((C.sum() - n) / (n * (n - 1)))


@torch.no_grad()
def probe(run):
    cfg = json.load(open(Path(run) / "metrics.json"))["config"]
    model, n_tok = build(cfg)
    sd = torch.load(Path(run) / "best_model.pt", map_location=DEVICE)
    sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd else sd
    model.load_state_dict(sd, strict=False); model.eval()
    n_sink = int(cfg.get("attn_sink", 0) or 0)
    sink_layers = patch_sink_capture(model) if (n_sink > 0 and cfg.get("sparse_attn")) else []

    # hooks for x0 (input of encoder[0]) and H (output of encoder[-1])
    cap = {}
    h1 = model.encoder[0].register_forward_pre_hook(lambda m, a: cap.__setitem__("x0", a[0].detach()))
    h2 = model.encoder[-1].register_forward_hook(lambda m, a, o: cap.__setitem__("H", o.detach()))

    def run_split(d):
        recs = []
        tok, msk, y, attr, phi, noc = d["tok"], d["msk"], d["y"], d["attr"], d["phi"], np.clip(d["noc"],1,5)
        for s in range(0, len(tok), 128):
            T = torch.from_numpy(tok[s:s+128]).to(DEVICE); M = torch.from_numpy(msk[s:s+128]).to(DEVICE)
            o = model(T, M)
            H = cap["H"].cpu().numpy(); X0 = cap["x0"].cpu().numpy()
            reps = model.cls_decoder_module.last_reps.detach().cpu().numpy()   # (b,45,d)
            prob = torch.sigmoid(o["logits_cls"]).cpu().numpy()               # (b,45)
            attr_pred = o["logits_attr"].argmax(-1).cpu().numpy() if "logits_attr" in o else None
            # sink mass per donor query (mean over decoder layers & heads), last n_sink keys
            sink = None
            if sink_layers:
                am = np.mean([l._last_attn.cpu().numpy() for l in sink_layers], axis=0)  # (b,h,45,Nk)
                sink = am[..., -n_sink:].sum(-1).mean(1)                                  # (b,45)
            mm = msk[s:s+128].astype(bool)
            for j in range(len(H)):
                valid = mm[j]
                pres = np.where(y[s+j] == 1)[0]
                if len(pres) == 0:
                    continue
                ph = phi[s+j][pres]
                fmin = pres[int(np.argmin(ph))]                # faintest present donor
                ap = attr[s+j]
                rec = dict(
                    noc=int(noc[s+j]),
                    Hcos=mean_pairwise_cos(H[j], valid),
                    X0cos=mean_pairwise_cos(X0[j], valid),
                    repcos=mean_pairwise_cos(reps[j], y[s+j] == 1),
                    margin=float(prob[j][pres].mean() - prob[j][np.where(y[s+j] == 0)[0]].mean()),
                    fmin_prob=float(prob[j][fmin]),
                )
                if attr_pred is not None:
                    pk = valid & (ap < ATTR_BG)                # peaks assigned to a real donor
                    pkf = valid & (ap == fmin)                 # faint-minor private peaks
                    rec["attr_all"] = float((attr_pred[j][pk] == ap[pk]).mean()) if pk.any() else np.nan
                    rec["attr_fmin"] = float((attr_pred[j][pkf] == fmin).mean()) if pkf.any() else np.nan
                if sink is not None:
                    rec["sink_pres"] = float(sink[j][pres].mean())
                    rec["sink_fmin"] = float(sink[j][fmin])
                recs.append(rec)
        return recs

    res = {"SYNTH": run_split(SYNTH), "REAL": run_split(REAL)}
    h1.remove(); h2.remove()
    return cfg, n_sink, res


def agg(recs, key, k):
    v = [r[key] for r in recs if r["noc"] == k and key in r and not (isinstance(r[key], float) and math.isnan(r[key]))]
    return float(np.mean(v)) if v else float("nan")


# ── run + report ────────────────────────────────────────────────────────────────────────
ALL = {}
for run in RUNS:
    name = Path(run).name
    print(f"\n>>> {name}")
    ALL[name] = probe(run)

METRICS = [
    ("Hcos",     "H token cosine (over-smoothing; lower=de-smoothed)"),
    ("repcos",   "present-donor rep cosine (identity collapse; lower=separated)"),
    ("attr_all", "attr top-1 on present peaks (encoder readability)"),
    ("attr_fmin","attr top-1 on FAINT-minor private peaks (absorption)"),
    ("fmin_prob","faint-minor donor prob (final score)"),
    ("margin",   "present-vs-absent prob margin"),
    ("sink_pres","sink mass / present query  (B4 only)"),
    ("sink_fmin","sink mass / faint-minor query (B4 only)"),
]
for split in ("SYNTH", "REAL"):
    print(f"\n================== {split} ==================")
    hdr = f"{'metric':<46}" + "".join(f"{n[:18]:>20}" for n in [Path(r).name for r in RUNS])
    for key, desc in METRICS:
        # skip sink rows for runs w/o sink unless any run has it
        if key.startswith("sink") and not any(ALL[Path(r).name][1] > 0 for r in RUNS):
            continue
        print(f"\n  -- {desc}")
        print("    " + f"{'NOC':<6}" + "".join(f"{Path(r).name[-14:]:>18}" for r in RUNS))
        for k in range(1, 6):
            row = f"    {k:<6}"
            for r in RUNS:
                _, _, res = ALL[Path(r).name]
                row += f"{agg(res[split], key, k):>18.3f}"
            print(row)
print("\ndone.")
