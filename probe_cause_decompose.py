"""
probe_cause_decompose.py — EVAL-ONLY. CAREFUL root-cause separation for the low-phi N5 miss.

Combines the model-INDEPENDENT info classification (measure_noc5_ceiling logic) with the model's
actual per-donor recall, on TRAIN-fit (seen combos) vs DEV (held-out combos, seed=0 == make_dev_split).

Each TRUE contributor X in a sample is classified by PRESENCE alone (physics, no model):
  RANKABLE : X has a PRIVATE allele present in the peaks  -> info IS there; a miss = MODEL failure.
  DROPOUT  : X has private alleles but NONE present       -> distinguishing peak didn't amplify (info gone).
  MASKED   : X has NO private allele (covered by others)  -> only height can separate (info weak).

Decisive cross-tab (per NOC, per phi bin):  recall  ×  {RANKABLE, DROPOUT, MASKED}, on TRAIN vs DEV.
  * DEV: of the MISSED low-phi minors, what fraction are RANKABLE?  high => H1 combo-generalization (fixable).
                                                                     low  => H2 info-limited dropout (not fixable by model).
  * TRAIN: recall of DROPOUT/MASKED donors. ~1.0 => memorization leakage (model can't physically rank them).

Usage: python probe_cause_decompose.py <run_dir> [data_dir]
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, torch

RUN  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/inc6_minorw_seed42")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

META = json.load(open(DATA / "meta_set.json"))
KNOWN = META["known_donors"]; LOCUS_TO_IDX = META["locus_to_idx"]
cfg = json.load(open(RUN / "metrics.json"))["config"]
n_tok = cfg.get("n_token_feats", 8); tok_prefix = f"tokens{n_tok}" if n_tok > 3 else "tokens"

def allele_key(v):
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None"): return ""
    if s == "X": return "-2.0"
    if s == "Y": return "-1.0"
    try: return f"{round(float(s),1):.1f}"
    except ValueError: return ""

def load_raw_genotypes():
    f = glob.glob("data_raw/**/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx", recursive=True)[0]
    df = pd.read_excel(f, sheet_name=0)
    loci_cols = [c for c in df.columns if c in LOCUS_TO_IDX]
    geno = {}
    for _, r in df.iterrows():
        try: d = int(r["Sample ID"])
        except (ValueError, TypeError): continue
        if d not in KNOWN: continue
        g = {}
        for loc in loci_cols:
            cell = r[loc]
            if pd.isna(cell): continue
            ks = {allele_key(a) for a in str(cell).split(",")}; ks.discard("")
            if ks: g[LOCUS_TO_IDX[loc]] = ks
        geno[d] = g
    print(f"raw genotypes: {len(geno)} known donors")
    return geno

def obs_set(tok, mask):
    obs = set(); li = tok[:, 0].astype(int); al = tok[:, 1]
    for j in np.where(mask)[0]:
        a = float(al[j]); ak = "-2.0" if a == -2.0 else ("-1.0" if a == -1.0 else f"{round(a,1):.1f}")
        obs.add((int(li[j]), ak))
    return obs

def classify(X_int, true_ints, geno, obs):
    gX = geno.get(X_int, {})
    if not gX: return "NO_GENO"
    others = [geno.get(o, {}) for o in true_ints if o != X_int]
    private = set()
    for L, alleles in gX.items():
        oh = set().union(*[o.get(L, set()) for o in others]) if others else set()
        for a in alleles:
            if a not in oh: private.add((L, a))
    if not private: return "MASKED"
    return "RANKABLE" if (private & obs) else "DROPOUT"

# ── data + dev split seed=0 ──────────────────────────────────────────────────────────────
tok = np.load(DATA / f"{tok_prefix}_train.npy").astype(np.float32)
msk = np.load(DATA / "mask_train.npy"); ymat = np.load(DATA / "y_train_set.npy")
noc = np.load(DATA / "noc_train.npy").astype(int); phi = np.load(DATA / "phi_train.npy").astype(np.float32)

def dev_mask_seed0(y, noc, combo_frac=0.15, noc1_frac=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int), 1, 5); N = len(noc); m = np.zeros(N, bool)
    for k in [2, 3, 4, 5]:
        idx = np.where(noc == k)[0]; combos = {}
        for i in idx: combos.setdefault(tuple(np.where(y[i] == 1)[0].tolist()), []).append(i)
        uniq = list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * combo_frac)))]: m[combos[c]] = True
    idx1 = np.where(noc == 1)[0]
    m[rng.choice(idx1, size=int(round(len(idx1) * noc1_frac)), replace=False)] = True
    return m
dmask = dev_mask_seed0(ymat, noc)
geno = load_raw_genotypes()

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
def probs(t, m):
    out = []
    for i in range(0, len(t), 256):
        o = model(torch.from_numpy(t[i:i+256]).to(DEVICE), torch.from_numpy(m[i:i+256]).to(DEVICE))
        out.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
    return np.concatenate(out)

def run(sel, label, noc_keep=(4, 5), cap=8000):
    idx = np.where(sel & np.isin(noc, noc_keep))[0]
    if len(idx) > cap: idx = np.random.default_rng(1).choice(idx, cap, replace=False)
    P = probs(tok[idx], msk[idx])
    # records: (cat, recalled, phi, noc)
    recs = []
    for j, gi in enumerate(idx):
        k = int(np.clip(noc[gi], 1, 5)); top = set(np.argsort(P[j])[::-1][:k].tolist())
        obs = obs_set(tok[gi], msk[gi]); true_cols = np.where(ymat[gi] == 1)[0]
        true_ints = [KNOWN[c] for c in true_cols]
        for c in true_cols:
            cat = classify(KNOWN[c], true_ints, geno, obs)
            recs.append((cat, int(c in top), float(phi[gi, c]), k))
    print(f"[{label}] {len(idx)} samples, {len(recs)} donor instances")
    return recs

print("\nforwarding ...")
R_tr  = run(~dmask, "TRAIN-fit")
R_dev = run(dmask,  "DEV")

CATS = ["RANKABLE", "DROPOUT", "MASKED", "NO_GENO"]
def summarize(R, label, noc):
    sub = [r for r in R if r[3] == noc]
    print(f"\n==== {label}  NOC{noc}  (n_donor={len(sub)}) ====")
    print(f"  {'category':10s} {'n':>6} {'%donors':>8} {'recall':>8} | low-phi(<.10): {'n':>5} {'recall':>7}")
    for cat in CATS:
        cc = [r for r in sub if r[0] == cat]
        if not cc: continue
        rec = np.mean([r[1] for r in cc])
        lp = [r for r in cc if r[2] < 0.10]
        lpr = np.mean([r[1] for r in lp]) if lp else float("nan")
        print(f"  {cat:10s} {len(cc):>6} {100*len(cc)/len(sub):7.1f}% {rec:8.3f} | "
              f"{len(lp):>15} {lpr:7.3f}")
    # of the MISSED donors, composition by category (the key question)
    missed = [r for r in sub if r[1] == 0]
    if missed:
        print(f"  -- of {len(missed)} MISSED donors: " + ", ".join(
            f"{cat} {100*sum(1 for r in missed if r[0]==cat)/len(missed):.0f}%" for cat in CATS
            if any(r[0] == cat for r in missed)))

for noc in (4, 5):
    summarize(R_tr,  "TRAIN-fit", noc)
    summarize(R_dev, "DEV",       noc)

print("\nKEY READ:")
print(" * DEV missed donors mostly RANKABLE  => H1 combo-generalization (info present, model fails -> fixable).")
print(" * DEV missed donors mostly DROPOUT/MASKED => H2 info-limited (physics -> NOT fixable by model).")
print(" * TRAIN recall of DROPOUT ~1.0 => memorization leakage (train 0.99 is not real ranking).")
