"""
Decisive follow-up: is an N5 missed donor's signal PHYSICALLY PRESENT & DISTINCTIVE
in the raw observed peaks, or is it genuinely ambiguous (information ceiling)?

Uses donor reference genotypes (data/donor_geno.npy) to compute, per true donor:
  present  = ref alleles found among the observed peaks (locus,allele match)
  private  = present alleles NOT in ANY OTHER true donor of this mixture (panel-distinctive here)
and the height of the private peaks (readability vs dropout).

Compares MISSED minors (decoder top-5) vs CORRECTLY-IDENTIFIED minors.
If missed donors have many tall private alleles -> model (encoder) fails despite evidence.
If missed donors have ~0 private present alleles -> genuine ambiguity (no model fix possible).
"""
import json, numpy as np, torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data_insilico_w"
CKPT = ROOT / "results" / "inc6_maskp_seed42"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

def L(n): return np.load(DATA / f"{n}.npy", allow_pickle=True)
tokens = L("tokens8_test").astype(np.float32); mask = L("mask_test").astype(bool)
y = L("y_test_set").astype(np.float32); noc = L("noc_test").astype(int)
attr = L("attr_test").astype(int); phi = L("phi_test").astype(np.float32)
B = len(tokens)

g  = np.load(ROOT / "data" / "donor_geno.npy")            # (45,46,11) col0=locus col1=allele
gm = np.load(ROOT / "data" / "donor_geno_mask.npy").astype(bool)

def key(locus, allele):  # integer key for (locus, allele) matching
    return locus.astype(int) * 1000 + np.round(allele * 10).astype(int)

# per-donor reference allele key set
ref_keys = [set(key(g[c, gm[c], 0], g[c, gm[c], 1]).tolist()) for c in range(45)]

# build model + forward
cfg = json.load(open(CKPT / "metrics.json"))["config"]
model = SetTransformerMixture(
    n_loci=cfg["n_loci"], d_locus=cfg["d_locus"], d_model=cfg["d_model"],
    n_heads=cfg["n_heads"], n_isab=cfg["n_isab"], m_inducing=cfg["m_inducing"],
    n_classes=cfg["n_classes"], n_noc=cfg["n_noc"], dropout=cfg["dropout"],
    cls_decoder=cfg["cls_decoder"], n_token_feats=cfg["n_token_feats"],
    encoder=cfg["encoder"], num_embed=cfg["num_embed"],
    periodic_sigma=cfg["periodic_sigma"], aux_heads=cfg["aux_heads"],
    sparse_attn=cfg["sparse_attn"]).to(DEV)
model.load_state_dict(torch.load(CKPT / "best_model.pt", weights_only=True, map_location=DEV))
model.eval()

@torch.no_grad()
def fwd():
    P = []
    for s in range(0, B, 256):
        tk = torch.from_numpy(tokens[s:s+256]).to(DEV); mk = torch.from_numpy(mask[s:s+256]).to(DEV)
        _, H, pad = model._encode_set(tk, mk)
        P.append(torch.sigmoid(model.cls_decoder_module(H, pad_mask=pad)).cpu().numpy())
    return np.concatenate(P)
probs = fwd()

# ── per-sample observed peak keys + height map ────────────────────────────────
n5 = np.where(noc == 5)[0]
hgt = np.expm1(tokens[:, :, 2])           # height per peak

def analyse(i):
    v = mask[i]
    pk = key(tokens[i, v, 0], tokens[i, v, 1])
    h  = hgt[i, v]
    peakmax = {}                          # key -> max height observed
    for k_, hh in zip(pk.tolist(), h.tolist()):
        peakmax[k_] = max(peakmax.get(k_, 0.0), hh)
    obs = set(peakmax.keys())
    true = np.where(y[i] == 1)[0].tolist()
    rows = []
    for c in true:
        present = ref_keys[c] & obs
        others = set().union(*[ref_keys[o] for o in true if o != c])
        private = present - others                       # present & not in any other true donor
        priv_h = [peakmax[k_] for k_ in private]
        rows.append(dict(donor=c, phi=float(phi[i, c]),
                         n_ref=len(ref_keys[c]), n_present=len(present),
                         n_private=len(private),
                         priv_h_max=(max(priv_h) if priv_h else 0.0),
                         priv_h_tall=sum(hh > 100 for hh in priv_h)))   # >100 rfu = clearly readable
    return true, rows

# classify each true donor in each N5 sample as missed/hit by decoder top-5
miss, hit = [], []
for i in n5:
    top5 = set(np.argsort(probs[i])[::-1][:5].tolist())
    _, rows = analyse(i)
    for r in rows:
        r["i"] = int(i)
        (miss if r["donor"] not in top5 else hit).append(r)

def stats(rows, tag):
    a = lambda f: np.array([r[f] for r in rows], float)
    print(f"\n--- {tag} (n={len(rows)}) ---")
    print(f"  phi:           median={np.median(a('phi')):.3f}")
    print(f"  #present:      median={np.median(a('n_present')):.1f}  mean={a('n_present').mean():.1f}")
    print(f"  #PRIVATE:      median={np.median(a('n_private')):.1f}  mean={a('n_private').mean():.2f}  "
          f"(=0: {(a('n_private')==0).mean()*100:.0f}%  >=1: {(a('n_private')>=1).mean()*100:.0f}%  "
          f">=3: {(a('n_private')>=3).mean()*100:.0f}%)")
    print(f"  #PRIVATE tall(>100rfu): median={np.median(a('priv_h_tall')):.1f}  "
          f"(>=1 tall private: {(a('priv_h_tall')>=1).mean()*100:.0f}%)")
    print(f"  max private peak height: median={np.median(a('priv_h_max')):.0f} rfu")

print("="*70)
print("N5 true-donor evidence: MISSED vs CORRECTLY-IDENTIFIED (decoder top-5)")
print("="*70)
stats(hit,  "HIT minors+majors (in top-5)")
# restrict hit to minors (phi<0.2) for fair comparison
hit_minor = [r for r in hit if r["phi"] < 0.1999]
stats(hit_minor, "HIT minors only (phi<0.20)")
stats(miss, "MISSED donors (not in top-5)")

# recall as a function of #private present alleles
print("\n" + "="*70)
print("Decoder recall@top5 vs #PRIVATE present alleles (all N5 true donors)")
print("="*70)
allr = hit + miss
ishit = np.array([1]*len(hit) + [0]*len(miss))
npriv = np.array([r["n_private"] for r in allr])
for lo, hi in [(0,0),(1,2),(3,4),(5,99)]:
    m = (npriv >= lo) & (npriv <= hi)
    if m.sum(): print(f"  #private in [{lo}..{hi if hi<99 else '+'}]: recall={ishit[m].mean():.3f}  (n={m.sum()})")

# recall vs tall private alleles
print("\nDecoder recall@top5 vs #TALL(>100rfu) private alleles")
ntall = np.array([r["priv_h_tall"] for r in allr])
for lo, hi in [(0,0),(1,1),(2,2),(3,99)]:
    m = (ntall >= lo) & (ntall <= hi)
    if m.sum(): print(f"  #tall-private in [{lo}..{hi if hi<99 else '+'}]: recall={ishit[m].mean():.3f}  (n={m.sum()})")
