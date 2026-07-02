"""
probe_encoder_minor.py — EVAL-ONLY. Decisive separation for the user's question:
  is the deep-miss (true minor ranked >8 by the decoder) caused by the DECODER ranking
  (info preserved in encoder H but mis-scored) or the ENCODER washing out the minor's signal?

Method: the model's attr_head reads per-peak donor identity DIRECTLY from encoded tokens H (a shallow
linear on H -> it reflects what H preserves). For each TRUE donor d in a NOC5 sample, find d's PRIVATE
allele peaks that are PHYSICALLY PRESENT (from reference genotype, model-independent), and read what
attr_head says at exactly those peak positions. Stratify by the decoder's rank of d:
    recalled (rank<=5) | near-miss (6-8) | deep-miss (>8)

Read:
  * deep-miss minors STILL have high private-peak attr accuracy  => encoder H PRESERVES the minor;
        the decoder ranking loses it -> fix the DECODER (0.83 is NOT an encoder limit).
  * deep-miss minors have LOW private-peak attr accuracy         => encoder washed the minor out
        (context-mixing) -> the loss is upstream in the ENCODER.

Usage: python probe_encoder_minor.py [run] [data]
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, torch

RUN  = Path("results") / (sys.argv[1] if len(sys.argv) > 1 else "inc6_minorw_seed42")
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("data_insilico_w")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture

META = json.load(open(DATA / "meta_set.json")); KNOWN = META["known_donors"]; LOCUS_TO_IDX = META["locus_to_idx"]
cfg = json.load(open(RUN / "metrics.json"))["config"]; n_tok = cfg.get("n_token_feats", 8); tp = f"tokens{n_tok}"

def akey(v):
    s = str(v).strip()
    if s in ("","nan","NaN","None"): return ""
    if s == "X": return "-2.0"
    if s == "Y": return "-1.0"
    try: return f"{round(float(s),1):.1f}"
    except ValueError: return ""

def load_geno():
    f = glob.glob("data_raw/**/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx", recursive=True)[0]
    df = pd.read_excel(f, sheet_name=0); cols = [c for c in df.columns if c in LOCUS_TO_IDX]; g = {}
    for _, r in df.iterrows():
        try: d = int(r["Sample ID"])
        except (ValueError, TypeError): continue
        if d not in KNOWN: continue
        gg = {}
        for loc in cols:
            if pd.isna(r[loc]): continue
            ks = {akey(a) for a in str(r[loc]).split(",")}; ks.discard("")
            if ks: gg[LOCUS_TO_IDX[loc]] = ks
        g[d] = gg
    return g

tok = np.load(DATA / f"{tp}_train.npy").astype(np.float32); msk = np.load(DATA / "mask_train.npy")
ymat = np.load(DATA / "y_train_set.npy").astype(np.float32); noc = np.load(DATA / "noc_train.npy").astype(int)
phi = np.load(DATA / "phi_train.npy").astype(np.float32)

def dev_mask_seed0(y, noc, combo_frac=0.15, noc1_frac=0.06, seed=0):
    rng = np.random.default_rng(seed); noc = np.clip(noc.astype(int),1,5); N=len(noc); m=np.zeros(N,bool)
    for k in [2,3,4,5]:
        idx=np.where(noc==k)[0]; combos={}
        for i in idx: combos.setdefault(tuple(np.where(y[i]==1)[0].tolist()),[]).append(i)
        u=list(combos); rng.shuffle(u)
        for c in u[:max(1,int(round(len(u)*combo_frac)))]: m[combos[c]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*noc1_frac)),replace=False)]=True
    return m
dv_idx = np.where(dev_mask_seed0(ymat, noc))[0]
geno = load_geno()

model = SetTransformerMixture(
    n_loci=cfg.get("n_loci",24), d_locus=cfg.get("d_locus",16), d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4), n_isab=cfg.get("n_isab",2), m_inducing=cfg.get("m_inducing",32),
    n_classes=cfg.get("n_classes",45), n_noc=cfg.get("n_noc",6), dropout=cfg.get("dropout",0.1),
    cls_decoder=cfg.get("cls_decoder","pooled"), decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok, encoder=cfg.get("encoder","isab"), dec_layers=cfg.get("dec_layers",2),
    num_embed=cfg.get("num_embed","raw"), n_freq=cfg.get("n_freq",8), d_num_emb=cfg.get("d_num_emb",8),
    periodic_sigma=cfg.get("periodic_sigma",1.0), aux_heads=True,
    d_proj=cfg.get("d_proj",64), sparse_attn=cfg.get("sparse_attn",False),
).to(DEVICE)
sd = torch.load(RUN/"best_model.pt", map_location=DEVICE); sd = sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
model.load_state_dict(sd, strict=False); model.eval()

@torch.no_grad()
def fwd(idx, bs=256):
    PP, AT = [], []
    for i in range(0,len(idx),bs):
        b=idx[i:i+bs]
        o=model(torch.from_numpy(tok[b]).to(DEVICE), torch.from_numpy(msk[b]).to(DEVICE))
        PP.append(torch.sigmoid(o["logits_cls"]).cpu().numpy())
        AT.append(torch.softmax(o["logits_attr"],dim=-1).cpu().numpy())   # (B,N,46)
    return np.concatenate(PP), np.concatenate(AT)

sel5 = dv_idx[noc[dv_idx]==5]
P, AT = fwd(sel5)

# buckets by decoder rank of each true donor; for each, attr accuracy on the donor's PRIVATE present peaks
buckets = {"recalled(<=5)": [], "near-miss(6-8)": [], "deep-miss(>8)": []}
def bucket(r): return "recalled(<=5)" if r<=5 else ("near-miss(6-8)" if r<=8 else "deep-miss(>8)")

for s, gi in enumerate(sel5):
    li = tok[gi][:,0].astype(int); al = tok[gi][:,1]; valid = msk[gi].astype(bool)
    # map (locus, allele_key) -> token positions present
    pos = {}
    for j in np.where(valid)[0]:
        a=float(al[j]); ak="-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
        pos.setdefault((int(li[j]),ak),[]).append(j)
    true_cols = np.where(ymat[gi]==1)[0]; true_ints=[KNOWN[c] for c in true_cols]
    order = np.argsort(P[s])[::-1]; rankof={d:r+1 for r,d in enumerate(order)}
    for c in true_cols:
        X=KNOWN[c]; gX=geno.get(X,{})
        if not gX: continue
        others=[geno.get(o,{}) for o in true_ints if o!=X]
        # private present peaks of X
        priv_pos=[]
        for L,alleles in gX.items():
            oh=set().union(*[o.get(L,set()) for o in others]) if others else set()
            for a in alleles:
                if a not in oh and (L,a) in pos: priv_pos += pos[(L,a)]
        if not priv_pos: continue
        # attr_head accuracy at those peaks: argmax==c, and prob mass on c
        acc=np.mean([AT[s,j].argmax()==c for j in priv_pos])
        prob=np.mean([AT[s,j,c] for j in priv_pos])
        buckets[bucket(rankof[c])].append((acc, prob, float(phi[gi,c]), rankof[c]))

print(f"run={RUN.name}  (NOC5 dev, n_samples={len(sel5)})\n")
print("== ENCODER preservation of the minor: attr_head on the minor's OWN private present peaks ==")
print(f"  {'decoder-rank bucket':18s} {'n_donors':>9} {'attr-acc':>9} {'attr-prob(d)':>12} {'mean-phi':>9} {'med-rank':>9}")
for name, rows in buckets.items():
    if not rows: continue
    a=np.array(rows)
    print(f"  {name:18s} {len(rows):>9} {a[:,0].mean():>9.3f} {a[:,1].mean():>12.3f} {a[:,2].mean():>9.3f} {np.median(a[:,3]):>9.0f}")
print("\nREAD: deep-miss attr-acc HIGH (~recalled) => encoder PRESERVES minor, decoder loses it (fix DECODER).")
print("      deep-miss attr-acc LOW              => encoder WASHES OUT minor (loss is upstream in ENCODER).")
