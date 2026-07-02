"""
probe_context.py — EVAL-ONLY. Answers two questions:

Q1  WHY does the encoder lose the minor — is it mixing with "context", and what IS the context?
    COUNTERFACTUAL: for each deep-miss minor X, re-run the SAME frozen encoder but mask the input down
    to ONLY X's own genotype alleles (drop the major-donor peaks = remove the context). Read attr_head
    at X's PRIVATE peaks, full vs context-removed.
      * attr-acc JUMPS when major peaks removed => the encoder washes the faint minor by MIXING it with
        the major-dominated peak context (ISAB inducing-point summary is major-heavy). "Context" = the
        other contributors' PEAKS (height-dominant majors), not the combo label abstractly.
      * attr-acc stays low alone => the peak is intrinsically uninformative (not a mixing artifact).

Q2  Is the decoder actually BAD, or fair given ~0.67/0.83?  Decompose the NOC5 oracle gap and compare
    the decoder against the honest linear-H ceiling and the top-8 competition ceiling; for near-miss,
    measure whether the displacing decoy is an allele-shared "explained-away" donor.

Usage: python probe_context.py [run] [data]
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
    df = pd.read_excel(f, sheet_name=0); cols=[c for c in df.columns if c in LOCUS_TO_IDX]; g={}
    for _, r in df.iterrows():
        try: d=int(r["Sample ID"])
        except (ValueError,TypeError): continue
        if d not in KNOWN: continue
        gg={}
        for loc in cols:
            if pd.isna(r[loc]): continue
            ks={akey(a) for a in str(r[loc]).split(",")}; ks.discard("")
            if ks: gg[LOCUS_TO_IDX[loc]]=ks
        g[d]=gg
    return g

tok=np.load(DATA/f"{tp}_train.npy").astype(np.float32); msk=np.load(DATA/"mask_train.npy")
ymat=np.load(DATA/"y_train_set.npy").astype(np.float32); noc=np.load(DATA/"noc_train.npy").astype(int)
phi=np.load(DATA/"phi_train.npy").astype(np.float32)

def dev_mask_seed0(y,noc,combo_frac=0.15,noc1_frac=0.06,seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc.astype(int),1,5); N=len(noc); m=np.zeros(N,bool)
    for k in [2,3,4,5]:
        idx=np.where(noc==k)[0]; combos={}
        for i in idx: combos.setdefault(tuple(np.where(y[i]==1)[0].tolist()),[]).append(i)
        u=list(combos); rng.shuffle(u)
        for c in u[:max(1,int(round(len(u)*combo_frac)))]: m[combos[c]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*noc1_frac)),replace=False)]=True
    return m
dv_idx=np.where(dev_mask_seed0(ymat,noc))[0]; geno=load_geno()

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
sd=torch.load(RUN/"best_model.pt",map_location=DEVICE); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
model.load_state_dict(sd,strict=False); model.eval()

@torch.no_grad()
def fwd(toks, masks, bs=256):
    PP, AT = [], []
    for i in range(0,len(toks),bs):
        o=model(torch.from_numpy(toks[i:i+bs]).to(DEVICE), torch.from_numpy(masks[i:i+bs]).to(DEVICE))
        PP.append(torch.sigmoid(o["logits_cls"]).cpu().numpy()); AT.append(torch.softmax(o["logits_attr"],-1).cpu().numpy())
    return np.concatenate(PP), np.concatenate(AT)

sel5=dv_idx[noc[dv_idx]==5]
Pfull, ATfull = fwd(tok[sel5], msk[sel5])

# build per-(sample,deep-miss-minor) rows with context-removed masks (keep only X's own alleles)
rows_tok, rows_mask, meta = [], [], []
def posmap(gi):
    li=tok[gi][:,0].astype(int); al=tok[gi][:,1]; valid=msk[gi].astype(bool); pos={}
    for j in np.where(valid)[0]:
        a=float(al[j]); ak="-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
        pos.setdefault((int(li[j]),ak),[]).append(j)
    return pos

for s,gi in enumerate(sel5):
    pos=posmap(gi); true_cols=np.where(ymat[gi]==1)[0]; true_ints=[KNOWN[c] for c in true_cols]
    order=np.argsort(Pfull[s])[::-1]; rankof={d:r+1 for r,d in enumerate(order)}
    for c in true_cols:
        if rankof[c]<=8: continue                       # deep-miss only
        X=KNOWN[c]; gX=geno.get(X,{})
        if not gX: continue
        others=[geno.get(o,{}) for o in true_ints if o!=X]
        priv=[]
        for L,alls in gX.items():
            oh=set().union(*[o.get(L,set()) for o in others]) if others else set()
            for a in alls:
                if a not in oh and (L,a) in pos: priv += pos[(L,a)]
        if not priv: continue
        # context-removed mask: keep only tokens whose (locus,allele) in X's genotype
        keep=np.zeros(tok.shape[1],bool)
        for (L,ak),js in pos.items():
            if ak in gX.get(L,set()):
                for j in js: keep[j]=True
        rows_tok.append(tok[gi]); rows_mask.append(keep); meta.append((s,c,priv,float(phi[gi,c])))

rows_tok=np.array(rows_tok,np.float32); rows_mask=np.array(rows_mask)
# CONTROL: remove the SAME NUMBER of peaks at random (keep the private peaks) — tests sparsity vs major-specific
rng=np.random.default_rng(0); rows_mask_rand=[]
for r,(s,c,priv,ph) in enumerate(meta):
    gi=sel5[s]; valid=np.where(msk[gi].astype(bool))[0]; ktot=int(rows_mask[r].sum())
    keep=set(priv); pool=[j for j in valid if j not in keep]
    rng.shuffle(pool)
    for j in pool[:max(0,ktot-len(keep))]: keep.add(j)
    mrand=np.zeros(tok.shape[1],bool); mrand[list(keep)]=True; rows_mask_rand.append(mrand)
rows_mask_rand=np.array(rows_mask_rand)
_, ATred = fwd(rows_tok, rows_mask)
_, ATrnd = fwd(rows_tok, rows_mask_rand)
print(f"run={RUN.name}\n")
print("== Q1  COUNTERFACTUAL: remove major-peak context, re-encode, read attr at minor's PRIVATE peaks ==")
acc_full=[]; acc_red=[]; acc_rnd=[]; npk=[]
for r,(s,c,priv,ph) in enumerate(meta):
    acc_full.append(np.mean([ATfull[s,j].argmax()==c for j in priv]))
    acc_red.append(np.mean([ATred[r,j].argmax()==c for j in priv]))
    acc_rnd.append(np.mean([ATrnd[r,j].argmax()==c for j in priv]))
    npk.append(len(priv))
acc_full=np.array(acc_full); acc_red=np.array(acc_red); acc_rnd=np.array(acc_rnd)
print(f"  deep-miss minors tested: {len(meta)}  (mean #private peaks={np.mean(npk):.1f})")
print(f"  attr-acc on private peaks:  FULL = {acc_full.mean():.3f}  |  MAJORS-REMOVED = {acc_red.mean():.3f}  |  RANDOM-same-count = {acc_rnd.mean():.3f}")
print(f"  recovered (0->1): majors-removed {100*np.mean((acc_full<0.5)&(acc_red>=0.5)):.1f}%  vs  random {100*np.mean((acc_full<0.5)&(acc_rnd>=0.5)):.1f}%")
print("  READ: majors-removed >> random => the encoder washes minor specifically by MIXING with the MAJOR peaks (height-dominant context),")
print("        not mere sparsity. 'Context' = the major contributors' peaks, NOT the combo label.")

# ── Q2  is the decoder bad? gap decomposition + decoy analysis ─────────────────────────────
def oracle_topk(P, k):
    em=[]
    for s,gi in enumerate(sel5):
        top=np.argsort(P[s])[::-1][:k]; pred=np.zeros(45); pred[top]=1; em.append((pred==ymat[gi]).all())
    return np.mean(em)
top5=oracle_topk(Pfull,5)
# top-8 containment ceiling
cont8=np.mean([set(np.where(ymat[gi]==1)[0]).issubset(set(np.argsort(Pfull[s])[::-1][:8].tolist())) for s,gi in enumerate(sel5)])
print("\n== Q2  is the decoder bad?  NOC5 oracle gap decomposition ==")
print(f"  decoder top-5 oracle           = {top5:.3f}")
print(f"  honest linear-on-H ceiling     ~ 0.52   (decoder BEATS it by +{top5-0.52:.2f} => decoder extracts real signal)")
print(f"  competition ceiling (top-8)    = {cont8:.3f}")
print(f"  => DECODER-recoverable (in top8, not top5) = {cont8-top5:.3f}  | ENCODER-bound (not in top8) = {1-cont8:.3f}")

# near-miss decoy: is the displacing decoy 'explained-away' by the true majors?
shareA=[]
for s,gi in enumerate(sel5):
    order=np.argsort(Pfull[s])[::-1]; top5s=set(order[:5].tolist()); true=set(np.where(ymat[gi]==1)[0].tolist())
    # only samples with a near-miss (a true donor at rank 6-8)
    if not any(6<=({d:r+1 for r,d in enumerate(order)}[c])<=8 for c in true): continue
    majors=sorted(true, key=lambda c:-phi[gi,c])[:2]; majint=[geno.get(KNOWN[c],{}) for c in majors]
    for d in (top5s-true):                                  # the decoy(s)
        gD=geno.get(KNOWN[d],{})
        if not gD: continue
        tot=cov=0
        for L,alls in gD.items():
            present=set().union(*[m.get(L,set()) for m in majint])
            for a in alls: tot+=1; cov+=int(a in present)
        if tot: shareA.append(cov/tot)
if shareA:
    print(f"\n  near-miss decoy: mean fraction of its alleles COVERED by the 2 true majors = {np.mean(shareA):.3f}")
    print("  (high => decoy is 'explained-away' by majors; a competition-aware decoder should demote it)")
