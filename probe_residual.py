"""
probe_residual.py — EVAL-ONLY. After the encoder height-dominance fix (counterfactual: majors removed),
~13-19% of deep-miss minor private peaks STILL aren't recovered. What is that residual? = the part a
height-robust encoder CANNOT reach -> tells whether ~0.9 is a real ceiling or a probe limit.

For each deep-miss minor's private present peak, after MAJOR-REMOVAL re-encode, split recovered (attr argmax
== true donor) vs NOT, and compare physical token features + stutter-overlap with a major + donor phi.
Hypotheses for the residual: (a) intrinsic faintness (low height), (b) stutter overlap with a major (the
private allele sits at a major's n-1 position -> confusable even with the major's main peak gone),
(c) very low absolute phi floor.

Usage: python probe_residual.py [run]
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, torch

RUN = Path("results") / (sys.argv[1] if len(sys.argv) > 1 else "inc6_minorw_seed42")
DATA = Path("data_insilico_w"); DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
META = json.load(open(DATA/"meta_set.json")); KNOWN=META["known_donors"]; LOCUS_TO_IDX=META["locus_to_idx"]
cfg = json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8); tp=f"tokens{n_tok}"

def akey(v):
    s=str(v).strip()
    if s in ("","nan","NaN","None"): return ""
    if s=="X": return "-2.0"
    if s=="Y": return "-1.0"
    try: return f"{round(float(s),1):.1f}"
    except ValueError: return ""
def aval(v):
    try: return float(v)
    except (ValueError,TypeError): return None

def load_geno():
    f=glob.glob("data_raw/**/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx",recursive=True)[0]
    df=pd.read_excel(f,sheet_name=0); cols=[c for c in df.columns if c in LOCUS_TO_IDX]; g={}; gnum={}
    for _,r in df.iterrows():
        try: d=int(r["Sample ID"])
        except (ValueError,TypeError): continue
        if d not in KNOWN: continue
        gg={}; gn={}
        for loc in cols:
            if pd.isna(r[loc]): continue
            ks={akey(a) for a in str(r[loc]).split(",")}; ks.discard("")
            nums={aval(a) for a in str(r[loc]).split(",")}; nums.discard(None)
            if ks: gg[LOCUS_TO_IDX[loc]]=ks; gn[LOCUS_TO_IDX[loc]]=nums
        g[d]=gg; gnum[d]=gn
    return g, gnum

tok=np.load(DATA/f"{tp}_train.npy").astype(np.float32); msk=np.load(DATA/"mask_train.npy")
ymat=np.load(DATA/"y_train_set.npy").astype(np.float32); noc=np.load(DATA/"noc_train.npy").astype(int)
phi=np.load(DATA/"phi_train.npy").astype(np.float32)
def dev_mask_seed0(y,noc,cf=0.15,n1=0.06,seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc.astype(int),1,5); N=len(noc); m=np.zeros(N,bool)
    for k in [2,3,4,5]:
        idx=np.where(noc==k)[0]; cmb={}
        for i in idx: cmb.setdefault(tuple(np.where(y[i]==1)[0].tolist()),[]).append(i)
        u=list(cmb); rng.shuffle(u)
        for c in u[:max(1,int(round(len(u)*cf)))]: m[cmb[c]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*n1)),replace=False)]=True
    return m
dv_idx=np.where(dev_mask_seed0(ymat,noc))[0]; geno,genonum=load_geno()

model=SetTransformerMixture(
    n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),
    n_classes=45,n_noc=6,dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),
    decoder_source=cfg.get("decoder_source","encoded"),n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),
    dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),n_freq=cfg.get("n_freq",8),
    d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),aux_heads=True,
    d_proj=cfg.get("d_proj",64),sparse_attn=cfg.get("sparse_attn",False)).to(DEVICE)
sd=torch.load(RUN/"best_model.pt",map_location=DEVICE); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
model.load_state_dict(sd,strict=False); model.eval()

@torch.no_grad()
def fwd(toks,masks,bs=256):
    PP,AT=[],[]
    for i in range(0,len(toks),bs):
        o=model(torch.from_numpy(toks[i:i+bs]).to(DEVICE),torch.from_numpy(masks[i:i+bs]).to(DEVICE))
        PP.append(torch.sigmoid(o["logits_cls"]).cpu().numpy()); AT.append(torch.softmax(o["logits_attr"],-1).cpu().numpy())
    return np.concatenate(PP),np.concatenate(AT)

sel5=dv_idx[noc[dv_idx]==5]; Pfull,_=fwd(tok[sel5],msk[sel5])
rows_tok,rows_mask,meta=[],[],[]
for s,gi in enumerate(sel5):
    li=tok[gi][:,0].astype(int); al=tok[gi][:,1]; valid=msk[gi].astype(bool); pos={}
    for j in np.where(valid)[0]:
        a=float(al[j]); ak="-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
        pos.setdefault((int(li[j]),ak),[]).append(j)
    tcols=np.where(ymat[gi]==1)[0]; tints=[KNOWN[c] for c in tcols]
    order=np.argsort(Pfull[s])[::-1]; rankof={d:r+1 for r,d in enumerate(order)}
    majors=sorted(tcols,key=lambda c:-phi[gi,c])[:2]; majnum=[genonum.get(KNOWN[c],{}) for c in majors]
    for c in tcols:
        if rankof[c]<=8: continue
        X=KNOWN[c]; gX=geno.get(X,{})
        if not gX: continue
        others=[geno.get(o,{}) for o in tints if o!=X]
        priv=[]
        for L,alls in gX.items():
            oh=set().union(*[o.get(L,set()) for o in others]) if others else set()
            for a in alls:
                if a not in oh and (L,a) in pos:
                    for j in pos[(L,a)]: priv.append((j,L,a))
        if not priv: continue
        keep=np.zeros(tok.shape[1],bool)
        for (L,ak),js in pos.items():
            if ak in gX.get(L,set()):
                for j in js: keep[j]=True
        rows_tok.append(tok[gi]); rows_mask.append(keep); meta.append((s,c,priv,float(phi[gi,c]),majnum))
rows_tok=np.array(rows_tok,np.float32); rows_mask=np.array(rows_mask)
_,ATred=fwd(rows_tok,rows_mask)

# per private-peak: recovered? + features (height, SR, Hb, phi, stutter-overlap-with-major)
rec_feat={"recovered":[], "not-recovered":[]}
for r,(s,c,priv,ph,majnum) in enumerate(meta):
    gi=sel5[s]
    for (j,L,a) in priv:
        ok = ATred[r,j].argmax()==c
        height=float(np.expm1(tok[gi,j,2])); Hb=float(tok[gi,j,3]); SR=float(tok[gi,j,4])
        # stutter overlap: is this private allele at a major's n-1 (back-stutter) position?
        try: av=float(a)
        except ValueError: av=None
        stut=False
        if av is not None:
            for mn in majnum:
                for amaj in mn.get(L,set()):
                    if abs((amaj-1.0)-av)<0.05 or abs((amaj+1.0)-av)<0.05: stut=True
        rec_feat["recovered" if ok else "not-recovered"].append((height,SR,Hb,ph,int(stut)))

print(f"run={RUN.name}  (NOC5 dev deep-miss private peaks)\n")
print("== Residual after MAJOR-REMOVAL: recovered vs NOT, physical features ==")
print(f"  {'group':14s} {'n':>6} {'med-height':>10} {'med-SR':>7} {'med-Hb':>7} {'med-phi':>8} {'%stutter-of-major':>18}")
for g,rows in rec_feat.items():
    a=np.array(rows)
    if not len(a): continue
    print(f"  {g:14s} {len(a):>6} {np.median(a[:,0]):>10.1f} {np.median(a[:,1]):>7.2f} {np.median(a[:,2]):>7.2f} {np.median(a[:,3]):>8.3f} {100*a[:,4].mean():>17.1f}%")
print("\n  (low height / high SR / high %stutter in NOT-recovered => residual is physical faintness+stutter, ~0.9 near a real ceiling)")

# absolute-phi floor: samples whose true set is NOT even in top-12
print("\n== top-12-miss samples: is there an absolute phi floor? ==")
below=[]; above=[]
for s,gi in enumerate(sel5):
    top12=set(np.argsort(Pfull[s])[::-1][:12].tolist()); true=set(np.where(ymat[gi]==1)[0].tolist())
    minphi=min(phi[gi,c] for c in true)
    (below if not true.issubset(top12) else above).append(minphi)
print(f"  min-phi of weakest donor:  in-top12 samples med={np.median(above):.3f}  |  beyond-top12 samples med={np.median(below):.3f}")
print(f"  beyond-top12 = {len(below)}/{len(sel5)} samples; frac with min-phi<0.04 = {100*np.mean(np.array(below)<0.04):.0f}%")
