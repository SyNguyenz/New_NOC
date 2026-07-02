"""
probe_faint_check.py — EVAL-ONLY. Stress-test the "residual = too faint / physical floor" claim, given
that WE generated the mixtures (make_insilico AT=14 RFU dropout; surviving peaks are clean scaled-real).

For deep-miss NOC5 minors, AFTER major-removal re-encode, on each PRESENT private peak:
  (1) attr-recovery vs peak HEIGHT band — if recovery rises monotonically with height for PRESENT peaks,
      the residual is the SAME height-underweighting bias (FIXABLE), not a physical floor.
  (2) #surviving private peaks per donor (recovered donor vs not) — few-evidence (AT-dropout removed the rest).
  (3) allele PANEL-commonness (how many of the 45 known donors carry that locus-allele) — closed-set
      discriminability, distinct from faintness.

Usage: python probe_faint_check.py [run]
"""
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd, torch
RUN=Path("results")/(sys.argv[1] if len(sys.argv)>1 else "inc6_minorw_seed42")
DATA=Path("data_insilico_w"); DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
META=json.load(open(DATA/"meta_set.json")); KNOWN=META["known_donors"]; LOCUS_TO_IDX=META["locus_to_idx"]
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8); tp=f"tokens{n_tok}"
def akey(v):
    s=str(v).strip()
    if s in ("","nan","NaN","None"): return ""
    if s=="X": return "-2.0"
    if s=="Y": return "-1.0"
    try: return f"{round(float(s),1):.1f}"
    except ValueError: return ""
def load_geno():
    f=glob.glob("data_raw/**/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx",recursive=True)[0]
    df=pd.read_excel(f,sheet_name=0); cols=[c for c in df.columns if c in LOCUS_TO_IDX]; g={}
    for _,r in df.iterrows():
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
def dev_mask_seed0(y,noc,cf=0.15,n1=0.06,seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc.astype(int),1,5); N=len(noc); m=np.zeros(N,bool)
    for k in [2,3,4,5]:
        idx=np.where(noc==k)[0]; cmb={}
        for i in idx: cmb.setdefault(tuple(np.where(y[i]==1)[0].tolist()),[]).append(i)
        u=list(cmb); rng.shuffle(u)
        for c in u[:max(1,int(round(len(u)*cf)))]: m[cmb[c]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*n1)),replace=False)]=True
    return m
dv_idx=np.where(dev_mask_seed0(ymat,noc))[0]; geno=load_geno()
# panel commonness: # of 45 known donors carrying each (locus, allele_key)
from collections import Counter
panel=Counter()
for d,gg in geno.items():
    for L,alls in gg.items():
        for a in alls: panel[(L,a)]+=1

model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),aux_heads=True,
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
    al=tok[gi][:,1]; li=tok[gi][:,0].astype(int); valid=msk[gi].astype(bool); pos={}
    for j in np.where(valid)[0]:
        a=float(al[j]); ak="-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
        pos.setdefault((int(li[j]),ak),[]).append(j)
    tcols=np.where(ymat[gi]==1)[0]; tints=[KNOWN[c] for c in tcols]
    order=np.argsort(Pfull[s])[::-1]; rankof={d:r+1 for r,d in enumerate(order)}
    for c in tcols:
        if rankof[c]<=8: continue
        X=KNOWN[c]; gX=geno.get(X,{})
        if not gX: continue
        others=[geno.get(o,{}) for o in tints if o!=X]; priv=[]
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
        rows_tok.append(tok[gi]); rows_mask.append(keep); meta.append((s,c,priv))
rows_tok=np.array(rows_tok,np.float32); rows_mask=np.array(rows_mask); _,ATred=fwd(rows_tok,rows_mask)
print(f"run={RUN.name}  (NOC5 deep-miss, after major-removal; AT=14 RFU so all present peaks are >=14)\n")
# (1) recovery vs height band (present peaks only)
print("== (1) attr-recovery vs PRESENT-peak HEIGHT (RFU) — monotone with height => height-underweighting bias, NOT a floor ==")
bands=[(14,25),(25,40),(40,70),(70,1e9)]
agg={b:[0,0] for b in bands}
perdonor_priv={"recovered":[], "not":[]}
for r,(s,c,priv) in enumerate(meta):
    gi=sel5[s]; donor_ok=False
    for (j,L,a) in priv:
        h=float(np.expm1(tok[gi,j,2])); ok=int(ATred[r,j].argmax()==c)
        for b in bands:
            if b[0]<=h<b[1]: agg[b][0]+=ok; agg[b][1]+=1; break
        if ok: donor_ok=True
    perdonor_priv["recovered" if donor_ok else "not"].append(len(priv))
print(f"  {'height band':14s} {'n_peaks':>8} {'attr-recovery':>14}")
for b in bands:
    ok,n=agg[b]
    if n: print(f"  {str(b[0])+'-'+(str(b[1]) if b[1]<1e9 else 'inf'):14s} {n:>8} {ok/n:>14.3f}")
# (2) #surviving private peaks per donor
print("\n== (2) # PRESENT private peaks per deep-miss donor (few-evidence from AT-dropout?) ==")
for g in ("recovered","not"):
    a=np.array(perdonor_priv[g]); print(f"  donor {g:10s}: n={len(a):>4}  median #private-present-peaks={np.median(a):.0f}  %with-only-1={100*np.mean(a==1):.0f}%")
# (3) panel commonness of the private allele
print("\n== (3) allele PANEL-commonness (#of 45 donors sharing it) — recovered vs not (closed-set discriminability) ==")
com={"recovered":[], "not":[]}
for r,(s,c,priv) in enumerate(meta):
    for (j,L,a) in priv:
        ok=ATred[r,j].argmax()==c; com["recovered" if ok else "not"].append(panel.get((L,a),0))
for g in ("recovered","not"):
    a=np.array(com[g]); print(f"  {g:10s}: median #donors-sharing-allele={np.median(a):.0f}  mean={a.mean():.1f}")
print("\nVERDICT: recovery rising with height (present peaks) => same height-bias, fixable, NOT physical floor.")
