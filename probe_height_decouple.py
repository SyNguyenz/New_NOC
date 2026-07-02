"""
probe_height_decouple.py — EVAL-ONLY no-train test of the CHEAP lever "decouple height from the attention".

Mechanism hypothesis (F31 + massive-activation/attention-sink lit): high-MAGNITUDE (tall=major) peaks
dominate the ISAB pooling so the faint minor's per-peak H gets washed. The major-REMOVAL counterfactual
(probe_context) recovers the minor (attr .07->.81) but removes peaks (= peeling). This probe keeps EVERY
peak present and instead NEUTRALISES the height features (set to feat_mean -> standardized 0), so all
alleles stay but their height/rank/dominance signal is flattened. Re-encode frozen model, read attr_head
at the minor's PRIVATE peaks.

  * minor RECOVERS under height-neutralisation (toward the .81 major-removal ceiling) => it IS height-
    magnitude dominance -> a height-decoupled key / height-robust aggregation lever is VALIDATED (cheap).
  * stays low (~full)  => it's the PRESENCE/structure of major peaks, not their height -> need the
    aggregation-normalization (scaled-weighted-sum), height-decoupling alone won't fix it.

Token feats (1+idx): col1 allele, col2 log_h(idx1), col3 Hb(idx2), col4 SR(idx3), col5 rank_inv(idx4),
col6 n/10(idx5), col7 glob_rel(idx6). Height-dominance feats = log_h, Hb, rank_inv, glob_rel.

Usage: python probe_height_decouple.py [run]
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
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),aux_heads=True,
    d_proj=cfg.get("d_proj",64),sparse_attn=cfg.get("sparse_attn",False)).to(DEVICE)
sd=torch.load(RUN/"best_model.pt",map_location=DEVICE); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
model.load_state_dict(sd,strict=False); model.eval()
feat_mean=model.feat_mean.cpu().numpy()   # (n_num,) original-scale means; idx i -> token col (1+i)
@torch.no_grad()
def fwd(toks,masks,bs=256):
    PP,AT=[],[]
    for i in range(0,len(toks),bs):
        o=model(torch.from_numpy(toks[i:i+bs]).to(DEVICE),torch.from_numpy(masks[i:i+bs]).to(DEVICE))
        PP.append(torch.sigmoid(o["logits_cls"]).cpu().numpy()); AT.append(torch.softmax(o["logits_attr"],-1).cpu().numpy())
    return np.concatenate(PP),np.concatenate(AT)
sel5=dv_idx[noc[dv_idx]==5]; Pfull,ATfull=fwd(tok[sel5],msk[sel5])
# collect deep-miss minors + private peak positions
items=[]
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
                    for j in pos[(L,a)]: priv.append(j)
        if priv: items.append((s,gi,c,priv))

def neutralise(cols_to_flatten):
    """build a token batch (one row per deep-miss item) with given height feats set to feat_mean on valid peaks."""
    rows=np.stack([tok[gi].copy() for (_,gi,_,_) in items]); masks=np.stack([msk[sel5[s]] for (s,_,_,_) in items])
    for r,(s,gi,c,priv) in enumerate(items):
        v=msk[gi].astype(bool)
        for fidx in cols_to_flatten:
            rows[r][v, 1+fidx] = feat_mean[fidx]
    return rows.astype(np.float32), masks

def attr_acc(AT, mode):
    accs=[]
    for r,(s,gi,c,priv) in enumerate(items):
        idx = s if mode=="full" else r
        accs.append(np.mean([AT[idx,j].argmax()==c for j in priv]))
    return np.mean(accs)

print(f"run={RUN.name}  (NOC5 deep-miss minors n={len(items)}; attr-acc on their PRIVATE peaks)\n")
print(f"  FULL (baseline)                         : {attr_acc(ATfull,'full'):.3f}")
# log_h only
t1,m1=neutralise([1]); _,A1=fwd(t1,m1); print(f"  neutralise log_h only                   : {attr_acc(A1,'x'):.3f}")
# all global height-dominance feats: log_h(1), Hb(2), rank_inv(4), glob_rel(6)
t2,m2=neutralise([1,2,4,6]); _,A2=fwd(t2,m2); print(f"  neutralise log_h+Hb+rank_inv+glob_rel   : {attr_acc(A2,'x'):.3f}")
print(f"  (reference) MAJORS-REMOVED ceiling      : ~0.81  (probe_context, removes peaks = peeling)")
print("\nREAD: big jump under height-neutralisation (toward .81) => height-MAGNITUDE dominance -> height-decoupled")
print("      key / robust aggregation lever VALIDATED. Flat ~baseline => it's major PRESENCE, need scaled-weighted-sum.")
