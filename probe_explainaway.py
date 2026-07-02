"""
On PRODUCTION N5, signature PRESENCE is blind to the decoy (AUC missed-true vs decoy = 0.501).
Question: does a HEIGHT / EXPLAINING-AWAY signal separate them, where presence cannot?

For each candidate donor c (deployable features; GT used only to label groups):
  frac_present   : fraction of c's reference alleles observed in the mix
  n_priv_present : # of c's PRIVATE alleles observed (a decoy is absent -> its privates should drop)
  coherence      : mean/std of (obs_height / dosage) over c's present alleles
                   (one real contributor -> all alleles share one proportion -> low CV -> high coherence;
                    a decoy's alleles are produced by DIFFERENT true donors -> incoherent)
  exclusive_ev   : # of c's present alleles NOT carried by ANY OTHER model-top-k donor
                   (EXPLAINING-AWAY: a decoy's alleles are all explained by the true set -> ~0 exclusive;
                    a true faint minor contributes some allele no one else does)
Report AUC(missed-true vs decoy) for each -> which signal is the lever.
"""
import os, json, itertools
from pathlib import Path
import numpy as np, torch

DATA=Path(os.environ.get("STR_DATA_DIR","data_insilico_w")); RUN=Path(os.environ.get("RUN","results/inc6_maskp_seed42"))
GENO=Path("data/donor_geno.npy"); DEVc=torch.device("cuda" if torch.cuda.is_available() else "cpu")
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))

g=np.load(GENO); gm=np.load(str(GENO).replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
# per donor: allele -> dosage (count in genotype) ; and private set
ditems=[{} for _ in range(C)]
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]:
            it=kk(g[c,j,0],g[c,j,1]); ditems[c][it]=ditems[c].get(it,0)+1
own={}
for c in range(C):
    for it in ditems[c]: own.setdefault(it,set()).add(c)
priv=[set(it for it in ditems[c] if own[it]=={c}) for c in range(C)]

from models.set_transformer import SetTransformerMixture
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8); tp=f"tokens{n_tok}"
def load(sp): return (np.load(DATA/f"{tp}_{sp}.npy").astype(np.float32),np.load(DATA/f"mask_{sp}.npy"),
                      np.load(DATA/f"y_{sp}_set.npy").astype(bool),np.load(DATA/f"noc_{sp}.npy").astype(int))
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","per_donor"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab++"),dec_layers=cfg.get("dec_layers",2),
    num_embed=cfg.get("num_embed","periodic"),n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),
    periodic_sigma=cfg.get("periodic_sigma",0.3),aux_heads=cfg.get("aux_heads",False),sparse_attn=cfg.get("sparse_attn",False)).to(DEVc)
sd=torch.load(RUN/"best_model.pt",map_location=DEVc); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
model.load_state_dict(sd,strict=False); model.eval()
@torch.no_grad()
def logits(tok,msk):
    L=[]
    for i in range(0,len(tok),256):
        o=model(torch.from_numpy(tok[i:i+256]).to(DEVc),torch.from_numpy(msk[i:i+256].astype(bool)).to(DEVc))
        L.append(o["logits_cls"].cpu().numpy())
    return np.concatenate(L)
tk,mk,y,noc=load("test"); Lg=logits(tk,mk)

def auc(pos,neg):
    pos,neg=np.asarray(pos,float),np.asarray(neg,float)
    if not len(pos) or not len(neg): return float("nan")
    a=np.concatenate([pos,neg]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    return (rk[:len(pos)].sum()-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def feats(i, c, obs, others_alleles):
    al=ditems[c]
    if not al: return None
    pres=[(it,obs[it]) for it in al if it in obs]
    fp=len(pres)/len(al)
    npp=sum(1 for it in priv[c] if it in obs)
    if pres:
        hn=np.array([h/al[it] for it,h in pres])
        coh = hn.mean()/(hn.std()+1e-6)
    else: coh=0.0
    excl=sum(1 for it,_ in pres if it not in others_alleles)
    return fp, npp, coh, excl

for NV in [5,4]:
    sel=np.where(noc==NV)[0]
    F={k:{"miss":[],"dec":[]} for k in ["frac_present","n_priv_present","coherence","exclusive_ev"]}
    for i in sel:
        order=np.argsort(Lg[i])[::-1]; top=set(order[:NV]); tru=set(np.where(y[i])[0])
        miss=sorted(tru-top); dec=sorted((set(range(C))-tru)&top)
        # observed peaks
        obs={}
        for kk_ in np.where(mk[i])[0]:
            it=kk(tk[i,kk_,0],tk[i,kk_,1]); h=np.expm1(tk[i,kk_,2]); obs[it]=max(obs.get(it,0.0),h)
        for grp,clist in [("miss",miss),("dec",dec)]:
            for c in clist:
                others=set(range(C))&top; others=others-{c}
                others_al=set().union(*[set(ditems[o]) for o in others]) if others else set()
                f=feats(i,c,obs,others_al)
                if f is None: continue
                F["frac_present"][grp].append(f[0]); F["n_priv_present"][grp].append(f[1])
                F["coherence"][grp].append(f[2]); F["exclusive_ev"][grp].append(f[3])
    print(f"\n===== N{NV} (samples={len(sel)}) — AUC(missed-true vs decoy), 0.5=blind =====")
    print(f"  reference: signature-presence AUC = 0.501 (N5) / 0.418 (N4)")
    keys=["frac_present","n_priv_present","coherence","exclusive_ev"]
    for k in keys:
        mi,de=F[k]["miss"],F[k]["dec"]
        print(f"  {k:>16}: AUC={auc(mi,de):.3f}   miss_mean={np.mean(mi):.2f}  decoy_mean={np.mean(de):.2f}  (n={len(mi)})")
    # COMBINED upper bound: in-sample Fisher LDA over all 4 features
    Xm=np.array([F[k]["miss"] for k in keys]).T; Xd=np.array([F[k]["dec"] for k in keys]).T
    X=np.vstack([Xm,Xd]); X=(X-X.mean(0))/(X.std(0)+1e-9)
    nm=len(Xm); mu1=X[:nm].mean(0); mu0=X[nm:].mean(0)
    Sw=np.cov(X[:nm].T)+np.cov(X[nm:].T)+1e-3*np.eye(X.shape[1])
    w=np.linalg.solve(Sw, mu1-mu0); s=X@w
    print(f"  >> COMBINED (LDA in-sample UPPER BOUND): AUC={auc(s[:nm],s[nm:]):.3f}")
