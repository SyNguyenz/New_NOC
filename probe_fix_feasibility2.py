"""
NO-TRAIN feasibility for Increment-8 component (4): REFERENCE-CONDITIONING (geno_query, deployable).

At inference we do NOT have per-peak attribution. geno_query's premise: a donor's KNOWN reference
genotype (donor_geno) tells us WHICH (locus,allele) peaks are its evidence — so we can locate a faint
minor's evidence without attribution, and focus past the shared-allele bleed (faint query splits 47%
own / 42% major). Test: pool H over peaks matching donor d's REFERENCE alleles (no attribution), remove
the shared carrier, and read identity SEEN vs NOVEL by faintness. Compare to oracle attribution pooling.
  ref-guided ~= attr-guided AND combo-invariant => geno_query GO + deployable (solves no-attr-at-test).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0,".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc7_masspool_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
G=np.load("kaggle_bundle/donor_geno.npy"); GM=np.load("kaggle_bundle/donor_geno_mask.npy")
# reference (locus,allele) set per donor
REF=[set((int(l),round(float(al),1)) for l,al in G[d][GM[d]][:,:2]) for d in range(45)]
dg=dgm=None
if cfg.get("geno_query"):
    dg=torch.from_numpy(G.astype(np.float32)); dgm=torch.from_numpy(GM)
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),dec_layers=cfg.get("dec_layers",2),num_embed=cfg.get("num_embed","raw"),
    n_freq=cfg.get("n_freq",8),d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),
    aux_heads=True,sparse_attn=cfg.get("sparse_attn",False),geno_query=cfg.get("geno_query",False),
    donor_geno=dg,donor_geno_mask=dgm,vib=cfg.get("vib",False),mass_pool=cfg.get("mass_pool",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
print(f"loaded {RUN.name}")

y=np.load(DATA/"y_train_set.npy"); noc=np.load(DATA/"noc_train.npy")
tk=np.load(DATA/"tokens8_train.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_train.npy").astype(bool)
at=np.load(DATA/"attr_train.npy"); LH=tk[:,:,2]; LOC=tk[:,:,0]; ALL=tk[:,:,1]
def dev_mask_seed0(y,noc,cf=0.15,nf=0.06,seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc.astype(int),1,5); N=len(noc); m=np.zeros(N,bool)
    for k in (2,3,4,5):
        idx=np.where(noc==k)[0]; cb={}
        for i in idx: cb.setdefault(tuple(np.where(y[i]==1)[0].tolist()),[]).append(i)
        u=list(cb); rng.shuffle(u)
        for c in u[:max(1,int(round(len(u)*cf)))]: m[cb[c]]=True
    i1=np.where(noc==1)[0]; m[rng.choice(i1,size=int(round(len(i1)*nf)),replace=False)]=True
    return m
dmask=dev_mask_seed0(y,noc); seen_idx=np.where(~dmask)[0]; novel_idx=np.where(dmask)[0]

@torch.no_grad()
def encode(idxs,bs=128):
    """per TRUE donor: attr-pooled and REF-pooled reps (both carrier-removed), donor, rank, noc, ratio."""
    ATTR=[];REF_=[];D=[];R=[];N=[];RA=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; valid=np.where(a>=0)[0]
            if len(valid)==0: continue
            dh={int(d):float(np.exp(LH[gi][a==d]).sum()) for d in np.unique(a[valid])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}; maj=dh[order[0]]
            carrier=H[j][valid].mean(0)               # deployable carrier (all peaks)
            peakkey={int(p):(int(LOC[gi,p]),round(float(ALL[gi,p]),1)) for p in valid}
            for d in np.unique(a[valid]):
                d=int(d)
                attrpk=np.where(a==d)[0]
                refpk=[p for p in valid if peakkey[int(p)] in REF[d]]   # NO attribution used
                if len(refpk)==0: continue
                ATTR.append(H[j][attrpk].mean(0)-carrier); REF_.append(H[j][refpk].mean(0)-carrier)
                D.append(d); R.append(ro[d]); N.append(int(noc[gi])); RA.append(dh[d]/maj)
    return np.array(ATTR),np.array(REF_),np.array(D),np.array(R),np.array(N),np.array(RA)

rng=np.random.default_rng(0)
fit=rng.choice(seen_idx,size=9000,replace=False)
seen5=seen_idx[noc[seen_idx]==5]; novel5=novel_idx[noc[novel_idx]==5]
Af,Rf,Df,_,_,_=encode(fit)
As,Rs2,Ds,Rs,Ns,RAs=encode(seen5)
An,Rn2,Dn,Rn,Nn,RAn=encode(novel5)

def probe(Xf,yf,Xs,Xn):
    clf=nn.Linear(Xf.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
    Xt=torch.from_numpy(Xf).to(DEV); yt=torch.from_numpy(yf).long().to(DEV); lf=nn.CrossEntropyLoss()
    for ep in range(120):
        p=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=p[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
    with torch.no_grad():
        return (clf(torch.from_numpy(Xs).to(DEV)).argmax(1).cpu().numpy(),
                clf(torch.from_numpy(Xn).to(DEV)).argmax(1).cpu().numpy())
psA,pnA=probe(Af,Df,As,An)                 # oracle attribution pooling (carrier-removed)
psR,pnR=probe(Rf,Df,Rs2,Rn2)               # reference-guided pooling (deployable, carrier-removed)

def acc(pr,D,m): return (pr[m]==D[m]).mean() if m.sum() else float('nan')
print("\n=== P4: faint-minor (N5 rank>=3) readability — ATTR-pooled(oracle) vs REF-pooled(deployable, no attribution), carrier-removed ===")
print(f"  {'ratio':>12} | {'ATTR seen':>9} {'ATTR nov':>8} {'gap':>6} | {'REF seen':>8} {'REF nov':>8} {'gap':>6}")
for lo,hi in [(0.0,0.02),(0.02,0.05),(0.05,0.10),(0.10,0.25),(0.0,1.0)]:
    ms=(Ns==5)&(Rs>=3)&(RAs>=lo)&(RAs<hi); mn=(Nn==5)&(Rn>=3)&(RAn>=lo)&(RAn<hi)
    print(f"  {('[%.2f,%.2f)'%(lo,hi)) if hi<1 else 'ALL faint':>12} | {acc(psA,Ds,ms):9.3f} {acc(pnA,Dn,mn):8.3f} {acc(psA,Ds,ms)-acc(pnA,Dn,mn):+6.3f} | {acc(psR,Ds,ms):8.3f} {acc(pnR,Dn,mn):8.3f} {acc(psR,Ds,ms)-acc(pnR,Dn,mn):+6.3f}")
print("\n  REF nov ~ ATTR nov => reference locates a faint minor's evidence WITHOUT attribution => geno_query GO + deployable.")
