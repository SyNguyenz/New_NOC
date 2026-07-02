"""
NO-TRAIN feasibility for the Increment-8 directions, on frozen inc7 H.

P1 (encoder over-smoothing / shared-context carrier removal):
    carrier_i = per-sample MEAN of donor-pooled reps (the component common to all donors = the
    combo-specific shared context). residual = rep - carrier. If the residual reads faint-minor
    identity BETTER on novel and with a SMALLER seen->novel gap (and within-sample donor cosine
    drops from ~0.90), then de-smoothing / carrier-removal is a GO encoder lever.

P2 (premise of the compositional-consistency loss):
    among faint minors on NOVEL combos, is cross-combo SELF-STABILITY of a donor's rep correlated
    with being correctly readable? If yes, pushing reps toward combo-invariance (what the consistency
    loss does) should raise novel readability => GO.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0,".")
from models.set_transformer import SetTransformerMixture

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc7_masspool_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; torch.manual_seed(0); np.random.seed(0)
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
dg=dgm=None
if cfg.get("geno_query"):
    dg=torch.from_numpy(np.load(DATA/"donor_geno.npy").astype(np.float32)); dgm=torch.from_numpy(np.load(DATA/"donor_geno_mask.npy"))
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
at=np.load(DATA/"attr_train.npy"); LH=tk[:,:,2]
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
    """per-donor: raw pooled rep, carrier-removed (ORACLE donor-pool mean) and (NO-ORACLE all-peak mean)."""
    RAW=[];RES=[];RES2=[];D=[];R=[];N=[];RA=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        _,H,_=model._encode_set(t,m); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; valid=np.where(a>=0)[0]
            if len(valid)==0: continue
            dh={int(d):float(np.exp(LH[gi][a==d]).sum()) for d in np.unique(a[valid])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}; maj=dh[order[0]]
            reps={int(d):H[j][a==d].mean(0) for d in np.unique(a[valid])}
            carrier=np.mean(list(reps.values()),0)          # ORACLE per-donor-mean shared component
            carrier_all=H[j][valid].mean(0)                 # NO-ORACLE: mean over ALL valid peaks (deployable)
            for d in reps:
                RAW.append(reps[d]); RES.append(reps[d]-carrier); RES2.append(reps[d]-carrier_all)
                D.append(d); R.append(ro[d]); N.append(int(noc[gi])); RA.append(dh[d]/maj)
    return (np.array(RAW),np.array(RES),np.array(RES2),np.array(D),np.array(R),np.array(N),np.array(RA))

rng=np.random.default_rng(0)
fit=rng.choice(seen_idx,size=9000,replace=False)
seen5=seen_idx[noc[seen_idx]==5]; novel5=novel_idx[noc[novel_idx]==5]
RAWf,RESf,RES2f,Df,_,_,_=encode(fit)
RAWs,RESs,RES2s,Ds,Rs,Ns,RAs=encode(seen5)
RAWn,RESn,RES2n,Dn,Rn,Nn,RAn=encode(novel5)

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
ps_raw,pn_raw=probe(RAWf,Df,RAWs,RAWn)
ps_res,pn_res=probe(RESf,Df,RESs,RESn)
ps_r2,pn_r2=probe(RES2f,Df,RES2s,RES2n)

def acc(pr,D,m): return (pr[m]==D[m]).mean() if m.sum() else float('nan')
print("\n=== P1: faint-minor (N5 rank>=3) novel readability — RAW vs carrier-removed(ORACLE) vs carrier(ALL-PEAK,deployable) ===")
print(f"  {'ratio':>12} | {'RAW nov':>8} {'gap':>6} | {'RESorc nov':>10} {'gap':>6} | {'RESall nov':>10} {'gap':>6}")
for lo,hi in [(0.0,0.02),(0.02,0.05),(0.05,0.10),(0.10,0.25),(0.0,1.0)]:
    ms=(Ns==5)&(Rs>=3)&(RAs>=lo)&(RAs<hi); mn=(Nn==5)&(Rn>=3)&(RAn>=lo)&(RAn<hi)
    rn=acc(pn_raw,Dn,mn); rg=acc(ps_raw,Ds,ms)-rn
    en=acc(pn_res,Dn,mn); eg=acc(ps_res,Ds,ms)-en
    an=acc(pn_r2,Dn,mn); ag=acc(ps_r2,Ds,ms)-an
    tag=f"[{lo:.2f},{hi:.2f})" if hi<1 else "ALL faint"
    print(f"  {tag:>12} | {rn:8.3f} {rg:+6.3f} | {en:10.3f} {eg:+6.3f} | {an:10.3f} {ag:+6.3f}")
# within-sample donor cosine before/after carrier removal (novel)
def wsim(X,idx5):
    from collections import defaultdict
    byi=defaultdict(list)
    # rebuild per-sample grouping via recompute is costly; approximate: use consecutive donors share sample order
    return None
print("\n  GO if RES novel-readability > RAW novel-readability AND RES gap < RAW gap (carrier removal helps + de-biases).")

# P2: does cross-combo self-stability predict novel readability? (correlation, novel faint minors)
from collections import defaultdict
byd=defaultdict(list)
for raw,d,r,n in zip(RAWn,Dn,Rn,Nn):
    if n==5 and r>=3: byd[d].append(raw)
def cos(a,b): return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
stab={};
for d,L in byd.items():
    if len(L)<3: continue
    R=np.stack(L); cs=[cos(R[i],R[k]) for i in range(len(R)) for k in range(i+1,len(R))]
    stab[d]=np.mean(cs)
correct=defaultdict(list)
for pr,d,r,n in zip(pn_raw,Dn,Rn,Nn):
    if n==5 and r>=3 and d in stab: correct[d].append(pr==d)
xs=[stab[d] for d in stab if d in correct]; ys=[np.mean(correct[d]) for d in stab if d in correct]
if len(xs)>5:
    cc=np.corrcoef(xs,ys)[0,1]
    lo=np.array(ys)[np.array(xs)<np.median(xs)].mean(); hi=np.array(ys)[np.array(xs)>=np.median(xs)].mean()
    print(f"\n=== P2: self-stability vs novel readability (per donor, faint minors) n_donors={len(xs)} ===")
    print(f"  Pearson r = {cc:+.3f} | readability: low-stability donors {lo:.3f} vs high-stability {hi:.3f}")
    print("  r>0 / high>low => making reps combo-stable (consistency loss target) should raise novel readability => GO.")
