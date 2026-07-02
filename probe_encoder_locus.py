"""
LOCALIZE where combo-dependence enters the encoder.
Pipeline: tokens -> _project_tokens -> x0 (input proj) -> ISAB blocks -> H (encoder out).
Per-donor representation = MEAN over that donor's true peaks, at x0 vs at H.
Measure SEEN(train combos) vs NOVEL(dev combos) readability of FAINT minors (N5 rank>=3),
matched on minor/major template ratio. Both groups synthetic (no real confound).

  gap already at x0  => combo-dependence comes from the INPUT FEATURES (relational/combo-relative
                        token fields computed over the whole mixture).
  gap only/larger at H => the ISAB cross-token MIXING ADDS the combo-dependence (entanglement).
Also reports absolute level: how much donor identity each representation carries.
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
print(f"loaded {RUN.name} mass_pool={cfg.get('mass_pool')}")

y=np.load(DATA/"y_train_set.npy"); noc=np.load(DATA/"noc_train.npy")
tk=np.load(DATA/"tokens8_train.npy")[:,:,:n_tok].astype(np.float32); mk=np.load(DATA/"mask_train.npy").astype(bool)
at=np.load(DATA/"attr_train.npy"); LH=tk[:,:,2]
def dev_mask_seed0(y,noc,combo_frac=0.15,noc1_frac=0.06,seed=0):
    rng=np.random.default_rng(seed); noc=np.clip(noc.astype(int),1,5); N=len(noc); m=np.zeros(N,bool)
    for k in (2,3,4,5):
        idx=np.where(noc==k)[0]; combos={}
        for i in idx: combos.setdefault(tuple(np.where(y[i]==1)[0].tolist()),[]).append(i)
        uniq=list(combos); rng.shuffle(uniq)
        for c in uniq[:max(1,int(round(len(uniq)*combo_frac)))]: m[combos[c]]=True
    idx1=np.where(noc==1)[0]; m[rng.choice(idx1,size=int(round(len(idx1)*noc1_frac)),replace=False)]=True
    return m
dmask=dev_mask_seed0(y,noc); seen_idx=np.where(~dmask)[0]; novel_idx=np.where(dmask)[0]

@torch.no_grad()
def encode_pooled(idxs,bs=128):
    """Per-DONOR pooled reps: mean of x0 and H over the donor's true peaks."""
    X0=[];H_=[];D=[];R=[];N=[];RA=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        x0,H,_=model._encode_set(t,m); x0=x0.cpu().numpy(); H=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; valid=np.where(a>=0)[0]
            if len(valid)==0: continue
            dh={int(d):float(np.exp(LH[gi][a==d]).sum()) for d in np.unique(a[valid])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}; maj=dh[order[0]]
            for d in np.unique(a[valid]):
                pk=np.where(a==d)[0]
                X0.append(x0[j][pk].mean(0)); H_.append(H[j][pk].mean(0)); D.append(int(d))
                R.append(ro[int(d)]); N.append(int(noc[gi])); RA.append(dh[int(d)]/maj)
    return (np.array(X0),np.array(H_),np.array(D),np.array(R),np.array(N),np.array(RA))

rng=np.random.default_rng(0)
fit_sel=rng.choice(seen_idx,size=9000,replace=False)
seen_ev=rng.choice(np.setdiff1d(seen_idx,fit_sel),size=4000,replace=False)
X0f,Hf,Df,_,_,_=encode_pooled(fit_sel)
X0s,Hs,Ds,Rs,Ns,RAs=encode_pooled(seen_ev)
X0n,Hn,Dn,Rn,Nn,RAn=encode_pooled(novel_idx)
print(f"per-donor reps: fit {len(Df)} | seen {len(Ds)} | novel {len(Dn)}")

def fit_eval(Xf,yf,Xs,Xn):
    clf=nn.Linear(Xf.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
    Xt=torch.from_numpy(Xf).to(DEV); yt=torch.from_numpy(yf).long().to(DEV); lf=nn.CrossEntropyLoss()
    for ep in range(120):
        p=torch.randperm(len(yt),device=DEV)
        for s in range(0,len(yt),8192):
            b=p[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
    with torch.no_grad():
        ps=clf(torch.from_numpy(Xs).to(DEV)).argmax(1).cpu().numpy()
        pn=clf(torch.from_numpy(Xn).to(DEV)).argmax(1).cpu().numpy()
    return ps,pn
ps_x0,pn_x0=fit_eval(X0f,Df,X0s,X0n)
ps_H ,pn_H =fit_eval(Hf ,Df,Hs ,Hn )

def acc(pr,D,mask): return (pr[mask]==D[mask]).mean() if mask.sum() else float('nan')
print("\n=== Faint-minor (N5 rank>=3) donor-id readability: x0(INPUT) vs H(ENCODER), SEEN vs NOVEL combo ===")
print(f"  {'ratio bin':>12} | {'x0 SEEN':>8} {'x0 NOVEL':>8} {'x0 gap':>7} | {'H SEEN':>8} {'H NOVEL':>8} {'H gap':>7}")
for lo,hi in [(0.0,0.02),(0.02,0.05),(0.05,0.10),(0.10,0.25),(0.0,1.0)]:
    ms=(Ns==5)&(Rs>=3)&(RAs>=lo)&(RAs<hi); mn=(Nn==5)&(Rn>=3)&(RAn>=lo)&(RAn<hi)
    a0s,a0n=acc(ps_x0,Ds,ms),acc(pn_x0,Dn,mn); aHs,aHn=acc(ps_H,Ds,ms),acc(pn_H,Dn,mn)
    tag=f"[{lo:.2f},{hi:.2f})" if hi<1.0 else "ALL faint"
    print(f"  {tag:>12} | {a0s:8.3f} {a0n:8.3f} {a0s-a0n:+7.3f} | {aHs:8.3f} {aHn:8.3f} {aHs-aHn:+7.3f}")
print("\n  Read: x0-gap>0 => INPUT features already combo-dependent; H-gap>>x0-gap => ISAB mixing ADDS it.")
print("        absolute level (SEEN cols): how much donor identity each stage carries.")
