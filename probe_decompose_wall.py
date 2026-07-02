"""
Reconcile: a ~5-6pp encoder seen-vs-novel gap vs the ~34pp N5 set-oracle wall.
On SEEN(train) vs NOVEL(dev) N5, by donor height-rank, measure:
  (1) ENCODER readability   = independent linear probe on H-pooled donor rep (fit on seen), acc.
  (2) DECODER inclusion      = model's own sigmoid(logits_cls[true donor]) > 0.5.
  (3) DECODER margin         = (2) - (1)  [how much the trained decoder beats a plain readout].
Then show set-level compounding: prod over the 5 donors of per-donor inclusion ~ set EM.
If the decoder MARGIN collapses seen->novel much more than encoder readability drops,
the wall is mostly DECODER combo-memorization, with the encoder a smaller (readability-ceiling) share.
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
def run(idxs,bs=128):
    """per true-donor: H-pooled rep, donor id, rank, decode prob (sigmoid logits_cls)."""
    HP=[];D=[];R=[];P=[]; SETprob=[]; SETtrue=[]
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        x0,H,pad=model._encode_set(t,m)
        out=model(t,m); pr=torch.sigmoid(out["logits_cls"]).cpu().numpy()
        Hc=H.cpu().numpy()
        for j,gi in enumerate(sel):
            a=at[gi]; valid=np.where(a>=0)[0]
            if len(valid)==0: continue
            dh={int(d):float(np.exp(LH[gi][a==d]).sum()) for d in np.unique(a[valid])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}
            for d in np.unique(a[valid]):
                pk=np.where(a==d)[0]
                HP.append(Hc[j][pk].mean(0)); D.append(int(d)); R.append(ro[int(d)]); P.append(pr[j][int(d)])
            SETprob.append(pr[j]); SETtrue.append(set(int(d) for d in np.unique(a[valid])))
    return np.array(HP),np.array(D),np.array(R),np.array(P),SETprob,SETtrue

rng=np.random.default_rng(0)
fit_sel=rng.choice(seen_idx[noc[seen_idx]>=2],size=9000,replace=False)
HPf,Df,_,_,_,_=run(fit_sel)
# only N5 for the readout table
seen5=seen_idx[noc[seen_idx]==5]; novel5=novel_idx[noc[novel_idx]==5]
HPs,Ds,Rs,Ps,_,_=run(seen5)
HPn,Dn,Rn,Pn,SETp,SETt=run(novel5)

# encoder readability probe (fit on seen donors, all NOC)
clf=nn.Linear(HPf.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4)
Xt=torch.from_numpy(HPf).to(DEV); yt=torch.from_numpy(Df).long().to(DEV); lf=nn.CrossEntropyLoss()
for ep in range(120):
    p=torch.randperm(len(yt),device=DEV)
    for s in range(0,len(yt),8192):
        b=p[s:s+8192]; opt.zero_grad(); lf(clf(Xt[b]),yt[b]).backward(); opt.step()
@torch.no_grad()
def rd(HP,D): return clf(torch.from_numpy(HP).to(DEV)).argmax(1).cpu().numpy()==D
ENs=rd(HPs,Ds); ENn=rd(HPn,Dn)

print("\n=== N5 by donor height-rank (0=major .. 4=faintest minor): SEEN(train) vs NOVEL(dev) ===")
print(f"  {'rank':>5} | {'ENC-read SEEN':>13} {'NOVEL':>7} {'denc':>6} | {'DEC-incl SEEN':>13} {'NOVEL':>7} {'ddec':>6} | {'margin SEEN':>11} {'NOVEL':>7}")
for r in range(5):
    ms=Rs==r; mn=Rn==r
    es,en=ENs[ms].mean(), ENn[mn].mean()
    ds,dn=(Ps[ms]>0.5).mean(), (Pn[mn]>0.5).mean()
    print(f"  {r:>5} | {es:13.3f} {en:7.3f} {en-es:+6.3f} | {ds:13.3f} {dn:7.3f} {dn-ds:+6.3f} | {ds-es:+11.3f} {dn-en:+7.3f}")

# set-level compounding check on novel
incl=[]
for pr,tru in zip(SETp,SETt):
    incl += [pr[d]>0.5 for d in tru]
print(f"\n  novel per-donor inclusion (all N5 donors) mean = {np.mean(incl):.3f}  -> ^5 ~ {np.mean(incl)**5:.3f}  (set-EM compounding)")
print("  Read: denc small (encoder share) vs margin collapse SEEN->NOVEL big (decoder combo-memorization share).")
