"""
WHERE EXACTLY is the snag in the encoder ceiling (.82)? For each N5 set-miss, take each dropped true donor d
and the decoy wd that took its slot. Look ONLY at the loci where d and wd DIFFER (the discriminating loci),
and ask whether d's DISTINGUISHING alleles (the ones wd does NOT have) are OBSERVED in the mixture:

  - d's distinguishing alleles PRESENT  > wd's distinguishing alleles present  => the evidence FAVORS d, the
    model ranked wrong despite present discriminating evidence  => MODEL ERROR (recoverable).
  - d's distinguishing alleles mostly ABSENT (dropped out / never there)        => the observed peaks cannot
    distinguish d from wd  => genuine INFORMATION FLOOR (the discriminating evidence is gone).

Also split present-but-FAINT: a distinguishing allele observed at low height (near the dropout threshold) is
'present but washable' — counts the in-between case. Deployable: present = mask peaks, allele>0 (no at>=0).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
sys.path.insert(0,"."); from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN
DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1]) if len(sys.argv)>1 else Path("results/inc11_nc_mab0_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; geno=load_raw_genotypes()
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),num_embed=cfg.get("num_embed","raw"),n_freq=cfg.get("n_freq",8),
    d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),aux_heads=cfg.get("aux_heads",False),
    sparse_attn=cfg.get("sparse_attn",False),nc_attn=cfg.get("nc_attn","none"),nc_learnable_bias=cfg.get("nc_learnable_bias",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
def load(s): return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
                     np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
@torch.no_grad()
def encodeH1(t,m):
    _,H,_=model._encode_set(t,m); return H.cpu().numpy()
def fit(Hs,Ds):
    X=torch.from_numpy(np.concatenate(Hs).astype(np.float32)).to(DEV); y=torch.from_numpy(np.concatenate(Ds).astype(int)).long().to(DEV)
    clf=nn.Linear(X.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(50):
        perm=torch.randperm(len(y),device=DEV)
        for s in range(0,len(y),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(X[b]),y[b]).backward(); opt.step()
    return clf.weight.detach().cpu().numpy(), clf.bias.detach().cpu().numpy()
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); sel=rng.choice(len(at_tr),size=5000,replace=False)
HH=[];DD=[]
for s in range(0,len(sel),128):
    b=sel[s:s+128]; H=encodeH1(torch.from_numpy(tk_tr[b]).to(DEV),torch.from_numpy(mk_tr[b]).to(DEV))
    for j,gi in enumerate(b):
        av=at_tr[gi]; v=np.where(av>=0)[0]; HH.append(H[j][v]); DD.append(av[v])
W,B=fit(HH,DD)
def gdict(d): return geno.get(KNOWN[d],{})
tk,mk,at=load("test")
# global height stats for faint threshold
allh=tk[:,:,2][mk & (tk[:,:,1]>0)]; faint_thr=np.percentile(np.exp(allh),20)
d_pres=[]; wd_pres=[]; d_faint=[]; ndisc=[]; favors_d=0; favors_wd=0; tie=0; npairs=0
for s in range(0,len(at),64):
    bids=list(range(s,min(s+64,len(at)))); H=encodeH1(torch.from_numpy(tk[bids]).to(DEV),torch.from_numpy(mk[bids]).to(DEV))
    for j,g in enumerate(bids):
        av=at[g]; v=np.where(av>=0)[0]
        if len(v)==0: continue
        true=set(int(x) for x in np.unique(av[v]))
        if len(true)!=5: continue
        z=H[j][v]@W.T+B; z-=z.max(1,keepdims=True); P=np.exp(z);P/=P.sum(1,keepdims=True); vote=P.sum(0)
        top=set(np.argsort(vote)[::-1][:5].tolist())
        if top==true: continue
        dropped=true-top; decoys=top-true
        # present alleles -> max height
        ph={}
        for p in range(tk.shape[1]):
            if mk[g,p] and float(tk[g,p,1])>0:
                ph[(int(tk[g,p,0]),akey(tk[g,p,1]))]=max(ph.get((int(tk[g,p,0]),akey(tk[g,p,1])),0.0),float(np.exp(tk[g,p,2])))
        for d in dropped:
            wd=max(decoys,key=lambda x: vote[x])   # the decoy that most took the slot
            gd=gdict(d); gw=gdict(wd)
            d_dist=[]; w_dist=[]
            for L in set(gd)|set(gw):
                ad=gd.get(L,set()); aw=gw.get(L,set())
                for a in ad-aw: d_dist.append((L,a))     # d's distinguishing alleles
                for a in aw-ad: w_dist.append((L,a))     # wd's distinguishing alleles
            if not d_dist and not w_dist: continue
            npairs+=1; ndisc.append(len(d_dist))
            dp=sum(1 for x in d_dist if x in ph); wp=sum(1 for x in w_dist if x in ph)
            d_pres.append(dp); wd_pres.append(wp)
            d_faint.append(sum(1 for x in d_dist if x in ph and ph[x]<faint_thr))
            if dp>wp: favors_d+=1
            elif wp>dp: favors_wd+=1
            else: tie+=1
print(f"=== {RUN.name}: discriminating-allele evidence at N5 set-misses (dropped d vs winning decoy wd, n={npairs}) ===")
print(f"  mean #discriminating alleles d-has-not-wd                : {np.mean(ndisc):.1f}")
print(f"  d's distinguishing alleles PRESENT (observed)            : mean {np.mean(d_pres):.2f}")
print(f"  wd's distinguishing alleles present                      : mean {np.mean(wd_pres):.2f}")
print(f"  of d's present distinguishing alleles, # that are FAINT  : mean {np.mean(d_faint):.2f}  (bottom-20% height)")
print(f"  cases where present evidence FAVORS d (dp>wp)            : {favors_d}/{npairs} = {favors_d/max(npairs,1):.0%}  -> MODEL ERROR (recoverable)")
print(f"  cases FAVORS decoy (wp>dp)                                : {favors_wd}/{npairs} = {favors_wd/max(npairs,1):.0%}")
print(f"  cases TIE (dp==wp)                                        : {tie}/{npairs} = {tie/max(npairs,1):.0%}  -> evidence can't distinguish (INFO FLOOR)")
print("\n  FAVORS-d high => model ignores PRESENT discriminating evidence (recoverable).")
print("  TIE/FAVORS-decoy high => discriminating alleles dropped out => observed peaks under-determine d vs wd (info floor).")
