"""
Confirm whether the decoy-competition lever (deployable genotype-coverage added to the score) beats the
flat +.02, and WHERE it lands by rank. Two bases: the model's own decode (.747) and the H1 soft-vote ceiling
(.82). Two coverage scores (deployable: mask peaks incl noise, NO privileged at>=0):
    COV    = |present ∩ geno[d]| / |geno[d]|
    RARITY = same, each matched allele weighted by panel-rarity 1/freq (discriminative => decoy-suppressing)
Rank by  base_logit + lam * zscore(score); report N5 set-EM + per-RANK recall (rank0..4) + guard N1-N4.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
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
def gset(d): return set((L,a) for L,al in geno.get(KNOWN[d],{}).items() for a in al)
gsize={d:max(len(gset(d)),1) for d in range(45)}
freq=defaultdict(int)
for d in range(45):
    for p in gset(d): freq[p]+=1
@torch.no_grad()
def encodeH1(t,m):
    _,H,_=model._encode_set(t,m); return H.cpu().numpy()
@torch.no_grad()
def mscore(t,m):
    o=model(t,m); return torch.sigmoid(o["logits_cls"]).cpu().numpy()
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
def z(x): s=x.std(); return (x-x.mean())/s if s>1e-9 else x*0
tk,mk,at=load("test")
# precompute per-sample: model probs, softvote, coverage scores, true set+ranks
S=[]
for s in range(0,len(at),64):
    bids=list(range(s,min(s+64,len(at)))); t=torch.from_numpy(tk[bids]).to(DEV); m=torch.from_numpy(mk[bids]).to(DEV)
    H=encodeH1(t,m); MS=mscore(t,m)
    for j,g in enumerate(bids):
        av=at[g]; v=np.where(av>=0)[0]
        if len(v)==0: continue
        true=set(int(x) for x in np.unique(av[v])); noc=len(true)
        lh=tk[g][:,2]; load_h={int(d):float(np.exp(lh[av==d]).sum()) for d in true}
        order=sorted(true,key=lambda d:-load_h[d])
        zz=H[j][v]@W.T+B; zz-=zz.max(1,keepdims=True); Pp=np.exp(zz);Pp/=Pp.sum(1,keepdims=True); sv=Pp.sum(0)
        present=set()
        for p in range(tk.shape[1]):
            if mk[g,p] and float(tk[g,p,1])>0: present.add((int(tk[g,p,0]),akey(tk[g,p,1])))
        cov=np.array([len(present & gset(d))/gsize[d] for d in range(45)])
        rar=np.array([sum(1.0/freq[p] for p in (present & gset(d)))/sum(1.0/freq[p] for p in gset(d)) for d in range(45)])
        S.append((true,noc,order,np.log(MS[j]+1e-6)-np.log(1-MS[j]+1e-6),np.log(sv+1e-6),cov,rar))
def evalc(base_key,score_key,lam):
    em=defaultdict(lambda:[0,0]); rankrec=defaultdict(lambda:[0,0])
    for (true,noc,order,ml,svl,cov,rar) in S:
        base= ml if base_key=="model" else svl
        sc= z(cov) if score_key=="cov" else z(rar)
        comb=base+lam*sc; top=set(np.argsort(comb)[::-1][:noc].tolist())
        em[noc][0]+=(top==true); em[noc][1]+=1
        if noc==5:
            for r,d in enumerate(order): rankrec[r][0]+=(d in top); rankrec[r][1]+=1
    return em,rankrec
print(f"=== {RUN.name}: deployable coverage logit-add — set-EM + per-rank (N5) ===")
for base_key in ["model","softvote"]:
    base_em,_=evalc(base_key,"cov",0.0)
    b5=base_em[5][0]/base_em[5][1]
    print(f"\n  BASE={base_key}:  N5 set-EM {b5:.3f}  (N1 {base_em[1][0]/base_em[1][1]:.3f} N2 {base_em[2][0]/base_em[2][1]:.3f} N3 {base_em[3][0]/base_em[3][1]:.3f} N4 {base_em[4][0]/base_em[4][1]:.3f})")
    for score_key in ["cov","rar"]:
        best=None
        for lam in [0.25,0.5,1.0,1.5,2.0,3.0]:
            em,rr=evalc(base_key,score_key,lam); e5=em[5][0]/em[5][1]
            if best is None or e5>best[1]: best=(lam,e5,em,rr)
        lam,e5,em,rr=best
        guard=f"N1 {em[1][0]/em[1][1]:.3f} N2 {em[2][0]/em[2][1]:.3f} N3 {em[3][0]/em[3][1]:.3f} N4 {em[4][0]/em[4][1]:.3f}"
        rk=" ".join(f"r{r} {rr[r][0]/rr[r][1]:.3f}" for r in range(5))
        print(f"    +{score_key:4} best lam={lam:.2f}: N5 {e5:.3f} (delta {e5-b5:+.3f}) | guard {guard} | rank-rec {rk}")
print("\n  delta>>+.02 on a base, landing on rank2/3, without guard collapse => decoy-coverage lever is real & targeted.")
