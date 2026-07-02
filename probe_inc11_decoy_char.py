"""
Characterize the DECOY that wins a true faint donor's slot (the overlooked mechanism). For each N5 set-miss
and each DROPPED true donor d, find the winning decoy(s) (non-contributors in top5). Ask:
  - is the winning decoy a GENOTYPE NEAR-TWIN of the dropped donor?  Jaccard(geno d, geno decoy) vs the
    panel-baseline Jaccard(d, random donor).  high => panel-intrinsic ambiguity (info-ish floor).
  - how many PRIVATE alleles (vs the 4 OTHER true contributors) did the dropped donor actually have present?
    few => under-determined: its evidence is mostly shared, so a decoy covering the same shared alleles wins.
  - does the decoy's present-allele COVERAGE exceed the dropped donor's?  (decoy explains the peaks as well.)
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
def gset(d): return set((L,a) for L,al in geno.get(KNOWN[d],{}).items() for a in al)
def jacc(a,b):
    A=gset(a);Bs=gset(b); u=len(A|Bs); return len(A&Bs)/u if u else 0.0
# panel baseline jaccard
alld=list(range(45)); base_j=np.mean([jacc(i,j) for i in alld for j in alld if i!=j])

tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); sel=rng.choice(len(at_tr),size=5000,replace=False)
HH=[];DD=[]
for s in range(0,len(sel),128):
    b=sel[s:s+128]; H=encodeH1(torch.from_numpy(tk_tr[b]).to(DEV),torch.from_numpy(mk_tr[b]).to(DEV))
    for j,gi in enumerate(b):
        av=at_tr[gi]; v=np.where(av>=0)[0]; HH.append(H[j][v]); DD.append(av[v])
W,B=fit(HH,DD)
tk,mk,at=load("test")
twin_j=[]; base_list=[]; npriv_drop=[]; cov_drop=[]; cov_decoy=[]
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
        present=set((int(tk[g][p,0]),akey(tk[g][p,1])) for p in v)
        for d in dropped:
            # nearest decoy by genotype to the dropped donor
            jd=max((jacc(d,x) for x in decoys), default=0.0); twin_j.append(jd); base_list.append(base_j)
            others=[o for o in true if o!=d]
            oh=set().union(*[gset(o) for o in others])
            npriv=len(set(p for p in present if p in gset(d)) - oh); npriv_drop.append(npriv)
            cov_drop.append(len(present & gset(d))/max(len(gset(d)),1))
            # the winning decoy's coverage
            wd=max(decoys,key=lambda x: vote[x]); cov_decoy.append(len(present & gset(wd))/max(len(gset(wd)),1))
print(f"=== {RUN.name}: DECOY characterization on N5 set-misses (n_dropped={len(twin_j)}) ===")
print(f"  panel baseline Jaccard(random pair)            : {base_j:.3f}")
print(f"  Jaccard(dropped true, nearest winning decoy)   : {np.mean(twin_j):.3f}   (>> baseline => genotype NEAR-TWIN confusion)")
print(f"  dropped donor's #PRIVATE alleles present       : mean {np.mean(npriv_drop):.1f}  median {int(np.median(npriv_drop))}  (<=2 frac {np.mean(np.array(npriv_drop)<=2):.0%})")
print(f"  present-allele coverage  dropped {np.mean(cov_drop):.2f}  vs winning decoy {np.mean(cov_decoy):.2f}")
print("\n  near-twin Jaccard >> baseline AND dropped npriv low => the decoy is a genotype near-duplicate that")
print("  explains the same (mostly shared) peaks => panel-intrinsic under-determination (info-ish floor),")
print("  not a generic model error. coverage(decoy) >= coverage(dropped) => decoy explains the peaks as well.")
