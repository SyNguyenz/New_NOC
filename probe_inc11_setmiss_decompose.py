"""
Have we explained the ~0.18-0.21 ENCODER set-miss? We only ever studied the rank-4 faintest minor. But an
N5 set-EM miss needs ALL 5 right. Decompose the set-miss itself (H1 soft-vote = the encoder ceiling readout):

[A] per-RANK recall (rank0=loudest major .. rank4=faintest): is the donor of that rank in top-NOC?
    -> product of the 5 per-rank recalls vs ACTUAL set-EM. If product ~ set-EM, the miss is COMPOUNDING of
       small per-donor misses across ALL ranks (multiplicative), NOT a single rank-4 mechanism.
[B] for the set-misses: how many TRUE donors dropped, and of WHICH ranks (is it only rank4, or spread)?
[C] false-positive vs false-negative: among set-misses, is a DECOY (non-contributor) in top-NOC (false-pos)
    or is it purely a true donor dropping out (false-neg)? Decoys-in => a competition mechanism we have not
    probed (a non-contributor out-scoring a true faint donor), distinct from the faint-minor flip.
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,"."); from models.set_transformer import SetTransformerMixture
DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1]) if len(sys.argv)>1 else Path("results/inc11_nc_mab0_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"
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
tk,mk,at=load("test")
rank_hit=defaultdict(lambda:[0,0])   # rank -> [hit,total]
setmiss_droprank=defaultdict(int); n_set=0; n_setmiss=0; n_decoy_in=0; n_truedrop=0; ndrop_list=[]
for s in range(0,len(at),64):
    bids=list(range(s,min(s+64,len(at)))); H=encodeH1(torch.from_numpy(tk[bids]).to(DEV),torch.from_numpy(mk[bids]).to(DEV))
    for j,g in enumerate(bids):
        av=at[g]; v=np.where(av>=0)[0]
        if len(v)==0: continue
        true=set(int(x) for x in np.unique(av[v]))
        if len(true)!=5: continue
        lh=tk[g][:,2]; load_h={int(d):float(np.exp(lh[av==d]).sum()) for d in true}
        order=sorted(true,key=lambda d:-load_h[d])   # rank0=loudest
        z=H[j][v]@W.T+B; z-=z.max(1,keepdims=True); P=np.exp(z);P/=P.sum(1,keepdims=True); vote=P.sum(0)
        top=set(np.argsort(vote)[::-1][:5].tolist())
        n_set+=1
        for r,d in enumerate(order): rank_hit[r][0]+= (d in top); rank_hit[r][1]+=1
        if top!=true:
            n_setmiss+=1; dropped=true-top; ndrop_list.append(len(dropped))
            for d in dropped: setmiss_droprank[order.index(d)]+=1
            if any(x not in true for x in top): n_decoy_in+=1
            if dropped: n_truedrop+=1
print(f"=== {RUN.name}: N5 set-miss decomposition (H1 soft-vote = encoder ceiling) ===")
print(f"  set-EM = {1-n_setmiss/n_set:.3f}   (n={n_set})")
print("  [A] per-RANK recall (rank0=loudest major .. rank4=faintest):")
prod=1.0
for r in range(5):
    rec=rank_hit[r][0]/rank_hit[r][1]; prod*=rec
    print(f"      rank{r}: recall {rec:.3f}")
print(f"      PRODUCT of per-rank recalls = {prod:.3f}   vs actual set-EM {1-n_setmiss/n_set:.3f}")
print(f"      => if product ~ set-EM: the miss is COMPOUNDING across ranks (multiplicative), not one rank.")
print(f"  [B] set-misses: {n_setmiss}.  mean #true-donors dropped per miss = {np.mean(ndrop_list):.2f}")
print(f"      which rank drops (count): " + "  ".join(f"r{r}={setmiss_droprank[r]}" for r in range(5)))
print(f"  [C] of {n_setmiss} set-misses: {n_decoy_in} have a DECOY in top5 (false-pos), {n_truedrop} have a true donor dropped.")
print(f"      decoy-in fraction = {n_decoy_in/max(n_setmiss,1):.2f}  (high => an unprobed decoy-competition mechanism)")
