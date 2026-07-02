"""
DECODER mechanism, deeper: WHERE does a faint-minor donor query attend, SEEN vs NOVEL?
inc7 decoder = SparseMAB (sparsemax): each donor query is meant to attend to ITS OWN alleles.
If on novel combos the minor query's attention LEAKS off its own peaks (to majors / scatters),
the combo-overfit lives in the decoder's attention SELECTION (which peaks it picks), and that is
the instance/config memorization that collapses the seen margin.

Capture layer-0 cross-attention (donor query -> peaks). For each TRUE donor, by height-rank:
  own-mass   = attention on peaks attributed to that donor (ground truth)
  major-mass = attention on the 3 major donors' peaks
Compare SEEN(train) vs NOVEL(dev), both synthetic (seed=0 carve).
"""
import sys, json, numpy as np, torch
from pathlib import Path
sys.path.insert(0,".")
import models.set_transformer as st
from models.set_transformer import SetTransformerMixture, sparsemax

DATA=Path("data_insilico_w"); RUN=Path(sys.argv[1] if len(sys.argv)>1 else "results/inc7_masspool_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"
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
print(f"loaded {RUN.name} sparse_attn={cfg.get('sparse_attn')}")

# patch the FIRST decoder SparseMAB layer to stash its attention (mean over heads)
layer0=model.cls_decoder_module.layers[0]
_stash={}
def patched(X,Y,key_padding_mask=None,_self=layer0):
    B,Nq,_=X.shape; Nk=Y.size(1)
    q=_self.q(X).view(B,Nq,_self.h,_self.dh).transpose(1,2)
    k=_self.k(Y).view(B,Nk,_self.h,_self.dh).transpose(1,2)
    v=_self.v(Y).view(B,Nk,_self.h,_self.dh).transpose(1,2)
    scores=(q@k.transpose(-2,-1))/np.sqrt(_self.dh)
    if key_padding_mask is not None: scores=scores.masked_fill(key_padding_mask[:,None,None,:],-1e4)
    attn=sparsemax(scores,dim=-1)
    _stash['a']=attn.mean(1).detach()    # (B,Nq,Nk) mean over heads
    out=(_self.drop(attn)@v).transpose(1,2).reshape(B,Nq,-1)
    H=_self.norm1(X+_self.o(out)); return _self.norm2(H+_self.ff(H))
layer0.forward=patched

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
dmask=dev_mask_seed0(y,noc)
seen5=np.where((~dmask)&(noc==5))[0]; novel5=np.where(dmask&(noc==5))[0]
rng=np.random.default_rng(0); seen5=rng.choice(seen5,size=min(2000,len(seen5)),replace=False)

@torch.no_grad()
def attn_by_rank(idxs,bs=128):
    own=np.zeros(5); maj=np.zeros(5); cnt=np.zeros(5)
    for s in range(0,len(idxs),bs):
        sel=idxs[s:s+bs]; t=torch.from_numpy(tk[sel]).to(DEV); m=torch.from_numpy(mk[sel]).to(DEV)
        model(t,m); A=_stash['a'].cpu().numpy()       # (B,45,N)
        for j,gi in enumerate(sel):
            a=at[gi]; valid=np.where(a>=0)[0]
            dh={int(d):float(np.exp(LH[gi][a==d]).sum()) for d in np.unique(a[valid])}
            order=sorted(dh,key=lambda d:-dh[d]); ro={d:r for r,d in enumerate(order)}; majors=set(order[:3])
            majpk=np.where(np.isin(a,list(majors)))[0]
            for d in np.unique(a[valid]):
                r=ro[int(d)]; av=A[j,int(d)]            # donor d's query attention over peaks
                ownpk=np.where(a==int(d))[0]
                own[r]+=av[ownpk].sum(); maj[r]+=av[majpk].sum(); cnt[r]+=1
    return own/np.maximum(cnt,1), maj/np.maximum(cnt,1)

os_,ms_=attn_by_rank(seen5); on_,mn_=attn_by_rank(novel5)
print("\n=== Layer-0 decoder attention of the TRUE donor's query, by height-rank (0=major..4=faintest) ===")
print(f"  {'rank':>5} | {'own-mass SEEN':>13} {'NOVEL':>7} | {'major-mass SEEN':>15} {'NOVEL':>7}")
for r in range(5):
    print(f"  {r:>5} | {os_[r]:13.3f} {on_[r]:7.3f} | {ms_[r]:15.3f} {mn_[r]:7.3f}")
print("\n  If faint-minor own-mass DROPS and major-mass RISES seen->novel, the decoder's attention")
print("  selection is combo-overfit (picks the wrong/major peaks on novel) = the memorization locus.")
