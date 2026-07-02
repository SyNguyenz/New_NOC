"""
Counterfactual feasibility for the mechanism-grounded fix: if the mab1 writeback `a` muddies the faint
peak's residual identity, then SCALING the writeback (H1(lam)= (H0+lama) + FFN(SetNorm(H0+lama))) with lam<1 should
RECOVER the missed faint minor's margin — WITHOUT collapsing the kept minors or the majors. This is the
no-train proxy for a LayerScale / ReZero / gated-residual on the mab1 writeback branch (Touvron CaiT 2021;
Bachlechner ReZero 2020). Reports per-faint-peak margin AND a global-lam N5/N4 set-EM (recompute the whole
encoder layer1 with a scaled writeback) so we see the set-level trade.
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
isab1=model.encoder[1]; mab=isab1.mab1
@torch.no_grad()
def H1_lambda(t,m,lam):
    x0,pad=model._project_tokens(t,m); H0=model.encoder[0](x0,pad_mask=pad)
    I=isab1.I.expand(H0.size(0),-1,-1); Hind=isab1.mab0(I,H0,q_mask=None,kv_mask=pad)
    Xn=mab.norm_q(H0,pad) if mab.norm_q is not None else H0
    a,_=mab.attn(Xn, mab.norm_kv(Hind,None), Hind, key_padding_mask=None, need_weights=False)
    H=H0+lam*a; H1=H+mab.ff(mab.norm_h(H,pad))
    return H0.cpu().numpy(), H1.cpu().numpy(), pad
def fit(Hs,Ds):
    X=torch.from_numpy(np.concatenate(Hs).astype(np.float32)).to(DEV); y=torch.from_numpy(np.concatenate(Ds).astype(int)).long().to(DEV)
    clf=nn.Linear(X.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(50):
        perm=torch.randperm(len(y),device=DEV)
        for s in range(0,len(y),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(X[b]),y[b]).backward(); opt.step()
    return clf.weight.detach().cpu().numpy(), clf.bias.detach().cpu().numpy()
# fit W1 on the TRUE (lam=1) H1 train
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); sel=rng.choice(len(at_tr),size=4000,replace=False)
HH=[];DD=[]
for s in range(0,len(sel),128):
    b=sel[s:s+128]; _,H1,_=H1_lambda(torch.from_numpy(tk_tr[b]).to(DEV),torch.from_numpy(mk_tr[b]).to(DEV),1.0)
    for j,gi in enumerate(b):
        av=at_tr[gi]; v=np.where(av>=0)[0]; HH.append(H1[j][v]); DD.append(av[v])
W1,B1=fit(HH,DD)
tk,mk,at=load("test"); samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    samp[gi]={"info":info,"true":set(int(x) for x in np.unique(a[v])),"noc":len(order)}
ids=list(samp.keys())
def vote(H): z=H@W1.T+B1; z-=z.max(1,keepdims=True); P=np.exp(z);P/=P.sum(1,keepdims=True); return P.sum(0)
print(f"=== {RUN.name}: writeback-scale (lam) counterfactual ===")
print("  lam      N5 set-EM   N4 set-EM   N3 set-EM   N1 set-EM   (H1-vote, global lam on mab1 writeback)")
for lam in [1.0,0.75,0.5,0.25,0.0]:
    em={1:[0,0],3:[0,0],4:[0,0],5:[0,0]}
    for s in range(0,len(ids),48):
        bids=ids[s:s+48]; _,H1,_=H1_lambda(torch.from_numpy(tk[bids]).to(DEV),torch.from_numpy(mk[bids]).to(DEV),lam)
        for j,g in enumerate(bids):
            av=at[g]; v=np.where(av>=0)[0]
            if len(v)==0: continue
            noc=samp[g]["noc"]; t=set(np.argsort(vote(H1[j][v]))[::-1][:noc])
            if noc in em: em[noc][0]+=(t==samp[g]["true"]); em[noc][1]+=1
    def r(n): return em[n][0]/max(em[n][1],1)
    print(f"  {lam:.2f}     {r(5):.3f}       {r(4):.3f}       {r(3):.3f}       {r(1):.3f}")
print("\n  N5 rises as lam drops WITHOUT N1/N3/N4 collapsing => gated/scaled writeback (LayerScale/ReZero) is a GO.")
print("  N5 flat or everything drops => writeback scaling is not the lever.")
