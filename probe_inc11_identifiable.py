"""
RIGOROUS retraction test of the 'info floor' claim. For each N5 set-miss (mab0 soft-vote), the dropped true
donor d and the winning decoy wd: is the true set STRICTLY a better explanation than the swap-set
{other-4-contributors + wd}? i.e. does d have a PRESENT allele NOT explained by (other4 ∪ wd)?
  - |d_unique_present| > 0  => true set explains an allele the swap-set cannot => IDENTIFIABLE in principle
                              => the miss is a MODEL ERROR (consistent with the proven '100% rankable').
  - |d_unique_present| == 0 => swap-set explains everything d does => genuine ambiguity for THIS swap.
If identifiable fraction ~ 100% => there is NO info floor; the residual is model-limited (retracts my claim).
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
def gset(d): return set((L,a) for L,al in geno.get(KNOWN[d],{}).items() for a in al)
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
ident=0; ambig=0; uniq=[]
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
        present=set((int(tk[g][p,0]),akey(tk[g][p,1])) for p in v)
        for d in (true-top):
            wd=max((top-true),key=lambda x: vote[x]); other4=true-{d}
            swap_expl=set().union(*[gset(o) for o in other4]) | gset(wd)
            d_unique=(present & gset(d)) - swap_expl
            uniq.append(len(d_unique))
            if len(d_unique)>0: ident+=1
            else: ambig+=1
tot=ident+ambig
print(f"=== {RUN.name}: is the dropped true donor IDENTIFIABLE vs the swap-set (other4 + decoy)? (n={tot}) ===")
print(f"  IDENTIFIABLE (d has a present allele the swap-set can't explain) : {ident}/{tot} = {ident/max(tot,1):.0%}")
print(f"  AMBIGUOUS  (swap-set explains all of d's present alleles)        : {ambig}/{tot} = {ambig/max(tot,1):.0%}")
print(f"  mean # of d's present alleles unexplained by the swap-set        : {np.mean(uniq):.1f}")
print("\n  identifiable ~100% => NO info floor; the residual is MODEL-limited (the model fails to use present")
print("  discriminating evidence). ambiguous>0 => those specific swaps are genuinely under-determined.")
