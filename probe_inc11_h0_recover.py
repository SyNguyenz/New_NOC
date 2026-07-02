"""
Set-level CAUSAL confirmation of the layer1 over-smoothing claim: if layer1 destroys faint-minor identity
that H0 had, then a soft-vote readout on H0 should RECOVER N5 set-EM vs the same readout on H1 — while
possibly TRADING on N3/N4 (layer1 helps kept minors / majors). Probe refit per stage on train (combo-disjoint).
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path
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
def stages(t,m):
    x0,pad=model._project_tokens(t,m); H0=model.encoder[0](x0,pad_mask=pad); H1=model.encoder[1](H0,pad_mask=pad)
    return H0,H1,pad
def fit(Hs,Ds):
    X=torch.from_numpy(np.concatenate(Hs).astype(np.float32)).to(DEV); y=torch.from_numpy(np.concatenate(Ds).astype(int)).long().to(DEV)
    clf=nn.Linear(X.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(50):
        perm=torch.randperm(len(y),device=DEV)
        for s in range(0,len(y),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(X[b]),y[b]).backward(); opt.step()
    return clf.weight.detach().cpu().numpy(), clf.bias.detach().cpu().numpy()
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); sel=rng.choice(len(at_tr),size=4000,replace=False)
A0=[[],[]];A1=[[],[]]
for s in range(0,len(sel),128):
    b=sel[s:s+128]; H0,H1,pad=stages(torch.from_numpy(tk_tr[b]).to(DEV),torch.from_numpy(mk_tr[b]).to(DEV))
    H0=H0.cpu().numpy();H1=H1.cpu().numpy()
    for j,gi in enumerate(b):
        a=at_tr[gi]; v=np.where(a>=0)[0]; A0[0].append(H0[j][v]);A0[1].append(a[v]);A1[0].append(H1[j][v]);A1[1].append(a[v])
W0,B0=fit(A0[0],A0[1]); W1,B1=fit(A1[0],A1[1])
# CONCAT [H0;H1] probe = no-train proxy for DenseFormer-style depth aggregation (does a readout with access
# to BOTH the early private identity AND the late set-context recover N5 that H1 alone misses?)
Ac=[[np.concatenate([h0,h1],1) for h0,h1 in zip(A0[0],A1[0])],A1[1]]
Wc,Bc=fit(Ac[0],Ac[1])
def vote(H,W,B):
    z=H@W.T+B; z-=z.max(1,keepdims=True); P=np.exp(z); P/=P.sum(1,keepdims=True); return P.sum(0)
tk,mk,at=load("test")
from collections import defaultdict
em=defaultdict(lambda:{"H0":0,"H1":0,"cat":0,"n":0})
for s in range(0,len(at),64):
    bids=list(range(s,min(s+64,len(at)))); H0,H1,pad=stages(torch.from_numpy(tk[bids]).to(DEV),torch.from_numpy(mk[bids]).to(DEV))
    H0=H0.cpu().numpy();H1=H1.cpu().numpy()
    for j,g in enumerate(bids):
        a=at[g]; v=np.where(a>=0)[0]
        if len(v)==0: continue
        true=set(int(x) for x in np.unique(a[v])); noc=len(true)
        t0=set(np.argsort(vote(H0[j][v],W0,B0))[::-1][:noc]); t1=set(np.argsort(vote(H1[j][v],W1,B1))[::-1][:noc])
        hc=np.concatenate([H0[j][v],H1[j][v]],1); tc=set(np.argsort(vote(hc,Wc,Bc))[::-1][:noc])
        E=em[noc]; E["n"]+=1; E["H0"]+=(t0==true); E["H1"]+=(t1==true); E["cat"]+=(tc==true)
print(f"=== {RUN.name}: H0 / H1 / [H0;H1] depth-aggregation set-EM (probe refit per stage) ===")
print(f"  NOC    n     H1-vote   H0-vote   [H0;H1]   delta(cat-H1)")
for noc in range(1,6):
    E=em[noc]; n=max(E["n"],1)
    print(f"   {noc}   {E['n']:5d}    {E['H1']/n:.3f}    {E['H0']/n:.3f}    {E['cat']/n:.3f}    {E['cat']/n-E['H1']/n:+.3f}")
print("  cat-H1 > 0 on N5 => combining early(private) + late(context) RECOVERS faint minors")
print("                      => DenseFormer-style depth aggregation is a GO (feasibility).")
print("  cat ~ H1 => the info is not recoverable even by combining layers (fix family won't help).")
