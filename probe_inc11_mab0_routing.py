"""
Is the RESIDUAL faint-minor absorption (after mab0 is already sigmoid) STILL a mab0-routing problem,
or has it moved downstream? Two direct tests on the trained inc11_nc_mab0 checkpoint.

[STAGE]  linear identity probe (refit per stage) on the faint minor's PRIVATE peaks at:
            x0  (pre-encoder)  ->  H0 (after ISAB layer 0)  ->  H1 (after layer 1 = final H)
         The stage where ->minor collapses to ->major is where absorption is INTRODUCED.

[GATE]   for residual-ABSORBED faint private peaks, the mab0 sigmoid GATE vector over the m inducing slots:
            overlap(faint-peak gate, ABSORBER-major's peak gates)   vs   overlap(., a NON-absorber major)
         high overlap with the absorber => mab0 routes the faint peak into the SAME inducing slots as its
         absorber (a routing collision = the constraint IS mab0 routing). ~equal => absorption is NOT a mab0
         routing collision (it is downstream: mab1 readout / layer-2) => 'de-competition in mab0' is WRONG.
         Measured at BOTH ISAB layers.
"""
import sys, json, numpy as np, torch, torch.nn as nn, math
from pathlib import Path
sys.path.insert(0, ".")
from models.set_transformer import SetTransformerMixture
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w")
RUN=Path(sys.argv[1]) if len(sys.argv)>1 else Path("results/inc11_nc_mab0_seed42")
DEV="cuda" if torch.cuda.is_available() else "cpu"; geno=load_raw_genotypes()
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg["n_token_feats"]
model=SetTransformerMixture(n_loci=cfg.get("n_loci",24),d_locus=cfg.get("d_locus",16),d_model=cfg.get("d_model",128),
    n_heads=cfg.get("n_heads",4),n_isab=cfg.get("n_isab",2),m_inducing=cfg.get("m_inducing",32),n_classes=45,n_noc=6,
    dropout=cfg.get("dropout",0.1),cls_decoder=cfg.get("cls_decoder","pooled"),decoder_source=cfg.get("decoder_source","encoded"),
    n_token_feats=n_tok,encoder=cfg.get("encoder","isab"),num_embed=cfg.get("num_embed","raw"),n_freq=cfg.get("n_freq",8),
    d_num_emb=cfg.get("d_num_emb",8),periodic_sigma=cfg.get("periodic_sigma",1.0),aux_heads=cfg.get("aux_heads",False),
    sparse_attn=cfg.get("sparse_attn",False),nc_attn=cfg.get("nc_attn","none"),nc_learnable_bias=cfg.get("nc_learnable_bias",False)).to(DEV)
model.load_state_dict(torch.load(RUN/"best_model.pt",map_location=DEV,weights_only=True),strict=False); model.eval()
def load(s):
    return (np.load(DATA/f"tokens8_{s}.npy")[:,:,:n_tok].astype(np.float32),
            np.load(DATA/f"mask_{s}.npy").astype(bool), np.load(DATA/f"attr_{s}.npy"))
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")

@torch.no_grad()
def stages(t,m):
    """return x0,H0,H1 per-peak reps + pad_mask."""
    x0,pad=model._project_tokens(t,m)
    H0=model.encoder[0](x0,pad_mask=pad)
    H1=model.encoder[1](H0,pad_mask=pad)
    return x0,H0,H1,pad

@torch.no_grad()
def mab0_gate(layer, Xpeaks_rep, pad):
    """sigmoid gate A (B,h,m,L) for ISAB `layer`'s mab0, given the per-peak input to that layer."""
    isab=model.encoder[layer]; mab=isab.mab0
    I=isab.I.expand(Xpeaks_rep.size(0),-1,-1)            # inducing queries
    Yn=mab.norm_kv(Xpeaks_rep,pad)                       # keys from normed peaks
    B,Lq,d=I.shape; Lk=Yn.size(1)
    q=mab.wq(I).view(B,Lq,mab.h,mab.dh).transpose(1,2)
    k=mab.wk(Yn).view(B,Lk,mab.h,mab.dh).transpose(1,2)
    sc=(q@k.transpose(-2,-1))/math.sqrt(mab.dh)
    nval=(~pad).sum(1).clamp(min=1).to(sc.dtype); b=-torch.log(nval).view(B,1,1,1)
    if mab.bias is not None: b=b+mab.bias
    A=torch.sigmoid(sc+b)
    A=A.masked_fill(pad[:,None,None,:],0.0)
    return A      # (B,h,m,L)

def fit_probe(Hs,Ds):
    X=torch.from_numpy(np.concatenate(Hs).astype(np.float32)).to(DEV)
    y=torch.from_numpy(np.concatenate(Ds).astype(int)).long().to(DEV)
    clf=nn.Linear(X.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(50):
        perm=torch.randperm(len(y),device=DEV)
        for s in range(0,len(y),8192):
            bb=perm[s:s+8192]; opt.zero_grad(); lf(clf(X[bb]),y[bb]).backward(); opt.step()
    return clf.weight.detach().cpu().numpy(), clf.bias.detach().cpu().numpy()
def private_of(g,d,info,at,tk):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

# fit 3 stage probes on train
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); sel=rng.choice(len(at_tr),size=4000,replace=False)
Sx=[[],[]];S0=[[],[]];S1=[[],[]]
for s in range(0,len(sel),128):
    b=sel[s:s+128]; t=torch.from_numpy(tk_tr[b]).to(DEV); m=torch.from_numpy(mk_tr[b]).to(DEV)
    x0,H0,H1,pad=stages(t,m); x0=x0.cpu().numpy();H0=H0.cpu().numpy();H1=H1.cpu().numpy()
    for j,gi in enumerate(b):
        a=at_tr[gi]; v=np.where(a>=0)[0]
        Sx[0].append(x0[j][v]);Sx[1].append(a[v]);S0[0].append(H0[j][v]);S0[1].append(a[v]);S1[0].append(H1[j][v]);S1[1].append(a[v])
Wx,Bx=fit_probe(Sx[0],Sx[1]);W0,B0=fit_probe(S0[0],S0[1]);W1,B1=fit_probe(S1[0],S1[1])
def pred(H,W,B,contribs):
    Z=H@W.T+B; return contribs[int(np.argmax([Z[c] for c in contribs]))]

# test N5
tk,mk,at=load("test")
samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    if len(order)==5: samp[gi]={"info":info}
ids=list(samp.keys())

stage_cnt={"x0":[0,0],"H0":[0,0],"H1":[0,0]}   # [->minor, ->major]
ov_abs=[];ov_ctrl=[];ov_abs1=[];ov_ctrl1=[]
for s in range(0,len(ids),64):
    bids=ids[s:s+64]; t=torch.from_numpy(tk[bids]).to(DEV); m=torch.from_numpy(mk[bids]).to(DEV)
    x0,H0,H1,pad=stages(t,m)
    A0=mab0_gate(0,x0,pad).cpu().numpy(); A1=mab0_gate(1,H0,pad).cpu().numpy()
    x0=x0.cpu().numpy();H0n=H0.cpu().numpy();H1n=H1.cpu().numpy()
    for j,g in enumerate(bids):
        info=samp[g]["info"]; d=[dd for dd in info if info[dd]["rank"]==4][0]; contribs=list(info.keys())
        a=at[g]; v=np.where(a>=0)[0]; priv=private_of(g,d,info,at,tk)
        pk=[p for p in v if int(a[p])==d and (int(tk[g][p,0]),akey(tk[g][p,1])) in priv]
        for p in pk:
            for nm,(H,W,B) in [("x0",(x0[j],Wx,Bx)),("H0",(H0n[j],W0,B0)),("H1",(H1n[j],W1,B1))]:
                pr=pred(H[p],W,B,contribs)
                if pr==d: stage_cnt[nm][0]+=1
                elif info[pr]["rank"]<4: stage_cnt[nm][1]+=1
            # gate collision only for FINAL-absorbed peaks
            pr1=pred(H1n[j][p],W1,B1,contribs)
            if pr1!=d and info[pr1]["rank"]<4:
                absb=pr1
                ab_pk=[q for q in v if int(a[q])==absb]
                majs=[c for c in contribs if info[c]["rank"]<4 and c!=absb]
                if not ab_pk or not majs: continue
                def gate(A,peak): h=A[j][:,:,peak].reshape(-1); n=np.linalg.norm(h); return h/n if n>0 else h
                def mgate(A,pks): G=np.mean([A[j][:,:,q].reshape(-1) for q in pks],0); n=np.linalg.norm(G); return G/n if n>0 else G
                ga0=gate(A0,p); Gab0=mgate(A0,ab_pk); ctrlpk=[q for q in v if int(a[q])==majs[0]]; Gct0=mgate(A0,ctrlpk)
                ov_abs.append(float(ga0@Gab0)); ov_ctrl.append(float(ga0@Gct0))
                ga1=gate(A1,p); Gab1=mgate(A1,ab_pk); Gct1=mgate(A1,ctrlpk)
                ov_abs1.append(float(ga1@Gab1)); ov_ctrl1.append(float(ga1@Gct1))

print(f"=== {RUN.name}: where is the RESIDUAL faint-minor absorption introduced? ===")
print("  [STAGE] faint private-peak identity ->minor / ->major  (probe refit per stage)")
for nm in ["x0","H0","H1"]:
    a,b=stage_cnt[nm]; tot=a+b
    print(f"    {nm:3}:  ->minor {a/tot:.2f}   ->major {b/tot:.2f}   (n={tot})")
print(f"\n  [GATE] mab0 gate overlap of an ABSORBED faint peak with its ABSORBER vs a control major (n={len(ov_abs)})")
print(f"    layer0 mab0:  overlap(absorber) {np.mean(ov_abs):.3f}   overlap(control) {np.mean(ov_ctrl):.3f}   diff {np.mean(ov_abs)-np.mean(ov_ctrl):+.3f}")
print(f"    layer1 mab0:  overlap(absorber) {np.mean(ov_abs1):.3f}   overlap(control) {np.mean(ov_ctrl1):.3f}   diff {np.mean(ov_abs1)-np.mean(ov_ctrl1):+.3f}")
print("\n  diff>>0 => faint peak shares mab0 slots with its absorber = ROUTING COLLISION in mab0 (claim supported).")
print("  diff~0  => not a mab0 routing collision; residual absorption is downstream (claim NOT supported).")
