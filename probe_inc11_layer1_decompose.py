"""
Decompose LAYER1 into its sub-operations to find WHICH one flips the faint minor.
ISABpp layer1 (mab0-arm):  H1 = H0 + a + f
    a = mab1 attention writeback (each H0 peak softmax-attends over the inducing summary; value = inducing)
    f = FFN(SetNorm(H0 + a))
The final readout W1 is linear, so the faint minor's margin (minor vs absorber-major) splits ADDITIVELY:
    margin(H1) = w·H0 + w·a + w·f + b,   w = W1[minor] - W1[absorber]
The sub-operation whose contribution is most NEGATIVE on the missed/flipped peaks = the culprit.
Also: is the writeback `a` pointing toward the ABSORBER (w·a<0 => mab1 injects major content from a
major-dominated inducing summary), and does the faint peak ATTEND mostly to major-loaded inducing slots?
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

isab1=model.encoder[1]
@torch.no_grad()
def layer1_parts(t,m):
    """return per-peak H0, a (writeback), f (ffn), H1, and inducing donor-load. all numpy."""
    x0,pad=model._project_tokens(t,m); H0=model.encoder[0](x0,pad_mask=pad)
    I=isab1.I.expand(H0.size(0),-1,-1)
    Hind=isab1.mab0(I,H0,q_mask=None,kv_mask=pad)            # (B,m,d) inducing summary (sigmoid)
    mab=isab1.mab1
    Xn=mab.norm_q(H0,pad) if mab.norm_q is not None else H0
    a,attw=mab.attn(Xn, mab.norm_kv(Hind,None), Hind, key_padding_mask=None, need_weights=True, average_attn_weights=True)
    H=H0+a; f=mab.ff(mab.norm_h(H,pad)); H1=H+f
    return (H0.cpu().numpy(),a.cpu().numpy(),f.cpu().numpy(),H1.cpu().numpy(),Hind.cpu().numpy(),attw.cpu().numpy(),pad.cpu().numpy())

def fit(Hs,Ds):
    X=torch.from_numpy(np.concatenate(Hs).astype(np.float32)).to(DEV); y=torch.from_numpy(np.concatenate(Ds).astype(int)).long().to(DEV)
    clf=nn.Linear(X.shape[1],45).to(DEV); opt=torch.optim.Adam(clf.parameters(),lr=1e-2,weight_decay=1e-4); lf=nn.CrossEntropyLoss()
    for ep in range(50):
        perm=torch.randperm(len(y),device=DEV)
        for s in range(0,len(y),8192):
            b=perm[s:s+8192]; opt.zero_grad(); lf(clf(X[b]),y[b]).backward(); opt.step()
    return clf.weight.detach().cpu().numpy(), clf.bias.detach().cpu().numpy()
def private_of(g,d,info,at,tk):
    others=[KNOWN[o] for o in info if o!=d]; gX=geno.get(KNOWN[d],{}); priv=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for a in al:
            if a not in oh: priv.add((L,a))
    return priv

# fit W1 on H1 train
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); sel=rng.choice(len(at_tr),size=4000,replace=False)
HH=[];DD=[]
for s in range(0,len(sel),128):
    b=sel[s:s+128]; H0,a,f,H1,Hind,attw,pad=layer1_parts(torch.from_numpy(tk_tr[b]).to(DEV),torch.from_numpy(mk_tr[b]).to(DEV))
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
    if len(order)==5: samp[gi]={"info":info}
ids=list(samp.keys())
# per-inducing-slot dominant donor (probe argmax of Hind) for the attended-slot analysis
res={"missed":{"H0":[],"a":[],"f":[],"awAbsLoad":[],"n":0},"kept":{"H0":[],"a":[],"f":[],"awAbsLoad":[],"n":0}}
for s in range(0,len(ids),48):
    bids=ids[s:s+48]; H0,a,f,H1,Hind,attw,pad=layer1_parts(torch.from_numpy(tk[bids]).to(DEV),torch.from_numpy(mk[bids]).to(DEV))
    for j,g in enumerate(bids):
        av=at[g]; v=np.where(av>=0)[0]; info=samp[g]["info"]; d=[dd for dd in info if info[dd]["rank"]==4][0]
        majors=[c for c in info if info[c]["rank"]<4]; priv=private_of(g,d,info,at,tk)
        # missed/kept by H1 soft-vote
        z=H1[j][v]@W1.T+B1; z-=z.max(1,keepdims=True); P=np.exp(z);P/=P.sum(1,keepdims=True)
        grp="kept" if d in set(np.argsort(P.sum(0))[::-1][:5]) else "missed"
        # inducing-slot donor load: probe Hind, which slots carry the absorber
        zi=Hind[j]@W1.T+B1; slot_owner=zi.argmax(1)   # (m,)
        for p in [int(p) for p in v if int(av[p])==d and (int(tk[g][p,0]),akey(tk[g][p,1])) in priv]:
            zp=H1[j][p]@W1.T+B1; absb=max(majors,key=lambda c: zp[c]); w=W1[d]-W1[absb]
            R=res[grp]; R["n"]+=1
            R["H0"].append(float(w@H0[j][p])); R["a"].append(float(w@a[j][p])); R["f"].append(float(w@f[j][p]))
            # how much attention weight this peak puts on inducing slots OWNED by the absorber major
            R["awAbsLoad"].append(float(attw[j][p][slot_owner==absb].sum()))
print(f"=== {RUN.name}: LAYER1 sub-operation margin decomposition (faint minor vs absorber) ===")
print(f"  contribution to margin (W1·component);  total = H0 + a + f + bias")
for grp in ["kept","missed"]:
    R=res[grp]
    if R["n"]==0: continue
    print(f"  {grp:6} (n={R['n']:4d}):  H0 {np.mean(R['H0']):+.2f}   a(mab1 writeback) {np.mean(R['a']):+.2f}   f(FFN) {np.mean(R['f']):+.2f}   | attn-wt on absorber slots {np.mean(R['awAbsLoad']):.2f}")
print("\n  the most-negative term on 'missed' = the sub-operation that flips the faint minor.")
print("  a<<0 + high attn-wt on absorber slots => mab1 writeback injects major content from a major-loaded inducing summary.")
