"""
Probe EACH atomic symptom separately on inc11_nc_mab0, to avoid compounding a wrong label.

S0 [CONFOUND CHECK] for the faint private peaks of MISSED vs KEPT minors: was the peak isolated ALREADY at
   H0?  margin m0 = probe_logit(minor) - max_major  at H0 (W0) and m1 at H1 (W1).
     missed peaks m0<=0  -> never isolated -> the failure is LAYER0/encoder-info, NOT 'introduced at layer1'.
     missed peaks m0>0 but m1<0 (FLIP) -> layer1 caused it -> then S1/S2 say which mechanism.

S1 [OVER-SMOOTHING / rank collapse]  token uniformity with depth: mean pairwise cosine + effective rank of
   the valid-peak tokens at x0 / H0 / H1. Rising cosine / falling rank = low-pass smoothing. And per faint
   peak: cos(H1_peak, sample-mean H1) — does the missed peak collapse toward the GLOBAL mean?

S2 [DOMINANT-MAJOR INJECTION]  in H1 space: cos(faint-peak, its OWN-minor peak-mean) vs cos(faint-peak,
   ABSORBER-major peak-mean). Absorbed = high cos to the major, low to own. If cos(peak,absorber) >> cos(peak,
   global-mean), it is a SPECIFIC major injection (sink-like), not isotropic smoothing.
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
def stages(t,m):
    x0,pad=model._project_tokens(t,m); H0=model.encoder[0](x0,pad_mask=pad); H1=model.encoder[1](H0,pad_mask=pad)
    return x0,H0,H1,pad
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
def cos(a,b): na=np.linalg.norm(a);nb=np.linalg.norm(b); return float(a@b/(na*nb)) if na>0 and nb>0 else 0.0

# fit H0,H1 probes
tk_tr,mk_tr,at_tr=load("train"); rng=np.random.default_rng(0); sel=rng.choice(len(at_tr),size=4000,replace=False)
A0=[[],[]];A1=[[],[]]
for s in range(0,len(sel),128):
    b=sel[s:s+128]; x0,H0,H1,pad=stages(torch.from_numpy(tk_tr[b]).to(DEV),torch.from_numpy(mk_tr[b]).to(DEV))
    H0=H0.cpu().numpy();H1=H1.cpu().numpy()
    for j,gi in enumerate(b):
        a=at_tr[gi]; v=np.where(a>=0)[0]; A0[0].append(H0[j][v]);A0[1].append(a[v]);A1[0].append(H1[j][v]);A1[1].append(a[v])
W0,B0=fit(A0[0],A0[1]); W1,B1=fit(A1[0],A1[1])
def margin(H,W,B,minor,majors):
    z=H@W.T+B; return z[minor]-max(z[m] for m in majors)

tk,mk,at=load("test"); samp={}
for gi in range(len(at)):
    a=at[gi]; v=np.where(a>=0)[0]
    if len(v)==0: continue
    lh=tk[gi][:,2]; info={}
    for d in np.unique(a[v]): info[int(d)]={"h":float(np.exp(lh[a==d]).sum())}
    order=sorted(info,key=lambda d:-info[d]["h"])
    for r,d in enumerate(order): info[d]["rank"]=r
    if len(order)==5: samp[gi]={"info":info,"true":set(int(x) for x in np.unique(a[v]))}
ids=list(samp.keys())

# over-smoothing global + per-peak stratified
pc={"x0":[],"H0":[],"H1":[]}; er={"x0":[],"H0":[],"H1":[]}
def pairwise_cos(M):
    Mn=M/ (np.linalg.norm(M,axis=1,keepdims=True)+1e-9); C=Mn@Mn.T; n=len(M)
    return (C.sum()-n)/(n*(n-1)) if n>1 else 0.0
def eff_rank(M):
    sv=np.linalg.svd(M-M.mean(0,keepdims=True),compute_uv=False); return (sv.sum()**2)/(np.square(sv).sum()+1e-9)
rows={"missed":{"m0":[],"m1":[],"iso0":0,"flip":0,"n":0,"cmean":[],"cself":[],"cabs":[]},
      "kept":{"m0":[],"m1":[],"iso0":0,"flip":0,"n":0,"cmean":[],"cself":[],"cabs":[]}}
for s in range(0,len(ids),64):
    bids=ids[s:s+64]; x0,H0,H1,pad=stages(torch.from_numpy(tk[bids]).to(DEV),torch.from_numpy(mk[bids]).to(DEV))
    x0=x0.cpu().numpy();H0=H0.cpu().numpy();H1=H1.cpu().numpy()
    for j,g in enumerate(bids):
        a=at[g]; v=np.where(a>=0)[0]; info=samp[g]["info"]
        for nm,M in [("x0",x0[j][v]),("H0",H0[j][v]),("H1",H1[j][v])]:
            pc[nm].append(pairwise_cos(M)); er[nm].append(eff_rank(M))
        # H1 soft-vote to decide missed/kept
        z1=H1[j][v]@W1.T+B1; z1-=z1.max(1,keepdims=True); P=np.exp(z1);P/=P.sum(1,keepdims=True)
        st=set(np.argsort(P.sum(0))[::-1][:5]); d=[dd for dd in info if info[dd]["rank"]==4][0]
        grp="kept" if d in st else "missed"
        majors=[c for c in info if info[c]["rank"]<4]; priv=private_of(g,d,info,at,tk)
        meanH1=H1[j][v].mean(0); dpk=[p for p in v if int(a[p])==d]; H1d=H1[j][dpk].mean(0) if dpk else meanH1
        for p in [p for p in v if int(a[p])==d and (int(tk[g][p,0]),akey(tk[g][p,1])) in priv]:
            m0=margin(H0[j][p],W0,B0,d,majors); m1=margin(H1[j][p],W1,B1,d,majors)
            R=rows[grp]; R["n"]+=1; R["m0"].append(m0); R["m1"].append(m1)
            R["iso0"]+= (m0>0); R["flip"]+= (m0>0 and m1<0)
            # absorber major in H1 space
            zp=H1[j][p]@W1.T+B1; absb=max(majors,key=lambda c: zp[c])
            apk=[q for q in v if int(a[q])==absb]; Habs=H1[j][apk].mean(0) if apk else meanH1
            R["cmean"].append(cos(H1[j][p],meanH1)); R["cself"].append(cos(H1[j][p],H1d)); R["cabs"].append(cos(H1[j][p],Habs))

print(f"=== {RUN.name}: atomic layer1 mechanism probes ===")
print("S1 [OVER-SMOOTHING] token uniformity by stage (valid peaks):")
for nm in ["x0","H0","H1"]:
    print(f"    {nm:3}: mean pairwise cos {np.mean(pc[nm]):.3f}   effective rank {np.mean(er[nm]):.1f}")
print("\nS0 [CONFOUND] faint private-peak margins (minor vs best-major), missed vs kept minors:")
for grp in ["kept","missed"]:
    R=rows[grp]; n=max(R["n"],1)
    print(f"    {grp:6} (n={R['n']:4d}): mean m0(H0) {np.mean(R['m0']):+.2f}  mean m1(H1) {np.mean(R['m1']):+.2f}  | isolated@H0 {R['iso0']/n:.2f}  flip(H0pos->H1neg) {R['flip']/n:.2f}")
print("\nS2 [DOMINANT-MAJOR vs SMOOTHING] H1-space cosine of the faint private peak:")
for grp in ["kept","missed"]:
    R=rows[grp]
    print(f"    {grp:6}: cos(peak, own-minor) {np.mean(R['cself']):.2f}   cos(peak, absorber-major) {np.mean(R['cabs']):.2f}   cos(peak, global-mean) {np.mean(R['cmean']):.2f}")
print("\n  read: missed iso@H0 LOW => failure is layer0/info (not layer1).  iso@H0 high + flip high => layer1.")
print("        cos(absorber) >> cos(global-mean) => specific major injection (sink-like); ~equal => isotropic over-smoothing.")
