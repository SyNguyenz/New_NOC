"""
De-risk the soft-split (Sinkhorn/EM) mechanism BEFORE training. Take inc13_B's attr LOGITS as the
peak->donor compatibility S[k,c]; apply genotype mask (carriers + background sink); run unrolled EM
(= differentiable Sinkhorn-lite): A=softmax_c(S+log phi) over carriers, phi=height-weighted share,
iterate. PRIVATE peaks (single carrier) ANCHOR phi -> shared peaks split by compat*phi. Measure if
this phi separates the faint minor from the decoy (where hard-argmax attr failed: 0.12) and finds the
faintest donor (phi_head failed: 10.5% < random).  Feasibility-filter applied (drop no-carrier peaks).
NO privileged info (carriers/heights only); GT used to score.
"""
import os, json
from pathlib import Path
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
DA=Path("data_insilico_w"); RUN=Path("results/inc13_B_distill_seed42"); G="data/donor_geno.npy"
DEVc=torch.device("cuda" if torch.cuda.is_available() else "cpu")
NITER=int(os.environ.get("EM_ITER","5")); BG=float(os.environ.get("BG_PRIOR","0.02"))
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
dos=[{} for _ in range(C)]
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]: it=kk(g[c,j,0],g[c,j,1]); dos[c][it]=dos[c].get(it,0)+1
carr={}
for c in range(C):
    for it in dos[c]: carr.setdefault(it,[]).append(c)

cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
    dropout=0.1,cls_decoder="per_donor",decoder_source="encoded",n_token_feats=n_tok,encoder="isab++",dec_layers=2,
    num_embed="periodic",n_freq=8,d_num_emb=8,periodic_sigma=0.3,aux_heads=True,sparse_attn=True).to(DEVc)
sd=torch.load(RUN/"best_model.pt",map_location=DEVc); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
m.load_state_dict(sd,strict=False); m.eval()
tk=np.load(DA/f"tokens{n_tok}_test.npy").astype(np.float32); mk=np.load(DA/"mask_test.npy")
y_p=np.load(RUN/"y_test_pred.npy").astype(bool); y_t=np.load(DA/"y_test_set.npy").astype(bool); noc=np.load(DA/"noc_test.npy")
H=np.expm1(tk[:,:,2])
@torch.no_grad()
def attr_logits(t,k):
    o=[]
    for i in range(0,len(t),128):
        r=m(torch.from_numpy(t[i:i+128]).to(DEVc),torch.from_numpy(k[i:i+128].astype(bool)).to(DEVc))
        o.append(r["logits_attr"].cpu().numpy())
    return np.concatenate(o)
sel=np.where(noc==5)[0]; AL=attr_logits(tk[sel],mk[sel])

COMPAT=os.environ.get("COMPAT","attr")   # attr | uniform | coverage
def softsplit_phi(s_i,i):
    # feasible peaks (carrier exists) + heights + per-peak carrier list
    peaks=[];
    for k in np.where(mk[i])[0]:
        it=kk(tk[i,k,0],tk[i,k,1])
        if it in carr: peaks.append((k,it,H[i,k]))
    if not peaks: return np.zeros(C)
    lg=AL[s_i]; n=len(peaks); h=np.array([p[2] for p in peaks])
    Oset=set(it for _,it,_ in peaks)
    cov=np.array([len(set(dos[c])&Oset)/max(1,len(dos[c])) for c in range(C)])  # genotype-coverage prior
    S=np.full((n,C+1),-1e9);
    for r,(k,it,_) in enumerate(peaks):
        for c in carr[it]:
            S[r,c]= lg[k,c] if COMPAT=="attr" else (0.0 if COMPAT=="uniform" else 4.0*cov[c])
        S[r,C]= lg[k,C] if COMPAT=="attr" else -2.0   # background sink
    phi=np.ones(C+1)/(C+1); phi[C]=BG
    for _ in range(NITER):
        z=S+np.log(phi+1e-9); z-=z.max(1,keepdims=True)
        A=np.exp(z); A/=A.sum(1,keepdims=True)          # row softmax over carriers+bg
        w=(A[:,:C]*h[:,None]).sum(0)                     # height-weighted share per donor
        tot=w.sum()+ (A[:,C]*h).sum()
        phi=np.concatenate([w,[ (A[:,C]*h).sum() ]])/max(tot,1e-9)
    return phi[:C]

def auc(p,nq):
    p,nq=np.asarray(p,float),np.asarray(nq,float)
    if not len(p) or not len(nq): return float("nan")
    a=np.concatenate([p,nq]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
    cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
    return (rk[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(nq))

mt=[]; dc=[]; faint_hit=0; faint_tot=0; wcorr=[]
phi_t=np.load(DA/"phi_test.npy")
for s_i,i in enumerate(sel):
    phi=softsplit_phi(s_i,i)
    true=np.where(y_t[i])[0]
    for c in [c for c in true if not y_p[i,c]]: mt.append(phi[c])
    for c in [c for c in np.where(y_p[i])[0] if not y_t[i,c]]: dc.append(phi[c])
    # faint-minor id + within-sample corr
    cset=true
    if len(cset)>=2:
        fp=cset[np.argmin(phi[cset])]; ft=cset[np.argmin(phi_t[i,cset])]
        faint_tot+=1; faint_hit+=int(fp==ft)
        a=phi[cset]-phi[cset].mean(); b=phi_t[i,cset]-phi_t[i,cset].mean()
        d=a.std()*b.std(); wcorr.append(float((a*b).mean()/d) if d>1e-9 else 0.0)
print("=== SOFT-SPLIT (unrolled EM on masked attr logits) — N5 ===")
print(f"  decoy-AUC (missed-true vs decoy, phi):  {auc(mt,dc):.3f}   [hard-attr=0.12, attr-height=0.21, raw-height~0.61]")
print(f"  within-N5 phi corr:  {np.nanmean(wcorr):+.3f}   [phi_head was -0.14]")
print(f"  faint-minor id:      {faint_hit}/{faint_tot} = {faint_hit/max(1,faint_tot):.3f}   [phi_head 0.105, random 0.20]")
print(f"  (NITER={NITER}, bg_prior={BG})")
