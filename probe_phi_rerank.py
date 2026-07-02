"""
DECISIVE oracle test: does the independent soft-split phi (uniform compat + genotype-mask + height EM,
NO neural attr) RERANK the cls candidates to improve N5 oracle?  Deployable (no privileged info);
alpha tuned on real VAL, evaluated on real TEST (avoids the F34 re-rank artifact).
"""
import os, json
from pathlib import Path
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
DA=Path("data_insilico_w"); RUN=Path(os.environ.get("RUN","results/inc13_B_distill_seed42")); G="data/donor_geno.npy"
DEVc=torch.device("cuda" if torch.cuda.is_available() else "cpu"); NITER=int(os.environ.get("NITER","5"))
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
@torch.no_grad()
def clslogits(t,k):
    o=[]
    for i in range(0,len(t),128):
        r=m(torch.from_numpy(t[i:i+128]).to(DEVc),torch.from_numpy(k[i:i+128].astype(bool)).to(DEVc))
        o.append(r["logits_cls"].cpu().numpy())
    return np.concatenate(o)
def softphi(tk,mk):
    N=len(tk); PH=np.zeros((N,C))
    for i in range(N):
        peaks=[(k,kk(tk[i,k,0],tk[i,k,1]),np.expm1(tk[i,k,2])) for k in np.where(mk[i])[0] if kk(tk[i,k,0],tk[i,k,1]) in carr]
        if not peaks: continue
        n=len(peaks); h=np.array([p[2] for p in peaks])
        S=np.full((n,C+1),-1e9)
        for r,(k,it,_) in enumerate(peaks):
            for c in carr[it]: S[r,c]=0.0      # UNIFORM compat (independent of neural attr)
            S[r,C]=-2.0
        phi=np.ones(C+1)/(C+1)
        for _ in range(NITER):
            z=S+np.log(phi+1e-9); z-=z.max(1,keepdims=True); A=np.exp(z); A/=A.sum(1,keepdims=True)
            w=(A[:,:C]*h[:,None]).sum(0); bg=(A[:,C]*h).sum(); tot=w.sum()+bg
            phi=np.concatenate([w,[bg]])/max(tot,1e-9)
        PH[i]=phi[:C]
    return PH
def z(a): s=a.std(); return (a-a.mean())/(s if s>1e-9 else 1.0)
def load(sp): return (np.load(DA/f"tokens{n_tok}_{sp}.npy").astype(np.float32),np.load(DA/f"mask_{sp}.npy"),
                      np.load(DA/f"y_{sp}_set.npy").astype(bool),np.load(DA/f"noc_{sp}.npy").astype(int))
def oracle(L,PH,y,noc,k,alpha):
    sel=np.where(np.clip(noc,1,5)==k)[0]; hit=0
    for i in sel:
        sc=z(L[i]) + alpha*z(np.log(PH[i]+1e-6))
        top=np.argsort(sc)[::-1][:k]; pr=np.zeros(C,int); pr[top]=1; hit+=int((pr==y[i]).all())
    return hit/max(1,len(sel))
tkv,mkv,yv,nv=load("val"); tkt,mkt,yt,nt=load("test")
Lv,Lt=clslogits(tkv,mkv),clslogits(tkt,mkt); PHv,PHt=softphi(tkv,mkv),softphi(tkt,mkt)
alphas=[0,0.1,0.2,0.3,0.5,0.75,1.0]
valk=[k for k in [5,4,3] if (np.clip(nv,1,5)==k).any()]
best_a,best_v=0,-1
for a in alphas:
    v=np.mean([oracle(Lv,PHv,yv,nv,k,a) for k in valk])
    if v>best_v: best_v,best_a=v,a
print("=== phi-RERANK oracle (uniform soft-split, deployable; alpha tuned on VAL) ===")
print("  alpha:   "+"  ".join(f"{a:>4}" for a in alphas))
for k in [5,4,3,2,1]:
    print(f"  N{k} test:"+"  ".join(f"{oracle(Lt,PHt,yt,nt,k,a):.3f}" for a in alphas))
print(f"\n  >> VAL-selected alpha={best_a} (from NOC{valk})")
for k in [5,4,3,2,1]:
    print(f"     N{k}: model={oracle(Lt,PHt,yt,nt,k,0):.3f} -> rerank={oracle(Lt,PHt,yt,nt,k,best_a):.3f}")
