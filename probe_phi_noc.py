"""
User's claim: a good soft-split phi solves NOC too -> NOC = #donors with meaningful phi after
explaining-away. Test: derive count from the independent soft-split phi (threshold tuned on val,
and participation-ratio), compare to the model's cardinality head. Per-NOC count accuracy.
"""
import os, json
from pathlib import Path
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
DA=Path("data_insilico_w"); RUN=Path(os.environ.get("RUN","results/inc13_B_distill_seed42")); G="data/donor_geno.npy"
DEVc=torch.device("cuda" if torch.cuda.is_available() else "cpu"); NITER=5
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
carr={}
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]: carr.setdefault(kk(g[c,j,0],g[c,j,1]),[]).append(c)
cfg=json.load(open(RUN/"metrics.json"))["config"]; n_tok=cfg.get("n_token_feats",8)
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
    dropout=0.1,cls_decoder="per_donor",decoder_source="encoded",n_token_feats=n_tok,encoder="isab++",dec_layers=2,
    num_embed="periodic",n_freq=8,d_num_emb=8,periodic_sigma=0.3,aux_heads=True,sparse_attn=True).to(DEVc)
sd=torch.load(RUN/"best_model.pt",map_location=DEVc); sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
m.load_state_dict(sd,strict=False); m.eval()
@torch.no_grad()
def card_pred(t,k):
    o=[]
    for i in range(0,len(t),128):
        r=m(torch.from_numpy(t[i:i+128]).to(DEVc),torch.from_numpy(k[i:i+128].astype(bool)).to(DEVc))
        o.append(r["logits_card"].argmax(1).cpu().numpy()+1)
    return np.concatenate(o)
def softphi(tk,mk):
    N=len(tk); PH=np.zeros((N,C))
    for i in range(N):
        peaks=[(k,kk(tk[i,k,0],tk[i,k,1]),np.expm1(tk[i,k,2])) for k in np.where(mk[i])[0] if kk(tk[i,k,0],tk[i,k,1]) in carr]
        if not peaks: continue
        n=len(peaks); h=np.array([p[2] for p in peaks]); S=np.full((n,C+1),-1e9)
        for r,(k,it,_) in enumerate(peaks):
            for c in carr[it]: S[r,c]=0.0
            S[r,C]=-2.0
        phi=np.ones(C+1)/(C+1)
        for _ in range(NITER):
            z=S+np.log(phi+1e-9); z-=z.max(1,keepdims=True); A=np.exp(z); A/=A.sum(1,keepdims=True)
            w=(A[:,:C]*h[:,None]).sum(0); bg=(A[:,C]*h).sum(); phi=np.concatenate([w,[bg]])/max(w.sum()+bg,1e-9)
        PH[i]=phi[:C]
    return PH
def load(sp): return (np.load(DA/f"tokens{n_tok}_{sp}.npy").astype(np.float32),np.load(DA/f"mask_{sp}.npy"),
                      np.clip(np.load(DA/f"noc_{sp}.npy"),1,5))
tkv,mkv,nv=load("val"); tkt,mkt,nt=load("test")
PHv,PHt=softphi(tkv,mkv),softphi(tkt,mkt); Cv,Ct=card_pred(tkv,mkv),card_pred(tkt,mkt)
def thr_count(PH,t): return np.clip((PH>t).sum(1),1,5)
# tune absolute threshold on val (overall acc)
best_t,best=0.02,-1
for t in np.linspace(0.005,0.15,30):
    a=(thr_count(PHv,t)==nv).mean()
    if a>best: best,best_t=a,t
pr_t=(PHt.sum(1,keepdims=True)**2)/np.maximum((PHt**2).sum(1,keepdims=True),1e-9)  # participation ratio
def acc(pred,tr,k): s=tr==k; return (pred[s]==tr[s]).mean() if s.any() else float("nan")
print(f"=== NOC count accuracy ({RUN.name}; phi-threshold tuned on val tau={best_t:.3f}) ===")
print(f"  {'NOC':>4} {'model_card':>11} {'phi_thresh':>11} {'phi_partic':>11}")
ph_t=thr_count(PHt,best_t); pr_round=np.clip(np.round(pr_t[:,0]),1,5).astype(int)
for k in [1,2,3,4,5]:
    print(f"  {k:>4} {acc(Ct,nt,k):>11.3f} {acc(ph_t,nt,k):>11.3f} {acc(pr_round,nt,k):>11.3f}")
print(f"  {'ALL':>4} {(Ct==nt).mean():>11.3f} {(ph_t==nt).mean():>11.3f} {(pr_round==nt).mean():>11.3f}")
print("\n  (model_card = the trained cardinality head; phi_* = derived from independent soft-split phi)")
