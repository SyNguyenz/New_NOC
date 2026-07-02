"""
exp_lop_rerank.py — Step 1 (optimized rank) + Step 2 (noise channel), post-hoc on inc22_fixed.
Compares oracle EM (top-true-k) on TEST for ranking scores fit/tuned on VAL (C6-clean):
  logit-only -> +phi(hand-alpha, widened) -> +phi(LR weights) -> +phi+noise(LR weights).
The +phi+noise vs +phi delta is the MARGINAL-OVER-PHI of the noise/decoy mechanism.
"""
import json, numpy as np, torch
from models.set_transformer import SetTransformerMixture
import phi_rerank as pr, lop_rerank as lr

DEVICE="cpu"; RUN="results/inc22_fixed_aslot_seed42"
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt,yt,nt=LD("tokens8_test"),LD("mask_test").astype(bool),LD("y_test_set"),LD("noc_test").clip(1,5)
Xv,Mv,yv,nv=LD("tokens8_val"),LD("mask_val").astype(bool),LD("y_val_set"),LD("noc_val").clip(1,5)
g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")
ALLELE_OFF,n_cls,LUT_W=30,45,1024
owner_lut=torch.zeros(24,LUT_W,n_cls); gm=torch.from_numpy(gmask).bool()
for c in range(min(n_cls,g.shape[0])):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+ALLELE_OFF
            if 0<=li<24 and 0<=ab<LUT_W: owner_lut[li,ab,c]=1.0
model=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,
    n_classes=45,n_noc=6,dropout=0.1,cls_decoder="aslot",n_token_feats=8,encoder="isab++",
    num_embed="periodic",periodic_sigma=0.3,aux_heads=True,sparse_attn=True,
    donor_geno=torch.from_numpy(g),donor_geno_mask=torch.from_numpy(gmask),nc_attn="mab0",
    soft_geno_attr=True,feas_filter=True,set_of_set=True,owner_lut=owner_lut,
    n_slot_iters=3,ot_eps=0.05,ot_iters=5,noc_head_v2=True).to(DEVICE)
model.load_state_dict(torch.load(RUN+"/best_model.pt",weights_only=True,map_location=DEVICE),strict=False); model.eval()
@torch.no_grad()
def infer(X,M):
    L=[]
    for s in range(0,len(X),256): L.append(model(torch.tensor(X[s:s+256]),torch.tensor(M[s:s+256]))["logits_cls"].numpy())
    return np.concatenate(L)
L_te=infer(Xt,Mt); L_va=infer(Xv,Mv)

# channels: logit, log-phi, noise-support
PH_te=pr.deconv_phi(Xt,Mt,g,gmask,12); PH_va=pr.deconv_phi(Xv,Mv,g,gmask,12)
nb=lr.fit_noise_model(Xv,Mv,g,gmask,yv)
print(f"noise model Pr(noise)=sigmoid({nb[0]:.3f} + {nb[1]:.3f}*log1p(h))  [b1<0 => taller=cleaner, expected]")
S_te=lr.donor_support(Xt,Mt,g,gmask,nb); S_va=lr.donor_support(Xv,Mv,g,gmask,nb)

LP_te=np.log(PH_te+1e-6); LP_va=np.log(PH_va+1e-6)
ch2_va=[L_va,LP_va]; ch2_te=[L_te,LP_te]
ch3_va=[L_va,LP_va,S_va]; ch3_te=[L_te,LP_te,S_te]

C=45; yti=(yt>0.5).astype(int)
def oracle(score):
    e=np.zeros(len(score),bool)
    for i in range(len(score)):
        k=int(yti[i].sum()); pr_=np.zeros(C,int); pr_[np.argsort(score[i])[::-1][:k]]=1
        e[i]=(pr_==yti[i]).all()
    return {j:round(float(e[nt==j].mean()),4) for j in range(1,6)}, round(float(e.mean()),4)

def rep(name,score):
    pc,ov=oracle(score); print(f"  {name:34s} overall {ov:.4f} | N3 {pc[3]:.4f} N4 {pc[4]:.4f} N5 {pc[5]:.4f}")

# widened-alpha hand tune (oracle EM on val) for phi-only
def tune_alpha_wide(Lva,PHva,grid):
    best,bv=0.0,-1
    for a in grid:
        sc=np.stack([pr._z(Lva[i])+a*pr._z(np.log(PHva[i]+1e-6)) for i in range(len(Lva))])
        e=[]
        for k in (5,4,3):
            sel=np.where(nv==k)[0]; h=0
            for i in sel:
                p=np.zeros(C,int); p[np.argsort(sc[i])[::-1][:k]]=1; h+=int((p==(yv[i]>0.5)).all())
            e.append(h/max(1,len(sel)))
        if np.mean(e)>bv: bv,best=np.mean(e),a
    return best
aw=tune_alpha_wide(L_va,PH_va,np.linspace(0,3,16))
print(f"widened alpha* = {aw:.2f} (old grid capped at 1.0)\n")

w2=lr.fit_pool_weights(ch2_va,yv); w3=lr.fit_pool_weights(ch3_va,yv)
print(f"LR pool weights  2ch [logit, logphi] = {np.round(w2,3)}")
print(f"LR pool weights  3ch [logit, logphi, noise] = {np.round(w3,3)}\n")

rep("logit only (baseline)", L_te)
rep("+phi hand-alpha=1.0 (old)", pr.rerank_scores(L_te,PH_te,1.0))
rep(f"+phi hand-alpha={aw:.2f} (widened)", pr.rerank_scores(L_te,PH_te,aw))
rep("+phi LR-weights (step1)", lr.pool_scores(ch2_te,w2))
rep("+phi+noise LR-weights (step2)", lr.pool_scores(ch3_te,w3))
rep("ORACLE ceiling (=true set)", yti.astype(float)*1e6+L_te*1e-6)
