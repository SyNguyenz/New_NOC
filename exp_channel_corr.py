"""Measure pairwise correlation of the three ranking channels (per-row z-scored, pooled over
(sample,donor)). Tests the diversity claim: phi diverse from logit => helps; noise redundant with
phi => can't help (bias-variance floor at rho*sigma^2; Krogh-Vedelsby ambiguity needs diversity)."""
import json, numpy as np, torch
from models.set_transformer import SetTransformerMixture
import phi_rerank as pr, lop_rerank as lr
RUN="results/inc22_fixed_aslot_seed42"
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt,yt=LD("tokens8_test"),LD("mask_test").astype(bool),LD("y_test_set")
Xv,Mv,yv=LD("tokens8_val"),LD("mask_val").astype(bool),LD("y_val_set")
g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")
owner_lut=torch.zeros(24,1024,45); gm=torch.from_numpy(gmask).bool()
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+30
            if 0<=li<24 and 0<=ab<1024: owner_lut[li,ab,c]=1.0
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,
    n_noc=6,dropout=0.1,cls_decoder="aslot",n_token_feats=8,encoder="isab++",num_embed="periodic",
    periodic_sigma=0.3,aux_heads=True,sparse_attn=True,donor_geno=torch.from_numpy(g),
    donor_geno_mask=torch.from_numpy(gmask),nc_attn="mab0",soft_geno_attr=True,feas_filter=True,
    set_of_set=True,owner_lut=owner_lut,n_slot_iters=3,ot_eps=0.05,ot_iters=5,noc_head_v2=True)
m.load_state_dict(torch.load(RUN+"/best_model.pt",weights_only=True,map_location="cpu"),strict=False); m.eval()
@torch.no_grad()
def infer(X,M):
    L=[]
    for s in range(0,len(X),256): L.append(m(torch.tensor(X[s:s+256]),torch.tensor(M[s:s+256]))["logits_cls"].numpy())
    return np.concatenate(L)
L=infer(Xt,Mt)
PH=pr.deconv_phi(Xt,Mt,g,gmask,12); LP=np.log(PH+1e-6)
nb=lr.fit_noise_model(Xv,Mv,g,gmask,yv); S=lr.donor_support(Xt,Mt,g,gmask,nb)
def zr(A): return np.stack([pr._z(A[i]) for i in range(len(A))]).reshape(-1)
zl,zp,zs=zr(L),zr(LP),zr(S)
print("pooled per-row-z correlations (test):")
print(f"  corr(logit, phi)   = {np.corrcoef(zl,zp)[0,1]:.3f}   (diverse => phi adds; we SAW +7pp)")
print(f"  corr(logit, noise) = {np.corrcoef(zl,zs)[0,1]:.3f}")
print(f"  corr(phi,   noise) = {np.corrcoef(zp,zs)[0,1]:.3f}   (high => noise redundant w/ phi; we SAW no gain)")
