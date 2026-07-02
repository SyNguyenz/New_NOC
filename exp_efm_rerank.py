"""
Validation gate: does the PROPER EuroForMix-class deconvolution (copy-number + stutter + degradation)
beat the crude uniform-compat phi (oracle N5 0.906) and the raw logit (0.831)? Tuned on val, eval test.
"""
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
import phi_rerank as pr, efm_rerank as ef
RUN="results/inc22_fixed_aslot_seed42"
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt,yt,nt=LD("tokens8_test"),LD("mask_test").astype(bool),LD("y_test_set"),LD("noc_test").clip(1,5)
Xv,Mv,yv,nv=LD("tokens8_val"),LD("mask_val").astype(bool),LD("y_val_set"),LD("noc_val").clip(1,5)
St,Sv=LD("size_test"),LD("size_val")
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
L_te=infer(Xt,Mt); L_va=infer(Xv,Mv)

C=45; yti=(yt>0.5).astype(int)
def oracle(score):
    e=np.zeros(len(score),bool)
    for i in range(len(score)):
        k=int(yti[i].sum()); p=np.zeros(C,int); p[np.argsort(score[i])[::-1][:k]]=1; e[i]=(p==yti[i]).all()
    return {j:round(float(e[nt==j].mean()),4) for j in (3,4,5)}, round(float(e.mean()),4)
def rep(name,score):
    pc,ov=oracle(score); print(f"  {name:34s} overall {ov:.4f} | N3 {pc[3]:.4f} N4 {pc[4]:.4f} N5 {pc[5]:.4f}")

rep("logit only (baseline)", L_te)
# crude phi
PHc_te=pr.deconv_phi(Xt,Mt,g,gmask,12); PHc_va=pr.deconv_phi(Xv,Mv,g,gmask,12)
ac=pr.tune_alpha(L_va,PHc_va,yv,nv); rep(f"crude-phi rerank (a={ac})", pr.rerank_scores(L_te,PHc_te,ac))
# EFM phi, a few xi / degradation settings (tune alpha on val each)
for xi,lam,tag in [(0.08,0.0,"efm xi=.08"),(0.06,0.0,"efm xi=.06"),(0.10,0.0,"efm xi=.10"),(0.08,0.003,"efm xi=.08 degr")]:
    PHe_te=ef.deconv_efm(Xt,Mt,St,g,gmask,12,xi,lam); PHe_va=ef.deconv_efm(Xv,Mv,Sv,g,gmask,12,xi,lam)
    ae=pr.tune_alpha(L_va,PHe_va,yv,nv)
    rep(f"{tag} rerank (a={ae})", pr.rerank_scores(L_te,PHe_te,ae))
    if xi==0.08 and lam==0.0:
        # standalone EFM-phi ranking (no logit) — pure deconvolution quality
        rep("efm-phi STANDALONE (no logit)", np.log(PHe_te+1e-9))
        rep("crude-phi STANDALONE (no logit)", np.log(PHc_te+1e-9))
rep("ORACLE ceiling", yti.astype(float))
