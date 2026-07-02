"""Prove the mechanism: is the BETTER (efm) deconvolution LESS diverse from the logit than the crude
one? If corr(logit, efm-phi) > corr(logit, crude-phi), accuracy bought redundancy -> smaller LOP gain
(Krogh-Vedelsby: gain ∝ diversity, not member accuracy)."""
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
import phi_rerank as pr, efm_rerank as ef
RUN="results/inc22_fixed_aslot_seed42"
def LD(f): return np.load("data_insilico_w/%s.npy"%f)
Xt,Mt=LD("tokens8_test"),LD("mask_test").astype(bool); St=LD("size_test")
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
PHc=pr.deconv_phi(Xt,Mt,g,gmask,12); PHe=ef.deconv_efm(Xt,Mt,St,g,gmask,12,0.08,0.0)
def zr(A): return np.stack([pr._z(A[i]) for i in range(len(A))]).reshape(-1)
zl=zr(L); zc=zr(np.log(PHc+1e-6)); ze=zr(np.log(PHe+1e-6))
print(f"corr(logit, crude-phi) = {np.corrcoef(zl,zc)[0,1]:.3f}  (crude rerank N5 0.906)")
print(f"corr(logit, efm-phi)   = {np.corrcoef(zl,ze)[0,1]:.3f}  (efm rerank N5 0.895)")
print(f"corr(crude-phi, efm-phi) = {np.corrcoef(zc,ze)[0,1]:.3f}")
print("=> if efm corr higher: better deconv = less diverse from logit = smaller LOP gain (Krogh-Vedelsby).")
