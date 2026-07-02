"""Decisive: does the in-model torch _em_deconv_phi equal the PROVEN numpy phi_rerank.deconv_phi?
If yes -> EM is correct, the regression is the integration design (joint-trained head).
If no  -> concrete bug in the torch EM."""
import numpy as np, torch
from models.set_transformer import SetTransformerMixture
import phi_rerank as pr

g=np.load("data/donor_geno.npy").astype(np.float32); gmask=np.load("data/donor_geno_mask.npy")
Xt=np.load("data_insilico_w/tokens8_test.npy")[:500]; Mt=np.load("data_insilico_w/mask_test.npy")[:500].astype(bool)
noc=np.load("data_insilico_w/noc_test.npy")[:500]

ALLELE_OFF,n_cls,LUT_W=30,45,1024
owner_lut=torch.zeros(24,LUT_W,n_cls); gm=torch.from_numpy(gmask).bool()
for c in range(min(n_cls,g.shape[0])):
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); ab=int(round(float(g[c,j,1])*10))+ALLELE_OFF
            if 0<=li<24 and 0<=ab<LUT_W: owner_lut[li,ab,c]=1.0
m=SetTransformerMixture(n_classes=45,cls_decoder="aslot",n_token_feats=8,aux_heads=True,
    feas_filter=True,set_of_set=True,soft_geno_attr=True,
    donor_geno=torch.from_numpy(g),donor_geno_mask=torch.from_numpy(gmask),owner_lut=owner_lut,
    em_phi_feature=True).eval()

with torch.no_grad():
    phi_torch=m._em_deconv_phi(torch.tensor(Xt),torch.tensor(Mt),n_iters=10).numpy()
phi_np=pr.deconv_phi(Xt,Mt,g,gmask,n_iters=10)

d=np.abs(phi_torch-phi_np)
# row-wise rank correlation (what matters for ranking) and top-k agreement
def topk_overlap(a,b,k):
    return np.mean([len(set(np.argsort(a[i])[::-1][:k])&set(np.argsort(b[i])[::-1][:k]))/k for i in range(len(a))])
cors=[np.corrcoef(phi_torch[i],phi_np[i])[0,1] for i in range(len(Xt)) if phi_np[i].std()>1e-9 and phi_torch[i].std()>1e-9]
print(f"max|Δφ| = {d.max():.4e}   mean|Δφ| = {d.mean():.4e}")
print(f"mean row Pearson(torch,np) = {np.nanmean(cors):.4f}  (n={len(cors)})")
for k in (3,4,5):
    print(f"  top-{k} donor-set overlap torch vs np: {topk_overlap(phi_torch,phi_np,k):.4f}")
# N5 specifically
n5=np.where(noc==5)[0]
if len(n5):
    print(f"N5 (n={len(n5)}): mean|Δφ|={d[n5].mean():.4e}  top5 overlap={topk_overlap(phi_torch[n5],phi_np[n5],5):.4f}")
