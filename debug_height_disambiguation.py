"""DECISIVE no-train probe: in the N5 MISSES (decoy displaces a true donor), do PEAK HEIGHTS
distinguish the true donor from the decoy?

Forensic additive model (EuroForMix): H(allele) = T * sum_d Mx_d * copy_d(allele). For a candidate
5-donor set, fit non-negative Mx by NNLS to the observed heights -> residual = how well that hypothesis
explains the heights. Compare:
   residual(TRUE set)            vs   residual(true set with missed-donor -> DECOY)
If TRUE << DECOY-swap  -> heights FAVOR the truth: the model under-uses the height likelihood = a real,
                         forensic, untried lever (MODEL-limited).
If TRUE ~= DECOY-swap  -> heights do NOT disambiguate: the decoy is as height-consistent as the truth =
                         genuine deconvolution ambiguity (INFO-limited) -> 0.9 likely unreachable here.
"""
import json, numpy as np, torch
from pathlib import Path
from scipy.optimize import nnls
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
# copy-number per donor: (locus,allele)->copy (homozygous locus = copy 2)
copy=[{} for _ in range(45)]
for c in range(45):
    byloc={}
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); al=round(float(g[c,j,1]),1); byloc.setdefault(li,[]).append(al)
    for li,als in byloc.items():
        u=list(set(als))
        if len(u)==1: copy[c][(li,u[0])]=2.0
        else:
            for a in u: copy[c][(li,a)]=1.0
def gkeys(c): return set(copy[c].keys())

tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool); y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int)
B=len(tok)
cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
  periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV)); m.eval()
P=np.zeros((B,45))
with torch.no_grad():
    for s in range(0,B,256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        P[s:s+256]=torch.sigmoid(m(tk,mb)["logits_cls"]).cpu().numpy()

def obs_heights(i):
    h={}
    for j in np.where(mk[i])[0]:
        k=(int(tok[i,j,0]),round(float(tok[i,j,1]),1)); v=float(np.expm1(tok[i,j,2]))
        h[k]=max(h.get(k,0.0),v)
    return h

def residual(i, donors):
    """fit Mx by NNLS over the union of observed + donor-expected alleles; return relative residual."""
    h=obs_heights(i)
    alle=set(h.keys())
    for d in donors: alle |= gkeys(d)
    alle=sorted(alle)
    A=np.zeros((len(alle),len(donors))); b=np.array([h.get(a,0.0) for a in alle])
    for c,d in enumerate(donors):
        for r,a in enumerate(alle): A[r,c]=copy[d].get(a,0.0)
    if b.sum()<=0: return None
    x,_=nnls(A,b); pred=A@x
    return float(np.linalg.norm(pred-b)/(np.linalg.norm(b)+1e-9))

true_res=[]; dec_res=[]; favor=0; ntot=0; corr_res=[]
for i in np.where(noc==5)[0]:
    tru=list(np.where(y[i]>0.5)[0]); top=list(np.argsort(P[i])[::-1][:5])
    if set(top)==set(tru):
        r=residual(i,tru);
        if r is not None: corr_res.append(r)
        continue
    missed=[d for d in tru if d not in top]; decoys=[d for d in top if d not in tru]
    # hardest case: the lowest-scored missed true donor vs the highest-scored decoy
    md=min(missed,key=lambda d:P[i,d]); dc=max(decoys,key=lambda d:P[i,d])
    swap=[dc if d==md else d for d in tru]
    rt=residual(i,tru); rd=residual(i,swap)
    if rt is None or rd is None: continue
    true_res.append(rt); dec_res.append(rd); ntot+=1
    if rt<rd: favor+=1
print(f"N5 correct cases: mean height residual(true set) = {np.mean(corr_res):.3f}  (n={len(corr_res)})")
print(f"N5 MISSES (n={ntot}):")
print(f"  residual TRUE set            = {np.mean(true_res):.3f}")
print(f"  residual DECOY-swapped set   = {np.mean(dec_res):.3f}")
print(f"  margin (decoy - true)        = {np.mean(dec_res)-np.mean(true_res):+.3f}")
print(f"  misses where heights FAVOR truth (residual_true < residual_decoy) = {favor}/{ntot} = {100*favor/ntot:.0f}%")
print()
if np.mean(dec_res)-np.mean(true_res) > 0.02 and favor/ntot>0.6:
    print("=> HEIGHTS DISAMBIGUATE: model-limited; height-likelihood is an untried real lever.")
else:
    print("=> HEIGHTS DO NOT DISAMBIGUATE: decoy ~ as height-consistent as truth = INFO-limited.")
