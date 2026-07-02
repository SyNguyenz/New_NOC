"""GATE for the decoy-environment idea: is there a LEARNABLE signal that separates a true donor from a
compatible DECOY, beyond what the decoder already uses?

Forensic LR feature (deployable in spirit): marginal height-fit improvement of ADDING donor d to the set
of OTHER present donors -> residual(others) - residual(others + d). A TRUE donor explains extra peaks
(positive marginal); a DECOY whose alleles are already covered adds ~nothing (~0 marginal).
Candidates per mixture (NOC>=3): present-true donors (label 1) vs compatible-absent decoys (comp>=0.6, label 0).
Compare AUC(true vs decoy) of:  decoder score  |  marginal-height-LR  |  both.
  marginal-LR AUC >> decoder AUC  -> there IS an under-used height signal -> the decoy-environment can teach it -> GO.
  marginal-LR ~ decoder           -> no extra signal -> decoys genuinely ambiguous -> INFO-limited, environment can't help.
(Uses oracle co-donors = ceiling of the available signal; we ask only whether the signal EXISTS.)
"""
import json, numpy as np, torch
from pathlib import Path
from scipy.optimize import nnls
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
copy=[{} for _ in range(45)]; gset=[set() for _ in range(45)]
for c in range(45):
    byloc={}
    for j in range(g.shape[1]):
        if gm[c,j]:
            li=int(g[c,j,0]); al=round(float(g[c,j,1]),1); byloc.setdefault(li,[]).append(al); gset[c].add((li,al))
    for li,als in byloc.items():
        u=list(set(als));
        for a in u: copy[c][(li,a)]=2.0 if len(u)==1 else 1.0

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
def heights(i):
    h={}
    for j in np.where(mk[i])[0]:
        k=(int(tok[i,j,0]),round(float(tok[i,j,1]),1)); h[k]=max(h.get(k,0.0),float(np.expm1(tok[i,j,2])))
    return h
def resid(h,donors):
    if not donors:
        b=np.array(list(h.values())); return 1.0
    alle=set(h.keys())
    for d in donors: alle|=set(copy[d].keys())
    alle=sorted(alle); A=np.zeros((len(alle),len(donors))); b=np.array([h.get(a,0.0) for a in alle])
    for c,d in enumerate(donors):
        for r,a in enumerate(alle): A[r,c]=copy[d].get(a,0.0)
    if b.sum()<=0: return None
    x,_=nnls(A,b); return float(np.linalg.norm(A@x-b)/(np.linalg.norm(b)+1e-9))
lab=[]; f_dec=[]; f_lr=[]
rng=np.random.RandomState(0)
for i in np.where(noc>=3)[0]:
    h=heights(i); tru=list(np.where(y[i]>0.5)[0]); ok=set(h.keys())
    base_full=resid(h,tru)
    for d in tru:                                   # TRUE donor: marginal of adding d to the others
        others=[o for o in tru if o!=d]; mlr=(resid(h,others) or 1.0)-(resid(h,others+[d]) or 1.0)
        lab.append(1); f_dec.append(P[i,d]); f_lr.append(mlr)
    decoys=[d for d in range(45) if d not in tru and len(gset[d]&ok)/max(len(gset[d]),1)>=0.6]
    rng.shuffle(decoys)
    for d in decoys[:len(tru)]:                     # compatible DECOY: marginal of adding it to the true set
        mlr=(base_full or 1.0)-(resid(h,tru+[d]) or 1.0)
        lab.append(0); f_dec.append(P[i,d]); f_lr.append(mlr)
lab=np.array(lab); f_dec=np.array(f_dec); f_lr=np.array(f_lr)
zd=(f_dec-f_dec.mean())/(f_dec.std()+1e-9); zl=(f_lr-f_lr.mean())/(f_lr.std()+1e-9)
print(f"true-vs-compatible-decoy separation (N3-5, n={len(lab)}, {int(lab.sum())} true / {int((1-lab).sum())} decoy):")
print(f"  decoder score              AUC = {roc_auc_score(lab,zd):.3f}")
print(f"  marginal height-LR         AUC = {roc_auc_score(lab,zl):.3f}")
print(f"  decoder + height-LR        AUC = {roc_auc_score(lab,zd+zl):.3f}")
print(f"\n  mean marginal-LR: true={f_lr[lab==1].mean():.3f}  decoy={f_lr[lab==0].mean():.3f}")
