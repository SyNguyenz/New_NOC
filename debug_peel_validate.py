"""No-train validation of the GENERATIVE / iterative-subtraction idea (deployable):
greedily pick the donor that best explains the current RESIDUAL (rarity-weighted) minus a damning-absence
penalty, SUBTRACT its estimated contribution, repeat. If 'subtract-and-see' disambiguates decoys, the
peeled set should beat the decoder's independent top-k on N5. Also a HYBRID (decoder picks the confident
ones, residual disambiguates the rest). All deployable: genotype + observed peaks + decoder output, no labels.
"""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]; copy=[{} for _ in range(45)]
from collections import Counter; panel=Counter()
for c in range(45):
    byloc={}
    for j in range(g.shape[1]):
        if gm[c,j]:
            l=int(g[c,j,0]); a=round(float(g[c,j,1]),1); geno[c].add((l,a)); byloc.setdefault(l,[]).append(a)
    for l,als in byloc.items():
        u=list(set(als))
        for a in u: copy[c][(l,a)]=2.0 if len(u)==1 else 1.0
for c in range(45):
    for k in geno[c]: panel[k]+=1
tok=L("tokens8_test").astype(np.float32); mk=L("mask_test").astype(bool); y=L("y_test_set").astype(np.float32); noc=L("noc_test").astype(int); B=len(tok)
cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
  periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV)); m.eval()
P=np.zeros((B,45)); Lg=np.zeros((B,45))
with torch.no_grad():
    for s in range(0,B,256):
        tk=torch.from_numpy(tok[s:s+256]).to(DEV); mb=torch.from_numpy(mk[s:s+256]).to(DEV)
        lg=m(tk,mb)["logits_cls"]; Lg[s:s+256]=lg.cpu().numpy(); P[s:s+256]=torch.sigmoid(lg).cpu().numpy()
MED=np.median([float(np.expm1(tok[i,j,2])) for i in range(B) for j in np.where(mk[i])[0]])
def prof(i):
    loc={}
    for j in np.where(mk[i])[0]:
        l=int(tok[i,j,0]); a=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2])); d=loc.setdefault(l,{}); d[a]=max(d.get(a,0.),h)
    return loc
def damning(d,loc):
    return sum(1 for (l,a) in geno[d] if not(l in loc and a in loc[l]) and (l in loc and max(loc[l].values())>MED))
def res_score(d,r,loc):
    expl=sum(r.get((l,a),0.0)/panel[(l,a)] for (l,a) in geno[d])
    return expl - 3.0*damning(d,loc)
def peel(i,fixed=(),k=5,alpha=0.0):
    loc=prof(i); r={(l,a):loc[l][a] for l in loc for a in loc[l]}
    sel=list(fixed)
    for d in sel:                                   # subtract the pre-fixed donors first
        present=[r[(l,a)] for (l,a) in geno[d] if r.get((l,a),0)>0]
        mx=np.median(present) if present else 0.0
        for (l,a) in geno[d]:
            if (l,a) in r: r[(l,a)]=max(0.0,r[(l,a)]-mx*copy[d][(l,a)])
    while len(sel)<k:
        cand=[d for d in range(45) if d not in sel]
        sc=[alpha*Lg[i,d]+res_score(d,r,loc) for d in cand]
        d=cand[int(np.argmax(sc))]; sel.append(d)
        present=[r[(l,a)] for (l,a) in geno[d] if r.get((l,a),0)>0]
        mx=np.median(present) if present else 0.0
        for (l,a) in geno[d]:
            if (l,a) in r: r[(l,a)]=max(0.0,r[(l,a)]-mx*copy[d][(l,a)])
    return set(sel)
def n5_set(getset):
    e=[]
    for i in np.where(noc==5)[0]:
        pred=getset(i); e.append(pred==set(np.where(y[i]>0.5)[0].tolist()))
    return round(float(np.mean(e)),3)
def dec_top5(i): return set(np.argsort(P[i])[::-1][:5].tolist())
def hybrid(i,nfix): return peel(i,fixed=tuple(np.argsort(P[i])[::-1][:nfix].tolist()),k=5)
print(f"decoder top-5            N5 = {n5_set(dec_top5)}")
print(f"pure greedy peel (resid) N5 = {n5_set(lambda i: peel(i,k=5))}")
print(f"greedy peel + 0.3*decoder N5 = {n5_set(lambda i: peel(i,k=5,alpha=0.3))}")
for nf in [2,3,4]:
    print(f"hybrid: decoder top-{nf} fixed + residual peel rest  N5 = {n5_set(lambda i,nf=nf: hybrid(i,nf))}")
