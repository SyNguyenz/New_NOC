"""Is the N5 'decoy' INHERENT in the data, or ENCODER-INDUCED?
For each baseline N5 MISS: missed-true donor T vs false-included decoy D, measured on RAW evidence only
(observed mix peaks + known 45-donor panel genotypes; NO ground-truth peak filtering -> deployable, no F34 bug).
  T_private_present = T's panel alleles PRESENT in the mix that the OTHER 4 true donors CANNOT explain
                      (exclusive raw evidence FOR T that the model ignored).
  T_priv_height_pct = height percentile (0=faint..1=tallest) of those private peaks -> faint or clear?
  D_unique_support  = D's alleles present that the TRUE 5-set CANNOT explain (real competing evidence FOR D).
  D_damning_absence = D's panel alleles ABSENT at a locus that IS observed (raw evidence AGAINST D).
  rank_T / gap      = model's rank of the missed T, and P[bestDecoy]-P[T] (how close was it).
VERDICT:
  T_private_present>0 (clear height) & D_unique_support~0 & D_damning>=1  => ENCODER-INDUCED (info there, model failed)
  T_private_present~0                                                     => INHERENT ambiguity (data-limited)
"""
import json, numpy as np, torch
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"; DEV=torch.device("cuda")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)

g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]: geno[c].add((int(g[c,j,0]), round(float(g[c,j,1]),1)))

cfg=json.load(open(ROOT/"results/inc6_maskp_seed42/metrics.json"))["config"]
m=SetTransformerMixture(n_loci=24,d_locus=16,d_model=128,n_heads=4,n_isab=2,m_inducing=32,n_classes=45,n_noc=6,
  dropout=0.1,cls_decoder="per_donor",n_token_feats=8,encoder="isab++",num_embed="periodic",
  periodic_sigma=cfg["periodic_sigma"],aux_heads=True,sparse_attn=True).to(DEV)
m.load_state_dict(torch.load(ROOT/"results/inc6_maskp_seed42/best_model.pt",weights_only=True,map_location=DEV)); m.eval()
te_tok=L("tokens8_test").astype(np.float32); te_mk=L("mask_test").astype(bool); te_y=L("y_test_set").astype(np.float32); te_noc=L("noc_test").astype(int)
_n=L("tokens8_train")[:,:,1:8][L("mask_train").astype(bool)]
m.feat_mean.copy_(torch.tensor(_n.mean(0),device=DEV)); m.feat_std.copy_(torch.tensor(_n.std(0)+1e-6,device=DEV))
P=np.zeros((len(te_tok),45))
with torch.no_grad():
    for s in range(0,len(te_tok),128):
        x=torch.from_numpy(te_tok[s:s+128]).to(DEV); mb=torch.from_numpy(te_mk[s:s+128]).to(DEV)
        P[s:s+128]=torch.sigmoid(m(x,mb)["logits_cls"]).cpu().numpy()

def mix_of(i):
    loc={}
    for j in np.where(te_mk[i])[0]:
        l=int(te_tok[i,j,0]); al=round(float(te_tok[i,j,1]),1); h=float(np.expm1(te_tok[i,j,2]))
        loc.setdefault(l,{}); loc[l][al]=max(loc[l].get(al,0.),h)
    return loc
def present(c, loc): return {(l,al) for (l,al) in geno[c] if l in loc and al in loc[l]}

ii=np.where(te_noc==5)[0]
T=[]; D=[]; examples=[]
for i in ii:
    true=set(int(x) for x in np.where(te_y[i]>0.5)[0]); pred=set(int(x) for x in np.argsort(P[i])[::-1][:5])
    if pred==true: continue
    loc=mix_of(i); hs=sorted(h for ld in loc.values() for h in ld.values())
    pct=lambda h:(np.searchsorted(hs,h)/max(len(hs),1))
    rank=np.argsort(P[i])[::-1].tolist()
    missed=true-pred; decoys=pred-true
    expl_true=set().union(*[geno[c] for c in true])
    ex={"i":int(i),"T":[],"D":[]}
    for t in missed:
        expl_oth=set().union(*[geno[c] for c in (true-{t})])
        t_pres=present(t,loc); t_priv=t_pres-expl_oth
        ph=[pct(loc[l][al]) for (l,al) in t_priv]
        T.append((len(t_pres),len(t_priv), float(np.mean(ph)) if ph else -1.0, rank.index(t), float(P[i][max(decoys,key=lambda d:P[i][d])]-P[i][t])))
        ex["T"].append({"d":t,"present":len(t_pres),"private":len(t_priv),"priv_h":[round(x,2) for x in sorted(ph)],"rank":rank.index(t)})
    for d in decoys:
        d_pres=present(d,loc); d_uniq=d_pres-expl_true
        d_abs={(l,al) for (l,al) in geno[d] if l in loc and al not in loc[l]}
        D.append((len(d_pres),len(d_uniq),len(d_abs)))
        ex["D"].append({"d":d,"present":len(d_pres),"unique_support":len(d_uniq),"damning_absent":len(d_abs)})
    if len(examples)<4: examples.append(ex)

nmiss=len(set([])) ; nmiss=len(ii)-sum(1 for i in ii if set(int(x) for x in np.where(te_y[i]>0.5)[0])==set(int(x) for x in np.argsort(P[i])[::-1][:5]))
tp=np.array([r[1] for r in T]); tph=np.array([r[2] for r in T if r[2]>=0])
du=np.array([r[1] for r in D]); da=np.array([r[2] for r in D]); rk=np.array([r[3] for r in T]); gp=np.array([r[4] for r in T])
print(f"N5: {len(ii)} samples, {nmiss} MISSES | missed-true donors={len(T)} | decoys={len(D)}\n")
print("=== Missed TRUE donor (T) — does it have EXCLUSIVE raw evidence the model ignored? ===")
print(f"  T_private_present: mean={tp.mean():.2f} | ==0: {(tp==0).mean()*100:.0f}% | >=1: {(tp>=1).mean()*100:.0f}% | >=2: {(tp>=2).mean()*100:.0f}% | >=3: {(tp>=3).mean()*100:.0f}%")
print(f"  height-pctile of those private peaks: mean={tph.mean():.2f} (1=tallest) | faint(<0.33): {(tph<0.33).mean()*100:.0f}% | clear(>0.5): {(tph>0.5).mean()*100:.0f}%")
print(f"  model rank of missed T: median={int(np.median(rk))} | rank6(just-missed): {(rk==5).mean()*100:.0f}% | buried(>=10): {(rk>=10).mean()*100:.0f}%")
print(f"  prob gap P[decoy]-P[T]: mean={gp.mean():.3f}")
print("\n=== False-included DECOY (D) — does it ADD evidence, or is it a free-rider? ===")
print(f"  D_unique_support (alleles TRUE-set can't explain): mean={du.mean():.2f} | ==0: {(du==0).mean()*100:.0f}%")
print(f"  D_damning_absence (panel allele missing at an OBSERVED locus): mean={da.mean():.2f} | >=1: {(da>=1).mean()*100:.0f}% | >=2: {(da>=2).mean()*100:.0f}%")
print("\n=== concrete examples (read raw by hand) ===")
for ex in examples:
    print(f" sample {ex['i']}: missed T={ex['T']}")
    print(f"            decoys D={ex['D']}")
