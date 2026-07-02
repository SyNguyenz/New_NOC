"""Read the RAW miss cases: can a careful (forensic) reading distinguish the true donor from the decoy,
using signals the crude NNLS / decoder ignore? If yes -> model is suboptimal (NOT info-limited).

Key signal the global residual missed: DROPOUT-WEIGHTED ABSENCE. A donor's expected allele that is ABSENT
at a STRONG-template locus (tall peaks, low dropout prob) is damning evidence against that donor. The decoy
(67% compatible) has ~33% absent alleles — are they at strong loci (damning) while the true donor's absences
are only at weak loci (forgivable)? Also: panel-unique present alleles (definitive for the true donor).
"""
import json, numpy as np, torch
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
DEV=torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.set_transformer import SetTransformerMixture
def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]: geno[c].add((int(g[c,j,0]),round(float(g[c,j,1]),1)))
# panel rarity: how many of 45 donors carry each (locus,allele)
from collections import Counter
panel=Counter()
for c in range(45):
    for k in geno[c]: panel[k]+=1
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
def profile(i):
    """locus -> {allele: height}; and per-locus max height."""
    loc={}
    for j in np.where(mk[i])[0]:
        l=int(tok[i,j,0]); a=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2]))
        inner=loc.setdefault(l,{}); inner[a]=max(inner.get(a,0.0),h)
    return loc
# global median peak height (template scale)
allh=[float(np.expm1(tok[i,j,2])) for i in range(B) for j in np.where(mk[i])[0]]
MED=np.median(allh); print(f"global median peak height = {MED:.0f}\n")
def damning_absence(i,d,loc):
    """count d's expected alleles ABSENT at a STRONG-template locus (locus max height > median)."""
    dam=0; weak=0
    for (l,a) in geno[d]:
        present = (l in loc and a in loc[l])
        if present: continue
        strong = (l in loc and max(loc[l].values())>MED)
        if strong: dam+=1
        else: weak+=1
    return dam,weak
def panel_unique_present(i,d,loc):
    return sum(1 for (l,a) in geno[d] if panel[(l,a)]==1 and l in loc and a in loc[l])

# AGGREGATE over N5 misses: decoy vs true-missed-donor
dam_d=[]; dam_t=[]; puq_d=[]; puq_t=[]; examples=[]
for i in np.where(noc==5)[0]:
    tru=set(np.where(y[i]>0.5)[0].tolist()); top=set(np.argsort(P[i])[::-1][:5].tolist())
    if top==tru: continue
    loc=profile(i)
    missed=[d for d in tru if d not in top]; decoys=[d for d in top if d not in tru]
    md=min(missed,key=lambda d:P[i,d]); dc=max(decoys,key=lambda d:P[i,d])
    dd,_=damning_absence(i,dc,loc); dt,_=damning_absence(i,md,loc)
    dam_d.append(dd); dam_t.append(dt); puq_d.append(panel_unique_present(i,dc,loc)); puq_t.append(panel_unique_present(i,md,loc))
    if len(examples)<3: examples.append((i,md,dc,loc))
print("N5 MISSES aggregate (true-missed-donor vs the decoy that displaced it):")
print(f"  damning absences (expected allele MISSING at a tall-peak locus): true={np.mean(dam_t):.2f}  decoy={np.mean(dam_d):.2f}")
print(f"  panel-UNIQUE present alleles (only this donor has it, observed):  true={np.mean(puq_t):.2f}  decoy={np.mean(puq_d):.2f}")
print(f"  misses where decoy has MORE damning absences than true: {np.mean(np.array(dam_d)>np.array(dam_t)):.0%}")
print(f"  misses where true has a panel-unique allele the decoy lacks: {np.mean(np.array(puq_t)>np.array(puq_d)):.0%}\n")
# RAW dump of 3 cases
for (i,md,dc,loc) in examples:
    print(f"===== sample {i}: TRUE-missed donor {md} (score {P[i,md]:.2f}) vs DECOY {dc} (score {P[i,dc]:.2f}) =====")
    for l in sorted(set([x[0] for x in geno[md]]+[x[0] for x in geno[dc]]+list(loc.keys()))):
        obs=loc.get(l,{}); ta=sorted(a for (ll,a) in geno[md] if ll==l); da=sorted(a for (ll,a) in geno[dc] if ll==l)
        obs_s=' '.join(f'{a}:{int(h)}' for a,h in sorted(obs.items()))
        def mark(als):
            out=[]
            for a in als:
                sign='+' if (l in loc and a in loc[l]) else '-'
                out.append(f'{a}{sign}')
            return ' '.join(out)
        print(f"  L{l:2d} obs[{obs_s}] | true{md}:[{mark(ta)}] decoy{dc}:[{mark(da)}]")
    print()
