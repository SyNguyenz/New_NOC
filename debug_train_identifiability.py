"""
Does the synthetic generator's 'go beyond-real hard' tail create UNLEARNABLE labels?
For synthetic TRAIN N5 minors, count surviving PRIVATE alleles (present & not in any
other true donor). If a large fraction have 0-2 private present alleles, those donor
labels are effectively unidentifiable -> training NOISE, not useful hard examples.
Compare the regime to real test. Also report coverage of the real (balanced) regime.
"""
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
g=np.load(ROOT/"data"/"donor_geno.npy"); gm=np.load(ROOT/"data"/"donor_geno_mask.npy").astype(bool)
def key(lo,al): return lo.astype(int)*1000+np.round(al*10).astype(int)
ref=[set(key(g[c,gm[c],0],g[c,gm[c],1]).tolist()) for c in range(45)]

def L(n): return np.load(DATA/f"{n}.npy",allow_pickle=True)
def priv_dist(split, cap=6000):
    tk=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool)
    at=L(f"attr_{split}").astype(int); phi=L(f"phi_{split}").astype(np.float32); noc=L(f"noc_{split}").astype(int)
    n5=np.where(noc==5)[0]; rng=np.random.RandomState(0); rng.shuffle(n5); n5=n5[:cap]
    npriv=[]
    for i in n5:
        obs=set(key(tk[i,mk[i],0],tk[i,mk[i],1]).tolist())
        true=np.where(phi[i]>0)[0]
        for c in true:
            if not (0<phi[i,c]<0.2): continue
            others=set().union(*[ref[o] for o in true if o!=c]) if len(true)>1 else set()
            npriv.append(len((ref[c]&obs)-others))
    npriv=np.array(npriv)
    print(f"[{split}] N5 minor-donors n={len(npriv)} | "
          f"0 private={np.mean(npriv==0)*100:.0f}%  1-2={np.mean((npriv>=1)&(npriv<=2))*100:.0f}%  "
          f">=3={np.mean(npriv>=3)*100:.0f}%  median={np.median(npriv):.0f}")
    return npriv

print("Surviving PRIVATE alleles per N5 minor donor (identifiability of the label):")
priv_dist("train"); priv_dist("test")

# coverage: where does real sit in the synthetic min-phi / ratio distribution?
def stats(split):
    phi=L(f"phi_{split}").astype(np.float32); noc=L(f"noc_{split}").astype(int)
    mp=[]; ra=[]
    for i in np.where(noc==5)[0]:
        pv=phi[i][phi[i]>0]
        if len(pv)==5: mp.append(pv.min()); ra.append(pv.max()/pv.min())
    return np.array(mp), np.array(ra)
mp_s,ra_s=stats("train"); mp_r,ra_r=stats("test")
print("\nCoverage of the REAL regime by SYNTHETIC (min-phi):")
print(f"  real median min-phi = {np.median(mp_r):.3f}; synthetic percentile of that value = "
      f"{(mp_s<np.median(mp_r)).mean()*100:.0f}th  (high => real's typical case is synthetic's rare tail)")
print(f"  real balanced mixtures (ratio<=2): {np.mean(ra_r<=2)*100:.0f}% of real | "
      f"synthetic ratio<=2: {np.mean(ra_s<=2)*100:.0f}% of synth  (easy extreme coverage)")
