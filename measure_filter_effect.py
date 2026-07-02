"""Does filtering the artifact faint band IMPROVE true-vs-decoy separability (model-free)?
On real test N5, for each height filter F (drop peaks < F RFU before analysis):
  TRUE evidence kept   = min over true donors of (private alleles present)  [the faint minor's distinguishing evidence]
  DECOY false support  = best non-contributor's unique-support (alleles present that the true set can't explain)
If raising F removes DECOY support FASTER than it removes TRUE private -> filtering widens the gap -> helps discrimination."""
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]: geno[c].add((int(g[c,j,0]),round(float(g[c,j,1]),1)))
tok=np.load(DATA/"tokens8_test.npy").astype(np.float32); mk=np.load(DATA/"mask_test.npy").astype(bool)
y=np.load(DATA/"y_test_set.npy"); noc=np.load(DATA/"noc_test.npy").astype(int)
ii=np.where(noc==5)[0]

def loc_of(i,F):
    loc={}
    for j in np.where(mk[i])[0]:
        h=float(np.expm1(tok[i,j,2]))
        if h<F: continue
        l=int(tok[i,j,0]); a=round(float(tok[i,j,1]),1); loc.setdefault(l,{}); loc[l][a]=max(loc[l].get(a,0.),h)
    return loc

print(" filter | true min-private | best-decoy unique-support | decoy>=true-private? (ambiguous frac)")
for F in [0,15,25,35,50]:
    tp=[]; du=[]; amb=[]
    for i in ii:
        true=set(int(x) for x in np.where(y[i]>0.5)[0]); loc=loc_of(i,F)
        union=set().union(*[geno[c] for c in true])
        mp=99
        for t in true:
            eo=set().union(*[geno[c] for c in (true-{t})])
            pres={(l,a) for (l,a) in geno[t] if l in loc and a in loc[l]}
            mp=min(mp,len(pres-eo))
        best=0
        for c in range(45):
            if c in true: continue
            dp={(l,a) for (l,a) in geno[c] if l in loc and a in loc[l]}; best=max(best,len(dp-union))
        tp.append(mp); du.append(best); amb.append(1 if best>=mp else 0)
    print(f"  <{F:>3} | {np.mean(tp):6.2f}          | {np.mean(du):6.2f}                  | {100*np.mean(amb):5.1f}%")
print("\nread: want true min-private to STAY HIGH while best-decoy unique-support DROPS, and ambiguous-frac DROP.")
