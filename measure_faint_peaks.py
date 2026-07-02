"""Are the FAINT peaks in real mixtures TRUE donor alleles, or artifacts (stutter/drop-in/noise)?
For real test N5: each observed peak -> is (locus,allele) explained by ANY true contributor's genotype?
  in-union  = a true donor HAS that allele (could be a real faint allele, OR coincidental stutter on a true pos)
  NOT-union = NO true donor has it -> PURE ARTIFACT (stutter/drop-in/noise) -> a filter candidate
Stratify by height. Also: of the artifacts, how many are back-stutter (at a-1 of a larger peak)?
If faint peaks are mostly ARTIFACT -> filtering helps (cleaner, less decoy fodder). If mostly TRUE -> filtering removes the minor's evidence -> hurts."""
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

bins=[(0,20),(20,40),(40,75),(75,150),(150,1e9)]
cnt={b:[0,0] for b in bins}     # [in-union, not-union]
art_stut=0; art_tot=0
for i in np.where(noc==5)[0]:
    true=set(int(x) for x in np.where(y[i]>0.5)[0])
    union=set().union(*[geno[c] for c in true])
    loc={}
    for j in np.where(mk[i])[0]:
        l=int(tok[i,j,0]); a=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2]))
        loc.setdefault(l,{}); loc[l][a]=max(loc[l].get(a,0.),h)
    for l,d in loc.items():
        for a,h in d.items():
            inu = (l,a) in union
            for b in bins:
                if b[0]<=h<b[1]: cnt[b][0 if inu else 1]+=1; break
            if not inu:                                 # artifact: is it back-stutter? (parent at a+1, taller, ratio 3-20%)
                art_tot+=1; par=d.get(round(a+1.0,1),0.0)
                if par>0 and 0.03< h/par <0.20: art_stut+=1
print("height-band   | #peaks | %TRUE-donor-allele | %ARTIFACT(no true donor has it)")
for b in bins:
    t,n=cnt[b]; tot=t+n
    if tot: print(f"  {b[0]:>4}-{('inf' if b[1]>1e8 else int(b[1])):<5}RFU | {tot:5d}  |  {100*t/tot:5.1f}%  |  {100*n/tot:5.1f}%")
print(f"\nof all ARTIFACT peaks (n={art_tot}): {100*art_stut/max(art_tot,1):.0f}% are back-stutter-like (rest = drop-in/noise/forward-stutter)")
print("read: if faint bands are mostly ARTIFACT -> a stutter/AT filter cleans decoy fodder; if mostly TRUE -> filtering kills the faint minor.")
