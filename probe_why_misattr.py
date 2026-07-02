"""
WHY does the model mis-attribute the faint minor's peaks?
Hypothesis: attribution is HARD (1 owner/peak) but reality is ADDITIVE (a peak's height can be SHARED
by several donors). A faint minor's alleles are MOSTLY shared with majors; the hard softmax gives the
shared peak to the better-corroborated donor -> the minor is credited ONLY by its few PRIVATE peaks.

For each N5 missed-true T: split T's present alleles into PRIVATE (only T among true donors) vs SHARED;
see where attr_pred sends each (to T / to another true donor / to decoy or background).
"""
import numpy as np
D="results/inc13_B_distill_seed42"; DA="data_insilico_w"; G="data/donor_geno.npy"
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
dset=[set(kk(g[c,j,0],g[c,j,1]) for j in range(g.shape[1]) if gm[c,j]) for c in range(C)]
tk=np.load(f"{DA}/tokens8_test.npy"); mk=np.load(f"{DA}/mask_test.npy").astype(bool)
attr_p=np.load(f"{D}/attr_pred_test.npy"); y_p=np.load(f"{D}/y_test_pred.npy").astype(bool)
y_t=np.load(f"{DA}/y_test_set.npy").astype(bool); noc=np.load(f"{DA}/noc_test.npy")

sel=np.where(noc==5)[0]
sh_frac=[]; priv_to_T=[]; priv_n=0; sh_n=0
sh_owner={"T":0,"other_true":0,"decoy_or_bg":0}; pr_owner={"T":0,"other_true":0,"decoy_or_bg":0}
for i in sel:
    # map (locus,abin) -> peak index (max height) for this sample
    pidx={}
    for k in np.where(mk[i])[0]:
        it=kk(tk[i,k,0],tk[i,k,1])
        if it not in pidx or tk[i,k,2]>tk[i,pidx[it],2]: pidx[it]=k
    Oset=set(pidx); true=np.where(y_t[i])[0]; pred=np.where(y_p[i])[0]
    for T in [c for c in true if not y_p[i,c]]:                 # missed faint true donor
        Tpres=dset[T]&Oset
        if not Tpres: continue
        nsh=0
        for it in Tpres:
            sharers=[c for c in true if it in dset[c]]          # true donors carrying this allele
            owner=attr_p[i,pidx[it]]                            # who the model attributed the peak to
            if owner==T: tgt="T"
            elif owner in true: tgt="other_true"
            else: tgt="decoy_or_bg"
            if len(sharers)>=2:                                 # SHARED allele
                nsh+=1; sh_n+=1; sh_owner[tgt]+=1
            else:                                               # PRIVATE to T
                priv_n+=1; pr_owner[tgt]+=1
        sh_frac.append(nsh/len(Tpres))
print(f"=== N5 missed-true donors: where do their peaks go? ===")
print(f"  mean fraction of missed-T's present alleles that are SHARED (with another true donor) = {np.mean(sh_frac):.3f}")
print(f"  (private alleles = the only ones the faint minor can be safely credited for)\n")
def pct(d): tot=sum(d.values()); return {k:f'{v}/{tot}={v/max(1,tot):.2f}' for k,v in d.items()}
print(f"  SHARED   alleles (n={sh_n})  attributed to -> {pct(sh_owner)}")
print(f"  PRIVATE  alleles (n={priv_n}) attributed to -> {pct(pr_owner)}")
print(f"\n  => if SHARED mostly go to 'other_true' and PRIVATE go to 'T', the hard 1-owner softmax is")
print(f"     STEALING the minor's shared peaks; it survives only on its (few) private peaks.")
