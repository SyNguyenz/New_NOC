"""
Does lever A filter infeasible noise BEFORE the heads? No - it only masks the attribution OUTPUT.
Input-filtering is separate: drop peaks whose allele NO panel donor carries (deployable, closed-set ->
those peaks cannot come from any contributor => pure dropin/noise => safe to remove). Measure how much
noise there is, its height (is it the faint artifact band?), and whether faint TRUE alleles survive.
"""
import numpy as np
DA="data_insilico_w"; G="data/donor_geno.npy"
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
carriers=set()
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]: carriers.add(kk(g[c,j,0],g[c,j,1]))
dset=[set(kk(g[c,j,0],g[c,j,1]) for j in range(g.shape[1]) if gm[c,j]) for c in range(C)]
tk=np.load(f"{DA}/tokens8_test.npy"); mk=np.load(f"{DA}/mask_test.npy").astype(bool)
y=np.load(f"{DA}/y_test_set.npy").astype(bool); noc=np.load(f"{DA}/noc_test.npy"); H=np.expm1(tk[:,:,2])

for NV in [5]:
    sel=np.where(noc==NV)[0]
    feas_h=[]; inf_h=[]; n_feas=0; n_inf=0; inf_per=[]
    true_allele_kept=[]; true_allele_lost=[]
    for i in sel:
        ni=0
        true=np.where(y[i])[0]; true_all=set().union(*[dset[c] for c in true])
        for k in np.where(mk[i])[0]:
            it=kk(tk[i,k,0],tk[i,k,1]); h=H[i,k]
            if it in carriers: feas_h.append(h); n_feas+=1
            else: inf_h.append(h); n_inf+=1; ni+=1
        inf_per.append(ni)
        # safety: do any TRUE contributor alleles get dropped by the no-panel-carrier filter?
        for it in true_all:
            (true_allele_kept if it in carriers else true_allele_lost).append(1)
    print(f"=== N{NV}: input feasibility filter (no panel donor carries the allele -> drop) ===")
    print(f"  feasible peaks   : {n_feas}  (mean/sample {n_feas/len(sel):.1f})  median height {np.median(feas_h):.0f}")
    print(f"  INFEASIBLE peaks : {n_inf}  (mean/sample {np.mean(inf_per):.1f})  median height {np.median(inf_h):.0f}  "
          f"= {n_inf/(n_feas+n_inf)*100:.0f}% of all peaks")
    print(f"  infeasible height pct: p50={np.percentile(inf_h,50):.0f} p90={np.percentile(inf_h,90):.0f}  "
          f"(feasible p50={np.percentile(feas_h,50):.0f} p90={np.percentile(feas_h,90):.0f})")
    print(f"  SAFETY — true-contributor alleles: kept(feasible)={len(true_allele_kept)}  dropped={len(true_allele_lost)} "
          f"({len(true_allele_lost)/max(1,len(true_allele_kept)+len(true_allele_lost))*100:.1f}% lost)")
    print(f"\n  => infeasible = pure dropin (no contributor can make it). Filtering cleans encoder context.")
    print(f"     If infeasible are mostly LOW height + ~0% true alleles lost -> safe pre-encoder filter.")
