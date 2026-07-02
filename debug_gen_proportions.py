"""
Decompose the minor-peak height gap (synth vs real) into its generator factors:
   minor_peak_height = phi_minor * t_total * rel_h * jitter
Compare, for N5 mixtures, synth-train vs real-test:
   - mixture proportions phi (min phi, max/min ratio)   <- generator r_max=6+5*(k-2)
   - total profile RFU per sample                       <- generator t_total~LN(log32000,.55)
   - per-donor total RFU (minor loudness)
"""
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)

def summ(split):
    tk=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool)
    at=L(f"attr_{split}").astype(int); phi=L(f"phi_{split}").astype(np.float32); noc=L(f"noc_{split}").astype(int)
    h=np.expm1(tk[:,:,2])
    n5=np.where(noc==5)[0]
    minphi=[]; ratio=[]; tot=[]; minor_donor_rfu=[]; nminor_present=[]
    for i in n5:
        pv=phi[i][phi[i]>0]
        if len(pv)<5:
            # phi may be missing/zero for real; skip ratio stats then
            pass
        if len(pv):
            minphi.append(pv.min()); ratio.append(pv.max()/max(pv.min(),1e-6))
        tot.append(h[i][mk[i]].sum())
        # per-donor total rfu
        for c in np.where(phi[i]>0)[0] if (phi[i]>0).any() else []:
            pe=(at[i]==c)&mk[i]
            if 0<phi[i,c]<0.2:
                minor_donor_rfu.append(h[i][pe].sum()); nminor_present.append(int(pe.sum()))
    def q(a): a=np.array(a,float); return f"p10={np.percentile(a,10):.3g} med={np.median(a):.3g} p90={np.percentile(a,90):.3g}"
    print(f"\n[{split}] N5 n={len(n5)}")
    if minphi: print(f"  min phi (faintest donor proportion): {q(minphi)}")
    if ratio:  print(f"  phi ratio max/min:                   {q(ratio)}")
    print(f"  total profile RFU per sample:        {q(tot)}")
    if minor_donor_rfu: print(f"  per-MINOR-donor total RFU:           {q(minor_donor_rfu)}")
    if nminor_present:  print(f"  per-MINOR-donor #peaks present:      {q(nminor_present)}")

print("="*70); print("N5 mixture-proportion & template comparison: SYNTH train vs REAL test"); print("="*70)
summ("train")
summ("test")
print("\nGenerator (make_insilico.gen_mixture): r_max=6+5*(k-2)=21 at N5;")
print("  phi=exp(U(0,log r_max)) normalized; t_total~LN(log 32000, 0.55); AT=14 RFU.")
print("If synth min-phi << real min-phi OR synth ratio >> real ratio -> r_max over-skews minors.")
print("If synth total RFU << real total RFU -> t_total too low.")
