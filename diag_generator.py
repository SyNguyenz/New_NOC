"""Diagnose the overlay generator (make_insilico) BEFORE fixing, to avoid bugs:
 (1) which proportion mode built the canonical data_insilico_w? -> min-phi / ratio per NOC vs real (orig min-phi~.05, realistic~.125)
 (2) does the OVERLAY SUBSTRATE (real single-source = noc==1 rows) already contain stutter? (if yes -> DON'T add an explicit stutter model = double-count; the under-stutter is from the overlay PROCESS losing minor stutter to AT-dropout)
 (3) where is the height range lost: real-SS substrate vs synth-mix vs real-mix."""
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)
tok=L("tokens8_train").astype(np.float32); mk=L("mask_train").astype(bool); noc=L("noc_train").astype(int)
phi=L("phi_train") if (DATA/"phi_train.npy").exists() else None

def stutter_height(idxs):
    allh=[]; stut=[]
    for i in idxs:
        loc={}
        for j in np.where(mk[i])[0]:
            l=int(tok[i,j,0]); a=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2]))
            loc.setdefault(l,{}); loc[l][a]=loc[l].get(a,0.0)+h
        for l,d in loc.items():
            for a,h in d.items():
                allh.append(h); par=d.get(round(a+1.0,1),0.0)
                stut.append(1 if (par>0 and 0.03< h/par <0.20) else 0)
    allh=np.array(allh)
    return np.mean(stut), np.median(allh), np.percentile(allh,10), np.percentile(allh,90)

print("=== within data_insilico_w TRAIN: substrate (NOC1=real SS) vs synth-mix (NOC2/5) ===")
for nc in [1,2,4,5]:
    sel=np.where(noc==nc)[0][:4000]
    if len(sel)==0: continue
    sr,med,p10,p90=stutter_height(sel)
    extra=""
    if phi is not None and nc>=2:
        mp=[]; rt=[]
        for i in sel:
            ps=phi[i][phi[i]>0]
            if len(ps)>0: mp.append(ps.min()); rt.append(ps.max()/max(ps.min(),1e-6))
        extra=f" | min-phi med={np.median(mp):.3f} ratio med={np.median(rt):.1f}"
    print(f"  NOC{nc} (n={len(sel)}): stutter={sr:.3f}  height med={med:.0f} p10={p10:.0f} p90={p90:.0f}{extra}")

print("\nreal test N5 reference (measured earlier): stutter 0.135, height med 132 p10 10 p90 1346, min-phi~0.125 ratio~4.0")
print("read: if NOC1(real SS) stutter HIGH but NOC5(synth) LOW -> overlay LOSES minor stutter (no explicit model needed);")
print("      if min-phi med ~0.05 -> canonical data = ORIGINAL skewed mode (the realistic/wide fix NOT applied).")
