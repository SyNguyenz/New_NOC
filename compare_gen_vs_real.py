"""How does the in-silico GENERATOR (synthetic train) differ from REALITY (real test)? Measure the EPG
phenomena a forensic generator must capture, comparing synthetic train vs real test at matched NOC:
  peaks/sample, log-height distribution, alleles-per-locus occupancy, BACK-STUTTER signature
  (peak at allele a with a taller parent at a+1, ratio 3-20%), near-threshold (dropout-prone) fraction,
  heterozygote/intra-locus height imbalance, degradation slope (height vs locus-size proxy)."""
import numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
def L(n): return np.load(DATA/f"{n}.npy", allow_pickle=True)

def feats(split):
    tok=L(f"tokens8_{split}").astype(np.float32); mk=L(f"mask_{split}").astype(bool); noc=L(f"noc_{split}").astype(int)
    return tok,mk,noc

def describe(tok,mk,noc,split,target_noc=5):
    sel=np.where(noc==target_noc)[0]
    npk=[]; allh=[]; occ=[]; stut=[]; lowf=[]; imbal=[]; degr=[]
    for i in sel:
        js=np.where(mk[i])[0]
        loc={}
        for j in js:
            l=int(tok[i,j,0]); al=round(float(tok[i,j,1]),1); h=float(np.expm1(tok[i,j,2]))
            loc.setdefault(l,{}); loc[l][al]=loc[l].get(al,0.0)+h
        npk.append(len(js))
        for l,d in loc.items():
            occ.append(len(d)); hs=sorted(d.values())
            for a,h in d.items():
                allh.append(h)
                if h<60: lowf.append(1)
                else: lowf.append(0)
                # back-stutter: is there a taller peak at a+1 (one repeat up)?
                par=d.get(round(a+1.0,1),0.0)
                if par>0 and h>0 and 0.03< h/par <0.20: stut.append(1)
                else: stut.append(0)
            if len(hs)>=2: imbal.append(hs[0]/hs[-1])      # smallest/largest within locus
        # degradation: corr of height vs locus index (size proxy)
        if len(loc)>=4:
            ls=np.array(sorted(loc.keys())); mh=np.array([max(loc[l].values()) for l in ls])
            if mh.std()>0 and ls.std()>0: degr.append(np.corrcoef(ls,np.log1p(mh))[0,1])
    def s(x): return f"{np.mean(x):.3f}" if len(x) else "na"
    print(f"[{split} NOC{target_noc}] n={len(sel)}")
    print(f"  peaks/sample: mean={np.mean(npk):.1f}  | alleles/locus: mean={np.mean(occ):.2f}")
    print(f"  height RFU: median={np.median(allh):.0f}  p10={np.percentile(allh,10):.0f}  p90={np.percentile(allh,90):.0f}")
    print(f"  near-threshold (<60 RFU) frac: {np.mean(lowf):.3f}  | intra-locus imbalance (min/max): {s(imbal)}")
    print(f"  back-stutter-like frac (of peaks): {np.mean(stut):.3f}  | degradation slope (corr size,logH): {s(degr)}")

for nc in (4,5):
    for sp in ("train","test"):
        tok,mk,noc=feats(sp); describe(tok,mk,noc,sp,nc)
    print()
