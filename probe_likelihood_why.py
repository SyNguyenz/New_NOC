"""
WHAT does the likelihood/EuroForMix-spirit deconvolution actually DO to the faint N5 minor, and why does
that make it fail at ID? Direct mechanism readout (no neural needed).

Model (probe9-identical): G (peaks x 45) = candidate-donor reference-genotype PRESENCE; h = observed peak
RFU; w = nnls(G, h) = per-donor contribution that reconstructs the height vector. Rank donors by w.

For each N5 sample, report what NNLS assigns the faint (lowest-height) true minor:
  w_faint vs mean w_major            -> is the minor credited any height?
  frac w_faint == 0                  -> does NNLS literally ZERO the minor (explained away by majors)?
  #non-contributor decoys with w>w_faint, and the faint minor's rank-by-w -> precision/degeneracy.
Plus: faint recall@5 by w alone, split by #private alleles (does height help where private is sparse?).
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.optimize import nnls
from measure_noc5_ceiling import load_raw_genotypes, KNOWN

DATA=Path("data_insilico_w"); geno=load_raw_genotypes()
tk=np.load(DATA/"tokens8_test.npy")[:,:,:8].astype(np.float32)
at=np.load(DATA/"attr_test.npy"); noc=np.clip(np.load(DATA/"noc_test.npy").astype(int),1,5)
def akey(a):
    a=float(a); return "-2.0" if a==-2.0 else ("-1.0" if a==-1.0 else f"{round(a,1):.1f}")
panel=defaultdict(int)
for d,loc in geno.items():
    for L,al in loc.items():
        for a in al: panel[(L,a)]+=1
ref={c:set((L,a) for L,al in geno.get(KNOWN[c],{}).items() for a in al) for c in range(45)}
idxN5=[gi for gi in range(len(at)) if noc[gi]==5 and len(np.unique(at[gi][at[gi]>=0]))==5]

def fit(gi):
    v=np.where(at[gi]>=0)[0]
    keys=[(int(tk[gi][j,0]),akey(tk[gi][j,1])) for j in v]
    h=np.array([float(np.exp(tk[gi][j,2])) for j in v])
    G=np.zeros((len(v),45))
    for c in range(45):
        for r,k in enumerate(keys):
            if k in ref[c]: G[r,c]=1.0
    w,_=nnls(G,h)
    return v,keys,w

zero=0; decoy_beats=0; ranks=[]; wf_ratio=[]; rec=0
by_priv=defaultdict(lambda:[0,0])
for gi in idxN5:
    a=at[gi]; v=np.where(a>=0)[0]
    lh=tk[gi][:,2]; hsum={int(d):float(np.exp(lh[a==d]).sum()) for d in np.unique(a[v])}
    contribs=list(hsum); faint=min(hsum,key=hsum.get)
    _,keys,w=fit(gi)
    wf=w[faint]; wmaj=np.mean([w[c] for c in contribs if c!=faint])
    zero+=(wf<1e-6)
    noncon=[c for c in range(45) if c not in contribs]
    decoy_beats+=sum(1 for c in noncon if w[c]>wf+1e-9)
    ranks.append(int((w>wf+1e-9).sum()))                       # 0 = top by w
    wf_ratio.append(wf/(wmaj+1e-9))
    top5=set(np.argsort(w)[::-1][:5].tolist()); rec+=(faint in top5)
    # #private alleles present for the faint minor
    others=[KNOWN[o] for o in contribs if o!=faint]; gX=geno.get(KNOWN[faint],{}); pr=set()
    for L,al in gX.items():
        oh=set().union(*[geno.get(o,{}).get(L,set()) for o in others]) if others else set()
        for al2 in al:
            if al2 not in oh: pr.add((L,al2))
    npriv=len(pr & set(keys))
    b="[5+]" if npriv>=5 else ("[3-4]" if npriv>=3 else "[1-2]")
    by_priv[b][0]+=int(faint in top5); by_priv[b][1]+=1

n=len(idxN5)
print(f"N5 n={n}  (NNLS height-deconvolution, EuroForMix-spirit)\n")
print(f"  faint minor's NNLS weight == 0 (explained away by majors): {zero}/{n} = {zero/n:.2f}")
print(f"  mean w_faint / w_major ratio: {np.mean(wf_ratio):.3f}   (1.0 = equal credit)")
print(f"  mean rank of faint minor by w (0=top, 44=worst): {np.mean(ranks):.1f}")
print(f"  mean #non-contributor decoys outranking the faint minor: {decoy_beats/n:.1f}")
print(f"  faint recall@5 by w alone: {rec/n:.3f}")
print("\n  faint recall@5 by w, split by #PRIVATE alleles present (does height help where private is sparse?):")
for b in ["[1-2]","[3-4]","[5+]"]:
    h_,t_=by_priv[b]
    if t_: print(f"    {b:6s} priv: recall {h_/t_:.3f}  n={t_}")
print("\n  w_faint~0 + decoys outrank + low recall@5 => NNLS zeroes the faint minor (shared alleles explained")
print("  by majors, private height below noise) and degeneracy lets decoys win => why likelihood ID fails.")
