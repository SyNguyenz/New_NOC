"""DATA-SIDE check before any model work: is the generator 'giong that' enough — does the REALISTIC mode close
the gaps, and (crucially) does it reproduce the real DECOY STRUCTURE (the thing that drives the N5 wall)?
Compare synthetic N5 (realistic vs original) vs REAL test N5 on:
  EPG: height p10/50/90, back-stutter frac, alleles/locus, min-phi/ratio
  DECOY STRUCTURE: hardest true donor's PRIVATE-present alleles (min over the 5), best-decoy DAMNING-absence,
                   best-decoy UNIQUE-support (allele-compat non-contributor) — the difficulty the model trains on.
If realistic matches real here -> data side resolved, residual N5 misses are MODEL's fault. If gaps remain -> data first."""
import os
os.environ.setdefault("STR_DATA_DIR","data")
import numpy as np
from pathlib import Path
import make_insilico as MI
ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(0); pool=MI.build_ss_pool()
g=np.load(ROOT/"data/donor_geno.npy"); gm=np.load(ROOT/"data/donor_geno_mask.npy")
geno=[set() for _ in range(45)]
for c in range(45):
    for j in range(g.shape[1]):
        if gm[c,j]: geno[c].add((int(g[c,j,0]),round(float(g[c,j,1]),1)))

def mix_of_tok(tok,mask):
    loc={}
    for j in np.where(mask)[0]:
        l=int(tok[j,0]); a=round(float(tok[j,1]),1); h=float(np.expm1(tok[j,2]))
        loc.setdefault(l,{}); loc[l][a]=max(loc[l].get(a,0.),h)
    return loc

def decoy_struct(loc,true):
    expl=set().union(*[geno[c] for c in true])
    minpriv=99
    for t in true:
        oth=true-{t}; eo=set().union(*[geno[c] for c in oth]) if oth else set()
        pres={(l,a) for (l,a) in geno[t] if l in loc and a in loc[l]}
        minpriv=min(minpriv,len(pres-eo))
    best=(-1,0,0)
    for c in range(45):
        if c in true: continue
        dpres={(l,a) for (l,a) in geno[c] if l in loc and a in loc[l]}
        if len(dpres)>best[0]:
            duniq=dpres-expl; ddam=sum(1 for (l,a) in geno[c] if l in loc and a not in loc[l])
            best=(len(dpres),len(duniq),ddam)
    return minpriv,best[1],best[2]

def stats(get_iter,tag,n):
    allh=[];stut=[];occ=[];mp=[];rt=[];mpriv=[];duniq=[];ddam=[]
    cnt=0
    for loc,true,phi in get_iter:
        for l,d in loc.items():
            occ.append(len(d))
            for a,h in d.items():
                allh.append(h); par=d.get(round(a+1.0,1),0.0)
                stut.append(1 if (par>0 and 0.03<h/par<0.20) else 0)
        if phi is not None and len(phi)>0:
            mp.append(min(phi)); rt.append(max(phi)/max(min(phi),1e-6))
        a,b,c=decoy_struct(loc,true); mpriv.append(a); duniq.append(b); ddam.append(c)
        cnt+=1
        if cnt>=n: break
    allh=np.array(allh)
    print(f"[{tag}] n={cnt}")
    print(f"   EPG: height p10/50/90={np.percentile(allh,10):.0f}/{np.median(allh):.0f}/{np.percentile(allh,90):.0f} | stutter={np.mean(stut):.3f} | alleles/locus={np.mean(occ):.2f}"+(f" | min-phi={np.median(mp):.3f} ratio={np.median(rt):.1f}" if mp else ""))
    print(f"   DECOY: min true-donor private(present)={np.median(mpriv):.1f} (frac<=2: {np.mean(np.array(mpriv)<=2):.2f}) | best-decoy unique-support={np.median(duniq):.1f} | best-decoy damning={np.median(ddam):.1f}")

BIN_SIZE=MI.build_bin_size()
def synth_iter(mode, peak=False):
    while True:
        cols=sorted(rng.choice(45,5,replace=False).tolist())
        if peak: xf,y,k,tt,ab,ph=MI.gen_mixture_peak(cols,rng,BIN_SIZE,mode=mode)
        else:    xf,y,k,tt,ab,ph=MI.gen_mixture(cols,pool,rng,mode=mode)
        tok,mask,attr,size=MI.xflat_to_tokens(xf)
        phi=[ph[c] for c in cols]
        yield mix_of_tok(tok,mask), set(cols), phi

def real_iter():
    DATA=ROOT/"data_insilico_w"
    tok=np.load(DATA/"tokens8_test.npy").astype(np.float32)[:,:,:3]; mk=np.load(DATA/"mask_test.npy").astype(bool)
    y=np.load(DATA/"y_test_set.npy"); noc=np.load(DATA/"noc_test.npy").astype(int)
    phit=np.load(DATA/"phi_test.npy") if (DATA/"phi_test.npy").exists() else None
    for i in np.where(noc==5)[0]:
        true=set(int(x) for x in np.where(y[i]>0.5)[0])
        phi=[float(phit[i][c]) for c in true] if phit is not None and phit.ndim==2 and phit.shape[1]>=45 else None
        yield mix_of_tok(tok[i],mk[i]), true, phi

print("=== generator fidelity on N5: PEAK-MODEL realistic & wide vs REAL ===")
stats(synth_iter("realistic",peak=True),"PEAK realistic",1500)
stats(synth_iter("wide",peak=True),"PEAK wide",1500)
stats(real_iter(),"REAL test",10000)
print("\nverdict: realistic should match REAL on EPG AND decoy structure. Remaining gaps = data side NOT done.")
