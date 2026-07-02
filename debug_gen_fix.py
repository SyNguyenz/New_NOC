"""
Validate the root-cause fix cheaply (no retrain): regenerate synthetic N5 with
REAL-MATCHED mixture proportions + template, and show the minor-peak height
distribution and the synth-vs-real domain gap collapse toward real.

OLD generator (make_insilico.gen_mixture):
   r_max = 6 + 5*(k-2) = 21 ;  phi = exp(U(0,log r_max))/sum ;  t_total ~ LN(log 32000, .55)
FIX (matched to real-test N5 stats measured in debug_gen_proportions):
   ratio max/min ~ med 4 (incl. equal mixtures) ;  t_total ~ LN(log 40000, .8)
"""
import os, numpy as np
os.environ["STR_DATA_DIR"] = str((__import__("pathlib").Path(__file__).resolve().parent)/"data_insilico_w")
import importlib, make_insilico as M
importlib.reload(M)
from pathlib import Path
ROOT=Path(__file__).resolve().parent; DATA=ROOT/"data_insilico_w"
rng=np.random.default_rng(0)
pool=M.build_ss_pool()
BIN_LOCUS=M.BIN_LOCUS; N_FLAT=M.N_FLAT; AT=M.AT; GAMMA=M.GAMMA_SHAPE

def gen(cols, mode):
    k=len(cols)
    if mode=="old":
        r_max=6.0+5.0*(k-2); w=np.exp(rng.uniform(0,np.log(r_max),k)); phi=w/w.sum()
        t_total=float(np.exp(rng.normal(np.log(32000),0.55)))
    else:  # fix: real-matched. Dirichlet proportions (balanced, min-phi~0.125) + louder/wider template + more jitter
        if rng.random()<0.2:
            phi=np.full(k,1.0/k)                                   # equal mixture (real has these)
        else:
            phi=rng.dirichlet(np.full(k,6.0))                      # concentration 6 -> ratio med ~4, min-phi ~0.12
        t_total=float(np.exp(rng.normal(np.log(55000),1.0)))       # match real median 40k + heavy tail to ~180k
    gshape=5.0 if mode=="fix" else GAMMA                           # lower shape -> more per-peak variability (real)
    mix=np.zeros(N_FLAT); contrib=np.zeros((k,N_FLAT))
    for di,(c,p) in enumerate(zip(cols,phi)):
        prof=pool[c][rng.integers(len(pool[c]))]; h=np.expm1(prof.astype(np.float64)); s=h.sum()
        if s<=0: continue
        h=h/s; jit=rng.gamma(gshape,1.0/gshape,size=N_FLAT); contrib[di]=p*t_total*h*jit; mix+=contrib[di]
    drop=mix<AT; mix[drop]=0
    attr=np.full(N_FLAT,-1,np.int16)
    live=(~drop)&(contrib.max(0)>0); attr[live]=np.asarray(cols,np.int16)[contrib.argmax(0)[live]]
    return mix, attr, phi

def minor_heights(mode, n=3000):
    hs=[]; minphi=[]; ratio=[]
    for _ in range(n):
        cols=sorted(rng.choice(45,size=5,replace=False).tolist())
        mix,attr,phi=gen(cols,mode)
        minphi.append(phi.min()); ratio.append(phi.max()/phi.min())
        for di,c in enumerate(cols):
            if phi[di]<0.2:
                pe=(attr==c)&(mix>0)
                if pe.sum(): hs.extend(mix[pe].tolist())
    return np.array(hs), np.array(minphi), np.array(ratio)

# real test minor heights
def real_minor():
    tk=np.load(DATA/"tokens8_test.npy"); mk=np.load(DATA/"mask_test.npy").astype(bool)
    at=np.load(DATA/"attr_test.npy").astype(int); phi=np.load(DATA/"phi_test.npy"); noc=np.load(DATA/"noc_test.npy").astype(int)
    h=np.expm1(tk[:,:,2]); hs=[]
    for i in np.where(noc==5)[0]:
        for c in np.where((phi[i]>0)&(phi[i]<0.2))[0]:
            pe=(at[i]==c)&mk[i]
            if pe.sum(): hs.extend(h[i][pe].tolist())
    return np.array(hs)

def desc(a,t):
    print(f"  {t:<22} med={np.median(a):7.0f}  p10={np.percentile(a,10):6.0f}  p90={np.percentile(a,90):7.0f}  "
          f"frac<100rfu={np.mean(a<100):.2f}")

ho,mpo,ro=minor_heights("old"); hf,mpf,rf_=minor_heights("fix"); hr=real_minor()
print("N5 MINOR-peak height (rfu):")
desc(ho,"OLD synthetic"); desc(hf,"FIXED synthetic"); desc(hr,"REAL test")
print(f"\nmin-phi:  OLD med={np.median(mpo):.3f}  FIX med={np.median(mpf):.3f}  REAL med=0.125")
print(f"ratio:    OLD med={np.median(ro):.2f}   FIX med={np.median(rf_):.2f}   REAL med=4.0")

# domain gap: classifier old-vs-real and fix-vs-real on log1p heights
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
def auc_vs_real(hsyn):
    X=np.log1p(np.r_[hsyn,hr]).reshape(-1,1); ylab=np.r_[np.zeros(len(hsyn)),np.ones(len(hr))]
    p=np.random.RandomState(0).permutation(len(X)); X,ylab=X[p],ylab[p]; n=int(.7*len(X))
    clf=LogisticRegression().fit(X[:n],ylab[:n]); return roc_auc_score(ylab[n:],clf.predict_proba(X[n:])[:,1])
print(f"\ndomain AUC (height) vs real:  OLD={auc_vs_real(ho):.3f}  FIX={auc_vs_real(hf):.3f}  (0.5=matched)")
