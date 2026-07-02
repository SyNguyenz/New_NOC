"""
Richer-data lever #1 — REPLICATES (EFMrep). Does combining R replicate amplifications of the SAME
mixture raise the deconvolution oracle at N5? Replicates have INDEPENDENT stochastic dropout, so a
faint minor recovered across replicates is new information (not diversity-bounded like better
modeling of one profile). Generated from each test mixture's stored ground truth (contributors =
y_test_set, phi = phi_test, template = meta_template_test) via make_insilico's EuroForMix peak model;
combined by a shared-phi copy-number EM (EFMrep). Standalone oracle (phi argsort) — model-independent.
"""
import numpy as np, make_insilico as mi
N_FLAT=mi.N_FLAT; DOS=mi.DONOR_DOSAGE; EFF=mi.EFF_BIN; AT=mi.AT; CV=mi.PH_CV
ST=mi.STUTTER_TARGET; BA=mi.BIN_ALLELE; BL=mi.BIN_LOCUS; SIZE=mi.build_bin_size()
SR_S,SR_I,SR_SG=mi.SR_SLOPE,mi.SR_INTERCEPT,mi.SR_SIGMA; DEGMAX=mi.DEG_BETA_MAX
C=DOS.shape[0]
logdose=np.where(DOS>0, np.log(np.maximum(DOS,1)), -1e9)            # (C,N_FLAT) copy-number log weights
parent=np.array([mi._BININDEX.get((int(BL[j]), round(float(BA[j])+1.0,1)), -1) for j in range(N_FLAT)])

y=np.load("data_insilico_w/y_test_set.npy"); phiT=np.load("data_insilico_w/phi_test.npy")
noc=np.load("data_insilico_w/noc_test.npy").clip(1,5); Tmpl=np.load("data_insilico_w/meta_template_test.npy")

def gen_rep(cols, phc, T, beta, rng):
    deg=np.exp(-beta*np.maximum(SIZE-100.0,0.0)); shape=1.0/CV**2; mix=np.zeros(N_FLAT)
    for c,p in zip(cols,phc):
        mu=T*p*DOS[c]*deg*EFF; nz=np.where(mu>0)[0]
        if len(nz): mix[nz]+=rng.gamma(shape, mu[nz]*CV**2)
    parent_h=mix.copy(); js=np.where((ST>=0)&(parent_h>0))[0]
    if len(js):
        a=BA[js].astype(float); sr=np.clip(SR_S*a+SR_I,0.01,0.18)*rng.lognormal(0,SR_SG,len(js))
        np.add.at(mix, ST[js], sr*parent_h[js])
    mix[mix<AT]=0.0; return mix

def corrected(mix, xi=0.08):
    pres=np.where(mix>0)[0]; h=mix[pres].copy()
    pj=parent[pres]; has=pj>=0
    h[has]=np.maximum(mix[pres[has]]-xi*mix[pj[has]], 0.0)
    return pres, h

def efm_phi(reps, n_iters=12):
    Ss=[]; hs=[]
    for mix in reps:
        pres,h=corrected(mix)
        if len(pres)==0: continue
        S=np.full((len(pres),C+1),-1e9); S[:,:C]=logdose[:,pres].T; S[:,C]=-2.0
        Ss.append(S); hs.append(h)
    if not Ss: return np.zeros(C)
    S=np.vstack(Ss); h=np.concatenate(hs); phi=np.ones(C+1)/(C+1)
    for _ in range(n_iters):
        z=S+np.log(phi+1e-9); z-=z.max(1,keepdims=True); A=np.exp(z); A/=A.sum(1,keepdims=True)
        w=(A[:,:C]*h[:,None]).sum(0); bg=(A[:,C]*h).sum(); tot=w.sum()+bg
        phi=np.concatenate([w,[bg]])/max(tot,1e-9)
    return phi[:C]

yti=(y>0.5).astype(int)
def oracle(P):
    e=np.zeros(N,bool)
    for i in range(N):
        k=int(yti[i].sum())
        if k==0: e[i]=True; continue
        p=np.zeros(C,int); p[np.argsort(P[i])[::-1][:k]]=1; e[i]=(p==yti[i]).all()
    return {j:round(float(e[noc==j].mean()),4) for j in (3,4,5)}, round(float(e.mean()),4)

R=3; N=len(y)
print("Sweep template scale to find the regime matching real test (R=1 N5~0.876, ~140 present bins),")
print("then read the R=2/R=3 lift THERE (the honest gain at the real operating point):\n")
for tmu in (np.log(750.0), np.log(3000.0), np.log(8000.0)):
    rng=np.random.default_rng(42); PH={r:np.zeros((N,C)) for r in range(1,R+1)}; npres=[]
    for i in range(N):
        cols=np.where(y[i]>0.5)[0]
        if len(cols)==0: continue
        phc=phiT[i,cols]; phc=phc/max(phc.sum(),1e-9)
        T=float(np.exp(rng.normal(tmu, mi.PEAK_TSIG))); beta=rng.uniform(0,DEGMAX)
        reps=[gen_rep(cols,phc,T,beta,rng) for _ in range(R)]
        if noc[i]==5: npres.append(int((reps[0]>0).sum()))
        for r in range(1,R+1): PH[r][i]=efm_phi(reps[:r])
    print(f"--- template median={np.exp(tmu):.0f} | R=1 N5 present-bins median={np.median(npres):.0f} (real~140) ---")
    for r in range(1,R+1):
        pc,ov=oracle(PH[r]); print(f"    R={r}: overall {ov:.4f} | N3 {pc[3]:.4f} N4 {pc[4]:.4f} N5 {pc[5]:.4f}")
