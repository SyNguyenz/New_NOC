"""
Is the N5 decoy wall INFO-limited or MODEL-limited?  Check the RAW data (alleles + heights + full
45-donor panel), independent of the model. Use the model only to identify its actual hard confusions
(missed-true T, decoy D) on N5; then ask of the RAW mix:

  T_necessary : does T carry an allele PRESENT in the mix that NONE of the model's chosen 5-set
                (4 hits + decoy) can explain?  >0  => raw data PROVES T is present => MODEL-limited.
  D_superfluous: are ALL of decoy D's present alleles explained by the TRUE 5-set (without D)?
                => raw data says D is eliminable => MODEL-limited.
  height check : for the allele-ambiguous T (T_necessary=0), is there a HEIGHT excess at T's alleles
                that the model's set under-explains? (quantitative residual)

If T_necessary>0 / D superfluous are COMMON => info is THERE, model just failed.
If T has no private allele AND no height residual => genuinely INFO-limited.
"""
import numpy as np
D="results/inc13_B_distill_seed42"; DA="data_insilico_w"; G="data/donor_geno.npy"
def ab(a): return int(round(float(a)*10))
def kk(l,a): return (int(round(float(l))),ab(a))
g=np.load(G); gm=np.load(G.replace(".npy","_mask.npy")).astype(bool); C=g.shape[0]
ditems=[{} for _ in range(C)]
for c in range(C):
    for j in range(g.shape[1]):
        if gm[c,j]:
            it=kk(g[c,j,0],g[c,j,1]); ditems[c][it]=ditems[c].get(it,0)+1   # dosage
dset=[set(d) for d in ditems]

tk=np.load(f"{DA}/tokens8_test.npy"); mk=np.load(f"{DA}/mask_test.npy").astype(bool)
y_p=np.load(f"{D}/y_test_pred.npy").astype(bool); y_t=np.load(f"{D}/y_test_true.npy").astype(bool)
noc=np.load(f"{DA}/noc_test.npy")
H=np.expm1(tk[:,:,2])

def obs_heights(i):
    o={}
    for k in np.where(mk[i])[0]:
        it=kk(tk[i,k,0],tk[i,k,1]); o[it]=max(o.get(it,0.0),H[i,k])
    return o

for NV in [5,4]:
    sel=np.where(noc==NV)[0]
    nT=0; T_nec_pos=0; T_nec_counts=[]; nD=0; D_superf=0
    amb_resid=[]   # for allele-ambiguous T: height excess at T's private-to-set alleles
    for i in sel:
        O=obs_heights(i); Oset=set(O)
        pred=np.where(y_p[i])[0]; true=np.where(y_t[i])[0]
        miss=[c for c in true if not y_p[i,c]]; dec=[c for c in pred if not y_t[i,c]]
        pred_alleles=set().union(*[dset[c] for c in pred]) if len(pred) else set()
        true_alleles=set().union(*[dset[c] for c in true]) if len(true) else set()
        for T in miss:
            nT+=1
            Tpres=dset[T]&Oset
            Tnec=Tpres - pred_alleles            # T's present alleles NO chosen donor explains
            T_nec_counts.append(len(Tnec))
            if len(Tnec)>0: T_nec_pos+=1
        for Dd in dec:
            nD+=1
            Dpres=dset[Dd]&Oset
            Dnec_vs_true=Dpres - true_alleles    # D's present alleles the TRUE set can't explain
            if len(Dnec_vs_true)==0: D_superf+=1
    print(f"=== N{NV}  (miss T={nT}, decoy D={nD}) ===")
    print(f"  T_necessary (raw allele ONLY T explains, present): {T_nec_pos}/{nT} = {T_nec_pos/max(1,nT):.3f}  "
          f"mean#={np.mean(T_nec_counts):.2f}")
    print(f"     => {T_nec_pos/max(1,nT)*100:.0f}% of missed-true have a present allele the model's chosen set CANNOT explain (MODEL-limited)")
    print(f"  decoy fully superfluous (all its present alleles explained by TRUE set): {D_superf}/{nD} = {D_superf/max(1,nD):.3f}")
    print(f"     => {D_superf/max(1,nD)*100:.0f}% of decoys are raw-eliminable (the true set explains everything they do)")
    # of the T with NO private allele -> truly allele-ambiguous; how many?
    amb = sum(1 for c in T_nec_counts if c==0)
    print(f"  allele-AMBIGUOUS missed-true (T_necessary=0): {amb}/{nT} = {amb/max(1,nT):.3f}  <- the candidate info-limited core")
    # SYMMETRIC deployable test: vs the 4 confident hits, does T explain MORE unexplained peak-height than D?
    Tg_h=[]; Dg_h=[]; Tg_c=[]; Dg_c=[]
    for i in sel:
        O=obs_heights(i); pred=np.where(y_p[i])[0]; true=np.where(y_t[i])[0]
        hits=[c for c in true if y_p[i,c]]                         # confident common baseline
        base=set().union(*[dset[c] for c in hits]) if hits else set()
        unexp={it:h for it,h in O.items() if it not in base}        # peaks the 4 hits can't explain
        for T in [c for c in true if not y_p[i,c]]:
            gain=[unexp[it] for it in dset[T] if it in unexp]; Tg_h.append(sum(gain)); Tg_c.append(len(gain))
        for Dd in [c for c in pred if not y_t[i,c]]:
            gain=[unexp[it] for it in dset[Dd] if it in unexp]; Dg_h.append(sum(gain)); Dg_c.append(len(gain))
    def auc(p,n):
        p,n=np.asarray(p,float),np.asarray(n,float)
        if not len(p) or not len(n): return float("nan")
        a=np.concatenate([p,n]); _,inv,cnt=np.unique(a,return_inverse=True,return_counts=True)
        cs=np.cumsum(cnt); rk=((cs-cnt+cs+1)/2.0)[inv]
        return (rk[:len(p)].sum()-len(p)*(len(p)+1)/2)/(len(p)*len(n))
    print(f"  [SYMMETRIC vs 4-hit baseline] marginal unexplained-peak coverage T vs decoy:")
    print(f"     #peaks:  T mean={np.mean(Tg_c):.2f}  decoy mean={np.mean(Dg_c):.2f}   AUC(T>D)={auc(Tg_c,Dg_c):.3f}")
    print(f"     HEIGHT:  T mean={np.mean(Tg_h):.0f}  decoy mean={np.mean(Dg_h):.0f}   AUC(T>D)={auc(Tg_h,Dg_h):.3f}  <- deployable raw separability\n")
