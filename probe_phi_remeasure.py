"""
Why is production phi corr -0.23 when small-scale parallel phi is +0.642?
Re-measure carefully: global vs per-NOC vs WITHIN-sample (what recon actually needs).
phi_pred = softplus (unnormalized abundance); phi_true = proportion (sums to 1 over present).
Test: scale/normalization artifact? inverted? and does phi find the FAINT minor in N5?
"""
import numpy as np
D="results/inc13_B_distill_seed42"; DA="data_insilico_w"
pp=np.load(f"{D}/phi_pred_test.npy"); pt=np.load(f"{DA}/phi_test.npy")
y=np.load(f"{DA}/y_test_set.npy").astype(bool); noc=np.load(f"{DA}/noc_test.npy")

def pear(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float); a,b=a-a.mean(),b-b.mean()
    d=a.std()*b.std(); return float((a*b).mean()/d) if d>1e-9 else float("nan")
def spear(a,b):  # rank then pearson
    ra=np.argsort(np.argsort(a)); rb=np.argsort(np.argsort(b)); return pear(ra,rb)

pres=y
print("=== GLOBAL (all present donors, all NOC) ===")
print(f"  raw  phi_pred vs phi_true corr = {pear(pp[pres],pt[pres]):+.3f}  (reproduce the -0.23)")

print("\n=== PER-NOC (present donors) corr ===")
for k in [1,2,3,4,5]:
    sel=noc==k; m=pres&sel[:,None]
    print(f"  N{k}: raw corr={pear(pp[m],pt[m]):+.3f}   n={m.sum()}")

print("\n=== WITHIN-SAMPLE (the recon-relevant metric) ===")
for k in [2,3,4,5]:
    idx=np.where(noc==k)[0]; sp=[]; pe=[]
    for i in idx:
        c=np.where(y[i])[0]
        if len(c)<2: continue
        sp.append(spear(pp[i,c],pt[i,c])); pe.append(pear(pp[i,c],pt[i,c]))
    print(f"  N{k}: within-sample Spearman={np.nanmean(sp):+.3f}  Pearson={np.nanmean(pe):+.3f}  (does phi order the donors right?)")

print("\n=== FAINT-MINOR identification (N5) ===")
idx=np.where(noc==5)[0]; hit=0; tot=0; inv=0
for i in idx:
    c=np.where(y[i])[0]
    faint_true=c[np.argmin(pt[i,c])]; faint_pred=c[np.argmin(pp[i,c])]
    tot+=1; hit+=int(faint_true==faint_pred)
    # is phi_pred INVERTED? (predicts faint donor as the STRONGEST)
    strong_pred=c[np.argmax(pp[i,c])]; inv+=int(faint_true==strong_pred)
print(f"  phi_pred's argmin == true faintest donor: {hit}/{tot} = {hit/tot:.3f}  (random={1/5:.2f})")
print(f"  phi_pred INVERTED (calls the faint donor the STRONGEST): {inv}/{tot} = {inv/tot:.3f}")

print("\n=== scale check ===")
print(f"  phi_pred present mean={pp[pres].mean():.3f} std={pp[pres].std():.3f}  | phi_true present mean={pt[pres].mean():.3f}")
print(f"  phi_pred absent  mean={pp[~pres].mean():.3f}  (should be ~0)")
