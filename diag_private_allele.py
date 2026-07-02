"""
diag_private_allele.py — would a HARD 'max-score to any donor with a present private allele' rule
help or hurt N5?  A private allele = a present peak key carried by exactly ONE of the 45 candidates.
In a closed set where all N5 contributors are known, a NON-donor with a present private allele can
only be stutter / drop-in / noise (a phantom).  We measure how many phantoms exist and their height
vs real (true-donor) private alleles -- this decides hard-rule (height-blind) vs soft height-weighted.
"""
import numpy as np, phi_rerank as pr
from collections import defaultdict

DATA = "data_insilico_w/"
tok = np.load(DATA+"tokens8_test.npy"); mask = np.load(DATA+"mask_test.npy").astype(bool)
y = np.load(DATA+"y_test_set.npy"); noc = np.load(DATA+"noc_test.npy")
g = np.load("data/donor_geno.npy"); gmask = np.load("data/donor_geno_mask.npy")
carr, C = pr.build_carriers(g, gmask)
n5 = np.where(noc == 5)[0]

true_priv_n, nond_priv_n = [], []          # per-mixture: # true / # non donors with >=1 present private
true_priv_h, nond_priv_h = [], []          # heights of true-private and nondonor-private alleles
mix_with_phantom = 0                        # mixtures where >=1 non-donor has a present private (hard rule mis-fires)
for i in n5:
    owner_h = defaultdict(list)
    for p in np.where(mask[i])[0]:
        key = (int(round(float(tok[i,p,0]))), int(round(float(tok[i,p,1])*10)))
        h = float(np.expm1(tok[i,p,2]))
        if key in carr and len(carr[key]) == 1:        # private to exactly one candidate
            owner_h[carr[key][0]].append(h)
    T = set(np.where(y[i] > 0.5)[0])
    t_owners = [c for c in owner_h if c in T]
    d_owners = [c for c in owner_h if c not in T]
    true_priv_n.append(len(t_owners)); nond_priv_n.append(len(d_owners))
    for c in t_owners: true_priv_h += owner_h[c]
    for c in d_owners: nond_priv_h += owner_h[c]
    if d_owners: mix_with_phantom += 1

true_priv_n=np.array(true_priv_n); nond_priv_n=np.array(nond_priv_n)
true_priv_h=np.array(true_priv_h); nond_priv_h=np.array(nond_priv_h)
def pct(a,qs=(10,25,50,75,90)): return {q:round(float(np.percentile(a,q)),1) for q in qs} if len(a) else {}

print(f"N5 mixtures: {len(n5)}\n")
print("=== TRUE donors with >=1 present PRIVATE allele (correctly boostable) ===")
print(f"  per mixture: mean {true_priv_n.mean():.2f}  (of 5 true donors)  pctl {pct(true_priv_n)}")
print(f"  mixtures where ALL 5 true donors have a private allele: {(true_priv_n==5).mean()*100:.1f}%")
print(f"  mixtures where >=1 true donor has NO private allele      : {(true_priv_n<5).mean()*100:.1f}%")
print(f"  true-private allele heights (RFU) pctl {pct(true_priv_h)}  n={len(true_priv_h)}\n")
print("=== NON-donors with >=1 present PRIVATE allele (PHANTOM: stutter/drop-in/noise) ===")
print(f"  per mixture: mean {nond_priv_n.mean():.3f}   pctl {pct(nond_priv_n)}")
print(f"  mixtures with >=1 phantom-private non-donor (hard rule WOULD mis-fire): {mix_with_phantom}/{len(n5)} = {100*mix_with_phantom/len(n5):.1f}%")
print(f"  phantom-private allele heights (RFU) pctl {pct(nond_priv_h)}  n={len(nond_priv_h)}")
if len(nond_priv_h):
    print(f"  phantom privates below 150 RFU: {(nond_priv_h<150).mean()*100:.1f}% | below 300 RFU: {(nond_priv_h<300).mean()*100:.1f}%")
