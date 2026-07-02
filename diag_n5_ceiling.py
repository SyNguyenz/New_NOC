"""
diag_n5_ceiling.py — DECOMPOSE the N5 oracle ceiling: is it (A) minor signal too low
(dropout / information floor) or (B) decoys not suppressed (discrimination)?

Model-free.  Uses only peaks + reference genotypes + truth (data_insilico_w + data/donor_geno).
Forensic grounding:
  • coverage_c = fraction of donor c's obligate alleles that have a matching peak. Low coverage =
    allelic DROP-OUT (Balding-Buckleton / Tvedebrink). This is hypothesis (A).
  • present-PRIVATE alleles of a true donor = its present obligate alleles NOT carried by any OTHER
    true contributor. 0 present-private => donor is allele-REDUNDANT given the others => its presence
    is information-theoretically unprovable from allele presence (a HARD oracle ceiling, strongest form
    of (A)). Heights can still help, so we also report the height of those private alleles.
  • confuser decoys = non-donors whose coverage >= the weakest true donor's coverage. These are the
    non-donors a presence-ranker cannot push below the real minor => hypothesis (B), discrimination.
We also run the height-aware EM phi (phi_rerank.deconv_phi, the deployable independent signal) and,
for every FAILED oracle case, label the missed true donor as 'too-low' vs 'lost-to-decoy'.
"""
import numpy as np
import phi_rerank as pr

DATA = "data_insilico_w/"
tok = np.load(DATA + "tokens8_test.npy")          # (N,160,8) [locus, allele, log1p_h, ...]
mask = np.load(DATA + "mask_test.npy").astype(bool)
y = np.load(DATA + "y_test_set.npy")              # (N,45) true membership
noc = np.load(DATA + "noc_test.npy")
g = np.load("data/donor_geno.npy")                # (45,46,11) [locus, allele, ...]
gm = np.load("data/donor_geno_mask.npy")          # (45,46)

C = g.shape[0]
# obligate allele-key set per donor (same keying as phi_rerank.build_carriers)
oblig = []
for c in range(C):
    s = set()
    for j in range(g.shape[1]):
        if gm[c, j]:
            s.add((int(round(float(g[c, j, 0]))), int(round(float(g[c, j, 1]) * 10))))
    oblig.append(s)

n5 = np.where(noc == 5)[0]
print(f"N5 samples: {len(n5)}\n")

min_cov = []          # weakest true donor coverage per mixture
weak_height = []      # mean present-allele height of weakest true donor (RFU)
n_redundant = []      # # true donors with 0 present-private alleles (hard non-identifiable)
priv_min_height = []  # for identifiable donors, the min over true donors of (max private-allele height)
n_full_mimic = []     # # non-donors with coverage==1.0
n_confuser = []       # # non-donors with coverage >= weakest true-donor coverage

for i in n5:
    present = set()
    h_of = {}
    for p in np.where(mask[i])[0]:
        k = (int(round(float(tok[i, p, 0]))), int(round(float(tok[i, p, 1]) * 10)))
        h = float(np.expm1(tok[i, p, 2]))
        present.add(k); h_of[k] = max(h_of.get(k, 0.0), h)
    T = [c for c in range(C) if y[i, c] > 0.5]
    # coverage per candidate
    cov = np.array([ (len(oblig[c] & present) / max(1, len(oblig[c]))) for c in range(C) ])
    # weakest true donor
    covT = cov[T]
    wk = T[int(np.argmin(covT))]
    min_cov.append(cov[wk])
    pres_wk = oblig[wk] & present
    weak_height.append(np.mean([h_of[k] for k in pres_wk]) if pres_wk else 0.0)
    # private (among contributors) present alleles
    redun = 0; mins = []
    for c in T:
        others = set().union(*[oblig[d] for d in T if d != c])
        priv_present = [k for k in (oblig[c] & present) if k not in others]
        if not priv_present:
            redun += 1
        else:
            mins.append(max(h_of[k] for k in priv_present))
    n_redundant.append(redun)
    priv_min_height.append(min(mins) if mins else 0.0)
    # decoys
    nond = [c for c in range(C) if c not in T]
    n_full_mimic.append(sum(1 for d in nond if cov[d] >= 0.999))
    n_confuser.append(sum(1 for d in nond if cov[d] >= cov[wk] - 1e-9))

min_cov = np.array(min_cov); weak_height = np.array(weak_height)
n_redundant = np.array(n_redundant); n_full_mimic = np.array(n_full_mimic)
n_confuser = np.array(n_confuser); priv_min_height = np.array(priv_min_height)

def pct(a, qs=(10, 25, 50, 75, 90)):
    return {q: round(float(np.percentile(a, q)), 3) for q in qs}

print("=== (A) MINOR-SIGNAL-TOO-LOW (dropout / information floor) ===")
print(f"weakest true-donor coverage  pctl {pct(min_cov)}  mean {min_cov.mean():.3f}")
print(f"  mixtures with a true donor <100% covered : {(min_cov<0.999).mean()*100:.1f}%")
print(f"  mixtures with a true donor <70%  covered : {(min_cov<0.70).mean()*100:.1f}%")
print(f"  mixtures with a true donor <50%  covered : {(min_cov<0.50).mean()*100:.1f}%")
print(f"weakest-donor present-allele mean height (RFU) pctl {pct(weak_height)}")
print()
print("=== HARD CEILING: allele-redundant (non-identifiable) true donors ===")
print(f"  # redundant true donors per N5 mixture pctl {pct(n_redundant)}  mean {n_redundant.mean():.3f}")
print(f"  mixtures with >=1 redundant (presence-unprovable) true donor : {(n_redundant>=1).mean()*100:.1f}%")
print(f"  of identifiable donors, min private-allele height (RFU) pctl {pct(priv_min_height)}")
print()
print("=== (B) DECOY-NOT-SUPPRESSED (discrimination) ===")
print(f"  full-coverage non-donor mimics per mixture pctl {pct(n_full_mimic)}  mean {n_full_mimic.mean():.3f}")
print(f"  confuser decoys (cov >= weakest true) per mixture pctl {pct(n_confuser)}  mean {n_confuser.mean():.3f}")
print(f"  mixtures with >=1 confuser decoy : {(n_confuser>=1).mean()*100:.1f}%")
print()

# === height-aware EM phi ranking: classify each oracle FAILURE ===
PH = pr.deconv_phi(tok[n5], mask[n5], g, gm, n_iters=12)   # (n5,45) proportions
yk = y[n5]
too_low = 0; lost_decoy = 0; ok = 0; total_missed = 0
for r, i in enumerate(n5):
    T = [c for c in range(C) if y[i, c] > 0.5]
    top5 = set(np.argsort(PH[r])[::-1][:5])
    missed = [c for c in T if c not in top5]
    if not missed:
        ok += 1; continue
    present = set()
    for p in np.where(mask[i])[0]:
        present.add((int(round(float(tok[i, p, 0]))), int(round(float(tok[i, p, 1]) * 10))))
    cov = {c: len(oblig[c] & present)/max(1,len(oblig[c])) for c in range(C)}
    for c in missed:
        total_missed += 1
        others = set().union(*[oblig[d] for d in T if d != c])
        priv = [k for k in (oblig[c] & present) if k not in others]
        # 'too-low' = weak signal: low coverage OR no present-private allele (redundant)
        if cov[c] < 0.70 or len(priv) == 0:
            too_low += 1
        else:
            lost_decoy += 1

print("=== EM-phi (height-aware, deployable) oracle-N5 FAILURE classification ===")
print(f"  phi oracle-N5 exact-match: {ok}/{len(n5)} = {ok/len(n5):.3f}")
print(f"  total missed true donors in failed mixtures: {total_missed}")
print(f"    too-low (cov<0.70 or 0 private allele) : {too_low}  ({100*too_low/max(1,total_missed):.1f}%)")
print(f"    lost-to-decoy (well-covered, has private): {lost_decoy}  ({100*lost_decoy/max(1,total_missed):.1f}%)")
