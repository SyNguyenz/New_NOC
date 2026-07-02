"""Cross-locus mechanism (Yu-like locus-association): phi_rerank uses only the GLOBAL phi (summed mass).
The cross-locus signal it IGNORES = per-donor CONSISTENCY of implied-phi across loci. A true donor explains
its carried alleles across loci with ONE consistent implied-phi; a decoy is spiky (high at coincidental loci,
~0 elsewhere). We compute, per donor: global phi, locus COVERAGE, and implied-phi CONSISTENCY (1 - CV across
supported loci), then test each as an LOP rerank channel vs phi_rerank on in-silico oracle-NOC EM.

Decisive test of "is there ignored cross-locus info?": if consistency/coverage BEATS phi_rerank -> yes (the
locus-independence ceiling argument is wrong). If it ties -> confirms the generator is locus-independent and
the mechanism is banked for real data."""
import numpy as np, importlib.util
from pathlib import Path
PROJ = Path("."); DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"; CACHE = PROJ / "cache_insilico"; K = 45
def lm(n, p): s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pr = lm("pr", PROJ / "inc22_clean" / "phi_rerank.py")
g = np.load(GENO / "donor_geno.npy").astype(np.float32); gm = np.load(GENO / "donor_geno_mask.npy")

# carriers: (locus, allele*10) -> {donor: copies}
cn = {}
for c in range(g.shape[0]):
    for j in range(g.shape[1]):
        if gm[c, j]:
            key = (int(round(g[c, j, 0])), int(round(g[c, j, 1] * 10)))
            cn.setdefault(key, {})[c] = cn.setdefault(key, {}).get(c, 0) + 1
# loci each donor carries (count of distinct present-able loci)
donor_loci = [set() for _ in range(K)]
for (L, a), dd in cn.items():
    for c in dd: donor_loci[c].add(L)

def crosslocus(tokens, mask, n_iters=12):
    N = len(tokens); mask = mask.astype(bool)
    PHI = np.zeros((N, K)); COV = np.zeros((N, K)); CONS = np.zeros((N, K))
    for i in range(N):
        ps = np.where(mask[i])[0]
        info = []
        for p in ps:
            L = int(round(tokens[i, p, 0])); a = int(round(tokens[i, p, 1] * 10)); h = float(np.expm1(tokens[i, p, 2]))
            if (L, a) in cn: info.append((L, a, h))
        if not info: continue
        n = len(info); S = np.full((n, K + 1), -1e9); h = np.array([x[2] for x in info])
        for r, (L, a, _) in enumerate(info):
            for c, cp in cn[(L, a)].items(): S[r, c] = 0.0   # UNIFORM compat = canonical phi_rerank (was log(cp): efm-family, mislabeled)
            S[r, K] = -2.0
        phi = np.ones(K + 1) / (K + 1)
        for _ in range(n_iters):
            z = S + np.log(phi + 1e-9); z -= z.max(1, keepdims=True); A = np.exp(z); A /= A.sum(1, keepdims=True)
            w = (A[:, :K] * h[:, None]).sum(0); bg = (A[:, K] * h).sum(); phi = np.concatenate([w, [bg]]) / max(w.sum() + bg, 1e-9)
        PHI[i] = phi[:K]
        # per-locus decomposition
        loci = sorted(set(x[0] for x in info))
        # per-locus total height and per-locus per-donor mass & dosage
        massLC = {L: np.zeros(K) for L in loci}; dosLC = {L: np.zeros(K) for L in loci}; totL = {L: 0.0 for L in loci}
        for r, (L, a, hh) in enumerate(info):
            totL[L] += hh
            for c, cp in cn[(L, a)].items():
                massLC[L][c] += A[r, c] * hh; dosLC[L][c] += cp
        # per-locus scale s_L = total_L / sum_c phi_c * dosage_cL  (absorbs T*eff*deg per locus, per sample)
        sL = {}
        for L in loci:
            pred = float((phi[:K] * dosLC[L]).sum()); sL[L] = totL[L] / max(pred, 1e-9)
        for c in range(K):
            if phi[c] <= 1e-6: continue
            impl = []; sup = 0
            for L in donor_loci[c]:
                if L in dosLC and dosLC[L][c] > 0:
                    if massLC[L][c] > 0.02 * max(totL[L], 1e-9): sup += 1
                    impl.append(massLC[L][c] / max(dosLC[L][c] * sL[L], 1e-9))
            carried_present = sum(1 for L in donor_loci[c] if L in totL)
            COV[i, c] = sup / max(carried_present, 1)
            if len(impl) >= 2:
                im = np.array(impl); cv = im.std() / (im.mean() + 1e-9); CONS[i, c] = 1.0 / (1.0 + cv)  # 1 at perfect consistency
    return PHI, COV, CONS

def load(sp):
    d = {k: np.load(DATA / f"{k}_{sp}.npy") for k in ["tokens8", "mask", "phi", "noc"]}
    d["y"] = (d["phi"] > 0).astype(np.float32); d["cls"] = np.load(CACHE / f"cls_{sp}.npy")
    return d
va, te = load("val"), load("test")
def z(a): a = a.astype(np.float64); s = a.std(); return (a - a.mean()) / (s if s > 1e-9 else 1.0)
def decode(score, k):
    yp = np.zeros((len(score), K), int)
    for i in range(len(score)): yp[i, np.argsort(score[i])[::-1][:int(k[i])]] = 1
    return yp
def per_noc_em(yp, y, noc):
    yt = (y > 0.5).astype(int); per = {}
    for i in range(len(yp)): per.setdefault(int(noc[i]), []).append(bool((yp[i] == yt[i]).all()))
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}
def fmt(em): return "  ".join(f"{em.get(k,0):.3f}" for k in (1, 2, 3, 4, 5)) + f"   {np.mean([em.get(k,0) for k in (3,4,5)]):.3f}"
def lop(L, sig, a, log=True): return np.stack([z(L[i]) + a * z(np.log(sig[i] + 1e-6) if log else sig[i]) for i in range(len(L))])
def tune(L, sig, y, noc, log=True, grid=(0, .1, .2, .3, .5, .75, 1, 1.5, 2)):
    bv, ba = -1, 0
    for a in grid:
        em = per_noc_em(decode(lop(L, sig, a, log), noc.clip(1, 5)), y, noc); v = np.mean([em.get(k, 0) for k in (3, 4, 5)])
        if v > bv: bv, ba = v, a
    return ba

PHv, COVv, CONSv = crosslocus(va["tokens8"], va["mask"]); PHt, COVt, CONSt = crosslocus(te["tokens8"], te["mask"])
oN = te["noc"].clip(1, 5)
print("=== cross-locus channels vs phi_rerank (in-silico test, ORACLE-NOC EM) ===")
print(f"  {'channel':<30}{'a':>5}   N1     N2     N3     N4     N5    meanN345")
print(f"  {'baseline cls':<30}{'-':>5}   {fmt(per_noc_em(decode(te['cls'], oN), te['y'], te['noc']))}")
for nm, sv, st, log in [("phi_rerank (global phi)", PHv, PHt, True),
                        ("coverage (cross-locus)", COVv, COVt, False),
                        ("consistency (cross-locus)", CONSv, CONSt, False)]:
    a = tune(va["cls"], sv, va["y"], va["noc"], log); print(f"  {nm:<30}{a:>5.2f}   {fmt(per_noc_em(decode(lop(te['cls'], st, a, log), oN), te['y'], te['noc']))}")
# phi + consistency (two-channel): does adding consistency on top of phi help?
def lop2(L, s1, a1, s2, a2): return np.stack([z(L[i]) + a1 * z(np.log(s1[i] + 1e-6)) + a2 * z(s2[i]) for i in range(len(L))])
a1 = tune(va["cls"], PHv, va["y"], va["noc"], True)
bb, ba2 = -1, 0
for a2 in (0, .1, .2, .3, .5, .75, 1):
    em = per_noc_em(decode(lop2(va["cls"], PHv, a1, CONSv, a2), va["noc"].clip(1, 5)), va["y"], va["noc"]); v = np.mean([em.get(k, 0) for k in (3, 4, 5)])
    if v > bb: bb, ba2 = v, a2
print(f"  {'phi + consistency':<30}{a1:>2.1f}+{ba2:<2.1f} {fmt(per_noc_em(decode(lop2(te['cls'], PHt, a1, CONSt, ba2), oN), te['y'], te['noc']))}")
