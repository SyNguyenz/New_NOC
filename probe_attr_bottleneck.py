"""Decisive test of the user's causal chain (attr_head -> phi -> decoy -> rerank/NOC):
is attr_head the bottleneck, or is there an IDENTIFIABILITY floor BELOW attr that no phi-reasoner crosses?

For N5 samples the BEST method (symbolic phi_rerank) gets WRONG, ask: does OPTIMAL height-fit (NNLS over a
FIXED 5-set, per-locus-normalized to remove degradation) make the TRUE 5-set fit STRICTLY better than the
model's chosen (decoy-containing) 5-set?
  YES (true fits better in most errors) -> the separating signal EXISTS, our phi-reasoner just doesn't use it
       -> a better phi->attr reasoner (NN residual on EM) COULD help. Bottleneck = REASONING.
  NO  (residuals tie / true not better)  -> identifiability floor; no reasoner helps -> need NEW signal (cross-locus).
"""
import numpy as np, importlib.util
from pathlib import Path
try:
    from scipy.optimize import nnls; HAVE_NNLS = True
except Exception:
    HAVE_NNLS = False

PROJ = Path("."); DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"; CACHE = PROJ / "cache_insilico"; K = 45
def lm(n, p): s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pr = lm("pr", PROJ / "inc22_clean" / "phi_rerank.py")

def load(sp):
    d = {k: np.load(DATA / f"{k}_{sp}.npy") for k in ["tokens8", "mask", "phi", "noc"]}
    d["y"] = (d["phi"] > 0).astype(np.float32); d["cls"] = np.load(CACHE / f"cls_{sp}.npy")
    return d
va, te = load("val"), load("test")
dg = np.load(GENO / "donor_geno.npy").astype(np.float32); dgm = np.load(GENO / "donor_geno_mask.npy").astype(bool)
print(f"nnls={'scipy' if HAVE_NNLS else 'lstsq-clip'}; donor_geno {dg.shape}, test N5={int((te['noc']==5).sum())}")

# copy-number per candidate: cn[c][(locus, allele*10)] in {1,2}
cn = [dict() for _ in range(K)]
for c in range(min(K, dg.shape[0])):
    for j in range(dg.shape[1]):
        if dgm[c, j]:
            key = (int(round(dg[c, j, 0])), int(round(dg[c, j, 1] * 10)))
            cn[c][key] = cn[c].get(key, 0.0) + 1.0

def z(a): a = a.astype(np.float64); s = a.std(); return (a - a.mean()) / (s if s > 1e-9 else 1.0)
def oracle_em(scores, y, noc):
    per = {}
    for i in range(len(scores)):
        k = int(noc[i]); top = np.argsort(scores[i])[::-1][:k]; pred = np.zeros(K, bool); pred[top] = True
        per.setdefault(k, []).append(bool((pred == (y[i] > 0.5)).all()))
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}
def lop(L, sig, a):
    out = np.empty_like(L, np.float64)
    for i in range(len(L)): out[i] = z(L[i]) + a * z(np.log(sig[i] + 1e-6))
    return out
def tune(L, sig, y, noc, grid=(0, .1, .2, .3, .5, .75, 1, 1.5, 2)):
    bv, ba = -1, 0
    for a in grid:
        v = np.mean([oracle_em(lop(L, sig, a), y, noc).get(k, 0) for k in (3, 4, 5)])
        if v > bv: bv, ba = v, a
    return ba

SYv = pr.deconv_phi(va["tokens8"], va["mask"].astype(bool), dg, dgm)
SYt = pr.deconv_phi(te["tokens8"], te["mask"].astype(bool), dg, dgm)
alpha = tune(va["cls"], SYv, va["y"], va["noc"]); sc_te = lop(te["cls"], SYt, alpha)
print(f"symbolic rerank alpha={alpha}; test per-NOC EM:", {k: round(v, 3) for k, v in oracle_em(sc_te, te["y"], te["noc"]).items()})

def peaks(d, i):
    m = d["mask"][i].astype(bool); t = d["tokens8"][i][m]
    return np.round(t[:, 0]).astype(int), np.round(t[:, 1] * 10).astype(int), np.expm1(t[:, 2])
def fit_resid(loc, ab, h, donors, perlocus=True):
    if len(donors) == 0 or len(h) == 0: return 1.0
    A = np.zeros((len(loc), len(donors)))
    for ci, c in enumerate(donors):
        for p in range(len(loc)): A[p, ci] = cn[c].get((loc[p], ab[p]), 0.0)
    if A.sum() == 0: return 1.0
    if perlocus:
        ls = {}
        for p in range(len(loc)): ls[loc[p]] = ls.get(loc[p], 0.0) + h[p]
        w = np.array([1.0 / max(ls[loc[p]], 1e-6) for p in range(len(loc))])
        A = A * w[:, None]; h = h * w
    if HAVE_NNLS:
        phi, _ = nnls(A, h); r = np.linalg.norm(A @ phi - h)
    else:
        phi = np.clip(np.linalg.lstsq(A, h, rcond=None)[0], 0, None); r = np.linalg.norm(A @ phi - h)
    return r / (np.linalg.norm(h) + 1e-9)

for PL in (True, False):
    N5 = np.where(te["noc"] == 5)[0]; wins = ties = loss = nerr = 0; margins = []
    for i in N5:
        true = np.where(te["y"][i] > 0.5)[0]; pred = np.argsort(sc_te[i])[::-1][:5]
        if set(pred.tolist()) == set(true.tolist()): continue
        nerr += 1; loc, ab, h = peaks(te, i)
        rt = fit_resid(loc, ab, h, list(true), PL); rp = fit_resid(loc, ab, h, list(pred), PL)
        margins.append(rp - rt)
        if rt < rp - 1e-4: wins += 1
        elif rt > rp + 1e-4: loss += 1
        else: ties += 1
    print(f"\n=== {'per-locus-normalized' if PL else 'raw-height'} NNLS fit ===")
    print(f"N5={len(N5)}, symbolic-rerank ERRORS={nerr}")
    print(f"  TRUE set fits STRICTLY better : {wins} ({100*wins/max(nerr,1):.0f}%)  -> separating signal present")
    print(f"  PRED set fits better (true worse): {loss} ({100*loss/max(nerr,1):.0f}%)  -> identif. tie / true mis-fits")
    print(f"  tie                            : {ties} ({100*ties/max(nerr,1):.0f}%)")
    print(f"  median margin (resid_pred-resid_true; >0 = true better): {np.median(margins):.4f}; mean {np.mean(margins):.4f}")

# Exp B: is the "reasoning headroom" REALIZABLE? Use OPTIMAL-FIT (NNLS) phi over compatible candidates as a
# deployable rerank signal vs the crude uniform-compat symbolic phi. Does upgrading the phi-reasoner beat 0.901?
def nnls_phi(d):
    out = np.zeros((len(d["tokens8"]), K))
    for i in range(len(d["tokens8"])):
        loc, ab, h = peaks(d, i)
        if len(loc) == 0: continue
        comp = [c for c in range(K) if any((loc[p], ab[p]) in cn[c] for p in range(len(loc)))]
        if not comp: continue
        ls = {}
        for p in range(len(loc)): ls[loc[p]] = ls.get(loc[p], 0.0) + h[p]
        w = np.array([1.0 / max(ls[loc[p]], 1e-6) for p in range(len(loc))])
        A = np.zeros((len(loc), len(comp)))
        for ci, c in enumerate(comp):
            for p in range(len(loc)): A[p, ci] = cn[c].get((loc[p], ab[p]), 0.0)
        A = A * w[:, None]; hh = h * w
        phi = nnls(A, hh)[0] if HAVE_NNLS else np.clip(np.linalg.lstsq(A, hh, rcond=None)[0], 0, None)
        for ci, c in enumerate(comp): out[i, c] = phi[ci]
    return out
NPv, NPt = nnls_phi(va), nnls_phi(te)
aN = tune(va["cls"], NPv, va["y"], va["noc"]); emN = oracle_em(lop(te["cls"], NPt, aN), te["y"], te["noc"])
print(f"\n=== Exp B: NNLS-optimal-fit phi as rerank signal (alpha={aN}) ===")
print(f"  baseline cls          per-NOC EM:", {k: round(v, 3) for k, v in oracle_em(te['cls'], te['y'], te['noc']).items()})
print(f"  symbolic uniform-phi  per-NOC EM:", {k: round(v, 3) for k, v in oracle_em(sc_te, te['y'], te['noc']).items()})
print(f"  NNLS optimal-fit phi  per-NOC EM:", {k: round(v, 3) for k, v in emN.items()})
