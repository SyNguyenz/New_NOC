"""(1) efm_rerank (EuroForMix-class: copy-number + back-stutter + degradation) vs phi_rerank (crude
uniform-compat) on in-silico test, ORACLE-NOC EM.
(2) The HIDDEN bottleneck: NOC determination. We've been reporting ORACLE-k EM (true count given). Deployed,
k is UNKNOWN. Measure present-set EM under oracle-k vs RF-count-k vs the user's sequential-residual-k, and
the gap. Uses the canonical clean model's cached cls (cache_insilico) — no dirty-model rebuild."""
import numpy as np, importlib.util
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
PROJ = Path("."); DATA = PROJ / "data_insilico_w"; GENO = PROJ / "data"; CACHE = PROJ / "cache_insilico"; K = 45
def lm(n, p): s = importlib.util.spec_from_file_location(n, str(p)); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
pr = lm("pr", PROJ / "inc22_clean" / "phi_rerank.py")
ef = lm("ef", PROJ / "efm_rerank.py")
g = np.load(GENO / "donor_geno.npy").astype(np.float32); gm = np.load(GENO / "donor_geno_mask.npy")

def load(sp):
    d = {k: np.load(DATA / f"{k}_{sp}.npy") for k in ["tokens8", "mask", "phi", "noc"]}
    d["y"] = (d["phi"] > 0).astype(np.float32); d["cls"] = np.load(CACHE / f"cls_{sp}.npy")
    szp = DATA / f"size_{sp}.npy"
    d["size"] = np.load(szp).astype(np.float64) if szp.exists() else np.zeros(d["mask"].shape, np.float64)
    return d
va, te = load("val"), load("test")
print(f"size channel: {'ON' if va['size'].any() else 'OFF (degr disabled)'}; test NOC dist",
      {int(k): int(v) for k, v in zip(*np.unique(te["noc"], return_counts=True))})

def z(a): a = a.astype(np.float64); s = a.std(); return (a - a.mean()) / (s if s > 1e-9 else 1.0)
def sig(x): return 1.0 / (1.0 + np.exp(-x))
def lop(L, s, a): return np.stack([z(L[i]) + a * z(np.log(s[i] + 1e-6)) for i in range(len(L))])
def decode(score, k):
    yp = np.zeros((len(score), K), int)
    for i in range(len(score)): yp[i, np.argsort(score[i])[::-1][:int(k[i])]] = 1
    return yp
def per_noc_em(yp, y, noc):
    yt = (y > 0.5).astype(int); per = {}
    for i in range(len(yp)):
        per.setdefault(int(noc[i]), []).append(bool((yp[i] == yt[i]).all()))
    return {k: float(np.mean(v)) for k, v in sorted(per.items())}
def tune(L, s, y, noc, grid=(0, .1, .2, .3, .5, .75, 1, 1.5, 2)):
    bv, ba = -1, 0
    for a in grid:
        em = per_noc_em(decode(lop(L, s, a), noc.clip(1, 5)), y, noc)
        v = np.mean([em.get(k, 0) for k in (3, 4, 5)])
        if v > bv: bv, ba = v, a
    return ba
def fmt(em): return "  ".join(f"{em.get(k,0):.3f}" for k in (1, 2, 3, 4, 5)) + f"   {np.mean([em.get(k,0) for k in (3,4,5)]):.3f}"

# ---------- (1) efm vs phi rerank, ORACLE-NOC ----------
PHv = pr.deconv_phi(va["tokens8"], va["mask"].astype(bool), g, gm, n_iters=12)
PHt = pr.deconv_phi(te["tokens8"], te["mask"].astype(bool), g, gm, n_iters=12)
EFv = ef.deconv_efm(va["tokens8"], va["mask"].astype(bool), va["size"], g, gm, xi=0.08, degr_lambda=0.0)
EFt = ef.deconv_efm(te["tokens8"], te["mask"].astype(bool), te["size"], g, gm, xi=0.08, degr_lambda=0.0)
oN = te["noc"].clip(1, 5)
print("\n=== (1) ORACLE-NOC EM: efm vs phi rerank (in-silico test) ===")
print(f"  {'method':<26}{'a':>5}   N1     N2     N3     N4     N5    meanN345")
print(f"  {'baseline cls':<26}{'-':>5}   {fmt(per_noc_em(decode(te['cls'], oN), te['y'], te['noc']))}")
for nm, sv, st in [("phi_rerank (uniform)", PHv, PHt), ("efm_rerank (cn+stutter)", EFv, EFt)]:
    a = tune(va["cls"], sv, va["y"], va["noc"]); sc = lop(te["cls"], st, a)
    print(f"  {nm:<26}{a:>5.2f}   {fmt(per_noc_em(decode(sc, oN), te['y'], te['noc']))}")
if va["size"].any():
    EFvd = ef.deconv_efm(va["tokens8"], va["mask"].astype(bool), va["size"], g, gm, xi=0.08, degr_lambda=0.003)
    EFtd = ef.deconv_efm(te["tokens8"], te["mask"].astype(bool), te["size"], g, gm, xi=0.08, degr_lambda=0.003)
    a = tune(va["cls"], EFvd, va["y"], va["noc"]); sc = lop(te["cls"], EFtd, a)
    print(f"  {'efm_rerank (+degr)':<26}{a:>5.2f}   {fmt(per_noc_em(decode(sc, oN), te['y'], te['noc']))}")

# ---------- (2) the HIDDEN bottleneck: NOC determination ----------
# best ranking = phi rerank (a tuned)
a = tune(va["cls"], PHv, va["y"], va["noc"]); scV = lop(va["cls"], PHv, a); scT = lop(te["cls"], PHt, a)

def rf_feats(L, P):
    s = np.sort(sig(L), 1)[:, ::-1][:, :15]; return np.concatenate([s, sig(L).sum(1, keepdims=True), (sig(L) > 0.5).sum(1, keepdims=True)], 1)
rf = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(rf_feats(va["cls"], None), va["noc"].clip(1, 5))
kRF = rf.predict(rf_feats(te["cls"], None))

# user's sequential-residual NOC on the canonical model: order by phi-rerank score; NOC = first k whose
# cumulative donor allele-coverage leaves residual peak-height < tau (tuned on val).
dset = [set() for _ in range(K)]; carr = {}
for c in range(K):
    for j in range(g.shape[1]):
        if gm[c, j]:
            it = (int(round(g[c, j, 0])), int(round(g[c, j, 1] * 10))); dset[c].add(it); carr.setdefault(it, []).append(c)
def resid_curve(d, score):
    N = len(d["tokens8"]); R = np.ones((N, 6))
    for i in range(N):
        m = d["mask"][i].astype(bool); t = d["tokens8"][i][m]
        pk = [((int(round(t[p, 0])), int(round(t[p, 1] * 10))), float(np.expm1(t[p, 2]))) for p in range(len(t))]
        pk = [(it, h) for it, h in pk if it in carr]
        tot = sum(h for _, h in pk)
        if tot <= 0: continue
        order = np.argsort(score[i])[::-1]; cov = set()
        for k in range(1, 6):
            cov |= dset[order[k - 1]]
            R[i, k] = sum(h for it, h in pk if it not in cov) / tot
    return R
Rv, Rt = resid_curve(va, scV), resid_curve(te, scT)
def seqcount(R, tau): return np.clip([next((k for k in range(1, 6) if R[i, k] < tau), 5) for i in range(len(R))], 1, 5)
bt, bv = 0.05, -1
for t in np.linspace(0.01, 0.30, 30):
    acc = (seqcount(Rv, t) == va["noc"].clip(1, 5)).mean()
    if acc > bv: bv, bt = acc, t
kSEQ = seqcount(Rt, bt)
kCARD = None  # (model card head not cached; RF + seq are the deployable comparators here)

tn = te["noc"].clip(1, 5)
print("\n=== (2) NOC determination accuracy (deployable, no oracle) ===")
print(f"  RF-count overall {np.mean(kRF==tn):.3f} | seq-residual(user, tau={bt:.3f}) overall {np.mean(kSEQ==tn):.3f}")
print(f"  {'trueNOC':>8} {'n':>5} {'RF':>7} {'seq':>7}")
for k in (1, 2, 3, 4, 5):
    s = tn == k
    print(f"  {k:>8} {int(s.sum()):>5} {np.mean(kRF[s]==k):>7.3f} {np.mean(kSEQ[s]==k):>7.3f}")

print("\n=== (2) present-set EM: ORACLE-k (hidden ceiling) vs deployable k ===")
print(f"  {'k-source':<16}   N1     N2     N3     N4     N5    meanN345")
for nm, kk in [("ORACLE-k", tn), ("RF-count-k", kRF), ("seq-residual-k", np.asarray(kSEQ))]:
    print(f"  {nm:<16}   {fmt(per_noc_em(decode(scT, kk), te['y'], te['noc']))}")
print("\n  mean residual curve (test, by true NOC) k1..k5:")
for k in (1, 3, 5):
    s = tn == k; print(f"   trueNOC={k}: " + "  ".join(f"k{j}={Rt[s,j].mean():.3f}" for j in range(1, 6)))

# ---------- (3) does the rerank GAIN survive deployable NOC? ranking x k-source ----------
a_efm = tune(va["cls"], EFv, va["y"], va["noc"])
rankings = [("cls (no rerank)", te["cls"]), ("phi_rerank", lop(te["cls"], PHt, a)), ("efm_rerank", lop(te["cls"], EFt, a_efm))]
print("\n=== (3) is the rerank gain REAL or an oracle-NOC artifact? (ranking x k-source) ===")
print(f"  {'':<28}N1     N2     N3     N4     N5    meanN345")
for ksrc, kk in [("ORACLE-k", tn), ("RF-count-k", kRF)]:
    print(f"  -- {ksrc} --")
    for nm, sc in rankings:
        print(f"     {nm:<23} {fmt(per_noc_em(decode(sc, kk), te['y'], te['noc']))}")
