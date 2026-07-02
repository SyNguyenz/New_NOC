"""
No-train probe: do per-donor *minimal Jumping Emerging Patterns* (JEP), mined offline from the
45-donor reference panel, detect faint minors in real-test mixtures with good SPECIFICITY,
WITHOUT training and WITHOUT privileged info?

Idea (user): a donor = a SET-OF-SETS. Each signature S is a minimal allele-combo that is
discriminative for the donor — a private single allele (order 1), or a combo whose individual
alleles may be shared but the combo is not owned by any other single donor (order 2/3).
Donor "fires" if ANY of its signatures is fully present in the observed peaks (OR-of-ANDs / DNF).

Grounding (paper-check):
  - minimal JEP = Dong&Li jumping emerging pattern (present in this class, absent in comparison);
    minimal = no proper subset is a JEP.  (Springer JEP; OCLEP+ minimal-length EP)
  - scoring = CAEP: aggregate fired signatures, weighted by discriminativeness (growth-rate analog;
    here = panel rarity of the alleles).  (Dong&Zhang CAEP)
  - differentiable realization later = Neural DNF / Multi-Prototype.

LITERATURE GAP = our risk: EP/CAEP discriminate vs a *single* comparison class. A mixture is a
SUPERPOSITION of <=5 donors, so a signature "private vs one other donor" can still be ASSEMBLED
from 2+ other contributors (explaining-away). This probe measures that directly as the FALSE-FIRE
rate on absent donors. Signal survives specificity -> real lever; specificity collapses -> won't train.

DEPLOYABILITY: observed peaks = ALL masked tokens (optionally height>=AT). Ground-truth (y, attr,
phi) is used ONLY to score/diagnose, NEVER to build features -> avoids the F34 at>=0 confound.
"""
import os, json, itertools
from pathlib import Path
import numpy as np

DATA = Path(os.environ.get("STR_DATA_DIR", "data_insilico_w"))
GENO = Path(os.environ.get("STR_GENO", "data/donor_geno.npy"))
MAX_ORDER = int(os.environ.get("JEP_MAX_ORDER", "3"))
TOPK = int(os.environ.get("JEP_TOPK", "60"))            # cap order-2 and order-3 sigs/donor (by rarity)
AT_SWEEP = [float(x) for x in os.environ.get("JEP_AT_SWEEP", "0,30,50").split(",")]

def abin(a): return int(round(float(a) * 10))
def key(locus, a): return (int(round(float(locus))), abin(a))

# ───────────────────────── panel ─────────────────────────
g  = np.load(GENO)
gm = np.load(str(GENO).replace(".npy", "_mask.npy")).astype(bool)
C  = g.shape[0]
donor_items = []
for c in range(C):
    s = set()
    for j in range(g.shape[1]):
        if gm[c, j]:
            s.add(key(g[c, j, 0], g[c, j, 1]))
    donor_items.append(s)

owners = {}
for c in range(C):
    for it in donor_items[c]:
        owners.setdefault(it, set()).add(c)
n_owners = {it: len(o) for it, o in owners.items()}
def rarity(it): return float(np.log(C / n_owners[it]))   # high = rare = discriminative

# ───────────────── mine minimal JEPs per donor (vs any SINGLE other donor) ─────────────────
def mine(c):
    A = sorted(donor_items[c])
    priv = [it for it in A if owners[it] == {c}]                 # order-1: private singletons
    sigs = [{"items": (it,), "order": 1, "w": rarity(it)} for it in priv]
    privset = set(priv)
    nonpriv = [it for it in A if it not in privset]
    if MAX_ORDER >= 2:
        cand = []
        for x, y in itertools.combinations(nonpriv, 2):
            if not ((owners[x] & owners[y]) - {c}):              # no OTHER single donor has both
                cand.append((x, y, rarity(x) + rarity(y)))
        cand.sort(key=lambda t: -t[2])
        for x, y, w in cand[:TOPK]:
            sigs.append({"items": (x, y), "order": 2, "w": w})
    if MAX_ORDER >= 3:
        cand = []
        for x, y, z in itertools.combinations(nonpriv, 3):
            # minimal: every 2-subset must still be ambiguous (not itself a 2-JEP)
            if not ((owners[x] & owners[y]) - {c}): continue
            if not ((owners[x] & owners[z]) - {c}): continue
            if not ((owners[y] & owners[z]) - {c}): continue
            if not ((owners[x] & owners[y] & owners[z]) - {c}):  # no OTHER single donor has all 3
                cand.append((x, y, z, rarity(x) + rarity(y) + rarity(z)))
        cand.sort(key=lambda t: -t[3])
        for x, y, z, w in cand[:TOPK]:
            sigs.append({"items": (x, y, z), "order": 3, "w": w})
    return sigs

JEP = [mine(c) for c in range(C)]
n1 = [sum(s["order"] == 1 for s in J) for J in JEP]
n2 = [sum(s["order"] == 2 for s in J) for J in JEP]
n3 = [sum(s["order"] == 3 for s in J) for J in JEP]
print("="*78)
print(f"PANEL: {C} donors | items universe={len(owners)} | mean alleles/donor={np.mean([len(s) for s in donor_items]):.1f}")
print(f"signatures/donor: order1 mean={np.mean(n1):.1f} (donors w/ 0 private={sum(x==0 for x in n1)}) "
      f"| order2 mean={np.mean(n2):.1f} | order3 mean={np.mean(n3):.1f}")
print("="*78)

# ───────────────────────── test set ─────────────────────────
tk  = np.load(DATA / "tokens8_test.npy")
mk  = np.load(DATA / "mask_test.npy").astype(bool)
y   = np.load(DATA / "y_test_set.npy").astype(bool)
noc = np.load(DATA / "noc_test.npy").astype(int)
phi = np.load(DATA / "phi_test.npy")
N   = tk.shape[0]

def observed_set(i, AT):
    v = mk[i]
    h = np.expm1(tk[i, :, 2])
    sel = np.where(v & (h >= AT))[0]
    return set(key(tk[i, k, 0], tk[i, k, 1]) for k in sel)

# GATE: do true contributors' panel alleles actually appear in observed peaks? (index/bin sanity)
ov = []
for i in range(N):
    O = observed_set(i, 0.0)
    for c in np.where(y[i])[0]:
        di = donor_items[c]
        if di: ov.append(len(di & O) / len(di))
print(f"[GATE] mean fraction of a true contributor's panel alleles present in observed peaks = {np.mean(ov):.3f}")
print(f"       (expect high; faint donors lower from dropout. <0.3 would mean an index/bin mismatch)\n")

def auc(pos, neg):
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    allv = np.concatenate([pos, neg]); order = allv.argsort()
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.cumsum(cnt); start = csum - cnt
    avgrank = (start + csum + 1) / 2.0
    ranks = avgrank[inv]
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))

# ───────────────────────── score (per AT) ─────────────────────────
for AT in AT_SWEEP:
    fired     = np.zeros((N, C), bool)
    score_sum = np.zeros((N, C))
    for i in range(N):
        O = observed_set(i, AT)
        for c in range(C):
            tot = 0.0; f = False
            for s in JEP[c]:
                if all(it in O for it in s["items"]):
                    f = True; tot += s["w"]
            fired[i, c] = f; score_sum[i, c] = tot

    print("#"*78)
    print(f"# OBSERVED = all masked tokens with height >= AT={AT:g} RFU")
    print("#"*78)
    header = f"{'NOC':>4} {'n_pres':>7} {'recall':>7} {'spec':>6} {'falsefire':>10} {'AUC':>6}   {'faint_recall(by phi)':>20}"
    print(header)
    for nv in [1, 2, 3, 4, 5]:
        idx = np.where(noc == nv)[0]
        if len(idx) == 0: continue
        pres_fire, abs_fire, pos_s, neg_s = [], [], [], []
        faint_hits, faint_tot = 0, 0
        for i in idx:
            pres = np.where(y[i])[0]; absent = np.where(~y[i])[0]
            for c in pres: pres_fire.append(fired[i, c]); pos_s.append(score_sum[i, c])
            for c in absent: abs_fire.append(fired[i, c]); neg_s.append(score_sum[i, c])
            if len(pres) > 0:
                faint = pres[np.argmin(phi[i, pres])]           # smallest proportion present donor
                faint_tot += 1; faint_hits += int(fired[i, faint])
        rec = np.mean(pres_fire); ff = np.mean(abs_fire)
        a = auc(pos_s, neg_s)
        frec = faint_hits / max(1, faint_tot)
        print(f"{nv:>4} {len(idx):>7} {rec:>7.3f} {1-ff:>6.3f} {ff:>10.3f} {a:>6.3f}   {frec:>20.3f}")

    # N5 faint-minor rank breakdown + privileged diagnostic
    if int(AT) == 0:
        idx5 = np.where(noc == 5)[0]
        print(f"\n  [N5 detail, AT=0]  recall by faintness rank (1=faintest .. 5=strongest):")
        ranks_hit = [[] for _ in range(5)]
        for i in idx5:
            pres = np.where(y[i])[0]
            order = pres[np.argsort(phi[i, pres])]               # faintest first
            for r, c in enumerate(order):
                if r < 5: ranks_hit[r].append(fired[i, c])
        print("     " + "  ".join(f"r{r+1}={np.mean(h):.3f}" for r, h in enumerate(ranks_hit) if h))
        # privileged diagnostic (label only): for MISSED faintest, how many private alleles present?
        attr = np.load(DATA / "attr_test.npy").astype(int)
        miss_priv, hit_priv = [], []
        for i in idx5:
            pres = np.where(y[i])[0]
            faint = pres[np.argmin(phi[i, pres])]
            O = observed_set(i, 0.0)
            npriv = sum(1 for it in donor_items[faint] if owners[it] == {faint} and it in O)
            (hit_priv if fired[i, faint] else miss_priv).append(npriv)
        print(f"  [diag, privileged] faintest-minor #private-alleles-present: "
              f"fired mean={np.mean(hit_priv) if hit_priv else float('nan'):.2f} (n={len(hit_priv)}) | "
              f"MISSED mean={np.mean(miss_priv) if miss_priv else float('nan'):.2f} (n={len(miss_priv)})")
    print()
