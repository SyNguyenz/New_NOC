"""
make_insilico.py — Generate in-silico STR mixtures by OVERLAYING REAL single-source
profiles (not a peak model). Literature-grounded recipe:

  for each donor d with mixture proportion phi_d:
     ss   = random REAL single-source profile of d
     h    = expm1(ss)                  # -> RFU
     h    = h / h.sum()                # RELATIVE peak heights (Kelly/Bright: absolute
                                        #   height is template-dependent -> must normalize)
     h   *= gamma_jitter(per-peak)     # add variability (synthetic shows LESS than real)
     mix += phi_d * T_total * h        # weight by proportion (NIST: prop = sum h / total),
                                        #   shared alleles stack
  mix[mix < AT] = 0                     # detection threshold -> minor-contributor dropout
  -> log1p -> Xflat(590) + tokens(160,3) + mask + y(45) + noc

Two modes:
  python make_insilico.py                 # FIDELITY check vs real test combos (default)
  python make_insilico.py --build N        # generate N train mixtures -> data_insilico/

Split discipline: single-source from TRAIN only; real multi-person stays in val/test.
"""
from __future__ import annotations
import argparse, json, os, shutil
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
# Configurable for Kaggle/Colab: STR_DATA_DIR (input real data), STR_OUT_DIR (in-silico out)
DATA = Path(os.environ.get("STR_DATA_DIR", str(ROOT / "data")))
OUT = Path(os.environ.get("STR_OUT_DIR", str(ROOT / "data_insilico")))
META = json.load(open(DATA / "meta_set.json"))
KNOWN = META["known_donors"]
COL = {d: i for i, d in enumerate(KNOWN)}
FLAT_COLS = META["flat_cols"]
LOCUS_TO_IDX = META["locus_to_idx"]
MAX_SEQ = META["max_seq"]
N_FLAT = META["n_flat"]

# Precompute per-bin (locus_idx, allele_val) for token reconstruction
BIN_LOCUS = np.zeros(N_FLAT, dtype=np.int64)
BIN_ALLELE = np.zeros(N_FLAT, dtype=np.float32)
for j, col in enumerate(FLAT_COLS):
    locus, allele = col.split("_", 1)
    BIN_LOCUS[j] = LOCUS_TO_IDX[locus]
    BIN_ALLELE[j] = -2.0 if allele == "X" else (-1.0 if allele == "Y" else float(allele))


def build_bin_size():
    """Per-FLAT-bin fragment size (bp), median over real peaks. size(bp) is ~deterministic per
    (locus, allele) — gives in-silico mixtures a realistic size channel (degradation substrate,
    Increment 2a). Needs data/size_train.npy (extract_size.py). Returns (N_FLAT,) float32 or None."""
    sp = DATA / "size_train.npy"
    if not sp.exists():
        return None
    size = np.load(sp); tok = np.load(DATA / "tokens_train.npy"); mk = np.load(DATA / "mask_train.npy").astype(bool)
    bin_of = {(int(BIN_LOCUS[j]), round(float(BIN_ALLELE[j]), 1)): j for j in range(N_FLAT)}
    acc = [[] for _ in range(N_FLAT)]
    v = mk & (size > 0)
    li = tok[:, :, 0][v].astype(int); al = np.round(tok[:, :, 1][v], 1); sz = size[v]
    for a, b, s in zip(li, al, sz):
        j = bin_of.get((int(a), float(b)))
        if j is not None:
            acc[j].append(s)
    out = np.zeros(N_FLAT, np.float32)
    for j in range(N_FLAT):
        if acc[j]:
            out[j] = float(np.median(acc[j]))
    # fallback for unseen bins: per-locus mean size
    for L in range(int(BIN_LOCUS.max()) + 1):
        idx = np.where(BIN_LOCUS == L)[0]
        seen = [out[j] for j in idx if out[j] > 0]
        if seen:
            m = float(np.mean(seen))
            for j in idx:
                if out[j] == 0:
                    out[j] = m
    return out


# Real val/test combos (donor IDs) — for fidelity check / version-A generation and, critically, for
# EXCLUDING them from the in-silico train (build_train). READ FROM meta_set.json so they follow the
# donor fold: prepare_data_set/preprocess pick val/test combos from whatever donors are KNOWN in that
# fold, so hard-coding them would recreate the fold's real val/test combos inside the synthetic train.
# The fallbacks are the fold-0 literals, so a meta written before split_policy existed still works.
def _combos_from_meta(split, fallback):
    mp = META.get("split_policy", {}).get("multi_person_combos", {}).get(split)
    if mp is None:                      # pre-split_policy meta -> fall back to the fold-0 literals
        return fallback                 # NOTE: `{}` is a real answer (no combo in that split), not missing
    return {int(k.replace("NOC", "")): [tuple(sorted(int(d) for d in c)) for c in combos]
            for k, combos in mp.items()}


TEST_COMBOS = _combos_from_meta(
    "test", {2: [(31, 32)], 3: [(46, 47, 48)], 4: [(33, 34, 35, 36)], 5: [(31, 32, 33, 34, 35)]})
VAL_COMBOS = _combos_from_meta(
    "val", {2: [(33, 34)], 3: [(41, 42, 43)], 4: [(32, 33, 34, 35)], 5: [(35, 36, 37, 38, 39)]})

GAMMA_SHAPE = 10.0   # per-peak jitter CV ~ 1/sqrt(shape) ~ 0.32
# AT: real test (PROVEDIt) retains peaks to ~10 RFU (p10=10); the old 14 DROPPED real's faint minor tail
# (measure_gen_fidelity.py: synth p10 23-26 vs real 10). Lower to match THIS dataset's effective threshold
# (typical lab AT 50-150, but this research data is lower). Grounded by the real data's own faint tail.
AT = float(os.environ.get("STR_AT", "10.0"))   # analytical/detection threshold (RFU); peak-model is UNDER-occupied at
#   AT=14, so AT=10 closes BOTH the faint tail (real p10=10) and occupancy (->5.7~5.74). (Overlay over-occupied at low AT;
#   the peak model doesn't because it generates the faint tail physically rather than retaining spurious overlay peaks.)

# ── Back-stutter model (grounded; the overlay LOSES minor stutter in mixing -> synth 0.09 vs real 0.14). ──
# Physically: mixture back-stutter at allele a-1 = SR(a) * (summed parent height at a), SR linear in LUS
# (LUS~allele for simple repeats; NGM-SELect coeffs slope~0.007-0.011, intercept~-0.03..-0.058; SR~5-15%,
# log-normal noise). Applied to the SUMMED mixture (correct: total stutter = SR x total parent). NO double-count
# guard = re-measured final stutter must MATCH real (~0.136), not exceed it. Toggle STR_STUTTER=0 to disable.
STUTTER = int(os.environ.get("STR_STUTTER", "1"))
SR_SLOPE = 0.0066; SR_INTERCEPT = -0.040; SR_SIGMA = 0.30   # SR=clip(slope*allele+intercept, .01,.18)*lognormal(0,sigma); calibrated so final stutter~real 0.136 (not over)
_BININDEX = {(int(BIN_LOCUS[j]), round(float(BIN_ALLELE[j]), 1)): j for j in range(N_FLAT)}
STUTTER_TARGET = np.full(N_FLAT, -1, dtype=np.int64)   # bin at (same locus, allele-1 repeat) where back-stutter lands
for j in range(N_FLAT):
    a = float(BIN_ALLELE[j])
    if a >= 0:                                          # skip X(-2)/Y(-1) amel
        STUTTER_TARGET[j] = _BININDEX.get((int(BIN_LOCUS[j]), round(a - 1.0, 1)), -1)

# ── Drop-in (sporadic contamination/artifact alleles; Gill et al. 2000; Balding & Buckleton 2009) ──
# Poisson(lambda) PHANTOM alleles per profile at random allele bins, low height ~ AT + Exponential (the faint
# artifact regime), capped (drop-in is low-template by definition). No contributor -> contrib=0 -> attr=-1,
# exactly like stutter. Absent from the original generator (a realism gap vs real PROVEDIt). Toggle STR_DROPIN.
DROPIN = int(os.environ.get("STR_DROPIN", "0"))   # OPT-IN (default off) so inc22 stays bit-identical; LUPI sets it on
DI_LAMBDA = float(os.environ.get("STR_DI_LAMBDA", "0.8"))    # mean # drop-in alleles per profile (drop-in is rare)
DI_HMEAN = float(os.environ.get("STR_DI_HMEAN", "75.0"))     # exponential mean of drop-in height ABOVE AT
DI_HMAX = float(os.environ.get("STR_DI_HMAX", "350.0"))      # cap (drop-in is low-template)
DI_BINS = np.where(BIN_ALLELE >= 0)[0]                        # real allele positions (exclude amelogenin X/Y)

def add_dropin(mix, rng):
    """Sporadic drop-in (Balding & Buckleton 2009; Gill et al.): Poisson(# alleles), each a random allele bin
    with a low height ~ AT + Exponential, capped. Phantom (no contributor) -> attr=-1 downstream (like stutter)."""
    n_di = int(rng.poisson(DI_LAMBDA))
    if n_di > 0 and len(DI_BINS):
        js = rng.choice(DI_BINS, size=n_di, replace=True)
        h = np.minimum(AT + rng.exponential(DI_HMEAN, size=n_di), DI_HMAX)
        np.add.at(mix, js, h)
    return mix

# ── PEAK MODEL (EuroForMix-style, simplified+fast) — replaces overlay to reproduce real's faint tail. ──
# Overlay normalizes each real SS to relative heights -> DISCARDS the per-donor absolute structure -> can't
# render real's p10=10 faint tail. Peak model generates DIRECTLY from genotype: per (donor,allele) expected
# height mu = T*phi*dosage*deg(size); observed ~ Gamma(mean=mu, CV=PH_CV). The Gamma's natural low tail +
# degradation give the real heteroscedastic faint distribution. No SS pool needed (simpler+faster than overlay).
PH_CV = float(os.environ.get("STR_PH_CV", "0.60"))        # peak-height coeff of variation (gamma heteroscedasticity)
DEG_BETA_MAX = float(os.environ.get("STR_DEG", "0.0040")) # degradation slope/bp: deg=exp(-beta*(size-100)), beta~U(0,max) per mixture
# CALIBRATED to real test N5 (measure_gen_fidelity.py): median 137~132, p90~1000, occ 5.7~5.74, decoy structure matched.
PEAK_TMU = np.log(float(os.environ.get("STR_PEAK_T", "750.0"))); PEAK_TSIG = float(os.environ.get("STR_PEAK_TSIG", "1.10"))  # per-mixture template scale (lognormal)

def build_donor_dosage():
    """(45, N_FLAT) copy-number per donor per flat bin, from donor_geno (homozygous -> dosage 2 if 2 rows)."""
    gp = DATA / "donor_geno.npy"; gmp = DATA / "donor_geno_mask.npy"
    if not gp.exists(): return None
    g = np.load(gp); gm = np.load(gmp); dos = np.zeros((45, N_FLAT), np.float32)
    for c in range(45):
        for j in range(g.shape[1]):
            if gm[c, j]:
                bj = _BININDEX.get((int(g[c, j, 0]), round(float(g[c, j, 1]), 1)))
                if bj is not None: dos[c, bj] += 1.0
    return dos
DONOR_DOSAGE = build_donor_dosage()

def build_locus_eff():
    """Per-locus amplification efficiency from REAL single-source (mean-1). Real STR kits vary strongly per locus;
    the uniform-T model lacks this -> it's the biggest missing variance source widening the real height tails."""
    nL = int(BIN_LOCUS.max()) + 1
    try:
        X = np.load(DATA / "Xflat_train.npy").astype(np.float32); n = np.load(DATA / "noc_train.npy")
        H = np.expm1(X[n == 1])
    except Exception:
        return np.ones(nL, np.float32)
    eff = np.ones(nL, np.float32)
    for L in range(nL):
        v = H[:, np.where(BIN_LOCUS == L)[0]]; v = v[v > 0]
        if len(v): eff[L] = float(np.median(v))
    return (eff / np.median(eff)).astype(np.float32)        # median-1: keeps overall scale, high/low loci add the spread
LOCUS_EFF = build_locus_eff()
EFF_BIN = LOCUS_EFF[BIN_LOCUS]                              # (N_FLAT,) per-bin locus efficiency multiplier

# ── Realistic-proportion mode (root-cause fix for the synth->real N5 gap) ──────
# Diagnosis (debug_gen_*.py, 2026-06-13): the OLD proportion model r_max=6+5*(k-2)
# over-skews high-NOC mixtures — at N5 it gives min-phi med 0.054 / ratio med 8.1,
# vs REAL test min-phi 0.125 / ratio 4.0, so synthetic minor peaks are ~2x too faint
# (median 68 vs 155 rfu). log_h is 58% of the synth-vs-real domain-classifier signal.
# A readout re-fit on REAL H reaches N5 oracle 0.94 vs the synth-trained 0.79 -> the
# encoded info is there; the wall is this proportion/template DISTRIBUTION shift.
# PROP_MODE picks the mixture-proportion / template model (3-way generator experiment 2026-06-13):
#   "original"  : r_max=6+5*(k-2) skew (the shipped generator; medium->very-hard, OMITS balanced).
#   "realistic" : Dirichlet (min-phi~0.125) + louder/wider template + more jitter — MATCHES real N5
#                 (validated domain AUC(height) 0.70->0.57). NARROW: risks overfitting this test set.
#   "wide"      : SUPERSET = per-mixture coin-flip between original-skew and realistic-balanced, so the
#                 training span covers BOTH extremes (ratio 1..21) and CONTAINS real as a dense
#                 sub-region — the "easy->hard, beyond reality" curriculum done right (the bug was that
#                 "original" omitted the easy/balanced 40% of real: real min-phi 0.125 = orig's 97th pct).
PROP_MODE = os.environ.get("STR_PROP_MODE", "realistic" if int(os.environ.get("STR_REALISTIC", "0")) else "original")
REAL_TTOTAL_MU = np.log(40000.0); REAL_TTOTAL_SIGMA = 1.05  # heavier tail: match real height p90 (1346 vs old 903);
#   real heteroscedasticity is WIDER both ends (taller majors + fainter minors). sigma 0.85->1.05 widens across-mixture
#   template spread; taller template also lifts minor STUTTER above AT (helps the stutter gap without an explicit model).
REAL_DIRICHLET_ALPHA = 4.0                                  # concentration -> min-phi~0.12, ratio~4 like real
REAL_EQUAL_FRAC = 0.15                                      # fraction of perfectly balanced mixtures (real has these)
REAL_GAMMA_SHAPE = 6.0                                      # lower shape -> more per-peak variability (real-like)
WIDE_REAL_FRAC = 0.5                                        # wide mode: fraction drawn from the realistic regime


def build_ss_pool():
    """donor_col -> list of single-source Xflat rows (log1p)."""
    X = np.load(DATA / "Xflat_train.npy").astype(np.float32)
    y = np.load(DATA / "y_train_set.npy")
    n = np.load(DATA / "noc_train.npy")
    ss = n == 1
    Xs, ds = X[ss], y[ss].argmax(1)
    pool = {c: Xs[ds == c] for c in range(45)}
    return pool


def gen_mixture(donor_cols, pool, rng, t_total=None, mode=None):
    k = len(donor_cols)
    mode = PROP_MODE if mode is None else mode
    # "wide" superset = per-mixture coin-flip between the realistic and the original regimes.
    regime = mode
    if mode == "wide":
        regime = "realistic" if rng.random() < WIDE_REAL_FRAC else "original"
    if regime == "realistic":
        # real-matched proportions: Dirichlet (balanced, min-phi~0.125 like real test N5),
        # with a fraction of perfectly-equal mixtures; louder/wider template; more jitter.
        if rng.random() < REAL_EQUAL_FRAC:
            phi = np.full(k, 1.0 / k)
        else:
            phi = rng.dirichlet(np.full(k, REAL_DIRICHLET_ALPHA))
        if t_total is None:
            t_total = float(np.exp(rng.normal(REAL_TTOTAL_MU, REAL_TTOTAL_SIGMA)))
        gshape = REAL_GAMMA_SHAPE
    else:
        # ORIGINAL: ratio skew grows with k (higher NOC -> more extreme minors -> dropout)
        r_max = 6.0 + 5.0 * (k - 2)
        w = np.exp(rng.uniform(0, np.log(r_max), k)); phi = w / w.sum()
        if t_total is None:
            t_total = float(np.exp(rng.normal(np.log(32000), 0.55)))
        gshape = GAMMA_SHAPE
    mix = np.zeros(N_FLAT, dtype=np.float64)
    contrib = np.zeros((k, N_FLAT), dtype=np.float64)        # per-donor RFU contribution per bin
    for di, (c, p) in enumerate(zip(donor_cols, phi)):
        prof = pool[c][rng.integers(len(pool[c]))]
        h = np.expm1(prof.astype(np.float64))
        s = h.sum()
        if s <= 0:
            continue
        h = h / s
        jit = rng.gamma(gshape, 1.0 / gshape, size=N_FLAT)
        h = h * jit
        contrib[di] = p * t_total * h
        mix += contrib[di]
    if STUTTER:                                              # back-stutter on the SUMMED mixture (a-1 repeat); SR(a)*parent
        parent = mix.copy()
        js = np.where((STUTTER_TARGET >= 0) & (parent > 0))[0]
        if len(js):
            a = BIN_ALLELE[js].astype(np.float64)
            sr = np.clip(SR_SLOPE * a + SR_INTERCEPT, 0.01, 0.18) * rng.lognormal(0.0, SR_SIGMA, size=len(js))
            np.add.at(mix, STUTTER_TARGET[js], sr * parent[js])   # stutter peaks: no donor attribution (contrib=0 -> attr=-1)
    if DROPIN:                                              # sporadic phantom alleles (no contributor -> attr=-1)
        add_dropin(mix, rng)
    drop = mix < AT                                          # below detection -> minor dropout
    mix[drop] = 0.0
    xflat = np.log1p(mix).astype(np.float32)
    y = np.zeros(45, dtype=np.float32)
    phi45 = np.zeros(45, dtype=np.float32)                   # PRIVILEGED: mixture proportion target (Inc2 §5)
    for c, p in zip(donor_cols, phi):
        y[c] = 1.0
        phi45[c] = p
    # PRIVILEGED: per-bin dominant contributor (donor COLUMN 0..44), -1 where bin empty.
    # Supervises allele->donor attribution (Inc2 §5): which donor each observed peak came from.
    attr_bin = np.full(N_FLAT, -1, dtype=np.int16)
    if k > 0:
        live = (~drop) & (contrib.max(0) > 0)
        donor_col_arr = np.asarray(donor_cols, dtype=np.int16)
        attr_bin[live] = donor_col_arr[contrib.argmax(0)[live]]
    return xflat, y, k, t_total, attr_bin, phi45


def _proportions(k, rng, mode):
    regime = "realistic" if (mode == "wide" and rng.random() < WIDE_REAL_FRAC) else mode
    if regime == "realistic":
        return np.full(k, 1.0/k) if rng.random() < REAL_EQUAL_FRAC else rng.dirichlet(np.full(k, REAL_DIRICHLET_ALPHA))
    r_max = 6.0 + 5.0 * (k - 2); w = np.exp(rng.uniform(0, np.log(r_max), k)); return w / w.sum()

def gen_mixture_peak(donor_cols, rng, bin_size, mode="realistic"):
    """EuroForMix-style direct peak generation (no overlay). Same return signature as gen_mixture."""
    k = len(donor_cols); phi = _proportions(k, rng, mode)
    T = float(np.exp(rng.normal(PEAK_TMU, PEAK_TSIG)))
    beta = rng.uniform(0, DEG_BETA_MAX)
    size = bin_size if bin_size is not None else np.zeros(N_FLAT, np.float64)
    deg = np.exp(-beta * np.maximum(size - 100.0, 0.0))    # fragment-size degradation
    shape = 1.0 / PH_CV**2
    contrib = np.zeros((k, N_FLAT), dtype=np.float64)
    for di, (c, p) in enumerate(zip(donor_cols, phi)):
        mu = T * p * DONOR_DOSAGE[c] * deg * EFF_BIN        # expected height per bin (0 where donor has no allele); EFF_BIN = per-locus efficiency
        nz = np.where(mu > 0)[0]
        if len(nz): contrib[di, nz] = rng.gamma(shape, mu[nz] * PH_CV**2)   # Gamma(mean=mu, CV=PH_CV)
    mix = contrib.sum(0)
    if STUTTER:                                            # back-stutter on the summed mixture (same model as overlay)
        parent = mix.copy(); js = np.where((STUTTER_TARGET >= 0) & (parent > 0))[0]
        if len(js):
            a = BIN_ALLELE[js].astype(np.float64)
            sr = np.clip(SR_SLOPE * a + SR_INTERCEPT, 0.01, 0.18) * rng.lognormal(0.0, SR_SIGMA, size=len(js))
            np.add.at(mix, STUTTER_TARGET[js], sr * parent[js])
    if DROPIN: add_dropin(mix, rng)                        # sporadic phantom alleles (no contributor -> attr=-1)
    drop = mix < AT; mix[drop] = 0.0
    xflat = np.log1p(mix).astype(np.float32)
    y = np.zeros(45, np.float32); phi45 = np.zeros(45, np.float32)
    for c, p in zip(donor_cols, phi): y[c] = 1.0; phi45[c] = p
    attr_bin = np.full(N_FLAT, -1, dtype=np.int16)
    live = (~drop) & (contrib.max(0) > 0)
    dc = np.asarray(donor_cols, dtype=np.int16); attr_bin[live] = dc[contrib.argmax(0)[live]]
    return xflat, y, k, T, attr_bin, phi45

def gen_mixture_peak_labeled(donor_cols, rng, bin_size, mode="realistic"):
    """gen_mixture_peak + PRIVILEGED physical labels for LUPI (Inc-LUPI):
      beta      : per-mixture degradation slope (scalar)
      mu_bin    : CLEAN noise-free expected total height per bin (denoising target)
      var_bin   : heteroscedastic gamma variance per bin = (sum_c mu_c^2)*CV^2
      dropin_bin: bool per bin = a drop-in landed here (phantom artifact label)
    Faithful copy of gen_mixture_peak's draws so the saved (tokens, attr, phi) match it byte-for-byte."""
    k = len(donor_cols); phi = _proportions(k, rng, mode)
    T = float(np.exp(rng.normal(PEAK_TMU, PEAK_TSIG)))
    beta = float(rng.uniform(0, DEG_BETA_MAX))
    size = bin_size if bin_size is not None else np.zeros(N_FLAT, np.float64)
    deg = np.exp(-beta * np.maximum(size - 100.0, 0.0))
    shape = 1.0 / PH_CV**2
    contrib = np.zeros((k, N_FLAT), dtype=np.float64)
    mu_bin = np.zeros(N_FLAT, np.float64); mu2_bin = np.zeros(N_FLAT, np.float64)
    for di, (c, p) in enumerate(zip(donor_cols, phi)):
        mu = T * p * DONOR_DOSAGE[c] * deg * EFF_BIN
        mu_bin += mu; mu2_bin += mu**2                      # clean expected (sum) and sum-of-squares (for var)
        nz = np.where(mu > 0)[0]
        if len(nz): contrib[di, nz] = rng.gamma(shape, mu[nz] * PH_CV**2)
    mix = contrib.sum(0)
    if STUTTER:
        parent = mix.copy(); js = np.where((STUTTER_TARGET >= 0) & (parent > 0))[0]
        if len(js):
            a = BIN_ALLELE[js].astype(np.float64)
            sr = np.clip(SR_SLOPE * a + SR_INTERCEPT, 0.01, 0.18) * rng.lognormal(0.0, SR_SIGMA, size=len(js))
            np.add.at(mix, STUTTER_TARGET[js], sr * parent[js])
    dropin_bin = np.zeros(N_FLAT, bool)
    if DROPIN:
        before = mix.copy(); add_dropin(mix, rng)
        dropin_bin = ((mix - before) > 1e-9) & (contrib.max(0) == 0)   # PURE phantom drop-in (exclude contributor-overlap; data-check fix)
    drop = mix < AT; mix[drop] = 0.0
    xflat = np.log1p(mix).astype(np.float32)
    y = np.zeros(45, np.float32); phi45 = np.zeros(45, np.float32)
    for c, p in zip(donor_cols, phi): y[c] = 1.0; phi45[c] = p
    attr_bin = np.full(N_FLAT, -1, dtype=np.int16)
    live = (~drop) & (contrib.max(0) > 0)
    dc = np.asarray(donor_cols, dtype=np.int16); attr_bin[live] = dc[contrib.argmax(0)[live]]
    var_bin = mu2_bin * PH_CV**2
    return xflat, y, k, beta, attr_bin, phi45, mu_bin.astype(np.float32), var_bin.astype(np.float32), dropin_bin

PEAK_MODEL = int(os.environ.get("STR_PEAK_MODEL", "1"))   # 1 = EuroForMix-style peak model (default), 0 = legacy overlay
def gen_any(cols, pool, rng, bin_size, mode=None):
    """Dispatch: peak model (direct from genotype, reproduces faint tail) or legacy overlay."""
    if PEAK_MODEL and DONOR_DOSAGE is not None:
        return gen_mixture_peak(cols, rng, bin_size, mode=mode or PROP_MODE)
    return gen_mixture(cols, pool, rng, mode=mode)

def xflat_to_tokens(xflat, attr_bin=None, bin_size=None):
    nz = np.where(xflat > 0)[0]
    if len(nz) > MAX_SEQ:
        nz = nz[np.argsort(xflat[nz])[::-1][:MAX_SEQ]]   # keep tallest
    tok = np.zeros((MAX_SEQ, 3), dtype=np.float32)
    mask = np.zeros(MAX_SEQ, dtype=bool)
    attr = np.full(MAX_SEQ, -1, dtype=np.int16)          # per-peak dominant donor (-1 = pad)
    size = np.zeros(MAX_SEQ, dtype=np.float32)           # per-peak fragment size (bp)
    for i, j in enumerate(nz):
        tok[i] = [BIN_LOCUS[j], BIN_ALLELE[j], xflat[j]]
        mask[i] = True
        if attr_bin is not None:
            attr[i] = attr_bin[j]
        if bin_size is not None:
            size[i] = bin_size[j]
    return tok, mask, attr, size


def feat_summary(X, noc, tag):
    """per-NOC: total RFU, n active bins, MAC."""
    rfu = np.expm1(X)
    print(f"  {tag}:")
    for k in sorted(set(noc)):
        m = noc == k
        if not m.sum():
            continue
        # MAC: max alleles per locus (count bins>0 grouped by BIN_LOCUS)
        macs = []
        for i in np.where(m)[0][:300]:
            active = X[i] > 0
            cnt = np.bincount(BIN_LOCUS[active], minlength=24)
            macs.append(cnt.max() if cnt.size else 0)
        print(f"    NOC={k}: totalRFU={np.median(rfu[m].sum(1)):.0f}  "
              f"activeBins={np.median((X[m] > 0).sum(1)):.0f}  MAC={np.median(macs):.1f}  (n={m.sum()})")


def fidelity_check(pool, rng):
    """Generate in-silico for the real TEST combos; compare distributions to real test."""
    Xte = np.load(DATA / "Xflat_test.npy"); nte = np.load(DATA / "noc_test.npy")
    print("=== FIDELITY: in-silico vs REAL (same test combos) ===")
    feat_summary(Xte, nte, "REAL test")
    bin_size = build_bin_size()
    # generate in-silico matching each test combo
    Xs, ns = [], []
    for k, combos in TEST_COMBOS.items():
        n_real = int((nte == k).sum())
        for _ in range(n_real):
            combo = combos[rng.integers(len(combos))]
            cols = [COL[d] for d in combo]
            xf, y, kk, _, _, _ = gen_any(cols, pool, rng, bin_size)
            Xs.append(xf); ns.append(kk)
    Xs = np.array(Xs); ns = np.array(ns)
    feat_summary(Xs, ns, "IN-SILICO test-combo")
    print(f"\n  (tune GAMMA_SHAPE/AT/T_total if distributions diverge)")


def build_train(n_mix, pool, rng, exclude_test=True, oversample_test=0, noc_p=None):
    OUT.mkdir(exist_ok=True)
    test_set = {tuple(sorted(c)) for cs in TEST_COMBOS.values() for c in cs}
    val_set = {tuple(sorted(c)) for cs in VAL_COMBOS.values() for c in cs}
    print(f"  excluding real combos from in-silico train — test={sorted(test_set)} val={sorted(val_set)}")
    # keep real single-source train too
    Xtr = np.load(DATA / "Xflat_train.npy"); ytr = np.load(DATA / "y_train_set.npy"); ntr = np.load(DATA / "noc_train.npy")
    ss = ntr == 1
    Xf_list = [Xtr[ss]]; y_list = [ytr[ss]]; noc_list = [ntr[ss]]
    tok_real = np.load(DATA / "tokens_train.npy")[ss]; mask_real = np.load(DATA / "mask_train.npy")[ss]
    tok_list = [tok_real]; mask_list = [mask_real]
    # PRIVILEGED labels for real single-source: every valid peak belongs to that one donor;
    # phi is one-hot on that donor. (Inc2 §5 — provenance is known for in-silico/SS, absent for real mixtures.)
    d_ss = ytr[ss].argmax(1).astype(np.int16)
    attr_real = np.full(mask_real.shape, -1, dtype=np.int16)
    for r in range(len(d_ss)):
        attr_real[r, mask_real[r].astype(bool)] = d_ss[r]
    phi_real = np.zeros((len(d_ss), 45), np.float32); phi_real[np.arange(len(d_ss)), d_ss] = 1.0
    attr_list = [attr_real]; phi_list = [phi_real]
    # Increment 2a: per-peak size(bp). Real SS rows carry their real size (extract_size.py); in-silico
    # peaks get size from the per-bin table (size ~deterministic per locus,allele).
    bin_size = build_bin_size()
    sp = DATA / "size_train.npy"
    size_real = np.load(sp)[ss] if sp.exists() else np.zeros_like(mask_real, np.float32)
    size_list = [size_real]; size_new = []
    print(f"Keeping {ss.sum()} real single-source. Generating {n_mix} in-silico mixtures..."
          f" (size channel: {'ON' if bin_size is not None else 'OFF — run extract_size.py'})")
    Xf_new, y_new, noc_new, tok_new, mask_new, attr_new, phi_new = [], [], [], [], [], [], []
    # TRUE recreate-leak: oversample SYNTHETIC test combos (built from single-source).
    # Real multi-person stays held out in test; only the combo STRUCTURE is recreated.
    if oversample_test > 0:
        print(f"  OVERSAMPLING {oversample_test}x each test combo (synthetic, from single-source)")
        for cs in TEST_COMBOS.values():
            for combo in cs:
                cols = [COL[d] for d in combo]
                for _ in range(oversample_test):
                    xf, y, kk, _, ab, ph = gen_any(cols, pool, rng, bin_size)
                    tok, mask, attr, size = xflat_to_tokens(xf, ab, bin_size)
                    Xf_new.append(xf); y_new.append(y); noc_new.append(kk); tok_new.append(tok)
                    mask_new.append(mask); attr_new.append(attr); phi_new.append(ph); size_new.append(size)
    made = 0
    while made < n_mix:
        k = int(rng.choice([2,3,4,5], p=noc_p)) if noc_p is not None else int(rng.integers(2, 6))
        cols = sorted(rng.choice(45, size=k, replace=False).tolist())
        ids = tuple(sorted(KNOWN[c] for c in cols))
        if exclude_test and (ids in test_set or ids in val_set):
            continue
        xf, y, kk, _, ab, ph = gen_any(cols, pool, rng, bin_size)
        tok, mask, attr, size = xflat_to_tokens(xf, ab, bin_size)
        Xf_new.append(xf); y_new.append(y); noc_new.append(kk); tok_new.append(tok)
        mask_new.append(mask); attr_new.append(attr); phi_new.append(ph); size_new.append(size)
        made += 1
    Xf = np.concatenate([np.array(Xf_new)] + Xf_list).astype(np.float32)
    Y = np.concatenate([np.array(y_new)] + y_list).astype(np.float32)
    NOC = np.concatenate([np.array(noc_new)] + noc_list).astype(np.int32)
    TOK = np.concatenate([np.array(tok_new)] + tok_list).astype(np.float32)
    MASK = np.concatenate([np.array(mask_new)] + mask_list).astype(bool)
    ATTR = np.concatenate([np.array(attr_new, dtype=np.int16)] + attr_list).astype(np.int16)
    PHI = np.concatenate([np.array(phi_new, dtype=np.float32)] + phi_list).astype(np.float32)
    SIZE = np.concatenate([np.array(size_new, dtype=np.float32)] + size_list).astype(np.float32)
    perm = rng.permutation(len(Xf))
    np.save(OUT / "Xflat_train.npy", Xf[perm]); np.save(OUT / "y_train_set.npy", Y[perm])
    np.save(OUT / "noc_train.npy", NOC[perm]); np.save(OUT / "tokens_train.npy", TOK[perm]); np.save(OUT / "mask_train.npy", MASK[perm])
    np.save(OUT / "attr_train.npy", ATTR[perm]); np.save(OUT / "phi_train.npy", PHI[perm])
    np.save(OUT / "size_train.npy", SIZE[perm])
    print(f"  privileged labels: attr_train {ATTR.shape}, phi_train {PHI.shape}, size_train {SIZE.shape}")
    # copy real val/test/open + meta. phi_/condition_ (real NOMINAL mixture proportion +
    # DNA condition, from extract_phi_condition.py) are carried so the Inc2 phi head can be
    # EVALUATED on real and the in-silico->real gap stratified by condition.
    for split in ["val", "test"]:
        for pre in ["Xflat_", "y_", "noc_", "tokens_", "mask_", "phi_", "condition_",
                    "condition_level_", "meta_template_", "meta_qindex_", "attr_", "size_"]:
            suf = "_set" if pre == "y_" else ""
            src = DATA / f"{pre}{split}{suf}.npy"
            if src.exists():
                shutil.copy(src, OUT / src.name)
    # combo_id_{val,test}: leave-one-combo-out grouping for the post-hoc decode fit (-1 = single-source).
    # donor_geno*: CoSA/feas_filter and phi_rerank need them at train time — copied here so the
    # published OUT dir is self-contained and needs no follow-up cp before zipping/uploading.
    for f in ["tokens_open.npy", "mask_open.npy", "Xflat_open.npy", "size_open.npy", "meta_set.json",
              "combo_id_test.npy", "combo_id_val.npy", "fold_info.json",
              "donor_geno.npy", "donor_geno_mask.npy"]:
        if (DATA / f).exists():
            shutil.copy(DATA / f, OUT / f)
    print(f"Saved {len(Xf)} train ({len(Xf_new)} in-silico + {ss.sum()} real SS) -> {OUT}")
    feat_summary(Xf[perm][:2000], NOC[perm][:2000], "generated train (mixed)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=int, default=0, help="generate N in-silico train mixtures")
    ap.add_argument("--include_test_combos", action="store_true",
                    help="version A (fidelity/recreate-leak): allow test combos in train")
    ap.add_argument("--oversample_test", type=int, default=0,
                    help="generate K synthetic copies of each test combo (true recreate-leak)")
    ap.add_argument("--noc_weights", type=str, default=None,
                    help="relative NOC sampling weights for NOC2,3,4,5 e.g. '1,1.5,2.5,2' "
                         "(default uniform). Higher = more high-NOC mixtures.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--realistic", action="store_true",
                    help="root-cause fix: real-matched mixture proportions (min-phi~0.125, "
                         "balanced) + louder/wider template + more jitter; closes the synth->real "
                         "N5 height gap (domain AUC 0.70->0.57). Also via env STR_REALISTIC=1.")
    ap.add_argument("--wide", action="store_true",
                    help="SUPERSET curriculum: per-mixture coin-flip between original-skew and "
                         "realistic-balanced -> spans both extremes, contains real as a dense sub-region "
                         "(broad knowledge without overfitting the test prior). Also via STR_PROP_MODE=wide.")
    args = ap.parse_args()
    global PROP_MODE
    if args.wide:
        PROP_MODE = "wide"
    elif args.realistic:
        PROP_MODE = "realistic"
    assert PROP_MODE in ("original", "realistic", "wide"), f"bad PROP_MODE {PROP_MODE}"
    print(f"proportion model: {PROP_MODE.upper()}")
    rng = np.random.default_rng(args.seed)
    pool = build_ss_pool()
    noc_p = None
    if args.noc_weights:
        w = np.array([float(x) for x in args.noc_weights.split(",")], dtype=float)
        assert len(w) == 4, "noc_weights needs 4 values (NOC2,3,4,5)"
        noc_p = (w / w.sum()).tolist()
        print(f"NOC sampling distribution (2,3,4,5): {[round(p,3) for p in noc_p]}")
    if args.build > 0:
        build_train(args.build, pool, rng,
                    exclude_test=not (args.include_test_combos or args.oversample_test > 0),
                    oversample_test=args.oversample_test, noc_p=noc_p)
    else:
        fidelity_check(pool, rng)


if __name__ == "__main__":
    main()
