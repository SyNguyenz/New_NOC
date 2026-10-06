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
import argparse, json, math, os, shutil
from pathlib import Path
import numpy as np
import kit

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
        if len(acc[j]) >= BIN_SIZE_MIN_N:
            out[j] = float(np.median(acc[j]))
    # Bins seen too rarely take their size from a per-locus straight line in base pairs, x = whole repeats x repeat
    # length + the decimal's bases (15.3 = 15 repeats + 3 bp), fitted on the well-seen bins of that locus. The old
    # fallback, the locus MEAN, put 249 of 590 bins 5-28 bp off (FGA up to 116): the pull-up filter then looked for
    # other-channel neighbours at the wrong size and missed a 20x taller TH01 7 beside D1S1656 15.3, so the NOC4-5 n-1
    # the export deletes there (real kept 0 of 69) stayed in the generator (0.89). The line fits every well-seen bin
    # within 0.08-0.47 bp. Loci with fewer than three well-seen bins, and AMEL/Yindel, keep the mean.
    for L in range(int(BIN_LOCUS.max()) + 1):
        idx = np.where(BIN_LOCUS == L)[0]
        seen = idx[out[idx] > 0]
        if not len(seen):
            continue
        m = float(np.mean(out[seen]))
        rl = BIN_SIZE_RL.get(_LOC_NAME.get(L, ""), 4)
        al = BIN_ALLELE[idx]; wh = np.floor(al + 1e-6)
        x = wh * rl + np.round((al - wh) * 10)
        ok = (out[idx] > 0) & (al >= 0)
        fit = np.polyfit(x[ok], out[idx][ok], 1) if ok.sum() >= 3 else None
        for k, j in enumerate(idx):
            if out[j] == 0:
                out[j] = float(np.polyval(fit, x[k])) if fit is not None and al[k] >= 0 else m
    return out


BIN_SIZE_MIN_N = 3                                          # a median of fewer peaks is left to the per-locus line
# BIN_SIZE_RL = {"D22S1045": 3}
BIN_SIZE_RL = {"D22S1045": 3, "Penta D": 5, "Penta E": 5}                               # repeat length (bp); every other GlobalFiler STR is 4
_LOC_NAME = {v: k for k, v in LOCUS_TO_IDX.items()}

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
FWD_TARGET = np.full(N_FLAT, -1, dtype=np.int64)       # bin at allele+1: FORWARD stutter (n+1)
_OFF_TARGET: dict[int, np.ndarray] = {}                # offset (0.1 repeat units) -> target bin per bin
for j in range(N_FLAT):
    a = float(BIN_ALLELE[j])
    if a >= 0:                                          # skip X(-2)/Y(-1) amel
        STUTTER_TARGET[j] = _BININDEX.get((int(BIN_LOCUS[j]), round(a - 1.0, 1)), -1)
        FWD_TARGET[j] = _BININDEX.get((int(BIN_LOCUS[j]), round(a + 1.0, 1)), -1)
# Range must cover ART_SPAN: the table may name offsets out to 8 repeats, and an offset with no entry
# here is dropped in silence - which is what happened when the span was widened and nothing changed.
for d in range(-int(float(os.environ.get("STR_ART_SPAN", "100"))),
               int(float(os.environ.get("STR_ART_SPAN", "100"))) + 1):
    if d == 0:
        continue
    t = np.full(N_FLAT, -1, dtype=np.int64)
    for j in range(N_FLAT):
        a = float(BIN_ALLELE[j])
        if a >= 0:
            t[j] = _BININDEX.get((int(BIN_LOCUS[j]), round(a + d / 10.0, 1)), -1)
    if (t >= 0).any():
        _OFF_TARGET[d] = t

# ── PER-CONTRIBUTOR physics: the only generation path. ──
# Audit finding: the generator's ONLY per-contributor channel is phi_c, a single scalar. Degradation uses
# one beta for the whole mixture, locus efficiency is shared, and stutter is added to the SUMMED profile —
# all of which act identically on every candidate genotype and therefore cannot separate a true donor
# from a decoy even in principle. On real data the two per-contributor shape channels are strong
# (cross-locus coherence AUROC 0.714, degradation-fit 0.693) while on synth they sit at chance (0.494,
# 0.530), which is why mechanisms built on them kept measuring 0.50.
#   beta_c     per-contributor degradation slope. Real contributors in one mixture can differ in DNA
#              quality; a donor's peaks then follow ITS OWN curve in fragment size, which is what makes
#              the size channel identifying rather than a constant.
#   stutter    generated from each contributor's own parent peaks rather than from the sum, so a stutter
#              product is proportional to the contributor that made it. Total stutter is unchanged
#              (sum of per-contributor stutter = SR x summed parent), so the calibration to real ~0.136
#              still holds; only the provenance changes.
#   forward    n+1 stutter, absent entirely. FS is computed by enrich_tokens and then sliced off by
#              tokens8; adding the token field without generating the phenomenon would teach a mapping
#              that holds on synth and not on real, which is how the 11-field token lost to the 8-field.
#   noise      real single-source profiles are 61.2% non-donor peaks: 29.4% back-stutter, 7.8% forward,
#              and 62.7% neither — a floor of ~36 peaks per profile sitting just above threshold. The
#              generator emitted 0.8 (DI_LAMBDA), a 44x shortfall, and 50.9% of real artefacts land on a
#              panel bin so they survive feas_filter and reach the encoder. Counting a faint fifth donor
#              among ~29 on-panel junk peaks is a different problem from counting it in a clean profile,
#              which is why synthetic count accuracy in the low-evidence regime is 0.80 against real 0.66.
#
# Constants below are ESTIMATED FROM THE 5247 REAL SINGLE-SOURCE PROFILES IN TRAIN (derive_laws.py) —
# never from test, since an earlier calibration read its target off the evaluation set. A single-source
# profile needs no deconvolution: any peak outside that donor's genotype is an artefact by definition,
# and its degradation slope is a direct regression of log height on fragment size.
# Per-contributor degradation was originally modelled as beta_c = beta_mix + delta_c on top of the
# source profile. It was removed: the profile is a real run and already carries its own degradation, so
# the extra tilt double-counted it and halved the median true-peak height. Real degradation variety
# comes from the pool itself.
FS_RATE  = float(os.environ.get("STR_FS_RATE", "0.0155"))   # measured median n+1 / parent (back: 0.0681)
FS_SIGMA = 0.50                # real p90/median is heavier, but that tail is contaminated by noise peaks
# Ratio ceiling r_max = R_SPREAD_FLOOR * (k-1): 15 / 30 / 45 / 60 at k = 2..5. The reference data tops
# out at 9:1 at EVERY k — its {1,2,4,9} grid does not widen with contributor count — so 6+5(k-2) was
# NARROWER than reality at k=2,3 (min-phi p5 0.223 against real's 0.100) and wider at k=4,5. A ceiling
# a comfortable margin above 9 is needed at every k because a continuous spread cannot place 5% of its
# mass exactly at its endpoint, where real's 27% of 1:9 two-person mixtures sit. Of the ceilings tried
# this is the only one covering real at both ends for all k, and it also has the steepest k-ramp (3.78x
# against 3.08x and 2.61x), which matters because the count head reads skew as a cue for k. 60:1 at NOC5
# is far beyond anything in the reference set, deliberately: casework ratios are arbitrary and the
# training span is meant to contain reality rather than imitate it.
# Contributors were briefly given a shared locus-efficiency pattern, on the reasoning that one mixture
# is one amplification. A variance decomposition of the per-profile log-efficiency vectors, grouped by
# run, says otherwise: 0.0012 of the 0.4037 total variance lies BETWEEN runs and 0.4025 within, so the
# shared fraction is 0.003 and the mechanism describes nothing. The 0.62 that had been used was fitted
# to reproduce the observed inter-contributor correlation of 0.229, which must therefore arise from
# something else - locus allele density and the shared degradation environment - rather than from a
# common efficiency vector. Fitting a constant to an outcome is what that was; the decomposition is
# what the data supports.
# The offset table's rates are OBSERVED: an artefact only counts if it cleared the threshold and did
# not land on somebody's true allele. Emitting at the observed rate therefore under-delivers, and it
# showed - a reconstructed two-person mixture carried 45 artefacts against real's 65, uniformly ~25%
# short at every RFU level, and the model reads a cleaner profile as fewer people. The multiplier is
# tried at 1.6, which matched the observed artefact count (68 against 65) but cost 28 extra raw peaks
# and broke the monotone fall of count accuracy with NOC - junk pushing the gate up rather than
# fidelity - so emission stays at the measured rate.
# How far from a real allele an artefact may still be attributed to it. 2.5 repeats, and the wider
# setting was tried in full. Beyond that distance the same-locus peaks are NOT the parent's artefacts:
# they sit 3-8 repeats out at a steady ~4% of the parent, which is what a trace amount of somebody
# else's DNA looks like, not what slippage produces. They are a PROFILE-level phenomenon, and NOISE_N
# - a profile-level count - is the right carrier for them even though its placement is crude. Emitting
# them per parent instead multiplies them by the contributor count and drops them on positions that
# can be another donor's real allele, manufacturing evidence: the twin's composition matched better
# (same-locus 22.0/19.8/19.5/19.4% against real's 24.6/21.4/20.0/21.7%, flat as real is, where before
# it fell 18.2 -> 12.1) and yet ID fell 0.964 -> 0.950 and count 0.860 -> 0.833. Structure alone is
# not the criterion. Note the two derivations share this constant deliberately - NOISE_N counts what
# lies OUTSIDE it, so widening one without the other emits the population twice.
ART_SPAN = float(os.environ.get("STR_ART_SPAN", "25"))
# Per-sample error budget on the quantities the SPECIFICATION fixes. Quantities that are stochastic by
# nature - dropout, stutter scatter, the noise floor - are not in here: for those only the level the
# variance sits at can be right, and forcing them per sample would delete real variation.
# t_total is an OBSERVED total: it already contains the artefacts and the noise floor. Spending it as
# the ALLELE budget and adding those on top overshoots, and correcting that by ITERATION diverges,
# because rescaling shrinks the alleles and their stutter proportionally while the noise floor is an
# ABSOLUTE mass that does not move - each pass overshoots by relatively more (mean total ran
# 0.984 -> 0.865 -> 0.773 -> 0.700 at 2/4/6/8 passes). Solve it instead:
#     observed = B*(1 + a) + N   ->   B = (t_total - N) / (1 + a)
# with a the PROPORTIONAL artefact mass (stutter) and N the ABSOLUTE one (the floor). Fitted on the
# NOC1 artefact-fraction-versus-brightness curve, which runs 0.2694 at ~1200 RFU total down to 0.0474
# above 190000: a = 0.050, N = 279 RFU. Predicts 0.098 / 0.067 / 0.055 against the observed
# 0.100 / 0.064 / 0.051 at the three middle bands.
ART_MASS_A = float(os.environ.get("STR_ART_MASS_A", "0.050"))
ART_MASS_N = float(os.environ.get("STR_ART_MASS_N", "279.0"))
BUDGET_SOLVE = os.environ.get("STR_BUDGET_SOLVE", "1") == "1"
TOL_ITERS = int(os.environ.get("STR_TOL_ITERS", "2"))
ACCEPT = os.environ.get("STR_ACCEPT", "1") == "1"
TOL_SLOPE = float(os.environ.get("STR_TOL_SLOPE", "0.0025"))   # size decline against the treatment
TOL_SHARE = float(os.environ.get("STR_TOL_SHARE", "0.35"))     # each contributor against its phi
BASE_SLOPE = 0.0006                                            # untreated baseline, measured on NOC1
# Bins that at least one panel donor carries - the only ones feas_filter lets through, hence the only
# ones worth emitting into.
CARR_BIN = np.zeros(N_FLAT, dtype=np.float64)
CARR_N = np.zeros(N_FLAT, dtype=np.int64)      # HOW MANY panel donors carry each bin, not just any
try:
    _gg9 = np.load(DATA / "donor_geno.npy")
    _gm9 = np.load(DATA / "donor_geno_mask.npy").astype(bool)
    _cn9 = {}
    for c in range(_gg9.shape[0]):
        for j in range(_gg9.shape[1]):
            if _gm9[c, j]:
                _k9 = (int(round(float(_gg9[c, j, 0]))), int(round(float(_gg9[c, j, 1]) * 10)))
                _cn9[_k9] = _cn9.get(_k9, 0) + 1
    _cs9 = set(_cn9)
    for _j9 in range(N_FLAT):
        _k9 = (int(BIN_LOCUS[_j9]), int(round(float(BIN_ALLELE[_j9]) * 10)))
        CARR_N[_j9] = _cn9.get(_k9, 0)
        if _k9 in _cs9:
            CARR_BIN[_j9] = 1.0
except Exception:
    CARR_BIN[:] = 1.0
CARR_IDX = np.where(CARR_BIN > 0)[0]
CARR_SET = {(int(BIN_LOCUS[_j]), int(round(float(BIN_ALLELE[_j]) * 10)))
            for _j in range(N_FLAT) if CARR_BIN[_j] > 0}
if not len(CARR_IDX):
    CARR_BIN[:] = 1.0; CARR_IDX = np.arange(N_FLAT)
ART_EMIT = 1.0
R_SPREAD_FLOOR = 15.0
# Artefacts the offset table does NOT reach. The table covers everything within 2.5 repeats of a true
# allele, which is 33.9 of the 57 artefacts a real single-source profile carries; the remaining 23.0
# sit further away and need their own emission. Setting this to the full artefact count double-counted
# the near ones - with AT at its true value the profile then hit the MAX_SEQ ceiling of 160 peaks
# against real's 93. Measured against total RFU across a 20000x range the count barely moves
# (39/37/34/32/35, log-log slope -0.025), so it is a genuine per-profile constant and not a rate.

#   level. Calibrating this on NOC1 alone was wrong: at NOC1 phi=1 so the source profile's own
#   artefacts survive and only a small top-up is needed, while at NOC5 they are scaled away entirely
#   and the mixture needs the full floor. Stripping the source artefacts makes one constant correct at
#   every k, and 35.7 is the count measured directly on real single-source profiles.
NOISE_LN_MU, NOISE_LN_SD = 2.180, 0.530     # absolute log-normal height; re-read from NOC1 below
# A relative form was measured and tried: real artefact height is lognormal(-3.313, 1.831) times the
# locus maximum, and the two agree at NOC1 because a typical single-source locus peaks near 250 RFU.
# Scaling with the locus fixed part of the count gap (NOC5 count 0.145 -> 0.234 on the digital twin)
# but cost more in identification (0.890 -> 0.788), because that distribution's heavy tail reaches 58%
# of the locus maximum and placing peaks that tall at RANDOM bins manufactures evidence for the wrong
# donor. Real artefacts that tall are structured - n-2 stutter, microvariants beside a true allele -
# so the relative height needs structured POSITIONS before it can be used.

_NOC1 = None


# Declared before the first noc1_calib() call, which reads it. Without this the context built below
# raised NameError at import, cal() swallowed it, and EVERY calibrated constant in this file silently
# fell back to its literal - the calibration only ever reached the code that looks constants up at
# call time. The literals were copied from a good derivation so they were close, which is exactly why
# it went unnoticed for so long.
DONOR_DOSAGE = None


def noc1_calib(force: bool = False) -> dict:
    """Constants derived from the real single-source profiles; see code/calibrate.py."""
    global _NOC1
    if _NOC1 is not None and not force:
        return _NOC1
    import types
    import calibrate as _cal
    ctx = types.SimpleNamespace(
        DATA=DATA, N_FLAT=N_FLAT, BIN_LOCUS=BIN_LOCUS, BIN_ALLELE=BIN_ALLELE,
        CARR_SET=CARR_SET, ART_EMIT=ART_EMIT, ART_SPAN=ART_SPAN,
        build_bin_size=build_bin_size, cache=None, SZ=None,
        _BININDEX=_BININDEX, COL=COL, DONOR_DOSAGE=DONOR_DOSAGE, MAX_SEQ=MAX_SEQ)
    _NOC1 = _cal.derive(ctx)
    return _NOC1


def cal(key, default):
    """Calibrated value, falling back to the literal if the reference data is unavailable.

    Everything the generator needs is inferred from real single-source profiles; the literals below
    are only what those inferences produced on THIS dataset, kept so the module still imports without
    it. STR_AT_PC is the one intended override: the detection threshold is an instrument setting,
    while every other constant is a property of the samples and must not be dialled."""
    try:
        v = noc1_calib().get(key)
        return default if v is None else v
    except Exception:
        return default

SR_SCALE = 1.0             # Stripping the source profile's artefacts also strips its stutter, so the
#   generated stutter now carries the whole load: at 0.301 (calibrated while the source stutter was
#   still present) a reconstructed NOC5 mixture had 116 peaks against real's 129, and at 1.0 it has 122.
#   The value saturates above 1.0, and MAC still reads 10 against real's 12, so stutter alone does not
#   account for the busiest locus - that gap is still open. Historical note: the shipped formula yields
#   a single-source back-stutter median of 0.107
#   against the 0.068 measured on real single-source profiles; the NGM-SELect coefficients it is built
#   from describe a different kit, and this dataset's own profiles are the better authority.
# Gamma shape against expected peak height, from the measured CV curve (shape = 1/CV^2):
#   <30 RFU CV .381 -> 6.9 | 30-80 .397 -> 6.3 | 80-200 .396 -> 6.4 | 200-600 .313 -> 10.2
#   >600 .170 -> 34.6
# Allele dropout probability against that allele's expected RFU, measured on real single-source
# profiles (44.2% at 20-60 RFU, 21.4% at 60-150, 4.9% at 150-400, none above 400).

# Scaled by 0.70: the intact-source filter keeps profiles above 15000 RFU, whose MEDIAN dropout is
# zero but which still lose the occasional allele, so a fraction of the law is already realised in the
# source. At full strength the twin lost 8.9% of true alleles at NOC5 against real's 7.1%; at 0.70 it
# loses 6.2% and the model then reads it at ID .943 / count .866 against real's .954 / .870.
# Residual dropout in the intact-source pool measures 0.008, so essentially the whole law applies.

#   peak-height variation; the shipped CV 0.32 jitter adds a second copy of it. Within-profile
#   concentration (max/median of the donor's own peaks) reads 3.89 at shape 10 against real's 3.27, and
#   falls monotonically to 3.30 at shape 50. Once scale is divided out, the within-profile height shape
#   then matches real across the board (true p50/p90/max per total RFU: 20.10/40.83/66.78 against
#   20.74/43.57/69.64), so what remained of the pooled p90 gap was peak COUNT, not peak height.

#   AT=10 sits at the p25 of the observed height distribution and truncates the whole faint tail, which
#   is measurable on NOC1 without touching any mixture. AT_PC, NOISE_N and SR_SCALE interact - a lower
#   threshold retains more of both the noise floor and the stutter - so the three are solved together
#   against three single-source targets (peak count 93, artefact fraction 0.612, back-stutter 0.068)
#   rather than set one at a time. NOISE_N is the count that SURVIVES the threshold, not the count
#   emitted, which is why it falls when the threshold does.

# Calibrated constants. Defined after cal() so they can read the inference; the literals are the
# values this dataset produced, kept as a fallback when the reference data is absent.
AT_PC = float(os.environ["STR_AT_PC"]) if "STR_AT_PC" in os.environ else float(cal("at", 3.0))  # real single-source retains peaks to p1=3, p10=6;
# Baseline density per panel bin, measured beyond the offset table's reach where nothing structured
# can contribute. An integer bin carries 0.077 and a fractional one 0.040 - an off-ladder position is
# a LESS likely place for a peak, which is the opposite of what the old fractional artefact rows
# assumed. Height comes from the same population.
NOISE_P_INT = float(os.environ.get("STR_NOISE_P", cal("noise_p_int", 0.077)))
NOISE_P_FRAC = float(cal("noise_p_frac", 0.075))
# SHOULDER. A peak within one repeat of an allele peak is thinned by the analysis, integer and fractional
# bins alike, noise and artefacts alike; bp distance does not enter and survivors are no taller than far
# noise. Noise keeps a fraction that falls with the neighbour's height; an artefact beside another allele
# keeps a flat fraction. Both from NOC1; they replace one flat rate that covered fractional bins only.
SHOULDER_H = np.asarray(cal("shoulder_h", [49.0, 176.0, 534.0, 1630.0, 4732.0]), float)
SHOULDER_NOISE = np.asarray(cal("shoulder_noise", [0.647, 0.432, 0.310, 0.219, 0.220]), float)
SHOULDER_ART = float(cal("shoulder_art", 0.72))
_SH_ORDER = [s * d for d in range(1, 10) for s in (1, -1)]


def _shoulder_h(occ_h):
    """Height of the nearest allele peak within one repeat of every bin, 0 where there is none."""
    H = np.zeros(N_FLAT); done = np.zeros(N_FLAT, bool)
    for dd in _SH_ORDER:
        t = _OFF_TARGET.get(dd)
        if t is None:
            continue
        ok = (~done) & (t >= 0)
        hv = np.zeros(N_FLAT); hv[ok] = occ_h[t[ok]]
        hit = ok & (hv > 0)
        H[hit] = hv[hit]; done |= hit
    return H


# ...and the bp DISTANCE to that neighbour decides almost everything, contrary to the note above: re-measured on NOC1
# train+val with distance in real bp (repeat units x 4, not allele-name tenths - 12 vs 11.3 is 1 bp, not 7), an n-1
# landing 1 bp from a standing allele was seen 0 times in 178 while at 2-3 bp it stood about as often as a clean one
# (.78-.95); the one flat SHOULDER_ART (.72) averaged the two and let the generator show .59 at 1 bp and cut 2-3 bp to
# .57-.68. So the shoulder is looked up by distance: 1, 2 or 3 bp (data/shoulder_bp.json, calibrate_shoulder.py), for
# artefact survival and for the noise floor alike; without the file it falls back to the flat terms above.
_Wsh = np.floor(BIN_ALLELE + 1e-6); _UNsh = _Wsh + np.round((BIN_ALLELE - _Wsh) * 10) / 4.0
_NBP = {1: ([], []), 2: ([], []), 3: ([], [])}
for _L in np.unique(BIN_LOCUS):
    _b = np.flatnonzero((BIN_LOCUS == _L) & (BIN_ALLELE >= 0))
    _D = np.abs(_UNsh[_b][:, None] - _UNsh[_b][None, :])
    for _r, _c in zip(*np.nonzero((_D > 0) & (_D < 0.99))):
        _d = int(np.rint(_D[_r, _c] * 4))
        if _d in _NBP:
            _NBP[_d][0].append(int(_b[_r])); _NBP[_d][1].append(int(_b[_c]))
_NBP = {_d: (np.asarray(_j, np.int64), np.asarray(_k, np.int64)) for _d, (_j, _k) in _NBP.items()}


def _shoulder_hd(occ_h):
    """(height, bp distance) of the NEAREST standing allele 1-3 bp from every bin (taller one on a tie); 0, 0 where
    there is none."""
    H = np.zeros(N_FLAT); D = np.zeros(N_FLAT, np.int64)
    for d in (1, 2, 3):
        j, k = _NBP[d]
        if not len(j):
            continue
        v = np.zeros(N_FLAT); np.maximum.at(v, j, occ_h[k])
        new = (v > 0) & (D == 0)
        H[new] = v[new]; D[new] = d
    return H, D


_SBF = DATA / "shoulder_bp.json"
if _SBF.exists() and int(os.environ.get("STR_SHOULDER_BP", "1")):
    _sbj = json.load(open(_SBF))
    SHOULDER_ART_BP = np.array([1.0] + [float(x) for x in _sbj["art"]])
    SHOULDER_NOISE_H = np.asarray(_sbj["noise_h"], float)
    SHOULDER_NOISE_BP = {int(d): np.asarray(v, float) for d, v in _sbj["noise"].items()}
else:
    SHOULDER_ART_BP = np.array([1.0, SHOULDER_ART, SHOULDER_ART, SHOULDER_ART])
    SHOULDER_NOISE_H = SHOULDER_H
    SHOULDER_NOISE_BP = {1: SHOULDER_NOISE, 2: SHOULDER_NOISE, 3: SHOULDER_NOISE}
NOISE_RUN_SD = float(cal("noise_run_sd", 0.45))    # share of the height sd that belongs to the run
NOISE_LN_MU = float(cal("noise_ln_mu", NOISE_LN_MU))
NOISE_LN_SD = float(cal("noise_ln_sd", NOISE_LN_SD))
# The height is each locus's OWN measured distribution (quantiles of log height on NOC1's clean baseline
# bins), with the peaks of one profile correlated through a Gaussian copula - so every locus keeps its real
# marginal exactly and a loud run is still loud everywhere. A log-normal with the right sd put 0.311 of the
# noise at 8-12 RFU where real has 0.419; a common residual shape around per-locus levels did no better;
# splitting a run offset off and adding one back broke it again. The two numbers above remain only for
# when the reference data is absent.
_LQ = cal("noise_loc_q", None)
NOISE_LOC_Q = None
if _LQ:
    NOISE_LOC_Q = np.tile(np.mean(np.array(list(_LQ.values()), float), 0), (int(BIN_LOCUS.max()) + 1, 1))
    for _k, _v in _LQ.items():
        NOISE_LOC_Q[int(_k)] = np.asarray(_v, float)
NOISE_RHO = float(cal("noise_rho", 0.39))
# ONE RUN, ONE STATE. The stutter run offset (art_run_sd) and the noise run offset (NOISE_RHO) were drawn apart,
# but both belong to the same injection. On TRAIN NOC1, after removing what the total, the injection and the
# template explain, a profile's stutter offset and its noise-height offset correlate 0.27; drawn apart the twin read
# 0.05 through the same estimator, fully shared 0.66. The share that reproduces 0.27 is 0.36, so the estimator's own
# noise sits on both sides and cancels. Apart, the twin let artefact types vary against each other where real moves
# them together, and a profile's busiest locus - its MAC - came out short at bright templates.
RUN_COUPLE = float(os.environ.get("STR_RUN_COUPLE", "0.36"))
# THE RUN'S GAIN LIFTS ITS FLOOR. A run whose total stands above what its template and injection predict (the NOC1
# RFU law) was read with more gain, and the baseline is read with the same gain: on TRAIN NOC1 the noise heights of a
# profile rise 0.085 in log per unit of that excess, where the twin, which drew its floor blind to it, reads 0.015.
# NOISE_GAIN charges the difference. Only where the total, template and injection are all known.
NOISE_GAIN = float(os.environ.get("STR_NOISE_GAIN", "0.070"))
# PULL-UP FILTER. The arrays are PROVEDIt's "Filtered" export: pull-up removed by a rule on context - deleted
# with certainty when a peak in another dye channel within 0.5 bp is 20x taller, in part at 10-20x or 0.5-2 bp
# (calibrate.py, NOC1 train pairs). A NOC1 source was filtered in its OWN context, and every emission rate was
# calibrated on filtered NOC1 profiles, so both carry the filter at NOC1's crowding. A mixture puts other
# people's tall peaks beside them: 7% of kept true alleles sit in pull-up context at NOC1 against 20% at NOC5.
# Charged as the increment. An allele keeps (1 - P_mix) / (1 - P_own), its source filtered in its own context;
# an artefact the same, P_own read in the context of the contributor who dominates its parent bins. Noise, whose
# height is the instrument's and not the profile's, is emitted at its rate divided by the NOC1-average survival
# P_bar(bin, h) it already contains, then filtered in the mixture's context. (P_bar at a fixed h averages
# profiles of every loudness, which is right for noise and wrong for stutter: tried on stutter, the NOC1 twin
# lost 0.54 stutter peaks.)
from calibrate import pull_neighbours, pull_prob, PULL_H_GRID
PULL_P = np.asarray(cal("pull_p", []), float)
PULL_BAR = np.asarray(cal("pull_bar", []), float)
_LOCUS_NAMES = [k for k, _ in sorted(LOCUS_TO_IDX.items(), key=lambda kv: kv[1])]
_PULL_NB = {}
_PULL_LG = np.log(PULL_H_GRID)


def _pull_nb(bin_size):
    """Other-channel neighbours of every bin within 2 bp, on the bin sizes the profile is built with."""
    key = (len(bin_size), float(np.sum(bin_size)))
    if key not in _PULL_NB:
        _PULL_NB[key] = pull_neighbours(BIN_LOCUS, _LOCUS_NAMES, np.asarray(bin_size, float))
    return _PULL_NB[key]


def _pbar(idx, h):
    """NOC1-average pull-up deletion for a peak of height h at bins idx (what the emission rates carry)."""
    pos = np.interp(np.log(np.maximum(h, 1.0)), _PULL_LG, np.arange(len(_PULL_LG), dtype=float))
    i0 = np.minimum(pos.astype(np.int64), len(_PULL_LG) - 2); fr = pos - i0
    return PULL_BAR[idx, i0] * (1.0 - fr) + PULL_BAR[idx, i0 + 1] * fr


# PULL-UP IS ALSO EMITTED, not only filtered. PULL_P above deletes peaks in pull-up context, but the Filtered export
# keeps the ones the analysis could not call: on held-out NOC1 the twin has 0.9-1.8 % of its anomalous peaks in pull-up
# context against real's 6.7-8.5 %, and that missing process is the whole of the twin's shortfall in the high tail at
# stutter positions fed by two or more alleles (~0.3 peaks per profile at EVERY NOC - the reason NOC5's gate stayed
# short while NOC2-4 closed). Rate and height ratio per (cause height, distance) cell, measured on TRAIN+VAL NOC1 as
# real minus twin (calibrate_pullup.py); a spectral bleed is per bin and independent. Background in every label.
PU_EMIT = int(os.environ.get("STR_PU_EMIT", "1"))
_PUF = DATA / "pullup_emit.json"
PU_H_EDGES = PU_D_EDGES = None; PU_RATE = PU_RATIO = None
if PU_EMIT and _PUF.exists():
    _puj = json.load(open(_PUF))
    PU_H_EDGES = np.asarray(_puj["h_edges"], float); PU_D_EDGES = np.asarray(_puj["d_edges"], float)
    PU_RATE = np.asarray(_puj["rate"], float); PU_RATIO = np.asarray(_puj["ratio"], float)
PU_RANGE = (0.3, 1.7)                                    # training draws a scale on the rate, like the other artefacts
# SLIPPAGE HAS A LOCUS STATE. The ratio is drawn per parent bin, so several parents feeding one bin were independent
# lognormals - and a sum of those has its relative spread cut by ~sqrt(m), which is why the generator could not reach
# real's tall peaks where two or more alleles both cast stutter (per-step trace: 0.16 such peaks per NOC5 profile
# against real's 0.95, while the totals matched - the noise floor was filling in elsewhere). Real carries a shared
# locus term: on 900 real NOC1 profiles, at bins fed by ONE standing allele where the expectation clears twice the
# profile's baseline, log(observed / expected) has sd 0.249 and correlates +0.110 between two bins of the SAME locus,
# against the twin's 0.379 and -0.024. So the residual after the sample offset is split, a share ART_LOCUS_R of its
# variance moving to one draw per locus and the rest staying per peak; the pooled spread is unchanged, so every NOC1
# number calibrated on it still holds, but the scatter no longer averages away when parents stack.
# The same NOC1 measurement also says the twin's residual is too WIDE (0.368 against real's 0.249): the table's lsd was
# fitted to the pooled ratio distribution, which the median and p90 checks still match, but the spread around the
# expectation at stutter-dominated bins is not the same thing. ART_SCAT_K scales it; both constants are read off NOC1.
ART_LOCUS_R = float(os.environ.get("STR_ART_LOCUS_R", "0.50"))
# WIDTH OF THE RATIO DRAW, PER OFFSET. Scored on the stutter tail alone - bins where three times the expectation already
# exceeds the profile's tallest baseline peak, so the floor cannot reach it - the table's scatter is too narrow: real has
# 1.61 % of those bins standing above 3x the expectation at a single-source position, the generator 1.38 %, and widening
# lsd by 1.2 lands on real exactly with peaks per profile and the artefact share untouched. Artefact survival, which two
# earlier sweeps blamed, only reaches 1.49 % and costs 8.6 peaks per profile - those sweeps scored a statistic that was
# ~80 % baseline noise. Fitted per offset by calibrate_scatter.py on NOC1 TRAIN + VAL; an offset with no entry keeps 1.0.
ART_SCAT_K = float(os.environ.get("STR_ART_SCAT_K", "1.0"))
ART_SCAT_D = {}
_ASD = DATA / "art_scatter.json"
if _ASD.exists() and int(os.environ.get("STR_ART_SCAT_D", "1")):
    ART_SCAT_D = {int(k): float(v) for k, v in json.load(open(_ASD)).items()}
# The per-draw cap at the largest ratio NOC1 ever showed is the only mechanism that cuts the upper tail, and
# the tail is what is short where parents stack (P(>3x expected) at bins fed by 3 sources: real 2.8 % vs 1.1 %).
ART_CAP = int(os.environ.get("STR_ART_CAP", "1"))
# WHY THERE IS NO "emit from the realised parent" KNOB. It would be inert: in the path this generator uses (the
# filtered overlay) `contrib_exp = contrib.copy()`, so the template and the realised heights are the same numbers and
# there is nothing to couple - tried as STUT_REAL_W over 0..1 and every statistic was identical to three decimals.
# Which also voids the earlier reading of the generator's long LEFT tail in log(observed / expected) at stutter bins
# (skew -3.16 against real's +0.27): it is not the template/realised gap. It is the bins where the stutter did NOT
# survive and only the baseline stands - expectation 30+ RFU against an observed 9 - so the left tail is a property of
# artefact SURVIVAL, which is also the only stage that moves the tall-peak statistic at all.
_PBAR_NOISE = np.zeros(N_FLAT)
if PULL_P.size and PULL_BAR.size and NOISE_LOC_Q is not None:
    _q = NOISE_LOC_Q[BIN_LOCUS]
    _PBAR_NOISE = _pbar(np.repeat(np.arange(N_FLAT), _q.shape[1]), np.exp(_q).ravel()).reshape(_q.shape).mean(1)
# Where an artefact at a bin can have come from: the parents one and two repeats away in the same locus
# (one past the last bin where there is none).
_ART_PAR = np.stack([_OFF_TARGET.get(_d, np.full(N_FLAT, -1)) for _d in (10, -10, 20, -20)], 1)
_ART_PAR = np.where(_ART_PAR >= 0, _ART_PAR, N_FLAT)
_NCDF = np.vectorize(lambda _x: 0.5 * (1.0 + math.erf(_x / math.sqrt(2.0))))
_FRAC_BIN = (np.rint(BIN_ALLELE * 10).astype(np.int64) % 10) != 0
# Artefact visibility against how common the target allele is. Measured on NOC1 at fixed locus,
# parent height and parent repeat count: 0.29-0.46 where one or two panel donors carry the position
# against 0.72-0.92 where eleven or more do. This is the term that makes an otherwise NOC-invariant
# law behave differently at NOC5 - the n-1 positions still free there are the ones nobody carries.
_CARR_MULT = np.interp(CARR_N.astype(np.float64),
                       np.asarray(cal("art_carr_edges", [0, 1, 3, 6, 11, 21]), float),
                       np.asarray(cal("art_carr_mult", [1.0] * 6), float))
# ...with mean 1 over the population ART_SURV_P was fitted on, because that curve is NORMALISED to
# saturate at 1 and the emission rate supplies the ceiling. The multiplier as derived averages 0.939
# over the n-1 targets, so multiplying it in pulled the whole survival curve down 6 % and the twin
# showed an n-1 at a bright parent 0.79 of the time against real's 0.89 - while the table's own rate,
# weighted by where bright parents actually sit, is 0.8947 against real's 0.8850, i.e. already right.
# Only the SHAPE across carriage is this term's to set; its level belongs to the rate. Weighted by
# how many panel donors carry the parent, since that is how often the position is an n-1 target.
# Per-locus level and per-locus rate. Height is the strong one: 5.5 to 13.1 RFU across loci against a
# global 8.6, and it tracks neither amplification efficiency (0.370) nor fragment size (0.030).
_NL_MU = np.full(N_FLAT, NOISE_LN_MU)
_NL_P = np.where(_FRAC_BIN, NOISE_P_FRAC, NOISE_P_INT)
for _k, _v in (cal("noise_loc_mu", {}) or {}).items():
    _NL_MU[BIN_LOCUS == int(_k)] = float(_v)
for _k, _v in (cal("noise_loc_p", {}) or {}).items():
    _m = (BIN_LOCUS == int(_k))
    _NL_P[_m] = float(_v) * np.where(_FRAC_BIN[_m], NOISE_P_FRAC / max(NOISE_P_INT, 1e-9), 1.0)
# Per-BIN rate where NOC1 has enough observations to support one. Bins are not interchangeable inside
# a locus: their occupancy scatters 0.0153 where a common rate would give 0.0055, so 2.8x more than
# counting noise, and runs 0.063 to 0.157. Height carries no such structure - inside a locus bins
# differ by 1.15x and the loudest averages 18.8 RFU - so only the rate is per bin.
# THERE IS NO LOCUS-MAX TERM. One was carried here - unexplained peaks per locus falling 2.247 to
# 1.225 as the locus max rises, spearman -0.159 - and it is a between-locus confound. Loci whose
# peaks are tall are also loci with a low baseline rate, and that is already in the per-bin rates
# below. Re-derived against those rates and after the crowd term, the curve is 0.988 to 1.019 across
# ten decades of locus height: nothing. It was doing real work only as compensation - switching it
# off cut the faint band's excess noise peaks from 11.35 to 1.45 per profile - for the crowd term's
# footing error, which is fixed at its source now.
# The suppression is HEIGHT-DEPENDENT: a crowded profile loses its FAINT noise and keeps its tall
# noise. Measured on NOC1 alone, by how many bins carry SIGNAL and by the noise peak's own height,
# the rate relative to NOC1's mean - and the occupancy is counted the way the generator counts it,
# on the profile before the floor is added, because that is the only number the generator has when
# it makes the lookup. Read instead on the finished profile, as it was, the two footings are 73.7
# against 39.9 bins at faint templates: the generator handed the curve half the occupancy it was
# built on, read the profile as sparse and kept its faint noise, x1.35 at <8 RFU. That was the whole
# of the faint band's artefact excess - copying real's occupancy into the lookup takes the bias from
# +0.0438 to +0.0013 - and no parameter of this function could have fixed it. Rates are normalised
# by the per-bin base rate below, so bin-to-bin differences do not leak in, and the table is scaled
# so the NOC1 marginal is unchanged. (The >25 band is left at 1.0: there crowding and brightness are
# confounded and the rise is pull-up.)
NOISE_CR_H = np.asarray(cal("noise_cr_h", [4.0, 10.0, 14.0, 20.0, 40.0]), float)   # height mid-points
NOISE_CR_S = np.asarray(cal("noise_cr_s", [33.0, 46.0, 57.0, 68.0, 78.0, 90.0]), float)  # SIGNAL bins
NOISE_CR_M = np.asarray(cal("noise_cr_m", [
    [1.169, 1.100, 1.031, 0.977, 0.912, 0.822],     # <8 RFU
    [1.078, 1.079, 1.037, 0.968, 0.935, 0.908],     # 8-12
    [1.001, 1.022, 1.041, 0.946, 0.948, 1.040],     # 12-16
    [0.886, 0.913, 0.980, 0.898, 0.946, 1.356],     # 16-25
    [1.000, 1.000, 1.000, 1.000, 1.000, 1.000]]), float)   # >25: not modelled
# Past the last column: each band's NOC1 log-linear trend per unit of the abscissa where it is a suppression,
# held where it is not. A mixture at NOC5 carries 123 signal bins at the median against NOC1's ~90 ceiling;
# held at the table's edge its faint noise was suppressed to 0.82 where the NOC1 trend, extended,
# gives 0.63 - and the twin carried about ten extra peaks per mixture.
# Past NOC1's edge nothing is measured, so neither end is known to be right: holding at the edge and extending the
# trend are both NOC1 readings, and the truth lies between. Each profile draws where it sits, uniformly between them
# (a RANGE, not a pick). Extending always suppressed too much in the most crowded mixtures - the noise shortfall grew
# with occupancy, -0.9 peaks at 110-130 bins and -1.3 past 150 - and it is the locus with most peaks that sets MAC.
NOISE_CR_EXT = np.asarray(cal("noise_cr_ext", [-0.0060, -0.0034, -0.0002, 0.0, 0.0]), float)
# The crowding curve ABSCISSA: how many peaks at or above this height are on the plate. See the lookup in
# gen_mixture for why it is not the total peak count.
NOISE_CR_TH = float(cal("noise_cr_th", 50.0))
# ...CHARGED AS AN INCREMENT. The table above was read off finished real profiles, where crowding is tangled with
# everything else that moves the floor (the run's gain, its stutter state), so on top of the laws that now carry
# those it counted them twice: bright NOC1 twins lost 0.6 noise peaks and 0.3 of MAC. Read instead against the
# generator itself - each TRAIN NOC1 profile beside its twin made with this law off (code/calibrate_crowd.py) - the
# count barely moves with occupancy (x1.04 sparse, x0.99 crowded) -- which was an artefact of the ABSCISSA, the
# signal-bin count: re-read against the number of peaks at or above NOISE_CR_TH the faintest band falls 0.999 ->
# 0.867 monotonically across the columns, so the file now carries its own log-linear extension as well. What
# remains besides the count is height: real keeps more floor at 8-12 RFU and much less above 25 RFU, where the
# twin's floor runs tall.
_CRI = DATA / "noise_cr_inc.json"
if _CRI.exists():
    _cri = json.load(open(_CRI))
    NOISE_CR_H = np.asarray(_cri["noise_cr_h"], float); NOISE_CR_S = np.asarray(_cri["noise_cr_s"], float)
    NOISE_CR_M = np.asarray(_cri["noise_cr_m"], float)
    NOISE_CR_EXT = np.asarray(_cri.get("noise_cr_ext", np.zeros(len(NOISE_CR_H))), float)
    NOISE_CR_TH = float(_cri.get("noise_cr_th", NOISE_CR_TH))
_PAR_BIN = np.full(N_FLAT, -1, dtype=np.int64)      # bin cua cha n+1, -1 neu khong co
_posmap = {(int(BIN_LOCUS[_j]), int(round(float(BIN_ALLELE[_j]) * 10))): _j for _j in range(N_FLAT)}
for _j in range(N_FLAT):
    _PAR_BIN[_j] = _posmap.get((int(BIN_LOCUS[_j]), int(round(float(BIN_ALLELE[_j]) * 10)) + 10), -1)
# `_CR_MAX` is only the normaliser that keeps the per-peak keep probability inside [0, 1]: the floor is drawn this
# much richer and thinned back by the curve. Any value at or above the table's max works, and a larger one only adds
# over-draw variance - so it is CAPPED. Splitting the crowding table's top column let the tall band (>= 25 RFU) of
# the last column read 4.08 (212 real peaks against 52 of the twin's), which is a real ratio but not a crowding one:
# across the columns that band runs 0.16, 0.25, 0.27, 0.32, 0.46, 1.13, 4.08, i.e. real has FEWER tall floor peaks
# than the twin in quiet profiles and far more in bright ones - that is brightness (the tall unexplained peaks a
# bright run carries), not crowding. Left to set `_CR_MAX` it inflated every emission 4.4x instead of 1.5x and
# reshaped the surviving floor (tall noise kept at rate 1, faint at 0.16), which cost more than it bought: the noise
# excess moved +0.65/-0.26/+1.29/+1.23/+1.14 -> +0.44/+0.56/+1.34/+1.07/+0.61 but the gate went the wrong way,
# +0.118/+0.140/+0.188/+0.028 -> +0.151/+0.209/+0.210/+0.092. Capped, the keep probability saturates at 1 for that
# band - the most the mechanism can do for it without over-drawing everything else.
# Taken from the bands that CARRY THE COUNT and behave like a crowding law - the two faintest, whose values run
# 0.70 to 1.28 across the columns - and not from the tall bands, whose ratios are the brightness effect above
# (0.16 -> 4.08). `_keep` is clipped at 1 below, so a tall band simply keeps everything, which is the most the
# mechanism can do for it. Chosen structurally, not by any score on the mixtures.
_CR_MAX = float(max(NOISE_CR_M[:2].max(), 1.0)) if NOISE_CR_M.shape[0] >= 2 else float(max(NOISE_CR_M.max(), 1.0))

_bp = cal("noise_bin_p", {}) or {}
if _bp:
    _ba10 = np.rint(BIN_ALLELE * 10).astype(np.int64)
    for _j in range(N_FLAT):
        _v = _bp.get(f"{int(BIN_LOCUS[_j])}_{int(_ba10[_j])}")
        if _v is not None:
            _NL_P[_j] = float(_v) * (NOISE_P_FRAC / max(NOISE_P_INT, 1e-9) if _FRAC_BIN[_j] else 1.0)
RFU_LOG_C = float(cal('rfu_coef', [0.976, 0.948, 9.743])[2])          # log RFU = 0.976*log(ng) + 0.948*log(inj) + 9.743, both exponents ~1
RFU_LOG_SD = float(cal('rfu_sd', 0.603))         # residual scatter, 1.83x
JIT_H = np.asarray(cal('jit_h', [0.0, 15.0, 55.0, 140.0, 400.0, 1500.0, 1e9]), float)
JIT_SHAPE = np.asarray(cal('jit_shape', [6.9, 6.9, 6.3, 6.4, 10.2, 34.6, 34.6]), float)
# Artefact survival against the artefact's OWN expected height, from NOC1 once the flat baseline
# noise (0.095 per bin, measured at offsets that are not products) and the emission rate are both
# taken out. Half-loss lands at 26 RFU against the 27.5 the allele dropout law reports on the same
# profiles: one height law, applied to whatever peak has to clear the floor.
ART_SURV_H = np.asarray(cal('art_surv_h', [0.0, 4.0, 7.0, 11.0, 17.0, 26.0, 40.0, 65.0, 110.0, 200.0]), float)
ART_SURV_P = np.asarray(cal('art_surv_p', [0.043, 0.145, 0.220, 0.339, 0.491, 0.651, 0.773, 0.859, 0.902, 1.0]), float)
# The half-loss point read back off GENERATED profiles with the estimator that derived it - fit the
# size trend on each profile's own peaks, predict every allele, count what is missing - lands 15%
# above what the direct fit reports. Both sides measured the same way: at this gain the generated
# dropout curve sits on real's band for band (1.02 / 1.03 / 1.00 / 0.97 against real's 0.830 / 0.664
# / 0.491 / 0.262), where the uncorrected value reads 0.90 and 0.81 through the middle.
#
# It is NOT raised further to close the remaining 9% of TOTAL dropout. That gap is composition, not
# level: the generated profiles carry 20% fewer alleles at low copy count than real does, and forcing
# the total with this constant costs 10-18% on every band and 1.7-6.7x at the abundant end. The two
# structural suspects were measured and cleared - the template shrink accounts for 3 points and the
# increment form for 2.
# Which alleles an amplification keeps is drawn again rather than copied from the source profile; see
# _relottery. Real runs of one specification differ by 0.0561 where the binomial law predicts 0.0588, so the
# lottery is independent between runs and copying one realisation freezes an ensemble that should move.
RELOTTERY = os.environ.get("STR_RELOTTERY", "1") == "1"
# Chance an allele is lost, against its own RFU - measured on NOC1 across the whole dilution ladder, since
# 45 profiles at one level never reach the faint end. It says WHICH allele a redrawn lottery takes.
DROP_H = np.asarray(cal("drop_h", [0.0, 15.0, 40.0, 100.0, 250.0, 500.0, 1e9]), float)
DROP_P = np.asarray(cal("drop_p", [1.0, 0.5, 0.2, 0.05, 0.01, 0.0, 0.0]), float)
# How far apart those chances are within one profile: the share of the equal-chance variance that real
# actually shows, 0.32 over 656-1026 pairs per rung once the injection effect is taken out.
DROP_DISP = float(cal("drop_disp", 0.32))
# Per-allele deviation of a fresh amplification, measured on NOC1: 0.142 shared by a locus, 0.348 per allele (see
# _relottery).
REDRAW_LOC = float(os.environ.get("STR_REDRAW_LOC", "0.180"))
REDRAW_ALL = float(os.environ.get("STR_REDRAW_ALL", "0.250"))
SAT_RFU = float(cal("sat_rfu", 0.0))
# Peak-height scatter is realised at AMPLIFICATION and is INVARIANT across injections of the same
# product; the capillary adds only a small extra. Measured on 3601 NOC1 injection pairs (same well,
# same extract, same run, different seconds): sd(log height ratio) = 0.0944 / 0.1049 / 0.0555 for
# 5->15 / 5->25 / 15->25, mean 0.0850. Two injections each carry one capillary draw, so
# sigma_capillary = 0.0850 / sqrt(2) = 0.060. Everything above that belongs to the PCR and must not
# be redrawn when the same product is injected again.
# INJECTION LAST. Physically the injection is the last thing that happens: mass -> degradation ->
# amplification -> INJECTION -> detection. The file used to apply it FIRST, by scaling t_total, and
# then divide it back out of every law lookup (height/_ig). Two things go wrong that way. The two
# exponents do not match - RFU scales as inj^0.948 (rfu_coef) while the law keys use inj^0.80
# (INJ_BETA) - so a residual inj^0.148 rides on every lookup and two injections of ONE product come
# out with different draws; and the absolute-scale steps that belong to the capillary (noise floor,
# saturation, the analytical threshold) were being applied to numbers the injection had already
# moved. Building at the 15 sec reference and multiplying at the end puts every molecule-count law
# on an exactly injection-invariant quantity and leaves the threshold to do the injection's real
# work. RFU_INJ_BETA is the SAME 0.948 already fitted in rfu_coef, not a new constant. The old
# order is gone, not kept behind a flag: with it every law lookup divided by a factor that the
# reference build makes exactly 1.0, so the division was arithmetic with no effect left to have.
RFU_INJ_BETA = float(cal('rfu_coef', [0.976, 0.948, 9.743])[1])
# Baseline fragment-length slope that survives after a treatment's own cond_beta is taken out:
# -0.00063 (sd 0.00027 across the seventeen treatments on NOC1).
DEG_BASE = float(cal("deg_base", 0.00063))
DEG_RESET = os.environ.get("STR_DEG_RESET", "1") == "1"
TPL_SPREAD = os.environ.get("STR_TPL_SPREAD", "4")
# LEVEL CONVERSION. A source is a real run at a ladder level, and the target rarely sits exactly on
# one - and in the honest case, a different amplification of the same DNA, never: PROVEDIt holds ONE
# PCR product per donor, treatment and template, so the only other amplification is at another level.
# A source AT the target's level is charged nothing; its dropout is inherited, and at NOC1 that
# reproduces real's losses band by band with no model at all (0.0526 dropped against real's 0.0558).
# When the source is RICHER, the target's own amplification loses alleles the source kept. Measured
# on NOC1 dilution pairs - same donor, extract, treatment and injection, each level its own reaction -
# with the upper profile fed in as the source for the lower spec, scaling + scatter + threshold remove
# almost nothing (retention 0.7021 against the source's own 0.7072 at 0.0078 ng) while real loses
#     lower level  0.0078  0.0156  0.0313  0.0625  0.125   0.25
#     q per step   0.226   0.163   0.102   0.042   0.010   0.003
# Stochastic amplification failure - too few molecules entered the reaction - keyed on the ABSOLUTE
# template. The q(f) table this replaces was keyed on f = template / 0.5 ng and clamped below f = 0.085,
# i.e. flat across the whole band where the loss lives: it could not tell 0.0078 ng from 0.0156 ng,
# which is why charging only its difference once moved retention by 0.002.
# Stored as survival relative to the top of the ladder, PHI(t), so a step of any length is
# PHI(target) / PHI(source). Two steps compose to within 0.01 of the measured two-step loss except at
# the floor (0.304 measured against 0.353 composed - an allele weak at one level is weak at the next),
# and the draw prefers the level just above, so more than one step is taken on 0.2-0.6% of draws.
AMP_STEP_T = np.asarray(cal("amp_step_t", [0.0078, 0.0156, 0.0313, 0.0625, 0.125, 0.25, 0.5]), float)
AMP_STEP_PHI = np.asarray(cal("amp_step_phi", [0.5496, 0.7104, 0.8489, 0.9451, 0.9868, 0.9971, 1.0]),
                          float)
# The same chain per injection time: the sources are matched on injection, and at faint templates what a step loses
# depends on how much product reached the capillary (on the pooled chain one step under-dropped at 5 s by 0.016 and
# over-dropped at 15 s by 0.014). An injection the table does not hold takes the pooled chain.
AMP_STEP_PHI_INJ = {float(_k): np.asarray(_v, float) for _k, _v in (cal("amp_step_phi_inj", {}) or {}).items()}


AMP_STEP_SIZE = float(cal("amp_step_size", -0.378))
# ...per rung, and with the HEIGHT term the pooled fit used to throw away: an allele that came out faint at the
# richer rung is the one the step takes (logit coefficient on centred log height 0.20 at 0.0078 ng to 2.09 at
# 0.25 ng), which is why a faint contributor in real keeps FEWER alleles and TALLER ones than a height-blind
# step produces. Read per rung because pooling without each rung's intercept flattens the size slope as well.
AMP_STEP_SIZE_T = np.asarray(cal("amp_step_size_t", [AMP_STEP_SIZE] * len(AMP_STEP_T)), float)
AMP_STEP_H_T = np.asarray(cal("amp_step_h_t", [0.0] * len(AMP_STEP_T)), float)


def _step_coef(t):
    """(size, height) logit coefficients of the step down, at the TARGET template."""
    t = max(float(t), 1e-9)
    if t <= AMP_STEP_T[0]:
        return float(AMP_STEP_SIZE_T[0]), float(AMP_STEP_H_T[0])
    if t >= AMP_STEP_T[-1]:
        return float(AMP_STEP_SIZE_T[-1]), float(AMP_STEP_H_T[-1])
    lt = np.log(t); lg = np.log(AMP_STEP_T)
    return float(np.interp(lt, lg, AMP_STEP_SIZE_T)), float(np.interp(lt, lg, AMP_STEP_H_T))
# Treatment conversion, per code: retention ratio treated / untreated by rung, and its spread over size.
# See calibrate.py; applied only where a treated contributor had to be drawn from the untreated ladder.
TREAT_CONV = cal("treat_conv", {}) or {}


def _pin_mean(s, lin):
    """Per-allele survival whose MEAN is s, with per-allele logit offsets lin. The intercept is solved so the
    mean is kept exactly: the offsets move WHICH alleles go, never how many."""
    lin = np.clip(np.asarray(lin, dtype=float), -30.0, 30.0)
    s = float(min(s, 1.0))
    if s >= 1.0 or len(lin) == 0:
        return np.full(len(lin), s)
    s = min(max(s, 1e-4), 1.0 - 1e-6)
    a = float(np.log(s / (1.0 - s)))
    ex = lambda a_: 1.0 / (1.0 + np.exp(-np.clip(a_ + lin, -60.0, 60.0)))
    p = ex(a)
    for _ in range(30):
        f = float(p.mean()) - s
        if abs(f) < 1e-7:
            break
        a -= f / max(float((p * (1.0 - p)).mean()), 1e-9)
        p = ex(a)
    return p


def _spread_surv(s, sizes, slope):
    """Survival with mean s spread over fragment size alone (the treatment conversion's form)."""
    sizes = np.asarray(sizes, dtype=float)
    return _pin_mean(s, slope * ((sizes - sizes.mean()) / 100.0) if len(sizes) else sizes)


def _conv_surv(t, L, sizes, inj=None, logh=None):
    """Per-allele survival for taking a source at rung L down to template t: mean PHI(t) / PHI(L) at this
    injection, spread over fragment size AND over the allele's own height at the rung it comes from, at the
    coefficients measured between NOC1 dilution rungs. The mean is kept exactly; the terms move only WHICH."""
    sizes = np.asarray(sizes, dtype=float)
    s = float(min(_lvl_surv(t, inj) / max(_lvl_surv(L, inj), 1e-9), 1.0))
    bs, bh = _step_coef(t)
    lin = bs * ((sizes - sizes.mean()) / 100.0) if len(sizes) else sizes
    if logh is not None and bh > 0.0:
        lh = np.asarray(logh, dtype=float)
        if len(lh) == len(sizes):
            lin = lin + bh * (lh - lh.mean())
    return _pin_mean(s, lin)


def _treat_mean(cond, t):
    """Mean survival of an untreated source's alleles under treatment cond at template t (1 if unknown)."""
    tc = TREAT_CONV.get(str(cond))
    if not tc or not tc.get("t"):
        return 1.0
    return float(min(np.interp(np.log(max(float(t), 1e-9)), np.log(np.asarray(tc["t"], float)),
                               np.asarray(tc["r"], float)), 1.0))


def _treat_surv(cond, t, sizes):
    """Per-allele survival for an untreated source standing in for a contributor treated with cond."""
    tc = TREAT_CONV.get(str(cond)) or {}
    return _spread_surv(_treat_mean(cond, t), sizes, float(tc.get("b") or 0.0))


AMP_STEP_TILT = np.asarray(cal("amp_step_tilt", [0.00448, 0.00351, 0.00240, 0.00143, 0.00077, 0.00037,
                                                   0.0]), float)


# The scatter a step down adds beyond what the jitter increment charges (variance of log height around the size
# trend), cumulative from the top of the ladder like the tilt. See calibrate.py (amp_step_var).
AMP_STEP_VAR = np.asarray(cal("amp_step_var", [0.0] * len(AMP_STEP_T)), float)
# ...per injection, as PHI: the pooled chain was right on average but -0.021 / +0.017 / +0.020 in sd log height at
# 5 / 15 / 25 s on the same-extract hold-out.
AMP_STEP_VAR_INJ = {float(_k): np.asarray(_v, float) for _k, _v in (cal("amp_step_var_inj", {}) or {}).items()}


# GENERATE, THEN FILTER. A contributor is its donor's most complete real profile (highest template, injection at
# or above the target's), and which of its alleles stand is decided ONCE, by counting template molecules:
#   - PROVEDIt sets every template mass from the 80 bp qPCR target, and Q (80 bp / 214 bp, measured on the
#     extract that was amplified - for a mixture the mixed extract, treated as one) says how much of it is
#     intact at longer lengths. Under scission the intact template at an allele of size L is
#     m_eff = m * Q^(-a(L-80)/134), with a per treatment family (strand breaks hurt long amplicons harder than
#     the 80/214 bp ratio extrapolates);
#   - an allele receives k ~ Poisson(m_eff / 6.6 pg) copies (a homozygous bin twice that) - 6.6 pg per diploid
#     genome, every molecule counted, no efficiency factor;
#   - each copy contributes a Gamma-distributed amount with mean 1 and CV FILT_CV (the stochastic early cycles),
#     so the height is the source's per-copy height times that sum, stepped down to the target injection with the
#     capillary's scatter. The allele stands iff k >= 1 and it clears the threshold.
# Fitted on TRAIN NOC1 and checked on held-out NOC1: retention within ~0.02 in every template band (0.544 / 0.543
# below 0.012 ng ... 0.982 / 0.976 above 0.1 ng) and every treatment family; whole-locus loss within ~0.02;
# heterozygote balance within ~0.03. A per-locus efficiency on top was tried and removed: the source already
# carries its donor's locus pattern, and adding one widened bright profiles to 0.78-0.82 against real 0.63-0.70.
# Presence is decided here and nowhere else - the source is complete - so no loss is charged twice.
FILTER = os.environ.get("STR_FILTER", "1") == "1"
FILT_PG = 0.0066
FILT_CV = 0.3
FILT_A = {"a": 2.0, "DNase": 1.25, "Fragmentase": 1.25, "UV": 0.5, "sonication": 1.5, "humic": 0.5}
# ...and that exponent is not a constant: the damage beyond the Q law falls with Q itself. On TRAIN NOC1 the factor
# s on the exponent that best explains a treated sample's presence runs log2 s +0.53 / +0.10 / -0.18 / -0.47 / -0.81
# at Q 1.3-1.7 / 1.7-2.5 / 2.5-4 / 4-8 / >8 for DNase, the same shape for every family - the constant above is only
# the average over NOC1's own Q, and mixture tubes sit lower (NOC5 median 1.6), where it under-charges. Per family:
# log2 s = alpha + beta ln Q, then a residual (sd with the estimation noise removed) split into a donor part, drawn
# per contributor, and a tube part, drawn once per sample since a mixture is treated as one tube; the split keeps
# the proportions measured before the Q trend was taken out. (alpha, beta, sd donor, sd tube):
# HEIGHT_SHARE_C is read off NOC1 heterozygote balance: at 60 copies the three mass bands land at 0.700 / 0.747 /
# 0.830 against real's 0.693 / 0.748 / 0.838, where the old per-allele draw gave 0.694 / 0.728 / 0.806. Dropout is
# untouched by it (losing one allele of a pair 0.352 against 0.359, losing the locus 0.215 against 0.204).
HEIGHT_SHARE = float(os.environ.get("STR_HEIGHT_SHARE", "1.0"))     # 0 = the old per-allele height draw
HEIGHT_SHARE_C = float(os.environ.get("STR_HEIGHT_SHARE_C", "60.0"))
FILT_S = {"DNase": (0.539, -0.568, 0.232, 0.465), "Fragmentase": (0.555, -0.727, 0.242, 0.273),
          "UV": (1.134, -0.382, 0.190, 0.470), "sonication": (0.917, -0.785, 0.292, 0.155),
          "humic": (1.194, -0.756, 0.358, 0.352)}
# WHO IS REALLY IN THE TUBE. PROVEDIt mixes WHOLE BLOOD dilutions by approximate volume, then extracts, treats and
# quantifies the mixture as one (naming convention, RD14-0003 methods). A donor's share of the extract is therefore
# the number of its nucleated cells that went in - a Poisson count - over everyone's; the tube's total is re-set by
# the quantification, so the counts redistribute the shares and do not change the total. The expected count of
# donor c is R x (its nominal template) / 6.6 pg, R being how many times the amplified mass the extract holds:
# at least ~5.5 (a 55 uL eluate, at most 10 uL amplified) and up to ~460 (a 1:10 blood dilution, 50 uL, ~3.5e4
# cells, amplified at 0.5 ng). R is not recorded per sample, so it is drawn per tube, log-uniform over that range.
# A single source keeps share 1, so NOC1 is untouched.
CELL_R = (5.5, 460.0)
# CROWDED TUBES AMPLIFY EACH COPY LESS. In real mixtures a faint contributor's alleles are missing more often than
# its own template predicts, and the shortfall behaves as a loss of effective copies: equal on short and long loci
# (not degradation), common to the tube rather than to one donor, growing with the number of contributors, while
# the heavy contributors keep their designed ratios. Forensic practice describes the effect (minor templates
# outcompeted in the early cycles) but no source gives its size, so it is a RANGE: every tube draws kappa uniformly
# in [0, KAPPA_MAX] and every contributor's copies are scaled by 2^(-kappa (K-1)). The top is the most severe
# learnable case, not a fitted value: at the top the lightest designed minor (0.015 ng, ~2.3 copies per allele) in a
# five-person tube still keeps two thirds of its alleles, so its presence stays in the data. The heights
# are re-budgeted to the tube's total afterwards, so only which faint alleles stand changes. KAPPA_FIX pins it (for
# sweeps); a single source has K-1 = 0 and is untouched.
KAPPA_MAX = 0.26
# STUTTER AT LOW TEMPLATE. The artefact table carries one ratio per locus and offset, and NOC1 is dominated by bright
# parents, so it holds the bright-parent ratio at every height. Real is not flat: measured on held-out NOC1 with the
# template read off the OTHER allele of the locus (so the parent's own draw is not in the conditioning), the n-1 a
# parent casts is present about as often as the twin's, but its expected RFU is +55 / +29 / +19 / +11 % above the
# twin's at partner heights 0-100 / 100-200 / 200-400 / 400-1000 RFU, and equal above that; the parent itself is not
# lower. Stutter formed in the first cycles, while only a few copies are there, is over-represented - the known LCN
# effect, and why probabilistic genotyping lets stutter vary inversely with the parent. Kept as the smallest form that
# fits: the emitted ratio is multiplied by 1 + A_d (H / 100)^-G_d, H the parent's expected RFU, capped at STUT_LT_CAP.
# Fitted on VAL NOC1 as the increment over the table: n-1 A 0.5, G 1.1. Held-out TEST NOC1, twin/real - 1 by partner
# band 0-100 / 100-200 / 200-400 / 400-1000 / 1000-3000 RFU: -35 -19 -14 -9 -4 % before, -6 +6 +3 -2 -2 % after.
# n+1 is NOT a low-template effect - its shortfall is flat, -10..-13 % at every height on val, bright parents
# included - so it takes a flat factor (G = 0), 1.12; on test it moves -23 -27 -14 % to -19 -18 -2 % at >= 200 RFU.
# PROVEDIt's value is one lab's: training draws a scale s per tube (STUT_LT_RANGE) so the model sees weaker and
# stronger low-template stutter; a twin uses s = 1.
STUT_LT_A = {-10: float(os.environ.get("STR_STUT_LT_A1", "0.5")), 10: float(os.environ.get("STR_STUT_LT_A2", "0.12"))}
STUT_LT_G = {-10: float(os.environ.get("STR_STUT_LT_G1", "1.1")), 10: float(os.environ.get("STR_STUT_LT_G2", "0.0"))}
STUT_LT_CAP = 4.0
# ...but a raised MEAN is the wrong shape for n-1. Installed that way (A 0.5, G 1.1) the mixtures' faint-parent n-1
# stood 18.0 / 35.2 % of the time at parents 0-100 / 100-200 RFU against real's 13.5 / 25.1, and still came out
# SHORTER when present (0.145 vs 0.190). Real keeps the presence and puts the height in the tail: stutter made in the
# first cycles is all or nothing. So n-1 takes a mean-one Gamma factor with shape H / STUT_LT_H0 - wide at a faint
# parent, vanishing at a bright one - which is the dispersion-inverse-to-parent form probabilistic genotyping uses.
STUT_LT_H0 = {-10: float(os.environ.get("STR_STUT_LT_H0", "0.0"))}
# Tried at H0 10-200 on VAL NOC1: the Gamma lifts height-when-present only 0.131 -> 0.155 (real 0.190) while cutting
# presence at 200-1000 RFU parents, so it is left at 0. What the twin gets wrong is the BOTTOM of the heights: at
# parents under 400 RFU, 22 % of its standing n-1 are under 8 RFU against real's 7 %, while real's non-stutter
# artefacts under 8 RFU are as common as the twin's. A tiny peak four bases in front of its own parent is not called
# as a peak of its own. So an n-1 whose drawn height is under STUT_SMALL_H[0] RFU survives at STUT_SMALL_Q times the
# usual rate (ramping back to 1 at STUT_SMALL_H[1]), and with the small ones gone the raised mean (STUT_LT_A) no
# longer turns into extra presence. Fitted jointly on VAL NOC1 (A 0.5 at G 1.1, Q 0.15). Held-out TEST NOC1, n-1 by
# parent band 0-100 / 100-200 / 200-400 RFU: ratio-when-present 0.125 0.071 0.067 -> 0.151 0.094 0.078 (real 0.194
# 0.101 0.077), share of faint-parent n-1 under 8 RFU 22.9 -> 12.2 % (real 7.7), presence 13.9 / 34.7 / 58.4 % against
# real 14.5 / 30.2 / 54.1 (4 points over in the middle bands - the residual).
STUT_SMALL_H = (6.0, 12.0)
STUT_SMALL_Q = float(os.environ.get("STR_STUT_SMALL_Q", "0.15"))
STUT_LT_RANGE = (0.3, 1.7)
# DROP-IN: somebody else's allele. The peaks that open decoys on real are mostly NOT stutter: 46% stand at a stutter
# position far above the stutter law AND above the sample's own noise p90 (~28 RFU against a 9 RFU floor), and real's
# excess over the generator is nearly all of that kind. On NOC1 such peaks favour allele positions - per bin 2.06x
# more often on a bin some panel donor carries than on one nobody carries, against 1.52x for the twin, which had no
# drop-in. So a drop-in event is one allele of another person: a locus at random, then an allele drawn by how often
# the panel carries it (a common allele is a common drop-in - which is exactly a decoy's allele). WHERE it lands needs
# no fitting; HOW OFTEN (DI_RATE per profile) and HOW TALL (log-normal DI_LMU, DI_LSD at the reference injection,
# scaled by the injection like every molecule-borne peak) are the lab's and are read off VAL NOC1 as the increment
# over the twin (scratchpad dropin_fit.py): 1.5 events per profile, median e^3.0 = 20 RFU. Held-out TEST NOC1, peaks
# above the sample's noise p90 and 3x the stutter law, per profile on allele bins 1.89 -> 2.28 (real 2.37), on bins
# nobody carries 2.74 -> 2.61 (real 2.50), their height 16.9 -> 18.5 RFU (19.0), preference for allele bins 1.57x ->
# 1.99x (2.17x). It belongs to nobody: background in every label. Training draws a scale
# on the rate in DI_RANGE.
# ...and one event is one PERSON, not one allele. With each event drawing an allele on its own, the twin scattered its
# drop-in evenly: at NOC5 real puts 1.34 anomalous peaks per sample on a bin some decoy carries against the twin's
# 0.92, and real has 0.50 decoys collecting two or more of them against 0.24 - while the TOTAL number of anomalous
# peaks already matched (3.53 / 3.35). A contaminant is somebody's sample, so their alleles arrive together, several on
# the same donor's positions, which is exactly what opens a decoy. So an event draws one panel donor who is not a
# contributor, at a low level, and their alleles stand or drop on their own; the event rate is divided by the alleles
# an event leaves standing, so the peaks per profile stay at what NOC1 measured. DI_PERSON=0 restores the per-allele
# draw. Labels are unchanged: drop-in belongs to nobody.
DI_PERSON = int(os.environ.get("STR_DI_PERSON", "1"))
DI_RATE = float(os.environ.get("STR_DI_RATE", "1.5"))
DI_LMU = float(os.environ.get("STR_DI_LMU", "3.0"))
DI_LSD = float(os.environ.get("STR_DI_LSD", "0.5"))
DI_HSD = float(os.environ.get("STR_DI_HSD", "0.9"))      # per-allele scatter inside one contaminating sample
# In person mode the level is the CONTAMINANT's own level, not the height of what shows: only its tall tail clears AT,
# so the level sits near 3 RFU while the peaks that appear are 15-30. Level and event rate are fitted together on VAL
# NOC1 against the same three numbers as before (anomalous peaks on allele bins, on bins nobody carries, their height):
# 0.4 events per profile at a level of e^0.8 = 2.2 RFU. Held-out TEST NOC1, per profile: anomalous peaks on allele
# bins 1.89 -> 2.23 (real 2.37), on bins nobody carries 2.74 -> 2.61 (real 2.50), their height 16.9 -> 17.7 RFU
# (19.0), preference for allele bins 1.57x -> 1.95x (real 2.17x) - the same place the per-allele version reached,
# with the peaks now arriving in groups on one donor's positions.
DI_LMU_P = float(os.environ.get("STR_DI_LMU_P", "0.8"))
DI_RATE_P = float(os.environ.get("STR_DI_RATE_P", "0.4"))
DI_RANGE = (0.3, 1.7)
# SMALL PEAKS ARE HARDER TO SEE IN SOME RUNS, AND NOC1 CANNOT SAY WHICH. In the NOC4-5 test mixtures the n-1 of a
# private parent is seen less often than the peak's own expected height predicts from NOC1 (after conditioning on it:
# -.11 at clean private parents), the ones that ARE seen stand taller than the same donor's NOC1 n-1 (+.04..+.22 in log
# ratio - selection, not a taller stutter) and the smallest non-allele peak per bright locus is 18-28 RFU against
# 11-15 in NOC1 at the same locus brightness. No NOC1 variable reproduces it (dye brightness, crowding, totals,
# treatment, Q, injection, peak count near the target; the same donors are ordinary in NOC1), so it is not fitted:
# a non-allele peak (stutter, floor, pull-up, drop-in) survives as if its height were divided by f, i.e. with
# probability S(h/f)/S(h) on the ART_SURV curve. Twins use f = 1 (the NOC1 behaviour); training draws f per sample
# log-uniformly in DET_F_RANGE so the model meets runs from ordinary to markedly worse at small peaks. Alleles are
# not touched (their presence is matched as it is).
DET_F_RANGE = (1.0, 1.0)     # DISABLED 2026-09-29: the envelope check failed (real above f=1 at NOC4-5 E 15-80), see memory n1-detection-dye-ratio
# FAINT STUTTER IS A KIT WEAKNESS, NOT A MIXTURE LAW. On the GF 3500 data the n-1 of a faint parent is seen less often in
# the 4-5 person runs than NOC1 predicts at the same expected height (the same donor and allele: -.12 at E 7-25, -.03 at
# E 25-40, none above), but on the 3130 IDPlus runs of the SAME donors and mixture designs it is not (+.006), and the
# floor at control bins is untouched (.073 against NOC1's .068-.074). So nothing here models it as physics; instead the
# model is shown both states at every NOC, so it cannot lean on whether a faint stutter stands: training drops each
# surviving artefact below STUT_DROP_H with a probability q drawn per sample in STUT_DROP_RANGE. Measured on NOC1
# TRAIN+VAL with real alleles injected: q = .4 takes n-1 presence down -.11 / -.15 / -.04 / .00 at E 7-15 / 15-25 /
# 25-40 / 40+ (GF NOC4-5: -.14 / -.08 / -.03 / .00) with the control-bin floor untouched (.0707 against .0714); the
# range reaches every band's GF value. det_f above had the wrong shape: it thinned mid-height stutter too (-.14 at E
# 25-40 for f = 1.5) and the floor by a third. Twins use q = 0.
STUT_DROP_H = 25.0
STUT_DROP_RANGE = (0.0, 1.0)                 # the full range: from untouched to every faint artefact dropped
# EACH ALLELE STUTTERS AT ITS OWN RATE. Back stutter depends on the allele's sequence (its longest uninterrupted
# repeat stretch), not only on its length, and the table knows only the locus and a slope in allele number. On real
# NOC1 the n-1 log ratio of a (donor, allele) cell repeats across two independent halves of that donor's runs at
# r = 0.961 after the table's expectation is taken out, and the offset belongs to the ALLELE: between alleles 0.056,
# between donors carrying the same allele 0.009. Stored as the increment over what the generator already makes
# (real minus twin, bright parents >= 800 RFU, TRAIN + VAL NOC1, shrunk by its own standard error; scratchpad
# allele_stut_inc.py): 200 alleles, sd 0.154 log. On held-out TEST NOC1 the same real-minus-twin offsets correlate
# 0.868 with it. An allele the file does not name keeps the locus law.
# ...and not only n-1, and not only its height. Copying the answer into the generator layer by layer (v2 judging)
# left the synth-real count gap in place with the WHOLE allele layer copied, and reproduced all of it once real n-1
# AND n+1 were copied on top of real alleles - with the same totals: the stutter stood at different POSITIONS. On
# NOC1 the per-(donor, allele) real-minus-twin increment, independent halves on both sides, repeats at r = 0.556 for
# the n+1 log ratio (stable part 0.27 log), 0.553 for n+1 presence at parents >= 400 RFU and 0.389 for n-1 presence
# at 100-400 RFU (stable part ~0.12 in rate). So the file also carries, per parent allele:
#   "10"              n+1 log-ratio offset, measured like "-10" (both present, parents >= 400 RFU);
#   "lrate-10"/"lrate10" a LOGIT offset on that allele's emission rate, measured AFTER the ratio offsets are in (a
#                     taller stutter already survives more often, so only the presence left over is charged here).
#                     A logit offset, not a factor on the rate: presence is measured at parents where it sits at
#                     0.4-0.6, while the n-1 emission RATE is 0.93, so a factor of 1.5 measured on presence has
#                     nowhere to go on the rate and was clipped. Applied multiplicatively and clipped at 0.99 it
#                     lost the whole >1 side while the <1 side still bit: over the 203 panel bins with an n-1
#                     target the mean rate fell 0.9528 -> 0.8395 with 35 % of bins clipped, and real NOC1 showed
#                     an n-1 at a bright parent 0.89 of the time against the twin's 0.79 - 1.6 peaks per profile,
#                     which the noise floor then over-emitted to cover. On the odds scale both sides are the same
#                     quantity, the offset is bounded by construction and offset 0 reproduces the table exactly.
# STR_ALLELE_STUT=0 turns the ratio offsets off, STR_ALLELE_RATE=0 the rate offsets (calibrate_allele_stut.py uses
# them to measure each as the increment over the rest).
ALLELE_STUT = np.zeros(N_FLAT)
ALLELE_STUT_OFF = {-10: ALLELE_STUT, 10: np.zeros(N_FLAT)}
ALLELE_RATE = {-10: np.zeros(N_FLAT), 10: np.zeros(N_FLAT)}   # LOGIT offsets, 0 = the table's own rate
_ASF = DATA / "allele_stut.json"
if _ASF.exists():
    _asj = json.load(open(_ASF))
    for _d in (-10, 10):
        if int(os.environ.get("STR_ALLELE_STUT", "1")):
            for _b, _v in _asj.get(str(_d), {}).items():
                ALLELE_STUT_OFF[_d][int(_b)] = float(_v)
        if int(os.environ.get("STR_ALLELE_RATE", "1")):
            for _b, _v in _asj.get("lrate%d" % _d, {}).items():
                ALLELE_RATE[_d][int(_b)] = float(_v)
# HOW MUCH DNA IS REALLY THERE. The labelled template is a quantified and pipetted amount; what reaches the PCR differs
# from it per person. Measured on TRAIN NOC1 at the lowest templates (where nearly every profile drops alleles, so the
# presence-based estimate is unselected): the spread of log2(effective / labelled) after removing the estimator's own
# noise, minus what the generator already makes there, is 0.37 for untreated samples, mean 0. Treated samples already
# carry it through the damage range (FILT_S, refitted on the same profiles), so they are not charged again.
# SOME DONORS' ALLELES DO NOT STUTTER. Same allele length, different donor: on TRAIN+VAL NOC1 with parents >= 1200 RFU,
# D2S441 14 shows its n-1 1.00 of the time in donors 15/44/25/35 and .04 in donors 16/36; D5S818 11 1.00 against .00
# (donor 27); TH01 9, D22S1045 16, D19S433 13, D7S820 8, D13S317 8 alike - over 20-50 runs each, and when the n-1 does
# stand its ratio to the parent is the allele's usual one. Split in halves of the runs, the pairs flagged on one half
# stood .17-.19 on the other against the generator's .90-.95. Same-length alleles of different sequence (an
# interrupted repeat, a short longest-uninterrupted stretch) are the known reason. A per-(donor, allele) LOGIT offset
# on the n-1 emission, measured by calibrate_donor_stut.py on NOC1 as the increment over the generator and shrunk to 0,
# weighted by each contributor's share of the parent where several carry it. data/donor_stut.json {donor: {bin: logit}}.
DONOR_RATE = {}
_DSF = DATA / "donor_stut.json"
if _DSF.exists() and int(os.environ.get("STR_DONOR_STUT", "1")):
    DONOR_RATE[-10] = np.zeros((45, N_FLAT))
    for _c, _v in json.load(open(_DSF)).items():
        for _b, _o in _v.items():
            DONOR_RATE[-10][int(_c), int(_b)] = float(_o)
AMOUNT_SD = 0.37
# STUTTER BY PARENT HEIGHT. The table and the laws above get each offset's MEAN ratio right on NOC1 but not how it
# moves with the parent: on held-out TEST NOC1, real-minus-twin log(stutter/parent) when present, at parents
# <300 / 300-1k / 1k-3k / >3k RFU, was n-1 +.14 / .00 / .00 / +.02, n+1 .00 / +.22 / +.12 / -.11, n-2 -.15 / +.16 /
# +.23 / +.07, pooled within +-.02 - and the mixtures showed the same numbers whether one contributor or several
# carried the parent, so it is the law, not the mixing. That shape error is what moved the v2 gate at NOC4/5 (more
# faint parents there). A step increment per parent-height band and offset, one on the log ratio and one on the
# emission LOGIT, fitted jointly on TRAIN+VAL NOC1 by calibrate_stut_parent.py against real's stutter SEPARATED from
# the floor, so that both the height when present and the presence match real in every band (raising the height
# alone lets more faint stutter stand, which is how the earlier STUT_LT_A attempt overshot presence). Steps, not a
# line between band centres: the bands the increment is applied on must be the bands it was measured on, or a large
# value at one band leaks into its neighbour (it did: n-1's +0.87 at 50 RFU put 100-141 RFU parents over real).
# data/stut_parent.json {d: {"edges", "ratio", "logit"}}; STR_STUT_PH=0 turns it off (the calibration measures
# against the generator without it).
STUT_PH = {}
_SPF = DATA / "stut_parent.json"
if _SPF.exists() and int(os.environ.get("STR_STUT_PH", "1")):
    for _d, _v in json.load(open(_SPF)).items():
        STUT_PH[int(_d)] = (np.asarray(_v["edges"], float), np.asarray(_v["ratio"], float),
                            np.asarray(_v["logit"], float))

# NOT EVERY MARKER STUTTERS. Stutter is polymerase slippage on a repeat; AMEL (a 6 bp X/Y length difference) and
# Yindel (a Y-chromosome insertion/deletion) have no repeat to slip on. The artefact table still carried rows for them
# (AMEL n+1 rate .09 - which lands on the Y bin of a female - and Yindel n-1 .045), and the parent-height increments
# (STUT_PH) raised those to .17-.28 at bright parents, where real sits flat at the floor (.09-.15 at every parent
# height; female Y bin .080 at X >= 1200 RFU against the twin's .108). Every other locus of the kit, DYS391 included,
# shows n-1 at .85-1.00 over a floor of .04-.09. The same bins also sat in the NOC1 stutter fits and dragged their
# bright-band target to .88 while the STR positions are at .94-.96. No artefact is emitted from these loci.
NO_STUTTER_LOCI = ("AMEL", "Yindel")
NO_STUTTER_BIN = np.isin(BIN_LOCUS, [LOCUS_TO_IDX[_n] for _n in NO_STUTTER_LOCI if _n in LOCUS_TO_IDX])


# THE SCATTER OF A STUTTER SHRINKS WITH ITS PARENT. The ratio draw's width (run offset, locus offset and per-peak
# lognormal) was one number per locus and offset, fitted on the pooled distribution. Measured on TRAIN+VAL NOC1 at
# identical parents (real alleles injected), the within-(donor, allele) sd of log(n-1/parent) falls with the parent in
# real - .368 / .274 / .220 / .156 / .108 at 150-300 / 300-600 / 600-1.2k / 1.2-3k / >3k RFU - where the generator
# stayed at .33-.25; its sample-level part was twice real's and its per-peak part 1.75x at bright parents. That is the
# early-cycle copy lottery: many copies, little scatter. A factor per parent band on the whole width, fitted by
# calibrate_stut_spread.py (n-1 only; the other offsets are not measured this way). data/stut_spread.json.
STUT_SPREAD = {}
_SSF = DATA / "stut_spread.json"
if _SSF.exists() and int(os.environ.get("STR_STUT_SPREAD", "1")):
    for _d, _v in json.load(open(_SSF)).items():
        STUT_SPREAD[int(_d)] = (np.asarray(_v["edges"], float), np.asarray(_v["g"], float))


def _ph_band(h, edges):
    """index of the parent-height band each parent falls in (STUT_PH)"""
    return np.clip(np.digitize(np.asarray(h, float), edges) - 1, 0, len(edges) - 2)
KAPPA_FIX = None
_KO_OCC_TOTAL = 0      # DIAGNOSTIC: read the crowding curve with the old total-peak abscissa
_INJ_STEP = {tuple(float(_x) for _x in _k.split("|")): tuple(_v)
             for _k, _v in (cal("inj_step", {}) or {}).items()}


def _treat_family(cond):
    cd = "a" if cond is None else str(cond)
    if cd in ("b", "c", "d", "e"):
        return "DNase"
    if cd.startswith("-"):
        return "Fragmentase"
    if cd.startswith("U"):
        return "UV"
    if cd.startswith("S"):
        return "sonication"
    if cd.startswith("I"):
        return "humic"
    return "a"


def _filter_source(c, rng, inj):
    """Donor c's most complete untreated profile: the highest template holding a run at injection >= inj."""
    pl = (noc1_calib().get("pool_lvl") or {}).get(c)
    if not pl:
        return None
    for ent in sorted(pl, key=lambda e: -e[0]):
        if len(ent) < 4:
            return None
        ijs, qs = np.asarray(ent[2], float), np.asarray(ent[3], float)
        ok = np.flatnonzero((ijs >= float(inj)) & np.isfinite(qs))
        if len(ok):
            same = ok[ijs[ok] == float(inj)]
            k = int(rng.choice(same if len(same) else ok))
            return float(ent[0]), np.expm1(np.asarray(ent[1][k], np.float64)), float(ijs[k]), max(float(qs[k]), 1.0)
    return None


def _filter_contrib(c, rng, t, q_tube, cond, inj, bin_size, s_tube=0.0, eff=1.0):
    """(heights, expected heights) of donor c at template t in a tube of quality q_tube, or None."""
    src = _filter_source(c, rng, inj)
    if src is None or DONOR_DOSAGE is None:
        return None
    t_s, z, inj_s, q_s = src
    own = DONOR_DOSAGE[c] > 0
    if own.sum() < 10:
        return None
    dose = np.asarray(DONOR_DOSAGE[c][own], float)
    sz = np.asarray(bin_size, float)[own]
    sz = np.where(sz > 0, sz, float(np.median(sz[sz > 0])) if (sz > 0).any() else 200.0)
    x = (sz - 80.0) / 134.0
    lam_s = t_s * q_s ** (-FILT_A["a"] * x) / FILT_PG * dose
    fam = _treat_family(cond)
    a_t = FILT_A.get(fam, 1.0)
    if fam in FILT_S:
        al_s, be_s, sd_d, _ = FILT_S[fam]
        a_t = a_t * 2.0 ** (al_s + be_s * float(np.log(max(float(q_tube), 1.0))) + s_tube
                            + float(rng.normal(0.0, sd_d)))
    if fam not in FILT_S:
        eff = float(eff) * 2.0 ** float(rng.normal(0.0, AMOUNT_SD))      # this person's amount error (AMOUNT_SD)
    lam_t = float(eff) * float(t) * max(float(q_tube), 1.0) ** (-a_t * x) / FILT_PG * dose
    hs = z[own]
    per = np.where(hs > 0, hs, float(np.median(hs[hs > 0])) if (hs > 0).any() else 0.0) / np.maximum(lam_s, 1e-9)
    # ONE DRAW DECIDES PRESENCE, ANOTHER DECIDES HEIGHT. Whether an allele stands at all is an early-cycle event
    # of its own - real NOC1 loses ONE allele of a pair far more often than both (0.359 against 0.204 at the
    # faintest band), so that draw has to stay per allele. But once both stand, their heights come from the same
    # genomes, so their copy numbers move together: on bright real NOC1 the within-pair spread is 0.192 against
    # 0.605 between loci, while drawing the height from the same per-allele Poisson put the twin at 0.246 and cost
    # it the heterozygote balance (0.806 against 0.838). HEIGHT_SHARE is how much of the height draw is taken from
    # the locus's shared count instead of the allele's own; presence is untouched either way.
    k = rng.poisson(lam_t)
    k_h = k
    if HEIGHT_SHARE > 0:
        _loc = np.asarray(BIN_LOCUS, int)[own]
        _sh = np.zeros(len(lam_t))
        for _L in np.unique(_loc):
            _sel = _loc == _L
            _sh[_sel] = rng.poisson(float(np.mean(lam_t[_sel])))
        # ...and how much is shared GROWS with the copy number, because that is what the physics says: at a few
        # copies the early-cycle lottery of each allele dominates and the two drift apart, at many copies both
        # alleles are just counting the same genomes. w = lam / (lam + HEIGHT_SHARE_C).
        _w = HEIGHT_SHARE * lam_t / (lam_t + HEIGHT_SHARE_C)
        k_h = np.maximum((1.0 - _w) * k + _w * _sh, 0.0)
    ksum = np.where(k > 0, rng.gamma(np.maximum(k_h, 1e-6) / FILT_CV ** 2, FILT_CV ** 2), 0.0)
    _st = (1.0, 0.0, 0.0) if inj_s == float(inj) else _INJ_STEP.get((inj_s, float(inj)), (1.0, 0.0, 0.0))
    f = _st[0]
    sd_run = _st[1] if len(_st) > 2 else 0.0                 # one capillary event for the whole profile...
    sd_al = _st[2] if len(_st) > 2 else (_st[1] if len(_st) > 1 else 0.0)   # ...and what each allele does on top
    _jit = np.exp(rng.normal(0.0, sd_run)) if sd_run > 0 else 1.0
    h = per * ksum * f * _jit * (np.exp(rng.normal(0.0, sd_al, len(k))) if sd_al > 0 else 1.0)
    H = np.zeros(N_FLAT); Hm = np.zeros(N_FLAT)
    H[own] = np.where(k >= 1, h, 0.0)
    Hm[own] = per * lam_t * f
    return H, Hm


def _lvl_var(t, inj=None):
    """Extra log-height variance at template t relative to the top of the ladder, on the chain for this injection:
    linear in log t between rungs, one floor step per doubling below the lowest, 0 above the highest."""
    vv = AMP_STEP_VAR_INJ.get(float(inj), AMP_STEP_VAR) if inj else AMP_STEP_VAR
    t = max(float(t), 1e-9)
    if t >= AMP_STEP_T[-1]:
        return float(vv[-1])
    if t <= AMP_STEP_T[0]:
        return float(vv[0] + (vv[0] - vv[1]) * np.log2(AMP_STEP_T[0] / t))
    return float(np.interp(np.log(t), np.log(AMP_STEP_T), vv))


def _lvl_tilt(t):
    """Size-slope offset (log RFU per bp) at template t relative to the top of the ladder: linear in log t
    between rungs, one floor step per doubling below the lowest, 0 above the highest."""
    t = max(float(t), 1e-9)
    if t >= AMP_STEP_T[-1]:
        return float(AMP_STEP_TILT[-1])
    if t <= AMP_STEP_T[0]:
        return float(AMP_STEP_TILT[0] + (AMP_STEP_TILT[0] - AMP_STEP_TILT[1]) * np.log2(AMP_STEP_T[0] / t))
    return float(np.interp(np.log(t), np.log(AMP_STEP_T), AMP_STEP_TILT))


def _lvl_surv(t, inj=None):
    """Survival at template t relative to the top of the ladder, on the chain for this injection: geometric
    between levels, one floor step per doubling below the lowest, 1 above the highest."""
    ph = AMP_STEP_PHI_INJ.get(float(inj), AMP_STEP_PHI) if inj else AMP_STEP_PHI
    t = max(float(t), 1e-9)
    if t >= AMP_STEP_T[-1]:
        return float(ph[-1])
    if t <= AMP_STEP_T[0]:
        s0 = float(ph[0] / max(ph[1], 1e-9))
        return float(ph[0] * max(s0, 1e-6) ** float(np.log2(AMP_STEP_T[0] / t)))
    return float(np.exp(np.interp(np.log(t), np.log(AMP_STEP_T), np.log(ph))))


# Amplification failure is not independent between contributors. Measured on NOC1 by asking how
# often BOTH alleles of a HETEROZYGOUS locus fall together - two sets of molecules in one reaction at
# one locus, which is the same configuration as two contributors sharing a bin. They fall together
# far more often than independence allows: p_both 0.0722 against p^2 0.0333 at the second template
# quintile. Pairs drawn from DIFFERENT loci of the same sample show only about a third of that
# surplus (+0.0141 against +0.0390), so most of it belongs to the locus, not to the sample.
# Solving the three-level model - sample S, locus L, molecule u - leaves L/(L+u) at 0.299 / 0.203 /
# 0.260 / 0.222 over the four quintiles that carry any dropout at all: flat across four decades of
# template, which is what a primer-efficiency effect should look like and what a template effect
# should not. The fifth quintile is dropped from the average; it holds almost no dropouts at all.
AMP_SHARE_R = float(cal("amp_share_r", 0.246))


def _amp_split(q):
    """Split a measured total failure q into its locus-shared L and per-molecule u parts.

    Holds (1 - L)(1 - u) = 1 - q with L / (L + u) pinned at the NOC1 ratio, so a single-source
    sample reproduces the measured q(f) exactly and no second constant is introduced.
    """
    r = AMP_SHARE_R
    d = r * (1.0 - r)
    if q <= 0.0 or d <= 0.0:
        return 0.0, max(float(q), 0.0)
    m = (1.0 - float(np.sqrt(max(1.0 - 4.0 * d * float(q), 0.0)))) / (2.0 * d)
    return r * m, (1.0 - r) * m
AMP_PRODUCT = os.environ.get("STR_AMP_PRODUCT", "1") == "1"

INJ_REF = float(cal('inj_ref', 15.0))


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

# _CARR_MULT normalised here, where the panel and the offset targets both exist: see its definition.
_CW = np.zeros(N_FLAT)
if DONOR_DOSAGE is not None and -10 in _OFF_TARGET:
    _t1 = _OFF_TARGET[-10]; _cn = np.asarray(DONOR_DOSAGE, float).sum(0)
    _ok = (_t1 >= 0) & (_cn > 0)
    np.add.at(_CW, _t1[_ok], _cn[_ok])
    if _CW.sum() > 0:
        _cm = float(np.average(_CARR_MULT, weights=_CW))
        if _cm > 0:
            _CARR_MULT = _CARR_MULT / _cm


# The balanced regime gen_conditioned mixes in (see GEN_PHI_WIDE).
REAL_DIRICHLET_ALPHA = 1.0                                  # Dirichlet(1): uniform over every split of the shares
REAL_EQUAL_FRAC = 0.25                                      # of the balanced tubes, the share drawn exactly equal
WIDE_REAL_FRAC = 0.5                                        # fraction of tubes drawn from the balanced regime
# Template spread of the ORIGINAL mode. sigma=0.55 gives p90/p10 = exp(2*1.2816*0.55) = 4.1x, but real
# test mixtures span 19-34x (NOC2..5: 33.7 / 19.3 / 24.5 / 23.7; NOC1 matches at 60.6 vs 69.7 only
# because synth NOC1 IS real NOC1). Consequence: real NOC5 has p10 = 7.6k RFU while synth NOC5 has
# p10 = 16.0k, so the generator barely reaches the low-template band where the model actually fails.
# sigma = ln(24)/(2*1.2816) = 1.24 reproduces the observed spread. The old 0.55 was never derived from
# anything measured; a narrow template distribution is the unrealistic choice, since casework DNA
# quantity varies by orders of magnitude.
TTOTAL_SIGMA = float(os.environ.get("STR_TTOTAL_SIGMA", "0.55"))

def build_ss_pool():
    """donor_col -> list of single-source Xflat rows (log1p)."""
    X = np.load(DATA / "Xflat_train.npy").astype(np.float32)
    y = np.load(DATA / "y_train_set.npy")
    n = np.load(DATA / "noc_train.npy")
    ss = n == 1
    Xs, ds = X[ss], y[ss].argmax(1)
    pool = {c: Xs[ds == c] for c in range(45)}
    return pool


_BSZ = build_bin_size()


def _renorm_tpl(x, tpl, c=None):
    """Put a clean profile on the reference template's footing.

    Injection time needs no correction - it is structurally inert (1.01x) because it only decides how
    much finished product enters the capillary. Template does: at low copy number amplification is a
    lottery, so a faint run carries extra scatter that a real mixture's contributors, sharing ONE
    amplification, do not have against each other. The size trend is what damage and fragment length
    wrote and is kept; only the residual around it is rescaled, by the measured factor for this
    profile's distance from the reference (0.893 per doubling, 0.498 at 64x).
    """
    cal_ = noc1_calib()
    sh = cal_.get("tpl_shrink")
    ref = cal_.get("tpl_ref")
    if not sh or not ref or tpl <= 0 or tpl >= ref:
        return x
    kk = sorted(sh)
    m = float(np.interp(np.log2(ref / tpl), kk, [sh[k] for k in kk]))
    h = np.expm1(x.astype(np.float64))
    # Only the donor's OWN alleles. tpl_shrink was measured on heterozygote pairs, i.e. on alleles,
    # and the source profile's artefacts sit far below the size trend; pulling them toward it along
    # with the alleles changes the balance between the two groups and tilts the allele-only slope even
    # when the all-bin slope is held fixed. That is what pushed the untreated slope from 0.00064 to
    # 0.00111 while a trend correction fitted over all bins left it untouched.
    nz = h > 0
    if c is not None and DONOR_DOSAGE is not None:
        nz = nz & (DONOR_DOSAGE[c] > 0)
    if nz.sum() < 12:
        return x
    bz = np.polyfit(_BSZ[nz], np.log(h[nz]), 1)
    tr = np.polyval(bz, _BSZ[nz])
    h2 = h.copy()
    h2[nz] = np.exp(tr + m * (np.log(h[nz]) - tr))
    # Restore the size trend explicitly. Compressing the residual keeps the trend, but rescaling to
    # the original total afterwards does not: the residual distribution differs between short and long
    # fragments, so a single multiplier tilts the profile. Measured, it steepened the size slope from
    # 0.00064 to 0.00111, a 74% error that showed up hardest in the conditions where no damage is
    # applied at all - untreated and humic acid ran 1.56-1.70x against real while sonication at 30
    # cycles, where the real slope is twenty times larger, ran 1.04x.
    h2[nz] *= h[nz].sum() / max(h2[nz].sum(), 1e-9)
    b2 = np.polyfit(_BSZ[nz], np.log(h2[nz]), 1)
    h2[nz] *= np.exp((bz[0] - b2[0]) * (_BSZ[nz] - _BSZ[nz].mean()))
    h2[nz] *= h[nz].sum() / max(h2[nz].sum(), 1e-9)
    return np.log1p(h2).astype(np.float32)


def _clean_level(rng):
    """Pick the template level this mixture's contributors all share."""
    pc = noc1_calib().get("pool_clean")
    if not pc:
        return None
    return list(pc)[int(rng.integers(len(pc)))]


def _relottery(x, c, rng, tpl=None, cond=None):
    """Draw again WHICH of donor c's alleles this amplification kept, at the source's own rate."""
    if DONOR_DOSAGE is None or not RELOTTERY:
        return x
    h = np.expm1(np.asarray(x, dtype=np.float64))
    own = DONOR_DOSAGE[c] > 0
    n = int(own.sum())
    if n < 12:
        return x
    have = own & (h > 0)
    r = float(have.sum()) / n
    if r <= 0.0 or r >= 1.0:
        return x                                   # nothing lost: no lottery to redraw
    cal_ = noc1_calib()
    A = cal_.get("cond_allele_mean") or {}
    Pd = (cal_.get("donor_dev_log") or {}).get(c) or (cal_.get("donor_dev_log") or {}).get(str(c)) or {}
    if not A or not Pd:
        return x
    band = int(np.searchsorted([0, 0.02, 0.06, 0.15, 1e9], float(tpl if tpl else 0.25), "right")) - 1
    idx = np.where(own)[0]
    key = [f"{int(BIN_LOCUS[j])}_{int(round(float(BIN_ALLELE[j]) * 10))}" for j in idx]
    cd = str(cond) if cond is not None else "a"
    sysv = np.array([A.get(f"{cd}|{band}|{k_}", A.get(f"a|{band}|{k_}", np.nan)) + Pd.get(k_, 0.0) for k_ in key])
    ok = np.isfinite(sysv)
    if ok.sum() < 12:
        return x
    # Only alleles the tables can put back may be drawn at all. An allele with no modelled height could be
    # killed but never resurrected, and that one-sided draw is what pushed the NOC1 centre down by 0.008.
    idx, sysv = idx[ok], sysv[ok]
    r = float((h[idx] > 0).mean())
    if r <= 0.0 or r >= 1.0:
        return x
    mu = float(np.log(h[have]).mean())
    # Probability this allele is one of the ones kept: mean exactly the source's retention - so the count this
    # amplification produced is unchanged - shaped by WHICH allele is likely to go. The shape is the measured
    # dropout curve against the allele's own RFU (drop_h / drop_p), read at its height where the source has one
    # and at its modelled height where it does not, so a tall allele is essentially never the one dropped.
    _hk = np.where(h[idx] > 0, h[idx], np.exp(mu + sysv))
    _sv = np.clip(1.0 - np.interp(_hk, DROP_H, DROP_P), 1e-3, 1.0 - 1e-3)
    _lg = np.log(_sv / (1.0 - _sv)); _lg = _lg - float(_lg.mean())
    # ...and only as steeply as the data says. The spread of p decides how much the count can move: the variance
    # of the retention is mean p(1-p)/n_eff against r(1-r)/n_eff if every allele were alike, and real's ratio of
    # the two is DROP_DISP. The curve gives the direction, this gives the length.
    # The tilt is bounded so no allele is pinned at 0 or 1: a saturated p cannot be moved back to the mean and
    # the count starts drifting (-0.004 at the source before the bound went in).
    _tgt = DROP_DISP * r * (1.0 - r)
    _lo, _hi = 0.0, 6.0
    p = _pin_mean(r, np.clip(_lo * _lg, -4.0, 4.0))
    if float((p * (1.0 - p)).mean()) > _tgt:
        for _ in range(20):
            _md = 0.5 * (_lo + _hi)
            p = _pin_mean(r, np.clip(_md * _lg, -4.0, 4.0))
            if float((p * (1.0 - p)).mean()) > _tgt:
                _lo = _md
            else:
                _hi = _md
        p = _pin_mean(r, np.clip(0.5 * (_lo + _hi) * _lg, -4.0, 4.0))
    # One locus fails together: with probability AMP_SHARE_R an allele takes its locus's draw rather than
    # its own, which is what turns n into n_eff = n / (1 + rho).
    u = rng.random(len(idx))
    ul = rng.random(int(BIN_LOCUS.max()) + 2)[BIN_LOCUS[idx]]
    u = np.where(rng.random(len(idx)) < AMP_SHARE_R, ul, u)
    keep = u < p
    off = (rng.normal(0.0, REDRAW_LOC, size=int(BIN_LOCUS.max()) + 2)[BIN_LOCUS[idx]]
           + rng.normal(0.0, REDRAW_ALL, size=len(idx)))
    h2 = h.copy()
    h2[idx[~keep]] = 0.0
    _back = keep & (h[idx] <= 0)
    if _back.any():
        # An allele the draw brings back is a MARGINAL one - this amplification kept it and another lost it - so it
        # does not stand at its donor's usual height. Its height is drawn from the modelled one and then weighted by
        # drop(h) * (1 - drop(h)), the chance the measured curve gives an allele of being on exactly that edge; put
        # at the modelled height instead it stood at the profile's own average, which flattened the twin's height
        # scatter (0.687 -> 0.654 against real's 0.702) and its stutter with it.
        _ix = idx[_back]
        _cand = np.exp(mu + sysv[_back][:, None] + off[_back][:, None]
                       + rng.normal(0.0, REDRAW_ALL, size=(len(_ix), 8)))
        _d = np.interp(_cand, DROP_H, DROP_P)
        _w = _d * (1.0 - _d)
        _sw = _w.sum(1)
        _pick = np.where(_sw[:, None] > 0, np.cumsum(_w, 1) >= (rng.random(len(_ix)) * np.maximum(_sw, 1e-12))[:, None],
                         True).argmax(1)
        _h = _cand[np.arange(len(_ix)), _pick]
        _h50 = float(np.interp(0.5, DROP_P[::-1], DROP_H[::-1]))
        h2[_ix] = np.maximum(np.where(_sw > 0, _h, _h50), AT_PC)
    return np.log1p(h2).astype(np.float32)

_MATCH = [False]        # did the last _clean_source find a source under the SAME treatment?
_LVLM = [False]         # ...and was it drawn from the template ladder at all?
_SRCL = [None]          # ...and at which template level


def _source_pool(c, cond):
    """The ladder a draw for donor c comes from: this treatment's if the donor has one, else untreated."""
    pl = ((noc1_calib().get("pool_cond") or {}).get(str(cond)) or {}).get(c) if cond is not None else None
    if pl:
        return pl, True
    return (noc1_calib().get("pool_lvl") or {}).get(c), False


def _source_level(c, t_eff, cond):
    """The level the draw takes: the nearest at or ABOVE the target, else the highest there is.

    Above, because a richer source can be taken down - its extra alleles are charged at the measured
    conversion rate - while a poorer one cannot be brought up: what it failed to amplify is gone. The
    nearest-level rule broke ties toward the poorer side, and a tie is exactly the honest case: with
    the target's own amplification excluded, the candidates sit one doubling either side."""
    pl, _ = _source_pool(c, cond)
    if not pl or t_eff is None or t_eff <= 0:
        return None
    # 'At' means within 3%: spec names round the rungs (0.0313 for 1/32 ng), and without the tolerance a
    # target written 0.0313 finds the 0.03125 rung below it and is taken down from the next one up.
    up = [e[0] for e in pl if e[0] >= float(t_eff) * 0.97]
    return float(min(up)) if up else float(max(e[0] for e in pl))


def _clean_source(c, pool, rng, level=None, t_eff=None, cond=None, inj_sec=None):
    """Cleanest ACTUAL profile, not a smoothed one. Compressing a faint run's residual onto its own
    trend removes the sampling scatter and the donor's real per-locus structure together - there is no
    way to tell them apart inside one profile - so the sample stops being a sample. At the reference
    template no compression is needed at all, and PROVEDIt carries all 45 donors there."""
    """An UNTREATED profile of donor c, renormalised onto the reference template."""
    if t_eff is not None and t_eff > 0:
        # Take a REAL profile at the template this contributor actually has, and do not renormalise
        # it onto the reference. Whatever that profile has already lost, the instrument lost - which
        # at NOC1 is the whole of the story, since one contributor cannot interact with anything.
        # Matched on the TREATMENT first, because a treatment destroys alleles and the degradation
        # model only tilts heights; falls back to the untreated ladder for donors with no run under
        # that condition, which is why the fallback exists rather than being an oversight.
        pl, _cm = _source_pool(c, cond)
        _MATCH[0] = bool(_cm)
        if pl:
            _L = _source_level(c, t_eff, cond)
            _ent = next(e for e in pl if abs(e[0] - _L) < 1e-12)
            L, arr, ijs = _ent[0], _ent[1], _ent[2]
            if len(arr):
                # ...and matched on the INJECTION too, because what a profile has lost depends on how
                # many molecules reached the capillary, not on loaded mass alone. Real retention at
                # 0.0078 ng is 0.644 / 0.728 / 0.771 at 5 / 15 / 25 sec - a spread of 0.126 against a
                # replicate envelope of 0.0343 - and 0.068 at 0.0156 ng, while every level from 0.031
                # ng up agrees to 0.006. Falls back to the whole cell when this donor has no run at
                # the target's injection, which is the same rule the condition match already uses.
                idx = np.flatnonzero(ijs == float(inj_sec)) if inj_sec else np.array([], int)
                z = arr[idx[rng.integers(len(idx))]] if len(idx) else arr[rng.integers(len(arr))]
                _LVLM[0] = True
                _SRCL[0] = float(L)
                return _relottery(z, c, rng, L, cond)
    pc = noc1_calib().get("pool_clean_all") or {}
    lst = pc.get(c)
    if not lst:
        return pool[c][rng.integers(len(pool[c]))]
    tot = sum(len(a) for _, a in lst)
    j = int(rng.integers(tot))
    for tpl, arr in lst:
        if j < len(arr):
            return _renorm_tpl(arr[j], tpl, c)
        j -= len(arr)
    return pool[c][rng.integers(len(pool[c]))]


def gen_mixture(donor_cols, pool, rng, t_total=None, bin_size=None, phi=None,
                q_target=None, cond=None, ng_total=None, inj_sec=None, _depth=0, amp_cache=None, q_tube=None,
                kappa=None, stut_lt=None, dropin=None, pullup=None, det_f=None, stut_drop=None):
    """phi lets a caller reproduce a specific real mixture through THIS code path rather than
    reimplementing the physics - a twin built from a copy of the loop tests the copy, not the
    generator."""
    k = len(donor_cols)
    t_given = t_total is not None
    phi_given = phi
    if q_target is None:
        _qa = noc1_calib().get('q_all')
        q_target = float(rng.choice(_qa)) if _qa is not None and len(_qa) else None
    # Skewed shares (the skew grows with k). The draw is made even when the caller passes phi: it advances the random
    # stream, so a build with a given seed stays the build it always was.
    phi = _skew_phi(k, rng)
    if t_total is None:
        t_total = float(np.exp(rng.normal(np.log(32000), TTOTAL_SIGMA)))
    if phi_given is not None:
        phi = np.asarray(phi_given, dtype=float); phi = phi / phi.sum()
    if not t_given:
        # Total signal is not a free parameter: each contributor brings its own template, and the
        # single-source fit says RFU is linear in both total ng and injection time. Drawing k template
        # amounts from the observed single-source pool therefore makes t_total scale with contributor
        # count for free — the shipped k-independent lognormal made NOC1 2.0x too loud and NOC5 too
        # quiet from the same constant.
        cal = noc1_calib()
        if cal.get("template_ng") is not None and len(cal["template_ng"]):
            ng_tot = float(rng.choice(cal["template_ng"], size=k).sum())
            inj = float(rng.choice(cal["inj_levels"]))
            t_total = float(np.exp(RFU_LOG_C + 0.976 * np.log(max(ng_tot, 1e-6))
                                   + 0.948 * np.log(inj) + rng.normal(0.0, RFU_LOG_SD)))
    # Build at the reference injection; the gain goes back on after the artefacts, before the
    # capillary's own absolute-scale steps.
    _gain = ((float(inj_sec) / INJ_REF) ** RFU_INJ_BETA) if inj_sec else 1.0
    mix = np.zeros(N_FLAT, dtype=np.float64)
    contrib = np.zeros((k, N_FLAT), dtype=np.float64)        # per-donor RFU contribution per bin
    contrib_exp = np.zeros((k, N_FLAT), dtype=np.float64)   # the same before each allele's own draw
    contrib_pre = np.zeros((k, N_FLAT), dtype=np.float64)   # ...and before the dropout, for the share check
    # One shared locus-efficiency pattern for the whole mixture: divide out the pattern each source
    # profile brought from its own run, then impose a single drawn one on all contributors. Applying a
    # pattern without removing the old one would double-count it, which is the mistake that already
    # cost the degradation channel and the jitter channel.
    # t_total as an OUTPUT, not an input. The physical order is mass -> degradation -> amplification
    # -> detection, so the signal level is produced by the loaded mass and the injection, not imposed
    # on the profile before any of that happens. Imposing it is what forced the fixed-point budget
    # patch: t_total is an observed, post-threshold quantity being spent as a pre-threshold one.
    _lvl = _clean_level(rng)
    # One loss tendency for the whole sample, on top of the per-allele law: measured over-dispersion
    # of the per-profile dropout rate against binomial on NOC1 untreated.
    _t_in = t_total if (t_given and t_total is not None and t_total > 0) else None
    if BUDGET_SOLVE and t_given and t_total is not None and t_total > 0:
        # t_total is an OBSERVED total: it already has the artefacts and the noise floor in it, and it
        # is already net of everything that dropped. So the budget handed to the alleles has to be the
        # one whose SURVIVING mass, plus artefacts, plus floor, comes back to t_total:
        #     observed = B * surv * (1 + a) + N   ->   B = (t_total - N) / ((1 + a) * surv)
        # surv is computable before anything is drawn: the conversion removes the same fraction of every allele
        # of a contributor - they all enter at one copy number - so the mass it takes is that
        # contributor's share times q. Without this the twin's total came out 10% light (z +27).
        _surv = 1.0
        if ng_total is not None and ng_total > 0:
            # Only the CONVERSION loss: a source at the target's own level carries its dropout
            # already, and so does the observed total it is being scaled to.
            _sv = 0.0
            for _dc, _p in zip(donor_cols, np.asarray(phi, dtype=float)):
                _t = float(_p) * float(ng_total)
                _L = _source_level(_dc, _t, cond)
                _fac = (min(_lvl_surv(_t, inj_sec) / max(_lvl_surv(_L, inj_sec), 1e-9), 1.0)
                        if _L is not None and _L > _t else 1.0)
                if cond is not None and str(cond) != "a" and not _source_pool(_dc, cond)[1]:
                    _fac *= _treat_mean(cond, _t)
                _sv += float(_p) * _fac
            _surv = float(np.clip(_sv, 0.2, 1.0))
        t_total = max((float(t_total) - ART_MASS_N) / ((1.0 + ART_MASS_A) * _surv),
                      0.05 * float(t_total))
    _tsh = noc1_calib().get("tpl_shrink")
    _tref = noc1_calib().get("tpl_ref")
    _t_obs = t_total
    if _gain != 1.0:
        t_total = t_total / _gain
    _pb_sum = 0.0
    # ONE degradation draw for the whole sample. The tube was treated as a unit, so every contributor
    # in it carries the same damage; and the treatment's slope is not a single number but a
    # distribution, whose WIDTH comes from the damage mechanism (see cond_sd in calibrate.py). Left
    # as the fixed per-code value the twin had exactly zero between-sample degradation variance
    # against real's 0.00035-0.00306, and its mean landed on real's median rather than real's mean.
    _bsamp = None
    if cond is not None:
        _cb = noc1_calib().get("cond_beta", {})
        if str(cond) in _cb:
            _bmu = float(_cb[str(cond)]) + DEG_BASE
            _bsd = float(noc1_calib().get("cond_sd", {}).get(str(cond), 0.0))
            if _bsd > 0.0 and _bmu > 0.0:
                # GAMMA, not a normal clipped at zero. The clip bit exactly the low-dose treatments -
                # DNase b has sd/beta = 0.67, so 6.9% of normal draws landed below zero and were
                # truncated, and regenerating one sample 200 times returned only 0.75 of the spread
                # asked for against the 0.94 the clip alone predicts. Gamma wins on three separate
                # grounds: it is strictly positive so nothing is truncated; it is the continuous form
                # of a Poisson count, which is what the scission law says beta is; and it is
                # right-skewed, which is what real shows - mean steeper than median by 0.00035 /
                # 0.00022 / 0.00023 at c / d / e.
                _sh = (_bmu / _bsd) ** 2
                _bmu = float(rng.gamma(_sh, _bmu / _sh))
            _bsamp = max(_bmu, 0.0) - DEG_BASE
    # Drawn ONCE for the whole sample: a locus that fails to amplify fails for everyone in the tube.
    # Its rate is read at the sample's TOTAL template, not at any one contributor's share, because
    # what amplifies at a locus is every contributor's template together.
    # ONE common shock per locus, compared against EACH contributor's own L_c. Drawing a ready-made
    # mask at the sample's total template was wrong: it evaluates L at f_tot while u is evaluated at
    # f_c, so the constraint (1-L)(1-u) = 1-q holds at neither, and raising the shared fraction moved
    # failure mass out of a large u into a small L - single-carrier survival went UP when a stronger
    # shared component can only leave it unchanged. With the shock coupled comonotonically a single
    # carrier gets L_c + (1-L_c)u_c = q(f_c) exactly, so NOC1 is reproduced whatever r is, and r is
    # identifiable ONLY from bins that more than one contributor carries - which is what it describes.
    _ushock = None
    if ng_total is not None and ng_total > 0:
        _ushock = rng.random(int(BIN_LOCUS.max()) + 1)[BIN_LOCUS]
    # Generate-then-filter needs the tube's Q, the template and the injection (see FILTER).
    _filt = bool(FILTER and q_tube is not None and np.isfinite(q_tube) and ng_total and inj_sec
                 and bin_size is not None)
    _FH = np.zeros((k, N_FLAT)); _FHm = np.zeros((k, N_FLAT))
    if _filt:
        # Decided for the whole tube at once: if any donor has no complete source every contributor takes the old
        # path, so a sample is never half one model and half the other.
        _fs = FILT_S.get(_treat_family(cond))
        _s_tube = float(rng.normal(0.0, _fs[3])) if _fs else 0.0     # one tube, one draw (see FILT_S)
        # shares as the cells that actually went in (see CELL_R)
        _ph = np.asarray(phi, float)
        if k >= 2:
            _R = float(np.exp(rng.uniform(np.log(CELL_R[0]), np.log(CELL_R[1]))))
            _nc = rng.poisson(_R * _ph * float(ng_total) / FILT_PG).astype(float)
            if _nc.sum() > 0:
                _ph = _nc / _nc.sum()
        _kap = (float(kappa) if kappa is not None else
                float(KAPPA_FIX) if KAPPA_FIX is not None else float(rng.uniform(0.0, KAPPA_MAX)))
        _eff = 2.0 ** (-_kap * (k - 1))                       # one tube, one efficiency (see KAPPA_MAX)
        for _di, (_dc, _dp) in enumerate(zip(donor_cols, _ph)):
            _fr = _filter_contrib(_dc, rng, float(_dp) * float(ng_total), q_tube, cond, inj_sec, bin_size, _s_tube,
                                  _eff)
            if _fr is None:
                _filt = False
                break
            _FH[_di], _FHm[_di] = _fr
    for di, (c, p) in enumerate(zip(donor_cols, phi)):
        # Per-contributor degradation comes from WHICH real profile is drawn, not from a tilt applied
        # on top of one. The overlay's source profiles already carry their own degradation, so
        # multiplying by exp(-beta*(size-100)) counted it twice: it halved the median true-peak height
        # (188 against real's 356) and raised within-profile concentration to 4.86 against real's 3.27.
        # Drawing a target Q per contributor and taking the nearest observed profile gives the same
        # per-contributor variation using measured degradation instead of a modelled one.
        # Profiles are drawn uniformly. An earlier version picked the profile whose Q index was nearest
        # a drawn target, meaning to give each contributor its own degradation; measured one component
        # at a time on the digital twin it was the single most damaging thing in the generator - count
        # accuracy 0.553 with it against 0.869 without, where real is 0.870, while every other component
        # moved the result by at most 6 points. argmin returns ONE profile per target, so the same few
        # profiles recur and contributors of one mixture get forced to unrelated quality levels; that is
        # the source of the strongest/weakest tail of 24.5 against real's 8.7. Real degradation variety
        # is already in the pool and uniform sampling reaches it.
        _teff = (float(p) * float(ng_total)) if ng_total else None
        if _filt:
            continue                            # built before the loop
        _MATCH[0] = False
        _LVLM[0] = False
        _SRCL[0] = None
        prof = _clean_source(c, pool, rng, _lvl, t_eff=_teff, cond=cond, inj_sec=inj_sec)
        _cmatch, _lmatch, _lsrc = _MATCH[0], _LVLM[0], _SRCL[0]
        h = np.expm1(prof.astype(np.float64))
        _hsrc = h.copy()          # the source profile's OWN RFU, for the jitter increment below
        if DONOR_DOSAGE is not None:
            # A source profile carries the donor's alleles AND that run's artefacts. Only the alleles
            # belong to the contributor: baseline noise is a property of the capillary run, generated
            # from the total template, so it does not shrink when a contributor is minor. Scaling the
            # whole profile by phi shrank the artefacts too - in a NOC5 twin they fell from ~9 RFU to
            # ~3 and were cut by the threshold, leaving 20.3 artefacts against real's 40.2 and lifting
            # the height p10 from 11 to 65. Keeping only the alleles here and adding one noise floor at
            # mixture level below puts each phenomenon on the quantity it actually scales with.
            h = h * (DONOR_DOSAGE[c] > 0)
        s = h.sum()
        if s <= 0:
            continue
        h = h / s
        _nzs = h > 0                      # dung o ca khoi phan huy lan khoi neo ben duoi
        # Skipped when the source already came from THIS treatment: it then carries the treatment's
        # own slope, drawn from that treatment's real spread, and imposing a fresh one throws the real
        # value away and substitutes a modelled draw. Kept for the fallback case - only 7 to 31 donors
        # have runs under each treatment against 45 untreated - where the source is untreated and the
        # slope genuinely has to be put there. Measured with the draw off: paired |twin-real| on the
        # dynamic range 0.6905 against 0.7468, on the bp slope 0.00169 against 0.00204.
        if bin_size is not None and cond is not None and not _cmatch:
            # Damage as a per-bin multiplier is ORDER-INVARIANT: scaling each contributor and summing
            # equals scaling the sum, so where it is applied cannot be read off the twin. Applying it
            # to the summed mixture instead was tried and moved nothing except the per-contributor
            # renormalisation, which is worth keeping - t_total is an observed total, and letting the
            # damage drain it as well charges the loss twice. The saturation the paired mixtures show
            # (2.20x peak-spread widening on a pristine profile against 1.17x on a skewed one) is a
            # property of the chemistry, not of the order, and a multiplier of this form cannot
            # reproduce it from either side.
            _b = _bsamp
            if _b is not None:
                # cond_beta is the treatment's WHOLE slope, not an increment on top of one. Measured
                # per treatment on NOC1, slope + cond_beta comes out at -0.00063 +/- 0.00027 for all
                # seventeen of them, from untreated 'a' (-0.00056, beta 0) to S30 (-0.01272, beta
                # 0.01175). The source profile drawn from the pool already carries its own treatment's
                # slope, so multiplying beta on top adds a second one: the twin's mixtures come out at
                # -0.00599 against real's -0.00423, 1.42x too steep, and the excess grows with
                # contributor count (1.225 / 1.276 / 1.387 / 1.395 at NOC2-5) because every
                # contributor brings a source AND gets beta applied. Flatten the source's own slope
                # first, then impose the target's - the same divide-out-then-impose the locus pattern
                # already uses, and the same double-count the degradation channel was warned about.
                _sl = 0.0
                if DEG_RESET and _nzs.sum() >= 12:
                    # fitted on the SAME variable the imposition uses - max(bp-100, 0), not raw bp.
                    # Mixing the two left a third of the double-count in place (slope ratio 1.336).
                    _xb = np.maximum(bin_size[_nzs] - 100.0, 0.0)
                    if float(_xb.std()) > 1.0:
                        _sl = float(np.polyfit(_xb, np.log(h[_nzs]), 1)[0])
                _tot = _sl + _b + DEG_BASE          # flatten by _sl, then impose -(beta + baseline)
                if abs(_tot) > 1e-12:
                    h = h * np.exp(-_tot * np.maximum(bin_size - 100.0, 0.0))
                _s2 = h.sum()
                if _s2 > 0:
                    h = h / _s2
        # Skipped when the source came from the template ladder, which is the fourth instance in this
        # file of the same double count. The block exists because the source used to be renormalised
        # ONTO THE REFERENCE by _renorm_tpl, so a contributor sitting at a fraction p of the mixture
        # needed its spread widened back out to its own, lower template. A ladder-matched source is
        # already AT that template and already carries that spread. Widening it again multiplies the
        # curve by itself: _m is 1.24 at share 0.3 and 1.66 at share 0.1, and it is invisible at NOC1
        # because p = 1 gives _m = 1 exactly - which is why switching TPL_SPREAD off measured
        # byte-identical three times running and was read, three times, as the block being inert.
        # On real mixtures the cost is the whole height defect: sd log 1.8x the replicate envelope and
        # dynamic range 1.9x, where summing the donors' real single-source runs with nothing modelled
        # at all reads 0.2x and 0.1x. Kept for the pool_clean_all fallback, where the premise still
        # holds and the spread genuinely has to be put back.
        if (TPL_SPREAD in ("1", "2", "4")) and not _lmatch and _tsh and _tref and t_total > 0:
            # This contributor sits at p of the mixture, i.e. at its OWN template, which is below the
            # reference the source profile was put on. Widen its log spread by the same curve, read
            # backwards, and let the sum move: the boost that falls out IS the minor's lift.
            _rf = max(float(p) * float(t_total), 1e-9) / max(float(t_total), 1e-9)
            _dbl = np.log2(max(_rf, 1e-12)) / 0.976        # doublings of TEMPLATE, via RFU ~ ng^0.976
            _kk = sorted(_tsh)
            _m = float(np.interp(abs(_dbl), _kk, [_tsh[_q] for _q in _kk])) if _dbl < 0 else 1.0
            _m = 1.0 / max(_m, 1e-3)                        # going DOWN in template WIDENS
            _nz = h > 0
            if _nz.sum() >= 6 and abs(_m - 1.0) > 1e-6:
                _lh = np.log(h[_nz]); _mu = _lh.mean()
                # Widen the RESIDUAL around the size trend, not the trend itself. The bp trend lives
                # inside log h, so widening log h by m multiplies the degradation slope by m too -
                # at k=1 that is invisible (p=1, m=1, slope exact to +0.00000 of the imposed target)
                # but in a mixture p<1 gives m~1.24 at share 0.3 and ~1.66 at 0.1, averaging near
                # 1.3, which is exactly the 1.327 the twin's mixture slope came out too steep by.
                # Same rule _renorm_tpl already states: the size trend is what damage and fragment
                # length wrote and is kept; only the residual around it is rescaled.
                _tr = np.zeros_like(_lh)
                if bin_size is not None:
                    _xb = bin_size[_nz]
                    if float(_xb.std()) > 1.0:
                        _cf = np.polyfit(_xb, _lh, 1)
                        _tr = np.polyval(_cf, _xb) - _mu
                _new = np.exp(_mu + _tr + np.clip(_m, 1.0, 4.0) * (_lh - _mu - _tr))
                _bo = float(_new.sum() / max(h[_nz].sum(), 1e-12))
                if TPL_SPREAD in ("1", "4"):
                    h = h.copy(); h[_nz] = _new / max(_new.sum(), 1e-12)
                p = float(p) * _bo                          # Jensen boost -> realised share
        # Taken down from a richer rung, the heights also SCATTER more: a fainter template is a noisier lottery. The
        # jitter increment below charges the part that follows height; what a step adds beyond it (AMP_STEP_VAR,
        # same-extract NOC1 dilution pairs) is drawn here. Without it the honest NOC1 twin one rung down came out
        # 0.034 too narrow in sd log height; widening by the whole tpl_shrink curve overshot to -0.022, jitter's
        # share counted twice. Nothing is drawn when the source sits at the target's own level.
        if _lsrc is not None and _teff and _lsrc > _teff * (1.0 + 1e-9):
            _dv = _lvl_var(_teff, inj_sec) - _lvl_var(_lsrc, inj_sec)
            _nzv = h > 0
            if _dv > 0.0 and _nzv.sum() >= 2:
                h = h.copy()
                h[_nzv] = h[_nzv] * np.exp(rng.normal(0.0, np.sqrt(_dv), int(_nzv.sum())))
                h = h / max(float(h.sum()), 1e-12)
        # Taken down from a richer rung, the heights also change how they stand: the size slope flattens
        # with template (see AMP_STEP_TILT). Without this the honest twin - a different amplification one
        # rung up - came out steeper than real at faint templates by 0.0008-0.0011 on the very alleles both
        # kept, while the source copied from the answer came back exact. Applied before the jitter
        # increment, so scatter is looked up on the heights the target actually has.
        if _lsrc is not None and _teff and bin_size is not None and _lsrc > _teff * (1.0 + 1e-9):
            _dt = _lvl_tilt(_teff) - _lvl_tilt(_lsrc)
            _nzt = (h > 0) & (bin_size > 0)
            if _nzt.sum() >= 2 and _dt != 0.0:
                h = h.copy()
                h[_nzt] = h[_nzt] * np.exp(_dt * (bin_size[_nzt] - float(bin_size[_nzt].mean())))
                h = h / max(float(h.sum()), 1e-12)
        exp_h = p * t_total * h
        # Charged in FULL, not as an increment over what the source profile already carries. The
        # increment form is the one the dropout below uses and it looks like the same argument
        # applies, but it was tried and does not: the contributor tiers went from 19-31% too far
        # apart to 20-39%, and NOC2 flipped to 10% too close. Whatever sets the residual tier
        # spread, it is not this variance being counted twice.
        # Looked up on the COPY proxy, not on RFU. Injection decides how much finished product
        # reaches the capillary, and scatter is a molecule-count phenomenon, so two profiles at
        # the same RFU are not equally noisy if they were injected for different times. Pooled
        # over PROVEDIt's three injection times the RFU-keyed curve looks fine; split, the
        # heterozygote CV at 60-150 RFU reads 0.345 / 0.494 / 0.541 at 5 / 15 / 25 sec.
        gs = np.interp(exp_h, JIT_H, JIT_SHAPE)
        # As an INCREMENT over what the source profile already carries, not in full. jit_shape is
        # calibrated from heterozygote pairs in real NOC1 profiles - it encodes the scatter those
        # profiles contain - and the source IS one of them, so charging it again counts the same
        # variance twice. Measured: real NOC1 sd(log h) 0.6558 against the twin's 0.8023, and
        # splitting that as sqrt(s^2+j^2) versus sqrt(s^2+2j^2) gives j = 0.462 and s = 0.465 - the
        # excess is exactly one jitter variance. Removing it outright takes the KS from 0.215 to
        # 0.091, but removing it is not right either: the source carries scatter for ITS OWN height
        # and a twin built at another template needs scatter for the TARGET height. So subtract one
        # and add the other - the same divide-out-then-impose DEG_RESET uses for the slope and _gs2
        # below uses for the capillary term.
        # Charged as an ADDED VARIANCE, not by rescaling the source's realised residual. Rescaling was
        # tried, on the argument that adding can only ever widen - sqrt(s_src^2 + s_add^2) - so the
        # twin carries max(s_src, s_tgt) and the 19.7% of alleles that need LESS scatter than the
        # source has can never get it. That argument is wrong, and the measurement says so: |z| 12.0
        # -> 20.1, tong RFU -1.8 -> -6.2, NOC1 sd log KS 0.179 -> 0.189. The flaw is that cv_source is
        # the LAW's value at the source's height while the source's realised residual has its own
        # random magnitude around it, so scaling by cv_tgt/cv_src assumes an amplitude the source does
        # not have and injects error. Adding variances is correct in expectation whatever the source
        # realised, which is why it wins.
        # Both sides put on the SAME footing before the lookup. exp_h is built at the reference
        # injection - _gain goes back on at the very end - while _hsrc is the source run's raw RFU,
        # carrying its own 5/15/25 sec. Reading one at the reference and the other at the run made
        # the two land at different points of a steep curve even when the source IS the target: in
        # the identity test the target/source height ratio came out 0.609 / 0.999 / 2.982 at the
        # 5th / 50th / 95th percentile, which is 1/_gain at 25 / 15 / 5 sec to three digits. The
        # increment that should have been exactly zero was 0.0010 at the median but 0.1744 at the
        # 95th, so jitter was charging real scatter on a third of the samples for no reason but the
        # injection they happened to be run at.
        _cv2 = np.maximum(1.0 / np.maximum(gs, 1e-9)
                          - 1.0 / np.maximum(np.interp(_hsrc / _gain, JIT_H, JIT_SHAPE), 1e-9), 0.0)
        gs = 1.0 / np.maximum(_cv2, 1e-6)
        if AMP_PRODUCT:
            # amp_cache holds the realised amplification draw so a second injection of the SAME
            # product reuses it instead of rolling the dice again - without it the twin re-scattered
            # every peak, giving sd(log ratio) 0.335 between two injections where NOC1 pairs read
            # 0.085.
            # CAP_SD is NOT taken out here and NOT added back below. It used to be: the gamma carried
            # the whole observed scatter and the capillary share was split off into its own draw, so
            # that two injections of one product could differ. That split assumes the source supplies
            # no capillary noise of its own. It does - the source IS a real run, and its heights
            # already carry the capillary noise of the injection that produced them, so a second draw
            # counted it twice. Found by feeding a real profile in as its own source and asking what
            # came back: the pipeline added sd 0.0996 of per-allele scatter to data that already had
            # it, and removing this draw took that to 0.0746 while every audited statistic held or
            # improved - retention 0.92x -> 0.91x of the replicate envelope, artefact fraction 0.87x
            # -> 0.85x, dynamic range 0.89x -> 0.85x, bp slope 1.10x -> 1.07x.
            _gs2 = gs
            # Drawn UNCONDITIONALLY even when the cache supplies the answer: skipping the call
            # would advance the stream by a different amount on the second injection and desync
            # every draw after it, which is worse than the problem being fixed (sd 0.514 against
            # 0.335 for no cache at all).
            _dr = rng.gamma(_gs2, 1.0 / _gs2)
            _am = amp_cache[di] if (amp_cache is not None and di < len(amp_cache)
                                    and amp_cache[di] is not None) else None
            if _am is None:
                _am = _dr
                if amp_cache is not None:
                    while len(amp_cache) <= di:
                        amp_cache.append(None)
                    amp_cache[di] = _am
            h = h * _am
        else:
            h = h * rng.gamma(gs, 1.0 / gs)
        contrib_exp[di] = exp_h            # what the template says, before this allele's own draw
        contrib[di] = p * t_total * h
        if _lsrc is not None and _teff and _lsrc > _teff * (1.0 + 1e-9):
            # The source is RICHER than this contributor: charge the alleles its amplification kept
            # and this one's would not, at the rate measured between NOC1 dilution levels. Split into
            # the locus-shared and per-molecule parts measured on NOC1 heterozygotes, the locus part
            # drawn once per tube and compared against each contributor's own L - primer efficiency
            # in one reaction is common to everyone in it. Keyed on the contributor's own template,
            # the same key the source ladder is indexed by, so the step converts between two things
            # measured the same way.
            # Which alleles go depends on fragment size (see _conv_surv); the locus-shared part keeps
            # the pooled rate - primer efficiency does not care how long the amplicon is - and the
            # per-molecule part carries the size dependence, so (1-L)(1-u_i) = s_i allele by allele.
            _q = 1.0 - min(_lvl_surv(_teff, inj_sec) / max(_lvl_surv(_lsrc, inj_sec), 1e-9), 1.0)
            if _q > 0.0:
                _l, _u = _amp_split(float(_q))
                _ob = contrib[di] > 0
                _ui = np.zeros(N_FLAT)
                if bin_size is not None and _ob.sum() >= 2 and (bin_size[_ob] > 0).all():
                    _si = _conv_surv(_teff, _lsrc, bin_size[_ob], inj_sec,
                                     np.log(np.maximum(contrib[di][_ob], 1e-9)))
                    _ui[_ob] = np.clip(1.0 - _si / max(1.0 - _l, 1e-9), 0.0, 1.0)
                else:
                    _ui[_ob] = _u
                _live = rng.random(N_FLAT) >= _ui
                if _ushock is not None and _l > 0.0:
                    _live = _live & (_ushock >= _l)
                contrib[di] = contrib[di] * _live
                contrib_exp[di] = contrib_exp[di] * _live
        # A treated contributor drawn from the UNTREATED ladder (its donor has no run under this treatment)
        # loses what the treatment destroys, at the NOC1-measured rate for this code and template. This, not a
        # mixing effect, was the mixture shortfall: +0.006 untreated, +0.007 treated-and-matched, -0.071 where
        # a contributor had fallen back.
        if (cond is not None and str(cond) != "a" and not _cmatch and _teff and bin_size is not None
                and str(cond) in TREAT_CONV):
            _ob = contrib[di] > 0
            if _ob.sum() >= 2 and (bin_size[_ob] > 0).all():
                _kt = np.ones(N_FLAT, bool)
                _kt[_ob] = rng.random(int(_ob.sum())) < _treat_surv(cond, _teff, bin_size[_ob])
                contrib[di] = contrib[di] * _kt
                contrib_exp[di] = contrib_exp[di] * _kt
        contrib_pre[di] = contrib[di]
        _pb_sum += float(p)
        mix += contrib[di]
    if _filt:
        # The filtered contributors carry heights in the source's absolute scale; the tube's allele budget is
        # what the observed total leaves after artefacts and floor, at the reference injection. One factor for
        # the whole tube, so the contributors' relative heights - their copy numbers - stand as drawn.
        _bud = float(_t_in) if _t_in is not None else float(t_total)
        if BUDGET_SOLVE and _t_in is not None:
            _bud = max((_bud - ART_MASS_N) / (1.0 + ART_MASS_A), 0.05 * _bud)
        _bud = _bud / _gain
        _sc = _bud / max(float(_FH.sum()), 1e-9)
        contrib = _FH * _sc; contrib_exp = contrib.copy(); contrib_pre = _FHm * _sc
        mix = contrib.sum(0)
        _pb_sum = 1.0
    # The saturation-widening boost above multiplies each contributor's p by _bo, but phi was
    # normalised BEFORE that, so the boosted shares sum to 1.047 and every twin came out 4.7% heavy:
    # the budget had been solved for a total of 1.0. Measured directly - built mass / budget 1.0451,
    # sum of the boosted phi 1.0466, observed total / target 1.0441, all the same number. The boost
    # is a statement about the contributors' RELATIVE shares; the TOTAL is what t_total fixes and is
    # not the boost's to move. So put the sum back on 1 and let the relative effect stand.
    if _pb_sum > 0.0 and abs(_pb_sum - 1.0) > 1e-9:
        mix /= _pb_sum
        contrib /= _pb_sum
        contrib_exp /= _pb_sum
        contrib_pre /= _pb_sum
    if _gain != 1.0:
        # THE INJECTION, and everything below it is absolute capillary scale: the artefact SURVIVAL
        # curve, the noise floor, saturation and the threshold were all calibrated on heights as the
        # instrument reports them. Only the amplification itself - profile, degradation, jitter,
        # template dropout - is built at the reference. Putting the gain any later made the artefact
        # survival table read a 15 sec-equivalent height and let three times too much stutter through
        # at 5 sec (25.3 per profile against real's 16.3).
        mix *= _gain
        contrib *= _gain
        contrib_exp *= _gain
        t_total = _t_obs
    # NO modelled dropout curve here. The dilution-pair measurement says the whole loss is
    # accounted for by two things that are already in the file: scaling the heights (which pushes
    # alleles under the machine's own AT below) and the stochastic amplification failure q(f) charged
    # per contributor above. Scaling + AT predicts 0.9126 retention on those pairs and q(f) brings it
    # to 0.7852, which is what real profiles show - so a third, modelled threshold at 27 RFU was
    # duplicating the machine's real one at 3.
    art_tab = noc1_calib().get("art_table")
    # One slippage offset for the whole sample: the run's efficiency, shared by every locus. The
    # per-peak sd below keeps only what is left after it, so the pooled spread is unchanged.
    _rsd = float(noc1_calib().get("art_run_sd", 0.0))
    _roff = rng.normal(0.0, 1.0) if _rsd > 0 else 0.0
    _loff = rng.normal(0.0, 1.0, int(BIN_LOCUS.max()) + 1) if ART_LOCUS_R > 0 else None   # see ART_LOCUS_R
    art_loc = noc1_calib().get("art_table_loc")
    art_exp = np.zeros(N_FLAT); art_h = np.zeros(N_FLAT); _isart = np.zeros(N_FLAT, bool)
    _n1 = np.zeros(N_FLAT, bool)                                      # bins an n-1 landed on (STUT_SMALL_Q)
    # WHOSE ARTEFACT IS IT. A stutter is cast by a parent allele, so it belongs to whoever put that allele there -
    # in the same proportion they contributed to the parent. Kept here and written into the per-peak split labels,
    # so the model is taught that these peaks are ALREADY explained by the contributors it has, instead of seeing
    # them as evidence for one more. Measured: 75-83% of artefact HEIGHT sits at a stutter position of a standing
    # allele on both sides, and the remainder (8-10 RFU) is the baseline, which belongs to nobody.
    art_by_c = np.zeros((k, N_FLAT)) if k > 0 else None
    if art_tab:
        # Structured artefacts, emitted from the SUMMED profile at the measured offsets. Position and
        # height both come from the table, so back stutter, forward stutter and n-2 are one mechanism
        # rather than three formulas.
        #
        # Summed, not per contributor: slippage happens once per allele position per reaction, over
        # whatever template is sitting there, so the emission rate belongs to the position and not to
        # each donor separately. Drawing it per contributor makes P(any stutter) = 1 - (1-rate)^m for
        # m donors carrying the parent allele, which climbs to 1 as NOC rises - the twin carried 23.7
        # stutters per NOC5 profile against real's 19.2 while matching exactly at NOC1, and that
        # NOC-dependent gap is the whole signature. Attribution is unaffected: an artefact is attr=-1
        # whichever donor's template made it.
        # Emitted from the TEMPLATE amount, not from the parent's realised peak. Slippage copies the
        # template, so the allele peak and its stutter are two independent draws around one amount;
        # tying the stutter to the peak's own draw destroys that. It shows up as errors-in-variables:
        # conditioning on a low observed parent selects downward draws, so real's observed SR climbs
        # to 0.267 at faint parents against 0.061 at tall ones, and real's surviving n-1 stands at
        # 12 RFU under an 82 RFU parent where SR x that parent is 4.7. Taking the realised peak left
        # the twin's survivors at 7 while matching exactly at tall parents (54/54, 149/150).
        # ...but only the part of it that is NEW. A source profile is a real run, so the stutter a
        # donor's alleles cast onto that donor's OWN alleles is already sitting in the heights that
        # were drawn - and unlike the source's artefacts at every other bin, the DONOR_DOSAGE mask
        # cannot delete it, because at an allele bin the artefact and the allele are one number.
        # Emitting there charges it twice. The exposure is not marginal: 49.7% of a donor's allele
        # bins are the target of one of that donor's own alleles (26.3-68.3% over the 45 donors),
        # since n-1, n+-2 and n+1 carry 93% of the table's rate and heterozygote alleles sit one or
        # two repeats apart all the time. Measured by feeding a real profile in as its own source and
        # splitting the bins: the hit half came back +0.0061 in log and the untouched half -0.0064,
        # a 1.3% step between two halves of one profile, and it is the whole of the 0.0211 identity
        # residual that survived the noise floor and the jitter (0.0211 -> 0.0021 with the table off).
        # Only the OWN-donor share is removed; a stutter cast onto ANOTHER contributor's allele is
        # new to that profile and is exactly what makes mixtures hard, so it stays.
        _cross_cache: dict = {}

        def _cross(d):
            f = _cross_cache.get(int(d))
            if f is not None:
                return f
            t = _OFF_TARGET.get(d)
            f = np.ones(N_FLAT)
            if t is not None and DONOR_DOSAGE is not None:
                _tot = contrib_exp.sum(0)
                ok = (t >= 0) & (_tot > 0)
                if ok.any():
                    ti = t[ok]
                    own = np.zeros(int(ok.sum()))
                    for _ci, _c in enumerate(donor_cols):
                        own += contrib_exp[_ci][ok] * (DONOR_DOSAGE[_c][ti] > 0)
                    f[ok] = np.maximum(1.0 - own / _tot[ok], 0.0)
            _cross_cache[int(d)] = f
            return f

        for parent in [contrib_exp.sum(0)]:
            src = np.where((parent > 0) & ~NO_STUTTER_BIN)[0]            # see NO_STUTTER_LOCI
            if not len(src):
                continue
            if art_loc is not None:
                for L_ in np.unique(BIN_LOCUS[src]):
                    rows = art_loc.get(int(L_)) or art_loc.get(str(int(L_)))
                    if not rows:
                        continue
                    sl = src[BIN_LOCUS[src] == L_]
                    for row in rows:
                        d, rate, lmu, lsd = row[0], row[1], row[2], row[3]
                        _sl2 = row[4] if len(row) > 4 else 0.0
                        _a02 = row[5] if len(row) > 5 else 0.0
                        _flat = row[6] if len(row) > 6 else 0.0
                        tgt = _OFF_TARGET.get(d)
                        if tgt is None:
                            continue
                        if _flat:
                            continue                       # once per SAMPLE, below - not per donor
                        if int(d) in ALLELE_RATE or int(d) in STUT_PH:                # see ALLELE_STUT_OFF
                            _lo = ALLELE_RATE[int(d)][sl] if int(d) in ALLELE_RATE else 0.0
                            if int(d) in STUT_PH:                                     # see STUT_PH
                                _lo = _lo + STUT_PH[int(d)][2][_ph_band(parent[sl], STUT_PH[int(d)][0])]
                            _r0 = min(max(float(rate), 1e-6), 1.0 - 1e-6)
                            _rate = 1.0 / (1.0 + np.exp(-(np.log(_r0 / (1.0 - _r0)) + _lo)))
                            if int(d) in DONOR_RATE and len(sl):                     # see DONOR_RATE
                                _L0 = np.log(_r0 / (1.0 - _r0)) + _lo
                                _sh = contrib_exp[:, sl] / np.maximum(contrib_exp[:, sl].sum(0), 1e-12)
                                _rc = 1.0 / (1.0 + np.exp(-(_L0[None, :] + DONOR_RATE[int(d)][np.asarray(donor_cols)][:, sl])))
                                _rate = (_sh * _rc).sum(0)
                        else:
                            _rate = rate
                        j = sl[(tgt[sl] >= 0) & (rng.random(len(sl)) < _rate)]
                        if not len(j):
                            continue
                        _gsp = (STUT_SPREAD[int(d)][1][_ph_band(parent[j], STUT_SPREAD[int(d)][0])]   # see STUT_SPREAD
                                if int(d) in STUT_SPREAD else 1.0)
                        mu_j = (lmu + _sl2 * (BIN_ALLELE[j].astype(np.float64) - _a02)
                                + _roff * _rsd * lsd * _gsp)
                        if int(d) in ALLELE_STUT_OFF:
                            mu_j = mu_j + ALLELE_STUT_OFF[int(d)][j]        # see ALLELE_STUT(_OFF)
                        _alt = STUT_LT_A.get(int(d), 0.0) * (1.0 if stut_lt is None else float(stut_lt))
                        if _alt > 0:                                        # see STUT_LT_A
                            mu_j = mu_j + np.log(np.minimum(
                                1.0 + _alt * (np.maximum(parent[j], 1.0) / 100.0) ** (-STUT_LT_G[int(d)]), STUT_LT_CAP))
                        if int(d) in STUT_PH:                                       # see STUT_PH
                            mu_j = mu_j + STUT_PH[int(d)][1][_ph_band(parent[j], STUT_PH[int(d)][0])]

                        # Expectation and realisation accumulate SEPARATELY, both per bin, because
                        # survival is decided once on the summed peak - see below.
                        # The DRAW is capped at the largest ratio this offset ever reached on NOC1,
                        # not the median. n+1 and n+-2 genuinely scatter 0.6-1.0 in log - measured
                        # under a uniform detection criterion, so that width is real - but a lognormal
                        # with that width has no upper bound and the real ratio does. Uncapped it put
                        # an n+2 at 2.98x its own parent, 21094 RFU beside a 7076 RFU allele, where
                        # real carries no artefact above 2000 RFU in 1333 mixtures.
                        _fr = (ART_SCAT_K * ART_SCAT_D.get(int(d), 1.0) * lsd * _gsp
                               * np.sqrt(max(1.0 - _rsd ** 2, 0.0)))        # left after the sample offset
                        if _loff is not None:
                            _rt = np.exp(rng.normal(mu_j + _fr * np.sqrt(ART_LOCUS_R) * _loff[BIN_LOCUS[j]],
                                                    _fr * np.sqrt(max(1.0 - ART_LOCUS_R, 0.0))))
                        else:
                            _rt = np.exp(rng.normal(mu_j, _fr))
                        _h0 = STUT_LT_H0.get(int(d), 0.0) * (1.0 if stut_lt is None else float(stut_lt))
                        if _h0 > 0:                                         # see STUT_LT_H0
                            _shp = np.maximum(parent[j], 1e-3) / _h0
                            _rt = _rt * rng.gamma(_shp, 1.0 / _shp)
                        if ART_CAP and len(row) > 7 and row[7] > 0:
                            _rt = np.minimum(_rt, row[7])
                        _pj = parent[j] * _cross(d)[j]
                        np.add.at(art_exp, tgt[j], _pj * np.exp(mu_j))
                        np.add.at(art_h, tgt[j], _pj * _rt)
                        if int(d) == -10:
                            _n1[tgt[j]] = True
                        if art_by_c is not None:
                            _sh = contrib_exp[:, j] / np.maximum(parent[j], 1e-12)     # who made this parent
                            for _ci in range(k):
                                np.add.at(art_by_c[_ci], tgt[j], _pj * _rt * _sh[_ci])
            else:
                for d, rate, lmu, lsd in art_tab:
                    tgt = _OFF_TARGET.get(d)
                    if tgt is None:
                        continue
                    j = src[(tgt[src] >= 0) & (rng.random(len(src)) < rate)]
                    if len(j):
                        np.add.at(mix, tgt[j],
                                  parent[j] * _cross(d)[j] * np.exp(rng.normal(lmu, lsd,
                                                                               size=len(j))))
    elif STUTTER:
        parents = list(contrib)
        for parent in parents:
            js = np.where((STUTTER_TARGET >= 0) & (parent > 0))[0]
            if len(js):
                a = BIN_ALLELE[js].astype(np.float64)
                sr = np.clip(SR_SLOPE * a + SR_INTERCEPT, 0.01, 0.18) * rng.lognormal(0.0, SR_SIGMA, size=len(js))
                np.add.at(mix, STUTTER_TARGET[js], SR_SCALE * sr * parent[js])   # stutter: attr=-1
            jf = np.where((FWD_TARGET >= 0) & (parent > 0))[0]
            if len(jf):
                fr = FS_RATE * rng.lognormal(0.0, FS_SIGMA, size=len(jf))
                np.add.at(mix, FWD_TARGET[jf], fr * parent[jf])
    if art_exp.any():
        # ONE survival decision per bin, on the SUMMED expected height. Slippage is a per-molecule
        # event, so each contributor emits its own product, but the products land in the same bin and
        # the scanner reads what is there: at NOC5 five contributors each put a faint n-1 on one
        # allele and individually none of them clears the floor, while their sum plainly does.
        # Deciding it per contributor cost a quarter of the tall stutters (2.36 per profile at 70-150
        # RFU against real's 3.17) and broke the identity the per-contributor emission exists to
        # preserve, that summed stutter equals SR x summed parent.
        # Keyed on the height the peak ACTUALLY came out at, not the height it was aimed at. Detection
        # is a threshold on what is on the plate, so the survivors are biased tall - real's observed
        # n-1 sits at 15 RFU where the expectation is 9. Keying it on the expectation lets the faint
        # draws through at full rate and left the twin with 6.2 stutters below 15 RFU per NOC5 profile
        # against real's 3.6, at every NOC. The allele dropout law is already applied this way.
        _nz = np.where(art_exp > 0)[0]
        _shH, _shD = _shoulder_hd(np.where((contrib.sum(0) > 0) & (mix >= AT_PC), mix, 0.0))
        _kp = _nz[rng.random(len(_nz))
                  < np.interp(art_h[_nz], ART_SURV_H, ART_SURV_P) * _CARR_MULT[_nz]
                  * SHOULDER_ART_BP[_shD[_nz]]                                  # see _shoulder_hd
                  * np.where(_n1[_nz], np.interp(art_h[_nz], STUT_SMALL_H, (STUT_SMALL_Q, 1.0)), 1.0)]
        if stut_drop:                                                   # see STUT_DROP_RANGE
            _kp = _kp[~((art_h[_kp] < STUT_DROP_H) & (rng.random(len(_kp)) < float(stut_drop)))]
        mix[_kp] += art_h[_kp]
        if art_by_c is not None:
            _mask = np.zeros(N_FLAT, bool); _mask[_kp] = True
            art_by_c = art_by_c * _mask                                  # only the artefacts that survived
        _isart[_kp] = True
    _pb = _NL_P.copy()
    _shN, _shD2 = _shoulder_hd(np.where((contrib.sum(0) > 0) & (mix >= AT_PC), mix, 0.0))
    _sm = _shN > 0
    _pb = _pb.copy()
    for _dsh in (1, 2, 3):                                                 # see _shoulder_hd
        _smd = _sm & (_shD2 == _dsh)
        _pb[_smd] = _pb[_smd] * np.interp(np.log(_shN[_smd]), np.log(SHOULDER_NOISE_H), SHOULDER_NOISE_BP[_dsh])
    _pb = _pb * _CR_MAX              # drawn richer, then thinned by the height-dependent curve
    if PULL_P.size:
        _pb = _pb / np.maximum(1.0 - _PBAR_NOISE, 1e-3)     # the NOC1 pull-up survival the rate carries
    # NOT on a bin a contributor owns. The floor is deleted from each source by the DONOR_DOSAGE
    # mask and re-emitted here, which is right for every bin the mask actually clears - but the mask
    # keeps a source's value at that donor's own allele bins untouched, whatever it is. If real showed
    # only baseline noise at an allele that dropped, that noise is IN the source and comes through
    # with it; if the allele is there, its height already contains the floor that landed on it.
    # Firing here again both lifts peaks that already carry it and manufactures retained alleles out
    # of nothing. Isolated variable by variable, no parameter of this function fixes it - halving the
    # rate or the height trades the identity residual against the peak count - while changing WHERE
    # it may fire wins every column at once (all templates: identity 0.0304 -> 0.0110, invented
    # alleles 0.0177 -> 0.0000, peak count 9.62 -> 9.17 where switching the floor off entirely gives
    # 34.41, median log height 0.0459 -> 0.0377, retention 0.0210 -> 0.0102, sd log 0.0548 -> 0.0244).
    # At the faintest templates, the band that was still outside the envelope, retention goes 0.0342
    # -> 0.0013 and sd log 0.1007 -> 0.0054. The parameters were never wrong; the function was being
    # asked to cover ground the source already covers.
    if DONOR_DOSAGE is not None:
        _ownb = np.zeros(N_FLAT, bool)
        for _c in donor_cols:
            _ownb |= (DONOR_DOSAGE[_c] > 0)
        _pb = np.where(_ownb, 0.0, _pb)
    jn = np.where(rng.random(N_FLAT) < _pb)[0]
    if len(jn):
        # One offset for the whole profile - the baseline is an instrument state per injection,
        # so a loud floor is loud everywhere at once - and the per-peak residual keeps only what
        # is left after it, so the pooled spread is unchanged.
        if NOISE_LOC_Q is not None:
            _zr = RUN_COUPLE * _roff + np.sqrt(max(1.0 - RUN_COUPLE ** 2, 0.0)) * rng.normal()   # see RUN_COUPLE
            _zu = NOISE_RHO * _zr + np.sqrt(max(1.0 - NOISE_RHO ** 2, 0.0)) * rng.normal(size=len(jn))
            _pos = _NCDF(_zu) * (NOISE_LOC_Q.shape[1] - 1)
            _i0 = np.minimum(_pos.astype(np.int64), NOISE_LOC_Q.shape[1] - 2); _fr = _pos - _i0
            _qL = NOISE_LOC_Q[BIN_LOCUS[jn]]; _ar = np.arange(len(jn))
            _nh = np.exp(_qL[_ar, _i0] * (1.0 - _fr) + _qL[_ar, _i0 + 1] * _fr)
        else:
            _no = rng.normal(0.0, 1.0) * NOISE_RUN_SD * NOISE_LN_SD
            _ns = NOISE_LN_SD * np.sqrt(max(1.0 - NOISE_RUN_SD ** 2, 0.0))
            _nh = np.exp(rng.normal(_NL_MU[jn] + _no, _ns))
        if NOISE_GAIN and _t_in and ng_total and inj_sec:                       # see NOISE_GAIN
            _nh = _nh * np.exp(NOISE_GAIN * (np.log(_t_in) - RFU_LOG_C - 0.976 * np.log(float(ng_total))
                                             - 0.948 * np.log(float(inj_sec))))
        # A faint peak in a crowded profile is the one that does not survive - counted on the SAME footing the
        # curve was fitted on. calibrate's `_noise_crowd` measures it as the peaks standing at the SIGNAL bins,
        # a contributor's own allele positions and their +-1/+-2 stutter targets, and its own comment says so
        # ("mixtures carry 94-123 signal bins where NOC1 tops out near 90"); this read every peak above AT
        # instead. Two different quantities, and the total is always the larger, so every mixture landed past
        # the table's last column at 95 and was clamped - NOISE_CR_EXT is all zeros, so beyond the range there
        # is no suppression left at all. Measured: real's floor rate per free bin falls 0.0722 -> 0.0402 from
        # occupancy 60 to 180 while the twin only reached 0.0716 -> 0.0449, +8 % at 120-150 and +12 % above.
        # The total is also SELF-REFERENTIAL - the floor's own peaks count towards it - which is why real NOC1
        # runs the wrong way against it, 0.0641 at occupancy 100-115 rising to 0.0796 above 145, where the
        # mixtures fall to 0.0397. Counted over signal bins, or equivalently over peaks above 50 RFU, NOC1 and
        # the mixtures lie on ONE curve (mean |NOC1 - mixture| gap 0.0035 against 0.0067 for total occupancy
        # and 0.0175 for the NOISE_GAIN residual).
        _occ = float((mix >= NOISE_CR_TH).sum())
        if _KO_OCC_TOTAL:
            _occ = float((mix >= AT_PC).sum())              # DIAGNOSTIC: the old, mismatched footing
        _col = np.array([np.interp(_occ, NOISE_CR_S, NOISE_CR_M[_b])
                         for _b in range(len(NOISE_CR_H))])
        if _occ > NOISE_CR_S[-1]:
            _col = _col * np.exp(rng.random() * NOISE_CR_EXT * (_occ - NOISE_CR_S[-1]))   # see NOISE_CR_EXT
        _keep = np.minimum(np.interp(_nh, NOISE_CR_H, _col) / _CR_MAX, 1.0)
        _sel = rng.random(len(jn)) < _keep
        jn = jn[_sel]; _nh = _nh[_sel]
        np.add.at(mix, jn, _nh)
    _dir = DI_RATE * (1.0 if dropin is None else float(dropin))
    if _dir > 0 and DONOR_DOSAGE is not None:               # see DI_RATE / DI_PERSON: somebody else's sample
        if DI_PERSON:
            _nd = int(rng.poisson(DI_RATE_P * (1.0 if dropin is None else float(dropin))))
            _pool = [c for c in range(len(DONOR_DOSAGE)) if c not in set(donor_cols)]
            for _c in rng.choice(_pool, size=_nd, replace=False) if _nd and _nd <= len(_pool) else []:
                _ob = np.flatnonzero(np.asarray(DONOR_DOSAGE[_c]) > 0)
                _lv = float(np.exp(rng.normal(DI_LMU_P, DI_LSD)))
                _hh = _lv * np.asarray(DONOR_DOSAGE[_c], float)[_ob] * np.exp(rng.normal(0.0, DI_HSD, len(_ob)))
                _st = _ob[_hh >= AT]                        # only what clears detection: a couple of alleles
                if len(_st): np.add.at(mix, _st, _hh[_hh >= AT] * _gain)
        else:
            _nd = int(rng.poisson(_dir))
            if _nd:
                _dfreq = np.asarray(DONOR_DOSAGE, float).sum(0)
                _dloci = np.unique(BIN_LOCUS[_dfreq > 0])
                for _L in rng.choice(_dloci, size=_nd):
                    _bins = np.flatnonzero((BIN_LOCUS == _L) & (_dfreq > 0))
                    _b = int(rng.choice(_bins, p=_dfreq[_bins] / _dfreq[_bins].sum()))
                    mix[_b] += float(np.exp(rng.normal(DI_LMU, DI_LSD))) * _gain
    if SAT_RFU > 0:
        # The capillary saturates. Nothing above this is ever recorded: NOC1 has no peak past it and
        # real's 1333 mixtures top out at 29069 RFU with none above 30000, while the generator reached
        # 44436 - and a parent that tall carried its own stutter past 2000 RFU, which real never does.
        np.minimum(mix, SAT_RFU, out=mix)
    # The PULL-UP filter, on the finished profile (see PULL_P). An allele's source was already filtered in its
    # own context, so only the increment the other contributors' peaks bring is charged, and so is an artefact's,
    # in the context of whoever dominates its parents; noise was emitted with P_bar divided out and meets the
    # whole rule here.
    if PULL_P.size and bin_size is not None:
        _nbk, _ndk = _pull_nb(bin_size)
        _lv = np.where(mix > 0)[0]
        if len(_lv):
            _sv = 1.0 - pull_prob(mix, _nbk, _ndk, _lv, PULL_P)
            _ow = contrib[:, _lv].sum(0) > 0
            if _ow.any():
                _lo = _lv[_ow]; _dom = contrib[:, _lo].argmax(0); _po = np.zeros(len(_lo))
                for _c in np.unique(_dom):
                    _s = _dom == _c
                    _po[_s] = pull_prob(contrib[_c], _nbk, _ndk, _lo[_s], PULL_P)
                _sv[_ow] = np.minimum(_sv[_ow] / np.maximum(1.0 - _po, 1e-9), 1.0)
            _ar = (~_ow) & _isart[_lv]
            if _ar.any():
                _la = _lv[_ar]
                _cp = np.concatenate([contrib, np.zeros((contrib.shape[0], 1))], 1)[:, _ART_PAR[_la]].sum(2)
                _pc = _cp.argmax(0); _pa = np.zeros(len(_la))
                for _c in np.unique(_pc):
                    _s = _pc == _c
                    _v = contrib[_c].copy(); _v[_la[_s]] = mix[_la[_s]]
                    _pa[_s] = pull_prob(_v, _nbk, _ndk, _la[_s], PULL_P)
                _sv[_ar] = np.minimum(_sv[_ar] / np.maximum(1.0 - _pa, 1e-9), 1.0)
            mix[_lv[rng.random(len(_lv)) >= _sv]] = 0.0
        if PU_RATE is not None:                              # see PU_EMIT: the pull-up the analysis could not call
            _hp = np.append(mix, 0.0); _hh = _hp[_nbk]; _aa = _hh.argmax(1); _rr = np.arange(N_FLAT)
            _hm = _hh[_rr, _aa]; _dm = _ndk[_rr, _aa]
            _hi = np.searchsorted(PU_H_EDGES, _hm, side="right") - 1
            _di = np.searchsorted(PU_D_EDGES, _dm, side="right") - 1
            _fr = ((mix <= 0) & (_hi >= 0) & (_hi < len(PU_H_EDGES) - 1)
                   & (_di >= 0) & (_di < len(PU_D_EDGES) - 1))
            _j = np.flatnonzero(_fr)
            if len(_j):
                _pr = PU_RATE[_hi[_j], _di[_j]] * (1.0 if pullup is None else float(pullup))
                _j = _j[rng.random(len(_j)) < _pr]
                if len(_j):
                    _rt = PU_RATIO[_hi[_j], _di[_j]] * np.exp(rng.normal(0.0, 0.4, len(_j)))
                    mix[_j] += _hm[_j] * _rt
    _df = 1.0 if det_f is None else float(det_f)            # see DET_F_RANGE
    if _df > 1.0:
        _na = np.flatnonzero((mix > 0) & (contrib.sum(0) <= 0))
        if len(_na):
            _s1 = np.interp(mix[_na], ART_SURV_H, ART_SURV_P)
            _sf = np.interp(mix[_na] / _df, ART_SURV_H, ART_SURV_P)
            _gone = rng.random(len(_na)) >= np.minimum(_sf / np.maximum(_s1, 1e-9), 1.0)
            mix[_na[_gone]] = 0.0
    drop = mix < AT_PC                                       # below detection -> minor dropout
    mix[drop] = 0.0
    # WITHIN-SAMPLE acceptance. The constants are estimated by pooling thousands of profiles, so a
    # law can hold on average while no single sample obeys it. These three are fixed by the
    # specification - the size decline follows the treatment, and each contributor carries its own
    # proportion - so they are checked in the sample that was just built and the sample is rebuilt if
    # either falls outside its band. Quantities that are stochastic by nature (dropout, stutter
    # scatter, the noise floor) are deliberately NOT here: forcing those per sample would delete
    # variation that real data has.
    # The acceptance bands check the OLD path's modelled slope and shares; the filtered path derives both from Q
    # and copy numbers, so it is not rejected against a model it does not use.
    if _depth < TOL_ITERS and ACCEPT and bin_size is not None and k >= 2 and not _filt:
        _bad = None
        _liv = mix > 0
        if _liv.sum() >= 20:
            _own = contrib.sum(0) > 0
            _m2 = _liv & _own & (bin_size > 0)
            if _m2.sum() >= 15:
                _sl = -np.polyfit(bin_size[_m2], np.log(mix[_m2]), 1)[0]
                # against the value THIS sample drew, not the treatment's centre - otherwise the
                # check filters out exactly the draws the spread was added to produce.
                _exp = (float(_bsamp) if _bsamp is not None else 0.0) + BASE_SLOPE
                if abs(_sl - _exp) > TOL_SLOPE:
                    _bad = "slope"
        # A per-contributor copy-number band was tried and removed. Two things went wrong and both
        # are worth remembering: the test read contrib (before artefacts) while the defect is measured
        # on the finished profile (after artefacts land on those same bins), so it constrained a
        # different quantity; and rejecting samples skews the distribution in ways the constrained
        # statistic does not capture - ID fell 0.945 -> 0.930 against real's 0.948. The spread is real
        # (generator p10-p90 0.83-3.92 against real's 1.26-2.57) but rejection is the wrong tool for it.
        if _bad is None:
            for _di in range(k):
                # On the PRE-dropout mass. phi specifies how much template each contributor brings,
                # not how much of it survives detection - a minor is SUPPOSED to lose more. Checking
                # the post-dropout mass makes this a filter on the dropout itself: strengthen the
                # loss a minor takes and its realised share falls past TOL_SHARE, the sample is
                # rejected, and the accepted population is biased toward the draws where the minor
                # happened to survive. That is what silently undid the retention anchor - the twin's
                # multi-carrier survival went UP when the anchor should only ever push it down.
                _cp = contrib_pre if contrib_pre.any() else contrib
                _s1 = float(_cp[_di].sum()); _st = float(_cp.sum(0).sum())
                if _st > 0 and abs(_s1 / _st / max(float(phi[_di]), 1e-9) - 1.0) > TOL_SHARE:
                    _bad = "share"; break
        if _bad is not None:
            return gen_mixture(donor_cols, pool, rng, t_total=t_total, bin_size=bin_size,
                               phi=phi, q_target=q_target, cond=cond, ng_total=ng_total,
                               inj_sec=inj_sec, _depth=_depth + 1, amp_cache=amp_cache, q_tube=q_tube, kappa=kappa,
                               stut_lt=stut_lt, dropin=dropin, pullup=pullup, det_f=det_f, stut_drop=stut_drop)
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
    # ...and the FULL split, not just the winner. The generator knows how much of every peak each contributor put
    # there; keeping only the dominant one threw that away and left the trainer to rebuild it from a phi x copy-number
    # approximation - exactly the quantity that is wrong for the faint contributor a mixture stands or falls on.
    # frac_bin[i, j] = the share of bin j that contributor i (donor cols5[i]) contributed, 0 where the bin is empty.
    frac_bin = np.zeros((MAX_CONTRIB_LBL, N_FLAT), dtype=np.float32)
    cols5 = np.full(MAX_CONTRIB_LBL, -1, dtype=np.int16)
    if k > 0:
        # allele bins: the contributors' own heights; artefact bins: who cast the stutter that stands there.
        # A bin that is both (a donor's allele carrying somebody else's stutter) counts both.
        src = contrib + (art_by_c if art_by_c is not None else 0.0)
        tot_c = src.sum(0)
        ok = (~drop) & (tot_c > 0)
        m = min(k, MAX_CONTRIB_LBL)
        frac_bin[:m][:, ok] = src[:m][:, ok] / tot_c[ok]
        cols5[:m] = np.asarray(donor_cols[:m], dtype=np.int16)
    return xflat, y, k, t_total, attr_bin, phi45, frac_bin, cols5

def _log_spread(k, rng):
    """Upper end of the log-ratio range for one mixture.

    The shipped draw is w_i = exp(U(0, ln r_max)) for each of the k contributors, so the SPREAD of a
    mixture is emergent — an order statistic of k independent draws — and it concentrates: at k=5 the
    median max/min is 8.1 and mixtures near 1:1:1:1:1 essentially never occur. Balanced mixtures are
    where cross-locus coherence separates a true donor from a decoy (real: +0.074 there against -0.000
    at the faintest), and 21.5% of real NOC5 profiles are exactly equal, so the shipped generator omits
    the region the discriminating signal lives in while covering skews beyond anything real.

    Drawing the spread itself log-uniformly over [1, r_max] keeps r_max = 6+5(k-2) untouched — the
    deliberately-harder-than-real upper end stays, since casework ratios are arbitrary and PROVEDIt's
    {1,2,4,9} grid is an artefact of one experimental design — and simply makes the rest of the range
    reachable. E[ln spread | k] is still ln(r_max)/2, so skew still grows with k and the k-ratio
    contrast the count head depends on survives at half the log-slope.

    Replacing the shipped draw outright is wrong: covering the spread uniformly thins the extreme-skew
    tail, and at NOC5 that pushed min-phi p5 from 0.027 to 0.046 while real reaches 0.034, so the
    faintest real minors stopped being covered. A coin flip between the two keeps the shipped support
    intact and adds the rest, which is a union rather than a swap — the superset the whole exercise is
    supposed to build.

    The ramp also has to clear the widest ratio the reference data actually contains, which is 9:1 at
    EVERY k — PROVEDIt's design grid is {1,2,4,9} regardless of contributor count. r_max = 6+5(k-2)
    gives only 6 at k=2 and 11 at k=3, so the shipped generator is NARROWER than reality at low NOC
    while being wider at high NOC; measured min-phi p5 was 0.223 against real's 0.100 at k=2. Flooring
    the range at 9 restores coverage where it was missing without touching k=4,5, where the ramp
    already exceeds it."""
    return rng.uniform(0.0, np.log(R_SPREAD_FLOOR * max(1, k - 1)))

def _skew_phi(k, rng):
    """Proportions for the skewed regime.

    k independent draws on [0, ln R] realise a spread far below R — an order statistic of k samples,
    which at k=2 has median ratio ~3 against a ceiling of 9. That is why raising the ceiling alone left
    min-phi p5 at 0.178 where real reaches 0.100: the ceiling was never being approached. Anchoring one
    contributor at each end makes the realised spread equal to the drawn one, so the range becomes a
    property that is set rather than hoped for, and the positions are shuffled so the anchoring carries
    no information about which donor is which."""
    lr = _log_spread(k, rng)
    u = rng.uniform(0.0, lr, k)
    if k >= 2:
        u[0], u[1] = 0.0, lr
        rng.shuffle(u)
    w = np.exp(u)
    return w / w.sum()


# CONDITIONED TRAINING DRAWS. The laws that decide which alleles stand (the Q filter, the damage range, the
# cell counts, the tube efficiency) only act when the generator is told the sample's conditions. Each training
# mixture draws its conditions from ranges measured on TRAIN NOC1 only: the treatment code at its NOC1 frequency, the
# tube's Q from that code's NOC1 Q values, the shares (GEN_PERSON_NG, GEN_AMOUNT_FLOOR, GEN_PHI_WIDE), the tube total
# from PROVEDIt's design grid (GEN_DESIGN_NG), the injection among the three the instrument used, and the observed
# total from the NOC1 RFU law at that mass and injection. Nothing is read from a real mixture.
# Each person leaves DNA independently of how many others did: GEN_PERSON_NG, log-uniform over the NOC1 ladder range
# the presence law is validated on.
GEN_PERSON_NG = (0.008, 0.5)
# The smallest person's share relative to the largest. Independent log-uniform draws put it at p10 .025-.055
# (median .06-.28) where real's designs never go below 1:9 (p10 .111, median .25-.50); a floor on m relative to the
# largest keeps every labelled person within what real mixtures hold.
GEN_AMOUNT_FLOOR = 0.1
_CONDTAB = [None]


def _condition_table():
    if _CONDTAB[0] is None:
        import re as _re2
        # rx = _re2.compile(r"RD\d+-\d+-\d+d\d+([A-Za-z0-9\-]*?)-[\d.]+GF-Q([\d.]+)_")
        rx = _re2.compile(r"RD\d+-\d+-\d+d\d+([A-Za-z0-9\-]*?)-[\d.]+" + kit.TAG + r"-Q([\d.]+)_")
        by = {}
        for n in json.load(open(DATA / "meta_sample_names_train.json")):
            m = rx.search(str(n))
            if m:
                by.setdefault(m.group(1) or "a", []).append(float(m.group(2)))
        codes = sorted(by)
        w = np.array([len(by[c]) for c in codes], float)
        _CONDTAB[0] = (codes, w / w.sum(), {c: np.asarray(by[c], float) for c in codes})
    return _CONDTAB[0]


# HOW MUCH DNA A TUBE HOLDS, GIVEN THE NUMBER OF CONTRIBUTORS. The per-person amount (GEN_PERSON_NG) was
# calibrated on NOC1 and is right there - K=1 draws a median 0.064 ng against the real 0.062 - but summing K
# independent draws makes the TOTAL grow with K, and the only thing that capped it was a target load of 0.5-1.5 ng,
# a bound that almost never binds. Measured against the real template amounts: the generator's median total is
# x1.03 of real at K=1 and x1.71 at K=5, and the whole brightness error follows - real NOC5 profiles carry a
# median 40k total RFU, the generator's 67k (x1.68), with RFU/ng flat across K in real data and the injection mix
# balanced, so template is the lever. Worse, the FAINT TAIL is absent: the generator's K=5 p10 is 0.247 ng where
# the design says 0.120 and the test split 0.090, so a 5-person mixture at 0.075 ng - which is where the real
# undercount errors live - was never generated at all.
#
# The replacement is not a fit. PROVEDIt ran a fixed grid of (contributors x total template), it is written in the
# sample file names, and `-([\d.]+)(?:GF|IP|PP)` reads it off them exactly as extract_phi_condition.py does for
# the real arrays. So the total is drawn from the experiment's own grid for that K and `m` is demoted to shares.
# No constant is chosen. The grid comes from the raw GF 3500 file
# listing, not from our split - it agrees with the test split at every quantile of K=1..4 and in the K=5 tail.
# THE SHARES MUST CONTAIN THE BALANCED END (GEN_PHI_WIDE). Independent log-uniform shares (gen_conditioned hands
# gen_mixture phi = m / m.sum()) are essentially never balanced. Five independent
# draws over a 62x range are essentially never balanced: the synth NOC5 smallest share topped out at 0.181, while
# 102 of 372 real NOC5 profiles are 1:1:1:1:1 (0.200) - 27% of real NOC5 lay entirely outside training, and only
# 0.9% of synth NOC5 reached real's median smallest share (0.125). The aim is to CONTAIN real, not to copy the
# design's ratio grid, so the same superset is wired in here with its own constants: half the tubes keep the skewed
# draw (which covers mixtures harder than real), half take the balanced regime. The tube total is untouched.
_DESIGN_NG = [None]


def _design_ng(k):
    """Total template amounts (ng) PROVEDIt ran with exactly `k` contributors, clamped to 1..5."""
    if _DESIGN_NG[0] is None:
        import csv as _csv, glob as _glob, re as _re3
        # rx_t = _re3.compile(r"-([\d.]+)(?:GF|IP|PP)")
        rx_t = _re3.compile(r"-([\d.]+)(?:GF|IP|PP|F6C)")
        raw = Path(__file__).resolve().parent.parent / "data_raw"
        by = {}
        for f in _glob.glob(str(raw / "**" / "*.csv"), recursive=True):
            # if "3500_GF29cycles" not in f:
            if kit.KIT not in f or "Known Genotypes" in f:
                continue
            with open(f, newline="", encoding="utf-8", errors="ignore") as fh:
                r = _csv.reader(fh); next(r, None)
                seen = set()
                for row in r:
                    if not row or row[0] in seen:
                        continue
                    seen.add(row[0])
                    nm = str(row[0]); parts = nm.split("-")
                    if len(parts) < 3:
                        continue
                    kk = len([x for x in parts[2].split("_") if x.strip()])
                    mt = rx_t.search(nm)
                    if kk and mt:
                        try:
                            by.setdefault(min(kk, 5), []).append(float(mt.group(1)))
                        except ValueError:
                            pass
        # a cached copy so a build without data_raw still runs
        cp = DATA / "design_template_ng.json"
        if by:
            json.dump({str(a): sorted(set(b)) for a, b in by.items()}, open(cp, "w"))
        elif cp.exists():
            by = {int(a): list(b) for a, b in json.load(open(cp)).items()}
        if not by:
            raise RuntimeError("design_template_ng: no GF 3500 sample names found")
        # DISTINCT values, equal weight: the set of template amounts PROVEDIt's design ran at each K, without counting
        # how often each was run (those counts come from the evaluation samples' file names). The range at every K is
        # unchanged, which is what the training data needs: to contain the real cases.
        _DESIGN_NG[0] = {a: np.unique(np.asarray(b, float)) for a, b in by.items()}
    return _DESIGN_NG[0][int(np.clip(k, 1, 5))]


def gen_conditioned(cols, pool, rng, bin_size):
    """One tube: conditions, shares and total template drawn as described above, then gen_mixture."""
    codes, w, qs = _condition_table()
    cd = codes[int(rng.choice(len(codes), p=w))]
    q = float(rng.choice(qs[cd]))
    m = np.exp(rng.uniform(np.log(GEN_PERSON_NG[0]), np.log(GEN_PERSON_NG[1]), len(cols)))
    m = np.maximum(m, GEN_AMOUNT_FLOOR * m.max())                            # see GEN_AMOUNT_FLOOR
    if rng.random() < WIDE_REAL_FRAC:                                         # see GEN_PHI_WIDE
        _k = len(cols)
        _b = (np.full(_k, 1.0 / _k) if rng.random() < REAL_EQUAL_FRAC
              else rng.dirichlet(np.full(_k, REAL_DIRICHLET_ALPHA)))
        m = float(m.sum()) * _b                                               # same total, balanced shares
    ng = float(rng.choice(_design_ng(len(cols))))                             # see GEN_DESIGN_NG
    inj = float(rng.choice([5.0, 15.0, 25.0]))
    # The tube's efficiency (KAPPA_MAX) acts on every copy, so it lowers the PCR product - the total signal - by the
    # same factor it lowers the copies that decide presence; drawn here so both see one value.
    kap = float(KAPPA_FIX) if KAPPA_FIX is not None else float(rng.uniform(0.0, KAPPA_MAX))
    slt = float(rng.uniform(*STUT_LT_RANGE))                                   # see STUT_LT_A
    sdi = float(rng.uniform(*DI_RANGE))                                        # see DI_RATE
    spu = float(rng.uniform(*PU_RANGE))                                        # see PU_EMIT
    sdf = float(np.exp(rng.uniform(np.log(DET_F_RANGE[0]), np.log(DET_F_RANGE[1]))))  # see DET_F_RANGE
    sqd = float(rng.uniform(*STUT_DROP_RANGE))                                 # see STUT_DROP_RANGE
    eff = 2.0 ** (-kap * (len(cols) - 1))
    _eps = rng.normal(0.0, RFU_LOG_SD)                   # ONE scatter draw for the tube
    _tt = lambda j: eff * float(np.exp(RFU_LOG_C + 0.976 * np.log(ng) + 0.948 * np.log(j) + _eps))
    return gen_mixture(cols, pool, rng, t_total=_tt(inj), inj_sec=inj, bin_size=bin_size, phi=m / m.sum(), cond=cd,
                       ng_total=ng, q_tube=q, kappa=kap, stut_lt=slt, dropin=sdi, pullup=spu, det_f=sdf, stut_drop=sqd)

MAX_CONTRIB_LBL = 5          # how many contributors the per-peak split is kept for
VAL_COMBO_FRAC = 0.15        # share of each NOC's in-silico combos held out to the validation split


def xflat_to_tokens(xflat, attr_bin=None, bin_size=None, frac_bin=None):
    nz = np.where(xflat > 0)[0]
    if len(nz) > MAX_SEQ:
        nz = nz[np.argsort(xflat[nz])[::-1][:MAX_SEQ]]   # keep tallest
    tok = np.zeros((MAX_SEQ, 3), dtype=np.float32)
    mask = np.zeros(MAX_SEQ, dtype=bool)
    attr = np.full(MAX_SEQ, -1, dtype=np.int16)          # per-peak dominant donor (-1 = pad)
    size = np.zeros(MAX_SEQ, dtype=np.float32)           # per-peak fragment size (bp)
    frac = np.zeros((MAX_SEQ, MAX_CONTRIB_LBL), dtype=np.float32)   # per-peak split over the sample's contributors
    for i, j in enumerate(nz):
        tok[i] = [BIN_LOCUS[j], BIN_ALLELE[j], xflat[j]]
        mask[i] = True
        if attr_bin is not None:
            attr[i] = attr_bin[j]
        if bin_size is not None:
            size[i] = bin_size[j]
        if frac_bin is not None:
            frac[i] = frac_bin[:, j]
    return (tok, mask, attr, size) if frac_bin is None else (tok, mask, attr, size, frac)

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

def build_train(n_mix, pool, rng, noc_p=None):
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
    # The true per-peak split. The real single sources carry it too: one donor made every allele AND every stutter
    # those alleles cast, so both belong to that donor, exactly as the generator labels its own rows (art_by_c).
    # Left unlabelled they fell back to the trainer's approximation, which sends every peak that is not one of the
    # donor's alleles to background - so the real 10% of training taught "stutter is noise" while the in-silico 90%
    # taught "stutter belongs to its parent's donor", and the real rows are the ones carrying real stutter. Only a
    # peak at none of the artefact table's offsets from a STANDING allele of the donor stays background (baseline).
    frac_new, col_new = [], []
    frac_real = np.zeros((len(size_real), MAX_SEQ, MAX_CONTRIB_LBL), np.float32)
    col_real = np.full((len(size_real), MAX_CONTRIB_LBL), -1, np.int16)
    _offs = sorted({int(_rw[0]) for _rows in (noc1_calib().get("art_table_loc") or {}).values() for _rw in _rows})
    if DONOR_DOSAGE is not None and _offs:
        for r in range(len(d_ss)):
            c = int(d_ss[r]); _m = mask_real[r].astype(bool)
            _tb = np.full(MAX_SEQ, -1, np.int64)
            for p_ in np.flatnonzero(_m):
                _tb[p_] = _BININDEX.get((int(round(float(tok_real[r, p_, 0]))), round(float(tok_real[r, p_, 1]), 1)), -1)
            _own = np.asarray(DONOR_DOSAGE[c]) > 0
            _mine = np.zeros(N_FLAT, bool)
            for b in _tb[(_tb >= 0)]:
                if _own[b]:
                    _mine[b] = True
                    for d in _offs:
                        t = _OFF_TARGET[d][b] if d in _OFF_TARGET else -1
                        if t >= 0:
                            _mine[t] = True
            _lab = (_tb >= 0) & _mine[np.maximum(_tb, 0)]
            frac_real[r, _lab, 0] = 1.0
            col_real[r, 0] = c
    print(f"Keeping {ss.sum()} real single-source. Generating {n_mix} in-silico mixtures..."
          f" (size channel: {'ON' if bin_size is not None else 'OFF — run extract_size.py'})")
    Xf_new, y_new, noc_new, tok_new, mask_new, attr_new, phi_new = [], [], [], [], [], [], []
    made = 0
    while made < n_mix:
        k = int(rng.choice([2,3,4,5], p=noc_p)) if noc_p is not None else int(rng.integers(2, 6))
        cols = sorted(rng.choice(45, size=k, replace=False).tolist())
        ids = tuple(sorted(KNOWN[c] for c in cols))
        if ids in test_set or ids in val_set:
            continue
        xf, y, kk, _, ab, ph, fb, c5 = gen_conditioned(cols, pool, rng, bin_size)
        tok, mask, attr, size, frac = xflat_to_tokens(xf, ab, bin_size, fb)
        Xf_new.append(xf); y_new.append(y); noc_new.append(kk); tok_new.append(tok)
        mask_new.append(mask); attr_new.append(attr); phi_new.append(ph); size_new.append(size)
        frac_new.append(frac); col_new.append(c5)
        made += 1
    G = dict(Xflat=np.array(Xf_new, np.float32), y=np.array(y_new, np.float32), noc=np.array(noc_new, np.int32),
             tokens=np.array(tok_new, np.float32), mask=np.array(mask_new, bool), attr=np.array(attr_new, np.int16),
             phi=np.array(phi_new, np.float32), size=np.array(size_new, np.float32),
             attrfrac=np.array(frac_new, np.float16), attrcol=np.array(col_new, np.int16))
    # THE VALIDATION SPLIT: the real NOC1 val from data/ plus VAL_COMBO_FRAC of each NOC's unique in-silico combos, with
    # all their samples. Checkpoint selection and every decode parameter are chosen on it; no val combo is trained on.
    held = np.zeros(len(G["noc"]), bool); vrng = np.random.default_rng(0)
    for kk in (2, 3, 4, 5):
        combos = {}
        for i in np.flatnonzero(G["noc"] == kk):
            combos.setdefault(tuple(np.flatnonzero(G["y"][i] > .5).tolist()), []).append(i)
        uniq = sorted(combos); vrng.shuffle(uniq)
        for c in uniq[:max(1, int(round(len(uniq) * VAL_COMBO_FRAC)))]:
            held[combos[c]] = True
    keep = ~held
    Xf = np.concatenate([G["Xflat"][keep]] + Xf_list).astype(np.float32)
    Y = np.concatenate([G["y"][keep]] + y_list).astype(np.float32)
    NOC = np.concatenate([G["noc"][keep]] + noc_list).astype(np.int32)
    TOK = np.concatenate([G["tokens"][keep]] + tok_list).astype(np.float32)
    MASK = np.concatenate([G["mask"][keep]] + mask_list).astype(bool)
    ATTR = np.concatenate([G["attr"][keep]] + attr_list).astype(np.int16)
    PHI = np.concatenate([G["phi"][keep]] + phi_list).astype(np.float32)
    SIZE = np.concatenate([G["size"][keep]] + size_list).astype(np.float32)
    FRAC = np.concatenate([G["attrfrac"][keep], frac_real]).astype(np.float16)
    FCOL = np.concatenate([G["attrcol"][keep], col_real]).astype(np.int16)
    rv = {"Xflat": np.load(DATA / "Xflat_val.npy"), "y": np.load(DATA / "y_val_set.npy"), "noc": np.load(DATA / "noc_val.npy"),
          "tokens": np.load(DATA / "tokens_val.npy"), "mask": np.load(DATA / "mask_val.npy"), "attr": np.load(DATA / "attr_val.npy"),
          "phi": np.load(DATA / "phi_val.npy"), "size": np.load(DATA / "size_val.npy")}
    n_rv = len(rv["noc"])
    rv["attrfrac"] = np.zeros((n_rv,) + G["attrfrac"].shape[1:], np.float16)
    rv["attrcol"] = np.full((n_rv,) + G["attrcol"].shape[1:], -1, np.int16)
    for nm in G:
        out_nm = "y_val_set.npy" if nm == "y" else f"{nm}_val.npy"
        np.save(OUT / out_nm, np.concatenate([rv[nm].astype(G[nm].dtype), G[nm][held]]))
    nv = n_rv + int(held.sum())
    for nm in ("condition", "condition_level", "meta_template", "meta_qindex", "combo_id", "noc_true"):
        src = DATA / f"{nm}_val.npy"                       # real-only metadata: in-silico rows get -1 (noc_true: K)
        if src.exists():
            a = np.load(src)
            fill = G["noc"][held].astype(a.dtype) if nm == "noc_true" else np.full(int(held.sum()), -1, a.dtype)
            np.save(OUT / src.name, np.concatenate([a, fill]))
    perm = rng.permutation(len(Xf))
    np.save(OUT / "Xflat_train.npy", Xf[perm]); np.save(OUT / "y_train_set.npy", Y[perm])
    np.save(OUT / "noc_train.npy", NOC[perm]); np.save(OUT / "tokens_train.npy", TOK[perm]); np.save(OUT / "mask_train.npy", MASK[perm])
    np.save(OUT / "attr_train.npy", ATTR[perm]); np.save(OUT / "phi_train.npy", PHI[perm])
    np.save(OUT / "size_train.npy", SIZE[perm])
    np.save(OUT / "attrfrac_train.npy", FRAC[perm]); np.save(OUT / "attrcol_train.npy", FCOL[perm])
    # The laws these rows were made with, for the decode layer (gen_law.py): what the decode subtracts and expects
    # must be what the generator did.
    import types
    import gen_law as _gl
    np.save(OUT / _gl.LAW_FILE, _gl.build_law(types.SimpleNamespace(**globals())), allow_pickle=True)
    print(f"  privileged labels: attr_train {ATTR.shape}, phi_train {PHI.shape}, size_train {SIZE.shape}, "
          f"attrfrac_train {FRAC.shape} (true per-peak split on {int((FCOL >= 0).any(1).sum())} rows)")
    # copy real val/test/open + meta. phi_/condition_ (real NOMINAL mixture proportion +
    # DNA condition, from extract_phi_condition.py) are carried so the Inc2 phi head can be
    # EVALUATED on real and the in-silico->real gap stratified by condition.
    for split in ["test"]:
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
              "combo_id_test.npy", "fold_info.json",
              "donor_geno.npy", "donor_geno_mask.npy"]:
        if (DATA / f).exists():
            shutil.copy(DATA / f, OUT / f)
    print(f"Saved {len(Xf)} train ({int(keep.sum())} in-silico + {ss.sum()} real SS) and {nv} val "
          f"({n_rv} real NOC1 + {int(held.sum())} in-silico, combo-disjoint) -> {OUT}")
    feat_summary(Xf[perm][:2000], NOC[perm][:2000], "generated train (mixed)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", type=int, default=0, help="generate N in-silico train mixtures (conditioned draws)")
    ap.add_argument("--noc_weights", type=str, default=None,
                    help="relative NOC sampling weights for NOC2,3,4,5 e.g. '1,1.5,2.5,2' "
                         "(default uniform). Higher = more high-NOC mixtures.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    pool = build_ss_pool()
    noc_p = None
    if args.noc_weights:
        w = np.array([float(x) for x in args.noc_weights.split(",")], dtype=float)
        assert len(w) == 4, "noc_weights needs 4 values (NOC2,3,4,5)"
        noc_p = (w / w.sum()).tolist()
        print(f"NOC sampling distribution (2,3,4,5): {[round(p,3) for p in noc_p]}")
    assert args.build > 0, "--build N: the number of in-silico mixtures to generate"
    build_train(args.build, pool, rng, noc_p=noc_p)

if __name__ == "__main__":
    main()
