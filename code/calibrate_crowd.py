# Calibrates the noise-floor crowding law as an increment over the generator (TRAIN NOC1, real vs twin with the law off)
# and writes data/noise_cr_inc.json, which make_insilico reads (NOISE_CR_*). Fold-specific: re-run for every fold.
import importlib.util, json, os, re, sys, collections
from pathlib import Path
import numpy as np
import kit
CODE = Path(__file__).resolve().parent; REAL = Path(os.environ.get("STR_DATA_DIR", str(CODE / "data")))
os.environ["STR_DATA_DIR"] = str(REAL); sys.path.insert(0, str(CODE))
sp = importlib.util.spec_from_file_location("mi", CODE / "make_insilico.py")
MI = importlib.util.module_from_spec(sp); sp.loader.exec_module(MI)
pool = MI.build_ss_pool(); BSV = MI.build_bin_size(); BS = np.asarray(BSV, float)
BIDX = MI._BININDEX; NB = MI.N_FLAT; BL = MI.BIN_LOCUS.astype(int); BA = MI.BIN_ALLELE.astype(float)
UNITS = MI.BIN_UNITS                                  # repeat units (make_insilico.BIN_UNITS)
g = np.load(REAL / "donor_geno.npy"); gmb = np.load(REAL / "donor_geno_mask.npy").astype(bool)
TB = np.zeros((45, NB), bool)
for c in range(45):
    for j in range(g.shape[1]):
        if gmb[c, j]:
            k = BIDX.get((int(round(float(g[c, j, 0]))), round(float(g[c, j, 1]), 1)), -1)
            if k >= 0:
                TB[c, k] = True
# stutter partner of each bin: index of the bin one repeat below / above / two below / half step below, same locus
KEY = {(int(BL[j]), round(float(UNITS[j]), 2)): j for j in range(NB)}
def nb(j, d):
    return KEY.get((int(BL[j]), round(float(UNITS[j]) + d, 2)), -1)
OFF = {"lui -1": -1.0, "tien +1": 1.0, "lui -2": -2.0, "nua buoc": -0.5}
PART = {n: np.array([nb(j, d) for j in range(NB)]) for n, d in OFF.items()}
# Writes data/noise_cr_inc.json, read by make_insilico (NOISE_CR_M). Re-run after changing the noise floor.
# TRAIN NOC1: the crowding law as an INCREMENT. Each real profile against its twin generated with the crowding law off;
# noise = peaks off every allele and off every product offset (-3..+2, half step). The column is the number of peaks at
# or above CR_TH = 50 RFU, read on the REAL profile so both sides share it - NOT the signal-bin count this used before,
# and NOT the total peak count the generator looked the curve up with. Measured on real (NOC1 to fit, the mixtures as
# diagnosis): against the tall-peak count the floor rate per free bin runs 0.0794 below 20 to 0.0383 above 100 and NOC1
# and the mixtures agree inside 0.004 over their common range, while against TOTAL occupancy NOC1 runs the WRONG way
# (0.0641 at 100-115 rising to 0.0796 above 145, where the mixtures fall to 0.0397) because the total counts the
# floor own peaks - it is self-referential. Mean |NOC1 - mixture| gap: 0.0035 tall peaks, 0.0067 total occupancy,
# 0.0175 the NOISE_GAIN residual. NOC1 tops out near 99 tall peaks while the mixtures reach 144, so the log-linear
# extension below is what carries the law past NOC1 range; it used to be written as zero, leaving the curve flat
# exactly where the mixtures live.
MI.NOISE_CR_M = np.ones_like(MI.NOISE_CR_M); MI._CR_MAX = 1.0; MI.NOISE_CR_EXT = np.zeros_like(MI.NOISE_CR_EXT)
PRODS = [-3.0, -2.0, -1.0, 1.0, 2.0, -0.5]
HB = [0, 8, 12, 16, 25, 1e9]
CR_TH = 50.0                                     # a peak this tall or taller counts as signal on the plate
SB = [0, 25, 35, 45, 55, 70, 82, 1e9]            # columns in the number of such peaks, on TRAIN NOC1.
# The top column is SPLIT because pooling everything above 70 made the curve flat by construction exactly
# where the mixtures live: NOC1 carries 657 profiles at 70-80 tall peaks and 166 above 80, and the exact
# noise ledger says the twin needs about 5 % more suppression there (its own rate per free bin falls 26 %
# from NOC1 to NOC5 where real falls 30 %). Let NOC1 speak in that range instead of extrapolating from a
# single pooled point at 75.8.


def split(z, c):
    """(number of peaks >= CR_TH, histogram of the floor own peaks by height)"""
    a = z > 0; tru = TB[c]; nz = []
    for b in np.flatnonzero(a):
        if tru[b]:
            continue
        ta = UNITS[(BL == BL[b]) & tru]
        if not (len(ta) and any(np.isclose(UNITS[b] - ta, p).any() for p in PRODS)):
            nz.append(z[b])
    return float((z >= CR_TH).sum()), np.histogram(nz, HB)[0]


RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)" + kit.TAG + r"-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
NM = json.load(open(REAL / "meta_sample_names_train.json")); Xs = np.load(REAL / "Xflat_train.npy"); ns = np.load(REAL / "noc_train.npy")
CR = np.zeros((len(SB) - 1, len(HB) - 1)); CT = np.zeros_like(CR); NN = np.zeros(len(SB) - 1)
SM = np.zeros(len(SB) - 1)
for i in np.where(ns == 1)[0]:
    m = RX.search(str(NM[i]))
    if not m or int(m.group(1)) not in MI.COL:
        continue
    c = MI.COL[int(m.group(1))]; z = np.expm1(Xs[i].astype(np.float64))
    rng = np.random.default_rng(700 + i)
    w = np.expm1(MI.gen_mixture([c], pool, rng, t_total=float(z.sum()), bin_size=BSV, phi=np.array([1.0]), cond=m.group(3) or "a",
                                ng_total=float(m.group(4)), inj_sec=float(m.group(6)), q_tube=float(m.group(5)))[0].astype(np.float64))
    oa, ha = split(z, c); ob, hb = split(w, c)
    k = int(np.searchsorted(SB, oa, side="right") - 1)          # column by the REAL profile (paired)
    CR[k] += ha; CT[k] += hb; NN[k] += 1; SM[k] += oa
M = (CR / np.maximum(CT, 1)).T                                  # band x column
# A cell with too few real peaks INHERITS the column before it. Splitting the top column let the tall bands of the
# last column - 79 profiles, and exactly the bright NOC1 where pull-up rides along - read 1.49 and 4.42, which then
# became `_CR_MAX` in the generator: the emission was inflated 4.4x instead of 1.5x and the SHAPE of the surviving
# floor changed with it (tall noise kept at rate 1, faint at 0.16). The column trend is what this table is for, so a
# cell that cannot support one is not allowed to invent it.
MINP = 40
for _b in range(M.shape[0]):
    for _k in range(1, M.shape[1]):
        if CR[_k, _b] < MINP or CT[_k, _b] < MINP:
            M[_b, _k] = M[_b, _k - 1]
# THE TALL BANDS DO NOT FOLLOW THE COLUMN. Their ratio runs 0.155 -> 4.077 across the columns, i.e. real has FEWER
# tall floor peaks than the twin in quiet profiles and far more in bright ones - that is pull-up riding on brightness,
# not crowding. Left column-indexed, a mixture (100+ tall peaks) reads the LAST column, `_keep` clips at 1 and tall
# floor peaks are never thinned at all: measured on the 1333 test mixtures with alleles and artefacts copied from real,
# the twin puts 0.16-0.19 peaks above 40 RFU at free bins against real's 0.03, which is the whole +14...+21 % excess in
# the floor's RFU. Held at the sparse column, where pull-up is not driving it.
TALL_FROM = int(os.environ.get("CR_TALL_FROM", "3"))     # bands 16-25 RFU and up
for _b in range(TALL_FROM, M.shape[0]):
    M[_b, :] = M[_b, 0]
print("so dinh nhieu real theo cot (hang = dai cao):")
for _b in range(M.shape[0]):
    print("   dai %d: %s" % (_b, "  ".join("%6.0f" % CR[_k, _b] for _k in range(M.shape[1]))))
S = SM / np.maximum(NN, 1)                                      # each column MEAN count: the abscissa
# Past NOC1 range each band continues along its own log-linear trend when that trend is a SUPPRESSION; a band
# whose trend rises is held at its edge, since a rise is brightness riding along, not a crowding law.
EXT = []
for b in range(len(HB) - 1):
    w = np.sqrt(np.maximum(CR[:, b], 1.0))
    EXT.append(float(min(np.polyfit(S, np.log(np.maximum(M[b], 1e-6)), 1, w=w)[0], 0.0)))
json.dump({"noise_cr_s": [round(float(x), 2) for x in S], "noise_cr_h": [4.0, 10.0, 14.0, 20.0, 40.0],
           "noise_cr_m": np.round(M, 4).tolist(), "noise_cr_ext": [round(x, 6) for x in EXT],
           "noise_cr_th": CR_TH,
           "note": "real/twin noise-peak ratio on TRAIN NOC1, twin with the crowding law off; column = peaks >= 50 RFU on the real profile; code/calibrate_crowd.py"},
          open(REAL / "noise_cr_inc.json", "w"), indent=1)
print("cot (so dinh >= %.0f RFU) trung binh:" % CR_TH, np.round(S, 1))
print("do doc ngoai suy theo dai cao:", [round(x, 5) for x in EXT])
print("n theo cot do chat:", NN.astype(int))
print("ti so real / twin(khong giam) theo cot do chat (hang) va dai cao (cot):", HB)
for k in range(len(SB) - 1):
    print("%-10s" % ("%d-%s" % (SB[k], "" if SB[k + 1] > 1e8 else int(SB[k + 1]))) + "".join("%9.3f" % (CR[k, j] / max(CT[k, j], 1)) for j in range(len(HB) - 1))
          + "   tong %.3f" % (CR[k].sum() / max(CT[k].sum(), 1)))
