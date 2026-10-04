"""gen_law.py - a snapshot of the generator's laws, saved next to the training arrays.

make_insilico writes `gen_law.npy` with the data it builds: the stutter law (per offset, per parent bin, each allele's
own rate and ratio), the low-template stutter factor, the donor dosages and the presence curve (P(an allele stands |
the template its locus partner shows), fitted on real NOC1). The decode layer reads it, so what the decode subtracts
and expects is exactly what the data was made with.
"""
from __future__ import annotations

import numpy as np

LAW_FILE = "gen_law.npy"
STAND_THR = 10.0                 # an allele counts as standing above this RFU (decode and presence curve)


def build_law(mi) -> dict:
    """Snapshot of the generator's stutter law and donor alleles (mi = the make_insilico module)."""
    loc = mi.noc1_calib().get("art_table_loc") or {}
    offs = sorted({int(r[0]) for rows in loc.values() for r in rows})
    a = mi.BIN_ALLELE.astype(float)
    law = {}
    offsets = getattr(mi, "ALLELE_STUT_OFF", {-10: mi.ALLELE_STUT})
    rates = getattr(mi, "ALLELE_RATE", {})
    for d in offs:
        lratio = np.full(mi.N_FLAT, np.nan); rate = np.full(mi.N_FLAT, np.nan)
        for p in range(mi.N_FLAT):
            for r in (loc.get(int(mi.BIN_LOCUS[p])) or loc.get(str(int(mi.BIN_LOCUS[p]))) or []):
                if int(r[0]) == d:                              # the table: ratio and emission rate at this parent
                    lratio[p] = r[2] + (r[4] if len(r) > 4 else 0.0) * (a[p] - (r[5] if len(r) > 5 else 0.0))
                    rate[p] = float(r[1])
        if d in offsets:
            lratio = lratio + offsets[d]                         # each allele's own ratio
        if d in rates:
            rate = np.minimum(rate * rates[d], 0.99)             # each allele's own emission rate (as the generator)
        law[d] = lratio + np.log(np.maximum(rate, 1e-6))         # log(rate x ratio): the expected stutter share
    keys = np.array([(int(mi.BIN_LOCUS[j]), int(round(float(mi.BIN_ALLELE[j]) * 10))) for j in range(mi.N_FLAT)])
    return {"offs": offs, "law": law, "pres": presence_curve(mi, STAND_THR), "target": {d: np.asarray(mi._OFF_TARGET[d]) for d in offs},
            "lt_a": dict(mi.STUT_LT_A), "lt_g": dict(mi.STUT_LT_G), "lt_cap": float(mi.STUT_LT_CAP),
            "dos": np.asarray(mi.DONOR_DOSAGE, float), "bin_locus": np.asarray(mi.BIN_LOCUS, int), "keys": keys}


def presence_curve(mi, thr: float):
    """P(an allele stands above thr | the OTHER allele of its heterozygous locus stands at height H), logistic in
    log H, fitted on the data dir's real NOC1 single sources. The partner measures the template without carrying the
    allele's own draw, so this is the dropout law the walk needs, at the walk's own threshold."""
    X = np.load(mi.DATA / "Xflat_train.npy", mmap_mode="r"); Y = np.load(mi.DATA / "y_train_set.npy")
    N = np.load(mi.DATA / "noc_train.npy"); dos = np.asarray(mi.DONOR_DOSAGE); bl = np.asarray(mi.BIN_LOCUS, int)
    xs, ys = [], []
    for i in np.flatnonzero(N == 1):
        c = int(np.argmax(Y[i])); h = np.expm1(np.asarray(X[i], np.float64))
        for L in np.unique(bl[dos[c] > 0]):
            ab = np.flatnonzero((bl == L) & (dos[c] == 1))
            if len(ab) != 2:
                continue
            for a, b in ((ab[0], ab[1]), (ab[1], ab[0])):
                if h[b] > 0:
                    xs.append(np.log(h[b])); ys.append(float(h[a] > thr))
    x = np.array(xs); y = np.array(ys); w = np.zeros(2); A = np.stack([np.ones_like(x), x], 1)
    for _ in range(25):                                          # logistic regression by Newton steps
        pr = 1.0 / (1.0 + np.exp(-A @ w))
        w = w + np.linalg.solve((A * (pr * (1 - pr))[:, None]).T @ A + 1e-9 * np.eye(2), A.T @ (y - pr))
    return (float(w[0]), float(w[1]))


def load_law(path) -> dict:
    law = np.load(path, allow_pickle=True).item()
    law["index"] = {(int(l), int(a)): j for j, (l, a) in enumerate(law["keys"])}
    law["own"] = law["dos"] > 0
    law["carried"] = law["own"].any(0)                           # bins some panel donor carries: the background pool
    return law


def profile_heights(tokens: np.ndarray, mask: np.ndarray, law: dict) -> np.ndarray:
    """(N, 160, >=3) tokens [locus, allele, log1p height] -> (N, N_FLAT) RFU per allele bin."""
    ix = law["index"]; out = np.zeros((len(tokens), len(law["keys"])))
    for i in range(len(tokens)):
        for p in np.flatnonzero(mask[i]):
            j = ix.get((int(round(float(tokens[i, p, 0]))), int(round(float(tokens[i, p, 1]) * 10))))
            if j is not None:
                out[i, j] += float(np.expm1(tokens[i, p, 2]))
    return out


def stutter_of(contrib: np.ndarray, law: dict) -> np.ndarray:
    """Expected stutter RFU cast by an allele-height vector, under the generator's law."""
    e = np.zeros_like(contrib)
    for d in law["offs"]:
        tgt = law["target"][d]; lr = law["law"][d]
        p = np.flatnonzero((contrib > 0) & (tgt >= 0) & np.isfinite(lr))
        if not len(p):
            continue
        m = np.ones(len(p))
        a_ = law["lt_a"].get(d, 0.0)
        if a_ > 0 and law["lt_g"].get(d, 0.0) > 0:               # low-template factor (STUT_LT_A)
            m = np.minimum(1.0 + a_ * (np.maximum(contrib[p], 1.0) / 100.0) ** (-law["lt_g"][d]), law["lt_cap"])
        elif a_ > 0:                                              # flat factor (G = 0)
            m = m * (1.0 + a_)
        np.add.at(e, tgt[p], contrib[p] * np.exp(lr[p]) * m)
    return e
