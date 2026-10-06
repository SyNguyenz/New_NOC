"""calibrate_pullup.py - pull-up as a peak the generator EMITS, not only one it deletes.

The arrays are PROVEDIt's "Filtered" export, so the obvious pull-up is gone - but not all of it. On held-out NOC1 the
peaks that are neither a plausible stutter nor ordinary baseline sit in pull-up context (another dye channel within
1 bp, at least 10x taller) 6.7-8.5 % of the time in real against 0.9-1.8 % in the twin: the generator models the
FILTER (PULL_P) and never emits a surviving pull-up at all. That missing process is the size of the twin's whole
shortfall in the high tail at stutter positions fed by two or more alleles (~0.3 peaks per profile, at every NOC), and
those are the peaks that open a decoy.

Measured here on real NOC1 (TRAIN + VAL only) as the increment over what the generator already makes: for every bin
that is not one of the donor's alleles - INCLUDING the stutter positions, which is where it lands: pull-up comes from a
tall peak, the tall peaks are alleles, and an allele's stutter positions sit within a base pair of it, so the bleed
falls on them and pushes them above the stutter law (excluding those bins, as the first two versions of this file did,
found nothing at all) - the tallest peak in another dye channel within D_EDGES[-1] bp is the cause, and the pair
(cause height band, |distance| band) gives
  rate  = P(a peak stands here above BOTH the stutter law (x3) and the profile's baseline p90 (xTALL_K)) in real
          minus the same in its twin, floored at 0
  ratio = median (height - what the stutter law expects there) / cause height, from real, over those same peaks
"Above the noise floor" is TALL_K times the profile's own p90 of baseline peaks: counting any peak at all measures the
floor, which the twin already fills at the same rate (the first version of this file did that and found 0.03 extra
peaks per profile, while the class that is actually missing is the tall one). Emission is per bin and independent,
which is what a spectral bleed is.

Output: data/pullup_emit.json {"h_edges", "d_edges", "rate", "ratio"} read by make_insilico (PU_RATE / PU_RATIO).
Usage:  python calibrate_pullup.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import kit

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STR_DATA_DIR", str(HERE / "data")))
os.environ["STR_PU_EMIT"] = "0"                            # the increment is over the generator WITHOUT emission
os.environ.setdefault("STR_DATA_DIR", str(DATA))
_s = importlib.util.spec_from_file_location("make_insilico", HERE / "make_insilico.py")
MI = importlib.util.module_from_spec(_s); _s.loader.exec_module(MI)
import calibrate as CAL                                     # noqa: E402  (pull_neighbours, the edge grids)

# RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)GF-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)" + kit.TAG + r"-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
H_EDGES = [200.0, 600.0, 1500.0, 4000.0, 1e12]              # the causing peak's height
D_EDGES = [0.0, 0.5, 1.0, 2.0]                              # |distance| in bp
MIN_N = 40
TALL_K = 2.0                                                # a peak this many times the profile's baseline p90 counts


OFFS = (-20, -10, 10, 20)


def _free_bins(donor):
    """bins that are not an allele of this donor: the stutter positions are kept, see the header"""
    return ~(np.asarray(MI.DONOR_DOSAGE[donor]) > 0)


def _stut_exp(h, own):
    """what the generator's own stutter law expects at each bin, from this profile's standing alleles"""
    e = np.zeros(len(h))
    for d in OFFS:
        t = MI._OFF_TARGET.get(d)
        if t is None:
            continue
        src = np.flatnonzero(own & (h > 0) & (t >= 0))
        for p in src:
            rows = (MI.noc1_calib().get("art_table_loc") or {})
            row = None
            for r in (rows.get(int(MI.BIN_LOCUS[p])) or rows.get(str(int(MI.BIN_LOCUS[p]))) or []):
                if int(r[0]) == d:
                    row = r; break
            if row is None:
                continue
            lr = (row[2] + (row[4] if len(row) > 4 else 0.0) * (float(MI.BIN_ALLELE[p]) - (row[5] if len(row) > 5 else 0.0))
                  + np.log(max(float(row[1]), 1e-6)))
            if d in MI.ALLELE_STUT_OFF:
                lr = lr + MI.ALLELE_STUT_OFF[d][p]
            m = 1.0
            a_ = MI.STUT_LT_A.get(d, 0.0)
            if a_ > 0 and MI.STUT_LT_G.get(d, 0.0) > 0:
                m = min(1.0 + a_ * (max(h[p], 1.0) / 100.0) ** (-MI.STUT_LT_G[d]), MI.STUT_LT_CAP)
            elif a_ > 0:
                m = 1.0 + a_
            e[t[p]] += h[p] * np.exp(lr) * m
    return e


def _cells(h, free, nb, nd, tall, exp):
    """per bin: (cause height band, distance band, does an unexplained TALL peak stand, excess / cause height)"""
    hp = np.append(np.asarray(h, float), 0.0)
    idx = np.flatnonzero(free)
    hh = hp[nb[idx]]; a = hh.argmax(1); r_ = np.arange(len(idx))
    hm = hh[r_, a]; dm = nd[idx][r_, a]
    hi = np.searchsorted(H_EDGES, hm, side="right") - 1
    di = np.searchsorted(D_EDGES, dm, side="right") - 1
    ok = (hi >= 0) & (di >= 0) & (di < len(D_EDGES) - 1) & (hi < len(H_EDGES) - 1)
    hh_ = h[idx][ok]; ex = exp[idx][ok]
    st = (hh_ > tall) & (hh_ > 3.0 * np.maximum(ex, 1e-9))
    return hi[ok], di[ok], st, np.maximum(hh_ - ex, 0.0) / np.maximum(hm[ok], 1e-9)


def main():
    pool = MI.build_ss_pool(); bs = MI.build_bin_size()
    nb, nd = CAL.pull_neighbours(MI.BIN_LOCUS, MI._LOCUS_NAMES, np.asarray(bs, float), win=D_EDGES[-1])
    nH, nD = len(H_EDGES) - 1, len(D_EDGES) - 1
    seen = {t: np.zeros((nH, nD)) for t in ("real", "twin")}
    hit = {t: np.zeros((nH, nD)) for t in ("real", "twin")}
    rat = [[[] for _ in range(nD)] for _ in range(nH)]
    n = 0
    for split, seed0 in (("train", 31000), ("val", 700)):
        names = json.load(open(DATA / f"meta_sample_names_{split}.json"))
        X = np.load(DATA / f"Xflat_{split}.npy"); Y = np.load(DATA / f"y_{split}_set.npy")
        noc = np.load(DATA / f"noc_{split}.npy")
        for i in np.flatnonzero(noc == 1):
            m = RX.search(str(names[i]))
            if not m or int(m.group(1)) not in MI.COL:
                continue
            c = int(np.argmax(Y[i])); z = np.expm1(X[i].astype(np.float64))
            rng = np.random.default_rng(seed0 + n); n += 1
            xf = MI.gen_mixture([c], pool, rng, t_total=float(z.sum()), bin_size=bs, phi=np.array([1.0]),
                                cond=m.group(3) or "a", ng_total=float(m.group(4)), inj_sec=float(m.group(6)),
                                q_tube=float(m.group(5)))[0]
            free = _free_bins(c)
            for tag, h in (("real", z), ("twin", np.expm1(np.asarray(xf, np.float64)))):
                own_ = ~free
                exp_ = _stut_exp(h, own_)
                _bl = h[free & (h > 0) & (exp_ <= 0)]
                tall = TALL_K * (np.quantile(_bl, .9) if len(_bl) > 5 else 15.0)
                hi, di, st, rr = _cells(h, free, nb, nd, tall, exp_)
                np.add.at(seen[tag], (hi, di), 1.0)
                np.add.at(hit[tag], (hi, di), st.astype(float))
                if tag == "real":
                    for a_, b_, s_, v_ in zip(hi, di, st, rr):
                        if s_:
                            rat[a_][b_].append(float(v_))
    rate = np.zeros((nH, nD)); ratio = np.zeros((nH, nD))
    for a_ in range(nH):
        for b_ in range(nD):
            if seen["real"][a_, b_] < MIN_N or seen["twin"][a_, b_] < MIN_N:
                continue
            p_r = hit["real"][a_, b_] / seen["real"][a_, b_]; p_t = hit["twin"][a_, b_] / seen["twin"][a_, b_]
            rate[a_, b_] = max(p_r - p_t, 0.0)
            ratio[a_, b_] = float(np.median(rat[a_][b_])) if len(rat[a_][b_]) >= 10 else 0.0
    json.dump({"h_edges": H_EDGES, "d_edges": D_EDGES, "rate": rate.tolist(), "ratio": ratio.tolist()},
              open(DATA / "pullup_emit.json", "w"))
    print(f"pullup_emit.json from {n} NOC1 profiles")
    print("  cause RFU       " + "  ".join("%4.1f-%-4.1f bp" % (D_EDGES[b], D_EDGES[b + 1]) for b in range(nD)))
    for a_ in range(nH):
        lab = "%5.0f-%-6s" % (H_EDGES[a_], "inf" if H_EDGES[a_ + 1] > 1e11 else "%.0f" % H_EDGES[a_ + 1])
        print("  " + lab + "  " + "  ".join("rate %.4f r %.4f" % (rate[a_, b], ratio[a_, b]) for b in range(nD)))
    print("  expected extra peaks per NOC1 profile: %.2f" % float((rate * seen["real"] / max(n, 1)).sum()))


if __name__ == "__main__":
    main()
