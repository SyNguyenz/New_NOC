"""calibrate_scatter.py - width of the stutter ratio draw, per offset, from the tail real actually shows.

The artefact table's `lsd` was fitted to the pooled ratio distribution, whose median and p90 it still matches - but the
TAIL it does not. Isolating stutter from the baseline (keep only bins where three times the expectation already exceeds
the profile's tallest baseline peak, so the floor cannot lift the bin over 3x on its own) real has 1.61 % of those bins
standing above 3x the expectation at a single-source position against the generator's 1.38 %, and the shortfall is the
same fifth at two or more sources. Widening the draw closes it with peaks per profile and the artefact share untouched,
while artefact survival - which two earlier sweeps blamed, both of them scoring a statistic that turned out to be ~80 %
baseline noise - only gets two thirds of the way and costs 8.6 peaks per profile.

Per offset, a factor on `lsd` is chosen on TRAIN + VAL NOC1 so that the share of clean-zone single-source bins standing
above 3x the expectation matches real's. TEST NOC1 is not read.

Output: data/art_scatter.json {offset in tenths: factor}, read by make_insilico (ART_SCAT_D).
Usage:  python calibrate_scatter.py
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STR_DATA_DIR", str(HERE / "data")))
os.environ["STR_ART_SCAT_D"] = "0"                         # the factor is fitted over the generator WITHOUT it
os.environ.setdefault("STR_DATA_DIR", str(DATA))
_s = importlib.util.spec_from_file_location("make_insilico", HERE / "make_insilico.py")
MI = importlib.util.module_from_spec(_s); _s.loader.exec_module(MI)

RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)GF-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
OFFS = (-20, -10, 10, 20)
GRID = (1.0, 1.1, 1.2, 1.3, 1.45, 1.6)
MIN_N = 400


def _law():
    """log(rate x ratio) per parent bin, per offset, as the generator computes it"""
    loc = MI.noc1_calib().get("art_table_loc") or {}
    a = MI.BIN_ALLELE.astype(float); out = {}
    for d in OFFS:
        arr = np.full(MI.N_FLAT, np.nan)
        for p in range(MI.N_FLAT):
            for r in (loc.get(int(MI.BIN_LOCUS[p])) or loc.get(str(int(MI.BIN_LOCUS[p]))) or []):
                if int(r[0]) == d:
                    arr[p] = (r[2] + (r[4] if len(r) > 4 else 0.0) * (a[p] - (r[5] if len(r) > 5 else 0.0))
                              + np.log(max(float(r[1]), 1e-6)))
        if d in MI.ALLELE_STUT_OFF:
            arr = arr + MI.ALLELE_STUT_OFF[d]
        out[d] = arr
    return out


LAW = _law()


def expect(h, own):
    """per bin: the summed expectation, and the single offset feeding it (0 when several do)"""
    E = np.zeros(len(h)); who = np.zeros(len(h), int); cnt = np.zeros(len(h), int)
    for d in OFFS:
        t = MI._OFF_TARGET[d]; lr = LAW[d]
        for p in np.flatnonzero(own & (h > 0) & (t >= 0)):
            if not np.isfinite(lr[p]) or own[t[p]]:
                continue
            m = 1.0
            a_ = MI.STUT_LT_A.get(d, 0.0)
            if a_ > 0 and MI.STUT_LT_G.get(d, 0.0) > 0:
                m = min(1.0 + a_ * (max(h[p], 1.0) / 100.0) ** (-MI.STUT_LT_G[d]), MI.STUT_LT_CAP)
            elif a_ > 0:
                m = 1.0 + a_
            E[t[p]] += h[p] * np.exp(lr[p]) * m; who[t[p]] = d; cnt[t[p]] += 1
    return E, np.where(cnt == 1, who, 0)


def share(profiles, pool, bs, generate):
    """per offset: share of clean-zone single-source bins standing above 3x the expectation"""
    hit = {d: 0 for d in OFFS}; tot = {d: 0 for d in OFFS}
    for n, (c, z, m) in enumerate(profiles):
        own = np.asarray(MI.DONOR_DOSAGE[c]) > 0
        if generate:
            rng = np.random.default_rng(41000 + n)
            xf = MI.gen_mixture([c], pool, rng, t_total=float(z.sum()), bin_size=bs, phi=np.array([1.0]),
                                cond=m.group(3) or "a", ng_total=float(m.group(4)), inj_sec=float(m.group(6)),
                                q_tube=float(m.group(5)))[0]
            h = np.expm1(np.asarray(xf, np.float64))
        else:
            h = z
        E, who = expect(h, own)
        free = (E <= 0) & ~own
        fl = h[free & (h > 0)]
        if len(fl) < 20:
            continue
        clean = (3.0 * E > float(fl.max())) & (E > 0) & ~own
        for d in OFFS:
            sel = clean & (who == d)
            tot[d] += int(sel.sum()); hit[d] += int(((h > 3 * E) & sel).sum())
    return {d: (hit[d] / tot[d] if tot[d] else np.nan, tot[d]) for d in OFFS}


def main():
    pool = MI.build_ss_pool(); bs = MI.build_bin_size()
    prof = []
    for split in ("train", "val"):
        names = json.load(open(DATA / f"meta_sample_names_{split}.json"))
        X = np.load(DATA / f"Xflat_{split}.npy"); noc = np.load(DATA / f"noc_{split}.npy")
        for i in np.flatnonzero(noc == 1):
            m = RX.search(str(names[i]))
            if m and int(m.group(1)) in MI.COL:
                prof.append((MI.COL[int(m.group(1))], np.expm1(X[i].astype(np.float64)), m))
    print(f"NOC1 train+val: {len(prof)} profiles")
    real = share(prof, pool, bs, False)
    print("  real : " + "  ".join("%+d %.4f (n=%d)" % (d // 10, real[d][0], real[d][1]) for d in OFFS), flush=True)
    best = {}
    for k in GRID:
        MI.ART_SCAT_D = {d: k for d in OFFS}
        g = share(prof, pool, bs, True)
        print("  x%.2f : " % k + "  ".join("%+d %.4f" % (d // 10, g[d][0]) for d in OFFS), flush=True)
        for d in OFFS:
            if real[d][1] < MIN_N or not np.isfinite(g[d][0]):
                continue
            e = abs(g[d][0] - real[d][0])
            if d not in best or e < best[d][0]:
                best[d] = (e, k)
    # Only an offset whose target the grid actually REACHES is written. Measured per offset the picture is nothing like
    # the pooled one: n-2 real 0.0832 against 0.0562 and x1.45 lands on it; n+1 real 0.0397 against the generator's
    # 0.0444, so it needs no widening at all; n-1 real 0.0039 against 0.0009 and even x1.6 only reaches 0.0013 - short by
    # three times, so widening is not its mechanism and 1.6 would only be the edge of the grid. The pooled "x1.2 matches
    # real" that first pointed here was the average of n-1 under and n+1 over, i.e. an artefact of pooling.
    out = {}
    for d in best:
        e, k = best[d]
        if e <= 0.15 * real[d][0]:
            out[str(d)] = k
        else:
            print("  %+d: gan nhat x%.2f con lech %.4f tren muc tieu %.4f - KHONG ghi" % (d // 10, k, e, real[d][0]))
    json.dump(out, open(DATA / "art_scatter.json", "w"))
    print("art_scatter.json:", out)


if __name__ == "__main__":
    main()
