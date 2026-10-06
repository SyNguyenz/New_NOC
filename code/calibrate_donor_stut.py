"""calibrate_donor_stut.py - each DONOR's allele's n-1 emission, as a logit increment over the generator.

The generator's n-1 laws key on the allele's length (ALLELE_RATE per bin, the per-locus table, STUT_PH per parent
band). Real NOC1 says a same-length allele of one donor can almost never stutter while the same allele of the other
donors always does: D2S441 14 n-1 at parents >= 1200 RFU 1.00 in donors 15/44/25/35 and .04 in 16/36, D5S818 11 1.00
against .00 (donor 27), TH01 9, D22S1045 16, D19S433 13, D7S820 8, D13S317 8 alike, over 20-50 runs each, the ratio
to the parent normal whenever the n-1 does stand. Split in halves of the runs, 71 pairs flagged on one half stood
.17-.19 on the other against the generator's .90-.95. Same length, different sequence (an interrupted repeat, a short
longest uninterrupted stretch) is the known cause; the data here has no sequence, so it is measured per donor.

Per (donor, parent allele): n-1 presence at parents >= 600 RFU (where survival is ~1 and presence is emission), at
targets that are not an allele and that no allele reaches at a 1-3 bp offset (the shoulder), real vs the twin of the
same profile; offset = logit(real) - logit(twin), shrunk toward the twin by K_PSEUDO pseudo-runs (see main),
a pseudo-count, not the normal prior of calibrate_allele_stut.py. Written as an increment over the generator WITHOUT these offsets, so
calibrate_stut_parent.py (run after it) fits its parent-height law with them in place.
AMEL and Yindel are left out. TEST is not read.

Output: data/donor_stut.json {donor column: {parent bin: logit}}, read by make_insilico (DONOR_RATE).
Usage:  python calibrate_donor_stut.py
"""
from __future__ import annotations

import collections
import importlib.util
import json
import os
import re
from pathlib import Path

import numpy as np
import kit

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("STR_DATA_DIR", str(HERE / "data")))
os.environ["STR_DONOR_STUT"] = "0"                     # the increment is over the generator WITHOUT itself
os.environ.setdefault("STR_DATA_DIR", str(DATA))
_s = importlib.util.spec_from_file_location("make_insilico", HERE / "make_insilico.py")
MI = importlib.util.module_from_spec(_s); _s.loader.exec_module(MI)

# RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)GF-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)" + kit.TAG + r"-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
H_MIN = 600.0
MIN_N = 8
K_PSEUDO = 5.0                                          # runs of prior weight on the twin (see main)
BPOFF = np.array([-0.75, -0.5, -0.25, 0.25, 0.5, 0.75])
OWN = np.asarray(MI.DONOR_DOSAGE) > 0
BL = np.asarray(MI.BIN_LOCUS, int)
_W = np.floor(MI.BIN_ALLELE + 1e-6); UN = _W + np.round((MI.BIN_ALLELE - _W) * 10) / 4.0


def specs():
    out = []
    for split, seed0 in (("train", 40000), ("val", 1100)):
        names = json.load(open(DATA / f"meta_sample_names_{split}.json"))
        X = np.load(DATA / f"Xflat_{split}.npy"); noc = np.load(DATA / f"noc_{split}.npy")
        for n, i in enumerate(np.flatnonzero(noc == 1)):
            m = RX.search(str(names[i]))
            if m and int(m.group(1)) in MI.COL:
                out.append((MI.COL[int(m.group(1))], np.expm1(X[i].astype(np.float64)), m, seed0 + n))
    return out


def targets(c):
    """n-1 target of every allele of donor c that is not an allele itself and no allele reaches at 1-3 bp"""
    oa = np.flatnonzero(OWN[c] & ~MI.NO_STUTTER_BIN); t1 = MI._OFF_TARGET[-10]; out = []
    for p in oa:
        t = t1[p]
        if t < 0 or OWN[c, t]:
            continue
        al = oa[BL[oa] == BL[t]]
        if np.isclose((UN[t] - UN[al])[:, None], BPOFF[None, :]).any():
            continue
        out.append((int(p), int(t)))
    return out


def collect(sp, profiles):
    acc = collections.defaultdict(list)
    for (c, _, _, _), h in zip(sp, profiles):
        for p, t in targets(c):
            if h[p] >= H_MIN:
                acc[(c, p)].append(float(h[t] > 0))
    return acc


def logit(p):
    p = float(np.clip(p, 0.02, 0.98))
    return np.log(p / (1.0 - p))


def main():
    pool = MI.build_ss_pool(); bs = MI.build_bin_size()
    sp = specs()
    tw = [np.expm1(np.asarray(MI.gen_mixture(
        [c], pool, np.random.default_rng(sd), t_total=float(z.sum()), bin_size=bs, phi=np.array([1.0]),
        cond=m.group(3) or "a", ng_total=float(m.group(4)), inj_sec=float(m.group(6)), q_tube=float(m.group(5)))[0],
        np.float64)) for c, z, m, sd in sp]
    R = collect(sp, [s[1] for s in sp]); T = collect(sp, tw)
    rows = []
    for key in set(R) & set(T):
        a, b = np.array(R[key]), np.array(T[key])
        if len(a) < MIN_N or len(b) < MIN_N:
            continue
        pa = float(np.clip(a.mean(), 0.02, 0.98)); pb = float(np.clip(b.mean(), 0.02, 0.98))
        se2 = 1.0 / (len(a) * pa * (1 - pa)) + 1.0 / (len(b) * pb * (1 - pb))
        rows.append((key, logit(a.mean()) - logit(b.mean()), se2, a.mean(), b.mean(), len(a)))
    # Shrinkage by PSEUDO-COUNTS, not a normal prior: the true offsets are not normal around 0 - most pairs sit at 0
    # and a few near -7 (never stutters) - and a normal empirical-Bayes prior (tau .70) pulled those few to -1.3..-1.7,
    # which leaves the emission at ~.8 where real shows .02. Here each pair's real presence is pulled toward the twin's
    # by K_PSEUDO runs' worth: 1 of 45 becomes .12, 0 of 8 becomes .37.
    raw = {}
    for (c, p), diff, s2, ra, tb, n in rows:
        ppost = (ra * n + K_PSEUDO * tb) / (n + K_PSEUDO)
        raw[(c, p)] = (float(np.clip(logit(ppost) - logit(tb), -6.0, 3.0)), n)
    # CENTRED per allele (run-weighted mean 0): only the differences BETWEEN donors of one allele belong here. Left
    # uncentred, 349 pairs came out above +1 - not donors, but the generator's general n-1 shortfall on the normal pairs
    # (.950 against .970, the parent-height law having been fitted to a pool that held the non-stuttering pairs), which
    # calibrate_stut_parent.py, run next, charges again. An allele carried by one donor gets 0: its level is the
    # allele law's.
    byp = collections.defaultdict(list)
    for (c, p), (z, n) in raw.items():
        byp[p].append((c, z, n))
    out = collections.defaultdict(dict)
    for p, v in byp.items():
        mu = sum(z * n for _, z, n in v) / sum(n for _, _, n in v)
        for c, z, n in v:
            out[str(c)][str(p)] = round(float(z - mu), 4)
    json.dump(out, open(DATA / "donor_stut.json", "w"), indent=1)
    zs = np.array([v for dd in out.values() for v in dd.values()])
    print(f"{len(sp)} NOC1 profiles; {len(rows)} (donor, allele) pairs with >= {MIN_N} cases on both sides")
    print(f"offsets: sd {zs.std():.3f}, < -2: {(zs < -2).sum()}, < -1: {(zs < -1).sum()}, > +1: {(zs > 1).sum()}")
    worst = sorted(rows, key=lambda r: r[1])[:12]
    loci = json.load(open(DATA / "meta_set.json"))["loci"]
    inv = {v: k for k, v in MI.COL.items()}
    for (c, p), diff, s2, ra, tb, n in worst:
        print(f"  donor {inv.get(c, c):>3} {loci[BL[p]]:<9} {MI.BIN_ALLELE[p]:<5} real {ra:.2f} twin {tb:.2f} (n {n})"
              f"  offset {out[str(c)][str(p)]:+.2f}")


if __name__ == "__main__":
    main()
