"""calibrate_stut_spread.py - how the n-1 ratio's scatter shrinks with the parent, as a factor on the draw's width.

The artefact table carries one width (lsd) per locus and offset, shared by the run offset, the locus offset and the
per-peak lognormal. Real NOC1 says the scatter is not one number: within a donor's allele, the sd of log(n-1/parent)
falls from .368 at 150-300 RFU parents to .108 above 3000 (TRAIN+VAL, identical parents), while the generator stayed
near .33-.25 - the early-cycle copy lottery, many copies and little scatter. A factor g per parent band multiplies the
whole width (make_insilico STUT_SPREAD) and is iterated until the twin's within-band sd matches real's.

Statistic (identical on both sides, each side on its own profiles): at n-1 targets of standing parents that no other
allele of the donor reaches (target or parent, -3..+3 repeats or 1-3 bp; AMEL/Yindel out), log(n-1/parent) minus that
side's (donor, allele) mean at parents >= 1200 RFU, sd within each parent band, n-1 visible only. The same
decomposition on a fake-real reproduced every component to within .002-.007, so the statistic carries no harness bias.
TRAIN + VAL NOC1 (every second TRAIN profile, for time); TEST is not read. Run before calibrate_stut_parent.py: the
width moves how often a stutter clears the floor, which that law is fitted over.

Output: data/stut_spread.json {"-10": {"edges": [...], "g": [...]}}, read by make_insilico (STUT_SPREAD).
Usage:  python calibrate_stut_spread.py
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
os.environ["STR_STUT_SPREAD"] = "0"                    # fitted over the generator without it, then set in memory
os.environ.setdefault("STR_DATA_DIR", str(DATA))
_s = importlib.util.spec_from_file_location("make_insilico", HERE / "make_insilico.py")
MI = importlib.util.module_from_spec(_s); _s.loader.exec_module(MI)

# RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)GF-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)" + kit.TAG + r"-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
EDGES = np.array([0.0, 300.0, 600.0, 1200.0, 3000.0, 1e9])
ITERS = int(os.environ.get("STR_STUT_SPREAD_ITERS", "4"))
BADO = np.array([-3, -2, -1, 1, 2, 3, -.75, -.5, -.25, .25, .5, .75])
# A visible n-1 more than e (x2.7) below its (donor, allele) mean is not a stutter that scattered but a floor peak
# on a target whose stutter was not emitted or kept: real NOC1 has none (0 of 3957 at parents > 3000), the twin had
# 0.26 % at residual ~ -2.9, and those ten rows alone made its sd .163 where the stutter-carried rows gave .058 -
# which drove g to its floor and left the twin half as wide as real. Dropped on both sides.
TRIM = -1.0
OWN = np.asarray(MI.DONOR_DOSAGE) > 0
BL = np.asarray(MI.BIN_LOCUS, int)
_W = np.floor(MI.BIN_ALLELE + 1e-6); UN = _W + np.round((MI.BIN_ALLELE - _W) * 10) / 4.0
T1 = MI._OFF_TARGET[-10]


def specs():
    out = []
    for split, step, seed0 in (("train", 2, 50000), ("val", 1, 1300)):
        names = json.load(open(DATA / f"meta_sample_names_{split}.json"))
        X = np.load(DATA / f"Xflat_{split}.npy"); noc = np.load(DATA / f"noc_{split}.npy")
        for n, i in enumerate(np.flatnonzero(noc == 1)[::step]):
            m = RX.search(str(names[i]))
            if m and int(m.group(1)) in MI.COL:
                out.append((MI.COL[int(m.group(1))], np.expm1(X[i].astype(np.float64)), m, seed0 + n))
    return out


def rows(profiles, donors):
    out = []
    for c, h in zip(donors, profiles):
        oa = np.flatnonzero(OWN[c] & ~MI.NO_STUTTER_BIN)
        for p in oa[h[oa] >= 150]:
            t = T1[p]
            if t < 0 or OWN[c, t]:
                continue
            al = oa[BL[oa] == BL[p]]
            if any(q != p and (np.isclose(UN[t] - UN[q], BADO).any() or np.isclose(UN[p] - UN[q], BADO).any()) for q in al):
                continue
            out.append((c, int(p), h[p], np.log(h[t] / h[p]) if h[t] > 0 else np.nan))
    return out


def band_sd(R):
    m = collections.defaultdict(list)
    for c, p, hp, lr in R:
        if hp >= 1200 and np.isfinite(lr):
            m[(c, p)].append(lr)
    mu = {k: np.mean(v) for k, v in m.items() if len(v) >= 3}
    out = []
    for b in range(len(EDGES) - 1):
        x = [lr - mu[(c, p)] for c, p, hp, lr in R if EDGES[b] <= hp < EDGES[b + 1] and np.isfinite(lr) and (c, p) in mu
             and lr - mu[(c, p)] >= TRIM]
        out.append((float(np.std(x)), len(x)) if len(x) >= 100 else (np.nan, len(x)))
    return out


def main():
    pool = MI.build_ss_pool(); bs = MI.build_bin_size()
    sp = specs(); donors = [s[0] for s in sp]
    real = band_sd(rows([s[1] for s in sp], donors))
    g = np.ones(len(EDGES) - 1)
    print(f"{len(sp)} NOC1 profiles; real within-band sd: " + " ".join(f"{v:.3f}(n{n})" for v, n in real))
    for it in range(ITERS + 1):
        MI.STUT_SPREAD[-10] = (EDGES, g.copy())
        tw = [np.expm1(np.asarray(MI.gen_mixture(
            [c], pool, np.random.default_rng(sd), t_total=float(z.sum()), bin_size=bs, phi=np.array([1.0]),
            cond=m.group(3) or "a", ng_total=float(m.group(4)), inj_sec=float(m.group(6)), q_tube=float(m.group(5)))[0],
            np.float64)) for c, z, m, sd in sp]
        gen = band_sd(rows(tw, donors))
        print(f"iteration {it}: g {np.round(g, 3).tolist()}  twin sd " + " ".join(f"{v:.3f}" for v, _ in gen))
        if it == ITERS:
            break
        for b in range(len(g)):
            if np.isfinite(real[b][0]) and np.isfinite(gen[b][0]) and gen[b][0] > 0:
                g[b] = float(np.clip(g[b] * real[b][0] / gen[b][0], 0.2, 2.0))
    out = {"-10": {"edges": EDGES.tolist(), "g": [round(float(x), 4) for x in MI.STUT_SPREAD[-10][1]]}}
    json.dump(out, open(DATA / "stut_spread.json", "w"), indent=1)
    print("stut_spread.json", out)


if __name__ == "__main__":
    main()
