"""calibrate_stut_parent.py - how each stutter offset moves with its PARENT's height, as an increment over the generator.

The artefact table, the per-allele laws (allele_stut.json) and the low-template factor (STUT_LT_A) get each offset's
mean ratio right on NOC1 but not its shape in parent height.

FLOOR FIRST. A peak at a stutter position is floor only, stutter only, or both, and at the minor offsets the floor is
most of it: at n+1 / n-2 positions under 600-1200 RFU parents 40-85 % of the visible peaks are floor. Fitting the
stutter law to the raw visible heights there fits the floor, and does not converge (a first version tried: +0.74 on
the n-2 ratio moved the visible median by nothing). So real's stutter is separated from the floor BY LAW before it
is compared:
  - the floor next to an allele is measured on real itself at CONTROL positions - bins 1.2-1.8 repeats from the
    nearest of the donor's alleles that no allele can reach at a stutter (-3..+3 repeats) or 1-3 bp offset - in the
    same parent-height band; there the generator's floor matches real (presence within ~0.004, median 8 vs 8-9 RFU);
  - with pi_f its presence and F its visible-height distribution, at a stutter position with observed presence P
    and visible distribution G:  pi_s = 1 - (1 - P) / (1 - pi_f),  G_s = (G - w_f F) / (1 - w_f),
    w_f = pi_f (1 - pi_s) / P;
  - checked on the generator against its TRUE stutter (same seeds, floor off; stutter is drawn before the floor):
    pi_s and the quantiles agree where the floor share w_f <= ~0.55 and not above ~0.7, so only bands with
    w_f <= W_MAX on the real side are fitted, and the rest take the nearest fitted band's value.
The generator side goes through the SAME separation (floor on, its own controls), so what the separation gets wrong
cancels: reading the generator with the floor off instead left the separation's own bias (+.005..+.013 presence where
the floor is 30-46 % of the peaks) uncorrected on the real side, and the fit charged it to the law.

Per offset (n-1, n+1, n-2) and parent-height band, two increments are iterated jointly: the emission LOGIT, to
bring the stutter's presence to real's pi_s, and the log RATIO, to bring its median height when present to real's.
What they correct, measured on TRAIN+VAL NOC1 before fitting: n-1 presence short at parents >= 300 RFU (.64/.59,
.81/.73, .86/.79, .87/.80) with heights right, and +17 % height under 150 RFU; n+1 and n-2 heights grow much less
than proportionally with the parent (n+1 median 17/22/21/37 RFU against 11/13/21/47 at 300-600/600-1.2k/1.2-3k/>3k).
AMEL and Yindel are left out entirely (not STRs: no stutter; their bins once dragged the bright-band target to .88
against the STR positions' .94-.96). Measured only where the target bin can be reached by exactly one of the donor's alleles (its parent) at any stutter or
1-3 bp offset, on
every real NOC1 profile of TRAIN and VAL regenerated from its own spec (TEST is not read).

Applied as a STEP per band (the bands it is measured on), not interpolated between band centres - interpolated, the
n-1 value at 50 RFU leaked into 100-141 RFU and put presence there over real.
Output: data/stut_parent.json {d: {"edges": [...], "ratio": [...], "logit": [...]}}, read by make_insilico (STUT_PH).
Measured over the generator WITH allele_stut.json in place, so run after calibrate_allele_stut.py.
Usage:  python calibrate_stut_parent.py
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
os.environ["STR_STUT_PH"] = "0"                             # the increment is over the generator WITHOUT itself
os.environ.setdefault("STR_DATA_DIR", str(DATA))
_s = importlib.util.spec_from_file_location("make_insilico", HERE / "make_insilico.py")
MI = importlib.util.module_from_spec(_s); _s.loader.exec_module(MI)

RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)" + kit.TAG + r"-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
OFFS = (-10, 10, -20)                                       # n-1, n+1, n-2
# Offsets, in REPEAT units, at which an allele can put something that is not floor: stutter at -3..+3 repeats and
# the 1-3 bp artefacts (+-0.25/0.5/0.75 repeat). Kept in repeat units, not in the allele-name keys of _OFF_TARGET: a
# first version listed -2 bp as key -5, which exists for no allele (-2 bp is -8 from 12 -> 11.2 but -2 from 12.2 ->
# 12.0), so microvariant neighbours were never excluded.
BAD = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0, -0.25, 0.25, -0.5, 0.5, -0.75, 0.75])
CTL_D = (1.2, 1.8)                                          # control distance: as close to one repeat as a clean bin gets
EDGES = np.array([0.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0, 1e9])
KNOT = np.array([50.0, 141.0, 283.0, 566.0, 1131.0, 2263.0, 4525.0, 9051.0])
W_MAX = 0.45                                                # floor share above which the separation is not trusted
MIN_N = 100
ITERS = int(os.environ.get("STR_STUT_PH_ITERS", "8"))
UN = MI.BIN_UNITS                                     # repeat units (make_insilico.BIN_UNITS)
BL = np.asarray(MI.BIN_LOCUS, int)


def specs():
    out = []
    for split, seed0 in (("train", 20000), ("val", 700)):
        names = json.load(open(DATA / f"meta_sample_names_{split}.json"))
        X = np.load(DATA / f"Xflat_{split}.npy"); noc = np.load(DATA / f"noc_{split}.npy")
        for n, i in enumerate(np.flatnonzero(noc == 1)):
            m = RX.search(str(names[i]))
            if m and int(m.group(1)) in MI.COL:
                out.append((MI.COL[int(m.group(1))], np.expm1(X[i].astype(np.float64)), m, seed0 + n))
    return out


def twins(sp, pool, bs):
    return [np.expm1(np.asarray(MI.gen_mixture(
        [c], pool, np.random.default_rng(sd), t_total=float(z.sum()), bin_size=bs, phi=np.array([1.0]),
        cond=m.group(3) or "a", ng_total=float(m.group(4)), inj_sec=float(m.group(6)), q_tube=float(m.group(5)))[0],
        np.float64)) for c, z, m, sd in sp]


def band(v):
    return int(np.clip(np.digitize(v, EDGES) - 1, 0, len(KNOT) - 1))


def claims(oa, n):
    """per bin: how many of the alleles oa sit at a non-floor offset (BAD) from it, same locus"""
    cnt = np.zeros(n, int); near = np.full(n, np.inf); nidx = np.full(n, -1)
    for L in np.unique(BL[oa]):
        al = oa[BL[oa] == L]; b = np.flatnonzero(BL == L)
        D = UN[b][:, None] - UN[al][None, :]
        cnt[b] = np.isclose(D[:, :, None], BAD[None, None, :]).any(2).sum(1)
        k = np.abs(D).argmin(1); near[b] = np.abs(D)[np.arange(len(b)), k]; nidx[b] = al[k]
    return cnt, near, nidx


def collect(profiles, donors, controls=True):
    """values at clean stutter positions keyed (offset, band) - a target only ONE allele can reach (its parent) - and
    control-position values keyed band - bins no allele can reach, 1.2-1.8 repeats from the nearest allele"""
    own = np.asarray(MI.DONOR_DOSAGE) > 0
    ctl, pos = {}, {}
    for c, h in zip(donors, profiles):
        oa = np.flatnonzero(own[c] & ~MI.NO_STUTTER_BIN)   # AMEL and Yindel have no repeat to stutter on
        cnt, near, nidx = claims(oa, len(h))
        if controls:
            for bb in np.flatnonzero(~own[c] & (cnt == 0) & (near >= CTL_D[0]) & (near <= CTL_D[1])):
                if h[nidx[bb]] > 0:
                    ctl.setdefault(band(h[nidx[bb]]), []).append(h[bb])
        st = oa[h[oa] > 0]
        for d in OFFS:
            t = MI._OFF_TARGET[d][st]; ok = t >= 0
            for p, tt in zip(st[ok], t[ok]):
                if not own[c, tt] and cnt[tt] == 1:
                    pos.setdefault((d, band(h[p])), []).append(h[tt])
    return ctl, pos


def separated(ctl, pos, key):
    """(stutter presence, stutter median height when present, floor share) with the floor removed, or None"""
    a = np.array(pos.get(key, [])); f = np.array(ctl.get(key[1], []))
    if len(a) < MIN_N or len(f) < MIN_N or not (a > 0).any():
        return None
    P = float((a > 0).mean()); pf = float((f > 0).mean())
    ps = 1.0 - (1.0 - P) / (1.0 - pf)
    if ps <= 0:
        return None
    wf = pf * (1.0 - ps) / P
    x = np.unique(np.concatenate([a[a > 0], f[f > 0]]))
    Go = np.searchsorted(np.sort(a[a > 0]), x, side="right") / (a > 0).sum()
    Fv = np.searchsorted(np.sort(f[f > 0]), x, side="right") / max((f > 0).sum(), 1)
    Gs = np.maximum.accumulate(np.clip((Go - wf * Fv) / max(1.0 - wf, 1e-6), 0, 1))
    return ps, float(x[min(np.searchsorted(Gs, 0.5), len(x) - 1)]), wf


def direct(pos, key):
    a = np.array(pos.get(key, []))
    if len(a) < MIN_N or not (a > 0).any():
        return None
    return float((a > 0).mean()), float(np.median(a[a > 0]))


def logit(p):
    p = np.clip(p, 0.005, 0.995)
    return np.log(p / (1.0 - p))


def fill(v, ok):
    """bands not fitted take the nearest fitted band's value"""
    if not ok.any():
        return np.zeros_like(v)
    idx = np.flatnonzero(ok)
    return np.array([v[idx[np.argmin(np.abs(idx - k))]] for k in range(len(v))])


def main():
    pool = MI.build_ss_pool(); bs = MI.build_bin_size()
    sp = specs(); donors = [s[0] for s in sp]
    ctl, pos = collect([s[1] for s in sp], donors)
    R = {d: [separated(ctl, pos, (d, k)) for k in range(len(KNOT))] for d in OFFS}
    ok = {d: np.array([r is not None and r[2] <= W_MAX for r in R[d]]) for d in OFFS}
    print(f"{len(sp)} real NOC1 profiles (train+val); fitted bands (floor share <= {W_MAX}):")
    for d in OFFS:
        print(f"  {d:>4}: " + " ".join(f"{k:>6.0f}{'*' if o else ' '}" for k, o in zip(KNOT, ok[d]))
              + "   floor share " + " ".join("  -  " if r is None else f"{r[2]:.2f}" for r in R[d]))
    inc = {d: {"ratio": np.zeros(len(KNOT)), "logit": np.zeros(len(KNOT))} for d in OFFS}
    for it in range(ITERS + 1):
        for d in OFFS:
            MI.STUT_PH[d] = (EDGES, inc[d]["ratio"].copy(), inc[d]["logit"].copy())
        gctl, gpos = collect(twins(sp, pool, bs), donors)   # floor ON, separated exactly like real
        print(f"\niteration {it}: stutter presence and median RFU, floor removed the same way, real / generator")
        for d in OFFS:
            G = [separated(gctl, gpos, (d, k)) for k in range(len(KNOT))]
            dl = np.zeros(len(KNOT)); dr = np.zeros(len(KNOT)); good = ok[d].copy()
            cells = []
            for k in range(len(KNOT)):
                r, g = R[d][k], G[k]
                if good[k] and g is not None:
                    dl[k] = logit(r[0]) - logit(g[0]); dr[k] = np.log(r[1] / g[1])
                    cells.append(f"{r[0]:.3f}/{g[0]:.3f} {r[1]:5.1f}/{g[1]:5.1f}")
                else:
                    good[k] = False
                    cells.append(f"{'-':>23}")
            print(f"  {d:>4} " + " | ".join(cells))
            if it < ITERS:
                inc[d]["logit"] = fill(inc[d]["logit"] + np.where(good, dl, 0.0), good)
                inc[d]["ratio"] = fill(inc[d]["ratio"] + np.where(good, dr, 0.0), good)
    # the increments behind the LAST measurement (iteration ITERS) are the ones written
    out = {str(d): {"edges": EDGES.tolist(), "ratio": [round(float(v), 4) for v in MI.STUT_PH[d][1]],
                    "logit": [round(float(v), 4) for v in MI.STUT_PH[d][2]]} for d in OFFS}
    json.dump(out, open(os.environ.get("STR_STUT_PH_OUT", str(DATA / "stut_parent.json")), "w"), indent=1)
    for d in OFFS:
        print(f"stut_parent.json [{d:>4}] ratio {out[str(d)]['ratio']}  logit {out[str(d)]['logit']}")


if __name__ == "__main__":
    main()
