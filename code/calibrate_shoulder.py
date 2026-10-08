"""calibrate_shoulder.py - how a standing allele 1, 2 or 3 bp away thins a peak, by DISTANCE, as an increment.

The generator's SHOULDER terms (calibrate.py) looked for any allele "within one repeat" by allele-name tenths and kept a
flat fraction for artefacts (.72) and one curve in the neighbour's height for the noise floor, on the premise that bp
distance does not enter. Measured in real bp (repeat units x 4: 12 vs 11.3 is 1 bp, not 7 tenths) it decides almost
everything: on TRAIN+VAL NOC1 an n-1 that lands 1 bp from a standing allele was seen 0 times in 178 - two peaks 1 bp
apart are not resolved - while 2-3 bp away it stood about as often as a clean one (.78-.95). The flat .72 averaged the
two, so the generator showed .59 at 1 bp and cut 2-3 bp to .57-.68.

Two quantities per distance d = 1, 2, 3 bp, each measured on real and on the twin by the same code:
  art[d]       n-1 presence at a target with ONE standing allele d bp away / presence at a clean target
               (parents >= 1200 RFU, targets no other stutter offset reaches)
  noise[d][h]  floor presence at a bin d bp from ONE standing allele of height band h (not a stutter position)
               / floor presence far from every allele (>= 2.5 repeats)
and the generator's factors (SHOULDER_ART_BP, SHOULDER_NOISE_BP) are moved by real/twin until the twin matches -
the increment is charged against the generator as it is, crowding, pull-up and survival included. AMEL and Yindel are
left out (no stutter). TEST is not read.

Output: data/shoulder_bp.json {"art": [a1, a2, a3], "noise_h": [...], "noise": {"1": [...], "2": [...], "3": [...]}},
read by make_insilico (_shoulder_hd). STR_SHOULDER_BP=0 turns it off.
Usage:  python calibrate_shoulder.py
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
os.environ["STR_SHOULDER_BP"] = "0"                     # measured over the generator's flat shoulder, then set in memory
os.environ.setdefault("STR_DATA_DIR", str(DATA))
_s = importlib.util.spec_from_file_location("make_insilico", HERE / "make_insilico.py")
MI = importlib.util.module_from_spec(_s); _s.loader.exec_module(MI)

RX = re.compile(r"RD\d+-\d+-(\d+)d(\d+)([A-Za-z0-9\-]*?)-([\d.]+)" + kit.TAG + r"-Q([\d.]+)_\d+\.(\d+)\s*sec", re.I)
INT = np.array([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0])
HB = np.array([0.0, 300.0, 1000.0, 3000.0, 1e12])
ITERS = int(os.environ.get("STR_SHOULDER_ITERS", "3"))
MIN_N = 30
OWN = np.asarray(MI.DONOR_DOSAGE) > 0
BL = np.asarray(MI.BIN_LOCUS, int)
UN = MI.BIN_UNITS                                     # repeat units (make_insilico.BIN_UNITS)


def specs():
    out = []
    for split, seed0 in (("train", 30000), ("val", 900)):
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


def scan(profiles, donors):
    """(noise counts keyed (d, h band), far counts, n-1 counts keyed d, clean n-1 counts); each count is [n, visible]"""
    NZ = collections.defaultdict(lambda: [0, 0]); FAR = [0, 0]
    ST = collections.defaultdict(lambda: [0, 0]); STC = [0, 0]
    for c, z in zip(donors, profiles):
        oa = np.flatnonzero(OWN[c] & ~MI.NO_STUTTER_BIN)
        for L in np.unique(BL[oa]):
            al = oa[BL[oa] == L]; b = np.flatnonzero((BL == L) & ~OWN[c])
            D = UN[b][:, None] - UN[al][None, :]
            integ = np.isclose(D[:, :, None], INT[None, None, :]).any(2)
            bp = np.rint(np.abs(D) * 4).astype(int)
            for r, bb in enumerate(b):
                near = sorted((int(bp[r, k]), int(al[k])) for k in range(len(al)) if 0 < abs(D[r, k]) < 1)
                if integ[r].any():
                    k1 = [k for k in range(len(al)) if abs(D[r, k] + 1) < 1e-6]
                    if len(k1) == 1 and integ[r].sum() == 1 and z[al[k1[0]]] >= 1200:
                        if near:
                            d, q = near[0]
                            if z[q] > 0 and len(near) == 1:
                                ST[d][0] += 1; ST[d][1] += int(z[bb] > 0)
                        else:
                            STC[0] += 1; STC[1] += int(z[bb] > 0)
                    continue
                if near:
                    d, q = near[0]
                    if z[q] > 0 and len(near) == 1:
                        hb = int(np.searchsorted(HB, z[q], side="right") - 1)
                        NZ[(d, hb)][0] += 1; NZ[(d, hb)][1] += int(z[bb] > 0)
                elif np.abs(D[r]).min() >= 2.5:
                    FAR[0] += 1; FAR[1] += int(z[bb] > 0)
    return NZ, FAR, ST, STC


def rel(res):
    """art[d] and noise[d][h] as ratios (None where too few)"""
    NZ, FAR, ST, STC = res
    f0 = FAR[1] / max(FAR[0], 1); base = STC[1] / max(STC[0], 1)
    art = {d: (ST[d][1] / ST[d][0] / base if ST[d][0] >= MIN_N else None) for d in (1, 2, 3)}
    noise = {(d, h): (NZ[(d, h)][1] / NZ[(d, h)][0] / f0 if NZ[(d, h)][0] >= MIN_N else None)
             for d in (1, 2, 3) for h in range(4)}
    return art, noise


def main():
    pool = MI.build_ss_pool(); bs = MI.build_bin_size()
    sp = specs(); donors = [s[0] for s in sp]
    R = scan([s[1] for s in sp], donors); ra, rn = rel(R)
    # the noise curve's abscissa: one point per neighbour-height band (geometric centre, ends held at 100 / 10000)
    hb_med = []
    for h in range(4):
        lo, hi = HB[h], HB[h + 1]
        hb_med.append(float(np.sqrt(max(lo, 100.0) * min(hi, 10000.0))))
    art = np.array([MI.SHOULDER_ART] * 3, float)
    noise = {d: np.interp(np.log(hb_med), np.log(MI.SHOULDER_H), MI.SHOULDER_NOISE).astype(float) for d in (1, 2, 3)}
    print(f"{len(sp)} real NOC1 profiles (train+val)")
    print("real: art/clean by 1/2/3 bp " + " ".join("-" if ra[d] is None else f"{ra[d]:.3f}" for d in (1, 2, 3)))
    print("real: noise/far by d, band " + " | ".join(
        f"{d}bp " + " ".join("-" if rn[(d, h)] is None else f"{rn[(d, h)]:.2f}" for h in range(4)) for d in (1, 2, 3)))
    for it in range(ITERS + 1):
        MI.SHOULDER_ART_BP = np.array([1.0] + list(art))
        MI.SHOULDER_NOISE_H = np.asarray(hb_med)
        MI.SHOULDER_NOISE_BP = {d: noise[d].copy() for d in (1, 2, 3)}
        ta, tn = rel(scan(twins(sp, pool, bs), donors))
        print(f"\niteration {it}: factors art {np.round(art, 3).tolist()}")
        print("  art/clean real|twin: " + "  ".join(
            f"{d}bp " + ("-" if ra[d] is None or ta[d] is None else f"{ra[d]:.3f}|{ta[d]:.3f}") for d in (1, 2, 3)))
        for d in (1, 2, 3):
            print(f"  noise {d}bp real|twin by band: " + "  ".join(
                "-" if rn[(d, h)] is None or tn[(d, h)] is None else f"{rn[(d, h)]:.2f}|{tn[(d, h)]:.2f}"
                for h in range(4)))
        if it == ITERS:
            break
        for d in (1, 2, 3):
            if ra[d] is not None and ta[d] is not None and ta[d] > 0:
                art[d - 1] = float(np.clip(art[d - 1] * ra[d] / ta[d], 0.0, 1.5))
            for h in range(4):
                a, b = rn[(d, h)], tn[(d, h)]
                if a is not None and b is not None and b > 0:
                    noise[d][h] = float(np.clip(noise[d][h] * a / b, 0.0, 3.0))
    out = {"art": [round(float(x), 4) for x in art], "noise_h": [round(x, 1) for x in hb_med],
           "noise": {str(d): [round(float(x), 4) for x in noise[d]] for d in (1, 2, 3)}}
    json.dump(out, open(DATA / "shoulder_bp.json", "w"), indent=1)
    print("\nshoulder_bp.json", out)


if __name__ == "__main__":
    main()
