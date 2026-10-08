"""calibrate.py — derive every generator constant from single-source (NOC1) reference profiles.

Split out of make_insilico so the generator file holds the generator. Nothing here is fitted to the
mixture set: PROVEDIt keeps all 1333 real mixtures in the evaluation split, so a constant taken from
them could not be validated against them. The laws each constant encodes, and the hypotheses that were
measured and rejected, are in reports/generator_laws.md.

Called once through make_insilico.noc1_calib(), which owns the cache and the fallback literals.
"""
from __future__ import annotations
import json as _json
import os
import re as _re
import numpy as np
import re as _re_mod
_RE_EXTRACT = _re_mod.compile(r"RD\d+-\d+-(\d+)(d[^-]*)-")
from pathlib import Path
import kit


def _art_survival(ctx, ss, tok, mk, y, key, sr):
    """What fraction of an emitted artefact is actually seen, against its OWN expected height.

    Two things have to come out before this is measurable: the baseline noise that sits on a bin
    whether or not an artefact was emitted there, and the emission rate itself. The noise level is
    read at offsets that are NOT products - three and four repeats away the occupancy is 0.095 and
    does not move with the parent, where n-1 runs 0.124 to 0.845 across the same range - and the
    emission rate is what the curve saturates at once the expected height is large. What is left
    runs 0.04 at 0-4 RFU to 1.00 above 200, half of them lost at 26 RFU. The allele dropout law,
    measured independently on the same profiles, puts its own half-loss at 27.5: an artefact is lost
    by the same height law as a real allele, which is why this is a survival curve and not a second
    artefact model."""
    EH = [0.0, 4.0, 7.0, 11.0, 17.0, 26.0, 40.0, 65.0, 110.0, 200.0]
    num = np.zeros(len(EH)); den = np.zeros(len(EH)); hsum = np.zeros(len(EH)); bn = bd = 0
    for i in ss:
        c = int(y[i].argmax()); v = np.where(mk[i])[0]
        kk = [(int(round(float(tok[i, t, 0]))), int(round(float(tok[i, t, 1]) * 10))) for t in v]
        hm = dict(zip(kk, np.expm1(tok[i, v, 2].astype(np.float64))))
        for a in key[c]:
            ha = hm.get(a, 0.0)
            if ha <= 0:
                continue
            for d in (30, -30, 40, -40):                    # far offsets: baseline noise only
                if (a[0], a[1] + d) in key[c]:
                    continue
                bd += 1; bn += 1 if hm.get((a[0], a[1] + d), 0.0) > 0 else 0
            # n-1 ALONE. Pooling the four product offsets was tried and biases the curve down by a
            # tenth above 15 RFU: n+1 and n-2 carry a ratio near 0.007, so their whole population
            # sits in the first height bin and whatever their emission rate falls short of n-1's is
            # charged to survival instead. One offset with a well-determined rate measures the height
            # law cleanly, and the law is a property of the peak, not of which offset produced it.
            f = sr.get((a[0], -10))
            if f is None or (a[0], a[1] - 10) in key[c]:
                continue
            e_ = np.exp(f[0] + f[2] * (a[1] / 10.0 - f[3])) * ha
            q = min(int(np.searchsorted(EH, e_, "right")) - 1, len(EH) - 1)
            den[q] += 1; hsum[q] += e_
            num[q] += 1 if hm.get((a[0], a[1] - 10), 0.0) > 0 else 0
    bg = bn / max(bd, 1)
    ok = den >= 400
    cl = np.clip((num[ok] / den[ok] - bg) / max(1.0 - bg, 1e-9), 0.0, None)
    cl = np.maximum.accumulate(cl)                          # survival cannot fall with height
    # Anchored at each bin's MEAN expected height, not its left edge: the value is the average over
    # the bin, and hanging it on the edge makes every interpolation between bins read low.
    return ([float(x) for x in (hsum[ok] / den[ok])],
            [float(min(x / max(cl.max(), 1e-9), 1.0)) for x in cl], float(bg))


def _noise_floor(ctx, ss, tok, mk, y, key):
    """The baseline peak density, per panel bin, measured where nothing structured can reach.

    Density against distance from the nearest donor allele, on NOC1: 0.260 at one repeat and 0.101 at
    two - stutter and n-2 - then 0.075, 0.080, 0.077, 0.077, 0.077 from 2.5 repeats out to the end of
    the panel. It is FLAT, not a halo: the baseline is a property of the run, not of how close a real
    allele happens to be. Fractional positions sit at 0.040, half the integer rate, because an
    off-ladder bin is a less likely place for a peak, not a more likely one.

    This replaces a Poisson count spread over the histogram of all artefact positions. That histogram
    is dominated by stutter, so it concentrated the noise floor onto the very bins the stutter model
    already fills and starved everywhere else: matching the density beside an allele then needed 108
    peaks per profile where real carries 93. The three numbers here reproduce real's artefact count
    without a count being fitted - 0.077 x 547 bins of baseline, plus the structured excess at one and
    two repeats, less the shortfall at fractional bins, is 56.7 against real's 56.5."""
    bl = ctx.BIN_LOCUS.astype(int); ba = np.rint(ctx.BIN_ALLELE * 10).astype(int)
    per_loc = {}
    for j in range(ctx.N_FLAT):
        per_loc.setdefault(int(bl[j]), []).append(int(ba[j]))
    ni = di = nf = df = nn = dn = 0
    bin_n = {}; bin_d = {}
    hts = []
    per_h = {}; per_n = {}; per_d = {}; runs = []
    for i in ss:
        c = int(y[i].argmax()); v = np.where(mk[i])[0]
        kk = [(int(round(float(tok[i, t, 0]))), int(round(float(tok[i, t, 1]) * 10))) for t in v]
        hm = dict(zip(kk, np.expm1(tok[i, v, 2].astype(np.float64))))
        al = {}; cur = []
        for (L_, a_) in key[c]:
            al.setdefault(L_, []).append(a_)
        for L_, aa in al.items():
            aa = np.asarray(aa)
            for b in per_loc.get(L_, ()):
                if (L_, b) in key[c]:
                    continue
                _d = int(np.abs(aa - b).min())
                h_ = hm.get((L_, b), 0.0)
                # A fractional bin BESIDE a real allele is emptier than one out in the open: 0.040
                # against 0.075. Nothing structured reaches a fractional position, so this is the
                # baseline itself being suppressed - a shoulder next to a tall peak is what the
                # analysis software removes, the same depletion already measured for low peaks
                # sitting under a tall one in another dye.
                if b % 10 and _d < 10:
                    dn += 1; nn += 1 if h_ > 0 else 0
                elif _d < 25:
                    continue
                elif b % 10:
                    df += 1; nf += 1 if h_ > 0 else 0
                else:
                    di += 1; ni += 1 if h_ > 0 else 0
                if _d >= 25:
                    per_d[L_] = per_d.get(L_, 0) + 1
                    per_n[L_] = per_n.get(L_, 0) + (1 if h_ > 0 else 0)
                    bin_d[(L_, b)] = bin_d.get((L_, b), 0) + 1
                    bin_n[(L_, b)] = bin_n.get((L_, b), 0) + (1 if h_ > 0 else 0)
                if h_ > 0:
                    hts.append(np.log(h_))
                    if _d >= 25:
                        per_h.setdefault(L_, []).append(np.log(h_))
                        cur.append((L_, np.log(h_)))
        if len(cur) >= 6:
            runs.append(cur)
    hts = np.asarray(hts)
    # Noise height depends on the LOCUS and on nothing else that was tested. Its geometric mean runs
    # 5.5 to 13.1 RFU across loci - 2.4x - while it moves 8.2 to 9.4 across a sevenfold range of
    # profile total, does not move with template over a seventeenfold range, with injection time, with
    # fragment size (8.6/9.1/7.5/9.2), with distance from the nearest allele, or with whether the bin
    # is an integer position. The locus effect is not amplification efficiency either: those two
    # correlate 0.370 across loci, and not fragment size, which correlates 0.030. It is a property of
    # the locus's own channel. Removing it takes the log scatter from 0.480 to 0.385, so it is 36% of
    # the variance a single global constant was calling noise.
    loc_mu = {int(k): float(np.mean(v)) for k, v in per_h.items() if len(v) >= 300}
    _res = [np.asarray(v) - np.mean(v) for v in per_h.values() if len(v) >= 300]
    loc_sd = float(np.concatenate(_res).std()) if _res else (float(hts.std()) if len(hts) else 0.53)
    loc_p = {int(k): float(per_n[k] / per_d[k]) for k in per_d if per_d[k] >= 2000}
    # PER-BIN rate, not one rate per locus. Measured on NOC1 the occupancy across bins scatters 0.0153
    # where a common rate would give 0.0055 - 2.8x over-dispersed - and runs 0.063 at the tenth
    # percentile to 0.157 at the loudest bin. Height carries no such structure: once the locus level is
    # out, bins inside a locus differ by 1.15x and the loudest averages 18.8 RFU, so the height stays a
    # locus property and only the rate becomes a per-bin one. Shrunk toward the locus rate by the
    # count behind each bin, so a bin seen 400 times keeps its own value and a rare one does not
    # invent a rate from noise.
    bin_p = {}
    for k, d_ in bin_d.items():
        if d_ < 150:
            continue
        r_ = bin_n[k] / d_
        lr = float(per_n.get(k[0], 0) / max(per_d.get(k[0], 1), 1))
        w_ = d_ / (d_ + 300.0)
        bin_p[f"{k[0]}_{k[1]}"] = float(w_ * r_ + (1.0 - w_) * lr)
    # A fifth of the height scatter belongs to the RUN, not to the peak: sd 0.173 between profiles
    # against 0.343 within. The baseline is one instrument state per injection, so a profile with a
    # loud floor is loud everywhere at once, and drawing every peak independently makes each generated
    # profile equally average. The run's LEVEL and its COUNT are independent (-0.032), so only the
    # level takes this term - the same split already carried for stutter.
    _g = [np.array([v - loc_mu.get(L_, 0.0) for L_, v in r]) for r in runs
          if sum(1 for L_, _ in r if L_ in loc_mu) >= 6]
    _g = [x for x in _g if len(x) >= 6]
    if _g:
        _mu = np.array([x.mean() for x in _g]); _wi = np.concatenate([x - x.mean() for x in _g])
        run_f = float(np.sqrt(_mu.var() / max(_mu.var() + _wi.var(), 1e-12)))
    else:
        run_f = 0.0
    # The height's SHAPE, per locus. The spread is right as two numbers - locus spread plus a common
    # residual composes to 0.463 against real's pooled 0.480 - but real puts 0.419 of its noise at 8-12
    # RFU where that composition puts 0.311. Copying real heights in, one level of the answer at a time:
    # this sample's own heights fix the band shape, real heights pooled PER LOCUS fix it as well, and
    # per-locus heights with a run offset split off and redrawn break it again. So each locus keeps its
    # own marginal, as quantiles, and nothing is decomposed.
    loc_q = {int(k): np.quantile(np.asarray(v), np.linspace(0.0, 1.0, 101)).tolist()
             for k, v in per_h.items() if len(v) >= 300}
    # The run still matters - one instrument state per injection - but only as CORRELATION between the
    # peaks of one profile, not as an offset added on top of a marginal that already contains it. The
    # between-run share is taken with each run's own sampling noise removed: a run mean from a dozen
    # peaks carries within-variance / n of scatter that is not a run effect.
    _gc = [np.array([v - loc_mu[L_] for L_, v in r if L_ in loc_mu]) for r in runs]
    _gc = [x for x in _gc if len(x) >= 6]
    rho = 0.0
    if _gc:
        _mm = np.array([x.mean() for x in _gc]); _vv = np.array([x.var(ddof=1) for x in _gc])
        _nn = np.array([len(x) for x in _gc], float)
        _vb = max(float(_mm.var()) - float(np.mean(_vv / _nn)), 0.0)
        rho = float(np.sqrt(_vb / max(_vb + float(np.mean(_vv)), 1e-12)))
    return (float(ni / max(di, 1)), float(nf / max(df, 1)), float(nn / max(dn, 1)),
            float(hts.mean()) if len(hts) else 2.18, loc_sd, loc_mu, loc_p, run_f, bin_p, loc_q, rho)


def _art_rarity(ctx, ss, tok, mk, y, key, tabl, EH, SP, edges):
    """How much of the emission survives, against how COMMON the target allele is in the panel.

    A stutter landing where few donors carry an allele is far less likely to be called: at a fixed
    locus, parent height and parent repeat count, NOC1 gives 0.29-0.46 where one or two of the 45
    donors carry the n-1 position against 0.72-0.92 where eleven or more do, and the pattern holds on
    nearly every locus. The panel's common alleles are the ladder's well-defined bins; an unusual
    repeat length sits at the edge of one, and what the software will call there is not the same.

    This is what makes an otherwise NOC-invariant law look NOC-dependent. As contributors are added
    the n-1 positions still FREE are the ones nobody carries, so the mean rarity of an eligible
    position falls from 6.8 donors at NOC2 to 3.7 at NOC5, and real's per-position stutter rate falls
    with it - 0.516 to 0.340 - while a generator without this term holds flat at 0.49. Conditioning on
    parent height, total RFU, treatment, carrier count, locus maximum, neighbouring peak height and
    the peak's deviation from its own contributor's level explains none of that gap; rarity explains
    all of it, and is measurable on NOC1 where no mixture is involved."""
    carr = {}
    for k_ in key:
        for b in k_:
            carr[b] = carr.get(b, 0) + 1
    # Stratified by the model's OWN predicted rate. Rare positions sit under short parents, which the
    # SR-against-repeat term already makes quiet, so pooling divides a low count by a low prediction
    # and reports no effect at all - the residual came out 0.86-1.00 where the raw rates run 0.29
    # against 0.80. The multiplier has to be what is left INSIDE a stratum of equal prediction.
    PS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.75, 1.01]
    num = np.zeros((len(PS), len(edges))); den = np.zeros((len(PS), len(edges)))
    for i in ss:
        c = int(y[i].argmax()); v = np.where(mk[i])[0]
        kk = [(int(round(float(tok[i, t, 0]))), int(round(float(tok[i, t, 1]) * 10))) for t in v]
        hm = dict(zip(kk, np.expm1(tok[i, v, 2].astype(np.float64))))
        for a in key[c]:
            ha = hm.get(a, 0.0)
            if ha <= 0 or (a[0], a[1] - 10) in key[c]:
                continue
            row = None
            for r_ in (tabl.get(int(a[0])) or ()):
                if int(r_[0]) == -10:
                    row = r_; break
            if row is None:
                continue
            pr = row[1] * float(np.interp(np.exp(row[2] + row[4] * (a[1] / 10.0 - row[5])) * ha, EH, SP))
            q = min(int(np.searchsorted(edges, carr.get((a[0], a[1] - 10), 0), "right")) - 1, len(edges) - 1)
            s = min(int(np.searchsorted(PS, pr, "right")) - 1, len(PS) - 1)
            den[s, q] += pr
            num[s, q] += 1.0 if hm.get((a[0], a[1] - 10), 0.0) > 0 else 0.0
    m = np.ones(len(edges)); w = np.zeros(len(edges))
    for q in range(len(edges)):
        r = [(num[s, q] / den[s, q], den[s, q]) for s in range(len(PS)) if den[s, q] >= 60]
        if r:
            ww = sum(x[1] for x in r)
            m[q] = sum(x[0] * x[1] for x in r) / ww; w[q] = ww
    ok = w > 0
    # Relative to the WEIGHTED MEAN, not the maximum. The emission rate these multiply was solved over every
    # parent, so it already is the average over the carrier mix; dividing by the top bucket instead pushed
    # every other bucket down and left n-1 at tall parents at 0.94-0.95 of real in every bucket.
    m[ok] = m[ok] / max(float(np.sum(m[ok] * w[ok]) / np.sum(w[ok])), 1e-9)
    return [float(x) for x in m]


# PULL-UP. PROVEDIt's "Filtered" export - the one every array in this project is built from - has pull-up,
# minus-A and SE33 -2 bp removed by the criteria of Alfonse et al. (FSI Genet 2018;32:62-70). Read off the
# Filtered/UnFiltered pairs, the removal that touches real peaks is the pull-up rule, and it is a rule on
# observable context: a peak is deleted with certainty when a peak in ANOTHER dye channel within 0.5 bp stands
# 20x or more taller, in part at 10-20x or 0.5-2 bp, and almost never below 10x. True alleles of NOC1 and of the
# mixtures follow the same table as the artefacts it was read from. Pull-up is spectral bleed-through, so the
# context is sought only across channels; each marker's channel comes from the kit's panel (kit.DYE).
PULL_R_EDGES = [0.0, 5.0, 10.0, 15.0, 20.0, 30.0, 50.0, 100.0, 300.0, 1e12]
PULL_D_EDGES = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
PULL_H_GRID = np.exp(np.linspace(0.0, np.log(30000.0), 25))


def pull_neighbours(bin_locus, locus_names, bsz, win=2.0):
    """Per bin, the bins of OTHER dye channels within win bp of it (bin median sizes), padded with an index
    one past the last bin (height 0) and distance 99."""
    dye = np.array([kit.DYE.get(locus_names[int(L)], -1) for L in bin_locus])
    n = len(bsz); ks = []
    for j in range(n):
        if bsz[j] <= 0 or dye[j] < 0:
            ks.append(np.zeros(0, np.int64)); continue
        ks.append(np.where((dye >= 0) & (dye != dye[j]) & (bsz > 0) & (np.abs(bsz - bsz[j]) <= win))[0])
    M = max(1, max(len(k) for k in ks))
    nb = np.full((n, M), n, np.int64); nd = np.full((n, M), 99.0)
    for j, k in enumerate(ks):
        nb[j, :len(k)] = k; nd[j, :len(k)] = np.abs(bsz[k] - bsz[j])
    return nb, nd


def _pull_cell(h, nb, nd, idx):
    """Table cell of the peaks at bins idx: the TALLEST other-channel peak near each is the context."""
    hp = np.append(np.asarray(h, float), 0.0)
    hh = hp[nb[idx]]; a = hh.argmax(1); r_ = np.arange(len(idx))
    hm = hh[r_, a]; dm = nd[idx][r_, a]
    ri = np.clip(np.searchsorted(PULL_R_EDGES, hm / np.maximum(hp[idx], 1e-9), side="right") - 1, 0, len(PULL_R_EDGES) - 2)
    di = np.searchsorted(PULL_D_EDGES, dm, side="right") - 1
    ok = (hm > 0) & (di >= 0) & (di < len(PULL_D_EDGES) - 1)
    return ri, np.clip(di, 0, len(PULL_D_EDGES) - 2), ok


def pull_prob(h, nb, nd, idx, table):
    """P(the analysis deletes the peak as pull-up) for the peaks at bins idx, given the profile's heights h."""
    ri, di, ok = _pull_cell(h, nb, nd, idx)
    p = np.zeros(len(idx))
    p[ok] = np.asarray(table, float)[ri[ok], di[ok]]
    return p


def _pull_table(ctx, want, nb, nd, bidx, loci):
    """P(delete) by (R, |d|) cell from the Filtered/UnFiltered pairs of the named profiles, as the increment over
    each column's no-context rate (other rules and chance). Cached: the raw exports are read once."""
    import csv as _csv, glob as _glob, json as _js
    from pathlib import Path as _P
    cp = _P(ctx.DATA) / "pullup_pairs.json"
    if cp.exists():
        return _js.load(open(cp))
    raw = _P(ctx.DATA).parent / "data_raw"
    li = {name: i for i, name in enumerate(loci)}
    H = {"UnF": {}, "F": {}}
    for tag, sub in (("UnF", "PROVEDIt_1-5-Person CSVs UnFiltered"), ("F", "PROVEDIt_1-5-Person CSVs Filtered")):
        for f in [g for g in _glob.glob(str(raw / sub / f"*{kit.KIT}" / "**" / "*.csv"), recursive=True) if "Known Genotypes" not in g]:
            with open(f, newline="", encoding="utf-8-sig", errors="replace") as fh:
                rd = _csv.reader(fh); head = next(rd)
                si = head.index("Sample File"); mi = head.index("Marker")
                ai = [i for i, c in enumerate(head) if c.startswith("Allele ")]
                hi = [head.index(head[i].replace("Allele", "Height")) for i in ai]
                for row in rd:
                    if len(row) <= mi or row[si] not in want or row[mi] not in li:
                        continue
                    v = H[tag].setdefault(row[si], np.zeros(ctx.N_FLAT))
                    for a_, h_ in zip(ai, hi):
                        if h_ >= len(row):
                            continue
                        try:
                            av = float(row[a_]); hv = float(row[h_])
                        except ValueError:
                            continue
                        j = bidx.get((li[row[mi]], int(round(av * 10))))
                        if j is not None:
                            v[j] = max(v[j], hv)
    cnt = np.zeros((len(PULL_R_EDGES) - 1, len(PULL_D_EDGES) - 1, 2))
    for s_, hu in H["UnF"].items():
        hf = H["F"].get(s_)
        if hf is None:
            continue
        lv = np.where(hu > 0)[0]
        ri, di, ok = _pull_cell(hu, nb, nd, lv)
        dl = (hf[lv] <= 0).astype(float)
        np.add.at(cnt[:, :, 0], (ri[ok], di[ok]), dl[ok]); np.add.at(cnt[:, :, 1], (ri[ok], di[ok]), 1.0)
    rate = cnt[:, :, 0] / np.maximum(cnt[:, :, 1], 1.0)
    base = rate[0]
    p = np.clip((rate - base) / np.maximum(1.0 - base, 1e-9), 0.0, 1.0)
    p[cnt[:, :, 1] < 30] = 0.0
    p[0] = 0.0
    res = {"p": p.tolist(), "n": cnt[:, :, 1].tolist(), "profiles": len(H["UnF"])}
    _js.dump(res, open(cp, "w"))
    return res


def _pull_bar(X, own, nb, nd, table):
    """NOC1 average of P(deleted) for a candidate peak of height h at each bin, over the train profiles where the
    bin is not the donor's own allele: what emission rates calibrated on the Filtered NOC1 arrays already carry."""
    table = np.asarray(table, float)
    n = X.shape[1]; G = len(PULL_H_GRID); nR = len(PULL_R_EDGES) - 1; nD = len(PULL_D_EDGES) - 1
    s = np.zeros((n, G)); c = np.zeros(n)
    for a in range(0, len(X), 300):
        Hh = X[a:a + 300]; free = ~own[a:a + 300]
        Hp = np.concatenate([Hh, np.zeros((len(Hh), 1))], 1)
        hh = Hp[:, nb]
        am = hh.argmax(2)
        hm = np.take_along_axis(hh, am[..., None], 2)[..., 0]
        dm = nd[np.arange(n)[None, :], am]
        di = np.searchsorted(PULL_D_EDGES, dm, side="right") - 1
        okd = (hm > 0) & (di >= 0) & (di < nD)
        dic = np.clip(di, 0, nD - 1)
        c += free.sum(0)
        for g, hg in enumerate(PULL_H_GRID):
            ri = np.clip(np.searchsorted(PULL_R_EDGES, hm / hg, side="right") - 1, 0, nR - 1)
            s[:, g] += (np.where(okd, table[ri, dic], 0.0) * free).sum(0)
    return s / np.maximum(c[:, None], 1.0)


def derive(ctx) -> dict:
    """Everything the generator needs, estimated from the REAL SINGLE-SOURCE profiles.

    Single-source is the only place with per-peak ground truth: a peak is the donor's allele or it is
    an artefact, with no deconvolution in between, and the sample's template (ng), injection time and
    degradation index are all recorded. Calibrating here and letting mixtures follow as an untuned
    consequence keeps every constant observable and keeps the evaluation splits out of the loop —
    earlier calibrations read their targets off real test, which is selection pressure on the metric.

    Fitted on 5245 profiles:
      log RFU = 0.976*log(ng) + 0.948*log(inj) + 9.743   (R2 0.849, residual sd 0.603)
    Both exponents are 1 within noise, which is what the chemistry says: product scales with input DNA
    and signal scales with injection time. So t_total is not a free parameter — it follows from the
    template a mixture carries and how long it was injected, and it scales with contributor count
    because each contributor brings their own material. The shipped generator drew t_total from a
    k-independent lognormal, which made NOC1 twice as loud as real and NOC5 too quiet at once.

    Also returns the empirical artefact-bin weights: real artefacts land on a panel allele 50.9% of the
    time against the 35.4% a uniform draw over the 590 bins would give, because 37% of them are stutter
    and sit beside true alleles in the common allele range. Sampling positions from the observed
    histogram reproduces that without a tuning constant."""
    if ctx.cache is not None:
        return ctx.cache
    if not ctx.SZ:
        _b = ctx.build_bin_size()
        if _b is not None:
            ctx.SZ = {(int(ctx.BIN_LOCUS[j]), int(round(float(ctx.BIN_ALLELE[j]) * 10))): float(_b[j])
                   for j in range(ctx.N_FLAT) if _b[j] > 0}
    out = {}
    try:
        tok = np.load(ctx.DATA / "tokens_train.npy"); mk = np.load(ctx.DATA / "mask_train.npy").astype(bool)
        y = np.load(ctx.DATA / "y_train_set.npy"); noc = np.load(ctx.DATA / "noc_train.npy")
        ng = np.load(ctx.DATA / "meta_template_train.npy").astype(float)
        names = _json.load(open(ctx.DATA / "meta_sample_names_train.json"))
        gp = ctx.DATA / "donor_geno.npy"
        gg = np.load(gp); ggm = np.load(gp.parent / "donor_geno_mask.npy").astype(bool)
    except Exception:
        ctx.cache = {}
        return ctx.cache
    key = [{(int(round(float(gg[c, j, 0]))), int(round(float(gg[c, j, 1]) * 10)))
            for j in range(gg.shape[1]) if ggm[c, j]} for c in range(gg.shape[0])]
    import re as _re
    pat = _re.compile(r"[_.](\d+)\s*sec", _re.I)
    ss = np.where(noc == 1)[0]
    tpl, inj, art_w, sr = [], [], np.zeros(ctx.N_FLAT), []
    _INJ_REF, _INJ_BETA = 15.0, 0.80
    out["inj_ref"] = _INJ_REF
    rfu_tot, at_min, far_cnt, hb_pairs, drop_rec = [], [], [], [], []
    rfu_don, rfu_ex = [], []          # donor va lan chiet cua tung ho so, song song voi rfu_tot
    far_in = []
    for i in ss:
        m = pat.search(str(names[i]))
        if m is None or not (ng[i] > 0):
            continue
        tpl.append(float(ng[i])); inj.append(float(m.group(1)))
        # Both curves below are keyed on a COPY proxy, height / (injection/reference)^0.80, not on
        # RFU. Injection decides how much finished product reaches the capillary, so two profiles at
        # the same RFU carry different amounts of DNA when they were injected for different times -
        # and dropout and peak-height scatter are molecule-count phenomena. Pooled over PROVEDIt's
        # three injection times the RFU-keyed curve looks fine; split, it is not a law at all. Allele
        # dropout at a profile median of 60-150 RFU reads 0.090 / 0.253 / 0.359 at 5 / 15 / 25 sec, a
        # fourfold spread, and the heterozygote CV reads 0.345 / 0.494 / 0.541. Dividing the height by
        # injection^0.80 - the exponent that minimises the disagreement, scanned on NOC1 - puts 15 and
        # 25 sec on top of each other (0.397 / 0.387) and takes the mean spread from 0.70 to 0.39.
        _ig = (float(m.group(1)) / _INJ_REF) ** _INJ_BETA
        c = int(y[i].argmax()); v = np.where(mk[i])[0]
        _h = np.expm1(tok[i, v, 2].astype(np.float64))
        _keys = [(int(round(float(tok[i, p, 0]))), int(round(float(tok[i, p, 1]) * 10))) for p in v]
        _hm = dict(zip(_keys, _h))
        if len(_h):
            rfu_tot.append(float(_h.sum())); at_min.append(float(_h.min())); rfu_don.append(c)
            _exm_ = _RE_EXTRACT.search(str(names[i]))
            rfu_ex.append(_exm_.group(1) + _exm_.group(2) if _exm_ else f"?{c}")
            # Counted over every position, not just those feas_filter passes: the same filter runs on
            # both sides, so matching the raw population makes the visible one match by itself.
            _fk = [k_ for k_ in _keys if k_ not in key[c] and not any(
                a[0] == k_[0] and abs(a[1] - k_[1]) <= ctx.ART_SPAN for a in key[c])]
            far_cnt.append(len(_fk))
            far_in.append(sum(1 for k_ in _fk if k_ in ctx.CARR_SET))
            drop_rec.append((float(_h.sum() / max(len(key[c]), 1)) / _ig,
                             1.0 - len(key[c] & set(_keys)) / max(len(key[c]), 1)))
            _byl = {}
            for a in key[c]:
                _byl.setdefault(a[0], []).append(a)
            for _L, _al in _byl.items():
                if len(_al) == 2:
                    _x = [_hm.get(a, 0.0) for a in _al]
                    if min(_x) > 0:
                        hb_pairs.append((float(np.mean(_x)) / _ig, float(min(_x) / max(_x))))
        keys = [(int(round(float(tok[i, p, 0]))), int(round(float(tok[i, p, 1]) * 10))) for p in v]
        h = np.expm1(tok[i, v, 2].astype(np.float64)); hm = dict(zip(keys, h))
        for kk, hh in zip(keys, h):
            if kk in key[c]:
                continue
            # NON-stutter artefacts only: stutter is generated separately from each parent peak, so
            # weighting the noise floor by every artefact position would emit it twice and force
            # NOISE_N down to absorb the duplication.
            if (kk[0], kk[1] + 10) not in key[c] and (kk[0], kk[1] - 10) not in key[c]:
                j = ctx._BININDEX.get((kk[0], round(kk[1] / 10.0, 1)), -1)
                if j >= 0:
                    art_w[j] += 1.0
            par = (kk[0], kk[1] + 10)
            if par in key[c] and par in hm and hm[par] > 0:
                sr.append(hh / hm[par])
    # Degradation per TREATMENT, and sources restricted to untreated profiles. The naming convention
    # document names the designators outright, so none of this needs inferring from Q: "a" is
    # untreated, b-e are DNase I at 3/6/12/24 mU, -15/-30/-45 are Fragmentase minutes, S2/S10/S30 are
    # sonication cycles, U15/U60/U105 are UV minutes, and I15/I22/I35 are humic acid VOLUMES - which
    # inhibit PCR rather than break DNA. The measurement agrees: humic acid shows an excess slope of
    # 0.0000 while sonication at 30 cycles reaches 0.0118, so folding them onto one Q axis, as an
    # earlier version did, treats an inhibited sample as a degraded one. Q also bottoms out near 0.9
    # for pristine DNA, so a slope fitted against Q was charging the natural size decline as damage.
    try:
        lab_ = np.array(_json.load(open(ctx.DATA / "condition_train.json")))[ss]
        bmap = {}
        for i2, i in enumerate(ss):
            c = int(y[i].argmax()); v = np.where(mk[i])[0]
            kk3 = [(int(round(float(tok[i, t3, 0]))), int(round(float(tok[i, t3, 1]) * 10)))
                   for t3 in v]
            hh3 = np.expm1(tok[i, v, 2].astype(np.float64))
            ow = np.array([k_ in key[c] for k_ in kk3])
            sz3 = np.array([ctx.SZ.get(k_, 0.0) for k_ in kk3])
            m3 = ow & (sz3 > 0) & (hh3 > 0)
            if m3.sum() >= 12:
                b3, _ = np.polyfit(sz3[m3], np.log(hh3[m3]), 1)
                bmap.setdefault(str(lab_[i2]), []).append(-b3)
        base_ = float(np.median(bmap["a"]))
        # NOT clamped at zero. A treatment can be FLATTER than untreated and three of them are:
        # inhibition does not fragment DNA, it only starves the reaction, so an inhibited sample
        # declines with fragment length LESS than the untreated baseline. Real NOC1 gives I15/I22/I35
        # at -0.00019/-0.00011/-0.00028 against untreated 'a' at -0.00053. The clamp forbade the
        # negative beta those need, so they fell through to 0, inherited the whole of DEG_BASE and
        # came out at -0.00045/-0.00055/-0.00065 - too steep by half the range they should span. That
        # is what compressed the twin's slope range to 85% of real's and made the error change sign
        # across treatments: an additive floor on codes that needed to sit below it.
        out["cond_beta"] = {k: float(np.median(v)) - base_
                            for k, v in bmap.items() if len(v) >= 40}

        # BETWEEN-SAMPLE spread of the degradation slope, taken from the damage MECHANISM rather
        # than from a per-treatment table. Real NOC1 has a true spread (fit noise removed) running
        # 0.00035 to 0.00306 across treatments while the generator, imposing one fixed slope per
        # code, has exactly zero. Reading it off per code would mean memorising seventeen
        # distributions; grouping by mechanism and dose - which the PROVEDIt naming document gives
        # (b-e DNase I at 3/6/12/24 mU, -15..-45 Fragmentase, U15..U105 UV minutes, S2..S30
        # sonication cycles, I15..I35 humic acid inhibition) - leaves three laws.
        #
        # Fragmentation is random SCISSION: breaks fall as a Poisson process along the molecule,
        # which is why height decays as exp(-beta*length) and beta IS the break density. A Poisson
        # count has variance equal to its mean, so sd must go as sqrt(beta). Fitting the exponent
        # sd = A*beta^p with bootstrap CIs: enzymatic 0.53 [0.38, 0.74] - contains 0.5, EXCLUDES 1.0.
        # And the constant is the same for two unrelated enzymes: 0.0318 [0.0292, 0.0338] for DNase I
        # against 0.0315 [0.0291, 0.0338] for Fragmentase, difference 0.0002 [-0.0030, 0.0034].
        #
        # Sonication gives p = 0.94 [0.68, 1.44] - contains 1.0, EXCLUDES 0.5 - because mechanical
        # shearing delivers a dose and the variation is in the dose delivered, so it is
        # multiplicative: CV = 0.247 +/- 0.006 over a fourfold range of beta.
        #
        # UV gives p = 2.51 [1.63, 3.70], neither law, and should not follow either: UV makes
        # polymerase-blocking photoproducts rather than strand breaks, and beta SATURATES in the
        # lesion count - a fragment already blocked is not blocked harder. So the Poisson counting
        # shows up in the DOSE, not in beta: sd/sqrt(minutes) has cv 0.102 against 0.256 for
        # sqrt(beta) and 0.208 for beta. The same saturation is why the apparent exponent blows up.
        #
        # Untreated and inhibited samples get essentially nothing: measured true spread 0.00013 for
        # 'a' and 0.00000/0.00000/0.00027 for I15/I22/I35. No damage process, no damage variation -
        # and the scission law would have over-predicted them five to sevenfold.
        K_SCISSION, CV_SHEAR, C_UV, SD_BASE = 0.0317, 0.247, 0.00025, 0.0001

        def _mech_sd(code, beta):
            c = str(code)
            if _re.fullmatch(r"[b-e]", c) or _re.fullmatch(r"-\d+", c):
                return K_SCISSION * float(np.sqrt(max(beta, 0.0)))          # enzymatic scission
            m = _re.fullmatch(r"S(\d+)", c)
            if m:
                return CV_SHEAR * max(beta, 0.0)                            # mechanical shearing
            m = _re.fullmatch(r"U(\d+)", c)
            if m:
                return C_UV * float(np.sqrt(float(m.group(1))))             # UV, Poisson in dose
            return SD_BASE                                                  # untreated, inhibited

        out["cond_sd"] = {k: float(_mech_sd(k, out["cond_beta"][k] + base_))
                          for k in out["cond_beta"]}
        # Damage does not only tilt a profile by fragment size, it ROUGHENS it: paired against the same
        # donor's untreated run at the same template and injection, DNase at level e widens the spread
        # between peaks 1.82x. A size multiplier cannot make that, and without it the twin's peaks were
        # too orderly. The extra is fitted as added log-variance, s_out^2 = s_in^2 + g^2, which is the
        # form the data supports and which SATURATES for free: the same g lands on a mixture that is
        # already spread and lifts it far less. That prediction was checked against 1060 paired real
        # mixtures without being fitted to them - 1.08/1.13/1.16/1.22 predicted from single-source runs
        # alone against 1.07/1.12/1.16/1.18 measured - so mixtures need no table of their own and the
        # whole transform can be re-derived from one untreated profile per donor.
        Xu = np.load(ctx.DATA / "Xflat_train.npy").astype(np.float32)[ss]
        du = y[ss].argmax(1)
        # Grouped by TEMPLATE. 'Untreated' is not 'same conditions' - the runs still span a 64x range of
    # loaded mass, and every height relation is level-dependent. Same-run grouping was measured and
    # rejected: no run holds 5 untreated donors and two from one run are no closer than two from
    # different runs (1.512 vs 1.577), because template was varied WITHIN a run.
        BSZ = ctx.build_bin_size()
        OWNB = {}
        for c4 in range(gg.shape[0]):
            mm4 = np.zeros(ctx.N_FLAT, dtype=bool)
            for j4 in range(ctx.N_FLAT):
                if (int(ctx.BIN_LOCUS[j4]), int(round(float(ctx.BIN_ALLELE[j4]) * 10))) in key[c4]:
                    mm4[j4] = True
            OWNB[c4] = mm4
        tpl_u = np.load(ctx.DATA / "meta_template_train.npy").astype(np.float64)[ss]
        # ONE value per dilution rung. The names spell 1/32 ng both 0.0312 and 0.0313, which split that
        # rung into two levels: the untreated ladder lost the 0.0312 half outright (too few donors on
        # its own), the treated ladders carried it as a separate level, and the dilution pairs used for
        # the level conversion found almost no 'adjacent' neighbour for 0.0156. Anything within 3% of a
        # rung 0.5 / 2^k is that rung; amounts off the ladder (treated samples) are left as they are.
        # Snapped to the rung as the rest of this file spells it - rounded to 4 places - because every
        # level lookup below compares against round(t, 4) at 1e-6; snapping to 0.015625 exactly made
        # every rung under 0.0625 match nothing and emptied the faint half of the ladder.
        _rung = np.round(0.5 / 2.0 ** np.arange(8), 4)
        _near = np.abs(np.log(tpl_u[:, None] / _rung[None, :]))
        _snap = _near.min(1) < np.log(1.03)
        tpl_u = np.where(_snap, _rung[_near.argmin(1)], tpl_u)
        inj_u = np.array([float(_re.search(r"[_.](\d+)\s*sec", str(n)).group(1))
                          if _re.search(r"[_.](\d+)\s*sec", str(n)) else 0.0
                          for n in _json.load(open(ctx.DATA / "meta_sample_names_train.json"))])[ss]
        # The quality index Q of the extract each profile was amplified from (Quantifiler Trio 80 bp / 214 bp, in
        # the sample name). NaN for LAND - large target not detected - and for names without one.
        _qrx = _re.compile(r"-Q([\d.]+)_")
        q_u = np.array([float(_qrx.search(str(n)).group(1)) if _qrx.search(str(n)) else np.nan
                        for n in _json.load(open(ctx.DATA / "meta_sample_names_train.json"))])[ss]
        try:
            byd = {}
            for j4, i in enumerate(ss):
                if lab_[j4] != "a" or tpl_u[j4] <= 0 or inj_u[j4] <= 0:
                    continue
                byd.setdefault((int(du[j4]), float(inj_u[j4])), []).append(j4)
            sh = {}
            for _k4, v4 in byd.items():
                for p4 in v4:
                    for q4 in v4:
                        if tpl_u[q4] <= tpl_u[p4] * 1.5:
                            continue
                        sp = []
                        own4 = OWNB[int(du[p4])]
                        for z4 in (p4, q4):
                            hz = np.expm1(Xu[z4].astype(np.float64))
                            nzz = (hz > 0) & own4
                            if nzz.sum() < 12:
                                sp = []
                                break
                            bz = np.polyfit(BSZ[nzz], np.log(hz[nzz]), 1)
                            sp.append(float(np.std(np.log(hz[nzz]) - np.polyval(bz, BSZ[nzz]))))
                        if len(sp) == 2 and sp[0] > 0:
                            sh.setdefault(int(round(np.log2(tpl_u[q4] / tpl_u[p4]))), []).append(sp[1] / sp[0])
            out["tpl_shrink"] = {k4: float(np.median(v4)) for k4, v4 in sh.items() if len(v4) >= 60}
        except Exception:
            pass
        lv = np.array(sorted({round(float(t), 4) for t in tpl_u if t > 0}))
        clean = lab_ == "a"
        pc = {}
        for L3 in lv:
            m3 = clean & (np.abs(tpl_u - L3) < 1e-6)
            if m3.sum() < 20:
                continue
            # Only levels that are COMPLETE. A source has to have lost nothing, so that dropout is
            # applied once, here, rather than inherited from a faint run and then charged again. The
            # two faintest templates have already dropped alleles of their own; drawing from them and
            # scaling up cannot put them back, and sharing such a level across the contributors cost
            # count 0.839 -> 0.732.
            if float(np.median(np.expm1(Xu[m3]).sum(1))) < 15000.0:
                continue
            out.setdefault("tpl_ref", float(L3))
            out["tpl_ref"] = max(out["tpl_ref"], float(L3))
            d3 = {c: Xu[m3 & (du == c)] for c in range(gg.shape[0]) if (m3 & (du == c)).any()}
            if len(d3) >= 30:
                pc[float(L3)] = d3
        out["pool_clean"] = pc
        # EVERY template level, complete or not, kept SEPARATELY. pool_clean above deliberately keeps
        # only levels that have lost nothing, because a source scaled DOWN from there has its dropout
        # manufactured once, in the generator. That is the right rule if the twin must manufacture -
        # but it does not have to. PROVEDIt holds real untreated NOC1 profiles at all seven levels,
        # 0.0078 to 0.5 ng, covering 42-45 of the 45 donors. Each (donor, level) cell is ONE
        # amplification: its 1-3 runs are injections of a single PCR product, and only 0.9% of
        # untreated cells hold a second (the 76.6% once quoted here counted injections). So a twin
        # drawn from a real sample's own cell gets that sample's amplification; a genuinely different
        # one exists only at another level, which is what the generator's level conversion is for. A twin at a
        # faint template built from a real profile at that template inherits the machine's own
        # dropout instead of a modelled one - and at NOC1 there is no mixing and no interaction, so
        # that dropout is pure instrument error, which is exactly what NOC1 is for. The old pool
        # forced a scale-down of up to 64x; matched levels are 2x apart.
        pl = {}
        for L3 in lv:
            m3 = clean & (np.abs(tpl_u - L3) < 1e-6)
            if m3.sum() < 20:
                continue
            # The run's INJECTION rides along with each profile. Template alone does not fix what a
            # profile has lost: dropout is a molecule-count phenomenon and the injection decides how
            # much product reaches the capillary, so at 0.0078 ng real retention runs 0.644 / 0.728 /
            # 0.771 at 5 / 15 / 25 sec and at 0.0156 ng 0.822 / 0.904 / 0.891, against a replicate
            # envelope of 0.0343. From 0.031 ng up the three agree to within 0.006. The pool is split
            # in even thirds across the three times, so choosing on template alone hands back a run at
            # the wrong injection two draws out of three. Kept as a parallel array rather than as a
            # separate key so the cell counts stay whole and the generator can fall back.
            d3 = {c: (Xu[m3 & (du == c)], inj_u[m3 & (du == c)], q_u[m3 & (du == c)])
                  for c in range(gg.shape[0]) if (m3 & (du == c)).any()}
            if len(d3) >= 15:
                pl[float(L3)] = d3
        out["pool_lvl"] = {c: sorted([(float(L3), d3[c][0], d3[c][1], d3[c][2])
                                      for L3, d3 in pl.items() if c in d3], key=lambda e: e[0])
                           for c in range(gg.shape[0])}
        # ... and the same again keyed on the TREATMENT. An untreated source cannot stand in for a
        # DNase-digested one: the treatment does not merely tilt the heights, it destroys alleles, and
        # the degradation model only tilts. Measured on NOC1 with template-matched untreated sources,
        # retention comes out +0.0089 on untreated samples and +0.0749 on treated ones, the error
        # growing with severity - +0.1140 at DNase 24 mU and +0.1081 at 60 min UV against +0.0133 at
        # the mildest inhibition. The reference set carries 4232 treated single-source profiles over
        # 17 conditions, every condition at six to eight template levels, so the source can be matched
        # on condition as well. Donor coverage per condition is thinner than for untreated (45 donors
        # untreated, 31 for DNase, 12-14 for sonication and Fragmentase, 7 for 'b'), which is why the
        # lookup falls back to the untreated ladder when a donor has no run under that condition.
        pcnd = {}
        for cd_ in sorted({str(x) for x in lab_}):
            m4 = (lab_ == cd_)
            if m4.sum() < 60:
                continue
            per = {}
            for L3 in lv:
                m3 = m4 & (np.abs(tpl_u - L3) < 1e-6)
                if m3.sum() < 6:
                    continue
                for c in range(gg.shape[0]):
                    m5 = m3 & (du == c)
                    if m5.any():
                        per.setdefault(c, []).append((float(L3), Xu[m5], inj_u[m5]))
            if per:
                pcnd[cd_] = {c: sorted(v, key=lambda e: e[0]) for c, v in per.items()}
        out["pool_cond"] = pcnd
        # ONE pool over every complete level, each entry tagged with the template it came from, so the
        # draw keeps all 45 donors and all their runs instead of being confined to a single level.
        out["pool_clean_all"] = {c: [(float(L3), d3[c]) for L3, d3 in pc.items() if c in d3]
                                 for c in range(gg.shape[0])}
    except Exception:
        pass
    # Q index per pool profile (build_ss_pool ordering) and the degradation slope it implies. Real
    # degradation acts through FRAGMENT SIZE, not as a flat multiplier: measured on 4891 profiles the
    # slope runs from 0.00087 at Q<0.8 to 0.00689 at Q>4, correlation +0.453 with log Q. That
    # distinction is what decides WHO loses alleles - a size-dependent cut takes the weak contributor's
    # long fragments, which are already faint, while the strong contributor's merely shrink. Real
    # confirms it: the strongest contributor's dropout is 0.000 at every Q level, and all of the loss
    # sits on the middle and weak ones (0.154 and 0.231 at high Q).
    try:
        qidx = np.load(ctx.DATA / "meta_qindex_train.npy").astype(float)
        ds_ = y[ss].argmax(1); q_ss = qidx[ss]
        out["pool_q"] = {c: q_ss[ds_ == c] for c in range(gg.shape[0])}
        out["q_all"] = q_ss[np.isfinite(q_ss)]
        bq = []
        for i in ss:
            c = int(y[i].argmax()); v = np.where(mk[i])[0]
            kk2 = [(int(round(float(tok[i, t2, 0]))), int(round(float(tok[i, t2, 1]) * 10)))
                   for t2 in v]
            hh2 = np.expm1(tok[i, v, 2].astype(np.float64))
            ow = np.array([k_ in key[c] for k_ in kk2])
            sz2 = np.array([ctx.SZ.get(k_, 0.0) for k_ in kk2])
            m2 = ow & (sz2 > 0) & (hh2 > 0)
            if m2.sum() >= 12 and np.isfinite(qidx[i]):
                b2, _ = np.polyfit(sz2[m2], np.log(hh2[m2]), 1)
                bq.append((np.log(max(qidx[i], 0.05)), -b2))
        bq = np.array(bq)
        A2 = np.polyfit(bq[:, 0], bq[:, 1], 1)
        out["beta_q"] = [float(A2[0]), float(A2[1])]
    except Exception:
        pass
    # Sources indexed by DEGRADATION, restricted to profiles loud enough that nothing was lost to low
    # template. The two axes are different things and were being conflated: a faint profile has lost
    # alleles because there was little DNA, a degraded one because the fragments are broken. Drawing at
    # random let a MAJOR contributor be built from a faint profile that had already lost a fifth of its
    # alleles, which no real mixture does - real's strongest contributor loses 0.000 at every Q level,
    # while the twin's lost 0.067. Selecting a loud profile AT the wanted degradation keeps the two
    # apart, and it is the right order of operations besides: a sample degrades before it is mixed, so
    # degradation belongs to the source choice, not to a correction applied afterwards.
    try:
        Xl = np.load(ctx.DATA / "Xflat_train.npy").astype(np.float32)[ss]
        dl = y[ss].argmax(1); ql = np.load(ctx.DATA / "meta_qindex_train.npy").astype(float)[ss]
        loud = np.expm1(Xl).sum(1) >= 15000.0
        # The CLEANEST profiles: loud (nothing lost to low template) and least degraded (nothing lost
        # to broken fragments). Selecting a profile that already sits at the wanted degradation was the
        # wrong half of the choice - the loudness filter removes degraded profiles, since a degraded
        # sample is faint at the long end, so the pool it left was uniformly flat in fragment size and
        # a contributor's heights came out 118/122/127 RFU across 100-400 bp where real declines 2.2x.
        # With a flat source there is nothing for the height-dependent dropout to bite on, which is why
        # the twin eroded short fragments as hard as long ones and real does the opposite.
        # Starting clean and applying the degradation the mixture calls for puts that curve back under
        # control, and it is what a lab with a handful of profiles per donor would have to do anyway.
        pool_c = {}
        for c in range(gg.shape[0]):
            m_ = (dl == c) & loud
            if not m_.any():
                continue
            qi = ql[m_]; xi = Xl[m_]
            thr = np.nanpercentile(qi, 40) if np.isfinite(qi).any() else np.inf
            keep_ = np.where(~np.isfinite(qi) | (qi <= thr))[0]
            pool_c[c] = (xi[keep_], qi[keep_])
        out["pool_loud"] = pool_c
    except Exception:
        pass
    # Profiles with no allele dropout, for use as intact overlay sources.
    try:
        Xs_ = np.load(ctx.DATA / "Xflat_train.npy").astype(np.float32)[ss]
        ds_ = y[ss].argmax(1)
        keep = {}
        for c in range(gg.shape[0]):
            sel = np.where(ds_ == c)[0]
            good = [j for j in sel if np.expm1(Xs_[j]).sum() >= 15000.0]
            if good:
                keep[c] = Xs_[good]
        out["pool_full"] = keep
    except Exception:
        pass
    # STRUCTURED artefact table: for each offset from a true allele, how often an artefact appears
    # there and how tall it is relative to that allele. Real artefacts are not scattered - 48.6% sit one
    # repeat below, 16.6% one above, 10.4% two below, and 19.4% at FRACTIONAL positions such as +0.3,
    # which an integer-offset stutter formula cannot produce. Half of the extra alleles real shows at a
    # crowded locus are fractional. This also replaces an SR formula whose coefficients were fitted to
    # a different kit.
    try:
        off_cnt, off_rat, n_true_pk = {}, {}, 0
        loc_cnt, loc_rat, loc_true, loc_true_h = {}, {}, {}, {}
        for i in ss:
            c = int(y[i].argmax()); v = np.where(mk[i])[0]
            kk_ = [(int(round(float(tok[i, p, 0]))), int(round(float(tok[i, p, 1]) * 10))) for p in v]
            hh = np.expm1(tok[i, v, 2].astype(np.float64)); hmap = dict(zip(kk_, hh))
            n_true_pk += sum(1 for k_ in kk_ if k_ in key[c])
            for k_, x_ in zip(kk_, hh):
                if k_ in key[c]:
                    loc_true[k_[0]] = loc_true.get(k_[0], 0) + 1
                    loc_true_h.setdefault(k_[0], []).append((k_[1], float(x_), c))
            for k_, x_ in zip(kk_, hh):
                if k_ in key[c]:
                    continue
                # Counted against each donor allele in range - but for the PRODUCT offsets only where
                # the bin is CLEAN, i.e. not also a +-1 / +-2 repeat position of another of the donor
                # alleles. A bin that is n-1 of one allele and n-2 of another holds ONE peak, and it is
                # there because of the n-1 (present 90% of the time at tall parents); charging it to
                # n-2 as well made every minor offset look common and tall. Measured on NOC1: at clean
                # positions n+2 sits at the background rate (0.071-0.083) at EVERY parent height, while
                # the pooled count put it at 1.62x that; n-2 was inflated 1.45x and n+1 1.28x, and their
                # height ratios read at overlapping positions are the n-1 ratio. The generator emits
                # each offset from each parent independently, so the union at an overlapping bin is
                # produced there by itself - the table must hold the clean, single-process rates.
                for near in key[c]:
                    if near[0] != k_[0] or hmap.get(near, 0) <= 0:
                        continue
                    d = k_[1] - near[1]
                    if abs(d) > ctx.ART_SPAN and d != -30:
                        continue
                    if abs(d) in (10, 20, 30) and any(o != near and o[0] == k_[0] and (k_[1] - o[1]) in (-20, -10, 10, 20)
                                                  for o in key[c]):
                        continue
                    off_cnt[d] = off_cnt.get(d, 0) + 1
                    off_rat.setdefault(d, []).append(x_ / hmap[near])
                    loc_cnt[(k_[0], d)] = loc_cnt.get((k_[0], d), 0) + 1
                    loc_rat.setdefault((k_[0], d), []).append((x_ / hmap[near], near[1] / 10.0, hmap[near]))
        tab = []
        for d, cnt in off_cnt.items():
            r = np.array(off_rat[d]); r = r[r > 0]
            if cnt / max(n_true_pk, 1) < 0.004 or len(r) < 30:
                continue
            tab.append((int(d), min(ctx.ART_EMIT * cnt / n_true_pk, 1.0),
                        float(np.mean(np.log(r))), float(np.std(np.log(r)))))
        out["art_table"] = sorted(tab, key=lambda t: -t[1])
        # Stutter is LOCUS-SPECIFIC, which is why kit manufacturers publish a separate filter per
        # locus rather than one number: a long uninterrupted repeat slips far more often than a short
        # or compound one. Measured here the n-1 occurrence rate runs from 0.10 to 0.61 and the height
        # ratio from 0.017 to 0.090 across loci, and AMEL - which has no repeat unit at all - shows
        # 0.000, so a single global rate was emitting stutter off a sex marker. One rate for every
        # locus also spreads peaks evenly, while real concentrates them: at NOC5 real reaches MAC 12.10
        # from 138 peaks where the twin managed 11.58 from 148.
        # Only INTEGER offsets are products of the parent. Measured on NOC1, the rate at -1/-2/+1/+2
        # rises 1.9-4.8x from a faint parent to a tall one and their height rises with it (9 -> 150
        # RFU at -1), while every FRACTIONAL offset is flat or falling (+0.3 goes 0.025 -> 0.010) and
        # holds at 8-9 RFU whatever the parent does. Fractional bins in fact carry LESS than an
        # ordinary bin - 0.040 against the 0.077 baseline - so there is nothing there to emit: they
        # are the noise floor, and a thin patch of it. Treating them as products of the parent gave a
        # 3000 RFU allele a tall companion real never shows, 7.6 unexplained peaks above 20 RFU per
        # NOC5 profile against real's 1.4.
        #
        # The product offsets are estimated in three steps, because a rate pooled over every parent
        # measures emission and survival multiplied together and reports neither: the shipped table
        # read 0.542 where true emission is 0.75, which put five times too many faint stutters in the
        # twin (9.5 per NOC5 profile below 15 RFU against real's 3.6) and too few tall ones. First the
        # height ratio, read at the tallest parents where nothing is lost; then the survival curve
        # that ratio implies; then the emission rate, with each parent weighted by the survival its
        # own expected artefact would have. Weighting rather than cutting keeps n-2 and n+2, whose
        # ratio is small enough that a fixed parent-height cut leaves them no data at all.
        # Pass 1 - the height law, fitted where the artefact is large enough that almost none is lost.
        # Stutter grows with the parent's repeat count, the SR-against-allele regression every kit
        # validation reports, measured positive on 23 of 23 loci here. The scatter left after that
        # trend splits into a per-RUN offset and a per-peak residual: slippage efficiency belongs to
        # one amplification, so every locus in a sample leans the same way, and calling the whole
        # spread per-peak noise puts 37-39% of the variance in the wrong place and makes each
        # generated sample rougher than any real one.
        # n-3 is a product too, though the span stops at 2.5 repeats. On held-out NOC1 the peaks at exactly -3
        # rise with the parent (2.03 -> 2.30 per profile from 0.063 to 0.25-0.5 ng) while the twin, which put only
        # the noise floor there, fell to 1.69; at 0.25-0.5 ng it was the largest single artefact shortfall. It is
        # measured by the same three passes as the others, so its rate is net of the floor already there.
        _fit = {}
        for (L_, d), cnt in loc_cnt.items():
            if d % 10 or (abs(d) > 25 and d != -30) or loc_true.get(L_, 0) < 200:
                continue
            _rr = np.array(loc_rat[(L_, d)], float); _rr = _rr[_rr[:, 0] > 0]
            if len(_rr) < 30:
                continue
            _t = _rr[:, 2] >= 2500
            if _t.sum() < 20:
                _t = _rr[:, 2] >= np.percentile(_rr[:, 2], 80.0)
            _m = _rr[:, 2] * float(np.median(_rr[_t, 0])) >= 65.0
            if _m.sum() < 30:
                continue
            r = _rr[_m, 0]; _av = _rr[_m, 1]; _lg = np.log(r)
            if len(r) >= 60 and _av.std() > 0.3:
                _sl, _b0 = np.polyfit(_av, _lg, 1)
                _a0 = float(_av.mean())
                _ic = float(_sl * _a0 + _b0)                  # log-mean AT the mean parent allele
                _sd = float(np.std(_lg - (_sl * _av + _b0)))  # residual, trend removed
            else:
                _sl, _a0, _ic, _sd = 0.0, 0.0, float(_lg.mean()), float(np.std(_lg))
            _fit[(int(L_), int(d))] = (_ic, _sd, float(_sl), _a0, float(np.percentile(r, 99.9)))
        # POOLED over loci, per product offset, for loci whose own data is too thin to fit. Counted at
        # clean positions only, the minor offsets are rarer and shorter than the pooled-over-overlap count
        # made them (n-2 median ratio 0.0071 -> 0.0048), so the fit criterion above - parents whose
        # expected artefact reaches 65 RFU - asks for parents over 13000 RFU and most loci fail it. A locus
        # that fails used to be DROPPED from the table, i.e. given no n-2 at all: n-2 fell from 11 loci to
        # 4 and NOC1 came out short of real by 0.2 at tall parents. A locus without its own fit now takes
        # the offset's pooled law. Where even the pool cannot meet the 65 RFU criterion the tall-parent
        # set is used as it is, which reads a surviving population and leans slightly high.
        _fitp = {}
        for _d in (-30, -20, -10, 10, 20):
            _rows = [np.array(v, float) for k, v in loc_rat.items() if int(k[1]) == _d and len(v)]
            if not _rows:
                continue
            _rr = np.concatenate(_rows); _rr = _rr[_rr[:, 0] > 0]
            if len(_rr) < 30:
                continue
            _t = _rr[:, 2] >= 2500
            if _t.sum() < 20:
                _t = _rr[:, 2] >= np.percentile(_rr[:, 2], 80.0)
            _m = _rr[:, 2] * float(np.median(_rr[_t, 0])) >= 65.0
            if _m.sum() < 30:
                _m = _t
            _lg = np.log(_rr[_m, 0])
            _fitp[_d] = (float(_lg.mean()), float(_lg.std()), 0.0, 0.0, float(np.percentile(_rr[_m, 0], 99.9)))
        # Pass 2 - survival, read against EXACTLY the expected height the generator will compute. A
        # flat per-locus ratio was used here first and the table came out inconsistent with itself:
        # it predicted an n-1 rate of 0.477 on real's own parents where real measures 0.573, because
        # the allele trend spreads expected heights across a curve that is anything but linear.
        _EH, _SP, _ = _art_survival(ctx, ss, tok, mk, y, key, _fit)
        _PI, _PF, _PN, _NM, _NS, _LMU, _LP, _NR, _BP, _LQ, _RHO = _noise_floor(ctx, ss, tok, mk, y, key)
        out["noise_loc_q"], out["noise_rho"] = _LQ, _RHO
        out["noise_bin_p"] = _BP
        out["noise_run_sd"] = _NR
        out["art_surv_h"], out["art_surv_p"] = _EH, _SP
        out["noise_p_int"], out["noise_p_frac"] = _PI, _PF
        out["noise_ln_mu"], out["noise_ln_sd"] = _NM, _NS
        out["noise_loc_mu"], out["noise_loc_p"] = _LMU, _LP
        _BG = _PI                                    # what an integer bin carries with nothing emitted
        # ...and what each bin carries in the generator: the same per-locus and per-bin rates it builds its
        # noise floor from, so the background charged here is the one the generator will actually add.
        _BGBIN = {}
        _fr_ = _PF / max(_PI, 1e-9)
        for _j in range(ctx.N_FLAT):
            _Lj = int(ctx.BIN_LOCUS[_j]); _aj = int(round(float(ctx.BIN_ALLELE[_j]) * 10))
            _isf = (_aj % 10) != 0
            _v = _PF if _isf else _PI
            if (_LP or {}).get(_Lj) is not None:
                _v = float(_LP[_Lj]) * (_fr_ if _isf else 1.0)
            _bv = (_BP or {}).get("%d_%d" % (_Lj, _aj))
            if _bv is not None:
                _v = float(_bv) * (_fr_ if _isf else 1.0)
            _BGBIN[(_Lj, _aj)] = float(_v)
        # Pass 3 - the emission rate, solved so the table reproduces what NOC1 actually shows. A bin
        # is occupied when the artefact survives OR the baseline noise lands there, which for an
        # emission e and a mean survival s is e*s + bg*(1 - e*s), so e = (obs - bg) / ((1-bg) * s).
        # Estimating e by counting instead - even weighting each parent by its survival - leaves the
        # table inconsistent with the profiles it came from, because emission and survival correlate
        # across loci and the mean of the product is not the product of the means: read that way the
        # table predicted 0.392 for an offset real puts at 0.573.
        tabl = {}
        for (L_, d), cnt in loc_cnt.items():
            _rr = np.array(loc_rat[(L_, d)], float)
            _rr = _rr[_rr[:, 0] > 0]
            nt = loc_true.get(L_, 0)
            if nt < 200 or (len(_rr) < 30 and int(d) not in _fitp) or cnt / nt < 0.004 or (abs(d) > 25 and d != -30):
                continue
            # Only parents whose target bin is FREE can show an artefact there. A quarter of them are
            # not: the donor's other allele already sits one repeat down, and the peak that lands
            # there is a real allele, not a stutter. Counting those in the denominator understated
            # every rate by the same quarter.
            _free = [(a_, h_) for a_, h_, c_ in loc_true_h.get(L_, ())
                     if (L_, a_ + d) not in key[c_]
                     and not (abs(d) in (10, 20, 30) and any(o[0] == L_ and o[1] != a_ and (a_ + d - o[1]) in (-20, -10, 10, 20)
                                                         for o in key[c_]))]
            _ah = np.array([(a_ / 10.0, h_) for a_, h_ in _free], float)
            if len(_ah) < 60:
                continue
            f = _fit.get((int(L_), int(d))) or _fitp.get(int(d))
            if f is None:
                continue
            _ic, _sd, _sl, _a0, _rmax = f
            # Each parent's OWN background and OWN survival. One global background (the integer-bin rate)
            # sat below what the generator's noise actually puts on these target bins, and dividing by the
            # MEAN survival is only right when the background is the same for every parent.
            _bgv = np.array([_BGBIN.get((int(L_), int(a_ + d)), _BG) for a_, _ in _free], float)
            _si = np.interp(np.exp(_ic + _sl * (_ah[:, 0] - _a0)) * _ah[:, 1], _EH, _SP)
            _sm = float(_si.mean())
            if _sm <= 0.02:
                continue
            _e = max(float(cnt) - float(_bgv.sum()), 0.0) / max(float(((1.0 - _bgv) * _si).sum()), 1e-9) * _sm
            # A CEILING on the ratio, taken from the largest this offset was ever observed to reach
            # on NOC1. The allele slope is an extrapolation once the parent sits outside the range it
            # was fitted on, and unbounded it produced an n+2 product at 2.98x its own parent - 21094
            # RFU beside a 7076 RFU allele - where real carries no artefact above 2000 RFU in 1333
            # mixtures. Clipping the ALLELE instead was tried and is worse: the fit set is tall-parent
            # only, so its range starts at 17 repeats and clipping lifts every ordinary parent onto
            # it, which took the count to 1197.
            tabl.setdefault(int(L_), []).append(
                (int(d), min(ctx.ART_EMIT * _e / _sm, 1.0), _ic, _sd, _sl, _a0, 0.0,
                 _rmax, 0.0))
        _RUN_SD = 0.0
        try:
            _bp = {}
            for i in ss:
                c = int(y[i].argmax()); v = np.where(mk[i])[0]
                kk = [(int(round(float(tok[i, t, 0]))), int(round(float(tok[i, t, 1]) * 10))) for t in v]
                hh = np.expm1(tok[i, v, 2].astype(np.float64)); hm = dict(zip(kk, hh))
                for a in key[c]:
                    if hm.get(a, 0) <= 0:
                        continue
                    b = (a[0], a[1] - 10)
                    if b in hm and b not in key[c] and hm[b] > 0:
                        _bp.setdefault(int(i), []).append(np.log(hm[b] / hm[a]))
            gp = {k: np.array(v) for k, v in _bp.items() if len(v) >= 6}
            allv = np.concatenate(list(gp.values())); allv = allv - allv.mean()
            mu = np.array([v.mean() for v in gp.values()]); mu = mu - mu.mean()
            wi = np.concatenate([v - v.mean() for v in gp.values()])
            _f = mu.var() / max(mu.var() + wi.var(), 1e-12)     # ty trong phuong sai cua lan chay
            _RUN_SD = float(np.sqrt(_f))                        # nhan voi lsd cua tung offset
        except Exception:
            _RUN_SD = 0.0
        out["art_run_sd"] = _RUN_SD
        out["art_table_loc"] = {k_: sorted(v, key=lambda t: -t[1]) for k_, v in tabl.items()}
        _CE = [0, 1, 3, 6, 11, 21]
        out["art_carr_edges"] = _CE
        out["art_carr_mult"] = _art_rarity(ctx, ss, tok, mk, y, key, tabl, _EH, _SP, _CE)
    except Exception:
        pass
    out["template_ng"] = np.array(tpl)
    out["inj_levels"] = np.array(sorted(set(inj)))
    # ── the remaining constants, derived here instead of transcribed ──────────
    # RFU response. Both exponents come out at 1 within noise, which is what the chemistry says:
    # product scales with input DNA and signal with injection time.
    if len(tpl) >= 30:
        lr = np.log(np.array(rfu_tot)); X_ = np.column_stack(
            [np.log(np.maximum(tpl, 1e-9)), np.log(inj), np.ones(len(tpl))])
        b_, *_ = np.linalg.lstsq(X_, lr, rcond=None)
        out["rfu_coef"] = [float(b_[0]), float(b_[1]), float(b_[2])]
        out["rfu_sd"] = float((lr - X_ @ b_).std())
        # PER-DONOR AMPLIFICATION EFFICIENCY. The residual of that fit is not white: averaged per
        # donor it spreads 0.2368 in log-RFU (0.2435 observed, 0.0569 of it sampling noise over the
        # 45 donors with >=20 profiles each), i.e. the loudest person returns 1.67x the signal of the
        # quietest for the SAME nanograms and the SAME injection. It is a property of the person -
        # blood, extraction yield, template quality - not of the run, so it belongs on the donor.
        # The generator normalises each source profile (h = h / h.sum()) and then imposes the share
        # arithmetically, which divides this out and leaves every twin sitting exactly on its nominal
        # phi (+/-0.005) while real mixtures deviate by 0.019 per donor. Real mixtures are the test
        # set and cannot be fitted; this is the same quantity measured where it is legal to measure.
        # Only the PERSON-level part of that residual survives into a mixture. RD14-0003 mixes whole
        # bloods and extracts them TOGETHER, so whatever the extraction contributed is common to every
        # contributor and cancels out of the share. Decomposed over 822 distinct extracts: between
        # extracts of one person sd = 0.5213, between people sd = 0.1940. Averaging per extract first
        # and shrinking by the reliability keeps the person and drops the extraction.
        _dres = lr - X_ @ b_
        _don = np.asarray(rfu_don); _ex = np.asarray(rfu_ex)
        _exm = {}
        for _e in np.unique(_ex):
            _m = _ex == _e
            if _m.sum() >= 6:
                _exm[_e] = (int(_don[_m][0]), float(_dres[_m].mean()))
        _byd = {}
        for _d, _v in _exm.values():
            _byd.setdefault(_d, []).append(_v)
        _byd = {_d: _v for _d, _v in _byd.items() if len(_v) >= 3}
        if _byd:
            _wit = np.concatenate([np.asarray(_v) - np.mean(_v) for _v in _byd.values()])
            _bet = np.array([np.mean(_v) for _v in _byd.values()])
            _nr = float(np.mean([len(_v) for _v in _byd.values()]))
            _vw = float(_wit.var(ddof=1)); _vb = max(float(_bet.var(ddof=1)) - _vw / _nr, 0.0)
            _rel = _vb / max(_vb + _vw / _nr, 1e-12)          # shrink toward 0, not toward noise
            out["donor_eff"] = {int(_d): float(_rel * np.mean(_v)) for _d, _v in _byd.items()}
            out["donor_eff_sd"] = float(np.sqrt(_vb))
    # Detection threshold: the hard floor of the observed height distribution. It is an instrument
    # setting, and the only constant here that a user is meant to be able to override.
    out["at"] = float(np.median(at_min)) if at_min else 3.0
    # NOISE CROWDING, on the footing the generator reads it. Occupancy is the number of bins carrying
    # the donor's alleles or their product offsets (+-1, +-2 repeats) - what the generator's profile
    # holds before the floor is added - not the observed count, which contains the floor itself; the
    # two differ by half at faint templates (39.9 against 73.7). Noise is counted only where nothing
    # structured reaches (>= 2.5 repeats from any donor allele), the positions the base rate is
    # measured on, and normalised by the generator's own per-bin base rate so bin-to-bin differences
    # do not leak in. Relative to NOC1's mean, so the marginal is unchanged by construction.
    try:
        _bl = ctx.BIN_LOCUS.astype(int); _ba = np.rint(ctx.BIN_ALLELE * 10).astype(int)
        _fb = (_ba % 10) != 0; _fr = _PF / max(_PI, 1e-9)
        _base = np.where(_fb, _PF, _PI).astype(float)
        for _k, _v in (_LP or {}).items():
            _m = _bl == int(_k)
            _base[_m] = float(_v) * np.where(_fb[_m], _fr, 1.0)
        for _k, _v in (_BP or {}).items():
            _Lk, _ak = _k.split("_")
            _j = ctx._BININDEX.get((int(_Lk), int(_ak) / 10.0), -1)
            if _j >= 0:
                _base[_j] = float(_v) * (_fr if _fb[_j] else 1.0)
        _own, _prd, _elg = {}, {}, {}
        for _c in range(gg.shape[0]):
            _al = {}
            _o = np.zeros(ctx.N_FLAT, bool); _pr = np.zeros(ctx.N_FLAT, bool)
            for (_Lk, _a10) in key[_c]:
                _al.setdefault(_Lk, []).append(_a10)
                _j = ctx._BININDEX.get((_Lk, _a10 / 10.0), -1)
                if _j >= 0:
                    _o[_j] = True
                for _d in (-20, -10, 10, 20):
                    _j2 = ctx._BININDEX.get((_Lk, (_a10 + _d) / 10.0), -1)
                    if _j2 >= 0:
                        _pr[_j2] = True
            _e = np.zeros(ctx.N_FLAT, bool)
            for _j in range(ctx.N_FLAT):
                _aa = _al.get(int(_bl[_j]))
                if _aa and not _o[_j]:
                    _e[_j] = int(np.abs(np.asarray(_aa) - _ba[_j]).min()) >= 25
            _own[_c], _prd[_c], _elg[_c] = _o, _pr & ~_o, _e
        _ED = [0.0, 8.0, 12.0, 16.0, 25.0, 1e9]
        _occ, _nb, _eb = [], [], []
        for _i in range(len(ss)):
            _c = int(du[_i]); _h = np.expm1(Xu[_i].astype(np.float64)); _on = _h > 0
            _occ.append(float((_on & (_own[_c] | _prd[_c])).sum()))
            _e = _elg[_c]; _eb.append(float(_base[_e].sum()))
            _bi = np.clip(np.searchsorted(_ED, _h[_e & _on], side="right") - 1, 0, 4)
            _nb.append(np.bincount(_bi, minlength=5).astype(float))
        _occ = np.asarray(_occ); _nb = np.asarray(_nb); _eb = np.asarray(_eb)
        _q = np.quantile(_occ, np.linspace(0.0, 1.0, 7)); _q[-1] += 1.0
        _g = np.clip(np.searchsorted(_q, _occ, side="right") - 1, 0, 5)
        _tot = _nb.sum(0) / max(float(_eb.sum()), 1e-9)
        _S, _M = [], np.ones((5, 6))
        for _k in range(6):
            _s = _g == _k
            _S.append(float(np.median(_occ[_s])))
            _M[:4, _k] = (_nb[_s][:, :4].sum(0) / max(float(_eb[_s].sum()), 1e-9)) / np.maximum(_tot[:4], 1e-12)
        out["noise_cr_h"] = [4.0, 10.0, 14.0, 20.0, 40.0]
        out["noise_cr_s"] = _S
        out["noise_cr_m"] = _M.tolist()
        # Beyond NOC1's range. Mixtures carry 94-123 signal bins at the median where NOC1 tops out near
        # 90, so most of them would read the table's last column. Each band is extended along its own
        # NOC1 log-linear trend (per signal bin) when that trend is a SUPPRESSION; a band whose NOC1
        # trend rises is held at its edge, because that rise is pull-up riding on brightness, not a
        # crowding law, and there is nothing to extend.
        _ext = []
        for _b in range(5):
            if _b == 4:
                _ext.append(0.0); continue
            _w = np.array([float(_nb[_g == _k, _b].sum()) for _k in range(6)])
            _sl = float(np.polyfit(np.asarray(_S), np.log(np.maximum(_M[_b], 1e-6)), 1,
                                   w=np.sqrt(np.maximum(_w, 1.0)))[0])
            _ext.append(min(_sl, 0.0))
        out["noise_cr_ext"] = _ext
    except Exception:
        pass
    # LEVEL CONVERSION: survival relative to the top of the dilution ladder. Pairs of the SAME donor,
    # extract, treatment and injection one level apart are two separate amplifications of one DNA -
    # the step a source drawn one level above its target has to be taken through. Predicted retention
    # of the lower level is the upper profile scaled to the lower one's total and cut at the detection
    # threshold; what real loses beyond that is failure to amplify at all, keyed on the ABSOLUTE
    # template of the lower level. Chained from the top, PHI(0.5 ng) = 1.
    try:
        _AT = float(out["at"])
        _ex = _re.compile(r"RD\d+-\d+-(\d+)d(\d+)")
        _lad = {}
        for _i in range(len(ss)):
            _m = _ex.search(str(names[ss[_i]]))
            if _m:
                _lad.setdefault((int(du[_i]), int(_m.group(2)), str(lab_[_i]), float(inj_u[_i])),
                                {})[round(float(tpl_u[_i]), 4)] = _i
        # The DILUTION LADDER only. Treated samples carry amounts off it (0.003, 0.009, 0.041 ng ...),
        # and sorting every amount that occurs made 'adjacent' mean nothing: almost no pair was one
        # dilution step apart and PHI came out flat. Levels held by >= 20 untreated profiles are the
        # ladder itself; pairs of any treatment that sit on two of its adjacent rungs are used.
        _cnt = {}
        for _t, _lb in zip(tpl_u, lab_):
            if _t > 0 and str(_lb) == "a":
                _cnt[round(float(_t), 4)] = _cnt.get(round(float(_t), 4), 0) + 1
        _LV = sorted(_k for _k, _v in _cnt.items() if _v >= 20)
        _num, _den = {}, {}
        _numi, _deni, _cnti = {}, {}, {}
        for (_c, _x1, _x2, _x3), _dd in _lad.items():
            _o = _own[_c]
            if _o.sum() < 20:
                continue
            for _a in range(len(_LV) - 1):
                _lo, _hi = _LV[_a], _LV[_a + 1]
                if _lo not in _dd or _hi not in _dd:
                    continue
                _zl = np.expm1(Xu[_dd[_lo]].astype(np.float64))
                _zh = np.expm1(Xu[_dd[_hi]].astype(np.float64))
                if _zl.sum() <= 0 or _zh.sum() <= 0:
                    continue
                _pred = float(((_zh * (_zl.sum() / _zh.sum()))[_o] >= _AT).mean())
                if _pred > 0:
                    _num[_lo] = _num.get(_lo, 0.0) + float((_zl[_o] > 0).mean())
                    _den[_lo] = _den.get(_lo, 0.0) + _pred
                    _ki = (float(_x3), _lo)
                    _numi[_ki] = _numi.get(_ki, 0.0) + float((_zl[_o] > 0).mean())
                    _deni[_ki] = _deni.get(_ki, 0.0) + _pred; _cnti[_ki] = _cnti.get(_ki, 0) + 1
        _phi, _run = {_LV[-1]: 1.0}, 1.0
        for _a in range(len(_LV) - 2, -1, -1):
            _L = _LV[_a]
            _st = (_num[_L] / _den[_L]) if _den.get(_L, 0.0) > 0 else 1.0
            _run *= float(min(max(_st, 0.05), 1.0))
            _phi[_L] = _run
        out["amp_step_t"] = [float(_x) for _x in _LV]
        out["amp_step_phi"] = [float(_phi[_x]) for _x in _LV]
        # ...and the same chain PER INJECTION. The sources are matched on injection, and at faint templates what a
        # step loses depends on how much product reached the capillary: on a pooled PHI one step under-dropped at
        # 5 s by 0.016 and over-dropped at 15 s by 0.014 (same-extract NOC1 hold-out). A rung with fewer than 15
        # pairs at an injection takes the pooled step.
        _pinj = {}
        for _ij in sorted({_k[0] for _k in _cnti}):
            _phj, _rnj = {_LV[-1]: 1.0}, 1.0
            for _a in range(len(_LV) - 2, -1, -1):
                _L = _LV[_a]; _k2 = (_ij, _L)
                if _cnti.get(_k2, 0) >= 15 and _deni.get(_k2, 0.0) > 0:
                    _st = _numi[_k2] / _deni[_k2]
                else:
                    _st = (_num[_L] / _den[_L]) if _den.get(_L, 0.0) > 0 else 1.0
                _rnj *= float(min(max(_st, 0.05), 1.0)); _phj[_L] = _rnj
            _pinj[str(_ij)] = [float(_phj[_x]) for _x in _LV]
        out["amp_step_phi_inj"] = _pinj
        # ...and WHICH alleles the step takes. At the same upper-rung height, a long fragment is lost
        # more often: logit of surviving the step falls 0.37-0.69 per 100 bp at every rung (t -3 to
        # -17), while the height term runs 0.2 at 0.0078 ng to 2.1 at 0.25 ng - failure to amplify is
        # all-or-nothing at low template and near-threshold loss at high, and the threshold is already
        # the generator's. Pooled over all rungs, with height held in the fit so the size term is not
        # just 'long peaks are shorter'.
        _BSZ2 = ctx.build_bin_size()
        _rws = {}
        for (_c, _x1, _x2, _x3), _dd in _lad.items():
            _o = _own[_c] & (_BSZ2 > 0)
            for _a in range(len(_LV) - 1):
                _lo, _hi = _LV[_a], _LV[_a + 1]
                if _lo not in _dd or _hi not in _dd:
                    continue
                _zl = np.expm1(Xu[_dd[_lo]].astype(np.float64))
                _zh = np.expm1(Xu[_dd[_hi]].astype(np.float64))
                _s = _o & (_zh > 0)
                if _s.sum() < 10:
                    continue
                _sz = _BSZ2[_s]; _lh = np.log(_zh[_s])
                _rws.setdefault(_lo, []).append(np.column_stack([(_sz - _sz.mean()) / 100.0, _lh - _lh.mean(),
                                                                 (_zl[_s] > 0).astype(float)]))

        def _lfit(_A):
            _Xd = np.column_stack([np.ones(len(_A)), _A[:, 0], _A[:, 1]]); _wv = np.zeros(3)
            for _it in range(60):
                _p = 1.0 / (1.0 + np.exp(-(_Xd @ _wv))); _W = _p * (1.0 - _p) + 1e-9
                _wv = _wv + np.linalg.solve(_Xd.T @ (_Xd * _W[:, None]) + 1e-6 * np.eye(3),
                                            _Xd.T @ (_A[:, 2] - _p))
            return float(_wv[1]), float(_wv[2])
        _Apool = np.concatenate([_x for _v in _rws.values() for _x in _v])
        _sp, _hp = _lfit(_Apool)
        out["amp_step_size"] = _sp
        # PER RUNG, and the HEIGHT term kept. Pooling across rungs without their own intercepts pulls the size
        # slope to -0.378 where every rung reads -0.37 to -0.69; and the height coefficient - discarded until
        # now - runs 0.20 at 0.0078 ng to 2.09 at 0.25 ng: the alleles a step takes are the ones that came out
        # FAINT, which is what makes a faint contributor lose alleles and keep tall ones. The top rung copies
        # the rung below it (no step is taken down from the top).
        _szt, _hct = [], []
        for _a in range(len(_LV)):
            _A2 = np.concatenate(_rws[_LV[_a]]) if _rws.get(_LV[_a]) else None
            if _A2 is not None and len(_A2) >= 500 and 0.01 < float(_A2[:, 2].mean()) < 0.999:
                _s2, _h2 = _lfit(_A2)
            else:
                _s2, _h2 = (_szt[-1], _hct[-1]) if _szt else (_sp, _hp)
            _szt.append(_s2); _hct.append(_h2)
        out["amp_step_size_t"] = [float(x) for x in _szt]
        out["amp_step_h_t"] = [float(max(x, 0.0)) for x in _hct]
        # ...and how the survivors STAND. The lower rung's size slope is flatter: at low template every amplicon
        # stays in exponential phase, at high template the short ones pull ahead as the reaction nears plateau.
        # Read on the alleles present at BOTH rungs, at any height - the generator tilts every allele it keeps.
        # Read on alleles >= 50 RFU at both (0.0004-0.0011 per step) it came out 20-30% steeper, and the
        # same-extract NOC1 hold-out overshot by that much. Cumulative from the top of the ladder, so a step of any
        # length is B(t) - B(L).
        _dsl = {}
        for (_c, _x1, _x2, _x3), _dd in _lad.items():
            _o = _own[_c] & (_BSZ2 > 0)
            for _a in range(len(_LV) - 1):
                _lo, _hi = _LV[_a], _LV[_a + 1]
                if _lo not in _dd or _hi not in _dd:
                    continue
                _zl = np.expm1(Xu[_dd[_lo]].astype(np.float64))
                _zh = np.expm1(Xu[_dd[_hi]].astype(np.float64))
                _s = _o & (_zl > 0) & (_zh > 0)
                if _s.sum() < 8 or float(_BSZ2[_s].std()) < 1.0:
                    continue
                _dsl.setdefault(_lo, []).append(float(np.polyfit(_BSZ2[_s], np.log(_zl[_s]), 1)[0])
                                                - float(np.polyfit(_BSZ2[_s], np.log(_zh[_s]), 1)[0]))
        _B, _acc = {_LV[-1]: 0.0}, 0.0
        for _a in range(len(_LV) - 2, -1, -1):
            _v = _dsl.get(_LV[_a], [])
            _acc += float(np.mean(_v)) if len(_v) >= 15 else 0.0
            _B[_LV[_a]] = _acc
        out["amp_step_tilt"] = [float(_B[_x]) for _x in _LV]
    except Exception:
        pass
    # TREATMENT CONVERSION. A contributor whose donor has no run under the mixture's treatment is drawn
    # from the untreated ladder, and an untreated run carries none of what the treatment destroys: the
    # mixture shortfall that looked like an interaction is +0.006 on untreated mixtures, +0.007 where every
    # contributor had a run under the treatment, and -0.071 where one fell back. Per code, from NOC1 cells of
    # the same donor, rung and injection: the retention ratio treated / untreated at each rung (the mean the
    # conversion keeps), and the logit slope, on fragment size, of an allele surviving the treatment given it
    # survived untreated (how that loss spreads - the long fragments go first).
    try:
        _BSZ3 = ctx.build_bin_size()
        _cells = {}
        for _i in range(len(ss)):
            _cells.setdefault((int(du[_i]), str(lab_[_i]), round(float(tpl_u[_i]), 4), float(inj_u[_i])),
                              []).append(_i)

        def _slope(_A):
            _Xd = np.column_stack([np.ones(len(_A)), _A[:, 0]]); _w = np.zeros(2)
            for _it in range(40):
                _p = 1.0 / (1.0 + np.exp(-(_Xd @ _w))); _W = _p * (1.0 - _p) + 1e-9
                _w = _w + np.linalg.solve(_Xd.T @ (_Xd * _W[:, None]) + 1e-6 * np.eye(2), _Xd.T @ (_A[:, 1] - _p))
            return float(_w[1])

        _tc = {}; _allrows = []
        for _cd in sorted({str(_x) for _x in lab_} - {"a"}):
            _acc = {}; _rows = []
            for (_c, _lb, _L, _ij), _ii in _cells.items():
                if _lb != _cd:
                    continue
                _uu = _cells.get((_c, "a", _L, _ij))
                if not _uu:
                    continue
                _o = _own[_c] & (_BSZ3 > 0)
                if _o.sum() < 10:
                    continue
                _HT = [np.expm1(Xu[_k].astype(np.float64)) for _k in _ii]
                _HU = [np.expm1(Xu[_k].astype(np.float64)) for _k in _uu]
                _a = _acc.setdefault(_L, [0.0, 0.0, 0])
                _a[0] += float(np.mean([(_h[_o] > 0).mean() for _h in _HT]))
                _a[1] += float(np.mean([(_h[_o] > 0).mean() for _h in _HU])); _a[2] += 1
                for _hT in _HT:
                    for _hU in _HU:
                        _s = _o & (_hU > 0)
                        if _s.sum() >= 5:
                            _sz = _BSZ3[_s]
                            _rows.append(np.column_stack([(_sz - _sz.mean()) / 100.0, (_hT[_s] > 0).astype(float)]))
            _Ls = sorted(_L for _L, _v in _acc.items() if _v[2] >= 3 and _v[1] > 0)
            if not _Ls:
                continue
            _b = None
            if _rows:
                _A = np.concatenate(_rows); _allrows.append(_A)
                if len(_A) >= 300 and 0.02 < float(_A[:, 1].mean()) < 0.995:
                    _b = _slope(_A)
            _tc[_cd] = {"t": [float(_L) for _L in _Ls],
                        "r": [float(min(_acc[_L][0] / _acc[_L][1], 1.0)) for _L in _Ls], "b": _b}
        _bp = _slope(np.concatenate(_allrows)) if _allrows else 0.0
        for _cd in _tc:
            if _tc[_cd]["b"] is None:
                _tc[_cd]["b"] = float(_bp)
        out["treat_conv"] = _tc
    except Exception:
        pass
    # PULL-UP: the rule and the NOC1 baseline it leaves in every emission rate (see kit.DYE).
    try:
        import json as _js2
        _loci = sorted(_js2.load(open(ctx.DATA / "meta_set.json"))["locus_to_idx"].items(), key=lambda kv: kv[1])
        _loci = [k_ for k_, _ in _loci]
        _BSZ5 = ctx.build_bin_size()
        _nbp, _ndp = pull_neighbours(ctx.BIN_LOCUS, _loci, _BSZ5)
        _bidx = {(int(ctx.BIN_LOCUS[j]), int(round(float(ctx.BIN_ALLELE[j]) * 10))): j for j in range(ctx.N_FLAT)}
        _pt = _pull_table(ctx, {str(names[i]) for i in ss}, _nbp, _ndp, _bidx, _loci)
        _Xp = np.expm1(np.load(ctx.DATA / "Xflat_train.npy")[ss].astype(np.float64))
        # Its own name: _own is the per-donor bin mask the blocks around it read, and reusing the name here once
        # overwrote it for everything after this block (amp_step_var came out all zero).
        _ownp = np.zeros(_Xp.shape, bool)
        for _r, _i in enumerate(ss):
            for _kk in key[int(np.argmax(y[_i]))]:
                _j = _bidx.get(_kk)
                if _j is not None:
                    _ownp[_r, _j] = True
        out["pull_p"] = _pt["p"]
        out["pull_bar"] = _pull_bar(_Xp, _ownp, _nbp, _ndp, _pt["p"]).tolist()
    except Exception:
        pass
    # SHOULDER: a peak within one repeat of an allele peak is thinned by the analysis. Neither the bp
    # distance nor the peak's own height decides it - survivors beside a tall allele are no taller than
    # far noise - so it is a kept FRACTION. For noise it falls with the neighbour's height; for an n-1
    # stutter that happens to sit beside another allele it is flat, whatever that neighbour is.
    try:
        _idx = {(int(ctx.BIN_LOCUS[_j]), int(round(float(ctx.BIN_ALLELE[_j]) * 10))): _j for _j in range(ctx.N_FLAT)}
        _bl2 = ctx.BIN_LOCUS.astype(int); _ba2 = np.rint(ctx.BIN_ALLELE * 10).astype(int)
        _geo = {}
        for _c in range(gg.shape[0]):
            _al = {}
            for (_Lk, _a10) in key[_c]:
                _al.setdefault(_Lk, []).append(_a10)
            _par = np.full(ctx.N_FLAT, -1); _nb = np.full(ctx.N_FLAT, -1)
            _far = np.zeros(ctx.N_FLAT, bool); _ow = np.zeros(ctx.N_FLAT, bool)
            for _j in range(ctx.N_FLAT):
                _aa = _al.get(int(_bl2[_j]))
                if not _aa:
                    continue
                _aa = np.asarray(_aa); _d = _aa - _ba2[_j]
                if (_d == 0).any():
                    _ow[_j] = True
                    continue
                if (_d == 10).any():
                    _par[_j] = _idx.get((int(_bl2[_j]), int(_ba2[_j]) + 10), -1)
                _cl = (np.abs(_d) < 10) & (_d != 0)
                if _cl.any():
                    _nb[_j] = _idx.get((int(_bl2[_j]), int(_aa[_cl][np.argmin(np.abs(_d[_cl]))])), -1)
                _far[_j] = int(np.abs(_d).min()) >= 25
            _geo[_c] = (_par, _nb, _far, _ow)
        _HB = np.array([0.0, 100.0, 300.0, 1000.0, 3000.0, 1e12]); _HP = _HB
        _nf = np.zeros(2); _nn = np.zeros((5, 2)); _hmed = [[] for _ in range(5)]
        _stb = np.zeros((5, 2)); _stn = []
        for _i in range(len(ss)):
            _par, _nb, _far, _ow = _geo[int(du[_i])]
            _h = np.expm1(Xu[_i].astype(np.float64)); _on = _h > 0
            _nf += [float(_far.sum()), float((_far & _on).sum())]
            for _j in np.where(~_ow & (_par >= 0))[0]:
                _hp = _h[_par[_j]]
                if _hp <= 0:
                    continue
                _b = int(np.searchsorted(_HP, _hp, side="right") - 1)
                if _nb[_j] < 0 or _h[_nb[_j]] <= 0:
                    _stb[_b] += [1.0, 1.0 if _on[_j] else 0.0]
                else:
                    _stn.append((_b, 1.0 if _on[_j] else 0.0))
            for _j in np.where(~_ow & (_par < 0) & (_nb >= 0))[0]:
                _H = _h[_nb[_j]]
                if _H > 0:
                    _b = int(np.searchsorted(_HB, _H, side="right") - 1)
                    _nn[_b] += [1.0, 1.0 if _on[_j] else 0.0]; _hmed[_b].append(_H)
        _fr0 = _nf[1] / max(_nf[0], 1.0)
        _ok = [_b for _b in range(5) if _nn[_b, 0] >= 200]
        out["shoulder_h"] = [float(np.median(_hmed[_b])) for _b in _ok]
        out["shoulder_noise"] = [float(_nn[_b, 1] / _nn[_b, 0] / _fr0) for _b in _ok]
        _base = _stb[:, 1] / np.maximum(_stb[:, 0], 1.0)
        _exp = sum(_base[_b] for _b, _ in _stn); _obs = sum(_o for _, _o in _stn)
        out["shoulder_art"] = float(_obs / _exp) if _exp > 0 else 1.0
    except Exception:
        pass
    # DETECTOR CEILING. The capillary has a finite dynamic range and nothing is recorded above it: on
    # NOC1 no profile carries a peak past this value, and across 1333 real mixtures the tallest peak
    # anywhere is 29069 RFU with not one profile above 30000. The generator had eight, up to 44436,
    # and those inflated parents were what pushed their own stutter past anything real ever shows.
    try:
        _mx = [float(np.expm1(tok[i, np.where(mk[i])[0], 2].astype(np.float64)).max())
               for i in ss if mk[i].any()]
        out["sat_rfu"] = float(np.max(_mx)) if _mx else 0.0
    except Exception:
        pass
    # Artefacts beyond the offset table's reach, per profile. Verified to be a per-profile constant:
    # across a 20000x range of total RFU the count reads 39/37/34/32/35, log-log slope -0.025.
    out["noise_n"] = float(np.mean(far_cnt)) if far_cnt else 23.0
    # Per-locus efficiency splits into a DONOR part and a RUN part. Two untreated runs of the same
    # donor correlate 0.672 on the locus pattern, so the donor half is real and must be kept; the
    # remaining 35% of the variance belongs to the amplification. A real mixture is ONE amplification,
    # so its contributors share that half - the overlay gives each of them their own source profile
    # and therefore k independent runs. Returned as the donor mean pattern plus the run sd.
    try:
        _pl = {}
        for j5, i in enumerate(ss):
            if str(lab_[j5]) != "a":
                continue
            c = int(y[i].argmax())
            hm5 = {(int(round(float(tok[i, q, 0]))), int(round(float(tok[i, q, 1]) * 10))):
                   float(np.expm1(tok[i, q, 2])) for q in np.where(mk[i])[0]}
            byl = {}
            for a5 in key[c]:
                byl.setdefault(a5[0], []).append(a5)
            v5 = np.array([sum(hm5.get(z, 0.0) for z in byl.get(L, [])) for L in range(int(ctx.BIN_LOCUS.max()) + 1)])
            if (v5 > 0).sum() >= 20:
                lv = np.log(np.where(v5 > 0, v5, np.nan))
                _pl.setdefault(c, []).append(lv - np.nanmean(lv))
        mean_p = {c: np.nanmean(np.array(v), 0) for c, v in _pl.items() if len(v) >= 4}
        dv = []
        for c, vs in _pl.items():
            if c not in mean_p:
                continue
            for x5 in vs:
                m5 = np.isfinite(x5) & np.isfinite(mean_p[c])
                if m5.sum() >= 18:
                    dv.append(float(np.nanstd(x5[m5] - mean_p[c][m5])))
        if mean_p and dv:
            out["locus_donor"] = {int(c): np.nan_to_num(v).tolist() for c, v in mean_p.items()}
            out["locus_run_sd"] = float(np.median(dv))
    except Exception:
        pass
    # A per-SAMPLE dropout offset. Loss is not an independent coin per allele: at the templates where
    # anything is lost at all, the spread of the per-profile rate runs 1.5-2.6x wider than binomial,
    # so each run carries its own tendency to lose alleles on top of the per-allele law. Measured on
    # NOC1 untreated by comparing the observed spread of the profile rate against sqrt(p(1-p)/n).
    try:
        _tpl = np.load(ctx.DATA / "meta_template_train.npy").astype(np.float64)[ss]
        _gg2 = {}
        for j3, i in enumerate(ss):
            if str(lab_[j3]) != "a" or not (_tpl[j3] > 0):
                continue
            c = int(y[i].argmax())
            ob = {(int(round(float(tok[i, q, 0]))), int(round(float(tok[i, q, 1]) * 10)))
                  for q in np.where(mk[i])[0]}
            n3 = len(key[c])
            if n3:
                _gg2.setdefault(round(float(_tpl[j3]), 4), []).append((1.0 - len(key[c] & ob) / n3, n3))
        ex = []
        for v3 in _gg2.values():
            a3 = np.array(v3, float)
            if len(a3) < 40:
                continue
            p3 = a3[:, 0].mean()
            if p3 < 0.01 or p3 > 0.5:
                continue
            vb = p3 * (1 - p3) / a3[:, 1].mean()
            if a3[:, 0].var() > vb:
                ex.append(np.sqrt(a3[:, 0].var() - vb) / max(p3, 1e-6))   # sd tuong doi
        if ex:
            out["drop_run_cv"] = float(np.median(ex))
    except Exception:
        pass
    # The floor is TWO populations and they are counted separately. A peak at a position no panel
    # donor carries has no subject: feas_filter removes it, and no forensic law about alleles applies
    # to it, so it needs only to be present in the right number. A peak at a carried position is the
    # opposite - it survives into the model and reads as evidence for whoever carries that allele.
    # Both halves are the same height (9 RFU each) so one distribution generates them, but their
    # counts differ, 4.8 carried against 18.0 not, and a single placement histogram gets the split
    # wrong: artefact_w puts 27.6% of its mass on carried bins and so emits 6.4 there, a third too
    # many in exactly the half that reaches the model.
    out["noise_in"] = float(np.mean(far_in)) if far_in else 4.8
    # Dropout takes two variables: height and fragment size. At fixed height a long fragment is lost
    # 1.4-1.6x more often, measured after removing the size trend that height already carries.
    try:
        _ds = float(out.get("drop_s", 1.50))
        rr = []
        for i in ss:
            c = int(y[i].argmax())
            # Same copy proxy as drop_h/drop_p above. Leaving this one in RFU while the generator
            # scales it by the run's injection gain mixes two conventions in one law.
            _m7 = pat.search(str(names[i]))
            _g7 = ((float(_m7.group(1)) / _INJ_REF) ** _INJ_BETA) if _m7 else 1.0
            hm7 = {(int(round(float(tok[i, q, 0]))), int(round(float(tok[i, q, 1]) * 10))):
                   float(np.expm1(tok[i, q, 2])) / _g7 for q in np.where(mk[i])[0]}
            ob7 = [(ctx.SZ.get(k7, 0.0), hm7[k7]) for k7 in key[c]
                   if k7 in hm7 and ctx.SZ.get(k7, 0.0) > 0 and hm7[k7] > 0]
            if len(ob7) < 10:
                continue
            xy7 = np.array(ob7, float)
            bf = np.polyfit(xy7[:, 0], np.log(xy7[:, 1]), 1)
            for k7 in key[c]:
                z7 = ctx.SZ.get(k7, 0.0)
                if z7 > 0:
                    rr.append((float(np.exp(np.polyval(bf, z7))), z7, 1.0 if k7 in hm7 else 0.0))
        AA = np.array(rr, float)
        pts = []
        for a7, b7 in ((0, 150), (150, 190), (190, 230), (230, 270), (270, 310), (310, 1e9)):
            m7 = (AA[:, 1] >= a7) & (AA[:, 1] < b7)
            if m7.sum() < 500:
                continue
            h7 = AA[m7, 0]; ob = AA[m7, 2].mean()
            lo7, hi7 = 1.0, 400.0
            for _ in range(60):
                md = 0.5 * (lo7 + hi7)
                if (1.0 - 1.0 / (1.0 + (np.maximum(h7, 1e-9) / md) ** _ds)).mean() > ob:
                    lo7 = md
                else:
                    hi7 = md
            pts.append((0.5 * (a7 + min(b7, 350)), 0.5 * (lo7 + hi7)))
        if len(pts) >= 4:
            kk7, bb7 = np.polyfit([q[0] for q in pts], np.log([q[1] for q in pts]), 1)
            out["drop_h50_bp"] = [float(np.exp(bb7)), float(kk7)]
    except Exception:
        pass
    # Height-dependent gamma shape, from the dispersion of heterozygote pairs. CV is ~0.17 for tall
    # peaks and more than double that for faint ones, and a constant-CV jitter cannot reproduce the
    # dropout that follows from it.
    hb = np.array(hb_pairs) if hb_pairs else np.zeros((0, 2))
    jh, js = [0.0], [6.9]
    for lo, hi, mid in ((0, 30, 15.0), (30, 80, 55.0), (80, 200, 140.0),
                        (200, 600, 400.0), (600, 1e12, 1500.0)):
        q_ = (hb[:, 0] >= lo) & (hb[:, 0] < hi) if len(hb) else np.zeros(0, bool)
        if q_.sum() >= 100:
            cv = float(np.log(hb[q_, 1]).std() / np.sqrt(2))
            jh.append(mid); js.append(float(1.0 / max(cv, 1e-3) ** 2))
    if len(jh) > 1:
        jh.append(1e9); js.append(js[-1]); js[0] = js[1]
        out["jit_h"] = jh; out["jit_shape"] = js
    # SPREAD OF A STEP DOWN, beyond what the jitter increment charges. A source taken from a richer rung carries
    # that rung's scatter; the lower rung is a noisier lottery (tpl_shrink is the total change). The generator's
    # jitter increment already adds the part that follows height, so what is stored is the rest: on same-extract
    # adjacent-rung pairs, the rise in the variance of log height around the size trend on alleles present at both
    # rungs, minus the log variance the generator's jitter increment adds for that same step (heights scaled to the
    # lower profile's total, both read at the reference injection as the generator reads them). Cumulative from the
    # top of the ladder, like the tilt, so a step of any length is V(t) - V(L).
    try:
        _jh6 = np.asarray(out.get("jit_h", [0.0, 1e9]), float); _js6 = np.asarray(out.get("jit_shape", [6.9, 6.9]), float)
        _gb6 = float((out.get("rfu_coef") or [0.976, 0.948])[1]); _ir6 = float(out.get("inj_ref", 15.0))

        def _trig(k):
            k = np.asarray(k, float).copy(); acc = np.zeros_like(k)
            for _ in range(8):
                sm = k < 8.0
                acc = acc + np.where(sm, 1.0 / k ** 2, 0.0); k = np.where(sm, k + 1.0, k)
            return acc + 1.0 / k + 1.0 / (2.0 * k ** 2) + 1.0 / (6.0 * k ** 3) - 1.0 / (30.0 * k ** 5)
        _BSZ6 = ctx.build_bin_size()
        _dvs, _dvsi = {}, {}
        for (_c, _x1, _x2, _x3), _dd in _lad.items():
            _o = _own[_c] & (_BSZ6 > 0)
            _g6 = (float(_x3) / _ir6) ** _gb6 if _x3 else 1.0
            for _a in range(len(_LV) - 1):
                _lo, _hi = _LV[_a], _LV[_a + 1]
                if _lo not in _dd or _hi not in _dd:
                    continue
                _zl = np.expm1(Xu[_dd[_lo]].astype(np.float64)); _zh = np.expm1(Xu[_dd[_hi]].astype(np.float64))
                _s = _o & (_zl > 0) & (_zh > 0)
                if _s.sum() < 12 or float(_BSZ6[_s].std()) < 1.0 or _zh[_o].sum() <= 0:
                    continue
                _ll = np.log(_zl[_s]); _lh6 = np.log(_zh[_s])
                _rl = _ll - np.polyval(np.polyfit(_BSZ6[_s], _ll, 1), _BSZ6[_s])
                _rh = _lh6 - np.polyval(np.polyfit(_BSZ6[_s], _lh6, 1), _BSZ6[_s])
                _e6 = _zh[_s] * (_zl[_o].sum() / _zh[_o].sum())
                _cv2 = np.maximum(1.0 / np.interp(_e6 / _g6, _jh6, _js6) - 1.0 / np.interp(_zh[_s] / _g6, _jh6, _js6), 0.0)
                _jv = np.where(_cv2 > 1e-9, _trig(1.0 / np.maximum(_cv2, 1e-9)), 0.0)
                _dvs.setdefault(_lo, []).append(float(_rl.var() - _rh.var() - _jv.mean()))
                _dvsi.setdefault((float(_x3), _lo), []).append(_dvs[_lo][-1])
        _V6, _acc6 = {_LV[-1]: 0.0}, 0.0
        for _a in range(len(_LV) - 2, -1, -1):
            _v6 = _dvs.get(_LV[_a], [])
            _acc6 += max(float(np.mean(_v6)), 0.0) if len(_v6) >= 15 else 0.0
            _V6[_LV[_a]] = _acc6
        out["amp_step_var"] = [float(_V6[_x]) for _x in _LV]
        # ...and per INJECTION, as PHI is: pooled, the hold-out was right on average but -0.021 / +0.017 / +0.020 in
        # sd log height at 5 / 15 / 25 s. A rung with fewer than 15 pairs at an injection takes the pooled step.
        _vinj = {}
        for _ij in sorted({_k[0] for _k in _dvsi}):
            _Vj, _aj = {_LV[-1]: 0.0}, 0.0
            for _a in range(len(_LV) - 2, -1, -1):
                _vv = _dvsi.get((_ij, _LV[_a]), [])
                if len(_vv) < 15:
                    _vv = _dvs.get(_LV[_a], [])
                _aj += max(float(np.mean(_vv)), 0.0) if len(_vv) >= 15 else 0.0
                _Vj[_LV[_a]] = _aj
            _vinj[str(_ij)] = [float(_Vj[_x]) for _x in _LV]
        out["amp_step_var_inj"] = _vinj
    except Exception:
        pass
    # HOW SPREAD OUT the per-allele chances of surviving are. If every allele of a profile had the same chance r,
    # one run's retention would vary by r(1-r)/n_eff; if a faint allele is nearly certain to go and a tall one
    # nearly certain to stay, the variance is mean p(1-p)/n_eff, which is smaller. The ratio of the two is the
    # dispersion, and it is what a REDRAWN lottery has to reproduce.
    # This set holds one PCR product per cell, so the only repeated measurement of a specification is that product
    # injected for different numbers of seconds - and seconds move retention systematically (0.636 / 0.732 / 0.761
    # at 0.0078 ng), which would pose as lottery. So each profile is put against the mean of its own
    # (template, treatment, seconds) cell and only the residual is paired. What remains is the presence lottery of
    # the capillary: 0.29 / 0.30 / 0.31 / 0.41 / 0.36 / 0.40 from 0.0078 to 0.25 ng, pooled 0.32, on 656-1026
    # pairs per rung. The PCR lottery itself cannot be measured here - no two cells hold the same specification -
    # so it is not invented.
    try:
        _dx = _re.compile(r"RD\d+-\d+-(\d+)d(\d+)")
        _rw = []
        for _i in range(len(ss)):
            _m = _dx.search(str(names[ss[_i]]))
            _o = _own[int(du[_i])]
            if not _m or _o.sum() < 12 or float(tpl_u[_i]) <= 0:
                continue
            _rw.append((int(du[_i]), str(lab_[_i]), round(float(tpl_u[_i]), 4), int(_m.group(2)),
                        float(inj_u[_i]), float((np.expm1(Xu[_i].astype(np.float64))[_o] > 0).mean()),
                        int(_o.sum())))
        _cell = {}
        for _q in _rw:
            _cell.setdefault((_q[2], _q[1], _q[4]), []).append(_q[5])
        _cmu = {_k: float(np.mean(_v)) for _k, _v in _cell.items() if len(_v) >= 5}
        _grp = {}
        for _q in _rw:
            _k = (_q[2], _q[1], _q[4])
            if _k in _cmu:
                _grp.setdefault((_q[0], _q[1], _q[2], _q[3]), []).append((_q[5] - _cmu[_k], _q[5], _q[6], _q[4]))
        _rh = float(out.get("amp_share_r", 0.246) or 0.0)
        _vo, _vb = 0.0, 0.0
        for _v in _grp.values():
            for _a in range(len(_v)):
                for _b in range(_a + 1, len(_v)):
                    if _v[_a][3] == _v[_b][3]:
                        continue
                    _rr = 0.5 * (_v[_a][1] + _v[_b][1]); _nn = 0.5 * (_v[_a][2] + _v[_b][2])
                    _vo += (_v[_a][0] - _v[_b][0]) ** 2 / 2.0
                    _vb += _rr * (1.0 - _rr) / (_nn / (1.0 + _rh))
        if _vb > 0:
            out["drop_disp"] = float(min(max(_vo / _vb, 0.05), 1.0))
    except Exception:
        pass
    # INJECTION STEPS. One PCR product is injected for 5, 15 and 25 s, so same-product pairs differ only by the
    # capillary: the height of an allele at the shorter injection is the longer one times a single factor, with
    # the capillary's own scatter on top (25->15 s x0.577 sd 0.080, 15->5 s x0.350 sd 0.152, 25->5 s x0.202 sd
    # 0.155; the 25->15 step predicts presence at 15 s right for 97.1% of alleles). Only DOWNWARD steps exist:
    # what fell under the threshold at a short injection is not in the data to scale up.
    try:
        _ix2 = _re.compile(r"RD\d+-\d+-(\d+)d(\d+)")
        _grp2 = {}
        for _i in range(len(ss)):
            _m = _ix2.search(str(names[ss[_i]]))
            if _m:
                _grp2.setdefault((int(du[_i]), int(_m.group(2)), str(lab_[_i]), round(float(tpl_u[_i]), 4)),
                                 {})[float(inj_u[_i])] = _i
        _istep = {}
        for _hi, _lo in ((25.0, 15.0), (15.0, 5.0), (25.0, 5.0)):
            _med, _res = [], []
            for (_c2, _e2, _l2, _t2), _dd in _grp2.items():
                if _hi in _dd and _lo in _dd:
                    _o2 = _own[_c2]
                    _a2 = np.expm1(Xu[_dd[_hi]].astype(np.float64))[_o2]
                    _b2 = np.expm1(Xu[_dd[_lo]].astype(np.float64))[_o2]
                    _ok2 = (_a2 > 0) & (_b2 > 0)
                    if _ok2.sum() >= 5:
                        _r2 = np.log(_b2[_ok2] / _a2[_ok2])
                        _med.append(float(np.mean(_r2))); _res.append(_r2 - float(np.mean(_r2)))
            if len(_med) >= 30:
                _f2 = float(np.median(_med))
                # SPLIT, because the capillary is ONE event per injection and the scatter is not all per allele.
                # Half of it moves a whole profile (25->15 s: 0.054 of 0.083; 15->5 s: 0.119 of 0.154), and a
                # profile-wide factor cannot change heterozygote balance. Charging the pooled 0.083 to every allele
                # inflated the per-allele jitter 1.3-1.5x and cost the twin its balance at bright templates
                # (0.804 against real's 0.838, while the sources it copies sit at 0.883).
                _istep["%g|%g" % (_hi, _lo)] = [float(np.exp(_f2)), float(np.std(_med)),
                                                float(np.std(np.concatenate(_res)))]
        out["inj_step"] = _istep
    except Exception:
        pass
    # Allele dropout against RFU per allele. This is the one law that needs the reference set to span
    # SEVERAL template levels: 45 profiles at one dilution never reach the faint end and the curve
    # cannot be traced (0/12 subsamples recovered the 20-60 RFU point).
    dr = np.array(drop_rec) if drop_rec else np.zeros((0, 2))
    dh, dp = [0.0], [1.0]
    for lo, hi, mid in ((10, 20, 15.0), (20, 60, 40.0), (60, 150, 100.0),
                        (150, 400, 250.0), (400, 1e12, 500.0)):
        q_ = (dr[:, 0] >= lo) & (dr[:, 0] < hi) if len(dr) else np.zeros(0, bool)
        if q_.sum() >= 30:
            dh.append(mid); dp.append(float(np.median(dr[q_, 1])))
    if len(dh) > 2:
        dh.append(1e9); dp.append(0.0)
        out["drop_h"] = dh; out["drop_p"] = dp
        # Invert the profile-level law into a per-allele one against the observed within-profile
        # height spread, so the generator can apply dropout where it actually falls.
        try:
            ker = []
            for i in ss:
                c = int(y[i].argmax()); v = np.where(mk[i])[0]
                kk2 = [(int(round(float(tok[i, q, 0]))), int(round(float(tok[i, q, 1]) * 10)))
                       for q in v]
                hh2 = np.expm1(tok[i, v, 2].astype(np.float64))
                if hh2.sum() < 15000:
                    continue
                ow = np.array([k_ in key[c] for k_ in kk2])
                if ow.sum() >= 30:
                    t2 = hh2[ow]; ker.append(t2 / t2.mean())
            ker = np.concatenate(ker)
            ker = ker[np.random.default_rng(0).choice(len(ker), min(20000, len(ker)), replace=False)]
            Mm = np.array(dh[1:-1], float); Dd = np.array(dp[1:-1], float)
            best, bl = (27.5, 1.48), 1e9
            for h50 in np.exp(np.linspace(np.log(5), np.log(200), 40)):
                for sp_ in np.linspace(0.6, 3.0, 25):
                    pr = np.array([np.mean(1.0 / (1.0 + (m * ker / h50) ** sp_)) for m in Mm])
                    l = float(((pr - Dd) ** 2).sum())
                    if l < bl:
                        bl, best = l, (float(h50), float(sp_))
            out["drop_h50"], out["drop_s"] = best
        except Exception:
            pass
    # DONOR SHAPE. The donor's own per-allele pattern, estimated from EVERY NOC1 profile rather than
    # the handful in the clean pool. The clean pool holds 5-12 runs per donor, so a mean taken there
    # still carries 0.376/sqrt(8) = 0.133 of one amplification's realisation - more than the person
    # term itself - and the generator was copying that into every mixture. Over all ~54 amplifications
    # the standard error falls to 0.051. Dosage is taken out (a homozygous allele sits log 2 above a
    # heterozygous one, and that is genotype, not pattern) and the condition effect is removed per
    # (treatment, template band) because degradation bends the pattern the same way for everyone.
    try:
        _tb = [0.0, 0.02, 0.06, 0.15, 1e9]
        _pv = {}
        for i in ss:
            m_ = pat.search(str(names[i]))
            if m_ is None or not (ng[i] > 0):
                continue
            c = int(y[i].argmax()); v = np.where(mk[i])[0]
            _h = np.expm1(tok[i, v, 2].astype(np.float64))
            _k = [(int(round(float(tok[i, p_, 0]))), int(round(float(tok[i, p_, 1]) * 10))) for p_ in v]
            hm_ = dict(zip(_k, _h))
            w_ = {}
            for a_ in key[c]:
                h_ = hm_.get(a_, 0.0)
                if h_ <= 0:
                    continue
                j_ = ctx._BININDEX.get((a_[0], round(a_[1] / 10.0, 1)), -1)
                dz = float(ctx.DONOR_DOSAGE[c][j_]) if (ctx.DONOR_DOSAGE is not None and j_ >= 0) else 1.0
                w_[a_] = np.log(h_) - np.log(max(dz, 1e-9))
            if len(w_) < 15:
                continue
            mu_ = float(np.mean(list(w_.values())))
            # Keyed by TREATMENT and template band. Injection only moves the profile's level, which
            # the per-profile centring above already removes; treatment bends the pattern itself.
            _mt = _re.search(r"RD14-0003-\d+d\d([A-Za-z0-9\-]*?)-[0-9.]+" + kit.TAG, str(names[i]))
            _tt = (_mt.group(1).strip("-") or "a") if _mt else "a"
            _tr = (_tt, int(np.searchsorted(_tb, float(ng[i]), "right")) - 1)
            _pv.setdefault((c, _tr), []).append({a_: x - mu_ for a_, x in w_.items()})
        _cm = {}
        for (c, tr_), lst in _pv.items():
            for d_ in lst:
                for a_, x in d_.items():
                    _cm.setdefault((tr_, a_), {}).setdefault(c, []).append(x)
        CM2 = {q: float(np.mean([np.mean(z) for z in dd.values()]))
               for q, dd in _cm.items() if len(dd) >= 3}
        _dv = {}
        for (c, tr_), lst in _pv.items():
            for d_ in lst:
                for a_, x in d_.items():
                    if (tr_, a_) in CM2:
                        _dv.setdefault(c, {}).setdefault(a_, []).append(x - CM2[(tr_, a_)])
        dev = {c: {a_: float(np.mean(z)) for a_, z in dd.items() if len(z) >= 8}
               for c, dd in _dv.items()}
        _all = np.array([x for dd in dev.values() for x in dd.values()])
        if len(_all) > 200 and _all.std() > 1e-9:
            out["donor_dev"] = {int(c): {f"{a_[0]}_{a_[1]}": v / float(_all.std())
                                         for a_, v in dd.items()} for c, dd in dev.items()}
            out["donor_dev_n"] = int(len(_all))
            # In LOG units, shrunk by the reliability of the donor's own average, for a generator that
            # subtracts a source profile's residual rather than replacing the profile. Replacing it
            # was tried and collapses the twin (ID 0.781, count 0.620 against real's 0.948/0.894),
            # because the population mean carries none of that profile's own template and condition.
            # Subtracting keeps every systematic term the source already has and changes only the one
            # quantity a fresh amplification redraws.
            _rel = 0.52
            out["donor_dev_log"] = {int(c): {f"{a_[0]}_{a_[1]}": float(_rel * v)
                                             for a_, v in dd.items()} for c, dd in dev.items()}
            # A(condition, allele): what every donor shows at that treatment and template band.
            out["cond_allele_mean"] = {f"{q[0][0]}|{q[0][1]}|{q[1][0]}_{q[1][1]}": float(v)
                                       for q, v in CM2.items()}
    except Exception:
        pass
    out["artefact_w"] = art_w / max(art_w.sum(), 1.0)
    out["sr_median"] = float(np.median(sr)) if sr else None
    # Persist so a run can be reproduced and a reviewer can read what was inferred from what.
    try:
        import json as _j
        with open(ctx.DATA / "generator_calib.json", "w") as fh:
            _j.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                     for k, v in out.items()
                     if k in ("at", "noise_n", "rfu_coef", "rfu_sd", "inj_levels", "sr_median",
                              "jit_h", "jit_shape", "drop_h", "drop_p", "art_table",
                              "art_surv_h", "art_surv_p", "art_noise_bg")}, fh, indent=1)
    except Exception:
        pass
    ctx.cache = out
    return out
