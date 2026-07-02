"""
measure_noc5_ceiling.py — re-measure the NOC high-order "ceiling" from RAW reference
genotypes (NOT the 91%-coverage synth CSV), to replace the cited-but-unverified
"76% rankable / 24% genuine dropout" prior with a reproducible, decomposed number.

Genotype source = data_raw/.../*GF29cycles/*RD14-0003 GF Known Genotypes.xlsx
(the authoritative PROVEDIt reference: donor 1..50 × 24 loci, cells "a1,a2").

For each TRUE contributor X in each closed test sample, using ONLY presence/absence
(model-independent), classify X by whether its IDENTITY is recoverable from the peaks:
  RANKABLE : X has >=1 PRIVATE allele (not in any other contributor's genotype) that is
             PRESENT in the peaks -> info is there; a miss is MODEL failure, not physics.
  DROPOUT  : X has private alleles but NONE are present -> the distinguishing peak(s)
             physically did not amplify above AT -> identity absent by presence.
  MASKED   : X has NO private allele at all (genotype fully covered by the others) ->
             allele-sharing non-identifiability; can only be inferred from peak HEIGHT.

Then a SAMPLE-level COUNT check (the NOC question, distinct from ID): is the true number
of contributors forced by allele richness alone?
  count_lb = ceil(MAC/2)   (MAC = max distinct alleles at any one locus)
  count_forced = count_lb >= noc   -> the COUNT is provable from presence (model should get it)
  else the extra contributor's EXISTENCE is only inferable from height-stacking (or not at all).

A height-stacking INDICATOR (approx, labelled) for DROPOUT/MASKED donors: at the shared
loci where X contributes, is observed height elevated vs the per-locus norm? Rigorous
attribution needs a continuous-likelihood model (EuroForMix/pgNOC); this is only a flag.

Usage:  python measure_noc5_ceiling.py
"""
from __future__ import annotations
import glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
META = json.load(open(DATA / "meta_set.json"))
KNOWN = META["known_donors"]
COL = {d: i for i, d in enumerate(KNOWN)}        # donor_int -> y column
LOCUS_TO_IDX = META["locus_to_idx"]


def allele_key(v) -> str:
    """Canonical allele string matching the token float encoding (X=-2.0, Y=-1.0)."""
    s = str(v).strip()
    if s in ("", "nan", "NaN", "None"):
        return ""
    if s in ("X",):
        return "-2.0"
    if s in ("Y",):
        return "-1.0"
    try:
        return f"{round(float(s), 1):.1f}"
    except ValueError:
        return ""


def load_raw_genotypes() -> dict[int, dict[int, set]]:
    """geno[donor_int][locus_idx] = {allele_key,...} from the RAW Known Genotypes xlsx."""
    f = glob.glob("data_raw/**/PROVEDIt_RD14-0003 GF Known Genotypes.xlsx", recursive=True)[0]
    df = pd.read_excel(f, sheet_name=0)
    loci_cols = [c for c in df.columns if c in LOCUS_TO_IDX]
    geno: dict[int, dict[int, set]] = {}
    for _, r in df.iterrows():
        try:
            d = int(r["Sample ID"])
        except (ValueError, TypeError):
            continue
        if d not in COL:                                   # only the 45 known donors
            continue
        g: dict[int, set] = {}
        for loc in loci_cols:
            cell = r[loc]
            if pd.isna(cell):
                continue
            ks = {allele_key(a) for a in str(cell).split(",")}
            ks.discard("")
            if ks:
                g[LOCUS_TO_IDX[loc]] = ks
        geno[d] = g
    cov = np.mean([len(g) for g in geno.values()])
    print(f"raw genotypes: {len(geno)} known donors | mean loci/donor with data = {cov:.1f}/24")
    return geno


def observed_set(tok, mask):
    """{(locus_idx, allele_key)} present above AT for one sample, + per-locus allele counts + heights."""
    obs = set()
    per_locus_alleles: dict[int, set] = {}
    per_locus_h: dict[int, list] = {}
    li = tok[:, 0].astype(int); al = tok[:, 1]; lh = tok[:, 2]
    for j in np.where(mask)[0]:
        L = int(li[j]); a = float(al[j])
        ak = "-2.0" if a == -2.0 else ("-1.0" if a == -1.0 else f"{round(a,1):.1f}")
        obs.add((L, ak))
        per_locus_alleles.setdefault(L, set()).add(ak)
        per_locus_h.setdefault(L, []).append(float(np.expm1(lh[j])))
    return obs, per_locus_alleles, per_locus_h


def main():
    geno = load_raw_genotypes()
    tok = np.load(DATA / "tokens_test.npy"); mask = np.load(DATA / "mask_test.npy").astype(bool)
    y = np.load(DATA / "y_test_set.npy"); noc = np.load(DATA / "noc_test.npy")
    N = len(y)

    # donor-level identity classification + sample-level count check, per NOC
    cat_counts = {k: {"RANKABLE": 0, "DROPOUT": 0, "MASKED": 0, "NO_GENO": 0} for k in range(1, 6)}
    count_forced = {k: [0, 0] for k in range(1, 6)}     # [forced, total samples]
    stack_flag = {"DROPOUT": [0, 0], "MASKED": [0, 0]}  # [height-elevated, total]  (indicative)

    for i in range(N):
        k = int(noc[i])
        if k < 1 or k > 5:
            continue
        true_cols = np.where(y[i] == 1)[0]
        true_ints = [KNOWN[c] for c in true_cols]
        obs, pla, plh = observed_set(tok[i], mask[i])

        # COUNT: forced by allele richness?  count_lb = ceil(MAC/2)
        mac = max((len(s) for s in pla.values()), default=0)
        count_forced[k][1] += 1
        if math.ceil(mac / 2) >= k:
            count_forced[k][0] += 1

        # per-locus median height (norm for the stacking indicator)
        loc_med = {L: float(np.median(hs)) for L, hs in plh.items() if hs}

        for X in true_ints:
            gX = geno.get(X, {})
            if not gX:
                cat_counts[k]["NO_GENO"] += 1
                continue
            others = [geno.get(o, {}) for o in true_ints if o != X]
            private = set()
            for L, alleles in gX.items():
                others_here = set().union(*[o.get(L, set()) for o in others]) if others else set()
                for a in alleles:
                    if a not in others_here:
                        private.add((L, a))
            if not private:
                cat = "MASKED"
            elif private & obs:
                cat = "RANKABLE"
            else:
                cat = "DROPOUT"
            cat_counts[k][cat] += 1

            # indicative height-stacking flag for non-RANKABLE donors
            if cat in ("DROPOUT", "MASKED"):
                shared_loci = [L for L in gX if L in loc_med and any((L, a) in obs for a in gX[L])]
                elevated = any(
                    max(plh[L]) > 1.4 * loc_med[L] for L in shared_loci if len(plh.get(L, [])) > 1
                )
                stack_flag[cat][1] += 1
                stack_flag[cat][0] += int(elevated)

    print("\n== ID-recoverability of each TRUE contributor (presence-based, model-independent) ==")
    print(f"  {'NOC':>3} {'n_donors':>9} {'RANKABLE':>10} {'DROPOUT':>9} {'MASKED':>8} {'NO_GENO':>8}")
    for k in range(1, 6):
        c = cat_counts[k]; tot = sum(c.values())
        if tot == 0:
            continue
        pr = lambda x: f"{x:>4d}({100*x/tot:4.1f}%)"
        print(f"  {k:>3} {tot:>9} {pr(c['RANKABLE']):>10} {pr(c['DROPOUT']):>9} "
              f"{pr(c['MASKED']):>8} {c['NO_GENO']:>8}")

    print("\n== COUNT detectability per sample (is true NOC forced by allele richness, ceil(MAC/2)>=NOC) ==")
    print(f"  {'NOC':>3} {'forced/total':>14} {'%':>6}")
    for k in range(1, 6):
        f_, t_ = count_forced[k]
        if t_:
            print(f"  {k:>3} {f'{f_}/{t_}':>14} {100*f_/t_:5.1f}%")

    print("\n== [indicative] height-stacking among non-RANKABLE donors (needs likelihood to confirm) ==")
    for cat, (e, t) in stack_flag.items():
        if t:
            print(f"  {cat:>8}: {e}/{t} donors sit at a locus with an elevated shared peak ({100*e/t:.0f}%)")

    # headline recompute of the cited prior, on NOC5 specifically
    c5 = cat_counts[5]; t5 = c5["RANKABLE"] + c5["DROPOUT"] + c5["MASKED"]
    if t5:
        print(f"\nNOC5 recomputed: RANKABLE={100*c5['RANKABLE']/t5:.0f}%  "
              f"DROPOUT={100*c5['DROPOUT']/t5:.0f}%  MASKED={100*c5['MASKED']/t5:.0f}%  "
              f"(cited prior was 76% rankable / 24% dropout — note MASKED was not a separate bucket)")


if __name__ == "__main__":
    main()
