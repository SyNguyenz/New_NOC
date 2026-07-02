"""
measure_id_ambiguity.py — is the high-NOC ID weakness an INFORMATION limit of the
mixture (allele-sharing degeneracy against the full 45-donor panel), or model failure?

measure_noc5_ceiling.py only checked private-vs-CO-CONTRIBUTORS (=> detectability).
Unique IDENTIFICATION must distinguish the true donor from ALL 45. This script does the
presence-based set-identifiability test:

For each TRUE contributor X in a closed test sample:
  X_present = { (locus,allele) of X's reference genotype that are PRESENT in the peaks }
  substitutes = { decoy donor D not in the true set : geno(D) ⊇ X_present }
     i.e. a NON-contributor whose genotype already explains EVERY peak X explains.
  If substitutes != {} -> by presence alone the data cannot prefer X over D -> the
     mixture is AMBIGUOUS for X (would need peak HEIGHT/quantity to break the tie).
  If substitutes == {} -> X is uniquely required by presence (model failure if missed).

This is an UPPER BOUND on ambiguity: a height-aware model can rule out some decoys whose
*other* alleles would add peaks not observed. We also report the height-tightened count
(decoys that introduce NO extra allele beyond what is already observed at X's loci).

Genotypes from RAW data_raw (NOT the 91%-coverage synth CSV).
Usage:  python measure_id_ambiguity.py
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from measure_noc5_ceiling import load_raw_genotypes, observed_set, KNOWN

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    geno = load_raw_genotypes()                      # geno[donor_int][locus_idx] = {allele_key}
    # flat (locus, allele) set per donor, + per-locus dict for the height-tightened test
    flat = {d: {(L, a) for L, alleles in g.items() for a in alleles} for d, g in geno.items()}
    all_donors = list(geno.keys())

    tok = np.load(DATA / "tokens_test.npy"); mask = np.load(DATA / "mask_test.npy").astype(bool)
    y = np.load(DATA / "y_test_set.npy"); noc = np.load(DATA / "noc_test.npy")
    N = len(y)

    # per NOC: tallies of contributor identifiability
    per = {k: {"n": 0, "ambiguous": 0, "ambiguous_h": 0, "sub_counts": [],
               "best_cov": [], "xpres_sz": []} for k in range(1, 6)}
    sample_unique = {k: [0, 0] for k in range(1, 6)}   # [fully-unique samples, total]

    for i in range(N):
        k = int(noc[i])
        if k < 1 or k > 5:
            continue
        obs, pla, _plh = observed_set(tok[i], mask[i])      # obs = {(L, allele_key)}
        obs_by_locus: dict[int, set] = {}
        for (L, a) in obs:
            obs_by_locus.setdefault(L, set()).add(a)
        true_ints = [KNOWN[c] for c in np.where(y[i] == 1)[0]]
        decoys = [d for d in all_donors if d not in true_ints]

        all_unique = True
        for X in true_ints:
            gX = flat.get(X)
            if not gX:
                continue
            X_present = {(L, a) for (L, a) in gX if (L, a) in obs}
            if not X_present:
                continue
            # presence-based substitutes: decoy explains every peak X explains
            subs = [D for D in decoys if X_present <= flat[D]]
            # height-tightened: decoy introduces NO allele absent from the observed peaks
            #   at the loci it would touch (else it would predict an unseen peak)
            subs_h = [D for D in subs
                      if all(a in obs_by_locus.get(L, set()) for (L, a) in flat[D])]
            # margin: how much of X_present can the BEST decoy cover? (1.0 = knife-edge)
            best_cov = max((len(X_present & flat[D]) / len(X_present) for D in decoys), default=0.0)
            per[k]["best_cov"].append(best_cov)
            per[k]["xpres_sz"].append(len(X_present))
            per[k]["n"] += 1
            per[k]["sub_counts"].append(len(subs))
            if subs:
                per[k]["ambiguous"] += 1
                all_unique = False
            if subs_h:
                per[k]["ambiguous_h"] += 1
        sample_unique[k][1] += 1
        if all_unique:
            sample_unique[k][0] += 1

    print("\n== Presence-based ID identifiability vs the FULL 45-donor panel (raw genotypes) ==")
    print("  ambiguous = a NON-contributor donor explains every PRESENT allele of the true contributor")
    print(f"  {'NOC':>3} {'n_contrib':>9} {'AMBIG(pres)':>12} {'AMBIG(+height)':>14} {'mean#decoys':>12} {'uniq-set samples':>18}")
    for k in range(1, 6):
        p = per[k]
        if p["n"] == 0:
            continue
        amb = 100 * p["ambiguous"] / p["n"]
        ambh = 100 * p["ambiguous_h"] / p["n"]
        md = float(np.mean(p["sub_counts"]))
        su, st = sample_unique[k]
        print(f"  {k:>3} {p['n']:>9} {f'{amb:5.1f}%':>12} {f'{ambh:5.1f}%':>14} "
              f"{md:>12.2f} {f'{su}/{st} ({100*su/max(st,1):.0f}%)':>18}")
    print("\n== Margin (how unique, not knife-edge): best decoy's coverage of X's present alleles ==")
    print(f"  {'NOC':>3} {'median best-decoy cov':>22} {'95th pct':>10} {'median #present alleles':>24}")
    for k in range(1, 6):
        p = per[k]
        if not p["best_cov"]:
            continue
        bc = np.array(p["best_cov"])
        print(f"  {k:>3} {f'{np.median(bc):.2f}':>22} {f'{np.percentile(bc,95):.2f}':>10} "
              f"{int(np.median(p['xpres_sz'])):>24}")

    print("\n  Reading: AMBIG(+height) = decoys that survive the no-extra-peak constraint — the tighter,")
    print("  more honest info-limit estimate. High AMBIG(+height) at NOC5 => the MIXTURE DATA itself is")
    print("  degenerate for unique ID (height/quantity is the only tiebreak) = an information limit, not")
    print("  pure model failure. Low => the data determines the set and a miss is model-side.")


if __name__ == "__main__":
    main()
