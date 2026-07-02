"""
make_gf_control.py — locus-ablation CONTROL to disentangle the cross-kit OOD drop.

The IDPlus OOD test confounds TWO shifts: (a) kit/instrument covariate shift, and
(b) INFORMATION LOSS — IDPlus carries only 16 of GlobalFiler's 24 markers (drops SE33
+ 7 others). This script rebuilds the GF in-distribution REAL test (same kit, same
samples) at two locus resolutions so eval_crossfolder can A/B them:
  gf_full24  : all 24 GF loci (in-distribution reference)
  gf16       : GF test MASKED to the 16 loci shared with IDPlus (kit unchanged, loci cut)

If gf16's N4/N5 oracle collapses toward the IDPlus OOD level, the high-NOC failure is
LOCUS-LOSS (fewer private alleles; Green&Mortera §8), not kit covariate shift.

Usage:
  python make_gf_control.py --shared_from idplus28_rd14
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CROSS = ROOT / "data_cross"
from features.enrich import enrich_tokens, add_size_fields  # noqa: E402


def write_tag(tag, tokens, mask, size, y, noc, cond, shared, dropped):
    en9 = enrich_tokens(tokens, mask)
    en11 = add_size_fields(en9, mask, size)
    np.save(CROSS / f"tokens8_{tag}.npy", en9[:, :, :8].astype(np.float32))
    np.save(CROSS / f"tokens9_{tag}.npy", en9.astype(np.float32))
    np.save(CROSS / f"tokens11_{tag}.npy", en11.astype(np.float32))
    np.save(CROSS / f"mask_{tag}.npy", mask)
    np.save(CROSS / f"y_{tag}_set.npy", y.astype(np.float32))
    np.save(CROSS / f"noc_{tag}.npy", noc.astype(np.int64))
    np.save(CROSS / f"size_{tag}.npy", size.astype(np.float32))
    np.save(CROSS / f"is_closed_{tag}.npy", np.ones(len(y), bool))   # GF real test = closed
    np.save(CROSS / f"condition_{tag}.npy", cond.astype(np.int32))
    noc_dist = {int(k): int(v) for k, v in zip(*np.unique(noc, return_counts=True))}
    json.dump({"tag": tag, "folder": "3500_GF29cycles", "panel": "rd14-gf",
               "id_measurable": True, "n_samples": int(len(y)), "n_closed": int(len(y)),
               "noc_dist": noc_dist, "closed_noc_dist": noc_dist,
               "shared_loci": shared, "dropped_loci": dropped, "has_size": True},
              open(CROSS / f"summary_{tag}.json", "w"), indent=2)
    print(f"  wrote {tag}: n={len(y)} noc={noc_dist} loci_kept={len(shared) if shared else 24}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shared_from", default="idplus28_rd14",
                    help="cross tag whose summary lists the shared loci to mask down to")
    args = ap.parse_args()
    CROSS.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(DATA / "meta_set.json"))
    locus_to_idx = meta["locus_to_idx"]
    shared = json.load(open(CROSS / f"summary_{args.shared_from}.json"))["shared_loci"]
    shared_idx = np.array(sorted(locus_to_idx[m] for m in shared), dtype=np.int64)
    dropped = [m for m in meta["loci"] if m not in set(shared)]

    tokens = np.load(DATA / "tokens_test.npy").astype(np.float32)
    mask   = np.load(DATA / "mask_test.npy")
    size   = np.load(DATA / "size_test.npy").astype(np.float32) if (DATA / "size_test.npy").exists() \
             else np.zeros(mask.shape, np.float32)
    y      = np.load(DATA / "y_test_set.npy")
    noc    = np.load(DATA / "noc_test.npy")
    cond   = np.load(DATA / "condition_test.npy") if (DATA / "condition_test.npy").exists() \
             else np.full(len(y), -1, np.int32)

    print(f"GF real test: n={len(y)}  24-loci full + 16-loci masked (shared from {args.shared_from})")
    # gf_full24 — all loci
    write_tag("gf_full24", tokens, mask, size, y, noc, cond, shared=None, dropped=[])
    # gf16 — keep only peaks whose locus is in the shared set
    keep = mask & np.isin(tokens[:, :, 0].astype(np.int64), shared_idx)
    tok_m = tokens * keep[..., None]
    write_tag("gf16", tok_m, keep, size * keep, y, noc, cond, shared=shared, dropped=dropped)


if __name__ == "__main__":
    main()
