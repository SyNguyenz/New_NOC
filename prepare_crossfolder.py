"""
prepare_crossfolder.py — Cross-folder ZERO-SHOT OOD prep.

Parse an EXTERNAL PROVEDIt kit folder's Filtered CSVs with the SAME parsing logic
as data/prepare_data_set.py, but HARMONISE loci into the FROZEN GF29 locus map
(data/meta_set.json). Emits tokens8/9/11 + mask + y(45) + noc + condition + size in
the identical layout the model consumes, so a frozen GF29 checkpoint can be evaluated
ZERO-SHOT by eval_crossfolder.py (no re-training, no re-derivation of features).

Why this is a real OOD test (the current 'real test' is NOT):
  The in-silico mixtures + real test are all built from the 3500_GF29cycles folder
  (panel RD14-0003, GlobalFiler kit, 3500 instrument). A different kit folder is a
  genuine covariate shift (kit chemistry / instrument / cycle number).

Donor panels (decides whether closed-set ID is measurable zero-shot):
  rd14  (3130_IDPlus28cycles, 3500_F6C29cycles_hlfrxn) = SAME physical donors as
        train (RD14-0003), profiled with a DIFFERENT kit/instrument. The 45-class
        donor head is meaningful -> ID + NOC measurable zero-shot.
  rd12  (3500_IDPlus29cycles, 3130_PP16HS32cycles) = DIFFERENT people (RD12-0002).
        The donor integers in the filenames collide with RD14's numbering but are
        different individuals -> y is left ZERO, id_measurable=False -> NOC count +
        reject(open) transfer only.

Locus harmonisation: a peak whose Marker is not in the GF map (e.g. PowerPlex
Penta E/D) is DROPPED; GF-only markers (SE33/Yindel/DYS391/...) are simply absent
in the external kit -> fewer tokens per sample. Identifiler-Plus loci are an exact
subset of GlobalFiler, so IDPlus folders harmonise cleanly.

Usage:
  python prepare_crossfolder.py --folder 3130_IDPlus28cycles --panel rd14 --tag idplus28_rd14
  python prepare_crossfolder.py --folder 3500_IDPlus29cycles --panel rd12 --tag idplus29_rd12
"""
from __future__ import annotations
import argparse, glob, json, os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_FILTERED = ROOT / "data_raw" / "PROVEDIt_1-5-Person CSVs Filtered"
OUT_DIR = ROOT / "data_cross"

# Reuse the EXACT parsing/encoding helpers + enrichment from the canonical pipeline.
from data.prepare_data_set import allele_to_float, parse_donors  # noqa: E402
from features.enrich import enrich_tokens, add_size_fields        # noqa: E402
from extract_phi_condition import parse_condition, COND_CATS      # noqa: E402


def build_flat_col_index(meta: dict) -> dict[tuple[str, float], int]:
    """Reproduce data/prepare_data_set.py's flat_col_index ordering from meta_set.json
    so the harmonised Xflat lines up bin-for-bin with the trained model's flat columns."""
    loci = meta["loci"]
    bins = meta["locus_bin_lists"]
    idx: dict[tuple[str, float], int] = {}
    j = 0
    for loc in loci:
        for av in bins[loc]:
            idx[(loc, float(av))] = j
            j += 1
    assert j == meta["n_flat"], f"flat_col_index size {j} != n_flat {meta['n_flat']}"
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True,
                    help="substring of the kit subfolder, e.g. 3130_IDPlus28cycles")
    ap.add_argument("--panel", required=True, choices=["rd14", "rd12"],
                    help="rd14 = same donors as train (ID measurable); rd12 = different people")
    ap.add_argument("--tag", required=True, help="output tag, e.g. idplus28_rd14")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.load(open(DATA_DIR / "meta_set.json"))
    locus_to_idx = meta["locus_to_idx"]
    KNOWN_DONORS = meta["known_donors"]
    KNOWN_SET = set(KNOWN_DONORS)
    MAX_SEQ = meta["max_seq"]
    n_flat = meta["n_flat"]
    flat_col_index = build_flat_col_index(meta)
    id_measurable = (args.panel == "rd14")

    # ── Collect external-kit CSVs (peak exports only; skip genotype tables) ──────
    pattern = str(RAW_FILTERED / f"*{args.folder}*" / "**" / "*.csv")
    files = sorted(glob.glob(pattern, recursive=True))
    files = [f for f in files if "Known Genotypes" not in f]
    print(f"folder='{args.folder}' panel={args.panel}  matched {len(files)} CSV(s)")
    if not files:
        raise SystemExit(f"No CSVs matched {pattern}")

    dfs = []
    for f in files:
        df = pd.read_csv(f, low_memory=False)
        if "Marker" not in df.columns or "Sample File" not in df.columns:
            print(f"  skip (not a peak export): {os.path.basename(f)}")
            continue
        dfs.append(df)
    raw = pd.concat(dfs, ignore_index=True)
    allele_cols = [c for c in raw.columns if c.startswith("Allele ")]
    height_cols = [c.replace("Allele", "Height") for c in allele_cols]
    size_cols   = [c.replace("Allele", "Size") for c in allele_cols]
    has_size = all(c in raw.columns for c in size_cols)
    print(f"  rows={len(raw)}  allele_cols={len(allele_cols)}  size_cols={'yes' if has_size else 'no'}")

    markers = sorted(raw["Marker"].dropna().unique().tolist())
    shared = [m for m in markers if m in locus_to_idx]
    dropped = [m for m in markers if m not in locus_to_idx]
    print(f"  markers present={len(markers)} | shared-with-GF={len(shared)} | dropped={dropped}")

    # ── Build per-sample tokens / flat / size, harmonised to the GF locus map ────
    sample_tokens: dict[str, list[tuple[float, float, float]]] = {}
    sample_sizes:  dict[str, list[float]] = {}
    sample_flats:  dict[str, np.ndarray] = {}
    sample_donors: dict[str, list[int]] = {}

    for sf, grp in raw.groupby("Sample File"):
        donors = parse_donors(str(sf))
        if not donors:
            continue
        toks: list[tuple[float, float, float]] = []
        szs:  list[float] = []
        flat = np.zeros(n_flat, dtype=np.float32)
        for _, row in grp.iterrows():
            locus = row["Marker"]
            if locus not in locus_to_idx:                       # harmonise: drop non-GF loci
                continue
            locus_idx = float(locus_to_idx[locus])
            for k, (ac, hc) in enumerate(zip(allele_cols, height_cols)):
                av = allele_to_float(row[ac])
                if av is None:
                    continue
                h = row.get(hc, None)
                if pd.isna(h):
                    continue
                try:
                    h_val = float(h)
                except (ValueError, TypeError):
                    continue
                log_h = float(np.log1p(h_val))
                toks.append((locus_idx, av, log_h))
                # aligned fragment size(bp) sidecar (0 if this kit didn't export Size)
                sz = 0.0
                if has_size:
                    s = row.get(size_cols[k], None)
                    try:
                        sz = float(s) if not pd.isna(s) else 0.0
                    except (ValueError, TypeError):
                        sz = 0.0
                szs.append(sz)
                ci = flat_col_index.get((locus, av))            # bins unseen in GF train: skip
                if ci is not None:
                    flat[ci] = max(flat[ci], log_h)
        if not toks:
            continue
        sample_tokens[sf] = toks
        sample_sizes[sf]  = szs
        sample_flats[sf]  = flat
        sample_donors[sf] = donors

    names = sorted(sample_tokens.keys())
    N = len(names)
    print(f"  valid samples={N}")
    if N == 0:
        raise SystemExit("No valid samples after parsing.")

    tokens = np.zeros((N, MAX_SEQ, 3), dtype=np.float32)
    mask   = np.zeros((N, MAX_SEQ), dtype=bool)
    size   = np.zeros((N, MAX_SEQ), dtype=np.float32)
    for i, sf in enumerate(names):
        t = sample_tokens[sf]; s = sample_sizes[sf]
        n = min(len(t), MAX_SEQ)
        tokens[i, :n, :] = np.array(t[:n], dtype=np.float32)
        size[i, :n]      = np.array(s[:n], dtype=np.float32)
        mask[i, :n] = True

    flat = np.stack([sample_flats[sf] for sf in names])

    # ── Labels / NOC / closed-set flag ──────────────────────────────────────────
    # noc_true = number of contributors parsed from the name (donor-agnostic count).
    donors_all = [sample_donors[sf] for sf in names]
    noc_true = np.array([min(len(d), 5) for d in donors_all], dtype=np.int64)
    if id_measurable:
        y = np.zeros((N, 45), dtype=np.float32)
        for i, ds in enumerate(donors_all):
            for d in ds:
                if d in KNOWN_SET:
                    y[i, KNOWN_DONORS.index(d)] = 1.0
        has_unknown = np.array([any(d not in KNOWN_SET for d in ds) for ds in donors_all], bool)
        is_closed = ~has_unknown
    else:
        # rd12: donor integers are DIFFERENT people -> cannot use the 45-class head.
        y = np.zeros((N, 45), dtype=np.float32)
        is_closed = np.zeros(N, dtype=bool)

    # ── Condition (§12) ─────────────────────────────────────────────────────────
    cond_raw, cond_cat = [], []
    for i, sf in enumerate(names):
        category, raw_code, _level = parse_condition(str(sf), int(noc_true[i]))
        cond_raw.append(category)
        cond_cat.append(COND_CATS.index(category) if category in COND_CATS else -1)
    cond_cat = np.array(cond_cat, dtype=np.int32)

    # ── Enrich to 8/9/11 (identical transform to the trained pipeline) ──────────
    en9 = enrich_tokens(tokens, mask)
    en11 = add_size_fields(en9, mask, size)

    np.save(out / f"tokens8_{args.tag}.npy", en9[:, :, :8].astype(np.float32))
    np.save(out / f"tokens9_{args.tag}.npy", en9.astype(np.float32))
    np.save(out / f"tokens11_{args.tag}.npy", en11.astype(np.float32))
    np.save(out / f"mask_{args.tag}.npy",     mask)
    np.save(out / f"y_{args.tag}_set.npy",    y)
    np.save(out / f"noc_{args.tag}.npy",      noc_true)
    np.save(out / f"size_{args.tag}.npy",     size)
    np.save(out / f"is_closed_{args.tag}.npy", is_closed)
    np.save(out / f"condition_{args.tag}.npy", cond_cat)
    json.dump(names, open(out / f"names_{args.tag}.json", "w"))

    noc_dist = {int(k): int(v) for k, v in zip(*np.unique(noc_true, return_counts=True))}
    closed_noc = {int(k): int(v) for k, v in
                  zip(*np.unique(noc_true[is_closed], return_counts=True))} if is_closed.any() else {}
    summary = {
        "tag": args.tag, "folder": args.folder, "panel": args.panel,
        "id_measurable": bool(id_measurable), "n_samples": int(N),
        "n_closed": int(is_closed.sum()), "noc_dist": noc_dist, "closed_noc_dist": closed_noc,
        "shared_loci": shared, "dropped_loci": dropped, "has_size": bool(has_size),
    }
    json.dump(summary, open(out / f"summary_{args.tag}.json", "w"), indent=2)
    print(f"  noc_dist={noc_dist}  closed={int(is_closed.sum())}  id_measurable={id_measurable}")
    print(f"  wrote tokens8/9/11_{args.tag}.npy (+ mask/y/noc/size/is_closed/condition) -> {out}")


if __name__ == "__main__":
    main()
