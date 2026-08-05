"""
inc22_clean/preprocess.py — CONSOLIDATED raw -> full inc22 dataset, ONE command.

Design (per request): readers of the SAME source are MERGED; readers of DIFFERENT sources stay SEPARATE.
  • SAME source (GF29cycles Filtered CSVs): prepare_data_set + extract_size  -> MERGED into csv_pass():
    the CSVs are read ONCE (identical pd.concat) and BOTH the base arrays (tokens/mask/Xflat/y/noc/meta/
    names) AND per-peak size are produced from that single DataFrame. Verbatim logic from both scripts on
    the shared `raw` -> bit-identical to running them separately.
  • DIFFERENT sources stay as their own steps (subprocess; verbatim, bit-identical):
      build_donor_geno.py        (Known-Genotypes xlsx) -> donor_geno.npy + mask
      extract_phi_condition.py   (sample NAMES)         -> phi/condition
      synth/extract_genotypes.py (tokens)               -> donor_genotypes.csv
      build_real_attr.py         (geno csv + tokens)    -> attr
      features/enrich.py         (tokens)               -> tokens8/9/11
      make_insilico.py           (data/)                -> data_insilico_w/   [STR_DROPIN=0: inc22 has no drop-in]

All clean-local (HERE = inc22_clean/; data_raw is the junction to root). Verify with verify_preprocess.py.
"""
from __future__ import annotations
import glob, json, os, re, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_FILTERED = HERE / "data_raw" / "PROVEDIt_1-5-Person CSVs Filtered"
KIT_PATTERN = str(RAW_FILTERED / "*GF29cycles" / "**" / "*.csv")
MAX_SEQ = 160
RANDOM_SEED = 42
PY = sys.executable

# ── donor split (verbatim: prepare_data_set.py) — 10-fold, each donor unknown once ──
ALL_DONORS = list(range(1, 51))
N_FOLDS = 10
FOLD = int(os.environ.get("STR_FOLD", "0"))
if not 0 <= FOLD < N_FOLDS:
    raise SystemExit(f"STR_FOLD must be in [0, {N_FOLDS - 1}], got {FOLD}")
_rng = np.random.default_rng(RANDOM_SEED)
_shuffled = _rng.permutation(ALL_DONORS).tolist()
UNKNOWN_DONORS = sorted(_shuffled[5 * FOLD: 5 * FOLD + 5])
KNOWN_DONORS = sorted(set(ALL_DONORS) - set(UNKNOWN_DONORS))
KNOWN_SET = set(KNOWN_DONORS)


def allele_to_float(allele):                                  # verbatim: prepare_data_set
    a = str(allele).strip()
    if a in ("", "nan", "OL"):
        return None
    if a == "X":
        return -2.0
    if a == "Y":
        return -1.0
    try:
        return float(a)
    except ValueError:
        return None


def parse_donors(filename):                                   # verbatim: prepare_data_set
    parts = filename.split("-")
    if len(parts) < 3:
        return []
    contrib_part = parts[2]
    if "_" in contrib_part:
        donors = []
        for seg in contrib_part.split("_"):
            m = re.match(r"^(\d+)", seg)
            if m:
                donors.append(int(m.group(1)))
        return donors
    else:
        m = re.match(r"^(\d+)", contrib_part)
        return [int(m.group(1))] if m else []


def build_label_vector(donor_ids):                            # verbatim: prepare_data_set
    vec = np.zeros(45, dtype=np.float32)
    for d in donor_ids:
        if d in KNOWN_SET:
            vec[KNOWN_DONORS.index(d)] = 1.0
    return vec


def allele_key(v):                                            # verbatim: extract_size
    s = str(v).strip()
    if s in ("", "nan", "OL", "None"):
        return ""
    if s in ("X", "-2.0", "-2"):
        return "-2.0"
    if s in ("Y", "-1.0", "-1"):
        return "-1.0"
    try:
        return f"{round(float(s), 1):.1f}"
    except ValueError:
        return ""


def csv_pass():
    """MERGED prepare_data_set + extract_size — read the GF29 CSVs ONCE, emit base arrays + size."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Scanning GF29cycles CSVs …")
    csv_files = glob.glob(KIT_PATTERN, recursive=True)
    print(f"  Found {len(csv_files)} CSV files")
    dfs = [pd.read_csv(f, low_memory=False) for f in csv_files]
    raw = pd.concat(dfs, ignore_index=True)                   # <-- the single shared read
    print(f"  Total locus rows: {len(raw)}")

    allele_cols = [c for c in raw.columns if c.startswith("Allele ")]
    height_cols = [c.replace("Allele", "Height") for c in allele_cols]
    size_cols = [c.replace("Allele", "Size") for c in allele_cols]

    loci = sorted(raw["Marker"].dropna().unique().tolist())
    locus_to_idx = {loc: i for i, loc in enumerate(loci)}
    print(f"  Loci ({len(loci)}): {loci}")

    # ===== prepare_data_set: collect allele bins (verbatim) =====
    print("First pass: collecting allele bins …")
    allele_bins = {loc: set() for loc in loci}
    for _, row in raw.iterrows():
        locus = row["Marker"]
        if locus not in locus_to_idx:
            continue
        for ac in allele_cols:
            av = allele_to_float(row[ac])
            if av is not None:
                allele_bins[locus].add(av)
    locus_bin_lists = {loc: sorted(bins) for loc, bins in allele_bins.items()}
    flat_cols = []
    flat_col_index = {}
    for loc in loci:
        for av in locus_bin_lists[loc]:
            allele_str = str(int(av)) if av == int(av) else str(av)
            if av == -2.0:
                allele_str = "X"
            elif av == -1.0:
                allele_str = "Y"
            col_name = f"{loc}_{allele_str}"
            flat_col_index[(loc, av)] = len(flat_cols)
            flat_cols.append(col_name)
    n_flat = len(flat_cols)
    print(f"  Flat feature dims: {n_flat}")

    # ===== second pass: tokens + flats (prepare_data_set) =====
    print("Second pass: building sample features …")
    sample_tokens = {}
    sample_flats = {}
    sample_donors = {}
    for sf, grp in raw.groupby("Sample File"):
        donors = parse_donors(str(sf))
        if not donors:
            continue
        tokens = []
        flat = np.zeros(n_flat, dtype=np.float32)
        for _, row in grp.iterrows():
            locus = row["Marker"]
            if locus not in locus_to_idx:
                continue
            locus_idx = float(locus_to_idx[locus])
            for ac, hc in zip(allele_cols, height_cols):
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
                tokens.append((locus_idx, av, log_h))
                col_idx = flat_col_index.get((locus, av))
                if col_idx is not None:
                    flat[col_idx] = max(flat[col_idx], log_h)
        if not tokens:
            continue
        sample_tokens[sf] = tokens
        sample_flats[sf] = flat
        sample_donors[sf] = donors

    sample_files = sorted(sample_tokens.keys())
    print(f"  Valid samples: {len(sample_files)}")

    labels = np.stack([build_label_vector(sample_donors[sf]) for sf in sample_files])
    nocs = labels.sum(axis=1).astype(np.int32)
    has_unknown = np.array(
        [any(d not in KNOWN_SET for d in sample_donors[sf]) for sf in sample_files], dtype=bool)
    is_closed = ~has_unknown

    tokens_arr = np.zeros((len(sample_files), MAX_SEQ, 3), dtype=np.float32)
    mask_arr = np.zeros((len(sample_files), MAX_SEQ), dtype=bool)
    for i, sf in enumerate(sample_files):
        toks = sample_tokens[sf]
        n = min(len(toks), MAX_SEQ)
        tokens_arr[i, :n, :] = np.array(toks[:n], dtype=np.float32)
        mask_arr[i, :n] = True
    flat_arr = np.stack([sample_flats[sf] for sf in sample_files])

    # ===== group-aware split (verbatim: prepare_data_set) =====
    closed_idx = np.where(is_closed)[0]
    open_idx = np.where(~is_closed)[0]
    print(f"  Closed-set: {len(closed_idx)}  Open-set: {len(open_idx)}")
    # Multi-person: EVERY closed combo -> test. The network trains on in-silico only (build_train
    # discards real mixtures), so real combos parked in `train` protected nothing and just shrank the
    # evaluation. They are excluded from in-silico generation instead, and the post-hoc decode is fit
    # leave-one-combo-out at eval time via combo_id_test.npy. val stays single-source (k=1 rows).
    sample_combo = {}
    for i in closed_idx:
        sf = sample_files[i]
        donors_i = sample_donors[sf]
        noc_i = int(nocs[i])
        if noc_i >= 2:
            sample_combo[i] = tuple(sorted(donors_i))
    combos_by_noc = {}
    for i, combo in sample_combo.items():
        combos_by_noc.setdefault(len(combo), set()).add(combo)
    combos_by_noc = {noc: sorted(combos) for noc, combos in combos_by_noc.items()}
    all_combos = sorted({c for cs in combos_by_noc.values() for c in cs})
    combo_to_id = {c: j for j, c in enumerate(all_combos)}
    split_policy_combos = {"train": {}, "val": {}, "test": {}}
    for noc_i, combos in sorted(combos_by_noc.items()):
        split_policy_combos["test"][f"NOC{noc_i}"] = [[int(d) for d in c] for c in combos]
        n_samp = sum(1 for c in sample_combo.values() if len(c) == noc_i)
        print(f"  NOC={noc_i}: {len(combos)} combos -> test ({n_samp} samples)")
    multi_test = sorted(sample_combo)
    single_idx = np.array([i for i in closed_idx if int(nocs[i]) == 1])
    donor_labels = np.array([KNOWN_DONORS.index(sample_donors[sample_files[i]][0]) for i in single_idx])
    ss_train, ss_tmp, _, ss_lbl_tmp = train_test_split(
        single_idx, donor_labels, test_size=0.30, random_state=RANDOM_SEED, stratify=donor_labels)
    ss_val, ss_test = train_test_split(ss_tmp, test_size=0.50, random_state=RANDOM_SEED, stratify=ss_lbl_tmp)
    idx_train = ss_train                                   # single-source only
    idx_val = ss_val                                       # single-source only (k=1 rows for RF count)
    idx_test = np.concatenate([ss_test, np.array(multi_test, dtype=ss_test.dtype)])
    combo_id_all = np.full(len(sample_files), -1, dtype=np.int32)   # -1 = single-source
    for i, c in sample_combo.items():
        combo_id_all[i] = combo_to_id[c]
    splits = {"train": idx_train, "val": idx_val, "test": idx_test, "open": open_idx}
    for name, ix in splits.items():
        n_ss = int((nocs[ix] == 1).sum())
        n_mix = int((nocs[ix] >= 2).sum())
        print(f"  {name:5s}: {len(ix):5d} samples  (NOC=1: {n_ss}, NOC>=2: {n_mix})")

    print("Saving base arrays …")
    for split, ix in splits.items():
        np.save(DATA_DIR / f"tokens_{split}.npy", tokens_arr[ix])
        np.save(DATA_DIR / f"mask_{split}.npy", mask_arr[ix])
        np.save(DATA_DIR / f"Xflat_{split}.npy", flat_arr[ix])
        np.save(DATA_DIR / f"y_{split}_set.npy", labels[ix])
        np.save(DATA_DIR / f"noc_{split}.npy", nocs[ix])
        np.save(DATA_DIR / f"combo_id_{split}.npy", combo_id_all[ix])
    for split, ix in splits.items():
        names = [sample_files[i] for i in ix]
        with open(DATA_DIR / f"meta_sample_names_{split}.json", "w") as f:
            json.dump(names, f)
    meta = {
        "kit": "3500_GF29cycles", "loci": loci, "locus_to_idx": locus_to_idx,
        "locus_bin_lists": {loc: [float(v) for v in bins] for loc, bins in locus_bin_lists.items()},
        "flat_cols": flat_cols, "n_flat": n_flat, "max_seq": MAX_SEQ,
        "known_donors": KNOWN_DONORS, "unknown_donors": UNKNOWN_DONORS, "random_seed": RANDOM_SEED,
        "n_classes": 45, "split_sizes": {k: len(v) for k, v in splits.items()},
        "combo_list": [[int(d) for d in c] for c in all_combos],
        "split_policy": {
            "description": (
                "Single-source (NOC=1): stratified by donor (all 45 donors in every split). "
                "Multi-person (NOC>=2): EVERY closed donor-combo goes to test — the network trains "
                "on in-silico mixtures only, so no real combo is spent on train/val. All of them are "
                "excluded from in-silico generation, and the post-hoc decode (phi-rerank alpha + RF "
                "count) is fit leave-one-combo-out via combo_id_test.npy. val is single-source only."),
            "multi_person_combos": split_policy_combos,
        },
    }
    with open(DATA_DIR / "meta_set.json", "w") as f:
        json.dump(meta, f, indent=2)
    # fold provenance kept OUT of meta_set.json so fold 0 stays byte-identical to the published run
    with open(DATA_DIR / "fold_info.json", "w") as f:
        json.dump({"fold": FOLD, "n_folds": N_FOLDS, "random_seed": RANDOM_SEED,
                   "unknown_donors": UNKNOWN_DONORS, "known_donors": KNOWN_DONORS}, f, indent=2)
    print(f"  FOLD {FOLD}/{N_FOLDS}  unknown={UNKNOWN_DONORS}")

    # ===== extract_size: per-peak size from the SAME `raw` (verbatim logic) =====
    print("Building per-peak size from the same CSVs …")
    look = defaultdict(dict)
    for _, row in raw.iterrows():
        loc = row.get("Marker")
        if loc not in locus_to_idx:
            continue
        li = locus_to_idx[loc]
        for ac, sc in zip(allele_cols, size_cols):
            ak = allele_key(row.get(ac))
            if not ak or sc not in row:
                continue
            sz = row.get(sc)
            try:
                szf = float(sz)
            except (ValueError, TypeError):
                continue
            if np.isfinite(szf):
                look[str(row["Sample File"])][(li, ak)] = szf
    size_all = np.zeros((len(sample_files), MAX_SEQ), np.float32)
    for i, sf in enumerate(sample_files):
        smap = look.get(str(sf), {})
        for j in range(MAX_SEQ):
            if not mask_arr[i, j]:
                continue
            key = (int(round(float(tokens_arr[i, j, 0]))), allele_key(tokens_arr[i, j, 1]))
            if key in smap:
                size_all[i, j] = smap[key]
    for split, ix in splits.items():
        np.save(DATA_DIR / f"size_{split}.npy", size_all[ix])
    print("csv_pass done (base + size).")


def step(rel, env_extra=None, args=()):
    """Run a SEPARATE-source proven script as its own process (verbatim, bit-identical)."""
    env = {**os.environ, "STR_DROPIN": "0"}
    if env_extra:
        env.update(env_extra)
    print(f"\n>>> {rel} {' '.join(args)}")
    subprocess.run([PY, str(HERE / rel), *args], cwd=str(HERE), env=env, check=True)


def main():
    assert (HERE / "data_raw").exists(), f"data_raw junction missing at {HERE/'data_raw'}"
    csv_pass()                                                         # GF29 CSVs (MERGED) -> base + size
    step("build_donor_geno.py")                                       # xlsx        -> donor_geno + mask
    step("extract_phi_condition.py")                                  # names       -> phi/condition
    step("synth/extract_genotypes.py")                                # tokens      -> donor_genotypes.csv
    step("build_real_attr.py")                                        # geno+tokens -> attr
    step("features/enrich.py", args=(str(DATA_DIR),))                 # tokens      -> tokens8/9/11
    step("make_insilico.py",                                          # data/       -> data_insilico_w (no drop-in)
         env_extra={"STR_DATA_DIR": str(DATA_DIR), "STR_OUT_DIR": str(HERE / "data_insilico_w")},
         args=("--build", "50000", "--noc_weights", "1,1.5,2.5,2", "--seed", "42"))
    print("\nDONE — full inc22 dataset built in inc22_clean/ (data/ + data_insilico_w/).")


if __name__ == "__main__":
    main()
