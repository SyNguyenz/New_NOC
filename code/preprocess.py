"""
inc22_clean/preprocess.py — CONSOLIDATED raw -> full inc22 dataset, ONE command.

Design (per request): readers of the SAME source are MERGED; readers of DIFFERENT sources stay SEPARATE.
  • SAME source (the kit's Filtered CSVs, STR_KIT): base arrays + per-peak size -> ONE csv_pass():
    the CSVs are read ONCE (identical pd.concat) and BOTH the base arrays (tokens/mask/Xflat/y/noc/meta/
    names) AND per-peak size are produced from that single DataFrame. Verbatim logic from both scripts on
    the shared `raw` -> bit-identical to running them separately.
  • DIFFERENT sources stay as their own steps (subprocess; verbatim, bit-identical):
      build_donor_geno.py        (Known-Genotypes xlsx) -> donor_geno.npy + mask
      extract_phi_condition.py   (sample NAMES)         -> phi/condition
      synth/extract_genotypes.py (tokens)               -> donor_genotypes.csv
      build_real_attr.py         (geno csv + tokens)    -> attr
      features/enrich.py         (tokens)               -> tokens8/9/11
      make_insilico.py           (data/)                -> data_insilico_w/

All clean-local (HERE = inc22_clean/; data_raw is the junction to root). Verify with verify_preprocess.py.
"""
from __future__ import annotations
import glob, json, os, re, shutil, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import kit

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RAW_FILTERED = HERE / "data_raw" / "PROVEDIt_1-5-Person CSVs Filtered"
KIT_PATTERN = str(RAW_FILTERED / f"*{kit.KIT}" / "**" / "*.csv")
# Peaks per profile kept as tokens. 160 cut 7-10 % of the real NOC4-5 profiles (up to 176 peaks) and 20 % of the
# in-silico NOC5 (up to 191), and the two sides were cut differently: real in CSV order, which drops whole loci of the
# last dye, tall alleles included; the generator by keeping the tallest. The most any GF profile carries is 214, so
# 224 cuts nothing; if one ever exceeds it, the tallest are kept on both sides.
MAX_SEQ = 224
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
    """Read the kit's Filtered CSVs ONCE, emit base arrays + per-peak size."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Scanning {kit.KIT} CSVs …")
    csv_files = [f for f in glob.glob(KIT_PATTERN, recursive=True) if "Known Genotypes" not in f]
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
    n_cut = 0
    for i, sf in enumerate(sample_files):
        toks = sample_tokens[sf]
        if len(toks) > MAX_SEQ:                     # see MAX_SEQ: the tallest are kept, as the generator keeps them
            keep = sorted(np.argsort([-t[2] for t in toks], kind="stable")[:MAX_SEQ])
            toks = [toks[k] for k in keep]; n_cut += 1
        n = len(toks)
        tokens_arr[i, :n, :] = np.array(toks, dtype=np.float32)
        mask_arr[i, :n] = True
    print(f"  profiles over MAX_SEQ={MAX_SEQ} (tallest kept): {n_cut}")
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
    # Split by PCR TUBE, stratified by donor: the 5 / 15 / 25 sec injections of one amplification are near-copies of
    # each other, so a sample-level split put another injection of 91% of the test NOC1 profiles into train. All
    # injections of a tube now land in one split (20 folds of tubes: 3 -> test, 3 -> val, 14 -> train, ~70/15/15).
    tube = np.array([re.sub(r"\.\d+\s*sec.*$", "", sample_files[i]) for i in single_idx])
    folds = list(StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=RANDOM_SEED)
                 .split(single_idx, donor_labels, groups=tube))
    part = np.zeros(len(single_idx), int)                  # 0 train, 1 val, 2 test
    for f, (_, ix) in enumerate(folds):
        part[ix] = 2 if f < 3 else (1 if f < 6 else 0)
    ss_train, ss_val, ss_test = single_idx[part == 0], single_idx[part == 1], single_idx[part == 2]
    assert not (set(tube[part == 0]) & set(tube[part != 0])), "a PCR tube is split across train and val/test"
    assert set(donor_labels[part == 0]) == set(donor_labels), "a donor has no NOC1 training profile"
    print(f"  NOC1 split by PCR tube: train {len(ss_train)} / val {len(ss_val)} / test {len(ss_test)} "
          f"({len(set(tube))} tubes)")
    idx_train = ss_train                                   # single-source only
    idx_val = ss_val                                       # single-source only (k=1 rows for RF count)
    idx_test = np.concatenate([ss_test, np.array(multi_test, dtype=ss_test.dtype)])
    noc_true_all = np.array([len(set(sample_donors[sf])) for sf in sample_files], dtype=np.int32)
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
        # TRUE contributor count, including donors outside the 45-donor panel. `noc_{split}`
        # is labels.sum(1) and therefore counts KNOWN donors only, so it under-counts the
        # open split. NOC supervision does not need donor identity, so the open split is
        # usable training data for a count head — but only with this label.
        np.save(DATA_DIR / f"noc_true_{split}.npy", noc_true_all[ix])
    for split, ix in splits.items():
        names = [sample_files[i] for i in ix]
        with open(DATA_DIR / f"meta_sample_names_{split}.json", "w") as f:
            json.dump(names, f)
    meta = {
        "kit": kit.KIT, "loci": loci, "locus_to_idx": locus_to_idx,
        "locus_bin_lists": {loc: [float(v) for v in bins] for loc, bins in locus_bin_lists.items()},
        "flat_cols": flat_cols, "n_flat": n_flat, "max_seq": MAX_SEQ,
        "known_donors": KNOWN_DONORS, "unknown_donors": UNKNOWN_DONORS, "random_seed": RANDOM_SEED,
        "n_classes": 45, "split_sizes": {k: len(v) for k, v in splits.items()},
        "combo_list": [[int(d) for d in c] for c in all_combos],
        "split_policy": {
            "description": (
                "Single-source (NOC=1): stratified by donor (all 45 donors in every split). "
                "Multi-person (NOC>=2): EVERY closed donor-combo goes to test — the network trains "
                "on in-silico mixtures only, so no real combo is spent on train/val, and all of them "
                "are excluded from in-silico generation. The post-hoc decode is fit leave-one-combo-out "
                "via combo_id_test.npy. val is single-source only."),
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

    # ===== per-peak size from the SAME `raw` =====
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
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    print(f"\n>>> {rel} {' '.join(args)}")
    subprocess.run([PY, str(HERE / rel), *args], cwd=str(HERE), env=env, check=True)


def main():
    assert (HERE / "data_raw").exists(), f"data_raw junction missing at {HERE/'data_raw'}"
    # EVERYTHING BELOW BELONGS TO THIS FOLD. data/ and data_insilico_w/ are emptied first, so no step can read what an
    # earlier run (another fold) left there: each calibration step imports make_insilico, which loads every
    # calibration file present in data/, and calibrate.py caches pullup_pairs.json from that fold's profiles. Each
    # calibration is then an increment over the generator with exactly the laws calibrated before it in this run.
    for d in (DATA_DIR, HERE / "data_insilico_w"):
        if d.exists():
            shutil.rmtree(d)
    print(f"fold {FOLD}: data/ and data_insilico_w/ cleared, rebuilding from data_raw")
    import kit_raw
    kit_raw.main()                                                     # the kit's .xlsx exports -> .fromxlsx.csv
    csv_pass()                                                         # the kit's CSVs (MERGED) -> base + size
    step("build_donor_geno.py")                                       # xlsx        -> donor_geno + mask
    step("extract_phi_condition.py")                                  # names       -> phi/condition
    step("synth/extract_genotypes.py")                                # tokens      -> donor_genotypes.csv
    step("build_real_attr.py")                                        # geno+tokens -> attr
    step("features/enrich.py", args=(str(DATA_DIR),))                 # tokens      -> tokens8/9/11
    step("calibrate_scatter.py",                                      # NOC1 + twin -> art_scatter.json (width of the
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # stutter ratio draw, per offset, from its tail)
    step("calibrate_pullup.py",                                       # NOC1 + twin -> pullup_emit.json (the pull-up
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # the analysis could not call; an increment too)
    step("calibrate_allele_stut.py",                                  # NOC1 + twin -> allele_stut.json (per-allele
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # n-1 rate: an increment over the generator,
                                                                      # so it is re-measured against the one in use)
    step("calibrate_shoulder.py",                                     # NOC1 + twin -> shoulder_bp.json (a peak 1/2/3
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # bp from an allele: survival and floor by distance)
    step("calibrate_donor_stut.py",                                   # NOC1 + twin -> donor_stut.json (n-1 emission per
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # donor x allele: same-length alleles that do not stutter)
    step("calibrate_stut_spread.py",                                  # NOC1 + twin -> stut_spread.json (n-1 scatter
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # width by parent height: less scatter, more copies)
    step("calibrate_stut_parent.py",                                  # NOC1 + twin -> stut_parent.json (stutter vs
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # parent height, floor separated; after allele_stut)
    step("calibrate_crowd.py",                                        # TRAIN NOC1 + twin -> noise_cr_inc.json (noise
         env_extra={"STR_DATA_DIR": str(DATA_DIR)})                   # floor vs crowding, over all the laws above)
    step("make_insilico.py",                                         # data/       -> data_insilico_w (no drop-in)
         env_extra={"STR_DATA_DIR": str(DATA_DIR), "STR_OUT_DIR": str(HERE / "data_insilico_w")},
         # every in-silico mixture draws its treatment, tube Q, template and injection from ranges measured on
         # TRAIN NOC1 (gen_conditioned), which is what puts the calibrated presence/height laws into training.
         args=("--build", "50000", "--noc_weights", "1,1.5,2.5,2", "--seed", "42"))
    print("\nDONE — full inc22 dataset built in inc22_clean/ (data/ + data_insilico_w/).")


if __name__ == "__main__":
    main()
