# Increment 1 on Kaggle — enriched relational token (head-to-head)

Goal: does an ENRICHED 8-field token (Hb, stutter-ratio, rank, n-peaks, glob-rel — derived from the
existing locus/allele/height token) raise contributor-ID ranking (oracle NOC4/5) vs the 3-field baseline?
Compared for hybrid and pure-ST, on the same combo-diverse in-silico data.

## Steps
1. **Upload data**: zip your local `data_insilico_w/` (all `.npy` + `meta_set.json`) → Kaggle Dataset,
   name it e.g. `noc-insilico-w`. (tokens8_*.npy are NOT needed — generated on Kaggle by the script.)
2. **Upload code**: this `kaggle_bundle/` as a Kaggle Dataset, e.g. `noc-code` (or clone the repo).
3. **GPU notebook** (enable GPU):
   ```
   !cp -r /kaggle/input/noc-code/* /kaggle/working/
   !INSILICO_W=/kaggle/input/noc-insilico-w python /kaggle/working/kaggle_run_increment1.py
   ```
   Optional env: `RUNS=hybrid3,hybrid8,st3,st8` (subset to save time), `EPOCHS=150`.
4. **Download** `/kaggle/working/results/inc1_*` (each has `metrics.json` + `best_model.pt`).
   Send the four `metrics.json` back here.

## What to look at (the hypothesis)
- `per_noc_oracle` (ranking ceiling) for tok8 vs tok3 — does enrichment raise N4/N5 oracle?
- `em_two_stage` / `em_two_stage_pgnoc` (decoded), `reject_auroc`.
- Compare on the in-silico-dev-equivalent + the REAL test numbers the scripts print.

## Baseline to beat (tok3, data_insilico_w): decoded 0.954 / oracle 0.975 (N4 .733/.822, N5 .396/.604).

## Notes
- The script copies the dataset to a writable dir and runs `features/enrich.py` to build tokens8_*.npy.
- `--n_token_feats 8` loads tokens8 and auto-standardizes the new features from train statistics.
- Selection/eval discipline: trust in-silico-dev trends; REAL test strata are tiny (N4 n=45, N5 n=48) → wide CIs.
