# STR mixture contributor identification (NOC + who)

Raw PROVEDIt GlobalFiler profiles → the number of contributors and which reference donors they are.

Paths below are from the repo root. Step-by-step 10-fold / Kaggle run: [code/RUN_10FOLD.md](code/RUN_10FOLD.md).
Formulas: [code/FORMULAS.md](code/FORMULAS.md).

## Layout

```
preprocess.sh      raw -> data/foldK, one fold at a time
code/              all code (this is the Kaggle "code" dataset)
data_raw/          PROVEDIt CSVs + known genotypes
data/foldK/        a fold's finished dataset (built by preprocess.sh, not tracked)
reports/           generator / experiment reports
```

## Pipeline (raw → output)

```
data_raw/  (PROVEDIt CSVs + known genotypes)
   │  code/preprocess.py (preprocess.sh sets STR_FOLD):
   │      kit_raw · prepare/size pass · build_donor_geno · extract_phi_condition · synth/extract_genotypes ·
   │      build_real_attr · features/enrich · calibrate_*.py (fold's NOC1 train → code/data/*.json)
   ▼
code/data/  (real arrays + the generator's calibrated laws; working dir, rebuilt per fold)
   │  make_insilico.py --build 50000 --seed 42     (conditioned draws; writes gen_law.npy)
   ▼
code/data_insilico_w/  (in-silico train + real val/test/open)  ──copied to──▶  data/foldK/
   │  features/enrich.py                            (kaggle_run_increment1.py does this -> data_w_inc22/)
   ▼
code/train_set_transformer.py   model: code/models/set_transformer.py   decode: code/decode_layer.py
   ▼
code/results/<out_subdir>_seed<seed>/  {best_model.pt, last_model.pt, metrics.json, y_test_pred.npy, y_test_true.npy}
```

Every step from raw to output uses only the chosen fold: NOC1 is split by PCR tube, the calibrated laws are fitted
on that fold's NOC1 train, and the fold's real combos are excluded from the in-silico train.

## Run

```bash
bash preprocess.sh 0                                                       # raw -> data/fold0
INSILICO_W=data/fold0 python code/kaggle_run_increment1.py --seed 42       # enrich + train + decode
STR_DATA_DIR=<data_dir> python code/train_set_transformer.py --seed 42 --out_subdir <name>
STR_INIT_FROM=<ckpt> STR_EPOCHS=3 STR_LR=1e-4 ...                         # fine-tune
STR_EVAL_ONLY=1 STR_INIT_FROM=<ckpt> STR_DATA_DIR=<data_dir> ...          # decode/evaluate a checkpoint only
```

`bash preprocess.sh` with no argument prints its usage (several folds, ranges, `all`, `--override`).

Environment: `STR_DATA_DIR`, `STR_DEVICE` (cpu/cuda/mps), `STR_BATCH` (256 = recipe; 128 on a 4 GB card),
`STR_EPOCHS`, `STR_LR`, `STR_INIT_FROM`, `STR_EVAL_ONLY` (`STR_FOLD` is set by preprocess.sh).
Kit: `STR_KIT` = `3500_GF29cycles` (default) or `3500_F6C29cycles_hlfrxn`; raw folder, genotype file, dye
layout and repeat lengths come from `code/kit.py`, the number of loci from the data. Set it for preprocess and training alike.

## Model (`code/models/set_transformer.py`)

| component | choice |
|---|---|
| encoder | ISAB++ (SetNorm), `nc_attn = mab0` (SigmoidMABpp / MABpp) |
| token embed | periodic PLR (`sigma=0.3`), 8-field tokens |
| pre-encoder | set_of_set (private/shared split) + feas_filter (drop 0-carrier peaks) |
| decoder | AdaptiveSlot: CoSA genotype init → GSANet attribution refine → MESH Sinkhorn-OT loop → AdaSlot gate |
| count | the AdaSlot gate: k = round(Σ gate), corrected by the decode layer |
| aux (train only) | per-peak attribution (generator's true split where known), Kendall-weighted |

Objective (closed set only; the open split is never trained on): `ASL(γ_neg=4)` on `logits_cls` + `0.05·SmoothL1(Σgate, NOC)`
+ `Kendall(attr CE)`; `mask_peaks=0.15` on shared peaks. Selection = macro-over-NOC oracle Recall@k on val
(real NOC1 val + 15% of the in-silico combos, held out); no early stopping.

## Decode (`code/decode_layer.py`) — every parameter chosen on val

1. **Penalty + bonus on the gate**, over all donors:
   - penalty: `u_c` = largest single-peak share of the positive contributions to c's gate (gradient × input);
     if `u_c ≥ u_min`, that peak anchors c's level and `p_c` = how plausible c's other alleles are next to the
     majors; `logit_c -= a · u_c · (1 − p_c)`.
   - bonus: `f2_c` = c's proportion in a joint fit with the majors relative to the weakest (filter `f2 ≥ f2_min`),
     `f1_c` = allele coverage beyond the sample's background, `f3_c` = relative gate;
     `logit_c += b · max(f1_c, .05) · f2_c · f3_c`.
   - k = round(Σ corrected gate).
2. **Order**: set-conditional greedy on the phi-reranked logits (`code/phi_rerank.py`), each pick scored by what it
   newly explains (height over the profile median, per allele).
3. **Drop**: the k-th pick is dropped when it explains ≤ t.
4. **Open score** (a donor outside the panel?): attr background share, off-panel height, height the decoded set leaves
   unexplained, gate total, 6th gate, weakness of the k-th gate; each standardised on val, summed one-sided
   (excess in the open direction only), flagged above the 95th percentile of val. Pooled over NOC.

The eval prints COUNT (sum-of-gates and decoded k), ID (at the true k and at the decoded k) and OPEN SET reports,
each with per-class precision / recall / F1 and macro / micro averages.

## Fold 0 result (`code/results/dgw_ref_seed42`)

v3 (`results/inc22_v3_seed42`, 150 epochs) fine-tuned 3 epochs on `data_dgw` (12k, seed 7), decode as above
(chosen then on the old in-silico DEV split: α .2, t .0299, u_min .1, a 16, f2_min .3, b 32):

| | macro ID (NOC2-5) | macro count | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|---|---|
| in-silico DEV (old split) | .873 | | | | | |
| real test | .933 | .950 | .961 | .974 | .967 | .828 |
