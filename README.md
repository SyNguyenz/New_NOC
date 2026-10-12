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

## Results (10 folds, mean ± SD)

GlobalFiler (3500, 29 cycles). 10 donor-disjoint folds, each holding out 5 of the 50 donors; trained on
in-silico mixtures and real single-source profiles only; tested on every real profile (25,154 in total,
14,093 of them mixtures).

| task | metric | result |
|---|---|---|
| Count (NOC) | macro F1, NOC1–5 | 0.9403 ± 0.0101 |
| | accuracy | 0.9603 ± 0.0083 |
| Identification | macro F1 over donors, mixtures | 0.9838 ± 0.0028 |
| | exact donor set, macro NOC2–5 | 0.9042 ± 0.0145 |
| | exact donor set at the decoded k (all NOC) | 0.9487 ± 0.0098 |
| | exact donor set at the true k (all NOC) | 0.9777 ± 0.0050 |
| Open set | AUROC | 0.8811 ± 0.0295 |
| | open samples flagged / closed samples flagged | 0.627 ± 0.091 / 0.095 ± 0.010 |

| | NOC1 | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|---|
| count F1 | 0.9973 ± 0.0008 | 0.9649 ± 0.0053 | 0.9318 ± 0.0178 | 0.9078 ± 0.0264 | 0.9000 ± 0.0267 |
| exact donor set, decoded k | 0.9932 ± 0.0012 | 0.9408 ± 0.0101 | 0.9355 ± 0.0248 | 0.9450 ± 0.0175 | 0.7954 ± 0.0470 |
| exact donor set, true k | 0.9949 ± 0.0010 | 0.9955 ± 0.0036 | 0.9851 ± 0.0072 | 0.9727 ± 0.0120 | 0.8675 ± 0.0231 |
