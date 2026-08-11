# `inc22_clean` — clean, standalone pipeline for `inc22_fixed_aslot`

A self-contained slice of the project that runs **only** the `inc22_fixed_aslot` arm, raw → output,
without the 60-arm orchestrator, the flag zoo, or the ~2200-line train/model files. The training +
model are a **verified bit-identical extraction** of the inc22 path; the data generators are the
proven shared scripts, invoked unchanged.

## Pipeline (raw → output)

```
data_raw/  (PROVEDIt CSVs)
   │  extract_phi_condition.py · synth/extract_genotypes.py · build_real_attr.py · extract_size.py   [proven, invoked]
   ▼
data/  (real phi/condition, donor genotypes, allele→donor attr, per-peak size)
   │  make_insilico.py --build 50000 --noc_weights 1,1.5,2.5,2 --seed 42                              [proven, invoked]
   ▼
data_insilico_w/  (tokens/mask/Xflat/y/noc + phi/attr/size for val/test)
   │  copy → data_w_inc22/ ; make_dev_split.py ; features/enrich.py                                  [proven, invoked]
   ▼
data_w_inc22/  (tokens8_{train,val,test,open,dev} + dev split)
   │  train_set_transformer.py   ── model: models/set_transformer.py  (CLEAN REWRITE)                               [clean]
   ▼
results/inc22_fixed_aslot_seed42/  {best_model.pt, metrics.json, y_test_pred.npy, y_test_true.npy}
```

## Run

```bash
# from the project root
python inc22_clean/kaggle_run_increment1.py --from-raw --seed 42      # full chain from the PROVEDIt CSVs
python inc22_clean/kaggle_run_increment1.py --seed 42                 # from an already-built data_insilico_w (common)
python inc22_clean/kaggle_run_increment1.py --skip-prep --seed 42     # reuse an existing data_w_inc22/

# train directly (data dir already prepared):
STR_DATA_DIR=<data_w_dir> python inc22_clean/train_set_transformer.py --seed 42 --out_subdir inc22_fixed_aslot

# prove the clean model == the original inc22 path on the published checkpoint:
python inc22_clean/verify_inc22.py
```

## What the model is (fixed — no flags)

`models/set_transformer.py :: SetTransformerMixture` — the inc22 configuration, hard-wired:

| component | choice |
|---|---|
| encoder | ISAB++ (SetNorm, clean-path), `nc_attn = mab0` → mab0 = non-competitive **SigmoidMABpp**, mab1 = **MABpp** |
| token embed | **periodic** PLR (Gorishniy 2022), `sigma=0.3`, 8-field tokens |
| pre-encoder | **set_of_set** (private/shared split) + **feas_filter** (drop 0-carrier peaks; dedicated reject pool) |
| decoder | **AdaptiveSlot**: CoSA geno-init → GSANet attr-refine → MESH Sinkhorn-OT loop (`iters=3, eps=0.05, ot_iters=5`) → AdaSlot Gumbel-Sigmoid gate |
| count | **tierA_count**: MLP on permutation-invariant aggregates, fit on in-silico + real NOC1 only |
| aux (train only) | per-peak attribution (`soft_attr_label` = EuroForMix φ·CN soft CE) + φ regression, Kendall-weighted |

## Training objective

`ASL(γ_neg=4)` on `logits_cls` + `0.5·BCE(reject)` + `0.05·SmoothL1(Σgate, NOC)`
+ `0.3·CORN(logits_count_v2, NOC)` + `Kendall(soft_attr CE + L1 φ)`, with `mask_peaks=0.15` on shared
peaks and `mask_private` on single-carrier peaks. Selection = macro-over-NOC oracle Recall@k on the
in-silico **DEV** split.

## Decode

1. **phi-rerank** (`phi_rerank.py`): independent EM mixture-proportion (Mx) deconvolution → logarithmic-
   opinion-pool rerank of the per-donor logits, `alpha` tuned on val. Reranks the **ranking** (which
   donors / decode order) only.
2. **COUNT = `tierA_count`**: MLP on permutation-invariant aggregates (sorted prob profile,
   Σgate/sorted gate, sorted slot_mass, scale-invariant physical features), fit on in-silico + real
   NOC1 only — no labelled real mixture. **NOT count-on-rerank**: counting on the LOP-reranked score
   trades N3/N4 down for N5 (N3 −13pp, N4 −7pp). CORN is trained but never decoded.
3. decode top-`k_post` of the reranked ranking; `oracle` = top-true-k of the reranked ranking (ceiling).

`metrics.json` reports `per_noc_oracle` (test ceiling, reranked), `per_noc_at_pred_k` (deployed),
`dev_per_noc_oracle` (combo-generalization judge), `phi_rerank_alpha`, `count_acc`, `reject_auroc`.

## Excluded (everything inc22 does not use / what was measured-bad)

replicates, em_phi_feature, noise_gate, ref_match, soft_geno_attr, phi_inject, sparse_attn,
minor_weight, irm, vicreg, donor_recon, query_denoise, mass_pool, noc_contrast / noc_ord_head, sic,
the per_donor / additive / dsmil / sos / spen / pooled / hybrid decoders, geno_query, the
`decoder_source raw/local` branches, the unused `cardinality_head`, the `logits_card` CE (its
EM-optimal-k target fights tier B on the same gate tensor), and the post-hoc RF count. CORN is kept:
dropping it measurably hurt (EM .9162 vs .9260). Count-on-rerank is excluded (trades N3/N4).

## Bit-identical proof (`verify_inc22.py`)

Loads `results/inc22_fixed_aslot_seed42/best_model.pt` into **both** the original
(project-root `models/set_transformer.py`) `SetTransformerMixture` built with the inc22 kwargs **and**
the clean (`inc22_clean/models/set_transformer.py`) `SetTransformerMixture`, runs the same test batch
in `eval()`, and asserts identical outputs + gradients. Measured:

```
original load: missing=[] unexpected=[]          # inc22 kwargs reproduce the exact architecture
clean    load: missing=[] unexpected=[cardinality_head.*, ord_count_head.*]   # unused + excluded-CORN heads
forward  max|Δ| = 0.000e+00  on logits_cls / logits_card / phi / logits_attr / logit_reject
grads    max|Δ| = 0.000e+00  over all 135 shared parameters
PASS — clean SetTransformerMixture is bit-identical to the inc22 path.
```

### Reproducibility note (honest scope of "bit-identical")
The clean model computes the **exact same function and gradients** as the original inc22 path — proven
by loading the published checkpoint into both (above). It therefore reproduces the published
`inc22_fixed_aslot_seed42` result exactly from that checkpoint. A *from-scratch* retrain with the clean
pipeline trains the identical architecture/objective on the identical data and is internally
reproducible (fixed seed), but its bit-level weights will differ from the original seed-42 run because
removing the unused `cardinality_head` shifts the random-init stream (the dropped head consumed RNG
during construction in the original). The decode also now uses the deployable phi-rerank→count recipe,
so retrained `metrics.json` numbers reflect that recipe, not the bare inc22 arm's count-on-raw-probs.

## Data generators are invoked, not rewritten
`extract_*`, `make_insilico.py`, `make_dev_split.py`, `features/enrich.py` define the exact in-silico
dataset; they are shared (not arm-specific) and not the source of the complexity, so they are invoked
unchanged from the project root. Rewriting them would change the generated data and break
reproducibility. The clean rewrite is the training + model, where the bloat actually lived.
