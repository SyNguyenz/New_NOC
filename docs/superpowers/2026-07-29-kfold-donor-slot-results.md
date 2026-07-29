# Donor-Slot SetTransformer — GroupKFold(5) results (2026-07-29)

Runner: `code/kfold_donor_slot.py`. Model: repo `SetTransformerMixture` (inc22),
init from `Donor-Slot_Set_Transformer.pt`. Data: real GF (PROVEDIt 3500_GF29cycles)
built via `data/prepare_data_set.py` + `features/enrich.py` → 8829 closed samples,
24 loci, 8-feat tokens. GroupKFold(5) grouped by donor-combo (65 groups).
Task evaluated: contributor identification = 45-donor presence (`logits_cls`),
oracle top-true-k decode. GPU: RTX 5060 Ti.

## Results

| init | Donor-set EM | Micro-F1 | Macro-F1 | per-fold EM |
|---|---|---|---|---|
| pretrained (ckpt as init) | 0.9884 ± 0.0078 | 0.9912 ± 0.0060 | 0.3571 ± 0.0358 | 0.992, 0.991, 0.973, 0.993, 0.993 |
| scratch (leak-free)       | 0.0435 ± 0.0346 | 0.2539 ± 0.0532 | 0.0980 ± 0.0280 | 0.037, 0.108, 0.011, 0.044, 0.017 |

## ⚠️ Methodology finding — donor identification is ill-posed under this CV

- **pretrained** scores ~0.99 only because the checkpoint was trained on part of this
  same real GF pool (leakage). Not a valid cross-validation number.
- **scratch** collapses, and per-NOC EM shows why: **NOC=1 EM ≈ 0.0007**. GroupKFold
  groups every donor's single-source (NOC=1) samples under that donor's own group, so
  each donor's identity is held **entirely** in one fold. A model cannot identify a
  donor it never saw in training → donor identification under donor-disjoint CV is
  ill-posed.

**Conclusion:** GroupKFold(5)-by-donor-combo is the right protocol for the notebook's
actual target — **NOC count** (`target_noc`), which generalizes across held-out combos —
but NOT for identifying *which* donor. To get a defensible cross-validation number,
evaluate the NOC-count head (`logits_card`) instead of donor presence.

## Reproduce

```bash
# build data (once)
python code/data/prepare_data_set.py
python code/features/enrich.py code/data
# run
python code/kfold_donor_slot.py --ckpt /path/Donor-Slot_Set_Transformer.pt \
    --data-dir code/data --init scratch    # leak-free
python code/kfold_donor_slot.py --ckpt /path/Donor-Slot_Set_Transformer.pt \
    --data-dir code/data --init pretrained  # uses the checkpoint (leaky)
```
Outputs: `results/kfold_donor_slot/{kfold_metrics_<init>.json, per_fold_scores_<init>.json}`
(gitignored).
