# Donor-Slot SetTransformer — GroupKFold(5) results (2026-07-29)

Runner: `code/kfold_donor_slot.py`. Model: repo `SetTransformerMixture` (inc22), constructed
from the baked buffers of `Donor-Slot_Set_Transformer.pt`. Data: real GF (PROVEDIt
3500_GF29cycles) built via `data/prepare_data_set.py` + `features/enrich.py` → 8829 closed
samples, 24 loci, 8-feat tokens. GroupKFold(5) grouped by donor-combo (65 groups). GPU:
RTX 5060 Ti.

`--init scratch` = each fold trains from random init on its own train split (leak-free).
`--init pretrained` = each fold starts from the checkpoint (which was trained on part of this
same pool → optimistic).

## Headline — NOC count (`--task count`, well-posed; the notebook cell-14 target)

Predict number of contributors (`target_noc` ∈ 1..5) via the `logits_card` head, class-weighted
cross-entropy. Metric = Macro-F1 + Accuracy over the 5 count classes.

| init | Macro-F1 | Accuracy | per-fold accuracy |
|---|---|---|---|
| **scratch (leak-free)** | **0.4963 ± 0.1568** | **0.7851 ± 0.3290** | 0.970, 0.941, 0.950, **0.127**, 0.937 |
| pretrained (ckpt init) | 0.4830 ± 0.0752 | 0.9213 ± 0.0251 | 0.937, 0.924, 0.872, 0.935, 0.939 |

- Macro-F1 is ~equal across inits (0.496 vs 0.483) → NOC count **generalises across held-out
  donor-combos**, so this is a defensible cross-validation number.
- The accuracy gap is Fold 4: its held-out combos are hard from scratch (acc 0.127), and the
  pretrained checkpoint only recovers it (acc 0.935) because it had seen that data → the ±0.33
  scratch std is honest fold variance, not noise.
- **Report the scratch row** as the leak-free result.

## For the record — donor identification (`--task donor`, ill-posed under this CV)

Contributor identification = which of 45 known donors are present (`logits_cls`, oracle
top-true-k decode).

| init | Donor-set EM | Micro-F1 | NOC=1 EM |
|---|---|---|---|
| pretrained | 0.9884 ± 0.0078 | 0.9912 ± 0.0060 | ~0.99 |
| scratch (leak-free) | 0.0435 ± 0.0346 | 0.2539 ± 0.0532 | 0.0007 |

**Do not report these as a CV result.** GroupKFold groups every donor's single-source (NOC=1)
samples under that donor's own group, so each donor's identity is held **entirely** in one
fold — a model cannot identify a donor it never saw (scratch NOC=1 EM ≈ 0). The pretrained 0.99
is pure memorisation/leakage. Donor identification is simply not a well-posed target under
donor-disjoint cross-validation; NOC count (above) is.

## Reproduce

```bash
# build data (once)
python code/data/prepare_data_set.py
python code/features/enrich.py code/data
# well-posed NOC-count CV (leak-free)
python code/kfold_donor_slot.py --ckpt /path/Donor-Slot_Set_Transformer.pt \
    --data-dir code/data --task count --init scratch
# variants: --init pretrained (uses the ckpt) ; --task donor (donor-ID, ill-posed)
```
Outputs: `results/kfold_donor_slot/{kfold_metrics_<task>_<init>.json,
per_fold_scores_<task>_<init>.json}` (gitignored).
