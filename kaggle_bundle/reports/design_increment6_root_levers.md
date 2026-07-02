# Increment 6 — ROOT anti-memorization levers for the combinatorial-generalization wall

## Why (the root, restated)

The binding constraint for "oracle > 0.9 at every NOC" is the **N5 oracle wall** = combinatorial /
compositional generalization (F29/F30): train N5 oracle ~1.0 → held-out-combo DEV ~0.65, real ≈ DEV.
Each donor is individually seen; the model fails to **compose** known donors into a *novel* combo. The
mechanism is **decoder combo-memorization**.

Increments 4–5 attacked the **symptom** at decode time. Inc5 (residual-aug + gated peel) nudged real
N5 by +1–2pp (rand1 .742→.745, randk .656→.715) but **left DEV N5 oracle flat** (.6498), confirming
peeling reads the wall, it does not move it. Inc4 `p3_irm` is best in-domain (N5 .726) but worst OOD and
breaks the aux heads. **Conclusion: a decode patch cannot fix a representation that memorizes combos.**
Increment 6 = screening pass of published methods that attack the *learned representation*.

Dropped from this screen (user decision 2026-06-08): **Hướng A neural probabilistic genotyping**
(amortized latent-genotype + population prior) — highest root payoff but a decoder rebuild, not a
"quick test"; any quick proxy of it is decode-side = symptom again. Held as the moonshot if B/C/D fail.

## What (4 screening arms, each = ONE published method on the pe_s3+sparse base)

| arm | method (paper) | mechanism | code | cost |
|---|---|---|---|---|
| **D-mask** `inc6_maskp` | input dropout / masked modelling as implicit regularizer (arXiv 2601.22450; 2501.14687) | randomly drop a fraction of valid input peaks each train step → label (donor set) unchanged → decoder must infer donors from a random peak subset, **can't template-match a memorized combo** | `train_set_transformer.py --mask_peaks 0.15` | cheap |
| **D-andmask** `inc6_andmask` | ILC **AND-mask** (Parascandolo 2020, arXiv 2009.00329) | keep only cls-gradient components whose **sign agrees across {low-NOC, high-NOC} environments** → cls features that help BOTH easy (generalizing) and hard (memorizing) strata survive. Invariance **without** a penalty term, so the φ/attr aux heads stay intact (the thing IRM broke, F23) | `train_set_transformer.py --and_mask` | ~2× (2-env backward) |
| **C slot-fix** `inc6_slotfix` | **DETR** `eos_coef` + slot diversity (Carion 2020) | the P6 per-contributor slot binder collapsed (C8) because K=8 slots but ~85% of targets are ∅ → ∅ dominates the CE and every slot predicts ∅ (under-count). Down-weight ∅ in the matched CE + penalize duplicate slots → contributors bind to distinct slots | `train_p6_slot.py --slot_fix` (null_coef 0.1, diversity 0.05) | cheap-ish |
| **B meta** `inc6_meta` | **Reptile** (Nichol 2018) episodic finetune, proxy of Lake 2019 meta-seq2seq (arXiv 1906.05381) | each episode fast-adapts (k inner SGD steps) on samples sharing a target donor across **varying co-contributor contexts**; Reptile outer step moves base weights toward the adapted ones → pressures each donor to be encoded as a **context-invariant composable unit** rather than a per-combo template. Warm-starts from `inc5_res_rand1_seed42` | `train_inc6_meta.py --init_from …` | cheap (finetune) |

### Honest scope (C8 discipline)
- **B is a proxy, not full meta-learning.** Reptile averages adapted weights; it lacks FOMAML/MAML's
  explicit query objective. If DEV N5 oracle moves → invest in the strong FOMAML version (support/query
  **combo-disjoint** episodes). If flat → meta-finetune-lite is insufficient; the lever is a heavier
  episodic re-train. NOTE the data reality (F29): at N5 each combo is ~1 sample, so a faithful
  support/query split must be at the **donor** level (compose donors), not the combo level — this arm
  builds episodes by donor for exactly that reason.
- **C** applies the DETR remedy for *supervised closed-set* slot collapse (eos_coef), **not** the
  DINOSAUR target-encoder fix (that is for self-supervised reconstruction — wrong setting here).

## How (run)

```
# 4 arms, seed 42, 3 machines (M1 = both D arms, M2 = slot-fix, M3 = meta):
MACHINE=M1 SEEDS=42 python kaggle_run_increment1.py
MACHINE=M2 SEEDS=42 python kaggle_run_increment1.py
MACHINE=M3 SEEDS=42 python kaggle_run_increment1.py     # needs results/inc5_res_rand1_seed42 present
```
`measure_insilico_oracle.py` is auto-invoked (D arms) / self-invoked (C, B) → writes the
`generalization` block (per-NOC oracle/count on DEV/train/real) into each metrics.json.

## Judge (the ONLY readout that counts)

**DEV per-NOC oracle, combo-disjoint** (base ≈ 0.65 at N5, train ≈ 0.97). NOT test EM after peel — Inc5
proved that moves while DEV stays flat. A lever "works" only if **DEV N5 oracle rises vs base** while the
**N1/N2/N3 guard holds** (must stay ≥ .96). Because N4/N5 single-seed variance is ±.18 (F14/F16 trap),
treat every number here as a **direction-finder, not a conclusion** — re-run the winner at 3 seeds for a
CI before logging anything to `reports/empirical_findings_log.md` (consent first, [[conclusion-discipline]]).

## Batch-2 arms (added after the no-train feasibility screen; `feasibility_inc6{,b}.py`)

The probes returned GO for SAM and motivated building the remaining map branches. Same base, same
DEV-N5 judge.

| arm | method (paper) | mechanism | code | flag |
|---|---|---|---|---|
| **SAM** `inc6_sam` | Sharpness-Aware Minimization (Foret 2021) | two forward-backward/step: ascend to w+ρ·g/‖g‖ then step with the perturbed-point grad → FLAT minima generalize OOD/combinatorially. Probe: novel-N5 minimum 2.1× sharper than seen | NEW `train_inc6_sam.py` (isolated; core losses cls+noc+aux, reject omitted = irrelevant to ID ranking) | `--sam_rho 0.05` |
| **masked-pretrain** `inc6_maskpre` | masked self-supervised pretrain (SAINT/MAE-style) | N epochs reconstructing masked peak features from the ISAB context BEFORE supervised ID → encoder warmed toward context/combo-invariant features (stronger cousin of D-mask) | `train_set_transformer.py` | `--masked_pretrain_epochs 20` |
| **VIB** `inc6_vib` | variational information bottleneck (Alemi 2017) | reparam each donor latent D_i~N(μ_i,σ_i) + KL in `PerDonorDecoder` → keep only info to decide donor i, DROP combo nuisance = "disentangled donor latents" (binding problem) | `models/set_transformer.py` + train loop | `--vib --vib_weight 1e-3` |
| **FOMAML** `inc6_fomaml` | first-order MAML (Finn 2017) | episode adapts on donor d's SUPPORT, outer step along the QUERY-loss grad (held-out context-disjoint d samples) → explicit "identify d in a NOVEL context" objective (stronger than Reptile B-meta) | `train_inc6_meta.py --fomaml` | warm-starts inc5 base |

NO-GO from the screen (NOT built): A (ref-likelihood deconvolution), soup/SWA, Spectral-Decoupling, TTA/TENT.

3-machine split (8 arms): **M1**=maskp,andmask,sam · **M2**=vib,maskpre,slotfix · **M3**=meta,fomaml
(M3 warm-starts from `results/inc5_res_rand1_seed42`). Run `MACHINE=M1|M2|M3 SEEDS=42 python kaggle_run_increment1.py`.

## Smoke status (2026-06-08, local tok3/tok8 / synthetic) — ALL 8 arms pass end-to-end
D-mask · D-andmask (custom AND-mask backward, aux Kendall grads preserved) · C slot-fix (eos_coef +
diversity, grad→slots) · B meta (Reptile + self-measure) · **SAM** (2-step) · **masked-pretrain**
(recon_mse ↓) · **VIB** (+33k params, KL term) · **FOMAML** (query-loss ↓). Base regression unchanged
(params 1,442,622 with vib off). Root↔kaggle_bundle synced.
