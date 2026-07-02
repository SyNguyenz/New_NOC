# Increment 5 — Residual-aware (peeling) decoder for the N5 oracle wall

## Why (evidence chain, all measured 2026-06-08, no-train probes)

The binding constraint for "oracle > 0.9 at every NOC" is the **N5 oracle wall** =
combinatorial-generalization / combo-overfitting (F29; train N5 oracle ~1.0 → held-out-combo
dev ~.57). The current best in-domain arm `inc4_p3_irm` reaches only **N5 oracle .726** (full test).

A series of no-train probes localised the lever and bounded it:

| probe | result | lesson |
|---|---|---|
| `quick_peel.py` (oracle-peel, GT others) | buried (neural≥16) → top-5 **.65** | mechanism real: removing combo context recovers buried minor donors |
| `greedy_peel.py` (naïve greedy, no GT) | N5 set-EM **.726→.519** | naïve peel NET-NEGATIVE — breaks easy picks (≤5 bucket 1.00→.91) |
| `gated_peel.py` (cosine tail + gate τ=0.8) | **.726→.758** (+3.2pp) | **gating is mandatory** (keep neural's confident picks); cosine recovers only 27–34% of buried |
| `neural_peel.py` (FROZEN model on residual) | **.726** (≈baseline, < cosine) | frozen model is **OOD on residuals** (trained on full mixtures only) → a residual scorer must be **TRAINED** |

**Conclusion:** the peeling mechanism works, but (a) the scorer is the bottleneck and (b) a frozen
model cannot score residuals. The only way past the no-train ceiling (.758) is to **train the model
to score residuals end-to-end**. That is Increment 5.

## What (design — lowest-risk, evidence-aligned)

**Residual-aware training + gated recursive-peel decode.** Do NOT change the architecture (P6 slot
set-prediction already collapsed, C8). Instead change the *training distribution* and the *decode*:

### Training: residual augmentation (`build_residual_aug.py`)
For each NOC≥2 training mixture with true donor set S, generate a **residual copy**: subtract a random
subset R⊂S of donors' fitted contributions (NNLS on reference profiles G in linear RFU space — the
**exact** operation the decode uses), re-tokenize the residual, and supervise the model to predict the
**remaining** donors S∖R. The augmented training set = original ⊕ residual copies. The model is trained
(unchanged harness, base flags) on the union.

This single change does three things at once:
1. **Fixes OOD-on-residuals** (`neural_peel` failure): the model now sees residual/low-template
   profiles during training, so it can score them at decode.
2. **Per-donor combo-invariance** (the consistency lever, Veitch 2021 in spirit): a donor must be
   identified whether it appears in the full mixture or in a residual with random other donors removed
   → the per-donor decision is decoupled from *which* other donors co-occur → attacks combo-overfit.
3. **Curriculum of N1-like subproblems** (OMP / successive-interference-cancellation, Mallat 1993 /
   Verdú 1998): a deep residual is a low-NOC problem, where oracle is ~1.0 and generalization is solved.

Residual definition matches the decode EXACTLY (NNLS fit of the peeled subset to the full mixture, then
subtract) so train and inference are consistent — this is why `neural_peel`'s OOD failure is repaired.

### Decode: gated recursive peel (`eval_peel_decode.py`)
At inference, K given (oracle) or from the count head (deployable):
1. `P_full = model(mixture)`; **keep** donors with prob ≥ τ as confident picks (τ=0.8, measured best).
2. Subtract confident picks (NNLS) → residual → re-tokenize → `model(residual)` → pick the next donor.
3. Repeat until K picks. **Gating preserves the easy picks** (the net-negative fix); the residual scorer
   — now trained, not frozen — contests only the uncertain slots where buried minor donors live.

Plain top-k oracle is also reported (no-regression guard N1/N2/N3, and to isolate decode vs training gain).

## Arms (judge on DEV N5 oracle + REAL test; guard N1/2/3 like prior increments)

Base = adopted **pe_s3 + sparse** (tok8 · periodic σ0.3 · isab++ · per_donor · aux · sparse_attn) — same
base as Inc4 P1–P4. Only the training data (residual-augmented) and the decode (gated peel) differ.

- **P1 `inc5_res_rand1`** — residual aug peels **1 random** donor per NOC≥2 sample (gentle curriculum,
  closest to the gated decode which peels ~0.43 slot/sample on average).
- **P2 `inc5_res_randk`** — residual aug peels a **random 1..K-1** subset (fuller curriculum: teaches the
  model to score deeper residuals, matching multi-step peels at decode).

Both are decoded identically (gated peel τ-sweep {0, .7, .8, .9}); `eval_peel_decode.py` writes a `peel`
block into each arm's metrics.json: per-NOC oracle for **plain top-k** vs **gated peel**, on DEV and REAL.

A 3rd machine is **not** needed — 2 independent trainings. One machine can run both sequentially
(`RUNS=inc5_res_rand1,inc5_res_randk`); 2 machines parallelize (M1=P1, M2=P2).

## Scope / what this does NOT do
- Targets **Wall #1 (ID oracle)** only. The count head (Wall #2, F30) is unchanged here; the peel-stop
  criterion (number of peel steps until residual→noise) is reported as a secondary count signal but
  recalibrating count for EM>0.9 is a separate follow-up.
- Realistic expectation (from probes): learned residual scorer + gating ~**.80–.85** N5 oracle; the
  ~.91 ideal ceiling assumes oracle-peel recovery. Still likely <.9 alone — this is a bet that end-to-end
  training (unlike the frozen proxy) shifts the buckets at the source. Honest: 6 Inc4 arms didn't break
  the wall; this is a NEW train-time structure with a real shot, not a guarantee.

## Files (all NEW; existing code untouched)
- `build_residual_aug.py` — writes `data_res/` = base dir with residual-augmented train arrays.
- `eval_peel_decode.py` — gated-peel decode eval; writes `peel` block into the arm's metrics.json.
- `kaggle_run_increment1.py` — adds inc5 arms + prep wiring (dev_split → residual_aug → enrich) + M1/M2.
- Reuses: `train_set_transformer.py` (unchanged), `models/set_transformer.py` (unchanged),
  `make_insilico.xflat_to_tokens`, `features.enrich`, `train_set_transformer.build_pgnoc_refs`.
