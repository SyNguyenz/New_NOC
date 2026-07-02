# Increment 7 — Mass-preserving aggregation (the F31 root lever)

## Why (F31, 3-seed, logged)
The N5 oracle wall is **major-peak (height) mass dominance**, not info-limit / combo-memorization / physical floor:
- **Info present**: 99.9% of N5 contributors RANKABLE (private allele present); 99% of DEV misses RANKABLE (`probe_cause_decompose.py`).
- **Encoder washes the minor, REVERSIBLY**: attr_head on a deep-miss minor's OWN private peaks = **.07 on the full set → .81 when the MAJOR peaks are masked** (random same-count removal only .24; `probe_context.py`, 3-seed). The minor's signal is washed by mixing with the major-dominated peak context, not lost.
- **It's the MASS of the major MAJORITY, not the height FEATURE**: neutralising the height features at input recovers only **.07→.16** (`probe_height_decouple.py`) — far below the .81 major-removal ceiling. So a height-decoupled key is INSUFFICIENT; the fix must be in the AGGREGATION.
- **Decoder is competition-bound, not bad** (beats honest linear-H by +.15; `probe_root_levers.py`) — secondary lever, deferred.

Mechanism: ISAB++'s inducing-point compression (`mab0`) pools the N peaks into each inducing point by a **softmax-over-peaks WEIGHTED MEAN**. With many tall major peaks and few faint minor peaks, the normalized mass goes to the majors → the minor's encoded H is averaged away. This is the cardinality/assignment-mass erasure named in Fischer & Gärtner 2024 (arXiv 2407.04170) and the massive-activation/attention-sink literature (arXiv 2603.05498).

## What (the lever)
**Mass-preserving inducing compression** = replace `mab0`'s weighted-mean with a **scaled weighted SUM** under **competitive slot assignment**:
- softmax over the INDUCING points (each peak distributes its mass over inducing points), NOT over peaks;
- each inducing point aggregates its assigned peaks by a scaled weighted **sum** (not divided by the per-slot mass), normalized by √(N_valid) with a learned scale.
- A distinctive minor peak can claim its own inducing point and is preserved instead of washed.

Grounded: Fischer & Gärtner 2024 (arXiv 2407.04170, scaled-weighted-sum for cardinality generalization) + AttSets (Yang 2018, arXiv 1808.00758). `mab1` (X attends back to H) stays standard so per-peak outputs keep full context.

## Code (drop-in, +2 params)
- `models/set_transformer.py`: new `MassMABpp` + `MassISABpp`; `SetTransformerMixture(mass_pool=True)` swaps ISAB++ → MassISABpp (asserts encoder=isab++). Adds only the 2 learned scales.
- `train_set_transformer.py`: `--mass_pool` flag → cfg → model.
- `measure_insilico_oracle.py`: builds with `mass_pool` (eval must match).
- Smoke: forward/backward finite on random + real data, grad norm ~19, cls std ~1.0 (healthy init), byte-identical root↔kaggle_bundle.

## Arm + how to run (≤3 machines)
One arm, **3 seeds, one per machine** (N5 is single-seed-noisy → take the CI up front):
```
MACHINE=M1 SEEDS=42 INSILICO_W=/kaggle/input/<ds> python kaggle_run_increment1.py
MACHINE=M2 SEEDS=43 INSILICO_W=/kaggle/input/<ds> python kaggle_run_increment1.py
MACHINE=M3 SEEDS=44 INSILICO_W=/kaggle/input/<ds> python kaggle_run_increment1.py
```
`inc7_masspool` = pe_s3 + sparse + `--mass_pool`. Each run self-invokes `measure_insilico_oracle.py` (writes the generalization block).

## Judge (conclusion-discipline)
- **Primary**: DEV **N5 oracle** vs base ~.65 (3-seed CI). Ceiling from the major-removal probe ≈ .81 → a real win lands between .65 and ~.81.
- **Guard**: DEV N1/N2/N3 oracle must stay high (no regression); attr_head N5 acc should rise if H preserves minors.
- **Secondary**: N4 oracle, real-test per-NOC, reject AUROC.
- Only conclude N4/N5 effects from the 3-seed CI (F14/F16 trap). Append to `empirical_findings_log.md` only with consent.

## Expectations / risks
- If DEV N5 oracle rises toward ~.75–.81 → mass-dominance confirmed as the operative lever; then add the light decoder competition term (lever B) for the near-miss half.
- Risk: slot-style assignment could destabilize (cf. P6 slot collapse — but that was DECODER set-prediction with Hungarian matching; this is ENCODER pooling with SetNorm clean-path, far milder). If it underperforms base in-dist, it's a clean negative on the aggregation hypothesis.
- The ~17% deep-miss tail beyond the ceiling = faint-but-present low peaks the model underweights (same bias, milder) + a small phi<0.04 sliver near AT — not addressed here.
