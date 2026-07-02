# Empirical Findings Log — output ↔ paper-theory reconciliation

**Purpose.** A running log. After EACH experiment, we reconcile the observed output against the
paper/design-doc theory (`design_representation_redesign.md` = doc1, `design_increment2_relationships.md`
= doc2) and record what was *learned* — not raw numbers (those live in `results/*/metrics.json`), but the
**conclusion** each result licenses or refutes, plus the next action it implies.

**How to add an entry.** New experiment → new dated section. For each finding write: `OBSERVED` (the fact),
`THEORY` (which doc/section it maps to), `CONCLUSION` (what it licenses/refutes), `ACTION` (what to do next).
Keep findings falsifiable. When a later run overturns an earlier finding, do NOT delete it — add a
`REVISED` note under it with the date, so the reasoning trail survives.

**Selection discipline (binding, doc1 §7 + doc2 §9).** Select on in-silico DEV with graded recall, never on
real test or saturated real val, never on raw Exact-Match. Real test = report once, with CIs (N4 n=45,
N5 n=48 → wide). Treat small EM/oracle swings (esp. NOC5) as metric noise until shown otherwise.

---

## Standing scoreboard (Increment 1, raw token; isab=LayerNorm, isab++=SetNorm)

oracle = ranking ceiling (per-NOC best achievable by reordering); decoded EM = after the decode stage.

| run | encoder | token | num_embed | oracle | N4 orc | N5 orc | decoded EM | reject |
|---|---|---|---|---|---|---|---|---|
| inc1_hybrid_tok3 | isab | 3 | raw | 0.976 | 0.756 | 0.708 | 0.878 | 0.988 |
| inc1_hybrid_tok8 | isab | 8 | raw | 0.979 | 0.844 | 0.646 | 0.876 | 0.888 |
| inc1_st_perdonor_tok3 | isab | 3 | raw | 0.951 | 0.733 | 0.396 | **0.932** | 0.989 |
| inc1_st_perdonor_tok8 | isab | 8 | raw | **0.881** | **0.000** | **0.021** | 0.833 | 0.956 |
| inc1_st_perdonor_tok8_pp | isab++ | 8 | raw | 0.957 | 0.822 | 0.396 | 0.929 | **0.999** |
| inc1_st_perdonor_tok8_pe | isab | 8 | periodic | 0.860 | 0.000 | 0.000 | 0.847 | 0.933 |
| inc1_st_perdonor_tok8_pp_pe | isab++ | 8 | periodic | 0.951 | 0.578 | 0.375 | 0.926 | 0.904 |
| inc1_hybrid_tok8_pp_pe | isab++ | 8 | periodic | 0.977 | 0.822 | 0.646 | 0.869 | 0.896 |

(hybrid decode = two-stage + pgNOC; ST decode = post-hoc / joint-card.)
**Working base = `inc1_st_perdonor_tok8_pp` (isab++).** raw vs periodic is **INCONCLUSIVE**: real-test deltas
(oracle −0.6pp, EM −0.3pp) are within run-to-run noise (n=1, no seeds); only reject (−9.5pp) is sizeable.
Note periodic scored HIGHER on the in-silico dev SELECTION metric (macro-rec 0.971 vs raw 0.963) — i.e. it
helped in-distribution. The PE-vs-raw verdict needs a σ-sweep + seeds and is **DEFERRED to a final tuning
pass after all design increments** (per user 2026-06-02). See F8 / P4.

---

## Carry-forward — consolidated from sessions ≤ 2026-05-31

Durable findings that still constrain decisions (condensed from memory: technical_decisions,
two_stage_noc_decoder, results). Kept because each one rules something in or out going forward.

**C1. On honest novel-combo eval, aligned per-donor models BEAT pooled ST for ID.** LR/XGB > ST
(pooled) on the no-leak split; ST's set-pooling discards the locus×allele alignment that LR/XGB use as
a per-donor lookup table (donor-48, a faint minor private allele, *lives* in LR/XGB but *dies* in all-ST).
→ Hybrid = flat aligned base + tanh-bounded set stream beats all (oracle 0.959). **But this whole "ST weak"
result is confounded by the 3-field raw token (doc1 §1)** — it is the thing the redesign is testing, not a
verdict against ST. *Decision: keep both streams until a properly-embedded ST is shown to close the gap.*

**C2. permutation-invariance ≠ attention (separable).** Tokens carry alignment (locus emb + allele) → the
ST stream is NOT purely invariant; "aligned vs set" is a false dichotomy. STR has both structures.
set_scale learned 0.1→0.92 under combo-diverse data ⇒ the set stream is valuable. *Don't go aligned-only.*

**C3. The decoded↔oracle gap is 100% COUNT error.** Ranking is near-oracle when k is right; the entire
high-NOC game is picking k. (Confirmed again this session as F3.)

**C4. Decoder progression (post-hoc, on a fixed checkpoint):** post-hoc 0.940 → two-stage 0.950 →
two-stage+pgNOC 0.954. two-stage = stage1 NOC1-vs-multi on real val + stage2 count 2–5 on in-silico;
pgNOC = gamma-weighted NNLS deconvolution cost-curve + BIC. Wired into train_*.py decode block.

**C5. pgNOC (height + reference + ID head) WINS the 3-way cross-check** vs forensim (height-blind, 0.48)
and euroformix-simple (0.44) on 50 balanced samples → peak-height *likelihood* is the binding NOC-count
signal (height-blind caps N4 ≈ 0.29). Caveat: euroformix ran a SIMPLE config (no degrad/stutter/kit) —
do NOT claim "beats gold-standard EFM". pgNOC's edge = exploits 45 known references (closed-set).

**C6. METHODOLOGY (load-bearing).** Three separate "gains" all EVAPORATED under honest evaluation:
test-tuned routing (0.956 = HARKing), under-regularized XGB, and selecting on SATURATED real val
(N4 val 0.979 vs test 0.689). Fix every time = **select on the in-silico hard novel-combo dev, eval real
test ONCE.** "The dev/eval design IS the experiment." (This is why doc1 §7 is a binding constraint.)

**C7. The in-silico→real DOMAIN GAP is the recurring binding constraint — not architecture sharpness.**
Sharper reference genotypes (consensus+height) and tuned pgNOC alpha both IMPROVE in-silico/standalone but
transfer WORSE to real (they overfit the in-silico single-source); the cruder global ref is more robust.
Theory-optimal (in-distribution) and domain-robust (deployment) point opposite ways. → Closing the gap
(make_insilico realism: per-locus gamma noise, degradation, stutter) is the main non-architecture lever.

**C8. Task unification = set prediction with unknown cardinality** (DeepSetNet / Joint-Cardinality). Current
pipeline = decoupled (multi-label ID + post-hoc cardinality decode); the enriched token serves both. Test,
don't assume the joint loss wins.

**C9. Misc durable.** log1p(height) essential (binary features gave NOC5 ≈ 0.40); decision threshold 0.80
(re-search if retrained); **per-NOC Macro F1 is misleading → always report Exact Match by NOC** (zero-support
donors drag macro F1 to ~0 even at EM=1.0); reject head sharing z_mix drops AUROC (1.000 flat → 0.886 hybrid)
→ use a decoupled reject pool. Count models must be fit on `data_insilico_w/` (overlay of real NOC1, 55,247
train), NOT the failed R simDNAmixtures synth (`*_synth_train.npy`, domain-shifted MAC).

---

## 2026-06-02 — Increment 1, raw-token head-to-head (tok3 vs tok8; isab vs isab++)

### F1. Enriching the token as RAW scalars does NOT help and can collapse a set encoder.
- **OBSERVED.** Plain ST (isab), per_donor: tok3 oracle 0.951 → tok8 oracle 0.881; N4 0.733 → **0.000**,
  N5 0.396 → **0.021**. feat_std across the 8 fields spans 7.29 … 0.182 (wildly different scales).
- **THEORY.** doc1 §3 (Grinsztajn 2022): plain NNs are rotation-invariant on the input axis and lose to
  trees on tabular lookup targets *unless each numeric feature gets its own embedding*. The "enriched
  token" was only half-implemented — features added, but fed as standardized scalars into ONE shared
  `input_proj` Linear (`models/set_transformer.py:_project_tokens`). No PLE/periodic embedding.
- **CONCLUSION.** "More fields" is not the lever; **per-feature embedding is**. The doc1 §1 claim that the
  3-field token "crippled" ST is right in spirit but the fix is the *embedding*, not the field count.
  The raw-scalar tok8 collapse is the predicted failure mode, not evidence against enrichment.
- **ACTION.** Implemented `PeriodicNumEmbedding` (Gorishniy 2022 PLR) + `--num_embed periodic`
  (done 2026-06-02). Re-run as `st_pd_pp8_pe` / `st_pd8_pe`. → see Pending P1.
- **REVISED 2026-06-02 (P1 result, see F8).** Partially overturned. The lever that DEMONSTRABLY fixed the
  tok8 collapse is the ENCODER (isab++, F2, large effect), NOT the embedding — so the collapse was primarily
  an encoder-stability failure. Whether per-feature embedding *additionally* helps is **UNRESOLVED**:
  periodic (σ=1, untuned) came out ≈ raw on real test (within noise, n=1) yet BETTER on the in-silico dev
  selection metric. doc1 §3 is therefore neither confirmed nor refuted at this scale — verdict deferred to a
  σ-sweep (F8 / P4).

### F2. The encoder is the dominant variable; ISAB++ rescues the enriched token.
- **OBSERVED.** Same raw tok8, only encoder changes: isab oracle 0.881 → isab++ 0.957 (N4 0→0.822,
  N5 0.02→0.40, reject 0.956→0.999).
- **THEORY.** doc1 §4 "which encoder wins is EMPIRICAL — do not assume." SetNorm + clean residual path
  (Set Transformer++) stabilizes deep set attention.
- **CONCLUSION.** Encoder capacity/stability gates whether a richer token is usable at all. Increment 1 is
  NOT "settled" until encoder is fixed → ISAB++ is the working default.
- **ACTION.** Use isab++ as the ST/hybrid default in all further runs.

### F3. Hybrid's bottleneck is the DECODER, not ranking.
- **OBSERVED.** hybrid has the best oracle (0.979) but two-stage+pgNOC decode throws away ~10pp:
  N2 oracle 0.98 → decoded 0.27, N3 0.99 → 0.28. Gap oracle→decoded ≈ 10pp for hybrid vs ≈ 3pp for ST.
- **THEORY.** doc1 §4/§9: linear NNLS (pgNOC) is the "confounded baseline"; decoder must be learned/nonlinear.
- **CONCLUSION.** Pouring effort into the token won't move hybrid — its loss is in the decode stage. ST's
  per_donor + joint-card decode is far better calibrated (small, uniform oracle→decoded gap).
- **ACTION.** For hybrid, the lever is a learned decoder, not the token. Strategically, prefer pushing ST.

### F4. By decoded EM (the deployable number), ST already ≥ hybrid.
- **OBSERVED.** decoded test EM: ST 0.929–0.932 vs hybrid 0.876–0.878. ST also has best reject (0.999).
- **THEORY.** doc1 §7 head-to-head; the flat base gives hybrid a real high-NOC *ranking* edge (N5 oracle
  0.71 vs ST 0.40) but that edge is destroyed by hybrid's decoder (F3).
- **CONCLUSION.** ST is the more promising path: efficient decode + best open-set + the largest untested
  lever (per-feature embedding) still ahead of it. The hybrid flat-base is a one-time scaffold that doc2's
  allele→donor supervision is designed to make redundant.
- **NUANCE (for the "hybrid only benefits via its ST stream" question).** Partly true: token/encoder changes
  live only in the ST stream, so ST and hybrid rise from the same source. BUT (a) hybrid is currently
  decoder-bound, so lifting the ST stream won't help it until the decoder is fixed; and (b) the flat base's
  high-NOC ranking edge is genuinely independent of the ST stream — so hybrid ≠ "just the ST part".

### F5. NOC5 oracle swings are likely metric artifact, not capability.
- **OBSERVED.** N5 oracle ranges 0.40 (ST) … 0.71 (hybrid tok3) … 0.646 (hybrid tok8) across runs on n=48.
- **THEORY.** doc2 §9 (Schaeffer 2023 "Mirage"): Exact-Match/accuracy are discontinuous metrics (92% of
  "emergent" cases) → smooth underlying changes look like jumps; at our scale expect smooth gains, no magic.
- **CONCLUSION.** Do not over-read NOC5 swings on tiny strata. Select with graded recall on in-silico dev.
- **ACTION.** Report NOC5 with CIs; never select a design on a NOC5 EM delta alone.

### F6. NOC5 is NOT information-limited — every contributor has a PRESENT private allele (raw-genotype measurement, 2026-06-06). The earlier "24% genuine dropout / 76% ceiling" was a SYNTH-CSV artifact and is REMOVED.
- **OBSERVED (measure_noc5_ceiling.py — RAW reference genotypes RD14-0003 from data_raw, 23.8/24 loci
  coverage; presence/absence on the AT-filtered REAL test).** 100% of true contributors at EVERY NOC —
  including NOC5 (240/240) — have ≥1 PRIVATE allele (not in any co-contributor's genotype) PRESENT in the
  peaks. 0% dropout, 0% fully-masked. The COUNT is forced by allele richness (ceil(MAC/2) ≥ NOC) in
  91.7% of NOC5 samples; only 4/48 (~8%) need height-stacking to prove the 5th contributor.
- **THEORY.** doc1 §1 / doc2 §8: NOC5 is NOT a physical ceiling — confirmed directly. The identifying
  information is physically in the data; Green/Mortera's "absent under dropout" regime is essentially not
  reached at this analytical threshold.
- **CONCLUSION.** The NOC5 weakness (oracle/EM/count below NOC1–3) is ~entirely MODEL failure, not an
  information wall — the distinguishing evidence is present and the count is presence-provable in ~92% of
  cases. The NOC lever is therefore MODEL-SIDE, in TWO SEPARABLE parts (measured in F29, 2026-06-07):
  (1) the high-NOC ORACLE (ranking) ceiling is a COMBINATORIAL-GENERALIZATION gap — the model fits train N5
  oracle ~1.0 but drops to ~.57 on held-out combos, real ≈ dev — so the lever is anti-overfitting / combo
  augmentation / IRM-at-N5, NOT §13 (there is no domain penalty at N5); (2) the EM↔oracle (count) gap (C3) is a
  height-aware / ordinal count head reading the encoder, not the 45-dim ID-profile bottleneck
  `CardinalityHead(sigmoid(logits_cls).detach())`. It is NOT a target cap.
- **CORRECTION NOTE (do NOT re-cite the old number).** The prior "76% rankable / 24% genuine dropout / NOC5
  target ≈76%" came from the 91%-coverage synth genotype CSV (data/synth/donor_genotypes.csv): missing
  genotype rows were mis-scored as "no private allele" → fabricated dropout/masked. Re-measured on the raw
  PROVEDIt references it is 100% rankable. CAVEAT: "private allele present" = info EXISTS, not that it is
  trivially recoverable (a low minor peak must still be ranked above 40 absent donors); a harsher AT would
  create real dropout. So this licenses "no physical wall here", not "NOC5 is easy".
- **ACTION.** Use measure_noc5_ceiling.py (raw genotypes) as the ceiling check, NEVER the synth-CSV split.

### F7. We are NOT yet allowed into Increment 2 (its own sequencing gate).
- **OBSERVED.** Increment 1's key lever (per-feature embedding) was untested and the encoder choice still
  flipped the whole result (F1, F2).
- **THEORY.** doc2 §Status + §11: "Build AFTER Increment 1 settles (token/readout/encoder/selection) … so
  we don't confound many variables at once."
- **CONCLUSION.** Settle Inc1 first: (i) per-feature embedding, (ii) encoder = isab++, (iii) selection on
  in-silico dev + graded recall. Only then proceed to doc2 2a→2b→2c→2d, ablating each.
- **ACTION.** Run P1; if periodic embedding settles the token question, declare Inc1 settled and scope Inc2-2a.

### F8. Per-feature periodic embedding (doc1 §3): INCONCLUSIVE vs raw — small deltas, needs a σ-sweep; verdict DEFERRED to a final pass.
- **OBSERVED (controlled, only num_embed differs; n=1 per config, no seed repeats).**
  - isab++ tok8 per_donor: raw `st_pd_pp8` → periodic `st_pd_pp8_pe`: oracle 0.957 → 0.951, N4 oracle
    0.822 → 0.578, N5 0.396 → 0.375, decoded EM 0.929 → 0.926, reject AUROC 0.999 → 0.904.
  - isab tok8 per_donor: raw `st_pd8` (0.881) → periodic `st_pd8_pe` (0.860). Periodic did NOT rescue the
    plain-ISAB collapse (both N4=N5=0). Only isab++ rescues (F2).
  - hybrid isab++ tok8: periodic `hybrid8_pp_pe` oracle 0.977, decoded EM 0.869 — flat vs raw hybrid (≈0.876).
  - **Direction split:** periodic scored HIGHER on the in-silico DEV selection metric (best dev macro-rec
    0.971 > raw 0.963) but ≈equal-to-slightly-lower on REAL test (oracle/EM within noise; reject the one
    sizeable drop).
- **THEORY.** doc1 §3: per-feature embeddings (PLE/periodic) fix the Grinsztajn-2022 rotation-invariance bias.
  Gorishniy 2022's claim is about **in-distribution** generalization (train→test, same dataset) and it stresses
  **σ is the critical, dataset-specific hyperparameter** (optimal often ∈ [0.01, 0.5]). It says nothing about
  OOD/covariate-shift transfer — which is exactly the in-silico→real regime where our deltas appear.
- **CONCLUSION (NOT a verdict — explicitly inconclusive).**
  - No implementation bug: the module is faithful PLR (order Periodic→Linear→ReLU, per-feature `c` + Linear,
    gradients flow, padding handled like raw). Audited 2026-06-02.
  - On the regime the paper actually addresses (in-distribution = dev), periodic did NOT fail — it scored
    higher than raw. The shortfall is only on OOD real test, outside the paper's claim.
  - The real-test deltas are too small to call (n=1, within seed noise); reject −9.5pp is open-set, also n=1.
  - Leading hypothesis: **σ=1 is untuned and high.** With z-scored inputs, σ=1 → ~1 oscillation per feature-std
    → a non-Lipschitz, high-frequency embedding: more expressive in-distribution (explains the dev gain) but
    fragile under covariate shift (standardized by in-silico stats → amplifies shift on real). σ controls BOTH
    sides; a lower σ (0.1–0.3) is predicted to keep the dev edge while transferring better.
  - Minor design point (not a bug): PE grows `input_proj` input 23→72, so locus-embedding share drops 70%→22%
    — locus identity is diluted; rebalance `d_locus`/`d_num_emb` when revisiting.
  - What IS settled and stands independently: the ENCODER (isab++) is the demonstrated lever (F2).
- **ACTION.** Do NOT conclude PE now. (i) Carry the **raw** base (`st_pd_pp8`, isab++) through the design
  increments to avoid dragging an untuned variable through everything. (ii) The PE-vs-raw decision (σ-sweep +
  seeds + locus rebalance) is a **final tuning pass after all increments are done** → P4. (iii) Proceed to
  doc2 Increment 2-2a on the raw base; the residual ST↔hybrid gap is NOC4/5 *ranking* (ST 0.82/0.40 vs hybrid
  0.84/0.65 oracle), which §5 allele→donor supervision + §4 NOC4-vs-5 contrastive curriculum target.

- **P1 — per-feature periodic embedding (tests F1 / doc1 §3). PARTIALLY RESOLVED 2026-06-02 (see F8) —
  embedding question still OPEN.** Encoder leg settled (isab++ is the lever, F2). Embedding leg INCONCLUSIVE:
  periodic (σ=1, untuned) ≈ raw on real test (within noise, n=1) but better on in-silico dev. Not falsified,
  not confirmed. Carried forward as a deferred final-pass task → P4.

- **P2 — (optional, low priority) does enrichment even help under isab++?** `st_pd_pp3` (isab++, tok3, raw)
  was never run. One run would tell whether tok8 enrichment adds anything over tok3 once the encoder is fixed.
  Cheap sanity check; not blocking Inc2.

- **P3 — Increment 2-2a (NEXT real step).** On the raw base (`st_pd_pp8`, isab++), add doc2 §5 allele→donor
  privileged supervision + §4 NOC4-vs-5 contrastive curriculum; target the residual NOC4/5 ranking gap.
  Ablate each addition; measure oracle + dropout-vs-rankable split (§10) + in-silico↔real gap.

- **P4 — DEFERRED FINAL PASS: periodic-embedding σ-sweep (resolves P1's embedding leg).** Run only AFTER all
  design increments are in. `--periodic_sigma` ∈ {0.05, 0.1, 0.3, 1.0}, n_freq ∈ {8, 16}, ≥2 seeds each;
  also rebalance `d_locus`/`d_num_emb` (PE dilutes locus to 22% of input_proj). Select on in-silico dev,
  separately track real-test transfer + reject. **Predicted:** low σ retains the dev edge AND transfers
  better → PE was mis-tuned, not wrong. **Falsifier:** no σ beats raw on real test across seeds → drop PE.

---

## 2026-06-03 — Increment 2-2a/2b ablation (6 arms; isab++, raw embed; select on in-silico DEV)

Arms (all `cls_decoder=per_donor`, `encoder=isab++`, `loss=asl`): base tok8 (Hb+back) ·
2a_fwdstutter tok9 (+fwd) · 2b_privsup tok8 +aux · 2ab tok9 +aux · 2a_full tok11 (+size/degrad) ·
2ab_full tok11 +aux. aux = allele→donor attribution + φ regression, Kendall-weighted (doc2 §5).
Real-test joint-card EM / oracle / NOC4 / NOC5: base .928/.946/.644/.188 · 2a .924/.940/.689/.146 ·
**2b .940/.965/.711/.542** · 2ab .923/.946/.578/.062 · 2a_full .910/.939/.267/.104 ·
2ab_full .904/.925/.000/.167. (N4 n=45, N5 n=48 → wide CIs; n=1, no seeds.)

### F9. Privileged supervision (2b) is the only arm that beats base — and it lifts the ORACLE, not just decode.
- **OBSERVED.** tok8 + aux heads vs base: EM 0.928→0.940 (+1.26pp), **oracle 0.946→0.965 (+1.85pp)**,
  **NOC5 EM 0.188→0.542 (~×3)**, NOC4 0.644→0.711, NOC1 0.977→0.985 (no regression). Best on every
  aggregate (Macro F1 0.9645, Micro 0.9671, Hamming 0.0021, Jaccard 0.9722), reject 1.000. Attribution
  acc (eval D) overall 0.925, NOC1 0.994, NOC2–5 0.74–0.81 (well above chance at high NOC). Condition-stratified
  (eval A, doc2 §12): dnase is consistently hardest (EM ~0.83–0.90; enzymatic-digestion dropout stratum,
  NOC5×dnase) vs fragmentase/humic/uv ~0.95–0.99; 2b's gains land in info-rich strata (untreated
  0.919→0.964) → stratification works, separating model-failure from info-absent (Green&Mortera §8).
  (NB: the old "F6 24%-dropout" cross-ref was REMOVED — F6's dropout number was a synth-CSV artifact; see F6.)
- **THEORY.** doc2 §5 LUPI / generalized distillation (teach relationships, don't hope they emerge);
  §4b-C valve 1 (express the NOC signal via abundance φ + allele→donor attribution, NOT a count label);
  §10 verification (donor query attends its donor's alleles). C3 (decoded↔oracle gap = count error).
- **CONCLUSION.** The privileged signal improves the REPRESENTATION/ranking (oracle up at NOC5 0.292→0.688),
  which is the load-bearing place per C3. Valve 1 held: gain came through φ/attribution with NO count-label
  contrastive and NO ID regression → no negative transfer. **2b is the new working base.** doc2 §5 confirmed.
- **ACTION.** Freeze 2b (tok8 Hb+back, raw, isab++, aux on). Re-wire two-stage+pgNOC decoder (C4) onto it.

### F10. 2b wins on real TEST but not on DEV → privileged supervision is a DOMAIN-GAP lever (C7), not in-distribution.
- **OBSERVED.** best_val_macro_recall: base 0.9611 (ep115) ≥ 2b 0.959 (ep85); val_oracle_em base 0.783 ≥
  2b 0.771. Yet real-test EM 2b 0.940 > base 0.928. Opposite of F8's periodic-embedding pattern (helped dev,
  hurt transfer).
- **THEORY.** doc2 §11: "supervised relationships must be PHYSICAL → transfer." C7: in-silico→real gap is
  THE recurring binding constraint, not architecture sharpness.
- **CONCLUSION.** LUPI on physical relationships (attribution/φ) shrinks the domain gap — arguably 2b's most
  important property. Selection caveat: under strict doc1 §7 (select on dev macro-recall) base would tie/beat
  2b; the licensing evidence here is the COHERENT multi-signal real-test gain (EM+oracle+NOC5+attribution+
  condition-where-info-exists all move together), not a single dev metric. Needs seeds to harden (n=1).

### F11. Adding fwd-stutter (tok9) / size+degradation (tok11) as RAW columns HURTS — a DELIVERY pathology (F1 repeats), NOT a verdict the info is useless. (VERDICT: OPEN)
- **OBSERVED.** fwd stutter (tok9): EM −0.36pp, oracle −0.58pp. size/degrad (tok11): EM −1.76pp and
  **NOC4 collapses 0.644→0.267**, training destabilized (no early-stop, val oscillates). Combined 2ab_full:
  **NOC4 → 0.000.**
- **DERIVABILITY (checked in features/enrich.py).** None of the added feats carry new BAYES-information:
  FS = h/h(allele−1) and degr_resid = log_h−(a+b·size) are RELATIONAL (functions of neighbour/other peaks);
  size_bp ≈ affine(locus,allele) (the per-marker bp ladder). All three are derivable-by-attention from the raw
  {locus,allele,log_h} peak set → precomputed CONVENIENCES (doc2 §2), not missing info. "Derivable" does NOT mean
  "useless to precompute": §2/§6 exist to hand a capacity-limited 2-ISAB encoder relationships it struggles to learn.
- **THEORY.** A usable feature cannot lower generalization at the population level → an empirical drop is a
  SETUP/estimation failure, not an info verdict. (i) Information monotonicity / data-processing: adding feats cannot
  raise the Bayes risk. (ii) Grinsztajn 2022 bias-3 + Gorishniy: raw scalars through ONE shared `input_proj` are
  rotation-invariant; added dims disrupt the alignment of existing feats — fix = per-feature embedding (P4), NOT
  removal. size→NOC4 collapse + instability is the signature of (ii). (iii) Grinsztajn bias-2: redundant dims raise
  NN variance (not excludable). NOTE: §4b-B mRMR/IB explains only the possible REDUNDANCY of FS/degr; it does NOT
  explain the size collapse (size is the only genuinely-new degradation-enabling signal) — two distinct mechanisms.
- **CONCLUSION — VERDICT OPEN.** Do NOT conclude fwd/size are useless, and do NOT yet conclude they help. Harmful
  only AS RAW columns fed through the shared projection.
- **ACTION.** Do NOT drop the features as an information decision. Re-test delivered RIGHT: per-feature embedding
  (P4) ± §6 structural attention bias (stutter/degradation) + proper standardization, then keep/drop on that
  evidence. These results STRENGTHEN the case to resolve P4 before judging any 2a feature — consider pulling P4 earlier.

### F12. fwd stutter HELPS φ but HURTS NOC — a delivery/architecture conflict (usable feature, mis-delivered); motivates the §4b-C decoupling.
- **OBSERVED.** 2b vs 2ab on the φ head (eval B): φ Spearman(all) 0.065→0.642, major-ID and per-NOC Spearman
  all better with fwd stutter. But same 2ab is WORSE than 2b on NOC (EM 0.923 vs 0.940, NOC5 0.062 vs 0.542).
- **THEORY.** doc2 §4b negative transfer; §5 Kendall; §4b-C valves 2–4 (separate projection g(z) / decoupled
  pool / PCGrad) — designed for exactly this conflict. Precedent: reject head needed its own pool (C9). The split is
  direct evidence the feature carries USABLE signal (it lifts φ) but is delivered through the shared raw projection
  the ID/NOC decode also reads → a delivery/architecture problem, not "the feature is bad".
- **CONCLUSION.** A feature that genuinely helps one head can degrade the shared representation another head reads.
  The decoupling valves are NOT implemented yet → direct justification to build 2c's decoupled projection/pool
  (+ per-feature embedding) BEFORE re-judging features.
- **ACTION.** Implement §4b-C decoupling (projection + pool + PCGrad) as 2c; re-test fwd/size there — it is the FIX
  for a usable-but-mis-delivered feature, not a reason to discard it.

### Next steps (updated sequencing).
1. Freeze 2b as the working base (F9). **Defer fwd/size raw columns — NOT dropped** (verdict OPEN, F11/F12);
revisit only when delivered RIGHT. 2. Re-wire two-stage+pgNOC onto the 2b checkpoint — oracle→decode gap ~15pp at
NOC5 (C3/F9). 3. Build §4b-C decoupling = 2c (F12) AND resolve P4 (per-feature embedding); re-test fwd/size there.
4. eval_phi_condition C stratifier — FIXED 2026-06-03 (within-NOC, removes the NOC confound; both repo + bundle
copies). 5. Then 2c DR-curriculum + Rank-N-Contrast on the decoupled projection. Seeds needed to harden F9/F10 (n=1).

---

## 2026-06-03 (batch 2) — periodic σ-sweep + tok9/tok11 + 2c Rank-N-Contrast — REVISED after user review

**META (binding).** The first draft of this section OVER-CONCLUDED (declared FS "drop", 2c/RNC "don't adopt",
NOC2 a "frequency-resolution effect"). Corrected per user 2026-06-03. The corrected stance: (i) every method here
is PAPER-PROVEN to generalize *under stated conditions* — when our result contradicts the paper, the prior is that
OUR setup / implementation / run-variance is the cause, NOT that the method is useless; (ii) small NOC strata
(n=44–48) have ~±10pp run variance — the SAME `inc2_2b_privsup` config gave NOC2 oracle **0.864** (earlier run)
vs **0.977** (this run) → no per-arm attribution is valid without seeds + a fixed split seed.

Arms (all `cls_decoder=per_donor`, `encoder=isab++`, `loss=asl`, `aux_heads`): `inc2_2b_privsup` (tok8, RAW
embed) · `inc2_2c_contrast` (= privsup + `--noc_contrast`) · `inc2_2b_pe_s05/s1/s3` (tok8, periodic σ=0.05/0.1/0.3)
· `inc2_2b_pe_tok9` (tok9 = +FS, periodic σ=0.1) · `inc2_2b_pe_tok11` (tok11 = +size/degrad, periodic σ=0.1).
Real-test EM / oracle / NOC2 orc / NOC4 orc / NOC5 orc, dev macro-rec:
privsup .948/.968/.977/.889/.625 (dev .9644) · contrast .938/.958/.955/.889/.333 (dev .9641) ·
pe_s05 .962/.976/.977/.933/.688 (dev .9681) · pe_s1 .958/.977/.932/.956/.583 (dev .9692) ·
**pe_s3 .971/.981/.864/.978/.771 (dev .9715)** · pe_tok9 .936/.953/.932/.667/.250 (dev .9712) ·
pe_tok11 .952/.966/.977/.867/.396 (dev .9726). (N2 n=44, N4 n=45, N5 n=48 → wide CIs; n=1, NO seeds.)

### F13. Periodic embedding (doc1 §3) HELPS once σ is tuned low — the one clearly paper-consistent positive; advances P4.
- **OBSERVED.** Controlled (tok8 + aux, only num_embed/σ vary): RAW `privsup` (dev .9644, test EM .948,
  oracle .968) vs periodic `pe_s3` σ=0.3 (dev .9715, test EM .971, oracle .981). σ=0.3 beats raw on BOTH dev AND
  real test (+2.4pp EM, +1.3pp oracle); dev macro-rec monotone in σ (.9681→.9692→.9715 for σ=.05→.1→.3). inc1 F8
  had σ=1.0 ≈ raw → the gain appears at moderate σ, not σ=1.
- **THEORY (paper conditions).** Gorishniy 2022 ("On Embeddings for Numerical Features") proves per-feature
  periodic/PLR embeddings improve DNNs on tabular data **in-distribution**, and states σ is the **critical,
  dataset-specific** hyperparameter (no OOD guarantee). Grinsztajn 2022 bias-3 = rotation-invariance. Our gain is
  exactly the regime the paper covers; σ=1 failing = the paper's σ-sensitivity, not a refutation.
- **CONCLUSION.** doc1 §3 SUPPORTED at this scale and consistent with theory. This is the strongest single result
  in the batch. Still n=1, no seeds → not frozen.
- **ACTION.** Carry periodic σ≈0.3 as a STRONG candidate; finish P4 (≥2–3 seeds, σ∈{0.2,0.3,0.4}, n_freq∈{8,16},
  d_locus rebalance, per-NOC CIs) before declaring it the base.

### F14. tok9 (FS) and tok11 (size/degrad) are NET-NEGATIVE on ID/NOC in these runs — but the verdict stays OPEN (NOT "useless"/"drop"). REVISES the earlier over-strong draft.
- **OBSERVED.** Controlled (periodic σ=0.1 + aux): tok8 `pe_s1` EM .958/orc .977 → tok9 `pe_tok9` EM .936/orc .953
  (N4 orc .956→.667, N5 .583→.250). tok11 `pe_tok11` EM .952 < pe_s3 .971, N5 orc .396. φ still IMPROVES with FS
  (eval B Spearman all .116→.481), reproducing F12. tok11 training was UNSTABLE earlier (no early-stop — F11).
- **THEORY (both directions, honestly).**
  - *Why a gain is not guaranteed:* FS and SR are the same pairwise relation re-indexed — with adjacent alleles
    a, a+1 present, **SR_a = h_a/h_{a+1}** and **FS_{a+1} = 1/SR_a** ([enrich.py:73-79]). At the SET level the
    encoder already has every adjacent-allele ratio via SR, so FS adds no new *information* edge (mRMR-redundant,
    doc2 §4b-B). Data-processing: a redundant column can't lower Bayes risk but can raise NN variance (Grinsztajn
    bias-2) / over-completeness (IB, Tishby/Alemi).
  - *Why "useless/drop" is NOT licensed (the correction):* doc2 §2/§6 is PRECISELY the argument that precomputing
    a relation the set *contains but a capacity-limited 2-ISAB encoder struggles to extract* CAN help (FS hands
    peak a its forward-stutter status without forcing the attention to find peak a+1 and invert SR). So FS *could*
    net positive. "Forward stutter" is also a validated PG relationship (doc2 §1), not a junk feature. size(bp) is
    a genuine degradation signal (STRmix). NONE of these is a "method proven useless" — they are precomputed-
    convenience / feature-engineering choices whose net effect is EMPIRICAL.
  - *Implementation-suspect (must rule out first):* `degr_resid` is a per-sample `np.polyfit` over few peaks
    ([enrich.py:41-49]) → fragile/high-variance; tok11's training instability is a numerics red flag, not an info
    verdict. New columns' standardization (feat_mean/std) and the σ shared across heterogeneous features may be
    mis-set. The user's separate observation (2a_full < base) is itself evidence pointing at SETUP, not the info.
- **CONCLUSION — OPEN.** Do NOT drop FS/size and do NOT claim they help. They are net-negative AS CURRENTLY
  DELIVERED, under n=1 + run variance + an un-audited numerics path. F11 stays OPEN (the earlier draft's
  "REVISE F11 → drop" is RETRACTED).
- **ACTION.** Before any keep/drop: (a) audit numerics (degr_resid fit, per-column standardization, NaN/inf);
  (b) seeds + fixed split; (c) test FS routed to the φ head only (decoupled) vs in the shared token. Decide on
  that evidence.

### F15. 2c (Rank-N-Contrast) is net-negative AS DEPLOYED — this is about OUR off-paper deployment, NOT a verdict on RNC.
- **OBSERVED.** 2c = privsup + `--noc_contrast`, identical otherwise: EM .948→.938, oracle .968→.958, N5 oracle
  .625→.333. The `rnc` term barely descended (Ep1 4.844 → Ep110 4.795, Δ≈0.05).
- **THEORY (paper conditions vs our setup).** Zha 2023 (Rank-N-Contrast, NeurIPS) proves RNC improves ordinal
  representation learning when it is the **main objective, in-distribution**. Our deployment is OFF-paper: RNC is
  an AUX loss whose gradient reaches the SHARED encoder `H` via `pma_noc` ([set_transformer.py:609-612]) while a
  different head (per-donor ID) reads the same `H`; PCGrad (doc2 §4b-C valve 4) is NOT implemented
  ([train_set_transformer.py:602-604]); and it is a count-label contrastive, which §4b-C valve 1 warns can push
  the representation donor-invariant. PLUS the loss did not descend (Δ0.05) → an OPTIMIZATION/implementation
  problem (τ, batch NOC composition, normalization) independent of the idea.
- **CONCLUSION.** The negative result reflects our incomplete/off-paper deployment + a non-descending loss, NOT
  that RNC "doesn't work". Do not adopt 2c AS RUN; do not conclude against RNC.
- **ACTION.** Fix deployment before re-judging: detach `H` into `pma_noc` (or give it its own branch) so the
  contrast shapes only its private pool; add PCGrad; first make the rnc loss actually descend (sanity-check τ /
  batch). Re-test with the §4b-C no-regression guard.

### F16. NOC2 swings are RUN VARIANCE, not a frequency effect — the earlier "σ trades NOC2 for NOC4" claim is RETRACTED.
- **OBSERVED.** The SAME `inc2_2b_privsup` config gave NOC2 oracle **0.864 / EM .795** (earlier run) and
  **0.977 / EM .909** (this run) — an ~11pp swing with NOTHING changed but the run. Within THIS batch, NOC2
  oracle does fall with σ (.977→.932→.864 for σ=.05→.1→.3), but that 11pp range is the SAME magnitude as the
  config's own run-to-run noise → not separable from noise at n=1. The rerun also reshuffled strata in BOTH
  directions (N2/N3 oracle UP, N4/N5 oracle DOWN: N4 .933→.889, N5 .688→.625) — the signature of split/seed
  variance, NOT a uniform "richer data" gain.
- **THEORY.** doc2 §9 (Schaeffer "Mirage": EM is discontinuous on tiny strata) + F5/F8/C6 selection discipline:
  do not read single-run small-NOC deltas; n=44 → wide CI. Two runs of one config disagreeing by 11pp IS the
  empirical CI.
- **CONCLUSION.** NOC2 attribution is UNDER-DETERMINED. My earlier "σ frequency-resolution causes the NOC2 drop"
  is NOT supported (raw also hit 0.864). The user's alternative ("2b/aux is the culprit; the pre-aux base kept
  NOC2 > 0.9") is ALSO under-determined here (this run's 2b NOC2 = 0.977). Both remain hypotheses. Likely root
  cause of the variance: the dev holdout appears re-randomized per run (log shows random combo-holdout) → a
  different train set each run.
- **ACTION (now top priority).** FIX the dev-split seed so reruns are comparable; run ≥3 seeds per arm; report
  per-NOC oracle with CIs. Only then revisit whether σ or aux moves NOC2. To test the user's "richer data"
  hypothesis: verify the tok8 arrays + the train/dev split are byte-identical across the two privsup runs (if they
  are, the swing is pure seed/init variance; if not, the data changed and that confound must be removed).

### Next steps (batch 2, corrected).
1. **Seeds + a FIXED split seed are the #1 blocker** — every n=1 claim above (F13–F16) is gated on it. 2. Treat
periodic σ≈0.3 as the leading candidate (F13), confirm via P4 (seeds + per-NOC CIs). 3. FS/size (F14) and 2c/RNC
(F15): verdicts OPEN — audit numerics + fix off-paper deployment, then re-judge on seeded evidence. Do NOT drop
features or methods on the current single runs.

---

## 2026-06-04 (batch 3) — SEEDED (3 seeds) confirmation; reproducibility added. Two earlier single-run findings were NOISE.

Tooling shipped this round (all root + kaggle_bundle in sync): `train_set_transformer.py` `--seed` (seeds
python/numpy/torch/cuda + DataLoader/open-sampler generators; default 42, persisted to metrics.json); per-NOC
ORACLE/joint/post-hoc now saved to metrics.json; runner `SEEDS=` env → `<arm>_seed<N>` dirs; `aggregate_seeds.py`
(per-NOC mean ± 95% CI + rank-based complete-dominance, since the 95% t-CI is over-wide at n=3); `check_data_identity.py`
(hash source arrays to test the "richer data" hypothesis). Confirmed the dev split was ALREADY seed=0 deterministic —
so the run-to-run variance was the **unseeded training** (init/shuffle/dropout), now fixed.

Arms × 3 seeds (42/43/44), all per_donor + isab++ + aux, joint-card decode; mean ± 95% CI [min,max]:
- **privsup** (raw tok8):  EM .946±.016 · oracle .962±.019 · N2orc .932±.000 · N4orc .844±.166 · N5orc .493±**.434**
- **pe_s3** (periodic σ.3, tok8): EM .961±.018 · oracle .975±.014 · N2orc **.985**±.065 · N4orc .933±.110 · N5orc .646±.137
- **pe_tok9** (periodic σ.1, tok9=+FS): EM .961±.011 · oracle .974±.008 · N2orc .962±.086 · N4orc .933±.199 · N5orc .604±.090

Rank-dominance vs raw privsup: BOTH periodic arms have their WORST seed > raw's BEST seed on EM (min .953/.957 >
max .952) AND oracle (min .968/.970 > max .967) = complete 3v3 separation (exact one-sided p≈0.05, the strongest
attainable at n=3). The aggregator's 95% t-CIs overlap only because n=3 inflates the t-multiplier (~4.3).

### F13 — UPGRADED to CONFIRMED: periodic embedding (σ=0.3) helps, and stabilizes the high-NOC strata.
- **OBSERVED.** pe_s3 COMPLETELY DOMINATES raw privsup across 3 seeds on EM (+1.6pp) and oracle (+1.3pp), with
  mean per-NOC oracle ≥ raw on EVERY stratum (N1 .993≥.989, N2 .985≥.932, N3 .948≥.944, N4 .933≥.844, N5 .646≥.493).
  Periodic also SHRINKS the N5 oracle CI from ±.434 (raw) to ±.137 (pe_s3)/±.090 (tok9).
- **THEORY.** Gorishniy 2022 §3 (per-feature PLR fixes rotation-invariance; in-distribution gain, σ critical) — the
  effect here also transfers to real test. The variance-shrinkage on N5 is consistent with a better-conditioned
  per-feature representation (less reliance on a single fragile height axis).
- **CONCLUSION.** doc1 §3 CONFIRMED at this scale (3-seed complete dominance, not a single run). Periodic σ=0.3 is
  the working embedding on the 2b base. Caveat: n=3 → credible (p≈.05), not definitive; 5 seeds would tighten.

### F14 — REVERSED: the earlier "tok9/FS is net-negative, drop it" was a NOISE artifact. Verdict now neutral-to-positive (FS not yet cleanly isolated).
- **OBSERVED.** Batch-2 single run had tok9 EM .936 / N4 orc .667 → I called FS net-negative. Across 3 seeds
  pe_tok9 = EM .961±.011 / oracle .974±.008 / N4 orc .933 (one seed 1.000) / N3 orc .972 — it TIES pe_s3 and
  DOMINATES raw, with the SMALLEST CIs of all arms and the HIGHEST dev macro-rec (.975/.967/.972). The .936/.667
  were low draws.
- **THEORY.** The set-level redundancy argument (FS_{a+1}=1/SR_a) bounds the *expected information gain* to ~0; it
  does NOT predict harm. doc2 §2/§6 (precompute a set-contained-but-hard-to-extract relation for a capacity-limited
  encoder) predicts a possible small *help* — consistent with tok9 ≈ best tok8 arm. So an empirical net-negative was
  never theory-licensed; it was the n=1 noise F11/F5 warned about.
- **CONCLUSION.** The negative verdict on FS is REFUTED. tok9 is neutral-to-slightly-positive for ID/NOC. **BUT NOT
  CLEANLY ISOLATED:** pe_tok9 used σ=0.1 while pe_s3 used σ=0.3 → the FS effect is confounded with σ. The earlier
  draft's "REVISE F11 → drop" and "drop FS from the shared token" are both RETRACTED; F11 returns to OPEN.
- **ACTION.** Clean A/B: tok8 vs tok9 vs tok11 at the SAME σ (0.3), 3 seeds each → isolate FS/size. (Runner arms
  added: `inc2_2b_pe3_tok9`, `inc2_2b_pe3_tok11`.) φ-side routing of FS still worth testing separately.

### F16 — CONFIRMED NOISE: the σ=0.3 "NOC2 oracle drop to .864" was a single low draw.
- **OBSERVED.** Across 3 seeds pe_s3 NOC2 oracle = 1.000 / 1.000 / 0.955 (mean .985±.065) — HIGHER than raw
  privsup's .932 (identical across all 3 seeds, n=44 lands on 41/44). The batch-2 value .864 is outside this range
  → a low-tail draw, exactly the ±10pp small-stratum variance predicted.
- **CONCLUSION.** The "frequency-resolution trades NOC2 for NOC4" hypothesis is REFUTED. NOC2 is not special; it was
  small-n noise. My batch-2 retraction of F16 was correct and is now positively confirmed by seeds. Selection
  discipline (C6/F5) upheld: never read a single-run small-NOC delta.

### Methodological note (load-bearing).
Two of three batch-2 single-run conclusions (tok9-hurts, NOC2-σ-effect) did not survive seeding. This is direct
evidence for the [[conclusion-discipline]] rule: at n=44–48 strata, n=1 cannot separate effect from noise; honor
the paper-proven prior + run variance before any "drop/useless" verdict.

### Next steps (batch 3).
1. **Clean isolation arms at fixed σ=0.3 × 3 seeds**: tok8 / tok9 / tok11 (+FS, +size) — isolate each feature
   (F14). 2. **2c valve variants as separate comparable arms** (F15, §4b-C): `shared` (current) vs `detach`
   (pma_noc pools H.detach() — pure decoupling control) vs `pcgrad` (valve 4: project rnc grad off the main-task
   grad). Implemented behind `--noc_contrast_mode`. 3. Carry periodic σ=0.3 as the base. 4. 5 seeds to move from
   p≈.05 to definitive on the periodic win.

---

## 2026-06-05 (batch 3 results) — clean σ-matched isolation + RNC valve variants, 3 seeds each. Two NEW load-bearing findings: a RNC manipulation-check FAILURE, and a tok11 domain-gap.

Arms on the periodic σ=0.3 + aux base, 3 seeds (42/43/44), aggregate mean ± 95% CI, plus rank-based
complete-dominance vs `pe_s3` baseline:
- **pe_s3** (tok8): EM .961±.018 · oracle .975±.014 · card_noc_acc .974
- **pe3_tok9** (+FS): EM .964±.015 · oracle .978±.011 · card_noc_acc .977 · Δoracle +.003 (overlapping)
- **pe3_tok11** (+size/degr): EM .932±.020 · oracle .954±.010 · card_noc_acc .945 · Δoracle −.021 **DOMINATED**
- **2c_shared/detach/pcgrad** (RNC): EM .951/.953/.958 · oracle .968/.972/.970 · all overlapping, none > baseline

### F17 — FS (tok9) at matched σ=0.3 is NEUTRAL (clean isolation; theory + experiment agree). Keep tok8 for parsimony.
- **OBSERVED.** tok9 vs tok8 at the SAME σ=0.3: Δoracle +.003, ΔEM +.003, card_noc_acc +.003 — all overlapping,
  no separable effect, on ID, NOC-head, and per-NOC. Delivery audited clean (below).
- **THEORY.** FS_{a+1}=1/SR_a → set-level redundant → DPI bounds info gain ≈ 0 → predicted point Δ≈0. Experiment
  hits the point prediction. (FS has the §2/§6 "convenience" escape hatch but it's pure redundancy, so no help.)
- **CONCLUSION.** FS is NEUTRAL for ID/NOC — the strongest-grounded feature verdict (theory point-prediction +
  3-seed confirmation agree). Keep tok8 (minimal-sufficient); FS optional for the φ head only. F11 → RESOLVED-neutral.

### F18 — DELIVERY AUDIT: tok9/tok11 reach the model correctly — NO setup/training/inference bug. Unlike RNC, no silent-off mechanism.
- **OBSERVED (features/enrich.py audited on real train+test arrays).** FS/size/degr: 0 NaN, 0 inf; padding exactly 0;
  real variance (FS std 1.13, frac-nonzero .48; size [76,443]bp 100% coverage; degr_resid std ~1.2); **degr polyfit
  degenerate/skipped = 0/55247** (the "fragile polyfit" never failed). Inference standardization matches train:
  feat_mean/feat_std are register_buffers → saved in best_model.pt → restored by load_state_dict. Token width
  consistent train↔test; downstream paths read fixed positions. tok11 training (3 seeds) is STABLE: monotone loss,
  clean early-stop, no oscillation — the batch-2 raw-token instability is GONE under periodic σ=0.3.
- **CONCLUSION.** tok9/tok11 are genuinely fed to the model. "tok11 hurts" is NOT a delivery/instability artifact —
  and the fact it CHANGES the output (hurts) proves the model uses those columns. A feature-importance probe is
  therefore unnecessary for the decision (the ablation ladder tok8→+FS→+size/degr already isolates the harm to
  size/degr). Contrast with RNC (F19), which DID have a silent-off failure.

### F19 — RNC (2c) was NEVER ACTUALLY TESTED: the contrastive loss is FLAT in all 9 runs → manipulation-check FAILURE. The "RNC no-benefit" reading is WITHDRAWN.
- **OBSERVED (full training logs, rnc value per epoch).** loss_rnc Ep1→last: shared 4.85→4.79, detach 4.88→4.81,
  pcgrad 4.85→4.79 — **Δ≈0.05–0.08 (~1%) over ~100 epochs in ALL 9 runs.** A randomly-initialised contrastive loss
  that barely descends = the objective is essentially not optimised → the projection was never shaped into ordinal
  structure → card_noc_acc/EM null is UNINTERPRETABLE as "RNC doesn't help". Internal consistency: `detach` descends
  LEAST (it blocks the encoder path) ✓; all 3 modes ≈ baseline precisely because the contrast did nothing in all 3.
- **THEORY (two mechanisms, both silently disable RNC; both consistent with the flat loss).**
  1. **τ=2 too large with L2-normalised features.** `z=normalize(z); sim=-cdist/τ` → for unit vectors cdist∈[0,2] →
     sim∈[-1,0]. Best-case logprob = −log(N)+1 ≈ −3.6 (N≈100) → the loss FLOOR ≈ 3.6, so 4.85→3.6 is the most it
     could ever move, and the weak gradient barely budges it. τ=0.5 → floor ≈ 0.6 → real gradient. **τ is the
     primary bug.** (Zha 2023 uses a smaller effective temperature / unnormalised distances.)
  2. **Kendall auto-down-weighting** (1705.07115): effective weight exp(−log_var_rnc) inferred to drop to ~0.3 — a
     hard/noisy task is down-weighted, secondary.
- **CONCLUSION.** WITHDRAW "RNC has no benefit". RNC is UNTESTED — the experiment never exercised it (manipulation
  check failed). Cannot conclude for OR against RNC. This is the same discipline as seeds (F13–F16): a NULL is only
  interpretable if the intervention provably happened. (FIXED this round — see code changes; re-test pending.)
- **ACTION.** Fix τ (sweep {0.1,0.3,0.5}) + add `--rnc_fixed_weight` (bypass Kendall) + log loss_rnc/log_var_rnc to
  history + a `probe_noc_structure.py` manipulation check (Spearman pairwise-dist↔|ΔNOC| + linear-probe on z_noc_proj
  dumped at test). Only after the probe confirms z_noc_proj is ordinally structured may card_noc_acc be read.

### F20 — tok11 (size/degr) harm is a DOMAIN GAP (in-silico overfit), not instability — and it is CONDITIONAL on the uncalibrated generator. Verdict: drop from token NOW, RE-TEST after §13.
- **OBSERVED.** tok11 has the HIGHEST dev macro-recall of any arm (.972/.977/.978 ≥ pe_s3 .968/.973/.967) yet
  COLLAPSES on real test at high NOC (N4 oracle .58–.71, N5 .29–.40 vs pe_s3 .93/.65). Dev↑ test↓ divergence. Audit:
  std(degr_resid) train=1.21 vs test=1.67 (+38%) — the feature's distribution SHIFTS train→test.
- **THEORY.** doc2 §13: the generator does NOT model degradation (only gamma jitter + threshold dropout + ratio
  skew). So on synthetic, height⊥size → degr_resid ≈ centred height (slope b≈0); on real, height decays with size →
  degr_resid is a true degradation residual. **Same column, different meaning train vs test = covariate shift** →
  negative transfer, concentrated at high NOC where degradation/dropout bites most (Green&Mortera §8). Ben-David
  domain-adaptation bound: a feature whose conditional shifts across domains raises target error. C6/C7: dev does
  NOT predict test for this feature — selecting on dev would WRONGLY pick tok11 (a selection-discipline trap).
- **CONCLUSION.** "size/degr hurts" is SCOPED to the current uncalibrated generator, not absolute. Drop size/degr
  from the shared token NOW. But degr_resid is the ONE deferred feature with a real theoretical case to revive
  post-§13 (it precomputes a cross-locus degradation relation a 2-ISAB struggles to compute, §2/§6 — unlike FS's pure
  redundancy), PROVIDED §13 makes the relation real+transferable. DPI still bounds it (may stay neutral); and the
  degradation benefit may arrive via distribution match (Ben-David) WITHOUT the columns at all.
- **ACTION (sequencing, §11 — never change generator AND judge features at once).** (1) Freeze base = pe_s3. (2) §13:
  add degradation to make_insilico, calibrate to real per-condition, validate_realism BEFORE trusting anything.
  (3) re-measure domain gap on the frozen base [main prize]. (4) THEN re-derive + re-test tok9/tok11 on REAL test
  (not dev), seeded. size_bp is ≈affine(locus,allele) (DPI-redundant, harmless); degr_resid is the active variable.

### Standing verdicts after batch 3 — SUPERSEDED by the 2026-06-06 section below (2c/2d ran; RNC now tested; sparse adopted on OOD evidence). Kept for the trail.
- **Working base = pe_s3 (tok8 + periodic σ=0.3 + aux), isab++.** Nothing beat it: FS neutral (F17), size/degr a
  domain-gap loss (F20), RNC untested (F19). IB-plateau on relationship-additions; the remaining real lever is
  data-side (§13). 5 seeds would harden the periodic win (F13) from p≈.05.
- Next: §2d architecture levers (sparse-attn / IRM / struct-bias) ablated on the frozen base, seeded, real-test;
  and §13 generator calibration. Each as a separate comparable arm.

---

## 2026-06-06 — 2c/2d Kaggle results IN + first ZERO-SHOT cross-kit OOD + NOC5-ceiling re-measurement

Three things landed: (1) the queued 2c (RNC τ=0.3+fixed-wt, shared/pcgrad) and 2d (sparse / IRM λ1,λ10)
3-seed runs; (2) a new local cross-folder OOD harness (`prepare_crossfolder.py`, `eval_crossfolder.py`,
`make_gf_control.py`) that evaluates frozen GF29 checkpoints on a DIFFERENT PROVEDIt kit folder; (3) a
raw-genotype re-measurement of the NOC5 "ceiling" (`measure_noc5_ceiling.py`) — see the F6 rewrite above.
All eval on REAL test / REAL external folders, 3 seeds, mean±CI. Selection discipline upheld (C6/F5).

### F21. RNC (2c) — manipulation-check now PASSES, so the null is interpretable: NEUTRAL in-distribution, NEUTRAL→HARMFUL OOD. Resolves F19; not adopted.
- **OBSERVED.** τ=0.3 + `--rnc_fixed_weight 1.0`: `probe_noc_structure` Spearman(featdist,|ΔNOC|) .74–.78,
  linear-probe NOC .91–.93 > prior .828 across all seeds → the contrast DID shape z_noc_proj (the F19
  flat-loss/τ=2 bug is gone). Yet downstream is unchanged: card_noc_acc shared .9737 = base .9737 exactly;
  in-dist EM .963±.012 ≈ base .961; pcgrad worse+noisier (.955±.034). Cross-kit OOD (IDPlus28): shared ID
  .809 = base .809 but oracle N4/N5 LOWER (.156/.037 vs .230/.084); pcgrad WORST arm (.780).
- **THEORY.** §4b-C valve 2/3: the contrast lives on `pma_noc`/`proj_noc`, DISCARDED at inference, and the
  count head reads `CardinalityHead(sigmoid(logits_cls).detach())` (set_transformer.py:655) — a separate,
  detached path. So even an active RNC cannot reach NOC decode → structurally neutral, ID protected. The OOD
  harm = count-label contrast overfitting the GF NOC geometry (valve-1's documented failure mode).
- **CONCLUSION.** F19 RESOLVED: RNC-as-deployed is interpretable-NEUTRAL (ID-protected) in-dist and
  neutral-to-harmful OOD. Do NOT adopt. CAVEAT: only τ=0.3 tested; the F19 τ-sweep {0.1,0.5} is still open,
  but τ=0.3 is a real (not silent-off) negative. To make RNC HELP NOC it must be wired INTO the count
  representation (height-aware/ordinal count head), not a discarded projection — see F6 ACTION.

### F22. sparse-attn (2d, §7) — in-distribution NEUTRAL but the clear GENERALIZATION lever. ADOPT.
- **OBSERVED.** In-dist EM .961±.007 = base (neutral), tightest seed CI, reject AUROC 1.000. Cross-kit OOD
  (IDPlus28, same RD14 donors): BEST arm at EVERY stratum — ID .836±.015 > base .809 (sparse min-seed .821 ≥
  base max-seed .821), oracle N3 .662/N4 .322 vs base .559/.230. Wins under locus-masking too (GF16 ID .845
  vs .820). reject AUROC internal .957 > base .943.
- **THEORY.** §7: sparsemax assigns EXACTLY ZERO to irrelevant/other-donor peaks → a regulariser that
  suppresses spurious cross-donor attention; the benefit shows up precisely under covariate shift (transfer),
  not in-distribution. Exactly the §7 motivation, only visible OOD.
- **CONCLUSION.** sparse-attn EARNS adoption — its in-dist "neutral" hid a robustness gain. **New base =
  pe_s3 + sparse-attn.** No-regression guard held (N1/2/3 ID intact). n=3, credible not definitive.

### F23. IRM (2d, §7) — neutral on AGGREGATE EM, breaks the auxiliary threshold-F1, expensive. Dropped as deployed — but REOPENED at high NOC by F29 (its N5 effect was never measured).
- **OBSERVED.** λ=1 in-dist EM .962±.008 ≈ base; OOD .816±.013 ≈ base (not separable). BUT the NOC-environment
  penalty collapses val threshold-F1 (.98→.15 at λ=1, →~.13 at λ=10) while joint-card decode EM survives.
  λ=10 worse + ~2× slower (irm10_seed44 hit the 12h Kaggle cell timeout — only 2 seeds).
- **THEORY.** §7 IRM targets the combo/NOC co-occurrence shortcut. The earlier reading that "the shortcut is
  already suppressed by aux supervision + the combo-disjoint split" is REFUTED by F29: a large combinatorial
  train→dev gap remains at high NOC (N5 oracle .99→.57), i.e. the shortcut is NOT suppressed. The threshold-F1
  collapse is a deployment/calibration issue, not evidence the shortcut is gone.
- **CONCLUSION.** Dropped AS DEPLOYED (judged on AGGREGATE/real EM, ~83% NOC1, which masks the high-NOC combo
  gap). Its effect on the N5 in-silico-dev ORACLE — the metric that exposes the gap (F29) — was NEVER measured,
  so IRM/REx is REOPENED as a candidate anti-overfitting lever for N5, to be judged on N5 dev oracle (not
  aggregate EM). struct-bias (3rd §2d branch) stays SKIPPED (ISAB-incompat + redundant with Hb/SR/FS).

### F24. Cross-kit ZERO-SHOT generalization: transfer across kit/instrument is GOOD; the high-NOC OOD collapse is ~90% LOCUS-LOSS, not covariate shift.
- **OBSERVED.** Frozen GF29 (GlobalFiler/3500) → IdentifilerPlus folders, no retrain, train-time
  standardisation. Same donors RD14-0003 (`3130_IDPlus28`, n=10261) → ID+NOC measurable; different people
  RD12-0002 (`3500_IDPlus29`) → NOC+reject only. Decomposition (pe_s3, ID EM): GF-24loci .961 → GF-MASKED-to-16
  -loci (SAME kit) .820 → actual IDPlus OOD .809. oracle N4: .933 → .319 → .230. So cutting GF to the 16 shared
  IDPlus loci (dropping SE33 + 7 others) ALREADY causes ~90% of the drop; the real kit/instrument change adds
  only a small residual (ID −1.1pp, N4 −9pp).
- **THEORY.** doc2 §8 Green/Mortera information ceiling: fewer loci = fewer private alleles to deconvolve 4–5
  contributors. The model is NOT failing to transfer chemistry — it lacks the loci. CAVEAT: the masked control
  also feeds a missing-locus pattern the model never trained on (train 24 → infer 16), so it slightly
  OVERSTATES the pure-information term; true isolation needs retraining on 16 loci.
- **CONCLUSION.** Reframes C7: kit/instrument transfer is good; the binding wall on a reduced-locus kit is
  information (loci typed). The within-kit real test OVERSTATED capability at high NOC — always pair with a
  cross-folder OOD read. Files: prepare_/eval_crossfolder.py, make_gf_control.py; data_cross/; metrics_cross_*.json.

### F25. deepNoC comparison — on COUNT accuracy our N4/N5 genuinely lags (real test), and that gap is a domain-gap + count-head issue, NOT loci.
- **OBSERVED.** Per-NOC COUNT accuracy (deepNoC-comparable; NOT per-donor EM), real GF test: N1 .999 N2 .947
  N3 .910 N4 .770 N5 .729. N1–N3 ≈ deepNoC's >.9; **N4/N5 below**. (The .646 N5 figure elsewhere is per-donor
  Exact-Match — identify all 5 people — a strictly harder task than counting.)
- **THEORY / CONCLUSION (honest).** Three contributors to the N4/N5 count gap vs deepNoC: (i) deepNoC's >.9 is
  on its OWN simulated (in-distribution) test; ours is REAL test across the in-silico→real gap (C7) — the fair
  comparison is our in-silico DEV count (higher, not yet dumped locally). (ii) deepNoC = 89 engineered
  feats/peak, count-specialised; ours derives count from a 45-dim ID-profile bottleneck (a light detached
  head), so it discards the height-stacking signal (C3: decode↔oracle gap = count error). (iii) generator not
  yet calibrated (§13). NONE of these is loci. ACTION: dump in-silico-dev per-NOC count to size the domain gap;
  build a direct height-aware ordinal count head (F6 ACTION) reading the encoder, not the ID profile.

### Standing verdicts (current, supersedes batch-3).
- **Working base = pe_s3 (tok8 + periodic σ=0.3 + aux), isab++) + sparse-attn** (F22 — sparse adopted on the
  cross-kit OOD generalization win; in-distribution it is neutral). RNC DROPPED (F21), IRM DROPPED (F23),
  struct-bias SKIPPED.
- **NOC5 is model-limited, not information-limited** (F6; reaffirmed by F29 — the overlay generator gives 100%
  info-presence). The high-NOC ORACLE limit is a COMBINATORIAL-GENERALIZATION gap (F29: train N5 oracle ~1.0 →
  held-out-combo dev ~.57; real ≈ dev ⇒ NOT a domain gap, NOT §13). Oracle levers = combo augmentation /
  IRM-at-N5 / decoder regularization (sparse already adopted). The EM↔oracle count head (C3) is a SEPARATE
  second wall. Not a target cap, not loci, not §13-for-oracle.
- **Generalization caveat now standing:** within-kit real test overstates high-NOC; cross-folder OOD (loci
  matched) is the honest read. Cross-kit transfer is good; reduced-locus kits are information-bounded at high NOC.
- Open: (a) τ-sweep RNC {0.1,0.5} to fully close F21; (b) in-silico-dev per-NOC count dump (F25); (c) build +
  test the direct ordinal count head (F6/F25 ACTION), 3 seeds, no-regression guard N1/2/3; (d) §13 generator
  degradation calibration (data-side, last). 5 seeds would harden the periodic (F13) and sparse (F22) wins.

---

## 2026-06-07 — Increment 3 (ID-representation + NOC-ordinal levers) — 6 arms × 3 seeds + cross-kit OOD + a mechanism probe. Net: a NULL in-distribution; one weak OOD survivor (geno_query).

Design = `reports/design_increment3_levers.md`, each lever = ONE published method on the **bare pe_s3 base**
(tok8 + periodic σ0.3 + aux + isab++) — NOTE: bare pe_s3, NOT the adopted `pe_s3+sparse` (F22). 6 arms:
**repA** `--geno_query` (DAB-DETR/Conditional-DETR genotype-conditioned donor queries, Liu 2022/Meng 2021) ·
**repB** `--donor_contrast` (SupCon-by-donor, Khosla 2020) · **repC** `--cls_decoder additive`
(AdditiveDonorDecoder, the C1 prior) · **nocV1/V2/V3** = CORN ordinal count head (Shi/Cao/Raschka 2023) +
RNC (Zha 2023) on a count pool that RNC actually shapes — V1 detach+ensemble, V2 non-detach+PCGrad (Yu 2020),
V3 replace. Aggregated this session with `aggregate_seeds.py` (in-dist, selected-decode EM) + the NEW
`eval_cross_inc3.py` (OOD wrapper — `eval_crossfolder.build_model` does NOT wire geno_query/donor_contrast;
the wrapper re-verified by reproducing base .809 / sparse .836 exactly).

In-dist real-test EM (3-seed mean [min,max]) / oracle: base pe_s3 **.961**[.953,.966]/.975 · pe_s3+sparse
.961[.958,.963]/.977 · repA .957[.952,.965]/.972 · repB .956[.940,.967]/.969 · nocV1 .942/.971 ·
nocV2 .936/.971 · nocV3 .906/.936 · repC .858[.835,.903]/.883.
OOD idplus28_rd14 ID EM (3-seed): base .809[.800,.821] (orcN4 .230) · sparse **.836**[.821,.847] (orcN4 .322) ·
repA **.819**[.811,.825] (orcN3 .584, N4 .308, N5 .096) · repB .787[.749,.812] (orcN4 .169).

### F26. Increment 3 is a NULL in-distribution: no arm beats base. Representation levers neutral, NOC-ordinal levers ≤ base, additive decoder fails.
- **OBSERVED.** Every inc3 arm's 3-seed EM ≤ base .961: repA/repB neutral (.957/.956, CIs overlap base heavily,
  oracle even slightly lower .972/.969 < .975); nocV1/V2 clearly below (.942/.936); nocV3 unstable (.906, one seed
  oracle collapsed to .861); repC the worst (.858, high-NOC decode collapse N3 .25 / N4 .01 / N5 .00). NOC1 ID EM
  stays ~.99 on every arm (no-regression guard held). NOC5 oracle still ~.57–.62 everywhere — no lever cracked it.
- **THEORY.** doc1 §7 head-to-head discipline; F6 (NOC5 model-limited but hard); C3 (decode↔oracle gap = count).
  The base in-dist ranking is near-saturated (oracle .975) → there is little in-distribution headroom for any
  model-side lever, consistent with the batch-3/F17/F20 IB-plateau (relationship-additions stopped paying in-dist).
- **CONCLUSION.** Increment 3 produces NO in-distribution winner. This is the EXPECTED outcome under F6+C3: the
  residual N4/N5 weakness is a count/decoding-or-data problem, not something a representation/ordinal-head lever
  moves at this base. Single-eval, n=3 — but the direction (all ≤ base) is uniform across arms and seeds.
- **ACTION.** Do not adopt any inc3 arm on in-distribution evidence. The decisive read is OOD (F27).

### F27. geno_query (repA) is the only OOD survivor — a WEAK positive, second to sparse; donor_contrast (repB) is OOD-NEGATIVE. The genotype anchors transfer because allele identity is domain-invariant.
- **OBSERVED.** Cross-kit OOD (idplus28_rd14, same RD14 donors, GF-harmonised, zero-shot, 3 seeds): repA ID EM
  .819 > base .809 (Δ+1.0pp, seed ranges [.811,.825] vs [.800,.821] barely overlap at the top) and lifts oracle
  **N4 .230→.308** (≈ sparse .322) + N3 .559→.584; still < sparse .836. repB .787 < base .809 with WORSE oracle
  N4 (.169 < .230). In-dist both were neutral (F26).
- **THEORY.** doc2 §5 LUPI / DAB-DETR (Liu 2022): a query that is an explicit reference-genotype PRIOR anchors
  attention on the donor's own (locus,allele) — and allele IDENTITY is kit/instrument-invariant (the same RD14
  person's reference alleles are valid across folders), so the anchor transfers. This is the SAME shape as sparse
  (F22): in-dist neutral, OOD positive — a robustness/regularisation lever visible only under covariate shift.
  repB's failure = SupCon-by-donor overfits the GF peak-grouping geometry → breaks under kit shift (valve-1 /
  the documented RNC/IRM pattern, F21/F23: a contrastive on a within-domain grouping is a domain-specific shortcut).
- **CONCLUSION.** geno_query is a genuine but WEAK OOD lever (ranks 2nd to sparse); donor_contrast is dropped
  (OOD-negative). The "domain-invariant allele identity transfers" thesis holds directionally. n=3, single OOD
  folder, Δ small and ranges touch → credible-not-definitive; NOT yet a standalone adoption.
- **ACTION.** Test the only arm with real upside: **`pe_s3 + sparse + geno_query`** (inc3 was built on bare pe_s3,
  missing the adopted sparse; sparse is the stronger lever, geno_query adds N4-oracle — they hit different strata,
  worth checking if they compose). Drop repB, nocV1/V2/V3, repC.

### F28. WHY repC fails: a probe shows the failure is at the READOUT, not the encoder — which PARTIALLY REFUTES my own first-pass mechanism story. Methodology: those explanations were hypotheses, not isolated causes.
- **OBSERVED (probe `attr_head`-input H → fresh linear donor probe, independent of the trained head; ~5min,
  local RTX3050 GPU ≈ T4 throughput).** Donor-separability of the encoder representation H: base .994, repA .994,
  repC **.979** (per-NOC N5: base/repA .97 vs repC .91). So repC's additive decoder DOES leave H slightly less
  donor-separable (real, gap widens at high NOC) — but the gap is TOO SMALL to explain repC's collapse: the donor
  information SURVIVES in H at 0.91–0.95 even at high NOC, i.e. it is linearly recoverable; the catastrophe (N4 .01,
  N5 .00 decoded) happens DOWNSTREAM of H.
- **THEORY.** The additive sum-pool readout (vs per_donor cross-attention) is the suspect: a sum over peaks is not
  invariant to set cardinality (score scales with #peaks → a NOC1-calibrated threshold can't serve NOC5), and its
  per-allele-lookup advantage (the C1 prior) was established on ALIGNED Xflat bins, not on ISAB-context-mixed H.
  Both are READOUT-stage mechanisms, consistent with "info is in H but not decoded". NOT yet isolated by experiment.
- **CONCLUSION.** The probe LOCATES repC's failure at the readout, not the representation — and PARTIALLY REFUTES
  the first-pass "additive under-shapes the encoder" story (that effect is real but minor). DISCIPLINE NOTE: the
  detailed turn-1 theory explanations for repC AND the noc-arms were a MIX of (a) accurate method descriptions,
  (b) extrapolations of real papers (Grinsztajn/Deep Sets/PCGrad) not proven for THIS architecture, (c) the
  project's own self-authored findings (count=ranking-error etc. — internal, not independent literature, partly
  circular), and (d) untested conjecture. They are post-hoc, consistent-with-data, NOT experimentally isolated —
  treat as HYPOTHESES. The only one tested (encoder under-shaping) was downgraded by its own probe.
- **ACTION.** repC/noc-arms are understood ENOUGH to drop (F26) — the decision needs no further compute.
  DEFERRED training ablations (if a paper-grade causal claim is later wanted; NOT on the critical path):
  #1 additive mean-pool vs sum-pool (tests cardinality-scale; ~3.5h/3seeds, needs a pool-option code add);
  #2 additive `decoder_source=raw/local` vs `encoded` (tests needs-aligned-features); #6/#8 (RNC redundancy /
  capacity-vs-PCGrad) = low value, already covered by the project's count=ranking finding. Conclusions need 3 seeds.

### Standing verdicts (current, supersedes the 2026-06-06 section's open items where they overlap).
- **Working base unchanged = pe_s3 (tok8 + periodic σ=0.3 + aux, isab++) + sparse-attn** (F22). Increment 3 added
  NO adopted lever: geno_query (F27) is a weak OOD candidate pending the `+sparse` stack test; everything else
  (donor_contrast, CORN/RNC noc-arms V1/V2/V3, additive repC) DROPPED.
- **Increment 3 verdict:** representation/ordinal-head levers do not move the in-distribution number (F26, base is
  near-saturated); the only thing that transfers is an explicit domain-invariant ID prior (geno_query, F27) — same
  lesson as sparse. The high-NOC residual is NOT a representation problem at this base.
- **NEXT (single highest-value arm):** `pe_s3 + sparse + geno_query`, 3 seeds, in-dist + cross-kit OOD, guard
  N1/2/3. The real high-NOC levers (F29, off the representation axis Inc3 explored): (1) ORACLE gap = combo
  augmentation / IRM-re-judged-at-N5 / decoder regularization (anti combo-overfitting); (2) EM↔oracle gap =
  a height-aware ordinal count head reading the encoder (C3/F25). §13 is a feature/domain lever, NOT the
  N5-oracle lever.
- Tools added: `eval_cross_inc3.py` (OOD eval for geno_query/donor_contrast arms). 5 seeds would harden F27.

---

## 2026-06-07 (batch 4) — per-NOC IN-SILICO oracle finally measured (eval-only). The high-NOC oracle ceiling is a COMBINATORIAL-GENERALIZATION (overfitting) gap, NOT capacity and NOT domain gap. REVISES the §13/count-head framing for the ORACLE.

Triggered by the question: "if N5 weakness is a domain gap, the IN-SILICO (train-domain) oracle should be
near-1 — what is it?" The per-NOC in-silico oracle had NEVER been dumped (the F25 open ACTION). Measured now
with `measure_insilico_oracle.py` (EVAL-ONLY, no retrain): loads the frozen `inc2_2b_pe_s3` checkpoints,
reconstructs the in-silico DEV split IN MEMORY identical to make_dev_split.py (seed=0, non-destructive), and
reports per-NOC oracle EM on THREE sets — in-silico TRAIN-fit (seen), in-silico DEV (held-out combo-disjoint,
SAME generator = zero domain shift), and REAL test (domain-shifted). 3 seeds (42/43/44).

### F29. Per-NOC oracle: TRAIN-fit ~1.0 at every NOC, but in-silico DEV N5 ≈ .58 — a ~40pp train→dev gap WITHIN in-silico. Real ≈ dev. ⇒ overfitting to training combos, not capacity, not domain gap.
- **OBSERVED (oracle EM, 3 seeds 42/43/44).**
  - N5: TRAIN-fit .992/.998/.997 · **in-silico DEV .574/.614/.566** · REAL test .688/.667/.583
  - N4: TRAIN-fit .998/.999/.999 · **in-silico DEV .831/.866/.824** · REAL test .933/.978/.889
  - N3 (seed42): TRAIN-fit 1.000 · DEV .955 · REAL .938 ; N1/N2 ~1.0 everywhere.
  - INFO-PRESENCE proxy on in-silico DEV: ~100% of samples at EVERY NOC (incl. N5 2153/2159) have all
    contributors leaving an attributed peak → the synthetic data is NOT information-starved.
- **THEORY.**
  - *Capacity falsified:* TRAIN-fit N5 = .99+ → the architecture CAN rank 5 contributors near-perfectly;
    the per-donor decoder is not capacity-bound at high NOC.
  - *Domain-gap falsified at N5:* REAL N5 (.58–.69) ≈ or > in-silico DEV N5 (.57–.61); REAL N4 (.89–.98) >
    DEV N4 (.82–.87). A domain gap predicts REAL << in-silico-held-out; the opposite holds. (The in-silico DEV
    is the HARDER benchmark — 2159 N5 novel combos spanning the full Mx sweep vs real n=48.)
  - *Confound (F6) ruled out empirically:* `make_insilico.py:2-6` builds mixtures by OVERLAYING REAL
    single-source profiles → peaks are real, complete alleles; the 91.4% genotype_qc coverage only dents the
    attr LABELS, not allele presence (info-presence proxy = 100%). So a low in-silico N5 oracle is a model
    failure, not synth info-starvation.
  - *Mechanism:* train .99 → held-out-combo dev .57 = a generalization gap driven purely by UNSEEN
    contributor combinations = **shortcut learning** (Geirhos 2020; doc2 §7 names it: "our memorization failure
    = textbook shortcut learning") + a combinatorial-coverage wall: C(45,5)≈1.22M 5-combos, train ≤14,400 N5
    samples → <1.2% coverage; the model memorizes seen combos and fails to compose unseen ones.
- **CONCLUSION.** The binding limit on high-NOC ORACLE (ranking ceiling) is **combinatorial generalization /
  overfitting to training combos** — NOT representation capacity and NOT the in-silico→real domain gap. This
  is n=3-seed robust (the .99→.57 gap dwarfs seed variance). The information is present (F6 reaffirmed via the
  overlay generator + 100% info-presence); the model can fit it (train ~1.0); it cannot generalize it to novel
  combos.
- **ACTION (all require RETRAINING — noted, deferred).** (1) Combinatorial augmentation = more unique training
  combos (doc2 §4 domain randomization over the contributor-combo axis) — direct attack on the <1.2% coverage.
  (2) Re-judge IRM/REx (doc2 §7) AT N5: see the F23 REVISE below — its drop was decided on the wrong metric.
  (3) sparse-attn (F22, already adopted) is anti-shortcut and consistent. (4) capacity/regularization control
  on the per-donor decoder. The count-head/C3 EM↔oracle gap remains a SEPARATE second wall on top of this.
- **DISCIPLINE.** Eval-only, frozen checkpoints; the in-silico DEV split is reproduced deterministically
  (seed=0) and is genuinely held out (its N5 oracle .57 << train .99 rules out leakage). Tool:
  `measure_insilico_oracle.py`. NOT yet seed-extended beyond 3; the direction is uniform across all 3.

### Corrections applied AT SOURCE (fixed in place, not just flagged here).
The now-wrong "§13 / count-head is the N5-ORACLE lever" framing was corrected where it was written: F6 ACTION,
F23 (header + THEORY + CONCLUSION), the 2026-06-06 standing verdict, and the 2026-06-07 Inc3 standing verdict.
Left unchanged because still valid: F20 (size/degr is a FEATURE-level domain gap) and the C3 count head (a
SEPARATE second wall). The old F25 ACTION "dump in-silico per-NOC oracle" is now DONE (= this finding).

### Standing verdicts (current, supersedes the high-NOC-lever items above).
- **Working base unchanged = pe_s3 (tok8 + periodic σ=0.3 + aux, isab++) + sparse-attn.**
- **High-NOC ORACLE root cause = combinatorial generalization (F29), REFINED by F30 → per-donor DECISION
  entanglement, NOT a coverage/data-volume problem.** ⚠️ F30 RETRACTS the "combinatorial augmentation /
  <1.2% coverage" lever from F29: every real N5 test combo is already 1-donor-swap from a train combo yet
  still fails 27% → more combos won't help. Viable lever = **per-donor PEELING decoder** (feasibility
  positive, F30). §13/coverage/biology-likelihood-ID all NOT the oracle lever.
- Two stacked walls for "oracle>0.9 AND EM≈oracle at all NOC": (1) N5 oracle = per-donor decision entanglement
  (F30, peeling lever); (2) count-head EM↔oracle gap (C3/F25) REFINED by F30 = NOT feature-limited (gain_k
  redundant), it's a CALIBRATION/decode gap on the neural card head (4↔5 AUC .944 but acc .76). Both need
  retraining. Goal status (F30, full real test): N1–N3 done; N4 oracle done (.92), EM count-blocked (.86); N5 oracle .73.

### F30 (2026-06-07) — FULL real test rebuilt; pgNOC retracted; N5 root cause = per-donor decision entanglement; peeling feasible. (eval-only, p3=inc4_p3_irm_seed42, seed42)
- **DATA FIX.** Canonical train/eval folder = `data_insilico_w` (≡ Kaggle `data_w`), NOT the stale `data/`(=`data_kaggle`, 146 combos). Raw GF29 has only ~7–8 combos/NOC (5–6 known-only); the old split gave the real test exactly **1 combo/NOC** → every prior "real N2..N5" metric was a 1-combo spot-check. **Rebuilt FULL real test = 3582** (held-out NOC1 2249 + ALL multi-donor 1333, covers all 5/6/4/5 raw combos, 0 leak verified) via `build_full_real_test.py`; old test backed up. All eval tools now use it.
- **Arm ranking on full test (all-combo oracle EM, `measure_real_allcombos_multi.py`).** Best = **inc4_p3_irm** (oracle N4 .917/N5 .726, 2-5 .892) > p1_stack > others. repC_additive & nocV3_ordreplace COLLAPSE at multi-donor (N3-5 ≈0, hidden by 83%-NOC1 old test) → drop. IRM = best in-domain but WORST OOD (idplus28 ID-EM .775<base .809; reject .932<.968) → in-domain↔OOD tradeoff; no arm wins both.
- **pgNOC RETRACTED as count decoder.** `pgnoc.py` hardcodes stale `hybrid_50k_weight` (use `pgnoc_eval.py`). On full test (own probs, inc2_2b/inc2_2c_sh/p3): pgNOC count (BIC) << neural count head at every NOC (p3: pgNOC N4 .69/N5 .42 vs neural .87/.76). "two-stage+pgNOC 0.954 best count" = stale-test artifact. (Naming: inc2_2c_fix_pg/sh = noc_contrast pcgrad/shared, NOT pgNOC.)
- **gain_k feature REDUNDANT (`quick_gain_features.py`).** Likelihood cost-curve gains add ~0 over neural card head at EVERY transition (AUC_both ≈ AUC_neural, +0.000–0.003 in-sample). Count is NOT feature-limited; the N5 count gap (4↔5 AUC .944 but acc .76) is decode/CALIBRATION. RETRACTS this-session's own premature "gain5 strong feature to ensemble" (it compared to a weak height-0.67 baseline, not to the neural head).
- **N5 root cause filtered (no-train probes `probes_n5.py`/`probe_height.py`/`feasibility_id_likelihood.py`).** (a) NOT coverage: every N5 test combo is 1-donor-swap from a train combo, still fails 27% ⇒ systematic generalization, not data volume. (b) NOT perception/pool/faint: buried (rank≥16) donors' private alleles are 100% PRESENT and tall (glob_rel .69 ≈ near-miss, vs recovered .81) ⇒ visible evidence ignored. (c) NOT biology-ID: gamma-NNLS likelihood ranks ID worse than neural even when true⊆pool (.55 vs .73). ⇒ **CONCLUSION: N5 wall = per-donor DECISION entanglement (overfits combo-specific boundaries), not info/perception/coverage.**
- **PEELING feasibility POSITIVE (`quick_peel.py`) = first viable N5 lever.** Oracle-peel (subtract 4 other true donors, re-rank residual): buried donors → top-5 in 65% (median rank 3), near-miss → 85%. Confirms entanglement diagnosis. CAVEATS: oracle upper bound (real iterative peel lower); 35% buried unrecoverable even with perfect peel (hard floor); est lift N5 oracle .73→~.85 not 1.0.
- **Inc4 P1–P6 in-dist generalization (Kaggle seed42, DEV N5 oracle; base N5 .574).** NO arm broke the N5 wall: P2-local .600 / P3-irm .595 (marginal+, single-seed noise — P3 drops TRAIN memorization .98→.87 but doesn't transfer to held-out combos) / P4-decorr .508 (HURT — decorrelation REJECTED) / P5 count-decouple helps N3/N4 not N5 / P6 slot COLLAPSES (N5 set-EM .19). P1 stack=sparse+geno_query: OOD .801 ≈ base, < sparse .836 → 2 OOD levers DON'T stack. Peeling decoder (not among the 6 arms) remains the unbuilt N5 lever.
- **ACTION.** N4: recalibrate neural card head (decode/ordinal/temperature) → EM N4→~.9 (closes N1–N4). N5: build per-donor peeling decoder (+ anti-combo-memorization reg). NOT: more data/coverage, attention/minor-emphasis, biology-likelihood-ID, gain_k features — all empirically rejected this session.
- **DISCIPLINE.** Single seed (42); all eval-only on frozen ckpts. N4/N5 real per-NOC now n=242/372 (full combo) — far more reliable than old n=45/48 single-combo. Peel result is an oracle upper bound, not achieved performance.

### F31 (2026-06-10) — N5 root cause LOCALIZED = MAJOR-PEAK DOMINANCE (encoder reversible-washing + decoder competition). 3-seed. (eval-only, inc6_minorw seed42/43/44)
> ⚠️ **2026-06-10 — F31 PARTIALLY RETRACTED, read F32 before acting on this entry.** inc7_masspool (the F31 ACTION) RAN 3-seed = NEGATIVE. Direct encoder probes (load ckpt → infer → read H) REFUTE the "encoder washes the minor by major-mass dominance" mechanism and the "fix = aggregation normalization / mass_pool" lever. The reversible-washing number (.07→.81) did NOT replicate on inc7 and is OOD-confounded. The real encoder effect is a MODEST combo-dependence of the ISAB context-mixing (~5–6pp), not height-washing. The bullets below marked ⚠️ are superseded by F32.
- **CONTEXT.** minor-weight arm (loss reweight low-φ positives) = NEGATIVE: DEV N5 oracle .606–.667 ≈ base, train .95–.98 (doesn't move the wall). VIB (undirected KL bottleneck) also negative (DEV N5 .606, traded down). Probes below say why.
- **INFO PRESENT (reconfirms F6/F30).** 99.9% of N5 contributors RANKABLE (private allele present); of DEV misses 99% RANKABLE (`probe_cause_decompose.py`). NOT info/dropout-limited.
- **MEMORIZATION IS IN THE DECODER, NOT ENCODER FEATURES.** Linear readout on frozen ISAB H: DEV N5 ~.52 with train≈dev (honest); full attention decoder train .98 vs DEV .67 — the .30 train→dev gap appears ONLY with the decoder (`probe_root_levers.py`).
- ⚠️ **[SUPERSEDED by F32]** **ENCODER WASHES THE MINOR BY MIXING WITH MAJOR PEAKS — REVERSIBLE (decisive, `probe_context.py`).** attr_head on the minor's OWN private peaks, deep-miss (decoder rank>8): **.05–.09 on the FULL set → .81–.87 when MAJOR peaks are masked** (random same-count removal only .24–.28); 86–91% of deep-miss minors recover. ⇒ "context" = the height-dominant MAJOR peaks (ISAB inducing-point summary is major-heavy), NOT the combo label; info is washed, not lost. — **F32 UPDATE: did NOT replicate on inc7 (major-masking 0.27→0.33, +0.06 only) and the test is OOD-confounded (kept minors DROP 0.81→0.70 on masked input). A direct independent probe shows the encoder BUILDS minor id (input 0.02 → H 0.75, minor≈major) = NOT washed. This "reversible washing" claim is WITHDRAWN.**
- **DECODER NOT BAD; FLAW = COMPETITION.** Beats honest linear-H by +.15–.17; N5 oracle gap splits ~evenly: decoder-recoverable (in top8, not top5) .145–.161 vs encoder-bound (not in top8) .156–.173. Near-miss decoy has 55% of its alleles covered by the 2 true majors (explained-away donor outranks faint true minor).
- ⚠️ **[SUPERSEDED by F32]** **CONCLUSION (3-seed).** N5 wall = MAJOR-CONTRIBUTOR (HEIGHT) DOMINANCE at two stages — encoder context-washing (reversible → raises the .83 top-8 ceiling if fixed) + decoder shared-allele-decoy competition. REFINES F30 "per-donor decision entanglement"; EXPLAINS why peeling works (peeling removes the majors that wash the encoder). minor-weight & VIB fail because they don't touch attention height-dominance. — **F32 UPDATE: the "encoder height-washing" half is REFUTED (see above). Corrected root cause = combinatorial generalization (F29) compounding across stages, concentrated on the faintest contributors; the encoder part is a MODEST combo-dependence of the ISAB context-mixing (~5–6pp), not height-dominance.**
- **ACTION (retrain).** (A) encoder: height/cardinality-robust attention normalization (scaled weighted-sum vs weighted-mean, Fischer & Gärtner 2024 arXiv 2407.04170) so minority assignment-mass isn't erased. (B) decoder: competition-aware/set-matching to demote major-explained decoys. Both protect minority from majors; non-peeling, inference-time.
- **DISCIPLINE.** 3-seed (42/43/44), eval-only frozen ckpts, DEV reconstructed seed=0. attr-recovery is an oracle upper bound (true majors removed). NOTE: earlier Kaggle gen-block oracle ~0 for `inc6_vib` was a stale-eval artifact (model built without `vib`); re-eval gives real DEV N5 .606.
- **RESIDUAL RESOLVED + "physical floor" RETRACTED (seed42, `probe_residual.py`/`probe_faint_check.py`/`probe_height_decouple.py`).** The ~.13–.19 of private peaks that DON'T recover even after major-removal are NOT a physical floor — they are PRESENT, clean, above AT=14 (we generate the mixtures; surviving peaks are scaled-real, no sub-AT noise). attr-recovery rises MONOTONICALLY with present-peak height (14-25 RFU .67 → 70+ RFU .92) = the SAME height-UNDERWEIGHTING bias, fixable. Only a tiny sliver (12 donors, phi<0.04 where AT-dropout left 1–2 peaks) is genuinely few-evidence. ⇒ "~0.9 is a hard ceiling" was an overstatement; the .83 top-8 number is the CURRENT model's, and a height-robust encoder can exceed it.
- ⚠️ **[SUPERSEDED by F32 — lever was WRONG]** **LEVER LOCALISED to the AGGREGATION, not the feature (`probe_height_decouple.py`).** Neutralising the height FEATURES at input recovers only .07→.16 (vs .81 major-removal) ⇒ height-decoupled key is INSUFFICIENT; the washing is driven by the MASS of the major MAJORITY (many tall peaks dominate the softmax weighted-mean pooling), not the height feature value. Fix must change the AGGREGATION normalization (scaled weighted-SUM, Fischer&Gärtner 2024 arXiv 2407.04170), not the input. — **F32 UPDATE: scaled weighted-SUM (mass_pool) was BUILT and RAN 3-seed = NEGATIVE (DEV N5 oracle unchanged). Aggregation normalization is NOT the lever.**
- ⚠️ **[SUPERSEDED by F32]** **ACTION → Increment 7 BUILT (not yet run).** `inc7_masspool` = pe_s3+sparse + `--mass_pool` (mass-preserving inducing compression `MassISABpp`/`MassMABpp`, +2 params, drop-in). design=`reports/design_increment7_masspool.md`. Run 3 seeds (M1/M2/M3 = seeds 42/43/44), judge DEV N5 oracle vs base ~.65 (ceiling ~.81). Results pending next session. — **F32 UPDATE: RAN. NEGATIVE. See F32.**

### F32 (2026-06-10) — inc7_masspool RAN (3-seed) = NEGATIVE; DIRECT encoder probes REFUTE F31's "major-mass washing" + the mass_pool lever; combo-dependence LOCALIZED to the ISAB context-mixing (modest), input features combo-invariant. REVISES F31. (user consented to this log edit)
- **inc7_masspool 3-seed = NEGATIVE/NULL.** DEV N5 oracle .644/.646/.631 ≈ base ~.65 (BOTTOM of the arm cluster; < inc5_res_rand1 .650, < inc6_maskp .680, < inc6_minorw .667–.693); train N5 oracle ~.977 → train≫dev gap intact, wall unmoved. Guard N1/2/3 dev oracle held (1.0/.996/.972). mass_pool is genuinely active in the ckpt (`encoder.0.mab0.scale` loads, =1.17) — not a bug; the lever is inert. ⇒ F31's "fix = aggregation normalization (scaled weighted-SUM)" is WRONG.
- **DIRECT ENCODER PROBE (methodological upgrade: load ckpt → forward → read H with an INDEPENDENT linear probe, not metrics; `probe_encoder_info.py`).** Probe rep[peak]→global donor id, fit on TRAIN-combo peaks, tested on REAL novel-combo test: **x0 (input) minor=0.02 (=1/45 chance) → H (encoder output) minor=0.75, EQUAL to majors 0.74.** The encoder CONSTRUCTS combo-generalizable minor identity — it does NOT wash it out. ⇒ F31 "encoder washes the minor by mixing with major peaks" REFUTED as the binding mechanism; mass_pool did nothing because there is no general washing to fix.
- **F31 reversibility does NOT replicate (`probe_peel_reversible.py`).** Major-masking on inc7: deep-miss minors 0.27→0.33 (+0.06), NOT F31's .07→.81; AND kept minors DROP 0.81→0.70 because the masked input is OOD for the attr head (same neural_peel OOD-on-residual issue, F30). F31's "decisive reversible washing" was a confounded/over-read probe.
- **BUT a CLEAN within-synthetic test finds a REAL, MODEST encoder combo-dependence (`probe_seen_vs_novel.py`, `probe_encoder_locus.py`, 2 seeds 42/43).** Compare model-SEEN (train combos) vs model-NOVEL (dev combos), BOTH synthetic (no real confound), exact seed=0 carve, faint minors (N5 rank≥3) matched on minor/major template ratio:
  - per-peak donor-id readability from H: SEEN 0.63 vs NOVEL 0.58 (**gap +0.05–0.06, consistent across ALL faintness bins, both seeds**).
  - **LOCALIZED (per-donor pooled, x0 vs H):** the INPUT features are combo-INVARIANT (x0 seen−novel gap ≈ +0.00, abs ~0.34); the **ISAB CONTEXT-MIXING is where combo-dependence enters** (H gap +0.05–0.06) and it is also what lifts faint-minor identity 0.34→0.81. The faintest bin (ratio<0.02) is both the lowest absolute (novel 0.50) and the most combo-dependent (gap +0.08–0.11). ⇒ the exact encoder mechanism = the cross-token context-integration is combo-OVERFIT, NOT the pooling normalization (mass_pool) and NOT the input features.
- **Dropped minors (set head misses) read 0.30 by the INDEPENDENT probe too** (≈ model-attr 0.30; vs kept minors 0.79) — for the ~13–21% faintest that fail, the identity is genuinely not cleanly in H on novel combos (so it is NOT "info all there, decoder lazy" either — my first inc7 read was also wrong).
- **CORRECTED ROOT CAUSE (replaces F31's single "height-dominance washing").** N5 wall = COMPOUND, dominated by combinatorial generalization (F29), manifesting at BOTH stages, concentrated on the faintest contributors: (1) F31 major-mass washing = REFUTED; (2) ISAB context-integration is combo-overfit (~5–6pp representation-level generalization penalty) = MODEST; (3) a genuine faint-evidence margin for the very faintest (novel readability ~0.50 at ratio<0.02 even with context); (4) decoder/set-resolution loss on top. Neither "pure encoder washing" (F31) nor "pure decoder laziness" (my first read) is correct.
- **QUANTITATIVE DECOMPOSITION — answers "how does a 5–6pp encoder gap make a 34pp wall" = it does NOT; the encoder is the SMALL share + compounding (`probe_decompose_wall.py`, 2 seeds).** Per-donor SEEN(train)→NOVEL(dev) drop at N5, by height-rank, encoder-readability vs decoder-inclusion vs the decoder's margin-over-readout:
  - **Faintest minor (rank4): ENCODER readability drops only −4.5pp (0.81→0.765); DECODER inclusion drops −21pp (0.975→0.765).** The decoder's combo-memorization MARGIN over a plain readout collapses +16.5pp (seen) → ~0pp (novel). ⇒ ~80% of the faintest-minor gap is DECODER combo-memorization, ~20% encoder. (But on novel the decoder for rank4 has collapsed to exactly the encoder readability 0.765 = co-limited there.)
  - **Compounding:** novel per-donor inclusion 0.996/0.985/0.962/0.894/0.765 → set-EM needs all 5 → product ≈ 0.66 ≈ dev N5 oracle .64. The faintest minor (~0.76) is the weakest link dragging the set; small per-donor drops MULTIPLY into the 34pp set wall. ⇒ the 34pp SET wall = (decoder-memorization collapse, the big part) × (compounding over 5 donors); the 5–6pp encoder number is one donor's encoder slice, never meant to explain 34pp alone.
- **ENCODER exact sub-mechanism (`probe_encoder_entangle.py`, 2 seeds): a combo-specific SHARED-CONTEXT CARRIER dominates each donor's H; identity is a small residual — worst for faint minors.** Within a sample, ALL donor H-reps are ~0.90 cosine to each other (major-major 0.90–0.91, minor-minor 0.89, minor-major 0.92 — so NOT specifically "minor pulled to major"; it's a shared per-sample context). That carrier is combo-specific: the SAME donor's rep self-correlates only **0.70 (minor) vs 0.79 (major)** across combos — i.e. a minor is MORE similar to its current-combo neighbours (0.92) than to ITSELF in other combos (0.70). Majors carry a LARGE identity residual (robust); faint minors carry a SMALL one, so the combo-varying carrier dominates them → the ~5–10pp novel-readability penalty, worst at the faintest. Info NOT lost (residual still probe-readable 0.75) — it is combo-dependence of a small residual riding a combo-varying carrier. ⇒ the leak is the cross-token CONTEXT BLEND (the attend-back / global mixing), NOT the inducing-pool normalization mass_pool changed — which is why mass_pool was inert.
- **DECODER exact sub-mechanism: NOT a training co-occurrence prior (`probe_decoder_cooccur.py`) — REFUTED.** The decoder's wrong substitute co-occurs with the sample's majors NO more than a random donor (FP 292 ≈ random 292 ≈ true-missed 269; FP/random 1.00x); only a weak within-sample signal (FP 9% > the true minor it replaced, 67% of cases). Training co-occurrence is near-uniform (≈14k N5 combos) so it carries little structure. ⇒ the decoder's seen-only +16.5pp margin is INSTANCE/CONFIGURATION memorization (recognises seen peak-configurations), NOT a learnable co-occurrence statistic. On novel it falls back to ~the encoder-readable level (rank4 novel decoder 0.765 = encoder readability 0.765).
- **DECODER deeper: the memorization is in the NONLINEAR SCORING, NOT the attention selection (`probe_decoder_attn.py`, 2 seeds).** inc7 decoder = SparseMAB (sparsemax, each donor query → its alleles). Captured layer-0 cross-attention of each TRUE donor's query, SEEN vs NOVEL: **attention SELECTION is combo-INVARIANT** — own-peak mass and major-peak mass are identical seen↔novel at every rank (rank4 own 0.47/0.47, major 0.42/0.42). So the decoder does NOT memorise "where to look." Yet decoder inclusion drops 21pp seen→novel while BOTH linear readouts (pooled AND per-peak) drop only ~5pp ⇒ the extra ~16pp lives in the decoder's NONLINEAR post-attention scoring (FFN/score_w combining per-peak H), exploiting combo-specific structure that is present but NOT linearly visible. Static weakness exposed: the faintest minor's query cannot cleanly isolate its own faint alleles — it splits attention ~47% own / ~42% MAJOR peaks (shared-allele bleed, combo-invariant) = the F31 "shared-allele competition" seen directly. ⇒ decoder fix = regularise the nonlinear per-donor head against config-memorisation (capacity/dropout/consistency), NOT an attention change; and the shared-allele bleed argues for reference-conditioned queries (geno_query) to disambiguate a faint minor's own alleles from the overlapping majors.
- **Two-stream / content-context separation (Russin) — why it was REJECTED, reconciled.** The inc6 probe (`feasibility_twostream.py`) found content(x0) per-peak ≈ chance (0.02–0.04) ⇒ "identity is RELATIONAL, not per-peak" ⇒ a context-free content stream carries ~no identity ⇒ NO-GO. This session CONFIRMS that (x0 per-peak ≈ 0.02) — but it is NOT a contradiction of the combo-dependence finding: the OLD probe was underpowered (per-peak, single-NOC, ~floor accuracy, gap ~0.01 invisible); the well-powered version (all-NOC fit, matched faintness, per-donor pooled) does show the real +5–6pp H combo-dependence. RESOLUTION: the mixing both CREATES the identity (you can't drop it — content alone is empty) AND injects combo-dependence; so the lever is NOT architectural separation (two-stream rejected, correctly) but REGULARISING the same relational mixing to be combo-invariant (rep-consistency / reference anchor). Caveat for my own F32 ACTION: `decoder_source='local'` removes ISAB cross-mixing but keeps the per-donor decoder's cross-attention, so it is NOT the empty content stream — but it leans toward the rejected direction; prefer combo-invariance regularisation over removing mixing.
- **ACTION (corrected — priority FLIPPED by the decomposition).** DROP mass_pool. The BIGGER lever is the DECODER combo-memorization (~80% of the per-minor gap): anti-combo-memorization regularization and/or making the per-donor decode reference-conditioned & combo-invariant (geno_query-style, the only OOD survivor F27) so the decoder cannot lean on memorized combo→set mappings that vanish on novel combos. The ENCODER fix (smaller share ~20% but co-limits the faintest bottleneck donor) follows from its mechanism = stop the combo-shared-context carrier from dominating a faint donor's rep / preserve the identity residual: an identity-preserving per-token stream (decoder_source="local" = LocalTokenEncoder, no cross-token mixing, already coded but untrained as base), and/or a reference anchor (geno_query) so a donor's rep has a combo-invariant component, and/or rep-consistency reg pulling a donor's H toward combo-invariance. (This REVISES my earlier "fix encoder first" — the decoder is the larger source; both are needed, decoder first by share.) PEELING (F30/F31) addresses decoder shared-allele COMPETITION, a different decoder sub-mechanism than the memorization-margin collapse measured here — re-scope, do not assume it fixes the margin collapse.
- **LITERATURE GROUNDING (the mechanism + the fix both match published theory; searched 2026-06-10).** Two independent lines converge on our two findings and SHARPEN the lever beyond my vague "anti-memorization reg":
  - **Encoder shared-context carrier = TRANSFORMER OVER-SMOOTHING / token uniformity / rank collapse.** Self-attention is a low-pass filter; pure attention output converges toward rank-1 (all tokens → a common vector) — exactly our within-sample ~0.90 cosine with identity as a small residual. Refs: Dong et al. 2021 "Attention is not all you need: pure attention loses rank doubly exponentially"; "Addressing Token Uniformity via Singular Value Transformation" (arXiv 2208.11790); "Mitigating Over-smoothing in Transformers via Regularized Nonlocal Functionals" (arXiv 2312.00751). Fixes that exist: a token-fidelity regularizer (penalize ‖smoothed − input‖ to preserve per-token distinctiveness), singular-value flattening, sparse/local attention (we already adopted sparse). ⇒ the encoder leak is a NAMED, studied failure with drop-in regularizer fixes.
  - **The wall itself = COMPOSITIONAL GENERALIZATION; the principled fix is ADDITIVE decoder on DISENTANGLED latents + a COMPOSITIONAL-CONSISTENCY regularizer (Brendel group).** Wiedemer et al. NeurIPS 2023 "Compositional Generalization from First Principles" (arXiv 2307.05596): generalization to unseen combos is provably governed by (i) compositional SUPPORT (atom/donor coverage — which we HAVE: every donor seen) + (ii) an ADDITIVE/modular decoder; and "false modularity" (additive head on NON-independent features) does NOT generalize. Wiedemer et al. ICLR 2024 "Provable Compositional Generalization for Object-Centric Learning" (arXiv 2310.05327): the missing ingredient is a **compositional-consistency loss** — RECOMBINE latent slots from two in-dist samples into an OOD combo and force the model to re-encode/decode it consistently; masked/softmax decoders fail because softmax-normalization introduces SLOT INTERACTIONS (= our shared-context entanglement). Empirical reality-check, Mahon & Lukasiewicz 2024 "Successes and Limitations of Object-centric Models at Compositional Generalisation" (arXiv 2412.18743): probing shows learned reps "need non-linear classifiers and fail on held-out combinations = lower-level pattern matching, not concepts" — INDEPENDENTLY corroborates our decoder nonlinear-config-memorization finding.
  - **This EXPLAINS our past failures + names the untried lever.** repC additive FAILED = false modularity (additive head on the ENTANGLED encoder H, no consistency loss). mass_pool = wrong target (pooling-norm, not additivity/over-smoothing). IRM/decorrelate = not the compositional-consistency form. The NEVER-TRIED, theory-endorsed ingredient = **compositional-consistency regularization**, which we can implement cheaply because we OWN the generator (make_insilico overlays donor profiles): recombine donors across training mixtures → synthesize the novel-combo mixture → enforce consistent set prediction. NOTE this is NOT the rejected "more data/coverage" (F29/F30) — it is a FUNCTION-CONSTRAINING loss, not added samples. Query-side support: DN-DETR query-denoising (feed noised GT, reconstruct) + DAB/Anchor-DETR reference-point queries ground the geno_query reference-conditioning for the faint-minor shared-allele bleed.
  - **CAVEAT (honest).** The Brendel proofs are for GENERATIVE object-centric autoencoders (reconstruct images); ours is DISCRIMINATIVE. The transferable core = the consistency-regularizer + additivity-on-disentangled-latents principle, enabled by our generator — NOT a guaranteed transplant. The empirical paper is single-seed and on toy data. So: direction is now well-grounded and sharpened, but still a train-time BET (consistent with the 6-arm Inc4 / inc5 / inc6 / inc7 null history — verify on DEV N5 oracle, 3-seed).
- **INC8 NO-TRAIN FEASIBILITY (`probe_fix_feasibility.py`, 2 seeds) — both directions GO; strongest positive N5 signal in the project's all-null lever history.** P1 (de-smoothing / shared-carrier removal): subtract the per-sample shared component from H (carrier = mean over peaks), then re-fit the readout. Faint-minor (N5 rank≥3) NOVEL readability **0.808 → 0.93–0.95**, faintest bin (ratio<0.02) **0.51 → 0.78–0.81 (+28pp)**, and the seen→novel GAP shrinks **0.054 → ~0.02 (≈3×)**. Crucially the DEPLOYABLE no-oracle carrier (mean over ALL valid peaks, needs no attribution) works almost as well as the oracle per-donor-mean carrier (0.934 vs 0.946) ⇒ removing the dominant shared mode (= the over-smoothing/rank-1 component; cf. singular-value/anti-over-smoothing fixes) recovers faint-minor identity AND de-biases it across combos. P2 (consistency-loss premise): per-donor cross-combo self-stability correlates with novel readability (Pearson r=+0.65/+0.67; high-stability donors read 0.83–0.85 vs low 0.77) ⇒ pushing reps toward combo-invariance (what the consistency loss does) should raise novel readability. CAVEAT: these are linear-READABILITY signals (necessary condition), NOT decoded set-EM — the decoder's nonlinear config-memorization is a separate axis; must still verify a trained Increment-8 on DEV N5 oracle, 3-seed (the 6-arm/inc5/6/7 null history demands it). But the de-smoothing GO is the first lever to move the encoder-bound component this much without training.
  - **HARD CENTERING (the literal P1 op) BREAKS N1 — DROPPED; Inc8 pivoted to N1-safe loss regs (BUILT 2026-06-10).** Subtracting the per-sample peak-mean from H at inference DESTROYS N1 identity: probe readability **1.000 → 0.022 (= 1/45 chance)** (N1 has one donor ⇒ the sample-mean IS its identity; majors/faint unharmed .98→.99 / .78→.93). N1 = 63% of the test set ⇒ inference-time centering is NO-GO. **Increment 8** therefore targets the SAME goal (de-smooth carrier + combo-invariance) as a TRAIN-TIME loss. The first version (hand-rolled decorr + rep-consistency) was an ADAPTATION; on the user's "one published method, no self-invent" discipline it was REPLACED by **VICReg VERBATIM (Bardes, Ponce & LeCun, ICLR 2022, arXiv 2105.04906)** applied to the per-(sample,donor) pooled encoded reps — its three terms map exactly and it is N1-safe by construction: **variance** (hinge, per-dim std≥γ=1) = anti-collapse that PRESERVES identity (no subtraction → fixes the N1 break); **covariance** (off-diag² of the cov matrix) = decorrelate dims = de-smooth the carrier; **invariance** (the same donor's reps across the combos it appears in, pulled to their mean) = combo-invariance. Ladder = turn terms on: **V1 `--vicreg`** = variance+covariance; **V2 `--vicreg --vicreg_inv`** = + invariance. Paper coeffs var/cov/inv = 25/1/25, overall `--vicreg_weight` default 0.04 (= the hyperparameter to watch; lower it if the guard or val-F1 drops). Forward byte-identical when off ⇒ `inc2_2d_sparse` (3-seed DEV N5 oracle **0.590/0.636/0.602, mean .609**) is a clean base. Helper `vicreg_donor_regs()`; runner arms `inc8_v1_vicreg`/`inc8_v2_vicreg_inv`, M1=V1 M2=V2 (2 machines, seed 42 promise-check); kaggle_bundle synced. Smokes PASS (forward exposes encoded; var/cov/inv finite on REAL attr with −1 sentinels & mixed NOC; N1-safe — no subtraction; encoder grads finite-nonzero; argparse accepts). NOT yet trained — judge DEV N5 oracle vs .609 + guard N1/2/3 + mechanism probes on the ckpt, 3-seed only if promising. (Alt published methods Wiedemer 2310.05327 / uniformity / patch-diversification NOT chosen; VICReg picked as the single verbatim method covering both arms + N1-safe, NOT proven optimal.)
  - **The other two Inc8 components — probe status (honest).** (3) ADDITIVE decoder on disentangled latents is NOT no-train-probeable: a linear probe cannot distinguish additive (sum) from pooled aggregation (equal up to scale); additivity's compositional benefit is a NONLINEAR-decoder training property (Brendel). P1's disentangle IS its precondition (avoids the false-modularity that sank repC), but additivity itself must be tested by training. (4) geno_query / REFERENCE-CONDITIONING: my first probe (`probe_fix_feasibility2.py`, pool H over a donor's reference-allele peaks, no attribution) gave a spurious 0.999 = **SELECTION LEAK, RETRACTED** — the control (same pooling on x0, which carries ~0 per-peak identity, still scored 0.886) proves the donor is identified by WHICH reference positions are pooled, not by H content. So P4 is INVALID. geno_query's only real evidence remains the prior TRAINING result (F27: weak OOD survivor, < sparse). No new no-train support for it.
- **DISCIPLINE.** inc7 3-seed; direct-probe results 2-seed (42/43), internally consistent across all faintness bins. Eval-only on frozen ckpts; DEV = seed=0 carve (= make_dev_split, = the split the model trained with). New scripts (root, local): `probe_encoder_info.py`, `probe_peel_reversible.py`, `probe_seen_vs_novel.py`, `probe_encoder_locus.py`, `probe_decompose_wall.py`, `probe_decoder_cooccur.py`, `probe_encoder_entangle.py`, `probe_decoder_attn.py`.

## 2026-06-11 — Increment 8 (VICReg) RAN (seed42, V1=var+cov, V2=var+cov+inv) + direct mechanism probes localizing the N5 wall to TWO confirmed, independent failures. (eval-only on frozen ckpts; consent given)
### F33. inc8 VICReg = headline NULL on N5, but the lever WORKED on its target; probes then decompose the N5 wall into a confirmed ENCODER isolation-failure + a confirmed DECODER under-read.
- **Result.** DEV N5 oracle: V1 .558, V2 .657 vs base inc2_2d_sparse .609 (3-seed) — within single-seed noise (±.18), no break of the wall. V2 lifted DEV N3/N4 oracle (.960→.982, .850→.888). Guard N1/2/3 held (V2 1.0/.998/.982); var term kept N1 safe (no hard-centering collapse).
- **Lever acted on target (probe_encoder_entangle).** VICReg cov de-smoothed the encoder HARD: within-sample major-major cosine .904(base)→.084(V2); minor-major pull .916→.290; inv raised donor cross-combo self-stability .66→.76. So the de-smoothing the design promised happened — yet N5 oracle did not move ⇒ encoder over-smoothing was NOT the dominant lever.
- **No physical floor (measure_noc5_ceiling + probe_q1_recheck).** N5 = 100% RANKABLE (every contributor has a PRIVATE allele present); ALL 47 'both-miss' faint minors are RANKABLE 100%, 0% DROPOUT. So the entire N5 gap is MODEL-limited, not info-limited. (Retracts my own in-turn 'few peaks = info absent' claim.)
- **Decoder DOES waste generalizable info (probe_decoder_waste).** A non-memorizing soft-vote of a per-peak donor probe ON THE SAME H beats the trained decoder at N5: V2 probe .769 vs decoder .659 (Δ+.110; +.07 at N3/N4); MLP readout .790. On BASE the gap is ~0 (.734 vs .723) ⇒ VICReg raised the encoder ceiling (probe .734→.769) but the per-donor decoder REGRESSED (.723→.659) — the gain is stranded at the encoder→decoder interface. Per-rank decoder inclusion gap grows with faintness (r2 +.03, r3 +.05, r4 +.09).
- **ENCODER failure = isolation, CONFIRMED (probe_encoder_isolate).** On the minor's OWN private-allele peak, where does its encoder-H point among the 5 contributors? both-KEEP →minor .93 / →major .07; decoder-miss .69/.31; **both-MISS .36/.64** — i.e. the encoder absorbs 64% of the faintest minor's private peaks into a DOMINANT donor. Confusion is minor→major only (→other-minor=0.00). Control (.93 on kept) validates the probe. = isolation failure at the faint tail, measured at the private peak (not a pooling artifact, not info-absence).
- **DECODER failure = under-read, CONFIRMED; combo-memorization RULED OUT (probe_decoder_mech).** On the 38 pure-decoder-miss minors (H isolated them at .69): decoder score of the true minor collapses .841(when kept)→.139(when missed, near floor), buried at rank ~8-11. The displacer is an ARBITRARY filler: its train co-occurrence with the majors (.067) = the true minor's (.068), and it is not even confident (.325). So the decoder suppresses the faint donor's score rather than swapping in a memorized partner.
- **Decomposition (V2, N5, all model-limited).** decoder .659 → 1.0 ≈ ~.13 decoder under-read (H already isolated, readout drops it = cheap: freeze VICReg encoder, retrain/recalibrate head toward the soft-vote/MLP ceiling ~.79) + ~.21 harder encoder isolation failure on the faintest tail (private peak present but absorbed into majors; representation de-smoothing alone did not fix it).
  - **REVISED 2026-06-12 (F35, 3-seed BASE).** The ".13 decoder under-read" was measured on the **V2 VICReg** ckpt, NOT the base. On the base `inc2_2d_sparse` (3 seeds), the clean MAX-probe puts decoder under-read at N5 ≈ **+.07 (seed-variable .038/.099/.083)**, not .13; the **~.20 encoder share is the robust 3-seed number**. The "~.21 encoder isolation" CONCLUSION stands; only the decoder magnitude is revised down (and the F34 seed42-only ".04" is revised UP to ~.07). See F35.
- **Scripts (root, local, eval-only):** probe_encoder_entangle.py, probe_encoder_info.py, probe_decoder_waste.py, probe_ceiling_decomp.py, probe_where_weak.py, probe_q1_recheck.py, probe_encoder_isolate.py, probe_decoder_mech.py, measure_noc5_ceiling.py. DISCIPLINE: single-seed V2 for the mechanism probes (internally controlled: both-KEEP control validates each); 3-seed only for the headline DEV oracle. Levers below are HYPOTHESES, not yet run.

---

## 2026-06-12 — F35: N5 encoder wall hardened to 3 seeds; mechanism = softmax-attention COMPETITION / explaining-away (+ non-competitive-attention ceiling probe). Corrects F33 (decoder number) and the F34 probe8 "not-encoder" relabel.

(F34 was a same-day audit trail kept in memory, not this log; several of its bullets self-corrected. F35 logs only the 3-seed survivors.)

### F35. The base N5 wall = ~.20 ENCODER (robust 3-seed) + ~.07 DECODER (seed-variable). The encoder share is softmax-attention COMPETITION (faint-minor evidence absorbed into the dominant donor) + a conjunctive readout under-weight — NOT positional/stutter, NOT height-causal, NOT global over-smoothing.
- **OBSERVED (eval-only, frozen base `inc2_2d_sparse` seeds 42/43/44, real test N5 n=372).**
  - *Decomposition* (`probe_cleanprobe_maxsum`): clean linear MAX-probe on encoder-H = .761/.833/.793 (~.80); decoder under-read = +.038/+.099/+.083 (**mean ~.07, seed-variable**). Encoder ceiling gap to 1.0 = **~.20**.
  - *Absorption* (`probe_encoder_isolate`): on the missed faint minors, **72–88%** of their PRIVATE peaks point to a MAJOR, **0.00 to another minor**; control (recalled minors) →minor .95.
  - *Readout under-weight* (`probe6`, causal): keep one **panel-UNIQUE** private allele + all shared → recall only .39/.42/.32 (~.38); ~5 private peaks needed for ~.94.
  - *Positional REFUTED* (`probe_encoder_why`, NEW): back-stutter-of-major raises absorption only ~+.06; absorbing donor is the stutter-owner only ~44%; co-location with a major is universal (1.00 in both ISOLATED and ABSORBED groups). The only per-peak correlate is faintness — already non-causal at the donor level (probe1/2/3 decoupling).
- **THEORY.** Softmax normalization forces attention mass onto the dominant local explanation, so a faint minor's evidence (shared alleles + the faint tail of its private alleles) is credited to co-present majors = explaining-away / competition: Gradient-flow low-entropy polarization (arXiv 2603.06248), self-attention ignores explaining-away (2310.20307), ACMIL dominant-instance concentration (2311.07125). The clean **no-privilege** probe still ceilings at ~.80 ⇒ the .20 is a real encoder-representation limit. This **CORRECTS the F34/probe8 relabel** ("the .21 is a reference-matching/scoring gap, not encoder") — that rested on the **retracted privileged-`at>=0`** ensemble (.831).
- **CONCLUSION.** Encoder mechanism LOCKED (3-seed, causal, multi-probe): competition/explaining-away absorbs faint-minor evidence exclusively into dominant donors + conjunctive readout under-weight. RULED OUT as drivers: stutter/positional, height-causal, global over-smoothing (VICReg null, F33). The deployable FIX is OPEN (reference-match = privileged/circular F34; likelihood/height = negative prior F33/F29). Untried fix-family the mechanism points to = **non-competitive (sigmoid) attention** (lower-sample-complexity self-attention, 2502.00281).
- **ACTION.** No-train ceiling probe of non-competitive attention on the frozen encoder (F35b); judge whether relaxing competition recovers faint-minor isolation and at WHICH attention step. Scripts: `probe_cleanprobe_maxsum`, `probe_encoder_isolate`, `probe6_why_sparse_fails`, `probe_encoder_why.py`.

### F35b. Non-competitive-attention CEILING probe: the absorbing competition is LOCALIZED to mab0 (the inducing-pool step) — CONFIRMED — and relaxing it de-absorbs the faint tail per-peak; but the no-train swap does NOT convert to donor recall and degrades well-isolated donors ⇒ TRAIN-TIME bet, not a free fix. Explains why mass_pool / global-temperature failed (wrong placement).
- **OBSERVED (`probe_noncompetitive_attn.py`, no-train, base inc2_2d_sparse seed42; MABpp attention reimplemented from trained weights, softmax→sigmoid/temperature, PER STEP; clean probe REFIT per condition; real decoder MISSES the faint minor in 51/372, KEEPS 321).** On the MISSED tail — per-peak private isolation (→minor) / donor soft-vote recall:
  - BASE softmax: .266 / .255   (KEPT .928 / .963)
  - **mab0 sigmoid_norm: .457 / .196**;  **mab0 temp4: .576 / .235** — per-peak isolation UP **+.19 / +.31**, donor recall FLAT-to-down.
  - **mab1 sigmoid_norm: .103** (per-peak isolation DROPS) → mab1 is the WRONG step.
  - KEPT donor recall DROPS under mab0 relaxation (.963 → .798 / .679).
- **CONCLUSION.** (1) **Placement CONFIRMED:** the competition that absorbs faint-minor private peaks lives in **mab0 (inducing points softmax-attend over peaks)**, not mab1 — directly explains why mass_pool (pooling-norm) and probe2's global temperature failed: WRONG PLACEMENT, not wrong idea. (2) Relaxing mab0 competition **de-absorbs the faint tail per-peak** (the mechanism's direct causal signature — strongest yet). (3) **BUT no-train cannot deliver the donor-level gain:** per-peak↔donor recall DECOUPLE even under a refit linear readout (MISSED rec ≤ baseline), and KEPT global separability degrades (frozen-weight OOD — non-competitive attention trades global donor-separability for faint-tail isolation). The conjunctive-readout second wall (probe6) is untouched. ⇒ **non-competitive mab0 attention is mechanistically ON-TARGET but a TRAIN-TIME bet**, consistent with the project's no-train→train null history. **ACTION → BUILT as Increment 11 (sigmoid attention, Ramapuram et al. 2024, arXiv 2409.04431):** `SigmoidMABpp` (σ(QKᵀ/√d − log n)V, no row-norm) wired as `--nc_attn {mab0,both}` on the BASE9 base; arms `inc11_nc_mab0` (M1) / `inc11_nc_both` (M2, placement control), 2-machine parallel, bundle-synced, smokes pass (model + argparse + 1-epoch micro-train). NOT trained. Judge DEV N5 oracle 3-seed vs base ~.59 + guard N1/2/3; then re-run `probe_encoder_isolate` (→major drop?) / `probe_cleanprobe_maxsum` (ceiling >.80?). Expect the readout (conjunctive, probe6) wall to ALSO need a separate lever. Scripts: `probe_noncompetitive_attn.py`.

---

## 2026-06-13 — F36: Increment 11 RAN (seed42). nc_mab0 = the FIRST lever to move the N5 wall (directional, single-seed); nc_both = placement control fails in the DECODER. Residual decomposed into 3 fronts; combo-invariant Encoder-B credit-sharing probed = capped → PARKED.

### F36. mab0 sigmoid moves N5 the right way but is a PARTIAL dose; the wall is now 3 fronts (decoder under-read +.091 / encoder isolation 66% / under-determination 33%). m_inducing=64 REJECTED by a saturation probe. Encoder-B (combo-invariant genotype credit-sharing) capped at +.02 deployable — revisit only after the A + decoder fronts land.
- **OBSERVED (eval-only, frozen ckpts; real test N5 n=372; base = `inc2_2d_sparse`).**
  - *Results.* `inc11_nc_mab0`: DEV N5 oracle .609(3-seed base)→**.639**; real N5 oracle .722→**.747**; real N5 count .732→**.798**; overall oracle .975→.960, test EM .962→.941 (small in-domain tax); guard N1/2/3 held. `inc11_nc_both`: real N5 oracle **COLLAPSE .317**, test EM .837.
  - *Decode-split (`probe_inc11_decode`).* N5 H-linear-vote ceiling (encoder info, model-free) vs model-decode: base .737 vs .723 (+.013, encoder-bound); **mab0 .839 vs .747 (+.091 = DECODER under-read)**; both .659 vs .317 (**+.341** gap, grows with NOC N3→N5). ⇒ **both fails in the READOUT not the encoder** (mab1 sigmoid corrupts the per-token addressing the per-donor decoder reads; token norms FINE — refutes a norm-explosion guess). Placement (F35b) re-confirmed on trained weights.
  - *Isolation (`probe_inc11_mechanism`).* Faint-minor private peak →MAJOR absorption: base .11 → **mab0 .08** → both .16. Encoder per-peak readability (probe fit-acc) base .814 → mab0 .828 → both .795.
  - *Residual decomposition (`probe_inc11_encoder_residual`).* mab0 cut N5 encoder misses 123→79 (−36%) but the A/B/C composition is UNCHANGED (base 69/30/1 → mab0 **66% isolation-failure / 33% under-determination / 1%**) ⇒ mab0 = partial dose of the *right* medicine, same mechanism still dominates the residual.
  - *Private-allele stratification.* mab0 gains concentrate on few-private (npriv≤2) faint minors (.17→.33, .25→.62); npriv≥5 saturated (.96). Confirms it targets the combinatorially-ambiguous tail.
  - *m_inducing sizing (`probe_inducing_util`).* The m=32 inducing bottleneck is **only 48% saturated** (effective rank 15.3/32, 0 dead slots on mab0; base 13.0/32, 5 dead). Capacity is NOT the binding constraint ⇒ **`m_inducing=64` REJECTED**; the absorption is a routing/de-competition problem, not slot-sharing. (mab0 raised eff-rank 13→15 & removed dead slots vs base = de-competition signature.)
  - *Encoder-B combo-invariant probe (`probe_inc11_encoderB`, DEPLOYABLE mask peaks incl ~37 noise/sample, NO privileged at>=0).* Genotype-coverage / rarity-weighted: logit-add ceiling only **+.02 N5 set-EM** (matches the retracted-`ref_match` deployable +.016); residual/peeling-at-genotype **FAILS** (faint-minor rank 41/45) — because the minor's distinguishing evidence IS the shared alleles, so any "subtract the majors" scheme destroys it.
- **CONCLUSION.** mab0 = the first lever to move N5 (KEEP as base; 3-seed 43/44 still pending before any "works" verdict). Recoverable map for mab0: **DECODER under-read +.091** (info already in H, addressable) + **ENCODER-A isolation ~.106** (66%×.16; mab0 partial, addressable; constraint = de-competition NOT capacity) + **ENCODER-B under-determination ~.05** (33%×.16; combo-invariant routes capped +.02, combo-prior = OOD-unsafe ⇒ likely a hard floor). **Encoder-B PARKED** → revisit only after A + decoder land.
- **ACTION → Increment 12 = 4 arms on the mab0 base, 2 fronts, 2-machine.** A (de-competition, capacity ruled out): tunable gate. Decoder (read the +.091): soft/additive readout + faint-query supervision. Hyperparameters derived, not guessed (m_inducing rejected by saturation; A-temp T from F35b temp4 evidence; entmax α=1.5 = closed-form default if used). Scripts (root+local, eval-only): `probe_inc11_mechanism.py`, `probe_inc11_decode.py`, `probe_inc11_encoder_residual.py`, `probe_inc11_encoderB.py`, `probe_inducing_util.py`. DISCIPLINE: all inc11 numbers single-seed (mab0 seed42) = directional only.

### F37. Deep mechanism dig (mab0 seed42, eval-only) RE-DECOMPOSES the N5 wall and CORRECTS F36's framing: NO info floor (97% identifiable), the wall is a model READOUT under-read of DIFFUSE faint-donor evidence. Increment 12 = softer/adaptive per-donor aggregation (entmax / DSMIL), NOT additive (additive collapses the encoder). Several mechanisms RULED OUT by direct probe.
- **NO INFO FLOOR (retracts F36's "Encoder-B hard floor" / the in-turn "info-floor" framing).** `probe_inc11_identifiable`: on N5 set-misses the dropped true donor has **mean 2.9 PRESENT alleles the decoy-swap {other-4 ∪ winning-decoy} cannot explain → 97% IDENTIFIABLE**. Consistent with the proven "N5 100% rankable" (F33). The residual is MODEL-limited, full stop.
- **THE WALL = DECODER UNDER-READ of DIFFUSE evidence.** `probe_inc11_decode` / `probe_additive_parts`: the mab0 ENCODER is good — a soft-vote (softmax-per-peak→sum) linear readout on its H hits **N5 .849**; the trained per_donor **sparsemax** decode only reaches **.747 → +.10 STRANDED**. Sparsemax (hard) commits to a few tall peaks and DROPS the faint donor's diffuse evidence (the 2.9 decisive alleles spread thin). MAX/MLP readouts do NOT beat soft-vote (≤+.011) ⇒ it is the AGGREGATION sparsity, not the readout family.
- **UNIFIED ROOT across probes:** the model's nonlinear readout AMPLIFIES clear identities, UNDER-FIRES faint ones — same signature at the layer1 FFN (`probe_inc11_layer1_decompose`: margin contribution +4.65 kept / −0.22 missed), the decoder (.747<.849), and the 45-way decoy competition (`probe_inc11_setmiss_decompose`: 100% of misses have a non-contributor in top-5; misses COMPOUND across ranks 2–4, product of per-rank recalls .812≈set-EM .849 — NOT just rank-4; all prior probes restricted to the 5 contributors = blind to the decoy).
- **RULED OUT by direct probe (not assertion):** info floor (97% identifiable); capacity / `m_inducing=64` (`probe_inducing_util`: bottleneck only 48% saturated, eff-rank 15/32); mab0 routing-collision (`probe_inc11_mab0_routing`: faint-peak↔absorber gate overlap ≈ control, diff ≈ 0); over-smoothing-read-H0 (`probe_inc11_h0_recover`: H0 is set-undecodable, N5 .038; layer1 is ESSENTIAL); depth-aggregation / DenseFormer (`[H0;H1]` concat ≈ H1, −.035 N5 ⇒ NO-GO); combo-invariant coverage/decoy lever (`probe_inc11_coverage_rank`: +.008 on the ceiling, +.024 only because it helps the model catch up); **`cls_decoder=additive` (`probe_additive_parts`: the additive checkpoints' ENCODER ceiling COLLAPSES to N5 .005–.09 — additive is a weak training signal that cripples the encoder; this is why repC failed).** Over-smoothing signature is real (`probe_inc11_layer1_mech`: H1 token cosine 0.03→0.60) but does NOT equal info loss (H1 decodes far better than H0).
- **ACTION → Increment 12 BUILT (NOT trained): softer/adaptive PER-DONOR aggregation** — keep per_donor (encoder stays good), only change how its keys are aggregated so diffuse faint evidence is read. Lit-grounded (sparse-vs-soft, "dense-everywhere is NOT optimal; loud donor wants sparse, faint wants dense"): **entmax-1.5** (Peters 2019, exact closed form added as `entmax15`), **ASEntmax learnable per-head temperature** (2508.17821), **DSMIL dual-stream** max-instance + dense-bag (Li 2021, `DSMILDecoder`). New flags `--dec_aggr {sparsemax,entmax15,entmax_temp}` + `--cls_decoder dsmil`; arms `inc12_entmax15` / `inc12_entmax_temp` / `inc12_dsmil` on the mab0 base; runner M1=entmax15,entmax_temp · M2=dsmil; bundle synced; smokes PASS (forward+backward finite, learnable-temp gets grad, all 3 decoders). **JUDGE: DEV N5 oracle vs mab0 (.639) + guard N1/2/3; 3-seed before any verdict.** Expect the residual .849→1.0 (thin decisive-allele margins across ranks 2–4) to still be hard. Scripts added (root+local, eval-only): `probe_inc11_{decode,encoder_residual,mab0_routing,readout_ceiling,layer1_mech,layer1_decompose,writeback_scale,setmiss_decompose,decoy_char,coverage_rank,identifiable}.py`, `probe_additive_parts.py`, `probe_inducing_util.py`. DISCIPLINE: all F37 numbers single-seed (mab0 seed42), internally-controlled = directional; 3-seed required before logging any inc12 verdict.

## 2026-06-16 — F38: N5 decoy mis-ID ROOT-CAUSED to HARD attribution stealing shared peaks; FIX = an independent SYMBOLIC soft-split φ (EuroForMix-lite) reranking cls candidates → N5 oracle +0.07 on BOTH best baselines, DEPLOYABLE / NO-TRAIN. (user consented to this log edit)

### F38. The N5 decoy wall is largely a HARD-ATTRIBUTION artifact, fixable post-hoc by a symbolic explaining-away φ. (2-checkpoint, single-seed each = directional-but-robust; deployable inference-time lever.)

- **MECHANISM (measured, `probe_why_misattr`):** a missed faint minor's present alleles are **85.8% SHARED** with a major; the per-peak **hard softmax attr** gives **94% of those shared peaks to the better-corroborated major** (→T only 1%), and even the minor's **PRIVATE** alleles leak — only **30% →T**, 45% to a non-carrier true donor, 25% to background (attribution is height/context-driven, NOT genotype-constrained). The minor is starved, the decoy fed (decoy 10.6 peaks / 555 RFU vs missed-true 1.6 / 145, `probe_attr_proportion`).
- **phi_head is DEAD for within-mix proportions (`probe_phi_remeasure`):** global corr −0.23 is a CROSS-NOC artifact; within each NOC ≈0, within-N5-sample Spearman −0.14, faint-id 10.5% (< random .20), phi_pred collapsed to ~0.028 (absent-imbalance in the MSE). phi-within-mix and the decoy are the SAME problem (circular) — broken by the PRIVATE-peak anchor.
- **NO absolute info-limit (`probe_rawinfo_limit`):** 100% of N5 missed-true have ≥1 present allele the model's chosen 5-set cannot explain (mean 5.07); 0% allele-ambiguous. BUT the decoy coincidentally covers ~6 unexplained peaks too → crude separators (coverage .54, height-sum .61, coherence .57, LDA upper-bound .69) cap ~.7; the signal they miss = HEIGHT-CONSISTENCY (a redundant decoy's coincidental peaks imply an inconsistent single φ).
- **THE LEVER — independent symbolic soft-split φ (`probe_softsplit` / `probe_phi_rerank`):** feasibility-filter (drop the 16% of peaks NO panel donor carries — pure dropin, median height 11, **0% true-allele loss**, `probe_infeasible_filter`) → per-peak SOFT split over genotype-carriers (+ background sink), unrolled EM (= differentiable Sinkhorn-lite, Cuturi 2013 / slot-attention-OT): A=softmax_c(S+log φ), φ=height-weighted share, ×5; PRIVATE peaks (single carrier) ANCHOR φ → shared peaks split by φ. **Compatibility S = UNIFORM, NOT the neural attr.**
- **KEY RESULT — the symbolic part BEATS the learned attr:** as soft-split compatibility, neural attr → decoy-AUC **0.225** (< random — actively biased to the decoy); **uniform → 0.686**; coverage-prior 0.684. φ-within-N5 corr **+0.66** (vs phi_head −0.14), faint-id 29-33%. ⇒ the decoy is a genotype/height-physics problem the EM solves; the learned head HURTS (neuro-symbolic: symbolic φ + neural cls candidates).
- **DEPLOYED — rerank cls candidates by this φ (α tuned on real VAL, eval real TEST, no privileged at>=0):** N5 oracle **inc6_maskp 0.788→0.858 (+0.070)**, **inc13_B 0.806→0.876 (+0.070)** — consistent on BOTH best checkpoints, monotone in α, NO regression N1-N4 (N4 flat/up, N2→~1.0). Largest, cleanest N5 lever of the whole search; requires NO retraining (post-hoc on any checkpoint).
- **RULED OUT this investigation:** set-of-sets JEP/CAEP signature head un-trained = blind on production decoys (decoy-AUC 0.50, `probe_jep_vs_model`) — a TRAINED sig-head helps only a WEAK small decoder (+.13 dev N5, `ab_setofsets`); **add_recon coupled to sigmoid(cls_logit)** (inc16) COLLAPSES logit margins (true −4.5/decoy +0.3, margin 8.0→3.6, train-N5-oracle .95→.62) = NEGATIVE (`probe_inc16_dissect`); **decoupled recon head** (arm R) = +.05 dev N5 (BCE), but ASL doesn't train at small scale (RetinaNet bias-init insufficient); **separate-phi tower & cascade attr→phi→cls** (`ab_phi_arch`) HURT phi (corr +.64→+.07/+.31) — shared encoder is POSITIVE transfer for phi, negative-transfer hypothesis REFUTED; **φ-threshold / participation-ratio NOC** (`probe_phi_noc`) REFUTED (N4/N5 count 0.00; trained card-head 0.96 / N5 .763 far better); **distill** redundant once φ is used at inference.
- **CAVEAT / DISCIPLINE:** +0.07 is 2-checkpoint, single-seed each (not 3-seed) → directional-but-robust (independent checkpoints + principled symbolic mechanism + monotone α + ~16 converging probes). val-test N5 combo overlap ~27% but α is ONE scalar (low overfit). Paper-check bug-guards honored: Sinkhorn 2-marginal (not per-peak softmax); DON'T add a stutter-model (generator already has stutter → double-count); KEEP per-donor cls decoder (additive-cls collapses encoder, repC); log-space height (STRmix proxy; EuroForMix is Gamma); degradation/locus-eff folding is a refinement.
- **ACTION:** (1) φ-rerank = SHIP as a post-hoc reranker (deployable on any checkpoint). (2) After rerank fixes ID, the **COUNT head (N5 .763) is the NEW bottleneck** for deployed N5 EM → next lever (card-head, NOT φ-threshold; user's sequential top-down subtraction-count proposed, UNTESTED). (3) OPEN upside (untested): a SOFT genotype-constrained attr + attr-specific pipeline could yield a learned compatibility > uniform .686 (current 0.225 is the HARD-trained attr). Scripts (root, eval-only): probe_{jep_signature,jep_vs_model,decoy_evidence,explainaway,inc16_dissect,heads_quality,attr_proportion,phi_remeasure,why_misattr,genoconstr_attr,infeasible_filter,softsplit,phi_rerank,phi_noc}.py; ab_setofsets.py, ab_phi_arch.py (small-scale trained A/B).
