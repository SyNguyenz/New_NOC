# Design Doc — Increment 2: Relationship-Centric Training (express + supervise + curriculum + denoise)

**Status:** proposal (2026-05-31). Build AFTER Increment 1 settles (token/readout/encoder/selection),
so we add this on a proven base and don't confound many variables at once.
**Principle (user):** attention's strength is learning RELATIONSHIPS → (1) make the data EXPRESS the needed
relationships across all complexity levels, (2) SUPERVISE them directly (in-silico has ground truth), (3)
extract ALL per-peak information, (4) SUPPRESS attention to spurious/unnecessary relationships (noise/shortcut).
Everything below is paper-grounded.

## 1. Complete relationship catalog (the "enough relationships" requirement)
The validated PG models (EuroForMix Bleka 2016; STRmix; PG review PMC8535381) enumerate the COMPLETE set of
physical STR-mixture relationships → our target set:
- peak height (abundance) + height variation (CV)
- **mixture proportion Mx** (each contributor's fraction)
- **degradation** (height decays with fragment SIZE/bp; slope β)
- **stutter: back n−1 AND forward n+1** (proportion ξ)
- **allele drop-out** (minor below threshold) ; **drop-in** (sporadic spurious peaks)
- **Fst/θ + allele frequencies** (population)
- per-locus amplification efficiency
- **allele→donor genotype mapping** (which alleles belong to which contributor)
Current token covers only Hb + back-stutter. ADD: forward stutter, degradation (needs size bp), drop-in, Mx, locus-efficiency.

## 2. Token-field RELATIONSHIP MAP (explicit "relationships between fields" + "feature for what")
Each physical relationship = a function of specific token fields. Make these explicit so we know what each
feature is FOR and which field-interactions encode which relationship:

| relationship | token fields involved | physical meaning |
|---|---|---|
| Hb (het balance) | height_peak ÷ Σheight_at_locus | balanced pair = 1 donor; imbalance ⇒ ≥2 |
| back-stutter | height_peak ÷ height(allele−1) | distinguish true minor vs n−1 artifact |
| forward-stutter | height_peak ÷ height(allele+1) | n+1 artifact (ADD — missing) |
| degradation | height_peak vs SIZE(bp), across loci | larger fragment lower ⇒ degraded/old DNA (ADD size) |
| global tier / Mx proxy | height_peak ÷ max(height in profile) | which contributor "tier" a peak belongs to |
| count (NOC) | n_alleles_per_locus (by RFU thr), MAC | ⌈MAC/2⌉ lower bound; saturates ⇒ Hb/height needed |
| allele→donor | (allele, locus) matched to donor reference genotype | THE ID relationship; attention must group a donor's peaks |
| contributor grouping | a donor's peaks share consistent Mx/global-tier ACROSS loci | cross-locus consistency = same person (deconvolution) |

## 3. Where to query / query design (readout)
- Donor queries cross-attend the peak set (Query2Label/ML-Decoder). "Where to query" = each of 45 donor
  queries → the encoded peak set; per-donor logit. (Increment 1 already builds this = per_donor decoder.)
- **Query INITIALIZATION (privileged)**: initialize each donor query from that donor's REFERENCE genotype
  embedding (we have single-source profiles) so the query "knows" its donor's expected alleles → stronger,
  generalizes to the closed-set. (Query2Label uses learnable label embeddings; we enrich init with reference.)

## 4. Data program — express relationships at ALL complexity levels (the "enough tiers" requirement)
**Domain Randomization** (Tobin 2017; review PMC9038844) + **Curriculum** (Bengio 2009) + auto-curriculum.
We GENERATE in-silico → control every relationship parameter. For EACH relationship, span the parameter
clear→exaggerated(even-unrealistic)→subtle so the model learns INVARIANT structure and real cases fall inside:
- Mx: 1:1 (balanced, height-irrelevant) → 1:1000 (minor barely shows)
- allele-sharing: none → maximal (all alleles shared)
- dropout: none → severe ; drop-in: none → frequent
- stutter (back+fwd): 0 → exaggerated
- degradation slope: flat → steep
- **NOC4-vs-5 CONTRASTIVE pairs**: same 4 donors ± a 5th whose Mx sweeps large→tiny → directly teaches the
  4-vs-5 distinguishing (height-stacking) signal across its full strength range. (Generalized to ALL NOC
  adjacencies in §4b — this is the single hardest boundary, not the only one.)
Curriculum = schedule the domain-parameter distribution from clear→subtle over training.

## 4b. Discriminative-relationship learning: scope, SELECTION criterion, and ID-protection valves
§4's NOC4-vs-5 contrastive pairs are a SINGLE boundary; the count signal is ORDINAL (1<2<3<4<5) and the
unlearned differences may sit at ANY adjacency (measure, don't assume it's only 4-5). But more contrast /
more relationships is NOT free — a count-centric objective can collapse donor identity and hurt ID
(negative transfer). So three coupled moves: (A) BROADEN the contrast across the whole NOC axis;
(B) SELECT which relationships to learn by an explicit criterion (not greedy); (C) PROTECT ID with valves.

**(A) Ordinal contrast across ALL NOC (broaden), not one 4-vs-5 boundary.**
- NOC is ordinal → **Rank-N-Contrast** (Zha 2023): contrast every sample PAIR weighted by label distance
  |NOCᵢ−NOCⱼ|, ordering the representation along the full 1→5 axis (a NOC1–NOC5 pair pushed farther than
  NOC4–NOC5). §4's 4-vs-5 pairs = the special case at the hardest adjacency. (SupCon Khosla 2020 is the
  unordered variant; RNC wins here precisely because it respects the count ORDER.)
- NOC head: CORAL/CORN-style ordinal output consistent with the ordered axis.

**(B) Selection criterion — the "anchor" (anti-greedy; do NOT learn every relationship/contrast).**
deepNoC's 89 feats/peak is the greedy extreme; we SELECT instead:
- **Which relationships to feed** → **mRMR** (Peng 2005): keep max-Relevance MI(relationship; NOC/ID)
  − min-Redundancy MI(relationship; already-selected). Rank the §2 catalog (Hb, back/forward stutter, Mx,
  degradation, drop-in, locus-eff); keep the compact relevant-non-redundant subset.
- **When to stop** → **Information Bottleneck / Deep VIB** (Tishby 1999; Alemi 2017): a good representation
  is MINIMAL SUFFICIENT — keep info predictive of ID/NOC, "forget" the rest; over-complete (learn-everything)
  representations generalize worse / less robust. Stop adding relationships when marginal dev info-gain plateaus.
- **Which PAIRS to contrast** → **confusion-driven hard mining** (Robinson 2021): anchor = the empirical NOC
  CONFUSION MATRIX on in-silico dev; emphasize the confused adjacent pairs (high off-diagonal mass), NOT the
  already-separable ones (e.g. NOC1 vs NOC5). Margin ∝ ordinal distance.

**(C) ID-protection valves (binding constraint: this must NOT degrade donor-ID).**
NOC here is read DOWNSTREAM of ID (CardinalityHead reads the sorted donor-prob profile) ⇒ NOC ranking is
BOUNDED by ID ranking, and the signal distinguishing 4-vs-5 IS the faint-extra-contributor (ID) signal
(76% of NOC5 misses are rankable, not dropout). So learn it AS an ID signal, with valves:
1. **Express via abundance/attribution, NOT the count label.** Supervise the NOC difference through per-donor
   abundance φ + allele→donor attribution (§5); NOC = |φ>τ| emerges. AVOID a count-label contrastive that
   clusters by "how many" and becomes donor-INVARIANT — that is the negative-transfer failure mode.
2. **Separate projection head** (SimCLR, Chen 2020): put the ordinal-contrastive on a projection g(z) discarded
   at inference; the ID head reads z / its per_donor decoder. Representation-BEFORE-projection is empirically
   better downstream → contrastive geometry does not sit on the ID readout space.
3. **Decouple pooling** — PRECEDENT IN THIS PROJECT: the reject head needed its own PMA pool (`pma_reject`)
   because the ID gradient pulled z off-task (reject AUROC 0.993→0.945 when shared). Give the NOC-contrastive
   its own pool/projection likewise.
4. **Gradient/loss balancing**: **PCGrad** (Yu 2020) projects away conflicting (cosine-negative) gradients;
   **Kendall uncertainty** (§5) auto-weights the tasks.
5. **No-regression guard**: any change that drops per-donor ID EM on NOC1/2/3 beyond a small threshold
   (e.g. >1 pp) is a FAIL even if NOC4/5 improves.

**Verification (per §11 — ablate, don't assume).** Measure per-donor ID EM at all 5 NOC WITH vs WITHOUT the
discriminative-contrastive branch. ID stable/up ⇒ donor-level signal (keep). ID down ⇒ count-clustering or
gradient conflict ⇒ decouple projection / enable PCGrad / down-weight.

## 5. Auxiliary / PRIVILEGED supervision (the "supervise/teach the relationships" requirement)
in-silico is synthesized from known single-source → GROUND TRUTH for free (real test doesn't have/need it):
**Learning Using Privileged Information** (Vapnik) / **generalized distillation** (Lopez-Paz 2015, 1511.03643).
Auxiliary heads (teach attention the relationships directly, not hope they emerge):
- **allele→donor attribution** (per-peak source donor) — supervises the cross-attention itself.
- **mixture proportion φ** (regression) — the unmixing abundances.
- **per-peak stutter/artefact** classification (like deepNoC's peak classifier).
- main: donor-ID (ASL) + NOC.
**Multi-task loss weighting** = Kendall homoscedastic uncertainty (1705.07115) to auto-balance these losses
(hand-tuning weights is infeasible); caveat: prone to overfit/bad-init — monitor (GradNorm as alternative).

## 6. Structural attention bias (the "relationships between fields/positions" at attention level)
Graphormer-style ADDITIVE attention bias by known structure (relative position encoding for non-sequential
structure): bias attention by **same-locus** (peaks at the same locus interact strongly) and **allele-distance**
(n±1 ⇒ stutter relation). Builds the physical field/position relationships into attention as inductive bias,
helping it focus on real relationships. Refs: Graphormer / relative position encoding / structure-aware transformer.

## 7. Suppress UNNECESSARY relationships / noise (the "remove attention to spurious, avoid noise" requirement)
- **Sparse attention** — sparsemax / α-entmax (Martins 1705.07704): assign EXACTLY ZERO to irrelevant peaks
  → donor query attends only its donor's alleles, zeroing noise/other-donor peaks → focus + interpretable.
- **Invariant Risk Minimization** (Arjovsky 1907.02893) / REx: treat combo/NOC/ratio as ENVIRONMENTS, learn
  features INVARIANT across them → suppress the SPURIOUS combo-co-occurrence shortcut (our memorization failure
  = textbook shortcut learning, survey 2402.12715), keep the CAUSAL allele→donor relationship.
- The allele→donor auxiliary supervision (Sec 5) itself FORCES attention onto the causal relationship.

## 8. Information completeness (the "extract all info" requirement)
- Per-peak SUFFICIENT statistics for the PG likelihood = {allele, height, SIZE(bp)}. **CORRECTION (verified):**
  size(bp) is NOT missing and needs NO raw re-prep — the GeneMapper CSVs already export (Allele, **Size**,
  Height) triplets; prepare_data_set.py simply dropped the Size column. `extract_size.py` recovers it as a
  sidecar aligned to the existing tokens (size_{split}.npy), **100% peak coverage**, bp∈[76,443]. So we are
  info-complete at peak level NOW; derived relations (Hb/SR/degradation/Mx) are functions of it. (Dye column
  also dropped — low value, locus already implies the dye-lane.)
- HONEST CEILING ("Information gain from peak height", Green/Mortera 2018, PubMed 29990823): height info is
  SUBSTANTIAL for single-replicate UNBALANCED mixtures (our hard NOC4/5 minor cases) but ~IRRELEVANT for balanced
  ones, and ABSENT under dropout. So extract height-relations fully (they help the cases we care about) but
  expect a ceiling where information physically isn't present.

## 9. Emergence — tempered expectation (the "unexpected capabilities like LLM" question)
Wei 2022 (emergent abilities) vs **Schaeffer 2023 "Mirage" (2304.15004)**: apparent emergence is largely a
METRIC ARTIFACT — nonlinear/discontinuous metrics (Accuracy, **Exact-Match** = our headline metric, 92% of
"emergent" cases) turn SMOOTH underlying gains into apparent jumps. ⇒ Do NOT expect magic emergence at our scale
(1.4M params, structured 45-class task); expect SMOOTH capability gains from richer data/supervision. If EM
"jumps", suspect the metric (re-validates graded-recall selection). This program maximizes the smooth gains;
no over-claim.

## 10. Verification — did attention learn the RIGHT relationships?
- Attention-map inspection: donor query d should attend to donor-d's alleles (allele→donor), not spurious.
- Linear probing of encoder features for the supervised relationships.
- Dropout-vs-rankable split (already have) on NOC5; in-silico-dev vs real-test gap.

## 11. Risks & sequencing
- **Domain gap** (recurring): supervised relationships must be PHYSICAL (transfer); DR (wide span) helps; validate
  on real test. Synth-specific artifacts (e.g., wrong stutter/φ model) could teach non-transferable patterns.
- **Over-engineering / confounding**: this is many changes → ADD INCREMENTALLY on top of Increment 1's settled
  base; ablate each (relationship-supervision, DR-curriculum, sparse-attn, IRM, struct-bias, size) so we know
  what helps. Don't change everything at once.
- **Sequencing**: Increment 1 (token/readout/encoder/selection) → then 2a size+forward-stutter+drop-in features
  (§4b-B mRMR-selected, not all) → 2b auxiliary privileged supervision (+Kendall) → 2c DR-curriculum + §4b-A
  ordinal contrast (Rank-N-Contrast, confusion-driven pairs) on a §4b-C decoupled projection → 2d
  sparse-attn/IRM/struct-bias. Each measured on in-silico dev + real test, honest CIs, with the §4b-C
  no-regression guard on per-donor ID EM at every step.
  → **FINAL (only after the architecture is FROZEN): §13 generator-realism calibration to real conditions**
  — a DATA-side lever to close the in-silico→real gap; done LAST so it never confounds the model ablations.

## 12. Evaluating φ (Mx) + condition-stratified gap (the real data HAS φ — naming convention)
The PROVEDIt sample NAME encodes the mixture RATIO (→ nominal φ) and the DNA CONDITION
(untreated/UV/degraded/sonicated/inhibited + level) — both recoverable with NO raw re-prep
(`extract_phi_condition.py`; validated φ-support == y on all splits). So Increment 2's φ head can be
EVALUATED on real (not only supervised on in-silico), and the in-silico→real gap stratified physically.
- **φ / Mx estimation is a recognised sub-task**, not ad-hoc: it is feature (iii) of the Zhu 2026
  blueprint we adopted; continuous PG engines **EuroForMix** (Bleka 2016) and **STRmix** (Bright 2016)
  estimate Mx by MLE and validate estimated-vs-known Mx. In the unmixing framing (§2 of the
  representation doc) this is **abundance estimation** → metric = abundance RMSE/MAE + SAD (HSU lit.).
- **Nominal vs realized caveat**: the name ratio is the TEMPLATE mixing ratio, not the realized per-locus
  proportion (perturbed by degradation + per-locus amplification efficiency — STRmix degradation model).
  ⇒ evaluate with a SCALE-ROBUST metric: **Spearman rank-corr(φ̂, ratio)** + major/minor ordering, and
  report MAE as secondary, condition-conditioned.
- **Condition-stratified EM is the dataset's intended use**: PROVEDIt (Alfonse 2018) was built across
  144 conditions precisely to study interpretation under degradation/inhibition. Stratify EM per
  (condition × NOC) with honest CIs. Green & Mortera (§8) predicts the degraded/inhibited strata lose
  height information ⇒ separates "model failure" from "information physically absent" (NB: the old
  "24%-genuine-dropout NOC5" was a synth-CSV artifact — raw genotypes give ~100% rankable; findings-log F6).
  Dropout rises with falling peak-height/template + degradation
  (Tvedebrink 2009/2012; ISFG DNA-commission Gill 2015) ⇒ `condition` + `template_ng` + `Q` are the
  theory-grounded stratifiers (template_ng/Q already extracted via extract_metadata.py).

## 13. Generator-realism calibration to real conditions (FINAL stage — only after the model is FROZEN)
**Status: deferred — do LAST.** This is a DATA-side lever (improve `make_insilico.py` realism), not an
architecture change. Run it only after Increments 1–2 (2a–2d) are done and the architecture/decoder are
FROZEN. Rationale: changing the training distribution confounds every model ablation (§11 "don't change
everything at once"); and closing the in-silico→real GAP only matters once the model is already strong on
in-silico — otherwise you calibrate toward a weak target.

What it uses (now available): the recovered real **DNA condition** (untreated/DNase/Fragmentase/
sonication/UV/humic + level), **template_ng**, **Q**, **nominal φ** (`extract_phi_condition.py`), and now
real per-peak **SIZE(bp)** (`extract_size.py`, 100% coverage — NOT a re-prep, the CSVs already had it) give
the EMPIRICAL real degradation slope (height vs size) per condition. The current generator only models gamma
peak-height jitter (GAMMA_SHAPE) + hard threshold dropout (AT) + ratio skew — it MISSES the degradation
relationship (§1): height decay vs fragment SIZE(bp), UV/sonication damage, inhibition. Since size is now in
hand, the degradation model can be FIT to the real height-vs-size decay per condition (no re-prep blocker).

Plan (calibration, not blind randomization):
- Add a **degradation model** to `make_insilico.py`: per-locus height decay with fragment size (slope β),
  parameterised by condition; **condition-dependent dropout** as a function of (height, template, degradation)
  rather than a single AT threshold.
- **Calibrate** β / dropout / jitter parameters so the SYNTHETIC per-condition distribution matches the REAL
  per-condition distribution (use `synth/validate_realism.py` + the §12 condition strata as the target);
  sample the in-silico condition mix to match real prevalence (or span it, DR-style, for robustness).
- **Measure**: the in-silico-dev↔real-test gap (§7) BEFORE vs AFTER calibration, per condition × NOC. Success
  = gap shrinks WITHOUT regressing in-silico-dev (the §4b-C no-regression guard still applies).
- **Risk (explicit, §11)**: a wrong degradation/φ model teaches NON-transferable synth artifacts. Calibrate
  to measured real distributions; keep DR span wide; validate on real test only, once.
- Grounding: STRmix degradation model + Tvedebrink 2012 degradation-adjusted dropout (height/template-driven,
  §12 refs); Domain Randomization (Tobin 2017) for the span-wide variant; PROVEDIt conditions (Alfonse 2018)
  as the calibration target; Green & Mortera ceiling (§8) bounds what is recoverable under dropout.

## References (added this increment)
- Domain randomization: Tobin 2017; review "Robot Learning From Randomized Simulations" PMC9038844.
- Curriculum learning: Bengio et al. 2009.
- Privileged info / generalized distillation: Vapnik & Izmailov; Lopez-Paz et al. 2015 (arXiv 1511.03643).
- Multi-task loss weighting: Kendall, Gal, Cipolla 2018 (arXiv 1705.07115); GradNorm.
- Structural attention bias: Graphormer (Ying 2021); relative position encoding; structure-aware transformer (2202.03036).
- Sparse attention: Martins & Astudillo (sparsemax); Niculae & Blondel "Regularized Framework for Sparse/Structured Attention" 1705.07704; α-entmax.
- Shortcut/spurious: Geirhos 2020 shortcut learning; survey 2402.12715; IRM Arjovsky 2019 (1907.02893); REx.
- Information gain from peak height: Green & Mortera 2018 (PubMed 29990823); RSS-C 2021 (rssc.12498).
- **(§12) PROVEDIt dataset / condition-stratified eval**: Alfonse, Garrett, Lun et al. 2018 — "A large-scale
  dataset of single and mixed-source STR profiles… PROVEDIt", FSI:Genetics (PubMed 29091906; 25k profiles,
  144 conditions). Mx estimation/validation: Bleka, Storvik, Gill 2016 — EuroForMix, FSI:Genetics; Bright et
  al. 2016 — STRmix developmental validation, FSI:Genetics.
- **(§12) Allelic drop-out vs peak-height/template**: Tvedebrink et al. 2009 (PubMed 19647706) + degradation
  extension 2012; ISFG DNA-commission recommendations on drop-out/drop-in, Gill et al. 2015 (PMC4689582).
- Emergence: Wei 2022 (2206.07682); Schaeffer 2023 "Mirage" (2304.15004).
- **(§4b) Ordinal contrast**: Zha, Cao, Son, Yang, Katabi 2023 — Rank-N-Contrast, NeurIPS 2023 Spotlight
  (arXiv 2210.01189); Khosla et al. 2020 — Supervised Contrastive Learning (arXiv 2004.11362, unordered
  variant); Cao, Mirjalili, Raschka 2020 — CORAL rank-consistent ordinal regression.
- **(§4b) Relationship selection / stopping**: Peng, Long, Ding 2005 — mRMR, IEEE TPAMI 27(8):1226–1238;
  Tishby, Pereira, Bialek 1999 — Information Bottleneck Method; Alemi, Fischer, Dillon, Murphy 2017 — Deep
  Variational Information Bottleneck, ICLR (arXiv 1612.00410).
- **(§4b) Pair selection / hard mining**: Robinson, Chuang, Sra, Jegelka 2021 — Contrastive Learning with
  Hard Negative Samples, ICLR (arXiv 2010.04592).
- **(§4b) ID-protection valves**: Chen, Kornblith, Norouzi, Hinton 2020 — SimCLR, ICML (arXiv 2002.05709;
  projection head protects the representation); Yu, Kumar, Gupta, Levine, Hausman, Finn 2020 — Gradient
  Surgery / PCGrad, NeurIPS (arXiv 2001.06782; conflicting-gradient negative transfer).
- (carried from Increment 1: FT-Transformer/Gorishniy embeddings; Query2Label/ML-Decoder readout; Set
  Transformer++/SetNorm encoder; ASL; Cortes/Mohri cardinality; SAINT; DeepSetNet set prediction; deepNoC; Bright/Taylor PG.)
