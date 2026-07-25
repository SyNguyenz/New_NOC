# Design — Stacking-Explanation Mechanism (learn how alleles stack, with its own loss)

**Status:** DESIGN (not yet implemented). Capture-context spec. **Date:** 2026-07-12.
**Scope:** A dedicated, first-class mechanism + loss that teaches the model **how alleles STACK** at
shared positions — which donors contribute to each peak, how many, and how the summed height
decomposes — so the model can *explain* each observed peak as a sum of contributions. Companion to
`design_4head_decomposition.md` (this is the explicit, first-class form of that doc's §3.5 coupling).

> Epistemic note: internal numbers (`em_*`, F-findings) are our own single-seed runs, not externally
> verified. Design reasoning rests on (a) the generator's ground-truth decomposition (verified in
> code) and (b) external published methods (refs §8).

---

## 1. Motivation — shared/stacked alleles are the crux, and we CAN teach them

A peak at a **shared allele** has height = SUM of the contributions of every present donor carrying it.
This single fact drives three failures at once:
- **attr ambiguity** — which donor(s) does this peak belong to? (softmax attr can only pick one)
- **decoy false-positive** — a wrong donor whose alleles are all "covered" by the true donors' shared
  alleles looks present → `cls` false positive.
- **height mis-reading** — the summed height at a shared allele is NOT any single donor's height.

At *inference* a stacked peak is 1 number = k unknowns → locally unsolvable (needs cross-locus φ). BUT
at *training* the in-silico generator produces the **full ground-truth decomposition**: `contrib`
(k × N_FLAT) = each donor's contribution to each bin (`make_insilico.py:279-284`,
`gen_mixture_peak_labeled:314-320`). So the true split IS available as a **privileged label** — we can
*teach* the model what the split should be, and at inference it applies the learned cross-locus pattern.
This is what makes a dedicated stacking mechanism + loss well-posed.

---

## 2. The mechanism — a Stacking-Explanation head (three tiers)

### 2.1 "Which donors stack" — multi-label attribution
Per peak, predict the SET of contributing donors (multi-label), NOT a single argmax. Reuse
`ml_attr_head` (`set_transformer.py:1478`, multi-label sigmoid, already coded, off by default) instead
of the softmax `attr_head`. Ground truth: `(contrib > 0)` per bin. Loss: **ASL / BCE** (bounded)
[Ridnik ASL 2021]. This is the operator fix for softmax's forced-1-donor/peak defect.

### 2.2 "How many stack" — stacking-depth ordinal (NEW target)
Per peak, predict the number of contributing donors m ∈ {1..5}. Ground truth:
`m = (contrib > 0).sum(0)` per bin (currently NOT supervised). Loss: **CORN/CORAL ordinal** (bounded,
rank-consistent) [Cao CORAL 2020; Shi CORN 2023]. Rationale: knowing a peak is a 3-stack lets the model
read height correctly (a 3-stack peak's height ≈ 3× a single) — the SUM signal a multiset counter needs
[Deep Sets; GIN].

### 2.3 "Explain the height" — reconstruction / analysis-by-synthesis loss (the core)
The predicted decomposition must **re-sum to the observed height**:
`L_recon = ‖ Σ_c Â_c · φ̂_c · dosage_c · scale − h_obs ‖` (on `log1p` heights, per present peak). This
is the "let the model EXPLAIN each peak" objective — self-supervised, **bounded by the observed data**
(no variance-collapse mode) [Slot-Attention reconstruction; NMF]. It is per-sample → **NOT blocked by
the locus-independent generator**.

---

## 3. Privileged supervision is REQUIRED — and the prior LUPI collapse was a FIXABLE bug, not a barrier

Privileged in-silico physics labels are LUPI [Vapnik-Izmailov 2015; Lopez-Paz 2016], and — importantly
— **required**: reconstruction alone provably does NOT yield a semantic decomposition without an
inductive-bias/supervision signal [Locatello et al. ICML 2019, best paper]. So the `contrib` split
supervision is not optional; recon-alone is insufficient.

**Root cause of the prior LUPI collapse (code audit, `train_set_transformer.py:1462-1478`) — CORRECTED.**
The `results/lupi_v*` collapse (loss → −22, monotone v2→v5) was **NOT** the β-NLL variance collapse:
the inner β-NLL (line 1467-1469) was correctly guarded with the stop-grad weight per [Seitzer 2022].
The real culprit is the **OUTER Kendall wrapper** `exp(−log_var)·loss + log_var` applied to privileged
losses that are **fittable-to-≈0 on synthetic** (`degr` = predict one scalar β; `mu` = denoise clean
height). As loss→0 the learnable `log_var`→−∞ (no floor), so the weight `exp(−log_var)`→∞ → the
privileged loss gets **infinite weight on the shared encoder → hijacks it**; the term → `1+log(loss)`
→ −∞ (the −22). Baseline attr-CE/phi-L1 survived only because their targets are HARD (can't reach 0).
→ **Kendall is the wrong tool for collapsible privileged losses.** (This is our own analysis — high
confidence as code+math, but confirm by re-running one arm with `log_var` floored.)

**Safe design (mandatory):**
1. **Bounded losses ONLY** — reconstruction (bounded by height), multi-label ASL/BCE, ordinal CORN.
   Never an unbounded NLL.
2. **Presence-gated** — compute only over real (non-pad) peaks; never regress a sparse 45-vec target
   (the phi-head L1-collapse — a *math* property of L1 on an ~88%-zero target).
3. **Fixed / FLOORED weighting, NOT naive Kendall** — floor `log_var ≥ log(0.1)`, or use a fixed small
   weight, or **detach** the privileged head from the encoder. Gradient policy per
   `design_4head_decomposition.md` §4, but note: **do NOT Kendall-weight a collapsible privileged loss.**

The reconstruction loss is the **safe core** — self-supervised, data-bounded, no collapse mode — but it
is a *regularizer on top of* the supervised split, not a replacement (Locatello 2019).

---

## 4. Data step (required)

The generator currently saves only `attr = argmax` (dominant donor per peak) and `mu = Σ` (clean total
height) — **not** the per-donor split needed to supervise stacking. Add a cheap regen step: from the
existing `contrib` array, save per peak:
- `stack_depth` = `(contrib > 0).sum(0)` (ordinal target, §2.2)
- `stack_multilabel` = `(contrib > 0)` mapped to donor columns (multi-label target, §2.1)
- (optional) `contrib_frac` = `contrib / contrib.sum(0)` per bin (fractional split for a richer recon
  target)

This mirrors the existing `attr_bin` extraction (`make_insilico.py:296-298`); no new physics, just
persist what the generator already computes. Leak-safe (labels are per-instance latents, not donor
attributes).

---

## 4b. Training strategy — counterfactual nested-peeling (delta-consistency)

A training curriculum that teaches the stacking decomposition from *differences*, not just absolute
labels. Base-sample count is UNCHANGED (it restructures the same mixtures into nested families; cost is
extra forward passes per base, not more base data).

**Construction (exact, in-silico).** From a NOC5 mixture with per-donor `contrib` (`make_insilico.py:279-284`),
derive the nested family by REMOVING one donor at a time: `mix_4 = mix_5 − contrib[j]`, then NOC3/2/1
recursively. The removal is EXACT (subtract the donor's contribution), so the peak-level **delta between
NOC_k and NOC_{k−1} = exactly the removed donor's contribution** — a clean, privileged counterfactual
label with no extra measurement.

**Delta-consistency loss.** Supervise that the model's decomposition CHANGES by exactly the removed
donor: (i) the removed donor's slot/gate must turn OFF (present-set + count consistency:
`Σgate(mix_k) − Σgate(mix_{k−1}) ≈ 1`, and the dropped slot is the removed donor); (ii) the reconstructed
height delta must equal the removed `contrib` (`recon(mix_k) − recon(mix_{k−1}) ≈ contrib[j]`). Both are
**bounded / reconstruction-based** (LUPI-safe per §3) — a counterfactual form of the §2.3 explain-the-
height objective [Kaushik 2020 counterfactually-augmented data; Zhang 2018 Mixup structured aug].

**What it teaches (both cls AND noc):**
- **attr/cls** — donor signature as "which peaks drop when this donor is removed" (a counterfactual
  attribution signal, typically stronger than absolute per-peak labels).
- **noc (gate)** — count as *peel depth*: NOC_k vs NOC_{k−1} differ by exactly one slot → directly
  supervises `Σgate` calibration (§ 4-head doc §3.2), the count-consistency the RF crutch replaces.
- **additive/consistent decomposition** — enforces `f(mix_k) − f(mix_{k−1}) ≈ donor_removed`, i.e. the
  decomposition is linear/peelable → directly supports the inference-time **peeling** lever (internally
  the best N5 lever, feasibility-positive but 35%-buried; F30) [DeepSets additivity; NMF].

**Ceiling (honest).** Better training SIGNAL, not new information. When the removed donor is a **buried
minor** (near-far, peaks all shared), its delta is **tiny/noisy** → the counterfactual signal is weak
exactly at the hard N5 case. Bounded by the same identifiability floor (§5).

---

## 5. What it adds vs the ceiling (honest)

**Adds (currently missing):** explicit supervision of stacking depth + a reconstruction/"explain"
objective — the current model has soft OT attention but NO explicit "explain the observed height as a
sum" loss and NO stacking-depth target. Real gap, not redundant.

**May exceed the symbolic EM — the ceiling is the (intractable) Bayes-optimal decoder, NOT the EM.**
The true optimum = the Bayesian posterior over donor-sets + nuisances (φ_c, β_c, T) for the correct
model; it is OPTIMAL but **computationally intractable** (combinatorial C(45,5) × continuous). The
symbolic EM / EuroForMix are **suboptimal tractable approximations** (local-MLE, per-hypothesis scoring,
no global combinatorial search). A neural net is a **different tractable approximation — amortized
inference** — with no reason to be bounded by the EM's approximation quality; it can learn the search,
avoid EM's local optima, and marginalize nuisances from data. So "exceed the symbolic EM" is
theoretically valid (this is *why* a net is worth using at all — cf. the forensic field moving from
EuroForMix-MLE to ML methods, [[forensic-noc-estimation]]). What it CANNOT exceed is the Bayes-optimal
decoder = the information floor. On *synthetic* (model matched to the generator) the EM is near-optimal
so headroom over it is small; on *real* (EM mis-specified) the headroom is larger.

**Ceiling = shared-φ on the CURRENT generator.** The reconstruction loss has a null space (many splits
re-sum to the same height); privileged training breaks it *during training*, but at inference the only
cross-locus coupling in the data is **shared-φ** (exhausted — NNLS null, `exp_crosslocus` α→0). So on
current data the stacking module **reaches the EM ceiling (~0.83 N5), not beyond**. To let it EXCEED,
the generator must carry a **per-contributor cross-locus signature** (per-contributor degradation β_c /
efficiency) → see `design_generator_crosslocus.md`. The truly-ambiguous N5 peak (decoy fully covered +
coincidentally φ-consistent) stays at the information floor regardless.

---

## 6. Relationship to the other design docs

- `design_4head_decomposition.md` §3.5 — this doc is the **explicit, first-class** realization of that
  coupling (promoted from a loss term to a dedicated head + reconstruction objective).
- `design_generator_crosslocus.md` — the generator upgrade that lets this module exceed the shared-φ
  ceiling (provides the per-contributor cross-locus signal the reconstruction can exploit).
- `design_generalization_jtt.md` — the reconstruction/"explain" loss IS a form of always-on invariant
  (physics) supervision, so it also serves generalization.

---

## 7. Open decisions

- [ ] Recon target: raw RFU vs `log1p(h)`? needs per-locus `scale` — reuse `_em_deconv_phi`
      (`set_transformer.py:1819`) or the generator's `EFF_BIN`/`T`?
- [ ] Supervise `contrib_frac` (full fractional split) or only depth + multi-label? (fuller = richer
      but closer to LUPI regression risk — keep bounded/presence-gated).
- [ ] Depth head: per-peak ordinal, or derive depth from the multi-label head's active count?
- [ ] Is the reconstruction (self-supervised) loss alone enough, or is the privileged split needed?
      (self-sup is safest; add privileged only if recon underfits — measure first.)
- [ ] Gradient: run the cosine + frozen-H probe pre-flight (4-head §4) before deciding share/detach.
- [ ] §4b nested-peeling: how deep to peel per base (full 5→1 family vs a sampled subset, to bound the
      extra forward-pass cost)? and does the delta-consistency go on the gate (count), the recon
      (height), or both?

---

## 8. References

Confidence: ✓ verified citation.

**Stacking representation (multi-label / set)**
- ✓ Ridnik et al., *Asymmetric Loss for Multi-Label Classification* (ASL), ICCV 2021, arXiv:2009.14119
  (the bounded multi-label loss already used as `--loss asl`).
- ✓ Zaheer et al., *Deep Sets*, NeurIPS 2017, arXiv:1703.06114 (sum aggregation for stacking depth).
- ✓ Xu et al., *How Powerful are Graph Neural Networks?* (GIN), ICLR 2019, arXiv:1810.00826.

**Stacking-depth ordinal head**
- ✓ Cao, Mirjalili, Raschka, *Rank-consistent ordinal regression* (CORAL), Pattern Recognition Letters
  2020, arXiv:1901.07884.
- ✓ Shi, Cao, Raschka, *Deep NN for Rank-Consistent Ordinal Regression* (CORN), 2023, arXiv:2111.08851.

**Reconstruction / analysis-by-synthesis / unmixing**
- ✓ Locatello et al., *Object-Centric Learning with Slot Attention*, NeurIPS 2020, arXiv:2006.15055
  (reconstruction as the inductive bias for decomposition).
- ✓ Lee & Seung, *Learning the parts of objects by non-negative matrix factorization*, Nature 1999
  (linear unmixing: h ≈ Σ abundances · endmembers = the stacking model).

**Counterfactual nested-peeling training (§4b)**
- ✓ Kaushik, Hovy, Lipton, *Learning the Difference that Makes a Difference with Counterfactually-
  Augmented Data*, ICLR 2020, arXiv:1909.12434 (supervise on counterfactual differences).
- ✓ Zhang et al., *mixup: Empirical Risk Minimization*, ICLR 2018, arXiv:1710.09412 (structured
  compositional augmentation).
- Internal: F30 (peeling feasibility-positive, best N5 lever, 35%-buried) — the inference-time analog
  this training strategy supports (`reports/empirical_findings_log.md`).

**Privileged information + the LUPI-collapse warning**
- ✓ Vapnik & Izmailov, *Learning Using Privileged Information*, JMLR 2015.
- ✓ Lopez-Paz et al., *Unifying distillation and privileged information*, ICLR 2016, arXiv:1511.03643.
- ✓ Seitzer et al., *On the Pitfalls of Heteroscedastic Uncertainty Estimation* (β-NLL — why NOT
  unbounded NLL), ICLR 2022, arXiv:2203.09168.

**Multi-task gradient policy (LUPI-safety)**
- ✓ Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad), NeurIPS 2020, arXiv:2001.06782.
- ✓ Kendall, Gal, Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses*, CVPR 2018,
  arXiv:1705.07115.

**Forensic decomposition (the physics we explain)**
- ✓ Bleka, Storvik, Gill, *EuroForMix: an open source software...*, Forensic Sci. Int. Genetics 2016
  (continuous gamma model: per-contributor φ, degradation, stutter).
- ✓ Taylor, Bright, Buckleton, *The interpretation of single source and mixed DNA profiles* (STRmix
  continuous model), Forensic Sci. Int. Genetics 2013 (per-contributor mixture deconvolution).

---

## 9. Code anchors

- `make_insilico.py:279-284` — `contrib` (k × N_FLAT) per-donor per-bin ground-truth decomposition.
- `make_insilico.py:296-298` — `attr_bin = argmax(contrib)` (the argmax we currently save; extend to
  stack_depth + multi-label here).
- `make_insilico.py:314-320` — `gen_mixture_peak_labeled` (privileged labels; mu = Σ, not the split).
- `models/set_transformer.py:1462` — softmax `attr_head` (to replace); `:1478` — `ml_attr_head`.
- `models/set_transformer.py:958-995` — Sinkhorn-OT assignment `A` + `slot_mass` (soft attr/phi bones).
- `models/set_transformer.py:1819` — `_em_deconv_phi` (symbolic EM; recon scale reference).
- `train_set_transformer.py:211` — `pcgrad_backward`; `:1447` — the phi L1-collapse to avoid replicating.

**Companions:** `design_4head_decomposition.md` (§3.5), `design_generator_crosslocus.md` (ceiling
unblock), `design_generalization_jtt.md` (recon = invariant supervision).
