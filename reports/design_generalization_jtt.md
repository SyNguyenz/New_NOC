# Design — Generalization via Direct Invariant Supervision (JTT / LfF), NOT IRM

**Status:** DESIGN (not yet implemented). Capture-context spec. **Date:** 2026-07-12.
**Scope:** How to raise the N5 *combo-generalization* slice of the oracle ceiling — the
train→novel-combo gap driven by donor-combo memorization (shortcut learning). Argues for **directly
supervising the KNOWN invariant rule** (we have genotype ground truth) via JTT/LfF-style two-pass
training, and **demotes IRM** to a fragile last-resort control. Companion to
`design_4head_decomposition.md` (this replaces that doc's §6 framing).

> Epistemic note: internal findings (F29/F30/F38, `em_*` numbers) are our own single-seed runs, NOT
> externally verified — cited as *observations to re-measure*, not authority. The design reasoning
> rests on (a) external published methods (refs at bottom) and (b) the problem's causal structure.

---

## 1. Motivation — the wall this attacks, and why IRM is the wrong FIRST tool

**Wall:** at high NOC the model memorizes which donor COMBOS co-occur (a spurious correlation) and
fails to compose UNSEEN combos — classic shortcut learning [Geirhos 2020]. This is the
*generalization-limited* slice of the N5 oracle ceiling (distinct from the *information-limited* slice
= near-far minor buried in shared alleles, which no learning fixes). Our runs suggest a large
train→held-out-combo gap at N5 (to re-measure on a clean base).

**Why not IRM first:** IRM [Arjovsky 2019] exists precisely because you usually DON'T know which
feature is invariant, so it must be *discovered* from multiple environments. It is empirically fragile
— under fair model selection ERM ties/beats it [Gulrajani & Lopez-Paz 2021 DomainBed], and it can
provably fail with too few environments [Rosenfeld 2021]. A prior in-house IRM arm was also
mis-specified (NOC-partition environments, non-descending penalty, judged on N1-dominated aggregate EM).

**Key insight (the pivot):** for this problem we **KNOW the invariant rule from ground truth**, so we
do not need IRM's discovery at all — we can **supervise the invariant directly**, which is stronger,
simpler, and more stable.

---

## 2. What the invariant IS (and that we have its ground truth)

- **Invariant (causal) rule:** *a donor is present ⟺ its alleles are present in the peaks AND its
  implied per-locus proportion is consistent with a single φ (height-consistency)* — i.e. the
  EuroForMix physics [EuroForMix]. This rule is **invariant to which combo appears**.
- **Spurious rule:** *donors {a,b,c} tend to co-occur* (memorized co-occurrence).
- **Ground truth we already have:** per-donor genotype (`DONOR_DOSAGE`), per-peak attribution
  (`attr`), present-set (`y`), count (`noc`), proportion (`phi`). These *encode the invariant rule*.

So "teach the model which feature is invariant" = supervise the genotype/attribution/height-consistency
channel so the model relies on physics, not co-occurrence. This is the *correct* use of privileged
information [Vapnik-Izmailov LUPI 2015; Lopez-Paz generalized distillation 2016] — as a *feature-
selection signal*, NOT the unbounded-NLL form that made the earlier LUPI physical heads collapse.

---

## 3. Method — two-pass "probe then teach" = JTT / LfF

The user's "×2 resource: pass-1 test how well it understands, pass-2 teach the invariant" maps to:

- **JTT — Just Train Twice** [Liu et al. ICML 2021, arXiv:2107.09044]: train once → identify the
  examples the model gets wrong *for the shortcut reason* → upweight them + retrain. Simple, no
  environment labels, often more robust than IRM.
- **LfF — Learning from Failure** [Nam et al. NeurIPS 2020, arXiv:2007.02561]: train a deliberately
  *biased* model, use its failures to reweight a *debiased* model.
- **Invariant-feature supervision:** use ground-truth attribution/genotype to directly supervise WHICH
  peaks/features the model should key on (allele-matching + height-consistency), always-on.

---

## 4. Two variants (they compose)

### (a) Always-on invariant supervision — subsumed by the 4-head coupling
Enforce the physical invariant at *every* sample: the predicted `attr × phi` must reconstruct the
observed height (analysis-by-synthesis), and a donor's implied-φ must be consistent → rejects the
combo-plausible-but-physically-inconsistent decoy. **This is exactly the reconstruction/EM coupling in
`design_4head_decomposition.md` §3.5.** Per-sample → **NOT blocked by the locus-independent
generator**. Simplest; already on the 4-head roadmap.

### (b) JTT-targeted — the explicit two-pass
1. **Pass 1:** train the 4-head base normally.
2. **Detect** (see catch #2): on a **combo-disjoint dev set**, find samples where the model picks a
   DECOY (present-set false positive driven by shortcut).
3. **Pass 2:** retrain upweighting those samples + strengthening the invariant supervision
   (correct attribution / height-consistency) on them.

Cost = ×2 training (JTT's literal cost) — bounded and **cheaper/stabler than tuning IRM**.
Relative of the existing `minor_phi_ref` hard-example weighting (`train_set_transformer.py:1083,1225`),
but *targeted at shortcut-driven decoy errors*, not merely low-φ minors.

---

## 5. Two honest catches (decide these before coding)

**Catch 1 — converges with the 4-head fix; 4-head is a PREREQUISITE.**
We already supervise the invariant (attr/cls) yet the model still memorizes combos — because the
invariant *channel* (softmax `attr_head`) is broken and cannot carry "allele-matching +
height-consistency". "Teach the invariant better" therefore *requires* the corrected soft/coupled
attr+phi = the 4-head work. Do 4-head first; JTT operates on that clean base.

**Catch 2 — the detection signal must come from HELD-OUT-COMBO, not train errors.**
Vanilla JTT mines *training* errors. But combo-memorization *helps every seen combo* → the biased model
makes **few training errors** → train-error mining can MISS the shortcut cases. Detection must use a
**combo-disjoint dev set** (`make_dev_split.py` — held-out combos) to expose shortcut reliance. This is
a mandatory modification of stock JTT for this problem.

---

## 6. Ceiling (honest bounds)

Recovers the **generalization-limited** slice: decoys that ARE physically distinguishable
(inconsistent implied-φ) but the model missed by leaning on the combo shortcut. Does **NOT** recover
the **information-limited** slice: an N5 decoy fully covered by the true donors' shared alleles AND
coincidentally φ-consistent is unresolvable — the identifiability floor (`per_noc_oracle` N5 ≈ 0.83).
Neither JTT nor any supervision breaks it (needs more independent views = cross-locus, generator-
blocked; or more measurement = MPS).

---

## 7. Sequence / priority

1. **4-head first** (`design_4head_decomposition.md`) — fixes the invariant channel AND its always-on
   invariant supervision (variant (a)) is part of the coupling.
2. **Re-measure the N5 novel-combo gap on the clean 4-head base** (combo-disjoint dev). A physics-
   coupled decomposition is combo-invariant by construction → it may *absorb the gap for free*, making
   further work moot.
3. **If a real gap remains → variant (b) JTT-targeted.** If still short, try the coded **ILC AND-mask**
   (`and_mask_backward`, `train_set_transformer.py:236`) [Parascandolo 2020] or **combo-augmentation /
   domain-randomization** [Tobin 2017] with **donor-combo** environments (NOT NOC), judged on **N5
   held-out-combo oracle** (not aggregate EM).
4. **IRMv1 / V-REx / GroupDRO = fragile last-resort controls only** [Arjovsky 2019; Krueger 2021;
   Sagawa 2020; Rosenfeld 2021; DomainBed 2021].

---

## 8. Open decisions

- [ ] Detection metric for "shortcut-driven decoy pick" on combo-disjoint dev: implied-φ inconsistency
      of the picked decoy? present-set false-positive whose alleles are fully shared-covered?
- [ ] Pass-2 mechanism: sample upweighting (JTT) vs an explicit invariant-consistency loss upweighted
      on the detected set vs both.
- [ ] Is variant (a) (always-on coupling) alone enough, or is (b) targeting needed? Decide *after*
      step 2 re-measure.
- [ ] Reweighting interaction with the existing `minor_phi_ref` (Inc6) — unify or keep separate.
- [ ] Guard: N1–N4 EM must not regress; ≥3 seeds (N5 run-variance large).

---

## 9. References

Confidence: ✓ verified citation.

**Two-pass robustness (the core method)**
- ✓ Liu et al., *Just Train Twice: Improving Group Robustness without Training Group Information* (JTT),
  ICML 2021, arXiv:2107.09044.
- ✓ Nam et al., *Learning from Failure: De-biasing Classifier from Biased Classifier* (LfF),
  NeurIPS 2020, arXiv:2007.02561.
- ✓ Geirhos et al., *Shortcut Learning in Deep Neural Networks*, Nature Machine Intelligence 2020,
  arXiv:2004.07780.

**Privileged information / invariant supervision**
- ✓ Vapnik & Izmailov, *Learning Using Privileged Information*, JMLR 2015.
- ✓ Lopez-Paz et al., *Unifying distillation and privileged information* (generalized distillation),
  ICLR 2016, arXiv:1511.03643.

**Invariance / DG (deferred controls)**
- ✓ Arjovsky et al., *Invariant Risk Minimization* (IRM), 2019, arXiv:1907.02893.
- ✓ Krueger et al., *Out-of-Distribution Generalization via Risk Extrapolation* (V-REx), ICML 2021,
  arXiv:2003.00688.
- ✓ Sagawa et al., *Distributionally Robust Neural Networks* (GroupDRO), ICLR 2020, arXiv:1911.08731.
- ✓ Rosenfeld, Ravikumar, Risteski, *The Risks of Invariant Risk Minimization*, ICLR 2021,
  arXiv:2010.05761.
- ✓ Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization* (DomainBed), ICLR 2021,
  arXiv:2007.01434.
- ✓ Parascandolo et al., *Learning explanations that are hard to vary* (ILC AND-mask), 2020,
  arXiv:2009.00329.
- ✓ Tobin et al., *Domain Randomization for Transferring Deep Neural Networks...*, IROS 2017,
  arXiv:1703.06907 (the augmentation/randomization principle).

**Physical invariant (the rule we supervise)**
- ✓ Bleka, Storvik, Gill, *EuroForMix: an open source software...*, Forensic Sci. Int. Genetics 2016
  (Mx/φ + height-consistency = the decoy-rejection invariant).

---

## 10. Code anchors

- `train_set_transformer.py:236` — `and_mask_backward` (ILC AND-mask; a coded invariance surgery).
- `train_set_transformer.py:1083,1225` — `minor_phi_ref` (Inc6 hard-example weighting; JTT relative).
- `train_set_transformer.py:211-233` — `pcgrad_backward` (gradient policy, cross-ref 4-head doc §4).
- `make_dev_split.py` — combo-disjoint dev split (the held-out-combo detection set for catch #2).
- attr / `phi` / `DONOR_DOSAGE` labels — the ground-truth invariant channel.
- `results/inc22_fixed_aslot_seed42/metrics.json` — baseline; `per_noc_oracle` N5 ≈ 0.83 (info floor).

**Companion:** `reports/design_4head_decomposition.md` (prerequisite; §3.5 = always-on variant (a)).
