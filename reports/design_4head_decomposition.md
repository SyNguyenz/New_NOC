# Design — Unified 4-Head Slot Decomposition (present-set / count / attr / phi)

**Status:** DESIGN (not yet implemented). Capture-context spec. **Date:** 2026-07-12.
**Scope:** Redesign the four decomposition heads around ONE shared slot decomposition, so count
becomes *learned* (not post-hoc), attr/phi use the *right operator/loss*, and the four heads *couple*
to resolve shared-allele ambiguity. Attacks the DEPLOYABLE bottleneck; does NOT claim to break the
N5 information floor.

> Epistemic note: numbers below from `results/inc22_fixed_aslot_seed42/metrics.json` are our own
> single-seed runs (not externally verified). The *design reasoning* is grounded in (a) verified code
> structure with `file:line` anchors and (b) external published methods (refs at bottom). When
> checking the implementation, trust the code anchors + papers, re-measure the numbers.

---

## 1. Motivation — the diagnosis (verified from code)

The core (present-set `cls` head + `reject` head + encoder `H`) is **strong**; a readout refit on
frozen `H` reaches high oracle → `H` is information-rich, the encoder is NOT the bottleneck. The
weakness is **three mis-designed decomposition heads**, each using the wrong tool:

| Head | Current impl (verified) | Defect | Anchor |
|---|---|---|---|
| **count** | `CardinalityHead` / `noc_head_v2` read `sigmoid(logits_cls).detach()` (+ `slot_mass.detach()`, `emphi_hill.detach()`, MAC) | post-hoc reader: (a) gradient never shapes encoder, (b) capped by ID quality, (c) **blind to raw height-stacking** — reads a detached, thresholded ID posterior | `set_transformer.py:667-688`, `:1801-1816`, `:1914`, `:2018` |
| **attr** | `attr_head = Linear(d_model, n_classes+1)`, **softmax** | **wrong operator**: softmax forces 1-donor/peak → *structurally* cannot represent a shared-allele peak that belongs to ≥2 donors | `set_transformer.py:1462`, comment `:1191` |
| **phi** | `softplus(phi_head(z))`, `F.l1_loss(out["phi"], phi)` | **wrong loss/target**: L1 on a 45-dim ~88%-zero target from ONE pooled vector, no presence-gating → collapses to ≈0 everywhere | `set_transformer.py:1463`, `train_set_transformer.py:1447` |

Measured consequence (inc22, in-silico test — **our single-seed numbers, LOW confidence, not
externally verified**): `oracle_em 0.973`, `em_post_hoc 0.959`, `em_noc_v2 0.919`, `em_joint_card 0.860`,
`card_noc_acc 0.967`, `per_noc_oracle` N5 = 0.831.

**Read these correctly (code-fact, `train_set_transformer.py:1896-1919`):** joint_card / post_hoc /
oracle all decode with the SAME ranking `rank_te` — they differ ONLY in the count `k`. So the low
`em_joint_card 0.860` is **NOT the slot decoder being weak**: the slot RANKING reaches oracle 0.973
(with the true count). It is specifically the **gate-derived COUNT** that trails RF — a count-quality
gap, consistent with the gate being *under-trained for counting* (§3.2), not a broken/weak decoder and
(audit found) not a wiring bug. So the design's target is narrow: strengthen the **gate-count**, keep RF
as a fallback until the gate-count demonstrably beats it on the same eval.

The count/attr/phi HEAD DEFECTS above are **code+math facts** (high confidence), independent of any
training number: softmax cannot represent a multi-donor peak; L1 on an ~88%-zero target minimizes to ≈0;
count reads `.detach()`. Those stand; the metrics only illustrate.

---

## 2. Unifying principle

`present-set`, `count`, `attr`, `phi` are **four facets of ONE decomposition**: *which k donors, at
what proportions, explaining which peaks*. The right architecture is a **shared generative
decomposition (slots) + specialized heads + mutual-consistency constraints** = a **learned,
differentiable EuroForMix** [EuroForMix]. The bones already exist in `AdaptiveSlotDecoder`
(`set_transformer.py:858-1027`):

- **Sinkhorn OT (MESH)** `:958-981` — soft fractional peak→slot assignment = the **attr** stream.
- **`slot_mass`** `:983-995` — mass-preserving per-slot assignment count = the **phi** stream + the
  SUM signal counting provably needs [DeepSets, GIN, Fischer-Gärtner].
- **gate (AdaSlot)** `:997-1020` — per-slot existence = the **count / present-set** facet.

Slots = 45 panel donors. `logits_cls = cls_head(slots) + gate_logit` (`:1016`); `gate` is
Binary-Concrete/Gumbel-Sigmoid with the *correct* Logistic noise (`:1000-1006`) [Concrete, Gumbel-
Softmax]; `logits_card = noc_head(gate)` with **no `.detach()`** (`:1018-1020`) → the gate is ALREADY
learned end-to-end, just under-trained for counting.

---

## 3. The four heads — target design + paper ref

### 3.1 Present-set (cls) — KEEP
Strong (macro-recall ~0.99). `logits_cls = cls_raw + gate_logit`. No change except it benefits from a
sharper gate (below). [Slot Attention]

### 3.2 Count (gate) — STRENGTHEN (the priority fix)  ✅ IMPLEMENTED 2026-07-15 (point 2+5; opt-in `--gate_count`)
Make the gate a **calibrated, mass-aware, count-consistent** existence gate so `count = Σ gate` is
*learned*, replacing the RF/CardinalityHead crutches.

> **Implemented (2026-07-15):** `--gate_count` adds `smooth_l1(Σ sigmoid(gate_logit), NOC)` (weight
> `--gate_count_weight`, default = beta). Gradient flows gate_head→slots→MESH→encoder (NOT detached).
> `gate`/`gate_logit`/`slot_mass` surfaced in `_forward_aslot`; decode adds a `gate_sum` diagnostic
> (`em_gate_sum`, `gate_count_acc`, `per_noc_gate_sum` in metrics.json). aslot-only; **default OFF →
> bit-identical (verified vs `git stash` baseline)**. Covers points **2** (Σgate≈NOC consistency) + **5**
> (gradient to encoder). **Point 3** (CORN on Σgate) still deferred — `noc_head_v2` does CORN but on
> DETACHED features.
>
> **UPDATE 2026-07-16 — points 1 + 4 now implemented (opt-in, default OFF, bit-identical verified):**
> • Point **1 mass-aware gate** `--gate_mass`: a grad-enabled per-slot mass (recomputed with gradient,
>   separate from the detached count-signal) is fed into `gate_logit` via a ReZero **zero-init** `mass_gate`
>   (`set_transformer.py` AdaptiveSlotDecoder) → no-op at start, gradient gate→slots→MESH→encoder.
> • Point **4 anneal** `--gate_temp_final`: trainer linearly sets `cls_decoder_module.gumbel_temp` from
>   `--gumbel_temp` → final over epochs (sharper gate late).
> Arms: `inc28_massgate_aslot` (=gate_count+gate_mass), `inc28_gatefull_aslot` (+anneal+phi_gated). NOT
> yet ablated for a win (needs full train + ≥3 seeds + no-regression N1–N4 vs RF/post_hoc on in-silico dev).

1. **Mass-aware gate.** Feed `slot_mass` (`:983-995`) INTO the gate/count decision (currently it is
   detached and only feeds `noc_v2`). "A slot is a real contributor iff it explains real peak mass" →
   injects the SUM aggregation counting needs. [DeepSets; GIN — sum injective on multisets;
   Fischer-Gärtner arXiv:2407.04170]
2. **Count-consistency constraint `Σ gate ≈ NOC`.** Direct supervision / regularizer on the sum of
   gate probabilities = true count → calibrates the gate for *counting*, not just *ranking* (present-
   set only needs top-k ORDER; counting needs a sharp on/off boundary at exactly k). [AdaSlot]
3. **Ordinal count head** — CORN/CORAL on `Σ gate` (or on sorted `slot_mass`) instead of
   `Linear(45→6)` + CE → rank-consistent, monotone-in-mass. [CORAL; CORN]
4. **Anneal Gumbel temperature** so the count reads a sharper gate (train-time noise → high variance).
   [Concrete; Gumbel-Softmax]
5. **Keep gradient to the encoder** (no detach) so `H` learns count-relevant (height/mass) features —
   the exact thing the current detached CardinalityHead cannot induce.

### 3.3 Attr — SOFT / MULTI-LABEL (fix the operator)
Replace hard softmax with the soft assignment. Two options already present:
- the **Sinkhorn OT assignment `A`** (`:958-981`) — fractional peak→slot, natively multi-donor, OR
- **`ml_attr_head`** (`set_transformer.py:1478`) — multi-label sigmoid (already coded, off by default).

Rationale: a shared-allele peak's height is a SUM of contributors; a soft/fractional assignment can
represent "belongs to A and B", a softmax cannot. [Slot Attention; Sinkhorn/Cuturi]

### 3.4 Phi — PRESENCE-GATED PROPORTION (fix the loss/target)  ✅ IMPLEMENTED 2026-07-16 (opt-in `--phi_gated`)
Drop the L1-on-sparse `phi_head`. Define phi as **normalized `slot_mass` among ACTIVE slots**
(gate-gated), i.e. proportion *within the predicted present set*, not a raw 45-dim regression. If a
learned head is kept, gate it on presence + use a proper heteroscedastic loss (β-NLL) to avoid
variance-collapse.

> **Implemented (2026-07-16):** `--phi_gated` keeps the learned `phi_head`, adds a `phi_logvar_head`, and
> replaces the L1 loss with a **presence-gated β-NLL** (Seitzer 2022): the loss is masked to the truly-
> present donors (`y>0.5`) so the ~88%-zero absent slots can't collapse phi→0, with a stop-grad β=0.5
> weight against variance-collapse. **Weighting caveat honoured (§4/§6b):** β-NLL can be NEGATIVE, so it
> uses a **FIXED weight** (`--phi_gated_weight`, default 0.3), NOT the Kendall wrapper (which would
> detonate `exp(−log_var)·(neg loss)→−∞` — the audited LUPI pitfall). Default OFF → L1 path unchanged. Also note the label caveat: the stored `phi` target is the **nominal design ratio**
(from the sample name), NOT the realized per-locus proportion — a partial label mismatch that bounds
any phi head. [β-NLL/Seitzer; NMF for the unmixing framing]

### 3.5 Coupling — the mechanism that resolves ambiguity (attr ↔ phi ↔ recon)
The reason attr+phi together beat attr alone (concretely): phi estimated from a donor's **private
alleles at clean loci** *predicts* how a shared-allele peak splits (`h ≈ Σ_c φ_c·dosage_c·scale`) →
resolves the shared-allele attribution. This is the EM E/M loop and is exactly the deployed +0.07
soft-split. Realize as:

- **Reconstruction-consistency (analysis-by-synthesis):** predicted `attr × phi` must re-sum to the
  observed height; add a reconstruction loss `‖ Σ_c A_c·φ_c·dosage_c − h_obs ‖`. **Per-sample → NOT
  blocked by the locus-independent generator** (unlike cross-locus). [NMF; Slot Attention recon]
- **Unrolled EM:** attr uses phi to bias shared-peak assignment (E-step); phi aggregates attr
  (M-step). The MESH iteration (`:965-981`) is already a soft version of this. [EuroForMix]

> Ceiling caveat: coupling = EM, so its ceiling = the identifiability floor (N5 minor's evidence lives
> in alleles SHARED with majors; phi from few/faint private alleles is low-SNR). It closes the
> broken-head→EM-ceiling gap and is trainable/calibratable end-to-end (unlike the post-hoc symbolic
> EM), but does NOT reach 1.0 at N5.

---

## 4. Gradient policy (do NOT blanket-detach)

Blanket detach = post-hoc = the weak-count problem we are fixing, AND it kills positive transfer
between four facets of one latent. Use **surgical** isolation instead:

1. **Share the trunk** for the four core heads (they mostly reinforce — same latent). *Test, don't
   assume:* task groupings are empirically non-obvious and related tasks can still interfere
   [Standley et al. ICML 2020]. Measure it (pre-flight below).
2. **START with tuned scalarization (weighted sum) — NOT gradient surgery.** Hard empirical evidence:
   specialized multi-task optimizers (PCGrad/MGDA/CAGrad/GradNorm) do **not reliably beat** careful
   loss-weighting under fair evaluation [Xin et al. NeurIPS 2022, "Do Current MTO Methods Even Help?"],
   and plain unitary scalarization + regularization matches/beats them [Kurin et al. NeurIPS 2022, "In
   Defense of the Unitary Scalarization"]. So: fixed/tuned weights first; add **PCGrad protect-main**
   (`train_set_transformer.py:211`) [Yu 2020] ONLY for a pair with *measured* negative gradient cosine.
3. **Weight cap, NOT naive Kendall on collapsible losses.** Kendall uncertainty [Kendall 2018] is fine
   for HARD tasks (loss bounded away from 0) but **detonates on privileged/easy losses**: the audit of
   `results/lupi_v*` traced the −22 collapse to `exp(−log_var)→∞` as a fittable privileged loss →0
   (§ stacking-doc §3). → use fixed weights or **floor `log_var`** for any loss that can reach ≈0.
4. **Decision rule for detach (per head, testable):** detach a head IFF a *deep* probe on **frozen H**
   reaches that head's ceiling (its features already in `H`). If not (suspected for count — needs
   height/mass features cls may not induce) → **don't detach**, share.
5. **Reserve detach** for genuinely orthogonal aux: `reject`/OOD (already detached via `pma_reject`,
   `:1741`) and privileged physics heads that a floored-weight can't tame.

Pre-flight (cheap, eval-only): **measure gradient cosine** between the four heads on a few batches. If
~0/positive → plain scalarized sum is fine (no surgery). Only turn on PCGrad for measured-conflict pairs.

---

## 5. What this fixes vs. what it does NOT

**Fixes (addressable, not generator-blocked):**
- Count becomes *learned* (Σ gate, mass-aware, ordinal) → removes the RF/post-hoc crutch; closes the
  `em_joint_card 0.86 → toward oracle 0.97` decode gap.
- attr/phi use the right operator/loss + couple → reaches the EM ceiling *end-to-end* (trainable,
  calibratable, combo-generalizable — unlike post-hoc symbolic EM).
- A physics-coupled decomposition is **combo-invariant by construction** (uses physics, not memorized
  co-occurrence) → may absorb part of the N5 combo-generalization gap for free.

**Does NOT fix:**
- The **N5 information floor** (near-far minor buried in shared alleles; `per_noc_oracle` N5 ≈ 0.83).
  No re-representation/coupling breaks it — it is a source-separation identifiability limit. Escaping
  it needs *more independent views* (cross-locus, blocked by the locus-independent generator) or *more
  measurement* (sequence-based MPS) — both out of scope here.

---

## 6. Relationship to the generalization direction (→ `design_generalization_jtt.md`)

Different wall: the 4-head work reaches ceilings deployably (count/decode + EM); the generalization
direction tries to raise the *N5 oracle generalization slice* (combo-memorization shortcut). **Full
spec now lives in `reports/design_generalization_jtt.md`.** Summary + sequence (not either/or):
1. **4-head first** — deployable bottleneck, high-confidence, not blocked, and it produces the CLEAN
   base needed to fairly judge any generalization method (testing on the current broken heads is
   uninterpretable). Its recon/EM coupling (§3.5) IS the *always-on invariant supervision* variant.
2. **Re-measure the N5 novel-combo gap on the clean base** (combo-disjoint dev). The physics-coupled
   decomposition is combo-invariant by construction → it may absorb the gap for free → direction moot.
3. If a real gap remains → **direct invariant supervision (JTT/LfF)**, not IRM: we KNOW the invariant
   (genotype ground truth) so supervise it directly [Liu 2021 JTT; Nam 2020 LfF] rather than discover
   it. Then the coded **ILC AND-mask** (`and_mask_backward`, `train_set_transformer.py:236`)
   [Parascandolo 2020] / **combo-augmentation** [Tobin 2017], with **donor-combo** environments (not
   NOC), judged on **N5 held-out-combo oracle**. IRMv1 / V-REx / GroupDRO = fragile last-resort
   controls [Arjovsky 2019; Krueger 2021; Sagawa 2020; Rosenfeld 2021; Gulrajani-Lopez-Paz 2021].

---

## 6b. Empirical risks & caveats (paper-grounded; our own metrics are LOW-confidence)

Evidence discipline: **our single-seed, un-peer-reviewed training numbers are low-confidence** and are
NOT used to justify design choices; only **code+math facts** and **published papers** are. Under that
filter:

- **Slot-attention is finicky on non-toy data** [Seitzer et al. ICLR 2023 "Bridging the Gap to
  Real-World Object-Centric Learning" (DINOSAUR); Locatello 2020]: raw slot attention often needs
  strong feature priors and can collapse. *Mitigant here:* the audit shows the slot **ranking** is fine
  (shared `rank_te` → oracle 0.973); only the **gate-count** trails → keep RF as fallback until the
  gate-count wins on the same eval.
- **Gradient surgery ≈ no better than tuned weights** [Xin 2022; Kurin 2022] → lead with scalarization,
  PCGrad only on measured conflict (§4).
- **Reconstruction alone ≠ semantic decomposition** [Locatello et al. ICML 2019 best paper] →
  privileged `contrib` supervision is REQUIRED, not optional (stacking-doc §3).
- **Kendall detonates on collapsible privileged losses** — the audited cause of the LUPI −22 collapse,
  NOT β-NLL (stacking-doc §3) → floor `log_var` / fixed weights.
- **Stacking many objectives rarely compounds** [Kurin 2022; Standley ICML 2020] → **ablation-first**:
  add ONE mechanism at a time with a no-regression guard (N1–N4), ≥3 seeds; do not ship the full stack
  blind.
- **Two things to AUDIT before building** (both consistent with a possible pipeline bug, not a
  fundamental limit): (a) the gate-count under-performance — re-derive on in-silico dev, seeds;
  (b) re-run one LUPI arm with `log_var` floored to confirm the collapse was the Kendall pitfall.

---

## 7. Open decisions / next steps (to discuss before coding)

- [ ] Attr stream: use the Sinkhorn OT `A` directly, or the separate `ml_attr_head`? (OT reuses the
      decomposition; ml_attr_head is a cleaner isolated stream.)
- [ ] Phi: fully symbolic (normalized `slot_mass`, no learned head) vs a presence-gated learned head
      with β-NLL? (Symbolic already feeds count well; learned adds trainability but needs the recon
      coupling to not collapse.)
- [ ] Count-consistency: soft regularizer `Σgate≈NOC` vs hard supervision on `Σ slot_mass`? Ordinal on
      `Σgate` vs on sorted `slot_mass`?
- [ ] Reconstruction target: reconstruct raw `h` (RFU) or `log1p(h)`? needs `DONOR_DOSAGE` + per-locus
      scale (efm-style) — reuse `_em_deconv_phi` (`set_transformer.py:1819`)?
- [ ] Gradient: run the **cosine + frozen-probe** pre-flight first; decide PCGrad on/off per pair.
- [ ] No-regression guard: N1–N4 EM must not drop; ≥3 seeds (N5 run-variance is large).

---

## 8. References

Confidence marked: ✓ verified citation; ⚠ cited in existing code comments, **verify exact
citation before relying on** (may be author shorthand).

**Slot decomposition / object-centric**
- ✓ Locatello et al., *Object-Centric Learning with Slot Attention*, NeurIPS 2020, arXiv:2006.15055.
- ⚠ AdaSlot — *Adaptive Slot Attention: object discovery with dynamic slot number*, CVPR 2024 (Gumbel-
  Sigmoid existence gate for adaptive slot count). [code: `set_transformer.py:856,997`]
- ⚠ CoSA (ICLR 2024) — genotype/external-signal-conditioned slot init. [code comment `:853`]
- ⚠ GSANet (CVPR 2024) — attr-soft guided slot-init refinement. [code comment `:854`]
- ⚠ MESH (ICML 2023) — Sinkhorn OT replaces competitive softmax in the slot loop. [code comment `:855`]
- ✓ Cuturi, *Sinkhorn Distances: Lightspeed Computation of Optimal Transport*, NeurIPS 2013,
  arXiv:1306.0895.

**Discrete gate / reparameterization**
- ✓ Maddison, Mnih, Teh, *The Concrete Distribution*, ICLR 2017, arXiv:1611.00712.
- ✓ Jang, Gu, Poole, *Categorical Reparameterization with Gumbel-Softmax*, ICLR 2017, arXiv:1611.01144.
  (Logistic noise = difference of two Gumbels; a single Gumbel biases the gate "active" — see
  `set_transformer.py:1000-1006`.)

**Counting / multiset aggregation**
- ✓ Zaheer et al., *Deep Sets*, NeurIPS 2017, arXiv:1703.06114 (sum is injective on multisets).
- ✓ Xu et al., *How Powerful are Graph Neural Networks?* (GIN), ICLR 2019, arXiv:1810.00826.
- ⚠ Fischer & Gärtner 2024, arXiv:2407.04170 — mass-preserving count signal (cited `set_transformer.py:983`).

**Ordinal regression (count head)**
- ✓ Cao, Mirjalili, Raschka, *Rank-consistent ordinal regression* (CORAL), Pattern Recognition Letters
  2020, arXiv:1901.07884.
- ✓ Shi, Cao, Raschka, *Deep NN for Rank-Consistent Ordinal Regression* (CORN), 2023, arXiv:2111.08851.

**Multi-task gradient policy (+ the evidence AGAINST leading with surgery — §4, §6b)**
- ✓ Yu et al., *Gradient Surgery for Multi-Task Learning* (PCGrad), NeurIPS 2020, arXiv:2001.06782.
  [code: `train_set_transformer.py:211`]
- ✓ Kendall, Gal, Cipolla, *Multi-Task Learning Using Uncertainty to Weigh Losses*, CVPR 2018,
  arXiv:1705.07115.
- ✓ Xin et al., *Do Current Multi-Task Optimization Methods in Deep Learning Even Help?*, NeurIPS 2022,
  arXiv:2209.11379 (MTO methods don't reliably beat tuned scalarization).
- ✓ Kurin et al., *In Defense of the Unitary Scalarization for Deep Multi-Task Learning*, NeurIPS 2022,
  arXiv:2201.04122.
- ✓ Standley et al., *Which Tasks Should Be Learned Together in Multi-Task Learning?*, ICML 2020,
  arXiv:1905.07553 (task groupings are non-obvious; ablate).

**Decomposition needs supervision + slot fragility (§6b)**
- ✓ Locatello et al., *Challenging Common Assumptions in the Unsupervised Learning of Disentangled
  Representations*, ICML 2019 (best paper), arXiv:1811.12359 (recon alone ≠ semantic factors).
- ✓ Seitzer et al., *Bridging the Gap to Real-World Object-Centric Learning* (DINOSAUR), ICLR 2023,
  arXiv:2209.14860 (slot attention needs strong feature priors on real data).
- ✓ Parascandolo et al., *Learning explanations that are hard to vary* (ILC AND-mask), 2020,
  arXiv:2009.00329. [code: `train_set_transformer.py:236`]

**Heteroscedastic / variance-collapse (phi)**
- ✓ Seitzer et al., *On the Pitfalls of Heteroscedastic Uncertainty Estimation* (β-NLL), ICLR 2022,
  arXiv:2203.09168.
- ✓ Lee & Seung, *Learning the parts of objects by non-negative matrix factorization*, Nature 1999
  (unmixing / reconstruction framing).

**Forensic model (the physics we reconstruct)**
- ✓ Bleka, Storvik, Gill, *EuroForMix: an open source software...*, Forensic Sci. Int. Genetics 2016
  (continuous gamma peak-height model: Mx/φ, degradation, stutter, drop-in/out).

**Deferred invariance direction (§6)**
- ✓ Arjovsky et al., *Invariant Risk Minimization*, 2019, arXiv:1907.02893.
- ✓ Krueger et al., *Out-of-Distribution Generalization via Risk Extrapolation* (V-REx), ICML 2021,
  arXiv:2003.00688.
- ✓ Sagawa et al., *Distributionally Robust Neural Networks* (GroupDRO), ICLR 2020, arXiv:1911.08731.
- ✓ Rosenfeld, Ravikumar, Risteski, *The Risks of Invariant Risk Minimization*, ICLR 2021,
  arXiv:2010.05761.
- ✓ Gulrajani & Lopez-Paz, *In Search of Lost Domain Generalization* (DomainBed), ICLR 2021,
  arXiv:2007.01434.

---

## 9. Code anchors (for implementation check)

- `models/set_transformer.py:667-688` — `CardinalityHead` (detached post-hoc count; to be replaced).
- `models/set_transformer.py:764-824` — donor-slot private/shared attention (Level-1/2).
- `models/set_transformer.py:858-1027` — `AdaptiveSlotDecoder` (CoSA/GSANet/MESH/AdaSlot). Gate
  `:997-1020`; `slot_mass` `:983-995`; `logits_cls = cls_raw + gate_logit` `:1016`;
  `logits_card = noc_head(gate)` `:1018-1020`.
- `models/set_transformer.py:1191` — comment: softmax attr structurally can't learn shared-allele.
- `models/set_transformer.py:1462` — `attr_head` (softmax); `:1478` — `ml_attr_head` (multi-label).
- `models/set_transformer.py:1463` — `phi_head`; softplus readouts `:1902,1918,1980,2028,2146`.
- `models/set_transformer.py:1819` — `_em_deconv_phi` (symbolic EM phi, feeds `emphi_hill`).
- `train_set_transformer.py:1443-1449` — attr(CE)+phi(L1) aux loss, Kendall-weighted.
- `train_set_transformer.py:211-233` — `pcgrad_backward` (protect-main).
- `train_set_transformer.py:236+` — `and_mask_backward` (ILC AND-mask).
- `results/inc22_fixed_aslot_seed42/metrics.json` — baseline numbers.
