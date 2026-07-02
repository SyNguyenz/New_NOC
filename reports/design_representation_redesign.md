# Design Doc — Representation & Architecture Redesign for STR Mixture Deconvolution (Donor ID + NOC)

**Status:** proposal v1.0 (2026-05-31).
**Scope:** the DECONVOLUTION task — reconstruct WHICH of 45 known donors contributed (a set) + HOW MANY
(NOC). This is mathematically distinct from likelihood-ratio (LR) evaluation [Zhu 2026]; **LR is out of scope**.
Metric = contributor/genotype concordance (per-donor Exact Match), NOT LR calibration.
**Grounding:** related-task literature (below), NOT prior in-house experiments — those are CONFOUNDED by an
impoverished 3-field token that crippled the Set Transformer (ST). Direct evidence of the confound: switching
the token from per-allele to per-marker alone produced a large jump on the small real set.

---

## 1. Problem & motivation

One mixed STR electropherogram (EPG) → two coupled outputs:
- **ID** (primary): the contributor SET (subset of 45 known donors).
- **NOC**: number of contributors = cardinality of that set. NOC misspecification substantially degrades ID
  reconstruction [Zhu 2026] → the two are coupled, not independent.

Findings this session that motivate a redesign (not a decoder tweak):
- The decoded↔oracle EM gap is pure **cardinality**; ranking is near-oracle.
- **NOC5 is NOT a physical ceiling.** Raw-genotype re-measurement (2026-06-06, measure_noc5_ceiling.py):
  **100% of NOC5 contributors have a PRIVATE allele PRESENT** (rankable; model under-ranks); count is
  presence-provable in 92% of NOC5 samples. [The earlier "only 24% genuine dropout" was a synth-genotype
  coverage artifact — REMOVED; see findings-log F6.] Independently corroborated: ML counters reach NOC5
  60–94% [Mao 2026, Table 2] → counting/ranking high-NOC is learnable, representation-dependent.
- Current token = `(locus_idx, allele_raw, log_height_raw)`: 3 fields, RAW scalars, NO relational features,
  NO per-feature embedding, NO pretraining. By contrast deepNoC uses 24 loci × 50 peaks × **89 features/peak**.
- ⇒ All prior "ST weak / hybrid best / count-features-useless / NOC5 physical" conclusions are **confounded by
  this token** and are NOT evidence against the redesign.

## 2. Task framing — supervised unmixing with known endmembers (+ cardinality)

A mixture EPG ≈ a non-negative combination of the contributors' single-source profiles — structurally a
**supervised hyperspectral-unmixing** problem (observed = Σ abundanceₖ·endmemberₖ), and we HAVE the endmembers
(45 donor single-source references). Frame as:
- predict **abundances** φ ∈ ℝ⁴⁵ (φ_d = contribution of donor d), constrained ANC (φ≥0) [+ optional ASC Σ=1];
- **ID** = {d : φ_d > τ}; **NOC** = |{d : φ_d > τ}| → ID and NOC emerge from ONE head (unification);
- a **learned decoder** reconstructs the observed EPG from φ + references → **reconstruction loss** that is
  physics-grounded and self-supervised (no donor labels → usable on abundant in-silico + unlabeled real).

This task is a **deconvolution-specific architecture** [Zhu 2026 blueprint]; we adopt 4 of its 5 features:
(i) **NOC as model-selection with a parsimony penalty** (our cardinality-aware / AIC–BIC work);
(ii) **multimodal / top-N output** — report MULTIPLE high-probability contributor sets, not just top-1,
because in allele-sharing cases several sets explain the data nearly equally (directly targets our 76%-rankable
NOC5 misses); (iii) **mixture-proportion estimation** = the abundances φ; (iv) **marker-specific noise**
(stutter/dropout, not uniform) = enriched stutter features (§3) + a learned decoder (§4); (v) **inter-locus
dependency modeling** = the encoder (§4), shown by Yu 2025 to add ~30 pp genotype concordance.

## 3. Input representation (enriched per-peak token)

Per observed peak, fields — EACH embedded, never fed as a raw scalar:

| field | meaning | basis |
|---|---|---|
| locus | which locus | Embedding(24, d) (have) |
| allele | repeat number | numerical embedding (PLE / periodic) |
| log_height | RFU | numerical embedding (PLE / periodic) |
| **height / locus-total** | heterozygote balance (Hb) | Bright 2012 — relational |
| **stutter ratio** = h / h(allele+1) at locus | true-minor vs artefact | Taylor 2013 |
| **rank-in-locus**, **n-peaks-at-locus** | intra-locus context | PACE / deepNoC |
| **height / profile-max** | global contributor tier | deconvolution |
| **size (bp)** | fragment size → degradation | STRmix degradation model (in CSV already — `extract_size.py`, no re-prep) |

**Per-feature numerical embeddings** fix the harmful rotation-invariance that makes plain NNs lose to
trees/aligned on lookup-style tabular targets [Grinsztajn 2022]:
- **PLE** (piecewise-linear): bins via quantile or target-tree; scalar → [e₁…e_T], lossless; or
- **Periodic**: concat[sin(2πc_i x), cos(2πc_i x)], c_i ~ N(0,σ) trainable, then linear(+ReLU).
Refs: FT-Transformer (Gorishniy 2021), On Embeddings for Numerical Features (Gorishniy 2022).
Note: Hb / SR / rank / n-peaks / global-rel are **derivable from EXISTING tokens** (no re-prep). **CORRECTION
(verified):** size(bp) also needs NO re-prep — the GeneMapper CSVs already export (Allele, Size, Height) and
prepare_data_set.py only dropped the Size column; `extract_size.py` recovers it as a token-aligned sidecar
(100% coverage). So the whole enriched token incl. size + degradation is reachable without instrument re-prep.

## 4. Architecture (multiple encoders for a fair head-to-head)

```
enriched tokens (N, 160, K) --per-field embedding--> token embeddings (N, 160, d)
        │
   ENCODER  (one of, compared head-to-head — §7,§8):
     A. ST/attention over peaks  (within-donor inter-locus + Hb/SR relations; locus/allele identity in
        tokens ⇒ NOT pure permutation-invariance — alignment kept)
     B. CNN/ResNet over aligned loci  (Yu 2025 used this for inter-locus dependency — attention NOT proven necessary)
     C. flat-only (Xflat aligned base = LR/XGB-like per-donor lookup)
     D. hybrid = aligned base (C) + set/attention stream (A)        ← current best under the OLD token
        │
        ├─ abundance head  φ ∈ ℝ⁴⁵  (ANC: softplus/ReLU)   → ID (φ>τ) + NOC (|φ>τ|) + multimodal top-N sets
        ├─ reject head (open-set)                          → has-unknown contributor
        └─ DECODER (LEARNED, NONLINEAR): φ, refs → reconstruct EPG → reconstruction loss (self-supervised)
```
- **Decoder MUST be learned/nonlinear** (model stutter/dropout/degradation implicitly). Linear NNLS = our
  confounded pgNOC baseline; HSU autoencoders and Yu 2025 use learned decoders.
- **"Which encoder wins is EMPIRICAL"** — do not assume. External evidence is mixed: FT-Transformer beats GBDT
  only 7/11; Yu 2025's strongest deep result is ResNet that RE-WEIGHTS a continuous engine (i.e., hybrid-like,
  not pure-deep). So the head-to-head {A,B,C,D} on the SAME rich tokens is the deciding experiment.
- **Multimodal output**: emit the top-N candidate contributor sets (not only argmax) for allele-sharing cases.

## 5. Losses

`L = L_ID + α·L_NOC + β·L_reject + γ·L_recon  (+ δ·L_contrastive during pretraining)`
- **L_ID**: ASL (or BCE) on membership φ vs the true 45-label set. (ASL won earlier — emphasizes faint minorities.)
- **L_NOC**: NOC as model-selection — cardinality-aware (Cortes/Mohri) or AIC/BIC over k, parsimony-penalized
  [Zhu 2026 (i)]; OR keep the external two-stage decoder if it remains better — decide empirically.
- **L_recon**: learned decoder reconstructs observed heights from φ·refs (physics self-supervision; label-free).
- **ANC** (+ optional ASC) constraints on φ.

## 6. Pretraining (we have abundant in-silico, scarce real labels)

- **Primary — reconstruction pretraining**: run the unmixing autoencoder on in-silico + unlabeled real (no
  donor labels) → learn peak/contributor representations → fine-tune ID+NOC on labels. Physics-grounded and
  label-free; preferred over generic SSL because it uses the additive mixture structure.
- **Optional — SAINT-style** (Somepalli 2021): contrastive (CutMix raw p=0.3 + mixup in embedding space α=0.2,
  InfoNCE τ=0.7) + denoising. ADAPT, do NOT copy: SAINT's CLS→single-label head is REJECTED (our output is a SET).

## 7. Evaluation protocol (measurement is a binding constraint — handle explicitly)

- **Primary dev = in-silico held-out NOVEL combos** (large, hard, labeled). Real val N4 is SATURATED
  (0.979 vs test 0.689) ⇒ it CANNOT select for the hard-NOC regime → do not select on it.
- **Real test = evaluation ONLY, once**; report with explicit caveat (N4 n=45, N5 n=48 → wide CIs).
- **Head-to-head on the SAME rich tokens**: encoders {A ST, B CNN/ResNet, C flat, D hybrid}. Select architecture
  + hyperparameters on in-silico dev; report test once. NO test-tuning (this session: "routing 0.956" and
  "+pgNOC 0.964" both EVAPORATED under honest selection).
- **Metrics**: oracle (ranking ceiling) + decoded per-NOC EM; **top-N set recall** (is the true set among the
  model's top-N candidate sets? — the multimodal metric, captures the 76% rankable NOC5 misses); NOC count
  accuracy; dropout-vs-rankable split for NOC5; reconstruction error.
- **Quantify the in-silico→real domain gap** (in-silico-dev metric minus real-test metric) and report it — it
  is the standing risk, not a footnote.

## 8. Experiment plan (incremental; each independently testable; cheap → expensive)

1. **Enriched token + per-feature embeddings** (Hb/SR/rank from existing tokens; PLE or periodic). Retrain &
   compare encoders {A,B,C,D}. Measure oracle + decoded per-NOC + top-N recall on in-silico dev + real test.
2. **Reconstruction (unmixing) loss** added (learned decoder) → does it lift ranking (oracle N4/N5) vs (1)?
3. **Reconstruction pretraining** on in-silico → fine-tune → test data-efficiency and real transfer.
4. **Add Size(bp)** — already in the CSVs (`extract_size.py` sidecar, no re-prep); wire into the token +
   a degradation feature (height-vs-size). For in-silico, attach size via a per-(locus,allele) size table
   (size is ~deterministic per bin). Enables the degradation relationship without instrument re-prep.
5. **Optional**: intersample attention; contrastive pretraining; explicit joint set-prediction loss.

## 9. Risks & open questions

- **Domain gap (in-silico→real)** — the recurring binding constraint. A richer representation MAY generalize
  better (the per-marker jump improved real too), but this is unproven. **Parallel lever (not architecture):**
  improve `make_insilico.py` realism — per-locus gamma peak-height noise, degradation slope, stutter — so the
  reconstruction/representation learned on in-silico transfers. Track the gap (§7).
- **Measurement** — tiny real high-NOC strata + saturated val. Mitigation: in-silico dev primary, honest CIs,
  top-N recall, never select on real test.
- **Inter-locus memorization** — learn WITHIN-donor associations (train the dependency model on single-source,
  as Yu 2025 does); the combo-diverse in-silico prevents cross-donor co-occurrence from being memorized.
- **Linear vs learned decoder** — must be learned/nonlinear; linear NNLS (pgNOC) is the confounded baseline.
- **Pure-deep vs hybrid** — external evidence (Yu 2025, FT-Transformer) leans hybrid (deep + continuous), but
  this is exactly what the §7 head-to-head must settle on rich tokens — do not pre-judge.
- **Effort** — re-prep (only for size) + new heads/decoder + retrain + pretrain. Increments 1–2 are cheap-ish
  (no re-prep); 3–4 heavier.

## 10. Decision log (adopt / adapt / reject)
- **ADOPT**: per-feature numerical embeddings (PLE/periodic); enriched relational tokens (Hb/SR/rank/size);
  learned nonlinear unmixing decoder + ANC; reconstruction self-supervision; NOC as parsimony-penalized
  model-selection; multimodal top-N output + top-N set-recall metric; CNN/ResNet encoder in the head-to-head
  (Yu 2025); within-donor inter-locus modeling; deconvolution-only framing (no LR).
- **ADAPT**: SAINT contrastive/denoising pretraining — optional, after reconstruction pretraining.
- **REJECT**: copying SAINT wholesale (single-label CLS head — output is a SET); linear-only decoder; pure
  permutation-invariance that discards locus×allele alignment; selecting ANY choice on real test or on the
  saturated real val; treating NOC5 as a physical ceiling; LR objectives.

## 11. References (key)
- Yu et al. 2025 — deep ST R mixture deconvolution via locus-association ResNet re-weighting a continuous engine, +30pp concordance. Int J Legal Med, 10.1007/s00414-025-03677-x.
- Zhu, Mao, Zhang 2026 — DNA Mixture Deconvolution: A Four-Strategy Framework. Genes 17(4):434 (10.3390/genes17040434). [deconvolution≠LR; 5-feature blueprint]
- Mao et al. 2026 — Mixed STR profile deconvolution methods & intelligence trends. Forensic Science and Technology 51(1):57–66. [NOC ML Table 2; CNN/BNN/RNN/generative directions]
- deepNoC 2024 — arXiv 2412.09803 (24×50×89 features, MHCNN artefact classifier).
- Grinsztajn et al. 2022 — Why tree-based models still outperform DL on tabular. arXiv 2207.08815.
- Gorishniy et al. 2021 — Revisiting DL for tabular (FT-Transformer). arXiv 2106.11959.
- Gorishniy et al. 2022 — On Embeddings for Numerical Features. arXiv 2203.05556 (PLE, periodic).
- Somepalli et al. 2021 — SAINT. arXiv 2106.01342 (intersample attn, contrastive+denoising pretrain).
- Rezatofighi et al. — DeepSetNet (arXiv 1611.08998) / Deep Perm-Set Net (1805.00613); Joint Cardinality (1709.04093). [set prediction = membership+cardinality]
- HSU autoencoder unmixing + simultaneous #endmembers — Sensors 2025, 1424-8220/25/8/2592.
- Bright et al. 2012 (heterozygote balance), Taylor et al. 2013 (allelic+stutter peak-height models) — FSI:Genetics.
- EuroForMix (Bleka 2016), NOCIt (Swaminathan 2015), STRmix (Bright 2016).
