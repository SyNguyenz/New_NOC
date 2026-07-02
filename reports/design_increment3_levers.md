# Design Doc — Increment 3: ID-representation + NOC-ordinal levers (each = ONE published method, ablated)

**Status:** proposal 2026-06-06. Built AFTER the F6/ID-ambiguity correction (NOC5 ID is MODEL-limited,
NOT information-limited: 100% of contributors have a present private allele, no decoy donor substitutes —
`measure_noc5_ceiling.py` / `measure_id_ambiguity.py`). So the levers below are model-side (use the
demonstrably-present signal better), each tied to ONE established method — **no self-invented mechanism**.
Every arm is a separate flag on the frozen pe_s3 base (tok8 + periodic σ0.3 + aux, isab++), ablated 3 seeds,
no-regression guard on N1/2/3 ID EM, eval on REAL test + cross-folder OOD.

Base control = `inc2_2b_pe_s3` (already have 3-seed results). Each lever = base + ONE flag.

---

## Group 1 — REPRESENTATION levers (target: lift faint-minor ID at high NOC; the count signal & ID info are present, F6)

### Lever A — `--geno_query` : reference-genotype-conditioned donor queries
- **METHOD (not invented).** DAB-DETR (Liu et al. 2022, ICLR, arXiv 2201.12329) + Conditional-DETR
  (Meng 2021, ICCV 2108.06152) + Anchor-DETR (Wang 2022, AAAI): **a decoder query should be an explicit
  PRIOR/anchor of what to attend to, not a from-scratch learnable vector** — this provably fixes slow
  convergence + small-object localization. Query2Label (Liu 2021, 2107.10834) / ML-Decoder = the label-query
  base we already use. LUPI / privileged info (Vapnik; Lopez-Paz 2015 — doc2 §3/§5). Forensic optimality:
  genotype-conditioned cross-attention ≈ the EuroForMix/STRmix likelihood that weighs each peak by whether it
  matches a donor's reference (Bleka 2016; Taylor-Bright 2013) — the validated gold-standard computation.
- **EXACT CHANGE.** Donor query for donor d = `learnable_query_d + geno_proj(geno_emb_d)`, where `geno_emb_d`
  = masked-mean of d's reference alleles encoded through the SAME `_project_tokens` (locus_embed + input_proj),
  so it lives in the peak space. Reference alleles from the RAW genotype xlsx (allele identity is
  domain-invariant → less C7-overfit than the height-based pgNOC reference). `geno_proj` = Linear(d,d),
  zero-init so the arm starts == base (byte-identity at init).
- **MECHANISM.** Query d, knowing d's expected (locus,allele), attends to d's faint minor peak (it seeks that
  allele, not the tallest peak) → groups d's peaks across loci → lifts minor ID. Maps the F6 finding (info is
  present) onto the proven anchor-query fix for faint/small targets.

### Lever B — `--donor_contrast` : supervised-contrastive peak grouping by contributor
- **METHOD.** Supervised Contrastive Learning (Khosla et al. 2020, NeurIPS, arXiv 2004.11362). "Class" of a
  peak = its true source donor (we already have per-peak `attr` labels, doc2 §5). Decoupled projection (SimCLR
  Chen 2020 / §4b-C valve 2) so the contrastive geometry sits on a discarded head, not the ID readout.
- **EXACT CHANGE.** `proj_peak: H → z_peak (B,N,d_proj)`; SupCon loss over valid peaks grouped by `attr`
  donor id (ignore attr=-1). Aligned WITH the ID objective (both want donor-discriminative features) → lower
  negative-transfer risk than the count-label RNC (F21). Kendall-weighted, guard N1/2/3.
- **MECHANISM.** Peaks of one contributor cluster in representation → the encoder explicitly "groups a
  contributor's peaks", the deconvolution primitive (doc2 §2 contributor-grouping).

### Lever C — `--cls_decoder additive` : additive (non-competing) donor readout  [EXISTS, no new code]
- **METHOD.** `AdditiveDonorDecoder` already in the model (set_transformer.py:304), grounded in C1 (LR/XGB
  per-donor additive lookup beats pooled softmax on minor donor-48) and the forensic log-LR's additive-over-
  loci structure. Softmax cross-attention normalizes across peaks → suppresses minors (documented in code).
- **EXACT CHANGE.** None — run base with `--cls_decoder additive` (+ enriched tok8/periodic, which it was
  never tested with). Ablation arm only.

## Group 2 — NOC-ordinal levers (wire RNC INTO the count representation; the count signal is present — 92% presence-forced, F6)

Root cause RNC was inert for NOC (F21): the count head reads `CardinalityHead(sigmoid(logits_cls).detach())`
— a separate detached path — while RNC shaped a discarded `z_noc_proj`. These arms give the count head its
OWN encoder pool that RNC actually shapes, with a rank-consistent ordinal output.

- **CORN ordinal head (verified).** Shi, Cao, Raschka 2023, *Rank-consistent ordinal regression based on
  conditional probabilities* (arXiv 2111.08851; coral-pytorch). K−1=4 conditional binary tasks
  P(y>k | y≥k); rank-consistent by construction. Replaces the unordered 5-way softmax count head (which
  ignores the 1<2<3<4<5 order — doc2 §4b-A "CORAL/CORN-style ordinal output").
- **RNC (verified, already in code).** Rank-N-Contrast (Zha 2023, NeurIPS, 2210.01189) on the count pool.
- **PCGrad (verified, already in code).** Yu et al. 2020, NeurIPS (2001.06782) gradient surgery.

### Lever V1 — `--noc_ord_head --noc_ord_detach --noc_contrast --rnc_tau 0.3 --rnc_fixed_weight 1.0`
- New `pma_card` pools **H.detach()** → `z_card`; CORN head on z_card; RNC on `proj(z_card)` (SAME pool the
  count head reads — the fix). Final count = ENSEMBLE of the ID-derived `logits_card` and the CORN count.
  ID untouched (detach) → no negative transfer; guard verifies. Safest, designed (§4b-A + valve 2/3).

### Lever V2 — `--noc_ord_head --noc_contrast --noc_contrast_mode pcgrad --rnc_tau 0.3 --rnc_fixed_weight 1.0`
- `pma_card` pools **H (not detached)** → RNC+CORN gradient reaches the encoder → count-aware features;
  PCGrad projects away the ID↔count gradient conflict. Higher ceiling, higher ID risk (guard mandatory).

### Lever V3 — `--noc_ord_head --noc_ord_replace`
- REPLACE the ID-derived count with the CORN-on-encoder count (no ensemble). Tests whether reading the
  encoder directly beats the 45-dim ID-profile bottleneck. Drops the "count bounded by ID" property
  (C3/valve) → expected weaker; included for completeness per request.

---

## Runs — 6 arms × 3 seeds (42/43/44), split across 3 Kaggle machines (~balanced compute)
- **Machine 1:** `inc3_repA_genoq`, `inc3_repB_donorcon`
- **Machine 2:** `inc3_repC_additive`, `inc3_nocV1_ordrnc`
- **Machine 3:** `inc3_nocV2_ordrnc_pcgrad` (PCGrad ~2× backward), `inc3_nocV3_ordreplace`
Each arm writes `<arm>_seed<N>/metrics.json`; aggregate with `aggregate_seeds.py`; eval cross-kit OOD with
`eval_crossfolder.py`. Selection on in-silico DEV; report REAL test + OOD once, 3-seed CIs, guard N1/2/3.

**Discipline:** all behind flags, defaults unchanged (byte-identity for existing arms); each lever = one
published method (above); no conclusion until 3-seed + guard, per [[conclusion-discipline]].
