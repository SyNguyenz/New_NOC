# Draft Sections — Workshop Paper

## 2. Related Work

### 2.1 Number-of-Contributors (NOC) Estimation

The problem of inferring the number of contributors (NOC) from a DNA mixture
has received considerable attention. Early rule-based methods such as MAC
(maximum allele count) are simple but systematically underestimate NOC.
Marciano & Adelman [PACE, 2017] demonstrated that non-linear SVMs with
locus-specific thresholds achieve 98.5% accuracy on 1–4 contributor mixtures
using the Identifiler kit. Taylor & Humphries [deepNoC, 2024] scaled this to
1–10 contributors using a 16-layer deep network trained on 100,000 simulated
GlobalFiler profiles, fine-tuned on PROVEDIt laboratory data, achieving 90%
overall accuracy on 1–5 contributor mixtures. Both works output a single NOC
class; they do not identify which specific individuals are present in a mixture.

Our Set Transformer treats NOC estimation as an auxiliary task (the NOC head).
On the same PROVEDIt GF29cycles test set, our NOC head achieves 99.55% overall
accuracy — exceeding deepNoC's 90% despite NOC being a secondary objective.
This confirms that the mixture embedding z_mix captures contributor structure
beyond what the primary identification task requires.

### 2.2 Probabilistic Genotyping and Likelihood Ratio Methods

Traditional forensic mixture interpretation computes a likelihood ratio (LR)
comparing the probability that a person of interest (POI) contributed to the
mixture versus an unrelated individual. Systems such as EuroForMix and STRmix
implement fully continuous models with explicit parameters for mixture ratio,
stutter, and peak height variance. Slooten [2021] extends this to multi-POI
scenarios, deriving LR jointly over 2^n hypotheses for n persons of interest,
at O(2^n) computational cost. These methods assume POIs are pre-specified and
known; they do not select identities from a large reference database, nor do
they handle the open-set case where some contributors are unknown.

Our task is fundamentally different: given a mixture and a reference database
of 45 known donors, identify which subset contributed, and detect whether any
unknown contributor is present. This framing matches emerging forensic needs
for intelligence-level screening before resource-intensive LR computation.

### 2.3 Mixture Deconvolution

Recent work by Yu et al. [2025] proposes a deep learning approach for mixture
*deconvolution* — inferring genotype probability distributions for unknown
contributors by capturing inter-locus dependencies, then integrating these
distributions with a fully continuous model. Validated on PROVEDIt with
2–4 contributor mixtures, their method improves deconvolution accuracy by up
to 30 percentage points over conventional probabilistic models. Importantly,
their output is a *genotype distribution over the allele space*, not an
identity assignment from a reference database.

These tasks are complementary: deconvolution infers what alleles are present
for unknown contributors; contributor identification assigns database identities
to those alleles. The two approaches could be composed — first deconvolve to
obtain contributor profiles, then match profiles against a database. We address
the second step directly from raw peak heights without an intermediate
deconvolution stage.

### 2.4 Open-Set Recognition

Closed-set classifiers assume all test classes appeared during training. In
forensic contributor identification, this assumption fails whenever a mixture
contains contributors absent from the reference database. Bendale & Boult [2016]
introduced Openmax — fitting Weibull distributions to activation vectors of
correctly classified training samples, then recalibrating logits to reserve
probability mass for an "unknown" class. Yoshihashi et al. [CROSR, 2019]
combine reconstruction-based regularization with classification to improve
latent-space coverage of known classes and thereby tighten rejection of
unknowns.

We adopt a simpler but effective approach: a dedicated reject head trained with
a binary BCE loss on closed-set (label=0) and open-set (label=1) samples,
interleaved during training. The trained reject head achieves AUROC=1.000 on
1,325 closed vs 1,366 open samples. To verify this is task-level separability
rather than a scoring artifact, we evaluate four additional post-hoc methods
on the same frozen model (Table 3): Mahalanobis distance (AUROC 0.9934),
Openmax (0.9914), MSP (0.9103), and Energy score (0.9006). All five methods
confirm that known and unknown mixtures occupy well-separated regions in z_mix
space.

### 2.5 Set-Structured Inputs

Zaheer et al. [Deep Sets, 2017] prove that any permutation-invariant function
on a set can be decomposed as rho(sum_i phi(x_i)), with rho and phi as
universal approximators. While theoretically complete, Deep Sets' mean-pooling
encoder cannot model interactions between elements. Our ablation confirms this
directly: replacing the Set Transformer encoder with a Deep Sets mean-pool
baseline drops Macro F1 from 0.953 to 0.145 (Table 2) — evidence that
contributor identification requires modeling allele co-occurrence patterns
across loci, not just individual token statistics.

Lee et al. [Set Transformer, 2019] address this by parameterizing the encoder
with self-attention (SAB) and Induced Set Attention Blocks (ISAB), achieving
universal approximation for permutation-invariant functions with O(nm)
complexity. We adopt ISAB(d=128, h=4, m=32) × 2 followed by PMA (k=1 seed
vector), which is permutation-invariant by Proposition 1 of Lee et al., and
verified empirically (permutation invariance error = 2.38e-07, floating-point
only).

---

## 4. Experiments

### 4.5 Ablation Study

Table 2 reports ablation results under identical training conditions (60 epochs,
batch size 64, AdamW lr=3e-4).

**Encoder architecture.** Replacing ISAB+PMA with Deep Sets (mean pool over
phi(x_i)) collapses performance to MacroF1=0.145 and Exact Match=0.002 — near
random. This confirms that modeling allele interactions via attention is
critical for contributor identification. A flat 590-dim MLP encoder (with no
set structure) achieves MacroF1=0.973, outperforming the Set Transformer at 60
epochs. This is consistent with the observation that ISAB converges more slowly
than MLPs on fixed-length inputs; with 100 training epochs and patience
scheduling, the full Set Transformer achieves MacroF1=0.981 (Table 1),
substantially exceeding the MLP baseline (0.973).

**Auxiliary tasks.** Removing the NOC head (no_noc) reduces Exact Match from
0.922 to 0.891 (Δ=−3.1%), demonstrating that multi-task supervision with the
NOC count regularizes the mixture embedding. Removing the reject head
(no_reject) slightly improves closed-set metrics (MacroF1 0.962 vs 0.953),
as expected: the reject head's BCE penalty on closed samples (always labeled 0)
competes with the classification objective. The performance difference is small
(+0.9% MacroF1) and the open-set capability is entirely lost without the
reject head.

---

## 5. Limitations

**Dataset scope.** All experiments use a single kit (GlobalFiler, RD14-0003,
50 donors). Yu et al. [2025] demonstrate significant cross-platform degradation
when training and testing on different capillary systems. We observe a 1.1%
Exact Match drop when shifting from Filtered to UnFiltered data (Table 7),
suggesting robustness to peak filtering; cross-kit generalization requires
future work.

**Reference database size.** The 45-donor database is small compared to
operational forensic databases. Scaling to hundreds of donors would require
more training data and likely larger model capacity. The attention mechanism is
architecturally compatible with larger databases, but we have not validated
this.

**Comparison to probabilistic genotyping.** We do not report direct comparison
to EuroForMix or STRmix, as these require per-sample model fitting by a
forensic expert and produce LR values on a different scale. Our system performs
intelligence-level screening (which donors are likely present?) rather than
providing courtroom-ready statistical weight-of-evidence. Composing our
identification output with a subsequent LR computation is a natural extension.

**NOC imbalance.** The PROVEDIt dataset is heavily skewed toward single-source
samples (85% NOC=1), reflecting typical forensic caseloads. Metrics on
complex mixtures (NOC=4,5) are computed on small subsets (33 and 64 samples
respectively) and should be interpreted with caution.

**Open-set protocol.** The 5 held-out donors are selected by random seed
(seed=42) without cherry-picking. AUROC=1.000 reflects that in PROVEDIt,
mixtures containing unknown donors have systematically different peak patterns
from fully-known mixtures — unknown contributors introduce alleles not present
in any known donor's reference profile, creating a strong signal. Robustness
to more subtle unknowns (e.g., previously unseen donors with similar allele
profiles) is an open question.
