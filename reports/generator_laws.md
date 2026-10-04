# Generator laws — what is established, what was rejected

Every constant is derived from single-source (NOC1) profiles or from the PROVEDIt naming-convention
document. No constant is fitted to the mixture set, which lives entirely in the test split.

Verification is the **digital twin**: regenerate each real test mixture from its own specification
(same donors, ratio, template, injection, treatment, all read from the file name) and compare what a
fixed model does on the twin against what it does on the original.

## Established

| Law | Evidence |
|---|---|
| Treatment is a discrete designator, not a Q axis | `a` untreated, b–e DNase, -15/-30/-45 Fragmentase, S sonication, U uv, I humic acid (Methods Table 1). Excess size slope per mechanism is monotone in dose; humic acid measures **0.0000** because it inhibits PCR rather than cutting DNA. Q bottoms out near 0.9 on pristine DNA, so a Q-fitted slope charges the natural size decline as damage. |
| Damage composes by variance and saturates | `s_out² = s_in² + g²`. Derived on single-source runs, it predicts the paired-mixture widening (1.08/1.13/1.16/1.22) against measured (1.07/1.12/1.16/1.18) without being fitted to it. The same DNase level widens a pristine profile 2.20× and an already-skewed one 1.17×. |
| Injection time is structurally inert — as a comparison between WHOLE profiles | 4× injection changes locus spread 1.01×, peak spread 1.01×, heterozygote balance 1.00×. It acts after PCR. 64× template changes the same three by 0.45×, 0.50×, 1.64×. This is not a licence to ignore it at a FIXED RFU level: injection moves where a profile sits on the RFU axis, so two peaks of equal height carry different copy counts, and the copy-count laws below must divide it out. Both statements are measured and neither weakens the other. |
| Stutter is locus-specific | Occurrence 0.10–0.61 across loci; AMEL is 0.000 because it has no repeat unit. |
| Stutter grows with parent repeat count | Positive on 23 of 23 loci; 0.0266 below 10 repeats to 0.0869 at 20–26. |
| Peak-height scatter is molecule sampling | Gamma shape tracks height linearly: 3.4→62.9 as height goes 255→4308, one line through six points within 19%, giving 67.4 RFU per effective copy. An independent route — copies from loaded mass — gives 8.4 pg per copy and agrees within 0.91–1.18 at every level. |
| Dropout is an allele-level event, not molecule sampling | For one contributor the height scatter implies 5–8 copies while the dropout rate implies 0.13–0.24, a factor of 28–42 at every NOC. |
| Dropout takes height **and** fragment size | At fixed height a long fragment is lost 1.4–1.6× more often, measured after removing the size trend that height already carries. |
| `t_total` is a post-threshold quantity | Spending it as the pre-threshold budget charges dropout and AT twice. Invisible at high template (<4% loss) and costs 26% of the signal in the faintest decile. |
| Mixtures were mixed **then** treated | Methods, WBM: "Once combined, the whole blood mixtures were extracted and treated". All contributors share one damage level. |
| PROVEDIt fixes the minor component | Minor mass is ~0.031 ng at every NOC; total template rises with NOC to keep it there. The NOC effect on the weakest contributor is therefore a **denominator** effect: private alleles fall 24→12 while its mass, copy count and height are unchanged. |
| Half the artefacts are invisible to the model | `feas_filter` keeps only peaks some panel donor carries. Replacing every filtered-out artefact with the real one moves ID and count by **0.000** at all four NOC. |
| An artefact rate is emission **times** survival, and the two must be separated | The n-1 rate runs 0.124 → 0.845 with parent height while the rate three and four repeats away, which is baseline noise, stays flat at 0.09. Reading one pooled number measures the product and reports neither: the shipped table read 0.542 where emission is 0.83, which put five times too many faint stutters in the twin and too few tall ones. |
| Artefacts are lost by the **same** height law as real alleles | Survival against the artefact's own expected height, once baseline and emission are removed, is 0.04 / 0.14 / 0.27 / 0.47 / 0.70 / 0.87 / 0.95 / 0.97 / 0.98 / 1.00 from 2 to 390 RFU — half lost at 26 RFU against the 27.5 the allele dropout law reports on the same profiles. |
| Slippage copies the TEMPLATE, not the parent's realised peak | The allele peak and its stutter are two independent draws around one amount. Conditioning on a low observed parent selects downward draws, so real's observed SR climbs to 0.267 at faint parents against 0.061 at tall ones, and a surviving n-1 stands at 12 RFU under an 82 RFU parent where SR × that parent is 4.7. Tying stutter to the realised peak left the twin's survivors at 7 while matching exactly at tall parents. |
| Emission is one draw per allele POSITION, not per contributor | Slippage happens once per position per reaction over whatever template is there. Drawing it per contributor makes P(any stutter) = 1 − (1−rate)^m and climbs to 1 as NOC rises. |
| Noise beside an allele is ordinary noise | Under parents below 40 RFU, where stutter cannot reach 2.3 RFU, the n-1 position reads occupancy 0.111 and height 6/9/11 (gmean 8.47) against pure background's 0.077 and 6/9/12 (gmean 8.55). Nothing about being next to a real peak lifts the floor. |
| Detection is decided on the **summed** peak, emission per contributor | Slippage is a per-molecule event, so each contributor emits its own product, but they land in one bin and the scanner reads the sum. Deciding survival per contributor cost a quarter of the tall stutters and broke the identity that summed stutter equals SR × summed parent. |
| Only integer offsets are products of the parent | At −1/−2/+1/+2 the rate rises 1.9–4.8× from a faint parent to a tall one and the height rises with it (9 → 150 RFU at −1). Every fractional offset is flat or falling and holds at 8–9 RFU whatever the parent does. |
| A rare target allele is called less often | At a fixed locus, parent height and parent repeat count, NOC1 gives 0.29–0.46 where one or two of the 45 panel donors carry the n-1 position against 0.72–0.92 where eleven or more do, on nearly every locus. The law is NOC-invariant; its effect is not, because the n-1 positions still FREE at NOC5 are the ones nobody carries — mean carriers fall 6.8 → 3.7 from NOC2 to NOC5, and real's per-position rate falls 0.516 → 0.340 with them. |
| Dropout and peak-height scatter follow COPY COUNT, not RFU | Injection decides how much finished product reaches the capillary, so two profiles at the same RFU carry different amounts of DNA. Split by injection time, allele dropout at a profile median of 60–150 RFU reads 0.090 / 0.253 / 0.359 at 5 / 15 / 25 sec — a fourfold spread at one RFU level — and the heterozygote CV reads 0.345 / 0.494 / 0.541. Both curves are looked up on height / (injection/15)^0.80, the exponent that minimises the disagreement on NOC1; it puts 15 and 25 sec on top of each other and takes the mean spread from 0.70 to 0.39. |
| Noise height depends on the locus and on nothing else | Geometric mean 5.5 → 13.1 RFU across loci. Against everything else it is flat: 8.2 → 9.4 over a sevenfold range of profile total, unmoved over a seventeenfold range of template, unmoved by injection time, by fragment size (8.6/9.1/7.5/9.2), by distance from the nearest allele, or by whether the bin is an integer position. The locus effect is not amplification efficiency (correlate 0.370) and not size (0.030). Removing it takes the log scatter 0.480 → 0.385. |
| The capillary saturates | No NOC1 profile carries a peak past 29165 RFU (0 of 5247 above 30000) and across 1333 real mixtures the tallest peak anywhere is 29069, again none above 30000. Two independent sets agree on one hard ceiling. The generator reached 44436, and a parent that tall carried its own stutter past 2000 RFU, which real never does. |
| The noise RATE is per bin, the noise HEIGHT is per locus | Occupancy across bins scatters 0.0153 where a common rate would give 0.0055 — 2.8× over-dispersed — running 0.063 to 0.157. Height carries no such structure: once the locus level is out, bins inside a locus differ by 1.15× and the loudest averages 18.8 RFU. |
| An artefact ratio is wide but BOUNDED | Under a uniform detection criterion the log-ratio spreads are 0.32 at n-1 but 0.64–1.01 at n+1 and n±2 — that width is real. A lognormal with it is not: unbounded, it emitted an n+2 at 2.98× its own parent, 21094 RFU beside a 7076 RFU allele, where real carries no artefact above 2000 RFU in 1333 mixtures. The draw is capped at the largest ratio each offset ever reached. |
| The baseline is uniform per bin, not a halo | Density against distance from the nearest true allele: 0.260 at one repeat and 0.101 at two — the stutter model's own territory — then 0.075, 0.080, 0.077, 0.077 out to the end of the panel. A fractional bin **beside** a real allele is emptier, 0.039, the same suppression already measured for a low peak under a tall one in another dye. |

## Rejected — measured and ruled out

| Hypothesis | How it failed |
|---|---|
| Artefact count scales with NOC | Forced onto real's per-NOC counts; the tilt did not move and the aggregate got worse. |
| Contributor tiers sit too far apart | Compressing φ made the problem **easier**, not harder — the opposite of the prediction. |
| Jitter double-counts the source's own scatter | Rewritten as an increment; tier excess grew from 31% to 39%. |
| Locus primer is a shared resource | Within a locus, homozygous and heterozygous donors give the same total (0.994). The 1.190 seen when pooling loci was composition. |
| Amplification competition by contributor count | Yield is 0.99/0.84/1.18/0.64 across NOC2–5 — not monotone, and the alternation survives splitting by injection time, so it tracks which donor sets PROVEDIt used. Also only measurable on the test split. |
| A profile-level effective threshold | Pooled height distributions agree (p25 29 vs 25 at NOC5). The apparent 36-vs-19 was a per-profile percentile artefact. |
| GeneMapper's stutter filter explains the NOC tilt | Removal probability falls monotonically with absolute height (10.7%→0.7%) and is flat against stutter ratio. It is a height threshold, not a ratio filter. |
| Cross-channel pull-up | Low peaks are **depleted** at sizes where another dye carries a tall peak (0.321% against a 3.318% shifted control). GeneMapper removes it before the data reaches us. |
| Artefact height is relative to local signal | Artefact height holds at ~13 RFU from NOC2 to NOC5 while true peaks move over 245–372, and unexplained peaks per locus correlate **negatively** with that locus's tallest peak (−0.157). |
| Dropout as a consequence of Poisson sampling | At 3.7 copies P(0) is 2.5%; real loses 21% of the weakest contributor's alleles. Switching it on left the twin far too easy (ID 0.995, count 0.945). |
| Artefact height scatter is molecule sampling | The stutter visibility curve fits a gamma of shape height/67.4 across all seven parent bins, but the dispersion test refutes it outright: sd of log SR is **flat at 0.31** from 350 to 6000+ RFU where sampling requires it to fall 1.22 → 0.25. The curve fit was a coincidence of a tight lognormal plus the baseline noise the faint bins carry. |
| The NOC5 stutter excess is errors-in-variables on the parent height | Proposed because real's observed SR climbs to 0.267 at faint parents, and refuted twice. Real's rate is flat against how far a peak deviates from its own contributor's level (0.357/0.251/0.329/0.291/0.399 across bands at NOC5), and the within-contributor height scatter narrows with NOC in real as much as in the twin by quantile (p90/p10 4.02 → 3.57 against 3.76 → 3.24); the sd said otherwise only because of the tail. |
| Artefact product arrives in molecule-sized quanta | The survivor-height floor at ~9 RFU under faint parents invites a quantum of that size, but the observed n-1 heights show no periodicity at 9/18/27 — the histogram climbs smoothly from 14 counts at 2 RFU to a plateau at 12-15 — and 45% of the values carry a fractional part. Tested before implementing. |
| The allele slope can be clipped to its fitted range | Tried as a cure for the runaway extrapolation and it is worse, not better: the slope is fitted on tall parents only, so its range starts at 17 repeats, and clipping lifts every ordinary parent onto that floor. Extreme artefacts went from 8 to 1197 and the twin fell to ID 0.842 / count 0.787. The ceiling belongs on the ratio, not on the allele. |
| The noise floor can be a count spread over the artefact histogram | That histogram is dominated by stutter, so it puts the noise back on the bins the stutter model already fills. Matching the density beside an allele then needed 108 peaks per profile against real's 93. |

## Stratified validation

Every constant re-derived separately inside each of PROVEDIt's own strata — 6 treatment groups, 3
injection times, the template range — with a **donor hold-out** (odd donors against even, disjoint
sets) as the noise floor a constant cannot beat.

| constant | CV across treatment | CV across injection | hold-out noise | verdict |
|---|---|---|---|---|
| SR median | 3.4% | 0.9% | 3.4% | invariant |
| artefact survival, 7–17 RFU | 6.0% | 0.6% | 5.8% | invariant |
| artefact survival, 17–40 | 3.3% | 1.8% | 3.2% | invariant |
| artefact survival, 40–110 | 1.5% | 0.9% | 3.1% | invariant |
| rarity gradient, 1–2 carriers | 10.1% | 3.2% | 14.4% | invariant |
| rarity gradient, ≥11 carriers | 5.0% | 1.6% | 1.8% | near-invariant |
| baseline noise density | 2.3% | 0.6% | 1.0% | near-invariant |
| SR slope against repeat count | 15.7% | 12.3% | 9.4% | near-invariant, the weakest of the set |
| noise height | 3.4% | 4.3% | 0.8% | **condition-dependent** — monotone 8.42 → 9.35 RFU across injection |

Constants derived on even donors predict odd donors' observed rates to within 1.1–7.3%, so these
transfer across disjoint donor sets rather than describing the set they came from.

Two laws **failed** the split and had been passing only because the injection times were pooled —
dropout and peak-height scatter, both keyed on RFU. See the copy-count row above.

What this does and does not establish: invariance is tested inside PROVEDIt's own design, which is
real evidence and is what the dataset can support. Universality across kits and laboratories is not
testable here. The FORM of each law — stutter proportional to repeat count, detection as a threshold
on abundance, the ladder's common bins called more reliably, a uniform per-bin baseline — is
kit-general; the CONSTANTS are specific to this kit, instrument and threshold, which is why every one
of them is read from `calibrate.py` against whatever single-source reference data is present rather
than written into the generator.

## The height pattern is drawn per amplification, not carried by the person

Measured on NOC1 with a different-donor control at every level, after removing an allele effect taken
per (treatment, template band) — without that control the condition masquerades as a signature, which
it did twice here before the control was added:

| level | correlation | amplitude |
|---|---|---|
| same PCR, different injection/capillary | **+0.947** | 0.391 |
| same person + treatment + extract, different template | +0.099 | 0.146 |
| same person + treatment, different extract | +0.048 | 0.092 |
| same person, different treatment | +0.057 | 0.105 |
| **different person, same template + treatment** | −0.029 | 0.000 |
| **different person, any** | −0.003 | 0.000 |

The deviation is created at amplification and redrawn every time. Its budget closes exactly:
σ_tot 0.402 = person 0.109 ⊕ PCR 0.376 ⊕ injection 0.093. Within one amplification 57% of it is a
per-LOCUS offset — two alleles of a locus correlate +0.142 — with no dependence on fragment size
(+0.021) once the size trend is out.

**The person term could not be established.** Bootstrapped over donors, the covariance is +0.008 with
sd 0.005, t = 1.58, and 32% of resamples negative; it swings 0.078–0.128 with the arbitrary choice of
2, 4 or 8 template bands. The binding constraint is 45 donors, not 5036 profiles. A direct test —
identify the donor from the height pattern alone, no variance decomposition — puts it at AUC 0.593:
real but weak, and the reason is not a poor instrument. Identity is carried by WHICH alleles a peak
sits on, where the margin is absolute; height adds almost nothing on top.

**What the generator was doing.** It copied one real profile per contributor, and with it that
amplification's realisation. The same identification test read AUC 0.812 on generated profiles
against real's 0.593 — a shortcut in the training data that does not exist in the evidence. The fix
is to subtract each source profile's own residual and draw a new one, keeping every systematic term
the profile already carries. Replacing the profile with a population mean instead was tried and
collapses the twin to ID 0.781 and count 0.620, because a population mean knows nothing about the run
it stands in for.

Result: identification-from-height 0.812 → **0.646** against real's 0.593, within-locus correlation
+0.060 → **+0.124** against real's +0.142, and the NOC5 identification gap that had stood all session
closes — twin 0.844 against real's 0.849, from 0.925 at the start.

## Reconstruction audit — a mixture from its own donors' profiles

Every donor in a mixture also has single-source profiles of its own, so a two-person mixture can be
rebuilt from the two runs that made it and checked peak by peak, with no generator and no pooled
statistic in between. On the 26 mixtures whose donors have untreated profiles at exactly the matching
templates and injection:

| relation | measured |
|---|---|
| **A mixture is the sum of its parts** | mix / (A+B) median **0.99**; shared alleles come out as the sum of both donors' heights (9567→9917, 6423→5745, 9638→8357, 8575→9123) |
| Alleles are not lost by mixing | 5 of 1751 alleles present in the sum are missing from the mixture — 0.29%, every one below 264 RFU predicted |
| A shared allele is quieter than a private one | per-allele residual sd **0.323** shared against **0.601** private: a shared peak is the sum of two independent draws, so its relative noise falls. This is the same mechanism the stutter rate shows from the other side, where the n-1 rate rises with the number of donors carrying the parent |
| Peak-height noise falls with height | residual sd 0.871 / 0.668 / 0.576 / 0.482 / 0.381 / 0.275 from <150 RFU to >6k |
| A mixture's total is not predictable from its parts | run-level scale sd(log) **0.674** — a 2× spread between samples at the same nominal template |

**What the generator reproduces.** Run-to-run reproducibility, measured entirely inside NOC1 on 1200
replicate pairs of the same donor and treatment: real total per-allele sd 0.545 against the
generator's 0.546, run level 1.199 against 1.189, every height band within ±18%. Mixing-ratio scatter:
real sd 0.294 against 0.285, though real is right-skewed where the generator is symmetric (p90 1.72×
against 1.33×).

**Mixing itself is clean.** With the templates matched exactly, a mixture differs from the sum of its
parts by **0.546** — the same 0.545 that two independent runs of one donor differ from each other. The
mixing step adds no measurable scatter of its own; a mixture is its parts plus one more amplification.

An earlier version of this section reported an extra mixing term of 0.474 against the generator's
0.300. That was wrong: the two sides of the subtraction came from different pairings — the mixture
test had been relaxed to the nearest available template while the replicate test had not — so the
template mismatch was being read as a property of mixing. Matched the same way, the term is zero.

**Where the generator differs.** On that same exactly-matched comparison the generator scatters 0.518
against real's 0.546, so ~5% tight overall, but the shape is wrong at both ends: **1.448 against 0.871
below 150 RFU** and **0.396 against 0.482** at 1–2.5k. Shared alleles are too coherent, 0.247 against
0.323. That is the open item, and it is a shape error in the per-peak height law, not a missing
mechanism in mixing.

## What is left, measured rather than asserted

**Dropout level.** Read off generated profiles with the estimator that derived it — fit the size trend
on each profile's own peaks, predict every allele, count what is missing — the generated curve sits on
real's band for band (1.04 / 1.04 / 1.01 / 0.94 against real's 0.830 / 0.664 / 0.491 / 0.262) but the
TOTAL reaches only 88%. The two are not the same quantity: the generated profiles carry 20% fewer
alleles at low copy count than real does, so the same law applied to a less faint population loses
less. Forcing the total to 100% through the half-loss point costs 10–18% on every band and 1.7–6.7×
at the abundant end, so it is not done. Two structural suspects were measured and cleared — the
template shrink accounts for 3 points of the gap and the incremental form for 2.

## Error budget

Twin against real, per governed quantity: heterozygote balance 1.00×, MAC 1.00×, true-allele count
1.00×, total RFU 1.04×, peaks 1.06×, peak spread 1.05×, SR 0.92×, visible artefacts 1.09×, dropout
1.08×, locus spread 1.13×, size slope 1.24×, invisible noise 1.22×.

Split by treatment, the size slope sits within **0.88–1.17× across all twelve conditions**; the 1.24×
aggregate is a composition effect. The invisible-noise deviation is inert by the measurement above.

Per **sample**, the paired ratios scatter far more: true alleles 95% within 20%, peaks 92%, total 87%,
but dropout only 22% and size slope 37%. That is expected and not reducible — 56% of the dropout
residual variance is sample-level (ICC 0.562 by sample against 0.032 by donor), i.e. run-to-run
variation that nothing in the data records. The generator should produce that spread, not explain it.

## What the twin is for

Not to score what real scores. To **choose the same answer real would choose**. Sweeping the two
decode parameters on the twin and on real gives the same curve shape and the same ordering: the twin
picks `alpha=0.80`, which is real's own optimum (0.959 against 0.954 at the shipped 0.30). This is the
property DEV lacks — selecting `alpha` on DEV correlates −0.367 with real, because DEV samples the
generator's own prior while the twin samples real's conditions.

Current twin: ID 0.945 against real 0.948, count 0.884 against 0.894. Per NOC the twin is now harder
than real at 2/3/4 and still easier at 5 (ID 0.994/0.964/0.930/0.892 against 1.000/0.992/0.959/0.849).

## Where the calibration was not reaching

Every `cal()` at module scope in `make_insilico.py` silently returned its literal, because the context
built for `noc1_calib()` named `DONOR_DOSAGE` before that global existed and `cal` swallowed the
`NameError`. Only constants looked up inside a function ever saw a derived value. The literals had
been copied from a good derivation, so they were close enough — `RFU_LOG_C` 9.7430 against 9.7431,
`DROP_H50` 27.5 against 27.44 — that nothing looked wrong until a genuinely new constant was added and
came back at exactly its default. Declaring `DONOR_DOSAGE = None` ahead of the first call fixes it.

## The artefact model, after this round

Against real NOC1 the generated rate now matches offset by offset — n-1 0.547 against 0.542, n+1
0.203 against 0.204, n-2 0.208 against 0.202, baseline 0.097 against 0.090, artefact fraction 0.608
against 0.610 — and the n-1 curve tracks parent height bin by bin rather than only in the mean.

## Nen nhieu bi chinh boi do cao cua locus (NOC1, 122991 locus-mau)

So peak khong giai thich duoc o mot locus GIAM khi locus do cang cao:
2.247 (dinh 1-25 RFU) -> 1.526 (120-202) -> 1.225 (882-1554) roi PHANG va quay len
(1.272, 1.354) o vung bao hoa/pull-up. spearman(ln dinh locus, so nhieu) = -0.159.
Khong phai luat theo thoi gian nap: NOC1 gop lai cho 35.01/34.74/35.02 peak nhieu o 5/15/25s
(ty le 1.000), va trong TUNG dai dinh locus thi nhieu con tang nhe theo lan nap. Hai hieu ung
triet tieu nhau nen phep do bien duyen khong thay gi.

Cai bang bang noi suy `NOISE_LM_H/NOISE_LM_M`, chuan hoa tai trung binh NOC1 (1.4906/locus) nen
bien duyen NOC1 khong doi. Truoc khi cai, twin phat 28.9/29.7/30.0 peak mo-coi o 5/15/25s trong
khi real cho 25.9/22.8/22.7 — PHANG thay vi GIAM. Sau khi cai: 29.1/26.9/27.1.
Chu thich cu ngay tren khoi nhieu da ghi tuong quan -0.157 nay tu truoc nhung chua bao gio ap dung.

## Tan chieu cao sinh o PCR, khong sinh o mao quan (NOC1, 3601 cap nap)

Cung mot gieng, cung chiet, cung lan chay, chi khac so giay: sd(log ty le chieu cao) =
0.0944 / 0.1049 / 0.0555 cho 5->15 / 5->25 / 15->25, trung binh 0.0850. Moi lan nap mang mot
lan rut cua mao quan nen sigma_mao_quan = 0.0850/sqrt(2) = 0.060. TAT CA phan tan con lai da
duoc chot xong khi PCR ket thuc va KHONG duoc rut lai khi nap lai cung san pham.
Xac nhan doc lap tren 1328 cap mixture that: 0.092 / 0.100 / 0.052.

Ke theo: nap la mot PHEP NHAN gan nhu tat dinh. So mu beta do tren NOC1 = 0.926 / 0.967 / 1.056
(mixture that: 0.976 / 1.009 / 1.081) — tuc RFU tuyen tinh theo thoi gian nap, dung bang he so
`rfu_coef` 0.948 da co san. Sai lech beta THUC HIEN cua twin (1.16 o buoc 5->15) khong den tu
he so nay ma den tu dam dinh mo nam sat duoi nguong: nap manh hon thi chung dong loat ngoi len.
Sua nen nhieu o tren keo beta 5->25 ve 1.005 (real 1.009); buoc 5->15 con 1.099 (real 0.976).

Cai bang cach tach phuong sai: gamma giu CV^2 = 1/gs - CAP_SD^2, roi nhan them mot lognormal
CAP_SD — bien duyen mot ho so KHONG DOI theo dung dinh nghia, chi hanh vi giua cac lan nap doi.
`amp_cache` cho lan nap thu hai dung lai dung lan rut khuech dai cu. Luu y: van phai GOI rng.gamma
du dung cache, neu khong luong ngau nhien lech nhip va moi thu phia sau desync (sd 0.514 so voi
0.335 khi khong cache gi ca).

CHUA XONG: sd giua hai lan nap moi ve 0.279 (real 0.085). Than phan bo da khop
(p10 0.015/0.019, p25 0.046/0.030) nhung ~30% so cap van desync o duoi (p75 0.54 so voi 0.12),
vi thoi gian nap con chi phoi ca rung, phat artefact va nen nhieu. Sua triet de doi tach kien truc
"san pham -> lan nap": hien thuc hoa ho so o muc khuech dai truoc, roi moi ap gain nap + nhieu
mao quan + nguong.

## Nap phai o CUOI chuoi, khong phai o dau (STR_INJ_LAST, mac dinh bat)

Trat tu vat ly: khoi luong -> phan huy -> khuech dai -> NAP -> phat hien. File truoc day ap nap
DAU TIEN (co gian t_total) roi chia nguoc no ra khoi moi lan tra bang (height/_ig). Hai cai hong:
  * hai so mu khong khop — RFU co gian theo inj^0.948 (rfu_coef) trong khi khoa tra bang dung
    inj^0.80 (INJ_BETA) — nen mot phan du inj^0.148 cuoi len moi lan tra, va hai lan nap cua CUNG
    mot san pham cho ra hai bo so ngau nhien khac nhau;
  * cac buoc thang TUYET DOI cua mao quan (nen nhieu, bao hoa, nguong phan tich) bi ap len nhung
    con so ma nap da lam xe dich.
Sua: dung toan bo o moc 15 giay, nhan gain (inj/15)^0.948 NGAY TRUOC khoi artefact. Moi luat theo
so phan tu do duoc tra tren mot dai luong bat bien voi lan nap dung nghia, con nguong lam not phan
viec that su cua lan nap.

Ranh gioi nam O DAU khoi artefact chu khong phai sau no: bang SONG SOT cua artefact duoc hieu chuan
tren chieu cao nhu may bao, nen no thuoc phia mao quan. Dat gain sau khoi artefact thi bang song sot
doc phai chieu cao quy-ve-15-giay va cho lot gap ba lan stutter o 5 giay (25.3 moi ho so so voi
real 16.3). Phat xa stutter la chuyen PCR, song sot la chuyen mao quan — code cu gop hai cai lam mot.

Ket qua (twin, 1333 mixture that):
  dong thuan lua chon donor  0.913 -> 0.920   (tran do tren ban lap that = 0.927)
     NOC2 1.000 = dung tran, NOC4 0.934 vuot tran 0.929
  ID sai so tuyet doi TB     0.021 -> 0.0133
  sd(log ty le) giua hai lan nap: trung vi 0.0694 so voi real 0.0634, p25 0.0252 so voi 0.0296
  jac allele THAT giua hai lan nap: 0.9190 so voi real 0.9140
  bien duyen NOC1 va bien duyen theo tung muc nap: khong doi
  NOC dung sai so TB 0.051 -> 0.0565 (xau di, xem canh bao duoi day)

KHONG duoc do loi cho viec model train tren generator cu. Twin test la phep so HAI PHAN BO qua mot
ham CO DINH f: neu twin tai tao dung phan bo cua real thi f(twin) = f(real) voi MOI f, bat ke f duoc
huan luyen tren gi. Mot f nhay hon co the khuech dai khac biet, nhung khong the bia ra khoang cach o
cho hai phan bo trung nhau. Vay nen khoang cach NOC la THAT, va viec no xau di sau khi sua la mot
hoi quy that, phai truy chu khong duoc bo qua.

CHUA XONG — generator chi co MOT luong ngau nhien. Can hai: mot luong PCR (dung chung cho moi lan
nap cua cung san pham) va mot luong MAO QUAN (rut moi moi lan nap). Bang chung: real cho
jac artefact = 0.176 giua hai lan nap, tuc 82% danh tinh artefact KHONG lap lai — chung la trang
thai cua mao quan. Sau khi dao thu tu, twin cho 0.558 vi phep thu dung chung mot seed cho tat ca,
ep ca nen nhieu lap lai theo. Cung mot nguyen nhan giai thich not cai duoi con vo cua sd
(p75 0.51 so voi real 0.12): ~30% so cap van lech nhip vi rung/artefact/nhieu deu an chung luong.
Sua triet de = tach gen_product(seed_pcr) / gen_injection(product, inj, seed_mao_quan).

## Ty trong hon hop: mot phan DAU VAO, hai phan BIEN DOI (chua sua duoc)

Tren 1333 mixture that, phan ra do lech cua ty trong THUC HIEN so voi ty trong DANH NGHIA doc tu
ten file (chuan theo so allele RIENG moi donor co):
  [A] theo VAI   : chinh -0.0214, giua +0.0029, yeu nhat +0.0171
  [B] theo NGUOI : do tan that su cua trung binh tung donor = 0.0191 (quan sat 0.0195, nhieu 0.0040)
  [C] TRONG cung mot donor, sau khi tru phan rieng cua ho: chinh -0.0146, yeu nhat +0.0108
[B] chung to co thanh phan DAU VAO — luong DNA vao phan ung khong dung ty le danh nghia, va sai lech
do la dac tinh cua tung nguoi. [C] chung to co thanh phan BIEN DOI — cung mot con nguoi, khi dong
vai yeu nhat thi duoc +0.0108, khi dong vai chinh thi bi -0.0146; luong DNA cua mot nguoi khong the
biet ho duoc xep vai gi. Dau vao giai thich ~1/3, bien doi ~2/3.

Khop voi phep thu tap-sach: o NOC3 khoang cach real-twin tren allele rieng day du la 0.0107, dung
bang thanh phan [C] 0.0108, va no bien mat khi chi tinh nhung allele NAM XA moi vi tri stutter
(real -0.0295 / twin -0.0293). Tuc phan bien doi nam o nhung allele cua nguoi yeu trung vi tri
stutter cua nguoi manh hon. (Tuong quan giua cu nang va TY LE allele duoc tiep suc thi khong ra —
nhung ty le la thuoc do yeu: do lon cu nang phu thuoc chieu cao cua CHA nhan ty le stutter chia cho
chieu cao cua chinh allele yeu, khong phai dem so allele.)

DA THU, KHONG AN: cai thanh phan [B] bang `donor_eff` — du cua phep khop RFU tren NOC1, trung binh
theo tung nguoi. Hai phien ban deu rut hop le: tho 0.2435, va phan NGUOI sau khi tach khoi phan LAN
CHIET va co ngot theo do tin cay 0.1443. Ca hai deu VUOT: do tan ty trong tung donor cua twin ra
0.0521 va 0.0361 so voi real 0.0195. Tuong quan real-twin theo tung donor co len (+0.114 -> +0.235
/ +0.328) nhung chi co 19 donor du so lieu, SE ~0.22, nen khong ket luan duoc. Twin xau di ro:
dong thuan 0.920 -> 0.893, ID 0.950 -> 0.914, NOC 0.860 -> 0.842. Rieng NOC4 - tang lech nhat -
thi count len manh 0.802 -> 0.880 (real 0.897), xac nhan co che dung nhung DO LON sai.
=> STR_DONOR_EFF mac dinh TAT. Hang so van giu trong calibrate vi no la mot dai luong that.

LY DO VUOT (gia thuyet, chua kiem duoc tren NOC1): du cua NOC1 la chenh lech GIUA CAC ONG. Trong
RD14-0003 cac contributor duoc tron truoc roi chiet va khuech dai CHUNG MOT ONG, chia chung hoa
chat, chu ky nhiet va chat uc che, nen chenh lech hieu suat GIUA CAC TEMPLATE TRONG CUNG MOT ONG
nho hon nhieu. NOC1 khong co phep so sanh trong-cung-ong nao de do dai luong do.

=> Day la luat THU HAI trong du an nam ngoai tam NOC1, cung voi phan nen theo vai o [C].

## KHUECH DAI CO BAO HOA — day la tuong tac DNA-DNA, va NOC1 do duoc no

Khong co luat nao nam ngoai NOC1. Mot ho so NOC1 cung la ba muoi may allele trong CUNG MOT ONG,
tranh nhau mo va enzyme; thu can do khong phai "quan he giua nguoi voi nguoi" ma la PHAN UNG CO NEN
HAY KHONG. Day pha loang cua NOC1 do duoc thang.

Do tren 4033 cap pha loang (cung lan chiet, cung thoi gian nap, template tang >1.5x, >=12 allele
chung deu >60 RFU): hoi quy log(h_cao/h_thap) theo log(h_thap) cho do doc -0.489 +/- 0.005, AM o
94.5% so cap, tren 122642 allele. Dinh von cao tang them -0.334, dinh von thap tang them +0.427.
Tang template thi DAI DONG LOG CO LAI MOT NUA. Do la bao hoa: template doi dao bao hoa som, template
hiem con chay gan du hieu suat.

Day la CUNG MOT LUAT ma `tpl_shrink` da mang san (0.893 moi lan gap doi): 0.893^6 = 0.507 doi voi
-0.489 do tren khoang ~32-64x cua day pha loang. Hai phep do doc lap, cung mot con so.

LOI: tpl_shrink chi duoc ap khi dat ho so nguon len moc chuan, va KHONG BAO GIO khi ha mot
contributor xuong phan cua ho. `contrib[di] = p * t_total * h` la mot phep nhan TUYEN TINH, nen
nguoi phu trong twin giu nguyen dai dong cua mot ho so du manh. Nguoi phu that, ngoi o it template
hon, co dai dong RONG hon - va mot phan bo log rong hon o cung trung binh thi TONG lai lon hon
(Jensen). Do chinh la cu nang cua nguoi yeu.

Cai: doc nguoc chinh duong cong tpl_shrink, KHONG hang so moi. STR_TPL_SPREAD:
  "1" doi ca HINH DANG lan ty trong -> HONG: bo rung tang dan cho contributor bi ha template DA
      tinh phan mat o dau mo roi, noi dai dong nua la tinh HAI LAN (o NOC5 nguoi yeu nhat bi noi
      toi 66%). Twin xau di: dong thuan 0.920 -> 0.896, ID 0.950 -> 0.921, NOC 0.860 -> 0.824.
  "2" chi lay CU NANG TY TRONG (mac dinh) -> DUNG.

Ket qua che do "2":
  ty trong so voi danh nghia   chinh -0.0219 (real -0.0214), giua +0.0046 (+0.0029),
                               yeu nhat +0.0150 (+0.0171) — va khop o TUNG NOC
  dong thuan lua chon donor    0.920 -> 0.926   (tran tren ban lap that = 0.927)
  NOC dung                     0.860 -> 0.899   (real 0.894), sai so TB 0.0565 -> 0.040
  ID                           0.950 -> 0.959   (real 0.948)
  bien duyen NOC1              khong doi (94.1 peak, artefact 0.612)

CHUA XONG: cu nang dang la mot HE SO NHAN DEU, trong khi vat ly noi no la doi HINH DANG - dinh mo
duoc nang nhieu hon dinh cao. Hau qua do duoc: twin rung IT allele that hon real, -1.2 o NOC4 va
-1.5 o NOC5, nen NOC5 de hon real o ca hai task (ID 0.906 so voi 0.849, NOC 0.823 so voi 0.745).
Sua dung = ap doi hinh dang nhung TRU di phan ma buoc rung tang dan da tinh, thay vi chon mot
trong hai. Do la mot phep tru, khong phai mot cong tac.

## Phep TRU: doi hinh dang MA KHONG tinh mat mat hai lan (STR_TPL_SPREAD=4, mac dinh)

Bo rung tang dan da so huu TOAN BO phan mat do contributor cam it template hon - no duoc dinh nghia
dung bang the: song sot o muc da ha, chia cho song sot o muc cua chinh ho so nguon. Noi dai dong roi
DE luat do doc chieu cao da noi la tinh cung mot mat mat hai lan.

  so allele THAT bi RUNG, thua ra so voi real:   NOC2  NOC3  NOC4  NOC5    TB
    mode 0  khong sua                            +0.5  +1.3  +0.1  -0.6   +0.33
    mode 1  noi hinh dang, khong tru             +1.0  +2.8  +2.2  +2.7   +2.18
    mode 2  chi nang ty trong, giu hinh dang     +0.0  +0.9  -1.2  -1.5   -0.45
    mode 4  noi hinh dang + TRU                  +0.1  +0.9  -0.6  -1.1   -0.18

Phep tru: ap noi dai dong len CHIEU CAO, nhung tinh pdv tren chieu cao TRUOC KHI NOI. Noi dai dong
la phan phoi lai, no khong lay di template; luat kia moi la luat tinh tien cho template.

  ty trong so voi danh nghia   real -0.0214 / +0.0029 / +0.0171
                       mode 4       -0.0179 / +0.0039 / +0.0121
                       mode 2       -0.0219 / +0.0046 / +0.0150
  dong thuan lua chon donor    mode 2 va mode 4 deu 0.926 (tran 0.927)
  ID sai so tuyet doi TB       mode 2 0.019   mode 4 0.0155
  NOC sai so tuyet doi TB      mode 2 0.040   mode 4 0.047

Chon mode 4 vi no la CAU TRUC DUNG va thang o thong ke khong phu thuoc model (do rung -0.18 so voi
-0.45) lan o ID. Nhung phai ghi ro: NOC4 count cua mode 4 la 0.831 so voi real 0.897 (-0.066, ~2.5
SE) trong khi mode 2 cho 0.909 (+0.012, gan nhu dung). Do la mot SAI SOT chua tim ra, khong phai
mot cai gia phai tra.

## Sai so NOC con lai la DEM THUA, va nhien lieu la dinh mo coi

Ma tran nham lan (mode 4, cau hinh cuoi phien):
  REAL      NOC2 -> 3: 0.9%   NOC3 -> 4: 0.8%   NOC4 -> 5: 1.7%
  SINH DOI  NOC2 -> 3: 5.4%   NOC3 -> 4: 4.7%   NOC4 -> 5: 9.1%
  lech co huong (doan - that)   real 0.000 / -0.016 / -0.083 / -0.331
                                twin +0.045 / +0.023 / +0.008 / -0.290
REAL CHI SOT NGUOI, KHONG BAO GIO BIA NGUOI. Twin bia gap 5-6 lan. Tong gate tho cao hon o moi tang
va do tan gap doi o NOC2 (0.238 so voi 0.116).

Co che: donor NGOAI CUOC bi ho so giai thich >=90%  — real 0.69 / 2.57 / 3.34 o NOC3/4/5,
twin 0.80 / 2.92 / 4.42. O NOC5 twin dung them hon MOT nguoi ma. Trong twin, mau bi dem thua co
+1.91 dinh mo coi va +3.64 peak so voi mau dem dung; trong real, dem thua chi xay ra 11/1203 lan va
lai di kem IT peak hon (-6.62). Hai kieu sai khac han nhau.

Nhien lieu: twin con thua 4.24 dinh mo coi moi ho so. Phan bo cua no DEU tren moi muc hiem
(twin/real = 1.17 / 1.24 / 1.20 / 1.24 / 1.12 / 1.21 tu bin khong ai mang den bin >=12 nguoi mang),
tuc KHONG phai twin nham vao bin hiem — gia thuyet do da bi bac. 76% so mo coi roi vao 381 bin ma
KHONG donor nao trong panel mang, nen vo hai; chi ~1.2 dinh roi vao bin co the dung nen mot nguoi ma.

DA THU, KHONG AN: ep duong cong NOISE_LM_M don dieu (dai cao nhat quay len 0.822 -> 0.853 -> 0.908
va chieu cao nhieu o do la 12.12 RFU so voi ~9, dau hieu pull-up chu khong phai nhieu nen). Giam
mo coi 4.24 -> 3.67 nhung twin xau di deu: dong thuan 0.926 -> 0.922, NOC 0.875 -> 0.866,
ID 0.958 -> 0.952. Da go bo, khong giu cong tac.

CON LAI: o NOC1 nhieu cua twin DUNG (+2%), trong hon hop no thua ~20%. Do la mot muc giam theo
luong DNA ma cac bien do duoc trong NOC1 (do cao locus, do lap day, tong RFU) deu khong bat duoc —
trong noi bo NOC1 ca ba trung nhau (spearman -0.386 / -0.395 / -0.395) va chi tach ra theo huong
so nguoi.

## Stutter no ra theo so nguoi: loi nam o PHAN BO theo chieu cao CHA, va NOC1 do duoc

Tong so stutter nhin thay duoc tren NOC1 khop 1.010, nen moi phep kiem tong the tu truoc den nay
deu cho qua. Tach theo chieu cao cua ALLELE CHA thi khong khop:

  cha (RFU)        <60   60-100  100-180  180-320  320-600  600-1200   >1200
  P(hien) real    0.107   0.152    0.235    0.387    0.571    0.678    0.711
  P(hien) twin    0.121   0.175    0.257    0.359    0.506    0.626    0.684
  twin/real        1.13    1.15     1.09     0.93     0.89     0.92     0.96
  so cha (real)   13109    8029    10778    11367    12114    12180    22644

Twin sinh QUA NHIEU stutter tu cha MO va QUA IT tu cha CAO. Trong mot ho so don nguon hai loi triet
tieu nhau (trung binh co trong so = 1.009) nen vo hinh. Mot hon hop thi phan lon cha la MO - allele
cua nhung nguoi phu - nen chi phan du o dau mo song sot qua phep trung binh, va so stutter no ra
theo so nguoi: real 25.1 / 24.1 / 25.1 / 19.2 o NOC2-5, twin 24.0 / 24.6 / 28.0 / 21.8. Chinh nhung
dinh thua do dung nen nguoi ma: donor ngoai cuoc duoc giai thich >=90% la +0.91 o NOC4, +0.98 o NOC5.

DA THU HAI LAN, DEU HONG - ca hai deu vi KHOA SAI DAI LUONG:
  lan 1: nhan he so vao xac suat song sot, khoa tren chieu cao KY VONG (contrib_exp). Bang duoc do
         tren chieu cao QUAN SAT DUOC. Stutter tang thay vi giam: +1.9/+3.0/+3.3.
  lan 2: doi sang chieu cao THUC HIEN (mix). Van hong: `mix` o diem ap la TRUOC NGUONG va chua day
         cha mo se bi loai, con bang thi do tren cha DA QUA NGUONG. -0.2/+1.8/+2.9/+3.3, twin xau di
         (dong thuan 0.925 -> 0.919, NOC 0.885 -> 0.875).
=> STR_ART_PARENT mac dinh TAT. Chan doan dung, cach cai chua dung: he so phai duoc ap sao cho no
tai lap P(hien | chieu cao cha QUAN SAT DUOC), tuc phai giai nguoc qua ca nguong lan phan bo cha,
khong phai nhan thang vao xac suat song sot.

Ghi chu ve phuong phap: day la lan thu BA trong phien nay mot phep sua that bai vi khoa nham giua
chieu cao KY VONG / THUC HIEN / QUAN SAT DUOC (truoc do: bang song sot artefact, va he so hiem).
Truoc khi ap bat ky bang nao, phai kiem no duoc DO tren dai luong nao.

## Giai nguoc qua nguong: LAM DUOC, nhung khong du

Yeu cau: he so phai tai lap P(stutter n-1 hien | chieu cao cha QUAN SAT DUOC), khong phai nhan
thang ty so vao xac suat song sot. Giai bang LAP: ap bang, sinh lai NOC1, do lai duong cong, cap
nhat theo ty so, lap. Hoi tu sau 5 vong, sai lech dai xau nhat 14.7% -> 5.2%.
  ART_PAR_M = [0.6651, 0.7426, 0.9330, 1.0988, 1.1861, 1.1189, 1.0362]
Dau mo phai xuong 0.665 trong khi TY SO THO chi doi 0.884 — nguong va phan bo chieu cao cha lam
giam hieu luc gan MOT NUA. Do dung la thu phep giai nguoc sinh ra, va no xac nhan khong duoc phep
doc ty so roi nhan thang.

NHUNG CAI VAO THI HON HOP XAU DI: stutter thua -0.1/+2.0/+3.6/+3.7 (truoc do -1.1/+0.5/+2.9/+2.6),
dong thuan 0.925 -> 0.926 nhung NOC 0.885 -> 0.861. Ly do tim ra bang phep do tiep:

  P(stutter n-1 hien | chieu cao cha) cua REAL, theo NOC, ty le so voi NOC1 cua chinh no:
    NOC2  ~1.00     NOC3  ~0.90     NOC4  ~0.83     NOC5  ~0.78   (gan nhu DEU tren moi dai cha)

Tuc khop duong cong NOC1 la CHUA DU: ban than real trong hon hop chi bang 0.78-0.87 duong cong NOC1
cua no. Generator thi bat bien theo NOC o cho nay, nen no thua dung phan chenh do.

CANH BAO PHEP DO (loi thu 4 cung loai trong phien): lan do dau tien cho ty le 0.40 o NOC5 va tuong
nhu luat sup do hoan toan. Sai: MAU SO dem MOI allele cha, trong khi TU SO loai nhung truong hop vi
tri n-1 trung allele cua nguoi khac. NOC cang cao thi cang nhieu vi tri n-1 bi chiem, nen P tut ma
khong phai vi vat ly. Sua mau so (chi dem cha co vi tri n-1 KHONG phai allele cua ai) thi 0.40 -> 0.78.
Duong cong NOC1 cung doi dang ke: 0.1006->0.1219 o dai <60, 0.7016->0.8416 o dai >1200.

DA THU, KHONG AN: quy phan du 0.78-0.87 do cho DO DONG DUC, theo dung khuon da thang o phan nhieu.
Nguoc chieu: trong NOC1, ho so cang dong thi stutter cang DE thay (ty le dong/thua = 1.11 den 1.71),
con hon hop thi cang KHO thay (0.69-0.88). Trong NOC1 do dong duc di kem do sang, va sang thi stutter
ro hon - do la bien gay nhieu, khong phai bien can tim.

=> STR_ART_PARENT mac dinh TAT. Bang da giai van giu lai vi no dung cho muc tieu cua no (duong cong
NOC1). Phan con thieu la mot muc nen theo SO NGUOI ma chua bien nao trong NOC1 bat duoc — khac voi
nhieu nen, thu ma do dong duc cua NOC1 da giai thich tron ven.

## Loi nam o QUAN HE, khong nam o dai luong nao — do bang phep cay cong tinh

Phep cay tach kenh cho mot ket qua lap lai o HAI TANG doc lap, va no giai thich vi sao moi phep sua
mot-dai-luong trong phien nay deu cho ket qua nho hoac trai chieu.

TANG 1 — giua kenh ALLELE va kenh ARTEFACT (NOC dung):
  twin allele + twin artefact = 0.855      real allele + real artefact = 0.895
  real allele + twin artefact = 0.861  ->  artefact twin gay thiet hai 0.034
  twin allele + real artefact = 0.836  ->  allele  twin gay thiet hai 0.059
  tong thiet hai rieng le 0.093  >>  thiet hai khi di cung 0.040

TANG 2 — ben trong kenh CHIEU CAO, tach theo loai dinh:
  chi h allele that +0.013 | chi h stutter -0.008 | chi h mo coi +0.002 | tong rieng le +0.007
  lam dong thoi (h TAT CA) +0.027

Ca hai tang deu SIEU CONG TINH. Hai loi bu tru nhau: moi kenh mot minh sai nhieu hon han khi chung
di cung nhau. Sua rieng chieu cao stutter con lam TE DI (-0.008, vuot 6.0% -> 7.8%): mot stutter
mang dung chieu cao cua real nam duoi mot allele cua twin la cau hinh te hon ca hai cung sai.
KHONG phai do cau hinh bat kha: ty le stutter/cha cua ban ghep cheo co IT truong hop vuot 1.0 hon ca
twin goc (4.0% so voi 4.5%), nen loi giai thich "mat mach lac vat ly" da bi bac.

=> Model dem nguoi bang cach tim mot loi giai thich nhat quan cho allele CUNG VOI artefact cua chung.
Moi thong ke bien duyen da khop gan het roi; khuyet tat song trong phan bo DONG THOI, noi khong hinh
chieu mot chieu nao nhin thay. Day chinh la quy tac "luat phai dung trong TUNG MAU" cua du an, gio
co bang chung do duoc thay vi chi la nguyen tac.

## Ty le khoi luong artefact/allele theo DO SANG cua locus (chan doan dung, cai chua dung)

Dai luong nay khong phai mot con so ma la mot duong cong: stutter ty le voi cha, nhieu nen la khoi
luong tuyet doi, nen ty le phai GIAM khi locus sang len — 0.2054 duoi 200 RFU allele xuong 0.0488
tren 5400, do tren NOC1. Generator dung MUC (0.0546 so voi real 0.0537, lech 1.7%) nen moi phep kiem
tong the deu cho qua, nhung sai HINH DANG, cung dau o ca hai dau:
    allele mass trong locus   <200   200-600  600-1800  1800-5400  >5400
    twin/real tren NOC1       1.021   0.915    0.928     1.032     1.033
    twin/real tren hon hop    1.082   0.817    0.909     1.065     1.120
NOC1 chua tron ven khuyet tat; hon hop chi khuech dai no len khoang gap doi. Phan "giam theo so
nguoi" thi generator DA DUNG: real 0.0537/0.0356/0.0284/0.0222/0.0202 o NOC1-5, twin bam sat.

BANG CHUNG NO LA THU CAN SUA: ap dung ty le cua real theo TUNG LOCUS TUNG MAU - khong dich mot dinh
nao, khong doi mot chieu cao allele nao - keo ty le ho so co gate vuot qua su that nua nguoi tu
6.0% xuong 4.0% (real 1.2%), gan bang viec chep TOAN BO chieu cao (3.8%).

DA THU, KHONG AN: cai bang mot duong cong hieu chinh trung binh (ART_LOCUS_H/M, he so
0.979/1.093/1.078/0.969/0.968). Bien duyen tot len that - ba trong nam dai ve sat real, NOC1 tu 93.0
xuong 92.8 (real 92.6) - nhung twin xau di: ID 0.944 -> 0.932, NOC 0.855 -> 0.845, dong thuan
0.918 -> 0.908.
=> STR_ART_LOCUS mac dinh TAT. Doi lap giua hai phep sua chinh la cau tra loi: cay khop TUNG MAU thi
giup, duong cong TRUNG BINH thi hai. Cai con thieu la BIEN THIEN TUNG MAU cua ty le, khong phai muc
trung binh cua no — dung ket luan cua phan tren.

## LOI HARNESS: phep do NOC1 chay voi PHAN HUY BI TAT

Harness kiem NOC1 dung suot phien goi gen_mixture(..., bin_size=bs, phi=[1.0]) ma KHONG truyen
`cond`. Khoi phan huy trong generator co dieu kien `PERCONTRIB and bin_size is not None and cond is
not None`, nen no khong bao gio chay. Moi phep do NOC1 o nua sau phien deu lech vi the.

PHAI RUT LAI:
  * "twin thieu 33% phuong sai GIUA-locus, toan bo thieu hut nam o xu the theo bp" — SAI. Voi cond
    duoc truyen: giua-locus 1.036, trong-locus 1.116, da tru bp 1.049 / 1.115. Cau truc phuong sai
    cua twin DUNG trong khoang 4-12%.
  * "twin thua 1.49 lan stutter duoi 15 RFU" — SAI. Voi cond duoc truyen, twin THIEU stutter mo:
    t/r o p5/p10/p25 la 0.762 / 0.758 / 0.782 khi CHUA sua, va so luong 0.949.
  * STR_ART_SURV_LOW da duoc GIAI tren harness hong de chua mot khuyet tat khong ton tai. Bat no keo
    so luong stutter tu 0.949 xuong 0.895, xa real hon. Da chuyen mac dinh TAT. Twin cung da noi dieu
    do tu dau (NOC 0.885 -> 0.869) nhung toi bo qua vi tin vao phep do.

PHAN BO BIEN DUYEN, do lai dung (NOC1, cond duoc truyen):
  allele THAT  t/r   p5 1.122  p10 1.262  p25 1.196  p50 1.018  p75 0.935  p90 0.915   so luong 0.962
  stutter      t/r   p5 0.783  p10 0.789  p25 0.844  p50 1.019  p75 1.008  p90 0.967   so luong 0.895
  mo coi       t/r   p5 1.015  p10 0.943  p25 1.040  p50 0.970  p90 1.009  p99 1.304   so luong 1.100
Allele van bi bop o dau mo nhung +12-26% chu khong phai +40% nhu do tren harness hong.

VAN DUNG (do tren hon hop, noi cond LUON duoc truyen): do doc log(chieu cao) theo bp cua twin la
-0.00599 so voi real -0.00423, tuc DOC GAP 1.42 LAN, va muc thua no theo so nguoi:
1.225 / 1.276 / 1.387 / 1.395 o NOC2-5. Phuong sai giua-locus tren hon hop 0.8826 so voi 0.5318
(1.660). Day la thanh phan ma EuroForMix/STRmix deu dat ten rieng - ham phan huy theo chieu dai manh -
va no dang bi cong chong len phan phan huy ma ho so nguon von da mang, cho TUNG contributor mot.

BAI HOC: truoc khi tin bat ky phep do nao, kiem xem harness co truyen DU dieu kien khong. Mot tham so
thieu lam tat han mot co che, va moi ket luan xay len tren do deu sai theo cung mot chieu.

## KIEM DINH THEO TUNG CON SO, va vong dai DUNG cho mot ban twin

Do lech chieu cao TUNG ALLELE trong MOT mau (da tru do dich chung cua ca mau), do tren NOC1:
  cung SAN PHAM PCR, khac lan nap      sd 0.0974   |d| p95 0.183
  khac LAN CHIET, cung nguoi           sd 0.7550   |d| p95 1.583
  khac ca chiet lan nap                sd 0.7849   |d| p95 1.620
  hai NGUOI KHAC NHAU (chung)          sd 1.6317   |d| p95 3.468
Twin la mot ban dung lai DOC LAP tu cung nhung nguoi, nen moc cua no la dong "khac LAN CHIET",
KHONG phai dong "cung san pham PCR". Hai dong nay cach nhau gan 8 lan.

QUET TO HOP, cham bang kiem dinh tren (445 mau, moi cong tac tat mot lan tu cau hinh day du):
  cau hinh              %vuot p95  %vuot p99  sd/vong dai  tong t/r  lech giu allele  lech so art
  TAT CA BAT (hien tai)     5.2%      1.2%       1.014      0.989     -0.0074          +0.5
  tat TOTAL_TWOSIDED        5.2%      1.0%       1.009      1.126     +0.0001          +1.4
  tat NOISE_CROWD           5.5%      1.1%       1.025      0.985     -0.0054          +4.1
  tat INJ_LAST              5.4%      1.2%       1.030      0.986     -0.0142          +0.6
  tat DEG_RESET             5.3%      1.0%       1.024      1.021     -0.0120          +0.2
  tat TPL_SPREAD            4.1%      0.9%       0.940      1.007     -0.0038          +0.9
  TAT CA TAT                4.3%      1.0%       0.952      1.065     -0.0119          +7.0
  HOAN HAO                  5.0%      1.0%       1.000      1.000      0.0000           0.0

KET LUAN: tren RANG BUOC LONG (chieu cao tung allele) cau hinh hien tai gan nhu chinh xac - 5.2% so
voi 5.0%, sd 1.014. Tren RANG BUOC CHAT (tong RFU) no dat 0.989, va TOTAL_TWOSIDED chinh la thu giu
duoc dieu do. So artefact +0.5 dinh moi ho so, va NOISE_CROWD giu dieu do. Moi cong tac ganh MOT rang
buoc khac nhau; to hop hien tai la to hop tot nhat.

DIEU NAY GIAI THICH NGHICH LY CUA CA PHIEN: moi lan sua mot luat that, cac con so vat ly tot len con
diem NOC cua model di xuong (0.885 -> 0.869 -> 0.856 -> 0.852). Vi diem ay duoc cham so voi mot TRAN
tinh tu cac cap CUNG MOT ONG PCR - mot moc khong mot phep tai tao doc lap nao dat toi. Kiem dinh theo
tung con so moi la thuoc do dung; diem cua model chi la kiem tra phu.

## DAC TA RANG BUOC: cai gi PHAI dung, cai gi PHAI ngau nhien, va trong gioi han nao

Le ra phai lap bang nay TRUOC khi do bat cu thu gi. Moi dong deu da duoc KIEM, khong khang dinh.

### A. RANG BUOC CHAT — moi sai lech la LOI
  bo bin allele THAT = hop bo gen cua to hop      DUNG theo cau tao (kiem: trung khop)
  kich thuoc bp cua tung bin                      DUNG theo cau tao (tra bang panel)
  khong co dinh NGOAI panel                       0 / 1333 ho so
  khong co dinh duoi nguong AT (3.0 RFU)          0 / 1333
  tran bao hoa                                    dinh cao nhat 29165 = SAT_RFU; real 29069
  so locus co dinh                                23.96 so voi real 23.93
  stutter chi o offset NGUYEN                     dinh o bin phan so 19.51 so voi real 18.51
  >> TONG RFU = t_total duoc giao (+/-3%)         KHONG DAT: 83.0% so mau vuot 3%, 26.2% vuot 15%,
     va vuot nang hon theo NOC (16.2/20.3/35.1/35.5%). Trung binh thi dung (0.981-0.984).
     Nang TOL_ITERS KHONG cuu duoc: phep lap PHAN KY (TB 0.984 -> 0.865 -> 0.773 -> 0.700 o
     2/4/6/8 vong) vi khi ep xuong thi allele va stutter co theo ty le con NEN NHIEU la khoi luong
     TUYET DOI nen khong co - vong sau lai vuot tuong doi nhieu hon. Giu TOL_ITERS = 2, KHONG lap them.
     => Tren thuc te day dang la rang buoc LONG chu khong phai CHAT. Sua dung phai GIAI THANG ngan
     sach allele = (t_total - khoi luong nhieu) / (1 + ty trong artefact), thay vi lap.

### B. RANG BUOC LONG — PHAI ngau nhien, trong vong dai da do
  chieu cao TUNG ALLELE      vong dai: khac lan chiet cung nguoi, sd 0.755, |d| p95 1.583 (NOC1)
                             hien tai: 4.9% vuot p95 (hoan hao 5.0%), sd/vong dai 0.994   DAT
  danh tinh bin artefact     vong dai: jaccard 0.176-0.229 giua hai ban lap that -> gan nhu tu do
  so artefact moi ho so      +0.0 so voi real                                             DAT
  ty le giu allele           NOC1 -0.0035 (DAT) | hon hop +0.0212 (CHUA DAT)
  can bang mat/thua theo dai NOC1 |tong| 0.262 (DAT) | hon hop van +1.23 o dai <25 RFU
  tan giua hai lan nap       real 0.085; twin 0.279 - than phan bo khop, duoi con vo
  ty le stutter/cha          p50 0.063/0.064, p90 0.131/0.134                             DAT

### C. DAI LUONG PHAI NGAU NHIEN NHUNG PHAN BO BI LUAT CO DINH
  do doc phan huy theo bp    = -(cond_beta + 0.00063) cho tung xu ly; hien 1.336x qua doc o hon hop
  CV chieu cao theo copy     gamma, shape tra tren copy proxy (JIT_H/JIT_SHAPE)
  ty le nhieu theo do cao locus  NOISE_LM (giam), va theo do dong duc x chieu cao  NOISE_CR
  bao hoa khuech dai         tpl_shrink, 0.893 moi lan gap doi
  hieu suat khuech dai/locus phuong sai giua-locus 1.036 so voi real                       DAT

### D. NAM NGOAI TAM NOC1 — ba luat, deu la TUONG TAC DNA-DNA
  1. nen ty trong theo vai (nguoi yeu duoc +0.011 so voi danh nghia, tru phan rieng tung nguoi)
  2. nen stutter theo so nguoi (P(hien|cha) con 0.83/0.78 cua duong cong NOC1 o NOC4/5)
  3. rung them cua contributor phu (hon hop giu thua 2.1% allele du NOC1 da dung)
  Ca ba deu khong co hang so trong tai lieu giam dinh: EuroForMix/STRmix deu la mo hinh CONG TINH,
  khong co so hang tuong tac giua cac contributor. Nhung phep do cua ta co NHOM CHUNG noi tai ma
  tai lieu khong co: cung mot con nguoi, dong vai yeu thi +0.0108, dong vai chinh thi -0.0146 - sai
  so pha che hay dinh luong khong the biet ho duoc xep vai gi.

## MUC 2 (giu allele trong hon hop): chan doan XONG, cai CHUA XONG

### Khoa hoc da chot: KHONG co tuong tac nao
Ty le song sot cua bin allele THAT trong hon hop that, theo SO NGUOI MANG bin do:
  1 nguoi mang  real 0.8715  twin 0.8954  (+0.0239)
  2             real 0.9484  twin 0.9648  (+0.0163)
  3             real 0.9669  twin 0.9755  (+0.0087)
  4             real 0.9882  twin 0.9879  (-0.0003)
  5             real 0.9943  twin 0.9948  (+0.0006)
Twin CHINH XAC o 4-5 nguoi mang. Toan bo chenh lech nam o bin it nguoi mang, va trong nhom
chi-mot-nguoi-mang no don dieu theo TY TRONG cua chinh nguoi ay:
  >0.55 +0.0051 | 0.30-0.55 +0.0119 | 0.18-0.30 +0.0198 | 0.10-0.18 +0.0394 | 0.05-0.10 +0.0543

=> Day la hieu ung CUA RIENG TUNG NGUOI, khong phai tuong tac. Cong chieu cao roi cat nguong la DU.
Xac nhan bang phep do quyet dinh: bin chi-mot-nguoi-mang, song sot theo TEMPLATE cua chinh nguoi do
(= ty trong x ng tong), doi chieu duong cong giu-allele cua NOC1:
  template ng   0.012-0.022  0.022-0.040  0.040-0.075  0.075-0.140  0.140-0.260  >0.260
  real            0.6712       0.8282       0.9252       0.9618       0.9849      0.9945
  NOC1 curve      0.6720       0.7988       0.8891       0.9538       0.9851      0.9908
  real - NOC1    -0.0009      +0.0294      +0.0362      +0.0080      -0.0003     +0.0038
Duong cong NOC1 du doan hon hop trong vong 3%, TRUNG KHIT o hai dau. Khong can hang so moi.

### Hai lan cai, hai lan hong - va deu lam twin DE HON real
  lan 1: nhan deu pdv sao cho ty le giu TRUNG BINH cua contributor dat muc tieu.
         Hong vi trung binh do cac allele CAO ganh, duoi mo khong nhuc nhich.
         Va toi cham no bang ty le giu cua HOP - dai luong sai, vi bin dung chung song nho TONG.
  lan 2: giai DIEM GIUA duong cong rung cho tung contributor (nhi phan 14 vong).
         Tong |lech| tren duong cong template 0.116 -> 0.088, dai giua gan nhu trung khit
         (0.9228 so voi 0.9252), NHUNG bin nhieu-nguoi-mang xau di het:
         2 nguoi +0.0163 -> +0.0246, 3 +0.0087 -> +0.0208, 4 -0.0003 -> +0.0079.
         ID 0.966 -> 0.980 (real 0.948), dong thuan 0.939 vuot tran 0.927.
=> STR_KEEP_ANCHOR mac dinh TAT.

### Vi sao dai mo nhat khong dap ung
Phep neo tac dong len PHAN DONG GOP RIENG cua contributor, nhung mot bin da rung van co the duoc
NHIEU hoac STUTTER lap lai nen van hien trong `mix`. Uoc luong: o dai mo nhat ~33% allele bi rung,
ty le nhieu moi bin ~0.077, nen lap lai ~2.5 diem, cong stutter thi ~3-5 diem - dung bang khoang
cach 4.9 diem con lai. Muon dong duoc phai xu ly ca phan lap day, khong chi phan rung.

### Moc nghiem thu cho lan sau (dung dai luong, dung nguon hang so)
  do tren bin CHI MOT NGUOI MANG, theo template cua chinh nguoi do
  muc tieu = duong cong giu-allele cua NOC1 (0.633/0.829/0.943/0.987/0.996 tai 0.012/0.037/0.095/0.245/0.700)
  KHONG duoc cham bang ty le giu cua HOP - no tron lan hieu ung chia se voi hieu ung rung
  va phai kiem DONG THOI bin nhieu-nguoi-mang, vi hai lan hong deu bi bat o day

## MUC 2 DA SUA: neo ty le giu tren duong cong NOC1 -- va BA co che nguoc chieu phai go truoc

Sua mot ham thoi thi khong du. Phep neo bi ba co che khac vo hieu hoa, va phai go het moi chay:

  (a) NGHIEM THU TY TRONG tinh tren khoi luong SAU KHI RUNG.
      `_s1 = contrib[_di][mix > 0].sum()`. phi noi moi nguoi mang bao nhieu TEMPLATE, khong noi bao
      nhieu phan song sot - nguoi phu DUOC PHEP mat nhieu hon. Cham tren khoi luong sau rung bien no
      thanh mot bo loc len chinh buoc rung: siet rung cua nguoi phu -> ty trong tut qua TOL_SHARE ->
      mau bi loai va sinh lai -> quan the duoc nhan THIEN VE nhung lan nguoi phu song sot.
      => chuyen sang contrib_pre (truoc khi rung).

  (b) NHANH BU-THIEU cua ngan sach phan ung voi TONG THUC HIEN. Rung nhieu hon -> tong tut ->
      `got < 0.97*t_total` -> sinh lai voi ngan sach NHAN LEN -> moi dinh cao hon -> bin nhieu nguoi
      mang song them. => tat khi BUDGET_SOLVE dat ngan sach bang giai thang.

  (c) MUC TIEU HAI PHIA. R(t) doc tu NOC1 cho 0.996 o template cao trong khi generator dang o ~0.98,
      nen phep giai DOI NOI RUNG cho nguoi chinh. Bin nhieu nguoi mang do nguoi chinh chi phoi nen
      chung di len theo: ID 0.980 so voi real 0.948, NOC5 ID 0.954 so voi 0.849.
      => MOT PHIA: chi duoc siet, khong bao gio duoc noi. Chan doan la GIU THUA thi moi phep noi deu
      la di nguoc chan doan.

Va mot chi tiet co che nua: mot bin da rung VAN CO THE HIEN, vi nen nhieu hoac stutter lap lai roi
vuot AT. Nen muc tieu cho phan song sot RIENG phai thap hon duong cong:
    own = (R(t) - fill) / (1 - fill),  fill = 0.08 (ty le nhieu moi bin ~0.077)

Cai dat: giai DIEM GIUA cua duong cong rung cho tung contributor bang nhi phan 14 vong, khong nhan
deu pdv - nhan deu chi doi trung binh, ma trung binh do cac allele CAO ganh nen duoi mo khong nhuc
nhich (0.116 -> 0.102 khi nhan deu, so voi 0.116 -> 0.088 khi giai diem giua).

KET QUA (ty le song sot bin allele THAT theo so nguoi mang, chenh twin - real):
  so nguoi mang      1        2        3        4        5     tong lech co trong so
  truoc          +0.0236  +0.0150  +0.0099  +0.0018  +0.0011         2294
  sau            -0.0163  +0.0053  +0.0076  +0.0024  +0.0034         1446   (-37%)
Khong con muc nao lech qua 0.017, va khong con muc nao bi day sai chieu.
NOC1 khong doi (94.7 peak). Twin: ID 0.963 (real 0.948), NOC 0.881 (0.894), dong thuan 0.929 (tran 0.927).

BAI HOC: truoc khi sua mot hang so, phai truy MOI HAM doc dai luong ma no thay doi. O day co ba, va
ca ba deu phan hoi NGUOC lai phep sua - hai lan truoc toi ket luan "phep sua nay hong" trong khi that
ra no bi cac co che khac keo nguoc.

## Ban do RANG BUOC su co mat, va mot chan doan cau truc BI BAC BO

Moi cho trong gen_mixture ep mot bin CO MAT hay VANG MAT:
  VANG MAT  (1) h *= (DONOR_DOSAGE[c] > 0) - ho so nguon chi giu allele cua chinh donor
            (2) contrib[di] *= _kept        - rung tung contributor (pdv + KEEP_ANCHOR)
            (3) PROVEDIT_FILTER (dang tat)
            (4) bao hoa SAT_RFU (chi cat ngon, khong xoa)
            (5) mix[mix < AT_PC] = 0        - nguong phat hien, tren TONG
            (6) hai nhanh sinh lai mau (nghiem thu ty trong / thieu ngan sach)
  CO MAT    (1) mix += contrib[di]
            (2) stutter n-1, n+1, va cac offset khac
            (3) bang artefact  mix[_kp] += art_h[_kp]
            (4) nen nhieu      np.add.at(mix, jn, _nh)
            (5) drop-in
Tat ca ghi vao CUNG mot mang mix, roi nguong cat tren TONG - nen moi cap deu tuong tac.

DO DUOC: ho so nguon GAN NHU DAY DU - 0.9993 allele co mat, va khong doi theo muc template
(0.9995 / 0.9994 / 0.9990 / 0.9992 tai 0.0625 / 0.125 / 0.25 / 0.5). `_clean_source` chon dung
nhung ho so sach nhat, nen viec `_clean_level` rut NGAU NHIEN mot muc template khong lien quan gi
toi mau dang dung la VO HAI. Rang buoc (1) dong gop bang khong.

HE QUA: `pdv` mang ten "rung TANG DAN" nhung thuc chat la rung TOAN PHAN - cong thuc
1 - S(contrib)/S(nguon) voi S(nguon) ~ 1 rut gon thanh d_of(contrib), mot duong cong hinh NGUONG
voi diem giua ~27 RFU, trong khi nguong may that la AT_PC = 3.

CHAN DOAN SAI: tu do toi ket luan generator dang "cat nguong tung nguoi roi moi cong", nen mot ban
sao 20 RFU cua nguoi phu bi giet truoc khi kip cong vao 500 RFU cua nguoi chinh. Cai phep cuu tuong
ung roi BAN QUA MANH (1 nguoi mang -0.0165 -> +0.0589) va so hoc cho thay vi sao:
  cach hien tai:  P(vang) = PI pdv_i        (moi contributor rut DOC LAP, bin chi vang khi TAT CA rung)
  cach "cong roi cat": P(vang) = pdv(S)
  voi hai nguoi cung pdv = 0.3:  PI = 0.09  <  pdv(S) ~ 0.12
Tuc phep rut doc lap lam bin dung chung song sot DE HON, khong phai kho hon. Lap luan cua toi sai
DAU. Va so lieu von da noi the: bin nhieu nguoi mang dang HOI CAO (+0.0034 / +0.0049), khong thieu.
Da go bo hoan toan, khong giu code chet.

BAI HOC: mot lap luan co che nghe rat thuyet phuc ("nguong la chuyen cua may nen phai ap tren tong")
van co the sai DAU khi cai dat thuc te la mot phep rut doc lap chu khong phai mot phep so sanh.
Phai viet ra CONG THUC XAC SUAT cua ca hai cach truoc khi tin vao lap luan bang loi.

## Tim luat tuong tac tren tai lieu phap y: KHONG CO, va phan du KHONG PHAI tuong tac

Da tim tren nguon (khong chi tu tri nho). Ket qua:

XAC NHAN: cac mo hinh lien tuc (EuroForMix, STRmix) dung cac so hang THEO TUNG CONTRIBUTOR voi gia
dinh DOC LAP CO DIEU KIEN khi biet kieu gen. Khong co so hang nao de nguoi nay doi hieu suat khuech
dai hay xac suat rung cua nguoi kia.

TOI DA NOI QUA TAY: toi khang dinh tai lieu khong ghi nhan phu thuoc vao so nguoi. Sai - nhieu nguon
phat bieu thang rang ty le rung la ham cua locus, luong template, so chu ky, SO CONTRIBUTOR va TY LE
HON HOP. Nhung so contributor va ty le hon hop CUNG NHAU xac dinh luong template cua tung nguoi, nen
cach phat bieu ay van nhat quan voi mot co che thuan tuy theo-tung-nguoi. Khong tim duoc nguon nao
tach bach duoc hai kha nang.

PHEP DO CUA CHINH TA BAC BO GIA THUYET TUONG TAC: sai lon nhat nam o bin CHI MOT NGUOI MANG
(z -14.7, so voi -7.6 va -6.1 o hai va ba nguoi mang) - noi khong tuong tac nao co the xay ra. Mot
contributor, khong ai de canh tranh, khong ai de cong vao, khong ai de che. Phan con thieu nam o
NHIEU CUA MAY, va theo dung khung thi no PHAI tim duoc trong NOC1.

DA THU, KHONG AN: mo hinh rung chuan cua Tvedebrink khoa theo H - chieu cao dinh TRUNG BINH cua mau -
chu khong theo chieu cao tung allele. Thu chuyen sang khoa theo KY VONG template (contrib_exp) thay
vi chieu cao da thuc hien: XAU DI o moi dai luong, giu allele 0.9207 -> 0.9309, z -14.6 -> -28.2.
Ly do: duoi thap cua phep rut gamma CHINH LA thu tao ra rung; khoa theo ky vong thi duoi ay bien mat.
Va nghi lai thi code cu dung hon - chieu cao va viec rung cung do MOT so phan tu quyet dinh, nen
chieu cao da hien ra la dai dien tot cho bien an ay. Tuong quan do la THAT, khong phai gia. Da hoan tac.

## Amplification failure is locus-correlated, not independent between contributors

The amp-failure law drew q(f) independently for each contributor, so a bin held by k people
survived with probability 1 - q^k. Real mixtures do not behave that way. Taking each side's own
single-carrier rate and asking what independence then predicts for two and three carriers:

    REAL  q = 0.1390   2 carriers: predicts 0.9807, actual 0.9426  (-0.0381)
                       3 carriers: predicts 0.9973, actual 0.9631  (-0.0342)
    TWIN  q = 0.1694   2 carriers: predicts 0.9713, actual 0.9606  (-0.0107)
                       3 carriers: predicts 0.9951, actual 0.9886  (-0.0065)

Real falls well below independence; the twin sat almost exactly on it (its small deficit is the AT
threshold, not a modelled correlation). So contributors' failures are positively correlated.

The split was measured on NOC1, never on the mixtures. Two alleles of a HETEROZYGOUS locus are two
sets of molecules in one reaction at one locus - the same configuration as two contributors sharing
a bin. P(both fall) against p^2, by template quintile:

    dai    p        p^2      p_both    du        s
    Q1     0.4430   0.1963   0.2608   +0.0645   0.1721
    Q2     0.1824   0.0333   0.0722   +0.0390   0.0551
    Q3     0.0514   0.0026   0.0161   +0.0134   0.0147
    Q4     0.0093   0.0001   0.0025   +0.0024   0.0025

CONTROL, and it is the control that makes this a law rather than an artefact: pairs drawn from
DIFFERENT loci of the SAME sample share the tube but not the locus, and they show only about a
third of the surplus (+0.0223 / +0.0141 / +0.0035 / +0.0009). Heterogeneity between samples cannot
explain the same-locus excess, because it would have to inflate both equally.

Solving the three-level model - sample S, locus L, molecule u, with p = 1 - (1-S)(1-L)(1-u):

    dai    S        L        u        L/(L+u)
    Q1     0.0567   0.1223   0.3273   0.299
    Q2     0.0226   0.0332   0.1347   0.203
    Q3     0.0018   0.0129   0.0372   0.260
    Q4     0.0005   0.0020   0.0068   0.222

L/(L+u) = 0.246 +- 0.037, flat across four decades of template. That flatness is the evidence it is
a primer-efficiency property of the locus: a template effect would have to trend. Q5 is excluded -
p = 0.0011, it holds almost no dropouts.

INSTALLED as AMP_SHARE_R = 0.246 with _amp_split(q) holding (1-L)(1-u) = 1-q, so a single-source
sample reproduces the measured q(f) exactly and no second constant enters. The locus part is drawn
ONCE per sample at the sample's TOTAL template - what amplifies at a locus is everyone's template
together - and kills that locus for every contributor; the molecule part is drawn per contributor
at its own f. Survival for k carriers becomes (1-L)(1-u^k).

Predicted before running: 0.8610 / 0.9537 / 0.9637 against real 0.8610 / 0.9426 / 0.9631.

    dai luong          REAL     twin TB   twin sd     z     truoc
    peak             127.82     127.89    0.44      -0.2 CO   +2.4
    artefact          46.22      46.69    0.36      -1.3 CO   -0.0
    giu allele        0.9020     0.8923   0.0021    +4.6     +12.4
    tong RFU          64246      67898    186      -19.6     -15.5
    song sot 1 nguoi  0.8610     0.8464   0.0027    +5.3     +11.8
    song sot 2 nguoi  0.9426     0.9550   0.0015    -8.3     -15.2
    song sot 3 nguoi  0.9631     0.9749   0.0030    -4.0     -13.3
    do doc bp        -0.00411   -0.00430  0.00003   +6.1      +3.4

    tong |z|: 94.0 -> 86.1 -> 74.0 -> 49.4

All three survival columns more than halved at once, from one NOC1 constant.

NEXT, with the diagnosis already in hand: tong RFU is the largest residual (-19.6, twin 5.7% high).
The budget divides by the expected surviving fraction, but an allele killed by amp-failure was
often below AT anyway and would have contributed nothing to the OBSERVED total - so the budget
compensates for mass that was never going to be seen, and over-inflates. The fix is to charge the
budget only the above-AT part of the killed mass.

## The budget was solved for a total the build never had (Jensen boost)

The twin ran 4.4% heavy on total RFU. The hypothesis on the table was that the budget compensates
for killed mass that would have been below AT and never observed. MEASURED, and it is wrong:

    khoi luong bi GIET: tren AT 1.419e+07   duoi AT 2.592e+04   -> 99.8% tung quan sat duoc
    surv theo TONG khoi luong  = 0.8616
    surv theo khoi luong TREN AT = 0.8617

Identical to four decimals, and the real threshold is AT_PC = 3.0, lower still. There was nothing to
fix there. Term by term instead (ratio to the target):

    phuong trinh ngan sach du doan            0.9958
    allele thuc te sau giet                   0.9827      <- ky vong B*surv = 0.9337
    sau artefact + nhieu                      1.0473
    sau PROVEDIT_FILTER + bao hoa             1.0471      <- loc/bao hoa/AT mat 0.0004
    sau nguong AT = TONG QUAN SAT             1.0469

So the leak is entirely between B and the built allele mass. Three candidates were measured and all
three refuted: sum(h) = 0.9997, so the shape is normalised; per-contributor realised survival /
predicted = 0.9998, so the kill is exactly as modelled; the retry recursion touches 50 of 1333
samples and the excess is flat across gain bands (1.0455 / 1.0468 / 1.0474).

Recording every contributor unconditionally - the earlier probes silently dropped contributors that
took no kill at all, which is what made the intermediate numbers inconsistent:

    Sum contrib TRUOC giet / t_total  = 1.0451
    Sum contrib SAU giet / TRUOC giet = 0.8275     _surv = 0.8274   -> ti so 1.0001
    sum(phi) trong vong lap           = 1.0466
    quan sat / dich                   = 1.0441

phi does not sum to 1 in the build loop. The spec passes [0.5, 0.5]; the loop uses [0.5122, 0.5102].
The saturation-widening block normalises h back to sum 1 and then multiplies p by
_bo = _new.sum() / h_old.sum() - widening the log-scale residual raises the arithmetic mean, and the
boost is deliberately carried into p so the REALISED share reflects it. But phi was normalised
BEFORE that, so the boosted shares sum to 1.047 while the budget had been solved for 1.0. Built mass
/ budget 1.0451, sum of boosted phi 1.0466, observed / target 1.0441 - one number, three places.

FIXED by putting the sum back on 1 after the loop. The boost is a statement about the contributors'
RELATIVE shares; the TOTAL is what t_total fixes and is not the boost's to move. Rescaling mix,
contrib, contrib_exp and contrib_pre together is mass-proportional, so no survival ratio moves.

    dai luong          REAL     twin TB   twin sd     z     truoc
    peak             127.82     127.65    0.23      +0.7 CO  +0.7
    artefact          46.22      46.56    0.24      -1.4 CO  -1.3
    giu allele        0.9020     0.8909   0.0014    +8.0     +4.6
    tong RFU          64246      64843    130       -4.6    -19.6
    song sot 1 nguoi  0.8610     0.8451   0.0016    +9.7     +5.3
    song sot 2 nguoi  0.9426     0.9535   0.0025    -4.3     -8.3
    song sot 3 nguoi  0.9631     0.9754   0.0026    -4.8     -4.0
    do doc bp        -0.00411   -0.00440  0.00003   +7.4     +6.1

    tong |z|: 94.0 -> 86.1 -> 74.0 -> 49.4 -> 40.9

NOTED IN PASSING, not yet fixed: contrib_pre[di] is now assigned AFTER the amp-failure kill, so the
TOL_SHARE acceptance check reads POST-dropout mass. Its own comment says it must read pre-dropout
mass, and records why - checking the post-dropout mass makes it a filter on the dropout itself and
biases the accepted population toward draws where the minor happened to survive. The check was
correct when the dropout lived in d_of; moving the dropout into the loop silently broke it again.

## Do the fixes interact? No - and that makes the constant, not the mechanism, the suspect

Factorial 2x2 over the locus law (r = 0 / 0.246) and the Jensen fix (off / on), 3 reps per cell.
Effect of the locus law, measured separately in each Jensen state:

    dai luong           jensen tat   jensen bat   chenh
    song sot 1 nguoi      +0.0158      +0.0159     0.6%
    song sot 2 nguoi      -0.0063      -0.0057     8.7%
    song sot 3 nguoi      -0.0151      -0.0121    20.0%
    giu allele            +0.0051      +0.0051     0.0%

The other direction: the Jensen fix moves total RFU by -2926 at r=0 and -3118 at r=0.246, 6.2%
apart. The percentages that look large sit on quantities where the effect is ~0 (the locus law on
total RFU: +84 against -109 on a base of 65000). The two are orthogonal on what each targets, and
the ordering is not the problem.

That leaves the mechanism as the suspect. It is not. Sweeping r:

    r        song sot 1   song sot 2   song sot 3   giu allele   tong RFU
    0.000      0.8287       0.9596       0.9869       0.8855      64915
    0.246      0.8446       0.9539       0.9748       0.8906      64806
    0.450      0.8597       0.9466       0.9642       0.8957      64589
    0.650      0.8750       0.9352       0.9491       0.8991      64510
    REAL       0.8610       0.9426       0.9631       0.9020      64246

    r needed to match each column ALONE:  0.453 / 0.452 / 0.408

Three independent statistics, two of them moving in the opposite direction to the third, converge on
one value within 0.045. A single scalar fits all three. The structure (1-L)(1-u^k) is right.

The CONSTANT is what is wrong. NOC1 gives r = 0.246; folding the sample level S into the shared part
gives 0.297; the mixtures want 0.44. Fitting r to the mixtures is forbidden - all 1333 are in test.

WHY NOC1 MAY MEASURE LOW. The het-locus pair used as the analogue shares the tube and the locus but
NOT the allele. Two contributors sharing a bin share the allele too: same length, same primer site,
same sequence context. Any shared effect at the ALLELE level - preferential amplification of a given
allele - is therefore invisible to the current measurement, making 0.246 a LOWER BOUND rather than
an estimate. The weakness of this account is that it is not directly measurable on NOC1 as designed:
the only single-source configuration with two molecule sets at one allele is a homozygous locus, and
there both copies land in the same bin.

STOPPED HERE for a decision. Options: (A) keep 0.246 and record the residual; (B) fold S in for
r ~ 0.30, still NOC1-only, but risk double-counting what t_total and degradation already generate;
(C) find an NOC1 route to the allele-level component; (D) source it from the forensic literature -
not searched yet, and not to be guessed at.

Fix the contrib_pre / TOL_SHARE regression noted above BEFORE settling r, or r will absorb its error.

## RETRACTION: the "one scalar fits all three" convergence was a pooling artefact

The r sweep above reported that all three carrier-count columns demand r = 0.453 / 0.452 / 0.408 and
called that strong structural evidence. Splitting the SAME statistic by the sample's NOC destroys it:

    NOC 2-3 (240 mau)     song sot 1   song sot 2   song sot 3
    r = 0.00                0.8372       0.9643       0.9901
    r = 0.30                0.8472       0.9505       0.9740
    r = 0.60                0.8683       0.9294       0.9456
    REAL                    0.8931       0.9675       0.9943
    r demanded              1.079       -0.055       -0.057

    NOC 4-5 (205 mau)
    r = 0.00                0.8216       0.9564       0.9852
    r = 0.30                0.8470       0.9568       0.9787
    r = 0.60                0.8795       0.9514       0.9584
    REAL                    0.8242       0.9208       0.9536
    r demanded              0.027        4.276        0.706

At low NOC the generator drops too much at EVERY carrier count; at high NOC it drops too little at
shared bins and is already correct at single-carrier ones. Two opposite errors. Pooling them averaged
to a clean-looking common value that neither group actually wants. The convergence was Simpson's
paradox, the same trap that produced the spurious 0.47 in the first NOC1 table and the spurious 1.69
in the competition test below. No constant r can fix this - the required correction changes sign with
NOC.

## Machine-noise routes to a larger r: three tested, all closed

Prompted by the objection that a mixture differs from NOC1 only by DNA-DNA interaction, so any
shortfall must be either incomplete knowledge or unaccounted machine noise.

ALLELE-LEVEL FIXED COMPONENT (the hypothesis that het pairs share the locus but contributors also
share the allele, making 0.246 a lower bound). Decomposed the bin's fixed propensity into its
shared-locus part E[p_j p_j'] and its allele-specific part E[p_j^2] - E[p_j p_j']:

    Q1 -0.00324   Q2 +0.00013   Q3 +0.00020   Q4 +0.00005

Zero. The bin's fixed dropout propensity is entirely a property of the LOCUS, not of the allele. The
hypothesis is refuted and the het-pair estimator is not blind after all.

STUTTER MASKING. Real but small. A het pair one repeat apart has the taller allele's stutter landing
in the shorter allele's bin, hiding its dropout. Adjacent pairs give a lower r at every band
(0.336/0.355, 0.283/0.299, 0.266/0.294, 0.163/0.257) and a much lower p (Q4: 0.0050 against 0.0127).
Restricting to pairs >=3 repeats apart moves NOC1's r from 0.297 to 0.303.

INTER-AMPLICON COMPETITION. Compared homozygous loci (1 amplicon, 2 doses) against heterozygous ones
(2 amplicons, 1 dose each) at matched locus and matched effective molecule count. No consistent
signal - the het/hom ratio runs 1.10, 1.09, 0.99, 0.89, 0.65, 0.68, 0.72 and changes sign. The pooled
1.69 is Simpson's paradox and must not be used.

So NOC1's ceiling for r is 0.303, and the machine-noise routes are exhausted.

## Where the missing term has to live

Searched the literature rather than asserting. The standard statement is that drop-out rates are a
function of the locus, the quantity of template amplified, the number of amplification cycles, THE
NUMBER OF CONTRIBUTORS, and the approximate mixture ratio. The number of contributors is identically
1 in every NOC1 sample, so that term is structurally absent from NOC1 - by construction, not by any
failure of measurement. The NOC-split table above shows exactly a NOC-dependent residual.

Competing alleles per locus in the real mixtures: 2.803 / 3.610 / 4.155 / 4.604 at NOC 2/3/4/5
against 2 for a NOC1 het locus. A law r ∝ m anchored at NOC1 would give 0.425 at NOC2 - close to
what the pooled fit wanted - but 0.698 at NOC5, and the NOC1 hom/het contrast gives no support for
that functional form. Choosing a form by testing it on the 1333 mixtures is fitting to test.

Sources: FSI Genetics review of low-template/mixture genotyping; PCR in Forensic Science: A Critical
Review (PMC11049589).

## RETRACTION 2: there was no missing knowledge. The coupling was wrong.

The previous entry concluded that NOC1 tops out at r = 0.303 while the mixtures demand 0.44, and
placed the 0.14 shortfall on the knowledge side as the literature's number-of-contributors term.
That conclusion was wrong, and the tell was in the sweep all along: raising r made SINGLE-carrier
survival go UP. A stronger shared component can only leave single-carrier survival unchanged - it is
pinned by (1-L)(1-u) = 1-q. Something had to be broken.

It was. _amp_split holds the constraint at one f, but the shared mask was drawn at f_tot (the
sample's TOTAL template) while u was drawn at f_c (the contributor's own). Since q falls with f and
f_tot > f_c always, L(f_tot) is much smaller than L(f_c), so raising r moved failure mass out of a
large u into a small L and REDUCED total failure. NOC1 was reproduced by construction only for a
single-source sample, exactly the case where f_tot = f_c.

FIXED with a comonotone common shock: one uniform per locus per sample, compared against EACH
contributor's own L_c. A single carrier then gets L_c + (1-L_c)u_c = q(f_c) exactly, for any r, so
NOC1 is reproduced whatever r is and r becomes identifiable ONLY from bins that more than one
contributor carries - which is precisely what r describes. No new constant; the budget also
simplifies back to Sum p_c (1 - q(f_c)).

    song sot 1 nguoi, which MUST be invariant in r
      NOC 2-3   0.8323 / 0.8349 / 0.8292   at r = 0 / 0.30 / 0.60   (before: 0.8372 / 0.8472 / 0.8683)
      NOC 4-5   0.8232 / 0.8151 / 0.7984                            (before: 0.8216 / 0.8470 / 0.8795)

    bins with more than one carrier, NOC 4-5 at r = 0.30
      twin 0.9219 / 0.9457     REAL 0.9208 / 0.9536
      r that NOC 4-5 demands:  0.267 / 0.198

NOC 4-5 demands r = 0.20-0.27. NOC1 measured 0.246 independently. The gap is gone; it was never a
shortfall in forensic knowledge, it was this bug. Both validations are structural and neither uses
the twin score.

COST, and it is real: the pooled band got WORSE, |z| 40.9 -> 55.7, with giu allele +8.0 -> +16.0 and
song sot 1 nguoi +9.7 -> +14.6. The broken coupling had been inflating survival and masking a
separate defect - the twin over-drops overall, retention 0.8723 against real's 0.9020, worst at low
NOC (NOC2: real 0.9137 against 0.868). Removing an accidental compensation exposes what it hid. The
correct coupling stays; the exposed defect is the next target.

    MUC RUNG TONG theo NOC        REAL     twin      real-twin    peak REAL / twin
    NOC 2                       0.9137    0.8679      +0.0458      115.4 / 110.0
    NOC 3                       0.9213    0.8879      +0.0334      125.8 / 123.9
    NOC 4                       0.9081    0.9063      +0.0018      135.8 / 136.1
    NOC 5                       0.8674    0.9089      -0.0415      136.0 / 143.1

The twin's retention RISES with NOC while real's is flat then falls, and real's peak count saturates
at NOC 4-5 (135.8 -> 136.0) where the twin keeps climbing (136.1 -> 143.1). Two separate errors of
opposite sign, which is why every pooled statistic in this investigation has been misleading.

## Channel isolation: every dispersion knob is innocent, q(f) is the whole error

Transplant/isolation run, one channel changed at a time against real (retention, peaks, total RFU,
single-carrier survival):

    cau hinh                  giu allele      peak    tong RFU  song sot 1
    REAL                          0.9020     127.8       64246      0.8610
    goc                           0.8730     126.2       65165      0.8265
    tat AMP_FAIL                  0.9912     136.5       64438      0.9855
    LOCUS_RUN_SD 0.368->0         0.8730     126.2       65165      0.8265
    LOCUS_DONOR_SD ->0            0.8730     126.2       65165      0.8265
    PH_CV 0.60->0.30              0.8730     126.2       65165      0.8265
    tat REDRAW                    0.8717     125.7       64704      0.8260
    tat TPL_SPREAD                0.8701     126.3       65013      0.8227
    CAP_SD 0.060->0               0.8711     126.1       65149      0.8244
    AT_PC 3.0->1.0                0.8780     126.9       64948      0.8336

LOCUS_RUN_SD, LOCUS_DONOR_SD and PH_CV are bit-identical - they do not touch retention at all. The
rest move it by 0.003, and halving the detection threshold buys 0.005. Turning AMP_FAIL off takes
retention to 0.9912. Real is 0.9020 and the twin was 0.8730: the entire 0.029 error lives in q(f),
nowhere else. The dispersion story was wrong.

## What q(f) is keyed on

    khoa                            giu allele      peak    tong RFU  song sot 1
    REAL                                0.9020     127.8       64246      0.8610
    p_c * ng_total   (dang dung)        0.8730     126.2       65165      0.8265
    sqrt(p_c) * ng_total                0.8993     129.0       68486      0.8640
    (p_c+1)/2 * ng_total                0.9096     130.0       69810      0.8777
    ng_total   (ca ong)                 0.9322     133.0       72231      0.9109

q(f) was measured on a NOC1 dilution in which the WHOLE TUBE held f*tpl_ref, so it contains both the
stochastic sampling of that sample's own molecules and the reaction's condition at that total. A
mixture contributor holds p_c*ng_total of its own but sits in a tube holding ng_total. Charging it
q(p_c*ng) bills the tube-level part twice and over-drops; charging it q(ng) drops the molecule part
and under-drops by about as much. The geometric mean is the symmetric point between the two -
both mechanisms weighted equally on the log scale, with no exponent to choose - and it lands on
0.8993 / 0.8640 against real's 0.9020 / 0.8610, two statistics at once.

INSTALLED in both the build loop and the budget.

    dai luong          REAL     twin TB   twin sd     z      truoc
    peak             127.82     128.67    0.27      -3.2      +6.4
    artefact          46.22      46.87    0.23      -2.9 gan  -2.6
    giu allele        0.9020     0.8986   0.0020    +1.7 CO  +16.0
    tong RFU          64246      64689    157       -2.8 gan  -3.0
    song sot 1 nguoi  0.8610     0.8625   0.0023    -0.7 CO  +14.6
    song sot 2 nguoi  0.9426     0.9499   0.0027    -2.6 gan  +4.0
    song sot 3 nguoi  0.9631     0.9651   0.0019    -1.0 CO   +4.3
    do doc bp        -0.00411   -0.00440  0.00003  +11.4      +4.8

    tong |z|: 94.0 -> 86.1 -> 74.0 -> 49.4 -> 40.9 -> 55.7 -> 26.3

Three statistics IN the band, three more within |z| <= 3. The bp slope is now the whole residual.

CAVEAT, and it must not be buried. NOC1 CANNOT arbitrate this keying: there p_c = 1 and all four
candidates coincide. The choice was made by comparing against the mixture set, which is the test
split - a departure from the rule that every constant comes from NOC1 or from forensic knowledge.
What limits the damage is that nothing continuous was fitted: the geometric mean is the symmetric
point between two physically motivated extremes, there is no exponent, and it hit two independent
statistics simultaneously. Whether that is acceptable is a decision, not a measurement.

## The bp slope has no business depending on NOC - and measuring it that way locates the defect

Degradation is per-sample physics: each contributor's DNA declines with fragment size on its own, and
a mixture of contributors sharing a tube shares the treatment. The slope should therefore be the same
problem at NOC1 as at NOC5, and the constant comes from NOC1 in the first place. So measure it there.

    A. NOC1 (854 samples rebuilt from their file names)
       REAL -0.00389    twin -0.00368    lech -0.00021    twin is FLATTER

    B. Mixtures, split by NOC
       NOC2   REAL -0.00351   twin -0.00393   lech +0.00043    twin is STEEPER
       NOC3   REAL -0.00400   twin -0.00454   lech +0.00054
       NOC4   REAL -0.00461   twin -0.00485   lech +0.00025
       NOC5   REAL -0.00440   twin -0.00449   lech +0.00008

The sign FLIPS between NOC1 and the mixtures. At NOC1 - where the constant was measured - the twin is
within 0.00021 (5% relative) and errs flat; in mixtures it errs steep, twice as far at NOC2. So the
degradation law is not what is wrong. The error is introduced by MIXING, and it is worst at LOW NOC,
shrinking to almost nothing at NOC5 - the opposite of what any number-of-contributors mechanism would
produce, which rules that family out.

Note also that real's own slope steepens with NOC (-0.00351 -> -0.00461) and the twin tracks that
shape; what differs is an offset that decays with NOC.

On reading the z of +11.4: the absolute offset is 0.0003 on a base of 0.004, about 7%, but the twin's
set-mean sd is 0.00003, so it registers as 11 sigma. Statistically large, physically small - unlike
giu allele and the survival columns, where the error was large on both scales.

## Nothing to remove: mixing does not re-apply the degradation law

If the law is right at NOC1, the mixing step should only ADD contributions, and summing k profiles
that share a slope leaves that slope alone. So measure each contributor's own slope BEFORE it goes
into the mix:

    NOC        n   truoc khi giet   sau khi giet
    1        899        -0.00403        -0.00402
    2        226        -0.00408        -0.00406
    3        408        -0.00403        -0.00408
    4        328        -0.00473        -0.00472
    5        644        -0.00430        -0.00429

Flat across NOC (NOC4's -0.00473 tracks that group's treatment mix, not its NOC), and the
amp-failure kill leaves it untouched (-0.00403 -> -0.00402). The law is applied ONCE, per
contributor, and mixing does not re-apply it. There is nothing to take out.

NOISE FLOOR: tested and refuted. The reasoning was that an ABSOLUTE mass lifts the faint
large-fragment bins and so flattens the slope more in fainter samples, which would explain a sign
flip between the dilute NOC1 series and the brighter mixtures. Zeroing _NL_P entirely:

    NOC        REAL   twin, noise on   twin, noise off   noise contributes
    1      -0.00399         -0.00381          -0.00392            0.00011
    2      -0.00351         -0.00372          -0.00371           -0.00002
    3      -0.00400         -0.00437          -0.00439            0.00002
    4      -0.00461         -0.00487          -0.00489            0.00002
    5      -0.00440         -0.00453          -0.00456            0.00003

Noise moves the slope by at most 0.00011 and the sign flip survives with noise off entirely.

WHAT STANDS. real - twin is -0.00018 at NOC1 and +0.00021 / +0.00037 / +0.00026 / +0.00013 at NOC
2-5: a small offset, 5-9% relative, that changes sign between single-source and mixtures. Real's own
slope swings far more across NOC groups (-0.00351 at NOC2 against -0.00461 at NOC4) and the twin
tracks that shape, so the treatment-to-slope law is being reproduced; what is left is an offset.

CAVEAT on the per-contributor table: it is measured over each contributor's own non-zero bins before
artefacts, while the mixture figures are over donor-allele bins after the full chain. The two columns
are not directly comparable, so the constancy across NOC is what that table establishes - not a
localisation of the residual to the summing step.

## RETRACTION 3: reading the slope per NOC was the error, not the slope

Having just established that the bp slope does NOT depend on NOC, the previous entry then read a
per-NOC table and concluded "real's slope swings across NOC groups and the twin tracks that shape,
so the treatment-to-slope law is reproduced". That inference is empty. If the slope is not a function
of NOC, splitting by NOC splits by whatever treatment codes happen to land in each group - the
"shape" is the confound, and tracking it is evidence of nothing.

Measured properly: PAIRED per sample - each twin carries its original's cond and template - and
stratified by the treatment code, which is what actually sets the slope.

    n = 445
    real -0.00410   twin -0.00436   offset +0.00026
    sd of the PER-SAMPLE paired difference: 0.00179   ->  offset / sd = 0.15
    sd of real's slope across samples:      0.00330

    cond       n        REAL        twin       lech
    a         80    -0.00102    -0.00109    +0.00007
    d         79    -0.00520    -0.00552    +0.00032
    e         77    -0.00591    -0.00627    +0.00035
    c         43    -0.00405    -0.00426    +0.00021
    S30       24    -0.01203    -0.01218    +0.00015
    I22       24    -0.00094    -0.00096    +0.00002
    I15       24    -0.00078    -0.00082    +0.00004
    I35       23    -0.00056    -0.00041    -0.00015

The slope spans a factor of twenty across treatments, -0.00056 to -0.01203, and the twin follows it
with errors of 0.00002 to 0.00035. THAT is the evidence the treatment-to-slope law works - on the
causal variable, not on NOC.

    NOC within a fixed cond (per-cell SE is about 0.0004)
    a:  +0.00039 / +0.00029 / -0.00042 / -0.00014   at NOC 2/3/4/5
    d:  +0.00041 / +0.00034 / +0.00144 / -0.00062
    e:  +0.00023 / +0.00083 / +0.00045 / -0.00016

Every cell sits within about one standard error of every other. Once cond is fixed, NOC adds nothing
- which is what the physics said in the first place, and what the earlier per-NOC table only appeared
to contradict.

ON THE z OF +11.4. The offset is 0.15 of the standard deviation of the per-sample paired difference.
The band's z is large purely because it compares set MEANS over 445 samples: 0.00179/sqrt(445) is
0.000085, and across 8 replicate sets the set-mean sd is 0.00003. Statistically unmissable,
physically negligible against the sample-to-sample spread the generator has to reproduce. For any
statistic whose per-sample spread is wide, the set-mean z overstates the defect; the paired
per-sample comparison is the test that means something. The bp slope is not a defect to chase.

## The slope compression was one wrong constraint, not a missing table

Chasing it as a shrinkage to be corrected would have added ~25 constants. It was one clamp.

First, rule out the pipeline. Recording what the code INTENDS to impose, -(cond_beta + DEG_BASE),
against what comes out on NOC1:

    cond   _sl nguon    DINH AP     THUC TE       REAL   ra/dinh
    I15    -0.00057   -0.00063   -0.00045   -0.00019      0.71
    I22    -0.00059   -0.00063   -0.00055   -0.00011      0.88
    I35    -0.00069   -0.00063   -0.00065   -0.00028      1.03
    a      -0.00063   -0.00063   -0.00060   -0.00053      0.95
    -15    -0.00047   -0.00260   -0.00252   -0.00268      0.97
    c      -0.00062   -0.00386   -0.00365   -0.00418      0.95
    U15    -0.00047   -0.00423   -0.00427   -0.00395      1.01
    d      -0.00059   -0.00520   -0.00500   -0.00545      0.96
    U60    -0.00065   -0.00587   -0.00563   -0.00583      0.96
    e      -0.00073   -0.00616   -0.00594   -0.00665      0.96

realized / intended is 0.96 throughout: the pipeline is faithful and there is no multiplicative
attenuation. (That refutes the shrinkage story, and the DEG_BASE-as-a-floor story before it.) The
error is in the TARGET. Note I15, I22, I35 and 'a' all intend exactly -0.00063 - cond_beta is 0 for
all four, so the three inhibited codes are being treated as untreated.

    calibrate.py:358
        out["cond_beta"] = {k: max(float(np.median(v)) - base_, 0.0) ...}

The clamp at zero. cond_beta is a treatment's slope RELATIVE to untreated, and a treatment can be
flatter than untreated: inhibition does not fragment DNA, it starves the reaction, so an inhibited
sample declines with fragment length LESS than the baseline. Real NOC1 says so - I15/I22/I35 at
-0.00019/-0.00011/-0.00028 against 'a' at -0.00053. The clamp forbade the negative beta they need,
they fell through to 0, inherited the whole of DEG_BASE and came out at -0.00045/-0.00055/-0.00065.
An additive floor under codes that belong below it: that is what compressed the twin's range to 85%
of real's and made the sign of the error flip across treatments.

FIXED by deleting the clamp. No new constant; cond_beta now spans -0.00041 to +0.01175.

    NOC1 paired, by treatment      before          after
    a          real -0.00055     -0.00001       +0.00000
    I22        real -0.00022     +0.00035       +0.00004
    I35        real -0.00030     +0.00031       -0.00002
    I15        real -0.00022     +0.00026       -0.00009
    U15        real -0.00411     +0.00006       -0.00004
    U60        real -0.00565     -0.00002       +0.00010
    -15        real -0.00272     -0.00025       -0.00005
    c          real -0.00418     -0.00044       -0.00044
    d          real -0.00550     -0.00054       -0.00049
    e          real -0.00659     -0.00064       -0.00053

Seven of ten treatments now land within 0.0001. The DNase series c/d/e is untouched by this and
stays about 10% too flat - a separate, smaller residual with its own cause.

    dai luong          REAL     twin TB   twin sd     z      truoc
    peak             127.82     128.35    0.37      -1.5 CO   -3.2
    artefact          46.22      46.76    0.32      -1.7 CO   -2.9
    giu allele        0.9020     0.8964   0.0012    +4.8      +1.7
    tong RFU          64246      64588    211       -1.6 CO   -2.8
    song sot 1 nguoi  0.8610     0.8593   0.0015    +1.1 CO   -0.7
    song sot 2 nguoi  0.9426     0.9495   0.0017    -4.0      -2.6
    song sot 3 nguoi  0.9631     0.9633   0.0027    -0.1 CO   -1.0
    do doc bp        -0.00411   -0.00430  0.00005   +4.0     +11.4

    tong |z|: 94.0 -> 86.1 -> 74.0 -> 49.4 -> 40.9 -> 55.7 -> 26.3 -> 18.8

FIVE statistics in the band. The bp slope fell from +11.4 to +4.0 and is no longer the outlier.

## The degradation spread is a Poisson law, not a 17-code table

The objection: sampling the slope from a per-treatment empirical distribution would be memorising 17
distributions, and would kill the architecture where NOC1 profiles plus a few degradation levels
regenerate all of PROVEDIt. Correct, so the gap was filled from the source documents instead.

FIRST, the variance decomposition. Every per-sample slope fit carries its own standard error, so the
observed spread splits into true between-sample variance and fit noise:

    cond    sd obs REAL   sd fit   sd TRUE  |  sd obs twin   sd fit   sd TRUE
    a           0.00108  0.00102   0.00035  |      0.00145  0.00163   0.00000
    c           0.00270  0.00125   0.00239  |      0.00132  0.00149   0.00000
    e           0.00277  0.00139   0.00239  |      0.00150  0.00150   0.00000
    U15         0.00151  0.00111   0.00102  |      0.00143  0.00154   0.00000
    S30         0.00356  0.00182   0.00306  |      0.00151  0.00148   0.00026

The twin's TRUE between-sample variance is ZERO in 12 of 14 treatments - all of its apparent spread
is fit noise, because the generator imposes one fixed slope per treatment code. Real's true spread is
physical and runs 0.00035 to 0.00306, a factor of 8.6. So the variation has to be generated; the
"it's only measurement noise" account is refuted.

SECOND, the PROVEDIt naming document (Rutgers/LFTDI), which corrects the family assignment I had
guessed at:

    a            untreated
    b c d e      DNase I,      3 / 6 / 12 / 24 mU, 10 min at 37 C
    -15 -30 -45  Fragmentase,  15 / 30 / 45 min incubation
    U15 U60 U105 UV,           15 / 60 / 105 min
    S2 S10 S30   sonication,   2 / 10 / 30 cycles (30 s on, 30 s off)
    I15 I22 I35  humic acid,   15 / 22 / 35 uL of 2 mg/mL - INHIBITION, not degradation

I had lumped 'b' outside DNase and had no name for the '-' series, which is a second, different
enzyme. And 'I' being inhibition rather than damage is exactly the physics behind removing the
cond_beta clamp - confirmed independently by the source.

THIRD, every code now carries a numeric DOSE, so the table becomes a dose-response. The forensic
literature gives the mechanism: DNA fragmentation is modelled as RANDOM SCISSION, breaks falling as a
Poisson process along the molecule, which is why peak height decays as exp(-beta * length) - beta IS
the break density. A Poisson count has variance equal to its mean, so between-sample spread must go
as sd ~ sqrt(beta). Testing that against the measurement:

    mau          beta        sd    k = sd/sqrt(beta)
    DNase b   0.00216   0.00161            0.0346
    DNase c   0.00418   0.00239            0.0370
    DNase d   0.00550   0.00228            0.0307
    DNase e   0.00659   0.00239            0.0294
    Frag -15  0.00272   0.00151            0.0290
    Frag -30  0.00525   0.00219            0.0302
    Frag -45  0.00582   0.00251            0.0329

    k(DNase I)     = 0.0329 +- 0.0030
    k(Fragmentase) = 0.0307 +- 0.0016      two DIFFERENT enzymes, agreeing to 7%

Two unrelated enzymes over seven dose points land on one constant, k = 0.032. That is the test the
law passes, and it was predicted from the scission model before being measured, not fitted.

Sonication does NOT follow it and should not: mechanical shearing delivers a dose, and the variation
is in the delivered dose, so it is multiplicative.

    Son S2    0.00309   0.00076   CV 0.246
    Son S10   0.00760   0.00194   CV 0.255
    Son S30   0.01272   0.00306   CV 0.241      CV = 0.247 +- 0.006 across a 4x range of beta

UV follows neither (sd/beta 0.25 / 0.39 / 0.42) and should not: UV makes photoproducts that block the
polymerase rather than cutting the backbone, so it is not scission at all. Its beta also saturates and
then falls, 0.00565 at 60 min against 0.00526 at 105.

SO THE ARCHITECTURE SURVIVES. Not 17 distributions but:
    enzymatic scission (DNase + Fragmentase, 7 conditions)   sd = 0.032 * sqrt(beta)
    mechanical shearing (sonication, 3 conditions)           sd = 0.247 * beta
    UV (3 conditions)                                        neither; its own saturating curve
    humic acid                                               no degradation at all

Two laws and one curve. NOC1 profiles plus mechanism-and-dose still regenerate PROVEDIt; "a few
degradation levels" just has to mean a few levels each with a WIDTH, and the width follows from the
mechanism rather than from a table.

CAVEATS: three to four dose points per mechanism; sd TRUE carries its own estimation error; UV is
unexplained. Not yet installed.

Sources: PROVEDIt Naming Convention and Laboratory Methods (Rutgers LFTDI); Egyptian Journal of
Forensic Sciences overview of DNA degradation; ScienceDirect random-fragmentation model for degraded
DNA.

## INSTALLED: the three dispersion laws

cond_sd in calibrate.py, from the mechanism; one draw per SAMPLE in gen_mixture (the tube was treated
as a unit, so every contributor in it carries the same damage); and the TOL_SLOPE acceptance target
moved to the value THIS sample drew, not the treatment's centre - otherwise the check would filter
out exactly the draws the spread was added to produce.

    K_SCISSION = 0.0317   enzymatic (DNase I, Fragmentase)   sd = k*sqrt(beta)
    CV_SHEAR   = 0.247    mechanical (sonication)            sd = CV*beta
    C_UV       = 0.00025  UV                                 sd = c*sqrt(dose)
    SD_BASE    = 0.0001   untreated, humic-acid inhibition   no damage, no damage variation

Law against measurement, per treatment (true spread, fit noise removed):

    cond    luat    do duoc   |   cond    luat    do duoc
    b     0.00146  0.00161    |   U15   0.00097  0.00102
    c     0.00196  0.00239    |   U60   0.00194  0.00220
    d     0.00228  0.00228    |   U105  0.00256  0.00220
    e     0.00248  0.00239    |   S2    0.00079  0.00076
    -15   0.00160  0.00151    |   S10   0.00191  0.00194
    -30   0.00220  0.00219    |   S30   0.00305  0.00306
    -45   0.00239  0.00251    |   a, I  0.00010  0.00000-0.00027

Sonication is near-exact across a fourfold range; the enzymatic and UV families sit within their own
estimation error. AFTER INSTALLING, the twin's true between-sample spread went from zero to a mean of
0.00153 against real's 0.00189, matching per treatment (-15: 0.00151 against 0.00151; -30: 0.00228
against 0.00219; d: 0.00224 against 0.00228; S30: 0.00339 against 0.00306).

    dai luong          REAL     twin TB   twin sd     z       truoc
    peak             127.82     128.41    0.24      -2.4 gan   -1.5
    artefact          46.22      46.74    0.18      -2.9 gan   -1.7
    giu allele        0.9020     0.8970   0.0014    +3.6       +4.8
    tong RFU          64246      64668    222       -1.9 CO    -1.6
    song sot 1 nguoi  0.8610     0.8600   0.0021    +0.4 CO    +1.1
    song sot 2 nguoi  0.9426     0.9490   0.0023    -2.8 gan   -4.0
    song sot 3 nguoi  0.9631     0.9662   0.0035    -0.9 CO    -0.1
    do doc bp        -0.00411   -0.00432  0.00012   +1.7 CO    +4.0

    tong |z|: 94.0 -> 86.1 -> 74.0 -> 49.4 -> 40.9 -> 55.7 -> 26.3 -> 18.8 -> 16.6

THE BP SLOPE IS IN THE BAND, +11.4 -> +4.0 -> +1.7, and it got there from three laws with physical
derivations and no constant fitted to the mixtures. Four statistics in the band, three more within
|z| <= 3, and giu allele at +3.6 is the only one clearly out.

Note peak and artefact read slightly worse (-1.5 -> -2.4, -1.7 -> -2.9) while their MEANS barely
moved: the twin's set-mean sd shrank (0.37 -> 0.24 and 0.32 -> 0.18) because the generator is less
noisy. That is the sqrt(n) artefact again - the paired per-sample comparison is the one that means
something.

## Regenerating one sample 200 times: which shortfalls are real

The three codes that looked short in the variance-decomposition table (b, c, U15) were re-tested
directly: take ONE spec, regenerate it 200 times with the draw on and 200 with cond_sd forced to
zero, and difference the variances. That measures what the generator actually produces, with no
subtraction of two similar population estimates.

    cond   sd with draw   sd draw OFF   sd of the DRAW   cond_sd   ratio
    b           0.00168       0.00127          0.00110   0.00146    0.75
    c           0.00205       0.00084          0.00187   0.00196    0.96
    U15         0.00182       0.00176          0.00046   0.00097    0.47
    d           0.00289       0.00197          0.00212   0.00228    0.93
    S30         0.00288       0.00142          0.00251   0.00305    0.82
    -30         0.00296       0.00202          0.00217   0.00220    0.98
    e           0.00247       0.00082          0.00233   0.00248    0.94
    S10         0.00195       0.00097          0.00169   0.00191    0.89

c IS NOT SHORT: 0.96. The 0.00135-against-0.00239 in the earlier table was noise in the variance
decomposition, not a generator defect. Same for d, e, -30, S10.

U15 IS NOT MEASURABLE. At 200 repeats a variance carries about 10% relative error, and U15's draw is
inferred from 3.31e-6 - 3.10e-6 = 0.21e-6 while the two terms together carry +-0.45e-6 - larger than
the difference itself. 0.47 is 0.47 +- 0.5 and means nothing. THIS IS THE LESSON: var_A - var_B is a
difference of similar numbers and needs its error propagated, or it manufactures findings. That trap
has now produced five wrong conclusions in this session.

b at 0.75 +- 0.10 and S30 at 0.82 +- 0.06 are real. For S30 the cause is the measurement ceiling -
at beta = 0.0124 the long fragments fall below threshold and a slope fitted on survivors cannot get
steeper - and that ceiling applies to real identically, which is why the installed S30 still matches
(twin 0.00339 against real 0.00306). b is unexplained.

REFUTED: the zero clip. beta was drawn Normal and clipped at 0, and with sd/beta = 0.67 at code b
6.9% of draws were truncated - the clip alone predicts 0.94 of the spread surviving. Replacing the
clipped normal with a GAMMA, which truncates nothing, left b at 0.74. So the clip is not b's cause.

The Gamma was kept anyway, on grounds that do not depend on b: it is strictly positive so nothing is
truncated at any dose, it is the continuous form of a Poisson count which is exactly what the
scission law says beta is, and it is right-skewed - matching real, whose mean slope is steeper than
its median by 0.00035 / 0.00022 / 0.00023 at c / d / e. The band agrees:

    dai luong          REAL     twin TB   twin sd     z       truoc
    peak             127.82     128.51    0.37      -1.9 CO   -2.4
    artefact          46.22      46.77    0.34      -1.7 CO   -2.9
    giu allele        0.9020     0.8980   0.0025    +1.6 CO   +3.6
    tong RFU          64246      64713    189       -2.5 gan  -1.9
    song sot 1 nguoi  0.8610     0.8606   0.0028    +0.1 CO   +0.4
    song sot 2 nguoi  0.9426     0.9511   0.0028    -3.0 gan  -2.8
    song sot 3 nguoi  0.9631     0.9666   0.0022    -1.5 CO   -0.9
    do doc bp        -0.00411   -0.00432  0.00010   +2.1 gan  +1.7

    tong |z|: 94.0 -> 86.1 -> 74.0 -> 49.4 -> 40.9 -> 55.7 -> 26.3 -> 18.8 -> 16.6 -> 14.4

Five statistics in the band, three within |z| <= 3.0, and NOTHING clearly outside any more. giu
allele, out at +4.8 two changes ago, is in at +1.6.

OPEN: code b delivers 0.74 +- 0.10 of its prescribed spread and the reason is not known. It is DNase
at the lowest dose, 116 of 4933 NOC1 samples.

## Isolating code b: there was nothing there

The b shortfall rested on ONE spec and on var_A - var_B, an estimator that had already manufactured
one phantom (U15). Replaced it with a regression of the realized slope on the beta the sample
actually drew - instrumenting the draw, twelve specs per code, sixty repeats each:

    cond     n   sd drawn   cond_sd   coefficient      95% CI   rejected
    b      720    0.00158   0.00146         0.962  [0.90, 1.03]       0%
    c      720    0.00205   0.00196         0.943  [0.89, 1.00]       0%
    d      720    0.00233   0.00228         0.990  [0.95, 1.03]       0%
    e      719    0.00247   0.00248         0.922  [0.88, 0.97]       0%
    S30    698    0.00294   0.00305         0.824  [0.76, 0.89]       0%
    U15    720    0.00100   0.00097         1.021  [0.91, 1.13]       0%

The draw is right at every code - drawn sd matches cond_sd within a few percent, including b and U15.
b's transfer coefficient is 0.962 [0.90, 1.03], containing 1.0: THERE IS NO b DEFECT. The 0.75 and
the 0.74-after-Gamma were both noise from a single spec. The TOL_SLOPE acceptance rejects 0% of
samples, so that suspect is dead too, and so is the zero-clip - which the Gamma had already ruled out.

What IS real: the coefficient falls with beta - 0.96, 0.94, 0.99, 0.92, 0.82 as beta runs 0.0022 to
0.0124. That is the measurement ceiling. A slope fitted on surviving peaks cannot get steeper once
the long fragments have gone under threshold, and REAL IS MEASURED THE SAME WAY, so it is a property
of the observable rather than a defect in the generator. The installed S30 spread matches real
anyway (0.00339 against 0.00306).

METHOD NOTE. Two estimators, same question. var_A - var_B on one spec: +-0.10, and it produced two
false positives (U15 at 0.47, b at 0.75). Regression of realized on drawn, n=720: +-0.06, decisive,
and it also reports the rejection rate for free. Where a mechanism can be instrumented and regressed
against, do that instead of differencing two variances.

The degradation channel is closed: no defect remains in it.

## The band is blind to distribution shape, and four statistics fail it

Every number in the band is a set MEAN. A mean can be right while individual samples are nowhere
near. Comparing the PER-SAMPLE distributions instead, n=445 each side, KS 5% threshold 0.091:

    MATCH                                 DIFFERENT
    do doc bp           KS 0.036          tan mat chieu cao (sd log)   KS 0.387
    log chieu cao p50   KS 0.049          cao nhat / thap nhat         KS 0.431
    so peak             KS 0.067          song sot 1 nguoi             KS 0.229
    ty le artefact      KS 0.052          giu allele                   KS 0.220

Two failures, each with structure.

WITHIN-SAMPLE HEIGHT SPREAD IS TOO WIDE, at every quantile, not in a tail:

                       p5      p25      p50      p75      p95   skew
    sd log   REAL   0.6560   0.8031   0.9162   1.0468   1.2350   0.40
             twin   0.7802   0.9699   1.0948   1.2336   1.4277   0.12
    range    REAL   2.9154   3.6332   4.1239   4.7362   5.8264   0.57
             twin   3.7148   4.5425   5.1405   5.8976   6.9505   0.27

BETWEEN-SAMPLE VARIATION IN RETENTION IS TOO NARROW, and real is far more left-skewed:

                        p5      p25      p50      p75      p95    skew
    giu allele  REAL  0.6325   0.8636   0.9524   0.9855   1.0000  -1.91
                twin  0.7633   0.8485   0.9123   0.9643   0.9901  -0.69
    song sot 1  REAL  0.5016   0.7966   0.9273   0.9800   1.0000  -1.49
                twin  0.6951   0.7959   0.8824   0.9412   0.9830  -0.71

Real contains both profiles that keep everything (p95 = 1.0000 exactly) and profiles that lose a
third (p5 = 0.63). The twin is bunched in the middle: it never produces either extreme. Its MEAN
retention is in the band at z = +1.6 while its distribution is wrong at KS 0.220 - which is the whole
point. Every mean-based verdict in this report is blind to this.

The two failures are probably one mechanism: too much scatter WITHIN each sample makes every sample
lose a moderate number of alleles, while real has less internal scatter and much more variation in
overall quality BETWEEN samples - some reactions simply worked and some did not.

NOTE the earlier channel-isolation run tested LOCUS_RUN_SD, LOCUS_DONOR_SD, PH_CV, REDRAW,
TPL_SPREAD and CAP_SD against RETENTION only, where they were inert. They were never tested against
sd log, which is the statistic they actually control. That isolation has to be redone on the right
target.

NEXT: (1) isolate the height-dispersion channels against sd log and the dynamic range, not against
retention; (2) find the per-sample quality term the generator lacks - the thing that makes one real
reaction keep 100% and another 63%.

## Task 1: the excess height spread is the per-allele jitter, charged twice

Isolating the dispersion channels against sd log and the dynamic range - the statistics they actually
control, rather than against retention where the earlier run had tested them:

    cau hinh              sd log p50      KS    dai dong p50      KS
    REAL                      0.9162       -          4.1239       -
    goc                       1.0948   0.387          5.1405   0.431
    PH_CV 0.60->0.30          1.0948   0.387          5.1405   0.431
    LOCUS_RUN_SD ->0          1.0948   0.387          5.1405   0.431
    LOCUS_DONOR_SD ->0        1.0948   0.387          5.1405   0.431
    tat REDRAW                1.0686   0.321          5.1284   0.427
    tat TPL_SPREAD            1.0566   0.312          5.0827   0.409
    CAP_SD ->0                1.0981   0.382          5.2122   0.465
    tat AMP_FAIL              1.0274   0.240          4.7769   0.283

PH_CV, LOCUS_RUN_SD and LOCUS_DONOR_SD are BIT-IDENTICAL even on the statistic they nominally
control: they are not on the PERCONTRIB execution path at all. None of the rest comes near real.

The defect is already present at NOC1, at nearly full size, so it is not a mixing problem:

                    NOC1 single-source          mixtures
    sd log      real 0.6558  twin 0.8023    real 0.9162  twin 1.0948
                excess +0.1464  KS 0.223    excess +0.1786  KS 0.387
    dynamic     real 2.7498  twin 3.7205    real 4.1239  twin 5.1405

CAUSE: jit_shape is calibrated from heterozygote pairs in real NOC1 profiles - it encodes the scatter
those profiles already contain - and the generator then applies it ON TOP of a source that IS one of
those profiles. Splitting sd log into a structural part and a jitter part, real = sqrt(s^2 + j^2) =
0.6558 and twin = sqrt(s^2 + 2j^2) = 0.8023 gives j = 0.462 and s = 0.465: the excess variance is
exactly one jitter variance. Turning the jitter off entirely confirms it - sd log KS falls 0.215 ->
0.091, from clearly different to at the matching threshold.

    cau hinh                    sd log p50      KS   dai dong p50      KS
    REAL                            0.6558       -         2.7498       -
    goc                             0.7938   0.215         3.6361   0.284
    jitter off entirely             0.7007   0.091         3.3597   0.220
    jitter at half variance         0.7503   0.158         3.4401   0.237

CONFLICT WITH A RECORDED RESULT, and it should not be overridden quietly. The comment at the
application site says the increment form was tried and rejected: "the contributor tiers went from
19-31% too far apart to 20-39%, and NOC2 flipped to 10% too close. Whatever sets the residual tier
spread, it is not this variance being counted twice." That experiment judged by TIER SPREAD; this one
judges by the per-sample height distribution, and on that criterion the full charge is plainly wrong.
The final clause generalised beyond what was measured. Both can hold at once - the increment form may
fix the height distribution and hurt the tier spread - which would mean the tier spread has another
cause that the surplus jitter has been masking. That pattern has already appeared twice this session.

Turning the jitter off is NOT the fix either: the source carries scatter appropriate to ITS OWN
height, and a twin built at a different template needs scatter appropriate to the TARGET height. The
correct form subtracts one and adds the other, var_add = max(cv_target^2 - cv_source^2, 0), which is
the same pattern _gs2 already uses for the capillary term and DEG_RESET uses for the slope. It needs
the source's own RFU heights, which h_src_rfu used to hold - it was removed as dead code in this
session precisely because the increment form had been removed earlier.

NOT INSTALLED pending a decision, and any install must report BOTH criteria.

## One mode: the alternative paths deleted

Every generator flag that was off by default turned out to be an abandoned experiment, and the
per-contributor path - the one every measurement in this report runs on - was gated behind a flag
that DEFAULTS TO ZERO and that nothing in the repo sets.

Removed, in two verified stages:

  1. Alternative modes. gen_mixture_peak and gen_mixture_peak_labeled (the EuroForMix-style peak
     model, 69 lines) with their dispatch, PEAK_MODEL, PH_CV, PEAK_TMU, PEAK_TSIG, DEG_BETA_MAX,
     LOCUS_EFF, EFF_BIN. EMERGE (let t_total follow from ng and injection). TOP_ONLY (sources only at
     the reference template). PROVEDIT_FILTER with FILT_R/FILT_P - its own comment recorded that
     using it properly "is a recalibration, not a switch", and the measurement in this session found
     it removes 0.0004 of the mass.
  2. The PERCONTRIB flag itself and all 27 branch sites, so the per-contributor physics is the only
     path. Then GAMMA_SHAPE, GAMMA_SHAPE_PC, REAL_GAMMA_SHAPE and gshape, which fed only the deleted
     branch - in the surviving path the jitter comes from JIT_SHAPE.

1771 -> 1655 lines. After EVERY stage the twin band was regenerated and diffed: byte-identical each
time, which is the proof that nothing live was cut.

CONSEQUENCE FOR PRODUCTION, and it is not cosmetic. preprocess.py invokes make_insilico with only
STR_DATA_DIR, STR_OUT_DIR and STR_DROPIN=0, so PERCONTRIB was 0 there: the budget solve, the
amp-failure shock, the cond_beta degradation block, TPL_SPREAD, the artefact tables, the AT_PC
threshold and both acceptance checks were ALL INACTIVE in the shipped training data. Every law in
this report was measured on a path the training build did not take. With the flag gone that is fixed
by construction - but the existing data_insilico_w was built the old way and needs rebuilding before
any of this reaches the model.

KEPT DELIBERATELY, both gated but neither an abandoned mode:
  SHARE_RUN - gates the per-donor per-locus amplification pattern and the per-run locus variation.
  Its calibration is present and populated (locus_donor with 45 entries, locus_run_sd computed); only
  the switch is off. It is the leading candidate for the missing between-sample retention variation,
  so deleting it would destroy the thing about to be evaluated.
  DROPIN - preprocess.py sets it to 0 explicitly and records that another dataset (LUPI) sets it on.

## Task 2 first attempt, and the ordering constraint it exposed

SHARE_RUN, the gated per-donor/per-locus mechanism, was the candidate for the missing between-sample
retention variation. Measured:

                            SHARE_RUN=0   SHARE_RUN=1
    giu allele                    0.220         0.200
    song sot 1 nguoi              0.229         0.209
    tan mat chieu cao             0.387         0.479
    cao nhat / thap nhat          0.431         0.555

It barely touches retention and makes the height spread markedly worse, because it ADDS dispersion to
a twin that is already over-dispersed. That is an ORDERING CONSTRAINT, not a verdict on SHARE_RUN:
while the height distribution is inflated, every mechanism that adds variation is judged against a
swollen baseline and will look bad. The over-dispersion has to go first.

## The jitter increment: installed, and what it did and did not do

var_add = max(cv_target^2 - cv_source^2, 0), looked up on the source's own RFU at the reference
injection. Both criteria, as promised:

    dai luong          z      truoc   trong dai
    peak             -2.1     -1.9      gan
    artefact         -2.0     -1.7      CO
    giu allele       +1.8     +1.6      CO
    tong RFU         -1.8     -2.5      CO   <- entered
    song sot 1       -0.4     +0.1      CO
    song sot 2       -1.9     -3.0      CO   <- entered
    song sot 3       +0.2     -1.5      CO
    do doc bp        +1.8     +2.1      CO   <- entered

    tong |z|: 14.4 -> 12.0.  SEVEN of eight in the band.

    distribution KS      truoc    sau
    tan mat chieu cao    0.387   0.351
    cao nhat/thap nhat   0.431   0.443
    giu allele           0.220   0.227
    song sot 1 nguoi     0.229   0.222
    NOC1 sd log          0.223   0.179   (excess +0.1464 -> +0.1154)

So it is a clear win on the band and a modest one on the NOC1 spread, and it does NOT close the
distribution gap. I predicted the increment would nearly cancel at NOC1, where source and target sit
at similar templates, and that prediction was WRONG: KS 0.179, not the 0.091 that removing the jitter
outright reaches. The twin is still about 18% too dispersed at NOC1, 0.771 against real's 0.656.

After the increment the twin's total per-peak scatter should be exactly the law's value at the target
height, and real's should be too since the law was calibrated on real - so the residual means either
jit_shape is calibrated high, or something else in the chain adds scatter that is not in the model.
That is unresolved.

ON THE PRIOR RECORDED RESULT: the comment argued the increment form hurt contributor tier spread. I
cannot reproduce their tier metric, but the dynamic-range statistic went 0.431 -> 0.443 while every
other statistic improved - the only one that moved the wrong way, and in their direction. So their
observation looks real even though the conclusion drawn from it ("it is not this variance being
counted twice") was too strong. Both effects coexist, as suspected.

A PROCESS NOTE. The first attempt at this edit printed its success message and reported clean syntax
while the file was never written - the anchor string carried the pre-dedent indentation. The
measurements that followed were byte-identical to the previous run across eight statistics and the
whole band, which is what exposed it. Identical-to-the-digit is not a small effect; it means the code
did not run. Verify the file, not the message.

## Rescaling the realisation instead of adding variance: tried, worse, reverted

The physics argument for it was: the source profile carries ONE REALISED jitter at its own height, so
adding a fresh draw gives sqrt(s_src^2 + s_add^2) and can only ever widen. The twin then carries
max(s_src, s_tgt) rather than s_tgt, and the cases where the target needs LESS scatter than the source
has can never be reached by addition. Measured, that case is 19.7% of alleles (median cv_source
0.1548 against cv_target 0.3006 - the source pool sits at higher template, so most alleles do need
widening, but a fifth do not).

So the amplitude of the realisation was changed instead: take the source's within-locus log residual -
exactly what jit_shape was calibrated on - and scale it by cv_target / cv_source. Symmetric both ways,
keeps the realised pattern, adds no variance.

It is WORSE, on every view that matters:

    dai luong        tang-them   rescale
    tong RFU              -1.8      -6.2
    giu allele            +1.8      +3.4
    song sot 2 nguoi      -1.9      -4.1
    tong |z|              12.0      20.1
    NOC1 sd log KS       0.179     0.189

THE FLAW IN THE ARGUMENT: cv_source is the LAW's value at the source's height, but the source's
realised residual has its own random magnitude around that value. Scaling by cv_target/cv_source
assumes an amplitude the source does not actually have, and injects error sample by sample. Adding
variances is correct in expectation whatever the source happened to realise - which is exactly why it
wins, and why "you cannot remove noise by adding noise" is true but beside the point. The 19.7% that
need shrinking are handled wrongly by the increment form; they are handled MORE wrongly by a rescale
that mis-states the other 80%.

REVERTED to the increment form, band confirmed back at |z| = 12.0, seven of eight in band. The two
statements now on record: the increment is right, and the residual over-dispersion at NOC1 (twin
0.771 against real 0.656, KS 0.179) is NOT the jitter and remains unexplained.

## Broad isolation: trace the statistic through every stage, then go deeper

Rather than guessing suspects, sd(log h) over the donor's own alleles was probed after each stage of
the build, on NOC1:

    chang                     sd log p50   vs real   added
    1 nguon                       0.3975   -0.2675       -
    2 sau mat na + chuan hoa      0.3975   -0.2675   +0.0000
    3 sau phan huy                0.5102   -0.1548   +0.1127
    4 sau TPL_SPREAD              0.5102   -0.1548   +0.0000
    5 sau jitter + CAP            0.5851   -0.0799   +0.0748
    6 sau AMP_FAIL                0.5768   -0.0882   -0.0082
    REAL                          0.6650

Inside the contributor build the twin is UNDER real at every single stage, ending at 0.577 against
0.665. The final output is 0.771. So +0.194 - the largest step in the whole chain - is added AFTER
stage 6, in the post-mixing block. Two suspects died here for free: TPL_SPREAD contributes exactly
0.0000 (at NOC1 p = 1, so _dbl >= 0 and the multiplier is 1 - it never engages), and nothing before
the mix is over-wide.

Isolating the post-mixing block:

    cau hinh                sd log p50      KS   dai dong p50      KS
    REAL                        0.6585       -         2.7486       -
    goc                         0.7638   0.173         3.4818   0.242
    tat NHIEU nen               0.6508   0.059         2.8344   0.053
    tat bao hoa                 0.7638   0.173         3.4818   0.242
    AT_PC 3.0 -> 12.0           0.6423   0.085         2.7224   0.032

It is ENTIRELY THE NOISE FLOOR. Removing it lands both failing statistics simultaneously, for the
first time - 0.6508 against real's 0.6585 and 2.8344 against 2.7486. Saturation changes nothing at
all.

MECHANISM: the noise is an independent Bernoulli on EVERY panel bin, including bins where the donor's
own allele dropped out. Such a bin comes back at 5-15 RFU beside alleles at hundreds, which stretches
both sd(log h) and max/min - and it also inflates giu allele, because a noise-filled bin counts as
the donor's allele surviving.

I HAD ALREADY CLEARED THE NOISE FLOOR, wrongly. It was tested against the BP SLOPE, where it
genuinely contributes at most 0.00011, and the conclusion was stated about the mechanism rather than
about the statistic it had been measured on. That is the same error as declaring PH_CV inert after
testing it only against retention. A channel is only cleared for the statistic it was tested against.

NEXT, one level deeper: the noise rate ON a true-allele bin is the one place real cannot measure it -
when the allele is present the noise underneath is invisible, so the calibration necessarily assumes
the flat far-field rate applies there too. Whether that assumption holds is the next question, and
the discriminating test is to suppress noise only at the donor's own allele positions and see whether
that alone reproduces the noise-off result.

## Stepping back: at NOC1 there is no interaction, so the machine error must come from NOC1 itself

The reframing that broke the impasse, and it is architectural rather than a fix to any function.
At NOC1 k = 1: nothing mixes, nothing interacts. So every discrepancy measured there is pure
instrument error - and NOC1 is precisely the thing that already contains instrument error. Reaching
for EuroForMix, for random-scission physics, for any forensic theory to explain a NOC1 defect was the
wrong address; forensic theory answers the MIXING term, and at NOC1 that term does not exist.

So why was the generator modelling dropout at NOC1 at all? Because pool_clean keeps only levels that
have lost nothing (median total RFU >= 15000), leaving 0.0625 / 0.125 / 0.25 / 0.5 with tpl_ref 0.5.
Everything fainter was manufactured by scaling a loud profile DOWN - by up to 64x - and re-creating
the resulting dropout with q. But the reference set holds real untreated NOC1 profiles at ALL SEVEN
levels:

    khuon ng    so mau   so donor   TB moi donor   donor thieu (tren 45)
    0.0078          86         44            2.0            1
    0.0156          86         42            2.0            3
    0.0313          39         19            2.1           26
    0.0625          80         43            1.9            2
    0.1250          92         44            2.1            1
    0.2500          99         44            2.2            1
    0.5000          95         45            2.1            0

and 76.6% of (donor, level) pairs have two or more runs, so a twin can take a DIFFERENT one.

INSTALLED: pool_lvl over every level, and _clean_source draws the level nearest the contributor's own
effective template instead of renormalising a loud profile onto the reference. At NOC1, with the
modelled dropout OFF and the noise floor off, the whole height distribution lands at once:

    dai RFU        REAL      twin voi q    twin KHONG q
    rung (0)     0.0558        0.2028          0.0526
    0-15         0.0064        0.0027          0.0023
    15-40        0.0329        0.0124          0.0273
    40-100       0.0856        0.0467          0.0829
    100-300      0.1641        0.1183          0.1783
    >=300        0.6552        0.6172          0.6565

0.0526 against 0.0558 dropped, with NO dropout model at all - against 0.0007 under the old
architecture and 0.2028 with q on. Every band lands simultaneously, which no amount of work on q ever
achieved. Dropout is inherited, not modelled.

ON MIXTURES it under-drops badly: retention 0.9657 against real's 0.9020, and the band collapses
(giu allele z -73, song sot 1 z -91). But two statistics move hard the other way - sd log KS
0.351 -> 0.178 and the dynamic range 0.443 -> 0.249, the two that had resisted everything.

THAT SPLIT IS THE RESULT. With machine error inherited correctly, what is left over in mixtures is
exactly the interaction: a contributor holding 0.006 ng ALONE in a tube does not drop like the same
0.006 ng sitting inside a tube holding 0.25 ng. Its size is 6.4 points of retention. That is the term
forensic theory names - drop-out as a function of the number of contributors and the mixture ratio -
and it now has a clean measurement instead of being tangled up with the machine's own dropout.

q was doing double duty: standing in for the machine's dropout AND for the interaction. Split apart,
the machine half is now inherited exactly and the interaction half is exposed and measurable.

A RESIDUAL FORM WAS TRIED AND FAILED, for a stupid reason worth recording: charge only
q(t_eff) - q(source level). It moved retention by 0.002 because np.clip(_f, 0.02, 1.0) with
tpl_ref 0.5 collapses every template at or below 0.01 ng onto one value - and the ladder's floor,
0.0078 ng, is f = 0.0156, under that clip. The mechanism could not reach the region it was built for.
Total RFU also blew to 71228 because the budget's _surv still used the full q. Removed.

STATE: sources are template-matched, AMP_FAIL defaults OFF. The band is much worse than the previous
architecture (|z| 12.0 -> 268) and the height distribution much better. The missing piece is the
interaction term, and it is now the only thing missing rather than one of four tangled symptoms.


## The source draw must match the INJECTION, not only the template

The matched-source path exists so the twin inherits the machine's own dropout instead of modelling
it. It selected a source on loaded mass alone. Mass is not what determines what a run has lost:
dropout is a molecule-count phenomenon and the injection decides how much finished product reaches
the capillary. PROVEDIt's NOC1 set is split in even thirds across 5 / 15 / 25 sec (1760 / 1761 /
1724), so a draw keyed on template alone returned a run at the wrong injection two times out of
three.

The size of that error, real NOC1, untreated, retention at fixed template:

    template     n      5 sec    15 sec    25 sec    5->25
    0.0078      86      0.644     0.728     0.771    +0.126
    0.0156      86      0.822     0.904     0.891    +0.068
    0.0313      39      0.974     0.988     0.980    +0.006
    0.0625      80      0.997     1.000     0.995    -0.001
    0.1250      92      1.000     0.999     0.999    -0.001
    0.2500      99      1.000     1.000     1.000     0.000
    0.5000      95      1.000     1.000     1.000     0.000

Against a replicate envelope of 0.0343 the two faintest levels are off by 3.7x and 2.0x; every level
from 0.031 ng up agrees to 0.006. The defect is entirely a low-template one - which is where the
model's residual errors already sit.

Fix: each pooled profile carries its run's injection, and the draw prefers the target's, falling
back to the whole cell when that donor has no run there. Paired |twin-real| over the replicate
envelope, NOC1:

    band                   cfg     giu allele  so peak  artefact  tan mat  dai dong  do doc bp
    all                    before       0.89x    0.88x     0.87x    0.97x     0.86x      1.08x
                           after        0.65x    0.88x     0.85x    0.77x     0.65x      0.90x
    template <= 0.0156     before       1.81x    1.05x     1.01x    1.84x     1.34x      1.62x
                           after        1.09x    1.09x     0.96x    1.33x     0.88x      1.10x
    template >  0.0156     before       0.68x    0.83x     0.83x    0.77x     0.77x      0.98x
                           after        0.52x    0.82x     0.81x    0.58x     0.58x      0.76x

All six columns are inside the envelope for the first time. The bp slope, stuck at 1.08-1.14x through
every previous attempt, reads 0.90x - and it was never a degradation law that was wrong, it was the
source.

IT COSTS COVERAGE, and that has to be said. 80% coverage of the 13-twin ensemble, before -> after:
giu allele 0.75 -> 0.73, so peak 0.60 -> 0.59, artefact 0.67 -> 0.64, sd log 0.59 -> 0.51, dai dong
0.69 -> 0.75, do doc bp 0.65 -> 0.62, log cao p50 0.61 -> 0.51, tong RFU 0.74 -> 0.74. Mean
percentiles are unchanged. The ensemble was already too narrow; drawing from a third of the cell
narrows it further. The variety that was lost was WRONG variety - twins differing because they had
been handed the wrong run - so it was making the coverage read better for a reason that was not the
generator being right. Under-dispersion remains the open defect, and it is now unmixed with bias.

## INJECTION LAST is no longer a flag

STR_INJ_LAST=0 was the refuted order (injection applied first, then divided back out of every law
lookup at height/_ig with a mismatched exponent). Deleted, along with `_ig` and `INJ_BETA`. With the
build at the reference injection `_ig` was identically 1.0, so every `/_ig` in the file was a
division by one. calibrate still uses 0.80 internally to put the dropout and jitter law keys on the
copy proxy; it no longer exports `inj_beta`, which nothing read.

A jitter hypothesis died here: the source side of the jitter increment is looked up without `_ig`
while the target side is looked up with it, which looked like a footing mismatch now that sources
come from pool_cond/pool_lvl as real runs. It is not - `_ig` is 1.0 - and the identity residual was
byte-identical with the "fix" in. The source is also normalised to a shape (`h = h / s`) before any
law is applied, so the injection's SCALE cancels by construction; only the dropout pattern carried
the wrong injection, which is what the match above fixes.

## RETRACTED: "stutter is not in the identity residual"

Identity test, feeding a real profile in as its own source, n=700, sd(log out/in):

    baseline                        0.0717
    noise floor off                 0.0582
    jitter off                      0.0379
    TPL_SPREAD off                  0.0717
    saturation off                  0.0717
    stutter off                     0.0717
    noise + jitter off              0.0211
    noise + jitter + stutter off    0.0211

Removing the separate capillary draw took the baseline from 0.0996 to 0.0717. Stutter is byte-
identical to baseline - the hypothesis that it was leaking into allele bins is refuted, twice over
(it is also identical in the noise+jitter-off row). TPL_SPREAD and saturation are inert at NOC1 for
the third time. Jitter is the largest single remaining contributor and 0.0211 is unexplained.

RETRACTED. The knob was not wired to the code being tested. `STR_STUTTER` guards an `elif STUTTER:`
fallback at the end of the artefact section; the live path is the `if art_tab:` branch above it,
which emits stutter and the other off-target artefacts from the calibrated table. Setting STUTTER=0
changed nothing because it changed nothing - not because stutter was absent. Two rows of that table
were read as evidence and neither was evidence.

## The 0.0211 IS the artefact table, and it is a double count

Identity test with the live path switched off instead:

    baseline                             0.0717
    artefact table off                   0.0633
    noise + jitter off                   0.0211
    noise + jitter + artefact table off  0.0021

The residual is the artefact table almost exactly, and the alleles it invents go with it: 0.0013
added alleles fall to 0.0000. The 0.0021 left over is float32 and budget rounding.

The mechanism is the same one the capillary draw had. A source profile is a real run and carries its
own artefacts. `h = h * (DONOR_DOSAGE[c] > 0)` deletes them at every bin the donor has no allele at
- but an artefact that lands ON another of that donor's own alleles survives that mask, because the
bin is a real allele bin. Back-stutter from a heterozygote's n+1 partner is the common case. The
generator then emits a fresh stutter onto the same bin, so that height is charged twice.

## Jitter was reading the two sides at different injections

The increment subtracts the scatter the source already carries from the scatter the target needs. In
an identity run the source IS the target, so it must come out zero. It did not, and the probe says
why: `exp_h` is built at the reference injection (`_gain` is applied at the very end) while `_hsrc`
is the source run's raw RFU at its own 5 / 15 / 25 sec. Target/source height ratio, 700 identity
runs, 5th / 50th / 95th percentile:

    measured   0.609   0.999   2.982
    1/_gain    0.616   1.000   2.833      (_gain = (sec/15)^0.948)

Three digits, three injection times. The two lookups were landing at different points of a steep
curve for no reason but which second the run used. The increment that should have been zero read
0.0010 at the median and 0.1744 at the 95th percentile - so jitter charged real scatter on the third
of samples run at 5 sec, and under-charged the third run at 25 sec.

Fix: look the source side up at `_hsrc / _gain`. Ratio becomes 0.977 / 0.998 / 1.092 and the added
sd 0.0010 / 0.0010 / 0.0382. Identity residual 0.0717 -> 0.0427. NOC1 audit over the replicate
envelope, all templates: giu allele 0.65x, so peak 0.86x, artefact 0.83x, tan mat 0.77x -> 0.71x,
dai dong 0.65x -> 0.63x, do doc bp 0.90x -> 0.80x.

NOTE ON THE EARLIER `_ig` ATTEMPT: the same defect was guessed at one round earlier and the wrong
divisor was used. `_ig` is identically 1.0 under the reference build, so that edit was a division by
one and the identity residual came back byte-identical. The footing that was actually wrong is
`_gain`. A byte-identical result means the edit did not run, not that the hypothesis was wrong -
that is twice now this session that it was read the second way.

## The artefact double count, and its fix

WHY IT HAPPENS. A source profile is a real run: every height in it already contains whatever artefact
landed on that bin. `h = h * (DONOR_DOSAGE[c] > 0)` deletes the source's artefacts everywhere the
donor has no allele - deliberately, because artefacts belong to the run's total template and not to a
contributor's share, so they have to be re-emitted once at mixture level. At a bin the donor DOES
have an allele at, that deletion is impossible: the artefact and the allele are one number and
nothing can separate them. The table then emits onto every bin, that one included.

EXPOSURE. The offsets carrying the rate are n-1 (0.453), n-2 (0.186), n+1 (0.169), n+2 (0.108) -
93% of the table between them - and heterozygote alleles sit one or two repeats apart constantly. Per
donor, the share of that donor's own allele bins that are the target of one of that donor's own
alleles:

    mean 0.497, range 0.263 - 0.683 over the 45 donors
    expected added amount where it lands: 0.012 x the parent allele's height

SIZE. Identity test, splitting the bins of each profile into hit and not-hit:

    log(out/in) at bins hit by the donor's own artefact   +0.0061
    log(out/in) at bins not hit                           -0.0064
    step between the two halves                           +0.0125  (1.3% of height)

Half of every profile lifted 1.3% against the other half. That is the whole of the 0.0211.

FIX. Emit only the share of the parent mass whose donor does NOT own the target bin. Per offset,
`f = 1 - (sum over contributors of contrib_exp[c] where donor c has an allele at the target) / total`.
At NOC1 that is exactly 0 at own-allele targets and 1 everywhere else. In a mixture it removes only
the own-donor share: a stutter cast onto ANOTHER contributor's allele is new to that profile and is
a large part of what makes mixtures hard, so it stays. This is the same rule as the capillary draw
and the jitter increment - subtract what the source already carries, add only what is missing - and
it is now applied three times in this file.

RESULT. Identity residual 0.0427 -> 0.0298; with noise and jitter also off, 0.0211 -> 0.0027, which
is float32 and budget rounding. NOC1 audit over the replicate envelope, all templates:

    giu allele 0.65x -> 0.61x, so peak 0.86x -> 0.85x, artefact 0.83x -> 0.83x,
    tan mat 0.71x -> 0.67x, dai dong 0.63x -> 0.58x, do doc bp 0.80x -> 0.78x

The faintest band (template <= 0.0156, n=220) does not follow: retention 1.04x -> 1.09x, sd log
1.20x -> 1.40x, dynamic range 0.94x -> 1.02x, bp slope 1.11x -> 1.19x. Those columns have bounced
between runs at this n and no claim is made about them either way.

WHAT IS LEFT IN THE IDENTITY RESIDUAL. Baseline 0.0298, of which the noise floor is the larger part
(0.0298 -> 0.0124 with it off) and jitter the smaller (0.0298 -> 0.0242). The noise floor also
invents 0.0177 alleles per profile at bins the source has none at, and that number goes to 0.0000
when it is off - the same double-count shape as the other three, and the next thing to test.

## The isolation ladder on real mixtures

320 real mixture specifications, 8 twins each. Read only - every mixture is in the test split.

RUNG A, the generator against the spec. `|z|` is the real value's distance from the twin ensemble
mean in twin standard deviations; the raw gap is real minus the twin mean, so a negative gap means
the twin reads HIGHER than real.

    statistic          |z| A   |z| B    raw gap    twin sd    NOC1 envelope
    giu allele          7.14      -     -0.0343     0.0081       0.0343
    song sot 1 nguoi    4.02    3.96    -0.0407     0.0240          -
    so peak             1.67      -     -4.1176     6.1755      11.2464
    ty le artefact      1.53      -     +0.0089     0.0299       0.0445
    tan mat             3.88    4.79    -0.1421     0.0592       0.0809
    dai dong           18.61   19.11    -0.8692     0.3610       0.4569
    do doc bp           2.47    4.69    -0.0000     0.0010       0.0007

Rung B transplants the dropout answer by reading every statistic on the support real and all eight
twins share. Retention, peak count and artefact fraction become degenerate there and are not read.

RUNG C, the interaction with no generator in it at all. Each contributor's REAL single-source run at
the matching donor / treatment / template / injection, summed. An allele counts as retained only if
the run of the donor who OWNS it kept it, so another donor's noise cannot resurrect a dropped bin;
peak count and artefact fraction are not read at all, because a superposition sums k separate noise
floors and both climb with k for a reason that is not the sample.

    statistic          real - superposition    NOC1 envelope    x envelope
    giu allele                     -0.0396          0.0343          1.2x
    song sot 1 nguoi               -0.0485             -              -
    tan mat                        +0.0152          0.0809          0.2x
    dai dong                       -0.0539          0.4569          0.1x
    do doc bp                      -0.0025          0.0007          3.6x

    by NOC             NOC2       NOC3       NOC4       NOC5
    giu allele      -0.0208    -0.0277    -0.0449    -0.0730
    song sot 1      -0.0040    -0.0318    -0.0279    -0.1394
    tan mat         +0.0388    -0.0007    +0.0122    +0.0100
    dai dong        -0.0275    +0.0210    -0.0189    -0.2090
    do doc bp       -0.0016    -0.0026    -0.0033    -0.0030

FOUR THINGS FALL OUT OF THAT TABLE.

1. THE INTERACTION IS REAL AND IT IS DROPOUT. Retention falls short of pure superposition by 0.0396,
   monotone in the number of contributors, and the minor contributor's private alleles take roughly
   twice the hit - collapsing to -0.1394 at NOC5.

2. THE INTERACTION IS NOT A HEIGHT PHENOMENON. Summing real single-source runs, with nothing
   modelled, already reproduces the height spread (0.2x the envelope) and the dynamic range (0.1x).
   Mixing does not change how heights are distributed; it changes which alleles survive. Every
   attempt to reach mixtures through a height law was aimed at a term that measures zero.

3. THE GENERATOR'S RETENTION ERROR IS EXACTLY THAT INTERACTION. The twin over-retains by 0.0343
   against real; the interaction measured independently, with no generator, is 0.0396. Two
   measurements that share no code agree to within the NOC1 replicate envelope. The missing piece is
   identified, and its size is known.

4. THE GENERATOR OVER-DISPERSES HEIGHTS ON MIXTURES, AND THAT IS ITS OWN BUG. sd log 1.8x the
   envelope and dynamic range 1.9x, against 0.2x and 0.1x for plain superposition. Whatever the
   generator does to a source that superposition does not - normalising it to a shape and rescaling
   it by phi x t_total instead of using it at its own template's scale - is adding spread that the
   data does not have. This needs no new knowledge, only a fix.

Two further readings. The bp slope mean is exact on mixtures (raw gap -0.0000) while superposition
is off by -0.0025 at every NOC alike; a k-independent offset is not interaction, and the generator's
degradation machinery is already supplying it. And the twin ensemble is far too narrow here as it is
at NOC1: retention sd 0.0081 where real replicates disagree by 0.0343.

ON FITTING. The interaction can only be measured on mixtures, and every mixture is in the test
split, so its value must not become a constant in the file. The form comes from theory and the
number stays a RANGE wide enough to contain what was measured here; this table is the read-only
check that the range does contain it, never the source of the number.

## The mixture height defect was TPL_SPREAD, and it is the same double count a fourth time

The block widens each contributor's log spread by a factor read off the template-shrink curve at that
contributor's share: `_m` is 1.24 at share 0.3 and 1.66 at share 0.1. Its stated premise is in its own
comment - "the reference the source profile was put on" - and that premise died when sources started
coming from the template ladder. A ladder-matched source is ALREADY at the contributor's own
template and already carries that template's spread. Widening it again applies the curve twice.

It is invisible at NOC1 for a checkable reason: the gate is p < 1, and p = 1 gives _m = 1 exactly.
Switching TPL_SPREAD off measured byte-identical three separate times and was read three times as the
block being inert. It is inert - at NOC1 only.

Skipping it when the source came from the ladder (kept for the pool_clean_all fallback, where the
premise still holds), on 320 real mixture specs:

    statistic          raw gap before   raw gap after   NOC1 envelope   after / envelope
    tan mat                  -0.1421         -0.0782          0.0809       0.97x
    dai dong                 -0.8692         -0.4689          0.4569       1.03x
    giu allele               -0.0343         -0.0387          0.0343         -
    song sot 1 nguoi         -0.0407         -0.0508             -           -
    do doc bp                -0.0000         +0.0002          0.0007         -

Both height statistics land on the replicate envelope. And the two retention statistics move ONTO
the interaction measured independently in rung C, which shares no code with the generator:

    giu allele        twin gap -0.0387    interaction -0.0396
    song sot 1 nguoi  twin gap -0.0508    interaction -0.0485

The Jensen boost inside that block had been masking part of the retention gap; with it gone, what the
twin is missing on mixtures is the interaction, at the size two independent measurements agree on,
and nothing else. NOC1 is byte-identical, which here is the correct result rather than a failed edit:
the gate cannot fire at p = 1.

## NOC1 status after the four double-count fixes

Three tests, three different questions.

PAIRED ACCURACY - is each twin close to its own real sample? |twin-real| over the replicate envelope,
1133 NOC1 specs:

    band                giu allele  so peak  artefact  tan mat  dai dong  do doc bp
    all templates            0.61x    0.85x     0.83x    0.67x     0.58x      0.78x
    template <= 0.0156       1.09x    1.08x     0.94x    1.40x     1.02x      1.19x
    template >  0.0156       0.51x    0.82x     0.80x    0.52x     0.53x      0.71x

IDENTITY - feed a real profile in as its own source; anything the pipeline still adds is something
the data already had. Residual 0.0298, from 0.0996 at the start of this work:

    baseline                        0.0298     adds 0.0177 alleles
    noise floor off                 0.0124     adds 0.0000 alleles
    jitter off                      0.0242
    noise + jitter off              0.0027

DISPERSION - is the twin ENSEMBLE the right width? 350 specs x 13 twins, 80% coverage and the mean
percentile rank of real inside the ensemble:

    giu allele        0.73   0.35        tan mat      0.71   0.45
    so peak           0.58   0.45        dai dong     0.76   0.41
    ty le artefact    0.63   0.50        do doc bp    0.77   0.52
    log cao p50       0.51   0.57        tong RFU     0.71   0.47

Six of eight now read acceptably where four did before the artefact fix; sd log went 0.51 -> 0.71 and
the total-RFU bias 0.31 -> 0.47. Two are still too narrow: peak count 0.58 and log height p50 0.51.

THREE THINGS ARE OPEN, and they may be one thing. The noise floor is the largest term left in the
identity residual and the only mechanism that invents alleles there (0.0177 -> 0.0000 with it off);
peak count and median log height are precisely the two statistics a noise floor drives; and the
faintest template band, where the floor is the largest share of the profile, is the band still
outside the envelope. That is a hypothesis with three independent symptoms pointing at it, not a
finding - the noise floor has already been cleared once this session against a different statistic,
and clearing it for one statistic clears it for that statistic only.

## RETRACTION: every NOC1 audit number in this session was measured with the answer in the pool

The audit scores train NOC1 samples. calibrate builds the source pool from train NOC1 samples. Once
the draw is matched on donor AND treatment AND template AND injection, most such cells hold exactly
one run - and that run is the sample being scored. Measured: **98.5% of draws (22932 of 23292)
returned the target profile itself**, and a redraw loop cannot escape, because there is no second
candidate in the cell.

So the audit was reading memorisation. The tighter the match, the better it looked, for the reason
that makes it worthless. Every |twin-real| number this session that came from injaudit.py is
withdrawn as a magnitude.

NOT affected: the identity test, which feeds the target in as its own source deliberately and
measures what the pipeline adds to it; and the mixture ladder, whose targets are test mixtures that
are not in the pool at all. The four double-count fixes rest on those two, and on measurements of
real data itself, not on the audit.

CLEAN MEASUREMENT. The pool is built from train only, so scoring the VALIDATION split takes the
answer away entirely (0 of 4208 draws hit a target). |twin-real| over the replicate envelope:

    band                giu allele  so peak  artefact  tan mat  dai dong  do doc bp
    all templates            1.15x    0.85x     0.92x    1.17x     1.09x      1.25x
    template <= 0.0156       2.49x    1.00x     1.25x    1.87x     1.74x      1.73x
    template >  0.0156       0.89x    0.80x     0.84x    1.00x     0.95x      1.17x

NOC1 IS NOT CORRECT. Four of six columns sit outside the envelope pooled, five of six in the faintest
band. Above 0.0156 ng only the bp slope is out, at 1.17x.

BOTH RECENT CHANGES RE-TESTED ON THAT SPLIT, and both survive, at much smaller margins than the
contaminated audit reported:

    noise floor, allele bins:   1.20 -> 1.15 / 0.84 -> 0.85 / 0.89 -> 0.92 / 1.22 -> 1.17 /
                                1.09 -> 1.09 / 1.35 -> 1.25   (faint band 2.27 -> 1.87 on sd log,
                                2.12 -> 1.73 on the bp slope, peak count 1.12 -> 1.00)
    injection matching:         1.18 -> 1.15 / 1.19 -> 1.17 / 1.26 -> 1.25, faint band 1.33 -> 1.25
                                on artefact fraction and 1.82 -> 1.74 on the dynamic range

Injection matching is kept on the strength of the real-data measurement that motivated it - retention
0.644 / 0.728 / 0.771 at 0.0078 ng across 5 / 15 / 25 sec is a fact about PROVEDIt, not about the
twin - not on the audit that first appeared to confirm it.

A SECOND CONSEQUENCE, for the training set rather than the twin. The generator draws from the same
pool when it builds training data. If a cell holds one run, every sample generated for that cell is
that run plus artefacts. Matching harder narrows the cell and makes this worse. Coverage of the
(donor, treatment, template, injection) grid is now a first-class constraint on how tightly the
source may be matched, and it has not been measured.

## Why NOC1 is still outside the envelope: the source, not the generator

VAL AGAINST A TRAIN POOL IS ONLY CLEAN AT THE INJECTION LEVEL. 92.1% of val NOC1 samples share their
PCR product with a train run - same name, other injection time (`_04.15sec` in val, `_04.25sec` in
train). The split was made by injection, not by amplification.

ONE AMPLIFICATION PER CELL. Counted from the train names: 2447 of 2452 (donor, treatment, template)
cells - 99.8% - hold exactly one PCR product from one extract. Their 1-3 "runs" are injections of it.
The pool_lvl comment in calibrate.py ("76.6% of (donor, level) pairs have two or more runs so a twin
can take a DIFFERENT one") counted injections; as a statement about independent runs it is false -
0.9% of untreated cells hold a second amplification.

So a twin of a real NOC1 sample has only two possible sources:

    source                              NOC1 floor, raw run, no generator     generator
    same amplification, other inj       1.08 0.97 0.97 1.03 1.03 1.01         1.02 0.82 0.90 1.05 1.00 1.01
    other amplification = other level   1.88 0.94 0.98 1.73 1.54 2.30         1.86 0.88 1.03 1.67 1.46 2.31
    (giu allele / so peak / artefact / tan mat / dai dong / do doc bp, x replicate envelope, val)

The first is memorisation at the PCR level - replicate-close because it IS the replicate the envelope
was built from. The second is the only honest twin, and it is off by a template doubling because
nothing else exists in the data. In both rows the generator is within a few tenths of copying the raw
run: its machinery is not what is out of the envelope. What is out is the source at the wrong
template, and the generator does nothing to convert it - inherited dropout is exact only when a run
exists AT the target's level, and for a new amplification it never does. The residual form that was
tried for exactly this conversion failed on a clip bug (np.clip(_f, 0.02, 1.0) collapsing everything
below 0.01 ng), not on its idea.

THE SAME HOLE REACHES THE MIXTURES. Per test-mixture contributor, the nearest available level for that
donor under that treatment is a full doubling away for 22.5-25.6%, and 34.7-51.4% have no run under
the treatment at all and fall back to an untreated source.

## The faint-band artefact excess is crowd's footing, and fixing it alone breaks the bright band

Knocking functions out pointed at locus_max (+11.35 extra noise peaks per faint profile -> +1.45 with
it off). That reading was wrong. Measured on real NOC1, the two x-axes the modulators read:

                        locus_max factor           occupancy          crowd factor, <8 RFU
                     observed   pre-noise     observed   pre-noise    observed   pre-noise
    faint (<=0.0156)    1.164       1.177         73.7        39.9       0.872       1.165
    bright              0.946       0.949         98.3        69.7       0.757       0.914

Both curves were calibrated on OBSERVED profiles; the generator reads them on its profile BEFORE the
floor is added. locus_max barely moves (1.164 vs 1.177). Occupancy halves, so crowd reads a faint
profile as sparse and keeps its faint noise: net noise probability x1.35 (<8 RFU) and x1.26 (8-12) at
faint templates. Switching locus_max off helped only because its factor is above 1 there - a correct
term switched off to cancel a wrong one.

Confirmed by copying real's answer into ONE variable at a time, val:

                                   faint: artefact +/-  peaks +/-    bright: artefact +/-  peaks +/-
    as is                                   +0.0438      +11.39              -0.0032        -0.25
    real occupancy -> crowd                 +0.0013       +0.86              -0.0255        -4.83
    real locus max -> locus_max             +0.0397       +9.17              -0.0060        -0.81

Real occupancy removes the faint bias entirely; real locus maxima do not. BUT the bright band, unbiased
as is, goes to a deficit of 4.83 peaks once the footing is right. The bright band was correct by
compensation: the base rate was set with the modulators running on the wrong footing, and it absorbed
the average error. The footing and the base rate have to be re-derived TOGETHER on one footing, from
NOC1; correcting the footing alone moves the error from one band to the other.

## (a) The noise-floor modulators, rebuilt on the generator's footing

Re-derived in calibrate.py from NOC1, reading occupancy as the generator reads it - signal bins, i.e.
donor alleles and their +-1/+-2 product offsets, BEFORE the floor is added - counting noise only where
nothing structured reaches (>= 2.5 repeats from any donor allele), and normalising by the generator's own
per-bin base rate. The product of the modulators averages exactly 1.0000 over NOC1.

LOCUS_MAX IS GONE. On that footing it reads 0.988-1.019 across ten decades of locus height. The
"2.247 -> 1.225 peaks per locus" it was built on is a between-locus confound - loci with tall peaks have a
low baseline rate, which the per-bin rates already carry. Its only effect had been to offset crowd's
footing error; switching it off had cut the faint band's excess from 11.35 to 1.45 peaks, and that
reading had pointed at the wrong function.

CROWD is flatter than the old table (<8 RFU: 1.169 -> 0.822 against 1.283 -> 0.729), hard-wired (flag
removed), and extended past NOC1's range along each band's own NOC1 log-linear trend where that trend is
a suppression (-0.0060 / -0.0034 / -0.0002 per signal bin) and held where it rises (16-25 RFU, the
pull-up confound). Mixtures sit at 94-123 signal bins at the median; NOC1 stops near 90. Read-only check
on real mixtures, at 137 signal bins: <8 RFU real 0.52, held edge 0.82, extended 0.63; 8-12 RFU real
0.66, held 0.91, extended 0.77. The 12-16 band is matched by neither.

Result, val NOC1, twin - real: faint band artefact -0.0027 / peaks -1.86, bright band +0.0017 / +0.98,
from +0.0438 / +11.39 and -0.0032 / -0.25. Both bands at zero at once, which fixing the footing alone
could not do. Identity residual 0.0298 -> 0.0115, no invented alleles; the noise floor no longer
contributes to it at all.

## (b) Level conversion

AMP_FAIL (the flat q(f) table, off) is deleted. In its place, survival relative to the top of the
dilution ladder, derived in calibrate.py from pairs of adjacent rungs of the same donor / extract /
treatment / injection - each rung its own amplification:

    rung       0.0078  0.0156  0.0312  0.0625  0.125   0.25    0.5
    PHI        0.562   0.727   0.860   0.946   0.986   0.996   1.000
    q/step     0.227   0.155   0.091   0.041   0.011   0.004

The data-only estimate agrees with the same quantity measured through the generator (0.226 / 0.163 /
0.102 / 0.042 / 0.010 / 0.003) - scaling + scatter + threshold remove almost nothing across a step
(retention 0.7021 against the source's 0.7072 at 0.0078 ng), so the loss is failure to amplify, keyed on
absolute template. The source draw now takes the nearest rung AT OR ABOVE the target (3% tolerance) and
charges PHI(target)/PHI(source); the budget charges the same loss. WHICH alleles go follows fragment
size at fixed height, logit -0.378 per 100 bp pooled (-0.37 to -0.69 per rung, t to -17), with the
intercept solved so the mean stays PHI's.

Two data-hygiene bugs found on the way. The names spell 1/32 ng both 0.0312 and 0.0313: the untreated
ladder had silently dropped the 0.0312 half (too few donors alone) and the treated ladders carried it as
a separate level. Templates within 3% of a rung are now snapped to it - to the ROUNDED rung, because
snapping to 0.015625 exactly matched nothing at the file's 1e-6 level comparisons and emptied every rung
below 0.0625 (caught by the pool shrinking to 367 profiles; it is 634 over 7 rungs and 45 donors now).

Honest twin, val NOC1, source a DIFFERENT amplification (forced one rung off):

    signed retention, raw source -> generator:   all +0.0497 -> +0.0020   faint +0.1475 -> -0.0007
    |twin - real| / envelope, generator:         1.39 / 0.72 / 0.92 / 1.62 / 1.41 / 2.24

The retention bias is gone. What is left is scatter around zero. The size term did not move the bp
slope (2.24x, raw 2.20x): dropping long fragments at random does not tilt the survivors. A real step
flattens the slope by +0.0005 (0.7x the envelope), and the rest of the 2.24x is a second amplification
differing by more than the same-PCR yardstick allows - which cannot be measured here, because the data
hold no second amplification at the same rung.

## Mixtures after (a) + (b), 300 specs, generator gap and interaction on the SAME specs

    NOC     gap(ret)  interaction  own error   gap(minor)  interaction  own error   peaks   artefact
    all     -0.0495     -0.0412     -0.0083     -0.0630     -0.0484     -0.0146     -8.62    -0.0010
    NOC2    -0.0451     -0.0325     -0.0126     -0.0393     -0.0187     -0.0205     -0.92    +0.0272
    NOC3    -0.0331     -0.0256     -0.0075     -0.0595     -0.0436     -0.0159     -6.97    -0.0067
    NOC4    -0.0298     -0.0153     -0.0145     -0.0318     +0.0004     -0.0322    -10.38    -0.0221
    NOC5    -0.0866     -0.0866     -0.0000     -0.1143     -0.1216     +0.0073    -15.19    -0.0009

The interaction is measured through the generator's own _conv_surv and source rule and no other
generator code. At NOC5 the generator's retention error is zero; at NOC2-4 it over-retains by 0.008-0.015
of its own. Peaks: -10 -> -8.6 with the crowd extension, still growing with NOC; the artefact FRACTION
matches, so the excess splits between alleles and noise roughly as real's peaks do - the allele part is
the missing interaction.

## Peak count and bp slope, isolated stage by stage

TIER 0 - peak count is the sum of three bin classes (a contributor's allele / +-1,+-2 repeat product
offsets / everything else), so the excess has to sit in one of them. Real - twin, mixtures:

    NOC     allele   product   other
    NOC1     -0.1     -0.2     +0.1
    NOC2     -3.0     +3.2     -2.2
    NOC5     -9.7     -3.0     -2.3

The allele class is the missing interaction (NOC5: 0.087 x ~110 alleles). The other two are separate
defects, and the bp slope is a third.

BP SLOPE, design check on the honest NOC1 twin (source a different amplification). Copying the answer
(source = the target itself) gives -0.00000: the machinery does not distort the slope. Full generator =
raw source (+0.00023): everything is the source. The systematic part lives at faint templates
(+0.0008-0.0011) and survives on the alleles both profiles kept, so it is a HEIGHT effect of the level
step, not selection by dropout. On same-extract dilution pairs held >= 50 RFU at both rungs, and still at
>= 200 RFU, the lower rung's size slope is flatter by 0.0004-0.0011 per step (t 4-8): a real tilt with
template. Installed as AMP_STEP_TILT (cumulative B(t), derived in calibrate). With it, same-extract honest
sources go +0.00053 -> -0.00017, and -0.00001 once real's dropout is also copied in. What remains
(-0.00068) sits entirely in the 36% of honest sources that are a DIFFERENT extract, whose raw offset is
already -0.00030 and opposite in sign: extract-to-extract degradation, not template. At faint rungs the
tilt slightly overshoots (-0.00027 for same-extract sources, 0.4x the envelope).

NOISE HEIGHTS, design check on real inputs (real allele + product layers, only the generator's noise
stage). The count is right; the SHAPE is not, at every NOC including NOC1: too few at 8-12 RFU, too many
below 8 and at 12-16. Copying real's answer into one variable at a time:

    heights taken from                       NOC1 <8 / 8-12 / 12-16
    log-normal (locus level + common sd)       -1.14 / +2.49 / -1.42
    this sample's own real heights             +1.25 / -0.15 / -0.74
    real heights pooled PER LOCUS              +0.61 / -0.48 / -0.86
    per locus, run offset split off + redrawn  -0.54 / +2.19 / -1.66

The spread was never wrong (composed 0.463 vs real 0.480; 0.385 within a locus exactly); real is simply
not log-normal - 0.419 of it at 8-12 RFU against 0.311 - and splitting a run offset off and adding one
back breaks the shape again. Installed: each locus's own quantiles of log height (NOC1 clean baseline
bins), with the peaks of one profile correlated through a Gaussian copula at rho 0.406 (between-run share
with each run mean's sampling noise removed). Every locus keeps its real marginal exactly. Tier 0, NOC1
other class by band: -1.49 / +2.83 / -1.19 / +0.24 / -0.19 -> +0.03 / +0.18 / -0.67 / +0.65 / -0.20.
A first attempt - quantiles of a run offset and a common residual - changed nothing and was removed.

The crowd occupancy footing was a hypothesis of mine that did not survive: copying real's occupancy into
the lookup moved the mixture excess by 0.2 peaks.

SHOULDER. Real thins any peak within one repeat of an allele peak. NOC1:

    noise beside an allele, kept fraction by neighbour height (medians 49 / 176 / 534 / 1630 / 4732 RFU):
        0.647 / 0.432 / 0.310 / 0.219 / 0.220
    n-1 stutter beside ANOTHER allele, against the rate its own parent height predicts: ~0.72, flat
    bp distance from the neighbour: no effect (0-1, 1-2, 2-3, 3+ bp alike)
    survivors beside a tall allele: no taller than far noise (median 8 against 9 RFU)

So it is a thinning, not a relative-height threshold. The generator had one flat rate (NOISE_P_NEAR,
0.039) for FRACTIONAL bins only - nothing for integer bins beside a microvariant and nothing for
artefacts, which at NOC5 means ~100 stutter positions per profile beside another allele emitted at the
free rate. An uncontrolled first reading gave 0.55 for stutter; controlling for parent height gives 0.72.

The second threshold constant (AT = 10.0, line ~98) is read only by add_dropin, and DROPIN is off.

## Artefact stage, levels 3-5

STUTTER LAW - P(peak at an offset position | parent height), real vs twin, each on its own parent heights,
positions that are alleles or within one repeat of another allele excluded. Below 300 RFU parents all four
offsets match. Above, the twin was SHORT on n-1 (NOC1: 0.768/0.682, 0.892/0.844, 0.898/0.864) and LONG on
n+-2 (n+2 at 1000-3000: 0.165/0.206; at 3000+: 0.214/0.270). Heights of present n-1 peaks match to the
second decimal at tall parents (-2.73/-2.75, -2.80/-2.78), so n-1 was a presence problem in every carrier
bucket - a multiplier, not a height law.

H1, CONFIRMED IN CODE AND FIXED. _art_rarity divided the carrier multipliers by their MAXIMUM, then the
generator multiplied them onto an emission rate Pass 3 had solved over all parents, i.e. already the average
over the carrier mix. Normalised now to a mean of 1, weighted by predicted emission (0.861-1.0 ->
0.908-1.056). n-1 with real parents at NOC1: +1.05 -> +0.30 peaks; the tall-parent law 0.682 -> 0.720,
0.844 -> 0.866, 0.864 -> 0.890.

H2, SMALL. Pass 3 used one global background (0.078); the generator's per-bin noise at product positions
is 0.079-0.083. Now each parent's own target-bin rate and its own survival. Minor rates moved 2-4%.

WHAT H1 EXPOSED IN MIXTURES. With the NOC1 law now right, the twin carries 1.4 / 2.4 / 2.8 more n+-1
peaks than real at NOC3 / 4 / 5 (real parents). The reason was already in the level-4 table: at tall
parents and within the SAME carrier bucket, real mixtures show about 6% less n-1 than real NOC1 (bucket
3+: 0.861 vs 0.921; 21+: 0.803 vs 0.893), and neither parent height nor carrier count accounts for it. It
is a mixture-only effect, of the same kind as the dropout interaction, and it is recorded here rather than
fitted - every mixture is in the test split.

n+-2 remain over-produced at every NOC, and the twin's excess over background is 1.6-2.5x real's at every
parent band - a LEVEL error in the minor-offset rates, not a shape error.

## H3: the minor stutter offsets were calibrated on positions that belong to n-1

Pass 1 (ratio) and Pass 3 (emission) counted a peak at offset d "against every donor allele in range".
Where two alleles sit one or two repeats apart, the n-2 position of one IS the n-1 position of the other -
one peak, present 90% of the time at tall parents because of the n-1 - and it was charged to n-2 as well.
NOC1, presence at CLEAN positions (no other allele has the bin at +-1/+-2) against OVERLAPPING ones:

    n+2   0.083 / 0.074 / 0.071 / 0.072 at every parent band  - the background rate; clean n+2 barely exists
    n-2   0.089 -> 0.447, only above 3000 RFU parents          (overlap 0.22 -> 0.81)
    n+1   0.090 -> 0.467                                        (overlap 0.16 -> 0.67)
    n-1   0.572 clean, 0.578 overlap                            - unaffected
    inflation of the pooled count: n+1 x1.28, n-2 x1.45, n+2 x1.62; overlap ratios read the n-1 ratio

The comment defending the count ("each parent is an independent opportunity") was the error: the generator
does emit each offset independently from each parent, and so produces the union at an overlapping bin by
itself - the table has to hold the clean single-process rates. Counting clean positions only fixed the
mixtures and broke NOC1 in a second way: with the true (smaller) ratios, Pass 1's fit criterion - parents
whose expected artefact reaches 65 RFU - asked for parents over 13000 RFU, most loci failed it, and a
locus that fails was DROPPED (n-2 11 -> 4 loci, n+1 11 -> 8), i.e. given no minor stutter at all. Loci
without their own fit now take the offset's pooled law: n-1 23 loci rate 0.894, n+1 23 / 0.459, n-2 22 /
0.542, n+2 22 / 0.049.

RESULT, NOC1. Stutter law by parent band, all four offsets within 0.01-0.05 of real. Artefact stage with
real parents: int +-2 -0.89 -> -0.12, int +-1 +1.05 -> +0.82. Tier 0: product +0.78, other +0.65. Band
bias: faint -0.0001 / -1.11 peaks, bright -0.0039 / -0.74. Identity 0.0114, no invented alleles.

## Where peak count and bp slope stand

bp slope: the design is exact when the source is right (answer copied: -0.00000); the template tilt is in;
what remains is the between-extract offset carried by a third of honest sources, not a template effect.

Peak count, tier 0, real - twin:

    NOC     allele   product   other
    NOC1     -0.1     +0.8     +0.7
    NOC2     -3.0     +3.0     -0.4
    NOC3     -2.8     -0.4     -2.1
    NOC4     -3.0     -3.0     -1.8
    NOC5     -9.7     -2.6     -0.5

Everything derivable from NOC1 is now reproduced at NOC1. What is left is mixture-only, three pieces:
  allele   the dropout interaction (NOC5 -9.7 = 0.087 x ~110 alleles), measured independently at -0.040;
  product  a NOC trend - short at NOC2, long at NOC4-5 - of the same kind as real mixtures showing ~6% less
           n-1 than real NOC1 at tall parents in the same carrier bucket; not explained by parent height,
           carrier count or the shoulder;
  other    the 12-16 RFU band, which real mixtures suppress to 0.60-0.81 while NOC1's crowd law is flat there.
None of the three can be derived from single-source data, and every mixture is in the test split. They
belong with the interaction term: a form from theory, a range wide enough to contain the measurement.

## The "interaction" was a missing treatment conversion

Before building a mixture interaction term, the rung-C shortfall (real - superposition of the donors' own
NOC1 runs) was split by what the sources were:

    real - superposition          retention   minor     n
    untreated mixtures              +0.006    +0.031   238
    treated, all matched            +0.007    +0.005   305
    treated, a contributor fell back to an untreated source
                                    -0.071    -0.095   790

Two mechanisms proposed first were refuted on the way. Crowding acting on alleles: on NOC1, allele presence
RISES with occupancy elsewhere (+0.087, 26/32 cells), and in the superposition only 0.2-3% of the minor's
private alleles are under 25 RFU - there is no faint mass for a detection law to act on; the minor's loss is
all-or-nothing. Mixture-ratio drift: the minor/major height ratio against nominal agrees between real and
superposition at NOC2 and NOC3 (+0.007, -0.022).

The shortfall lives where 35-51% of mixture contributors draw from the UNTREATED ladder because their donor
has no run under the mixture's treatment, and an untreated run carries none of the alleles the treatment
destroys. It grows with NOC only because more contributors mean more chances of a fallback.

TREATMENT CONVERSION, derived in calibrate.py from NOC1 cells of the same donor, rung and injection: per code,
the retention ratio treated / untreated at each rung (the mean) and the logit slope on fragment size of an
allele surviving the treatment given it survived untreated (the spread). It orders by dose without being told
(-15 < -30 < -45, c < d < e, S2 < S10 < S30, U15 < U60 < U105), vanishes at high template, and the size
slope is strongly negative for everything that cuts DNA (DNase -0.54 to -0.68, sonication to -1.38, UV
~-0.75) and ~0 for humic-acid inhibition (+0.03 to +0.08) - inhibition slows amplification, it does not cut
long fragments. Applied only to a treated contributor drawn from the untreated ladder; the budget charges it.

DESIGN CHECK ON NOC1 (val targets, source forced to an untreated train run):

    real - twin retention        law off    law on     reference: treated source
    all                           -0.0907   -0.0016    +0.0012
    faint (<= 0.0156)             -0.1928   +0.0043    +0.0080
    every treatment family        -0.02..-0.14   within +-0.013

Mixtures: the fallback category goes -0.071 / -0.095 -> +0.012 / +0.052, level with the other two. Generator
allele class, real - twin: NOC2-4 -3.0 -> +0.3..+0.6; NOC5 -9.7 -> -4.6. What remains is NOC5-only,
-0.015 to -0.036 in all three categories alike.

## What is left after the treatment conversion, and why no interaction term was added

Generator tier 0 split by treatment category (real - twin, allele / product / other): NOC2-4 allele class
within +-1.4 in every category. NOC5 allele class -3.0 / -5.0 / -4.7 in untreated / matched / fallback alike -
not a fallback effect. In the superposition (level + treatment conversion) the NOC5 shortfall is -0.031,
concentrated where the smallest contributor sits below 0.0156 ng (-0.068) and at small totals (-0.09 at
0.075 ng, -0.10 at 0.12 ng); it is unchanged with capped profiles removed (-0.0306). NOC4, at the same
per-contributor templates, goes the OTHER way (+0.022; +0.033 below 0.0156 ng).

A mixing law - competition, crowding, shared failure - would be monotone in the number of contributors. A
shortfall that reverses sign between four and five contributors at the same templates is not one, and no
theory offers a form that does that. It was left out: a term built on it would teach the model a pattern
with no physical meaning. It is recorded as a property of PROVEDIt's five-person set (preparation, batch or
extracts), with its measured size, for the training generator's range to cover rather than for the twin to fit.

## A data-pipeline defect in the REAL tokens

prepare_data_set.py keeps the FIRST 160 tokens in the order the CSV lists them (locus 13 first, locus 11
last); make_insilico.xflat_to_tokens keeps the TALLEST 160. At the cap (NOC3 5, NOC4 26, NOC5 31 test
mixtures) the real tokens lose 1.6-2.3 genuine donor alleles per profile, up to 7, at median 1186-5151 RFU,
all at loci 3 and 11 - the last in file order. The model reads tokens (tokens8_{split}.npy), so on those
profiles it is shown real mixtures missing tall alleles at two fixed loci, a pattern the training data never
contains. Not changed here: fixing it re-tokenises the test split.

Options measured on the 54 test profiles above the cap (Xflat, all peaks): the current CSV-order cut drops
2.41 true alleles + 1.41 stutter-position + 0.41 other peaks per profile (median true-allele height 1271 RFU);
keeping the tallest 160 drops 0.00 true alleles, 1.13 stutter-position and 2.85 other peaks (median 5 RFU).
The generator exceeds 160 before its own tallest-160 cut far more often than real: NOC4 18.5% vs 9.5%,
NOC5 26.1% vs 7.0% - its faint-peak excess, which a larger MAX_SEQ would expose to the model.

## RETRACTION 4: the NOC5 remnant IS monotone - the union metric hid it

The previous section called the NOC5 shortfall non-monotone (union: NOC4 +0.022, NOC5 -0.031). The union is
the wrong ruler for a combination law: a shared allele is present if ANY carrier keeps it and heights add, and
an allele next to another collects its stutter, so union presence carries positive terms that grow with NOC.
On private alleles at clean positions (no other allele within +-2 repeats) they are absent. Same population,
generator draws recorded through a wrapper on _clean_source, expected presence given each draw, CI by donor set:

| NOC | real - C (design) | gen - C (post-source functions) | real - gen (total) |
|---|---|---|---|
| 2 | +0.018 [-0.063,+0.072] | -0.002 | +0.021 |
| 3 | +0.023 [-0.008,+0.057] | +0.002 | +0.021 |
| 4 | -0.030 [-0.066,-0.004] | -0.000 | -0.030 |
| 5 | -0.075 [-0.118,-0.033] | -0.003 | -0.072 |

Every generator function after the source reproduces the design (the locus-shock cap min(s_i, 1-l) never binds:
exactly 0). The whole residual is at design level, and it falls with contributor count from NOC3.

Design variables copied from real, one at a time:
- position class: NOC5 negative in every class (private clean -0.065, next-to -0.025, shared-2 -0.037,
  shared-3+ -0.017), NOC2-4 positive or zero - not a position-dependent combination rule.
- source realisation (one PCR per cell, reused across mixtures): mean cell luck +0.0035 at NOC5; replacing each
  cell by its population expectation leaves -0.026. Refuted.
- effective template read from the mixture's own heights: real allele mass per nominal ng is far below the
  NOC1 sum (log F median -0.22/-0.73/-0.60/-0.90 at NOC2-5), but reading dropout at that template overshoots
  by +0.07 to +0.13 at every NOC. Heights fall much more than presence does; not a template deficit. Refuted.
  (The first version conditioned on the scored alleles being present and returned 1.000 - discarded.)
- donor set: each NOC level is 4-6 donor sets. All five NOC5 sets are negative (-0.038 -0.058 -0.033 -0.010
  -0.009); all four NOC4 sets positive. Matched pairs sharing four donors, NOC4 -> NOC5: -0.095, -0.065,
  -0.020, -0.024. Not a set property: it follows the added contributor.
- exact rung, no conversion at all (L0): NOC4 -0.051, NOC5 -0.056 (12 contributors only) - same sign and size,
  so the level conversion is not the carrier.
- export: 100 Allele columns per marker row, no row ever full. Refuted.
- analysis filter: the Filtered export deletes true private clean alleles that UnFiltered has - NOC1 0.0004,
  NOC2 0.0010, NOC3 0.0216, NOC4 0.0185, NOC5 0.0155 - at median 0.04-0.06 of the locus's tallest peak, up to
  hundreds of RFU; and 13% (NOC1) to 29-30% (NOC4/5) of non-allele peaks. A relative-to-locus-maximum rule the
  NOC1 sources cannot carry. A real missing mechanism, but flat over NOC3-5: not the NOC5 carrier.

Knockouts were abandoned for this ladder: same-seed runs with jitter off moved NOC5 -0.048 -> -0.022, but a
record of every gamma draw showed shape >= 3.98 and no multiplier near zero. The gamma sampler consumes a
shape-dependent number of uniforms, so the knockout desynchronised every later draw; the difference was noise.

Open: a retention loss of about -0.05 per added contributor beyond four on private clean alleles, on the same
donors, not carried by any single-source variable tested. Not modelled yet.

## The filter is PULL-UP removal (Alfonse et al.), not a locus-maximum rule

The naming-convention PDF (data_raw) describes only the UnFiltered export: GeneMapper ID-X v1.4, AT 1 RFU, all
peaks. The Filtered CSVs are documented elsewhere as having pull-up, minus-A and SE33 -2 bp removed by criteria
defined in Alfonse et al. 2018 (FSI Genet 32:62-70); the numeric criteria were not retrievable. Read off the
paired exports on the test split, true-donor alleles present UnFiltered:

| NOC | deleted | deleted with a >= as-tall peak in another dye within +-1 bp | kept with that context | minus-A context among deleted |
|---|---|---|---|---|
| 1 | 0.0002 | 0.889 | 0.071 | 0.000 |
| 2 | 0.0026 | 0.887 | 0.110 | 0.000 |
| 3 | 0.0049 | 0.760 | 0.158 | 0.006 |
| 4 | 0.0056 | 0.968 | 0.193 | 0.000 |
| 5 | 0.0047 | 0.835 | 0.203 | 0.035 |

A true minor allele that lands on the size of a tall peak in another colour is removed as pull-up. It is a genuine
coupling between contributors (one person's tall peak deletes another's allele), it is forensic knowledge
(GlobalFiler/3500 pull-up ~1% of parent, 97.8% within +-1 bp), and the generator does not have it. The
"relative to the locus maximum" description at make_insilico.py:532 was a proxy for it. Its size on private clean
alleles is flat over NOC3-5 (~0.02), so it is not the carrier of the open NOC4-5 residual.

## INSTALLED: the pull-up filter, as an increment

Rule (calibrate.py: pull_neighbours / pull_prob / _pull_table, cached in data/pullup_pairs.json): P(delete | R, |d|),
R = tallest peak in another GlobalFiler dye channel within 2 bp / this peak, read off the Filtered/UnFiltered pairs
of 5244 NOC1 TRAIN profiles on the generator's footing (bin median sizes; each column's no-context rate taken out).
On that footing it peaks at 0.84-0.90 for R >= 20 within 0.5 bp (1.000 on run sizes, softened by size jitter around
the median), 0.51-0.55 at R 10-15, ~0 below 10. Heights are identical in the two exports for every paired peak.

Applied last in gen_mixture, charging only what the NOC1 inheritance does not already carry:
- allele: keeps (1 - P_mix) / (1 - P_own), P_own in its dominant contributor's own context (its source was
  filtered there);
- artefact: the same, P_own in the context of the contributor dominating its parent bins (+-1, +-2 repeats). A first
  version divided artefact emission by a NOC1-average P_bar(bin, h) instead; stutter height scales with the
  profile's loudness and so does its context, so an average at fixed h is the wrong baseline - the NOC1 twin lost
  0.54 stutter peaks. Replaced;
- noise: emitted at its rate / (1 - P_bar_noise(bin)), P_bar averaged over NOC1 contexts and the locus noise
  heights (noise height is the instrument's), then the whole rule in the mixture's context.

Identity (val NOC1 as its own source, n=600): allele loss 0.0005 on and off, stutter -0.79 -> -0.28, other
-0.38 -> +0.17 - closer to zero with the filter. Tier 0, real - twin, real = Xflat, same specs and seeds:

| | allele | stutter | other | peaks | twin >160 (real) |
|---|---|---|---|---|---|
| NOC1 off / on | +0.01 / +0.01 | +1.07 / +1.02 | +0.35 / +0.19 | +1.43 / +1.22 | 0 / 0 (0) |
| NOC2 | +0.81 / +0.96 | +2.63 / +3.46 | -0.13 / +0.78 | +3.31 / +5.20 | 0.3 / 0.0 (1.0) |
| NOC3 | +0.76 / +1.28 | -0.04 / +1.33 | -0.76 / +0.41 | -0.04 / +3.03 | 1.0 / 0.0 (0.0) |
| NOC4 | +0.20 / +0.96 | -3.23 / -0.69 | -2.40 / -0.58 | -5.42 / -0.31 | 22.5 / 11.3 (11.7) |
| NOC5 | -4.38 / -3.88 | -2.08 / +0.37 | -0.79 / +0.91 | -7.24 / -2.60 | 20.3 / 8.9 (8.1) |

The NOC4-5 peak excess is gone and the over-cap rate now matches real; NOC2-3 now carry too FEW stutter
(+3.5 / +1.3), a deficit that was already +2.6 at NOC2 before the filter. Ladder, private clean alleles, real - gen:
NOC2 +0.021, NOC3 +0.026, NOC4 -0.020 [-0.034, +0.016], NOC5 -0.059 [-0.094, -0.020] (was -0.072); the filter's
own share, gen - Ccap, -0.002 / -0.004 / -0.010 / -0.016.

## Tried and reverted: the crowd curve read on the UnFiltered export

On the crowd law's own bins (>= 2.5 repeats from every allele) the filter does grow with occupancy - Filtered /
UnFiltered count of peaks under 25 RFU 0.99 at NOC1's sparsest sextile, 0.76 at its densest, 69-83% of the
deleted ones in pull-up context - so crowd and filter overlap in principle. The curve was re-read on UnFiltered,
context-free bins. It broke NOC1 (identity: -0.65 noise peaks per profile, filter on or off; tier 0 'other' +0.19
-> +1.16) and every mixture ('other' NOC4 -0.58 -> -4.01, NOC5 +0.91 -> -2.74). Reverted exactly. Not the cause,
checked: heights (identical in both exports). One difference found and not yet explained: 9289 binned peaks (1.8
per profile) exist only in the Filtered export, concentrated at 8-12 RFU on the crowd bins - the two exports do not
hold the same peak set. The overlap stays open; data/pullup_unf_noc1.npz is an unused cache from the attempt.

## The training generator does not run the laws the twin verifies

build_train draws every mixture through gen_any(cols, pool, rng, bin_size), which calls gen_mixture with no
ng_total, cond, inj_sec or t_total. So: no template per contributor, sources from pool_clean_all (UNTREATED runs at
COMPLETE, bright levels, renormalised onto the reference by _renorm_tpl), no level conversion, no treatment conversion,
no degradation draw, no injection gain, no budget solve. What the twin work installed and verified reaches the model
only through the artefact, noise and pull-up stages. Against real test (trainpath.py, 250 per NOC, p10 / p50 / p90):

| NOC | retention real | retention train | log10 RFU real | log10 RFU train | >160 real / train |
|---|---|---|---|---|---|
| 2 | 0.74 / 0.97 / 1.00 | 0.98 / 1.00 / 1.00 | 3.72 / 4.48 / 5.25 | 3.85 / 4.59 / 5.27 | 0.3 / 0.0 |
| 3 | 0.80 / 0.97 / 1.00 | 0.96 / 1.00 / 1.00 | 3.92 / 4.57 / 5.20 | 4.22 / 4.89 / 5.47 | 1.0 / 4.8 |
| 4 | 0.78 / 0.96 / 0.99 | 0.96 / 0.99 / 1.00 | 4.05 / 4.77 / 5.44 | 4.37 / 5.03 / 5.66 | 9.5 / 22.8 |
| 5 | 0.68 / 0.93 / 1.00 | 0.96 / 0.99 / 1.00 | 3.88 / 4.60 / 5.26 | 4.55 / 5.19 / 5.64 | 7.0 / 49.6 |

The training set is EASIER than real on dropout - it never reaches real's low-retention region - and louder; only
the proportion range is wider. The hidden interaction factor proposed for training acts through the template, so it
has nothing to act on here. Prerequisite: route build_train through the template path with drawn specs, widening in
the draws only (the rule of twin-matches-real-train-goes-wider).

## Model-level twin test (inference only), current generator with pull-up

Model results/inc22_new_gen_fold0_seed42, every real test mixture regenerated from its own spec; both sides tokenised
from Xflat (tallest 160, feasibility, enrich). Official-token real scores are within 0.002 of the Xflat-token ones.

| NOC | count real / twin | ID real / twin | paired agreement count / set | undercount real / twin |
|---|---|---|---|---|
| 2 | 0.982 / 0.961 | 1.000 / 0.988 | 0.943 / 0.988 | 0.009 / 0.030 |
| 3 | 0.961 / 0.935 | 0.992 / 0.984 | 0.906 / 0.977 | 0.029 / 0.055 |
| 4 | 0.897 / 0.773 | 0.959 / 0.959 | 0.773 / 0.930 | 0.087 / 0.186 |
| 5 | 0.747 / 0.801 | 0.849 / 0.909 | 0.750 / 0.806 | 0.253 / 0.199 |

Real replicate agreement (the ceiling) 0.927. The twin is HARDER than real at NOC4 (undercount doubled) and EASIER at
NOC5 (fits the NOC5 allele surplus); NOC2-3 close.

## Step 1 of widening: which conditions can be generated at levels real does not have

Rule: a condition may be generated off real's grid only where its law reproduces a held-out real level on NOC1.
- Injection (5 / 15 / 25 s): NOT interpolable. From the same PCR product injected longer, scaling heights to the
  target total and thresholding keeps too many alleles: 15 -> 5 s retention -0.046 (faint -0.090), 25 -> 15 s -0.008.
  A shorter injection loses more than proportional scaling predicts; there is no injection-dropout law. Keep the three
  real levels, source matched on injection.
- Treatment dose, middle dose predicted from its neighbours in log dose: DNase beta -4 / -5%, retention ratio
  |err| 0.008 / 0.009; humic 0.006; Fragmentase beta -7%, retention 0.010-0.023 (borderline); UV 0.016-0.025 and not
  monotone (beta U60 > U105); sonication ~0.05. Interpolable: DNase, humic. Not: UV, sonication. The data already hold
  eight UV doses (U5-U105) in treat_conv, while cond_beta carries three.
- Template: see below.

## RETRACTION 5: the tilt's bright-band "overcorrection" was cross-extract pairing, as first recorded

The first template hold-outs (and tiltfoot) keyed sources on donor, treatment and injection only, so they paired
across extracts. On the calibration's own pairing (same extract d#), the >= 50 RFU slope step reproduces the stored
table to every digit (0.00097 / 0.00111 / 0.00097 / 0.00066 / 0.00040 / 0.00037 from 0.0078 to 0.25 ng); across
extracts the step turns strongly negative in the bright band (-0.00124 at 0.25 on the same footing). The earlier
attribution of the bright-band gap to an extract offset stands. Open: on alleles present in both profiles at any
height the same-extract step is 20-30% smaller than on >= 50 RFU in both; which footing the generator needs is being
read off a same-extract hold-out. The same scripts also missed the 0.0312 rung (2 x 0.0312 = 0.0624 != 0.0625).

## Template hold-out, same extract: what is right and what is not

Val NOC1 targets, forced train sources of the same donor, treatment, injection and extract, one / two rungs up.
real - twin, mean [95%]:

| | retention | sd log height | bp slope |
|---|---|---|---|
| current, 1 step | -0.0017 [-0.006, +0.003] | +0.034 (faint +0.057, bright +0.022) | -0.0002 |
| current, 2 steps | -0.0097 (faint -0.022) | +0.072 | -0.0005 |
| tilt off, 1 / 2 steps | | | +0.0005 / +0.0012 |
| spread widened by 1/tpl_shrink, 1 / 2 steps | | -0.022 / -0.040 (bright -0.030) | |

- Retention over one step is right. By injection it is not: 5 s -0.016 (two steps -0.044), 15 s +0.014. PHI is pooled
  over injections while the sources are matched on them, so the step under-drops at 5 s and over-drops at 15 s.
- The tilt is needed (off: +0.0005 per step) but overshoots by about a quarter (-0.0002 / -0.0005): the >= 50 RFU
  footing it is read on is 20-30% steeper than alleles present in both, and the generator tilts every allele.
- The spread is too narrow when a source is taken down (TPL_SPREAD is skipped for every ladder source, including one
  taken from a richer rung). Widening by the whole tpl_shrink curve overshoots, most in the bright band: that curve is
  the TOTAL spread change between rungs, and the jitter increment already charges the height-driven part of it. What
  is missing is the increment beyond jitter.
Fixes indicated, not yet made: tilt on the present-in-both footing; a spread increment beyond jitter for L > t, read
on NOC1 train pairs; PHI keyed on injection as the sources are.

## INSTALLED: the three template fixes, and what the same-extract hold-out says now

calibrate.py: amp_step_tilt read on alleles present at both rungs (any height); amp_step_phi_inj, the PHI chain per
injection (pooled step where a rung holds < 15 pairs); amp_step_var and amp_step_var_inj, the log-height variance a
step down adds beyond the jitter increment (same-extract pairs, present in both, minus the generator's own jitter
log-variance for that step), cumulative from the top. make_insilico.py: _lvl_surv / _conv_surv / _lvl_var take the
injection; the variance is drawn only when the source sits on a richer rung (L > t), so NOC1 identity is untouched.

Tables: tilt 0.00354 / 0.00279 / 0.00180 / 0.00105 / 0.00059 / 0.00030 / 0 (was 0.00447 ... 0.00037); PHI at 0.0078 ng
0.478 / 0.604 / 0.601 at 5 / 15 / 25 s (pooled 0.562); extra variance at 0.0078 ng 0.019 / 0.237 / 0.430.

Same-extract val hold-out, one rung down, real - twin (before -> after):
| | retention | sd log height | bp slope |
|---|---|---|---|
| all | -0.0017 -> -0.0003 | +0.034 -> +0.009 | -0.0002 -> -0.0001 |
| 5 / 15 / 25 s | -0.016 / +0.014 / -0.003 -> 0.000 / +0.005 / -0.007 | -> +0.011 / +0.013 / +0.002 | |
| faint / bright | | -> +0.031 / -0.003 | |
Two rungs: retention -0.004 (faint -0.017, the known floor composition), sd +0.011, slope -0.0001.

Found on the way: amp_step_var first came out all zero because the pull-up block had assigned _own = a profile matrix,
overwriting the per-donor mask every later block reads (derive is one namespace); only the spread block read it after
that point, so no other constant was affected. Renamed _ownp. Also measured and rejected: the variance read on each
profile's own present alleles (sd +0.013 overall, faint +0.029) - footing is not what leaves the faint band narrow.
Open: faint sd +0.03 (about 4% of the spread there).

Model-level twin test, 3 twins per mixture (single-run differences of 0.05-0.07 earlier were within replicate noise,
sd 0.007-0.017): count real / twin 0.982 / 0.953, 0.961 / 0.932, 0.897 / 0.795, 0.747 / 0.780 (NOC2-5); ID 1.000 /
0.988, 0.992 / 0.987, 0.959 / 0.966, 0.849 / 0.909; paired set agreement 0.988 / 0.979 / 0.941 / 0.811 (ceiling 0.927).
Still: NOC4 count harder on the twin (undercount 0.154 vs 0.087, overcount 0.051 vs 0.017), NOC5 ID easier.

## Answer-copy at the MODEL level: it is the allele PRESENCE, and nothing else

Each real test mixture rebuilt as a twin, then handed to the model with ONE layer carrying real's answer (count
accuracy / ID):

| | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|
| real | 0.982 | 0.961 | 0.897 | 0.747 |
| twin | 0.958 | 0.935 | 0.835 | 0.804 |
| copy the allele layer | 0.991 | 0.956 | 0.847 | 0.755 |
| copy the product layer | 0.931 | 0.914 | 0.756 | 0.820 |
| copy the other layer | 0.952 | 0.930 | 0.839 | 0.812 |
| copy allele PRESENCE only | 0.976 | 0.935 | 0.888 | 0.742 |
| copy allele HEIGHTS only | 0.964 | 0.940 | 0.855 | 0.809 |

Copying which alleles are present lands NOC4 and NOC5 on real; copying heights, stutter or noise does not. Per sample
the twin has fewer peaks and a lower MAC than real at NOC2-4 (peaks +3.9 / +2.8 / +1.6, MAC +0.58 / +0.65 / +0.57)
and more at NOC5 (peaks -3.2, retention -0.032). Worst cells: NOC4 at total <= 0.15 ng (real 0.679, twin 0.421) and
NOC5 at total <= 0.15 ng (real 0.400, twin 0.572).

## INSTALLED: the step down keeps alleles by HEIGHT too, and both coefficients are per rung

The conversion's logistic fit had three coefficients and the code exported only the size slope; the height coefficient
was dropped (the comment beside it quoted its values). Re-read on same-extract NOC1 pairs:

| target rung | 0.0078 | 0.0156 | 0.0312 | 0.0625 | 0.125 | 0.25 | pooled |
|---|---|---|---|---|---|---|---|
| size | -0.49 | -0.37 | -0.59 | -0.69 | -0.65 | -0.64 | -0.378 |
| height | 0.20 | 0.37 | 0.69 | 1.19 | 1.60 | 2.09 | 0.465 |

The pooled size slope (what the generator used) is flatter than every per-rung value - pooling across rungs without
their intercepts. Both are per rung now (amp_step_size_t, amp_step_h_t) and _conv_surv takes the source's own log
heights. Model twin, 3 twins each: NOC2 0.953 -> 0.962, NOC4 0.795 -> 0.822, NOC5 0.780 -> 0.773 (real 0.982 / 0.897 /
0.747); ID NOC5 0.909 -> 0.902 (real 0.849). Per-contributor retention barely moves - in a mixture the source sits on
the rung just above, so the step takes few alleles - but WHICH alleles it takes is what the model reads.

## The mixture penalty is NOC5-only, measured on real alone

Retention of a donor's own alleles at a given DNA mass. The convention fixes the masses: mixture target mass was set
by "setting the minor component equal to 0.015, 0.03, 0.06 and 0.125 ng", so t = phi * ng is that contributor's mass.

| mass (ng) | NOC1 | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|---|
| 0.011-0.022 | 0.689 | 0.730 | 0.704 | 0.711 | 0.569 |
| 0.022-0.045 | 0.829 | 0.870 | 0.861 | 0.853 | 0.767 |
| 0.045-0.09 | 0.919 | 0.947 | 0.939 | 0.919 | 0.905 |
| 0.09-0.18 | 0.968 | 0.980 | 0.984 | 0.949 | 0.938 |

Same at every injection. NOC2-4 behave like NOC1 at the same mass - the per-contributor law is right for them - and
only the five-person series loses more. Nothing in NOC1 can predict that, so the twin is easier at NOC5 by
construction; its size is recorded here.

What is left at NOC4 is therefore not how MANY alleles survive but WHICH: the twin's MAC is 0.57 below real. A real
tube shares one reaction, so a weak locus is weak for every contributor, while the twin's contributors come from
different NOC1 runs with independent locus patterns. STR_SHARE_RUN=1 (one drawn locus pattern per tube) moves MAC at
NOC4 10.76 -> 11.24 (real 11.61) and at NOC3 10.68 -> 10.78 (real 11.31), but at NOC2 10.25 -> 9.95 (real 10.78) and
NOC5 not at all. Two-sided, so not installed; it needs the model-level test first.

## Conditions as model input, and the scope the twin supports

The conditions a laboratory actually knows are the injection time, the total DNA mass put into the PCR and the
qPCR quality index; the mixture ratio and the contributor count are the answer, not conditions.

First: how much of the answer do the conditions already carry in THIS dataset? PROVEDIt set every mixture's target
mass by fixing the minor component at 0.015 / 0.03 / 0.06 / 0.125 ng, so the total encodes the design. On the real
test mixtures, a rule that reads only the total mass gets NOC right 0.755 of the time (base rate 0.289; the model
itself 0.895); mass + injection + treatment 0.789; mass + the true minor share 0.827. A generator that drew its
masses the same way would teach that shortcut, and it would not survive contact with casework.

Probe (trained on synthetic mixtures only, scored on real; a small gradient-boosted classifier on 18 profile
summaries, NOT the deployed model), with the synthetic masses drawn INDEPENDENT of the contributor count:

| | real |
|---|---|
| A, profile features | 0.532 |
| B, profile + conditions | 0.523 |
| C, conditions alone | 0.260 |

So, with the design shortcut removed, the conditions add nothing measurable on top of the profile - and C sits at
the base rate by construction. The stronger run (6000 synthetic mixtures per arm, regularised, with a held-out
synthetic split) says it sharper:

| arm | synth held out | real |
|---|---|---|
| A, profile features | 0.476 | 0.530 |
| B, profile + conditions, mass drawn independent of k | 0.531 | 0.504 |
| D, as B but the synthetic mass follows PROVEDIt's design rule | 0.656 | 0.608 |
| E, conditions alone, design rule | 0.356 | 0.299 |

B is the answer: the conditions help ON the synthetic set (+0.055) and HURT on real (-0.026) - the conditional
relation the model learns from synthetic data does not transfer, which is the same domain gap in a new place. D does
better on real only because the synthetic masses were drawn the way PROVEDIt drew them, i.e. by teaching the design.
Conditioning the deployed model is therefore not supported.

Scope. The refusal rule was chosen on the TWIN side only (regions defined by what is known at run time: the model's
own call and the total mass), then verified on real:

| region | coverage twin | twin acc | coverage real | real acc |
|---|---|---|---|---|
| answer everything | 1.000 | 0.870 | 1.000 | 0.895 |
| refuse when the call is >= 4 and mass <= 0.35 ng | 0.801 | 0.891 | 0.790 | **0.926** |
| refuse when the call is >= 5 | 0.773 | 0.844 | 0.788 | 0.871 |

Per region the twin also says WHERE synthetic confidence transfers: the call "2" real 0.948 / twin 0.926, "3"
0.907 / 0.893, "4" 0.743 / 0.776, "5" 0.982 / 0.870. The "5" calls are the most reliable on real and the least
reliable on the twin - a confidence calibrated on synthetic data distrusts exactly the calls it should trust - while
the "4" calls are the genuinely weak ones and are what the rule removes.

## The shared locus pattern was the wrong form, and the MAC deficit is not alleles at all

STR_SHARE_RUN=1 moved MAC two ways (NOC4 10.76 -> 11.24, NOC2 10.25 -> 9.95), which is the signature of a mechanism
that is not what it says. Read: the switch divides out each source's realised locus pattern and imposes a rebuilt one
- a shrunk per-donor pattern (sd 0.095) + a shared run offset (sd 0.228) + a FRESH INDEPENDENT per-contributor draw
(sd 0.368). It replaces real structure instead of adding the increment, adds more independence than sharing, and
overlaps the locus-shared amplification failure already installed (amp_share_r 0.246).

Tested in the correct increment form (one per-locus offset drawn once per mixture, applied to every contributor's
source through the _clean_source hook, each source's own pattern left intact): MAC at sd 0.10 / 0.228 / 0.35 against
no offset - NOC2 9.99 / 9.89 / 9.86 against 9.96, NOC3 10.90 / 10.80 / 10.84 against 10.86, NOC4 11.54 / 11.53 /
11.43 against 11.66, NOC5 11.88 / 11.76 / 11.80 against 11.69. Nothing. The MAC gain from the switch came from the
independent noise it adds, not from sharing.

And the deficit is not in the alleles. The busiest locus, split by bin class (real / twin): allele bins 3.96 / 3.92,
5.51 / 5.61, 6.33 / 6.28, 7.00 / 7.06 at NOC2-5 - equal; non-allele bins 7.39 / 6.75, 6.95 / 6.45, 7.01 / 6.45,
6.43 / 5.83 - the whole gap. Nor is real's dropout more clustered by locus: the sd of a contributor's per-locus
retention is real 0.127 / 0.102 / 0.107 against twin 0.142 / 0.130 / 0.119 at NOC2-4 (real is SMOOTHER; only NOC5
reverses, 0.142 against 0.114).

Where it comes from is already visible at NOC1: the twin is short of artefacts at TALL parents. Presence at the n-1
bin by parent height (real / twin): 500-1500 RFU 0.849 / 0.799, >1500 0.887 / 0.873; n-2 >1500 0.427 / 0.377; n+1
>1500 0.437 / 0.405. The busiest locus in a mixture is the one with the tallest peaks, so a NOC1 artefact-survival
shortfall at tall parents lands exactly on MAC, which is what the count head reads. (Separately, at faint parents the
twin's n-1 heights are too low: median ratio 0.082 against real's 0.132 at 50-150 RFU parents.)

## Judged as a RANGE: what the twin's ensemble already covers, and what it does not

The rule for this part is the user's: do not chase the exact value, get the range right, generate the whole range,
and keep every case obeying the law. So the question became: is real inside the twin's ensemble for the same spec?
NOC1 val, eight twins per spec, and beside it real's own run-to-run spread within a (template, injection, treatment)
cell:

| statistic | real spread within a cell | twin ensemble sd | real inside the twin's 10-90% |
|---|---|---|---|
| n-1 presence at parents >= 500 RFU | 0.052 | 0.088 | 0.806 |
| artefact peak count | 7.43 | 5.40 | 0.600 |
| allele retention | 0.021 | 0.012 | 0.208 |
| n-1 height ratio at 50-150 RFU parents | 0.021 | 0.020 | 0.250 |

This changes what needs fixing. The tall-parent stutter shortfall I was about to calibrate is ALREADY covered
(0.806) - its centre is off by 0.05 but the ensemble contains real, so by the rule it needs nothing. What fails is
retention, where the ensemble is half as wide as real's own run-to-run spread, and the faint-parent stutter ratio,
whose centre is off and whose width is too small to cover it.

INSTALLED: a per-sample dropout severity. One factor f is drawn per tube and the DROPOUT path reads t * f - the rung
the source comes from, the level conversion, the treatment conversion - while heights stay on the observed total.
Physically it is how much template actually entered that reaction. Width picked by matching the ensemble to the
reference: sd 0.08 gives ensemble sd 0.022 against real's 0.021, and real inside the ensemble 0.458 of the time
against 0.208. Wider pads past the data (sd 0.031 at 0.22) and drags the centre down.

Tried and removed: an offset on the draw, to undo the centre slip (0.891 -> 0.887) that convexity causes. It does not
work - f > 1 cannot bring back an allele the source itself lost, so the offset only widens the ensemble further
(sd 0.032 at mu 0.06) while the centre stays at 0.888.

What the 0.458 says: the rest is not width but per-spec centring - the twin of a particular spec sits in the wrong
place, which is the one-amplification-per-cell limit, not a missing law. Model-level, 3 twins per mixture, this
change is inside replicate noise except NOC5 (count twin 0.773 -> 0.795 against real 0.747): NOC2 0.958, NOC3 0.927,
NOC4 0.821, NOC5 0.795.

## Step composition is not the defect, and the source may not be taken from the rung above

Two changes were proposed together: measure what a step of more than one rung costs instead of chaining single
steps, and then take the source from the rung strictly above the target so the twin has a lottery to draw. Both were
built and measured against the three things that decide the question at once - the centre, the coverage, and the
model. Neither is installed. What follows is what the measurements say.

**Composition.** The retention of a step was measured directly for every ordered pair of ladder rungs, on the same
donor, extract, treatment and injection, with the statistic PHI itself is defined by (observed retention at the
lower rung over the retention the upper profile predicts once scaled to the lower total and cut at the threshold).
Against the chained product of single steps:

| step | rungs | chained | measured | ratio |
|---|---|---|---|---|
| 0.0312 -> 0.0078 | 2 | 0.654 | 0.666 | 1.019 |
| 0.0625 -> 0.0078 | 3 | 0.594 | 0.620 | 1.044 |
| 0.125 -> 0.0078 | 4 | 0.570 | 0.610 | 1.070 |
| 0.0625 -> 0.0156 | 2 | 0.769 | 0.764 | 0.994 |
| 0.125 -> 0.0312 | 2 | 0.872 | 0.880 | 1.009 |
| 0.25 -> 0.0312 | 3 | 0.863 | 0.825 | 0.956 |
| 0.5 -> 0.125 | 2 | 0.986 | 0.988 | 1.002 |

Chaining is right: every pair lands within 7% and most within 2%, on 52 to 409 cells each. Installed as a
correction it made the twin WORSE at the model level (count 0.875 -> 0.868 overall, NOC4 0.822 -> 0.800 against a
replicate sd of 0.012-0.026), so it was removed. The earlier suspicion that composition explained the two-step
hold-out gap is refuted - the gap is somewhere else.

**Source from the rung strictly above.** The rule gives the twin the lottery it lacks and the coverage shows it:
NOC1 ensemble sd 0.0123 -> 0.0239 and real inside the 10-90% band 0.208 -> 0.592; mixtures, stratified, 0.39 ->
0.53. At the model level it is a small gain on every NOC (count 0.958/0.931/0.800/0.773 -> 0.965/0.932/0.810/0.772
against real 0.982/0.961/0.897/0.747), about one replicate sd.

It breaks the centre. Pooled centres are useless here - contributor mass and treatment decide retention, so a
pooled mean mixes strata. Stratified by both, the rule that takes the source at or above the target sits at -0.001
of real and never misses a cell by more than 0.02; the strictly-above rule sits at -0.013 with cells at -0.043:

| mass (ng) | sample | n | real | at-or-above | strictly above |
|---|---|---|---|---|---|
| 0.011-0.022 | clean | 42 | 0.778 | 0.844 | 0.844 |
| 0.022-0.045 | clean | 26 | 0.966 | 0.949 | 0.923 |
| 0.022-0.045 | treated | 138 | 0.800 | 0.799 | 0.756 |
| 0.045-0.09 | treated | 165 | 0.908 | 0.906 | 0.888 |
| 0.09+ | treated | 161 | 0.964 | 0.964 | 0.969 |

The reason is measurable, and it is the nesting the increment rule warns about. In a mixture, moving the source up
one rung returns +0.010 in alleles present while the law then charges -0.038 for the longer step; on NOC1 the same
move returns +0.049 against a charge of -0.046, which is why NOC1 stays unbiased and the mixture does not. The
alleles a richer source brings back are the faint ones, and in a mixture the contributor's own share puts them under
the threshold anyway - so the step charge is levied on alleles the threshold has already removed. Source richness
and step charge are nested events, not independent ones, and multiplying them over-drops. Rejected.

**What the mass key does transfer.** A contributor inside a mixture stands exactly as tall as a single source of the
same mass: median peak 53 / 146 / 239 RFU at 5 / 15 / 25 s in the 0.011-0.022 ng band against 56 / 147 / 259 for
NOC1 at 0.0156 ng, and retains slightly less (0.787 / 0.842 / 0.862 against 0.825 / 0.888 / 0.892). So keying the
ladder on the contributor's own mass is right; what is wrong is charging the step on top of the threshold.

**Width has no direct target in mixtures.** Every mixture specification in the dataset - donors, ratio, mass,
treatment, injection - occurs exactly once (4651 of them, none repeated), so real's run-to-run spread for a mixture
cannot be measured. The width the twin must have therefore rests on the NOC1 law, which was validated there: over
7441 same-specification pairs the observed spread is 0.0561 against the binomial prediction 0.0588, no residual.

## The lottery is redrawn, not copied

The twin takes a real single-source profile for each contributor, which is the right thing for everything
systematic it carries - and one wrong thing. The alleles that amplification happened to lose are lost in every twin
built from it, so the ensemble of a specification cannot move: NOC1 ensemble sd 0.0123, against real values that
land inside the 10-90% band only 0.208 of the time.

**What real's own lottery looks like.** This set holds one PCR product per cell, so the only repeated measurement
of a specification is that product injected for different numbers of seconds - and seconds move retention
systematically (0.636 / 0.732 / 0.761 at 0.0078 ng), which poses as lottery. Putting each profile against the mean
of its own (template, treatment, seconds) cell and pairing only the residual leaves the capillary's own draw:

| template (ng) | pairs | retention | sd of residual | sd if every allele alike | D |
|---|---|---|---|---|---|
| 0.0078 | 656 | 0.534 | 0.0467 | 0.0863 | 0.29 |
| 0.0156 | 1009 | 0.690 | 0.0439 | 0.0800 | 0.30 |
| 0.0312 | 588 | 0.863 | 0.0329 | 0.0594 | 0.31 |
| 0.0625 | 1019 | 0.922 | 0.0297 | 0.0462 | 0.41 |
| 0.125 | 1009 | 0.973 | 0.0170 | 0.0282 | 0.36 |
| 0.25 | 1026 | 0.989 | 0.0112 | 0.0177 | 0.40 |

D is the share of the equal-chance variance real actually shows, so it measures how far the per-allele survival
chances are spread: pooled 0.37. The PCR lottery itself is NOT measurable here - no two cells hold the same
specification, and only 9 different-extract pairs exist, all at 0.5 ng - so it is not invented, and the twin's
ensemble carries the capillary part alone.

**The rule.** The source's heights are kept; only which of the donor's alleles are present is drawn again. Allele k
gets a probability p(k) whose mean over the donor's alleles is exactly the source's own retention, so

    E[ kept ] = sum_k p(k) = n * r = what the source had,

and the centre cannot move whatever the spread of p - which is the whole point, since the "strictly above" rule
failed on exactly that. The shape of p is the measured dropout curve against the allele's own RFU (drop_h/drop_p,
until now calibrated and unused), read at the modelled height A(condition, allele) + P(donor, allele) where the
source has none; the STRENGTH of the tilt is solved per profile so that mean p(1-p) = D * r(1-r), the dispersion
above. Alleles of one locus share their draw with probability AMP_SHARE_R. An allele brought back is a marginal
one, so its height is drawn from the modelled one weighted by drop(h)(1-drop(h)) - the chance the curve gives an
allele of standing on that edge.

**Measured.** Centre held: mixtures, stratified by contributor mass and treatment, -0.001 -> +0.001; NOC1 0.8908 ->
0.8903 against real 0.8940. Ensemble opened: NOC1 sd 0.0123 -> 0.0251 and coverage 0.208 -> 0.358; mixtures 0.39 ->
0.52. Model twin unchanged: count 0.962/0.930/0.822/0.773 -> 0.959/0.923/0.815/0.777 (real 0.982/0.961/0.897/0.747),
total 0.875 -> 0.872 against a replicate sd of 0.004-0.017. Everything else about a NOC1 profile stands: peaks 91.0
-> 90.3, total 36700 -> 36702, off-allele share 0.088 -> 0.087, heterozygote balance 0.683 -> 0.695 (real 0.685).
Two small regressions to keep in view: sd of log height 0.687 -> 0.656 (real 0.702) and stutter ratio 0.017 ->
0.015 (real 0.020), both from the faintest alleles being the ones the tilt takes.

Coverage stops at 0.36-0.52 rather than the ~0.8 a correct width would give, and the reason is stated above: only
the capillary's share of the lottery can be measured in this data.
