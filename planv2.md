## Plan 6 tuần

**Tuần 1 — Setup + Literature**

Đọc theo thứ tự ưu tiên (đọc tóm tắt + figures cho mọi cái, đọc sâu cho top 3):
1. **Alfonse 2017** (PROVEDIt original, FSI Genetics) — hiểu data
2. **Naming Convention PDF** (link ở trang Rutgers) — hiểu filename encoding
3. **Yu et al. 2025** (IJLM) — closest competitor, t hơi lo cái này, m phải đọc thật kỹ để defend differentiation
4. **Set Transformer** (Lee et al. 2019, ICML) — architecture core
5. **Deep Sets** (Zaheer et al. 2017, NeurIPS) — baseline architecture
6. **deepNoC** (Taylor 2024, arxiv 2412.09803) — DL on STR, học cách họ xử lý simulation
7. **Slooten 2021** (FSI Genetics) — multi-POI LR formulation (background)
8. **CROSR** (Yoshihashi 2019) — open-set recognition technique
9. **PACE** (Marciano 2017) — historical ML on NOC

Code setup:
- Colab Pro nếu m có ($10/tháng) — đáng tiền cho 6 tuần. Free Colab cũng được nhưng session timeout phiền.
- Download data, extract, verify naming convention.
- Implement donor ID parser từ filename.
- Verify split 45/5: list 50 donor IDs trong PROVEDIt, hold out 5 với fixed seed, document choice. **Quan trọng: m phải verify cách repo NOC_DNA split — nếu họ split sample-level mà không donor-level thì có leakage.**

Output: data loader trả về `(mixture_features, contributor_45bit, has_unknown_flag, noc)`.

**Tuần 2 — Baselines**

Implement 4 baselines, train trên cùng split:
1. **Per-donor logistic regression** — independent classifier per donor. Mimic LR-paradigm. Floor.
2. **Multi-label MLP** — single network, 45 sigmoid heads. Standard ML baseline.
3. **kNN multi-label** — non-parametric. Useful nếu data sparse.
4. **XGBoost per-donor** — strong tabular baseline.

Đặt evaluation pipeline đầy đủ: Macro F1, Micro F1, Hamming loss, Exact match, sample-wise Jaccard, per-NOC breakdown (1/2/3/4/5-person), per-donor accuracy. Plus reliability/calibration plots.

**Tại sao tuần 2 quan trọng nhất**: nếu main method fail (Tuần 3-4), m vẫn có "first systematic ML benchmark trên PROVEDIt cho closed-set contributor identification" — fallback paper.

**Tuần 3 — Main method (Set Transformer + open-set)**

Architecture concrete:
- Input encoding: mixture → set of tokens `{(marker_id_embed, allele_value_embed, log_peak_height)}`. Variable size set.
- Encoder: Set Transformer (2-3 ISAB blocks, 4 heads, hidden 128). Output: fixed-dim mixture embedding `z_mix`.
- 45 donor reference profiles: also encoded as set of (marker, allele) — no peak heights vì reference là clean genotype. Output 45 vectors `z_donor[i]`.
- Scoring: cross-attention hoặc bilinear: `logit_i = z_donor[i]^T W z_mix + b`. Get 45 logits.
- Reject head: MLP(z_mix) → 1 logit cho has-unknown.
- Aux head (optional): MLP(z_mix) → 6 softmax cho NOC.

Loss: `L = BCE(45 logits) + α·BCE(reject) + β·CE(NOC)`. α, β tune trên validation.

**Tuần 4 — Experiments + Ablation**

Mandatory experiments:
- Compare main vs all baselines, all metrics
- Per-NOC accuracy curve (model should degrade gracefully với mixture phức tạp)
- Per-donor analysis (identify hard donors → discussion)
- Open-set evaluation: AUROC trên has-unknown flag, ROC curve

Ablations (để chứng minh từng component contribute):
- Set Transformer vs Deep Sets vs vanilla MLP encoder
- With/without multi-task NOC head
- With/without open-set head (degrade to closed-set only)
- Different open-set techniques (sigmoid threshold vs energy-based vs Mahalanobis)

**Tuần 5 — Polish**

- Robustness: train/test trên filtered, evaluate trên unfiltered (domain shift)
- Interpretability: attention weight visualization → which markers most informative per donor
- Failure analysis: pick 5-10 hard mixtures, dissect
- Statistical significance: multiple seeds (3-5), confidence intervals, paired t-test với baselines

**Tuần 6 — Writing**

Workshop paper format ~6-8 pages. Bố cục:
1. Intro: problem, contributions (3 cái: formulation, architecture, open-set)
2. Related work: NoC methods, deconvolution methods (Yu 2025), LR-based POI, ML open-set
3. Method: data, architecture, training
4. Experiments: baselines table, ablations, per-NOC, interpretability
5. Limitations: chưa compare EuroForMix; chỉ 1 dataset; closed-set của 45 nhỏ
6. Conclusion

## Critical pitfalls — t list để m tránh ngay từ đầu

1. **Donor-level leakage**: nếu cùng một donor xuất hiện trong train mixture và test mixture, không tính là leakage (vì task là identify từ database). Nhưng nếu cùng **mixture sample** xuất hiện cả train và test → leakage. Cẩn thận với PROVEDIt vì có replicate.

2. **Unknown donor selection bias**: 5 hold-out donors phải fixed seed, document choice, **không cherry-pick**. T recommend dùng seed của repo NOC_DNA để consistent.

3. **NOC imbalance**: PROVEDIt có nhiều 1-2 person mixture hơn 4-5 person. Cần report per-NOC metrics, không chỉ aggregate.

4. **Class imbalance per donor**: một số donor xuất hiện trong nhiều mixture, một số ít. Per-donor precision/recall sẽ thấy. Có thể cần class weighting trong loss.

5. **Comparison với LR-based methods**: forensic reviewer sẽ đòi. Workshop ML reviewer thì có thể bỏ qua. **Limitations section phải acknowledge** và explain tại sao không include (compute, setup phức tạp, domain expertise gap).

6. **Yu et al. 2025 differentiation**: t lo nhất chỗ này. M phải đọc paper họ rất kỹ và viết rõ trong related work: "Yu et al. addresses mixture deconvolution (infer unknown contributor genotypes); we address contributor identification from a known database with open-set unknown detection — orthogonal problems." Nếu m không defend được điểm này, reviewer sẽ reject.