# Kế hoạch cá nhân: Closed-set 45-class Classifier
## Bài toán: DNA Mixture Identification (45 known + flag unknown)

---

## 1. Bối cảnh

- **Bài toán nhóm**: Identify 45 known contributors + flag presence of unknown contributor trong DNA mixture (PROVEDIt dataset).
- **Dataset**: PROVEDIt (50 donors total) — chia 45 làm "known database", 5 hold-out đóng vai "unknown".
- **Phân vai**:
  - Bạn cùng nhóm (manhngvu): focus OOD detection (component "flag unknown"). Đã có baseline binary unknown detection F1 0.7482 (XGBoost) trên 500 samples.
  - **Phần của tôi**: closed-set multi-label classifier 45 classes — với 1 mixture, output vector 45-dim binary chỉ ra donor nào là contributor.
- **Tổng timeline**: 6 tuần đến deadline báo cáo cuối.
- **Deliverables**: code, báo cáo, kết quả riêng (mỗi người 1 báo cáo).

---

## 2. Mục tiêu cụ thể

**Task chính**: Multi-label classification 45 classes
- Input: features từ DNA mixture (peak heights + derived features từ STR loci)
- Output: vector 45-dim binary, mỗi dim = donor *i* có/không trong mixture
- Metric đánh giá: macro/micro F1, hamming loss, exact match accuracy, per-class precision/recall

**Constraint của phase này**: Bài toán closed-set — train + evaluate trên mixtures chỉ chứa contributors thuộc 45 known. Component flag unknown được handle bởi manhngvu, ghép vào pipeline cuối ở tuần 4.

---

## 3. Câu hỏi PHẢI xác nhận TRƯỚC khi bắt đầu code

Không bắt đầu code đến khi có câu trả lời:

1. **Hỏi thầy**: Task chính xác là multi-label 45 classes (1 mixture có thể có 2-5 contributors trong 45) hay multi-class 45 classes (1 mixture chỉ có 1 contributor)?
2. **Hỏi thầy**: Có bắt buộc dùng DL không, hay tabular ML (XGBoost/CatBoost) cũng được?
3. **Hỏi manhngvu**: Pipeline parse PROVEDIt raw + extract 260 features có share được không? Format input/output cụ thể?
4. **Hỏi manhngvu**: Cách kết hợp 45-classifier (của tôi) + OOD module (của bạn) ở pipeline cuối — rule-based hay joint model?

---

## 4. Plan chi tiết 6 tuần

### Tuần 1: Setup + Baseline 45-class

**Day 1-2: Data preparation**
- [ ] Download PROVEDIt raw từ https://lftdi.camden.rutgers.edu/provedit/files/
- [ ] Đọc PROVEDIt Naming Convention PDF, hiểu format file và cách donor IDs encode trong tên file
- [ ] Sync với manhngvu, lấy pipeline parse + 260 features (nếu share được)
- [ ] Inspect dataset format, hiểu cấu trúc CSV (peak heights, allele calls per locus per sample)

**Day 3-4: Label engineering**
- [ ] Chia 50 donors → 45 known + 5 hold-out unknown (document rõ tiêu chí chia: ngẫu nhiên với seed cố định, hoặc theo balance kit/condition)
- [ ] Filter mixtures:
  - Closed-set train/val/test: mixtures chỉ chứa contributors thuộc 45 known
  - Open-set evaluation set: mixtures có chứa ≥1 hold-out donor (để integration với OOD module sau)
- [ ] Build multi-label encoding: vector 45-dim binary cho mỗi mixture
- [ ] **Validate labels**: random sample 50 mixtures, kiểm tra labels đúng bằng tay → catch parse errors sớm

**Day 5-7: Baseline classifier**
- [ ] Implement baseline XGBoost multi-label (one-vs-rest hoặc multi-output)
- [ ] Train + evaluate trên closed-set
- [ ] Metrics: macro F1, micro F1, hamming loss, exact match, per-class P/R
- [ ] **Milestone tuần 1**: phải có baseline numbers, không kéo sang tuần 2

### Tuần 2: Exploration đợt 1 — 2 variant nhẹ

- [ ] **Variant 1**: MLP trên 260 features
  - Architecture: 260 → 512 → 256 → 45 (sigmoid output)
  - Loss: BCE per class với class weighting
- [ ] **Variant 2**: 1D CNN hoặc Set Transformer trên peak height matrix per locus
  - Input: matrix [num_loci × num_alleles_max] hoặc [num_loci × features_per_locus]
  - Lý do: tận dụng cấu trúc spatial/set của loci, không flatten thành vector
- [ ] Phân chia với manhngvu: tôi làm về architecture, bạn làm về OOD method — tránh trùng
- [ ] **Milestone tuần 2**: 3 cột số liệu (XGBoost baseline + 2 variant) để compare

### Tuần 3: Exploration đợt 2 có direction

- [ ] Phân tích kết quả tuần 2: variant nào tốt? Tại sao? Có pattern gì? Class nào khó nhất?
- [ ] **Variant 3 (follow-up có direction từ insight tuần 2)**: 1 variant duy nhất, có lý do
  - VD: nếu MLP và CNN đều tốt → kết hợp (CNN feature extractor + MLP head)
  - VD: nếu class imbalance là bottleneck → focal loss, oversampling, contrastive learning
  - VD: nếu một vài locus carry nhiều info → attention over loci
- [ ] Bắt đầu hyperparameter tuning sơ bộ cho variant best (Optuna)
- [ ] Sync chính thức với manhngvu về integration: structure pipeline cuối thế nào?

### Tuần 4: Thống nhất với manhngvu + tối ưu sâu

- [ ] Thống nhất pipeline cuối với manhngvu:
  - Structure: input → preprocessing → 45-classifier (của tôi) || OOD module (của bạn) → integrated output
  - Cách kết hợp: rule-based (OOD score > threshold thì flag unknown) hay learned (joint model)?
  - Quyết định: phân vai phân tích trong báo cáo (mỗi người focus góc khác nhau dù pipeline chung)
- [ ] Tuning kỹ variant best:
  - Learning rate schedule
  - Regularization (dropout, weight decay, early stopping)
  - Data augmentation cho minority classes
- [ ] **Bắt đầu draft báo cáo**: intro + dataset + preprocessing (không đợi đến tuần 6)

### Tuần 5: Ablation + Validation

- [ ] **Ablation study** (đóng góp chính cho điểm báo cáo):
  - Bỏ subset features → metric đổi thế nào?
  - Bỏ component architecture (attention, residual, augmentation...) → metric đổi thế nào?
  - Class weighting on/off
  - Different loss functions
- [ ] Multi-run với 3-5 seeds khác nhau → report mean ± std (stability)
- [ ] Thử ensemble nếu còn thời gian (XGBoost + DL variant best)
- [ ] Tiếp tục viết báo cáo: methodology + experiments + results

### Tuần 6: Hoàn thiện

- [ ] Viết phần kết quả + discussion + kết luận + future work
- [ ] Polish: figures (training curves, confusion matrix per class, ablation bar chart), tables, citations
- [ ] Buffer 2 ngày cho lỗi cuối phút (luôn có)
- [ ] Kiểm tra code: clean, comment, README, reproducible với seed cố định
- [ ] Final review báo cáo + sync với manhngvu để 2 báo cáo consistent về phần chung (intro, dataset, preprocessing)

---

## 5. Deliverables

**Code repository**:
- `data/` — script parse PROVEDIt, build labels, split sets
- `features/` — feature engineering pipeline
- `models/` — baseline (XGBoost) + variants (MLP, CNN, Transformer)
- `train.py`, `evaluate.py`, `ablation.py`
- `configs/` — hyperparameters per experiment
- `README.md` — setup, reproduce instructions

**Báo cáo cá nhân** (~10-20 pages):
- Intro + bài toán (phần chung với manhngvu)
- Related work
- Dataset + preprocessing (phần chung với manhngvu)
- **Methodology — phần riêng** (45-class classifier, architecture choices, training)
- **Experiments + results — phần riêng** (variants comparison, tuning, integration)
- **Ablation study — phần riêng**
- Discussion + limitations
- Kết luận + future work

**Results artifacts**:
- Metrics tables (per variant, per class)
- Training curves (loss, F1 over epochs)
- Confusion matrix / per-class P-R
- Ablation table

---

## 6. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| PROVEDIt naming parse sai → labels sai | Medium | High | Validate 50 mixtures bằng tay, cross-check với manhngvu |
| Class imbalance nặng (một số donors xuất hiện ít trong mixtures) | High | Medium | Class weighting, oversampling, focal loss |
| DL không vượt XGBoost trên tabular | High | Medium | Báo cáo cả 2 phương pháp, tập trung novelty (architecture, training trick) thay vì chỉ chasing metric |
| Sync với manhngvu không đều → chồng chéo work | High | High | Sync 2-3 ngày/lần, shared doc track progress |
| Pipeline manhngvu không share được hoặc format không match | Medium | High | Tuần 1 build pipeline backup riêng, không phụ thuộc 100% vào manhngvu |
| Đi sai task (multi-class vs multi-label) | Low | Critical | Confirm với thầy NGAY trước khi code |
| Reproduction tốn thời gian hơn dự kiến | Medium | High | Cắt scope: bỏ variant 3, giữ 2 variant tuần 2 |

---

## 7. Check-ins & Sync points

- **Daily log**: viết 5 dòng cuối ngày — hôm nay làm gì, blocker gì, mai làm gì
- **Sync với manhngvu**: 2-3 ngày/lần, tối thiểu 2 lần/tuần. Format: progress + blocker + next steps
- **Check-in với thầy**: cuối mỗi tuần, gửi 1 paragraph progress + câu hỏi cần giải đáp
- **Milestone reviews**:
  - Cuối tuần 1: baseline numbers có trong tay
  - Cuối tuần 3: 3 variant compared, chọn best
  - Cuối tuần 4: integration với manhngvu hoàn tất
  - Cuối tuần 5: ablation done, draft báo cáo 70%
  - Cuối tuần 6: deliverables đầy đủ

---

## 8. Notes & References

### Bài báo tham khảo
- **Phan et al. 2021** — "High-performance deep learning pipeline predicts individuals in mixtures of DNA using sequencing data" — bài gần nhất, nhưng dùng MPS sequencing data và chỉ 6 individuals. Tham khảo kiến trúc, không reproduce trực tiếp được.
- **Petty et al. 2025 (arXiv)** — "Bayesian Forensic DNA Mixture Deconvolution Using a Novel String Similarity Measure" — Bayesian approach, useful cho discussion + comparison.
- **deepNoC 2024**, **TAWSEEM 2022** — bài về NoC trên PROVEDIt, tham khảo preprocessing pipeline.
- **Probabilistic genotyping**: STRmix, EuroForMix, LRmix — state-of-the-art truyền thống của field, đề cập trong related work.

### Dataset references
- PROVEDIt main: https://lftdi.camden.rutgers.edu/provedit/files/
- manhngvu binary detection baseline (bài cũ): https://huggingface.co/datasets/manhngvu/dna_noc

### Tools
- Python, PyTorch
- XGBoost, CatBoost (tabular baseline)
- Optuna (hyperparameter tuning)
- Weights & Biases hoặc TensorBoard (logging)
- scikit-learn (metrics, splitting)

---

**Lần cập nhật cuối**: 2026-05-09
**Trạng thái**: Draft v1 — chờ confirm task từ thầy và phân vai cụ thể với manhngvu