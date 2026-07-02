# Kế hoạch: Sửa Split + Sinh Data Synthetic (gold-standard) + Validate trên Real

> Mục tiêu: loại bỏ leakage cấu trúc trong evaluation và chứng minh khả năng
> generalize sang **hỗn hợp mới** bằng mô phỏng synthetic **chuẩn forensic**
> (`simDNAmixtures` — cùng công cụ deepNoC), rồi validate trên hỗn hợp PROVEDIt
> thật theo protocol zero-shot novel-combo.
> Scale-up (thêm donor / cross-kit) **gác lại sau**.

---

## 0. Bối cảnh & vấn đề (đã xác minh bằng dữ liệu)

PROVEDIt GF29cycles, closed-set 8,829 mẫu, hiện split random seed=42 → 6180/1324/1325.

**Phát hiện leakage cấu trúc:**
- Trùng file train/test: **0** (row-level sạch ✓); open-set tách biệt ✓
- NHƯNG: **100% tổ hợp donor trong test có trong train**; 0 tổ hợp mới.
- Multi-person: **206/206 (100%)** mẫu NOC≥2 có tổ hợp đã thấy trong train.

**Nguyên nhân gốc — không gian hỗn hợp quá nhỏ:**

| NOC | #tổ hợp duy nhất | #mẫu | replicate/tổ hợp |
|---|---|---|---|
| 1 | 45 (đủ 45 donor) | 7,496 | ~167 |
| 2 | 5 | 334 | ~67 |
| 3 | 6 | 385 | ~64 |
| 4 | 4 | 242 | ~60 |
| 5 | 5 | 372 | ~74 |

→ Chỉ **20 hỗn hợp multi-person riêng biệt**. Exact Match cao ở NOC≥2 (NOC2=1.000)
**một phần do memorization tổ hợp**, chưa chứng minh generalize sang hỗn hợp mới.

---

## So sánh với deepNoC (Taylor & Humphries 2024) — định vị phương pháp

| | deepNoC | Của chúng ta |
|---|---|---|
| Task | Đếm NoC (donor-agnostic) | **Contributor identification** (45 donor cụ thể) |
| Genotype synthetic | Sample từ allele freq DB | **45 genotype donor THẬT** (identity quan trọng) |
| Mức fidelity cần | Thô (đếm allele/peak) | **Mịn (allele-level)** để biết donor nào |
| Mô phỏng | simDNAmixtures + **GAN (EPG)** + peak detection | simDNAmixtures **(chỉ tầng peak-info)** |
| Eval | fine-tune real → test real | **zero-shot novel-combo** (sạch hơn) + biến thể fine-tune |

**Điểm mấu chốt:**
- Dùng **cùng simulator `simDNAmixtures`** với deepNoC → directly comparable, defensible.
- **Bỏ tầng GAN/EPG**: model của ta ăn token peak-level (locus, allele, height),
  KHÔNG ăn EPG thô → chỉ cần tầng peak-info của simDNAmixtures. Đơn giản hơn deepNoC.
- Identification **nâng chuẩn realism** (cần fidelity allele-level, không chỉ đếm) →
  realism-validation là **bắt buộc**, không tùy chọn.
- Chưa có tiền lệ "synthetic cho identification" → đây là đóng góp mới, cũng là lý do
  phải validate kỹ.

---

## Nguyên tắc thiết kế (đọc trước khi code)

1. **Single-source (NOC=1) = reference database.** "Đã thấy donor X" là HỢP LỆ.
   Test NOC=1 trên replicate khác của donor đã biết là công bằng. Không hold-out donor.
   Single-source thật được dùng làm (a) nguồn trích genotype, (b) reference trong train.

2. **Multi-person (NOC≥2) = test trên tổ hợp CHƯA thấy.** Chia không gian tổ hợp
   thành {train combos} và {test combos} rời nhau. Khi hold out {31,32}, donor 31 và 32
   VẪN có mặt trong train (single-source + tổ hợp khác). Đúng kịch bản "nhận ra 31 và 32
   trong một cặp chưa từng ghép".

3. **Synthetic làm TRAIN chính; real multi-person làm TEST.** 20 hỗn hợp thật quá ít để
   train. Train trên synthetic (tổ hợp không giới hạn) → test zero-shot trên real.

4. **Donor-pool isolation:** genotype + (nếu trộn) single-source replicate dùng để
   sinh synthetic chỉ lấy từ train-split → không peek vào replicate dùng cho test thật.

---

## Phase 1 — Sửa split & audit leakage (LÀM TRƯỚC)

### Tasks
- [x] **1.1** Tạo `audit_split.py` — kiểm tra leakage tái sử dụng được:
  - In: (a) trùng filename giữa split; (b) overlap tổ hợp donor train↔val↔test;
    (c) overlap mixture-design (combo+ratio); (d) phân bố NOC mỗi split;
    (e) với NOC≥2: % mẫu test có tổ hợp mới.
  - Exit code ≠ 0 nếu phát hiện multi-person combo leakage (dùng trong verify/CI).
- [x] **1.2** Sửa `data/prepare_data_set.py` — logic split mới:
  - Tách `closed_idx` → `single_idx` (NOC=1) và `multi_idx` (NOC≥2).
  - **Single-source:** `train_test_split(single_idx, stratify=donor_id, seed)` →
    mọi donor có mặt train (reference) + test (recognition).
  - **Multi-person:** group-aware theo tổ hợp — gán **nguyên tổ hợp** vào
    train/val/test, stratify nhóm tổ hợp theo NOC. Số combo nhỏ (4–6/NOC):
    vd NOC=2 (5 combo) → ghi rõ combo nào ở split nào vào meta để tái lập.
  - Lưu `split_policy` + danh sách combo held-out vào `meta_set.json`.
- [x] **1.3** Regenerate feature arrays + `meta_sample_names_{split}.json`.
- [x] **1.4** Chạy `audit_split.py` → xác nhận multi-person test combos = NOVEL. **PASS 0 errors.**
- [x] **1.5** Train lại model real-only split mới → ghi nhận **EM tụt ở NOC≥2**
  (kỳ vọng, trung thực). Đây là baseline "real-only, no-leak".
  **Kết quả: Macro F1=0.8961, EM=0.7894; NOC1=0.952, NOC2=0.023, NOC3-5=0.000**

### Files
- Tạo: `audit_split.py`
- Sửa: `data/prepare_data_set.py` (split, dòng ~205–227); đồng bộ `extract_metadata.py`

### Acceptance
- `audit_split.py`: multi-person test combos NOT in train = 100% novel.
- Mỗi split đủ NOC 1–5. Ghi lại "real-only no-leak baseline".

### Rủi ro
- Combo ít → per-NOC variance cao. Mitigation: báo cáo kèm n nhỏ + nhấn mạnh đây
  chính là lý do cần synthetic. NOC=4 (4 combo) → 2 train/1 val/1 test, ghi rõ.

---

## Phase 2 — Sinh synthetic chuẩn forensic bằng `simDNAmixtures` (gold-standard)

### Tổng quan kiến trúc
```
single-source THẬT  --(2.1 trích genotype)-->  45 donor genotypes (24 loci)
                                                      |
        kit config GlobalFiler + model params  --(2.2 simDNAmixtures R)-->  peak list
        (template/degradation/stutter, Mx)                                  (locus,allele,height)
                                                      |
                              (2.3 convert)  -->  tokens/mask/Xflat/y/noc (đúng bins meta_set.json)
                                                      |
                              (2.4 realism validation vs 20 real mixtures)
```

### Tasks
- [x] **2.0** Cài môi trường R + package:
  - R ≥ 4.x; `install.packages("simDNAmixtures")` (Kruijver & Bright 2022, ref [29]
    trong deepNoC); kèm dependency `pesatools`/allele freq nếu cần.
  - Cầu nối: viết script R xuất CSV, gọi từ Python bằng `subprocess` (đơn giản,
    không cần rpy2). Ghi rõ version vào plan/paper.
- [x] **2.1** `synth/extract_genotypes.py` — trích genotype 45 donor:
  - Từ single-source (NOC=1) thuộc **train-split** của mỗi donor.
  - Mỗi locus: lấy **consensus allele** (1–2 allele xuất hiện ở đa số replicate
    chất lượng cao, lọc stutter/artifact theo height ratio).
  - **QC bắt buộc:** allele trong 20 hỗn hợp THẬT phải nằm trong hợp của genotype
    các donor thành phần → kiểm tra & log tỉ lệ khớp (phát hiện genotype sai).
  - Lưu `data/synth/donor_genotypes.csv` (45 × 24 × {a1,a2}) — artifact reference DB.
- [x] **2.2** `synth/simulate_mixtures.R` — mô phỏng mixture bằng simDNAmixtures:
  - Input: `donor_genotypes.csv`, danh sách tổ hợp + tham số (NOC, Mx, template, seed).
  - Cấu hình kit GlobalFiler (24 loci), peak-height model (gamma/log-normal mặc định
    của package), stutter + degradation.
  - **Calibration thực dụng:** dùng model params mặc định nhưng đặt **range
    template/degradation khớp PROVEDIt** (đã parse: template 0.003–0.76 ng,
    injection 5/15/25s từ `extract_metadata.py`). Ghi rõ: full lab-calibration
    ngoài scope; dùng published params + PROVEDIt-matched ranges (trung thực).
  - Output CSV: mỗi mixture → (sample_id, donors, locus, allele, height, Mx, NOC).
- [x] **2.3** `synth/generate_dataset.py` DONE: 43,800 train + 1,800 val (NOC 2-5 balanced) — orchestrate + convert:
  - Định nghĩa **không gian tổ hợp**: {train combos} (nhiều, kể cả tổ hợp mới chưa
    có trong PROVEDIt) và {test combos} (sẽ test trên real → dùng đúng 20 combo thật,
    chia thành held-in/held-out cho protocol Phase 3).
  - Gọi R (2.2) sinh peak list cho synthetic train/val (chỉ {train combos}).
  - Convert peak list → tokens/mask/Xflat/y/noc dùng allele bins `meta_set.json`.
  - Lưu `data/synth/{tokens,mask,Xflat,y_set,noc}_synth_{train,val}.npy`.
  - Sinh ~vài chục nghìn mixture, phân bố NOC 2–5 cân bằng.
- [x] **2.4** `synth/validate_realism.py` DONE: n_alleles KS=0.105, total_height KS=0.146 (means khop tot); peaks_per_locus lech do filtering artifact — **BẮT BUỘC** (vì identification cần fidelity):
  - So phân bố synthetic vs 20 hỗn hợp THẬT trên: tổng peak height, heterozygote
    balance, #allele/locus, Mx ước lượng, #peaks/profile.
  - KS-test mỗi metric + plot histogram chồng. Mục tiêu: synthetic phủ được real.
  - Nếu lệch lớn → tinh chỉnh template/degradation range ở 2.2 và lặp lại.

### Files
- Tạo: `synth/extract_genotypes.py`, `synth/simulate_mixtures.R`,
  `synth/generate_dataset.py`, `synth/validate_realism.py`
- Đọc: `data/meta_set.json`, single-source train-split, real 20 mixtures (validation)

### Acceptance
- `donor_genotypes.csv` 45 donor; QC khớp với real mixtures ≥ ngưỡng (log rõ).
- Sinh ≥ vài chục nghìn synthetic mixture (gồm tổ hợp mới) qua simDNAmixtures.
- `validate_realism.py`: phân bố synthetic overlap hợp lý real (KS-stat + plot lưu).
- Donor-pool isolation đảm bảo (không replicate test thật bị dùng để trích/sinh).

### Rủi ro / caveat (ghi vào paper)
- Genotype consensus sai → mọi mixture của donor đó sai. Mitigation: QC 2.1.
- Calibration không phải full-lab → ghi nhận; dùng PROVEDIt-matched ranges.
- Phụ thuộc R/simDNAmixtures → ghi version, đóng gói script tái lập.

---

## Phase 3 — Train synthetic → test REAL (zero-shot novel-combo, sạch hơn deepNoC)

### Tasks
- [ ] **3.1** `train_synth.py` (hoặc flag trong `train_set_transformer.py`):
  - Train = `synth_train` (mixture của {train combos}) **+ single-source thật**
    (reference, để head identification học mapping 45 donor).
  - Val = `synth_val`. Giữ kiến trúc/loss/threshold-search hiện tại.
- [ ] **3.2** `eval_on_real.py` — **TEST CHÍNH = real mixtures của {test combos} held-out**:
  - Các combo này **chưa từng** xuất hiện trong synthetic train → **zero-shot
    novel-combo generalization**. Không fine-tune → không leakage.
  - Báo cáo: Macro/Micro F1, Exact Match, **confusion-style per-NOC** + **per-donor F1**
    (giống Table 2 deepNoC), reject AUROC.
- [ ] **3.3** (Biến thể deepNoC-style, tùy chọn) fine-tune nhẹ trên một phần real
  mixtures (held-in combos) → test trên held-out combos → so zero-shot vs fine-tuned.
- [ ] **3.4** Bảng so sánh **3 cấu hình** trong 1 chỗ:
  1. Real-leaky (model cũ, split cũ) — số cao nhưng leak.
  2. Real-only no-leak (Phase 1) — trung thực, n nhỏ.
  3. **Synth-trained → test real zero-shot** (Phase 3) — câu chuyện chính.

### Files
- Tạo: `train_synth.py`, `eval_on_real.py`; sửa `evaluate.py` (chế độ eval real-test).

### Acceptance
- Model synth-trained đạt EM hợp lý trên real novel-combo mixtures (zero-shot).
- Bảng 3-cấu-hình đầy đủ; confusion per-NOC + per-donor F1 trên real test.

---

## Phase 4 — Tích hợp báo cáo & paper

### Tasks
- [ ] **4.1** Limitations (`reports/paper_draft.md`): nêu rõ phát hiện 20-mixture,
  lý do cần synthetic, cách evaluation mới (zero-shot novel-combo) khắc phục.
- [ ] **4.2** Method section: mô phỏng `simDNAmixtures` (genotype 45 donor thật,
  peak model, Mx/template/degradation, bỏ tầng GAN vì input peak-level), donor-pool
  isolation, realism validation. **Cite deepNoC** làm tiền lệ paradigm + nêu khác biệt
  (NoC counting → identification, raises fidelity bar).
- [ ] **4.3** Thêm bảng 3-cấu-hình + realism-validation plot vào `print_paper_tables.py`
  và `generate_report3.py`.
- [ ] **4.4** Cập nhật memory (`results.md`, `project_state.md`).

### Acceptance
- Paper có: thừa nhận trung thực vấn đề cũ; phương pháp simDNAmixtures rõ ràng,
  comparable deepNoC; số liệu synth→real zero-shot defensible; realism validation.

---

## Thứ tự thực thi & ước lượng

| Phase | Nội dung | Ước lượng | Phụ thuộc |
|---|---|---|---|
| 1 | Fix split + audit | 0.5–1 ngày | — |
| 2 | Genotype + simDNAmixtures + realism | 2–3 ngày (gồm học R package) | Phase 1 |
| 3 | Train synth + test real zero-shot | 1 ngày + GPU train | Phase 2 |
| 4 | Báo cáo/paper | 0.5 ngày | Phase 3 |

**Đường găng:** Phase 1 → 2 → 3. Phase 4 song song cuối. Phase 2 nặng hơn Option A
(do R + calibration) nhưng đổi lại paper-standard + comparable deepNoC.

---

## Tiêu chí nghiệm thu tổng (Definition of Done)

- [ ] `audit_split.py`: 0 multi-person combo leak ở split mới.
- [ ] `donor_genotypes.csv` 45 donor + QC khớp real mixtures.
- [ ] simDNAmixtures sinh mixture tổ hợp-mới; realism validation đạt (KS + plot).
- [ ] Donor-pool isolation: không replicate test thật lọt vào genotype/synthetic.
- [ ] Eval zero-shot novel-combo trên real; confusion per-NOC + per-donor F1.
- [ ] Bảng 3-cấu-hình (leaky / no-leak / synth→real) trong paper tables.
- [ ] Method (simDNAmixtures, cite deepNoC) + Limitations cập nhật trung thực.
- [ ] Memory + paper_tables + báo cáo docx regenerate.

## Non-goals (gác lại)
- Mở rộng database >45 donor (IDPlus29 16 donor mới — lệch loci).
- Cross-kit robustness (GF→Fusion6C) — bonus, làm sau nếu dư thời gian.
- Tầng GAN/EPG của deepNoC (không cần — input là peak token, không phải EPG thô).
- Full lab-calibration của simDNAmixtures (dùng published params + PROVEDIt ranges).
