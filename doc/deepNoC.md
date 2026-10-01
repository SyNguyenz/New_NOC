# deepNoC — Tóm tắt paper

**Paper**: "deepNoC: A deep learning system to assign the number of contributors to a short tandem repeat DNA profile"
**Tác giả**: Duncan Taylor (Forensic Science SA / Flinders Uni), Melissa Humphries (Uni of Adelaide)
**Năm**: 2024 (arXiv: 2412.09803v1)

---

## 1. Vấn đề giải quyết

Gán NOC (Number of Contributors) cho STR DNA profile — bước đầu tiên trước khi chạy probabilistic genotyping (STRmix). Hiện tại chủ yếu làm thủ công bằng MAC (Maximum Allele Count) hoặc ML đơn giản (Decision Tree, Random Forest), bị giới hạn bởi:
- Cần dữ liệu thật (đắt, ít)
- Lọc artifacts/stutter trước → mất thông tin
- Chỉ dùng summary features (MAC, total peak count)
- Giới hạn NOC 3-5

## 2. Ý tưởng chính

**Dùng GAN simulate EPG signal** → tạo 100,000 profiles (NOC 1-10) → train deep NN → **finetune trên vài trăm profiles thật** từ lab.

Điểm khác biệt so với các phương pháp trước:
- **Không lọc** artifacts/stutter
- **Không áp analytical threshold**
- **Không dùng summary** — dùng raw electrophoretic signal
- Hỗ trợ **NOC 1-10** (không chỉ 1-5)

## 3. Data Pipeline

### 3.1 Simulation (GAN-based)
1. Dùng R package `simDNAmixtures` [29] simulate peak information (locus, allele, height, size)
2. Calibrate theo GlobalFiler kit (Forensic Science SA lab conditions)
3. GAN [16] convert peak info → simulated EPG signal (raw electropherogram)
4. Peak detection (Woldegebriel method [31]) → detect peaks + relabel
5. MHCNN [9] classify peaks → gán **peak label probability (plp)** = P(allelic | not stutter/artifact)
6. → 100,000 labeled profiles, NOC 1-10

### 3.2 Input format
**[24 × 50 × 89]** per profile:
- 24 loci (GlobalFiler)
- Max 50 peaks/locus (padding zeros)
- 89 features/peak:
  - **1-24**: locus one-hot
  - **25**: allele designation (/100)
  - **26**: size in bp (/100)
  - **27**: height in RFU (/33,000)
  - **28**: allele frequency (Australian Caucasian)
  - **29**: peak label probability (plp)
  - **30-53**: stutter relationships (back, double-back, forward, point-2-repeat) — mỗi type 6 features: parent allele, parent height, ratio, expected ratio, parent freq, parent plp
  - **54-77**: parent-of-stutter relationships (same 4 types × 6 features)
  - **78**: total peaks at locus (/100)
  - **79**: total peaks in profile (/1000)
  - **80-89**: expected mixture proportions (top-1 đến top-10 contributor) từ "smart start" algorithm

**Lọc**: peaks có P(artefactual) > 0.97 bị loại khỏi input.

### 3.3 Labeled outputs (6 outputs, multi-task)
1. **Peak proportion allelic** [24×50×1] — P(peak is allelic), continuous
2. **Peak number of alleles** [24×50×21] — one-hot, #alleles at that position (0-20)
3. **Locus mixture proportions** [24×10] — proportion of each donor at locus
4. **Locus number of alleles** [24×20] — one-hot, #alleles at locus (1-20)
5. **Profile mixture proportions** [10] — proportion of each donor overall
6. **Profile NoC** [10] — one-hot, NOC (1-10)

## 4. Architecture

**16 layers** từ input → profile NoC output. Hierarchical:
- **Peak-level** layers → peak outputs (proportion allelic, #alleles)
- **Locus-level** aggregation → locus outputs (mixture proportions, #alleles)
- **Profile-level** aggregation → profile outputs (mixture proportions, NoC)

**Secondary outputs feed back** vào main branch (explainability + additional signal). Trials cho thấy bỏ secondary outputs không ảnh hưởng performance, nhưng giữ lại vì explainability.

## 5. Training

### 5.1 Pre-training (simulated data)
- 100,000 profiles: 90k train / 10k test
- Batch size: 100
- Epochs: 200
- Optimizer: Adam, LR = 1e-5, beta = 0.5
- Loss: MSE (outputs 1, 3, 5) + Categorical CE (outputs 2, 4, 6)
- Accuracy ~72% trên simulated test (NOC 1-10)
- Accuracy tăng log-linear với training size (40% at 50 → 72% at 90k)

### 5.2 Fine-tuning (lab-generated ProvedIt profiles)
- **743 profiles** từ ProvedIt (GlobalFiler, ABI 3500, 25s injection)
  - NOC 1: 68, NOC 2: 175, NOC 3: 158, NOC 4: 186, NOC 5: 156
- Split: 371 train / 372 test (every second profile)
- **2000 epochs** fine-tuning
- Performance ban đầu trên real: 40-60% → sau fine-tune: **~90%**
- Model learns rằng ProvedIt chỉ có NOC 1-5 → không predict >5

> ⚠️ **CẢNH BÁO 1 — split không donor-disjoint.** Nguyên văn paper (mục 2.6):
> *"For fine tune training every second profile was used to train (371 profiles) and the
> remaining (372 profiles) were used to test."* Không có bất kỳ đảm bảo combination-disjoint
> hay donor-disjoint nào, và paper **không liệt kê điều này trong limitations**.
>
> ProvedIt dựng mixture từ pool donor nhỏ, mỗi tổ hợp lặp qua nhiều mức template / tỉ lệ /
> replicate — nên 175 profile 2-người **không thể** đến từ 175 cặp donor khác nhau. Mọi phép
> chia 50/50 đều bắt buộc đặt cùng một tổ hợp donor vào cả hai nửa.
> *(Phần cấu trúc ProvedIt là suy luận từ thiết kế archive, không phải paper nói.)*
>
> **Đo được trên data của ta**: cùng model, cùng feature, chỉ đổi cách chia —
> protocol every-second-profile cho `acc_mix` 0.639 so với combo-disjoint 0.527.
> **Chênh ~11.7 điểm trên mẫu hỗn hợp.**

### 5.3 Kết quả fine-tune (Table 2, ProvedIt test set)

Confusion matrix — **hàng = NOC DỰ ĐOÁN, cột = NOC THẬT** (nhãn nguyên văn của paper:
`Predicted NoC` ở trục dọc, `known NoC` ở trục ngang), tổng 372:

| đoán \ thật | 1 | 2 | 3 | 4 | 5 | **tổng hàng**<br>*(số lần đoán)* |
|---|---|---|---|---|---|---|
| 1 | **34** | 0 | 0 | 0 | 0 | 34 |
| 2 | 0 | **88** | 2 | 1 | 0 | 91 |
| 3 | 0 | 0 | **69** | 2 | 1 | 72 |
| 4 | 0 | 0 | 6 | **82** | 16 | 104 |
| 5 | 0 | 0 | 2 | 8 | **61** | 71 |
| **tổng cột**<br>*(số mẫu thật)* | 34 | 88 | 79 | 93 | 78 | 372 |

Bảng chỉ số họ in kèm:

| NOC | Accuracy | Precision *(nhãn của paper)* | Recall *(nhãn của paper)* | F1 |
|-----|----------|-----------|--------|----|
| 1 | 1.000 | 1.000 | 1.000 | 1.000 |
| 2 | 0.992 | 1.000 | 0.967 | 0.983 |
| 3 | 0.965 | 0.873 | 0.958 | 0.914 |
| 4 | 0.911 | 0.882 | 0.788 | 0.832 |
| 5 | 0.927 | 0.782 | 0.859 | 0.819 |

> ⚠️ **CẢNH BÁO 2 — hai nhãn cột `Precision` và `Recall` bị ĐẢO, và cột `Accuracy` là one-vs-rest.**
>
> Vì hàng là *dự đoán* còn cột là *thật*, dựng lại từ chính ma trận:
>
> | phép tính | ra | trùng cột nào của paper | thực chất là |
> |---|---|---|---|
> | đường chéo ÷ **tổng cột** | 1.000, 1.000, 0.873, 0.882, 0.782 | cột **`Precision`** | **recall chuẩn** |
> | đường chéo ÷ **tổng hàng** | 1.000, 0.967, 0.958, 0.788, 0.859 | cột **`Recall`** | **precision chuẩn** |
> | `(TP+TN)/372` | 1.000, 0.992, 0.965, 0.911, 0.927 | cột **`Accuracy`** | **one-vs-rest** |
>
> Cả **15 giá trị** khớp từng chữ số — không phải trùng hợp. Ví dụ hàng 4: tổng hàng
> `0+0+6+82+16 = 104` → `82/104 = 0.7885` = cột "Recall" 0.788; tổng cột
> `0+1+2+82+8 = 93` → `82/93 = 0.8817` = cột "Precision" 0.882.
>
> Kiểm tra độc lập cho cột `Accuracy`: với 5 lớp và accuracy đa lớp `a`, trung bình
> one-vs-rest phải là `1 − 2(1−a)/5`. Thay `a = 334/372 = 0.8978` → **0.9591**, đúng bằng
> trung bình của `[1.000, 0.992, 0.965, 0.911, 0.927]` = **0.9591**.
>
> **Cột dùng để so sánh với ta là recall chuẩn:**
>
> | | 1p | 2p | 3p | 4p | 5p | macro |
> |---|---|---|---|---|---|---|
> | Họ in trong Table 3 (one-vs-rest) | 1.000 | 0.992 | 0.965 | 0.911 | 0.927 | 0.9591 |
> | **Recall chuẩn** | 1.000 | 1.000 | **0.873** | **0.882** | **0.782** | **0.9074** |
> | *chênh (điểm)* | 0 | −0.8 | **+9.2** | +3.0 | **+14.5** | |
>
> Thổi phồng lớn nhất ở **5 người (+14.5 điểm)** rồi **3 người (+9.2)**, không phải ở 4 người.
>
> **Cột `F1` của họ thì ĐÚNG.** F1 = 2PR/(P+R) đối xứng theo precision và recall, nên việc đảo
> nhãn hai cột không ảnh hưởng tới nó — dựng lại từ ma trận ra `[1.000, 0.983, 0.914, 0.832,
> 0.819]`, khớp cả 5 chữ số. **macro F1 = 0.9097.**
>
> Vì vậy **macro F1 là thước đo an toàn nhất để so với deepNoC**: nó miễn nhiễm với chính lỗi
> nhãn ở trên, và cũng là thước đo notebook decoder-FT báo cáo (0.9456).
>
> *(Bản trước của ghi chú này đọc ma trận lộn hàng–cột, dẫn tới ghi nhầm recall 4p = 0.788 và
> 5p = 0.859. Đã sửa 2026-09-04 sau khi render trực tiếp trang 18 của PDF.)*

**Overall ~90% accuracy**: 334/372 = **0.8978**. ✅ Con số tổng này **là accuracy đa lớp
đúng nghĩa và hoàn toàn đứng vững** — vấn đề chỉ nằm ở bảng per-NOC và bảng so sánh.

## 6. So sánh với các phương pháp khác (Table 3, ProvedIt)

| Method | 1p | 2p | 3p | 4p | 5p |
|--------|----|----|----|----|-----|
| **deepNoC** *(one-vs-rest — đúng số in trong Table 3)* | **1.000** 🟩 | **0.992** 🟩 | **0.965** 🟩 | 0.911 | **0.927** 🟩 |
| **deepNoC** *(recall chuẩn)* | 1.000 | 1.000 | **0.873** | **0.882** | **0.782** |
| PACE-GF | 0.986 | 0.963 | 0.944 | 0.959 🟥 | — |
| TAWSEEM | 0.979 | 0.951 | 0.941 | 0.937 🟩 | 0.969 🟥 |
| MAC | 0.627 | 0.806 | 0.84 | 0.583 | 0.087 |
| Decision Tree | 0.910 | 0.960 | 0.819 | 0.602 | 0.660 |
| NOCIt | 0.806 | 0.939 | 0.84 | 0.825 | 0.573 |
| LDA-26 | 0.925 | 0.970 | 0.904 | 0.825 | 0.806 |
| Human | — | — | 0.915 | 0.618 | 0.768 |

🟩 = ô paper tô XANH (cao nhất). 🟥 = ô paper tô ĐỎ ("may be skewed by experimental
design", bị loại khỏi việc xét cao nhất — luôn là mức NOC cao nhất trong dải của phương pháp đó).
Màu đọc trực tiếp từ bản render trang 23. Ngoài ra TAWSEEM còn có thể bị skew do tính
per-locus thay vì per-profile (chính paper nêu, mục 4.1).

> ⚠️ **CẢNH BÁO 3 — bảng này đang so hai thước đo khác nhau.**
> Hàng deepNoC lấy **nguyên cột one-vs-rest** của Table 2, còn các phương pháp khác
> gần như chắc chắn báo **recall**.
>
> Bằng chứng: **MAC 5p = 0.087**. Nếu là one-vs-rest thì gần như bất khả — MAC là luật
> tất định, hiếm khi đoán 5, nên one-vs-rest của nó phải ~0.8+. Còn hiểu là recall thì 0.087
> đúng y hệt điểm yếu kinh điển của MAC ở 5 người.
> *(Suy luận từ giá trị — chưa đọc Kruijver/Marciano/Alotaibi để xác nhận.)*
>
> **Nếu các phương pháp kia báo recall, thay hàng deepNoC bằng recall chuẩn thì ngôi đầu đổi
> ở 2 trong 4 ô xanh:**
>
> | | deepNoC in ra | deepNoC recall chuẩn | cao nhất trong bảng (bỏ ô đỏ) |
> |---|---|---|---|
> | 1p | 1.000 🟩 | 1.000 | deepNoC **giữ** ✅ *(kế tiếp: PACE-GF 0.986)* |
> | 2p | 0.992 🟩 | 1.000 | deepNoC **giữ** ✅ *(kế tiếp: LDA-26 0.970)* |
> | 3p | 0.965 🟩 | **0.873** | **MẤT** → PACE-GF 0.944, TAWSEEM 0.941, LDA-26 0.904 đều cao hơn |
> | 4p | 0.911 | **0.882** | vốn đã không phải ô xanh (TAWSEEM 0.937 🟩) — kết luận không đổi |
> | 5p | 0.927 🟩 | **0.782** | **MẤT** → LDA-26 0.806 cao hơn *(TAWSEEM 0.969 là ô đỏ, bị loại)* |
>
> Tức là mệnh đề "*deepNoC cao nhất ở phần lớn các mức*" của paper phụ thuộc vào việc hàng
> của nó là one-vs-rest còn các hàng khác là recall. Nếu quy về cùng thước đo, deepNoC giữ
> được 1p và 2p, mất 3p và 5p.
>
> ⚠️ Điều kiện: giả định "các phương pháp kia báo recall" **chưa được kiểm chứng** — nó suy
> ra từ giá trị MAC 5p = 0.087 (xem trên), chưa đọc Kruijver/Marciano/Alotaibi để xác nhận.
> Trước khi đưa vào bài phải mở ba nguồn đó ra đọc.

> ⚠️ **CẢNH BÁO 4 — hàng simulated của Table 3 cũng là one-vs-rest.**
> Hàng đó ghi 1p–10p = `0.9998 … 0.933`, trung bình **0.9426**. Nhưng chính paper nói
> accuracy trên simulated chỉ **72%** (mục 5.1).
>
> Với 10 lớp, nếu accuracy thật = 0.72 thì one-vs-rest kỳ vọng = `1 − 2(1−0.72)/10` = **0.9440**.
> **0.9426 vs 0.9440** — không thể trùng ngẫu nhiên.
>
> Người đọc lướt Table 3 sẽ nghĩ deepNoC đạt ~94% trên simulated. Con số thật là **72%**.

## 7. Key insights cho project của mình

### 7.0 ⚠️ Mốc so sánh đúng khi viết bài

Có **hai nguồn thổi phồng độc lập** trong bảng của deepNoC. Đừng cộng dồn ngây thơ, nhưng
cũng đừng bỏ qua cái nào:

| Nguồn | Ảnh hưởng |
|---|---|
| **Metric** — one-vs-rest thay vì recall | +14.5 điểm ở 5p (0.927 vs 0.782), +9.2 ở 3p, +3.0 ở 4p |
| **Protocol** — trùng donor/combo giữa train-test | +11.7 điểm trên mẫu hỗn hợp *(đo trên data của ta)* |

Số của ta (accuracy đa lớp + macro F1, protocol combination-disjoint) đang chịu **cả hai**
hình phạt. Đặt thẳng cạnh bảng của họ là tự làm hại mình.

**Mốc đúng để so, theo recall chuẩn dựng lại từ Table 2:**

| | 1p | 2p | 3p | 4p | 5p | macro | acc đa lớp |
|---|---|---|---|---|---|---|---|
| deepNoC — recall chuẩn | 1.000 | 1.000 | 0.873 | 0.882 | 0.782 | 0.9074 | **0.8978** |
| **deepNoC — F1 *(cột này của họ đúng)*** | 1.000 | 0.983 | 0.914 | 0.832 | 0.819 | **0.9097** | |

**So bằng macro F1** — cùng protocol every-second-profile, đo trên data của ta:

| | 1p | 2p | 3p | 4p | 5p | macro F1 |
|---|---|---|---|---|---|---|
| deepNoC *(89 feat/peak, GAN simulator, 2000 ep)* | 1.000 | 0.983 | 0.914 | 0.832 | 0.819 | **0.9097** |
| ta, `noc_branch` *(9 feat/peak, không GAN, 15 ep)* | 0.986 | 0.945 | 0.896 | 0.826 | **0.863** | **0.9032** |

Kém 0.0065 tổng thể, **hơn 0.044 ở 5 người**. Dòng combination-disjoint của ta (`posthoc_rf`,
0 tổ hợp chung) là **0.8817** — chênh 0.023 so với dòng trên chính là giá của việc gặp lại tổ
hợp donor quen.

Con số tổng **0.8978 = 334/372** là accuracy đa lớp đúng nghĩa và đứng vững; vấn đề chỉ nằm
ở bảng per-NOC và Table 3.

Cách xử lý đề xuất: báo cáo **cả hai protocol** và coi khoảng cách là một kết quả —
ta là đo được phần nào của hiệu năng NOC đã công bố đến từ việc gặp lại tổ hợp donor quen.

### 7.1 Fine-tuning approach
- deepNoC **CÓ real mixture data** cho fine-tuning (ProvedIt: 175×NOC2, 158×NOC3, 186×NOC4, 156×NOC5)
- Mình chỉ có **5171 single-source** (NOC1) trong real train → **không thể fine-tune NOC counting cho mixtures**
- deepNoC cần **2000 epochs** fine-tuning trên 371 samples → rất lâu, nhưng data đủ đa dạng NOC
- deepNoC accuracy trước fine-tune: 40-60% → sau: 90%. Tương tự domain gap mình thấy

### 7.2 Input features
- deepNoC dùng **89 features/peak** (rất rich) vs mình chỉ **8 features** (locus, allele, log_h, Hb, SR, rank_inv, n/10, glob_rel)
- deepNoC có **stutter relationship features** (back/forward/double-back stutter ratios + parent info) — 48 features dành cho stutter
- deepNoC có **peak label probability** (plp) = P(allelic) — mình không có
- deepNoC có **allele frequency** — mình không có
- deepNoC có **smart-start mixture proportions** (10 features) — tương tự phi output của mình

### 7.3 Architecture differences
- deepNoC: **hierarchical CNN** (peak→locus→profile), spatial structure [24×50×89]
- Mình: **Set Transformer** (attention-based), flat token sequence [N×8]
- deepNoC: **multi-task** (6 outputs, secondary outputs feed back) → explainability
- Mình: **multi-task** nhưng focus ID + NOC (gate counting)

### 7.4 Điều có thể áp dụng
1. **Thêm stutter features** vào tokens: SR/FS đã có, nhưng chưa có parent info (parent height, parent plp, expected stutter ratio). Đây là 48/89 features của deepNoC
2. **Thêm allele frequency** feature: population frequency → giúp phân biệt common vs rare alleles
3. **Peak label probability**: nếu có MHCNN hoặc tương tự, thêm plp feature sẽ giúp lọc noise
4. **Mixture proportions as input**: deepNoC dùng "smart start" mixture proportions → tương tự phi_rerank output. Có thể feed phi estimate ngược lại làm input feature
5. **Fine-tuning cần real mixture data**: nếu muốn fine-tune, phải tạo/thu thập real mixture data (NOC 2-5), không chỉ single-source

### 7.5 Confidence threshold
- deepNoC đề xuất: nếu max probability < threshold → "unclassified"
- Threshold 0.95 → accuracy ~95% nhưng chỉ classify ~90% profiles
- Có thể áp dụng cho gate: nếu sum(gate) không rõ ràng (e.g., 2.5 — NOC 2 hay 3?) → flag "uncertain"

## 8. Limitations noted by authors
- Chưa test cross-population (train Australia, test khác)
- Chưa hỗ trợ multi-PCR replicate
- "Apparent NoC" vs "true NoC" — minor contributor DNA quá ít có thể invisible
- Sensitivity to peak height stochasticity chưa được khảo sát
