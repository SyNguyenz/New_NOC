# Giải thích toàn bộ dự án: Nhận dạng cá nhân trong hỗn hợp DNA

> Tài liệu này giải thích từ đầu cho người không có nền tảng về sinh học phân tử hay học máy.

---

## 1. Bài toán là gì?

### 1.1 Hỗn hợp DNA trong pháp y

Khi cảnh sát thu thập dấu vết sinh học tại hiện trường — chẳng hạn máu, tóc, hoặc mồ hôi — mẫu vật đó thường chứa DNA của nhiều người cùng một lúc. Ví dụ, một vết máu có thể chứa DNA của nạn nhân lẫn hung thủ. Đây được gọi là **hỗn hợp DNA** (DNA mixture).

Câu hỏi đặt ra là: **mẫu này chứa DNA của những ai?**

Các hệ thống truyền thống như STRmix hay EuroForMix giải quyết câu hỏi này bằng mô hình xác suất phức tạp, đòi hỏi chuyên gia vận hành và mất nhiều giờ để phân tích một mẫu. Dự án này áp dụng học máy để tự động hóa bước nhận dạng.

### 1.2 Bài toán cụ thể

Có một cơ sở dữ liệu **45 người đã biết**. Khi nhận một mẫu hỗn hợp mới, cần xác định: **trong số 45 người đó, ai có mặt trong hỗn hợp này?**

Đây là bài toán **phân loại đa nhãn (multi-label classification)**:
- Đầu vào: một mẫu hỗn hợp DNA
- Đầu ra: vector nhị phân 45 chiều — `y[i] = 1` nếu người thứ `i` có mặt, `y[i] = 0` nếu không

Ví dụ: nếu mẫu chứa DNA của người số 3 và người số 17, thì đầu ra là vector có giá trị 1 tại vị trí 3 và 17, còn lại là 0.

### 1.3 Giới hạn của bài toán

Bài toán có hai phần:

- **Closed-set identification**: xác định người đóng góp từ danh sách 45 người đã biết (đầu ra 45-bit).
- **Open-set detection**: phát hiện khi mẫu chứa ít nhất một người *không* nằm trong danh sách 45 người đó (còn gọi là "unknown contributor"). Model cần trả lời "có người lạ hay không?" thay vì cố ép vào một trong 45 nhãn.

Để phục vụ cả hai, model được huấn luyện với **reject head** — một đầu ra riêng dự đoán xác suất "có người lạ". 5 người được giữ lại (donors 6, 21, 26, 40, 50) tạo thành tập open-set để huấn luyện và đánh giá đầu ra này.

---

## 2. DNA và STR là gì?

### 2.1 DNA là gì?

DNA (deoxyribonucleic acid) là phân tử mang thông tin di truyền trong mỗi tế bào người. Mỗi người có một bộ DNA gần như duy nhất (trừ cặp sinh đôi cùng trứng).

### 2.2 STR là gì?

**STR** (Short Tandem Repeat — vi tandem lặp ngắn) là những đoạn DNA ngắn lặp đi lặp lại tại các vị trí cố định trên nhiễm sắc thể, gọi là **locus** (số nhiều: loci). Ví dụ, tại locus D3S1358, một người có thể có đoạn "AGAT" lặp 14 lần, một người khác lặp 15 lần.

Số lần lặp này gọi là **allele**. Mỗi người có **2 allele tại mỗi locus** (một từ bố, một từ mẹ).

Kit GlobalFiler (dùng trong dự án này) phân tích **24 loci** chuẩn. Đây là kit được FBI và nhiều cơ quan pháp y thế giới sử dụng.

### 2.3 Phân tích STR cho ra kết quả gì?

Máy phân tích điện di (capillary electrophoresis) đọc mẫu DNA và cho ra:
- **Allele**: giá trị allele phát hiện được (ví dụ: 14, 15, 17.2)
- **Size**: kích thước đoạn DNA (đơn vị base pair)
- **Height (RFU)**: độ cao đỉnh tín hiệu, đơn vị Relative Fluorescence Unit — **tỉ lệ với lượng DNA của người đó trong hỗn hợp**

Trong một hỗn hợp 2 người, nếu người A đóng góp 70% DNA và người B đóng góp 30%, thì đỉnh allele của người A sẽ cao hơn đỉnh của người B khoảng 2.3 lần.

---

## 3. Dữ liệu PROVEDIt

### 3.1 PROVEDIt là gì?

**PROVEDIt** (Proficiency Review of Evidentiary DNA Interpretation Tools) là bộ dữ liệu mở do Đại học Rutgers (Mỹ) tạo lập, bao gồm hàng nghìn mẫu STR được phân tích dưới nhiều điều kiện thực nghiệm.

Phiên bản dùng trong dự án: **RD14-0003**, kit GlobalFiler (3500_GF29cycles), tập **Filtered** (đã lọc nhiễu). Gồm 50 người duy nhất (donor ID từ 1 đến 50), với các loại mẫu:
- **1 người** (single-source): mẫu chỉ chứa DNA của một người
- **2–5 người** (mixture): mẫu chứa DNA của nhiều người theo tỉ lệ khác nhau

### 3.2 Cách tên file mã hóa thông tin

Mỗi file kết quả phân tích có tên theo quy ước chuẩn. Ví dụ:

```
A02_RD14-0003-31_32-1;1-M2c-0.03GF-Q2.0_01.5sec.hid
```

Giải mã:
- `A02` — vị trí giếng trên bảng mẫu (capillary position)
- `RD14-0003` — mã nghiên cứu
- `31_32` — **donor ID 31 và 32** (phân cách bằng `_`)
- `1;1` — tỉ lệ trộn 1:1
- `M2c` — điều kiện thí nghiệm
- `0.03GF` — nồng độ DNA (nanogram) và kit GlobalFiler
- `Q2.0` — chỉ số chất lượng
- `01.5sec` — thời gian tiêm mẫu

Từ tên file, có thể trích xuất chính xác danh sách donor đóng góp — đây là nhãn ground truth.

### 3.3 Quy mô dữ liệu (sau tiền xử lý)

| Tập | Số mẫu | Mô tả |
|---|---|---|
| Train | 6.180 | Dùng để huấn luyện model |
| Validation | 1.324 | Dùng để chọn hyperparameter và early stopping |
| Test | 1.325 | Đánh giá cuối cùng, không dùng trong quá trình train |
| Open-set | 1.366 | Mẫu chứa ít nhất 1 người "lạ" (unknown donor) |

**Lưu ý quan trọng:** 85% mẫu trong tập train là single-source (1 người). Đây là đặc điểm cố hữu của PROVEDIt, không phải lỗi. Nó có nghĩa là model phải học cách nhận dạng hỗn hợp phức tạp từ rất ít ví dụ.

---

## 4. Tiền xử lý dữ liệu

### 4.1 Dữ liệu thô trông như thế nào?

File CSV thô chứa kết quả từ máy điện di. Mỗi hàng là một cặp (mẫu, locus):

| Sample File | Marker | Dye | Allele 1 | Height 1 | Allele 2 | Height 2 | ... |
|---|---|---|---|---|---|---|---|
| A02_RD14-0003-31_32-... | D3S1358 | G | 15 | 1250 | 16 | 890 | ... |
| A02_RD14-0003-31_32-... | vWA | B | 14 | 2100 | 17 | 1800 | ... |

Một mẫu hỗn hợp 2 người tại một locus có thể có 4 allele (2 từ người A + 2 từ người B), mỗi allele kèm theo độ cao đỉnh.

Allele "OL" (off-ladder) là allele không nằm trong thang chuẩn — bị loại bỏ.

### 4.2 Cách tạo đặc trưng

**Cách tiếp cận dự án này (Báo cáo II):**

Mỗi allele quan sát được (không phải OL, có height > 0) trở thành một **token** với 3 giá trị:
```
token = (locus_idx, allele_value, log1p(height))
```

Ví dụ: phát hiện allele 15 tại locus D3S1358 với height 1250 RFU → token = `(3, 15.0, 7.13)` (trong đó 3 là chỉ số của D3S1358, `log(1+1250) ≈ 7.13`).

Mỗi mẫu là một **tập hợp (set)** các token như vậy. Một mẫu hỗn hợp 5 người có thể có tới 100–150 tokens.

**Tại sao dùng log1p?**
- Peak height có thể từ 50 đến 10.000 RFU — dải rộng
- `log1p(x) = log(1 + x)` nén dải này lại (50→3.9, 10000→9.2) giúp mô hình học tốt hơn
- Allele không xuất hiện có height = 0, và `log1p(0) = 0` — giữ nguyên ý nghĩa "vắng mặt"

**Tại sao không dùng binary (0/1)?**
Báo cáo I chỉ dùng binary (allele có mặt hay không). Điều này bỏ mất thông tin về **tỉ lệ đóng góp**: trong hỗn hợp 3 người, allele của người đóng góp 60% sẽ cao gấp 3 lần người đóng góp 20% — thông tin quan trọng để phân biệt ai đóng góp bao nhiêu.

---

## 5. Kiến trúc mô hình

### 5.1 Tại sao dùng Set Transformer?

Vấn đề với MLP và CNN thông thường: chúng nhận **vector có thứ tự cố định**. Nhưng dữ liệu STR là một **tập hợp** (set) — thứ tự các allele không có ý nghĩa sinh học. Nếu đổi thứ tự các allele trong input, kết quả dự đoán phải như nhau.

**Set Transformer** (Lee et al., ICML 2019) được thiết kế để xử lý đúng cấu trúc tập hợp này. Kiến trúc đảm bảo tính **permutation-invariant**: hoán đổi thứ tự các token đầu vào không thay đổi đầu ra.

### 5.2 Kiến trúc Set Transformer

```
Đầu vào: tập hợp tokens {(locus_idx, allele_val, log1p_height)}
          Số tokens khác nhau mỗi mẫu (min=39, max=218, trung bình=100)

Bước 1 — Input Projection:
  locus_idx → Embedding(24 loci, 16 chiều)
  Nối với [allele_val, log1p_height]
  → Linear(18 → 128)  ← mỗi token thành vector 128 chiều

Bước 2 — ISAB × 2 (Induced Set Attention Block):
  Mỗi ISAB có 32 "điểm quy nạp" (inducing points) học được
  Token trong set "hỏi" 32 điểm này → 32 điểm tổng hợp thông tin
  Rồi mỗi token "hỏi lại" 32 điểm → token được cập nhật ngữ cảnh
  → Phức tạp O(N×32) thay vì O(N²) — tiết kiệm bộ nhớ

Bước 3 — PMA (Pooling by Multihead Attention):
  1 vector "hạt giống" (seed) học cách tổng hợp toàn bộ set
  → z_mix ∈ R^128 — biểu diễn tổng hợp của cả hỗn hợp

Bước 4 — Ba đầu ra (multi-task):
  cls_head:    z_mix → Linear(128 → 45)  [45 logits, dùng BCELoss]
  reject_head: z_mix → Linear(128 → 1)   [1 logit, có unknown hay không]
  noc_head:    z_mix → Linear(128 → 6)   [softmax, ước lượng 0-5 người]
```

**Tham số:** ~1 triệu (1,025,732) — nhỏ gọn, phù hợp dataset ~6.000 mẫu.

### 5.3 Hàm loss (Multi-task Learning)

```
L = BCE(cls) + 0.5 × BCE(reject) + 0.3 × CE(NOC)
```

- `BCE(cls)`: Binary Cross-Entropy cho 45 logits — mục tiêu chính. Có `pos_weight` để bù imbalance (trung bình mỗi donor chỉ xuất hiện trong ~3% samples)
- `BCE(reject)`: Binary Cross-Entropy cho đầu ra "có người lạ không" — học từ cả mẫu closed-set (nhãn=0) và open-set (nhãn=1)
- `CE(NOC)`: Cross-Entropy cho ước lượng số người — auxiliary task giúp model học biểu diễn tốt hơn

### 5.4 Các baseline so sánh

Để đánh giá Set Transformer có thực sự tốt hơn không, cần so sánh với các phương pháp đơn giản hơn. Tất cả đều dùng cùng dữ liệu và split:

| Model | Ý tưởng |
|---|---|
| **Logistic Regression** | 45 bộ phân loại tuyến tính độc lập, mỗi bộ cho một donor |
| **kNN** (k=11, cosine) | Tìm 11 mẫu train giống nhất trong không gian 590-dim, lấy nhãn đa số |
| **XGBoost** | 45 cây quyết định gradient boosting độc lập (mỗi cây cho một donor) |
| **MLP** | Mạng nơ-ron 590→512→256→45, chia sẻ biểu diễn giữa 45 lớp |
| **1D CNN** | Reshape 590-dim thành ma trận 24×72 (loci × allele bins), Conv1D học pattern |

---

## 6. Huấn luyện

### 6.1 Class imbalance

Vấn đề: mỗi donor chỉ xuất hiện trong khoảng 3% samples (tức ~185 mẫu dương / ~6.000 mẫu train). Nếu model luôn dự đoán "không có ai", accuracy sẽ đạt 97% nhưng vô dụng hoàn toàn.

Giải pháp: `BCEWithLogitsLoss(pos_weight=5)` — tăng trọng số cho các dự đoán dương sai (false negative) gấp 5 lần so với dự đoán âm sai (false positive). Buộc model chú ý đến minority class.

### 6.2 Quy trình huấn luyện

```
For each epoch:
  1. Lấy batch 64 mẫu từ closed-set train
  2. Lấy thêm 16 mẫu từ open-set (reject_label=1) → train reject head
  3. Forward pass → tính loss L
  4. Backward + AdamW update
  5. Sau epoch: tính val Macro F1 trên 1.324 mẫu val
  6. Nếu val F1 không cải thiện sau 5 epoch → giảm lr × 0.5
  7. Nếu val F1 không cải thiện sau 15 epoch → dừng (early stopping)
```

Tốt nhất ở epoch 90-100. Thời gian: ~10 phút trên RTX 3050 Laptop.

### 6.3 Tìm threshold tối ưu

Sau training, model cho ra xác suất từ 0 đến 1 cho mỗi donor. Mặc định threshold 0.5 nghĩa là: nếu xác suất > 0.5 thì dự đoán "có mặt". Nhưng threshold tối ưu không nhất thiết là 0.5.

Tìm trên tập val: thử threshold từ 0.2 đến 0.8 (bước 0.05), chọn giá trị cho Macro F1 cao nhất trên val. Kết quả: **threshold = 0.80** — model conservative, chỉ dự đoán "có mặt" khi xác suất rất cao.

---

## 7. Đánh giá — Các metric

### 7.1 Macro F1

F1 = harmonic mean của Precision và Recall. **Macro F1** tính F1 cho từng trong 45 lớp rồi lấy trung bình đều nhau (không phụ thuộc vào số mẫu mỗi lớp).

Tốt hơn accuracy cho bài toán imbalanced.

### 7.2 Exact Match (Subset Accuracy)

Tỉ lệ mẫu được dự đoán **hoàn toàn đúng** — tất cả 45 nhãn đúng cùng một lúc. Đây là metric khắt khe nhất: chỉ cần sai một donor là không tính.

Đặc biệt quan trọng trong pháp y: một báo cáo sai tên một người có thể gây hậu quả nghiêm trọng.

### 7.3 Hamming Loss

Tỉ lệ nhãn bị dự đoán sai (trên tổng số nhãn B×45). Càng nhỏ càng tốt.

### 7.4 Exact Match theo NOC

Chia kết quả theo số người đóng góp (NOC = Number of Contributors):
- NOC=1: single-source — dễ nhất
- NOC=5: 5-person mixture — khó nhất

Metric này cho thấy model suy giảm như thế nào khi mixture phức tạp hơn.

### 7.5 Reject AUROC

AUROC (Area Under ROC Curve) của reject head: khả năng phân biệt mẫu closed-set (không có người lạ) và open-set (có người lạ). AUROC = 1.0 là hoàn hảo.

---

## 8. Kết quả

### 8.1 So sánh các model (1.325 mẫu test)

| Model | Macro F1 | Exact Match | Recall | Precision |
|---|---|---|---|---|
| **Set Transformer** | **0.9813** | **0.9653** | **0.9737** | 0.9895 |
| XGBoost | 0.9736 | 0.9449 | 0.9502 | **0.9996** |
| MLP | 0.9731 | 0.9426 | — | — |
| Logistic Regression | 0.9622 | 0.9238 | 0.9313 | 0.9968 |
| 1D CNN | 0.9622 | 0.9253 | — | — |
| kNN | 0.8832 | 0.7766 | 0.8014 | 0.9963 |

Set Transformer vượt tất cả baseline. XGBoost có Precision cao hơn (ít false positive hơn) nhưng Recall thấp hơn (bỏ sót nhiều donor hơn).

### 8.2 Exact Match theo độ phức tạp của hỗn hợp

| Số người trong hỗn hợp | Exact Match | Số mẫu test |
|---|---|---|
| 1 người | 96.6% | 1.119 |
| 2 người | **100%** | 46 |
| 3 người | 98.4% | 63 |
| 4 người | 93.9% | 33 |
| 5 người | **92.2%** | 64 |

Đáng chú ý: với 5 người (phức tạp nhất), vẫn đạt 92.2%. Báo cáo I với XGBoost và binary features chỉ đạt ~40% ở NOC=5 — cải thiện gấp hơn 2 lần.

### 8.3 Phát hiện người lạ (Open-set)

Reject head phân biệt mẫu "chỉ có người biết" vs "có người lạ" với AUROC = **1.000** — hoàn hảo trên tập test. Đây là kết quả đặt nền tảng để tích hợp với module OOD của nhóm.

---

## 9. Tóm tắt đóng góp

1. **Tái cấu trúc biểu diễn dữ liệu**: Từ binary allele-presence → token allele có peak height. Khai thác thông tin tỉ lệ đóng góp, cốt lõi của việc phân biệt contributor trong mixture phức tạp.

2. **Mở rộng dataset**: Xử lý trực tiếp từ raw CSV thay vì file đã tiền xử lý. Dataset tăng từ 3.342 → 10.195 mẫu.

3. **Set Transformer**: Kiến trúc phù hợp tự nhiên với cấu trúc tập hợp của dữ liệu STR; được xác minh tính bất biến hoán vị.

4. **Multi-task learning**: Một model duy nhất cho ba mục tiêu — nhận dạng donor, phát hiện unknown, ước lượng NOC.

5. **Benchmark hệ thống**: 5 baseline trên cùng data/split, so sánh công bằng.

---

## Phụ lục: Glossary

| Thuật ngữ | Giải thích |
|---|---|
| STR | Short Tandem Repeat — đoạn DNA lặp ngắn, dùng để nhận dạng cá nhân |
| Locus (loci) | Vị trí cố định trên nhiễm sắc thể nơi STR được phân tích |
| Allele | Số lần lặp của một STR tại một locus cụ thể của một người |
| RFU | Relative Fluorescence Unit — đơn vị đo cường độ tín hiệu DNA |
| NOC | Number of Contributors — số người đóng góp DNA vào hỗn hợp |
| Closed-set | Tất cả contributor đều có trong database (đã biết) |
| Open-set | Ít nhất một contributor không có trong database (người lạ) |
| Multi-label | Mỗi mẫu có thể thuộc nhiều lớp đồng thời |
| Multi-task | Một model học nhiều mục tiêu cùng lúc |
| ISAB | Induced Set Attention Block — khối attention cho set data |
| PMA | Pooling by Multihead Attention — gộp set thành một vector |
| AUROC | Area Under ROC Curve — đo khả năng phân biệt nhị phân (1.0 = hoàn hảo) |
| Exact Match | Dự đoán đúng hoàn toàn tất cả 45 nhãn cùng lúc |
| Macro F1 | F1 trung bình đều nhau trên tất cả 45 lớp |
| pos_weight | Trọng số tăng cường cho class dương để xử lý imbalance |
| Early stopping | Dừng training khi validation metric không cải thiện sau N epoch |
