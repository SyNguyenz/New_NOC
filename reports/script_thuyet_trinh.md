# Script Thuyết Trình
## Nhận dạng cá nhân trong hỗn hợp DNA — Bộ phân loại đa nhãn 45 lớp

---

## SLIDE 1 — Tiêu đề

**Tiêu đề:** Nhận dạng cá nhân trong hỗn hợp DNA: Bộ phân loại đa nhãn 45 lớp trên dữ liệu PROVEDIt

**Thuyết trình:**
Báo cáo này trình bày kết quả cá nhân trong dự án nhóm — xây dựng bộ phân loại đa nhãn (multi-label classifier) 45 lớp xác định các cá nhân đã biết trong một hỗn hợp DNA. Phần OOD detection (phát hiện cá nhân chưa biết) là phần của thành viên khác trong nhóm.

---

## SLIDE 2 — Phương pháp sử dụng trong Báo cáo I

**Tiêu đề slide:** Phương pháp Báo cáo I — Ba variant

**Bài toán:** Cho mẫu hỗn hợp x ∈ ℝ^252, dự đoán y ∈ {0,1}^45, trong đó y_i = 1 khi donor i có mặt trong hỗn hợp. Đây là bài toán multi-label classification, mỗi mẫu có thể có nhiều nhãn dương đồng thời.

**V0 — XGBoost One-vs-Rest (Baseline)**
- Huấn luyện 45 bộ phân loại nhị phân độc lập, mỗi bộ dự đoán sự có mặt của một donor
- Không khai thác tương quan giữa các donor, nhưng mạnh trên tabular data thưa
- `scale_pos_weight=5` bù imbalance (trung bình ~3% sample của mỗi lớp là dương)

**V1 — MLP (Multi-Layer Perceptron)**
- Xử lý vector 252-dim qua các lớp fully-connected, chia sẻ biểu diễn giữa 45 lớp
- Kiến trúc: 252 → [512, BN, ReLU, Drop(0.3)] → [256, BN, ReLU, Drop(0.3)] → 45
- Tham khảo Phan et al. (2021) — backbone phân loại contributor trên sequencing data

**V2 — 1D CNN theo locus**
- Tái cấu trúc 252-dim thành ma trận 2D (24 loci × 33 allele bins) theo cấu trúc sinh học thực tế
- Conv1D học pattern allele combination trong từng locus và tương tác cross-locus
- Tham khảo deepNoC (2024) — xử lý STR profile như tín hiệu 1D

| Lớp (CNN) | Shape đầu ra | Mô tả |
|---|---|---|
| Input (sau LociMatrix) | (B, 24, 33) | 24 loci × 33 allele bins |
| Conv1d(24→64, k=3) + BN + ReLU | (B, 64, 33) | Pattern trong locus |
| Conv1d(64→128, k=3) + BN + ReLU | (B, 128, 33) | Pattern cross-locus |
| AdaptiveMaxPool1d(1) | (B, 128) | Global max pooling |
| Linear(128→256) + ReLU, Linear(256→45) | (B, 45) | Logits |

**Thuyết trình:**
Báo cáo I xây dựng ba phương pháp theo hướng từ đơn giản đến phức tạp dần. XGBoost là baseline không tham số hóa theo cấu trúc dữ liệu; MLP chia sẻ biểu diễn giữa 45 lớp; CNN tận dụng cấu trúc locus-allele. Cả ba đều tiếp cận dữ liệu như vector đặc trưng phẳng.

---

## SLIDE 3 — Tiền xử lý dữ liệu trong Báo cáo I

**Tiêu đề slide:** Tiền xử lý Báo cáo I — Pipeline NOC_DNA → vector 252-dim

**Pipeline (kế thừa từ module NOC_DNA):**

| Bước | Mô tả |
|---|---|
| 1. Melt wide → long | Biến ma trận Allele/Size/Height thành dạng long (mỗi peak một hàng) |
| 2. Lọc ngưỡng | Loại bỏ peaks có Height < 50 RFU (nhiễu nền), Allele = OL (off-ladder) |
| 3. Stutter filter | Loại stutter artifact: peak ở vị trí N−1 có chiều cao < 15% peak chính |
| 4. Pivot genotype | Pivot về ma trận thưa: Sample File × (Locus, Allele) → Height |
| 5. Flatten header | Đổi tên cột thành `Locus_Allele` (ví dụ: `D3S1358_15`, `vWA_17`) |

**Kết quả:** `combined_preprocessed_data.csv` — 3.342 mẫu, 252 cột đặc trưng (binary: 1 nếu allele được phát hiện, 0 nếu không).

**Phân chia dữ liệu:**

| Tập | Số mẫu | Tỷ lệ |
|---|---|---|
| Train | 2.023 | 70% |
| Validation | 434 | 15% |
| Test | 434 | 15% |
| Open-set (OOD) | 451 | — |

**Phân chia donor:** `numpy.random.default_rng(42)` → 45 known, 5 unknown = [6, 21, 26, 40, 50]

**Tiền xử lý đặc trưng cho DL:** `log1p(x)` — nén dải giá trị peak height (50–10.000 RFU) trong khi giữ nguyên zeros (allele vắng mặt). StandardScaler không dùng vì biến zeros thành giá trị âm.

**Thuyết trình:**
Bước pivot ở Bước 4 là thao tác then chốt: từ dữ liệu dạng long (nhiều hàng mỗi mẫu) chuyển về một hàng duy nhất mỗi mẫu với 252 cột. Giá trị cuối cùng là binary — mất toàn bộ thông tin về độ cao đỉnh, tức mất thông tin về tỉ lệ đóng góp của từng donor.

---

## SLIDE 4 — Cấu hình tham số và Huấn luyện trong Báo cáo I

**Tiêu đề slide:** Cấu hình Báo cáo I

**XGBoost:**

| Tham số | Giá trị |
|---|---|
| n_estimators | 300 |
| max_depth | 6 |
| learning_rate | 0.1 |
| subsample / colsample_bytree | 0.8 / 0.8 |
| scale_pos_weight | 5 |
| Chiến lược | MultiOutputClassifier (One-vs-Rest), n_jobs=−1 |

**MLP và CNN (dùng chung training loop):**

| Tham số | Giá trị |
|---|---|
| Optimizer | Adam, lr=1e-3, weight_decay=1e-4 |
| LR scheduler | ReduceLROnPlateau ×0.5, patience=5 |
| Early stopping | patience=15, monitor=val Macro F1 |
| Batch size | 64 |
| Max epochs | 150 |
| Loss | BCEWithLogitsLoss(pos_weight=neg/pos per class) |
| MLP params | 447.021 |
| CNN params | 74.349 |
| Hardware | CPU |

**Thuyết trình:**
Hàm loss BCE với pos_weight là lựa chọn chuẩn cho multi-label imbalance. Giá trị pos_weight ≈ 30× cho các donor ít mẫu — mô hình phải đánh đổi giữa Precision và Recall trên từng lớp. Đây là một hạn chế của phương pháp huấn luyện độc lập từng lớp.

---

## SLIDE 5 — Kết quả đã chạy trong Báo cáo I

**Tiêu đề slide:** Kết quả Báo cáo I — Ba variant trên 434 mẫu test (closed-set)

**Bảng kết quả Báo cáo I:**

| Metric            | XGBoost (Baseline) | MLP (V1) | 1D CNN (V2) |
|-------------------|--------------------|----------|-------------|
| Macro F1          | **0.832**          | 0.760    | 0.752       |
| Micro F1          | **0.884**          | 0.825    | 0.817       |
| Hamming Loss      | **0.007**          | 0.013    | 0.013       |
| Exact Match       | 0.749              | **0.809**| 0.747       |
| Macro Precision   | **0.931**          | 0.709    | 0.694       |
| Macro Recall      | 0.770              | **0.882**| 0.856       |

**Nhận xét:**
- XGBoost dẫn đầu về Macro F1 và Precision — ưu thế của tree-based method trên tabular data thưa
- MLP có Exact Match cao nhất (0.809) nhờ Recall tốt hơn — ít bỏ sót contributor hơn
- Cả ba đều suy giảm khi NOC tăng: Exact Match NOC=1 ≈ 0.90+, NOC=5 ≈ 0.40
- Tập train chỉ có 2.023 mẫu và đặc trưng binary (allele presence) — giới hạn khả năng khái quát hóa

**Thuyết trình:**
Báo cáo I đã thiết lập ba baseline trên bộ dữ liệu với 252 đặc trưng binary mô tả sự hiện diện của allele theo locus. XGBoost vượt trội về F1 tổng thể, nhưng cả ba đều chưa khai thác hai nguồn thông tin quan trọng: độ cao đỉnh (peak height) và cấu trúc tập hợp (set structure) của dữ liệu STR. Đây là xuất phát điểm để xây dựng Báo cáo II.

---

## SLIDE 6 — Hạn chế của Báo cáo I và động lực thay đổi

**Tiêu đề slide:** Tại sao cần thay đổi ở Báo cáo II?

| Hạn chế (Báo cáo I) | Hệ quả | Thay đổi (Báo cáo II) |
|---|---|---|
| Đặc trưng binary — mất peak height | Không phân biệt được tỉ lệ đóng góp; NOC=5 Exact Match ≈ 0.40 | Token (locus, allele, log1p_height) |
| Flat vector 252-dim — áp đặt thứ tự | STR profile không có thứ tự tự nhiên; MLP/CNN không phản ánh cấu trúc thực | Set Transformer — permutation-invariant |
| Pipeline NOC_DNA lọc nhiều mẫu | Tập train chỉ 2.023 mẫu — giới hạn khả năng khái quát | Xử lý raw CSV trực tiếp → 6.180 train |
| Single-task (chỉ classification) | Không phát hiện open-set; cần tích hợp rời rạc với OOD module | Multi-task: cls + reject + NOC |
| Binary pivot bỏ qua stutter/peak ratio | Hai donor có allele giống nhau ở một locus không thể phân biệt qua binary | Peak height phân biệt được qua RFU |

**Luận điểm trung tâm:**
> Dữ liệu STR là *tập hợp các quan sát allele*, không phải vector có thứ tự cố định. Peak height mang thông tin định lượng về tỉ lệ đóng góp. Báo cáo II tái cấu trúc bài toán để phản ánh đúng hai điều này.

**Thuyết trình:**
Kết quả Báo cáo I cho thấy rõ một điểm yếu hệ thống: Exact Match giảm từ 0.90+ ở NOC=1 xuống 0.40 ở NOC=5. Đây không phải vấn đề về kiến trúc model mà là vấn đề về biểu diễn dữ liệu — khi nhiều donor cùng đóng góp, binary features không đủ để phân biệt ai. Peak height, vốn tỉ lệ với lượng DNA của từng donor, là tín hiệu cần thiết. Đồng thời, cấu trúc set phù hợp hơn về mặt nguyên lý so với flat vector.

---

## SLIDE 7 — Phương pháp sử dụng trong Báo cáo II

**Tiêu đề slide:** Phương pháp — Set Transformer với học đa nhiệm

**Hai điểm cốt lõi:**

**1. Set Transformer (Lee et al., ICML 2019)**
- Hỗn hợp DNA là một *tập hợp các quan sát* (set of observations): mỗi allele được phát hiện là một phần tử, không có thứ tự tự nhiên
- Set Transformer học được biểu diễn bất biến với hoán vị (permutation-invariant) qua cơ chế attention
- Kiến trúc: ISAB × 2 (Induced Set Attention Block) → PMA_1 (Pooling by Multihead Attention) → z_mix

```
Input tokens: {(locus_idx, allele_val, log1p_height)} per observed allele
Encoder:  ISAB(128, 4 heads, m=32) × 2
Pooling:  PMA_1 → z_mix ∈ R^128
Heads:    cls_head (45 logits), reject_head (1 logit), noc_head (6 logits)
```

**2. Học đa nhiệm (Multi-task Learning)**
Hàm loss kết hợp ba mục tiêu:

```
L = BCE(45 logits, pos_weight) + 0.5 × BCE(reject) + 0.3 × CE(NOC)
```

- `BCE(cls)`: nhận dạng 45 donor (mục tiêu chính)
- `BCE(reject)`: phân biệt closed-set vs. open-set (has-unknown contributor)
- `CE(NOC)`: ước lượng số người đóng góp (auxiliary task)

**Thuyết trình:**
Điểm khác biệt cốt lõi so với Báo cáo I là: thay vì vector đặc trưng phẳng 252-dim, chúng tôi biểu diễn mỗi mẫu như một tập hợp token allele. Set Transformer phù hợp tự nhiên với cấu trúc này vì dữ liệu STR vốn không có thứ tự. Ngoài ra, mô hình được huấn luyện đồng thời trên ba mục tiêu: nhận dạng donor, phát hiện unknown, và ước lượng NOC.

---

## SLIDE 8 — Tiền xử lý dữ liệu trong Báo cáo II

**Tiêu đề slide:** Tiền xử lý — Từ raw CSV đến set-structured tokens

**Quy trình (so sánh với Báo cáo I):**

| Yếu tố              | Báo cáo I                        | Báo cáo II                              |
|---------------------|----------------------------------|-----------------------------------------|
| Nguồn dữ liệu       | combined_preprocessed_data.csv   | Raw CSVs — GF29cycles Filtered          |
| Đặc trưng           | 252-dim binary (allele presence) | Set tokens (locus, allele, log1p_height)|
| Giá trị đặc trưng   | 0/1 (có/không)                   | log1p(RFU) — giữ thông tin cường độ    |
| Flat features       | 252 cột                          | 590 cột (log1p height per allele bin)   |
| Số mẫu              | 3.342 (all kits combined)        | **10.195** (GF29cycles only)            |
| Tập train           | 2.023                            | **6.180**                               |
| Tập test            | 434                              | **1.325**                               |
| Open-set            | 451                              | 1.366                                   |

**Token format cho Set Transformer:**
```
Mỗi allele được phát hiện (non-OL, height > 0) → 1 token:
  dim 0: locus_idx  (int, 0–23, dùng Embedding(24, 16))
  dim 1: allele_val (float: X→-2, Y→-1, numeric→float)
  dim 2: log1p(height)  (float)

Max sequence length = 160 tokens (p99 = 158 trên GF29cycles)
Padding mask: True = valid token
```

**Phân chia donor:** seed=42, 45 known / 5 unknown [6, 21, 26, 40, 50] — giữ nguyên từ Báo cáo I.

**Thuyết trình:**
Thay đổi quan trọng nhất so với Báo cáo I là tích hợp thông tin peak height — một nguồn thông tin bị bỏ qua trước đây. Peak height phản ánh tỉ lệ đóng góp của mỗi donor, giúp phân biệt các cá nhân trong hỗn hợp phức tạp. Ngoài ra, việc xử lý trực tiếp từ raw CSV của kit GF29cycles cho phép mở rộng tập dữ liệu từ 3.342 lên 10.195 mẫu.

---

## SLIDE 9 — Cấu hình tham số và Huấn luyện trong Báo cáo II

**Tiêu đề slide:** Cấu hình — Set Transformer và 5 baselines

**Cấu hình Set Transformer:**

| Tham số           | Giá trị   | Lý do                                     |
|-------------------|-----------|-------------------------------------------|
| d_model           | 128       | Đủ lớn cho 24 loci, không overfit         |
| n_heads           | 4         | 4 × 32 = 128, đủ capacity per head       |
| n_isab            | 2         | Trade-off giữa depth và training time     |
| m_inducing        | 32        | O(N×m) thay vì O(N²) — tiết kiệm bộ nhớ  |
| d_locus embedding | 16        | Compact encoding cho 24 loci              |
| dropout           | 0.1       | Nhẹ vì dataset lớn hơn                    |
| Tổng params       | **1,025,732** | ~1M                                   |

**Cấu hình huấn luyện:**

| Tham số           | Giá trị                   |
|-------------------|---------------------------|
| Optimizer         | AdamW, lr=3e-4, wd=1e-4   |
| LR scheduler      | ReduceLROnPlateau ×0.5 (patience=5) |
| Early stopping    | patience=15, monitor=val Macro F1 |
| Batch size        | 64 closed + 16 open (reject ratio=0.25) |
| Max epochs        | 100                        |
| pos_weight (BCE)  | neg/pos per class ≈ 5×     |
| α (reject loss)   | 0.5                        |
| β (NOC loss)      | 0.3                        |
| Threshold search  | Val set, range [0.2, 0.8], step 0.05 |
| Hardware          | RTX 3050 Laptop 4GB        |
| Thời gian         | ~10 phút (100 epochs)      |

**5 Baselines (cùng tập dữ liệu, StandardScaler):**
- Per-donor Logistic Regression (`MultiOutputClassifier`, C=1, balanced)
- kNN multi-label (k=11, cosine distance)
- XGBoost per-donor (300 trees, depth=6, scale_pos_weight=5)
- MLP 590→512→256→45 (BCE + pos_weight, Adam, early stop)
- 1D CNN (24 loci × 72 allele bins → Conv1D × 2 → MaxPool → 45)

**Thuyết trình:**
Threshold tối ưu 0.80 — cao hơn mặc định 0.50 — cho thấy mô hình khá conservative, chỉ dự đoán positive khi xác suất cao. Tất cả baselines và Set Transformer được huấn luyện và đánh giá trên cùng train/val/test split để đảm bảo so sánh công bằng.

---

## SLIDE 10 — Kết quả đạt được trong Báo cáo II

**Tiêu đề slide:** Kết quả — So sánh trên tập test (1.325 mẫu)

**Bảng kết quả Báo cáo II:**

| Mô hình              | Macro F1     | Micro F1     | Recall       | Precision    | Exact Match  |
|----------------------|--------------|--------------|--------------|--------------|--------------|
| **Set Transformer**  | **0.9813**   | **0.9821**   | **0.9737**   | 0.9895       | **0.9653**   |
| XGBoost              | 0.9736       | 0.9779       | 0.9502       | **0.9996**   | 0.9449       |
| MLP                  | 0.9731       | 0.9756       | —            | —            | 0.9426       |
| LR                   | 0.9622       | 0.9669       | 0.9313       | 0.9968       | 0.9238       |
| CNN                  | 0.9622       | 0.9648       | —            | —            | 0.9253       |
| kNN                  | 0.8832       | 0.9000       | 0.8014       | 0.9963       | 0.7766       |

**Exact Match theo NOC — Set Transformer:**

| NOC | Exact Match | n mẫu |
|-----|-------------|--------|
| 1   | 0.966       | 1.119  |
| 2   | **1.000**   | 46     |
| 3   | 0.984       | 63     |
| 4   | 0.939       | 33     |
| 5   | 0.922       | 64     |

**Reject head — Open-set detection:**
- AUROC = **1.000** (phân biệt hoàn hảo closed-set vs. open-set trên 1.325 + 1.366 mẫu)

**Thuyết trình:**
Set Transformer vượt XGBoost — baseline mạnh nhất — trên mọi metric trừ Precision (0.9895 vs 0.9996). Điểm cải thiện đáng chú ý nhất là Exact Match tăng từ 0.9449 lên 0.9653, tức ~20 mẫu được dự đoán đúng hoàn toàn hơn. Với NOC=5 (hỗn hợp 5 người — phức tạp nhất), Exact Match vẫn đạt 0.922, cải thiện đáng kể so với ~0.40 ở Báo cáo I.

---

## SLIDE 11 — Đóng góp

**Tiêu đề slide:** Đóng góp

1. **Tái cấu trúc biểu diễn dữ liệu:** Chuyển từ vector đặc trưng binary 252-dim sang tập token allele có peak height (locus, allele, log1p_height), khai thác thông tin cường độ tín hiệu bị bỏ qua trong pipeline cũ.

2. **Mở rộng tập dữ liệu:** Xử lý trực tiếp từ raw CSV thay vì file đã tiền xử lý, tăng từ 3.342 lên 10.195 mẫu.

3. **Set Transformer với học đa nhiệm:** Kiến trúc bất biến với hoán vị phù hợp tự nhiên với cấu trúc tập hợp của dữ liệu STR; huấn luyện đồng thời ba mục tiêu (contributor identification, open-set detection, NOC estimation).

4. **Benchmark hệ thống:** Năm baseline trên cùng dữ liệu và split, cho phép so sánh công bằng; Set Transformer vượt tất cả baseline về Macro F1 (0.9813) và Exact Match (0.9653).

5. **Open-set detection tích hợp:** Reject head với AUROC = 1.000 trên tập test, đặt nền tảng cho tích hợp với module OOD của nhóm.

---

*Tài liệu tham khảo chính:*
- Lee et al. (2019). Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks. ICML.
- Zaheer et al. (2017). Deep Sets. NeurIPS.
- Taylor (2024). deepNoC: Deep learning for Number of Contributors on PROVEDIt.
- PROVEDIt Dataset. Rutgers University LFTDI.
