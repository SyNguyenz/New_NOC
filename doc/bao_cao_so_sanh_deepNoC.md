# Báo cáo: đếm số người góp mẫu (NoC) so với deepNoC

Ngày 2026-10-02. Viết để làm phần Kết quả và Thảo luận của bài (đích: *Forensic Science International: Genetics*).

**Nguồn số liệu:**
- Đợt **10 fold** chạy trên Kaggle: CODE `529535b846`, dữ liệu `noc-cond50k-fold0..9`, mỗi fold một backbone train riêng.
- Kế hoạch và giả thuyết **ghi trước khi chạy**: `doc/chay_10fold_zeroshot50k.md` (mục 9).
- Số từng fold: `doc/ket_qua_tung_fold.csv`, `doc/ket_qua_noc_tung_fold.md`.

Mọi số dưới đây là **macro F1 của NoC (1–5 người)**. Đây là thước đo duy nhất so được trực tiếp với deepNoC (xem mục 2).

---

## 1. Phạm vi so sánh

- **So trong panel.** Tập test là hồ sơ mà mọi người góp đều thuộc 45 donor có genotype tham chiếu của fold.
  Đây là tình huống tương đương với deepNoC: cả 372 hồ sơ test của họ đều chứa donor đã gặp lúc fine-tune.
- **Hai hệ thống giải hai bài toán khác nhau.**
  - deepNoC chỉ đếm, không cần genotype tham chiếu; định danh làm ở bước sau (STRmix + likelihood ratio).
  - Hệ thống của mình **vừa đếm vừa định danh trong một mô hình**, và cần panel tham chiếu.
- **Không cùng tập hồ sơ cá thể.**
  - deepNoC chấm 372 hồ sơ (GlobalFiler, ABI 3500, 25 s).
  - Mình chấm toàn bộ hồ sơ trong panel của từng fold: 2.240–2.770 hồ sơ mỗi fold, 25.155 hồ sơ cộng dồn 10 fold, gồm cả tiêm 5 / 15 / 25 s.

## 2. Mốc deepNoC (dựng lại từ Table 2 của paper)

Table 2 có hàng là NoC **dự đoán**, cột là NoC **thật**. Cột "Precision" và "Recall" trong paper **bị đổi nhãn cho nhau**;
dựng lại từ ma trận thì khớp cả 15 giá trị. Cột F1 thì đúng, vì F1 đối xứng.

| deepNoC | NOC1 | NOC2 | NOC3 | NOC4 | NOC5 | macro |
|---|---|---|---|---|---|---|
| F1 | 1.000 | 0.983 | 0.914 | 0.832 | 0.819 | **0.9097** |
| recall đúng (đường chéo ÷ tổng cột) | 1.000 | 1.000 | 0.873 | 0.882 | 0.782 | 0.907 |
| accuracy đa lớp | | | | | | 0.898 (334/372) |

- **Khoảng tin cậy 95%** của macro F1 deepNoC: **[0.882, 0.934]**. Bootstrap theo từng hồ sơ trên 372 hồ sơ; họ không công bố tổ hợp donor, nên khoảng thật còn rộng hơn.
- Cột "accuracy" trong Table 2 và Table 3 của họ là **one-vs-rest**, không phải accuracy đa lớp. Không dùng cột đó để so.

## 3. Thiết lập của mình

**Pretrain (Phase 1), mỗi fold:**
- train: 50.000 hỗn hợp **sinh** (`--conditioned`, chỉ từ donor trong panel, không trùng tổ hợp test) cộng hồ sơ **đơn nguồn thật**. Sau khi tách DEV còn khoảng 47.300 hồ sơ.
- validation: DEV in-silico, khoảng 7.800 hồ sơ, tách theo tổ hợp. Dùng để chọn epoch; không có nhãn hỗn hợp thật.

**Đếm không fine-tune (zero-shot):** đọc đầu `logits_card` của backbone. **Không dùng một hồ sơ hỗn hợp có nhãn nào.**

**Fine-tune cùng ngân sách nhãn với deepNoC:**
- Backbone đóng băng, head nhỏ `head_only` (1.093 tham số).
- **Mỗi mô hình chỉ thấy ≤ 371 hồ sơ thật**, chia NOC1–5 = 34 / 87 / 79 / 93 / 78, đúng như deepNoC.
- Thực tế 370–371 nhãn ở mọi fold, trừ fold 9 (359), vì fold đó chỉ có 96 hồ sơ NOC5.
- Không có tập validation: công thức cố định, chọn trên fold 0. Lặp 3 lần rút.

**Hai giao thức chấm fine-tune:**

| | chia | tổ hợp donor giữa fit và chấm |
|---|---|---|
| **strict** (của mình) | test chia 5 phần tách tổ hợp; fit trên 4 phần, chấm phần còn lại | **không trùng**: chấm trên tổ hợp chưa gặp |
| **giao thức deepNoC** | xếp hồ sơ, lấy cách một; fit trên nửa chẵn, chấm nửa lẻ | **trùng**: cùng tổ hợp ở cả hai nửa, giống paper deepNoC |

**Không rò rỉ:** `SHARED = 0` tổ hợp giữa dữ liệu sinh và test ở cả 10 fold. 5 donor của mỗi fold bị giấu hoàn toàn khỏi
train, val và test.

## 4. Kết quả chính

### 4.1. Tổng hợp

| hệ thống | nhãn hỗn hợp thật | giao thức | **macro F1 NoC** | 95% CI |
|---|---|---|---|---|
| **deepNoC** | 371 | xen kẽ (của họ) | **0.910** | [0.882, 0.934] |
| **Mình, zero-shot** | **0** | (không fit) | **0.888** gộp; 0.885 ± 0.040 TB 10 fold | [0.869, 0.905] |
| **Mình, fine-tune `head_only`** | 371 | **giao thức deepNoC** | **0.920 ± 0.012** (fold 1–9) | |
| **Mình, fine-tune `head_only`** | 371 | **strict** | **0.902 ± 0.014** (fold 1–9) | |

CI của mình là bootstrap theo **tổ hợp donor** (659 nhóm), vì các hồ sơ cùng tổ hợp đúng hoặc sai cùng nhau.

### 4.2. Theo từng NoC (F1)

| | NOC1 | NOC2 | NOC3 | NOC4 | NOC5 | macro |
|---|---|---|---|---|---|---|
| deepNoC | 1.000 | 0.983 | 0.914 | 0.832 | 0.819 | 0.910 |
| zero-shot, gộp 10 fold | 0.996 | 0.966 | 0.911 | 0.816 | **0.751** | 0.888 |
| `head_only`, giao thức deepNoC, TB fold 1–9 | 0.992 | 0.961 | 0.917 | **0.872** | **0.859** | 0.920 |
| `head_only`, strict, TB fold 1–9 | 0.991 | 0.952 | 0.900 | 0.839 | 0.829 | 0.902 |

Recall đúng, cùng định nghĩa với dòng recall của deepNoC ở mục 2:
- zero-shot, gộp 10 fold: 0.995 / 0.967 / 0.949 / 0.875 / **0.631**;
- `head_only` strict, gộp fold 1–9: 0.985 / 0.951 / 0.898 / 0.857 / **0.828**;
- deepNoC: 1.000 / 1.000 / 0.873 / 0.882 / 0.782.

### 4.3. Từng fold

| fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| zero-shot | 0.803 | 0.910 | 0.866 | 0.914 | 0.878 | 0.838 | 0.888 | 0.920 | 0.900 | 0.934 |
| `head_only` strict | 0.869* | 0.887 | 0.932 | 0.913 | 0.891 | 0.903 | 0.894 | 0.893 | 0.902 | 0.905 |
| `head_only` giao thức deepNoC | 0.928* | 0.918 | 0.938 | 0.924 | 0.937 | 0.921 | 0.910 | 0.917 | 0.904 | 0.912 |

\* Fold 0 là fold dùng để chọn `head_only`, nên không tính vào số chính của fine-tune.

### 4.4. Định danh (ID), cùng mô hình, cùng lượt chạy

| | TB 10 fold |
|---|---|
| oracle EM (biết trước NoC đúng) | **0.978** |
| EM@k (k = NoC đếm zero-shot) | **0.924** |
| donor F1@k | 0.978 |
| reject AUROC (nhận ra hồ sơ có người ngoài panel) | 1.000 |

deepNoC không định danh, nên phần này không có mốc so trực tiếp.

## 5. Phân tích độ nhạy: chỉ hồ sơ tiêm 25 s (khớp điều kiện deepNoC; **không** chốt trước)

| | TB 10 fold | TB fold 1–9 |
|---|---|---|
| zero-shot | 0.898 | 0.908 |
| `head_only` strict | 0.901 | 0.906 |
| `head_only` giao thức deepNoC | 0.919 | 0.917 |

Fine-tune trên 25 s chỉ có **308–360 nhãn** mỗi lần fit (ít hơn deepNoC), vì tập 25 s nhỏ.

## 6. Lợi thế khi dùng lại tổ hợp donor (giao thức deepNoC trừ strict)

Cùng mô hình, cùng 371 nhãn, chỉ đổi cách chia. TB 10 fold.

| head fine-tune | tham số | strict | giao thức deepNoC | **lợi thế** |
|---|---|---|---|---|
| `card_cal` (hiệu chỉnh `logits_card`) | 6 | 0.900 | 0.908 | +0.008 |
| `head_only` | 1.093 | 0.899 | 0.921 | +0.022 |
| `lora_inv` | ~25.700 | 0.867 | 0.901 | +0.034 |
| `card_ft` (đọc gate theo thứ tự donor) | 230 | 0.836 | 0.914 | +0.078 |
| `head_unf` (đọc toàn bộ peak) | 4.901 | 0.337 | 0.813 | **+0.476** |

Thêm một kết quả, fold 0 và 6 (`doc/failed_experiments.md`, mục 27–28): mạng đếm riêng `noc_deep`, fine-tune **toàn bộ mạng**
theo cách deepNoC, có strict 0.67 / 0.61 nhưng giao thức deepNoC 0.75 / 0.73 (+0.07 / +0.12).

**Ý nghĩa:** giao thức chia xen kẽ của deepNoC **thưởng cho việc học thuộc tổ hợp**. Mô hình càng có nhiều đường đọc danh tính
hoặc dấu vết tổ hợp thì phần thưởng càng lớn. Vì deepNoC fine-tune toàn bộ mạng (2000 epoch) trên phép chia này, một phần
bước nhảy 40–60% → 90% mà họ báo cáo có thể đến từ việc dùng lại tổ hợp. Họ không đo được điều này vì không có phép chia
tách tổ hợp.

## 7. Giả thuyết đã ghi trước, và kết quả

| | nội dung | kết quả |
|---|---|---|
| T1 | zero-shot, TB 10 fold ≥ 0.9097 | **sai** (0.885) |
| T2 | zero-shot, gộp ≥ 0.9097 | **sai** (0.888) |
| T3 | fold 0: bộ dữ liệu tự sinh ≈ bộ của nhóm (lệch ≤ 0.02) | **sai** (0.803 so với 0.910; hai bộ giống nhau về thống kê) |
| T4 | mọi kiểm tra sạch | **đúng** |
| T5 | `head_only` 371 nhãn, strict, fold 1–9 ≥ 0.9097 | **sai** (0.902, thiếu 0.008) |
| T6 | headline là zero-shot gộp | theo luật |
| T7 | hiệu chỉnh prior không nhãn: NOC5 +0.10, fold 4–9 | **sai** (+0.075) |

## 8. Câu viết đề xuất cho bài

> Without any labelled real mixture, our model reached a macro-F1 of 0.888 (95% CI 0.869–0.905) for 1–5 contributors over
> ten donor-held-out folds evaluated on unseen donor combinations, against 0.910 (0.882–0.934) reported for deepNoC after
> fine-tuning on 371 labelled profiles. Given the same budget of 371 labelled profiles, our model reached 0.920 under
> deepNoC's own alternating split and 0.902 under a combination-disjoint split. The 0.02 difference between the two splits
> is the gain from re-encountering donor combinations, and it grew to 0.08–0.48 for count heads with more access to donor
> identity.

**Không viết** "vượt deepNoC" mà không nói rõ giao thức. Ở giao thức strict, chênh lệch nằm trong khoảng tin cậy của deepNoC.

## 9. Phần công bố bắt buộc (đi kèm mọi phép so với deepNoC)

1. Không cùng tập hồ sơ cá thể. Mình dùng toàn bộ hồ sơ trong panel của từng fold, kể cả tiêm 5 / 15 / 25 s.
2. Đầu vào là bảng peak **Filtered** của PROVEDIt, không phải tín hiệu EPG thô như deepNoC.
3. Mô hình dùng **panel 45 genotype tham chiếu** (khởi tạo slot, lọc peak). deepNoC không dùng.
4. Phase 1 dùng hồ sơ **đơn nguồn thật** (nhãn NOC1) và hồ sơ **open thật** để dạy reject head (không có nhãn NoC).
5. Kit: deepNoC ghi GlobalFiler / ABI 3500 / 25 s nhưng không ghi số chu kỳ. `3500_GF29cycles` là cấu hình GlobalFiler duy nhất trong archive PROVEDIt 1–5 người.
6. Dữ liệu sinh là 50.000 hỗn hợp mỗi fold, sinh bằng generator của nhóm (`--conditioned`). Lệnh và md5 nằm trong `build_info.json` của từng fold.
7. Mỗi fold một backbone, **một seed**. Chưa đo dao động giữa các seed; T3 gợi ý mức dao động này có thể đáng kể.

## 10. Giới hạn cần nêu trong bài

- **Gọi thiếu NOC5:** zero-shot có recall NOC5 chỉ 0.63, vì đầu `logits_card` học nhãn suy từ ID. Fine-tune 371 nhãn sửa phần lớn (0.83).
- **NoC phụ thuộc ID:** đầu đếm đọc gate hoặc điểm ID. Ca nào ID không chắc chắn thì đếm cũng kém.
- **Ngoài panel:** NoC giảm mạnh khi có người không thuộc panel. Ngoài phạm vi so sánh; nêu trong phần giới hạn.
- **Khoảng 7–10% hồ sơ NOC5 thật** có người góp gần như vô hình (dưới 30% allele riêng còn hiện). Đây là trần chung cho mọi phương pháp.
- **Dao động khi train:** fold 0 cho 0.80 trên bộ tự sinh, 0.91 trên bộ của nhóm (cùng thống kê dữ liệu).
- Dòng 25 s và các head phụ là phân tích sau, không chốt trước.

## 11. Thăm dò sau đợt 10 fold (không thuộc số chính)

Ghi ở `doc/failed_experiments.md`, mục 24–31.
- Mạng đếm riêng `noc_deep` (không dính ID): khoảng 0.66–0.70, kém đầu đếm dính ID khoảng 0.13 trên cùng dữ liệu.
- Khởi tạo head từ dữ liệu sinh rồi mới fine-tune 371 nhãn: tăng ở 5/5 fold đã thử (TB khoảng +0.02 strict). Chưa chốt trước cho 10 fold.
