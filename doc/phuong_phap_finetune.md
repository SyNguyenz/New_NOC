# Phương pháp fine-tune NoC với ngân sách 371 nhãn (`head_only`), để ghi vào Methods

Đúng với code đã chạy đợt 10 fold (CODE `529535b846`): `code/noc_lora.py` (`head_kfold`, `budget_rows`, `InvariantCountHead`),
`code/train_set_transformer.py` (arm `head_only`, cờ `--lora_budget`).

## 1. Điểm xuất phát

- **Backbone DS-ST** của fold, đã train ở Phase 1 trên dữ liệu sinh và hồ sơ đơn nguồn thật, rồi **đóng băng hoàn toàn**.
  Fine-tune không cập nhật trọng số nào của backbone. Vì vậy phần định danh (ID) giống từng bit với lượt zero-shot; log kiểm
  bằng `[ID GUARD] params_unchanged=True` và `fp(logits_cls)` không đổi.
- Fine-tune chỉ train **một head đếm nhỏ mới** (1.093 tham số), đặt trên đầu ra của backbone.

## 2. Đầu vào của head (28 đặc trưng, không chứa danh tính donor)

Chạy backbone **một lần** trên mỗi hồ sơ để lấy hai vector, mỗi vector 45 chiều (45 donor trong panel):
- `gate`: xác suất "slot có người" của bộ giải mã AdaSlot;
- `sigmoid(logits_cls)`: xác suất định danh của 45 donor.

Mỗi vector được **sắp xếp giảm dần**, rồi lấy **12 giá trị lớn nhất**, **tổng** của cả vector, và **số phần tử ≥ 0,5**.
Được 14 đặc trưng mỗi vector, 28 đặc trưng tổng cộng.

Sắp xếp làm mất thông tin "donor nào" và chỉ giữ "bao nhiêu slot hoặc danh tính bật, bật mạnh tới đâu". Nhờ vậy head khó học
thuộc tổ hợp donor.

Đặc trưng được chuẩn hoá (trừ trung bình, chia độ lệch chuẩn) bằng thống kê **tính trên chính các hồ sơ dùng để fit**.


## 2b. Danh sách đủ 28 đặc trưng

Ký hiệu: g₁ ≥ g₂ ≥ … ≥ g₄₅ là 45 giá trị `gate` sau khi sắp xếp giảm dần; p₁ ≥ … ≥ p₄₅ là 45 xác suất định danh sau khi sắp xếp.

| # | đặc trưng | ý nghĩa |
|---|---|---|
| 1–12 | g₁, g₂, …, g₁₂ | độ "có mặt" của 12 slot sáng nhất, từ mạnh tới yếu. Slot thứ 5 có bật hay không là bằng chứng trực tiếp cho NoC ≥ 5 |
| 13 | Σ gᵢ (cả 45 slot) | tổng mức có mặt, xấp xỉ số người theo cách đếm của bộ giải mã |
| 14 | #{gᵢ ≥ 0,5} | số slot được coi là "có người" |
| 15–26 | p₁, p₂, …, p₁₂ | xác suất định danh của 12 donor được tin nhất, từ cao tới thấp. Khoảng rơi giữa pₖ và pₖ₊₁ cho biết danh sách người góp dừng ở đâu |
| 27 | Σ pᵢ (cả 45 donor) | tổng xác suất định danh, xấp xỉ số người theo đầu ID |
| 28 | #{pᵢ ≥ 0,5} | số donor được đầu ID coi là có góp mẫu |

## 3. Kiến trúc head

```
28 đặc trưng chuẩn hoá -> Linear(28, 32) -> GELU -> Linear(32, 5) -> logit NoC 1..5      (1.093 tham số)
```

## 4. Huấn luyện

| | giá trị |
|---|---|
| nhãn | **NoC thật** của các hồ sơ thật được rút (mục 5) |
| loss | cross-entropy có trọng số lớp: wₖ ∝ 1/√nₖ (nₖ là số hồ sơ lớp k trong tập fit), chuẩn hoá về trung bình 1, kẹp trong [0,6; 1,6] |
| tối ưu | AdamW, lr 3·10⁻⁴, weight decay 10⁻⁴ |
| batch | 64, xáo ngẫu nhiên mỗi epoch |
| số epoch | **200, cố định** |
| validation, early stopping | **không có** |
| seed | cố định theo fold, lần rút và phần chia |
| thời gian | vài giây mỗi lần fit trên CPU, vì đặc trưng của backbone được tính sẵn một lần |

**Công thức cố định từ trước:** chọn `head_only` trên fold 0 giữa bốn cách (`card_cal`, `card_ft`, `head_only`, `lora_inv`),
với luật chọn ghi trước; sau đó áp nguyên cho mọi fold. Không có siêu tham số nào được chỉnh trên tập test.

## 5. Ngân sách nhãn: giống deepNoC

- Mỗi mô hình fine-tune thấy **tối đa 371 hồ sơ thật có nhãn**, chia NoC1–5 = **34 / 87 / 79 / 93 / 78**. Đây là đúng tập
  fine-tune của deepNoC: tổng hồ sơ của Table 2 trừ đi tập test của họ.
- Hồ sơ được rút ngẫu nhiên **theo từng lớp**, **chỉ từ phía fit** của phép chia. Nếu phía fit không đủ hồ sơ của một lớp thì
  lấy hết số có. Thực tế mỗi lần fit dùng 370–371 nhãn ở 9/10 fold; fold 9 là 359, vì cả fold chỉ có 96 hồ sơ NoC5.
- Lặp **3 lần rút** khác nhau (seed 0, 1, 2). Báo trung bình của 3 lần rút cho mỗi fold, rồi trung bình ± độ lệch chuẩn qua 10 fold.
- Nguồn hồ sơ thật là **tập test trong panel của chính fold đó**. Không có hồ sơ nào vừa được dùng để fit vừa được chấm.

## 6. Hai giao thức chấm

**a) Giao thức strict (tách tổ hợp donor).**
- Tập test thật của fold được chia 5 phần bằng `StratifiedGroupKFold(5, shuffle=True, random_state=42)`.
  Nhóm là **tổ hợp donor** của hồ sơ (hồ sơ đơn nguồn: nhóm là chính donor đó); phân tầng theo NoC.
- Mỗi phần lần lượt làm phần chấm: rút ≤ 371 hồ sơ từ 4 phần còn lại để fit một head mới, rồi chấm phần đó.
- Sau 5 lượt, **mọi hồ sơ test được chấm đúng một lần**, bởi một mô hình **chưa từng thấy tổ hợp donor của nó**.
- Năm lượt fit dùng năm bộ ≤ 371 khác nhau, nhưng mỗi mô hình chỉ thấy ≤ 371.

**b) Giao thức deepNoC (theo đúng mục 2.6 của paper deepNoC).**
- Xếp hồ sơ test theo thứ tự, chia **cách một**: hồ sơ thứ 0, 2, 4, … là nửa fit; hồ sơ thứ 1, 3, 5, … là nửa chấm.
- Rút ≤ 371 hồ sơ (cùng quota theo lớp) từ nửa fit, fit một head, rồi chấm **toàn bộ nửa còn lại** (1.120–1.385 hồ sơ mỗi fold).
- Cùng tổ hợp donor xuất hiện ở cả hai nửa, giống deepNoC.

## 7. Giải mã và chấm điểm

- Xác suất NoC p = softmax(logit). Số người dự đoán: **k = round(Σᵢ i·pᵢ)**, kẹp trong [1, 5]. Đây là giải mã theo kỳ vọng.
- Thước đo: **macro F1** trên 5 lớp NoC (và F1 từng lớp). Cùng thước đo với cột F1 của deepNoC; cột này đúng dù hai cột
  Precision và Recall của họ bị đổi nhãn.
- **Định danh:** lấy k donor có điểm cao nhất trong bảng xếp hạng ID (đã phi-rerank). Bảng xếp hạng không đổi khi fine-tune,
  chỉ k thay đổi.

## 8. Đoạn Methods tiếng Anh (gợi ý)

> **Fine-tuning with a deepNoC-sized label budget.** The DS-ST backbone of each fold was frozen after pre-training, so the
> identification outputs were bit-identical to the zero-shot run. On top of it we trained a small counting head
> (1,093 parameters): the 45 slot-existence gates and the 45 identification probabilities were each sorted in descending
> order and summarised by their 12 largest values, their sum and the number of values ≥ 0.5 (28 identity-blind features,
> standardised on the fitting profiles). The features passed through a two-layer perceptron (28–32–5, GELU). The head was
> trained with class-weighted cross-entropy (weights ∝ 1/√n, normalised and clipped to [0.6, 1.6]), using AdamW
> (learning rate 3×10⁻⁴, weight decay 10⁻⁴), batch size 64, for a fixed 200 epochs, without a validation set or early
> stopping. The recipe was fixed on fold 0 by a pre-registered rule and then applied unchanged to every fold.
> Each fitted head saw at most 371 labelled real profiles (34/87/79/93/78 for one to five contributors), the same budget as
> deepNoC's fine-tuning set. They were drawn per class, at random, only from the fitting side of the split, and the draw was
> repeated three times. We evaluated two splits of each fold's real test profiles:
> (i) a **combination-disjoint** split: five folds grouped by donor combination and stratified by NoC, so every profile is
> scored by a head that never saw its donor combination;
> (ii) **deepNoC's split**: alternate profiles for fitting and testing, so donor combinations are shared.
> The number of contributors was decoded as the rounded expectation of the predicted distribution, and performance is
> reported as macro-F1 over one to five contributors (mean ± SD over ten donor-held-out folds).
