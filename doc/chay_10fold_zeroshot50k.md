# Chạy 10 fold zero-shot 50k: mỗi người 2 fold

Ngày 2026-10-01. Người điều phối: **Mạnh**. Thời gian mỗi người: **khoảng 7 giờ GPU Kaggle, một lần commit**.

## 1. Mục tiêu

Đếm số người góp (NoC) **không dùng một nhãn hỗn hợp thật nào**, đo trên cả 10 fold bằng một công thức duy nhất.
Mỗi fold làm như sau:

- **Phase 1:** train 150 epoch trên 50.000 hỗn hợp sinh (`--conditioned`) và các hồ sơ đơn nguồn thật của fold.
- **NoC:** đọc thẳng `logits_card` của backbone, giải mã bằng argmax. Đây là con số chính.
- **ID:** lấy top-k của bảng xếp hạng ID, với k là số người vừa đếm được.
- **Dòng phụ:** fine-tune với **đúng ngân sách nhãn của deepNoC**, tức 371 hồ sơ (NOC1–5 lần lượt 34/87/79/93/78).
  Chạy 3 lần rút mẫu trên chính backbone vừa train, khoảng 12 phút mỗi fold. Mỗi lần rút chạy **bốn cách fine-tune**
  trên đúng cùng các hàng:

  | cách | train cái gì | tham số |
  |---|---|---|
  | `card_cal` | nhiệt độ + 5 bias trên `logits_card` (hiệu chỉnh đầu zero-shot) | 6 |
  | `card_ft` | bản sao `Linear(45→5)` của `logits_card`, khởi tạo từ trọng số zero-shot | 230 |
  | `head_only` | head bất biến danh tính trên feature cố định | ~1.100 |
  | `lora_inv` | LoRA r8 trên decoder + head | ~25.700 |

  Cả bốn cách đều không chạm vào đường ID. Cách dùng làm dòng chính được **chọn trên fold 0 trước khi chạy 10 fold**
  (mục 3, bước 0). **Kết quả: `head_only`.** Phạm vi đánh giá là **trong panel** (mục 1c), nên dùng luật chọn ban đầu.

Backbone luôn đóng băng sau Phase 1. Dòng `[ID GUARD] ... params_unchanged=True` trong log là bằng chứng ID không bị chạm vào.

### Vì sao đi hướng này (đã đo trên fold 0)

| cách đếm | nhãn hỗn hợp thật | macro F1 (strict) |
|---|---|---|
| zero-shot, bộ 50k | **0** | **0.9102** |
| zero-shot, 12k rút từ chính bộ 50k | 0 | 0.7488 |
| LoRA trên backbone 12k, ngân sách 371 nhãn | 371 | 0.8646 ± 0.017 |
| LoRA trên backbone 12k, không giới hạn (~2.000 hàng) | ~2.000 | 0.9201 |
| deepNoC (F1 họ công bố, dựng lại từ Table 2) | 371 | 0.9097 |

Rút ra từ bảng: muốn zero-shot tốt thì **cần 50k hỗn hợp sinh**. Còn LoRA chỉ vượt deepNoC khi dùng nhiều nhãn thật hơn họ khoảng 5 lần.
Chi tiết ở `doc/deepNoC.md` và `doc/failed_experiments.md`.

## 1b. Kết quả chọn cách fine-tune trên fold 0 (01/10)

Backbone 12k rút từ bộ 50k (`fp 753d6437b9`), 371 nhãn, 3 lần rút, cùng các hàng cho cả bốn cách, giải mã expect.

| cách | tham số | strict | giao thức deepNoC | lợi thế giao thức (deepNoC − strict) | macro F1 hỗn hợp ngoài panel |
|---|---|---|---|---|---|
| **`head_only`** | 1.093 | **0.9026 ± 0.006** | 0.9021 ± 0.005 | **≈ 0** | 0.227 |
| `lora_inv` | 25.685 | 0.8646 ± 0.017 | 0.9036 ± 0.012 | +0.039 | 0.315 |
| `card_cal` | 6 | 0.7961 ± 0.009 | 0.8428 ± 0.001 | +0.047 | 0.491 |
| `card_ft` | 230 | 0.7405 ± 0.014 | 0.8675 ± 0.001 | **+0.127** | 0.483 |
| zero-shot | 0 | 0.7488 (argmax 0.8128) | — | — | 0.514 |
| deepNoC | — | — | 0.9097 | — | — |

- Theo luật chọn đã chốt trước, cách được chọn là **`head_only`**. Giả thuyết F2 đúng: với 371 nhãn thì **không cần LoRA**.
- `head_only` gần như **không có lợi thế giao thức**: dùng lại tổ hợp donor không giúp nó, tức nó không học thuộc tổ hợp.
  Ngược lại, `card_ft` đọc gate theo thứ tự donor nên học thuộc nặng nhất (+0.127).
- F1 sai: hiệu chỉnh 6 tham số chưa đủ để sửa lỗi gọi thiếu NOC5.
- Đánh đổi cần nhớ: `head_only` đọc xác suất ID nên yếu ở hồ sơ có người ngoài panel (0.227). Zero-shot và `card_cal` giữ được khoảng 0.5.

### Lần chọn thứ hai (luật sửa ngày 01/10, trước khi chạy các đầu mới)

Yêu cầu: **NoC phải đếm được cả khi có người ngoài panel**, nên đầu đếm không được dựa vào điểm ID.
Luật mới:
1. Chỉ xét những đầu có macro F1 trên hỗn hợp ngoài panel ≥ 0.464 (tức zero-shot 0.514 trừ 0.05).
2. Trong số đó, chọn đầu có strict cao nhất. Hai đầu chênh nhau dưới 0.005 thì chọn đầu ít tham số hơn.
3. Không đầu nào đạt thì định tuyến: dùng `head_only` cho hồ sơ trong panel, zero-shot cho hồ sơ ngoài panel.

Cell `cell_ft371_compare_fold0.py` (CODE `529535b846`) chạy năm đầu sau:

| đầu | đọc điểm ID | train cái gì | tham số |
|---|---|---|---|
| `card_cal` | không | hiệu chỉnh `logits_card` | 6 |
| `head_free` | không | gate đã sắp xếp | 645 |
| `head_mac` | không | gate đã sắp xếp + 19 đặc trưng MAC | 1.253 |
| `head_unf` | không | encoder trên **mọi** peak (bỏ qua bộ lọc panel) + MAC | 4.901 |
| `head_only` | **có** | để đối chiếu | 1.093 |

**Lần chọn thứ hai bị huỷ** do phạm vi đã chốt là trong panel (mục 1c). Các đầu `head_free` / `head_mac` / `head_unf` vẫn chạy
trong cell 10 fold vì gần như không tốn thời gian, nhưng chỉ để tham khảo.

## 1c. Phạm vi: chỉ trong panel (Mạnh chốt 01/10)

- NoC được đánh giá và so với deepNoC **chỉ trên tập trong panel**: hồ sơ mà mọi người góp đều thuộc 45 donor của fold.
  Lý do: 372 hồ sơ test của deepNoC đều chứa donor đã xuất hiện lúc fine-tune, nên tập trong panel là phép so tương đương.
- Số ngoài panel (nhánh `open`) vẫn có trong báo cáo nhưng **không dùng** cho quyết định hay so sánh nào.
- Khi viết bài: nêu rõ bối cảnh là **có panel tham chiếu**, và thêm một câu trong phần giới hạn.

## 2. Phân công

| người | fold | dataset cần gắn |
|---|---|---|
| **Mạnh** | **0**, 1 | `noc-cond50k-fold0`, `noc-cond50k-fold1` |
| **Nguyên** | 2, 3 | `noc-cond50k-fold2`, `noc-cond50k-fold3` |
| **Dũng** | 4, 5 | `noc-cond50k-fold4`, `noc-cond50k-fold5` |
| **Quang** | 6, 7 | `noc-cond50k-fold6`, `noc-cond50k-fold7` |
| **Giang** | 8, 9 | `noc-cond50k-fold8`, `noc-cond50k-fold9` |

Ai cũng phải gắn thêm dataset code **`noc-fold-code`**, đúng bản có CODE **`529535b846`**.

Fold 0 còn dùng để **kiểm tái lập dữ liệu**. Bộ 50k của nhóm (`noc-insilico-w-fold0`) đã cho 0.9102. Bộ `noc-cond50k-fold0`
được sinh lại bằng cùng script nhưng không trùng từng byte với bộ đó. Nếu fold 0 của bộ mới lệch quá 0.02 thì Mạnh sẽ báo cả nhóm
**trước khi** gộp kết quả.

## 3. Mạnh làm trước khi phát cho nhóm

0. **Chọn cách fine-tune trên fold 0**, khoảng 15 phút. Chạy `kaggle_upload/cells/cell_ft371_compare_fold0.py`, gắn
   `noc-sub12k-of50k-fold0` và backbone `backbone_sub12k_fold0` (qua Add Data → Notebook Output). Cell in ra cách được chọn
   theo luật đã chốt trước. Bước này **không chặn** nhóm: cell 10 fold vẫn chạy cả bốn cách. Fold 0 chỉ quyết định cách nào là dòng chính.
1. Sinh dữ liệu trên máy (đã chạy):
   `python3 kaggle_upload/tools/build_cond50k_folds.py`.
   Lệnh này ra 10 file `kaggle_upload/team_10fold/noc-cond50k-fold<F>.zip` (khoảng 118 MB mỗi file) và file `kaggle_upload/team_10fold/cond50k_manifest.json` chứa md5 của từng fold.
   Mọi fold sinh trên **một máy, một lệnh**. Generator cho kết quả tất định trên cùng máy.
2. Tạo 10 dataset Kaggle, tên `noc-cond50k-fold0` … `noc-cond50k-fold9`. Mỗi dataset là một file zip, bên trong có thư mục `data_cond50k_fold<F>/`.
3. Upload `kaggle_upload/noc-fold-code.zip` thành **New Version** của `noc-fold-code`.
4. Chia sẻ các dataset cho nhóm: mở dataset, chọn **Settings → Sharing**, thêm tài khoản Kaggle của từng người (hoặc để Public).
5. Dán md5 từ `cond50k_manifest.json` vào biến `MANIFEST` trong `kaggle_upload/team_10fold/cell_zeroshot50k_10fold.py`, rồi gửi cell cho nhóm.
   Có `MANIFEST` thì cell tự dừng nếu ai đó gắn nhầm dataset.

## 4. Từng người làm

1. Kiểm tài khoản Kaggle đã xác minh số điện thoại (có xác minh mới dùng được GPU) và còn **ít nhất 8 giờ** hạn mức GPU trong tuần.
   Xem ở **Settings → Accelerator quota**.
2. Tạo notebook mới. Chọn **Accelerator: GPU T4 x2** hoặc **P100**, và **Internet: Off** (không cần mạng).
3. **Add Data:** gắn `noc-fold-code` cùng 2 dataset `noc-cond50k-fold<F>` của mình. **Không** gắn các fold khác.
4. Dán toàn bộ nội dung `cell_zeroshot50k_10fold.py` vào **một** cell, rồi chỉ sửa hai dòng đầu:
   ```python
   NGUOI = 'dung'      # tên mình, không dấu
   FOLDS = [4, 5]      # 2 fold của mình, theo bảng ở mục 2
   ```
5. Bấm **Save Version**, chọn **Save & Run All (Commit)**.
   **Không chạy interactive.** Đóng tab hoặc mất mạng thì phiên chết, mất cả backbone. Nhóm đã mất backbone ba lần vì đúng lỗi này.
6. Khoảng 7 giờ sau, mở version vừa chạy, vào tab **Output**, tải file `send_<tên>.zip` và gửi cho Mạnh.
   Trong file có, với mỗi fold: `fold<F>_seed42.json` (file để gộp), các file metrics, log và backbone.

## 5. Kiểm log khi đang chạy

Mở version đang chạy, vào tab **Logs**. Mỗi fold phải thấy:

| dòng | phải là |
|---|---|
| `CODE ... OK` | `529535b846` |
| `fold F: ... md5 ...` | khớp `MANIFEST` (cell tự kiểm) |
| `[FOLD] data dir: unknown_donors=...` | đúng donor ẩn của fold (bảng ở mục 7) |
| `params: 1,505,737` | đúng con số này |
| `Ep  10 \| ... (Ns)` | khoảng 70–80 s mỗi epoch (bộ 50k), cả Phase 1 khoảng 3–3,3 giờ |
| `>>> backbone da luu -> /kaggle/working/send/fold<F>/backbone` | xuất hiện ngay sau `Training done in` |
| `[SPLIT AUDIT] ... SHARED=0  (clean)` | bắt buộc là 0 |
| `[ID GUARD] ... params_unchanged=True` | bắt buộc là True |
| các lượt 371 nhãn | `fit on 371 of ...` (hoặc ít hơn nếu fold thiếu một lớp), cùng `fp(logits_cls)` với lượt zero-shot |

Cuối log có khối `===== KET QUA <tên>: fold [...]`. Chụp hoặc dán khối này vào nhóm chat.

## 6. Sự cố hay gặp

| hiện tượng | cách xử lý |
|---|---|
| `dataset code la CODE xxx, can 529535b846` | Vào **Input**, bấm ⋮ cạnh `noc-fold-code`, chọn **Check for updates** (hoặc gỡ ra rồi gắn lại) |
| `can dung 1 thu muc data_cond50k_fold<F>` | Chưa gắn dataset của fold đó, hoặc gắn cùng một dataset hai lần |
| `md5 ... khac ban da chot` | Gắn nhầm bản dataset. Báo Mạnh, **không chạy tiếp** |
| `!!! FOLD F LOI` | Cell vẫn chạy fold còn lại. Gửi `send/fold<F>/*.log` cho Mạnh |
| Phiên chết sau khi Phase 1 xong | Backbone đã ở `send/fold<F>/backbone`. Mạnh sẽ gửi cell ngắn chạy lại phần đếm với `--init_from`, khoảng 20 phút |
| Hết hạn mức GPU giữa chừng | Chạy từng fold một: `FOLDS = [4]`, tuần sau `FOLDS = [5]` |

## 7. Bảng 10 fold

Donor ẩn của mỗi fold không có trong panel 45 người, chỉ xuất hiện ở nhánh `open`.

| fold | donor ẩn | tổ hợp thật N2/N3/N4/N5 |
|:---:|---|---|
| 0 | 6, 21, 26, 40, 50 | 5 / 6 / 4 / 5 |
| 1 | 18, 28, 46, 47, 48 | 7 / 4 / 6 / 5 |
| 2 | 19, 24, 32, 38, 41 | 6 / 4 / 4 / 3 |
| 3 | 5, 8, 25, 30, 39 | 7 / 6 / 5 / 3 |
| 4 | 10, 22, 27, 29, 35 | 7 / 6 / 4 / 3 |
| 5 | 16, 17, 33, 42, 43 | 6 / 6 / 5 / 3 |
| 6 | 4, 11, 31, 44, 45 | 6 / 5 / 6 / 4 |
| 7 | 7, 12, 20, 23, 36 | 8 / 6 / 6 / 4 |
| 8 | 1, 3, 13, 15, 49 | 7 / 5 / 7 / 6 |
| 9 | 2, 9, 14, 34, 37 | 7 / 6 / 4 / 2 |

Fold 2–5 và fold 9 chỉ có 2–3 tổ hợp NOC5, nên F1 NOC5 của từng fold sẽ dao động mạnh. Khi báo cáo phải dùng **số gộp 10 fold** và ghi kèm `n`.

## 8. Gộp kết quả (Mạnh)

```bash
python3 code/report_folds.py send_all/fold* --out doc/ket_qua_10fold_zeroshot50k.md --csv doc/ket_qua_10fold_zeroshot50k.csv
```

Trong đó `send_all/` chứa các `send_<tên>.zip` đã giải nén. Lệnh trên gộp zero-shot. Fine-tune 371 nhãn gộp bằng
`python3 code/report_budget.py send_all/ [--exclude_selection]` (bỏ fold 0 cho T5).
Script gộp sẽ kiểm mọi fold dùng chung code, chung cách đếm và chung công thức. Nó in một dòng cho mỗi fold, rồi in mean ± sd và số gộp (cộng dồn ma trận nhầm lẫn).

## 9. Giả thuyết đặt trước

Ghi ngày 01/10, trước khi chạy. **Không sửa sau khi xem số.**

| | nội dung |
|---|---|
| **T1** | zero-shot macro F1, trung bình 10 fold, ≥ 0.9097 (deepNoC) |
| **T2** | zero-shot macro F1 gộp (cộng ma trận nhầm lẫn 10 fold) ≥ 0.9097 |
| **T3** | fold 0: bộ `noc-cond50k-fold0` và bộ của nhóm (`noc-insilico-w-fold0`) lệch nhau ≤ 0.02 macro F1 zero-shot |
| **T4** | mọi fold `params_unchanged=True`, `SHARED=0`; mọi lượt 371 nhãn có cùng `fp(logits_cls)` với lượt zero-shot của fold đó |
| **T5** | `head_only` (đã chọn trên fold 0), dùng 371 nhãn, trung bình 9 fold còn lại × 3 lần rút, strict ≥ 0.9097 |
| **T6** | Headline là **zero-shot, số gộp 10 fold** bất kể kết quả. Fine-tune 371 nhãn luôn là dòng phụ. Bảng từng fold kèm số tổ hợp NOC4/NOC5 |

Luật đọc kết quả, chốt trước:

- **T4 sai:** có lỗi lập trình. Không dùng số nào cho tới khi tìm ra lỗi.
- **T3 sai:** bộ dữ liệu tự sinh không tương đương bộ đã cho 0.9102. Báo cả hai, không trộn, và tìm nguyên nhân trước khi viết Methods.
- **T1 và T2 đúng:** headline là *"ngang hoặc vượt deepNoC, không dùng nhãn hỗn hợp thật, 10 fold, giao thức tách tổ hợp"*.
- **T1 hoặc T2 sai:** báo đúng con số và khoảng cách. **Không** viết "vượt".
- **T5** đọc độc lập. Nó chỉ trả lời câu hỏi: với cùng 371 nhãn như deepNoC, fine-tune có vượt họ hay không.
  Fold 0 là fold dùng để **chọn** cách fine-tune, nên T5 tính trên 9 fold còn lại. Bảng 10 fold vẫn in fold 0, kèm ghi chú.
- Các fold sẽ chênh nhau, chủ yếu vì có fold chỉ có 2–3 tổ hợp NOC5. Chênh lệch đó được **báo cáo**, không dùng làm lý do đổi headline.

Mọi so sánh với deepNoC phải kèm phần công bố bắt buộc ở `doc/notes/2026-10-01_chien_luoc_nop_bai.md`, mục 8.
Riêng hướng zero-shot phải nói thêm: Phase 1 có dùng hồ sơ **đơn nguồn thật** (nhãn NOC = 1). Đây là nhãn rẻ, nhưng vẫn là nhãn.
