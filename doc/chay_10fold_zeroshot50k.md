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
| **T7** | *(ghi 02/10, sau khi xem fold 0–3, TRƯỚC khi có fold 4–9)* Zero-shot cộng **hiệu chỉnh prior không dùng nhãn**: chia xác suất `softmax(logits_card)` cho tỉ lệ NOC1..5 trong tập train Phase 1 của fold (sau khi cắt DEV), rồi lấy argmax. Đánh giá **chỉ trên fold 4–9**, số gộp: recall NOC5 tăng **≥ 0.10** so với zero-shot thường, **và** macro F1 không giảm. Fold 0–3 chỉ để xem trước, không tính. Bị bác thì giữ zero-shot thường làm headline, không thử biến thể khác trên cùng các fold này |

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

## 10. Thử nghiệm S2: mạng đếm riêng `noc_deep` (ghi 02/10, TRƯỚC khi train)

Mạng đếm **tách hẳn** khỏi backbone ID, theo kiểu deepNoC:
- Đọc **mọi peak**: không qua bộ lọc panel, không dùng genotype, không đọc điểm ID hay gate.
- Đặc trưng mỗi peak chỉ là vật lý: allele, chiều cao, thứ hạng trong locus, quan hệ stutter.
- Gộp peak → locus → hồ sơ, cộng 19 đặc trưng MAC.
- Dạy bằng **NOC thật** trên dữ liệu sinh của fold (nhãn miễn phí).
- Chọn epoch trên DEV in-silico, không nhìn nhãn thật.

ID giữ nguyên từng bit: backbone không bị chạm vào.

Chạy local trên MPS/CPU, trên fold 0 (bộ `noc-cond50k-fold0`). Mốc so sánh là backbone 50k của chính fold đó:

| | strict | giao thức deepNoC |
|---|---|---|
| `logits_card` zero-shot | 0.8025 | — |
| `card_cal` (371 nhãn) | 0.8528 | 0.8837 |
| `head_only` (371 nhãn) | 0.8691 | 0.9277 |

| | giả thuyết |
|---|---|
| **N1** | `noc_deep` zero-shot (argmax) ≥ 0.8025 |
| **N2** | `noc_deep` + hiệu chỉnh 371 nhãn (`card_cal` trên đầu ra của `noc_deep`), strict, TB 3 lần rút ≥ 0.8528 |
| **N3** | recall NOC5 của `noc_deep` zero-shot ≥ 0.368 + 0.10 (`logits_card` fold 0 chỉ đạt 0.368) |

Luật đọc kết quả:
- N1 hoặc N2 đúng: chạy thêm fold 6 (CPU) để xác nhận, rồi mới tính đưa `noc_deep` vào pipeline.
- Cả hai sai: ghi lại vào `failed_experiments.md`, không chỉnh siêu tham số trên fold 0 để cố cho đạt.

## 11. Sửa tỉ lệ người góp trong dữ liệu sinh (D2), kiểm bằng `noc_deep` (ghi 02/10, TRƯỚC khi sinh và train)

**Nguyên tắc:** không chỉnh dữ liệu sinh theo tập test.
- Không dùng chế độ "realistic" (Dirichlet α = 4, 15% chia đều): α đó được chọn để khớp NOC5 của test.
- Không dùng chế độ "wide" của nhóm, vì nửa cân bằng của nó chính là "realistic".

**Thay đổi duy nhất (`STR_COND_PHI=mixed`):** trong `gen_conditioned`, với xác suất 0.5 thì chia lại tổng lượng DNA của
hỗn hợp theo **Dirichlet(1, …, 1)**. Đây là phân bố đều trên mọi cách chia, tiên nghiệm không thông tin, không lấy từ
dữ liệu nào. Nửa còn lại giữ nguyên cách rút hiện tại (mỗi người log-uniform 0,008–0,5 ng). Mọi định luật đã hiệu chỉnh
khác giữ nguyên.

**Thiết kế A/B:**
- **Fold 6.** Chưa dùng để chấm `noc_deep`.
- Hai bộ 20k cùng code, cùng seed 42: `base` (như hiện tại) và `mixed`.
- `noc_deep` train như nhau trên mỗi bộ (20 epoch, chọn epoch theo DEV in-silico), rồi chấm zero-shot argmax trên test thật fold 6.

| | giả thuyết |
|---|---|
| **W1** | macro F1 `noc_deep`(mixed) ≥ `noc_deep`(base) + 0.05 |
| **W2** | recall NOC4 của `noc_deep`(mixed) ≥ của `noc_deep`(base) + 0.15 |

Luật đọc kết quả:
- W1 đúng: dữ liệu `mixed` đáng thử cho backbone. Bước kế là chạy 12k trên Kaggle, A/B cùng seed, 1 fold.
- W1 sai: ghi lại; D2 không phải nút thắt của `noc_deep`.

## 12. `noc_deep` ở 20k và ghép với `card_cal` (ghi 02/10, TRƯỚC khi chạy)

**H1.** Cấu hình E3 (`--loss ordinal --aux 0.5 --feats rich`), train trên 20k hồ sơ sinh, ghép 3 seed (42/43/44).
Zero-shot ≥ bản 6k ghép 3 seed + 0.03, ở **cả** fold 0 (0.664) lẫn fold 6 (0.664).

**H3.** Ghép có học, fit trên 371 nhãn, cùng các lần rút và các phần chia như mọi lần fit trước:
`z = W·[log p(logits_card) ; log p(noc_deep)] + b`, 55 tham số.
Strict ≥ `card_cal` (fit lại trong cùng script) + 0.01, ở **cả** fold 0 lẫn fold 6.
`noc_deep` dùng là bản tốt nhất có được trước khi ghép (bản 20k nếu H1 đúng, không thì bản 6k).

## 13. M-h: slot đếm chung trong `noc_deep`, "đếm bằng tổng" không dính ID (ghi 02/10, TRƯỚC khi chạy)

- Thêm 6 slot **chung**: khởi tạo học được, không genotype, không gắn donor.
- Slot attention 3 vòng trên peak, kèm ngữ cảnh locus. Các slot tranh nhau peak (softmax theo slot).
- Mỗi slot có cổng "tồn tại". Dạy SmoothL1(tổng cổng, NOC thật), trọng số 0.5. Cổng đã sắp xếp và tổng cổng được đưa vào đầu đếm thứ tự.
- Mọi thứ khác giống E3.

**Cùng cỡ dữ liệu:**
- fold 0: đúng 15.095 hồ sơ train của bộ 12k (như mục 28).
- fold 6: rút 15.095 hồ sơ từ train fold 6, cùng các hồ sơ cho cả hai bản.
- Ghép 3 seed (42/43/44) cho cả E3 lẫn M-h.

| | giả thuyết |
|---|---|
| **S1** | M-h zero-shot ≥ E3 + 0.03, ở **cả** fold 0 lẫn fold 6 |
| **S2** | tổng cổng có tương quan với NOC thật trên test thật ≥ 0.90 |

## 14. Đo chắc bằng 5 seed, và M-h2: dạy thẳng cổng thứ k (ghi 02/10, TRƯỚC khi chạy)

**M-h2** = M-h, nhưng thay SmoothL1(tổng cổng) bằng BCE trên **các cổng đã sắp xếp giảm dần**, với đích
`[1]*NOC + [0]*(6 - NOC)`. Tức là cổng thứ k bật khi và chỉ khi NOC ≥ k. Trọng số 0.5.

Cùng cỡ dữ liệu như mục 13 (15.095 hồ sơ, fold 0 và fold 6). **5 seed (42–46)** cho E3, M-h, M-h2. Báo
trung bình ± sd của zero-shot argmax qua các seed, và bản ghép 5 seed.

| | giả thuyết |
|---|---|
| **R1** | M-h2 (TB 5 seed) ≥ M-h + 0.02, ở **cả hai** fold |
| **R2** | khoảng cách trung vị tổng cổng giữa NOC5 và NOC4 của M-h2 ≥ 2 lần của M-h (M-h: 0.23 / 0.20) |
| **R3** | chỉ khi chênh lệch TB > 2 × sd giữa các seed mới được coi là cải tiến thật |

## 15. KẾT QUẢ 10 FOLD, đọc theo giả thuyết T1–T7 (02/10)

Toàn bộ 10 fold, CODE `529535b846`, mọi kiểm tra đạt.
Bảng chi tiết: `doc/ket_qua_10fold_zeroshot50k.md`, `doc/ket_qua_10fold_finetune371.md`, `doc/ket_qua_9fold_finetune371.md`.

| | đo được | mốc | kết quả |
|---|---|---|---|
| **T1** zero-shot, TB 10 fold | 0.885 ± 0.040 | ≥ 0.9097 | **SAI** |
| **T2** zero-shot, gộp | **0.888**, 95% CI [0.869, 0.905] (bootstrap theo tổ hợp donor) | ≥ 0.9097 | **SAI** |
| **T3** fold 0: bộ tự sinh so với bộ của nhóm | 0.8025 so với 0.9102 | lệch ≤ 0.02 | **SAI** (hai bộ giống nhau về thống kê; nghi do dao động khi train) |
| **T4** kiểm tra sạch | 10/10 fold | | **ĐÚNG** |
| **T5** `head_only` 371 nhãn, strict, fold 1–9 | 0.902 ± 0.014 (gộp 0.902) | ≥ 0.9097 | **SAI** (thiếu 0.008) |
| T5, giao thức deepNoC | 0.920 ± 0.012 | | cao hơn mốc |
| **T6** headline = zero-shot gộp | 0.888 | | theo luật đã chốt |
| **T7** hiệu chỉnh prior, gộp fold 4–9 | macro F1 0.8955 → 0.9043; recall NOC5 +0.075 | NOC5 ≥ +0.10 | **SAI** |

**deepNoC:** 0.9097, 95% CI [0.882, 0.934]. Đây là bootstrap theo từng hồ sơ trên 372 hồ sơ của Table 2; họ không công bố
tổ hợp, nên khoảng thật còn rộng hơn. Cả zero-shot (0.888) lẫn `head_only` (0.902) đều **nằm trong khoảng tin cậy của deepNoC**.

**Câu viết đúng theo luật:** *"không dùng nhãn hỗn hợp thật, macro F1 0.888 [0.869–0.905] qua 10 fold tách tổ hợp donor,
so với 0.910 [0.882–0.934] của deepNoC có fine-tune 371 hồ sơ; với cùng 371 nhãn: 0.902 (strict) và 0.920 (giao thức của
deepNoC)."* **Không** viết "vượt deepNoC" ở giao thức strict.

**Kết quả phụ (không chốt trước, báo như phân tích):**
- `card_cal` (6 tham số): fold 1–9 strict 0.906 (gộp 0.909), giao thức deepNoC 0.910.
- Lợi thế khi dùng lại tổ hợp, tính TB 10 fold (giao thức deepNoC trừ strict): `head_unf` +0.48, `card_ft` +0.08, `lora_inv` +0.03, `head_only` +0.02, `card_cal` +0.01.

**Dòng "chỉ 25 s"** (phân tích độ nhạy, khớp điều kiện deepNoC; **không** chốt trước). Script `work/eval_25s.py`, chi tiết ở
`doc/ket_qua_10fold_25s.txt`. Fine-tune được fit lại offline, với 371 nhãn rút từ chính hồ sơ 25 s.

| | TB 10 fold | TB fold 1–9 |
|---|---|---|
| zero-shot, toàn bộ hồ sơ | 0.885 | 0.894 |
| **zero-shot, chỉ 25 s** | **0.898** | **0.908** |
| `head_only` 25 s: strict / giao thức deepNoC | 0.901 / 0.919 | 0.906 / 0.917 |
| `card_cal` 25 s: strict / giao thức deepNoC | 0.899 / 0.920 | 0.909 / 0.923 |
| *deepNoC (25 s, 371 nhãn, chia xen kẽ)* | *0.910* | |

## 16. Chia dữ liệu cụ thể của mỗi fold (đợt 10 fold, 02/10)

**A. Pretrain (Phase 1, train backbone):** 150 epoch, train lại từ đầu ở mỗi fold.

| tập | nội dung | số hồ sơ mỗi fold | dùng để |
|---|---|---|---|
| **train** | 50.000 hỗn hợp **sinh** (`--conditioned`, chỉ từ donor trong panel, không trùng tổ hợp test) cộng hồ sơ **đơn nguồn thật** của 45 donor trong panel, trừ phần đã tách sang DEV | **47.267–47.494** (NOC1 4.743–4.932; NOC2 khoảng 6.000; NOC3 khoảng 9.100; NOC4 khoảng 15.250; NOC5 khoảng 12.100) | học trọng số |
| **DEV in-silico** (validation) | tách **theo tổ hợp donor** từ train: khoảng 15% tổ hợp mỗi NOC, cộng khoảng 6% hồ sơ đơn nguồn | **7.733–7.843** (NOC1 khoảng 310) | **chọn epoch tốt nhất** (macro recall ID). Không nhãn thật nào |
| val thật | hồ sơ **đơn nguồn thật** (NOC1) của donor trong panel | 1.081–1.124 | chỉ để in log (F1 ID), không chọn gì |
| open thật | hồ sơ thật có người **ngoài panel** | 1.097–1.705 | dạy **reject head** (nhãn "ngoài panel", không có nhãn NoC) |

**B. Zero-shot:** không fit gì thêm, chấm thẳng trên toàn bộ **test**.

**C. Fine-tune 371 nhãn:** backbone đóng băng, head nhỏ (`head_only` 1.093 tham số). Dữ liệu thật **chỉ lấy từ tập test của fold**.

| giao thức | train (fine-tune) | validation | test (chấm) |
|---|---|---|---|
| **strict** | chia test thành 5 phần **tách tổ hợp**; mỗi lần fit rút **≤ 371** hồ sơ (34/87/79/93/78) từ 4 phần còn lại | **không có**: công thức cố định, chọn trên fold 0 | phần thứ 5; ghép 5 phần lại thì mọi hồ sơ test đều được chấm đúng 1 lần |
| **giao thức deepNoC** | rút **≤ 371** hồ sơ từ nửa chẵn (cách một) | không có | toàn bộ nửa lẻ (1.120–1.385 hồ sơ); tổ hợp **dùng chung** với nửa chẵn |

Lặp 3 lần rút khác nhau, báo trung bình. Mỗi mô hình chỉ thấy ≤ 371 nhãn. Số nhãn thực tế: 370–371 ở mọi fold, trừ fold 9
(359, vì chỉ có 96 hồ sơ NOC5). Với dòng "chỉ 25 s" là 308–360.

**D. Test thật (trong panel):** mọi hồ sơ thật mà mọi người góp đều thuộc 45 donor của fold. Gồm toàn bộ hỗn hợp của các tổ hợp
thuộc fold, cộng hồ sơ đơn nguồn test.

| fold | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 2458 | 2577 | 2240 | 2535 | 2458 | 2440 | 2464 | 2761 | 2770 | 2452 |
| NOC1 / 2 / 3 / 4 / 5 | 1125/334/385/242/372 | 1109/479/293/324/372 | 1103/383/290/296/168 | 1112/481/385/389/168 | 1113/478/385/284/198 | 1082/382/437/341/198 | 1095/383/338/378/270 | 1119/526/436/434/246 | 1109/479/337/425/420 | 1095/481/436/344/96 |

**Không rò rỉ:** không tổ hợp test nào nằm trong 50k hỗn hợp sinh (`SHARED=0` ở cả 10 fold). Không hồ sơ test nào nằm trong
train hay DEV của Phase 1. 5 donor của mỗi fold bị giấu hoàn toàn khỏi train, val và test; chỉ xuất hiện ở nhánh open.

## 17. Dạy lại đầu đếm của backbone bằng NOC THẬT trên dữ liệu sinh (sửa M1), backbone đóng băng (ghi 02/10, TRƯỚC khi chạy)

Dùng backbone 50k đợt 10 fold (fold 0 và 6). Trích đặc trưng cố định (gate, sigmoid(logits_cls), `logits_card`, 19 MAC) trên
**6.000 hồ sơ sinh** rút ngẫu nhiên (seed 42) từ train Phase 1 của fold. Dạy các head nhỏ bằng **NOC thật** của các hồ sơ đó.
Không dùng nhãn hỗn hợp thật, ID không đổi.

| head | đọc gì | tham số |
|---|---|---|
| `sim_card` | `Linear(45→5)` trên gate (cùng cấu trúc `logits_card`) | 230 |
| `sim_inv` | gate + xác suất ID đã sắp xếp (đặc trưng của `head_only`) | 1.093 |
| `sim_invmac` | `sim_inv` + 19 MAC | 1.701 |

| | giả thuyết |
|---|---|
| **P1** | head tốt nhất (chọn trên fold 0) có zero-shot ≥ `logits_card` + 0.02, ở **cả** fold 0 lẫn fold 6 |
| **P2** | `head_only` khởi tạo từ `sim_inv` rồi fine-tune 371 nhãn, strict ≥ `head_only` train từ đầu + 0.01, ở cả hai fold |

*Bổ sung trước khi chạy (yêu cầu của Mạnh):*
- **`sim_sum`**: head chỉ đọc [Σgate, Σgate²], dạy bằng NOC thật trên dữ liệu sinh.
- **Cấu hình fine-tune như deepNoC** cho phần NoC: Adam lr 1e-5, betas (0.5, 0.999), batch 100, 2000 epoch, cross-entropy.
  So với cấu hình hiện tại: AdamW lr 3e-4, wd 1e-4, batch 64, 200 epoch.
- **P3** (chỉ báo cáo, không có ngưỡng): `head_only` strict theo cấu hình deepNoC so với cấu hình hiện tại, train từ đầu và khởi tạo từ `sim_inv`.

## 18. `noc_deep` với 89 đặc trưng/peak như deepNoC, không dính ID (ghi 03/10, TRƯỚC khi chạy)

Yêu cầu của Mạnh: nhánh NoC riêng, không phụ thuộc ID, vẫn đếm được; mẫu nhỏ, local; tham khảo 89 đặc trưng của deepNoC.
Không đổi dữ liệu sinh. Không nạp backbone, nên ID không thể đổi.

**Thay đổi duy nhất là đầu vào** (`--feats deep89`, code `code/noc_deep.py`); kiến trúc, loss, aux, lịch train giữ nguyên như E3
(`--loss ordinal --aux 0.5`, d = 64, 2 tầng, 25 epoch, chọn epoch trên DEV in-silico).

| cột | đặc trưng | so với paper |
|---|---|---|
| 0–23 | one-hot locus | như paper |
| 24, 25, 26 | allele/100, kích thước bp/100, log1p(chiều cao)/10 | paper chia chiều cao cho 33.000 |
| 27 | tần số allele | **lấy từ 45 genotype trong panel** (không có bảng quần thể offline); bản `deep89nf` đặt về 0 |
| 28 | plp = P(peak là allele) | paper dùng MHCNN; ở đây là HistGB fit trên peak của dữ liệu sinh (nhãn `attr`) |
| 29–52 | peak này là stutter (lùi 1, lùi 2, tiến 1, −2 bp): allele, chiều cao, tỉ lệ, tỉ lệ kỳ vọng, tần số, plp của peak cha | như paper; tỉ lệ kỳ vọng là hồi quy tuyến tính theo allele cha, mỗi locus và mỗi loại, fit trên dữ liệu sinh |
| 53–76 | peak này là cha: cùng 6 thông tin về peak stutter con | như paper |
| 77, 78 | số peak trong locus, trong hồ sơ | như paper |
| 79–88 | tỉ lệ pha (proxy): trung bình qua các locus thường của chiều cao allele thứ r / tổng, r = 1..10 | paper dùng "smart start" |
| hồ sơ | 19 MAC cũ + 8 MAC chỉ đếm peak có plp > 0,5 | mới |

Bảng plp, tỉ lệ kỳ vọng và tần số fit **chỉ trên train sinh**, cố định 15.095 hồ sơ (rng 0). Cùng cỡ dữ liệu như E3 ở mục 14:
15.095 hồ sơ train, DEV 2.152, seed 42–46, fold 0 (bộ sub12k) và fold 6.

| | giả thuyết (mốc E3, mục 30 failed_experiments: fold 0 0.669 ± 0.012, fold 6 0.676 ± 0.010) |
|---|---|
| **F1** | `deep89` zero-shot, TB 5 seed ≥ E3 + 0.03 ở **cả** fold 0 lẫn fold 6 |
| **F2** | recall NOC4 zero-shot (bản ghép 5 seed) tăng ≥ 0.10 so với E3 ở cả hai fold |
| **F3** | `deep89nf` (bỏ tần số) không kém `deep89` quá 0.02 (tức tần số từ panel không phải nguồn chính) |
| **F4** | fine-tune toàn nhánh 371 nhãn (công thức M-g, mục 27), strict ≥ E3 cùng công thức + 0.03 ở cả hai fold |
| kiểm phụ | AUC của plp trên peak DEV sinh ≥ 0.90 (không thì đặc trưng plp vô nghĩa) |

Báo cả giao thức deepNoC cho F4. Nếu F1 sai thì dừng, không chỉnh siêu tham số trên fold 0/6.

## 19. Nhánh NoC trong chính Set Transformer, không dính ID, CHỈ zero-shot (ghi 03/10, TRƯỚC khi chạy)

Yêu cầu của Mạnh (03/10): bỏ fine-tune, chỉ zero-shot. Không cần 89 đặc trưng tự thiết kế: để Set Transformer tự học quan hệ
giữa các peak.

**Thiết kế** (`enable_noc_branch` trong `code/models/set_transformer.py`, mặc định tắt; driver `work/stbranch.py`):
- Nhánh đọc **mọi peak**, không lọc panel (`apply_feas=False`). Token 8 trường tính lại trên toàn bộ peak, không qua panel.
- Nhánh **không đọc** slot genotype, gate hay `logits_cls`.
- 2 ISAB riêng (d = 64, 16 điểm cảm ứng) để các peak chú ý lẫn nhau (stutter, allele chung), rồi PMA 4 hạt.
- Ghép 19 MAC, rồi đầu thứ tự P(NOC > k).
- Hai nguồn đầu vào:
  - `backbone`: đầu ra từng peak của encoder backbone 50k **đóng băng**;
  - `scratch`: embedding riêng của nhánh trên token thô.
- Train bằng NOC thật của hỗn hợp sinh: **6.000 hồ sơ cố định (rng 42)**, DEV sinh 1.000, **10 epoch** (demo), AdamW 1e-3, batch 128
  (Mạnh 03/10: mẫu nhỏ). Mốc deep89 chạy lại ở **cùng 6.000 / 1.000**, cùng 3 seed, để so cùng cỡ dữ liệu.
  Chọn epoch theo DEV sinh; giải mã argmax; seed 42, 43, 44; fold 0 và 6.
- MPS: ISAB/PMA cần `.contiguous()` cho truy vấn mở rộng (chỉ khi tensor ở MPS; CPU/CUDA giữ nguyên từng bit). Lượt ID
  của `[ID GUARD]` chạy trên CPU.
- Log `[ID GUARD]`: tham số backbone và `fp(logits_cls)` trước và sau phải trùng.

| | giả thuyết (mốc: deep89 zero-shot ở cùng 6.000 hồ sơ, chạy cùng lúc; `logits_card` 50k 0.8025 / 0.8884) |
|---|---|
| **G1** | nhánh tốt hơn trong hai bản, TB 3 seed ≥ deep89 + 0.03 ở **cả** fold 0 lẫn fold 6 |
| **G2** | nhánh tốt hơn ≥ `logits_card` của cùng backbone ở cả hai fold |
| **G3** | `backbone` ≥ `scratch` + 0.02 ở cả hai fold (biểu diễn học cho ID có giúp đếm) |
| bất biến | `[ID GUARD] params_unchanged=True`, fp trùng, ở mọi lần chạy |

*Sửa 03/10 (Mạnh: demo chạy ít epoch): 25 → 10 epoch cho MỌI bản, kể cả mốc deep89. Lượt 25 epoch fold 0 đã chạy xong và
đã xem (backbone 0.725 / 0.741 / 0.710; scratch 0.689 / 0.689 / 0.645), lưu ở `work/stbranch/ep25`; không dùng để chọn gì.*

Không chỉnh siêu tham số theo test. Nếu G1 sai thì ghi lại và dừng hướng này ở cấu hình này. Một cấu hình không đủ để đóng
hướng (bài học 15), nên nếu sai sẽ ghi rõ "sai ở cấu hình này".

## 20. Đường cong theo cỡ dữ liệu: nhánh ST `backbone`, CHỈ strict, zero-shot (ghi 03/10, TRƯỚC khi chạy)

Mạnh 03/10: chỉ chạy strict, phải khó hơn paper deepNoC.
- Zero-shot không dùng nhãn thật nào. Mỗi lần chạy in `[STRICT]` và dừng nếu có tổ hợp hỗn hợp nào của test nằm trong train/DEV.
- Cấu hình y hệt mục 19 (10 epoch, DEV sinh 1.000, seed 42–44, fold 0 và 6). Chỉ đổi cỡ train: 6.000 (đã có),
  **15.000** và **toàn bộ** train sinh của fold (47.494 / 47.267).
- Không đổi siêu tham số theo cỡ dữ liệu.

| | giả thuyết |
|---|---|
| **C1** | TB 3 seed tăng theo cỡ dữ liệu ở cả hai fold, và bản toàn bộ ≥ bản 6k + 0.03 ở **cả** fold 0 lẫn fold 6 |
| **C2** | khoảng cách tới `logits_card` (0.8025 / 0.8884) ở bản toàn bộ thu hẹp ≥ một nửa so với bản 6k (6k: 0.099 / 0.195) ở cả hai fold |
| bất biến | `[ID GUARD]` đạt và `[STRICT] shared 0` ở mọi lần chạy |

## 21. Train 60.000 hồ sơ: nhánh ST `backbone`, CHỈ strict, zero-shot (ghi 03/10, TRƯỚC khi sinh và chạy)

Mạnh 03/10: "thử nâng mức 60k train". Đây là lần đầu **sinh thêm** dữ liệu, theo yêu cầu trực tiếp của Mạnh.
- Generator, cờ và bundle **không đổi** (`--build N --noc_weights 1,1.5,2.5,2 --conditioned`); chỉ khác `--seed 43`.
- Train cũ (47.494 / 47.267 hàng) giữ nguyên từng hàng. Thêm hỗn hợp mới cho đủ **60.000**; không thêm NOC1 thật.
- Bỏ mọi hỗn hợp mới có tổ hợp nằm trong DEV hoặc test. DEV 1.000 và test không đổi.
- Script `work/build_extra.py`; dữ liệu ở `work/offline/fold*_60k`.
- Nhánh, cấu hình, 10 epoch, seed 42–44 y hệt mục 20.

| | giả thuyết (mốc: bản toàn bộ ~47k, mục 34 failed_experiments: 0.795 ± 0.009 / 0.803 ± 0.015) |
|---|---|
| **K1** | 60k ≥ 47k + 0.01 (TB 3 seed) ở **cả** fold 0 lẫn fold 6 |
| **K2** | (chỉ báo) ở 60k vẫn chọn epoch cuối, tức là chưa bão hoà |
| bất biến | `[STRICT] shared 0`, `[ID GUARD]` đạt |

Nếu K1 sai thì kết luận "từ 47k lên 60k không đủ để thấy khác biệt ở 10 epoch". Không khái quát thành "thêm dữ liệu không giúp".
