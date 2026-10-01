# Chạy 10 fold NoC + ID: mỗi người 2 fold

Người điều phối: **Mạnh**. Mỗi người tốn khoảng **7 giờ GPU Kaggle**, trong **một lần commit**.

Mỗi fold cho ra **hai kết quả để so với deepNoC**, đo trên cùng một backbone:

| dòng | nhãn hỗn hợp thật | fold 0 đã đo |
|---|---|---|
| **zero-shot** (không fine-tune) | 0 | macro F1 0.9102 |
| **fine-tune `head_only`** | 371 (bằng deepNoC: 34/87/79/93/78) | 0.9026, đo trên backbone 12k; trên backbone 50k sẽ có sau lần chạy này |
| *deepNoC (Taylor & Humphries 2024)* | *371* | *0.9097* |

Phạm vi: **chỉ tính trong panel**, tức hồ sơ mà mọi người góp đều thuộc 45 donor của fold. Tập test 372 hồ sơ của deepNoC cũng toàn donor đã gặp lúc fine-tune.
Lý do chọn thiết kế này: `doc/chay_10fold_zeroshot50k.md`.

---

## 1. Phân công

| người | `NGUOI` | `FOLDS` | file dữ liệu cần |
|---|---|---|---|
| Mạnh | `'manh'` | `[0, 1]` | `noc-cond50k-fold0.zip`, `noc-cond50k-fold1.zip` |
| Nguyên | `'nguyen'` | `[2, 3]` | `noc-cond50k-fold2.zip`, `noc-cond50k-fold3.zip` |
| Dũng | `'dung'` | `[4, 5]` | `noc-cond50k-fold4.zip`, `noc-cond50k-fold5.zip` |
| Quang | `'quang'` | `[6, 7]` | `noc-cond50k-fold6.zip`, `noc-cond50k-fold7.zip` |
| Giang | `'giang'` | `[8, 9]` | `noc-cond50k-fold8.zip`, `noc-cond50k-fold9.zip` |

Ai cũng cần thêm **`noc-fold-code.zip`** (code) và **`cell_zeroshot50k_10fold.py`** (cell chạy).

## 2. Các file trong thư mục này

| file | là gì | ai cần |
|---|---|---|
| `noc-fold-code.zip` | code, CODE **`529535b846`** | mọi người |
| `noc-cond50k-fold<F>.zip` (10 file, ~116 MB mỗi file) | dữ liệu fold F: 50.000 hỗn hợp sinh `--conditioned` + hồ sơ thật của fold | người chạy fold F |
| `cell_zeroshot50k_10fold.py` | cell Kaggle, chỉ sửa 2 dòng | mọi người |
| `cond50k_manifest.json` | md5 của 10 bộ dữ liệu (cell tự kiểm) | để đối chiếu |
| `noc-cond50k-fold<F>.build.log` | log sinh dữ liệu | để lưu vết |

Cả 10 bộ dữ liệu được sinh **trên một máy, bằng một lệnh**: `python3 kaggle_upload/tools/build_cond50k_folds.py`.
Không fold nào có tổ hợp test lọt vào dữ liệu sinh (`shared 0`). Mỗi file zip có `build_info.json` ghi lệnh, md5 và môi trường.

## 3. Đưa dữ liệu lên Kaggle

Chọn **một** trong hai cách.

**Cách A (nên dùng): Mạnh upload rồi chia sẻ.**
1. Mạnh tạo 11 dataset: `noc-fold-code` (New Version nếu đã có) và `noc-cond50k-fold0` … `noc-cond50k-fold9`. Mỗi dataset upload đúng **một** file zip.
2. Với mỗi dataset: vào **Settings → Sharing**, thêm tài khoản Kaggle của 4 người còn lại (hoặc để Public).

**Cách B: mỗi người tự upload.**
Mỗi người tạo dataset riêng từ `noc-fold-code.zip` và 2 file `noc-cond50k-fold<F>.zip` của mình. Tên dataset đặt tuỳ ý: cell tìm theo **tên thư mục bên trong zip**, không theo tên dataset.

Khi upload, **không giải nén trước**. Kaggle tự giải nén và giữ nguyên thư mục `data_cond50k_fold<F>/`.

## 4. Chạy (mỗi người)

1. **Kiểm tài khoản:** số điện thoại đã xác minh (có xác minh mới dùng được GPU), và còn **≥ 8 giờ** GPU trong tuần (**Settings → Accelerator quota**).
2. **Tạo notebook mới:**
   - **Accelerator:** GPU T4 x2 hoặc P100.
   - **Internet:** Off.
3. **Add Data:**
   - `noc-fold-code`
   - **2** dataset `noc-cond50k-fold<F>` của mình. **Không** gắn fold của người khác.
4. Dán toàn bộ `cell_zeroshot50k_10fold.py` vào **một** cell, rồi **chỉ** sửa hai dòng đầu:
   ```python
   NGUOI = 'dung'      # tên mình, không dấu, theo bảng mục 1
   FOLDS = [4, 5]      # 2 fold của mình
   ```
   Giữ nguyên `RUN_BUDGET = True` để có cả dòng fine-tune.
5. Bấm **Save Version → Save & Run All (Commit)**.
   **Không chạy interactive.** Đóng tab hoặc mất mạng là phiên chết, mất hết. Nhóm đã mất backbone ba lần vì lỗi này.

## 5. Cell làm gì với mỗi fold

```
dữ liệu noc-cond50k-foldF
  -> Phase 1: train backbone TỪ ĐẦU, 150 epoch (~3,3 giờ). Mỗi fold một backbone riêng.
     backbone được lưu NGAY khi train xong -> send/foldF/backbone/
  -> zero-shot: đếm NoC bằng backbone, 0 nhãn thật, rồi ID top-k                  (~5 phút)
  -> fine-tune 371 nhãn trên CHÍNH backbone đó, 3 lần rút, 7 cách, 2 giao thức      (~12 phút)
     dòng chính là head_only; các cách còn lại để làm bảng phụ
```

Backbone luôn đóng băng sau Phase 1, nên ID không bị phần đếm chạm vào. Mỗi lần chạy, log in `[ID GUARD] ... params_unchanged=True` để chứng minh điều đó.

## 6. Kiểm log khi đang chạy

Mở version đang chạy, vào tab **Logs**. Mỗi fold phải có:

| dòng | phải là |
|---|---|
| `CODE 529535b846 OK` | đúng hash này |
| `fold F: ... md5 ...` | cell tự so với manifest; sai thì cell dừng |
| `[FOLD] data dir: unknown_donors=...` | đúng donor ẩn của fold (bảng mục 9) |
| `params: 1,505,737` | đúng con số này |
| `Ep  10 \| ... (Ns)` | khoảng 70–80 s mỗi epoch |
| `>>> backbone da luu -> .../send/foldF/backbone` | ngay sau `Training done in` |
| `[SPLIT AUDIT] ... SHARED=0  (clean)` | **bắt buộc 0** |
| `[ID GUARD] ... params_unchanged=True` | **bắt buộc True**, ở lượt zero-shot và cả 3 lượt fine-tune |
| `fit on 371 of ...` | lượt fine-tune dùng đúng 371 nhãn (lớp nào không đủ hàng thì có thể ít hơn) |

## 7. Gửi kết quả

Khi version chạy xong (khoảng 7 giờ):
1. Mở version đó → tab **Output** → tải file **`send_<tên>.zip`**.
2. Gửi file đó vào nhóm cho Mạnh, kèm ảnh chụp khối `===== KET QUA <tên>: fold [...]` ở cuối log.

Trong `send_<tên>.zip`, mỗi fold có:
- `fold<F>_seed42.json`: file để gộp 10 fold
- metrics và log của zero-shot cùng 3 lượt fine-tune
- backbone (`best_model.pt` + `best_model.fold.json`)

## 8. Sự cố hay gặp

| hiện tượng | cách xử lý |
|---|---|
| `dataset code la CODE xxx, can 529535b846` | **Input** → bấm ⋮ cạnh `noc-fold-code` → **Check for updates** (hoặc gỡ ra rồi gắn lại) |
| `can dung 1 thu muc data_cond50k_fold<F>` | Chưa gắn dataset của fold đó, hoặc gắn trùng hai lần |
| `md5 ... khac ban da chot` | Gắn nhầm dữ liệu. Báo Mạnh, **không chạy tiếp** |
| `assert NGUOI != 'ten_ban'` | Chưa sửa dòng `NGUOI` |
| `!!! FOLD F LOI` | Cell vẫn chạy fold còn lại. Gửi `send/fold<F>/*.log` cho Mạnh |
| Phiên chết sau khi Phase 1 xong | Backbone đã ở `send/fold<F>/backbone`. Mạnh sẽ gửi cell ngắn (~20 phút) chạy lại phần đếm với `--init_from` |
| Hết hạn mức GPU | Chạy từng fold: `FOLDS = [4]`, tuần sau `FOLDS = [5]` |

## 9. Bảng 10 fold

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

## 10. Gộp kết quả (Mạnh)

Giải nén mọi `send_<tên>.zip` vào cùng thư mục `send_all/` (sẽ ra `send_all/fold0/` … `send_all/fold9/`), rồi chạy:

```bash
python3 code/report_folds.py send_all/fold* --out doc/ket_qua_10fold_zeroshot50k.md --csv doc/ket_qua_10fold_zeroshot50k.csv
```

Lệnh trên gộp **zero-shot** (`fold<F>_seed42.json`): một dòng cho mỗi fold, mean ± sd và số gộp (cộng dồn ma trận nhầm lẫn).

Gộp **fine-tune 371 nhãn** (`budget_fold<F>_d<D>.metrics.json`, đủ các cách, 3 lần rút) bằng lệnh riêng:

```bash
python3 code/report_budget.py send_all/ --out doc/ket_qua_10fold_finetune371.md
python3 code/report_budget.py send_all/ --exclude_selection --out doc/ket_qua_9fold_finetune371.md
```

Lệnh thứ hai bỏ fold 0 (fold dùng để chọn `head_only`). Đó là số dùng cho giả thuyết T5.
Kết quả được đọc theo giả thuyết T1–T6 đã ghi **trước khi chạy**, ở `doc/chay_10fold_zeroshot50k.md`, mục 9.
