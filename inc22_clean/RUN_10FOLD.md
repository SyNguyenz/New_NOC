# Chạy 10-fold unknown-donor cho `inc22_fixed_aslot`

Mục tiêu: mỗi donor trong 50 donor PROVEDIt được làm **unknown (hold-out)** đúng **một lần**.
Fold 0 (`unknown = 6, 21, 26, 40, 50`) **đã chạy xong** — tài liệu này hướng dẫn chạy **fold 1 → 9**.

Quy trình: **preprocess ở local** (CPU, nhẹ) → **upload dataset lên Kaggle** → **train trên GPU Kaggle**.

---

## 0. Bảng 10 fold

Fold = cắt permutation seed-42 của `[1..50]` thành 10 khối 5 người. `STR_FOLD` chọn khối.

| fold | unknown donors | tổng mẫu mixture thật (→ test) | open |
|:---:|---|---:|---:|
| 0 | 6, 21, 26, 40, 50 | 1333 | 1366 |
| 1 | 18, 28, 46, 47, 48 | 1468 | 1339 |
| 2 | 19, 24, 32, 38, 41 | 1137 | 1705 |
| 3 | 5, 8, 25, 30, 39 | 1423 | 1365 |
| 4 | 10, 22, 27, 29, 35 | 1345 | 1435 |
| 5 | 16, 17, 33, 42, 43 | 1358 | 1628 |
| 6 | 4, 11, 31, 44, 45 | 1369 | 1526 |
| 7 | 7, 12, 20, 23, 36 | 1642 | 1097 |
| 8 | 1, 3, 13, 15, 49 | 1661 | 1143 |
| 9 | 2, 9, 14, 34, 37 | 1357 | 1543 |

`n_classes` luôn = **45** ở mọi fold (50 − 5), nên không chỗ nào trong model cần đổi.

⚠️ **Fold 0 cũng phải chạy lại.** Split policy đã đổi (mục 1), test mang nghĩa khác nên con số
`inc22_fixed_aslot_seed42` cũ không so sánh được với 10 fold mới. Chạy **cả 10 fold**, không phải 9.

---

## 1. Những gì đã được patch (đọc để hiểu, không cần làm gì)

| file | thay đổi |
|---|---|
| `prepare_data_set.py` | **chuyển từ `data/` lên `inc22_clean/`** (để `data/` thuần output, không phải zip); (a) donor split đọc `STR_FOLD` (0–9), ghi `data/fold_info.json`; (b) **mọi combo thật → test**, val/train chỉ còn single-source; (c) ghi `combo_id_{split}.npy` |
| `preprocess.py` | y hệt, cho bản one-command |
| `kaggle_run_increment1.py` | gọi `prepare_data_set.py` ở đường dẫn mới |
| `make_insilico.py` | combo loại trừ **đọc từ `meta_set.json`** thay vì hard-code fold 0; tự copy `combo_id_*`, `fold_info.json`, `donor_geno*` sang `data_insilico_w/` → thư mục xuất ra là self-contained, không cần `cp` thủ công |
| `train_set_transformer.py` | (a) `DEVICE`: CUDA → **MPS (Mac)** → CPU, override `STR_DEVICE`; (b) decode fit **leave-one-combo-out** (`loco_decode`) |

### 1.1. Vì sao đổi split policy

Network **không bao giờ train trên mixture thật** — `make_insilico.build_train` giữ real `noc == 1` rồi
sinh 50000 mixture in-silico, còn real mixture trong split train bị **drop thẳng**:

```python
ss = ntr == 1
Xf_list = [Xtr[ss]]; tok_real = np.load(DATA/"tokens_train.npy")[ss]
```

Nên việc giữ combo thật ở `train` không bảo vệ gì cả, chỉ làm teo tập đánh giá: fold 0 chỉ report được
**233 / 1333** mẫu mixture thật (1 combo mỗi NOC), 858 mẫu đi vào train rồi bị vứt. Cộng 10 fold là
**9156 mẫu mixture thật** bị lãng phí.

Chính sách mới: **mọi combo thật vào test**, và loại toàn bộ chúng khỏi sinh in-silico. Test mixture
tăng **~5.7×** (182–338 → 1137–1661 mỗi fold).

### 1.2. Vì sao leave-one-combo-out chứ không giữ val

Thứ duy nhất fit trên mixture thật là **1 scalar `alpha`** (phi-rerank) và **1 RandomForest nhỏ**
(count). Đốt cả combo để làm tập fit cho hai thứ đó là lãng phí khi dữ liệu khan. `loco_decode` thay
bằng: với mỗi combo `c`, fit `alpha` + RF trên **tất cả combo khác** (+ val single-source làm ví dụ
k=1), rồi predict trên `c`. Mẫu single-source trong test là một group riêng, fit trên toàn bộ mixture.

Bảo đảm chống leak **y hệt** held-out cũ (group-disjoint ở mức combo, network chưa thấy combo nào),
nhưng không combo nào bị đốt. Có `assert not fit[g].any()` trong `loco_decode` chặn tự-fit.

Trong `metrics.json`: `phi_rerank_alpha` = alpha fit trên **toàn bộ** mixture (artifact để deploy),
`phi_rerank_alpha_per_group` = alpha từng group (chỉ để estimate trung thực), `n_decode_groups`,
`n_test_mixtures`.

Khi viết paper phải gọi đúng tên: đây là **cross-validated estimate** của decode stage, không phải một
held-out set duy nhất.

### 1.3. Vì sao patch `make_insilico.py` là bắt buộc

Hai dict combo cũ (`{2:[(31,32)], 3:[(46,47,48)], …}`) đúng bằng combo val/test **của riêng fold 0**.
Ở fold khác, `preprocess` chọn combo khác → nếu giữ literal thì `build_train` sẽ sinh in-silico mixture
**trùng đúng combo test thật của fold đó** → leak, hỏng metric. Giờ đọc từ meta nên tự động đúng, và
với policy mới nó loại **toàn bộ** combo thật.

### 1.4. Chú ý về `alpha` và số cũ

`phi_rerank` trước đây tune alpha trên real val — val giờ chỉ còn single-source nên đường đó không còn
nghĩa, LOCO thay thế hoàn toàn. Nếu chạy trên bundle cũ (không có `combo_id_test.npy`),
`train_set_transformer` tự rơi về đường legacy (tune trên val) và ghi rõ trong `decode`.

---

## 2. Chuẩn bị local (làm 1 lần)

### 2.1. Tạo junction/symlink `data_raw`

`inc22_clean/` cần thấy thư mục raw PROVEDIt. Chạy **ở project root**:

Windows (PowerShell / cmd, cần quyền tạo junction):

```bash
cmd /c mklink /J "inc22_clean\data_raw" "data_raw"
```

macOS / Linux:

```bash
ln -s ../data_raw inc22_clean/data_raw
```

Kiểm tra:

```bash
ls inc22_clean/data_raw
```

Phải thấy `PROVEDIt_1-5-Person CSVs Filtered/` và file `PROVEDIt_RD14-0003 GF Known Genotypes.xlsx`.

### 2.2. Môi trường Python

Cần Python **≥ 3.10** (code dùng type hint `float | None`), `numpy`, `pandas`, `scikit-learn`, `openpyxl`.
Bước preprocess **không cần GPU, không cần torch** — chạy được trên máy Mac bình thường.

```bash
pip install -r requirements.txt
```

---

## 3. Bước 1 — Preprocess local cho từng fold

Chạy **ở project root**, đổi `STR_FOLD` cho mỗi fold:

```bash
STR_FOLD=1 python inc22_clean/preprocess.py
```

Trên Windows PowerShell:

```bash
$env:STR_FOLD=1; python inc22_clean/preprocess.py
```

Lệnh này làm hết chuỗi: raw CSV → `inc22_clean/data/` (tokens, mask, Xflat, y, noc, size, phi/condition,
donor genotypes, attr) → `make_insilico.py --build 50000` → `inc22_clean/data_insilico_w/`.

**Thời gian (đo thực tế fold 0 trên CPU):** phần đọc CSV + build array + donor_geno + phi/condition +
enrich mất **~5 phút**. Phần `make_insilico --build 50000` chưa đo riêng, ước lượng 10–30 phút. Chạy nền,
làm lần lượt từng fold.

### 3.1. Sanity-check ngay sau khi chạy xong

```bash
python -c "import json;m=json.load(open('inc22_clean/data/meta_set.json'));f=json.load(open('inc22_clean/data/fold_info.json'));mp=m['split_policy']['multi_person_combos'];print('fold',f['fold']);print('unknown',m['unknown_donors']);print('n_classes',m['n_classes']);print('sizes',m['split_sizes']);print('n_test_combos',{k:len(v) for k,v in mp['test'].items()});print('val combos (must be empty)',mp['val'])"
```

Đối chiếu:

- `fold` = số fold bạn vừa chạy
- `unknown` khớp cột "unknown donors" trong bảng mục 0
- `n_classes` = 45
- `n_test_combos` khớp cột "combo N2/N3/N4/N5" — **toàn bộ** combo thật của fold đó phải nằm ở test
- `split_sizes["test"]` ≈ (single-source test) + (tổng mẫu mixture ở bảng mục 0)
- `val` **không được** chứa combo nào (`multi_person_combos["val"] == {}`)

Trong log của `make_insilico` phải thấy dòng:

```
excluding real combos from in-silico train — test=[...] val=[]
```

và danh sách `test=` phải **khớp đủ** `split_policy.multi_person_combos["test"]` của fold đó — đây chính
là bằng chứng chống leak: không combo thật nào được tái tạo trong in-silico train.

### 3.2. Đổi tên thư mục kết quả theo fold (tránh ghi đè khi chạy fold tiếp theo)

`data/` và `data_insilico_w/` bị **ghi đè** mỗi lần chạy preprocess. Trước khi chạy fold kế tiếp:

```bash
mv inc22_clean/data_insilico_w inc22_clean/folds/fold1_insilico_w
```

(tạo `inc22_clean/folds/` trước nếu chưa có). Thư mục `inc22_clean/data/` không cần giữ — mọi thứ Kaggle
cần đã nằm trong `*_insilico_w`.

---

## 4. Bước 2 — Đóng gói & upload lên Kaggle

> **Bạn tự zip.** Phần dưới chỉ quy định *nội dung* và *tên* dataset.

### 4.1. Dataset CODE (upload 1 lần, dùng cho cả 10 fold)

- **Zip nội dung:** toàn bộ `inc22_clean/`, **bỏ hẳn các thư mục dữ liệu**: `data/`, `data_raw/`,
  `data_insilico_w/`, `data_w_inc22/`, `folds/`, `results/`, mọi `__pycache__/`, và `diagrams/`
  (không cần khi train). Còn lại chỉ ~200 KB code — **không còn file .py nào nằm trong `data/`**,
  `prepare_data_set.py` đã chuyển lên `inc22_clean/`.
- **Cấu trúc bên trong zip** (quan trọng — giữ nguyên cây thư mục):

  ```
  inc22_clean/
    kaggle_run_increment1.py
    prepare_data_set.py          <- đã chuyển từ data/ lên đây
    preprocess.py
    train_set_transformer.py
    phi_rerank.py
    make_dev_split.py
    make_insilico.py
    build_donor_geno.py
    build_real_attr.py
    extract_phi_condition.py
    extract_size.py
    verify_inc22.py
    models/__init__.py
    models/set_transformer.py
    features/enrich.py
    synth/extract_genotypes.py
  ```

  `data/` sẽ được `prepare_data_set.py` / `preprocess.py` tự tạo khi chạy, nên không cần đưa vào zip.

- **Kaggle → Datasets → New Dataset**
  - Title: `noc-inc22-code`
  - slug (URL) sẽ là: `noc-inc22-code`
  - → mount tại `/kaggle/input/noc-inc22-code/inc22_clean/`

Khi sửa code, tạo **New Version** của chính dataset này (giữ nguyên slug).

### 4.2. Dataset DATA (một dataset cho mỗi fold)

- **Zip nội dung:** thư mục `foldK_insilico_w/` (đổi tên thư mục gốc bên trong thành `data_insilico_w`)
- **Cấu trúc bên trong zip:**

  ```
  data_insilico_w/
    tokens_train.npy  mask_train.npy  Xflat_train.npy  y_train_set.npy  noc_train.npy
    attr_train.npy    phi_train.npy   size_train.npy
    tokens_val.npy    mask_val.npy    ...  (val + test đầy đủ)
    tokens_open.npy   mask_open.npy   Xflat_open.npy   size_open.npy
    combo_id_test.npy combo_id_val.npy        <- BẮT BUỘC, thiếu là rơi về đường legacy
    meta_set.json     fold_info.json
    donor_geno.npy    donor_geno_mask.npy
  ```

  Tất cả các file trên do `preprocess.py` sinh/copy sẵn vào `data_insilico_w/` — chỉ cần zip nguyên
  thư mục, không phải gom thêm gì.

- **Title / slug:** `noc-inc22-fold0`, `noc-inc22-fold1`, … `noc-inc22-fold9`
  → mount tại `/kaggle/input/noc-inc22-fold1/data_insilico_w/`

Mỗi dataset khoảng **300–450 MB**. Tách theo fold để upload dần và train song song nhiều notebook.

> Đường dẫn mount thật luôn hiện ở sidebar "Input" của notebook. Nếu khác, chạy `!ls /kaggle/input` và
> `!ls /kaggle/input/<slug>` để lấy đúng path rồi sửa lại biến trong cell.

---

## 5. Bước 3 — Notebook Kaggle (GPU)

Bật **Accelerator: GPU T4 x2 / P100** và **Internet: off** (không cần mạng).

### Cell 1 — copy code vào working dir

```python
!cp -r /kaggle/input/noc-inc22-code/inc22_clean/* /kaggle/working/
import os; os.chdir('/kaggle/working')
!ls
```

### Cell 2 — chọn fold & verify trước khi train

```python
FOLD = 1
INSILICO = f'/kaggle/input/noc-inc22-fold{FOLD}/data_insilico_w'

import json, os, numpy as np
m = json.load(open(f'{INSILICO}/meta_set.json'))
f = json.load(open(f'{INSILICO}/fold_info.json'))
cid = np.load(f'{INSILICO}/combo_id_test.npy')
print('fold        :', f['fold'])
print('unknown     :', m['unknown_donors'])
print('n_classes   :', m['n_classes'])
print('sizes       :', m['split_sizes'])
print('test combos :', len(np.unique(cid[cid >= 0])), '| test mixtures:', int((cid >= 0).sum()))
assert f['fold'] == FOLD, 'dataset không khớp fold đang định chạy!'
assert m['split_policy']['multi_person_combos']['val'] == {}, 'val còn combo — dataset build bằng code cũ'
```

Đối chiếu `unknown` và `test mixtures` với bảng mục 0. **Bước này quan trọng** — nó chặn việc train nhầm
fold, và chặn việc dùng dataset build bằng code trước khi đổi split policy.

### Cell 3 — train

```python
!INSILICO_W=/kaggle/input/noc-inc22-fold1/data_insilico_w \
    python kaggle_run_increment1.py --seed 42 --out_subdir inc22_fixed_aslot_fold1
```

Đổi `fold1` ở **cả 2 chỗ** cho mỗi fold. Lệnh này sẽ:

1. copy dataset → `/kaggle/working/data_w_inc22/` (writable)
2. `make_dev_split.py` — carve DEV split combo-disjoint
3. `features/enrich.py` — build `tokens8/9/11`
4. `train_set_transformer.py` — train + phi-rerank + post-hoc RF count

**Không cần set `STR_FOLD` trên Kaggle** — fold đã "đóng băng" trong dataset từ bước preprocess local.
`STR_FOLD` chỉ có tác dụng ở bước preprocess.

Nếu notebook bị restart mà `data_w_inc22/` vẫn còn, chạy lại với `--skip-prep` để bỏ qua bước 1–3.

### Cell 4 — thu kết quả

```python
import json
r = f'/kaggle/working/results/inc22_fixed_aslot_fold{FOLD}_seed42'
print(json.dumps(json.load(open(f'{r}/metrics.json')), indent=2)[:2000])
!ls -la {r}
```

Kết quả nằm ở `results/inc22_fixed_aslot_fold{K}_seed42/`:

| file | nội dung |
|---|---|
| `metrics.json` | `per_noc_oracle`, `per_noc_post_hoc`, `dev_per_noc_oracle`, `count_acc`, `reject_auroc`, `phi_rerank_alpha` (bản deploy), `phi_rerank_alpha_per_group`, `n_decode_groups`, `n_test_mixtures` |
| `best_model.pt` | checkpoint |
| `y_test_pred.npy` / `y_test_true.npy` | dự đoán trên real test |

Kiểm tra nhanh: `n_decode_groups` phải = (số combo thật của fold) + 1 (group single-source), và
`decode` phải chứa `leave-one-combo-out fit`. Nếu thấy `[legacy: no combo_id_test]` thì dataset thiếu
`combo_id_test.npy` — số đo vẫn chạy nhưng chỉ trên đường cũ, phải upload lại.

Download 4 file này (hoặc "Save Version" để giữ output). Lưu về local theo cây:

```
results_10fold/
  fold0/metrics.json …
  fold1/metrics.json …
  …
  fold9/metrics.json …
```

Khi gộp 10 fold: mỗi ô per-NOC phải kèm `n` (số mixture của NOC đó trong fold), vì số combo mỗi NOC
khác nhau giữa các fold. Đừng lấy trung bình trần của 10 con số.

---

## 6. Lưu ý & xử lý sự cố

**Chạy trên Mac.** Preprocess là numpy/pandas thuần → chạy bình thường. Nếu muốn train thử ngay trên Mac:

```bash
STR_DATA_DIR=inc22_clean/data_w_inc22 python inc22_clean/train_set_transformer.py --seed 42 --out_subdir smoke
```

Tự chọn MPS trên Apple Silicon. Nếu gặp lỗi op chưa hỗ trợ trên MPS, ép CPU:

```bash
STR_DEVICE=cpu STR_DATA_DIR=inc22_clean/data_w_inc22 python inc22_clean/train_set_transformer.py --seed 42 --out_subdir smoke
```

MPS chậm hơn T4 nhiều — chỉ nên dùng để smoke-test, không dùng chạy thật 150 epoch.

**Số combo mỗi NOC thay đổi theo fold.** Fold 9 chỉ còn 2 combo NOC=5, fold 2/3/4/5 còn 3, fold 8 có 6.
Không ảnh hưởng training (train là in-silico), nhưng ảnh hưởng độ chắc của ô NOC5 → luôn báo cáo kèm `n`.
Với LOCO, combo NOC5 ít cũng có nghĩa alpha/RF của group đó được fit trên ít mixture NOC5 hơn — đây là
biến động thật của fold, không phải bug.

**"Novel combo" mạnh yếu khác nhau theo NOC.** Với 50000 mixture in-silico và `noc_weights=1,1.5,2.5,2`:
NOC2 phủ ~99.9% của C(45,2)=990 combo, NOC3 ~53%, NOC4 ~11%, NOC5 ~1.2%. Nghĩa là ở NOC2 model đã thấy
gần như toàn bộ không gian combo — combo thật bị loại chỉ là vài cái duy nhất chưa thấy. Claim
"novel-combo held-out" chỉ thực sự mạnh ở NOC4/NOC5; viết paper nên nói rõ thay vì gộp một chữ.

**`noc_open.npy` đếm thiếu.** `nocs = labels.sum(1)` chỉ đếm known donor, nên mẫu open bị undercount NOC
thật. Vô hại với pipeline này (`OpenSetDataset` chỉ đọc `tokens8_open` + `mask_open`), nhưng đừng dùng
`noc_open.npy` để phân tích.

**Dung lượng `/kaggle/working`.** `data_w_inc22/` sau enrich khoảng 1.5–2 GB (tokens9 + tokens11). Vẫn dưới
giới hạn 20 GB, nhưng nếu "Save Version" thì nên xóa `data_w_inc22/` trước khi commit để output nhẹ.

**Tối ưu (tùy chọn, chưa làm).** `tokens/mask/Xflat/size` là **fold-invariant** — chỉ phần chia split
thay đổi theo fold. Về lý thuyết có thể cache pass đọc CSV và tái dùng cho cả 10 fold. Chưa làm vì sẽ
phải sửa cấu trúc `csv_pass()`; thực đo fold 0 chỉ mất ~4 phút nên lợi ích không đáng rủi ro.

