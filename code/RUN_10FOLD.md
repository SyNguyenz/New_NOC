# Chạy 10-fold unknown-donor cho `inc22_fixed_aslot`

Mỗi donor trong 50 donor PROVEDIt được làm **unknown (hold-out)** đúng một lần.
Quy trình: **preprocess ở local** (CPU) → **upload dataset lên Kaggle** → **train trên GPU Kaggle**.

Mọi lệnh chạy từ **repo root** (thư mục chứa `code/`).

---

## 1. Bảng 10 fold

Fold = cắt permutation seed-42 của `[1..50]` thành 10 khối 5 người; `STR_FOLD` chọn khối.

| fold | unknown donors | combo thật N2/N3/N4/N5 | mẫu mixture (→ test) | open |
|:---:|---|---|---:|---:|
| 0 | 6, 21, 26, 40, 50 | 5 / 6 / 4 / 5 | 1333 | 1366 |
| 1 | 18, 28, 46, 47, 48 | 7 / 4 / 6 / 5 | 1468 | 1339 |
| 2 | 19, 24, 32, 38, 41 | 6 / 4 / 4 / 3 | 1137 | 1705 |
| 3 | 5, 8, 25, 30, 39 | 7 / 6 / 5 / 3 | 1423 | 1365 |
| 4 | 10, 22, 27, 29, 35 | 7 / 6 / 4 / 3 | 1345 | 1435 |
| 5 | 16, 17, 33, 42, 43 | 6 / 6 / 5 / 3 | 1358 | 1628 |
| 6 | 4, 11, 31, 44, 45 | 6 / 5 / 6 / 4 | 1369 | 1526 |
| 7 | 7, 12, 20, 23, 36 | 8 / 6 / 6 / 4 | 1642 | 1097 |
| 8 | 1, 3, 13, 15, 49 | 7 / 5 / 7 / 6 | 1661 | 1143 |
| 9 | 2, 9, 14, 34, 37 | 7 / 6 / 4 / 2 | 1357 | 1543 |

`n_classes` = **45** ở mọi fold (50 − 5).

---

## 2. Split & flags

- **NOC=1**: stratify theo donor → mọi donor có mặt ở train + val + test.
- **NOC≥2**: toàn bộ combo thật nằm ở `test`; `train`/`val` chỉ single-source.
- **Decode**: `alpha` (phi-rerank) + count fit leave-one-combo-out theo `combo_id_test.npy`.
- **Checkpoint**: chọn theo in-silico DEV, không early stopping → `best_model.pt` + `last_model.pt`.

Env flag (mặc định đã đúng, chỉ đổi khi cần):

| flag | mặc định | tác dụng |
|---|---|---|
| `STR_FOLD` | `0` | chọn khối 5 donor unknown |
| `STR_PEAK_MODEL` | `0` | `0` = overlay profile NOC1 thật, `1` = peak model |
| `STR_ENRICH_AFTER_FEAS` | `1` | enrich sau feasibility filter |
| `STR_DEVICE` | auto | ép `cuda` / `mps` / `cpu` |
| `STR_NOC_ARM` | `0` | tương đương `--noc_arm` |

### `--noc_arm`

| arm | thay đổi |
|---|---|
| 0 | baseline |
| 1 | `+ 0.05·SmoothL1(Σgate, NOC)` |
| 2 | arm 1, bỏ `logits_card` + post-hoc RF; đếm bằng `tierA_count` |
| 3 | arm 2, bỏ CORN |

Mọi arm ghi thêm `count_acc_gate`, `count_macro_f1_gate_mix`, `corr_sumgate_noc` vào `metrics.json`.

---

## 3. Preprocess local (từng fold)

Cần Python **≥ 3.10** + `numpy`, `pandas`, `scikit-learn`, `openpyxl`. Không cần GPU/torch.
`code/data_raw/` đã có sẵn trong repo — không phải tạo symlink gì.

```bash
STR_FOLD=1 python code/preprocess.py
```

PowerShell:

```bash
$env:STR_FOLD=1; python code/preprocess.py
```

Sinh `code/data/` (real arrays) → `code/data_insilico_w/` (train in-silico + val/test/open thật).
Thời gian: đọc CSV + build array ~5 phút, `make_insilico --build 50000` thêm 10–30 phút.

### 3.1. Sanity-check

```bash
python -c "import json;m=json.load(open('code/data/meta_set.json'));f=json.load(open('code/data/fold_info.json'));mp=m['split_policy']['multi_person_combos'];print('fold',f['fold']);print('unknown',m['unknown_donors']);print('n_classes',m['n_classes']);print('sizes',m['split_sizes']);print('n_test_combos',{k:len(v) for k,v in mp['test'].items()});print('val combos (must be empty)',mp['val'])"
```

- `fold`, `unknown` khớp bảng mục 1
- `n_classes` = 45
- `n_test_combos` khớp cột "combo thật N2/N3/N4/N5" — **toàn bộ** combo của fold nằm ở test
- `multi_person_combos["val"]` phải là `{}`

Trong log `make_insilico` phải thấy `excluding N real combos from in-silico train: [...]`, với **N =
tổng combo thật** của fold. Không combo thật nào được tái tạo trong in-silico train.

### 3.2. Đổi tên trước khi chạy fold kế tiếp

`code/data/` và `code/data_insilico_w/` bị ghi đè mỗi lần chạy:

```bash
mkdir -p code/folds && mv code/data_insilico_w code/folds/fold1_insilico_w
```

Chỉ cần giữ `*_insilico_w` — nó đã self-contained (bao gồm `donor_geno*`, `combo_id_*`, `fold_info.json`).

---

## 4. Đóng gói & upload lên Kaggle

> Bạn tự zip. Phần dưới quy định *nội dung* và *tên* dataset.

### 4.1. Dataset CODE (upload 1 lần, dùng cho cả 10 fold)

Zip thư mục `code/`, **bỏ hẳn**: `data/`, `data_raw/`, `data_insilico_w/`, `data_w_inc22/`, `folds/`,
`results/`, `notebooks/`, mọi `__pycache__/`. Còn lại ~200 KB.

```
code/
  kaggle_run_increment1.py
  prepare_data_set.py
  preprocess.py
  train_set_transformer.py
  train_noc_head.py
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

`data/` sẽ được script tự tạo khi chạy.

- Title / slug: `noc-inc22-code` → mount tại `/kaggle/input/noc-inc22-code/code/`
- Sửa code thì tạo **New Version** của chính dataset này (giữ nguyên slug).

### 4.2. Dataset DATA (một dataset mỗi fold)

Zip `code/folds/foldK_insilico_w/`, đổi tên thư mục gốc trong zip thành `data_insilico_w`:

```
data_insilico_w/
  tokens_train.npy  mask_train.npy  Xflat_train.npy  y_train_set.npy  noc_train.npy
  attr_train.npy    phi_train.npy   size_train.npy
  tokens_val.npy    mask_val.npy    ...  (val + test đầy đủ)
  tokens_open.npy   mask_open.npy   Xflat_open.npy   size_open.npy
  combo_id_test.npy combo_id_val.npy
  meta_set.json     fold_info.json
  donor_geno.npy    donor_geno_mask.npy
```

- Title / slug: `noc-inc22-fold0` … `noc-inc22-fold9` → mount tại `/kaggle/input/noc-inc22-fold1/data_insilico_w/`
- Mỗi dataset ~350 MB.

> Đường dẫn mount thật hiện ở sidebar "Input". Nếu khác, chạy `!ls /kaggle/input/<slug>` rồi sửa biến.

---

## 5. Notebook Kaggle (GPU)

Accelerator: **GPU T4 x2 / P100**. Internet: off.

### Cell 1 — copy code

```python
!cp -r /kaggle/input/noc-inc22-code/code/* /kaggle/working/
import os; os.chdir('/kaggle/working')
!ls
```

### Cell 2 — chọn fold & verify

```python
FOLD = 1
INSILICO = f'/kaggle/input/noc-inc22-fold{FOLD}/data_insilico_w'

import json, numpy as np
m = json.load(open(f'{INSILICO}/meta_set.json'))
f = json.load(open(f'{INSILICO}/fold_info.json'))
cid = np.load(f'{INSILICO}/combo_id_test.npy')
print('fold        :', f['fold'])
print('unknown     :', m['unknown_donors'])
print('sizes       :', m['split_sizes'])
print('test combos :', len(np.unique(cid[cid >= 0])), '| test mixtures:', int((cid >= 0).sum()))
assert f['fold'] == FOLD, 'dataset không khớp fold đang định chạy!'
assert m['split_policy']['multi_person_combos']['val'] == {}, 'dataset build bằng code cũ'
```

### Cell 3 — train

```python
!INSILICO_W=/kaggle/input/noc-inc22-fold1/data_insilico_w \
    python kaggle_run_increment1.py --seed 42 --out_subdir inc22_fixed_aslot_fold1 --noc_arm 0
```

Ba arm chạy trên cùng một dataset, chỉ đổi `--noc_arm` và `--out_subdir`:

```python
for arm in (1, 2, 3):
    !INSILICO_W=/kaggle/input/noc-inc22-fold{FOLD}/data_insilico_w \
        python kaggle_run_increment1.py --seed 42 --out_subdir arm{arm}_fold{FOLD} --noc_arm {arm}
```

Đổi `fold1` ở **cả 2 chỗ**. Không cần `STR_FOLD` trên Kaggle — fold đã đóng băng trong dataset.
Notebook restart mà `data_w_inc22/` còn thì thêm `--skip-prep`.

### Cell 4 — thu kết quả

```python
import json
r = f'/kaggle/working/results/inc22_fixed_aslot_fold{FOLD}_seed42'
print(json.dumps(json.load(open(f'{r}/metrics.json')), indent=2)[:2000])
!ls -la {r}
```

Kết quả: `metrics.json`, `best_model.pt`, `y_test_pred.npy`, `y_test_true.npy`.

Kiểm tra: `n_decode_groups` = (số combo test) + 1, và `decode` chứa `leave-one-combo-out fit`.
Nếu thấy `[legacy: no combo_id_test]` → dataset thiếu `combo_id_test.npy`, phải upload lại.

Lưu về local:

```
results_10fold/fold0/  fold1/  …  fold9/
```

---

## 6. Lưu ý

**Gộp 10 fold.** Số combo mỗi NOC khác nhau giữa các fold → mỗi ô per-NOC phải kèm `n`, đừng lấy
trung bình trần của 10 con số.

**"Novel combo" mạnh yếu khác nhau theo NOC.** Với 50000 mixture in-silico và `noc_weights=1,1.5,2.5,2`,
in-silico phủ ~99.9% không gian combo NOC2, ~53% NOC3, ~11% NOC4, ~1.2% NOC5. Claim novel-combo chỉ thực
sự mạnh ở NOC4/NOC5.

**Chạy trên Mac.** `train_set_transformer.py` tự chọn CUDA → MPS → CPU. Ép thiết bị bằng `STR_DEVICE`:

```bash
STR_DEVICE=cpu STR_DATA_DIR=code/data_w_inc22 python code/train_set_transformer.py --seed 42 --out_subdir smoke
```

MPS chậm hơn T4 nhiều — chỉ dùng smoke-test.

**Dung lượng `/kaggle/working`.** `data_w_inc22/` sau enrich ~1.5–2 GB. Xóa trước khi "Save Version".
