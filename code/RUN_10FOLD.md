# Chạy 10-fold unknown-donor

Quy trình: **preprocess ở local** (CPU) → **upload dataset lên Kaggle** → **train trên GPU Kaggle**.
Mọi lệnh chạy từ **repo root** (thư mục chứa `code/`).

---

## 1. Bảng 10 fold

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

`n_classes` = 45 ở mọi fold.

## 2. Env flag

| flag | mặc định | tác dụng |
|---|---|---|
| `STR_FOLD` | `0` | fold cho `code/preprocess.py` (`preprocess.sh` tự đặt) |
| `STR_KIT` | `3500_GF29cycles` | kit: `3500_GF29cycles` hoặc `3500_F6C29cycles_hlfrxn` (xem `code/kit.py`); đặt giống nhau cho preprocess và train |
| `STR_DEVICE` | auto | ép `cuda` / `mps` / `cpu` |
| `STR_BATCH` | `256` | batch train; card 4 GB dùng `128` |
| `STR_EPOCHS` / `STR_LR` | `150` / `6e-4` | số epoch / learning rate |
| `STR_INIT_FROM` | — | checkpoint để fine-tune |
| `STR_EVAL_ONLY` | `0` | `1` = chỉ decode + đánh giá checkpoint `STR_INIT_FROM` |

---

## 3. Preprocess local: `preprocess.sh` (ở repo root)

Cần Python ≥ 3.10 + `numpy`, `pandas`, `scikit-learn`, `openpyxl`. Không cần GPU/torch.

```bash
bash preprocess.sh                  # in hướng dẫn
bash preprocess.sh 1                # fold 1 -> data/fold1
bash preprocess.sh 0 2 5            # các fold 0, 2, 5
bash preprocess.sh 3-6              # fold 3 đến 6
bash preprocess.sh all              # fold 0 đến 9
bash preprocess.sh --override 0     # build lại fold 0 dù data/fold0 đã có
```

- Fold đã có `data/foldK` thì bỏ qua, trừ khi có `--override`.
- Kết quả mỗi fold: `data/foldK/` (self-contained, gồm cả `gen_law.npy`, `donor_geno*`, `combo_id_*`, `fold_info.json`).
- Log: `data/logs/foldK.log`. Thời gian: ~30 phút / fold.
- `code/data/` và `code/data_insilico_w/` là thư mục làm việc, bị xóa và dựng lại ở mỗi fold.

### 3.1. Sanity-check

```bash
K=1; python -c "import json;d='data/fold$K';m=json.load(open(d+'/meta_set.json'));f=json.load(open(d+'/fold_info.json'));mp=m['split_policy']['multi_person_combos'];print('fold',f['fold']);print('unknown',m['unknown_donors']);print('n_classes',m['n_classes']);print('sizes',m['split_sizes']);print('n_test_combos',{k:len(v) for k,v in mp['test'].items()});print('val combos (must be empty)',mp['val'])"
```

- `fold`, `unknown` khớp bảng mục 1; `n_classes` = 45
- `n_test_combos` khớp cột "combo thật N2/N3/N4/N5"
- `multi_person_combos["val"]` phải là `{}`
- `data/logs/foldK.log` có dòng `excluding real combos from in-silico train`

---

## 4. Upload lên Kaggle

### 4.1. Dataset CODE (một lần cho cả 10 fold)

`kaggle_upload/code/`: thư mục `code/`.

```
code/
  kaggle_run_increment1.py  train_set_transformer.py  decode_layer.py  gen_law.py  phi_rerank.py
  preprocess.py  make_insilico.py
  build_donor_geno.py  build_real_attr.py  extract_phi_condition.py  kit.py  kit_raw.py
  calibrate.py  calibrate_*.py
  models/__init__.py  models/ordinal.py  models/set_transformer.py
  features/enrich.py  synth/extract_genotypes.py
```

- Slug: `noc-inc22-code` → `/kaggle/input/noc-inc22-code/code/`
- Sửa code thì tạo **New Version** của dataset này.

### 4.2. Dataset DATA (một dataset mỗi fold)

`data/foldK/` = nội dung `code/data_insilico_w/` của fold K (~520 MB).

- Slug: `noc-inc22-fold0` … `noc-inc22-fold9`
- `INSILICO_W` phải trỏ vào thư mục chứa `meta_set.json`: upload nội dung `foldK` thì là `/kaggle/input/noc-inc22-foldK`;
  dataset chứa thư mục `foldK` thì là `/kaggle/input/noc-inc22-foldK/foldK`. Kiểm tra bằng `!ls /kaggle/input/<slug>`.

---

## 5. Notebook Kaggle (GPU)

Accelerator: GPU T4 x2 / P100. Internet: off.

### Cell 1 — copy code

```python
!cp -r /kaggle/input/noc-inc22-code/code/* /kaggle/working/
import os; os.chdir('/kaggle/working')
!ls
```

### Cell 2 — chọn fold & verify

```python
FOLD = 1
INSILICO = f'/kaggle/input/noc-inc22-fold{FOLD}'          # thư mục chứa meta_set.json

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
!INSILICO_W={INSILICO} python kaggle_run_increment1.py --seed 42 --out_subdir inc22_fixed_aslot_fold{FOLD}
```

- Không cần `STR_FOLD` trên Kaggle.
- Notebook restart mà `data_w_inc22/` còn thì thêm `--skip-prep`.
- ~100 s/epoch, 150 epoch ≈ 4–4.5 giờ + vài phút decode.

### Cell 4 — thu kết quả

```python
import json
r = f'/kaggle/working/results/inc22_fixed_aslot_fold{FOLD}_seed42'
mt = json.load(open(f'{r}/metrics.json'))
print('decode :', mt['decode'])
print('ID     :', mt['macro_id_mix'], mt['per_noc_at_pred_k'])
print('count  :', mt['macro_count_mix'], mt['per_noc_count'])
!ls -la {r}
```

Kết quả: `metrics.json`, `best_model.pt`, `last_model.pt`, `y_test_pred.npy`, `y_test_true.npy`.
Chỉ số cuối cùng: `macro_id_mix` / `macro_count_mix` (macro NOC2–5).

Lưu về local: `results_10fold/fold0/ … fold9/`.

---

## 6. Lưu ý

- Gộp 10 fold: mỗi ô per-NOC phải kèm `n`, không lấy trung bình trần của 10 con số.
- `/kaggle/working`: `data_w_inc22/` sau enrich ~1.5–2 GB, xóa trước khi "Save Version".
- Smoke-test local (CPU/Mac):

```bash
STR_DEVICE=cpu STR_EPOCHS=1 STR_DATA_DIR=code/data_w_inc22 python code/train_set_transformer.py --seed 42 --out_subdir smoke
```
