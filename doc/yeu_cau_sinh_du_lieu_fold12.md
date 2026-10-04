# Yêu cầu sinh dữ liệu fold 1 và fold 2 bằng generator mới

Gửi nhóm, 2026-09-29. Mục đích: so generator mới với generator cũ trên **đúng hai fold ta đã có mốc đầy
đủ**, thay vì so chéo fold như bộ `data_ft_att` (fold 0).

## Cần sinh

Hai bộ, **cùng generator và cùng tham số đã dùng cho `data_ft_att`**, chỉ đổi fold:

| | fold 1 | fold 2 |
|---|---|---|
| `fold_info.json` → `fold` | 1 | 2 |
| donor ẩn (`unknown_donors`) | 18, 28, 46, 47, 48 | 19, 24, 32, 38, 41 |
| donor biết (panel) | 45 donor còn lại | 45 donor còn lại |
| số hỗn hợp sinh | **12.000** (giữ nguyên như fold 0 để so được) | 12.000 |

Cấu trúc file giữ y như `data_ft_att`: `tokens8_*`, `mask_*`, `y_*_set`, `noc_*`, `attr_*`, `phi_*`,
`attrfrac_train`/`attrcol_train` (cell dùng file này để nhận diện), `donor_geno*`, `combo_id_test`,
`tokens_open`, `meta_set.json`, `fold_info.json`, và split `dev` đã cắt sẵn.

## Ba điều kiện bắt buộc

1. **Hỗn hợp chỉ chồng từ hồ sơ đơn nguồn của PHÍA TRAIN**, không dùng hàng đơn nguồn của `val`/`test`.
2. **Không tổ hợp hỗn hợp nào của train trùng với test** (ở fold 0 ta đo được SHARED = 0, giữ nguyên vậy).
3. **Donor ẩn của fold không xuất hiện ở train/val/test**, chỉ ở `open`.

Ta sẽ tự kiểm lại cả ba bằng `work/probe_sim2real.py` và `code/open_labels/donor_map.npz` trước khi chạy.

## Xin thêm, nếu tiện

- **File generator** đã sinh `data_ft_att` (file ghi ra `attrfrac_train.npy`). Có nó thì ta tự sinh cho 7
  fold còn lại, không phải chờ.
- Nó khác `code/make_insilico.py` ở điểm nào? Đo được: bộ cũ gần như không có đuôi peak mờ
  (`logh_p10` 24 RFU so với 9 RFU của hồ sơ thật), bộ mới đã sửa. Muốn biết chỗ sửa nằm ở đâu.
- **Checkpoint dùng để warm start** trong lần chạy 2 epoch của nhóm. Nếu nó train trên bộ không theo fold
  thì nó đã thấy donor ẩn của fold 0, và con số 0.9552 không dùng được.

## Chạy gì khi có dữ liệu

`kaggle_upload/cells/cell_datagen_moi_fold12.py` — khoảng 2,5 giờ cho cả hai fold (Phase 1 chỉ ~1 giờ/fold
vì 15k hàng thay vì 47k). Cell tự so với mốc dữ liệu cũ đã lưu sẵn:

| | fold 1 (cũ) | fold 2 (cũ) |
|---|---|---|
| zero-shot | 0.4256 | 0.6588 |
| lora_inv strict | 0.8864 | 0.8869 |
| lora_inv deepnoc | 0.9158 | 0.9300 |
| oracle EM | 0.9534 | 0.9710 |
| EM@k | 0.9018 | 0.9241 |
