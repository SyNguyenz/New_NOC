# Pipeline NoC + ID — trạng thái, kết quả và cách chạy

Cập nhật 2026-09-22. CODE hiện tại **`7c16d6ddc9`** (md5 của `train_set_transformer.py` + `noc_clone.py` +
`noc_lora.py` + `fold_report.py` + `models/set_transformer.py`). Mọi số dưới đây là **fold 1, test 2577 hồ sơ
thật, tách tổ hợp donor**, backbone fold 1 (fingerprint `fp(logits_cls) = 056326a272`) trừ khi ghi khác.
Lịch sử thí nghiệm chi tiết kèm nguồn: [failed_experiments.md](failed_experiments.md).

---

## 1. Một fold chạy như thế nào

```
dataset fold (fold_info.json, tokens_*, donor_geno ...)
  -> make_dev_split.py      tách DEV in-silico (chọn checkpoint)
  -> features/enrich.py     token 8 trường mỗi peak
  -> train_set_transformer.py
       Phase 1   train backbone trên in-silico của 45 donor đã biết (150 epoch, ~3,1 giờ)
                 hoặc --init_from <best_model.pt> để bỏ qua
       ID        logits_cls của backbone đóng băng -> phi-rerank (alpha fit bỏ-một-tổ-hợp)
       NoC       cách đếm chọn bằng --count, 5-fold tách tổ hợp trên test của fold
       decode    top-k của bảng xếp hạng ID, k = số người đếm được
       báo cáo   results/<run>/report.json + reports/fold<F>_seed<S>.json
  -> outputs/fold<F>/       báo cáo, metrics, backbone + best_model.fold.json (stamp fold)
```

Phần ID **không bao giờ bị phần đếm chạm vào**: mọi cách đếm train trên bản sao hoặc trên LoRA tách riêng.
Log in `[ID GUARD] ... params_unchanged=True` mỗi lần chạy để kiểm chứng.

## 2. File trong `code/`

| file | dòng | vai trò |
|---|---|---|
| `kaggle_run_increment1.py` | ~175 | chạy một fold từ đầu đến cuối (runner) |
| `train_set_transformer.py` | ~820 | Phase 1, ID, gọi phần đếm, ghi báo cáo (file gốc, dọn tại chỗ) |
| `noc_clone.py` | ~190 | đếm bằng bản sao backbone; `InvariantCountHead` |
| `noc_lora.py` | ~220 | đếm bằng LoRA; head bất biến; inner val |
| `fold_report.py` | ~245 | fingerprint, stamp fold, nhãn open, bảng chẩn đoán, báo cáo JSON |
| `models/set_transformer.py` | ~625 | model; `enable_noc_lora(rank, scope)`, `noc_pass()` |
| `report_folds.py` | 185 | gộp báo cáo nhiều fold |
| `compare_counts.py` | ~445 | script thí nghiệm so nhiều cách đếm (có RF) — sẽ chuyển vào `archive/` |

Code thí nghiệm cũ (trainer 4.651 dòng, model 930 dòng, notebook cũ...) nằm ở `archive/2026-09-15_truoc_clean/`.

## 2b. KIẾN TRÚC CHỐT (2026-09-29)

Một backbone duy nhất, **1.505.737 tham số**, đóng băng sau Phase 1. Hai đầu ra:

| đầu ra | đường đi | tham số được train sau Phase 1 |
|---|---|---|
| **ID** | `forward()` → `logits_cls` (45 donor) → phi-rerank → lấy top-k | **0** — `fp(logits_cls)` không đổi ở mọi lần chạy, `params_unchanged=True` |
| **NoC** | `noc_pass()` = chạy lại encoder + decoder với trọng số `W + B·A` (LoRA **rank 8**, chỉ 10 ma trận của decoder) → head bất biến đọc gate và xác suất **đã sắp xếp** → `k = round(Σ i·pᵢ)` | **25.915** = 24.592 LoRA + 1.323 head |

Ba điều dễ nói sai khi viết paper:

1. **Rank 8, KHÔNG phải 8-bit.** Không có lượng tử hoá; r=8 là chiều của tích `B·A`.
2. **5 adapter trong báo cáo là GIAO THỨC ĐÁNH GIÁ**, không phải mô hình triển khai. 5-fold tách tổ hợp
   để mỗi hàng được chấm bởi adapter chưa từng thấy tổ hợp của nó. Khi triển khai chỉ có **một** adapter,
   fit trên toàn bộ hỗn hợp có nhãn.
3. **Chi phí suy luận là 2 lượt forward.** Vì scope=decoder nên đầu ra encoder `H` của hai lượt giống hệt
   nhau; cache `H` lại thì phần đếm gần như miễn phí. Hiện chưa tối ưu chỗ này.

Điều kiện triển khai: đầu ID không cần nhãn thật nào; đầu NoC cần một ít hỗn hợp thật có nhãn (mỗi fold ta
dùng 17–22 tổ hợp). Nếu không có nhãn nào thì còn `logits_card` zero-shot làm phương án lui
(0.4256 / 0.6588 với dữ liệu cũ, 0.7644 với dữ liệu sinh mới).

Số để báo cáo (3 fold, dữ liệu cũ): strict **0.8956 ± 0.0156**, deepnoc **0.9258 ± 0.0087**
(deepNoC công bố 0.9097), ID EM@k ~0.92, oracle EM 0.953–0.971.

## 3. Các cách đếm NoC (`--count`)

| `--count` | là gì |
|---|---|
| `clone` | bản sao backbone fine-tune qua `logits_card = noc_head(gate)`; `noc_head` đọc 45 cột **theo thứ tự donor** |
| `clone_inv` | cùng bản sao, nhưng đếm bằng **head bất biến**: gate và xác suất 45 donor **đã sắp xếp** (12 lớn nhất + tổng + số ≥ 0.5 mỗi loại), scaler cố định. Không còn trục "ai là ai" để học thuộc |
| `lora` | LoRA rank 8 trên backbone đóng băng, head theo thứ tự donor |
| **`zero_shot`** (2026-09-30) | `logits_card` của Phase 1 — **không dùng nhãn hỗn hợp thật nào**. Trước đây chỉ in làm mốc; nay là một arm đầy đủ (ID EM@k, open, per-NOC, ngưỡng tin cậy). Fold 0 bộ của họ: **0.9102** so với lora 0.9238 |
| **`gate_cal`** (2026-09-30) | `sum(gate) → k` hiệu chỉnh bằng hồi quy logistic trên **DEV in-silico**. **Không dùng nhãn hỗn hợp thật nào.** Bỏ qua lượt deepnoc vì không fit trên test |
| **`lora_inv`** (mặc định 2026-09-24) | LoRA r8 ở decoder (encoder đóng băng) + head bất biến; **5 epoch**, 25.915 tham số. Fold 1: 0.8802 / 0.8864 với expect, so với clone_inv 0.8665. **Lưu ý:** ngân sách 5 epoch mới xác nhận trên 1 fold; ở 15 epoch arm này thua clone_inv tại fold 2 → chạy kèm clone_inv ở mọi fold |

Cờ đi kèm:

| cờ | mặc định | ý nghĩa |
|---|---|---|
| `--count_kfold` | 5 | số fold tách tổ hợp trên test |
| `--clone_full_ep`, `--clone_full_lr` | 15, 1e-5 | ngân sách giai đoạn FULL của clone |
| `--lora_scope` | `decoder` | `all` (33 ma trận, 99.264 tham số) · `encoder` (20 / 69.632) · `encoder_attn` (12 / 28.672) · `decoder` (10 / 24.592) |
| `--lora_epochs` | **5** | ngân sách của `lora_inv`; `0` = dùng `--lora_steps` |
| `--lora_steps` | 3000 | ngân sách của `lora` |
| `--inner_val` | tắt | `lora_inv`: dừng sớm theo ~1/5 số tổ hợp giữ lại khỏi phần train |
| `--posthoc_rf` | tắt | thêm RandomForest post-hoc trên hồ sơ xác suất 45 donor đã sắp xếp (`noc_posthoc.py`), fit trên **đúng các split** bộ đếm dùng; in rf riêng, đếm riêng, và ensemble 50/50, ở cả hai giao thức. Đây là **mô hình thứ hai**, chỉ để báo cáo |
| `--protocol` | `both` | `strict` = K-fold tách tổ hợp (kết quả chính) · `deepnoc` = chia y hệt deepNoC §2.6, lấy cách một hồ sơ 50/50, **không tách gì**, hai nửa dùng chung tổ hợp · `both` = in cả hai, hiệu số là phần thưởng của việc dùng lại tổ hợp |
| `--count_decode` | **`expect`** | `expect` = lấy kỳ vọng rồi làm tròn (giữ thứ tự NoC); mọi lần chạy in `[DECODE]` cả hai số |
| `--compare` | trống | chạy thêm các cách đếm khác trên cùng backbone, chỉ báo cáo (vd `--compare clone_inv`); notebook: `COMPARE = ['clone_inv']` |

Luôn in thêm hai mốc miễn phí: **zero-shot** (`logits_card` của Phase 1, chưa thấy hỗn hợp thật có nhãn) và
**readout tổng gate**.

LoRA gắn thẳng lên từng ma trận Linear (`W + B·A`): wq/wk/wv/wo và feed-forward của mỗi khối ISAB, q/k/v,
GRU, slot_ffn, gate_head của decoder, và input/geno/attr projection. Không đụng norm, bias, embedding,
`pma`, `reject_head`, `cls_head`, `noc_head`.

## 4. Kết quả đã đo

### 4.1 Thang tách tổ hợp (test fold 1, 2577 hồ sơ, không tổ hợp nào của test xuất hiện lúc train)

| cách đếm | bước/fold | macro F1 | F1 NOC4 | F1 NOC5 | EM@k (ID) | acc open* |
|---|---|---|---|---|---|---|
| zero-shot `logits_card` | 0 | 0.426 | .184 | .172 | — | — |
| readout tổng gate | 0 | 0.492 | .283 | .640 | — | — |
| **clone_inv** | ~490 | **0.8665** | **.722** | **.808** | **0.891** | 0.201 |
| clone | ~490 | 0.8031 | .552 | .699 | 0.846 | 0.485 |
| clone | 3000 | 0.6918 | .312 | .646 | 0.772 | 0.467 |
| lora_inv, scope all | 3000 | 0.7715 | .557 | .724 | 0.832 | — |
| lora_inv, scope encoder_attn | 3000 | 0.7581 | .461 | .690 | 0.821 | — |
| lora_inv, scope decoder | 3000 | 0.7570 | .492 | .682 | 0.820 | — |
| lora_inv, scope encoder | 3000 | 0.7246 | .373 | .629 | 0.803 | — |
| lora | 3000 | 0.5115 | .129 | .390 | 0.646 | 0.618 |
| LoRA học in-silico | 3000 | 0.4958 | .315 | .237 | 0.652 | 0.692 |
| LoRA in-silico rồi fine-tune thật | 3000 | 0.5789 | .188 | .542 | 0.696 | 0.589 |
| RF (đã bỏ khỏi pipeline) | — | 0.8817 | .780 | .815 | 0.907 | 0.203 |
| **lora_inv 5 epoch, decoder, expect (mặc định)** | ~160 | **0.8864** | **.769** | **.851** | **0.902** | 0.132 |
| *(3 fold: lora 0.8956±0.0156 strict, 0.9258±0.0087 deepnoc; clone 0.8804±0.0307 / 0.9598±0.0043; deepNoC 0.9097)* | | | | | | |
| lora_inv 15 epoch, scope decoder | ~490 | 0.8644 | .708 | .814 | 0.888 | 0.142 |
| lora_inv 15 epoch, scope encoder | ~490 | 0.8523 | .678 | .783 | 0.878 | 0.190 |
| lora_inv 15 epoch, scope encoder_attn | ~490 | 0.8452 | .660 | .786 | 0.877 | 0.197 |
| lora_inv 15 epoch, scope all | ~490 | 0.8248 | .624 | .752 | 0.864 | 0.167 |

\* acc open = độ chính xác trên 1.339 hồ sơ có ít nhất một người ngoài panel 45 donor.

Fold 2 (LoRA 3000 bước, head theo thứ tự donor): 0.609. ID fold 1 / fold 2: oracle EM 0.953 / 0.971,
reject AUROC ~0.9997.

### 4.2 Thang tổ hợp dùng chung (protocol của deepNoC)

5 seed, 25 s, cùng ngân sách fine-tune với deepNoC: **LoRA 0.9505 ± 0.003**, clone 0.943, adapter B 0.935,
adapter A 0.926; **deepNoC 0.9097**. deepNoC (mục 2.6, trang 11) lấy *mỗi hồ sơ thứ hai* để fine-tune,
không tách tổ hợp, không tách mẫu — nên số của họ cũng ở thang này, so sánh là cùng điều kiện. Paper của họ
không có câu nào nói con số đó là tổng quát hoá hay không.

## 5. Điều rút ra

1. **Bài toán là số tổ hợp, không phải kiến trúc.** Cả archive chỉ có 30 tổ hợp hỗn hợp thật; test fold 1
   chiếm 22, mỗi mức NoC chỉ 4–7 tổ hợp. Học trên vài tổ hợp thì dễ thuộc lòng.
2. **Các tổ hợp cùng mức NoC giống nhau** (độ lệch giữa tổ hợp chỉ 0,11–0,64 lần độ lệch trong tổ hợp),
   nên về nguyên tắc suy ra được. Chỗ bí là ranh giới 4/5: đếm allele chỉ tách được với AUC 0.505.
3. **Train càng mạnh trên ít tổ hợp càng học thuộc:** clone 490 → 3000 bước tụt 0.803 → 0.692; mọi lần fit
   đạt train F1 ~1.0.
4. **Chặn đường danh tính ở đầu vào bộ đếm là thứ có tác dụng lớn nhất:** clone 0.803 → clone_inv 0.867;
   LoRA 0.512 → lora_inv 0.772 (cùng 3000 bước).
5. **Vị trí gắn LoRA ảnh hưởng ít** (all / encoder_attn / decoder chênh ≤ 0.014, 1 seed).
6. **LoRA thấp hơn clone chủ yếu vì ngân sách** (3000 bước, lr 5e-4 so với ~490 bước, lr 1e-5). Công thức
   mới cho lora_inv cùng ngân sách với clone_inv để so công bằng.
7. **Vị trí gắn LoRA không quyết định.** Fold 1: decoder 0.864 > all 0.825; fold 2: decoder 0.827 < all
   0.853. Thứ tự đảo giữa hai fold, chênh giữa các scope (≤ 0.04) nhỏ hơn chênh giữa các fold. LoRA decoder
   có gap học thuộc nhỏ hơn thật, nhưng train F1 chỉ ~0.95–0.97 → **thiếu dung lượng**, không phải chống học
   thuộc. clone_inv thắng cả hai fold và tốt nhất ở open → giữ clone_inv làm mặc định (mục 10
   `failed_experiments.md`).

## 6. Chạy trên Kaggle

### 6.1 Chuẩn bị (mỗi lần code đổi)

1. Dataset code `noc-fold-code`: **New Version** → upload `kaggle_upload/noc-fold-code.zip`.
2. Trong notebook: Input → dấu ba chấm cạnh `noc-fold-code` → **Check for updates**. Kaggle ghim version lúc
   gắn; không cập nhật thì dòng assert hash sẽ báo `dataset code la CODE <cũ>, can <mới>`. Vẫn còn bản cũ
   thì Factory reset session.
3. Dữ liệu: `noc-inc22-folds` (fold 1–3 đã có). Fold 0/4/5/6: upload `kaggle_upload/noc-inc22-fold{F}.zip`.
   Fold 7–9 chưa được tạo.
4. Backbone: fold 1 có sẵn `noc_inc22_backbone` (không có stamp, nhưng fingerprint `056326a272` xác minh
   đúng fold 1). Fold khác: phải chạy Phase 1, hoặc gắn Output của lần chạy fold đó (chỉ còn nếu chạy bằng
   **Save Version**).

### 6.2 Notebook fold (báo cáo 10 fold)

`kaggle_upload/fold_notebooks/noc_fold{0..9}.ipynb`, CODE_HASH đã là `7c16d6ddc9`.
**Đã sửa 2026-09-22:** cell cuối từng bắt buộc `source == 'noc_lora'`, sẽ báo lỗi với cách đếm mặc định mới;
giờ nhận mọi nguồn `noc_*`. Notebook cũ đang mở trên Kaggle thì import lại, hoặc sửa tay hai chỗ:
`CODE_HASH = '7c16d6ddc9'` và `rep['noc']['source'].startswith('noc_')`.
Thời gian: ~3,3 giờ/fold (Phase 1 3,1 giờ + đếm ~5 phút); ~15 phút nếu có backbone của đúng fold.

### 6.3 Cell thí nghiệm (dùng backbone có sẵn, không cần Phase 1)

| file | chạy gì | thời gian |
|---|---|---|
| `kaggle_upload/cells/cell_lora_inv_cong_thuc_moi.py` | lora_inv 15 epoch, scope all / encoder_attn / all + inner val; so với mốc 0.8665 | ~15 phút |
| `kaggle_upload/cells/cell_so_sanh_clone_vs_clone_inv.py` | clone rồi clone_inv trên fold 1 | ~25 phút |
| **`kaggle_upload/cells/cell_bo_cua_ho_fold0.py`** | fold 0 với bộ `noc-insilico-w` THẬT của nhóm (50k, `--conditioned`), 4 cách đếm gồm `gate_cal` không nhãn | ~3,5 giờ |
| `kaggle_upload/cells/cell_datagen_moi_fold12.py` | fold 1+2 với dữ liệu sinh MỚI, so thẳng với mốc dữ liệu cũ trên cùng fold. Chờ nhóm sinh dữ liệu (`doc/yeu_cau_sinh_du_lieu_fold12.md`) | ~2,5 giờ |
| `kaggle_upload/cells/cell_cu_vs_moi_fold0.py` | fold 0, cùng code cùng test: dữ liệu sinh CŨ so với MỚI, mỗi bên zero-shot / gate / lora_inv / clone_inv × hai giao thức + ID. Mặc định chỉ chạy bên CŨ vì bên MỚI đã có số | ~3,5 giờ |
| `kaggle_upload/cells/cell_datagen_moi_fold0.py` | chạy pipeline chuẩn trên bộ dữ liệu sinh MỚI (fold 0): zero-shot / sum(gate) / lora_inv / clone_inv × hai giao thức. Trả lời: dữ liệu tốt rồi thì còn cần fine-tune không | ~3,5 giờ |
| `kaggle_upload/cells/cell_fold123.py` | fold 1+2+3 một lần chạy: cả hai cách đếm × hai giao thức × hai cách giải mã, RF post-hoc, cột open, trung bình ± độ lệch, so với deepNoC 0.9097. Lưu backbone ngay khi Phase 1 xong | ~7 giờ |
| `kaggle_upload/cells/cell_fold1_day_du.py` | fold 1, một lần chạy: lora_inv + clone_inv × strict + deepnoc × expect + argmax, kèm RF post-hoc và ensemble | ~15 phút |
| `kaggle_upload/cells/cell_hai_giao_thuc_fold1.py` | fold 1: lora_inv 5 ep và 15 ep + clone_inv, mỗi arm ra 4 số (strict/deepnoc × argmax/expect) | ~12 phút |
| `kaggle_upload/cells/cell_lora_budget_2fold.py` | fold 1: quét ngân sách fit thật (5/10/15/25/40 epoch + rank 16) + clone_inv. Có ứng viên vượt mốc mới chạy tiếp fold 2 (tự train Phase 1) để xem chênh lệch có giữ dấu | ~40 phút, hoặc ~4 giờ nếu sang chặng 2 |
| `kaggle_upload/cells/cell_lora_decoder_2fold.py` | fold 1+2: LoRA decoder (+ clone_inv cùng backbone) và LoRA all; kiểm 3 giả thuyết đặt trước H1 hoà / H2 thứ tự / H3 gap học thuộc | ~2 giờ |

### 6.4 Phải kiểm tra trong log

- dòng đầu `CODE <hash>` đúng với dataset vừa upload;
- `[ID GUARD] fp(logits_cls)=056326a272 ... params_unchanged=True` (fold 1);
- `[SPLIT AUDIT] ... SHARED=0 (clean)`;
- `[REPORT] checks: ... params_unchanged True | split-audit shared combos 0`;
- chạy lại `clone` phải ra đúng 0.8031 — nếu lệch thì có gì đó khác, phải tìm trước khi tin số mới.

### 6.5 Đóng gói lại sau khi sửa code

```
python kaggle_upload/tools/build_fold_kits.py
```

Sinh lại `noc-fold-code.zip` và 10 notebook với CODE_HASH mới.

## 7. Hạn chế phải công bố trong paper

- **Hai thang đánh giá khác nhau:** số so với deepNoC ở thang tổ hợp dùng chung; số 10 fold ở thang tách
  tổ hợp — không đặt cạnh nhau.
- **clone_inv (và RF) gần như hỏng với người ngoài panel:** acc open 0.201, vì đọc xác suất 45 donor. clone
  0.485, LoRA 0.618. Casework có người lạ cần gác cổng bằng reject head (AUROC 0.9997).
- **Lịch 15 epoch @1e-5 của clone ban đầu được chọn khi nhìn số eval** — phải công bố, hoặc chạy lại với lịch
  đăng ký trước.
- Kết quả tách tổ hợp hiện chỉ fold 1 (và fold 2 cho LoRA cũ), 1 seed.
- Backbone fold 1 dùng cho các so sánh không có stamp fold (xác minh qua fingerprint).
- Input là bảng peak Filtered của PROVEDIt, deepNoC dùng EPG thô; panel 45 donor; 666 so với 675 hồ sơ hỗn
  hợp 25 s.

## 8. Việc còn lại

1. ~~lora_inv cùng ngân sách~~: đã chạy — scope decoder 0.8644, gần hoà clone_inv 0.8665 với ít tham số hơn
   ~58 lần. **Đã chọn làm mặc định (quyết định của nhóm, 2026-09-22).**
2. ~~Xác nhận trên fold 2~~: **đã chạy 2026-09-23 — decoder thua clone_inv 0.027 ở fold 2, mặc định đã trả
   về `clone_inv`.** Backbone fold 2 mới (có stamp) nằm trong output của lần chạy đó, nên fold 2 không cần
   train Phase 1 lại.
3. Chốt `--count_decode`: fold 1 đã xong, `expect` thắng `argmax` ở cả 4 arm (+0.0036..+0.0074; mục 11–12
   `failed_experiments.md`). Chỉ còn chờ fold 2 — số này in ra miễn phí ở mọi lần chạy fold sau, không cần
   lần chạy riêng. Hai script đo tại chỗ: `work/probe_count_feats.py`,
   `work/probe_count_decode.py` (CPU, dùng `work/smoke_fold1/` + `kaggle_upload/backbone/best_model.pt`).
4. Chạy đủ 10 fold bằng notebook; gộp bằng `python code/report_folds.py <thư mục chứa fold*_seed42.json>`.
5. Tạo dữ liệu fold 7–9 (preprocess ở local).
6. Sau khi chốt: ghi kết quả vào `failed_experiments.md`, chuyển `compare_counts.py` vào `archive/`.
7. Cập nhật `code/README.md` và `code/RUN_10FOLD.md` (đang mô tả cách chạy cũ).
8. Commit; lưu ý `.gitignore` chặn `*.npz` nên `code/open_labels/donor_map.npz` cần `git add -f`.

## 9. Nơi lưu trữ

| gì | ở đâu |
|---|---|
| lịch sử thí nghiệm có nguồn | `doc/failed_experiments.md` |
| tóm tắt deepNoC | `doc/deepNoC.md` |
| code thí nghiệm cũ | `archive/2026-09-15_truoc_clean/` |
| bundle tái lập so sánh deepNoC | `kaggle_upload/noc-bundle-3a4eab6847-archive.zip` |
| kit Kaggle, notebook, cell, script đóng gói | `kaggle_upload/` (`tools/`, `cells/`, `fold_notebooks/`) |
| quy tắc làm thí nghiệm (skill) | `.claude/skills/noc-thi-nghiem/SKILL.md` |
| code thí nghiệm đã gỡ | `archive/<ngày>_<tên>/` — chuyển vào, không xoá |
| tập smoke để test trên máy | `work/smoke_fold1/` (đã nằm trong `.gitignore`) |
