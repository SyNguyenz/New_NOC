# Đếm NoC mà không neo vào ID — chốt hiện trạng 2026-10-01

Note để đọc lại. Mọi số trong đây đều từ log đã chạy, không có số suy đoán. Chỗ nào chưa đo thì
ghi rõ **CHƯA ĐO**.

Nền tham chiếu cho toàn bộ note: **fold 0**, dữ liệu 50k `--conditioned` của synznguyen, backbone
`fp(logits_cls) = 6b61691655`, 2.458 hàng test thật, `params_unchanged=True`, `shared combos 0`.

---

## 1. Câu hỏi gốc: sao `sum(gate)` tệ hơn `logits_card` nhiều thế

### Khác nhau về code — hai dòng trong `AdaptiveSlotDecoder` (set_transformer.py:314-325)

```python
gate        = torch.sigmoid(gate_logit)      # (B, 45) — mỗi slot một số
logits_card = self.noc_head(gate)            # Linear(45, 5) = 225 trọng số + 5 bias
```

| | `round(sum gate)` | `logits_card` |
|---|---|---|
| đọc gì | **một** số (tổng) | **cả 45** số |
| trọng số mỗi slot | cố định 1.0 | học được, riêng từng slot |
| ngưỡng | **cố định** 1.5/2.5/3.5/4.5 | **5 bias học được** |
| tham số | 0 | 230 |

### Số đo

| cách đọc | macro F1 | mix macro F1 | acc | gọi thiếu | gọi thừa |
|---|---|---|---|---|---|
| `logits_card`, expect | 0.9014 | **0.8773** | 0.9349 | 108 | 47 |
| `logits_card`, argmax | **0.9102** | chưa lưu lần đó | 0.9406 | — | — |
| `round(sum gate)` | 0.7158 | 0.6937 | 0.7339 | 142 | 148 |
| *nhánh synznguyen, `sum gate`* | — | *0.8632* | *0.9288* | *37* | *136* |

**`corr(sum gate, NOC) = 0.9535` trong chính lần chạy đó.** Tương quan 0.9535 mà macro F1 0.7158
→ **thông tin có đủ trong gate, chỗ mất là `round()`**.

### Chỗ mất, bằng số học

`sum()` không phân biệt được:

| | tổng | `round()` | thật |
|---|---|---|---|
| 1 slot 0.98 + 44 slot rò rỉ 0.02 | 1.86 | 2 | **1 người** ❌ |
| 2 slot 0.93 | 1.86 | 2 | 2 người ✓ |

45 slot × rò rỉ 0.02 = **0.9**, gần trọn một người góp, chỉ từ nhiễu. Đúng kiểu hỏng đo được:

| lớp | n | hướng sai chính | recall gate | recall logits_card |
|---|---|---|---|---|
| NOC1 | 1125 | **355 → NOC2** | **0.676** | **0.996** |
| NOC2 | 334 | 19 → NOC3 | 0.925 (precision chỉ **0.446**) | 0.940 |
| NOC3 | 385 | 65 → NOC4 | 0.751 | 0.935 |
| NOC4 | 242 | 61 → NOC5 | 0.678 | 0.934 |
| NOC5 | 372 | **82 → NOC4** | 0.755 | 0.747 |

NOC1 là 46% tập test; gate mất 355 hàng chỉ ở đó. Gọi **thừa** ở đáy, gọi **thiếu** ở đỉnh →
nén về giữa kèm sàn bị đẩy lên, **không** phải một độ lệch cố định.

### Lý do thứ hai: đầu nào được dạy

```python
loss = ... + beta * loss_noc                                   # beta_card = 0.30 -> logits_card
if w_gate > 0:
    loss = loss + w_gate * F.smooth_l1_loss(gate.sum(1), noc)  # w_gate    = 0.05 -> sum(gate)
```

Chênh **6 lần**, và khác bản chất: CE 5 lớp học ranh giới; SmoothL1 trên một tổng thì thiếu xác định
(`∂Σgate/∂gate_i = 1` cho mọi slot — gradient không bao giờ nói slot nào nên tắt).

### Nhưng KHÔNG phải "đọc tổng thì dở"

Nhánh synznguyen dồn **toàn bộ** ngân sách đếm vào `sum(gate)` **cộng** warm start 2 epoch, vẫn chỉ
được 0.8632 so với 0.8773 của `logits_card`. So sòng phẳng, đọc vector 45 chiều thắng **+0.0141**.
Phân bổ gradient giải thích phần lớn khoảng cách 0.184, không phải tất cả.

---

## 2. Vấn đề thật: `logits_card` neo vào ID

Lúc suy luận `logits_card` **không** đọc `logits_cls`. Neo nằm ở hai chỗ khác:

1. **Trục của `gate` là trục donor.** Slot `i` khởi tạo từ bộ gen donor `i` (CoSA), nên
   `Linear(45,5)` đọc theo thứ tự donor → học được "cột 17 và 29 sáng → 4 người", tức **nhớ tổ hợp**.
2. **Nhãn dạy nó suy từ ID** (train_set_transformer.py:369):
   ```python
   card_tgt = cardinality_target(torch.sigmoid(out["logits_cls"]).detach(), y, card_lam)
   ```
   Nó học "đầu ID tin là có mấy người", **không** học NOC thật.

### Bằng chứng: 1.366 hàng có donor NGOÀI panel, nơi ID buộc phải sai

| | tập đóng | **ngoài panel** |
|---|---|---|
| `lora_inv` (neo theo ID) | **0.9238** | **0.198** |
| `gate_cal` (bất biến danh tính) | 0.6854 | **0.630** |

**Đảo chiều hoàn toàn.** Cách đếm neo theo ID sụp đúng ở nơi ID sụp. Đây là lý do kỹ thuật để
chuyển sang thứ tự **NoC trước → ID sau**.

---

## 3. PHÁT HIỆN: đầu đếm không neo ID đã có sẵn và đã được train

`noc_head_v2` (CORN ordinal head) được train **trong mọi lần chạy** từ đầu tới giờ:

| | |
|---|---|
| nhãn dạy | `noc.clamp(1,5)` — **NOC THẬT**, không qua `logits_cls` |
| đặc trưng | profile xác suất **đã sắp xếp** (mất trục donor) + **19 đặc trưng MAC vật lý** + `slot_mass` đã sắp xếp |
| MAC gồm | số allele mỗi locus ở ngưỡng RFU 0/50/150/500 + thống kê chiều cao log |
| nguồn MAC | PACE (Marciano & Adelman 2017) và **deepNoC 2024** — chính bộ đặc trưng của họ |
| đầu vào | **detach hết**, không làm nhiễu slot hay ranking |
| trọng số loss | `beta_card` = 0.3, **ngang** `logits_card` |
| giải mã | `corn_probs()` đã viết sẵn ở `models/ordinal.py` |
| **ai gọi `corn_probs`** | **KHÔNG AI.** Trainer chỉ import `corn_loss` |

Tức đầu này được dạy đủ 150 epoch mỗi lần chạy rồi bị bỏ không đọc.

**Vì sao từng bị bỏ:** sập N3/N4 (~0.21/0.17). Nhưng phép đo đó có **TRƯỚC** dữ liệu
`--conditioned`. MAC phụ thuộc nặng vào **chiều cao đỉnh**, đúng thứ generator cũ làm sai nhất
(thiếu đuôi đỉnh mờ: 24 RFU so với 9 RFU thật).

**Dự báo đặt trước, không chỉnh sau khi xem số:**
- **H1** — CORN trên dữ liệu `--conditioned` vượt `round(sum gate)` 0.7158
- **H2** — CORN trên nhánh ngoài panel vượt `lora_inv` 0.198

Không đoán mức. H1 sai → đóng hướng CORN, ghi lại. H2 sai → giả thuyết "bất biến danh tính thì
giữ được ngoài panel" chưa vững.

---

## 4. Hình dạng pipeline đề xuất: NoC → đã biết/chưa → ID

1. **NoC** từ một đầu **bất biến danh tính**. Ứng viên: CORN (miễn phí, đo được ngay) hoặc
   `sum(gate)` + ngưỡng học trên hàng thật.
2. **reject head** quyết trong/ngoài panel. **Đã gần như xong: AUROC = 1.0000** trên fold 0.
3. **ID** chỉ chạy cho hàng trong panel, lấy top-k với k từ bước 1.

Mốc ID hiện tại để so: oracle EM **0.9813**, EM@k 0.9451 (`lora_inv`) / 0.9317 (`zero_shot`),
donor F1@k 0.9841. Không dùng nhãn thật nào thì EM@k **0.9251** (alpha khớp trên DEV in-silico).

---

## 5. Cần chạy lại những gì

> **BACKBONE FOLD 0 ĐÃ MẤT** (xác nhận 2026-10-01). Đây là **lần thứ ba**: fold 2 ngày 23/09,
> fold 0 ngày 30/09, và giờ là backbone của bộ 50k. Nguyên nhân mỗi lần đều giống nhau — cell lưu
> backbone sau khi cả lần chạy xong, hoặc phiên bị ngắt trước khi Save Version.
> **Cell `cell_neo_id_12k.py` đã sửa: lưu backbone ngay tại dòng `Training done in`.**
> Nhớ **Save Version → Save & Run All (Commit)**, không chạy interactive.

| việc | cần Phase 1? | thời gian | trạng thái code |
|---|---|---|---|
| **arm `corn`** (giải mã `logits_count_v2`) | **CÓ** (vì mất backbone) | đi kèm mọi lần chạy, **miễn phí** | **ĐÃ VIẾT** |
| **lưu `S_te`** (mảng tổng gate từng hàng) | kèm theo | 0 | **ĐÃ VIẾT** |
| `--card_feats sorted` (cắt trục donor) | **CÓ** | ~1,1 giờ trên 12k | **ĐÃ VIẾT** |
| `--card_target true` (nhãn là NOC thật) | **CÓ** | ~1,1 giờ trên 12k | **ĐÃ VIẾT** |
| trần ngưỡng của `sum(gate)` | KHÔNG — tính từ `S_te` đã lưu | vài giây CPU | chưa có |
| arm `gate_kf` (gate + 4 ngưỡng, k-fold tách tổ hợp) | KHÔNG | ~1 phút | chưa có |
| 15k so với 50k (`--max_mix`) | **CÓ** | ~1 giờ | chưa viết |
| `beta_card=0, w_gate=0.3` (tái lập phân bổ của họ) | **CÓ** | ~3,2 giờ | cờ đã có, chờ quyết |

### KHÔNG cần chạy lại

Phần dọn code đã được chứng minh **không đổi một con số nào**: smoke CPU cùng seed 42 trên
`work/smoke_fold1`, 4 arm × 2 giao thức, chạy qua từng bước dọn:

```
fp(logits_cls)      b6dc7969c8  ==  b6dc7969c8   (cả 4 lần)
params_unchanged          True  ==  True
metrics.json so với trước khi dọn: 1 khác biệt duy nhất
   -> count_decode_per_source, trường mới cố ý thêm
```

---

## 5b. `zero_shot` nghĩa là gì — nói cho chặt

Pipeline hai chặng:
- **Chặng 1** train backbone 150 epoch trên **50.000 hỗn hợp sinh** + **5.247 đơn nguồn thật**.
  Cả hai loại nhãn đều miễn phí: hỗn hợp sinh thì ta biết đáp án, đơn nguồn thì NOC = 1.
- **Chặng 2** (`lora_inv`, `clone_inv`) khớp adapter **trên hỗn hợp thật của tập test**, dùng
  **nhãn NOC thật**, theo k-fold tách tổ hợp.

`zero_shot` = **bỏ hẳn chặng 2**, đọc thẳng `logits_card` từ backbone đóng băng.

Nên "zero-shot" ở đây nghĩa là **không nhãn hỗn hợp thật**, KHÔNG phải "không train" — `logits_card`
có được dạy đếm, bằng `beta_card=0.3` trên dữ liệu sinh. Thứ nó chưa bao giờ thấy là **một hỗn hợp
thật kèm số người thật**. Kiểm toán mỗi lần chạy chứng minh:
`in-silico train combos=39653  real test combos=20  SHARED=0`.

Trục này đáng đo vì một lab mới lấy được hồ sơ đơn nguồn (việc thường ngày) và dữ liệu sinh (chạy
software), nhưng **rất khó** có thư viện hỗn hợp thật đã biết số người — muốn có nhãn thì phải biết
trước đáp án, mà đó đúng là thứ cần dự đoán.

| | trước fine-tune | sau fine-tune | khoảng cách |
|---|---|---|---|
| **deepNoC** (họ tự báo) | 40–60% | ~90% | **30–50 điểm** |
| ta, bộ cũ 50k | 0.6153 | 0.9026 | 28,7 điểm |
| **ta, bộ 50k `--conditioned`** | **0.9102** | 0.9238 | **1,4 điểm** |

**Lưu ý khi viết bài:** trong tài liệu ML, "zero-shot" thường nghĩa là model làm việc nó *chưa được
train*. Ở đây nó đã được train để đếm, chỉ trên dữ liệu sinh. Nên gọi **"không có giám sát từ hỗn hợp
thật"** / **"sim-to-real không thích ứng"** / **"trước fine-tune"** (cách deepNoC gọi). Tên arm trong
code giữ `zero_shot` thì được.

**Và phần ID thì chưa label-free:** alpha của `phi_rerank` khớp leave-one-combination-out trên nhãn
thật của test. Bản sạch tuyệt đối dùng alpha khớp trên DEV in-silico: NoC 0.9102, **EM@k 0.9251**
(so với 0.9317).

---

## 5c. Thí nghiệm đang chờ chạy: cắt dây neo, trên 12k

Cell: `kaggle_upload/cells/cell_neo_id_12k.py`, CODE `f14264174e`, hai lần chạy ~2,2 giờ.

| | `card_feats` | `card_target` | nghĩa |
|---|---|---|---|
| **R1** | `donor` | `id` | hiện tại (mặc định) |
| **R2** | `sorted` | `true` | cắt **cả hai** dây neo |

Mỗi lần chạy kèm arm `corn` **miễn phí**, cộng `gate` để đối chứng.

**Giả thuyết đặt trước, ghi trước khi chạy:**

| | nội dung |
|---|---|
| **H1** (có thật bị neo) | trong R1: sụt closed→open của `logits_card` **>** sụt của `corn` |
| **H2** (cắt được) | R2 `logits_card` open acc ≥ R1 open acc **+ 0.05** |
| **H3** (giá phải trả) | R2 oracle EM ≥ R1 oracle EM **− 0.01** |
| **H4** (corn có dùng được) | trong R1: `corn` macro F1 **>** `gate` macro F1 |

Luật đọc kết quả, chốt trước:
- H1 + H2 đúng → cắt neo là hướng thật, làm lại trên 50k
- H1 đúng, H2 sai → neo có thật nhưng **hai công tắc này không phải cách cắt**. Đổi sang `corn`
- **H1 sai → giả thuyết neo KHÔNG được số liệu ủng hộ. Đóng hướng và ghi lại**
- H3 sai → fix làm hỏng ID, phải báo đánh đổi

**Đã biết trước và không phải lỗi:** `fp(logits_cls)` của R2 **chắc chắn khác** R1. `gate` không
detach nên sửa cách dạy phần đếm lan vào encoder qua `gate_logit` → đổi cả ID. Đó chính là lý do
phải có H3.

**Phải kiểm trong log:** dòng `enrich-after-feas keeps`. **0.779** = bộ cũ, **0.788** = `ft_att` 12k
(KHÔNG có `--conditioned`), **0.811** = bộ `--conditioned`. Nếu bộ 12k mới của Nguyên ra ~0.788 thì
nó cùng build với `ft_att`, và số tuyệt đối sẽ không chuyển sang được bộ 50k.

**12k có đủ để kết luận không?** Đủ cho phép so A/B, vì R1 và R2 dùng **cùng** dữ liệu nên chất
lượng generator bị triệt tiêu. Chỉ **số tuyệt đối** là không chuyển trực tiếp sang bộ 50k.

---

## 6. Trạng thái code sau khi dọn (2026-10-01)

**CODE: `7c16d6ddc9` → `08f718edb3`** (dọn) **→ `f14264174e`** (thêm `corn`, `--card_feats`, `--card_target`, lưu `S_te`). Danh sách 5 file. Mọi cell còn chốt hash cũ sẽ assert fail —
đúng hành vi mong muốn với cell lịch sử, nhưng cell đang dùng thì phải bump.

| | trước | sau |
|---|---|---|
| `train_set_transformer.py` | 1.039 | 965 |
| trong đó `def train()` | 607 | **459** |
| file trong CODE hash | 7 | **5** |
| arm đếm | 7 | **5** |

**Đã bỏ** (code nằm ở `archive/2026-10-01_truoc_clean/`, không xoá):
- `clone` / `clone_inv` + `clone_warm_ep`/`clone_full_ep`/`clone_full_lr` + 2 cờ CLI + `noc_clone.py`
- `posthoc_rf` + `noc_posthoc.py` — việc này **ghi đè** quyết định "giữ code sau cờ" ở mục 14
- `inner_val` — mặc định tắt, chưa từng dùng
- `noc_v2_weight` — luôn `None`, luôn phân giải thành `beta_card`

**Suýt hỏng:** `noc_lora.py` import `InvariantCountHead` **từ** `noc_clone.py`, tức arm đang ship
`lora_inv` phụ thuộc file sắp bỏ. Đã chuyển class sang `noc_lora.py` rồi chạy lại smoke **sau khi**
`noc_clone.py` đã rời `code/`.

**Đã tách:**

| tách ra | dòng | về đâu |
|---|---|---|
| vòng lặp Phase 1 | 112 | `run_phase1()` cùng file |
| chấm điểm một arm | 55 | `fold_report.score_count_arm()` |
| chấm điểm nhánh open | 15 | `fold_report.score_open_split()` |

**Đã sửa một lỗi báo cáo thật:** bảng per-NOC in `zero_shot` (baseline, argmax) và `noc_zero_shot`
(arm, expect) thành hai dòng tên gần giống nhau mà **không nhãn giải mã**. Trên fold 0 đó là 0.9102
và 0.9014 — đã có lần trích sai vì chuyện này. Giờ mỗi dòng mang tên giải mã (`zero_shot/argmax`,
`noc_zero_shot/expect`) và `metrics.json` có thêm `count_decode_per_source`.

**Comment đã gỡ:** comment `w_gate` ghi `round(sum gate) alone gives .838 macro-F1` — thực đo
**0.6937**, sai gần 0.15 và nằm ngay cạnh tham số. Đã gỡ toàn bộ nhật ký số liệu khỏi `CFG`
(5 khối, 26 dòng) và trỏ về `failed_experiments.md`.

---

## 7. Chỗ chưa biết, ghi rõ để không quên

- **Trần của `sum(gate)` + ngưỡng học được.** Với corr 0.9535, chưa biết lấy lại được bao nhiêu.
  `gate_cal` thất bại (0.6854 < 0.7158) nhưng nó khớp trên **DEV in-silico**, nơi sàn rò rỉ khác
  miền thật. Khớp trên hàng thật theo k-fold tách tổ hợp thì **chưa ai làm**.
- **15k so với 50k.** Điểm dữ liệu nhỏ duy nhất (`ft_att` 12k) **khác generator** → lượng và chất
  bị trộn. Đây đúng là confound đã làm kết luận mục 19 sai. Dữ liệu hiện tại: 55.247 hàng =
  50.000 hỗn hợp + 5.247 đơn nguồn thật; 39.653 tổ hợp hỗn hợp khác nhau trên 1.385.934 khả dĩ =
  **2,86% độ phủ**. Xuống 15k thì còn ~1,0%.
  Dự báo đặt trước: **NoC tụt, ID gần như không đổi** — vì qua 3 bộ dữ liệu, oracle EM chỉ dao động
  0.016 (0.9650 → 0.9813) còn zero-shot NoC dao động 0.295 (0.6153 → 0.9102), tức NoC nhạy hơn ~18 lần.
- **Vì sao gate của ta kém gate của họ trên CÙNG dữ liệu** (acc 0.7339 so với 0.9288). Trainer khác
  nhau; chưa truy ra hết.
- **Mục 7.2 trong `RUN_10FOLD.md` của họ** viết "Ep 1 phải có `attr=` quanh 0.3; nếu 3.3 thì nhãn
  lệch hàng". Log của chính họ có `attr=3.518`, của ta `attr=3.192`. Theo tiêu chí họ tự đặt thì
  **cả hai bên đều lệch**. Cần hỏi nhóm.
- **Checkpoint nào** cho lần warm start 2 epoch của họ. Chưa biết.

---

## 8. Mốc ngoài để so

deepNoC công bố **macro F1 0.9097**. `doc/deepNoC.md` đã dựng lại từ ma trận nhầm lẫn của họ, 15/15
giá trị khớp từng chữ số, và chỉ ra bảng của họ in **one-vs-rest** chứ không phải recall:

| | 1p | 2p | 3p | 4p | 5p | macro |
|---|---|---|---|---|---|---|
| họ in trong Table 3 (one-vs-rest) | 1.000 | 0.992 | 0.965 | 0.911 | 0.927 | 0.9591 |
| **recall chuẩn** | 1.000 | 1.000 | **0.873** | **0.882** | **0.782** | **0.9074** |

Cột F1 của họ thì đúng (F1 đối xứng theo precision/recall nên việc đảo nhãn hai cột không ảnh hưởng)
→ **0.9097 là mốc dùng được**. Khối in `recall` và `(one-vs-rest)` cạnh nhau trong trainer tồn tại
đúng để không bao giờ trích lẫn hai thước đo này.
