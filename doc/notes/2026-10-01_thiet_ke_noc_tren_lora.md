# Thiết kế đếm NoC — đặt toàn bộ trên LoRA, không chạm đường ID

Ngày 2026-10-01. CODE `ca3afecba8`.
Doc này là **thiết kế**. Số liệu và bằng chứng dẫn tới nó nằm ở
[2026-10-01_dem_khong_neo_vao_id.md](2026-10-01_dem_khong_neo_vao_id.md).

---

## 0. Ràng buộc gốc, quyết định mọi thứ còn lại

> ID đã rất tốt rồi, không được ảnh hưởng tới ID. Làm gì thì làm trên LoRA.

ID hiện tại: oracle EM **0.9813**, EM@k 0.9451, donor F1@k 0.9841, reject AUROC **1.0000**.

Nên **mọi thay đổi cho phần đếm phải nằm sau lớp LoRA**. Cái này kiểm được, không phải tin lời:

```
p.requires_grad = n.startswith("noc_lora.")      # noc_lora.py:37 — chỉ LoRA được train
[ID GUARD] fp(logits_cls)=... params_unchanged=True
```

`fp(logits_cls)` là fingerprint của chính mảng logits ID trên tập test. Nếu nó không đổi thì đường ID
bit-identical. Mỗi lần chạy đều in ra.

**Phản ví dụ, để không ai lặp lại:** hôm nay đã thử sửa trong Phase 1 bằng hai cờ `--card_feats sorted`
và `--card_target true`. Cả hai **đổi `fp` từ `b6dc7969c8` sang `3e96c8cb15`** — vì `gate` KHÔNG detach
nên nó nối vào `logits_cls` qua `gate_logit`, và sửa cách dạy phần đếm lan thẳng vào encoder.
**Đã gỡ cả hai.** Đây chính là lý do ràng buộc "làm trên LoRA" là đúng chứ không phải cẩn thận quá.

---

## 1. Ba nguồn tín hiệu, và nguồn nào mang ID

Phase 1 train một lần rồi đóng băng. Từ nó lấy ra ba nguồn cho phần đếm:

| nguồn | là gì | có ID không |
|---|---|---|
| `gate` (45) | AdaSlot slot existence, `sigmoid(gate_logit)` | **không** |
| `MAC` (19) | số allele mỗi locus ở ngưỡng RFU 0/50/150/500 + thống kê chiều cao log | **không** |
| posterior ID (45) | `sigmoid(logits_cls)` | **CÓ** |

`MAC` là bộ đặc trưng vật lý của PACE (Marciano & Adelman 2017) và deepNoC 2024. Nó đo **bằng chứng
điện di**, không đi qua danh tính donor và không đi qua ID. `_mac_feats_torch` đã là `@torch.no_grad()`
nên nó là đặc trưng **cố định** — thêm vào LoRA head không sinh thêm một gradient nào vào backbone.

Đây là nguồn duy nhất **không thể hỏng khi ID hỏng**.

---

## 2. Sáu cách đọc ra NoC

| head | gate | MAC | posterior ID | nhãn thật | đóng | ngoài panel |
|---|---|---|---|---|---|---|
| `gate` — `round(sum gate)` | ✓ | — | — | không | 0.716 | chưa ghi |
| `gate_cal` — hiệu chỉnh trên DEV | ✓ | — | — | không | 0.685 | **0.630** |
| `zero_shot` — `logits_card` | ✓ **trục donor** | — | — | không | **0.910** | chưa ghi |
| `corn` — `noc_head_v2` | qua `slot_mass` | ✓ | ✓ đã sắp xếp | không | chưa đo | chưa đo |
| **`lora_inv`** *(đang ship)* | ✓ | — | ✓ | **cần** | **0.924** | **0.198** |
| `lora_mac` *(mới)* | ✓ | ✓ | ✓ | cần | chưa đo | chưa đo |
| `lora_free` *(mới)* | ✓ | ✓ | — | cần | chưa đo | chưa đo |

Số là fold 0, bộ 50k `--conditioned`, backbone `fp 6b61691655`. "ngoài panel" = 1.366 hàng có ít nhất
một donor ngoài panel 45 người, tức **nơi ID không thể đúng**.

**Hai ô in đậm là toàn bộ vấn đề.** `lora_inv` mạnh nhất ở tập đóng (0.924) nhưng sụp xuống **0.198**
ngoài panel, còn `gate_cal` yếu ở tập đóng (0.685) mà giữ **0.630**. Đảo chiều, và đảo đúng ở nơi ID sai.

---

## 3. Chỗ neo vào ID nằm ở đâu — sửa lại chẩn đoán

Ban đầu tưởng chỗ neo là **trục donor**: `gate[i]` là donor `i` (slot khởi tạo từ bộ gen donor `i`), nên
`noc_head = Linear(45,5)` đọc theo thứ tự donor và học được "cột 17 và 29 sáng → 4 người".

**Đúng cho `logits_card`, nhưng KHÔNG đúng cho `lora_inv`** — `InvariantCountHead` đã sắp xếp gate giảm
dần, tức đã bỏ trục donor từ trước.

Chỗ neo thật của `lora_inv` là **giá trị** lấy từ `sigmoid(logits_cls)`, và docstring của chính nó nói
nó là đặc trưng áp đảo:

> gate một mình được ~0.49 macro F1 trên fold 1, còn profile xác suất đỡ tới ~0.88

Sắp xếp bỏ được **ai**, không bỏ được **số liệu từ ID**. Đó là vì sao `lora_inv` sụp ngoài panel.

**Sửa một chỗ nói sai trước đó:** `corn` **cũng** đọc posterior ID (profile đã sắp xếp), nên nó không
phải head "không neo ID". Cái phân biệt `corn` là **nhãn dạy nó là NOC thật** và nó có MAC.
Head thật sự không đọc posterior ID là `lora_free`, `gate`, `gate_cal`.

---

## 4. Thiết kế: một `InvariantCountHead`, ba cấu hình đặc trưng

Toàn bộ thay đổi nằm trong `noc_lora.py`. Không file nào khác đổi hành vi.

```python
LORA_FEATS = {"lora_inv":  dict(use_probs=True,  use_mac=False),
              "lora_mac":  dict(use_probs=True,  use_mac=True),
              "lora_free": dict(use_probs=False, use_mac=True)}
```

| head | d_in | tham số head | LoRA r8 | tổng train được |
|---|---|---|---|---|
| `lora_inv` | 28 | 1.093 | 24.592 | 25.685 |
| `lora_mac` | 47 | 1.701 | 24.592 | 26.293 |
| `lora_free` | 33 | 1.253 | 24.592 | 25.845 |

Backbone 1.505.737 tham số **đóng băng toàn bộ** ở cả ba.

---

## 5. Hai mức tham vọng — quyết trước khi chạy

**Mức 1 — một head duy nhất, KHÔNG định tuyến.**
Nếu `lora_mac` tốt ở **cả hai** cột thì không cần định tuyến gì. Thiết kế gọn hơn hẳn và dễ bảo vệ trong
bài: một backbone, một lượt chạy, một head đếm, một đường ID.

**Mức 2 — định tuyến bằng reject head.**
Chỉ đáng làm nếu `lora_mac` vẫn phải đánh đổi (mạnh trong panel thì yếu ngoài panel). Lúc đó reject head
(AUROC 1.0) chọn: trong panel dùng head đọc ID probs, ngoài panel dùng head không đọc.

**Nên thử mức 1 trước** — rẻ hơn (cùng một lần chạy) và nếu đạt thì thiết kế đơn giản hơn nhiều.

> **Chỗ còn để ngỏ, phải nói rõ:** bước định tuyến **chưa bao giờ được đo**. Mọi con số "ngoài panel"
> trong doc này là **một** head chạy trên **toàn bộ** nhánh open, không phải một bộ định tuyến thật.
> Reject AUROC 1.0 nói rằng về nguyên tắc tách được, không nói rằng pipeline định tuyến sẽ đạt bao nhiêu.

---

## 6. Phải chạy gì

Vì **backbone fold 0 đã mất** (lần thứ ba), phải train lại Phase 1 đằng nào cũng vậy. Nên:

**Một Phase 1 trên 12k (~1,1 giờ), rồi năm head trên CÙNG backbone đó:**
`lora_inv`, `lora_mac`, `lora_free`, `corn`, `gate`.

Mọi so sánh trong cùng một model, cùng split, nên không confound nào.

### Giả thuyết đặt trước — ghi trước khi chạy, không chỉnh sau khi xem số

| | nội dung |
|---|---|
| **H1** | `lora_mac` ≥ `lora_inv` ở tập đóng (thêm kênh không được làm tệ đi) |
| **H2** | `lora_mac` ngoài panel ≥ `lora_inv` ngoài panel **+ 0.05** |
| **H3** | `lora_free` ngoài panel > `lora_inv` ngoài panel |
| **H4** | cả năm head cùng `fp(logits_cls)`, `params_unchanged=True`. Sai → **dừng**, có lỗi |
| **H5** | H1 và H2 đều đúng → **mức 1 đủ, không cần định tuyến** |

Luật đọc kết quả, chốt trước:
- H1 + H2 đúng → ship `lora_mac`, làm lại trên 50k, bỏ hướng định tuyến
- H2 sai nhưng H3 đúng → MAC chỉ giúp khi **bỏ** ID probs → đi mức 2 (định tuyến)
- H2 và H3 đều sai → MAC không phải mảnh còn thiếu. **Đóng hướng và ghi lại**
- H4 sai → có lỗi lập trình, không phải kết quả

### Phải kiểm trong log

- **`enrich-after-feas keeps`**: 0.779 = bộ cũ, **0.788** = `ft_att` 12k (KHÔNG `--conditioned`),
  **0.811** = bộ `--conditioned`. Nếu bộ 12k mới ra ~0.788 thì nó cùng build với `ft_att`, và **số
  tuyệt đối không chuyển sang bộ 50k được** — chỉ phép so A/B là còn giá trị.
- **Save & Run All (Commit)**, không interactive. Cell lưu backbone **ngay tại dòng `Training done in`**.
  Đã mất backbone ba lần, cả ba lần cùng một nguyên nhân: lưu sau khi cả lần chạy xong.

### 12k có đủ để kết luận không

**Đủ cho phép so A/B**, vì năm head dùng **cùng** backbone và **cùng** dữ liệu nên chất lượng generator
bị triệt tiêu hoàn toàn. Chỉ **số tuyệt đối** là không chuyển trực tiếp sang bộ 50k.

---

## 7. Kết quả smoke — chạy được, nhưng ĐỘ LỚN KHÔNG TIN ĐƯỢC

CPU, 1 epoch, `work/smoke_fold1`, cùng backbone cho cả ba head:

| head | đóng F1 | ngoài panel acc |
|---|---|---|
| `lora_inv` | 0.0903 | 0.0333 |
| `lora_mac` | 0.3897 | 0.6933 |
| `lora_free` | 0.1927 | 0.3933 |

**Vì sao con số này bị thổi phồng:** ở 1 epoch, `sigmoid(logits_cls)` gần như vô dụng, nên head nào dựa
vào nó thì trắng tay, còn MAC là đo vật lý nên vẫn chạy. Trên backbone 150 epoch thật, ID posterior rất
tốt (oracle EM 0.98) nên `lora_inv` sẽ khá hơn nhiều và khoảng cách **phải co lại mạnh**.

**Thứ smoke thật sự chứng minh:**

```
fp(logits_cls)      b6dc7969c8  ==  b6dc7969c8     (y hệt bản gốc)
params_unchanged          True
oracle_em / em_at_pred_k / reject_auroc / alpha / per_noc_oracle:  GIỐNG HẾT
```

Đường ống chạy đúng, ID bất biến, ba head so được trên cùng backbone cùng split. Không hơn.

---

## 8. Trạng thái code

| file | đổi gì |
|---|---|
| `noc_lora.py` | `InvariantCountHead` thêm `use_probs` / `use_mac`; `_NocLoraView` truyền MAC; `noc_lora_kfold` thêm hai tham số |
| `train_set_transformer.py` | `LORA_FEATS`, arm `lora_mac` / `lora_free` / `corn`, lưu `sum_gate_per_row`, `count_decode_per_source` |
| `models/set_transformer.py` | **không đổi hành vi** (hai cờ Phase 1 đã gỡ sạch) |

`models/ordinal.py` có `corn_probs()` — tồn tại từ đầu dự án và **chưa ai từng gọi**. Giờ arm `corn`
gọi nó, nên `noc_head_v2` lần đầu tiên được đọc ra sau khi đã train 150 epoch trong mọi lần chạy.

**Cell cũ `cell_neo_id_12k.py` giờ SAI THIẾT KẾ** — nó bật hai cờ Phase 1 đã bị gỡ. Phải viết lại
thành một Phase 1 + năm head.

**Code đã bỏ nằm ở `archive/2026-10-01_truoc_clean/`**: `noc_clone.py`, `noc_posthoc.py`, và bản
`train_set_transformer.py` trước khi dọn. Không xoá gì.
