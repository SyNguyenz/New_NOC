# Thí nghiệm không được giữ lại — NoC + ID (DS-ST)

Ghi ngày 2026-09-15, lúc dọn code. Code chuẩn còn lại chỉ là pipeline một fold:
`code/kaggle_run_increment1.py` → `code/train_set_transformer.py` (Phase 1 → ID phi-rerank → đếm NoC bằng LoRA r8, 5-fold tách
tổ hợp → báo cáo `noc_id_fold_report/v1`), với `code/noc_lora.py` (phần đếm) và `code/fold_report.py`
(fingerprint, stamp fold, nhãn open, chẩn đoán, ghi báo cáo); gộp fold bằng `code/report_folds.py`.

Mọi con số dưới đây lấy từ comment/docstring/commit/log của chính các lần chạy đó (cột **nguồn**), không
đo lại. Tên file ở cột nguồn là **bản thí nghiệm** trong `archive/2026-09-15_truoc_clean/` (trừ README.md, commit git, log Kaggle); `train_set_transformer.py`, `kaggle_run_increment1.py` và `models/set_transformer.py` trong `code/` là bản gốc đã được dọn lại. Các con số nằm trên **những split khác nhau** (strict single split, K-fold 2577 hồ sơ tách tổ hợp,
protocol deepNoC, pooled 25 s) — chỉ so được trong cùng một dòng/nhóm, không so chéo giữa các dòng.

Code của mọi thí nghiệm ở đây đã chuyển vào `archive/2026-09-15_truoc_clean/` (giữ nguyên đường dẫn
tương đối). Bundle đầy đủ tái lập so sánh với deepNoC: `kaggle_upload/noc-bundle-3a4eab6847-archive.zip`.

---

## 1. Đếm NoC — các cách không được chọn

| thí nghiệm | kết quả đo | vì sao không giữ | nguồn |
|---|---|---|---|
| CORN ordinal head (`noc_head_v2`) dùng làm số đếm | recall NOC3/NOC4 ~0.21 / 0.17 | sụp ở NOC3/4. Head vẫn được **train** trong Phase 1 (gradient của nó nằm trong `clip_grad_norm_`, bỏ đi sẽ đổi quá trình train) nhưng không bao giờ decode | README.md, models/set_transformer.py |
| Đếm trên điểm đã phi-rerank ("lop count") | NOC3 0.917→0.784, NOC4 0.930→0.860, NOC5 +8pp | đổi NOC3/4 lấy NOC5 | README.md |
| Readout NoC từ vector ID pooled `z` | 0.96 trong phân phối, **0.18** trên tổ hợp donor mới; 0.59 khi fit trên 18k tổ hợp | `z` mã hoá *ai*, không phải *bao nhiêu* → học thuộc tổ hợp | train_noc_branch.py, train_noc_head.py |
| Feature vật lý từ peak thô | macro-F1 NOC2–5 0.44 | yếu | train_noc_head.py |
| Encoder fork + pooled vector (4 biến thể: fork last/full × init copy/random) | macro-F1 NOC2–5 0.73 | thua RF posthoc thời đó (0.875) | train_noc_head.py |
| Head FiLM trên feature encoder H (`NocFiLMHead`) | 0.46 (so với count features 0.72, sum gate 0.59) | generator in-silico giữ tổng template cố định theo k nên RFU/peak giảm ~1/k, còn dữ liệu thật thì RFU **tăng** theo NoC → H học tín hiệu RFU bị đảo chiều | models/set_transformer.py (NocCountHead docstring) |
| Head trên feature đóng băng: NocCountHead linear (280 params) / MLP-32 / GBT-300 | macro-F1 0.523 / 0.527 / 0.526 | cả ba ~0.49–0.54, tăng capacity không giúp → biểu diễn đóng băng là nút thắt | train_set_transformer.py (Phase 2b comment) |
| Per-locus NoC stream (`PerLocusNocStream`) | không còn số đo trong repo | chưa từng bật trong cấu hình chuẩn (`noc_film=False`) | models/set_transformer.py |
| Sum(gate) làm số đếm (`--noc_arm`, `tierA_count`) | gate ban đầu tương quan **−0.966** với NoC; thêm loss 0.05·SmoothL1 → +0.976. round(Sum gate) 0.838 macro-F1 NOC2–5 (fold 0, 1 seed); tierA 0.882 vs 0.854 | **loss gate được giữ** trong Phase 1 (`w_gate=0.05`); bản thân số đếm gate chỉ còn là baseline, tierA bị thay | commit b434dff |
| `noc_branch`: clone cả backbone rồi fine-tune (Phase 2b), single split | lịch chọn **bằng eval**: 15ep@1e-5 0.842 / 60ep@1e-5 0.758 / 60ep@5e-5 0.572 | số epoch là hyperparameter chọn trên test → không sạch | train_set_transformer.py `[SCHEDULE PROVENANCE]` |
| `noc_branch` K-fold (2577 hồ sơ, tách tổ hợp) | macro-F1 **0.8507** (recall .985 .908 .894 .694 .780) vs RF posthoc **0.8817** | thua RF → chỉ còn là ablation | train_set_transformer.py |
| Inner val không nhóm (`--branch_val_grouped none`, như notebook tham chiếu) | inner val .8719→.8859→.8867 trong khi eval .9032→.8997→.8925 | val rò rỉ (chung mẫu và tổ hợp với train) nên chỉ sai hướng | train_set_transformer.py |
| Ablation `branch_init=random` | 0.844 → 0.119 | không phải method; cho thấy prior Phase 1 là bắt buộc | models/set_transformer.py (widen_for_branch) |
| `branch_extra` (+6 feature stutter/size cho nhánh đếm), `branch_noc1_frac`, `branch_unfreeze=decoder` | không còn số đo trong repo | nhánh `noc_branch` bị thay toàn bộ | train_set_transformer.py |

## 2. RF posthoc và các biến thể

| thí nghiệm | kết quả đo | vì sao không giữ | nguồn |
|---|---|---|---|
| `posthoc_rf`: RandomForest 10 feature trên profile xác suất 45 donor, fit leave-one-combo-out | K-fold 0.8817 (recall .998 .946 .918 .836 .712); protocol deepNoC 0.8873 | cần hồ sơ hỗn hợp thật có nhãn để fit; **bị giới hạn panel** (off-panel chỉ 0.67–0.82) vì donor ngoài panel không có cột xác suất. Bỏ hẳn 2026-09-11, thay bằng NoC LoRA | train_set_transformer.py; log Kaggle 5 seed 2026-09-10 (CODE b7de359feb) |
| `posthoc_rf_x` (+15 feature phi + 55 feature count của backbone) | 0.8594 (K-fold) / 0.8681 (deepNoC) / 0.8537 (strict) — mốc đăng ký trước 0.8817 | thua cả 3 lần; NOC5 sụp .602 / .645 / .626 (70 chiều thêm làm rừng bỏ lớp cao nhất) | train_set_transformer.py |
| `posthoc_rf_free` (+18 feature không cần tham chiếu, đọc thẳng từ peak) | closed −0.0015 (mốc +0.01), open +0.0185 (mốc +0.10). Riêng feature: 0.7138 trong panel vs 0.7118 ngoài panel, nhưng chỉ 0.4977 khi đứng một mình | trượt cả hai mốc đăng ký trước | train_set_transformer.py |
| `posthoc_rf_ft_free` | closed +0.0184 nhưng 0.9056 < 0.9121 của `posthoc_rf_ft` | bị biến thể khác trội hơn | train_set_transformer.py |
| `rf_calib` (nghiêng posterior theo thứ tự, tau chọn trên OOB) | ghi nhận là lần thất bại thứ ba khi sửa RF (sau phi+count và reference-free); bệnh gốc: precision NOC5 0.953 vs NOC4 0.730, 90/372 hồ sơ NOC5 bị gọi là 4 | không cải thiện | train_set_transformer.py |
| Ensemble 0.5·RF + 0.5·noc_branch | **không thất bại**: qua bài test đăng ký trước trên protocol deepNoC (0.9089 vs 0.8873, lợi ích dồn ở NOC4+5); K-fold 0.8896 vs 0.8817; so với deepNoC 5 seed 0.951±0.009 | **bị thay** bởi C_lora8 (0.9505±0.003): không cần RF, một model, phương sai thấp hơn | train_set_transformer.py; log Kaggle 5 seed 2026-09-10 (CODE b7de359feb) |

## 3. Các cách gắn nhánh đếm vào một backbone (5 seed 42–46, protocol deepNoC 25 s)

Nguồn: log Kaggle 5 seed ngày 2026-09-11 (CODE b6f0ee8dca / fdedbd8a82, chỉ khác dòng in luật). Luật chọn viết **trước khi chạy**: chọn arm ít tham số nhất đạt mức 1 (P(Δmacro ≤ 0) trung bình < 0.025
so với deepNoC 0.9097); mức 2 = trung bình macro > 0.9097.

| arm | tham số train | macro F1 | mức | kết quả |
|---|---|---|---|---|
| clone (copy cả backbone) | 1,505,737 | 0.943 | 2 | không chọn (tái lập đúng noc_branch cũ) |
| B_block (adapter sau mỗi khối ISAB) | 16,934 | 0.935 | 2 | không chọn |
| A_mid (adapter giữa encoder và decoder) | 8,582 | 0.926 | 2 | không chọn |
| **C_lora8 (LoRA r8, 33 ma trận)** | 99,494 | **0.9505 ± 0.003** | **1** | **được chọn → pipeline chuẩn** |

Lưu ý: 4 arm so trên cùng tập test; hiệu chỉnh Bonferroni cho 4 arm thì p hai phía bảo thủ ~0.02 chưa qua
0.0125 (p theo seed 0.002 thì qua). Cần một lần chạy xác nhận C_lora8 trên seed mới (47–51).

**C_lora8 trên protocol tách tổ hợp (2026-09-18, CODE 259766cf23, seed 42) — THẤT BẠI.** Mức 1 ở trên đo
trên protocol deepNoC, nơi tổ hợp donor **chung** giữa fine-tune và test. Chạy lần đầu trên test tách tổ hợp
(LoRA 5-fold trong test của từng fold archive):

| fold | macro F1 | F1 NOC1..5 | train pool F1 mỗi fold | so với cùng test fold 1 (code cũ) |
|---|---|---|---|---|
| 1 (2577 hồ sơ, tổ hợp NOC2..5 = 7/4/6/5) | **0.5115** | .982 .709 .348 .129 .390 | 0.997–1.000 | posthoc_rf 0.8817, noc_branch 0.8507, ensemble 0.8896 |
| 2 (2240 hồ sơ, tổ hợp = 6/4/4/3) | **0.609** | .972 .835 .828 .214 .196 | 0.999–1.000 | — |

Phase 1 và ID không đổi: backbone fold 1 train lại cho đúng fp(logits_cls) 056326a272, oracle EM 0.953 như
bản cũ. Hỏng nằm ở phần đếm: mỗi fold LoRA chỉ học 3–5 tổ hợp mỗi mức NoC, thuộc lòng chúng (train F1 ~1.0)
rồi đảo NOC4↔NOC5 trên tổ hợp mới (fold 1: 224/324 NOC4 gọi thành 5, 191/372 NOC5 gọi thành 4). LoRA sửa
encoder nên mang thông tin danh tính → học thuộc tổ hợp (đúng bài học 1 ở mục 8). Split `open` chỉ thêm 1–4
tổ hợp mỗi mức, không đủ bù. Kết luận: 0.9505 chỉ dùng được cho so sánh cùng protocol với deepNoC, KHÔNG
phải số tổng quát hoá sang tổ hợp mới.

**Thử cứu LoRA, fold 1 (2026-09-18, `compare_counts.py`, backbone fp 056326a272, luật đặt trước: thay RF chỉ khi
thắng RF ở fold 1 và không thua ở fold 2).** Cùng 2577 hồ sơ test, cùng nhóm tổ hợp:

| cách đếm | macro F1 | F1 NOC1..5 | EM@k (ID) | acc open (có người ngoài panel) |
|---|---|---|---|---|
| `rf` (RF LOCO trên profile xác suất) | **0.8817** | .989 .947 .878 .780 .815 | **0.907** | **0.203** |
| `lora_sim` (LoRA chỉ học in-silico) | 0.4958 | .921 .625 .382 .315 .237 | 0.652 | 0.692 |
| `lora_sim_real` (in-silico rồi fine-tune thật) | 0.5789 | .979 .792 .393 .188 .542 | 0.696 | 0.589 |
| `ens_rf_sim` (0.5/0.5) | 0.7987 | .957 .823 .740 .721 .754 | 0.845 | 0.659 |
| `ens_rf_simreal` (0.5/0.5) | 0.7574 | .987 .928 .767 .456 .649 | 0.825 | 0.420 |

- RF tái lập **đúng** 0.8817 của code cũ trên cùng test → mốc đối chiếu khớp.
- LoRA học in-silico cũng thua (lệch domain in-silico → thật, cùng kiểu với head FiLM ở mục 1); train trước
  giúp LoRA thật từ 0.5115 lên 0.5789 nhưng vẫn xa RF. Ensemble kéo RF xuống.
- Cổng sau fold 1: không cách nào thắng RF → luật quyết định, fold 2 không chạy. **RF được dùng cho báo cáo
  10 fold tách tổ hợp.**
- **Giới hạn phải công bố:** RF gần như hỏng trên hồ sơ có người ngoài panel (acc 0.203 trên 1339 hồ sơ open
  của fold 1), vì người lạ không có cột xác suất; LoRA giữ được 0.59–0.69 ở đó.

**Clone (noc_branch) trên cùng test, 2026-09-20 — lượng fine-tune mới là thứ quyết định.** Cùng backbone
(fp 056326a272), cùng 5 fold tách tổ hợp, chỉ đổi ngân sách train:

| cách đếm | số bước/fold | macro F1 | F1 NOC1..5 | EM@k | acc open |
|---|---|---|---|---|---|
| `rf` | — | **0.8817** | .989 .947 .878 .780 .815 | 0.907 | 0.203 |
| `clone` (công thức đã công bố: 10 ep head @3e-4 + 15 ep toàn bộ @1e-5) | ~490 | 0.8031 | .983 .932 .851 .552 .699 | 0.846 | 0.485 |
| `clone_long` (cùng clone, ngân sách của LoRA) | 3000 | 0.6918 | .979 .881 .641 .312 .646 | 0.772 | 0.467 |
| `ens_rf_clone` | — | 0.8699 | .990 .953 .879 .738 .791 | 0.898 | 0.401 |
| `ens_rf_clonelong` | — | 0.8128 | .990 .952 .879 .526 .718 | 0.860 | 0.340 |
| `lora_real` (đo 2026-09-18) | 3000 | 0.5115 | .982 .709 .348 .129 .390 | 0.646 | 0.773 |

- **Cùng kiến trúc, tăng bước train 490 → 3000 thì tụt 0.803 → 0.692** (train loss 0.20 → 0.09). Vậy khác
  biệt clone/LoRA trước đây phần lớn là do ngân sách fine-tune, không phải kiến trúc.
- Ô còn trống: **LoRA ở ~490 bước** chưa đo.
- Clone nhẹ (0.8031) vẫn dưới mốc cũ 0.8507. Không tra được log của lần đo 0.8507 (không còn trong repo),
  nên không biết lần đó dùng K mấy; lần này K=5 để ghép cặp với các fold của LoRA. Cách chia fold, công
  thức train và đường dự đoán thì giống hệt code cũ.
- Cổng dừng sau fold 1: không cách nào thắng rf → **rf được dùng cho báo cáo 10 fold**, fold 2 không chạy.
- Off-panel: clone 0.485 và LoRA 0.773 đều hơn hẳn rf 0.203.

**Head đếm bất biến (`clone_inv`) — 2026-09-21, fold 1, CODE eb968de11c, backbone fp 056326a272. ĐẠT.**
Luật đặt trước: chỉ nhận nếu vượt clone 0.8031. Cơ chế: `gate[i]` gắn cứng với donor i nên
`noc_head = Linear(45,5)` nhớ được tổ hợp; `clone_inv` thay bằng head đọc **gate và profile xác suất đã
sắp xếp** (12 giá trị lớn nhất + tổng + số ≥ 0.5 mỗi loại), chuẩn hoá bằng scaler cố định. Encoder vẫn
nhận gradient qua gate, phần ID không đổi.

| | macro F1 | acc | F1 NOC1..5 | EM@k | acc open (người ngoài panel) |
|---|---|---|---|---|---|
| clone (head theo thứ tự donor) | 0.8031 | 0.864 | .983 .932 .851 .552 .699 | 0.846 | **0.485** |
| **clone_inv (head bất biến)** | **0.8665** | **0.910** | .992 .962 .848 **.722** **.808** | **0.891** | 0.201 |
| rf (đã bỏ khỏi pipeline theo yêu cầu) | 0.8817 | 0.918 | .989 .947 .878 .780 .815 | 0.907 | 0.203 |

- Lợi ích dồn đúng chỗ yếu: NOC4 .552 → .722, NOC5 .699 → .808.
- Train loss mỗi fold còn 0.04–0.07 (thấp hơn clone 0.16–0.23) nhưng held-out **cao hơn** → không phải học
  thuộc nhiều hơn, mà tổng quát hoá tốt hơn.
- Ngưỡng tin cậy: đạt accuracy 0.95 ở t=0.83 với 91,2% hồ sơ được phân loại (clone chỉ 77,7%).
- **Đánh đổi phải công bố:** trên hồ sơ có người ngoài panel, clone_inv chỉ 0.201 so với clone 0.485 — vì
  nó đọc profile xác suất 45 donor, đúng điểm yếu của rf (0.203). Dùng cho casework có người lạ thì phải
  cân nhắc, hoặc gác cổng bằng reject head.
- Ba lỗi phải sửa khi dựng head này: LayerNorm xoá mất độ lớn tuyệt đối của gate (tín hiệu đếm) → scaler
  affine cố định; head mới không học nổi ở lr 1e-5 của backbone → nhóm lr riêng 3e-4; head mới không hội
  tụ trong 10 epoch qua encoder → warm-up 200 epoch trên feature đã cache.
- Chưa có: xác nhận trên fold 2 và nhiều seed.

**LoRA + head bất biến, theo vị trí gắn (`lora_inv --lora_scope`) — 2026-09-22, fold 1, CODE b1615241a9,
backbone fp 056326a272, 3000 bước/fold, lr LoRA 5e-4.** Luật đặt trước: chỉ nhận nếu vượt clone_inv 0.8665.
KHÔNG ĐẠT.

| scope | ma trận / tham số LoRA | macro F1 | F1 NOC1..5 | EM@k |
|---|---|---|---|---|
| all | 33 / 99.264 | **0.7715** | .993 .963 .621 .557 .724 | 0.832 |
| encoder_attn (chỉ attention encoder) | 12 / 28.672 | 0.7581 | .990 .935 .715 .461 .690 | 0.821 |
| decoder | 10 / 24.592 | 0.7570 | .977 .896 .737 .492 .682 | 0.820 |
| encoder (attention + FFN) | 20 / 69.632 | 0.7246 | .992 .960 .669 .373 .629 | 0.803 |
| *tham chiếu:* lora (head theo thứ tự donor, đo 2026-09-18) | 33 / 99.264 | 0.5115 | .982 .709 .348 .129 .390 | 0.646 |
| *tham chiếu:* clone_inv | copy cả backbone | 0.8665 | .992 .962 .848 .722 .808 | 0.891 |

- **Head bất biến cứu LoRA rất nhiều:** cùng scope all, cùng 3000 bước: 0.5115 → 0.7715 (+0.26). Chặn
  đường danh tính có tác dụng với cả LoRA, không riêng clone.
- **Vị trí gắn ảnh hưởng ít:** encoder_attn ≈ decoder ≈ all (chênh ≤ 0.014, 1 seed). Thêm FFN của encoder
  (scope encoder) là tệ nhất.
- **Vẫn dưới clone_inv 0.8665** — nhưng lại lẫn biến ngân sách: LoRA 3000 bước ở lr 5e-4, clone_inv ~490
  bước ở lr 1e-5. Clone từng tụt 0.803 → 0.692 khi tăng lên 3000 bước. Ô chưa đo: **lora_inv ~500 bước**.
- Cột open trong cell hiện `nan` vì cell đọc khoá `count_info`, còn báo cáo lưu ở `noc.lora` — số liệu vẫn
  nằm trong JSON.

**lora_inv cùng ngân sách với clone_inv (15 epoch ≈ 490 bước, lr head 3e-4) — 2026-09-22, fold 1, CODE
0a6b742cbe.** Luật: chỉ nhận nếu vượt 0.8665. KHÔNG VƯỢT, nhưng scope decoder gần như hoà.

| scope | tham số train | macro F1 | F1 NOC4 | F1 NOC5 | EM@k | acc open | train F1 cuối |
|---|---|---|---|---|---|---|---|
| decoder | 25.915 | **0.8644** | .708 | .814 | 0.888 | 0.142 | ~0.95 |
| encoder | 70.955 | 0.8523 | .678 | .783 | 0.878 | 0.190 | ~0.99 |
| encoder_attn | ~30.000 | 0.8452 | .660 | .786 | 0.877 | 0.197 | — |
| all | 100.587 | 0.8248 | .624 | .752 | 0.864 | 0.167 | ~0.999 |
| *clone_inv* | copy 1,5 triệu | *0.8665* | *.722* | *.808* | *0.891* | *0.201* | — |

- **Ngân sách là thủ phạm chính:** scope all 3000 bước 0.7715 → 15 epoch 0.8248 (+0.053).
- **decoder chỉ kém clone_inv 0.002** với số tham số train ít hơn ~58 lần — trong phạm vi nhiễu của 1 seed.
  Theo luật (phải vượt) thì clone_inv giữ nguyên.
- **Càng ít đụng encoder càng tổng quát tốt:** decoder > encoder > encoder_attn > all. Scope decoder không
  fit hết được phần train (train F1 cuối ~0.95, loss ~0.12) còn scope all fit gần tuyệt đối (~0.999, loss
  ~0.01) — ít chỗ học thuộc hơn thì held-out cao hơn. Khớp với giả thuyết: học thuộc nằm ở feature encoder.
- Ngưỡng tin cậy: decoder đạt acc 0.95 ở 91,1% hồ sơ (clone_inv 91,2%).
- Off-panel vẫn yếu ở mọi bản (0.14–0.20), cùng điểm yếu của head đọc xác suất 45 donor.
- Lỗi in nhỏ: banner đầu log ghi "LoRA r8 x 3000 steps" dù chạy 15 epoch (dòng tiêu đề phần đếm thì đúng).

## 4. Lỗi protocol / rò rỉ đã phát hiện (đã sửa, ghi lại để không lặp lại)

| lỗi | đo được | cách sửa | nguồn |
|---|---|---|---|
| Leak guard chỉ ở **mức dòng** | 27–32 trên 33 hồ sơ NOC1 test (mỗi seed) có **cùng extract** ở 5/15 s nằm trong train Phase 1 → 0.957 bị thổi phồng | guard ở mức extract → 0.951 | log Kaggle 5 seed 2026-09-10 (CODE b7de359feb) |
| Chia single-source **theo mẫu** | NOC1 0.998 trong khi hỗn hợp 0.056 (đếm bằng cách nhận ra donor) | chia theo donor | train_set_transformer.py |
| Dùng backbone của **fold khác** | 9/17 tổ hợp test fold 2 và 14/21 của fold 3 nằm trọn trong donor đã biết của fold 1 | stamp fold cạnh `best_model.pt`, `--init_from` từ chối khi lệch | train_set_transformer.py (`_fold_stamp`) |
| Build cũ có **4 tổ hợp thật lọt vào in-silico train** | 4 tổ hợp | `[SPLIT AUDIT]` chạy mỗi lần, `SHARED` phải = 0 | train_set_transformer.py |
| Reject head train trên **mọi** hồ sơ open | loss reject 0.000 ở epoch 150 → đánh giá known/unknown trên chính nhãn train | withhold donor khỏi Phase 1 khi cần đánh giá known/unknown | train_set_transformer.py |
| In accuracy eval trong lúc fine-tune | đường eval phẳng quanh ep 170 dễ bị dùng để chọn 200 epoch | chỉ in train loss; eval báo cáo một lần sau cùng | train_set_transformer.py |
| Protocol deepNoC (lấy xen kẽ mỗi hồ sơ thứ hai) | hai nửa chung tổ hợp donor | chỉ dùng để đặt cạnh số đã công bố, không coi là ước lượng tổng quát hoá | train_set_transformer.py |
| So với notebook tham chiếu 0.9453 ± 0.0079 (ta 0.8896) | notebook gộp toàn archive (train 5,064 hồ sơ thật, cả off-panel), nhóm theo extract (chung tổ hợp), 10 lần GroupShuffleSplit có test chồng lấn | không phải lợi thế phương pháp; ±0.0079 đánh giá thấp độ dao động | train_set_transformer.py |

## 5. Generator dữ liệu

| thí nghiệm | kết quả đo | vì sao không giữ | nguồn |
|---|---|---|---|
| Peak model (`STR_PEAK_MODEL=1`) thay cho overlay | fold 0: oracle .9272 / post-hoc .6788 / count .7029 vs overlay .9636 / .9126 / .9258. Peak model quá sạch: tỉ lệ peak không donor nào giải thích .036–.040 (thật .17–.22), MAC 7–9 (thật 11–12) | lệch xa dữ liệu thật; quay về overlay | commit 242a573 |

## 6. Phía ID

| thí nghiệm | kết quả đo | vì sao không giữ | nguồn |
|---|---|---|---|
| phi-rerank bằng phi **neural** (attr-phi) | tương quan ~0.95 với logits → dư thừa, không nâng oracle | log opinion pool cần hai nguồn độc lập; giữ phi EM uniform-compat (tương quan ~0.73, nâng oracle NOC5 0.831→0.901) | phi_rerank.py |
| Các module ID bị loại từ đời inc22: replicates, em_phi_feature, noise_gate, ref_match, soft_geno_attr, phi_inject, sparse_attn, minor_weight, irm, vicreg, donor_recon, query_denoise, mass_pool, noc_contrast/noc_ord_head, sic, decoder per_donor/additive/dsmil/sos/spen/pooled/hybrid, geno_query, `cardinality_head` | README ghi "không dùng / đo xấu", không còn số chi tiết trong repo | — | README.md |

## 7. Code đã chuyển vào archive mà không có kết quả ghi lại trong repo

- `diagnose_domain_gap.py` — công cụ so feature thật vs in-silico theo từng tham số generator.
- `make_test_50donors.py` — tạo test in-silico từ cả 50 donor.
- `package_real_train.py`, `notebooks/id-finetune-kaggle.ipynb` — fine-tune ID trên real train (`real_*`).
- `notebooks/decoder-ft-10fold.ipynb` — fine-tune decoder trên dữ liệu thật (trần 0.944 macro-F1 theo train_noc_branch.py).
- `models/set_transformer_10mb.py` — biến thể model lớn hơn.
- `verify_inc22.py` — kiểm tra bit-identical với model gốc (model gốc không còn trong repo).
- `train_noc_branch.py`, `train_noc_head.py` — nhánh/head đếm riêng.
- Bản thí nghiệm của `train_set_transformer.py` (4,651 dòng), `kaggle_run_increment1.py` và `models/set_transformer.py` (930 dòng). File gốc cùng tên trong `code/` được giữ và dọn lại, không xoá.
- Các notebook `kfold_noc`, `noc_final_go`, `paper_kaggle`, `st-noc*`.

## 8. Bài học

1. Thứ **không mang danh tính** (profile xác suất đã sắp xếp, gate, số allele) chuyển giao sang tổ hợp donor
   mới; thứ **mang danh tính** (vector `z`, feature encoder) học thuộc tổ hợp.
2. Mọi con số NoC phải **tách tổ hợp** (hỗn hợp) và **tách extract** (NOC1, mọi thời gian inject).
3. Luật chọn / mốc so sánh viết **trước khi chạy**; lịch train và epoch không được chọn trên test.
4. Kiểm tra rò rỉ trên dữ liệu thật (row hash, extract, stamp fold), không tin vào file cấu hình.
5. Mỗi thí nghiệm kết thúc: ghi kết quả vào file này và **xoá nhánh code** — không để lại trong trainer.

## 9. Tái lập so sánh với deepNoC

Bundle `kaggle_upload/noc-bundle-3a4eab6847-archive.zip` (CODE 3a4eab6847), backbone fold 1
(`mashwoo/noc-inc22-backbone`, fp(logits_cls) 056326a272), seed 42–46:

```
python kaggle_run_increment1.py --seed <S> --init_from <backbone>/best_model.pt \
    --pooled_kfold 1 --pooled_sweep 1 --pooled_steps 3000 --pooled_inject 25 --pooled_arms A,B,C
```

Kết quả: C_lora8 0.9505 ± 0.003 vs deepNoC 0.9097 (ensemble RF+branch cũ 0.951 ± 0.009). Khi đưa vào paper
phải công bố kèm: không cùng từng hồ sơ (666 vs 675 hồ sơ hỗn hợp 25 s); panel 45 genotype; Phase 1 train
trên 5,171 hồ sơ NOC1 thật; input là bảng peak Filtered của PROVEDIt, deepNoC dùng EPG thô; reject head đã
thấy hồ sơ off-panel (không có nhãn NoC); backbone fold 1 không có stamp; 3000 step cố định trước sweep.


## 10. LoRA decoder không giữ được lợi thế sang fold 2 (2026-09-22/23, CODE 94c17a2f60)

Luật đặt trước: H1 hoà (decoder ≥ clone_inv − 0.01), H2 thứ tự (decoder > all), H3 cơ chế (gap decoder < gap all).
Chạy fold 1 (backbone cũ) và fold 2 (Phase 1 train lại trong cùng notebook), cùng 15 epoch, tách tổ hợp.

| fold | count | macro F1 | F1 NOC1..5 | train F1 | held F1 | gap | open |
|---|---|---|---|---|---|---|---|
| 1 | lora_inv decoder | 0.8644 | .990 .954 .857 .708 .814 | 0.949 | 0.900 | +0.048 | 0.142 |
| 1 | clone_inv | **0.8665** | .992 .962 .848 .722 .808 | 0.986 | 0.900 | +0.086 | 0.201 |
| 1 | lora_inv all | 0.8248 | .992 .959 .796 .624 .752 | 0.996 | 0.862 | +0.134 | 0.167 |
| 2 | lora_inv decoder | 0.8268 | .995 .962 .785 .723 .669 | 0.968 | 0.867 | +0.101 | 0.344 |
| 2 | clone_inv | **0.8542** | .994 .970 .894 .782 .631 | 0.991 | 0.884 | +0.107 | 0.379 |
| 2 | lora_inv all | 0.8533 | .993 .971 .882 .788 .633 | 0.997 | 0.887 | +0.111 | 0.367 |

- **H1 SAI** (fold 2: decoder thấp hơn clone_inv 0.027) → theo luật, mặc định quay lại `clone_inv`.
- **H2 SAI**: thứ tự scope đảo giữa hai fold. Fold 1 decoder > all (+0.040), fold 2 decoder < all (−0.027).
- **H3 ĐÚNG nhưng không kéo theo kết quả tốt hơn**: gap của decoder nhỏ hơn all ở cả hai fold, vậy mà fold 2
  decoder vẫn thua. Train F1 của decoder chỉ 0.949/0.968 → **decoder-only thiếu dung lượng (underfit)**, chứ
  không phải "chống học thuộc" đúng nghĩa. Fold 1 tình cờ có lợi, fold 2 thì hại.
- Kết luận: **vị trí gắn LoRA không phải yếu tố quyết định**; chênh lệch giữa các scope (≤ 0.04) nhỏ hơn chênh
  lệch giữa các fold, tức là nhiễu theo tổ hợp lấn át. clone_inv thắng cả hai fold và tốt nhất ở open
  (0.201 / 0.379).
- `heldout_f1_mean` chính là điểm trên các hàng test đã gộp, nên KHÔNG dùng được để chọn mô hình; muốn chọn
  không nhìn test thì phải dựng tập giữ lại riêng (inner val) hoặc chọn theo fold khác.
- Tái lập: fold 1 ra đúng các số của lần chạy trước (0.8644 / 0.8665 / 0.8248); fold 2 train lại Phase 1 từ
  đầu cho fp(logits_cls) = e102bc28c6, trùng với lần chạy 2026-09-18 → pipeline xác định (deterministic).


## 11. Đặc trưng hồ sơ KHÔNG giúp, công thức giải mã thì có (2026-09-23, đo trên máy, backbone fold 1 đóng băng)

880 hàng test thật của fold 1, tách tổ hợp 5-fold, head nhỏ (Linear 64 + GELU), không train backbone.

**a) Thêm đặc trưng hồ sơ (đếm allele mỗi locus, thống kê chiều cao/Hb/SR — 31 chiều): THẤT BẠI.**

| đầu vào head | macro F1 | NOC4 vs 5 AUC |
|---|---|---|
| gate + probs đã sắp xếp (hiện tại) | **0.677** | **0.911** |
| cộng đặc trưng hồ sơ | 0.625 | 0.824 |
| chỉ đặc trưng hồ sơ | 0.386 | 0.563 |

Đếm allele thô gần như không tách được NOC4/5 (AUC 0.563, khớp AUC 0.505 đo trước), còn gate của backbone
tách tới 0.911 → tín hiệu đếm nằm trong biểu diễn, không nằm ở thống kê bề mặt. Thêm 31 chiều nhiễu làm loãng
head (~700 hàng train) và gắn với tổ hợp nên học thuộc nhiều hơn.

**b) Đổi công thức giải mã: CÓ TÁC DỤNG, +0.010 (3 seed, macro F1 trên NOC2..5).**

| công thức | macro F1 | lệch ≥ 2 bậc |
|---|---|---|
| argmax (đang dùng) | 0.8484 ± 0.0027 | 0.017 |
| **kỳ vọng rồi làm tròn** | **0.8588 ± 0.0032** | 0.014 |
| chia prior tập fit | 0.8574 ± 0.0026 | 0.040 |
| CORN (thứ tự) | 0.8472 | 0.015 |

Kỳ vọng thắng argmax ở cả 3 seed (chênh 0.010 so với lệch seed 0.003) và giảm số ca sai nặng. NoC là biến có
thứ tự, argmax bỏ qua điều đó. Chia prior nâng macro F1 nhưng làm lệch ≥ 2 bậc tăng 2,5 lần → loại.
Đã thêm `--count_decode {argmax,expect}`; mặc định vẫn argmax, mọi lần chạy in `[DECODE]` cả hai số để quyết
định trên fold 1+2 bằng pipeline thật.


**Bằng chứng công thức giải mã trên pipeline thật (fold 1, 2026-09-23, CODE 6c48f02066).** Cùng một lần chạy,
cùng posterior, chỉ khác cách lấy k:

| count | argmax | expect | lệch ≥ 2 bậc (argmax → expect) |
|---|---|---|---|
| lora_inv decoder | 0.8644 | **0.8718** | 0.0066 → 0.0062 |
| clone_inv | 0.8665 | **0.8701** | 0.0078 → 0.0062 |

Cả hai cách đếm đều tốt lên, khớp hướng của phép đo tại chỗ (+0.010, 3 seed). Còn chờ fold 2 để chốt theo luật.


## 12. Hâm nóng bằng in-silico trước khi fit tổ hợp thật: THẤT BẠI (2026-09-23, fold 1, CODE 6c48f02066)

Ý tưởng: mỗi fold chỉ có 22 (fold 1) / 17 (fold 2) tổ hợp hỗn hợp thật, mỗi lần fit thấy ~4/5 số đó, nên đẩy
LoRA + head học đếm trên hỗn hợp in-silico (6000 hàng, **4212 tổ hợp**, cùng 45 donor, SHARED=0 với test)
trước, rồi mới fit trên tổ hợp thật. Luật: nhận nếu ≥ +0.005.

| arm | macro F1 | expect | NOC3 | NOC4 | NOC5 | train F1 | gap | open |
|---|---|---|---|---|---|---|---|---|
| presim 0 | **0.8644** | **0.8718** | .857 | .708 | .814 | 0.949 | +0.048 | 0.142 |
| presim 10 ep | 0.8575 | 0.8630 | .885 | .663 | .789 | 0.951 | +0.052 | 0.144 |
| presim 30 ep | 0.8534 | 0.8575 | .880 | .672 | .771 | 0.954 | +0.072 | 0.149 |
| clone_inv | 0.8665 | 0.8701 | .848 | .722 | .808 | 0.986 | +0.086 | 0.201 |

- **Càng hâm nóng nhiều càng tệ** (0.8644 → 0.8575 → 0.8534), đơn điệu, nên không phải nhiễu. H1 sai → đóng
  hướng, không chạy fold 2. Code `--presim_ep/--presim_n` đã gỡ khỏi `noc_lora.py` + `train_set_transformer.py`;
  cell nằm ở `archive/2026-09-23_presim/`.
- **Nó dịch chuyển ranh giới 4/5 theo hướng của in-silico:** NOC3 tăng (.857 → .885) nhưng NOC4 giảm
  (.708 → .663) và NOC5 giảm (.814 → .771). Trên chính tập in-silico, warm-up chỉ đạt recall NOC4 0.90 /
  NOC5 0.86 — tức bản thân miền in-silico có ranh giới 4/5 khác với hồ sơ thật.
- Gap học thuộc còn tăng (+0.048 → +0.072): hâm nóng làm fit phần train thật tốt hơn nhưng held-out kém đi.
- **Kết luận cho paper:** khoảng cách in-silico ↔ thật đủ lớn để thêm dữ liệu mô phỏng cho bộ đếm là phản tác
  dụng, dù Phase 1 vẫn hưởng lợi từ chính dữ liệu đó. Trần hiện tại là số tổ hợp THẬT, và không lấp được bằng
  mô phỏng.
- **`expect` thắng `argmax` ở cả 4 arm** (+0.0074, +0.0055, +0.0041, +0.0036). Bản tốt nhất toàn bảng là
  lora_inv decoder + expect = **0.8718**, cao hơn clone_inv/argmax 0.8665 và clone_inv/expect 0.8701.


## 13. Ngân sách fit thật: ÍT EPOCH HƠN THÌ TỐT HƠN (2026-09-23/24, fold 1, CODE dda225a05a / kit 93ce24e253)

lora_inv scope=decoder, cùng backbone fold 1, chỉ đổi số epoch fit trên tổ hợp thật (và một lượt rank 16).

| arm | macro F1 | expect | F1 NOC4 | F1 NOC5 | train F1 | gap | held F1 | open | EM@k |
|---|---|---|---|---|---|---|---|---|---|
| **r8 × 5 ep** | **0.8802** | **0.8864** | **.747** | **.847** | 0.928 | **+0.023** | **0.904** | 0.156 | **0.899** |
| r8 × 10 ep | 0.8725 | 0.8772 | .726 | .831 | 0.939 | +0.035 | 0.904 | 0.162 | 0.893 |
| r8 × 15 ep (mốc cũ) | 0.8644 | 0.8718 | .708 | .814 | 0.949 | +0.048 | 0.900 | 0.142 | 0.888 |
| r8 × 25 ep | 0.8468 | 0.8525 | .665 | .779 | 0.965 | +0.080 | 0.885 | 0.152 | 0.878 |
| r8 × 40 ep | 0.7988 | 0.8152 | .557 | .732 | 0.977 | +0.136 | 0.842 | 0.161 | 0.849 |
| r16 × 15 ep | 0.8572 | 0.8667 | .684 | .783 | 0.960 | +0.064 | 0.896 | 0.152 | 0.885 |
| clone_inv | 0.8665 | 0.8701 | .722 | .808 | 0.986 | +0.086 | 0.900 | **0.201** | — |

- **Đơn điệu giảm theo ngân sách:** 5 → 40 epoch là 0.8802 → 0.7988. Gap học thuộc tăng đều +0.023 → +0.136.
  Dải 3000 bước (~94 epoch) đo trước đây cho 0.7570, nằm đúng trên đường cong này.
- **Rank 16 cũng tệ hơn rank 8** (0.8572 vs 0.8644) → thêm dung lượng có hại, không chỉ thêm bước.
- **Bác bỏ giả thuyết cũ của tôi** ("decoder underfit nên thua ở fold 2"): thực tế fit ít hơn thì tổng quát
  tốt hơn. Với ~13–18 tổ hợp hỗn hợp mỗi lần fit, mỗi epoch thêm chủ yếu là học thuộc.
- **r8 × 5 ep + expect = 0.8864**, hơn clone_inv/argmax 0.8665 tới +0.020 và hơn clone_inv/expect 0.8701
  tới +0.016; NOC4 .747 và NOC5 .847 đều là cao nhất bảng. Nhưng **open vẫn thua clone_inv** (0.156 vs 0.201).
- **Đỉnh có thể còn dưới 5 epoch** — dải 1–4 epoch chưa quét.
- `expect > argmax` ở **cả 7 arm** của fold 1, cộng fold 2 × 15 ep (+0.0054) là **8/8**.
- Đang chờ chặng 2 (fold 2, backbone Phase 1 vừa train lại, fp e102bc28c6 khớp lần trước) cho 5 ep và 10 ep.


## 14. RandomForest post-hoc: KHÔNG giúp (2026-09-24, fold 1, CODE 988f4aa2e0 / kit ca3f88da81)

RF trên hồ sơ xác suất 45 donor đã sắp xếp (công thức cũ, `noc_posthoc.py`), fit trên **đúng các split**
bộ đếm dùng, rồi ensemble 50/50 với bộ đếm. Giải mã bằng kỳ vọng.

| giao thức | RF một mình | lora_inv một mình | ensemble | ens − đếm |
|---|---|---|---|---|
| strict | 0.8533 | **0.8864** | 0.8727 | **−0.0137** |
| deepnoc | 0.8791 | **0.9158** | 0.8900 | **−0.0258** |

Ensemble **kéo xuống** ở cả hai giao thức, vì RF yếu hơn bộ đếm ở mọi thang. RF cũ đạt 0.8817 (2026-09-18)
khi bộ đếm lúc đó mới 0.8665; giờ bộ đếm 0.8864 nên RF hết chỗ đứng. (Con số RF thấp hơn 0.8817 vì ở đây RF
fit trên đúng split của bộ đếm, không phải LOCO kèm hàng val — đổi lại so sánh không bị lệch dữ liệu.)
→ Đóng hướng RF lần thứ hai, giữ code ở `code/noc_posthoc.py` sau cờ `--posthoc_rf`, mặc định tắt.

## 15. Fold 1 đầy đủ: hai cách đếm × hai giao thức (2026-09-24)

| cách đếm | strict (expect) | strict (argmax) | deepnoc (expect) | deepnoc (argmax) | **premium** |
|---|---|---|---|---|---|
| lora_inv decoder 5 ep | **0.8864** | 0.8802 | 0.9158 | 0.9209 | **+0.0294** |
| clone_inv | 0.8701 | 0.8665 | **0.9646** | 0.9656 | **+0.0945** |

Mốc ngoài: deepNoC công bố **0.9097** (cùng giao thức deepnoc); số cũ của ta 0.9505 (deepnoc, công thức cũ).

- **Đổi thang là đổi người thắng.** LoRA thắng ở strict (+0.0163), clone thắng đậm ở deepnoc (+0.0488).
- **Phần thưởng dùng lại tổ hợp tỉ lệ với dung lượng fine-tune:** clone (1,5 triệu tham số) hưởng +0.0945,
  LoRA (25.915 tham số) chỉ +0.0294. Cùng dữ liệu, cùng split, cùng seed → đây là **đo trực tiếp**, và là một
  phát hiện đáng đưa vào paper: giao thức của deepNoC ưu ái mô hình nào fine-tune nhiều tham số nhất.
- **Per-NOC (strict):** LoRA tốt hơn ở đúng ba lớp khó — NOC3 .869 vs .851, NOC4 .769 vs .731, NOC5 .851 vs
  .815; clone chỉ hơn ở NOC1/NOC2.
- **ID hưởng lây:** EM@k lên **0.9018** (cao nhất từ trước tới nay), donor F1@k 0.9695, fp không đổi.
- **`expect` chỉ thắng ở thang khó:** strict +0.006 và +0.004, nhưng ở deepnoc `argmax` nhỉnh hơn
  (0.9209 vs 0.9158; 0.9656 vs 0.9646). Hợp lý: khi posterior đã sắc nét thì lấy kỳ vọng chỉ làm mờ đi.
- **`expect` làm hỏng cột open:** clone_inv open 0.2009 (argmax, các lần chạy trước) → 0.1352 (expect);
  lora 0.1561 → 0.1322. Cùng mô hình, chỉ khác cách giải mã. Hàng ngoài panel có posterior tản, lấy kỳ vọng
  kéo về giữa. Đã thêm log `[OPEN DECODE]` in cả hai để theo dõi ở mọi fold.


## 16. Đánh giá bộ dữ liệu sinh mới `data_ft_att` (fold 0) — 2026-09-24

Nhận từ nhóm: `incoming/ft_att_2026-09-24/` (data zip, backbone, metrics, model + trainer của họ).
15.148 hàng in-silico (NOC1 4.932 là hồ sơ thật đơn nguồn), dev 2.099, val 1.124 thật, test 2.458 thật.

**Sạch về rò rỉ:** 9.260 tổ hợp hỗn hợp in-silico, 20 tổ hợp thật ở test, **CHUNG = 0**. dev cũng tách
khỏi cả train lẫn test. 0 hàng trùng lặp. 45 donor dùng đều (lệch 3%).

**Truyền sang dữ liệu thật tốt hơn hẳn bộ cũ.** Cùng một phép thử: đặc trưng hồ sơ mù danh tính →
HistGradientBoosting, fit trên in-silico, chấm trên test thật của chính bộ đó (`work/probe_sim2real.py`).

| bộ dữ liệu | acc | macro F1 | macro F1 (hỗn hợp) |
|---|---|---|---|
| **mới (fold 0)** | 0.7274 | **0.6094** | **0.5183** |
| cũ (`data_w`) | 0.6840 | 0.3456 | 0.2272 |

**Khoảng cách miền (AUC phân biệt sinh với thật, 0.5 = không phân biệt được):**

| bộ | NOC1 | NOC2 | NOC3 | NOC4 | NOC5 |
|---|---|---|---|---|---|
| mới | 0.488 | 0.922 | 0.924 | 0.899 | 0.938 |
| cũ | 0.489 | **1.000** | **1.000** | **1.000** | 0.997 |

**Bộ cũ bị lộ ở đâu:** `logh_p10` (phân vị 10 của log chiều cao) AUC **0.993** — sinh 3.198 so với thật
2.282, tức khoảng 24 RFU so với 9 RFU. **Bộ cũ hầu như không sinh ra đuôi peak mờ** mà hồ sơ thật luôn có;
dải động cũng hẹp hơn (logh_std 1.232 vs 1.637). Đó là lý do hâm nóng bằng in-silico cũ làm hỏng ranh giới
4/5 (mục 12): mô hình học đếm trên những hồ sơ không có vùng mờ.

**Bộ mới còn lệch ở đâu (thứ yếu):** tỉ lệ stutter `SR>0.15` 0.306 so với 0.346, `Hb<0.1` 0.534 so với 0.472
(sinh quá nhiều peak rất lệch), `logh_std` 1.775 so với 1.637 (giờ hơi rộng quá).

**Con số của họ (`metrics.json`, fold 0, 2 epoch fine-tune):** đếm bằng `round(sum gate)` — **không dùng
nhãn hỗn hợp thật nào** — đạt acc **0.9552**, macro F1 trên hỗn hợp **0.9139**, corr(sum gate, NOC) 0.9874,
EM@k theo gate 0.9382. Trên dữ liệu cũ, cùng readout zero-shot chỉ được acc 0.6197 (fold 1) và 0.7942
(fold 2). Khác fold nên không so trực tiếp được, nhưng mức chênh vượt xa chênh lệch giữa các fold.

**Hệ quả nếu lặp lại được trên fold 1 và 2:** phần fine-tune đếm (LoRA/clone, mục 10–13) có thể **không còn
cần thiết** — đếm zero-shot của họ đã cao hơn cả bộ đếm fine-tune tốt nhất của ta (acc 0.9212 ở fold 1).
Việc cần làm là sinh lại dữ liệu fold 1 và fold 2 bằng generator mới rồi đo lại, vì đó là hai fold ta đã
có đầy đủ mốc.

**Điểm cần lưu ý về dữ liệu:** NOC2 chỉ có 1.417 hàng trên 680 tổ hợp (2,1 bản mỗi tổ hợp) trong khi NOC4 có
3.695 hàng trên 3.658 tổ hợp (1,0 bản) — mỗi tổ hợp gần như chỉ xuất hiện một lần, tức **không dạy bất biến
theo điều kiện chạy** (cùng tổ hợp ở các mức template khác nhau). Hàng sinh cũng không có `condition_train`
/`meta_template_train`, nên không kiểm được điều đó. 67% hàng có nhãn tách peak thật (`attrfrac`).


### 16b. Kiểm rò rỉ bộ `data_ft_att` — sáu kênh, tất cả sạch (2026-09-24)

| kênh | cách kiểm | kết quả |
|---|---|---|
| tổ hợp hỗn hợp train ↔ test | tập donor của mỗi hàng | 9.260 vs 20 tổ hợp, **chung 0** |
| dev ↔ train, dev ↔ test | như trên | **0** và **0** |
| trùng hồ sơ chính xác | md5 của (locus, allele, chiều cao) mọi peak | **0** ở cả 6 cặp split |
| trùng theo tập allele (bỏ qua chiều cao) | md5 của tập (locus, allele) | **0** ở train↔test, dev↔test, val↔test |
| donor ẩn của fold | tra `open_labels/donor_map.npz` | test 0/2.458, val 0/1.124, train 0/4.932 hàng thật; open **1.366/1.366** đều chứa donor ngoài panel |
| nhãn có đúng không | so `y_*_set` và `noc_*` với donor_map | test 2.458/2.458 khớp, val 1.124/1.124 khớp, **0 lệch** |

Thêm hai quan sát:
- **NOC1 của train là 4.932 hồ sơ ĐƠN NGUỒN THẬT** (tra được trong donor_map), còn NOC2–5 thì 0/10.216 hàng
  tra được, tức hỗn hợp là hoàn toàn tổng hợp. Hồ sơ đơn nguồn thật được chia ba đường rời nhau:
  train 4.932 / val 1.124 / test 1.125.
- **Tỉ lệ peak giải thích được bởi chính donor được gán nhãn** khớp gần như tuyệt đối với thật:
  sinh 0.39 / 0.61 / 0.703 so với thật 0.39 / 0.63 / 0.706 ở NOC 1/3/5. Lượng peak artefact và nhiễu trong
  hồ sơ sinh đúng bằng mức của hồ sơ thật — đây là thứ bộ cũ làm sai (thiếu hẳn đuôi mờ, mục 16).

**Kênh thứ bảy — pool thành phần — đã thử truy ngược và KHÔNG phân giải được, vì lý do có lợi cho ta.**
Nhóm xác nhận generator chỉ chồng hồ sơ NOC1 thật. Câu hỏi còn lại là các hồ sơ đó lấy từ 4.932 hàng phía
train hay cả 7.181 hàng. Ba phép thử (`work/probe_component_pool.py`, `work/probe_component_trace.py`):

1. **Theo chiều cao:** nếu chồng nguyên một lần chạy thì trên allele riêng của donor, tỉ số chiều cao phải
   là hằng số. Thực tế CV trung vị **0.375**, chỉ 9/299 lần có CV < 0.15, và kho khớp nhất phân bố
   74/14/11% — đúng bằng tỉ lệ kích thước ba kho, tức **khớp ngẫu nhiên**. Generator không sao chép hiện
   thực chiều cao của lần chạy nào.
2. **Theo dấu vết artefact riêng:** dấu vết "riêng của một lần chạy" hiện trong hỗn hợp với tỉ lệ 87,1%
   (train), 88,2% (test), 82,8% (val) — bằng nhau, tức không phân biệt được.
3. **Theo vị trí artefact duy nhất toàn kho:** toàn bộ 7.181 lần chạy chỉ dùng **588 vị trí artefact khác
   nhau**, và **không vị trí nào xuất hiện đúng một lần**. Không gian vị trí quá nhỏ để mang dấu vết cá thể.

→ Kết luận: **dù pool thành phần có lấy từ phía test đi nữa, hỗn hợp tổng hợp cũng không mang theo thông tin
cá thể nào của một lần chạy cụ thể** — không trùng hàng, không quan hệ tỉ lệ chiều cao, không dấu vết vị trí.
Thứ được tái sử dụng chỉ là phân bố chung (stutter, nhiễu, dropout), vốn là mục đích của việc mô phỏng.
Vẫn nên kiểm một dòng trong code generator xem nó lấy mẫu thành phần từ mảng nào, và ghi vào Methods.


## 17. BA FOLD ĐẦY ĐỦ — chốt ngân sách 5 epoch và chốt LoRA (2026-09-25, CODE b4b1a18c60)

Một lần chạy 6,7 giờ: fold 1, 2, 3 × (lora_inv decoder 5 ep / clone_inv) × (strict / deepnoc) ×
(expect / argmax), kèm RF post-hoc. fp ID: fold 1 `056326a272`, fold 2 `e102bc28c6`, fold 3 `6e2f43b6ef`;
mọi fold `params_unchanged=True`, `shared combos 0`, tách tổ hợp OK.

| fold | lora strict | clone strict | lora deepnoc | clone deepnoc | premium lora | premium clone | EM@k |
|---|---|---|---|---|---|---|---|
| 1 | 0.8864 | 0.8701 | 0.9158 | 0.9646 | +0.0294 | +0.0945 | 0.902 |
| 2 | **0.8869** | 0.8562 | 0.9300 | 0.9562 | +0.0431 | +0.1000 | 0.924 |
| 3 | 0.9136 | 0.9150 | 0.9317 | 0.9587 | +0.0181 | +0.0437 | 0.933 |
| **TB** | **0.8956 ± 0.0156** | 0.8804 ± 0.0307 | **0.9258 ± 0.0087** | **0.9598 ± 0.0043** | **+0.030** | **+0.079** | 0.920 |

**1. Ngân sách 5 epoch: XÁC NHẬN, và nó cứu đúng fold từng thua.** Fold 2 với 15 epoch cho 0.8268; với
5 epoch cho **0.8869**, tăng **+0.060**. Fold 1 tăng +0.0220. Luật đặt trước (hơn mốc 15 ep ≥ 0.005 ở cả hai
fold) đạt. Kết luận mục 13 lặp lại được: với mười mấy tổ hợp, epoch thêm là học thuộc.

**2. LoRA thắng clone ở thang chặt: XÁC NHẬN.** Chênh từng fold +0.0163 / +0.0307 / −0.0014, trung bình
+0.015, và **ổn định hơn hẳn** (±0.0156 so với ±0.0307). Kết luận cũ "LoRA thua ở fold 2" là hệ quả của
ngân sách sai, không phải của LoRA. Một mô hình đóng băng + 25.915 tham số thắng bản sao 1,5 triệu tham số.

**3. Phần thưởng dùng lại tổ hợp tỉ lệ với dung lượng: XÁC NHẬN trên cả 3 fold.** clone +0.079 trung bình,
LoRA +0.030, không fold nào đảo dấu. Đây là kết quả phương pháp luận mạnh nhất của bài: **giao thức của
deepNoC ưu ái mô hình nào fine-tune nhiều tham số nhất**, và mức ưu ái đo được là 2,6 lần.

**4. So với deepNoC 0.9097 (cùng giao thức của họ):** LoRA **0.9258 ± 0.0087** (+0.016), clone
**0.9598 ± 0.0043** (+0.050). Cả hai đều vượt. Số trung thực của ta ở thang chặt là **0.8956 ± 0.0156**.

**5. Giải mã:** `expect` thắng `argmax` ở **6/6 ô thang chặt**; ở deepnoc thì `argmax` nhỉnh hơn ở 5/6 ô;
ở cột open `argmax` thắng **6/6** (ví dụ fold 2 clone 0.379 so với 0.367). Giữ `expect` làm mặc định vì
thang chặt là kết quả chính, nhưng **bảng open trong paper nên báo theo `argmax`** và nói rõ lý do.

**6. RF post-hoc: đóng hẳn.** Giúp 1/6 ô (fold 2 strict +0.0093), hại 5/6. Không đáng thêm mô hình thứ hai.

**7. Độ khó khác nhau giữa các fold là có thật:** fold 3 dễ nhất cho cả hai (0.91), fold 2 khó nhất cho clone
(0.8562). Độ lệch giữa fold (±0.03 với clone) vẫn lớn hơn mọi hiệu ứng kỹ thuật đã đo — lý do phải chạy đủ
10 fold trước khi viết con số cuối.


## 18. Dữ liệu sinh mới, chạy pipeline chuẩn từ đầu (fold 0, 2026-09-25, CODE b1cff38080)

150 epoch từ đầu, không warm start, trên `data_ft_att` (15.148 hàng). Test 2.458 hàng thật của fold 0.

| cách đếm | strict | deepnoc | premium | open |
|---|---|---|---|---|
| zero-shot `logits_card` (không nhãn thật) | 0.7644 | — | — | — |
| `sum(gate)` (không nhãn thật) | 0.5024 | — | — | — |
| **lora_inv 5 ep** | **0.9062** | 0.9305 | +0.0243 | 0.203 |
| clone_inv | 0.8953 | 0.9413 | +0.0460 | 0.173 |

**H1 — fine-tune còn đáng giá: CÓ, rất rõ.** lora 0.9062 so với 0.7644 của bản tốt nhất không fine-tune,
chênh **+0.1418**. Dữ liệu tốt hơn KHÔNG làm phần fine-tune thừa ra; nó nâng cả hai đầu.

**H2 — theo thước đo đã đặt (sum gate ≥ 0.80): KHÔNG ĐẠT (0.3848), nhưng thước đo sai chỗ.**
`corr(sum gate, NOC) = +0.9001`, tức thứ tự đúng, nhưng thang bị lệch: gate đoán NOC3 608 lần (thật 385)
và NOC4 505 lần (thật 242), trong khi NOC2 chỉ 113 (thật 334). Đây là lỗi **hiệu chỉnh thang**, không phải
lỗi biểu diễn. Thước đo đúng cho "dữ liệu dạy đếm tốt đến đâu" là zero-shot `logits_card`:
**0.7644** so với 0.4256 (fold 1) và 0.6588 (fold 2) của dữ liệu cũ.

**KHÔNG tái lập được con số 0.9552 của nhóm.** Ta được `count_acc_gate` **0.6709**. Khác biệt duy nhất:
họ warm start 2 epoch từ một checkpoint có sẵn (gate đã được dạy đếm từ trước), ta train từ đầu. Nghĩa là
con số của họ **không quy hết cho dữ liệu**; phần lớn năng lực đếm của gate đến từ checkpoint nền.

**ID không hề hỏng, thậm chí là bản tốt nhất từng đo:** oracle EM 0.9703, EM@k 0.9243,
**donor F1@k 0.9763** (cao nhất từ trước tới nay), reject AUROC 0.9994, fp `da8d1a25a1`, mọi kiểm toán sạch.

**Ngưỡng tin cậy tốt hơn hẳn:** đạt acc 0.95 khi phân loại **97,3%** số hồ sơ, so với 91,3% ở fold 1 dữ liệu cũ.

**Hạn chế phải nêu:** đây là fold 0, mà ta CHƯA từng chạy fold 0 với dữ liệu cũ. Mọi so sánh ở trên đều là
so chéo fold. Phép thử quyết định là chạy **fold 0 với dữ liệu CŨ** (`kaggle_upload/noc-inc22-fold0.zip`
đã có sẵn) rồi so hai lần chạy trên cùng một fold, cùng một test.


## 19. QUYẾT ĐỊNH: generator cũ vs mới trên CÙNG fold 0 (2026-09-29, CODE b1cff38080)

Cùng code, cùng `combo_md5 9c00a1125dc1`, cùng 2.458 hàng test, cùng 5.247 hồ sơ đơn nguồn thật.
Khác duy nhất: 50.000 hỗn hợp sinh (cũ) so với 12.000 (mới). Phase 1 150 epoch từ đầu cả hai bên.

| | CŨ (50k) | MỚI (12k) | MỚI − CŨ |
|---|---|---|---|
| **zero-shot `logits_card`** | 0.6153 | **0.7644** | **+0.1491** |
| sum(gate) macro F1 hỗn hợp | **0.4876** | 0.3848 | −0.1028 |
| sum(gate) accuracy | **0.7181** | 0.6709 | −0.0472 |
| **lora_inv strict (expect)** | 0.9026 | 0.9062 | **+0.0036** |
| lora_inv strict (argmax) | 0.9036 | 0.9033 | −0.0003 |
| lora_inv deepnoc | 0.9271 | 0.9305 | +0.0034 |
| count accuracy | 0.9312 | 0.9361 | +0.0049 |
| oracle EM | 0.9650 | 0.9703 | +0.0053 |
| EM@k | 0.9125 | 0.9243 | +0.0118 |
| donor F1@k | 0.9738 | 0.9763 | +0.0025 |
| reject AUROC | 0.9972 | 0.9994 | +0.0022 |
| **open (ngoài panel)** | **0.2482** | 0.2028 | **−0.0454** |
| thời gian Phase 1 | 11.315 s (3,1 h) | **3.647 s (1,0 h)** | **−3,1 lần** |

**1. Không có nhãn thật thì generator mới thắng đậm: +0.1491.** Đây là hiệu ứng thật của generator, đo
tách bạch, với dữ liệu ít hơn 4 lần. Khớp với chẩn đoán trước đó: bộ cũ thiếu đuôi peak mờ.

**2. Sau khi fine-tune thì lợi thế gần như biến mất: +0.0036 với `expect`, −0.0003 với `argmax`.**
LoRA fit trên ~17–20 tổ hợp thật đã bù trọn khoảng cách miền. Đây là câu trả lời cho câu hỏi lớn: **dữ liệu
sinh tốt hơn KHÔNG nâng được kết quả cuối cùng khi đã có fine-tune.**

**3. Nhưng nó rẻ hơn 3 lần.** 12k hàng cho Phase 1 1,0 giờ thay vì 3,1 giờ, mà kết quả cuối bằng nhau.
Với chiến dịch 10 fold, đó là ~10 giờ GPU thay vì ~32 giờ.

**4. ID nhích lên đều ở mọi chỉ số** (oracle EM +0.005, EM@k +0.012, donor F1@k +0.003, reject +0.002).
Nhỏ nhưng cùng chiều.

**5. Hai chỗ bộ mới KÉM hơn:**
- **open −0.0454** (0.2482 → 0.2028). Đáng lưu ý vì reject AUROC lại tốt hơn (0.9994 vs 0.9972): phát hiện
  donor ngoài panel thì tốt hơn, nhưng ĐẾM những hồ sơ đó thì kém hơn.
- **sum(gate) kém hơn rõ** (−0.10 macro F1 hỗn hợp). Gate vẫn đúng thứ tự (corr 0.90) nhưng lệch thang:
  đoán thừa NOC3/NOC4, thiếu NOC2/NOC5. Nhánh tier-B `w_gate` hiệu chỉnh kém trên phân bố mới.

**Khuyến nghị (BẢN CŨ, ĐÃ SAI — xem mục 20):** không cần sinh lại 10 fold chỉ để tăng điểm.

> ⚠️ **SỬA 2026-09-30:** kết luận trên chỉ đúng cho `data_ft_att`, và `data_ft_att` KHÔNG phải bộ dữ liệu
> đã tạo ra kết quả tốt của nhóm. Xem mục 20.

## 20. `data_ft_att` không phải bộ dữ liệu của lần chạy tốt (2026-09-30)

Notebook `st-noc.ipynb` của nhóm (out_subdir `inc22_v2`, fold 0, 150 epoch từ đầu) dùng dataset
`synznguyen/noc-insilico-w` với **train 55.247 hàng (50.000 hỗn hợp sinh)**, và đạt:

| | log của nhóm (`noc-insilico-w`) | ta chạy `data_ft_att` | ta chạy bộ CŨ |
|---|---|---|---|
| `sum(gate)` accuracy | **0.9288** | 0.6709 | 0.7181 |
| `sum(gate)` macro F1 hỗn hợp | **0.8632** | 0.3848 | 0.4876 |
| corr(sum gate, NOC) | **+0.9822** | +0.9001 | +0.9117 |
| ID @ k=sum gate | **0.913** | — | — |
| oracle EM | 0.975 | 0.9703 | 0.9650 |

**Dấu vân tay generator:** tỉ lệ peak qua feasibility filter (`enrich-after-feas keeps`) —
bộ CŨ **0.779**, `data_ft_att` **0.788**, `noc-insilico-w` **0.810**. `data_ft_att` nằm sát bộ cũ, còn bộ
cho kết quả tốt thì khác hẳn → **hai build khác nhau**. Khớp với thứ tự nghịch lý ở bảng trên: nếu cùng
generator thì `data_ft_att` không thể kém bộ cũ trên chính thước đo gate.

Khác biệt khác giữa hai pipeline:
- build 50.000 so với 12.000 hỗn hợp;
- bundle code của họ có `calibrate_crowd.py`, `calibrate_budget.py`, `preprocess.py` (generator mới +
  hiệu chỉnh); kit của ta chỉ có `make_insilico.py` bản cũ;
- trainer của họ **đọc `attrfrac`** (nhãn phân rã peak thật), trainer của ta không load file đó;
- họ lấy k từ `sum(gate)`, ta lấy từ LoRA. Chênh lệch nằm ở chất lượng gate (corr 0.9822 so với ≤0.912).

**Chỗ cần hỏi nhóm:** `RUN_10FOLD.md` mục 7.2 của họ viết "Ep 1 phải có `attr=` quanh 0.3; nếu 3.3 thì
nhãn lệch hàng". Log của chính họ có `attr=3.518` ở Ep 1, của ta `attr=3.192`. Theo tiêu chí họ tự đặt thì
**cả hai bên đều đang lệch nhãn attr**.

**Bài học lặp lại:** mục 19 khái quát từ một bộ dữ liệu mà mình tưởng là "bộ mới". Trước khi so, phải xác
minh bộ dữ liệu đúng là bộ đã tạo ra con số đang muốn tái lập — bằng một dấu vân tay đo được, không bằng tên
file.



## 21. Đã nhận bộ dữ liệu THẬT của họ + bundle code (2026-09-30)

`incoming/noc_bundle_2026-09-30/` — `data_insilico_w` (128 MB) và 22 file code của họ.

**Xác nhận đây đúng là bộ tạo ra kết quả tốt:** keep rate **0.811** (log của họ 0.810), train **55.247**
hàng = 50.000 hỗn hợp + 5.247 đơn nguồn thật, 39.653 tổ hợp train, 20 tổ hợp test, **CHUNG 0**.
Chưa có `tokens8` và `dev` nên pipeline của ta sẽ tự prep, đúng luồng.

**Generator của họ khác ta:**

| | ta | họ |
|---|---|---|
| `make_insilico.py` | 33,5 KB | **163 KB** |
| lệnh build | `--build 50000 --noc_weights 1,1.5,2.5,2 --seed 42` | **thêm `--conditioned`** |
| `AMOUNT_SD=0.37`, `KAPPA_MAX=0.26`, `RUN_COUPLE=0.36`, `NOISE_GAIN=0.070` | không có | có |
| `data/noise_cr_inc.json` | không có | có |
| script hiệu chỉnh | không có | `calibrate.py` (114 KB), `calibrate_budget.py`, `calibrate_crowd.py` |
| nhãn `attrfrac`/`attrcol` | không sinh, không đọc | sinh và dùng |

Chú thích trong `preprocess.py` của họ: **không có `--conditioned` thì build rơi về `gen_any` và không
luật nào tới được tập train.** Đây gần như chắc chắn là chuyện đã xảy ra với bộ `data_ft_att` 12k
(keep 0.788, sát bộ cũ 0.779).

**Thêm cách đếm `gate_cal` vào pipeline của ta (CODE c324f5b24e).** `sum(gate)` có thứ tự tốt
(corr 0.90–0.98) nhưng lệch thang, nên `round()` làm mất gần hết thông tin đó. `gate_cal` khớp ánh xạ
một biến `sum(gate) → k` bằng hồi quy logistic **trên DEV in-silico**, tức không dùng nhãn hỗn hợp thật
nào. Đây là cách kiểm luận điểm của họ trong khung đo của ta: nếu `gate_cal` đạt ~0.93 thì **toàn bộ tầng
LoRA là không cần thiết** và pipeline hết phụ thuộc nhãn thật. Vì nó không fit trên test, arm này bỏ qua
lượt deepnoc (strict và deepnoc là cùng một số).

Cell chạy: `kaggle_upload/cells/cell_bo_cua_ho_fold0.py` (~3,5 giờ). Nó so bốn cách đếm và đặt cạnh ba bộ
dữ liệu sinh đã đo trên cùng fold 0.


## 22. BỘ DỮ LIỆU THẬT CỦA HỌ, PIPELINE CỦA TA — fold 0 (2026-09-30, CODE b352dafd85)

50.000 hỗn hợp `--conditioned`, Phase 1 150 epoch từ đầu, cùng 2.458 hàng test, cùng `combo_md5`.

| cách đếm | cần nhãn thật? | strict | deepnoc | open |
|---|---|---|---|---|
| **zero-shot `logits_card`** | **KHÔNG** | **0.9102** | — | — |
| `round(sum gate)` | KHÔNG | 0.7158 | — | — |
| `gate_cal` (hiệu chỉnh trên DEV) | KHÔNG | 0.6854 | — | **0.630** |
| **lora_inv** | CẦN | **0.9238** | 0.9519 | 0.198 |

**Ba bộ dữ liệu sinh, cùng fold 0, cùng pipeline:**

| bộ | zero-shot | gate acc | gate F1 hỗn hợp | lora strict | oracle EM |
|---|---|---|---|---|---|
| cũ 50k | 0.6153 | 0.7181 | 0.4876 | 0.9026 | 0.9650 |
| `ft_att` 12k | 0.7644 | 0.6709 | 0.3848 | 0.9062 | 0.9703 |
| **họ 50k `--conditioned`** | **0.9102** | 0.7339 | 0.7757 | **0.9238** | **0.9813** |

**1. Generator của họ tốt hơn ở MỌI chỉ số.** zero-shot +0.295 so với bộ cũ; lora +0.021; oracle EM +0.016.
Kết luận ở mục 19 ("dữ liệu tốt hơn không nâng được kết quả sau fine-tune") **SAI** — nó chỉ đúng cho bộ
`ft_att` 12k, vốn không có `--conditioned`.

**2. Fine-tune gần như hết cần thiết.** Với bộ cũ, LoRA mua được **+0.287** (0.6153 → 0.9026). Với bộ của
họ, chỉ còn **+0.0136** (0.9102 → 0.9238). Nghĩa là **có thể bỏ nhãn hỗn hợp thật** và chỉ mất 0.014.

**3. ID tốt nhất từ trước tới nay:** oracle EM **0.9813**, EM@k **0.9451**, donor F1@k **0.9841**,
reject AUROC **1.0000**. Ngưỡng tin cậy: acc 0.95 khi phân loại **99,8%** hồ sơ (bộ cũ: 95,5%).

**4. `gate_cal` THẤT BẠI (0.6854 < gate thô 0.7158).** Hiệu chỉnh `sum(gate)→k` học trên DEV in-silico
không chuyển được sang hàng thật: thang gate lệch giữa hai miền. Giữ arm lại nhưng tắt mặc định; phương án
không nhãn đúng đắn là **zero-shot `logits_card`**, không phải gate.

**5. PHÁT HIỆN ĐÁNG GIÁ NHẤT — đếm off-panel:** trên 1.366 hàng có donor ngoài panel, `gate_cal` đạt
**0.630** còn `lora_inv` chỉ **0.198**. Hợp lý: gate đếm slot, không đọc danh tính 45 donor. Mà reject
AUROC lại **bằng 1.0**, tức tách hoàn hảo hồ sơ trong panel với ngoài panel.
→ **Định tuyến theo reject head**: hàng trong panel dùng LoRA, hàng ngoài panel dùng gate. Đây là cách sửa
điểm yếu open đã đeo bám suốt dự án, và mọi thành phần đều đã có sẵn.

**6. Gate của ta vẫn kém gate của họ trên cùng dữ liệu** (acc 0.7339 so với 0.9288). Trainer khác nhau, nên
phần đó chưa giải thích được — nhưng không còn quan trọng, vì `logits_card` của ta đạt 0.9406 acc.


## 23. HAI CÁCH ĐỌC SỐ NGƯỜI, CÙNG MỘT BACKBONE (fold 0, bộ 50k của synznguyen, 2026-09-30)

Cùng `fp(logits_cls) = 6b61691655` với lần chạy trước, tức **mô hình bit-identical**; chỉ đổi cách đọc ra k.

| cách đọc | macro F1 | macro F1 hỗn hợp | acc | acc hỗn hợp | gọi thiếu | gọi thừa |
|---|---|---|---|---|---|---|
| **`logits_card`** (Linear 45→5, đọc cả vector gate) | **0.9014** ¹ | **0.8773** | **0.9349** | 0.8837 | 108 | 47 |
| `round(sum gate)` (chỉ đọc tổng) | 0.7158 | 0.6937 | 0.7339 | 0.7824 | 142 | 148 |
| *nhánh synznguyen, `sum gate`* | — | *0.8632* | *0.9288* | *0.8702* | *37* | *136* |

¹ với `expect` (cách giải mã đang ship). Cùng arm đó dùng `argmax` cho **0.9102** trên 5 lớp; macro F1
**hỗn hợp** dưới `argmax` thì lần chạy này KHÔNG lưu, nên không có số để so — đã sửa code
(CODE a17a9269b1) để từ nay lưu per-NOC, mix và accuracy cho **cả hai** cách giải mã.

**1. Trong mô hình của repo này, đọc vector 45 chiều hơn đọc tổng rất xa: +0.184 macro F1.**
Kiểu hỏng của gate rất đặc trưng: **NOC1 recall chỉ 0.676** — 355 trên 1.125 hồ sơ đơn nguồn bị gọi thành
2 người, vì tổng gate bị thổi phồng. `logits_card` không mắc lỗi đó (NOC1 recall 0.996).

**2. Nhưng gate của nhánh synznguyen KHÔNG hỏng như vậy** (NOC1 recall 0.998, mix macro F1 0.8632).
→ Không phải "đọc tổng là dở", mà là **đầu nào được dạy thì đầu đó chạy**. Repo này dồn tín hiệu đếm vào
`logits_card` (`beta_card=0.3`), nhánh kia dồn vào `sum(gate)` (`w_gate=0.05`, không có `beta_card`).
Mỗi bên đạt ~0.86–0.88 mix macro F1 ở đầu được ưu tiên, và đầu còn lại thì kém hẳn.

**3. So sòng phẳng, cùng không nhãn thật:** ta 0.8773 (`logits_card`) so với họ 0.8632 (`sum gate`),
hơn **+0.0141**. Khác biệt nhỏ hơn nhiều so với vẻ ban đầu.

**4. Thiên lệch ngược nhau:** ta gọi thiếu (108 thiếu / 47 thừa), họ gọi thừa (37 / 136). Gate của ta thì
cân bằng nhưng sai nhiều (142 / 148).

**5. Giải mã phải chọn theo từng arm, không đặt chung.** `expect` giúp LoRA và clone (+0.004…+0.007) nhưng
**hại `zero_shot`** (0.9014 so với 0.9102 của `argmax`). Posterior của `logits_card` đã sắc nét nên lấy kỳ
vọng chỉ làm mờ. Nếu ship `zero_shot` thì phải dùng `--count_decode argmax`.
