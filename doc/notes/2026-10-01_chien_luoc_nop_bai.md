# Chiến lược nộp bài — đánh giá mức sẵn sàng, 2026-10-01

Doc này để đọc lại từ phiên khác. Mục đích: trả lời "bài này nộp journal được chưa, và còn thiếu gì".

Tạp chí đích đã chọn: **Forensic Science International: Genetics** (Elsevier).
Overleaf (read-only): https://www.overleaf.com/read/gmqrkjypdqgr#48a749 — project `DNA_NOC_Latex`.

**Kết luận ngắn: CHƯA nộp được.** Không phải vì kết quả yếu, mà vì **cách đo chưa nhất quán**.

---

## 1. Vấn đề lớn nhất — bốn thế hệ số, không so được với nhau

| ghi ngày | số | cấu hình | còn tái lập được không |
|---|---|---|---|
| **07/08** *(abstract Overleaf **hiện tại**)* | 0.9374 ± 0.0118 | 5-fold, **trước** khi có bảo vệ rò rỉ | không — protocol đã bị thay |
| **10/09** | **0.951 ± 0.009** vs 0.910 | ensemble + `posthoc_rf`, chỉ 25s, `pooled_kfold`, 5 seed (42–46) | **KHÔNG** — xem §2 |
| **25/09** | 0.9258 ± 0.0087 (deepnoc)<br>0.8956 ± 0.0156 (strict) | `lora_inv`, **3 fold**, một model, hai giao thức | có |
| **30/09** | 0.9238 (strict) / 0.9519 (deepnoc) | fold 0 **một fold**, dữ liệu 50k `--conditioned` | có, nhưng backbone đã mất |

Bốn dòng khác **protocol, dữ liệu, số fold, số seed**. Không dòng nào thay được dòng nào.

Câu đầu tiên reviewer FSI:Genetics sẽ hỏi: *"cho tôi MỘT công thức, N fold, mean ± sd."*
**Ta chưa có.** Đây là thứ duy nhất đang chặn đường nộp.

---

## 2. Con số headline 0.951 phụ thuộc code đã bị bỏ

`0.951 ± 0.009` (10/09) là **ensemble có `posthoc_rf`**. Nhưng:

- `posthoc_rf` đã bị **đóng hai lần** (mục 14: ensemble kéo xuống −0.0137 strict / −0.0258 deepnoc)
- `clone` / `clone_inv` đã bỏ theo quyết định 01/10
- cả hai file (`noc_posthoc.py`, `noc_clone.py`) đã chuyển vào `archive/2026-10-01_truoc_clean/`

→ **Giữ claim 0.951 thì phải khôi phục code và chạy lại. Bỏ thì cần headline mới.**
Phải quyết chuyện này trước khi viết abstract.

---

## 3. Claim nên chốt

**Đừng** lấy headline là "vượt deepNoC". Nó phụ thuộc protocol, reviewer bẻ được ngay bằng một câu hỏi.

Headline đề xuất — ghép hai thứ, không tranh hơn thua:

> Một mô hình đóng băng vừa đếm số người góp vừa định danh họ, **không dùng một nhãn hỗn hợp thật nào**;
> và giao thức đánh giá của công trình trước **thổi phồng kết quả tỉ lệ với số tham số được fine-tune**.

### Bốn đóng góp, xếp theo độ mạnh

| | đóng góp | bằng chứng | sẵn sàng |
|---|---|---|---|
| **1** | **Premium do dùng lại tổ hợp**: cùng một model, chuyển từ tách-tổ-hợp sang giao thức công bố thì được **+0.030** với adapter 26k tham số và **+0.079** với clone 1,5M. Tỉ lệ **2,6×**, 3 fold, không fold nào đảo dấu | đã đo | gần đủ, cần thêm fold |
| **2** | **Một model hai tác vụ**, một lượt forward: `logits_cls` → ID, `logits_card` → NoC | đã đo | đủ |
| **3** | **Không cần nhãn hỗn hợp thật**: NoC 0.9102, ID EM@k 0.9251 (alpha khớp trên DEV in-silico) | đã đo, **một fold** | **thiếu fold** |
| **4** | Bảng 3 của deepNoC in one-vs-rest chứ không phải recall | dựng lại khớp **15/15 chữ số** | cần kiểm độc lập |

Đóng góp **1 mạnh nhất về phương pháp luận** — nó định khung lại toàn bộ so sánh thay vì tranh thắng thua.

---

## 4. Số nào vào mục nào, số nào phải chạy lại

| mục của bài | số cần | hiện có | phải làm |
|---|---|---|---|
| Abstract | macro F1 ± sd, N fold, N seed | **không có** | **chạy 10 fold × ≥3 seed một công thức** |
| Kết quả chính (strict) | mean ± sd theo fold | 0.8956 ± 0.0156 (3 fold) | mở rộng lên 10 fold |
| So với deepNoC (giao thức họ) | mean ± sd | 0.9258 ± 0.0087 (3 fold) | mở rộng lên 10 fold |
| **Bảng premium** | strict vs deepnoc cùng model | +0.030 / +0.079 (3 fold) | mở rộng; đây là đóng góp 1 |
| Zero-shot / không nhãn thật | macro F1 + ID EM@k | 0.9102 / 0.9251 (**1 fold**) | **chạy nhiều fold** |
| ID | oracle EM, EM@k, donor F1@k | 0.9813 / 0.9451 / 0.9841 (1 fold) | nhiều fold |
| Open-set | acc ngoài panel | `gate_cal` 0.630 vs `lora_inv` 0.198 (1 fold) | nhiều fold |
| Reject | AUROC | 1.0000 (1 fold) | nhiều fold |
| Ablation cách đếm | 5 head cùng backbone | **đang chạy** (`lora_mac` / `lora_free` / `corn`) | chờ |
| Mô tả dữ liệu sinh | thống kê generator | keep rate, phân bố NOC | **xem §5 — vấn đề** |

---

## 5. Hai thứ phải xử lý trước khi viết Methods

### 5a. Bộ dữ liệu sinh — ta KHÔNG tái lập được

Ảnh hưởng của generator là khổng lồ: zero-shot đi từ **0.6153 lên 0.9102 chỉ vì đổi bộ sinh**, cùng
fold, cùng code, cùng test.

Nhưng kit của ta **không dựng lại được** bộ `--conditioned` của synznguyen — thiếu `calibrate.py`
(114 KB), `calibrate_budget.py`, `calibrate_crowd.py`, và `make_insilico.py` của họ là 163 KB so với
33,5 KB của ta.

**Bài báo không thể mô tả một dữ liệu mà nhóm không tự dựng lại được.** Phải lấy đủ bộ script sinh,
hoặc chấp nhận báo trên bộ ta tự dựng (kém hơn hẳn).

Thêm: lần chạy 01/10 xác nhận bộ 12k của Nguyên vẫn là build **không `--conditioned`**
(`enrich-after-feas keeps 0.787`, so với 0.811 của bộ tốt). Cờ đó vẫn chưa được bật khi sinh.

### 5b. Nhãn `attr` — cả hai bên đều lệch theo tiêu chí của chính họ

`RUN_10FOLD.md` mục 7.2 của nhóm viết *"Ep 1 phải có `attr=` quanh 0.3; nếu 3.3 thì nhãn lệch hàng"*.
Log của họ: `attr=3.518`. Log của ta: `attr=3.192` (và 3.199 ở lần 01/10).
**Theo tiêu chí họ tự đặt thì cả hai bên đều đang lệch.** Chưa ai trả lời. Phải làm rõ trước khi nộp.

---

## 6. Ba câu reviewer chắc chắn hỏi

**"Chỉ 20 tổ hợp donor thật mỗi fold?"**
Đây là ràng buộc gắt nhất của cả dự án, và nó giải thích vì sao ngân sách fine-tune 5 epoch là tối ưu
(nhiều hơn là học thuộc). **Nêu thẳng trong Limitations**, đừng để reviewer tìm ra.

**"Dữ liệu sinh của các anh có hợp lệ không?"**
Cần một mục so phân bố in-silico với thật. Công cụ đã có: chính `logh_p10` chỉ ra generator cũ thiếu
đuôi đỉnh mờ (24 RFU so với 9 RFU thật), và đó giải thích vì sao nó phá biên giới NOC4/5.

**"PROVEDIt Filtered hay UnFiltered?"**
Đã quyết dùng Filtered và không đề xuất chạy lại UnFiltered. Phải nói rõ và nói vì sao.

---

## 7. Cách viết phần về bảng của deepNoC

Đây là **đính chính một bài đã xuất bản**, nên phải viết rất cẩn thận.

Số đã dựng lại từ chính ma trận nhầm lẫn Table 2 của họ, khớp **cả 15 giá trị**:

| | 1p | 2p | 3p | 4p | 5p | macro |
|---|---|---|---|---|---|---|
| họ in trong Table 3 (**one-vs-rest**) | 1.000 | 0.992 | 0.965 | 0.911 | 0.927 | 0.9591 |
| **recall chuẩn** | 1.000 | 1.000 | 0.873 | 0.882 | 0.782 | **0.9074** |

Kiểm độc lập: với 5 lớp và accuracy đa lớp `a`, trung bình one-vs-rest phải là `1 − 2(1−a)/5`.
Thay `a = 334/372` ra **0.9591**, khớp đúng.

**Cách viết nên dùng:** trình bày như *làm rõ để so sánh trên cùng thước đo*, không phải tố lỗi. Và
**nêu luôn rằng cột F1 của họ ĐÚNG** — F1 đối xứng theo precision/recall nên việc đảo nhãn hai cột
không ảnh hưởng; dựng lại ra `[1.000, 0.983, 0.914, 0.832, 0.819]`, macro **0.9097**. Việc ta chỉ ra
cả chỗ đúng cho thấy đọc kỹ chứ không bới móc.

→ **Mốc dùng để so là macro F1 0.9097.**

Trước khi nộp: nhờ một người trong nhóm **dựng lại độc lập**.

---

## 8. Phần công bố bắt buộc — giữ nguyên, không được bỏ dòng nào

Mọi so sánh với deepNoC phải kèm:

- Không cùng tập hồ sơ cá thể (666 so với 675 hỗn hợp)
- Panel 45 bộ gen tham chiếu
- Phase 1 train trên 5.171 hồ sơ NOC1 **thật**
- Đầu vào là bảng đỉnh **Filtered** của PROVEDIt, không phải EPG thô như deepNoC
- Reject head **có** thấy hồ sơ ngoài panel (không kèm nhãn đếm)
- Backbone không có fold stamp (với các lần chạy cũ)
- Kit: deepNoC ghi GlobalFiler / ABI 3500 / 25s nhưng **không ghi số chu kỳ**. `3500_GF29cycles` là
  cấu hình GlobalFiler duy nhất trong archive PROVEDIt 1-5 người, nên coi là cùng kit.

---

## 9. Việc tiếp theo, theo thứ tự

1. **Chờ thí nghiệm `lora_mac` / `lora_free` / `corn`** xong → chốt cách đếm
2. **Quyết chuyện 0.951**: khôi phục `posthoc_rf` + `clone` và chạy lại, hay bỏ claim đó
3. **Chạy 10 fold × ≥3 seed một công thức duy nhất** ← thứ duy nhất chặn đường nộp
4. Lấy đủ script sinh dữ liệu từ nhóm, hoặc chấp nhận báo trên bộ ta tự dựng
5. Làm rõ chuyện `attr` lệch
6. Thêm baseline MAC (và probabilistic genotyping nếu làm được)
7. Cập nhật abstract Overleaf — số hiện tại ở đó là của **07/08**, đã lỗi thời hai thế hệ

---

## Liên quan

- [2026-10-01_thiet_ke_noc_tren_lora.md](2026-10-01_thiet_ke_noc_tren_lora.md) — thiết kế phần đếm
- [2026-10-01_dem_khong_neo_vao_id.md](2026-10-01_dem_khong_neo_vao_id.md) — số liệu và bằng chứng
- `doc/deepNoC.md` — dựng lại bảng của deepNoC, đầy đủ 4 cảnh báo
- `doc/failed_experiments.md` — mọi thí nghiệm đã chạy, mục 10–23
