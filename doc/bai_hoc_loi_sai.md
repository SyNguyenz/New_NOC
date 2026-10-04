# Bài học: các lỗi Claude đã mắc trong dự án NoC (DS-ST vs deepNoC)

Mỗi mục ghi: **lỗi**, **ai/cái gì phát hiện**, **luật từ nay**. Đọc file này trước khi kết luận hoặc viết vào paper.
Skill đi kèm: `.claude/skills/noc-deepnoc/SKILL.md`.

## A. So sánh không công bằng

1. **Khoe LoRA 0.9201 "vượt deepNoC" khi đã fine-tune ~2.000 nhãn.**
   Mạnh bắt: "deepNoC finetune có 376 thôi".
   → Luật: mọi fine-tune ≤ 371 nhãn thật (34/87/79/93/78), rút chỉ từ phía fit. Ghi số nhãn cạnh MỌI con số fine-tune.

2. **So `noc_deep` train 6k/20k với backbone train 50k.**
   Mạnh bắt: "phải so tất cả cùng train 20k chứ?".
   → Luật: so các đầu đếm ở CÙNG cỡ dữ liệu train (ví dụ cùng 12k), cùng seed, cùng fold.

3. **So zero-shot bằng hai cách giải mã khác nhau** (argmax 0.9102 vs expect 0.7488) rồi kết luận về dữ liệu.
   → Luật: cùng một cách giải mã khi so. Zero-shot = argmax (đã chốt trước); fine-tune = expect.

4. **Chọn head chỉ tối ưu trong panel**, chọn `head_only` mà không nói rõ nó đọc xác suất ID.
   Mạnh phản đối: NoC phụ thuộc ID.
   → Luật: khi chọn head, nói rõ nó đọc gì (gate, xác suất ID, …). Phạm vi đã chốt 01/10: chỉ trong panel, nhưng phải ghi công khai.

## B. Đọc sai số liệu hoặc paper

5. **Đọc nhầm số MAC hoặc recall là recall của deepNoC, nói "deepNoC recall yếu".**
   Mạnh bắt: "Recall nó có yếu đâu, m sai số liệu à".
   → Luật: Table 2 của deepNoC có hàng = dự đoán, cột = thật; hai cột Precision/Recall bị đổi nhãn.
   Recall thật là 1.000/1.000/0.873/0.882/0.782. Chỉ dùng cột F1 (đúng). Ghi rõ nguồn của từng con số.

6. **Nói deepNoC "coi mọi người là người lạ" hoặc có lợi thế ngoài panel.**
   Mạnh bắt: "làm gì có người lạ trong deepNoC đâu, m linh tinh".
   → Luật: deepNoC chia cách một, tổ hợp donor dùng chung giữa fit và test. Đọc lại mục 2.6 của paper trước khi mô tả giao thức.

7. **Nói "test lớn hơn = khó hơn".**
   Mạnh bắt: "Chấm nhiều hơn đâu có nghĩa là dễ hơn".
   → Luật: cỡ test chỉ ảnh hưởng độ rộng CI, không phải độ khó. Độ khó phải chỉ ra bằng thứ khác (tách tổ hợp, 5/15 s, …).

8. **Hiểu "NOC5 lấy hết" thành không còn NOC5 để test.**
   Mạnh bắt: "fine tune mà noc 5 lấy hết rồi, thì còn gì mà test nữa".
   → Luật: khi mô tả chia dữ liệu, luôn ghi số hồ sơ fit / test theo từng lớp; "lấy hết" là lấy hết của NỬA FIT.

9. **Kết luận `ft_att` 12k "không có cờ --conditioned" dựa vào keep rate.**
   → Luật: keep rate không phân biệt được (là hệ quả của tỉ lệ đơn nguồn). Kiểm bằng dấu hiệu RFU/bins theo NoC.

10. **Ghi head 1.323 tham số trong doc**, thực ra có 230 tham số của một bản `noc_head` chết. `pipeline_noc_id.md` để cũ.
    → Luật: đếm tham số bằng code (chỉ phần có gradient), cập nhật doc ngay khi pipeline đổi.

## C. Đổi phạm vi mà không được phép

11. **Đề xuất sửa dữ liệu** (lọc nhãn, sinh thêm, đổi generator, chỉnh theo tập test).
    Mạnh cấm: "không chỉnh dữ liệu theo tập test", "dữ liệu tốt rồi, giờ sửa model thôi".
    → Luật: chỉ sửa model, loss, head, cách train. Không đụng dữ liệu sinh.

12. **Đề xuất đầu đếm pma(z)** trong khi đã có bằng chứng z mã hoá "AI" (danh tính).
    → Luật: trước khi đề xuất, kiểm lại kết quả cũ trong `doc/failed_experiments.md`; không đề xuất lại hướng đã bị bác.

13. **NoC phải là nhánh riêng, không ảnh hưởng ID.**
    → Luật: backbone đóng băng khi train đầu NoC; log `[ID GUARD] params_unchanged=True` mỗi lần.

## D. Thống kê và khái quát vội

14. **Coi chênh ±0.02 trên 1 seed là cải tiến.** Chạy 5 seed thấy sd 0.01–0.02.
    → Luật: ≥ 3 seed (tốt hơn 5), báo TB ± sd; chênh nhỏ hơn 2 sd là không có kết luận.

15. **Một điểm siêu tham số được dùng để đóng một hướng.**
    → Luật: thử ít nhất vài điểm; phân tán lớn giữa các fold là dấu hiệu học thuộc, không phải "hướng tệ".

16. **Sửa D2 bằng Dirichlet(1) nhưng đại lượng nhắm tới không đổi** (phần người phụ vẫn ~0.03).
    → Luật: trước khi đo F1, kiểm xem thay đổi có thực sự làm dịch chuyển đại lượng mục tiêu không.

17. **Kết quả sau khi xem số (sim-init, dòng 25 s) dễ bị viết như kết quả chính.**
    → Luật: chỉ thứ ghi trong kế hoạch trước khi chạy mới là kết quả chính; còn lại ghi "phân tích sau, không chốt trước".

## E. Lỗi kỹ thuật khi chạy

18. Đặt biến `path` trong zsh làm hỏng PATH → dùng tên khác (`fp`).
19. macOS không có `timeout` → dùng vòng lặp chờ hoặc tham số timeout của tool.
20. Cross-entropy nhận nhãn sai dtype → ép `int64`.
21. Job nền chạy quá 30 phút bị dừng (simhead dừng sau fold 4) → chia job nhỏ theo fold, ghi kết quả từng fold ra file.
22. Lỗi cú pháp và parse tham số trong script tạm (`checks_d_m.py`, `fuse_eval.py`) → chạy smoke 1 fold trước khi chạy hết.
23. **MPS crash ("buffer is not large enough") bị đổ cho dữ liệu lớn, rồi định chuyển sang CPU.**
    Mạnh nhắc: chạy lượng data nhỏ. Thực tế lỗi xảy ra cả ở batch 8, do truy vấn `expand()` (không liền bộ nhớ) trong ISAB/PMA.
    → Luật: giữ mẫu nhỏ (≤ 6k), và khi MPS lỗi thì thu nhỏ để tìm đúng op gây lỗi trước khi đổi thiết bị.
    Sửa: `.contiguous()` chỉ khi tensor ở MPS, nên CPU/CUDA giữ nguyên từng bit (`.contiguous()` trên CPU làm `logits_cls` lệch 8e-4).
24. **Ghi file bằng Python làm mất CRLF**, diff thành cả file (1.305 dòng).
    → Luật: `code/models/set_transformer.py` dùng CRLF. Sửa bằng Edit, hoặc đọc/ghi với `newline=''`. Kiểm `git diff --stat` sau mỗi lần sửa.
