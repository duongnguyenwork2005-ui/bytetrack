# R92 — PF chuyển động thẳng/CTRV khi Lost trên UA-DETRAC

Ngày 06/11/2026. Theo phạm vi người dùng, UA-DETRAC là camera giao thông cố định; UAVDT/UTUAV không dùng chọn PF. Giao thức và cổng khóa trước ở `00_protocol.md`. Ứng viên opt-in `pf_lost_cv_fraction=0.5` là **mixture chuyển trạng thái hạt theo từng bước Lost**, không phải IMM đầy đủ hoặc thuật toán ghép cặp mới. ByteTrack association và detection cache giữ nguyên.

## Kiểm kỹ thuật

`src/r92_pf_modes/test_modes.py` đạt: giá trị xác suất ngoài [0,1] bị từ chối, nhánh đi thẳng cho dịch chuyển CV, nhánh CTRV giữ phép cũ, RNG từng track và K1 prior của `predict` đơn trùng `multi_predict` kể cả đảo thứ tự track. Khi opt-in tắt, **6/6 file MOT** trên UA smoke 2 video × 3 seed trùng SHA với R72; ba điểm TrackEval smoke gốc tái tạo đúng HOTA/IDSW. Cache, conf, lớp, fps, buffer, elapsed mode, evaluator không đổi.

## UA dev12, cùng detector và ba seed

Trên 12 video train dev đã quan sát trước, PF R72 motion2 → ứng viên:

- HOTA COMBINED trung bình seed **57,006% → 57,440%** (+0,435 điểm).
- IDF1 **68,361% → 69,356%** (+0,995 điểm).
- IDSW **697,0 → 642,33** (−54,67; −7,84%). Chín video giảm, ba tăng; bootstrap ghép cặp video 5.000 mẫu cho chênh IDSW [−103,7; −14,0], chỉ mang tính mô tả vì dev đã dùng nhiều lượt.
- FP **13.485 → 13.482,33**, gần như không đổi.

Mặc dù các chỉ số di chuyển theo chiều có lợi, cổng R92 yêu cầu IDSW ≤627,3 (giảm ≥10%) trước khi chọn và mở BDD như bước chọn cấu hình. Ứng viên chỉ đạt 642,33, nên **không chọn/không đổi YAML luận văn hay Nano**. `dev_gate.json`, `dev_scores.json` và từng raw fields ở `results/trackeval/DETRAC-all/r92-ua-dev-cv50-s{0,1,2}/` lưu bằng chứng. BDD được mở sau đó thành **R93 riêng** để trả lời câu hỏi chuyển miền camera gắn xe của người dùng, không hồi tố sửa cổng R92.

[Nghiên cứu mô hình đa chế độ](https://onlinelibrary.wiley.com/doi/10.1155/2012/870890) gợi ý kết hợp đường thẳng và rẽ khi không chắc chế độ chuyển động; không phải bằng chứng biến thể R92 có lợi tổng quát. R92 vẫn dùng một hộp trung bình hạt cho ByteTrack, nên chưa xử lý trực tiếp bất định ghép cặp khi nhiều xe ở gần nhau.
