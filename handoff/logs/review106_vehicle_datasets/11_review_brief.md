# R106 — bàn giao reviewer độc lập, chỉ đọc

Phạm vi: kiểm nguồn tác giả, tiêu chí chọn/GT, con số từ đầu ra cũ. Không chạy tracker/inference, không sửa file hoặc mở R107. Người dùng giao R106 trực tiếp; coder không có tài liệu để tự xác nhận review R105 đã hoàn tất.

Đọc `00_protocol.md`, `10_ket_qua.md`, `tier1_checks.json`; mã là `audit_r106.py`, `verify_evidence.py`, `check_gram.py`, `audit_gram_sample.py`, `fetch_sources.py`, `run_tier1.py`. Dữ liệu đủ kiểm tại `gt_cohort.json`, `combined_verified.csv`, `scenario_{summary,per_clip}.csv`, `verification.json`, `integrity_verified.json`. Không cần quét data/results; từng file đầu vào được liệt kê hash trong JSON.

Các điểm cần soi:

- Đây là hậu kiểm trên dữ liệu đã xem điểm; dùng cố định COCO c25, giữ cả kết quả âm, không tuyên bố đăng ký trước độc lập.
- GT-only phase chạy trước audit; quy tắc R99, không đổi cổng khi Sherbrooke/GRAM thiếu đoạn rẽ. Kiểm nguyên chuỗi trước phân tầng, không reset ID tại episode.
- 156 clip×filter CLEAR khớp; 12 raw COMBINED và sáu MOT RideSafe tái chấm. RideSafe ba COMBINED mới khác trung bình clip lịch sử, không so hai phép gộp như cải thiện thuật toán.
- Phân biệt RoundaboutHD COCO c25 IDSW rẽ 22/22/22 với mixed5 c25 lịch sử 28/32/29. HELMET 43 khung-xe rẽ không đại diện toàn mẫu.
- Visibility thật chỉ ở SVMOT. Các cột dummy của tập khác không có nghĩa không bị che. RideSafe stride5/ego-motion không phải camera cố định.
- Lineage 12 run, metadata cache/association/certificate/checkpoint đã kiểm; không rehash ảnh/NPZ. RideSafe không còn NPZ bất biến, chỉ MOT/XML/metadata và mã pilot hiện có.
- GRAM nguồn tác giả mới, gói nhãn công khai <25 MiB, chỉ 250 XML Urban1 được kiểm CRC, giữ nhãn park và kết quả không có rẽ. Không chạy code/kết quả baseline kèm kit; chưa có metric tracking GRAM. KITTI link tải yêu cầu login, Urban Tracker 404 được ghi rõ.
- Kiểm lỗi verifier ban đầu exit 1 đã sửa và HEAD GRAM 403 trước GET 200 trong hồ sơ. `tier1_integrity.log` là log tự kiểm cuối được capture; stdout GT/audit đầu không được redirect.

Mục 0 PLAN đã cập nhật trước báo cáo. Không có review R106 độc lập tại thời điểm bàn giao; orchestrator mở phiên reviewer riêng.
