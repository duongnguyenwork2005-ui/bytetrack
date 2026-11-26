# FPS và chất lượng tracking trên 80 video

Đo xong ngày 17/11/2026 trên Jetson Nano. So trực tiếp v21combo với YOLO11n 50/50, lọc hộp trước NMS, NMS ma trận và giải mã JPEG gối đầu. Cùng 150 ảnh JPEG từ mỗi video, 40 video BDD100K val và 40 video UA-DETRAC test, TensorRT FP16 384×640, ByteTrack KF c25, tắt hiển thị. Đổi thứ tự hai cấu hình theo từng video. [Giao thức](00_protocol.md)

## Tốc độ

- **Cả 80 video:** FPS trung bình **13,12 → 18,63**, tăng **41,99%**. FPS tính từ tổng khung/tổng thời gian đo ổn định **13,07 → 18,59**. Trung vị mức tăng ghép cặp **42,52%**. Cấu hình tối ưu nhanh hơn trên cả 80 video và đạt **≥15 FPS ở 80/80**, so với **1/80** của v21combo. Thấp nhất bản tối ưu **16,66 FPS**.
- **BDD100K:** FPS trung bình **13,01 → 19,43** (+49,35%). Trung vị **12,92 → 19,44**. Trung vị p90 theo video **90,50 → 119,44 ms**.
- **UA-DETRAC:** FPS trung bình **13,23 → 17,82** (+34,76%). Trung vị **13,42 → 17,92**. Trung vị p90 theo video **89,30 → 127,58 ms**.

FPS dùng cửa sổ ổn định sau 25 khung đầu. Trung vị p90 trên toàn bộ 80 video **90,08 → 123,88 ms**. P90 tính từ lúc bắt đầu đọc khung đến lúc tracker xử lý xong, gồm thời gian chờ của ảnh JPEG đã giải mã trong cấu hình mới. Chưa đo camera trực tiếp hoặc HDMI. Launcher chưa được chuyển.

## Đầu ra tracking

- **BDD100K:** HOTA **33,51% → 43,72%**, IDF1 **38,96% → 52,97%**, IDSW **971 → 1.534**, FN **37.403 → 28.539**.
- **UA-DETRAC:** HOTA **62,81% → 66,57%**, IDF1 **73,72% → 78,37%**, IDSW **280 → 245**, FN **22.287 → 17.590**.

Đây là điểm gộp TrackEval của các đoạn đã đo. Phép thử đổi cả detector và cách xử lý luồng ảnh. Trong R108, khi giữ cùng detector trên 16 đoạn, giải mã gối đầu cho MOT và lớp trùng byte với cấu hình NMS ma trận. [Đối chiếu R108](../review108_followup/nano_evidence/decode_overlap/score_verified.json)

## Kiểm tra kết quả

Đủ 160/160 lượt, 24.000 khung suy luận, mỗi lượt đủ 150 khung. Đối chiếu đúng 80 dòng manifest R86, cùng hash ảnh cho hai lượt mỗi video, xác minh engine, kiểm chuỗi timestamp, tính lại FPS/p90 và đối chiếu hash các tệp MOT/lớp. Cùng boot trong thời gian đo, lỗi thẻ 0 và swap 83,30 → 127,50 MiB. Mã SHA256 gói bằng chứng `24ca388d8c5d4e06d58a8c6b764fa123f2f0e0130c8a46fa13b04270e64ef217`. [Số liệu tổng hợp đã xác minh](results/score_verified.json) · [Log Nano](run.log)
