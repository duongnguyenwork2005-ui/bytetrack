# R83 — khảo sát các hướng tăng tốc tiếp trên Nano

Ngày 04/11/2026; theo yêu cầu người dùng thử các cách khả thi và báo thống kê ngắn. Baseline là C1 đã triển khai, engine 384×640 `a6912393…`, CV/EKF c25; không thay thuật toán tracking, ngưỡng hay metric. Mọi ứng viên ở gói riêng và mặc định tắt; launcher C1 chỉ đổi nếu cổng sau đạt.

Thứ tự: (1) copy hậu xử lý R82 đã thử, giữ kết quả âm; (2) đọc trước frame bằng worker và queue tối đa 2, giữ đủ frame/thứ tự, thời gian chờ tính vào trễ; (3) LUT tiền xử lý R62 và kết hợp với (2) nếu (2) qua smoke; (4) đường video MP4 với giải mã phần cứng nếu có MP4 phù hợp trên Nano; (5) C2–C4 chỉ là ứng viên đổi kích thước nội dung, đòi cổng chất lượng PC riêng trước triển khai; (6) pruning/distillation, INT8, CUDA graph chỉ khảo sát tính khả thi, không tính là FPS đạt khi chưa có model/benchmark tương ứng.

Smoke mỗi ứng viên 300 JPG CV tắt hiển thị: đủ frame, MOT/lớp trùng byte C1, không lỗi, swap/mmc không tăng bất thường. So tốc độ dùng **cùng một đường ảnh**, cùng engine/boot, thứ tự đảo, ghi median FPS/p90 và decode/pre/infer/post. Nếu smoke tăng <5% hoặc nhiễu decode lớn hơn hiệu, dừng ứng viên trước ma trận HDMI. Ứng viên qua smoke mới mở 4 cặp CV + 4 cặp EKF HDMI, cùng nguồn và điều kiện, ngưỡng tăng FPS trung vị theo cặp ≥5% ở cả hai tracker, p90 không tăng >5%, parity MOT/lớp, không drop frame/swap/mmc. Chỉ sau cổng đó mới thay launcher; lưu bản cũ để hoàn tác.

Không lấy FPS của JPG làm FPS MP4; nếu không có MP4 nguồn phù hợp thì ghi chưa đo, không suy từ plugin GStreamer. C2–C4 giảm pixel nội dung (khác C1 bỏ đệm), phải kiểm HOTA/IDF1/IDSW/AP/recall xe nhỏ và che khuất trên dev trước khi xem là ứng viên triển khai.
