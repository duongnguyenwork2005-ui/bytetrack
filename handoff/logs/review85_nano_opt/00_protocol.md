# R85 — khảo sát tối ưu tiếp cho Jetson Nano (05/11/2026)

Nguồn giao việc: người dùng yêu cầu **thử tất cả các cách khả thi** rồi thống kê ngắn gọn. Đây là ủy quyền mở vòng thử nghiệm mới; kết quả cũ và cấu hình luận văn không bị sửa hồi tố.

## Điểm xuất phát

- Launcher đang dùng `v21combo`, engine C1 384x640 FP16; R83 đã xác minh LUT + giảm copy trùng MOT/lớp và nhanh hơn trong phép đo ghép cặp.
- C2 352x576, C3 288x512, C4 256x448 mới có thời gian engine, chưa có chất lượng hay tốc độ pipeline được chấp nhận.
- Mục tiêu chất lượng: xe nhỏ, xe máy và vật bị che khuất. Không chọn ứng viên chỉ vì FPS.

## Thử nghiệm và thứ tự

1. C2 trên dev: cùng checkpoint `models/yolov8n.pt`, cùng split UA-DETRAC dev12 và BDD dev139, cache conf 0.05; đối chứng là C1 **và** vuông 640. Chấm CV/EKF c25, AP và recall theo cỡ/che khuất. Kiểm ánh xạ ảnh và kích thước tensor 352x576. C1 là đối chứng triển khai; vuông 640 là đối chứng chất lượng R63. Giữ cache, tracker, metric, GT, FPS và ngưỡng NMS của R63. c10 chỉ làm phân tích nếu c25 qua cổng.
2. Cổng so với C1: HOTA, IDF1, AP50, AP50-95 mỗi miền giảm không quá 0,5 điểm phần trăm; IDSW không tăng quá 5%; recall xe nhỏ và che khuất giảm không quá 2 điểm phần trăm, tầng thiếu GT ghi rõ. Nếu trượt, không triển khai C2 và không mở C3/C4 theo quy tắc R63. Nếu qua, kiểm Nano-PC và đo pipeline ghép cặp CV/EKF, FPS trung vị tăng ít nhất 5%, p90 không tăng quá 5%, không vi phạm giám sát.
3. NMS ma trận `v22nms` giữ nguyên engine C1 và tensor: trước tải kiểm parity decode trên PC/Nano, sau đó 4 cặp cân bằng A/B cho CV và EKF, 300 JPG có HDMI mỗi lượt. Mọi cặp phải đủ 300 frame, replay/live và MOT/lớp A/B trùng byte. Cổng tốc độ: trung vị tăng ≥5% cho cả CV/EKF, p90 trễ không tăng >5%, không vi phạm giám sát; chỉ khi đạt mới cân nhắc chuyển launcher.
4. Truyền bộ nhớ CUDA: trước hết probe pageable/pinned đúng kích thước I/O C1; nếu có lợi, dùng bản sao gói riêng chỉ đổi buffer host (`v23pin`), thử output-only và input+output pinned trên 300 frame. Parity detection/MOT/lớp phải trùng byte với `v21combo`. Sau đó chỉ ứng viên tốt nhất (có thể ghép NMS) mới mở 4 cặp CV + 4 cặp EKF HDMI theo cùng cổng ≥5% FPS trung vị, p90 ≤+5%, không vi phạm giám sát. Không đổi launcher khi chưa qua cổng.
5. Đánh giá các cách khác theo nút thắt và độ an toàn: prefetch/offline, pipeline decode, TensorRT, pruning/distillation. Chỉ triển khai ứng viên qua cổng tốc độ, chất lượng và ổn định. Không thử ép xung, thay xung SD hay INT8 không hỗ trợ phần cứng.

## Tái lập và dừng

- Mọi đầu ra mới dùng tên `r85*`, không ghi đè kết quả R63/R83; lưu lệnh, hash, log, version và kết quả âm.
- Chỉ dùng dev để chọn. Test40/BDD val chỉ báo sau khi chốt ứng viên.
- Nano trước tải: kiểm bộ đếm lỗi thẻ, swap, RAM, nhiệt, tiến trình khác. Cổng ổn định R63 vẫn áp dụng; trượt thì không đo tải Nano.
- Dừng khi kernel panic/softlockup, lỗi SD mới, nhiệt quá 80 C, swap tăng trong lượt, frame thiếu hoặc parity sai.
