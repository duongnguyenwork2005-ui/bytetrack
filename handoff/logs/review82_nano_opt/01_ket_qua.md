# R82 — thử giảm copy trong detector Nano (04/11/2026)

Mục tiêu: kiểm hai chỗ sao chép/cấp phát dư sau C1 mà không thay engine, tracker, detector threshold hoặc đầu ra. Ứng viên riêng `v18/src/nano_detector.py`: lấy `argmax` từ ma trận `p` đã lọc thay vì lọc `cls_scores` lần nữa; `astype(np.float32, copy=False)` để dùng output float32 của TensorRT trực tiếp cho `decode_nms`. Buffer output chỉ hợp lệ đến lần inference kế tiếp; luồng hiện tại giải mã/NMS trước khi gọi inference tiếp.

PC: `test_decode_parity.py` đạt 18/18 trường hợp NMS trùng chính xác C1. Nano: gói độc lập `~/nano_run/r63_20261030/v18copy/nano_transfer`; engine 384×640 `a6912393…`. Smoke CV 300 JPG đủ frame; live/replay MOT trùng byte; tệp MOT và lớp ứng viên trùng byte với C1. Sau chạy, `mmcblk1p1/errors_count=0`.

| Lượt | Nguồn ảnh | FPS | Decode trung vị | Infer trung vị | P90 trễ |
|:--|:--|--:|--:|--:|--:|
| B đầu (`v18copy_smoke_20261104`) | bản sao ảnh B | 13,73 | 11,57 ms | 44,82 ms | 82,48 ms |
| A1 (`v18copy_pair_c1_cv_20261104`) | ảnh C1 | 12,44 | 19,78 ms | 45,45 ms | 91,65 ms |
| B chung nguồn (`v18copy_shared_cv_20261104`) | symlink đến ảnh C1 | 12,69 | 17,89 ms | 44,89 ms | 90,35 ms |
| A2 (`v18copy_pair2_c1_cv_20261104`) | ảnh C1 | 13,64 | 11,54 ms | 45,45 ms | 82,99 ms |

Nguồn ảnh ban đầu khác vị trí trên thẻ, nên chênh 13,73→12,44 FPS **không** là lợi ích của mã. Cả sau khi dùng một thư mục ảnh, thời gian decode dao động 11,54–19,78 ms giữa ba lượt; FPS ứng viên 12,69 nằm giữa hai lượt C1 12,44 và 13,64. Thay đổi mã giảm `infer_ms` khoảng 0,56 ms ở các lượt chung nguồn, phù hợp việc bớt copy, nhưng chưa chứng minh tăng FPS toàn luồng và không đạt cổng ≥5% đã dùng cho tối ưu Nano. Không mở lượt EKF/HDMI đầy đủ, không chọn ứng viên. Launcher và cấu hình C1 đã triển khai giữ nguyên.

Muốn tìm tăng tốc tiếp theo cần xử lý hoặc đo tách dao động giải mã JPG và thử đọc trước frame có giới hạn hàng đợi, rồi đánh giá lại bằng cặp cùng boot, cùng nguồn, có hiển thị và cả CV/EKF. Không cộng ~0,56 ms vào kết quả tốc độ C1 như lợi ích đã đạt.
