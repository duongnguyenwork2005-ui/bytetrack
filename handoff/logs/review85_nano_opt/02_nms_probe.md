# R85 — NMS ma trận trên Nano, trạng thái trước đo pipeline

Ứng viên `v22/src/nano_detector.py` sao từ gói `v21combo`, chỉ thay đường gọi NMS. Với 8–256 box sau lọc conf, tính ma trận IoU một lần rồi giữ đúng thứ tự greedy stable; ngoài khoảng này dùng NMS gốc để giới hạn bộ nhớ. Engine C1, checkpoint, tensor, conf và tracker đều giữ nguyên. Gói thử tách biệt trên Nano: `~/nano_run/r63_20261030/v22nms/nano_transfer`; `diff -qr` với gói đang dùng chỉ báo một tệp `src/nano_detector.py` khác. Launcher chưa chuyển.

Kiểm CPU: 100/100 bộ box tổng hợp cho chỉ số NMS trùng gốc; 30/30 tensor decode tổng hợp cho output trùng byte trên PC NumPy 2.4.6 và trên Nano NumPy 1.19.5. Microbenchmark Nano (`10_nms_nano_micro.log`), ms/ca:

Thêm kiểm trên **300 ảnh thật** MVI_39031, raw YOLOv8n FP32 trên PC: output decode v21/v22 trùng byte 300/300; số box trước NMS trung vị 41, p90 56, tối đa 66, không frame nào cần fallback >256 (`12_nms_real_pc.log`). Đây là parity đầu ra thật trên PC; vẫn cần parity TensorRT FP16 và MOT trên Nano.

Đã chuyển **đầu vào NMS thật** từ 300 ảnh đó sang Nano (NPZ 185 KB) và chạy xen kẽ bốn lượt CPU, 1.200/1.200 lần chỉ số NMS trùng. Trên Nano NumPy 1.19.5: trung vị NMS gốc **2,50 ms** và ma trận **0,81 ms** (p90 3,70→0,97 ms; `14_nms_real_nano.log`). Mức tiết kiệm này là riêng NMS, dự đoán mức tăng FPS toàn luồng chỉ vài phần trăm; không được cộng trực tiếp vào số R83.

| Số box vào NMS | Gốc | Ma trận |
|---:|---:|---:|
| 20 | 5,77 | 0,83 |
| 50 | 14,74 | 1,39 |
| 100 | 29,90 | 2,63 |
| 200 | 61,34 | 6,42 |

Số này dùng **box tổng hợp**, chưa chứng minh số box thực tế, FPS, độ trễ hay chất lượng pipeline. OpenCV 3.2 hệ thống trên Nano thiếu module `dnn`, nên lựa chọn `cv2.dnn.NMSBoxes` dù nhanh trên PC không thể dùng trực tiếp. Đã chuẩn bị phép đo 4 cặp cân bằng × CV/EKF HDMI (`run_nms_hdmi_pairs.sh`, `analyze_nms_hdmi.py`) với cùng nguồn/engine; phải qua tiền kiểm ổn định trước khi chạy.
