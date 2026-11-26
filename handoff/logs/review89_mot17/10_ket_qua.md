# R89 — MOT17 train: kết quả chuyển miền cho che khuất

> **ĐÃ LOẠI KHỎI PHẠM VI KHÓA LUẬN (06/11/2026).** Người dùng xác nhận dự án chỉ theo dõi xe cộ. MOT17 là tập người đi bộ; số liệu dưới đây chỉ giữ làm dấu vết của phép thử chọn sai dataset. Không dùng chúng để chọn bộ lọc, so IDSW của xe, viết kết quả luận văn hay đề xuất cấu hình triển khai.

Nguồn chính thức: https://motchallenge.net/data/MOT17/ . Tải từ hai ZIP chính thức về F:, kiểm đủ byte, CRC và SHA256 **cục bộ** tại `F:/LABOTARY_data/goc_tai_ve/MOT17-R89/{images,labels}_manifest.json`. Đây là GT train công khai, không có kết quả test ẩn hay nộp leaderboard. Giao thức trước tải/test `00_protocol.md`; 7 scene duy nhất dùng biến thể tên `-FRCNN` nhưng **không dùng detection FRCNN**. Detector trong mọi lượt là checkpoint YOLOv8n COCO R54 (SHA256 `f59b3d83…`), conf cache>0,05, imgsz640; CV/EKF/UKF/PF đọc đúng một cache. GT/seqinfo giữ byte gốc; bản DPM/FRCNN/SDP của cùng scene có GT trùng SHA256 và chỉ đếm một lần.

Nhãn 7 scene gồm 5.316 frame, 112.297 pedestrian marked, FPS nguồn 30/14/25. Trong đó 45.904 box (40,9%) có visibility<0,5; visibility là giá trị nguồn, không phải cờ occlusion nhị phân. Detector smoke 10 frame/scene đạt cổng 805/1.398 (57,58%) và cả 7 scene >20%. Trên **toàn bộ** frame, recall one-to-one IoU≥0,5 của detection person: conf>0,05 **64.049/112.297 = 57,0%**, visibility<0,5 **11.150/45.904 = 24,3%**, visibility≥0,5 **52.899/66.393 = 79,7%**. Ở conf>0,25, các tỷ lệ là **41,8% / 10,7% / 63,2%**. Chênh lệch lớn này là giới hạn quan trọng khi đọc metric tracker.

Ma trận gồm CV, EKF, UKF, PF500 mặc định và PF500 K1/K2 motion100 opt-in; PF ba seed 0/1/2; c25/c10; track buffer 1,2 s đổi thành 36/17/30 bước theo 30/14/25 fps. `summary.json` hậu kiểm đạt **30/30 ô cấu hình, 54/54 evaluator và 126/126 lượt video/seed**; không cache ảnh lỗi/truncated, tệp MOT hữu hạn/không trùng (frame, ID), đủ frame timing, GT hash đúng, và HOTA/IDF1/MOTA/IDSW tái cộng raw fields bằng `boot_stats.combine` khớp CSV evaluator. Không lấy trung bình HOTA theo video hoặc theo nhóm FPS.

Trên 7 video, số thứ tự sau là **HOTA / IDF1 (%) / IDSW**, PF là trung bình ba seed của metric COMBINED:

- c25: CV **36,475 / 42,945 / 595**; EKF **36,416 / 42,802 / 586**; UKF **36,050 / 42,318 / 649**; PF mặc định **35,629 / 41,440 / 657,67**; PF K1/K2 **36,071 / 41,913 / 622,00**.
- c10: CV **36,653 / 42,931 / 591**; EKF **37,072 / 43,677 / 598**; UKF **36,789 / 43,082 / 608**; PF mặc định **36,422 / 42,711 / 650,00**; PF K1/K2 **36,372 / 42,439 / 605,33**.

PF K1/K2 giảm IDSW so PF mặc định **35,67** ở c25 (−5,4%) và **44,67** ở c10 (−6,9%). Bootstrap ghép cặp 7 video, 5.000 lượt, dùng raw fields trung bình ba seed trước tái cộng: CI95 của chênh IDSW (PF K1/K2 − PF mặc định) lần lượt **[−70,0; −5,7]** và **[−67,7; −20,7]**. C25 HOTA PF tăng 0,441 điểm, CI [−0,028; +0,781] điểm; c10 HOTA PF **giảm 0,051 điểm**, CI [−0,319; +0,585] điểm. CI này phản ánh lấy mẫu scene, không tính độ bất định của tập GT hay detector; chỉ 7 scene.

Tại c10, PF K1/K2 cao hơn CV **14,33 IDSW (2,4%)** và EKF **7,33 (1,2%)**, thấp hơn UKF 2,67; khoảng bootstrap PF−CV là **[−7,7; +36,3] IDSW**, chưa đủ chứng minh tương đương với biên hẹp. HOTA PF thấp hơn EKF 0,701 điểm và IDF1 thấp hơn 1,238 điểm. Tại c25 PF còn cao hơn CV 27 IDSW, EKF 36, nên không tuyên bố PF tốt nhất hay ngang bằng toàn diện. Dữ liệu per-video và per-seed trong `summary.json` cho phép xem chênh lệch từng scene.

Giới hạn: MOT17 là **người đi bộ**, phù hợp để thử che khuất/chuyển động không đều nhưng không chứng minh khả năng bám xe máy. Ở visibility thấp, detector bỏ sót đa số GT; nhiều lỗi là trần của detection, không thể gán nhân quả cho bộ lọc. Đây là đánh giá nội bộ trên train với YOLOv8n COCO, không so trực tiếp bảng xếp hạng MOT17 vốn dùng giao thức/detector riêng. Không đổi cấu hình luận văn hoặc Nano; ứng viên R87 Rao–Blackwell trước đó không đạt cổng SVMOT và vẫn tắt mặc định.
