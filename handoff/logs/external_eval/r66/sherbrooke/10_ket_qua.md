# R66 Sherbrooke — kết quả 1.001 frame

Đã hoàn tất 36/36 lượt v2 trên Sherbrooke, một camera giao thông cố định, 15 ID và 2.012 box hợp lệ. Mỗi detector chạy trên ảnh gốc, mỗi bộ lọc dùng cùng detections ở từng ngưỡng. FPS 30, buffer 36, nhóm xe theo SQLite cars, đầu vào tracker COCO car. GT và kết quả được chấm theo cùng mask của tác giả. Nguồn cần trích Jodoin, Bilodeau và Saunier, WACV 2014. Xem `v2/review/02_geometry.md` về hai góc box và điều kiện sử dụng nguồn trong biên bản smoke.

HOTA thang 100. Mỗi dòng là toàn bộ sequence. PF500 là trung bình ba seed.

| Detector | Ngưỡng | KF/CV | EKF | UKF | PF500 |
|---|---:|---:|---:|---:|---:|
| YOLOv8n COCO | 0,25 | 32,39 | 32,36 | 32,36 | 26,14 |
| YOLOv8n COCO | 0,10 | 31,74 | 31,71 | 31,71 | 25,21 |
| YOLOv8n mixed5 | 0,25 | 33,92 | 33,83 | 33,79 | 26,92 |
| YOLOv8n mixed5 | 0,10 | 33,21 | 33,11 | 33,10 | 26,04 |
| YOLO11n mixed5 | 0,25 | 33,92 | 33,92 | 33,85 | 26,20 |
| YOLO11n mixed5 | 0,10 | 33,25 | 33,23 | 33,12 | 25,72 |

PF500 thấp hơn KF/CV từ 6,25 đến 7,73 điểm HOTA. EKF và UKF cách KF nhiều nhất 0,14 điểm. Nhãn có xe đỗ/chờ trong ROI không có GT, khiến một số phát hiện đúng phương tiện vẫn chịu FP. Thành viên của SQLite cars bao gồm cả mô tả xe tải và xe buýt. Kết quả này mô tả một cảnh, không có phép phân tầng xe máy, chuyển hướng hay che khuất.

Điểm từng seed, DetA, AssA, IDF1, MOTA, FP, FN, IDSW và min–max theo seed ở `v2/full/filter_summary.csv`; điểm nguồn ở `v2/full/scores.csv`. Mỗi seed chấm COMBINED riêng. Khoảng min–max chỉ thể hiện dao động giữa ba seed. Việc chạy mất khoảng 13 phút theo đồng hồ bên trong lượt full; có đọc ổ F: để kiểm file HELMET, nên không dùng thời gian này làm benchmark tốc độ.

Smoke v2 đạt 36 lượt và hai lần lặp PF. Cổng cuối so 8 chỉ số với các thống kê gốc, kiểm lại hash ảnh, GT, cache, MOT thô và MOT sau ROI. `v2/full/coverage.json` xác nhận đủ số lượt và hash của CSV. Lượt mở sớm trước review được giữ theo `v2/noncanonical_manifest.json`, không đưa vào bảng. Giữ nguyên tất cả bộ lọc và detector của R64.
