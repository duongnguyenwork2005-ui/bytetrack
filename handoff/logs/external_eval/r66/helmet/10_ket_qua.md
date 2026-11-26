# HELMET test36 — theo dõi xe máy trên 12 nhóm cảnh

**EKF và UKF đều tăng HOTA so với KF/CV ở cả sáu tổ hợp detector/ngưỡng.** Trên 36 clip, mức tăng lần lượt là 0,18–1,16 và 0,25–1,12 điểm HOTA. PF500 trung bình ba seed thấp hơn KF/CV 1,35–3,93 điểm.

## Phạm vi và độ phủ

Mẫu gồm 3 clip test từ mỗi nhóm trong 12 nhóm tên cảnh, chọn theo SHA-256 cố định trước khi xem điểm. Mỗi clip có 100 frame, 10 fps, reset tracker ở đầu clip, buffer 12 bước. Tổng cộng 3.600 frame, 9.371 box xe máy và 287 ID tính riêng theo clip. Ba detector dùng ảnh gốc 1920×1080 với cùng thông số sinh cache R64. Tracker lọc lớp motorcycle COCO=3. So các bộ lọc với cùng cache và cùng ngưỡng conf >0,25 hoặc >0,10. Nguồn HELMET của Lin, Deng, Albers và Siebert, CC BY 4.0. Đây là đánh giá bổ sung cho tracking trên một tập con cân bằng nhóm của test182. [OSF HELMET](https://osf.io/4pwj8/) · [Kho tác giả](https://github.com/LinHanhe/Helmet_use_detection).

Thông tin nguồn và dữ liệu chuyển đổi ở `manifest_test36.json`, `nguon_kiem_sha_crc.json`, `giai_nen_test36.json`. Tất cả 7 archive ảnh đạt kích thước, SHA-256 và CRC. Chỉ giải nén ảnh của mẫu đã khóa (1.186 MiB). 3 × 36 NPZ detection full chiếm 4,69 MiB. Không phát hiện box trùng hoặc xung đột trong 36 clip.

## Kết quả

Điểm HOTA COMBINED thang 100. Cột PF500 ghi trung bình và khoảng nhỏ nhất–lớn nhất của ba seed. Khoảng này mô tả dao động giữa seed.

| Detector | Conf | KF/CV | EKF | UKF | PF500 TB [min, max] |
|---|---:|---:|---:|---:|---:|
| YOLOv8n COCO | 0,25 | 30,11 | 30,34 | 30,36 | 26,22 [25,79–26,46] |
| YOLOv8n COCO | 0,10 | 30,58 | 30,77 | 31,01 | 26,65 [26,25–26,96] |
| YOLOv8n mixed5 | 0,25 | 17,72 | 18,87 | 18,85 | 16,37 [16,22–16,47] |
| YOLOv8n mixed5 | 0,10 | 21,00 | 22,16 | 22,11 | 19,25 [18,86–19,82] |
| YOLO11n mixed5 | 0,25 | 16,80 | 17,25 | 17,27 | 14,20 [14,16–14,27] |
| YOLO11n mixed5 | 0,10 | 18,55 | 19,09 | 19,64 | 16,38 [15,80–16,89] |

Với YOLO11n mixed5 ở 0,10, UKF tăng 1,10 điểm HOTA, tăng 0,72 điểm IDF1 và giảm IDSW từ 78 xuống 54. Theo HOTA từng nhóm, UKF hơn KF ở 9 nhóm, thấp hơn ở 2 và bằng ở 1. EKF cũng có 54 IDSW. Xét cả sáu tổ hợp, EKF và UKF luôn giảm IDSW so với KF. Riêng với YOLOv8n COCO, IDF1 giảm nhẹ dù HOTA tăng. Dữ liệu chi tiết ở `full/combined_tung_run.csv`, `full/filter_summary.csv`, `full/per_clip.csv` và `full/per_scene.csv`. COMBINED được tái tính từ raw fields của các clip. Không lấy trung bình điểm HOTA của các clip thay cho COMBINED.

## Hậu kiểm và giới hạn

Smoke 12 clip × 36 cấu hình đạt 432 đơn vị. Lặp PF500 cùng seed 0 trên YOLOv8n mixed5 c25 cho MOT, raw fields và bảng điểm trùng byte. Full đạt 36 × 36 = 1.296 đơn vị. Sau chạy đã kiểm lại hash ảnh đầu vào, cache, nhật ký từng clip, MOT đã xuất và tám chỉ số tổng hợp với ngưỡng sai khác 1e-9. Dấu vết `smoke/final_review.json`, `full/phase_inputs.json`, `full/tong_hop_trang_thai.json`, `full/final_review.json`. Nhật ký ngân sách giai đoạn full từ 05:44 đến 06:02 ngày 03/11 và sau đó là hậu kiểm. Mỗi bộ lọc giữ cấu hình R64.

GT thường bao xe máy và cả người ngồi. Trên smoke, recall ở IoU ≥ 0,5, conf >0,25 là 60,03% cho YOLOv8n COCO, 27,69% cho YOLOv8n mixed5 và 23,96% cho YOLO11n mixed5. `smoke_review/01_ngu_nghia.md` ghi ảnh đối chiếu và giới hạn. Mức chênh mô tả **toàn mẫu test36**. Lượt này không gán nhãn riêng thời điểm rẽ, quỹ đạo cong hoặc che khuất nên chưa thể quy lợi thế EKF/UKF cho riêng một trong các tình huống đó.
