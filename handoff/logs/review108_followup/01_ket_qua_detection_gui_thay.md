# Kết quả phát hiện phương tiện

Báo cáo tiến độ lần 4 dùng **YOLOv8n COCO** trên SVMOT và trong cấu hình Jetson Nano v21combo, và **YOLO11n mixed5** trên HELMET. Mixed5 là mô hình huấn luyện cho năm lớp car, truck, bus, motorcycle và bicycle.

## BDD100K validation

Đánh giá trên **39.973 ảnh, 200 video**, theo giao thức COCO-video của dự án. Các số dưới đây là phần trăm.

- **YOLOv8n COCO** đạt **24,39 mAP50** và **15,71 mAP50–95**. AP50 của car, truck, bus, motorcycle và bicycle lần lượt là **55,77 / 18,31 / 24,87 / 11,75 / 11,26**. [Chi tiết từng lớp](../../../results/phase2/r53_detector/full/bdd_val/yolov8n_coco/theo_lop.csv)
- **YOLO11n mixed5** đạt **37,87 mAP50** và **24,94 mAP50–95**. AP50 từng lớp theo cùng thứ tự là **77,40 / 37,26 / 36,66 / 20,16 / 17,88**. [Chi tiết từng lớp](../../../results/phase2/r53_detector/full/bdd_val/yolo11n_mixed5/theo_lop.csv)

## UA-DETRAC test40

Đánh giá trên **40 video, 56.167 ảnh có nhãn**, sau khi loại 173 ảnh thiếu mục khung hình trong tệp nhãn. Giao thức đã khóa chấm AP cho hai lớp có ánh xạ nhãn tương ứng là car và bus.

- **YOLOv8n COCO** đạt **87,01 mAP50** và **59,97 mAP50–95**. [Kết quả UA-DETRAC](../../../results/phase2/r53_detector/full/detrac_test/yolov8n_coco/tong_hop.csv)
- **YOLO11n mixed5** đạt **88,84 mAP50** và **62,22 mAP50–95**. [Kết quả UA-DETRAC](../../../results/phase2/r53_detector/full/detrac_test/yolo11n_mixed5/tong_hop.csv)

## HELMET test, lớp xe máy

**YOLO11n mixed5** được đánh giá trên **182 clip, 18.200 ảnh, 54.529 hộp GT**. Ghép một-một theo confidence tại IoU ≥0,50. Ở ngưỡng confidence >0,10 dùng trong slide, detector đạt **precision 45,17% và recall 42,38%**. Ở ngưỡng >0,25, precision tăng lên **63,26%** và recall giảm còn **30,77%**. [Số đếm và kết quả theo tập con](helmet182/motorcycle_detection_operating.json)

AP50 chấm hộp ở IoU 0,50. mAP50–95 lấy trung bình qua các ngưỡng IoU từ 0,50 đến 0,95 và các lớp được chấm. Đây là số liệu **detection** đã chấm trên PC. Mỗi tập dữ liệu có quy tắc chấm và điều kiện ghi hình riêng. Benchmark v21combo dùng TensorRT FP16 384×640 trên Nano. [Đối chiếu kích thước ảnh 384×640](../review63/bao_cao_cuoi/10_bao_cao_cuoi.md)
