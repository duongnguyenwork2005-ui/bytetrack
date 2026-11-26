# R125 — PF mới trên BDD, khóa trước kết quả

Yêu cầu 19/11/2026: thử PF mới R123 trên BDD. Giữ nguyên Cartesian PF100, không tinh chỉnh tham số trên BDD, không sửa mặc định hoặc khóa luận.

## Phạm vi

Đủ 200 video BDD100K COCO-video validation, 39.973 ảnh, manifest và cache YOLOv8n COCO của R61/R74. Cache conf .05, tracking conf>.25, COCO vehicle classes [1,2,3,5,7], max300, FPS5, buffer1,2giây=6khung, elapsed steps. Dùng mot_crowd như R74, không thay bằng MOT evaluator thường. Không gắn nhãn che khuất giả từ visibility mặc định của bản chuyển đổi.

Chạy PF100 Cartesian ba seed0/1/2, Cartesian500 cùng mô hình ba seed để tách hiệu ứng số hạt, và chạy lại KF/EKF/UKF mỗi loại một lần. Tổng1.800lượt clip mới, dùng chung detection. Đối chiếu thêm PF cổ điển500 K1/K2 motion2 R74 đã lưu (600lượt), không gọi đó là cùng mô hình với Cartesian. Kết quả không thuận cũng báo.

## Cổng và thống kê

Trước full: kiểm SHA nguồn/cache/GT/seqinfo theo marker R61, freeze mã/cấu hình/evaluator/baseline; hai video smoke R74, ba seed: PF cổ điển phát lại khớp MOT R74, PF100 cùng seed lặp khớp. CFG100 và CFG500 chỉ khác số hạt. Giữ nguồn R123/R118 không chỉnh.

Chấm HOTA, IDF1, IDSW, FN, FP, MOTA; kết hợp raw fields đúng TrackEval. PF lấy trung bình metric COMBINED từng seed, không trộn raw seeds rồi gọi chính xác cùng đại lượng. Bootstrap ghép cặp theo200video,5.000mẫu, tính lại COMBINED riêng từng seed trước lấy trung bình. BDDval đã được dự án xem nhiều vòng, không phải test mù. Không dùng điểm để chọn N hoặc sửa bộ lọc.

Hậu kiểm1.800marker/MOT/timing, nguồn/cache/GT không đổi, tái hợp9nhómraw. KF/EKF so tổng R61 (cùng cache đã kiểm); nếu khác, chẩn đoán và ghi rõ trước kết luận.

## Tốc độ

Sau khi mọi job chất lượng kết thúc, đo tuần tự PC trên hai video chọn trước điểm bằng mức trung bình detection ở phân vị50/90 theo manifest, tối đa200ảnh/video, bỏ16khung nóng. SoKF/PF cổ điển500/Cartesian500/PF100, ba lần đảo thứ tự (24lượt), seed0, cùng process và input. Kiểm MOT giữa các lần. Chỉ báo chi phí cập nhật tracker trên mẫu, không suy ra FPSNano hoặc FPS toàn hệ thống. Timing của các job đồng thời không dùng để so tốc độ.
