# R124 — PF100 trên KITTI: ít IDSW hơn, còn đánh đổi HOTA/FN

## Phạm vi và kết quả tổng

Kiểm miền camera gắn xe sau R123, giữ nguyên Cartesian PF100 và ba seed, không chọn video theo điểm. Cùng 20 chuỗi Car / 7.863 ảnh / 24.070 hộp, detector COCO c25, 10 Hz, buffer một giây và native KITTI evaluator như R119. Mỗi seed có workspace riêng để tránh race seqmap. Không có PF500 cùng mô hình trên KITTI trong vòng này; không suy ra hiệu ứng riêng của số hạt.

Theo thứ tự **KF / EKF / UKF / PF100 trung bình ba seed**:

- IDSW: **280 / 260 / 260 / 247,67**.
- HOTA: **49,764 / 49,680 / 49,717 / 49,487%**.
- IDF1: **65,157 / 65,105 / 65,223 / 65,021%**.
- FN: **9.416 / 9.422 / 9.416 / 9.477,33**.
- FP: **1.852 / 1.875 / 1.866 / 1.929,33**.

Ba seed PF100 có IDSW 248/250/245. So KF, PF giảm IDSW 11,55% nhưng HOTA giảm 0,277 điểm và FN tăng 61,33 hộp, tương đương 0,65%. Vì tracker bỏ sót nhiều hơn, không thể chỉ dùng IDSW để nói PF tốt hơn toàn diện. Bootstrap ghép cặp 20 video, 10.000 mẫu, PF100−KF IDSW **[−66,67,−4]**, so EKF **[−25,−0,33]**, so UKF **[−25,0]** (sai số số học cận trên 2e−15 làm tròn 0). Đây là nghiên cứu trên tập train công khai đã được dự án phân tích, không phải benchmark test mù.

## Theo kịch bản đã khóa R119

Rẽ ≥25° trên ảnh, 399 hộp: IDSW **8/8/8/7,67**, FN **96/106/104/106,33**. Chênh IDSW nhỏ, PF bỏ sót nhiều hơn KF. Góc trên ảnh còn chịu tác động ego-motion, không xác nhận quỹ đạo thế giới.

Native occlusion2, 4.259 hộp: IDSW **43/42/42/43,33**, FN **3.384/3.415/3.411/3.413,67**. PF không giải quyết tốt hơn nhóm che này so ba bộ lọc còn lại. Giao rẽ và che 50 hộp có IDSW 3/2/2/1 nhưng FN 30/37/37/38, mẫu ít và nhiều mất ghép; không lấy giao nhóm đó để quảng bá ưu thế.

## Kiểm chứng và quyết định

Đủ **60 lượt PF mới**, đối chiếu cache/GT R119 đạt 60/60, từng lượt tái ghép CLEAR để kiểm IDSW/FN trước cộng nhóm, 80 hash nguồn/MOT đạt. `summary.json`, `verification.json`, `bootstrap.json`; mã tái hiện `run_job.py`.

Giữ PF100 như ứng viên nghiên cứu có lợi về chi phí và IDSW tổng, chưa chọn mặc định cho mọi camera. R123 đã cho thấy giữ chất lượng so Cartesian500 trên UA/SVMOT và giảm 60,19% thời gian cập nhật PC; R124 bổ sung đánh đổi HOTA/FN khi chuyển camera. Không dùng phép đo PC để tuyên bố FPS Nano. Không chỉnh detector/association hay tham số để cứu điểm KITTI và không sửa khóa luận.
