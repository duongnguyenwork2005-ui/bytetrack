# R112b — RBPF với cập nhật Student-t, thăm dò có kiểm soát

Ghi sau khi đã thấy RBPF R112 đạt dev12 nhưng trượt IDSW SVMOT, trước khi chạy ứng viên robust. Chỉ thử một cấu hình, bậc tự do Student-t ν=4, áp dụng chung cho mọi video. Không chọn tham số riêng cho dataset.

Mỗi hạt dùng mật độ dự đoán Student-t trên innovation bốn chiều để cập nhật trọng số. Cập nhật Gaussian điều kiện được xấp xỉ bằng R tăng theo max(1,(ν+Mahalanobis²)/(ν+4)), Joseph covariance. Đây là xấp xỉ robust, chưa phải tích phân chính xác của mô hình nhiễu Student-t với prior Gaussian.

Dev12 giữ cache, CTRV, 500 hạt và các tham số ghép đã có; ba seed 0/1/2 chạy trong các tiến trình độc lập. Cổng dev như R112 đối với PF500 K1/K2 motion2, gồm IDSW giảm 10%, HOTA giảm tối đa 0,5 điểm, FN tăng tối đa 0,5%, không seed tăng IDSW và số clip tốt nhiều hơn xấu. Nếu đạt, kiểm lại đủ SVMOT test9 với ba seed, dùng đối chứng base R112 đã phát lại đầy đủ. Yêu cầu IDSW toàn phần và ở visibility ≤0,5 không tăng, HOTA giảm tối đa 0,5 điểm và FN tăng tối đa 0,5%. Dữ liệu đã được xem qua nhiều vòng, kết quả chỉ mô tả và phát triển.

Sau cấu hình này, tổng hợp cả ứng viên đạt và trượt. Không tiếp tục dò ν hoặc đổi ngưỡng từ điểm SVMOT trong lượt này. Chỉ cấp tên tracker thử cho biến thể có kiểm số học và chạy từ registry bình thường, ghi rõ trạng thái lựa chọn theo cổng.
