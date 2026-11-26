# R126 — giảm chi phí PF trước, rồi kiểm chất lượng

Yêu cầu 19/11: PF phải có lợi cả về thời gian và chất lượng, chưa hài lòng với R125. Không đổi mặc định, detector, association, metric hoặc khóa luận.

Giai đoạn A: profile PF100 hiện hành trên đúng hai video timing BDD R125, prefix200, cùng cache. Profiling chỉ để xác định nút thắt, không lấy thời gian cProfile làm benchmark. Tối ưu toán số trong module nghiên cứu mới, giữ mô hình,100hạt,thứ tự RNG,resampling,Q/R/init. Ưu tiên thay solve/cholesky2×2 bằng công thức chuyên biệt có kiểm điều kiện và giữ Joseph covariance; tránh tính lặp không cần thiết nhưng giữ đủ moment/covariance của giao diện. Kiểm số học với bản gốc, PSD, trường hợp omega0/nhỏ/lớn, mất quan sát, lặp seed và MOT hồi quy.

Đo tuần tự cùng process sau khi không còn job chất lượng. Cổng hiệu quả thăm dò: thời gian update PF mới thấp hơn PF100 gốc ít nhất20% và không chậm hơn PF cổ điển trên trung bình mẫu BDD; không nói FPSNano từ PC. Chạy lại đủBDD200×3seed nếu thay toán số có thể ảnh hưởng đầu ra, so chính xác raw fields với R125. Nếu toàn bộ MOT/metric giữ nguyên thì cải thiện chỉ về chi phí, không gọi đó là tăng accuracy.

Giai đoạn B sẽ ghi bổ sung giả thuyết trước điểm nếu cần thay thuật toán. Dùng UA train dev12 để phát triển và chọn duy nhất một ứng viên trước khi chuyển BDD. BDD đã quan sát, không gọi là test mù; không dò nhiều tham số trên BDD để tìm thắng. Không giảm IDSW bằng tăng FN hay bỏ mục tiêu. Báo cả thất bại.
