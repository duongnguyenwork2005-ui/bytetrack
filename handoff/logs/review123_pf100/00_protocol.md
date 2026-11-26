# R123 — một chiều lấy mẫu có cần 500 hạt không?

Sau R120 trượt cổng (tăng IDSW và FN), dừng làm mềm R, không dò ν. Giữ nguyên Cartesian RBPF R118, chỉ giảm N500→100 để kiểm tra độ hiệu quả của marginalization một chiều omega. Không thay Q/R/init/association, ba seed. Đây là đánh đổi chất lượng và chi phí, không mặc định ít hạt sẽ tốt hơn.

Dev UA train12, cùng đầu vào R118,36 lượt. Mở chuyển SVMOT9 nếu IDSW trung bình <=1,02 lần R118, HOTA không giảm quá0,25điểm phần trăm, FN không tăng quá0,5%. Nếu trượt dừng, không thử thêm N khác. Nếu đạt, giữ100cho SVMOT9×3seed và báo so cả R118500 lẫn R112500 đã lưu, không gọi độc lập vì các tập đã dùng nghiên cứu. Chỉ chọn ứng viên nhanh hơn nếu giữ chất lượng cả hai tập; không thay mặc định toàn dự án.

Numerical zero-rate/PSD sử dụng kiểm R118; lặp seed100, smoke500 đối chiếu R118. Đo tuần tự PC hai prefix512×ba lần đảo thứ tự, bỏ32khung đầu, sau khi mọi job chất lượng/GPU kết thúc, cùng process và input. Không dùng timing job đồng thời hay suy ra FPSNano. Hash đầu vào/nguồn/metric/MOT trước-sau, raw fields tái hợp.
