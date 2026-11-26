# Tích hợp RBPF sau phép đo

Người dùng đã yêu cầu sửa bộ lọc. Cổng dev12 của RBPF R112 đã đạt cả năm tiêu chí với IDSW trung bình 469 so với 697. Đang kiểm chuyển miền trên SVMOT. Sau khi các lượt đo kết thúc và kiểm hash cũ, có thể đưa ConditionalPF và các lớp cầu nối vào src/rbpf_ctrv.py, đăng ký tên bytetrack_rbpf_ctrv trong runner chung và cung cấp YAML riêng dưới configs/phase2/trackers/r112/. Đây là phần tích hợp được bổ sung sau phạm vi ghi ban đầu của R112, để chạy bản mới từ entry point của dự án.

Giữ nguyên bản plugin đã đo trong thư mục R112 để tái lập. Lớp ứng dụng mới phải khớp MOT prefix của plugin ở ba seed và cả hai backend. Cấu hình mới dùng đúng 500 hạt, ESS 0,5, ước lượng mean, CTRV và ngưỡng ghép như phép đo. Các thuật toán K1 và roughening không chạy trong RBPF, do cập nhật Gaussian điều kiện đảm nhiệm phần quan sát.

Kiểm CLI có liệt kê tên mới, nạp đúng YAML, chạy replay qua đường run_tracking thông thường và kiểm hồi quy PF cũ qua một prefix đã lưu. Không dùng phép đo này để báo FPS Nano.

Quyết định sau phép đo ngày 18/11. Gaussian RBPF tăng IDSW SVMOT 434→440, Student-t trượt cổng FN dev nên không chạy transfer. Chỉ đăng ký Gaussian với tên nghiên cứu rõ ràng để tái hiện và phát triển, không coi tích hợp là vượt qua cổng chuyển miền. Cấu hình mới tắt các cờ K1/K2 không được thuật toán sử dụng và chặn các tùy chọn PF cũ chưa hỗ trợ. Xác minh hash trước tích hợp lưu riêng, các hash của runner/fingerprint sau tích hợp được ghi bằng postintegration_provenance.py, không ghi đè manifest thí nghiệm đã khóa.
