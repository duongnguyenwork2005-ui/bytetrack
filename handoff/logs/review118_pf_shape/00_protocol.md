# R118 — RBPF vận tốc Cartesian

Đổi phần Gaussian điều kiện sang vị trí và vận tốc Cartesian [cx,cy,vx,vy]. Lấy mẫu một chiều tốc độ quay omega. Hai khối hình dạng [a,va] và [h,vh] độc lập với góc được cập nhật Gaussian dùng chung. Dùng transition coordinated turn liên tục tại omega=0. Q cho vx và vy bằng nhau, theo hệ số nhiễu vận tốc hiện hành. Đây là thay đổi mô hình nhiễu và biểu diễn trạng thái, không khẳng định tương đương PF cực trước đó.

Khởi tạo chuyển động cùng quyết định và hai quan sát như cũ, chuyển bất định tốc độ/hướng sang hiệp phương sai vận tốc bằng Jacobian. Không đổi cửa sổ ghép, detector, cutoff hay lifecycle. Thử 500 hạt, seed 0/1/2, đủ train dev12 UA-DETRAC. So trực tiếp RBPF R112 (469 IDSW, HOTA 58,96%) và báo PF K1K2 cũ (697 IDSW). Tập dev đã xem nhiều vòng nên kết quả là phát triển. Đánh giá trước số học, liên tục tại omega=0, giới hạn CV, covariance PSD, cùng seed và parity RBPF nền trước full.

Cổng mở kiểm chuyển miền là IDSW trung bình giảm so RBPF, HOTA giảm tối đa 0,5 điểm phần trăm, FN tăng tối đa 0,5%, đa số seed giảm IDSW. Nếu đạt, chạy lại cả RBPF nền và ứng viên trên SVMOT test9 theo cache/cutoff R112. Báo thăm dò do miền đã quan sát. Nếu không đạt, kết thúc phép thử và lưu nguyên kết quả.

Nguồn cấu trúc RBPF [Doucet và cộng sự](https://arxiv.org/abs/1301.3853), mô hình coordinated turn [MathWorks trackingCT](https://www.mathworks.com/help/fusion/ref/trackingct.html). Hai khối hình dạng dùng chung có thể vì Q chiều cao đóng băng theo trung bình và ban đầu không có tương quan với chuyển động. Covariance cực 9 chiều dùng làm giao diện được xấp xỉ bằng Jacobian; ghép ByteTrack dùng hộp vị trí/kích thước.
