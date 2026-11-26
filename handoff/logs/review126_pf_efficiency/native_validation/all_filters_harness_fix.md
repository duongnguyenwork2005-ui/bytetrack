Lượt đo bổ sung đầu tiên dừng trước khi ghi bất kỳ timing row nào: đường tham
chiếu KF/EKF/UKF bị thêm hậu tố seed, trong khi R125 lưu ba tracker xác định
không có seed trong tên. Đã sửa dùng hàm `bdd.tag` đúng giao thức R125;
giữ script, manifest và error log ban đầu. Không đổi filter/cấu hình hay
độ dung sai. Lượt đo 24 hàng bắt đầu lại từ đầu và lưu manifest mới.
