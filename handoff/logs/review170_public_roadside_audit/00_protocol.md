# R170 — kiểm tra nguồn dữ liệu camera ven đường công khai

Ủy quyền từ yêu cầu hiện tại: tìm dữ liệu cho đổi hướng, che khuất và xe máy,
đồng thời ưu tiên tối ưu Nano/cắt tỉa. Chỉ đọc nguồn công khai, không gửi mail,
đăng ký tài khoản hoặc tiếp tục tác vụ tài khoản TUMTraf-V2X đang tạm dừng.

Lượt này chỉ kiểm metadata và gói nhãn nhỏ: không mở tracker, không thay các
cổng PF/Gaussian, không chạy GPU hay đo tốc độ PC/Nano. Không tải ZIP động
không rõ dung lượng; dừng GET khi server bỏ qua Range. Giới hạn mỗi response
metadata 2 MiB, từng thao tác công khai read-only, không thực thi mã dataset.

Nguồn tác giả:

- https://accident-dataset.github.io/ có link tải công khai Nextcloud,
  nhưng số khung/lớp/tần số khác trang https://tum-traffic-dataset.github.io/tumtraf-a/
  và abstract https://arxiv.org/abs/2508.14567. Không trộn các con số giữa
  release; phải kiểm README/nhãn/timestamp thật trước khi lập converter.
- https://github.com/lihaoliu-cambridge/trafficmot có link Drive công khai;
  chỉ xem metadata nếu cần xác minh thay đổi, không gửi lại yêu cầu truy cập.
- https://github.com/I24-MOTION/I24-3D-dataset là camera đường với GT 3D và
  timestamp đã hiệu chỉnh; không coi hộp chiếu 3D là cùng GT 2D hiện hành.

Điều kiện trước một thí nghiệm tương lai: ảnh và ID ổn định trong cùng camera;
thứ tự/timestamp hoặc fps+stride đã xác minh; nhãn phương tiện có ánh xạ rõ;
hộp 2D có định nghĩa rõ và chính sách bỏ qua nhất quán; quyền tải và đủ đĩa.
Nhóm đổi hướng/che khuất chọn từ GT và quy tắc đã công bố, trước mọi điểm
tracker. Không chọn mẫu vì EKF/UKF thắng. Không hứa dataset mới sẽ giúp thắng
KF; giữ cả kết quả âm. Chưa cho phép chuyển PF sang full/scenario khi cổng
Gaussian full BDD/UA hiện tại còn trượt. Không sửa launcher hay khóa luận.

GET ZIP công khai ban đầu trả HTTP200 dù yêu cầu suffix Range, không có
Content-Length; đã đóng stream, không tải body. Endpoint WebDAV công khai
trả207 cho PROPFIND Depth1 và cho thấy thư mục accid3nd_dataset. Đây là API
đọc của chính share công khai, không dùng credential hoặc đường vòng auth.
