# R170 — nguồn camera ven đường mới, chưa phải benchmark đã sẵn sàng

Đã truy cập được một gói TUM Traffic Accid3nD công khai từ [trang tác giả](https://accident-dataset.github.io/).
Không đăng nhập, đăng ký hoặc tiếp tục yêu cầu tài khoản TUMTraf-V2X đã tạm dừng.
Chưa có tracking, detection AP hay bằng chứng EKF/UKF thắng KF trên bộ này.

Năm ZIP có tổng 46.587.516.755 byte. Đã đọc đủ central directory bằng
41.835.824 byte HTTP Range, đối chiếu size/Content-Range/ETag. Chưa tải toàn
bộ ZIP. Có 111.700 JPG và 111.656 JSON ở 45 nhóm sequence/camera; không có
đường dẫn file trùng giữa các ZIP. Đây là số đếm gói cụ thể, không thay bằng
số khung tổng quát của bài báo hoặc cộng cả camera và LiDAR thành ảnh.

Có 95.345 cặp khớp tên ảnh–JSON trực tiếp. Tám nhóm không khớp tên. Đọc JSON
đầu tiên ở từng nhóm cho thấy bốn mẫu tham chiếu được ảnh thật qua
`frame_properties.image_file_names` dù tên camera trong filename khác tên
thư mục. Bốn mẫu của r09_s05 vẫn trỏ ảnh không có trong danh mục; ảnh cùng
thư mục thuộc khoảng thời gian khác. Không tự dịch timestamp, ghép gần nhất
hoặc giả định toàn bộ các nhóm đã được sửa. Phát hiện này chỉ ở một mẫu/nhóm.

Ảnh mẫu đầu tiên theo thứ tự tên có độ phân giải 1920×1200, cảnh cao tốc từ
camera cố định. JSON OpenLABEL có 20 object, 16 CAR và 4 TRUCK, UUID, hộp
`full_bbox`, mask, timestamp và tham chiếu ảnh rõ. ZIP CRC và SHA của ảnh/
nhãn đã kiểm. Mẫu này không có thuộc tính che khuất; không suy điều đó cho
toàn bộ dataset và không coi tên full_bbox là bằng chứng đủ về hộp amodal.
Chưa kiểm liên tục UUID hoặc số xe máy trên cả bộ.

Timestamp tên tệp của các cặp đã khớp có 7.301 khoảng >42 ms. Đây là chẩn
đoán so với nhịp danh nghĩa 25 Hz, không phải bằng chứng mọi khoảng khác
đều là lỗi. Không được đọc ảnh kế tiếp rồi mặc định Δt=40 ms cho mọi cặp.
Trang [Accid3nD](https://accident-dataset.github.io/) và trang
[TUMTraf-A](https://tum-traffic-dataset.github.io/tumtraf-a/) mô tả các quy
mô/nhịp khác nhau; giữ riêng revision/gói, kiểm dữ liệu thực trước converter.

Nếu dùng tiếp: xác minh explicit image reference trên toàn nhãn, chính sách
frame thiếu và cadence, sự bền vững UUID, tọa độ/semantic hộp và lớp. Sau đó
mới chọn kịch bản từ GT trước khi xem điểm tracker. Giao thông cao tốc và tai
nạn không tự bảo đảm mô hình constant-turn tốt hơn KF; phải báo cả nhóm
thẳng, đổi hướng và nhóm thua. Chưa tải hàng chục GB khi mapping chưa rõ.

V1 dừng vì directory lớn hơn response 2 MiB, giữ source/failed index. V2 chỉ
chia Range, giữ 2 MiB/response, 20 MiB/ZIP và giới hạn request, không fallback
tải cả archive. Tám nhãn kiểm thêm dùng local ZIP header/deflate/CRC, tổng
61.855 byte payload, không cần tải lại central directory.

Bằng chứng: `inventory_verified.json`, năm `train*.zip.index_v2.json`,
`sample/verification.json`, `unmatched_samples/verification.json`. PASS chỉ
xác nhận tính toàn vẹn metadata/transport, không xác nhận chất lượng nhãn
hay mức sẵn sàng của benchmark. Không đổi bộ lọc, cấu hình thí nghiệm,
launcher hoặc khóa luận; cổng Gaussian full UA và PF vẫn giữ nguyên.
