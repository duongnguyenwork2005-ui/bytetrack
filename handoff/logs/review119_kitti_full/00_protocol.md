# R119 — mở rộng KITTI trước khi đọc điểm

Nguồn: yêu cầu 19/11 của người dùng, tiếp tục tự nghiên cứu PF và tìm dữ liệu xe rẽ/che khuất/xe máy. Không cập nhật khóa luận.

Chạy tất cả chuỗi KITTI Tracking training có Car hợp lệ theo cùng chính sách evaluator R115 (truncation<=0, occlusion<=2). Chuỗi không có Car hợp lệ chỉ thống kê, không dùng benchmark người. Không chọn chuỗi theo kết quả. R115 đã xem 0001/0011/0020, báo riêng phần mới và tổng.

Detector dùng nguyên YOLOv8n COCO của R115, SHA đã khóa, 640 rect FP32, conf cache .05, NMS .7, max_det 3000; tracking c25 Car, cap300, buffer 1 giây, 10Hz, elapsed steps, cùng KF/EKF/UKF. Dùng lại ba cache cũ nguyên byte. Native KITTI evaluator xử lý Van/DontCare như R115. Khóa GT, hash nguồn và danh sách khung trước khi sinh cache/chấm.

Nhóm kịch bản giữ định nghĩa R115: native occlusion2; turn25 theo hướng bottom-center trên ảnh, cửa sổ một giây hai phía và đoạn bền; giao nhóm. Camera gắn trên xe làm góc trên ảnh chịu tác động ego-motion. Không gọi đó là quỹ đạo thế giới. Giữ sự kiện occlusion2 dài>=3 ảnh với hai đầu GT liên tục để chẩn đoán khôi phục ID. Báo IDSW cùng FN và HOTA, không diễn giải ít IDSW là tốt hơn nếu mất nhiều mục tiêu.

Tiêu chí kiểm: cache/GT/MOT và nguồn hash trước/sau; từng nhóm CLEAR IDSW/FN tái ghép khớp evaluator; ba chuỗi cũ khớp toàn metric/MOT R115; aggregate từ raw fields, bootstrap paired video 10.000 mẫu. Mọi kết quả kể cả không thuận đều báo. Không đổi tracker/association/mặc định.
