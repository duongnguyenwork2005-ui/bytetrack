# SVMOT – Kiểm tra riêng các đoạn xe đổi hướng

Dùng 18 đầu ra R64 của ByteTrack với KF/CV, EKF và UKF trên chín chuỗi SVMOT test. Mỗi cấu hình dùng chung detector, ngưỡng và cache R64 giữa ba bộ lọc. Tổng IDSW và FN khi phát lại CLEAR khớp 162/162 cặp chuỗi và lượt chạy.

Quy tắc quỹ đạo từ RoundaboutHD R99 được quy đổi từ 15 sang 30 fps. Dùng góc giữa hai vectơ từ trung vị vị trí đáy hộp GT, cửa sổ ±1 giây, có ngưỡng dịch chuyển tối thiểu. Đoạn ≥25° yêu cầu ít nhất 16 khung liên tiếp. Mẫu có 12 đoạn và 240 khung-xe. Ngưỡng ≥35° có 7 đoạn và 125 khung-xe. Ngưỡng gần thẳng ≤8° có 108 đoạn và 5.799 khung-xe.

- YOLOv8n COCO c25, toàn chín chuỗi có 435, 407, 418 IDSW theo thứ tự KF/CV, EKF, UKF. Nhóm ≥25° có 2, 2, 3 IDSW và 64, 65, 65 FN. Nhóm ≥35° cũng có 2, 2, 3 IDSW.
- Cả sáu cấu hình detector/ngưỡng đều thiếu bằng chứng cả EKF và UKF giảm IDSW trên nhóm đổi hướng. Tính theo từng đoạn, cũng không có đoạn mà cả hai cùng có IDSW thấp hơn KF trong cùng một cấu hình.
- Hai góc camera quan sát xe chủ yếu chạy dọc đường. Góc của đáy hộp GT trên ảnh có thể đổi khi box bị cắt ở biên, khi bị che hoặc khi điểm đáy dao động lúc xe dừng. Số đoạn trên là số nhận diện theo góc ảnh, cần kiểm ảnh trước khi gọi là thao tác rẽ vật lý.
- TUMTraf-V2X R4_S4 (drive_22 theo Hình 7 phụ lục) là ứng viên kiểm chứng độc lập. Nguồn tác giả xác nhận nhiều xe quay đầu và có ảnh camera ven đường cùng hộp 3D, track ID và hiệu chuẩn. Cần kiểm tính dùng được của nhãn 2D chiếu lên ảnh, vùng nhìn và visibility trước khi chấm. Chưa có kết quả ByteTrack của dự án trên tập này.

Nguồn dự án: handoff/logs/external_eval/SVMOT.md. Mã và bảng chi tiết: audit.py và audit.json. Nguồn dữ liệu: https://zenodo.org/records/19468203. Nguồn TUMTraf: https://tum-traffic-dataset.github.io/tumtraf-v2x/.
