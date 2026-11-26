# FastTrack và KITTI Tracking

Đã chạy 36 lượt trên 7.032 khung hình nguồn. Mọi dãy số bên dưới theo thứ tự KF, EKF, UKF và PF500 K1K2 motion2. PF lấy trung bình ba seed 0, 1 và 2. Mỗi chuỗi và cả bốn cấu hình dùng chung cache YOLOv8n COCO với ngưỡng 0,25. Đây là thử nghiệm thăm dò ô tô trên PC. [Giao thức](00_protocol.md)

## FastTrack

Ba ZIP từ kho tác giả khớp SHA-256 và cả 5.375 JPEG giải mã đúng kích thước. Mỗi ZIP có hai metadata đều ghi 30 FPS. Dữ liệu dùng lớp 6 sau khi kiểm ảnh thực tế, gồm 137.480 hộp và 257 ID tính theo chuỗi. Điểm dùng nhãn từ thư mục `train` của bản công bố, với ba chuỗi chọn theo tên cảnh. [Phiên bản nguồn](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark/tree/7e16fad855e64fae3fe7754f1bec8c8fbb50fd87/train)

- Toàn bộ ba chuỗi có IDSW **318 / 297 / 302 / 372,67** và HOTA **30,35 / 32,29 / 32,17 / 31,48%**. EKF và UKF đều ít đổi ID hơn KF trên cả ba chuỗi. IDSW của từng chuỗi ngày che, ngày rẽ phải kèm che và hầm lần lượt là **86/82/84**, **159/144/152** và **73/71/66** cho KF/EKF/UKF.
- Trong 366 hộp thuộc sáu đoạn đổi hướng trên ảnh ≥25°, IDSW **0 / 0 / 0 / 0,33**, FN **78 / 78 / 78 / 84,33**. Mẫu rẽ này chưa cho thấy ưu thế của EKF hoặc UKF.
- 15 lượt ô tô bị xe buýt hoặc xe tải che đã được xác nhận trên ảnh trước, lúc che và sau khi xe gây che đi qua. Cả bốn bộ lọc được ghép ở hai đầu cho cùng **8/15 lượt**. Trong tám lượt đó, giữ ID **0 / 4 / 5 / 5,33** và đổi ID **8 / 4 / 3 / 2,67**. Trong bảy lượt còn lại, cả bốn thiếu ghép ở đầu trong năm lượt và thiếu ghép ở cuối trong hai lượt. Bảy xe gây che gồm năm xe trong chuỗi che ban ngày và hai xe trong chuỗi rẽ phải. [Mốc theo sự kiện](fasttrack_occlusion_events.json) · [Kết quả từng sự kiện](scores/fasttrack_verified_occlusion.json)
- Độ phủ ứng viên Car ở IoU ≥0,5 là **40.576/137.480 (29,51%)**. Với chiều cao hộp dưới 25 px, chỉ **23/21.147 (0,11%)** có candidate. Với hộp cao từ 25 px, có **40.553/116.333 (34,86%)**. Đây là cận trên trước ghép một-một.

Nhóm 15 lượt là kiểm tra thăm dò, được tìm bằng hộp của xe tải/xe buýt đi trước phủ ≥45% hộp ô tô trong 6–120 khung, đỉnh ≥65%, rồi xem ảnh để giữ các lượt xác nhận được. Số liệu tính ở hai mốc cách cửa sổ ứng viên 10 khung. Đã xem 19 ứng viên và giữ 15 lượt thấy rõ xe gây che. Visibility trong ZIP đều ghi 1, kể cả khi xe bị xe khác che trên ảnh. Bốn gián đoạn nội bộ trong nhãn Car không đủ đáng tin để dùng làm chỉ báo che khuất. Chưa xác nhận sự kiện qua vật cố định. Mẫu ảnh có một xe máy chạy qua và một xe máy đỗ, chưa đủ cho phép so sánh riêng lớp này.

## KITTI Tracking

Ba chuỗi 0001, 0011 và 0020 có số hộp Car gắn cờ occlusion=2 cao nhất trước khi áp dụng các quy tắc bỏ qua. Chấm 1.657 khung ở 10 FPS bằng TrackEval Kitti2DBox, giữ cách bỏ qua Van, DontCare, đối tượng cắt biên và các hộp không hợp lệ của trình chấm. [Giao thức KITTI](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)

- Toàn ba chuỗi có IDSW **111 / 109 / 108 / 129** và HOTA **49,32 / 48,79 / 48,94 / 48,31%**.
- Trong 2.205 hộp Car có cờ che nặng, IDSW **20 / 16 / 17 / 19,33** và FN **1.752 / 1.767 / 1.764 / 1.762,33**. EKF và UKF giảm IDSW nhưng bỏ sót thêm lần lượt 15 và 12 hộp.
- Với 71 sự kiện che nặng đủ nhãn hai đầu, KF/EKF/UKF cùng giữ đúng ID **15/71**, đổi ID hai lượt, thiếu ghép đầu 45 lượt và thiếu ghép cuối chín lượt. PF giữ **15/13/15** lượt theo ba seed.

## Nguồn khác và hậu kiểm

TSBOW trả HTTP 401 cho file nguồn, và tài liệu mô tả nhãn dùng cho detection. FE-DETRAC lỗi phân giải tên miền trong phiên này. TUMTraf V2X tải qua đăng ký và chưa có link gói của người dùng. [TSBOW](https://huggingface.co/datasets/SKKUAutoLab/TSBOW) · [FE-DETRAC](https://www.scidm.tw/en/dataset/fe-detrac) · [TUMTraf](https://innovation-mobility.com/en/project-providentia/a9-dataset/)

Đã xác minh 6 hash cache, 6 hash GT chuyển đổi và 36 hash MOT. IDSW và FN từ bảng ghép CLEAR khớp kết quả TrackEval ở cả 36 lượt. Tổng IDSW, FN và FP cộng đủ từ từng chuỗi. Toàn bộ cache và MOT nằm dưới `F:/LABOTARY_data/goc_tai_ve/FastTrack-R115`. Mã phát lại và báo cáo lưu ở thư mục này của kho mã. Điểm tổng và theo từng chuỗi nằm trong [thư mục scores](scores/).
