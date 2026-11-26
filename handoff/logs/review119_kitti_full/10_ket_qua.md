# R119 — KITTI Tracking đầy đủ ô tô và khảo sát nguồn mới

Ngày 19/11/2026, người dùng cho phép tự nghiên cứu khi nghỉ. Không chỉnh khóa luận, slide, cấu hình tracking hoặc mặc định. Giao thức được lưu trước khi đọc kết quả ở `00_protocol.md`.

## Phạm vi và kiểm chứng

KITTI Tracking training có 21 chuỗi cục bộ. Chuỗi 0017 không có Car hợp lệ theo evaluator nên chỉ audit, không chạy benchmark người. Toàn bộ 20 chuỗi còn lại có **7.863 ảnh / 24.070 hộp Car**. Dùng cùng YOLOv8n COCO c25, 10 FPS, buffer một giây, native KITTI evaluator xử lý Van/DontCare/truncation. Giữ định nghĩa nhóm rẽ trên ảnh R115, không gọi đó là quỹ đạo trong thế giới vì camera gắn xe.

Hoàn tất **60 lượt** KF/EKF/UKF. Chín lượt ở ba chuỗi R115 khớp toàn bộ metric, nhóm, khôi phục ID và hash MOT. Mỗi lượt tái ghép CLEAR để đối chiếu IDSW/FN trước khi cộng nhóm. Hậu kiểm 35 hash nguồn, 20 cache, 60 MOT và **7.863 hash ảnh** trước/sau đều đạt. Chi tiết trong `verification.json` và `image_verification.json`.

## Kết quả theo đúng thứ tự KF / EKF / UKF

Toàn 20 chuỗi: **IDSW 280 / 260 / 260**, HOTA **49,764 / 49,680 / 49,717%**, IDF1 **65,157 / 65,105 / 65,223%**, FN **9.416 / 9.422 / 9.416**, FP **1.852 / 1.875 / 1.866**. EKF và UKF giảm IDSW 7,14%, HOTA gần nhau. Bootstrap ghép cặp 20 chuỗi, 10.000 mẫu, chênh IDSW so KF: EKF **[−52,+5]**, UKF **[−47,+3]**, chưa loại được khả năng không khác biệt.

Phần mới thực sự có 17 chuỗi / 6.206 ảnh / 14.046 hộp. IDSW **169 / 151 / 152**, HOTA **50,069 / 50,295 / 50,256%**, FN **5.431 / 5.413 / 5.413**. Khóa JSON `new18` là tên nội bộ được ghi trước khi audit; 18 chuỗi chưa chạy gồm một chuỗi không có Car, danh sách thực chấm chỉ có **17**, không phải 18. Không cộng lại ba chuỗi cũ vào phần mới.

Nhóm rẽ ≥25° trên ảnh chỉ có **399 hộp**, IDSW **8 / 8 / 8**, FN **96 / 106 / 104**. Nhóm che nặng do KITTI gắn cờ có **4.259 hộp**, IDSW **43 / 42 / 42**, FN **3.384 / 3.415 / 3.411**. Giao rẽ và che chỉ có 50 hộp, IDSW 3/2/2 nhưng FN 30/37/37. Không dùng giao nhóm nhỏ và bỏ sót nhiều hơn để tuyên bố phi tuyến thắng. Có 177 lượt che đủ điều kiện GT ở hai đầu; từng bộ lọc giữ ID 21 lượt, đổi ID 12 lượt, còn lại thiếu ghép một đầu. Đây là đếm theo từng bộ lọc, không phải mẫu đủ ghép chung đã cân bằng.

**Kết luận:** có tín hiệu giảm IDSW toàn bộ KITTI, chưa cho thấy lợi thế ở nhóm rẽ hoặc khôi phục ID sau che. KITTI không thay được bằng chứng RoundaboutHD/HELMET hay trở thành một benchmark phi tuyến mới chỉ từ số tổng này.

## Nguồn dữ liệu được kiểm tra

- **GMOT-40:** tác giả phát hành [nhãn ID và ảnh trên GitHub](https://spritea.github.io/GMOT40/download.html). Tải riêng đủ bốn chuỗi car bằng HTTP Range của ZIP release v0.1, kiểm CRC nguồn và SHA từng ảnh, không tải các lớp người/động vật. Dữ liệu ở `F:/LABOTARY_data/goc_tai_ve/GMOT40-R119`, manifest `source_probes/gmot_downloads.json`. Đã kiểm thử tiếp ở R121/R122; chưa có cadence từng chuỗi được xác nhận.
- **DSEC-MOT:** [repo tác giả](https://github.com/CASR-HKU/DSEC-MOT) có box, ID và lớp motorcycle, nhưng nhãn thuộc event camera 640×480; không thể lấy box đó áp trực tiếp lên RGB camera khác. Link Google Drive duyệt được thư mục gốc chứa train/test, cả hai trả danh sách rỗng lúc kiểm; gdown không tải được nhãn. Đây là trạng thái truy cập hiện tại, không khẳng định dữ liệu không tồn tại. Log/HTML đã lưu ở `source_probes/dsec*`.
- **MEVDT:** [bài báo](https://arxiv.org/abs/2407.20446) công bố grayscale frames và nhãn tracking 24 Hz, camera cố định. [DOI dữ liệu](https://doi.org/10.7302/d5k3-9150) tiếp tục trả 403 từ PC, không tải ảnh hoặc chấm.
- **TRAF:** tìm được [link tác giả còn hoạt động](https://go.umd.edu/TRAF-Dataset), lấy listing 5.557 file; phần lớn video/ảnh và một det.txt. Đọc danh mục ZIP12 (698 file) và ZIP20 (1.211 file) bằng HTTP Range, chỉ thấy ảnh PNG, không có nhãn TXT/XML/CSV/JSON/INI. Không tải trọn hai ZIP 911/351 MB; chỉ cần khoảng 54/93 KB để audit danh mục. Mẫu TestSet.txt trong repo TrackNPred tác giả có năm cột frame/ID/toạ độ/giá trị bổ sung, không đủ bốn tọa độ hộp; không dựng hộp giả từ các điểm đó. Chưa có căn cứ các tài nguyên đã liệt kê chứa đủ GT bbox/ID/cadence để chạy MOT. Listing và audit ở `source_probes/traf_listing.json`, `source_probes/traf_zip_audit.json`.
- **IDD-X:** [trang tác giả](https://idd-x.github.io/) yêu cầu đăng ký và request tải, nhãn chỉ dành cho important objects. Chưa phù hợp để đưa mọi detection phương tiện vào evaluator thường vì các xe khác có thể không được gắn nhãn.

Chưa tìm thêm tập mới đủ bằng chứng rằng EKF và UKF cùng hơn KF trên cả rẽ và che. Giữ các trường hợp không thuận trong báo cáo; không chọn dataset bằng việc bỏ các chuỗi KF thắng.
