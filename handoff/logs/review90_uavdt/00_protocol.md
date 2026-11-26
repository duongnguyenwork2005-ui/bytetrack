# R90 — UAVDT: kiểm tra tập xe thay cho MOT17

Ngày khóa giao thức: 06/11/2026, trước khi tải/chấm. Người dùng xác nhận phạm vi chỉ theo dõi phương tiện. R89 MOT17 là ngoài phạm vi và không được dùng để chọn cấu hình.

## Nguồn và cổng chấp nhận

- Ưu tiên gói `UAVDT-Benchmark-M` và `DET/MOT toolkit` do tác giả liên kết tại trang chính thức https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5 . Không sử dụng phần SOT hay nhãn người.
- Kiểm đủ byte, kiểm CRC và SHA256 cục bộ; lưu manifest cùng nguồn URL. Xem cấu trúc ZIP trước giải nén lớn.
- Chỉ chấp nhận clip có ảnh liên tiếp, GT theo `(frame, track_id, bbox)` và lớp xe (car/truck/bus). Kiểm số thứ tự frame, ID trùng trong frame, box hợp lệ, và split; không dùng nhãn test ẩn. Nếu gói thiếu GT MOT công khai, dừng và báo.
- Tác giả ghi video 30 fps; phải xác minh ảnh phát hành tương ứng các frame liên tiếp của chuỗi đã chọn trước khi dùng buffer 1,2 s = 36 bước. Nếu không xác minh được, dừng benchmark tracker.

## Giao thức dự kiến sau cổng dữ liệu

Chọn tập con trước khi nhìn điểm bộ lọc: ưu tiên các chuỗi train có xe đổi hướng/che khuất dựa vào thuộc tính nguồn, đồng thời lấy mẫu điều kiện thường; chỉ sử dụng GT công khai. Ghi danh sách và quy tắc chọn trước detector smoke. Dùng cùng một cache detection cho CV/EKF/UKF/PF, chỉ lớp xe, cùng checkpoint và ngưỡng; đo recall class-agnostic IoU≥0,5 tại conf >0,05 trên mẫu ảnh hệ thống. Nếu recall quá thấp để so tracking, dừng tại smoke. PF dùng ba seed. Báo HOTA/IDF1/MOTA/IDSW theo clip và gộp, tách lỗi detector; đây là phép thử chuyển miền nội bộ, không so điểm trực tiếp với leaderboard tác giả. Không đổi tracker/YAML luận văn/launcher Nano theo phép thử này.

Nhãn đã kiểm trước detector: 30 clip train, GT MOT và GT DET đi kèm cùng `(frame, ID, box)`, lớp DET chỉ 1/2/3 = car/truck/bus. Quy tắc chọn cố định trong `selection.json`: clip có ≥300 frame, ≥10 ID, ≥1.000 GT hợp lệ; lấy 4 clip có tỷ lệ box che khuất cao nhất, 2 clip còn lại có tỷ lệ đổi hướng trong cửa sổ 10 frame cao nhất, 2 clip còn lại có tỷ lệ che khuất thấp nhất, hòa thì theo tên. Danh sách khóa là **M1306, M1005, M1202, M0206** (che khuất), **M0204, M0501** (đổi hướng), **M0301, M1008** (đối chứng). Tổng 4.367 frame. Đây là chọn theo nhãn nguồn trước khi xem điểm detector hoặc tracker, không phải chọn clip PF thắng.

Smoke detector cố định 10 frame rải đều mỗi clip, ba checkpoint R54 (YOLOv8n COCO, YOLOv8n mixed5, YOLO11n mixed5), conf >0,05, IoU ghép ≥0,5, chỉ xe car/truck/bus. Cổng mở full cho **một detector tốt nhất theo recall gộp** là recall ≥20% và ít nhất 4/8 clip ≥15%; nếu không đạt thì chỉ báo smoke, không xếp hạng bộ lọc trên UAVDT. Cổng đạt với YOLOv8n COCO (510/979 = 52,09%; 7/8 clip). Ma trận full cố định c25/c10, CV/EKF/UKF/PF500 mặc định và PF500 K1/K2 motion100 opt-in, PF ba seed 0/1/2, cùng cache detector, buffer 1,2 giây/36 bước. K1/K2 chỉ là ứng viên thăm dò, không đổi cấu hình đã khóa. Bản chuyển GT đưa `gt_ignore` thành lớp distractor 8 của TrackEval; đây là quy ước đánh giá nội bộ. Đã đối chiếu mã MATLAB tác giả: đánh giá chính thức bỏ prediction nằm trọn trong box ignore, nên hai quy tắc có thể cho số FP/metric khác nhau và **không so với leaderboard chính thức**.

Trang nguồn mô tả ba lớp car/truck/bus, thuộc tính che khuất, 30 fps, 1080×540, và MOT train/test. **Tám ảnh đầu chuỗi trong ZIP tải về thực tế là 1024×540**, cần kiểm tất cả ảnh dùng để chấm; không tự kéo ảnh hay box về kích thước trang giới thiệu. UAVDT không có xe máy; UTUAV và HELMET mới là bằng chứng trực tiếp cho xe máy.
