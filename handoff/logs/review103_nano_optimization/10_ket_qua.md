# R103 — bốn phép thử tối ưu Jetson Nano

Đối chứng là YOLO11n mixed5 50/50 R102, TensorRT FP16 384×640, ByteTrack KF, cutoff 0,25. Cùng bốn đoạn BDD ở 5 fps và bốn đoạn UA-DETRAC ở 25 fps, mỗi đoạn 150 JPEG, cùng ảnh trên tmpfs và tắt hiển thị. Hoàn thành **40/40 lượt**. Cả hai pha có MOT đối chứng khớp từng byte với R102, tổng **16/16**. Đo tốc độ bằng trung vị mức thay đổi FPS ổn định từng cặp. HOTA gộp bằng TrackEval trên đúng các đoạn GT, với cùng quy tắc ignore. Tối đa 50 °C, lỗi ext4 bằng 0. Pha pipeline có khoảng 214 MB swap nền và đọc thêm 7 trang; pha student có khoảng 218 MB swap nền và không đổi số trang đọc/ghi. Đây là phép đo trên các đoạn thử, chưa đo hiển thị hoặc camera trực tiếp.

## Chạy YOLO cách một khung

Gói thử đọc đủ 150 khung và đánh dấu riêng 75 khung không chạy YOLO. Khung có cờ này vẫn tăng thời gian và dự đoán tracker, nhưng không chạy vòng ghép, tạo ID hoặc đánh dấu Lost do detector rỗng. Smoke trên Nano kiểm cùng ID sau khung bỏ suy luận, rồi kiểm chuyển Lost khi một khung chạy detector thực sự không có detection. Chỉ thay trong bản sao thí nghiệm.

- **BDD 5 fps:** FPS trung vị **14,50 → 23,54**, tăng ghép cặp **61,52%**. P90 trễ xử lý **78,80 → 76,99 ms**. HOTA **45,64 → 34,31%**, IDSW **45 → 105**, FN **1.669 → 2.095**. Trong 97 GT ID xuất hiện sau đầu đoạn, số ID được khớp ít nhất một lần **58 → 50**. Trong 49 ID được khớp bởi cả hai nhánh, thời gian khớp đầu tiên tăng trung vị **600 ms** theo thời gian nguồn 5 fps.
- **UA-DETRAC 25 fps:** FPS trung vị **14,47 → 25,40**, tăng ghép cặp **75,53%**. P90 trễ xử lý **86,54 → 84,13 ms**. HOTA **68,66 → 68,41%**, IDSW **49 → 23**, FN **1.810 → 1.911**. Trong 16 GT ID xuất hiện sau đầu đoạn, cả hai nhánh khớp cùng 14 ID ít nhất một lần. Thời gian khớp lần đầu trên 14 ID tăng trung vị **80 ms** theo thời gian nguồn. Đây là ứng viên tăng tốc mạnh nhất cho UA trong phép thử. BDD mất nhiều chất lượng ở cadence này.

## Cắt vùng camera cố định

Chọn dải toàn chiều ngang trước khi xem điểm bằng ảnh mẫu đầu mỗi camera. Dải cao 384 px trên ảnh 960×540, giữ tỷ lệ co theo chiều ngang khi đưa vào engine **256×640**. Cắt trong detector sau khi đọc ảnh gốc, cộng lại tọa độ dọc vào hộp trước tracker, chấm GT toàn ảnh. Có **255/9.279** hộp GT hoàn toàn ngoài dải.

GPU `trtexec` A/B/B/A **41,61 → 29,07 ms (−30,14%)**. Trên bốn đoạn UA, FPS **14,47 → 18,38** và tăng ghép cặp **26,97%**, p90 **86,54 → 71,50 ms**. HOTA **68,66 → 65,13%**, IDSW **49 → 43**, FN **1.810 → 2.262**, FP **1.243 → 1.510**. HOTA giảm trên cả bốn đoạn. Cấu hình ROI 256×640 này đánh đổi đáng kể chất lượng.

## Lọc confidence trước NMS

Đặt cutoff detector **0,25** trước NMS trong nhánh live đang dùng cùng cutoff tracker. Hai clip BDD và hai clip UA cho MOT và lớp trùng byte **4/4**. FPS tăng **2,29% BDD** và **2,04% UA**. Hậu xử lý trung vị **3,71 → 2,22 ms** BDD và **6,42 → 5,03 ms** UA. Kết quả áp dụng cho giao thức c25 và bốn đoạn đã thử. Nếu cấu hình tracker dùng detection điểm thấp, cutoff detector cần giữ đủ dải confidence được dùng.

## Detector YOLO11n nhẹ hơn đã huấn luyện lại

Bỏ attention trong C2PSA và nhánh bottleneck cuối backbone C3k2. Số tham số **2.590.815 → 2.259.167 (−12,8%)**. Giữ đầu vào 384×640. Fine-tune ba epoch với 1.200 ảnh mixed5 train và 200 ảnh RideSafe train từ R101. Chấm dev mixed5 7.190 ảnh và hai clip RideSafe test R102 ngoài ngày train. Checkpoint best SHA `741f1c87dc2fcb68bbd2fb404558b26674b8d99ed3f09187a4317161a419b0b7`. Engine thử SHA `b46d02d99f448dc2a3e2c08fa09c5520ec62e6c792510237563dc5cb81520787`.

GPU **41,64 → 40,24 ms (−3,36%)**. FPS tăng **1,17% BDD** và **1,93% UA**. HOTA BDD **45,64 → 44,18%**, IDSW **45 → 51**. HOTA UA **68,66 → 65,44%**, IDSW **49 → 53**. Mixed5 dev mAP50 **49,56 → 45,02%**. Trên hai clip RideSafe, recall xe máy c25 **32,28 → 42,29%**, precision **75,17 → 65,59%**, KF HOTA **12,02 → 17,85%**, IDSW **19 → 24**. Lợi ích RideSafe trong mẫu nhỏ đi cùng độ chính xác giảm ở các tập dùng để đo Nano và trên dev năm lớp.

## Bằng chứng và kết thúc

`pipeline/assessment.json` và `student/assessment.json` có tốc độ, chất lượng và thời gian khớp lần đầu theo đoạn. `pairs.json` lưu từng lượt, `engine_abba.json` lưu phép đo GPU, `mot_outputs.tar.gz` chứa đầu ra đã chấm. Script và giao thức trong cùng thư mục. Tất cả lượt thử đã kết thúc. Engine và thay đổi pipeline nằm trong `~/nano_run/r103_optimization/` dưới gói thử riêng. Chưa chuyển launcher.
