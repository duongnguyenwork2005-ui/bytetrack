# R114 — bốn bộ lọc trên Nano và bản khóa luận đã cập nhật

KF và EKF cùng đạt ít nhất 15 FPS trên cả 16 đoạn. UKF đạt 14 đoạn. RBPF500 đạt sáu đoạn, do thời gian cập nhật tracker tăng ở các đoạn đông xe. Tốc độ được đo trực tiếp trên Jetson Nano với cùng YOLO11n 50/50, TensorRT FP16 384×640, JPEG từ tmpfs và tắt hiển thị. Mỗi đoạn có 150 ảnh, bỏ 25 khung đầu trong tổng hợp tốc độ. Phép so có tám đoạn mỗi tập, lấy cách đều từ danh sách 40 đoạn mỗi tập R109. RBPF500 dùng seed 0.

Kết quả mỗi dòng theo thứ tự KF, EKF, UKF, RBPF500.

- BDD100K, FPS trung bình **19,60 / 19,61 / 19,62 / 13,80**. Trung vị p90 theo đoạn **115,5 / 116,1 / 130,3 / 611,9 ms**. Số đoạn đạt mốc **8 / 8 / 8 / 4** trên tám đoạn.
- UA-DETRAC, FPS trung bình **18,01 / 18,04 / 16,39 / 8,68**. Trung vị p90 theo đoạn **124,0 / 127,8 / 146,7 / 911,5 ms**. Số đoạn đạt mốc **8 / 8 / 6 / 2** trên tám đoạn.
- Gộp 16 đoạn, FPS trung bình **18,80 / 18,83 / 18,01 / 11,24**. Trung vị p90 theo đoạn **121,1 / 121,9 / 143,1 / 833,3 ms**. FPS thấp nhất **17,06 / 17,10 / 8,93 / 2,10**.

Ở đoạn MVI_40855, GPU chạy khoảng 42 ms mỗi khung ở cả bốn lượt. Thời gian cập nhật tracker trung vị là **25,43 / 34,84 / 110,33 / 470,25 ms**. RBPF500 tại đây đạt **2,10 FPS** với p90 **3,03 giây**. EKF là cấu hình phi tuyến đã đạt mốc tốc độ ở toàn bộ mẫu. Chất lượng RBPF tốt hơn PF500 K1/K2 ở phép phát lại PC R113 và có giới hạn về tốc độ trên Nano.

Tám lượt smoke riêng đạt đầy đủ. 64 lượt chính hoàn thành 9.600 khung suy luận. Cache detection trùng SHA256 với R110 ở 64/64 lượt, KF trùng byte MOT và lớp của R109 ở 16/16 đoạn. Toàn bộ MOT khớp số khung, ID, điểm và tọa độ phát lại PC trong dung sai 0,001 pixel. FPS và p90 được tính độc lập từ dữ liệu timestamp rồi so với log của runner. Cùng mã boot, lỗi SD bằng 0, swap tăng từ 5,99 lên 14,69 MiB. Nhiệt độ lớn nhất của các trường CPU/GPU/thermal trong log là 38 °C. Lượt smoke KF đầu mất 68,9 giây khởi tạo, được tách khỏi thời gian xử lý. Phép đo dùng gói thử riêng.

[Kết quả máy đọc và từng đoạn](measure/verified_summary.json) · [Giao thức](00_protocol.md) · [Log Nano](nano_measure.log) · [Bằng chứng gốc](measure_evidence.tar.gz)

## Khóa luận

Bản A4 43 trang giữ HELMET cho xe máy, RoundaboutHD cho đổi hướng và UA-DETRAC cho che khuất. Chương thực nghiệm bổ sung bảng IDSW/FN theo nhóm, đối chiếu theo ngưỡng detector trên SVMOT và kết quả ô tô, xe máy trên BDD/METEOR. Bootstrap theo 40 đoạn UA-DETRAC cho khoảng phân vị 95% của chênh lệch IDSW nhóm đang che là [−8, 2] với EKF và [−15, 3] với UKF. Khóa luận ghi đây là mức giảm quan sát được và cả hai khoảng đều chứa 0. Chương Nano tách điểm chất lượng trên 80 đoạn phát lại PC khỏi phép đo tốc độ trên 16 đoạn Nano. Đã kiểm biên dịch, phông nhúng, trích xuất nội dung, bố cục tóm tắt và hai bảng mới, CRC và nội dung zip 22 tệp.

[PDF R114](../../../outputs/khoa_luan_Nguyen_Huu_Thai_Duong_R114_20261118.pdf) · [Gói nguồn LaTeX R114](../../../outputs/khoa_luan_Nguyen_Huu_Thai_Duong_LaTeX_R114_20261118.zip)

## V2X-Seq-SPD

Hai gói mẫu, ảnh camera cố định train/val và metadata train/val tiếp tục trả trang giới hạn tải của Google Drive. Quy trình xác nhận tải chính thức của cả hai gói mẫu cũng trả trang giới hạn tải. Cổng tác giả BAAI không kết nối được từ PC. Giao thức nhánh camera cố định giữ lớp Car và Motorcyclist, dự kiến gắn nhãn sự kiện trước khi chấm. Công cụ kiểm tra metadata, ảnh, 2D box, ID và mã che đã chạy qua fixture mô phỏng dựa trên schema công bố. Khi có archive gốc, cần kiểm lại ánh xạ ảnh, nhãn và ý nghĩa mã che trước khi chạy thí nghiệm. Hiện chưa có kết quả tracking V2X.

[Nguồn dữ liệu và đường tải chính thức](https://github.com/AIR-THU/DAIR-V2X-Seq) · [Kết quả truy cập](v2x_access.json) · [Kiểm xác nhận tải](v2x_confirmation.json) · [Công cụ kiểm ảnh và nhãn](audit_v2x_inputs.py)
