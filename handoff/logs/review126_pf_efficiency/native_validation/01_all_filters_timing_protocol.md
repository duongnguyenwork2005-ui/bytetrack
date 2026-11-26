# Đo bổ sung chi phí so KF/EKF/UKF

Khóa sau khi R126-A2 đạt kiểm đủ 600 lượt BDD, trước kết quả đo bổ sung.
Lý do: yêu cầu hiện tại đặt câu hỏi liệu PF còn đáng dùng khi tốn thời gian hơn các bộ lọc khác.
18 lượt trước chỉ so các phiên bản PF; cần cùng lượt đo cho KF/EKF/UKF.

Giữ hai video đã khóa theo workload của R125, mỗi video 200 khung, bỏ 16 khung đầu,
seed 0, cùng cache/detection/FPS/buffer và association. Đo tuần tự KF/EKF/UKF/native PF100,
ba lần, đảo thứ tự ở lần thứ hai: 24 lượt. Không chạy đồng thời với quality jobs.
Chỉ thống kê `tracker.update` PC; không suy FPS Nano hoặc tốc độ toàn pipeline.
Kiểm nguyên frame/ID, confidence và bbox so MOT R125 ở mỗi prefix, và lặp lại
phải trùng mảng hoàn toàn trong từng mode/video. Không thay tham số theo tốc độ.

Kết quả bổ sung là đối chiếu, không thay thế 18 lượt đã ghi hoặc dùng để chọn lại
gate của R126. Lưu các giá trị và sáu cặp theo video/lần, kể cả khi PF vẫn chậm hơn.
