# R62 — chốt ứng viên trước khi sửa mã

Ngày 28/10/2026. Căn cứ baseline ngắn 8/8 lượt đạt và ba phép đo riêng trên Nano, chọn **một điểm tối ưu: chuyển uint8 BGR HWC thành float32 RGB NCHW** trong `nano_detector.preprocess`. Không đổi letterbox, engine, detector threshold, NMS, tracker hay cách đo thời gian.

Baseline ngắn: tiền xử lý gồm letterbox + chuyển tensor có trung vị khoảng 13,3–14,0 ms/frame. Các cách chia tại chỗ và đổi thứ tự copy giảm ít. Phép đo v3 (`16_so_tien_xu_ly_v3.json`) trên OpenCV/NumPy của Nano, 8 vòng cân bằng: legacy 9,778 ms, `np.take` 6,745 ms, OpenCV LUT trên HWC 4,658 ms, **OpenCV LUT trên CHW uint8 được làm phẳng thành một cột: 2,396 ms**. 11 trường hợp × 4 cách đều trùng byte, dtype, shape, layout; input không bị sửa. Những số này chỉ đo chuyển tensor, không được gọi là cải thiện FPS pipeline.

Ứng viên: tạo một bảng 256 giá trị bằng đúng phép chia float32 `arange(256) / 255.0`; chuyển BGR→RGB/CHW ở uint8; `cv2.LUT` đọc một cột uint8 một kênh, trả float32; reshape về NCHW liên tục. Bảng chỉ đọc, không dùng buffer ảnh chung. Input khác uint8 hoặc rỗng dùng lại đường cũ để giữ hành vi API. Không nhân nghịch đảo để tránh khác bit.

Kiểm trước đo: giữ tương thích cú pháp Python 3.6; kiểm tensor so với torch trên toàn miền uint8 và các layout; chạy test detector hiện có trên PC; kiểm tensor trên Nano với mã production. Sau đó đối chiếu detection/MOT/lớp với v16 và đo theo cặp như kế hoạch. Chỉ nghiệm thu tăng tốc khi đạt các điều kiện FPS, p90, frame, tài nguyên đã chốt.

Baseline dài đầu tiên đã bị gián đoạn do Nano reboot trước khi có kết quả, đang chờ người dùng xác nhận nguyên nhân. Có thể chuẩn bị mã và kiểm PC trong thời gian này; chưa triển khai launcher hoặc gọi ứng viên là đạt.
