# Chuẩn bị nhận SVMOT và UTUAV — 31/10/2026

**Cập nhật 01/11/2026:** hai archive UTUAV đã tải xong và pilot đã chạy. Xem kết quả mới tại `02_pilot_utuav.md`; phần dưới là trạng thái chuẩn bị tại 31/10.

## Tình trạng

- Người dùng đang tải `svmot_data_v1.0.zip` và `UTUAV_B_C_3and6class.tar.gz` (ảnh B/C). Chưa có tệp hoàn chỉnh để kiểm trực tiếp.
- Môi trường `khoaluan-mot` có NumPy, SciPy, OpenCV, Torch, Ultralytics, TrackEval và YAML. YOLOv8n cùng cấu hình ByteTrack KF/CV, EKF/CTRV, UKF/CTRV, PF/CTRV đã có.
- UTUAV còn cần **gói nhãn MOT riêng** `UTUAV_MOT_VOC_3_6classes.tar.gz` để chấm IDF1/HOTA/IDSW. Gói ảnh B/C không đủ để đánh giá tracker có nhãn chuẩn.
- Ổ C trống 37,9 GiB, ổ D trống 46,8 GiB lúc kiểm. Hai archive đang tải khoảng 34,2 GB theo trình duyệt; không giải nén đồng thời hoặc sao chép thêm bản đầy đủ khi chưa đo dung lượng thực tế.

## Khi tải xong

Chạy tiền kiểm chỉ đọc từ gốc repo, sửa đường dẫn nếu trình duyệt lưu ở nơi khác:

```powershell
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/external_mot_preflight.py --downloads-dir C:/Users/Mikuno/Downloads
```

Tiền kiểm mặc định đọc danh mục ZIP SVMOT và header gzip UTUAV; nó **không chứng minh nội dung archive nguyên vẹn**. Sau khi tệp tải xong, thêm `--full-check` để đọc hết và kiểm CRC (có thể lâu), đồng thời đối chiếu checksum nếu nguồn phát hành có công bố. Chỉ giải nén các chuỗi đã chọn trên ổ còn đủ chỗ. Không đặt mật khẩu UTUAV trong lệnh, mã hay hồ sơ.

## Cổng trước benchmark

1. Kiểm cây thư mục thực tế, nhãn GT, FPS, độ phân giải, lớp, ID, khung không có nhãn và quy định ignore/occlusion của từng bộ; không tự suy từ tên archive. Chốt ánh xạ lớp sang năm lớp của dự án và phát hiện lớp ngoài phạm vi.
2. Chọn trước các chuỗi rẽ/đổi hướng, che khuất ngắn, xe máy nhỏ và nhóm đối chứng đi thẳng. Tách theo **chuỗi**, ghi danh sách/khung cụ thể trước khi nhìn điểm số; không chọn đoạn theo kết quả thuận lợi cho bộ lọc nào.
3. Chạy một detector và lưu cùng một detection cache cho KF/EKF/UKF/PF. Giữ nguyên tracker YAML, ngưỡng, `track_buffer`, phép chấm và mã nguồn đã khóa; mọi thay đổi giao thức cần người dùng chốt riêng.
4. Kiểm converter nhãn bằng vài frame đối chiếu ảnh và GT; chỉ sau đó chạy smoke ngắn. Ghi HOTA, IDF1, IDSW, số frame, thời gian, cùng lỗi theo nhóm tình huống. Không coi UTUAV `occlusion` là nhãn mức che khuất nếu trường này chỉ là placeholder.

Chưa làm adapter/đổi nhãn hoặc chạy benchmark vì hai archive chưa xong và giao thức dataset mới chưa được chốt. Phần R63/Nano đang dở được giữ nguyên.
