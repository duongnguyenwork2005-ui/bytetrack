# R114 — bằng chứng theo kịch bản và thông lượng theo bộ lọc

## Nano

Mẫu đối chứng gồm tám đoạn BDD và tám đoạn UA-DETRAC. Lấy chỉ số 0, 5, 10, 15, 20, 25, 30, 35 từ 40 đoạn mỗi tập đã khóa trong R109. Mỗi đoạn có 150 JPEG liên tiếp. Chạy bốn bộ lọc trên cùng ảnh: KF, EKF, UKF, RBPF500 seed 0. Luân phiên thứ tự theo vị trí và tập. Smoke riêng dùng đoạn đầu mỗi tập, chạy đủ bốn bộ lọc. Kết quả smoke không cộng vào 64 lượt chính.

Mỗi lượt dùng engine TensorRT FP16 YOLO11n 50/50, kích thước 384x640, cutoff 0,25 trước NMS ma trận, giải mã gối đầu, hàng đợi 4, không bật hiển thị. BDD dùng 5 Hz và elapsed steps, UA dùng 25 Hz và elapsed skip. Giữ nguyên giới hạn track, định dạng hộp, lớp và vùng bỏ qua theo lệnh R109. Bản nguồn riêng sao từ overlap, bổ sung registry UKF/RBPF, seed RBPF theo video và các module R112. Dùng cùng bản nguồn cho cả bốn phép đo.

Khớp mã SHA256 150 JPEG với R109, engine với R109, cache detection từng lượt với cache R110 tương ứng. KF phải khớp MOT và lớp từng byte với R109. Tính FPS ổn định là 125 chia chênh timestamp hoàn tất khung 150 và 25. P90 độ trễ tính từ bắt đầu đọc ảnh đến xong tracking trên khung 26–150 theo cùng quy tắc chọn phân vị của runner. Báo trung bình FPS, thấp nhất, số clip đạt 15 FPS và trung vị p90 theo clip, tách từng tập. Giữ log RAM, swap, lỗi SD, mã boot, cấu hình và đầu ra. FPS áp dụng cho JPEG trên tmpfs, có suy luận trên mọi ảnh, tắt hiển thị.

## V2X-Seq-SPD

Ưu tiên nhánh camera cố định cho lớp Car và Motorcyclist. Chỉ chấm sau khi xác minh timestamp ảnh, nguồn bbox 2D, ý nghĩa từng mã occluded_state và phạm vi hộp Motorcyclist. Dùng sequence và camera làm khóa, không nối ID qua hai camera hay qua khoảng trống thời gian. Chốt độ phủ GT và danh sách sự kiện trước khi xem điểm.

- Đổi hướng xác định từ đường tâm đáy hộp GT với điều kiện góc từ 25°, lọc độ dịch chuyển và yêu cầu dấu góc ổn định như R99. Camera cố định giảm nhiễu do chuyển động camera; góc vẫn tính trên mặt ảnh.
- Che khuất dùng mã che đã xác minh của nhãn. Xét riêng khung đang che và lần đầu xuất hiện lại trong một giây. Không suy rằng mọi khoảng trống nhãn là che. Báo nhóm giao giữa đổi hướng và che khuất khi đủ mẫu.
- Giữ một detection cache và thiết lập ghép, chấm KF/EKF/UKF theo lớp GT hoặc giao thức class-aware đã công bố. Đối chiếu IDSW/FN trên timeline đầy đủ trước khi lấy nhóm sự kiện. Báo IDF1, HOTA toàn video và độ phủ detection theo nhóm, tránh diễn giải IDSW ít khi FN lớn.

Nguồn tác giả là [bản mô tả nhãn](https://github.com/AIR-THU/DAIR-V2X-Seq/blob/main/dataset/v2x-seq-spd/README.md) và [danh mục tải](https://github.com/AIR-THU/DAIR-V2X#dataset-download). Hai gói sample và metadata train/val đều báo quota ở lượt kiểm trực tiếp. Trang dự án BAAI không kết nối được từ PC. Cuộc thi MEIS dẫn về bộ gốc cho dữ liệu, chưa tìm thấy gói ảnh và nhãn độc lập. Số mục và FPS thiết bị không được suy ra từ các phép kiểm nguồn này.
