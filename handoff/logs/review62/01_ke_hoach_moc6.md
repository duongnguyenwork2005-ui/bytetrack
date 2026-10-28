# R62 — Kế hoạch mốc 6: tối ưu tốc độ hệ thống Jetson Nano

Ngày mở: 28/10/2026. Người dùng yêu cầu bắt đầu mốc 6 sau khi mốc 5 đã đóng. Thực hiện trong cửa sổ chat hiện tại. Cập nhật 29/10: người dùng cho phép tự tiếp tục sau reset hạn mức; đã bật lịch mỗi giờ gắn chat, không chạy orchestrator CLI cũ song song. Trạng thái **lúc lập** kế hoạch: mới đối chiếu tài liệu và công cụ trên PC, chưa chạy Nano; trạng thái hiện hành đọc PLAN mục 0.

## 1. Mục tiêu và ranh giới

Mục tiêu chính là tăng **FPS xử lý toàn luồng** của chương trình đang triển khai trên Jetson Nano: đọc ảnh → TensorRT FP16 → ByteTrack CV/EKF → hiển thị HDMI. Báo cáo riêng FPS khi tắt hiển thị, FPS hiển thị, trễ p50/p90, thời gian khởi tạo, bộ nhớ, swap, nhiệt độ và xung. So cùng gói hiện hành `~/nano_run/v16`, cùng nguồn ảnh, engine, cấu hình tracker và điều kiện nguồn. Không suy ra tốc độ từ số PC.

Vòng đầu giữ nguyên checkpoint, engine `~/nano_run/engine/yolov8n_fp16.engine`, TensorRT FP16, ngưỡng c25, lớp [1,2,3,5,7], YAML tracker, buffer 1,2 s, `elapsed_mode=steps`, dữ liệu đánh giá và thuật toán tracking/metric. Chỉ tối ưu cách thực thi pipeline sau khi xác định nút thắt; nếu cần đổi engine/precision, model hoặc tham số thuật toán thì lập quyết định riêng trước khi làm. Gói `v16` và kết quả cũ chỉ đọc; mọi thử nghiệm mới dùng thư mục kết quả mới, bản mã mới nếu có là `v17` hoặc cao hơn, không ghi đè v16.

## 2. Baseline đã có, không dùng thay phép đo mới

- `handoff/REVIEW21_CHI_TIET.md` và `REVIEW22_CHI_TIET.md`: 300 JPG, không hiển thị, CV/EKF khoảng **9,1–9,4 FPS**. Inference TensorRT khoảng **74 ms/frame**, tracker khoảng **6–8 ms/frame**. Các số này gợi ý detector/đọc ảnh chi phối throughput, nhưng chưa chỉ ra một thay đổi tối ưu cụ thể.
- `handoff/REVIEW23_CHI_TIET.md`: 1.470 JPG, bốn lượt CV→EKF→EKF→CV, CV 9,22 và EKF 9,05 FPS. Decode rơi vào hai mức ~11,5/16,3 ms theo lượt nên **không xếp hạng CV/EKF từ FPS đó**. Frame 1–1470 đủ; MOT live/replay trùng byte; nhiệt CPU tối đa 38 °C, GPU 35,5 °C trong bốn lượt.
- `handoff/REVIEW25_CHI_TIET.md`: 300 JPG, bật hiển thị giảm throughput khoảng **2%** so với tắt (CV 9,21 so 9,39; EKF 9,18 so 9,38 FPS, 2 lượt/nhóm); 1.470 JPG bật hiển thị đạt 1470/1470 frame với CV và EKF. `handoff/REVIEW26_CHI_TIET.md`: v16 sửa vòng đời UI, không có benchmark toàn luồng mới trên v16.

Vì gói, điều kiện nhiệt và tầng decode có thể khác hôm nay, các số trên chỉ là mốc lịch sử. Không gọi một số FPS đơn lẻ là cải thiện.

## 3. Trình tự thực hiện

### A. Tiền kiểm chỉ đọc trên Nano

Xác nhận `hostname`, L4T/Ubuntu, Python detector và env `nano`, engine SHA256, manifest gói v16, dữ liệu `MVI_39031` 300/1470 ảnh, dung lượng trống, thiết bị lưu trữ, `nvpmodel`, `jetson_clocks`, xung/thermal, RAM/swap, phiên desktop `:0`, tiến trình tracking đang chạy. Ghi stdout và thời điểm vào hồ sơ R62. Không `sudo`, `apt upgrade`, đổi power mode, đổi clocks, build engine hay dọn dữ liệu trong bước này. Nếu ảnh, engine, gói hoặc nguồn điện/SD không như mốc đối chiếu, ghi sai khác và xử lý trước khi đo.

### B. Đo lại v16 trong điều kiện hiện tại

Chạy cùng 300 JPG gốc `MVI_39031_f0001-0300` tại 25 fps và cấu hình c25 cho CV/EKF. Nếu desktop `:0` sẵn sàng, lịch 8 lượt như vòng 25: CV bật–tắt–tắt–bật, EKF tắt–bật–bật–tắt; nghỉ 20 s giữa lượt, mỗi lượt thư mục mới, bỏ 25 frame đầu khỏi thống kê ổn định nhưng giữ toàn bộ CSV. Nếu chưa có desktop, đo headless trước, ghi rõ chưa có baseline hiển thị. Sau đó chạy 1.470 JPG với hiển thị ở lịch cân bằng CV→EKF→EKF→CV để thấy throughput dài và nhiệt/swap; chỉ chạy khi 300 ảnh đạt.

Mỗi lượt phải exit 0, `hoan_tat`, đủ frame, không bỏ frame xử lý; lưu `run_meta.json`, `pipeline/`, `timing/`, `tegrastats.log`, `hien_thi.json` (khi bật), hash MOT/lớp, các giá trị xung và nguồn. So live với replay từ detection cache của chính lượt và kiểm parity trước khi nhận số tốc độ. Nếu kết quả/thiết bị bất thường, giữ nguyên bằng chứng và không chọn lượt đẹp để thay thế.

### C. Chọn đúng một điểm tối ưu từ profile

Tách thời gian decode, tiền xử lý, infer, hậu xử lý, truyền giữa tiến trình, `tracker.update()`, vẽ/hiển thị và thời gian chờ. Chọn điểm có đóng góp lớn và có khả năng sửa mà vẫn giữ đầu ra. Ghi giả thuyết và tiêu chí trước khi sửa. Mọi sửa mã chạy trong nhánh/gói riêng; kiểm thử PC, kiểm trên Nano và đối chiếu MOT/lớp/detection với v16 trước khi so FPS. Không thay đổi thuật toán hoặc giảm chất lượng để lấy tốc độ.

### D. So sánh v16 với ứng viên và chốt

Dùng cùng ảnh/engine/điều kiện, lịch v16↔ứng viên cân bằng thứ tự, ít nhất bốn cặp cho mỗi cấu hình CV/EKF trên 300 ảnh; lưu **mọi** lượt. Báo cáo trung vị hiệu FPS **theo cặp** và độ phân tán, cùng trễ p90, RAM/swap, nhiệt/xung. Chạy dài 1.470 ảnh với ứng viên nếu phép đo ngắn và parity đạt. Chỉ gọi là tăng tốc khi trung vị FPS xử lý tăng ít nhất **5%** ở cả CV và EKF trong điều kiện có hiển thị, không thiếu/drop frame so với v16, MOT/lớp trùng byte với đối chứng tương ứng, không tăng trễ p90 quá 5% hoặc gây swap/throttling mới. Nếu chưa đạt, giữ v16 là bản dùng và báo cáo kết quả âm.

### E. Báo cáo và triển khai

Ghi cấu hình thiết bị, lệnh, hash đầu vào/mã/engine, số thô từng lượt, phép tính so sánh, parity, giới hạn. Chỉ chuyển launcher sang gói mới sau khi ứng viên đạt và có cách quay về v16; xác minh lại launcher + HDMI bằng người dùng hoặc bài kiểm đã có. Cập nhật PLAN mục 0 trước báo cáo bàn giao. Chưa đưa dữ liệu ảnh/video hoặc cache thô lên GitHub; snapshot mã, báo cáo và bảng tổng hợp nhỏ sau khi kiểm xong.

## 4. Trạng thái truy cập lúc mở mốc

Ban đầu kết nối bị từ chối do chưa có khóa công khai. Người dùng đã thêm khóa; alias `nano` trỏ `10.11.61.127`. Sau trạng thái boot eMMC không nhận thẻ, người dùng xác nhận đã đổi Jetson và dùng lại thẻ cũ; lần kết nối mới thấy root trên SD và đầy đủ đường dẫn v16/engine/env. Cần baseline đo mới trên thiết bị hiện tại; số lịch sử không làm đối chứng trực tiếp cho tối ưu. SSH đã nối lại sau gián đoạn: engine hash và manifest 562/562 đạt; đang đo baseline từ bản sao v16 riêng cho R62. Chi tiết ở `05_tien_kiem_va_khoi_phuc.md`. Chưa sửa boot/mount/cài gói hoặc thuật toán.
