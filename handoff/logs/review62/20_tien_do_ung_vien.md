# R62 — tiến độ trước khi tiếp tục phép đo GPU

Ngày 28/10/2026, giờ Việt Nam. Trạng thái chuẩn: PLAN mục 0.

## Đã hoàn tất

- Truy cập Nano đổi từ `10.11.61.127` sang `192.168.1.4`, dùng lại cùng khóa SSH và SD. Engine nguyên bản `c32690cd…` còn nguyên; đã sao lưu về `handoff/archive/nano_engine_r62_c32690cd.engine`, hash khớp. Gói v16 manifest đạt 562/562; bản sao dành cho phép đo cũng đạt 562/562.
- Baseline ngắn **8/8 lượt × 300 ảnh** hoàn tất, tracking đủ frame, live/replay MOT+lớp trùng byte, detection giống nhau cả 8 lượt. Số từng lượt ở `10_baseline300_tong_hop.json`, kiểm detection ở `12_kiem_detection_baseline300.json`. CV bật 8,593/8,803 FPS, CV tắt 8,959/8,943; EKF bật 8,972/9,192, EKF tắt 9,408/9,400. Decode hai mức 11,4/16,5 ms: không xếp hạng CV/EKF từ số này. Lượt đầu bỏ 1 frame xem trước lúc mở cửa sổ, tracking vẫn đủ; ba lượt bật khác bỏ 0.
- Raw logs/CSV/replay nhỏ được sao lưu trong `baseline300_evidence.tar.gz`, SHA256 `2d78eb0e7bace8b78acc9ee6e096ec4222a1d7df5f74f70052a73d03d571f922`, khớp PC/Nano. Cache raw giữ trên Nano. **1.470/1.470 ảnh** khớp hash gốc (`13_kiem_1470.log`).
- Thử các cách chuyển tensor bằng ba script `so_tien_xu_ly*.py`, giữ nguyên mọi kết quả (`11_`, `14_`, `16_`). Không chạy phép đo CPU cùng lúc với benchmark pipeline. Micro v1 kết thúc trước khi đóng gói/sao lưu raw, theo timestamp file (kết thúc 21:12:14, archive hoàn tất 21:12:32).
- Chọn đúng một thay đổi `preprocess`: uint8 CHW → bảng 256 float32 giá trị đúng phép chia cũ → `cv2.LUT` trên cột một kênh → NCHW. Micro v3: legacy trung vị 9,778 ms, ứng viên 2,396 ms; 11 trường hợp trùng byte. Chưa suy ra mức cải thiện toàn pipeline.
- Mã ứng viên trong `src/nano_detector.py`; `src/test_nano_detector.py` bổ sung toàn miền uint8, layout khác, dtype fallback, ảnh rỗng và quyền sở hữu output. PC **7/7 PASS** (`18_test_nano_detector_pc_lan02.log`), cú pháp Python 3.6 đạt. Lần test đầu lỗi ở oracle `torch.flip(uint16)` chưa hỗ trợ; đã đổi oracle sang float trước flip, giữ log thất bại `18_test_nano_detector_pc.log`.
- Kiểm mã production trên CPU Nano: **11 trường hợp đạt**, NumPy 1.19.5/OpenCV 3.2.0 (`19_test_production_tensor_nano.json`); SHA256 file được chuyển `6a1165f06ad5f36e48c95a43a1e1c96022c0355058c9669830055dee0e057afe`. File tạm `~/nano_run/r62_20261028/nano_detector_candidate.py`; chưa thay file trong v16, chưa dựng gói v17 triển khai.
- Rà mã độc lập trong cùng chat không phát hiện lỗi cần sửa; đây là review tĩnh, không thay phép đo GPU thật.

## Lượt bị gián đoạn và việc còn lại

Lượt `~/nano_run/r62_20261028/baseline1470/01_cv_on` đã tạo ba file `before.json`, `command.json`, `live.log`, nhưng sau reboot chỉ còn file 0 byte, không có progress/run_meta. Không ghi nhận frame thành công, không tính là lượt đạt và không ghi đè thư mục này. `15_gian_doan_baseline_dai.json`: boot mới lúc 21:51 Việt Nam, reset source báo software reset, ext4 phục hồi 14 orphan inode, `errors_count=0`; không có persistent journal của boot trước. Các dữ kiện này không xác định ai/nguyên nhân đã reset.

Đã hỏi người dùng có chủ động restart hay máy tự treo/tắt. Chờ câu trả lời trước khi tiếp tục GPU dài. Không có benchmark GPU đang chạy ở thời điểm ghi hồ sơ.

Sau khi xác nhận máy ổn định:

1. Tạo thư mục **lần thử mới** cho baseline dài và hoàn tất CV→EKF→EKF→CV 1.470 ảnh, cùng điều kiện hiện tại.
2. Đóng gói ứng viên từ v16, chỉ đổi file preprocess và provenance/manifest cần thiết; kiểm trên Nano.
3. Chạy so sánh baseline↔ứng viên cân bằng, 4 cặp cho CV và 4 cặp cho EKF, bật hiển thị; so detection/MOT/lớp, p90, RAM/swap, nhiệt/xung. Tiêu chí giữ ở kế hoạch 01.
4. Nếu đạt, chạy dài ứng viên và xác minh launcher trước chuyển bản dùng; nếu không đạt, giữ v16 và báo kết quả âm. Hiện **chưa nghiệm thu mức tăng tốc hoặc mốc 6**.
