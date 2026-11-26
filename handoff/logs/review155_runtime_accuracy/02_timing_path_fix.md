# Sửa đường dẫn nguồn của runner, giữ nguyên giao thức đo

Sau Nano đổi boot từ46e76ac2... sang51280481..., swap0MB, lượt pilot đầu
dừng ở FileNotFoundError cho SOURCE/drivers/r86_driver.py, trước khi ghi
inputs hoặc chạy ảnh. Bản R105 trên Nano đặt drivers ở thư mục gốc; import
đã dùng đúng module gốc nhưng fingerprint dựng nhầm đường dẫn thư mục.

Giữ log pilot.log và thư mục runs/pilot trống. Version mới fingerprint
đúng __file__ của bench_nano/r86_driver/r101_driver/r102_driver thực sự
đã import. Không tạo một bản drivers giả để fingerprint nhầm code. Giữ
toàn bộ mô hình/AP/cổng/ảnh/package/timing/counters/parity/selectionrule.

run_models_v2.py/timing_v2.py và timing_models_v2.json khóa trước chạy
pilot mới ở runs_v2/pilot, rồi full cùng parent. Không sửa nguồn/config
cũ đã dùng trong GPUtiming, không hồi tố các log cũ. Đây là sửa harness,
không tối ưu tốc độ hoặc thay thuật toán tracking.
