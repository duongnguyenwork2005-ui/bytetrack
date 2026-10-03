# Bằng chứng review `1454b3d`

Đọc `CODEX_REVIEW_1454b3d.md` trước. R39-F1 là finding cần sửa; không có training GPU do reviewer chạy.

- `REPRO_REVIEW_1454b3d.py --repo tree --out repro_local.json`: dùng hai hàm nhật ký thật, trả exit 1 / bug_count 3 trên target. Có một ca đối chứng sạch đạt. Không sửa source hoặc dùng checkpoint.
- `check_receipts.py --tree tree --out receipts_local.json`: tính lại 111 điều kiện từ CSV/JSON/hash, không nạp checkpoint.
- `run_monitor_checks.py --repo tree --out monitor_local.json`: cần psutil; chỉ chọn các bài có ý nghĩa trên POSIX, không chạy ba nhóm đặc thù Windows. Kết quả reviewer 19/19.
- `check_package.py --zip <ZIP delta gốc> --out <thư mục mới>/evidence/package.json`: cần Git và ZIP gốc 1.607.478 byte, không nằm trong gói bằng chứng. Chỉ kiểm 214 đường dẫn thay đổi, không kiểm toàn Git tree. Script sẽ tạo `tree/` và `base_preimages/` bên cạnh `evidence/`; dùng thư mục mới để giữ bản bằng chứng nguyên vẹn.

`tree/` chỉ chứa source và tài liệu/log phục vụ review này, KHÔNG phải toàn repo. `package_metadata/` giữ manifest và khai báo phụ thuộc của delta. `base_preimages/` là 8 file trước sửa đã được so byte với snapshot trước; kết quả và tên snapshot ở `evidence/base_continuity.json`.

Môi trường reviewer: Linux, Python 3.12, psutil 7.2.2. Không có torch/Ultralytics. Lần monitor đầu trong sandbox thất bại do khả năng quan sát PID qua /proc; giữ `monitor_posix.log/json`. Một lần chẩn đoán console cho thấy `psutil.Process()` báo không thấy ngay PID của chính monitor đang chạy. Chạy lại ngoài sandbox với cùng source/test đạt, lưu `monitor_posix_native.log/json`. Không gán các lỗi môi trường đó cho mã dự án.

Lịch sử CPU/GPU dưới `tree/handoff/logs/review39/` là bằng chứng do tác giả cung cấp, giữ byte gốc. Không nhầm với log chạy độc lập của reviewer dưới `evidence/`.

Mỗi file trong gói (trừ chính SHA256SUMS.txt) có hash trong `SHA256SUMS.txt`. Dữ liệu model, ảnh, weights, dataset và thư viện psutil cài thêm không được đóng gói.
