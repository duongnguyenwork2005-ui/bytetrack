# Đường chạy Nano cho lượt tự động

CLI coder trên Windows không đọc được thư mục SSH riêng. Phiên chính đã xác minh alias `nano` bằng host key đã ghim và sẽ chạy phần SSH/SCP của R105, dùng đúng ba script đã chuẩn bị trong thư mục này. Không sao chép khóa hoặc đổi host key.

Đường dẫn bằng chứng là `D:/LABOTARY/automation/runs_local/r105_107_20261112_runtime/scratch/r105_nano_host/`. Kiểm `bridge_status.json` và `smoke.log` hoặc `full.log`. Smoke đã hoàn tất 8/8 và Nano đang chạy 30 lượt. Phiên chính sẽ tải `full/` tự động sau khi benchmark kết thúc. Coder lấy bằng chứng này để chấm, kiểm số lượt và SHA256 rồi tiếp tục review.

Runner trên Nano được chép trước khi coder bổ sung hai trường `package_sha256` và `driver_manifest_sha256`. Bằng chứng gốc ghi SHA của các file đã chạy ở `inputs.json` → `package_sources` và `code_sha256`, đồng thời `complete.json` ghi SHA archive và code. Khi chấm, xác minh riêng schema này cùng hash driver, engine, ONNX, ảnh và tệp kết quả. Giữ các JSON gốc để reviewer kiểm trực tiếp. Nếu cần một lệnh Nano bổ sung, ghi yêu cầu và lý do vào log R105 để phiên chính xử lý.
