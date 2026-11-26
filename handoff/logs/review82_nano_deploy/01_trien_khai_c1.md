# Triển khai C1 lên Jetson Nano — 04/11/2026

Theo yêu cầu người dùng, dùng SSH alias `nano` để chuyển launcher HDMI từ v16 sang ứng viên C1 của R63. Giữ nguyên thư mục `~/nano_run/v16/nano_transfer` và engine v16.

- Gói: `~/nano_run/r63_20261030/v17c/nano_transfer`.
- Engine: `~/nano_run/r63_20261030/g3_c1_384x640_toc_do_01/engine/yolov8n_384x640_op12_fp16.engine`, SHA256 `a6912393829f8c811d3cc33008f3e4056a626434df327c85aae7beb33af01749`.
- ONNX provenance: `~/nano_run/r63_20261030/ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx`.
- Sửa riêng `configs/nano_ui.json` trong gói C1. Bản trước sửa: `configs/nano_ui.json.pre_deploy_20261104`. `scripts/nano/07_giao_dien.sh --kiem-tra`: `loi=[]`, `canh_bao=[]`, engine/ONNX đúng, Python detector OK.
- Smoke: `~/nano_run/r63_20261030/deploy_c1_smoke_20261104/`. CV, 300 JPG, không hiển thị, đủ frame; live/replay MOT trùng byte; `fps_xu_ly=11.5556393372` trong điều kiện này. Không so trực tiếp số này với kết quả HDMI 13,2 FPS của R63 G3. Sau lượt chạy, ext4 `mmcblk1p1/errors_count=0`.
- Hai launcher desktop và menu cùng trỏ tới gói C1 và có SHA256 giống nhau `f36bd4fe…`. Bản v16 sao lưu cạnh từng launcher với đuôi `.v16_backup_20261104`. Không có tiến trình `nano_live` còn chạy sau smoke.
- Thử mở giao diện qua `DISPLAY=:0` trong 15 giây: log mới ghi đúng gốc C1, không có traceback; lệnh thoát mã 1 sau timeout, nên lượt này chưa xác minh trực quan thao tác chuột/HDMI. Người dùng có thể bấm biểu tượng để kiểm giao diện tại máy.

Quay về v16 nếu cần: chép hai tệp `.v16_backup_20261104` đè lại `nano-tracking.desktop` tương ứng ở Desktop và `~/.local/share/applications/`; cấu hình và engine v16 vẫn nguyên. Vòng này chưa thử hai ứng viên giảm copy/buffer sau C1, chưa có số đo tăng FPS thêm.
