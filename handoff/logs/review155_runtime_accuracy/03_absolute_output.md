# Đường dẫn đầu ra tuyệt đối cho subprocess

Lượt runs_v2/pilot truyền output tương đối; r102_driver chạy subprocess
với cwd=package, nên child tạo output trong package thay vì thư mục của
runner. Giữ cả hai thư mục và pilot_v2.log làm bằng chứng lỗi harness.
Không lấy FPS/quality từ lượt chưa được runner xác minh này.

Gọi nguyên timing_v2.py với output **tuyệt đối** trong runs_absolute/pilot,
không thay thuật toán/config/guard/metrics/selectionrule hoặc package.
Full sử dụng runs_absolute/full cùng parent để kiểm pilot provenance.
Không sửa files v2 đã fingerprint; fix nằm ở đối số gọi script.
