# Giữ symlink khi sao chép gói thử nghiệm

Lần stage đầu dùng shutil.copytree mặc định, làm các symlink configs/data/
models/third_party được sao chép thành nội dung. Kiểm kê không khớp nên
dừng trước mọi phép đo FPS. Không đổi nguồn hay dữ liệu đối chứng.

Giữ log package_stage.log và bản sao đầu trong package_stage_v1_failed.
stage_pipeline_v2.py giữ symlink, quy về cùng referent tuyệt đối và kiểm
toàn bộ entry của gói nhỏ. Chỉ nano_detect_stream.py đổi; thêm gpu_pre.py,
gpu_detector.py và fused_pre.so. nano_detector.py, nguồn tracker, NMS và
các referent dữ liệu/config/model/vendor giữ nguyên. Lần sửa này chỉ
sửa công cụ stage; kernel/oracle/capture đang chạy không thay đổi.
