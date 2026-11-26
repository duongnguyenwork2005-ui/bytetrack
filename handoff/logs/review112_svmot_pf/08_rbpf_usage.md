# Chạy RBPF nghiên cứu qua runner chuẩn

Tên tracker là bytetrack_rbpf_ctrv. YAML ở configs/phase2/trackers/r112/rbpf500.yaml, mã ở src/rbpf_ctrv.py. Bản này cải thiện UA-DETRAC train dev12 nhưng không đạt cổng IDSW SVMOT test9. Không thay mặc định luận văn hoặc launcher Nano. Student-t bị loại vì FN dev tăng, không tích hợp vào registry.

Ví dụ PowerShell dùng cache sẵn của dự án

```powershell
& C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name DETRAC-all --tracker-yaml configs/phase2/trackers/r112/rbpf500.yaml --tracker-name r112-rbpf-demo --videos MVI_20052 --fps 25 --elapsed-mode skip --seeds 0 1 2
```

Kết quả tách ba seed trong data/processed/trackers/DETRAC-all/. Khi chạy bộ dữ liệu khác phải dùng đúng cache, lớp, FPS, elapsed và thời gian giữ track theo giao thức của tập đó. Không dùng điểm gộp một video làm kết luận cho cả tập.

RBPF xử lý Gaussian điều kiện, vì vậy không dùng K1 optimal proposal hoặc K2 roughening của PF cũ. Cấu hình không tương thích như pf_estimate=map, pf_optimal_proposal=true, roughening, chuyển động lost trộn CV hoặc association thực nghiệm PF sẽ bị từ chối. Chỉ dùng pf_measurement_height=observation. Chẩn đoán pf_diag cũ được thiết kế cho PF toàn hạt, chưa được triển khai cho các Gaussian điều kiện; xem bộ đếm pf_stats và các phép kiểm R112 thay vì diễn giải bảng diag cũ cho RBPF.

Runner nhận biết tên mới là ngẫu nhiên, tạo seed riêng và ghi fingerprint bao gồm rbpf_ctrv.py. Phép đo R112 dùng plugin cô lập có hash được giữ lại, còn các phép kiểm registered_* đối chiếu triển khai chuẩn với toàn bộ MVI_20052 ở ba seed trên cả backend NumPy và Ultralytics. Các kiểm này xác minh kết quả và đường chạy, không phải đo tốc độ Nano.
