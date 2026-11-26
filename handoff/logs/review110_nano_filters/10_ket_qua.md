# IDSW của bốn bộ lọc trên 80 đoạn với detector Nano mới

Cùng detector YOLO11n 50/50 của R109 trên 40 đoạn BDD100K và 40 đoạn UA-DETRAC, mỗi đoạn 150 khung. Lưu detection một lần trên Nano và phát lại trên PC NumPy qua KF, EKF-CTRV, UKF-CTRV và PF500 K1/K2 motion2. Các lượt dùng cùng ngưỡng, lọc và ánh xạ lớp, cách tính bước thời gian và buffer 1,2 giây. PF dùng ba seed 0, 1, 2. [Giao thức](00_protocol.md)

## BDD100K

- KF đạt **1.534 IDSW**, HOTA **43,72%**, FN **28.539**.
- EKF đạt **1.354 IDSW**, HOTA **43,87%**, FN **28.662**. IDSW giảm **180 lần (11,73%)** so với KF, giảm trên 28/40 đoạn và tăng trên 7/40.
- UKF đạt **1.426 IDSW**, HOTA **43,74%**, FN **28.612**. IDSW giảm **108 lần (7,04%)** so với KF, giảm trên 22/40 đoạn và tăng trên 13/40.
- PF500 đạt trung bình **1.569,67 IDSW**, HOTA **42,81%**, FN **29.147,67**. Ba seed có IDSW **1.594, 1.562 và 1.553**.

## UA-DETRAC

- KF đạt **245 IDSW**, HOTA **66,57%**, FN **17.590**.
- EKF đạt **250 IDSW**, HOTA **66,59%**, FN **17.593**.
- UKF đạt **248 IDSW**, HOTA **66,61%**, FN **17.591**.
- PF500 đạt trung bình **276,33 IDSW**, HOTA **65,94%**, FN **18.193**. Ba seed có IDSW **287, 275 và 267**.

## Kiểm chứng

Hoàn tất **480 lượt tracking, 72.000 khung phát lại**. Từng đoạn kiểm SHA256 ảnh và engine, MOT và lớp của lượt lưu cache trùng từng byte với R109 ở 80/80 đoạn. KF phát lại trên PC khớp hộp trong 0,001 pixel, frame/ID/lớp/điểm chính xác và toàn bộ chỉ số theo video trùng lượt R109. Tổng IDSW, FN và FP theo video khớp điểm gộp ở cả sáu lượt/tập. Điểm FPS Nano R109 đo với KF. [Kết quả TrackEval theo video và seed](replay_full/scores.json)
