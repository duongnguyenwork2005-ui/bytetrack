# R165: chưa đạt mục tiêu IDSW

Một bước resample–move sau tái lấy mẫu của guided PF R153, vẫn100hạt,
Q/R/khởi tạo/ghép cặp như parent. Đủ36lượt trên12video UA dev, ba seed.
IDSW seed mới446/431/433, parent439/435/439; mean437,67→436,67, giảm1ID.
EKF435 vàUKF430 là đối chứng mạnh đã khóa, KF449. Khôngseed nào<=430,
nên cả mean/two-seeds gate đều trượt; không đổi mặc định/chạy full/transfer.

HOTA61,22110→61,22776%, FN35071→35068,67; hai guard này đạt. Bootstrap
ghép12video, mean ba seed,5000draw seed165: CI95 chênhIDSW so parent
[-3,3333;+1,3333]. Khoảng chứa0, trên dev đã xem nhiều vòng và chưa hiệu
chỉnh selectionbias/đa kiểm định, nên không khẳng định cải thiện chắc chắn.

Numerical và smoke đều đạt: tắt move khớp parent/RNG, exact Gaussian
likelihood soSciPy, detailedbalance, stationary CT, ancestry/eta/shape,
nativeposterior soexpm/Joseph kể cả accepted/rejected. Tracking có916600
đề xuất và916182accept. Các con số này xác nhận nhánh đã chạy, không chứng
minh nguyên nhân thay đổi IDSW. Chưa đo tốc độ hợp lệ vì PC đang train YOLO.

`final_verification.json` tái kiểm source/raw/36MOT/CLS/timing/marker,
giữ quality_gate_pass=false. Thuật toán tham khảo resample–move của
[Gilks & Berzuini](https://doi.org/10.1111/1467-9868.00280), triển khai
riêng cho conditionalincomingomega R153; không phải mã của tác giả.
