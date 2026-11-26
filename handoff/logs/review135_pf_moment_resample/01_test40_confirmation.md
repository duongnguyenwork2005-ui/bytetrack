# R135 — xác nhận trên UA test40 nếu qua UAdev và BDD

Toàn bộ40 video/56.340 frame, cùng cache COCOc25/classes/GT/evaluator ordinaryMOT
với R132. Nền ứng viên là R132Native100/Rprediction; thay resampling bằng
systematic + joint moment correction, không dùng mixture hoặc quantileinit.
Chạy120 lượt mới (ba seed0/1/2); dùng lại240 lượt controls KF/EKF/UKF/Native100
và120 lượt parent R132 đã khóa hash. Smoke6 parent controls/6 repeat trước full.

Đích meanIDSW<2498 và ít nhất2seed<2498; HOTA>=Native100-0,001,
FN<=1,002*Native100. Không đổi threshold/N/numeric tolerance/seed sau điểm.
Tái hợp raw, kiểm hash từng MOT/classes/timing/marker, paired video bootstrap
5000 với seed133 như helper R133, không đổi bootstrap để chọn CI thuận lợi.
Helper R133 chỉ cung cấp I/O/evaluator/reference logic; factory thực là R135.
Các tập đã quan sát, không gọi test mù; báo cả FN/HOTA/IDF1/FP/CI.
