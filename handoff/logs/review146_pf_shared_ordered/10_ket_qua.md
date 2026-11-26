# R146 — ordered stratified resampling với confidence-R chung

Đủ36lượt UA train dev12×3seed, giữ nguyên PF R142 và chỉ đổi systematic selection sang stratified selection sau sortomega. Mọi filter dùng cùng công thức R(c), KF/EKF/UKF reuse36lượt R144 đã xác minh hash.

PF từngseed:449/441/440 IDSW; trung bình443,33. KF449, EKF439, UKF439. HOTA PF61,2324%, FN35.078,33, FP9.960,67. HOTA/FN giữ được nhưng meanPF>439 và không seed nào<=439, nên không đạt cổng. Không mở BDD, UA test40 hoặc dataset kịch bản, không chỉnh ESS/N/seed để cứu.

Numerical600bước scored-control parity qua resample/PSD/RNG,100CDFcases, source R140 qua manifest chain đạt. Smoke6parentPFparity/6candidate repeat và6controller offcontrols/6repeat đạt. Marker/MOT/classes/timing/raw/sourcehash36lượt đạt. Log numerical đầu dừng do truy cập sai tầng manifest; đã sửa đọc qua inputs, giữ log lỗi và chạy lại trước preflight, không ảnh hưởng dữ liệu tracking.

Các tập đã được quan sát nhiều vòng. Chưa đo tốc độ, không đổi src/default/Nano/khóa luận. PLAN mục0 cập nhật trước báo cáo. R147 thử một cơ chế khác là jointmomentcorrection, không trộn ordering của nhánh này.
