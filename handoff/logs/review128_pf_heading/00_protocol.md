# R128 — đưa nhiễu hướng vào Cartesian PF bằng moment matching

Khóa trước điểm ứng viên, 19/11/2026. Người dùng yêu cầu tiếp tục cho đến khi PF
có IDSW thấp hơn KF/EKF/UKF trên cả UA-DETRAC và BDD. Không bảo đảm trước kết quả;
giữ mọi lượt trượt, không chọn video hoặc seed có lợi, không đổi metric/detector.

## Giả thuyết

R118/R126 đã đổi mô hình nhiễu sang Cartesian đẳng hướng. `_std_theta` của CTRV
không được truyền vào Gaussian vận tốc ở bản này. Đây là giới hạn mô hình đã
chọn trước, không phải tuyên bố các kết quả cũ bị tính sai. Nhiễu hướng trực tiếp
có thể giúp thích nghi khi hướng thay đổi mà omega chưa học kịp.

Thử đúng một ứng viên: sau coordinated-turn predict, cho vận tốc quay thêm
delta ~ N(0, sigma_theta^2), dùng nguyên `_std_theta` hiện có, không dò hệ số.
Giữ 100 hạt omega IID, Q Cartesian/R/init/resampling/association như R126.
Tích phân delta bằng moment chính xác rồi xấp xỉ kết quả bằng Gaussian trong
mỗi hạt. Đây là thay đổi mô hình/process approximation, không còn nói toàn bộ
posterior trong mỗi hạt chính xác Gaussian hay chỉ tối ưu cài đặt như R126.

Với u=E[v], M=Cov(v)+u*u^T, a=exp(-s²/2), b=exp(-2s²):
E[v']=a*u; E[v'v'^T]=b*M+(1-b)*tr(M)/2*I;
Cov(v')=E[v'v'^T]-a²*u*u^T; Cov(x,v')=a*Cov(x,v).
Không tạo RNG draw mới hoặc tăng số hạt. Q isotropic được giữ.

## Kiểm và cổng

Numerical: đối chiếu moment với Gauss-Hermite tích phân góc, zero-noise đúng R126,
PSD và cross-covariance; smoke native IID khớp R123, ứng viên lặp seed.
UA train dev12, cùng cache R123, seed0/1/2, đủ36lượt. Cổng mở BDD:
IDSW trung bình <472 (min KF489/EKF472/UKF474), ít nhất2/3seed cũng <472;
HOTA>=R123-0,001 và FN<=1,002*R123. Không được giảm IDSW bằng mất nhiều track.

Nếu đạt UA, giữ source nguyên vẹn chạy đủ BDD200×3seed, cùng R125/mot_crowd.
Mục tiêu BDD: IDSW trung bình <5548 (min6345/5548/5655), ít nhất2/3seed <5548;
HOTA>=R125-0,001 và FN<=1,002*R125. Báo cả HOTA/IDF1/FN/FP và CI ghép video;
đạt điểm trung bình không tự chứng minh khác biệt có ý nghĩa thống kê.
Nếu trượt, không chọn; chỉ thử hướng cơ chế khác bằng giao thức mới, không dò
sigma_theta trên BDD hoặc UA test. UA ở đây là dev12 từ train, không phải test40.
BDD val đã xem nhiều vòng, dùng để xác nhận thăm dò; không gọi test mù.

Nếu qua cả hai, đo chi phí tuần tự trên cùng mẫu R126; không suy FPS Nano từ PC.
Không sửa mặc định, src, thesis hoặc các kết quả đã khóa.

## Nguồn

Rao–Blackwellisation/marginalization: Doucet et al., UAI2000,
https://arxiv.org/abs/1301.3853. Tổng quan thực hành PF:
Elfring et al., Sensors2021, https://www.mdpi.com/1424-8220/21/2/438.
Công thức moment góc ở trên được suy ra từ characteristic function của Gaussian
và kiểm tích phân độc lập; hai nguồn không bảo đảm ứng viên này thắng tracking.
