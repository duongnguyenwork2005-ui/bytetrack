# R135 — sửa sai moment do tái lấy mẫu của Cartesian RBPF

Khóa trước điểm R135, ngày19/11/2026. Mục tiêu vẫn PF ítIDSW hơn cảKF/EKF/UKF
trên BDDval200 và UA test40. R134 thắngBDD nhưng UA2515,33>KF2498; không dò
grid/sigma/seed của R134. R135 dùng lại nền R132 Native100/Rprediction, initIID,
không dùng Gaussian-mixture R133 hoặc quantile-init R134.

## Cơ sở và giới hạn

Systematic resampling giữ phân phối theo kỳ vọng, nhưng mean/covariance của một
ensemble hữu hạn có thể lệch sau selection. Tham khảo hướng khớp hai moment ở
[Acevedo/de Wiljes/Reich,2017](https://arxiv.org/abs/1608.08179), đặc biệt cả cảnh
báo second-order correction có thể tạo mẫu không nhất quán với posterior.
R135 là **suy diễn riêng cho mixture Gaussian điều kiện**, không phải tái hiện
ETPF, NETF, optimal transport hay thuật toán nguyên bản của bài báo.

Thử sửa hai moment joint(omega,x) với x=[cx,cy,vx,vy] sau systematic selection.
Giữ đúng cả covariance trong Gaussian điều kiện và covariance giữa các hạt.
Không chỉ recenter omega rồi bỏ qua phụ thuộc với body. Đây là biến đổi xấp xỉ
posterior hữu hạn; giữ hai moment không bảo đảm giữ multimodality/higher moments
hay giảmIDSW. Không tự nhận ưu thế thống kê hoặc tính nhất quán hữu hạn.

## Một thay đổi, không có tham số dò

Tính trước-resample mean(mu_w,mu_x), variance v_w, cross c và Sigma_x từ
weighted mixture, gồm sum_i w_i P_i. Sau selection bằng đúng systematic RNG
như nền, tính các moment tương ứng với uniform weights, ký hiệu dấu s.

- omega' = mu_w + sqrt(v_w/v_ws)*(omega_s-mu_ws).
- beta=c/v_w, beta_s=c_s/v_ws.
- S=Sigma_x-c*c^T/v_w và S_s tương tự (Schur complement).
- T=chol(S)*inverse(chol(S_s)).
- m'_i=mu_x+beta*(omega'_i-mu_w)
  +T*(m_si-mu_xs-beta_s*(omega_si-mu_ws)).
- P'_i=T*P_si*T^T.

Như vậy ensemble Gaussian-mixture mới khớp đủ mean5D/cov5D của weighted
ensemble trước selection trong sai số số học, khi các ma trận không suy biến.
Body vẫn Gaussian điều kiện, omega vẫn là một giá trị mỗi hạt. Shape Gaussian
chung không đổi. Kết quả update hiện tại được tính trước resample như nền;
chỉ cloud cho bước tương lai được sửa. Uniform logweights/counters/cache giữ
đúng giao thức. Không thêm random draw, chỉ cùng một systematic uniform.
Nếu variance omega suy biến theo tolerance64eps hoặc Cholesky/finite guard
không đạt, dùng nguyên selection thông thường; không bơm covariance/jitter.
Không đổi ESS0,5, N100, Q/R/init, detector, lớp, FPS/buffer/association/metric.

## Kiểm và cổng

Kiểm độc lập joint-moment trên nhiều Gaussian-mixture ngẫu nhiên; PSD/input
ownership, trường hợp degenerate trả selection, no extra RNG. Tắt correction
phải khớp R132 đầy đủ qua nhiều bước có resample. Smoke parent6/repeat6 trước
UA train dev12×3seed0/1/2. MeanIDSW<472/ít nhất2seed<472,
HOTA>=R123PF100-0,001,FN<=1,002*R123. Đạt mới BDD200×3seed/mot_crowd:
meanIDSW<5548/2seed<5548/guardrails soR125PF100. Đạt mới UA test40×3seed,
meanIDSW<2498/2seed<2498/guardrails soNative100R132. Cùng refs đã khóa hash.
Nếu trượt, không nới cổng/đổi epsilon/cov transform/seed/N để cứu kết quả.

Các tập đã được quan sát nhiều lần; báo selection bias, CI bootstrap video5000,
FN/FP/HOTA/IDF1. Không đổi src/mặc định/Nano/khóa luận. Chỉ đo PC tuần tự nếu
qua chất lượng hai tập, không gọi tracker.update là FPS Nano.
