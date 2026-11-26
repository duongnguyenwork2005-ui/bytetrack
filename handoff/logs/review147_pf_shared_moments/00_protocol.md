# R147 — bảo toàn moment sauresample khi mọi bộ lọc cùngconfidence-R

R146 đã đủ3seedUAdev:449/441/440, mean443,33>439, chỉ0seed<=439;
HOTA61,2324%, FN35078,33, không chọn và không mở BDD/test40/kịch bản.
Không dò ordering/seed/N/ESS để cứu. Giữ R144commonR controls449/439/439.

R147 xuất phát lại từ R142 Native100/Rprediction/confidenceR/systematic,
không ghép orderedR146. Thêm joint-moment correction đã kiểm ởR135 sau
systematic resampling: khớp mean/cov5D(omega,cx,cy,vx,vy), gồm within-Gaussian
covariance. Giữ shape Gaussian chung, không thêmRNG và không đổiQ/R/N100/
ESS0,5/init/association/metric. Mục đích giảmdrift do lựa chọn hạt hữu hạn,
đặc biệt khiR(c) nhỏ làmweights tập trung; đây là giả thuyết, không bảođảm
giảmIDSW hoặc giữ higher moments/multimodality. Không phải ETPF chuẩn.

Sử dụng nguyên moment_resample.py R135 vớiscore hiện tại đượcR142forward.
Kiểm nguồnR135 hash qua finalproof/inputs. Numerical200mixture jointmoment,
720scored-control steps khi tắt correction qua nhiềuresample/PSD/RNG và
enabledfinite/PSD. Smoke helperR146 chỉ dùng làmharness: controls commonR
vàPFparentR142parity, candidate moment mới lặp3seed/2video; không dùng
resamplingR146 trong candidate này.

UAdev12×3seed: PFmean<=439, ít nhất2seed<=439, HOTA>=R142-.001,
FN<=1.002R142. Đạt mớiBDD200/commonR3controls+PF3seed, rồiUA test40;
chỉ đạt cảhai full với EKF<KF,UKF<KF,PF<=min(EKF,UKF) vàguardrails mới
kịch bản. Reuse controlsdev36R144 sauSHA; không cherry-pick clip/seed.
Không dò tolerance/strength/tuneQ/R sauđiểm. Tốiđa3job gồmtrainingGPU.
Không sửa src/default/Nano/khóa luận. Các tập đã xem nhiềuvòng, không
gọi test mù hoặc bảo đảmưu thếphổquát. Có thể tốn thêmCPU, timingchỉđo
sau quality khi không có jobtraining/quality khác.

Nguồn hướngmoment correction: [Acevedo etal.](https://arxiv.org/abs/1608.08179).
Transform mixtureGaussian là triểnkhai riêngR135 đã kiểm, không nhận là
toàn thuật toán paper. Code/phép thử cũ giữ nguyên.
