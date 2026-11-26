# R165 — một bước resample–move cho incoming angular innovation

Người dùng21/11 cho phép thử thêm cải thiện PF; thực hiện CPU trong lúc
R164 phục hồi detector trên GPU PC. Không đo timing PC đồng thời training,
tối đa3 logical Python job, không sửa nguồn/default/khóa luận/detection.

Parent R153 guided N100, cùng Q/R/init/ESS .5/systematic/association và
chính sách Lost. Chỉ khi có cache predict hợp lệ, incoming variance>0 và
parent vừa resample: một independent Metropolis–Hastings move cho từng
incoming omega, q Gaussian của chính proposal R153, cùng ancestor đã chọn.
Không tăng N, không dò số bước/scale/ESS/seed sau khi xem điểm.

Target có điều kiện ancestor là priorNormal(omega;mu,s2) nhân likelihood
Gaussian vị trí sau T(omega) thật, gồm covariance phụ thuộc omega. Acceptance
logratio = [LL(new)+logprior/q(new)] − [LL(old)+logprior/q(old)]. q cố định
theo ancestor và measurement hiện tại. Slice đúng prebody mean/cov, mu,
eta mới theo last_selected; giữ eta mới độc lập. Accepted move tính lại
Kalman bodyposterior, omega[t]=new+eta; shape đã update không update lần2.
Trả estimate parent trước resample/move, như cách parent trả trước resample;
chỉ particle population cho predict kế tiếp được rejuvenate. Không dùng GT,
tương lai hoặc đổi likelihood để ép giữ ID. Không gọi là fixed-lag smoothing.

Nguồn khái niệm: Gilks & Berzuini, Following a Moving Target, JRSS B63,
127–146,2001, https://doi.org/10.1111/1467-9868.00280;
https://www.stats.ox.ac.uk/~doucet/smc_resources.html (MCMC invariant moves).
Đây là áp dụng riêng cho R153, không phải mã tác giả hay bảo đảm giảm IDSW.

Trước điểm: off bitparity parent arrays/RNG gồm Lost/no-predict/motioninit;
2x2 likelihood so SciPy; CT oracle bằng expm; detailed balance independent
MH và linear-Gaussian q optimal acceptance1; actual resample ancestry/shape/
PSD và native posterior; zero variance/no-resample parity, installed restore.
Smoke offprefix và onrepeat 2video×3seed. Khóa source/cache/GT trước dev36.

Cổng dùng reference mạnh R154 EKF435/UKF430, KF449; PFmean<=430 và ít nhất
2seed<=430; HOTA>=R153-.001,FN<=R153*1.002. Trượt đóng nhánh, không tune
để cứu. Đạt chỉ giữ component; full BDD/UA/PF/scenario vẫn cần Gaussian
fullE/U<KF theo điều kiện người dùng. Dev đã xem nhiều vòng, không blindtest.
