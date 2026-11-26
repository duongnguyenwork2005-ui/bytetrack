# R130 — noise vận tốc theo hướng chuyển động, khóa trước điểm

User yêu cầu PF ít IDSW hơn KF/EKF/UKF trên UA và BDD. R128 chưa dướiEKFBDD,
R129 initializer mềm cóUAIDSW465,67 nhưngFN37529,33 tăng0,267% soPF100, trượt
cổng0,2%; không mởBDD, không nới cổng hoặc dò gain sau điểm.

Thử một cơ chế khác trên native PF100 R126, giữ hard init: sửa hình học process
noise vận tốc. Q Cartesian R118 thêm sigma_v² vào cảvx/vy, trace2*sigma_v²;
trong mô hình polar, noise tốc độ là một chiều theo hướng xe, còn noise hướng
là thành phần khác. Thử rank-adaptive speed noise cộng heading diffusion.

Sau deterministic coordinated-turn và Qpos nhưcũ, dùng
M=Pvv+u*u^T; D=M/tr(M), nếuM=0 dùngI/2; thêmQspeed=sigma_v²*D vàoPvv.
D là xấp xỉ ratio-moments cho E[direction*direction^T], **không phải tích phân
chính xác hướng của Gaussian**. Nó PSD, trace1, thànhrank1 khi vận tốc biết rõ,
thànhI/2 khi bất định đẳng hướng quanh0. Tích phân noise góc Gaussian bằng
moment nhưR128, dùng nguyênsigma_theta hiện hành, không dò hệ số.

Giữ Qheight-velocity gốc (không đổi shape), Qposition/R/init/100hạt omega IID/
resampling/association. Không thaykhốiQomega. Đây là thay đổi mô hình/process
approximation, không gọi càiđặt tươngđươngR126 hay sửa số liệu cũ.

Numerical: PSD/trace/giới hạn biết rõ hướng/đẳng hướng/zero-noise native parity,
Gauss-Hermite kiểm moment heading đã có ởR128; smoke Native100 soR123 vàrepeat.
UAtrain dev12×3seed. CổngUA meanIDSW<472,>=2seed<472,HOTA>=R123-0,001,
FN<=1,002*R123. Đạt mớiBDD200×3seed/mot_crowd cùngR125, đíchmeanIDSW<5548,
>=2seed<5548,HOTA>=R125-0,001,FN<=1,002*R125. Không dò tensor/scale sau điểm.
Nếuđạt cảhai, mởxácnhậnUA test40 vớiđối chứngmới vàPFbaseline nhưkếhoạch.
Báođúngvai tròdev/testvàselectionbias, khôngchọnsubset/seed. Không sửadefault,
src/thesis/detector/metric. Chưa tuyênbốFPSNano.
