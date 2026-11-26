# R153 — proposal góc quay có quan sát, giữ mô hình thời gian R142

Thiết kế trước điểm, ngày20/11. R152 đang kiểm N500; không chọn seed/N
từ điểm đó. Ứngviên này giữ N100, Q/R/init/ESS.5/systematic/mean/shape,
association/lifetime/cache R142. Không trộn ordered/moment/robustR.
Không thay EKF/UKF/KF hoặc mở kịchbản khi prerequisiteUA chưa đạt.

Nguồn phương pháp chung: [Doucet, Godsill, Andrieu,2000](https://www.stats.ox.ac.uk/~doucet/doucet_godsill_andrieu_sequentialmontecarloforbayesfiltering.pdf),
phần local-linearisation proposals và Rao–Blackwellisation. Thiết kế
dướiđây là áp dụng riêng cho kernel hiện tại; không gọi tác giả đã thử
trên ByteTrack hoặc bảođảm giảmIDSW.

## Mốc thời gian cần giữ đúng

KernelR142: body[t]=T(omega[t-1])*body[t-1]+noise; sau đó mới sinh
omega[t]=omega[t-1]+eta[t]. Vì vậy measurement[t] không hướng dẫn
eta[t] mới, nhưng có thể hướng dẫn eta[t-1] đang đi vào T. Không đổi
thànhomega-first rồi quy mọi cải thiện cho proposal.

Trong mỗi particle giữ incomingcentre và incomingvariance củaomega:
ởinit meanomega/covomega; saupredict là omega trước cộngeta vàsigma²Q.
Conditional body sauupdate độc lập với eta mới trong cùng hạt/histories.
Resampling phải copy incomingcentre theo đúng ancestry, giữvariance.

Predict vẫn lấy mẫu prior nhưcũ để có bboxchoassociation, lưu prebody
mean/cov, incomingcentre/variance, Qheight, lost và eta[t] mới. Update
chỉsau khiByteTrack ghép measurement thật, không dùngGT hoặc future.
Chỉ proposal cuối cùng được thay; không sửa phép ghép cặp hiện tại.

## Proposal và hiệu chỉnh

Tại incomingcentre mu, dùng m=T(mu)m_prev vàP=T(mu)P_prevT'+Q.
J=d(T(omega)m_prev)/domega. Linearisation chỉ để lập q:
S=H(P+J*s²*J')H'+R, gain=s²*J'H'S^-1,
qmean=mu+gain*(z-Hm), qvar=s²-gain*S*gain'.

Rút omega_prev từN(qmean,qvar), tính lại bodyprior bằng T thực và Q
gốc từ snapshot, rồi Kalmanupdate tuyến tính nhưNativeC. Cộnglogweight:
loglikelihood thực + logNormal(omega_prev;mu,s²)
                     - logNormal(omega_prev;qmean,qvar).
Covariance phụthuộcomega được tính đúng khi tínhlikelihood thực; đạo
hàm của covariance bỏ qua trong q không cho phép bỏ tỉsố prior/q.
omega[t]=omega_prev+eta[t] đãrút độc lập, incomingcentre=omega_prev.
Shape đãpredict độc lậpomega được giữ nguyên, khôngpredict hai lần.

Zero incomingvariance dùngpriorDirac, không thêm draw/jitter. Không có
predict cache hoặc motioninitialization chen giữa predict/update thì
khôngguide; invalidatesnapshot sauinit thànhcông vàsau mỗiupdate.
Cần giữ đúng trườnghợp unconfirmed update khôngpredict, lost nhiều
bước, reactivation, confidenceNone/0/1 và resampleancestry. Nativekernel
failure phải fail-loud, không âmthầm bỏprior/q hoặc resetID.

Đây không là posteriorchínhxác: Q/R plug-in height và motioninit vẫn
theoR142; q làGaussianxấpxỉ, có thể thiếu multimode. Importancecorrection
đúng cho conditionalstep không bảođảm ítIDSW hoặc nhanh hơn.

## Kiểm trướctracking và cổng

1. Guideoff khớp parentR142 cả arrays/state/RNG/PSD qua predict/update,
   gồm Lost/no-predict/motioninit/reinit; q không tiêu thụdraw ở off.
2. Jacobian sai phân/ODE; oracle linearGaussian cho qmean/qvar và
   incrementalweights hằng đúng evidence, không chỉ test code tựsoi.
3. Quadrature1D CT độc lập: posterior mean/marginal likelihood và
   MonteCarlo với tỉsốp/q, PSD/ranges/normalisation/ancestry/counters.
4. Smoke offparentGaussian/PFprefix, guideonrepeat3seed/2video; khóa
   source/cache/GT/config trước điểm, kiểm raw/MOT/CLS/timing/marker.

Dev12×3seed=36lượt trước. meanIDSW<=439, ít nhất2seed<=439, HOTA không
giảm>.001 vàFN không tăng>.2% soPF100R142. Trượt thì đóng nhánh,
không dò varianceinflation/strength/N/seed/ESS để cứu. Đạt chỉ giữ thành
phần; full/kịchbản vẫn cần GaussianfullE/U<KF của người dùng. Đo thời
gian tuần tự chỉ sau quality và khi không có train/qualityjob khác.
Không sửa src/default/Nano/khóa luận. Kếtquả khámphá, không blindtest.
