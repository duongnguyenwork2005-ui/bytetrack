# R139 — dự phòng nếu R138 chưa đạt: damped trend của chiều cao khi Lost

Chỉ mở quality nếu R138 kết thúc nhưng chưa đạt mục tiêu hai tập. Chuẩn bị trước
điểm R139; nếuR138 đạt thì dừng, không mở thêm. NềnR132Native100/Rprediction/
initIID/mean/systematic, không ghépR133–R138. Giữdetector/association/lifetime.

## Một thay đổi mô hình extent, không dò hệ số

Thay clamp Lost của Gaussian[h,vh] bằng dự báo vận tốc kích thước giảm dần:
vh(t+dt)=phi*vh(t), h(t+dt)=h(t)+tau*(1-phi)*vh(t), phi=exp(-dt/tau).
tau là **thời gian giữ track Lost hiện hành**, không chọn theoIDSW:30frameUA,
6frameBDD, tương ứng1,2s ởcadence25/5fps đã khóa. Tracker vẫnxóa track sau
lifetime cũ; đây không phải kéo dài buffer. Native state dùngdt1/frame nên
tau lần lượt30/6 theođơnvịframe. Không chỉnh std/Q/R, sốhạt/ESS/seed.

F_extent=[[1,tau*(1-phi)],[0,phi]], truyền mean/cov bằng Fm vàFPF^T+Q cũ.
Động lực mean đã học tiếp tục một đoạn khi xe bị che, rồi tắt dần để hạn chế
extent chạy xa. Không giữv_h vô hạn. Không bơm thêm variance hay dùngGTởruntime.
Shapeaspect và bodyomega/vx/vy Gaussian giữ nguyên. Cfloorh>=1e-3 giữ như nền.
Đây là giảđịnh extent mới, không phải ablation thuần thuật toánPF hoặc khẳngđịnh
mọi xe cóđộng lực này. Khôngnhận bảođảm giảmIDSW/HOTA.

Implementation riêng choLost: lưupriorh/vh, gọiNativeC đểtruyềnbody/aspect với
lostflag0, rồi thayGaussianh/vh bằngDampedF vàQ tínhởheightprior nhưcũ. KhôngLost
phải khớpNativeparentbitwise. Context UA/BDD lấyhorizon30/6 đãkhóa từframebuffer,
khôngtunehorizon hay thửnhiềuphi sauđiểm.

## Kiểm và cổng

Kiểm mean/cov2x2 independentoracle, PSD/ownership/RNG; semigroup củaF (khôngnhận
Qdiscrete hiện tại cósemigroup theo thời gianliên tục). Giớihạntau->0 khôi phục
clampGaussian cũ, tau->infinity khôi phục CVextent; dùngexpm1 đểổnđịnhsố.
KhôngLost nhiều bước khớp parent; controls6/repeat6 trên video.

UAdev12×3seed0/1/2:meanIDSW<472/ít nhất2seed<472/HOTA>=R123PF100-0,001/
FN<=1,002R123. Đạt mớiBDD200×3seed/mot_crowd <5548/2seed/guardrailssoR125PF100;
đạt mớiUA test40×3seed<2498/2seed/guardrailssoNative100R132. Cùnghashrefs/cache/GT.
Raw/MOT/CLS/timing/marker/sourcehash/pairedvideo bootstrap5000. Nếuđạt cảhai
mới đoPCtuầntự42lượt, khôngsuyNanoFPS. Khôngdòscale/dampingtime/N/seed/subset;
các tập đãquansát nhiều vòng nênbáoselectionbias/CI. Khôngsửa src/default/
Nano/khóaluận.

Kích hoạt sauR138: BDD5424,33 qua cổng nhưngUAtest40mean2539,33>KF2498;
khôngdò covariance củaR138, thực hiệnđúng môhìnhextent đãghi trướcđiểmUA này.
