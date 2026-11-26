# R138 — phương án kế nếu R137 chưa đạt: giữ bất định kích thước khi Lost

Chuẩn bị trước điểm R138, chỉ mở quality sau khi R137 kết thúc/không đạt.
Mục tiêu PF ítIDSW hơn KF/EKF/UKF trên cả UA test40 vàBDD200. NềnR132Native100/
Rprediction/initIID/mean/systematic, không ghép các thay đổi R133–R137.

## Phát hiện từ mã và thay đổi duy nhất

NativeC`kernel.c` dòng48 khiLost đặt mean(vh)=0 và đồng thời xóa cả covariance
row/column vh của Gaussian[h,vh]. Đây là mô hình clamp vh xác định của PF cũ;
không mặc định gọi nó lỗi số học. Tuy nhiên không có quan sát mới nào để biết
chắc tốc độ thay đổi kích thước bằng0 khi xe bị che. Thử giả định khác:
**chỉ reset mean(vh)=0, giữ covariance/cross covariance h-vh** rồi predictGaussian
như cũ. Không giữ mean(vh) cũ để hộp tự phình. Không bơm thêm variance/noise.

Implementation có thể reset self.shape_mean[1,1]=0 khiLost rồi gọi cùng Ckernel
với lostflag0; flag chỉ điều khiển clamp shape, body/Q/R/RNG không đổi.
Không chỉnh aspect Gaussian,position/velocity Gaussian,omega hayresampling.
Đây là sửa mô hình bất định khi mất quan sát, không phải phép chiếu vh=0 xác định
chính xác: nếu thật sự áp constraint xác định thì xóa covariance là nhất quán.
Không nhận bảo đảm giảm IDSW; cổng chấm sẽ quyết định.

## Kiểm và cổng

KhôngLost nhiều bước phải khớp parentbitwise kể cảRNG/resampling. Lostshape
phải khớp oracle2x2 FPF^T+Q với mean-onlyreset; covariancePSD, không xóa variance
đã có. Tại gap đầu, body/omega/RNG giữ nguyên so parent. Kiểmupdate/sync/ownership.
Parentcontrols6/repeat6 trướcUAdev12×3seed0/1/2: meanIDSW<472/2seed<472,
HOTA>=R123PF100-0,001,FN<=1,002R123. Đạt mớiBDD200×3seed<5548/2seed/guardrails
soR125PF100; đạt mớiUA test40×3seed<2498/2seed/guardrails soNative100R132.
Giữ cùngGT/cache/association/metric; không dò covariance scale/seed/N sau điểm.

Raw/hash/MOT/CLS/timing/marker/pairedvideo bootstrap5000. Nếu đạt cả hai tập mới
timingPC tuần tự42 lượt, không suyNanoFPS. Các tập đã quan sát nhiều lần nên báo
selectionbias/CI. Không sửa src/default/Nano/khóa luận. NếuR137đạt mục tiêu,
không mởR138 trừ yêu cầu mới.

Kích hoạt sauR137: BDD5456 qua cổng,UA test40mean2525 vẫn>KF2498.
Không thay bất cứ scale/noise/threshold nào theo điểm này; thực hiện đúng
mean-only reset đã mô tả trước khi có kết quảtest40R137.
