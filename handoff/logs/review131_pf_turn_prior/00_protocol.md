# R131 — hiệu chỉnh prior tốc độ quay trên BDD train nội bộ

Khóa trước khi fit tham số hoặc chạy điểm tracking. Mục tiêu của user vẫn là
PF ít IDSW hơn KF/EKF/UKF trên cả UA và BDD. R130 chưa đạt UA:472,33 soEKF472,
chỉ1/3seed<472; dừng hướng đó, không đổi tensor để cứu.

## Dữ liệu hiệu chỉnh và mô hình

Chỉ dùng139video `dev` của split train nội bộ đã khóa
`configs/phase2/detector/splits/bdd5_train_v1.json`. Đây là nhãn BDD MOT train
2020 Scalabel repo đang dùng, provenance có điều kiện theoR34/R35, không phải
BDDval200 hoặcUA test40. Không đọc ảnh hoặc sửa dataset/detector.

Với mỗi ID xe thuộc5lớp, khôngcrowd/ignore: chia đoạn liên tiếp theoframeIndex,
ít nhất7khung. Savitzky–Golay window5/poly2, đạo hàm1/2 với delta0,2giây,
bỏhaiđầu. omega=(vx*ay-vy*ax)/(vx²+vy²). Chỉ giữ tốc độ>=0,5pixel/frame
(ngưỡng khởi tạo CTRV hiện có); chỉ tạo cặp từ hai mẫu omega liên tiếp.
Việc smoothing offline chỉ dùng cho calibration; online không dùng khungtươnglai.

Fit mộtOU zero-mean chung bằng Gaussian conditional MLE:
omega_next=rho*omega+epsilon; rho=exp(-lambda*delta);
Var(epsilon)=S*(1-rho²). Delta calibration0,2s, profileS=MSE/(1-rho²).
Tối ưu scalar lambda trong[0,01;100]s^-1; nếu nghiệm sátbiên, không đủ>=1000cặp
hoặc tham số khôngfinite/dương thì dừngcalibration, không sửa cận sau kếtquả.
Không fit mean riêng từngcamera/dataset, không dò tham số qua trackingIDSW.

Online: native PF100 R126, giữhardinit/QCartesian/Qshape/R/association. Thay
duy nhất prioromega banđầu và randomwalkomega bằngOU đãfit. InternalCTRVdt vẫn1
theoframe. Với cadenceTgiây từ protocol (UA1/25,BDD1/5): sigma_init_frame=
sqrt(S)*T, rho_frame=exp(-lambda*T), q_frame=sqrt(S*(1-rho_frame²))*T.
Vẫn100drawGaussian mỗi predict, không chọnseed. Predictxy trong mỗi frame vẫn
coordinated-turn piecewise constant theoomega trướcstep, không gọi tích phân
liên tụcOU chínhxác choxy. Không trộnR128/R129/R130.

Nguồn mô hình tương quan mũ: tài liệu primaryimplementation Singer củaMathWorks
https://fr.mathworks.com/help/fusion/ref/fusion.tracker.transition.singeraccelerationmodel.html.
Singer chuẩn nói về acceleration; ởđây áp dụngOU choomega, không nhậnđây là
triển khaiSingeracceleration đầyđủ hoặc nguồn chứng minh tracking sẽ thắng.

## Kiểm/cổng

Khóa hash139labels/split/protocol/fitter và tham sốfit trướctracking. Kiểm
OUmoment/chuyểncadence/stationaryvariance/PSD/nativecontrols/repeat.
UAtrain dev12×3seed: meanIDSW<472,>=2seed<472,HOTA>=PF100R123-0,001,
FN<=1,002*R123. Đạt mới chạyBDD200×3seed/mot_crowd cùngcache R125:
meanIDSW<5548,>=2seed<5548,HOTA>=PF100R125-0,001,FN<=1,002*R125.
Nếuđạt cảhai, xácnhậnđủUA test40 bằngđốichứngmới vàPFbaseline. Nếutrượt
không đổiwindow/threshold/MLEbounds/variance/cadence hoặc chọnvideo/seed đểcứu.
Các kếtquả đãqua nhiềuvòng lựa chọn nên không gọitestmù hay ưu thế tổngquát.
Không sửa mặcđịnh/src/thesis, chưa tuyênbốFPSNano.
