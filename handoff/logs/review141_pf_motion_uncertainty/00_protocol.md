# R141 — dự phòng nếu R140 chưa đạt: uncertainty floor khi khởi tạo vận tốc

Chỉ mở quality nếu R140 hoàn tất/không đạt mục tiêu cảhai tập; nếuR140đạt thì
dừng, không mở R141. NềnR132Native100/Rprediction/initIID/mean/systematic/
shapeclampcũ, không trộnR133–R140. Giữ detector/association/metric vàmọi scaleQ/R.

## Nhận xét và giả thuyết

Native hard motion-init lấyv từ hai quan sát nhưng covariance ngang hướng là
(speed*heading_std)^2. Khi speed nhỏ, covariance này rất nhỏ dù detection tâm
cónoise theoheight. Thửgiữ **nguyên pointvelocitymean/initgating/position** như
nền, nhưng không cho velocitycovariance thấp hơn uncertainty từ hai observations.
Không dùng soft-init R129 vốn đổi velocitymean vàđãtrượtFN.

Theo giảđịnh measurementnoise hai khung độc lập nhưmôhình chuẩn, velocityobs=
(z2-z1)/T cócovarianceVobs=(R1+R2)/T². T=pf.dt*max(n_frames,1), đúngper-frame
units. R1 lànoise vịtrí của observationtrước (ghi trướcupdate bằngheightprediction
vàconfidence thực); R2 tạiinit dùngheightprediction/confidenceNone vìhook không
truyềnscore. Nêu rõapproximation này vàgiảđịnh temporalindependence, khôngnhận
Vobs làlowerbound thật nếu detectionerrors tươngquan giữa khung.

VớiC0 làvelocityblock2x2 của Native sauinit, D=C0-Vobs;
Cnew=Vobs+PSD_part(D), PSD_part giữ eigenvalue dương, clipâm về0.
NếuDđãPSD thìtrảC0nguyên đểgiữbitwise. Cnew-C0 vàCnew-Vobs PSD theođạisố;
đây làconservative covariancefloor, khôngnhậnposteriorBayeschínhxác. Position-
velocitycross vẫnnhưNative đểtáchablation, không nhận đãsửađầyđủcorrelated-
measurementinitialization. Không đổimean, không bổsungRNG, khôngtune gain/floor.

## Kiểm và cổng

RandomSPD2x2cases kiểmPSD/noop/ownership vàđạisốfloor; gapscalingT^-2.
Motion-init fixture giữ nguyên means/logw/RNG/shape vàchỉđổivelocitycovariance;
disablefloor phải khớpNative qua nhiều bước/actualinit/PSD/finite.
Controls6/repeat6 trướcUAdev12×3seed0/1/2: meanIDSW<472/2seed<472/
HOTA>=R123PF100-0,001/FN<=1,002R123. Đạt mớiBDD200×3seed/mot_crowd<5548/
2seed/guardrailssoR125PF100; đạt mớiUA test40×3seed<2498/2seed/guardrails
soNative100R132. Cùngrefs/GT/cache/hash, raw/MOT/CLS/timing/marker/boot5000.

Khôngdòfloorweight/gain/N/seed/subset sauđiểm. Nếuđạt cảhai tập mới timingPC42lượt,
khôngsuyNanoFPS. Các tập đãquansátnhiềuvòng, báo selectionbias/CI. Khôngsửa src/
default/Nano/khóaluận.

Kích hoạt sauR140: BDD5446,67 qua cổng,UA2518,67 vẫn>KF2498. Khôngtuneordering/
seedcủaR140, thực hiện đúng covariancefloor đãghi trướcđiểmUA đó. R1 lưucontext
height/confidence trướcupdate vàtínhRkhiinit đểtránh tạo matrixthừa mỗi frame;
tham sốnoise đãkhóa, nêncontext tạođúngR đãdùng ởobservationtrước.
