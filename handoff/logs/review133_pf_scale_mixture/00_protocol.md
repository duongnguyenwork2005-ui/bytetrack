# R133 — likelihood vị trí là Gaussian scale mixture, khóa trước điểm

Tiếp yêu cầu user: PF ítIDSW hơnKF/EKF/UKF trênUA vàBDD. R132 đãđạtBDD5464,67
so6345/5548/5655; UAtest40 đầyđủ lại2520 soKF2498/EKF2525/UKF2543, chưađạtKF.
Không đổi seed, video, threshold hoặc hệsốR đểcứuR132. Đây là giả thuyết môhình
đo mới, kiểm lại bằngdev trước khi chuyểnbenchmark. Test40đãđượcquan sát nên
các lần tiếp là thăm dò, không gọi độc lập/mù hoặc chọn tham số theo test40.

## Mô hình được khóa

NềnR132: nativeCartesianPF100, R theoheightprediction, hardinit/Q/resampling/
association giữnguyên. Chỉ thay likelihood/conditionalupdate vịtrí bằng hỗnhợp
3Gaussian zero-mean với các covariance scale. Hỗnhợp xây dựng từ Gauss–Laguerre
3node choGamma shape nu/2=2 (nu4 cốđịnh), weightsdươngchuẩnhoá, rawscale=1/node,
chuẩnhoáscaleđể sum(weight*scale)=1. Nhưvậy covariance đo tổng bằngRposnền,
không đơnthuần tăngRđểgiảmIDSW. Không dònu/node/weights/scales sauđiểm.

Đây là finiteGaussianmixture cókurtosis/đuôi lớn hơnGaussiannền, lấyýtưởng
Student-t scale mixture; **không phải Student-t chínhxác** hoặc VBStudentfilter.
Với mỗi hạtomega, chạy3Gaussianupdatevịtrí bằnglõiC, tínhlikelihoodGaussian
đúng từngcomponent và logsumexp mixture; updateomegaweight theo likelihood
marginal. Gộp posterior3component thành1Gaussian bằng momentmatching trong
mỗi hạt (có between-component covariance). Đây là approximation ởbướccollapse;
shapea/h vẫnGaussian dùngRnền, không robustshapeR120đãthấtbại. Giữ100hạtomega,
khôngthêmRNGdraw; mỗiupdate có3conditionalGaussiancompute thayvì1, cầnđochiphí.

Nguồn ýtưởng: Roth etal., Robust Bayesian Filtering and Smoothing Using
Student's t Distribution, https://arxiv.org/abs/1703.02428;
Li/Sun2019, https://www.mdpi.com/1424-8220/19/22/4830.
Các nguồn khôngbảođảm implementationnày thắng; Rothcũngcảnhbáo approximation
momentmatching cóthểkhôngmongmuốn, dođó bắtbuộckiểmthựctế vàcổngFN/HOTA.

## Kiểm/cổng

Numerical: covariancechuẩnhoá/kurtosis, mixturelikelihood vàposteriormoment so
nghiệmGaussianmixture độc lập, PSD; onecomponent scale1 khôi phụcR132/seed;
smoke Native100/R132 control khớpMOTđãkhóa vàrepeatứngviên.

UAtrain dev12×3seed: meanIDSW<472,>=2seed<472,HOTA>=R123PF100-0,001,
FN<=1,002*R123. Đạt mớiBDD200×3seed/mot_crowd/cùngR125:
meanIDSW<5548,>=2seed<5548,HOTA>=R125PF100-0,001,FN<=1,002*R125PF100.
Đạt cảhai mới UAtest40×3seedứngviên (120lượt), so9nhómR132 đãkhóahash,
dùnglạiKF/EKF/UKF/native100chạyfresh cùnginput; khôngđổireference.
Đíchtest40meanIDSW<2498,>=2seed<2498,HOTA>=native100test40-0,001,
FN<=1,002*native100test40. BáocảCIghépvideo5000mẫu, HOTA/IDF1/FN/FP.
Nếutrượt dừngứngviên, khônglấysubset/seedcólợi, khôngđổiweights/scales.
Chưađổimặcđịnh/src/thesis, khôngsuyFPSNano; timingtuầntựsauqualitynếuđạt.
