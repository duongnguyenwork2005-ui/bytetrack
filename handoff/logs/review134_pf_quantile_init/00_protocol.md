# R134 — Gaussian quantile initialization cho omega, khóa trước điểm

Tiếp yêu cầu PF ítIDSW hơnKF/EKF/UKF trênUA vàBDD. R133quaUAdev/BDD nhưng
UAtest40mean2518,33 (2516/2523/2516) vẫn>KF2498; khôngđổinu/scales/seed đểcứu.
Thử một cơ chế variance-reduction khác, độc lập trênR132nativePF100/Rprediction,
không dùnglikelihoodmixtureR133 vàkhông tăngsốhạt.

## Thay đổi duy nhất

Khi sinhtrack, thay100omega GaussianIID bằng midpointquantiles củaGaussianprior:
z_i=Phi^-1((i+0,5)/N), chuẩnhoáz vềmean0/variance1 trênNđiểm; omega_i=
mu_prior+sigma_prior*z_i. N=100giữ nguyên. Khôngdòquantileoffset/cắtđuôi/
σ/scales theođiểm. KhôngthaybodyGaussian/shape hoặc logweightsuniform.

Đây là xấp xỉ prior bằngdeterministicquadrature cóhai momentchínhxác, không
phải phânphốiGaussianliêntục chínhxác hoặcSQMCđầyđủ. Highermoments/tails
cósaixấp xỉ. Processnoise/resampling sauinit vẫnngẫu nhiên nhưR132; giữseed0/1/2,
khôngthêmbớtRNGdrawsoR132constructor (cácdrawinitialparticlescũvẫnđượcconsume).
Khôngchọnseedmaymắn; chưa bảođảmIDSWgiảm. GaussianCDF grid cache readonly.

## Kiểm và cổng

Numerical: mean/variance/symmetry/readonly/inputownership/RNGkhôngđổi;
zero-priorvariance hoặcđầu vàoNativeđãcóquantilenodes phải khớpstate/kernel;
PSD/finite qua predict/update. NativeR132controlprefix khớpMOT vàlặpứngviên.
UAtrain dev12×3seed trướcBDD: meanIDSW<472,>=2seed<472,HOTA>=R123PF100-0,001,
FN<=1,002*R123. Đạt mớiBDD200×3seed/mot_crowd cùngR125:
meanIDSW<5548,>=2seed<5548,HOTA>=R125PF100-0,001,FN<=1,002*R125.
Đạt cảhai mớiUAtest40ứngviên120lượt, dùngfreshrefsR132đãhash (KF/EKF/UKF/
Native100) cùngGT/cache, đíchmean<2498/2seed<2498/guardrailssoNative100.
Nếutrượt khôngđổigrid/sigma/N, khôngchọnsubsetvideo hoặcseed sauđiểm.

Tấtcảtestdatasetsđãxemqua cácvòng, khônggọitestmù vàphảibáoselectionbias,
CI/FN/HOTA. Không đổi detector/association/metric/src/mặcđịnh/thesis. Nếuđạt,
đochiphíPCtuầntựriêng, khôngsuyNanoFPS.
