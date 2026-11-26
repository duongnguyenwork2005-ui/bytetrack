# R140 — omega-ordered stratified resampling, khóa trước điểm

Tiếp mục tiêuPF ítIDSW hơn cảKF/EKF/UKF trênBDD200 vàUA test40. R139dampedextent
trượtHOTA/FN ởUAdev, khôngdòhorizonđểcứu. R140 dùnglạiR132Native100/Rprediction/
initIID/mean vàshapeclampcũ, khôngtrộnR133–R139.

## Thay đổi duy nhất

Khi ESS<0,5N nhưcũ, thay systematic bằng stratified resampling sau sắp omega:
order=argsort(omega,stable), cdf=cumsum(normalized_weights[order]);
u_i~U[0,1), target_i=(i+u_i)/N, selected_i=order[searchsorted(cdf,target_i)].
Gathersđồng bộomega/bodymeans/bodycov; uniformlogweights,counters/cache nhưcũ.
GiữN100,Q/R/init/mean/association, khôngsửaCDFprocessnoise nhưR127.
Resample dùngN uniformdraw thay vì1 systematicdraw, nên chuỗiRNG sauresample đổi;
giữseeds0/1/2, khôngchọnseedmaymắn. Nếukhôngresample phải khớpparent/RNG.

Nguồnprimary:
[Li et al.,Stratification and Optimal Resampling](https://arxiv.org/abs/2004.01975),
[Gerber/Chopin/Whiteley,Ordering and convergence](https://arxiv.org/abs/1707.01845).
Áp dụngordering mộtchiều cho tọađộsampledomega; Gaussianbody phụthuộclịchsửnhiều
chiều nên không tự nhận mọiđiềukiện địnhlý mộtchiều cho quan sátbbox được thỏa.
KhôngdùngHilbertcurve/optimaltransportsolver/SQMC, không bảođảm giảmIDSW.

## Kiểm và cổng

Đối chiếu scalarCDF độc lập với nhiềuweights/omega/uniform; permutation
equivariance khiomega khácnhau, zero-weightboundary/uniformweights/ownership;
đối chiếu expectedoffspring phân tầng bằngintervalintersections. Tắt thayđổi
phải khớpparentqua nhiều resample; PSD/finite và controls6/repeat6 trướcreplay.

UAdev12×3seed:meanIDSW<472/ít nhất2seed<472/HOTA>=R123PF100-0,001/
FN<=1,002R123. Đạt mớiBDD200×3seed/mot_crowd<5548/2seed/guardrailssoR125PF100;
đạt mớiUA test40×3seed<2498/2seed/guardrailssoNative100R132. CùngGT/cache/refs
đãkhóa. Raw/MOT/CLS/timing/marker/sourcehash/pairedvideo bootstrap5000.
Nếuđạt cảhai mới đoPCtuầntự42lượt; khôngsuyNanoFPS. Khôngdòordering/blending/
N/ESS/seed/subsettheođiểm. Các tập đãquansátnhiềuvòng,báoselectionbias/CI;
khôngsửa src/default/Nano/khóaluận.
