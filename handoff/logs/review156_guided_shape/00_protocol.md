# R156 — giữ bất định kíchthước khi Lost trên guided PF

Khóa trước điểm: R153guidedPF devmean437,67 (439/435/439), E/U R154 mới
435/430 soKF449. Không chọn Gaussian kém đểPF thắng: dùng đúng cặp
reference mới, targetPF<=430 vàít nhất2seed<=430.

Chỉđổi chínhsách shape khiLost trênN100guidedR153: meanvh=0 nhưcũ,
nhưng giữcovarianceh-vh vàvh-vh, rồi predictGaussian. KF/EKF/UKF ByteTrack
hiện tại chỉresetmeanvh, khôngzeroP; đây làđồngbộ handlinguncertainty,
không gọi baselinePF cũ saithuậttoánBayes. R138fixedR đãthử chínhsách
này:devđỡ nhưngfullUA khôngđạt, không quy kết sẽthắng vớiRconfidence/
guidedproposal. Khôngtrộn resampling/moment/N/Q/R hayassociation mới.

Triểnkhai wrapper: khiLost đặtshape_mean[1,1]=0 rồi gọi predict với
lostFalse đểNativeC khôngxóacov; prebody/Qheight/eta đúng R153. Cache
proposal phảigiữshapeprior này, khôngapplyLostclamp trởlạikhi replaybody.
Statekhác vàrandomdraw khôngđổi; khiTracked phảikhớp parentbitwise.

Numerical: disabled Lostpolicy bitparity parentR153/RNG/PSD qua600bước;
oracle2x2 A P A'+Q với meanvh=0 ởnhiềuSPDcov/dt; body/omega/logw/RNG
trước measurement phảikhớpparent dùcovshape khác; tracked/no-Lost và
guidedsnapshotrestore. Smokeoffprefix parentR153 vàonrepeat2video×3seed.
Khóa source/cache/GT/YAML/refs trước36lượt dev12×3seed0/1/2.

Cổngdev: PFmean<=430,two-seeds<=430; HOTA>=R153-.001,
FN<=R153*1.002; E435/U430<KF449 giữ nguyên. Trượtđóngnhánh, khôngdò
gain/damping/horizon/N/seed đểcứu. Đạt chỉgiữcomponent; Gaussianfull
BDD/UA phảiđạt prerequisite trước PFfull/kịchbản theo yêu cầu người dùng.
Không đổi src/default/Nano/khóa luận, không đoPCtime chung vớiquality.
Dữliệu đãthấy nhiềuvòng, kếtquảkhámphá khôngblindtest.
