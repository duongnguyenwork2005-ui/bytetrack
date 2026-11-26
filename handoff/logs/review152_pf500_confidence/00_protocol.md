# R152 — ablation số hạt canonical500 với confidence-R

Giữ nguyênR142 NativeCartesian RBPF/PredictedNoisePF, confidenceR,
Q/init IID/systematic ESS.5/shapeLost/mean/association. Chỉ thayN100
bằngN500 (sốhạt canonical đã dùng trướcR123), không gridsearch/chọnN
theoIDSW. Nhằm kiểm giới hạn xấp xỉ MonteCarlo khiR sắc hơn theo score.
Không đảm bảo nhiều hạt luôn làmMOT tốt hơn, và phải đo thêm chi phí nếu
ứngviên cólợi. Không trộn ordered/moment-resample/covariancefloor.

Dev12UA,16919ảnh ×3seed0/1/2 =36lượt mới. KF449/EKF439/UKF439 commonR
R144 vàPF100448/436/438 giữ nguyên. NumericalN500 directkernel/parity/
RNG/PSD vàoracleCVidenticalcomponents100/500, guardthực tế phải500;
smoke Gaussian controls/parentPF100prefix vàPF500repeat trước chất lượng.
Khóa source/cache/GT/config theoR142/R144 và hash ởinputs trước điểm.

Cổng PFdev: mean<=min(E,U)=439, ít nhất2seed<=439, HOTA không giảm>
.001 vàFN không tăng>.2% soR142PF100. Báo cảchênh soN100/từngseed,
không đổiESS/N/seed/threshold nếu trượt. Đây là devablation, không mở
BDD/UAfullmới hoặckịchbản từ kết quả này vì R148 đã cho E/U>KF trên
UAfull; phải sửa/kiểm điều kiệnGaussian đầyđủ trước. Nếu PFdevđạt chỉ
giữ thành phần để thử kếtiếp sauprerequisite, không tuyên bố đạt đích.

Tối đa3job bềnvững kể cả GPUtrain; một worker PF tuần tự để dành tài
nguyên chấm model/đoNano. Không đo timingPC khiquality/trainđangchạy.
Không đổi src/default/Nano/khóa luận. Dữliệuđãthấynhiềuvòng, kếtquả
khám phá, không gọi blindtest hay ưu thếthuầnPF phổquát.
