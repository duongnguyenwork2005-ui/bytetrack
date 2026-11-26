# R136 — trung vị marginal của Gaussian mixture cho vị trí báo cáo

Khóa trước điểm19/11/2026. R135 thắngBDD nhưng UA2521,67>KF2498; không đổi
transform/epsilon/seed của R135. Mục tiêu PF ítIDSW hơn KF/EKF/UKF trên cả hai
tập giữ nguyên. R136 quay lại nền R132Native100/Rprediction, initIID và
systematic resampling nguyên bản, không trộn các thay đổi R133–R135.

## Một thay đổi

Thay vị trí cx/cy **báo cáo** từ weighted mean bằng median của từng marginal
Gaussian mixture. Mỗi marginal gồm100 Gaussian với mean và variance từ Gaussian
body điều kiện, weights=exp(logw). Tìm nghiệm sum_i w_i Phi((m-m_i)/sigma_i)=0,5
trong[min(mean_i),max(mean_i)], brentq xtol1e-10px/rtol4eps cố định. Không dùng
median của100 mean đơn thuần: phải tính cả within-Gaussian uncertainty.

Đây là Bayes action cho loss |cx-cxhat|+|cy-cyhat|, do đạo hàm expected absolute
error là2CDF-1; **không** nhận là nghiệm tối ưu IDSW/IoU hoặc luôn tốt hơn mean.
Kích thước, cloud/weights/omega/kernel/Q/R/RNG/resampling và association giữ
nguyên. Point estimate ảnh hưởng association qua bbox như mọi filter estimate.
Không chỉnh nội bộ Gaussian theo median, không overwrite output quá khứ.

Ma trận tương thích trả về là second moment quanh point estimator mới:
P_action=P_mean+(mean-median)*(mean-median)^T; không gọi nó central covariance
của posterior. Native Gaussian core không dùng ma trận này để predict/update.
Các coordinate ngoài cx/cy giữ nguyên. Single/symmetric Gaussian phải khôi phục
mean. Variance0 xử lý CDF Dirac bằng bisection, không bơm noise/variance.

## Kiểm và cổng

Independent CDF via math.erf, mixture ngẫu nhiên và outlier/symmetry/degenerate
cases; kernel arrays/RNG phải khớp parent khi ép cùng chuỗi observations,
PSD/risk matrix/ownership. Parent controls6/repeat6 trướcUAdev12×3seed0/1/2.
UAmeanIDSW<472/2seed<472/HOTA>=R123PF100-0,001/FN<=1,002*R123.
Đạt mớiBDD200×3seed/mot_crowd (<5548/2seed/guardrailssoR125PF100), đạt mới
UAtest40×3seed (<2498/2seed/guardrailssoNative100R132). Cùng GT/cache/refs đã
khóa hash, không dò root tolerance/axis/mean-median blending/seed/N sau điểm.

Nếu đạt hai tập mới đo PC tuần tự để báo chi phí thêm của CDF solving; không
suy NanoFPS. Báo HOTA/IDF1/FN/FP/CIbootstrap5000 cùng IDSW. Các tập đã quan sát
nhiều vòng, không gọi test mù; không đổi src/default/Nano/khóa luận.
