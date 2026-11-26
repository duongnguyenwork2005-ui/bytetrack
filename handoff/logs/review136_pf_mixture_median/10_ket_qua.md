# R136 — median marginal của Gaussian mixture

Hoàn tất36 lượt UAdev ngày19/11/2026; **dừng, không chọn** do trượt cổng hai seed.

NềnR132Native100/Rprediction. Chỉ vị trí báo cáo cx/cy chuyển từ weighted mean
sang median của mixture100Gaussian, có tính within-Gaussian variance. Giữ cloud,
weights,Q/R/init/systematic/RNG/association. Đây là Bayes action choL1, không phải
nghiệm tối ưu IDSW/IoU. Risk matrix quanh point estimator được tính bằng
P_mean+delta*delta^T, không gọi là central posterior covariance.

Numerical500mixture soCDF độc lập qua math.erf đạt, maxCDFerror1,01e−12.
360 bước với cùng observations giữ nguyên toàn bộ kernel arrays/RNG; kiểm
single/symmetric/outlier/Dirac/PSD/ownership đạt. Smoke6parent controls/6repeat đạt.

UAtrain dev12×3seed0/1/2,16.919frame/seed, đủ36 lượt:

- IDSW **470**, từngseed **476/478/456**; tham chiếuKF/EKF/UKF **489/472/474**.
- HOTA **59,2875%**,IDF1 **72,5360%**.
- FN **37.436,67**,FP **12.144,33**.
- MeanIDSW<472 vàHOTA/FN qua cổng; **chỉ1/3seed<472**, trượt yêu cầuít nhất2/3.

Raw metric tái hợp,36MOT/classes/timing/marker vàsourcehash đạt ởverification.json.
Không mởBDD/test40 hoặc đo timing, không thayroot tolerance/axis/blending/N/seed.
CDF solving có thêm công tính toán nhưng chưa đo thời gian để đưa ra phần trăm.

Các tập phát triển đã quan sát qua nhiều vòng nên có selectionbias; mộtseed456
không thay cho kết quả ba seed. Giữ nguyên src/default/Nano/khóa luận.
BướcR137 thử defer angular innovation sau selection với cùngQ/RNG, không chỉnh
median để cứu kết quả.
