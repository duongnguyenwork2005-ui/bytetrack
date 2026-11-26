# R146 — PF stratified resampling với cùng confidence-R giữa các bộ lọc

R144 UAdev12 đã chạy mới36 lượt: KF449/EKF439/UKF439/PF440,67 IDSW;
EKF/UKF đã ít hơn KF, HOTA/FN mọi filter đạt, nhưng PF mean>439 nên dừng
R144 trước BDD/test40/kịch bản. Không nới ngưỡng hoặc chọn seed.

R146 thêm đúng một cơ chế vàoPF R142: ordered stratified resampling như
R140, giữ100hạt/ESS0,5/Q/R/init/mean/shapeclamp. Giữ confidenceR hiện tại,
cùng công thức với KF/EKF/UKF củaR144. Sort theoomega, target(i+u_i)/N,
N uniformdraw thay1 systematic; không đổi processnoise hoặc dò ESS/N.
Lý do là giảm mất đa dạng sauresample, không bảo đảm giảm IDSW. Đây là
kiểm tương tác của hai kỹ thuật, không coi kết quả fixed-R R140 là kết
quả đã biết của tổ hợp này. Không ghép các kỹ thuật khác để cứu sauđiểm.

Nguồnprimary: [Li etal.](https://arxiv.org/abs/2004.01975),
[Gerber etal.](https://arxiv.org/abs/1707.01845). Chỉ orderomega mộtchiều,
không khẳng định toàn Gaussianbody thỏa điều kiện định lý mộtchiều.

Trình tự: numerical/nguồn -> smoke -> UAdev12×3seed. Reuse đúng36controls
chungR củaR144 sauSHA. Đạt meanPF<=439/ít nhất2seed<=439/HOTA>=R142-.001/
FN<=1.002R142 mới BDD200, rồi UA test40. Mỗi tập full chạy mới3controls
commonR vàPF3seed, giữ fixed-R controls cũ riêng; không làm yếu KF.
Chỉ đạt **cảBDD vàUAtest40** với EKF<KF,UKF<KF,PFmean<=min(EKF,UKF) và
ít nhất2seedPF<=min đó, HOTA/FN khôngsuygiảm quá ngưỡng mới mở kịch bản.
Không đổi detector/classes/c25/association/lifetime/FPS/GT/metric.

Numerical gồm direct scored-update parity khi tắt ordering qua nhiều
resample/PSD/RNG/contextrestore, CDFselection độc lập, sourceR140 hash.
Smoke controller route-off khớp fixed-R, onrepeat; PF route-off ordering
khớpR142 và cùng seedcandidate lặp. Full raw/MOT/CLS/timing/marker/hash.
Các tập đã xem nhiềuvòng; ghi selectionbias, không gọi test mù. Không
chỉnh mặc định/src/Nano/khóa luận. Tối đa3job bền vững kể cảtrainingGPU.
