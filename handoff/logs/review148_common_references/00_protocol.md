# R148 — kiểm đầy đủ điều kiện tiên quyết trên BDD và UA-DETRAC

Yêu cầu người dùng19/11 đặt thứ tự: BDD/UA trước; EKF/UKF ít IDSW hơn KF,
PF<=cảhai thì mới testkịchbản. R144dev12 với cùngconfidenceR đã có
KF449/EKF439/UKF439/PF440,67; R146/R147 thửresampling đều443,33, khôngđạt,
không chọn các ứng viên đó hoặc nới cổng để mở full.

R148 là **audit mốc so sánh**, không thử/tune PF mới và không cứu R146/147.
Cần biết điều kiện EKF/UKF<KF có đúng trên đủBDD200 vàUA test40, thay vì
suy từ12video dev hoặc tiếp tục dò PF theo một chênh lệch dev rất nhỏ.
Giữ đúng3Gaussianfilters cùng cơ chếR(c)R144 và PF R142 hiện tại. Các
cấu hình đã khóa trước khi mở điểmfull này; không sửa chúng sauđiểm.

Chạy mới KF/EKF/UKF commonNSA ở BDD200/39973ảnh và UA test40/56340ảnh:
tổng720 lượtvideo (600BDD+120UA). Dùng lại PF R142 ba seed đã chạy đủ
200/40video sauSHA và6prefix parity mỗi tập; không gọi các lượtPF này là
chạy mới. Giữ fixed-R results riêng để phân biệt noiseadaptation với
thuậttoán.PF nguyênR142, khôngN500/ordered/momentcorrection trongaudit.

Cùngcache/classes/c25/FPS5(buffer6)BDD/FPS25(buffer30)UA vàmetricmot_crowd/
mot củaR142. R(c)=max(1-c,0.05)R0 cho cả4, không giảm chất lượngKF đểđạt
thứhạng. Smokeoffcontrols khớpfixed-R, onrepeat, parentPFkhớpR142; source/
cache/GT/MOT/CLS/timing/marker/raw kiểm lại. Không quá3job gồmtrainingGPU.
Khôngsuy timing từqualityjob sang FPS, đặc biệt khiNanođang tắt.

Báo thực tế trên từng tập: EKF<KF,UKF<KF,PFmean<=min(EKF,UKF),ít nhất2/3
seedPF<=min; HOTA>=fixed-R samefilter-.001,FN<=1.002samefilter. PFkhông
thay đổi soR142 nên không tự tạo hiệu quả PF mới. Nếu bất kỳtậpfullkhông
đạt thì không tự mởscenario. DevR144/R146/R147 vẫn ghi làtrượt theo đúng
giao thức riêng; fullauditkhông biến các thất bại đó thành thành công.

Các tập đã quan sát nhiều vòng, không độc lập/testmù. Không chọn lại
clip/seed/ngưỡng sauđiểm hoặc gọi baseline cónoiseadaptation làpurefilter
ablation. Không sửa src/default/Nano/khóa luận.
