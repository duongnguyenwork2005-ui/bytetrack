# R149 — thử biểu diễn Cartesian cho EKF/UKF

Người dùng yêu cầu EKF và UKF ít IDSW hơn KF trên cả BDD200/UA test40,
PF không cao hơn hai bộ đó rồi mới mở các tập kịch bản. R148 full audit
cho thấy UA KF2319/EKF2359, đã đủ biết điều kiện EKF<KF không đạt; UKF
đang hoàn tất. Không làm yếu KF, không mở kịch bản để tìm bảng thuận.

Giả thuyết mới: tham số vận tốc cực (v,theta) cần reset từ hai quan sát
và có suy biến hướng khi v=0. Thử CT vận tốc Cartesian (vx,vy) cho cả
EKF/UKF: [cx,cy,vx,vy,omega,a,h,va,vh], chuyển động quay tích phân chính
xác, liên tục ở omega=0. Vận tốc bắt đầu bằng0 với covariance như KF,
tự học qua tương quan vị trí/vận tốc, không đặt vận tốc từ hai quan sát.

Đây là thay mô hình trạng thái/khởi tạo/nhiễu tương ứng, KHÔNG phải chỉ
đổi phép tính đại số của bộ lọc cực, và không gọi là sửa lỗi baseline.
Q(cx,cy,vx,vy,a,h,va,vh) giữ hệ số XYAH/CV1/20,1/160,1e-2,1e-5;
omega giữ std và init_std đang khóa. Không có theta nên bỏ nhiễu theta,
không dò Q/omega/alpha/threshold trên kết quả. UKF giữ alpha/beta/kappa
hiện hành; sigma root PSD dùng eigen khi Cholesky không được, không thêm
nhiễu omega giả trong oracle CV với varianceomega=0. Đo tuyến tính dùng
cùng cập nhật Joseph chính xác cho E/U,
kiểm với sigma-point update độc lập trước khi chấm.

Tất cả dùng current-score R(c)=max(1-c,.05)R0 R144. KF và PF100 R142
không đổi. Giữ nguyên backend Ultralytics, cache, FPS, buffer, lớp,
ngưỡng c25, association, vòng đời, metric. Thử nghiệm cô lập trong thư
mục này; không thay src, mặc định, launcher, Nano, khóa luận.

Đầu tiên numerical: đạo hàm sai phân trung tâm, ODE quay độc lập,
giới hạn CV, bất biến quay, PSD, khởi động vx/vy từ0, UKF sigma update
khớp Joseph và phục hồi hook. Smoke đối chứng cực khớp R148, lặp mới
khớp, PF vẫn khớp R142. Chỉ sau đó chạy24lượt EKF/UKF dev12. Dùng lại
KF449 và PF448/436/438 đã kiểm R144. Cổng riêng cho thành phần E/U:
E<KF,U<KF, IDSW từng bộ không cao hơn Gaussian cực commonR tương ứng,
HOTA/FN từng bộ không tệ hơn tương ứng quá .001/0.2%. Như vậy không
chọn một Gaussian kém đi để PF dễ thắng. Nếu thành phần đạt nhưng PF
chưa<=min(E,U), chỉ giữ kết quả dev để nghiên cứu PF tiếp, không mở full.
Cổng đầy đủ còn yêu cầu PFmean<=min(E,U), ít nhất2seedPF<=min.

Nếu dev trượt: đóng nhánh, không nới cổng/chọn seed/dò tham số để cứu.
Nếu dev đạt: chạy BDD200 trước, đạt mới UA40, cùng tiêu chí với reference
R148 của từng tập. Chỉ cả hai đạt mới kịch bản; PF vẫn là R142 lịch sử,
không quy thành một cải tiến PF mới. Các tập đã thấy nhiều vòng, ghi rõ
khám phá, không khẳng định test mù hoặc vượt trội phổ quát.

Nguồn: [Roth, Hendeby, Gustafsson, FUSION2014](https://liu.diva-portal.org/smash/get/diva2%3A734112/FULLTEXT01.pdf)
so sánh biểu diễn CT cực/Cartesian và cho thấy độ nhạy vào tọa độ.
Bài đó ủng hộ cực trong thí nghiệm của họ; không trích thành bằng chứng
Cartesian sẽ tốt hơn ở đây. Lượt này kiểm riêng giả thuyết cho bbox ảnh.

Tối đa3job bền vững kể cả trainingGPU; không đo timingPC chung với job
chất lượng, và không suy ra FPS Nano khi máy chưa bật được.
