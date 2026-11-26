# R129 — khởi tạo vận tốc mềm theo covariance, khóa trước điểm

Tiếp yêu cầu PF ít IDSW hơn KF/EKF/UKF trên UA-DETRAC và BDD. R128 đưa nhiễu
hướng vào velocity giảm BDD từ5609,33 xuống5590 nhưng chưa <5548. Giữ nguyên
R128/log, không dò sigma. Hướng mới thử độc lập trên native PF100 R126, không
trộn thêm nhiễu heading chưa đạt mục tiêu.

## Một thay đổi

R118/R126 gán vận tốc bằng toàn bộ chênh hai detection đầu, đặt lại covariance
và xóa position/velocity cross-block. Lần này giữ điều kiện nhận biết chuyển
động và việc tách block khi khởi tạo, nhưng dùng fusion mềm giữa Gaussian vận
tốc prior và pseudo-observation d=(z_xy-ref_xy)/(gap*dt).

R_d=(R_xy hiện tại+R_xy lần quan sát trước)/(gap*dt)^2. R hiện tại dùng
confidence=None vì API initializer không truyền confidence; R trước được lưu
sau update với confidence của update. Với cấu hình hiện hành, scaling theo
confidence tắt nên hai đường dùng cùng quy tắc R. Theo đó:
K=Pvv*(Pvv+R_d)^-1; u_new=u_old+K*(d-u_old);
P_new=(I-K)Pvv(I-K)^T+K*R_d*K^T (Joseph).
R dùng chiều cao/confidence tương ứng đã lưu; ở activate chưa biết confidence
thì dùng measurement_noise mặc định. Không dò hệ số gain.

Pseudo-observation có tương quan với prior và phép đo vị trí cùng frame. Việc
tách block như initializer cũ là xấp xỉ; không gọi đây là posterior Bayes chính
xác hay tính độc lập của hai phép đo đã được chứng minh. Cần kiểm bằng tracking.
Không thêm draw, vẫn100hạt omega và Q/R/association nhưR126.

## Cổng và phạm vi

Numerical fusion đối chiếu nghiệm Gaussian độc lập, PSD/singular-no-motion;
smoke Native100 khớpR123, lặpứngviên 6cặp. UAtrain dev12×3seed đủ36lượt trướcBDD.
CổngUA: meanIDSW<472, >=2seed<472; HOTA>=PF100R123-0,001,FN<=1,002*R123.
Đạt mới chuyển nguyênsourceBDD200×3seed/mot_crowd/cùngR125. BDD đíchmean<5548,
>=2seed<5548, HOTA>=R125PF100-0,001,FN<=1,002*R125PF100.
Nếu đạt hai cổng, xác nhận đủUAtest40 theo kế hoạch riêng trước điểm test40.
Nếu trượt không dò gain hoặc tùy biến theo dataset để cứu, giữ log và nghiên cứu
cơ chế khác. Mục tiêu chưa đạt thì ghi đúng, không gọi thắng bằng video/seed chọn.

Các dữ liệu đã xem nhiều lượt; mọi lựa chọn theo dev và kết quả BDD thăm dò đều
cần nêu thiên lệch lựa chọn, không gọi bằng chứng test mù. Không sửa source cũ,
detector, metric, launcher, src hay khóa luận. Chưa suy tốc độ Nano.
