# R158 — Cartesian CT với cùng chính sách khởi tạo chuyển động của PF

Khóa trước điểm. R149 Cartesian EKF/UKF tự học velocity từ0 đã trượt dev;
R154 polar Q floor không đạt E/U<KF trên UAfull. Không tự mở kịch bản.
Thử một nguyên nhân có thể kiểm: chính sách khởi tạo khác nhau giữa R149
và PF. Giữ CT Cartesian f/J/Q/Joseph update/UKF sigma weights/R(c) R149;
chỉ thêm initializer hai quan sát theo chính sách Native PF R132/R142.

vxy=(zxy-refxy)/(dt*max(gap,1)), speed theo norm và điều kiện min-speed
cũ. Cov velocity = sigma_speed² dd' + (speed*sigma_heading)² ee', với
sigma_speed=2*(h/160)*10, sigma_heading hiện hành; xóa cross-block của
velocity như initializer PF, giữ position/omega/shape. Không thêm noise,
gain, GT hoặc tuning. Nếu đứng yên chưa init thì thử lại ở lần sau. Đây
là heuristic chính sách đầu vào của mô hình, không gọi exact Bayes hoặc
chỉ thay biểu diễn đại số; không làm yếu KF để thuận lợi cho phi tuyến.

Numerical: tắt init khớp R149 cả2filter, công thức chord độc lập nhiều
dt/gap/heading, covariance PSD/marginals/không sửa input/no-motion,
giữ nguyên f/J/UKFupdate, chuỗi curved+Lost, hookrestore. Smoke đối chứng
polar commonR khớp R144, ứngviên lặp, PF R142 giữ nguyên. Sau đó chạy
24dev (EKF/UKF×12video). Khóa sources/cache/GT/YAML/reference trước chạy.

Đánh giá với đối chứng mạnh R154dev E435/U430 (không chọn cặp yếu439):
từng bộ IDSW<KF449 và <=435/430 tương ứng, HOTA>=ref-.001,
FN<=ref*1.002. Báo thêm đối chứng R149461/476 để phân biệt tác động init.
PF dùng lại R142, không gọi kết quả một cải tiến PF. Chỉ component đạt mới
xem xét audit full mới riêng; không full hoặc kịch bản tự động từ dev.
Trượt đóng nhánh, không dò noise/initgain. Dữ liệu đã thấy nhiều vòng,
không test mù. Không sửa src/default/launcher/khóa luận, không NanoFPS
từ PC, tốiđa3job bềnvững.
