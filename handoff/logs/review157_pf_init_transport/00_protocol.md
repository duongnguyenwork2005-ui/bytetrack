# R157 — giữ tương quan vị trí/vận tốc khi khởi tạo chuyển động

Khóa trước điểm mới. R153 guided PF N100 là parent; R156 Lost covariance
là nhánh riêng, không trộn. R154 full đã trượt điều kiện E/U<KF trên UA,
nên thử này chỉ ở dev; không tự mở full hoặc dataset kịch bản.

Initializer hiện gán vận tốc từ hai detection, đặt covariance vận tốc theo
chính sách cũ rồi xóa tương quan position–velocity. Giả thuyết: có thể giữ
tương quan đã học ở prior bằng một phép đổi covariance đồng dạng PSD.
Với Lold và Ltarget là Cholesky của covariance vận tốc trước/sau initializer,
A=Ltarget*inverse(Lold), D=diag(I2,A), Pnew=D*Pold*D'. Giữ nguyên marginal
position và target marginal velocity; giữ tương quan đã chuẩn hóa. Không
thêm hệ số/gain/noise, không lấy GT, không thêm random draw. Mean, omega,
weights reset, Q/R/N100/ESS/resampling/association theo parent. Đây vẫn là
chính sách heuristic của initializer, không gọi posterior Bayes chính xác.
Nếu không nhận biết chuyển động, đường chạy phải trùng parent.

Numerical: 500 SPD congruence độc lập, PSD/target/position/whitened cross;
480 bước tắt trùng parent kể cả initializer/RNG; actual enabled init nhiều
dt/gap/heading với covariance/caches đúng; no-motion parity và hook restore.
Smoke tắt khớp 2video×3seed R153, bật lặp đúng. Sau đó dev12×3seed.

Dùng đối chứng mạnh R154dev E435/U430, KF449; meanPF<=430 và ít nhất
2seed<=430, HOTA>=R153-.001, FN<=R153*1.002. Giữ cả kết quả trượt,
không dò strength/seed để cứu. Nếu đạt cũng chỉ giữ component vì Gaussian
full chưa đạt prerequisite. Khóa source/cache/GT/YAML/reference trước chạy.
Không sửa src/default/launcher/khóa luận. Tối đa3job bền vững; không đo
PC timing cùng quality, không suy ra Nano FPS. Dữ liệu dev đã thấy nhiều
vòng, kết quả khám phá có selection bias, không gọi test mù.
