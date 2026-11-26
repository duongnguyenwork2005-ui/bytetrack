# R149 — CT Cartesian Gaussian, loại tại dev

Hoàn tất24lượt mới EKF/UKF trên12video UAdev,16919ảnh mỗi bộ. Thay vận
tốc cực bằng Cartesian và tự học từ0, Q Cartesian đẳng hướng cùng hệsố
CV, omega giữhiệnhành. Cùngcurrent-scoreR cho mọi bộ lọc; KF vàPF giữ
nguyên. Đây là thay mô hình/khởi tạo, không sửa lỗi bộ lọc cực cũ.

IDSW: KF449; Gaussian cực EKF439/UKF439; Cartesian EKF461/UKF476;
PF R142mean440,67 (448/436/438). HOTA Cartesian E61,0282%/U61,2211%,
FN35222/35197. CổngIDSW cảhai trượt; EtrượtHOTA soGaussiancực, cảhai
trượtFN. Tuy PF ítIDSW hơn hai bộCartesian, không chọn kết quả này vì
haiGaussian kém đi, không đáp ứng điều kiện tiên quyết E/U<KF.

Numerical đạt96caseJacobian/ODE/rotation,160CVsteps,100sigma-update,
320PSDsteps; maxJacobian3,17e-10, ODE5,35e-12, CV3,30e-14,
sigma-update1,13e-14. Hookrestore đạt. Smoke4historicalcontrols,4candidate
repeats,6PFprefix đạt. 24MOT/CLS/timing/marker kiểmhash, rawtáihợp đạt;
dev/verification.json và inputs.json giữ bằngchứng nguồn/đầu vào.

Đóng nhánh tại dev, không mở BDD/UAfull/kịchbản hoặc dò thêm thông số
để cứu. Kết quả không chứng minh tọa độ Cartesian kém phổquát; chỉ loại
cấu hình đã khóa này cho bài toán/dữliệu đang dùng. Không sửa mặc định,
nguồn src, Nano hay khóa luận.
