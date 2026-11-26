# R148 — common confidence-R trên BDD200 và UA test40

Đủ720lượt Gaussian mới:600BDD và120UA; dùng lại720lượt PF R142 ba
seed sauSHA/prefixparity. Cùng ảnh/cache/association/FPS/buffer/classes,
cùngR(c)=max(1-c,.05)R0 cho KF/EKF/UKF/PF. PF không được sửa hoặc tune.

BDD200 (39.973ảnh): IDSW KF5319, EKF4854, UKF5028, PF4788,33.
HOTA tương ứng35,4212/35,5073/35,5337/35,4798%. PFseed4785/4785/4795.
Cả thứ hạng người dùng và HOTA/FN guardrails đạt trên tập này.

UA40 (56.340ảnh): IDSW KF2319, EKF2359, UKF2372, PF2362,67.
HOTA tương ứng59,0257/59,2155/59,1480/59,2696%. PFseed2347/2364/2377.
EKF/UKF vẫn cao hơn KF; PF thấp hơn UKF nhưng cao hơn EKF theo mean.
Cổng thứhạng vàtwo-seeds không đạt; HOTA/FN guardrails vẫn đạt.

Bootstrapghép theo video5000mẫu/tập seed148, tính meanCOMBINED PF thật
trong mỗi mẫu thay vì gộp rawassociations xấp xỉ. UA CI95%IDSW EKF−KF
[-19;96],UKF−KF[-8;113], bao0. TrênBDD, EKF−KF=-465,CI[-700;-249];
UKF−KF=-291,CI[-484,025;-100,975]. Đây là mô tả có điều kiện trên dữ liệu
đã quan sát nhiều vòng; CI không loại selectionbias và không chứng minh
ưu thế phổ quát. PF mới không được tạo trong audit này.

Kiểm toàn vẹn đạt:3529record đầu vào/nguồn/bằngchứng,720marker và bộ
MOT/CLS/timing, raw/combined tái hợp; smoke tắtcommonR khớpfixedR,
commonR lặpđúng và PFprefix khớpR142. Chi tiết trongfinal_summary.json,
final_verification.json và {bdd,test40}/bootstrap.json.

Không mở tracking kịch bản vì điều kiện cảhai tập chưa đạt. Giữ nguyên
các devfail R144/R146/R147, không dùng fullaudit để cứu chúng. Bước tiếp
R149 kiểm CTCartesian cho E/U trêndev với KF/PF nguyên trạng. Nano đang
tắt theo người dùng; không có FPS mới. Không sửa src/default/khóa luận.
