# R161 — vị trí phát sinh IDSW trên dev, không đổi tracker

Trong lúc Nano đo FPS, phát lại **matching CLEAR hiện hành** trên MOT đã
chấm của cùng12video dev. Đối chiếu từng video đúng sốIDSW trongraw,
không chạy detector/tracking mới hoặc mở dataset kịch bản.

Đối chứng KF/E/UcommonR R144 và E/UQfloorR154; PF R142/R153/R156/
R157/R159 đủ3seed. Dùng trace củaR96, hash nguồn và kiểm lại trên dữ
liệu thực; không thay TrackEval/metric. Nhóm: chuyển ID sau >=1 frame GT
không được match, chuyển giữa các GTmatch liên tiếp, và subset<=3frame
sau tái xuất hiện. Chi tiết gap1/2–3/4–8/9–30/>30frame.

Những nhóm này mô tả matching GT–tracker, **không suy ra** trạng thái
Lost nội bộ hoặc che khuất vật lý/quan hệ nhân quả. So thay đổi nhóm không
đảm bảo cùng tập điều kiện đã match. Mục đích chẩn đoán để tránh tiếp tục
dò PFnoise/resampling khi khác biệt IDSW chỉ vài lần vàCI chứa0.
Giữ output/cấu hình/GT/depth/dataset/frame/seed/thresholds; không chọn
video thuận lợi, không sửa source trước đây, không thay mặc định/khóa luận.
