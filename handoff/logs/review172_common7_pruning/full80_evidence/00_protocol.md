# R172 — đối chứng cắt tỉa ở mốc bảy epoch hoàn tất chung

R164 joint50 thất bại trong epoch8; R169 control thất bại trong epoch10
dù blocking1. Primary paired study10epoch chưa hoàn tất, không đổi nhãn
thành PASS. Hai lỗi và phần tính toán bỏ đi vẫn được lưu nguyên trạng.

Phép đánh giá phụ này được khóa trước điểm dev/runtime mới: dùng checkpoint
epoch6 (zero-based), tức bảy epoch đầu liên tục của **R164 cho cả hai arm**.
Không dùng control10, control resumed R169 hoặc chọn giữa các lần restart.
Không huấn luyện mới. Cùng 5200 ảnh, teacher/KD/hyper/seed/geometry R164.
Chọn **checkpoint cuối tại mốc hoàn tất chung**, không chọn epoch theo AP.
Budget được chọn do sự cố thực thi, không do điểm tracker hoặc student dev.
Control monitor cũ đã từng được xem; đây vẫn là phân tích phát triển lặp lại,
không phải kết quả test mù hoặc primary10epoch đã hoàn thành.

Trước PC quality: R171 bốn fresh-process diagnostic phải PASS; điều đó không
chứng minh sửa lỗi training dài hạn. Kiểm checkpoint/hash/bảy batch-shape
history/data/source. PC FP32 dùng nguyên evaluator R169/R151, argmax/NMS
384×640 trên7190 ảnh, đủ5 lớp. Export ONNX và10 ORT/PyTorch fixtures giữ
ngưỡng đã khóa. Không thay kiến trúc/precision/loss/metric để cứu kết quả.

Sau khi R167 Nano collection kết thúc, build hai engine control7/joint50_7
trong thư mục mới, cùng FP16/workspace512 và R162 CUDA preprocessing.
Giữ rawparity6px/.02 và cả FAIL, actualTensorRT7190 trước FPS. Macro AP50/
AP50–95 giảm<=0,5 điểm phần trăm, từng AP50class<=2 điểm so cả original và
matchedcontrol. Giữ FAIL để báo trade-off, không promote.

Pilot4 rồi full80 nếu student nhanh hơn original, giữ control, cùng40BDD+
40UA,150JPEG/đoạn,warm25,KF,displayoff,fullinference và các guard/metric cũ.
Swapstart<100MiB, không NanoGPU job khác. So mới với cùngCUDApre, không gộp
SiLU R167 vào pruning. Chưa thay mặc định/launcher/khóa luận hay cổng PF.

Write_paths: logs/review172_common7_pruning,models/r172_common7_pruning,
Nano~/nano_run/r172_common7_pruning_20261121, PLAN mục0. Tối đa3 job.
