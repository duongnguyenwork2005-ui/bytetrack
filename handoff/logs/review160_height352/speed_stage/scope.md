# R160 — giảm một hàng stride32, giữ chiều rộng640

Khóa trước điểm ứngviên. R155 đã thấy352x608/320x576 mất>.5điểmAP50.
Một giả thuyết hình học riêng: thử **352x640**, cùng weights anchorR101,
không pruning/training mới. Ảnh16:9 được letterbox thành khoảng626x352,
so640x360 trongcanvas384x640; scale tuyến tính giảm2,22%, canvas giảm
8,33%pixel. Không kéo méo/cắt ảnh, không bỏ detection/frame hoặc lớp.
Đây là chọn ứngviên phát triển sau R155, không gọi holdout chưa thấy.

Giữ original R109 là đối chứng; ONNXFP32static1x3x352x640/opset12,
NMS ngoài graph, FP16 TensorRTworkspace512. Kiểm ONNX/checker và10ảnh
CPUORT/PyTorch; Nano rawparity ngưỡng cũ box6px/score.02, ghiFAIL nếu
trượt. Không nới ngưỡng. Sau đó audit APengine thật bằng capture.py/
score.py/dataset/archive7190 R155 bất biến; actualAP không đổi rawFAIL
thànhPASS. NhưR155, đủhash/finite/indices/GT/NMS trước timing.

Chất lượng khuyến nghị: AP50 và AP50-95 giảm<=.005, từng AP50class
giảm<=.02 soactual originalR155. Báo5lớp, GTmoto226/bike407. ĐoGPU
original/candidate/candidate/original, pilot4clip indices0,10 mỗiBDD/UA,
cùngR109 full-inference/KF/displayoff/tmpfs/warmup25/150JPEG vàguard.
Đo cảqualityFAIL để ghi trade-off, khôngpromote. Nếu pilot nhanh hơn,
đo80clips ghépcặp nguyênmanifest rồi chấmHOTA/IDSW/FN/FP; nếu không
nhanh hơn thì kết thúc mứcpilot, khôngtuyên bố80. Khôngchọnvideo/seed.

Nguồn/config/ảnh/weights/GT khóa trước điểm; không chạyGPU build/capture/
timing đồngthời trênNano. ChờR150capture3model hoàn tất trước build.
Khôngdeploy/đổilauncher/default/khóa luận, không mởscenarios vì filter
prerequisite chưa đạt, khôngsuy FPStừpixel/MAC. Tốiđa3job bềnvững.
