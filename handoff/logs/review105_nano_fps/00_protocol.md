# R105 — tăng FPS bằng lịch suy luận YOLO thích ứng

- Mục tiêu là FPS ổn định của toàn pipeline trên Jetson Nano. So trực tiếp v21combo và YOLO11n 50/50 TensorRT FP16 384×640. Giữ KF, lớp, c25, vùng bỏ qua, hàng đợi bốn khung và quy tắc chấm của R102. Không chỉnh launcher.
- Trên bốn đoạn UA-DETRAC 25 fps, so v21combo, YOLO11n đủ khung, YOLO11n cách khung cố định, cách khung có bật lại khi phát hiện chuyển động ngoài hộp, và biến thể bật lại thêm khi hộp vừa biến mất. Trên bốn đoạn BDD 5 fps, chạy v21combo và YOLO11n đủ khung. Hai đoạn BDD đầu có thêm kiểm thích ứng phải tự chạy đủ khung và giữ đầu ra.
- Các chế độ thích ứng chỉ xét bỏ YOLO ở khung chẵn. So ảnh xám 160×90 với ảnh liền trước, loại vùng ignore và hộp YOLO c25 gần nhất nới bốn pixel ở lưới thu nhỏ. Khi một ô 10×10 có ít nhất 12 pixel thay đổi quá 16 mức xám, chạy YOLO. Nhánh thứ hai cũng chạy nếu có hộp c25 nội vùng vừa biến mất, với ngưỡng IoU 0,15. Các ngưỡng cố định trước khi chấm. Khung bỏ YOLO vẫn được đọc và KF dự đoán, phân biệt với kết quả detector rỗng.
- Giữ giao thức 150 JPEG/đoạn và tmpfs, tắt hiển thị, chạy xen kẽ theo đoạn, kiểm boot/nhiệt/swap/lỗi thẻ. Đo số lần chạy YOLO, FPS, p90, HOTA, IDSW, FN và độ trễ khớp lần đầu cho các ID mới được cả hai chế độ khớp. So tăng tốc trực tiếp với v21combo và đánh đổi chất lượng với YOLO11n đủ khung. Đánh giá thăm dò trên mẫu R102, không dùng các đoạn này để tinh chỉnh tham số hay đổi cấu hình luận văn.

## Cổng chạy và bằng chứng (bổ sung triển khai 12/11)

- `bench_nano.py smoke` chạy 8 lượt trên clip đầu BDD và UA, mỗi clip 150 frame, sau smoke ảnh giả lập trên PC. `full` yêu cầu smoke hoàn tất với cùng mã/package/engine, rồi chạy đúng 30 lượt mới.
- v21combo dùng package gốc R86/R102; YOLO11n dùng bản sao R105 của R103. Driver R86/R101/R102 giữ nguyên byte trong `drivers/`, kiểm bằng `driver_manifest.json`. Engine/ONNX phải khớp R102, không dựng lại engine.
- BDD 3/3/2/2 lượt; UA 5/5/5/5 lượt; đảo thứ tự chế độ theo clip. Lưu hash từng ảnh, command, metadata, CSV pipeline đúng frame 1–150, MOT và lớp trong `evidence.tar.gz`. BDD guarded phải không skip và MOT/lớp trùng full.
- `eval_benchmark.py <logs-full> <scratch-stage-moi>` dùng staging/first-match R103, metric/cấu hình R102, bổ sung hỗ trợ chấm nhóm hai clip. FPS ghép cặp là trung vị mức tăng từng cặp với v21combo/full; báo từng clip và gộp, HOTA/IDF1/IDSW/FN/FP, p90, inference và first-match theo R103.
- `test_r105.py` chỉ CPU/fixture, dùng source PC sao riêng rồi ghép R103→R105; không chứng minh package Nano đã chạy. `make_bundle.py` tạo bundle để giải nén vào `~/nano_run/r105_nano_fps`; evaluator chạy PC, stage ở scratch đã cấp. Không có bước train, cài gói, thay launcher/boot/config hay engine.

## Đối chiếu bằng chứng bridge đã nhận

Bridge thực tế chạy bản R105 sớm có runner SHA `94be03b3e12ca8d3da1f9f16db0c136a2a6ee16b92ff333554155ae271f96ef1`, trước bổ sung hash toàn bộ package/driver-manifest của bundle local. Giữ nguyên JSON cũ; evaluator có cổng riêng ghim runner này, kiểm snapshot source và driver gốc, package_sources, code/engine/ONNX/archive, smoke cùng source/engine. Mỗi run ghi thêm sáu hash source LF trọng yếu; không coi đó là hash đầy đủ toàn package. Chưa tái chạy bản bundle mới chỉ để đổi schema; hạn chế này phải có trong báo cáo/review.

`eval_benchmark.py` kiểm lại cấu hình từng run và tính lại FPS/p90 trực tiếp từ timestamp CSV (bỏ 25 frame đầu). `audit_benchmark.py` kiểm cả 8 smoke + 30 full, giữ CSV từng lượt/từng ID, hash GT/evaluator và đối chiếu MOT R102. Các kiểm này không đổi lịch, ngưỡng scheduler hoặc metric đã chốt. `run_tier1.py` chạy đồng bộ test/audit/syntax/kiểm bundle, ghi exit thực ở `tier1_checks_current.json`. Lượt chấm/fixture cuối đặt `pc_stage_verified/` và `pc_test_scratch/` ngay trong write_paths R105 của repo; không ghi staging mới ngoài repo. Đây là Tier 1 của coder, chưa thay thế phiên reviewer Codex độc lập.
