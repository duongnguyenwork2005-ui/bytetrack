# R86 — tốc độ v16 so với v21combo trên nhiều video BDD và UA-DETRAC

Đã chạy xong ngày 06/11/2026 trên Jetson Nano, một boot `5daa0db4…`. Giao thức khóa trước: `00_protocol.md`. A = v16 640×640, engine `9fae6b43…`; B = v21combo+C1 384×640, engine `a6912393…`. 40 video BDD val chọn cách đều từ manifest 200 video, 40/40 video UA-DETRAC test. Mỗi video lấy 150 JPEG liên tiếp ở giữa; đây là **mẫu frame**, không phải toàn bộ 39.973 ảnh BDD hay 56.167 ảnh UA. Tất cả lượt tắt HDMI, cùng nguồn ảnh/FPS/regions/tracker trong từng cặp. CV và EKF được chấm riêng.

## Kết quả chính

- **BDD, CV (40 video):** FPS trung vị A 8,57, B 12,96; trung vị tỷ lệ tăng ghép cặp **51,96%**, CI95 bootstrap theo video [49,91; 53,69]%; B nhanh hơn ở 40/40 video. p90 latency trung vị theo cặp giảm 31,53%.
- **BDD, EKF (40 video):** A 8,66, B 12,85 FPS; tăng **47,74%**, CI95 [45,62; 49,06]%; nhanh hơn 40/40. p90 latency giảm 29,35%.
- **UA-DETRAC, CV (40 video):** A 8,79, B 13,52 FPS; tăng **52,38%**, CI95 [51,53; 54,27]%; nhanh hơn 40/40. p90 latency giảm 31,10%.
- **UA-DETRAC, EKF (40 video):** A 8,85, B 12,92 FPS; tăng **47,74%**, CI95 [45,85; 49,66]%; nhanh hơn 40/40. p90 latency giảm 28,07%.

Chỉ số phần trăm chính là **trung vị của `100 × (FPS_B/FPS_A − 1)` theo từng video**, không phải tỷ lệ của hai FPS trung vị. CI dùng 5.000 bootstrap theo video, seed 0. Mỗi nhóm 40 cặp = 80 lượt; tổng cộng 160 cặp, 320 lượt, 48.000 frame được xử lý (12.000 JPEG gốc × hai bản × hai tracker). `fps_xu_ly` là 150 frame chia thời gian cửa sổ xử lý; không cộng tải engine/ghi kết quả.

## Nhiễu đọc ảnh và thời gian toàn lượt

Nano có các tầng decode khác nhau theo lần đọc ảnh. Trong các cặp có độ lệch decode <1 ms: BDD CV 17 cặp tăng trung vị 48,29%, BDD EKF 18 cặp 50,02%; UA CV 20 cặp 51,22%, UA EKF 20 cặp 51,43%. Các nhóm này nhỏ hơn và được chọn theo điều kiện quan sát sau đo, chỉ dùng để kiểm độ nhạy, không thay kết quả chính. Trung vị infer A ~73,02–73,05 ms, B ~44,41–44,46 ms; tiền xử lý A ~11,26–13,36 ms, B ~3,14–4,98 ms.

Nếu tính toàn lượt 150 frame gồm khởi tạo engine, tốc độ hoàn thành video tăng trung vị 29,09% BDD CV, 27,18% BDD EKF, 28,47% UA CV, 26,88% UA EKF. Trung vị thời gian toàn lượt A ~26,9–27,3 s, B ~21,1–21,3 s. Tỷ lệ này phụ thuộc độ dài clip và cách khởi động mỗi lượt, không dùng thay FPS xử lý cho video liên tục.

## Kiểm toán

`audit.json`: 320/320 metadata duy nhất, 150/150 frame mỗi lượt, 0 unreadable/truncated, hash engine/ONNX đúng, BDD FPS 5 và buffer 6/steps, UA FPS 25 và buffer 30/skip, vùng bỏ qua UA được nạp, cùng boot. Telemetry tối đa 50 °C và swap 11 MB; ext4 `errors_count` vẫn 0. Không thay launcher desktop/menu, vẫn v21combo.

Nguồn tái lập: `sample_manifest.json` (80 video và các khoảng frame), `sample80_middle150.sha256` (SHA archive đã xác thực trên Nano trước giải nén), `run_inputs.json`, `meta320.tar.gz` (320 `run_meta.json` và lệnh tương ứng), `final.json` (160 cặp và thống kê), `audit.json`; mã chuẩn bị/chạy/chấm ở cùng thư mục. Archive tar ảnh tạm đã xóa sau khi kiểm hash và giải nén; 12.000 ảnh mẫu gốc vẫn ở `~/nano_run/r86_multivideo_20261106/sample/` trên Nano.

Giới hạn: BDD chỉ lấy 40/200 video và 150 frame giữa mỗi video; UA lấy đủ 40 video nhưng cũng chỉ 150 frame giữa. Đây là benchmark **JPEG lưu trên thẻ, tắt HDMI**, không phải FPS camera, MP4 hay phép đo toàn bộ frame. Chất lượng tracking/detection của C1 so 640×640 đã được đánh giá riêng trong R63; R86 chỉ đo tốc độ, không chấm lại HOTA/IDSW.
