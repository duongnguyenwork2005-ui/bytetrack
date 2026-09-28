# Review vòng 22 — c4b39fe

**Kết luận: PASS WITH LIMITATIONS.** Đóng N21-F1, N21-F2 và N21-F3 của review `9f210e2`. Không phát hiện lỗi chặn mới trong phạm vi đã kiểm.

Nghiệm thu ở phạm vi **luồng offline ảnh/video → TensorRT → tracker CV/EKF trên Nano**, với bằng chứng thực nghiệm 300 frame, không render. Chưa nghiệm thu camera, hiển thị trực tiếp hoặc độ ổn định khi chạy dài. Reviewer kiểm mã, chạy fixture độc lập và đối chiếu dữ liệu bàn giao; không trực tiếp điều khiển Nano trong vòng review này.

## 1. Snapshot được review

- Base: `45e086ac47d778c463719b9b29398b6d99078542`.
- Target: `c4b39fe3861baac76211b2cb82c62d970b719767`.
- ZIP: `review_delta_latest.zip`, **5.036.951 byte**, 346 thành viên.
- SHA256: `536dee9402bde020d85bbf0af0fc9910213571c823738e57bfaa813ee3c1db84`.
- `testzip` đạt; **340/340 payload** khớp hash. Áp patch lên snapshot base đã lưu: **394/394 đường dẫn trong patch** khớp target theo manifest. Không có đường dẫn riêng bị cấm hoặc weights/engine/video/ZIP khác trong gói và patch.
- Có 8 NPZ bị loại thuộc delta vòng 21–22; thêm 1 NPZ vòng 16 đã thiếu từ snapshot base. Các phần thiếu được khai báo trong `DEPENDENCIES.md`. Không xem đây là gói đủ để tự chạy lại TensorRT.

Đối chiếu bổ sung với snapshot `9f210e2`: 145 file hiện có trong các phạm vi code dùng chung, configs và kết quả đã kiểm không đổi byte. Bao gồm `run_tracking.py`, `det_cache.py`, `tracker_ctrv.py`, `ekf_ctrv.py`, `ukf_ctrv.py` và công cụ đóng delta cùng test của nó. Không có căn cứ từ đợt sửa này để chạy lại các thí nghiệm UA-DETRAC/BDD đã khóa.

## 2. Ba finding đã đóng

| Finding | Kiểm độc lập | Kết luận |
|:--|:--|:--|
| **N21-F1 — timer bắt đầu sau khi detector đã xử lý** | Đọc giao thức 2: child chờ `G` sau HELLO; parent bật timer trước khi gửi `G`. Chạy phép 15a/15b qua entry point thật, chèn trễ 1 giây sau HELLO: child chờ khoảng 1.007 ms; cả hoàn tất và dừng sớm đều có mọi frame trong cửa sổ đo. Với dữ liệu Nano, kiểm lại timestamp của **1.800/1.800 frame** thuộc 6 lượt: không frame nào nằm ngoài cửa sổ. | **Đóng** |
| **N21-F2 — replay không tìm cache-root riêng** | Phép 16 ghi vào root riêng có dấu cách; chạy chính `replay_argv` trong metadata, chỉ thay interpreter và placeholder đầu ra. Wrapper `nano_replay.py` gọi runner dùng chung. Exit 0, parity đạt và MOT khớp byte. | **Đóng** |
| **N21-F3 — cấu hình lỗi vẫn khởi động detector** | Phép 17a: 7 ca YAML/override/tham số lỗi đều exit 2, không `detector.pid`, không `run_meta`. Phép 17b/17c: gây lỗi sau spawn, lần lượt trước xử lý và giữa vòng lặp; trả mã 2/3, child thoát và được thu hồi, không báo hoàn tất. | **Đóng** |

Lệnh reviewer chạy trên cây đã tái dựng:

```bash
PYTHONPATH=/workspace/scratch/39272c88b7a6/review_9f210e2/deps \
PYTHONUNBUFFERED=1 python src/test_nano_stream.py
```

**Kết quả độc lập: 20/20 PASS, 1 SKIP, exit 0.** Có chạy SIGINT trên Linux. Phép 14 dùng detector PyTorch thật bị SKIP vì workspace không có checkpoint/ảnh DETRAC; không tính là đã kiểm detector thật tại đây. Môi trường reviewer: Python 3.12.14, NumPy 2.3.5, lap 0.5.13, OpenCV headless 4.11.0.86.

Log `handoff/logs/review22/tests_20260928-1246.log` ghi **34/34 suite PASS, exit 0**, HEAD `b0c645b`. Đây là log bàn giao đã đối chiếu, không phải 34 suite do reviewer chạy lại. Log thiết bị `03_trien_khai_test_v8.log` ghi parity 20/20, run-cache 9/9, stream 20/20 cùng một phép SKIP PyTorch.

## 3. So trực tiếp MOT live/replay của Nano

Nguồn: `handoff/logs/review22/nano_live_v8/results_nano/live_v8/parity/`.

| Live → replay | Frame xử lý | Số hàng MOT mỗi bên | ID phân biệt | Đối chiếu |
|:--|--:|--:|--:|:--|
| `B_trt_cv` → `B_replay_cv` | 300 | 1.272 | 14 | **Byte-identical** |
| `B_trt_ekf` → `B_replay_ekf` | 300 | 1.270 | 15 | **Byte-identical** |
| `D3_stop_file` → `D3_replay_dung` | 56 | 91 | 3 | **Byte-identical** |

Đây là so **byte gốc**, không chuẩn hóa xuống dòng, không remap ID và không dựa riêng vào JSON kết luận parity.

SHA256 chung của mỗi cặp:

```text
CV   15ab38b4c303f5a202a797b29bca6dd9c3ae49b8a2067dce2d5e6c39aced6701
EKF  238dbb2ae5c6d948ff843a008b808cc271760bec177e7c0b6985549840b00ee1
STOP fd410cf417daabec432dde5164688c9dfe2683cb94f9983ac4502c1f9347c739
```

## 4. Tính lại sáu lượt đo Nano

Nguồn: `.../live_v8/bench/luot*/run_meta.json`, `pipeline/MVI_39031.csv`, `timing/MVI_39031.csv` và `bang_luot.csv`.

Mỗi lượt đủ frame 1–300, timing đủ 300 hàng; thống kê sau warm-up dùng 275 frame. Tính lại `FPS = 300 / xu_ly_s`, FPS ổn định từ timestamp kết thúc frame 25 và 300, cùng trung vị decode/infer/tracker/độ trễ: khớp metadata và bảng tổng hợp trong sai số số thực.

| Lượt | FPS xử lý | FPS ổn định | Decode trung vị (ms) | Tracker update trung vị (ms) | Độ trễ trung vị (ms) |
|:--|--:|--:|--:|--:|--:|
| CV r1 | 9,0741 | 9,0341 | 16,381 | 5,777 | 118,113 |
| EKF r1 | 9,4239 | 9,4224 | 11,407 | 6,481 | 113,865 |
| EKF r2 | 9,2033 | 9,1743 | 16,174 | 6,437 | 117,607 |
| CV r2 | 9,1268 | 9,1005 | 16,367 | 5,633 | 118,552 |
| CV r3 | 9,1871 | 9,1919 | 15,929 | 5,574 | 114,867 |
| EKF r3 | 9,3625 | 9,3845 | 11,432 | 6,342 | 114,272 |

- Trung vị FPS: **CV 9,1268**, **EKF 9,3625**; các khoảng min–max khớp báo cáo.
- Cửa sổ xử lý dài hơn khoảng từ bắt đầu đọc frame 1 đến hoàn tất tracker frame 300 **18,07–20,12 ms**. Không còn thiếu phần đầu như vòng 21.
- Trong sáu lượt benchmark, frame 1 bắt đầu sau lệnh START **0,329–1,288 ms**. Con số 0,27–0,34 ms nêu ở phần parity không nên dùng thay cho phạm vi của sáu lượt benchmark.
- **36/36 hash module** trong provenance của sáu lượt khớp mã target theo quy ước LF.
- **170/170 file nguồn Nano hiện diện** khớp danh sách SHA256 bàn giao; 4 NPZ còn lại nằm trong danh sách loại có chủ ý.

Số FPS trên là thông lượng trong cửa sổ xử lý, **không gồm khởi tạo, kết thúc tiến trình, ghi kết quả hoặc render**. Dữ liệu decode khác nhau giữa lượt; không có cơ sở kết luận EKF làm toàn luồng nhanh hơn CV. Kết quả này cũng không thay thế benchmark tracker-from-cache +0,752 ms/frame ở vòng 20.

## 5. Giới hạn và bước tiếp theo

1. Mới có một đoạn 300 frame, ba lượt mỗi cấu hình. Chưa chứng minh ổn định trên toàn sequence 1.470 frame, nguồn camera hoặc vận hành dài; nguyên nhân mất nguồn trước vòng này vẫn chưa rõ.
2. Reviewer không có engine, ảnh và NPZ trong delta để tự chạy lại TensorRT. Kết luận trên thiết bị dựa vào mã, log, timestamp, hash và output gốc đã đối chiếu.
3. Khi chính tracker ném ngoại lệ, runner chưa trả phần MOT đã tích lũy; đường mới dọn child, ghi trạng thái lỗi và giữ thông tin nhận được, nhưng chưa bảo đảm cứu MOT trong mọi lỗi nội bộ. Giới hạn này đã được khai báo, không được mô tả thành khả năng phục hồi hoàn chỉnh.
4. Không suy ra độ chính xác HOTA/IDF1, năng lực camera thời gian thực hay lợi ích huấn luyện detector từ phép nghiệm thu này.

**Có thể chốt vòng 22 và dùng `c4b39fe` làm base review tiếp theo sau khi lưu đúng gói đã kiểm.** Mốc tiếp theo nên chạy toàn bộ 1.470 frame trên Nano với cấu hình hiện tại, giữ log/timing và kiểm live↔replay cùng tính toàn vẹn đầu ra; sau đó mới bổ sung hiển thị/camera. Không cần mở lại tuning hoặc các thí nghiệm tracking đã khóa để hoàn thành mốc vận hành này.

Review được thực hiện trên bản sao trong workspace; không sửa source triển khai, commit, index, cấu hình Git hoặc trạng thái Nano.
