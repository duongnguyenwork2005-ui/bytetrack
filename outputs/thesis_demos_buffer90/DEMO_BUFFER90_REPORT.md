# Demo `track_buffer=90`: EKF/CTRV giữ ID, KF/CV không giữ ID

## Kết luận

Có đủ ba kết quả KF/CV, EKF/CTRV và UKF/CTRV trên cùng tập `DETRAC-all` train, cùng detector và cùng tham số tracking với runtime `track_buffer=90`. Đây không phải điều kiện A hoặc B nguyên bản; đây là cấu hình Stage E mở rộng: giống A ở `conf=0.25`, nhưng dùng `track_buffer=90` thay vì 30.

Quét toàn bộ 2.766 sự kiện che khuất đã gộp cho ra 1.003 sự kiện đánh giá được. Có 482 sự kiện EKF `id_match=1`; 13 sự kiện đồng thời có CV `id_match=0`; 7 sự kiện có ID EKF ổn định trong cửa sổ ±25 frame; 5 sự kiện còn lại có chuỗi mất-ghép/tái-ghép thực sự giao với chính khoảng che khuất và có đủ dữ liệu để đối chiếu. Ứng viên xếp hạng cao nhất là `MVI_40871`, GT ID 1, frame 149–161.

Trong sự kiện được chọn, CV switched từ ID 5 sang 115; EKF và UKF đều preserved ID 5. Cả ba có `id_continuous=0` vì đều không được ghép trong 6 frame 156–161. Đây là một ví dụ cụ thể, không phải bằng chứng EKF tốt hơn trên toàn bộ dữ liệu.

## Nguồn và cấu hình

- Dataset/split: `DETRAC-all`, `train`, 60 video. Không dùng test để tìm hoặc xếp hạng demo.
- Kết quả đã lưu:
  - `data/processed/trackers/DETRAC-all/sweep-cv-buf90/data/`
  - `data/processed/trackers/DETRAC-all/sweep-ekf_ctrv-buf90/data/`
  - `data/processed/trackers/DETRAC-all/sweep-ukf_ctrv-buf90/data/`
- Mỗi thư mục có 60 file kết quả, không có file rỗng và có cùng danh sách video.
- Kết quả lịch sử được tạo ở commit `02b0a02ff464fd771c7a220a66576534c51f6434`. Demo cũ trong `outputs/thesis_demos/` không được dùng để chọn hoặc kết luận.

| Thuộc tính | KF/CV | EKF/CTRV | UKF/CTRV |
|---|---:|---:|---:|
| YAML | `configs/buffer_sweep/cv_buf90.yaml` | `configs/buffer_sweep/ekf_ctrv_buf90.yaml` | `configs/buffer_sweep/ukf_ctrv_buf90.yaml` |
| Motion implementation khi rerun | `KalmanFilterXYAH` | `EKFTrackerCTRV` | `UKFTrackerCTRV` |
| `track_high_thresh` | 0.25 | 0.25 | 0.25 |
| `track_low_thresh` | 0.10 | 0.10 | 0.10 |
| `new_track_thresh` | 0.25 | 0.25 | 0.25 |
| `track_buffer` | 90 | 90 | 90 |
| `match_thresh` | 0.80 | 0.80 | 0.80 |
| `fuse_score` | true | true | true |

Detector là Ultralytics YOLO với weights `models/yolov8n.pt`, SHA-256 `F59B3D833E2FF32E194B5BB8E08D211DC7C5BDF144B90D2C8412C47CCFC83B36`, `conf=0.25`, COCO vehicle classes `[2, 3, 5, 7]`. Script lịch sử không truyền seed qua CLI; cả ba dùng cùng mặc định Ultralytics `seed=0`, `deterministic=true`. Ngoài `tracker_type`/motion model, ba YAML có cùng tham số tracking.

## Lọc và xếp hạng

Đánh giá dùng định nghĩa sự kiện gộp hiện tại trong `src/occlusion_eval.py`. Từ 3.349 đoạn thô, 583 đoạn trùng được gộp, còn 2.766 sự kiện. Trong đó 1.003 sự kiện có GT đủ trước/sau; 1.186 thiếu GT trước và 577 thiếu GT sau bị loại.

Luồng lọc là `2.766 → 1.003 → 482 → 13 → 7 → 5`:

1. Sự kiện đánh giá được.
2. EKF có `id_match=1`.
3. CV có `id_match=0`.
4. ID EKF ổn định trong ±25 frame quanh lần tái ghép.
5. Chuỗi frame không ghép ngay trước lần tái ghép phải giao với khoảng che khuất được gắn nhãn; frame ảnh, predicted pool và detection phải tồn tại để replay phép ghép.

Không dùng UKF làm điều kiện lọc. Sau khi hai ứng viên ưu tiên `id_continuous` bị loại vì lần tái ghép không bắt nguồn từ sự kiện được gắn nhãn, năm ứng viên còn lại đều có EKF và CV không liên tục. `MVI_40871` đứng đầu nhờ có lần tái xuất ngay sau sự kiện, khoảng điểm ghép EKF–CV cắt qua ngưỡng rõ ràng, GT đổi hướng 133,56°, và che khuất rõ. Toàn bộ 1.003 hàng cùng lý do giữ/loại nằm trong `candidates_buffer90.csv`.

## Sự kiện được chọn

| Trường | Giá trị |
|---|---|
| Video / GT ID | `MVI_40871` / 1 |
| Sự kiện gộp | frame 149–161, partial occlusion |
| Mức che khuất | mean 54,49%; max 75,38% |
| Chuyển động GT | 0,871 px/frame; đổi hướng 133,558° |
| Lần tái xuất | frame 162, ngay sau cuối sự kiện |
| CV | ID 5 → 115; switched; `id_match=0`; `id_continuous=0` |
| EKF | ID 5 → 5; preserved; `id_match=1`; `id_continuous=0` |
| UKF | ID 5 → 5; preserved; `id_match=1`; `id_continuous=0` |
| Frame không quan sát | 6 cho mỗi model, frame 156–161 |
| Ổn định EKF ±25 | chain chỉ có ID 5; 15 frame khớp sau tái xuất trong cửa sổ |

`id_match` chỉ hỏi ID sau sự kiện có bằng ID trước sự kiện hay không. `id_continuous` yêu cầu ID được quan sát liên tục qua toàn bộ sự kiện; vì có 6 frame không ghép, `id_continuous=0` ngay cả khi EKF/UKF tái dùng đúng ID cũ.

## Đối chiếu tại frame tái liên kết 162

Detection mục tiêu có box `[926.7870, 196.6947, 959.5979, 247.5615]`, score `0.651432`, IoU với GT `0.707019`.

ByteTrack hiện tại tính `iou_distance`, sau đó áp dụng `fuse_score`: `cost = 1 - IoU × detection_score`. `linear_assignment` dùng ngưỡng cost `0.8`, tương đương fused similarity ít nhất `0.2`.

| Model | Box predict của ID 5 | IoU predict–detection | Fused similarity | Cost | Kết quả |
|---|---|---:|---:|---:|---|
| KF/CV | `[933.9541, 186.1548, 954.7812, 217.3682]` | 0.227996 | 0.148524 | 0.851476 | Trượt ngưỡng; ID 5 không ghép detection mục tiêu, mục tiêu xuất hiện với ID 115 |
| EKF/CTRV | `[932.6166, 197.4478, 953.4437, 228.6612]` | 0.389510 | 0.253739 | 0.746261 | Qua ngưỡng; phép LAP đầy đủ ghép detection 16 với ID 5 |
| UKF/CTRV | `[929.6007, 195.8201, 950.4279, 227.0335]` | 0.374509 | 0.243967 | 0.756033 | Qua ngưỡng; phép LAP đầy đủ ghép detection 16 với ID 5 |

Chênh fused similarity EKF–CV là `0.105215`. Đây là phân tích/đối chiếu tại frame tái liên kết từ trạng thái lượt chạy lại, không phải log association runtime đầy đủ của lượt chạy lịch sử.

## Chạy lại không dùng cache

Cả ba tracker được chạy mới từ đầu video đến frame 187 bằng YAML buffer=90; script không đọc cache prediction. Bảng dưới so sánh cửa sổ 64 frame 124–187 với prediction đã lưu. Trajectory mục tiêu tái lập đúng tuyệt đối ở cả ba model: CV 5→115, EKF 5→5, UKF 5→5.

| Model | Frame đối chiếu | Box khớp hình học IoU ≥ 0,9 | Nhãn ID thô khớp | Nhãn sau ánh xạ trajectory 1–1 |
|---|---:|---:|---:|---:|
| KF/CV | 64 | 1.349/1.349 (100%) | 1.349/1.349 (100%) | 1.349/1.349 (100%) |
| EKF/CTRV | 64 | 1.341/1.348 (99,481%) | 1.320/1.348 (97,923%) | 1.337/1.348 (99,184%) |
| UKF/CTRV | 64 | 1.337/1.347 (99,258%) | 1.131/1.347 (83,964%) | 1.327/1.347 (98,515%) |

Sai khác nhãn ở các track khác trong cảnh phản ánh việc kết quả buffer=90 lịch sử có trước các sửa lỗi filter/tracker trên nhánh hiện tại. Vì target ID 5, hình học target và quyết định ghép frame 162 đều tái lập, bằng chứng cho sự kiện được chọn không dựa vào ánh xạ suy đoán; tuy nhiên toàn cảnh không phải tái lập bit-identical.

## Đầu ra

- `demo_ekf_buffer90.mp4`: 1920×1080, H.264 High/yuv420p, 12 fps, 106 frame, 8,83 giây.
- `before_occlusion.png`: frame 143.
- `during_occlusion.png`: frame 156.
- `reappearance.png`: frame 162.
- `metadata.json`: cấu hình, nguồn, phép lọc, kết quả và kiểm chứng rerun.
- `match_cost_at_reacquisition.csv`: box, IoU, fused score, cost, ngưỡng và quyết định ghép.
- `candidates_buffer90.csv`: toàn bộ 1.003 sự kiện đánh giá được và lý do xếp hạng/loại.
- `evaluation/`: các bảng sự kiện gộp và paired evaluation làm nguồn audit.

## Giới hạn diễn giải

- Đây là một sự kiện train được chọn có chủ đích theo tiêu chí EKF thắng CV; không phải ước lượng hiệu năng tổng quát và không dùng test.
- Trên toàn bộ 1.003 sự kiện đánh giá được, preserved là CV 489, EKF 482, UKF 482; demo không đảo ngược kết quả tổng thể đó.
- Xe có đổi hướng lớn nhưng dịch chuyển chậm trong đoạn gắn nhãn; không nên suy rộng rằng CTRV luôn thắng trên mọi cú rẽ hoặc mọi che khuất dài.
- Cả ba model đều không quan sát target trong 6 frame; EKF thắng ở tái nhận đúng ID, không phải ở duy trì detection liên tục.
- Kết quả lịch sử có trước các sửa lỗi trên nhánh hiện tại; rerun tái lập outcome mục tiêu nhưng chỉ khớp 98,5–100% toàn cảnh sau ánh xạ trajectory.
- Các cost là replay đúng implementation trên predicted pool và detection của lượt chạy lại, không phải telemetry đầy đủ được lưu trong lượt chạy lịch sử.
- Sai khác số học GPU rất nhỏ vẫn có thể xảy ra trên phần cứng hoặc phiên bản thư viện khác.
