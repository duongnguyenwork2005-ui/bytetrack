# Hệ thống tracking phương tiện trên PC — hướng dẫn sử dụng và kiểm chứng

Cập nhật 2026-09-20. Entry point: **`src/track_video_file.py`** (đặt ngoài `src/demo/**`).

## 1. Chạy một video

```bash
# File video bất kỳ (.mp4/.avi/...): fps đọc từ metadata; xem trực tiếp + lưu video có bbox/ID/lớp
python src/track_video_file.py --source C:\videos\xe.mp4 \
    --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --classes 1 2 3 5 7 --conf-min 0.25 \
    --device 0 --out results/track_video/xe --save-video --show

# Sequence dataset đã có (không cần ghép video)
python src/track_video_file.py --dataset detrac --part test --video MVI_39031 --tracker-yaml bytetrack.yaml

# Thư mục ảnh .jpg (cần --fps)
python src/track_video_file.py --source D:\anh\cam01 --fps 25 --tracker-yaml configs/bytetrack_ukf_ctrv.yaml
```

Phím khi `--show`: **q** thoát (kết quả đã xử lý **vẫn được ghi đầy đủ**), **SPACE** tạm dừng/tiếp.

| Nhóm | Tham số | Mặc định | Ghi chú |
|:--|:--|:--|:--|
| Nguồn | `--source` / `--dataset --part --video` | — | file video, thư mục ảnh, hoặc sequence dataset |
| | `--fps` | metadata video / dataset | **bắt buộc** với thư mục ảnh |
| | `--max-frames` | hết | |
| Detector | `--model` | `models/yolov8n.pt` | checkpoint YOLO; sha256 ghi vào `run_meta` |
| | `--device` | `0` | `0` (GPU) hoặc `cpu` |
| | `--det-conf` / `--det-max-det` | `0.05` / `3000` | ngưỡng **thô** — **cùng** DetectorSpec của det cache; xem mục 3 |
| Lọc | `--conf-min` | `0.25` | detection > conf vào tracker (điều kiện c25; `0.10` = c10) |
| | `--classes` | `1 2 3 5 7` | COCO: 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck |
| | `--class-map` | none | vd `7:2,5:2` gộp lớp trước khi vào tracker |
| | `--max-det` | `300` | cắt tiền tố theo frame (sau lọc) |
| Tracker | `--tracker-yaml` | `bytetrack.yaml` (KF-CV) | `configs/bytetrack_ekf_ctrv.yaml`, `..._ukf_ctrv.yaml`, `configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml` |
| | `--set k=v ...` | — | ghi đè khóa YAML — **đi vào tracker thật** (cùng `load_tracker_cfg` với runner) |
| | `--track-buffer-sec` | `1.2` | đổi sang bước theo fps |
| | `--elapsed-mode` | `steps` | Δt từ chỉ số frame; frame bỏ qua/không đọc được vẫn tính đúng số bước |
| | `--seed` | `pf_seed` trong YAML | PF |
| Đầu ra | `--out` | `results/track_video/<tên>` | |
| | `--save-video` / `--show` | tắt | |
| | `--save-cache CACHE_ID` | tắt | ghi detection thô vào det cache → replay bằng `run_tracking.py` |

**GT không tham gia**: entry point không đọc `gt.txt`/annotation. GT chỉ dùng ở bước đánh giá.

## 2. Đầu ra

```
<out>/data/<tên>.txt        MOTChallenge: frame,id,x,y,w,h,conf
<out>/data_cls/<tên>.csv    frame,id,cls  (lớp của từng track theo frame)
<out>/timing/<tên>.csv      thời gian TRACKER theo frame: t_predict_ms, t_assoc_ms, t_update_ms, n_steps
<out>/pipeline/<tên>.csv    thời gian TOÀN PIPELINE theo frame: decode_ms, detect_ms, n_det_raw, n_det_loc (+ t_update_ms)
<out>/run_meta.json         mọi thứ để tái hiện (mục 4)
<out>/<tên>_tracked.mp4     nếu --save-video
```

## 3. Thống nhất với đường chạy cache — ba chỗ dùng **chung** mã

| Bước | Hàm dùng chung | Vì sao quan trọng |
|:--|:--|:--|
| Detector | `det_cache.detect_frame(model, img, spec)` | live và build cache gọi **đúng một dòng** `model.predict(...)` với cùng `DetectorSpec` (conf 0,05, mọi lớp, `max_det` 3000, imgsz 640). ByteTrack cần detection **confidence thấp** cho vòng ghép thứ hai (`track_low_thresh` 0,1), nên detector **không** được chạy ở 0,25 — ngưỡng đó áp ở bước lọc |
| Lọc | `det_cache.filter_dets(arr, conf_min, cls_set, max_det, class_map)` | y hệt `VideoDetections.frame()` |
| Vòng lặp tracker | `run_tracking.track_video(vd, cfg, dt, elapsed_mode, seed, on_frame=...)` | cùng `make_tracker` (YAML override vào tracker thật, `with_elapsed_steps`), cùng `n_steps`, `tracker.update` gọi **cả khi frame rỗng** → vòng đời vẫn cập nhật |

`LiveVideoDetections` chỉ là **adapter** có cùng interface với `VideoDetections` (`iter_frames()`, `orig_shape`). Không có vòng lặp tracking thứ hai.

`profile_pipeline.py` (cũ) gọi `tr.update(boxes, None)` **không** truyền `n_steps` và chạy detector ở `conf=0.25` — hai chỗ lệch với runner; entry point mới **không** kế thừa cách đó. Script cũ giữ nguyên cho profiling, không dùng làm đường chạy chính.

## 4. Tái hiện

`run_meta.json` ghi: nguồn (đường dẫn, **sha256** file video, fps, số frame), `detector_spec` đầy đủ (weights sha256, imgsz, conf, iou, max_det, device, phiên bản ultralytics/torch/cuda/cudnn/GPU), bộ lọc, tracker (YAML + **sha256**, override, **cấu hình hiệu lực**, `track_buffer` theo bước, `elapsed_mode`, seed, backend), `code` fingerprint (commit + sha256 20 module), thống kê thời gian, và **lệnh replay** sẵn dùng khi có `--save-cache`:

```bash
python src/track_video_file.py --dataset detrac --part test --video MVI_39031 \
    --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --save-cache tv_xe --out results/track_video/xe
python src/run_tracking.py --cache-id tv_xe --split-name TV --tracker-yaml configs/bytetrack_ekf_ctrv.yaml \
    --tracker-name ekf --videos MVI_39031 --conf-min 0.25 --classes 2 3 5 7 --max-det 300 --fps 25 \
    --track-buffer-sec 1.2 --elapsed-mode steps
python src/nano_parity.py --ref results/track_video/xe --test data/processed/trackers/TV/ekf
```

## 5. Đã kiểm chứng (`src/test_track_video_file.py`, 11/11)

| # | Kiểm | Kết quả |
|:--|:--|:--|
| 6b | **Live → save-cache → replay bằng `run_tracking.py` → parity** | **ĐẠT, lệch 0,000e+00** (bit-identical) — cùng detector, cùng lọc, cùng vòng lặp |
| 7 | Detection thô live vs **cache gốc của thí nghiệm** (`yolov8n-coco_conf0.05_all`) | cùng số detection từng frame, cùng lớp; bbox lệch **6·10⁻⁵ px**, conf **3·10⁻⁵** — nhiễu GPU (cuDNN), không phải lỗi |
| 8 | File **.mp4 thật** + `--save-video` | exit 0, video ra đủ 60 frame, `run_meta` đủ trường, fps đọc từ metadata |
| 9 | Video toàn frame **đen** | exit 0, 0 hàng tracking, `timing` có đủ 15 frame → vòng đời cập nhật cả frame rỗng |
| 10 | Checkpoint / nguồn không tồn tại | exit 2 với thông báo rõ / exit ≠ 0 |
| 2, 5 | Frame không đọc được; `on_frame` trả False (bấm q) | unreadable=True vào cache arrays; dừng sớm **vẫn giữ hàng của frame cuối** |

### Sai khác do detector ≠ sai khác do tracker

Hai lượt chạy **độc lập** cùng cấu hình trên toàn bộ `MVI_39031` (1470 frame):

| | CV lượt 1 vs 2 | EKF lượt 1 vs 2 |
|:--|:--|:--|
| Hàng tracking / track | 5842 / 72 (cả hai) | 5845 / 70 (cả hai) |
| Liên kết ID | **y hệt** (phân hoạch khớp, không lệch đánh số) | **y hệt** |
| max lệch bbox | 1,2·10⁻⁴ px | 1,2·10⁻⁴ px |
| max lệch conf | 8,6·10⁻⁶ | 3,1·10⁻⁶ |
| `nano_parity` | KHÔNG ĐẠT (conf > 1e-6) + `[chẩn đoán] liên kết ID y hệt → khác biệt ở đầu vào detector` | như CV |

Detector trên GPU này **không bit-reproducible** (đã biết từ Giai đoạn 1); nhiễu ~10⁻⁴ px / ~10⁻⁵ conf và **không** làm đổi quyết định ghép nào trong 1470 frame × 2 cấu hình. Tolerance conf 1e-6 **giữ nguyên** như đã công bố (đặt cho sai số làm tròn giữa kiến trúc, không phải cho nhiễu GPU); khi so live với một cache có sẵn, kết luận đúng nằm ở mức **phân hoạch**, không ở mức conf.

## 6. Thời gian — tracker riêng, pipeline riêng

Toàn bộ `MVI_39031` (1470 frame, 960×540, GTX 1650, `--device 0`), trung bình **sau khi bỏ 25 frame warm-up**, **một lượt**, máy có việc khác — số mô tả, **không** dùng xếp hạng:

| Cấu hình | tracker `update` | detector | decode | **pipeline không vẽ** | vẽ + ghi | wall |
|:--|--:|--:|--:|--:|--:|--:|
| CV | 0,83 ms | 14,3 | 0,4 | **15,6 ms/frame** | — | 47,5 fps |
| CV `--save-video` | 0,82 | 14,1 | 0,4 | 15,3 | 0,6 + 4,0 | 39,6 fps |
| EKF | 0,99 | 14,8 | 0,4 | **16,2** | — | 47,1 fps |
| EKF `--save-video` | 1,03 | 15,2 | 0,4 | 16,7 | 0,7 + 4,3 | 36,9 fps |

- **pipeline không vẽ** = decode + detect + `tracker.update`; **không** gồm vẽ/ghi/hiển thị.
- **wall** gồm tất cả kể cả warm-up và (khi bật) vẽ + ghi video. `--show` thêm chi phí `imshow`/`waitKey` không ghi ở đây.
- Detector chiếm ~90 %; chênh CV↔EKF ở tracker ~0,16 ms/frame ≈ 1 % pipeline.
- Nguồn: `results/track_video/MVI_39031_{cv,cv_video,ekf,ekf_video}/run_meta.json` (`thoi_gian`) và `pipeline/*.csv`.

## 7. Giới hạn

- Camera trực tiếp chưa smoke (không có camera trong đợt này); `NguonFrame` nhận file/thư mục/dataset, thêm `cv2.VideoCapture(index)` là việc nhỏ nhưng **chưa làm và chưa kiểm**.
- Thời điểm frame lấy từ **chỉ số frame × 1/fps**, không dùng `CAP_PROP_POS_MSEC` (nhiều codec trả 0/lệch). Video có fps biến thiên sẽ bị coi là đều.
- Không có Δt "bỏ qua" tự nhiên trong video file; đường xử lý `n_steps` dùng chung với runner và đã có test ở `test_time_units`/`test_timestep_integration`.
- Chỉ dùng cho dữ liệu đã có; **không** dùng kết quả này để tuning trên test/BDD val.
