# Hệ thống tracking phương tiện trên PC — hướng dẫn sử dụng và kiểm chứng

Cập nhật 2026-09-21 (sau review `923305d`). Entry point: **`src/track_video_file.py`** (ngoài `src/demo/**`).
Giao diện: **`src/pc_launcher.py`** — xem `handoff/LAUNCHER_PC.md`. Mốc PC được Codex nghiệm thu ở `074ceca`
(PASS WITH LIMITATIONS).

## 1. Chạy một video

```bash
# File video bất kỳ (.mp4/.avi/...): fps đọc từ metadata; xem trực tiếp + lưu video có bbox/ID/lớp
python src/track_video_file.py --source C:\videos\xe.mp4 \
    --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --classes 1 2 3 5 7 --conf-min 0.25 \
    --device 0 --out results/track_video/xe --save-video --show

# Sequence dataset đã có (không cần ghép video)
python src/track_video_file.py --dataset detrac --part test --video MVI_39031 --tracker-yaml bytetrack.yaml

# Thư mục ảnh .jpg (cần --fps). Thứ tự = GIÁ TRỊ SỐ ở cuối tên (1, 2, 10 — không phải 1, 10, 2)
python src/track_video_file.py --source D:\anh\cam01 --fps 25 --tracker-yaml configs/bytetrack_ukf_ctrv.yaml
# Tên không suy ra được thứ tự (tiền tố khác nhau, không có số, trùng số) -> phải chỉ định tường minh
python src/track_video_file.py --source D:\anh\cam01 --fps 25 --frames-manifest thu_tu.json
```

Phím khi `--show`: **q** thoát (kết quả, cache **và timing của frame cuối** vẫn được chốt đầy đủ — generator được
đóng tường minh tại điểm dừng, không chờ thu gom), **SPACE** tạm dừng/tiếp.

| Nhóm | Tham số | Mặc định | Ghi chú |
|:--|:--|:--|:--|
| Nguồn | `--source` / `--dataset --part --video` | — | file video, thư mục ảnh, hoặc sequence dataset |
| | `--fps` | metadata video / dataset | **bắt buộc** với thư mục ảnh |
| | `--frames-manifest` | — | JSON liệt kê thứ tự file khi tên mơ hồ |
| | `--max-frames` | hết | |
| Detector | `--model` | `models/yolov8n.pt` | sha256 ghi vào `run_meta` |
| | `--device` | `0` | `0` (GPU) hoặc `cpu` |
| | `--det-conf` / `--det-max-det` | `0.05` / `3000` | ngưỡng **thô** của detector = DetectorSpec của det cache; phải `≤ --conf-min` / `≥ --max-det` |
| | `--allow-truncated` | tắt | chấp nhận frame chạm trần `--det-max-det` (mặc định **từ chối**, y hệt replay) |
| Lọc | `--conf-min` | `0.25` | **ngưỡng hiệu lực** vào tracker (c25; `0.10` = c10) |
| | `--classes` | `1 2 3 5 7` | COCO: 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck |
| | `--class-map` | — | vd `7:2,5:2` gộp lớp trước khi vào tracker |
| | `--max-det` | `300` | cắt tiền tố theo frame (sau lọc) |
| Tracker | `--tracker-yaml` | `bytetrack.yaml` (KF-CV) | `configs/bytetrack_ekf_ctrv.yaml`, `..._ukf_ctrv.yaml`, `configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml` |
| | `--set k=v ...` | — | ghi đè YAML — **đi vào tracker thật** (cùng `load_tracker_cfg` với runner) |
| | `--track-buffer-sec` | `1.2` | đổi sang bước theo fps |
| | `--elapsed-mode` | `steps` | Δt từ chỉ số frame |
| | `--seed` | `pf_seed` trong YAML | PF |
| Đầu ra | `--out` | `results/track_video/<tên>` | |
| | `--save-video` / `--show` | tắt | |
| | `--save-cache CACHE_ID` | tắt | ghi detection thô vào det cache → replay bằng `run_tracking.py` |
| | `--warmup-frames` | `25` | số frame đầu bỏ khỏi **tổng kết** thời gian (CSV vẫn đủ) |
| Điều phối | `--progress-file` / `--progress-every` / `--stop-file` | tắt | cho launcher: JSON tiến trình ghi giới hạn tần suất, nguyên tử; stop-file → dừng qua đường phím q (xem `LAUNCHER_PC.md`) |

### Ngưỡng nào thực sự vào tracker

`--det-conf 0.05` là ngưỡng **thô** để det cache dùng lại được cho cả c25 lẫn c10. Nó **không** tự bảo đảm
vòng ghép confidence thấp của ByteTrack có detection: với `--conf-min 0.25`, mọi detection < 0,25 bị lọc
**trước** khi vào tracker (fixture ba frame `[0.9, 0.15, 0.9]` cho `n_dets = [1, 0, 1]` ở c25). Đây đúng
là điều kiện c25 đã khóa trong thí nghiệm. Muốn vòng ghép thứ hai thấy detection 0,1–0,25 thì đặt
`--conf-min 0.1` (c10). Không đổi cấu hình khóa.

### GT trong từng chế độ

- `--source` file/thư mục: **không** đọc GT hay annotation.
- `--dataset`: có thể đọc **vùng bỏ qua** (`gt/ignored_regions.txt` của UA-DETRAC) để tô xám ảnh theo quy
  ước benchmark — **không** đưa ID/box GT vào association. `run_meta.nguon.doc_gt` ghi rõ điều này.

## 2. Đầu ra

```
<out>/data/<tên>.txt         MOTChallenge: frame,id,x,y,w,h,conf
<out>/data_cls/<tên>.csv     frame,id,cls
<out>/timing/<tên>.csv       tracker theo frame: t_predict_ms, t_assoc_ms, t_update_ms, n_steps
<out>/pipeline/<tên>.csv     TỪNG STAGE theo frame (mục 6) — tính lại được mọi tổng
<out>/frames_manifest.json   quy ước thứ tự; `nguon` = toàn bộ danh sách nguồn; `da_xu_ly` = frame THỰC SỰ đã chạy
<out>/run_meta.json          tái hiện (mục 4)
<out>/<tên>_tracked.mp4      nếu --save-video
```

## 3. Thống nhất với đường chạy cache — bốn chỗ dùng **chung** mã

| Bước | Hàm dùng chung | Vì sao |
|:--|:--|:--|
| Detector | `det_cache.detect_frame(model, img, spec)` | live và build cache gọi **đúng một dòng** `model.predict(...)` |
| Lọc | `det_cache.filter_dets(...)` | y hệt `VideoDetections.frame()` |
| Vòng lặp tracker | `run_tracking.track_video(..., on_frame=...)` | cùng `make_tracker`, cùng `n_steps`, `update` cả frame rỗng |
| **Tương thích producer/consumer** | `det_cache.check_compatible(...)` **trước** khi chạy | `conf_min ≥ det_conf`, `max_det ≤ det_max_det`, classes ⊆; vi phạm → **exit 2 sớm**. Frame chạm trần `det_max_det` → **exit 3, không ghi kết quả** (y hệt `VideoDetections` từ chối khi replay); `--allow-truncated` để chấp nhận, được ghi vào `run_meta` và **lệnh replay** |

`LiveVideoDetections` chỉ là **adapter** cùng interface với `VideoDetections`. Không có vòng lặp tracking thứ hai.
`profile_pipeline.py` (cũ) lệch runner ở hai chỗ (không truyền `n_steps`, detector ở 0,25) — không dùng làm đường chạy chính.

## 4. Tái hiện

`run_meta.json` ghi: nguồn (đường dẫn, **sha256** file video, fps, quy ước thứ tự), `detector_spec` đầy đủ
(weights sha256, imgsz, conf, iou, max_det, device, phiên bản), bộ lọc (+ `allow_truncated`, kết quả
`check_compatible`), tracker (YAML + sha256, override, **cấu hình hiệu lực**, seed, backend), `code`
fingerprint (commit + 20 module), **`entry_point`** (sha256 LF của chính `track_video_file.py` +
`working_tree_src`: các file `src/` khác HEAD lúc chạy), thống kê thời gian, và replay:

- **`replay_argv`** — nguồn chuẩn: danh sách đối số, **mỗi đối số một phần tử** (tên nguồn/YAML có dấu cách
  vẫn là một phần tử); `<split>` và `<ten>` là hai phần tử placeholder → thay bằng giá trị thật rồi
  `subprocess.run(argv, shell=False)`.
- **`replay`** — chuỗi để sao chép, quote theo shell của **nền tảng lúc chạy** (`replay_quote`: Windows →
  `list2cmdline`, POSIX → `shlex.join`). Trên shell khác phải quote lại; **không** `.split()` chuỗi này.

Ví dụ khớp cấu hình (cùng `--classes`, cùng ngưỡng):

```bash
python src/track_video_file.py --dataset detrac --part test --video MVI_39031 \
    --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --classes 2 3 5 7 --conf-min 0.25 \
    --save-cache tv_xe --out results/track_video/xe
# lệnh replay đã có sẵn trong results/track_video/xe/run_meta.json -> "replay", chỉ thay <split>/<ten>:
python src/run_tracking.py --cache-id tv_xe --split-name TV --tracker-yaml configs/bytetrack_ekf_ctrv.yaml \
    --tracker-name ekf --videos MVI_39031 --conf-min 0.25 --classes 2 3 5 7 --max-det 300 --fps 25 \
    --track-buffer-sec 1.2 --elapsed-mode steps
python src/nano_parity.py --ref results/track_video/xe --test data/processed/trackers/TV/ekf
```

## 5. Đã kiểm chứng (`src/test_track_video_file.py`, **27/27**)

| # | Kiểm | Kết quả |
|:--|:--|:--|
| F1 | bộ đọc giả lập **20 ms/frame** | `decode_ms` = 20,5 (trước sửa: 0,003); Σ(stage + overhead) == `frame_total` sai số 3,6·10⁻¹⁵ |
| F2a | live → cache → replay bằng **đúng lệnh trong `run_meta`** | exit 0/0, parity ĐẠT lệch 0 |
| F2b | `--det-conf 0.5 --conf-min 0.1` | live **exit 2 sớm** (trước: exit 0, replay ValueError) |
| F2c | `--det-max-det 1`, 1 det/frame | mặc định live **exit 3, không ghi**; `--allow-truncated` → exit 0, lệnh replay có `--allow-truncated`, replay exit 0, parity ĐẠT |
| F2d | `max_det` consumer 10 > producer 5 | live **exit 2** |
| F4 | `1,2,10` / `001,002,010` / `img_7,img_08,img_10` | theo số; tiền tố khác / không số / trùng số → lỗi có hướng dẫn; `--frames-manifest` được tôn trọng; `frames_manifest.json` ghi ra |
| Q1/Q2 | bấm **q** ở frame 1 / 2 (GUI giả lập, warm-up 0) | mọi dòng pipeline **hữu hạn**, `show_ms` frame cuối > 0, `n_frame_tong_ket` == số dòng, Σstage+overhead == `frame_total` (sai số 0), manifest `nguon` 3 / `da_xu_ly` 1 hoặc 2, replay cache tương đương |
| S1/S2 | nguồn `traffic clip` + YAML `cfg dir/ekf ctrv.yaml` | `replay_argv` giữ mỗi giá trị một phần tử; chuỗi `replay` quote `"traffic clip"`; replay **chạy từ `replay_argv`** exit 0, parity ĐẠT; cả ca `--allow-truncated` |
| 6b | **live → save-cache → replay (YOLO thật, GPU)** | **ĐẠT, lệch 0,000e+00** |
| 7 | detection live vs cache gốc thí nghiệm | cùng số det/lớp; bbox lệch 1,8·10⁻⁴ px, conf 3,2·10⁻⁵ |
| 8/9/10 | .mp4 thật + `--save-video`; video đen; đầu vào sai | 60/60 frame; 0 hàng + timing đủ 15 frame; exit 2 / ≠0 |

Tái hiện trước sửa: `handoff/logs/review10/repro_truoc_khi_sua.log` (cả 4 = True); sau sửa:
`repro_sau_khi_sua.log` (cả 4 = False). Hai ca của review `923305d`: `handoff/logs/review11/repro_*.log`
(trước: frame cuối `NaN`, `--videos traffic clip` thành hai video; sau: cả hai hết tái hiện).

### Hai lượt độc lập cùng cấu hình — điều **quan sát được**

Trên `MVI_39031` (1470 frame), cặp `v2_*` không vẽ vs có vẽ (hai lần chạy detector độc lập):

| | hàng/track | khóa `(frame,id)` | phân hoạch | max Δbbox | max Δconf | `nano_parity` |
|:--|:--|:--|:--|--:|--:|:--|
| CV | 5842/72 cả hai | khớp | khớp | 1,2·10⁻⁴ px | 1,19·10⁻⁶ | **KHÔNG ĐẠT** (conf vừa quá 1e-6) |
| EKF | 5845/70 cả hai | khớp | khớp | 1,2·10⁻⁴ px | 6,6·10⁻⁷ | **ĐẠT** |

Tolerance giữ nguyên. Với cặp CV, công cụ báo *"ID và phân hoạch: KHỚP … nguyên nhân: CHƯA XÁC ĐỊNH từ hai
file tracking"* — **không** kết luận nguyên nhân từ hai file tracking. Bằng chứng tách biệt, **từ phép kiểm riêng** trên npz detection thô
(test 7): hai lần chạy detector cho cùng số detection/lớp với lệch cỡ 10⁻⁴ px / 10⁻⁵ conf. Kết hợp hai
điều đó: khác biệt giữa hai lượt nằm ở đầu ra detector; **không** có bằng chứng tracker ra quyết định
khác. Đây là suy luận từ hai phép kiểm, không phải thông báo mặc định của comparator.

## 6. Thời gian — phạm vi từng số

Toàn bộ `MVI_39031` (1470 frame, 960×540, GTX 1650, `--device 0`, backend `ultralytics`), **một lượt**,
trung bình trên **cùng tập 1445 frame sau 25 frame warm-up** cho **mọi** cột; máy có việc khác — số mô tả,
**không** dùng xếp hạng. Nguồn: `results/track_video/v2_MVI_39031_{cv,cv_video,ekf,ekf_video}/`
(`pipeline/MVI_39031.csv` theo frame, `run_meta.json` → `thoi_gian`). Bản cũ (timer sai) giữ ở
`results/track_video/MVI_39031_*` để đối chiếu, **không** dùng.

| ms/frame | decode | mask | detect | tracker | vẽ | ghi | overhead | **frame_total** | fps theo stage | fps vòng lặp |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV | 3,74 | 0,47 | 17,50 | 1,03 | — | — | 0,36 | **23,10** | 43,3 | 40,3 |
| CV `--save-video` | 3,56 | 0,49 | 17,80 | 1,02 | 0,84 | 4,89 | 0,60 | **29,19** | 34,3 | 32,2 |
| EKF | 3,63 | 0,52 | 18,69 | 1,28 | — | — | 0,42 | **24,54** | 40,7 | 38,4 |
| EKF `--save-video` | 3,90 | 0,57 | 20,50 | 1,38 | 0,92 | 5,43 | 0,63 | **33,33** | 30,0 | 28,6 |

Định nghĩa (cột trong `pipeline/*.csv`, tính lại được):

- `decode` quanh đúng `cv2.imread`; `mask` tô xám vùng bỏ qua; `detect` = `det_cache.detect_frame`
  (preprocess + suy luận + NMS); `tracker` = `tracker.update` (gồm predict); `vẽ`/`ghi` = render /
  `VideoWriter.write` (0 khi tắt; `--show` có cột `show_ms` riêng).
- `frame_total` đo trong adapter **từ trước khi đọc frame đến sau khi tracker + vẽ/ghi xong**;
  `overhead` = `frame_total` − Σ stage (lọc, generator, ghi raw) — ghi riêng, không gộp vào stage nào.
- **fps theo stage** = 1000 / `frame_total` (sau warm-up). **fps vòng lặp** = 1470 / `wall_vong_lap`
  (**kể cả** warm-up, kể cả vẽ/ghi khi bật; **không** gồm khởi tạo và ghi kết quả). Hai số khác phạm vi.
- Đối chiếu: Σ`frame_total` (mọi frame) = `wall_vong_lap` − 12…26 ms trên 1470 frame (`wall_ngoai_frame_s`
  trong `run_meta`). Khoảng chênh 5–6 s của bản cũ **nhất quán với** decode chưa được đo (≈3,6 ms × 1470
  ≈ 5,3 s) — là đối chiếu số học sau sửa, không phải phép đo trực tiếp của bản cũ. Khởi tạo (nạp detector
  + import tracker ~0,4 s) 0,8–1,0 s ghi ở `wall_khoi_tao_s`; ghi
  MOT/CSV/pipeline/manifest/cache 0,1 s ở `wall_ghi_s` (mốc đo dừng **trước** khi ghi `run_meta.json`).
- Detector chiếm ~75 %; chênh CV↔EKF ở tracker ~0,25 ms/frame ≈ 1 % `frame_total`. Biến thiên detector
  giữa 4 lượt (17,5–20,5 ms) là **quan sát**; một lượt đo không tách được nguyên nhân (tải máy, clock GPU…
  đều chỉ là giả thuyết).

## 7. Giới hạn

- Camera trực tiếp chưa smoke; `NguonFrame` chưa nhận `cv2.VideoCapture(index)`.
- Thời điểm frame = chỉ số × 1/fps; video **fps biến thiên** bị coi là đều (chưa xử lý VFR).
- Chỉ dùng cho dữ liệu đã có; **không** dùng kết quả này để tuning trên test/BDD val.
- `nano_parity.py` chỉ có hai file tracking: nó báo điều quan sát được; nguyên nhân cần phép kiểm riêng.
