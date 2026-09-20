# REVIEW_BRIEF — vòng 9: hệ thống tracking trên PC + F2 của review `8bab154`

<!-- BASE_TARGET -->

Base delta **giữ `b623465`** (review `8bab154` còn REQUEST CHANGES; gói này chứa trọn cả vòng 8 lẫn
vòng 9). Ưu tiên đã đổi: hoàn thiện hệ thống trên **PC**; Nano do người dùng tự chuẩn bị.
Chi tiết: **`handoff/HE_THONG_PC.md`**. Không chạy lại thí nghiệm; metric/GT/lock giữ nguyên.

## 1. Rà soát đường chạy

Đã có và kiểm chứng: `run_tracking.track_video()` (vòng lặp cache), `det_cache` (detector + cache),
`load_tracker_cfg`/`make_tracker` (YAML override vào tracker thật). Trùng lặp/lệch tìm thấy:
`profile_pipeline.py` gọi `tr.update(boxes, None)` **không truyền `n_steps`** và chạy detector ở
`conf=0.25` — lệch runner ở cả thời gian lẫn ngưỡng; `src/demo/live_view.py` là vùng riêng, không đụng.

## 2. Đã làm — một entry point, ba chỗ dùng **chung** mã

`src/track_video_file.py`: video file / thư mục ảnh / sequence dataset → YOLO → ByteTrack → MOT +
lớp + timing + `run_meta.json` (đủ tái hiện: sha256 video, checkpoint, YAML; cấu hình hiệu lực; lệnh
replay) + video bbox/ID/lớp (`--save-video`, `--show`, q thoát vẫn ghi đủ) + det cache (`--save-cache`).

| Bước | Dùng chung | Lý do |
|:--|:--|:--|
| Detector | `det_cache.detect_frame()` — tách từ `build_video_yolo` | cùng `DetectorSpec` cache: conf **0,05**, mọi lớp, `max_det` 3000; ByteTrack cần detection thấp cho vòng ghép 2 |
| Lọc | `det_cache.filter_dets()` — tách từ `VideoDetections.frame` | y hệt replay |
| Vòng lặp | `run_tracking.track_video(..., on_frame=None)` | cùng `n_steps`, `update` cả frame rỗng; `LiveVideoDetections` chỉ là adapter |

CV/EKF chỉ khác `--tracker-yaml`; UKF/PF/SORT dùng YAML tương ứng. GT không tham gia.

## 3. Kiểm chứng (`test_track_video_file` **11/11**)

- **Live → save-cache → replay `run_tracking.py` → parity ĐẠT, lệch 0,000e+00.**
- Detection live vs **cache gốc thí nghiệm**: cùng số det/lớp từng frame; lệch bbox **6·10⁻⁵ px**,
  conf **3·10⁻⁵** — nhiễu GPU. Hai lượt độc lập trên 1470 frame (CV và EKF): **liên kết ID y hệt**,
  công cụ parity chẩn đoán đúng "khác biệt ở đầu vào detector, không ở tracker".
- .mp4 thật + `--save-video` → 60/60 frame, `run_meta` đủ; video toàn đen → 0 hàng nhưng timing đủ
  15 frame (vòng đời cập nhật frame rỗng); checkpoint/nguồn sai → exit 2 / ≠ 0; frame không đọc được
  → `unreadable`; bấm q → giữ hàng frame cuối.
- Camera: **chưa smoke** (không có camera).

**Thời gian** (1470 frame, GTX 1650, sau 25 frame warm-up, một lượt): tracker CV **0,83** / EKF
**0,99** ms; detector ~14,5; **pipeline không vẽ 15,6 / 16,2 ms/frame** (47 fps); vẽ + ghi thêm
~4,7 ms → 40 / 37 fps wall. Số toàn pipeline **không gồm** hiển thị/ghi trừ khi ghi rõ.

## 4. Review `8bab154` theo phạm vi mới

**F2 đã sửa**: ghép bằng khóa `(frame,id)`; phân hoạch qua ghép hình học 1-1 (Hungarian, ngưỡng
tolerance, báo mơ hồ) + song ánh ID cố định toàn sequence; tolerance **giữ nguyên**.
`test_nano_parity` **15/15**, gồm hai fixture của Codex (lệch thật 0,0002 px, trước báo 400 px; đảo
thứ tự dòng, trước báo 10 px), hoán đổi ID, và ca đổi liên kết thật (vẫn FAIL). Sửa thêm: hoán đổi
ID làm tập khóa vẫn bằng nhau nên kiểm phân hoạch phải chạy khi mức (a) không đạt trọn vẹn.
**F1/F3 và tài liệu NANO_STATUS mục 5: hoãn, còn mở.** Review này **chưa** PASS.

## 5. Toàn vẹn

Đổi `run_tracking.py`/`det_cache.py` (trong `CODE_FILES`): `test_run_tracking_replay` 10/10,
`test_bytetrack_np_parity` 10/10 (bit-identical cache thật), `test_det_cache` 51/51 → **không** chạy
lại thí nghiệm. `stage_eval` lần sau sẽ báo `code` đổi và đánh giá lại với metric y hệt. Không tuning
BDD val; không đụng `.gitignore`/`src/demo/**`/`live.bat`/`outputs/**`; mp4 mẫu **không** stage.
Full suite: `handoff/logs/review9/`.
