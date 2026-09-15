# Giai đoạn 2 — Giao thông hỗn hợp, PF–CTRV, BDD100K-MOT, cache detection

Nhánh: `phase2-mixed-traffic` (tách từ `fix/tracker-evaluation-correctness` @ `35a560f`). Chỉ commit cục bộ, **không push/merge**. Môi trường: `khoaluan-mot` (Python 3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124, trackeval 1.3.0, GTX 1650).

## 0. Tóm tắt trạng thái

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1 | Kiểm tra baseline; sửa lỗi `shared_kalman` bỏ qua tham số bộ lọc | ✅ 5 test cũ PASS (77 phép); lỗi xác minh + sửa (late-binding); `test_filter_config.py` 9 test / 32 phép; mã pre-fix (`35a560f`) vs mã mới **bit-identical 9/9** trên cache thật (`verify_shared_kalman_regression.py`) — mục 2, 11.3 |
| 2 | Thời gian / đơn vị / mô hình nhiễu (6 yêu cầu) | ✅ `time_units.py` (audit + `ElapsedStepsMixin` + đổi đơn vị tương hợp), `test_time_units.py` 8 test / 30 phép (frame rỗng ≠ frame bỏ qua; PF cùng quy ước) |
| 3 | Detection cache + runner đọc cache | ✅ `det_cache.py`, `run_tracking.py`; `test_det_cache.py` 37/37; `test_run_tracking_replay.py` 7/7: capture-and-replay **bit-identical** với `model.track` (CV/EKF/UKF) + **trực tiếp vs cache cùng tiến trình** 100 % (frame,id) trùng |
| 4 | Adapter BDD100K-MOT, crowd-ignore IoA, fixture, gap events | ✅ `mot_datasets/bdd100k_mot.py`, `trackeval_ext.py`, `check_bdd100k.py`; `test_bdd100k_adapter.py` 42/42, `test_trackeval_ext.py` 10/10 (**trùng số** với `trackeval.datasets.BDD100K` gốc; 3 trường hợp ignore; gap → `gap_evidence`, không suy diễn); regression GĐ1 trùng khít |
| 5 | PF–CTRV | ✅ `pf_ctrv.py`, `PFSTrack`/`PFByteTracker`; `test_pf_ctrv.py` 26/26 |
| 6 | Pilot Nhóm A (bộ lọc) + Nhóm B (SORT, baseline = SORT nguyên bản) trên dev UA-DETRAC (12 video) | ✅ chạy thật — mục 7 (kèm tầng che khuất) |
| 7 | SORT nguyên bản / SORT–CV / SORT–EKF | ✅ `sort_tracker.py`, `test_sort_tracker.py` 21/21 |
| 8 | Runner thí nghiệm + tổng hợp + smoke end-to-end | ✅ `run_experiment.py`, `aggregate_phase2.py`, `test_smoke_phase2.py` 11/11 |
| 9 | Detector ứng viên: export ONNX + kiểm tra op TRT 8.2; dataset trộn; train script | ✅ chuẩn bị (không train dài) — mục 8 |
| 10 | Profiling PC + scripts Nano | ✅ đo trên PC; **Nano chưa đo** — mục 7.3, 9 |
| 11 | **Bàn giao mốc 1** (file, lệnh/test, kết luận `shared_kalman`, kết quả, điểm chặn) | ✅ mục 11 |

Dữ liệu/thiết bị: **BDD100K-MOT không có trên máy Windows này** (đã tìm toàn bộ D:, Downloads, thư mục dự án; `ByteTrack + EKF + Kitty/datasets/BDD100K` rỗng). Bộ nhãn dự phòng "gộp lớp 0" cũng không có. Nano chưa kết nối. Mọi phần BDD chạy trên fixture tổng hợp.

## 1. Hiện trạng đã xác minh (không suy đoán)

- Pipeline GĐ1: `baseline_track.py` gọi `model.track(frame, persist=True)` từng frame → YOLO chạy lại cho mỗi motion model; **không có cache**. Detector khóa: `models/yolov8n.pt` (sha256 `f59b3d83…`), classes COCO [2,3,5,7], conf 0,25 (A) / 0,10 (B), imgsz 640, iou 0,7.
- Bộ lọc: state 9-D `[cx,cy,v,theta,omega,a,h,va,vh]`, `dt = 1 bước`, mọi tham số nhiễu theo bước ([time_units.py](src/time_units.py) mục A).
- **Lỗi tìm thấy**: [tracker_ctrv.py](src/tracker_ctrv.py) `CTRVSTrack.multi_predict` dùng thuộc tính lớp `shared_kalman` (tạo lúc import). `BYTETracker.update` chỉ gọi `multi_predict`, nên mọi tham số bộ lọc đặt trên instance của tracker (từ YAML) **bị bỏ qua ở bước dự đoán**. Đo được: đặt `std_theta = 1.0` trên bộ lọc của tracker → sau `multi_predict`, `P[θ,θ]` vẫn tăng đúng bằng mặc định 0,010². GĐ1 chỉ dùng mặc định nên **kết quả cũ không bị ảnh hưởng** (đã chứng minh bit-identical, mục 2.1).
- TrackEval pip 1.3.0: lớp `BDD100K` dùng schema cũ (`index`, `Crowd`); `MotChallenge2DBox` không có vùng bỏ qua (crowd) — cần lớp mở rộng (mục 5).

## 2. Bước 1 — Sửa `shared_kalman` + regression

Thay đổi: [ekf_ctrv.py](src/ekf_ctrv.py) giữ tham số nhiễu/khởi tạo **trên instance** (`std_theta, std_omega, init_std_theta, init_std_omega, heading_std_after_init, min_speed_for_heading`); [ukf_ctrv.py](src/ukf_ctrv.py) chuyển `**kwargs`; `CTRVSTrack.multi_predict` dùng `stracks[0].kalman_filter` (**không** ghi đè biến dùng chung nào; `shared_kalman` chỉ còn là dự phòng khi track không có bộ lọc); `CTRVByteTracker.get_kalmanfilter()` đọc khóa `ctrv_*` từ YAML. **Late-binding** (sửa sau rà soát): 4 tham số khởi tạo (`init_std_theta`, `init_std_omega`, `heading_std_after_init`, `min_speed_for_heading`) nếu **không** truyền tường minh thì đọc `config.*` **lúc gọi** (đúng ngữ nghĩa pre-fix mà `tune_omega_noise.py` dựa vào: tạo bộ lọc rồi mới ghi đè `config.CTRV_INIT_STD_OMEGA`); truyền tường minh (YAML) thì cố định trên instance. `explicit_params()` cho biết tham số nào là tường minh.

Kiểm tra ([test_filter_config.py](src/test_filter_config.py), 9 test / 32 phép):
1. Bộ lọc mặc định == `shared_kalman` cũ: initiate / try_initiate_from_motion / multi_predict / update **bit-identical** (EKF, UKF).
2. Qua `BYTETracker.update` 12 frame (có frame rỗng): output mới == cũ bit-identical.
3. **Đi đúng đường lỗi**: override `ctrv_std_theta = 1.0` qua YAML → `P[θ,θ]` tăng +1,028 (mặc định +0,029) sau `multi_predict` do `BYTETracker.update()` gọi.
4. Kịch bản 79 frame tổng hợp: mã tại `35a560f` (trước sửa) vs mã mới → EKF/UKF/clamped **bit-identical** (243/242/243 box).
5. **Dự đoán đơn == dự đoán nhóm**: `predict(m_i, P_i)` từng track vs `multi_predict(M, C)` — EKF/UKF bit-identical, `KalmanFilterXYAH` `allclose` 1e-12 (ultralytics vector hóa khác đường).
6. **Hai tracker khác cấu hình trong cùng tiến trình**: A (`ctrv_std_theta = 1.0`) và B (mặc định) cập nhật **xen kẽ** cùng chuỗi detection → `P[θ,θ]` của A tăng ~1,0², của B ~0,01²; box của B chạy xen kẽ == B chạy một mình (so theo frame, bỏ qua ID); `CTRVSTrack.shared_kalman._std_theta` không đổi. Hạn chế đã biết (không thuộc bộ lọc): `BaseTrack._count` của ultralytics là bộ đếm ID toàn cục nên hai tracker chạy xen kẽ cho ID khác nhau — so sánh theo box.
7. **Late-binding**: tạo `EKFTrackerCTRV()` → đổi `config.CTRV_INIT_STD_OMEGA = 0.55` → `initiate()` cho `P0[ω,ω] = 0.55²` (legacy giữ nguyên); bộ lọc có tham số tường minh không đổi.

**Bằng chứng trên dữ liệu thật** ([verify_shared_kalman_regression.py](src/verify_shared_kalman_regression.py)): nạp mã pre-fix (`git show 35a560f:src/{ekf_ctrv,ekf_ctrv_clamped,ukf_ctrv,lost_association,tracker_ctrv}.py`, sha256 ghi trong [baseline_provenance.json](results/phase2/shared_kalman/baseline_provenance.json) cùng phiên bản thư viện, sha256 `yolov8n.pt`, 3 YAML, `meta.json` cache) và mã mới, chạy cả hai trên cache thật `MVI_39031` + `MVI_20052` + `MVI_39931` → [diff_summary.csv](results/phase2/shared_kalman/diff_summary.csv): EKF/UKF/clamped mặc định **bit-identical 9/9** (5845/8106/4759 dòng EKF; 5840/8106/4763 UKF; 5845/8106/4758 clamped); override `ctrv_std_theta = 0.05`: **pre-fix == mặc định** (override bị bỏ qua, 3/3 video), **fixed ≠ mặc định** (3/3) — chứng minh lỗi và bản sửa trên detection thật.

## 3. Bước 2 — Thời gian, đơn vị, mô hình nhiễu ([time_units.py](src/time_units.py))

Theo 6 yêu cầu đã thống nhất:
1. **Legacy giữ nguyên**: dt = 1 bước, tham số GĐ1 không đổi; regression bit-identical (mục 2) + capture-and-replay (mục 4).
2. **BDD Δt thực**: adapter đọc `timestamp` (ms) per-frame nếu có, không thì `frameIndex/5`; ghi `dt_source`; `check_bdd100k.py` thống kê phân bố Δt (median/min/max, số khoảng > 1 bước, khoảng lẻ). Đơn vị: v [px/bước], ω [rad/bước]; px/s = v·fps, rad/s = ω·fps.
3. **Audit trước, quy đổi sau** (docstring mục A): KF–CV không có dt (F cố định dt=1), Q/R/P0 tỉ lệ theo h theo bước; EKF/UKF/PF cùng Q/R/P0; `std_theta` [rad/bước] là gia số random-walk của hướng, `std_omega` [rad/bước²] là gia số của turn rate — chọn ở GĐ1 từ giới hạn vật lý + quét độ nhạy, **không** áp √Δt mặc định. Với FPS mới, giá trị theo bước phải **chọn lại bằng cùng quy trình trên dev** (thí nghiệm, không quy đổi công thức). Đổi đơn vị state/cov chỉ để báo cáo: `convert_state_units` dùng biến đổi tương hợp `S P Sᵀ` (nhất quán cả phần tử chéo, bảo toàn PSD; test round-trip).
4. Bảng audit đủ 4 bộ lọc; đổi FPS = đổi độ dài bước, khác với đổi mô hình nhiễu (→ Nhóm C).
5. **Không khẳng định** "bước ≡ giây" (chế độ giây chưa triển khai).
6. `track_buffer` theo giây → bước (`--track-buffer-sec 1.2` → 30 @25 fps, 6 @5 fps; ghi cả hai vào `run_meta.json`). `ElapsedStepsMixin.update(n_steps=k)`: `frame_id += k`, `multi_predict` lặp k lần (Q cộng dồn đúng số bước), khởi tạo 2 khung hình nhận đúng số bước, `track_buffer` đếm theo bước. `k = 1` bit-identical lớp gốc (CV/EKF/UKF); runner tính k từ Δt thực (`steps_between`).

## 4. Bước 3 — Detection cache ([det_cache.py](src/det_cache.py)) + runner ([run_tracking.py](src/run_tracking.py))

- Cache xây **một lần**: conf 0,05, **tất cả 80 lớp COCO**, max_det 3000, iou 0,7, imgsz 640, mask vùng bỏ qua như GĐ1; `meta.json` ghi sha256 weights, phiên bản ultralytics/torch/cuda/cudnn, GPU, mọi tham số predictor, git commit. Lọc conf (`>` ngặt), lớp, `max_det` (cắt tiền tố) **lúc load**.
- **Tính tương thích chứng minh bằng số** ([test_det_cache.py](src/test_det_cache.py), 37/37): trên tensor dự đoán thô, `NMS(0.05, all, 3000)` lọc `>0.25/0.10`, lớp con, `[:300]` **== NMS trực tiếp** với điều kiện đó (5 điều kiện × 2 kích thước); cache bị cắt `max_det` **phá** tính chất (chứng minh và strict-mode từ chối); `Boxes` từ cache có dtype/giá trị y hệt đường ultralytics.
- **Capture-and-replay** ([test_run_tracking_replay.py](src/test_run_tracking_replay.py)): bọc `BYTETracker.update` trong một phiên `model.track` thật (150 frame `MVI_39031`), ghi lại detection, nạp lại qua runner → output **bit-identical** cho CV/EKF/UKF; đồng thời khớp 100% (IoU≥0,9) với `testA-*` đã lưu.
- **Đối chiếu toàn video với GĐ1**: cache 0,05 lọc 0,25/0,10 chạy CV trên `MVI_39031` (1470 frame): cùng **5842/6064 dòng, 72/71 track, 100% cặp (frame,id) trùng** `testA-cv`/`testB-cv`; tọa độ lệch median 0, p99 6e-5 px; 3,3% dòng lệch > 0,5 px (GPU không bit-reproducible giữa hai lần chạy detector — đã biết từ GĐ1). Tracking từ cache: **~1000 fps** (CV) so với ~30 fps pipeline đầy đủ.
- Nguồn `--source gt`: detection tổng hợp từ GT (jitter, dropout, seed) — dùng cho fixture và làm "oracle detector".

## 5. Bước 4 — BDD100K-MOT ([mot_datasets/bdd100k_mot.py](src/mot_datasets/bdd100k_mot.py))

Schema đã đối chiếu tài liệu (box_track_20, Scalabel): list frame `{name, videoName, frameIndex, [timestamp], labels[{id, category, attributes{occluded, truncated, crowd}, box2d}]}`; 5 fps, ~200 frame/40 s, 1280×720; 8 lớp đánh giá + 3 distractor; `crowd` → vùng bỏ qua (IoA > 0,5). **Phải xác minh lại trên dữ liệu thật** — `check_schema()` fail-loud với schema cũ (`index`/`Crowd`), frameIndex không liên tục, id trùng, category lạ, thiếu attributes/box2d.

Quyết định thiết kế (ghi rõ trong docstring):
- Interim chung với DETRAC, **giữ nguyên** `category`/`occluded`/`truncated`/`crowd` gốc; `occlusion_ratio = 0.5` **quy ước** cho cờ `occluded` (che một phần, mức độ không biết; không bao giờ ≥ 0,90). Khoảng trống annotation (**gap**) của track **không được suy diễn** là che hoàn toàn: `occlusion_eval --gaps` tạo sự kiện gap `[last_seen+1, reappear−1]` với `has_full = False`, `has_gap = True` và `gap_evidence` = `occluded_adjacent` (frame ngay trước và sau gap đều `occluded`) hoặc `unknown` (chưa xác định); `aggregate_phase2.py` báo cáo tách tầng `partial` / `full` (chỉ DETRAC) / `gap-occluded_adjacent` / `gap-unknown`. Nhãn đã gộp lớp / mất thuộc tính chỉ dùng kiểm tra pipeline, **không** dùng báo cáo xe máy hay che khuất.
- Xuất MOTChallenge theo lớp: chính `BDD-<part>` (car+truck+bus+motorcycle+bicycle → class 1; crowd/distractor/train → class 13, conf 0), phụ `BDD-<part>_car`, `_motorcycle`, … **Rider/pedestrian chỉ thống kê, không đánh giá** (detector COCO không có lớp rider; không gộp người với xe).
- Tách output tracker theo lớp bằng **majority vote** một lớp/track (`STrack.cls` bị ghi đè mỗi frame).
- [trackeval_ext.py](src/trackeval_ext.py) `MotChallengeCrowdIgnore`: class 13 → `gt_crowd_ignore_regions`, det chưa khớp có **IoA > 0,5** bị loại (theo `trackeval/datasets/bdd100k.py`). Vì sao không dùng "class 8 distractor": det của một xe trong box crowd lớn có IoU ≪ 0,5 với crowd nên không bị loại → FP sai ([test_trackeval_ext.py](src/test_trackeval_ext.py), 10/10: (1) det chưa khớp trong vùng ignore → loại (gốc 3 FP, lớp mới 0 FP); (2) det **khớp GT hợp lệ nằm chồng vùng ignore → vẫn TP**, không bị loại — thứ tự đúng: ghép GT trước, ignore sau; (3) FP thường ngoài vùng (IoA 0,4) → vẫn FP ở cả hai; UA-DETRAC không class 13 → **trùng khít** lớp gốc; **đối chiếu với `trackeval.datasets.BDD100K` gốc** trên fixture chuyển sang schema cũ: HOTA/MOTA/IDSW/FP/FN **trùng số** cho car/truck/motorcycle — phải đặt `should_classes_combine = use_super_categories = False` vì TrackEval 1.3.0 raise IndexError khi gộp super-category với một lớp).
- `run_trackeval.py`: `<BENCHMARK>-<split>`, `--dataset-class mot|mot_crowd` (mặc định theo tiền tố). `occlusion_segments.py`: `--parquet --fps --out-tag`, CSV rỗng có header. `occlusion_eval.py`: `--tag --fps --context-s --gaps`, lọc GT class 1. **Regression**: chạy lại trên DETRAC-test tái tạo **trùng khít** 4 CSV của `results/test_eval/A` và `occlusion_segments_test.csv`/`full_*`.
- Fixture [data/fixtures/bdd100k_mini](data/fixtures/bdd100k_mini) (3 video: crowd, gap, occluded, truncated, distractor, timestamp có khoảng bỏ frame). [test_bdd100k_adapter.py](src/test_bdd100k_adapter.py) 42/42 đi hết đường (fixture có car G gap `unknown` và car M gap `occluded_adjacent`): JSON → interim → MOT theo lớp → cache gt-synthetic → runner CV/EKF (`n_steps = 2` tại chỗ timestamp nhảy 0,4 s; `track_buffer` 1,2 s = 6 bước) → tách lớp → TrackEval crowd-ignore (FP = 0) → occlusion_segments/eval với gap.
- Diễn giải: camera trên xe (ego-motion) + 5 fps làm IoU association yếu hơn nhiều so với camera cố định UA-DETRAC; giao thức khác chính thức ở chỗ dùng MOTChallenge + IoA thay vì toolkit `bdd100k`.

## 6. Bước 5 — PF–CTRV ([pf_ctrv.py](src/pf_ctrv.py))

Bootstrap PF, mỗi track N particle; **dùng chung** f (vector hóa `f_batch`), Q, R, `initiate`, `try_initiate_from_motion` với EKF/UKF (cùng giả định nhiễu). Log-weight + logsumexp; ESS → systematic resampling (1 số ngẫu nhiên); trung bình vòng θ; cov thực nghiệm (residual wrap); suy biến (Mahalanobis² tối thiểu > `pf_degenerate_maha2`) → đặt lại vị trí quanh z, đếm; thiếu measurement → chỉ predict, `vh = 0` áp lên mọi particle khi LOST; roughening tùy chọn (mặc định tắt). RNG từng track `SeedSequence([seed, video, track_id])` → không phụ thuộc thứ tự pool; cùng seed bit-identical. Bbox cho association = ước lượng có trọng số (`mean`) hoặc particle MAP; `track.mean/covariance` luôn được cập nhật nên `project`/`gating_distance` (Mahalanobis) đúng nghĩa. `PFTrackView` giữ giao diện `KalmanFilterXYAH` để `STrack.update/re_activate` của ultralytics chạy nguyên vẹn.

Kiểm tra ([test_pf_ctrv.py](src/test_pf_ctrv.py), 26/26): `f_batch == f` (1e-12); ω≈0; θ qua ±π; 30 bước không measurement (hữu hạn, trace tăng); suy biến; seed/video reproducibility; đảo thứ tự pool; `multi_predict == predict()`; resampling; **nhất quán với EKF khi N = 3000** trên đường thẳng nhiễu Gaussian (RMSE 0,96 vs 0,97 px, v 4,92/5, θ 30,9°/30°); vòng đời bookkeeping == EKF; runner `--seeds`.

**Quan sát quan trọng (không phải lỗi)**: với N = 100, PF suy biến (sample impoverishment) trên cùng đường thẳng: RMSE 26,5 px, θ lệch (−49° vs 30°); hướng/turn-rate không sửa được sau khi resample vì Q của θ/ω rất nhỏ (0,010/0,005 mỗi bước — thiết kế cho bộ lọc Gaussian). N = 500 ≈ EKF. Trên kịch bản tổng hợp có frame rỗng, PF-100 sinh 9 ID cho 3 xe (EKF: 3); N = 500 + roughening 0,5: 3 ID. Đây là hệ quả của giữ **cùng giả định nhiễu** với EKF; cải tiến (roughening, nhiễu tham số) là ablation Nhóm C.

Cấu hình: `configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml` (khóa `pf_*`; tham số ghép cặp giữ nguyên `bytetrack.yaml`).

## 7. Bước 6 — Pilot Nhóm A trên dev UA-DETRAC

Manifest [results/phase2/manifests/detrac_dev.json](results/phase2/manifests/detrac_dev.json): 12/60 video train, phân tầng tercile(`n_full_occ_segs`) × tercile(`pct_curved`), ưu tiên đa dạng thời tiết, seed 0 (chọn **trước** khi xem kết quả); `detrac_test.json` = 40 video test (đánh giá cuối, chưa chạy). Cache `yolov8n-coco_conf0.05_all` (12 video, 16 919 frame, 5,5 phút GPU).

Thí nghiệm [configs/phase2/experiments/groupA_detrac_dev.yaml](configs/phase2/experiments/groupA_detrac_dev.yaml): điều kiện A (>0,25) và B (>0,10); tracker cv / ekf / ukf / pf100 / pf300 / pf500 (PF × seed 0,1,2); cùng cache, cùng association, `track_buffer` 1,2 s. Kết quả: `results/phase2/groupA_detrac_dev/{A,B}/summary.csv`, `per_video.csv`, `paired_bootstrap.csv`, `timing.csv`, `event_outcomes_*.csv`, `bootstrap_*.csv`.

### 7.1 Nhóm A — so sánh bộ lọc trong cùng ByteTrack (dev, 12 video, cùng cache)

COMBINED_SEQ của TrackEval (PF: trung bình 3 seed, `HOTA_sd_seed` ≤ 0,005); `preserved`/`id_continuous` = tỉ lệ giữ ID qua 184 sự kiện che khuất đủ điều kiện (xác định từ GT; PF gộp 3 seed = 552 lượt chấm). `predict` = ms/frame của bộ lọc (CPU, tracking từ cache).

**Điều kiện A (conf > 0,25 — cấu hình khóa GĐ1)**

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | predict ms/frame | preserved | id_continuous |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cv | 0.5900 | 0.5200 | 0.6789 | 0.7177 | 0.5706 | 489 | 12119 | 37564 | 0.13 | 0.505 | 0.310 |
| ekf | 0.5925 | 0.5207 | 0.6838 | 0.7245 | 0.5719 | 472 | 12103 | 37448 | 0.33 | 0.489 | 0.310 |
| ukf | 0.5879 | 0.5206 | 0.6734 | 0.7151 | 0.5716 | 474 | 12128 | 37458 | 2.24 | 0.495 | 0.310 |
| pf100 | 0.4933 | 0.4692 | 0.5383 | 0.5509 | 0.4566 | 2069 | 16605 | 44821 | 1.26 | 0.294 | 0.217 |
| pf300 | 0.5182 | 0.4809 | 0.5763 | 0.5909 | 0.4762 | 1549 | 16173 | 43480 | 2.54 | 0.342 | 0.248 |
| pf500 | 0.5269 | 0.4847 | 0.5896 | 0.6056 | 0.4838 | 1352 | 15975 | 42992 | 3.71 | 0.350 | 0.261 |

**Điều kiện B (conf > 0,10)**

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | predict ms/frame | preserved | id_continuous |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| cv | 0.5989 | 0.5274 | 0.6899 | 0.7319 | 0.5690 | 404 | 15746 | 34212 | 0.12 | 0.565 | 0.408 |
| ekf | 0.5993 | 0.5282 | 0.6899 | 0.7336 | 0.5700 | 384 | 15814 | 34041 | 0.31 | 0.554 | 0.397 |
| ukf | 0.5990 | 0.5283 | 0.6891 | 0.7318 | 0.5699 | 390 | 15810 | 34054 | 2.51 | 0.560 | 0.402 |
| pf100 | 0.4938 | 0.4758 | 0.5329 | 0.5487 | 0.4501 | 1992 | 20091 | 42170 | 1.49 | 0.315 | 0.277 |
| pf300 | 0.5213 | 0.4871 | 0.5766 | 0.5937 | 0.4722 | 1498 | 19680 | 40493 | 3.01 | 0.366 | 0.319 |
| pf500 | 0.5329 | 0.4908 | 0.5962 | 0.6131 | 0.4786 | 1322 | 19479 | 40124 | 3.54 | 0.388 | 0.328 |

Chênh lệch **trung bình theo video** so với `cv` (CI 95% bootstrap theo video, 5000 lần, seed 0; PF: trung bình seed trong video):

| cond | tracker | ΔHOTA | ΔIDF1 | ΔAssA | ΔIDSW |
|:--|:--|:--|:--|:--|:--|
| A | ekf | −0.0003 [−0.0037, +0.0028] | +0.0001 [−0.0061, +0.0063] | −0.0008 [−0.0069, +0.0053] | −1.4 [−4.4, +1.9] |
| A | ukf | −0.0041 [−0.0087, −0.0005] | −0.0062 [−0.0146, +0.0003] | −0.0085 [−0.0174, −0.0018] | −1.3 [−4.3, +1.7] |
| A | pf100 | −0.1277 [−0.1746, −0.0803] | −0.1996 [−0.2804, −0.1186] | −0.1934 [−0.2667, −0.1171] | +131.7 [+59.0, +210.5] |
| A | pf300 | −0.0943 [−0.1342, −0.0557] | −0.1536 [−0.2249, −0.0855] | −0.1425 [−0.2045, −0.0803] | +88.4 [+36.4, +145.1] |
| A | pf500 | −0.0827 [−0.1182, −0.0483] | −0.1383 [−0.2051, −0.0747] | −0.1243 [−0.1801, −0.0687] | +71.9 [+28.8, +118.8] |
| B | ekf | +0.0001 [−0.0017, +0.0021] | +0.0003 [−0.0028, +0.0038] | −0.0004 [−0.0038, +0.0033] | −1.7 [−4.3, +0.6] |
| B | ukf | −0.0008 [−0.0023, +0.0007] | −0.0015 [−0.0053, +0.0018] | −0.0021 [−0.0050, +0.0008] | −1.2 [−3.7, +1.1] |
| B | pf100 | −0.1313 [−0.1780, −0.0850] | −0.2042 [−0.2853, −0.1271] | −0.2024 [−0.2736, −0.1297] | +132.4 [+59.1, +212.6] |
| B | pf300 | −0.0956 [−0.1349, −0.0588] | −0.1550 [−0.2252, −0.0905] | −0.1461 [−0.2055, −0.0879] | +91.1 [+39.1, +148.4] |
| B | pf500 | −0.0822 [−0.1189, −0.0482] | −0.1362 [−0.2043, −0.0739] | −0.1231 [−0.1778, −0.0702] | +76.5 [+31.1, +126.0] |

Giữ ID qua che khuất (`occlusion_eval`, paired bootstrap theo video, Δ tỉ lệ preserved so với cv): A: ekf −0.016 [−0.049, +0.005], ukf −0.011 [−0.044, +0.008], pf500 −0.14…−0.17 (CI không chứa 0); B: ekf −0.011 [−0.051, +0.022], ukf −0.005 [−0.047, +0.025].

Tỉ lệ giữ ID theo **tầng che khuất từ GT** (`occlusion_strata.csv`; full = ratio ≥ 0,90 — n = 33 sự kiện; partial = 0,10 ≤ ratio < 0,90 — n = 151; PF gộp 3 seed):

| tracker | A full | A partial | B full | B partial |
|:--|--:|--:|--:|--:|
| cv | 0.091 | 0.596 | 0.061 | 0.675 |
| ekf | 0.061 | 0.583 | 0.061 | 0.662 |
| ukf | 0.061 | 0.589 | 0.061 | 0.669 |
| pf100 | 0.020 | 0.353 | 0.000 | 0.384 |
| pf300 | 0.040 | 0.408 | 0.040 | 0.437 |
| pf500 | 0.030 | 0.419 | 0.040 | 0.464 |

Che hoàn toàn: mọi bộ lọc ≤ 9 % (2–3/33 sự kiện; switched ≈ 0,64–0,70) — đúng như GĐ1: với che ≥ 0,90 kéo dài, khác biệt motion model không thể hiện qua IoU association; n = 33 quá nhỏ để tách biệt (xem memory `stratum-curved-long-occlusion-qua-nho`).

Đọc kết quả (dev, chưa phải test):
- EKF ≈ CV ≈ UKF (mọi CI chứa 0, trừ UKF-A hơi kém CV về HOTA/AssA) — nhất quán với GĐ1.
- **PF–CTRV bootstrap với cùng Q/R của EKF kém rõ rệt** ở cả 3 N (HOTA −0,08…−0,13; IDSW gấp 3–4 lần; FN tăng mạnh). Cơ chế: sample impoverishment (mục 6) → dự đoán vị trí kém → IoU thấp → track mất/sinh mới. Tăng N cải thiện đơn điệu (100→300→500) nhưng còn xa CV; seed ổn định (sd HOTA ≤ 0,005). Kết luận có điều kiện: **PF không mặc nhiên giải quyết che khuất**; muốn PF hợp lý phải đổi thiết kế nhiễu/proposal (Nhóm C), và khi đó không còn "cùng giả định nhiễu".
- Điều kiện B tốt hơn A cho mọi bộ lọc (vòng association 2 có dữ liệu) — như GĐ1.

### 7.2 Nhóm B — thuật toán dùng chung bộ lọc (dev, điều kiện A, **baseline = SORT nguyên bản**)

Cấu hình [groupB_detrac_dev.yaml](configs/phase2/experiments/groupB_detrac_dev.yaml): mỗi cặp so sánh chỉ đổi **một** yếu tố — sort-orig1 → sort-orig30: bộ đệm (max_age 1 → 30 = `track_buffer` của ByteTrack); sort-orig30 → sort-cv30: biểu diễn trạng thái + nhiễu (KF 7-D `[u,v,s,r]` → `KalmanFilterXYAH`); sort-cv30 → sort-ekf30: motion model (CV → EKF–CTRV, cùng xyah, cùng vòng đời SORT); sort-orig1 → sort-ekf1: chỉ đổi bộ lọc, giữ vòng đời nguyên bản; sort-ekf30 → bt-ekf: vòng đời/association (SORT → ByteTrack: 2 vòng ghép, unconfirmed, fuse_score). SORT-CV-XYAH là **đối chứng** để tách ảnh hưởng của biểu diễn trạng thái khỏi ảnh hưởng của EKF.

| tracker | ý nghĩa | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | preserved | id_continuous |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **sort-orig1** | **SORT nguyên bản** (KF 7-D, iou 0,3, max_age 1, min_hits 3) — baseline | 0.5731 | 0.5392 | 0.6180 | 0.6583 | 0.6031 | 1038 | 8918 | 36423 | 0.348 | 0.315 |
| sort-orig30 | SORT nguyên bản, max_age 30 | 0.5872 | 0.5380 | 0.6494 | 0.6947 | 0.6045 | 726 | 8766 | 36716 | 0.505 | 0.321 |
| sort-cv30 | SORT + `KalmanFilterXYAH` (bộ lọc ByteTrack-CV), max_age 30 — đối chứng | 0.5690 | 0.5117 | 0.6428 | 0.6831 | 0.5596 | 747 | 11278 | 39437 | 0.451 | 0.294 |
| sort-ekf30 | SORT + EKF–CTRV, max_age 30 | 0.5666 | 0.5117 | 0.6377 | 0.6777 | 0.5601 | 742 | 11236 | 39419 | 0.446 | 0.299 |
| sort-ekf1 | SORT + EKF–CTRV, max_age 1 | 0.5513 | 0.5128 | 0.6038 | 0.6376 | 0.5579 | 1062 | 11414 | 39187 | 0.337 | 0.299 |
| bt-ekf | ByteTrack + EKF–CTRV (tham chiếu Nhóm A) | 0.5925 | 0.5207 | 0.6838 | 0.7245 | 0.5719 | 472 | 12103 | 37448 | 0.489 | 0.310 |

Δ so với **sort-orig1** (trung bình theo video, CI 95 % bootstrap theo video, 5000 lần, seed 0; `paired_bootstrap.csv`):

| tracker | ΔHOTA | ΔIDF1 | ΔAssA | ΔDetA | ΔIDSW | Δ preserved (n = 184) |
|:--|:--|:--|:--|:--|:--|:--|
| sort-orig30 | +0.0178 [+0.0090, +0.0295] | +0.0419 [+0.0271, +0.0607] | +0.0349 [+0.0192, +0.0559] | −0.0007 [−0.0018, +0.0008] | −26.0 [−37.3, −17.6] | +0.158 [+0.092, +0.232] |
| sort-cv30 | −0.0028 [−0.0264, +0.0218] | +0.0306 [−0.0053, +0.0690] | +0.0226 [−0.0099, +0.0588] | −0.0291 [−0.0444, −0.0154] | −24.3 [−31.8, −16.9] | +0.103 [+0.055, +0.149] |
| sort-ekf30 | −0.0034 [−0.0276, +0.0221] | +0.0291 [−0.0080, +0.0697] | +0.0213 [−0.0124, +0.0593] | −0.0292 [−0.0446, −0.0155] | −24.7 [−32.5, −18.0] | +0.098 [+0.049, +0.146] |
| sort-ekf1 | −0.0256 [−0.0417, −0.0104] | −0.0213 [−0.0419, −0.0029] | −0.0223 [−0.0397, −0.0047] | −0.0288 [−0.0439, −0.0153] | +2.0 [−0.5, +4.8] | −0.011 [−0.040, +0.008] |
| bt-ekf | +0.0218 [−0.0077, +0.0534] | +0.0682 [+0.0217, +0.1177] | +0.0643 [+0.0190, +0.1144] | −0.0196 [−0.0347, −0.0057] | −47.2 [−63.9, −33.3] | +0.141 [+0.100, +0.181] |

Tỉ lệ giữ ID theo tầng che khuất (full n = 33 / partial n = 151): sort-orig1 0.000 / 0.424; sort-orig30 0.091 / 0.596; sort-cv30 0.091 / 0.530; sort-ekf30 0.091 / 0.523; sort-ekf1 0.000 / 0.411; bt-ekf 0.061 / 0.583.

Đọc kết quả theo từng cặp (dev, chưa phải test): (i) **bộ đệm** (orig1 → orig30): cải thiện rõ và có ý nghĩa (HOTA +0.018, IDSW −26, preserved +0.158, CI không chứa 0) — yếu tố lớn nhất trong SORT; (ii) **biểu diễn trạng thái** (orig30 → cv30): XYAH + nhiễu tỉ lệ theo h của ByteTrack **kém** KF 7-D nguyên bản về DetA/MOTA (FP tăng 8766 → 11278, FN tăng) trong vòng đời SORT — tức so sánh "SORT vs SORT-EKF" mà không có đối chứng này sẽ gán sai phần suy giảm cho EKF; (iii) **motion model** (cv30 → ekf30): CV → EKF–CTRV **không khác** (ΔHOTA −0.0006, mọi chỉ số trong CI) — cùng kết luận với Nhóm A; (iv) đổi bộ lọc mà giữ max_age 1 (ekf1) **kém** baseline có ý nghĩa (−0.026) vì mất ưu điểm KF 7-D và không có bộ đệm; (v) **vòng đời** (ekf30 → bt-ekf): ByteTrack tăng AssA/IDF1, IDSW giảm mạnh, đổi lại DetA/MOTA thấp hơn (nhiều FP do vòng ghép 2 và không có min_hits). Kết luận tạm: thứ tự ảnh hưởng **bộ đệm > vòng đời association > biểu diễn trạng thái ≫ motion model (CV vs CTRV)**; phải kiểm chứng lại trên 40 video test.

### 7.3 Chi phí tính toán (PC, GTX 1650, `profile_pipeline.py`, MVI_39031, 450 frame đo sau 50 warm-up, 960×540→640)

| tracker | inference ms (mean/p50/p95) | predict ms | assoc ms | total ms | fps end-to-end |
|:--|:--|--:|--:|--:|--:|
| bytetrack (CV) | 10.67 / 9.36 / 15.63 | 0.12 | 0.66 | 19.44 | 51.4 |
| bytetrack_ekf_ctrv | 12.06 / 11.43 / 16.31 | 0.20 | 0.83 | 22.01 | 45.4 |
| bytetrack_ukf_ctrv | 11.06 / 9.83 / 15.92 | 0.95 | 0.96 | 20.81 | 48.1 |
| bytetrack_pf_ctrv n300 | 11.46 / 10.48 / 16.06 | 0.82 | 1.40 | 21.66 | 46.2 |

Decode ~3 ms, preprocess ~1,5 ms, postprocess ~2 ms/frame; RSS ~1,16 GB; GPU ≤ 60 MB reserved. Bộ lọc chiếm < 5 % thời gian; detector + decode là nút thắt trên PC. **Không suy ra cho Nano.** File: `results/phase2/profile/*.json|csv`.

**Tốc độ tracker khi chạy từ cache** (KHÔNG có detector, CPU, 12 video dev = 16 919 frame, điều kiện A; `data/processed/trackers/DETRAC-all/p2-*/summary.csv`): cv 745 fps (1,34 ms/frame), ekf 697 (1,43), ukf 267 (3,75), pf100 ≈ 300 (3,3), pf300 ≈ 200 (5,0), pf500 ≈ 150 (6,7); sort-orig1 1011 (0,99), sort-orig30 1051 (0,95), sort-cv30 744 (1,34), sort-ekf30 771 (1,30), bt-ekf 601 (1,66). Hai con số này (tracker-từ-cache vs toàn pipeline ở bảng trên) **không so sánh với nhau**: pipeline đầy đủ bị chi phối bởi detector + decode.

## 8. Detector ứng viên (export/kiểm tra op) và huấn luyện — chuẩn bị, chưa train

- `export_detector.py` (ONNX opset 12, static 1×3×640×640, `nms=False`) + bảng op TensorRT 8.2 (`scripts/nano/trt82_ops.txt`), kết quả `results/phase2/export/*.json`:
  - yolov8n: 262 node, op ⊂ bảng TRT 8.2 → có thể build.
  - **yolo11n**: 353 node, op ⊂ bảng → **ứng viên duy nhất** (10,2 MB ONNX).
  - yolo26n: end2end, dùng op `Mod` ngoài bảng → loại (không plugin).
  - Kết luận cuối cùng chỉ khi build engine bằng `trtexec` trên Nano.
- `prepare_mixed_yolo_dataset.py` (5 lớp, chia theo video, mask vùng bỏ qua/crowd; dry-run DETRAC stride 5: 13 199 ảnh train / 3 228 val; nhãn train car 92 581, bus 5 262, truck 545; BDD chưa có), `configs/phase2/detector/mixed5.yaml` (giả định `van→car`, `others→truck` ghi rõ; hạn chế xung đột nhãn 2 bánh), `train_detector.py --dry-run`. **Chưa chạy train** (GTX 1650 4 GB; BDD train chưa có).

## 9. Lệnh chính xác

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe        # (Git Bash) hoặc conda activate khoaluan-mot

# Test (script thuần)
$P src/test_filter_config.py; $P src/test_time_units.py; $P src/test_det_cache.py
$P src/test_run_tracking_replay.py --n-frames 150          # cần ảnh DETRAC + GPU/CPU
$P src/test_trackeval_ext.py; $P src/test_bdd100k_adapter.py; $P src/test_pf_ctrv.py; $P src/test_sort_tracker.py; $P src/test_smoke_phase2.py
$P src/test_ekf_ctrv.py; $P src/test_ukf_ctrv.py; $P src/test_init_regression.py; $P src/test_tracker_lifecycle.py; $P src/test_occlusion_eval.py
$P src/verify_shared_kalman_regression.py --videos MVI_39031 MVI_20052 MVI_39931 --commit 35a560f   # pre-fix vs fixed tren cache that

# Chọn video pilot + cache + thí nghiệm Nhóm A (dev / test)
$P src/select_pilot_videos.py --n 12 --seed 0
$P src/det_cache.py --dataset detrac --part train --manifest results/phase2/manifests/detrac_dev.json --cache-id yolov8n-coco_conf0.05_all --skip-existing
$P src/det_cache.py --dataset detrac --part test  --manifest results/phase2/manifests/detrac_test.json --cache-id yolov8n-coco_conf0.05_all --skip-existing
$P src/run_experiment.py --exp configs/phase2/experiments/groupA_detrac_dev.yaml           # --stage track|eval|aggregate, --dry-run
$P src/run_experiment.py --exp configs/phase2/experiments/groupA_detrac_test.yaml          # SAU khi chốt tham số PF trên dev
$P src/run_experiment.py --exp configs/phase2/experiments/groupB_detrac_dev.yaml           # Nhom B (SORT)
$P src/aggregate_phase2.py --exp configs/phase2/experiments/groupA_detrac_dev.yaml
$P src/profile_pipeline.py --video MVI_39031 --part test --n-frames 500 --warmup 50 --tracker-yaml configs/bytetrack_ekf_ctrv.yaml
$P src/export_detector.py --weights models/candidates/yolo11n.pt --opset 12 --imgsz 640
$P src/prepare_mixed_yolo_dataset.py --dry-run; $P src/train_detector.py --dry-run          # khong train dai

# Chạy lẻ một tracker từ cache
$P src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name DETRAC-test --tracker-yaml configs/bytetrack_ekf_ctrv.yaml \
   --tracker-name p2-ekf --videos MVI_39031 --conf-min 0.25 --track-buffer-sec 1.2
$P src/run_trackeval.py --split-name DETRAC-test --tracker p2-ekf

# BDD100K khi có dữ liệu (giải nén vào data/raw/bdd100k/{labels/box_track_20,images/track}/{train,val})
$P src/check_bdd100k.py --part val                                  # schema + thống kê theo lớp + Δt + manifest final
$P src/check_bdd100k.py --part train --make-manifest --n-dev 40 --seed 0
$P -c "import sys; sys.path.insert(0,'src'); from mot_datasets.bdd100k_mot import *; ds=BDD100KMOTDataset(); df=ds.gt_dataframe('val'); [print(export_motchallenge(df,'val',t)) for t in ('vehicle','car','motorcycle','truck','bus','bicycle')]"
$P src/det_cache.py --dataset bdd100k --part val --manifest results/phase2/manifests/bdd100k_final_val.json --cache-id yolov8n-coco_conf0.05_all_bdd
$P src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all_bdd --split-name BDD-val --tracker-yaml configs/bytetrack_ekf_ctrv.yaml \
   --tracker-name p2-bdd-ekf --manifest results/phase2/manifests/bdd100k_final_val.json --classes 1 2 3 5 7 --track-buffer-sec 1.2 --elapsed-mode steps
$P -c "import sys; sys.path.insert(0,'src'); from pathlib import Path; import config; from mot_datasets.bdd100k_mot import split_tracker_output_by_class as s; print(s(config.PROCESSED_DIR/'trackers/BDD-val/p2-bdd-ekf', config.PROCESSED_DIR, 'val', ['car','motorcycle']))"
$P src/run_trackeval.py --split-name BDD-val --tracker p2-bdd-ekf            # tự chọn mot_crowd
$P src/occlusion_segments.py --parquet data/interim/bdd100k_val_annotations.parquet --fps 5 --out-tag bdd100k_val
$P src/occlusion_eval.py --split-name BDD-val --trackers p2-bdd-cv p2-bdd-ekf --tag bdd100k_val --fps 5 --context-s 1.2 --gaps --out-dir results/phase2/bdd_val
# Fixture (không cần dữ liệu):
$P src/mot_datasets/make_bdd100k_fixture.py; $P src/check_bdd100k.py --part val --data-root data/fixtures/bdd100k_mini --no-images
```

## 10. Chưa làm / điểm chặn / hạn chế

- **Chưa làm**: chạy Nhóm A/B trên 40 video test (`groupA_detrac_test.yaml`; cần cache 40 video ≈ 20 phút GPU); Nhóm C ablation (PF roughening / nhiễu tham số, `track_buffer`); `boot_stats.py` (bootstrap trên thống kê đủ của TrackEval — hiện `aggregate_phase2.py` bootstrap trên **trung bình theo video**, khác COMBINED_SEQ); vòng lặp ByteTrack/SORT không phụ thuộc torch cho Nano; `benchmark_detector.py`; train detector.
- **Dữ liệu**: BDD100K-MOT (labels box_track_20 + images track val/train-subset) chưa có trên máy → mọi phần BDD mới kiểm tra bằng fixture. Khi có: lệnh mục 8. Tham số nhiễu θ/ω cho 5 fps phải chọn lại trên dev BDD (mục 3.3), chưa có giá trị.
- **Thiết bị**: Nano chưa kết nối → chưa đo gì trên thiết bị (`scripts/nano/README.md`, `00_check_stack.sh`…).
- **Hạn chế diễn giải**: (i) PF ở N nhỏ suy biến dưới cùng giả định nhiễu — so sánh "PF vs EKF" phụ thuộc N và roughening; (ii) BDD `occluded` là cờ nhị phân; gap của track chỉ được gắn `gap_evidence` (`occluded_adjacent`/`unknown`), không suy diễn là che hoàn toàn; (iii) rider không đánh giá; (iv) GPU không bit-reproducible → kết quả GĐ1 chỉ tái lập ở mức cấu trúc track (100% (frame,id)) chứ không bit-exact; từ GĐ2 mọi tracker dùng **cùng** cache nên ghép cặp chính xác; (v) bootstrap trong `aggregate_phase2.py` là trên trung bình theo video.

## 11. Bàn giao mốc 1 (sửa tính đúng đắn pipeline + detection cache + thử nghiệm nhỏ trên PC)

Commit cục bộ trên `phase2-mixed-traffic` (không push): `697bccb` (mục 1), `e052dcd` (mục 2–3), `6bb2b6f` (mục 4), commit bàn giao (mục 5–6). File của người dùng chưa commit (`.gitignore`, `src/demo/*`, `live.bat`, `outputs/*`) **không** bị động vào.

### 11.1 File đã sửa / thêm và mục đích

| Nhóm | File | Mục đích |
|:--|:--|:--|
| `shared_kalman` | [src/ekf_ctrv.py](src/ekf_ctrv.py) | Tham số bộ lọc trên instance; late-binding 4 tham số khởi tạo (giữ ngữ nghĩa pre-fix cho script cũ); `explicit_params()` |
| | [src/ukf_ctrv.py](src/ukf_ctrv.py) | Chuyển `**kwargs` xuống EKF (cùng tham số) |
| | [src/tracker_ctrv.py](src/tracker_ctrv.py) | `CTRVSTrack.multi_predict` dùng `stracks[0].kalman_filter`; `CTRVByteTracker.get_kalmanfilter()` đọc `ctrv_*` từ YAML; không ghi đè biến toàn cục |
| | [src/test_filter_config.py](src/test_filter_config.py) | 9 test/32 phép: mặc định == cũ; qua `BYTETracker.update`; đơn == nhóm; 2 tracker cách ly; late-binding |
| | [src/verify_shared_kalman_regression.py](src/verify_shared_kalman_regression.py) | Chạy mã pre-fix (`git show 35a560f`) vs mã mới trên cache thật; ghi provenance + `diff_summary.csv` |
| | `results/phase2/shared_kalman/{baseline_provenance.json, diff_summary.csv}` | Bằng chứng (sha256 mã/weights/YAML/cache, phiên bản thư viện; bit-identical 9/9; override bị bỏ qua ở bản cũ) |
| Thời gian / nhiễu | [src/time_units.py](src/time_units.py) | Audit đơn vị & mô hình nhiễu 4 bộ lọc (mục A–D); `ElapsedStepsMixin` (`n_steps`), `steps_between`, `track_buffer_frames`, `convert_state_units` (S P Sᵀ) |
| | [src/test_time_units.py](src/test_time_units.py) | 8 test/30 phép: `k=1` bit-identical; k bước = k lần predict; frame rỗng ≠ frame bỏ qua; PF cùng quy ước; round-trip đổi đơn vị |
| Cache | [src/det_cache.py](src/det_cache.py), [src/test_det_cache.py](src/test_det_cache.py) | Cache conf 0,05/80 lớp/max_det 3000 + `meta.json`; lọc lúc load; chứng minh NMS-prefix; `--source gt` |
| | [src/run_tracking.py](src/run_tracking.py), [src/test_run_tracking_replay.py](src/test_run_tracking_replay.py) | Runner từ cache (registry, ElapsedSteps, timing, `run_meta.json`); capture-and-replay bit-identical; **trực tiếp vs cache cùng tiến trình** |
| BDD100K | [src/mot_datasets/{base,detrac,bdd100k_mot,make_bdd100k_fixture}.py](src/mot_datasets/) | Adapter fail-loud, giữ nguyên lớp/thuộc tính, Δt từ timestamp, `track_gaps`, xuất MOT theo lớp, tách output theo lớp; fixture (gap `unknown` + gap `occluded_adjacent`) |
| | [src/trackeval_ext.py](src/trackeval_ext.py), [src/test_trackeval_ext.py](src/test_trackeval_ext.py) | `MotChallengeCrowdIgnore` (ghép GT trước, ignore IoA > 0,5 sau); 3 trường hợp; đối chiếu trùng số adapter TrackEval BDD100K gốc |
| | [src/occlusion_eval.py](src/occlusion_eval.py), [src/occlusion_segments.py](src/occlusion_segments.py), [src/run_trackeval.py](src/run_trackeval.py), [src/check_bdd100k.py](src/check_bdd100k.py) | Gap events `has_full=False` + `gap_evidence`; `--videos`; benchmark prefix; `mot_crowd`; thống kê schema/lớp/Δt |
| Thử nghiệm | [src/pf_ctrv.py](src/pf_ctrv.py), [src/sort_tracker.py](src/sort_tracker.py) (+ test) | PF–CTRV bootstrap (cùng f/Q/R với EKF); SORT nguyên bản / SORT-CV-XYAH / SORT-EKF |
| | [src/run_experiment.py](src/run_experiment.py), [src/aggregate_phase2.py](src/aggregate_phase2.py), [src/select_pilot_videos.py](src/select_pilot_videos.py), [src/test_smoke_phase2.py](src/test_smoke_phase2.py) | Runner thí nghiệm (track/eval/aggregate), bootstrap ghép cặp theo video, `occlusion_strata.csv`, chọn dev phân tầng |
| | `configs/phase2/experiments/{groupA_detrac_dev,groupA_detrac_test,groupB_detrac_dev,smoke_bddfx}.yaml`, `configs/phase2/trackers/*.yaml` | Nhóm A (cv/ekf/ukf/pf100/300/500 × seed), Nhóm B baseline `sort-orig1`, cấu hình tracker tách khỏi GĐ1 |
| | `results/phase2/{manifests,groupA_detrac_dev,groupB_detrac_dev,profile,export,bdd100k}/` | Kết quả dev, tầng che khuất, profiling PC, kiểm tra op ONNX |
| Khác | [src/profile_pipeline.py](src/profile_pipeline.py), [src/export_detector.py](src/export_detector.py), [src/prepare_mixed_yolo_dataset.py](src/prepare_mixed_yolo_dataset.py), [src/train_detector.py](src/train_detector.py), `scripts/nano/*`, `requirements.txt` (+onnx, fastparquet, psutil) | Chuẩn bị detector/Nano (chưa train, chưa đo Nano) |

### 11.2 Lệnh đã chạy, kết quả test thực tế, lỗi còn tồn tại

Lệnh: mục 9 (test, cache, `run_experiment.py --exp groupA_detrac_dev.yaml` / `groupB_detrac_dev.yaml`, `aggregate_phase2.py`, `verify_shared_kalman_regression.py`, `profile_pipeline.py`). Kết quả lần chạy cuối (sau mọi sửa đổi, `khoaluan-mot`, 2026-09-15):

| Bộ test | Kết quả | Bộ test | Kết quả |
|:--|:--|:--|:--|
| `test_ekf_ctrv.py` | 14/14 | `test_det_cache.py` | 37/37 |
| `test_ukf_ctrv.py` | 12/12 | `test_run_tracking_replay.py` (GPU, 150 frame) | 7/7 |
| `test_init_regression.py` | 17/17 | `test_trackeval_ext.py` | 10/10 |
| `test_tracker_lifecycle.py` | 20/20 | `test_bdd100k_adapter.py` | 42/42 |
| `test_occlusion_eval.py` | 14/14 | `test_pf_ctrv.py` | 26/26 |
| `test_filter_config.py` | 32/32 | `test_sort_tracker.py` | 21/21 |
| `test_time_units.py` | 30/30 | `test_smoke_phase2.py` | 11/11 |

Lỗi / hạn chế còn tồn tại (không phải lỗi mã của dự án):
- **pyarrow bị Windows Smart App Control chặn** (`WinError 4551`, `VerifiedAndReputablePolicyState = 1`) → parquet dùng `fastparquet` (đã ghim trong `requirements.txt`). Đây là thiết lập máy, chưa thay đổi; người dùng tự quyết.
- **TrackEval 1.3.0**: `BDD100K` gốc raise `IndexError` trong `combine_classes_det_averaged` khi đánh giá một lớp có super-category → khi đối chiếu đã đặt `should_classes_combine = use_super_categories = False`. Không ảnh hưởng đường `MotChallengeCrowdIgnore`.
- **ID toàn cục** `BaseTrack._count` (ultralytics): hai tracker trong cùng tiến trình cho ID khác nhau khi chạy xen kẽ; các test/runner so sánh theo box hoặc chạy tuần tự (không sửa thư viện).
- GPU không bit-reproducible giữa hai lần forward → "trực tiếp vs cache" chỉ trùng ở mức (frame,id) 100 % + tọa độ ≤ 1,2e-4 px; capture-and-replay mới là bit-identical.

### 11.3 Kết luận có bằng chứng: `shared_kalman` và kết quả Giai đoạn 1

Cơ chế lỗi: `BYTETracker.update()` chỉ gọi `STrack.multi_predict()` (không gọi `predict()` từng track); `CTRVSTrack.multi_predict` cũ dùng thuộc tính lớp `shared_kalman` (tạo lúc `import tracker_ctrv`) thay vì bộ lọc gắn trên track → tham số đặt trên instance của tracker bị bỏ qua **ở bước dự đoán**.

| Đường chạy GĐ1 | Cách đặt tham số | Đi qua đường lỗi? | Kết luận |
|:--|:--|:--|:--|
| `baseline_track.py` (mọi kết quả runA/runB/testA/testB, buffer_sweep, lost_assoc, clamped, ft) | mặc định; tuỳ chọn `--ctrv-init-std-omega` ghi `config.CTRV_INIT_STD_OMEGA` ở dòng 190 **trước** `import tracker_ctrv` dòng 215, và `initiate()` pre-fix đọc `config` lúc gọi → `shared_kalman` và bộ lọc của tracker cùng giá trị | có | **không ảnh hưởng** |
| `smoke_three_models.py`, `benchmark_fps.py` (end-to-end), `demo/rerun_with_state_log.py`, `demo/live_view.py`, `demo/build_buffer90_demo.py`… | mặc định | có | **không ảnh hưởng** |
| Biến thể clamped (`EKFTrackerCTRVClamped`) | clamp thực hiện trong `multi_predict` của chính lớp; `ClampedSTrack.shared_kalman` là instance lớp đó | có | clamp có tác dụng → **không ảnh hưởng** |
| Script offline `tune_omega_noise.py`, `omega_clamp_experiment.py`, `omega_smoother.py`, `compare_extrapolation.py`, `diagnose_*`, `demo/omega_suppress.py`, `repro_init_covariance.py` | gọi thẳng `predict/update` trên instance | không | **không ảnh hưởng** |
| `configs/*.yaml` GĐ1 | không có khóa tham số bộ lọc | — | không có cấu hình nào bị "bỏ qua" |

Bằng chứng số: (a) `test_filter_config.py` — mặc định mới == `shared_kalman` cũ bit-identical (initiate/multi_predict/update; qua `BYTETracker.update`); (b) `verify_shared_kalman_regression.py` — mã `35a560f` vs mã mới trên cache thật 3 video: EKF/UKF/clamped mặc định **bit-identical 9/9**; override `std_theta = 0.05`: bản cũ **== mặc định** (lỗi tái hiện trên dữ liệu thật), bản mới **≠** (sửa có hiệu lực). **Verdict: không có kết quả GĐ1 nào cần chạy lại.** Regression duy nhất phát hiện là do chính bản sửa đầu (gắn tham số lúc `__init__`) làm `tune_omega_noise.py` (ghi đè `config` **sau** khi tạo bộ lọc) mất tác dụng — đã khắc phục bằng late-binding + test 7.

### 11.4 Kết quả chạy thử (dev UA-DETRAC, 12 video, cùng cache `yolov8n-coco_conf0.05_all`)

- Nhóm A (mục 7.1): HOTA cond A cv 0.5900 / ekf 0.5925 / ukf 0.5879 / pf100 0.4933 / pf300 0.5182 / pf500 0.5269; cond B 0.5989 / 0.5993 / 0.5990 / 0.4938 / 0.5213 / 0.5329. EKF ≈ CV ≈ UKF (CI chứa 0); PF cùng giả định nhiễu kém rõ (IDSW ×3–4), tăng N cải thiện đơn điệu; tầng che hoàn toàn (n = 33) ≤ 9 % với mọi bộ lọc.
- Nhóm B (mục 7.2, baseline SORT nguyên bản): bộ đệm +0.018 HOTA (có ý nghĩa); XYAH kém KF 7-D về DetA trong SORT (đối chứng SORT-CV-XYAH); CV → EKF trong SORT **không khác**; ByteTrack tăng IDF1 +0.068, IDSW −47 nhưng DetA thấp hơn.
- Timing: bộ lọc (predict) 0,13 / 0,33 / 2,24 / 1,3–3,7 ms/frame (CV/EKF/UKF/PF) khi chạy từ cache — CV 745 fps, EKF 697 fps, UKF 267 fps, PF300 ≈ 200 fps, SORT nguyên bản ≈ 1000 fps; **toàn pipeline** (có detector, mục 7.3) 45–51 fps trên GTX 1650 — hai con số không so sánh với nhau. Nano chưa đo.

### 11.5 Phần bị chặn, đầu vào cần bổ sung, lệnh tiếp theo

| Bị chặn | Cần | Lệnh khi có |
|:--|:--|:--|
| BDD100K-MOT thật (xe máy, che khuất, Δt 5 fps) | `labels/box_track_20/{train,val}` + `images/track/val` (+ ~40 video train dev) đặt vào `data/raw/bdd100k/`; **không** dùng bộ nhãn gộp lớp | `check_bdd100k.py --part val`; `--part train --make-manifest --n-dev 40 --seed 0`; `det_cache.py --dataset bdd100k …`; `run_tracking.py … --elapsed-mode steps`; chọn lại `std_theta/std_omega` cho 5 fps trên dev BDD (mục 3.3) |
| Đánh giá cuối UA-DETRAC | cache 40 video test (≈ 20 phút GPU) | `det_cache.py --part test --manifest results/phase2/manifests/detrac_test.json …`; `run_experiment.py --exp configs/phase2/experiments/groupA_detrac_test.yaml`; Nhóm B test (tạo `groupB_detrac_test.yaml` từ dev) |
| Nhóm C ablation | quyết định trục: PF roughening / nhiễu tham số; `track_buffer`; `std_theta`/`std_omega` | thêm YAML `configs/phase2/trackers/*`, `configs/phase2/experiments/groupC_*.yaml` |
| Bootstrap trên thống kê đủ (COMBINED) | `boot_stats.py` (hiện bootstrap trên trung bình theo video) | — |
| Jetson Nano | thiết bị kết nối, JetPack 4.6 (TRT 8.2); vòng lặp tracker không phụ thuộc torch | `scripts/nano/00_check_stack.sh`, `01_build_engine.sh` (yolo11n ONNX), `02_benchmark_engine.sh` |
| Detector trộn 5 lớp | BDD train; GPU thời gian dài (không chạy trên GTX 1650 4 GB trong phạm vi này) | `prepare_mixed_yolo_dataset.py`; `train_detector.py` (không `--dry-run`) |
| parquet qua pyarrow | thiết lập Smart App Control (người dùng quyết) | không cần: `fastparquet` đủ dùng |
