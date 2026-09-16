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

> ⚠️ **Số trong mục này là bản TRƯỚC khi sửa lỗi vòng đời SORT** (baseline xuất track sớm hơn upstream một
> frame — review độc lập, mục 18.1). Giữ lại để đối chiếu lịch sử. **Bảng dùng cho luận văn**: mục 18.2
> (kết quả `expB_detrac_dev_fix1` / `expB_detrac_test_fix1`).

Cấu hình [groupB_detrac_dev.yaml](configs/phase2/experiments/groupB_detrac_dev.yaml): mỗi cặp so sánh chỉ đổi **một** yếu tố — sort-orig1 → sort-orig30: bộ đệm (max_age 1 → 30 = `track_buffer` của ByteTrack); sort-orig30 → sort-cv30: **gói biểu diễn trạng thái + thiết lập bộ lọc** (KF 7-D `[u,v,s,r]` với Q/R/P₀ của SORT → `KalmanFilterXYAH` với Q/R/P₀ tỉ lệ theo chiều cao của ByteTrack) — **không phải** chỉ đổi biểu diễn; sort-cv30 → sort-ekf30: motion model (CV → EKF–CTRV, cùng xyah, cùng vòng đời SORT); sort-orig1 → sort-ekf1: chỉ đổi bộ lọc, giữ vòng đời nguyên bản; sort-ekf30 → bt-ekf: vòng đời/association (SORT → ByteTrack: 2 vòng ghép, unconfirmed, fuse_score). SORT-CV-XYAH là **đối chứng** để tách ảnh hưởng của biểu diễn trạng thái khỏi ảnh hưởng của EKF.

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

Đọc kết quả theo từng cặp (dev, chưa phải test): (i) **bộ đệm** (orig1 → orig30): cải thiện rõ và có ý nghĩa (HOTA +0.018, IDSW −26, preserved +0.158, CI không chứa 0) — yếu tố lớn nhất trong SORT; (ii) **gói biểu diễn + thiết lập bộ lọc** (orig30 → cv30; đổi đồng thời state 7-D→XYAH **và** Q/R/P₀ — không tách được riêng phần nào): XYAH + nhiễu tỉ lệ theo h của ByteTrack **kém** KF 7-D nguyên bản về DetA/MOTA (FP tăng 8766 → 11278, FN tăng) trong vòng đời SORT — tức so sánh "SORT vs SORT-EKF" mà không có đối chứng này sẽ gán sai phần suy giảm cho EKF; (iii) **motion model** (cv30 → ekf30): CV → EKF–CTRV **không khác** (ΔHOTA −0.0006, mọi chỉ số trong CI) — cùng kết luận với Nhóm A; (iv) đổi bộ lọc mà giữ max_age 1 (ekf1) **kém** baseline có ý nghĩa (−0.026) vì mất ưu điểm KF 7-D và không có bộ đệm; (v) **vòng đời** (ekf30 → bt-ekf): ByteTrack tăng AssA/IDF1, IDSW giảm mạnh, đổi lại DetA/MOTA thấp hơn (nhiều FP do vòng ghép 2 và không có min_hits). Kết luận tạm: thứ tự ảnh hưởng **bộ đệm > vòng đời association > biểu diễn trạng thái ≫ motion model (CV vs CTRV)**; phải kiểm chứng lại trên 40 video test.

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
- Nhóm B (mục 7.2, baseline SORT nguyên bản; **số của mốc 2 trước khi sửa lỗi vòng đời SORT** — bản sau khi sửa ở mục 18): bộ đệm +0.018 HOTA (có ý nghĩa); gói XYAH + Q/R/P₀ của ByteTrack kém KF 7-D + Q/R/P₀ của SORT về DetA trong vòng đời SORT (đối chứng SORT-CV-XYAH); CV → EKF trong SORT **không khác**; ByteTrack tăng IDF1 +0.068, IDSW −47 nhưng DetA thấp hơn.
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

---

# MỐC 2 — Nguyên nhân mất ID, Nhóm C, khóa cấu hình, đánh giá trên tập test

*(Mục 1–11 ở trên là bàn giao mốc 1, giữ nguyên làm nguồn đối chiếu. Mọi số liệu dưới đây dẫn tới file kết quả tương ứng.)*

## 12. Ba kiểm tra tính đúng đắn còn hở (mốc 2, mục 1)

| Yêu cầu | Việc đã làm | Bằng chứng |
|:--|:--|:--|
| `multi_predict` dùng `stracks[0].kalman_filter`: danh sách luôn thuộc cùng tracker, cấu hình tương thích | [tracker_ctrv.py](src/tracker_ctrv.py): `CTRVSTrack.multi_predict`, `PFSTrack.multi_predict`, `CVSTrack.multi_predict` **kiểm tra mọi track trong pool dùng cùng instance bộ lọc**, khác → `AssertionError` (fail-loud, O(n)); đếm số lần gọi `n_multi_predict_calls` | `test_filter_config.py` test 10: pool trộn 2 tracker → raise; pool một tracker → không raise; guard chạy ở cả 22 lần gọi của 12 frame × 2 tracker |
| Late-binding không rò rỉ cấu hình giữa hai tracker | test 11: tracker A có `ctrv_init_std_omega` tường minh (0,33), B và UKF mặc định; đổi `config.CTRV_INIT_STD_OMEGA` giữa chừng → chỉ B/UKF đổi, A giữ 0,33²; `explicit_params()` phân biệt; `shared_kalman` không đổi | `test_filter_config.py` 44/44 |
| Timestep đi từ **timestamp** → dự đoán → vòng đời trên **đường chạy đầy đủ** (CV/EKF/UKF/PF) | **[test_timestep_integration.py](src/test_timestep_integration.py)** (mới): cache tổng hợp 20 frame có (a) timestamp **nhảy 3 bước** giữa f8–f9, (b) f12 không đọc được, (c) f15 rỗng, (d) một xe mất dấu quá `track_buffer`; chạy qua `run_tracking.track_video` | 32/32: `n_steps` đúng từng frame ({9:3, 13:2}); `frame_id` cuối = 22 (số bước) ≠ 19 (số frame); **box dự đoán tại f9 == predict 3 lần thủ công** (CV/EKF/UKF ≤ 1e-9, PF bit-identical cùng seed); track LOST bị xóa tại f10 ở chế độ `steps` nhưng f13 ở `skip`; frame rỗng → LOST rồi khớp lại cùng ID |
| Kiểm chứng tương đương phải so **liên kết ID xuyên suốt sequence** | **[track_compare.py](src/track_compare.py)** `id_bijection`: ghép dòng theo (frame, box) rồi kiểm tra tập cặp (id_a, id_b) là **song ánh trên toàn sequence** (không ánh xạ lại từng frame) | `test_run_tracking_replay.py` 10/10 (direct vs cache: song ánh + identity, GPU); `verify_shared_kalman_regression.py` thêm cột `id_bijection` (9/9); `test_filter_config.py` test 10 (xen kẽ 2 tracker: ID **khác** nhưng song ánh 1-1 cố định); [check_identical_runs.py](src/check_identical_runs.py) — xem 12.1 |

### 12.1 "288/288" nghĩa là gì, và "bit-identical" đo ở mức nào

**Đơn vị đếm**: một đơn vị = **một cặp (lần chạy tracker, video)**. Ở đây 288 = **24 lần chạy**
(12 cấu hình tracker — cv/ekf/ukf + pf100/pf300/pf500 × 3 seed — × **2 điều kiện** conf) × **12 video dev**.

**Hai mức so sánh, không dùng lẫn nhau** ([check_identical_runs.py](src/check_identical_runs.py) docstring):

| Mức | Cách đo | Kết quả expA (có `assoc_log`) vs mốc 1 |
|:--|:--|:--|
| `bit_identical` | so sánh **BYTE** của file `data/<video>.txt` (MOTChallenge 7 cột) — **không** chuẩn hóa, không sắp xếp lại, không đổi tên ID | **288/288** |
| `id_bijection` | đọc file thành bảng, ghép dòng theo **(frame, box)** với dung sai `tol` (ở đây **0,0** = trùng khít), rồi kiểm tra tập cặp (id_a, id_b) là **song ánh trên toàn sequence** — chấp nhận đánh số ID khác nhưng **không** chấp nhận track bị cắt/gộp khác đi | **288/288** |
| `id_identity` | song ánh đó chính là ánh xạ đồng nhất (ID trùng tuyệt đối) | **288/288** |

Bit-identical ⇒ song ánh; chiều ngược lại **không** đúng (ví dụ: `test_filter_config.py` test 10 — hai tracker
chạy xen kẽ cho ID khác nhau nên **không** bit-identical, nhưng song ánh 1-1 cố định vẫn thành lập).
Bằng chứng: [identical_vs_moc1.csv](results/phase2/expA_detrac_dev/identical_vs_moc1.csv),
log [handoff/logs/check_identical_expA.log](handoff/logs/check_identical_expA.log).

Bằng chứng đối chiếu GĐ1 (mục 11.3) **giữ nguyên**; ba kiểm tra trên không phát hiện vi phạm nào mới → **không mở lại việc chạy GĐ1**.

## 13. Nguyên nhân mất ID trên 12 video dev (mốc 2, mục 2)

### 13.1 Tập sự kiện và định nghĩa (mẫu số rõ ràng)

`failure_analysis.py --check-events` ([events_check.csv](results/phase2/failure_analysis/expA_detrac_dev/events_check.csv)): 717 đoạn partial + 245 đoạn full = 962 đoạn thô → **717 sự kiện** sau khi gộp đoạn chồng lấn/kề nhau của cùng (video, track) (không đếm lặp; 0 sự kiện chồng frame trong cùng track) → **184 sự kiện đủ điều kiện** (GT có mặt cả trước và sau, cửa sổ 30 frame = 1,2 s) = **33 full + 151 partial**; loại 369 `gt_missing_before`, 164 `gt_missing_after`. Số này **khớp mốc 1**.

| Chỉ số | Định nghĩa | Mẫu số |
|:--|:--|:--|
| `preserved` (= `id_match`) | ID tracker ghép GT ở frame **cuối trước** sự kiện == ID ở frame **đầu sau** sự kiện | 184 sự kiện đủ điều kiện |
| `switched` | hai ID khác nhau | 184 |
| `lost_after` | có ID trước, **không** ghép lại được trong cửa sổ sau (GT vẫn còn) | 184 |
| `no_match_before` | tracker chưa từng ghép GT trong cửa sổ trước → không đánh giá được phần "giữ" | 184 (33 sự kiện với mọi tracker) |
| `id_continuous` | `preserved` **và** mọi frame có GT trong sự kiện đều được ghép đúng ID đó | 184 |
| `has_full` (tầng "full") | sự kiện **có ít nhất một đoạn** với tỉ lệ che khuất GT **≥ 0,90** (`occlusion_eval.build_events`) — **không** có nghĩa mọi frame bị che 100 %; tầng "partial" = sự kiện chỉ có đoạn 0,10 ≤ ratio < 0,90 | 33 (full) + 151 (partial) |
| `wrong_refind_rate` | trong các lần **nối lại track** (refind), tỉ lệ det được ghép thuộc **GT khác** GT mà track bám trước khi mất | số refind xác định được cả hai GT |

Kết quả ([outcome_counts.csv](results/phase2/failure_analysis/expA_detrac_dev/outcome_counts.csv)), điều kiện c25 (conf > 0,25) và c10 (conf > 0,10):

| tracker | preserved | switched | lost_after | no_match_before | id_continuous | preserved (c10) |
|:--|--:|--:|--:|--:|--:|--:|
| cv | 93/184 = 0,505 | 44 | 14 | 33 | 57 (0,310) | 104 (0,565) |
| ekf | 90 (0,489) | 46 | 15 | 33 | 57 (0,310) | 102 (0,554) |
| ukf | 91 (0,495) | 45 | 15 | 33 | 57 (0,310) | 103 (0,560) |
| pf500 | 64 (0,348) | 72 | 14 | 34 | 46 (0,250) | 72 (0,391) |

### 13.2 Bảng nguyên nhân thất bại (có bằng chứng, nhiều nhãn cho phép)

Nguồn: [assoc_log.py](src/assoc_log.py) ghi **quyết định ghép từng frame** (pool sau dự đoán: id/state/`n_lost`/box dự đoán; det: idx/tier/conf/box/được track nào lấy/ở vòng nào/lý do loại; sự kiện lost/refind/new/removed) — bật bằng `assoc_log: true`, **không đổi hành vi** (kiểm chứng bit-identical trong `test_timestep_integration.py` và `check_identical_runs.py`). [failure_analysis.py](src/failure_analysis.py) đối chiếu log với GT (GT **chỉ** dùng để phân tích, không tham gia tracking).

Thất bại (`switched` + `lost_after`), điều kiện c25 ([cause_summary.csv](results/phase2/failure_analysis/expA_detrac_dev/cause_summary.csv)):

| tracker | tầng | prediction_drift | buffer_expired | score_fusion_reject | detector_miss | competition | hijacked_during | khác | tổng |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| cv | full (33 sự kiện) | **11** | 5 | 5 | 0 | 1 | 0 | 0 | 22 |
| cv | partial (151) | 9 | 7 | 7 | **9** | 2 | 1 | 1 | 36 |
| ekf | full | **13** | 6 | 2 | 0 | 2 | 0 | 0 | 23 |
| ekf | partial | 10 | 9 | 6 | **10** | 2 | 0 | 1 | 38 |
| ukf | full | **14** | 6 | 2 | 0 | 1 | 0 | 0 | 23 |
| ukf | partial | 10 | 8 | 6 | 10 | 2 | 0 | 1 | 37 |
| pf500 | full | **13** | 9 | 1 | 0 | 0 | 0 | 0 | 23 |
| pf500 | partial | **24** | 18 | 5 | 9 | 4 | 1 | 2 | 63 |

Hiện tượng đo được (tách khỏi suy luận):
- **`prediction_drift`** (nguyên nhân số 1 ở **tầng che hoàn toàn**): tại frame detector thấy lại đối tượng, track cũ **vẫn còn sống** nhưng IoU(box dự đoán, det) trung bình **0,06** (trung vị 0,05; max 0,19), lệch tâm trung vị **80 px** (trung bình 102 px) sau trung vị **25 bước** mất dấu (n = 43 ca cv + ekf). Vì `match_thresh` = 0,8 trên cost = 1 − IoU·conf, mọi cặp có IoU < 0,2 **không thể** được ghép, bất kể conf.
- **`buffer_expired`**: track bị xóa (quá `track_buffer` = 30 bước) trước khi detector thấy lại.
- **`score_fusion_reject`**: IoU ≥ 0,2 nhưng `1 − IoU·conf > 0,8` — cặp hợp lệ bị `fuse_score` đẩy qua ngưỡng.
- **`detector_miss`** chỉ xuất hiện ở tầng **partial** (9–10 ca) và **0 ca ở tầng full**: trong đó **6/9–10 ca detector vẫn thấy vật ở mức conf 0,05–0,10** (bị ngưỡng `track_low_thresh` loại), 3–4 ca detector không thấy gì.
- Nguồn của ID mới sau khi `switched`: **86 %** là **track mới sinh từ detection** (cv 38/44, ekf 41/46), không phải track khác có sẵn.

**Tách ba câu hỏi khác nhau** (không được trộn — [cause_origin_table.py](src/cause_origin_table.py)):

| Câu hỏi | Cột | Ý nghĩa |
|:--|:--|:--|
| **Nguyên nhân**: vì sao track **cũ** không được ghép lại? | `primary_cause` / `causes` | `prediction_drift`, `buffer_expired`, `score_fusion_reject`, `competition`, `detector_miss`, `hijacked_during`, `unknown` |
| **Hậu quả**: ID mới của GT **đến từ đâu**? | `id_after_origin` | `new_track_from_det` (detection sinh track mới) / `existing_other_track` / trống (`lost_after` — không có ID sau) |
| **Khả năng đánh giá**: sự kiện có chấm được không? | `outcome` | `no_match_before` = không đánh giá được phần "giữ ID" |

"ID mới sinh từ detection" **không phải một nguyên nhân** — nó là mô tả hậu quả; nguyên nhân nằm ở phía
track cũ. Bảng chéo hai trục (c25, [cause_by_origin_c25.csv](results/phase2/failure_analysis/expA_detrac_dev/cause_by_origin_c25.csv)), tracker `ekf`:

| nguyên nhân (track cũ không ghép được) | ID sau = track mới từ det | = track khác có sẵn | không có ID sau (`lost_after`) |
|:--|--:|--:|--:|
| prediction_drift | 20 | 3 | 0 |
| buffer_expired | 13 | 1 | 1 |
| score_fusion_reject | 4 | 0 | 4 |
| competition | 3 | 1 | 0 |
| detector_miss | 0 | 0 | 10 |
| matched_but_eval_mismatch | 1 | 0 | 0 |
| **unknown** | 0 | 0 | 0 |

**Cột log là bằng chứng cho từng nhãn** ([cause_evidence_c25.csv](results/phase2/failure_analysis/expA_detrac_dev/cause_evidence_c25.csv); điều kiện gán nhãn ghi trong docstring `failure_analysis.analyze_event`):

| nhãn | cột số liệu (trong `event_causes_*.csv`) | điều kiện gán |
|:--|:--|:--|
| `prediction_drift` | `state_at_fdet`, `n_lost_at_fdet`, `iou_pred_det`, `center_offset_px`, `det_conf` | track còn sống tại frame detector thấy lại GT, nhưng IoU(dự đoán, det) < 0,2 |
| `buffer_expired` | `state_at_fdet = removed`, `removed_frame`, `first_det_frame` | track bị xóa trước frame detector thấy lại GT |
| `score_fusion_reject` | `iou_pred_det`, `det_conf`, `fused_cost` | IoU ≥ 0,2 nhưng 1 − IoU·conf > `match_thresh` |
| `competition` | `det_matched_track`, `competitor_iou`, `det_stage` | cặp hợp lệ nhưng det bị track khác lấy |
| `detector_miss` | `best_iou_high_after`, `best_iou_low_after`, `best_iou_cache_after`, `best_cache_conf`, `detector_miss_sub` | không detection tier-high nào IoU ≥ 0,5 với GT trong cửa sổ sau |
| `hijacked_during` | `hijack_frame`, `hijack_other_gt` | track cũ (đang Tracked) ghép det thuộc GT khác ngay trong sự kiện |
| `unknown` | — | không nhãn nào đủ bằng chứng (**giữ lại**; c25: 0 ca cho cv/ekf/ukf, 0 ca cho pf500 — pf500 có 1 ca ở c10) |

Với mỗi sự kiện, `representative_cases_*.md` in kèm dòng "truy log" chỉ chính xác bộ lọc cần dùng trên
`event_window_log.csv` (theo `video`, `gt_track`, `event_id`) để xem lại từng frame: box GT, box dự đoán của
track, các detection chồng lên GT (IoU ≥ 0,3) kèm tier/conf/được track nào lấy/ở vòng nào.
- Ghép nhầm khi nối lại ([refind_quality.csv](results/phase2/failure_analysis/expA_detrac_dev/refind_quality.csv), c25): cv 3,52 % (52/1478), ekf 2,20 % (34/1545), ukf 2,00 %, pf500 7,43 %.

Ca tiêu biểu (≥ 2 thất bại cho mỗi nguyên nhân + ca thành công, dẫn video/track/frame/box/IoU/quyết định để truy ngược [event_window_log.csv](results/phase2/failure_analysis/expA_detrac_dev/event_window_log.csv)): [representative_cases_c25_ekf.md](results/phase2/failure_analysis/expA_detrac_dev/representative_cases_c25_ekf.md), `..._cv.md`, `..._ukf.md`, `..._pf500.md` (sinh bởi [representative_cases.py](src/representative_cases.py)).

## 14. Nhóm C — ba trục, mỗi trục đổi một thành phần (mốc 2, mục 3)

Đăng ký trước: [PREREG_moc2.md](configs/phase2/PREREG_moc2.md) (ghi **trước** khi chạy: giá trị khảo sát, ngân sách tìm kiếm, tiêu chí chọn, giả thuyết). Ma trận đầy đủ: [ablation_matrix.csv](results/phase2/ablation_matrix.csv) (39 dòng: thành phần cố định / thành phần thay đổi / chỉ số / CI / preserved theo tầng / ghép nhầm).

**Tiêu chí (đăng ký trước)**: HOTA ở c25 trên dev, chỉ coi là cải thiện khi CI 95 % bootstrap **COMBINED** (lấy mẫu lại theo video, 5000 lần, seed 0, tái tổ hợp đúng công thức TrackEval — [boot_stats.py](src/boot_stats.py)) **không chứa 0**.

### 14.1 C1 — thời gian giữ track mất dấu ([expC1_buffer_dev.yaml](configs/phase2/experiments/expC1_buffer_dev.yaml))

`track_buffer_sec` ∈ {0,4; 0,8; 1,2; 2,0; 3,0; 4,0} s (= 10/20/30/50/75/100 bước @ 25 fps) × {CV, EKF}; mọi thứ khác cố định. Quy tắc xóa trong cài đặt: track LOST bị xóa khi `frame_id − end_frame > track_buffer`.

| buffer | HOTA (CV) | ΔHOTA vs 1,2 s (CV) | IDSW | preserved full/partial | ghép nhầm | HOTA (EKF) | ΔHOTA vs 1,2 s (EKF) | ghép nhầm |
|:--|--:|:--|--:|:--|--:|--:|:--|--:|
| 0,4 s | 0.5891 | −0.0009 [−0.0019, −0.0003] | 483 | 0,061 / 0,576 | 2,49 % | 0.5887 | −0.0038 [−0.0070, −0.0001] | 1,69 % |
| 0,8 s | 0.5900 | +0.0000 [−0.0004, +0.0005] | 475 | 0,091 / 0,596 | 3,09 % | 0.5896 | −0.0029 [−0.0060, +0.0004] | 1,89 % |
| **1,2 s** | 0.5900 | — | 489 | 0,091 / 0,596 | 3,52 % | 0.5925 | — | 2,20 % |
| 2,0 s | 0.5901 | +0.0001 [−0.0002, +0.0004] | 500 | 0,091 / 0,596 | 4,57 % | 0.5929 | +0.0004 [−0.0003, +0.0010] | 2,61 % |
| 3,0 s | 0.5903 | +0.0003 [−0.0002, +0.0006] | 508 | 0,091 / 0,596 | 4,87 % | 0.5931 | +0.0005 [−0.0003, +0.0013] | 2,79 % |
| 4,0 s | 0.5898 | −0.0002 [−0.0014, +0.0006] | 513 | 0,091 / 0,596 | 5,31 % | 0.5930 | +0.0005 [−0.0004, +0.0013] | 3,04 % |

**Kết luận C1**: từ 0,8 s trở lên, HOTA **bão hòa** (mọi CI chứa 0) và tỉ lệ giữ ID qua che khuất **không tăng thêm một sự kiện nào** (0,091 / 0,596 giữ nguyên từ 0,8 s đến 4,0 s với CV); ngược lại **ghép nhầm tăng đơn điệu** (CV 2,5 % → 5,3 %; EKF 1,7 % → 3,0 %), FP và IDSW tăng nhẹ. Buffer 0,4 s kém rõ rệt ở cả hai bộ lọc. → **giữ 1,2 s**; nới thời gian giữ track **không** phải cách cải thiện phục hồi sau che khuất trên UA-DETRAC. Đây là bổ sung định lượng (thêm chiều **ghép nhầm**) cho kết luận "`track_buffer` là trần cứng" của GĐ1.

### 14.2 C3 — tham số nhiễu ([expC3_noise_dev.yaml](configs/phase2/experiments/expC3_noise_dev.yaml))

21 cấu hình (EKF 4 × 3 cho `std_theta` × `std_omega`; CV 3 × 3 cho `std_weight_position` × `std_weight_velocity`) công bố trước; detector/association/buffer cố định. **Không cấu hình nào cải thiện HOTA có ý nghĩa**: EKF tốt nhất 0.5931 (`std_theta` 0,01–0,02 với `std_omega` 0,01) so với 0.5925 hiện tại — ΔHOTA +0.0006 [−0.0007, +0.0013]; cấu hình xấu nhất `std_theta` 0,05 / `std_omega` 0,01 (−0.0023). CV: mọi cấu hình ≤ 0, xấu nhất 1/10 và 1/320 (−0.0094 [−0.0197, −0.0019]).

#### "Hòa nhau" nghĩa là gì — đo ở ba mức ([verify_c3_effect.py](src/verify_c3_effect.py) → [c3_effect_check.csv](results/phase2/expC3_noise_dev/c3_effect_check.csv))

"Hòa" ở đây là **không phân biệt được về mặt thống kê**, **không** phải "kết quả giống nhau". Cụ thể, so với
cấu hình tham chiếu của cùng bộ lọc:

(Nói "hòa nhau" cho **toàn bộ 21 cấu hình** là sai: đúng phát biểu là **không cấu hình nào cải thiện HOTA có
ý nghĩa**, trong khi **một số cấu hình CV kém hơn có ý nghĩa** — ví dụ `std_weight_position` 1/10 với
`std_weight_velocity` 1/320: ΔHOTA −0.0094 [−0.0197, −0.0019].)

| Mức so sánh | EKF (11 cấu hình khác tham chiếu) | CV (6 cấu hình có tham số hiệu dụng khác) |
|:--|:--|:--|
| HOTA **làm tròn 4 chữ số** | 0.5925–0.5931 (chênh ≤ 0.0006) | 0.5806–0.5900 |
| HOTA **đầy đủ độ chính xác** | **khác** ở chữ số thứ 4–5 (vd 0.592521195986 vs 0.593053988403) | **khác** ở chữ số thứ 3–4 |
| **Toàn bộ đầu ra tracking** | **0/12 video bit-identical**; song ánh ID **không** thành lập; số track 1841 → 1840–1847; **99,2–99,7 %** số dòng có box trùng trong 1 px, 99,6–99,9 % trong 5 px | **0/12 bit-identical**; số track 1852 → 1836–2008; chỉ **61–83 %** số dòng trùng trong 1 px (88–98 % trong 5 px) |

**Override có thực sự đi vào bộ lọc không** — có, kiểm bằng hai cách: (a) tạo tracker từ **chính cấu hình đã
chạy** rồi đọc thuộc tính thật của bộ lọc (`_std_theta/_std_omega`, `_std_weight_position/_std_weight_velocity`)
— khớp giá trị trong YAML; (b) chạy trên cache thật (`MVI_20052`, 400 frame) và ghi **quỹ đạo covariance** của
pool: `P[θ,θ]` trung bình đổi rõ theo `std_omega` (2,456 → 2,833 → 3,880 khi ω = 0,0025 / 0,005 / 0,01), và
`covariance_giong_het = False` với mọi cấu hình có tham số hiệu dụng khác.

→ Kết luận đúng mức: **tham số đã có hiệu lực và trạng thái/covariance thay đổi rõ rệt**, nhưng với **nhánh
EKF** phần lớn quyết định ghép không đổi (dự đoán chỉ dịch dưới 1 px ở ~99 % số dòng) nên chênh lệch nằm trong
khoảng nhiễu lấy mẫu (CI chứa 0). **Với nhánh CV thì khác**: chỉ 61–83 % số dòng lệch dưới 1 px và hai cấu
hình lệch tỉ số nhiều nhất **kém hơn có ý nghĩa**. Tỉ lệ "~99 % dưới 1 px" **chỉ áp dụng cho EKF**, không phải
cho cả lưới. Hai cấu hình CV chỉ **trùng tham số hiệu dụng** mới cho đầu ra
bit-identical (xem dưới).

**Phát hiện phụ (kiểm chứng bằng số)**: `KalmanFilterXYAH` của ultralytics **bất biến** khi nhân **cả hai** trọng số nhiễu với cùng hệ số, vì nhiễu **đo** cũng dùng `_std_weight_position`: `cv-p40-v320` và `cv-p10-v80` **bit-identical** với `cv-p20-v160` (12/12 video, `check_identical_runs.py`) → lưới 3 × 3 thực chất chỉ có **5 cấu hình phân biệt**; chỉ **tỉ số** vị trí/vận tốc là tham số thật. Phải tính đến điều này khi thiết kế tuning cho BDD100K.

→ **giữ tham số mặc định**. Đây là **tuning tham số**, không phải đổi mô hình nhiễu (dạng Q/R giữ nguyên).

### 14.3 C2 — nối lại track mất dấu ([expC2_recover_dev.yaml](configs/phase2/experiments/expC2_recover_dev.yaml))

Chọn **sau** mục 13 và ghi vào PREREG **trước** khi chạy: nguyên nhân nổi bật nhất thuộc cơ chế ghép là `prediction_drift` (cv 20, ekf 23 ca) → thay đổi nhỏ: **vòng ghép thứ 3** chỉ giữa track **LOST** chưa ghép và det high còn dư, đặt sau vòng 1 và trước khi sinh track mới, **gate Mahalanobis vị trí theo đúng hiệp phương sai của chính bộ lọc** (≤ 5,991 = χ²(2 dof, 95 %)), xếp hạng theo khoảng cách tâm chuẩn hóa ([lost_recovery.py](src/lost_recovery.py); mặc định **tắt** → bit-identical, `test_lost_recovery.py` 11/11). **Đối chứng**: cùng vòng 3 nhưng gate IoU 0,05 — để tách "có thêm một vòng ghép" khỏi "dùng thông tin bất định của bộ lọc". Áp dụng **giống hệt** cho CV và EKF; giữ cấu hình gốc của từng bộ lọc làm mốc.

| cấu hình | HOTA | ΔHOTA vs gốc | IDSW | preserved (Δ vs cv gốc) | preserved full / partial | số refind | ghép nhầm |
|:--|--:|:--|--:|:--|:--|--:|--:|
| cv (gốc) | 0.5900 | — | 489 | 0,505 | 0,091 / 0,596 | 2147 | 3,52 % |
| cv + maha | 0.5919 | +0.0019 [−0.0015, +0.0044] | 465 | 0,554 (**+0,049 [+0,011, +0,080]**) | **0,152** / 0,642 | 3536 | **9,28 %** |
| cv + iou (đối chứng) | 0.5886 | −0.0014 [−0.0089, +0.0029] | 399 | 0,565 (+0,060 [+0,036, +0,084]) | 0,091 / 0,669 | 3867 | 10,26 % |
| ekf (gốc) | 0.5925 | — | 472 | 0,489 | 0,061 / 0,583 | 2196 | 2,20 % |
| ekf + maha | 0.5901 | −0.0024 [−0.0055, +0.0011] | 470 | 0,505 | **0,121** / 0,589 | 3679 | 10,54 % |
| ekf + iou (đối chứng) | 0.5865 | −0.0060 [−0.0114, +0.0020] | 418 | 0,538 | 0,091 / 0,636 | 3880 | 11,52 % |

**Đọc kết quả — hai chỉ số không thống nhất, phải nói rõ**:
- Trên **chỉ số sự kiện**: vòng ghép thứ 3 **cải thiện có ý nghĩa** khả năng giữ ID qua che khuất (CV +0,049, CI không chứa 0); ở **tầng che hoàn toàn** tỉ lệ giữ ID **gần gấp đôi** (CV 0,091 → 0,152; EKF 0,061 → 0,121). Cơ chế đúng như giả thuyết: `score_fusion_reject` giảm (cv 12 → 4 ca, ekf 8 → 1 ca).
- Trên **chỉ số toàn cục**: HOTA **không cải thiện** (mọi CI chứa 0; EKF hơi âm), vì số lần nối lại tăng ~65 % nhưng **tỉ lệ ghép nhầm tăng 3–5 lần** (cv 3,5 % → 9,3 %; ekf 2,2 % → 10,5 %) và `hijacked_during` tăng (0–1 → 5–8 ca). Lợi ích ở các sự kiện được chấm bị bù trừ bởi liên kết sai ở nơi khác.
- Đối chứng `iou` cho thấy phần lớn hiệu ứng đến từ **việc có thêm một vòng ghép cho track LOST**, không riêng thông tin hiệp phương sai; gate Mahalanobis chặt hơn (ít refind hơn, ghép nhầm thấp hơn đối chứng) nhưng vẫn không đủ để HOTA tăng.
- → **Theo tiêu chí đăng ký trước (HOTA), C2 không được chọn.** Vì cải thiện trên chỉ số sự kiện là có ý nghĩa, C2 được chạy trên tập test như **nhánh thăm dò**, khai báo trước trong [LOCK_moc2.md](configs/phase2/LOCK_moc2.md) mục 2.

### 14.4 Cấu hình kết hợp

Vì **không trục nào** đạt tiêu chí cải thiện (C1 bão hòa, C3 không khác, C2 không đạt tiêu chí HOTA), cấu hình "kết hợp các thay đổi được chọn" **trùng cấu hình hiện tại** — không có tổ hợp nào để chạy thêm. Đây là kết quả âm và được báo cáo đúng như vậy, kèm nguyên nhân quan sát được (mục 13.2 và 14.1–14.3).

## 15. PF và thống kê (mốc 2, mục 4)

### 15.1 Dấu hiệu suy giảm chất lượng của PF

[pf_ctrv.py](src/pf_ctrv.py) ghi chẩn đoán **tùy chọn** mỗi lần cập nhật (`enable_diag`, không đổi hành vi). Tổng hợp bằng [pf_diag_report.py](src/pf_diag_report.py) → [results/phase2/pf_diag/expA_detrac_dev/](results/phase2/pf_diag/expA_detrac_dev/).

**Định nghĩa và thời điểm đo** (docstring `PFCTRV.enable_diag`):

| Đại lượng | Công thức | Đo lúc nào |
|:--|:--|:--|
| `ess_ratio` | ESS / N với ESS = 1 / Σ wᵢ², w = exp(logw − logsumexp(logw)) — trọng số **đã chuẩn hóa sau khi nhân likelihood** của quan sát | **TRƯỚC** khi resample |
| `resampled` | lần cập nhật đó có resample hay không (systematic, một số ngẫu nhiên) | — |
| `n_unique` | số giá trị **khác nhau** của toạ độ `cx` trong đám mây (xấp xỉ "số particle phân biệt": resample tạo bản sao trùng nhau) | **SAU** bước resample (nếu có) |
| `spread_xy` / `spread_theta` / `spread_v` | độ lệch chuẩn có trọng số của vị trí (px) / **độ lệch chuẩn vòng** của hướng (rad) / tốc độ (px/bước) | sau resample |
| `degenerate` | không particle nào gần quan sát (Mahalanobis² nhỏ nhất > `pf_degenerate_maha2`) → đặt lại vị trí quanh quan sát | tại bước cập nhật |
| `n_lost` | số bước track đã mất dấu tính đến frame đang cập nhật (0 = đang bám) | — |

**Cấu hình đã chạy** (ghi trong `summary.csv`, cột `pf_*`): `pf_resample_ess_ratio` = 0,5 (resample khi
ESS < 0,5 N), `pf_estimate` = `mean`, `pf_roughening` = **0** (tắt), `pf_degenerate_maha2` = 1e4, Q/R **dùng
chung với EKF**, `track_buffer` = 30 bước; **seed 0, 1, 2**; RNG mỗi track = `SeedSequence([seed, hash(video),
track_id])`. Kết luận dưới đây **chỉ áp dụng cho bootstrap PF này với các cấu hình đã thử** — không suy rộng
cho PF nói chung (proposal khác, roughening > 0, Q riêng cho PF đều chưa chạy).

| N | ESS/N (± SD giữa 3 seed) | % update phải resample | particle khác nhau / N | suy biến |
|:--|:--|--:|--:|--:|
| 100 | 0,248 ± 0,0006 | 98,4 % | 0,374 | 0 |
| 300 | 0,256 ± 0,0010 | 99,9 % | 0,382 | 0 |
| 500 | 0,260 ± 0,0010 | 100,0 % | 0,387 | 0 |

**Theo số bước đã mất dấu** (c25, trung bình 3 seed — [by_n_lost.csv](results/phase2/pf_diag/expA_detrac_dev/by_n_lost.csv)) — chính là lúc phải nối lại track:

| n_lost | ESS/N (N=500) | particle khác nhau / N (N=500) | ESS/N (N=100) | particle khác nhau / N (N=100) |
|:--|--:|--:|--:|--:|
| 0 (đang bám) | 0,260 | 0,387 | 0,251 | 0,378 |
| 1–2 | 0,124 | 0,240 | 0,122 | 0,228 |
| 3–5 | 0,077 | 0,173 | 0,081 | 0,170 |
| 6–10 | 0,035 | 0,097 | 0,038 | 0,092 |
| 11–30 | 0,011 | 0,034 | 0,017 | 0,035 |

→ **Điều các con số này chứng minh** (và chỉ chừng đó): ở những lần **cập nhật xảy ra sau một khoảng mất dấu
dài**, trọng số tập trung mạnh ngay sau khi nhân likelihood (ESS/N ~1 % — đo **trước** resample) và đám mây chỉ
còn ~3 % số particle phân biệt **sau** resample. Đây là dấu hiệu **mất đa dạng hạt (sample impoverishment)**
trong các lần nối lại **thành công** mà log ghi được.

**Không** được suy ra từ đây rằng (i) sự mất đa dạng đã xảy ra **trước khi** có quan sát mới — log không đo
phân bố tiên nghiệm ngay trước quan sát, chỉ đo sau likelihood; (ii) đây là **nguyên nhân** khiến các track
khác **không** nối lại được — những lần thất bại đó không có bước cập nhật nên không có dòng log tương ứng.
Muốn khẳng định mạnh hơn phải đo riêng tiên nghiệm/hậu nghiệm ở từng bước (chưa làm).

Điều này **nhất quán** với kết luận mốc 1 (PF kém hơn dưới **cùng giả định nhiễu** với EKF — không phải lỗi cài
đặt): Q rất nhỏ cho θ/ω + resample gần như mỗi bước (98–100 % số lần cập nhật); tăng N cải thiện đơn điệu.
Giới hạn: kết luận gắn với **implementation bootstrap PF này** và **các cấu hình đã liệt kê ở trên**; cải tiến
PF (roughening, nhiễu tham số, proposal khác) là **đổi mô hình nhiễu** → thí nghiệm riêng, **chưa làm**.

### 15.2 Cách tổng hợp thống kê (nhất quán với bảng chính)

[boot_stats.py](src/boot_stats.py) (mới — mục còn nợ của mốc 1): `stage_eval` lưu thêm `raw_fields.json` (HOTA_TP/FN/FP và AssA/AssRe/AssPr theo 19 ngưỡng α, IDTP/IDFN/IDFP, CLR_TP/FN/FP, IDSW cho **từng sequence**). Bootstrap **ghép cặp theo video** (5000 lần, seed 0) **cộng thống kê đủ** của các video được chọn rồi **tái tổ hợp đúng công thức `combine_sequences` của TrackEval** → CI cho chênh lệch của **chính chỉ số COMBINED** in trong bảng chính. Kiểm chứng: tái tổ hợp toàn bộ video **trùng** `combined_metrics.csv` tới 1e-9 trên dữ liệu thật (`test_boot_stats.py` 6/6).

`aggregate_phase2.py` nay xuất **cả hai**: `paired_bootstrap_combined.csv` (**bảng chính**) và `paired_bootstrap.csv` (trung bình theo video — chỉ tham khảo, ghi rõ là đại lượng khác). Với chỉ số **sự kiện** (preserved), bootstrap vẫn **cụm theo video** (`occlusion_eval.bootstrap_diff_by_video`) vì các sự kiện trong cùng một video không độc lập.

**PF — hai cách tổng hợp theo seed là hai đại lượng khác nhau** (metric phi tuyến): bảng chính lấy **trung
bình của metric** từng seed; bootstrap lấy **trung bình thống kê đủ** theo seed rồi mới tái tổ hợp metric.
Đã đo chênh lệch trên chính dữ liệu này: |ΔHOTA| = 6,7·10⁻⁶ (test c25), 4,2·10⁻⁶ (test c10), 9,5·10⁻⁶ (dev
c25 pf500), 4,5·10⁻⁶ (dev c25 pf100); |ΔIDF1| ≤ 1,2·10⁻⁶ — nhỏ hơn 3–4 bậc so với khoảng cách PF–EKF
(≈ 3·10⁻²), nên không ảnh hưởng kết luận nào; vẫn ghi rõ ở đây để người đọc không nhầm hai đại lượng.
Ngoài ra, CI bootstrap **theo video** không bao gồm bất định do **chọn seed**; biến thiên giữa 3 seed được
báo cáo riêng (SD HOTA ≤ 0,005 trên dev, ≤ 0,0004 trên test).

### 15.3 Nhóm B trình bày theo từng cặp

`aggregate_phase2.py` nhận khai báo `pairs` trong file thí nghiệm → `paired_pairs.csv`, mỗi cặp ghi rõ **yếu tố được kiểm soát** ([groupB_detrac_dev.yaml](configs/phase2/experiments/groupB_detrac_dev.yaml), [expB_detrac_test.yaml](configs/phase2/experiments/expB_detrac_test.yaml)): (1) SORT nguyên bản với hai thời gian giữ track; (2) SORT nguyên bản vs SORT–CV–XYAH **cùng vòng đời**; (3) SORT–CV–XYAH vs SORT–EKF, **các thành phần khác cố định**; (4) chỉ đổi bộ lọc ở `max_age` = 1; (5) SORT–EKF vs ByteTrack–EKF (chỉ đổi vòng đời/association). Diễn giải ở mục 7.2 chỉ nói về yếu tố được kiểm soát trong từng cặp.

## 16. Kết quả trên 40 video test UA-DETRAC (mốc 2, mục 5)

### 16.1 Bản khóa và tính hợp lệ của phép đánh giá

Cấu hình được **khóa trước** khi chạy: [LOCK_moc2.md](configs/phase2/LOCK_moc2.md) +
[lock_moc2.json](results/phase2/lock/lock_moc2.json) (sinh bởi [make_lock.py](src/make_lock.py)) ghi commit
`a64cb754`, sha256 từng file thí nghiệm và từng YAML tracker, 40 sequence, `meta.json` của cache (sha256
`yolov8n.pt` = `f59b3d83…`), phiên bản thư viện.

**Kiểm chứng thứ tự thời gian**: `lock_moc2.json.created` = **2026-09-16T02:33:51**; `run_meta.json.created`
sớm nhất trong các thư mục `p2-exp*_detrac_test-*` = **2026-09-16T02:44:45**, muộn nhất 04:43:16 →
**22/22 lần chạy test đều diễn ra SAU khi khóa**. (`lock_moc2.json` ghi `git_dirty: true` vì trong working
tree có hai file **của người dùng** đang sửa — `.gitignore`, `src/demo/rerun_with_state_log.py` — không liên
quan đến mã chạy thí nghiệm.)

**Không đổi tiêu chí sau khi xem kết quả**: tiêu chí đăng ký trước (HOTA `c25` trên **dev**, CI 95 % không
chứa 0) chỉ dùng để **chọn cấu hình**; kết quả test dưới đây là **đánh giá**, không dùng để chọn lại. Không
có tuning hay ablation mới nào chạy sau khi xem kết quả test.

Manifest tái lập: `results/phase2/{expA,expB,expC}_detrac_test/{experiment.yaml, runs.csv, videos.json}`,
`run_meta.json` từng lần chạy (cache_id, sha YAML, conf, classes, `track_buffer`, seed, commit, thời điểm).

### 16.2 Thí nghiệm A — so sánh **bộ lọc** trong cùng ByteTrack (40 video, 1324 sự kiện che khuất)

| tracker | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | FP | FN | preserved | id_continuous |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| **c25** cv | 0.5824 | 0.5355 | 0.6393 | 0.7012 | 0.5923 | 2498 | 73 478 | 199 544 | 0.480 | 0.297 |
| c25 ekf | 0.5835 | 0.5355 | 0.6418 | 0.7036 | 0.5926 | 2525 | 73 140 | 199 661 | 0.487 | 0.297 |
| c25 ukf | 0.5824 | 0.5353 | 0.6396 | 0.7018 | 0.5924 | 2543 | 73 248 | 199 671 | 0.485 | 0.298 |
| c25 pf500 (3 seed) | 0.5519 | 0.5118 | 0.6036 | 0.6596 | 0.5518 | 3555 | 84 178 | 215 123 | 0.435 | 0.272 |
| **c10** cv | 0.5797 | 0.5357 | 0.6334 | 0.6929 | 0.5792 | 2076 | 100 794 | 181 462 | 0.488 | 0.366 |
| c10 ekf | 0.5830 | 0.5358 | 0.6405 | 0.6982 | 0.5798 | 2058 | 100 467 | 181 420 | 0.491 | 0.369 |
| c10 ukf | 0.5815 | 0.5357 | 0.6373 | 0.6960 | 0.5797 | 2103 | 100 621 | 181 336 | 0.496 | 0.370 |
| c10 pf500 (3 seed) | 0.5505 | 0.5127 | 0.5997 | 0.6539 | 0.5391 | 3128 | 110 244 | 198 082 | 0.446 | 0.339 |

Chênh lệch COMBINED + CI 95 % (bootstrap ghép cặp theo video, 5000 lần, seed 0 — `paired_pairs.csv`):

| cặp | điều kiện | ΔHOTA | ΔIDF1 | ΔAssA | ΔIDSW |
|:--|:--|:--|:--|:--|:--|
| ekf − cv | c25 | +0.0012 [−0.0003, +0.0027] | +0.0024 [−0.0004, +0.0051] | +0.0025 [−0.0004, +0.0056] | +27 [−25, +77] |
| ekf − cv | **c10** | **+0.0033 [+0.0011, +0.0057]** | **+0.0053 [+0.0019, +0.0089]** | **+0.0072 [+0.0024, +0.0123]** | −18 [−74, +35] |
| ukf − cv | c25 | +0.0001 [−0.0012, +0.0014] | +0.0007 [−0.0022, +0.0036] | +0.0004 [−0.0024, +0.0031] | +45 [−13, +102] |
| ukf − cv | c10 | +0.0019 [−0.0002, +0.0041] | +0.0031 [−0.0007, +0.0073] | +0.0040 [−0.0004, +0.0088] | +27 [−27, +82] |
| ukf − ekf | c25 | **−0.0011 [−0.0019, −0.0002]** | −0.0018 [−0.0040, +0.0004] | **−0.0022 [−0.0039, −0.0004]** | +18 [−24, +57] |
| pf500 − ekf | c25 | **−0.0316 [−0.0389, −0.0255]** | **−0.0440 [−0.0555, −0.0340]** | **−0.0382 [−0.0489, −0.0294]** | **+1030 [+787, +1307]** |

Giữ ID qua che khuất (bootstrap theo video, cùng 1324 sự kiện; baseline `cv`): ekf +0.0068 [−0.0065, +0.0198]
(c25), +0.0030 [−0.0052, +0.0115] (c10); ukf +0.0045 [−0.0061, +0.0125] (c25); pf500 −0.042…−0.048 (mọi seed,
CI không chứa 0). Theo tầng (full n = 182 / partial n = 1142, c25): cv 0.060 / 0.547; ekf 0.060 / 0.555;
ukf 0.060 / 0.553; pf500 0.053 / 0.495.

**Đọc kết quả A (test)**:
1. Ở `c10` (ngưỡng thấp, vòng association 2 có dữ liệu), **EKF–CTRV hơn KF–CV có ý nghĩa thống kê** trên 40
   video: ΔHOTA +0.0033, ΔIDF1 +0.0053, ΔAssA +0.0072 — CI không chứa 0. Đây là **lần đầu** trong toàn bộ đề
   tài một khác biệt giữa motion model đạt ý nghĩa; nhưng **độ lớn rất nhỏ** (+0,57 % tương đối HOTA) và chỉ
   thấy được nhờ mẫu lớn (40 video, ghép cặp). Ở `c25` (cấu hình khóa của GĐ1) CI vẫn chứa 0 (+0.0012).
2. UKF **không** hơn CV, và **kém EKF** một chút có ý nghĩa ở c25 (−0.0011) — nhất quán với GĐ4 (UKF ngoại suy
   kém hơn EKF ở CTRV).
3. **PF–CTRV kém rõ rệt** ở mọi chỉ số, nhất quán dev; biến thiên giữa 3 seed nhỏ (SD HOTA ≤ 0,0004) → chênh
   lệch không do seed.
4. Giữ ID qua che khuất **không** khác biệt có ý nghĩa giữa CV/EKF/UKF. Ở **tầng che nặng** (`has_full`:
   sự kiện có đoạn che ≥ 0,90 — xem định nghĩa ở mục 13.1), **tại điều kiện c25** cả ba đều 0,060 (11/182 sự
   kiện); **tại c10** UKF nhỉnh hơn một sự kiện (12/182 = 0,066) còn CV/EKF vẫn 11/182 — khác biệt một sự kiện,
   không có ý nghĩa thống kê. Kết luận "cả ba như nhau" chỉ phát biểu cho c25, và phù hợp với phân tích nguyên
   nhân (mục 13.2): rào cản là IoU ở bước ghép, không phải độ chính xác của motion model.

### 16.3 Thí nghiệm B — so sánh **thuật toán**, mỗi cặp đổi một yếu tố (40 video, c25)

> ⚠️ **Bản TRƯỚC khi sửa lỗi vòng đời SORT** (mục 18.1). Kết quả sau khi sửa: mục 18.2 — mọi kết luận
> định tính không đổi, chênh lệch ≤ 0.002 HOTA, IDSW giảm 9–12 %.

| tracker | ý nghĩa | HOTA | DetA | AssA | IDF1 | MOTA | IDSW | preserved |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| sort-orig1 | SORT nguyên bản (KF 7-D, max_age 1) — baseline | 0.5304 | 0.5455 | 0.5212 | 0.5917 | 0.5997 | 5890 | 0.329 |
| sort-orig30 | + bộ đệm 30 | 0.5584 | 0.5417 | 0.5807 | 0.6522 | 0.5976 | 4438 | 0.486 |
| sort-cv30 | SORT + KalmanFilterXYAH | 0.5451 | 0.5283 | 0.5677 | 0.6452 | 0.5835 | 4517 | 0.449 |
| sort-ekf30 | SORT + EKF–CTRV | 0.5471 | 0.5284 | 0.5720 | 0.6491 | 0.5835 | 4433 | 0.451 |
| sort-ekf1 | SORT + EKF–CTRV, max_age 1 | 0.5164 | 0.5312 | 0.5078 | 0.5835 | 0.5843 | 6158 | 0.310 |
| bt-ekf | ByteTrack + EKF–CTRV | 0.5835 | 0.5355 | 0.6418 | 0.7036 | 0.5926 | 2525 | 0.487 |

Từng cặp (COMBINED + CI 95 %; chỉ diễn giải yếu tố được kiểm soát):

| cặp (yếu tố duy nhất thay đổi) | ΔHOTA | ΔIDF1 | ΔIDSW | Δpreserved |
|:--|:--|:--|:--|:--|
| bộ đệm 1 → 30 (SORT nguyên bản) | **+0.0280 [+0.0207, +0.0350]** | **+0.0605 [+0.0468, +0.0743]** | **−1452 [−1862, −1099]** | **+0.158 [+0.119, +0.194]** |
| gói biểu diễn + thiết lập bộ lọc: KF 7-D + Q/R/P₀ của SORT → XYAH + Q/R/P₀ của ByteTrack (cùng vòng đời SORT) | **−0.0133 [−0.0185, −0.0077]** | −0.0070 [−0.0144, +0.0014] | +79 [−71, +236] | −0.037 (0.486 → 0.449) |
| motion model CV → EKF–CTRV (cùng XYAH, cùng vòng đời) | +0.0020 [−0.0001, +0.0042] | +0.0038 [−0.0001, +0.0075] | **−84 [−162, −4]** | +0.002 |
| chỉ đổi bộ lọc ở max_age 1 | **−0.0140 [−0.0167, −0.0116]** | −0.0082 [−0.0108, −0.0059] | **+268 [+135, +410]** | −0.018 [−0.026, −0.011] |
| vòng đời/association SORT → ByteTrack (cùng EKF) | **+0.0364 [+0.0298, +0.0428]** | **+0.0545 [+0.0452, +0.0641]** | **−1908 [−2341, −1492]** | +0.036 |

**Thứ tự ảnh hưởng trên test** (chỉ trong đúng các cặp đã kiểm soát ở trên, **không** phải một phép phân rã
cộng hay thứ hạng phổ quát): vòng đời/association (+0.036) > bộ đệm (+0.028) > gói biểu diễn + thiết lập bộ
lọc (−0.013: XYAH + Q/R/P₀ của ByteTrack **kém hơn** KF 7-D + Q/R/P₀ của SORT **trong vòng đời SORT**)
≫ motion model CV→EKF (+0.002, CI chạm 0; chỉ IDSW giảm có ý nghĩa). Trùng hướng với dev, nay có mẫu 40 video.
Lưu ý: cặp thứ ba đổi **hai thứ cùng lúc** (biểu diễn state và tham số nhiễu) nên không được gán toàn bộ
−0.0133 cho riêng "biểu diễn trạng thái".

### 16.4 Thí nghiệm C — nhánh **thăm dò** (không phải cấu hình được chọn)

Theo tiêu chí đăng ký trước, **không trục nào của Nhóm C được chọn** (mục 14). `cv-maha`/`ekf-maha` chạy trên
test như **nhánh kiểm chứng giả thuyết**, khai báo trước trong LOCK mục 2:

| tracker | HOTA | ΔHOTA vs gốc | IDSW | FP | preserved | preserved full / partial |
|:--|--:|:--|--:|--:|--:|:--|
| cv | 0.5824 | — | 2498 | 73 478 | 0.480 | 0.060 / 0.547 |
| cv + maha | 0.5778 | **−0.0046 [−0.0072, −0.0021]** | 2759 | 81 428 | 0.516 (**+0.0355 [+0.0185, +0.0508]** so với `cv`) | **0.121** / 0.579 |
| ekf | 0.5835 | — | 2525 | 73 140 | 0.487 | 0.060 / 0.555 |
| ekf + maha | 0.5801 | **−0.0034 [−0.0058, −0.0012]** | 2844 | 81 332 | 0.526 (**+0.0385 [+0.0197, +0.0535]** so với `ekf`) | **0.121** / 0.590 |

**Cặp đối chứng đúng cho `preserved`** (mỗi nhánh so với **chính bộ lọc của nó**, bootstrap theo video 5000
lần, seed 0 — tính lại từ `paired_DETRAC-test.csv` / `paired_DETRAC-all.csv`):

| cặp | test (40 video, 1324 sự kiện) | dev (12 video, 184 sự kiện) |
|:--|:--|:--|
| `cv-maha − cv` | **+0.0355 [+0.0185, +0.0508]** | **+0.0489 [+0.0107, +0.0803]** |
| `ekf-maha − ekf` | **+0.0385 [+0.0197, +0.0535]** | +0.0163 [−0.0387, +0.0691] (CI **chứa 0**) |

Con số "+0,045" trong bản trước là `ekf-maha` so với **`cv`** — gộp cả việc đổi bộ lọc, **không** dùng để mô
tả riêng tác động của cơ chế nối lại; đã thay bằng cặp đối chứng đúng ở trên. Trên **dev**, nhánh EKF **không**
đạt ý nghĩa cho `preserved` (chỉ nhánh CV đạt) — thêm một lý do để C2 không được chọn theo tiêu chí đăng ký
trước, và để đọc kết quả test của C2 như **kiểm chứng giả thuyết**, không phải khẳng định cải thiện.

**Kết luận C trên test (kết quả âm, báo cáo đúng như đo được)**: vòng ghép thứ 3 cho track LOST **tăng có ý
nghĩa** khả năng giữ ID qua che khuất (+0,036…+0,045; ở tầng che hoàn toàn **gấp đôi**: 0,060 → 0,121, tức
11/182 → 22/182 sự kiện) nhưng **làm giảm HOTA có ý nghĩa** (−0.0046 / −0.0034) do FP tăng ~8 000 và IDSW
tăng 261–319. Trên dev, ΔHOTA có CI chứa 0; trên test (mẫu lớn hơn) dấu âm trở nên rõ ràng. Nghĩa là **cơ chế
này không nên bật** nếu mục tiêu là HOTA; nó chỉ hữu ích nếu bài toán ưu tiên liên tục ID và chấp nhận thêm
FP/ghép nhầm.

### 16.5 Tốc độ

Tốc độ **tracker chạy từ cache** trên 40 video test (56 340 frame, CPU; đo bằng `summary.csv` của từng lần
chạy, **không** có detector): sort-orig1 774 fps, sort-orig30 734, bt-ekf 498, cv 420–570, ekf 333–478,
ukf 175–187, pf500 106–114 fps (predict 0,09–3,6 ms/frame). Các lần chạy diễn ra khi máy còn chạy việc khác
nên **chỉ mang tính tham khảo tương đối**; con số đo có kiểm soát (450 frame, sau warm-up) nằm ở mục 7.3.
**Toàn pipeline có detector** (GTX 1650): 45–51 fps (mục 7.3) — hai loại số này **không** so sánh trực tiếp.
**Chưa đo trên Jetson Nano.**

### 16.6 Đối chiếu dev ↔ test

| Kết luận | dev (12 video, dùng để chọn) | test (40 video, đánh giá) | Nhất quán? |
|:--|:--|:--|:--|
| EKF vs CV | +0.0025 [−0.0018, +0.0060] (c25) | +0.0012 [−0.0003, +0.0027] (c25); **+0.0033 [+0.0011, +0.0057]** (c10) | có (dấu dương), test có ý nghĩa ở c10 |
| UKF vs EKF | −0.0046 (CI chứa 0) | −0.0011 [−0.0019, −0.0002] | có |
| PF vs EKF | −0.063…−0.097 | −0.0316 [−0.0389, −0.0255] | có |
| Bộ đệm > bộ lọc (Nhóm B) | +0.018 (buffer) vs −0.003 (bộ lọc) | +0.028 vs +0.002 | có |
| Vòng đời ByteTrack > SORT | +0.022 (CI chứa 0) | +0.0364 [+0.0298, +0.0428] | có, test rõ hơn |
| C2 (vòng ghép 3): preserved tăng, HOTA không tăng | preserved +0.049 [+0.011, +0.080]; HOTA CI chứa 0 | preserved +0.036 [+0.019, +0.051]; HOTA **−0.0046 [−0.0072, −0.0021]** | có (test cho thấy HOTA **giảm**) |
| Tầng che hoàn toàn ≈ 6 % với mọi bộ lọc | 0,061–0,091 (n = 33) | 0,060 (n = 182) | có |

## 17. BDD100K và Jetson Nano — đã làm, chưa làm, đầu vào còn thiếu (mốc 2, mục 6)

| Hạng mục | Đã làm (có trong repo) | Chưa làm | Đầu vào còn thiếu |
|:--|:--|:--|:--|
| Adapter BDD100K-MOT | `src/mot_datasets/bdd100k_mot.py` (schema box_track_20 fail-loud, giữ nguyên lớp/thuộc tính, Δt từ `timestamp`, `track_gaps`, xuất MOTChallenge theo lớp, tách output theo lớp bằng majority vote); fixture `data/fixtures/bdd100k_mini`; `src/check_bdd100k.py`; 42/42 kiểm tra | chạy trên **nhãn thật** | `labels/box_track_20/{train,val}` + `images/track/val` (≈ 200 video val, ~40 video train cho dev) đặt vào `data/raw/bdd100k/` |
| Vùng bỏ qua (crowd) | `src/trackeval_ext.py` `MotChallengeCrowdIgnore` (ghép GT trước, IoA > 0,5 sau), **đối chiếu trùng số** với `trackeval.datasets.BDD100K` gốc; 3 trường hợp kiểm tra | — | — |
| Khoảng trống annotation | `gap_evidence ∈ {occluded_adjacent, unknown}` — **không** suy diễn gap = che hoàn toàn; tầng riêng trong `occlusion_strata.csv` | — | — |
| Δt và Q theo thời gian | `src/time_units.py` (đơn vị bước, `steps_between`, `track_buffer_frames`, `convert_state_units`); **`src/test_timestep_integration.py`** chứng minh timestamp → dự đoán → vòng đời cho CV/EKF/UKF/PF (32/32) | kiểm tra **phân bố Δt thật** của BDD (5 fps, có frame lệch) và chọn lại `std_theta`/`std_omega` cho bước 0,2 s | nhãn BDD thật (để `check_bdd100k.py` thống kê Δt) |
| Hai thí nghiệm tách biệt | `configs/phase2/experiments/bdd_transfer_val.yaml` (**chuyển cấu hình** từ UA-DETRAC, không tuning) và `bdd_tuned_val.yaml` (**tuning trên train-dev BDD** rồi đánh giá trên val; quy trình + tiêu chí ghi sẵn trong file) | chạy cả hai | như trên + thời gian GPU tạo cache BDD |
| Detector cho giao thông hỗn hợp | `src/prepare_mixed_yolo_dataset.py` (5 lớp, chia theo video, mask vùng bỏ qua), `configs/phase2/detector/mixed5.yaml`, `src/train_detector.py --dry-run`, export ONNX + kiểm tra op TRT 8.2 (`yolo11n` là ứng viên duy nhất) | **huấn luyện** | ảnh BDD train; GPU thời gian dài (GTX 1650 4 GB) |
| Jetson Nano | `scripts/nano/{README.md, 00_check_stack.sh, 01_build_engine.sh, 02_benchmark_engine.sh}` + checklist 5 bước khi có thiết bị; ONNX đã export và kiểm tra op | **mọi phép đo trên thiết bị**; vòng lặp tracker không phụ thuộc torch (ultralytics 8.4 cần Python ≥ 3.8, Nano gốc chỉ có 3.6) | thiết bị Nano (JetPack 4.6, TRT 8.2) |

**Nguyên tắc đã ghi trước** (`bdd_transfer_val.yaml`): kết quả "chuyển cấu hình" và "tuning trên BDD" **không
được gộp** thành một kết luận về khả năng chuyển miền; thay đổi detector là **thí nghiệm riêng**, chỉ làm sau
khi pipeline tracking đã cố định.

## 18. Xử lý review độc lập (Codex) — sửa tính đúng đắn sau lần đánh giá test ban đầu

Bảng trạng thái đầy đủ từng phát hiện: [REVIEW_moc2_codex.md](results/phase2/REVIEW_moc2_codex.md).
Snapshot được review là commit `502f011`; `git diff --stat 502f011..28ca79b` cho thấy **chỉ có 7 file thuộc
`handoff/` + `CODEX_HANDOFF.md`** thay đổi (không đụng `src/`, `configs/`, `results/`) — nguyên nhân là ZIP
được dựng **trước** commit cuối, đã sửa quy trình (nay `make_handoff.py` **từ chối** đóng gói khi còn file
trong phạm vi gói khác HEAD).

**Kết quả cũ của mốc 2 được giữ nguyên**; mọi lần chạy sau khi sửa nằm trong thí nghiệm `*_fix1` (thư mục và
tên tracker mới).

### 18.1 Lỗi P1 — baseline SORT xuất track sớm hơn upstream một frame

`SORTTrack.__init__` khởi tạo `hits = 1; hit_streak = 1`, trong khi `KalmanBoxTracker` của
[abewley/sort](https://github.com/abewley/sort) khởi tạo **0** rồi chỉ tăng khi có update ghép. Với
`min_hits = 3`, track tạo ở frame 10 được xuất ở **frame 12** thay vì **frame 13**. Test cũ
(`test_2_min_hits`) lại khẳng định đúng hành vi lệch đó.

**Đã sửa**: `hits = 0; hit_streak = 0`. Bằng chứng tương đương:
[test_sort_upstream_parity.py](src/test_sort_upstream_parity.py) chạy **chính mã upstream đã ghim**
([third_party/sort_upstream/](third_party/sort_upstream/), commit `bce9f0d`, sha256 `f5b10481…`, GPL-3.0;
`filterpy` cài riêng vào `third_party/_testdeps`) trên **cùng detection** và so **toàn chuỗi**:

| kịch bản | dòng (repo vs upstream) | track | song ánh ID | lệch toạ độ tối đa |
|:--|:--|:--|:--|--:|
| min_hits (xuất từ frame 13) | 4 vs 4 | 1 vs 1 | ✔ đồng nhất | 6,7·10⁻⁶ px |
| track sinh muộn + mất 3 frame, `max_age` 1 / 30 | 64/65 vs 64/65 | 3/2 | ✔ | 3,1·10⁻⁵ px |
| 2 đối tượng, 1 bị che 5 frame, `max_age` 1 / 30 | 52/53 vs 52/53 | 3/2 | ✔ | 3,1·10⁻⁵ px |
| **detection thật** `MVI_39031` 200 frame, `max_age` 1 / 30 | 305/308 vs 305/308 | 8/6 | ✔ | 5,8·10⁻⁵ px |

Khác biệt còn lại **chỉ là số học** (repo: numpy + dạng Joseph; upstream: filterpy), tối đa **5,8·10⁻⁵ px**
trên mọi kịch bản; vòng đời, số track và ánh xạ ID trùng khít. Hai khác biệt **có chủ ý** vẫn còn, đã ghi
trong `PROVENANCE.json`: repo lọc `conf > det_thresh` (upstream nhận detection đã lọc sẵn) và cách xử lý
track có box NaN.

### 18.2 Chạy lại Nhóm B với baseline đã sửa — trước/sau

Cấu hình **y hệt bản đã khóa**, chỉ đổi `name` → [expB_detrac_dev_fix1.yaml](configs/phase2/experiments/expB_detrac_dev_fix1.yaml),
[expB_detrac_test_fix1.yaml](configs/phase2/experiments/expB_detrac_test_fix1.yaml). Không tuning, không đổi
tiêu chí. **`bt-ekf` (ByteTrack) không dùng `SORTTrack`** → chạy lại cho ra kết quả **bit-identical** bản cũ:
12/12 video (dev) và 40/40 (test), song ánh ID đồng nhất (`check_identical_runs.py`).

**Chỉ số tuyệt đối (c25)** — trước → sau:

| tracker | dev HOTA | dev IDSW | test HOTA | test IDSW | test preserved |
|:--|:--|:--|:--|:--|:--|
| sort-orig1 | 0.5731 → **0.5715** | 1038 → **912** | 0.5304 → **0.5311** | 5890 → **5369** | 0.329 → 0.327 |
| sort-orig30 | 0.5872 → **0.5854** | 726 → **681** | 0.5584 → **0.5585** | 4438 → **4314** | 0.486 → 0.485 |
| sort-cv30 | 0.5690 → **0.5677** | 747 → **680** | 0.5451 → **0.5452** | 4517 → **4318** | 0.449 → 0.450 |
| sort-ekf30 | 0.5666 → **0.5652** | 742 → **679** | 0.5471 → **0.5472** | 4433 → **4210** | 0.451 → 0.452 |
| sort-ekf1 | 0.5513 → **0.5500** | 1062 → **925** | 0.5164 → **0.5171** | 6158 → **5555** | 0.310 → 0.310 |
| bt-ekf (đối chứng) | 0.5925 → 0.5925 | 472 → 472 | 0.5835 → 0.5835 | 2525 → 2525 | 0.487 → 0.487 |

Hướng thay đổi đúng như dự đoán từ cơ chế: track được xuất muộn hơn một frame ⇒ ít track "non" hơn ⇒ **IDSW
giảm rõ** (test: −9…−12 %), HOTA gần như không đổi.

**Từng cặp so sánh trên TEST** (COMBINED, bootstrap ghép cặp theo video 5000 lần, seed 0) — trước → sau:

| cặp (yếu tố được kiểm soát) | trước khi sửa | sau khi sửa |
|:--|:--|:--|
| bộ đệm 1 → 30 | +0.0280 [+0.0207, +0.0350] | **+0.0274 [+0.0201, +0.0345]** |
| gói biểu diễn + thiết lập bộ lọc (KF 7-D → XYAH) | −0.0133 [−0.0185, −0.0077] | **−0.0132 [−0.0185, −0.0077]** |
| motion model CV → EKF–CTRV | +0.0020 [−0.0001, +0.0042] | **+0.0020 [−0.0001, +0.0042]** |
| chỉ đổi bộ lọc ở `max_age` 1 | −0.0140 [−0.0167, −0.0116] | **−0.0140 [−0.0167, −0.0116]** |
| vòng đời SORT → ByteTrack | +0.0364 [+0.0298, +0.0428] | **+0.0363 [+0.0296, +0.0428]** |

Trên **dev** (trung bình theo video, so với `sort-orig1` — đại lượng duy nhất có sẵn ở cả hai bản):
orig30 +0.0178 → **+0.0177**; cv30 −0.0028 → **−0.0026**; ekf30 −0.0034 → **−0.0033**; ekf1 −0.0256 →
**−0.0253**; bt-ekf +0.0218 → **+0.0238**. Bản mới còn có bootstrap COMBINED cho từng cặp:
bộ đệm **+0.0139 [+0.0071, +0.0285]**; KF 7-D → XYAH **−0.0178 [−0.0406, −0.0026]**; CV → EKF
−0.0024 [−0.0052, +0.0013]; `max_age` 1 đổi bộ lọc −0.0215 [−0.0425, −0.0086]; SORT → ByteTrack
**+0.0273 [+0.0199, +0.0327]**.

**Kết luận Nhóm B sau khi sửa** (giữ nguyên tiêu chí): mọi kết luận định tính **không đổi** — thứ tự ảnh
hưởng vẫn là vòng đời/association > bộ đệm > gói biểu diễn + thiết lập bộ lọc ≫ motion model (CV→EKF vẫn có
CI chạm/chứa 0 ở cả dev và test; chỉ IDSW giảm có ý nghĩa trên test). Khác biệt số học ở mức ≤ 0.002 HOTA,
không đảo chiều cặp nào. **Các bảng Nhóm B dùng cho luận văn phải lấy từ `*_fix1`** (mục 7.2 và 16.3 giữ lại
để đối chiếu lịch sử, đã ghi chú rõ).

### 18.3 Lỗi P1 — runner có thể chấm tracking mới bằng metric cũ

`stage_eval` trước đây bỏ qua TrackEval **chỉ vì ba file kết quả tồn tại**; vòng occlusion bỏ qua khi có
`bootstrap_*.csv`; `--no-skip` chỉ tác động bước tracking; `stage_track` ghi `run_meta.json` (commit, cấu
hình, `created`) **trước** khi bỏ qua video đã có; lỗi subprocess chỉ được in còn `main()` luôn trả 0.

**Đã sửa** ([run_experiment.py](src/run_experiment.py) + [run_fingerprint.py](src/run_fingerprint.py)):
mỗi bước đánh giá ghi **dấu vân tay** (`eval_meta.json`, `occl_meta.json`) gồm sha256 của **từng file
tracking**, cấu hình, cache (`meta.json` + từng `.npz`), danh sách sequence, **mã nguồn** (git commit +
sha256 của 17 module ảnh hưởng kết quả) và phiên bản `trackeval`. Chỉ tái sử dụng khi dấu vân tay **trùng
khít**, lệch thì chạy lại và **in rõ trường nào lệch**. Bước tracking cũng có `input_fingerprint`: đổi cấu
hình khi "tiếp tục" một run ⇒ chạy lại toàn bộ. `run_meta.json` chỉ ghi khi thực sự chạy; khi tái sử dụng thì
**giữ provenance gốc** và ghi thêm mục `resumed`. Thêm `--force` / `--force-eval`; lỗi đánh giá/tổng hợp ⇒
**exit code 1**.

Kiểm chứng: [test_run_experiment_guard.py](src/test_run_experiment_guard.py) **7/7** trên fixture BDD100K —
(1) chạy lại không đổi gì ⇒ "TÁI SỬ DỤNG"; (2) sửa một file tracking ⇒ đánh giá chạy lại (lý do `tracking`)
và HOTA đổi; (3) đổi cấu hình khi resume ⇒ tracking chạy lại toàn bộ, provenance giữ nguyên + có `resumed`;
(4) `--force` ⇒ mọi bước chạy lại; (5) evaluator ném lỗi ⇒ đếm thất bại và exit code ≠ 0.

### 18.4 Lỗi P2 — chính sách detection cache

`DetectorSpec.STRICT_FIELDS` thiếu `conf`, `classes`, `max_det` ⇒ có thể **ghi thêm** detection với ngưỡng/lớp
hẹp hơn trong khi `meta.json` vẫn tuyên bố cấu hình rộng hơn. **Đã sửa** ([det_cache.py](src/det_cache.py)):
thêm ba trường vào `STRICT_FIELDS`; `versions` thành `WARN_FIELDS` (cảnh báo + ghi `version_history`);
`write_meta(mode="append"|"create")`; `save_video` ghi `spec_sha256` cho **từng sequence**; `open_cache` kiểm
`spec_sha256` (lệch ⇒ lỗi ở chế độ strict). Việc **đọc** cache rộng xuống cấu hình hẹp hơn vẫn hợp lệ như cũ.
Test: `test_det_cache.py` **47/47** (+10 phép mới).

**Cache hiện có**: [check_cache_consistency.py](src/check_cache_consistency.py) →
[cache_consistency.csv](results/phase2/cache_consistency.csv): 56 sequence, **0 mismatch**; mỗi `cache-id` chỉ
có **một** giá trị `conf`/`max_det`/`weights_sha256`/`versions` trong `meta.json`, `version_history` rỗng.
Các sequence ở trạng thái `legacy` (sinh trước khi có fingerprint) nên không thể khẳng định bằng hash, nhưng
không có dấu hiệu trộn ⇒ **không tái tạo cache**.

### 18.5 Lỗi P2 — đóng gói

ZIP cũ vẫn chứa `.gitignore` (file của người dùng). **Đã sửa** ([make_handoff.py](src/make_handoff.py)):
danh sách `USER_FILES`/`USER_PREFIXES` (`.gitignore`, `live.bat`, `src/demo/**`, `outputs/**`) bị loại tuyệt
đối; gói **từ chối dựng** khi còn file trong phạm vi gói khác HEAD (`--allow-dirty` để bỏ qua, khi đó danh
sách dirty được ghi vào `versions.json` và `INDEX.md`); thêm `handoff/FILE_HASHES.csv` (sha256 từng file) và
`handoff/external/bytetrack.yaml` (bản ultralytics mà LOCK tham chiếu, để đối chiếu hash được).

### 18.6 Các điểm diễn giải đã chỉnh

Xem trực tiếp trong mục 7.2, 13.1, 14.2, 15.1, 15.2, 16.2, 16.3, 16.4: (a) cặp KF 7-D → XYAH đổi **cả** biểu
diễn lẫn Q/R/P₀ nên không quy riêng cho biểu diễn; (b) PF chỉ khẳng định đúng cái đã đo (ESS/N sau likelihood
trước resample; số hạt phân biệt sau resample; **không** suy ra "mất đa dạng trước khi có quan sát" hay
nguyên nhân của các lần nối lại thất bại); (c) thêm cặp đối chứng đúng `ekf-maha − ekf` cho `preserved`
(test **+0.0385 [+0.0197, +0.0535]**, dev +0.0163 [−0.0387, +0.0691] — CI chứa 0); (d) C3 nói "**không cấu
hình nào cải thiện** HOTA có ý nghĩa" (một số cấu hình CV **kém hơn** có ý nghĩa), tỉ lệ ~99 % dưới 1 px chỉ
áp cho nhánh EKF; (e) `has_full` = sự kiện **có đoạn che ≥ 0,90**, và "cả ba bộ lọc 11/182" chỉ đúng ở c25
(c10: UKF 12/182); (f) hai cách tổng hợp PF theo seed là hai đại lượng khác nhau, chênh lệch đo được
≤ 9,5·10⁻⁶ HOTA.

