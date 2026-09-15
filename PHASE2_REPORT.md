# Giai đoạn 2 — Giao thông hỗn hợp, PF–CTRV, BDD100K-MOT, cache detection

Nhánh: `phase2-mixed-traffic` (tách từ `fix/tracker-evaluation-correctness` @ `35a560f`). Chỉ commit cục bộ, **không push/merge**. Môi trường: `khoaluan-mot` (Python 3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124, trackeval 1.3.0, GTX 1650).

## 0. Tóm tắt trạng thái

| Bước | Nội dung | Trạng thái |
|---|---|---|
| 1 | Kiểm tra baseline; sửa lỗi `shared_kalman` bỏ qua tham số bộ lọc | ✅ 5 test cũ PASS (77 phép); lỗi xác minh + sửa; `test_filter_config.py` 21/21; mã HEAD và mã mới **bit-identical** |
| 2 | Thời gian / đơn vị / mô hình nhiễu (6 yêu cầu) | ✅ `time_units.py` (audit + `ElapsedStepsMixin` + đổi đơn vị tương hợp), `test_time_units.py` 20/20 |
| 3 | Detection cache + runner đọc cache | ✅ `det_cache.py`, `run_tracking.py`; `test_det_cache.py` 37/37; capture-and-replay **bit-identical** với `model.track` (CV/EKF/UKF) |
| 4 | Adapter BDD100K-MOT, crowd-ignore IoA, fixture, gap events | ✅ `mot_datasets/bdd100k_mot.py`, `trackeval_ext.py`, `check_bdd100k.py`; `test_bdd100k_adapter.py` 41/41, `test_trackeval_ext.py` 5/5; regression GĐ1 trùng khít |
| 5 | PF–CTRV | ✅ `pf_ctrv.py`, `PFSTrack`/`PFByteTracker`; `test_pf_ctrv.py` 26/26 |
| 6 | Pilot Nhóm A trên dev UA-DETRAC (12 video) | ✅ chạy thật — xem mục 7 |
| — | SORT / SORT–EKF, detector mới, profiling Nano, aggregate bằng `boot_stats` | ⏳ chưa làm (mục 9) |

Dữ liệu/thiết bị: **BDD100K-MOT không có trên máy Windows này** (đã tìm toàn bộ D:, Downloads, thư mục dự án; `ByteTrack + EKF + Kitty/datasets/BDD100K` rỗng). Bộ nhãn dự phòng "gộp lớp 0" cũng không có. Nano chưa kết nối. Mọi phần BDD chạy trên fixture tổng hợp.

## 1. Hiện trạng đã xác minh (không suy đoán)

- Pipeline GĐ1: `baseline_track.py` gọi `model.track(frame, persist=True)` từng frame → YOLO chạy lại cho mỗi motion model; **không có cache**. Detector khóa: `models/yolov8n.pt` (sha256 `f59b3d83…`), classes COCO [2,3,5,7], conf 0,25 (A) / 0,10 (B), imgsz 640, iou 0,7.
- Bộ lọc: state 9-D `[cx,cy,v,theta,omega,a,h,va,vh]`, `dt = 1 bước`, mọi tham số nhiễu theo bước ([time_units.py](src/time_units.py) mục A).
- **Lỗi tìm thấy**: [tracker_ctrv.py](src/tracker_ctrv.py) `CTRVSTrack.multi_predict` dùng thuộc tính lớp `shared_kalman` (tạo lúc import). `BYTETracker.update` chỉ gọi `multi_predict`, nên mọi tham số bộ lọc đặt trên instance của tracker (từ YAML) **bị bỏ qua ở bước dự đoán**. Đo được: đặt `std_theta = 1.0` trên bộ lọc của tracker → sau `multi_predict`, `P[θ,θ]` vẫn tăng đúng bằng mặc định 0,010². GĐ1 chỉ dùng mặc định nên **kết quả cũ không bị ảnh hưởng** (đã chứng minh bit-identical, mục 2.1).
- TrackEval pip 1.3.0: lớp `BDD100K` dùng schema cũ (`index`, `Crowd`); `MotChallenge2DBox` không có vùng bỏ qua (crowd) — cần lớp mở rộng (mục 5).

## 2. Bước 1 — Sửa `shared_kalman` + regression

Thay đổi: [ekf_ctrv.py](src/ekf_ctrv.py) giữ tham số nhiễu/khởi tạo **trên instance** (`std_theta, std_omega, init_std_theta, init_std_omega, heading_std_after_init, min_speed_for_heading`, mặc định = `config.py`); [ukf_ctrv.py](src/ukf_ctrv.py) chuyển `**kwargs`; `CTRVSTrack.multi_predict` dùng `stracks[0].kalman_filter`; `CTRVByteTracker.get_kalmanfilter()` đọc khóa `ctrv_*` từ YAML.

Kiểm tra ([test_filter_config.py](src/test_filter_config.py), 21/21):
1. Bộ lọc mặc định == `shared_kalman` cũ: initiate / try_initiate_from_motion / multi_predict / update **bit-identical** (EKF, UKF).
2. Qua `BYTETracker.update` 12 frame (có frame rỗng): output mới == cũ bit-identical.
3. **Đi đúng đường lỗi**: override `ctrv_std_theta = 1.0` qua YAML → `P[θ,θ]` tăng +1,028 (mặc định +0,029) sau `multi_predict`.
4. Kịch bản 79 frame tổng hợp: mã tại `HEAD` (trước sửa) vs mã mới → EKF/UKF/clamped **bit-identical** (243/242/243 box).

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
- Interim chung với DETRAC; `occlusion_ratio = 0.5` **quy ước** cho cờ `occluded` (che một phần, mức độ không biết; không bao giờ ≥ 0,90). Che hoàn toàn trong BDD = **gap** của track (không annotate) → `occlusion_eval --gaps` tạo sự kiện gap `[last_seen+1, reappear−1]`.
- Xuất MOTChallenge theo lớp: chính `BDD-<part>` (car+truck+bus+motorcycle+bicycle → class 1; crowd/distractor/train → class 13, conf 0), phụ `BDD-<part>_car`, `_motorcycle`, … **Rider/pedestrian chỉ thống kê, không đánh giá** (detector COCO không có lớp rider; không gộp người với xe).
- Tách output tracker theo lớp bằng **majority vote** một lớp/track (`STrack.cls` bị ghi đè mỗi frame).
- [trackeval_ext.py](src/trackeval_ext.py) `MotChallengeCrowdIgnore`: class 13 → `gt_crowd_ignore_regions`, det chưa khớp có **IoA > 0,5** bị loại (theo `trackeval/datasets/bdd100k.py`). Vì sao không dùng "class 8 distractor": det của một xe trong box crowd lớn có IoU ≪ 0,5 với crowd nên không bị loại → FP sai ([test_trackeval_ext.py](src/test_trackeval_ext.py) chứng minh: gốc 3 FP, lớp mới 0 FP; IoA 0,4 vẫn FP ở cả hai; UA-DETRAC không class 13 → **trùng khít** lớp gốc).
- `run_trackeval.py`: `<BENCHMARK>-<split>`, `--dataset-class mot|mot_crowd` (mặc định theo tiền tố). `occlusion_segments.py`: `--parquet --fps --out-tag`, CSV rỗng có header. `occlusion_eval.py`: `--tag --fps --context-s --gaps`, lọc GT class 1. **Regression**: chạy lại trên DETRAC-test tái tạo **trùng khít** 4 CSV của `results/test_eval/A` và `occlusion_segments_test.csv`/`full_*`.
- Fixture [data/fixtures/bdd100k_mini](data/fixtures/bdd100k_mini) (3 video: crowd, gap, occluded, truncated, distractor, timestamp có khoảng bỏ frame). [test_bdd100k_adapter.py](src/test_bdd100k_adapter.py) 41/41 đi hết đường: JSON → interim → MOT theo lớp → cache gt-synthetic → runner CV/EKF (`n_steps = 2` tại chỗ timestamp nhảy 0,4 s; `track_buffer` 1,2 s = 6 bước) → tách lớp → TrackEval crowd-ignore (FP = 0) → occlusion_segments/eval với gap.
- Diễn giải: camera trên xe (ego-motion) + 5 fps làm IoU association yếu hơn nhiều so với camera cố định UA-DETRAC; giao thức khác chính thức ở chỗ dùng MOTChallenge + IoA thay vì toolkit `bdd100k`.

## 6. Bước 5 — PF–CTRV ([pf_ctrv.py](src/pf_ctrv.py))

Bootstrap PF, mỗi track N particle; **dùng chung** f (vector hóa `f_batch`), Q, R, `initiate`, `try_initiate_from_motion` với EKF/UKF (cùng giả định nhiễu). Log-weight + logsumexp; ESS → systematic resampling (1 số ngẫu nhiên); trung bình vòng θ; cov thực nghiệm (residual wrap); suy biến (Mahalanobis² tối thiểu > `pf_degenerate_maha2`) → đặt lại vị trí quanh z, đếm; thiếu measurement → chỉ predict, `vh = 0` áp lên mọi particle khi LOST; roughening tùy chọn (mặc định tắt). RNG từng track `SeedSequence([seed, video, track_id])` → không phụ thuộc thứ tự pool; cùng seed bit-identical. Bbox cho association = ước lượng có trọng số (`mean`) hoặc particle MAP; `track.mean/covariance` luôn được cập nhật nên `project`/`gating_distance` (Mahalanobis) đúng nghĩa. `PFTrackView` giữ giao diện `KalmanFilterXYAH` để `STrack.update/re_activate` của ultralytics chạy nguyên vẹn.

Kiểm tra ([test_pf_ctrv.py](src/test_pf_ctrv.py), 26/26): `f_batch == f` (1e-12); ω≈0; θ qua ±π; 30 bước không measurement (hữu hạn, trace tăng); suy biến; seed/video reproducibility; đảo thứ tự pool; `multi_predict == predict()`; resampling; **nhất quán với EKF khi N = 3000** trên đường thẳng nhiễu Gaussian (RMSE 0,96 vs 0,97 px, v 4,92/5, θ 30,9°/30°); vòng đời bookkeeping == EKF; runner `--seeds`.

**Quan sát quan trọng (không phải lỗi)**: với N = 100, PF suy biến (sample impoverishment) trên cùng đường thẳng: RMSE 26,5 px, θ lệch (−49° vs 30°); hướng/turn-rate không sửa được sau khi resample vì Q của θ/ω rất nhỏ (0,010/0,005 mỗi bước — thiết kế cho bộ lọc Gaussian). N = 500 ≈ EKF. Trên kịch bản tổng hợp có frame rỗng, PF-100 sinh 9 ID cho 3 xe (EKF: 3); N = 500 + roughening 0,5: 3 ID. Đây là hệ quả của giữ **cùng giả định nhiễu** với EKF; cải tiến (roughening, nhiễu tham số) là ablation Nhóm C.

Cấu hình: `configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml` (khóa `pf_*`; tham số ghép cặp giữ nguyên `bytetrack.yaml`).

## 7. Bước 6 — Pilot Nhóm A trên dev UA-DETRAC

Manifest [results/phase2/manifests/detrac_dev.json](results/phase2/manifests/detrac_dev.json): 12/60 video train, phân tầng tercile(`n_full_occ_segs`) × tercile(`pct_curved`), ưu tiên đa dạng thời tiết, seed 0 (chọn **trước** khi xem kết quả); `detrac_test.json` = 40 video test (đánh giá cuối, chưa chạy). Cache `yolov8n-coco_conf0.05_all` (12 video, 16 919 frame, 5,5 phút GPU).

Thí nghiệm [configs/phase2/experiments/groupA_detrac_dev.yaml](configs/phase2/experiments/groupA_detrac_dev.yaml): điều kiện A (>0,25) và B (>0,10); tracker cv / ekf / ukf / pf100 / pf300 / pf500 (PF × seed 0,1,2); cùng cache, cùng association, `track_buffer` 1,2 s. Kết quả: `results/phase2/groupA_detrac_dev/{A,B}/summary.csv`, `per_video.csv`, `paired_bootstrap.csv`, `timing.csv`, `event_outcomes_*.csv`, `bootstrap_*.csv`.

KẾT QUẢ: _(điền sau khi chạy xong — xem mục 7.1)_

### 7.1 Bảng kết quả dev

_(đang chạy)_

## 8. Lệnh chính xác

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe        # (Git Bash) hoặc conda activate khoaluan-mot

# Test (script thuần)
$P src/test_filter_config.py; $P src/test_time_units.py; $P src/test_det_cache.py
$P src/test_run_tracking_replay.py --n-frames 150          # cần ảnh DETRAC + GPU/CPU
$P src/test_trackeval_ext.py; $P src/test_bdd100k_adapter.py; $P src/test_pf_ctrv.py
$P src/test_ekf_ctrv.py; $P src/test_ukf_ctrv.py; $P src/test_init_regression.py; $P src/test_tracker_lifecycle.py; $P src/test_occlusion_eval.py

# Chọn video pilot + cache + thí nghiệm Nhóm A (dev / test)
$P src/select_pilot_videos.py --n 12 --seed 0
$P src/det_cache.py --dataset detrac --part train --manifest results/phase2/manifests/detrac_dev.json --cache-id yolov8n-coco_conf0.05_all --skip-existing
$P src/det_cache.py --dataset detrac --part test  --manifest results/phase2/manifests/detrac_test.json --cache-id yolov8n-coco_conf0.05_all --skip-existing
$P src/run_experiment.py --exp configs/phase2/experiments/groupA_detrac_dev.yaml           # --stage track|eval|aggregate, --dry-run
$P src/run_experiment.py --exp configs/phase2/experiments/groupA_detrac_test.yaml          # SAU khi chốt tham số PF trên dev
$P src/aggregate_phase2.py --exp configs/phase2/experiments/groupA_detrac_dev.yaml

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

## 9. Chưa làm / điểm chặn / hạn chế

- **Chưa làm**: SORT nguyên bản + SORT–CV + SORT–EKF (Nhóm B); Nhóm C ablation template; detector ứng viên (export ONNX/opset cho TensorRT 8.2, dataset trộn, train/benchmark script); `profile_pipeline.py` + `scripts/nano/`; `boot_stats.py` (bootstrap trên thống kê đủ của TrackEval — hiện `aggregate_phase2.py` bootstrap trên **trung bình theo video**, khác COMBINED_SEQ); chạy Nhóm A trên 40 video test (đợi chốt tham số PF trên dev).
- **Dữ liệu**: BDD100K-MOT (labels box_track_20 + images track val/train-subset) chưa có trên máy → mọi phần BDD mới kiểm tra bằng fixture. Khi có: lệnh mục 8. Tham số nhiễu θ/ω cho 5 fps phải chọn lại trên dev BDD (mục 3.3), chưa có giá trị.
- **Thiết bị**: Nano chưa kết nối → chưa đo gì trên thiết bị.
- **Hạn chế diễn giải**: (i) PF ở N nhỏ suy biến dưới cùng giả định nhiễu — so sánh "PF vs EKF" phụ thuộc N và roughening; (ii) BDD `occluded` là cờ nhị phân, che hoàn toàn = gap; (iii) rider không đánh giá; (iv) GPU không bit-reproducible → kết quả GĐ1 chỉ tái lập ở mức cấu trúc track (100% (frame,id)) chứ không bit-exact; từ GĐ2 mọi tracker dùng **cùng** cache nên ghép cặp chính xác; (v) bootstrap trong `aggregate_phase2.py` là trên trung bình theo video.
