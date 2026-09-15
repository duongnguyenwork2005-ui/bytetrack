# CODEX_HANDOFF — điểm bắt đầu cho người review (mốc 2)

Tài liệu này dành cho **người/agent review không có quyền truy cập máy của tác giả**. Mọi đường dẫn là
**tương đối từ gốc repository**. Gói review: `handoff/phase2_review_latest.zip` (xem `handoff/INDEX.md` để
biết file nào có trong gói, `handoff/large_inputs.md` để biết đầu vào lớn nào **chỉ có trên máy** và lệnh
dùng chúng).

## 1. Mục tiêu, phạm vi mốc 2, quyết định đã chốt

**Đề tài**: theo vết đa đối tượng (MOT) cho giao thông hỗn hợp Việt Nam — YOLOv8n + ByteTrack, so sánh
**motion model**: KF–CV (baseline) / EKF–CTRV / UKF–CTRV / PF–CTRV, hướng tới triển khai Jetson Nano.

**Mốc 2 làm gì**: (1) đóng ba kiểm tra tính đúng đắn còn hở; (2) **tìm nguyên nhân mất ID** trên 12 video
dev bằng log quyết định ghép; (3) Nhóm C ba trục (thời gian giữ track, cơ chế nối lại track, tham số nhiễu),
mỗi trục **chỉ đổi một thành phần**; (4) hoàn thiện đối chứng PF + thống kê nhất quán; (5) **khóa cấu hình**
rồi chạy 40 video test UA-DETRAC; (6) chuẩn bị BDD100K + Jetson Nano; (7) bàn giao gói review này.

**Quyết định đã chốt (từ mốc 1, vẫn áp dụng)**:
- Bộ lọc làm việc ở **đơn vị bước** (dt = 1), không có chế độ giây; `track_buffer` quy đổi theo giây →
  bước theo fps; bỏ frame → `n_steps` (xem `src/time_units.py` mục A–D).
- Detection **cache một lần** (conf 0,05, 80 lớp COCO, `max_det` 3000) rồi lọc lúc load → mọi tracker so sánh
  trên **cùng detection**; `data/interim/det_cache/yolov8n-coco_conf0.05_all` (12 video dev + 40 video test).
- Dev = 12 video train UA-DETRAC (phân tầng, seed 0); **test = 40 video test**, chỉ dùng **sau khi khóa**.
- BDD100K: final = val công khai, dev = ~40 video train — **chưa có dữ liệu trên máy**.
- Khoảng trống annotation (**gap**) không được suy diễn là che khuất hoàn toàn (`gap_evidence`).
- Chọn bộ lọc/cấu hình **dựa trên thực nghiệm**; không đặt trước kết luận EKF tốt hơn CV.

## 2. Nhánh, commit, thay đổi chưa commit

- Nhánh: **`phase2-mixed-traffic`**, HEAD **`e761623`** (chỉ commit cục bộ, **không push/merge** — yêu cầu của tác giả).
- Commit của mốc 2 (từ `e353b84` = cuối mốc 1):

```
e761623 README: trang thai GD2 moc 2
225d34e Moc 2: bao cao muc 12-15 (kiem tra tinh dung dan, nguyen nhan mat ID, Nhom C, PF + thong ke), ablation_matrix.py
e69675f Moc 2 (4): chan doan PF theo so buoc mat (ESS, mat da dang hat) + phan tich nguyen nhan cho cac nhanh C2
d8a6c24 Moc 2 (2-5a): ket qua dev Nhom C (C1 buffer, C2 noi lai track, C3 nhieu), phan tich nguyen nhan + ca tieu bieu, khoa cau hinh test
a64cb75 Moc 2 (2-3c): sua truy GT cho refind (lan ghep gan nhat), frame hien tai cho chan doan PF; chon C2 = vong 3 gate Mahalanobis (prediction_drift noi bat), them doi chung IoU
dd67321 Moc 2 (6-7a): mau thi nghiem BDD (chuyen cau hinh vs tuning, tach biet), checklist Nano, pf_diag_report, run_all_tests, make_handoff, large_inputs.md
134d359 Moc 2 (3b): lost_recovery.py - ung vien C2 (lost_no_fuse, vong 3 iou/maha) mac dinh tat = bit-identical; log stage 4; test 11/11
7ca312a Moc 2 (2-3a): dang ky truoc Nhom C (PREREG_moc2.md, C1/C3 YAML), failure_analysis.py, check_identical_runs.py, aggregate: bootstrap COMBINED + pairs
6e28ffa Moc 2 (1): 3 kiem tra tinh dung dan - guard cung bo loc, timestep tich hop, song anh ID; assoc_log, pf_diag, boot_stats
```

- File **tracked** đang sửa (không thuộc công việc mốc 2 — **của người dùng**, không được đóng gói):

```
M .gitignore
 M src/demo/rerun_with_state_log.py
```

- Ngoài ra người dùng có các file **untracked** riêng (`live.bat`, `outputs/**`, `src/demo/*buffer90*`,
  `src/demo/live_view.py`, …) — **không** nằm trong ZIP và không bị sửa.

## 3. Thứ tự đọc

1. `CODEX_HANDOFF.md` (file này).
2. `PHASE2_REPORT.md` — mục **12–16** là mốc 2; mục 1–11 là mốc 1 (giữ nguyên làm nguồn đối chiếu).
3. `configs/phase2/PREREG_moc2.md` — đăng ký trước Nhóm C (giá trị, ngân sách, tiêu chí, giả thuyết).
4. `configs/phase2/LOCK_moc2.md` + `results/phase2/lock/lock_moc2.json` — bản khóa trước khi chạy test.
5. Kết quả: `results/phase2/failure_analysis/expA_detrac_dev/` (nguyên nhân mất ID),
   `results/phase2/ablation_matrix.csv` (Nhóm C), `results/phase2/{expA,expB,expC}_detrac_test/` (test),
   `results/phase2/pf_diag/expA_detrac_dev/` (PF).
6. Code theo thứ tự phụ thuộc: `src/det_cache.py` → `src/run_tracking.py` → `src/tracker_ctrv.py`
   (+ `src/ekf_ctrv.py`, `src/ukf_ctrv.py`, `src/pf_ctrv.py`, `src/lost_association.py`,
   `src/lost_recovery.py`) → `src/run_experiment.py` → `src/run_trackeval.py`, `src/occlusion_eval.py`,
   `src/boot_stats.py`, `src/aggregate_phase2.py` → `src/assoc_log.py`, `src/failure_analysis.py`.
7. `handoff/logs/tests_latest.log` + `handoff/logs/tests_summary.csv` — kết quả test thực tế (exit code).

## 4. Bảng đối chiếu từng yêu cầu của mốc 2

| # | Yêu cầu | Trạng thái | Đã làm | File / bằng chứng | Giới hạn còn lại |
|:--|:--|:--|:--|:--|:--|
| 1a | `multi_predict` dùng `stracks[0].kalman_filter`: pool luôn cùng tracker, cấu hình tương thích | ✅ | Guard fail-loud trong `CTRVSTrack`/`PFSTrack`/`CVSTrack.multi_predict` | `src/tracker_ctrv.py`; `src/test_filter_config.py` test 10 (44/44) | Guard là kiểm tra danh tính instance, không kiểm tra "hai instance có cùng tham số" |
| 1b | Late-binding không rò rỉ giữa các tracker | ✅ | Test chéo A tường minh / B mặc định / UKF | `src/test_filter_config.py` test 11 | — |
| 1c | Timestep timestamp → dự đoán → vòng đời trên đường chạy đầy đủ, cả CV/EKF/UKF/PF | ✅ | Test tích hợp mới qua `run_tracking.track_video` | `src/test_timestep_integration.py` (32/32) | Dữ liệu là cache tổng hợp (UA-DETRAC không bỏ frame); Δt thực của BDD chưa kiểm được trên dữ liệu thật |
| 1d | So sánh tương đương phải kiểm **liên kết ID xuyên suốt sequence** | ✅ | `id_bijection` (song ánh ID toàn sequence) dùng ở 3 nơi | `src/track_compare.py`; `src/check_identical_runs.py`; `results/phase2/expA_detrac_dev/identical_vs_moc1.csv` (288/288) | — |
| 2 | Phân tích sự kiện mất ID, log trước/trong/sau, phân loại nguyên nhân | ✅ | `assoc_log` (không đổi hành vi) + `failure_analysis` 4 tracker × 2 điều kiện | `results/phase2/failure_analysis/expA_detrac_dev/*`; `PHASE2_REPORT.md` mục 13 | Nhãn nguyên nhân là **suy luận có quy tắc** từ số liệu; ngưỡng (IoU < 0,2; conf) ghi trong docstring `src/failure_analysis.py` |
| 3 C1 | Dải thời gian giữ track, cùng buffer khi so CV–EKF, báo cáo phục hồi/FP/IDSW/ghép nhầm | ✅ | 6 mức × 2 bộ lọc, có ghép nhầm | `configs/phase2/experiments/expC1_buffer_dev.yaml`; `PHASE2_REPORT.md` mục 14.1 | Chỉ trên dev; UA-DETRAC là camera cố định |
| 3 C2 | Một thay đổi nhỏ trong cơ chế ghép, giả thuyết ghi trước, có đối chứng CV và EKF | ✅ | Vòng ghép 3 cho track LOST (gate Mahalanobis) + đối chứng IoU | `src/lost_recovery.py`; `configs/phase2/PREREG_moc2.md` mục "C2 — đã chọn"; mục 14.3 | HOTA không cải thiện; chạy test như **nhánh thăm dò** |
| 3 C3 | Tuning tham số nhiễu, ngân sách công bố trước, giữ detector/ghép/bộ đệm cố định | ✅ | 21 cấu hình | `configs/phase2/experiments/expC3_noise_dev.yaml`; mục 14.2 | Lưới CV chỉ có 5 cấu hình phân biệt (bất biến tỉ lệ — mục 14.2) |
| 3 | Ma trận ablation, cấu hình kết hợp | ✅ | `ablation_matrix.csv`; không trục nào đạt tiêu chí → không có tổ hợp để chạy | `src/ablation_matrix.py`; mục 14.4 | Kết quả âm |
| 4 | Dấu hiệu suy giảm PF (ESS, đa dạng hạt, khởi tạo), ≥ 3 seed | ✅ | Chẩn đoán theo `n_lost`, 3 seed × 3 mức N | `src/pf_diag_report.py`; `results/phase2/pf_diag/expA_detrac_dev/`; mục 15.1 | Chưa thử roughening/proposal khác (đổi mô hình nhiễu → thí nghiệm riêng) |
| 4 | Bootstrap ghép cặp theo video, 5000, seed 0, **nhất quán với bảng chính** | ✅ | `boot_stats.py` tái tổ hợp COMBINED; kiểm chứng == TrackEval 1e-9 | `src/boot_stats.py`, `src/test_boot_stats.py`; mục 15.2 | Với PF, thống kê đủ trung bình theo seed trước khi cộng (xấp xỉ) |
| 4 | Nhóm B trình bày theo từng cặp | ✅ | `pairs` trong YAML → `paired_pairs.csv` | `configs/phase2/experiments/{groupB_detrac_dev,expB_detrac_test}.yaml`; mục 15.3 | — |
| 5 | Khóa cấu hình rồi chạy 40 video test | ✅ | `LOCK_moc2.md` + `lock_moc2.json` ghi trước; 3 thí nghiệm test | mục 16 | — |
| 6 | Chuẩn bị BDD100K (2 thí nghiệm tách biệt) + Nano | ⏳ chuẩn bị | YAML `bdd_transfer_val` / `bdd_tuned_val` + checklist Δt/Q; checklist Nano | `configs/phase2/experiments/bdd_*.yaml`; `scripts/nano/README.md` | **Chưa có dữ liệu BDD100K, chưa kết nối Nano** |
| 7 | Bàn giao để review không cần máy | ✅ | File này + ZIP + log test + patch/diff + versions | `handoff/` | Log assoc_log đầy đủ (parquet) giữ tại máy; trong ZIP là bản trích quanh sự kiện |

## 5. Kết luận chính — tách "đo được" / "suy luận" / "chưa xác định"

**Đo được (có số, có CI, dẫn file)**

1. Trên dev 12 video, **CV ≈ EKF ≈ UKF**: ΔHOTA(EKF − CV) = +0,0025 [−0,0018, +0,0060] (COMBINED, bootstrap
   theo video) — CI chứa 0. PF–CTRV kém rõ rệt ở cả 3 mức N (ΔHOTA −0,063 … −0,097; CI không chứa 0).
2. Nguyên nhân thất bại ở **tầng che hoàn toàn** chủ yếu là **dự đoán lệch**: IoU(box dự đoán, det) trung bình
   0,06 sau trung vị 25 bước mất dấu → không thể vượt `match_thresh`. **86 %** ID mới sau khi mất là **track
   mới sinh từ detection**.
3. **Nới `track_buffer` không giúp**: từ 0,8 s đến 4,0 s, tỉ lệ giữ ID **không đổi** trong khi **ghép nhầm
   tăng đơn điệu** (CV 2,5 % → 5,3 %).
4. **Tuning tham số nhiễu không giúp** (21 cấu hình, mọi CI chứa 0). `KalmanFilterXYAH` **bất biến** khi nhân
   cả hai trọng số nhiễu cùng hệ số (bit-identical, 12/12 video).
5. **Thêm một vòng ghép cho track LOST** làm **tăng có ý nghĩa** tỉ lệ giữ ID qua che khuất (+0,049
   [+0,011, +0,080] với CV; tầng che hoàn toàn gần gấp đôi) **nhưng không cải thiện HOTA**, vì tỉ lệ **ghép
   nhầm tăng 3–5 lần**.
6. **PF suy biến đúng lúc cần**: khi track đang mất dấu > 10 bước, chỉ còn ~3 % particle phân biệt, ESS ~1 % N.
7. Mọi kết quả GĐ1 và mốc 1 **tái lập được**: expA (có log) vs mốc 1 bit-identical 288/288 video.

**Suy luận (có cơ sở nhưng không đo trực tiếp)**

- Vì `match_thresh` 0,8 trên cost 1 − IoU·conf, IoU < 0,2 là **rào cứng**: mọi cải thiện motion model chỉ đổi
  vị trí dự đoán vài chục px sẽ không vượt được rào này ở các đoạn che dài → giải thích vì sao CV/EKF/UKF
  gần như hòa nhau trên chỉ số toàn cục.
- Lợi ích của EKF (nếu có) nằm ở nhóm sự kiện nhỏ (che khuất trung bình, quỹ đạo cong) — mẫu tự nhiên của
  UA-DETRAC quá nhỏ để tách (kết luận này đã ghi từ GĐ5).

**Chưa xác định**

- Hành vi trên **giao thông hỗn hợp thật** (xe máy, BDD100K 5 fps, camera chuyển động): chưa có dữ liệu.
- Tốc độ trên **Jetson Nano**: chưa có thiết bị. Mọi con số tốc độ trong repo là trên PC (GTX 1650).
- PF với thiết kế nhiễu khác (roughening/proposal): chưa chạy.
- Detector huấn luyện lại cho 5 lớp: chưa chạy (thí nghiệm riêng, sau khi cố định pipeline tracking).

## 6. Lệnh tái hiện / tiếp tục

Chạy từ **gốc repository**; `P` = python của env `khoaluan-mot` (xem `handoff/versions.json`).

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe    # Windows + Git Bash

# 1. Toàn bộ test (ghi log + exit code)
$P src/run_all_tests.py                      # -> handoff/logs/tests_<ngay>.log, tests_summary.csv

# 2. Tái tạo cache detection (CẦN GPU + ảnh UA-DETRAC; bỏ qua nếu dùng cache trong ZIP)
$P src/det_cache.py --dataset detrac --part train --manifest results/phase2/manifests/detrac_dev.json \
   --cache-id yolov8n-coco_conf0.05_all --skip-existing
$P src/det_cache.py --dataset detrac --part test  --manifest results/phase2/manifests/detrac_test.json \
   --cache-id yolov8n-coco_conf0.05_all --skip-existing

# 3. Chạy lại thí nghiệm (CPU, từ cache): track -> TrackEval -> occlusion_eval -> aggregate
$P src/run_experiment.py --exp configs/phase2/experiments/expA_detrac_dev.yaml
$P src/run_experiment.py --exp configs/phase2/experiments/expC1_buffer_dev.yaml
$P src/run_experiment.py --exp configs/phase2/experiments/expC2_recover_dev.yaml
$P src/run_experiment.py --exp configs/phase2/experiments/expC3_noise_dev.yaml
$P src/run_experiment.py --exp configs/phase2/experiments/expA_detrac_test.yaml     # 40 video test
$P src/run_experiment.py --exp configs/phase2/experiments/expB_detrac_test.yaml
$P src/run_experiment.py --exp configs/phase2/experiments/expC_detrac_test.yaml

# 4. Phân tích nguyên nhân mất ID + ca tiêu biểu + ma trận ablation
$P src/failure_analysis.py --exp configs/phase2/experiments/expA_detrac_dev.yaml --check-events \
   --trackers cv ekf ukf pf500
$P src/representative_cases.py --tracker ekf --cond c25
$P src/ablation_matrix.py

# 5. Chẩn đoán PF, kiểm chứng bit-identical, khóa cấu hình, gói bàn giao
$P src/pf_diag_report.py --exp configs/phase2/experiments/expA_detrac_dev.yaml
$P src/check_identical_runs.py --pairs-from configs/phase2/experiments/expA_detrac_dev.yaml \
   --map "groupA_detrac_dev:A=c25,B=c10"
$P src/verify_shared_kalman_regression.py       # mã trước/sau khi sửa shared_kalman, trên cache thật
$P src/make_lock.py --exps configs/phase2/experiments/expA_detrac_test.yaml \
   configs/phase2/experiments/expB_detrac_test.yaml configs/phase2/experiments/expC_detrac_test.yaml
$P src/make_handoff.py --base e353b84           # -> handoff/phase2_review_latest.zip
```

**Đầu vào cần thiết**: ảnh + annotation UA-DETRAC (`data/extracted/`), `models/yolov8n.pt`
(sha256 `f59b3d83…`), GT MOTChallenge (`data/processed/DETRAC-{all,test}/`) — xem `handoff/large_inputs.md`.
Bước cần **GPU**: tạo cache, `test_run_tracking_replay.py`, `profile_pipeline.py`. Các bước khác chạy CPU.

## 7. Những điểm nên kiểm tra kỹ

1. **Nhãn nguyên nhân** trong `src/failure_analysis.py` (`analyze_event`): thứ tự ưu tiên, ngưỡng IoU 0,2,
   cách xác định "detector thấy lại" (IoU ≥ 0,5 với GT, tier high) — đây là chỗ dễ tranh luận nhất.
2. **`boot_stats.combine_stacked`** có tái tổ hợp đúng công thức TrackEval không (so với
   `trackeval/metrics/{hota,clear,identity}.py`); test `test_boot_stats.py` chỉ kiểm trên 4 video.
3. **`lost_recovery.LostRecoveryMixin`**: vòng ghép 3 đặt ở `_post_first_association` — kiểm tra nó không
   phá thứ tự vòng 2 / unconfirmed / sinh track mới của ByteTrack, và mặc định tắt là bit-identical.
4. **`assoc_log`**: cách suy ra "track được ghép ở vòng nào" (so `track.frame_id` và `track.idx` sau mỗi
   vòng) — có thể bỏ sót trường hợp hiếm; kiểm tra lại với `stage == 0` nhưng `matched_det >= 0`.
5. **Phân biệt tên**: nhóm thí nghiệm A/B/C vs điều kiện confidence `c25`/`c10` (mốc 1 dùng `A`/`B` cho
   điều kiện — đã đổi tên từ mốc 2, ánh xạ A→c25, B→c10).
6. **Bản khóa**: `results/phase2/lock/lock_moc2.json` có `created` **trước** `created` trong mọi
   `run_meta.json` của thư mục `p2-exp*_detrac_test-*` không.
7. **Kết quả âm**: đọc kỹ mục 14 — không trục nào của Nhóm C đạt tiêu chí đăng ký trước; nhánh C2 trên test
   là **thăm dò**, không phải "cấu hình tốt nhất".

## 8. Kết quả trên 40 video test UA-DETRAC

_(dang chay - se cap nhat)_

