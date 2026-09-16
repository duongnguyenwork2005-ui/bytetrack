# CODEX_HANDOFF — điểm bắt đầu cho người review (mốc 2, bản sau review vòng 1)

Tài liệu này dành cho **người/agent review không có quyền truy cập máy của tác giả**. Mọi đường dẫn là
**tương đối từ gốc repository**. Gói review: `handoff/phase2_review_latest.zip` (xem `handoff/INDEX.md` để
biết file nào có trong gói, `handoff/FILE_HASHES.csv` để đối chiếu sha256 từng file, `handoff/large_inputs.md`
để biết đầu vào lớn nào **chỉ có trên máy**, cách nhận diện và lệnh dùng chúng).

## 0. Đợt sửa sau review độc lập (Codex) — đọc trước

Review của Codex trên snapshot `502f011` nêu 4 lỗi (2×P1, 2×P2) + 6 điểm diễn giải. **Cả 4 lỗi đã được tái
hiện trên mã hiện tại và đã sửa**; phần Nhóm B bị ảnh hưởng **đã chạy lại** trong thư mục mới. Bảng trạng thái
đầy đủ: `results/phase2/REVIEW_moc2_codex.md`; chi tiết + số liệu trước/sau: `PHASE2_REPORT.md` **mục 18**.

| # | Phát hiện | Trạng thái | Bằng chứng |
|:--|:--|:--|:--|
| P1 | Baseline "SORT nguyên bản" xuất track sớm hơn upstream 1 frame (`hits`/`hit_streak` khởi tạo 1 thay vì 0) | **Đã sửa + chạy lại Nhóm B** | `src/sort_tracker.py`; `src/test_sort_upstream_parity.py` **8/8** so toàn chuỗi với mã upstream đã ghim (`third_party/sort_upstream`, commit `bce9f0d`), lệch toạ độ tối đa **5,8·10⁻⁵ px**, ánh xạ ID đồng nhất; kết quả mới: `results/phase2/expB_detrac_{dev,test}_fix1/` |
| P1 | Runner có thể chấm tracking mới bằng metric cũ; `--no-skip` không tới bước đánh giá; `run_meta` bị ghi đè provenance; lỗi subprocess không thành exit code | **Đã sửa** | `src/run_fingerprint.py` + `src/run_experiment.py`; `src/test_run_experiment_guard.py` **7/7** |
| P2 | Cache có thể ghi thêm với `conf`/`classes`/`max_det`/phiên bản khác mà metadata giữ nguyên | **Đã sửa** | `src/det_cache.py` (`STRICT_FIELDS`, `spec_sha256` từng sequence, `write_meta(mode=)`); `src/test_det_cache.py` **47/47**; kiểm cache hiện có: `results/phase2/cache_consistency.csv` — **0 mismatch**, không tái tạo |
| P2 | ZIP chứa `.gitignore`; commit trong ZIP lệch tin nhắn (ZIP dựng trước commit) | **Đã sửa** | `src/make_handoff.py`: `USER_FILES`/`USER_PREFIXES`, **từ chối dựng khi còn file khác HEAD**, `handoff/FILE_HASHES.csv`, `handoff/external/bytetrack.yaml` |
| — | 6 điểm diễn giải (Nhóm B, PF, C2, C3, `has_full`, tổng hợp PF theo seed) | **Đã sửa trong báo cáo** | `PHASE2_REPORT.md` mục 7.2, 13.1, 14.2, 15.1, 15.2, 16.2, 16.3, 16.4, 18.6 |

**Kết quả Nhóm B sau khi sửa** (cấu hình y hệt bản đã khóa, chỉ đổi `name`; không tuning):

| cặp (yếu tố được kiểm soát) — TEST 40 video, c25 | trước khi sửa | **sau khi sửa** |
|:--|:--|:--|
| bộ đệm `max_age` 1 → 30 | +0.0280 [+0.0207, +0.0350] | **+0.0274 [+0.0201, +0.0345]** |
| gói biểu diễn + thiết lập bộ lọc (KF 7-D → XYAH) | −0.0133 [−0.0185, −0.0077] | **−0.0132 [−0.0185, −0.0077]** |
| motion model CV → EKF–CTRV | +0.0020 [−0.0001, +0.0042] | **+0.0020 [−0.0001, +0.0042]** |
| chỉ đổi bộ lọc ở `max_age` 1 | −0.0140 [−0.0167, −0.0116] | **−0.0140 [−0.0167, −0.0116]** |
| vòng đời SORT → ByteTrack | +0.0364 [+0.0298, +0.0428] | **+0.0363 [+0.0296, +0.0428]** |

IDSW giảm 9–12 % ở mọi nhánh SORT (track xuất muộn hơn 1 frame ⇒ ít track "non"); **mọi kết luận định tính
không đổi**. `bt-ekf` (ByteTrack) **bit-identical** bản cũ: 12/12 video dev, 40/40 test → bản sửa không đụng
ByteTrack, nên **không** cần chạy lại Nhóm A, Nhóm C hay Giai đoạn 1 (đã kiểm: chỉ Nhóm B dùng `SORTTrack`).


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

- Nhánh: **`phase2-mixed-traffic`**, HEAD **`60c946d`** (chỉ commit cục bộ, **không push/merge** — yêu cầu của tác giả).
- Commit của mốc 2 (từ `e353b84` = cuối mốc 1):

```
60c946d Review-fix (7b): CODEX_HANDOFF ban sau review vong 1 (muc 0 trang thai phat hien, lenh thuc chay + exit code, provenance/hash), luu ZIP cu vao handoff/archive
fb360e0 Review-fix (4,6,7a): chay lai Nhom B sau khi sua SORT (dev+test, thu muc _fix1), sua dien giai, dong goi loai file nguoi dung
9ec7d8e Review-fix (5): chinh sach cache - STRICT_FIELDS them conf/classes/max_det, fingerprint tung sequence, tach GHI vs DOC
1c7ce8c Review-fix (1-3): runner fingerprint + provenance + exit code; SORT vong doi khop upstream da ghim
28ca79b Moc 2 (7): dung lai goi review voi ket qua test cuoi cung; sua 2 tham chieu treo trong CODEX_HANDOFF
502f011 Moc 2 (5): ket qua 40 video test UA-DETRAC (A/B/C), muc 16-17 bao cao, CODEX_HANDOFF ban cuoi
76748c8 Moc 2 (lam ro 1-4): verify_c3_effect (override vao bo loc + covariance khac nhung dau ra gan nhu trung), cause_origin_table (nguyen nhan vs nguon ID sau), dinh nghia ESS/n_unique + cau hinh PF, don vi dem cua check_identical_runs
a5730f1 Moc 2 (7a): CODEX_HANDOFF.md, gioi han dong goi, ZIP review (tao boi src/make_handoff.py; ZIP khong commit)
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
 M handoff/INDEX.md
 M handoff/patches/README.md
 M handoff/patches/commits.txt
 M handoff/versions.json
 M src/demo/rerun_with_state_log.py
 M src/make_handoff.py
```

- Ngoài ra người dùng có các file **untracked** riêng (`live.bat`, `outputs/**`, `src/demo/*buffer90*`,
  `src/demo/live_view.py`, …) — **không** nằm trong ZIP và không bị sửa.

## 3. Thứ tự đọc

1. `CODEX_HANDOFF.md` (file này).
2. `results/phase2/REVIEW_moc2_codex.md` — trạng thái từng phát hiện của review vòng 1.
3. `PHASE2_REPORT.md` — mục **18** (đợt sửa sau review, có bảng trước/sau), rồi mục **12–17** (mốc 2),
   mục 1–11 là mốc 1 (giữ nguyên làm nguồn đối chiếu).
4. `configs/phase2/PREREG_moc2.md` — đăng ký trước Nhóm C (giá trị, ngân sách, tiêu chí, giả thuyết).
5. `configs/phase2/LOCK_moc2.md` + `results/phase2/lock/lock_moc2.json` — bản khóa trước khi chạy test.
6. Kết quả: `results/phase2/expB_detrac_dev_fix1/`, `results/phase2/expB_detrac_test_fix1/` (Nhóm B **sau khi
   sửa** — dùng cho luận văn), `results/phase2/failure_analysis/expA_detrac_dev/` (nguyên nhân mất ID),
   `results/phase2/ablation_matrix.csv` (Nhóm C), `results/phase2/expA_detrac_test/`,
   `results/phase2/expB_detrac_test/`, `results/phase2/expC_detrac_test/` (test),
   `results/phase2/pf_diag/expA_detrac_dev/` (PF), `results/phase2/expC3_noise_dev/c3_effect_check.csv`.
7. Code theo thứ tự phụ thuộc: `src/det_cache.py` → `src/run_tracking.py` → `src/tracker_ctrv.py`
   (+ `src/ekf_ctrv.py`, `src/ukf_ctrv.py`, `src/pf_ctrv.py`, `src/lost_association.py`,
   `src/lost_recovery.py`) → `src/run_experiment.py` → `src/run_trackeval.py`, `src/occlusion_eval.py`,
   `src/boot_stats.py`, `src/aggregate_phase2.py` → `src/assoc_log.py`, `src/failure_analysis.py`;
   **mới sau review**: `src/run_fingerprint.py`, `src/check_cache_consistency.py`,
   `src/test_sort_upstream_parity.py`, `src/test_run_experiment_guard.py`, `third_party/sort_upstream/`.
8. `handoff/logs/tests_latest.log` + `handoff/logs/tests_summary.csv` — kết quả test thực tế (exit code).

## 4. Bảng đối chiếu từng yêu cầu của mốc 2

| # | Yêu cầu | Trạng thái | Đã làm | File / bằng chứng | Giới hạn còn lại |
|:--|:--|:--|:--|:--|:--|
| 1a | `multi_predict` dùng `stracks[0].kalman_filter`: pool luôn cùng tracker, cấu hình tương thích | ✅ | Guard fail-loud trong `CTRVSTrack`/`PFSTrack`/`CVSTrack.multi_predict` | `src/tracker_ctrv.py`; `src/test_filter_config.py` test 10 (44/44) | Guard kiểm tra danh tính instance, không kiểm tra "hai instance có cùng tham số" |
| 1b | Late-binding không rò rỉ giữa các tracker | ✅ | Test chéo A tường minh / B mặc định / UKF | `src/test_filter_config.py` test 11 | — |
| 1c | Timestep timestamp → dự đoán → vòng đời trên đường chạy đầy đủ, cả CV/EKF/UKF/PF | ✅ | Test tích hợp qua `run_tracking.track_video` | `src/test_timestep_integration.py` (32/32) | Dữ liệu là cache tổng hợp (UA-DETRAC không bỏ frame); Δt thực của BDD chưa kiểm trên dữ liệu thật |
| 1d | So sánh tương đương phải kiểm **liên kết ID xuyên suốt sequence** | ✅ | `id_bijection`; định nghĩa và đơn vị đếm ghi ở `PHASE2_REPORT.md` mục 12.1 | `src/track_compare.py`; `src/check_identical_runs.py`; `results/phase2/expA_detrac_dev/identical_vs_moc1.csv` | — |
| 2 | Phân tích sự kiện mất ID, log trước/trong/sau, phân loại nguyên nhân, giữ `unknown` | ✅ | `assoc_log` (không đổi hành vi) + `failure_analysis` 4 tracker × 2 điều kiện; **tách nguyên nhân (track cũ) khỏi nguồn ID sau** (`cause_origin_table.py`) | `results/phase2/failure_analysis/expA_detrac_dev/*` (`cause_summary.csv`, `cause_by_origin_c25.csv`, `cause_evidence_c25.csv`, `event_window_log.csv`, `representative_cases_*.md`); mục 13 | Nhãn nguyên nhân là **suy luận có quy tắc** từ số liệu; ngưỡng ghi trong docstring `src/failure_analysis.py` |
| 3 C1 | Dải thời gian giữ track, cùng buffer khi so CV–EKF, báo cáo phục hồi/FP/IDSW/ghép nhầm | ✅ | 6 mức × 2 bộ lọc, có ghép nhầm | `configs/phase2/experiments/expC1_buffer_dev.yaml`; mục 14.1 | Chỉ trên dev; UA-DETRAC là camera cố định |
| 3 C2 | Một thay đổi nhỏ trong cơ chế ghép, giả thuyết ghi trước, đối chứng CV và EKF | ✅ | Vòng ghép 3 cho track LOST (gate Mahalanobis) + đối chứng IoU | `src/lost_recovery.py`; `configs/phase2/PREREG_moc2.md` mục "C2 — đã chọn"; mục 14.3, 16.4 | Không đạt tiêu chí HOTA trên dev; **trên test HOTA giảm có ý nghĩa** |
| 3 C3 | Tuning tham số nhiễu, ngân sách công bố trước, giữ detector/ghép/bộ đệm cố định | ✅ | 21 cấu hình + **kiểm chứng override có hiệu lực** (`verify_c3_effect.py`) | `configs/phase2/experiments/expC3_noise_dev.yaml`; `results/phase2/expC3_noise_dev/c3_effect_check.csv`; mục 14.2 | Lưới CV chỉ có 5 cấu hình phân biệt (bất biến tỉ lệ) |
| 3 | Ma trận ablation, cấu hình kết hợp | ✅ | `ablation_matrix.csv`; không trục nào đạt tiêu chí → không có tổ hợp để chạy | `src/ablation_matrix.py`; mục 14.4 | Kết quả âm |
| 4 | Dấu hiệu suy giảm PF (ESS, đa dạng hạt, khởi tạo), ≥ 3 seed, ghi rõ cách tính | ✅ | Chẩn đoán theo `n_lost`, 3 seed × 3 mức N; định nghĩa + thời điểm đo + cấu hình ghi trong `PFCTRV.enable_diag` và mục 15.1 | `src/pf_diag_report.py`; `results/phase2/pf_diag/expA_detrac_dev/` | Kết luận giới hạn trong implementation bootstrap PF này và các cấu hình đã thử |
| 4 | Bootstrap ghép cặp theo video, 5000, seed 0, **nhất quán với bảng chính** | ✅ | `boot_stats.py` tái tổ hợp COMBINED; kiểm chứng == TrackEval 1e-9 | `src/boot_stats.py`, `src/test_boot_stats.py`; mục 15.2 | Với PF, thống kê đủ trung bình theo seed trước khi cộng (xấp xỉ) |
| 4 | Nhóm B trình bày theo từng cặp | ✅ | `pairs` trong YAML → `paired_pairs.csv` | `configs/phase2/experiments/{groupB_detrac_dev,expB_detrac_test}.yaml`; mục 15.3, 16.3 | — |
| 5 | Khóa cấu hình rồi chạy 40 video test, báo cáo đủ kể cả kết quả không cải thiện | ✅ | `LOCK_moc2.md` + `lock_moc2.json` ghi trước (kiểm chứng mốc thời gian); 3 thí nghiệm test (22 lần chạy) | mục 16 | — |
| 6 | Chuẩn bị BDD100K (2 thí nghiệm tách biệt) + Nano | ⏳ chuẩn bị | YAML `bdd_transfer_val` / `bdd_tuned_val` + checklist Δt/Q; checklist Nano 5 bước | mục 17; `scripts/nano/README.md` | **Chưa có dữ liệu BDD100K, chưa kết nối Nano** |
| 7 | Bàn giao để review không cần máy | ✅ | File này + ZIP + log test + patch/diff + versions | `handoff/` | Log `assoc_log` đầy đủ (parquet) giữ tại máy; trong ZIP là bản trích quanh sự kiện |

## 5. Kết luận chính — tách "đo được" / "suy luận" / "chưa xác định"

**Đo được (có số, có CI, dẫn file)**

1. **Trên 40 video test**: ở `c10`, **EKF–CTRV hơn KF–CV có ý nghĩa** (ΔHOTA +0.0033 [+0.0011, +0.0057],
   ΔIDF1 +0.0053, ΔAssA +0.0072); ở `c25` CI vẫn chứa 0 (+0.0012 [−0.0003, +0.0027]). Độ lớn rất nhỏ
   (≈ +0,6 % tương đối). UKF **kém** EKF một chút có ý nghĩa ở c25 (−0.0011 [−0.0019, −0.0002]).
   PF–CTRV kém rõ rệt (−0.0316 [−0.0389, −0.0255] so với EKF).
2. **Thứ tự ảnh hưởng (test, Nhóm B)**: vòng đời/association SORT → ByteTrack **+0.0364** > bộ đệm 1 → 30
   **+0.0280** > biểu diễn trạng thái KF 7-D → XYAH **−0.0133** ≫ motion model CV → EKF **+0.0020** (CI chạm 0).
3. **Nguyên nhân mất ID** (dev, 184 sự kiện): ở tầng che hoàn toàn, **dự đoán lệch** là số 1 (cv 11/22,
   ekf 13/23 ca) — IoU(dự đoán, det) trung bình 0,06 sau trung vị 25 bước mất dấu; `detector_miss` **không**
   xuất hiện ở tầng full; 86 % ID mới là **track mới sinh từ detection**.
4. **Nới `track_buffer` không giúp**: 0,8 → 4,0 s giữ nguyên tỉ lệ giữ ID trong khi **ghép nhầm tăng đơn điệu**
   (CV 2,5 % → 5,3 %).
5. **Tuning tham số nhiễu không giúp** (21 cấu hình, mọi CI chứa 0) — dù tham số **thực sự có hiệu lực**
   (covariance đổi rõ: `P[θ,θ]` trung bình 2,46 / 2,83 / 3,88 theo `std_omega`) và đầu ra **không** giống hệt
   (0/12 bit-identical; ~99 % dòng lệch dưới 1 px với EKF).
6. **Thêm vòng ghép cho track LOST**: giữ ID qua che khuất tăng có ý nghĩa (dev +0.049; test +0.036; tầng che
   hoàn toàn **gấp đôi**: 0,060 → 0,121) **nhưng HOTA giảm có ý nghĩa trên test** (−0.0046 / −0.0034) vì ghép
   nhầm tăng 3–5 lần (3,5 % → 9,3 %).
7. **PF suy biến đúng lúc cần**: khi track mất dấu > 10 bước, chỉ còn ~3 % particle phân biệt (sau resample),
   ESS/N ~1 % (trước resample).
8. Mọi kết quả GĐ1 và mốc 1 **tái lập được**: expA (có log) vs mốc 1 **288/288 cặp (lần chạy, video)**
   bit-identical **và** song ánh ID.

**Suy luận (có cơ sở nhưng không đo trực tiếp)**

- Vì `match_thresh` = 0,8 trên cost 1 − IoU·conf, IoU < 0,2 là **rào cứng**: cải thiện motion model chỉ dịch
  dự đoán vài chục px sẽ không vượt được rào này ở các đoạn che dài → giải thích vì sao CV/EKF/UKF gần như hòa
  nhau trên chỉ số toàn cục, và vì sao tầng che hoàn toàn đứng yên ở ~6 % với mọi bộ lọc.
- Lợi ích của EKF (nếu có) tập trung ở nhóm sự kiện nhỏ (che khuất trung bình, quỹ đạo cong) — mẫu tự nhiên
  của UA-DETRAC quá nhỏ để tách (đã ghi từ GĐ5).

**Chưa xác định**

- Hành vi trên **giao thông hỗn hợp thật** (xe máy, BDD100K 5 fps, camera chuyển động): chưa có dữ liệu.
- Tốc độ trên **Jetson Nano**: chưa có thiết bị. Mọi con số tốc độ trong repo là trên PC (GTX 1650).
- PF với thiết kế nhiễu khác (roughening/proposal/Q riêng): chưa chạy.
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

# 4. Phân tích nguyên nhân mất ID + ca tiêu biểu + bảng chéo + ma trận ablation
$P src/failure_analysis.py --exp configs/phase2/experiments/expA_detrac_dev.yaml --check-events \
   --trackers cv ekf ukf pf500
$P src/cause_origin_table.py --cond c25
$P src/representative_cases.py --tracker ekf --cond c25
$P src/ablation_matrix.py

# 5. Kiểm chứng: PF, C3, bit-identical, khóa cấu hình, gói bàn giao
$P src/pf_diag_report.py --exp configs/phase2/experiments/expA_detrac_dev.yaml
$P src/verify_c3_effect.py                       # override co vao bo loc? covariance doi? dau ra khac bao nhieu?
$P src/check_identical_runs.py --pairs-from configs/phase2/experiments/expA_detrac_dev.yaml \
   --map "groupA_detrac_dev:A=c25,B=c10"
$P src/verify_shared_kalman_regression.py        # mã trước/sau khi sửa shared_kalman, trên cache thật
$P src/make_lock.py --exps configs/phase2/experiments/expA_detrac_test.yaml \
   configs/phase2/experiments/expB_detrac_test.yaml configs/phase2/experiments/expC_detrac_test.yaml
$P src/make_handoff.py --base e353b84            # -> handoff/phase2_review_latest.zip
```

**Đầu vào cần thiết**: ảnh + annotation UA-DETRAC (`data/extracted/`), `models/yolov8n.pt`
(sha256 `f59b3d83…`), GT MOTChallenge (`data/processed/DETRAC-all/`, `data/processed/DETRAC-test/`) — xem
`handoff/large_inputs.md`. Bước cần **GPU**: tạo cache, `test_run_tracking_replay.py`, `profile_pipeline.py`.
Các bước khác chạy CPU từ cache (cache có trong ZIP nếu ≤ 60 MB — xem `handoff/INDEX.md`).

## 6b. Lệnh THỰC CHẠY trong đợt sửa này (log + exit code)

| Lệnh | Exit | Log |
|:--|:--|:--|
| `python src/run_experiment.py --exp configs/phase2/experiments/expB_detrac_dev_fix1.yaml` | 0 | `handoff/logs/expB_detrac_dev_fix1.log` |
| `python src/run_experiment.py --exp configs/phase2/experiments/expB_detrac_test_fix1.yaml` | 0 | `handoff/logs/expB_detrac_test_fix1.log` |
| `python src/check_identical_runs.py --split DETRAC-all --a p2-groupB_detrac_dev-A-bt-ekf --b p2-expB_detrac_dev_fix1-A-bt-ekf` | 0 | 12/12 bit-identical |
| `python src/check_identical_runs.py --split DETRAC-test --a p2-expB_detrac_test-c25-bt-ekf --b p2-expB_detrac_test_fix1-c25-bt-ekf` | 0 | 40/40 bit-identical |
| `python src/check_cache_consistency.py` | 0 | `results/phase2/cache_consistency.csv` (0 mismatch) |
| `python src/run_all_tests.py` | 0 | `handoff/logs/tests_latest.log`, `tests_summary.csv` — **19/19 PASS**, 0 FAIL, 0 SKIPPED |

Toàn bộ log của mọi lần chạy (kể cả các lần của mốc 2) nằm trong `handoff/logs/`; mỗi file kết thúc bằng dòng
`exit=<mã>`.

## 6c. Provenance và hash cần để tái lập

| Thành phần | Cách xác định |
|:--|:--|
| Mã nguồn | commit trong `handoff/versions.json` (`git_head`) + `handoff/patches/moc2_*.diff` + `commits.txt` |
| File trong gói | `handoff/FILE_HASHES.csv` (sha256 từng file) |
| Cấu hình đã khóa | `results/phase2/lock/lock_moc2.json` (sha256 từng YAML, 40 sequence, cache meta) |
| Detector | `models/yolov8n.pt` sha256 `f59b3d83…` (ghi trong `meta.json` của cache và trong lock) |
| Cache detection | `data/interim/det_cache/<id>/meta.json` (+ `spec_sha256` từ nay) và từng `<video>.json` |
| SORT upstream | `third_party/sort_upstream/PROVENANCE.json` (commit `bce9f0d`, sha256 `f5b10481…`) |
| `bytetrack.yaml` của ultralytics | `handoff/external/bytetrack.yaml` (bản đang dùng, để đối chiếu hash trong lock) |
| Mỗi lần chạy | `data/processed/trackers/<split>/<dir>/run_meta.json` (cfg, cache, commit, `created`, `input_fingerprint`, `resumed`) và `results/trackeval/<split>/<dir>/eval_meta.json` (dấu vân tay đánh giá) |

## 7. Những điểm nên kiểm tra kỹ

1. **Nhãn nguyên nhân** trong `src/failure_analysis.py` (`analyze_event`): thứ tự ưu tiên, ngưỡng IoU 0,2,
   cách xác định "detector thấy lại" (IoU ≥ 0,5 với GT, tier high) — chỗ dễ tranh luận nhất. Bảng ánh xạ
   nhãn ↔ cột số liệu: `results/phase2/failure_analysis/expA_detrac_dev/cause_evidence_c25.csv`.
2. **`boot_stats.combine_stacked`** có tái tổ hợp đúng công thức TrackEval không (so với
   `trackeval/metrics/{hota,clear,identity}.py`); `test_boot_stats.py` chỉ kiểm trên 4 video.
3. **`lost_recovery.LostRecoveryMixin`**: vòng ghép 3 đặt ở `_post_first_association` — kiểm tra nó không phá
   thứ tự vòng 2 / unconfirmed / sinh track mới của ByteTrack, và mặc định tắt là bit-identical.
4. **`assoc_log`**: cách suy ra "track được ghép ở vòng nào" (so `track.frame_id` và `track.idx` sau mỗi vòng)
   — kiểm tra trường hợp `stage == 0` nhưng `matched_det >= 0`.
5. **Phân biệt tên**: nhóm thí nghiệm A/B/C vs điều kiện confidence `c25`/`c10` (mốc 1 dùng `A`/`B` cho điều
   kiện; từ mốc 2 đổi tên, ánh xạ A→c25, B→c10).
6. **Bản khóa**: trường `created` của `results/phase2/lock/lock_moc2.json` (02:33:51) **trước** mọi `run_meta.json.created`
   của `p2-exp*_detrac_test-*` (sớm nhất 02:44:45) — đã kiểm, xem mục 16.1 của báo cáo.
7. **Kết quả âm**: mục 14 (không trục Nhóm C nào đạt tiêu chí) và mục 16.4 (nhánh thăm dò C2 làm **giảm** HOTA
   trên test) — đọc kỹ để không hiểu nhầm là "cấu hình tốt nhất".
8. **Hai chỉ số ngược chiều**: `preserved` (chỉ số sự kiện) và HOTA (chỉ số toàn cục) có thể đi ngược nhau
   (mục 14.3, 16.4) — mọi kết luận phải nói rõ đang dùng chỉ số nào.
9. **(sau review vòng 1)** `src/run_fingerprint.py`: danh sách module trong `CODE_FILES` có đủ để phát hiện
   mọi thay đổi ảnh hưởng kết quả không; `eval_meta.json`/`occl_meta.json` có bị bỏ qua ở nhánh nào không.
10. **(sau review vòng 1)** `src/test_sort_upstream_parity.py`: cách stub `matplotlib`/`skimage`, và hai khác
   biệt có chủ ý so với upstream (lọc `conf > det_thresh`, xử lý track NaN) — xem `PROVENANCE.json`.
11. **(sau review vòng 1)** `src/det_cache.py`: chính sách `STRICT_FIELDS` vs `WARN_FIELDS` có hợp lý không;
   các cache `legacy` (chưa có `spec_sha256`) được xử lý ra sao khi đọc.

## 8. Kết quả trên 40 video test UA-DETRAC

Cấu hình được **khóa trước** khi chạy (`configs/phase2/LOCK_moc2.md`, `results/phase2/lock/lock_moc2.json`,
commit `a64cb754`, ghi lúc **02:33:51**); lần chạy test sớm nhất lúc **02:44:45** → **22/22 lần chạy sau khi
khóa**. Không có tuning hay ablation mới sau khi xem kết quả test. Chi tiết: `PHASE2_REPORT.md` mục 16.

**A — bộ lọc** (40 video, 1324 sự kiện che khuất; COMBINED + CI 95 % bootstrap ghép cặp theo video):

| cặp | c25 (conf > 0,25) | c10 (conf > 0,10) |
|:--|:--|:--|
| EKF − CV (ΔHOTA) | +0.0012 [−0.0003, +0.0027] | **+0.0033 [+0.0011, +0.0057]** |
| UKF − CV | +0.0001 [−0.0012, +0.0014] | +0.0019 [−0.0002, +0.0041] |
| UKF − EKF | **−0.0011 [−0.0019, −0.0002]** | −0.0015 [−0.0040, +0.0007] |
| PF500 − EKF | **−0.0316 [−0.0389, −0.0255]** | **−0.0325 [−0.0394, −0.0264]** |

HOTA tuyệt đối (c25): cv 0.5824, ekf 0.5835, ukf 0.5824, pf500 0.5519. Giữ ID qua che khuất: cv 0.480,
ekf 0.487, ukf 0.485, pf500 0.435; **tầng che hoàn toàn (182 sự kiện): cả ba bộ lọc đều 0,060**.

**B — thuật toán** (c25, mỗi cặp chỉ đổi một yếu tố): bộ đệm 1 → 30 **+0.0280 [+0.0207, +0.0350]**;
biểu diễn trạng thái KF 7-D → XYAH **−0.0133 [−0.0185, −0.0077]**; motion model CV → EKF–CTRV +0.0020
[−0.0001, +0.0042] (IDSW −84 [−162, −4]); vòng đời SORT → ByteTrack **+0.0364 [+0.0298, +0.0428]**.

**C — nhánh thăm dò** (vòng ghép thứ 3 cho track LOST, gate Mahalanobis; **không** phải cấu hình được chọn):
giữ ID qua che khuất **tăng có ý nghĩa** (+0.036 [+0.019, +0.051] với CV; tầng che hoàn toàn 0,060 → **0,121**)
nhưng HOTA **giảm có ý nghĩa** (cv −0.0046 [−0.0072, −0.0021]; ekf −0.0034 [−0.0058, −0.0012]) vì FP +8 000 và
IDSW +261…+319 → **kết quả âm, không bật cơ chế này nếu mục tiêu là HOTA**.

**Tốc độ tracker từ cache** (56 340 frame, CPU, không có detector, đo khi máy còn chạy việc khác → tham khảo):
sort-orig1 774 fps, bt-ekf 498, cv 420–570, ekf 333–478, ukf 175–187, pf500 106–114. Toàn pipeline có detector
(GTX 1650): 45–51 fps (mục 7.3). **Chưa đo trên Nano.**

Kết quả đầy đủ: `results/phase2/{expA,expB,expC}_detrac_test/<cond>/{summary.csv, paired_bootstrap_combined.csv,
paired_pairs.csv, occlusion_strata.csv, bootstrap_DETRAC-test.csv}`, log chạy trong `handoff/logs/exp*_detrac_test.log`.

