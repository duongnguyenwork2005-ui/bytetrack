# CODEX_HANDOFF — điểm bắt đầu cho người review (mốc 2 đã chốt, đang chuẩn bị mốc 3)

Tài liệu này dành cho **người/agent review không có quyền truy cập máy của tác giả**. Mọi đường dẫn là
**tương đối từ gốc repository**.

> **Bàn giao có HAI mức — vòng sửa thông thường KHÔNG gửi lại cả repository.**
>
> | | Gói **DELTA** (mặc định mỗi vòng sửa) | Gói **ĐẦY ĐỦ** (mốc lớn / lưu trữ) |
> |:--|:--|:--|
> | File | `handoff/review_delta_latest.zip` | `handoff/phase2_review_latest.zip` |
> | Dựng bằng | `python src/make_review_delta.py` | `python src/make_handoff.py --base e353b84` |
> | Nội dung | `REVIEW_BRIEF.md`, patch so với **đúng snapshot đã đọc**, file code/test/config thêm-sửa trong vòng, log kiểm thử mới, CSV/JSON mới, `MANIFEST.csv`/`.json` (base/target + thêm/sửa/xóa + sha256), `DEPENDENCIES.md` | toàn bộ mã, cấu hình, kết quả, cache detection, log |
> | Đọc theo thứ tự | `REVIEW_BRIEF.md` → `patch/` → `logs/` + test → `data/` → `DEPENDENCIES.md` | tài liệu này → `PHASE2_REPORT.md` → mục 3 dưới đây |
>
> **Base review không tự chạy theo HEAD.** Snapshot mà người review đã đọc được ghi trong
> `handoff/review_base.json` và chỉ đổi khi có xác nhận (`make_review_delta.py --set-base <commit> --round <n>`)
> — dựng một gói mới **không** làm base nhảy sang HEAD.
>
> `DEPENDENCIES.md` của gói delta liệt kê file cần nhưng **không** đóng gói, kèm **vị trí trong snapshot base**
> và sha256. **Không giả định người review còn giữ bản gốc**: thiếu file nào, báo lại để bổ sung đúng file đó
> ở gói delta kế tiếp.

Gói đầy đủ: xem `handoff/INDEX.md` để biết file nào có trong gói, `handoff/FILE_HASHES.csv` để đối chiếu
sha256 từng file, `handoff/large_inputs.md` để biết đầu vào lớn nào **chỉ có trên máy**, cách nhận diện và
lệnh dùng chúng.

## 0. Trạng thái sau HAI vòng review độc lập — đọc trước

| Vòng | Phát hiện | Trạng thái | Bằng chứng |
|:--|:--|:--|:--|
| 1 (P1) | Baseline "SORT nguyên bản" xuất track sớm hơn upstream 1 frame | **Đã sửa + chạy lại Nhóm B** | `src/test_sort_upstream_parity.py` **8/8** so toàn chuỗi với mã upstream ghim (`third_party/sort_upstream`, commit `bce9f0d`), lệch ≤ **5,8·10⁻⁵ px**; kết quả `results/phase2/expB_detrac_{dev,test}_fix1/` |
| 1 (P1) | Runner chấm tracking mới bằng metric cũ | **Đã sửa** | `src/run_fingerprint.py`; `src/test_run_experiment_guard.py` **7/7** |
| 1 (P2) | Cache ghi thêm với `conf`/`classes`/`max_det` khác | **Đã sửa** | `src/det_cache.py`; `src/test_det_cache.py` **51/51** |
| 1 (P2) | ZIP chứa `.gitignore`, commit lệch | **Đã sửa** | `src/make_handoff.py` (`USER_FILES`, từ chối dựng khi dirty, `FILE_HASHES.csv`) |
| **2 (P1)** | **Resume sau khi bị gián đoạn / thiếu metadata có thể trộn cấu hình** | **Đã tái hiện + sửa** | Trạng thái **theo từng video** (`video_meta/<video>.json`, ghi nguyên tử) + `run_meta.status=running` trước khi ghi đè; `src/test_run_resume_guard.py` **10/10** qua đường chạy runner thật với lỗi giả lập |
| **2 (P2)** | **Cache cho ghi khi đổi phiên bản thư viện rồi từ chối đọc** | **Đã tái hiện + sửa** | `GEN_FIELDS` là **một** định nghĩa dùng cho cả đường ghi và đường đọc (gồm `versions`, `device`; bỏ đường dẫn `weights`); `src/test_det_cache.py` **51/51** |
| 2 | Diễn giải: IDSW "9–12 %", "0 mismatch", nhãn cặp KF 7-D → XYAH, C3, số Nhóm B cũ trong handoff | **Đã sửa** | `PHASE2_REPORT.md` mục 18.2, 18.4, 19; mục 5 của file này |

**Audit nguồn gốc `_fix1`** (`src/audit_run_provenance.py` → `results/phase2/run_provenance_audit.csv`):
**12/12** run Nhóm B `_fix1` có `run_meta` hợp lệ, **0 lần resume**, mỗi run là **một lượt chạy liên tục**
(14–123 s), `run_meta` ghi **sau** file tracking cuối, và **sha256 của từng file tracking hiện tại KHỚP
`eval_meta.json`** của lần tính metric ⇒ **giữ nguyên kết quả `_fix1`, không chạy lại**. 113 thư mục còn lại
thuộc bản runner cũ (**LEGACY** — không xác minh được bằng fingerprint), **0 nghi vấn**.

**Kết quả Nhóm B sau khi sửa** (cấu hình y hệt bản đã khóa, chỉ đổi `name`; không tuning) — **đây là bảng dùng
cho luận văn**, số cũ chỉ còn giá trị lịch sử:

| cặp (yếu tố được kiểm soát) — TEST 40 video, c25 | trước khi sửa | **sau khi sửa (`_fix1`)** |
|:--|:--|:--|
| bộ đệm `max_age` 1 → 30 | +0.0280 [+0.0207, +0.0350] | **+0.0274 [+0.0201, +0.0345]** |
| **gói** biểu diễn trạng thái **+ thiết lập bộ lọc** (KF 7-D + Q/R/P₀ của SORT → XYAH + Q/R/P₀ của ByteTrack) | −0.0133 [−0.0185, −0.0077] | **−0.0132 [−0.0185, −0.0077]** (thô −0.0132726 → −0.0132500) |
| motion model CV → EKF–CTRV (cùng XYAH, cùng vòng đời) | +0.0020 [−0.0001, +0.0042] | **+0.0020 [−0.0001, +0.0042]**, ΔIDSW **−108 [−186, −28]** |
| chỉ đổi bộ lọc ở `max_age` 1 | −0.0140 [−0.0167, −0.0116] | **−0.0140 [−0.0167, −0.0116]** |
| vòng đời SORT → ByteTrack (cùng EKF) | +0.0364 [+0.0298, +0.0428] | **+0.0363 [+0.0296, +0.0428]** |

IDSW giảm **2,8–9,8 % tùy nhánh** (orig30 2,79 %; cv30 4,41 %; ekf30 5,03 %; orig1 8,85 %; ekf1 9,79 %) —
**không** phải "9–12 % ở mọi nhánh" như bản trước viết. Mọi kết luận định tính **không đổi**. `bt-ekf`
(ByteTrack) **bit-identical** bản cũ: 12/12 video dev, 40/40 test ⇒ bản sửa SORT không đụng ByteTrack, nên
**không** cần chạy lại Nhóm A, Nhóm C hay Giai đoạn 1 (chỉ Nhóm B dùng `SORTTrack`).

Các hiệu số trên chỉ có nghĩa **trong đúng cặp được kiểm soát**; chúng **không** tạo thành một phép phân rã
cộng hay một thứ hạng áp dụng cho mọi dataset.


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

- Nhánh: **`phase2-mixed-traffic`**, HEAD **`8665480`** (chỉ commit cục bộ, **không push/merge** — yêu cầu của tác giả).
- Commit của mốc 2 (từ `e353b84` = cuối mốc 1):

```
8665480 Review vong 5 (muc 5,6): adapter COCO-video + smoke BDD that + ha muc dien giai
0520c01 Sua thu tu quy tac .gitattributes: git lay dong KHOP CUOI CUNG
2895b71 Review vong 5: sua 2 loi P1 + 3 loi tich hop, kiem chung QUA RUNNER THAT
ed4fda2 Ban giao vong 4: handoff + brief + trang thai Nano sau khi co backend NumPy
11603cb Backend ByteTrack khong phu thuoc torch (CV + EKF) cho Jetson Nano
96c5478 Review vong 4 (muc 1-3): sua tai lieu, kiem ke BDD100K that, chuan bi smoke rieng
5c170b3 Chuyen base review sang a8ecaf6 (Codex xac nhan doc + PASS WITH LIMITATIONS)
ffd1dd4 Chot bao cao moc 2 + kiem ke BDD100K + giao thuc do CV/EKF tren Nano
a8ecaf6 Kiem tra goi delta vong 3: quet duong dan ben trong .diff + ap patch roi so toan bo cay code
5380221 REVIEW_BRIEF vong 3: rut gon (897 -> 801 tu)
324d752 REVIEW_BRIEF vong 3 + chuyen base review sang 9a1c404 (Codex xac nhan da doc)
5658c3b Smoke YOLO that cho provenance cache (30 frame dev, GPU that) - dong diem con mo cua ban va
534146c Khoi phuc trang thai Git cua cac file rieng bi commit nham o 67445dc
e5d076c Sua hai loi dong goi delta: noi dung file rieng lot vao patch, patch khong tai tao dung target
8cfbca1 Tich hop ban va cache review vong 3 (phase2_cache_fix_review3): provenance checkpoint
9a1c404 Goi delta: khong luu hai lan cung mot noi dung (tests_latest.log == tests_<ngay>.log)
5c5596e REVIEW_BRIEF khong ghi cung base/target: placeholder <!-- BASE_TARGET --> dien luc dong goi
4ccfc65 Ban giao hai muc: goi DELTA cho moi vong sua, goi day du cho moc lon
35dd59e Review vong 2: artifact goi review tai HEAD 7341812 (FILE_HASHES khop 1256/1256 file)
7341812 Review vong 2: ghi ro dong HEAD trong handoff luon dung TRUOC git_head cua goi ZIP
cdbf911 Review vong 2: sinh lai CODEX_HANDOFF tai HEAD 61fe76a
61fe76a Review vong 2 (4,5): dong bo tai lieu + dua dau van tay danh gia vao goi ban giao
67445dc Review vong 2 (1,2,3): trang thai tracking theo tung video + chinh sach cache thong nhat + audit provenance
23f63ed Review-fix (7i): artifact goi review tai HEAD 4889209 (FILE_HASHES khop toan bo 1229 file)
4889209 Review-fix (7h): ghi FILE_HASHES.csv sau cung (sau INDEX.md) de hash trong gói la ban cuoi
c6213b0 Review-fix (7g): artifact goi review tai HEAD c601d91
c601d91 Review-fix (7f): ghi ro quan he giua commit va goi ZIP trong CODEX_HANDOFF
7610b56 Review-fix (7e): artifact cua goi review (INDEX, versions, FILE_HASHES, patches, external/bytetrack.yaml) tai HEAD 6a1921f
6a1921f Review-fix (7d): khong tinh artifact do chinh make_handoff sinh vao kiem tra dirty (ghi ro trong versions.json)
e281e1a Review-fix (7c): bo qua duong dan mau khi kiem link, sua tham chieu truong created trong CODEX_HANDOFF
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
 M CODEX_HANDOFF.md
 M handoff/REVIEW_BRIEF.md
 M src/demo/rerun_with_state_log.py
```

- Ngoài ra người dùng có các file **untracked** riêng (`live.bat`, `outputs/**`, `src/demo/*buffer90*`,
  `src/demo/live_view.py`, …) — **không** nằm trong ZIP và không bị sửa.

> **Quan hệ giữa commit và gói ZIP** (rút kinh nghiệm từ review vòng 1 và 2): gói được dựng **sau khi mọi
> thay đổi mã/cấu hình/kết quả đã commit** — `src/make_handoff.py` **từ chối dựng** nếu còn file trong phạm vi
> gói khác HEAD.
>
> Con số trên dòng HEAD ở trên là commit **tại thời điểm sinh ra chính tài liệu này**. Tài liệu được sinh ra
> **rồi mới commit**, nên commit chứa nó — và `git_head` trong `handoff/versions.json`, tức commit mà gói ZIP
> được dựng từ đó — luôn đứng **sau** con số này. Đây là thứ tự bình thường, **không** phải dấu hiệu gói lệch
> commit. Nguồn đối chiếu chính xác duy nhất về nội dung gói là `handoff/FILE_HASHES.csv` (sha256 từng file);
> `git_head` trong `handoff/versions.json` cho biết cây mã mà gói được dựng từ đó. Các artifact do chính script sinh (`handoff/versions.json`, `INDEX.md`, `FILE_HASHES.csv`, `patches/`,
> `external/`) được commit **ngay sau** lần dựng, nên commit cuối của nhánh có thể là commit "artifact" đứng
> **sau** `git_head` ghi trong `handoff/versions.json` và chỉ chứa đúng các file đó. Đối chiếu chính xác nội
> dung gói bằng `handoff/FILE_HASHES.csv`.

## 3. Thứ tự đọc

1. `CODEX_HANDOFF.md` (file này).
1b. **`handoff/MOC2_FINAL_SUMMARY.md` — báo cáo thực nghiệm CHỐT của mốc 2** (Nhóm A theo từng điều kiện
   confidence, Nhóm B `_fix1` với thành phần thay đổi của từng cặp, Nhóm C gồm cả kết quả âm, tách dev/test,
   tách tốc độ tracker-từ-cache với tốc độ toàn pipeline, và giới hạn provenance legacy). **Đọc file này
   trước `PHASE2_REPORT.md`** nếu chỉ cần kết quả.
1c. `handoff/BDD100K_INVENTORY.md` — kiểm kê BDD100K **thực tế 2026-09-18**: đã có **ảnh** `val`
   (200 video / 39.973 jpg, đúng vị trí), **chưa có nhãn** `box_track_20` và **chưa có** split `train`
   ⇒ smoke BDD còn bị chặn. Kèm gói cần tải, vị trí đặt, lệnh kiểm tra, kế hoạch hai thí nghiệm tách biệt.
1d. **`handoff/BACKEND_NP_NOTES.md`** — phụ lục vòng 5: bảng finding → sửa ở đâu → kiểm thử → kết quả;
   lệnh chạy + dependencies của backend NumPy; nhãn BDD COCO-video; kết quả smoke.
1e. `third_party/ultralytics_trackers/PROVENANCE.json` + `src/ul_backend.py` — **backend ByteTrack không
   phụ thuộc torch** (`BYTETRACK_BACKEND=np`), cho CV và EKF–CTRV. **Quy ước hash: sha256 của nội dung
   chuẩn hóa LF** (`HASH_CONVENTION`) — áp dụng thống nhất cho PROVENANCE, `fingerprint()` và test.
2. `results/phase2/REVIEW_moc2_codex.md` — trạng thái từng phát hiện của review vòng 1.
3. `PHASE2_REPORT.md` — mục **19** (review vòng 2: trạng thái run bị gián đoạn, chính sách cache, audit
   nguồn gốc), mục **18** (review vòng 1, có bảng trước/sau), rồi mục **12–17** (mốc 2); mục 1–11 là mốc 1
   (giữ nguyên làm nguồn đối chiếu).
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
   **mới sau review vòng 1**: `src/run_fingerprint.py`, `src/check_cache_consistency.py`,
   `src/test_sort_upstream_parity.py`, `src/test_run_experiment_guard.py`, `third_party/sort_upstream/`;
   **mới sau review vòng 2**: `src/audit_run_provenance.py`, `src/test_run_resume_guard.py` (+ phần
   `video_meta` / `status` trong `src/run_experiment.py`, `GEN_FIELDS` trong `src/det_cache.py`).
8. `results/phase2/run_provenance_audit.csv` — audit nguồn gốc **mọi** thư mục run (vòng 2).
   Riêng khi nhận **gói delta**: đọc `REVIEW_BRIEF.md` trước tiên, tài liệu này chỉ dùng để tra bối cảnh.
9. `handoff/logs/tests_latest.log` + `handoff/logs/tests_summary.csv` — kết quả test thực tế (exit code).

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
| 6 | Chuẩn bị BDD100K (2 thí nghiệm tách biệt) + Nano | ⏳ chuẩn bị | YAML `bdd_transfer_val` / `bdd_tuned_val`; adapter + `check_bdd100k.py` fail-loud; smoke fixture 11/11; **kiểm kê đầy đủ + gói cần tải + lệnh kiểm tra** (`handoff/BDD100K_INVENTORY.md`); Nano: export ONNX + đối chiếu op TRT 8.2 + **giao thức đo CV ↔ EKF cùng điều kiện** (`scripts/nano/README.md` mục 4c) | mục 17; `handoff/BDD100K_INVENTORY.md`; `scripts/nano/README.md` | **Chưa có dữ liệu BDD100K** (đã tìm toàn ổ C:/D:) ⇒ chưa chạy được smoke BDD. **Chưa kết nối Nano** ⇒ chưa có số nào trên thiết bị. Vòng lặp ByteTrack không phụ thuộc torch **chưa làm** |
| 7 | Bàn giao để review không cần máy | ✅ | File này + ZIP + log test + patch/diff + versions; **từ vòng 2** gói mang thêm `eval_meta.json` / `occl_meta.json` / `video_meta/` của các run có chúng | `handoff/`, `handoff/INDEX.md` mục "Ghi chu dong goi" | Log `assoc_log` đầy đủ (parquet) giữ tại máy; trong ZIP là bản trích quanh sự kiện. **Các run chạy bởi runner cũ không có dấu vân tay đánh giá** — số lượng ghi trong `INDEX.md`, không sinh hồi tố |

## 5. Kết luận chính — tách "đo được" / "suy luận" / "chưa xác định"

**Đo được (có số, có CI, dẫn file)**

1. **Trên 40 video test**: ở `c10`, **EKF–CTRV hơn KF–CV có ý nghĩa** (ΔHOTA +0.0033 [+0.0011, +0.0057],
   ΔIDF1 +0.0053, ΔAssA +0.0072); ở `c25` CI vẫn chứa 0 (+0.0012 [−0.0003, +0.0027]) ⇒ **không đủ bằng
   chứng về khác biệt**, **không** phải bằng chứng về sự tương đương (không có phép kiểm tương đương nào
   được đăng ký trước; CI c25 vẫn cho phép khác biệt thật tới +0.0027). Độ lớn rất nhỏ (≈ +0,6 % tương đối).
   UKF **kém** EKF một chút có ý nghĩa ở c25 (−0.0011 [−0.0019, −0.0002]).
   PF–CTRV kém rõ rệt (−0.0316 [−0.0389, −0.0255] so với EKF).
2. **Thứ tự ảnh hưởng (test, Nhóm B — số của bản `_fix1` sau khi sửa lỗi SORT)**: vòng đời/association
   SORT → ByteTrack **+0.0363 [+0.0296, +0.0428]** > bộ đệm 1 → 30 **+0.0274 [+0.0201, +0.0345]** > **gói**
   biểu diễn trạng thái **+ thiết lập bộ lọc** (KF 7-D + Q/R/P₀ của SORT → XYAH + Q/R/P₀ của ByteTrack)
   **−0.0132 [−0.0185, −0.0077]** ≫ motion model CV → EKF **+0.0020 [−0.0001, +0.0042]** (CI chạm 0; chỉ
   ΔIDSW **−108 [−186, −28]** có ý nghĩa). Chỉ đọc trong đúng cặp được kiểm soát, không phải thứ hạng phổ quát.
3. **Nguyên nhân mất ID** (dev, 184 sự kiện): ở tầng che hoàn toàn, **dự đoán lệch** là số 1 (cv 11/22,
   ekf 13/23 ca) — IoU(dự đoán, det) trung bình 0,06 sau trung vị 25 bước mất dấu; `detector_miss` **không**
   xuất hiện ở tầng full; 86 % ID mới là **track mới sinh từ detection**.
4. **Nới `track_buffer` không giúp**: 0,8 → 4,0 s giữ nguyên tỉ lệ giữ ID trong khi **ghép nhầm tăng đơn điệu**
   (CV 2,5 % → 5,3 %).
5. **Tuning tham số nhiễu: không cấu hình nào CẢI THIỆN HOTA có ý nghĩa** (21 cấu hình; một số cấu hình CV
   **kém hơn có ý nghĩa**, ví dụ `std_weight_position` 1/10 với `std_weight_velocity` 1/320: −0.0094
   [−0.0197, −0.0019]) — dù tham số **thực sự có hiệu lực**
   (covariance đổi rõ: `P[θ,θ]` trung bình 2,46 / 2,83 / 3,88 theo `std_omega`) và đầu ra **không** giống hệt
   (0/12 bit-identical; ~99 % dòng lệch dưới 1 px với EKF).
6. **Thêm vòng ghép cho track LOST**: giữ ID qua che khuất tăng có ý nghĩa (dev +0.049; test +0.036; tầng che
   hoàn toàn **gấp đôi**: 0,060 → 0,121) **nhưng HOTA giảm có ý nghĩa trên test** (−0.0046 / −0.0034) vì ghép
   nhầm tăng 3–5 lần (3,5 % → 9,3 %).
7. **PF — đúng cái đã đo**: ở các lần **cập nhật sau một khoảng mất dấu dài**, ESS/N ~1 % (đo **trước**
   resample) và chỉ còn ~3 % particle phân biệt (đo **sau** resample). Khoảng cách so với EKF **thu hẹp khi
   N tăng** (100 → 300 → 500) ⇒ số hạt là **một phần** nguyên nhân, nhưng ở N = 500 PF vẫn kém rõ rệt nên
   số hạt **chưa giải thích được toàn bộ**. **Không** suy ra được từ log này rằng
   mất đa dạng xảy ra *trước khi có quan sát mới*, cũng **không** phải bằng chứng đó là *nguyên nhân* của các
   lần nối lại thất bại (những lần đó không có bước cập nhật nên không có dòng log).
8. Mọi kết quả GĐ1 và mốc 1 **tái lập được**: expA (có log) vs mốc 1 **288/288 cặp (lần chạy, video)**
   bit-identical **và** song ánh ID.

**Suy luận (có cơ sở nhưng không đo trực tiếp)**

- Vì `match_thresh` = 0,8 trên cost 1 − IoU·conf, IoU < 0,2 là **rào cứng**: cải thiện motion model chỉ dịch
  dự đoán vài chục px sẽ không vượt được rào này ở các đoạn che dài → **một cách giải thích nhất quán** với
  việc CV/EKF/UKF không tách được nhau trên chỉ số toàn cục, và tầng che hoàn toàn đứng yên ở ~6 % với cả ba.
  **Không loại trừ** motion model là một phần nguyên nhân: ba bộ lọc này dùng chung một họ giả định (chuyển
  động trơn, nhiễu Gauss) và chung khung association, nên có thể cùng thất bại vì cùng một lý do; một họ dự
  đoán khác hẳn chưa được thử.
- Lợi ích của EKF (nếu có) tập trung ở nhóm sự kiện nhỏ (che khuất trung bình, quỹ đạo cong) — mẫu tự nhiên
  của UA-DETRAC quá nhỏ để tách (đã ghi từ GĐ5).

**Chưa xác định**

- **Nguồn gốc cấu hình của 53/56 sequence cache `legacy`**: chúng được sinh trước khi có `spec_sha256` nên
  không xác minh được bằng fingerprint (chỉ 3 sequence fixture ở trạng thái `ok`, 0 mismatch). Cache
  UA-DETRAC không đổi byte giữa hai gói bàn giao. Không bổ sung fingerprint hồi tố.
- **Nguồn gốc của 113 thư mục run `LEGACY`** (chạy bởi bản runner cũ): không có `eval_meta`/`video_meta` để
  đối chiếu; 0 dấu hiệu nghi vấn nhưng cũng không chứng minh được bằng hash.
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
$P src/make_handoff.py --base e353b84            # goi DAY DU  -> handoff/phase2_review_latest.zip
$P src/make_review_delta.py                      # goi DELTA   -> handoff/review_delta_latest.zip
$P src/make_review_delta.py --dry-run            # xem truoc se gom / bo qua nhung gi
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
| `python src/test_run_resume_guard.py` (vòng 2) | 0 | 10/10 — A→B bị ngắt→A, thiếu/hỏng `video_meta`, resume giữ provenance |
| `python src/audit_run_provenance.py --pattern "p2-expB_detrac_*_fix1-*" --verify-eval-fingerprint` (vòng 2) | 0 | `handoff/logs/audit_run_provenance_fix1.log` — 12/12 OK, 0 nghi vấn |
| `python src/audit_run_provenance.py --pattern "p2-*" --verify-eval-fingerprint` (vòng 2) | 0 | `handoff/logs/audit_run_provenance_all.log` — 21 OK, 113 LEGACY, **0 nghi vấn** |
| `python src/aggregate_phase2.py --exp .../expB_detrac_{dev,test}_fix1.yaml` (vòng 2) | 0 | chỉ sinh lại bảng dẫn xuất sau khi sửa **nhãn** cặp (không đổi tham số thí nghiệm) |
| `python src/make_review_delta.py` (vòng 2) | 0 | `handoff/review_delta_latest.zip` — 67 file, 0,38 MB (gói đầy đủ 66,9 MB), `MANIFEST.csv` khớp hash |
| `python src/run_all_tests.py --out-dir handoff/logs/review5` (vòng 5) | 0 | **25/25 PASS**, 0 FAIL, 0 SKIPPED, 0 NOT_RUN |
| `python src/test_runner_backend_np.py` (vòng 5) | 0 | **9/9** — CLI **và** `stage_track()` **thật** chạy trong tiến trình **chặn** `torch`/`ultralytics`; sửa YAML cùng đường dẫn ⇒ chạy lại (**4 dòng → 0 dòng**); đổi backend ⇒ chạy lại |
| `python src/test_bdd100k_coco.py` (vòng 5) | 0 | **8/8** — adapter COCO-video: ID theo **(video, instance_id)**, trường thiếu không bị bịa |
| `python -m src.mot_datasets.bdd100k_coco --export-gt --benchmark BDDSMOKE` (vòng 5) | 0 | GT MOTChallenge cho 3 sequence smoke, split **riêng** `BDDSMOKE-val` |
| `python src/run_experiment.py --exp configs/phase2/experiments/bdd_smoke_val.yaml` (vòng 5) | 0 | **smoke BDD chạy thật**: nhãn → GT → cache (GPU) → tracking → evaluator → tổng hợp |
| `python src/run_all_tests.py --out-dir handoff/logs/backend_np` (vòng 4) | 0 | **23/23 PASS**, 0 FAIL, 0 SKIPPED, 0 NOT_RUN |
| `python src/test_bytetrack_np_parity.py` (vòng 4) | 0 | **8/8** — CV 17.348 hàng + EKF 17.333 hàng **bit-identical**, chạy trong tiến trình **chặn** `torch`/`ultralytics` |
| `python src/run_all_tests.py --out-dir handoff/logs/review_delta_fix_full` (vòng 3) | 0 | **22/22 PASS**, 0 FAIL, 0 SKIPPED, 0 NOT_RUN |
| `python src/smoke_cache_provenance_yolo.py` (vòng 3) | 0 | YOLO **thật** + GPU thật, MVI_20052 30 frame: **5/5 PASS** — `weights_sha256` khớp `model.ckpt_path` |
| `python src/check_bdd100k.py --part val` (mốc 3) | **1** | **đúng như mong đợi**: fail-loud vì chưa có dữ liệu BDD100K trên máy — xem `handoff/BDD100K_INVENTORY.md` |
| `python src/check_cache_consistency.py` + `audit_run_provenance.py --pattern "p2-*"` (mốc 3, chạy lại để chốt số) | 0 | cache **3 `ok` + 53 `legacy` + 0 `mismatch`**; run **21 `OK` + 113 `LEGACY` + 0 nghi vấn** |

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
| Mỗi lần chạy | `data/processed/trackers/<split>/<dir>/run_meta.json` (cfg, cache, commit, `created`, `input_fingerprint`, `status`, `resumed`), `video_meta/<video>.json` (bằng chứng hoàn tất từng video — từ vòng 2), `results/trackeval/<split>/<dir>/eval_meta.json` + `results/phase2/<exp>/<cond>/occl_meta.json` (dấu vân tay đánh giá) |
| Audit nguồn gốc | `results/phase2/run_provenance_audit.csv` (mọi thư mục run: status, resume, hash khớp `eval_meta` hay không) |

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
11. **(sau review vòng 2)** `src/det_cache.py`: `GEN_FIELDS` — **một** tập trường dùng cho **cả** đường ghi
   (`write_meta` so sánh, khác ⇒ từ chối và không đổi gì trên đĩa) **và** đường đọc (`spec_sha256` hash đúng
   tập đó). Cần kiểm: tập này có **thiếu** trường nào thực sự ảnh hưởng detection không, và việc **loại**
   đường dẫn `weights` (giữ `weights_sha256`) có đúng không. Cache `legacy` (chưa có `spec_sha256`) chỉ được
   cảnh báo, **không** backfill.
12. **(sau review vòng 2)** `src/run_experiment.py`: `can_reuse_video()` + `video_meta/<video>.json` +
   `run_meta.status`. Cần kiểm: có đường nào tái sử dụng đầu ra mà **không** đi qua `can_reuse_video` không;
   `mark_run_started` có chạy **trước** mọi thao tác ghi đè không; `mark_run_finished` hợp nhất provenance có
   làm mất lịch sử `resumed` không. Test đối chứng: `src/test_run_resume_guard.py` (10 phép, chạy qua runner
   thật với lỗi giả lập).
13. **(sau review vòng 2)** `src/audit_run_provenance.py`: tiêu chí phân loại `OK` / `LEGACY` / `NGHI VAN`.
   Kết luận "giữ nguyên `_fix1`, không chạy lại" **dựa trên** `tracking_hash_match=True` với `eval_meta.json`
   — nếu tiêu chí này chưa đủ mạnh thì kết luận đó cần xem lại.
14. **(sau review vòng 4)** `third_party/ultralytics_trackers/` + `src/ul_backend.py`: bản vendor có thật sự
   chỉ khác upstream ở **5 dòng import** không (test đối chiếu tự sinh lại phép so), `np_compat.bbox_ioa`
   có chép đúng thân hàm gốc không, và `BYTETRACK_BACKEND` có đường nào **không** đi qua `ul_backend`
   (tức vẫn kéo ultralytics vào) không.
15. **(sau review vòng 4)** `src/bytetrack_np.Detections`: hợp đồng NumPy đã đủ chưa so với `Boxes` mà
   `BYTETracker` thực sự dùng (`.xywh`, `.conf`, `.cls`, `len`, cắt theo mask) — nếu upstream đổi, phép so
   phải phát hiện được.
16. **(sau review vòng 5)** `input_fingerprint` nay gồm `tracker_cfg` (qua `cfg_to_dict`) **và**
   `tracker_yaml_sha256_12`. Cần kiểm: còn đầu vào nào ảnh hưởng kết quả mà **chưa** nằm trong dấu vân tay
   không (ví dụ `--class-map`, `allow_truncated`).
17. **(sau review vòng 5)** `src/mot_datasets/bdd100k_coco.py`: nhãn là **COCO-video**, không phải
   `box_track_20` native. Cần kiểm: `crowd = iscrowd | ignore` có đúng ý nghĩa vùng bỏ qua của TrackEval
   không; `visibility = 1.0` ở `gt.txt` đã được ghi rõ là **placeholder** chưa; và việc đánh số `track_id`
   theo `(video, instance_id)` có chỗ nào còn rò rỉ ID giữa các video không.

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
ekf 0.487, ukf 0.485, pf500 0.435. **Tầng che nặng** (`has_full` = sự kiện **có đoạn che ≥ 0,90**, không
phải che 100 % ở mọi frame; 182 sự kiện): **tại c25** cả ba bộ lọc đều 0,060 (11/182); **tại c10** UKF 12/182
(0,066), CV/EKF vẫn 11/182 — chênh **một** sự kiện, không có ý nghĩa thống kê. Phát biểu "cả ba như nhau" chỉ
áp dụng cho **c25**.

**B — thuật toán** (c25, mỗi cặp chỉ đổi một yếu tố) — **số của bản `_fix1`**, tức sau khi sửa lỗi vòng đời
SORT (`results/phase2/expB_detrac_test_fix1/c25/paired_pairs.csv`); số trong ngoặc là bản trước khi sửa, chỉ
còn giá trị lịch sử:

| cặp (yếu tố được kiểm soát) | ΔHOTA `_fix1` | (trước khi sửa) | ΔIDSW `_fix1` |
|:--|:--|:--|:--|
| bộ đệm `max_age` 1 → 30 | **+0.0274 [+0.0201, +0.0345]** | (+0.0280) | −1055 [−1390, −759] |
| **gói** biểu diễn trạng thái **+ thiết lập bộ lọc** (KF 7-D + Q/R/P₀ của SORT → XYAH + Q/R/P₀ của ByteTrack) | **−0.0132 [−0.0185, −0.0077]** | (−0.0133) | +4 [−140, +156] (CI chứa 0) |
| motion model CV → EKF–CTRV (cùng XYAH, cùng vòng đời SORT) | +0.0020 [−0.0001, +0.0042] (CI chạm 0) | (+0.0020) | **−108 [−186, −28]** |
| chỉ đổi bộ lọc, `max_age` 1 | **−0.0140 [−0.0167, −0.0116]** | (−0.0140) | +186 [+62, +308] |
| vòng đời/association SORT → ByteTrack (cùng EKF) | **+0.0363 [+0.0296, +0.0428]** | (+0.0364) | −1685 [−2090, −1292] |

Mỗi hiệu số chỉ có nghĩa **trong đúng cặp được kiểm soát** — không phải một phép phân rã cộng, cũng không phải
thứ hạng áp dụng cho mọi dataset. `bt-ekf` **bit-identical** giữa hai bản (40/40 video test).

**C — nhánh thăm dò** (vòng ghép thứ 3 cho track LOST, gate Mahalanobis; **không** phải cấu hình được chọn):
giữ ID qua che khuất **tăng có ý nghĩa** (+0.036 [+0.019, +0.051] với CV; tầng che hoàn toàn 0,060 → **0,121**)
nhưng HOTA **giảm có ý nghĩa** (cv −0.0046 [−0.0072, −0.0021]; ekf −0.0034 [−0.0058, −0.0012]) vì FP +8 000 và
IDSW +261…+319 → **kết quả âm, không bật cơ chế này nếu mục tiêu là HOTA**.

**Tốc độ tracker từ cache** (56 340 frame, CPU, không có detector, đo khi máy còn chạy việc khác → tham khảo):
sort-orig1 774 fps, bt-ekf 498, cv 420–570, ekf 333–478, ukf 175–187, pf500 106–114. Toàn pipeline có detector
(GTX 1650): 45–51 fps (mục 7.3). **Chưa đo trên Nano.**

Kết quả đầy đủ: `results/phase2/{expA,expB,expC}_detrac_test/<cond>/{summary.csv, paired_bootstrap_combined.csv,
paired_pairs.csv, occlusion_strata.csv, bootstrap_DETRAC-test.csv}`, log chạy trong `handoff/logs/exp*_detrac_test.log`.

