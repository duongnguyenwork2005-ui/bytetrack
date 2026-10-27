# R60A — Hồ sơ quyết định chuyển sang mốc 5 (đánh giá cuối BDD100K val + UA-DETRAC test)

> **DỰ THẢO — CHƯA KHÓA — CHƯA CẤP QUYỀN CHẠY.** Hồ sơ này chỉ kiểm kê và đối chiếu tĩnh. Không phương án nào trong
> phiếu quyết định (mục 5) được chọn. Codex approve hồ sơ chỉ nghiệm thu tài liệu, không mở bước thực thi nào.

Ngày 2026-10-23. Người soạn: Claude (`claude-opus-5-5`, effort high). Reviewer: Codex. Repo `D:/LABOTARY/bytetrack`,
nhánh `phase2-mixed-traffic`, HEAD `117d62da027e8ce3e4c9f455d576a924b4d82bbb` (`git rev-parse HEAD`), cây có nhiều thay
đổi chưa commit, giữ nguyên.

**Nguồn ủy quyền:** plan R60 do Codex (GPT) đề xuất sau lượt `bytetrack_r59_sort_test40_20261023_1053` (lời gọi
`0012_planner`, 23/10 18:32), có hiệu lực vì người dùng đã duyệt; một bước R60A, write_paths: `handoff/PLAN_DA_CHOT.md`
(chỉ mục 0) và file này. PLAN mục 0 đã cập nhật trước file này.

**Cách làm và giới hạn của lượt:** chỉ Read/Grep/Glob và `git status/diff/rev-parse`. Không chạy Python, `py_compile`,
fixture, tracker, evaluator, bootstrap hay `kiem-khoa`; không băm file. Mọi hash và kết quả runtime dưới đây là **trích
từ bằng chứng đã lưu** (ghi nguồn), lượt này không tự xác minh lại. Không quét dữ liệu lớn (`data/`, cache NPZ, MOT).
Ký hiệu: **[nguồn]** = đọc trực tiếp ở file dẫn; **[đọc mã]** = suy từ đọc mã, chưa kiểm runtime; **[chưa xác minh]** =
lượt này không có bằng chứng.

---

## 1. Đóng trạng thái R59

| Mục | Nội dung | Nguồn |
|---|---|---|
| Verdict | `approve`, `issues: []`, `ideas: []`, `next_task` rỗng | `D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_005.json` |
| P1 | Đóng: reviewer băm độc lập 3 file kiểm khóa cuối (đã chuyển ra vùng automation), khớp `SHA256SUMS_truoc_khi_chuyen.txt`; `dat=true`, 1557 file, `lech=[]`, đúng khóa A; log exit 0 | review_005 `evidence[0]`; báo cáo `handoff/logs/review59/03_ket_qua_sort_test40.md` §2a |
| P2 | Đóng: tầng điểm thấp ByteTrack rỗng ở c25 (120/120 `n_det_bt_thap=0`), hiệu số không đo association điểm thấp | review_005 `evidence[1]`; báo cáo 03 §5 |
| Phiên | Nghiệm thu đúng `results/phase2/r59_sort_test40/thuc_thi/phien_20261023_160539_122500_pid12864`; dừng ở kết quả đối chứng mốc 4 | review_005 `evidence[5]` |
| Tổng hợp | `tong_hop/phien_20261023_160539_122500_pid12864/` (`doi_chieu_6_cap.csv`, `nguon.json`, `kiem_chung.json`, `SHA256SUMS`); reviewer ghi ba file tổng hợp khớp `SHA256SUMS` | review_005 `evidence[4]` |
| Giới hạn reviewer | Review tĩnh, không chạy lại fixture/tracker/evaluator/kiem-khoa; không băm lại 120 NPZ cache, không đếm lại D4; hạn chế `buoc_kiem_khoa` luôn ghi vào `chuan_bi/` (`src/r59_sort_test40/dieu_phoi_sort_test40.py:991`) vẫn tồn tại, không sửa mã đã khóa | review_005 `limitations` |

Đối chiếu với mục 0: mục 0 trước lượt này ghi "chờ Codex review lại toàn bộ R59B"; nay đã sửa thành R59B COMPLETE (review
\_005) và không còn việc review R59 đang chờ. Bảng "Thứ tự các mốc" dòng 4 còn ghi thiếu SORT/ước lượng biến thiên là
lịch sử; đã đính chính **trong mục 0**, không sửa bảng, không sửa báo cáo R59.

---

## 2. Kiểm kê bằng chứng hiện có cho hai tập đánh giá cuối

UA-DETRAC test = 40 video, 56.340 ảnh, 173 ảnh D4 (fingerprint R47; PLAN mục 0 "Kết quả đã xác minh"); "test40" trong
các lượt R54–R59 chính là **toàn bộ** tập này. BDD val = bản COCO-video chuyển đổi, 200 video, 39.973 ảnh
(`configs/phase2/LOCK_bdd_transfer.md` §5). Cả hai tập **đã được xem** ở các vòng trước (PLAN "Quy tắc phạm vi"), không
gọi là chưa từng quan sát.

| # | Bằng chứng | Phạm vi | Cấu hình chính | Nghiệm thu | Chưa được chứng minh |
|---|---|---|---|---|---|
| E1 | **Detector R53** — `handoff/logs/review53/04_ket_qua_danh_gia_day_du.md`; số liệu `results/phase2/r53_detector/full/{detrac_test,bdd_val}/<tên>/` | 3 detector (`yolov8n_coco`, `yolov8n_mixed5` best e15, `yolo11n_mixed5` best e13) × UA-DETRAC test (D3, 56.167 ảnh, chỉ car/bus) + BDD val (5 lớp, 39.973 ảnh) | imgsz 640, conf 0,001, NMS IoU 0,7, max_det 300, FP32; điểm vận hành conf ≥ 0,25, IoU 0,5 (04 §1) | GPT approve (`D:/LABOTARY/automation/runs_local/hoi_codex_sau_danh_gia_full_20261020/answer.json`, theo 04 §0); R53B approve review_005 lượt `bytetrack_r53b_lock_smoke_20261020_0356` | Không phải tracking. NPZ dự đoán (conf 0,001, max_det 300) **khác hợp đồng** cache tracking (mục 3.3). Nguyên nhân recall bus thấp chưa rõ (PLAN "Điểm chưa rõ"). Suy luận lại GPU có thể lệch nhẹ (04 §7) |
| E2 | **ByteTrack test40 R56** — `handoff/logs/review56/03_thuc_thi_tracking_test40.md`; bảng `results/phase2/r56_tong_hop/phien_20261021_161832_735800_pid11640/` (`tong_hop_36_run.csv`, `pf500_trung_binh.csv`, `nguon.json`) | 3 detector × c25/c10 × CV/EKF/UKF/PF500 s0–s2 = 36 run × 40 video | Giao thức `configs/phase2/r54_tracking/giao_thuc_test40.json` (mục 3.1); khóa R56 `khoa_20261021_155510_440500_pid8768` (`ce85628e…9d35247b`, theo PLAN) | Codex approve lần 2 `D:/LABOTARY/automation/nghiem_thu_test40/lan_013/answer.json` (theo PLAN mục 0) | Chỉ điểm (không CI ở R56). Reviewer chỉ băm mẫu cache/MOT; tái lập chỉ từ cùng cache, không suy luận lại detector giữa tiến trình (PLAN mục 0). 173 ảnh D4: chưa biết sót nhãn hay âm tính thật |
| E3 | **Biến thiên R57** — `handoff/logs/review57/01_bien_thien_test40.md`; phiên `results/phase2/r57_bien_thien/phien_20261023_011207_108057_pid8500/` | 36 run R56 × 40 video; 108 CI mỗi thiết kế (video 40 / nhóm 39) | Giao thức `configs/phase2/r57_bien_thien/giao_thuc.json` (cố định **sau** khi xem điểm R56, trước CI); B = 5000, `default_rng(0)`, phân vị 2,5/97,5; PF500 = trung bình COMBINED 3 seed trong từng mẫu | Codex approve `D:/LABOTARY/automation/runs_local/bytetrack_r57_bien_thien_20261023_0029/reports/review_002.json` (theo PLAN mục 0) | CI riêng lẻ, thăm dò, chưa hiệu chỉnh đa so sánh (đề xuất hiệu chỉnh đã bị người dùng từ chối — không đưa lại); giả định video/nhóm độc lập; kiểm cùng cảnh chỉ bao cặp đã biết |
| E4 | **SORT c25 test40 R59** — `handoff/logs/review59/03_ket_qua_sort_test40.md`; `results/phase2/r59_sort_test40/tong_hop/phien_20261023_160539_122500_pid12864/doi_chieu_6_cap.csv` | 3 detector × c25 × SORT-CV30/SORT-EKF30 = 6 run × 40 video, đối chứng 6 run ByteTrack R56 tái dùng | Giao thức `configs/phase2/r59_sort_test40/giao_thuc.json`; quyết định Q-A..Q-F (`review59/01` §2) | Codex approve `review_005.json` (mục 1) | Chỉ c25, chỉ điểm, không CI; tầng điểm thấp rỗng; không tách được thành phần vòng đời/association/khởi tạo; không đo tốc độ |
| E5 | **BDD transfer cũ** — `configs/phase2/LOCK_bdd_transfer.md`, `configs/phase2/experiments/bdd_transfer_cocovid_val.yaml`, `results/phase2/lock/lock_bdd_transfer.json`, `results/phase2/manifests/bdd100k_transfer_cocovid_val.json`, `results/phase2/bdd_transfer_cocovid_val/bao_cao/` (`bang_chinh.md`, `cap_so_sanh.md`, `pf_seed.md`) | **Chỉ `yolov8n_coco`** × c25/c10 × CV/EKF/UKF/PF500 s0–s2 = 12 run × 200 video | Khóa 2026-09-20T03:49:13, `git_commit` `d1f6d62a…` (dirty 3 file), ultralytics 8.4.141, torch 2.6.0+cu124, TrackEval 1.3.0 (`lock_bdd_transfer.json`); chi tiết mục 3 | PLAN mục 0 liệt kê ở "Kết quả đã xác minh" (dòng `bao_cao/`); verdict review cụ thể của lượt đó **[chưa xác minh]** trong lượt này | Không có detector mixed5; khác commit/mã với R54–R56; không báo che khuất, không báo theo lớp (GT gộp vehicle); nhãn COCO-video chuyển đổi, không phải `box_track_20` native; 3 sequence đã xem metric ở smoke; 615 ảnh trong timeline không có annotation, chưa rõ nguyên nhân (mục 3.1) |

Ngoài bảng: R51 (dev12) và mốc 2 cũ không thuộc mốc 5; không chuyển kết quả R51 sang test40/BDD (`review59/01` §5).

**Phân biệt bắt buộc khi báo cáo mốc 5:**
- (i) **Kết quả BDD cũ** (E5) = thí nghiệm *chuyển cấu hình* với detector COCO, mã ở `d1f6d62`; chỉ là lịch sử cho tới khi
  người dùng quyết định điều kiện tái dùng (M5-Q5).
- (ii) **Đánh giá detector mixed5 trên BDD val** (E1) = AP/P/R detector, không phải tracking.
- (iii) **Tracking BDD với detector mixed5** = **chưa thực hiện**; chưa có cache, khóa, smoke hay run nào.

---

## 3. Đối chiếu giao thức BDD cũ với giao thức hiện hành (test40) — đọc tĩnh

### 3.1 Bảng đối chiếu

"Khóa cũ" = giá trị trong E5. "Test40" = `configs/phase2/r54_tracking/giao_thuc_test40.json` (R54, dùng ở R55/R56/R59).
Cột cuối chỉ nêu **lựa chọn cần duyệt** cho lượt mới, không chọn.

| Hạng mục | Khóa BDD cũ (E5) | Test40 (R54) | Ghi chú / lựa chọn chưa duyệt |
|---|---|---|---|
| Checkpoint | `models/yolov8n.pt` `f59b3d83…` (`cache_weights_sha256`) | 3 checkpoint: COCO `f59b3d83…`, v8n mixed5 `542b1381…`, 11n mixed5 `9a5e34a1…` (`giao_thuc_test40.json` `detector`) | Phạm vi detector BDD: M5-Q1 |
| Ánh xạ lớp sau suy luận | COCO gốc | COCO gốc; mixed5: `model.names → anh_xa_model_sang_lop5 → sang_id_coco` (car 2, truck 7, bus 5, motorcycle 3, bicycle 1), lưu `cls_goc` trong NPZ | Đường ánh xạ lop5 hiện nằm trong wrapper R54 gắn DETRAC: `dataset="detrac", part="test"` (`src/r54_tracking/chuan_bi_test40.py:1059`), dùng `DetracDataset` (:1066–1104), ghi `cls_goc` (:1118) **[đọc mã]**. `src/det_cache.py` CLI không có ánh xạ lop5 (Grep `lop5|anh_xa|cls_goc` không thấy) ⇒ BDD + mixed5 cần mã chuẩn bị mới (M5-Q7) |
| Lớp đưa vào tracker | `[1, 2, 3, 5, 7]` (có bicycle) | `[2, 3, 5, 7]` (bicycle bị loại lúc load, `tai.ghi_chu_bicycle`) | GT BDD "vehicle" gồm car+truck+bus+motorcycle+bicycle (LOCK §6) nên khóa cũ đưa bicycle vào tracker. **Không** tự sao `[2,3,5,7]` của DETRAC sang BDD: M5-Q4 |
| Lọc/cắt lúc load | conf > c (ngặt) → lớp → `[:300]` (`det_cache.filter_dets`, `src/det_cache.py:412–425`) | Như cũ: `conf → lop → cat [:max_det]`, `max_det` 300 | Cùng hàm [đọc mã]. Cache phải không có frame `truncated` (VideoDetections chặn, `det_cache.py:446–451`) |
| Cache detector | `yolov8n-coco_conf0.05_all_bddcocovid`: conf 0,05, mọi lớp, `max_det` 3000, ultralytics 8.4.141, torch 2.6.0+cu124, GTX 1650 | conf 0,05, iou 0,7, imgsz 640, `max_det` 3000, `classes` null, `half` false, batch 1, `mask_ignored_regions` true | Tồn tại và nguyên vẹn của cache cũ trên đĩa: **[chưa xác minh]** (không quét `data/interim/det_cache/`). Khóa cũ không ghi đủ provenance cache (mục 3.4). Tái dùng: M5-Q5 |
| Tô xám vùng bỏ qua trên ảnh | Không có tác dụng: `BDD100KCocoDataset.ignored_regions` trả `[]` (`src/mot_datasets/bdd100k_coco.py:131–133`) | Tô xám `ignored_regions.txt` trước suy luận (114,114,114) | Khác bản chất dataset; vùng bỏ qua BDD xử lý ở evaluator (dưới) |
| fps / dt | 5 fps, `time_s = frame_id / 5` (`bdd100k_coco.py:117–128`, `FPS = 5.0` ở `bdd100k_mot.py:92`); một bước 0,2 s | 25 fps, `time_s = (i−1)/25`; một bước 0,04 s | Tham số nhiễu bộ lọc **giữ theo bước** của DETRAC ⇒ mô hình vật lý khác (bước BDD dài gấp 5) (LOCK §3; `src/time_units.py:52`, :76–83) |
| Thời gian giữ track | `track_buffer_sec` 1,2 ⇒ 6 bước (`track_buffer_frames = round(1,2 × 5)`, `time_units.py:217–221`) | 1,2 s ⇒ 30 bước | Cùng thời gian, khác số bước |
| `elapsed_mode` | `steps` (Δt từ chỉ số frame; val không có khoảng bỏ qua theo kiểm kê) | `skip` | Theo `src/run_tracking.py:230–240`: khi không có frame không đọc được và Δt đều, `steps` cho `n_steps = 1` mỗi frame như `skip` **[đọc mã, chưa kiểm runtime]**. Vẫn là lựa chọn phải ghi khóa: M5-Q4 |
| Association | ByteTrack khóa (`match_thresh` 0,8, `fuse_score`, 2 vòng) | Như cũ (track_high 0,25, low 0,1, new 0,25, match 0,8, fuse_score) | Không đổi thuật toán |
| Backend tracker | `ultralytics` toàn ma trận (8.4.141) | `ultralytics` | Phiên bản thư viện hiện hành so với khóa cũ **[chưa đối chiếu]** (khóa R54/R56 không đọc phần phiên bản trong lượt này) |
| YAML bộ lọc | sha256: `bytetrack.yaml` `395701d9…`, EKF `68694ad4…`, UKF `917fe0ea…`, PF500 `ff43d493…` (`lock_bdd_transfer.json` `tracker_yaml_sha256`) | Cùng 4 đường dẫn YAML (`giao_thuc_test40.json` `tracker.bo_loc`) | Hash YAML/`config.py` hiện hành so với khóa cũ **[chưa đối chiếu]**; phải kiểm ở bước chuẩn bị |
| `assoc_log` / `pf_diag` | `assoc_log: false`; `pf_diag` không khai báo trong YAML thí nghiệm | `assoc_log: true`, `pf_diag: true` | Logging khác nhau giữa hai đời; theo thiết kế R58/R59A logging không đổi quyết định tracker, chưa kiểm cho BDD |
| Evaluator | `mot_crowd` = `MotChallengeCrowdIgnore` (`src/trackeval_ext.py:43`); chọn theo loại dataset (`src/run_trackeval.py:130–155`) | `MotChallenge2DBox` (`mot`) | BDD **phải** giữ `mot_crowd`: GT class 13 = vùng bỏ qua, loại FP chưa ghép có IoA > 0,5 (`trackeval_ext.py:2–29`, :118–125) |
| GT / ignore | Split `BDDCOCOVID-val`; vehicle = car+truck+bus+motorcycle+bicycle → class 1; crowd/ignore, 3 distractor (`other person`, `trailer`, `other vehicle`), `train` → class 13 (`bdd100k_mot.py:58–63`, :81–91, :349); `seqLength` từ timeline ảnh; `visibility` 1,0 là placeholder; `gt_seqmap_sha256` `181a419d…` | GT DETRAC-test hiện hành, vehicle gộp car/bus/van/others; 173 D4 giữ trong timeline | **Phân biệt hai khái niệm** (sửa R60-F1): (i) *ảnh thiếu trên đĩa* — `n_missing_images_on_disk` 0, `n_videos_with_frame_gaps` 0; (ii) *ảnh có trong timeline nhưng không có annotation* — `n_images_without_annotation` **615** (`results/phase2/bdd_cocovid_check/val_report.json`, số trích từ báo cáo cũ, lượt này không đếm lại). Mã ghi ví dụ video `b23b2649-8349d2a1` có ảnh tới frame 202 nhưng annotation cuối ở frame 94, nên `seqLength` lấy từ timeline ảnh (`bdd100k_coco.py:155–157`). ⇒ Không kết luận "BDD không có tương đương D4". Nguyên nhân 615 ảnh (sót nhãn, cắt nhãn khi chuyển đổi, hay thật sự không có đối tượng), phân bố theo video và tác động lên metric **[chưa xác minh]**; không suy ra là âm tính thật, không tự đổi chính sách đánh giá. Khóa cũ giữ các ảnh này trong timeline; cách `mot_crowd` chấm frame có ảnh nhưng GT trống **chưa đối chiếu** trong lượt này. Chính sách cho các ảnh này: M5-Q4 |
| Che khuất | Tắt (nguồn thiếu `occluded`) | Không thuộc R54 | Không báo phục hồi sau che khuất trên BDD |
| PF ba seed | Bảng chính: trung bình **metric** theo seed; bootstrap: trung bình **thống kê đủ** rồi tái tổ hợp (LOCK §6) | R56: trung bình/min/max metric COMBINED qua 3 seed (`pf500_trung_binh.csv`); R57: trung bình COMBINED 3 seed trong từng mẫu bootstrap | Hai đời dùng quy tắc bootstrap PF khác nhau: M5-Q6 |
| Thống kê | Bootstrap ghép cặp theo video, 5000 lần, seed 0; 5 cặp khóa trước (`ekf−cv`, `ukf−cv`, `ukf−ekf`, `pf500−cv`, `pf500−ekf`) | R56 chỉ điểm; R57 CI hậu nghiệm (6 hiệu số/nhóm) | M5-Q6 |
| Tổng hợp video | `COMBINED_SEQ` tái tổ hợp từ thống kê đủ (`src/boot_stats.py`) | `COMBINED_SEQ` của TrackEval | Cùng nguyên tắc, khác mã điều phối |
| Mã điều phối | `src/run_experiment.py` (YAML thí nghiệm), commit `d1f6d62` | Wrapper R54/R55 + khóa phiên (hash mã, thư viện, cấu hình hiệu lực) | PLAN mục 0: "Mọi so sánh chỉ dùng các run tạo tại cùng commit, theo giao thức" ⇒ ràng buộc trực tiếp cho M5-Q5 |

### 3.2 Khác biệt BDD–DETRAC cần nêu ở mọi báo cáo mốc 5 (không tự san bằng)

1. **Thời gian:** 5 fps vs 25 fps; tham số nhiễu theo bước giữ nguyên ⇒ không phải cùng mô hình nhiễu vật lý. Chọn lại
   tham số theo đơn vị vật lý hay tuning trên BDD là **việc mới**, chỉ được làm trên dev nội bộ BDD và cần người dùng duyệt
   riêng; không đề xuất trong hồ sơ này.
2. **Lớp:** tracker BDD cũ nhận bicycle (COCO 1), DETRAC không; GT DETRAC gộp van/others, GT BDD gộp 5 lớp xe và đưa
   `train` vào vùng bỏ qua. Detector mixed5 có lớp bicycle/motorcycle học từ dữ liệu trộn; detector COCO dùng ID COCO gốc.
3. **Evaluator/ignore:** `mot_crowd` (IoA > 0,5 với vùng class 13) vs `MotChallenge2DBox` + tô xám ảnh. Hai tập **không**
   cùng quy tắc chấm ⇒ không gộp số BDD với DETRAC, không so trực tiếp HOTA giữa hai tập.
4. **Thành phần GT:** motorcycle 902 / bicycle 4.135 box so với car 341.436 trên BDD val (R53 04 §3) ⇒ BDD val không đại
   diện giao thông hỗn hợp nhiều xe máy; GT tracking gộp một lớp nên không báo theo lớp.

### 3.3 NPZ detector R53 không phải cache tracking

- NPZ đánh giá R53: conf 0,001, `max_det` 300 (04 §1). Cache tracking test40: conf 0,05, `max_det` 3000, `meta.json` với
  `spec_sha256` từng sequence, cờ `truncated`, `cls_goc` (`giao_thuc_test40.json` `cache`; `det_cache.py:165–200`, :509–530).
- `max_det` 300 ở NPZ đánh giá có thể cắt bớt detection trước bước lọc conf của tracker, nên tính chất "lọc lúc load ==
  chạy trực tiếp" (`det_cache.py:31–35`) **không** được bảo đảm **[đọc mã]**. Định dạng, metadata và quy trình tô
  xám/ánh xạ của NPZ R53 **[chưa xác minh]** tương thích với `det_cache.open_cache`.
- ⇒ Hồ sơ này **không** đề xuất dùng NPZ R53 làm đầu vào tracking. Phương án tái dùng chỉ được đưa ra nếu lượt chuẩn bị
  chứng minh hợp đồng tương thích; mặc định là tạo cache tracking mới cho BDD.

### 3.4 `open_cache(strict=True)` không đủ chứng minh nguồn cache (sửa R60-F2)

Đọc mã `src/det_cache.py` **[đọc mã, chưa kiểm runtime]**:
- `check_compatible` (:213–230) chỉ kiểm conf yêu cầu ≥ conf cache, lớp yêu cầu ⊆ lớp cache, `max_det` yêu cầu ≤ cache.
  Không so `weights_sha256`, `dataset`, `imgsz`, `iou`, `half`, `device`, `versions` của `meta.json` với giá trị mong đợi.
- `sequence_spec_status` (:491–506) trả `legacy` khi thiếu `<video>.json` hoặc thiếu `spec_sha256` ở meta/sequence;
  `open_cache` (:509–530) chỉ dừng ở `mismatch` ⇒ sequence `legacy` **vẫn qua** ở strict.
- `open_cache` không băm NPZ, không so `n_frames` của `<video>.json` (:346) với timeline ảnh của dataset; `VideoDetections`
  chỉ chặn frame `truncated` (:446–451).

Bảng phân biệt cho cache BDD COCO cũ `yolov8n-coco_conf0.05_all_bddcocovid`:

| Thông tin | Khóa cũ thực có (`results/phase2/lock/lock_bdd_transfer.json`) | Chưa có bằng chứng trong khóa |
|---|---|---|
| Checkpoint | `cache_weights_sha256` `f59b3d83…` (:229) | Đối chiếu với `weights_sha256` trong `meta.json` của cache trên đĩa |
| Cấu hình sinh | `cache_conf` 0,05, `cache_max_det` 3000 (:230–231); LOCK §2 ghi "80 lớp COCO" | `imgsz`, `iou`, `half`, `batch`, `device`, `dataset` của cache (Grep khóa không thấy các khóa này); `spec_sha256` của `meta.json` |
| Phiên bản/thiết bị | `cache_versions`: ultralytics 8.4.141, torch 2.6.0+cu124, CUDA 12.4, cuDNN 90100, GTX 1650 (:232–238) | Trùng với `meta.json` thực tế |
| Dataset / nhãn | `dataset` `bdd100k_coco` cho thí nghiệm (:362), `manifest_sha256` `c3337134…`, `nguon_nhan_sha256` `074ff795…`, `gt_seqmap_sha256`, `gt_sha256_per_video` (16 ký tự) (:364–381) | Cache được sinh trên đúng nguồn nhãn/timeline đó |
| Provenance từng sequence | — (khóa không có hash `meta.json`, `<video>.json` hay NPZ) | `spec_sha256` từng sequence (có thể `legacy`), hash NPZ, `n_frames` so timeline, `n_truncated`, `n_unreadable` |
| Commit sinh cache | Khóa ghi commit thí nghiệm `d1f6d62a…` (dirty 3 file) | `git_commit`/`created` trong `meta.json` cache, và có trước thời điểm khóa hay không |

⇒ Khóa cũ **không** chứa đủ bằng chứng để chứng minh cache trên đĩa (nếu còn) đúng là cache đã dùng cho E5 và còn nguyên
vẹn. Tình trạng cache trên đĩa **[chưa xác minh]** (lượt này không mở `data/interim/det_cache/`).

---

## 4. Ràng buộc đã chốt áp dụng cho mốc 5 (không phải lựa chọn)

- Không chọn detector, conf, bộ lọc hay tham số bằng điểm UA-DETRAC test hoặc BDD val; không tuning trên hai tập này
  (PLAN "Quy tắc phạm vi"; LOCK BDD §7).
- Không gọi hai tập là chưa từng quan sát; ghi rõ BDD val đã dùng ở R43, R53 và E5, 3 sequence đã xem metric ở smoke.
- Không mặc nhiên kế thừa ma trận/quyền chạy từ R56, R59 hay khóa BDD cũ.
- Không mở lại: SORT c10, sửa mixin `assoc_log` (P74d51e88), fixture lọc kép (P3b9dd663), CI cho SORT, mở rộng R57, hiệu
  chỉnh đa so sánh (Pf01dab37), mở rộng kiểm cùng cảnh (P33464a9d), sửa đường ghi `kiem-khoa` R59, QJTF, Nano.

---

## 5. Phiếu quyết định mốc 5 (ô "Quyết định" để trống — người dùng điền)

Đơn vị đếm: 1 run = 1 tracker-config(-seed) × 1 detector × 1 điều kiện conf, chạy trên toàn bộ video của tập
(BDD 200 video, 39.973 ảnh mỗi run theo kiểm kê cũ). Một nhóm Kalman+PF đủ = CV, EKF, UKF, PF500 s0/s1/s2 = 6 run.

### M5-Q1 — Phạm vi detector trên BDD val (tracking)

| Phương án | Nội dung | Căn cứ | Số run BDD mới (với 6 run/nhóm, 2 điều kiện) |
|---|---|---|---|
| A | 3 detector như test40 (`yolov8n_coco`, `yolov8n_mixed5`, `yolo11n_mixed5`) | Cùng tập detector với E2/E4 ⇒ hai tập đánh giá cuối cùng phạm vi; lý do là **đối xứng giao thức**, không phải điểm R53 | 36 run × 200 video = 7.200 đơn vị run–video |
| B | Chỉ 2 detector mixed5; COCO lấy từ E5 | Giảm 12 run nhưng ghép run khác commit/mã (mâu thuẫn quy tắc cùng commit) trừ khi M5-Q5 cho phép | 24 run × 200 = 4.800 |
| C | Không tracking BDD mới; mốc 5 BDD = E1 (detector) + E5 (tracking COCO cũ) | Không tốn chạy; nhưng không có tracking mixed5 trên BDD | 0 |

Điều kiện kiểm chứng: hash checkpoint = `giao_thuc_test40.json` `detector.*.sha256`; ánh xạ lop5 cùng hàm
`detector_classes.anh_xa_model_sang_lop5`. Không dùng AP BDD/DETRAC để loại detector.
**Quyết định:** ☐ A ☐ B ☐ C ☐ khác: ______

### M5-Q2 — Bộ lọc / tracker trên BDD

| Phương án | Nội dung | Hệ quả số run / detector / điều kiện |
|---|---|---|
| A | CV, EKF, UKF, PF500 s0–s2 (như E2/E5) | 6 |
| B | CV, EKF, UKF (không PF) | 3 |
| C | Chỉ CV, EKF | 2 |

Căn cứ: E5 và E2 đều dùng A; PF500 thấp hơn Kalman ở cả hai đời (E5 `cap_so_sanh.md`, E3) — đây là mô tả, **không** dùng
để loại PF khỏi đánh giá cuối nếu không có lý do thiết kế. SORT trên BDD **không** nằm trong phiếu (không được nêu trong
plan R60; nếu muốn phải là đề xuất riêng). Kiểm chứng: hash YAML bộ lọc + `config.py` + cấu hình hiệu lực từ instance như
K1 R59.
**Quyết định:** ☐ A ☐ B ☐ C ☐ khác: ______

### M5-Q3 — Điều kiện confidence

| Phương án | Nội dung | Hệ quả |
|---|---|---|
| A | c25 và c10 (như E2/E5) | × 2 |
| B | Chỉ c25 | × 1 |

Căn cứ: hai điều kiện được khóa ở E5 và R54 **trước** khi xem kết quả tương ứng; chọn một điều kiện bằng điểm BDD/DETRAC là
vi phạm quy tắc. Lưu ý: ở c25, tầng điểm thấp ByteTrack rỗng (R59 P2) — phương án B không đo association điểm thấp.
**Quyết định:** ☐ A ☐ B

**Tổng số run BDD mới theo tổ hợp** (Q1 × Q2 × Q3): A/A/A = 36; A/A/B = 18; A/B/A = 18; A/C/A = 12; B/A/A = 24;
B/A/B = 12; C = 0. Thời gian chạy BDD **chưa ước lượng** trong lượt này (cần đo ở smoke; `bao_cao/toc_do.md` của E5 chưa
đọc). Cache mới cần: 1 cache BDD cho mỗi detector mới (3 với Q1-A trừ khi COCO tái dùng cache cũ theo Q5).

### M5-Q4 — Quy ước thời gian, lớp và evaluator cho BDD

| Hạng mục | Phương án A (giữ khóa BDD cũ) | Phương án khác (cần lý do riêng) |
|---|---|---|
| fps / buffer | 5 fps, `track_buffer_sec` 1,2 ⇒ 6 bước | — (đổi = thí nghiệm mới, ngoài mốc) |
| `elapsed_mode` | `steps` | `skip` như test40 (theo đọc mã tương đương khi không có frame lỗi; phải kiểm bằng fixture) |
| Lớp tracker | `[1, 2, 3, 5, 7]` | `[2, 3, 5, 7]` như DETRAC (loại bicycle trong khi GT BDD tính bicycle) |
| Evaluator | `mot_crowd`, split `BDDCOCOVID-val`, GT/seqmap hiện có (`181a419d…`) | Xuất lại GT trong namespace mới, kiểm trùng hash với khóa cũ |
| `assoc_log` | `false` (E5) | `true` (như R54) |
| Tham số nhiễu | Theo bước của DETRAC (không đổi) | — (đổi = việc mới) |
| Ảnh có trong timeline nhưng không annotation (615 theo `val_report.json`, mục 3.1) | Giữ trong timeline và chấm như khóa cũ; báo cáo ghi số ảnh và giới hạn "chưa rõ nguyên nhân" | Bất kỳ cách loại/đánh dấu khác = thay chính sách đánh giá, cần người dùng duyệt riêng; không được chọn bằng điểm BDD |

Kiểm chứng: fixture cho `steps` vs `skip`, kiểm `n_steps` = 1 ở mọi frame BDD; parity cache–tracker (số frame = timeline);
hash GT/seqmap; evaluator chọn theo dataset `bdd100k_coco` → `mot_crowd`; ở bước chuẩn bị, đếm tĩnh từ file nhãn (không
dùng metric) số ảnh không annotation theo video, đối chiếu tổng với 615 của báo cáo cũ; fixture nhỏ xác nhận cách
`mot_crowd` xử lý frame GT trống.
**Quyết định:** fps/buffer ☐ A; `elapsed_mode` ☐ steps ☐ skip; lớp ☐ [1,2,3,5,7] ☐ [2,3,5,7]; GT ☐ dùng hiện có ☐ xuất lại;
`assoc_log` ☐ false ☐ true; ảnh không annotation ☐ giữ như khóa cũ ☐ khác: ______

### M5-Q5 — Điều kiện tái dùng kết quả cũ

| Đối tượng | Phương án | Điều kiện tối thiểu nếu tái dùng |
|---|---|---|
| E5 (12 run BDD COCO cũ) | (a) chỉ trích dẫn lịch sử, chạy lại COCO theo giao thức mới; (b) tái dùng làm phần COCO của mốc 5 | (b) cần: cache cũ đạt **toàn bộ** điều kiện của dòng cache dưới; hash YAML/`config.py`/mã tracker hiện hành khớp hoặc khác biệt được giải thích; chấm lại MOT cũ bằng evaluator hiện hành trùng byte; tái lập tracker 1 video trùng byte. Thiếu một điều kiện ⇒ không tái dùng |
| Cache BDD COCO cũ (`yolov8n-coco_conf0.05_all_bddcocovid`) | (a) tạo mới; (b) tái dùng | (b) — `open_cache` strict đạt là **cần nhưng không đủ** (mục 3.4). Điều kiện dự kiến, kiểm ở lượt chuẩn bị: (1) `meta.json` `weights_sha256` = `f59b3d83…` và = hash checkpoint thực tế; (2) `dataset` = `bdd100k_coco`, cấu hình sinh (`imgsz`, `iou`, `conf` 0,05, `max_det` 3000, `classes` null, `half`, `batch`, `device`, `versions`) khớp khóa ở các trường khóa có, các trường khóa không có được ghi lại và người dùng chấp nhận; (3) mọi sequence trong 200 video có `spec_sha256` = `meta.json` (trạng thái `ok`, **không** `legacy`); (4) `n_frames` mỗi sequence = timeline ảnh từ nguồn nhãn `074ff795…`, `n_truncated` = 0, `n_unreadable` ghi lại; (5) `created`/`git_commit` của cache trước thời điểm khóa E5; (6) băm NPZ/JSON/`meta.json` ghi vào khóa mới. **Thiếu provenance (bất kỳ `legacy`, hoặc không chứng minh được (1)–(5)) ⇒ dừng phương án (b) và mọi phương án phụ thuộc (E5 (b)), báo người dùng quyết định** — không mặc nhiên tạo mới hay chấp nhận. Suy luận GPU mới có thể lệch nhẹ nên cache mới ≠ cũ không phải lỗi |
| NPZ detector R53 | Không tái dùng làm cache tracking (mục 3.3) | Chỉ xét lại nếu có bằng chứng hợp đồng tương thích — ngoài lượt này |
| Phần UA-DETRAC (E1–E4) | (a) dùng nguyên làm phần UA-DETRAC của mốc 5, không chạy lại; (b) chạy lại | (a): trích đúng phiên/khóa đã approve, không tính lại metric, giữ nguyên giới hạn (không CI cho R56 điểm, CI R57 thăm dò, SORT chỉ c25) |

**Quyết định:** E5 ☐ a ☐ b; cache COCO ☐ a ☐ b; UA-DETRAC ☐ a ☐ b

### M5-Q6 — Thống kê dự kiến cho BDD

| Phương án | Nội dung | Căn cứ |
|---|---|---|
| A | Chỉ điểm COMBINED + hiệu số, không CI (như R56/R59) | Đơn giản; không suy luận |
| B | A + bootstrap ghép cặp theo video, B = 5000, seed 0, cặp khai báo **trước khi chạy** | E5 đã khóa cách này; giao thức BDD có thể tiền đăng ký (khác R57 là hậu nghiệm) |

Nếu B, cần chốt thêm: (i) danh sách cặp (E5: 5 cặp; R57: 6 hiệu số/nhóm); (ii) quy tắc PF trong bootstrap — trung bình
thống kê đủ theo seed rồi tái tổ hợp (E5) **hay** trung bình COMBINED 3 seed trong từng mẫu (R57); (iii) quy tắc PF ở bảng
chính — trung bình metric theo seed (E5, R56). Không đề xuất hiệu chỉnh đa so sánh (đã bị từ chối ở R57); CI ghi là riêng
lẻ. Không so detector bằng CI.
**Quyết định:** ☐ A ☐ B — nếu B: cặp ______; PF bootstrap ☐ thống kê đủ ☐ COMBINED/mẫu

### M5-Q7 — Đường mã chuẩn bị

| Phương án | Nội dung | Hệ quả |
|---|---|---|
| A | Wrapper mới riêng cho BDD (namespace mới), gọi helper hiện có (`det_cache`, `run_tracking`, `run_trackeval`, `bdd100k_coco`) không sửa helper | Cần mã + fixture mới cho ánh xạ lop5 trên BDD (đường R54 gắn DETRAC) |
| B | Dùng `src/run_experiment.py` + YAML như E5 | Không có ánh xạ lop5 cho mixed5 qua `det_cache` CLI **[đọc mã]** ⇒ chỉ đủ cho COCO |

**Quyết định:** ☐ A ☐ B ☐ khác: ______

### M5-Q8 — Báo cáo tổng hợp mốc 5

Phương án: một báo cáo cuối gồm hai phần tách biệt (UA-DETRAC: E1–E4; BDD: E1 + tracking mới/E5 theo Q5), không gộp số
giữa hai tập, ghi đủ giới hạn mục 3.2. **Quyết định:** ☐ đồng ý ☐ khác: ______

---

## 6. Trình tự dự kiến sau khi người dùng quyết định (mỗi bước cần duyệt riêng)

1. **Chuẩn bị mã/fixture/giao thức** (lượt riêng, chưa chạy GPU): giao thức JSON BDD mới (namespace riêng, ví dụ
   `configs/phase2/r6x_bdd/`), wrapper + fixture CPU trên dữ liệu tổng hợp: ánh xạ lop5 → COCO và `cls_goc` trên BDD;
   `steps`/`skip`; lọc conf → lớp → `[:300]`; evaluator `mot_crowd` với class 13; chặn `truncated`; quy tắc PF/thống kê
   theo Q6. Review.
2. **Khóa + smoke có review (cổng A):** tiền kiểm (hash checkpoint, YAML, `config.py` byte thô, GT/seqmap, manifest 200
   video, phiên bản thư viện), tạo cache (GPU — chỉ khi bước này được duyệt ghi rõ), parity cache–timeline, smoke vài video
   ngắn, chấm được, khóa; Codex review A.
3. **Lượt đầy đủ (cổng B), duyệt riêng:** mở phiên ràng buộc khóa A + review A; chạy theo video có tiếp tục; chấm; tái lập
   tracker 1 video + chấm lại trùng byte; hậu kiểm; tổng hợp; kiểm khóa cuối; Codex review B.

**Thiết kế đường ghi (tránh lặp hạn chế `chuan_bi/` của R59):**
- Mọi bước nhận **vùng ghi theo cổng** tường minh: A ghi `.../chuan_bi/<phiên>`, B ghi `.../thuc_thi/<phiên>/...`;
  `kiem-khoa` gọi trong B ghi vào **thư mục phiên B** (ví dụ `thuc_thi/<phiên>/kiem_khoa/`), không vào `chuan_bi/`.
  write_paths của plan B phải liệt kê đủ các vùng này.
- **Log:** `cli_log.json` (argv, exit, stdout/stderr, Python) trong cùng phiên; hash ghi cả byte thực tế trên đĩa (bài học
  LF/CRLF của R56).
- **Bytecode:** đặt `PYTHONDONTWRITEBYTECODE=1` hoặc `PYTHONPYCACHEPREFIX` trỏ vào vùng ghi của phiên, để không tạo
  `__pycache__` ngoài write_paths; fixture kiểm không có file `.pyc` mới ngoài vùng ghi.
- **Cấu hình thư viện:** thư mục cấu hình mà thư viện ghi (Ultralytics `settings.json`, và tương tự nếu có) trỏ vào
  `<phiên>/_cau_hinh_thu_vien/`; R59 đã có thư mục này trong phiên kiểm khóa
  (`_cau_hinh_thu_vien/ultralytics/Ultralytics/settings.json`, báo cáo 03 §2a) — cơ chế đặt biến môi trường của R59
  **[chưa xác minh]** trong lượt này; hash lại sau phiên.
- **Kiểm khóa:** kết quả kiểm khóa thuộc gói bằng chứng của cổng gọi nó; khóa ghi danh sách file và vùng ghi được phép;
  fixture kiểm rằng mỗi bước chỉ ghi vào vùng khai báo (so danh sách file trước/sau).
- Job dài: chạy foreground theo đoạn có giới hạn thời gian và tiếp tục theo video (như R59 `--gioi-han-giay`), không để
  tác vụ nền sau lượt.

---

## 7. Tự kiểm của lượt R60A

| Kiểm | Lệnh / cách | Kết quả |
|---|---|---|
| HEAD | `git rev-parse HEAD` | `117d62da027e8ce3e4c9f455d576a924b4d82bbb` |
| Phạm vi file | `git status --short handoff/` | `M handoff/PLAN_DA_CHOT.md` (đã có thay đổi chưa commit từ trước lượt), `?? handoff/logs/review60/` mới; các mục khác trong danh sách là thay đổi có sẵn trước lượt (khớp git status đầu phiên) |
| PLAN chỉ đổi mục 0 | `git diff -U0 handoff/PLAN_DA_CHOT.md` (so với HEAD) + Grep vị trí | Sửa của lượt nằm ở dòng 13–37 (khối R60A + tiêu đề R59B), dòng 72 (đánh dấu "Bước tiếp" R59B đã làm) và dòng 1148 (đánh dấu "Bước tiếp theo (23/10)" là lịch sử); `## Thứ tự các mốc` ở dòng 1235 ⇒ mọi sửa nằm trong mục 0. Diff so với HEAD còn có hunk ở dòng 1237/1241 (bảng mốc) — là thay đổi chưa commit **có trước lượt** (bảng ghi "dòng 3–4 cập nhật 22/10/2026, người dùng cho phép"); lượt này không có baseline trước-lượt để diff riêng, không sửa bảng |
| Không chạy | Không Python/fixture/tracker/evaluator/bootstrap/kiem-khoa; không băm | Đúng theo lượt |

**Vòng sửa 2 (review_001 `r60_ho_so_moc5_20261023_183448`, verdict revise, R60-F1/F2 mức P2):**

| Lỗi | Sửa | Nguồn đọc tĩnh |
|---|---|---|
| R60-F1 | Mục 3.1 dòng GT/ignore: bỏ kết luận "không có tương đương D4"; tách ảnh thiếu trên đĩa (0) với ảnh không annotation (615); nguyên nhân/cách chấm **[chưa xác minh]**; thêm hàng chính sách + điều kiện kiểm chứng ở M5-Q4 (ô trống); thêm giới hạn ở E5 | `results/phase2/bdd_cocovid_check/val_report.json:8–15`; `src/mot_datasets/bdd100k_coco.py:152–160`; LOCK §5 dòng 75 |
| R60-F2 | Mục 3.4 mới: giới hạn của `open_cache` strict + bảng "khóa cũ thực có / chưa có bằng chứng"; M5-Q5 dòng cache: 6 điều kiện dự kiến, thiếu provenance ⇒ dừng phương án phụ thuộc, hỏi người dùng | `src/det_cache.py:213–230`, :340–350, :438–451, :491–530; `lock_bdd_transfer.json:1–23`, :228–238, :362–381 (Grep khóa không có `imgsz/iou/half/device/batch`, không hash `meta.json`/NPZ) |

Kiểm của vòng: `git rev-parse HEAD` → `117d62d…` (khớp); `git status --short handoff/` → chỉ PLAN (đã `M` từ trước) và
`review60/` đổi so với đầu phiên; `git diff -U0 handoff/PLAN_DA_CHOT.md` → sửa vòng này nằm ở dòng 13–14 và 32–43 (khối R60A), mục 0
kết thúc trước `## Thứ tự các mốc` (dòng 1245); `git diff --check handoff/PLAN_DA_CHOT.md` → không báo lỗi whitespace (chỉ
cảnh báo LF→CRLF). Không có bản PLAN baseline đầu mốc trong repo để diff riêng phần sửa của lượt (bản ngoài repo không dùng
được với `git diff --no-index` theo quyền lượt này). Không đổi ô quyết định nào; không chạy Python, không mở cache.

**Chưa làm / giới hạn:** không mở `bao_cao/toc_do.md`, `pf_seed.md`, `lop_gt_det.md` của E5; không đọc phần phiên bản thư
viện/hash YAML trong khóa R54/R56 để so với khóa BDD cũ; không kiểm cache BDD cũ trên đĩa; verdict review của E5 chưa xác
định trong lượt này. Các điểm này được ghi **[chưa xác minh]** ở trên và phải kiểm ở lượt chuẩn bị nếu người dùng chọn
phương án phụ thuộc chúng.
