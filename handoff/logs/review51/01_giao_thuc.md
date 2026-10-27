# R51A — Giao thức tiền đăng ký: so sánh có kiểm soát trên UA-DETRAC dev12

Ngày lập: 2026-10-16. Người triển khai: Claude (`claude-opus-5-5`, effort high). Reviewer: Codex ở phiên riêng.
Bước này chỉ đọc và kiểm. Chưa chạy tracking, chưa chấm lại metric nào. Mọi số liệu dưới đây lấy từ:

- Script chỉ đọc: `src/r51_controlled/audit_protocol.py`. Test: `src/r51_controlled/test_protocol.py` (19/19 PASS sau
  vòng sửa R51A-F1/F2 ngày 18/10, xem mục 11).
- Đầu ra: `results/phase2/r51_controlled/protocol/`, gồm `audit_tong_hop.json`, `kiem_ke_run.{csv,json}`,
  `frame_khong_nhan_theo_{video,doan}.csv`, `frame_khong_nhan_danh_sach.json`,
  `box_tracker_tren_frame_khong_nhan.csv`, `so_khop_dau_ra.csv`, `ma_tran_tai_dung.csv`, `danh_sach_run_B.json`.

Sau khi reviewer approve, mục 2–8 được cố định. Bước B chỉ thực thi đúng các mục này. Muốn đổi gì phải hỏi người dùng.

## 1. Hai câu hỏi độc lập

| | Câu hỏi 1 (Q1): bộ lọc trong ByteTrack | Câu hỏi 2 (Q2): SORT so với ByteTrack |
|---|---|---|
| Giữ cố định | detection đầu vào, ngưỡng, association, vòng đời ByteTrack, evaluator | detection đầu vào, **lớp bộ lọc và tham số bộ lọc**, evaluator |
| Thay đổi | chỉ bộ lọc: KF-CV, EKF-CTRV, UKF-CTRV, PF-CTRV N=100, PF-CTRV N=100 + K1 | chỉ association và vòng đời: SORT ↔ ByteTrack |
| Đối chứng chính | ekf−cv, ukf−cv, ukf−ekf, pf100−ekf, pf100k1−pf100, pf100k1−ekf | bt-cv−sort-cv30, bt-ekf−sort-ekf30 |

Hai câu hỏi không được ghép thành một bảng xếp hạng. Không suy hiệu ứng bộ lọc từ Q2, cũng không suy hiệu ứng
association từ Q1.

## 2. Đầu vào chung (mọi đối chứng của cả hai câu hỏi)

| Thành phần | Giá trị đã khóa | Bằng chứng |
|---|---|---|
| Dataset / split | UA-DETRAC phần train, dev12 nội bộ, 12 video theo `results/phase2/manifests/detrac_dev.json` | `MANIFEST` trong script |
| Ảnh / frame có nhãn | 16.919 ảnh / 16.347 frame `<frame>` | `audit_tong_hop.json` |
| Detector / cache | `yolov8n-coco_conf0.05_all`; weights sha256 `f59b3d83…`, npz `9d596eb4…`, meta `870cb45f…`; vùng bỏ qua được tô trước khi detect (`det_cache.mask_ignored_regions=True`) | `kiem_ke_run.csv` (một giá trị npz duy nhất trên mọi run có eval_meta) |
| Điều kiện A | `conf > 0,25`, lớp COCO `[2, 3, 5, 7]`, `max_det 300`, fps 25, `elapsed_mode skip` | `run_meta.json` từng run |
| Đơn vị thời gian | 0,04 s là khoảng **danh nghĩa** giữa hai ảnh (`run_meta.dt_nominal_s`, fps 25). Bước thời gian **nội bộ của bộ lọc** là 1 bước = 1 frame (`config.CTRV_DT = 1.0`, `config.py:268–272`; KF-CV của ultralytics cũng dùng 1 bước). Vận tốc tính theo px/bước, turn rate theo rad/bước. `elapsed_mode skip`: mỗi frame là một bước | `kiem_ke_run.csv` cột `dt_danh_nghia_s`, `bl_dt_buoc` |
| Mặc định bộ lọc | Không YAML nào ở đây có khóa `ctrv_*`/`ukf_*`, nên CTRV/UKF dùng mặc định `src/config.py` (giá trị ở mục 5). `config.py` **không** nằm trong `CODE_FILES` của `run_fingerprint`; provenance kiểm riêng: sha256 (LF) `0e3f8b75…` giống nhau ở commit lúc chạy của mọi run, ở HEAD và ở cây làm việc; lần sửa cuối `c7713b1` (08/09) | `audit_tong_hop.json` → `config_py`; cột `config_py_*` |
| Không đổi khi chạy | `assoc_log=false`, `pf_diag=false`, không override `track_buffer_sec` | `danh_sach_run_B.json` |
| Backend | ultralytics 8.4.141, numpy 2.4.6, scipy 1.17.1, lap 0.5.13 | `audit_tong_hop.json` → `ma_hien_tai.backend` |
| GT | `data/processed/DETRAC-all/<v>/gt/gt.txt` sha256 `07de8402…`, seqinfo `024df16d…` (đã tính lại hôm nay, trùng eval_meta kc48) | `gt_hien_tai_trung_eval_meta = true` |
| Evaluator | TrackEval 1.3.0, `MotChallenge2DBox` gốc (`run_trackeval.DATASET_EVALUATOR['detrac']='mot'`), IoU 0,5, HOTA + CLEAR + Identity | `src/run_trackeval.py:80-134` |
| Mã | HEAD `117d62d`; hash module (quy ước `run_fingerprint`) ở `audit_tong_hop.json` → `ma_hien_tai.modules` | |

Lưu ý điều kiện A: `conf_min = 0,25` trùng `track_high_thresh = 0,25` của ByteTrack. Vì vậy vòng ghép thứ hai của
ByteTrack (detection có `0,1 < score < 0,25`) **luôn rỗng** trong mọi run ở đây. Q2 không kiểm được ý tưởng "ghép
detection điểm thấp" của BYTE; xem mục 5.

## 3. 572 ảnh dev12 không có `<frame>`: kiểm trực tiếp

### 3.1 Chuỗi xử lý (đọc mã)

1. **XML** (`data/extracted/DETRAC-Train-Annotations-XML/<v>.xml`): `parse_detrac_xml.py:215` chỉ duyệt các thẻ
   `<frame>` có mặt. Ảnh không có thẻ thì không sinh dòng nào.
2. **Chuẩn bị GT** (`to_motchallenge.py:136-175`): `gt.txt` chỉ có dòng của frame có trong XML. `seqLength` lấy theo
   **số ảnh trên đĩa** (`count_images`). Kết quả: 12/12 video có `seqLength` bằng số ảnh.
   `ignored_regions.txt` được ghi riêng, nhưng lớp `mot` không đọc file này.
3. **Tracker**: chạy trên mọi ảnh của video. Bằng chứng: có box tracker trên chính các frame thiếu nhãn (mục 3.2).
4. **Evaluator** (`MotChallenge2DBox`, TrackEval 1.3.0): duyệt timestep `1..seqLength`. Frame không có dòng GT thì
   mọi box tracker ở đó là FP. Fixture `test_evaluator_dem_fp_o_frame_khong_gt` dựng 1 xe đúng vị trí ở 4 frame; GT
   thiếu frame 3. Kết quả: TP 3, FP 1, IDSW 0; DetA = AssA = HOTA = 0,75; IDF1 = 6/7. **AssA cũng giảm**, vì box của
   cùng ID ở frame không GT nằm trong FPA. Ước lượng ban đầu của tôi cho rằng chỉ DetA đổi; test đã bác giả định này,
   nên script chỉ giữ phần CLEAR tính chính xác (`clear_after_mask`). Bỏ box ở frame đó thì HOTA = IDF1 = 1
   (`test_mask_r1_bo_frame_khong_nhan`). Frame không GT và không có box thì không ảnh hưởng gì
   (`test_frame_rong_khong_box_khong_anh_huong`).

### 3.2 Bằng chứng trực tiếp trên dữ liệu

- 572 ảnh thuộc 5/12 video, gom thành **8 đoạn liên tục, đều nằm giữa video** (không đoạn nào ở đầu hay cuối). Chi
  tiết ở `frame_khong_nhan_theo_doan.csv`:
  MVI_39851 162–295 (134), MVI_39931 677–835 (159) và 977–1005 (29), MVI_40213 1167–1170 (4) và 1287–1290 (4),
  MVI_40992 487–490 (4) và 1557–1590 (34), MVI_63544 892–1095 (204).
- Đếm lại độc lập bằng regex trên văn bản XML và `os.listdir` trên thư mục ảnh: khớp CSV, tổng 572
  (`test_dem_frame_doc_lap_regex`).
- **Không có frame XML nào có danh sách target rỗng**: 0/16.347 (`frame_xml_khong_target`). Như vậy, ít nhất trên
  dev12, định dạng XML bỏ hẳn những frame không có target, thay vì ghi frame rỗng.
- Ở **cả 8 đoạn**, mọi track GT ở frame a−1 đều kết thúc đúng tại a−1, và mọi track GT ở frame b+1 đều bắt đầu đúng
  tại b+1. Không track nào xuyên qua đoạn (`n_xuyen = 0`). Ngay trước và ngay sau mỗi đoạn chỉ có 1–2 xe có nhãn.
- Xem trực tiếp 3 ảnh (MVI_39931 #750, MVI_63544 #1000, MVI_39851 #230): làn đường trong vùng gán nhãn trống. Chỉ
  thấy xe hai bánh / người đi bộ sát mép trái (39931), xe đỗ phía xa (63544), hoặc xe rất xa ở mép trên (39851).
- Box tracker trên 572 frame của run neo ByteTrack-EKF: 371/91.500 box (0,41 %). Trong đó 290 box ở MVI_39931
  677–835, có tâm trung vị (68; 218) px, tức làn sát mép trái, khớp vị trí xe hai bánh trên ảnh #750. 34 box ở
  977–1005 có tâm y ≈ 60 và cao ≈ 17 px (xe xa ở mép trên). 41 box ở MVI_63544 (1 ID). Các đoạn còn lại có 0–5 box.
  0 box có IoA > 0,5 với vùng bỏ qua. Trên mọi run ứng viên, tỉ lệ box rơi vào 572 frame là 0,36–0,41 %
  (`kiem_ke_run.csv`, cột `ty_le_box_frame_khong_nhan`).

### 3.3 Kết luận và giới hạn

Các bằng chứng trên **phù hợp với** cách hiểu: đây là frame không có đối tượng thuộc diện gán nhãn, tức âm tính theo
quy ước nhãn UA-DETRAC. Chúng không có dấu hiệu là frame bị bỏ sót nhãn trong khi vẫn có xe. Các box tracker ở đó
cùng loại với FP ở frame có nhãn: xe hai bánh (UA-DETRAC không gán nhãn) và xe ở xa.

Giới hạn: mới xem 3/572 ảnh; không loại trừ được việc người gán nhãn sót xe trong một phần các đoạn. Kết luận này chỉ
áp dụng cho dev12. Các frame tương tự ở test40 (173) và train60 (1.706) chưa kiểm và nằm ngoài phạm vi.

### 3.4 Quy tắc chấm (cố định)

- **R0 — chính:** giữ nguyên evaluator hiện hành (`MotChallenge2DBox`, GT và seqinfo như trên). 572 frame được chấm
  như frame âm tính. Đây là quy tắc đã dùng cho mọi số cũ, nên không đổi metric chuẩn.
- **R1 — độ nhạy:** wrapper mới ở `src/r51_controlled/` bỏ mọi dòng tracker có frame nằm trong
  `frame_khong_nhan_danh_sach.json`, giữ nguyên GT và seqinfo, rồi chấm bằng cùng evaluator. R1 tương đương xem 572
  frame là "không chấm".
- Áp dụng **đồng nhất cho mọi run** (mới và tái dùng) của cả Q1 lẫn Q2. Ghi vào namespace mới
  `results/phase2/r51_controlled/eval/`. Không ghi đè `results/trackeval/…`.
- Ảnh hưởng đã biết trước khi chấm, chính xác cho CLEAR: FP giảm đúng bằng số box rơi vào 572 frame. Ví dụ EKF:
  12.103 → 11.732, MOTA 0,5719 → 0,5751; xem các cột `r1_*` của `kiem_ke_run.csv`. HOTA, AssA và IDF1 chỉ biết khi
  chấm lại ở B, vì các box này còn nằm trong FPA/IDFP (mục 3.1).
- Quy tắc báo cáo: kết luận chính theo R0. Nếu trạng thái CI (chứa 0 hay không) của bất kỳ đối chứng chính nào khác
  giữa R0 và R1, phải nêu cả hai và đánh dấu đối chứng đó là "nhạy với quy tắc frame không nhãn".

## 4. Kiểm kê và tiêu chí tái dùng

### 4.1 Tiêu chí

Một run cũ chỉ được tái dùng khi đạt **đủ** các điều kiện sau. Trùng tên cấu hình, hoặc trùng byte file MOT, đều
**không đủ** nếu đứng riêng. **Thiếu bằng chứng là không đạt**: mỗi điều kiện kiểm cả sự có mặt của trường lẫn giá
trị; không có phép kiểm nào bỏ qua video, trường hay run thiếu dữ liệu. Tiêu chí 1–6 được cài trong hàm
`kiem_provenance` (cột `tai_dung_du_dieu_kien`, `ly_do_khong_dat` của `kiem_ke_run.csv`), có fixture dương tính và
25 fixture âm tính, mỗi fixture làm hỏng đúng một trường (`test_provenance_*`).

1. `run_meta.status = complete`; `run_meta.videos` và `eval_meta.tracking.per_video` **bằng đúng** tập 12 video của
   manifest (không thiếu, không thừa), `n_missing = 0`, đủ 12 file đầu ra.
2. Cùng cache: `eval_meta.cache` có đủ `cache_id`, `meta_sha256`, `npz_sha256`, `weights_sha256`; cả bốn bằng dấu vân
   tay cache **tính lại trên đĩa hôm nay** (`run_fingerprint.cache_fingerprint`); `run_meta.cache_id` trùng. Cùng điều
   kiện A (mục 2).
3. sha256 của YAML tracker lúc chạy có mặt và bằng sha256 của file hiện tại. Với ByteTrack: tham số ghép và vòng đời là
   `0,25 / 0,1 / 0,25 / 0,8 / fuse_score / 30`.
4. Mã và backend: `run_meta.code.modules` có đủ module liên quan (`run_tracking`, `det_cache`, `time_units`,
   `ul_backend`, `bytetrack_np` cộng module theo loại bộ lọc); từng hash bằng HEAD `117d62d` và bằng hash ghi trong
   `eval_meta` (mã lúc chạy = mã lúc chấm). `run_meta.code.backend` có đủ 5 trường (`backend`, `ultralytics`, `numpy`,
   `scipy`, `lap`) và bằng backend hiện tại. `config.py` tại commit lúc chạy bằng HEAD và cây làm việc.
5. Hash đầu ra theo quy ước hiện hành (`.txt` chuẩn hóa LF) khớp `eval_meta.tracking.per_video` trên **đủ** 12 video
   (`hash_dau_ra_khop`; hash rỗng hoặc thiếu video là không khớp).
6. `eval_meta.gt` có `gt_sha256`, `seqinfo_sha256`, `n_missing_gt = n_missing_seqinfo = 0`, `gt_per_video` đủ 12
   video, và dấu vân tay trùng GT hiện tại. Run thiếu dấu vân tay GT được liệt kê ở
   `audit_tong_hop.json → run_thieu_dau_van_tay_gt`, không bị loại khỏi phép kiểm.
7. Ở bước B: run neo chạy lại tại HEAD phải trùng sha256 12/12 với run tái dùng cùng họ (mục 6).

Giới hạn còn lại: hash module và `config.py` chỉ chứng minh mã **đã commit** tại commit lúc chạy; thay đổi chưa commit
lúc chạy chỉ được bắt bởi hash module trong `run_meta` (có cho module, không có cho `config.py`). Bằng chứng cuối cùng
cho phần này là tiêu chí 7 (chạy lại neo tại HEAD).

### 4.2 Kết quả kiểm kê (`kiem_ke_run.csv`, `ma_tran_tai_dung.csv`)

- Cổng `kiem_provenance`: **16/16** run `kc48_pf_dev` (ekf; pf100, pf100-k1, pf100-k2, pf500, pf500-k1 × seed 0/1/2)
  đạt tiêu chí 1–6; **mọi** run cũ khác (10 run expA/expB) không đạt, lý do ghi ở `ly_do_khong_dat`
  (`test_tien_quyet_tai_dung`). Ma trận viết tay ở mục 4.3 được test đối chiếu với cổng (`test_ma_tran_tai_dung_khop_cong`). Đầu ra ekf trùng sha256 12/12 với `expA c25-ekf` (`a64cb75`) và `expB bt-ekf`
  (`1c7ce8c`). pf100 trùng 12/12 với `expA c25-pf100` theo từng seed (`so_khop_dau_ra.csv`). Như vậy đầu ra EKF và
  PF ổn định qua ba commit.
- `expA c25-cv` / `c25-ukf` (`a64cb75`): lệch module `run_tracking.py`, `det_cache.py` (UKF còn lệch thêm
  `tracker_ctrv.py`, `lost_*.py`). `eval_meta` (`28ca79b`) không có dấu vân tay GT; hash đầu ra theo quy ước cũ trước
  `935a445`. Vì vậy **không tái dùng**.
- `expB_fix1 sort-cv30` / `sort-ekf30` (`1c7ce8c`): `sort_tracker.py` lúc chạy `f59ace28…` khác HEAD `a4ab4b26…`;
  thêm `run_tracking.py`, `det_cache.py` (và `tracker_ctrv.py` với ekf). Vì vậy **không tái dùng**.
- `sort-orig1`: dùng bộ lọc khác (SortKF7 7-D, `max_age 1`), không thỏa "cùng bộ lọc", nên không vào so sánh Q2.

### 4.3 Ma trận

| Đối chứng | Seed | Quyết định | Nguồn |
|---|---|---|---|
| bt-ekf (Q1 + Q2) | — | tái dùng, có neo | `p2-kc48_pf_dev-A-ekf` |
| bt-pf100 | 0/1/2 | tái dùng | `p2-kc48_pf_dev-A-pf100-s{0,1,2}` |
| bt-pf100-k1 | 0/1/2 | tái dùng, có neo s0 | `p2-kc48_pf_dev-A-pf100-k1-s{0,1,2}` |
| bt-pf100-k2, bt-pf500, bt-pf500-k1 | 0/1/2 | chỉ tái dùng, phụ lục | `p2-kc48_pf_dev-A-…` |
| bt-cv (Q1 + Q2) | — | **chạy mới** | `bytetrack.yaml` (`395701d947a7`) |
| bt-ukf | — | **chạy mới** | `configs/bytetrack_ukf_ctrv.yaml` (`5d1286e41ebe`) |
| sort-cv30 | — | **chạy mới** | `configs/phase2/trackers/sort_cv_age30.yaml` (`8d1ea5b98684`) |
| sort-ekf30 | — | **chạy mới** | `configs/phase2/trackers/sort_ekf_ctrv_age30.yaml` (`e1b055b31cea`) |

Số cũ của các run không tái dùng vẫn được giữ nguyên, chỉ dùng để đối chiếu thông tin (sha256 đầu ra mới so với cũ).
Mọi YAML là cấu hình đã khóa và giữ nguyên. Nếu bước B cần file cấu hình thí nghiệm trong
`configs/phase2/r51_controlled/`, file đó chỉ diễn đạt lại các trường ở mục 2 và trỏ tới đúng các YAML trên.

## 5. Q1 — bộ lọc trong cùng ByteTrack

- Cố định: lớp tracker ByteTrack của ultralytics 8.4.141 (CV dùng `BYTETracker`; CTRV dùng `CTRVByteTracker`, với
  `LostAwareAssociation` mặc định tắt). Tham số ghép và vòng đời giống hệt nhau (mục 4.1.3). Cả hai lớp đều đặt vận tốc
  chiều cao bằng 0 khi predict cho track không ở trạng thái Tracked (`byte_tracker.py:82`, `tracker_ctrv.py:101-103,
  459-460`).
- Thay đổi: bộ lọc và mô hình chuyển động. Box dự đoán của bộ lọc đi vào IoU khi ghép; đây là hệ quả của bộ lọc,
  không phải thay đổi association.
- Thông số **hiệu lực** (đã khóa, không thay). Giá trị đọc từ instance bộ lọc tạo bằng đúng đường của tracker
  (`get_kalmanfilter` / `SORT._new_filter`) trên `tracker_cfg` của từng run, không suy từ YAML; cột `bl_*` của
  `kiem_ke_run.csv`, test `test_bang_thong_so_hieu_luc_kc48`. Đơn vị theo bước (1 bước = 1 frame, mục 2):

  | Thông số | KF-CV (XYAH 8-D) | EKF-CTRV | UKF-CTRV | PF-CTRV |
  |---|---|---|---|---|
  | dt nội bộ | 1 bước | 1 bước | 1 bước | 1 bước |
  | Nhiễu vị trí / vận tốc (hệ số × h) | 1/20, 1/160 (mặc định ultralytics) | 1/20, 1/160 | 1/20, 1/160 | 1/20, 1/160 |
  | `std_theta` [rad/bước], `std_omega` [rad/bước²] | — | 0,010; 0,005 | 0,010; 0,005 | 0,010; 0,005 |
  | Khởi tạo: `init_std_theta` [rad], `init_std_omega` [rad/bước] | — | 1,50; 0,10 | 1,50; 0,10 | 1,50; 0,10 |
  | `heading_std_after_init` [rad], `min_speed_for_heading` [px/bước], `omega_eps` | — | 0,30; 0,5; 1e-4 | 0,30; 0,5; 1e-4 | 0,30; 0,5; 1e-4 |
  | UKF α, β, κ | — | — | 1,0; 2,0; 0,0 | — |
  | PF | — | — | — | N 100 (phụ lục 500); ESS 0,5, systematic; `estimate mean`; roughening 0 (k2: 1,0); `degenerate_maha2` 1e4; K1 = `pf_optimal_proposal: true` |
  | Nguồn | `bytetrack.yaml` + ultralytics 8.4.141 | `config.py` (không khóa `ctrv_*` trong YAML) | như EKF + `config.UKF_*` | như EKF + khóa `pf_*` trong YAML |

- Seed: PF là 0, 1, 2. **Seed hiệu lực là seed của runner** (`run_meta.seed`): `run_tracking.run_video` gọi
  `tracker.set_seed(seed, video)` sau khi tạo bộ lọc, nên `pf_seed: 0` trong YAML bị ghi đè; RNG của mỗi track là
  `(seed, hash tên video, track id)`. Cột `bl_pf_seed_hieu_luc` ghi giá trị này, `bl_pf_seed_yaml` ghi giá trị YAML.
  CV, EKF và UKF tất định nên không có seed.
- Run chạy mới (bt-cv, bt-ukf, sort-cv30, sort-ekf30): YAML hiện tại trùng sha256 YAML lúc chạy của run cũ cùng tên
  (cột `yaml_sha12_*`), nên thông số hiệu lực ở các hàng tương ứng của `kiem_ke_run.csv` cũng là thông số sẽ dùng ở B
  (với `config.py` và backend như mục 2). Bước B ghi lại `tracker_cfg` của run mới để đối chiếu.
- Phụ lục (chỉ tái dùng, không thêm chạy mới): pf100-k2, pf500, pf500-k1.

## 6. Q2 — SORT so với ByteTrack, cùng bộ lọc và cùng detection

Cùng: detection (cache, điều kiện A; SORT lọc `conf > det_thresh = 0,25`, trùng bộ lọc `conf_min`). Cùng lớp và tham
số bộ lọc: `sort-cv30` bọc đúng `KalmanFilterXYAH` của ByteTrack-CV (`sort_tracker.py:101-118`); `sort-ekf30` bọc
`EKFTrackerCTRV` với cùng `ctrv_*` mặc định và cùng cách khởi tạo chuyển động 2 frame (`sort_tracker.py:121-160`).
Bộ đệm mất dấu 30 frame ở cả hai: `sort_max_age 30`, và `track_buffer 30` mà ultralytics 8.4.141 dùng thẳng làm
`max_frames_lost` (`byte_tracker.py:262`, không nhân theo fps).

Thuật toán thực sự khác nhau (đây là thứ được so):

| Khía cạnh | SORT trong repo (`sort_tracker.py`) | ByteTrack (ultralytics 8.4.141) |
|---|---|---|
| Ghép | 1 vòng Hungarian trên IoU, nhận cặp IoU ≥ 0,3, mọi track | Vòng 1: detection điểm cao với tracked + lost, chi phí 1 − IoU·score, nhận khi ≤ 0,8. Vòng 2 (điểm thấp): **rỗng ở điều kiện A**. Vòng 3: unconfirmed với detection còn lại, ngưỡng 0,7 |
| Sinh / xác nhận track | Mỗi detection chưa ghép sinh track; chỉ xuất khi `hit_streak ≥ 3` (hoặc frame ≤ 3) | Sinh khi score ≥ 0,25; kích hoạt ngay chỉ ở frame 1, sau đó cần ghép thêm 1 frame; unconfirmed bị xóa sau 1 frame hụt |
| Track tạm chưa xác nhận | Sống tới 30 frame | Xóa sau 1 frame hụt |
| Predict khi mất dấu | Không chỉnh vận tốc | Vận tốc chiều cao = 0 khi không Tracked |
| Xuất | Track vừa được cập nhật ở frame hiện tại | Track đã kích hoạt, đang Tracked |

Cách gọi đúng: "association và vòng đời kiểu SORT, dùng bộ lọc của ByteTrack, `max_age 30`". **Không** gọi là SORT
nguyên bản: SORT gốc dùng KF 7-D và `max_age 1`, tham chiếu là `sort-orig1`. Cũng không gọi là "chỉ đổi lifecycle",
vì cả quy tắc ghép lẫn predict khi mất dấu đều khác. Vì vòng 2 rỗng, Q2 không đo được lợi ích của việc ghép detection
điểm thấp (xem ideas; điều kiện c10 là việc mới, cần người dùng duyệt).

## 7. Metric, đơn vị, mẫu số, thống kê

- Metric báo cáo: HOTA, AssA, DetA, IDF1, IDSW. MOTA và FP chỉ để tham chiếu. Gộp COMBINED theo TrackEval trên 12
  video, từ trường thô (`raw_fields`).
- PF: tính COMBINED **riêng từng seed** rồi lấy trung bình 3 seed (cách của `src/kc48/boot_seed.py`). Không dùng
  min–max seed làm CI.
- Bootstrap ghép cặp theo **video** (n = 12): 5.000 mẫu, RNG seed 0. Trong mỗi mẫu, mọi đối chứng và mọi seed PF dùng
  **cùng chỉ số video**; PF tính COMBINED theo thứ tự seed 0, 1, 2 rồi lấy trung bình. CI 95 % percentile cho hiệu số.
  Đơn vị độc lập là video; frame, box và track không được coi là độc lập. "CI chứa 0" nghĩa là không phân biệt được,
  không có nghĩa là "bằng nhau".
- Phân tầng (bước C, chỉ khi đủ mẫu: ≥ 5 video đóng góp và ≥ 20 đơn vị; nếu không đạt thì chỉ báo số đếm). Dùng lại
  định nghĩa đã có, không định nghĩa mới:
  - Che khuất: `occlusion_eval` (context 1,2 s, `gaps=false`), như expA.
  - Cua: theo R48B, dev12 có 32 track cua (8 theo p90); `handoff/logs/review48/03_audit_du_lieu.md`.
  - Car nhỏ cạnh xe lớn: định nghĩa pipeline R48D (42 sự kiện / 7 video; `07_tong_hop_bao_cao_17_10.md`). Con số 88 sự
    kiện của R48B dùng định nghĩa audit khác, nên không trộn hai con số này.
  - Phải ghi rõ đơn vị (track / sự kiện / video) và mẫu số. Không có tầng motorcycle hoặc bicycle, vì GT UA-DETRAC
    không gán nhãn xe hai bánh.

## 8. Thực thi ở bước B (khi đã được approve)

- Run mới để so sánh (4, tất định): bt-cv, bt-ukf, sort-cv30, sort-ekf30.
- Run neo kiểm tái dùng (2): bt-ekf và bt-pf100-k1 seed 0, chạy tại HEAD qua runner mới. Yêu cầu sha256 (LF) trùng
  12/12 với run kc48 tương ứng. Lệch thì **dừng nhánh** bị ảnh hưởng, ghi lại và để người dùng quyết định; không tự
  chạy lại toàn bộ họ đó, không sửa thuật toán.
- Kiểm chấm: chấm lại R0 cho các run tái dùng phải tái tạo `combined_metrics.csv` cũ, sai số ≤ 1e-12. Sau đó mới chấm
  R1 cho mọi run.
- Đầu ra: `results/phase2/r51_controlled/{runs,eval,tests}/`. Không ghi vào `data/processed/trackers`,
  `results/trackeval` hay `results/phase2/kc48_pf_dev`.
- Chỉ chạy CPU, từ cache, ở foreground, có checkpoint theo từng run. Dự toán theo thời gian ms/frame của smoke R48C
  (EKF 1,62; pf100 3,48), áp cho 16.919 frame mỗi run: khoảng 0,5–1 phút mỗi run, 6 run dưới 10 phút. Chấm R0 + R1
  khoảng 30 lần. Đây là kịch bản dự toán, không phải cận trên.
- Không làm: tập test, BDD, c10, tuning hay quét tham số, H2/H3, phân tích nguyên nhân DetA, chạy lại test11, GPU.

## 9. Lệnh đã chạy ở bước A

| Lệnh | Exit | Ghi chú |
|---|---|---|
| `python -X utf8 src/r51_controlled/audit_protocol.py` | 0 | chạy 4 lần trong lúc phát triển; lần cuối mất 6,7 s; chỉ ghi `protocol/` |
| `python -X utf8 src/r51_controlled/test_protocol.py` | 1, rồi 0 | lần 1: 10/11, bác giả định "AssA không đổi" (mục 3.1); sau khi sửa: 13/13 PASS |

Không chạy tracking, không chấm metric trên dữ liệu thật, không sửa mã hay cấu hình sẵn có.

Vòng sửa R51A-F1/F2 (18/10), Python `khoaluan-mot`, cwd repo:

| Lệnh | Exit | Ghi chú |
|---|---|---|
| `python -B -X utf8 src/r51_controlled/audit_protocol.py` | 1, rồi 0 | lần 1 lỗi `SortKF7` không có `_std_weight_*` (run tham chiếu sort-orig1) → ghi None; lần cuối ~10 s, chỉ ghi `protocol/` |
| `python -B -X utf8 src/r51_controlled/test_protocol.py` | 0 | 19/19 PASS |

## 11. Vòng sửa R51A-F1/F2 (18/10)

Nguồn: review thủ công R51A (`automation/runs_local/bytetrack_r51_controlled_20261013_1806/review_thu_cong_R51A/answer.json`,
revise). Chỉ sửa trong `write_paths` của R51A; không sửa mã hay cấu hình nguồn, không thay tham số, chưa chạy B.

- **F1 — cổng provenance:** biểu thức cũ ở `audit_protocol.py` chỉ so video có trong `eval_meta` (thiếu hash một
  video vẫn True) và phép kiểm GT chung loại run thiếu dấu vân tay. Nay: `hash_dau_ra_khop` đòi đủ đúng tập video;
  `kiem_provenance` kiểm đủ tiêu chí 1–6 mục 4.1 cho từng run (cache tính lại trên đĩa, GT, backend, module lúc chạy /
  lúc chấm / HEAD, `config.py`, YAML, status, danh sách video); `gt_hien_tai_trung_eval_meta` tính trên mọi run đạt cổng
  và liệt kê riêng run thiếu GT. Test: fixture dương tính, tái hiện đúng ca reviewer (biểu thức cũ True, mới False), 24
  fixture âm tính từng trường; test trên dữ liệu thật: đúng 16 run kc48 đạt, mọi run khác không đạt, ma trận 4.3 khớp.
- **F2 — bảng thông số hiệu lực:** thêm cột `bl_*` (dt nội bộ, nhiễu, khởi tạo, UKF α/β/κ, PF N/ESS/estimate/roughening/
  degenerate/K1, seed YAML và seed hiệu lực, SORT iou/max_age/min_hits/det_thresh, khóa YAML bộ lọc), `backend_luc_chay`,
  `config_py_sha256_luc_chay`, `config_py_khop_HEAD_va_cay`, `dt_danh_nghia_s`. Mục 2 làm rõ 0,04 s danh nghĩa và dt
  nội bộ 1 bước; mục 5 ghi giá trị mặc định đã khóa kèm nguồn.
- Kết quả số của mục 3 và các quyết định tái dùng ở mục 4.3 **không đổi**.

## 10. Ghi chú thêm

- Reviewer R50A (`review_002.json`) nêu một issue không chặn: `09_thiet_ke_phan_tich_deta_pf_ekf.md:64` ghi "9 YAML",
  thực tế là 10. File này nằm trong vùng protected (`handoff/logs/review4*/**`), nên R51A **không sửa**; chỉ ghi nhận ở
  đây và ở mục 0.
