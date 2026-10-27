# R54A — Khóa giao thức tracking test40 (chuẩn bị, chưa chạy full)

Bước `R54A_LOCK` của plan R54 (Codex đề xuất sau lượt `bytetrack_r53c_report_full_20261020_1346`, người dùng duyệt).
Phạm vi: chỉ nhánh so bộ lọc ByteTrack tương ứng expA (CV / EKF / UKF / PF500) trên 40 video test UA-DETRAC, cho ba
detector đã nghiệm thu ở R53. Mốc A **không suy luận, không tracking thật, không GPU**. Lượt dài cần người dùng duyệt riêng.

Máy đọc được: `configs/phase2/r54_tracking/giao_thuc_test40.json` (nguồn chuẩn của số và tên; tài liệu này giải thích).
Wrapper: `src/r54_tracking/chuan_bi_test40.py`; fixture: `src/r54_tracking/test_chuan_bi_test40.py`.

## 1. Căn cứ

- R53C approve: `D:/LABOTARY/automation/runs_local/bytetrack_r53c_report_full_20261020_1346/reports/review_001.json`
  (verdict `approve`, không issue). Giới hạn reviewer ghi: review tĩnh, không chạy lại scorer/test/`kiem-khoa`/GPU,
  không băm lại NPZ/checkpoint/toàn bộ ảnh; approve chỉ cho R53C, **không** phê duyệt tracking hay Git.
- Ba quyết định người dùng (20/10 ~14:10):
  1. Ba detector `yolov8n_coco`, `yolov8n_mixed5`, `yolo11n_mixed5` cùng một giao thức.
  2. PF500 seed 0/1/2; không PF100/PF300, không K1/K2.
  3. Vehicle gộp, tô xám `ignored_region`, c25/c10, evaluator/timeline hiện hành; không chuyển R0/R1 dev12 sang test40.
- Kế thừa: `configs/phase2/experiments/expA_detrac_test.yaml`, `configs/phase2/LOCK_moc2.md` (mục 3),
  `results/phase2/lock/lock_moc2.json`, `results/phase2/manifests/detrac_test.json`. Mã/cấu hình lịch sử giữ nguyên.

## 2. Ma trận dự kiến (chỉ khai báo)

3 detector × 2 điều kiện (c25, c10) × (CV, EKF, UKF, PF500-s0, PF500-s1, PF500-s2) = **36 run**, mỗi run **40 video**.
Tên run `r54-<detector>-<c25|c10>-<cv|ekf|ukf|pf500>[-s<seed>]`; danh sách đầy đủ ở `ma_tran.danh_sach` của JSON và
`khoa.json`. Không SORT/expB, không expC/`lost_recover`, không ablation, không BDD tracking, không occlusion_eval/bootstrap
trong R54 (bootstrap/phân tích là việc sau, cần duyệt).

| Detector | Checkpoint | SHA256 | Loại tên lớp | Sau suy luận |
|:--|:--|:--|:--|:--|
| `yolov8n_coco` | `models/yolov8n.pt` | `f59b3d83…3b36` | `coco80` | giữ ID COCO gốc |
| `yolov8n_mixed5` | `models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt` | `542b1381…0694` | `lop5` | ID → COCO theo tên |
| `yolo11n_mixed5` | `models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt` | `9a5e34a1…86fb` | `lop5` | ID → COCO theo tên |

SHA256 đầy đủ ở JSON; tiền kiểm đối chiếu với `results/phase2/r53_detector/chuan_bi/khoa_danh_gia.json` (khóa R53) và
dòng của nó trong `SHA256SUMS` R53. Không chọn lại checkpoint, không chọn tham số từ AP test hay metric smoke.

## 3. Cache detection (mỗi detector một cache mới)

- Hàm hiện có: `det_cache.build_video_yolo` → `det_cache.detect_frame` (`model.predict`), `det_cache.write_meta(mode="create")`,
  `det_cache.save_video`. Không dùng NPZ detector R53 hay cache legacy `yolov8n-coco_conf0.05_all`.
- Tham số: imgsz 640, conf 0,05, IoU NMS 0,7, max_det 3000, FP32 (`half` false), NMS theo lớp (`agnostic_nms` false),
  không augment, batch 1, device "0", **mọi lớp gốc** (`classes` null).
- Tô xám: `baseline_track.mask_ignored_regions` (qua `det_cache._mask_regions`), màu `config.IGNORE_MASK_COLOR_BGR` =
  (114, 114, 114), vùng từ `data/processed/DETRAC-test/<video>/gt/ignored_regions.txt`.
- Ánh xạ lớp (phần mới duy nhất chạm vào detection): sau suy luận, với checkpoint `lop5`, cột `cls` đổi theo
  `model.names` → `detector_classes.anh_xa_model_sang_lop5` → `sang_id_coco` (car 2, truck 7, bus 5, motorcycle 3,
  bicycle 1). Box, conf, thứ tự NMS giữ nguyên byte; ID gốc lưu thêm ở mảng `cls_goc` của NPZ. Thiếu/trùng tên lớp hoặc
  loại checkpoint khác loại đã khóa → lỗi, không đoán. Baseline `coco80` giữ nguyên ID.
- Frame không đọc được hoặc chạm trần max_det → dừng (không `allow_truncated`).

## 4. Lọc lúc load và tracker

- `det_cache.filter_dets` (qua `VideoDetections`): conf **> 0,25** (c25) hoặc **> 0,10** (c10) — so sánh ngặt; lớp
  [2, 3, 5, 7]; cắt tiền tố `[:300]`; không `class_map`. **Bicycle (COCO 1) không thuộc tập lớp tracking kế thừa** nên bị
  loại lúc load cho cả ba detector.
- Tracker: `run_tracking.load_tracker_cfg` + `run_tracking.track_video`, ghi đầu ra y hệt `run_video_to_dir` (fixture D2
  kiểm trùng byte). Backend `ultralytics`. Association ByteTrack gốc (0,25 / 0,1 / 0,25 / 0,8 / fuse_score), không
  `lost_recover`. `track_buffer` 1,2 s = 30 bước @ 25 fps (`time_units.track_buffer_frames`), `elapsed_mode: skip`,
  `assoc_log` và `pf_diag` bật như expA (chỉ ghi chẩn đoán, không đổi quyết định).
- YAML: `bytetrack.yaml` (resolve vào `ultralytics/cfg/trackers/bytetrack.yaml`), `configs/bytetrack_ekf_ctrv.yaml`,
  `configs/bytetrack_ukf_ctrv.yaml`, `configs/phase2/trackers/bytetrack_pf_ctrv_n500.yaml` (N = 500,
  `pf_optimal_proposal` không đặt = K1 tắt, `pf_roughening` 0,0 = K2 tắt). Đường dẫn đã resolve, SHA256 và cấu hình hiệu
  lực của từng bộ lọc được ghi ở `tien_kiem.json` → `cau_hinh_hieu_luc` và `khoa.json`.
- PF: seed 0/1/2 truyền qua `set_seed(seed, video)`; wrapper ghi seed thực dùng (`kalman_filter.seed`), N, K1, K2.

## 5. Đánh giá

TrackEval `MotChallenge2DBox`, metric HOTA / CLEAR / Identity, ngưỡng 0,5, `CLASSES_TO_EVAL = ["pedestrian"]` (class 1 =
vehicle). Cấu hình từ `run_trackeval.build_configs`; wrapper chỉ thay `GT_FOLDER`, `TRACKERS_FOLDER`, `OUTPUT_FOLDER`,
`SEQ_INFO`, `LOG_ON_ERROR`. Kết quả ghi như `run_experiment.stage_eval`: `per_video_metrics.csv`, `combined_metrics.csv`,
`raw_fields.json` (`boot_stats.save_raw_fields`).

GT: `data/processed/DETRAC-test/<video>/gt/gt.txt` hiện hành (vehicle gộp car/bus/van/others, class 1); seqinfo hiện hành
(`seqLength` = số ảnh trên đĩa). Timeline: mọi ảnh `img%05d.jpg` từ 1, `time_s = (i−1)/25`, như `DetracDataset.frames`.

**Khác giao thức detector D3/D4/D5 (R53):** không loại 173 ảnh D4 (chúng ở lại timeline, tracker vẫn chạy, evaluator chấm
như frame không có GT); không biến van/others thành vùng ignore; không áp bộ chấm AP/IoA của detector. 173 ảnh không
`<frame>` không được diễn giải là nhãn âm tính chắc chắn đầy đủ (điểm chưa rõ của mục 0 vẫn mở).

## 6. Wrapper và các bước

`chuan_bi_test40.py --buoc {tien-kiem, khoa, kiem-khoa, smoke, hau-kiem}`; **không có bước full**, không tự khởi chạy
lượt dài, không tham số CLI mở rộng mẫu (`MAU_SMOKE` cố định trong mã và JSON; `kiem_giao_thuc` từ chối giao thức thêm
nhánh, đổi seed, đổi mẫu).

- Mỗi lần chạy tạo **phiên mới** (`<buoc>_<thời điểm>_<µs>_pid<pid>` hoặc `--lan`); thư mục đã có → exit 1. Đường dẫn
  ghi được resolve rồi kiểm nằm trong vùng của bước (`chuan_bi/`, `smoke/`, `tests/`), kể cả `OUTPUT_FOLDER` và
  `LOG_ON_ERROR` của TrackEval.
- `sys.dont_write_bytecode` đặt trước import mã dự án; `YOLO_CONFIG_DIR`, `MPLCONFIGDIR` trỏ vào
  `<phiên>/_cau_hinh_thu_vien`, `YOLO_OFFLINE=1`, `HF_HUB_OFFLINE=1`.
- Mã lỗi: 0 đạt; 1 từ chối ghi đè; 2 đầu vào/khóa sai (trước mọi ghi kết quả; riêng `khoa` lệch sau khi tạo phiên chỉ
  ghi `khoa_that_bai.json`, `smoke` lệch trước suy luận chỉ ghi `ket_qua_smoke.json` có `dung_truoc_suy_luan`);
  3 tiền kiểm không đạt (có báo cáo); 4 smoke/hậu kiểm có lỗi; 5 smoke ở chế độ fixture dừng sau kiểm trước suy luận.
- `tien-kiem` (chỉ đọc dữ liệu): xem mục 7. Ghi thêm `dau_vao_sha256` (giao thức, wrapper, 6 file nguồn, 3 checkpoint,
  120 file GT/seqinfo/ignored; băm đầu và cuối lượt, khác nhau → lệch) và `ma_da_nap_sha256` (module đã nạp).
- `khoa`: chỉ khi tiền kiểm (mới nhất có `tien_kiem.json`, hoặc `--tien-kiem`) đạt, giao thức chưa đổi, có kết quả fixture
  `tat_ca_dat` trên **đúng** SHA256 wrapper + test hiện tại, **và** (F4) trạng thái hiện tại khớp bằng chứng tiền kiểm:
  `dau_vao_sha256` + SHA256 `bang_tien_kiem.csv`/`mau_smoke.json` kiểm **trước** khi tạo phiên (lệch → exit 2, không ghi);
  sau khi nạp phụ thuộc, `ma_da_nap_sha256` và cấu hình tracker hiệu lực phải khớp (lệch → exit 2, chỉ
  `khoa_that_bai.json`, không `khoa.json`). Khóa SHA256 đầy đủ: giao thức, tài liệu này, wrapper, test, kết quả fixture,
  ba file tiền kiểm, manifest/fingerprint/bảng hash/D4, khóa R53 + `SHA256SUMS`, 3 checkpoint, GT/seqinfo/ignored_regions
  của 40 video (120 file), 4 YAML đã resolve, `src/run_experiment.py` (đối chiếu), và (F2) **mọi file nguồn đang nạp**
  thuộc `src/`, gói `ultralytics` và gói `trackeval` sau khi nạp đường smoke + `MODULE_ULTRALYTICS` (YOLO/predictor/
  AutoBackend PyTorch/NMS/letterbox/byte_tracker/basetrack/matching/KalmanFilterXYAH, kể cả module ultralytics nạp lười
  trong `predict`) + `YOLO(ckpt)` ba checkpoint trên CPU (không suy luận) + (F2 vòng 2) **khởi tạo `DetectionPredictor`
  thật** của từng checkpoint đúng như `Model.predict` lúc chưa có predictor (`_smart_load("predictor")(overrides=…)`, kw lấy
  từ `tao_spec` như `det_cache.detect_frame`, `project` = thư mục phiên) — **không** `setup_model`/AutoBackend/chọn thiết
  bị/predict/warmup; `__init__` nạp lười `ultralytics/utils/events.py` (qua `add_integration_callbacks`) nên file này vào
  khóa; sau đó import lại `MODULE_ULTRALYTICS_SAU_PREDICTOR` (no-op). Khởi tạo lỗi nếu: lớp khác `DetectionPredictor`,
  `setup_model` đã chạy, `save_dir` ngoài phiên hoặc bị tạo, `save` bật, analytics `events.enabled` bật (không offline),
  CUDA đã khởi tạo. Cộng file `RECORD` dist-info của thư viện
  (torch, torchvision, numpy, pandas, opencv-python, scipy, lap, fastparquet, trackeval, ultralytics). `khoa.json` (phiên
  bản 3) còn ghi `backend_tracker` (backend thực nạp + file của từng lớp nền), `cau_hinh_hieu_luc`, `predictor_khoi_tao`
  (theo detector: lớp, `save_dir`, cờ kiểm, `module_moi` do chính predictor nạp), phiên bản thư viện. Ghi `khoa.json`,
  `SHA256SUMS`, `thoi_diem_khoa.json`.
- `kiem-khoa` (F3): băm lại toàn bộ danh sách của khóa mới nhất **có** `khoa.json` (hoặc `--khoa`); khớp → exit 0.
  **Mặc định chỉ đọc** (in kết quả + exit, không tạo phiên); `--luu chuan_bi` (R54A) hoặc `--luu smoke` (R54B) mới ghi
  `ket_qua.json` vào phiên mới của vùng đó. Nhờ vậy lệnh `--buoc kiem-khoa` của R54B không ghi ra ngoài `smoke/`.
- `smoke` (R54B): xem mục 8. `hau-kiem` (R54B): băm lại mọi file đã ghi hash trong phiên smoke + `kiem-khoa`.

## 7. Tiền kiểm (theo manifest, không quét thư mục dữ liệu)

Đối chiếu: SHA256 manifest / fingerprint / bảng hash / D4 với khóa R53; khóa R53 với `SHA256SUMS` R53; 3 checkpoint
(SHA256, đường dẫn, loại tên lớp qua `model.names` đọc trên CPU — không suy luận); thứ tự video; bảng hash 56.340 dòng
(tính lại `danh_sach_anh_co_thu_tu_sha256` và `noi_dung_anh_sha256_tong_hop` so fingerprint); mỗi video: ảnh liên tục từ 1,
kích thước 960×540, số ảnh = fingerprint = `seqLength`; GT 9 cột, frame trong 1..N, conf = class = 1; ảnh D4 không có dòng
GT; số `<frame>` XML = số ảnh − D4; số vùng bỏ qua = XML; tổng 56.340 ảnh / 173 D4 / 56.167 frame XML; cấu hình hiệu lực
4 bộ lọc (track_buffer 30, không `lost_recover`, PF N = 500, K1/K2 tắt). Ghi bảng 40 dòng `bang_tien_kiem.csv`,
`tien_kiem.json` (hash nguồn, cấu hình hiệu lực, giới hạn), `mau_smoke.json` (256 ảnh).

**Hash ảnh kế thừa và tự băm lại:** 56.340 ảnh chỉ được `stat` (tồn tại + size khớp bảng hash R47 — hash kế thừa, không
băm lại); 256 ảnh mẫu smoke được tự băm lại và so bảng. GT/seqinfo/ignored_regions không có hash tham chiếu kế thừa
(lock_moc2 không ghi), nên tiền kiểm ghi hash hiện tại và khóa R54 cố định từ đây. Mọi lệch → `khong_dat`, exit 3, chặn
`khoa` và do đó chặn smoke; không tự sửa dữ liệu.

Mẫu smoke chốt **trước** suy luận: `MVI_39031`, `MVI_39051`, frame 1–128 mỗi video, cùng 256 ảnh cho ba detector.

## 8. Smoke (R54B, chỉ sau khi R54A được approve)

`kiem-khoa` → phiên `smoke/lan_<id>/` → (F2) **kiểm trước suy luận**: `BYTETRACK_BACKEND` kế thừa phải trống hoặc
`ultralytics`, backend thực nạp và file lớp nền khớp khóa, (F2 vòng 2) khởi tạo `DetectionPredictor` thật của ba
checkpoint như ở `khoa` (`project` = phiên smoke, không tạo thư mục, không `setup_model`), mọi file nguồn đang nạp
(src/Ultralytics/TrackEval, kể cả `utils/events.py` do predictor nạp) nằm trong khóa và băm khớp, cấu hình tracker resolve và phiên bản thư viện khớp khóa; lệch → exit 2, không suy luận → GT smoke (`DETRAC-smoke128`: lọc dòng frame 1–128 giữ nguyên text cột và frame ID,
seqinfo chỉ đổi `seqLength=128`; GT gốc không sửa) → mỗi detector: kiểm SHA256 checkpoint, tự băm lại 128 ảnh/video, đối
chiếu khung với `DetracDataset.frames` và vùng bỏ qua với `DetracDataset.ignored_regions` / `baseline_track.load_ignored_regions`,
suy luận đúng 256 ảnh, đổi ID, lưu cache riêng → 36 tổ hợp × 2 video = 72 lượt video ngắn, mọi tracker của một detector
dùng nguyên cache đó → chấm từng run → lặp PF500 s0/s1/s2 c25 trên `MVI_39031` cho mỗi detector (9 phép, so byte MOT) →
chấm lại từ MOT cho cv/ekf/ukf/pf500-s0 c25 mỗi detector (12 phép, so byte 3 file metric) → kiểm hash cache trước/sau,
số frame, `n_steps`, seed thực dùng, metric hữu hạn → module nạp (src/Ultralytics/TrackEval) phải nằm trong khóa →
`kiem-khoa` lần hai. `hau-kiem` (F5) băm lại và đòi tồn tại: `meta.json`, NPZ, JSON từng video của cache; GT/seqinfo smoke;
MOT, `run_meta_r54.json`, ba file metric (`per_video`, `combined`, `raw_fields`) của 36 run; MOT gốc và MOT lặp của 9 phép
lặp PF; ba file của 12 phép chấm lại (so hash ghi và so lần chấm đầu); phiên dừng trước suy luận bị từ chối. Dự toán lượt
dài ghi dạng kịch bản theo giả định, không bảo đảm. Báo cáo smoke chỉ kết luận tính hoạt động và tái lập, không xếp hạng.

## 9. Kiểm tra đã làm ở R54A

Fixture không suy luận thật: detector là mô hình giả (tìm hình chữ nhật trắng) đi qua `det_cache.build_video_yolo` thật;
tracker, evaluator, ghi cache, lọc lúc load là mã thật. 23 ca (19 ca vòng đầu, mở rộng D1/H1/I2 và thêm I3–I5 cho F1–F5;
vòng sửa 2 thêm I6 và mở rộng I2 cho F2 còn mở):

| Ca | Nội dung |
|:--|:--|
| G1 | giao thức thật: 36 run, 3 detector; bản sửa thêm seed / thêm PF100 / đổi mẫu smoke bị `kiem_giao_thuc` từ chối |
| A1–A2 | ánh xạ coco80 giữ ID; lop5 → {0:2, 1:7, 2:5, 3:3, 4:1}, box/conf/thứ tự không đổi; thiếu tên, trùng tên, hoán vị tên (loại `khac`), sai loại → lỗi |
| B1 | `VideoDetections` + `mo_cache`: 0,25 bị loại ở c25, 0,1 bị loại ở c10 (ngặt), bicycle bị loại theo lớp, 305 box → đúng tiền tố 300; conf/max_det vượt cache → lỗi; `write_meta(create)` trên cache có sẵn → lỗi |
| C1 | `det_cache._mask_regions`: pixel trong vùng = 114, ngoài vùng không đổi (kể cả vùng tràn mép ảnh) |
| D1 | smoke fixture đầy đủ 3 detector giả × 36 tổ hợp × 2 video (72 lượt), TrackEval thật; detector giả nhận vùng bỏ qua đã xám ở mọi ảnh, hình trong vùng không bị phát hiện; MOT của detector coco giả và mixed5 giả (sau ánh xạ) trùng byte; frame không GT (3 frame có XML không box + 3 frame D4) vẫn qua tracker (20 bước, `n_steps` = 1) và FP được chấm; PF seed thực dùng 0/1/2, N 500, K1 tắt, K2 = 0; lặp PF 9/9 trùng byte; chấm lại 12/12 trùng byte; `hau_kiem` = 0 lỗi; (F5) 8 ca sửa riêng rẽ rồi khôi phục — sửa NPZ, sửa `meta.json`, sửa JSON video của cache, xóa `raw_fields.json`, sửa `per_video_metrics.csv`, sửa MOT lặp PF, xóa file chấm lại, sửa GT smoke — mỗi ca `hau_kiem` báo đúng file; khôi phục → 0 lỗi |
| D2 | `chay_tracker` ghi `data/`, `data_cls/` trùng byte và `timing` trùng 4 cột không phải thời gian với `run_tracking.run_video_to_dir` (CV, PF500 s0, có assoc_log/pf_diag) |
| E1–E7 | `tien_kiem` trên fixture: hợp lệ → đạt; sai hash bảng so khóa, sai hash ảnh mẫu, thiếu GT, mất 1 ảnh trong timeline, `seqLength` lệch, ảnh D4 có GT → `khong_dat` |
| F1 | đường dẫn thoát vùng (tuyệt đối, `..`), tên phiên xấu, `cham()` với `OUTPUT_FOLDER` ngoài vùng → lỗi trước khi ghi; phiên/file đã có → từ chối ghi đè |
| H1 | (F1) đường thật, không suy luận: khung mẫu **đủ 2 × 128 frame** khớp `DetracDataset.frames` (index, path, `time_s` so bằng `==`); hồi quy: cách cũ `(i-1)/fps` lệch 13/128 frame mỗi video (vd frame 36: 1.4 ≠ 1.4000000000000001), cách mới `(i-1)*(1/fps)` khớp 128/128; vùng bỏ qua khớp `DetracDataset.ignored_regions` và `baseline_track.load_ignored_regions` |
| I1 | CLI thật: `--buoc full`, `--lan ../thoat`, `--khoa ..\x`, env trỏ ra ngoài `tests/`, smoke / kiem-khoa / khoa / hau-kiem khi thiếu tiền đề → exit 2, không tạo thư mục nào |
| I2 | CLI thật trên dữ liệu thật, vùng ghi trong `tests/`: tiền kiểm exit 0; chạy lại cùng `--lan` → exit 1; khóa exit 0 (120 file GT/seqinfo/ignored, 3 checkpoint, 4 YAML, ≥ 10 module src, 158 file `ma_ultralytics` gồm byte_tracker/basetrack/matching/kalman_filter/predictor/detect.predict/nms/autobackend/backends.pytorch/tasks/augment/utils.events/utils.files, 36 file `ma_trackeval`, 10 file RECORD, `backend_tracker`, `cau_hinh_hieu_luc`); (F3) kiem-khoa mặc định exit 0 và **không đổi gì** trong cây thư mục; `--lan` không `--luu` → 2; `--luu chuan_bi` chỉ ghi `chuan_bi/`; `--luu smoke` chỉ ghi `smoke/`; `--luu khac`, `--luu` với bước khác → 2 |
| I3 | (F2) CLI thật, bản sao khóa trong `tests/`: khóa hợp lệ → smoke kiểm trước suy luận đạt rồi dừng exit 5 (chế độ fixture không bao giờ suy luận), không có thư mục cache; `BYTETRACK_BACKEND=np` → exit 2; SHA256 `byte_tracker.py` trong khóa sai → kiem-khoa 2, smoke 2 không tạo phiên; bỏ `kalman_filter.py` khỏi khóa → kiem-khoa 0 nhưng smoke 2 "module nạp ngoài khóa"; `cau_hinh_hieu_luc` (track_buffer EKF 31), phiên bản ultralytics, backend trong khóa sai → smoke 2; hau-kiem trên phiên dừng trước suy luận → 4 |
| I6 | (F2 vòng 2) khóa CLI thật trong `tests/` chứa `ultralytics/utils/events.py` (vai trò `ma_ultralytics`); `predictor_khoi_tao` đủ 3 detector, lớp `DetectionPredictor`, không `setup_model`, không tạo thư mục, `save` tắt, callback events có mặt, analytics tắt, CUDA chưa khởi tạo; `module_moi` của predictor đầu tiên = đúng `events.py` (chính `__init__` nạp); smoke hợp lệ của I3 (exit 5) có thông tin predictor, 0 lệch; bản sao khóa **thiếu** `events.py` → kiem-khoa 0 nhưng smoke exit 2 "module nạp ngoài khóa … events.py", không cache; bản sao khóa **sai hash** `events.py` → kiem-khoa 2, smoke 2 không tạo phiên; hàm thật trong tiến trình fixture: `project` ngoài `results/phase2/r54_tracking` → `LoiDauVao`, trong `tests/` → không tạo thư mục, `model.predictor` vẫn `None`, không module Ultralytics ngoài khóa |
| I4 | (F4) hàm: tiền kiểm fixture đạt, rồi sửa GT thật của fixture → `doi_chieu_tien_kiem` báo đúng 1 lệch `…/gt/gt.txt`, sửa thêm checkpoint → 2 lệch; CLI thật: bản sao bằng chứng tiền kiểm có hash GT khác / `bang_tien_kiem.csv` bị sửa → khoa exit 2, không tạo phiên; `ma_da_nap_sha256` của `src/run_tracking.py` khác → khoa exit 2, chỉ `khoa_that_bai.json`, không `khoa.json`, và phiên đó không được chọn làm khóa mặc định |
| I5 | `khoa.json` bị sửa (bản sao) → kiem-khoa 2, smoke 2 không tạo thư mục smoke |
| Z1 | snapshot thư mục nhạy cảm trước/sau: 0 khác ngoài thư mục lần test |

## 10. Kết quả R54A

### 10.0 Vòng sửa 2 sau review R54A (`review_002`, verdict revise, F2 chưa đóng) — bằng chứng hiện hành

`review_002` đóng F1, F3, F4, F5; F2 [P1] còn mở: khóa chưa chứa `ultralytics/utils/events.py`, file mà
`BasePredictor.__init__` (`predictor.py:159` → `callbacks/base.py:185`) nạp vô điều kiện khi `Model.predict`
(`model.py:530`) tạo predictor → smoke thật sẽ trả exit 4 ở kiểm module sau smoke; I3 dừng trước khởi tạo predictor
nên không bắt được.

| Sửa (chỉ `src/r54_tracking/`, tài liệu này) | Ca hồi quy |
|:--|:--|
| `khoi_tao_predictor`: khởi tạo `DetectionPredictor` thật cho 3 checkpoint (không `setup_model`/predict/warmup/GPU, `project` trong phiên, kiểm không tạo thư mục, analytics tắt, CUDA chưa khởi tạo); `phu_thuoc_thuc_dung` gọi nó ở cả `khoa` và kiểm trước suy luận của `smoke`; thêm `ultralytics.utils.files` vào `MODULE_ULTRALYTICS` và `ultralytics.utils.events` vào `MODULE_ULTRALYTICS_SAU_PREDICTOR` (import sau khởi tạo, no-op nếu predictor đã nạp); `khoa.json` phiên bản 3 ghi `predictor_khoi_tao` | I2 (thêm `events.py`, `files.py` vào tập bắt buộc), I6 |

Rà tĩnh import lười trên đường `Model.predict` → `setup_model` → `stream_inference` (predictor, autobackend,
backends/pytorch, torch_utils, nms, ops, results, loaders, checks, cfg): ngoài `utils.events` và `utils.files` (đã có
trong khóa cũ) chỉ còn `utils.autodevice` (chỉ khi `device` chứa `-1`, giao thức dùng thiết bị cố định),
`utils.benchmarks`, `nn.distill_model`, exporter/tuner/trainer (không thuộc đường predict). Phần này là rà tĩnh, không
chạy `setup_model`; module nạp lười chưa thấy vẫn bị kiểm sau smoke bắt (exit 4).

- Fixture: **23/23 đạt, exit 0** — `results/phase2/r54_tracking/tests/lan_20261020_211038_548300_pid20336/ket_qua_test.json`
  (276,5 s). Lần ngay trước `lan_20261020_210620_308100_pid10128`: 22/23, I6 lỗi do chính ca test (`gt["detector"]` là
  dict, viết `[0]` → `KeyError`); sửa ca test, wrapper không đổi giữa hai lần.
- Tiền kiểm mới: **đạt, 0 lệch, exit 0** — `results/phase2/r54_tracking/chuan_bi/tien_kiem_20261020_211554_265400_pid2664/`
  (10,3 s); tổng như 10.2 (56.340 ảnh, 173 D4, 56.167 frame có GT, 675.774 dòng GT, 163 vùng, 56.340 ảnh stat đúng size,
  256 ảnh mẫu tự băm lại), `backend_env` = null.
- Khóa mới và `kiem-khoa --luu chuan_bi`: đường dẫn và SHA256 ở mục 0 `handoff/PLAN_DA_CHOT.md`.
- Phiên của 10.1 (`tien_kiem_20261020_205236_960200_pid336`, `khoa_20261020_205404_464000_pid5196`, fixture
  `lan_20261020_205046_281500_pid22480`) giữ nguyên, **không dùng cho R54B**.

### 10.1 Vòng sửa sau review R54A (`review_001`, verdict revise, F1–F5) — lịch sử

| Lỗi | Sửa (chỉ trong `src/r54_tracking/`, cấu hình/tài liệu R54) | Ca hồi quy |
|:--|:--|:--|
| F1 [P1] `time_s=(i-1)/FPS` lệch 1 ulp so `DetracDataset.frames` ở 13/128 frame → smoke thật sẽ dừng trước suy luận | `khung_mau` tính `(i-1)*dt`, `dt = 1.0/DetracDataset.fps` (= `dt_nominal`), kiểm `DetracDataset.fps == 25` | H1 đủ 2 × 128 frame |
| F2 [P1] khóa bỏ sót mã Ultralytics thực thi detector/ByteTrack; backend kế thừa env; kiểm module chỉ sau smoke | khóa mọi file nguồn đang nạp của src/ultralytics/trackeval + RECORD thư viện + `backend_tracker` + `cau_hinh_hieu_luc`; `kiem_truoc_suy_luan` trước mọi suy luận; chế độ fixture không suy luận | I2, I3 |
| F3 [P2] `kiem-khoa` luôn ghi `chuan_bi/` → R54B ghi ngoài write_paths | mặc định chỉ đọc; `--luu chuan_bi|smoke` | I2 |
| F4 [P2] `khoa` không đối chiếu bằng chứng tiền kiểm | `dau_vao_sha256`/`ma_da_nap_sha256` trong tiền kiểm; `doi_chieu_tien_kiem` trước tạo phiên và sau nạp phụ thuộc; khóa thất bại không có `khoa.json` | I4 |
| F5 [P2] `hau_kiem` tin cờ đã lưu | băm lại meta/JSON/NPZ cache, GT smoke, 3 file metric × 36 run, MOT lặp PF, file chấm lại | D1 (8 ca sửa) |

- Fixture: **22/22 đạt, exit 0** — `results/phase2/r54_tracking/tests/lan_20261020_205046_281500_pid22480/ket_qua_test.json`
  (107 s; chạy trên giao thức cuối). Lần ngay trước `lan_20261020_204818_235000_pid18552`: cũng 22/22 nhưng trước khi thêm
  mục `khoa_va_kiem_tra` vào giao thức (chỉ tài liệu hóa, không đổi tham số).
- Tiền kiểm chính thức mới: **đạt, 0 lệch, exit 0** —
  `results/phase2/r54_tracking/chuan_bi/tien_kiem_20261020_205236_960200_pid336/` (10,8 s); tổng số như 10.2 (56.340 ảnh,
  173 D4, 56.167 frame có GT, 675.774 dòng GT, 163 vùng, 56.340 ảnh stat đúng size, 256/256 ảnh mẫu tự băm lại khớp),
  `backend_env` = null (không kế thừa `BYTETRACK_BACKEND`).
- Khóa mới và `kiem-khoa --luu chuan_bi`: đường dẫn và SHA256 ghi ở mục 0 của `handoff/PLAN_DA_CHOT.md` (tài liệu này nằm
  trong khóa nên không tự chứa hash khóa).
- Phiên cũ giữ nguyên, **không còn hiệu lực cho R54B**: `tien_kiem_20261020_151626_732400_pid22580`,
  `khoa_20261020_151705_972100_pid21744` (khóa vòng đầu, thiếu mã Ultralytics), các phiên `kiem_khoa_*` cũ. Chọn khóa mặc
  định = khóa mới nhất có `khoa.json`, tức khóa mới.

### 10.2 Vòng đầu (lịch sử)

- Fixture: **19/19 đạt, exit 0** — `results/phase2/r54_tracking/tests/lan_20261020_151541_604300_pid25164/ket_qua_test.json`
  (35–40 s). Lần trước `lan_20261020_151435_592300_pid19428`: 18/19, ca I2 lỗi vì wrapper đọc sai tên khóa của khóa R53
  (`hash_anh`, `loai_d4`) → tiền kiểm thật báo 2 lệch giả và exit 3; đã sửa tên khóa (và kiểm thêm `duong_dan`), chạy lại đạt.
  Cùng lần sửa: bảng in của TrackEval chuyển vào `trackeval_stdout.txt` của từng run (không đổi cấu hình).
- Tiền kiểm chính thức: **đạt, 0 lệch, exit 0** —
  `results/phase2/r54_tracking/chuan_bi/tien_kiem_20261020_151626_732400_pid22580/` (9,5 s).
  Tổng: 56.340 ảnh (stat đúng size 56.340), `seqLength` cộng 56.340, D4 173, frame có GT 56.167 (= số `<frame>` XML),
  675.774 dòng GT (= số box XML), 163 vùng bỏ qua; mọi video `so_frame_khong_gt_ngoai_D4` = 0; 256/256 ảnh mẫu tự băm lại
  khớp bảng. Checkpoint: 3/3 SHA256 khớp giao thức và khóa R53; `yolov8n_coco` = `coco80` (ánh xạ null), hai mixed5 =
  `lop5` (ánh xạ {0:2, 1:7, 2:5, 3:3, 4:1}). Backend tracker `ultralytics`; YAML resolve: CV →
  `…/site-packages/ultralytics/cfg/trackers/bytetrack.yaml` (`395701d9…3b8b`), EKF `2c35f2da…ab69`, UKF `5d1286e4…5661`,
  PF500 `ff43d493…880b`.
- Khóa và `kiem-khoa`: thư mục `chuan_bi/khoa_*` và `chuan_bi/kiem_khoa_*` (đường dẫn và SHA256 ghi ở mục 0 của
  `handoff/PLAN_DA_CHOT.md`; tài liệu này nằm trong khóa nên không tự chứa hash khóa).

## 11. Giới hạn

- Fixture không chạy `DetectorSpec.fill_env()`, `predict` thật hay GPU; đường đó chỉ chạy ở R54B. `YOLO(ckpt)` chỉ được
  nạp trên CPU (đọc `names`, nạp lớp nn) ở tiền kiểm/khóa/kiểm trước suy luận.
- F2: torch/torchvision/numpy/OpenCV/scipy… chỉ khóa bằng phiên bản + SHA256 `RECORD` dist-info (bắt cài lại/đổi bản,
  không bắt sửa tay một file nhị phân sau cài). `filterpy` không cài trong env (phiên bản null) nên không có `RECORD`. Module ultralytics
  nạp lười mà `MODULE_ULTRALYTICS` chưa liệt kê sẽ chỉ bị phát hiện ở kiểm sau smoke (smoke lỗi exit 4, phải quay lại
  chuẩn bị) — không âm thầm chạy. F2 vòng 2: predictor thật chỉ được **khởi tạo**; `setup_model` (AutoBackend, chọn thiết
  bị, fuse), `stream_inference`, warmup, NMS trên GPU không chạy ở R54A — import lười trong các bước đó chỉ được rà tĩnh.
  Predictor ở `khoa`/kiểm trước suy luận dùng `project` = thư mục phiên; predictor thật ở smoke dùng `project` mặc định
  của Ultralytics với `save=False` (không tạo thư mục — chưa kiểm bằng chạy thật). Ca "nguồn Ultralytics đổi" được mô phỏng bằng bản sao khóa có SHA256 sai, không sửa
  file thư viện thật.
- F4: ca CLI mô phỏng đầu vào đổi bằng bản sao bằng chứng tiền kiểm (không sửa dữ liệu thật); ca hàm sửa file fixture thật.
- Tiền kiểm không băm lại toàn bộ ảnh, không đọc lại XML, không kiểm tọa độ box GT; không kiểm pixel tô xám trên ảnh thật.
- Cổng nguồn so SHA256 tại thời điểm kiểm; không chứng minh file chưa từng bị sửa rồi khôi phục giữa hai lần kiểm.
- Snapshot "không ghi ngoài vùng" của fixture chỉ phủ các thư mục liệt kê trong `snapshot()`.
- Không commit; người dùng quyết định Git riêng.
