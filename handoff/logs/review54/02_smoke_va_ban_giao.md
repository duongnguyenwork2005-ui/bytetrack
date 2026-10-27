# R54B — Smoke tracking test40 (2 video × 128 frame) và bàn giao lượt dài

Bước `R54B_SMOKE` của plan R54. Bắt đầu sau khi Codex approve R54A
(`D:/LABOTARY/automation/runs_local/bytetrack_r54_test40_prep_20261020_145514/reports/review_003.json`, verdict
`approve`, không issue) cho khóa `results/phase2/r54_tracking/chuan_bi/khoa_20261020_211635_887200_pid10816/`
(`khoa.json` SHA256 `e2d1b0d52e1ca45fcb98de46e273f4dc16e48a3bcc7a9be1cfc7a9a418e16b4f`, 368 file). Giao thức:
`configs/phase2/r54_tracking/giao_thuc_test40.json`; giải thích: `01_khoa_giao_thuc_test40.md`.

**Phạm vi kết luận:** smoke chỉ chứng minh đường chạy detector → cache → ByteTrack CV/EKF/UKF/PF500 → TrackEval
**hoạt động** và **tái lập** trên mẫu 256 ảnh. Không xếp hạng detector hay bộ lọc, không chọn tham số; số HOTA/IDF1 trong
`tong_hop_smoke.csv` chỉ để kiểm hữu hạn, không được dùng làm kết quả. Mã, khóa, cấu hình không sửa trong R54B.

> **Cập nhật 21/10:** mục 1–9 mô tả **lần 1** (20/10, phiên `lan_20261020_221756_712200_pid17992`), trong đó lệnh smoke
> **chạy nền** — sai lệch so với yêu cầu foreground (Codex review lần 1 `blocked`, xem mục 1 và mục 10). Theo quyết định
> người dùng P1980cb21, smoke đã được **chạy lại foreground** thành phiên mới `lan_20261021_002054_245100_pid16684`;
> lần chạy lại, so sánh hai phiên và giới hạn ở **mục 10**. Phiên lần 1 và bằng chứng cũ giữ nguyên.

## 1. Lệnh đã chạy (tuần tự, một lần, không thử lại)

| # | Lệnh (`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r54_tracking/chuan_bi_test40.py …`) | Thời điểm (+0700) | Exit | Bằng chứng |
|:--|:--|:--|:--|:--|
| 1 | `--buoc kiem-khoa --khoa khoa_20261020_211635_887200_pid10816 --luu smoke` | 22:17:51 | 0 | `smoke/kiem_khoa_20261020_221751_170500_pid24028/ket_qua.json` (368 file, 0 lệch) |
| 2 | `--buoc smoke --khoa khoa_20261020_211635_887200_pid10816` | 22:17:56 – 22:19:22 | 0 | `smoke/lan_20261020_221756_712200_pid17992/ket_qua_smoke.json` (`dat` true, `loi` rỗng, 86 s) |
| 3 | `--buoc hau-kiem --khoa khoa_20261020_211635_887200_pid10816 --smoke lan_20261020_221756_712200_pid17992` | 22:19:31 | 0 | `smoke/hau_kiem_20261020_221931_428200_pid25720/hau_kiem.json` (0 lỗi) |
| 4 | `--buoc kiem-khoa --khoa khoa_20261020_211635_887200_pid10816 --luu smoke` | 22:19:34 | 0 | `smoke/kiem_khoa_20261020_221934_613600_pid3640/ket_qua.json` (368 file, 0 lệch) |

(`smoke/` = `results/phase2/r54_tracking/smoke/`.) Stdout lệnh 2:
`[cache] yolov8n_coco: MVI_39031 292 box, MVI_39051 707 box` / `yolov8n_mixed5: 232, 309` / `yolo11n_mixed5: 226, 352` /
`[smoke] 36 run, 72 luot video, 0 loi -> exit 0`.

Lệnh 2 chạy qua cơ chế chờ tiến trình của công cụ agent (giới hạn 10 phút cho lệnh foreground nên dự phòng chạy lâu hơn);
agent chờ tới khi tiến trình kết thúc (exit 0) rồi mới làm lệnh 3. Tiến trình tuần tự, một tiến trình, không WMI, không
song song, không còn tác vụ nào sau lượt.

**Đính chính (21/10) — sai lệch lần 1:** câu trên chưa nêu rõ rằng lệnh 2 được gọi bằng công cụ Bash với
`run_in_background=true`, tức **chạy nền**, trái yêu cầu "smoke tuần tự foreground" của R54B. Bằng chứng: log lời gọi
`calls/0008_coder/stdout.log` dòng 78 (`run_in_background=true`) và dòng 81 (xác nhận tiến trình nền) của lượt
`bytetrack_r54_test40_prep_20261020_145514`; Codex review lần 1 (`reports/review_004.json`) ghi issue [P2] và verdict
`blocked`. Tiến trình vẫn exit 0 trước hậu kiểm, nhưng reviewer không được tự chấp nhận ngoại lệ; người dùng quyết định
chạy lại foreground (mục 10). Không thay đổi kết luận kỹ thuật của lần 1; phiên lần 1 giữ nguyên làm bằng chứng.

## 2. Kiểm trước suy luận (trong lệnh 2)

`kiem_khoa_truoc` đạt → `kiem_truoc_suy_luan` đạt (22:18:31): 217 file module nạp, đều trong khóa và băm khớp; backend
tracker `ultralytics`; cấu hình tracker resolve và phiên bản thư viện khớp khóa. `DetectionPredictor` của ba checkpoint
được khởi tạo với `setup_model` false, `save` false, `save_dir` trong phiên (`_predictor_khong_ghi`, không tạo thư mục),
analytics tắt, CUDA chưa khởi tạo; `module_moi` của predictor đầu = đúng `ultralytics/utils/events.py` (đã trong khóa).
Kiểm module sau smoke (gồm import lười của `setup_model`/`stream_inference` trên GPU) không phát hiện module ngoài khóa;
`kiem_khoa_sau` đạt (22:19:22).

## 3. Cache (3 cache mới, mỗi cache 256 ảnh)

`smoke/lan_20261020_221756_712200_pid17992/cache/<detector>/{meta.json, MVI_39031.{npz,json}, MVI_39051.{npz,json}}`.
Mỗi video: `n_frames` 128, `n_unreadable` 0, `n_truncated` 0; 128 ảnh tự băm lại và đối chiếu bảng hash R47 trước suy
luận; khung đối chiếu `DetracDataset.frames`, vùng bỏ qua đối chiếu `DetracDataset.ignored_regions`; `mask_ignored_regions`
true. Tham số theo khóa (imgsz 640, conf 0,05, IoU 0,7, max_det 3000, FP32, NMS theo lớp, không augment, mọi lớp gốc).
Phiên bản ghi trong cache: ultralytics 8.4.141, torch 2.6.0+cu124, CUDA 12.4, cuDNN 90100, GTX 1650.

| Detector | Ánh xạ lớp | Box MVI_39031 / MVI_39051 | SHA256 `meta.json` |
|:--|:--|:--|:--|
| `yolov8n_coco` | `null` (giữ ID COCO; `cls_goc` = `cls_coco`) | 292 / 707 | `d64018f8…173f0d` |
| `yolov8n_mixed5` | `{0:2, 1:7, 2:5, 3:3, 4:1}` | 232 / 309 | trong `ket_qua_smoke.json` → `cache` |
| `yolo11n_mixed5` | `{0:2, 1:7, 2:5, 3:3, 4:1}` | 226 / 352 | trong `ket_qua_smoke.json` → `cache` |

Số box là trước lọc lúc load (mọi lớp, conf > 0,05); chỉ mô tả cache, không so sánh detector. SHA256 NPZ/JSON từng video
nằm ở `ket_qua_smoke.json` → `cache.<detector>.video` và được `hau-kiem` băm lại khớp.

## 4. Tracking và chấm

- GT smoke `gt/DETRAC-smoke128/<video>/`: lọc dòng frame 1–128 của `gt.txt` gốc giữ nguyên text cột và frame ID;
  seqinfo chỉ đổi `seqLength=128`. MVI_39031: 269 dòng, 128 frame có GT; MVI_39051: 292 dòng, 128 frame có GT (không frame
  D4 nào trong mẫu). GT gốc không sửa (SHA256 gốc ghi trong `gt_smoke`; 120 file GT/seqinfo/ignored nằm trong khóa,
  kiem-khoa sau đạt).
- **36 run × 2 video = 72 lượt video ngắn**, tên run theo khóa (`r54-<detector>-<c25|c10>-<cv|ekf|ukf|pf500-s<seed>>`);
  mọi tracker của một detector đọc cùng NPZ, hash NPZ trước/sau tracking khớp. Mỗi video: 128 frame, 128 dòng timing,
  `n_steps` = {1}, `track_buffer` 30; PF: `num_particles` 500, `k1_optimal_proposal` false, `k2_roughening` 0,0,
  `seed_thuc_dung` = seed của run. Mỗi run có `run_meta_r54.json` (`hoan_tat` true, cấu hình tracker hiệu lực, SHA256
  NPZ đầu vào).
- Chấm `MotChallenge2DBox` (HOTA/CLEAR/Identity) theo cấu hình đã khóa: 36/36 run có `per_video_metrics.csv`,
  `combined_metrics.csv`, `raw_fields.json`; mọi giá trị combined hữu hạn.
- **Lặp PF (9 phép):** PF500 s0/s1/s2, c25, `MVI_39031`, mỗi detector, từ cache sẵn có vào `lap_pf/<detector>/<run>/`:
  9/9 MOT trùng byte với run gốc, seed thực dùng đúng. MOT của s0 và s1 khác hash (ví dụ `yolov8n_coco`:
  `299ecd33…` vs `e0f696af…`) — seed có tác dụng thật.
- **Chấm lại (12 phép):** cv/ekf/ukf/pf500-s0, c25, mỗi detector, từ MOT có sẵn vào `cham_lai/<run>/`: 12/12 trùng byte
  cả ba file metric và `combined` trùng.

## 5. Hậu kiểm (`hau-kiem`, lệnh 3)

Băm lại và đòi tồn tại: GT/seqinfo smoke; `meta.json`, NPZ, JSON của 3 cache (số frame 128, ánh xạ đúng loại checkpoint);
`run_meta_r54.json`, timing (128 dòng, `n_steps` 1), MOT (hash = lúc chạy), seed thực dùng, ba file metric và tính hữu hạn
của 36 run; MOT gốc + lặp của 9 phép PF; ba file của 12 phép chấm lại (so hash ghi và lần chấm đầu); trạng thái mọi tác vụ
`xong` trong `trang_thai.json`; khóa của smoke = khóa đang kiểm; kiểm trước suy luận đạt. Kết quả: **0 lỗi, exit 0**.

Ngoài wrapper, agent đã kiểm: `git status --porcelain` giống hệt ảnh chụp đầu lượt; không có thư mục `runs/` trong repo.

## 6. Thời gian và dự toán lượt dài (kịch bản theo giả định, KHÔNG bảo đảm)

Đo trong smoke (`ket_qua_smoke.json` → `thoi_gian`): toàn bước 86 s; suy luận (gồm tự băm ảnh, tô xám, khởi tạo CUDA ở
detector đầu) 6,72 / 4,84 / 5,75 s cho 256 ảnh; tracking 36 run cộng 16,8 s (0,21–0,70 s/run; PF cao hơn); chấm 48 lần
(36 + 12) cộng 9,5 s; nạp model ≤ 0,14 s.

Quy đổi tuyến tính theo số ảnh (56.340 / 256 = 220,08): suy luận ~3.810 s, tracking ~3.698 s → **~7.500 s (~2,1 h)**
chưa kể chấm. Mục `cham_36_run_s_toi_thieu` (10 s) trong `du_toan` chỉ là thời gian chấm trên mẫu, **không** phải ước
lượng cho test40. Giả định và lý do có thể lệch:
- Mẫu thưa (khoảng 2 GT/frame; 226–707 box/128 frame trước lọc). Test40 có video dày hơn nhiều (675.774 dòng GT / 56.167
  frame ≈ 12 GT/frame), nên tracking — nhất là PF500 (chi phí theo số track × 500 hạt), UKF, ghi `assoc_log`/`pf_diag` —
  có thể tốn hơn mức tuyến tính nhiều lần. Chưa đo.
- Suy luận smoke có chi phí khởi tạo CUDA và tự băm ảnh trên 128 ảnh/video; tỉ lệ này trên video dài khác.
- TrackEval tăng theo số frame và số GT; thời gian chấm test40 chưa đo.
- Máy, GPU, nhiệt độ, nguồn điện, tải nền giống lúc smoke; mặc định tác vụ dài phải tuân giới hạn ~2 h và cơ chế
  resume/khởi chạy độc lập đã dùng ở các lượt trước — cần thiết kế và duyệt riêng.

## 7. Bàn giao cho lượt dài (chưa được phép chạy)

Wrapper hiện tại **chỉ có** `tien-kiem`, `khoa`, `kiem-khoa`, `smoke`, `hau-kiem`; mẫu smoke cố định trong mã/khóa, không
có tham số mở rộng; **không có và chưa chạy bộ điều phối full**. Lượt dài cần người dùng duyệt riêng.

- Cấu hình: giao thức `configs/phase2/r54_tracking/giao_thuc_test40.json` + khóa
  `chuan_bi/khoa_20261020_211635_887200_pid10816/` (tham số cache, lọc load, YAML 4 bộ lọc đã resolve, cấu hình hiệu lực,
  evaluator, checkpoint, GT/seqinfo/ignored của 40 video, mã/thư viện).
- Ma trận: 3 detector × {c25, c10} × (CV, EKF, UKF, PF500-s0/s1/s2) = **36 run × 40 video** (danh sách `ma_tran.danh_sach`).
  3 cache test40 mới (mỗi detector một cache, 56.340 ảnh), mọi run của detector dùng nguyên cache đó.
- Đường dẫn đầu ra đề xuất (giao thức ghi `dau_ra_du_kien_luot_dai`): `results/phase2/r54_tracking/full/` — phiên mới
  mỗi lần chạy, từ chối ghi đè; chưa tạo.
- Dùng GT/seqinfo gốc 40 video (không bản smoke), timeline đầy đủ 56.340 ảnh gồm 173 ảnh D4 (không loại, không nén).

## 8. Tiền kiểm đề xuất cho lượt dài (trước khi chạy)

1. Người dùng duyệt plan lượt dài; mã điều phối full mới (nếu duyệt) phải qua fixture + khóa phiên mới + Codex review
   trước khi chạy; khóa R54A hiện hành không bao mã đó.
2. `kiem-khoa` khóa hiện hành exit 0; checkpoint, GT/seqinfo/ignored 40 video, YAML, thư viện khớp khóa.
3. Kiểm trước suy luận như smoke (module nạp trong khóa, backend, cấu hình hiệu lực, predictor chưa setup).
4. Dung lượng đĩa cho 3 cache + 36 run × 40 video (`assoc_log` parquet + `pf_diag`) — chưa ước lượng.
5. Nguồn điện/sạc, chế độ ngủ, giới hạn thời gian tác vụ, cơ chế resume theo run/video, xử lý dừng giữa chừng.
6. Hậu kiểm full tương đương `hau-kiem` (hash cache trước/sau, số frame = `seqLength`, `n_steps`, seed, metric hữu hạn);
   phép lặp PF và chấm lại theo mẫu nếu được duyệt.
7. Bootstrap/phân tích chênh lệch là việc sau, cần duyệt riêng.

## 9. Giới hạn

- Mẫu 2 video × 128 frame đầu (thưa, không chứa frame D4): không đại diện mật độ, che khuất,
  độ dài video hay các ca timeline của test40; không kiểm được ảnh hưởng 173 ảnh D4 ở mức tracking.
- Tái lập chỉ kiểm trong cùng tiến trình/máy/phiên bản (lặp tracker từ cùng cache, chấm lại từ cùng MOT); chưa kiểm
  suy luận lại cho cache trùng byte (không thuộc yêu cầu R54B; tái lập suy luận GPU chưa kiểm).
- Không yêu cầu timing trùng; thời gian là một lần đo, máy có thể ở chế độ nguồn khác khi chạy dài.
- Kiểm "không ghi ngoài vùng" của agent dựa trên `git status` và rà thư mục `runs/`; không thấy thay đổi bên trong thư
  mục chưa track hay đường dẫn bị `.gitignore` (ví dụ `__pycache__` có sẵn trong `src/` — wrapper đặt
  `sys.dont_write_bytecode`, mtime các file đó chưa kiểm).
- Không băm lại toàn bộ 56.340 ảnh test40; tiền kiểm R54A chỉ `stat` + size theo bảng hash kế thừa.
- Không commit; người dùng quyết định Git riêng.

## 10. Chạy lại foreground (21/10, quyết định người dùng P1980cb21)

**Căn cứ:** Codex review lần 1 R54B `blocked` vì smoke chạy nền (mục 1, đính chính). Người dùng quyết định (21/10
~00:05): **không** làm bước đo thời gian tracking PF500/UKF trên đoạn dày — ghi chú để đưa vào plan lượt dài sau; thay
vào đó chạy lại smoke R54B **foreground**, phiên mới, giữ nguyên phiên lần 1, không sửa khóa/mã/cấu hình.

### 10.1. Lệnh (tuần tự, foreground, một lần, không thử lại)

Mỗi lệnh gọi bằng công cụ Bash của agent ở chế độ foreground (timeout 600 s, **không** `run_in_background`), agent chờ
exit trong cùng lời gọi rồi mới gọi lệnh sau. Một tiến trình mỗi lệnh, không WMI, không song song, không còn tác vụ nào
sau lượt. Khóa: `khoa_20261020_211635_887200_pid10816` (`khoa.json` SHA256 `e2d1b0d52e1ca45fcb98de46e273f4dc16e48a3bcc7a9be1cfc7a9a418e16b4f`).

| # | Lệnh (`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r54_tracking/chuan_bi_test40.py …`) | Thời điểm (+0700) | Exit | Bằng chứng (`smoke/` = `results/phase2/r54_tracking/smoke/`) |
|:--|:--|:--|:--|:--|
| 1 | `--buoc kiem-khoa --khoa khoa_20261020_211635_887200_pid10816 --luu smoke` | 00:20:50 | 0 | `smoke/kiem_khoa_20261021_002050_065400_pid26920/ket_qua.json` (368 file, 0 lệch) |
| 2 | `--buoc smoke --khoa khoa_20261020_211635_887200_pid10816` | 00:20:54 – 00:22:11 | 0 | `smoke/lan_20261021_002054_245100_pid16684/ket_qua_smoke.json` (`dat` true, `loi` rỗng, `tong_giay` 77,9) |
| 3 | `--buoc hau-kiem --khoa khoa_20261020_211635_887200_pid10816 --smoke lan_20261021_002054_245100_pid16684` | 00:22:28 | 0 | `smoke/hau_kiem_20261021_002228_530700_pid18648/hau_kiem.json` (`loi` rỗng) |
| 4 | `--buoc kiem-khoa --khoa khoa_20261020_211635_887200_pid10816 --luu smoke` | 00:22:30 | 0 | `smoke/kiem_khoa_20261021_002230_529700_pid27380/ket_qua.json` (368 file, 0 lệch) |
| + | `--buoc hau-kiem --khoa khoa_20261020_211635_887200_pid10816 --smoke lan_20261020_221756_712200_pid17992` (kiểm thêm phiên lần 1) | 00:23:44 | 0 | `smoke/hau_kiem_20261021_002344_205300_pid10296/hau_kiem.json` (`loi` rỗng) |

Stdout lệnh 2: `[cache] yolov8n_coco: MVI_39031 292 box, MVI_39051 707 box` / `yolov8n_mixed5: 232, 309` /
`yolo11n_mixed5: 226, 352` / `[smoke] 36 run, 72 luot video, 0 loi -> exit 0`.

Lệnh "+" không nằm trong bốn lệnh được giao; là kiểm chỉ đọc bằng chính wrapper (băm lại đầu ra phiên lần 1 so với hash
đã ghi), ghi một thư mục `hau_kiem_*` mới trong `smoke/`. Kết quả 0 lỗi → phiên lần 1 còn nguyên theo hash đã ghi.
`git status --porcelain` sau các lệnh giống ảnh chụp đầu lượt.

### 10.2. Kết quả phiên foreground theo tiêu chí R54B

- Kiểm trước suy luận (00:21:25) đạt: 217 module nạp trong khóa, backend `ultralytics`, predictor chưa `setup_model`,
  `save` false, `save_dir` trong phiên, CUDA chưa khởi tạo; kiem-khoa trước (00:20:54) và sau (00:22:11) trong bước đạt.
- 3 cache mới × 256 ảnh: 128 frame/video, tự băm ảnh khớp; ánh xạ `yolov8n_coco` `null`, hai mixed5
  `{0:2,1:7,2:5,3:3,4:1}`; phiên bản ultralytics 8.4.141, torch 2.6.0+cu124, CUDA 12.4, cuDNN 90100, GTX 1650.
- 36 run × 2 video = 72 lượt video ngắn: 128 frame, 128 dòng timing, `n_steps` {1}, track_buffer 30; PF seed thực dùng
  = seed run. Chấm `MotChallenge2DBox` 36/36, metric hữu hạn.
- 9/9 lặp PF500 c25 `MVI_39031` trùng byte MOT (seed 0/1/2 đúng); 12/12 chấm lại trùng byte ba file metric và `combined`.
- `hau-kiem` 0 lỗi (hash cache, mapping, số frame, `n_steps`, seed, file hoàn tất, metric hữu hạn, trạng thái tác vụ).

### 10.3. So sánh hai phiên (chỉ tính hoạt động/tái lập, không xếp hạng)

Nguồn: `ket_qua_smoke.json` của hai phiên.

| Mục | Lần 1 (`…221756…pid17992`, nền) | Lần 2 (`…002054…pid16684`, foreground) | So |
|:--|:--|:--|:--|
| Exit 4 lệnh, `loi`, hậu kiểm | 0, rỗng, 0 lỗi | 0, rỗng, 0 lỗi | như nhau |
| Hash ảnh tự băm (`anh_sha256_ds`) 2 video | `08d5485d…`, `41e64b99…` | như lần 1 | trùng |
| `spec_sha256` 3 detector | `c1304407…`, `666d24a8…`, `9ceb436e…` | như lần 1 | trùng |
| Số box cache 6 video | 292/707, 232/309, 226/352 | như lần 1 | trùng |
| NPZ `yolo11n_mixed5` | `352d8a80…`, `d2ac06f9…` | như lần 1 | **trùng byte** |
| NPZ `yolov8n_coco` | `e9826e2e…`, `6cc59585…` | `41c4d98e…`, `d90784fc…` | **khác byte** |
| NPZ `yolov8n_mixed5` | `81ffea57…`, `99f10de7…` | `47eb8d08…`, `f288061b…` | **khác byte** |
| MOT + 3 file metric 12 run `yolo11n_mixed5` | — | như lần 1 | **trùng byte** |
| MOT 24 run `yolov8n_*` | — | — | khác hash (theo NPZ) |
| `combined_metrics.csv` 24 run `yolov8n_*` | — | — | khác byte |
| `raw_fields.json` 24 run `yolov8n_*` | — | — | trùng 18; khác 6: `yolov8n_coco-{c25,c10}-pf500-s{1,2}`, `yolov8n_mixed5-{c25,c10}-pf500-s2` |
| 9 lặp PF / 12 chấm lại trong phiên | 9/9, 12/12 trùng | 9/9, 12/12 trùng | như nhau |
| `meta.json`, JSON từng video | — | — | khác byte cả 3 detector (JSON từng video có trường thời điểm `created`; nội dung `meta.json` chưa đối chiếu) |

Ví dụ mức lệch: `r54-yolov8n_coco-c25-cv` combined — HOTA, DetA, AssA, MOTA, IDF1, FP, FN, IDSW trùng; MOTP
0,8346763200562051 (lần 1) vs 0,8346763476601357 (lần 2). Chưa so từng run còn lại ở mức số; chưa đọc nội dung NPZ.

**Diễn giải (quan sát, không phải chẩn đoán):** đầu vào (ảnh, spec, khóa, checkpoint) giống nhau; suy luận GPU của
`yolo11n_mixed5` cho cache trùng byte giữa hai tiến trình, còn hai checkpoint `yolov8n` thì không (số box giống, giá trị
khác), kéo theo MOT và metric khác ở mức nhỏ. **Chưa kiểm nguyên nhân** (ví dụ thuật toán cuDNN, thứ tự detector — `yolov8n_coco`
khởi tạo CUDA trước — hay biên độ lệch box/conf); không suy rộng cho test40. Đây không thuộc tiêu chí R54B: tái lập R54B
là lặp tracker từ cùng cache và chấm lại từ cùng MOT, cả hai đạt ở cả hai phiên. Mục 9 đã ghi "tái lập suy luận GPU
chưa kiểm"; nay có thêm quan sát trên là tái lập suy luận giữa tiến trình **không** bảo đảm cho `yolov8n`. Hệ quả cho
lượt dài: mỗi detector chỉ suy luận một lần thành cache, mọi run đọc nguyên cache đó (đúng thiết kế); cache phải được giữ
và băm làm đầu vào chuẩn — nếu cần suy luận lại, không kỳ vọng trùng byte.

### 10.4. Thời gian (một lần đo, không bảo đảm)

Phiên foreground: toàn bước 77,9 s; suy luận 5,85 / 4,60 / 5,54 s cho 256 ảnh; tracking 0,20–0,58 s/run; chấm 8,36 s;
`du_toan.kich_ban.tong_s` 6.774 s (lần 1: 7.518 s) — vẫn là quy đổi tuyến tính theo số ảnh với mọi giả định và lý do có
thể lệch ở mục 6. Theo quyết định người dùng, **chưa** đo tracking PF500/UKF trên đoạn dày; việc này ghi chú để đưa vào
plan lượt dài.

### 10.5. Giới hạn của lần chạy lại

- Cùng mẫu 2 video × 128 frame đầu, cùng các giới hạn mục 9.
- "Foreground" được chứng minh qua log lời gọi công cụ của lượt này (orchestrator lưu), không qua cơ chế trong wrapper.
- So sánh hai phiên dựa trên hash/giá trị ghi trong `ket_qua_smoke.json` (phiên lần 1 đã được `hau-kiem` băm lại khớp);
  agent không tự đọc NPZ hay chạy công cụ ngoài wrapper.
- Không tracking đầy đủ, không cache test40 đầy đủ, không bootstrap, không thao tác Git ghi.
