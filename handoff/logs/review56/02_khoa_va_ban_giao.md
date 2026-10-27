# R56B — Tiền kiểm, khóa R56 (liên kết khóa R55 cha) và bàn giao (chỉ chuẩn bị)

Ngày 2026-10-21. Nhánh `phase2-mixed-traffic`, HEAD `117d62d`, không commit. Mục 0 `handoff/PLAN_DA_CHOT.md` đã cập
nhật **trước** file này. R56 chỉ chuẩn bị; **không** có cache test40, suy luận, tracking, chấm, MOT, metric hay phép đo
đoạn dày trên dữ liệu thật. Chờ Codex review R56B; không chuyển sang thực thi test40.

## 1. Căn cứ

- **R55 complete:** lượt `bytetrack_r55_prepare_20261021_051232`, Codex `review_003.json` approve R55A, `review_004.json`
  approve R55B (`D:/LABOTARY/automation/runs_local/bytetrack_r55_prepare_20261021_051232/reports/`). Không review nào cấp
  quyền chạy thật.
- Plan R56 do Codex đề xuất (lời gọi `0017_planner`, 21/10 15:12), người dùng duyệt phương án A: thêm
  `tracking --video <v>` (toàn timeline video), fixture CPU, khóa mới liên kết khóa R55 cha; đoạn đo cố định MVI_40864
  frame 1–1515.
- **R56A được Codex approve:**
  `D:/LABOTARY/automation/runs_local/bytetrack_r56_video_prepare_20261021_151440/reports/review_001.json` (review tĩnh;
  không xác nhận R56B, không cấp quyền thực thi test40/đo đoạn dày/GPU). Thiết kế bổ sung:
  `handoff/logs/review56/01_bo_sung_chon_video.md`.
- Mã/test/hợp đồng/thiết kế **giữ nguyên bản R56A đã review** (không sửa trong R56B):
  `src/r55_tracking/dieu_phoi_test40.py` `257f4f028e5a39aec4ea61a8387efbab518631d05021942ce8a3d7d4a973961e`,
  `src/r55_tracking/test_dieu_phoi_test40.py` `87689fbcce7f4274d8dc152ac88ed1fbbc65fffebc9046d8d8f8ba31ef4e57a6`,
  `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json` `488238d4fc669ffac3ce98800b7031b08e8233504a84751afe3c29e4239cf95d`
  (ba giá trị do tiền kiểm và khóa R56 tự băm, trùng kết quả fixture R56A).

## 2. Lệnh đã chạy (foreground, tuần tự, chờ exit)

| # | Lệnh (`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/dieu_phoi_test40.py …`) | Exit | Bằng chứng |
|---|---|---|---|
| 1 | `--buoc tien-kiem` | 0 (`dat`, 0 lệch, 19,0 s) | `results/phase2/r55_tracking/chuan_bi/tien_kiem_20261021_155443_527300_pid26076/` |
| 2 | `--buoc khoa --tien-kiem tien_kiem_20261021_155443_527300_pid26076 --tests lan_20261021_152952_196200_pid9756` | 0 (`R56_LOCK`, 384 file) | `results/phase2/r55_tracking/chuan_bi/khoa_20261021_155510_440500_pid8768/` |
| 3 | `--buoc kiem-khoa --khoa khoa_20261021_155510_440500_pid8768 --luu chuan_bi` | 0 (384 + 368 file, khóa cha đúng hash, 0 lệch) | `results/phase2/r55_tracking/chuan_bi/kiem_khoa_20261021_155524_561000_pid12384/ket_qua.json` |

Tên phiên ở bước 2–3 lấy từ dòng in của bước trước; phiên test là phiên fixture R56A đã được approve
(`lan_20261021_152952_196200_pid9756`, 28/28, exit 0). Không chọn "phiên mới nhất". Tự kiểm sau khi viết mục 0 và file
này: xem mục 7.

## 3. Tiền kiểm (`tien_kiem.json` SHA256 `aec04ded1df31154b7aa74ef0f525ab311ef8763f81b51542f5d147031b143ba`)

Cùng phạm vi R55 (không thêm kiểm mới ngoài đối chiếu đoạn cố định đã có trong mã R56A):

- **Khóa R54:** băm lại 368 file, 0 lệch, `khoa.json` `e2d1b0d5…a418e16b4f`.
- **Checkpoint/names:** đọc trên CPU qua Ultralytics, không suy luận, không `setup_model`/warmup/forward, không GPU;
  SHA256 3 checkpoint (`yolov8n_coco`, `yolov8n_mixed5`, `yolo11n_mixed5`) nằm trong `dau_vao_sha256` và trùng khóa R54.
- **Dữ liệu:** manifest `detrac_test.json` `ff2b4b9c…3077f47c`, fingerprint R47 `dd0f2a0e…156343a`, bảng hash ảnh R47
  `3cea3488…8717b166`, bảng D4 `4d731253…85723551`. `bang_tien_kiem.csv` 40 video, SHA256
  `c026e173f80a213c8bb80a196b3d49fca6348fef1de32092a5ec22da87d2c793` — **trùng byte** bảng R55. Tổng **56.340 ảnh** (stat +
  size đúng cả 56.340), seqLength 56.340, **173 ảnh D4**, 56.167 frame có GT, 675.774 dòng GT, 163 vùng ignored.
- **Ảnh — phạm vi kiểm:** stat/size theo bảng hash R47 kế thừa + **băm lại 256 ảnh mẫu cố định** (MVI_39031, MVI_39051,
  frame 1–128); `mau_anh.json` `a902a88d1459e3254bb6fc689adb1fb531e6237a07f873d1bd4e3b7da76844dc` trùng byte bản R55.
  **Không băm lại toàn bộ 56.340 ảnh**; công việc cache (đường thực thi) băm từng ảnh ngay trước suy luận.
- **Cấu hình/thư viện/backend:** backend `ultralytics`, `track_buffer` 30 bước, CV/EKF/UKF/PF500 seed 0/1/2 trùng khóa
  R54; Python 3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124, trackeval 1.3.0, numpy 2.4.6, opencv-python 5.0.0.93, scipy
  1.17.1, lap 0.5.13 (đủ danh sách trong `tien_kiem.json`); `BYTETRACK_BACKEND` không đặt.
- **Đoạn đo cố định (đối chiếu bảng, không đo):** MVI_40864, frame 1–1515, `toan_video` true, 44.689 dòng GT căn cứ
  (≈ 29,50 dòng GT/ảnh) → 0 lệch so với bảng tiền kiểm. Trạng thái: **CHƯA ĐO**.
- **Dung lượng đĩa (kịch bản theo giả định, không bảo đảm):** như R55 — cần ~3,13 GB, trống ~44,5 GB.
- **Vùng ghi:** `results/phase2/r55_tracking/thuc_thi/` không tồn tại (không tạo).

## 4. Khóa R56

- Phiên `results/phase2/r55_tracking/chuan_bi/khoa_20261021_155510_440500_pid8768/`, `buoc` `R56_LOCK`, `phien_ban` 2,
  `khoa.json` SHA256 **`ce85628e34bc8507f9577c51a06d381a2d8e5d0f35cdf099c18330429d35247b`** (ghi trong `SHA256SUMS`,
  `thoi_diem_khoa.json`).
- **Liên kết:**
  - khóa R55 cha `results/phase2/r55_tracking/chuan_bi/khoa_20261021_121500_644100_pid22868/`, `khoa_json_sha256`
    `5a07efb8485466ef71366b94c6ae10051df22dacf606e8b6558eef204cf6e42b`, 376 file, `duoc_thay` đúng 3 file;
  - khóa R54 `results/phase2/r54_tracking/chuan_bi/khoa_20261020_211635_887200_pid10816/`, `khoa_json_sha256`
    `e2d1b0d52e1ca45fcb98de46e273f4dc16e48a3bcc7a9be1cfc7a9a418e16b4f`, 368 file (kiểm nghiêm ngặt, không ngoại lệ);
  - giao thức R54 `configs/phase2/r54_tracking/giao_thuc_test40.json`
    `eb1f8f3087401eaf3641c1bc215a864ad23850ec5ddd6970d7d74280715431ad`; hợp đồng `488238d4…4239cf95d`;
  - fixture R56A `results/phase2/r55_tracking/tests/lan_20261021_152952_196200_pid9756/ket_qua_test.json` (28 test),
    tiền kiểm bước 1; ma trận **36 run**; thứ tự cache → xác nhận cache → tracking → chấm → hậu kiểm.
- **Bảng khác biệt so khóa cha** (`khac_biet_khoa_cha` trong `khoa.json`; 384 = 373 giữ + 3 thay + 8 mới):

| Loại | File | SHA256 cha → mới |
|---|---|---|
| thay | `src/r55_tracking/dieu_phoi_test40.py` | `b9e7f199…3f23c047` → `257f4f02…a973961e` |
| thay | `src/r55_tracking/test_dieu_phoi_test40.py` | `9f8a1bcd…6859af05` → `87689fbc…ef4e57a6` |
| thay | `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json` | `626dd24d…7df12c76` → `488238d4…4239cf95d` |
| mới (ket_qua_test_r55) | `results/phase2/r55_tracking/tests/lan_20261021_152952_196200_pid9756/ket_qua_test.json` | `a8abf3c67e179eea6ef286f0baf5262ecb1d589debb4b7784a78db3f07412a73` |
| mới (tien_kiem_r55) | `…/tien_kiem_20261021_155443_527300_pid26076/tien_kiem.json` | `aec04ded1df31154b7aa74ef0f525ab311ef8763f81b51542f5d147031b143ba` |
| mới (tien_kiem_r55) | `…/tien_kiem_20261021_155443_527300_pid26076/bang_tien_kiem.csv` | `c026e173f80a213c8bb80a196b3d49fca6348fef1de32092a5ec22da87d2c793` |
| mới (tien_kiem_r55) | `…/tien_kiem_20261021_155443_527300_pid26076/mau_anh.json` | `a902a88d1459e3254bb6fc689adb1fb531e6237a07f873d1bd4e3b7da76844dc` |
| mới (bo_sung_r56) | `handoff/logs/review56/01_bo_sung_chon_video.md` | `861b682b9f697d79fc47b1694dceb26c207d541b72cce8d3c1f6c335b3e70778` |
| mới (khoa_cha_r55) | `…/khoa_20261021_121500_644100_pid22868/khoa.json` | `5a07efb8485466ef71366b94c6ae10051df22dacf606e8b6558eef204cf6e42b` |
| mới (khoa_cha_r55) | `…/khoa_20261021_121500_644100_pid22868/SHA256SUMS` | `451ef98fe9208638d4da5de8be3a832c0dd523d1186e230d2309e1e1e96c4449` |
| mới (khoa_cha_r55) | `…/khoa_20261021_121500_644100_pid22868/thoi_diem_khoa.json` | `c90697c93abcbd58e4fcfcd70b602a175fb205751b9929f7640297623eda3ae1` |

  373 file giữ = 368 file khóa R54 + thiết kế R55 `handoff/logs/review55/01_thiet_ke_dieu_phoi.md` (`9687aae7…2419a598`)
  + kết quả fixture R55 `tests/lan_20261021_115928_508900_pid21264/ket_qua_test.json` + 3 file tiền kiểm R55
  `chuan_bi/tien_kiem_20261021_121436_943200_pid22720/` — tất cả nằm trong khóa R56 (vai trò `khoa_cha_r55:*`) với hash
  bằng hash cha; khi tạo khóa R56 bộ điều phối băm lại 376 file cha, 0 lệch ngoài 3 file được phép. Bằng chứng R55 cũ vì
  vậy nguyên byte và được khóa R56 giữ; bằng chứng mới (fixture R56A, tiền kiểm R56B) là thành phần dùng cho khóa R56.
- **Ý nghĩa ngoại lệ 3 file:** chỉ mô tả sự chuyển tiếp lúc tạo khóa R56. `kiem-khoa` khóa R56 (và mọi công việc thực thi
  sau này) băm **mọi** file theo hash khóa R56, không ngoại lệ, cộng kiểm liên kết khóa cha + khóa R54. Khóa cha R55 là
  bằng chứng lịch sử bất biến; **không** tuyên bố khóa cha khớp mã hiện tại (nó khác đúng 3 file trên).
- `handoff/PLAN_DA_CHOT.md` và file này **không** nằm trong khóa R56 (cập nhật chúng không làm lệch khóa — kiểm ở mục 7).
- **Giới hạn import lười (giữ như R55):** khóa không khởi tạo predictor/`setup_model`; module Ultralytics nạp lười trong
  `setup_model`/`stream_inference` chỉ có trong khóa nhờ kế thừa khóa R54 (rà tĩnh + smoke R54B). Đường thực thi kiểm
  module nạp **trước** công việc (ngoài khóa/lệch hash → exit 2) và **sau từng đơn vị** trước marker (ngoài khóa → không
  marker, exit 4); chưa thử trên đường thật.
- Sửa bất kỳ file nào trong khóa R56 sau thời điểm này → phiên khóa mới (giữ phiên cũ); sửa mã/cấu hình → quay lại bước
  sửa mã + fixture mới + review.

## 5. Điều kiện cho lượt thực thi (chưa có — cần plan + người dùng duyệt riêng)

1. **Plan thực thi test40 do người dùng duyệt**, cấp `--duyet <mã plan>`; R56 không cấp quyền này. Đường thật chỉ nhận
   khóa `R56_LOCK` (`khoa_20261021_155510_440500_pid8768`, `ce85628e…9d35247b`) hoặc khóa mới hơn tạo đúng quy trình.
2. **Thứ tự bắt buộc:** (a) tạo/xác nhận cache **đủ 40 video** cho từng detector cần dùng (manifest công bố
   `HOAN_TAT.json`) theo khóa R56; (b) **đo thời gian PF500/UKF trên đoạn cố định MVI_40864 frame 1–1515 (toàn video)**
   bằng `tracking --run <run PF500/UKF> --video MVI_40864`, cùng cache đã xác nhận, foreground — **chưa đo**, thuộc
   P1980cb21; ghi thời gian từng run, detector/cấu hình/seed, phạm vi tính thời gian và giả định ngoại suy; nội dung đo
   phải được duyệt trước; không xếp hạng/chọn tham số/đổi giao thức từ phép đo; (c) chỉ sau đó mới tới full 36 run ×
   40 video, theo phê duyệt riêng. Đầu ra tracking một video hoàn tất được đường đầy đủ tái sử dụng khi cùng ràng buộc;
   chấm/hậu kiểm vẫn đòi đủ 40 video.
3. Căn cứ đoạn: mật độ GT (44.689/1.515 ≈ 29,50 dòng GT/ảnh, bảng tiền kiểm), **không** dựa trên metric tracking.
4. **Dự toán tạm thời** (hợp đồng `du_toan_tam_thoi`, không đổi): suy luận 3 detector ~3.519 s, tracking 36 run
   ~3.247 s, chấm ≥ 8 s, tổng ~6.774 s; đĩa ~3,13 GB — kịch bản theo giả định, **không thay** phép đo ở mục 2(b).
5. **QJTF** — ý tưởng đo chi phí kiểm module (trước/sau) mỗi tác vụ, ghi nhận cho plan thực thi; **chưa triển khai**,
   R56 không thêm instrumentation/benchmark/phép đo. Không làm lại P6fb1cad7 (tiêm module ngoài khóa) hay P9857b26c
   (marker nguyên tử).
6. Lượt dài tuân giới hạn tác vụ foreground/ngân sách thời gian của plan thực thi; bộ điều phối tiếp tục theo đơn vị
   (video/run), có thể chia nhiều lần gọi.

## 6. Nguyên tắc cache và tính tái lập (giữ như R55)

- **Cache bất biến duy nhất:** mỗi detector đúng một manifest cache công bố, chỉ khi đủ 40 video đạt kiểm; đó là đầu vào
  chuẩn duy nhất cho c25/c10 và mọi tracker, kể cả khi tiếp tục hay khi chạy `--video`. Lọc `--video` không giảm tập
  cache bắt buộc hay kiểm hash các video khác. Không dùng cache legacy, cache smoke R54 hay NPZ detector R53.
- **Sai hash → chặn**, không tự suy luận lại thay cache.
- **Không kỳ vọng suy luận detector lại trùng byte**; `lap-tracker` (cùng cache/seed → so byte MOT) và `cham-lai` (cùng
  MOT đã băm → so byte metric) không gọi detector.
- Quan sát smoke R54B (`yolov8n_*` khác giá trị cache giữa hai phiên, `yolo11n_mixed5` trùng byte) giữ nguyên, không
  suy rộng. 173 ảnh D4 vẫn mở: giữ trong timeline, chấm như frame không GT bằng evaluator hiện hành.

## 7. Đã làm / chưa làm / giới hạn

- Đã làm: 3 lệnh chuẩn bị trên đầu vào thật (exit 0); cập nhật mục 0; file này. Sau khi viết mục 0 và file này chạy lại
  `--buoc kiem-khoa --khoa khoa_20261021_155510_440500_pid8768` (không `--luu`, chỉ đọc) — kết quả ghi trong báo cáo lượt
  R56B gửi Codex (không ghi ngược vào file này để file không đổi sau kiểm).
- Không làm: cache test40, suy luận, tracking, chấm, đo đoạn dày, `setup_model`/warmup/forward, GPU, đọc NPZ, sửa
  mã/test/hợp đồng/thiết kế, sửa khóa R54/R55 hay bằng chứng cũ, Git ghi. Không tạo `results/phase2/r55_tracking/thuc_thi/`.
- Thay đổi mới của R56B chỉ nằm trong `results/phase2/r55_tracking/chuan_bi/` (3 phiên mới), mục 0 PLAN và file này.
- Giới hạn: tiền kiểm chỉ băm lại 256 ảnh mẫu (phần còn lại stat/size theo hash R47 kế thừa); GT không đối chiếu tọa độ
  box với XML (chỉ số đếm qua fingerprint R47); không kiểm pixel tô xám trên ảnh thật ở bước chuẩn bị; dự toán đĩa/thời
  gian là kịch bản; bao phủ import lười dựa vào khóa R54 + kiểm trước/sau đơn vị, chưa thử trên đường thật; chọn video
  mới được kiểm trên fixture tổng hợp (R56A), chưa chạy trên dữ liệu thật.
