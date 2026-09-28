# Phụ lục vòng 24 — đóng N23-F1 (kiểm số frame hiển thị), kiểm lại hai video Nano, chốt phiên bản OpenCV của bộ kiểm

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Review trước: `handoff/CODEX_REVIEW_1d5ffaa.md` — tracking 1.470 frame **PASS WITH LIMITATIONS**;
bộ kiểm render **REQUEST CHANGES** (N23-F1, P2). Base review **giữ `c4b39fe`** đến khi Codex nghiệm thu vòng này. Đầu vòng: HEAD
`1d5ffaa`, nhánh `phase2-mixed-traffic`, 0 commit sau đó, working tree chỉ có file riêng (`logs/review24/00_trang_thai_dau.log`).
Không chạy lại bốn lượt tracking, không build engine, không nâng cấp môi trường Nano; tracker, runner, luồng trực tiếp, YAML khóa,
kết quả UA-DETRAC/BDD, benchmark và bằng chứng Nano vòng 21–23 không đổi.

## 1. Tái hiện N23-F1 bằng script của Codex (không sửa script)

`python handoff/CODEX_REPRO_1d5ffaa_render_counter.py` từ gốc repo (script sha256 `d9c5f23c…`). Trên mã cũ `1d5ffaa`
(`01_repro_codex_truoc_khi_sua.log`, `repro_codex_truoc/`): video đúng exit 0; video ghi `f999` trên 27/30 frame có box **exit 0,
`dat` true**, phép "mọi frame: khung bbox/màu ID/số frame đúng chỗ" true, `ti_le_loi_max` 0,2662 như video đúng → script báo
`REGRESSION: FAIL`, exit 1. Nguyên nhân xác nhận trên mã: `ky_vong` và `nen` đều đã vẽ cùng số frame, nên trên frame có box
`mask = ky_vong != nen` chỉ còn bbox/nhãn; nhánh kiểm số frame chỉ chạy khi không có box. Hai video Nano có `n_frame_co_box = 300`,
tức số frame **chưa từng được kiểm** ở vòng 23.

## 2. Sửa (`9b1805c`, chốt phiên bản ở `4f1fa2f`)

- `render_mot_video.py`: tách `ve_so_frame()` dùng chung cho render và bộ kiểm; `ve_frame(..., ve_so=True)`. Đầu ra không đổi
  (render lại trên Nano **byte-giống** video vòng 23, mục 4).
- `kiem_video_render.py` — **(0) số frame, độc lập, trên mọi frame** (kể cả frame có box): dựng bản ảnh gốc + bbox/nhãn **không**
  số frame, rồi (a) trên các điểm chữ `f<i>` kỳ vọng, lỗi so với bản kỳ vọng ≤ 0,5 × lỗi so với bản không số và ≤ 40 (ngưỡng cũ,
  không nới); (b) nhận dạng: vẽ lại trên chính nền frame mọi ứng viên cách `f<i>` **một** phép sửa chữ số (thay/xóa/chèn) cộng bản
  không số — 31 ứng viên cho số một chữ số, 49–50 cho hai chữ số, 67–69 cho ba chữ số; trên hợp điểm các ứng viên, frame giải mã phải
  gần `f<i>` nhất. (1) bbox/màu ID trên frame có box và (2) tên lớp giữ nguyên. Phép cũ gộp tách thành hai phép riêng; JSON thêm
  `n_frame_kiem_so`, `ti_le_loi_so_max`, `loi_so`. Cú pháp Python 3.6.
- **Chốt phiên bản OpenCV** (`4f1fa2f`, lý do ở mục 5): nếu `<video>.json` của render ghi OpenCV khác OpenCV đang chạy → exit 2
  `KHONG KIEM DUOC`, không chạy phép nào; `--cho-phep-khac-opencv` chạy đủ phép để chẩn đoán nhưng exit vẫn 2, `dat` false. JSON ghi
  `opencv_render`. Cùng phiên bản thì hành vi không đổi.
- Regression `test_render_mot_video.py` thêm 8–13: `f999` trên 27 frame có box (kịch bản Codex), **mất** số trên frame có box, số
  **lệch một** (`f<i+1>`), số sai **chỉ** trên 3 frame không box — mỗi ca exit 1, phép số frame KHÔNG ĐẠT, phép bbox/ID vẫn ĐẠT; 12–13
  chốt phiên bản. Video giả được ghi bằng `cv2.putText` trực tiếp, không gọi mã đang kiểm.

## 3. Kết quả sau sửa (PC)

| Kiểm | Kết quả | Log |
|:--|:--|:--|
| Script Codex, mã `9b1805c` | video đúng exit 0 (`dat` true, 30/30 frame kiểm số, tỷ lệ max 0,246); video sai **exit 1** (`dat` false; bắt đúng 27 frame 1–10, 14–30; 3 frame 11–13 đúng không bị báo; bbox/ID ĐẠT) → `REGRESSION: PASS`, exit 0 | `02_*`, `repro_codex_sau/` |
| Script Codex, mã cuối `4f1fa2f` | như trên, `REGRESSION: PASS`, exit 0 | `05_*`, `repro_codex_ma_cuoi/` |
| `test_render_mot_video` | 11/11 (`9b1805c`), **13/13** (mã cuối) | `03_*`, `04_*` |
| Test mới chạy trên mã cũ `1d5ffaa` | 7/11: ca 8, 9, 10 exit 0 (chấp nhận video sai); ca 11 exit 1 nhưng không có phép số frame riêng | `regression_bat_ma_cu.log` |
| Full suite PC | **35/35** trên `8111892` (mã = `9b1805c`). Trên mã cuối `974affc`: lần 1 **34/35** — `test_pc_launcher` phép 9 (L-F1, `timeout_dung` = 3 s) mất 3,23 s nên launcher terminate; mã launcher không đổi trong vòng; chạy riêng 3 lần 17/17 (phép 9 mất 0,77–1,07 s); lần 2 **35/35** | `tests_20260928-1740.log`, `tests_20260928-1804.log` (lần FAIL, giữ lại), `08_test_pc_launcher_lap_lai.log`, `tests_20260928-1820.log` |

## 4. Kiểm lại hai video Nano (Python hệ thống 3.6.9, OpenCV 3.2.0 — cùng OpenCV đã render)

Gói **v10** (`9b1805c`, `a3bb5841…`) và **v11** (mã cuối, `6f80806d…`) ở `~/nano_run/v10`, `~/nano_run/v11`: manifest khớp, biên dịch
3.6 được; so với v9 chỉ khác `render_mot_video.py`, `kiem_video_render.py`, `test_render_mot_video.py` (mã tracking không đổi).
Driver `logs/review24/kiem_lai_v24.sh`; kết quả `kiem_lai_nano_v10/`, `kiem_lai_nano_v11/` (lệnh, sha256 mã, exit code từng bước).

**Hash** (`00_hash_va_phien_ban.txt`): video CV `5f55fcb4…` (20.406.950 B), EKF `c7d40019…` (20.409.877 B); MOT/lớp CV `5017f37b…`/
`a14e06bf…`, EKF `645bba3a…`/`a22bbf7d…` — **khớp** `review23/render_v23/nguon_sha256.txt` và sidecar; mtime vẫn là lúc render vòng 23.

Kết quả bộ kiểm **mã cuối (gói v11)**; gói v10 (`9b1805c`) cho cùng mọi số, chỉ thiếu trường `opencv_render`:

| Kiểm | CV lượt 1 | EKF lượt 2 |
|:--|:--|:--|
| Frame giải mã / `CAP_PROP_FRAME_COUNT` / số ảnh | 300 / 300 / 300 | 300 / 300 / 300 |
| FPS / kích thước | 25 / 960×540 | 25 / 960×540 |
| **Số frame hiển thị** (độc lập, mọi frame) | **300/300 đúng**, tỷ lệ lỗi max 0,164 | **300/300 đúng**, tỷ lệ lỗi max 0,164 |
| bbox/màu ID (frame có box) | 300 frame, 1.272 box, 0 lỗi, tỷ lệ max 0,318 | 300 frame, 1.270 box, 0 lỗi, tỷ lệ max 0,307 |
| Tên lớp, mẫu mỗi 5 frame (tham số vòng 23) | 255/255 box đúng (car 252, truck 3) | 255/255 box đúng (car 252, truck 3) |
| exit / thời gian kiểm | **0** / 81,5 s | **0** / 81,7 s |
| Bổ sung: tên lớp **mọi box** (`--buoc-lop 1`) | 1.271/1.272 (car 1.260, truck 12), exit 1 | 1.269/1.270 (car 1.258, truck 12), exit 1 |
| `opencv` / `opencv_render` | 3.2.0 / 3.2.0 | 3.2.0 / 3.2.0 |

- **Render lại** bằng `render_mot_video` v10 ra file mới (`render_v10_doi_chieu/`): sha256 **giống hệt** hai video vòng 23 → video là bản
  render đúng của MOT/lớp đã bàn giao và refactor không đổi đầu ra. Hai video cũ **giữ nguyên**, không render lại.
- **Box duy nhất không nhận dạng được khi kiểm mọi box**: frame 90, ID 30 (cùng box ở cả hai video), lớp ghi `car`, bộ kiểm chọn `bus`
  (12,30 vs 12,57 trên 205 điểm). Phân tích trên Nano (`phan_tich_f90_id30_nano.py` → `kiem_lai_nano_v10/phan_tich_f90_id30_*/f90_id30.json`): nhãn `30 car` nằm trọn trong ảnh (x 909–952 < 960),
  không chồng box khác; chữ màu ID 30 nét 1 px trên nền xám, tương phản chữ/nền 30,3 trong khi lỗi giải mã trên điểm chữ 22,5; trên
  129 điểm phân biệt car/bus: 15,54 vs 15,11. Đây là mất thông tin do nén mp4v, không phải vẽ sai (render lại byte-giống). Muốn mọi
  nhãn đọc được bằng máy thì phải đổi codec/chất lượng hoặc cỡ chữ — đổi định dạng video, **chưa làm**.

## 5. Phát hiện trong vòng: bộ kiểm chỉ hợp lệ khi cùng OpenCV với lúc render

Chạy thử bộ kiểm `9b1805c` trên PC (OpenCV 5.0.0) với chính video CV (hash khớp): số frame 299/300 sai, tên lớp 255/255 sai, nhưng
bbox/ID 0 lỗi. Tách nguồn trên 11 frame (`07_nguon_lech_pc_vs_nano.log`, PNG không mất mát xuất trên Nano): (a) giải mã mp4v PC vs
Nano lệch ≤ 4 mức (MAE 0,65–0,71); (b) `imread` JPEG giống hệt; (c) nét vẽ trên cùng ảnh gốc: khung chữ nhật **trùng từng điểm**,
toàn bộ lệch nằm ở chữ `putText` (774–5.315 điểm/frame, tới 251 mức) — OpenCV 5.0 vẽ font Hershey khác 3.2. Vì vậy thêm chốt phiên
bản (mục 2): với mã cuối, cùng lệnh trên PC trả **exit 2 `KHONG KIEM DUOC`** (`06_kiem_tren_pc_khac_opencv.log`). Kiểm trên Nano hợp
lệ (JSON v11 ghi `opencv_render` 3.2.0). Reviewer không có video (không nằm trong delta) và cần OpenCV 3.2.0 để kiểm lại; bằng chứng là
log/JSON/hash. Test tổng hợp (render và kiểm cùng một OpenCV) chạy được ở mọi phiên bản.

## 6. Đính chính diễn đạt vòng 23 (không đổi số)

- Phạm vi ổn định: **bốn lượt tracking, mỗi lượt khoảng 158–163 giây xử lý, trong phiên khoảng 13 phút 33 giây, có nghỉ và replay
  giữa lượt** — không phải một tiến trình chạy liên tục 14 phút, chưa phải kiểm chạy nhiều giờ.
- Nhiệt: **CPU tối đa 38 °C, GPU tối đa 35,5 °C** (tegrastats thô), không viết "mọi cảm biến ≤ 38 °C".
- Bộ nhớ: MemAvailable +3 MB giữa các mốc — không thấy tích lũy giữa bốn lượt; không suy ra đã loại trừ mọi rò rỉ.
- Kết luận "kiểm render ĐẠT" vòng 23 chưa gồm số frame hiển thị; thay bằng mục 4.

Đã sửa trong `REVIEW23_CHI_TIET.md` (ghi chú đầu phụ lục + mục 5), `NANO_STATUS.md` mục 4i; brief mới dùng cách viết đúng.
Số tracking/benchmark giữ nguyên. Render là **hậu kỳ** từ ảnh + MOT, không phải hiển thị trong lúc tracking.

## 7. Giới hạn còn lại

- Nhận dạng số frame chỉ xét ứng viên cách một chữ số; sai từ hai chữ số trở lên dựa vào phép tỷ lệ (a) — bắt được `f999` và mất số
  trong regression, chưa chứng minh cho mọi chuỗi.
- Tên lớp: mặc định lấy mẫu mỗi 5 frame (255 box/video); kiểm mọi box là bổ sung, còn 1 box không phân định do nén.
- Video không có `<video>.json` thì không xác minh được phiên bản OpenCV render (JSON ghi `null`).
- Hiển thị trực tiếp khi tracking và camera chưa làm; chưa chạy nhiều giờ; nguyên nhân hai mức decode của OpenCV 3.2 chưa rõ.

## 8. File và commit

Mã: `src/render_mot_video.py`, `src/kiem_video_render.py`, `src/test_render_mot_video.py`. Bằng chứng `handoff/logs/review24/`:
`00`–`07`, `repro_codex_{truoc,sau,ma_cuoi}/`, `regression_bat_ma_cu.log`, `kiem_lai_v24.sh`, `kiem_lai_nano_v10/`, `kiem_lai_nano_v11/`,
`kiem_tren_pc_khac_opencv/`, `nguon_lech_pc_vs_nano/`, `giai_ma_png_nano.py`, `so_nguon_lech_pc_nano.py`, `phan_tich_f90_id30_nano.py`,
`08_test_pc_launcher_lap_lai.log`, full suite. Bản sao từ Nano giữ nguyên byte (`.gitattributes` `* -text`, `sha256_tren_nano.txt`).
Video, PNG khung hình và gói Nano không commit. Commit: `c16d32b` (review), `9b1805c` (sửa N23-F1), `8111892` (bằng chứng sửa),
`4f1fa2f` (chốt phiên bản), `b21dc70`, `974affc` (bằng chứng/script), `7e12322` (bằng chứng Nano + full suite), rồi commit tài liệu
vòng 24 (phụ lục này, brief, `REVIEW23_CHI_TIET.md`, `NANO_STATUS.md`, `CODEX_HANDOFF.md`).
