# Phụ lục vòng 23 — chạy đầy đủ 1.470 frame trên Nano, kiểm ổn định, render video trên thiết bị

> **Đính chính vòng 24** (review `1d5ffaa`, chi tiết `handoff/REVIEW24_CHI_TIET.md`) — số liệu không đổi:
> (1) Bộ kiểm render ở mục 3 **chưa kiểm số frame hiển thị trên frame có box** (N23-F1); hai video có box ở cả 300 frame nên số frame
> chưa được kiểm. Vòng 24 kiểm lại trên Nano bằng bộ kiểm đã sửa: số frame 300/300 đúng, bbox/ID 0 lỗi, 255/255 nhãn mẫu, exit 0.
> (2) Mục 5: không phải "chạy liên tục ~14 phút" mà là **bốn lượt tracking, mỗi lượt khoảng 158–163 giây xử lý, trong phiên khoảng
> 13 phút 33 giây, có nghỉ và replay giữa lượt**. (3) Nhiệt: **CPU tối đa 38 °C, GPU tối đa 35,5 °C** — không viết "mọi cảm biến
> ≤ 38 °C". (4) MemAvailable +3 MB: không thấy tích lũy giữa các lượt; không suy ra đã loại trừ mọi rò rỉ bộ nhớ.

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Review trước: `handoff/CODEX_REVIEW_c4b39fe.md` (PASS WITH LIMITATIONS, đóng N21-F1/F2/F3).
Base review **chuyển sang `c4b39fe`** (gói `review_delta_c4b39fe.zip` 5.036.951 B, sha256 `536dee94…` khớp báo cáo); 9 NPZ không bàn
giao được khai báo tiếp trong `DEPENDENCIES.md`. Đầu vòng: HEAD `c4b39fe`, 0 commit sau đó, working tree chỉ có file riêng
(`logs/review23/00_trang_thai_dau.log`). Không training/tuning/đổi detector/INT8; không đổi tracker, runner, YAML khóa, kết quả UA-DETRAC/BDD,
benchmark vòng 20, bằng chứng vòng 21–22.

## 1. Thiết bị và đầu vào (`logs/review23/01_*`, `02_*`)

- Nano `10.11.56.132` (không đổi), L4T R32.7.5, boot 11:53:15 (không khởi động lại mới), thẻ `mmcblk1p1` rw, ext4 `errors_count` 0,
  28 GB trống, RAM trống ~2,57 GB, swap 15 MB, MAXN, `schedutil`. Con: Python 3.6.9, TensorRT 8.2.1.9, OpenCV 3.2.0; cha: Python 3.8.20,
  numpy 1.24.4, pandas 2.0.3, lap 0.5.13. Engine `c32690cd…` (12.626.138 B, build vòng 21, **không build lại**).
- Mã chạy: gói **v8** (`~/nano_run/v8`, đóng gói `b0c645b`): 7 file lõi (`nano_live`, `nano_detect_stream`, `nano_stream_proto`,
  `nano_detector`, `nano_replay`, `run_tracking`, `det_cache`) có sha256 LF **bằng HEAD `c4b39fe`**; `git diff b0c645b..c4b39fe` trên
  `src/ scripts/ configs/ third_party/` rỗng.
- Ảnh: chép **1.470 JPG gốc** `MVI_39031` (132.708.676 B) sang `~/nano_run/data_full/MVI_39031`; SHA256 **1470/1470 khớp** danh sách PC
  (`pc_sha256_MVI_39031_1470.txt`); thứ tự `liet_ke_anh` = `img00001…img01470`; frame 1–300 **byte giống** bộ smoke đã nghiệm thu.
  (Lần chép đầu bằng `tar | ssh` không tạo file nào — phát hiện ngay, chép lại bằng `scp`.)

## 2. Bốn lượt 1.470 frame (điều kiện công bố trước `b7da4f3`: `dieu_kien_chay_dai_v23.md`, driver `chay_dai_1470.sh`)

Đọc ảnh → TensorRT FP16 → tracker bằng entry point đã nghiệm thu; CV → EKF → EKF → CV, nghỉ 20 s, K = 4, không render, không `--regions`,
`--save-cache` riêng mỗi lượt; sau mỗi lượt: replay bằng chính `replay_argv` → `nano_parity` → so byte → kiểm tài nguyên. Log:
`nano_dai_v23/results_nano/dai_v23.log` (Nano 14:28:32 → 14:42:05).

| Lượt | live exit / trạng thái | con sau lượt | replay exit | parity exit | MOT live vs replay |
|:--|:--|:--|:--|:--|:--|
| 1 CV | 0 / hoàn tất | đã thoát (mã 0) | 0 | 0 | **byte giống hệt** |
| 2 EKF | 0 / hoàn tất | đã thoát | 0 | 0 | **byte giống hệt** |
| 3 EKF | 0 / hoàn tất | đã thoát | 0 | 0 | **byte giống hệt** |
| 4 CV | 0 / hoàn tất | đã thoát | 0 | 0 | **byte giống hệt** |

**Tính lại độc lập trên PC** (`phan_tich_chay_dai.py` → `05_phan_tich_chay_dai.{log,json}`, từ bản sao 77 file sha256/byte khớp nguồn):

- Mọi lượt: nhận = xử lý = EOS = **1.470**; cột `frame` của pipeline và timing đúng **1…1470, mỗi frame một lần**; 11 frame không còn
  detection sau lọc c25 (0 frame rỗng ở mức thô) vẫn qua tracker. MOT: CV **8.780 hàng / 74 ID**, EKF **8.765 hàng / 78 ID**.
- Live vs replay: **byte giống hệt** cả `data/` và `data_cls/`; so từng hàng (frame, id, bbox, conf) **không ánh xạ lại ID**: cùng số
  hàng, ID từng hàng trùng, lệch bbox/conf **0**. `nano_parity` (bbox 1e-3, conf 1e-6, ID toàn sequence) ĐẠT.
- Tính xác định: det cache 4 lượt **giống hệt** (`8eb7ab80…`); MOT 2 lượt CV giống hệt (`5017f37b…`), 2 lượt EKF giống hệt (`645bba3a…`).
- Cửa sổ đo: cả 4 lượt frame 1 bắt đầu đọc sau BẮT ĐẦU và frame 1470 xong trước hết cửa sổ; FPS tính lại khớp metadata (lệch < 1e-9).

| Lượt | `fps_xu_ly` | `fps_on_dinh` | `xu_ly_s` | decode tv | infer tv | `tracker.update()` tv | trễ tv (p90) | khởi tạo / kết thúc / ghi (s) |
|:--|--:|--:|--:|--:|--:|--:|--:|:--|
| 1 CV | 9,310 | 9,311 | 157,89 | 11,46 | 73,78 | 6,74 | 115,3 (122,5) | 8,44 / 0,73 / 0,70 |
| 2 EKF | 9,049 | 9,045 | 162,45 | 16,31 | 73,71 | 7,93 | 120,0 (128,3) | 9,52 / 0,73 / 0,70 |
| 3 EKF | 9,045 | 9,048 | 162,52 | 16,25 | 73,71 | 8,00 | 120,3 (127,7) | 8,14 / 0,73 / 0,69 |
| 4 CV | 9,137 | 9,141 | 160,89 | 11,87 | 73,72 | 6,78 | 116,9 (126,7) | 8,22 / 0,73 / 0,74 |

(ms/frame trừ khi ghi s; tóm tắt bỏ 25 frame đầu.) Theo cấu hình (2 lượt): **CV 9,22 FPS (9,14–9,31)**, **EKF 9,05 FPS (9,04–9,05)**;
`tracker.update()` CV 6,76 / EKF 7,96 ms. **Cách đọc**: ở vòng này hai lượt EKF rơi vào mức decode chậm (~16,3 ms) còn hai lượt CV ở mức
nhanh (~11,5 ms) — **ngược** vòng 22. Chênh FPS giữa hai cấu hình vì thế chủ yếu phản ánh decode, không cho phép xếp hạng; phần tracker
(~1,2 ms/frame) chạy song song với detector ở tiến trình kia. Không kiểm định; không suy ra HOTA/IDF1.

**Tài nguyên / ổn định** (`tai_nguyen.csv`, tegrastats thô từng lượt): MemAvailable trước lượt 2.627.884 → 2.630.892 kB (+3 MB qua 4 lượt,
không tăng dần); tegrastats RAM tối đa 2.667 MB mọi lượt, swap 16 MB; nhiệt tối đa CPU 38 °C, GPU 35,5 °C; CPU 1224–1479 MHz; 0 tiến trình
con còn chạy sau mọi lượt; boot không đổi; ext4 0 lỗi; số frame đang bay tối đa 2 (cửa sổ K = 4) ở cả 4 lượt.

## 3. Render video trên Nano (render **hậu kỳ**, không phải hiển thị trong lúc tracking) — `logs/review23/06_*`, `render_v23/`

- Sửa tối thiểu `src/render_mot_video.py` để chạy bằng **Python 3.6 hệ thống** (có OpenCV 3.2): bỏ `from __future__ import annotations`,
  tách `doc_mot`/`doc_lop`/`ve_frame` dùng chung; ghi `<out>.json` (số frame, fps, kích thước, số box, sha256). Không tạo môi trường mới,
  không cài gói. Mới: `src/kiem_video_render.py` (cú pháp 3.6) — số frame/FPS/kích thước; **mọi frame** vẽ lại bằng chính `ve_frame` và so
  trên các điểm vẽ (khung bbox, màu theo ID, số frame — *đính chính vòng 24: trên frame có box, số frame không thực sự được so*); **tên lớp**: mỗi box ở mỗi 5 frame, chọn ứng viên tên lớp gần frame giải mã nhất.
  Regression `src/test_render_mot_video.py` **7/7** (đúng → ĐẠT; MOT lệch 25 px, sai FPS, thiếu frame, sai tên lớp → KHÔNG ĐẠT; cú pháp 3.6).
- Gói **v9** `ea1fb21f…` (63.864.847 B) → `~/nano_run/v9`, manifest 556/556; mã tracking v9 == v8 (7/7 file). Trên Nano, từ MOT/lớp của
  lượt 1 CV và lượt 2 EKF, 300 frame đầu: render 17,3 s / 17,6 s, video mp4v 960×540 25 fps ~20,4 MB (`5f55fcb4…`, `c7d40019…`).
  **Kiểm trên Nano: ĐẠT** (bộ kiểm vòng 23, chưa gồm số frame trên frame có box — xem đính chính vòng 24) — 300/300 frame, 25 fps, kích thước khớp, 0 frame lỗi vị trí (tỷ lệ lỗi tối đa 0,32 / ngưỡng 0,5), 255/255 box
  mẫu đúng tên lớp (car 252, truck 3). Video không đưa vào repo/delta (bản sao để xem: `handoff/logs/review23/video_khong_commit/`, không commit).
- **Hiển thị trực tiếp: chưa làm.** Nano có màn hình HDMI 1920×1080 đang kết nối, nhưng tiến trình tracker không có OpenCV và giao thức chưa
  có kênh trả kết quả về tiến trình có ảnh; cần thiết kế riêng. Camera không thuộc mốc này.

## 4. Kiểm thử PC

`test_render_mot_video` 7/7, `test_nano_detector` 6/6; full suite: `**35/35 PASS** (`logs/review23/tests_20260928-1349.log`, trên `cf8b760`)`.

## 5. Giới hạn

Một sequence (1.470 frame), 2 lượt/cấu hình: bốn lượt tracking, mỗi lượt khoảng 158–163 giây xử lý, trong phiên khoảng 13 phút
33 giây, có nghỉ và replay giữa lượt (đính chính vòng 24) — chưa phải kiểm vận hành nhiều giờ. Hai mức decode của OpenCV 3.2 chưa
rõ nguyên nhân. Reviewer không có engine/ảnh/NPZ để tự chạy lại TensorRT; bằng chứng thiết bị là log + CSV + MOT + hash. Khi chính tracker
ném ngoại lệ, runner chưa trả MOT đã tích lũy (giới hạn cũ, không đổi). Hiển thị trực tiếp và camera chưa làm.

## 6. File

Mã: `src/render_mot_video.py`, `src/kiem_video_render.py` (mới), `src/test_render_mot_video.py` (mới), `src/run_all_tests.py`.
Bằng chứng `handoff/logs/review23/`: 00–06, `dieu_kien_chay_dai_v23.md`, `chay_dai_1470.sh`, `pc_sha256_MVI_39031_1470.txt`,
`phan_tich_chay_dai.py`, `nano_dai_v23/` (77 file từ Nano: run_meta, pipeline/timing CSV, MOT live + replay, parity, tegrastats, tài nguyên;
sha256/byte khớp nguồn; 4 NPZ cache bị loại khỏi delta, khai báo hash), `render_v23/` (JSON render + kiểm), full suite.
