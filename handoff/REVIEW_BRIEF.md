# REVIEW_BRIEF — vòng 23: chạy đầy đủ 1.470 frame trên Nano, kiểm ổn định, render video trên thiết bị

<!-- BASE_TARGET -->

Base **`c4b39fe`** (review `handoff/CODEX_REVIEW_c4b39fe.md`, PASS WITH LIMITATIONS). Tracker, runner, luồng trực tiếp, YAML khóa,
UA-DETRAC/BDD, benchmark vòng 20 và bằng chứng vòng 21–22 không đổi. Chi tiết: **`handoff/REVIEW23_CHI_TIET.md`**.

## Đầu vào và mã chạy

Nano không đổi IP, không khởi động lại, ext4 0 lỗi; engine `c32690cd…` dùng lại; mã chạy là gói **v8** (7 file lõi có hash
bằng `c4b39fe`). 1.470 JPG gốc `MVI_39031` chép sang Nano: SHA256 **1470/1470** khớp PC, đúng thứ tự.

## Bốn lượt 1.470 frame (CV → EKF → EKF → CV; điều kiện và driver commit trước `b7da4f3`)

Mỗi lượt đều: live exit 0 và hoàn tất; con thoát 0; replay bằng chính `replay_argv` exit 0; `nano_parity` exit 0; **MOT live == replay
từng byte**. Tính lại độc lập trên PC (`logs/review23/05_*`) từ 77 file bản sao (sha256/byte khớp nguồn):

- Frame 1–1470 xử lý **đúng một lần**, kể cả 11 frame không còn detection sau lọc. Nhận = xử lý = EOS = 1.470.
- So từng hàng không ánh xạ ID: cùng số hàng (CV 8.780 / EKF 8.765), ID trùng, lệch bbox/conf 0.
- Tính xác định: det cache 4 lượt giống hệt; MOT cùng cấu hình giống hệt.
- Mọi frame đếm nằm trong cửa sổ đo; FPS tính lại khớp metadata.
- MemAvailable +3 MB qua 4 lượt, RAM tối đa 2.667 MB, nhiệt ≤ 38 °C, không còn tiến trình con.

| | `fps_xu_ly` (2 lượt) | `tracker.update()` | decode | trễ trung vị |
|:--|:--|:--|:--|:--|
| CV | 9,22 (9,14–9,31) | 6,76 ms | ~11,5 ms | 115–117 ms |
| EKF | 9,05 (9,04–9,05) | 7,96 ms | ~16,3 ms | 120 ms |

Hai lượt EKF rơi vào mức decode chậm của OpenCV 3.2 (ngược vòng 22), nên chênh FPS không cho phép xếp hạng CV/EKF. Detector (infer
~73,7 ms) quyết định thông lượng. Mô tả, không kiểm định, không suy ra HOTA/IDF1.

## Render video trên Nano (hậu kỳ)

- `render_mot_video.py` sửa tối thiểu để chạy bằng Python 3.6 hệ thống (có OpenCV), không cài gói. Bộ kiểm mới
  `kiem_video_render.py` vẽ lại từng frame bằng chính hàm vẽ: vị trí bbox, màu ID, số frame, tên lớp. `test_render_mot_video` 7/7.
- Gói v9 `ea1fb21f…` (mã tracking == v8). Trên Nano, 300 frame CV và EKF: render ~17,4 s; **kiểm trên Nano ĐẠT**: 300/300 frame,
  25 fps, 0 frame lỗi, 255/255 nhãn lớp mẫu đúng.
- **Hiển thị trực tiếp chưa làm**: có màn hình HDMI nhưng tiến trình tracker không có OpenCV, cần kênh hiển thị riêng. Camera chưa làm.

## Kiểm thử và giới hạn

Full suite PC: `**35/35 PASS** (`logs/review23/tests_20260928-1349.log`, trên `cf8b760`)`. Chạy liên tục ~14 phút trên một sequence, chưa phải vận hành nhiều giờ. Chưa rõ nguyên nhân hai mức
decode. Delta từ `c4b39fe` gồm MOT live + replay, CSV timing, metadata và telemetry; không có ảnh, engine, video hay ZIP; 4 NPZ cache
mới được khai báo hash.
