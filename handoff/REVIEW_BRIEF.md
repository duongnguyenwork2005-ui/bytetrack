# REVIEW_BRIEF — vòng 25: dùng tại chỗ trên Jetson Nano (HDMI + chuột + bàn phím)

<!-- BASE_TARGET -->

Base **`f86732b`** (review `handoff/CODEX_REVIEW_f86732b.md`: PASS WITH LIMITATIONS, đóng N23-F1; delta đã review được lưu, hash
khớp). Detector, ngưỡng, YAML khóa, tracker và kết quả thí nghiệm giữ nguyên. Chi tiết: **`handoff/REVIEW25_CHI_TIET.md`**; hướng dẫn
người dùng: **`handoff/NANO_LOCAL_UI.md`**.

## Đã làm

- **Launcher Tk tiếng Việt** (`src/nano_launcher.py`, env `nano`): chọn video/thư mục ảnh, CV/EKF, thư mục lưu, Bắt đầu/Dừng, trạng
  thái, tóm tắt. Engine/Python đọc từ `configs/nano_ui.json`; sai thì báo cách sửa và khóa nút. Tái sử dụng
  `pc_launcher.LauncherController`; mỗi lượt một thư mục mới.
- **Hiển thị trực tiếp** trong tiến trình detector (Python 3.6, OpenCV GTK — nơi duy nhất có ảnh đã giải mã). Giao thức 3: sau
  `tracker.update` của frame *i*, `nano_live` gửi `R(i, hàng MOT/lớp)` thay credit; con ghép ảnh gốc **cùng frame_idx**, vẽ bằng
  `render_mot_video.ve_frame`. Hàng đợi hiển thị 2 frame, đầy thì bỏ và đếm; tracking không chờ hiển thị. Tắt hiển thị: y hệt vòng 24.
- **Dừng**: nút, `q`/`Esc`, đóng cửa sổ xem, đóng launcher → cùng stop-file → dừng có kiểm soát (mã 4), ghi đủ phần đã xử lý.
- Kiểm màn hình trước khi khởi động; `progress.json`, `hien_thi.json`, `tong_s`, `fps_hien_thi`; không có tạm dừng. Lệnh mở và
  biểu tượng desktop/menu (không sudo); chặn khóa màn hình khi rảnh chỉ trong lúc chạy.

## Bằng chứng

- `test_nano_hien_thi`: PC 16/16 (+1 SKIP POSIX), Nano 17/17 (gồm Tk trên HDMI); `test_nano_stream` 20/20; `test_pc_launcher` 17/17.
  Full suite PC: **36/36 PASS** trên mã cuối `a73eb1e` (`logs/review25/tests_20260929-1041.log`); lần trước 36/36 trên `91c95c7`.
- TensorRT thật, 14 lượt: detection giống hệt; **bật/tắt hiển thị → MOT + lớp giống byte**; live == replay, `nano_parity` ĐẠT; ảnh xem
  trước khớp pixel với render từ MOT đã ghi.
- **1.470 ảnh CV và EKF có hiển thị trên HDMI**: hiện 1470/1470, bỏ 0; MOT + lớp **giống byte kết quả vòng 23**.
- Ảnh hưởng hiển thị (300 ảnh, ABBA, 2 lượt/nhóm, cùng mức decode): FPS xử lý CV 9,21 bật / 9,39 tắt, EKF 9,18 / 9,38 (≈ −2 %); FPS
  hiển thị ≈ 9,2; CPU +6 điểm %. Mô tả, không xếp hạng CV/EKF.
- Vòng đời: hai lượt một phiên, Dừng giữa chừng, nguồn mp4, đóng launcher, **phím q** và **đóng cửa sổ xem** trên HDMI, mở từ **biểu
  tượng desktop** (cha systemd, không thuộc SSH) — đúng trạng thái, không còn tiến trình/cửa sổ. Ảnh màn hình:
  `logs/review25/anh_man_hinh/`.

## Chặn đã gặp

Màn hình Nano tự khóa sau 5 phút rảnh nên xdotool không vào được cửa sổ (có log). Không dùng mật khẩu qua SSH; người dùng mở khóa, pha
HDMI chạy tự động. Sau khi xem trên HDMI (gói v14): bỏ dòng `fatal:` trong nhật ký, biểu tượng desktop hiện đúng tên. Packager
delta gặp `WinError 206` (dòng lệnh quá dài): chia lô `git diff` (`a73eb1e`), test 19 bắt được mã cũ.

## Giới hạn và phần của người dùng

Chưa camera; không tạm dừng; một lượt tại một thời điểm; FPS phụ thuộc hai mức decode của OpenCV 3.2. Thao tác được tự động hóa
(`--tu-dong-*`, xdotool, gtk-launch); nút bấm bằng chuột thật và hộp chọn file chưa được người bấm. Người dùng đã mở khóa màn hình;
checklist 5 bước (`NANO_LOCAL_UI.md` mục 8) là tùy chọn, không chặn bàn giao.
