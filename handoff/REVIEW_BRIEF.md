# REVIEW_BRIEF — vòng 26: sửa N25-F1 và N25-F2 (giao diện tại chỗ trên Jetson Nano)

<!-- BASE_TARGET -->

Review `handoff/CODEX_REVIEW_20e66e9.md`: REQUEST CHANGES phần giao diện, hai finding P2; tracking và benchmark vẫn đạt. **Base giữ
`f86732b`**, chưa đánh dấu vòng 25 PASS. **Trạng thái: đã sửa, chờ Codex nghiệm thu.** Chi tiết: **`handoff/REVIEW26_CHI_TIET.md`**.

## Tái hiện trên mã chưa sửa

Script Codex giữ nguyên (`handoff/CODEX_REPRO_20e66e9_ui.py`). PC Windows: F1 tái hiện (`imshow` 1 lần nhưng `n_hien_thi` 2, frame
cuối 2, FPS 439,75), F2 SKIP (cần POSIX), exit 1. Trên Nano, hai hàm của script được gọi riêng vì không có Python nào vừa có OpenCV vừa
≥ 3.7: F1 (Python 3.6 + OpenCV 3.2) và F2 (env `nano`, tiến trình chọn file còn sống sau khi đóng launcher) đều exit 1. Kiểm tiến trình
theo executable: dòng “còn 1 picker” của `pgrep -f` là chính shell tìm kiếm.

## Sửa

- **N25-F1**: chỉ tính “đã hiển thị” khi cửa sổ còn mở (kiểm trước mỗi `imshow`) và `imshow` chạy xong. Frame tới sau khi đóng →
  không vẽ, không tạo lại cửa sổ, đếm riêng `n_khong_hien_thi_cua_so_dong`; MOT/lớp vẫn ghi đủ. `fps_hien_thi = (N−1)/(mốc cuối −
  mốc đầu)`, N < 2 → null; định nghĩa đồng bộ ở mã, `run_meta`, tóm tắt launcher, `NANO_LOCAL_UI.md`. Số benchmark cũ giữ nguyên.
- **N25-F2**: zenity chạy bằng `Popen`, launcher giữ handle, chờ bằng `after()` trên luồng Tk. Đóng launcher: hủy callback chờ, SIGTERM
  tới đúng PID đó rồi thu hồi; picker vừa tự thoát → chỉ thu hồi, bỏ kết quả; chưa thoát sau 2 s mới SIGKILL. Mọi `after()` bị hủy
  trước `destroy()`. Nhật ký cả phiên lưu vào `~/.cache/nano_tracking/`.

## Kiểm chứng

- Script Codex sau sửa: **exit 0** trên PC và cả hai phần trên Nano.
- `test_nano_hien_thi` thêm 10/10b/10c/11 (F1) và 12a–e (F2, tiến trình chọn file thật, cửa sổ Tk thật): PC **20/20** (+2 SKIP
  POSIX), Nano **23/23** (+1 SKIP: 10 cần cv2). Cùng test trên mã cũ: PC 16/20, Nano 18/23 — các ca F1/F2 FAIL. Suite liên quan:
  `test_nano_stream` 20/20, `test_pc_launcher` 17/17, render 13/13. Full suite PC: **36/36 PASS** (`logs/review26/tests_20260929-1401.log`, trên `6dbf40f`).
- Nano thật, gói **v16** (`~/nano_run/v16`, v14 giữ nguyên): TensorRT CV 300 ảnh, đóng cửa sổ xem bằng Alt+F4 giữa chừng → `da_dung`,
  **180 frame hiện, 1 frame không hiện vì cửa sổ đã đóng**, FPS hiển thị 9,35; live == replay giống byte, `nano_parity` ĐẠT; MOT + lớp
  181 frame giống tiền tố lượt không hiển thị.
- Zenity thật: mở → hủy → mở lại → chọn đúng đường dẫn; đóng launcher khi hộp chọn mở (chưa chạy và khi tracker đang chạy) → zenity bị
  SIGTERM đúng PID và thu hồi, tracker `da_dung`; mở từ biểu tượng desktop, 2 lượt, thoát sạch. Sau mỗi ca 0 tiến trình con, 0 cửa sổ.
- Ảnh màn hình Nano: `logs/review26/anh_man_hinh/`.

## Giới hạn

Thao tác cửa sổ được tự động hóa (xdotool, `gtk-launch`, tham số `--tu-dong-*` gọi đúng hàm của nút), chưa có người bấm chuột trong
vòng này. Nhánh SIGKILL (picker không phản hồi SIGTERM) chưa gặp với zenity thật. Trong harness của Codex, `scheduled_callbacks` vẫn là
1 vì stub không có `after_cancel`; callback đó là no-op, cửa sổ Tk thật cho 0 callback còn lại. Camera ngoài phạm vi.
