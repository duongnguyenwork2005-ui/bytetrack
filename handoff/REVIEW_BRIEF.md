# REVIEW_BRIEF — vòng 12: chốt mốc PC (`074ceca`) + launcher Tkinter

<!-- BASE_TARGET -->

Review `074ceca`: **PASS WITH LIMITATIONS** cho mốc hệ thống PC Windows. Gói khớp chính xác 6.360.571 byte /
`fa03d7ea…` → lưu `handoff/archive/review_delta_074ceca.zip`; **base review → `074ceca`**. Nano F1/F3 hoãn.
Không tuning/train; 4 run `v2`, kết quả UA-DETRAC/BDD, GT, lock **không đổi**. Chi tiết:
**`handoff/REVIEW12_CHI_TIET.md`**; hướng dẫn: **`handoff/LAUNCHER_PC.md`**.

## Chốt mốc

Assertion S1 phụ thuộc Windows (mục 6.4) đã sửa: kiểm **roundtrip theo nền tảng** — Windows truyền chuỗi `replay`
với `shell=False` (CreateProcess → `CommandLineToArgvW`, không qua cmd) rồi so với `replay_argv`; POSIX `shlex.split`.
Không đổi tracking/tolerance. `test_track_video_file` 27/27.

## Launcher (`python src/pc_launcher.py`)

Tkinter/ttk có sẵn; **không** viết lại detector/association/vòng lặp — chỉ điều phối `track_video_file.py` trong
**worker process** (argv có cấu trúc, `shell=False`). `LauncherController` tách khỏi Tk (kiểm thử được);
`LauncherApp` là widget mỏng, `poll()` mỗi 200 ms, GUI không bị chặn khi nạp mô hình.

- Chọn video/thư mục ảnh (+FPS bắt buộc, frames manifest), checkpoint, bộ lọc **CV/EKF/UKF** (YAML đã khóa) hoặc
  YAML khác, thiết bị **thực sự hỗ trợ** (`0` nếu thấy CUDA, `cpu`), thư mục gốc; show / lưu video / lưu cache; nâng
  cao truyền **đúng** và ghi vào `run_meta` (mặc định = entry point). Mỗi lượt **một thư mục riêng**; **một** tác vụ.
- Tiến trình theo **frame thực sự đã xử lý**/tổng; không biết tổng → chỉ số frame. FPS nhãn rõ *fps vòng lặp*.
- **Dừng**: `STOP.flag` → worker hoàn tất frame hiện tại → **cùng đường với phím q** → ghi đủ tracking/cache/timing,
  `dung_som` + `ly_do_dung="stop-file"`. Đóng cửa sổ → dừng có kiểm soát, chờ 30 s, còn sống mới terminate (ghi log).
- Entry point thêm **tùy chọn** `--progress-file` (JSON giới hạn mỗi 10 frame hoặc ≥0,5 s, ghi nguyên tử) và
  `--stop-file` (kiểm trong `on_frame` sau khi vẽ/ghi). Hành vi CLI cũ không đổi.

## Kiểm chứng

| Loại | Kết quả |
|:--|:--|
| **Fixture** `test_pc_launcher` **10/10** — subprocess thật, entry point thật, chỉ giả lập detector/`imread`/GUI; mọi đường dẫn có dấu cách | build_argv đúng; đầu vào sai → không khởi chạy; hoàn tất; **dừng giữa chừng → replay parity ĐẠT**; hai tác vụ → `DangChay`; worker lỗi → LỖI (không "hoàn tất"); `shutdown()` dừng sạch; worker bỏ qua stop → terminate có log |
| **Thật, không GUI** (`logs/review12/smoke_launcher_that.log`) | controller → YOLO/GPU 150 frame, **CV và EKF** → HOAN_TAT → cache → replay subprocess → **parity ĐẠT lệch 0** |
| **Thật, cửa sổ Tk** (`handoff/anh/launcher_1..5_*.png`) | form điền bằng script, **Bắt đầu/Dừng bấm thật qua `invoke()`**, worker YOLO/GPU, dừng ở 153/1470 → ĐÃ DỪNG; ảnh chỉ chứa cửa sổ launcher |

Chưa smoke chuột tay; chưa camera/VFR/exe/Nano. Full suite trên code chốt: `handoff/logs/review12/`.
