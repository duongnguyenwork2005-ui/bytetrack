# Phụ lục vòng 12 — chốt mốc PC (`074ceca`) và launcher

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Hướng dẫn launcher: `handoff/LAUNCHER_PC.md`. Hệ thống PC: `handoff/HE_THONG_PC.md`.

## 0. Trạng thái trước khi sửa

`handoff/logs/review12/trang_thai_truoc_khi_sua.log`: HEAD **đúng `074ceca`** (0 commit sau); working tree chỉ có
2 file riêng — không đụng. Gói đã nghiệm thu khớp **chính xác** 6.360.571 byte / sha256 `fa03d7ea…` → lưu
`handoff/archive/review_delta_074ceca.zip`; base review → `074ceca` (vòng 12).

## 1. Chốt mốc PC

| Việc | Kết quả |
|:--|:--|
| Review `074ceca` | **PASS WITH LIMITATIONS** — PC-F1…F4 + hai ca bổ sung đã đóng; lưu `handoff/CODEX_REVIEW_074ceca.md` |
| Assertion S1 phụ thuộc Windows (mục 6.4) | Bỏ `'"traffic clip"' in replay`. Thay bằng **roundtrip theo nền tảng**: Windows → truyền **chuỗi** `replay` với `shell=False` (CreateProcess → `CommandLineToArgvW` trong tiến trình con, **không** qua cmd) rồi so với `replay_argv`; POSIX → `shlex.split`. Không đổi logic tracking/tolerance. `test_track_video_file` 27/27 |
| Nano F1/F3 | vẫn hoãn |

## 2. Launcher — thiết kế

| Thành phần | Nội dung |
|:--|:--|
| `src/pc_launcher.py` `LauncherController` | **không phụ thuộc Tk**: `build_argv(cfg)` kiểm đầu vào trước khi khởi chạy (`CauHinhSai`), tạo thư mục lượt riêng; `start()` → `subprocess.Popen(argv, shell=False)` + thread đọc stdout → queue; `request_stop()` → tạo `STOP.flag`; `poll()` (GUI gọi mỗi 200 ms, không chặn) đọc `progress.json` nguyên tử, log, exit code → trạng thái; `force_terminate()` đường cuối có log; `shutdown()` khi đóng cửa sổ; `open_results()` |
| `LauncherApp (tk.Tk)` | widget mỏng bọc controller: 4 nhóm (Nguồn / Detector & bộ lọc / Kết quả / Nâng cao), Bắt đầu / Dừng / Mở kết quả, nhãn trạng thái, thanh tiến trình (determinate khi biết tổng, indeterminate khi khởi tạo), nhãn fps **ghi rõ phạm vi vòng lặp**, vùng log; dò CUDA chạy nền |
| Entry point (`track_video_file.py`) | thêm **tùy chọn** `--progress-file` (JSON, ghi giới hạn mỗi 10 frame hoặc ≥0,5 s, `.tmp` + `replace`), `--stop-file` (kiểm trong `on_frame` **sau** vẽ/ghi frame hiện tại → `return False` → **cùng đường với phím q**), `ket_qua.ly_do_dung`. Không đổi hành vi tracking; `test_track_video_file` 27/27 sau thay đổi |

Trạng thái: SAN_SANG → DANG_KHOI_TAO → DANG_CHAY → (DANG_DUNG →) HOAN_TAT | DA_DUNG | LOI. Worker exit ≠ 0 hoặc
bị terminate → LOI với thông báo; exit 0 mà không có trạng thái cuối trong `progress.json` → cũng LOI (không
báo "hoàn tất" vội).

## 3. Kiểm chứng (phân biệt fixture / GUI giả lập / thật)

**Fixture** — `src/test_pc_launcher.py` **10/10**: worker là **subprocess thật** chạy `track_video_file.main()`
thật qua wrapper chỉ giả lập `det_cache.detect_frame`, `cv2.imread` (30 ms/frame) và GUI cv2. Đường dẫn wrapper,
checkpoint, nguồn (`traffic clip`), YAML (`cfg dir/ekf ctrv.yaml`) đều có dấu cách. Các luồng: build_argv, đầu vào
sai → không khởi chạy, hoàn tất, dừng giữa chừng + replay parity, hai tác vụ, worker lỗi, `shutdown()`, worker
bỏ qua stop-file → terminate có log (timeout 2 s, mất 2,1 s).

**Thật, không GUI** — `handoff/logs/review12/smoke_launcher_that.log`: controller → `track_video_file.py`
**YOLO/GPU thật**, 150 frame `MVI_39031`, CV và EKF → HOAN_TAT exit 0 (9,6 s / 8,6 s) → cache → replay
`subprocess.run(replay_argv)` → **parity ĐẠT lệch 0** (394 và 395 hàng). Output: `results/track_video/launcher_smoke/`.

**Thật, có GUI** — `handoff/anh/launcher_{1..5}_*.png`: cửa sổ Tk thật, form điền bằng script, **Bắt đầu** và
**Dừng** bấm thật qua `.invoke()`, worker YOLO/GPU thật, dừng ở 153/1470 → ĐÃ DỪNG, log worker hiển thị trong
cửa sổ. Không phải thao tác chuột tay. Ảnh chỉ chứa cửa sổ launcher (đã cắt bỏ viền); lượt chụp đầu bị lẫn cửa sổ
khác của người dùng đã **xóa**, không đưa vào gói.

## 4. Toàn vẹn

Không sửa runner dùng chung (`run_tracking.py`, `det_cache.py`) trong vòng này; `track_video_file.py` chỉ thêm
điều phối tùy chọn. Full suite trên code chốt: `handoff/logs/review12/`. 4 run `v2`, kết quả UA-DETRAC/BDD, GT,
lock **không đổi**. Không đụng file riêng. mp4 của smoke GUI **không stage**.

## 5. Giới hạn

Chưa smoke chuột/bàn phím tay; chưa camera trực tiếp, VFR, đóng gói executable, training/tuning, Nano.
