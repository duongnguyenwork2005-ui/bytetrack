# Phụ lục vòng 13 — sửa L-F1…L-F4 (review `2d1a8df`, REQUEST CHANGES)

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Hướng dẫn launcher: `handoff/LAUNCHER_PC.md`. Review: `handoff/CODEX_REVIEW_2d1a8df.md`.
Base review **giữ `074ceca`** (Codex chưa nghiệm thu `2d1a8df`).

## 0. Trạng thái trước khi sửa

`handoff/logs/review13/trang_thai_truoc_khi_sua.log`: HEAD **đúng `2d1a8df`** (0 commit sau); working tree chỉ có
2 file riêng (`.gitignore`, `src/demo/rerun_with_state_log.py`) — không đụng. Môi trường thực: Python 3.11.9
Windows 11, numpy 2.4.6, scipy 1.17.1, OpenCV 5.0.0, torch 2.6.0+cu124 (CUDA có), ultralytics 8.4.141, Tk 8.6.

## 1. Tái hiện TRƯỚC khi sửa — `handoff/logs/review13/repro_truoc_khi_sua.log` (exit 0, script chạy hết)

Cùng một script (`repro_lf.py`, scratch) chạy trước và sau khi sửa. Phạm vi giả lập ghi rõ: worker là **subprocess
thật** chạy `track_video_file.main()` thật qua wrapper chỉ thay detector / `cv2.imread` / GUI cv2 (`waitKey` trả
SPACE ở frame 2, 255 trong vòng pause); L-F2/L-F4 dùng **cửa sổ Tk thật (ẩn)** với controller / messagebox thay thế.

| Finding | Tái hiện trên `2d1a8df` |
|:--|:--|
| L-F1 SPACE → Dừng | vào vòng pause → `request_stop()` (timeout fixture 3 s) → **terminate** sau 3,19 s, `exit=None`, `da_terminate=True`; **không có** `run_meta.json`, `data/`, `timing/` |
| L-F1b SPACE → `shutdown()` | → **LOI**, terminate, không `run_meta` |
| L-F3 trễ 1 s sau vòng lặp | `run_meta.fps_vong_lap` **27,36** (wall_vong_lap 0,146 s) vs progress cuối **2,36 fps** (t 1,696 s) → lệch |
| L-F2 thanh tiến trình | khởi tạo indeterminate → DANG_CHAY 10/100 value 10 → bơm event loop 0,5 s, worker vẫn frame 10 → value **14** (tự tăng) |
| L-F4 nhập `abc` | `max_frames` và `seed`: **`ValueError` thoát khỏi `_bat_dau`**, 0 hộp thoại |

## 2. Thay đổi

| Finding | File | Sửa |
|:--|:--|:--|
| L-F1 | `src/track_video_file.py` (`25c11b6`) | vòng pause `waitKey(50)` kiểm **stop-file** mỗi vòng → `return False` → **cùng đường với phím q** (`track_video` break → generator close → chốt timing frame cuối → ghi đủ tracking/cache/timing); không đọc thêm frame. `ly_do_dung="stop-file"` đặt ngay trong `_ve_hien_thi`; `on_frame` chỉ gán `"phim q"` khi chưa có lý do → q trong pause vẫn là `"phim q"`. Progress ghi `tam_dung=True/False` (ép) khi vào/ra pause |
| L-F3 | `src/track_video_file.py` (`25c11b6`) | `ghi_tien_trinh(..., t_vong_lap=wall_vong_lap)` ở trạng thái cuối (hoan_tat/da_dung/loi) → `fps_vong_lap_tam = n_frame / wall_vong_lap` **== `run_meta.thoi_gian.fps_vong_lap`** (cùng biến, cùng công thức); thêm `wall_khoi_tao_s / wall_vong_lap_s / wall_ghi_s / wall_tong_s` và **`fps_toan_luot`** (n_frame / wall_tong, kèm `pham_vi_fps_toan_luot`). Không đổi cách đo vòng lặp; kết quả lịch sử `v2` không đổi |
| L-F2 | `src/pc_launcher.py` (`2e96bc7`) | khi chuyển sang determinate: `pb.stop()` **trước** rồi mới `configure(maximum, value)` từ frame thật |
| L-F4 | `src/pc_launcher.py` (`2e96bc7`) | `_so()` parse + kiểm miền **mọi** trường số (`conf_min`, `max_det`, `track_buffer_sec`, `det_conf`, `det_max_det`, `warmup_frames`, `max_frames`, `seed`, `fps`) trong `build_argv` **trước** khi khởi chạy → `CauHinhSai` ghi **tên ô** + giá trị đã nhập; giá trị hợp lệ truyền đúng như nhập; `_bat_dau` bắt thêm `Exception` (không exception nào thoát callback) |
| Đóng cửa sổ (ghi chú review) | `src/pc_launcher.py` (`2e96bc7`) | `_dong()`: `request_stop()` rồi **chờ bằng `after(100, _cho_dong)`** — GUI phản hồi, `_tick` vẫn cập nhật; chỉ `destroy()` sau khi worker kết thúc; worker không phản hồi → `poll()` terminate sau `timeout_dung` (đường cuối) + `showwarning` "kết quả có thể chưa đầy đủ". **Không đổi timeout** |
| Hardening | `src/pc_launcher.py` (`2e96bc7`) | `_nap_thiet_bi` không gọi Tk từ thread phụ (vỡ khi app điều khiển bằng `update()`, nhãn kẹt "đang dò CUDA…" trong ảnh smoke đầu); ghi kết quả, `_tick` áp vào widget. `snapshot()` thêm `tam_dung` → nhãn "TẠM DỪNG (SPACE trong cửa sổ tracking)" |

## 3. Kiểm chứng SAU khi sửa

**Cùng script tái hiện** — `handoff/logs/review13/repro_sau_khi_sua.log` (exit 0): L-F1 → **DA_DUNG exit 0, không terminate**,
0,9–1,0 s sau yêu cầu, `run_meta` có, `ly_do_dung=stop-file`, `n_frame_xu_ly=2`; L-F1b `shutdown()` → DA_DUNG;
L-F3 progress **30,60 = run_meta 30,60** (wall_ghi 1,02 s không ảnh hưởng); L-F2 value **10 → 10**; L-F4 hộp thoại 1/1,
không exception, không worker.

**Regression** — `src/test_pc_launcher.py` **17/17** (`handoff/logs/review13/tests_sau_khi_sua.log`), thêm:

| # | Phép kiểm | Trên mã cũ `2d1a8df` (`regression_bat_ma_cu.log`) |
|:--|:--|:--|
| 9 | SPACE ở frame 2 → Dừng: worker tự kết thúc exit 0 **trước timeout**, DA_DUNG, `stop-file`, `n_frame_xu_ly == 2` (không đọc thêm frame), timing/pipeline hữu hạn, manifest `da_xu_ly == 2`, cache → **replay parity ĐẠT** | **FAIL** (LOI, terminate, không meta) |
| 9b | SPACE → `shutdown()`: DA_DUNG, không terminate, run_meta có | **FAIL** |
| 9c | phím q **trong** pause → `ly_do_dung="phim q"` (guard, không gán nhầm stop-file) | PASS (đúng: vốn không lỗi) |
| 10 | trễ 1 s quanh `save_video`: progress `fps_vong_lap_tam` **== chính xác** `run_meta.fps_vong_lap`; `t_vong_lap_s == wall_vong_lap_s < 1 s`; `wall_ghi_s ≥ 1 s`; `fps_toan_luot` < fps vòng lặp, có `pham_vi` | **FAIL** (3,10 vs 30,81) |
| 11 | Tk thật (ẩn): indeterminate → DANG_CHAY 10/100; bơm event loop 0,5 s, worker giữ frame → thanh **đứng yên 10**; frame 37 → value 37, max 100 | **FAIL** (10 → 14 → 38) |
| 12 | Tk thật, `_bat_dau()` thật: `abc`/`x`/`0`/`-1` vào max_frames / seed / conf_min / track_buffer → hộp thoại ghi **tên ô**, không worker, không exception; sửa hợp lệ → HOAN_TAT 3 frame | **FAIL** (ValueError thoát; conf_min `x` khởi chạy worker rồi LOI) |
| 13 | Tk thật: SPACE → `_dong()` (= nút X): trả về ngay (cửa sổ còn), event loop chạy, worker tự kết thúc → DA_DUNG, không terminate, cửa sổ đóng sau đó, không còn tiến trình | **FAIL** (terminate, LOI) |

Test 11–13 cần display; không có → in `[BO QUA]` và **không** tính PASS. Runner dừng ở FAIL đầu (9) theo thiết kế;
bảng "trên mã cũ" lấy từ **bản sao test không dừng** (chỉ bỏ `assert`, ghi rõ trong log). Test 8 (worker bỏ qua
stop-file → terminate sau timeout) vẫn giữ. `src/test_track_video_file.py` **27/27**.

**Smoke thật trên Windows** — `handoff/logs/review13/smoke_launcher_that.log`, output `results/track_video/launcher_smoke13/`,
YOLO/GPU thật, cửa sổ Tk + cửa sổ cv2 thật. **Thao tác tự động, không phải tay**: SPACE = `PostMessage(WM_KEYDOWN/UP)`
vào cửa sổ cv2 thật; nút = `.invoke()`; đóng = `_dong()` với `askyesno` trả True.

| Bước | Kết quả |
|:--|:--|
| A. controller, CV và EKF 150 frame, lưu cache | HOAN_TAT exit 0 (12,4 s / 13,5 s); replay subprocess **parity ĐẠT lệch 0** (394 / 395 hàng); progress cuối fps **33,0844 = run_meta 33,0844**, **29,4490 = 29,4490**; `fps_toan_luot` 28,8 / 25,5 |
| B. GUI EKF `--show` + cache | thanh tiến trình == frame worker **10/10 lần** (61…141); SPACE → `tam_dung` sau 0,05 s ở frame 148; 2 s tạm dừng: thanh **đứng yên (148, 148) ×8**; ảnh `launcher_6_tam_dung_space.png`; **Dừng** → DA_DUNG exit 0 **không terminate** sau **1,16 s** (gồm ghi kết quả/cache), `ly_do_dung=stop-file`, `n_frame_xu_ly=148` = frame lúc tạm dừng; replay **parity ĐẠT** (387 hàng); fps cuối **17,0018 = 17,0018** (gồm 2 s pause trong vòng lặp — đúng phạm vi); ảnh `launcher_7_da_dung_tu_tam_dung.png` |
| C. GUI CV `--show` → SPACE → đóng cửa sổ (X) | `_dong()` trả về ngay (cửa sổ còn), event loop 61 lần, cửa sổ đóng sau **1,32 s** khi worker đã DA_DUNG, không terminate; `run_meta` `stop-file`, 42 frame; `tasklist` PID worker **không còn** |

Ảnh chỉ chứa cửa sổ launcher. Dòng log đầu trong ảnh hiển thị đường dẫn interpreter conda (đã có trong
`CODEX_HANDOFF.md` từ các vòng trước) — không có đường dẫn riêng khác.

**Full suite** (`src/run_all_tests.py --out-dir handoff/logs/review13`, không sửa/commit trong lúc chạy):

| Lần | Commit | Kết quả | Ghi chú |
|:--|:--|:--|:--|
| 1 | `2e96bc7` | **30/31 FAIL** | `test_pc_launcher` phép 12: `UnicodeEncodeError` — runner bắt stdout **cp1252**, chi tiết phép kiểm chứa thông báo L-F4 có dấu; phép 13 không chạy. Các lần chạy tay đặt `PYTHONIOENCODING=utf-8` nên không lộ. Sửa: test `reconfigure` stdout utf-8 (`46daaa3`, chỉ đổi cách in). Log: `lich_su_that_bai/tests_20260921-1209.log` |
| 2 | `f685c67` | **30/31 FAIL** | phép 5: `DANG_DUNG` thay vì `DA_DUNG` — **race trong helper `_cho()` của test** (điều kiện `not dang_chay()` đánh giá sau `poll()`, worker thoát giữa hai lệnh → snapshot cũ). Lộ dưới tải suite; 4 lần chạy tay 17/17 không gặp. Sửa: poll lại sau khi thỏa (`56ce2c1`, chỉ đổi test). Log: `lich_su_that_bai/tests_20260921-1220.log` |
| 3 | `56ce2c1` | **31/31 PASS** trên code chốt `56ce2c1` (`handoff/logs/review13/tests_20260921-1231.log`) | code chốt của vòng; `test_pc_launcher` 17/17, `test_review_delta` 14/14, `test_track_video_file` 27/27 |

Hai lần FAIL đều là lỗi của **test** (in / race), không phải mã sản phẩm; log giữ nguyên trong `lich_su_that_bai/` kèm `GHI_CHU.md`.

## 4. Toàn vẹn

Không sửa `run_tracking.py`, `det_cache.py`, tracker, cấu hình khóa, GT, kết quả UA-DETRAC/BDD, 4 run `v2`.
`track_video_file.py` chỉ đổi điều phối (vòng pause + progress cuối); hành vi tracking không đổi
(`test_track_video_file` 27/27; replay parity ĐẠT trong mọi smoke). Không đụng file riêng; không mp4 mới
(smoke 13 không bật lưu video).

## 5. Giới hạn

- Thao tác phím/nút trong smoke đều **tự động**; chưa có smoke chuột/bàn phím tay.
- Chưa camera trực tiếp, VFR, đóng gói executable, Nano (F1/F3 vẫn hoãn).
- `fps_vong_lap` gồm cả thời gian người dùng tạm dừng (SPACE) — đúng định nghĩa "vòng lặp", nhãn ghi rõ.
