# Launcher PC — hướng dẫn sử dụng

Cập nhật 2026-09-21 (vòng 13, sau review `2d1a8df`). Entry point: **`src/pc_launcher.py`** (Tkinter/ttk, có sẵn trong Python; ngoài `src/demo/**`).

```bash
python src/pc_launcher.py
```

Launcher **không** viết lại detector, association, vòng đời track hay vòng lặp tracking. Nó chỉ điều phối
`src/track_video_file.py` (đường chạy đã nghiệm thu ở review `074ceca`) trong một **worker process** riêng
(argv có cấu trúc, `shell=False`), đọc tiến trình từ `--progress-file` và yêu cầu dừng qua `--stop-file`.

## Chọn nguồn → chọn bộ lọc → chạy/dừng → xem kết quả

| Bước | Thao tác | Ghi chú |
|:--|:--|:--|
| 1. Nguồn | nút **…** → file video (`.mp4/.avi/…`) hoặc thư mục ảnh | thư mục ảnh **bắt buộc** nhập FPS; tên ảnh không suy ra được thứ tự → chọn *frames manifest* |
| 2. Detector / bộ lọc | checkpoint `.pt`; **Bộ lọc**: CV, EKF, UKF (YAML đã khóa) hoặc *Khác* → chọn YAML ở phần nâng cao; **Thiết bị**: chỉ liệt kê những gì pipeline chạy được (`0` nếu thấy CUDA, `cpu` luôn có) | dò CUDA chạy nền, không chặn giao diện |
| 3. Kết quả | thư mục gốc; tùy chọn *Xem cửa sổ tracking* (`--show`), *Lưu video*, *Lưu detection cache* | **mỗi lượt tạo thư mục con riêng** `<tên nguồn>_<bộ lọc>_<YYYYmmdd-HHMMSS>` — không ghi đè |
| 4. Nâng cao | conf-min, classes, max-det, track-buffer-sec, det-conf, det-max-det, warmup-frames, seed, allow-truncated | **mặc định = mặc định của `track_video_file.py`**; giá trị nhập được truyền **đúng** và ghi vào `run_meta.json`; launcher không âm thầm đổi ngưỡng/lớp/buffer/seed. **Mọi trường số** (kể cả *Số frame tối đa*, *Seed*, *FPS*) được kiểm **trước** khi khởi chạy: sai kiểu / ngoài miền → hộp thoại ghi **tên ô** và giá trị đã nhập, không tạo worker |
| ▶ Bắt đầu | kiểm đầu vào **trước** khi khởi chạy (nguồn/checkpoint/YAML tồn tại, FPS hợp lệ) → lỗi báo ngay, không tạo tiến trình | chỉ **một** tác vụ tại một thời điểm |
| ■ Dừng | tạo `STOP.flag` trong thư mục lượt → worker **hoàn tất frame hiện tại** rồi dừng qua **cùng đường với phím q** (`track_video` break → generator close → chốt timing → ghi đủ tracking/cache/timing). Có tác dụng **cả khi đang tạm dừng bằng SPACE** trong cửa sổ tracking (vòng chờ SPACE kiểm `STOP.flag` mỗi 50 ms; không đọc thêm frame) | trạng thái *ĐÃ DỪNG*; `run_meta.ket_qua.dung_som=true`, `ly_do_dung="stop-file"` (phím q → `"phim q"`) |
| 📂 Mở thư mục kết quả | mở thư mục lượt vừa chạy | |

**Trạng thái**: SẴN SÀNG → ĐANG KHỞI TẠO (nạp mô hình) → ĐANG CHẠY → (ĐANG DỪNG →) HOÀN TẤT / ĐÃ DỪNG / LỖI.
Worker thất bại (exit ≠ 0) hoặc bị terminate → **LỖI** kèm thông báo từ `progress.json` và log; không bao giờ báo
"hoàn tất" khi worker thất bại.

**Tiến trình**: thanh tiến trình theo **frame thực sự đã xử lý** / tổng khai báo (video/thư mục có tổng); nếu
không biết tổng thì hiện số frame đã xử lý, không phần trăm giả. Khi chuyển từ hoạt ảnh khởi tạo sang đếm frame,
timer tự tăng được **dừng trước** rồi mới đặt giá trị — thanh không tự nhích khi worker đứng yên (kể cả lúc tạm
dừng SPACE; nhãn hiện *TẠM DỪNG*). **FPS** trên giao diện là *fps vòng lặp* (kể cả warm-up, vẽ/ghi/hiển thị nếu
bật, cả thời gian tạm dừng; **không** gồm khởi tạo và ghi kết quả) đọc từ `progress.json`. Ở trạng thái cuối, worker
ghi `n_frame / wall_vong_lap` **đã chốt** nên số này **bằng đúng** `run_meta.thoi_gian.fps_vong_lap`; tốc độ toàn lượt
(gồm khởi tạo + ghi) ghi riêng là `fps_toan_luot`. Không phải fps tracker/detector.

**Đóng cửa sổ khi đang chạy**: hỏi xác nhận → yêu cầu dừng có kiểm soát → **chờ bằng `after()`** (giao diện vẫn
phản hồi, trạng thái vẫn cập nhật) → chỉ đóng cửa sổ **sau khi worker đã kết thúc**. Worker không phản hồi sau
`timeout_dung` (30 s) → **terminate** (đường cuối, ghi rõ vào log, trạng thái LỖI, hộp thoại cảnh báo kết quả có thể
chưa đầy đủ). Không để worker chạy ngầm. Đóng khi đang tạm dừng SPACE cũng dừng có kiểm soát.

## Thư mục một lượt

```
<gốc>/<tên>_<bộ lọc>_<thời điểm>/
    launcher_argv.json     argv đã chạy (mỗi đối số một phần tử)
    progress.json          tiến trình cuối: trang_thai, frame, n_frames_khai_bao, fps_vong_lap_tam (== run_meta
                           fps_vong_lap), wall_*_s, fps_toan_luot (ghi nguyên tử)
    STOP.flag              chỉ có khi đã bấm Dừng
    data/, data_cls/, timing/, pipeline/, frames_manifest.json, run_meta.json   (như track_video_file.py)
    <tên>_tracked.mp4      nếu Lưu video
```

Replay: `run_meta.json` → `replay_argv` (thay `<split>`/`<ten>`) → `subprocess.run(argv, shell=False)`.

## Cơ chế điều phối trong entry point (tùy chọn, không đổi hành vi tracking)

- `--progress-file p.json`: JSON `{trang_thai, frame, n_frames_khai_bao, t_vong_lap_s, fps_vong_lap_tam, pham_vi_fps, …}`,
  ghi **giới hạn** mỗi `--progress-every` (10) frame hoặc ≥ 0,5 s, **nguyên tử** (`.tmp` + replace); trạng thái
  `dang_khoi_tao / dang_chay / hoan_tat / da_dung / loi` (+ `loi`, `out`, `ly_do_dung`).
- `--stop-file f`: kiểm trong `on_frame` **sau** khi vẽ/ghi frame hiện tại **và** trong vòng chờ tạm dừng SPACE
  (mỗi 50 ms) → trả `False` → đường dừng đã kiểm chứng; `ly_do_dung="stop-file"` (q → `"phim q"`).
- Trạng thái cuối (`hoan_tat`/`da_dung`): `fps_vong_lap_tam = n_frame / wall_vong_lap` đã chốt (== `run_meta`), kèm
  `wall_khoi_tao_s / wall_vong_lap_s / wall_ghi_s / wall_tong_s` và `fps_toan_luot` (+ `pham_vi_fps_toan_luot`).

## Đã kiểm chứng

`src/test_pc_launcher.py` **17/17** — worker là **subprocess thật** chạy `track_video_file.main()` thật qua wrapper
chỉ giả lập detector, `cv2.imread` (có độ trễ) và GUI cv2 (`waitKey` giả SPACE/q); 1–10 không cần Tk, 11–13 dùng
cửa sổ Tk thật (ẩn; không có display → `[BO QUA]`, không tính PASS):

| # | Luồng | Kết quả |
|:--|:--|:--|
| 1 | `build_argv`: nguồn `traffic clip`, checkpoint `ck dir/gia.pt`, YAML `cfg dir/ekf ctrv.yaml`, conf-min/max-det/track-buffer/seed/`--set` | mỗi giá trị **một phần tử**; tham số hiệu lực đúng; thư mục lượt riêng |
| 2 | nguồn/checkpoint/YAML không tồn tại, thiếu FPS, FPS âm | `CauHinhSai` **trước** khi khởi chạy |
| 3 | Bắt đầu → hoàn tất (8/40 frame) | HOAN_TAT, `progress.json` `hoan_tat` 8/40, đủ output |
| 4 | Bắt đầu → **Dừng** ở frame ≥ 5 | DA_DUNG; `dung_som`, `ly_do_dung="stop-file"`; timing == frame xử lý; pipeline hữu hạn; manifest 40 / đã xử lý; **replay từ cache (subprocess) parity ĐẠT** |
| 5 | Bắt đầu lần hai khi đang chạy | `DangChay`; tác vụ đầu vẫn dừng được bình thường |
| 6 | worker lỗi (det-conf 0,5 > conf-min 0,1 → exit 2) | **LỖI** + thông báo từ `progress.json`; không "hoàn tất" |
| 7 | `shutdown()` khi đang chạy | dừng có kiểm soát, DA_DUNG, **không** terminate |
| 8 | worker **bỏ qua** stop-file | sau timeout → terminate (đường cuối), có log, trạng thái LỖI |
| 9 / 9b / 9c | đang **tạm dừng SPACE** → Dừng / `shutdown()` / phím q | DA_DUNG, **không terminate**, `stop-file`, không đọc thêm frame, timing hữu hạn, manifest đúng, replay parity ĐẠT; q → `"phim q"` |
| 10 | trễ 1 s sau vòng lặp (lưu cache) | fps cuối trong progress **== chính xác** `run_meta.fps_vong_lap`; `fps_toan_luot` riêng |
| 11 | Tk thật: khởi tạo → chạy 10/100, worker đứng yên 0,5 s | thanh **đứng yên** ở 10; frame 37 → 37 |
| 12 | Tk thật: `abc`/`x`/`0`/`-1` vào max_frames / seed / conf_min / track_buffer, bấm Bắt đầu | hộp thoại ghi **tên ô**, không worker, không exception; sửa hợp lệ → HOAN_TAT |
| 13 | Tk thật: đang SPACE → đóng cửa sổ (X) | `_dong()` trả về ngay, chờ bằng `after()`, cửa sổ đóng sau khi worker DA_DUNG, không terminate |

**Smoke thật vòng 12** (`handoff/logs/review12/smoke_launcher_that.log`): controller → `track_video_file.py` với **YOLO
thật/GPU** trên 150 frame `MVI_39031`, CV và EKF: HOAN_TAT, exit 0; cache → replay `subprocess.run(replay_argv)` →
**parity ĐẠT lệch 0**. Cửa sổ Tk thật: ảnh `handoff/anh/launcher_{1..5}_*.png` — Bắt đầu/Dừng qua `invoke()`, dừng
ở 153/1470 → ĐÃ DỪNG.

**Smoke thật vòng 13** (`handoff/logs/review13/smoke_launcher_that.log`, output `results/track_video/launcher_smoke13/`;
**thao tác tự động, không phải tay**: SPACE = `PostMessage(WM_KEYDOWN/UP)` vào cửa sổ cv2 thật, nút = `.invoke()`,
đóng = `_dong()`): (A) CV/EKF 150 frame → HOAN_TAT → replay parity ĐẠT lệch 0, fps cuối progress == run_meta;
(B) EKF `--show` + cache: thanh == frame worker 10/10 lần; SPACE → *TẠM DỪNG* ở frame 148, thanh đứng yên 2 s;
**Dừng** → ĐÃ DỪNG sau 1,16 s, `stop-file`, 148 frame, replay parity ĐẠT (ảnh `launcher_6_tam_dung_space.png`,
`launcher_7_da_dung_tu_tam_dung.png`); (C) CV `--show`: SPACE → đóng cửa sổ → đóng sau 1,32 s khi worker ĐÃ DỪNG,
không terminate, PID worker không còn.

**Full suite** (`src/run_all_tests.py`, đã đăng ký `test_pc_launcher`): vòng 12 31/31 (`logs/review12/`); vòng 13
**31/31 PASS** trên code chốt `56ce2c1` (`handoff/logs/review13/tests_20260921-1231.log`).

## Giới hạn

- Chưa smoke bằng tay (chuột/bàn phím thật); nút qua `invoke()`, SPACE qua `PostMessage`, đóng qua `_dong()`.
- Chưa có camera trực tiếp, VFR, đóng gói executable, Nano.
- `--show` dùng cửa sổ cv2 của entry point (q/SPACE trong cửa sổ đó); launcher chưa có bộ phát video riêng.
