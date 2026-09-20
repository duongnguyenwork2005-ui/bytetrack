# Launcher PC — hướng dẫn sử dụng

Cập nhật 2026-09-21. Entry point: **`src/pc_launcher.py`** (Tkinter/ttk, có sẵn trong Python; ngoài `src/demo/**`).

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
| 4. Nâng cao | conf-min, classes, max-det, track-buffer-sec, det-conf, det-max-det, warmup-frames, seed, allow-truncated | **mặc định = mặc định của `track_video_file.py`**; giá trị nhập được truyền **đúng** và ghi vào `run_meta.json`; launcher không âm thầm đổi ngưỡng/lớp/buffer/seed |
| ▶ Bắt đầu | kiểm đầu vào **trước** khi khởi chạy (nguồn/checkpoint/YAML tồn tại, FPS hợp lệ) → lỗi báo ngay, không tạo tiến trình | chỉ **một** tác vụ tại một thời điểm |
| ■ Dừng | tạo `STOP.flag` trong thư mục lượt → worker **hoàn tất frame hiện tại** rồi dừng qua **cùng đường với phím q** (`track_video` break → generator close → chốt timing → ghi đủ tracking/cache/timing) | trạng thái *ĐÃ DỪNG*; `run_meta.ket_qua.dung_som=true`, `ly_do_dung="stop-file"` |
| 📂 Mở thư mục kết quả | mở thư mục lượt vừa chạy | |

**Trạng thái**: SẴN SÀNG → ĐANG KHỞI TẠO (nạp mô hình) → ĐANG CHẠY → (ĐANG DỪNG →) HOÀN TẤT / ĐÃ DỪNG / LỖI.
Worker thất bại (exit ≠ 0) hoặc bị terminate → **LỖI** kèm thông báo từ `progress.json` và log; không bao giờ báo
"hoàn tất" khi worker thất bại.

**Tiến trình**: thanh tiến trình theo **frame thực sự đã xử lý** / tổng khai báo (video/thư mục có tổng); nếu
không biết tổng thì hiện số frame đã xử lý, không phần trăm giả. **FPS** trên giao diện là *fps vòng lặp*
(kể cả warm-up, vẽ/ghi/hiển thị nếu bật) đọc từ `progress.json` — nhãn ghi rõ, **không** phải fps tracker.

**Đóng cửa sổ khi đang chạy**: hỏi xác nhận → yêu cầu dừng có kiểm soát → chờ tối đa `timeout_dung` (30 s) →
worker vẫn sống thì **terminate** (đường cuối, ghi rõ vào log, trạng thái LỖI). Không để worker chạy ngầm.

## Thư mục một lượt

```
<gốc>/<tên>_<bộ lọc>_<thời điểm>/
    launcher_argv.json     argv đã chạy (mỗi đối số một phần tử)
    progress.json          tiến trình cuối: trang_thai, frame, n_frames_khai_bao, fps_vong_lap_tam (ghi nguyên tử)
    STOP.flag              chỉ có khi đã bấm Dừng
    data/, data_cls/, timing/, pipeline/, frames_manifest.json, run_meta.json   (như track_video_file.py)
    <tên>_tracked.mp4      nếu Lưu video
```

Replay: `run_meta.json` → `replay_argv` (thay `<split>`/`<ten>`) → `subprocess.run(argv, shell=False)`.

## Cơ chế điều phối trong entry point (tùy chọn, không đổi hành vi tracking)

- `--progress-file p.json`: JSON `{trang_thai, frame, n_frames_khai_bao, t_vong_lap_s, fps_vong_lap_tam, pham_vi_fps, …}`,
  ghi **giới hạn** mỗi `--progress-every` (10) frame hoặc ≥ 0,5 s, **nguyên tử** (`.tmp` + replace); trạng thái
  `dang_khoi_tao / dang_chay / hoan_tat / da_dung / loi` (+ `loi`, `out`, `ly_do_dung`).
- `--stop-file f`: kiểm trong `on_frame` **sau** khi vẽ/ghi frame hiện tại → trả `False` → đường dừng đã kiểm chứng.

## Đã kiểm chứng

`src/test_pc_launcher.py` **10/10** — worker là **subprocess thật** chạy `track_video_file.main()` thật qua wrapper
chỉ giả lập detector, `cv2.imread` (có độ trễ) và GUI cv2; `LauncherController` không cần Tk:

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

**Smoke thật** (`handoff/logs/review12/smoke_launcher_that.log`): controller → `track_video_file.py` với **YOLO thật/GPU**
trên 150 frame `MVI_39031`, CV và EKF: DANG_KHOI_TAO → DANG_CHAY → HOAN_TAT, exit 0; cache → replay bằng
`subprocess.run(replay_argv)` → **parity ĐẠT lệch 0** (394/394 và 395/395 hàng). Cửa sổ Tk thật: ảnh
`handoff/anh/launcher_{1..5}_*.png` — điền form, bấm **Bắt đầu** và **Dừng** thật qua `invoke()`, worker YOLO/GPU
thật, dừng ở frame 153/1470 → ĐÃ DỪNG (chụp bằng script điều khiển `after()`, không phải thao tác chuột tay).

**Full suite** (`src/run_all_tests.py`, đã đăng ký `test_pc_launcher`): **31/31 PASS** trên code chốt —
`handoff/logs/review12/tests_20260921-0304.log`.

## Giới hạn

- Chưa smoke bằng tay (chuột/bàn phím thật); thao tác nút đã kiểm qua `invoke()` và controller.
- Chưa có camera trực tiếp, VFR, đóng gói executable, Nano.
- `--show` dùng cửa sổ cv2 của entry point (q/SPACE trong cửa sổ đó); launcher chưa có bộ phát video riêng.
