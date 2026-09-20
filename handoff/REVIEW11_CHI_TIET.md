# Phụ lục vòng 11 — xử lý review `923305d` (hai ca còn lại của PC-F1/PC-F2)

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Hướng dẫn: `handoff/HE_THONG_PC.md`. Vòng trước: `handoff/REVIEW10_CHI_TIET.md`.

## 0. Trạng thái trước khi sửa

`handoff/logs/review11/trang_thai_truoc_khi_sua.log`: HEAD **đúng `923305d`** (0 commit sau), working tree chỉ
có 2 file riêng — không đụng.

## 1. Tái hiện → sửa → kiểm chứng

Fixture: detector, `cv2.imread` (20 ms/frame) và GUI (`namedWindow/imshow/waitKey/destroyAllWindows`) giả lập;
tracker, det cache và hai entry point là **mã thật**, gọi `main()` thật. Log:
`handoff/logs/review11/repro_truoc_khi_sua.log` (exit 0 = tái hiện cả 2) và `repro_sau_khi_sua.log` (exit 1).

| Ca | Tái hiện trước sửa | Thay đổi | Sau sửa |
|:--|:--|:--|:--|
| **1. bấm `q`** | `q` ở frame 2, warm-up 0: frame 2 `frame_total_ms=NaN`, `overhead=NaN`, render/write/show = 0 dù đã hiển thị; `n_frame_tong_ket=2` nhưng chỉ 1 dòng hữu hạn; `wall_ngoai_frame` 28,5 ms chứa frame cuối; `frames_manifest` liệt kê cả 3 ảnh nguồn như "đã dùng" | `run_tracking.track_video`: giữ generator, thân vòng lặp tách ra `_vong_lap_tracker`, **`gen.close()` trong `finally` ngay tại `break`** (không chờ GC; no-op vô hại với `VideoDetections`). Adapter: `try: yield … finally: chốt frame_total + render/write/show` → chạy trên cả resume bình thường lẫn `GeneratorExit`. Không đọc thêm frame, không điền 0. Manifest tách `nguon` (toàn nguồn) / `da_xu_ly` (chỉ số frame thực chạy) + `dung_som` | `q` ở frame 1 và 2: **0 NaN**, `show_ms` frame cuối > 0, `n_frame_tong_ket` == số dòng hữu hạn, Σstage+overhead == `frame_total` (sai số 0), `wall_ngoai_frame` 28,5 → **0,82 ms**, manifest `nguon 3 / da_xu_ly 1|2`, tracking + cache đủ, replay tương đương |
| **2. tên có dấu cách** | nguồn `traffic clip`: live exit 0, cache đúng, nhưng `replay` = `--videos traffic clip` → argparse thấy **hai** video → `FileNotFoundError … traffic.npz` | `replay_argv`: list, **mỗi đối số một phần tử**, placeholder `<split>`/`<ten>` là hai phần tử riêng; `replay` (chuỗi sao chép) quote theo **nền tảng lúc chạy** — Windows `subprocess.list2cmdline`, POSIX `shlex.join` — ghi `replay_quote`; hướng dẫn: thay placeholder trong `replay_argv` rồi `subprocess.run(argv, shell=False)`, **không** `.split()`. Test helper bỏ `.split()` | nguồn `traffic clip` + YAML `cfg dir/ekf ctrv.yaml`: `replay_argv` giữ cả hai là một phần tử; chuỗi `replay` có `"traffic clip"`; replay **từ `replay_argv`** exit 0, parity ĐẠT; ca `--allow-truncated` cũng vậy |

`test_track_video_file` **27/27** (thêm Q1, Q2, S1, S2; F4f cập nhật schema manifest). Runner sau khi sửa:
`test_run_tracking_replay` 10/10, `test_bytetrack_np_parity` 10/10 (bit-identical cache thật).

## 2. Smoke thật trên PC (`handoff/logs/review11/smoke_that_pc.log`)

YOLO thật, GPU, **cửa sổ `--show` thật** (không bấm phím), `--save-video`, `--save-cache`, 60 frame `MVI_39031`:
`show_ms` 1,47 ms/frame được ghi; replay bằng **`subprocess.run(replay_argv, shell=False)` thật** (không
in-process) → exit 0, **95/95 hàng, parity ĐẠT lệch 0,000e+00**. Ca bấm `q` thật trên GUI chưa smoke tay
(chỉ GUI giả lập trong test) — phạm vi giả lập ghi rõ.

## 3. Tài liệu (mục 7 của review)

| Điểm | Xử lý |
|:--|:--|
| Docstring `nano_parity` nói conf tham gia ứng viên | Sửa: ứng viên **chỉ hình học**; conf kiểm riêng ở mức (a) |
| `wall_ghi_s` | Mô tả đúng mốc: MOT/CSV/pipeline/manifest/cache, dừng **trước** khi ghi `run_meta.json` (file chứa chính số đo) |
| "403 ms giải thích phần lớn chênh 5–6 s" | Bỏ. Ghi thay: chênh cũ **nhất quán với** decode chưa đo (≈3,6 ms × 1470 ≈ 5,3 s) — đối chiếu số học sau sửa, không phải đo trực tiếp bản cũ |
| "chênh detector là tải máy" | Sửa thành **quan sát**, chưa đủ bằng chứng quy nguyên nhân |

## 4. Toàn vẹn

`run_tracking.py` (trong `CODE_FILES`) đổi → `code` fingerprint đổi; bằng chứng không đổi hành vi: hai test
parity/replay ở trên. **Không** chạy lại 4 run `v2`, UA-DETRAC, BDD; lock/kết quả kiểm `git diff` không đổi.
Nano F1/F3 hoãn. Base delta giữ `b623465`; review `923305d` **chưa** đánh dấu PASS.
