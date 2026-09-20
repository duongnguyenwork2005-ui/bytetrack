# REVIEW_BRIEF — vòng 11: hai ca còn lại của review `923305d` (PC-F1/PC-F2)

<!-- BASE_TARGET -->

Base delta **giữ `b623465`**; review `923305d` **chưa** đánh dấu PASS. Nano F1/F3 hoãn. Không tuning/train;
4 run `v2`, kết quả UA-DETRAC/BDD và lock **không đổi** (`git diff`). Chi tiết: **`handoff/REVIEW11_CHI_TIET.md`**.

## Tái hiện → sửa → kiểm chứng

Cả hai tái hiện bằng fixture (detector/`imread`/GUI giả lập; tracker, cache, hai entry point **thật**) đúng như
Codex mô tả — `handoff/logs/review11/repro_truoc_khi_sua.log`; sau sửa hết tái hiện — `repro_sau_khi_sua.log`.

| Ca | Trước | Sửa | Sau |
|:--|:--|:--|:--|
| **1. bấm `q`** | frame cuối `frame_total=NaN`, show/render = 0 dù đã hiển thị; `n_frame_tong_ket` lệch số dòng hữu hạn; manifest gọi cả 3 ảnh là "đã dùng" | `track_video` giữ generator và **`close()` trong `finally` ngay tại `break`** (không chờ GC); adapter chốt timing trong `finally` quanh `yield`; không đọc thêm frame, không điền 0; manifest tách `nguon` / `da_xu_ly` | `q` ở frame 1 và 2: **0 NaN**, `show_ms` frame cuối > 0, `n_frame_tong_ket` == số dòng, Σstage+overhead == `frame_total` (sai số 0), `wall_ngoai_frame` 28,5 → 0,82 ms, tracking/cache đủ, replay tương đương |
| **2. tên có dấu cách** | `--videos traffic clip` → hai video → replay `FileNotFoundError` | `replay_argv` (mỗi đối số một phần tử, placeholder riêng); `replay` quote theo nền tảng (`replay_quote`: Windows `list2cmdline` / POSIX `shlex`); test bỏ `.split()` | `traffic clip` + YAML `cfg dir/ekf ctrv.yaml`: replay **từ `replay_argv`** exit 0, parity ĐẠT; cả ca `--allow-truncated` |

`test_track_video_file` **27/27** (+Q1, Q2, S1, S2). Runner đổi (`run_tracking.py`): `test_run_tracking_replay`
10/10, `test_bytetrack_np_parity` 10/10 bit-identical.

## Smoke thật trên PC

YOLO + GPU, **cửa sổ `--show` thật**, `--save-video`, `--save-cache`, 60 frame: `show_ms` ghi 1,47 ms/frame;
replay bằng **`subprocess.run(replay_argv, shell=False)` thật** → exit 0, 95/95 hàng, parity **ĐẠT lệch 0**
(`handoff/logs/review11/smoke_that_pc.log`). Bấm `q` thật trên GUI chưa smoke tay — chỉ GUI giả lập trong test.

## Tài liệu

Docstring parity: ứng viên **chỉ hình học**, conf kiểm riêng. `wall_ghi_s`: dừng **trước** khi ghi `run_meta.json`.
Bỏ "403 ms giải thích phần lớn chênh cũ" → chênh 5–6 s **nhất quán với** decode chưa đo (≈3,6 ms × 1470),
là đối chiếu số học, không phải đo bản cũ. "Chênh detector là tải máy" → **quan sát**, chưa quy nguyên nhân.

## Kiểm thử

Full suite trên code chốt (runner đổi): `handoff/logs/review11/` — không commit/sửa source trong lúc chạy.
