# REVIEW_BRIEF — vòng 16: chạy thật trên Jetson Nano — mốc 1 ĐẠT (parity lệch 0), mốc 2 chạy trọn (TensorRT FP16 → tracker)

<!-- BASE_TARGET -->

Base review **giữ `a5eec39`**. `CODEX_REVIEW_7a2feb6.md` chưa nhận được. Không tuning/train; tracker, cấu hình khóa, GT,
kết quả UA-DETRAC/BDD, 4 run `v2` **không đổi**; `pc-ref-*` **không tạo lại**. Chi tiết: **`handoff/REVIEW16_CHI_TIET.md`**.

## Thiết bị (đo thật, `logs/review16/`)

Jetson Nano 4 GB, JetPack **4.6.5** / L4T 32.7.5, TensorRT 8.2.1.9 (+binding Py 3.6), CUDA 10.2, `trtexec`, torch 1.10 CUDA,
cv2 3.2.0, không `lap`. SSH bằng khóa riêng (IP `10.11.59.212`); `sudo` cần mật khẩu → không apt. Môi trường riêng
**Miniforge** user-space → env `nano` **Python 3.8.20**, numpy 1.24.4, pandas 2.0.3, pyyaml 6.0.2, **lap 0.5.13** (wheel);
Python 3.6/torch hệ thống nguyên vẹn. Gói v3 `4dc0530a…` → manifest OK; `nano_env_check` ĐẠT (1–4); `test_nano_parity` 20/20.

## Mốc 1 — tracker CV/EKF từ cache: **ĐẠT trên Nano**

`python src/nano_run_cache.py` (chỉ ghi `nano-*`, chặn torch/ultralytics): smoke 545/545 ×2 và full **5842 / 5845** parity
**ĐẠT**, `max|Δbbox| = max|Δconf| = 0,0`, ID thô + phân hoạch khớp; hash `pc-ref-*` khớp bản đóng gói và **không đổi sau
chạy**. Sau chuẩn hóa CRLF→LF (pandas `os.linesep`), cả 4 output **byte-identical** với `pc-ref`. Lặp lại trên gói v4: ĐẠT.

## Mốc 2 — TensorRT → tracker: chạy trọn

- Engine FP16 build **trên Nano** (485 s); `trtexec`: GPU compute **68,1 ms** mean (14,7 qps).
- Chặn 1: `libgomp static TLS` khi import cv2 sau tensorrt → `LD_PRELOAD` libgomp. Chặn 2: bộ đệm GPU bằng torch → swap 984 MB,
  execute 80 ms … >100 s. Đường **ctypes libcudart** ổn định **68–78 ms**. Sửa tối thiểu `0423616` (detector + script; không
  đụng tracker; `test_nano_detector` 6/6) → gói **v4** `handoff/nano_transfer_latest.zip` **63.792.855 B** sha256
  `3a592ea01cb7cad14514084b9a2c793cd6d2aaf76693a1137579b1ea559dfba0`, manifest OK trên Nano.
- `03_e2e_nano.sh` (v4) exit 0: detector TRT FP16 300 frame, 865 box; **decode 15,9 / pre 13,1 / infer 73,3 / post 5,2 =
  107,6 ms/frame** (9,2 fps vòng detector, Python 3.6 tuần tự, MAXN, chưa `jetson_clocks`). So cache PC mức vào tracker:
  **579/581 ghép**, IoU 0,988, |Δconf| 0,0085 (baseline PC FP32 letterbox-640: IoU 0,996 → FP16 thêm ≈0,008). Tracker CV
  548 hàng/12 track, EKF 547 — khác `pc-ref` 545 như dự kiến (khác detector; mốc 1 chứng minh tracker đúng). Video render
  trên PC từ MOT Nano (venv thiếu cv2).

## Còn thiếu / giới hạn

`CODEX_REVIEW_7a2feb6.md`; cv2 trong env 3.8 (render tại chỗ); flag `bit_identical` nên chuẩn hóa LF; số đo một lượt, chưa
tối ưu, chưa gộp thành FPS e2e (hai tiến trình qua file).

## Commit / bước tiếp theo

`0423616` sửa detector; sau đó log/tài liệu. Bước tiếp theo: tối ưu nút thắt thật (infer 73 ms FP16 trên Maxwell là trần
của yolov8n 640; decode+pre 29 ms có thể giảm bằng cv2 4.x/`jetson_clocks`), đo AB/BA CV↔EKF từ cùng cache trên Nano
(`nano_bench_tracker.py`), ghép hai tiến trình nếu cần live.
