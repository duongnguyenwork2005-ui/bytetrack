# Phụ lục vòng 16 — chạy thật trên Jetson Nano: mốc 1 (tracker từ cache, parity) ĐẠT; mốc 2 (TensorRT → tracker) chạy trọn

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Trạng thái: `handoff/NANO_STATUS.md`. Log: `handoff/logs/review16/`. Base review **giữ `a5eec39`**.
`CODEX_REVIEW_7a2feb6.md` vẫn chưa nhận được file — sẽ lưu khi có.

## 0. Thiết bị thật và cách vào

| Mục | Giá trị (đo trên máy, `logs/review16/02_check_stack.log`, `nano_reference/nano_stack.txt`) |
|:--|:--|
| Máy | NVIDIA Jetson Nano Developer Kit 4 GB (p3448-0002, eMMC 16 GB + thẻ), `jetson-desktop`, aarch64, 4 CPU |
| Hệ | Ubuntu 18.04.6, L4T R32.7.5, JetPack **4.6.5**, kernel 4.9.337-tegra, glibc 2.27, gcc 7.5 |
| CUDA/TensorRT | CUDA 10.2.300, cuDNN 8.2.1.32, **TensorRT 8.2.1.9** (+ `python3-libnvinfer` cho Python 3.6), `/usr/src/tensorrt/bin/trtexec`, `nvcc` có |
| Python hệ thống | 3.6.9; numpy 1.19.5, scipy 0.19.1, pandas 0.22.0, **cv2 3.2.0**, torch 1.10.0 (`cuda.is_available()=True`), **không** `lap`, **không** pycuda; pip 9.0.1 |
| Điện/nhiệt | `NV Power Mode: MAXN`; tegrastats lúc rảnh CPU 1224 MHz, ~42 °C; RAM 3955 MB, swap 6 GB (zram) |
| Mạng | IP mới `10.11.59.212` (host key ed25519 khác key đã lưu của `.239`). SSH bằng **khóa riêng** `~/.ssh/id_ed25519_nano` (không mật khẩu; alias `nano`); `sudo` cần mật khẩu → không dùng apt từ phiên này |
| `apt-cache policy python3.8*` | có sẵn 3.8.0-3ubuntu1~18.04.2 (bionic-updates/universe) — **không cài** vì cần sudo |
| Môi trường riêng đã tạo | **Miniforge** aarch64 (user-space `~/miniforge3`, `-b`, không sửa `.bashrc`) → env `nano`: **Python 3.8.20**, numpy 1.24.4, pandas 2.0.3, pyyaml 6.0.2, **lap 0.5.13** (wheel aarch64 từ PyPI, cùng phiên bản PC), không scipy/cv2. Python 3.6/torch hệ thống **nguyên vẹn** |

## 1. Chuỗi lệnh đã chạy (nguyên văn, exit code) — `logs/review16/01…13_*.log`

| # | Lệnh (trên Nano, `~/nano_run/…/nano_transfer`) | Kết quả |
|:--|:--|:--|
| 1 | `scp nano_transfer_latest.zip` (v3) → `sha256sum` | `4dc0530a…` khớp; `unzip` 544 file; `sha256sum -c MANIFEST_SHA256.txt --quiet` → **MANIFEST_OK**, exit 0 |
| 2 | `bash scripts/nano/00_check_stack.sh reference/nano_stack.txt` | exit 0 (bảng mục 0) |
| 3 | `apt-cache policy python3.8 python3.8-venv python3.8-dev` | candidate 3.8.0-3ubuntu1~18.04.2; `sudo -n true` → "a password is required" |
| 4 | Miniforge `-b -p ~/miniforge3`; `conda create -n nano python=3.8 numpy pandas pyyaml`; `pip install -r requirements_nano.txt` | exit 0; lap 0.5.13 wheel |
| 5 | `python3 src/nano_env_check.py` (3.6.9) | **KHÔNG ĐẠT** đúng như thiết kế: Python < 3.8, thiếu lap (exit 1) |
| 6 | `~/miniforge3/envs/nano/bin/python src/nano_env_check.py --json reference/env_check_nano.json` | **ĐẠT (1–4)**: backend np, `torch/ultralytics` không bị nạp; 2 cache đủ (300 frame/845 box; 1470/10.145, 34 frame rỗng); 4 `pc-ref` |
| 7 | `python src/test_nano_parity.py` | **20/20**, exit 0 |
| 8 | `python src/nano_run_cache.py` (v3, rồi lặp lại trên v4) | **ĐẠT**, exit 0 (mục 2) |
| 9 | `bash scripts/nano/03_e2e_nano.sh … yolov8n_fp16.engine nano-trt` (v3) | build engine **ĐẠT** (485 s), `trtexec` PASSED; bước detector **LỖI** `libgomp.so.1: cannot allocate memory in static TLS block` → exit 1 (mục 3) |
| 10 | chẩn đoán (chỉ đo) + sửa tối thiểu `0423616` → gói **v4** `3a592ea0…` → chuyển, manifest OK, chép engine đã build | mục 3 |
| 11 | v4: `nano_run_cache.py` → `03_e2e_nano.sh` | mốc 1 ĐẠT lại; mốc 2 chạy trọn exit 0 (mục 4) |

## 2. Mốc 1 — tracker CV/EKF từ cache, parity với PC: **ĐẠT** (`nano_reference/nano_run_2026…-142213` v3, `…-145512` v4)

| Lượt | replay exit / wall | parity | `max\|Δbbox\|` | `max\|Δconf\|` | ID thô | phân hoạch | LF-normalized sha256 == pc-ref |
|:--|:--|:--|:--|:--|:--|:--|:--|
| NANO-smoke / CV (300 fr) | 0 / 3,2 s | **ĐẠT** 545/545 | 0,0 | 0,0 | khớp | khớp | **bằng** (`e3aa27bb…`) |
| NANO-smoke / EKF | 0 / 3,1 s | **ĐẠT** 545/545 | 0,0 | 0,0 | khớp | khớp | bằng (`f22f615a…`) |
| NANO-full / CV (1470 fr) | 0 / 9,8–10,1 s | **ĐẠT** 5842/5842 | 0,0 | 0,0 | khớp | khớp | bằng (`09419c20…`) |
| NANO-full / EKF | 0 / 11,0–11,1 s | **ĐẠT** 5845/5845 | 0,0 | 0,0 | khớp | khớp | bằng (`63e209df…`) |

Tolerance giữ nguyên (bbox 1e-3, conf 1e-6, ID không tolerance). Script **chỉ ghi `nano-*`**; hash cây `pc-ref-*` khớp bản
đóng gói trước khi chạy và **không đổi sau** (`1_pc_ref_khop_ban_dong_goi=true`, `4_pc_ref_khong_doi_sau_chay=true`).
`bit_identical_data_txt=False` trong JSON **chỉ vì kết thúc dòng**: pandas `to_csv` dùng `os.linesep` (PC CRLF, Nano LF);
sau `tr -d '\r'` cả 4 file **byte-identical** với `pc-ref` (`logs/review16/06_bit_identical_lf.log`). Kết luận: tracker
NumPy trên aarch64 cho **đúng từng số** so với PC x86-64 (numpy 1.24.4 vs 2.4.6, cùng lap 0.5.13). Tốc độ replay (chỉ mô tả,
gồm cả ghi CSV): ~148 frame/s full CV.

## 3. Mốc 2 — chẩn đoán và sửa tối thiểu trước khi chạy detector

- Engine: `trtexec --onnx=models/export/yolov8n_640_op12.onnx --saveEngine=yolov8n_fp16.engine --workspace=1024 --fp16` build
  **trên Nano** 485 s, PASSED; engine sha256 `e01b99cb…`. `trtexec --loadEngine … --iterations=200`: **GPU Compute Time
  mean 68,1 ms, median 67,3 ms, p99 81,4 ms; 14,7 qps** (`results_nano/nano-trt/trtexec_bench.txt`) — latency detector thuần.
- Lỗi 1 (`logs/review16/07`, `08`): `import tensorrt, numpy, cv2` → `libgomp.so.1: cannot allocate memory in static TLS
  block` (lỗi Jetson kinh điển). `import cv2` trước OK; `LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1` OK.
- Lỗi 2 (`09`, `10`): đường bộ đệm GPU bằng **torch** — `import torch` (~1 GB) đẩy máy 4 GB vào **swap 984 MB, lfb 1×4 MB**:
  torch CUDA init 27,9 s, `TRTEngine` init 150 s, execute 121 s → 17 s → 0,37 s → 0,08 s (dao động), D2H tới 48 s.
  Đường **ctypes `libcudart`** (cudaMalloc/cudaMemcpy/execute_v2, không torch, không pycuda): deserialize 20,8 s (lần đầu,
  I/O), execute lần 1 10,5 s (khởi tạo lười), sau đó **67,9–78,5 ms**, H2D 2,7 ms, D2H 1,6 ms — bằng `trtexec`.
- Sửa tối thiểu, commit riêng `0423616` (chỉ `src/nano_detector.py` + `scripts/nano/03_e2e_nano.sh`; **không** đụng tracker):
  `TRTEngine(bo_dem="cudart"|"torch"|"pycuda")` mặc định `cudart`, CLI `--bo-dem`; script `export LD_PRELOAD` cho bước
  detector. PC: `test_nano_detector` 6/6, verify gói ĐẠT. Gói **v4** `nano_transfer_latest.zip` **63.792.855 B**, sha256
  `3a592ea01cb7cad14514084b9a2c793cd6d2aaf76693a1137579b1ea559dfba0` → chuyển, manifest OK; engine build-trên-Nano chép sang.
  Thử trực tiếp `TRTEngine` v4 trên máy: init 5,4 s, lần 1 1,5 s, sau đó **73–77 ms**.

## 4. Mốc 2 — end-to-end trên Nano (v4, `logs/review16/13_v4_moc2_e2e.log`, `results_nano/nano-trt/`): **chạy trọn, exit 0**

| Khâu | Kết quả thật trên Nano |
|:--|:--|
| Detector TRT FP16 (Python 3.6 hệ thống, cudart, `LD_PRELOAD` libgomp), 300 JPG, che vùng bỏ qua như PC | 300 frame, **865 box** (thô, conf 0,05), 0 frame rỗng; wall vòng lặp 32,8 s (**9,2 fps** gồm decode+pre+infer+post+ghi mảng); khởi tạo 7,7 s. Trung bình sau 5 frame warm-up (295 frame): **decode 15,9 ms** (cv2 3.2 đọc JPG 960×540), **pre 13,1 ms** (letterbox+RGB+/255 NumPy), **infer 73,3 ms**, **post 5,2 ms** (decode 84×8400 + NMS NumPy) = **107,6 ms/frame** |
| So detection với cache PC (`.pt` FP32 rect 640×384, GTX 1650) — **mô tả** | mức vào tracker c25: **579/581 ghép (99,66 %)**, IoU tb **0,988** (min 0,908), \|Δconf\| tb **0,0085** max 0,185, trùng lớp 99,3 %, 292/300 frame cùng số box; mức thô: 837/845 ghép, IoU 0,983, \|Δconf\| tb 0,0104 max 0,232 |
| Baseline PC (torch FP32 + letterbox 640×640, vòng 15) để tách phần FP16 | c25: 579/581, IoU 0,996, \|Δconf\| 0,0078 ⇒ letterbox 640×640 gây phần lớn khác biệt về *tập* box; **FP16 thêm** IoU 0,996 → 0,988 và Δconf max 0,122 → 0,185 |
| Tracker CV/EKF từ cache TRT (env 3.8, backend np, `NANO-e2e/nano-trt-{cv,ekf}`) | CV **548 hàng, 12 track** (1,2 s, 244 fps replay); EKF **547 hàng, 12 track**; `run_meta` `status=complete`, `tracker_backend=np` |
| So tracking với `pc-ref` (khác detector → **không** kỳ vọng parity) | CV 548 vs 545, EKF 547 vs 545 hàng → "KHÔNG ĐẠT: số hàng khác" — đúng dự kiến, giống baseline PC (549/546). Không phải lỗi tracker (mốc 1 đã chứng minh tracker đúng từng số) |
| Video minh họa | venv Nano không có cv2 → render **trên PC** từ MOT của Nano: `logs/review16/results_nano/nano-trt-cv_render_tren_PC.mp4` (300 frame, 548 box, 19,7 MB, không stage) |

Phạm vi số đo (không cộng gộp): `trtexec` = GPU compute detector; `detector_timing.json` = pipeline detector (Python 3.6,
tuần tự, chưa tối ưu); tracker replay riêng. **Chưa có** "FPS end-to-end một tiến trình" (kiến trúc hai tiến trình qua file).
Điều kiện: MAXN, `jetson_clocks` chưa bật, ~42–44 °C, desktop đang chạy (RAM dùng ~1,5–1,8 GB), một lượt.

## 5. Đã / chưa

| Mốc | Trạng thái |
|:--|:--|
| Tracker-from-cache trên Nano: import ✔, parity CV/EKF smoke + full ✔ (lệch 0, byte-identical sau chuẩn hóa LF), `pc-ref` bất biến ✔ | **ĐẠT** |
| End-to-end: ảnh → detector TRT FP16 → cache → tracker CV/EKF → MOT/metadata/timing ✔, video ✔ (render PC) | **ĐẠT (chạy trọn)** — hai tiến trình qua det cache |
| Hiệu năng | chỉ số **thật**, phạm vi ghi rõ (mục 4); chưa tối ưu, chưa `jetson_clocks`, chưa đo nhiều lượt |
| Còn thiếu | `CODEX_REVIEW_7a2feb6.md`; cv2 trong env 3.8 (render tại chỗ; `pip install opencv-python-headless` tùy chọn); quyết định về bit-identical flag (chuẩn hóa LF trong `nano_run_cache`) |

Không đổi tracker/cấu hình khóa/GT/kết quả/`v2`; `pc-ref-*` không tạo lại. Commit: `0423616` (sửa detector), sau đó log/tài liệu.
