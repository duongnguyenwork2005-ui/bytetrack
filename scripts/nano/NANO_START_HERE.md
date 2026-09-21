# NANO_START_HERE — gói chuyển sang Jetson Nano (mốc 1: chạy CLI)

Gói này **độc lập** với repository PC: giải nén ở bất kỳ đâu trên Nano, `cd` vào thư mục gói và chạy các lệnh
dưới đây theo **đúng thứ tự**. Mỗi bước phải đạt mới sang bước sau. Mọi đường dẫn trong tài liệu là **tương đối
từ thư mục gói**. Thông tin nguồn (commit, HEAD, phiên bản môi trường PC, hash) ở `PACKAGE_INFO.json`; hash
SHA256 của **từng file** ở `MANIFEST_SHA256.txt`.

> Gói này **chưa được chạy trên Nano**. Việc kiểm trên PC (repo PC: `handoff/logs/nano_transfer/kiem_goi_sach.log`) chỉ xác nhận gói
> **đủ file và đủ phụ thuộc Python** để chạy đường replay/parity trong một thư mục sạch; nó **không** chứng minh
> tương thích Nano (Python/JetPack/aarch64). Không giả định phiên bản Python hay JetPack — bước 1 đo thật.

## 0. Cấu trúc gói

| Thư mục / file | Nội dung | Dùng ở bước |
|:--|:--|:--|
| `NANO_START_HERE.md`, `PACKAGE_INFO.json`, `MANIFEST_SHA256.txt` | tài liệu này; commit/HEAD/môi trường PC/hash đầu vào; hash từng file | 0 |
| `src/` | mã nguồn (không gồm `src/demo/**` — file riêng của tác giả) | 2–6 |
| `src/mot_datasets/` | adapter dataset (chỉ dùng `detrac` cho fps/vùng bỏ qua) | 3 |
| `configs/` | YAML tracker đã khóa: `bytetrack_ekf_ctrv.yaml`, `bytetrack_ukf_ctrv.yaml` (+ cấu hình thí nghiệm, không cần cho mốc này) | 3 |
| `third_party/ultralytics_trackers/` | bản **vendor** vòng lặp ByteTrack của ultralytics 8.4.141 (AGPL-3.0, `LICENSE`, `PROVENANCE.json` ghi sha256) — chạy **không cần torch/ultralytics** | 2–3 |
| `scripts/nano/` | `00_check_stack.sh`, `01_build_engine.sh`, `02_benchmark_engine.sh`, `trt82_ops.txt`, `README.md` | 1, 5 |
| `models/export/yolov8n_640_op12.onnx` + `.export.json` | ONNX export từ **chính** checkpoint `models/yolov8n.pt` của PC (hash checkpoint, hash ONNX, lệnh và tham số export, kết quả đối chiếu trọng số trong `.export.json`) | 5 |
| `data/interim/det_cache/yolov8n-coco_conf0.05_all/{meta.json, MVI_39031.npz, MVI_39031.json}` | **detection cache** PC của sequence `MVI_39031` (1470 frame, YOLOv8n COCO, conf 0,05, mọi lớp, max_det 3000, đã che vùng bỏ qua) + metadata + sidecar | 3–4 |
| `data/processed/trackers/NANO-smoke/pc-ref-cv/`, `pc-ref-ekf/` | **kết quả tham chiếu PC** (backend NumPy) chạy từ chính cache trên: `data/MVI_39031.txt`, `data_cls/`, `timing/`, `run_meta.json` | 4 |
| `reference/` | lệnh tái hiện (`lenh_tai_hien.sh`), đối chiếu PC np ↔ ultralytics (`parity_pc_np_vs_ul_*.json`) | 3–4 |
| `data/smoke/MVI_39031_f0001-0300/` (300 `.jpg`) + `MVI_39031_f0001-0300.mp4` + `MVI_39031_smoke.json` | **đoạn video smoke**: frame 1–300 của `MVI_39031` (UA-DETRAC test, 25 fps, 960×540, 12,0 s); JPG là bản **y nguyên** dữ liệu gốc, mp4 là bản mã hóa lại (**lossy**, chỉ để tiện `--source`) | 6 |
| `data/smoke/MVI_39031/{seqinfo.ini, ignored_regions.txt}` | fps/kích thước và **vùng bỏ qua** đã dùng khi dựng cache (che xám trước detector) | 6 |

**Không có trong gói (cố ý)**: môi trường Conda/venv Windows, dataset đầy đủ, lịch sử kết quả, engine TensorRT
(phải build **trên Nano**), checkpoint `.pt` (Nano không chạy được; hash ghi trong `.export.json`), nhãn GT.

## 1. Kiểm môi trường (chạy đầu tiên, ghi lại nguyên văn)

```bash
bash scripts/nano/00_check_stack.sh                      # JetPack/L4T, CUDA, TensorRT, RAM, nvpmodel
uname -m; cat /etc/nv_tegra_release; python3 -VV
python3 -c "import numpy, scipy, lap, pandas, cv2; print(numpy.__version__, scipy.__version__, lap.__version__, pandas.__version__, cv2.__version__)"
```

Phụ thuộc Python **bắt buộc** cho bước 2–4 (đường tracker từ cache): `numpy`, `scipy`, `lap`, `pandas`, `pyyaml`
(bản PC ghi trong `PACKAGE_INFO.json`; **không** yêu cầu cùng phiên bản, nhưng phải import được). Bước 6 thêm
`opencv-python` (đọc ảnh/video). **Không** cần torch/ultralytics cho bước 2–4. Bản vendor dùng
`from __future__ import annotations` ⇒ cần **Python ≥ 3.7**; JetPack 4.6 mặc định Python 3.6 — nếu gặp, dùng
miniforge aarch64 hoặc tương đương (ghi lại cách làm). Thiếu gì → ghi lại, **không** đoán.

## 2. Import backend NumPy (không kéo torch/ultralytics)

```bash
cd <thư mục gói>
BYTETRACK_BACKEND=np python3 -c "import sys; sys.path.insert(0,'src'); import ul_backend; print(ul_backend.fingerprint()); print('torch?', 'torch' in sys.modules, '| ultralytics?', 'ultralytics' in sys.modules)"
```

Đạt khi in được fingerprint và cả hai cờ là `False`. Kiểm tra phép so kết quả có chạy được trên máy:

```bash
python3 src/test_nano_parity.py        # tự tạo fixture, không cần dữ liệu; mong 20/20 PASS
```

## 3. Replay cache (tracker CV và EKF từ detection PC)

Lệnh **y hệt** lệnh đã tạo bản tham chiếu PC, chỉ đổi `--tracker-name` (`reference/lenh_tai_hien.sh`):

```bash
BYTETRACK_BACKEND=np python3 src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name NANO-smoke \
    --tracker-yaml bytetrack.yaml --tracker-name nano-cv --videos MVI_39031 \
    --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps
BYTETRACK_BACKEND=np python3 src/run_tracking.py --cache-id yolov8n-coco_conf0.05_all --split-name NANO-smoke \
    --tracker-yaml configs/bytetrack_ekf_ctrv.yaml --tracker-name nano-ekf --videos MVI_39031 \
    --conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --fps 25 --track-buffer-sec 1.2 --elapsed-mode steps
```

Kết quả: `data/processed/trackers/NANO-smoke/nano-{cv,ekf}/data/MVI_39031.txt` (+ `timing/`, `run_meta.json`).
Cấu hình này = điều kiện **c25** của thí nghiệm (ngưỡng 0,25, lớp COCO 1 2 3 5 7, buffer 1,2 s = 30 frame @25 fps).

## 4. Parity với bản tham chiếu PC — **quyết định có đi tiếp hay không**

```bash
python3 src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-cv  --test data/processed/trackers/NANO-smoke/nano-cv  --json reference/parity_nano_cv.json
python3 src/nano_parity.py --ref data/processed/trackers/NANO-smoke/pc-ref-ekf --test data/processed/trackers/NANO-smoke/nano-ekf --json reference/parity_nano_ekf.json
```

Tolerance (công bố trước, in ở đầu mỗi lần chạy): bbox `|Δ| ≤ 1e-3` px, conf `|Δ| ≤ 1e-6`, **liên kết ID không có
tolerance** (kiểm hai mức: ID thô và phân hoạch). `exit != 0` ⇒ **DỪNG**, gửi lại `parity_nano_*.json` +
`run_meta.json` + kết quả bước 1; không đo tốc độ, không chạy detector khi tracker còn cho kết quả khác PC.
Trên PC, backend NumPy và backend ultralytics cho kết quả **y hệt** (`reference/parity_pc_np_vs_ul_*.json`).

## 5. Detector trên Nano — phần **phải triển khai trên thiết bị** (chưa có mã)

Đã có: ONNX opset 12, static `1×3×640×640`, đầu ra `1×84×8400` (không NMS trong graph), tập op đã đối chiếu
**trên giấy** với bảng TensorRT 8.2 (`scripts/nano/trt82_ops.txt`, `models/export/*.export.json`). Chưa có: engine,
mã suy luận TensorRT, NMS NumPy, và **lớp nối detector vào vòng lặp tracking**.

```bash
bash scripts/nano/01_build_engine.sh models/export/yolov8n_640_op12.onnx yolov8n_fp16.engine   # trtexec --fp16, build TRÊN Nano
bash scripts/nano/02_benchmark_engine.sh yolov8n_fp16.engine                                   # latency GPU của detector (chưa phải pipeline)
```

Điểm nối trong mã hiện tại (đọc trước khi viết): `src/det_cache.py::detect_frame(model, img, spec)` là **một** hàm
detector dùng chung cho cache và live — hiện gọi `model.predict(...)` của ultralytics và trả `(n, 6) float32
[x1, y1, x2, y2, conf, cls]` theo thứ tự NMS; `filter_dets()` lọc (conf_min, classes, max_det) **sau** đó;
`run_tracking.track_video(vd, cfg, ...)` chỉ cần một đối tượng có interface như `VideoDetections` /
`LiveVideoDetections` (`src/track_video_file.py`). Việc cần làm trên thiết bị: (a) preprocess letterbox 640 +
BGR→RGB, chuẩn hóa /255 **giống ultralytics**; (b) TRT infer; (c) decode `84×8400` → xyxy + conf + cls, NMS
(iou 0,7, conf 0,05, max_det 3000 — đúng `DetectorSpec` của cache) bằng NumPy; (d) một adapter tương đương
`LiveVideoDetections` cấp frame cho `track_video`. Kết quả detector FP16/TRT **sẽ khác** cache PC (FP32 CUDA) —
đó là điều **cần đo** ở bước 6, không phải lỗi; parity tracker (bước 4) mới là điều kiện phải bằng.

## 6. Chạy đoạn video smoke bằng detector + ByteTrack CV/EKF

Sau khi có (5): chạy trên `data/smoke/MVI_39031_f0001-0300/` (JPG y nguyên, 25 fps; thứ tự theo số trong tên
file) **và/hoặc** `data/smoke/MVI_39031_f0001-0300.mp4`. Ghi rõ dùng nguồn nào: cache PC được dựng từ **JPG đã
che vùng bỏ qua** (`data/smoke/MVI_39031/ignored_regions.txt`); mp4 là bản nén lại nên detection **không** so
được với cache. Báo cáo: số frame, fps vòng lặp (kèm điều kiện `nvpmodel`, clocks, nhiệt độ), và kết quả tracking
so với `pc-ref-*` **chỉ** ở mức mô tả (khác detector ⇒ không kỳ vọng parity).

## 7. Những gì đã làm trên PC / chưa làm trên thiết bị

| Hạng mục | Trạng thái |
|:--|:--|
| Backend ByteTrack NumPy (không torch) | ✅ chạy thật trên PC, bit-identical với backend ultralytics (`reference/parity_pc_np_vs_ul_*.json`) |
| Cache + tham chiếu CV/EKF cho `MVI_39031` | ✅ trong gói |
| Bộ so parity + test | ✅ `src/nano_parity.py`, `src/test_nano_parity.py` 20/20 trên PC |
| ONNX từ checkpoint đang dùng | ✅ đã đối chiếu trọng số với bản re-export (`.export.json`) |
| Engine TensorRT | ❌ phải build trên Nano |
| Suy luận TRT + NMS NumPy + adapter vào vòng lặp | ❌ **chưa có mã** — mục 5 |
| Chạy video thật trên Nano | ❌ chưa |
| Số đo tốc độ Nano | ❌ chưa có; mọi FPS trong gói là PC |
