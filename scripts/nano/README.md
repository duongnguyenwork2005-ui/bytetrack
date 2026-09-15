# Jetson Nano (bản gốc, Maxwell 128 CUDA cores, 4 GB) — đo và tối ưu

**Trạng thái: CHƯA ĐO TRÊN THIẾT BỊ** (Nano chưa kết nối). Mọi con số FPS/bộ nhớ trong repo là trên PC (GTX 1650) và **không được suy ra** cho Nano.

## 1. Kiểm tra stack thực tế trên thiết bị (chạy đầu tiên)
```bash
bash scripts/nano/00_check_stack.sh      # ghi results/phase2/profile/nano_stack.txt
```
Nano gốc chỉ lên tối đa JetPack 4.6.x: L4T 32.7, Ubuntu 18.04, **CUDA 10.2, cuDNN 8.2, TensorRT 8.2, Python 3.6**. Hệ quả:
- `ultralytics 8.4` (cần Python ≥ 3.8, torch ≥ 1.8 mới) **không chạy native** trên Nano gốc → không chạy `model.track`/`run_tracking.py` trực tiếp.
- Lộ trình: export ONNX trên PC (opset ≤ 13, static shape) → build engine TensorRT FP16 **trên Nano** (`trtexec`) → inference bằng TensorRT Python API (`tensorrt` + `pycuda`, Python 3.6) → NMS numpy → tracker numpy (`ekf_ctrv.py`, `ukf_ctrv.py`, `pf_ctrv.py` chỉ dùng numpy; vòng lặp ByteTrack/SORT hiện phụ thuộc `ultralytics` — cần bản không phụ thuộc torch, **chưa làm**).

## 2. Export ONNX trên PC (kiểm tra khả năng export sớm)
```bash
python src/export_detector.py --weights models/yolov8n.pt --opset 12 --imgsz 640      # + --candidate yolo11n.pt
```
In danh sách op ONNX và đối chiếu bảng op TensorRT 8.2 (`scripts/nano/trt82_ops.txt`); op ngoài bảng → ứng viên bị loại hoặc cần plugin.

## 3. Build engine + benchmark trên Nano
```bash
bash scripts/nano/01_build_engine.sh models/export/yolov8n_640_op12.onnx yolov8n_fp16.engine   # --fp16
bash scripts/nano/02_benchmark_engine.sh yolov8n_fp16.engine                                 # trtexec --iterations
```
`trtexec` in latency GPU compute (ms), throughput (qps), bộ nhớ; ghi kèm `nvpmodel -q`, `jetson_clocks`, nhiệt độ (`tegrastats`) — điều kiện chạy phải được ghi vào JSON kết quả.

## 4. Đo end-to-end trên Nano
Chưa có (cần vòng lặp tracker không phụ thuộc torch). Khi có: đo decode / preprocess / inference / NMS / predict / assoc / render như `src/profile_pipeline.py`, đồng bộ CUDA trước/sau inference, ghi độ phân giải, batch=1, warm-up, số frame bỏ, phiên bản JetPack/TensorRT.

## 4b. Checklist khi có thiết bị (mốc 2, mục 6 — chưa thực hiện)
1. `bash scripts/nano/00_check_stack.sh` → dán `results/phase2/profile/nano_stack.txt` vào báo cáo (JetPack, L4T, CUDA, TensorRT, RAM, swap, `nvpmodel -q`).
2. Chép `models/export/yolo11n_640_op12.onnx` (ứng viên duy nhất qua kiểm tra op TRT 8.2, `results/phase2/export/*.json`) và `yolov8n_640_op12.onnx` (đối chứng) sang Nano; `01_build_engine.sh` FP16 từng file; ghi thời gian build và lỗi (nếu op nào không hỗ trợ → loại ứng viên đó, không thay bằng plugin trong phạm vi đề tài).
3. `02_benchmark_engine.sh` với `--iterations 200 --warmUp 2000` ở `nvpmodel -m 0` + `jetson_clocks`; ghi `tegrastats` song song 60 s (nhiệt độ, RAM). Kết quả là **latency GPU của detector**, chưa phải toàn pipeline.
4. Toàn pipeline (decode → preprocess → TRT → NMS → tracker) chỉ đo được sau khi có vòng lặp tracker không phụ thuộc torch (chưa làm); dùng cùng bộ đếm/đồng bộ như `src/profile_pipeline.py`, 450 frame sau 50 warm-up, video `MVI_39031`, 960×540→640, ghi JSON cùng schema `results/phase2/profile/*.json` để so với PC.
5. Mọi con số Nano phải kèm điều kiện chạy (mode, clocks, nhiệt độ, JetPack/TRT) — **không** dùng số PC (GTX 1650) thay thế.

## 5. Không hạ độ chính xác bộ lọc
FP16 chỉ cho detector (TensorRT). Bộ lọc (P, Q, R, Cholesky của UKF, trọng số PF) giữ float64 numpy. Mọi cấu hình tăng tốc phải chạy lại đánh giá tracking (HOTA/IDF1/IDSW + giữ ID qua che khuất) trước khi kết luận.
