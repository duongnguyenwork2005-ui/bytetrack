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

## 5. Không hạ độ chính xác bộ lọc
FP16 chỉ cho detector (TensorRT). Bộ lọc (P, Q, R, Cholesky của UKF, trọng số PF) giữ float64 numpy. Mọi cấu hình tăng tốc phải chạy lại đánh giá tracking (HOTA/IDF1/IDSW + giữ ID qua che khuất) trước khi kết luận.
