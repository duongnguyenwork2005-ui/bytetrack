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
2. Chép `models/export/yolo11n_640_op12.onnx` và `yolov8n_640_op12.onnx` (đối chứng) sang Nano — **cả hai** qua kiểm tra op TRT 8.2, chỉ `yolo26n` bị loại vì op `Mod` (`results/phase2/export/*.json`); `01_build_engine.sh` FP16 từng file; ghi thời gian build và lỗi (nếu op nào không hỗ trợ → loại ứng viên đó, không thay bằng plugin trong phạm vi đề tài).
3. `02_benchmark_engine.sh` với `--iterations 200 --warmUp 2000` ở `nvpmodel -m 0` + `jetson_clocks`; ghi `tegrastats` song song 60 s (nhiệt độ, RAM). Kết quả là **latency GPU của detector**, chưa phải toàn pipeline.
4. Toàn pipeline (decode → preprocess → TRT → NMS → tracker) chỉ đo được sau khi có vòng lặp tracker không phụ thuộc torch (chưa làm); dùng cùng bộ đếm/đồng bộ như `src/profile_pipeline.py`, 450 frame sau 50 warm-up, video `MVI_39031`, 960×540→640, ghi JSON cùng schema `results/phase2/profile/*.json` để so với PC.
5. Mọi con số Nano phải kèm điều kiện chạy (mode, clocks, nhiệt độ, JetPack/TRT) — **không** dùng số PC (GTX 1650) thay thế.

## 4c. Giao thức đo **CV so với EKF trong cùng điều kiện** (bổ sung mốc 3 — chưa thực hiện)

**Trạng thái: CHƯA ĐO. Nano chưa kết nối.** Mục này chỉ định nghĩa trước cách đo, để khi có thiết bị thì
không phải quyết định giữa chừng.

Quan sát trên PC (xem `handoff/MOC2_FINAL_SUMMARY.md` mục 6b): chênh lệch giữa CV và EKF ở phần tracker là
**~0,24 ms/frame**, trong khi `detector_wall` **khác nhau 2,2 ms giữa bốn lượt đã chạy** — mỗi cấu hình chỉ
một lượt, nên đây là **chênh lệch quan sát được**, chưa phải ước lượng nhiễu đã kiểm chứng. Dù vậy nó **đủ
để cảnh báo**: nếu đo kiểu "chạy CV một lượt, chạy EKF một lượt" thì biến thiên ở phần detector có thể át
hiệu ứng cần đo (trên PC, thứ tự fps end-to-end đã đảo ngược so với chi phí tracker).

Trên Nano, tản nhiệt kém và throttling theo nhiệt độ là **rủi ro cần kiểm**, **không** phải kết luận — chưa
có phép đo nào trên thiết bị. Giao thức dưới đây được thiết kế để phép đo vẫn có nghĩa kể cả khi rủi ro đó
xảy ra:

1. **Cùng detection, không chạy lại detector.** Đo tracker từ **cùng một detection cache** (`det_cache`) cho
   cả CV và EKF. Đây là phép đo *duy nhất* tách được chi phí motion model. Số end-to-end đo riêng và báo cáo
   riêng, không trộn (đúng như tách 6a / 6b của UA-DETRAC).
2. **Một lịch đo duy nhất, cân bằng AB/BA** (không phải "toàn bộ A rồi toàn bộ B", và cũng không phải
   A/B/A/B thuần — thứ tự cố định làm nhiệt độ tích lũy luôn thiệt cho cấu hình chạy sau):

   | khối | thứ tự | số cặp |
   |:--|:--|--:|
   | 1 | A → B | 3 |
   | 2 | B → A | 3 |

   Tổng **6 cặp** (3 mỗi chiều). Mỗi **cặp** cho một hiệu số `EKF − CV`; báo cáo **trung vị + IQR của các
   hiệu số theo cặp**, **không** phải hiệu của hai trung bình.
3. **Lưu MỌI lượt, kể cả lượt bị throttling.** Ghi điều kiện chạy ở mỗi lượt: `nvpmodel -q`,
   `jetson_clocks` bật/tắt, nhiệt độ và tần số từ `tegrastats` (lấy mẫu song song), RAM/swap. **Không loại
   lượt nào khỏi dữ liệu thô.**
   - **Số chính thức là hiệu năng chạy LIÊN TỤC** — tính trên toàn bộ các lượt, vì đó mới là điều kiện một
     thiết bị Nano thật gặp phải khi chạy dài. Throttling là **đặc tính của thiết bị**, không phải nhiễu đo.
   - Bảng "không throttling" chỉ là **phân tích bổ sung**, kèm **tiêu chí loại công bố TRƯỚC khi đo**
     (ví dụ: loại lượt có `tegrastats` báo nhiệt > X °C hoặc tần số GPU < Y MHz ở bất kỳ mẫu nào).
   - **Loại theo CẶP, không theo lượt đơn**: nếu một lượt trong cặp bị loại thì **loại cả cặp**. Hiệu số chỉ
     có nghĩa khi cả hai nửa của cặp còn nguyên; bỏ một nửa sẽ làm lệch hiệu số ghép cặp.
   - Báo cáo **cả hai** bảng và ghi rõ đã loại bao nhiêu **cặp**, theo tiêu chí nào.
4. **Cân bằng thứ tự**: chạy xen kẽ cả **A/B** lẫn **B/A** (số cặp mỗi chiều bằng nhau) để nhiệt độ tích lũy
   không dồn lợi thế cho một bộ lọc. Mọi lượt dùng **cùng warm-up** (bỏ 50 frame đầu, đo 450 frame tiếp
   theo, batch 1, cùng độ phân giải) và **cùng điều kiện bắt đầu** (chờ nhiệt độ về ngưỡng đã định trước khi
   bắt đầu lượt kế tiếp, ghi lại nhiệt độ đó).
5. **Đọc đúng ba cột timer** — chúng **KHÔNG** phải ba pha độc lập (xem `src/run_tracking.py`:177–183):
   `update_ms` là **tổng** lời gọi `tracker.update()`; `predict_ms` chỉ là `multi_predict`; `assoc_ms =
   update − predict` nên **bao gồm cả bước cập nhật (correction) của bộ lọc**, quản lý vòng đời và loại
   track trùng. Vì vậy **không** kỳ vọng `assoc` của CV và EKF bằng nhau — phần cập nhật của hai bộ lọc
   khác nhau thật. Đại lượng để so là **`update_ms`** (tổng chi phí tracker); `predict_ms` dùng để biết phần
   chênh lệch nằm ở bước dự đoán hay bước cập nhật.
6. **Đối chứng độ chính xác**: mọi cấu hình tăng tốc (FP16 detector, engine khác) phải chạy lại đánh giá
   tracking trên PC trước khi so tốc độ — số tốc độ không đi kèm số chất lượng thì không kết luận được gì.

**Backend không phụ thuộc torch — ĐÃ CÓ trên PC, CHƯA XÁC MINH trên thiết bị** (2026-09-18):

```bash
BYTETRACK_BACKEND=np python <script chạy tracker từ cache>     # CV + EKF, không import torch/ultralytics
```

- `third_party/ultralytics_trackers/` — bản **vendor** 5 module tracker của ultralytics 8.4.141 (AGPL-3.0,
  `PROVENANCE.json` ghi sha256 upstream/vendor), chỉ đổi **5 dòng import** sang `np_compat.py`.
- `src/ul_backend.py` chọn nguồn lớp nền; **một** bản mã CTRV/CV chạy trên cả hai backend.
- **Dependencies thực tế**: `numpy`, `scipy`, `lap` (bản PC: 2.4.6 / 1.17.1 / 0.5.13). Không cần torch,
  không cần Ultralytics.
- Kiểm chứng trên PC: `src/test_bytetrack_np_parity.py` **8/8** — CV **17.348 hàng** và EKF **17.333 hàng**
  **bit-identical** với backend hiện tại trên 3 sequence dev, chạy trong tiến trình **chặn** `torch` và
  `ultralytics`.

⚠️ **Parity trên PC KHÔNG chứng minh tương thích hay tốc độ Nano.** Rủi ro đã biết, **chưa kiểm**: bản
vendor dùng `from __future__ import annotations` (cần **Python ≥ 3.7**) trong khi JetPack 4.6 chỉ có
**Python 3.6** ⇒ cần cài Python mới hơn trên Nano (miniforge aarch64 hoặc tương đương), và `scipy`/`lap`
phải build được cho aarch64. Chưa bước nào trong số này được chạy trên thiết bị.

**Còn thiếu để đo end-to-end**: đường detector trên Nano (TensorRT engine + NMS numpy) chưa nối vào backend
này — mục 4 vẫn chưa làm. Backend mới cho phép đo **phần tracker** (CV ↔ EKF) từ detection cache, đúng
phép đo mà mục 4c cần.

**Tóm tắt trạng thái chuẩn bị Nano**

| Bước | Trạng thái |
|:--|:--|
| Export ONNX trên PC (opset 12, static shape) | ✅ `models/export/{yolov8n,yolo11n,yolo26n}_640_op12.onnx` + `results/phase2/export/*.json` |
| Đối chiếu op với bảng TensorRT 8.2 | ✅ `scripts/nano/trt82_ops.txt`; **2/3 qua**: `yolov8n`, `yolo11n` (verdict *"cần kiểm tra bằng trtexec trên Nano"*). `yolo26n` **bị loại**: có op `Mod` ngoài bảng TRT 8.2 |
| Script kiểm stack / build engine / benchmark | ✅ `00_check_stack.sh`, `01_build_engine.sh`, `02_benchmark_engine.sh` |
| Giao thức đo CV ↔ EKF | ✅ mục 4c này |
| Bộ so kết quả Nano ↔ PC | ✅ `src/nano_parity.py` — tolerance công bố trước, liên kết ID **không** có tolerance; `src/test_nano_parity.py` **10/10** |
| Khung đo AB/BA theo mục 4c | ✅ `src/nano_bench_tracker.py` — 6 cặp cân bằng, warm-up, `tegrastats`, lưu **mọi** lượt, báo trung vị + IQR của hiệu số **theo cặp** |
| Vòng lặp ByteTrack không phụ thuộc torch | ✅ **đã có** (`BYTETRACK_BACKEND=np`), parity bit-identical trên PC — **chưa chạy trên thiết bị** |
| Python ≥ 3.7 trên Nano (bản vendor yêu cầu) | ❓ **chưa xác minh** — JetPack 4.6 mặc định Python 3.6 |
| `scipy` / `lap` / **`pandas`** cho aarch64 | ❓ **chưa xác minh** — `src/run_tracking.py` có import `pandas` để ghi `timing/*.csv`, nên đây là phụ thuộc **bắt buộc** của đường tracker-từ-cache |
| Đường detector trên Nano nối vào backend mới | ❌ **chưa làm** |
| Thiết bị Nano | ❌ **không tiếp cận được** — kiểm lại 2026-09-20: 4 tên không phân giải, ARP không có OUI NVIDIA `00-04-4b`, không cổng serial, không thiết bị PnP Jetson/Tegra/APX. Bằng chứng: `handoff/logs/review8/thu_ket_noi_nano.log` |
| Số đo trên Nano | ❌ **chưa có số nào**; mọi FPS trong repo là PC (GTX 1650) |

### Lệnh thu thập thông tin khi có thiết bị (chạy TRƯỚC mọi phép đo)

```bash
# tren Nano
bash scripts/nano/00_check_stack.sh            # JetPack/L4T, CUDA, TensorRT, RAM, nvpmodel
uname -m && cat /etc/nv_tegra_release           # kien truc (phai la aarch64) + ban L4T
python3 -VV && python3 -c "import sys;print(sys.version_info >= (3,7))"   # bản vendor CAN >= 3.7
python3 -c "import numpy, scipy, lap; print(numpy.__version__, scipy.__version__, lap.__version__)"
free -h && swapon --show && nvpmodel -q && cat /sys/devices/virtual/thermal/thermal_zone*/temp
```

Chỉ tuyên bố "đã chạy trên Nano" khi có **output thật từ thiết bị** dán vào
`results/phase2/profile/nano_stack.txt`.

## 5. Không hạ độ chính xác bộ lọc
FP16 chỉ cho detector (TensorRT). Bộ lọc (P, Q, R, Cholesky của UKF, trọng số PF) giữ float64 numpy. Mọi cấu hình tăng tốc phải chạy lại đánh giá tracking (HOTA/IDF1/IDSW + giữ ID qua che khuất) trước khi kết luận.
