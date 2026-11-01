# R63 — công cụ đo (chuẩn bị 30/10/2026, chưa chạy trên Nano)

Tất cả chỉ ghi vào thư mục đầu ra truyền vào; không sửa gói v16, `src/`, engine có sẵn. Script chạy trên Nano
tương thích Python 3.6 (đã kiểm `ast.parse(feature_version=(3,6))`) và bash Ubuntu 18.04 (`bash -n`).

| File | Chạy ở | Mục đích |
|:--|:--|:--|
| `kiem_suc_khoe_nano.sh <ra> [giay]` | Nano | Chỉ đọc: L4T/JetPack, gói TensorRT, nvpmodel, xung CPU/GPU, nhiệt, RAM/swap, đĩa, tiến trình, log kernel (panic/softlockup/hung/RCU/I2C) boot hiện tại + trước, pstore, tegrastats. Không sudo (chỉ `sudo -n`), không đổi clocks, không reboot. Ra `tom_tat.json`. |
| `do_trtexec.sh <engine> <ra> [dur] [nghi]` | Nano | trtexec TRT 8.2 trên engine có sẵn, batch 1, shape của engine: A mặc định (có H2D/D2H, gần pipeline), B `--noDataTransfers`, C `--useSpinWait`, D CUDA graph, E profile từng lớp; `--exportTimes`, tegrastats nền, nhiệt sau mỗi lượt. |
| `phan_tich_trtexec.py` | PC/Nano | Đọc log trtexec → JSON (GPU Compute, Latency, H2D/D2H, Enqueue, walltime, top lớp, median/p90 từ exportTimes). Test: `tests/test_phan_tich_trtexec.py` (fixture tự soạn, 5/5 PASS). |
| `bench_engine_pipeline.py` | Nano (Py3.6) | Dùng `nano_detector` của gói truyền vào; xen kẽ pass `dong_bo` (đúng cách đo R62) và `su_kien` (CUDA event quanh H2D/execute_async_v2/D2H trên stream riêng) → tách GPU thực thi engine khỏi thời gian gọi/enqueue/chờ đồng bộ; đo riêng `astype(float32)` và decode_nms (ứng viên A khảo sát 32). Letterbox theo shape engine (chạy được engine chữ nhật mà không sửa gói). |
| `xuat_onnx.py` | PC | Xuất ONNX tĩnh (H,W) cùng thiết lập v16 vào thư mục mới, manifest + so đồ thị với tham chiếu. |
| `build_engine_nano.sh <onnx> <ra>` | Nano | trtexec `--fp16 --workspace=1024 --verbose` như engine v16; chỉ chạy sau khi kế hoạch được duyệt. |

## Tiêu chí "đủ ổn định để đo tải dài" (đề xuất, xem kế hoạch mục 4)

`kiem_suc_khoe_nano.sh` không có dấu hiệu panic/softlockup/hung_task mới trong boot hiện tại và pstore không có
bản ghi mới so với R62; nvpmodel giống R62 (ghi lại, không đổi); nhiệt nhàn rỗi < 50 °C; MemAvailable > 1,5 GB;
swap dùng < 100 MB; không có tiến trình python/trtexec khác. Lượt ngắn 300 ảnh đạt trước rồi mới chạy dài 1.470.
Nếu panic tái diễn: dừng mọi phép đo GPU, giữ log, báo người dùng.

## Đã kiểm trên PC (30/10/2026)

- Xuất lại 640×640 từ `models/yolov8n.pt` (`f59b3d83…`): đồ thị tương đương ONNX v16 `a66a5e03…`
  (262 node, 127 initializer trùng từng byte; chỉ khác metadata `date`). Manifest: `_kiem_xuat/640x640/manifest.json`.
- ONNX ứng viên `../ung_vien_onnx/{384x640,352x576,288x512,256x448}/` (manifest kèm sha256): đầu ra 1×84×5040 /
  4158 / 3024 / 2352; mọi op nằm trong `scripts/nano/trt82_ops.txt`. Chưa build engine, chưa đo. File `.onnx`
  bị git ignore (tái tạo bằng `xuat_onnx.py`), chỉ giữ manifest.
