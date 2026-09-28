# Điều kiện đo thời gian toàn luồng trên Jetson Nano — công bố TRƯỚC khi đo (vòng 21)

Chỉ chạy **sau khi** `scripts/nano/05_parity_live.sh` ĐẠT (A: giả lập == `pc-ref`; B: TensorRT live == replay cho CV và EKF).
Nếu parity không đạt: không đo, tìm nguyên nhân trước.

## Đối tượng đo

Luồng thực tế hai tiến trình `src/nano_live.py`: đọc frame → TensorRT FP16 (Python 3.6 hệ thống, bộ đệm `cudart`) →
hậu xử lý NumPy → pipe (cửa sổ credit K = 4) → lọc → tracker (env `nano`, Python 3.8, backend NumPy) → MOT/timing trong bộ nhớ.
Ghi file kết quả nằm **sau** vòng lặp và được đo riêng (`wall_ghi_s`).

| Hạng mục | Giá trị cố định |
|:--|:--|
| Nguồn | `data/smoke/MVI_39031_f0001-0300`: 300 JPG 960×540, byte giống dataset, fps 25 |
| Chế độ | sử dụng thông thường: **không** `--regions` (không mask từ nhãn), **không** render, **không** GT |
| Detector | engine FP16 build trên Nano từ `models/export/yolov8n_640_op12.onnx`; imgsz 640, conf 0,05, IoU 0,7, max_det 3000 |
| Tracker | điều kiện c25: `--conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --track-buffer-sec 1.2 --elapsed-mode steps`; CV = `bytetrack.yaml`, EKF = `configs/bytetrack_ekf_ctrv.yaml` |
| Lịch | **6 lượt**: CV, EKF, EKF, CV, CV, EKF (3 lượt/cấu hình), nghỉ 20 s giữa lượt, mỗi lượt một thư mục mới; **không loại lượt nào**, lượt lỗi được giữ và báo cáo |
| Warm-up | khởi tạo (spawn con + nạp engine + 3 lần suy luận làm nóng) đo riêng `wall_khoi_tao_s`, không vào FPS; tóm tắt stage bỏ **25 frame đầu** |
| Thiết bị | MAXN, governor `schedutil` giữ nguyên; **không** `jetson_clocks`, không đổi `nvpmodel`/nguồn/xung; desktop và tác vụ nền để nguyên như lúc đo (ghi lại) |

## Đại lượng báo cáo (mô tả, không kiểm định)

- `fps_vong_lap` = số frame đã xử lý / `wall_vong_lap_s` (cả 300 frame, kể cả warm-up); `fps_on_dinh` = (N − 25) / (t_xong[N] − t_xong[25]).
- Trung vị theo frame sau warm-up: decode, preprocess, infer (tách h2d/exec/d2h, lệnh đồng bộ nên GPU đã hoàn tất), postprocess,
  chờ credit ở con, truyền qua pipe, chờ đọc ở cha, lọc, `tracker.update()`, xử lý ở cha; trễ mỗi frame (trung vị, p90) từ trước khi
  đọc frame ở con đến khi tracker xong ở cha (đồng hồ `monotonic` chung của máy).
- `wall_khoi_tao_s`, `wall_ghi_s` riêng.
- Theo cấu hình: trung vị và min–max của 3 lượt. Không xây dựng khoảng tin cậy, không kiểm định CV vs EKF.
- Thiết bị: tegrastats **dòng thô** 500 ms (RAM, swap, CPU %/MHz, GR3D, nhiệt), `scaling_cur_freq`, GPU `devfreq` trước/sau nếu
  đọc được, `nvpmodel`, governor; thiếu thì ghi thiếu.

## Không làm

Không cộng latency detector vòng 16 hay benchmark tracker vòng 20 vào số này; không khái quát từ 300 frame ra toàn dataset hay cảnh
khác; không so FPS này với số PC. Một lượt mp4 (bước D2 của `05_parity_live.sh`) chỉ kiểm chức năng đường video, không vào bảng tốc độ.
