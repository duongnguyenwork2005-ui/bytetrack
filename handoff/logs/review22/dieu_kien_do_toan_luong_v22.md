# Điều kiện đo lại thời gian toàn luồng trên Jetson Nano — công bố TRƯỚC khi đo (vòng 22, sau sửa N21-F1)

Giữ **nguyên** điều kiện vòng 21 (`handoff/logs/review21/dieu_kien_do_toan_luong.md`, `42dff0c`); chỉ khác **ranh giới timer** theo
review `9f210e2` N21-F1. Số vòng 21 giữ làm lịch sử, không bị thay thế.

Chỉ chạy **sau khi** `scripts/nano/05_parity_live.sh` trên gói mới ĐẠT (giả lập == `pc-ref`; TensorRT live == replay CV/EKF; biên).
Nếu Nano không truy cập được hoặc nguồn điện không ổn định trong lúc đo: dừng, giữ mọi lượt đã có, báo phần bị chặn.

| Hạng mục | Giá trị cố định (như vòng 21) |
|:--|:--|
| Nguồn | `data/smoke/MVI_39031_f0001-0300`: 300 JPG 960×540, fps 25 |
| Chế độ | không `--regions`, không render, không GT |
| Detector | engine FP16 đã build trên Nano `~/nano_run/engine/yolov8n_fp16.engine` (sha256 `c32690cd…`), cudart; imgsz 640, conf 0,05, IoU 0,7, max_det 3000 |
| Tracker | c25: `--conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --track-buffer-sec 1.2 --elapsed-mode steps`; CV `bytetrack.yaml`, EKF `configs/bytetrack_ekf_ctrv.yaml` |
| Luồng | hai tiến trình, cửa sổ credit K = 4 |
| Lịch | 6 lượt: CV, EKF, EKF, CV, CV, EKF; nghỉ 20 s; mỗi lượt một thư mục mới; không loại lượt nào |
| Warm-up | tóm tắt stage và `fps_on_dinh` bỏ 25 frame đầu |
| Thiết bị | MAXN, `schedutil` giữ nguyên; không `jetson_clocks`, không đổi nguồn/xung |

## Ranh giới thời gian (giao thức 2)

Con gửi HELLO sau khi khởi tạo xong rồi **chờ lệnh BẮT ĐẦU**; cha chuẩn bị cấu hình/telemetry, **bật timer rồi mới gửi BẮT ĐẦU**;
con chỉ đọc frame 1 sau khi nhận lệnh. Báo cáo, cho từng lượt:

- `khoi_tao_s` (spawn → HELLO), `chuan_bi_cha_s` (HELLO → trước khi gửi BẮT ĐẦU) — **không** vào FPS.
- `xu_ly_s` = trước khi gửi BẮT ĐẦU → `track_video` trả về (frame 300 đã qua tracker, nhận EOS); `fps_xu_ly` = 300 / `xu_ly_s`.
- `ket_thuc_s` (sau vòng lặp → con thoát), `ghi_s` (ghi kết quả) — riêng.
- `fps_on_dinh` = (N − 25) / (t_xong[N] − t_xong[25]), theo mốc xong tracker ở cha.
- `kiem_cua_so_do.moi_frame_trong_cua_so` phải là `true` (mốc `monotonic`: frame 1 bắt đầu đọc ≥ lúc gửi BẮT ĐẦU, frame cuối xong ≤ hết cửa sổ).
- Trung vị stage như vòng 21; trễ/frame; tegrastats dòng thô, xung/nhiệt/RAM/swap nếu đọc được.

Mô tả, không kiểm định; không kết luận EKF nhanh/chậm hơn CV từ chênh lệch nhỏ giữa 3 lượt; không cộng số vòng 16/20; không so với
số vòng 21 như thể cùng phạm vi timer (vòng 21 thiếu phần đầu của frame đầu tiên — xem đính chính trong phụ lục vòng 22).
