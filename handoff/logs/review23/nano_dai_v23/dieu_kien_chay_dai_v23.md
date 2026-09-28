# Điều kiện chạy đầy đủ 1.470 frame trên Jetson Nano — công bố TRƯỚC khi chạy (vòng 23)

Mục tiêu: **kiểm vận hành và độ lặp lại** của luồng trực tiếp đã nghiệm thu (`c4b39fe`) trên toàn sequence, không phải thí nghiệm tối ưu
mới. Tham số giữ nguyên vòng 22; chỉ khác nguồn (1.470 frame thay vì 300).

| Hạng mục | Giá trị cố định |
|:--|:--|
| Mã | gói Nano **v8** `f4da9f08…` ở `~/nano_run/v8` (đóng gói tại `b0c645b`; `src/` == `c4b39fe`, 7 file lõi so hash LF khớp — `logs/review23/01_*`) |
| Nguồn | `~/nano_run/data_full/MVI_39031`: 1.470 JPG gốc UA-DETRAC `img00001…img01470`, 132.708.676 B, SHA256 1470/1470 khớp PC (`logs/review23/02_*`) |
| FPS nguồn | 25 (thời gian `(idx−1)/25`) |
| Detector | engine FP16 `~/nano_run/engine/yolov8n_fp16.engine` sha256 `c32690cdb9f14d5f…` (build vòng 21, không build lại); cudart; imgsz 640, conf 0,05, IoU 0,7, max_det 3000 |
| Tracker | c25: `--conf-min 0.25 --classes 1 2 3 5 7 --max-det 300 --track-buffer-sec 1.2 --elapsed-mode steps`; CV `bytetrack.yaml`, EKF `configs/bytetrack_ekf_ctrv.yaml` |
| Luồng | hai tiến trình, giao thức 2 (lệnh BẮT ĐẦU), cửa sổ credit K = 4; không `--regions`, không render, không GT |
| Warm-up | tóm tắt stage và `fps_on_dinh` bỏ 25 frame đầu; `fps_xu_ly` tính trên cả 1.470 frame trong cửa sổ BẮT ĐẦU → `track_video` trả về |
| Lịch | **4 lượt: CV → EKF → EKF → CV** (hai cặp AB/BA), nghỉ 20 s giữa lượt, mỗi lượt thư mục mới, `--save-cache` riêng |
| Sau mỗi lượt | replay bằng chính `replay_argv` → `nano_parity` (bbox 1e-3, conf 1e-6, ID toàn sequence, không nới) → so byte MOT; tiến trình con đã thoát?; MemAvailable/swap/nhiệt/ext4 trước–sau |
| Thiết bị | MAXN, `schedutil` giữ nguyên; không `jetson_clocks`, không đổi nguồn/xung, không `apt` |
| Driver | `handoff/logs/review23/chay_dai_1470.sh` (chỉ gọi entry point đã nghiệm thu) |

Giữ **mọi** lượt kể cả thất bại; không loại lượt chậm. Nếu Nano khởi động lại, mất kết nối, hoặc ext4 báo lỗi: dừng phần chạy dài, lưu
bằng chứng, không tự quy nguyên nhân.

Báo cáo riêng: FPS toàn luồng (`fps_xu_ly`, `fps_on_dinh`), trễ từng frame, `tracker.update()`; mô tả, không kiểm định; không kết luận
EKF nhanh/chậm hơn CV từ 2 lượt/cấu hình; không suy ra HOTA/IDF1. Kiểm frame 1–1.470 xử lý đúng một lần (kể cả frame không detection)
và mọi frame đếm nằm trong cửa sổ đo; tính lại từ CSV gốc trên PC.
