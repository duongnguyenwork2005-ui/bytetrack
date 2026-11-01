# R64 — RoundaboutHD prefix3000 (4 camera × 3.000 frame): đánh giá hệ thống 3 detector × 4 bộ lọc (02/11/2026) — XONG 36/36

**Giao thức** (GPT 01/11 Q2; sửa theo review smoke 02/11, B1):
- Dữ liệu: `imagesc001`–`imagesc004`, mỗi camera frame 1–3000 liên tục, tổng 12.000 frame 3840×2160, 15 fps. Detector
  đọc trực tiếp `video.mp4`, không ghi lại JPG. Tập con được **định trước**, không chọn camera theo kết quả.
- GT: `SCT/SCT_GT.txt` (xyxy → xywh), giữ ID trong từng camera, gộp class 1. Nguồn có lỗi: cùng (frame, id) cho hai box
  khác nhau. Trong prefix lỗi này chỉ có ở cam4: 8 cặp, 16 dòng, ID 105, frame 2536–2543. Theo B1, bỏ cả 16 dòng (không
  bỏ cả track, không đổi ID, không tạo vùng ignore); danh sách dòng bị bỏ nằm trong
  `data/processed/RoundaboutHD-prefix3000/gt_log.json`. Kiểm bằng tracker oracle: TrackEval cho cam4 HOTA 1,0.
- **Giới hạn nhãn:** SCT_GT không gán nhãn xe đỗ trong bãi. Mọi dự đoán không ghép được GT đều chịu FP, kể cả xe đỗ thật
  ⇒ đây là đánh giá **theo phạm vi SCT**, không phải benchmark toàn bộ RoundaboutHD. Không dùng điểm này để xếp hạng giữa
  các dataset, và không diễn giải mọi FP là nhận nhầm vật thể.
- Detector, cache và tracker giống SVMOT, chỉ khác fps: 15 fps, buffer 1,2 s = 18 bước (`max_frames_lost` thực tế 18),
  evaluator `mot`. Cache `r64_roundabout_<det>_c0.05`: 12/12 đơn vị qua cổng. Đơn vị yolov8n_coco cam1–2 được sinh sớm
  (sai lệch Q4, ghi trong `smoke/00_smoke.md`) và có biên bản xác minh bổ sung.
- Thống kê chỉ mô tả, không CI (4 camera của cùng một nút giao, chưa chứng minh độc lập). Bảng: `roundabout/*.csv`.
  Log: `../07_roundabout_cache.log`, `../08_*`, `../09_*`, `roundabout/track_logs/`.

## COMBINED (điểm thang 100; PF500 = trung bình 3 seed)

| Detector | Ngưỡng | HOTA CV / EKF / UKF / PF500 | IDF1 CV / EKF / UKF / PF500 | IDSW CV / EKF / UKF / PF500 | DetA CV / PF500 | MOTA (CV) |
|:--|:--|:--|:--|:--|:--|--:|
| yolov8n_coco | c25 | 38,44 / 38,07 / 38,05 / 26,79 | 51,59 / 51,00 / 51,03 / 32,70 | 388 / 402 / 409 / 1031,3 | 34,03 / 30,23 | 28,98 |
| yolov8n_coco | c10 | 38,71 / 38,31 / 38,38 / 26,77 | 50,24 / 49,74 / 49,93 / 31,64 | 338 / 336 / 341 / 982,3 | 33,14 / 29,12 | 19,99 |
| yolov8n_mixed5 | c25 | 42,83 / 42,48 / 42,44 / 31,27 | 55,33 / 54,57 / 54,59 / 37,18 | 471 / 477 / 494 / 1047,7 | 41,85 / 37,64 | 43,53 |
| yolov8n_mixed5 | c10 | 43,79 / 43,22 / 43,21 / 31,44 | 55,49 / 54,50 / 54,60 / 36,47 | 431 / 445 / 455 / 1038,3 | 42,25 / 37,98 | 40,99 |
| yolo11n_mixed5 | c25 | 44,30 / 43,96 / 43,91 / 32,13 | 57,67 / 57,04 / 57,00 / 38,72 | 358 / 393 / 376 / 984,7 | 41,39 / 36,71 | 41,66 |
| yolo11n_mixed5 | c10 | 44,23 / 44,06 / 44,02 / 32,14 | 55,89 / 55,71 / 55,72 / 37,89 | 327 / 367 / 356 / 935,7 | 40,30 / 35,78 | 34,80 |

PF500 từng seed: HOTA trong khoảng 26,5–32,5, chênh giữa các seed ≤ 0,5 điểm ⇒ mức thấp của PF500 **không phải do một
seed**.

## HOTA từng camera (c25)

| Detector | cam1 CV/EKF/UKF/PF | cam2 | cam3 | cam4 |
|:--|:--|:--|:--|:--|
| yolov8n_coco | 26,9 / 26,6 / 26,5 / 18,1 | 39,8 / 39,1 / 39,1 / 26,9 | 45,9 / 46,1 / 45,9 / 31,7 | 43,6 / 42,8 / 43,0 / 31,9 |
| yolov8n_mixed5 | 34,0 / 33,7 / 33,7 / 24,0 | 44,7 / 43,4 / 43,4 / 30,9 | 38,3 / 38,1 / 38,0 / 27,6 | 51,7 / 51,9 / 51,9 / 39,5 |
| yolo11n_mixed5 | 32,3 / 32,0 / 32,1 / 22,3 | 39,9 / 39,5 / 39,4 / 27,3 | 51,6 / 51,0 / 50,8 / 37,2 | 50,3 / 50,2 / 50,2 / 38,1 |

## Quan sát mô tả (không chọn hay tinh chỉnh gì theo đây)

- **Detector:** khác SVMOT, hai detector mixed5 cao hơn yolov8n_coco 4–6 điểm HOTA và có MOTA cao hơn nhiều.
  yolov8n_coco có FP lớn nhất (13,9 nghìn ở c25 CV); một phần là xe đỗ ngoài SCT_GT (xem giới hạn nhãn). cam1 thấp nhất
  với mọi detector.
- **CV/EKF/UKF gần như bằng nhau** (chênh ≤ 0,6 điểm HOTA; CV nhỉnh nhất ở hầu hết ô).
- **PF500 thấp hơn rõ và nhất quán:** so với CV, HOTA −11,6 đến −12,4 điểm, AssA giảm 17,6–20,2 điểm, IDSW gấp 2,2–2,9 lần, DetA
  giảm 3,8–4,7 điểm. Hiện tượng có ở cả 4 camera, cả 3 detector, cả 2 ngưỡng và cả 3 seed. Smoke 128 frame cũng đã cho thấy
  chiều này. Chưa chẩn đoán nguyên nhân. Một **giả thuyết chưa kiểm** là tham số nhiễu theo bước của PF/CTRV được đặt
  theo các tập trước, trong khi ở đây ảnh 4K (đơn vị pixel lớn hơn) và 15 fps (bước thời gian dài hơn). Nếu muốn kiểm
  cần lượt chẩn đoán riêng; không đổi cấu hình khóa.
- So với SVMOT (nơi PF500 nhỉnh hơn CV ở scene_06), chiều của PF500 **đảo ngược** giữa hai tập ⇒ không khái quát được kết
  luận về bộ lọc từ một tập ngoài.
