# R88 — Urban Tracker: kết quả thăm dò car-only

Nguồn tác giả và hash archive: `handoff/logs/review69_filters/datasets.json`. Giao thức và hai lần sửa **trước ma trận so bộ lọc**: `00_protocol.md`. Mã chuyển GT/cache/chạy: `src/r88_urbantracker/`. Kết quả máy: `source_audit.json`, `detector_smoke.json`, `gt_audit.json`, `matrix_progress.json`, `summary.json`. Hậu kiểm `summary.json` đạt 36 lượt tracker/TrackEval, mỗi lượt 1.000 frame đủ timeline, cache/GT/FPS/seed/conf cùng giao thức và MOT hữu hạn, không trùng (frame, ID).

Hai cảnh có 1.000 frame nhãn mỗi cảnh. René có 6.600 box car/23 ID và 2.818 box distractor, St-Marc có 912 car/7 ID và 7.463 distractor. GT nguồn có `unknown` 1.176/2.686 box tương ứng. Ảnh JPG được ánh xạ GT frame `t` sang ảnh `t+1` theo 0/1-based; hình kiểm đầu `alignment_first_frame.jpg`, chưa có bằng chứng căn chỉnh tuyệt đối cho St-Marc. FPS René 30000/1001 theo MOV, St-Marc 30 theo trang tác giả.

YOLOv8n COCO tại conf>0,05 và IoU≥0,5 trên mẫu hệ thống 50 frame/cảnh: car René 187/336, St-Marc 40/46; bicycle 0/67 và 2/22. YOLOv8n mixed5: car 7/336 và 36/46, bicycle 0/67 và 0/22. Vì vậy chỉ chấm car bằng COCO class2. Trên toàn René, detection car ở conf>0,25 khớp 1.448/6.600 GT (21,94%) trước tracking. Mask ROI tác giả giữ 6.592/6.600 GT tâm nằm trong nhưng chỉ 3.904/10.256 detection car c25; cache ROI là bản dẫn xuất có hash mask và nguồn. Lượt CV toàn khung c25 HOTA 11,82%, sau ROI HOTA 15,47% (phép chẩn đoán, không dùng chọn cấu hình).

Các số sau là HOTA / IDF1 (%) và IDSW, với PF là trung bình **ba seed**, mỗi cảnh báo riêng:

- René ROI, c25: CV 15,47 / 19,69 / 28; EKF 15,76 / 19,78 / 23; UKF 15,43 / 19,26 / 29; PF mặc định 13,68 / 16,18 / 40,67; PF K1K2 motion100 14,22 / 16,89 / 38,00.
- René ROI, c10: CV 19,31 / 21,89 / 16; EKF 19,11 / 21,47 / 16; UKF 18,73 / 20,72 / 19; PF mặc định 16,88 / 17,91 / 26,67; PF K1K2 motion100 17,77 / 19,65 / 21,67.
- St-Marc, c25: CV 36,32 / 30,81 / 7; EKF 36,05 / 31,25 / 13; UKF 36,01 / 30,54 / 13; PF mặc định 34,68 / 30,46 / 9; PF K1K2 motion100 35,93 / 30,86 / 8,67.
- St-Marc, c10: CV 31,41 / 24,02 / 7; EKF 31,56 / 24,21 / 13; UKF 31,52 / 24,21 / 13; PF mặc định 30,06 / 24,01 / 7; PF K1K2 motion100 30,97 / 23,97 / 8.

PF K1K2 motion100 tốt hơn PF mặc định trên HOTA ở cả bốn cặp cảnh/ngưỡng, IDSW giảm tại René nhưng **không bằng CV/EKF**. Cảnh St-Marc c10 PF mặc định có IDSW 7 như CV, song HOTA thấp hơn và chỉ 7 car ID nên không dùng một điểm này tuyên bố ngang bằng. MOTA René ROI c25/c10 âm 8–41%, St-Marc âm 212–383% vì nhiều FP và vùng nhãn hẹp; điều này hạn chế mạnh diễn giải chất lượng tracking. Không có motorcycle hay cờ che khuất riêng, và hai scene không đủ để tính CI khái quát. Đây là phép đo nội bộ với detector của dự án, **không so trực tiếp với Polytrack ABSDIST** của tác giả.

Không chọn cấu hình PF mới, không thay baseline luận văn/Nano. Bộ MOT17 nhiều chuỗi đang được chuẩn bị cho kịch bản che khuất; kết quả MOT17 phải báo riêng vì là người đi bộ.
