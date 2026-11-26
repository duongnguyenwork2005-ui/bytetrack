# R70 — kiểm chứng các cách cải thiện PF (03/11/2026)

## Phạm vi và giao thức

Theo yêu cầu cải thiện PF, lượt này dùng lại 12 video UA-DETRAC train dev của R48, detection cache `yolov8n-coco_conf0.05_all`, confidence 0,25, lớp COCO 2/3/5/7, FPS 25, elapsed skip. Không dùng tập test mới để chọn cấu hình. Giao thức và 70 hash đầu vào đã khóa trước tracking tại `00_protocol.json` và `input_lock.json`; hậu kiểm 70/70 hash khớp.

Ba công thức PF đều có 500 hạt, mỗi công thức chạy seed 0/1/2:

- PF500: bootstrap gốc.
- PF500-K1: optimal proposal đã có trong mã.
- PF500-K1K2: thêm roughening 1,0, giá trị đã chọn từ R48.

Mỗi công thức chạy R theo chiều cao hộp quan sát (mặc định cũ) hoặc trung bình có trọng số chiều cao của các hạt đã dự đoán (`-rpred`). CV, EKF, UKF là đối chứng. Tổng 21 cấu hình × 12 video = 252 đơn vị. ByteTrack association, detector, GT và cách chấm giữ giống nhau giữa các cấu hình.

## Tính hợp lệ

- 8/8 kiểm thử cho tham số `pf_measurement_height` đạt, gồm đường nạp YAML thực, công thức likelihood/K1, giá trị sai và chế độ mặc định cũ trùng kết quả lẫn RNG.
- Smoke 21 × 2 đơn vị thành công. Lặp lại PF500-K1K2-R dự đoán seed 0 trên hai video cho tệp MOT trùng byte.
- Full dev: 21/21 run, 252/252 marker; TrackEval COMBINED tái tính từ trường thô lệch tối đa `5,56e-17`.
- EKF, PF500 và PF500-K1 (ba seed mỗi PF): 84/84 tệp MOT trùng byte với R48; GT và metadata detector trùng.

## Kết quả chính

HOTA COMBINED, lấy trung bình số học ba seed với PF (CV/EKF/UKF một run):

- CV **59,000%**, EKF **59,252%**, UKF **58,786%**.
- PF500 **52,692%** (seed 52,600–52,767%).
- PF500-K1 **55,476%** (55,280–55,606%).
- PF500-K1K2 **56,600%** (56,590–56,614%).
- PF500-R dự đoán **52,805%**, PF500-K1-R dự đoán **55,423%**, PF500-K1K2-R dự đoán **56,776%**.

Khoảng 95% mô tả từ bootstrap ghép cặp theo video, 5.000 lần lấy mẫu (12 video, seed bootstrap 0):

- K1 − PF500: **+2,784 điểm phần trăm HOTA** [1,124; 5,514], AssA +5,729 điểm, IDSW trung bình giảm 516,3.
- K1K2 − K1: **+1,124 điểm phần trăm HOTA** [0,504; 2,142], DetA +1,397 điểm, IDSW trung bình giảm 73,3.
- R dự đoán − R quan sát trong K1K2: **+0,177 điểm phần trăm HOTA** [−0,214; 0,547]. Hai công thức PF còn lại cũng có khoảng HOTA đi qua 0. Chưa đủ cơ sở đổi mặc định R.
- PF tốt nhất (K1K2-R dự đoán) thấp hơn CV **2,224 điểm phần trăm HOTA** [−4,593; −0,711] và EKF **2,476 điểm** [−4,488; −1,255]. Vì R chưa có bằng chứng rõ, ứng viên thực dụng ở lượt này là K1K2 với R quan sát: **56,600%**, thấp hơn CV 2,401 và EKF 2,653 điểm.

Các khoảng chỉ mô tả biến thiên giữa 12 video đã dùng để phát triển và lựa chọn; đây không phải kiểm định trên tập test độc lập, và không hiệu chỉnh cho nhiều so sánh.

## Chẩn đoán chuyển động

Sai số tâm trước association được chia cho chiều cao GT hiện tại. Liên kết track ID với GT bằng MOT output **trước frame dự đoán**, IoU tối thiểu 0,5; kết quả chỉ tính khi liên kết trước đó còn hợp lệ và GT ở frame hiện tại tồn tại. Vì vậy đây là chẩn đoán có điều kiện, không phải độ chính xác dự đoán vô điều kiện; số mẫu khác nhau giữa bộ lọc.

Ở nhóm cách lần xuất MOT gần nhất 3–8 frame, trung vị sai số/tầm cao GT: CV 0,292; EKF 0,253; PF500 0,593; PF500-K1 0,534; PF500-K1K2 0,492. Tỷ lệ mẫu chấm được trong nhóm này tương ứng 26,6%, 26,4%, 24,5%, 29,2%, 29,3%. K1/K2 cải thiện PF nhưng lỗi dự đoán vẫn lớn hơn EKF trong các mẫu có liên kết. Khoảng frame này có thể chứa bỏ sót detection, frame skip hoặc đứt track; không gán nhãn trực tiếp là che khuất vật lý. Nhóm ≥9 frame còn thiên lệch chọn mẫu mạnh hơn, nên không dùng để tuyên bố ưu thế riêng của bộ lọc.

## Tốc độ và quyết định

`serial_speed.json` phát lại cùng 160 frame của MVI_20052, bỏ 32 frame khởi động, đo 128 frame × 4 lượt theo thứ tự cân bằng, một luồng. Thời gian chỉ bao quanh `tracker.update`; không gồm detector, đọc cache, chuyển detection, ghi MOT. Đây là so sánh tốc độ cục bộ trên một video, không phải tốc độ hệ thống đầu cuối.

Phép đo phát lại được chạy lại ngày 03/11 sau R72 để khôi phục tệp tốc độ R70. Trung vị của bốn lượt hiện lưu (ms/frame): CV 1,53; EKF 2,19; UKF 6,15; PF500 9,84; PF500-K1 10,26; PF500-K1K2 11,27; PF500-K1K2-R dự đoán 11,48. K1K2-R quan sát khoảng 5,15 lần EKF trong phép đo cục bộ này; số tuyệt đối thay đổi theo tải máy.

K1 và K2 có lợi ích HOTA rõ trên **dev12 này**, nhưng PF vẫn chậm hơn và thấp điểm hơn CV/EKF. Giữ cấu hình luận văn hiện hành; K1K2-R quan sát là ứng viên để kiểm chứng sau trên tập độc lập. Giữ `pf_measurement_height: observation` làm mặc định tương thích; `prediction` chỉ là tùy chọn thử nghiệm. Chưa mở thay đổi association, resample-move hoặc RBPF trong lượt này.

## Đường dẫn tái kiểm

- Giao thức: `handoff/logs/review70_pf/00_protocol.json`; khóa nguồn: `handoff/logs/review70_pf/input_lock.json`.
- Cấu hình: `configs/phase2/experiments/kc48_r70_pf_dev.yaml`, `configs/phase2/trackers/r70/`.
- Run, MOT, thời gian từng frame: `results/phase2/kc48_r70_pf_dev/_processed/`.
- TrackEval: `results/trackeval/DETRAC-all/p2-kc48_r70_pf_dev-A-*/`.
- Điểm và khoảng ghép cặp: `results/phase2/kc48_r70_pf_dev/analysis/summary.csv`, `paired_intervals.csv`, `metrics_validation.json`.
- Chẩn đoán: `results/phase2/kc48_r70_pf_dev/analysis/prediction_summary.csv`, `prediction_by_seed.csv`, `prediction_by_video.csv`, `prediction_method.json`.
- Tốc độ: `handoff/logs/review70_pf/serial_speed.json`.
