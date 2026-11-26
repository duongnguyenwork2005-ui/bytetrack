# R72 — đa dạng chuyển động của PF (03/11/2026)

Theo yêu cầu tiếp tục cải thiện PF, lượt này đo số tổ tiên khác nhau sau resampling và thử riêng mức roughening của năm thành phần chuyển động `v, theta, omega, va, vh`. Bốn thành phần hộp giữ roughening 1,0 vì R71 không ủng hộ việc giảm chúng. Cấu hình K1K2, 500 hạt, R theo chiều cao quan sát, ByteTrack association và detection cache giữ như R70/R71 trên UA-DETRAC train dev12. Ba mức chuyển động được khóa trước: 0,5 / 1,0 / 2,0, mỗi mức seed 0/1/2. Giao thức ở `00_protocol.json`, 78 hash đầu vào ở `input_lock.json`.

## Kiểm chứng

2/2 kiểm thử R72 và 3/3 hồi quy R71 đạt. Smoke 18/18, full 108/108 lượt video; TrackEval COMBINED tái tính từ các trường thô lệch tối đa 5,56e-17. Bản mức 1,0 trùng byte 36/36 tệp MOT R71. Một worker smoke từng bị Windows từ chối tạo thư mục; chạy lại riêng worker đó thành công, sau đó smoke và full đạt đủ. 78/78 hash đầu vào khớp sau full.

## Kết quả tracking

HOTA COMBINED trung bình ba seed: mức 0,5 **56,423%**; mức 1,0 **56,600%**; mức 2,0 **57,006%**. Mức 2,0 tăng 0,406 điểm phần trăm HOTA so với 1,0; khoảng bootstrap ghép cặp video (5.000 mẫu, 12 video) **[−0,051; +1,052]** đi qua 0. DetA tăng 0,184 điểm [0,025; 0,364]; IDSW trung bình giảm 65,3 [−116,0; −19,7]. IDF1 tăng 0,601 điểm [−0,272; +1,664], chưa tách khỏi 0. Mức 0,5 giảm 0,177 điểm HOTA [−0,794; +0,197] và tăng IDSW trung bình 51,7 [16,7; 91,3]. Các khoảng mô tả chưa hiệu chỉnh nhiều so sánh; tập dev này đã dùng để phát triển và chọn PF.

## Tổ tiên và sai số dự đoán

Chỉ số cũ `n_unique` đếm cx **sau** jitter. Trên smoke MVI_20052 seed0, 4.745/4.745 lần resampling đều báo 500 giá trị cx khác nhau, nhưng trung vị số tổ tiên thật chỉ là 272. Cột mới `n_ancestors` đếm các chỉ số từ systematic resampling, không dùng tọa độ sau jitter. Trên full, trung bình ba seed của trung vị tổ tiên mỗi lần resampling lần lượt khoảng **237 / 241 / 246** cho mức 0,5/1,0/2,0. Số lần có dưới 100 tổ tiên giảm từ khoảng 9.221 → 8.233 → 6.636 mỗi seed. Chỉ số này nói về một lần resampling, chưa đo được dòng dõi xuyên nhiều frame.

Với các dự đoán có thể nối với GT ở 3–8 frame kể từ lần xuất MOT trước, trung vị sai số tâm/chiều cao GT là **0,566 / 0,492 / 0,411**. Tỷ lệ mẫu có thể chấm lần lượt **30,7% / 29,3% / 27,9%**. Vì tập mẫu khác nhau giữa cấu hình, đây chỉ là chẩn đoán có điều kiện; không dùng chênh lệch sai số làm bằng chứng độc lập rằng tracking tốt hơn. Khoảng frame không tự xác định che khuất vật lý.

Phát lại tuần tự 128 frame sau 32 frame khởi động của MVI_20052, bốn lượt, một luồng: trung vị `tracker.update` **11,25 / 11,28 / 11,07 ms/frame**. Phép đo không gồm detector, đọc cache và ghi MOT; không có bằng chứng mức 2,0 làm tracker chậm rõ trong phép đo này.

## Quyết định

Giữ mặc định `pf_roughening_motion_scale: 1.0` để tương thích. Mức 2,0 là **ứng viên** vì điểm trung bình, DetA và IDSW trên dev cải thiện, nhưng khoảng HOTA vẫn qua 0 và dev12 không độc lập. Chưa đổi cấu hình luận văn đã khóa. Bước kiểm chứng tiếp theo cần tập chưa dùng để chọn tham số, cùng detection cho PF 1,0/2,0 và CV/EKF, trước khi tuyên bố hiệu quả tổng quát.

Bằng chứng: `results/phase2/kc48_r72_pf_dev/analysis/summary.csv`, `paired_intervals.csv`, `validation.json`, `ancestor_summary.csv`, `prediction_summary.csv`; `handoff/logs/review72_pf/serial_speed.json`. Mã: `src/pf_ctrv.py`, `src/tracker_ctrv.py`; cấu hình `configs/phase2/trackers/r72/`.
