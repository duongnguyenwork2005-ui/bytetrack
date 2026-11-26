# R92 — PF hai nhánh chuyển động khi track Lost (cố định trước khi chạy)

Ngày 06/11/2026. Người dùng xác nhận phạm vi chính là camera giao thông ven đường hoặc camera gắn trên xe đang di chuyển; **không dùng UAVDT/UTUAV làm tập chọn PF**. Nền camera cố định: UA-DETRAC train dev12. Nền camera di động: BDD COCO-video val200. Cả hai đã được dự án quan sát nhiều lần, nên kết quả là **phát triển/thăm dò**, không phải test mù.

## Giả thuyết và một ứng viên

R91/R65 cho thấy PF hay lệch khi track Lost. R72 tăng roughening cải thiện một phần nhưng PF vẫn cao IDSW hơn EKF. Thử **một** biến thể chuyển trạng thái PF khi track đã Lost: trước mỗi lần dự đoán, mỗi hạt độc lập chọn nhánh chuyển động thẳng với xác suất cố định `0.5` (đặt omega=0 trước `f_batch`), hoặc giữ CTRV với xác suất `0.5`; sau đó thêm cùng process noise Q và cập nhật K1 như cũ. Track đang Tracked, detection, ByteTrack association/threshold, GT, evaluator, số hạt, seed và roughening motion2 không đổi. Đây là **transition mixture mỗi bước**, không gọi là IMM đầy đủ hoặc data-association PF. Mặc định `pf_lost_cv_fraction=0` phải trùng bit với R72; ứng viên opt-in 0,5. Không dò nhiều xác suất trên dev đã dùng nhiều lần.

[Tổng quan mô hình đa chế độ](https://onlinelibrary.wiley.com/doi/10.1155/2012/870890) giải thích vì sao chuyển động thẳng và rẽ có thể cần hai mô hình; [Rao–Blackwellized data association PF](https://users.aalto.fi/~ssarkka/pub/nrbmcda-preprint.pdf) cho thấy bất định ghép cặp là vấn đề riêng. Hai tài liệu không chứng minh biến thể đơn giản ở đây sẽ giảm IDSW.

## Cổng kỹ thuật và đánh giá

1. Unit: `fraction` ngoài [0,1] bị từ chối; nhánh CV có omega0 và dịch chuyển đúng `v·dt`; nhánh CTRV giữ công thức cũ; RNG riêng theo track; `predict` đơn và `multi_predict` bằng nhau với cùng trạng thái/RNG; `fraction=0` tạo output MOT trùng byte baseline R72 trên hai video smoke UA cho 3 seed.
2. UA smoke MVI_20052/20062, rồi UA dev12: cùng cache `yolov8n-coco_conf0.05_all`, conf>0,25, class xe 2/3/5/7, max300, fps25, buffer1,2s, elapsed skip, PF500 K1/K2 roughening motion2, seeds0/1/2. Dùng baseline R72 đã có nếu parity đạt, chấm cùng TrackEval. Cổng mở BDD: IDSW COMBINED trung bình ba seed giảm ≥10% so baseline R72 (697,0 → ≤627,3), HOTA không giảm >0,5 điểm (≥56,506%), IDF1 không giảm >0,5 điểm (≥67,861%), FP không tăng >5%.
3. Nếu UA qua cổng, chạy BDD smoke ba video đã định ở R74, rồi BDD val200 cùng cache YOLOv8n COCO c25, 5 fps, buffer1,2s, elapsed steps, seeds0/1/2. Cổng chọn cho mô tả: IDSW PF ≤10% cao hơn EKF BDD 5.548 (≤6.102,8), HOTA/IDF1 không giảm >0,5 điểm so PF R74. Không thay YAML luận văn hoặc Nano cho đến khi có kiểm độc lập mới và đo tốc độ.

Theo dõi từng video/seed và hậu kiểm hash cache, GT, output hữu hạn, không trùng `(frame, ID)`, raw fields tái cộng. Dừng ứng viên nếu cổng thất bại. Không diễn giải giảm IDSW đơn lẻ nếu HOTA/IDF1 hoặc FP xấu đi.
