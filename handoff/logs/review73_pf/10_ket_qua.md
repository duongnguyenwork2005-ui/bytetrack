# R73 — kiểm mức roughening chuyển động 2,0 trên tập ngoài (03/11/2026)

## Câu hỏi và thiết kế

R72 thấy PF500-K1K2 mức 2,0 tăng HOTA trung bình 0,406 điểm phần trăm trên UA-DETRAC dev12, nhưng khoảng video đi qua 0. Theo yêu cầu kiểm nốt, R73 so đúng hai mức 1,0 và 2,0 trên dữ liệu ngoài dev12. Ba seed 0/1/2, cùng detection cho hai mức trong mỗi tập, cùng GT, ByteTrack association, confidence >0,25, buffer 1,2 s, elapsed steps. Giao thức khóa trước ở `00_protocol.json`.

- **SVMOT test:** 9 chuỗi/8.250 frame; YOLOv8n COCO cache R64, 30 fps, ảnh toàn khung theo giao thức dự án. CV/EKF cùng cache từ R64.
- **RoundaboutHD:** bốn camera, 3.000 frame đầu mỗi camera; YOLOv8n mixed5 cache R64, 15 fps. GT SCT có xe đỗ ngoài nhãn; dùng đúng GT R64 sau sửa trùng (frame, ID). CV/EKF cùng cache từ R64.

Mỗi tập smoke 128 frame qua 6/6 PF run và TrackEval. Full 12 PF run × các chuỗi tương ứng = 78/78 lượt video. Cổng R64 kiểm hash marker, NPZ, JSON, meta, nội dung cache và GT trước tracking/chấm; bằng chứng ở `source_*_full.json`. Hậu kiểm 16/16 run gồm CV/EKF tham chiếu, TrackEval COMBINED tái tính từ raw fields lệch tối đa 5,56e-17. Không chạy lại detector và không huấn luyện lại mô hình.

## Kết quả

**SVMOT test, HOTA COMBINED:** CV 48,660%; EKF 49,649%; PF mức 1,0 **50,289%**; PF mức 2,0 **50,275%**. Mức 2,0 thấp hơn 0,014 điểm phần trăm. Theo seed ghép cặp, chênh HOTA là −0,070 / −0,114 / +0,142 điểm. IDSW trung bình 436,3 → 434,0, chênh rất nhỏ và không cùng dấu theo seed. Ba chuỗi có chênh HOTA dương, sáu chuỗi âm; không thấy cải thiện nhất quán trên SVMOT.

**RoundaboutHD, HOTA COMBINED:** CV 42,832%; EKF 42,483%; PF mức 1,0 **39,746%**; PF mức 2,0 **40,658%**. Mức 2,0 tăng 0,912 điểm phần trăm; ba seed đều tăng (+1,159 / +0,934 / +0,643 điểm), cả bốn camera đều tăng (từ +0,425 đến +1,645 điểm khi lấy trung bình seed). IDSW trung bình giảm 628,7 → 588,7, cả ba seed đều giảm 32–45 lần. PF 2,0 vẫn thấp hơn CV khoảng 2,174 điểm HOTA và EKF khoảng 1,825 điểm trên tập này.

## Diễn giải và quyết định

Tăng roughening chuyển động lên 2,0 có lợi trên RoundaboutHD, nhưng không lặp lại trên SVMOT. SVMOT chỉ gồm các chuỗi thuộc hai cảnh, RoundaboutHD gồm bốn camera của một nút giao; không coi frame hoặc camera là các mẫu độc lập để tạo khoảng tin cậy giả. Hai tập cũng đã được dự án xem ở R64/R65, nên đây là kiểm ngoài tập dev12 đã chọn tham số chứ không phải benchmark mù cuối cùng.

**Giữ 1,0 làm mặc định** và giữ cấu hình luận văn đã khóa. Có thể báo 2,0 như một ứng viên theo cảnh nút giao, kèm cả kết quả SVMOT không cải thiện. Chưa có bằng chứng để tuyên bố PF 2,0 vượt CV/EKF tổng quát hoặc coi nó là phương pháp tốt nhất cho mọi phương tiện rẽ/che khuất.

Tệp tái kiểm: `results/phase2/r73_external_pf/summary.csv`, `per_seed.csv`, `per_video.csv`, `paired_descriptive.json`, `validation.json`; nguồn và nhật ký trong `handoff/logs/review73_pf/`. Mã chạy: `src/r73_pf/validate_external.py`, phân tích `src/r73_pf/analyze.py`.
