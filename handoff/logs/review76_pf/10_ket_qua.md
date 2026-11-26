# R76 — roughening chuyển động 3,0 và 4,0 (04/11/2026)

R72 đã cho thấy IDSW UA-DETRAC giảm khi tăng mức roughening chuyển động 0,5 → 1,0 → 2,0. R76 thử trước hai mức 3,0 và 4,0, cùng PF500-K1K2, N=500, R theo quan sát, ba seed, YOLOv8n COCO c25 và ByteTrack association như R72. Chỉ mở mức >2 bằng `pf_allow_high_motion_roughening: true`; khi tắt, cận kiểm cũ và mặc định không đổi. Giao thức ở `00_protocol.json`.

Kiểm thử mới 2/2, hồi quy R72 2/2 và R75 2/2. Smoke 2 video × 2 mức × 3 seed đủ 12/12. Full train dev12 đủ 72/72 lượt video, sáu run được chấm, marker/fingerprint và raw-field COMBINED được kiểm trong `src/r76_pf/analyze.py`.

**UA-DETRAC train dev12 (COMBINED trung bình ba seed):** mức 2,0 có IDSW 697,0 và HOTA 57,006%; mức 3,0 có **665,7 IDSW, 57,230% HOTA**; mức 4,0 có **644,3 IDSW, 57,153% HOTA**. Bootstrap ghép cặp 12 video, 5.000 mẫu: IDSW mức 3,0−2,0 là −31,3 [−67,7; +0,3]; mức 4,0−2,0 là −52,7 [−115,3; +0,3]. Cả hai khoảng chạm 0. HOTA mức 3,0 tăng +0,225 điểm [0,016; 0,443]; mức 4,0 tăng +0,147 điểm [−0,229; +0,589]. IDF1 cũng tăng theo điểm ước lượng ở cả hai mức.

Cổng mở BDD khóa trước yêu cầu giảm IDSW ít nhất 10% so với 2,0 mà HOTA không giảm quá 0,5 điểm. Mức 4,0 chỉ giảm 7,6%, mức 3,0 giảm 4,5%; **không mức nào qua cổng**, nên R76 không chạy BDD. PF vẫn cách EKF (472 IDSW) đáng kể. Chưa chọn mức mới, chưa đổi cấu hình luận văn; đây là tìm kiếm tham số trên tập dev đã quan sát, không phải kiểm chứng độc lập.

Bằng chứng: `results/phase2/kc48_r76_pf_dev/analysis/{per_seed,summary,paired_intervals}.csv`, `results/phase2/kc48_r76_pf_dev/eval.log`, YAML `configs/phase2/trackers/r76/`.
