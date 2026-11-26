# R75 — giữ tâm hạt sau resampling (04/11/2026)

Người dùng yêu cầu giảm IDSW của PF để tiến gần CV/EKF. R75 thử một thay đổi PF có thể tắt: sau systematic resampling và roughening, tịnh tiến đám hạt để trung bình của nó khớp trung bình hậu nghiệm có trọng số trước resampling (góc hướng dùng trung bình vòng). Giả thuyết là loại bỏ sự lệch tâm ngẫu nhiên trước dự đoán kế tiếp. `pf_recenter_resample` mặc định false; association, detector, metric và cấu hình luận văn cũ giữ nguyên. Giao thức định trước ở `00_protocol.json`.

Hai kiểm thử mới đạt 2/2, hồi quy R72 2/2, PF gốc 26/26 assertion. Smoke UA (2 video × 3 seed) và BDD (3 video × 3 seed) đều chạy/chấm đủ. Full UA 12 video × 3 seed và BDD val 200 video × 3 seed cũng đủ marker, raw fields và COMBINED tái tính khớp. BDD dùng đúng cache/GT R61 đã kiểm SHA như R74, `mot_crowd`; UA dùng cache/GT R72, `mot`.

**UA-DETRAC dev12:** so với PF500-K1K2 motion2 R72, IDSW trung bình ba seed **697,0 → 685,3** (−11,7; bootstrap ghép cặp video 5.000 mẫu, CI 95% [−40,0; +16,0]). HOTA **57,006% → 57,132%** (+0,127 điểm, CI [−0,322; +0,488]). Theo seed, IDSW 689→690, 702→666, 700→700: chỉ một seed giảm rõ. EKF tham chiếu có 472 IDSW.

**BDD val:** so với PF500-K1K2 motion2 R74, IDSW trung bình **6.291,3 → 6.299,3** (+8,0; CI [−56,0; +71,7]). HOTA **32,951% → 32,915%** (−0,035 điểm, CI [−0,110; +0,040]). EKF tham chiếu có 5.548 IDSW.

**Quyết định:** loại ứng viên giữ tâm vì mức giảm UA nhỏ, không rõ và không lặp trên BDD. Nó không giải quyết khoảng cách IDSW tới EKF. Tùy chọn còn trong mã để tái lập, mặc định false; không thay cấu hình thí nghiệm đã khóa. Hai tập đều đã được dự án quan sát nên đây là kiểm thăm dò, không phải test mù.

Bằng chứng: `results/phase2/kc48_r75_pf_dev/analysis/{per_seed,summary,paired_intervals}.csv`; `results/phase2/r75_pf_bdd/analysis/{per_seed,summary,paired_intervals}.csv`. Mã thí nghiệm `src/r75_pf/`; YAML `configs/phase2/trackers/r75/recenter.yaml`.
