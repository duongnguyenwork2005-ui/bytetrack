# R93 — PF R92 trên BDD100K camera gắn xe

Ngày 06/11/2026. Giao thức `00_protocol.md` khóa trước cùng một ứng viên R92 `pf_lost_cv_fraction=0.5` trên BDD COCO-video val200. Đây là câu hỏi chuyển miền riêng sau khi cổng UA của R92 đã trượt; không dùng BDD để hồi tố chọn R92. BDD val200 cũng đã được dự án quan sát, nên kết quả vẫn là thăm dò.

`src/r93_pf_bdd_transfer/run.py` dùng cache, GT, detector, conf>0,25, tập lớp, 5 fps, buffer 1,2 s, ByteTrack association, PF500 K1/K2 roughening motion2 và ba seed như R74. Preflight đối chiếu hash cache/GT; smoke 3 video × 3 seed chạy và chấm đủ. Full đủ 200 video × 3 seed. `audit.py` xác nhận mỗi seed có 39.973 frame, 200 tệp MOT và marker hoàn tất, số dòng tương ứng 139.069/139.182/139.028, tất cả giá trị hữu hạn và không trùng `(frame, ID)`. TrackEval đã chấm 200 video mỗi seed.

So R74 PF motion2 → ứng viên R92, trung bình ba seed của COMBINED: HOTA **32,9508% → 32,9645%**; IDF1 **37,0857% → 37,1268%**; MOTA 24,7465% → 24,6444%; **IDSW 6.291,33 → 6.339,33**, tăng 48; FP 18.710 → 19.194,67. Theo trung bình seed của từng video, IDSW giảm ở 82 video, tăng ở 96, bằng nhau ở 22; bootstrap ghép cặp video cho chênh IDSW [−25,67; +122,34] (95%). Cổng yêu cầu giảm ít nhất 5% IDSW (≤5.976,77) cùng ràng buộc HOTA/IDF1/FP và đa số video giảm. **Không đạt cổng.**

Do R92 cũng trượt cổng ≥10% IDSW trên UA-DETRAC, không chọn biến thể này cho benchmark luận văn hay Nano. Mặc định PF và YAML khóa giữ nguyên. Số liệu và từng video ở `full_gate.json`, `full_scores.json`, raw fields `results/trackeval/BDDCOCOVID-val/r93-bdd-full-cv50-s{0,1,2}/`; lệnh chấm trong `full_eval.log`. Nguồn BDD này không phải tập test mù và khoảng bootstrap không khắc phục việc đã quan sát tập trước đó.
