# R93 — kiểm chuyển miền camera gắn xe cho PF R92, không cứu cổng UA

Ngày 06/11/2026. R92 fixed-camera UA dev12 giảm IDSW 7,8%, tăng HOTA/IDF1 nhưng **trượt cổng 10%**. Không đổi kết luận R92. Người dùng xác nhận camera gắn trên xe là phạm vi chính, nên phép R93 hỏi độc lập: biến thể **đã cố định** `pf_lost_cv_fraction=0.5` có làm PF tệ hơn trên camera di động không? Không chỉnh xác suất hay ngưỡng theo BDD.

Nguồn BDD COCO-video val200, 39.973 frame, 200 video, cache YOLOv8n COCO R61, conf>0,25, lớp 1/2/3/5/7, max300, fps5, buffer1,2s, elapsed steps, PF500 K1K2 roughening motion2, seeds0/1/2. Baseline R74 motion200 cùng cache/GT/evaluator đã có: HOTA TB seed 32,9508%, IDF1 37,0857%, IDSW 6.291,33, FP 18.710. Trước full, smoke trên đúng ba video R74 để kiểm hình dạng, nguồn, tính hữu hạn, seed; **không dùng điểm smoke quyết định chọn video hoặc thay tham số**. Sau đó chạy đủ 200 video × 3 seed. Không dùng BDD từng dò nhiều vòng làm bằng chứng test mù.

Kết quả mô tả đầy đủ IDSW, HOTA, IDF1, FP COMBINED theo seed; số video IDSW tăng/giảm, bootstrap ghép cặp 200 video cho IDSW. Chỉ gọi ứng viên "đáng kiểm trên tập độc lập" nếu IDSW giảm ≥5% (≤5.976,77), HOTA≥32,4508%, IDF1≥36,5857%, FP≤19.645,5 và số video IDSW giảm > số tăng. Đây **không phải cổng chọn triển khai**: R92 đã trượt cổng UA 10%, và BDD đã quan sát trước. Cấu hình luận văn/Nano không thay đổi.
