# R66 Sherbrooke, rà cổng trước smoke v2

**Trạng thái khi rà là chưa duyệt full.** Bằng chứng lịch sử và smoke v2 dùng namespace riêng. Lượt này chỉ đọc hồ sơ và tái tổ hợp CSV bằng CPU.

Phiên bản đã rà có `sher_matrix.py` SHA-256 `71ebaa31d17c64f220dbbb6fdf092d6c8c97f64d511730b0aa6383a434196d69`, `danh_gia_r66.py` `dc1d50fca4f8cc10b69185378babfab1d99b33508b25cc9c1ba5d932b4c1990d`, `sher_faults.py` `a73e199cd3f2dd427c018e3a98c0859e242f907a3bb46d28e81d41263d74ae2f` và `sher_metric.py` `446bcd158d07506f0e2e99cbae14fd462217f001e3dc8baa9bcafebe9ee33c63`. Đối chiếu theo biên bản độc lập `hoi_codex_R66_review_smoke_sherbrooke_20261102/answer.json`.

- **B2, chặn smoke.** JSON đọc `hw` thành list, trong khi bảng Python của `giao_thuc(ds)` dùng tuple. Mô phỏng serialize/deserialize trong bộ nhớ cho kết quả bất đẳng. Cache vừa sinh sẽ dừng trước marker. Cần so với `lock["inputs"]` đã chuẩn hóa về JSON.
- **B1/B6, điều kiện duyệt full.** Kiểm thành phần chính của `cfg_inputs` trên metadata thật của CV và PF500 seed 0 ở smoke lịch sử khớp toàn bộ khóa mong đợi. Cổng full hiện chỉ nối hash của `coverage.json` tới quyết định. Cần kiểm tiếp hash CSV và phạm vi/marker của 36 lượt cùng bằng chứng repeat để phát hiện mất output smoke sau review.
- **B6, độ phân biệt fixture.** Cần một đối chứng dương trên bản sao trước và sau khi tiêm lỗi, và bắt đúng lỗi dự kiến. Nếu đường nguồn không đọc được, `check_fails` hiện có thể nhận OSError ở mọi ca. Cần ca seqinfo đổi, ID trùng và một seed dở trong khi seed khác có marker. Temp trong workspace tránh lỗi đổi tên tệp của Windows sandbox TMP.
- **B5.** Runner xét D/F, trần thời gian thật và từng seed riêng. Việc so parity lịch sử phải cho phép ghi trạng thái thiếu khi không có mẫu cũ, vì cổng chính căn cứ vào dữ liệu và provenance của lượt v2.
- **B3, kiểm dữ liệu lịch sử.** Tái tổ hợp độc lập toàn bộ 36 `raw_fields.json` của smoke lịch sử bằng helper R66 mới, so với `combined_metrics.csv`. Sai lệch lớn nhất bằng 0 cho HOTA, DetA, AssA, IDF1, MOTA, IDSW, FP, FN khi đọc CSV với csv.DictReader. Phần này chưa chứng nhận smoke v2.

Để duyệt full cần bằng chứng cùng phiên bản mã sau sửa, gồm khóa nguồn, 36 marker đã hậu kiểm, lặp PF có ROI và điểm khớp, fixture dương/âm đạt, và biên bản dữ liệu/sai lệch B4 và B7. Thời điểm kiểm trạng thái ổ đĩa lúc 22:00, sandbox chỉ thấy C: và D:.
