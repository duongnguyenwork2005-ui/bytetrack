# R95 — cứu ghép track PF vừa Lost từ phân bố hạt

Ngày 10/11/2026. Giao thức và cổng khóa trước ở `00_protocol.md`. Phép thử dành cho xe cộ từ camera ven đường và camera gắn xe. R95 chỉ sửa chi phí association khi track PF vừa Lost và detection đều không có cặp baseline vượt ngưỡng, phân bố hạt gần detection và lựa chọn đủ rõ ràng. Detector, PF predict/update, K1/K2, roughening và ByteTrack vòng ghép thứ hai giữ nguyên. Vì association đổi, đây là **PF + association**, không phải bằng chứng rằng riêng bộ lọc PF tốt hơn. Tùy chọn `pf_assoc_particle_rescue` mặc định tắt.

## Kiểm kỹ thuật

`src/r95_pf_reacquire/test_cost.py` đạt các ca cặp duy nhất, cặp đã ghép được, track Lost quá lâu, tranh với track Tracked và cặp mơ hồ. Khi tắt tùy chọn, **6/6 tệp MOT** trên hai video UA smoke × ba seed trùng SHA với PF nền cũ. Bộ test PF hiện có đạt **26/26**; fixture R92 vẫn đạt. Cùng cache, GT, confidence, tập lớp, FPS, buffer, 500 hạt và seed với đối chứng R72/R74. BDD preflight khớp SHA manifest, cache, GT và checkpoint trước chạy. UA đủ 12 video × 3 seed, 16.919 frame/seed; BDD đủ 200 video × 3 seed, 39.973 frame/seed. `audit.py` xác nhận mọi tệp/marker hoàn tất, hộp hữu hạn, không trùng `(frame, ID)`. Các metric COMBINED tái tính từ raw fields TrackEval khớp dưới (10^{-9}).

## Chất lượng

- **UA-DETRAC dev12 (camera cố định):** PF nền → R95, trung bình ba seed COMBINED: HOTA **57,006% → 57,251%** (+0,246 điểm); IDF1 **68,361% → 69,012%** (+0,651 điểm); IDSW **697 → 646,67** (−50,33; −7,22%); FP **13.485 → 13.542**. IDSW giảm ở 10/12 video, tăng ở 2. Bootstrap ghép cặp video 95% cho delta [−84,67; −20,67]. **Qua cổng UA** yêu cầu giảm ≥5% IDSW và ràng buộc HOTA/IDF1/FP.
- **BDD100K COCO-video val200 (camera gắn xe):** HOTA **32,951% → 33,187%** (+0,237 điểm); IDF1 **37,086% → 37,533%** (+0,447 điểm); IDSW **6.291,33 → 6.162,33** (−129; −2,05%); FP **18.710 → 19.452** (+3,97%). IDSW giảm 120 video, tăng 58, bằng 22; bootstrap ghép cặp video 95% cho delta [−197,67; −54,33]. **Trượt cổng BDD** yêu cầu giảm ≥3% IDSW, tức ≤6.102,59, dù các điều kiện HOTA/IDF1/FP và đa số video giảm đều đạt.

R95 giảm IDSW trên cả hai miền nhưng **không đạt cổng chọn đã khóa cho cả hai**. Trên UA, IDSW của R92 (642,33) còn thấp hơn R95 (646,67), song R92 tăng IDSW trên BDD; không dùng so sánh hậu nghiệm này để đổi cổng. R95 vẫn cao IDSW hơn EKF đối chứng: UA 646,67 so 472; BDD 6.162,33 so 5.548. Không khẳng định PF đã ngang các bộ lọc khác.

## Chi phí và giới hạn

Trên 600 cặp video-seed BDD, 119.919 frame, timing tracker trong các lượt phát lại PC khác thời điểm cho thấy `t_update_ms` trung bình **6,176 → 7,090 ms/frame** (+14,8%); phần association **3,789 → 4,614 ms/frame** (+21,8%). Đây là thống kê mô tả, chưa phải so A/B kiểm soát trên Nano. Phép thử association đọc thêm phân bố 500 hạt nên cần tối ưu/đo riêng nếu tiếp tục nghiên cứu triển khai.

UA dev và BDD val đã được dùng nhiều lượt; khoảng bootstrap chỉ mô tả biến thiên giữa video hiện có, không biến chúng thành test mù. Không triển khai R95 vào cấu hình luận văn hoặc Jetson Nano; dừng trước HELMET/SVMOT theo cổng đã khóa. Bằng chứng số ở `ua_dev_gate.json`, `bdd_full_gate.json`, audit và scores tương ứng; chi phí ở `bdd_speed_audit.json`; raw fields ở `results/trackeval/DETRAC-all/r95-ua-dev-rescue-s{0,1,2}/` và `results/trackeval/BDDCOCOVID-val/r95-bdd-full-rescue-s{0,1,2}/`.
