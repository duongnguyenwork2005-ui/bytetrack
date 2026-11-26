# R87 — kiểm ước lượng điều kiện cho Particle Filter

Ủy quyền 06/11/2026: người dùng cho tiếp tục hoàn thiện các bộ lọc rồi kiểm các dataset khác. Kết quả R70–R81 cho thấy PF đã được cải thiện bằng proposal K1/K2 nhưng IDSW trên BDD/UA vẫn cao hơn EKF; không tiếp tục dò scalar roughening hay cửa sổ Lost trên những tập đã dùng nhiều lần.

Giả thuyết mới, cố định trước chạy: ở K1, mỗi hạt đã có phân bố Gaussian điều kiện cho `(cx,cy,a,h)`, nhưng hộp dùng ghép IoU lấy trung bình các **mẫu** rút từ phân bố đó. Tính kỳ vọng và hiệp phương sai theo luật kỳ vọng toàn phần (`m_i`, `s_i²`) có thể giảm dao động Monte Carlo mà không đổi mô hình động học, detector, association hay proposal/resampling. Thử bằng opt-in `pf_rb_observed_estimate` mặc định `false`; chỉ cho `pf_optimal_proposal=true`, `pf_estimate=mean`. Không sửa YAML luận văn hay kết quả cũ.

Thứ tự kiểm: (1) unit analytic Gaussian, seed/reproducibility, mặc định trùng byte; (2) smoke cùng cache trên tập ngoài SVMOT 9 chuỗi (đã dùng ở R73, vì vậy kết quả là thăm dò); (3) nếu HOTA không giảm >0,5 điểm và IDSW giảm ≥10% với 3 seed so baseline, mở RoundaboutHD 4 camera làm kiểm chuyển miền, rồi cân nhắc BDD/UA. Nếu cổng không đạt, không dùng làm cấu hình mới. Không chọn theo BDD/UA val/test đã được xem nhiều lần.

Tách rõ hiệu quả tốc độ tracker và chất lượng HOTA/IDF1/IDSW; hash detector cache/GT phải giống baseline; ghi seed và so từng video. Không gọi tập đã quan sát là test mù.
