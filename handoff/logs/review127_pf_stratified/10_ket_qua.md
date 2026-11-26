# R127 — phân tầng nhiễu omega: không chọn

Ngày 19/11/2026. Sau khi R126 giảm chi phí cài đặt, thử đúng một giả thuyết giảm biến thiên Monte Carlo cho phần omega còn lấy mẫu của Cartesian PF100.

## Thay đổi đã khóa trước điểm

Mỗi bước predict, chia [0,1] thành 100 ô, lấy một uniform ngẫu nhiên trong mỗi ô, qua inverse-normal CDF rồi hoán vị ngẫu nhiên trước gán cho hạt. Mỗi hạt có marginal Gaussian đúng; nhiễu giữa các hạt có tương quan. Không đổi số hạt, Q/R, khởi tạo, resampling hoặc association. Dùng cùng lõi native R126 đã qua kiểm hồi quy.

Đây không phải triển khai đầy đủ Sequential Quasi-Monte Carlo; tài liệu nguồn trong `00_protocol.md` không bảo đảm cách thay nhiễu predict này luôn tốt hơn IID hoặc systematic resampling hiện hành.

## Kết quả

UA-DETRAC train dev12, 16.919 khung mỗi seed, seed 0/1/2, đủ **36 lượt video**; cùng detection, cadence, buffer và evaluator R123.

- PF100 IID nền: IDSW **458**, HOTA **59,0198%**, FN **37.429,33**.
- PF100 phân tầng: IDSW **465,67**, HOTA **58,9950%**, FN **37.439,33**.

IDSW tăng 7,67; HOTA giảm 0,0248 điểm phần trăm; FN tăng 10. IDSW từng seed ứng viên **463/468/466**, so nền **456/461/457**. Mức chênh nhỏ nhưng trượt cả ba điều kiện không suy giảm đã ghi trước; không đủ cơ sở chọn.

**Đã dừng, không mở BDD, không thay N/strata/cách hoán vị để cứu kết quả.** Không đo thêm tốc độ hoặc chuyển mặc định cho một ứng viên đã trượt cổng chất lượng. Giữ native PF100 IID R126 làm ứng viên tốc độ.

## Kiểm và giới hạn

200.000 giá trị nhiễu qua kiểm finite và moment: mean −1,82e−5, variance 1,000743. Sáu prefix native IID khớp R123, sáu cặp ứng viên lặp seed trùng mảng và hook được khôi phục. Hậu kiểm 36 marker/MOT, ba raw kết hợp và 70 hash nguồn đạt.

Moment và khả năng lặp không chứng minh variance tracking tốt hơn. Đây là dev đã được xem nhiều lần, không phải test độc lập. Không sửa `src/`, detector, mặc định hay khóa luận.

Bằng chứng: `inputs.json`, `smoke.json`, `dev_summary.json`, `verification.json`; log mỗi seed tại `dev_s0.log`, `dev_s1.log`, `dev_s2.log`. Hậu kiểm liên vòng tại `../review126_pf_efficiency/native_validation/closeout_verification.json`.
