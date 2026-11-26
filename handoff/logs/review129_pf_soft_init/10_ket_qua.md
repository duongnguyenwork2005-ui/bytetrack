# R129 — khởi tạo vận tốc mềm: dừng vì tăng FN

19/11/2026. Thử độc lập trên native PF100 R126, không trộn nhiễu hướng R128. Giao thức ghi trước điểm tại `00_protocol.md`.

Thay việc gán toàn bộ độ dịch hai detection thành vận tốc bằng fusion giữa Gaussian vận tốc prior và pseudo-observation độ dịch theo thời gian, có covariance từ hai phép đo. Gain tính theo covariance, không dò hệ số; dùng Joseph form, giữ 100 hạt/Q/R/association. Tương quan giữa pseudo-observation, prior và vị trí cùng frame khiến initializer này là xấp xỉ, không phải posterior Bayes chính xác.

UA-DETRAC train dev12 × ba seed, đủ **36 lượt**:

- Native PF100 nền: IDSW **458**, HOTA **59,0198%**, FN **37.429,33**.
- Initializer mềm: IDSW **465,67** (462/464/471), HOTA **58,9988%**, FN **37.529,33**.
- KF/EKF/UKF: IDSW **489/472/474**. Ứng viên vẫn ít IDSW hơn cả ba theo điểm ước lượng, nhưng kém PF100 nền và tăng FN **100 hộp / 0,267%**, vượt cổng **0,2%** đã ghi.

**Không chọn, không mở BDD, không dò gain hoặc nới cổng sau điểm.** Không chạy test40 hoặc timing cho ứng viên này. Tiếp một cơ chế process-noise khác ở R130.

Kiểm 750 fusion case ở ba thang covariance, sai khác covariance chia scale tối đa 3,50e−16; Gaussian information form khớp Joseph. PSD, no-motion không đổi trạng thái/RNG, hook restore đạt. Sáu prefix native khớp R123, sáu cặp ứng viên lặp seed trùng mảng. Hậu kiểm toàn bộ 36 marker/MOT và raw ba seed đạt, nguồn đóng băng không đổi.

Bằng chứng: `smoke.json`, `dev_summary.json`, `verification.json`, `inputs.json`, log từng seed. UA ở đây là dev từ train, không phải toàn test40; không sửa source cũ, mặc định hoặc khóa luận.
