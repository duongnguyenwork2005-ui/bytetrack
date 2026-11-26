# R128 — moment matching nhiễu hướng chưa đưa PF dưới EKF trên BDD

19/11/2026. Tiếp mục tiêu người dùng: PF có IDSW thấp hơn KF/EKF/UKF trên UA-DETRAC và BDD. Thử một ứng viên, không đổi số hạt hoặc dò sigma.

R118/R126 dùng Q Cartesian đẳng hướng, chưa đưa `_std_theta` vào Gaussian vận tốc. R128 tích phân một góc Gaussian độc lập sau coordinated-turn bằng moment chính xác rồi xấp xỉ Gaussian. Đây là thay đổi mô hình/process approximation, khác với tối ưu cài đặt R126. Công thức và nguồn nền tảng Rao–Blackwellisation ở `00_protocol.md`; nguồn không bảo đảm thắng tracking.

## Kết quả đủ tập

- UA-DETRAC **train dev12 × ba seed**, 36 lượt: IDSW **461** (462/466/455), so native PF100 **458**, KF/EKF/UKF **489/472/474**. HOTA **59,0201%**, so PF100 **59,0198%**; FN **37.438**, so **37.429,33**. Vẫn ít IDSW hơn cả ba đối chứng theo điểm ước lượng, qua cổng phát triển đã ghi. Đây chưa phải UA test40.
- BDD **200 video / 39.973 khung mỗi seed / ba seed**, 600 lượt: IDSW **5.590** (5.609/5.560/5.601), so native PF100 **5.609,33**; HOTA **33,4991%**, so **33,4975%**; FN **261.393,33**, so **261.378,67**. Cùng `mot_crowd`, cache, detector, cadence và association R125.
- KF/EKF/UKF BDD có **6.345/5.548/5.655 IDSW**. R128 ít hơn KF và UKF ở điểm trung bình, **vẫn cao hơn EKF 42 IDSW**. Không seed nào dưới EKF. Trượt mục tiêu đã đặt, không chọn làm bản cuối.

Chưa chạy UA test40 hoặc timing cho một ứng viên chưa qua cổng BDD. Kế hoạch xác nhận đủ test40 và harness đã chuẩn bị, giữ để tái sử dụng khi có ứng viên qua dev/BDD. Không điều chỉnh sigma sau điểm, không chọn seed/video có lợi. Các tập đã xem nhiều vòng, không gọi test mù hoặc tuyên bố khác biệt có ý nghĩa thống kê chỉ từ điểm trung bình.

## Kiểm và bằng chứng

Gauss–Hermite 64 điểm ở bốn mức sigma đối chiếu moment/cross-covariance, sai khác covariance tối đa 8,09e−13; PSD đạt. Zero-noise khôi phục native đúng mảng/RNG trên 240 bước. Sáu control UA và sáu cặp lặp ứng viên đạt. BDD cũng có sáu control và sáu cặp lặp.

Hậu kiểm UA 36 marker/MOT/raw và BDD 600 marker/MOT/classes/timing/raw, 1.123 hash BDD đạt. Bootstrap ghép video 5.000 mẫu lưu đầy đủ, không dùng CI để đổi cổng sau kết quả.

Bằng chứng: `smoke.json`, `dev_summary.json`, `verification.json`, `bdd/summary.json`, `bdd/verification.json`, `bdd/bootstrap.json`, các log seed. Không sửa mặc định, source cũ hoặc khóa luận. Bước tiếp là initializer mềm theo covariance R129, thử độc lập trên native PF100 R126.
