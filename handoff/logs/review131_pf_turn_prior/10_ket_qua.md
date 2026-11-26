# R131 — prior quay OU hiệu chỉnh từ train không đạt cổng UA

19/11/2026. Dùng đủ 139 video `dev` thuộc BDD MOT train nội bộ đã khóa từ R35; không dùng BDD val200 hoặc UA test40 để fit. Nguồn labels có provenance có điều kiện theo R34/R35. Chỉ đọc nhãn, không cần ảnh hoặc thay detector. Giao thức đã ghi trước fit/điểm tại `00_protocol.md`.

Từ quỹ đạo tâm hộp liên tiếp, Savitzky–Golay window5/poly2 và ngưỡng tốc độ khởi tạo hiện có, thu **145.744 cặp omega**. Fit một OU zero-mean bằng conditional Gaussian MLE, lambda trong khoảng khóa trước; nghiệm trong miền, không đổi giới hạn sau kết quả.

lambda **9,30738 s⁻¹**, stationary variance **10,9742 rad²/s²**. Chuyển đúng đơn vị giữa frame và giây:

- UA 25 FPS: rho **0,68915**, sigma omega ban đầu **0,132509 rad/frame**, process std **0,096019 rad/frame**.
- BDD 5 FPS: rho **0,15544**, sigma ban đầu **0,66255**, process std **0,65449 rad/frame**.

Chỉ thay prior/process omega trên native PF100, giữ hard init, Q Cartesian/shape/R/association. Đây là OU prior cho turn rate; không phải triển khai đầy đủ Singer acceleration hoặc tích phân xy liên tục chính xác. Nguồn primaryimplementation tham khảo về tương quan mũ đã ghi trong protocol.

UA train dev12 × ba seed, đủ **36 lượt**:

- PF100 nền: IDSW **458**, HOTA **59,0198%**, FN **37.429,33**.
- OU ứng viên: IDSW **479** (472/480/485), HOTA **58,9136%**, FN **37.455,33**.
- KF/EKF/UKF: IDSW **489/472/474**. Không đạt dưới cả ba, không seed nào dưới 472; HOTA giảm 0,1062 điểm, vượt cổng 0,1 điểm. FN guardrail đạt.

**Không chọn, không mở BDD/test40, không đổi window/threshold/bounds/variance để cứu.** Fit Gaussian từ angular derivatives của bbox có thể chịu nhiễu annotation; chưa xác định nhân quả cho kết quả tracking. Việc fit tham số không tự bảo đảm prior phù hợp hơn.

Kiểm stationary variance, semigroup giữa năm bước 25 FPS và một bước 5 FPS, chuyển đơn vị, 240 native-kernel control steps đạt. Sáu prefix native khớp R123, sáu cặp ứng viên lặp seed trùng mảng. Hậu kiểm raw ba seed, 36 marker/MOT và mọi hash calibration/source đạt.

Bằng chứng: `calibration_inputs.json`, `turn_prior.json`, `fit.log`, `smoke.json`, `dev_summary.json`, `verification.json`, các log seed. Không sửa source cũ, mặc định, dataset hoặc khóa luận. Tiếp R132 về noise measurement theo chiều cao dự đoán.
