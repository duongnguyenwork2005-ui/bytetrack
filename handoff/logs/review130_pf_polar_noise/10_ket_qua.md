# R130 — Q vận tốc theo hướng chưa đạt IDSW UA

19/11/2026. Thử một ứng viên process-noise khác, không chỉnh initializer/gain R129. Native PF100 R126 làm nền, giữ 100 hạt, hard init, shape/R và association.

Thay Q vận tốc Cartesian đẳng hướng bằng noise tốc độ một chiều theo tensor M/tr(M), M=Pvv+u*u^T, cộng moment heading Gaussian với sigma cũ. Tensor là xấp xỉ ratio-moments, không phải tích phân chính xác hướng của Gaussian. Không dò scale hay thay noise theo điểm dataset. Công thức/gates ghi trước tại `00_protocol.md`.

UA-DETRAC train dev12 × ba seed, đủ **36 lượt**:

- Native PF100: IDSW **458**, HOTA **59,0198%**, FN **37.429,33**.
- Ứng viên: IDSW **472,33** (475/466/476), HOTA **58,9676%**, FN **37.424,33**.
- KF/EKF/UKF: IDSW **489/472/474**. Ứng viên chưa thấp hơn EKF theo điểm trung bình; chỉ một seed dưới 472. Trượt hai điều kiện IDSW đã ghi, dù guardrail HOTA/FN đạt.

**Không chọn, không mở BDD, không đổi tensor/scale để cứu.** Không chạy test40 hoặc timing cho ứng viên này. Tiếp R131 với prior tốc độ quay được hiệu chỉnh trên phần train nội bộ.

Kiểm 1.000 tensor ngẫu nhiên, PSD/trace1 và giới hạn hướng biết rõ/đẳng hướng/zero đạt; 240 bước zero-noise khớp native theo mảng/RNG; 200 bước non-zero giữ finite/PSD. Sáu control UA khớp R123, sáu cặp ứng viên lặp seed trùng mảng. Hậu kiểm 36 marker/MOT/raw và 73 hash nguồn đạt.

Bằng chứng: `smoke.json`, `dev_summary.json`, `verification.json`, `inputs.json`, các log seed. UA ở đây là dev từ train; chưa sửa mặc định, source cũ hoặc khóa luận.
