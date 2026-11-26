# R132 — R dự đoán đã thắng IDSW BDD, chưa thắng KF trên UA test40

19/11/2026. Hoàn tất **996 lượt video chất lượng và 36 lượt timing PC**. Mục tiêu người dùng là PF có IDSW thấp hơn cả KF/EKF/UKF trên UA-DETRAC và BDD; **R132 chưa đạt mục tiêu ở UA test40**, không chọn seed hoặc đổi cổng để nhận đạt.

## Thay đổi duy nhất so native PF100 R126

Dùng chiều cao Gaussian shape dự đoán trước update để tính R, thay chiều cao detection vừa nhận. Cùng hàm noise/hệ số/confidence scaling cho cả likelihood vị trí, conditional Gaussian update và shape update. Giữ 100 hạt omega, hard init, Q, resampling và toàn bộ association/detector/metric.

Đây là thay đổi measurement-noise approximation. Numeric equal-height khôi phục native; hàm noise tạm được khôi phục cả khi exception. Backend thử riêng bằng context, YAML gốc vẫn ghi observation theo guard R112; metadata nghiên cứu ghi rõ **effective height prediction cho candidate100**, còn native100 dùng observation. YAML đơn lẻ không tái tạo backend mới. Không sửa launcher, `src/`, mặc định hoặc khóa luận.

## BDD đủ 200 video

39.973 khung mỗi seed, ba seed; cùng YOLOv8n COCO c25, năm lớp phương tiện, 5 FPS, buffer 6, elapsed steps, evaluator `mot_crowd` R125.

- PF R132: **5.464,67 IDSW**, các seed **5.447/5.453/5.494**.
- KF/EKF/UKF: **6.345/5.548/5.655 IDSW**. Cả ba seed PF thấp hơn cả ba đối chứng.
- HOTA PF **33,7055%**, so Native PF100 **33,4975%**; FN **261.064,67**, so **261.378,67**. IDSW giảm **144,67** so Native **5.609,33**.
- So EKF, PF ít **83,33 IDSW**, nhưng CI95% ghép video **[−177,33; +9,67]** còn chứa 0. HOTA thấp hơn EKF **0,121 điểm**, CI chênh HOTA **[−0,215; −0,030] điểm**. Không gọi PF tốt nhất toàn diện hoặc có ưu thế IDSW chắc chắn ngoài tập.

## UA train dev12 và xác nhận đầy đủ test40

Trên train dev12, ba seed/36 lượt: R132 có **466,67 IDSW** (460/483/457), thấp hơn KF/EKF/UKF **489/472/474**; HOTA **59,2893%**, FN **37.453,33**. Cổng phát triển đạt nên mới mở BDD, rồi test40. Không đánh đồng dev12 với test40.

Test40: **40 video / 56.340 khung mỗi lượt**, cùng cache COCO c25, lớp [2,3,5,7], 25 FPS, buffer 30, elapsed skip và CTRV dt=1 theo frame. Đối chứng mới KF/EKF/UKF mỗi bộ một lượt, Native100 ba seed và R132 ba seed; đủ **360 lượt video**:

- KF: **2.498 IDSW**, HOTA **58,2353%**, FN **199.544**.
- EKF: **2.525 IDSW**, HOTA **58,3525%**, FN **199.661**.
- UKF: **2.543 IDSW**, HOTA **58,2438%**, FN **199.671**.
- Native100: **2.509,33 IDSW** (2.497/2.523/2.508), HOTA **58,4341%**, FN **199.600,67**.
- R132: **2.520 IDSW** (2.504/2.514/2.542), HOTA **58,4594%**, IDF1 **70,5904%**, FN **199.583**, FP **73.393**.

R132 thấp hơn EKF/UKF theo điểm trung bình nhưng cao hơn KF **22 IDSW**; không seed nào dưới KF. Trượt hai điều kiện IDSW; HOTA/FN guardrails đạt. CI95% IDSW PF−KF **[−41,34; +83,67]**, chứa 0. HOTA/IDF1 có điểm cao hơn KF dù số switch hơi cao hơn; các chỉ số đánh giá khía cạnh khác nhau. Giữ nguyên kết quả không đạt, không chọn Native seed0 2.497 để nhận thắng hoặc thay seed mới.

Các tập đã được dự án quan sát nhiều vòng; đây không phải test mù và việc tiếp tục nghiên cứu phải nêu selection bias. Các ứng viên sau chọn bằng dev/giao thức mới, không dò tham số trên test40 để cứu R132.

## Tốc độ PC và kiểm chứng

Hai prefix BDD workload50/90 đã khóa từ R125, 200 khung/video, bỏ16 khung nóng, ba lần với thứ tự đảo ở lần2, đo tuần tự 36 lượt sau khi quality kết thúc:

KF/EKF/UKF/PF cổ điển500/Native100/R132 trung bình **0,960 / 1,126 / 2,745 / 5,050 / 4,123 / 4,090 ms/update**. R132 gần bằng Native100 trên mẫu này, vẫn đắt hơn KF/EKF/UKF. Không diễn giải khác biệt rất nhỏ với Native thành tăng tốc chắc chắn; chưa đo Nano hoặc toàn pipeline. Mọi mode có lặp MOT; các prefix ứng viên/control khớp full tương ứng.

Numerical 240 equal-height steps, PSD, hook restore và exception restore đạt; smoke Native/control/repeat đạt. Sau lần user tạm dừng, đã kiểm **179 marker và 851 hash** rồi replay các video thiếu marker, không resume giữa một video. Toàn bộ 360 marker/MOT/classes/timing/raw test40 và 600 BDD đã hậu kiểm; bootstrap 5.000 video-resamples mỗi tập. Hậu kiểm tổng hợp **PASS**, nhưng `goal_observed_met:false`.

## Bằng chứng

`final_summary.json`, `final_verification.json`, `test40/{summary.json,verification.json,bootstrap.json}`, `bdd/{summary.json,verification.json,bootstrap.json}`, `serial_timing.json`, `resume_check.json`; protocols `00_protocol.md`, `01_test40_confirmation.md`, `02_timing_protocol.md`.

Bước tiếp: R133 thử likelihood vị trí finite Gaussian scale mixture có covariance tổng bằng R nền, giữ các đối chứng R132 đã khóa. Không tăng số hạt hoặc đổi association để thắng bằng một biến khác.
