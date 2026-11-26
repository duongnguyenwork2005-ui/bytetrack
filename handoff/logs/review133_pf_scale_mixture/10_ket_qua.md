# R133 — Hỗn hợp Gaussian cho likelihood vị trí PF

Hoàn tất ngày 19/11/2026. **Chưa đạt mục tiêu PF ít IDSW hơn cả KF/EKF/UKF
trên đồng thời BDD và UA-DETRAC test40.** Không chọn làm mặc định.

## Thay đổi và kiểm chứng

Trên R132 Native Cartesian PF100, dùng ba Gaussian có scale khác nhau cho
nhiễu đo vị trí. Gauss–Laguerre ba node với Gamma shape2 được khóa trước điểm;
chuẩn hóa để covariance hỗn hợp bằng R của mô hình nền. Mỗi hạt omega cập nhật
ba Gaussian rồi ghép posterior theo hai moment, kể cả covariance giữa các
component. Nhiễu đo kích thước và toàn bộ Q/init/association giữ nguyên.
Đây là xấp xỉ likelihood đuôi nặng bằng hỗn hợp hữu hạn, không phải Student-t
chính xác. Moment collapse cũng là xấp xỉ cho những bước tiếp theo.

Kiểm độc lập 100 trường hợp hỗn hợp, 240 bước giới hạn một component, PSD,
ownership, fallback khi C kernel từ chối và kiểm đầu vào đạt. Sai khác covariance
so công thức độc lập tối đa 7,11e-15. Kurtosis vị trí hỗn hợp 3,77778; covariance
chuẩn hóa giữ nguyên. Smoke Native/R132/one-component và chạy lặp seed đều đạt.

## Kết quả đầy đủ

- UA train dev12 × ba seed: IDSW **461,33** (462/468/454), HOTA **59,3166%**,
  FN **37.440,33**. Qua cổng phát triển đã khóa.
- BDD COCO-video val200, 39.973 frame/seed, đủ **600 lượt video-seed**:
  IDSW **5.527** (5.532/5.524/5.525), HOTA **33,6401%**, IDF1 **38,4321%**,
  FN **261.131,67**, FP **17.744,67**. Cả ba seed thấp hơn KF/EKF/UKF
  **6.345/5.548/5.655**. Qua cổng BDD, nhưng IDSW cao hơn R132 62,33.
- UA-DETRAC test40, 56.340 frame/seed, đủ **120 lượt ứng viên**:
  IDSW **2.518,33** (2.516/2.523/2.516), HOTA **58,4597%**,
  IDF1 **70,5880%**, FN **199.573,33**, FP **73.390,67**.
  Các tham chiếu cùng dữ liệu: KF/EKF/UKF **2.498/2.525/2.543**;
  Native PF100 **2.509,33**; R132 **2.520**. Ứng viên còn cao hơn KF **20,33**,
  cả ba seed đều chưa dưới KF. Cổng IDSW và hai-seed trượt; HOTA/FN đạt.

Không đánh đồng kết quả UA dev12 với UA test40. Native/R132 tham chiếu test40
được dùng lại từ R132 sau xác minh hash, không chạy lại để chọn seed thuận lợi.
Tổng cộng **756 lượt chất lượng mới**, 240 lượt controls test40 và 120 lượt
parent ablation dùng lại. Không chạy timing vì chưa qua mục tiêu chất lượng.

## Tính toàn vẹn và giới hạn

Trong lượt chạy, Codex Desktop update làm gián đoạn ba worker ở 13/40 video
mỗi seed. `resume_check.json` xác minh 39 marker và 1.346 hash trước khi tiếp tục;
các worker hoàn tất từ video kế tiếp, không đổi thuật toán hoặc tham số.
Tái hợp raw metric, MOT/classes/timing, marker và hash nguồn đạt. Bootstrap
ghép cặp theo video 5.000 mẫu mỗi tập đã lưu: chênh IDSW so EKF trên BDD
−21, CI95 [−108,01; +69,34]; so KF trên UA +20,33, CI95 [−41,33; +80,67].
Cả hai còn chứa 0, chưa có bằng chứng ưu thế IDSW chắc chắn so các mốc này.

Các tập đã được quan sát qua nhiều vòng nghiên cứu, nên có selection bias;
đây không phải kiểm thử mù độc lập. Không thay src, detector, metric,
mặc định Nano hay khóa luận. Bước tiếp R134 thử giảm sai số prior omega bằng
phân vị Gaussian, trên nền R132, không chỉnh scales của R133 sau kết quả.

Bằng chứng: `00_protocol.md`, `numerical.json`, `dev_summary.json`,
`bdd/{summary.json,bootstrap.json,verification.json}`,
`test40/{summary.json,bootstrap.json,verification.json}`,
`final_summary.json`, `final_verification.json` (`goal_observed_met:false`).
