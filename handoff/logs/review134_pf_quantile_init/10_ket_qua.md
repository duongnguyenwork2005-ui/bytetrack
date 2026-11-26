# R134 — khởi tạo omega bằng các phân vị Gaussian

Hoàn tất19/11/2026. **Chưa đạt mục tiêu đồng thời BDD và UA test40.**

Trên R132Native100/Rprediction, thay100 omega IID ban đầu bằng midpoint
quantiles Gaussian, chuẩn hóa đúng mean/variance của prior. Không thêm draw,
không đổi process noise/resampling/Q/R/association hay detector. Đây là prior
hữu hạn xấp xỉ higher moments, không phải Gaussian liên tục chính xác.

Numerical240 bước parity với parent có cùng node,80 bước zero variance,
PSD/input ownership/no-constructor-draw đạt. Smoke parent controls và chạy
lặp trên UAdev,BDD,test40 đạt. Không dò grid/sigma/N/seed sau điểm.

- UAtrain dev12×3seed,36 lượt: IDSW **467,33** (467/473/462),
  HOTA **59,3157%**, FN **37.440,33**; qua cổng phát triển.
- BDDval200×3seed,600 lượt,39.973 frame/seed:
  IDSW **5.464,67** (5.471/5.454/5.469), HOTA **33,7476%**,
  IDF1 **38,6110%**, FN **261.073,33**, FP **17.729,67**.
  Cả ba seed ít IDSW hơn KF/EKF/UKF **6.345/5.548/5.655**;
  qua toàn bộ cổng BDD. IDSW trung bình bằng R132 nhưng phân bố từng video
  khác; HOTA cao hơn R132 khoảng0,0421 điểm phần trăm.
- UAtest40×3seed,120 lượt mới,56.340 frame/seed:
  IDSW **2.515,33** (2.515/2.515/2.516), HOTA **58,4290%**,
  IDF1 **70,5264%**, FN **199.562,33**, FP **73.390,67**.
  KF/EKF/UKF cùng đầu vào **2.498/2.525/2.543**. PF còn cao hơn KF
  **17,33**, không seed nào dưới KF; trượt IDSW/two-seeds dù HOTA/FN đạt.

Tổng **756 lượt chất lượng mới**; dùng lại240 lượt controls test40 và120
parent ablation từ R132 sau xác minh hash. Kiểm source/MOT/classes/timing/
marker/raw recombination đạt; paired video bootstrap5000 mẫu mỗi tập đã lưu.
Chênh IDSW so EKF trên BDD −83,33, CI95 [−180,67; +13,67]; so KF trên UA
+17,33, CI95 [−43,33; +74]. Các khoảng này còn chứa0.
Integrity PASS không đồng nghĩa mục tiêu chất lượng PASS:
`final_verification.json` và `final_summary.json` ghi `goal_observed_met:false`.
Không chạy timing vì chưa qua cả hai tập.

Không đánh đồng thắng trên UAdev với test40 hoặc chọn một seed thuận lợi.
Các tập đã quan sát nhiều vòng, có selection bias; không gọi kiểm thử mù
hoặc ưu thế tổng quát từ chênh nhỏ khi CI còn chứa0. Giữ nguyên src/default/
Nano/khóa luận. R135 tiếp theo sửa sai hai moment do resampling, không sửa
quantile grid của R134 sau kết quả.

Bằng chứng: `00_protocol.md`, `numerical.json`, `dev_summary.json`,
`bdd/{summary.json,bootstrap.json,verification.json}`,
`test40/{summary.json,bootstrap.json,verification.json}`,
`final_summary.json`, `final_verification.json`.
