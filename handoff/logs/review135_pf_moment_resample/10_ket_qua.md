# R135 — hiệu chỉnh hai moment joint sau systematic resampling

Hoàn tất19/11/2026. **Chưa đạt mục tiêu PF ítIDSW hơn KF/EKF/UKF trên cả hai tập.**

Trên R132Native100/Rprediction/initIID, sau systematic selection hiệu chỉnh
mean/covariance joint của omega và Gaussian[cx,cy,vx,vy] bằng Schur/Cholesky.
Tính cả within-Gaussian covariance, không thêm RNG hoặc noise, không đổi N100,
Q/R/association. Variance suy biến hoặc Cholesky lỗi giữ selection thông thường.
Đây là xấp xỉ posterior hữu hạn, không phải tái hiện ETPF/NETF; giữ hai moment
không bảo đảm giữ higher moments. Tham khảo moment matching và giới hạn tại
[Acevedo/de Wiljes/Reich2017](https://arxiv.org/abs/1608.08179).

Numerical500 mixture độc lập đạt: maxmeanerror2,84e−14,maxcoverror8,81e−13,
PSD/ownership/degenerate đạt. Khi tắt correction,720 bước với50 resample khớp
parent cả state/RNG/counters. Toy bật correction có48 lần hiệu chỉnh đạt PSD.
Smoke parent controls và candidate repeat trên cả ba stage đạt.

- UAtrain dev12×3seed,36 lượt: IDSW **465** (460/473/462),
  HOTA **59,3055%**,FN **37.453,67**; đạt cổng phát triển.
- BDDval200×3seed,600 lượt,39.973frame/seed:
  IDSW **5.463** (5.440/5.460/5.489),HOTA **33,6938%**,
  IDF1 **38,5433%**,FN **261.056,33**,FP **17.725,67**.
  Dưới KF/EKF/UKF **6.345/5.548/5.655** cả3seed. SoEKF−85,
  CI95bootstrapvideo **[−179,01;+8,33]**, còn chứa0.
- UAtest40×3seed,120 lượt mới,56.340frame/seed:
  IDSW **2.521,67** (2.527/2.521/2.517),HOTA **58,4233%**,
  IDF1 **70,5171%**,FN **199.580,67**,FP **73.416,33**.
  KF/EKF/UKF **2.498/2.525/2.543**; PF vẫn cao hơnKF **23,67**,
  CI95 **[−33,67;+78,67]**. Không seed nào dướiKF; trượt IDSW/two-seeds,
  dù cổng HOTA/FN đạt.

756 lượt chất lượng mới; dùng lại240controls và120parent ablationR132 đã khóa
hash. Raw tái hợp,source/MOT/classes/timing/marker vàbootstrap5000 mỗi tập đạt.
`final_verification.json` integrityPASS nhưng `goal_observed_met:false`;
không chạy timing sau khi trượt chất lượng. Không dò transform/epsilon/N/seed.

Lượt smoke đầu UA khởi động khi preflight chưa xong nên dừng ngay tại assert
inputs.json, chưa phát lại video; log `test40_smoke.log` được giữ. Sau khi
preflight exit0, chạy lại smoke đạt tại`test40_smoke_after_preflight.log`.
Không sửa nguồn/metric/tham số trong quá trình khắc phục thứ tự khởi động.

Các tập đã quan sát nhiều vòng, có selection bias, không gọi test mù hoặc ưu
thế tổng quát. Không thay src/default/Nano/khóa luận. R136 thử Bayes action L1
bằng median marginal Gaussian-mixture, giữ nguyên cloud của R132.
