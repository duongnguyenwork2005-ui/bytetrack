# R136 — UA test40 sau khi qua UAdev và BDD

Toàn40 video/56.340 frame với cùng cache COCOc25/classes/GT/evaluator R132.
Factory MedianPF trên nền R132Native100/Rprediction, chỉ thay point estimate
cx/cy bằng marginal Gaussian-mixture median; risk matrix quanh point estimator.
Không đổi cloud/weights/Q/R/init/resampling/association, không dùng R133–R135.
Chạy120 lượt mới (seed0/1/2), dùng lại240 controls và120 parent ablation R132.
Parent controls6/repeat6 trước full, hash khóa trước/sau, raw/MOT/CLS/timing/marker.

Đích meanIDSW<2498/ít nhất2seed<2498/HOTA>=Native100-0,001/FN<=1,002Native100.
Paired video bootstrap5000, seed133 như helper. Không dò root tolerance, axis,
blending mean/median,seed,N sau điểm. Helper R133 chỉ dùng I/O/evaluator/ref;
factory thực R136. Các tập đã quan sát, báo selection bias và CI.
