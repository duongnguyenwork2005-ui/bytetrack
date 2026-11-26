# R137 — phương án kế tiếp nếu R136 không đạt: defer angular innovation

Khóa trước điểm R137; chỉ mở quality sau khi R136 kết thúc/không đạt.
Nền R132Native100/Rprediction/initIID/systematic; không trộn R133–R136.
Mục tiêu và cổng giữ nguyên: UAdev12×3seed<472/2seed/guardrails soR123;
đạt mớiBDD200×3seed<5548/2seed/guardrails soR125; đạt mớiUA test40×3seed
<2498/2seed/guardrails soNative100R132. HOTA>=base-0,001,FN<=1,002base.
Không dò qomega/N/ESS/seed hoặc chọn subset sau điểm.

## Nhận xét về mô hình đã khóa

Native predict dùng F(omega_old) để truyền body Gaussian, rồi mới thêm
eta_i~N(0,qomega²) vào omega. Likelihood vị trí ở frame hiện tại chỉ phụ thuộc
body đã truyền, **không phụ thuộc eta_i**. Nếu selection nhân bản một ancestor,
cách hiện tại nhân bản cả eta_i vốn chưa được observation hiện tại quan sát.

Thử hoãn gắn eta vào omega cho đến sau selection: predict vẫn rút N Gaussian
như cũ và lưu pending_noise; body dùng oldomega như cũ. Estimate/full_particles
cho compatibility vẫn báo oldomega+pending_noise trước update. Gaussian update
và likelihood không đổi; systematic selection chọn body/oldomega như cũ, với
đúng uniform draw và thứ tự RNG như nền. Sau đó omega_new=oldomega_selected
+eta_slot, **không** eta_selected. Các descendant của cùng ancestor được angular
innovation độc lập; không bơm artificial noise, không đổi Q hay số draw.

Nếu không resample, commit cùng slot nên khớp nền. Nếu không có observation,
commit pending trước predict kế tiếp, bảo toàn động học qua gap. Motion-init và
cache/full_particles phải giữ đúng covariance/compatibility trước update.
Khi qomega0 phải khớp nền đầy đủ kể cả resampling/RNG. RNG consume vẫn nằm ở
predict; chỉ dời lúc gắn innovation để selection không nhân bản noise độc lập.

Đây là suy diễn riêng từ conditional independence của mô hình R132, không phải
tái hiện framework probabilistic programming. Tham khảo ý tưởng trì hoãn sampling
để tận dụng cấu trúc giải tích ở
[Murray et al.,AISTATS2018](https://proceedings.mlr.press/v84/murray18a.html).
Không nhận bảo đảm giảm IDSW hoặc đã chứng minh giảm variance mọi estimator.

## Kiểm bắt buộc

- Nhiều bước không-resample: returned state/covariance/full_particles và RNG
  khớp parent; sau update cả omega/body/logw khớp bitwise.
- Nhiều bước không-observation/lost: commit trước predict kế tiếp khớp parent.
- Qomega0 có resample: khớp đầy đủ state/RNG/counters, không noise nhân tạo.
- Fixture ép ESS thấp: công thức hậu-resample làoldomega[selected]+eta_slot,
  body/weights/counters như parent ở sự kiện đầu, RNG trạng thái không đổi.
- Input ownership/cache/PSD/sync compatibility, controls6/repeat6 trên video.

Sau quality: raw/hash/MOT/CLS/timing/marker/CIbootstrap5000 như các vòng trước.
Nếu đạt cả hai tập mới timing PC tuần tự42 lượt; không suy NanoFPS.
Các tập đã quan sát nhiều lần, báo selection bias và CI. Không sửa src/default/
Nano/khóa luận. Không mở R137 nếu R136 đã đạt mục tiêu, trừ yêu cầu mới.

Kích hoạt sau khi R136dev đủ36 lượt: mean470 nhưngseed476/478/456, trượt
two-seeds; không mởBDD choR136. NumericalR137 đạt360 bước không-resample,
120 bước mất observation,480 bướcqomega0 với31resample, forced selection6
ancestors đúng công thức/no-extra-RNG. Fixture đầu z150 có thể chọn đủ100
ancestor khác nhau nên chưa kiểm được nhánh duplicate; đổi riêng observation
toy thànhz500 để ép6ancestor trước khi khóa quality. Không sửa filter/Q/seed.
