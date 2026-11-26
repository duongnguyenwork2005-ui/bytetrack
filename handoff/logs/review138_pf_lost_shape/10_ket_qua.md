# R138 — giữ covariance kích thước khi Lost

Hoàn tất19/11/2026. **Chưa đạt mục tiêu PF ítIDSW hơn cả ba bộ lọc trên hai tập.**

NativeCkernel.c48 đặtmeanvh=0 vàxóacovarianceh-vh khiLost; đây làclampxácđịnh,
không phải lỗi số học. R138thử giảđịnhkhác: chỉresetmeanvh=0, giữcovariance/
crosscovariance rồiFPF^T+Q cũ. Khôngbơmnoise/inflatevariance, khôngđổiByteTrack
Loststate/buffer/association. N100, body/omega/init/Rprediction nhưR132.

Numerical600notLoststeps khớp parentbitwise/RNG;36Loststeps khớp independent
Gaussian2x2oracle với maxerror0;PSD/ownershipđạt. Toy12framegap giữhvar1494,14
thay451,87 vàgiảmheightresidual sauupdate, chỉlàtoy, khôngsuyraIDSWthật.
Controls6/repeat6 ởcảbastageđạt.

- UAtrain dev12×3seed,36lượt: IDSW **465** (468/469/458),cả3<472,
  HOTA **59,2875%**,FN **37.460,33**; qua cổng.
- BDDval200×3seed,600lượt,39.973frame/seed:
  IDSW **5.424,33** (5.400/5.421/5.452),HOTA **33,7810%**,
  IDF1 **38,6713%**,FN **261.100,33**,FP **17.709,33**.
  KF/EKF/UKF **6.345/5.548/5.655**, cả3seedthấp hơn cácmốc.
  SoEKF−123,67,CI95bootstrapvideo **[−221,34;−25,65]**. Dữliệuđãquansát vàchọn
  nhiềubiếnthể, khônggọilàxácnhậnđộclập sau chọnmodel.
- UAtest40×3seed,120lượtmới,56.340frame/seed:
  IDSW **2.539,33** (2.544/2.534/2.540),HOTA **58,4578%**,
  IDF1 **70,6143%**,FN **199.645**,FP **73.309**.
  KF/EKF/UKF **2.498/2.525/2.543**; chỉthấp hơnUKF ởmean, chưa dướiKF/EKF.
  SoKF+41,33,CI95 **[−17;+99]**. IDSW/two-seedstrượt, HOTA/FNđạt.

756lượtchấtlượngmới; dùnglại240controls+120parentablationR132 sauhash. Rawtáihợp,
MOT/CLS/timing/marker/sourcehash vàboot5000mỗitậpđạt. `final_verification.json`
integrityPASS nhưng `goal_observed_met:false`; khôngchạytiming. Khôngdòcovariancescale/
seed/N đểcứu. Giữnguyên src/default/Nano/khóaluận.

R139thửdamped extenttrend khiLost với horizonbằngbuffer đãkhóa30/6frame,
phươngánđãghi trướckhi cóđiểmUA củaR138; khôngđổi buffer hoặcassociation.
