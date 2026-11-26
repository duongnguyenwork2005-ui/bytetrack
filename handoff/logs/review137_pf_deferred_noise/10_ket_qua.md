# R137 — hoãn gắn nhiễu omega đến sau selection

Hoàn tất19/11/2026; **chưa đạt mục tiêu PF ítIDSW hơn cả ba bộ lọc trên hai tập**.

NativeR132 truyềnbodyGaussian bằngoldomega; innovationomega mới không xuất hiện
tronglikelihood của observation hiện tại. R137 vẫn rút cùngN Gaussian ởpredict,
nhưng sauselection gắneta_slot thayvì nhân bảneta_selected. Descendantcùngancestor
nhận innovationđộclập; không thêmartificialnoise/khôngđổiQ hayRNGcallorder.
Estimate/full_particlestrướcupdate vẫnprojectoldomega+eta; thiếuobservation thì
commit trướcpredictkếtiếp. N100,Rprediction,init,mean/systematic/association giữ.

Đây làsuydiễn riêng từconditionalindependence của môhình, không táihiệnframework
[Murray et al.,AISTATS2018](https://proceedings.mlr.press/v84/murray18a.html).
Không nhận bảođảm giảmvariance mọi estimator hoặcIDSW.

Numerical360noresample/120noobservation/480q0steps với31resample khớp parentbitwise
vàRNG. Fixtureépselection6ancestor xácnhậnoldomega[selected]+eta_slot thayvì
(oldomega+eta)[selected], đúngbody/counters/uniformweights/RNGstate. PSD/ownership,
controls6/repeat6 ởcảbastageđạt. Fixtuređầu z150 cóselection khôngduplicate nên
đổi observationtoy sangz500 trướcquality để kiểmđúngnhánh, khôngđổi filter/Q/seed.

- UAtrain dev12×3seed,36lượt: IDSW **467,33** (458/476/468),
  HOTA **59,2899%**,FN **37.447,33**; qua cổng.
- BDDval200×3seed,600lượt,39.973frame/seed:
  IDSW **5.456** (5.426/5.456/5.486),HOTA **33,7078%**,
  IDF1 **38,5589%**,FN **261.069,67**,FP **17.730**.
  KF/EKF/UKF **6.345/5.548/5.655**; cả3seedđều dưới ba mốc.
  SoEKF−92,CI95bootstrapvideo **[−181,68;−1,33]**. Đây là CIthămdò trên tập đã
  quan sát nhiều vòng; không suy thành xác nhận độc lập sau chọn nhiều biến thể.
- UAtest40×3seed,120lượt mới,56.340frame/seed:
  IDSW **2.525** (2.512/2.539/2.524),HOTA **58,4336%**,
  IDF1 **70,5449%**,FN **199.579,67**,FP **73.376**.
  KF/EKF/UKF **2.498/2.525/2.543**; chưa dướiKF, chỉ bằngEKF ởmean.
  SoKF+27,CI95 **[−32;+82,67]**, khôngseednào dướiKF. IDSW/two-seedstrượt,
  HOTA/FNđạt. KhôngđánhđồngUAdev vàUAtest40.

756lượt chấtlượng mới; dùnglại240controls và120parentablationR132 saukhóahash.
Rawtáihợp/MOT/classes/timing/marker/sourcehash/boot5000mỗitậpđạt.
`final_verification.json` integrityPASS, `goal_observed_met:false`; khôngruntiming.
Khôngdòqomega/ESS/N/seedhoặcsubsetđểcứu. Giữnguyên src/default/Nano/khóaluận.
BướcR138 thửgiữcovarianceh-vh khiLost nhưngvẫnresetmeanvh=0 theo giảđịnhbấtđịnh
khác, đãghi phươngántrướcđiểmtest40R137.
