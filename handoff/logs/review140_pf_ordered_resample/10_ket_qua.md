# R140 — omega-ordered stratified resampling

Hoàn tất19/11/2026; **chưa đạt mục tiêu trên đồng thời BDD vàUA test40**.

TrênR132Native100/Rprediction/initIID/mean/shapeclampcũ, khiESS<0,5N đổi
systematic thành stratifiedCDF sau sắpxếpparticles theoomega, gatherđồngbộ
bodymeans/covariances/omega. N100,Q/R/association giữnguyên. DùngNuniformdraw
thay1 nênchuỗiRNGsauresample thayđổi, giữseeds0/1/2, khôngchọnseedthuậnlợi.
Đây làresampling củaweights, khôngphải processnoiseCDF nhưR127.

Nguồnprimary: [Li et al.](https://arxiv.org/abs/2004.01975),
[Gerber/Chopin/Whiteley](https://arxiv.org/abs/1707.01845). Chỉomega đượcorder,
bodyGaussians phụthuộclịchsửnhiềuchiều nênkhôngnhậnmọiđiềukiện địnhlý1D
đượcđápứngcho mọiobservablebbox. KhôngdùngHilbert/SQMC/optimaltransportsolver.

Numerical500selection so scalarCDF/permutation/expectedoffspring/zero/uniform/
ownership đạt; maxexpectedoffspringerror6,14e−14. Tắtthayđổi600steps với43
resample khớpparent/RNG;PSDđạt. Controls6/repeat6 ởcảbastageđạt.

- UAtrain dev12×3seed,36lượt: IDSW **462,67** (461/472/455),
  HOTA **59,3042%**,FN **37.438,67**; qua cổng.
- BDDval200×3seed,600lượt,39.973frame/seed:
  IDSW **5.446,67** (5.417/5.442/5.481),HOTA **33,7137%**,
  IDF1 **38,5828%**,FN **261.066,33**,FP **17.725,67**.
  KF/EKF/UKF **6.345/5.548/5.655**, cả3seedthấp hơn cácmốc.
  SoEKF−101,33,CI95 **[−195,34;−7,32]**; dữliệuđãquansát/chọnnhiềubiếnthể,
  khônggọixácnhậnđộclậpsau chọnmodel.
- UAtest40×3seed,120lượtmới,56.340frame/seed:
  IDSW **2.518,67** (2.516/2.516/2.524),HOTA **58,4392%**,
  IDF1 **70,5701%**,FN **199.589**,FP **73.404**.
  KF/EKF/UKF **2.498/2.525/2.543**; chưa dướiKF, khôngseednàođạt.
  SoKF+20,67,CI95 **[−43,67;+80,67]**. TrượtIDSW/two-seeds,HOTA/FNđạt.

756lượtchấtlượngmới; dùnglại240controls+120parentablationR132 sauhash. Rawtáihợp/
MOT/CLS/timing/marker/sourcehash/boot5000mỗitậpđạt. `final_verification.json`
integrityPASS nhưng `goal_observed_met:false`; khôngchạytiming. Khôngdòordering/
ESS/N/seed/subsetđểcứu. Khôngđổi src/default/Nano/khóaluận. R141thửcovariancefloor
khi motion-init dựatừuncertaintyhaiobservations, giữpointvelocitymean nhưnền.
