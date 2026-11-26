# R142 — forward detection confidence vào measurement update PF

Khóa trước điểmR142. R141covariancefloor trượtUAdev473,33 (481/473/466), không
nới cổng/floorweightđểcứu. Audit thực tếNPbackend: STrack.update/re_activate gọi
kalman_filter.update(mean,cov,z) khôngscore; RBPFTrackView.forwardconfidenceNone,
nênR141noisecontext trongreplay thực cũngNone. HàmEKF._measurement_noise đãcó
R(c)=max(1-c,0,05)*R0 nhưng chưa được kíchhoạt bởi đường này. Không tuyên bố
cácbenchmarkcũsai: chúng là cấu hìnhfixedR đãkhóa, giữ làmcontrols.

## Thay đổi duy nhất

NềnR132Native100/Rprediction/initIID/mean/systematic/shapeclampcũ, **không có
covariancefloor R141**, khôngghépR133–R141. Adaptertrack riêng truyềnscore hiện
tại qua update/re_activate đếnRBPFTrackView.update, rồiParentPF.update.
Chỉ kíchhoạt côngthứcNSAmeasurementR hiệncó, floor0,05 giữ nguyên từrepo,
không thửcácfloor/gain/exponents. KhôngđổiQ,detector,threshold,association,
tracklifetime,classes hoặcmetric. Khôngdùngscorecủaframecũ. Birthinit giữnhưcũ.

Adapter không thực hiệnmatching; baseSTrack vẫn gọi nguyênvẹn vàcùngsốlần.
Trạngtháipendingconfidence đượcrestorefinally. Khiadapterroutingoff phải MOT
khớpParentR132. Nếuexplicitconfidence đãtruyềnthì giữ nó; chỉ lấy pending khiNone.
Đây làPF+confidence-conditioned R, khôngablationthuầnthuậttoánPF vàkhôngnhận
confidenceclassificationlàlocalizationuncertaintyđãcalibrate.

Nguồnprimary: [StrongSORT](https://arxiv.org/abs/2202.13514),
[Ultralytics Kalman source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/trackers/utils/kalman_filter.py).
R142không triểnkhaiReID/AFLink/GSI/cảStrongSORT; chỉứngdụngnoiseadaptationvào
Gaussianupdate/likelihood củaRBPF. Corefunctionđãcótrongsrc/ekf_ctrv.py584–600.

## Kiểm và cổng

Fakeview routing current/explicit/none, no staleconfidence, restorecontexts/
exceptions. DirectGaussianupdate phải khớpParentkhi truyềncùngscore/R;
c=0 khôi phụcR0,caoScorecoRđúngfloor0,05. Route-off video controls6 soR132 và
parentcontrols6/repeat6 trướcUAdev12×3seed0/1/2. Audit current/deliveredscore
ởsmoke, khônggiữlogkhổnglồởproductionquality.

UAmeanIDSW<472/2seed<472/HOTA>=R123PF100-0,001/FN<=1,002R123. Đạt mớiBDD200×3seed/
mot_crowd<5548/2seed/guardrailssoR125PF100; đạt mớiUA test40×3seed<2498/2seed/
guardrailssoNative100R132. Cùngrefs/cache/GT/hash, raw/MOT/CLS/timing/marker,
pairedvideo bootstrap5000. Đạt cảhai mới timingPC42lượt, khôngNanoFPS.
Khôngdòconfidencefloor/gain/seed/N/subset sauđiểm; báoselectionbias/CI vì các
tập đãquan sát. Khôngsửa src/default/Nano/khóaluận.
