# R133 — xác nhận ứng viên120lượt, dùnglại đối chứng R132 đãkhóa

Chỉmở sauUAdev vàBDDđạt nguyênsource. Dùngcùng40video/56.340khung/cacheGT/
COCOc25/classes[2,3,5,7]/max300/25FPS/buffer30/elapsedskip/CTRVdt1 R132.
ChạyR133candidate100s0/s1/s2 đủ120lượt. KF/EKF/UKF/native1006nhóm240lượt của
R132 dùnglại nguyênraw, freezehash/truynguồn cuốiR132; khôngthayđốichứng.
ThêmR132candidate100ba seed làmparent reference chođánhgiáablation.

CổngmeanR133IDSW<2498,>=2seed<2498,HOTA>=Native100R132-0,001,
FN<=1,002*Native100R132. Cáccon số2498/2525/2543 khôngphải đối tượngfit.
Khôngđổiweights/scales/seed hoặc bỏvideo sauđiểm. Test40đãxem R132,
mọikếtluậnphảighi thăm dò vàselectionbias, khônggọitestmù.

Smoke2videođầu/cuối,3seed64khung: R132contextkhớpMOTparent, ứngviên lặpđúng.
Verify120MOT/marker/classes/timing, rawtáihợp, reference240units/hash.
Bootstrapghép40video5000mẫu seed133, PF−KF/EKF/UKF/Native100/R132Parent,
fieldsHOTA/IDF1/IDSW/FN. BáocảCIchứa0. TimingPCđoriêngsauqualitynếuđạt;
khôngsuyNanoFPS. Khôngđổisrc/mặcđịnh/thesis.
