# Xác nhận đủ UA-DETRAC test40 cho R132

Khóa trước điểm test40. Chỉ mở ứng viên sau khi qua dev12 và BDD200 nguyênsource.
Manifest `results/phase2/manifests/detrac_test.json`, đủ40video, cùngcache
`yolov8n-coco_conf0.05_all`,c25,[2,3,5,7],max300,buffer30,25FPS,elapsedskip,
CTRVdt1/frame. GT/evaluator UA MOT giữ nguyên; không dùng test đểfit/tune.

ChạyKF/EKF/UKF mỗi bộ mộtlượt xácđịnh; nativePF100R126 ba seed; R132PF100 ba
seed, tổng9nhóm/360lượtvideo. Nhãn `candidate100` làR132Rprediction, input ghi
tườngminh effective_measurement_noise_height=prediction; native100 làR126.
Không sửaYAMLguardR112 hoặc mặcđịnhlauncher. Nguồnbackend/contextphảiđi kèm.

Khóa source/GT/cache/config trước. Smoke hai videođầu/cuối64khung, tấtcảnhóm
lặpmảngtrùngnhau. Verifyđủ360marker/MOT/classes/timing vàrawtáihợp.
Đạt mục tiêu khi meanR132IDSW<min(KF,EKF,UKF),>=2/3seed cũngthấp hơnmin;
HOTA>=nativePF100-0,001,FN<=1,002*nativePF100. Không chọnlọcvideo/seed.
CI ghépvideo5000resamples seed132 chochênhHOTA/IDF1/IDSW/FN vớitừngđốichứng
vànativePF100; báođúngkhiCIchứa0, khôngnhậntươngđươngthốngkê chỉvìmeanthấp.
Không hiệu chỉnh tham sốtheotest40nếutrượt. TốcđộPCtáchkhỏichấtlượng, không
suyFPSNano. Các tậpđãxemnhiềuvòng, khônggọitestmù hoặc ưu thế tổngquát.
