# R132 — R theo chiều cao dự đoán cho Cartesian PF100

Khóa một giả thuyết độc lập trước điểm, chuẩn bị trong lúcR131 đang chạyUA.
User muốn PF ítIDSW hơnKF/EKF/UKF trênUA vàBDD. Không chọn video/seed hoặc đổi
detector/metric/association. DùngnativePF100R126, không trộnR128/129/130/131.

R126 tính R từ chiều cao detection hiện tại. Thử dùng chiều cao shape Gaussian
dự đoán trướcupdate để tínhR cho cảlikelihoodvịtrí/Josephupdate vàshapeupdate.
Khôngđổi hàmnoise/hệsố/confidencescaling. Vớiheightpred=heightobs, phải khớp
Nativeđúngmảng; vớiheightbịclipped/jitter, Rkhôngbịthu nhỏ trực tiếp theoobs.
Đây là state-dependent noise approximation thay thế observation-dependent R,
khônggọiposteriorchínhxác hoặc bảođảm tăngchấtlượng. R70 đã thử chiều caoR trên
PF cổ điển; chưa đủ để suy kếtquả cho CartesianRBPF hiện tại.

Researchcontext thay hàmnoise tạm trongmộtupdate, khôi phục cảkhi exception;
PF object dùngchung nhưngupdate tuần tự, khôngđược ròra track/lượt khác. YAML
gốc giữ `pf_measurement_height: observation` vì guardbackend R112; input nghiên
cứu ghi tườngminh `effective_measurement_noise_height: prediction`, hash code
vàprotocol. Không coi YAMLđơnlẻ táitạođược backend này hoặc đổi launcher.

Kiểm equal-height parity/PSD/khôiphục/hook/nativecontrol/repeat. UAtrain dev12
×3seed trướcBDD. CổngUA:meanIDSW<472,>=2seed<472,HOTA>=R123PF100-0,001,
FN<=1,002*R123. Đạt mớiBDD200×3seed/mot_crowd/cùngR125:
meanIDSW<5548,>=2seed<5548,HOTA>=R125PF100-0,001,FN<=1,002*R125PF100.
Đạt cảhai mớixácnhậnđủUA test40 vàđochiphí tuần tự. KhôngđổihệsốR/window
hoặc trộn haiR theođiểm test đểcứu. Giữmọilượttrượt; nêuthiênlệch lựa chọn
qua nhiềuvòng, không gọitestmù. Không sửa src/mặcđịnh/thesis.
