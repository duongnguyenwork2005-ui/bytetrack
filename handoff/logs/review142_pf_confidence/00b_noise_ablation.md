# Bổ sung phân tích parent để tách hiệu ứng confidence

Không thay đổi cổng/metric/model/cấu hình đã khóa ở00_protocol.md. Ngoài các
controlsKF/EKF/UKF/fixedRNative100, đối chiếu trực tiếpR142 vớiR132Parent100
(cùngNative100/Rprediction, chỉthiếucurrentscoreadapter). UA test40helper đãcó
parent100. VớiBDD, dùngba rawR132 đãđượckhóa trongR132final_verification.json
vàđượckiểm lạihash; không chạy lại để chọn kết quả.

KhiBDDqualityhoàn tất, táihợp rawcủacảhai, meanba seed0/1/2, pairtheo200video.
Bootstrap5000mẫu, seed128nhưbootstrapBDDchính; HOTA/IDF1/IDSW/FN/FP vàdelta
R142-R132. Khôngđổibootstrapseed/ngưỡngtheoCI. CóthểchạyphântíchnàysauUAquality
đểkhôngvượt3Pythonjob; phải xongtrướcđotimingnếuR142đạt.

R142làPF+Rthíchnghiconfidence. BaselineKF/EKF/UKFvẫnfixedR; bấtkỳưu thếquan
sátnàođều bao gồmnoiseadaptation, khôngchứngminhPFthuầnưu việthơn khi mọi
filter dùngcùngR(c). Khôngsửa mốcbaseline hoặc âmthầmđổinhãnso sánh.
