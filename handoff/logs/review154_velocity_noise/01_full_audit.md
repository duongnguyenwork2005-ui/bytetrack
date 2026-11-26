# Audit cố định cảhai tập đầyđủ trước bước PF/kịchbản

Ghi trước điểmfull: dev E435/U430 soKF449 vàE/Ucũ439/439, componentpass;
PF R142440,67 chưađạt cặp mới. R153guidedPFdev437,67 đãqua cổng riêng
vớiGaussianR144 nhưng chưađo full. Khôngpromote PF từ auditGaussian.

Khóa cùng chínhsáchQ R154 cho cảBDD200 vàUA40, chạy480lượtGaussian mới
(400BDD+80UA), dùnglạiKFcommonR R148 vàPF R142 đểmô tả. Audit phảiđo
cảhai tập, không suyUA từBDD hoặc bỏUA nếu một cổng trênBDD trượt.
Source/params/cache/metric không đổi giữa hai tập hoặc sauđiểm.

Báo hai nhómcổng riêng: componentstrict như00_protocol (gồmIDSW không
tăng soGaussianR148), và prerequisite người dùng E<KF/U<KF với HOTA/FN
guardsoGaussianR148. Giữ cảkếtquảtrượt/cảsooldpolar; không dùngaudit để
nói Gaussian tốt hơn mọi mặt nếu cổngstrict trượt. Không làm yếuKF hay
đổiGaussian theo điểm để PF dễthắng.

Chỉ prerequisiteGaussian đạt trên cảhai tập mới xét kiểmPF đãqua dev
trên các fullreferences đãkhóa này; cổngPFmean<=cảhai/two-seeds/HOTA/FN
vẫnphảiđạt trước kịchbản. Audit không chạy/tunePF mới, khôngchọnseed,
khôngmởscenario tựđộng. Chỉ là khámphá trên dữliệu đãthấy nhiềuvòng.
