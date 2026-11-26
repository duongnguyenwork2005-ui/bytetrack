# R154 — phương sai hướng theo nhiễu vận tốc Cartesian

Trước điểm mới20/11: R148commonR có UAfull KF2319/E2359/U2372 nên
chưađạt prerequisite. R149CartesianGaussiandev461/476 đãloại. Giảthuyết
mới: polarQ hiện có nhiễu tốcđộh/160 nhưng theta cốđịnh.01rad, có thể
thiếu bấtđịnh vận tốc vuông góc khi xe đi chậm và gán jitter choomega.

Giữ nguyên stateCTRVcực/f/Jacobian/init2obs/stdomega/initomega/UKFweights,
Rconfidence/association/lifecycle. ChỉQtheta tại prior:
sigma_v=h/160; qtheta_min=(sigma_v/max(abs(v),sigma_v))²;
Qtheta=max(Qtheta_cũ,qtheta_min). Không có gain hoặccap đượcfit theoIDSW.
Mẫu sốsigma_v là giới hạn SNR=1 để khôngchia0, capvật lý1rad² ởtốcđộ
chưa phânbiệt với nhiễu; đây là approximation có chủđích, không tuyênbố
đúng Bayesian hoặc tươngđương môhình Cartesian khi v thấp.

Khi abs(v)>=sigma_v vàphysicalfloor chi phối, ánhxạJacobian từ(v,theta)
sang(vx,vy) đưa diag(sigma_v²,qtheta_min) thànhsigma_v²I. Giữ Qtheta_cũ
làmminimum khi lớn hơn. Không đổiQv/omega/kíchthước hoặcKF/PF, không
làm yếu Gaussian để PF thắng. Đây là cải tiến noise-model, khôngpure
so sánh EKF/UKF/PF chỉkhác thuậttoán suy luận.

Thamkhảo vấnđề phụthuộc tọađộ: [Roth etal.FUSION2014](https://liu.diva-portal.org/smash/get/diva2%3A734112/FULLTEXT01.pdf).
Côngthứcfloor trên là giảthuyết triểnkhai riêng, không gán cho tácgiả.

Numericaloff phảibitparity parentGaussian qua predict/update/confidence,
QPSD/finite/low-speed/rotation/Jacobianmatching và contextrestore trước
tracking. SmokeGaussianoldprefix/onrepeat/PFunchanged trên2video. Dev24
lượt E/U trước; cổng riêngcomponent: mỗi E/U<KF449 vàIDSW<=439của bản
cũ, HOTA>=samefilterR144-.001,FN<=1.002samefilterR144. Trượtđóng nhánh,
không dògain/cap/std đểcứu. Đạtcomponent mới audit E/Ufull BDD200 rồiUA40,
cùng cổng soR148commonR vàKF. Không đòi PFdev phải thắng để được kiểm
điều kiện riêngE/Ufull; PFđược giữ nguyên đểmô tả, khôngpromotechỉvìaudit.

ChỉkhiE/Ufull cảhai tập đạt thì xét PFứngviên đãqua dev, giữ cổngPFmean
<=cảE/U/two-seeds/HOTA/FN trướckịchbản như yêu cầu. Không mởscenario từ
componentdev hoặcchọn lại clip/seed/ngưỡng sauđiểm. Dữliệuđãthấy nhiều
vòng, kếtquả khámphá khôngtestmù. Khôngđổi src/default/Nano/khóa luận.
Tốiđa3job bềnvững kể cả GPUtrain; khôngbenchmarkPCtime cùngquality.
