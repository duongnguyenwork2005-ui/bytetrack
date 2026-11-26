# R159 — Auxiliary ancestry trên guided Rao–Blackwell PF

Khóa trước điểm. Parent R153guidedN100, không trộn R156/R157. R117 đã
thử APF trên full-state500 diagonal Q, chưa đạt; đây là áp dụng khác lên
conditional Gaussian Cartesian + proposal incoming-omega của R153, không
gọi một ý tưởng chưa ai thử hoặc lặp nguyên thuật toán R117.

Ở update có cache prior và incoming omega variance>0, tính xấp xỉ
predictive likelihood m_i bằng Gaussian vị trí đã marginalize tuyến tính
incoming omega: S_i=Pxy_i+Rxy+prior_var*J_i*J_i'. First-stage weights
proportional w_i*m_i; chỉ chọn ancestor bằng systematic khi ESS<.5N,
đúng ratio cũ. Sao chép conditional prior means/covs/incoming_mu và eta
độc lập hiện tại theo ancestor. Proposal omega vẫn R153. Second-stage
logweight = -log(m_selected)+log(prior/q)+exact conditional loglikelihood.
Giữ hiệu chỉnh này, không coi m_i là likelihood thật khi CT phi tuyến.
Nếu không auxiliary-resample, đường update giữ nguyên R153. Không dùng
GT/frame tương lai, không đổi association, Q/R/init/N100/shape/estimate.

Đây là auxiliary PF theo nguyên lý [Pitt và Shephard1999](https://www.nuffield.ox.ac.uk/economics/papers/1999/w13/pittshephard.pdf),
kết hợp riêng với RBPF của dự án, không mã ByteTrack do tác giả cung cấp.
Ý nghĩa của first-stage và second-stage weights đối chiếu bài gốc.

Numerical: disabled parent bitparity/RNG, log-predictive Gaussian độc lập,
linear Gaussian exact-proposal second-stage weights constant dù ancestor
khác nhau, actual CT ancestry/cache/eta checks và enabled PSD, zero-var
fallback, hookrestore. Smoke parentoff 2video×3seed vàonrepeat; sau đó
36dev12×3seed. Khóa toàn source/cache/GT/YAML/refs trước điểm.

Refs mạnh R154dev E435/U430, KF449; meanPF<=430, ít nhất2seed<=430;
HOTA>=parentR153-.001, FN<=parent*1.002. Trượt đóng nhánh, không dò
ESS/lookahead scale/N/seed. Cả khi dev đạt cũng không full/kịchbản vì
Gaussianfull prerequisite còn trượt UA. Không sửa src/default/launcher/
khóa luận; không PC timing cùng quality, tốiđa3job. Dữ liệu đã quan sát
nhiều vòng, kết quả khám phá có selection bias, không blindtest.
