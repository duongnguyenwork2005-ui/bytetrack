# R168 — giảm phương sai lấy mẫu normal, không thêm hạt

Khóa trước điểm: parent guidedR153, không chồng MH R165 đã trượt. Lấy mẫu
Latin hypercube một chiều ngẫu nhiên cho angularinnovation ở predict và
incomingomega guidedproposal ởupdate. Vẫn N100,Q/R/init/ESS/association,
importancecorrection p/q và tất cảGaussianreference R154435/430/KF449.

u_i=(permutation_i+Uniform_i)/N, z_i=Phi_inverse(u_i). Random permutation
gán strata độc lập với ancestor/weights, nên từng particle có marginal
normal trong số học lý tưởng; giữa các hạt có phụ thuộc. Không gọi là full
SQMC/Hilbert sorting hoặc bảo đảm phương sai mọi estimator/tracking thấp.
Clip đúng endpoint nextafter(0,1)/nextafter(1,0) để tránh inversePhi vô hạn
do floatingpoint; không tune cutoff. Hai batch predict/update độc lập.
Nguồn LHS: McKay/Beckman/Conover1979,
https://doi.org/10.1080/00401706.1979.10489755.
Phân biệt SQMC: https://arxiv.org/abs/1402.4039.

Chỉ thay standard_normal(N) trong predict/update đủđiều kiện; không đổi
drawinitialization/measurementnoise/resampling. Zeroomega variance dùng
parent, không thêm draw. Proxyrng phải restorefinally, cảlỗi. Không dò
ordering/strata/jitter/N/seed sau điểm, không tăng sốseed để cứu gate.

Numerical: off bitparity arrays/RNG/Lost/no-predict/motioninit; strata đủ
1 mỗi tầng vớiN1/3/17/100; finite/normal marginals độc lập qua4000batch;
samplemean variance soIID (không suy tracking); evidenceGaussian đóng so
4000batchimportance p/q; actualPSD vàzero variance/hookrestore. Smoke
offprefix/onrepeat2video×3seed, khóa source/cache/GT/refs trước dev36.

Gate meanIDSW<=430 và>=2seed<=430,HOTA>=R153-.001,FN<=R153*1.002. Trượt
đóng nhánh. Đạt chỉ component; GaussianfullUA prerequisite chưa đạt nên
không full/transfer. Không sửa src/default/khóa luận/Nano. Dev tái dùng,
khôngblindtest. Tốiđa3logicalPythonjob (R164train,R167Nano,R168CPU), không
đoPCtiming đồngthời train. Đây là đổi samplingproposal, khác R140/146
stratifiedresampling đã thử.
