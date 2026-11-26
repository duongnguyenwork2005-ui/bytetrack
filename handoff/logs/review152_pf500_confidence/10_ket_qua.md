# R152 — tăng N100 lên N500 không giảm IDSW trung bình

36lượt dev12×3seed đủ16919ảnh/seed. Giữ kernelR142 cùngconfidenceR,
Q/init/ESS.5/systematic/mean/shapeLost/association; chỉ sốhạt tăng500.
Guard thực tế trên từngparticleobject xác nhận500, không chỉ đổiYAML.

PF500 IDSW445/437/440, mean440,67; PF100R142448/436/438, mean440,67.
KF/EKF/UKF commonR R144449/439/439. Chỉ1seed<=439, mean>439: trượt
cổngIDSW/two-seeds. HOTA61,2308% soPF10061,2376%, FN35071,33 so35084,33;
cảhai guardrails đạt. Không đủ bằngchứng chỉ tăngN giải quyết được vấnđề.

Numerical600directparitysteps500/RNG/PSD,160CVidenticalcomponentsteps
100/500 vàfail-loudcountguard đạt. Smoke Gaussianoff/repeat vàparentPF100
prefixparity,PF500repeat6case đạt. 36MOT/CLS/timing/marker,rawrecombine
vàfrozeninput/sourcehash kiểmđạt trongdev/verification.json vàinputs.json.

Không mở full/kịchbản, không chọn seed/tăngN tiếp hoặc đổiESS đểcứu.
Không đo timing khi PC đang training/chấm model, nên không suy chi phí
tăng gấp5 hay FPS từ sốhạt. Giữ PF100 hiện tại, thử proposal có quan sát
ở R153 sau numerical/oracles, không trộn thêm thayđổi vào thí nghiệm này.
Không sửa src/default/Nano/khóa luận. Kếtquả dev đã được quan sát nhiều
vòng, không gọi test mù hoặc chứng minh cơ chếMonteCarlo không quan trọng.
