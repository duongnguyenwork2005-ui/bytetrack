# R169 — khôi phục đối xứng sau lỗi CUDA R164

R164 joint50 dừng illegalmemoryaccess ởepoch8 (zero-based7), sau291batch.
LogWindows cónvlddmkm13/153 cùnglúc, chưa xác định nguyên nhân. Giữ nguyên
R164source/log/runs/inputs. Checkpointepoch6 của cảcontrol/joint cònfinite,
cóoptimizer/scaler/EMA. FreshCUDAprocess thực hiện16-element test thànhcông.
Không cập nhật driver/registry/TDR, không thay clock hay cấu hình hệ thống.

Trước khi resume: syntheticKDforward/backward trên cảhai checkpoint với
CUDA_LAUNCH_BLOCKING=1, finite và đồng bộ GPU. Đây là diagnostic không có
epoch/checkpoint được lựa chọn. Giữ lỗi nếu tái diễn, không lờ exception.

Ultralytics checkpoint lưu EMA/optimizer chứ không giữ rawtrainingmodel và
RNG đầyđủ. Vìvậy không gọi resume là tiếp nối bit-exact. Để đối xứng, cảhai
arm bắt đầu lại từ **epoch6** (7epoch hoàn tất), cùng freshseed164, total10,
giữ SGD/lr/scheduler/data5200/monitor300/rect/batch4/KD củaR164. Chỉ train
3epoch7/8/9. Dùng cùng CUDA_LAUNCH_BLOCKING=1 cho cảhai, chỉ thay thứ tự
đồng bộ, không đổi math/precision/batch/budget. Phần epoch8 thấtbại bị bỏ,
không có trong checkpoint tiếp tục. Ghi rõ chi phí tính toán bịdiscard.

Primary paired comparison gồm7checkpoint cũ0..6 và3checkpoint resumed7..9
cho mỗiarm. Control cũ7..9/best/monitorselection giữ đểlưu vết, **không**
chọn giữa control cũ và mới theo AP. Chọn lại trên đúng10checkpoint/arm theo
monitor300 argmax/NMS384×640 AP50-95,tieAP50/earlier, trước dev7190. Không
đọc dev7190 để quyết định resume/epoch. Cảhai sốparam/ConvMAC không đổi.

Tệp mới ởlogs/review169_paired_cuda_resume vàmodels/r169_paired_resume,
NativeR169package riêng sauR167fullGPUđãxong. Không sửaR164source/default/
launcher/khóa luận/metric. CổngactualAP7190 vàpilot4/full80 cùngCUDApreR162
giữ nhưR164, qualityFAIL vẫnFAIL. Nguồn của pair mới sẽ khóaSHA trướctrain.
Tốiđa3logicalPythonjob; NanoR167timing tiếp tục độc lập, không chạy NanoGPU
build/capture khác đồngthời.
