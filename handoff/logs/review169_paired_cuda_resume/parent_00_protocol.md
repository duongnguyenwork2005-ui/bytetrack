# R164 Cắt cấu trúc theo độ trễ thực và phục hồi ở hình học triển khai

Người dùng21/11 giao trực tiếp thử thêm các phương pháp giảm mô hình,
đánh giá FPS/detection/IDSW. R150 đã giảm10,73%MAC nhưng chỉ+0,62%FPS và
mất3,28 điểmAP50. R164 chọn cấu trúc bằng độ trễ GPU thực trước huấn luyện;
không dùng BDDval/UAtest hoặc điểm tracking để chọn cấu trúc/epoch.

Phạm vi ghi: handoff/logs/review164_latency_pruning/,models/r164_latency_pruning/,
tmp/r164_latency_pruning/,Nano~/nano_run/r164_latency_pruning_20261121/;
PLAN mục0. Không đổi nguồn/default/launcher/khóa luận hoặc giao thức cũ.
Tối đa3 logical Python job, không đo timingPC đồng thời training/quality.

## Bốn cấu trúc khóa trước điểm

Teacher/anchor YOLO11n50/50 SHA d6aad8667ea7d344592c5aa170a7dbcda968c4aa30930cecc08d3541566daa82.
Giữ shape384x640, đủ5lớp, cả3scale, output1x9x5040,reg_max16.
Slicing vật lý tensor, BN và các dependency; không chỉ zero weights.

1. head50: giảm nửa hidden channels ở regression cv2 và classification
   cv3 của cả3head. Cắt đúng DWConv/group/pointwise/BN nối nhau, giữ
   nguyên5class logits và64DFL logits, không bỏ scale nhỏ.
2. body50: giảm nửa hidden channels các Bottleneck/SPPF/C2PSA-FFN như
   kiểu slice đã kiểmR104; bảo vệ QKV/num_heads/output channels của block.
3. joint50: kết hợp head50+body50. Mức50% chỉ tại vị trí được chọn;
   báo params/ConvMAC thực, không gọi toàn mạng giảm50%.
4. depth2: bỏ module m cuối ở hai C3k2 có tổng thời gian profile cao nhất
   trong các block2/4/6/8/13/16/19/22, đồng thời slice cv2 input concat.
   Profile engine hiện tại trước khi mở điểm ứng viên. Fused layer có
   nhiều model.index được chia thời gian đều cho các index, báo coverage
   và coi attribution là proxy. Nếu không map được>=80% thời gian layer
   tên model.index, dùng ConvMAC các block làm fallback đã khai báo.

Chọn channel theo L2norm(convweight)*abs(BNscale), thứ tự index ổn định,
round xuống/tới bội8, tối thiểu8. Đây là slice thủ công có kiểm dependency,
không gọi là đã dùng thư viện DepGraph. Oracle so mỗi cấu trúc với bản
teacher có mask zero đúng các channel/weight tương ứng: tensorraw finite,
shape/classes/stride đúng, allclose box rtol1e-4/atol.02, score1e-4/1e-5.
Kiểm các mask trên ảnh thật train/monitor và random trước export; ONNX
checker+10CPUORT/PyTorch parity cho từng model. Lưu cả lỗi, không nới cổng.

## Đo tốc độ trước khi tốn thời gian train

Baseline engine13f8cd0f9777840052fc671a621dc0473a103eeecce71ce655b54c74a4ba9823.
Profile bằng trtexec/separateProfileRun; build ứng viên FP16 workspace512
cùngTensorRT8.2, lần lượt, không chạy build/capture/timingGPU đồng thời.
Screen chỉ GPU compute bằng CUDAevents,trtexec warm500ms/duration6s/
iterations100, thứ tự thuận rồi đảo. Ghi boot,nhiệt,SDerror,swap/pswp.
Không dùng FPS của model chưatrain vì ít detection sẽ giảm tải tracking.

Đây là GPU-only screen: không nới guard các protocol R150/R155/R160/R162.
Ghi swap cả trước/sau; chỉ đo khi MemAvailable>512MiB,swap<512MiB,
SDerror0,không jobGPU khác,nhiệt<75. Nếu2median/mode lệch>2% hoặcboot/
SD/nhiệt khôngđạt, đánh dấu screen không hợp lệ; chưa chọn training từ đó.
Không gọi con số này là FPS end-to-end. Pipeline sau cùng vẫn bắt đầu
swap<100MiB và dùng guard frozen của80đoạn, không suy FPStừparams/byte/MAC.

Chỉ cấu trúc GPU compute nhanh hơn anchor>=8% mới đủ điều kiện train.
Chọn tối đa2: cấu trúc nhanh nhất và cấu trúc có phần trăm ConvMAC giảm
nhỏ nhất trong nhóm đủ điều kiện(nếu khác), tie theo tên alphabet.
Nếu không cấu trúc nào đạt, kết thúc negative screen, không train để cứu.

## Phục hồi chất lượng và đánh đổi

Control không cắt và tối đa2student bắt đầu từ cùnganchor/cấu trúcinit,
cùng5200ảnhtrain củaR145(5000mixed5+200RideSafe khácngàytest), monitor300
đãkhóa, đủhash và không trùngbyte train/dev. Không lấy BDDval/UAtest hay
HELMET/SVMOT làmtrain. Rectangular training rect=True,imgsz640,batch4,
10epochs,SGD lr.002,warmup.5,mosaic/mixup0,ampFalse,seed164,workers0.
Với ảnh16:9, mục tiêu batch384x640 gần deployment; ghi shape thực từng
epoch, không giả định mọi ảnh đều16:9. Đây là thay cả geometry/dataorder,
không gọi ablation chỉ pruning soanchor. So matchedcontrol đểtách kiến trúc.

Teacher FP32 frozen, supervised +.5confidence-weighted classBCE +.25DFLKL,
cùng công thức/hệ sốR145. Best theo monitor300, báo đủ history; chấm
7190dev bằng exact argmax/NMS/runtimegeometry, báo5lớp/P-R/mAP50/50-95.
NếuOOM ghi lỗi, chỉ chuyểnbatch2 cho cảcontrol/student trước cóđiểm;
không âmthầm đổi epoch/dữliệu. Không tự kéo dài budget sau xemdev.

Build lại checkpoint đãtrain, rawparity riêng giữngưỡng6px/.02 vàFAIL,
auditactualTensorRT7190 cùngGPUpreR162 trước finalFPS. Qualitygate:
macroAP50/AP50-95 giảm<=.005, mỗiAP50class giảm<=.02 vsoriginal vàcontrol.
Giữ qualityFAIL đểbáotrade-off; khôngpromote. Dùng cùngCUDApre cho tấtcả
engine trong finalpilot4 rồi full80 theo rule pilotnhanhhơn original.
So baseline CUDA20,48 đã kiểm, không chỉ baselineCPU18,82; mỗi trial đo
original mới. ChấmHOTA/IDSW/FN/FPcùngTrackEval; báo đầyđủ mẫu số.

INT8: kiểm capability thực và supportmatrix trước. SM5.3 không có đường
INT8 inference đượcNVIDIA hỗtrợ; khôngtrainQAT hoặc gọi weight-onlyINT8
là FPS nếu engine không chạy INT8. Ghi bằngchứng và giới hạn thiếtbị;
FP16 hiện tại đã có, không coi néncheckpointFP16 là cải tiến compute mới.

Nguồn phương pháp:
[NVIDIA Nano precision](https://forums.developer.nvidia.com/t/onnx-tensorrt-engines-fp16-32/346691/7),
[DepGraph paper](https://arxiv.org/abs/2301.12900),
[Torch-Pruning tác giả](https://github.com/VainF/Torch-Pruning).
