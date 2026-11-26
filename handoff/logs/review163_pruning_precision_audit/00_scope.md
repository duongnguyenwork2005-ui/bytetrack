# R163 — tách đường chấm FP32 và engine pruning

R150actualTensorRT7190 đã thấypruned AP50.438681, control.461715,
original.471467. R145standard Ultralytics val dùng multi-label/NMS khác
đường argmax của Nano; không lấy hai protocol khác nhau để quy toàn bộ
mức giảm cho FP16 hoặc pruning.

Chấm control/pruned FP32 trên PC, cùng384x640, argmax/c001/IoU.7/max300,
GT/7190ảnh/matching/P-Rrematch bất biến, tái dùng evaluate củaR151.
Không huấn luyện hoặc sửaweights/params/detection. So vớiR151baseline
FP32Nano-path vàR150actualTensorRT để chẩn đoán, không mở lại APgate.

Giới hạn: OpenCV trênPC vàNano khác phiênbản (Nano3.2.0); PyTorchFP32
vàTensorRT khác backend/fusion. Vì vậy đây không phải ablation chỉ thay
precision. Không gọi thời gianCPU làNanoFPS. Hash source/weights/data
trước/sau, kết quả riêng từngmode; tốiđa3job bềnvững.
