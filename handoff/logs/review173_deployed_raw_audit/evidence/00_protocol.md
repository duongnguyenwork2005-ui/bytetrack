# R173 — hậu kiểm tham chiếu tensor, không thay cổng R167

R167 rawparity so với fixture ORT FP32 (đã kiểm với PyTorch), không phải
so trực tiếp engine FP16 đang triển khai. Matched rebuild cũng FAIL tại
6px/.02. Khóa diagnostic sau kết quả full80: cùng10inputfloat32, chạy
original deployed/control/precise/fast đã có, không build/tune/train mới.

Đo raw so ORT FP32 và so original deployed, báo cả max toànanchor và các
bucket mà original hoặc candidate có score>=.001/.25. Cùng ngưỡng6px/.02,
không coi bucket khác là cách cứu FAIL. Không sửa R167 rawFAIL, actualAP,
full80 hoặc cơ chế promote. Không quy toàn bộ lệch FP32/FP16 cho SiLU.

Đây là posthoc diagnostic10fixture, không FPS/benchmark mới. Nano R167 đã
kết thúc; R172 còn chấm PC, chưa export/build. Chỉ một NanoGPU job, giữ
source/engine/plugin/fixture SHA, available>512MiB,swap<512MiB,SD0,nhiệt<75.
Không đổi default/launcher/khóa luận hoặc guard pipeline<100MiBswap.
