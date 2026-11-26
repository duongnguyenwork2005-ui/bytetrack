# R167 — CUDA fusion Sigmoid×input, giữ trọng số YOLO

Phạm vi người dùng21/11 tối ưu Nano, thử độc lập trong lúc R164 train
pruning; không đổi launcher/default/khóa luận hay nguồn R164 đã khóa.
R164 profile anchor cho Sigmoid12,693ms, act/Mul4,738ms; 77 SiLU đang
thành hai lớp. Đây là attribution để đặt giả thuyết, chưa hứa speedup.

Giữ anchor R101,384×640,5lớp,3scale,output1×9×5040. ONNX thay đúng mẫu
y=x*Sigmoid(x) có Sigmoidoutput chỉ dùng một lần bằng plugin một kernel;
không thay sigmoid class, không đổi weights. TensorProto initializer phải
trùng byte. FP32: x*(1/(1+expf(-x))). FP16: tính sigmoid FP32, round HALF,
rồi multiply HALF (giữ intermediate rounding thay vì silently full-FP32).
Không gọi là pruning, lượng tử mới, nén weights hoặc bảo đảm bit-exact.

Plugin IPluginV2DynamicExt, native TRT8.2.1.9/CUDA10.2/sm53, F32 LINEAR;
HALF LINEAR/CHW2/CHW4/CHW16/CHW32/HWC8 cùngformat vào/ra. Elementwise trên
số phần tử vật lý có channelpadding, half2 có xử lý phần tử lẻ. No workspace.
Zero serialized state; giữ serializer/deserializer/namespace đúng ABI.
Tham chiếu API: header NVIDIA thực tế trên Nano và
https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/plugins-api-migration.html
(phần V2, không sử dụng V3 của bản mới cho Nano8.2).

Trước enginepoints: kiểm kernel trực tiếp F32/HALF với0,±small/±60 và random,
size1/3/17/1025/1048577, finite, allclose F32 rtol2e-6/atol1e-6 và HALF
rtol.002/atol1e-4 so NumPy float64→F32/HALF nested rounding. Kiểm graph/
weights/count77, load/serialization actualengine,10 realraw inputfixtures
so originalORT/PyTorch, giữ tolerance6px/.02 vàFAIL, không đổi math để cứu AP.

Build matched control từ ONNX gốc và fused bằng cùng FP16/workspace512;
so cả original deployed và matched rebuild. GPU-only thuận/đảo warm500ms,
duration6s/iterations100,repeat<=2%,sameboot/SD0/temp<75/avail>512MiB/
swap<512MiB; không song song NanoGPU job. Chỉ fused GPUtime giảm>=8% vs
original mới đi tiếp actualTensorRT7190 và fullpipeline trade-off; nếu
trượt đóng negative screen. Không train/tune coefficients/precision sauđiểm.

ActualAP7190/P-R/5class, caps macro.005/perclassAP50.02 so original và
matched rebuild. Source/dataset/engine/plugin .so khóa SHA; actualCUDApre
R162 cho tấtcả. Finalpilot4 rồi80nếufusednhanhhơn original, giữmatchedcontrol;
KF/full inference/warm25/JPEG/displayoff/metric frozen và swapstart<100.
Ghi raw/quality FAIL dù đo trade-off; không promote. Plugin phải được load
trong mọi process deserialize, qua gói thử/LD_PRELOAD có manifest; không
thay thư viện hệ thống. Tốiđa3 logical Python job, không đo PCtiming khitrain.
