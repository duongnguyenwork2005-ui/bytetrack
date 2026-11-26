# R174 — kiểm CUDA Graph trên engine SiLU hiện tại

Người dùng26/11 yêu cầu SSH và tiếp tục tối ưu Nano. Đây là kiểm lại
kỹ thuật runtime đã thử trên engine cũ (C1), nay trên engine SiLU R167;
không thêm detector, trọng số, precision, training hoặc metric mới.

Không sửa engine/plugin/gói chạy cũ. Bốn chế độ cố định: original,
original+graph,preciseSiLU,preciseSiLU+graph. Cùng engineSHA của R167
cho mỗi cặp graph/off; FP16,384×640,pluginprecise giữ nguyênSHA.
Trtexec8.2 thực tại Nano có --useCudaGraph và có thể bỏ qua cờ khi capture
thất bại. Giữ log mọi warning/lỗi; không coi chỉ in cờEnabled là đã capture.
Đối chiếu source sampleInference/CUDA wrapper đi kèm để xác định dấu fallback.

Mỗi mode3lần, thứ tự thuận/đảo/đảo nhóm đã ghi trong screen.py trước điểm.
Giữ thông số R167 screen: warmUp500,duration6,iterations100; không spinwait,
không đổi clock/power. Screen GPU-only, không gọi throughput trtexec là FPS
toàn hệ thống. Không GPUjob khác; sameboot,SDerror0,swapstart<100MiB,
MemAvailable>512MiB,nhiệt<75°C; các guard lỗi là dừng và giữ bằng chứng.

Chỉ cân nhắc triển khai graph vào pipeline khi capture thành công, độ lệch
max/min median trong3lần<=2% và trung bình medianGPUtime giảm>=5% so cùng
engine không graph. Đây là cổng sàng lọc trước điểm, không nới để cứu.
Nếu trượt, kết thúc negative/không triển khai runtime. Nó không chứng minh
mọi workload đều không được lợi từ CUDA Graph. Nếu đạt, cần kiểm tensor
thực và đầy đủ detection/tracking/FPS theo giao thức cũ trước khi chọn.

Write_paths: logs/review174_silu_graph/, Nano~/nano_run/r174_silu_graph_20261126/,
PLANmục0. Không đổi defaults/launcher/khóa luận/cổngPF. Tối đa3job toàn hệ,
chỉ1jobGPU trên Nano; không chạy transfer/PC-to-Nano tác vụ nặng lúc timing.

Tài liệu NVIDIA: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/pdf/TensorRT-Developer-Guide.pdf
§CUDA Graphs; khả năng/cờ thực tế lấy từ --help8.2 và source cài trên Nano,
không áp dụng cú pháp enqueueV3/TensorRT11 vào TensorRT8.2.
