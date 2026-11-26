# R174 — CUDA Graph giảm chi phí enqueue, GPU SiLU chỉ giảm0,43%

Ngày26/11/2026 đã chạy đủ12lượt trên Nano vừa khởi động lại: original và
preciseSiLU R167, mỗi engine có graph/off, mỗi chế độ ba lần. Tất cả dùng
cùng trọng số/engine/plugin cũ,384×640/FP16, không rebuild hoặc train.

## Kết quả đo

Các số sau là trung bình của ba median trtexec, không phải FPS pipeline:

- Original GPU41,63740→graph41,33893ms: **giảm0,71682%**.
- PreciseSiLU GPU30,28003→graph30,14980ms: **giảm0,43010%**.
- Original enqueueCPU15,81697→1,26703ms: giảm91,98942%.
- PreciseSiLU enqueueCPU11,09810→1,50134ms: giảm86,47207%.
- End-to-End Host Latency trtexec: original41,95523→41,66340ms;
  precise30,59650→30,47047ms. Đây vẫn là phép đo engine của trtexec,
  không gồm decode/preprocessing/NMS/tracker của ứng dụng.

Enqueue giảm lớn nhưng GPUtime thay đổi nhỏ. Không cộng thời gian enqueue
vào GPUtime để suy ra phần trăm tăngFPS, vì việc gửi lệnh và thực thi có
thể chồng lấp. Kết quả không chứng minh mọi workload đều không được lợi
từ CUDA Graph, cũng không có phép đo FPS ứng dụng để khẳng định lợi ích0%.

Cổng khóa trước điểm yêu cầu capture hợp lệ, repeatspread<=2% và giảm
ít nhất5%GPUtime so chính engine không graph. Cả hai cặp ổn định nhưng
trượt cổng5%, nên kết thúc screen, **không triển khai graph vào pipeline**.
Không hạ cổng sau khi xem điểm hoặc gọi đây là tăngFPS mới của hệ thống.

## Capture và tính toàn vẹn

Tệp `verification.json` tái tính từ cả12log, kiểm thứ tự thuận/đảo/đảo nhóm,
engine/plugin/source/protocol/bindingsSHA, thông sốwarm500/duration6/
iterations100, version8201, PASSED và toàn bộ guard. Các loggraph cóEnabled,
khôngwarning/error/fallback. Source `sampleInference.cpp`/`sampleDevice.h`
đi kèm cài đặt chỉ ra nhánh capture/enqueueGraph và warning khi bỏ qua cờ;
enqueue giảm lớn là bằng chứng bổ sung. **Chưa có trace Nsight độc lập**.

GPU repeatspread original/originalgraph/precise/precisegraph tương ứng
0,20437/0,14720/0,07765/0,08925%, đều dưới2%. Cùngboot
`dccc433e-ae31-4ca5-9e13-0d89f0f00195`, giữ nguyên CPUgovschedutil và
GPUmin76,8/max921,6MHz; không chỉnh power/clock. Swap0,MemAvailable>512MiB,
SDerrors0 ở mọi snapshot, nhiệt tối đa50°C. Cuối lượt không còn jobGPU
do agent mở. Không đo chất lượng detection/tracking cho chế độgraph,
không suy bitexact chỉ từ việc dùng chung engine.

Screen đã hoàn tất trước lỗi đóng gói tarv1 (`.: file changed as we read it`).
Giữ nguyên archivev1/source/log. `collect_v2.ps1` chỉ đóng gói danh sách
tệp cố định, không chạy lại12benchmark hoặc sửa cổng; archivev2 33.909byte,
SHA256 `a34605cd2daea5e8e87da5b02707531027e2663d276cae8d77d15fa1a3a4d477`.
Chi tiết ở `01_collection_failure.md`.

## Quyết định hiện tại

R167 CUDA preprocessing+SiLU vẫn là ứng viên đã đo FPS80 tốt nhất trong
các nhánh này:26,90604FPS, mAP5047,04839%; có thay đổi tracking nhỏ và
rawtensorparityFAIL đã ghi riêng. R172pruning22,26731FPS nhưng mAP50
40,90797% và IDSW tăng sooriginal trên cảBDD/UA, nên không chọn.
Không cộng gain giữa các lượt hoặc gọi SiLUfusion là nén trọng số.

Không thay defaults, launcher, khóa luận hoặc cổng thí nghiệmPF. Primary
training10epoch của pruning vẫn chưa hoàn tất; R172chỉlà common7secondary.

Nguồn phương pháp: [NVIDIA TensorRT Developer Guide8.4.3, CUDA Graphs](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-843/pdf/TensorRT-Developer-Guide.pdf).
Cờ và nhánh capture thực tế được đối chiếu với trtexec8.2 và source cài
trên Nano; không áp dụng cú pháp TensorRT11/enqueueV3.
