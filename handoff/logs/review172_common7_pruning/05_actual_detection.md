# R172 — actual TensorRT detection: cắt tỉa chưa đạt cổng chất lượng

Đủ7.190 ảnh/engine, cùng source/GT/preprocess CUDA R162/NMS/AP của Nano.
`actual_quality_comparison.json` đã kiểm lại hash và nguồn quality; tốc độ
và tracking R172 **chưa đo** tại thời điểm báo cáo này.

- Original: mAP50 **47,14673%**, mAP50–95 **31,98483%**.
- Đối chứng7 epoch: **46,47791% /31,22281%**.
- Joint50 sau7 epoch: **40,90797% /26,67362%**.

Student giảm **6,23876 điểm phần trăm AP50** so original và **5,56995 điểm**
so matched control. AP50–95 giảm5,31121/4,54919 điểm tương ứng. Không quy
toàn bộ suy giảm cho FP16: trên PC FP32 cùng geometry/postprocess, mức
giảm đã hiện diện47,05447→40,98639%. Đây không phải ablation precision thuần
vì còn khác backend/preprocessing giữa PC và Nano.

AP50 theo lớp của student: car81,95602%, truck38,03466%, bus44,43605%,
motorcycle26,35918%, bicycle13,75391%. So original, giảm2,50171/6,00757/
6,97023/8,05818/7,65611 điểm. Tất cả cổng macro và từng lớp của student đều
FAIL so cả original lẫn matched control. Control7 cũng trượt macro và lớp
xe máy so original, nên cần giữ cả hai mốc trong phép so.

Mô hình thực sự giảm2.590.815→2.019.391 tham số (22,05576%), ConvMAC giảm
24,09253%. ONNX10.500.936→8.218.728byte (21,73338%). Không coi số byte hoặc
MAC là bằng chứng tăng FPS. Cổng GPU-only9,02% trước train là số đo khác.

Đây là phân tích phụ dùng cố định checkpoint6 từ hai arm R164,7 epoch đầu
liên tục, không chọn checkpoint theo AP, không dùng tail control10 hoặc
R169resumed. Primary10 thất bạiCUDA vẫn chưa hoàn thành. Dữ liệu dev đã xem
nhiều vòng, không test mù. Không chọn student này làm mặc định.

Actualcapture/build/source đã hoàn tất và speedstage đã tải lên Nano.
`timing_pending.json` chặn FPS vì swap310,625MiB, ngưỡng khóa<100MiB.
SSH không có sudo; đã chuẩn bị và kiểm cú pháp helper đưa trang swap vềRAM
rồi bật lại đúng swapfile/zram. Chờ người dùng chạy tại terminal, không
nới guard hoặc chạy timing trong điều kiện chưa đạt. Giữ lại chất lượng
FAIL nếu tiếp tục đo đủ80 đoạn để báo trade-off.
