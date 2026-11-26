# Kết quả tối ưu Nano trên 80 đoạn BDD và UA-DETRAC

**Nên dùng phương án tiền xử lý CUDA giữ nguyên engine384×640.** FPS
trung bình tăng **18,81983→20,47623 (+8,80136%)**, nhanh hơn trên80/80
đoạn; detection và tracking giữ nguyên. Pruning đang có và shape352×640
đều tăng tốc ít hơn và làm tăng IDSW, nên chưa chọn hai nhánh đó.

## Mẫu đo và kiểm tra

40 đoạn BDD và40 đoạn UA-DETRAC theo đúng manifestR109;150 JPEG/đoạn,
warmup25, ByteTrack KF, detector YOLO11n50/50, displayoff. Không bỏ khung
hoặc suy luận. Cùng nguồn ảnh, GT, threshold0,25, NMS0,7, lớp, buffer,
mask, decoder overlap, tmpfs và mã tracking. Đây là cửa sổ150 khung
trên80 video nguồn, chưa phải chạy hết từng video hay đo camera/HDMI.

Ba nhánh có160+240+160=560 lượt/84.000 inference; đối chứng được đo mới
trong từng nhánh và cùng boot với ứng viên. Source/hash/config/ảnh/GT,
timestamp/FPS/p90, schema MOT7 cột và TrackEval đều kiểm lại đạt. Mọi
MOT/CLS đối chứng trùng R109. PASS ở verifier xác nhận tính toàn vẹn;
quality FAIL của pruning/height352 vẫn được giữ nguyên. Ba pilot đều
nhanh hơn theo rule đã khóa nên đều mở full80; helper negative-pilot
chỉ được compile, không có phép thử negative-pilot phát sinh.

## CUDA giữ nguyên chất lượng

- BDD40: **19,60334→20,88571FPS (+6,54162%)**, nhanh hơn40/40.
- UA40: **18,03632→20,06674FPS (+11,25744%)**, nhanh hơn40/40.
- Chung80: **18,81983→20,47623FPS (+8,80136%)**. Aggregate FPS theo
  tổng thời gian18,77945→20,45778; không trộn với meanFPS theo đoạn.
- Median mức tăng ghép cặp8,60347%; CI95%[8,39662;9,00098],5000 lần
  bootstrap/seed155, mỗi mẫu vẫn40 BDD+40 UA. CI này áp dụng cho median
  paired gain, không phải meanFPS gain. Một lượt/model/đoạn nên CI không
  thay cho đo lặp nhiều phiên.
- Tất cả80 đoạn≥15FPS ở cả hai bản; ứng viên thấp nhất18,64787FPS.
  Median p90 latency theo đoạn121,67594→111,39042ms.

Kỹ thuật: một kernel CUDA gộp resize/letterbox114, BGR→RGB, HWC→CHW và
chuẩn hóa bằng cùng LUT256float32; upload BGR8bit, tái sử dụng buffer và
ghi trực tiếp vào input binding TensorRT. Giữ nguyên weights và engine
FP16 đầu vào384×640. Đây là tối ưu đường xử lý, không cắt mô hình.

Oracle44 tensor,20 raw và40 adapter đều trùng bit. Toàn bộ candidate
arrays trên7190 ảnh dev cũng trùng bit. Actual mAP50 **47,14673%**,
mAP50–95 **31,98483%** giữ nguyên; AP50 car/truck/bus/motorcycle/bicycle
**84,4577/44,0422/51,4063/34,4174/21,4100%**. Dev có226 hộp xe máy và
407 hộp xe đạp; đây là dev nội bộ đã quan sát, không phải test mù mới.

MOT/CLS trùng80/80 và TrackEval xác nhận:

- BDD: HOTA43,71884%, IDF1 52,96834%, IDSW1534, FN28539, FP2194 ở cả hai.
- UA: HOTA66,57320%, IDF1 78,37333%, IDSW245, FN17590, FP9144 ở cả hai.

Microbenchmark cùng engine/context: tiền xử lý+upload7,19135→2,37751ms;
GPU YOLO vẫn khoảng41,9ms. Trong pipeline, CPUpre về0 do chuyển việc
sang GPU; CUDApre nằm trong infer_ms. Không gọi toàn bộ tiền xử lý bằng0,
không cộng các median thành wall time vì decode có overlap.

## Pruning và distillation không đáng chọn ở cấu hình này

Nhánh cắt kênh+kèm distillation đã huấn luyện5 epoch, có matched dense
control cùng ngân sách. Params−10,62%, MAC−10,73%, nhưng GPU compute
41,54245→41,34075ms và meanFPS **18,82208→18,93809 (+0,61634%)**.
Nhánh pruned nhanh hơn74/80 đoạn. So matched control18,65206FPS,
pruned nhanh hơn khoảng1,53%; hai phép so có mẫu số khác nhau.

Actual mAP50 original/control/pruned **47,14673/46,17151/43,86807%**.
Pruned mất3,27865 điểm so original và2,30343 điểm so control; raw parity
FAIL giữ nguyên. Audit FP32 cũng cho pruned43,87884%, nên suy giảm lớn
đã xuất hiện trước đường TensorRTFP16; so hai backend chưa là ablation
chỉ đổi precision.

- BDD original→pruned: HOTA43,71884→43,30411%; IDSW1534→1644;
  FN28539→28457; FP2194→2436.
- UA original→pruned: HOTA66,57320→64,77225%; IDSW245→336;
  FN17590→19654; FP9144→8423.

Mức tăng chưa tới1%FPS đi kèm giảm detection và tracking. Chưa mở thêm
huấn luyện để cứu nhánh cắt kênh hiện tại.

## Shape352×640 cũng không phải phương án tốt hơn

GPU compute41,64825→39,29120ms, nhưng full80 meanFPS chỉ
**18,81030→19,03562 (+1,19786%)**, nhanh hơn41/80 đoạn:

- BDD **19,58920→19,30612FPS (−1,44507%)**.
- UA **18,03141→18,76513FPS (+4,06912%)**.

Profile BDD: CPUpre3,75252→7,41993ms, GPUexec41,97943→39,70926ms;
phần CPU tăng lớn hơn phần GPU giảm. Ở UA, CPUpre5,47981→6,09110ms
và GPUexec41,94007→39,68179ms nên có lợi tổng thể. Nhánh resize khác
khi rời kích thước giảm đúng một nửa là giả thuyết phù hợp với hình học
và profile; chưa có ablation riêng để quy toàn bộ chênh lệch cho nó.

Actual mAP50 **46,19008%**, giảm0,95665 điểm; AP50 bicycle giảm2,14273
điểm. Raw parity FAIL vẫn giữ. Tracking:

- BDD HOTA43,71884→43,66353%; IDSW1534→1577; FN28539→28411.
- UA HOTA66,57320→66,73017%; IDSW245→339; FN17590→17466.

HOTA UA tăng nhẹ đi cùng IDSW tăng94; không dùng riêng HOTA để gọi đây
là cải thiện khả năng giữID. Cả APgate và mục tiêu IDSW đều chưa đạt.

## Bằng chứng và trạng thái bàn giao

[Tổng hợp đã kiểm hash](D:/LABOTARY/bytetrack/handoff/logs/review155_runtime_accuracy/extra_final_summary.json),
[CUDA](D:/LABOTARY/bytetrack/handoff/logs/review162_cuda_pre/full80_evidence/runs_absolute/full/score_verified.json),
[pruning](D:/LABOTARY/bytetrack/handoff/logs/review150_nano_pruning/full80_evidence/runs_absolute/full/score_verified.json),
[height352](D:/LABOTARY/bytetrack/handoff/logs/review160_height352/full80_evidence/runs_absolute/full/score_verified.json).

Cả ba archive đã đối chiếu SHA256 với Nano. Boot51280481…, SDerror0;
swap đầu/cuối mỗi full lần lượt9,24609→9,13281MiB,9,13281→11,31250MiB,
11,31250→11,32813MiB, đúng guard bắt đầu<100MiB. Tất cả job tự mở đã
kết thúc. Gói CUDA thử riêng tại~/nano_run/r162_cuda_pre_20261120/package;
source nằm trong review162_cuda_pre. Launcher/default/khóa luận và
các bộ lọc chưa bị thay đổi bởi lượt thử này.
