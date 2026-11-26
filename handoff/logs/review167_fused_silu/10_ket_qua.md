# R167 — 26,91 FPS trên 80 đoạn, tăng 31,45%, có sai khác tracking nhỏ

Đã hoàn tất và hậu kiểm 320 lượt, 48.000 inference trên 40 đoạn BDD và
40 đoạn UA-DETRAC. Mỗi đoạn 150 JPEG, bỏ 25 khung nóng, ByteTrack KF,
tắt hiển thị, không bỏ suy luận. Đây là FPS của giao thức đoạn ảnh đã khóa,
không phải toàn bộ 80 video hoặc FPS camera/HDMI thực tế.

## Tốc độ được đo trực tiếp

- Original: **20,46866 FPS**; matched rebuild control: **20,27988 FPS**.
- Precise expf: **26,90604 FPS**, tăng **31,44997%** so original.
- Fast __expf: **26,81362 FPS**, tăng **30,99841%**.
- Cả hai ứng viên nhanh hơn original ở **80/80 đoạn**; không loại đoạn thua.
  Mức thấp nhất tương ứng **24,09608 / 24,05143 FPS**, cả 80 đoạn đạt15FPS.
- BDD precise: **20,87494 → 27,59523 FPS (+32,19312%)**.
- UA precise: **20,06238 → 26,21686 FPS (+30,67673%)**.
- Median tăng theo cặp precise31,81125%, bootstrap phân tầng40+40,
  5000 lần seed155, CI95%[31,60529;31,91575]. Fast31,13826%,
  CI95%[31,06509;31,45277]. Đây là CI trung vị tăng theo cặp, không phải
  CI của meanFPS hoặc khoảng tin cậy test nhiều thiết bị.
- Aggregate FPS original/precise/fast20,45044/26,86686/26,77467.

Phương pháp: thay đúng77 mẫu x×Sigmoid(x) bằng một kernel CUDA cho mỗi
activation, plugin IPluginV2DynamicExt, vector half2 và format packed được
kiểm trên TensorRT8.2.1.9/CUDA10.2/sm53. Giữ initializer TensorProto nguyên
byte, shape384×640, đủ5 lớp và3scale. Class sigmoid cuối không bị thay.
Tất cả model dùng cùng gói loader R167 và CUDA preprocessing R162. Đây là
fusion phép tính, **không phải pruning, INT8 mới hoặc nén trọng số**.

GPU-only screen thuận/đảo original/control/precise/fast41,62130/42,08935/
30,28195/30,39035ms, giảm27,24%/26,98% thời gian so original. Trong pipeline
thực, median-of-clip-medians GPUexec41,91736→30,47261ms với precise,
infer44,63364→33,16931ms. Decode15,87830→15,38776ms, trackerupdate9,83970→
9,95261ms; không cộng các median vì decode chồng lấp. Lợi ích chính có ở
GPU inference, chưa có bằng chứng tracker được tăng tốc trong nhánh này.
R173 IEngineInspector cho348→280layer, nhưng Reformat62→68 và các tactic
convolution thay đổi. Đây là lợi ích của toàn bộ biến đổi graph/engine;
chưa tách phần tăng tốc riêng do kernel activation, layout hoặc tactic.

## Detection và tracking vẫn phải báo riêng

Actual TensorRT trên cùng7190 ảnh, evaluator/NMS/GT/source đã khóa:

- Original mAP50/mAP50–95 **47,14673 /31,98483%**.
- Matched rebuild **47,03259 /31,96109%**.
- Precise **47,04839 /31,95699%**: so original−0,09834/−0,02784 điểm %.
- Fast **47,13934 /31,94924%**: so original−0,00739/−0,03559 điểm %.

Cả hai đạt cổng macro giảm<=0,5 điểm % và mỗi AP50class<=2 điểm % so cả
original và control. AP50 precise/fast: car84,46057/84,44980;
truck44,02494/44,02977; bus51,41455/51,38250;
motorcycle33,98005/34,49072; bicycle21,36184/21,34389%.
GT theo lớp53204/3054/1469/226/407. Dữ liệu dev đã được xem nhiều vòng;
xe máy và xe đạp có mẫu nhỏ, đây không phải test mù.

KF trên40 đoạn BDD:

- IDSW original/control/precise/fast **1534/1543/1541/1539**;
  precise tăng7, fast tăng5 so original.
- HOTA **43,71884→43,80858 /43,70396%**.
- FN28539→28528/28542; FP2194→2175/2204.

KF trên40 đoạn UA:

- IDSW original/control/precise/fast **245/246/245/248**:
  precise bằng baseline, fast tăng3 so original.
- HOTA **66,57320→66,54156 /66,59263%**.
- FN17590→17597/17609; FP9144→9143/9199.

Không gọi tracking giữ nguyên hoặc mọi metric đều cải thiện. R167 chỉ chấm
KF, không ngoại suy mức tăng FPS/IDSW này sang EKF/UKF/PF.
Hậu kiểm theo video ở07_tracking_uncertainty.md: BDD precise11tăng/9giảm/
20bằng, CI95 deltaIDSW[−19;+35]; UA5tăng/7giảm/28bằng, CI[−12;+14].
Đây là bootstrap phát triển hậu kiểm, không chứng minh non-inferiority.

Raw tensor parity của **cả hai vẫn FAIL**, giữ nguyên ngưỡng6px/.02.
Tham chiếu ở phép kiểm này là ORT FP32 đã kiểm PyTorch, không phải engine
FP16 đang triển khai. Matched rebuild cũng FAIL17,03397px/0,013046; xem
R173 để tách hai tham chiếu. Không quy toàn bộ lệch FP32/FP16 cho SiLU.
Precise max box13,78397px/max score0,021777; fast17,04010px/0,013953.
Kernel toy oracle đạt không biến raw toàn mạng thành bit-exact. Tactic và
làm tròn FP16 có thể đổi đầu ra dù cùng weights/toán tử; AP nhỏ không xóa
FAIL. V1 compile namespace lỗi giữ nguyên log; v2 chỉ sửa namespace, hai
source precise/fast đã khóa trước điểm. Không tune kernel sau xem AP.

## Toàn vẹn và phạm vi triển khai

Native archive31.167.855byte, SHA256
34d38a1330dabbbe02ece5692b6991652e3041e7f94e13a4c7ea8e9257a2ae64.
Hostsession61179 thoát0; `final_summary.json` và
`full80_evidence/runs_absolute/full/score_verified.json` PASS hash/source,
timestamp, image/config, 320 lượt, 80 original MOT/CLS byte-parity R109 và
TrackEval. Samebootf52d349d-e4ce-4413-abee-d9f0f02dac2e, SDerror0,
swapcuối90,51953MiB, guard đầu đạt. Gov/min/max CPU/GPU trước–sau không đổi.

Đây là ứng viên thử nghiệm tăng tốc mạnh với thay đổi chất lượng nhỏ đã đo;
không tự promote/default, không đổi launcher hoặc khóa luận. Khi deserialize
engine cần đúng hai plugin `.so` đã ràng buộc SHA trong gói thử. So với R162
CUDApre, gain31,45% được đo cùng lượt/cùng boot ở đây; không cộng phần trăm
hoặc coi khác biệt giữa các nghiên cứu/boot là một phép đo ghép cặp mới.
