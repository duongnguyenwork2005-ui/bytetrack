# R172 — pruning tăng 8,75% FPS, nhưng chưa đạt chất lượng

Đã thu và chấm xong ngày 26/11/2026, từ lượt đo Nano ngày 21/11.
Cắt cấu trúc joint50 giúp mean FPS **20,47635 → 22,26731 (+8,74648%)**,
nhưng mAP50 **47,14673 → 40,90797%** và tracking xấu đi trên cả hai tập.
Không chọn cấu hình này để thay bản đang dùng.

## Giao thức và giới hạn huấn luyện

Hai nhánh lấy cố định checkpoint epoch6 (zero-based) của R164: bảy epoch
đầu hoàn tất liên tục, cùng 5.200 ảnh train, teacher/KD/geometry/budget.
Không dùng tail đối chứng10epoch, checkpoint resumed R169 hoặc chọn epoch
theo điểm dev mới. Đây là phân tích phụ common7; **thí nghiệm chính10epoch
chưa hoàn tất do lỗi CUDA**, không đổi nhãn đó thành thành công. Dữ liệu
phát triển đã được xem nhiều vòng, không phải test mù.

Joint50 cắt nửa kênh tại các vị trí được chọn trong body và head; không
phải giảm50% toàn mạng. Tham số2.590.815→2.019.391 (−22,05576%),
ConvMAC−24,09253%, ONNX10.500.936→8.218.728byte (−21,73338%).
Giữ đủ5lớp,3scale,đầu vào384×640 và FP16. Mọi bản dùng chung CUDA
preprocessing R162; chưa ghép tối ưu SiLU R167 vào phép so này.

## Tốc độ trên đủ80 đoạn

40BDD +40UA-DETRAC,150JPEG/đoạn,warmup25,KF,displayoff,không bỏ inference:

- Mean FPS original/control7/joint7: **20,47635 /20,25659 /22,26731**.
- Joint so original: **+8,74648%**; so matchedcontrol: **+9,92625%**.
- BDD:20,89611→22,74855FPS (**+8,86498%**).
- UA:20,05659→21,78608FPS (**+8,62303%**).
- Nhanh hơn80/80đoạn; FPS thấp nhất20,21918,80/80đạt15FPS.
- Aggregate FPS20,45741→22,24562; không trộn với mean FPS.
- Trung vị gain theo cặp8,75659%; bootstrap phân tầng40+40,5000lần,
  seed155,CI95%[8,70734;8,87193]. Đây không phải CI của mean FPS.
- Median-of-clip-medians GPUexec41,91915→38,14291ms;
  inference44,65785→40,87326ms. Trackerupdate9,94962→10,03458ms:
  chưa có bằng chứng tracker nhanh hơn. Không cộng các median vì decode
  chạy chồng lấp với xử lý.

Đây là FPS của80đoạn ảnh theo giao thức đã khóa; không phải FPS của toàn
bộ80video, camera trực tiếp hoặc cửa sổ HDMI.

## Detection trên7.190ảnh thực tế qua TensorRT

Original/control7/joint7:

- mAP50: **47,14673 /46,47791 /40,90797%**.
- mAP50–95: **31,98483 /31,22281 /26,67362%**.
- Joint mất6,23876điểm AP50 so original,5,56995điểm so matchedcontrol.
- AP50 joint theo lớp car/truck/bus/motorcycle/bicycle:
  **81,95602 /38,03466 /44,43605 /26,35918 /13,75391%**.
- Mọi cổng macro và AP50 từng lớp đều FAIL so cả hai mốc.

Suy giảm đã xuất hiện ở PC FP32 cùng geometry/postprocess, không quy
toàn bộ cho FP16. Raw tensor parity control/joint vẫn FAIL ở ngưỡng6px/.02;
không nới ngưỡng hoặc dùng PASS kiểm nguồn thay PASS chất lượng.

Phân tích phụ tại confidence.25/IoU.5 từ số đếm detection có sẵn:
original→joint TP44.059→43.060,FP10.395→14.004,FN14.301→15.300.
Recall xe máy40,70796→34,51327%, với226GT. Xe đạp có recall tăng nhẹ
22,11302→23,83292% nhưng precision40,72398→22,87736% và AP giảm;
không kết luận mọi suy giảm đều chỉ do bỏ sót. Đây là FN/FP detection,
khác FN/FP tracking bên dưới. Không tìm ngưỡng confidence mới.

## Tracking KF

BDD40, original/control7/joint7:

- IDSW **1534 /1657 /1643**.
- HOTA **43,71884 /43,94464 /42,33337%**.
- FN **28.539 /27.848 /29.136**; FP **2.194 /2.663 /2.517**.

UA40, original/control7/joint7:

- IDSW **245 /305 /401**.
- HOTA **66,57320 /65,63049 /63,63511%**.
- FN **17.590 /18.297 /19.168**; FP **9.144 /10.145 /9.406**.

So original, joint tăng109/156IDSW và597/1.578FN trên BDD/UA. So control,
BDD giảm14IDSW nhưng tăng1.288FN và giảm1,61127điểm HOTA; không gọi đây
là thắng tracking. UA so control tăng96IDSW và871FN. Chỉ chạy KF trong
phép đo này; không suy các con số đó sang EKF/UKF/PF.

## Toàn vẹn, kết luận và tệp bằng chứng

Scorer và summarize.py thoát0, đủ240lượt/36.000inference. Kiểm hash nguồn,
ảnh/config/frame/timestamp/MOT/CLS/GT và TrackEval đạt; original MOT/CLS
trùng byte R109 cả80đoạn. Cùng bootf52d349d-e4ce-4413-abee-d9f0f02dac2e
trước/sau phép đo,swap0,pswpin/outkhông tăng,SDerrors0. Boot mới lúc thu
26/11 không được dùng thay boot này. Archive23.620.003byte SHA256:
a62cae8257d3a76a094fafa07e1f723fa595d94d467fed17ad0b7095d7d892ed.

`final_summary.json` và `full80_evidence/runs_absolute/full/score_verified.json`
là các nguồn tổng hợp đã kiểm; `common_checkpoint_proof.json` giữ giới hạn
huấn luyện; `fixed_confidence_diagnosis.json` giữ phân tích c25.

R167 SiLU fusion vẫn là ứng viên đã đo tốt hơn về đánh đổi:26,90604FPS,
+31,44997% so original của chính lượtR167, AP5047,04839%. Đó là fusion
CUDA/engine, không phải pruning hoặc INT8. Không cộng phần trăm giữa các
lượt. R167 cũng có sai khác tracking và rawparityFAIL đã báo riêng.
Không thay default/launcher/khóa luận, không mở thêm training để cứu điểm.
