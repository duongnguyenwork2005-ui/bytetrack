# R144 — nhiễu đo theo confidence chung cho các bộ lọc

Đã chạy đủ36lượt mới trên12video UA train dev, giữ nguyên cache/detection/classes/FPS/buffer và metric. KF/EKF/UKF đều nhận score hiện tại trong update bằng R(c)=max(1-c,0.05)R0, giống PF R142. Mốc fixed-R cũ giữ nguyên; không cố tình làm yếu KF.

- KF:449IDSW, HOTA60,7845%, FN35.172.
- EKF:439IDSW, HOTA61,2760%, FN35.065.
- UKF:439IDSW, HOTA61,1966%, FN35.072.
- PF R142:440,67IDSW (448/436/438), HOTA61,2376%, FN35.084,33; dùng lại36lượt sau hash và6prefix parity.

Cả EKF và UKF đã ít IDSW hơn KF. PF có2/3seed<=439 nhưng mean440,67>439 nên **chưa đạt cổng**. Mọi guardrail HOTA/FN đạt. Không mở BDD/UA test40/kịch bản trong R144, không đổi floor/gain/seed để cứu.

Numerical360bước direct-confidence parity/PSD/context ownership/exceptionrestore đạt;6 smoke controls routingoff khớp historical,6routingon repeat,6PF R142prefix parity đạt. Backend thực là Ultralytics đã cài; `dev/inputs.json` lưu đườngSTrack và hash. Full36MOT/classes/timing/marker/raw tái hợp đã kiểm.

Đây là dữ liệu phát triển đã xem nhiều lần, không phải test mù. Noiseadaptation là kỹ thuật bổ sung; không quy toàn bộ gain cho thuật toán suy luận. Bước R146 kiểm một cơ chế resampling PF riêng với cùngnoiseadaptation, giữ nguyên các mốc R144. Không đổi src/default/launcher/khóa luận. PLAN mục0 đã được cập nhật trước báo cáo.
