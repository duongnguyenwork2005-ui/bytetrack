# Giảm input xuống352×640 chưa mang lại lợi ích phù hợp

Đủ80 đoạn: meanFPS **18,81030→19,03562 (+1,19786%)**. BDD chậm1,45%,
UA nhanh4,07%; mAP50 giảm0,95665 điểm và IDSW tăng ở cả hai tập.
Không chọn shape này thay384×640.

## Phạm vi và kết quả

Cùng YOLO11n50/50 weights; build engine352×640 FP16 workspace512.
Giữ letterbox đúng tỉ lệ, không kéo méo/crop/skip. Đối chứng mới cùngboot,
40BDD+40UA,150JPEG/đoạn,warm25,ByteTrack KF,displayoff,full inference.
Đủ160 lượt/24.000 inference. Mẫu là cửa sổ150 khung trên80 video nguồn,
chưa phải full-video/camera/HDMI.

- BDD **19,58920→19,30612FPS (−1,44507%)**, nhanh hơn1/40 đoạn.
- UA **18,03141→18,76513FPS (+4,06912%)**, nhanh hơn40/40.
- Chung80 **18,81030→19,03562FPS (+1,19786%)**,41/80 nhanh hơn.
  Aggregate18,77066→19,02108FPS. Median paired gain1,44950%,
  stratified CI95%[1,20444;1,58381],5000bootstrap/seed155.
- GPU mean2median **41,64825→39,29120ms**; không phải FPS pipeline.

BDD CPUpre **3,75252→7,41993ms**, GPUexec **41,97943→39,70926ms**.
UA CPUpre **5,47981→6,09110ms**, GPUexec **41,94007→39,68179ms**.
CPU tăng lớn ởBDD đã lấn lợi íchGPU; ởUA phầnCPU tăng nhỏ hơn nên
còn lợi tổng thể. Nhánh OpenCV khác khi resize rời hệ số giảm đúng một
nửa là giả thuyết dựa hình học/profile; chưa là ablation xác nhận cơ chế.
Decode overlap nên không cộng median từng bước thành wall time.

## Detection và tracking

Actual7190 ảnh: AP50/AP50–95 **46,19008/31,76984%** sooriginal
**47,14673/31,98483%**. AP50 bicycle21,41002→19,26730%.
Trượt cổng macro0,5 điểm và từng lớp2 điểm. Raw parity FAIL ởsample04:
score.020578891>.02,box.325195px; giữ log và ngưỡng.

- BDD HOTA43,71884→43,66353%;IDSW1534→1577;
  FN28539→28411;FP2194→1718.
- UA HOTA66,57320→66,73017%;IDSW245→339;
  FN17590→17466;FP9144→8635.

UA có HOTA tăng nhẹ và ít FN/FP hơn, đồng thời thêm94 lần đổiID.
Không dùng riêng HOTA để kết luận giữID tốt hơn. Với mục tiêu detection
năm lớp và giữID, mức tăngFPS chung1,2% chưa đáng chọn.

## Bằng chứng

full80_evidence/runs_absolute/full/score_verified.json xác minh source/
config/ảnh/GT/hash/counters/timestamp/FPS/p90/MOT7 cột/TrackEval.
Original trùngR10980/80;swap11,31250→11,32813MiB,SD0,cùngboot51280481….
Archive16.358.359byte SHA256
`c25a283fac926a521e651769315f06fab958f65ad870b544f52b1a504e0315c2`.
00_protocol.md đã khóa trước điểm; pilot4 dương nên mở đủ80 theo rule,
không bỏ đo khi APFAIL. Chưa đổi default/launcher/khóa luận.
