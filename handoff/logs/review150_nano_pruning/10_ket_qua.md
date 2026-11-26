# Pruning YOLO trên Nano tăng rất ít FPS và giảm chất lượng

Đủ80 đoạn BDD/UA: nhánh cắt kênh+kèm distillation đạt
**18,82208→18,93809FPS (+0,61634%)** so original. Actual mAP50 giảm
**47,14673→43,86807%**, IDSW tăng ở cả hai tập. Không chọn nhánh này.

## Phép đo có đối chứng

40BDD+40UA,150JPEG/đoạn,warm25,ByteTrack KF,displayoff,full inference;
cùng ảnh/GT/mask/threshold/lớp/NMS/buffer/decoder. Ba model original,
matched dense control và pruned đều chạy đủ80:240 lượt/36.000 inference.
Control/pruned đều train5 epoch cùng ngân sách theoR145; control có
resume từepoch3, không cam kết RNG như chạy liên tục. Đây là cửa sổ150
khung trên80 video nguồn, chưa là full-video hay camera/HDMI.

## Tốc độ và chất lượng

- Original/control/pruned meanFPS **18,82208/18,65206/18,93809**.
- BDD original→pruned **19,59733→19,71853 (+0,61845%)**.
- UA **18,04683→18,15765 (+0,61406%)**.
- Pruned nhanh hơn74/80 đoạn; median paired gain0,59771%,
  CI95%[0,49713;0,77873] theo5000paired bootstrap/seed155.
- GPU mean2median original/control/pruned **41,54245/42,06360/41,34075ms**.
  Params−10,62%,MAC−10,73% chưa chuyển thành giảm thời gian tương ứng.
- ActualAP7190 ảnh: mAP50 original/control/pruned
  **47,14673/46,17151/43,86807%**, AP50–95 **31,98483/31,12498/29,28391%**.
  Cổng macro0,5 điểm/từng lớp2 điểm trượt. Pruned rawFAIL.035185>.02
  vẫn giữ; AP không sửa lại kết luận kiểm raw.

BDD original/control/pruned: HOTA43,71884/44,06853/43,30411%;
IDSW1534/1695/1644;FN28539/27799/28457;FP2194/2540/2436.
UA: HOTA66,57320/65,90643/64,77225%;IDSW245/273/336;
FN17590/18176/19654;FP9144/9456/8423.

So matchedcontrol, pruned nhanh hơn khoảng1,53%, nhưng AP50 mất2,30343
điểm. So original, tăng chỉ0,62% và UA có thêm91IDSW/2064 bỏ sót.
Không lấy giảm params/MAC làm bằng chứng tăng FPS, không kết luận
tác động kiến trúc chỉ từ so pruned với anchor khác lịch sử huấn luyện.

## Kiểm chứng và giới hạn

Host kiểm lại timestamp/FPS/p90, counters150inference/lượt, source/config,
hash ảnh/GT/engine/ONNX/MOT/CLS, schema7 cột và chấm lại TrackEval đạt.
Original trùngR10980/80. Swap9,13281→11,31250MiB,SD0,cùngboot51280481….
Archive24.129.917byte SHA256
`a099241467a28c15427f6f4da95b4bed3a9f5c9175e9e892c964ddb5cffb6d92`.

Nguồn: full80_evidence/runs_absolute/full/score_verified.json,
actual_AP_summary.json,speed_preflight.json và R163comparison.json.
FP32 pruned43,87884% cũng thấp; hai backend còn khác OpenCV nên chưa
là ablation precision thuần. Capture pruned từng ngắt5000 ảnh và resume
2190 với journal/hash, sau đó chấm đủ7190; không dùng partial làm kết quả.
QualityFAIL giữ nguyên; không đổi default/launcher/khóa luận.
