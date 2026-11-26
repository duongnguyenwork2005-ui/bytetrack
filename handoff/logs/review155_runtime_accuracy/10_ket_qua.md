# Tốc độ và chất lượng YOLO trên 80 đoạn BDD và UA-DETRAC

Giảm đầu vào TensorRT từ384×640 xuống320×576 tăng mean FPS từ
**18,804 lên22,354 (+18,88%)**, nhanh hơn trên cả80 đoạn. Tuy nhiên,
mAP50 giảm1,1624 điểm phần trăm và UA-DETRAC tăng45 IDSW, nên giữ
ứng viên này để mô tả trade-off, chưa chuyển vào cấu hình sử dụng.

## Phạm vi và đối chứng

40 đoạn BDD và40 đoạn UA-DETRAC theo đúng manifest đã khóa củaR109;
mỗi đoạn150 JPEG, bỏ25 frame đầu khi tính FPS ổn định. Cả hai nhánh
suy luận đủ150 frame, dùng cùng ByteTrack KF, ngưỡng0,25, NMS0,7,
ánh xạ lớp, mask, buffer, decoder overlap, tmpfs và chế độ không hiển thị.
Hai nhánh đo trong cùng boot; luân phiên thứ tự theo đoạn. Đây là FPS
trên các cửa sổ150 ảnh, chưa phải toàn bộ video/camera/HDMI.

160 lượt/24.000 inference được kiểm đếm. MOT và lớp của đối chứng
trùng byte R109 ở80/80 đoạn. Kiểm lại timestamp/FPS/p90, nguồn thực thi,
hash ảnh/engine/ONNX/GT, schema MOT7 cột và chấm lại TrackEval đều đạt.
Swap bắt đầu0, kết thúc49,49MiB; lỗi thẻ0, nhiệt dưới75°C.

## Tốc độ đo mới

- BDD40: mean19,6081→22,8475FPS, tăng16,5204%; nhanh hơn40/40.
- UA40: mean17,9998→21,8607FPS, tăng21,4498%; nhanh hơn40/40.
- Gộp80: mean18,8040→22,3541FPS, tăng18,8797%. Aggregate FPS tính
  theo tổng thời gian18,7622→22,3294; không trộn hai cách tổng hợp.
- Median mức tăng ghép cặp theo video19,4080%; bootstrap95%
  [19,0008;19,4979],5000 lần, seed155, mỗi mẫu vẫn40 BDD+40 UA.
- Cả hai đạt≥15FPS trên80/80 đoạn; ứng viên thấp nhất20,3202FPS.
  Median p90 latency theo đoạn121,671→104,627ms.

Profile median của các median theo đoạn: GPU YOLO41,952→33,497ms;
CPU tiền xử lý4,611→5,530ms; H2D1,748→1,378ms; suy luận gồm sao chép
44,137→35,268ms; cập nhật tracker10,876→10,257ms. Decode chồng lấp
xử lý nên không cộng các median thành tổng độ trễ. CPU tiền xử lý chưa
nhanh hơn; đây là phần đang thử chuyển sang CUDA trongR162.

## Đánh đổi chất lượng

Actual TensorRT trên7190 ảnh dev, cùng argmax/NMS/matching:
mAP50 **47,1467→45,9844%**; mAP50–95 **31,9848→30,9973%**.
AP50 car84,4577→82,9029%; truck44,0422→42,2498%; bus51,4063→48,6969%;
motorcycle34,4174→37,0599%; bicycle21,4100→19,0123%.
GT xe máy226 hộp và xe đạp407 hộp; chênh trên hai lớp nhỏ không đủ
khẳng định khả năng tổng quát. Ứng viên trượt cổng macro0,5 điểm và
từng lớp2 điểm đã đặt trước. Raw parity FAIL vẫn giữ nguyên.

- BDD: HOTA43,7188→42,7495%; IDSW1534→1518; FN28539→29304.
  Ít16 lần đổiID đi kèm nhiều765 bỏ sót, không coi riêng IDSW là thắng.
- UA: HOTA66,5732→65,8941%; IDSW245→290; FN17590→18571.
  Tăng tốc đồng thời giảm chất lượng tracking trên cùng cửa sổ đánh giá.

Không đổi engine mặc định, launcher, cấu hình luận văn hay metric.
Các phép thử đều dùng dữ liệu phát triển đã xem nhiều lần, chưa là kiểm
định mù. Mốc so tốc độ là original đo mới18,804FPS, không lấy mốc lịch
sử18,63FPS làm mẫu số.

## Bằng chứng và sửa lỗi công cụ

Kết quả đã xác minh:
`full80_evidence/runs_absolute/full/score_verified.json`;
`actual_AP_summary.json`; `selection.json`; `score_full_v2.log`.
Archive `full80_evidence_20261120.tar.gz`,17.323.656 byte,
SHA256 `86cf4727aa9b06731b73631c6c7036a6125c0aa8b92590f6cedef65d75fe2af1`.
PASS trong verifier xác nhận tính toàn vẹn, không biến quality FAIL thànhPASS.

Giữ log hai lỗi harness trước đo: đường dẫn fingerprint drivers sai và
output tương đối bị cwd tiến trình con đổi đích; sửa bằng runner version2
và đường dẫn tuyệt đối. Verifier đầu giả định MOT10 cột, đã giữ bảnv1/log
và sửa auditor theo schema7 cột thực tế. Không sửa track, thêm cột giả
hoặc đổi metric. Chi tiết ở notes02–05.
