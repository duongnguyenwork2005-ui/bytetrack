# R143 — chuyển nguyên PF R142 sang các tập kịch bản

Yêu cầu người dùng19/11/2026: chạy test trên các tập kịch bản. Giữ nguyên R142,
không chọn/tune tham số từ điểm mới. Chỉ tracking từ cache đã có, không cần đọc
ảnh trên ổ F hiện không gắn. Không thay detector, src, cấu hình mặc định, Nano,
slide hoặc khóa luận. Đây là kiểm chuyển cấu hình, các tập đã được dự án xem
trước; không phải holdout mù.

## Phạm vi khóa trước kết quả

- RoundaboutHD: đủ4 camera prefix1–3000,15fps, buffer18, YOLOv8n mixed5 c25,
  classes[1,2,3,5,7], max_det300, cùng GT SCT đã bỏ đúng16 dòng trùng R64.
- SVMOT: đủ9 chuỗi test/8.250 ảnh,30fps, buffer36, YOLOv8n COCO c25,
  classes[1,2,3,5,7], max_det300, full-image R64, giữ native visibility.
- HELMET: đủ36 clip đã chọn trước/3.600 ảnh,10fps, buffer12, YOLO11n mixed5,
  classes[3], max_det300. Chạy **cả c25 và c10**, giữ riêng hai điều kiện,
  không chọn ngưỡng thắng rồi bỏ ngưỡng còn lại.12 nhóm cảnh,3 clip/nhóm.

Mỗi điều kiện chạy mới KF/EKF/UKF fixed-R (mỗi loại1 lượt) và R132 PF100
fixed-R/R142 PF100 confidence-R (mỗi loại3 seed0/1/2):9 nhóm. Tổng765 lượt
video:36Roundabout+81SVMOT+324HELMETc25+324HELMETc10. Tracking elapsed steps
như R64/R66; ảnh cache liên tục nên bước chuyển trạng thái là1 khung.

Không đòi mọi tập đều thắng mới báo. Báo HOTA/IDF1/IDSW/FN/FP, từng seed,
từng camera/chuỗi/nhóm; đánh dấu IDSW giảm kèm FN tăng hoặc HOTA giảm.
Không gộp điểm tuyệt đối giữa các tập. PF mới có thêm R theo confidence,
controls KF/EKF/UKF vẫn fixed-R: không kết luận pure-PF superiority.

## Trước full

Khóa hash runner, dependency/PF native C/DLL, mã STrack thực nạp, metric,
config, cache NPZ/JSON/meta/marker, GT/seqinfo/gt_log, manifest và baseline
MOT/raw/certificate. Cache phải khớp timeline,FPS,classes,source GT và không
unreadable/truncated. Kiểm đúng36clip HELMET và số box/ID đã chuyển đổi.
Không sinh lại cache hoặc thay nhãn để cứu kết quả.

Smoke ở chuỗi đầu/cuối mỗi điều kiện,64 ảnh đầu:3 bộ lọc deterministic phải
khớp MOT lịch sử; PF route-off khớp parent R132 ở3 seed, R142 cùng seed lặp
trùng MOT/lớp. Chỉ sau preflight và smoke PASS mới full. Không quá3 Python
job bền vững đồng thời. Full lưu MOT/lớp/timing/marker; resume chỉ sau kiểm
hash, replay từ đầu video chưa có marker.

## Chấm và phân tầng

TrackEval MOT HOTA/CLEAR/Identity, giữ helper kết hợp đủ thống kê R66
`sher_metric.combine`; đối chiếu COMBINED với output TrackEval trực tiếp
và mọi metric deterministic với lịch sử. Không trung bình HOTA các clip.
Tổng PF là trung bình COMBINED ba seed, không cộng hoặc xem seed như video.

- Roundabout: dùng nguyên GT image-space turn25/turn35/straight8 R99,
  không chọn lại góc hoặc camera theo điểm. Phát lại CLEAR toàn chuỗi trước,
  xác nhận IDSW/FN khớp rồi mới đếm sự kiện trong mask; không reset ở đoạn rẽ.
- SVMOT: native visibility<=0,5; phát lại CLEAR toàn chuỗi, kiểm tổng IDSW/FN.
  Đây là độ nhìn thấy trong nhãn, không mặc nhiên chứng minh vật che cố định.
- HELMET: toàn sample là xe máy, box thường bao cả người ngồi. Không đặt tên
  các clip thành che khuất/rẽ nếu chưa có nhãn tương ứng.

HELMET bootstrap ghép cặp5000 mẫu theo **12 nhóm cảnh**, seed143, mỗi lần
lấy lại đủ3 clip của nhóm, tính COMBINED rồi meanPF3seed. Báo CI HOTA,IDF1,
IDSW,FN so KF/EKF/UKF/parent. Roundabout là4camera cùng nút giao, SVMOT9clip
từ2scene: chỉ mô tả, không dùng CI video giả độc lập. Mọi CI có giới hạn do
các tập đã được quan sát nhiều vòng. Không timingbenchmark từ job song song.

Kết thúc: kiểm lại hash/765marker/raw/score, cập nhật PLAN mục0 trước báo cáo.
