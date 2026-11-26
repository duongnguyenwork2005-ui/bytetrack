# R88 — Urban Tracker: giao thức trước chạy

Hai cảnh René-Lévesque và St-Marc đã tải/kiểm ZIP CRC tại R69. Mục tiêu là khảo sát chuyển miền cho ByteTrack CV/EKF/UKF/PF trên giao thông hỗn hợp, ưu tiên xe đạp/xe rẽ và cảnh che khuất. Đây là hai cảnh cùng bộ dữ liệu, không phải hai mẫu độc lập cho CI tổng quát.

Nguồn nhãn là SQLite Polytrack `*_gt.sqlite`. Dùng đúng 1000 frame có GT mỗi cảnh: René 7200–8199, St-Marc 1000–1999; map frame nguồn t sang JPG `t+1` vì GT 0-based và ZIP JPG 1-based. Lưu phép ghép và ảnh minh họa, không tuyên bố căn chỉnh tuyệt đối nếu không có video nguồn St-Marc để xác nhận. Chỉ đánh giá object có loại rõ: car/truck/bicycle ở René, car/bicycle/pedestrian ở St-Marc; `unknown` loại khỏi GT và phải báo số lượng, không gộp tùy tiện. Không có motorcycle trong hai cảnh. Không dùng nhãn `occluded`; tập này không cho phân tầng che khuất chính thức.

Trước full: đo recall detector trên mẫu hệ thống 50 frame/cảnh tại conf>0,05, IoU≥0,5, dùng YOLOv8n COCO và YOLOv8n mixed5 R54; nếu detector quá kém, dừng tracking và chỉ báo giới hạn. Nếu đủ khả năng, khóa một detector rồi tạo cache dùng chung và chạy CV/EKF/UKF/PF500 seed0/1/2, c25, buffer 1,2 s, FPS René 30000/1001; St-Marc 30 fps theo trang tác giả. HOTA/IDF1/MOTA/IDSW tính bằng TrackEval hiện có; so cùng cache/GT, không sửa thuật toán/chốt cấu hình cũ. Kết quả là thăm dò chuyển miền vì chỉ có 2 cảnh, không so trực tiếp với metric Polytrack ABSDIST chính thức.

Điểm cần kiểm trước full: ROI mask René, phạm vi nhãn St-Marc; box không hợp lệ, trùng frame-ID; nguồn nhãn JPEG lệch tối đa một frame; recall detector, đặc biệt bicycle. Nếu dừng, ghi lý do và không xếp hạng bộ lọc từ cache detector yếu.

## Sửa giao thức sau smoke detector, trước chạy tracking

Mẫu 50 frame/cảnh cho thấy YOLOv8n COCO ở conf>0,05, IoU≥0,5 nhận car René 187/336 (55,7%) và St-Marc 40/46 (87,0%), nhưng bicycle 0/67 và 2/22, truck René 0/16. YOLOv8n mixed5 nhận car 7/336 và 36/46, bicycle 0/67 và 0/22. Vì vậy **khóa YOLOv8n COCO, chấm car-only** để tránh gọi sự bỏ sót bicycle/truck là lỗi bộ lọc. Mỗi cảnh đánh giá xe ô tô `road_user_type=1`; tất cả loại khác, kể cả unknown, thành distractor class8; báo rõ số nhãn bị loại khỏi metric. Không dùng pedestrian St-Marc để kết luận về phương tiện. COCO class2 là detection duy nhất vào tracker. Giữ ảnh toàn khung; mask René chỉ dùng phân tích phạm vi GT, không sửa ảnh suy luận. So CV/EKF/UKF/PF500 mặc định ba seed và PF500 K1K2 motion100 opt-in ba seed; c25, cùng cache. Hai cảnh báo riêng và tổng chỉ mô tả. Đây là sửa lựa chọn đối tượng theo kết quả smoke; **không phải đánh giá mù**.

## Sửa giao thức sau CV René c25, trước ma trận bộ lọc

Lượt CV René toàn khung c25 vừa chạy để kiểm pipeline cho HOTA 11,82%, MOTA −98,68%; kiểm tiếp cache cho thấy ở c25 detection car chỉ khớp 1.448/6.600 GT, trong khi 6.352/10.256 detection car nằm ngoài ROI mask tác giả cung cấp. 6.592/6.600 GT car có tâm trong ROI. Vì vậy đánh giá chính René dùng **cache dẫn xuất lọc tâm detection theo mask ROI**, giữ GT gốc, cùng một cache cho mọi bộ lọc; cache toàn khung và lượt CV đầu giữ làm chứng cứ thăm dò, không gộp. St-Marc không có mask tương ứng nên giữ cache toàn khung. Chạy cả c25 và c10 (R54 có hai ngưỡng này) để thấy ảnh hưởng recall; không chọn ngưỡng theo HOTA tracker. Kết quả là thăm dò sau khi đã xem một điểm CV, tránh diễn giải là benchmark mù.
