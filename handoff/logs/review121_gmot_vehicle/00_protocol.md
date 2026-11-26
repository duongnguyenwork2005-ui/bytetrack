# R121 — GMOT-40 chỉ xe đường bộ, pilot có bất định FPS

Đã tải nhãn và chọn lọc đủ bốn chuỗi car từ release v0.1 của tác giả, kiểm CRC nguồn/SHA từng ảnh. Duyệt ảnh đầu/giữa/cuối trước điểm: car-0 vòng xuyến đêm góc cao cố định, car-1 đường đêm, car-2 đường tuyết; car-3 đua xe trên trường đua, loại khỏi phạm vi camera giao thông đường bộ. Không chọn chuỗi bằng điểm tracker. Giữ cả car-0/1/2, 1.027 ảnh, 22.301 hộp, 145 ID theo chuỗi. Không có lớp xe máy hoặc cờ che khuất.

**Giới hạn:** bài báo chỉ ghi FPS toàn tập trong khoảng24–30, ZIP ảnh không có seqinfo/video/timestamp. Chưa biết FPS thật của từng chuỗi. Chỉ chạy thăm dò độ nhạy với hai giả định đầu mút24/30Hz, cùng buffer1giây, dt theo giả định; không gọi đó là benchmark cuối hay lấy30làmFPSđãxácminh. Muốn kết luận vật lý cần xác nhận cadence. Báo cả hai mức, không chọn mức có lợi.

Detector YOLOv8n COCO FP32 640rect, cache .05/NMS.7/max3000 nhưR115; trackingc25/max300. Mục tiêu GMOT gọi chungcar; gộp detection car/bus/truck (2/5/7) thành một loại phương tiện bốn bánh vì ảnh/nhãn có cả bus/truck. Đây là giao thức detector riêng của dự án, không phải one-shotGMOT của tác giả. DùngKF/EKF/UKFđãkhóa, cùngcache; khôngReID/egomotion/associationmới.

Chuyển frame0→1 và padding nhãn→MOTsingleclass1, giữ mọi hộpnguồn. ChấmHOTA/CLEAR/IDF1quaTrackEvalkhôngpreprocess. Nhómturn25 giữ hàmR115vớifpsgiảđịnh tươngứng; vìảnhcar-0chỉ7,6–9,5giâynêncóthểítđoạnđạt. Không suy ra che khuất từvis=-1hoặcsựvắngnhãn. KiểmCLEARIDSW/FN từnglượt và hashnguồn/GT/cache/MOT, luôn báocảFN/HOTA. Nếu hiệu ứng thắng chỉ xuất hiện ởmộtFPS, ghi rõ không ổnđịnh.
