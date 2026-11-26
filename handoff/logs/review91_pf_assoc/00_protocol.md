# R91 — chẩn đoán IDSW PF bằng quyết định ghép cặp

Ngày 06/11/2026. Phạm vi chỉ phương tiện. Không dùng MOT17. R90 trên UAVDT train8 cho thấy PF K1/K2 motion100 giảm IDSW so PF mặc định nhưng vẫn hơn CV 69,67 IDSW ở c10; M1306 đóng góp 58 chênh lệch. Đây là tập phát triển đã xem, nên mọi thay đổi thử tại đây là **thăm dò**, không phải bằng chứng tổng quát.

## Bước chẩn đoán trước thay đổi

Dùng cache detection R90 YOLOv8n COCO, c10, class xe 2/5/7, GT/ignore/evaluator và buffer 36 bước y hệt R90. Chạy lại **M1306** (che khuất cao, 1.200 frame) và **M0204** (đổi hướng, 350 frame) với `--assoc-log` cho CV và PF K1/K2 motion100 seed0; xác nhận tệp MOT trùng byte R90 trước khi diễn giải log. Log gồm hộp dự đoán trước association, detection được chọn, trạng thái lost/refind/new. Đếm tình huống GT còn tồn tại và detection hợp lệ nhưng track cũ không ghép; so phân bố IoU/hạng ứng viên của CV và PF. Không tự đồng nhất mọi khoảng mất detection với che khuất vật lý.

Nếu log hỗ trợ giả thuyết hộp dự đoán PF làm sai thứ tự ghép, thử **một** biến thể opt-in đọc phân bố hạt khi tính chi phí ghép, giữ cache/GT, không thay mặc định hoặc YAML luận văn. Cổng mở full: giảm IDSW trên hai cảnh development mà HOTA không giảm quá 0,5 điểm, không tăng FP quá 5%. Sau đó chỉ kiểm độc lập trên chuỗi phương tiện chưa dùng để chọn tham số; không dùng lại BDD/UA đã dò nhiều vòng làm kiểm mù.

Nền nghiên cứu: [Rao–Blackwellized data association PF cho nhiều vật](https://oa.upm.es/7270/1/2010_Visual_tracking_of_multiple_interacting_objects_through_Rao-Blackwellized_data_association_particle_filtering.pdf) tách bất định ghép cặp khỏi trạng thái; [particle filtering đa mục tiêu với Gibbs association](https://ieeexplore.ieee.org/document/1039400) cho thấy liên kết quan sát là bài toán riêng; [PKF probabilistic association](https://arxiv.org/abs/2411.06378) dùng xác suất ghép khi ứng viên mơ hồ. Các công trình này gợi ý tận dụng phân bố dự đoán, không chứng minh trước biến thể cụ thể ở repo sẽ cải thiện.

## Bổ sung detector trước khi xem checkpoint huấn luyện phụ

Mixed5 train có 2.235 box motorcycle/476.062 box gộp năm lớp (0,47%); BDD dev có 226 box motorcycle. Thử chẩn đoán cắt ảnh UTUAV 4K bằng ô 960 px/overlap 160 px rồi 640 px/overlap 80 px trên **cố định** B frame 1/369/738 và C frame 1/376/751, hai checkpoint YOLOv8n COCO và YOLO11n mixed5, suy luận imgsz640/conf0,05/NMS0,7. Chỉ đo recall một-một theo lớp đối với nhãn xe động IoU≥0,5, số detection và thời gian tham khảo; không suy ra AP/precision từ sáu frame hoặc nhãn xe tĩnh bị loại. [SAHI](https://arxiv.org/abs/2202.06934) là cơ sở kỹ thuật cho thử ô ảnh, không đảm bảo hiệu quả trên UTUAV.

Thử huấn luyện riêng YOLO11n mixed5 trên 5.000 ảnh train lấy từ manifest đã khóa, ưu tiên ảnh có motorcycle và vẫn giữ đủ car/truck/bus/bicycle; không dùng ảnh dev/test, không ghi đè checkpoint gốc. Hai epoch SGD lr0=0,001, ảnh640, batch8, seed9106; so cùng lệnh validation trên toàn mixed5 dev 7.190 ảnh. **Cổng chọn ứng viên cho bước thử ngoài:** AP50 motorcycle tăng ít nhất 0,03 tuyệt đối, không lớp nào trong bốn lớp còn lại giảm AP50 quá 0,02, mAP50 gộp không giảm. Nếu đạt cổng này, chấm riêng 5.559 ảnh BDD dev và 1.631 ảnh UA dev, yêu cầu không lớp car/truck/bus của UA nào giảm AP50 quá 0,03, rồi mới diễn giải recall motorcycle trên HELMET 36 clip × frame 1/50/100 và UTUAV; HELMET đã được dự án quan sát nên đây vẫn là thăm dò, không phải test mù. Nếu cổng không đạt, không thay detector đã khóa dù một vài số cải thiện.

## Cổng chọn bộ dữ liệu theo mục tiêu

Không chọn tập dựa trên việc PF thắng KF. Ưu tiên (1) GT ID của **phương tiện**, có xe máy khi cần kết luận riêng về lớp đó; (2) timeline/fps rõ; (3) tình huống rẽ hoặc che khuất được nhãn trực tiếp hay có thể gắn nhãn từ quỹ đạo với quy tắc công bố trước; (4) detector có đủ recall để IDSW có nghĩa; (5) chia scene/train/dev/test tránh rò rỉ. Báo riêng từng tập vì quy ước GT/ignore khác nhau.

- **HELMET test36:** phù hợp nhất cho xe máy (36 clip/12 nhóm, 10 fps), nhưng box thường gồm người và xe; chưa có nhãn rẽ/che khuất nên chỉ kết luận theo dõi xe máy tổng thể trước khi tự gắn sự kiện.
- **RoundaboutHD:** phù hợp nhất cho quỹ đạo cong/che khuất giao thông vòng xuyến theo [bài giới thiệu của tác giả](https://arxiv.org/abs/2507.08729); 4 camera, phép chấm hiện có là 3.000 frame đầu mỗi camera. Không tự suy ra cải thiện riêng lúc rẽ khi chưa chia sự kiện.
- **SVMOT:** có visibility, cặp che khuất, depth và trạng thái dừng theo [kho dữ liệu](https://zenodo.org/records/19468203); hữu ích kiểm IDSW sau che khuất. Test hiện chỉ 9 chuỗi thuộc 2 scene, phải phân tích theo scene và dùng split cấp scene nếu huấn luyện thêm.
- **UAVDT:** car/truck/bus, cờ mức che khuất và góc nhìn theo [bài ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.pdf); 8 train scene R90 là development đã xem, bộ test chính thức nên dành cho đánh giá sau khi khóa ứng viên. Không đại diện xe máy.
- **UTUAV:** có motorbike, car, bus và timeline B/C đã được tác giả xác nhận, nhưng detector hiện khớp <2% GT động ở IoU0,5; tạm chỉ dùng để chẩn đoán detector cho đến khi có phương án tốt hơn. **VisDrone MOT** đạt cổng detector nhưng stride ảnh hiệu dụng chưa rõ; **TrafficMOT** thiếu nhãn test/FPS; chưa dùng hai bộ này để kết luận tracking. BDD/UA giữ làm đối chứng đã quan sát nhiều lượt, không làm test mù cho PF mới.

Lớp **bicycle** hiện có nhãn trong BDD dev/val nhưng chưa có một tập ngoài độc lập đạt đủ cổng timeline, GT và detector. Không suy từ kết quả motorcycle/car sang bicycle; AP riêng trên BDD dev là cổng tối thiểu trước khi tuyên bố detector năm lớp tốt hơn.

Để xác định HELMET có hỗ trợ riêng kịch bản rẽ/chen lấn hay chỉ hỗ trợ xe máy tổng thể, chấm từ GT trước khi xem điểm bộ lọc: với ID có đủ t−5/t/t+5, đánh dấu proxy rẽ nếu hai vector tâm đổi hướng ≥30° và mỗi vector đi ≥0,5 lần chiều cao box ở t; đánh dấu proxy giao nhau/che khuất nếu IoU box GT cùng frame ≥0,25. Đây chỉ là **proxy hình học**, không phải nhãn hành vi hay cờ che khuất nguồn. Báo số clip/ID/frame thỏa, không dùng proxy để đổi ngưỡng sau khi nhìn kết quả.
