# R67 — UTUAV B/C validation với FPS xác nhận (03/11/2026)

## Kết luận

**Hoàn tất phép thử hệ thống theo Q3:** 36/36 run trên toàn bộ B validation (738 frame) và C validation (751 frame), dùng 3 detector R54, KF/CV–EKF–UKF–PF500 (3 seed), hai ngưỡng c25/c10. Đây là kiểm tra chuyển miền và tính tái lập của pipeline. **Không dùng để chứng minh bộ lọc nào theo dõi xe máy/rẽ/che khuất tốt hơn KF:** ngay detector có cận trên recall cao nhất cũng chỉ khớp IoU≥0,5 với 1,75% GT động B và 0,89% GT động C ở conf >0,05. Mọi HOTA COMBINED dưới 0,5%.

## Nguồn thời gian và phạm vi

- Thư trả lời trực tiếp của Jorge Ernesto Espinosa Oviedo do người dùng chuyển trong chat xác nhận B/C gốc `30000/1001 fps`, ảnh phát hành stride 1, từng block Sequential_Gap300 liên tiếp. Bước thời gian `1001/30000 s`; buffer 1,2 s làm tròn thành 36 frame. Nội dung và giới hạn provenance ở [00_nguon_fps.md](00_nguon_fps.md). Trang [bài báo UTUAV](https://www.mdpi.com/2504-446X/10/1/15) ghi *Drones* 2026, 10(1), 15 (đăng trực tuyến 27/12/2025), nhưng không ghi tường minh FPS/stride.
- Chỉ dùng B `26341–27078` và C `20271–21021`, nội bộ ánh xạ 1..N. Không dùng training/evaluation, không nhảy qua gap 300 frame, reset tracker giữa B và C. Split `UTUAVSYS-validation` có 1.489 JPG 3840×2160, giải nén chọn lọc trên F. Archive ảnh đã kiểm CRC trước đó và SHA lúc chuyển ổ; từng JPG đã giải mã, kiểm kích thước, hash; gói XML nhãn SHA được ghi trong `source_manifest.json` và `images_verified.json`.
- Nhãn 6 lớp XML: motorbike/car/bus động → MOT class 1; ba lớp `_stat` → distractor class 8. Tổng 261.303 box nguồn, trong đó 154.761 box động (B 29.756; C 125.005). Audit `source_xml_audit.json`: 45 box ở mép ảnh vượt tối đa 4 px (B 3, C 42), các box còn lại hợp lệ; **giữ nguyên tọa độ XML, không cắt/chữa**. ID duy nhất từng frame trước lọc distractor. `occluded` toàn 0, không phân tầng che khuất.
- Bộ kiểm ảnh đầu/giữa/cuối cả B/C: `gt_visual_first_mid_last.jpg`. Giao thức khóa: `01_giao_thuc.json`; mã mới: `src/r67_utuav/`. Không sửa tracker, checkpoint, YAML hay metric đã khóa.

## Cổng kỹ thuật và độ phủ detector

Smoke `128` frame đầu mỗi chuỗi hoàn thành 36/36 run, 72/72 đơn vị chuỗi. PF500 cùng seed chạy lại cho MOT và ba file điểm trùng byte. Fixture converter (động, distractor, ID trùng, biên ảnh, frame rỗng) và fixture evaluator (oracle, distractor, output rỗng) đều đạt. `smoke/review.json` PASS sau khi kiểm lại hash cache, từng đơn vị tracking và phép tái tổ hợp metric. **Cổng này chỉ kiểm tính đúng kỹ thuật, không đặt ngưỡng điểm/recall.**

Trên toàn bộ validation, dùng mọi detection thuộc lớp phương tiện của tracker với conf **>0,05**, cận trên recall là tỷ lệ GT động có *ít nhất một* detection giao IoU≥0,5, không ràng buộc class hay ghép một-một:

- YOLOv8n COCO: B **520/29.756 = 1,75%**; C **1.107/125.005 = 0,89%**.
- YOLOv8n mixed5: B **160/29.756 = 0,54%**; C **3/125.005 = 0,0024%**.
- YOLO11n mixed5: B **210/29.756 = 0,71%**; C **29/125.005 = 0,023%**.

Đây là **cận trên**, không phải recall class-aware hoặc AP. Ngưỡng tracker `>0,10`/`>0,25` chỉ có thể làm độ phủ thấp hơn. Toàn bộ 18 tổ hợp detector×B/C×ngưỡng có số box và số frame phát hiện trong `full/detector_upper_bound_recall.json`. Ảnh UAV 4K chứa nhiều xe rất nhỏ ở kích thước đầu vào 640; quan sát này phù hợp với lỗi chuyển miền nhưng không tự chứng minh nguyên nhân duy nhất của điểm thấp.

## Điểm tracking

Các số sau là **HOTA COMBINED theo phần trăm**, tái tổ hợp từ raw fields của hai chuỗi; PF500 là **trung bình ba seed**. Cặp c25/c10 tương ứng conf tracker >0,25/>0,10:

- YOLOv8n COCO c25: KF **0,2262**; EKF **0,2272**; UKF **0,2272**; PF500 **0,2301** (seed 0,2244–0,2371). c10: KF **0,4599**; EKF **0,4602**; UKF **0,4604**; PF500 **0,4580** (0,4566–0,4590).
- YOLOv8n mixed5 c25: KF **0,0892**; EKF **0,0900**; UKF **0,0900**; PF500 **0,0847** (0,0736–0,0907). c10: KF **0,1017**; EKF **0,1025**; UKF **0,1025**; PF500 **0,0965** (0,0865–0,1018).
- YOLO11n mixed5 c25: KF/EKF/UKF cùng **0,0909**; PF500 **0,0844** (0,0713–0,0909). c10: KF **0,1393**; EKF **0,1425**; UKF **0,1425**; PF500 **0,1365** (0,1342–0,1388).

Với YOLOv8n COCO c10, HOTA riêng B của KF là **0,7299%**, C **0,3263%**; COMBINED **0,4599%** được tính từ thống kê gốc, **không** lấy trung bình hai điểm riêng. Ở B, cùng cấu hình này có **0 TP theo CLEAR/IDF1 tại IoU 0,5** dù HOTA khác 0 vì HOTA gộp nhiều ngưỡng IoU thấp hơn. IDSW gần như 0 ở phần lớn cấu hình vì hầu như không ghép được GT; không diễn giải IDSW thấp là giữ ID tốt. Xem `full/per_video.csv`, `full/combined_tung_run.csv` và `full/filter_summary.csv` để kiểm từng detector, ngưỡng, filter, seed và B/C.

## Hậu kiểm, giới hạn và cách dùng trong khóa luận

- Full có 6/6 cache, 72/72 đơn vị tracking, 36/36 điểm; marker/hashes/metadata/GT/timeline/buffer đều khớp. `full/final_review.json` PASS; sai số tối đa khi tái tổ hợp 8 chỉ số COMBINED từ `raw_fields.json` là `9,89×10⁻¹⁷`. Mọi lượt chạy theo cùng cache detector cho bốn bộ lọc. Các artifact của pilot R64 30 fps nằm ở namespace khác, không bị ghi đè.
- Hai cảnh B/C không phải 1.489 quan sát độc lập; PF seed chỉ biểu diễn biến thiên thuật toán, không thêm cảnh. Không tính CI. Không so HOTA tuyệt đối với HELMET/SVMOT/RoundaboutHD để xếp hạng độ khó. Không dùng `occluded=0` để kết luận về che khuất thật; không có phép phân tầng rẽ trong full validation.
- Câu phù hợp cho luận văn: *"Sau khi xác nhận FPS và stride trực tiếp với tác giả, thí nghiệm UTUAV v2 B/C validation chạy hoàn chỉnh nhưng các detector đã huấn luyện ở miền khác có cận trên recall IoU 0,5 dưới 2% trên ảnh UAV. Vì tín hiệu phát hiện quá thưa, kết quả tracking không đủ để đánh giá lợi thế của EKF, UKF hoặc PF so với KF trong chuyển động rẽ/che khuất; đây là giới hạn chuyển miền của pipeline được thử theo giao thức đã khóa."*
- Muốn khảo sát khả năng bộ lọc trên UTUAV sâu hơn cần **giao thức mới** với detector phù hợp ảnh UAV hoặc một phép chẩn đoán detection tổng hợp riêng; không dùng số full này để chọn lại checkpoint hoặc tune tracker theo validation.
