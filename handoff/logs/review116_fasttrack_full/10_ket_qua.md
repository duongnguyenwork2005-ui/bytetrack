# FastTrack đủ 12 video

Đã tải thêm chín ZIP, 10,996 GB, vào `D:/LABOTARY/test dataset/FastTrack-R116`. Ba ZIP của R115 vẫn ở `F:/LABOTARY_data/goc_tai_ve/FastTrack-R115`. Tổng cộng 12 ZIP, 14,756 GB, khớp SHA-256 nguồn. Bản này có 19.480 khung, 545.151 hộp mã lớp ô tô và 907 ID tính theo từng video. Ba video ban đầu được dùng lại sau khi đối chiếu hash. [Danh sách tệp](sources.json) · [Nguồn tác giả](https://huggingface.co/datasets/Hamidreza-Hashemp/FastTracker-Benchmark/tree/main/train)

Thứ tự bên dưới là KF, EKF, UKF và PF500 K1K2 motion2. PF lấy trung bình ba seed trừ khi ghi từng seed. Mọi cấu hình dùng cùng detection YOLOv8n COCO c25 trong từng video. [Giao thức](00_protocol.md)

## Tổng quan

- **Toàn bộ 12 video** có **1.099 / 1.075 / 1.103 / 1.277 IDSW** và HOTA **31,22 / 31,88 / 32,03 / 31,11%**. EKF giảm 24 IDSW, UKF tăng 4 IDSW. Khoảng bootstrap ghép cặp theo 12 video cho chênh IDSW so KF là [−81, +26] cho EKF và [−47, +48] cho UKF. Số video ít và có góc quan sát lặp, nên chỉ đọc khoảng này ở mức mô tả.
- **Chín video mới** có **781 / 778 / 801 / 904,33 IDSW** và HOTA **31,49 / 31,71 / 31,95 / 30,96%**. Lợi thế IDSW nhỏ hơn kết quả ba video ban đầu.
- **2.327 hộp ô tô đổi hướng ≥25° trên ảnh** có IDSW **8 / 8 / 15 / 12,67**, FN **1.092 / 1.095 / 1.094 / 1.111,33**. Mẫu này chưa cho thấy EKF và UKF cùng giảm đổi ID trong đoạn rẽ.
- **Nhịp ảnh**. Hai chuỗi đêm ghi 15 FPS ở `seqinfo.ini` và 30 FPS ở metadata video tái dựng. Phép đo chính dùng 15 FPS. Khi chấm lại hai chuỗi ở 30 FPS, toàn bộ 12 video có IDSW **1.100 / 1.076 / 1.105 / 1.280,67**, với cùng diễn giải chung. Chênh thời gian giữ track ở hai mức FPS được tính theo quy tắc một giây đã chọn. [Điểm đầy đủ và độ nhạy FPS](scores/scores.json)

## Xe buýt hoặc xe tải che ô tô

Chín video mới có 97 ứng viên hình học. Duyệt ảnh trước, trong và sau che xác nhận 67 lượt. Gộp 15 lượt R115 thành **82 lượt** do 21 ID phương tiện gây che tính theo video. Một xe gây che có thể xuất hiện trong nhiều lượt, ví dụ xe buýt đi ngang che nhiều ô tô đang xếp hàng.

- **Chỉ 67 lượt mới** có 19 lượt cả bốn cấu hình và mọi seed PF đều ghép được hai đầu. Số giữ đúng ID **4 / 5 / 6**, PF theo seed **4 / 6 / 6**.
- **Gộp 82 lượt** có 27 lượt ghép hai đầu chung. Số giữ đúng ID **4 / 9 / 11**, PF theo seed **9 / 11 / 12**. Trên toàn bộ sự kiện, KF/EKF/UKF thiếu ghép trước ở 47 lượt và thiếu ghép sau ở 7 lượt. Kết quả giữ ID tốt hơn trong nhóm này đi cùng HOTA và IDSW khác nhau trên toàn chuỗi. [Kết quả theo xe và sự kiện](scores/occlusion.json)

## Cột và trụ cầu đi bộ

Góc `task_day_left_turn` có **48 lượt được xác nhận bằng ảnh**, gồm **29 lượt qua cột dải phân cách** và **19 lượt qua trụ cầu đi bộ**. Có 33 lượt thiếu ghép trước và 14 lượt thiếu ghép sau ở cả bốn cấu hình. Lượt duy nhất ghép được cả hai đầu bị đổi ID ở tất cả cấu hình.

Kiểm tra cache theo lớp Car và IoU ≥0,5 cho thấy chỉ **3/48 lượt** có hộp ứng viên ở cả hai đầu khi dùng ngưỡng 0,25. Khi hạ ngưỡng để **kiểm tra độ phủ trong cache** xuống 0,05, con số là **21/48**. Đây là số candidate trước ghép một-một. Với lượt tracking đã chạy, nhóm vật cố định chủ yếu đo giới hạn quan sát, và mới đại diện cho một góc quay. [Từng lượt qua cột hoặc trụ](scores/fixed_occlusion.json) · [Độ phủ detection](scores/fixed_detection.json)

## Hậu kiểm

Ảnh liên tiếp theo metadata, 12 ZIP khớp SHA-256. Đã kiểm hash 12 cache, 12 GT chuyển đổi và 72 kết quả MOT, gồm 54 lượt mới và 18 lượt sử dụng lại. Kiểm bổ sung 12 kết quả MOT ở FPS thay thế. Bộ đếm IDSW, FN và FP gộp khớp tổng từng video. Toàn bộ nhãn visibility ghi 1. Hai nhóm che khuất dùng kiểm ảnh thủ công vì trường đó thiếu thông tin về cảnh che. Có một số hộp xe tải nhẹ mang mã lớp ô tô trong ảnh mẫu. Chấm theo mã gốc và ghi nhận hạn chế này. Tỉ lệ GT có ứng viên Car ở IoU ≥0,5 toàn bản là **26,70%**, riêng hộp dưới 25 px là **0,11%**. Kết quả về ô tô chưa đủ để suy luận về xe máy.
