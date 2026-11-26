# Auxiliary Particle Filter cho tracking xe

Đã thử ba biến thể trên UA-DETRAC train dev12, ba seed mỗi biến thể. Hai biến thể có điểm thuận trên dev được kiểm tiếp đủ 12 video FastTrack với tham số cố định. Tổng **180 lượt**, gồm 108 dev và 72 chuyển miền. Bộ so trực tiếp là PF500 K1K2 motion2 đang dùng cho các phép thử R115/R116. Cùng cache detection, số hạt, tiêu chí ghép và trình chấm trong từng cặp so.

## Kỹ thuật đã thử

APF dùng xác suất quan sát hiện tại để chọn tổ tiên, rồi lấy mẫu hậu nghiệm mới từ từng tổ tiên được chọn. Khi một tổ tiên được chọn nhiều lần, các mẫu con được sinh độc lập, giúp tránh sao chép cả nhiễu chuyển động đã rút trước đó. Với Q chéo của mô hình hiện hành và phép đo hộp tuyến tính, tính được likelihood tiên đoán và bốn tọa độ Gaussian điều kiện. Giữ cơ chế clamp chiều cao của hệ thống nên vẫn có xấp xỉ ở biên chiều cao. [Whiteley và Johansen](https://people.maths.bris.ac.uk/~manpw/apf_chapter.pdf) · [Triển khai và phương pháp APF trong MRPT](https://www.mrpt.org/tutorials/programming/statistics-and-bayes-filtering/particle_filter_algorithms/)

- **APF thuần** lấy mẫu lại tổ tiên ở mọi lần update hợp lệ và dùng nhiễu hậu nghiệm, bỏ K2.
- **APF+K2** thêm roughening K2 đang dùng sau ước lượng để duy trì độ đa dạng chuyển động.
- **APF+K2 theo ESS** chỉ thực hiện bước APF khi ESS/N dưới 0,5. Khi ESS cao, dùng K1 hiện có. Ngưỡng này kế thừa PF hiện hành.

Đây là plugin trong tiến trình Python, khôi phục hook sau mỗi lượt. Mã thuật toán ở [apf.py](apf.py).

## Tập phát triển UA-DETRAC

12 video train, 16.919 khung/seed, cùng checkpoint YOLOv8n COCO c25. Số liệu là trung bình ba seed sau khi gộp đúng metric cho từng seed.

- **PF hiện hành** có IDSW **697**, HOTA **57,01%**, FN **39.395,67**.
- **APF thuần** có IDSW **769,33** (+10,38%), HOTA **56,31%**, FN **40.542,67** (+2,91%). Bản này giảm chất lượng và dừng ở dev.
- **APF+K2** có IDSW **675,67** (−3,06%), HOTA **57,28%**, FN **38.913,67** (−1,22%). IDSW ba seed **680/688/659**. Khoảng bootstrap ghép theo video cho chênh IDSW so nền **[−42,33,−3,0]**.
- **APF+K2 theo ESS** có IDSW **677,67** (−2,77%), HOTA **57,37%**, FN **39.057** (−0,86%). IDSW ba seed **670/682/681**. Khoảng bootstrap chênh IDSW **[−43,67,+3,33]**.

Cả ba chưa đạt mục tiêu giảm IDSW 10% đã ghi trước. Sau khi thấy mức giảm nhỏ và các điều kiện HOTA/FN/seed của hai bản K2 đạt, đã mở thử chuyển miền thăm dò để kiểm tính nhất quán. Quyết định này được ghi sau dev, trước chạy FastTrack. [Giao thức ban đầu](00_protocol.md) · [Quyết định mở kiểm tra bổ sung](01_transfer_extension.md) · [Biến thể ESS](02_ess_extension.md)

Để đối chiếu với hướng đã có, **RBPF R112** trên cùng dev đạt **469 IDSW và HOTA 58,96%**. Số này là kết quả R112 đã kiểm trước đó, không phải phép chạy RBPF mới trong R117. APF trong lượt này chưa vượt được RBPF ở dev. [Nguồn RBPF R112](../review112_svmot_pf/dev_gate.json)

## Chuyển miền FastTrack

19.480 khung của đủ 12 video. Dùng lại cache và nhãn R115/R116, cả hai mức FPS theo metadata ảnh của R116. Không chọn hoặc sửa biến thể theo điểm FastTrack.

- **PF hiện hành** có IDSW **1.277**, HOTA **31,11%**, FN **414.493**.
- **APF+K2** có IDSW **1.282,67** (+0,44%), HOTA **31,54%**, FN **414.212,67**. Trên 82 sự kiện ô tô bị xe buýt/xe tải che đã xác nhận bằng ảnh, giữ đúng ID **8/8/10** theo seed, trong khi nền **9/11/12**. Số lượt ghép thiếu ở hai đầu cũng thay đổi, nên đọc cùng các bộ đếm đầy đủ.
- **APF+K2 theo ESS** có IDSW **1.285,33** (+0,65%), HOTA **31,47%**, FN **414.173**.

Cải thiện IDSW ở dev chưa lặp lại trên FastTrack. Hai bản tăng HOTA nhẹ và giảm FN nhưng tăng IDSW. [APF+K2](transfer_summary.json) · [Bản ESS](transfer_ess_summary.json) · [Theo từng sự kiện che của APF+K2](transfer_occlusion.json)

## Chi phí xử lý

Chạy tuần tự trên PC, hai đoạn MVI_20052/MVI_20062, mỗi đoạn 512 khung, bỏ 32 khung nóng, ba lần lặp và đảo thứ tự. Update tracker trung bình PF hiện hành / APF+K2 / ESS là **10,53 / 12,99 / 11,97 ms/frame**, tăng thời gian **23,33% / 13,71%**. Đây là thời gian cập nhật tracker trên hai prefix PC. Chưa có phép đo APF trên Jetson Nano. [Các lượt đo](serial_timing.json)

## Hậu kiểm và diễn giải

Kiểm Gaussian điều kiện bằng nghiệm một chiều và moment của hỗn hợp, chuẩn hóa trọng số, mẫu con độc lập khi lặp tổ tiên, tái tạo cùng seed và khôi phục hook đều đạt. Smoke đối chứng sáu lượt khớp MOT đã lưu. Cả 180 lượt hoàn tất và hash cache/GT/MOT khớp. Đối chiếu 54 hash đầu vào và nguồn, giữ snapshot hai file của phiên APF đầu sau khi bổ sung nhánh ESS. Snapshot khớp hash đã khóa từ trước full. [Hậu kiểm](verification.json)

Kết quả gợi ý việc lấy mẫu con độc lập có lợi nhỏ cho PF K1K2 trên dev, còn kỹ thuật regularization K2 vẫn cần thiết trong thiết lập này. Việc APF resample ở mọi update làm tăng chi phí và chưa giảm IDSW ở miền khác. Độ đa dạng hạt là một thành phần của bài toán. Mô hình chuyển động trên ảnh, detection bị sai hộp khi che khuất và các quyết định association vẫn ảnh hưởng đến khả năng giữ ID. Đây là diễn giải từ các phép đối chiếu, chưa phải chứng minh một nguyên nhân duy nhất. Ba ứng viên chưa đủ bằng chứng để thay cấu hình mặc định.
