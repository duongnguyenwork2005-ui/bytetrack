# RBPF vận tốc Cartesian

Đã thử biểu diễn chuyển động bằng [cx,cy,vx,vy] có Gaussian điều kiện, chỉ lấy mẫu tốc độ quay omega. Hai khối [a,va] và [h,vh] dùng chung giữa các hạt. Giữ 500 hạt, ba seed, detector và ghép ByteTrack. Nhiễu vx/vy đẳng hướng theo hệ số vận tốc hiện hành, nên có thay đổi mô hình nhiễu so với RBPF cực R112. Đây là biến thể nghiên cứu mới. [Cấu trúc Rao–Blackwell](https://arxiv.org/abs/1301.3853) · [Mô hình constant turn của MathWorks](https://www.mathworks.com/help/pdf_doc/fusion/fusion_ug.pdf)

## Chất lượng

- **UA-DETRAC dev12** có 36 lượt cho mỗi cách khởi tạo. RBPF R112 / Cartesian với hai quan sát / Cartesian tự học có IDSW **469 / 464,67 / 514**, HOTA **58,963 / 58,995 / 58,981%**. Bản giữ khởi tạo hai quan sát giảm 0,92% IDSW so với RBPF. Khoảng bootstrap ghép theo video **[−30,+18,33]** đi qua 0. Bản tự học vận tốc bị loại. [Dev chính](dev_summary.json) · [Đối chiếu khởi tạo](automatic/dev_summary.json)
- **SVMOT test9** chạy lại cả RBPF cũ và bản Cartesian đã chọn, 54 lượt. IDSW **440→435,33**, giảm **1,06%**. HOTA **51,014→50,772%**, giảm **0,242 điểm phần trăm**. FN **14.633→14.615**, FP **22.383,33→22.259,33**. Bộ đối chứng tái tạo đúng MOT R112 ở cả 27 lượt. [Điểm SVMOT](transfer_summary.json)
- **Nhóm che khuất SVMOT**, 12.051 hộp visibility≤0,5, IDSW **136→132,33**, FN **6.479→6.478**. Các sự kiện IDSW và FN toàn chuỗi từ phép ghép CLEAR khớp kết quả trình chấm ở 54/54 lượt trước khi phân nhóm. [Điểm theo nhóm che](occlusion_summary.json)

Mức giảm IDSW nhỏ và dev đã được xem nhiều vòng. SVMOT cũng đã được quan sát trong các vòng trước. Các điểm trên chưa chứng minh ưu thế tổng quát. So với PF500 K1K2 motion2 cũ, dev 697→464,67 IDSW, nhưng đối chứng gần nhất của lượt này là RBPF R112 với 469 IDSW.

## Chi phí xử lý

Hai prefix MVI_20052/MVI_20062, mỗi đoạn 512 khung, bỏ 32 khung nóng, ba lần đo tuần tự và đảo thứ tự. Update RBPF R112 **42,20 ms/frame**, Cartesian **33,90 ms/frame**, giảm **19,67% thời gian tracker**. Cả sáu cặp đều nhanh hơn, mức giảm theo cặp từ **8,62% đến 38,98%**. Thời gian tuyệt đối dao động mạnh qua các lần, với đối chứng từ 23,82 đến 65,22 ms trên các prefix. Chưa đo trên Nano và chưa có số FPS toàn hệ thống. [Các lượt đo](serial_timing.json)

## Hậu kiểm và phạm vi

Giới hạn omega=0 khớp Kalman CV tham chiếu trong cả predict và update. Kiểm quỹ đạo quay 90°, tính liên tục quanh 0, covariance PSD, repeat seed và khôi phục hook đạt. Smoke RBPF đối chứng 6/6 khớp MOT R112. Cả 126 lượt và 56 hash đầu vào/nguồn được đối chiếu. [Hậu kiểm](verification.json)

Mã thử ở [cartesian.py](cartesian.py), dùng hook trong tiến trình và registry RBPF sẵn có. Khởi tạo từ hai detection theo cùng tiêu chí cũ và chuyển bất định vận tốc bằng Jacobian. Covariance giao diện cực cũng dùng xấp xỉ Jacobian, trong khi association sử dụng mean hộp vị trí/kích thước. Bản tự học được thêm trước khi đọc điểm dev của vòng này. [Giao thức](00_protocol.md) · [Đối chiếu khởi tạo](01_initialization.md)
