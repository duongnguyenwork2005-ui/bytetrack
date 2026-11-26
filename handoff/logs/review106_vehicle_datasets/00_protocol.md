# R106 — giao thức hậu kiểm dữ liệu xe, 12/11/2026

Phạm vi do người dùng giao trực tiếp: R106. Mục 0 còn ghi R105 chờ review; phiên này không xác nhận hay thay kết luận review R105. Không mở R107. Không inference, GPU, train, SSH hoặc chạy lại tracker/benchmark có bằng chứng hợp lệ. Chỉ đọc GT, cache metadata, MOT và raw fields đã có; ghi trong thư mục R106 và cập nhật mục 0.

## Khóa trước phân tích R106

Đã biết kết quả tổng hợp và một số phân tầng cũ qua mục 0, R64/R66/R99. Vì vậy đây là **hậu kiểm mô tả**, không phải đăng ký trước độc lập hay tập chưa xem điểm. Không chọn detector, ngưỡng, clip hoặc bộ lọc từ kết quả mới.

- Toàn bộ mẫu đã khóa: SVMOT test9, RoundaboutHD prefix3000 bốn camera, HELMET test36, Sherbrooke cars v2 1001 frame, hai RideSafe pilot R100. Giữ riêng từng tập và phạm vi xe; không gộp HOTA giữa dataset.
- Phép so sánh chung cố định: YOLOv8n COCO, conf >0,25, KF/CV–EKF–UKF trên cùng cache của từng tập và cấu hình đã khóa. Các điểm tổng hợp còn lại có thể dẫn nguồn lịch sử nhưng không dùng chọn giả thuyết.
- Gán nhóm bằng **GT duy nhất**, trước khi mở MOT/raw fields trong runner R106. Tái dùng nguyên hàm `labels_for_camera` R99: điểm giữa đáy hộp, hỗ trợ hai giây, trung vị quanh ba mốc, mỗi chân dịch chuyển >= max(8 px, 0,25 chiều cao hộp), đổi hướng cùng dấu >=25 độ kéo dài ceil(8*fps/15) frame; >=35 độ là độ nhạy đã có, gần thẳng <=8 độ làm đối chứng. Không nội suy qua khoảng thiếu GT. Lưu toàn bộ episode và hash GT. Với camera gắn xe đây chỉ là góc trên ảnh, chứa chuyển động ego, không suy ra rẽ trên mặt đường.
- Che khuất: chỉ SVMOT dùng visibility GT thật, khóa nhóm 0<=visibility<=0,5 và 0,5<visibility<=1; -1 là không biết. Không gọi cột visibility chuyển đổi/dummy, khoảng mất match hay hộp chồng nhau là nhãn che khuất. RideSafe ghi số cờ `occluded` thực, nếu toàn 0 thì không ước lượng mức che khuất. HELMET/RoundaboutHD/Sherbrooke không có thuộc tính đủ chứng cứ trong GT MOT: N/A.
- Không lọc GT theo điểm hoặc thay MOT để chấm riêng các episode. Phát lại association CLEAR nguyên chuỗi với evaluator hiện có rồi đếm IDSW/FN tại các khung-xe đã gán nhóm; IDSW ở biên episode giữ lịch sử nguyên chuỗi. HOTA/IDF1 chỉ báo toàn mẫu từ raw fields tái tổ hợp, không gán điểm toàn tập cho nhóm rẽ/che khuất. FN và IDSW nhóm có mẫu số khung-xe riêng.
- Xác minh số clip/ID/box từ GT, ánh xạ ID preprocessing, IDSW/FN toàn chuỗi khớp điểm cũ; tái tổ hợp HOTA/Identity/CLEAR bằng TrackEval và so CSV gốc. Hash code/GT/MOT/raw/manifest dùng làm dấu vết. Không gọi PASS từ báo cáo cũ.
- Hai demo chọn sau điểm trong slide là minh họa, không tiêu chí chọn mẫu và không bằng chứng tổng hợp mới.

## Nguồn và tải mẫu

Chỉ nguồn tác giả: HELMET GitHub/OSF, SVMOT Zenodo, Sherbrooke Urban Tracker, RoundaboutHD GitHub/Hugging Face, RideSafe DashCop/Hugging Face. Kiểm nguồn chính thức trực tuyến; lưu URL, thời điểm, HTTP và SHA256 của tài liệu nhỏ đã lấy. Không đăng nhập, gửi thư hoặc tải lại dữ liệu có sẵn. KITTI tracking là ứng viên camera xe mới để kiểm khả năng lấy mẫu; nếu cổng yêu cầu login thì chỉ ghi blocker, không vòng qua cổng. Trần tải mới tự giới hạn 25 MiB cho **tài liệu/nhãn mẫu công khai**, không tải full archive ảnh/video. Không có quyền mở thí nghiệm detector mới từ khảo sát nguồn.

## Tự kiểm

Runner có hai lệnh nối tiếp `gt` rồi `audit`; `audit` chỉ dùng cohort GT đã lưu và kiểm lại hash giao thức/GT. Chạy kiểm tất định trên dữ liệu thực, lưu lệnh, exit và tóm tắt giới hạn. Cập nhật mục 0 trước báo cáo `10_ket_qua.md`. Reviewer độc lập do orchestrator mở ở phiên khác.
