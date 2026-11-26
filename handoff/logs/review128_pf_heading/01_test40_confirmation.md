# Xác nhận UA-DETRAC test40 sau khi đạt UA dev và BDD

Khóa trước khi chạy hoặc xem điểm test40 của ứng viên. User yêu cầu PF ít IDSW
hơn cả KF/EKF/UKF trên hai tập, nên UA train dev12 chưa đủ để tuyên bố thắng
UA-DETRAC test. Không điều chỉnh ứng viên theo test40.

Nếu R128 đạt cổng BDD, chạy nguyên source trên manifest
`results/phase2/manifests/detrac_test.json`, đủ40video; cùng YOLOv8n COCO c25,
cache `yolov8n-coco_conf0.05_all`, lớp[2,3,5,7], buffer30, cadence25FPS,
elapsed skip, CTRV dt1 theo frame như baseline. Không đánh đồng cadence thời
gian giây với tham số CTRV dt; runner dùng cadence để đếm bước, không tự
đổi tham số CTRV dt. GT/MOT evaluator giữ nguyên của protocol UA hiện hành.

Chạy mới KF/EKF/UKF mỗi bộ một seed xác định và PF ba seed:240lượt.
Chạy thêm native PF100 R126 ba seed nếu chưa có toàn test40 cùng đầu vào,
để kiểm HOTA/FN so baseline trực tiếp:120lượt, tổng360.
Khóa hash GT/cache/seqinfo/config/source trước chạy. Smoke kiểm repeat và
đường KF/EKF/UKF so mốc lịch sử nếu đủ metadata, không dùng nguồn lịch sử
khác cache/cấu hình để thay đối chứng mới.

Tuyên bố đạt IDSW trên test40 chỉ khi mean PF<min(KF,EKF,UKF), ít nhất2/3seed
cũng thấp hơn min. HOTA>=native PF100-0,001, FN<=1,002*native PF100.
Tính CI ghép video, báo khi CI chứa0. Nếu test40 không đạt, giữ kết quả như
phép kiểm chuyển tập không đạt; không tự sửa tham số theo test40 hoặc chọn
ít video có lợi để công bố. Hướng nghiên cứu tiếp phải dựa dev/diagnosis và
được ghi như thăm dò, không biến test40 thành dev mà không nói rõ.
