# Pilot METEOR camera gắn xe

Lựa chọn trước khi mở bất kỳ metric tracking nào của METEOR. Chọn đúng clip đầu tiên thuộc nhóm Frame XML Annotations/REC_2020_*_F.zip theo thứ tự archive tác giả, REC_2020_07_12_02_55_27_F. Đã kiểm gói nhãn có 1.800 XML và trường track_id trong attributes. Các lớp MotorBike và Scooter có tổng 2.023 hộp. Dùng toàn clip để tránh chọn đoạn theo thứ hạng tracker.

Đọc riêng video 80 MiB và gói nhãn từ ZIP chia phần 93,38 GB trên kho Hugging Face được tác giả liên kết. Chỉ dùng HTTP Range công khai, ghim revision, kiểm Content-Range và CRC của thành viên được lấy. Không tải toàn bộ dataset. Ghi video tại F:/LABOTARY_data/goc_tai_ve/METEOR-R112/.

Phép đo mới chỉ chấm phương tiện hai bánh MotorBike/Scooter. Dùng YOLOv8n COCO của R54, conf sinh 0,05, IoU NMS 0,7, FP32, batch 4, rect, ngưỡng tracker >0,25 và lớp COCO 3. Cache dùng chung cho KF, EKF, UKF và PF500 K1/K2 motion2 cùng RBPF500 (ba seed mỗi biến thể PF). Bước thời gian lấy từ FPS kiểm bằng video, buffer 1,2 giây. Nếu độ phủ detection IoU 0,5 trên xe máy dưới 10%, chỉ báo chẩn đoán đầu vào và không diễn giải thứ hạng bộ lọc.

GT chỉ lấy ID có hộp hợp lệ, loại EgoVehicle khỏi mục tiêu. Nếu một ID đồng thời gán nhiều hộp trong cùng frame hoặc thay đổi giữa lớp mục tiêu và lớp ngoài, báo lỗi trước tracking. Giữ thuộc tính occluded và kiểm số khung ảnh/nhãn. Chấm TrackEval MOT toàn clip. Mốc rẽ theo R99 được lấy từ GT trước điểm tracking, nhưng phản ánh góc trên ảnh và chịu chuyển động camera. Đây là pilot một clip đã chọn theo thứ tự archive, chưa đại diện toàn bộ METEOR.
