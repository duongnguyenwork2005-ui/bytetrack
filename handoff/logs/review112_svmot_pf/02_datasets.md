# Dữ liệu bổ sung cho ba kịch bản

Kiểm nguồn tác giả ngày 18/11/2026. Ưu tiên góc nhìn camera giao thông cố định hoặc camera gắn xe. Việc có cảnh rẽ chỉ xác nhận tính phù hợp của dữ liệu; thứ hạng bộ lọc cần được đo với cùng detection và giao thức.

## METEOR

Đã xác minh từ kho tác giả GAMMA-UMD/METEOR và kho dữ liệu XijunWang/METEOR do chính README tác giả liên kết. Đây là dashcam 1920×1080 ở 30 fps, có xe máy, scooter, các hành vi rẽ và chuyển làn. Đã đọc danh mục ZIP chia phần bằng HTTP Range, ghim revision ecedf70e3d2d44c7e6e794246bb4bd1e4020662c. Tải được riêng một video 80 MiB cùng gói nhãn, kiểm CRC từng thành viên được lấy và SHA256 cục bộ. Video nằm tại F:/LABOTARY_data/goc_tai_ve/METEOR-R112/REC_2020_07_12_02_55_27_F.MP4.

Clip chọn trước điểm có 1.800 khung, 2.023 hộp MotorBike/Scooter, 10 ID và trường track_id. GT xe máy của clip này có occluded bằng 0 ở mọi hộp. Tiêu chí rẽ R99 nhận 81 khung-xe ở bốn đoạn ≥25°. Đã chạy pilot trên cùng YOLOv8n COCO c25; độ phủ candidate IoU 0,5 là 31,19%. KF/EKF/UKF có IDSW 16/23/19 và HOTA 18,17/17,51/17,43%. PF500 K1/K2 motion2 và RBPF500 cùng IDSW trung bình 17,67; HOTA 16,80% và 19,47%. Đây là pilot một clip, không có bằng chứng EKF/UKF tốt hơn KF trên clip này. Chi tiết trong meteor_pilot/verified_summary.json và giao thức 05_meteor_pilot_protocol.md.

- [Kho tác giả](https://github.com/GAMMA-UMD/METEOR)
- [Dữ liệu do tác giả liên kết](https://huggingface.co/datasets/XijunWang/METEOR)
- [Bài báo và đặc điểm camera, nhãn](https://arxiv.org/abs/2109.07648)

## V2X-Seq-SPD

Ưu tiên cao cho lượt dữ liệu mới. Kho tác giả công bố hơn 15.000 khung của 95 cảnh, có nhánh infrastructure-side và vehicle-side. Schema nhãn tracking chứa track_id, 2d_box, occluded_state và truncated_state, cùng lớp Motorcyclist, Car, Truck, Van và Bus. Có thể dùng riêng camera RGB cho ByteTrack mà không bắt buộc chạy LiDAR hoặc phép chiếu hộp 3D. Cần kiểm ảnh/nhãn cùng timestamp, ID theo sequence, độ phủ 2D và định nghĩa hộp Motorcyclist trước khi chốt ánh xạ với detector xe máy.

- [Kho tác giả và đường tải](https://github.com/AIR-THU/DAIR-V2X-Seq)
- [Định nghĩa nhãn](https://github.com/AIR-THU/DAIR-V2X-Seq/blob/main/dataset/v2x-seq-spd/README.md)
- [Mẫu chính thức](https://drive.google.com/file/d/1gjOmGEBMcipvDzu2zOrO9ex_OscUZMYY/view)

Đã xác nhận lần đầu endpoint trả đúng ZIP 585.329.481 byte và đọc magic PK. Khi tải thực, Google Drive trả trang Quota exceeded. Chưa tải được archive hoàn chỉnh và chưa tracking. Bằng chứng tại v2x_example_access.json, v2x_example_download_response.txt và download_v2x_sample.log. Thư mục dữ liệu dự kiến trên F là F:/LABOTARY_data/goc_tai_ve/V2X-Seq-R112/.

## CityFlow-V2

Phù hợp mở rộng nhiều camera cố định và nhiều nút giao, giảm phụ thuộc một vòng xuyến. Bản AI City 2022 có 46 camera tại 16 giao lộ, 313.931 hộp cho 880 ID. Theo tác giả, chỉ phương tiện đi qua ít nhất hai camera được chú thích. Khi chấm đơn camera cần khóa vùng và quy tắc bỏ qua xe ngoài nhãn; không thể coi mọi xe được detector tìm thấy là một đối tượng có GT. Cần kiểm số đoạn đổi hướng theo GT trước khi chọn làm tập chính phi tuyến. Chưa tải hoặc chạy mới trong R112.

- [Giao thức tác giả](https://www.aicitychallenge.org/2022-data-and-evaluation/)
- [Đường tải chính thức](https://www.aicitychallenge.org/2022-track1-download/)

## KITTI Tracking

Phù hợp camera gắn xe và có nhãn occlusion/truncation. Đánh giá xe ô tô theo giao thức lớp Car. Chuyển động trên ảnh còn bao gồm chuyển động camera. Vì vậy nên báo tách xe rẽ và ego-camera rẽ, thay vì suy góc ảnh thành độ cong vật lý của riêng phương tiện. Gói ảnh của cổng tác giả yêu cầu truy cập qua trang tải, R106 chưa có gói ảnh đầy đủ. Chưa chạy mới R112.

- [Tracking benchmark](https://www.cvlibs.net/datasets/kitti/eval_tracking.php)
- [Thiết lập camera và nhịp lấy mẫu](https://www.cvlibs.net/datasets/kitti/setup.php)

## TUMTraf Intersection R2 và V2X R4

Cảnh rẽ trái, rẽ phải, quay đầu và che khuất phù hợp phạm vi nghiên cứu. R2 có 4.800 ảnh và nhãn 3D kèm ID; R4 có camera cố định lẫn gắn xe, 5.000 ảnh được gán nhãn và 30.000 hộp 3D kèm ID. Cần cố định cách dùng nhãn 2D hoặc chiếu hộp 3D, khả kiến trong từng camera và hiệu chuẩn. Trang tác giả hiện yêu cầu đăng ký. Người dùng đã cho biết đang chờ tài khoản được duyệt trong trao đổi trước.

- [Cổng tác giả](https://innovation-mobility.com/en/project-providentia/a9-dataset/)
- [Devkit](https://github.com/tum-traffic-dataset/tum-traffic-dataset-dev-kit)

## GRAM-RTM Urban1

Đã kiểm toàn bộ 23.435 XML trong gói annotation v4 có sẵn, CRC khi đọc từng thành viên và SHA256 archive khớp R106. Có 59.693 hộp, 243 ID, không trùng frame/ID. Trường occluded có 7.766 hộp giá trị 1. Tiêu chí góc ảnh R99 hiện tại chỉ tìm 52 khung-xe trong hai đoạn rẽ ≥25°, và 24 khung-xe trong một đoạn ≥35°. Có thể dùng làm kiểm tra che khuất ngoài miền; số mẫu rẽ này chưa thích hợp làm tập chính cho phi tuyến.

Gói XML thực có 58.542 hộp car, 740 motorbike và 411 bus. Những con số và lớp này khác mô tả cũ trên trang web tác giả, nên phải lấy định nghĩa từ bản gói đã ghim và kiểm tra hình/nhãn trước khi chọn phương tiện hai bánh. Link ảnh Urban1 chính thức hiện chuyển tới đăng nhập Microsoft của trường. Chưa có kết quả tracking mới. Nhãn và thống kê ở gram_full_audit.json, gram_Urban1_full_gt.csv.

- [Nguồn và đường tải](https://gram.web.uah.es/data/datasets/rtm/index.html)

## D2-City

Bài báo tác giả mô tả dashcam, detection và tracking, có nhãn che khuất và cắt biên. Portal outreach.didichuxing.com/d2city/ hiện không truy cập được trong lần kiểm này. Chưa xác minh được gói đang tải, không đưa vào hàng chờ test trước các nguồn có schema và đường tải đã kiểm.

- [Bài báo tác giả](https://arxiv.org/abs/1904.01975)

## Thứ tự thực hiện

1. Giữ HELMET và SVMOT làm đối chứng có sẵn cho xe máy và che khuất.
2. METEOR đã có pilot và adapter khả dụng, mở rộng mẫu theo nhãn sự kiện được khóa trước điểm để kiểm camera xe và xe máy.
3. V2X-Seq-SPD khi đường mẫu tải lại được, kiểm nhãn và độ phủ sự kiện trước khi inference.
4. TUMTraf khi được cấp quyền, hoặc CityFlow-V2 nếu cần thêm nhiều nút giao cố định.
5. KITTI cho camera xe, báo riêng ảnh hưởng chuyển động camera.

Ảnh chụp nguồn và kết quả HTTP nằm trong dataset_sources.json và các source_*.txt. Một số trang HTTP 403 với requests nhưng nội dung đọc được qua công cụ web, vì vậy tình trạng này chỉ phản ánh đường truy cập đã thử. TrafficMOT đã có trong lịch sử dự án, không coi là dataset mới của R112.
