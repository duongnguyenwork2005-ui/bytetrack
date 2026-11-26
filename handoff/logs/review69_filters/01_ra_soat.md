# R69 — PF mất độ chính xác ở bước nào?

## Trên video đã chấm

Trong phần quỹ đạo nối được với GT trên RoundaboutHD, trên 35.465 bộ (video, GT, frame) chung, sai số tâm trung vị ở khung kế tiếp là 0,095–0,096 lần chiều cao hộp GT với PF500, so với 0,060–0,063 với các bộ lọc còn lại. Khi thiếu ít nhất ba quan sát, chỉ 46–59% trường hợp có tâm GT nằm trong bán kính bằng hai lần độ trải của đám hạt quanh tâm dự đoán. Sai số dự đoán làm giảm IoU và có thể khiến ByteTrack từ chối một detection đúng, trước lúc bộ lọc được cập nhật. Trong mẫu chẩn đoán, tỷ lệ có detection phù hợp nhưng chi phí ghép vượt ngưỡng là 8,6–8,8% với PF và 3,6–3,7% với các đối chứng. Các trường hợp này tập trung ở track Lost. Đây là phép đọc lại [chẩn đoán R65](../review65_pf_roundabout/10_ket_luan.md).

## Đối chiếu đường chạy

- KF/CV trong ma trận ngoài dùng `bytetrack.yaml` của Ultralytics, trạng thái hộp XYAH và bốn vận tốc. EKF dùng CTRV, Jacobian và cập nhật tuyến tính cho phép đo XYAH. UKF dùng cùng mô hình CTRV, truyền 19 điểm sigma, tính trung bình và phần dư góc theo vòng tròn. Kiểm tra Jacobian, moment, hiệp phương sai và đường xử lý theo lô đã qua các fixture hiện có. Phép so KF với các biến thể CTRV gồm cả ảnh hưởng của mô hình chuyển động và phương pháp ước lượng.
- PF500 trong ma trận dùng bootstrap, một đám hạt cho mỗi ID, lấy ước lượng trung bình có trọng số trước resampling. Bước dự đoán giữ trọng số và truyền từng hạt qua CTRV. Khi Lost, vận tốc chiều cao được đặt về 0 trước khi truyền. Bước đo nhân likelihood trong miền log. Khi ESS dưới 250, systematic resampling chọn lại 500 hạt. Trên R65, ESS/N trung vị 0,19 và còn khoảng 36% tổ tiên phân biệt sau resampling. Việc lặp tổ tiên có thể làm mất giả thuyết vận tốc/hướng trong đoạn dài thiếu đo. ESS đo độ tập trung trọng số, tự nó chưa xác định nguyên nhân mất ID.
- Các bộ lọc đưa một hộp đại diện cho bước ghép IoU. Trong backend thực nạp, `STrack.update` và `re_activate` chuyển phép đo vào bộ lọc mà không truyền confidence. Điểm detection vẫn tham gia ghép theo cấu hình fuse-score. EKF/UKF tính R từ chiều cao dự đoán, PF dùng chiều cao quan sát. Với cùng hệ số nhiễu, nếu chiều cao đo nhỏ bằng nửa dự đoán thì phương sai đo tâm của PF chỉ bằng một phần tư. Q của PF còn thay đổi theo chiều cao từng hạt. Vì vậy các mô hình có cùng dạng công thức nhưng các ma trận số có thể khác.

## Hai kiểm tra đích danh

1. **Bảo toàn phần trạng thái không được khởi tạo lại.** Trong `src/pf_ctrv.py`, hàm `PFTrackView.try_initiate_from_motion` chỉ thay vận tốc và hướng trong đám hạt, sau đó đặt toàn bộ trọng số đều. Với 250 hạt ở x=96 mang tổng trọng số 0,75 và 250 hạt ở x=104 mang 0,25, ESS=400 vượt ngưỡng resampling. Bước khởi tạo này làm tâm x tăng từ 98 lên 100 trước khi áp likelihood mới. Có thể sửa trong một biến thể riêng bằng cách giữ logw khi vẽ lại vận tốc/hướng, hoặc resample cả trạng thái trước khi chuyển về trọng số đều. Tiêu chí kiểm là giữ phân bố biên của phần trạng thái còn lại. Cảnh tích hợp đứng yên bảy khung rồi đi, PF500 mặc định, cho trọng số đã đều ở 30/30 seed lúc khởi tạo. Kiểm lại **12 log pf_diag R65 có sẵn** theo đúng thứ tự sự kiện của cùng ID trong mỗi camera. Dự đoán giữ logw. Trong 1.654 / 1.714 / 1.680 lần init theo ba seed, 4.854 lần ngay sau kích hoạt track và 194 lần sau một lần update đặt đều trọng số. Cả **5.048/5.048** đã đều, nên thao tác đặt đều này không giải thích chênh lệch R65. [Bộ đếm từ log](check_saved_init.py) và [kết quả từng camera](saved_init.json).
2. **Proposal dùng phép đo (K1).** Fixture dùng prior không đều, Q khác nhau theo hạt, so trọng số cập nhật với mật độ dự đoán Gaussian tính độc lập. Sai số tuyệt đối tối đa dưới 2e-17. K1 được gọi 22 lần mỗi lượt K1 trong hai cảnh dưới, không fallback. K1 dùng detection đã ghép để lấy mẫu lại bốn tọa độ đo. Vì cập nhật diễn ra sau ghép cặp, nó chỉ có thể tác động tới chi phí ghép của các khung tiếp theo. K2 thêm nhiễu vào hạt sau resampling. Tăng N lấy nhiều mẫu hơn từ cùng mô hình. R48D đã thử các lựa chọn này trên dev12, K1 giúp PF100 nhưng cả PF500-K1 vẫn thấp hơn EKF khoảng 3,8 điểm HOTA trên dev12. Xem [báo cáo R48](../review48/07_tong_hop_bao_cao_17_10.md).

## Mô phỏng qua sáu khung mất quan sát

Một xe có hộp 40×30 px, tốc độ 3 px/bước. Hai cảnh có cùng đầu vào cho mọi bộ lọc. Cảnh đầu đi thẳng, cảnh sau đổi hướng 0,06 rad/bước từ khung 10. Bỏ detection ở khung 23–28 và đo ngay trước khi ghép ở khung 29. Dùng ba seed 0, 1, 2 cho PF. Sai số tâm dưới đây tính bằng pixel, với PF báo trung vị và khoảng theo seed.

- Đi thẳng: KF 0,30; EKF 0,10; UKF 0,16; PF500 2,43 [1,77–9,57].
- Đang cua: KF 12,95; EKF 4,61; UKF 4,52; PF500 4,22 [1,11–4,63].
- Ở cảnh cua, trung vị của PF500-K1 là 7,95; PF500-K2 là 11,13; PF1000 là 3,35 với khoảng 1,11–15,63. Mọi cấu hình đều giữ đúng ID khi xe tái xuất hiện. Đây là hai quỹ đạo của một xe với quan sát chính xác ngoài đoạn mất đo. Không dùng các số này để ước lượng mức thay đổi HOTA, chọn seed hoặc cấu hình thắng trên dữ liệu thực.

## Khi các hộp quan sát có nhiễu
Giữ cùng hai quỹ đạo và sáu khung mất quan sát. Cộng nhiễu Gaussian vào tâm x/y (độ lệch chuẩn 1 px) và kích thước w/h (1,5 px). Mọi bộ lọc và mọi lượt nhận **cùng một chuỗi nhiễu**. Dùng 30 seed PF từ 0 đến 29. Đo sai số tâm ở khung 29 trước khi ghép và kiểm ID sau khi ghép. Mỗi lần mất dấu ở đây chỉ mô phỏng việc detector không trả hộp.

- Khi cua, KF đạt 14,03 px, EKF 5,94 px và UKF 5,91 px. PF500 bootstrap có trung vị 10,05 px, p90 29,83 px, nối lại đúng ID 24/30 lượt. Cùng 500 hạt, K1 đạt trung vị 5,53 px, p90 10,71 px, đúng ID 30/30. Tăng bootstrap lên 1.000 hạt đạt trung vị 8,98 px và đúng ID 29/30.
- Khi đi thẳng, KF 2,41 px, EKF 3,17 px, UKF 3,26 px. PF500 bootstrap đạt trung vị 8,30 px (26/30 đúng ID), PF500-K1 đạt 4,17 px (30/30), PF1000 đạt 4,57 px (29/30).
- Qua các cập nhật từ khung 2 đến 22, lấy trung vị tỷ lệ số hạt cha phân biệt sau resampling trên N trong mỗi lượt, rồi trung vị qua seed. Cảnh cua cho 0,403 với PF500 bootstrap và 0,592 với K1. Khung không resample có tỷ lệ 1. Điều này phù hợp với giả thuyết K1 duy trì nhiều giả thuyết chuyển động hơn, chưa tách được đóng góp nhân quả của độ đa dạng so với việc K1 lấy mẫu vị trí theo quan sát. Sai số trên 30 seed chỉ phản ánh ngẫu nhiên của bộ lọc trên một chuỗi hộp cho mỗi cảnh.

Lệnh tái hiện là `python -B handoff/logs/review69_filters/probe.py --noisy`. [Số liệu từng seed](noisy.json) và [biểu đồ phân bố sai số](pf_noisy_comparison.png).

**Ưu tiên vòng sau.** Thử PF500-K1 đối chiếu bootstrap với cùng Q, R, detections và association trên tập phát triển khi mở lại thực nghiệm. Giữ PF500 hiện hành làm mốc. Phân tầng số khung mất quan sát và so sai số tâm ngay trước ghép, p90 cùng tỷ lệ nối lại đúng ID. Sửa tính nhất quán của trọng số lúc khởi tạo là việc độc lập. Nếu K1 vẫn còn lệch, có thể tách riêng giả thuyết R dùng chiều cao đo so với dự đoán.

Tái hiện từ gốc repo bằng Python môi trường khoaluan-mot: `python -B handoff/logs/review69_filters/probe.py`. [Số liệu từng seed và fixture](probe_result.json). Các bộ kiểm sẵn dùng là `test_filter_config.py` (44/44 assertion), `test_ekf_ctrv.py` (14/14), `test_ukf_ctrv.py` (12/12), `kc48/test_pf_dung_dan.py` (29/29) và `kc48/chay_test_pf_cu.py` (21/21, chọn 10 test, bỏ fixture số 11 có ghi đè cache). R69 giữ mã bộ lọc và cấu hình benchmark ở trạng thái đã khóa.

## Dữ liệu đã tải bổ sung

Đã chọn René-Lévesque và St-Marc để bổ sung góc camera giao lộ cho Sherbrooke. Nhãn có ID ô tô và xe đạp. René-Lévesque nhìn ba giao lộ, St-Marc có dòng giao thông hỗn hợp. [Trang dữ liệu Urban Tracker của tác giả](https://www.jpjodoin.com/urbantracker/dataset.html).

Các tệp nằm tại `F:/LABOTARY_data/goc_tai_ve/UrbanTracker-R69/`.

- **St-Marc** có `stmarc_frames.zip` 225.012.984 byte, chứa 2.000 JPG, cùng `stmarc_annotations.zip`. GT tổng hợp gồm 8.375 hộp, 28 ID ở các frame 1000–1999. SQLite riêng có 8 ID trong cars, 2 trong bike và 18 trong pedestrians. Trang giới thiệu ghi 7 xe, 2 xe đạp, 19 người. Cần dùng số đếm từ nhãn thực và kiểm ánh xạ lớp trước khi lập giao thức.
- **René-Lévesque** có `rene_video.mov` 199.595.856 byte, `rene_annotations.zip` và `rene_mask.png`. Video đã giải mã đủ 8.501 khung ở 1280×720, báo FPS 30000/1001. GT tổng hợp có 9.418 hộp, 29 ID (27 trong cars, 2 trong bike), frame 7200–8199. Trang giới thiệu ghi 29 xe và 2 xe đạp, README ghi mốc cuối 8200. Nhãn SQLite thực kết thúc ở 8199. Gói ảnh gốc `rene_frames.zip` cũng đã tải đủ 1.569.006.177 byte, chứa 8.501 JPG liên tiếp từ 00000001 đến 00008501, CRC đạt.

Toàn bộ ZIP hoàn chỉnh qua CRC, 7 cơ sở dữ liệu SQLite qua integrity_check, PNG đọc hợp lệ. Cả hai gói ảnh đã giải mã thử JPG đầu, giữa và cuối, đều 1280×720. Thư mục tải đã hết tệp .partial. [Danh sách URL, SHA256 cục bộ và số đếm](datasets.json) được tạo bởi [bộ kiểm tệp](verify_downloads.py). SHA256 là giá trị tính trên tệp tải được.

Trước khi chấm cần kiểm số frame trong nhãn tương ứng chỉ số ảnh/video nào, vùng ROI và FPS hiệu dụng St-Marc (trang ghi 30 fps, ZIP thiếu metadata thời gian). Hai cảnh cung cấp thêm dữ liệu ô tô và xe đạp; mức phù hợp cho nghiên cứu xe máy cần dựa vào các tập xe máy hiện có.

[Manifest mã và môi trường của kiểm tra PF](manifest.json) lưu Python, phiên bản backend, cấu hình và SHA256 của mã thực nạp.
