# R112 — SVMOT, PF và dữ liệu giao thông bổ sung

Hoàn tất ngày 18/11/2026 theo yêu cầu trực tiếp của người dùng. Phần mềm và phép chấm cũ được giữ làm đối chứng. Kết luận chính là detector giới hạn mạnh nhóm che khuất SVMOT, RBPF cải thiện rõ trên UA dev nhưng chưa chuyển được lợi thế IDSW sang SVMOT. Đã cung cấp một backend nghiên cứu chạy được từ runner chuẩn, chưa thay cấu hình mặc định.

## 1. Vì sao SVMOT không lặp lại ưu thế ở mọi nhóm

Đã dựng lại lịch sử ghép CLEAR trên 30 chuỗi cho KF/EKF/UKF, đối chiếu đúng IDSW và FN từng clip, tổng cộng 90 phép kiểm. Trên test chính thức 9 chuỗi, IDSW toàn video là 435/407/418, riêng khung visibility ≤0,5 là 125/115/125. Trên 21 chuỗi train bổ sung, toàn video là 1.189/1.162/1.162 nhưng riêng khung che là 149/152/156. Hai kết quả trả lời hai câu hỏi khác nhau, không mâu thuẫn số học.

Trong 428 ID có che ở train21, so EKF với KF có 10 ID giảm đổi ID, 404 bằng, 14 tăng. UKF có 7/404/17. Không thấy một lỗi chung khiến mọi track của EKF/UKF đều kém. Số liệu chi tiết theo chuỗi, xe và sự kiện nằm trong svmot_per_clip.csv, svmot_per_id.csv, svmot_events.csv.

Nếu tính thêm sự kiện ở khung mà GT đã che trong một giây trước đó, IDSW train21 thành 474/454/455. Đây là chẩn đoán hậu kiểm về thời điểm tái xuất hiện, không thay định nghĩa đã khóa hoặc dùng làm kết quả xác nhận mới. Nó cho thấy chỉ xét visibility tại đúng khung đổi ID có thể bỏ qua phần sau che.

Vấn đề lớn hơn là quan sát đầu vào. Cùng cache YOLOv8n COCO c25, chỉ 49,03% GT khả kiến thấp test9 và 46,40% train21 có ít nhất một detection IoU ≥0,5. Với KF, khoảng 93,88% FN trong nhóm này không có detection đạt điều kiện đó. Trên test9 là 6.116/6.515, train21 là 11.863/12.637. Đây là độ phủ candidate không ghép một-một, không phải AP hay recall detector chính thức. Bộ lọc khó khắc phục phần lớn những mẫu thiếu quan sát chỉ bằng cách thay cập nhật trạng thái.

Chẩn đoán đầy đủ ở [03_svmot_diagnosis.md](03_svmot_diagnosis.md), số máy đọc ở svmot_diagnosis.json. Chưa xác lập nguyên nhân ở mức từng phép gán nội bộ hoặc quan hệ nhân quả chỉ từ lịch sử GT ghép.

## 2. PF đã sửa theo hướng nào và kết quả ra sao

PF cũ lấy mẫu cả chín tọa độ CTRV. K1 cập nhật bốn tọa độ đo nhưng không lưu Gaussian điều kiện đầy đủ cho bảy tọa độ và tương quan vị trí–vận tốc qua thời gian. Với 500 hạt, sai số lấy mẫu, resampling và roughening có thể làm hộp dự đoán lệch qua ngưỡng ghép rời rạc. PF tổng quát hơn không bảo đảm thắng KF trong dữ liệu gần tuyến tính, Gaussian. Đây là lý giải theo cấu trúc mã và bằng chứng thực nghiệm, chưa phải chứng minh một nguyên nhân duy nhất.

Ứng viên Gaussian RBPF chỉ lấy mẫu hướng và tốc độ quay. Bảy tọa độ còn lại có trung bình và covariance Gaussian riêng cho mỗi hạt. Cập nhật dùng gain đầy đủ, covariance dạng Joseph, likelihood có covariance dự đoán và ESS resampling. Q phụ thuộc chiều cao được xấp xỉ tại trung bình từng Gaussian; vẫn chặn chiều cao dương. Không dùng K1/K2. Phép so thay cả phương pháp suy luận, chưa tách đóng góp từng thành phần. Nguồn phương pháp là [Doucet và cộng sự](https://arxiv.org/abs/1301.3853).

### UA-DETRAC train dev12

Cùng cache COCO c25, 500 hạt, CTRV, ghép ByteTrack và ba seed 0/1/2. Đã chạy đủ 36 lượt, 50.757 khung replay. Đối chứng đầy đủ lấy từ R72, kiểm parity prefix PF cũ trước chạy. Đây là dữ liệu phát triển đã được xem nhiều lần.

- PF500 K1/K2 motion2 → Gaussian RBPF, IDSW trung bình 697 → 469, giảm 32,71%.
- HOTA 57,0057% → 58,9635%, IDF1 68,3610% → 71,7956%.
- FN 39.395,67 → 37.431,33, giảm 4,99%. FP 13.485 → 12.113,33.
- IDSW từng seed mới 480/467/460 so với nền 689/702/700. 11 clip giảm, một clip tăng.
- KF/EKF/UKF cùng phép so có IDSW 489/472/474. RBPF mới đã gần nhóm này trên dev, không có nghĩa thắng ở mọi clip hoặc tập khác.

Đạt cả năm tiêu chí dev đã khóa. Chi tiết ở dev_gate.json, kết quả raw trong dev/evaluation/.

### SVMOT test9, kiểm chuyển miền

Chạy lại cả PF nền và ứng viên trên đủ 9 chuỗi × 3 seed × 2 phương pháp, 54 lượt, cùng cache và cấu hình quan sát.

- HOTA 50,2753% → 51,0145%, IDF1 52,7412% → 53,6720%.
- IDSW 434 → 440, tăng 1,38%. Ở khung khả kiến thấp 125,67 → 136.
- FN 14.756 → 14.633, FP 22.080 → 22.383,33. MOTA 46,9240% → 46,6586%.

Ứng viên trượt cổng IDSW và điều kiện không tăng ở mọi seed. Không chọn thay mặc định toàn miền, không bỏ kết quả bất lợi. Đây cũng là tập đã xem trong các vòng trước. Chi tiết ở transfer_svmot/verified_summary.json và verification.json.

### Thử Student-t chống ngoại lai

Sau khi thấy kết quả Gaussian, khóa một biến thể ν=4, không quét tham số theo SVMOT. Dùng trọng số Student-t và tăng R theo innovation, cập nhật Gaussian điều kiện là xấp xỉ. Nguồn tham khảo [Roth và cộng sự](https://arxiv.org/abs/1703.02428), không tự nhận triển khai trọn thuật toán bài báo.

Chạy đủ 36 lượt dev. IDSW 697 → 617, HOTA 57,0057% → 56,7585%, nhưng FN 39.395,67 → 42.127,33, tăng 6,93%. Trượt cổng FN nên dừng, không chạy SVMOT hoặc dò thêm ν. Giảm IDSW bằng cách bỏ sót nhiều xe hơn không đạt mục tiêu. Xem robust/dev_gate.json và robust/00_protocol.md.

## 3. Backend nghiên cứu đã tích hợp và kiểm thế nào

Mã ở src/rbpf_ctrv.py, YAML configs/phase2/trackers/r112/rbpf500.yaml, tên bytetrack_rbpf_ctrv. Runner nhận diện seed và fingerprint bao gồm module mới. Gaussian RBPF khả dụng để phát triển, không được trình bày như ứng viên đã vượt cổng SVMOT. Student-t vẫn nằm trong thư mục nghiên cứu, chưa đăng ký dùng thường xuyên.

Kiểm số học gồm Kalman tham chiếu 30 bước khi cố định góc và cùng Q, covariance PSD, góc ±π, mất quan sát, lặp cùng seed. Trên cả NumPy và Ultralytics, 12 lượt prefix gồm PF cũ/mới khớp snapshot, sáu lượt CLI đầy đủ MVI_20052 × ba seed khớp toàn bộ MOT ứng viên trong sai số văn bản 5,1e−11. Tổng kiểm CLI 4.164 khung. Tùy chọn PF cũ chưa được hỗ trợ sẽ bị từ chối thay vì bỏ qua âm thầm. Hướng dẫn ở [08_rbpf_usage.md](08_rbpf_usage.md).

Đã tái hợp độc lập 24 nhóm raw bằng TrackEval HOTA/Identity và tổng trực tiếp IDSW/FN/FP. Trước tích hợp, kiểm 95 hash Gaussian, 99 hash robust, 127 hash chẩn đoán SVMOT. Sau tích hợp chỉ run_tracking trong manifest 99 tệp đổi đúng dự kiến, 98 hash còn nguyên. Manifest cũ không có run_fingerprint, hash module này được ghi trong provenance tích hợp mới. Không viết lại manifest thí nghiệm để hợp thức hóa mã mới. Xem verification.json, robust/verification.json, postintegration_provenance.json, registered_np.json, registered_ultralytics.json.

Không đo FPS RBPF trên Nano trong lượt này. Thêm module vào fingerprint có thể khiến runner yêu cầu chấm lại khi người dùng chủ động chạy resume, nhằm tránh tái dùng metric của mã khác. Không sửa các kết quả lịch sử hoặc launcher. Các lần lỗi harness ban đầu và cách phục hồi được ghi trong [01_method_limit.md](01_method_limit.md).

## 4. Dataset mới đã làm đến đâu

METEOR là dữ liệu dashcam có xe máy, scooter và hành vi rẽ/chuyển làn từ [nguồn tác giả](https://github.com/GAMMA-UMD/METEOR). Đã dùng HTTP Range lấy riêng video và nhãn từ kho dữ liệu tác giả liên kết, ghim revision, kiểm CRC và SHA256, không tải cả 93 GB. Tệp tại F:/LABOTARY_data/goc_tai_ve/METEOR-R112/REC_2020_07_12_02_55_27_F.MP4.

Clip được chọn trước điểm tracking có 1.800 khung 30 fps, 2.023 hộp MotorBike/Scooter, 10 ID. Có bốn đoạn đạt góc ảnh ≥25°, nhưng mọi hộp xe máy ở clip này có occluded=0. Độ phủ candidate detector chỉ 31,19%. Pilot KF/EKF/UKF có IDSW 16/23/19, HOTA 18,17/17,51/17,43%. PF/RBPF trung bình ba seed cùng IDSW 17,67, HOTA 16,80/19,47%. Không kết luận EKF/UKF mạnh hơn KF từ clip này, không đại diện toàn METEOR. Kịch bản góc ảnh còn chịu chuyển động camera.

Các lựa chọn tiếp theo đã kiểm nguồn

- [V2X-Seq-SPD](https://github.com/AIR-THU/DAIR-V2X-Seq) có camera cố định và camera xe, 2D box, track_id, occlusion, Motorcyclist. Mẫu chính thức trả quota exceeded khi tải thực, chưa có archive hoàn chỉnh hoặc điểm tracking mới.
- [TUMTraf R2/R4](https://innovation-mobility.com/en/project-providentia/a9-dataset/) có giao lộ và chuyển động quay đầu, phù hợp phi tuyến/che khuất. Cần quyền tải và kiểm quy tắc chiếu nhãn 3D sang từng camera. Người dùng đã nói đang chờ duyệt tài khoản.
- [CityFlow-V2](https://www.aicitychallenge.org/2022-data-and-evaluation/) bổ sung nhiều giao lộ cố định. Cần quy tắc ROI/ignore vì chỉ xe qua ít nhất hai camera được chú thích. Chưa tải/chạy mới.
- [KITTI Tracking](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) dùng camera xe và có nhãn occlusion, phù hợp đối chứng ô tô. Không coi đây là benchmark xe máy.
- [GRAM Urban1](https://gram.web.uah.es/data/datasets/rtm/index.html) đã audit đủ 23.435 XML của gói nhãn v4, 59.693 hộp/243 ID, 7.766 hộp occluded. Chỉ có hai đoạn rẽ đạt tiêu chí hiện tại nên hợp kiểm che hơn làm tập chính phi tuyến. Link ảnh chuyển tới đăng nhập trường, chưa tracking mới.

Thông số gói, khác biệt nhãn với mô tả cũ, đường tải và tình trạng truy cập được giữ ở [02_datasets.md](02_datasets.md), dataset_sources.json, gram_full_audit.json, meteor_pilot/verified_summary.json. Chỉ tính METEOR là dữ liệu mới đã tải và chạy pilot trong lượt này.

## 5. Việc tiếp theo có căn cứ

Mở rộng METEOR theo mẫu khóa trước điểm, thống kê nhãn sự kiện và độ phủ detection trước khi chạy nhiều. Ưu tiên cải thiện quan sát xe máy/che khuất khi độ phủ quá thấp. Thử V2X khi tải được hoặc TUMTraf khi được cấp quyền để tăng số cảnh phi tuyến độc lập. Khóa định nghĩa sự kiện sau che trước lượt dữ liệu mới, giữ điểm toàn video bên cạnh điểm từng nhóm. Với PF, cần tách các thành phần RBPF và kiểm trên dữ liệu ngoài dev, không chọn ngưỡng theo nơi bộ lọc đã thắng.
