# Manifest tài nguyên slide khóa luận ByteTrack

Ngày tạo: 2026-09-10

Nhánh khi tạo: `fix/tracker-evaluation-correctness`

Commit nguồn khi tạo: `1dfe63dc7d4823abdff9e3574e3c600a400c08a9`

Thư mục này chỉ chứa tài nguyên trình bày được tạo từ dữ liệu và kết quả hiện có. Không có tracking, đánh giá hoặc bootstrap nào được chạy lại; không có file trong `data/` hay `results/` bị sửa.

## Đường dẫn nguồn

- Ảnh UA-DETRAC đã giải nén: `data/extracted/DETRAC-Images/<video>/imgNNNNN.jpg`.
- GT test theo định dạng MOT: `data/processed/DETRAC-test/<video>/gt/gt.txt`.
- Tracking lượt A: `data/processed/trackers/DETRAC-test/testA-<model>/data/<video>.txt`.
- Tracking lượt B: `data/processed/trackers/DETRAC-test/testB-<model>/data/<video>.txt`.
- `<model>` nhận một trong ba giá trị: `cv`, `ekf_ctrv`, `ukf_ctrv`.
- Danh sách 40 video và cấu hình khóa trước khi chạy: `results/test_eval/manifest.json`.
- Chỉ số tổng hợp: `results/test_eval/summary/configuration_metrics.csv`.
- Khoảng tin cậy và phép so sánh ghép cặp: `results/test_eval/summary/paired_contrasts_bootstrap.csv`, `results/test_eval/summary/configuration_bootstrap.csv`.
- Kiểm tra tính nhất quán: `results/test_eval/summary/validation.csv`, `results/test_eval/summary/metadata.json`.
- Sự kiện che khuất và kết cục lượt B: `results/test_eval/B/events_DETRAC-test.csv`, `results/test_eval/B/event_outcomes_DETRAC-test.csv`, `results/test_eval/B/paired_DETRAC-test.csv`.
- Phân đoạn che khuất GT: `data/interim/occlusion_segments_test.csv`, `data/interim/full_occlusion_segments_test.csv`.

Mỗi thư mục tracking A/B × CV/EKF/UKF có 40 file video. `validation.csv` xác nhận TrackEval và sự kiện đều phủ 40 video cho cả sáu cấu hình; 39 video có ít nhất một sự kiện đủ điều kiện.

## Slide 9 — ảnh dataset

### `slide09_dataset/dataset_overview_MVI_39031_f00001.png`

- Nguồn: `data/extracted/DETRAC-Images/MVI_39031/img00001.jpg`.
- Video/frame: `MVI_39031`, frame 1.
- Kích thước: 960 × 540 px, RGB.
- Lý do chọn: khung toàn cảnh giao thông nhiều làn, thể hiện mật độ và góc nhìn giám sát của UA-DETRAC.
- Đây là ảnh dataset gốc, không có GT hoặc kết quả tracking chồng lên ảnh. PNG khớp từng pixel RGB với JPG sau khi giải mã.

### `slide09_dataset/dataset_occlusion_MVI_39401_f00989.png`

- Nguồn: `data/extracted/DETRAC-Images/MVI_39401/img00989.jpg`.
- Video/frame: `MVI_39401`, frame 989.
- Kích thước: 960 × 540 px, RGB.
- Lý do chọn: xe buýt che khuất rõ một phương tiện trong cảnh giao thông đô thị.
- Đây là ảnh dataset gốc, không có GT hoặc kết quả tracking chồng lên ảnh. PNG khớp từng pixel RGB với JPG sau khi giải mã.

## Slide 13 — biểu đồ kết quả

Hai biểu đồ đọc trực tiếp `results/test_eval/summary/configuration_metrics.csv`. Tỷ lệ trong CSV được nhân 100 để trình bày theo phần trăm; số đếm được giữ nguyên. Lượt A dùng `conf=0.25`, lượt B dùng `conf=0.10`. CV có màu xanh dương, EKF–CTRV màu cam, UKF–CTRV màu tím; A dùng sắc độ nhạt và B dùng sắc độ đậm.

### `slide13_results/metrics_hota_idf1_assa.png`

- Kích thước: 1920 × 1080 px, nền trắng, tỷ lệ 16:9.
- Ba ô biểu diễn HOTA, IDF1 và AssA cho sáu cấu hình.
- Trục tung được thu hẹp riêng theo từng chỉ số và được ghi chú ngay trên hình, vì vậy chỉ dùng để so sánh điểm ước lượng trong cùng ô.
- Không đánh dấu mô hình “tốt nhất” hoặc suy diễn ý nghĩa thống kê.

### `slide13_results/tradeoffs_ids_fp_fn_continuity.png`

- Kích thước: 1920 × 1080 px, nền trắng, tỷ lệ 16:9.
- Bốn ô biểu diễn IDSW, FN, FP và `id_continuous`.
- Mỗi ô có trục tung riêng. Biểu đồ cho thấy đúng đánh đổi quan sát được khi chuyển từ A sang B: IDSW và FN giảm, `id_continuous` tăng, còn FP tăng ở cả ba motion model.

### Giá trị dùng trong hai biểu đồ

| Lượt | Model | HOTA (%) | IDF1 (%) | AssA (%) | IDSW | FP | FN | id_continuous (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | CV | 58,48 | 70,22 | 64,17 | 2.500 | 72.646 | 198.712 | 29,68 |
| A | EKF–CTRV | 58,60 | 70,47 | 64,43 | 2.532 | 72.314 | 198.835 | 29,68 |
| A | UKF–CTRV | 58,49 | 70,29 | 64,21 | 2.549 | 72.421 | 198.844 | 29,76 |
| B | CV | 58,22 | 69,41 | 63,58 | 2.082 | 99.903 | 180.571 | 36,63 |
| B | EKF–CTRV | 58,55 | 69,94 | 64,30 | 2.063 | 99.579 | 180.532 | 37,01 |
| B | UKF–CTRV | 58,40 | 69,72 | 63,97 | 2.110 | 99.731 | 180.446 | 37,01 |

Các giá trị trên hình được làm tròn tới hai chữ số thập phân; dấu chấm trong số đếm là dấu phân tách hàng nghìn theo cách trình bày tiếng Việt.

## Slide 14 — tình huống trực quan

Hai bảng ảnh dùng cùng GT, cùng ba frame và cùng crop giữa CV, EKF–CTRV và UKF–CTRV. Khung GT dùng nét đứt xanh lá; khung tracking thực dùng màu của từng model. Đây là kết quả tracking thực được đối chiếu với GT, không phải minh họa ngoại suy bằng GT.

`id_match` chỉ cho biết ID ghép trước và sau sự kiện giống nhau. `id_continuous` yêu cầu ID đó được duy trì liên tục theo định nghĩa của script trong khoảng đánh giá. Hai khái niệm không thay thế cho nhau.

### `slide14_case_study/case_different_MVI_40701_GT21.png`

- Kích thước: 1920 × 1080 px, RGB.
- Video/GT/sự kiện: `MVI_40701`, GT ID 21, event 1.
- Sự kiện partial: frame 343–443, 101 frame; sự kiện có che khuất hoàn toàn.
- Đoạn full tương ứng: frame 370–415, 46 frame; quãng đường tâm bbox trong đoạn full 87,94 px, độ dời đầu-cuối 87,31 px.
- Ba frame trình bày: trước 342, trong 392, sau 444.
- Occluder được gắn nhãn `vehicle` trong dữ liệu sự kiện.

| Model | ID trước | ID tại frame 392 | ID sau | Kết cục | id_match | id_continuous | recall_during |
|---|---:|---|---:|---|---|---|---:|
| CV | 328 | không có ghép IoU ≥ 0,5 | 423 | switched | false | false | 0,3366 |
| EKF–CTRV | 335 | không có ghép IoU ≥ 0,5 | 502 | switched | false | false | 0,3366 |
| UKF–CTRV | 329 | không có ghép IoU ≥ 0,5 | 329 | preserved | true | false | 0,3366 |

IoU đối chiếu trực tiếp tại frame trước/sau lần lượt là: CV 0,924/0,925; EKF–CTRV 0,923/0,907; UKF–CTRV 0,913/0,924. Ở frame 392, IoU lớn nhất với GT của cả ba cấu hình nhỏ hơn 0,5. Ví dụ được chọn vì ba model cho kết cục trước/sau khác nhau trong cùng một sự kiện che khuất dài. UKF–CTRV có `id_match=true` nhưng `id_continuous=false`, giúp minh họa rõ sự khác nhau giữa hai chỉ số. Đây là một trường hợp mô tả, không chứng minh UKF–CTRV tốt hơn trên toàn tập.

Nguồn cụ thể:

- Ảnh: `data/extracted/DETRAC-Images/MVI_40701/img00342.jpg`, `img00392.jpg`, `img00444.jpg`.
- GT: `data/processed/DETRAC-test/MVI_40701/gt/gt.txt`.
- Tracking: `data/processed/trackers/DETRAC-test/testB-{cv,ekf_ctrv,ukf_ctrv}/data/MVI_40701.txt`.
- Kết cục: `results/test_eval/B/event_outcomes_DETRAC-test.csv`.
- Đoạn full: `data/interim/full_occlusion_segments_test.csv`.

### `slide14_case_study/case_same_success_MVI_39401_GT92.png`

- Kích thước: 1920 × 1080 px, RGB.
- Video/GT/sự kiện: `MVI_39401`, GT ID 92, event 0.
- Sự kiện partial: frame 962–1016, 55 frame; sự kiện có che khuất hoàn toàn.
- Đoạn full tương ứng: frame 981–1007, 27 frame; quãng đường tâm bbox trong đoạn full 144,27 px, độ dời đầu-cuối 143,95 px.
- Ba frame trình bày: trước 961, trong 989, sau 1017.
- Occluder được gắn nhãn `vehicle` trong dữ liệu sự kiện.

| Model | ID trước | ID tại frame 989 | ID sau | Kết cục | id_match | id_continuous | recall_during |
|---|---:|---:|---:|---|---|---|---:|
| CV | 1007 | 1007 | 1007 | preserved | true | true | 1,0000 |
| EKF–CTRV | 1030 | 1030 | 1030 | preserved | true | true | 1,0000 |
| UKF–CTRV | 1032 | 1032 | 1032 | preserved | true | true | 1,0000 |

IoU đối chiếu trực tiếp ở frame trước/trong/sau là: CV 0,934/0,934/0,921; EKF–CTRV 0,934/0,935/0,921; UKF–CTRV 0,934/0,934/0,921. Ví dụ được chọn để cân bằng với trường hợp khác biệt: cả ba model cùng thành công trên một sự kiện có che khuất hoàn toàn. Nó không đại diện cho tỷ lệ thành công của toàn tập.

Nguồn cụ thể:

- Ảnh: `data/extracted/DETRAC-Images/MVI_39401/img00961.jpg`, `img00989.jpg`, `img01017.jpg`.
- GT: `data/processed/DETRAC-test/MVI_39401/gt/gt.txt`.
- Tracking: `data/processed/trackers/DETRAC-test/testB-{cv,ekf_ctrv,ukf_ctrv}/data/MVI_39401.txt`.
- Kết cục: `results/test_eval/B/event_outcomes_DETRAC-test.csv`.
- Đoạn full: `data/interim/full_occlusion_segments_test.csv`.

## Giới hạn diễn giải

- Hai case study được chọn có chủ đích để có một trường hợp khác kết cục và một trường hợp cùng thành công. Không dùng chúng để ước lượng tần suất hoặc suy luận nhân quả.
- Khoảng tin cậy `id_match` giữa các motion model đều chứa 0; dữ liệu chưa đủ bằng chứng về khác biệt và cũng không chứng minh các model tương đương.
- Ở lượt B, một số khoảng tin cậy danh nghĩa của EKF–CTRV so với CV dương cho HOTA/IDF1/AssA, nhưng chưa hiệu chỉnh nhiều so sánh và không chứng minh EKF–CTRV thắng toàn diện.
- HOTA, AssA và IDF1 của B so với A giảm theo điểm ước lượng trên test, dù B giảm IDSW/FN và tăng `id_continuous`; FP tăng rõ rệt.
- Tập test đã từng được sử dụng hoặc xem trong quá trình trước, nên không phải held-out hoàn toàn.
- Không suy luận nguyên nhân từ IDSW, FP, FN, tổng số ID hoặc hai ví dụ trực quan. Muốn kết luận cơ chế association cần log runtime về track, detection, candidate pair, cost và quyết định ghép.
- Không dùng thời gian chạy hiện có như benchmark vì tải máy không được kiểm soát.

## Kiểm tra chất lượng đã thực hiện

- Mở trực quan cả sáu PNG; tiêu đề, chú thích và nhãn ID không bị cắt hoặc chồng lấn.
- Giải mã bằng Pillow: 6/6 file hợp lệ, kích thước đúng, không file rỗng và không ảnh đơn sắc/trống.
- So sánh pixel: hai PNG slide 9 khớp ảnh JPG nguồn sau khi giải mã.
- Đọc lại sáu dòng cấu hình từ `configuration_metrics.csv` và đối chiếu mọi nhãn số trên hai biểu đồ.
- Đọc trực tiếp GT và ba file tracking lượt B cho từng case; xác nhận ID và IoU ở cả sáu frame trình bày.
- Đối chiếu `event_outcomes_DETRAC-test.csv` và `full_occlusion_segments_test.csv` để xác nhận khoảng frame, kết cục, `id_match` và `id_continuous`.

## Phần người dùng cần bổ sung trên Canva

- Logo và quy chuẩn nhận diện của trường/khoa.
- Tên sinh viên, mã số sinh viên, giảng viên hướng dẫn, hội đồng và ngày bảo vệ.
- Font, cỡ chữ, màu nền và vị trí cuối cùng theo template Canva đang dùng.
- Caption ngắn phù hợp với thời lượng thuyết trình; giữ các giới hạn diễn giải nêu trên.
- Căn lại crop trong Canva nếu template dùng khung ảnh khác 16:9, nhưng không cắt mất bbox/ID trong hai case study.
