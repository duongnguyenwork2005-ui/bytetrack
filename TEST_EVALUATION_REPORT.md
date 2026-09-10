# TEST_EVALUATION_REPORT — Đánh giá lại 40 video test UA-DETRAC

Ngày bàn giao: 10/09/2026, múi giờ Asia/Saigon. Repo `D:\LABOTARY\bytetrack`, nhánh `fix/tracker-evaluation-correctness`.

## 1. Kết quả và phạm vi

Đã hoàn tất 40 video × CV/EKF/UKF × A/B: **240/240 file tracking, không thiếu, không rỗng**. Lượt B đã kết thúc trước khi tiếp quản; không khởi chạy lại tracking. Đã chấm B bằng cùng `run_trackeval.py` và `occlusion_eval.py` của A, đối chiếu A từ CSV, rồi bổ sung tổng hợp và bootstrap ghép cặp theo video. Không sửa thuật toán, detector, ngưỡng hay dữ liệu.

**Chưa đủ bằng chứng để tuyên bố EKF/UKF vượt CV về `id_match`.** Cả bốn KTC 95% của `id_match` giữa motion model đều chứa 0; điều đó không chứng minh tương đương. Riêng ở lượt B, hiệu EKF−CV của HOTA, IDF1 và AssA có KTC danh nghĩa dương, nhưng chưa hiệu chỉnh cho nhiều so sánh và không kéo theo kết luận EKF thắng trên mọi chỉ số. B giảm IDSW/FN và tăng `id_continuous`, nhưng tăng FP; HOTA/AssA/IDF1 test B−A giảm theo điểm ước lượng, ngược xu hướng train. Phải đọc KTC từng chỉ số ở mục 5, không suy ra ý nghĩa thống kê từ dấu của điểm ước lượng.

Test này **đã từng được xem và dùng trong nghiên cứu trước** (Giai đoạn D, mục 4.6 của `BAO_CAO_KHOA_LUAN.md`). Đây là đánh giá xác nhận lại trên tập đã biết, không phải tập kiểm định hoàn toàn chưa từng sử dụng. Manifest được tạo trước lượt A/B hiện tại; điều đó không xóa lịch sử sử dụng test.

## 2. Commit, môi trường và cấu hình khóa

- Code chạy tracking/chấm điểm: **`aa45f6c87484e2194039149fd79e1b723f8691bf`**. Working tree lúc tiếp quản chỉ có chỉnh sửa đã biết ở `FIX_REPORT.md` và kết quả test chưa commit.
- Commit kết quả đánh giá là **`b5bd086f8e4a625850ab35c977d1ed40776e810d`** với thông điệp **`Chay danh gia tap test 40 video`**, parent `aa45f6c`, đã push lên `origin/fix/tracker-evaluation-correctness`. Nhánh này chưa merge vào `main`; `main` vẫn ở `568ba2a`. SHA này là commit kết quả đánh giá, không phải nhãn `HEAD` cho các cập nhật tài liệu về sau.
- Python `3.11.9`, env `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`; Windows build 26200; numpy `2.4.6`, scipy `1.17.1`, pandas `2.3.3`, torch `2.6.0+cu124`, ultralytics `8.4.141`, TrackEval `1.3.0`, OpenCV `5.0.0.93`.
- GPU NVIDIA GeForce GTX 1650 4 GB, driver `555.97`; CUDA runtime của torch `12.4` (nvidia-smi hiển thị driver hỗ trợ CUDA `12.5`). GPU không có tiến trình lúc tiếp quản.
- Weights `models/yolov8n.pt`, 6.549.796 byte; SHA-256 **`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`**, kiểm tra khớp manifest. COCO classes `[2,3,5,7]`, device `0`, mask ignored regions như pipeline hiện tại.
- A `conf=0.25`; B `conf=0.10`. `track_high_thresh=0.25`, `track_low_thresh=0.10`, `new_track_thresh=0.25`, `track_buffer=30`, `match_thresh=0.8`, `fuse_score=True`, giống nhau ở cả ba YAML. Chỉ `tracker_type` chọn CV/EKF/UKF; các tham số CTRV giữ ở code gốc.
- Che khuất: IoU ≥0,5; cửa sổ GT trước/sau 30 frame; partial ≥0,10 và full ≥0,90; nguồn segment giữ quy tắc gap=2, minimum length=2 frame. Đơn vị thống kê là sự kiện đã gộp chồng lấn/kề nhau của `(video, track_id)`.

Nguồn cấu hình: [manifest.json](results/test_eval/manifest.json), [environment.json](results/test_eval/provenance/environment.json), [locked_config_verified.json](results/test_eval/provenance/locked_config_verified.json). Hash source/config/weights nằm trong [protected_files_before.csv](results/test_eval/provenance/protected_files_before.csv).

Danh sách 40 video khóa trong manifest; trùng GT, seqmap và bộ ảnh, tổng **56.340 frame**, **675.774 bbox GT**:

```text
MVI_39031 MVI_39051 MVI_39211 MVI_39271 MVI_39311
MVI_39361 MVI_39371 MVI_39401 MVI_39501 MVI_39511
MVI_40701 MVI_40711 MVI_40712 MVI_40714 MVI_40742
MVI_40743 MVI_40761 MVI_40762 MVI_40763 MVI_40771
MVI_40772 MVI_40773 MVI_40774 MVI_40775 MVI_40792
MVI_40793 MVI_40851 MVI_40852 MVI_40853 MVI_40854
MVI_40855 MVI_40863 MVI_40864 MVI_40891 MVI_40892
MVI_40901 MVI_40902 MVI_40903 MVI_40904 MVI_40905
```

## 3. Kiểm tra hoàn tất và nguồn lệnh

Các thư mục được xác định từ **manifest và log/lệnh gốc**, không suy đoán từ tên. Sáu thư mục tracking chính xác là:

```text
data/processed/trackers/DETRAC-test/testA-cv/data/
data/processed/trackers/DETRAC-test/testA-ekf_ctrv/data/
data/processed/trackers/DETRAC-test/testA-ukf_ctrv/data/
data/processed/trackers/DETRAC-test/testB-cv/data/
data/processed/trackers/DETRAC-test/testB-ekf_ctrv/data/
data/processed/trackers/DETRAC-test/testB-ukf_ctrv/data/
```

Mỗi thư mục có đúng 40 file `<video>.txt`, **0 file rỗng**, không dư video. Đã kiểm tra toàn bộ 240 file: 7 cột số hữu hạn; frame/ID nguyên dương, frame trong seqLength; bbox có chiều rộng/cao dương; không trùng `(frame,id)`; dòng theo thứ tự frame; số bbox/ID trùng từng dòng summary. Tên ảnh liên tục từ 1 đến seqLength. `MVI_39511` có output cuối ở frame 377/380 ở cả sáu cấu hình; không dùng frame cuối có tracking làm bằng chứng thiếu ảnh, vì frame có thể không có track xác nhận.

Kiểm tra ảnh giải mã và kiểm tra MOT chi tiết: [independent_tracking_audit.json](results/test_eval/provenance/independent_tracking_audit.json), [image_decode_audit.json](results/test_eval/provenance/image_decode_audit.json).

Log gốc ghi A hoàn tất **03:22:36**, B **04:32:37**, ngày 10/09/2026, giờ +07. Lúc tiếp quản không có Python/GPU chạy tracking; tất cả summary đã đủ 40 dòng có thời gian. Vì vậy không cần resume hay chạy lại model/video nào.

**Giới hạn log cũ:** lệnh gốc dùng `2>&1 | grep ...` và không có `pipefail`. Mã 0 của shell và dòng “XONG” không tự chứng minh Python không gặp cảnh báo/lỗi bị lọc. Output và summary đầy đủ, kiểm tra MOT/ảnh không phát hiện lỗi; vẫn không thể phục hồi toàn bộ stderr lịch sử để khẳng định tuyệt đối “không cảnh báo bị bỏ qua”. Lượt chấm B hiện tại lưu toàn bộ stdout/stderr và kiểm tra exit code Python cho từng lệnh.

Lệnh tracking tương ứng với lệnh gốc (biến video lấy đúng 40 tên ở manifest; bản shell nguyên văn và dấu thời gian nằm trong [original_tracking_command_provenance.json](results/test_eval/provenance/original_tracking_command_provenance.json)):

```powershell
$py = 'C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe'
$manifest = Get-Content results/test_eval/manifest.json -Raw | ConvertFrom-Json
$videos = $manifest.danh_sach_video
foreach ($mm in $manifest.motion_model) {
  & $py src/baseline_track.py --videos $videos --split-name DETRAC-test --motion-model $mm --conf 0.25 --tracker-name "testA-$mm" --skip-existing
}
foreach ($mm in $manifest.motion_model) {
  & $py src/baseline_track.py --videos $videos --split-name DETRAC-test --motion-model $mm --conf 0.10 --tracker-name "testB-$mm" --skip-existing
}
```

Các lệnh trên chỉ ghi lại để tái lập, **không được chạy lại trong lần bàn giao này**. Cùng quy trình chấm A/B:

```powershell
foreach ($mm in $manifest.motion_model) {
  & $py src/run_trackeval.py --split-name DETRAC-test --tracker "testB-$mm"
  if ($LASTEXITCODE -ne 0) { throw 'TrackEval failed' }
}
& $py src/occlusion_eval.py --split-name DETRAC-test --part test --trackers testB-cv testB-ekf_ctrv testB-ukf_ctrv --out-dir results/test_eval/B --n-boot 5000
& $py src/summarize_test_evaluation.py
```

A đã chấm bằng các lệnh tương ứng với `testA-*`, `--out-dir results/test_eval/A`. B chạy mới; A được đối chiếu lại CSV per-video, combined, detailed, paired và event outcomes, không ghi đè kết quả chấm A.

## 4. Sáu cấu hình test

HOTA/IDF1/AssA/DetA/LocA/MOTA dưới đây dùng **thang 0–100**. Đây là kết quả **COMBINED_SEQ**, không phải trung bình điểm của 40 video. LocA lấy từ TrackEval detailed CSV, không thay bằng MOTP. HOTA được gộp đúng theo từng alpha rồi lấy trung bình, không tính `sqrt(mean(DetA) × mean(AssA))`.

| Cấu hình | HOTA | IDF1 | AssA | DetA | LocA | MOTA | IDSW | FP | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| testA-cv | 58.484 | 70.225 | 64.174 | 53.801 | 84.700 | 59.475 | 2,500 | 72,646 | 198,712 |
| testA-ekf_ctrv | 58.600 | 70.469 | 64.427 | 53.795 | 84.700 | 59.501 | 2,532 | 72,314 | 198,835 |
| testA-ukf_ctrv | 58.490 | 70.292 | 64.206 | 53.781 | 84.691 | 59.481 | 2,549 | 72,421 | 198,844 |
| testB-cv | 58.217 | 69.407 | 63.581 | 53.817 | 84.327 | 58.188 | 2,082 | 99,903 | 180,571 |
| testB-ekf_ctrv | 58.548 | 69.937 | 64.298 | 53.830 | 84.327 | 58.244 | 2,063 | 99,579 | 180,532 |
| testB-ukf_ctrv | 58.399 | 69.716 | 63.971 | 53.817 | 84.325 | 58.228 | 2,110 | 99,731 | 180,446 |

Nguồn: [configuration_metrics.csv](results/test_eval/summary/configuration_metrics.csv) và `results/trackeval/DETRAC-test/<cấu hình>/combined_metrics.csv`. Mỗi `per_video_metrics.csv` tương ứng có 40 video. Sáu detailed CSV được giữ trong [summary/trackeval_detailed/](results/test_eval/summary/trackeval_detailed/) để tái tính LocA, pooled metrics và KTC khi raw tracking không được Git lưu.

**Che khuất:** 2.174 partial + 484 full = 2.658 đoạn thô. **484/484 full nằm trong partial**; sau gộp còn **2.174 sự kiện**, không đếm full thêm lần nữa. Tập GT A/B và điều kiện đánh giá giống nhau tuyệt đối ở cả sáu cấu hình:

| Cấu hình | Tổng đã gộp | Đủ điều kiện | Bị loại | gt_missing_before | gt_missing_after |
| --- | --- | --- | --- | --- | --- |
| testA-cv | 2.174 | 1.324 | 850 | 559 | 291 |
| testA-ekf_ctrv | 2.174 | 1.324 | 850 | 559 | 291 |
| testA-ukf_ctrv | 2.174 | 1.324 | 850 | 559 | 291 |
| testB-cv | 2.174 | 1.324 | 850 | 559 | 291 |
| testB-ekf_ctrv | 2.174 | 1.324 | 850 | 559 | 291 |
| testB-ukf_ctrv | 2.174 | 1.324 | 850 | 559 | 291 |

Chỉ 39 video có sự kiện đủ điều kiện; `MVI_40774` không có. Điều kiện đủ dựa duy nhất vào GT có mặt trước và sau trong cửa sổ 30 frame. Lý do loại là nhóm loại trừ nhau theo script: nếu thiếu cả trước và sau thì ghi `gt_missing_before` trước; không diễn giải hai cột này là các tập biên độc lập.

| Cấu hình | id_match = preserved | id_continuous | switched | lost_after | no_match_before |
| --- | --- | --- | --- | --- | --- |
| testA-cv | 638 (48.187%) | 393 (29.683%) | 351 (26.511%) | 78 (5.891%) | 257 (19.411%) |
| testA-ekf_ctrv | 648 (48.943%) | 393 (29.683%) | 343 (25.906%) | 77 (5.816%) | 256 (19.335%) |
| testA-ukf_ctrv | 645 (48.716%) | 394 (29.758%) | 346 (26.133%) | 78 (5.891%) | 255 (19.260%) |
| testB-cv | 647 (48.867%) | 485 (36.631%) | 350 (26.435%) | 80 (6.042%) | 247 (18.656%) |
| testB-ekf_ctrv | 652 (49.245%) | 490 (37.009%) | 345 (26.057%) | 80 (6.042%) | 247 (18.656%) |
| testB-ukf_ctrv | 659 (49.773%) | 490 (37.009%) | 338 (25.529%) | 80 (6.042%) | 247 (18.656%) |

Mẫu số cho mọi tỷ lệ trong bảng trên là **1.324**. `id_match` là ID ghép gần nhất trước bằng ID ghép đầu tiên sau trong cửa sổ, không yêu cầu bám liên tục. `id_continuous` còn đòi hỏi cùng ID ở mọi frame có GT trong sự kiện. `preserved`, `switched`, `lost_after`, `no_match_before` cộng đúng 1.324; `id_continuous` là thuộc tính con của preserved, không cộng như kết cục thứ năm.

## 5. Chênh lệch và bootstrap ghép cặp

**5.000 lần, seed=0, percentile 95%, lấy mẫu video có hoàn lại**, cùng mẫu cho hai cấu hình trong từng hiệu. TrackEval dùng cả 40 video, tái gộp TP/FN/FP theo 19 alpha, AssA/LocA theo TP và Identity theo IDTP/IDFN/IDFP bằng công thức `combine_sequences` của TrackEval đang cài. Đã đối chiếu cả per-video, combined và mẫu lặp với API chính thức. Không lấy trung bình HOTA/IDF1 theo video để thay cho pooled metric.

Che khuất dùng 39 video có sự kiện đủ điều kiện; cộng số kết cục và chia tổng sự kiện được chọn ở mỗi bootstrap, **trọng số theo sự kiện, cụm lấy mẫu là video**. Các video không có sự kiện đủ điều kiện vẫn có trong chấm TrackEval và bảng tổng/loại. Script đối chiếu lại đúng KTC `id_match` của A/B hiện có. KTC chưa điều chỉnh so sánh nhiều chỉ số; các đoạn video có thể chung camera/cảnh, nên độc lập giữa video vẫn là giả định.

Mỗi ô là **chênh lệch [cận dưới; cận trên]**. Điểm tỷ lệ dùng **điểm phần trăm**, số đếm dùng đơn vị đếm. Tên `testX-ekf_ctrv - testX-cv` nghĩa EKF−CV; `testB-* - testA-*` nghĩa B−A.

| So sánh | HOTA | IDF1 | AssA |
| --- | --- | --- | --- |
| testA-ekf_ctrv minus testA-cv | +0.116 [-0.031; +0.267] | +0.244 [-0.041; +0.518] | +0.253 [-0.044; +0.557] |
| testA-ukf_ctrv minus testA-cv | +0.006 [-0.126; +0.139] | +0.067 [-0.219; +0.367] | +0.032 [-0.241; +0.311] |
| testB-ekf_ctrv minus testB-cv | +0.331 [+0.112; +0.569] | +0.531 [+0.189; +0.891] | +0.716 [+0.244; +1.238] |
| testB-ukf_ctrv minus testB-cv | +0.181 [-0.021; +0.409] | +0.309 [-0.067; +0.729] | +0.390 [-0.045; +0.869] |
| testB-cv minus testA-cv | -0.267 [-0.745; +0.185] | -0.818 [-1.409; -0.255] | -0.593 [-1.271; +0.074] |
| testB-ekf_ctrv minus testA-ekf_ctrv | -0.052 [-0.440; +0.345] | -0.532 [-1.036; -0.033] | -0.129 [-0.524; +0.321] |
| testB-ukf_ctrv minus testA-ukf_ctrv | -0.091 [-0.612; +0.400] | -0.576 [-1.275; +0.072] | -0.235 [-0.972; +0.478] |

| So sánh | DetA | LocA | MOTA |
| --- | --- | --- | --- |
| testA-ekf_ctrv minus testA-cv | -0.006 [-0.038; +0.024] | -0.000 [-0.011; +0.011] | +0.026 [-0.019; +0.074] |
| testA-ukf_ctrv minus testA-cv | -0.020 [-0.050; +0.008] | -0.009 [-0.023; +0.004] | +0.007 [-0.029; +0.043] |
| testB-ekf_ctrv minus testB-cv | +0.014 [-0.012; +0.038] | +0.000 [-0.026; +0.024] | +0.057 [-0.005; +0.116] |
| testB-ukf_ctrv minus testB-cv | +0.000 [-0.029; +0.028] | -0.003 [-0.017; +0.012] | +0.040 [-0.031; +0.114] |
| testB-cv minus testA-cv | +0.016 [-0.587; +0.571] | -0.373 [-0.440; -0.311] | -1.287 [-2.257; -0.381] |
| testB-ekf_ctrv minus testA-ekf_ctrv | +0.035 [-0.556; +0.580] | -0.373 [-0.453; -0.300] | -1.257 [-2.222; -0.362] |
| testB-ukf_ctrv minus testA-ukf_ctrv | +0.036 [-0.551; +0.581] | -0.367 [-0.435; -0.301] | -1.254 [-2.206; -0.375] |

| So sánh | IDSW | FP | FN |
| --- | --- | --- | --- |
| testA-ekf_ctrv minus testA-cv | +32 [-19; +82] | -332 [-555; -131] | +123 [-50; +279] |
| testA-ukf_ctrv minus testA-cv | +49 [-8; +105] | -225 [-369; -88] | +132 [-45; +282] |
| testB-ekf_ctrv minus testB-cv | -19 [-76; +36] | -324 [-722; +75] | -39 [-333; +244] |
| testB-ukf_ctrv minus testB-cv | +28 [-28; +84] | -172 [-562; +206] | -125 [-420; +158] |
| testB-cv minus testA-cv | -418 [-605; -251] | +27257 [+22086; +32546] | -18141 [-23661; -13323] |
| testB-ekf_ctrv minus testA-ekf_ctrv | -469 [-643; -311] | +27265 [+22172; +32454] | -18303 [-23876; -13474] |
| testB-ukf_ctrv minus testA-ukf_ctrv | -439 [-618; -276] | +27310 [+22119; +32553] | -18398 [-23972; -13551] |

| So sánh | id_match | id_continuous |
| --- | --- | --- |
| testA-ekf_ctrv minus testA-cv | +0.755 [-0.584; +2.035] | +0.000 [-0.431; +0.421] |
| testA-ukf_ctrv minus testA-cv | +0.529 [-0.519; +1.309] | +0.076 [-0.309; +0.480] |
| testB-ekf_ctrv minus testB-cv | +0.378 [-0.498; +1.316] | +0.378 [-0.079; +0.812] |
| testB-ukf_ctrv minus testB-cv | +0.906 [-0.548; +2.246] | +0.378 [-0.080; +0.872] |
| testB-cv minus testA-cv | +0.680 [-0.331; +1.897] | +6.949 [+5.202; +8.703] |
| testB-ekf_ctrv minus testA-ekf_ctrv | +0.302 [-1.447; +2.394] | +7.326 [+5.477; +9.285] |
| testB-ukf_ctrv minus testA-ukf_ctrv | +1.057 [-0.463; +2.901] | +7.251 [+5.473; +9.150] |

Đầy đủ KTC của sáu cấu hình, mọi kết cục và số lượng sự kiện: [configuration_bootstrap.csv](results/test_eval/summary/configuration_bootstrap.csv); mọi chênh lệch gồm cả các kết cục phụ: [paired_contrasts_bootstrap.csv](results/test_eval/summary/paired_contrasts_bootstrap.csv). Mô tả phương pháp, roster, hash nguồn và kiểm tra số: [metadata.json](results/test_eval/summary/metadata.json), [validation.csv](results/test_eval/summary/validation.csv).

KTC chứa 0 → **chưa đủ bằng chứng về khác biệt**, không nói “tương đương”. Chỉ kết luận theo đúng chỉ số và đối chiếu; không chọn một điểm ước lượng thuận lợi để tuyên bố tracker thắng toàn diện.

## 6. So với train từ CSV

Train có 60 video, 598.281 bbox GT, 2.766 sự kiện gộp, 1.003 đủ điều kiện trên 55 video; 1.186 loại thiếu GT trước và 577 loại thiếu GT sau. Bảng nguồn train A/B có cùng tập sự kiện. HOTA/IDF1/AssA dùng thang 0–100:

| Train | HOTA | IDF1 | AssA | IDSW | FP | FN | id_match % | id_continuous % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runA-cv | 61.043 | 77.828 | 65.407 | 2251 | 53332 | 146919 | 48.255 | 30.209 |
| runA-ekf_ctrv | 61.005 | 77.799 | 65.340 | 2289 | 53246 | 147086 | 47.458 | 30.010 |
| runA-ukf_ctrv | 60.896 | 77.585 | 65.126 | 2304 | 53338 | 147122 | 47.557 | 30.010 |
| runB-cv | 61.427 | 77.951 | 65.998 | 1982 | 70469 | 133530 | 51.147 | 37.787 |
| runB-ekf_ctrv | 61.384 | 77.955 | 65.954 | 1988 | 70739 | 133684 | 50.748 | 37.787 |
| runB-ukf_ctrv | 61.352 | 77.884 | 65.892 | 2007 | 70808 | 133624 | 50.548 | 37.787 |

Nguồn: `results/trackeval/DETRAC-all/run{A,B}-{cv,ekf_ctrv,ukf_ctrv}/combined_metrics.csv`, `results/run{A,B}/event_outcomes_DETRAC-all.csv`, `events_DETRAC-all.csv`, `bootstrap_DETRAC-all.csv`. Đây là các CSV hiện hữu, được đọc lại; không chạy hoặc sửa kết quả train.

Train A EKF−CV/UKF−CV `id_match` = −0,798/−0,698 điểm phần trăm, KTC [−1,832; +0,194]/[−1,618; +0,099]. Test A đổi dấu thành +0,755/+0,529, KTC [−0,584; +2,035]/[−0,519; +1,309]. Train B có hiệu −0,399/−0,598 với KTC [−1,486; +0,706]/[−1,763; +0,523]; đọc hiệu test B ở mục 5. Các KTC `id_match` giữa motion model chứa 0: chưa đủ bằng chứng về khác biệt trên cả hai tập.

**B−A đổi hướng giữa train và test** ở HOTA/AssA/IDF1, theo điểm ước lượng (điểm phần trăm):

| Model | ΔHOTA train / test | ΔAssA train / test | ΔIDF1 train / test |
| --- | --- | --- | --- |
| cv | +0.385 / -0.267 | +0.592 / -0.593 | +0.123 / -0.818 |
| ekf_ctrv | +0.380 / -0.052 | +0.614 / -0.129 | +0.156 / -0.532 |
| ukf_ctrv | +0.456 / -0.091 | +0.766 / -0.235 | +0.299 / -0.576 |

IDSW/FN giảm và FP tăng trên cả train/test. `id_continuous` tăng trên cả hai tập, nhưng điều đó không bảo đảm HOTA/IDF1 tổng thể tăng. Trên A, HOTA/AssA/IDF1 của EKF/UKF so CV đổi hướng mô tả giữa train/test; IDSW của EKF/UKF vẫn cao hơn CV ở cả train A và test A. Không tinh chỉnh hoặc thay baseline dựa trên các quan sát test này.

## 7. Thời gian chỉ có giá trị mô tả

| Cấu hình | Video | Ảnh | BBox đầu ra | Tổng ID theo video | Giây | Ảnh/giây mô tả |
| --- | --- | --- | --- | --- | --- | --- |
| testA-cv | 40 | 56340 | 549708 | 7369 | 1222.6 | 46.08 |
| testA-ekf_ctrv | 40 | 56340 | 549253 | 7448 | 1223.9 | 46.03 |
| testA-ukf_ctrv | 40 | 56340 | 549351 | 7428 | 1465.3 | 38.45 |
| testB-cv | 40 | 56340 | 595106 | 6965 | 1377.0 | 40.92 |
| testB-ekf_ctrv | 40 | 56340 | 594821 | 6991 | 1267.0 | 44.47 |
| testB-ukf_ctrv | 40 | 56340 | 595059 | 6984 | 1489.4 | 37.83 |

Nguồn: `data/interim/baseline_track_summary_DETRAC-test_<cấu hình>.csv`. Tổng giây là tổng `seconds` đã làm tròn theo video; phép đo gồm khởi tạo model, đọc ảnh, detector và tracker, nhưng dừng trước khi ghi CSV đầu ra. Ảnh/giây = tổng ảnh / tổng giây, không phải trung bình cột fps. Máy không được kiểm soát tải, chưa lặp/xen kẽ hoặc cố định tần số GPU. **Không coi đây là benchmark tốc độ**, không dùng để xếp hạng hiệu suất thuật toán. Tổng ID là tổng số ID riêng trong từng video, không phải ID toàn dataset và không chứng minh quan hệ nhân quả với association vòng hai.

## 8. Đường dẫn kết quả, kiểm tra và bàn giao

- Manifest gốc: `results/test_eval/manifest.json`, giữ nguyên byte.
- Tracking: sáu đường dẫn đầy đủ ở mục 3. Raw `.txt` giữ local theo `.gitignore` hiện tại, không đưa dataset/weights/raw tracking vào commit.
- Summary tracking: sáu `data/interim/baseline_track_summary_DETRAC-test_test{A,B}-{cv,ekf_ctrv,ukf_ctrv}.csv`.
- TrackEval: sáu `results/trackeval/DETRAC-test/test{A,B}-{cv,ekf_ctrv,ukf_ctrv}/`, mỗi thư mục có `combined_metrics.csv` và `per_video_metrics.csv`.
- Che khuất: `results/test_eval/A/` và `results/test_eval/B/`, mỗi thư mục có `events_DETRAC-test.csv`, `event_outcomes_DETRAC-test.csv`, `paired_DETRAC-test.csv`, `bootstrap_DETRAC-test.csv`.
- Tổng hợp, KTC và nguồn detailed: `results/test_eval/summary/`; script mới **chỉ tổng hợp CSV**: `src/summarize_test_evaluation.py`.
- Log/provenance: `results/test_eval/provenance/`; gồm log gốc A/B đã lọc, lệnh nguyên văn, log đầy đủ chấm B, môi trường, audit ảnh/MOT, tests, snapshot hash và kết quả kiểm tra bảo toàn.

Năm file test đã chạy trực tiếp bằng đúng env, kết quả **77/77 PASS**: `test_ekf_ctrv.py` 14/14, `test_ukf_ctrv.py` 12/12, `test_init_regression.py` 17/17, `test_occlusion_eval.py` 14/14, `test_tracker_lifecycle.py` 20/20. Xem [tests.log](results/test_eval/provenance/tests.log). Không thêm test chỉ để đếm; tổng hợp mới có assertion đối chiếu kết quả gốc và công thức TrackEval.

```powershell
foreach ($test in 'test_ekf_ctrv','test_ukf_ctrv','test_init_regression','test_occlusion_eval','test_tracker_lifecycle') {
  & $py "src/$test.py"
  if ($LASTEXITCODE -ne 0) { throw 'Test failed' }
}
& $py results/test_eval/provenance/verify_preserved.py
git diff --check
git diff --cached --check
git commit -m 'Chay danh gia tap test 40 video'
```

Kiểm tra SHA-256 trước/sau cho **2.258 file hiện hữu** (results, GT/raw tracking, interim, source/config và weights) để chứng minh không ghi đè train A/B, ba test cũ `yolov8n-test-bytetrack`, `yolov8n-test-ekf-ctrv`, `yolov8n-test-ukf-ctrv`, hoặc test A/B tracking hiện tại. Xem [preservation_check.json](results/test_eval/provenance/preservation_check.json). Nội dung `FIX_REPORT.md` có sẵn lúc tiếp quản được giữ nguyên nguyên văn; chỉ thêm ghi chú hiện hành và mục test mới.

File đã thay đổi: `FIX_REPORT.md`. File mới: `TEST_EVALUATION_REPORT.md`, `src/summarize_test_evaluation.py`, sáu summary test, sáu thư mục TrackEval test mới A/B và `results/test_eval/` với manifest, A/B, summary, provenance. Danh sách cụ thể của commit kết quả xem `git show --stat --oneline b5bd086`; không có file thuật toán/config/data gốc bị sửa.

## 9. Giới hạn và số liệu dùng cho slide

1. Test đã được sử dụng trước đó; không gọi là kiểm định độc lập hoàn toàn. Cấu hình hai lượt hiện tại khóa trước chạy, không tối ưu lại theo test.
2. Chỉ một detector YOLOv8n pretrained COCO, một bộ dữ liệu và một lần tracking mỗi cấu hình. Chưa chứng minh tổng quát cho detector fine-tune, dataset/camera độc lập hoặc mọi mức che khuất.
3. KTC video không đo biến thiên giữa nhiều lần chạy hoặc nhiều dataset; video có thể chia sẻ cảnh, và chưa hiệu chỉnh nhiều so sánh. Các KTC chạm/chứa 0 không chứng minh tương đương.
4. 850 sự kiện không đủ điều kiện bị loại có lý do GT; tỷ lệ che khuất chỉ áp dụng cho 1.324 sự kiện đủ điều kiện, không đại diện mọi xe/lần che. Full là thuộc tính trong sự kiện gộp.
5. Log tracking gốc lọc stderr; không thể chứng nhận không có mọi cảnh báo lịch sử. Audit hiện tại không phát hiện output lỗi; không có bằng chứng lỗi thuật toán cần sửa hoặc buộc chạy lại trong lần này.
6. FP tăng có thể ảnh hưởng đếm xe/mật độ; chưa đánh giá ứng dụng hạ nguồn. Không dùng tổng ID để suy luận nhân quả; chưa có log association vòng hai xuyên suốt 40 video.
7. Thời gian chỉ mô tả; raw tracking và ảnh/weights không nằm trong commit. Tái lập tracking cần dataset và env tương ứng; tái tổng hợp có các CSV detailed đã lưu.

**Có thể đưa lên slide:** quy mô 40 video/56.340 frame/675.774 bbox GT; 240/240 file hợp lệ; 2.174 sự kiện sau gộp (loại đếm lặp 484 full), 1.324 đủ điều kiện; bảng sáu cấu hình ở mục 4; hiệu kèm KTC ở mục 5; biểu hiện B giảm IDSW nhưng tăng FP và không cải thiện đồng loạt HOTA/IDF1 test; hướng train/test khác nhau. Với `id_match`, ghi “chưa đủ bằng chứng EKF/UKF khác CV” và hiển thị KTC, không ghi “EKF/UKF thắng” hoặc “tương đương”. Nếu đưa tốc độ, ghi rõ “mô tả một lần chạy, tải máy không kiểm soát”.

**Việc còn lại ngoài phạm vi lần bàn giao:** commit kết quả đã được push; merge vào `main` chỉ thực hiện khi có yêu cầu riêng. Nếu cần kết luận tổng quát hơn thì thiết kế đánh giá trên dữ liệu độc lập và benchmark thời gian có kiểm soát, đo ảnh hưởng FP, log association đầy đủ. Detector fine-tune và ngưỡng conf trung gian không được thử trong lượt này; mọi nghiên cứu bổ sung phải chốt trước, không tinh chỉnh trên test đã xem.
