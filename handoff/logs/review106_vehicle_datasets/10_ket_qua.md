# R106 — dữ liệu xe đường bộ và hậu kiểm kịch bản GT

Ngày 12/11/2026. Người triển khai đã tự kiểm; **chưa có review Codex độc lập cho R106**. Mục 0 PLAN được cập nhật trước báo cáo này. Người dùng giao trực tiếp bước R106; phiên này không có tài liệu review độc lập R105 để tự xác nhận mốc đó, và không thay kết luận R105.

Kết quả hiện có **không chứng minh EKF/UKF tốt hơn KF nói chung trên quỹ đạo phi tuyến**. Đối chứng cố định YOLOv8n COCO c25 cho nhóm đổi hướng RoundaboutHD có IDSW 22/22/22; SVMOT 2/2/3 (KF/EKF/UKF). HELMET tăng nhẹ HOTA toàn mẫu nhưng chỉ có 43 khung-xe đạt cổng đổi hướng. Gói nhãn GRAM-RTM mới tải được và kiểm mẫu GT thành công, chưa có điểm tracking GRAM.

## Phạm vi và cách chọn

Không mở tracker, detector, GPU, train hoặc SSH. Tái sử dụng đầu ra đã có của toàn bộ năm mẫu khóa: SVMOT test9, RoundaboutHD prefix3000 bốn camera, HELMET test36, Sherbrooke cars v2 và hai RideSafe pilot R100. Tổng **52 clip/camera, 25.451 khung chấm, 162.440 hộp GT, 1.093 ID cộng theo clip/camera**; không phải 1.093 xe duy nhất giữa các chuỗi/camera. Không cộng các chỉ số chất lượng giữa dataset.

`00_protocol.md` được ghi trước runner mở điểm R106. Tuy nhiên mục 0 và báo cáo lịch sử đã chứa kết quả, nên đây là **hậu kiểm mô tả trên dữ liệu đã quan sát**, không phải thí nghiệm xác nhận độc lập. Dùng COCO c25 làm đối chứng chung có sẵn ở cả năm mẫu, không chọn detector tốt nhất riêng từng tập. Ba bộ lọc của mỗi tập dùng cùng cache, ngưỡng và thiết lập association đã khóa; lịch chạy, FPS và bộ đệm theo từng tập được giữ lại trong metadata. Không lấy test để tune.

Gán nhóm trước khi runner đọc MOT/raw fields, bằng nguyên hàm R99 `labels_for_camera`: điểm giữa đáy hộp GT, hỗ trợ 2 giây, trung vị quanh ba mốc; mỗi chân dịch chuyển ≥max(8 px, 0,25 chiều cao hộp); đổi hướng cùng dấu ≥25° đủ ceil(8×fps/15) khung liên tiếp. Nhóm ≥35° là độ nhạy đã có; gần thẳng ≤8° là đối chứng. Không nối qua khoảng thiếu GT. Đây là **góc ảnh**, không phải nhãn rẽ vật lý; camera xe còn chứa chuyển động ego. Các nhóm rẽ và thẳng không phủ mọi GT, nhóm 35° nằm trong nhóm 25°.

Phát lại association CLEAR trên nguyên chuỗi rồi đếm IDSW/FN tại khung-xe thuộc nhóm; giữ lịch sử ID qua biên episode. HOTA/IDF1 chỉ báo toàn mẫu, không lấy điểm toàn tập làm điểm nhóm. Không tạo GT đã lọc để chấm lại episode. Hai demo slide từng được chọn theo điểm vẫn chỉ là minh họa và không tham gia chọn cohort R106.

## Nguồn chính thức, góc nhìn và khả năng sử dụng

URL/thời điểm/HTTP/kích thước/SHA256 trong `sources_manifest.json`; nội dung chụp tại `source_*.txt`. Đây là tài liệu nguồn, không phải bằng chứng đã tải toàn bộ dataset.

- **HELMET:** [kho tác giả LinHanhe](https://github.com/LinHanhe/Helmet_use_detection) dẫn [OSF](https://osf.io/4pwj8/). Camera ven đường, ID xe máy; mẫu test36 đã chuẩn bị ở R66, 12 nhóm tên cảnh. Nhãn thường bao cả người ngồi, khác box detector motorcycle. HTTP 200 của OSF trả trang ứng dụng, không xác minh lại toàn bộ archive hay giấy phép qua HTML đó. Không tải lại dữ liệu.
- **SVMOT:** [bản phát hành tác giả trên Zenodo](https://zenodo.org/records/19468203), kèm README chính thức đã tải, mô tả ID, visibility, quan hệ che khuất và xe dừng. Camera giao thông cố định; chín chuỗi thuộc hai scene test. Dùng full-image của dự án, không tô ROI như một số giao thức khác; dự đoán xe xa ngoài GT vẫn chịu FP. Không mở archive 18,3 GB.
- **RoundaboutHD:** [repo tác giả](https://github.com/siri-rouser/RoundaboutHD) dẫn [bản dữ liệu](https://huggingface.co/datasets/yl4300/RoundaboutHD), camera cố định quanh vòng xuyến, SCT ID, chuyển động cong và che khuất do hạ tầng. Dùng bốn prefix 3.000 frame ở 15 fps đã khóa. Giữ GT R64 đã loại đúng 16 dòng trùng cam4; xe đỗ ngoài SCT GT chịu FP. Bốn camera cùng một nút giao, không phải bốn cảnh độc lập. Không tải lại archive.
- **Sherbrooke:** [Urban Tracker của tác giả](https://www.jpjodoin.com/urbantracker/dataset.html), camera thấp tại giao lộ; R66 đã chuẩn bị bản cars/ROI v2 1.001 frame. Hai URL `www` và không `www` hiện đều trả **404**, đã lưu body; không xem kết quả tìm kiếm cũ là tải nguồn thành công. GT/MOT và source lock cục bộ vẫn đủ để hậu kiểm mẫu cũ, không tuyên bố đã tải lại từ tác giả hôm nay.
- **RideSafe-400:** [DashCop của tác giả](https://github.com/dash-cop/DashCop) dẫn [bản phát hành](https://huggingface.co/datasets/DeepBug/RideSafe-400), camera gắn xe, nhãn CVAT motorcycle track. Dùng hai pilot R100 có source manifest/XML/MOT sẵn; CVAT track ID sau audit, không thay bằng thuộc tính `motor_track_id`. Nhãn cách 5 khung nguồn 25 fps, chấm 5 fps. Đây là pilot đã quan sát, không phải holdout mới; không gộp với hai clip test R102 hay detector nội suy R101. Card ghi CC BY-NC-SA; không đăng nhập hoặc liên hệ tác giả.

## Chất lượng toàn mẫu đã kiểm trực tiếp

Các dãy dưới đây luôn theo **KF/CV – EKF – UKF**. HOTA và IDF1 theo phần trăm. `combined_verified.csv` giữ số đầy đủ và FP; `verification.json` chứa từng kiểm tra và hash.

- **SVMOT:** 9 clip, 8.250 frame, 168 ID theo chuỗi, 70.220 hộp. HOTA **48,660 / 49,649 / 49,209**; IDF1 **50,301 / 51,908 / 51,181**; IDSW **435 / 407 / 418**; FN **14.665 / 14.668 / 14.672**.
- **RoundaboutHD:** 4 camera, 12.000 frame, 352 ID theo camera, 75.671 hộp. HOTA **38,444 / 38,073 / 38,055**; IDF1 **51,593 / 51,002 / 51,025**; IDSW **388 / 402 / 409**; FN **39.480 / 39.472 / 39.494**. Cả hai bộ lọc phi tuyến giảm HOTA và tăng IDSW so KF trên đối chứng này.
- **HELMET:** 36 clip, 3.600 frame, 287 ID theo clip, 9.371 hộp xe máy. HOTA **30,105 / 30,339 / 30,359**; IDF1 **38,726 / 38,228 / 38,171**; IDSW **109 / 88 / 93**; FN **5.623 / 5.643 / 5.642**. HOTA tăng nhưng IDF1 giảm và FN tăng nhẹ; không bỏ qua đánh đổi.
- **Sherbrooke:** 1 clip, 1.001 frame, 15 ID cars trong ROI, 2.012 hộp. HOTA **32,394 / 32,364 / 32,364**; IDF1 **38,796 / 38,802 / 38,822**; IDSW **4 / 4 / 4**; FN **804 / 804 / 804**. Khác biệt rất nhỏ, một camera.
- **RideSafe pilot R100:** 2 clip, 600 khung nhãn, 271 ID theo clip, 5.166 hộp xe máy. HOTA **16,401 / 16,213 / 16,283**; IDF1 **17,876 / 17,632 / 17,724**; IDSW **20 / 26 / 23**; FN **4.531 / 4.524 / 4.525**. Đây là **COMBINED mới ghép bằng TrackEval**, không phải trung bình HOTA hai clip trong báo cáo cũ. Sáu điểm từng clip chấm lại từ MOT đều khớp điểm R100 ≤1e-9.

## Đổi hướng, đoạn thẳng và che khuất

`gt_cohort.json` chứa toàn bộ episode và hash GT trước đọc điểm. `scenario_per_clip.csv` và `scenario_summary.csv` có số ID/clip/khung-xe từng nhóm; ID có thể xuất hiện ở cả nhóm rẽ và thẳng, không cộng số ID nhóm như các tập rời nhau.

- **SVMOT:** nhóm ≥25° có 12 episode, 11 ID/6 clip, **240 khung-xe**, IDSW **2/2/3**, FN **64/65/65**. Nhóm gần thẳng có 108 episode, 89 ID/9 clip, **5.799 khung-xe**, IDSW **20/19/21**, FN **371/379/384**. Mức cải thiện IDSW toàn mẫu không tự trở thành cải thiện nhóm rẽ.
- **RoundaboutHD:** nhóm ≥25° có 318 episode, 226 ID theo camera, **6.601 khung-xe**, IDSW **22/22/22**, FN **2.633/2.637/2.637**. Nhóm thẳng có 834 episode, 333 ID theo camera, **36.002 khung-xe**, IDSW **233/234/236**, FN **17.265/17.223/17.240**. Các số 28/32/29 của mục 0 lịch sử dùng **mixed5 c25**, không mâu thuẫn với đối chứng COCO c25 ở đây và không được trộn hai detector.
- **HELMET:** nhóm ≥25° có 5 episode, 5 ID/3 clip, **43 khung-xe**, IDSW **0/0/0**, FN **16/17/17**. Nhóm thẳng có 116 episode, 111 ID/34 clip, **2.443 khung-xe**, IDSW **20/25/26**, FN **1.446/1.461/1.458**. Số nhóm rất nhỏ; không quy mức tăng HOTA toàn mẫu cho chuyển động rẽ. Không đồng nhất cổng 25° R99 này với proxy 30° khác ở R91.
- **Sherbrooke:** không episode ≥25° qua cổng; nhóm thẳng 18 episode/15 ID, **761 khung-xe**, IDSW **1/1/1**, FN **159/159/159**. Không nới cổng để tìm đoạn có lợi.
- **RideSafe:** nhóm đổi hướng ảnh ≥25° có 34 episode/26 ID, **123 khung-xe**, IDSW **3/3/2**, FN **97/97/97**. Nhóm thẳng 54 episode/46 ID, **342 khung-xe**, IDSW **1/1/1**, FN **303/298/301**. Bỏ sót rất cao và ego-motion/nhịp nhãn 5 fps hạn chế diễn giải.

**Visibility SVMOT:** nhóm ≤0,5 có **12.051 hộp (17,16% GT), 94 ID/9 clip**, IDSW **125/115/125**, FN **6.515/6.517/6.497** (54,06/54,08/53,91%). Nhóm >0,5 có **58.169 hộp**, IDSW **310/292/293**, FN **8.150/8.151/8.175** (14,01/14,01/14,05%). Đây là phân tầng theo visibility GT với cổng R106, không phải lớp mức độ chính thức hay bằng chứng nguyên nhân IDSW; không phân tích stopped/pairwise occlusion JSON. Nhóm visibility có thể giao với nhóm chuyển động.

HELMET/RoundaboutHD/Sherbrooke không có visibility thật đủ chứng cứ trong GT chuyển đổi: **N/A**, không diễn giải cột dummy=1 như hoàn toàn không bị che. Hai XML RideSafe có 0/5.166 hộp hiển thị gắn `occluded=1`; nhãn này không đủ để ước lượng che khuất thực. Không dùng hộp chồng lấn hoặc FN làm nhãn che khuất thay thế.

## Ứng viên mới và mẫu công khai

**GRAM-RTM:** [trang tác giả GRAM/ĐH Alcalá](https://gram.web.uah.es/data/datasets/rtm/index.html) cung cấp ID xe và quyền sử dụng nghiên cứu; yêu cầu ảnh chỉ để test, không train/tune. Urban1 là camera giám sát giao lộ, đúng phạm vi. Gói devkit/nhãn công khai tải được **24.732.734 byte**, SHA256/manifest từng thành viên trong `gram_manifest.json`. HEAD urllib ban đầu 403, sau đó GET công khai với user-agent tác vụ trả 200; không đăng nhập, không thực thi code trong kit. Không đọc các kết quả baseline kèm gói để chọn mẫu.

Mẫu theo tên **Urban1 frame 0–249**, không chọn theo điểm. Đã xem một XML frame 0 để hiểu schema trước runner; không xem điểm tracking hoặc so prefix khác. Runner cố định danh sách 250 tên trước khi đọc đối tượng. Kiểm đủ 250 XML và CRC của **chỉ các thành viên được đọc**: 600×360, **752 hộp car, 8 ID**, không trùng `(frame, ID)`, hộp hợp lệ. Có 250 hộp `trajectory=park`; giữ chúng trong audit. Cờ occluded nguồn có **681 giá trị 0 và 71 giá trị 1**, chưa chuẩn hóa thành tỷ lệ visibility. Cùng quy tắc góc ảnh có **0 đoạn rẽ ≥25°**, nhóm thẳng 5 episode/158 khung-xe. Giữ kết quả âm, không tìm prefix khác. `gram_sample_gt_audit.json` chứa hash từng XML và episode; `gram_Urban1_prefix250_gt.csv` chỉ là bản dẫn xuất cho audit hình học. **Chưa chấm HOTA/IDF1/IDSW/FN GRAM:** chưa có ảnh và cache YOLO tương ứng; không lấy điểm devkit tác giả thay điểm hệ thống. Không tải ảnh SharePoint, không mở thí nghiệm mới.

**KITTI tracking:** [trang benchmark chính thức](https://www.cvlibs.net/datasets/kitti/eval_tracking.php) có tracklet, trạng thái che khuất và box 2D; [trang setup](https://www.cvlibs.net/datasets/kitti/setup.php) mô tả nền tảng gắn xe. Link nhãn/ảnh hiện trỏ `user_login.php` trong HTML đã lưu. **Chưa tải mẫu/chấm**, không đăng nhập hoặc thử đường tải vòng qua cổng. Không dùng bảng xếp hạng của KITTI làm kết quả hệ thống này.

Tổng byte mới tải/lưu cho tài liệu và gói nhãn là **25.121.837 (<25 MiB = 26.214.400)**, được kiểm tự động. Không tải lại dữ liệu cũ, không mở TUMTraf đang tạm dừng hoặc chạy lại DAIR đã có blocker lịch sử.

## Tự kiểm, provenance và giới hạn

Lệnh dùng `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`, `PYTHONDONTWRITEBYTECODE=1`. `tier1_checks.json` ghi command, exit, bằng chứng và cách thu stdout:

1. `audit_r106.py gt` — **exit 0**, cohort GT-only 52 clip; GT/seqinfo/code/giao thức và ma trận nhóm có hash.
2. `audit_r106.py audit` — **exit 0**, **156/156** ID mapping/present/IDSW/FN nguyên chuỗi khớp; **12** COMBINED từ raw fields khớp CSV ≤1e-9, **6** lượt RideSafe chấm từ MOT khớp điểm cũ; ghép **15** kết quả toàn mẫu.
3. `fetch_sources.py` — **exit 0**, 12 yêu cầu nguồn gồm 10 HTTP 200 và 2 HTTP 404; không đánh đồng HTTP 200 với tải dữ liệu thành công. `check_gram.py` — **exit 0**, trang và gói nhãn HTTP 200.
4. `audit_gram_sample.py` — **exit 0**, kiểm 250 XML được chọn, CRC/box/ID và nhóm GT; không phải benchmark tracking.
5. `run_tier1.py` — **exit 0**, gọi `verify_evidence.py`, stdout/stderr nguyên văn ở `tier1_integrity.log`. Kiểm lại hash đầu vào/nguồn, checkpoint, **12 lineage run/cache**, các setting association chung, certificate/MOT HELMET; **12 phép ghép HOTA/Identity trực tiếp bằng TrackEval**, fixture đổi ID sau gap, **15 đối chiếu ALL/COMBINED**, hash mẫu GRAM và trần tải. Raw lịch sử thiếu LocA; phép ghép trực tiếp điền 0 riêng LocA không được báo cáo, HOTA/AssA/DetA không phụ thuộc LocA. Không tuyên bố đã kiểm LocA.

Verifier lần đầu **exit 1** vì so toàn bộ hash certificate HELMET giữa các bộ lọc (certificate chứa config nên đương nhiên khác). Đã sửa so cache-marker từng clip, đồng thời kiểm hash certificate và MOT riêng từng bộ lọc; lần cuối exit 0. Lỗi này và HEAD GRAM 403 được giữ trong `tier1_checks.json`, không xóa dấu vết hoặc coi lỗi thành PASS. Hai lệnh GT/audit ban đầu có exit quan sát trực tiếp trong phiên; stdout ban đầu không redirect, điều này được ghi rõ trong hồ sơ thay vì dựng lại log nguyên văn.

Không hash lại ảnh/NPZ nguồn hay chạy lại inference/tracker; lineage dựa trên metadata/certificate đã lưu và hash checkpoint hiện có. RideSafe R100 không có cache NPZ bất biến lưu theo từng bộ lọc; code pilot dùng chung detections trong một lượt và metadata ghi hash detector, nhưng R106 không phục dựng được byte detection cache lịch sử. Mọi kiểm mới ở đây là CPU/offline từ GT/MOT đã chỉ định. Không test GPU, FPS hoặc chất lượng detector mới. Không chạy toàn bộ CRC GRAM, chỉ 250 XML mẫu. Đã chờ kết thúc các tiến trình tự mở; không để job nền.

`verification.json` và `integrity_verified.json` phục vụ kiểm độc lập, không thay review của phiên Codex khác. R106 đủ hồ sơ nguồn, phép chấm trên mẫu có chứng cứ và giới hạn ứng viên chưa chấm để bàn giao reviewer. Không chọn bộ lọc/config mới từ kết quả này; chưa mở R107.
