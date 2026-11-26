# R113 — xe rẽ, che khuất và RBPF trên detection Nano

HELMET giữ vai trò mốc xe máy. Các số trong mục kịch bản theo thứ tự KF/EKF/UKF và dùng cùng detector trong từng phép so. Quy tắc nhóm và chọn mẫu ở [giao thức](00_protocol.md).

## Che khuất

- **UA-DETRAC test40**. Trong 7.486 hộp visibility ≤0,5, IDSW **53/50/48** và FN **4.941/4.940/4.939**. EKF giảm 5,7%, UKF giảm 9,4% IDSW so KF. Tính cả khung có che trong một giây gần nhất, IDSW **86/83/77** và FN **5.793/5.792/5.791** trên 10.097 hộp. Riêng các khung đã trở lại visibility >0,5 trong nhóm đó, IDSW **33/33/29**. Theo clip ở đúng khung đang che, EKF giảm ở 5, bằng ở 33, tăng ở 2. UKF giảm ở 5, bằng ở 32, tăng ở 3. Tính toàn 40 đoạn, IDSW **245/250/248**. Nhóm vừa rẽ vừa che chỉ có 34 hộp và 32 FN mỗi bộ lọc. Đây là hậu kiểm trên 40 đoạn đã dùng R110, chủ yếu xe bốn bánh. [Số theo clip và nhóm](ua_occlusion.json).
- **SVMOT**. Độ phủ ứng viên IoU ≥0,5 ở ngưỡng c25/c10/c05 là **49,03/67,42/73,60%** ở test9 và **46,40/69,38/77,68%** ở train21. Đây là cận trên trước ghép một-một. Điểm tracking đã có của R64 cho thấy KF c25→c10 giảm FN **14.665→12.924**, tăng FP **21.964→29.836** và giảm IDSW toàn tập **435→311**. Riêng ở khung thấp visibility tại c10, IDSW **105/107/106**, FN **5.685/5.635/5.597**. C05 chỉ đo ứng viên detection. [Độ phủ và điểm test9 theo ngưỡng](svmot_thresholds.json).

## Chuyển động phi tuyến trên ảnh

- **RoundaboutHD R111** vẫn là phép thử rõ nhất về chuyển động đổi hướng từ camera cố định. Nhóm rẽ ≥25° có **13.509 hộp** và IDSW **34/28/28**, FN **5.275/5.264/5.263**. Tính cả cửa sổ tương ứng, IDSW **716/726/725**. Bốn camera ở cùng một vòng xuyến. [Số đã chấm R111](../review111_roundabout/evaluation/scores.json).
- **BDD100K R110, 40 đoạn**. Nhóm rẽ ≥25° có 2.205 hộp, IDSW **64/61/59**, FN **1.201/1.206/1.207**. Riêng 1.896 hộp ô tô rẽ, IDSW **58/56/54**, FN **973/977/978**. Phép ghép vehicle gốc được phân nhóm theo lớp GT. Có 147 hộp xe máy nhưng không có đoạn xe máy đạt điều kiện rẽ. Cờ che khuất native vắng trong nhãn COCO-video cục bộ. Góc ảnh chịu ảnh hưởng camera xe. [Phân nhóm BDD](bdd_turns.json).
- **METEOR mới**. Tải và kiểm CRC/SHA hai video ở `F:/LABOTARY_data/goc_tai_ve/METEOR-R113`, 3.600 khung, 30 fps. Chọn bằng số đoạn rẽ từ nhãn trước điểm, có 22 đoạn ô tô và 29 đoạn xe máy/scooter đạt tiêu chí. Với 425 hộp ô tô rẽ, IDSW **0/0/0**, FN **294/294/294**. Với 670 hộp xe máy rẽ, IDSW **2/1/1**, FN **658/658/658**. Detector có ứng viên đúng ở 18/670 hộp xe máy rẽ. Toàn hai clip, ô tô IDSW **72/84/81** và HOTA **49,25/49,78/49,75%**, xe máy IDSW **62/59/58** và HOTA **12,00/12,03/11,42%**. Hai clip còn ít sự kiện ghép được để khẳng định ưu thế. Cả 89.017 hộp ô tô/xe hai bánh trong tám gói nhãn hợp lệ audit đều ghi `occluded=0`. Một gói 150 XML bị loại trước khi chọn clip vì không xác nhận được nhịp ảnh. [Nguồn METEOR](https://github.com/GAMMA-UMD/METEOR) · [Các clip đã xét](meteor_selection.json) · [Kết quả theo lớp và nhóm](meteor_scores/scores.json).

## Nguồn phù hợp cho kiểm tra tiếp theo

- **V2X-Seq-SPD** có triển vọng nhất cho một tập kết hợp cả hai kịch bản. Có camera cố định/gắn xe, nhãn 2D với track ID và `occluded_state`, lớp Car và Motorcyclist. Đã mở được danh mục gói gốc và thử gói mẫu cũ, mẫu thay thế chính thức, metadata train/val và gói ảnh camera cố định. Cả bốn đường trả HTML quota, chưa tải được nhãn/ảnh để kiểm có bao nhiêu sự kiện. [Định nghĩa nhãn](https://github.com/AIR-THU/DAIR-V2X-Seq/blob/main/dataset/v2x-seq-spd/README.md) · [Danh mục gốc](https://drive.google.com/drive/folders/1zeMHE_k8mzxTxYHzXQpzDtpbzS9Z5dPL) · [Kết quả dò tải](v2x_access.json).
- **TUMTraf V2X** có camera đường/gắn xe, rẽ, quay đầu và được tác giả nêu rõ các tình huống xe này che xe kia. Quyền tải vẫn qua đăng ký, điểm MOT 2D cần hiệu chuẩn và kiểm nhãn theo từng góc camera. [Cổng tác giả và mô tả cảnh](https://innovation-mobility.com/en/project-providentia/a9-dataset/).
- **FE-DETRAC** có MOT camera fisheye cố định, nhãn Bike cần kiểm lại có tách xe máy hay không, thử URL archive bị timeout. **D2-City** có ID, lớp motorcycle/car và cờ che, trang tải tác giả trả 404 trong lượt kiểm. **VN2V-Weather** có xe máy/ô tô nhưng là hành lang giao thông một chiều và gói ảnh/nhãn yêu cầu đăng nhập. **RCooper** có hình ảnh giao lộ với nhãn dựng từ LiDAR, cần xử lý độ khả kiến theo camera trước khi dùng trong pipeline 2D. [FE-DETRAC](https://www.scidm.tw/en/dataset/fe-detrac) · [D2-City](https://arxiv.org/abs/1904.01975) · [VN2V-Weather](https://huggingface.co/datasets/vehicle-research/multi-weather_traffic_data) · [RCooper](https://github.com/AIR-THU/DAIR-RCooper) · [Log truy cập](source_probe.log).

## RBPF với detection từ Jetson Nano

Phát lại trên PC đủ **240 lượt**, 80 đoạn × ba seed, 36.000 khung. Dùng cache R110 và cùng các tham số phát hiện/ghép với đối chứng. Số RBPF là trung bình ba seed sau khi gộp đúng theo từng tập.

- **BDD40**, PF → RBPF IDSW **1.569,67→1.450,33** (−7,60%), HOTA **42,81→43,60%**, FN **29.147,67→28.610**. KF **1.534 IDSW**, HOTA **43,72%**.
- **UA40**, PF → RBPF IDSW **276,33→246,33** (−10,86%), HOTA **65,94→66,69%**, FN **18.193→17.603,33**. KF **245 IDSW**, HOTA **66,57%**.

RBPF cải thiện cả IDSW và HOTA so PF cũ ở cả hai tập. Số FPS RBPF trên thiết bị vẫn chưa được đo. Đối chiếu cache SHA từng đoạn, xác minh 150 khung mỗi lượt. IDSW/FN/FP của 40 đoạn mỗi tập cộng khớp từng seed. Các nhánh phân nhóm kiểm parity với TrackEval trước khi tính kết quả. Hậu kiểm METEOR có một lỗi giả định 1.800 XML cho mọi clip được sửa bằng cách ghi rõ gói sai nhịp. Hậu kiểm BDD ban đầu đọc sai tên trường IDSW/FN của JSON R110, đã sửa và đối chiếu đủ 120 lượt. [Điểm đầy đủ](nano80/scores.json) · [Hậu kiểm](verification.log).
