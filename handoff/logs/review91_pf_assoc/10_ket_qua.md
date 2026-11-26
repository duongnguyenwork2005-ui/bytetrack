# R91 — PF IDSW, xe máy và chọn bộ dữ liệu theo mục tiêu

Ngày 06/11/2026. Đây là thăm dò trên tập đã quan sát; không đổi cấu hình luận văn, checkpoint detector khóa hoặc launcher Nano. Chỉ đánh giá **phương tiện**. Giao thức/cổng và căn cứ dataset ở `00_protocol.md`.

## 1. PF: lỗi tập trung ở track Lost, nhưng ứng viên chưa qua cổng

Chạy lại trên cache YOLOv8n COCO R90 c10 chung cho CV và PF K1/K2 motion100 seed0, UAVDT M1306 (che khuất nguồn 60,9% box) và M0204 (proxy đổi hướng 10 frame 5,9%). Tệp MOT của hai run có/không assoc log trùng SHA với R90. `src/r91_pf_assoc/diagnose.py` ghép GT với output/detection tại IoU≥0,5 và chỉ xét GT đã từng gắn track ID trong ≤8 frame. Đây là **proxy continuity có điều kiện**, không phải sự kiện IDSW của TrackEval và mẫu CV/PF không đồng nhất hoàn toàn.

M1306: PF K1/K2 có 73/803 trường hợp track cũ đang Lost với IoU giữa hộp dự đoán và detection đúng GT dưới 0,2; CV có 8/655. Trong các trường hợp GT gắn output ID khác, PF có 56 track Lost và 57 track Tracked, CV 18 và 77 tương ứng. IoU dự đoán trung vị toàn mẫu đủ điều kiện PF 0,862, CV 0,890; trong nhóm đổi ID PF 0,520, CV 0,700. Những con số này hỗ trợ giả thuyết **drift khi Lost**, chưa chứng minh nguyên nhân nhân quả cho mọi IDSW. Dữ liệu: `diagnostic.json` và `M1306_*_continuity.csv`.

Ứng viên opt-in `pf_assoc_expected_iou_lost` ở `src/r91_pf_assoc/cost.py` tính IoU kỳ vọng theo trọng số hạt cho track Lost rồi trộn 50/50 với IoU hộp trung bình; track Tracked và vòng ghép thứ hai giữ nguyên. Đây là **PF + thay đổi association**, không phải ablation thuần bộ lọc. Hàm kỳ vọng đạt fixture hai hạt (trọng số 0,75/0,25 → IoU kỳ vọng 0,75); cấu hình mặc định sau thêm opt-in cho M0204 vẫn tạo MOT trùng byte R90. YAML thử ở `configs/phase2/trackers/r91/pf_expected_iou_lost.yaml`.

- M1306: PF nền seed0 HOTA **35,48%**, IDSW **193**, FP **1.447** → ứng viên **37,13%**, **187**, **1.420**.
- M0204: PF nền **43,90%**, **22**, **261** → ứng viên **42,91%**, **24**, **263**.
- Gộp hai scene: HOTA 36,75→37,98%, IDSW 215→211. Tuy nhiên cổng trước đã yêu cầu IDSW giảm trên **cả hai** scene và HOTA không giảm quá 0,5 điểm mỗi scene. M0204 vi phạm cả hai điều kiện; **loại ứng viên, không mở full hay UAVDT test**. CV đối chứng gộp HOTA 40,50%, IDSW 152; PF vẫn chưa ngang CV.

Các công trình về [Rao–Blackwellized data association PF](https://oa.upm.es/7270/1/2010_Visual_tracking_of_multiple_interacting_objects_through_Rao-Blackwellized_data_association_particle_filtering.pdf) và [PF đa mục tiêu lấy mẫu liên kết](https://ieeexplore.ieee.org/document/1039400) xử lý bất định **quyết định ghép**, không chỉ thay IoU của một hộp. Chúng gợi ý một nhánh tiếp theo có nhiều giả thuyết liên kết trong thời gian che khuất, nhưng cần thiết kế tracker mới, kiểm tốc độ và tập xe chưa dùng để chọn tham số. K1 proposal, roughening, giữ tâm sau resample, noise đối dấu và Rao–Blackwell hóa hộp đã được dự án thử ở R70–R87; không lặp lại dò hệ số scalar trên BDD/UA đã quan sát nhiều lần.

## 2. Detection xe máy: hai phép thử không đạt

Mixed5 train khóa có **2.235/476.062** box motorcycle (0,47%); BDD dev có 226. Trên full mixed5 dev 7.190 ảnh, checkpoint YOLO11n mixed5 gốc có AP50 car/truck/bus/motorcycle/bicycle lần lượt **0,8511 / 0,4824 / 0,5522 / 0,3785 / 0,2183**, mAP50 **0,4965**.

Pilot huấn luyện riêng chọn 5.000 ảnh từ **train** manifest, 4.742 BDD + 258 UA, đủ năm lớp; toàn bộ 1.822 ảnh train có xe máy được chọn. Hai epoch SGD lr0 0,001, 640 px, batch8, seed9106, khởi tạo từ YOLO11n mixed5 gốc. Checkpoint mới SHA `3144673bd1fff4d85f45cdbc3bf0705f4c90697448e6bd15c4cd90d4daaadd6b`. So cùng lệnh trên full dev: AP50 motorcycle **0,3785→0,3622** (−0,0163), mAP50 **0,4965→0,4931** (−0,0034). Car −0,0022; truck −0,0042; bus +0,0005; bicycle +0,0053. Cổng định trước yêu cầu motorcycle +0,03 và mAP không giảm, nên **loại**. Không mở kiểm HELMET hoặc chấm theo nguồn cho checkpoint đã loại; không sửa checkpoint khóa. Log/mẫu/metric/cổng ở `balanced_pilot/`.

Pilot ảnh UTUAV 4K cố định B frame 1/369/738, C frame 1/376/751, 148 motorcycle GT động, 439 car, 45 bus. YOLOv8n COCO và YOLO11n mixed5 suy luận ảnh toàn khung resize640, ô 960→640 (15 ô) và ô 640→640 (28 ô), conf0,05, NMS0,7; chấm một-một **đúng lớp** IoU≥0,5, bỏ GT tĩnh. **Cả ba cách đều 0/148 motorcycle đúng lớp** cho cả hai checkpoint. Ở ô 640, chỉ 7/148 motorcycle GT có hộp *bất kỳ lớp xe* khớp của COCO, 2/148 ở mixed5; COCO car tăng 1/439 (toàn khung) lên 35/439, nhưng vẫn quá thấp để mở tracking xe máy. Median thời gian/ảnh PC COCO ~0,10 s toàn khung so ~0,55 s với 28 ô; số này chỉ là pilot máy tính, không dùng suy FPS Nano. Kết quả `tile_pilot.json`, `tile_pilot_640.json`. [SAHI](https://arxiv.org/abs/2202.06934) là động cơ thử ô ảnh, không bảo đảm cải thiện trên nhãn và miền UTUAV này. Cần dữ liệu huấn luyện đúng miền/nhãn hoặc detector khác rồi đánh giá lại **cả năm lớp** trước khi thay detector.

## 3. Bộ dữ liệu nên dùng cho từng câu hỏi

Chọn theo GT phương tiện, trường hợp nghiên cứu, timeline và khả năng detector, **không** chọn vì PF tình cờ thắng KF.

- **Xe máy và giao nhau:** HELMET test36 (36 clip/12 nhóm, 9.371 box motorcycle). Audit GT theo ngưỡng đã khóa tìm proxy đổi hướng rõ ở **1/36 clip, 3 cửa sổ**; hộp GT chồng lấn IoU≥0,25 có ở **27/36 clip, 361/2.685 frame có ≥2 xe**. Dùng cho xe máy và giao nhau, ghi rõ hộp nguồn có thể bao cả người ngồi. Chồng lấn không đồng nghĩa che khuất vật lý.
- **Đường cong/rẽ:** RoundaboutHD 4 camera vòng xuyến; [bài tác giả](https://arxiv.org/abs/2507.08729) mô tả chuyển động phi tuyến và che khuất. Mẫu 3.000 frame đầu/camera đã thử, nhưng chưa có phân tầng theo sự kiện rẽ; cần khóa quy tắc sự kiện trước khi kết luận riêng cho rẽ.
- **Che khuất có nhãn:** [SVMOT](https://zenodo.org/records/19468203) có visibility, cặp che khuất và depth; test hiện 9 chuỗi/2 scene, phân tích cấp scene. [UAVDT](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.pdf) có nhãn mức che khuất car/truck/bus; M1306 là ca khó, nhưng **không có xe máy** và detector recall che nặng chỉ 26,7% ở conf0,05. Tám train scene R90 đã là development; test chính thức nên giữ cho ứng viên khóa sau này.
- **Chưa đủ cổng tracking cho mục tiêu này:** UTUAV có xe máy/timeline chuẩn nhưng detector recall cực thấp; VisDrone thiếu stride hiệu dụng xác thực; TrafficMOT thiếu nhãn test/FPS. BDD và UA-DETRAC là đối chứng năm lớp đã dùng nhiều lần, không phải test mù cho PF mới. Lớp bicycle chưa có tập ngoài độc lập đạt mọi cổng; không suy kết quả car/motorcycle cho bicycle.

## Kết luận và việc còn mở

R91 xác định vấn đề PF nghiêng về **drift của track Lost**, nhưng biến thể association thử không đủ ổn định để chọn. Hai cách cải thiện detector được thử không tăng recall motorcycle theo tiêu chí đã khóa. Giữ baseline luận văn/Nano. Hướng tiếp theo nếu tiếp tục: (1) thiết kế PF nhiều giả thuyết liên kết trong che khuất và đánh giá trên test xe mới; (2) dùng dữ liệu huấn luyện xe máy đúng góc nhìn/định nghĩa box, kiểm AP từng lớp cùng chi phí suy luận. Không gán cải thiện IDSW cho PF khi detector hoặc thuật toán association cũng đã đổi.
