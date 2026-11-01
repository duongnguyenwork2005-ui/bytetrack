# R63 — báo cáo cuối: C1 384×640 so với đối chứng vuông 640×640 trên UA-DETRAC test40 và BDD val (01–02/11/2026)

> **Sửa theo review GPT 02/11** (`D:/LABOTARY/automation/runs_local/hoi_codex_R64_review_smoke_20261102/answer.json`,
> A: DAT_CO_SUA): làm tròn mAP50 BDD val đối chứng 0,2440; câu kết luận; tái lập AP test40 "khớp ở độ chính xác báo
> cáo"; script `g4_bao_cao_cuoi.sh` thêm fail-fast/pipefail (cho lần sau; kết quả đã đối chiếu, không chạy lại).

**Chỉ báo cáo** — C1 đã chốt trên dev trước khi chạy (GPT 01/11, `g4/10_ket_qua_g4.md`); số dưới đây không dùng để chọn
lại hay tinh chỉnh. Detector `models/yolov8n.pt` (`f59b3d83…`, kiểm sha256 trong script) FP32 trên PC (GTX 1650,
ultralytics 8.4.141); đối chứng = letterbox vuông 640×640 (Nano v16, `det_cache --no-rect` / `--hinh-dang 640x640`),
ứng viên C1 = 384×640 (rect; trùng pixel engine C1 trên Nano). Script: `cong_cu/g4_bao_cao_cuoi.sh` (log `01_…08_*.log`).

- **test40:** giao thức khóa R54 (`configs/phase2/r54_tracking/giao_thuc_test40.json`): 40 video, lớp 2 3 5 7, 25 fps,
  buffer 1,2 s (30 bước), elapsed skip, evaluator `mot`, tô xám vùng bỏ qua; AP theo D1–D5 (`mAP (car, bus)`).
- **BDD val:** giao thức khóa R61 (`configs/phase2/r61_bdd/giao_thuc.json`): 200 video COCO-video val, lớp 1 2 3 5 7,
  5 fps, buffer 6 bước, elapsed steps, evaluator `mot_crowd`; AP 5 lớp.
- Cache `data/interim/det_cache/r63_yolov8n-coco_c0.05_{test40,bddval}_{vuong640,rect640}`: 80 + 400 video,
  0 unreadable/truncated, 0 cảnh báo NMS quá giờ.

**Kiểm tái lập:** nhánh C1 (rect640) khớp các lượt lịch sử cùng detector — tracking test40 chênh tối đa ~6e-9 trên mọi
cột COMBINED (IDSW trùng), BDD val trùng tuyệt đối HOTA/IDF1/IDSW — test40 c25 CV/EKF HOTA
0,5824 / 0,5835, IDSW 2498 / 2525 = `p2-expA_detrac_test-c25-{cv,ekf}`; BDD val c10 CV/EKF HOTA 0,3410 / 0,3435, IDSW
6262 / 5437 = R61 `cong_b_20261027_01/tong_hop/bang_chinh.csv`; AP test40 384×640 mAP50 0,87013 khớp lượt 10/10
(`results/phase2/detector_eval/detrac_test/20261010_yolov8n_coco`) **ở độ chính xác báo cáo, không trùng tuyệt đối**
(chênh mAP50 ~6,2e-10, mAP50–95 ~1,24e-8). Tức C1 chính là đường PC mọi vòng R53–R61 đã dùng;
thứ thay đổi trong R63 là **đối chứng** (tái hiện đúng Nano v16 vuông 640×640). AP BDD val 384×640 cũng trùng R43.

## Tracking (Δ = C1 − 640×640, điểm thang 100; CI95 bootstrap ghép cặp theo video, 5.000, seed 0)

`tracking_{test40,bddval}_so_cap.json` (`g4_tong_hop_tracking.py`, `boot_stats.paired_bootstrap`, COMBINED_SEQ).

| Tập | Điều kiện | HOTA 640 → C1 | ΔHOTA [CI] | ΔIDF1 [CI] | IDSW 640 → C1 | Đọc cổng Q2 |
|:--|:--|:--|:--|:--|:--|:--|
| test40 | c25 CV | 0,5814 → 0,5824 | +0,09 [−0,20; +0,41] | +0,24 [−0,28; +0,79] | 2701 → 2498 (−7,5 %) | đạt |
| test40 | c25 EKF | 0,5815 → 0,5835 | +0,20 [−0,11; +0,55] | +0,46 [−0,08; +1,07] | 2710 → 2525 (−6,8 %) | đạt |
| test40 | c10 CV | 0,5792 → 0,5797 | +0,05 [−0,26; +0,37] | +0,22 [−0,37; +0,81] | 2224 → 2076 (−6,7 %) | độ nhạy |
| test40 | c10 EKF | 0,5824 → 0,5830 | +0,06 [−0,27; +0,41] | +0,17 [−0,40; +0,74] | 2218 → 2058 (−7,2 %) | độ nhạy |
| BDD val | c25 CV | 0,3369 → 0,3368 | −0,02 [−0,11; +0,07] | −0,09 [−0,23; +0,05] | 6478 → 6345 (−2,1 %) | đạt |
| BDD val | c25 EKF | 0,3384 → 0,3383 | −0,02 [−0,09; +0,06] | −0,11 [−0,24; +0,03] | 5637 → 5548 (−1,6 %) | đạt |
| BDD val | c10 CV | 0,3416 → 0,3410 | −0,06 [−0,18; +0,06] | −0,11 [−0,29; +0,08] | 6381 → 6262 (−1,9 %) | độ nhạy |
| BDD val | c10 EKF | 0,3430 → 0,3435 | +0,05 [−0,03; +0,14] | +0,02 [−0,11; +0,16] | 5557 → 5437 (−2,2 %) | độ nhạy |

Mọi CI ΔHOTA chứa 0 và nằm trong ±0,6 điểm; IDSW của C1 thấp hơn ở cả 8 ô (không suy nguyên nhân).

## Recall detection theo tầng, c25 (`tang_{test40,bddval}_c25/`, ghép IoU ≥ 0,5, bootstrap theo video)

| Tập | Tầng | n GT (video có GT) | Δ recall [CI] (điểm %) |
|:--|:--|--:|:--|
| test40 | xe nhỏ (√diện tích < 24 px) | 11.689 (18/40) | −0,60 [−2,05; −0,09] |
| test40 | che nặng (visibility < 0,5) | 69.484 (37/40) | −0,81 [−1,34; −0,46] |
| test40 | xe cỡ vừa (24–48 px) | 184.682 (40) | −1,65 [−2,11; −1,10] |
| test40 | tất cả | 675.774 (40) | −0,75 [−0,98; −0,49] |
| BDD val | xe nhỏ | 71.156 (200) | −0,18 [−0,23; −0,14] |
| BDD val | tất cả | 382.903 (200) | −0,30 [−0,34; −0,27] |

Recall C1 **thấp hơn nhẹ và nhất quán** (CI không chứa 0) ở hầu hết tầng; ước lượng điểm hai tầng có cổng (xe nhỏ, che
nặng) trong ngưỡng 2 điểm; cận dưới CI xe nhỏ test40 (−2,05) và xe cỡ vừa (−2,11, tầng không có cổng) chạm/vượt −2.
**BDD val không có tầng che khuất:** GT COCO-video val có visibility = 1,0 ở 100 % dòng xe (kiểm 01/11: 382.903/382.903)
— thiếu thông tin, không phải "đạt". Precision BDD chưa trừ vùng bỏ qua lớp 13 (chỉ recall dùng để đọc).

## AP (`results/phase2/r63_g4/ap_{test40,bddval}_{640x640,384x640}/`)

| Tập | Chỉ số | 640×640 → C1 | Δ (điểm) |
|:--|:--|:--|:--|
| test40 (56.167 ảnh) | mAP50 (car, bus) | 0,8718 → 0,8701 | −0,17 |
| test40 | mAP50–95 (car, bus) | 0,6005 → 0,5997 | −0,08 |
| test40 | AP50 car / bus | 0,860 → 0,858 / 0,884 → 0,883 | −0,19 / −0,14 |
| BDD val (39.973 ảnh) | mAP50 (5 lớp) | 0,2440 → 0,2439 | −0,01 |
| BDD val | mAP50–95 (5 lớp) | 0,1573 → 0,1571 | −0,02 |
| BDD val | AP50 car / truck / bus / moto / xe đạp | 0,557 / 0,185 / 0,249 / 0,116 / 0,114 → 0,558 / 0,183 / 0,249 / 0,117 / 0,113 | — |

AP BDD val 384×640 trùng lượt R43 (`results/phase2/detector_eval/bdd_val_cocovid/r43_yolov8n_coco`: mAP50 0,24394,
mAP50–95 0,15707). 0 cảnh báo NMS quá giờ ở cả bốn lượt AP.

Chỉ có ước lượng điểm: **CI AP cho test40/BDD val chưa tính** (bootstrap theo video với công cụ hiện tại cần hàng chục
giờ trên 56 nghìn ảnh; việc còn lại, không chặn kết luận vì báo cáo cuối không đổi quyết định).

## Kết luận báo cáo cuối

Trên hai tập báo cáo, kết quả cùng chiều với dev: ước lượng điểm tracking c25 nằm trong ngưỡng suy giảm định trước
(ΔHOTA −0,02…+0,20 ở c25; −0,06…+0,20 nếu tính cả c10; CI ΔHOTA chứa 0 — không dùng CI chứa 0 để khẳng định không có
ảnh hưởng; IDSW thấp hơn), AP
test40 giảm 0,17 điểm mAP50 và BDD val 0,01 điểm, recall detection giảm nhẹ nhất quán (−0,2…−0,8 điểm ở tầng tổng / có cổng). Cách viết cho
thầy giữ nguyên câu đã chốt (nay có thêm bằng chứng test40/BDD val): FPS +43 % trên Nano này; chất lượng tracking và AP trong ngưỡng suy giảm
đã định trước; recall giảm nhẹ; **không** viết "hoàn toàn không ảnh hưởng". G6: đề xuất v17 = C1; launcher giữ v16.
