# R63 G4 — chất lượng C1 384×640 so với đối chứng vuông 640×640 (dev, 31/10/2026)

**Phạm vi:** lựa chọn chỉ trên dev (UA-DETRAC dev12 `results/phase2/manifests/detrac_dev.json`; BDD dev 139 video của
split mixed5 đã khóa). **Chưa đánh giá GT test40 / BDD val** khi chọn C1 (MVI_39031 thuộc test40 chỉ dùng đo tốc độ
và tương đương thiết bị); báo cáo cuối sau khi chốt: `bao_cao_cuoi/`. Detector `models/yolov8n.pt`
(`f59b3d83…`) FP32 trên PC (GTX 1650, ultralytics 8.4.141). **Cập nhật 01/11 theo review GPT** (`hoi_codex_R63_chot_c1_20261101/answer.json`): CI AP
tính lại theo video (bản 2), Nano↔PC A chạy lại sạch, hai wrapper báo cáo cuối có `--hinh-dang`, cách đọc cổng đã chốt.
Đối chứng = letterbox vuông 640×640 (tái hiện Nano v16,
`det_cache --no-rect` / `eval --hinh-dang 640x640`), ứng viên = 384×640 (rect; với ảnh 16:9 trùng pixel letterbox
(384,640) auto=False của engine C1 — `v17_c/test_v17_c.py`).

## Sửa mã (giữ hành vi mặc định)

- `src/det_cache.py`: `DetectorSpec.rect` (mặc định True), chỉ vào `gen_spec`/hash khi False ⇒ 4/4 meta có
  `spec_sha256` vẫn khớp; `detect_frame` chỉ truyền `rect=False` khi đặt; CLI `--no-rect`. Test: `test_det_cache`
  51/51, `test_cache_provenance` 12/12, `test_track_video_file` 27/27; rect=False trùng NanoDetector 640 (|d| ≤ 1,2e-4).
- `src/eval_detector_5lop.py`: `--hinh-dang HxW` → `imgsz=(H,W)`, `rect=False` (tensor cố định, không phụ thuộc thành
  phần batch — mặc định cũ chọn rect/vuông theo batch có cùng kích thước hay không), ghi vào `quy_tac`. Test
  `test_detector_5lop` 92/92. Hai wrapper `danh_gia_detector_bdd_val/detrac_test` có `--hinh-dang` (01/11; kiểm sớm
  giá trị sai; test 22/22, 14/14); `v17_c/test_v17_c.py` 7/7 (thêm ảnh BDD 1280×720 và tensor batch trộn).
- Công cụ R63: `cong_cu/g4_*.py|sh`, `chat_luong/phan_tang_kich_thuoc.py --nguon bdd` (test 5/5).

## Dữ liệu dev

- Cache: `data/interim/det_cache/r63_yolov8n-coco_c0.05_{dev12,bdddev}_{vuong640,rect640}` (conf 0,05, iou 0,7,
  max_det 3000, tô xám vùng bỏ qua như cũ); 24 + 278 video, 0 unreadable/truncated, 0 cảnh báo NMS quá giờ.
- Split tracking BDD dev (Q3): `g4/bdd_dev_split.json` — `BDDDEV-train`, 139 video, 27.513 frame, 230.179 box xe
  đánh giá (lớp 1 = car/truck/bus/motorcycle/bicycle), 12.893 box bỏ qua (lớp 13), 65,5 % box xe `occluded`.
- Tracker khóa: dev12 theo R54 (lớp 2 3 5 7, 25 fps, buffer 30 bước, elapsed skip, evaluator `mot`); BDD dev theo R61
  (lớp 1 2 3 5 7, 5 fps, buffer 6 bước, elapsed steps, `mot_crowd`). c25 = cổng; c10 = độ nhạy.

## Kết quả (Δ = ứng viên − đối chứng, điểm thang 100; CI95 bootstrap ghép cặp theo video, 5.000, seed 0)

### Tracking (`g4/tracking_{dev12,bdddev}_so_cap.json`, COMBINED_SEQ qua `boot_stats.paired_bootstrap`)

| Miền | c25 | HOTA đối chứng → ứng viên | ΔHOTA [CI] | ΔIDF1 [CI] | IDSW | Cổng |
|:--|:--|:--|:--|:--|:--|:--|
| dev12 | CV | 0,5892 → 0,5900 | +0,08 [−0,44; +0,44] | +0,36 [−0,58; +1,05] | 482 → 489 (+1,5 %) | ĐẠT |
| dev12 | EKF | 0,5917 → 0,5925 | +0,08 [−0,39; +0,42] | +0,31 [−0,48; +0,88] | 469 → 472 (+0,6 %) | ĐẠT |
| BDD dev | CV | 0,3526 → 0,3529 | +0,02 [−0,09; +0,14] | +0,07 [−0,11; +0,26] | 4618 → 4522 (−2,1 %) | ĐẠT |
| BDD dev | EKF | 0,3529 → 0,3531 | +0,02 [−0,11; +0,17] | +0,03 [−0,13; +0,22] | 4094 → 4039 (−1,3 %) | ĐẠT |

c10 (độ nhạy): dev12 ΔHOTA +0,52 (CV) / +0,79 (EKF), IDSW −4,3 / −8,0 %; BDD dev ΔHOTA −0,01 / +0,03.

### Recall detection theo tầng, c25 (`g4/tang_{dev12,bdddev}_c25/`)

| Miền | Tầng | n GT | Δ recall [CI] | Cổng (giảm ≤ 2 điểm) |
|:--|:--|--:|:--|:--|
| dev12 | xe nhỏ (√diện tích < 24 px) | 58 (4/12 video có GT) | 0,00 [0,00; 0,00] | **không đủ GT ⇒ không tính đạt** |
| dev12 | che nặng (visibility < 0,5) | 15.269 | −1,01 [−1,77; −0,40] | ĐẠT |
| dev12 | tất cả | 116.845 | −0,55 [−0,81; −0,17] | — |
| BDD dev | xe nhỏ | 35.675 (137/139) | −0,35 [−0,46; −0,25] | ĐẠT |
| BDD dev | `occluded` (cờ nhị phân) | 150.806 | −0,45 [−0,52; −0,39] | ĐẠT |
| BDD dev | tất cả | 230.179 | −0,38 [−0,44; −0,33] | — |

Recall detection của ứng viên **thấp hơn nhẹ và nhất quán** (CI không chứa 0 ở hầu hết tầng) nhưng nằm trong ngưỡng;
tracking không giảm. Tầng xe nhỏ chỉ đánh giá được trên BDD dev (dev12 gần như không có xe < 24 px).

### AP 5 lớp, dev mixed5 (`results/phase2/r63_g4/ap_dev_mixed5_{640x640,384x640}/`, `g4/ap_dev_theo_mien_v2.json`)

| Miền | mAP50 đối chứng → ứng viên | ΔmAP50 [CI] | mAP50–95 | ΔmAP50–95 [CI] | Cổng (≥ −0,5, ước lượng điểm) |
|:--|:--|:--|:--|:--|:--|
| UA-DETRAC dev12 (1.631 ảnh, 12 video) | 0,6122 → 0,6093 | −0,29 [−1,02; +0,24] | 0,4563 → 0,4545 | −0,19 [−0,76; +0,08] | ĐẠT theo điểm; cận dưới CI < −0,5 |
| BDD dev (5.559 ảnh, 139 video) | 0,2658 → 0,2659 | +0,01 [−0,18; +0,26] | 0,1600 → 0,1603 | +0,03 [−0,07; +0,11] | ĐẠT |
| gộp (7.190 ảnh) | 0,2989 → 0,2975 | −0,14 | 0,1944 → 0,1937 | −0,07 | — |

CI95 **bản 2** (01/11): bootstrap ghép cặp **theo video** 5.000 lần, seed 0 (`cong_cu/g4_ap_theo_mien.py` v2): ảnh → video
qua `data/yolo_mixed5_v1/manifest/anh.csv` (basename `anh_ra` → (video, nguồn)), kiểm đúng danh sách 12 video dev12 và
139 video BDD dev; ghép (ảnh, lớp) một lần, mỗi mẫu nối khối video rồi gọi `ev.ap_coco`; điểm toàn tập khớp
`ev.danh_gia` ≤ 1e-12; test hồi quy `cong_cu/tests/test_g4_ap_theo_mien.py` 4/4; log `g4/07_ap_theo_mien_v2.log`.
**Bản 1 sai** (`g4/ap_dev_theo_mien.json`, `06_ap_theo_mien.log`, giữ làm bằng chứng lỗi — GPT phát hiện): lấy video
bằng `rsplit('-')` nên dev12 thành 1.631 cụm (theo ảnh) ⇒ CI quá hẹp; không dùng.

**Cách đọc cổng (GPT 01/11, Q2):** cổng áp trên **ước lượng điểm** như ΔHOTA/ΔIDF1, luôn báo kèm CI hợp lệ. C1 đạt cổng
AP ở cả hai miền theo điểm; trên dev12, CI (12 video) chưa loại trừ mức giảm tới ~1,0 điểm mAP50 — không được viết "AP
không giảm".

### Tương đương Nano ↔ PC (`g4/nano_pc_{A,B}.json`, 300 frame MVI_39031, tiêu chí chốt trước)

| | ghép Nano→PC | PC→Nano | IoU trung vị | cùng lớp | |
|:--|--:|--:|--:|--:|:--|
| C1 384×640 (Nano FP16 v17-C ↔ PC FP32 rect) | 99,73 % | 99,66 % | 0,988 | 100 % | ĐẠT |
| baseline 640×640 (Nano v16 ↔ PC FP32 vuông) | 99,61 % | 99,61 % | 0,985 | 100 % | ĐẠT |

(conf > 0,25; MVI_39031 thuộc test40 nhưng chỉ dùng kiểm tương đương thiết bị như R16, không dùng chọn ứng viên.)
Lượt PC nhánh A đầu có một cảnh báo "NMS time limit" (GPU chạy AP song song); **đã chạy lại khi GPU rảnh**
(`g4/nano_pc_A_lan2.json`, log `g4/08_nano_pc_A_lan2.log`: 0 cảnh báo) — kết quả trùng lần đầu. Ở **conf > 0,05 chưa đạt
99 % hai chiều** (C1 99,0/98,9 %; baseline 98,8/99,0 %) ⇒ không khẳng định tương đương toàn bộ detection, kể cả tầng điểm
thấp ByteTrack dùng (`g3/04_dinh_chinh_provenance.md` mục 6).

## Kết luận dev và cách viết cho thầy (GPT 01/11, Q1/Q2)

C1 qua mọi cổng dev đã chốt theo ước lượng điểm ⇒ **chốt C1** làm ứng viên v17, không mở C2/C3/G2 trong vòng này; báo
cáo cuối test40 + BDD val (đối chứng 640×640 vs C1) chỉ để báo cáo. Giới hạn: tầng xe nhỏ dev12 không đủ GT (58 box) —
**giới hạn, không phải đạt**; recall giảm nhẹ nhất quán ở vài tầng. Câu đề xuất: *"Trên phép đo 300 JPG có hiển thị HDMI
của Nano hiện tại, C1 tăng FPS khoảng 43 %. Trên hai miền dev, ước lượng điểm tracking và AP nằm trong ngưỡng suy giảm đã
định; recall giảm nhẹ, trong đó một số tầng giảm nhất quán. Kết quả không chứng minh chất lượng hoàn toàn không bị ảnh
hưởng; tầng xe nhỏ dev12 chưa đủ dữ liệu."* Mức tăng FPS là của Nano này; thiết bị Jetson khác phải đo lại.
