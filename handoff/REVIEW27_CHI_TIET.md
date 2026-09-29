# Vòng 27 — chuẩn bị dữ liệu và đường fine-tune detector 5 lớp (chi tiết)

Base review: `239b9c6` (Codex: PASS WITH LIMITATIONS, `handoff/CODEX_REVIEW_239b9c6.md`). Phạm vi: `handoff/PLAN_DA_CHOT.md`.
Vòng này **không** chạy training dài, benchmark tracking hay tối ưu Nano. Nano v16 không bị đụng tới.

Cách đọc nhãn trạng thái trong phụ lục này:

- **chạy thật**: chạy trên dữ liệu/thiết bị thật của máy;
- **fixture**: chạy thật trên GPU nhưng với dữ liệu tổng hợp;
- **bị chặn**: chưa làm được vì thiếu dữ liệu.

## 1. Chốt nghiệm thu, base

- `handoff/CODEX_REVIEW_239b9c6.md`: 9.043 B, sha256 `3129e1d9…`, cùng byte với bản trong `~/Downloads`.
- Gói đã review được lưu thành `handoff/archive/review_delta_239b9c6.zip`: 11.188.466 B, sha256 `854b09a3…`, khớp báo
  cáo. Gói có 547 thành viên, testzip OK, MANIFEST `f86732b → 239b9c6`.
- Base chuyển sang `239b9c6` bằng `make_review_delta.py --set-base 239b9c6 --round 26 …`; 13 khai báo NPZ được mang sang.
- Trạng thái đầu vòng (`logs/review27/00_trang_thai_dau.log`): HEAD `239b9c6`, 0 commit sau đó. Working tree chỉ có file
  riêng của người dùng (`.gitignore`, `src/demo/rerun_with_state_log.py`).

## 2. Kiểm kê dữ liệu thật (**chạy thật**)

Chi tiết ở `handoff/BDD100K_INVENTORY.md`, mục “Vòng 27”; số liệu thô ở `logs/review27/01_kiem_ke_bdd100k.{json,log}`.

- **Có trên máy**: chỉ **BDD100K MOT val** — archive `images20-track-val-1.zip`, ảnh 200 video / 39.973 ảnh (toàn bộ
  1280×720), nhãn COCO-video 465.512 annotation.
- **Không có**: ảnh và nhãn MOT **train**, gói detection 100K.
- motorcycle trên val chỉ có 902 box trong 774 ảnh của 29/200 video.
- 2 frame val trùng byte: video `b1d7b3ac-5744370e`, frame 79 và 80.

## 3. Rà soát đường cũ (mã tại `239b9c6`)

| File | Vấn đề | Xử lý vòng 27 |
|:--|:--|:--|
| `prepare_mixed_yolo_dataset.py` | Mặc định trộn UA-DETRAC; thiếu BDD train thì lặng lẽ ghi “bỏ qua” (dòng 155–158) | Không còn nguồn mặc định (exit 1). BDD chỉ xuất qua `--config` BDD-only. Cấu hình trộn nguồn bị từ chối. BDD thiếu → exit 2 kèm đường dẫn cụ thể, không tạo thư mục đích |
| (cùng file) `export_bdd` | Bỏ mọi frame không có box 5 lớp (dòng 117), mất ảnh âm tính hợp lệ | Giữ âm tính trong khoảng có nhãn; loại đuôi video không nhãn |
| (cùng file) `export_bdd` | Tô xám crowd đè cả box có nhãn; không xử lý distractor (`other vehicle` thành nền) | Tô xám trừ pixel box có nhãn; distractor nhóm xe cũng được tô xám |
| (cùng file) `export_bdd` | Chia 10% ngẫu nhiên không phân tầng; không kiểm trùng tập đánh giá; mã hóa lại mọi JPEG (q92); chỉ đọc Scalabel | Xem mục 4 |
| `train_detector.py` | `--dry-run` = train 1 epoch, fraction 0.02 | `--dry-run` = kiểm cấu hình, không train, không tạo thư mục; smoke là chế độ riêng |
| (cùng file) | `exist_ok=True` → lượt mới ghi đè lượt cũ | Từ chối nếu thư mục lượt hoặc file kết quả đã tồn tại |
| (cùng file) | optimizer `auto`, batch 8 mặc định chưa kiểm VRAM; không kiểm hash checkpoint | Cấm `auto`; ghi giá trị ultralytics resolve lúc chạy; kiểm sha256 |
| `eval_detector.py` | Một lớp “vehicle”, bỏ class ID khi đọc GT (dòng 94–96); kích thước cố định 960×540; lọc COCO `[2,3,5,7]` (thiếu bicycle); AP nội suy mọi điểm kiểu VOC | Giữ nguyên để tái lập kết quả Giai đoạn C (docstring ghi rõ không dùng cho 5 lớp). Evaluator mới: `eval_detector_5lop.py` |

## 4. Dataset BDD → YOLO 5 lớp

Mã: `src/bdd_yolo_export.py`, CLI `prepare_mixed_yolo_dataset.py --config`, cấu hình `configs/phase2/detector/bdd5_train.yaml`.

- **Nguồn**: chỉ split `train` của BDD100K MOT 2020. Đọc được hai định dạng nhãn: Scalabel `box_track_20` và COCO-video.
  Test C1–C3 kiểm hai định dạng cho cùng split, cùng nhãn, cùng vùng bỏ qua và cùng ảnh xuất, giống từng byte.
- **Chia theo video**:
  - 10% dev, seed 0, phân tầng theo *sự có mặt* của motorcycle và bicycle trong video;
  - số dev mỗi tầng = `floor(0,10·n + 0,5)` (làm tròn nửa lên, không dùng `round()` kiểu banker);
  - danh sách được **khóa** vào `configs/phase2/detector/splits/<ten>.json` kèm `sha256_danh_sach` và hash nguồn nhãn.
    Split đã tồn tại thì không ghi đè; bước xuất từ chối nếu nhãn đổi sau khi khóa hoặc file split bị sửa tay.
- **Chọn frame**: giữ frame có chỉ số 0-based chia hết cho `buoc_frame` (5, tức ~1 ảnh/giây). Quy tắc này dùng chung cho
  mọi model.
- **Ảnh âm tính**:
  - frame nằm trong `[frame có nhãn đầu, frame có nhãn cuối]` của video → giữ, file nhãn rỗng;
  - frame ở đuôi video sau frame có nhãn cuối → loại, đếm riêng;
  - mọi ảnh xuất ra đều có `.txt`. Ultralytics coi ảnh thiếu `.txt` là nền, nên bước kiểm bắt trường hợp này (D9).
- **crowd và ignore**:
  - vùng crowd/ignore thuộc 5 lớp, cùng `other vehicle` và `trailer` → tô xám (114,114,114), **trừ** pixel nằm trong
    box có nhãn;
  - tọa độ vùng tô ghi vào `ignore/<split>/<anh>.txt`;
  - pedestrian, rider, train, other person: không gán nhãn, không tô.
- **Tọa độ và file ảnh**:
  - tọa độ theo kích thước **thật** đọc từ header file; nhãn khai báo kích thước khác → lỗi;
  - box được cắt vào khung; box còn dưới 1 px thì bỏ;
  - ảnh không tô được sao chép nguyên byte. Không dùng hard link vì ultralytics có thể ghi lại JPEG hỏng.
- **Kiểm trùng tập đánh giá** (BDD val thật, UA-DETRAC test thật): chạy **trước** khi ghi ảnh, theo tên video, tên ảnh
  và sha256 nội dung. Hash toàn bộ 39.973 ảnh BDD val được cache (theo size/mtime) ở `data/interim/hash_tap_danh_gia/`
  (bị ignore).
- **Đầu ra**:
  - `data.yaml`: `val: images/dev` là **dev nội bộ**, không phải BDD val chính thức;
  - `manifest/anh.csv`: mỗi ảnh một dòng, có sha256 nguồn/đích, W×H, số box theo lớp;
  - `manifest/tom_tat.json`: quy tắc, thống kê ảnh/box/video theo lớp, hash;
  - `manifest/kiem_sau_xuat.json`.

  Xuất vào thư mục tạm rồi đổi tên; thư mục đích đã có thì từ chối. Mỗi dataset có `.gitignore` `*` bên trong.
- **Tập con smoke**: `--tap-con smoke` gồm 16 video train-fit và 4 video dev, bước 10 frame, nhãn loại “SMOKE”.

## 5. Evaluator 5 lớp (`src/eval_detector_5lop.py`)

Cùng một quy tắc cho YOLOv8n COCO, YOLOv8n fine-tune và YOLO11n fine-tune.

- **Ánh xạ lớp** theo **tên** đọc từ checkpoint (`src/detector_classes.py`):
  - COCO: `2→car, 7→truck, 5→bus, 3→motorcycle, 1→bicycle`;
  - checkpoint 5 lớp: ánh xạ đồng nhất;
  - thiếu hoặc trùng lớp → lỗi.

  Ánh xạ được ghi cùng kết quả. `sang_id_coco()` giữ quy ước `--classes 1 2 3 5 7` để tạo cache sau này.
- **Dự đoán**: imgsz 640, conf 0,001, NMS 0,7 theo lớp, max_det 300, FP32; chỉ lấy lớp ánh xạ được, lọc trước NMS.
- **Ghép và AP**: theo COCO — 10 ngưỡng IoU, nội suy 101 điểm. Vùng tô xám xử lý như crowd (IoA). Lớp không có GT →
  AP `null`, không vào mAP. Precision/recall tính tại conf 0,25 và IoU 0,5, **theo lớp**; không gộp lớp.
- **Test**:
  - E1: AP50 và AP50–95 theo lớp khớp pycocotools, chênh 0;
  - E3: P/R tính tay;
  - E4: kích thước ảnh thật;
  - E5: thiếu nhãn → lỗi;
  - E6: YOLOv8n COCO thật (CPU) chỉ ra lớp 0..4.

## 6. Runner huấn luyện (`src/train_detector.py`, `configs/phase2/detector/train_bdd5.yaml`)

- **Chế độ**:
  - `kiem-cau-hinh` (= `--dry-run`): không train, không tạo thư mục;
  - `smoke`;
  - `day-du`: bắt buộc `--xac-nhan-train-dai`.
- **Không ghi đè**: tự kiểm thư mục lượt và file kết quả trước, rồi truyền `exist_ok=False`. Ultralytics tự đổi tên khi
  thư mục đã có, và runner coi việc đó là lỗi.
- **Chính sách chung** cho hai model: khối `huan_luyen` của cấu hình; chỉ `weights` khác nhau, kiểm sha256.
  `optimizer/batch/amp/workers/...` không được để `auto`.
- **Ghi lại**:
  - tham số khai báo và tham số `get_cfg` resolve;
  - giá trị hiệu lực lúc chạy (callback): optimizer và param group (lr, momentum, weight_decay, nesterov), `amp`,
    `accumulate`, số worker, số ảnh, `nc/names`;
  - loss từng epoch và kiểm hữu hạn; thời gian mỗi batch ổn định (bỏ 5 batch đầu);
  - VRAM (`torch.max_memory_*` và `nvidia-smi` toàn GPU), RAM trống tối thiểu;
  - sha256 `best.pt`/`last.pt` và kết quả nạp lại (đúng 5 lớp, `train_args` khớp cấu hình);
  - đánh giá dev bằng evaluator 5 lớp.
- **An toàn**: lỗi trong phần ghi nhận không làm hỏng lượt train, nhưng làm smoke FAIL.

## 7. Smoke đường chạy trên GPU — **fixture**, không phải dữ liệu thật

Bằng chứng ở `logs/review27/fixture_gpu/`: script `05_*.sh`, `06_*.sh`, `07_*.sh`, log từng lượt, `ket_qua/*.json` và
`tom_tat_smoke.csv` (10 lượt).

**Dữ liệu fixture**:

- sinh bằng `make_bdd5_train_fixture.py`: 30 video × 60 frame, 1280×720, JPEG ~140 KB, cấu hình `fixture_gpu/bdd5_fixture.yaml`;
- chạy đúng luồng thật: `kiem-nguon` → `chia` (27 video train-fit / 3 video dev) → `xuat --tap-con smoke`;
- kết quả: 719 ảnh train / 90 ảnh dev, `kiem_sau_xuat` 0 lỗi;
- kiểm trùng với **BDD val thật**: băm đủ 39.973 ảnh, 39.972 hash khác nhau, không trùng;
- xuất lại cho `anh.csv` cùng sha256 `ff162332…` (tái lập).

**Smoke cuối, đúng cấu hình đề xuất** (`train_bdd5.yaml` sha256 `8db4d265…`, không ghi đè; `07_smoke_cau_hinh_cuoi.log`):

| | YOLOv8n | YOLO11n |
|:--|:--|:--|
| `--dry-run` | exit 0, không tạo thư mục | exit 0, không tạo thư mục |
| Smoke 1 epoch | ĐẠT, 114,4 s | ĐẠT, 137,0 s |
| Hiệu lực (callback) | batch 8, accumulate 8, SGD nesterov lr 0,01 / momentum 0,937 / wd 0,0005 (nhóm weight), `amp=False`, 2 worker, nc 5 | như YOLOv8n |
| Loss (box / cls / dfl) | 0,977 / 2,649 / 0,988, hữu hạn | 1,004 / 3,067 / 1,056, hữu hạn |
| `best.pt` sha256 | `75aa5acc…` (6.220.963 B) | `84e9fc13…` (5.443.475 B) |
| Nạp lại | đúng 5 lớp; `train_args` khớp cấu hình | đúng 5 lớp; `train_args` khớp cấu hình |
| Đánh giá dev (evaluator 5 lớp) | chạy được: 90 ảnh, 2,7 s, ánh xạ `lop5` đồng nhất | chạy được: 2,8 s |
| s/batch · ảnh/s | 0,97 · 8,27 | 1,16 · 6,91 |
| VRAM (PyTorch reserved · nvidia-smi max) | 2.130 · 2.485 MiB | 2.346 · 2.696 MiB |
| GPU bận (trung vị) | 95% | 94% |

Số AP trên dev fixture chỉ chứng minh evaluator chạy được. Chúng **không** nói gì về chất lượng model, và không được
dùng để so sánh hai model.

**Chọn batch và worker** (bảng đầy đủ ở `DETECTOR_TRAIN_PLAN.md`, mục 4):

- batch 16 tràn VRAM với cả hai model, chậm 2–4 lần;
- batch 12 vừa YOLO11n nhưng không nhanh hơn;
- 2 worker nhanh hơn 0 worker; `deterministic` tắt chỉ nhanh ~6%.

**Log lỗi được giữ lại**:

- `04a_smoke_yolov8n_b16_LOI_tloss_dict.*`: lượt đầu, callback đọc `tloss` kiểu tensor trong khi ultralytics 8.4 trả
  dict → `TypeError`. Đã sửa, và thêm lớp bảo vệ để lỗi ghi nhận không làm hỏng lượt train mà chỉ làm smoke FAIL.
- `02a_xuat_smoke_dem_hash_cu.log`: bộ đếm cũ báo 39.972 là “số ảnh đã băm”, thực chất là số hash khác nhau. Giờ ghi
  riêng hai số.

## 8. Test

| Suite | Kết quả | Log |
|:--|:--|:--|
| `test_detector_5lop` lần đầu | 47/52 — 5 FAIL thật: `round()` kiểu banker; `data.yaml` trỏ thư mục tạm, kéo theo lỗi ở E/F1/F5 | `02a_test_detector_5lop_lan1_47_52.log` |
| `test_detector_5lop` sau sửa | **52/52** | `02b_test_detector_5lop.log` |
| `run_all_tests --only test_detector_5lop test_bdd100k_adapter test_bdd100k_coco` | 3/3 PASS (52/52, 42/42, 11/11) | `tests_20260929-1618.log` |

Không chạy lại full suite. Vòng này không sửa đường tracking, Nano, cache hay evaluator tracking đã nghiệm thu:
- `eval_detector.py` chỉ thêm docstring;
- `mixed5.yaml` chỉ thêm comment;
- `run_all_tests.py` chỉ thêm một dòng đăng ký test.

## 9. Bị chặn

Chưa có BDD100K MOT train, nên các bước sau chưa chạy được: khóa split thật, xuất dataset thật, smoke trên dữ liệu thật,
số thống kê motorcycle trên train, và ngân sách thời gian đo trên ảnh thật. Gói cần tải và đường dẫn: `BDD100K_INVENTORY.md`.

## 10. File và commit

- `202a05a`: chốt nghiệm thu gồm `CODEX_REVIEW_239b9c6.md`, `review_base.json` (base `239b9c6`), `PLAN_DA_CHOT.md`,
  `logs/review27/00_trang_thai_dau.log`.
- `4933d41`: mã gồm `detector_classes.py`, `bdd_yolo_export.py`, `prepare_mixed_yolo_dataset.py`,
  `mot_datasets/make_bdd5_train_fixture.py`, `kiem_ke_bdd100k.py`, `eval_detector_5lop.py`, `eval_detector.py` (chỉ
  thêm docstring), `train_detector.py`, `test_detector_5lop.py`, `run_all_tests.py` (đăng ký test) và cấu hình
  `configs/phase2/detector/{bdd5_train,train_bdd5,mixed5}.yaml`.
- `3b5d7d7`: bằng chứng trong `handoff/logs/review27/`.
- Commit tài liệu (commit này): `DETECTOR_TRAIN_PLAN.md`, `REVIEW27_CHI_TIET.md`, `BDD100K_INVENTORY.md` (mục Vòng 27),
  `REVIEW_BRIEF.md`, `CODEX_HANDOFF.md` (hàng 22: kết quả review; hàng 23: vòng 27).

Không nằm trong repo hay delta:

- dataset fixture và checkpoint smoke: nằm trong scratchpad của phiên, hash ghi trong `ket_qua/*.json`;
- cache hash BDD val: `data/interim/hash_tap_danh_gia/`, bị ignore;
- weights, video, dữ liệu gốc.

Ghi chú lịch sử:

- `PHASE2_REPORT.md` (dòng 204, 228) còn mô tả lệnh cũ `prepare_mixed_yolo_dataset.py --dry-run` và
  `train_detector.py --dry-run`. Từ vòng 27, hai lệnh này bị chặn có chủ ý:
  - prepare trả exit 1 vì không còn nguồn mặc định;
  - train đòi `--model` và không còn train.
- Lệnh hiện hành nằm ở `DETECTOR_TRAIN_PLAN.md` mục 6. Báo cáo cũ giữ nguyên làm ghi chép lịch sử.
