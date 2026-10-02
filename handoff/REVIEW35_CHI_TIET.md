# Vòng 35 — khóa split, xuất tập con smoke, smoke 2 model trên dữ liệu thật, dự toán training local (chi tiết)

> **Đính chính vòng 36** (review `2351c67`, R35-F1 và R35-F2; chi tiết ở `REVIEW36_CHI_TIET.md`). Số liệu, log, checkpoint
> và ngưỡng đánh giá **giữ nguyên**; chỉ sửa diễn giải.
>
> - **Mục 7 và 10 (R35-F1)**: checkpoint smoke có confidence thấp hơn checkpoint COCO 80 lớp; **nguyên nhân chưa xác
>   định**. Chuyển sang 5 lớp/khởi tạo và các bước học (gồm có hay không warmup) **chưa được tách riêng**.
> - **Mục 7 (R35-F1)**: kết quả conf 1e-6 chỉ cho thấy có dự đoán **car** khớp GT ở điểm thấp trên 84 ảnh dev smoke. Kết quả
>   này không chứng minh toàn đường ống, mọi lớp hay tọa độ không có lỗi.
> - **Mục 9 và 10 (R35-F2)**: các mức thời gian, RAM và VRAM là **kịch bản dự toán theo các giả định đã nêu**, không phải
>   cận trên hay bảo đảm. Thực tế có thể vượt cả kịch bản chậm hơn.
>
> **Đính chính vòng 37** (`REVIEW37_CHI_TIET.md`): runner cũ để lần gọi `on_fit_epoch_end` trong ultralytics `final_eval` ghi
> đè `val_s` và `metrics` của epoch cuối.
>
> - `val_s` ở mục 6 (33,51 s / 8,76 s) vì vậy gồm cả lưu checkpoint và validate `best.pt`, không chỉ val trong epoch.
> - Số đo giữ nguyên. Lỗi đã sửa cho các lượt sau.
> - Pilot vòng 37 đo được model 5 lớp **trước bước học** có điểm lớp gần 0 (mục 7 được bổ sung ở phụ lục vòng 37).

**Phạm vi**: làm đúng chuỗi người dùng duyệt: ghi nhận nghiệm thu `ccd801f` → khóa split → xuất tập con smoke → smoke trên
dữ liệu thật → cập nhật dự toán training local, **dừng sau smoke**.

- **Không** chạy training dài, không thêm epoch, không bật `--xac-nhan-train-dai`. Số epoch và ngân sách vẫn **chưa duyệt**.
- Không xuất dataset đầy đủ.
- Không SSH hay sửa Nano v16. Không chạy lại tracking.
- `src/` không đổi. `configs/` chỉ thêm file split đã khóa và `.gitattributes` giữ nguyên byte của nó.
- Không đụng file riêng của người dùng.

Nhãn trạng thái dùng trong tài liệu: **ĐÃ CHẠY / CHƯA CHẠY / BỊ CHẶN**; kết quả **PASS / FAIL / SKIP / NOT_RUN**. Số liệu thời gian
ở mục 9 là **ước tính**.

## 0. Trạng thái đầu vòng

Log: `logs/review35/00_trang_thai_dau.log`.

- Nhánh `phase2-mixed-traffic`, HEAD `ccd801f`, 0 commit sau đó. Base review lúc đầu vòng: `4e71d32`.
- Working tree chỉ có file riêng của người dùng và `CODEX_REVIEW_c4d64a1_EVIDENCE/` chưa track ở gốc repo; để nguyên.
- Môi trường:
  - Python 3.11.9, torch 2.6.0+cu124, cuDNN 90100, ultralytics 8.4.141, numpy 2.4.6, OpenCV 5.0.0;
  - GTX 1650 4.096 MiB, driver 555.97;
  - RAM 8,4 GB, lúc đầu còn trống 1,2 GB; ổ D: còn trống 63,5 GB.

## 1. Ghi nhận nghiệm thu `ccd801f` (ĐÃ CHẠY)

Commit `5cac59b`.

- **Báo cáo** `D:\LABOTARY\review_codex\CODEX_REVIEW_ccd801f.md`: 6.153 B, sha256 `af86db9c…`.
  - Kết luận **PASS WITH LIMITATIONS**, không finding chặn mới.
  - Lưu nguyên byte thành `handoff/CODEX_REVIEW_ccd801f.md`.
- **Bằng chứng** `CODEX_REVIEW_ccd801f_EVIDENCE.zip`: 65.537 B, sha256 `716dd900…`.
  - Giải nén vào thư mục mới cạnh ZIP; ZIP gốc giữ nguyên.
  - `SHA256SUMS.txt` khớp **15/15**.
  - Lưu ở `handoff/review_evidence/CODEX_ccd801f/`.
- **Gói đã nghiệm thu**: 805.621 B, sha256 `4880e54f…`, khớp báo cáo. Lưu ở `handoff/archive/review_delta_ccd801f.zip` (git ignore).
- **Base review**: `4e71d32` → **`ccd801f`**, mang sang 13 khai báo NPZ.
- **Phạm vi reviewer**: Codex áp patch và đối chiếu hash **77 đường dẫn thay đổi**. Codex **không** xác nhận độc lập toàn cây
  3.780 file. Log `kiem-nguon` trên Windows là bằng chứng bàn giao, reviewer không chạy lại.

## 2. Khóa split `bdd5_train_v1` (ĐÃ CHẠY)

Commit khóa: `4237149`, **trước** mọi bước xuất, train hay xem AP. Log `01`–`03`.

- `kiem-nguon` **exit 0**: 1.400 video nhãn = 1.400 video ảnh, 278.079 frame, 0 ảnh thiếu.
- `--buoc chia` **exit 0**. Quy tắc theo cấu hình (`bdd5_train.yaml`, sha256 `ca5d602b…`, không đổi):
  - chia **theo video**, tỷ lệ dev 0,10, seed 0;
  - phân tầng theo sự có mặt của motorcycle và bicycle; mỗi tầng làm tròn nửa lên.

| Tầng | Video | Dev |
|:--|--:|--:|
| motorcycle=0, bicycle=0 | 830 | 83 |
| motorcycle=0, bicycle=1 | 303 | 30 |
| motorcycle=1, bicycle=0 | 113 | 11 |
| motorcycle=1, bicycle=1 | 154 | 15 |

| Thống kê nhãn nguồn (trước lấy mẫu) | train-fit | dev |
|:--|--:|--:|
| Video | 1.261 | 139 |
| Frame / frame có nhãn | 250.566 / 247.063 | 27.513 / 27.086 |
| Box car | 1.891.481 | 207.621 |
| Box truck | 134.946 | 14.619 |
| Box bus | 52.948 | 4.836 |
| Box motorcycle | 11.051 | 1.131 |
| Box bicycle | 23.601 | 1.972 |
| Video có motorcycle / bicycle | 241 / 412 | 26 / 45 |
| Vùng tô xám / nhãn lớp khác | 123.369 / 365.596 | 11.490 / 42.255 |

- **0 video chung** giữa train-fit và dev.
- **Trùng tên video** với BDD val (200 video) và UA-DETRAC test (40 video): **0** (`03_*`). Nội dung ảnh được kiểm ở bước xuất,
  chỉ trên ảnh được xuất (mục 3).
- **Hash**:
  - `sha256_danh_sach` `a59a8d06…`: tính lại, khớp;
  - vân tay nguồn nhãn `d2024493…` (sha256 các dòng `tên\tsha256` đã sắp xếp của 1.400 JSON): tính độc lập từ manifest nhập
    vòng 34, khớp.
- **Provenance phương án C** giữ nguyên: nhãn là snapshot HF `jaffe03195/bdd100k_sot` @ `694197099692…`. Native val và COCO
  val không đổi.

### 2a. Giữ nguyên byte file split trong git

Commit `aeed110`; log `09_chuan_hoa_blob_split.log`.

- File split trên đĩa ghi trên Windows có CRLF: 35.795 B, sha256 **`b8a2848e…`**. Đây là hash ghi trong manifest dataset
  (`tom_tat.json` → `file_split.sha256`).
- Commit `4237149` lưu blob đã bị `* text=auto` chuẩn hóa thành LF (34.296 B, sha256 `bc9c85d1…`). Người nhận patch vì thế
  không kiểm lại được `b8a2848e…`.
- **Cách sửa**:
  - thêm `configs/phase2/detector/splits/.gitattributes` và `logs/review35/.gitattributes` với `* -text`;
  - `git add --renormalize`.
- **Kết quả**:
  - blob == file trên đĩa (`b8a2848e…`);
  - JSON parse giống hệt; danh sách video và `sha256_danh_sach` không đổi;
  - file trên đĩa không bị sửa (mtime 12:56:13).
- Làm trước khi có epoch hay AP nào.

## 3. Xuất tập con smoke (ĐÃ CHẠY)

Commit `10c3ef1`. Log `04`–`06`.

```text
python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc xuat --tap-con smoke
python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem --tap-con smoke
```

- Trước khi xuất, D: còn trống 63,4 GB. Thư mục mới `data/yolo_bdd5_v1_smoke/` (không ghi đè, không commit) có 997 file,
  55,9 MB.
- **Chọn tập con** (`chon_tap_con`, `06_kiem_chon_tap_con.json`):
  - `np.random.default_rng(0)`, `permutation(len)[:n]` trên danh sách **đã sắp xếp** của split đã khóa, train rồi dev;
  - 16 video train-fit + 4 video dev, bước 10 frame;
  - chọn **trước** mọi AP; hai model dùng **cùng** tập con.

| | train | dev |
|:--|--:|--:|
| Video | 16 | 4 |
| **Ảnh** (giới hạn ≤ 1.000 / ≤ 200) | **334** | **84** |
| Ảnh âm tính (nhãn rỗng hợp lệ) | 0 | 4 |
| Ảnh có vùng tô xám | 125 | 27 |
| Box car / truck / bus | 2.316 / 204 / 100 | 542 / 6 / 4 |
| Box motorcycle / bicycle | 31 / 65 | **0 / 0** |

- Frame nguồn: 4.043; lệch bước 3.623; ngoài khoảng có nhãn 2.
- **Kiểm trùng**, chạy trước khi ghi ảnh:
  - train/dev cùng nội dung sha256: **0**;
  - BDD val: trùng tên ảnh **0**, trùng nội dung **0**. Đã băm 39.973 ảnh val, 39.972 hash khác nhau (một cặp ảnh trùng nằm
    ngay trong BDD val);
  - UA-DETRAC test: **chỉ so tên video**, không so nội dung.
- **Hợp đồng `bdd5-dataset-v2`**: **0 lỗi** ở bước xuất và ở lần kiểm lại độc lập.
  - Đã băm 418 ảnh, 418 nhãn, 152 file ignore. Không có `.npy` cạnh ảnh.
  - `train` resolve về `images/train`, `val` về `images/dev`.
  - Bản sao split trong dataset khớp `b8a2848e…`.
  - Hash: `data.yaml` `9744ba88…`, manifest ảnh `c48998266…`.
- **Giới hạn của smoke**: dev smoke **không có** motorcycle và bicycle. AP của hai lớp này không xác định trong lần đánh giá
  smoke; chỉ nhánh "lớp không có GT" của evaluator được chạy.

## 4. Smoke lần 1 — FAIL ở công cụ đo, đã lưu log

Commit `f27895a`. Script `chay_smoke.sh`; log `07_*`, `08_*`; kết quả runner chép nguyên byte vào `08_ket_qua_runner_*.json`.

- **`--dry-run` exit 2** với cả hai model: `chua co dataset …\data\yolo_bdd5_v1\data.yaml`. Đây là thiết kế của runner: chế độ
  `kiem-cau-hinh` đọc **dataset đầy đủ**, mà vòng này chưa xuất. Lần 2 thêm `--data <dataset smoke>`; giá trị ghi đè được ghi
  vào kết quả.
- **Smoke exit 1** với cả hai model, sau 11–12 s, dừng ở `train: Fast image access` (trước vòng train: 0 epoch, 0 AP). Runner
  ghi `loi: "OSError: [Errno 22] Invalid argument"`.
- **Nguyên cơn** (stderr của tác vụ nền, `08_stderr_tac_vu_nen_lan1.txt`):
  1. `moc-gio` là bộ gắn mốc giờ cho stdout của **tôi**, nằm ngoài runner. Nó đọc stdin ở chế độ text theo locale (cp1252,
     `surrogateescape`).
  2. Thanh tiến độ ultralytics "━" (UTF-8 `E2 94 81`) thành `'\udc81'`. Ghi ra log UTF-8 thì gặp `UnicodeEncodeError`, và
     bộ đọc chết.
  3. Pipe đứt. Lần ghi stdout kế tiếp của runner trên Windows báo `OSError [Errno 22]`.
  4. Lệnh `print` cuối của runner cũng lỗi, không được bắt, nên exit là **1** chứ không phải mã 3 của runner.
- **Không phải** lỗi của runner, dữ liệu, RAM hay GPU.
- Thư mục lượt và kết quả lần 1 (`*_smoke`) giữ nguyên làm bằng chứng. Lần 2 dùng tên mới `*_smoke_lan2`, vì runner từ chối
  ghi đè.

## 5. Sửa công cụ đo + regression (ĐÃ CHẠY)

Commit `bf7c0a2`. Runner, cấu hình và dữ liệu không đổi.

- `do_tai_nguyen_ngoai.py moc-gio` sau khi sửa:
  - đọc stdin dạng **byte** (`os.read`), tách theo `\n` và `\r`;
  - giải mã UTF-8 với `errors="replace"`;
  - không bao giờ ngừng đọc giữa chừng: lỗi khi ghi một dòng chỉ được đếm, nên bên ghi không bao giờ bị đứt pipe.
- `test_do_tai_nguyen_ngoai.py` tự đặt môi trường cho tiến trình con:
  - E1: như smoke lần 1, không `PYTHONIOENCODING`, stdin là `cp1252 surrogateescape`;
  - E2: `PYTHONIOENCODING=utf-8`.

| Chạy | Kết quả |
|:--|:--|
| Mã `f27895a` (đã chạy ở smoke lần 1), `10b_*` | **FAIL 8/12**. E1 tái hiện đúng lỗi: bên ghi `OSError errno=22`, bộ đọc `UnicodeEncodeError '\udc81'` |
| Mã đã sửa, `10c_*` | **PASS 12/12** |
| `10a_*` | Lần chạy đầu để tiến trình con thừa kế `PYTHONIOENCODING`, nên không đúng điều kiện smoke. Giữ lại, có ghi chú |

## 6. Smoke lần 2 — PASS cả hai model (ĐÃ CHẠY)

Commit `4dcd311`. Script `chay_smoke_lan2.sh`, chạy **tuần tự**: YOLOv8n rồi YOLO11n.

```text
python src/train_detector.py --model <M> --dry-run --data data/yolo_bdd5_v1_smoke/data.yaml        # exit 0 ca hai
python -u src/train_detector.py --model <M> --che-do smoke --ten train_bdd5_v1_<M>_smoke_lan2 2>&1 \
  | python handoff/logs/review35/do_tai_nguyen_ngoai.py moc-gio --ra handoff/logs/review35/12_smoke_<M>_lan2.log
```

**Tham số hiệu lực** (ghi trong kết quả runner, giống nhau cho hai model):

- epochs 1, batch 8 (không `auto`), imgsz 640, workers 2, seed 0, `deterministic`;
- AMP tắt, `cache: false`, `cls_remap` (log: "Remapped 5/5 cls head rows");
- SGD nesterov, lr 0,01, accumulate 8, 42 batch/epoch.

**Cách đo**:

- **Khởi tạo**: mốc `moc-gio` từ lúc bắt đầu tới dòng `Starting training`.
- **`train_s`**: callback runner, từ `on_train_start` tới `on_train_epoch_end`; gồm thời gian chờ batch đầu.
- **s/batch**: trung vị khoảng giữa các batch, bỏ 5 batch đầu.
- **`val_s`**: từ `on_train_epoch_end` tới `on_fit_epoch_end`; gồm val, lưu checkpoint, dọn dẹp.
- **wall**: `date` quanh pipeline.
- **RAM**:
  - bộ đo ngoài: psutil RSS (working set) của tiến trình train + mọi tiến trình con, mỗi 1 s. Tổng có thể đếm trùng trang chia
    sẻ giữa các tiến trình, nên có thể cao hơn bộ nhớ thực dùng (đính chính câu chữ vòng 36);
  - RAM hệ thống còn trống: psutil `available`.
- **VRAM**:
  - torch `max_memory_allocated` / `max_memory_reserved` trong pha train, riêng tiến trình;
  - `nvidia-smi memory.used` của **cả GPU**, gồm desktop: ~415–422 MiB trước khi chạy.

| | YOLOv8n | YOLO11n |
|:--|--:|--:|
| Checkpoint khởi tạo (COCO 80 lớp) | `models/yolov8n.pt` `f59b3d83…` | `models/candidates/yolo11n.pt` `0ebbc80d…` |
| Exit runner / `moc-gio` | **0 / 0** | **0 / 0** |
| Ảnh/nhãn ultralytics thực nạp (train/dev) | 334/84 | 334/84 |
| Thiếu / thừa / nhãn khác / `.npy` | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| Khởi tạo | 40,0 s | 36,4 s |
| `train_s` (42 batch) | 50,86 s | 63,2 s |
| s/batch ổn định (ảnh/s) | **0,991** (8,07) | **1,131** (7,07) |
| `val_s` | 33,51 s | 8,76 s |
| `model.train` / wall | 124,2 s / 134 s | 110,6 s / 121 s |
| torch allocated / reserved | 1.695 / 2.134 MiB | 1.941 / 2.344 MiB |
| nvidia-smi max cả GPU | 2.624 MiB | 2.840 MiB |
| RSS train + con: max (trung vị) | 3.018 (1.469) MiB, quan sát nhiều nhất 7 con | 2.828 (1.693) MiB, quan sát nhiều nhất 8 con |
| RAM hệ thống còn trống thấp nhất (bộ đo ngoài / runner) | **16** / 51 MiB | 82 / 146 MiB |
| GPU util trung vị cả lượt (gồm khởi tạo, val) | 16% | 13% |
| Loss box / cls / dfl | 1,622 / 3,683 / 1,221 | 1,650 / 4,284 / 1,239 |
| `last.pt` (sha256, byte) | `6f0d71dd…`, 6.220.707 | `31f21a9a…`, 5.443.219 |
| `best.pt` (sha256) | `31a9ff9e…` | `e1162246…` |
| Nạp lại: đúng 5 lớp / `train_args` lệch cấu hình | ✓ / không | ✓ / không |

**AP theo lớp trên dev smoke** (84 ảnh). Evaluator `eval_detector_5lop`, cùng quy tắc: dự đoán conf 0,001, NMS IoU 0,7,
max_det 300; vận hành conf 0,25, IoU 0,5. Bảng **chỉ để kiểm pipeline, không so sánh model**.

| Lớp | GT | v8n AP50 | v8n AP50–95 | 11n AP50 | 11n AP50–95 |
|:--|--:|--:|--:|--:|--:|
| car | 542 | 0,293 | 0,184 | 0 | 0 |
| truck | 6 | 0 | 0 | 0 | 0 |
| bus | 4 | 0 | 0 | 0 | 0 |
| motorcycle, bicycle | 0 | — | — | — | — |
| mAP (3 lớp có GT) | | 0,098 | 0,061 | 0 | 0 |

Ghi chú:

- RAM là tài nguyên căng nhất. Các ứng dụng khác chiếm ~3,9 GB (VS Code 1,5 GB, Chrome 0,9 GB, Java 0,75 GB, ChatGPT 0,7 GB).
- `val_s` của v8n (33,5 s) gồm khoảng 23 s từ lúc in kết quả val tới hết epoch. Khoảng này chưa đo tách được; có thể do lưu
  checkpoint và dọn dẹp trong lúc RAM bị phân trang.

## 7. Chẩn đoán YOLO11n 0 detection (ĐÃ CHẠY, chỉ đọc) — diễn giải đã đính chính ở vòng 36 (R35-F1)

Commit `2ab107b`. Không train, không đổi tham số. Đối chiếu lại ở vòng 36: `logs/review36/04_doi_chieu_R35-F1.log`.

**Đã quan sát** (`13_*`, số liệu giữ nguyên):

- Phép so là **checkpoint COCO 80 lớp** (file khởi tạo; lấy các hàng lớp cùng tên với 5 lớp) với **checkpoint smoke 5 lớp**
  (`best.pt`).
- Bias lớp hai bên gần nhau: car P3 −9,633 so với −9,609 (v8n); −10,117 so với −10,094 (11n).
- Điểm lớp lớn nhất mỗi ảnh (trung vị) của checkpoint smoke **thấp hơn** nhiều:

  | Model | COCO 80 lớp | Smoke 5 lớp |
  |:--|:--|:--|
  | v8n | 0,82 | 0,013 |
  | 11n | 0,80 | 0,0003 (max 0,0005) |

- Điểm của 11n nằm dưới ngưỡng 0,001 của evaluator, nên ra 0 detection.

**Chưa đo**:

- Model **5 lớp sau khi khởi tạo/nạp trọng số nhưng trước bước tối ưu đầu tiên**. Log khởi tạo ghi
  `Overriding model.yaml nc=80 with nc=5`, "Remapped 5/5 cls head rows", và chỉ chuyển **322/355** (v8n), **451/499** (11n)
  tensor. Phép so trên vì vậy gộp ảnh hưởng của bước chuyển sang 5 lớp với ảnh hưởng của các bước học.
- **Warmup/LR/accumulate thực chạy theo từng bước**: runner chỉ chụp LR lúc `on_train_start` và cuối epoch (0,01). Nhận
  định "ultralytics 8.4.141 không warmup khi `epochs=1`" chỉ dựa trên **đọc mã nguồn**: hàm `_get_warmup_iterations`,
  `trainer.py` sha256 `7532ff0b…`, ghi ở `logs/review36/00_*`. Đây không phải log thực chạy.

**Kết luận đã đính chính**: checkpoint smoke có confidence thấp hơn checkpoint COCO 80 lớp; **nguyên nhân chưa xác định**.
Chuyển sang 5 lớp/khởi tạo và các bước học (gồm có hay không warmup) là các yếu tố **cần phân biệt**, chưa được tách. Bản
vòng 35 viết "smoke 1 epoch không warmup làm điểm lớp giảm mạnh" là **vượt bằng chứng** và được rút lại.

**Chẩn đoán conf 1e-6** (`13b_*`, riêng):

- Chạy chính evaluator với conf 1e-6. YOLO11n có 25.199 detection; **car AP50 ≈ 0,233** trên 84 ảnh dev smoke.
- Truck và bus AP50 = 0 (11n: 1 và 0 dự đoán). Motorcycle và bicycle không có GT.
- Điều này chỉ chứng minh **có dự đoán car khớp GT ở điểm thấp** trên tập đã kiểm. Câu "ánh xạ lớp và tọa độ đúng; không
  phải lỗi đường ống" ở bản vòng 35 là **suy rộng quá mức** và được rút lại.
- JSON 13b ban đầu có `NaN`; đã ghi lại thành `null`, có ghi chú trong log.

**Đánh giá chính giữ nguyên** ở conf 0,001: YOLO11n AP = 0 cho mọi lớp có GT. Dev smoke không có motorcycle hay bicycle.

**Đề xuất chờ duyệt** (chưa làm): nếu cần chẩn đoán nguyên nhân, đo model 5 lớp trước bước học và lưu số warmup
iteration/LR thực tế từng bước. Chỉ tăng epoch không tạo được phép đối chứng nguyên nhân sạch.

## 8. Đếm dataset đầy đủ (ĐÃ CHẠY, không xuất)

Commit `d782e67`. `14_dem_anh_day_du.py` dùng đúng `doc_split`, `chon_tap_con`, `chon_frame` và `vai_tro` của bộ xuất, nên số
ảnh là số bộ xuất sẽ chọn. Kiểm chéo cùng phương pháp trên tập con smoke: **khớp hoàn toàn** số ảnh, âm tính, tô xám và box
theo lớp của dataset đã xuất.

| Bước frame | train-fit | dev | Byte nguồn | Ảnh có motorcycle / bicycle (train-fit, xấp xỉ) |
|:--|--:|--:|--:|:--|
| **5** (cấu hình) | **50.487** | **5.559** | 6,85 GB | 1.822 / 3.598 |
| 10 | 25.848 | 2.848 | 3,52 GB | 947 / 1.852 |

Ước tính quy mô trước đây (~50.400 / ~5.600) gần đúng.

## 9. Dự toán training local (ƯỚC TÍNH)

Commit `d782e67`. Script `15_du_toan_training.py`; mọi giả định ghi trong `15_du_toan_training.json`.

**Mô hình**, tính cho từng model chạy tuần tự:

```text
tổng(E) = khởi tạo + chờ batch đầu + E × (train/epoch + val/epoch) + val cuối
```

- **train/epoch** = ⌈ảnh train / 8⌉ × s/batch ổn định đo ở smoke.
- **val/epoch** = `val_s` của smoke + (ảnh dev − 84) × ms/ảnh từ dòng `Speed:` của ultralytics.
- **Khởi tạo** cho dataset đầy đủ cộng thêm:
  - băm hợp đồng: 42 MB/s, đo ở smoke;
  - quét nhãn: ~1.100 ảnh/s.
- **Kịch bản chậm hơn** giả định thêm 5 ms/ảnh hậu xử lý val (model đã học có thể có nhiều box hơn), rồi nhân tất cả với
  **1,3** cho rủi ro chưa đo: GPU nóng hoặc giảm xung khi chạy nhiều giờ, phân trang RAM.
- **Hai mức là kịch bản dự toán theo các giả định trên, không phải cận trên hay bảo đảm** (đính chính vòng 36, R35-F2). Thời
  gian thực tế có thể vượt cả kịch bản chậm hơn. Số liệu không đổi: đối chiếu 162 giá trị số ở
  `logs/review36/03_doi_chieu_so_du_toan.log`.

| Phương án | YOLOv8n | YOLO11n | Tổng tuần tự |
|:--|--:|--:|--:|
| **Bước 5, 15 epoch** (đề xuất, **chưa duyệt**) | 1,74 h/epoch train + 1,3–1,7 ph val → **26,5 h** (chậm hơn 34,6 h) | 1,98 h/epoch + 1,2–1,7 ph → **30,2 h** (chậm hơn 39,4 h) | **56,64 h ≈ 2,4 ngày** (chậm hơn 73,95 h) |
| Bước 5, 30 epoch | 52,9 h (chậm hơn 69,0 h) | 60,2 h (chậm hơn 78,6 h) | 113,08 h (chậm hơn 147,61 h) |
| Bước 10, 15 epoch | 13,6 h (chậm hơn 17,8 h) | 15,5 h (chậm hơn 20,2 h) | 29,10 h (chậm hơn 37,99 h) |
| Bước 10, 30 epoch | 27,2 h (chậm hơn 35,5 h) | 30,9 h (chậm hơn 40,3 h) | 58,08 h (chậm hơn 75,81 h) |

- Mỗi epoch ở bước 5 có 6.311 batch. Khởi tạo dataset đầy đủ ~4,9 phút mỗi lượt.
- Bảng trên **không** tính dừng sớm (`patience 5` có thể rút ngắn). Cũng không tính thời gian xuất dataset đầy đủ: ước thô
  15–30 phút, chưa đo.
- s/batch thật gần với fixture vòng 27–28 (v8n 0,991 so với 0,95–0,97; 11n 1,131 so với 1,14–1,16). Ước tính cũ ~59 h; ước
  tính mới 56,6 h.

**RAM, VRAM, đĩa** (kịch bản ước tính từ smoke cho dataset đầy đủ, bước 5; **không phải bảo đảm**, thực tế có thể vượt):

| | YOLOv8n | YOLO11n |
|:--|--:|--:|
| RSS train + con: đo ở smoke → kịch bản | 3.018 → 3.118–3.518 MiB | 2.828 → 2.928–3.328 MiB |
| torch reserved: đo ở smoke → kịch bản | 2.134 → 2.134–2.434 MiB | 2.344 → 2.344–2.644 MiB |
| Cả GPU: đo ở smoke → kịch bản (card 4.096 MiB) | 2.624 → 2.624–2.924 MiB | 2.840 → 2.840–3.140 MiB |

- **RAM tăng thêm**: nhãn của ~56K ảnh nằm trong tiến trình chính và trong từng worker (Windows dùng spawn). Giả định cộng
  100–500 MiB. **Chưa đo.**
- **VRAM tăng thêm**: bộ gán nhãn TAL cấp phát theo số box lớn nhất trong lô. Giả định cộng 0–300 MiB; lô nhiều box nhất quan
  sát ở smoke ~180 box. **Chưa đo** trên dữ liệu đầy đủ. Lô thực tế có thể nhiều box hơn, và VRAM (kể cả OOM) có thể vượt
  kịch bản.
- **Đĩa**: dataset bước 5 ~**8,5 GB** (bước 10: ~4,3 GB).
  - Hiệu chỉnh trên smoke: ảnh tô xám ghi lại JPEG q95 nặng **1,55×** nguồn; ảnh khác chép nguyên byte.
  - Cộng file nhỏ theo cấp phát NTFS.
  - Hai lượt train ~0,1 GB. D: còn trống ~63 GB.
- **Rủi ro lớn nhất là RAM**: lúc smoke, RAM còn trống chỉ 16–146 MiB vì ứng dụng khác. Nên đóng bớt trước khi chạy dài. Hành
  vi trong nhiều giờ khi đang phân trang **chưa đo**.

## 10. Cần người dùng quyết định

1. **Ngân sách training dài**: số epoch (đề xuất 15) và bước frame (5) → kịch bản ~56,6 h GPU cho hai model (kịch bản chậm
   hơn ~74 h; không phải bảo đảm); phương án khác ở mục 9. **Chưa chạy, chưa bật cờ.**
2. **Xuất dataset đầy đủ** (bước 5, ~8,5 GB) là điều kiện trước khi train dài. Vòng này không làm.
3. **Quan sát từ smoke** (mục 7): checkpoint smoke có confidence thấp hơn checkpoint COCO 80 lớp; **nguyên nhân chưa xác
   định** (đính chính vòng 36).
   - **Pilot thêm epoch**: chưa duyệt, chưa làm. Chỉ tăng epoch không tách được nguyên nhân.
   - **Chẩn đoán nguyên nhân**, nếu cần: đo model 5 lớp trước bước học và lưu warmup/LR thực tế. Đây là đề xuất chờ duyệt,
     chưa làm.
4. **RAM**: đóng bớt ứng dụng (~3,9 GB) trong lúc train dài.
5. **Smoke chưa thử AP lớp hiếm**: dev smoke không có motorcycle hay bicycle, nên AP hai lớp này chỉ xác định được trên dev đầy đủ.

## 11. Chưa chạy, giới hạn

- **NOT_RUN**:
  - xuất dataset đầy đủ (không yêu cầu vòng này);
  - training dài (chưa duyệt);
  - tracking và Nano (ngoài phạm vi).
- **Test suite NOT_RUN** vì `src/` không đổi. Mã mới chỉ là công cụ đo trong `logs/review35/`, có regression riêng (12/12).
- Thời gian, RAM và VRAM ở mục 9 là **kịch bản dự toán theo giả định**, ngoại suy từ một epoch 42 batch. Đây không phải bảo
  đảm; thực tế có thể vượt.
- Các số đo trong vòng chịu ảnh hưởng của RAM thấp và ứng dụng khác đang mở.
- Nhãn vẫn là snapshot bên thứ ba (phương án C); chưa xác thực bằng archive gốc.

## 12. Commit vòng 35

| Commit | Nội dung |
|:--|:--|
| `5cac59b` | Ghi nhận nghiệm thu `ccd801f`, base → `ccd801f` |
| `4237149` | Khóa split `bdd5_train_v1` |
| `aeed110` | Giữ nguyên byte file split (CRLF) và log vòng 35; nội dung không đổi |
| `10c3ef1` | Log xuất và kiểm tập con smoke |
| `f27895a` | Smoke lần 1 FAIL: log, công cụ đo đúng như đã chạy |
| `bf7c0a2` | Sửa `moc-gio` + regression (mã cũ FAIL, mã mới 12/12) |
| `4dcd311` | Smoke lần 2 PASS cả hai model |
| `2ab107b` | Chẩn đoán YOLO11n 0 detection |
| `d782e67` | Đếm dataset đầy đủ, dự toán training local |
