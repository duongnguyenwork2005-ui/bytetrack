# Vòng 37 — pilot chẩn đoán 2 epoch (YOLOv8n, YOLO11n) trên dataset smoke (chi tiết)

**Phạm vi** (người dùng duyệt):

- YOLOv8n và YOLO11n, mỗi model **một lượt 2 epoch**, chạy lần lượt trên GPU PC, đúng dataset smoke đã khóa (334 / 84 ảnh).
- Đo model 5 lớp **trước bước học** và sau từng epoch.
- Không training dài, không xuất dataset đầy đủ, không thêm model / seed / cấu hình, không Nano.
- Base review vẫn là `2351c67`; vòng 36 (`a22a6f5`) chưa được nghiệm thu.

## 0. Trạng thái đầu vòng

Log: `logs/review37/00_trang_thai_dau.log`.

- HEAD `a22a6f5`, 0 commit sau đó. Working tree chỉ có file riêng của người dùng.
- ultralytics 8.4.141, torch 2.6.0+cu124, GTX 1650 4 GB.
- Lúc đầu vòng RAM còn trống 0,94 GB; ổ D: còn 63,4 GB.

## 1. Chuẩn bị đo đạc (ĐÃ CHẠY)

Commit `75fe5fa`.

- **`src/train_detector.py`**:
  - chế độ `pilot` (bắt buộc `--xac-nhan-pilot`) và `kiem-pilot` (không train);
  - cấu hình pilot **kế thừa nguyên** cấu hình gốc (sha256 nội dung LF) và **chỉ được đổi `epochs`**; runner so tham số hiệu
    lực với smoke, khác ở khóa nào khác thì từ chối;
  - `DoVram` giữ mẫu thô có mốc thời gian.
- **Sửa lỗi đo đã có từ trước**: ultralytics `final_eval` gọi lại `on_fit_epoch_end` sau vòng lặp (validate `best.pt`).
  - Runner cũ ghi đè `val_s` và `metrics` của epoch cuối.
  - Hệ quả cho smoke vòng 35: `val_s` (33,51 s / 8,76 s) **gồm cả lưu checkpoint và `final_eval`**. Số đo không đổi, chỉ đổi
    cách hiểu (mục 7).
  - Nay lần gọi đó được nhận ra và ghi riêng.
- **`src/pilot_chan_doan.py`**:
  - điểm đo; nguồn đo; quy trình đo bằng nguyên evaluator ở conf 0,001;
  - điểm trước ngưỡng (chẩn đoán riêng);
  - bằng chứng nạp/remap đối chiếu độc lập checkpoint COCO;
  - chụp và so trạng thái huấn luyện trước/sau mỗi phép đo, khôi phục RNG và cờ, còn khác thì dừng;
  - log theo batch.
- **Test**:

  | Lần | Kết quả | Ghi chú |
  |:--|:--|:--|
  | Lần 1 | **24/30** | Lỗi của phép đo: đọc `predictor.args.half`, trong khi ultralytics 8.4.141 đã đổi khóa này thành `quantize`. Nhờ vậy cũng thấy lỗi đo làm dừng train đúng thiết kế. Đã sửa; log giữ ở `02_*` |
  | Lần 2 | **34/34 PASS** (`03_*`) | Chi tiết bên dưới |

  Lần 2 gồm:
  - cấu hình / CLI;
  - `final_eval` không ghi đè (mã cũ `a22a6f5` tái hiện lỗi);
  - phát hiện thay đổi trạng thái;
  - lượt pilot thật trên fixture (GPU): điểm đo đúng trước cập nhật đầu; LR/momentum/accumulate khớp công thức; batch có cập
    nhật khớp mô phỏng; `train_s` không gồm phép đo;
  - **A/B trên CPU**: hai lượt không đo trùng nhau từng bit, và lượt **có đo** trùng chúng từng bit (loss từng batch + mọi
    tensor `last.pt`).
- **Hồi quy**: `test_detector_5lop.py` **92/92 PASS** (`04_*`).
- `test_pilot_chan_doan.py` đã đăng ký vào `run_all_tests.py`.

## 2. Khóa cấu hình trước khi train (ĐÃ CHẠY)

Commit `6f2ca14`; chi tiết ở `logs/review37/01_ke_hoach_pilot_da_khoa.md`.

- `configs/phase2/detector/train_bdd5_pilot.yaml` (sha256 `5d195826…`): kế thừa `train_bdd5.yaml` (`a466a4fa…`),
  `pilot.epochs: 2`.
- `kiem-pilot` (không train) cho hai file khóa `01_khoa_cau_hinh_pilot_<M>.json`:
  - khác smoke **chỉ** `{epochs: [1, 2]}`;
  - checkpoint `f59b3d83…` / `0ebbc80d…`;
  - dataset smoke 334/84 (manifest `c4899826…`, split `b8a2848e…`).
- **Dự đoán theo mã** (gọi chính hàm của ultralytics đang cài):
  - 42 batch/epoch, **nw = 42**: warmup phủ trọn epoch 1;
  - mosaic tắt từ epoch index 0, nên **cả hai epoch không mosaic**. Smoke 1 epoch thì có mosaic: giá trị cấu hình giống
    nhưng augmentation hiệu lực khác.
- `chay_pilot.sh` chạy `kiem-pilot` lại ngay trước mỗi lượt và so khớp với khóa (`so_khop_khoa.py`, đã tự kiểm 3 ca). Kết
  quả: **khớp 46/46** cho cả hai model (`05_*`).

## 3. Chạy pilot (ĐÃ CHẠY)

Commit `0f7b63b`.

```text
bash handoff/logs/review37/chay_pilot.sh      # tuan tu YOLOv8n -> YOLO11n
python -u src/train_detector.py --config configs/phase2/detector/train_bdd5_pilot.yaml --model <M> --che-do pilot --xac-nhan-pilot
```

| | YOLOv8n | YOLO11n |
|:--|:--|:--|
| Exit runner / `moc-gio` | **0 / 0** | **0 / 0** |
| Wall / `model.train` | 175 s / 159,1 s | 177 s / 161,7 s |
| Ảnh/nhãn thực nạp | 334/84; 0 thiếu/thừa/nhãn khác/`.npy` | như v8n |
| Tiêu chí hoàn chỉnh | **ĐẠT**: đủ 2 epoch, có cập nhật mỗi epoch, loss hữu hạn mọi batch, đủ 7 phép đo, bất biến | **ĐẠT** |
| Checkpoint (nạp lại 5 lớp, `train_args` khớp) | `last` `40f7bf47…`, `best` `c7d07c37…` (6.220.899 B) | `last` `0e3c1f52…`, `best` `e4878084…` (5.443.411 B) |

`best.pt` rơi vào epoch 2 ở cả hai model; nội dung trọng số trùng `last.pt`, phép đo hai file cho cùng kết quả.

## 4. Đầu ra model theo điểm đo (cùng 84 ảnh dev smoke)

**Quy trình đo**, giống nhau mọi điểm:

- nguyên `eval_detector_5lop`: imgsz 640 letterbox (ảnh vào 640×384), **conf 0,001**, NMS IoU 0,7 theo lớp, max_det 300, FP32;
- AP chỉ tính trên lớp có GT (car 542, truck 6, bus 4); **motorcycle/bicycle không có GT nên không đánh giá được**.

**"Điểm trước ngưỡng"** là chẩn đoán riêng, không phải AP: sigmoid điểm lớp lớn nhất trên mọi anchor của đầu ra thô, trước
NMS và ngưỡng, cùng predictor.

**Nguồn đo**:

| Nhãn | Nguồn |
|:--|:--|
| trước học | model 5 lớp trainer đã dựng, trước bước cập nhật đầu; trùng EMA |
| model | trọng số hiện tại, FP32 trong bộ nhớ |
| EMA | bản ultralytics dùng để validate và ghi checkpoint, FP32 trong bộ nhớ |
| ckpt | `last.pt` / `best.pt`, EMA FP16 nạp từ đĩa |

**YOLOv8n**

| Điểm đo | Detection ≥ 0,001 (ảnh có) | car AP50 / AP50–95 | mAP50 (3 lớp) | Điểm trước ngưỡng: trung vị max/ảnh |
|:--|--:|--:|--:|--:|
| Trước học | **0** (0) | 0 / 0 | 0 | **0,000185** |
| Sau epoch 1, model | 631 (77) | 0,375 / 0,204 | 0,125 | 0,0124 |
| Sau epoch 1, EMA | 554 (76) | 0,338 / 0,184 | 0,113 | 0,0079 |
| Sau epoch 2, model | 7.496 (84) | 0,510 / 0,288 | 0,170 | 0,109 |
| Sau epoch 2, EMA | 6.031 (84) | 0,498 / 0,278 | 0,166 | 0,070 |
| `last.pt` = `best.pt` | 6.037 (84) | 0,498 / 0,279 | 0,166 | 0,071 |

**YOLO11n**

| Điểm đo | Detection ≥ 0,001 (ảnh có) | car AP50 / AP50–95 | mAP50 (3 lớp) | Điểm trước ngưỡng: trung vị max/ảnh |
|:--|--:|--:|--:|--:|
| Trước học | **0** (0) | 0 / 0 | 0 | **0,000278** |
| Sau epoch 1, model | 18 (11) | 0,005 / 0,001 | 0,002 | 0,00055 |
| Sau epoch 1, EMA | 3 (3) | 0,005 / 0,000 | 0,002 | 0,00041 |
| Sau epoch 2, model | 3.798 (84) | 0,413 / 0,209 | 0,138 | 0,055 |
| Sau epoch 2, EMA | 3.606 (84) | 0,392 / 0,195 | 0,131 | 0,041 |
| `last.pt` = `best.pt` | 3.603 (84) | 0,393 / 0,195 | 0,131 | 0,041 |

- Truck và bus AP50 = 0 ở mọi điểm đo (6 và 4 GT): quá ít để diễn giải.
- Trước học, không ảnh nào có điểm lớp ≥ 0,001 ở cả hai model. Trung vị theo lớp đều trong khoảng 6e-5 đến 3e-4.

**Bằng chứng khởi tạo** (trước học; đối chiếu độc lập với checkpoint COCO, khớp dòng `Transferred` của ultralytics):

- Lớp `{0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle}`, lấy từ hàng COCO 2, 7, 5, 3, 1.
- Optimizer chưa có trạng thái, `ema.updates` 0, không gradient, model trùng EMA.

| | Chép nguyên | Remap theo tên lớp | Khởi tạo mới | Log ultralytics |
|:--|--:|--:|--:|:--|
| YOLOv8n | 319 | 3 (bias lớp cuối cv3 ×3) | **33** (nhánh cv3 của Detect) | `Transferred 322/355` |
| YOLO11n | 448 | 3 | **48** | `Transferred 451/499` |

Độ rộng ẩn của nhánh phân lớp phụ thuộc số lớp (80 → 5), nên các conv trong `cv3` và trọng số conv lớp cuối lệch shape, không
chép được; chỉ bias được remap.

## 5. Warmup, learning rate và cập nhật trọng số thực chạy

Giống hệt nhau ở hai model; đối chiếu bằng `07_tom_tat_pilot.py`.

- **nw theo mã = 42**, cả epoch 1. Từng batch của epoch 1 khớp công thức `BaseTrainer._do_train` (**0 batch lệch**):
  - LR nhóm weight/bn: 0 → 0,00976;
  - LR nhóm bias: 0,1 → 0,01214;
  - momentum: 0,8 → 0,934;
  - accumulate: 1 → 8.
- **Epoch 2**: LR = **0,00505** cho mọi nhóm, bằng 0,01 × lf(1); accumulate 8.
- **Cập nhật optimizer** (`ema.updates` tăng): epoch 1 có **12** lần (ni 0, 1, 2, 4, 6, 8, 11, 14, 18, 23, 29, 36), epoch 2 có
  **5** lần (ni 44, 52, 60, 68, 76). Khớp mô phỏng điều kiện `ni − last_opt_step ≥ accumulate`.
- **Mosaic thực tế**: tắt ở cả hai epoch ("Closing dataloader mosaic" ngay epoch đầu), đúng dự đoán.
- **Loss trung bình** epoch 1 → 2:

  | | box | cls | dfl |
  |:--|:--|:--|:--|
  | v8n | 1,59 → 1,53 | 3,66 → 1,95 | 1,17 → 1,14 |
  | 11n | 1,58 → 1,69 | 4,17 → 2,72 | 1,21 → 1,33 |

- Mọi batch có loss hữu hạn.

## 6. Thời gian, RAM, VRAM

**Cách lấy mẫu**:

- mốc callback trong runner;
- `DoVram` ~1 s: `nvidia-smi` cả GPU + psutil `available`;
- bộ đo ngoài ~1 s: RSS tiến trình runner và tổng với tiến trình con; tổng có thể đếm trùng trang chia sẻ.

Mẫu thô ở `06_tai_nguyen_nvidia_smi_*.csv` và `06_ram_ngoai_*_mau_tho.csv`.

| | YOLOv8n | YOLO11n |
|:--|--:|--:|
| Đo trước học | 15,2 s | 8,3 s |
| Train epoch 1 / 2 | 50,1 / 39,0 s | 58,0 / 46,4 s |
| s/batch ổn định epoch 1 / 2 (smoke vòng 35) | 0,930 / 0,931 (0,991) | 1,106 / 1,108 (1,131) |
| Val trong vòng epoch 1 / 2; từ hết val tới lưu xong checkpoint epoch 1 / 2 | 1,1 / 0,9 s; 1,0 / 0,3 s | 1,0 / 1,0 s; 2,1 / 0,7 s |
| Đo sau epoch 1 / 2 (model + EMA) | 14,1 / 9,9 s | 11,5 / 12,4 s |
| `final_eval` (validate `best.pt`) | 0,8 s | 1,0 s |
| Đo checkpoint `last` / `best` | 3,3 / 3,2 s | 3,2 / 3,1 s |
| torch max allocated / reserved (mỗi epoch) | 1.691–1.693 / 2.138–2.164 MiB | 1.918–1.932 / 2.352–2.386 MiB |
| nvidia-smi cả GPU: max (trung vị) | 2.655 (2.621) MiB | 2.892 (2.846) MiB |
| RSS runner max; runner + con max (quan sát nhiều nhất 7 con) | 809; 3.084 MiB | 945; 3.064 MiB |
| RAM hệ thống còn trống thấp nhất (bộ đo ngoài / runner) | **19** / 59 MiB | **7** / 35 MiB |
| GPU util trung vị cả lượt | 56% | 94% |

- RAM rất căng vì các ứng dụng khác đang mở (không đóng giúp người dùng). Pilot vẫn hoàn tất.
- Mỗi phép đo trong lúc train đều làm thay đổi **trạng thái RNG CUDA**; trạng thái này đã được khôi phục.

## 7. Kết luận và câu hỏi còn mở

**Có bằng chứng** (trong điều kiện pilot):

1. Đường chạy pilot hoàn chỉnh trên dữ liệu thật:
   - đầu vào đúng, 2 epoch, 17 lần cập nhật/model, loss hữu hạn, checkpoint đọc lại đúng 5 lớp;
   - phép đo không đổi trạng thái: kiểm hash từng lần, và A/B CPU trên fixture trùng từng bit.
2. **Model 5 lớp trước bước học đã có điểm lớp gần 0**: 0 detection ≥ 0,001; điểm max/ảnh trung vị 1,9e-4 (v8n) và 2,8e-4
   (11n). Checkpoint COCO 80 lớp ở vòng 35 là 0,82 / 0,80. Mức gần 0 xuất hiện ở bước chuyển 80 → 5 lớp, **trước** mọi
   bước tối ưu.
3. Bước chuyển đó khởi tạo mới 33 / 48 tensor của nhánh phân lớp; chỉ 3 bias lớp được remap theo tên.
4. Trong pilot, các bước học **làm điểm lớp và AP tăng**, không kéo xuống: car AP50 (model) 0 → 0,375 → 0,510 (v8n) và
   0 → 0,005 → 0,413 (11n).
5. Warmup thực chạy đúng mã: trọn epoch 1, từng batch khớp công thức. Mosaic tắt cả hai epoch như dự đoán.

**Với R35-F1 (thu hẹp, chưa đóng)**:

- Phép so "COCO 80 lớp → checkpoint smoke" ở vòng 35 gộp bước chuyển sang 5 lớp. Pilot cho thấy điểm gần 0 đã có trước khi
  học.
- **Chưa xác định**:
  - trạng thái trước học của **chính lượt smoke vòng 35**: không được đo; cùng seed và mã nên có thể giống, chưa kiểm;
  - ảnh hưởng của **không warmup** (smoke) so với có warmup (pilot) đến tốc độ điểm tăng: không có đối chứng;
  - ảnh hưởng của mosaic: pilot không có mosaic;
  - vì sao YOLO11n tăng chậm hơn ở epoch 1. Không xếp hạng model;
  - AP truck/bus (quá ít GT) và motorcycle/bicycle (không có GT);
  - 2 epoch trên 334 ảnh không cho biết hành vi training dài.

**Đính chính cách hiểu số vòng 35** (do lỗi `final_eval` ở mục 1):

- `val_s` của smoke gồm cả lưu checkpoint và validate `best.pt`.
- Khoảng ~23 s giữa kết quả val và hết epoch ở smoke v8n không tái hiện trong pilot: val + lưu chỉ ~1–2 s mỗi epoch. Nguyên
  nhân chưa xác định; có thể liên quan áp lực RAM lúc smoke.
- Dự toán vòng 35 dùng `val_s` smoke làm chi phí val cố định mỗi epoch. Chi phí đó vì vậy có thể bị tính dư, nhưng nhỏ so
  với ~1,7–2 h train/epoch. Dự toán chưa tính lại.

## 8. Đề xuất bước tiếp theo (CHƯA làm, chờ duyệt)

1. **Quyết định của người dùng**: xuất dataset đầy đủ, bước frame, số epoch, ngân sách training local.
   - s/batch pilot gần smoke (0,93 / 1,11 so với 0,99 / 1,13), nên kịch bản dự toán vòng 35 không đổi đáng kể.
2. **Khác biệt cần biết nếu train 15 epoch**:
   - warmup khi đó kéo dài 1 epoch ≈ 6.311 batch;
   - mosaic bật trừ 2 epoch cuối.

   Pilot không thử chế độ có mosaic qua nhiều epoch.
3. **Đóng hẳn câu hỏi warmup trong R35-F1** cần đối chứng có kiểm soát (cùng seed, có / không warmup). Đây là ablation,
   **chỉ làm khi được duyệt**. Pilot đã cho thấy điểm gần 0 có từ trước học, nên phép đối chứng này không bắt buộc để quyết
   định training dài.
4. **RAM**: RAM trống tối thiểu chỉ 7–59 MiB. Nên đóng bớt ứng dụng trước khi train dài.
5. Có thể tính lại chi phí val/epoch của dự toán bằng số pilot khi người dùng duyệt bước tiếp.

## 9. Chưa chạy, giới hạn

- **NOT_RUN**:
  - training dài, xuất dataset đầy đủ, thêm epoch / lượt;
  - ablation warmup hoặc mosaic;
  - full suite (`run_all_tests.py`): chỉ chạy test liên quan (pilot 34/34, detector 92/92);
  - Nano.
- Pilot không dùng để so sánh hay chọn model.
- Thư mục lượt pilot (weights, 5 bản sao chẩn đoán FP32 ~12 MB mỗi file) nằm ở `models/finetune_phase2/*_pilot/`, không commit.
- Lượt smoke vòng 35 và kết quả của nó giữ nguyên.
- Nhãn vẫn là snapshot bên thứ ba (phương án C).

## 10. Commit vòng 37

| Commit | Nội dung |
|:--|:--|
| `75fe5fa` | Runner chế độ pilot / kiem-pilot, đo chẩn đoán, sửa `final_eval`; test (lần 1 24/30, lần 2 34/34), hồi quy 92/92 |
| `6f2ca14` | Khóa cấu hình pilot trước khi train (cấu hình, khóa, kế hoạch, script chạy, so khớp khóa) |
| `0f7b63b` | Kết quả pilot hai model, tóm tắt chỉ đọc |
| commit chứa phụ lục này (target của delta) | Brief, phụ lục này, kế hoạch training, `PLAN_DA_CHOT`, `CODEX_HANDOFF`, đính chính `val_s` ở phụ lục vòng 35 |
