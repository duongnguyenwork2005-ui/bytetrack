# Kế hoạch huấn luyện detector 5 lớp (YOLOv8n, YOLO11n) — bản để review

Trạng thái: **đề xuất, chưa chạy training dài.** Vòng 27 chỉ làm chuẩn bị và smoke. Phạm vi chung: `handoff/PLAN_DA_CHOT.md`.
Cập nhật vòng 28 (review `4937c85`): thêm hợp đồng kiểm dataset dùng chung (mục 1) và sửa diễn giải về kiểm trùng và
batch 16 (mục 1, 4). Số epoch, bước frame và ngân sách **chưa được duyệt** và không đổi.

Tài liệu này dùng ba nhãn nguồn số liệu:

- **fixture**: đo thật trên GPU của PC, nhưng với dữ liệu tổng hợp;
- **ước tính**: suy ra từ số đo, chưa kiểm;
- **chờ dữ liệu**: cần BDD100K MOT train thật.

## 1. Dữ liệu, split và lớp (đã khóa quy tắc, chờ dữ liệu)

- **Nguồn**: chỉ split `train` chính thức của **BDD100K MOT 2020**. Cấu hình ở `configs/phase2/detector/bdd5_train.yaml`,
  quy tắc chi tiết ở docstring `src/bdd_yolo_export.py`.
  - Không trộn UA-DETRAC.
  - Không dùng BDD100K val (bản COCO-video đã định danh là nguồn đánh giá).
  - Gói detection 100K chưa dùng; muốn dùng phải hỏi người dùng trước.
- **Lớp đã khóa**: `0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle`.
  - pedestrian, rider, train, other person: không gán nhãn.
  - crowd/ignore nhóm xe và distractor `other vehicle`, `trailer`: tô xám, trừ pixel thuộc box có nhãn; tọa độ vùng tô
    ghi vào `ignore/`.
- **Chia train-fit / dev nội bộ theo video**:
  - 10% video vào dev, seed 0, phân tầng theo sự có mặt của motorcycle và bicycle;
  - làm tròn nửa lên trong mỗi tầng.

  Danh sách được khóa vào `configs/phase2/detector/splits/bdd5_train_v1.json` và **commit trước lần huấn luyện đầu**.
  File ghi seed, tỷ lệ, tầng, danh sách video, `sha256_danh_sach` và hash nguồn nhãn.
- **`val` trong `data.yaml` = dev nội bộ** (`images/dev`), không phải BDD val chính thức.
- **Chọn frame**: giữ frame có chỉ số 0-based chia hết cho 5 (~1 ảnh/giây video); quy tắc chung cho hai model.
- **Ảnh âm tính**: giữ frame trong khoảng có nhãn của video; loại đuôi video không nhãn; mọi ảnh đều có file `.txt`.
- **Kiểm trùng** (chạy trước khi ghi ảnh). Phạm vi kiểm **khác nhau theo từng tập** — không phát biểu là “đã kiểm trùng
  nội dung ở mọi split”:
  - **BDD val**: tên video, tên ảnh và sha256 nội dung, so với toàn bộ 39.973 ảnh val.
  - **UA-DETRAC test**: **chỉ tên video**, không so nội dung.
  - **Train-fit và dev**: khác video, và từ vòng 28 thêm khác sha256 ảnh nguồn. Trùng nội dung thì dừng xuất và chờ người
    dùng quyết định; không tự loại ảnh.

  Chưa có dữ liệu train thật nên chưa biết nguồn thật có trùng hay không.
- **Hợp đồng dataset dùng chung** (vòng 28, `bdd_yolo_export.kiem_hop_dong_dataset`), gọi ở ba nơi: cuối bước xuất,
  trong runner trước khi nạp model, và trong evaluator trước khi dự đoán.
  - Kiểm đúng đường dẫn ultralytics resolve từ `data.yaml`: `train` = `images/train`, `val` = `images/dev`, `path` tuyệt đối.
  - Manifest **phiên bản 2** khóa sha256 của từng ảnh, từng file nhãn (kể cả file rỗng) và từng file ignore.
  - Phân biệt rõ ba trường hợp: file nhãn **rỗng hợp lệ**, file nhãn **thiếu** (lỗi), và file ignore có khai báo hoặc không.
  - Dataset xuất bằng manifest v1 bị từ chối và phải xuất lại; không bổ sung hash vào metadata cũ.
  - Khi train, runner đối chiếu thêm ảnh và nhãn **ultralytics thực nạp**, gồm cả nhãn lấy từ file `.cache`, với dataset
    đã xác minh, trước vòng train.
  - Chi phí trên fixture: 1,7 s cho 809 ảnh. Với ~56K ảnh, ước vài phút.
- **Quy mô ước tính**, theo mô tả công bố của BDD100K (1.400 video train, ~200 frame/video), chưa xác nhận trên máy:
  train-fit ~50.400 ảnh, dev ~5.600 ảnh. Số thật lấy từ `--buoc chia` và `--buoc xuat`.
- **Ràng buộc cho mốc 4**: tập dev dùng để tinh chỉnh tracker trên BDD sau này phải lấy từ **dev nội bộ** này, không
  lấy từ train-fit. Nếu lấy từ train-fit, detection sẽ chạy trên chính ảnh đã huấn luyện.

## 2. Checkpoint khởi tạo

| Model | File | Byte | sha256 | Loại |
|:--|:--|--:|:--|:--|
| YOLOv8n | `models/yolov8n.pt` | 6.549.796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | COCO 80 lớp — cũng là **baseline** không fine-tune |
| YOLO11n | `models/candidates/yolo11n.pt` | 5.613.764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` | COCO 80 lớp |

Runner kiểm sha256 và kiểm checkpoint là COCO 80 lớp trước khi train.

Ultralytics 8.4.141 có `cls_remap=True` (mặc định, ghi tường minh trong cấu hình): 5 hàng phân lớp được khởi tạo từ lớp
COCO cùng tên. Log smoke xác nhận “Remapped 5/5 cls head rows”. Cơ chế này giống nhau cho cả hai model.

## 3. Chính sách huấn luyện chung (`configs/phase2/detector/train_bdd5.yaml`)

Cả hai model dùng **cùng** khối `huan_luyen` và cùng dataset; chỉ `weights` khác nhau. Không tham số nào để `auto`.

| Nhóm | Giá trị |
|:--|:--|
| Epoch / ảnh vào / batch | **15** (đề xuất, chờ duyệt) / 640 / **8** (theo đo VRAM, mục 4) |
| Optimizer | SGD, nesterov, lr0 0,01, lrf 0,01 (LR tuyến tính về 1e-4), momentum 0,937, weight_decay 0,0005 |
| Ultralytics tự suy ra | nbs 64 → accumulate 8 (bước tối ưu mỗi 64 ảnh), weight_decay hiệu lực 0,0005 |
| Warmup | 1 epoch (mặc định ultralytics là 3/100 epoch; giữ cùng tỷ lệ cho 15 epoch), momentum 0,8, bias lr 0,1 |
| Augmentation | mặc định ultralytics: HSV (0,015 / 0,7 / 0,4), translate 0,1, scale 0,5, fliplr 0,5, mosaic 1,0; tắt mosaic ở 2 epoch cuối (`close_mosaic 2`); không mixup, cutmix, copy-paste, xoay, lật dọc |
| Seed / tất định | seed 0, `deterministic: true` |
| AMP / worker / cache | AMP tắt: GTX 1650 nằm trong danh sách ultralytics tự tắt AMP, nên khai báo tường minh. 2 worker. Không cache |
| Chọn checkpoint | `best.pt` theo fitness ultralytics 8.4.141 = **mAP50–95 trên dev nội bộ** (trọng số [0, 0, 0, 1]) |
| Dừng sớm | `patience 5`: dừng sau 5 epoch fitness trên dev không tăng |

- **Báo cáo sau train**: `eval_detector_5lop.py` chạy trên dev cho cả **ba** detector (YOLOv8n COCO, YOLOv8n FT, YOLO11n FT)
  với cùng quy tắc: AP50 và AP50–95 **theo lớp**, mAP là trung bình các lớp có GT, P/R tại conf 0,25 và IoU 0,5 theo lớp.
- **Giá trị resolve được ghi lại**: runner lưu giá trị lúc chạy (optimizer, param group, `amp`, accumulate, số worker,
  số ảnh) và `train_args` của checkpoint.
- **Không dùng cho lựa chọn**: BDD val và UA-DETRAC test. Hai tập này đã được xem ở các vòng trước, nên không gọi là tập
  test chưa từng quan sát.

## 4. Batch và tài nguyên — đo trên **fixture**

Bằng chứng: `logs/review27/fixture_gpu/`.

Điều kiện đo:

- GTX 1650 4 GB, driver 555.97; PyTorch 2.6.0+cu124; ultralytics 8.4.141; RAM 7,8 GB.
- Fixture: ảnh 1280×720, ~140 KB, 719 ảnh train / 90 ảnh dev.
- 1 epoch, mosaic bật.
- s/batch là trung vị khoảng cách giữa các batch, bỏ 5 batch đầu.

| Model | batch | worker | s/batch | ảnh/s | VRAM PyTorch reserved | nvidia-smi max (cả GPU) |
|:--|--:|--:|--:|--:|--:|--:|
| YOLOv8n | 16 | 2 | 3,49 | 4,6 | 3.920 MiB | 3.885 MiB |
| YOLOv8n | 8 | 2 | 0,96 / 0,98 (lặp) | 8,3 / 8,2 | 2.130 MiB | 2.540 / 2.480 MiB |
| YOLOv8n | 8 | 0 | 1,13 | 7,1 | 2.130 MiB | 2.537 MiB |
| YOLOv8n | 8 | 2, `deterministic: false` | 0,91 | 8,8 | 1.908 MiB | 2.254 MiB |
| YOLO11n | 16 | 2 | 6,07 | 2,6 | 4.302 MiB | 3.892 MiB |
| YOLO11n | 12 | 2 | 1,75 | 6,9 | 3.394 MiB | 3.746 MiB |
| YOLO11n | 8 | 2 | 1,16 | 6,9 | 2.346 MiB | 2.704 MiB |
| **YOLOv8n — cấu hình cuối, không ghi đè** | 8 | 2 | 0,97 | 8,27 | 2.130 MiB | 2.485 MiB |
| **YOLO11n — cấu hình cuối, không ghi đè** | 8 | 2 | 1,16 | 6,91 | 2.346 MiB | 2.696 MiB |
| YOLOv8n — vòng 28, runner mới, dataset v2 | 8 | 2 | 0,95 | 8,4 | 2.130 MiB | 2.532 MiB |
| YOLO11n — vòng 28, runner mới, dataset v2 | 8 | 2 | 1,14 | 7,0 | 2.346 MiB | 3.965 MiB (\*) |

(\*) `nvidia-smi` đo **cả GPU**, gồm cả các ứng dụng desktop khác đang dùng GPU. PyTorch reserved của lượt này không đổi
(2.346 MiB), nên đỉnh 3.965 MiB không quy được cho tiến trình train. Bằng chứng vòng 28: `logs/review28/fixture_gpu/`.

Kết luận đo:

- **Batch 16 chạy hết nhưng chậm 2–4 lần** với cả hai model; **không có lỗi OOM**, các lượt đều hoàn tất.
  - Bộ nhớ PyTorch reserved của batch 16 (3.920 / 4.302 MiB) sát hoặc vượt phần VRAM còn trống của card 4 GB.
  - Giả thuyết: driver Windows dùng bộ nhớ hệ thống (paging). Điều này **chưa được đo trực tiếp**.
  - Số đo chỉ cho thấy áp lực bộ nhớ đi kèm chậm, không chứng minh nguyên nhân.
  - Chênh tốc độ trên fixture không dùng để kết luận chất lượng model hay thời lượng training thật.
- Batch 12 vừa YOLO11n nhưng không tăng thông lượng.
- → **batch 8 cho cả hai model**.
- GPU bận 94–95% (trung vị) → giới hạn là tính toán GPU FP32. `deterministic` chỉ tốn ~6%, nên giữ.
- **RAM hệ thống** trống tối thiểu chỉ 25–217 MiB khi dùng 2 worker (máy có sẵn các ứng dụng khác), 628 MiB khi dùng
  0 worker. Nên đóng bớt ứng dụng khi chạy dài.

## 5. Ngân sách thời gian — **ước tính**

Cách tính: số ảnh ước tính ở mục 1, nhân với thông lượng fixture batch 8 / 2 worker. Val mỗi epoch ước theo thông lượng
val của fixture.

| Lựa chọn | YOLOv8n | YOLO11n | Tổng GPU |
|:--|--:|--:|--:|
| **Đề xuất**: bước 5 frame, **15 epoch** | ~1,8 h/epoch → **~27 h** | ~2,15 h/epoch → **~32 h** | **~59 h (~2,5 ngày)** |
| Bước 5 frame, 30 epoch | ~53 h | ~64 h | ~117 h |
| Bước 10 frame, 30 epoch | ~27 h | ~32 h | ~59 h |

- **Vì sao đề xuất bước 5 + 15 epoch** thay vì bước 10 + 30 epoch cùng chi phí: nhiều khung hình khác nhau hơn cho lớp
  hiếm. Trên val, motorcycle chỉ có trong 29/200 video; bước 10 sẽ giảm một nửa số ảnh motorcycle khác nhau.
- **Mức bất định: khoảng ±30%.**
  - fixture khác ảnh thật về nội dung, dù cùng kích thước và dung lượng tương đương;
  - áp lực RAM phụ thuộc ứng dụng khác đang mở;
  - thời gian val mỗi epoch ước thô;
  - số ảnh thật chưa biết;
  - dừng sớm có thể rút ngắn.
- **Đo lại trước khi train dài**: số này phải đo lại bằng smoke trên **dữ liệu thật** (`--tap-con smoke`), rồi cập nhật
  mục này trước khi xin duyệt.
- **Ngân sách so sánh được giữa hai model**: tính theo **số epoch và dataset giống nhau**, không theo thời gian đồng hồ.
  YOLO11n chậm hơn ~20% mỗi ảnh.

## 6. Lệnh thực tế (theo thứ tự)

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
C=configs/phase2/detector/bdd5_train.yaml
# 0. sau khi đặt dữ liệu (BDD100K_INVENTORY.md mục "Vòng 27")
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc kiem-nguon          # phải exit 0
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc chia                # khóa split -> COMMIT file split
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc xuat --tap-con smoke
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc xuat                # dataset đầy đủ + kiểm trùng + kiểm sau xuất
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc kiem
# 1. kiểm cấu hình (không train), smoke trên dữ liệu thật cho cả hai model
for M in yolov8n yolo11n; do
  $P src/train_detector.py --model $M --dry-run
  $P src/train_detector.py --model $M --che-do smoke
done
# 2. SAU KHI DUYỆT kế hoạch + ngân sách: training dài (bắt buộc cờ xác nhận)
$P src/train_detector.py --model yolov8n --che-do day-du --xac-nhan-train-dai
$P src/train_detector.py --model yolo11n --che-do day-du --xac-nhan-train-dai
# 3. đánh giá chung trên dev nội bộ (cùng quy tắc cho 3 detector)
$P src/eval_detector_5lop.py --data data/yolo_bdd5_v1/data.yaml --split val \
   --model yolov8n_coco=models/yolov8n.pt \
   --model yolov8n_ft=models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt \
   --model yolo11n_ft=models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt \
   --out results/phase2/detector_eval/dev_bdd5_v1
```

## 7. Đầu vào còn thiếu

- Ảnh **BDD100K MOT 2020 train**, đặt tại `data/raw/bdd100k/images/track/train/<video>/…jpg`.
- Nhãn **MOT 2020 train**, đặt tại `data/raw/bdd100k/labels/box_track_20/train/<video>.json`. Bản Scalabel được khuyến
  nghị; nếu chỉ có bản COCO-video thì đổi `bdd.nhan` trong cấu hình.

Chi tiết và ước tính dung lượng: `handoff/BDD100K_INVENTORY.md`, mục “Vòng 27”.

## 8. Chưa làm / cần quyết định

- **Cần duyệt**: số epoch (15) và bước frame (5) — hai tham số quyết định ngân sách ~2,5 ngày GPU.
- **Cần hỏi người dùng**: có dùng gói detection 100K hay không.
- **Chưa làm (mốc 3)**: detection cache từ model fine-tune. `detector_classes.sang_id_coco()` đã sẵn để giữ quy ước
  `--classes 1 2 3 5 7`.
- **Ngoài phạm vi vòng 27**: đánh giá trên BDD val / UA-DETRAC test, và mọi so sánh giữa model.
