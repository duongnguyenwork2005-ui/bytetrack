# Kế hoạch huấn luyện detector 5 lớp (YOLOv8n, YOLO11n) — bản để review

Trạng thái: **đề xuất, chưa chạy training dài.** Vòng 27 chỉ làm chuẩn bị và smoke. Phạm vi chung: `handoff/PLAN_DA_CHOT.md`.
Cập nhật vòng 28 (review `4937c85`): thêm hợp đồng kiểm dataset dùng chung (mục 1) và sửa diễn giải về kiểm trùng và
batch 16 (mục 1, 4). Cập nhật vòng 29 (review `07615ba`, R28-F1): chính sách **cache ảnh** (mục 1, mục 3). Số epoch,
bước frame và ngân sách **chưa được duyệt** và không đổi.
**Vòng 30** (base `bcb92f8`): kiểm lại dữ liệu — **BDD100K MOT train vẫn chưa có** trên máy, nên khóa split, xuất, smoke thật
và cập nhật ước lượng (mục 5) **bị chặn**. Tập con smoke được đối chiếu với giới hạn người dùng đặt (mục 6).
**Vòng 35** (base `ccd801f`, nhãn train theo phương án C):

- **đã khóa split** `bdd5_train_v1`;
- đã xuất **tập con smoke** (334 / 84 ảnh);
- **smoke 1 epoch trên dữ liệu thật đạt cho cả hai model**;
- đã cập nhật dự toán (mục 5) từ số đo thật.

Dataset đầy đủ **chưa xuất**. Training dài **chưa chạy**. Epoch, bước frame và ngân sách **vẫn chờ duyệt**. Chi tiết:
`REVIEW35_CHI_TIET.md`.

**Vòng 36** (base `2351c67`): Codex `2351c67` **PASS WITH LIMITATIONS** cho split và smoke. Vòng này chỉ đính chính diễn
giải theo R35-F1 và R35-F2; số liệu, split, dataset smoke và checkpoint không đổi.

- **R35-F1**: nguyên nhân confidence thấp ở checkpoint smoke **chưa xác định** (mục 5a).
- **R35-F2**: dự toán là **kịch bản theo giả định**, không phải bảo đảm.

Chi tiết: `REVIEW36_CHI_TIET.md`.

**Vòng 37**: **pilot chẩn đoán 2 epoch đã chạy và ĐẠT** cho YOLOv8n và YOLO11n, trên dataset smoke, với cấu hình khóa trước
(`configs/phase2/detector/train_bdd5_pilot.yaml`, chỉ đổi `epochs`).

- Model 5 lớp **trước bước học** đã có điểm lớp gần 0: 0 detection ≥ 0,001. Nhánh phân lớp `cv3` có 33 / 48 tensor khởi tạo
  mới khi chuyển 80 → 5 lớp.
- Trong pilot, các bước học làm điểm lớp và AP tăng.
- Warmup thực chạy (trọn epoch 1) khớp công thức của mã.
- Runner sửa lỗi `final_eval` ghi đè `val_s`/`metrics` epoch cuối.
- Training dài, xuất dataset đầy đủ, epoch và ngân sách **vẫn chờ duyệt**.

Chi tiết: `REVIEW37_CHI_TIET.md`.

**Vòng 38** (base `0c0a07a`; Codex `0c0a07a` PASS WITH LIMITATIONS, đóng R35-F1/F2):

- **Đã xuất dataset đầy đủ** `data/yolo_bdd5_v1/` (bước 5, split đã khóa): **50.487 ảnh train-fit / 5.559 ảnh dev**, khớp số
  đếm vòng 35; 8,03 GB.
- Hợp đồng v2: 0 lỗi; kiểm chéo ĐẠT. Danh sách frame được kiểm bằng chính `chon_frame()` của exporter, nên đây không
  phải triển khai độc lập của quy tắc lấy mẫu (đính chính vòng 39, R38-N2).
- Dry-run hai model (không train) nhận đúng dataset.
- **Training dài, số epoch, ngân sách, thời điểm vẫn chờ duyệt.**

Chi tiết: `REVIEW38_CHI_TIET.md`.

**Vòng 39** (base `b83dc12`; Codex `b83dc12` PASS WITH LIMITATIONS; R38-N1 / R38-N2 đã đính chính):

- **Resume** cho runner: `--che-do day-du --xac-nhan-train-dai --tiep-tuc`; kiểm trước bằng `--dry-run --tiep-tuc`.
  Tiếp tục **sau epoch đã hoàn tất**, bằng resume của ultralytics với đường dẫn checkpoint. Không trùng từng bit với
  lượt liên tục: RNG, thứ tự dữ liệu và gradient tích lũy dở không được khôi phục.
- **Bộ giám sát** `src/chay_giam_sat.py` thay bộ đo RAM cũ: bám PID của tiến trình do chính công cụ tạo, giữ exit
  code, có `trang-thai` và `tinh-lai`.
- Đã kiểm trên fixture CPU và dataset smoke GPU cho cả hai model; không train trên dataset đầy đủ.
- **Training dài, số epoch, ngân sách, thời điểm vẫn chờ duyệt.** Lệnh khởi chạy trong terminal riêng: mục 6.

Chi tiết: `REVIEW39_CHI_TIET.md`.

**Vòng 40**: Codex `1454b3d` REQUEST CHANGES vì R39-F1. Đã sửa: trước lần ghi tiếp đầu tiên, resume sao lưu nguyên
byte nhật ký rồi xử lý phần đuôi ghi dở / thiếu newline. Cú pháp resume không đổi; `--dry-run --tiep-tuc` in thêm kế
hoạch nhật ký. Chi tiết: `REVIEW40_CHI_TIET.md`.

**Vòng 41**: Codex `421dd53` PASS WITH LIMITATIONS, đóng R39-F1 trong phạm vi sửa nhật ký. Đã sửa R40-N1/N2: trước
lần ghi tiếp đầu tiên, resume sao lưu nguyên byte `results.csv` / `theo_batch.csv` rồi xử lý hàng cuối ghi dở / thiếu
newline; hàng hỏng ở giữa hoặc header sai → dừng (exit 2) trước khi train. Cú pháp resume không đổi; `--dry-run
--tiep-tuc` in thêm kế hoạch CSV. Chi tiết: `REVIEW41_CHI_TIET.md`.

Tài liệu này dùng ba nhãn nguồn số liệu:

- **fixture**: đo thật trên GPU của PC, nhưng với dữ liệu tổng hợp;
- **smoke thật** (vòng 35): đo thật trên GPU của PC với tập con dữ liệu BDD100K MOT train thật;
- **ước tính**: suy ra từ số đo, chưa kiểm;
- **chờ dữ liệu**: cần BDD100K MOT train thật. Từ vòng 34 đã có dữ liệu, nên nhãn này không còn dùng.

## 1. Dữ liệu, split và lớp (split đã khóa ở vòng 35)

**Vòng 35, đã khóa** `configs/phase2/detector/splits/bdd5_train_v1.json`:

- commit `4237149`; sha256 file `b8a2848e…`, giữ nguyên byte CRLF trong git từ `aeed110`;
- **train-fit 1.261 video / 250.566 frame**, **dev 139 video / 27.513 frame**, 0 video chung;
- tầng (motorcycle, bicycle): 830→83, 303→30, 113→11, 154→15 video dev;
- trùng tên video với BDD val và UA-DETRAC test: 0.

Số ảnh thật của dataset đầy đủ đếm bằng chính hàm của bộ xuất (chưa xuất): **bước 5 → 50.487 ảnh train-fit / 5.559 ảnh dev**.
Bảng theo lớp ở `REVIEW35_CHI_TIET.md` mục 2 và 8.

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

  Vòng 35, trên **tập con smoke** (418 ảnh): trùng nội dung train/dev 0; BDD val trùng tên 0 và trùng nội dung 0 (so với
  39.973 ảnh val đã băm). Vòng 38, **dataset đầy đủ** (56.046 ảnh):
  - trùng nội dung train/dev: **0 cặp**;
  - BDD val: trùng tên 0, trùng nội dung 0;
  - UA-DETRAC test: chỉ so tên video, 0 trùng.
- **Hợp đồng dataset dùng chung** (vòng 28, `bdd_yolo_export.kiem_hop_dong_dataset`), gọi ở ba nơi: cuối bước xuất,
  trong runner trước khi nạp model, và trong evaluator trước khi dự đoán.
  - Kiểm đúng đường dẫn ultralytics resolve từ `data.yaml`: `train` = `images/train`, `val` = `images/dev`, `path` tuyệt đối.
  - Manifest **phiên bản 2** khóa sha256 của từng ảnh, từng file nhãn (kể cả file rỗng) và từng file ignore.
  - Phân biệt rõ ba trường hợp: file nhãn **rỗng hợp lệ**, file nhãn **thiếu** (lỗi), và file ignore có khai báo hoặc không.
  - Dataset xuất bằng manifest v1 bị từ chối và phải xuất lại; không bổ sung hash vào metadata cũ.
  - Khi train, runner đối chiếu thêm ảnh và nhãn **ultralytics thực nạp**, gồm cả nhãn lấy từ file `.cache`, với dataset
    đã xác minh, trước vòng train.
  - Chi phí trên fixture: 1,7 s cho 809 ảnh. Với ~56K ảnh, ước vài phút.
- **Hai loại cache khác nhau**, mỗi loại có chốt riêng:
  - **Cache NHÃN** `labels/<split>.cache` (vòng 28): ultralytics chỉ kiểm tổng kích thước + đường dẫn. Runner đối chiếu
    nhãn thực nạp với file nhãn đã xác minh ngay trước vòng train.
  - **Cache ẢNH** `<ảnh>.npy` (vòng 29, R28-F1): ultralytics 8.4.141 `load_image()` nạp `.npy` cùng tên **thay** JPEG
    **kể cả khi `cache=False`**.
    - Chính sách hiện tại là `cache: false` và **không dùng cache ảnh**: pixel nạp phải là JPEG đã khóa sha256.
    - Hợp đồng từ chối `.npy` cạnh chính các ảnh train/dev mà trainer sẽ nạp, nêu rõ split, đường dẫn, lý do; không tự xóa.
    - Runner từ chối cấu hình `cache` khác `false`.
    - Callback trước vòng train kiểm lại nguồn pixel của dataset thật (không cache ảnh, không `.npy`).
- **Quy mô**: ước tính cũ theo mô tả công bố của BDD100K (1.400 video train, ~200 frame/video) là train-fit ~50.400 ảnh, dev
  ~5.600 ảnh. Vòng 35 đếm thật: **50.487 / 5.559** ở bước 5. Vòng 38 xuất thật ra **đúng 50.487 / 5.559**; tập
  (video, frame) khớp từng phần tử với tập tính lại.
  - dev: motorcycle 226 box / 205 ảnh / 24 video; bicycle 407 box / 334 ảnh / 44 video;
  - thống kê đầy đủ ở `REVIEW38_CHI_TIET.md` mục 4.
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
| AMP / worker / cache | AMP tắt: GTX 1650 nằm trong danh sách ultralytics tự tắt AMP, nên khai báo tường minh. 2 worker. `cache: false` **bắt buộc**: runner từ chối giá trị khác; `.npy` cạnh ảnh train/dev bị chặn (mục 1) |
| Chọn checkpoint | `best.pt` theo fitness ultralytics 8.4.141 = **mAP50–95 trên dev nội bộ** (trọng số [0, 0, 0, 1]) |
| Dừng sớm | `patience 5`: dừng sau 5 epoch fitness trên dev không tăng |

- **Báo cáo sau train**: `eval_detector_5lop.py` chạy trên dev cho cả **ba** detector (YOLOv8n COCO, YOLOv8n FT, YOLO11n FT)
  với cùng quy tắc: AP50 và AP50–95 **theo lớp**, mAP là trung bình các lớp có GT, P/R tại conf 0,25 và IoU 0,5 theo lớp.
- **Giá trị resolve được ghi lại**: runner lưu giá trị lúc chạy (optimizer, param group, `amp`, accumulate, số worker,
  số ảnh) và `train_args` của checkpoint.
- **Không dùng cho lựa chọn**: BDD val và UA-DETRAC test. Hai tập này đã được xem ở các vòng trước, nên không gọi là tập
  test chưa từng quan sát.

## 4. Batch và tài nguyên — đo trên **fixture** (vòng 27–28) và **smoke thật** (vòng 35, cuối mục)

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

**Vòng 35 — smoke thật** (334 ảnh train, 1 epoch, 42 batch, cấu hình cuối không ghi đè). Bằng chứng:
`logs/review35/12_*`.

| Model | batch | worker | s/batch | ảnh/s | VRAM PyTorch reserved | nvidia-smi max (cả GPU) | RSS train + con max |
|:--|--:|--:|--:|--:|--:|--:|--:|
| YOLOv8n | 8 | 2 | 0,991 | 8,07 | 2.134 MiB | 2.624 MiB | 3.018 MiB |
| YOLO11n | 8 | 2 | 1,131 | 7,07 | 2.344 MiB | 2.840 MiB | 2.828 MiB |

- s/batch trên dữ liệu thật gần với fixture: v8n chậm hơn ~2–4%, 11n nhanh hơn ~1–3%. VRAM PyTorch gần như trùng fixture.
- RAM hệ thống còn trống thấp nhất trong lúc smoke chỉ **16–146 MiB**, vì các ứng dụng khác chiếm ~3,9 GB.

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

### 5a. Vòng 35 — từ smoke thật (thay cho 5b)

Script `logs/review35/15_du_toan_training.py`; mọi giả định ghi trong `15_du_toan_training.json`.

Cách tính:

- train/epoch = ⌈ảnh train / 8⌉ × s/batch ổn định đo ở smoke thật.
- val/epoch = `val_s` của smoke + (ảnh dev − 84) × ms/ảnh val của ultralytics.
- Khởi tạo dataset đầy đủ: thêm băm hợp đồng và quét nhãn, ~4,9 phút mỗi lượt.
- **Kịch bản chậm hơn**: giả định thêm 5 ms/ảnh hậu xử lý val, rồi ×1,3 cho rủi ro chưa đo (GPU nóng hoặc giảm xung khi chạy
  nhiều giờ, phân trang RAM).
- **Đính chính vòng 36 (R35-F2)**: mọi mức dưới đây là **kịch bản dự toán theo các giả định đã nêu**, không phải cận trên hay
  bảo đảm. Thời gian và bộ nhớ thực tế có thể vượt cả kịch bản chậm hơn. Số liệu không đổi (`logs/review36/03_*`).

| Lựa chọn | YOLOv8n | YOLO11n | Tổng GPU tuần tự |
|:--|--:|--:|--:|
| **Đề xuất**: bước 5 (50.487 ảnh), **15 epoch** | 1,74 h/epoch → **26,5 h** (chậm hơn 34,6 h) | 1,98 h/epoch → **30,2 h** (chậm hơn 39,4 h) | **56,64 h ≈ 2,4 ngày** (chậm hơn 73,95 h) |
| Bước 5, 30 epoch | 52,9 h | 60,2 h | 113,08 h (chậm hơn 147,61 h) |
| Bước 10 (25.848 ảnh), 15 epoch | 13,6 h | 15,5 h | 29,10 h (chậm hơn 37,99 h) |
| Bước 10, 30 epoch | 27,2 h | 30,9 h | 58,08 h (chậm hơn 75,81 h) |

- **Chưa tính**: dừng sớm (`patience 5`), và xuất dataset đầy đủ (ước thô 15–30 phút, chưa đo).
- **RAM / VRAM / đĩa** cho dataset đầy đủ: kịch bản ước tính từ smoke, **không phải bảo đảm**.
  - RSS train + con ~2,9–3,5 GB;
  - torch reserved ~2,1–2,4 GB (v8n), ~2,3–2,6 GB (11n);
  - cả GPU ~2,6–2,9 GB (v8n), ~2,8–3,1 GB (11n), trên card 4 GB. Lô nhiều box hơn smoke có thể làm VRAM vượt kịch bản;
  - dataset bước 5 ~8,5 GB (bước 10 ~4,3 GB). D: còn ~63 GB.
- **Bất định**: dự toán ngoại suy từ một epoch 42 batch (36 khoảng đo). Hành vi khi chạy nhiều giờ với RAM thấp chưa đo.
  Kịch bản chậm hơn **không** bao trùm mọi trường hợp xấu.
- **Smoke không kiểm động học training dài** (đã đính chính ở vòng 36, R35-F1):
  - checkpoint smoke 5 lớp có confidence thấp hơn checkpoint COCO 80 lớp; YOLO11n không còn detection nào ≥ 0,001;
  - **nguyên nhân chưa xác định**: chuyển sang 5 lớp/khởi tạo và các bước học (gồm warmup) chưa được tách. "Không warmup khi
    `epochs=1`" chỉ dựa trên đọc mã nguồn ultralytics, không phải log thực chạy;
  - cấu hình 15 epoch khai báo 1 epoch warmup.

  Xem `REVIEW35_CHI_TIET.md` mục 7 và `REVIEW36_CHI_TIET.md`.
- **Pilot vòng 37** (2 epoch, warmup trọn epoch 1, mosaic tắt cả hai epoch theo `close_mosaic 2`):
  - model 5 lớp trước bước học: điểm lớp max/ảnh trung vị 1,9e-4 (v8n) / 2,8e-4 (11n), 0 detection ≥ 0,001;
  - car AP50 (model) sau epoch 2: 0,510 / 0,413;
  - s/batch 0,93 / 1,11, gần smoke, nên kịch bản dự toán ở trên **không đổi đáng kể**;
  - `val_s` smoke (dùng làm chi phí val cố định/epoch) gồm cả `final_eval` do lỗi runner đã sửa, có thể tính dư nhưng nhỏ.
    Chưa tính lại.

  Xem `REVIEW37_CHI_TIET.md`.

### 5b. Ước tính cũ từ fixture (vòng 27–30, giữ để đối chiếu)

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
- **Vòng 30**: vẫn **chưa cập nhật** vì chưa có dữ liệu train. Bảng trên vẫn là ước tính từ fixture và từ số video BDD công bố.
- **Vòng 35**: đã đo lại bằng smoke trên dữ liệu thật. Kết quả ở mục 5a: ~56,6 h so với ~59 h của bảng này.
- **Ngân sách so sánh được giữa hai model**: tính theo **số epoch và dataset giống nhau**, không theo thời gian đồng hồ.
  YOLO11n chậm hơn ~20% mỗi ảnh.

## 6. Lệnh thực tế (theo thứ tự)

**Tập con smoke** (`bdd5_train.yaml`, `tap_con.smoke`): 16 video train-fit + 4 video dev, lấy từ split **đã khóa** (hai
phần không chung video), bước 10 frame, seed 0. Tập con được chọn **trước khi xem AP**; hai model dùng **cùng** tập con.

- Mỗi video BDD MOT có tối đa ~204 frame, nên mỗi video cho ≤ 21 ảnh; theo cấu tạo tối đa khoảng 336 ảnh train-fit và 84
  ảnh dev. Mức này nằm trong giới hạn người dùng đặt là **≤ 1.000 / ≤ 200**.
- Sau `--buoc xuat --tap-con smoke`, đọc `manifest/tom_tat.json` để xác nhận số ảnh thực.
- Nếu vượt giới hạn thì dừng và hỏi; không tự cắt bớt.
- **Vòng 35, đã chạy**:
  - số thật là **334 / 84 ảnh**;
  - `--dry-run` mặc định đọc **dataset đầy đủ**; khi dataset đầy đủ chưa xuất, thêm `--data data/yolo_bdd5_v1_smoke/data.yaml`;
  - smoke chạy với `--ten train_bdd5_v1_<model>_smoke_lan2`, vì lượt lần 1 (lỗi công cụ đo) được giữ làm bằng chứng và runner
    không ghi đè.

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
C=configs/phase2/detector/bdd5_train.yaml
# 0. sau khi đặt dữ liệu (BDD100K_INVENTORY.md mục "Vòng 27")
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc kiem-nguon          # phải exit 0
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc chia                # khóa split -> COMMIT file split
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc xuat --tap-con smoke
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc xuat                # dataset đầy đủ + kiểm trùng + kiểm sau xuất (vòng 38: ĐÃ CHẠY, exit 0)
$P src/prepare_mixed_yolo_dataset.py --config $C --buoc kiem                # (vòng 38: ĐÃ CHẠY lại, exit 0, 0 lỗi)
# 1. kiểm cấu hình (không train), smoke trên dữ liệu thật cho cả hai model
for M in yolov8n yolo11n; do
  $P src/train_detector.py --model $M --dry-run
  $P src/train_detector.py --model $M --che-do smoke
done
# 1b. (vong 37, DA CHAY) pilot chan doan 2 epoch tren dataset smoke: khoa cau hinh roi chay tuan tu
#     (bash handoff/logs/review37/chay_pilot.sh: kiem-pilot + so khop khoa + pilot, cho tung model)
for M in yolov8n yolo11n; do
  $P src/train_detector.py --config configs/phase2/detector/train_bdd5_pilot.yaml --model $M --che-do kiem-pilot
  $P -u src/train_detector.py --config configs/phase2/detector/train_bdd5_pilot.yaml --model $M --che-do pilot --xac-nhan-pilot
done
# 2. SAU KHI DUYỆT kế hoạch + ngân sách: training dài (bắt buộc cờ xác nhận), trong TERMINAL RIÊNG, qua bộ giám sát
#    (vòng 39). Lệnh PowerShell / cmd và cách lấy exit code: REVIEW39_CHI_TIET.md mục 8. Thư mục giám sát MỚI mỗi lần.
$P src/chay_giam_sat.py chay --thu-muc results/phase2/detector_train/giam_sat/yolov8n_lan0 -- \
   $P -u src/train_detector.py --model yolov8n --che-do day-du --xac-nhan-train-dai
#    chỉ chạy YOLO11n SAU KHI YOLOv8n kết thúc với exit 0 (tuần tự, không chạy đồng thời)
$P src/chay_giam_sat.py chay --thu-muc results/phase2/detector_train/giam_sat/yolo11n_lan0 -- \
   $P -u src/train_detector.py --model yolo11n --che-do day-du --xac-nhan-train-dai
# 2b. sau gián đoạn: trạng thái lần trước (không đoán từ log) -> kiểm tiếp tục (không train) -> tiếp tục
$P src/chay_giam_sat.py trang-thai --thu-muc results/phase2/detector_train/giam_sat/yolov8n_lan0
$P src/train_detector.py --model yolov8n --dry-run --tiep-tuc
$P src/chay_giam_sat.py chay --thu-muc results/phase2/detector_train/giam_sat/yolov8n_tiep_tuc_1 -- \
   $P -u src/train_detector.py --model yolov8n --che-do day-du --xac-nhan-train-dai --tiep-tuc
# 3. đánh giá chung trên dev nội bộ (cùng quy tắc cho 3 detector)
$P src/eval_detector_5lop.py --data data/yolo_bdd5_v1/data.yaml --split val \
   --model yolov8n_coco=models/yolov8n.pt \
   --model yolov8n_ft=models/finetune_phase2/train_bdd5_v1_yolov8n_day_du/weights/best.pt \
   --model yolo11n_ft=models/finetune_phase2/train_bdd5_v1_yolo11n_day_du/weights/best.pt \
   --out results/phase2/detector_eval/dev_bdd5_v1
```

## 7. Đầu vào

**Từ vòng 34, không còn thiếu**:

- Ảnh MOT train: vòng 31; 1.400 video, 278.079 ảnh.
- Nhãn Scalabel `box_track_20/train`: vòng 34, theo **phương án C**.
  - Snapshot bên thứ ba đã ghim (HF `jaffe03195/bdd100k_sot` @ `694197099692…`), hash từng file.
  - **Chưa xác thực** bằng archive gốc.
- `kiem-nguon` exit 0.

Chi tiết: `handoff/BDD100K_INVENTORY.md`, mục Vòng 31–35.

**Vòng 38**: dataset đầy đủ **đã xuất**, `data/yolo_bdd5_v1/data.yaml`.

- Hash: `data.yaml` `1280e57b…`, manifest `1ee6174f…`.
- 8.034.288.965 B, 130.699 file; không commit.

## 8. Chưa làm / cần quyết định

- **Cần duyệt**: số epoch (15) và bước frame (5). Hai tham số này quyết định ngân sách: vòng 35 có kịch bản **~56,6 h GPU**
  (kịch bản chậm hơn ~74 h; không phải bảo đảm), mục 5a.
- **Cần quyết định** (cập nhật vòng 37):
  - **Pilot chẩn đoán**: ĐÃ CHẠY ở vòng 37 (2 model × 2 epoch, có đo model 5 lớp trước bước học và log warmup/LR thực tế).
  - **Đối chứng warmup** (cùng seed, có / không warmup) nếu muốn đóng hẳn câu hỏi warmup của R35-F1: đây là ablation, chờ
    duyệt, chưa làm. Không bắt buộc cho quyết định training dài.
  - Xuất dataset đầy đủ: **ĐÃ LÀM ở vòng 38**.
  - Đóng bớt ứng dụng để có RAM khi train dài: RAM trống tối thiểu chỉ 7–59 MiB trong pilot; 1,26 GB lúc kết thúc vòng 38.
  - **Vòng 38**:
    - mỗi lượt dài khoảng 26–30 h theo kịch bản, vượt giới hạn tác vụ nền ~2 h, nên nên khởi chạy trong terminal riêng;
    - runner không có chế độ resume → **vòng 39 đã thêm** `--tiep-tuc` (`REVIEW39_CHI_TIET.md` mục 3);
    - bộ đo RAM ngoài chọn nhầm tiến trình → **vòng 39 đã thay** bằng `src/chay_giam_sat.py` (bám PID do công cụ
      tạo).

    Lệnh dự kiến ở `REVIEW38_CHI_TIET.md` mục 5.
- **Cần hỏi người dùng**: có dùng gói detection 100K hay không.
- **Chưa làm (mốc 3)**: detection cache từ model fine-tune. `detector_classes.sang_id_coco()` đã sẵn để giữ quy ước
  `--classes 1 2 3 5 7`.
- **Ngoài phạm vi vòng 27**: đánh giá trên BDD val / UA-DETRAC test, và mọi so sánh giữa model.
