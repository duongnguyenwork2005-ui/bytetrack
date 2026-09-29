# Kiểm kê BDD100K-MOT trên máy — trạng thái và phần còn thiếu

Kiểm ngày **2026-09-18**, máy `D:\LABOTARY\bytetrack`, nhánh `phase2-mixed-traffic`.
(Bản trước, 2026-09-17, ghi "chưa có dữ liệu"; nay **đã có ảnh**, **chưa có nhãn** — xem bảng dưới.)
**Cập nhật 2026-09-29 (vòng 27)**: nhãn val đã có dưới dạng COCO-video. Split **train** (ảnh + nhãn) vẫn chưa có, nên
chưa chuẩn bị được dataset huấn luyện detector. Chi tiết ở mục ngay dưới.

## Vòng 27 (2026-09-29) — kiểm kê lại cho huấn luyện detector 5 lớp

Lệnh kiểm: `python src/kiem_ke_bdd100k.py --out handoff/logs/review27/01_kiem_ke_bdd100k.json --bam-archive`.
Script chỉ đọc (không giải nén, sửa hay xóa gì); log ở `handoff/logs/review27/01_kiem_ke_bdd100k.{json,log}`.

### Trên máy có gì

| Thành phần | Đường dẫn | Trạng thái |
|:--|:--|:--|
| Archive ảnh MOT 2020 val (phần 1) | `D:\LABOTARY\images20-track-val-1.zip` | có — 4.983.938.716 B, sha256 `4d678810a14095ab…`; 40.174 mục, gồm 39.973 ảnh của 200 video, **chỉ** `images/track/val` |
| Ảnh MOT val đã giải nén | `data/raw/bdd100k/images/track/val/<video>/` | có — 200 video, 39.973 ảnh (4,8 GB); header **toàn bộ 1280×720**, khớp kích thước khai báo trong nhãn |
| Nhãn MOT val (bản chuyển đổi **COCO-video**) | `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` | có — 117.094.032 B, sha256 `074ff795…`; cùng byte với bản ngoài repo `D:\LABOTARY\01-bdd_box_track_val_cocofmt.json` |
| Ảnh MOT **train** | `data/raw/bdd100k/images/track/train/` | ❌ **không có** |
| Nhãn MOT **train** (Scalabel `box_track_20` hoặc COCO-video) | `data/raw/bdd100k/labels/box_track_20/train/` hoặc `labels_cocovid/*train*` | ❌ **không có** |
| Nhãn MOT val gốc Scalabel | `data/raw/bdd100k/labels/box_track_20/val/` | không có (bản COCO-video ở trên thay thế cho đánh giá) |
| Gói detection 100K (`images/100k`, `labels/det_20`) | — | không có |
| Thư mục `datasets/BDD100K` của dự án cũ | `D:\LABOTARY\ByteTrack + EKF + Kitty\datasets\BDD100K` | rỗng (0 file) |

Kết luận của script: `du_de_chuan_bi_train = false`.

### Thống kê nhãn MOT val (nguồn đánh giá đã định danh — không dùng làm train/dev)

Tổng cộng: 200 video, 39.973 ảnh, 465.512 annotation. Mỗi video có 102–204 frame, trung bình 199,9.
Có 615 ảnh không có annotation nào: 488 ảnh nằm trong khoảng có nhãn của video, 127 ảnh ở đuôi video sau frame có nhãn cuối.

| Lớp | box (không crowd/ignore) | crowd | ignore | ảnh có lớp | video có lớp |
|:--|--:|--:|--:|--:|--:|
| car | 341.436 | 20.079 | 2.825 | 38.964 | 200 |
| truck | 27.343 | 481 | 424 | 15.862 | 160 |
| bus | 9.087 | 4 | 0 | 7.018 | 93 |
| **motorcycle** | **902** | 13 | 0 | **774** | **29** |
| bicycle | 4.135 | 271 | 0 | 2.738 | 63 |
| pedestrian *(ngoài 5 lớp)* | 57.065 | 1.842 | 816 | 14.792 | 152 |
| rider *(ngoài 5 lớp)* | 2.546 | 0 | 0 | 2.260 | 69 |
| train *(ngoài 5 lớp)* | 308 | 0 | 0 | 308 | 6 |

- Cờ `ignore` chỉ xuất hiện ở car, truck và pedestrian. Đây là các distractor `other vehicle`, `trailer`, `other person`
  đã được ánh xạ sang lớp đánh giá khi chuyển sang COCO-video.
- **Ảnh trùng nội dung**: 2 frame liên tiếp của video val `b1d7b3ac-5744370e` (`-0000079.jpg`, `-0000080.jpg`) giống
  nhau từng byte, nên toàn tập có 39.973 ảnh nhưng chỉ 39.972 hash khác nhau. Đây là đặc điểm của dữ liệu, không phải
  ảnh thiếu.

### MOT và detection là hai nguồn khác nhau, không thay thế trực tiếp

- **MOT 2020**: ảnh 5 fps theo video, có ID đối tượng theo thời gian. Split `train` là nguồn của train-fit và dev nội
  bộ trong plan.
- **Detection 100K**: mỗi ảnh là một khung hình tĩnh lấy từ một video, nhãn `det_20`, không có ID theo thời gian.
- Pipeline tracking và `configs/phase2/detector/bdd5_train.yaml` chỉ nhận MOT train. Muốn dùng gói detection (thay
  thế hoặc bổ sung) phải hỏi người dùng trước.

### Gói còn thiếu và đường dẫn cần đặt

Không dùng UA-DETRAC hay BDD val để thay thế.

1. **Ảnh MOT 2020 train**: gói *MOT 2020 Images — train* trên trang tải BDD100K chính thức.
   - Tên gói: gói val đã tải tên `images20-track-val-1.zip`; gói train được chia nhiều phần theo cùng kiểu tên. Máy không
     truy cập được trang tải lúc kiểm, nên cần xem lại tên và số phần trên trang.
   - Giải nén vào `data/raw/`, **không** vào `data/raw/bdd100k/`, vì archive đã có sẵn tiền tố `bdd100k/`.
   - Kết quả cần có: `data/raw/bdd100k/images/track/train/<video>/<video>-<7 chữ số>.jpg`.
2. **Nhãn MOT 2020 train, bản Scalabel (khuyến nghị)**: gói nhãn box tracking `box_track_20`.
   - Kết quả cần có: `data/raw/bdd100k/labels/box_track_20/train/<video>.json`, mỗi video một file.
   - Nếu chỉ có bản COCO-video (`*train*cocofmt.json`): đặt vào `data/raw/bdd100k/labels_cocovid/`, rồi đổi `bdd.nhan`
     trong `bdd5_train.yaml` sang `dinh_dang: cocovid`. Bản này đọc cả file vào RAM: file val 117 MB cho 200 video, bản
     train sẽ lớn hơn nhiều, trong khi máy có 7,8 GB RAM.
3. **Ước tính dung lượng**, suy từ val (5 GB zip ≈ 4,8 GB ảnh cho 200 video):
   - ảnh train khoảng 7 lần val, tức ~35 GB zip và ~34 GB sau giải nén;
   - dataset YOLO xuất ở bước 5 frame thêm khoảng 7 GB.

   Ổ `D:` còn ~125 GB trống. Số video thật chưa biết: `kiem-nguon` sẽ in số thật, không dùng con số ước tính.
4. **Kiểm sau khi đặt xong**: `python src/prepare_mixed_yolo_dataset.py --config configs/phase2/detector/bdd5_train.yaml --buoc kiem-nguon`
   phải exit 0. Lệnh này đối chiếu nhãn với ảnh theo **từng video**. Thiếu gì thì trả exit 2 và in đúng đường dẫn thiếu.

---

*Các mục 1–7 dưới đây là kiểm kê ngày 2026-09-18 (giữ nguyên để đối chiếu).*

## 1. Đã tải những gì — kết quả kiểm tra thực tế

**Archive đã tải**: `D:\LABOTARY\images20-track-val-1.zip` — 4,75 GB, 40.174 entry, thư mục gốc bên trong là
`bdd100k/images/…`. Tên gói cho biết đây là **MOT 2020 → Images → track → val → part 1**.

**Đã giải nén đúng vị trí** (không lồng nhầm `bdd100k/bdd100k`):

| Thành phần | Đường dẫn pipeline yêu cầu | Trạng thái |
|:--|:--|:--|
| **Ảnh MOT, split `val`** | `data/raw/bdd100k/images/track/val/<video>/<video>-<7 chữ số>.jpg` | ✅ **200 video, 39.973 ảnh**, đặt tên đúng quy ước adapter |
| **Ảnh MOT, split `train`** | `data/raw/bdd100k/images/track/train/…` | ❌ **không có** |
| **Nhãn tracking `val`** | `data/raw/bdd100k/labels/box_track_20/val/<video>.json` | ❌ **không có** — thư mục `labels/` không tồn tại, **0 file JSON** trong toàn bộ `data/raw/bdd100k` |
| **Nhãn tracking `train`** | `data/raw/bdd100k/labels/box_track_20/train/<video>.json` | ❌ **không có** |

> **"Tải xong dataset" không có nghĩa là đủ cả hai.** Gói đã tải là **Images**; gói **Labels** là một file
> tải **riêng**. Ảnh chiếm 4,75 GB, nhãn chỉ khoảng **50 MB**.

## 2. Hệ quả: chưa chạy được gì trên dữ liệu BDD thật

Adapter lấy **danh sách video, danh sách frame, tên file ảnh và `timestamp` TỪ file nhãn JSON**, không phải
từ thư mục ảnh ([bdd100k_mot.py:221–243](../src/mot_datasets/bdd100k_mot.py)):

```python
def list_videos(self, part):          # glob thu muc LABELS
    d = self.labels_dir(part)
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []

def frames(self, video, part="val"):  # doc JSON nhan -> frameIndex, name, timestamp
    fr = self.load_json(video, part)
```

Vì vậy **thiếu nhãn là chặn cứng** cho cả ba bước, kể cả bước chỉ cần ảnh:

| Lệnh | Kết quả hiện tại |
|:--|:--|
| `python src/check_bdd100k.py --part val` | **exit 1** — fail-loud, in đúng đường dẫn nhãn còn thiếu |
| `python src/det_cache.py --dataset bdd100k …` | không chạy được: cần `ds.frames()` → cần JSON nhãn |
| evaluator (TrackEval) | không chạy được: không có GT |

**Không** thay bằng nhãn detection (`labels/det_20`, `bdd100k_labels_images_*`) hay bằng fixture
`data/fixtures/bdd100k_mini` — fixture là dữ liệu **tổng hợp**, chỉ để kiểm tra pipeline (README của nó ghi
rõ), không dùng để báo cáo kết quả thực nghiệm.

## 3. Cần tải thêm gì — nguồn, vị trí, lệnh kiểm tra

Đăng ký tại **bdd-data.berkeley.edu** (nguồn chính thức, BAIR/Berkeley DeepDrive), phần **MOT 2020**:

| Gói | Bắt buộc? | Giải nén sao cho có |
|:--|:--|:--|
| **MOT 2020 Labels** | ✅ **bắt buộc** (~50 MB) | `data/raw/bdd100k/labels/box_track_20/{train,val}/<video>.json` |
| MOT 2020 Images — `train` | chỉ khi làm **Thí nghiệm 2** (tuning) | `data/raw/bdd100k/images/track/train/<video>/…jpg` |
| MOT 2020 Images — `val` | ✅ **đã có** | — |

Gói `test` **không cần**: BDD không công bố nhãn test.

Giải nén cẩn thận: archive chứa sẵn tiền tố `bdd100k/`, nên giải nén vào `data/raw/` (**không** vào
`data/raw/bdd100k/`, sẽ thành `data/raw/bdd100k/bdd100k/…`).

Kiểm tra theo đúng thứ tự — **mỗi bước phải exit 0 trước khi sang bước sau**:

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe

# 1. Schema + ảnh + thống kê lớp + phân bố Δt, đồng thời sinh manifest final (val)
$P src/check_bdd100k.py --part val --make-manifest
#    -> results/phase2/bdd100k_check/val_{class_stats,dt_stats}.csv, val_schema_report.json
#    -> results/phase2/manifests/bdd100k_final_val.json
#    phải in "[schema] N/N video hop le; 0 loi" và "[images] thieu 0 anh"

# 2. Manifest train-dev (~40 video, seed 0) - CHỈ dùng để tuning, KHÔNG dùng đánh giá
#    (chỉ chạy được khi đã có nhãn + ảnh của split `train`)
$P src/check_bdd100k.py --part train --make-manifest --n-dev 40 --seed 0
#    -> results/phase2/manifests/bdd100k_dev.json

# 3. Cache detection riêng cho BDD (CẦN GPU)
$P src/det_cache.py --dataset bdd100k --part val \
   --manifest results/phase2/manifests/bdd100k_smoke_val.json \
   --cache-id yolov8n-coco_conf0.05_smoke_bdd

# 4. Smoke 1-3 sequence bằng YAML + output RIÊNG (xem mục 4)
$P src/run_experiment.py --exp configs/phase2/experiments/bdd_smoke_val.yaml
```

> ⚠️ `run_experiment.py` **không có tham số `--videos`** (chỉ có `--exp/--stage/--dry-run/--no-skip/--force/
> --force-eval`). Muốn chạy trên một tập con thì phải **tạo manifest riêng + YAML riêng**, như mục 4 — bản
> trước của tài liệu này ghi `--videos <v1> <v2> <v3>` là **sai**, lệnh đó sẽ lỗi.

Tên manifest train-dev do `check_bdd100k.py` sinh ra thực tế là **`bdd100k_dev.json`** (không phải
`bdd100k_dev_train.json` như bản trước viết).

## 4. Smoke BDD — đã chuẩn bị sẵn, chạy được ngay khi có nhãn

Tách hoàn toàn khỏi đánh giá đầy đủ, **tên riêng ở cả bốn chỗ**:

| | giá trị |
|:--|:--|
| Manifest | `results/phase2/manifests/bdd100k_smoke_val.json` (1–3 sequence, **chọn và ghi ra file TRƯỚC khi xem metric**) |
| YAML | [`configs/phase2/experiments/bdd_smoke_val.yaml`](../configs/phase2/experiments/bdd_smoke_val.yaml) |
| `name` / thư mục kết quả | `bdd_smoke_val` → `results/phase2/bdd_smoke_val/`, `data/processed/trackers/BDD-val/p2-bdd_smoke_val-*` |
| `cache_id` | `yolov8n-coco_conf0.05_smoke_bdd` |

⇒ không ghi đè `bdd100k_final_val.json`, `bdd_transfer_val`, hay cache đánh giá đầy đủ.

Smoke dùng **CV và EKF** với cấu hình đã khóa, kiểm toàn đường chạy cache → tracking → evaluator → bảng tổng
hợp. **Không** chọn tham số hay bộ lọc từ metric của smoke.

## 5. Ánh xạ lớp, thuộc tính và vùng bỏ qua — đã định nghĩa, chờ dữ liệu để xác nhận

| Hạng mục | Quy ước đã chốt |
|:--|:--|
| Lớp đánh giá | 8 lớp BDD: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle |
| Ánh xạ detector | COCO → BDD: `1→bicycle, 2→car, 3→motorcycle, 5→bus, 7→truck`. **`person`(0) KHÔNG map** vì COCO gộp pedestrian + rider |
| Mục tiêu chính | `vehicle` = car + truck + bus + motorcycle + bicycle → class 1 (liên tục với UA-DETRAC) |
| Vùng bỏ qua | mọi box `crowd` (bất kỳ lớp), 3 lớp distractor (*other person, trailer, other vehicle*), và `train` khi target = vehicle → class 13, conf 0 |
| `occluded` | **cờ NHỊ PHÂN** "có bị che một phần hay không". BDD **không** ghi mức độ che. Interim gán `occlusion_ratio = 0,5` — đây là **GIÁ TRỊ QUY ƯỚC cho một cờ nhị phân, KHÔNG phải mức che 50 % đo được**: 0,5 được chọn vì vượt ngưỡng partial (0,10) nhưng **không bao giờ** vượt ngưỡng full (0,90), tức không bao giờ biến một cờ nhị phân thành sự kiện "che gần hoàn toàn". `visibility = 1 − occlusion_ratio` mang **cùng tính chất quy ước**. **Không so trực tiếp** với `occlusion_ratio` **đo được** của UA-DETRAC |
| Khoảng trống annotation (**gap**) | **KHÔNG** suy diễn thành che hoàn toàn. `occlusion_eval --gaps` tạo sự kiện gap với `has_full=False` và `gap_evidence ∈ {occluded_adjacent, unknown}`, báo cáo **tách tầng** |
| Thời gian | 5 fps danh nghĩa; Δt thực lấy từ `timestamp` (ms) khi **mọi** frame của video có trường đó, **ngược lại** suy ra theo `frameIndex / 5` — cột `dt_source` trong interim ghi rõ từng video dùng nguồn nào; bộ lọc vẫn ở **đơn vị bước** (dt = 1); `track_buffer_sec` 1,2 s = 6 bước @ 5 fps |
| `rider` | box **riêng** với xe hai bánh, không gộp; chỉ thống kê, **không** đánh giá (detector COCO không có lớp này) |

⚠️ **Chưa xác nhận trên dữ liệu thật**: tỉ lệ video có `timestamp` ở mọi frame, số video có khoảng lệch nhịp,
thống kê lớp thật (nhất là `motorcycle`/`bicycle` — phần liên quan nhất tới giao thông hỗn hợp), và số vùng
`crowd` thật. `check_bdd100k.py` in đúng các con số này ở bước 1 (`dt_source`, `n_intervals_irregular`,
`val_class_stats.csv`).

## 6. Kế hoạch đánh giá — **hai thí nghiệm tách biệt, không gộp kết luận**

### A. Chuyển cấu hình UA-DETRAC → BDD100K (`bdd_transfer_val.yaml`)
Giữ **nguyên** tham số nghiên cứu đã khóa trên UA-DETRAC, xử lý thời gian theo quy ước pipeline.
**Không tuning bằng nhãn BDD.** Ghi rõ khi báo cáo: `std_theta`/`std_omega` là **theo bước**, mà một bước ở
BDD dài gấp 5 lần ở DETRAC (0,2 s so với 0,04 s) — thí nghiệm này **cố ý** giữ nguyên giá trị theo bước,
đúng nghĩa "chuyển cấu hình"; **không** quy đổi bằng công thức.

### B. Tuning trên BDD train-dev rồi đánh giá trên val (`bdd_tuned_val.yaml`)
**Hiện chưa đủ điều kiện**: chỉ có ảnh `val`, không có `train`. Theo nguyên tắc đã chốt, **train-dev bắt buộc
lấy từ split `train`** — không được cắt val ra làm tập tuning.

Khi có `train`: công bố **trước** khi chạy — tập phát triển (~40 video train, seed 0,
`bdd100k_dev.json`), phạm vi tham số (`track_buffer_sec` ∈ {0,4; 0,8; 1,2; 2,0} và lưới C3 với
`std_theta`/`std_omega` nhân hệ số {1, 2, 5}), ngân sách tìm kiếm, tiêu chí chọn (HOTA ở c25 + CI, giống
`PREREG_moc2.md`). **Khóa** lựa chọn trong `LOCK_moc2.md` **trước** khi chạy val.

**Không trùng sequence**: hai tập lấy từ **hai split khác nhau** của BDD (`train` vs `val`) ⇒ giao rỗng theo
cấu tạo. Vẫn kiểm lại bằng một phép giao trên hai manifest trước khi chạy đánh giá.

Hai kết quả báo cáo **riêng**; **không** gộp thành một kết luận "khả năng chuyển miền".

## 7. Việc còn lại theo thứ tự

1. **Tải gói MOT 2020 Labels** → `data/raw/bdd100k/labels/box_track_20/val/*.json` (bắt buộc).
2. `check_bdd100k.py --part val --make-manifest` phải exit 0, 0 lỗi schema, 0 ảnh thiếu.
3. Đối chiếu thống kê lớp và phân bố Δt thật với quy ước ở mục 5.
4. Chọn 1–3 sequence, **ghi ra `bdd100k_smoke_val.json` trước khi xem metric**; tạo cache
   `yolov8n-coco_conf0.05_smoke_bdd` (cần GPU — đi qua guard provenance checkpoint đã thêm ở review vòng 3).
5. Chạy `bdd_smoke_val.yaml` (CV + EKF), kiểm 5 điểm ở mục 5, rồi mới bàn tới Thí nghiệm 1.
6. Chỉ khi có split `train` mới mở Thí nghiệm 2.
