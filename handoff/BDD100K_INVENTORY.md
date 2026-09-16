# Kiểm kê BDD100K-MOT trên máy — trạng thái và phần còn thiếu

Kiểm ngày **2026-09-17**, máy `D:\LABOTARY\bytetrack`, nhánh `phase2-mixed-traffic`.

## 1. Kết luận: **chưa có dữ liệu BDD100K thật trên máy**

```
$ python src/check_bdd100k.py --part val                                   # exit 1
[ERROR] Khong co file JSON nao trong D:\LABOTARY\bytetrack\data\raw\bdd100k\labels\box_track_20\val.
```

Đã tìm toàn bộ ổ `C:` và `D:` (độ sâu 3): thư mục duy nhất khớp là
`D:\LABOTARY\ByteTrack + EKF + Kitty\datasets\BDD100K` — **rỗng** (0 json, 0 jpg, 0 byte).

⇒ **Không chạy được smoke BDD100K** (mục 3 của đợt này). Fixture tổng hợp
`data/fixtures/bdd100k_mini` **không** được dùng thay thế: chính README của nó ghi *"Chỉ để kiểm tra
pipeline; KHÔNG dùng để báo cáo kết quả."*

## 2. Phần mềm: **đã sẵn sàng**, không còn việc chuẩn bị nào làm được nếu thiếu dữ liệu

| Thành phần | File | Trạng thái |
|:--|:--|:--|
| Adapter BDD100K-MOT (Scalabel `box_track_20`) | [src/mot_datasets/bdd100k_mot.py](../src/mot_datasets/bdd100k_mot.py) | ✅ Đọc JSON → schema interim chung với UA-DETRAC; `check_schema()` **fail-loud** khi gặp schema cũ (`index`, `attributes.Crowd` của TrackEval 1.3.0) |
| Script kiểm tra dữ liệu | [src/check_bdd100k.py](../src/check_bdd100k.py) | ✅ Kiểm schema → ảnh → thống kê lớp → phân bố Δt → sinh manifest (`--make-manifest --n-dev 40 --seed 0`) |
| Thí nghiệm 1 (chuyển cấu hình) | [configs/phase2/experiments/bdd_transfer_val.yaml](../configs/phase2/experiments/bdd_transfer_val.yaml) | ✅ Viết xong, chưa chạy |
| Thí nghiệm 2 (tuning trên train-dev) | [configs/phase2/experiments/bdd_tuned_val.yaml](../configs/phase2/experiments/bdd_tuned_val.yaml) | ✅ Khung + quy trình viết trước, chờ điền giá trị đã khóa |
| Smoke end-to-end trên fixture | [configs/phase2/experiments/smoke_bddfx.yaml](../configs/phase2/experiments/smoke_bddfx.yaml), [src/test_smoke_phase2.py](../src/test_smoke_phase2.py) | ✅ **11/11 PASS** — cache gt-synthetic → 4 tracker → TrackEval crowd-ignore → occlusion_eval (gap, 5 fps) → aggregate |
| Test adapter | [src/test_bdd100k_adapter.py](../src/test_bdd100k_adapter.py) | ✅ **42/42 PASS** |
| Test Δt → dự đoán/vòng đời | [src/test_timestep_integration.py](../src/test_timestep_integration.py) | ✅ **32/32 PASS** (`elapsed_mode: steps`) |

**Những gì đã được xác minh bằng phần mềm** (fixture, không phải kết quả thực nghiệm): ánh xạ lớp
COCO → BDD, xuất MOTChallenge theo lớp, vùng bỏ qua `crowd` + 3 lớp distractor, `elapsed_mode: steps`, và
`occlusion_eval --gaps` **không** suy diễn khoảng trống annotation thành che hoàn toàn.

**Những gì CHƯA thể xác minh nếu không có dữ liệu thật**: phân bố Δt thật từ `timestamp`, tỉ lệ khoảng lệch
nhịp, thống kê lớp thật (đặc biệt `motorcycle`/`bicycle`/`rider` — phần liên quan nhất tới giao thông hỗn
hợp), số vùng `crowd` thật, và mọi số liệu tracking.

## 3. Đầu vào còn thiếu — gói, nguồn, vị trí đặt, lệnh kiểm tra

Đăng ký tài khoản tại **bdd-data.berkeley.edu** (nguồn chính thức, BDD100K của BAIR/Berkeley DeepDrive) rồi
tải **hai** gói của phần **MOT 2020**:

| Gói cần tải | Nội dung | Giải nén sao cho có đường dẫn |
|:--|:--|:--|
| **MOT 2020 Labels** | nhãn tracking `box_track_20` (train + val) | `data/raw/bdd100k/labels/box_track_20/{train,val}/<video>.json` |
| **MOT 2020 Images** | ảnh 5 fps, 1280×720 | `data/raw/bdd100k/images/track/{train,val}/<video>/<video>-<7 chữ số>.jpg` |

Gói `test` **không cần**: BDD không công bố nhãn test.

Kiểm tra sau khi giải nén, theo đúng thứ tự:

```bash
P=/c/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe

# 1. Schema + ảnh + thống kê lớp + phân bố Δt, đồng thời sinh manifest final (val)
$P src/check_bdd100k.py --part val   --make-manifest
#    -> results/phase2/bdd100k_check/val_{class_stats,dt_stats}.csv, val_schema_report.json
#    -> results/phase2/manifests/bdd100k_final_val.json

# 2. Manifest train-dev (~40 video, seed 0) - CHỈ dùng để tuning, KHÔNG dùng đánh giá
$P src/check_bdd100k.py --part train --make-manifest --n-dev 40 --seed 0
#    -> results/phase2/manifests/bdd100k_dev_train.json

# 3. Cache detection riêng cho BDD (CẦN GPU)
$P src/det_cache.py --dataset bdd100k --part val \
   --manifest results/phase2/manifests/bdd100k_final_val.json \
   --cache-id yolov8n-coco_conf0.05_all_bdd

# 4. Smoke 1-3 sequence trước khi chạy cả split
$P src/run_experiment.py --exp configs/phase2/experiments/bdd_transfer_val.yaml --videos <v1> <v2> <v3>
```

Bước 1 phải **exit 0** và báo `[schema] N/N video hợp lệ; 0 lỗi` trước khi làm bước sau. Nếu `check_schema`
báo lỗi, **dừng** — đó là dấu hiệu tải nhầm phiên bản schema, không được sửa vòng.

## 4. Split nào dùng cho gì (đã chốt, giữ nguyên từ mốc 1)

| Vai trò | Split BDD | Ghi chú |
|:--|:--|:--|
| Phát triển / tuning | **~40 video chọn từ `train`** (`bdd100k_dev_train.json`, seed 0) | công bố trước; chỉ dùng cho Thí nghiệm 2 |
| Đánh giá cuối | **toàn bộ `val`** công khai (`bdd100k_final_val.json`) | chỉ chạy **sau khi khóa** |

**Không trùng sequence**: hai tập lấy từ **hai split khác nhau** của BDD (`train` vs `val`), nên giao là
rỗng theo cấu tạo. Khi có dữ liệu, kiểm lại bằng một phép giao trên hai manifest trước khi chạy đánh giá.

## 5. Ánh xạ lớp, thuộc tính và vùng bỏ qua — đã định nghĩa, chờ dữ liệu để xác nhận

| Hạng mục | Quy ước đã chốt |
|:--|:--|
| Lớp đánh giá | 8 lớp BDD: pedestrian, rider, car, truck, bus, train, motorcycle, bicycle |
| Ánh xạ detector | COCO → BDD: `1→bicycle, 2→car, 3→motorcycle, 5→bus, 7→truck`. **`person`(0) KHÔNG map** vì COCO gộp pedestrian + rider |
| Mục tiêu chính | `vehicle` = car + truck + bus + motorcycle + bicycle → class 1 (liên tục với UA-DETRAC) |
| Vùng bỏ qua | mọi box `crowd` (bất kỳ lớp), 3 lớp distractor (*other person, trailer, other vehicle*), và `train` khi target = vehicle → class 13, conf 0 |
| `occluded` | **cờ nhị phân** "bị che một phần", **không** phải mức độ che → interim gán `occlusion_ratio` = 0,5 (vượt ngưỡng partial 0,10, **không bao giờ** vượt ngưỡng full 0,90) |
| Khoảng trống annotation (**gap**) | **KHÔNG** suy diễn thành che hoàn toàn. `occlusion_eval --gaps` tạo sự kiện gap với `has_full=False` và `gap_evidence ∈ {occluded_adjacent, unknown}`, báo cáo **tách tầng** |
| Thời gian | 5 fps danh nghĩa; Δt thực lấy từ `timestamp` (ms) → `n_steps`; bộ lọc vẫn ở **đơn vị bước** (dt = 1); `track_buffer_sec` 1,2 s = 6 bước @ 5 fps (cùng **thời gian** giữ track với DETRAC 30 bước @ 25 fps) |
| `rider` | box **riêng** với xe hai bánh, không gộp; chỉ thống kê, **không** đánh giá (detector COCO không có lớp này) |

⚠️ Điểm **chưa xác nhận trên dữ liệu thật**: trường `timestamp` có mặt ở mọi frame hay không (adapter có
đường dự phòng theo `frameIndex`), và số video có khoảng lệch nhịp. `check_bdd100k.py` in đúng hai con số này
(`dt_source`, `n_intervals_irregular`) ở bước 1.

## 6. Kế hoạch đánh giá — **hai thí nghiệm tách biệt, không gộp kết luận**

### A. Chuyển cấu hình UA-DETRAC → BDD100K (`bdd_transfer_val.yaml`)
Giữ **nguyên** tham số nghiên cứu đã khóa trên UA-DETRAC (cùng YAML tracker, cùng `match_thresh`, cùng
`std_theta`/`std_omega` **theo bước**), xử lý thời gian theo quy ước pipeline. **Không tuning bằng nhãn BDD.**
Điểm phải ghi rõ khi báo cáo: `std_theta`/`std_omega` là **theo bước**, mà một bước ở BDD dài gấp 5 lần ở
DETRAC (0,2 s so với 0,04 s) — thí nghiệm này **cố ý** giữ nguyên giá trị theo bước, đúng nghĩa "chuyển cấu
hình"; **không** quy đổi bằng công thức.

### B. Tuning trên BDD train-dev rồi đánh giá trên val (`bdd_tuned_val.yaml`)
Công bố **trước** khi chạy: tập phát triển (~40 video train, seed 0), phạm vi tham số
(`track_buffer_sec` ∈ {0,4; 0,8; 1,2; 2,0} và lưới C3 với `std_theta`/`std_omega` nhân hệ số {1, 2, 5}),
ngân sách tìm kiếm, và tiêu chí chọn (HOTA ở c25 + CI, giống `PREREG_moc2.md`). **Khóa** lựa chọn bằng một
mục mới trong `LOCK_moc2.md` **trước** khi chạy val.

Hai kết quả báo cáo **riêng**; **không** gộp thành một kết luận "khả năng chuyển miền".

**Chưa chạy benchmark BDD đầy đủ trong đợt này** (thiếu dữ liệu).

## 7. Việc còn lại theo thứ tự

1. Tải 2 gói MOT 2020 (Labels + Images) → đặt đúng `data/raw/bdd100k/...`.
2. `check_bdd100k.py --part val --make-manifest` phải exit 0.
3. Đối chiếu thống kê lớp thật với bảng ở mục 5 (nhất là `motorcycle`/`bicycle`) và phân bố Δt thật.
4. Tạo cache `yolov8n-coco_conf0.05_all_bdd` (cần GPU) — cache mới đi qua guard provenance checkpoint đã
   thêm ở review vòng 3 ⇒ `meta.json` sẽ mang `weights_sha256` của đúng checkpoint được nạp.
5. Smoke 1–3 sequence, kiểm 5 điểm ở mục 5, rồi mới chạy Thí nghiệm 1.
