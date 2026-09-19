# Bản khóa — Thí nghiệm chuyển cấu hình UA-DETRAC → BDD100K val

Ghi **trước** khi xem bất kỳ kết quả đánh giá nào của thí nghiệm này.
Bản máy đọc: `results/phase2/lock/lock_bdd_transfer.json` (commit, sha256 từng YAML, GT/seqmap,
checkpoint, sequence, backend, cách tổng hợp).

## 1. Câu hỏi và phạm vi

**Câu hỏi**: cấu hình đã khóa trên UA-DETRAC (`configs/phase2/LOCK_moc2.md`) chuyển sang BDD100K val
thì hành xử thế nào?

**Không** phải câu hỏi "bộ lọc nào tốt nhất trên BDD" — trả lời câu đó cần tuning trên BDD train-dev
(Thí nghiệm 2), hiện **chưa đủ điều kiện** (không có split `train`).

**Không đặt trước kết luận EKF phải tốt hơn CV.** Kết quả không cải thiện hoặc kém hơn vẫn được giữ
và giải thích theo bằng chứng.

## 2. Ma trận đã khóa

| | |
|:--|:--|
| Bộ lọc | **CV**, **EKF–CTRV**, **UKF–CTRV**, **PF–CTRV N=500** (seed **0, 1, 2**) |
| Điều kiện confidence | **c25** (conf > 0,25) và **c10** (conf > 0,10) |
| Tổng số lần chạy | **12** (6 tracker-run × 2 điều kiện) × **200 video** |
| Detector | `models/yolov8n.pt`, sha256 `f59b3d83…` — **cùng checkpoint** với UA-DETRAC |
| Detection cache | `yolov8n-coco_conf0.05_all_bddcocovid` (conf 0,05, 80 lớp COCO, `max_det` 3000) — **một** cache cho **cả hai** điều kiện, lọc lúc load ⇒ mọi tracker thấy **cùng** detection |
| Lớp đưa vào tracker | COCO `[1, 2, 3, 5, 7]` = bicycle, car, motorcycle, bus, truck |
| `max_det` (tracker) | 300 |
| Association | ByteTrack đã khóa (`match_thresh` 0,8, `fuse_score`, 2 vòng) — **không đổi** |
| Thời gian giữ track | **1,2 s** = 6 bước @ 5 fps (cùng **thời gian** với DETRAC 30 bước @ 25 fps) |
| Backend tracker | **`ultralytics`** cho toàn bộ ma trận |

**Vì sao dùng backend Ultralytics**: phạm vi backend NumPy đã kiểm chứng hiện chủ yếu là CV/EKF
(UKF/PF chưa mở). Dùng NumPy cho một phần ma trận sẽ **đổi đồng thời backend và bộ lọc** trong cùng
một phép so sánh — không làm.

## 3. Quy ước thời gian — điểm phải nói rõ khi báo cáo

Bộ lọc làm việc ở **đơn vị bước** (dt = 1). Thí nghiệm này **giữ nguyên con số tham số nhiễu theo
bước** của UA-DETRAC (đúng nghĩa "chuyển cấu hình"), nhưng:

| | một bước | fps |
|:--|:--|--:|
| UA-DETRAC | **0,04 s** | 25 |
| BDD100K | **0,2 s** | 5 |

⇒ một bước ở BDD **dài gấp 5 lần**. Vì vậy đây **KHÔNG** phải "giữ nguyên mô hình nhiễu theo đơn vị
vật lý"; nó là giữ nguyên **con số theo bước**. **Không** tự thêm cách quy đổi hay tuning nào trong
thí nghiệm này.

## 4. Nguồn nhãn và giới hạn

Nguồn: `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` — **COCO-video chuyển đổi**,
sha256 `074ff795…`, **không phải** `box_track_20` native.

| Thiếu trong nguồn | Hệ quả đã khóa |
|:--|:--|
| `attributes.occluded` | `occlusion_ratio`/`visibility` = **NaN**; **tắt** phân tích che khuất; **không** biến khoảng annotation trống thành sự kiện che hoàn toàn |
| `attributes.truncated` | ghi `None` |
| `timestamp` | Δt suy ra từ frame index với **5 fps danh nghĩa** (`elapsed_mode: steps` vẫn tính đúng khoảng frame bị bỏ qua; kiểm kê cho thấy val **không** có video nào có khoảng bỏ qua) |

`visibility = 1.0` trong `gt.txt` là **placeholder** do định dạng MOTChallenge bắt buộc — nghĩa là
"không biết".

## 5. Tập dữ liệu

**Toàn bộ 200 video val**, 39.973 ảnh — **không loại video nào**. Kiểm kê:
`results/phase2/bdd_cocovid_check/val_report.json` (0 orphan, 0 sai tên file, 0 ảnh thiếu, 0 bbox
lỗi, 0 video có khoảng frame bỏ qua).

⚠️ **Ba sequence đã được xem trước**: `b1c66a42-6f7d68ca`, `b1c81faa-3df17267`, `b1c81faa-c80764c5`
— đã xem metric ở bước smoke (review vòng 5–6). **Không** mô tả toàn bộ val là tập hoàn toàn chưa
xem. Ba sequence này vẫn nằm trong tập đánh giá (loại chúng ra sẽ là một lựa chọn sau khi đã thấy
số liệu).

## 6. Cách chấm điểm — khóa trước

| | |
|:--|:--|
| Evaluator | `mot_crowd` — chọn theo **loại dataset** (`bdd100k_coco`), không theo tên split |
| GT / seqmap | split **riêng** `BDDCOCOVID-val`; `seqLength` lấy từ **timeline ảnh** |
| Mục tiêu đánh giá | `vehicle` = car + truck + bus + motorcycle + bicycle → **một** lớp (class 1) |
| Vùng bỏ qua | mọi box `crowd`/`ignore`, 3 lớp distractor, và `train` → class 13, conf 0 |
| COMBINED | TrackEval `COMBINED_SEQ`, tái tổ hợp từ thống kê đủ (`src/boot_stats.py`) |
| Bootstrap | ghép cặp **theo video**, **5000** lần, **seed 0** |
| PF ba seed | **bảng chính**: trung bình **metric** theo seed; **bootstrap**: trung bình **thống kê đủ** rồi tái tổ hợp (hai đại lượng khác nhau với metric phi tuyến — `PHASE2_REPORT.md` mục 15.2) |

**Cặp so sánh đã khóa** (trong YAML): `ekf − cv`, `ukf − cv`, `ukf − ekf`, `pf500 − cv`, `pf500 − ekf`.

## 7. Điều không làm trong thí nghiệm này

- Không tuning bằng BDD val; không đổi tham số sau khi xem metric.
- Không báo chỉ số phục hồi sau che khuất (nguồn thiếu thuộc tính).
- Không báo metric riêng từng lớp: GT gộp 5 lớp xe thành một lớp, và đường tách lớp + chính sách
  ignore theo lớp **chưa được kiểm chứng** trên nguồn này.
- Không benchmark Nano, không training detector, không Thí nghiệm 2 (tuning trên train-dev).
