# Phụ lục — Thí nghiệm chuyển cấu hình UA-DETRAC → BDD100K val

Tài liệu **phương pháp và giới hạn**. Số liệu nằm ở `results/phase2/bdd_transfer_cocovid_val/bao_cao/`.
Bản khóa: `configs/phase2/LOCK_bdd_transfer.md` (người đọc) + `results/phase2/lock/lock_bdd_transfer.json`
(máy đọc, ghi lúc `2026-09-20T03:49:13`, commit `d1f6d62`, **trước** mọi lần xem kết quả).

## 1. Câu hỏi — và câu hỏi này *không* trả lời

**Trả lời**: cấu hình đã khóa trên UA-DETRAC, đem nguyên sang BDD100K val, cho kết quả thế nào.

**Không trả lời**: "bộ lọc nào tốt nhất trên BDD". Câu đó cần tuning trên BDD train-dev (Thí nghiệm 2) —
hiện **chưa đủ điều kiện**: máy chỉ có split `val`, không có `train`.

Không đặt trước kết luận EKF phải tốt hơn CV. Kết quả kém hơn vẫn được giữ và báo cáo.

## 2. Nguồn nhãn — điểm phải nói rõ

| | |
|:--|:--|
| File | `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` |
| sha256 | `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d` |
| Kích thước | 117.094.032 byte |
| Định dạng | **COCO-video đã chuyển đổi** — **không phải** `box_track_20` native |
| Adapter | `src/mot_datasets/bdd100k_coco.py` (**riêng**, không ép qua adapter native) |

**Thiếu so với native**: `attributes.occluded`, `attributes.truncated`, `timestamp`.

Hệ quả đã khóa trước:

- `occlusion_ratio` / `visibility` = **NaN**; **tắt** toàn bộ phân tầng che khuất. Khoảng annotation
  trống **không** được biến thành "sự kiện che hoàn toàn" — đó sẽ là bằng chứng bịa.
- `visibility = 1.0` trong `gt.txt` là **placeholder** do định dạng MOTChallenge bắt buộc có cột đó.
  Nó nghĩa là **"không biết"**, không phải "nhìn thấy đầy đủ".
- Δt suy ra từ chỉ số frame với **5 fps danh nghĩa**.

**Bẫy của nguồn này**: chỉ có **392** giá trị `instance_id` cho toàn bộ val, và **327** trong số đó
xuất hiện ở **nhiều video**. Đánh số track theo `instance_id` trần sẽ **trộn** các đối tượng khác nhau
giữa các video. Adapter khóa ID theo cặp **`(video, instance_id)`**.

## 3. Kiểm kê trước khi khóa

`python src/check_bdd100k_cocovid.py` → `results/phase2/bdd_cocovid_check/val_report.json`:

| | |
|:--|--:|
| Video | **200** / kỳ vọng 200 |
| Ảnh | **39.973** / kỳ vọng 39.973 |
| Annotation | 465.512 |
| Ảnh thiếu trên đĩa | **0** |
| `file_name` sai quy ước | **0** |
| Video có khoảng frame bị bỏ qua | **0** |
| bbox w/h ≤ 0 | **0** |
| Ảnh **không** có annotation | 615 (1,54 %) — vẫn thuộc timeline hợp lệ |
| `iscrowd` / `ignore` | 22.690 / 4.065 (**đếm riêng**, không cộng gộp) |

**Không video nào bị loại** khỏi tập đánh giá.

## 4. Vì sao `seqLength` lấy từ ảnh, không lấy từ annotation (F4)

Bằng chứng trên video thật `b23b2649-8349d2a1`:

| | |
|:--|--:|
| Ảnh trên đĩa | **202** |
| Frame cuối **có annotation** | **94** |
| `seqLength` đã ghi | **202** |

Lấy độ dài từ frame annotation cuối sẽ ghi `seqLength=94`, tức **sai độ dài thật** của sequence.

Điều đã kiểm chứng trên `TrackEval 1.3.0` (`datasets/mot_challenge_2d_box.py`, dòng 222–224): loader
**kiểm tra timestep nằm ngoài `seqLength`** rồi **ném `TrackEvalException`**
(`"Tracking data contains the following invalid timesteps in seq ..."`). Vì vậy phát biểu đúng là:

- độ dài 94 là **sai**; timeline phải giữ đến **202**;
- nếu tracker có kết quả ở frame vượt độ dài đã khai, evaluator **có thể từ chối dữ liệu** (báo lỗi),
  chứ **không** âm thầm bỏ qua.

> Bản trước của tài liệu này viết rằng mọi FP sau frame 94 sẽ "biến mất khỏi phép đánh giá" và gọi đó
> là "làm đẹp số liệu" — **đã rút lại** (review `ca93c1e`, mục 4): điều đó không được chứng minh, và
> hành vi thực tế của loader là **báo lỗi**.

`seq_lengths_from_images()` lấy độ dài từ **timeline ảnh**; CLI trả mã lỗi 2 nếu thiếu ảnh.

## 4b. Quy mô ma trận

**6 cấu hình/seed × 2 điều kiện = 12 run tổng cộng**, mỗi run chạy trên **200 video**.
Sáu cấu hình/seed là: `cv`, `ekf`, `ukf`, `pf500-s0`, `pf500-s1`, `pf500-s2`.

Pilot là **12 run riêng** trên **3 video** (`bdd_pilot_cocovid`), không nằm trong con số trên.

## 5. Quy ước thời gian — chỗ dễ đọc sai nhất

Bộ lọc làm việc ở **đơn vị bước** (dt = 1). Thí nghiệm giữ nguyên **con số tham số nhiễu theo bước**
của UA-DETRAC — đúng nghĩa "chuyển cấu hình". Nhưng:

| | một bước | fps |
|:--|:--|--:|
| UA-DETRAC | **0,04 s** | 25 |
| BDD100K | **0,2 s** | 5 |

⇒ một bước ở BDD **dài gấp 5 lần**. Vì vậy đây **KHÔNG** phải "giữ nguyên mô hình nhiễu theo đơn vị
vật lý". Cùng một `std_weight_velocity` mô tả một mức nhiễu quá trình **khác hẳn** khi bước dài gấp 5.
Không có cách quy đổi nào được thêm vào — việc chọn lại tham số thuộc Thí nghiệm 2.

`track_buffer_sec` 1,2 s = **6 bước** @ 5 fps, cùng **thời gian** giữ track với DETRAC (30 bước @ 25 fps).

## 6. Backend

Toàn ma trận chạy backend **`ultralytics`** (8.4.141). Phạm vi backend NumPy đã kiểm chứng hiện chủ yếu
là CV/EKF. Dùng NumPy cho một phần ma trận sẽ **đổi đồng thời backend và bộ lọc** trong cùng một phép
so sánh — không làm.

## 7. Chấm điểm

| | |
|:--|:--|
| Evaluator | `mot_crowd`, chọn theo **loại dataset** (`bdd100k_coco`), không theo tên thư mục |
| Split | **riêng**: `BDDCOCOVID-val` (tách khỏi `BDDSMOKE-val` và `BDD-val`) |
| Mục tiêu | `vehicle` = car + truck + bus + motorcycle + bicycle → **một** lớp |
| Vùng bỏ qua | mọi box `crowd`/`ignore`, 3 lớp distractor, và `train` → class 13, conf 0 |
| Bootstrap | ghép cặp **theo video**, **5000** lần, **seed 0** |
| PF ba seed | bảng chính: trung bình **metric**; bootstrap: trung bình **thống kê đủ** rồi tái tổ hợp |

## 8. Ba sequence đã xem trước

`b1c66a42-6f7d68ca`, `b1c81faa-3df17267`, `b1c81faa-c80764c5` đã được xem metric ở bước smoke
(review vòng 5–6) và ở pilot. Chúng **vẫn nằm trong** tập đánh giá. Vì vậy **không** mô tả toàn bộ val
là tập hoàn toàn chưa xem. Loại chúng ra *sau khi* đã thấy số liệu sẽ là một lựa chọn hậu nghiệm.

## 9. Điều không làm

- Không tuning bằng BDD val; không đổi tham số sau khi xem metric.
- Không báo chỉ số phục hồi sau che khuất (nguồn thiếu thuộc tính).
- Không báo metric **riêng từng lớp**: GT gộp 5 lớp xe thành một, và đường tách lớp + chính sách ignore
  theo lớp **chưa được kiểm chứng** trên nguồn này.
- Không benchmark Nano, không training detector, không Thí nghiệm 2.
