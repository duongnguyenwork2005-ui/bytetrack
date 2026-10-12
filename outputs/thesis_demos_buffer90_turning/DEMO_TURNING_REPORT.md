# Demo "xe rẽ thật sự" với `track_buffer=90`: KHÔNG tìm được ứng viên

Ngày: 2026-09-11 · Nhánh `fix/tracker-evaluation-correctness` · Env conda `khoaluan-mot` ·
Script: `src/demo/turning_case_buffer90.py` (dùng lại máy quét của `src/demo/single_case_buffer90.py`).

## 1. Kết luận

**Không có GT object nào trên 60 video train thỏa đồng thời "xe rẽ thật sự" và "EKF thắng CV ổn định".**
Vì vậy **không dựng** `demo_ekf_turning_buffer90.mp4`, ba keyframe `*_turn.png` hay `match_cost_at_key_frame.csv`
(tạo chúng sẽ ngụ ý có một demo hợp lệ). Không nới tiêu chí.

Luồng lọc với định nghĩa ở §3:

| Bước | Số lượng |
|---|---:|
| GT object trên 60 video train | 5 952 |
| Object có **đoạn rẽ thật sự** (≥ 30°, hai cửa sổ ổn định, tốc độ ≥ 3 px/frame, GT nhìn thấy ≥ 50 %, box không chạm biên) | **9** (10 đoạn rẽ) |
| Thời điểm CV thất bại (switched/lost, ID đã ổn định ≥ 5 frame) trên 9 object đó | 14 |
| … EKF giữ đúng ID ≥ 10 frame GT liên tiếp sau đó | 1 |
| … và CV thất bại kéo dài ≥ 10 frame (không nhấp nháy về ID cũ) | **0** |
| … và ổn định trước đó, `f_fail` trong cửa sổ quanh đoạn rẽ, dịch chuyển đạt | 0 |

Trên cả 10 đoạn rẽ thật sự, **ba tracker hành xử giống hệt nhau** (§5): 8 đoạn cả ba giữ ID suốt đoạn rẽ (kể cả các
frame không tracker nào ghép được, giống nhau ở cả ba), 1 đoạn cả ba đổi ID cùng frame (MVI_63562 GT 58, cua 148°), 1 đoạn cả ba mất dấu cùng lúc.
Không có khác biệt nào giữa motion model tại chỗ rẽ trên tập train ở cấu hình buffer=90.

Hai ứng viên duy nhất "đạt" khi **không** dùng hai điều kiện an toàn (GT nhìn thấy ≥ 50 %, box không chạm biên) đều là
**rẽ giả** — xem §4 — và bị loại đúng lý do, không phải vì nới/siết ngưỡng góc.

## 2. Xác nhận cấu hình

| | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|
| Kết quả đã lưu (train, 60 file, 0 rỗng, cùng danh sách video) | `sweep-cv-buf90` | `sweep-ekf_ctrv-buf90` | `sweep-ukf_ctrv-buf90` |
| YAML (`configs/buffer_sweep/`) | `cv_buf90.yaml` | `ekf_ctrv_buf90.yaml` | `ukf_ctrv_buf90.yaml` |
| `tracker_type` | `bytetrack` | `bytetrack_ekf_ctrv` | `bytetrack_ukf_ctrv` |
| `track_high/low/new_track_thresh` | 0,25 / 0,10 / 0,25 | 0,25 / 0,10 / 0,25 | 0,25 / 0,10 / 0,25 |
| `match_thresh` / `fuse_score` / `track_buffer` | 0,8 / true / **90** | 0,8 / true / **90** | 0,8 / true / **90** |

`diff` ba YAML chỉ khác `tracker_type`. Detector `models/yolov8n.pt` (SHA-256 `f59b3d83…83b36`), `conf=0,25`
(`config.YOLO_DEFAULT_CONF`), classes `[2,3,5,7]` (`config.YOLO_VEHICLE_CLASSES`, suy ra tự động cho model COCO trong
`baseline_track.py`). Ba lượt được tạo ở commit `02b0a02` (2026-09-06) bằng
`baseline_track.py --tracker-cfg configs/buffer_sweep/<model>_buf90.yaml`; lần chạy lại trong phiên này (cho demo
`thesis_demos_buffer90_single_case`) xác nhận `tracker.args.track_buffer = 90` ở runtime và bộ lọc
`KalmanFilterXYAH` / `EKFTrackerCTRV` / `UKFTrackerCTRV`. Không dùng test; không sửa thuật toán, detector, config,
dữ liệu hay kết quả tracking.

## 3. Định nghĩa đã dùng

### 3.1 "Xe rẽ thật sự" (từ GT; không dùng góc đổi hướng cũ của cửa sổ demo)

Hai cửa sổ ổn định **trước** và **sau** đoạn rẽ, mỗi cửa sổ `W = 10` frame GT liên tiếp (yêu cầu tối thiểu 5; dùng
10 để heading ít nhiễu hơn; độ nhạy với `W = 5` ở §4). Vector hướng trước `v1 = pos[a] − pos[a−W]`, vector hướng
sau `v2 = pos[b+W] − pos[b]`, `b − a ≤ 40` frame (đoạn rẽ dài tối đa 1,6 s). Đạt điều kiện khi, ở **mỗi** cửa sổ:

1. tốc độ trung vị ≥ **3 px/frame**;
2. `|v| ≥ 15 px` (vector quá nhỏ làm góc mất ý nghĩa);
3. GT nhìn thấy ≥ **50 %** ở mọi frame — vị trí GT khi xe bị che hoàn toàn là nội suy của người gán nhãn, heading từ
   đó không tin được (§4, MVI_40991 GT 27);
4. box GT **không chạm biên ảnh** (cách ≥ 2 px) — box bị cắt ở biên làm tâm box lệch, sinh "đổi hướng" giả (§4,
   MVI_40191 GT 92);

và góc giữa `v1`, `v2` ≥ **30°**. Mỗi object lấy các đỉnh góc lớn nhất không giao nhau. Đoạn rẽ = `[a, b]`.
CSV lưu `heading_before_deg`, `heading_after_deg`, `turn_angle_deg`, `speed_before/after_px`, `vec_before/after_px`,
`path_px`, `net_px`, `median_bbox_diag_px`, `net_over_diag`.

Điều kiện dịch chuyển cho cửa sổ demo `[a − W, max(b + W, f_fail + 25)]`: net ≥ 0,5 × đường chéo bbox trung vị,
path ≥ 30 px.

### 3.2 "EKF thắng CV" (giống `single_case_buffer90.py`)

CV `switched`/`lost` với ID đã ổn định ≥ 5 frame, thất bại **kéo dài** ≥ 10 frame (không quay về ID cũ — loại
nhấp nháy 1 frame); EKF có ID ổn định ≥ 5 frame trước đó, không ghép ID khác trong `[f_fail, f_fail+5]`, rồi giữ
đúng ID **≥ 10 frame GT liên tiếp**; CV và EKF không đổi ID trong 25 frame trước `f_fail`; `f_fail` nằm trong
`[a − W, b + W + 10]` (trong hoặc ngay sau đoạn rẽ, cho phép cả cửa sổ trước). UKF không dùng để lọc.

**Xếp hạng (cố định, không dùng đến vì không có ứng viên):** (1) `f_fail` trong/ngay sau đoạn rẽ trước, gần nhất
trước; (2) số frame EKF giữ đúng ID sau `f_fail` lớn nhất; (3) góc rẽ, rồi net/diag; (4) có che khuất.

## 4. Vì sao hai điều kiện an toàn là bắt buộc (và ảnh hưởng của chúng)

Không có (3) và (4), bộ phát hiện tìm thấy 149 object "rẽ" và đúng **2** ứng viên qua mọi bước còn lại. Cả hai là
rẽ giả:

| Ứng viên | "Góc rẽ" | Vì sao là giả |
|---|---|---|
| `MVI_40991` GT 27, `f_fail` 1727 (chính là case study ở `thesis_demos_buffer90_single_case/`) | 36° giữa cửa sổ [1721–1731] và [1746–1756] | Cửa sổ trước nằm **trong đoạn che hoàn toàn** (GT nhìn thấy 0–3 %, xe sau cột): heading GT nhảy −165° → −134° → −148° chỉ vì vị trí nội suy. Heading thật trước khi vào cột −165°, sau khi ra −160…−170°: xe **đi thẳng**. |
| `MVI_40191` GT 92, `f_fail` 642 | 32° giữa cửa sổ [601–611] và [636–646] | Cửa sổ trước là 5 frame đầu tiên của object, box GT **bị cắt ở biên phải ảnh** (x + w > 960) nên tâm box lệch, heading −120° không thật; từ frame 615 trở đi heading −142° → −152° → −146°: đổi hướng ~10°, dưới ngưỡng. Thất bại của CV ở 642 cũng không liên quan đến rẽ: detector gộp hai xe kề nhau (GT 92 và GT 91) thành một box 72×62 px trong 2 frame, CV bám box gộp còn EKF/UKF nhận box đúng. |

Bảng độ nhạy (`turn_sensitivity.csv`, `python src/demo/turning_case_buffer90.py` với các hằng số đổi):

| Biến thể bộ phát hiện rẽ | Object có rẽ | Thời điểm CV thất bại | EKF giữ ≥ 10 | + kéo dài | + ổn định | + trong cửa sổ rẽ | **Đạt** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **W=10, GT nhìn thấy ≥ 50 %, không chạm biên (CHÍNH)** | 9 | 14 | 1 | 0 | 0 | 0 | **0** |
| W=10, GT nhìn thấy ≥ 50 %, cho phép chạm biên | 128 | 211 | 4 | 1 | 1 | 1 | **1** — MVI_40191 GT92 (rẽ giả: biên ảnh) |
| W=10, không xét nhìn thấy, không chạm biên | 12 | 19 | 2 | 1 | 1 | 1 | **1** — MVI_40991 GT27 (rẽ giả: GT trong che khuất) |
| W=10, bỏ cả hai điều kiện an toàn | 149 | 242 | 6 | 2 | 2 | 2 | **2** — MVI_40191 GT92 (rẽ giả: biên ảnh); MVI_40991 GT27 (rẽ giả: GT trong che khuất) |
| W=5, GT nhìn thấy ≥ 50 %, không chạm biên | 47 | 68 | 3 | 0 | 0 | 0 | **0** |
| W=5, bỏ cả hai điều kiện an toàn | 531 | 462 | 11 | 4 | 3 | 2 | **2** — MVI_40191 GT92 (rẽ giả: biên ảnh); MVI_40991 GT27 (rẽ giả: GT trong che khuất) |

Kết luận "không tìm được" **không** phụ thuộc vào `W` (5 hay 10) hay vào việc siết góc; hai ứng viên chỉ xuất hiện
khi bỏ điều kiện an toàn và cả hai đều là rẽ giả. Điều kiện "không chạm biên" loại nhiều "đoạn rẽ" nhất (128 → 9
object với W=10): phần lớn góc ≥ 30° đo được trên UA-DETRAC xuất hiện khi xe **đang vào/ra khung hình** và box GT bị
cắt ở biên — tâm box dịch chuyển theo phần bị cắt, không phải theo xe. Với W=5 (mức tối thiểu của yêu cầu) và đủ
điều kiện an toàn, có 47 object rẽ nhưng vẫn **0** ứng viên đạt.

## 5. Mười đoạn rẽ thật sự và hành vi ba tracker (`turns_detected.csv`, `turns_tracker_chains.csv`)

| Video | GT | Đoạn rẽ `[a, b]` | Góc | Heading trước → sau | Tốc độ trước / sau (px/frame) | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---:|---|---|---|---|---|
| MVI_20032 | 3 | 49–51 | 33,7° | 75° → 109° | 4,0 / 3,7 | 3 | 3 | 3 |
| MVI_20034 | 26 | 427–457 | 30,4° | −112° → −143° | 3,7 / 3,2 | 153 | 155 | 155 |
| MVI_20034 | 40 | 695–716 | 33,7° | −112° → −146° | 4,6 / 3,1 | 366 | 368 | 369 |
| MVI_20062 | 21 | 499–535 | 32,0° | 75° → 107° | 3,1 / 8,7 | 274 | 274 | 266 |
| MVI_40152 | 45 | 1366–1401 | 53,5° | −61° → −7° | 3,2 / 7,6 | 452 (rớt 46 fr) | 459 (rớt 46 fr) | 448 (rớt 46 fr) |
| MVI_40152 | 45 | 1432–1460 | 32,5° | −23° → −55° | 4,8 / 3,0 | 452 (rớt 2) | 459 (rớt 2) | 448 (rớt 2) |
| MVI_40162 | 8 | 226 | 54,0° | 26° → 80° | 4,4 / 3,3 | 14 | 21 | 21 |
| MVI_40991 | 2 | 116–156 | 31,0° | −145° → −176° | 3,3 / 3,7 | 23 (rớt 24) | 23 (rớt 24) | 23 (rớt 24) |
| MVI_63544 | 13 | 456–466 | 30,7° | 146° → 177° | 4,4 / 4,1 | 224 (rớt 21) | 232 (rớt 21) | 231 (rớt 21) |
| MVI_63562 | 58 | 441–462 | **148,3°** | −22° → −171° | 42,0 / 13,9 | 359→369 (rớt 6) | 353→363 (rớt 6) | 354→364 (rớt 6) |

Chuỗi ID đo trong `[a − 10, b + 10]`; "rớt n" = n frame **không tracker nào** ghép được target với IoU ≥ 0,5 (giống nhau ở cả ba — do detector, không do motion model).
Cả 10 đoạn: **không có khác biệt giữa ba motion model tại chỗ rẽ**. Đoạn duy nhất có đổi ID (MVI_63562 GT 58, cua
gấp 148° ở 42 px/frame) thì cả ba đổi cùng frame 466 sau 6 frame mất detection.

14 thời điểm CV thất bại trên 9 object này (`candidates_turning.csv`): 9 lần **EKF cũng đổi ID cùng frame**, 3 lần
cả ba mất dấu (EKF giữ 0 frame), 1 lần CV nhấp nháy 1 frame rồi quay về ID cũ (MVI_63544 GT 13 @508, 42 frame sau
khi rẽ xong), 1 lần EKF chưa có ID ổn định. Không lần nào EKF giữ đúng ID ≥ 10 frame trong khi CV thất bại kéo dài.

## 6. Giới hạn và diễn giải

1. Tập train UA-DETRAC (60 video, camera cố định trên cao, chủ yếu đường thẳng/nút giao xa) có **rất ít** xe rẽ ≥ 30°
   đo được tin cậy: 9 object / 5 952. Đây là giới hạn dữ liệu, đã ghi nhận từ Giai đoạn 5
   (`stratum-curved-long-occlusion-qua-nho`).
2. Ở 10 đoạn rẽ đó ba tracker cho cùng kết cục — không có bằng chứng **theo hướng nào** về motion model tại chỗ rẽ.
3. Ngưỡng 30° và hai điều kiện an toàn là chọn trước; §4 cho thấy nới chúng chỉ sinh ứng viên giả. `W = 10` thay vì
   5 không đổi kết luận (§4).
4. Kết quả buffer=90 có trước các sửa lỗi trên nhánh này; không ảnh hưởng kết luận vì không có ứng viên để tái lập.
5. Case study xe đi thẳng qua che khuất (`outputs/thesis_demos_buffer90_single_case/`) vẫn là demo gần nhất hiện có;
   nó **không** minh họa rẽ.

## 7. Artefact và tái tạo

```
outputs/thesis_demos_buffer90_turning/
├── DEMO_TURNING_REPORT.md        (file này)
├── candidates_turning.csv        14 thời điểm CV thất bại trên object có rẽ thật: heading trước/sau, góc, tốc độ, path, net, kết cục CV/EKF/UKF, lý do loại
├── turns_detected.csv            10 đoạn rẽ thật sự
├── turns_tracker_chains.csv      chuỗi ID ba tracker qua từng đoạn rẽ
├── turn_sensitivity.csv          độ nhạy kết luận theo W / vis / biên ảnh
└── metadata.json                 cấu hình, định nghĩa, số đếm, danh sách đầu ra KHÔNG tạo và lý do
src/demo/turning_case_buffer90.py scan | build   (build tự dừng: "Khong co ung vien hang 1")
```
Không tạo: `demo_ekf_turning_buffer90.mp4`, `before_turn.png`, `during_turn.png`, `after_turn.png`,
`match_cost_at_key_frame.csv`. Không chạy lại tracker (không có ứng viên để kiểm chứng).

```bash
conda activate khoaluan-mot
python src/demo/turning_case_buffer90.py scan     # ~6 phút, chỉ đọc GT + prediction đã lưu
python src/demo/turning_case_buffer90.py build    # dừng với thông báo không có ứng viên
```
Không commit, không push. `.gitignore` đã có quy tắc bỏ `*.mp4`, `*.png`, `rerun/` cho thư mục này.
