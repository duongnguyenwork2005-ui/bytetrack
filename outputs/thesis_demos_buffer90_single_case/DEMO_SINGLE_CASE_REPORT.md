# Case study `track_buffer=90`: một video xe đang di chuyển, EKF+CTRV giữ ID lâu hơn KF+CV

Ngày: 2026-09-11 · Nhánh `fix/tracker-evaluation-correctness` · Env conda `khoaluan-mot` (ultralytics 8.4.141,
torch 2.6.0+cu124, GTX 1650, ffmpeg 7.1 qua `imageio-ffmpeg`). Script: `src/demo/single_case_buffer90.py`.

## 1. Kết luận

**Tìm được đúng một case study đạt toàn bộ tiêu chí cứng và xếp hạng cao nhất:**

| | |
|---|---|
| Video / GT ID | `MVI_40991` (train), GT **27** — xe taxi vàng chạy chéo từ phải-dưới lên trái-trên, đi qua sau cột cầu vượt |
| Cửa sổ demo (frame nguồn) | **1702 – 1777** (76 frame, 3,04 s @25 fps); GT tồn tại 1676 – 1806 |
| Frame quyết định | **1727**: xe xuất hiện lại sau **21 frame** (1706 – 1726, 0,84 s) bị che hoàn toàn (GT nhìn thấy 0 – 3 %) |
| KF + CV | ID **365 → 376** (switched): tạo track mới khi xe hiện lại; giữ ID gốc **0** frame sau 1727 |
| EKF + CTRV | ID **379 → 379**: nối lại đúng track cũ; giữ đúng ID **34 frame liên tiếp** (1727 – 1760), **57 frame** tới hết đời track (chỉ ngắt bởi detector rớt 1 frame ở 1761, 1763 và 5 frame 1784 – 1788 — rớt ở cả ba tracker) |
| UKF + CTRV (hiển thị, không dùng để lọc) | ID **378 → 378**: giống EKF, 34 / 57 frame |
| Chuyển động (GT, cửa sổ 1702 – 1752) | tốc độ trung vị **5,27 px/frame**, path **335,2 px**, net **326,7 px**, đường chéo bbox trung vị 125,4 px → net/diag **2,60**; trong 21 frame bị che xe đi 184 px (~9 px/frame) |
| Che khuất | **có** — che hoàn toàn (`full_occlusion_segments.csv`: 1711 – 1731, 21 frame; `occlusion_segments.csv`: 1676 – 1731, max 1,00) |
| Đổi hướng | **không** (−3,8° trên cửa sổ; +12° đo riêng trong đoạn bị che, dưới ngưỡng 10° của cửa sổ) |
| Đầu ra | `demo_ekf_wins_cv_buffer90.mp4` 1920×1080 H.264 (libx264 High, yuv420p), 12 fps, 118 frame, **9,83 s** |

Đây là **một trường hợp được chọn có chủ đích** theo tiêu chí "EKF thắng CV". Phép quét đối xứng (§8) cho thấy
chiều ngược lại (EKF thất bại, CV giữ ID) xảy ra với tần suất tương đương trên cùng dữ liệu; case study này
**không** là bằng chứng EKF vượt trội nói chung. Lưu ý thêm: lượt A đã lưu (buffer=30) cho cùng kiểu kết cục trên chính
object này (§9.8) — clip minh họa khác biệt ở tái nhận của motion model, không phải tác dụng của buffer=90.

## 2. Xác nhận cấu hình ba kết quả

| Thuộc tính | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|
| Kết quả đã lưu | `data/processed/trackers/DETRAC-all/sweep-cv-buf90/data/` | `…/sweep-ekf_ctrv-buf90/data/` | `…/sweep-ukf_ctrv-buf90/data/` |
| Số file / rỗng | 60 / 0 | 60 / 0 | 60 / 0 |
| YAML | `configs/buffer_sweep/cv_buf90.yaml` | `ekf_ctrv_buf90.yaml` | `ukf_ctrv_buf90.yaml` |
| `tracker_type` | `bytetrack` | `bytetrack_ekf_ctrv` | `bytetrack_ukf_ctrv` |
| `track_high_thresh` / `track_low_thresh` / `new_track_thresh` | 0,25 / 0,10 / 0,25 | 0,25 / 0,10 / 0,25 | 0,25 / 0,10 / 0,25 |
| `match_thresh` / `fuse_score` / `track_buffer` | 0,8 / true / **90** | 0,8 / true / **90** | 0,8 / true / **90** |
| Bộ lọc thật khi chạy lại (`type(tracker.kalman_filter)`) | `KalmanFilterXYAH` | `EKFTrackerCTRV` | `UKFTrackerCTRV` |

- `diff` ba YAML: chỉ khác dòng `tracker_type` (và chú thích) — ba lượt chỉ khác **motion model**.
- Detector: `models/yolov8n.pt`, SHA-256 `f59b3d83…83b36`; `conf = 0,25` (`config.YOLO_DEFAULT_CONF`); classes `[2, 3, 5, 7]`
  (`config.YOLO_VEHICLE_CLASSES`, suy ra tự động cho model COCO trong `baseline_track.py`); dataset `DETRAC-all` = 60 video **train**.
  Tham số runtime được đọc lại từ `tracker.args` trong lần chạy lại và ghi vào `metadata.json → configuration.tracker_runtime`.
- Nguồn gốc: 3 lượt được tạo ở commit `02b0a02` (2026-09-06, "Thuc nghiem toan tap o track_buffer=90") bằng
  `baseline_track.py --tracker-cfg configs/buffer_sweep/<model>_buf90.yaml`. Đây là cấu hình Stage E mở rộng
  (`conf=0,25`, `track_buffer=90`), **không** phải lượt A/B chính thức. Kết quả có trước các sửa lỗi filter/tracker trên
  nhánh này (`a20edd8`, `aa45f6c`); với video này lần chạy lại bằng mã hiện tại vẫn tái lập 100 % (§6).
- Không dùng test. Không sửa thuật toán, detector, YAML, dữ liệu hay kết quả tracking.

## 3. Cách quét (toàn bộ tracking train, không giới hạn trong sự kiện che khuất)

Lý do mở rộng: audit trước (`outputs/thesis_demos_buffer90_moving/`) chỉ xét 1 003 sự kiện che khuất "đủ điều kiện"
và không tìm được xe đang di chuyển. Chính object được chọn ở đây (**MVI_40991 GT 27**) không nằm trong tập đó vì
đoạn che bắt đầu **ngay frame đầu tiên** của object (1676) → bị loại `gt_missing_before`.

Với mỗi object GT của 60 video: ghép GT ↔ tracker từng frame bằng Hungarian trên toàn frame, IoU ≥ 0,5
(`occlusion_eval.match_video`, cùng phép ghép đã sửa ở `FIX_REPORT.md`), được timeline ID của CV, EKF, UKF.

**CV thất bại tại `f_fail`** (ID CV phải đã ổn định ≥ 5 frame trước đó):
- `switched`: CV ghép target bằng ID khác ID cũ;
- `lost`: CV không ghép được target ≥ 10 frame GT liên tiếp.
- Thất bại phải **kéo dài**: 10 frame GT kể từ `f_fail` CV không quay về ID cũ (loại "nhấp nháy" 1 frame — 40 trường hợp,
  ví dụ MVI_40162 t40: CV 264→256→264 trong 1 frame; MVI_40181 t18: 65→63→65).

**EKF thắng**: EKF có ID `e` ổn định ≥ 5 frame trước `f_fail`; trong [`f_fail`, `f_fail`+5] không ghép ID nào khác `e`
(cho phép None); kể từ frame ghép được đầu tiên giữ đúng `e` **≥ 10 frame GT liên tiếp**. EKF chỉ đúng 1–2 frame rồi
đổi/mất → loại tự động.

**Ổn định trước đó**: 25 frame GT trước `f_fail`, cả CV và EKF không đổi ID (loại 10 trường hợp cảnh nhấp nháy, ví dụ
MVI_40981 t26: EKF 159→182 ở 382 rồi CV thất bại ở 387 — "EKF vẫn khớp cùng ID" không còn nghĩa).

**Xe di chuyển rõ** (GT trong cửa sổ [`f_fail`−25, `f_fail`+25], ≥ 20 frame): tốc độ trung vị ≥ 3 px/frame; net ≥ 0,5 ×
đường chéo bbox trung vị; path ≥ 30 px.

**Xếp hạng**: có che khuất (GT vis < 1 quanh `f_fail` hoặc đoạn che giao cửa sổ) → đổi hướng ≥ 10° → thời gian EKF giữ đúng
ID sau `f_fail` (`ekf_hold_until_change`, cho phép detector rớt < 10 frame vì rớt xảy ra ở cả ba tracker; cột
`ekf_hold_after` đếm liên tiếp tuyệt đối được lưu kèm) → tốc độ. UKF không tham gia lọc/xếp hạng.

### Luồng lọc (`candidates_single_case.csv`, 2 592 dòng = mọi thời điểm CV thất bại)

| Bước | Số lượng |
|---|---:|
| Object GT có ≥ 10 frame | 5 826 |
| Thời điểm CV thất bại với ID đã ổn định (1 760 switched, 832 lost) | 2 592 |
| EKF giữ đúng ID ≥ 10 frame liên tiếp sau đó | 71 |
| + CV thất bại kéo dài ≥ 10 frame | 31 |
| + CV và EKF không đổi ID trong 25 frame trước | 21 |
| + xe di chuyển rõ (3 tiêu chí) | **14** |
| trong đó có che khuất / có đổi hướng ≥ 10° | 4 / 0 |

Lý do loại chính của 2 578 dòng còn lại: EKF cũng đổi ID tại `f_fail` (1 325), EKF không giữ ≥ 10 frame (1 007),
EKF chưa có ID ổn định trước đó (189), CV nhấp nháy (40), cảnh không sạch (10), xe chậm/đứng (7).

## 4. Bảng ứng viên (14 trường hợp đạt mọi tiêu chí cứng)

| # | Video | GT | `f_fail` | CV | gap trước | EKF ID | giữ liên tiếp / tới khi đổi | UKF ID (giữ) | v trung vị | path | net/diag | turn | che max | Che khuất |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **MVI_40991** | **27** | **1727** | 365 → 376 | 21 | 379 | 34 / **57** | 378 (57) | 5,27 | 335 | 2,60 | −3,8° | 0,99 | **có** |
| 2 | MVI_40241 | 143 | 997 | 872 → 801 | 0 | 873 | **51** / 51 | 878 (51) | 5,63 | 241 | 3,49 | 0,3° | 0,85 | có |
| 3 | MVI_40991 | 25 | 1450 | 311 → 324 | 20 | 326 | 23 / 27 | 324 (56) | 5,09 | 318 | 2,54 | −6,0° | 1,00 | có |
| 4 | MVI_40212 | 57 | 460 | 261 → 280 | 6 | 260 | 17 / 17 | 259 (17) | 21,3 | 857 | 4,76 | −4,0° | 0,58 | có |
| 5 | MVI_40141 | 28 | 1437 | 419 → 408 | 0 | 414 | 139 / 139 | 420 (139) | 5,64 | 218 | 1,64 | 4,2° | 0 | không |
| 6 | MVI_41073 | 92 | 965 | 489 → 476 | 0 | 473 | 71 / 71 | 474 (71) | 5,02 | 226 | 4,38 | 3,2° | 0 | không |
| 7 | MVI_40191 | 92 | 642 | 370 → 399 | 2 | 381 | 66 / 66 | 373 (66) | 4,15 | 206 | 3,23 | −0,8° | 0 | không |
| 8 | MVI_40241 | 140 | 964 | 833 → 792 | 0 | 833 | 53 / 53 | 838 (53) | 5,59 | 187 | 2,88 | −1,0° | 0 | không |
| 9 | MVI_40243 | 154 | 1133 | 823 → 790 | 0 | 815 | 40 / 48 | 817 (48) | 5,11 | 202 | 3,39 | −0,2° | 0 | không |
| 10 | MVI_63521 | 33 | 713 | 235 → 228 | 1 | 235 | 14 / 47 | 233 (47) | 3,30 | 182 | 1,71 | −3,4° | 0 | không |
| 11 | MVI_40241 | 245 | 1696 | 1563 → 1588 | 0 | 1541 | 44 / 44 | 1546 (44) | 5,59 | 299 | 4,51 | 1,4° | 0 | không |
| 12 | MVI_40244 | 45 | 260 | 207 → 212 | 1 | 203 | 42 / 42 | 203 (42) | 10,4 | 348 | 3,22 | 0,4° | 0 | không |
| 13 | MVI_63554 | 71 | 526 | 362 → 338 | 0 | 359 | 34 / 34 | 357 (34) | 14,4 | 602 | 4,09 | 0,1° | 0 | không |
| 14 | MVI_40243 | 126 | 908 | 659 → 677 | 4 | 653 | 11 / 11 | 655 (11) | 4,97 | 202 | 3,50 | −2,5° | 0 | không |

Cả 14 đều là `switched`; không có trường hợp `lost` nào qua được bước "EKF giữ ≥ 10 frame".

**Vì sao chọn #1.** Bốn ứng viên có che khuất (#1–#4), không ứng viên nào đổi hướng ≥ 10°, nên tiêu chí thứ ba quyết
định: thời gian EKF giữ đúng ID sau khi CV thất bại. #1 giữ **57** frame tới hết đời track (chỉ ngắt bởi detector rớt
chung của cả ba tracker); #2 giữ 51. **Lưu ý minh bạch:** nếu đếm *liên tiếp tuyệt đối* (không cho phép frame rớt
detection) thì #2 (51) xếp trên #1 (34). Hai cách đếm đều có trong CSV (`ekf_hold_until_change`, `ekf_hold_after`).
Tôi chọn cách đếm "tới khi đổi ID" vì frame rớt detection ở 1761/1763 xảy ra ở cả ba tracker và track EKF 379 vẫn
tiếp tục đúng ID sau đó — đó không phải EKF "mất" ID. Ngoài ra #1 là che khuất **hoàn toàn** (0,99 – 1,00) với đoạn mất
dấu 21 frame rõ ràng, còn #2 (MVI_40241 GT 143, xe 54×34 px chỉ nhìn thấy 13 – 36 %) thất bại do một track LOST cũ
(ID 801, tạo ở frame 947 cho xe khác, còn sống nhờ buffer 90) chiếm detection của target ở 997 — cơ chế khác và khó
quan sát bằng mắt. Nếu người đọc muốn dùng #2, `build --rank 2` dựng được ngay.

## 5. Diễn biến trong clip (từ prediction đã lưu)

| Frame | GT nhìn thấy | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|---|
| 1685 – 1705 | 100 → 45 % (đi vào sau cột) | ID 365 | ID 379 | ID 378 |
| 1706 – 1726 | 39 → 0 % (che hoàn toàn từ 1712) | không ghép (track 365 LOST, vẫn dự đoán) | không ghép (379 LOST) | không ghép (378 LOST) |
| **1727** | 0 %¹ (detector đã thấy lại: IoU det–GT 0,79) | **376** (track mới) | **379** (nối lại) | **378** (nối lại) |
| 1728 – 1760 | 1 → 100 % | 376 | 379 | 378 |
| 1761, 1763, 1784 – 1788 | 100 % | không ghép (rớt detection ở cả ba) | không ghép | không ghép |
| 1762, 1764 – 1783, 1789 – 1790 | 100 % | 376 | 379 | 378 |

¹ GT của UA-DETRAC ghi 0 – 3 % nhìn thấy đến hết 1731 nhưng detector phát hiện lại từ 1727 với IoU 0,79 so với box GT;
nhãn `vis` của GT thô hơn detector vài frame.

Track CV 365: box cuối ở 1706, sau đó LOST và **không bao giờ** được nối lại (buffer 90 giữ nó sống đến ~1796 nhưng nó
không ghép được detection nào). Track CV 376 tạo ở 1727 và theo target đến 1790. Track EKF 379 / UKF 378 sống liên tục
1679 – 1790 (80 box, 79 frame ghép đúng GT 27).

**`id_match` / `id_continuous`** (đoạn [1705, 1737] = frame ghép cuối trước khi mất dấu … `f_fail`+10):

| | id_before (1705) | id tại 1727 | `id_match` | `id_continuous` |
|---|---|---|---|---|
| KF + CV | 365 | 376 | **0** | 0 |
| EKF + CTRV | 379 | 379 | **1** | 0 |
| UKF + CTRV | 378 | 378 | 1 | 0 |

`id_continuous = 0` ở cả ba vì không tracker nào quan sát được target trong 21 frame che hoàn toàn; EKF/UKF thắng ở
**tái nhận đúng ID**, không phải ở duy trì detection liên tục.

## 6. Chạy lại ba tracker (không cache) và đối chiếu

`rerun_with_state_log.rerun()` chạy `model.track()` từ **frame 1 đến 1777** với đúng YAML buffer=90, `conf=0,25`, tô xám
`ignored_region` như `baseline_track.py`, `persist=True`; đọc `tracker.args` để xác nhận `track_buffer=90`. Được chạy
**hai lần độc lập** trong phiên này (lần 1 lúc 19:05, lần 2 lúc 19:19); cả hai cho kết quả giống nhau. Đối chiếu trong
cửa sổ 1702 – 1777 với prediction đã lưu:

| Model | Box đã lưu | Box chạy lại | Khớp hình học IoU ≥ 0,9 | Khớp cùng ID thô | Sau ánh xạ ID 1–1 | Timeline target trùng |
|---|---:|---:|---:|---:|---:|---:|
| KF + CV | 101 | 101 | 101 (100 %) | 100 % | 100 % (ánh xạ = đồng nhất) | 76 / 76 frame |
| EKF + CTRV | 101 | 101 | 101 (100 %) | 100 % | 100 % | 76 / 76 |
| UKF + CTRV | 101 | 101 | 101 (100 %) | 100 % | 100 % | 76 / 76 |

Kết cục target tại 1727 trong lần chạy lại: CV 376, EKF 379, UKF 378 — trùng file đã lưu. Video dùng box/ID từ
**file đã lưu**; quỹ đạo/box dự đoán trong lúc LOST (nét đứt) lấy từ trạng thái runtime của lần chạy lại (file đã lưu
không chứa track LOST). `rerun/verification.json` lưu toàn bộ số liệu trên; lần render cuối (sửa nhãn tiêu đề) dùng
`--reuse-rerun` đọc lại đúng kết quả chạy lại này, không chạy lại lần ba (ghi ở `metadata.json → rerun_source`).

## 7. Phép ghép tại frame quyết định 1727 (`match_cost_at_key_frame.csv`)

ByteTrack: `cost = 1 − IoU(box dự đoán, detection) × score`, ghép khi `cost < 0,8` ⇔ `IoU × score > 0,2`
(`fuse_score=true`, `match_thresh=0,8`). Detection của target (YOLO tính lại trên cùng frame đã tô xám):
box `[263,9; 169,8; 369,4; 240,9]`, tâm (316,6; 205,4), score **0,646**, IoU với GT 0,789. Box dự đoán lấy từ pool
ngay sau `multi_predict`, trước khi ghép (track đang LOST 21 frame):

| Model | Track cũ | Box dự đoán (tâm) | IoU dự đoán–detection | IoU × score | cost | Qua ngưỡng 0,8? | LAP đầy đủ: track cũ nhận detection? | ID target sau frame (đã lưu) |
|---|---|---|---:|---:|---:|---|---|---|
| KF + CV | 365 | (361,0; 185,9) | 0,300 | **0,193** | 0,807 | **không** (thiếu 0,007) | không → detection thành track mới 376 | 376 |
| EKF + CTRV | 379 | (360,3; 193,6) | 0,361 | **0,233** | 0,767 | có | **có** | 379 |
| UKF + CTRV | 378 | (369,4; 196,0) | 0,317 | **0,205** | 0,795 | có (dư 0,005) | có | 378 |

Đọc kết quả:
- Cả ba dự đoán đều **lệch về phía sau** (bên phải, cao hơn) detection ~45 px theo x: trong 21 frame bị che xe chạy
  nhanh hơn (~9 px/frame) so với trước khi vào cột (~5 px/frame) nên mọi bộ lọc đều dự đoán chưa tới. EKF cho box gần
  detection nhất theo y (193,6 so với 185,9 của CV), IoU cao hơn 0,06 → qua ngưỡng.
- Biên độ **mỏng**: CV thiếu 0,007, UKF dư 0,005, EKF dư 0,033. Kết cục "EKF/UKF nối lại, CV không" là thật (tái lập
  100 %) nhưng nằm sát ngưỡng; một thay đổi nhỏ về score detection hoặc vị trí dự đoán có thể đảo kết cục.
- Đây là replay offline đúng implementation `matching.iou_distance → fuse_score → linear_assignment` trên pool dự đoán
  của lần chạy lại và detection tính lại; **không** phải log association của lượt chạy lịch sử. Detection tính lại có thể
  lệch cỡ 10⁻³ do GPU không bit-reproducible — cùng bậc với biên 0,007/0,005 nêu trên.

## 8. Bối cảnh: phép quét đối xứng (`symmetric_context.csv`)

Cùng tiêu chí, đổi vai trò tracker (`python src/demo/single_case_buffer90.py context`):

| A thất bại, B giữ đúng ID ≥ 10 frame | Thất bại của A | B giữ ≥ 10 | + A kéo dài | + ổn định trước | + xe di chuyển rõ |
|---|---:|---:|---:|---:|---:|
| **CV thất bại, EKF giữ** (chiều của demo) | 2 592 | 71 | 31 | 21 | **14** |
| EKF thất bại, CV giữ | 2 564 | 87 | 35 | 20 | **9** |
| CV thất bại, UKF giữ | 2 592 | 75 | 39 | 22 | 15 |
| UKF thất bại, CV giữ | 2 569 | 93 | 44 | 23 | 7 |
| EKF thất bại, UKF giữ | 2 564 | 31 | 15 | 6 | 3 |
| UKF thất bại, EKF giữ | 2 569 | 32 | 17 | 10 | 1 |

Chiều "EKF thắng CV" (14) và chiều ngược lại (9) cùng bậc; số thời điểm thất bại của CV (2 592) và EKF (2 564) gần
bằng nhau. Trên chỉ số tổng hợp buffer=90 đã báo cáo (README, commit `02b0a02`): HOTA CV 0,6101 / EKF 0,6107, IDSW
2 360 / 2 307 — chênh rất nhỏ. Case study **không** thay đổi kết luận đó.

## 9. Giới hạn diễn giải

1. Một trường hợp train được chọn có chủ đích theo tiêu chí "EKF thắng"; không phải ước lượng hiệu năng; không dùng test.
2. Kết cục sát ngưỡng ghép (§7): lợi thế của EKF ở frame 1727 là 0,04 điểm fused similarity; UKF chỉ dư 0,005.
3. Xe đi thẳng (−3,8°); clip không minh họa được ưu thế của mô hình CTRV khi rẽ — không có ứng viên nào đạt tiêu chí
   kèm đổi hướng ≥ 10°.
4. `id_continuous = 0` ở cả ba: không tracker nào "theo dõi" xe trong lúc che; khác biệt chỉ ở quyết định tái nhận.
5. Kết quả buffer=90 lịch sử có trước các sửa lỗi trên nhánh; với video này tái lập 100 %, nhưng audit trước ghi nhận
   video khác chỉ khớp 98,5 – 100 % sau ánh xạ ID.
6. Cách đếm "thời gian giữ đúng ID" ảnh hưởng thứ hạng #1/#2 (§4); cả hai cách đều được lưu để người đọc tự kiểm.
7. GT `vis` của UA-DETRAC ghi 0 – 3 % đến 1731 trong khi detector thấy lại từ 1727; "che hoàn toàn 21 frame" dựa trên
   nhãn GT, còn "mất dấu 21 frame" (1706 – 1726) dựa trên ghép tracker — hai con số trùng nhau ở đây nhưng có nguồn khác nhau.
8. **Kết cục này không đặc thù cho `track_buffer=90`.** Đoạn mất dấu chỉ 21 frame (< 30), nên với buffer=30 các track
   LOST vẫn còn sống khi xe hiện lại. Đối chiếu với lượt A đã lưu (buffer=30, cùng `conf=0,25`,
   `data/processed/trackers/DETRAC-all/runA-*/data/MVI_40991.txt`) cho **cùng kiểu kết cục** trên chính object này:
   CV `368` (1680–1705) → mất 1706–1726 → **`379` mới** (1727–1790); EKF `380` → mất → **`380`** nối lại; UKF `379` → mất →
   **`379`** nối lại (ID khác vì là lượt chạy khác; các frame rớt detection 1761/1763/1784–1788 cũng trùng). Vì vậy clip
   minh họa khác biệt ở **quyết định tái nhận của motion model**, không minh họa tác dụng của việc nới buffer. Lượt
   buffer=60 không có file cho video này nên không đối chiếu được.

## 10. Artefact và tái tạo

```
outputs/thesis_demos_buffer90_single_case/
├── DEMO_SINGLE_CASE_REPORT.md        (file này)
├── candidates_single_case.csv        2 592 thời điểm CV thất bại, mọi đại lượng, từng điều kiện, lý do loại, rank
├── symmetric_context.csv             phép quét đối xứng 6 cặp tracker (§8)
├── demo_ekf_wins_cv_buffer90.mp4     1920×1080 H.264 12 fps, 118 frame, 9,83 s  (không commit)
├── before.png                        frame 1705 — frame ghép cuối trước khi mất dấu
├── failure_or_occlusion.png          frame 1717 — giữa đoạn che hoàn toàn, ba box dự đoán nét đứt
├── after.png                         frame 1737 — 10 frame sau frame quyết định: CV "DOI ID 365 -> 376", EKF 379, UKF 378
├── metadata.json                     cấu hình, runtime, ứng viên, chuyển động, che khuất, kết cục, đối chiếu chạy lại, replay ghép
├── match_cost_at_key_frame.csv       bảng §7
└── rerun/                            (không commit) box/trạng thái/pool dự đoán của lần chạy lại + verification.json
src/demo/single_case_buffer90.py      scan | build [--rank N] [--reuse-rerun] | context
```

Video: 3 panel bằng nhau KF+CV | EKF+CTRV | UKF+CTRV, cùng vùng crop cố định `[91, 0, 668, 540]` bao trọn quỹ đạo target
(letterbox trong panel 640×720); nét liền xanh lá = GT; nét đứt màu track = quỹ đạo/box dự đoán (từ lần chạy lại) khi
mất dấu; viền vàng = GT đang bị che; box đỏ + "DOI ID cũ -> mới" 15 frame khi đổi ID; dòng trạng thái từng panel; caption
cuối 3,5 s đúng nguyên văn: *"Ví dụ minh họa trên tập train với track_buffer=90: trong trường hợp này, EKF duy trì ID
của xe đang di chuyển lâu hơn CV. Đây là case study được chọn, không chứng minh EKF vượt trội trên toàn bộ dữ liệu."*

```bash
conda activate khoaluan-mot
python src/demo/single_case_buffer90.py scan            # ~2 phút, chỉ đọc GT + prediction đã lưu
python src/demo/single_case_buffer90.py build           # chạy lại 3 tracker đến frame 1777 (~7 phút GPU), đối chiếu, replay, render
python src/demo/single_case_buffer90.py context         # ~12 phút, phép quét đối xứng
python src/demo/single_case_buffer90.py build --reuse-rerun   # chỉ render lại từ rerun/ đã có
```

`.gitignore` đã thêm quy tắc bỏ `*.mp4`, `*.png`, `rerun/` của thư mục này (báo cáo, CSV, metadata vẫn được theo dõi).
Không commit, không push. Không sửa `outputs/thesis_demos*/` cũ, `data/`, `results/`, `configs/`, thuật toán hay ngưỡng.
