# So sánh KF / EKF / UKF cho Multi-Object Tracking xe cộ trên UA-DETRAC

Khóa luận: đánh giá 3 motion model (Kalman Filter với mô hình CV, Extended KF và
Unscented KF với mô hình CTRV) trong pipeline **YOLOv8 + ByteTrack**, tập trung
vào hiệu năng khi xe **bị che khuất** và **di chuyển phi tuyến** (rẽ, vòng xuyến).

---

## Trạng thái các giai đoạn

| Giai đoạn | Nội dung | Trạng thái |
|---|---|---|
| 1 | Tải dữ liệu & data pipeline | ✅ Xong |
| 2 | Baseline YOLOv8 + ByteTrack + TrackEval | ✅ Xong — **baseline đã khoá** |
| 3 | EKF + CTRV | ⏳ Chưa bắt đầu |
| 4 | UKF + CTRV | ⏳ Chưa bắt đầu |
| 5 | Thực nghiệm đầy đủ & phân tích | ⏳ Chưa bắt đầu |

---

## Cài đặt

```bash
python -m pip install -r requirements.txt
```

Đã kiểm tra với Python 3.11.9 trên Windows.

---

## Cấu trúc thư mục

```
UA-DETRAC test/
├── requirements.txt
├── README.md
├── src/
│   ├── config.py              # Cấu hình tập trung (đường dẫn, ngưỡng, hằng số)
│   ├── download_data.py       # Tải dữ liệu từ Google Drive
│   ├── extract_verify.py      # Giải nén + kiểm tra toàn vẹn
│   ├── parse_detrac_xml.py    # Parse XML -> DataFrame chuẩn hoá
│   ├── to_motchallenge.py     # Xuất format MOTChallenge cho TrackEval
│   ├── occlusion_segments.py  # Phát hiện đoạn che khuất, phân nhóm độ dài
│   ├── curvature.py           # Tính độ cong quỹ đạo, phân nhóm thẳng/cong
│   ├── baseline_track.py      # Baseline: YOLOv8 (pretrained) + ByteTrack
│   └── run_trackeval.py       # Chạy TrackEval -> MOTA / IDF1 / HOTA
├── models/                    # Trọng số YOLOv8 (.pt) tải về
├── data/
│   ├── raw/                   # File .zip tải về
│   ├── extracted/             # Ảnh + XML sau giải nén
│   ├── interim/                # Bảng trung gian (.parquet / .csv)
│   └── processed/             # Format MOTChallenge (gt.txt, seqinfo.ini) + trackers/
└── results/                   # Hình ảnh, bảng kết quả, output TrackEval
```

---

## Chạy pipeline Giai đoạn 1

```bash
python src/download_data.py --only train_annotations_xml   # tải annotation XML
python src/extract_verify.py                               # giải nén + kiểm tra
python src/parse_detrac_xml.py                             # XML  -> parquet
python src/to_motchallenge.py                              # -> gt.txt + seqinfo.ini
python src/occlusion_segments.py                           # đoạn che khuất
python src/curvature.py --plot                             # độ cong quỹ đạo
```

Các script đều **idempotent** — chạy lại nhiều lần không hỏng dữ liệu, bước nào
đã xong sẽ được bỏ qua.

---

## Chạy pipeline Giai đoạn 2

```bash
# 1) Tạo GT split cho 5 video mẫu (đã làm sẵn, chỉ cần chạy lại nếu đổi danh sách video)
python src/to_motchallenge.py --videos MVI_20011 MVI_40962 MVI_39811 MVI_40204 MVI_63521 \
    --split-name DETRAC-sample

# 2) Chạy YOLOv8 (pretrained) + ByteTrack, xuất kết quả sang format MOTChallenge
python src/baseline_track.py

# 3) Đánh giá bằng TrackEval
python src/run_trackeval.py --split-name DETRAC-sample --tracker yolov8n-bytetrack
```

Danh sách video mẫu nằm ở `config.SAMPLE_VIDEOS` — chỉ cần sửa ở đó, không cần
sửa lệnh.

---

## Giải thích từng file code

### `src/config.py`
Nơi duy nhất chứa đường dẫn và hằng số. Muốn đổi ngưỡng che khuất, ngưỡng phân
nhóm thẳng/cong, hay chỗ lưu dataset thì **chỉ sửa file này**, không sửa các
script khác. Các hằng số quan trọng:

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `FPS` | 25 | UA-DETRAC quay ở 25 fps → 1 frame = 0.04 s |
| `IMG_W`, `IMG_H` | 960 × 540 | Kích thước ảnh |
| `OCCLUSION_MIN_RATIO` | 0.10 | Trên ngưỡng này coi là "bị che" |
| `OCCLUSION_FULL_RATIO` | 0.90 | Trên ngưỡng này coi là "che gần như hoàn toàn" |
| `OCCLUSION_MERGE_GAP` | 2 | Gộp 2 đoạn che khuất cách nhau ≤ 2 frame |
| `OCCLUSION_BUCKETS` | <0.5s / 0.5–1.5s / >1.5s | Phân nhóm độ dài đoạn che khuất |
| `SAVGOL_WINDOW/POLY` | 11 / 2 | Làm trơn quỹ đạo trước khi lấy đạo hàm |
| `TURN_ANGLE_THRESHOLD_DEG` | 20 | Trên ngưỡng góc rẽ này coi là "cong" |
| `STRAIGHTNESS_THRESHOLD` | 0.97 | Dưới ngưỡng chỉ số thẳng này coi là "cong" |

### `src/download_data.py`
Tải dữ liệu từ Google Drive.

**Tại sao không dùng `wget` thuần?** Với file lớn (>100 MB), Google Drive không
trả về file mà trả về **trang HTML xác nhận virus-scan** kèm một `confirm token`.
`wget` thuần sẽ lưu chính trang HTML đó thành file `.zip` → giải nén báo lỗi
"not a zip file". Script dùng thư viện `gdown`, thư viện này tự động đọc cookie +
confirm token rồi gọi request thứ hai để lấy byte stream thật.

Sau khi tải, script **luôn kiểm tra** file có phải zip hợp lệ không, qua 3 tầng:
1. Đọc vài KB đầu — nếu bắt đầu bằng `<!doctype html` thì báo rõ đây là trang
   virus-scan / link hết hạn, không phải file.
2. `zipfile.is_zipfile()` — kiểm tra chữ ký zip.
3. `zf.testzip()` — kiểm tra CRC toàn bộ entry, phát hiện file tải thiếu byte.

Hàm tải tự dò signature của `gdown` lúc runtime nên chạy được với cả gdown 4/5/6
(gdown 6 đã bỏ tham số `fuzzy` và nhận thẳng `id=`).

### `src/extract_verify.py`
Giải nén zip và đối chiếu 3 nguồn thông tin: số video có XML, số frame được
annotate trong từng XML, và số file ảnh thực tế trên đĩa.

**Lưu ý về số frame:** thẻ `<frame>` chỉ xuất hiện cho frame *có ít nhất 1 xe*.
Nên `n_frames_annotated ≤ n_images` là **bình thường** (thực tế: 82.085 / 83.791,
tức 2.04% frame không có xe nào). Chỉ coi là lỗi khi `max_frame_num > n_images`
(annotation trỏ tới ảnh không tồn tại).

**Tự tìm thư mục ảnh.** Mỗi bản mirror lồng thư mục một kiểu
(`Insight-MVT_Annotation_Train/`, `DETRAC-Images/DETRAC-Images/`, hoặc lồng 3
cấp). Thay vì bắt bạn sắp xếp lại, `config.find_images_root()` duyệt cây thư mục
tối đa 4 cấp và chọn thư mục có nhiều thư mục con `MVI_*` chứa ảnh nhất. Đã test
với cả 3 kiểu bố cục.

**Giải nén chọn lọc.** Mirror thường đóng gói cả 60 video train lẫn 40 video test
trong một file ~10 GB. Tập test không có ground truth công khai nên vô dụng với
đề tài này → mặc định script **chỉ giải nén 60 video train**, tiết kiệm ~4 GB đĩa.
Dùng `--all-sequences` để giải nén tất cả.

Dùng `ET.iterparse` để không nạp cả cây XML vào RAM.

### `src/parse_detrac_xml.py`
Parse XML thành DataFrame chuẩn hoá. Đây là file **quan trọng nhất** của giai
đoạn 1 vì chứa 2 điểm dễ làm sai:

**Điểm 1 — XML không có sẵn trường `occlusion_ratio`.**
UA-DETRAC chỉ cho biết *vùng chồng lấn* (`region_overlap`). Phải tự tính:

```
occlusion_ratio = diện_tích( HỢP các vùng che  ∩  box ) / diện_tích(box)
```

Phải dùng phép **hợp** (union) chứ không phải **tổng**: một xe có thể bị 2 xe
khác che, hai vùng chồng lấn có thể giao nhau → cộng dồn sẽ đếm trùng và cho
ratio > 1. Script dùng thuật toán *nén toạ độ* (coordinate compression) để tính
diện tích hợp chính xác.

**Điểm 2 — ngữ nghĩa của `occlusion_status` (đã kiểm chứng bằng thực nghiệm).**
Mỗi quan hệ che khuất giữa 2 xe chỉ được khai báo **một lần, ở một phía** (đã
kiểm tra: không tồn tại cặp khai báo đối xứng nào trong toàn bộ dataset).

| `status` | `occlusion_id` | Ý nghĩa | Bằng chứng |
|---|---|---|---|
| `0` | là 1 xe trong frame | Xe khai báo **bị che** bởi xe kia | 99.2% trường hợp xe khai báo ở xa camera hơn (cạnh dưới cao hơn trong ảnh), box nhỏ hơn (median tỉ lệ diện tích 0.53) |
| `1` | là 1 xe trong frame | Xe khai báo **đang che** xe kia — bản thân **không** bị che | 99.4% trường hợp xe khai báo ở gần camera hơn, box lớn hơn (median 1.31) |
| `-1` | không phải target nào | Bị che bởi **vật nền tĩnh** (cây, cột, biển báo) | 100% trường hợp `occlusion_id` không ứng với target nào trong cùng frame |

**Hệ quả:** khi tính độ che khuất của xe V, phải gom (a) vùng V tự khai báo với
`status=0`, (b) vùng V tự khai báo với `status=-1`, (c) vùng do **xe khác** khai
báo với `status=1` và `occlusion_id = V`; đồng thời **loại bỏ** vùng V khai báo
với `status=1`. Script xử lý mỗi frame theo 3 lượt: đọc → phân phối lại vùng che
về đúng xe bị che → mới tính tỉ lệ. Nếu làm sai bước này thì `visibility` vừa
thừa vừa thiếu.

### `src/to_motchallenge.py`
Xuất sang format MOTChallenge chuẩn để TrackEval đọc được:

```
frame, id, bb_left, bb_top, bb_width, bb_height, conf, class, visibility
```

- `conf = 1`, `class = 1` (gộp cả 4 loại xe về một lớp — bài toán theo vết phương
  tiện không phân biệt loại; loại xe vẫn được giữ riêng trong
  `vehicle_types.csv` để phân tích sau).
- `visibility = 1 − occlusion_ratio`.

**Về `ignored_region`:** UA-DETRAC đánh dấu sẵn một số **vùng tĩnh** trong ảnh
(vỉa hè, bãi đỗ xe xa, đường ngược chiều) — xe trong đó **không** được annotate.
Nếu không xử lý, detector sẽ phát hiện xe ở đây và bị tính là False Positive oan.
Format MOTChallenge không có chỗ mô tả "vùng bỏ qua", nên script hỗ trợ 2 cách:
1. *(mặc định)* Ghi ra file riêng `gt/ignored_regions.txt` → Giai đoạn 2 sẽ lọc
   detection rơi vào các vùng này **trước khi** đưa vào TrackEval.
2. `--ignore-as-distractor`: ghi thành dòng GT `class = 13` (distractor của
   MOT17) để bước preproc của TrackEval tự loại. Cách này làm `gt.txt` phình to
   nên chỉ dùng khi cần.

Cấu trúc thư mục xuất ra theo đúng chuẩn `MotChallenge2DBox` của TrackEval:

```
data/processed/
  seqmaps/DETRAC-all.txt
  DETRAC-all/<video>/gt/gt.txt
  DETRAC-all/<video>/gt/ignored_regions.txt
  DETRAC-all/<video>/seqinfo.ini
  trackers/DETRAC-all/<tên_tracker>/data/<video>.txt    <- Giai đoạn 2 ghi vào đây
```

### `src/occlusion_segments.py`
Phát hiện các **đoạn** bị che khuất liên tục của từng track, tính độ dài theo
frame và giây, rồi phân nhóm `short` (<0.5s) / `medium` (0.5–1.5s) / `long`
(>1.5s).

Thuật toán: đánh dấu frame `occluded = ratio ≥ 0.10` → gom frame liên tiếp thành
đoạn → **gộp** hai đoạn cách nhau ≤ 2 frame (annotation có nhiễu, một frame lọt
ra ngoài ngưỡng không có nghĩa là xe đã hết bị che; không gộp sẽ bị cắt vụn
thành nhiều đoạn ngắn giả) → loại đoạn ngắn hơn 2 frame.

Script tính **3 bảng** ứng với 3 mức che khuất — xem phần "Kết quả" bên dưới.

### `src/curvature.py`
Tính độ cong quỹ đạo để phân nhóm track thẳng / cong.

Quỹ đạo là chuỗi tâm bbox `(x(t), y(t))`. Tâm bbox có nhiễu vài pixel, nếu lấy
sai phân trực tiếp thì đạo hàm bậc 2 sẽ bị nhiễu lấn át hoàn toàn. Script dùng
**Savitzky-Golay**: khớp đa thức bậc `p` lên cửa sổ `w` điểm lân cận rồi lấy đạo
hàm *giải tích* của đa thức đó — vừa làm trơn vừa lấy đạo hàm trong một bước.

Độ cong của đường cong tham số:

```
          | x'·y'' − y'·x'' |
kappa = ─────────────────────        [1/px]
          ( x'² + y'² )^(3/2)
```

`kappa = 1/R` với `R` là bán kính đường tròn mật tiếp. Đường thẳng: `kappa = 0`.
Mẫu số tiến về 0 khi xe đứng yên → những frame có tốc độ dưới `EPS_SPEED`
(0.5 px/frame) được đánh dấu `NaN`.

Hai đại lượng liên hệ trực tiếp với **CTRV**:

```
theta = atan2(y', x')            [rad]    — hướng di chuyển
omega = d(theta)/dt = kappa · v  [rad/s]  — tốc độ quay (turn rate)
```

`omega` chính là **biến trạng thái thứ 5** của CTRV `[cx, cy, v, θ, ω]`. Thống
kê `omega` ở đây cho biết giá trị thực tế của nó trong UA-DETRAC, dùng để đặt
nhiễu quá trình `Q` hợp lý ở Giai đoạn 3 và 4.

Nhãn phân nhóm: `static` (xe đỗ), `unknown` (track < 15 frame, không đủ điểm ước
lượng đạo hàm bậc 2), `straight` (góc rẽ < 20° **và** chỉ số thẳng ≥ 0.97),
`curved` (còn lại). Thêm cột `turn_class` chia mịn hơn: `straight` (<10°) /
`gentle` (10–45°) / `sharp` (≥45°).

### `src/baseline_track.py`
Chạy **YOLOv8 pretrained (COCO) + ByteTrack** (tracker tích hợp sẵn trong
`ultralytics`, `model.track(...)`) trên từng video, xuất kết quả sang format
MOTChallenge để TrackEval so với ground truth.

**Vì sao đọc từng ảnh `.jpg` gốc thay vì ghép lại thành video?** UA-DETRAC lưu
sẵn mỗi frame là 1 file ảnh riêng. Ghép lại thành `.mp4` sẽ nén mất chất lượng.
Script đọc từng ảnh bằng `cv2.imread()` rồi gọi
`model.track(frame, persist=True, ...)` cho từng frame theo đúng thứ tự — đã
đối chiếu với mã nguồn `ultralytics/trackers/track.py`: với `persist=True`,
tracker **không bị reset** giữa các lần gọi, đúng là cơ chế chính thức để
"streaming" từng frame rời rạc mà vẫn giữ trạng thái theo dõi liên tục.

**Reset tracker giữa các video bằng cách nào?** Thay vì gọi `tracker.reset()`,
script **tạo lại `YOLO(...)` từ đầu** cho mỗi video mới. Chi phí tải lại model
nhỏ (yolov8n: ~0.3–2s) nhưng đảm bảo tuyệt đối không còn sót track id / trạng
thái Kalman filter nào từ video trước — ưu tiên an toàn/dễ debug hơn tối ưu tốc độ.

**Xử lý `ignored_region`:** thay vì lọc kết quả *sau khi* đã tracking xong (có
rủi ro ByteTrack đã lỡ gán track id cho 1 detection sai trong vùng này), script
**tô xám** (màu `114,114,114` — trùng màu padding mặc định của YOLO, không tạo
cạnh giả) trực tiếp lên vùng đó *trước khi* đưa ảnh vào detector. Detector vì
vậy **không bao giờ** thấy vật thể trong vùng bị che, đúng với cách UA-DETRAC
định nghĩa "vùng không annotate". Đã kiểm chứng bằng hình minh hoạ — xem phần
kết quả bên dưới.

**Format file kết quả:** chỉ 7 cột `frame,id,bb_left,bb_top,bb_width,bb_height,conf`
(không kèm class/visibility như `gt.txt`). Đã đối chiếu mã nguồn
`trackeval/datasets/mot_challenge_2d_box.py`: file có dưới 8 cột thì TrackEval
tự gán `class = 1` cho mọi dòng — khớp với class ta dùng cho GT xe, và tránh
nhầm lẫn với 3 cột x/y/z (toạ độ 3D, thường để `-1`) nếu dùng đủ 10 cột.

### `src/run_trackeval.py`
Tích hợp **TrackEval** (`JonathonLuiten/TrackEval`, bản đóng gói `pip install
trackeval`) để tính **HOTA, MOTA, IDF1**.

**Vì sao GT của ta "vừa khít" với `MotChallenge2DBox`?** TrackEval không tự suy
luận tên lớp — nó dùng một bảng cố định (đã đối chiếu mã nguồn):

```python
class_name_to_class_id = {
    'pedestrian': 1, 'person_on_vehicle': 2, 'car': 3, 'bicycle': 4,
    'motorbike': 5, 'non_mot_vehicle': 6, 'static_person': 7,
    'distractor': 8, 'occluder': 9, 'occluder_on_ground': 10,
    'occluder_full': 11, 'reflection': 12, 'crowd': 13,
}
valid_classes = ['pedestrian']   # CLASSES_TO_EVAL chỉ nhận giá trị này
```

Ở Giai đoạn 1, ta gán `class = 1` cho **mọi** bbox xe. Vì `1 == class_name_to_class_id['pedestrian']`,
truyền `CLASSES_TO_EVAL=['pedestrian']` sẽ đánh giá **đúng** các bbox xe của ta
— TrackEval chỉ so khớp theo số, không quan tâm tên gọi ngữ nghĩa là gì.

> **Đã sửa 1 lỗi phát hiện khi đọc mã nguồn:** ở Giai đoạn 1, tùy chọn
> `--ignore-as-distractor` của `to_motchallenge.py` dùng nhầm `class = 13`
> ("crowd" theo bảng chuẩn) thay vì `class = 8` ("distractor" — lớp thực sự
> được bước tiền xử lý của TrackEval tự động loại bỏ). Đã sửa lại hằng số
> `DISTRACTOR_CLASS = 8`. Vì cờ này chưa từng được bật trong lần chạy nào ở
> Giai đoạn 1 (mặc định script chỉ ghi `ignored_regions.txt` riêng), không có
> dữ liệu nào đã tạo ra bị ảnh hưởng.

**Cách TrackEval dò thư mục** (đã đối chiếu `MotChallenge2DBox.__init__`):

```
gt_set = BENCHMARK + '-' + SPLIT_TO_EVAL
GT      : {GT_FOLDER}/{gt_set}/{seq}/gt/gt.txt
seqmap  : {GT_FOLDER}/seqmaps/{gt_set}.txt
tracker : {TRACKERS_FOLDER}/{gt_set}/{tên_tracker}/data/{seq}.txt
```

Với `BENCHMARK='DETRAC'`, `SPLIT_TO_EVAL='sample'`, cấu trúc thư mục ta đã tạo
sẵn ở Giai đoạn 1 (`to_motchallenge.py --split-name DETRAC-sample`) khớp
**hoàn toàn**, không cần đổi tên gì thêm.

**3 nhóm metric được tính:**
- **HOTA** = √(DetA × AssA) — cân bằng giữa độ chính xác phát hiện (DetA) và độ
  chính xác liên kết track (AssA). Là metric được khuyến dùng nhất hiện nay vì
  không thiên vị về 1 phía như MOTA/IDF1.
- **CLEAR** (cho MOTA) = 1 − (FN+FP+IDSW)/GT — thiên về độ bao phủ phát hiện,
  ít nhạy cảm với lỗi liên kết ID.
- **Identity** (cho IDF1) — đo độ chính xác gán đúng track id xuyên suốt video,
  nhạy cảm với ID switch hơn MOTA → phù hợp để đo riêng ảnh hưởng của occlusion
  lên khả năng giữ đúng ID của motion model (câu hỏi trọng tâm của đề tài).

---

## Kết quả Giai đoạn 1

### Toàn vẹn dữ liệu
| Chỉ số | Giá trị |
|---|---|
| Số video (annotation XML) | 60 |
| Frame có annotation | 82.085 |
| Bounding box | 598.281 |
| Track (xe) | 5.952 |
| Vùng chồng lấn `region_overlap` | 174.220 |
| Vùng `ignored_region` | 244 |

Không có bbox lỗi, không có bản ghi trùng `(video, frame, track_id)`, không có
bbox vượt khung ảnh, `visibility` đều nằm trong `[0, 1]`.

### Phân bố che khuất
| Nguồn che khuất | Số bbox | Tỉ lệ |
|---|---|---|
| Không bị che | 437.761 | 73.17% |
| Bị **xe khác** che | 83.938 | 14.03% |
| Bị **vật nền** che | 71.344 | 11.92% |
| Cả hai | 5.238 | 0.88% |

`occlusion_ratio` (trên các bbox bị che): median **0.342**, phân vị 90 = 0.910,
max = 1.000.

### Ba mức che khuất

**A. Che khuất một phần** (`ratio ≥ 0.10`) — **2.765 đoạn**, 37.6% số track dính phải.

| Nhóm | Số đoạn | Tỉ lệ | Trong đó che nặng (≥0.7) |
|---|---|---|---|
| `short` (<0.5s) | 609 | 22.0% | 55 |
| `medium` (0.5–1.5s) | 916 | 33.1% | 175 |
| `long` (>1.5s) | 1.240 | 44.8% | 506 |

Nguồn: 50.3% do xe khác, 47.7% do vật nền, 2.0% cả hai.

**B. Che khuất gần như hoàn toàn** (`ratio ≥ 0.90`) — **584 đoạn**, 420/5.952
track (7.1%). Median 7 frame, max 2.225 frame. Xe di chuyển được median 30 px
(max 906 px) trong lúc "vô hình".

| Nhóm | Số đoạn | Tỉ lệ |
|---|---|---|
| `short` (<0.5s) | 370 | 63.4% |
| `medium` (0.5–1.5s) | 157 | 26.9% |
| `long` (>1.5s) | 57 | 9.8% |

> **Đây là kịch bản then chốt của đề tài.** Xe vẫn có trong ground truth nhưng
> gần như vô hình trong ảnh → detector gần như chắc chắn không ra detection →
> tracker buộc phải ngoại suy vị trí **hoàn toàn** bằng motion model. Chính ở
> đây CV (đi thẳng đều) và CTRV (có tốc độ quay) sẽ cho kết quả khác nhau rõ rệt.

**C. Track gap** (xe biến mất khỏi annotation rồi quay lại) — **0 / 5.952 track**.

> UA-DETRAC annotate **liên tục** mọi frame từ lúc xe xuất hiện đến lúc rời khung
> hình, kể cả khi bị che 100%. Đây là điểm cần nêu rõ trong luận văn: không thể
> dùng "khoảng trống annotation" làm định nghĩa che khuất hoàn toàn trên bộ dữ
> liệu này — phải dùng `occlusion_ratio` như mục B.

### Phân nhóm độ cong quỹ đạo

| Nhóm | Số track | Tỉ lệ |
|---|---|---|
| `straight` | 3.417 | 57.4% |
| `curved` | 2.238 | 37.6% |
| `unknown` (track quá ngắn) | 179 | 3.0% |
| `static` (xe đỗ) | 118 | 2.0% |

Theo góc rẽ thực tế: `straight` (<10°) 36.5%, `gentle` (10–45°) 49.4%,
`sharp` (≥45°) 9.1%.

So sánh 2 nhóm (giá trị median):

| | n | Góc rẽ | Chỉ số thẳng | `kappa` | `omega` p90 | Tốc độ |
|---|---|---|---|---|---|---|
| `curved` | 2.238 | 32.6° | 0.9917 | 0.0028 /px | 1.328 rad/s | 96 px/s |
| `straight` | 3.417 | 7.8° | 0.9988 | 0.0007 /px | 0.612 rad/s | 218 px/s |

Nhóm `curved` có độ cong **gấp 4 lần** và tốc độ quay **gấp 2 lần** nhóm
`straight`, đồng thời đi chậm hơn (xe giảm tốc khi vào cua) — đúng như kỳ vọng
vật lý. Hình minh hoạ: `results/curvature_examples.png`.

> **Lưu ý cần nêu trong luận văn:** CTRV ở đây được áp dụng trong **hệ toạ độ
> ảnh**, không phải toạ độ mặt đường. Do phối cảnh, một xe đi thẳng vẫn có thể
> tạo ra quỹ đạo hơi cong trên ảnh. Đây là lựa chọn mô hình hoá tiêu chuẩn trong
> MOT (ByteTrack/SORT đều làm việc trên toạ độ ảnh), nhưng nên nói rõ để tránh bị
> hội đồng hỏi vặn.

---

## Kết quả Giai đoạn 2 — Baseline YOLOv8 + ByteTrack

### Môi trường chạy
GPU khả dụng (CUDA), tốc độ đo được **~55–90 FPS** tuỳ video (yolov8n +
ByteTrack, ảnh 960×540). Với tốc độ này, chạy đủ 60 video train (~83.791 ảnh)
ở Giai đoạn 5 dự kiến mất **~20–25 phút**.

### Chọn video mẫu
5 video được chọn từ bảng thống kê `data/interim/video_selection_stats.csv`
(tính từ dữ liệu Giai đoạn 1) để phủ đa dạng thời tiết / mức che khuất / độ cong:

| Video | Thời tiết | Occlusion (% track) | % track cong | Lý do chọn |
|---|---|---|---|---|
| `MVI_20011` | sunny | 32.1% | 67.9% | Đại diện "trung bình", cũng là ví dụ xuyên suốt tài liệu Giai đoạn 1 |
| `MVI_40962` | night | 1.1% | 52.2% | Đối chứng "dễ" — gần như không occlusion |
| `MVI_39811` | night | 100% | 0% | Cực đoan: **mọi** track đều bị che ít nhất 1 lần |
| `MVI_40204` | cloudy | 40.8% | 43.0% | Occlusion nặng nhất tập train (36 đoạn che ≥90%) |
| `MVI_63521` | rainy | 18.9% | 70.0% | Độ cong cao nhất (32 khúc cua gấp) + thời tiết mưa |

### Kết quả TrackEval (HOTA / MOTA / IDF1)

| Video | HOTA | DetA | AssA | MOTA | IDF1 | FP | FN | IDSW |
|---|---|---|---|---|---|---|---|---|
| MVI_20011 | 0.534 | 0.508 | 0.563 | 0.605 | 0.749 | 261 | 2.747 | 13 |
| MVI_39811 | 0.553 | 0.478 | 0.641 | 0.451 | 0.718 | 191 | 135 | 3 |
| MVI_40204 | 0.624 | 0.591 | 0.661 | 0.709 | 0.817 | 983 | 5.466 | 70 |
| MVI_40962 | 0.658 | 0.647 | 0.671 | 0.719 | 0.848 | 751 | 1.363 | 18 |
| MVI_63521 | 0.692 | 0.681 | 0.704 | 0.797 | 0.882 | 430 | 2.613 | 25 |
| **COMBINED** | **0.637** | **0.610** | **0.667** | **0.718** | **0.830** | **2.616** | **12.324** | **129** |

**Quan sát bước đầu** (chỉ mang tính tham khảo — 5 video, chưa đủ để kết luận
thống kê, phân tích chính thức sẽ ở Giai đoạn 5):
- `MVI_40962` (gần như không occlusion) đạt HOTA/IDF1 cao thứ nhì dù traffic
  đông (90 track) — occlusion thấp giúp giữ ID tốt hơn.
- `MVI_20011` (occlusion trung bình, nhiều khúc cua) có HOTA **thấp nhất**
  (0.534) dù ít xe nhất (53 track) — gợi ý chuyển động cong đã gây khó khăn
  ngay cả cho detector/tracker baseline dùng model CV.
- `MVI_63521` (occlusion thấp nhất trong nhóm "khó", độ cong cao nhất) lại có
  HOTA **cao nhất** — cho thấy ở baseline này, occlusion ảnh hưởng rõ hơn độ
  cong; cần tập dữ liệu lớn hơn ở Giai đoạn 5 để tách bạch 2 yếu tố.
- IDP (0.922) cao hơn nhiều IDR (0.754) ở mức tổng hợp → tracker **bỏ sót**
  (miss) nhiều hơn là **gán nhầm ID**; phần lớn lỗi đến từ FN (12.324) hơn là
  IDSW (129) — dấu hiệu detector pretrained COCO chưa tối ưu cho góc camera
  giám sát giao thông (xe nhỏ/xa, góc nhìn từ trên cao) hơn là do motion model.

### Kiểm chứng trực quan
`results/baseline_example_MVI_40204_frame300.png` — so sánh ảnh gốc / GT / kết
quả tracker trên cùng 1 frame. Xác nhận bằng mắt: vùng `ignored_region` (viền
đỏ) chứa 2 xe thật trong ảnh gốc nhưng **không** có box nào được tracker gán ở
đó (mask hoạt động đúng); các box GT và tracker khớp vị trí tốt; xe bị che một
phần (viền cam trong panel GT) vẫn được detector bám theo.

### ✅ Baseline đã khoá
Đã quyết định: giữ nguyên `yolov8n.pt` + `conf=0.25` + ByteTrack mặc định
(`bytetrack.yaml`) làm cấu hình **cố định** cho toàn bộ đề tài (đánh dấu trong
`config.py`). Lý do: mục tiêu chính của khoá luận là so sánh **motion model**
(KF/EKF/UKF) — dù FN hiện chiếm đa số lỗi (12.324 so với FP 2.616, có thể cải
thiện bằng model lớn hơn hoặc hạ `conf`), việc tối ưu detector không phải trọng
tâm và sẽ làm phép so sánh giữa Giai đoạn 3–5 mất công bằng nếu thay đổi giữa
chừng. Từ Giai đoạn 3 trở đi, **chỉ motion model thay đổi**; detector + data
association giữ nguyên để đảm bảo cả 3 phương pháp nhận cùng một đầu vào detection.

---

## Tải bộ ảnh train (thủ công)

Annotation XML đã tải xong (13.1 MB, 60 video). Còn thiếu **bộ ảnh train**.

### Tình trạng các nguồn (đã kiểm tra thực tế)

| Nguồn | Trạng thái |
|---|---|
| `detrac-db.rit.albany.edu/Data/DETRAC-train-data.zip` (chính thức) | ❌ **Chết** — server trả về trang HTML redirect, không còn phục vụ file |
| Trang download chính thức | ❌ Redirect sang trang lab chung của UAlbany, không còn link tải |
| HuggingFace `ShantyCam/ua-detrac` | ✅ **Sống, đã xác minh nội dung** |
| HuggingFace `abhineet123/ua_detrac` | ⚠️ Sống nhưng đã đổi cấu trúc thư mục |
| Kaggle | ⚠️ Cần đăng nhập, không xác minh được cấu trúc từ xa |

### Nguồn khuyến nghị

**`ua-detrac-orig.zip` — 9.91 GB** ([HuggingFace](https://huggingface.co/datasets/ShantyCam/ua-detrac))

```
https://huggingface.co/datasets/ShantyCam/ua-detrac/resolve/main/ua-detrac-orig.zip
```

Tôi đã đọc *central directory* của file zip này từ xa bằng HTTP Range (không cần
tải về) để xác minh nội dung thật:

- **141.039 entry**, giữ **nguyên cấu trúc gốc**: `DETRAC-Images/DETRAC-Images/MVI_20011/img00001.jpg`
- Đủ **60/60 video train** — tổng **83.791 ảnh**
- Kèm cả 40 video test, `DETRAC-Train-Annotations-XML`, `DETRAC-Test-Annotations-XML`,
  và `DETRAC-MOT-toolkit`
- **Đối chiếu chéo với annotation đã parse: 60/60 video khớp**, 59/60 video có
  `max_frame_num` bằng đúng số ảnh. Chênh lệch 83.791 ảnh − 82.085 frame có
  annotation = 1.706 frame không có xe nào (2.04%) — hợp lý.

Tải bằng trình duyệt, hoặc bằng lệnh có hỗ trợ **resume** (file lớn, nên dùng):

```bash
# curl: -C - để tải tiếp nếu bị đứt giữa chừng
curl -L -C - -o "data/raw/ua-detrac-orig.zip"   "https://huggingface.co/datasets/ShantyCam/ua-detrac/resolve/main/ua-detrac-orig.zip"
```

```bash
# hoặc dùng huggingface_hub (tự resume, tự kiểm tra hash)
python -m pip install huggingface_hub
huggingface-cli download ShantyCam/ua-detrac ua-detrac-orig.zip   --repo-type dataset --local-dir data/raw
```

Đặt file vào `data/raw/` (tên gì cũng được — script tự tìm file zip lớn nhất).

### Nguồn dự phòng

**`ua_detrac_training_set.zip` — 5.63 GB** ([HuggingFace](https://huggingface.co/datasets/abhineet123/ua_detrac))
— nhẹ hơn 4 GB vì chỉ có tập train, nhưng **đã đổi tên thư mục và tên ảnh**:
`detrac_60_MVI_63563/image000001.jpg` thay vì `MVI_63563/img00001.jpg`.
Dùng được nhưng phải viết thêm bước đổi tên. Chỉ chọn nếu bạn bị giới hạn dung lượng ổ đĩa.

### Sau khi tải xong — chạy 2 lệnh này

```bash
python src/extract_verify.py      # giải nén + đối chiếu với annotation
python src/to_motchallenge.py     # chạy lại để cập nhật seqLength cho chính xác
```

`extract_verify.py` sẽ:

1. **Tự tìm thư mục ảnh** dù zip lồng thư mục kiểu gì — đã test với 3 kiểu bố cục
   (`Insight-MVT_Annotation_Train/`, `DETRAC-Images/DETRAC-Images/`, lồng 3 cấp).
   Bạn **không cần sắp xếp lại thư mục**.
2. **Chỉ giải nén 60 video train**, bỏ qua 40 video test — tiết kiệm khoảng 4 GB
   ổ đĩa. Tập test không có ground truth công khai nên vô dụng với đề tài này.
   Dùng `--all-sequences` nếu bạn vẫn muốn giải nén hết.
3. Đối chiếu số ảnh với annotation và báo ngay nếu lệch.

Kết quả mong đợi: `60 video, status = OK`, tổng ảnh **83.791**, frame không có xe **1.706 (2.04%)**.

## Các file dữ liệu sinh ra

| File | Nội dung |
|---|---|
| `data/interim/dataset_integrity.csv` | Đối chiếu XML ↔ ảnh theo từng video |
| `data/interim/detrac_train_annotations.parquet` | Bảng chính, 1 dòng = 1 bbox (598.281 dòng) |
| `data/interim/detrac_train_annotations_sample.csv` | 2.000 dòng đầu, mở bằng Excel để xem |
| `data/interim/detrac_ignored_regions.csv` | 244 vùng bỏ qua |
| `data/interim/detrac_sequence_attributes.csv` | Thời tiết / trạng thái camera từng video |
| `data/interim/occlusion_segments.csv` | 2.765 đoạn che khuất một phần |
| `data/interim/full_occlusion_segments.csv` | 584 đoạn che khuất ≥90% |
| `data/interim/track_gaps.csv` | (rỗng — không có track gap) |
| `data/interim/track_occlusion_summary.csv` | Tổng hợp che khuất theo từng track |
| `data/interim/track_curvature.csv` | Độ cong + nhãn thẳng/cong theo từng track |
| `data/interim/frame_kinematics.parquet` | Động học từng frame: `vx, vy, speed, heading, omega, kappa, radius` |
| `data/processed/DETRAC-all/` | Ground truth format MOTChallenge (60 video) |
| `data/processed/DETRAC-sample/` | Ground truth format MOTChallenge (5 video mẫu, Giai đoạn 2) |
| `data/processed/trackers/DETRAC-sample/yolov8n-bytetrack/data/*.txt` | Kết quả baseline tracker |
| `data/interim/video_selection_stats.csv` | Thống kê mỗi video (occlusion/curvature/thời tiết) — dùng chọn video mẫu |
| `data/interim/baseline_track_summary_yolov8n-bytetrack.csv` | Thời gian chạy, số bbox/track theo video |
| `results/curvature_examples.png` | Minh hoạ quỹ đạo theo nhóm độ cong |
| `results/baseline_example_MVI_40204_frame300.png` | Minh hoạ trực quan GT vs baseline tracker |
| `results/trackeval/DETRAC-sample/yolov8n-bytetrack/*.csv` | Kết quả HOTA/MOTA/IDF1 chi tiết + tổng hợp |
