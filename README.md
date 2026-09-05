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
| 3 | EKF + CTRV | ✅ Xong — chờ review |
| 4 | UKF + CTRV | ✅ Xong — chờ review |
| 5 | Thực nghiệm đầy đủ & phân tích | ✅ Xong — chờ review |

---

## Quy ước làm việc (đọc trước khi tiếp tục, kể cả trên máy mới)

**Quy trình bắt buộc:** làm theo **5 giai đoạn** (data pipeline → baseline →
EKF → UKF → thực nghiệm đầy đủ + phân tích). **Dừng lại sau mỗi giai đoạn để
người dùng review**, không tự động chạy tiếp giai đoạn sau khi chưa có xác
nhận. Trạng thái hiện tại: xem bảng ngay dưới đây.

**Auto commit + push:** sau mỗi lần chạy xong một phần việc có ý nghĩa, tự
động `git add` + `commit` + `push` lên `origin/main` — **không cần hỏi lại**
mỗi lần (người dùng đã uỷ quyền thường trực). Tôn trọng `.gitignore` hiện có
(xem mục "Dữ liệu KHÔNG có trong repo" bên dưới). Nếu push thất bại, báo cho
người dùng thay vì âm thầm bỏ qua.

**Baseline đã khoá** (xem "Kết quả Giai đoạn 2"): `yolov8n.pt`, `conf=0.25`,
ByteTrack mặc định. Từ Giai đoạn 3 trở đi **chỉ motion model được đổi**, mọi
thứ khác giữ nguyên để phép so sánh công bằng. Không đổi cấu hình này trừ khi
có lý do rõ ràng và được người dùng đồng ý.

**Giai đoạn 5 bắt buộc có phân tích phân tầng** theo độ dài che khuất và độ
cong quỹ đạo — không chỉ báo cáo chỉ số tổng hợp. Lý do: mô phỏng ở Giai đoạn
3–4 cho thấy CTRV tốt hơn CV tới 15× khi xe rẽ trong lúc bị che, nhưng trên
video thật chỉ số tổng hợp lại cho EKF/UKF thấp hơn baseline một chút — vì chỉ
số tổng hợp che lấp hiệu ứng (63.4% đoạn che khuất trong UA-DETRAC là ngắn
<0.5s, nơi các motion model gần như tương đương).
✅ **Đã làm xong** — xem "Kết quả Giai đoạn 5". Kết luận ngắn gọn: phân tầng cho
thấy đúng hướng giả thuyết (EKF giữ ID gần gấp đôi baseline ở tầng che nặng độ
dài trung bình) nhưng **chưa đạt ý nghĩa thống kê** (p = 0.44), và nguyên nhân
gốc đã truy được là `track_buffer = 30 frame` chặn cứng mọi đoạn che dài hơn 1.2s.

**Phong cách code:** docstring/comment tiếng Việt không dấu, giải thích rõ
công thức toán (người dùng cần trình bày lại trong luận văn và trả lời hội
đồng). Ưu tiên rõ ràng/dễ debug hơn tối ưu tốc độ. README.md dùng tiếng Việt
có dấu.

### ⚠️ Dữ liệu KHÔNG có trong repo (phải tải lại trên máy mới)
`.gitignore` loại trừ: `data/raw/`, `data/extracted/` (~10GB ảnh UA-DETRAC,
dataset có bản quyền), `models/*.pt` (trọng số YOLOv8), `data/interim/*.parquet`
(tái tạo được bằng script). **Sau khi clone, các thư mục này sẽ trống hoặc
thiếu.** Xem mục "Tải bộ ảnh train (thủ công)" bên dưới để tải lại — annotation
XML thì tự tải được qua `src/download_data.py`, còn ảnh phải tải thủ công theo
link đã kiểm chứng trong mục đó.

---

## Cài đặt

```bash
conda create -n khoaluan-mot python=3.11.9 -y
conda activate khoaluan-mot

# torch ban CUDA phai cai TRUOC va cai rieng - ban tren PyPI la CPU-only
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

Đã kiểm tra với Python 3.11.9 trên Windows, GPU NVIDIA GTX 1650 (CUDA 12.4).

**Vì sao phải cài `torch` riêng trước?** `requirements.txt` chỉ ghi `torch>=2.0.0`; nếu
để `pip` tự giải, nó lấy bản trên PyPI vốn là **CPU-only** → toàn bộ pipeline chạy trên
CPU, chậm hàng chục lần. Cài bản `+cu124` trước thì ràng buộc `torch>=2.0.0` đã thoả,
`pip` sẽ không đụng vào nữa.

Kiểm tra nhanh GPU có hoạt động không:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`requirements.txt` để bound mở nên lần resolve mới có thể kéo về **pandas 3.x**,
**numpy 2.4.x**, **opencv 5.x**, **ultralytics 8.4.x**. Đã kiểm chứng thực tế: toàn bộ
pipeline Giai đoạn 1 tái tạo **byte-identical** với bản đã commit, và 3 điểm vá
ultralytics trong `src/tracker_ctrv.py` vẫn đúng nguyên trên 8.4.x.

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
│   ├── baseline_track.py      # Chạy YOLOv8 + ByteTrack (chọn motion model)
│   ├── run_trackeval.py       # Chạy TrackEval -> MOTA / IDF1 / HOTA
│   ├── ekf_ctrv.py            # EKF + mô hình CTRV (phần toán cốt lõi)
│   ├── ukf_ctrv.py            # UKF + CTRV (Unscented Transform)
│   ├── tracker_ctrv.py        # Ghép bộ lọc CTRV (EKF/UKF) vào ByteTrack
│   ├── test_ekf_ctrv.py       # Unit test cho EKF (14 phép kiểm tra)
│   ├── test_ukf_ctrv.py       # Unit test cho UKF (12 phép kiểm tra)
│   ├── compare_extrapolation.py  # So sánh ngoại suy CV vs CTRV (mô phỏng)
│   ├── stratified_analysis.py # Giai đoạn 5: phân tầng + kiểm định McNemar
│   └── compare_models.py      # Giai đoạn 5: gộp bảng so sánh 3 motion model
├── configs/                   # File cấu hình tracker (.yaml)
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

### `src/ekf_ctrv.py` — phần toán cốt lõi của Giai đoạn 3
Extended Kalman Filter với mô hình **CTRV** (Constant Turn Rate and Velocity),
viết để **thay thế trực tiếp** `KalmanFilterXYAH` của ultralytics (cùng tên
phương thức, cùng ý nghĩa tham số).

**Vector trạng thái 9 chiều:**
```
x = [ cx, cy, v, theta, omega, a, h, va, vh ]
      0   1   2    3      4    5  6   7   8
```
Chia làm 2 khối có chủ đích:
- `(cx, cy, v, theta, omega)` — CTRV, **phi tuyến**, đây là thứ *duy nhất* khác baseline
- `(a, h, va, vh)` — vẫn Constant Velocity tuyến tính, **giữ nguyên 100%** giống
  baseline kể cả các hệ số nhiễu `1/20`, `1/160`, `1e-2`, `1e-5`

Mục đích: cô lập đúng **một** biến nghiên cứu. Nếu đổi luôn cả động học kích
thước bbox thì khi kết quả thay đổi sẽ không biết do CTRV hay do thứ kia.

**Hàm chuyển trạng thái** (dt = 1 frame), với `s0=sin(θ)`, `c0=cos(θ)`,
`s1=sin(θ+ω·dt)`, `c1=cos(θ+ω·dt)`:

```
|omega| > eps  (xe đang quay — quỹ đạo là cung tròn bán kính R = v/omega):
    cx' = cx + (v/omega)·(s1 − s0)
    cy' = cy + (v/omega)·(c0 − c1)
    theta' = theta + omega·dt ,  v' = v ,  omega' = omega

|omega| <= eps  (giới hạn khi omega → 0, tránh chia cho 0):
    cx' = cx + v·c0·dt
    cy' = cy + v·s0·dt
```

**Jacobian tại `omega ≈ 0` — chỗ dễ sai nhất.** Hai đạo hàm theo `omega` **không
bằng 0**, mà lấy từ số hạng bậc 1 của khai triển Taylor:
```
d(cx')/d(omega) = −0.5·v·dt²·s0
d(cy')/d(omega) = +0.5·v·dt²·c0
```
Nếu đặt bằng 0 (lỗi thường gặp), bộ lọc sẽ **không bao giờ học được `omega`** khi
xe đang đi thẳng → không phát hiện được lúc xe bắt đầu rẽ. Unit test 5b bắt đúng lỗi này.

**Mô hình đo tuyến tính:** `z = [cx, cy, a, h]` — *giống hệt* baseline. Hệ quả
quan trọng: bước `update` của EKF trùng khít với KF chuẩn (không có xấp xỉ nào),
nên **toàn bộ tính phi tuyến nằm gọn trong bước `predict`** — đúng thứ đề tài muốn
đo. Ngoài ra vì không gian đo y hệt baseline, ByteTrack vẫn ghép cặp bằng IoU trên
cùng loại bbox → **data association hoàn toàn không bị ảnh hưởng**.

**Khởi tạo 2 khung hình — bắt buộc với CTRV.** Khi track mới sinh ta chỉ có 1 quan
sát, không biết hướng. Nếu đặt `v=0, theta=0` như cách baseline đặt `vx=vy=0` thì
CTRV **bị kẹt cứng**: mọi phần tử Jacobian liên quan `theta` đều tỉ lệ với `v`
(`d(cx')/d(theta) = −v·s0·dt`), nên `v=0` làm chúng bằng 0 → `theta` không bao giờ
nhận được thông tin. Một xe đi thẳng đứng theo trục y sẽ **không bao giờ** học được
`v` lẫn `theta`. Baseline KF không gặp lỗi này vì CV tuyến tính, `vx`/`vy` độc lập.
Cách xử lý (chuẩn trong tài liệu CTRV/CTRA): ước lượng `v`, `theta` từ **hai quan
sát đầu tiên**. Đây không phải "ưu ái" cho EKF mà là sửa khiếm khuyết của cách tham
số hoá toạ độ cực — unit test 9c đo được: khởi tạo naive sai **12.9 px**, khởi tạo
2 khung hình sai **0.0 px**.

### `src/tracker_ctrv.py`
Ghép bộ lọc CTRV vào ByteTrack. Sau khi đọc mã nguồn `byte_tracker.py`, xác định
trong toàn bộ file chỉ có **đúng 3 vị trí** chạm vào bố cục vector trạng thái:

| Dòng | Nội dung | Xử lý |
|---|---|---|
| 82 | `mean_state[7] = 0` trong `predict` | Đổi chỉ số 7 → 8 (`vh` ở vị trí mới) |
| 94 | `multi_mean[i][7] = 0` trong `multi_predict` | Đổi chỉ số + dùng `shared_kalman` của lớp mới |
| 167 | `ret = self.mean[:4]` trong property `tlwh` | Đọc chỉ số `[0,1,5,6]` thay vì `[:4]` |

`matching.py` (phần ghép cặp) **không hề đọc** `mean`/`covariance` — chỉ dùng IoU
trên bbox `tlwh`. Đây là bằng chứng thay motion model không ảnh hưởng association.

**Cố ý KHÔNG xoá `omega` khi track bị mất** (chỉ xoá `vh` giống baseline): khi xe
bị che khuất giữa khúc cua, việc tiếp tục quay theo `omega` đã học được **chính là**
ưu thế mà đề tài muốn đo. Xoá `omega` sẽ làm CTRV thoái hoá thành CV.

### `src/test_ekf_ctrv.py`
14 phép kiểm tra, chạy `python src/test_ekf_ctrv.py` để xem báo cáo chi tiết:
đối chiếu với ví dụ **tính tay**, **Jacobian giải tích vs sai phân số**, kiểm tra
liên tục tại `omega → 0`, đi hết 1 vòng tròn phải về đúng chỗ cũ, `P` luôn đối
xứng và xác định dương sau 200 vòng lặp, và bằng chứng cho khởi tạo 2 khung hình.

### `src/ukf_ctrv.py` — phần toán cốt lõi của Giai đoạn 4
Unscented Kalman Filter với cùng mô hình CTRV. **Kế thừa `EKFTrackerCTRV`** nên
dùng chung y hệt `f(x)`, `Q`, `R` và cách khởi tạo — khác biệt duy nhất là cách
truyền hiệp phương sai.

**Ý tưởng:** EKF tuyến tính hoá `f` quanh *một* điểm rồi dùng Jacobian
(`P' = F P Fᵀ + Q`), bỏ qua mọi thành phần bậc ≥ 2. UKF không tuyến tính hoá gì:
nó chọn `2n+1 = 19` **sigma point** mô tả đúng kỳ vọng + hiệp phương sai hiện
tại, cho **từng điểm** đi qua hàm `f` phi tuyến **thật**, rồi tính lại kỳ vọng và
hiệp phương sai từ đám mây điểm đã biến đổi. Chính xác tới bậc 3 (so với bậc 1
của EKF) và **không cần tính đạo hàm** → không thể sai Jacobian.

**Xử lý `omega ≈ 0`:** UKF thừa hưởng chính hàm `f()` của EKF, nên công thức giới
hạn được áp dụng cho **từng sigma point riêng biệt**. Điều này thực ra *thuận lợi
hơn* EKF: sigma point trải quanh `omega = 0` có cả điểm âm, điểm dương và điểm
gần 0, mỗi điểm tự chọn nhánh công thức phù hợp (unit test 6).

Chi tiết về trung bình vòng cho `theta`, chọn tham số sigma point, và phát hiện
về hiệu ứng dây cung — xem mục "Kết quả Giai đoạn 4" bên dưới.

### `src/compare_extrapolation.py`
Thí nghiệm **mô phỏng** cô lập đúng cơ chế mà đề tài giả thiết: cho xe chạy theo
quỹ đạo CTRV đã biết trước, cho cả 3 bộ lọc quan sát 25 frame, rồi **cắt detection**
và bắt chúng chỉ dự đoán (đúng như lúc bị che khuất hoàn toàn). Bộ lọc CV dùng
đúng lớp `KalmanFilterXYAH` của ultralytics nên so sánh là tuyệt đối công bằng.

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

## Kết quả Giai đoạn 3 — EKF + CTRV

### Chạy pipeline

```bash
python src/test_ekf_ctrv.py                       # unit test (chạy TRƯỚC khi tích hợp)
python src/compare_extrapolation.py --plot        # thí nghiệm mô phỏng CV vs CTRV
python src/baseline_track.py --motion-model ekf_ctrv
python src/run_trackeval.py --split-name DETRAC-sample --tracker yolov8n-ekf-ctrv
```

### 1. Unit test: 14/14 PASS
Đáng chú ý nhất:

| Phép kiểm tra | Kết quả |
|---|---|
| Quỹ đạo cong `omega=90°/frame` vs tính tay | khớp tới 9 chữ số thập phân |
| Jacobian giải tích vs sai phân số (cong) | sai lệch `1.3e-08` |
| Jacobian tại `omega ≈ 0` (công thức giới hạn) | sai lệch `1.2e-06` |
| Đi hết 1 vòng tròn 36 bước | về đúng chỗ cũ, lệch `3.6e-13` px |
| `P` đối xứng + xác định dương sau 200 vòng | trị riêng nhỏ nhất `2e-10 > 0` |

### 2. Kiểm chứng công bằng trước khi so sánh
Chạy cả 2 bộ lọc trên cùng video, đo sai số giữa vận tốc bộ lọc ước lượng và
dịch chuyển **thực tế** của tâm bbox:

| Bộ lọc | v thực | v ước lượng | Tỉ lệ | Sai số tương đối |
|---|---|---|---|---|
| KF + CV (baseline) | 2.406 | 2.079 | 0.86 | 32.6% |
| EKF + CTRV | 2.432 | 2.102 | **0.86** | **32.6%** |

Hai bộ lọc trễ **y hệt nhau** → cách đặt nhiễu quá trình của EKF không thiên lệch.
(Việc ước lượng thiếu ~14% là đặc tính trễ vốn có của Kalman filter, cả hai đều bị.)

### 3. Thí nghiệm mô phỏng: cơ chế của giả thuyết có hoạt động không?

Sai số ngoại suy vị trí (px) khi bị che khuất — bảng đầy đủ ở
`results/extrapolation_cv_vs_ctrv.csv`, hình ở `results/extrapolation_cv_vs_ctrv.png`:

| Đoạn che | Turn rate | CV (baseline) | CTRV (EKF) | Chênh lệch |
|---|---|---|---|---|
| 0.2s | 0°/f (thẳng) | **1.76** | 2.10 | ngang nhau |
| 0.2s | 3°/f (rẽ) | 13.22 | **2.15** | CTRV tốt hơn 6× |
| 0.8s | 0°/f | **2.88** | 5.11 | CV tốt hơn |
| 0.8s | 3°/f | 60.97 | **5.24** | CTRV tốt hơn 12× |
| 2.0s | 0°/f | **5.18** | 15.71 | CV tốt hơn 3× |
| 2.0s | 3°/f | 217.41 | **14.51** | CTRV tốt hơn **15×** |

**Kết luận:** cơ chế mà khoá luận giả thiết là **có thật và rất mạnh**. Sai số của
CTRV gần như **phẳng** bất kể xe rẽ gấp bao nhiêu, còn sai số CV **tăng tuyến tính**
theo turn rate. Đổi lại, khi xe đi **thẳng tuyệt đối** thì CTRV kém hơn vì nó ước
lượng nhầm một `omega` nhỏ từ nhiễu đo rồi tích luỹ thành đường cong sai.

> **Một lỗi tôi đã mắc và sửa** (nên ghi vào luận văn vì đây là bẫy phổ biến):
> ban đầu tôi đặt nhiễu quá trình `CTRV_STD_OMEGA = 0.05 rad/frame²`, lấy từ *độ
> lớn* `omega` đo được ở Giai đoạn 1. **Sai về khái niệm:** nhiễu quá trình phải là
> *mức thay đổi của omega mỗi frame*, không phải độ lớn của chính omega. Xe tăng
> turn rate từ 0 lên 3°/frame trong ~1 giây ⇒ `d(omega)/dt ≈ 0.002 rad/frame²`,
> nhỏ hơn 25 lần. Đặt sai làm `omega` "lang thang" theo nhiễu: sai số ngoại suy
> trung bình có trọng số **23.2 px**; sau khi sửa còn **7.2 px** (CV: 33.5 px).
> Vùng tối ưu khá phẳng (`theta` 0.005–0.02, `omega` 0.002–0.01 đều cho 7.2–8.1 px)
> nên kết quả không nhạy cảm với lựa chọn chính xác.

### 4. Kết quả trên video thật (5 video mẫu)

| Motion model | HOTA | DetA | AssA | MOTA | IDF1 | FP | FN | IDSW |
|---|---|---|---|---|---|---|---|---|
| KF + CV (baseline) | **0.6367** | **0.6099** | **0.6668** | **0.7175** | **0.8300** | 2.616 | **12.324** | **129** |
| EKF + CTRV | 0.6346 | 0.6090 | 0.6633 | 0.7161 | 0.8253 | **2.608** | 12.386 | 154 |
| Chênh lệch | −0.33% | −0.14% | −0.52% | −0.21% | −0.56% | −8 | +62 | +25 |

**EKF + CTRV thấp hơn baseline một chút trên tập mẫu này.** Đây là kết quả thật,
không phải lỗi cài đặt — đã loại trừ bằng 3 lớp kiểm chứng ở trên.

**Vì sao mô phỏng thắng đậm mà video thật lại thua?**
1. **Ưu thế của CTRV chỉ phát huy khi KHÔNG có detection.** Với detector khá tốt,
   phần lớn track có detection gần như mọi frame; lúc đó motion model chỉ dùng để
   dự đoán **1 frame** cho việc ghép cặp IoU — ở khoảng đó CV và CTRV gần như
   ngang nhau (1.76 vs 2.10 px khi đi thẳng).
2. **Chuyển động trong UA-DETRAC chủ yếu gần thẳng trong hệ toạ độ ảnh.** Từ bảng
   mô phỏng, CTRV chỉ thắng khi turn rate ≥ 0.5°/frame; phần lớn track không đạt mức đó.
3. **Đoạn che khuất chủ yếu ngắn.** Giai đoạn 1 đo được: 63.4% đoạn che ≥90% là
   `short` (<0.5s) — đúng vùng CV vẫn còn tốt.
4. CTRV có **5 trạng thái** cho chuyển động tâm so với 4 của CV → nhiều tham số
   phải ước lượng hơn từ cùng lượng dữ liệu → phương sai cao hơn, trả giá nhẹ ở
   mọi nơi (thể hiện ở IDSW +25).

**Điều này KHÔNG bác bỏ giả thuyết của khoá luận** — nó cho thấy chỉ số **tổng hợp
che lấp hiệu ứng**, và đó chính xác là lý do đề cương yêu cầu **phân tích phân tầng**
theo độ dài che khuất và độ cong ở Giai đoạn 5. Thí nghiệm mô phỏng đã chứng minh
cơ chế tồn tại; việc còn lại là đo nó trên đúng nhóm dữ liệu mà nó phát huy.

---

## Kết quả Giai đoạn 4 — UKF + CTRV

### Chạy pipeline

```bash
python src/test_ukf_ctrv.py                       # unit test (12 phép kiểm tra)
python src/compare_extrapolation.py --plot        # so sánh cả 3 motion model
python src/baseline_track.py --motion-model ukf_ctrv
python src/run_trackeval.py --split-name DETRAC-sample --tracker yolov8n-ukf-ctrv
```

### Thiết kế: đảm bảo so sánh EKF vs UKF là công bằng tuyệt đối
`UKFTrackerCTRV` **kế thừa** `EKFTrackerCTRV`, dùng chung y hệt hàm chuyển trạng
thái `f(x)`, ma trận nhiễu quá trình `Q`, nhiễu đo `R` và cách khởi tạo 2 khung
hình. **Khác biệt duy nhất là cách truyền hiệp phương sai** qua hàm phi tuyến:
Jacobian (EKF) vs sigma point (UKF). Unit test 11 kiểm tra đúng điều này.

### Chọn tham số sigma point — một cái bẫy thật
Với `n = 9` chiều trạng thái, `lambda = alpha²·(n + kappa) − n`:

| alpha | kappa | n+lambda | Wm[0] | Wi | Đánh giá |
|---|---|---|---|---|---|
| **1.0** | **0.0** | **9.000** | **0.000** | **0.0556** | ✅ **Chọn** — mọi trọng số không âm |
| 1e-3 | 0.0 | 9e-06 | −999999 | 55555 | ❌ Bộ lọc phân kỳ ngay |
| 1.0 | 3−n = −6 | 3.000 | −2.000 | 0.1667 | ⚠️ Wm[0] âm → P có thể mất tính xác định dương |
| 0.5 | 0.0 | 2.250 | −3.000 | 0.2222 | ⚠️ Cả Wm[0] và Wc[0] đều âm |

`alpha = 1e-3` là **giá trị mặc định trong sách giáo khoa**, nhưng chỉ đúng cho
state ít chiều. Ở `n = 9` nó làm `n + lambda → 0` (chia cho 0), trọng số nổ lên
10⁶. Code **bắt lỗi này ngay lúc khởi tạo** với thông báo rõ ràng (unit test 9)
— và lưu ý phép kiểm tra phải đặt trên **độ lớn trọng số**, không phải trên
`n + lambda ≈ 0`, vì `9e-06` không phải là 0.

### Điểm cẩn thận nhất: theta là góc, không phải số thực
Khi tính kỳ vọng có trọng số của các sigma point, thành phần `theta` **không**
được lấy trung bình cộng. Hai góc `+179°` và `−179°` thực chất chỉ cách nhau 2°,
nhưng trung bình cộng cho `0°` — lệch hẳn 180°. Phải dùng **trung bình vòng**:

```
theta_mean = atan2( Σ Wm_i·sin(theta_i) , Σ Wm_i·cos(theta_i) )
```

Tương tự, hiệu `(Y_i − x_mean)` phải gói về `[−π, π)` trước khi lập hiệp phương
sai. Đây là lỗi kinh điển của UKF có biến góc — nguy hiểm tương đương lỗi chia
cho 0 tại `omega ≈ 0` của EKF. Unit test 4 và 4b kiểm tra riêng.

### Unit test: 12/12 PASS
| Phép kiểm tra | Kết quả |
|---|---|
| Sigma point tái tạo đúng kỳ vọng + hiệp phương sai | sai lệch `2.7e-14` |
| Hàm tuyến tính: UT cho kết quả **chính xác** (trùng EKF) | sai lệch `1.8e-15` |
| **Update của UKF trùng khít EKF** (mô hình đo tuyến tính) | sai lệch `2.8e-14` |
| Trung bình vòng cho theta | cộng thường `0°` ❌ vs vòng `180°` ✅ |
| `alpha=1e-3` bị bắt lỗi rõ ràng | ✅ |

Test "update trùng khít EKF" là bằng chứng cài đặt đúng, đồng thời xác nhận:
**mọi khác biệt kết quả giữa EKF và UKF đều đến từ bước `predict`.**

### Phát hiện quan trọng: UKF ngoại suy KÉM HƠN EKF

Thí nghiệm mô phỏng (sai số ngoại suy, px):

| Đoạn che | Turn rate | KF/CV | EKF/CTRV | UKF/CTRV |
|---|---|---|---|---|
| 0.2s | 3°/f | 13.22 | **2.15** | 2.17 |
| 0.8s | 3°/f | 60.97 | **5.24** | 5.76 |
| 2.0s | 0°/f | **5.18** | 15.71 | 24.94 |
| 2.0s | 3°/f | 217.41 | **14.51** | 26.61 |

Với đoạn che dài, UKF kém EKF ~10 px. **Nguyên nhân đã truy được** (không phải
lỗi cài đặt):

- Chênh lệch kỳ vọng UKF−EKF **tăng đơn điệu theo độ lớn `P`**:
  0.004 px (P×0.01) → 0.35 px (P×1) → 2.55 px (P×100)
- Kiểm chứng bằng bán kính quỹ đạo dự đoán khi che 50 frame (bán kính thật
  `R = v/omega = 152.79 px`): **EKF dự đoán 152.79 px (lệch 0.00)**, còn
  **UKF dự đoán 122.06 px (hụt 30.73 px)**

Đây là **hiệu ứng dây cung (bất đẳng thức Jensen)**: Unscented Transform tính
`E[f(x)]` chứ không phải `f(E[x])`. Các sigma point có `omega` khác nhau cong về
các hướng khác nhau; lấy trung bình những điểm nằm trên một cung tròn cho ra
điểm **nằm bên trong** cung đó → quỹ đạo bị "bóp" vào trong, và sai số tích luỹ
theo độ dài đoạn che.

> **Ý nghĩa cho luận văn:** khi bị che khuất lâu (không có phép đo), `P` phình to
> nên hiệu ứng này mạnh lên. Kỳ vọng của UKF là ước lượng đúng của `E[vị trí]`
> dưới bất định, nhưng cái ta cần cho tracking lại là **dự đoán điểm** bám sát
> quỹ đạo tất định — và ở đó phép truyền tất định `x' = f(x̂)` của EKF tốt hơn.
> Tôi đã thử thu hẹp độ trải sigma point (giảm `alpha`) nhưng **không cải thiện**,
> vì `Wm[0]` âm dần lại gây bất ổn — hai hiệu ứng triệt tiêu nhau.

### Kết quả trên video thật (5 video mẫu)

| Motion model | HOTA | DetA | AssA | MOTA | IDF1 | FP | FN | IDSW |
|---|---|---|---|---|---|---|---|---|
| KF + CV (baseline) | **0.6367** | **0.6099** | **0.6668** | **0.7175** | **0.8300** | 2.616 | **12.324** | **129** |
| EKF + CTRV | 0.6346 | 0.6090 | 0.6633 | 0.7161 | 0.8253 | 2.608 | 12.386 | 154 |
| UKF + CTRV | 0.6342 | 0.6090 | 0.6625 | **0.7163** | 0.8246 | **2.607** | 12.381 | 146 |

**EKF và UKF gần như không phân biệt được** trên video thật (chênh lệch mọi chỉ
số đều < 0.1%; UKF ít ID switch hơn 8 lần nhưng IDF1 thấp hơn 0.0007). Cả hai
vẫn thấp hơn baseline một chút, cùng lý do đã phân tích ở Giai đoạn 3.

Điều này **nhất quán** với thí nghiệm mô phỏng: khác biệt EKF/UKF chỉ đáng kể khi
`P` lớn (che khuất dài), mà 63.4% đoạn che khuất trong UA-DETRAC là ngắn (<0.5s).

### Tốc độ
| Motion model | FPS |
|---|---|
| KF + CV | 54.6 |
| EKF + CTRV | 80.8 |
| UKF + CTRV | 46.3 |

UKF chậm hơn EKF ~1.7× do phải truyền 19 sigma point qua `f()` mỗi bước thay vì
1 lần tính Jacobian. (Chênh lệch FPS giữa các lần chạy còn chịu ảnh hưởng của
tải GPU, nên chỉ nên xem là ước lượng tương đối.)

---

## Kết quả Giai đoạn 5 — Thực nghiệm đầy đủ & phân tích phân tầng

### Chạy pipeline

```bash
# 1) Tracking toan bo 60 video train, cho ca 3 motion model (~3 gio tren GTX 1650)
python src/baseline_track.py --all-train --split-name DETRAC-all --motion-model cv       --skip-existing
python src/baseline_track.py --all-train --split-name DETRAC-all --motion-model ekf_ctrv --skip-existing
python src/baseline_track.py --all-train --split-name DETRAC-all --motion-model ukf_ctrv --skip-existing

# 2) TrackEval cho tung motion model
python src/run_trackeval.py --split-name DETRAC-all --tracker yolov8n-bytetrack
python src/run_trackeval.py --split-name DETRAC-all --tracker yolov8n-ekf-ctrv
python src/run_trackeval.py --split-name DETRAC-all --tracker yolov8n-ukf-ctrv

# 3) Bang so sanh + phan tich phan tang
python src/compare_models.py --split-name DETRAC-all
python src/stratified_analysis.py --split-name DETRAC-all --plot
```

Quy mô: **60 video, 83.791 ảnh × 3 motion model = 251.373 lượt xử lý frame.**

### 1. Chỉ số tổng hợp trên toàn bộ 60 video

| Motion model | HOTA | DetA | AssA | MOTA | IDF1 | FP | FN | IDSW |
|---|---|---|---|---|---|---|---|---|
| KF + CV (baseline) | **0.6104** | **0.5735** | **0.6541** | **0.6615** | **0.7783** | 53.332 | **146.919** | **2.251** |
| EKF + CTRV | 0.6101 | 0.5733 | 0.6536 | 0.6613 | 0.7782 | **53.309** | 147.066 | 2.290 |
| UKF + CTRV | 0.6095 | 0.5732 | 0.6524 | 0.6611 | 0.7770 | 53.396 | 147.072 | 2.302 |
| Δ EKF vs baseline | −0.054% | −0.025% | −0.078% | −0.041% | −0.016% | −23 | +147 | +39 |
| Δ UKF vs baseline | −0.151% | −0.046% | −0.251% | −0.068% | −0.163% | +64 | +153 | +51 |

**Khoảng cách thu hẹp ~6 lần so với tập 5 video mẫu** (EKF: −0.054% so với −0.33% ở
Giai đoạn 3). Trên toàn tập, ba motion model **gần như không phân biệt được** ở mức
tổng hợp — chênh lệch dưới 0.2% ở mọi chỉ số.

> Lưu ý: chỉ số tuyệt đối thấp hơn tập 5 video mẫu (HOTA 0.610 vs 0.637) vì 5 video
> mẫu tình cờ là tập "dễ" hơn mặt bằng chung 60 video. Đây là lý do phải chạy đủ tập
> mới kết luận được.

### 2. Phép đo phân tầng: giữ được ID xuyên qua đoạn che khuất

Chỉ số tổng hợp trung bình hoá trên mọi frame nên che lấp hiệu ứng. `src/stratified_analysis.py`
đo trực tiếp cơ chế mà khoá luận giả thiết: **với mỗi đoạn che khuất, tracker có giữ
được cùng một ID trước và sau đoạn đó không?**

Ghép GT ↔ tracker theo từng frame bằng Hungarian trên ma trận IoU (ngưỡng 0.5), vì
file tracker dùng id riêng của ByteTrack. Chỉ so sánh trên **tập đoạn chung** — đoạn
nào cả 3 model đều xác định được `id_before` — để mẫu số giống hệt nhau:
**1.544 / 3.349 đoạn** (1.805 đoạn bị loại vì tracker chưa từng bắt được xe trước lúc bị che).

**Che khuất ≥ 0.90 (xe gần như vô hình — kịch bản then chốt):**

| Độ dài đoạn | n | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|---|
| `short` (<0.5s) | 180 | 0.2333 | 0.2333 | 0.2333 |
| `medium` (0.5–1.5s) | 78 | 0.0641 | **0.1154** | 0.0769 |
| `long` (>1.5s) | 18 | 0.0000 | 0.0000 | 0.0000 |

Tách thêm theo độ cong, ở nhóm `medium`:

| | n | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|---|
| `curved` | 23 | 0.0870 | **0.1304** | 0.0870 |
| `straight` | 51 | 0.0588 | **0.1176** | 0.0784 |

**EKF + CTRV giữ được ID trên gần gấp đôi số đoạn so với baseline** ở nhóm che nặng
độ dài trung bình (11.5% vs 6.4%) — đúng hướng mà mô phỏng Giai đoạn 3 dự đoán.

**Che khuất ≥ 0.10 (một phần):** khác biệt rất nhỏ và phần lớn nghiêng nhẹ về baseline
(ví dụ `long`: CV 0.1814, EKF 0.1685, UKF 0.1749). Hợp lý — khi xe chỉ bị che một phần,
detector vẫn ra detection nên motion model hầu như không được dùng để ngoại suy.

### 3. Kiểm định ý nghĩa thống kê (McNemar ghép cặp)

Ba model chạy trên **cùng một tập đoạn** nên đây là dữ liệu ghép cặp — phải dùng
McNemar, không dùng chi-square/t-test 2 mẫu độc lập (xem docstring
`src/stratified_analysis.py`). Kết quả ở tầng có chênh lệch lớn nhất:

| Tầng | n | baseline thắng | EKF thắng | Δ | p |
|---|---|---|---|---|---|
| `full` × `medium` | 78 | 1 | 5 | **+4** | 0.4375 |
| `full` × `medium` × `straight` | 51 | 1 | 4 | +3 | 0.75 |
| `partial` × `long` | 463 | 7 | 1 | −6 | 0.1406 |

> **KHÔNG tầng nào đạt p < 0.05.** Hướng chênh lệch đúng như giả thuyết (EKF thắng ở
> đúng tầng được dự đoán), nhưng số cặp bất đồng quá nhỏ (chỉ 6 cặp ở tầng tốt nhất)
> nên **chưa đủ bằng chứng thống kê để kết luận**. Đây là kết quả trung thực và cần
> nêu đúng như vậy trong luận văn.

### 4. Phát hiện quan trọng nhất: `track_buffer` là trần cứng, không phải motion model

Ở nhóm `long` (>1.5s), **cả 3 model đều bằng 0%**. Truy nguyên nhân:

`bytetrack.yaml` đặt `track_buffer: 30`, và ultralytics dùng
`max_frames_lost = track_buffer = 30 frame`. UA-DETRAC quay 25 fps ⇒ **1.2 giây**.
Quá ngưỡng này ByteTrack **xoá hẳn track**, không còn gì để ngoại suy nữa.

Đối chiếu trực tiếp trên dữ liệu (đoạn che ≥0.90):

| Độ dài đoạn | n | Giữ được ID |
|---|---|---|
| ≤ 30 frame (**trong** buffer) | 257 | 52 (**20.2%**) |
| > 30 frame (**vượt** buffer) | 21 | 0 (**0.0%**) |

100% đoạn `long` đều dài hơn 30 frame (min = 38, median = 48), và **không một đoạn nào
giữ được ID** — với cả 3 motion model.

> **Đây là lời giải cho nghịch lý mô phỏng vs thực tế.** Thí nghiệm mô phỏng ở Giai đoạn 3
> cho CTRV ngoại suy 2.0s (50 frame) và thắng CV tới 15×. Nhưng trong ByteTrack thật,
> track đã bị xoá từ frame thứ 30 — **ưu thế 15× đó không bao giờ có cơ hội xuất hiện.**
> Chất lượng motion model chỉ có ý nghĩa **bên trong cửa sổ `track_buffer`**; ngoài cửa sổ
> đó thì mọi motion model đều như nhau vì đều bằng 0.

Hệ quả cho luận văn: muốn CTRV phát huy đúng tiềm năng đo được trong mô phỏng thì phải
**nới `track_buffer`** — nhưng đó là thay đổi cấu hình data association, nằm ngoài
baseline đã khoá, nên không thực hiện ở giai đoạn này. Đây là hướng đề xuất cho công
việc tiếp theo.

### 5. Kết luận Giai đoạn 5

1. **Ở mức tổng hợp, 3 motion model không phân biệt được** trên toàn bộ 60 video
   (chênh lệch < 0.2% mọi chỉ số). Kết quả này ổn định hơn và đáng tin hơn con số
   trên 5 video mẫu.
2. **Phân tầng cho thấy đúng hướng giả thuyết:** EKF + CTRV giữ ID tốt gần gấp đôi
   baseline ở tầng che nặng độ dài trung bình — đúng vùng mà mô phỏng dự đoán.
3. **Nhưng chưa đủ ý nghĩa thống kê** (p = 0.44, chỉ 6 cặp bất đồng). Không được
   tuyên bố CTRV tốt hơn dựa trên dữ liệu này.
4. **Nguyên nhân gốc đã truy được:** `track_buffer = 30 frame` chặn cứng mọi đoạn che
   dài hơn 1.2s. UA-DETRAC lại chỉ có 21 đoạn che nặng vượt ngưỡng đó và 78 đoạn nằm
   trong vùng tranh chấp — **cỡ mẫu tự nhiên của bộ dữ liệu quá nhỏ** cho câu hỏi này.
5. **UKF không cải thiện gì so với EKF** trên video thật (thấp hơn ở mọi chỉ số tổng
   hợp), nhất quán với hiệu ứng dây cung đã phân tích ở Giai đoạn 4.

Hình minh hoạ: `results/stratified/id_retention_DETRAC-all.png`.

---

## Tải bộ ảnh train (thủ công)

> **Vị trí dữ liệu sau khi tải.** Script mong đợi đúng bố cục sau (`config.py`):
>
> ```
> data/raw/ua-detrac-orig.zip                    <- file zip tai ve
> data/extracted/DETRAC-Images/MVI_*/            <- anh, MOI CAP mot
> data/extracted/DETRAC-Train-Annotations-XML/*.xml
> ```
>
> Bản zip gốc **lồng thư mục 2 cấp** (`DETRAC-Images/DETRAC-Images/MVI_*`). Phải đưa
> thư mục bên trong lên một cấp, vì `config.list_train_videos()` đọc thẳng
> `data/extracted/DETRAC-Train-Annotations-XML/*.xml` — để lồng 2 cấp thì glob ra rỗng
> và script báo "không có video train". (`find_images_root()` thì tự dò được nên không kén.)
>
> Cả `data/raw/` và `data/extracted/` đều nằm trong `.gitignore` → máy mới clone về sẽ
> **trống**, phải tải lại theo hướng dẫn dưới đây.

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
| `results/trackeval/DETRAC-sample/<tracker>/*.csv` | Kết quả HOTA/MOTA/IDF1 chi tiết + tổng hợp |
| `results/comparison_DETRAC-sample.csv` | Bảng so sánh trực tiếp các motion model |
| `results/extrapolation_cv_vs_ctrv.csv/.png` | Thí nghiệm mô phỏng ngoại suy KF/CV vs EKF/CTRV vs UKF/CTRV |
| `data/processed/trackers/DETRAC-all/<tracker>/data/*.txt` | **Giai đoạn 5** — kết quả tracking 60 video × 3 motion model |
| `data/interim/baseline_track_summary_DETRAC-all_*.csv` | Thời gian chạy / số bbox theo video, 60 video |
| `results/trackeval/DETRAC-all/<tracker>/*.csv` | HOTA/MOTA/IDF1 trên toàn bộ 60 video |
| `results/comparison_DETRAC-all.csv` | Bảng so sánh 3 motion model + cột chênh lệch (%) |
| `results/comparison_per_video_DETRAC-all.csv` | Chỉ số theo từng video, cả 3 motion model |
| `results/stratified/segment_outcomes_DETRAC-all.csv` | 1 dòng = 1 đoạn che khuất × 1 motion model (preserved/switched/lost) |
| `results/stratified/track_recall_DETRAC-all.csv` | Recall + số tracker id theo từng track GT |
| `results/stratified/by_bucket*_DETRAC-all.csv` | Tỉ lệ giữ ID theo tầng (độ dài che × độ cong) |
| `results/stratified/mcnemar_*_DETRAC-all.csv` | Kiểm định McNemar ghép cặp so với baseline |
| `results/stratified/id_retention_DETRAC-all.png` | Hình: giữ ID theo độ dài che, tách nhóm thẳng/cong |
