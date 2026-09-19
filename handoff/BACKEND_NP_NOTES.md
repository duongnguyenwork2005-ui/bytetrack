# Phụ lục — backend NumPy, nhãn BDD COCO-video, và bảng finding vòng 5

Chi tiết cho `handoff/REVIEW_BRIEF.md`. Review nguồn: `handoff/CODEX_REVIEW_ed4fda2.md`
(**REQUEST CHANGES**). Base review giữ **`a8ecaf6`** — Codex chưa chấp thuận chuyển sang `ed4fda2`.

## 1. Bảng finding → sửa ở đâu → kiểm thử tái hiện → kết quả sau sửa

| # | Finding | Sửa ở đâu | Kiểm thử tái hiện | Kết quả sau sửa |
|:--|:--|:--|:--|:--|
| **P1-A** | `resolve_tracker_yaml()` luôn gọi `check_yaml` của Ultralytics; CLI (`run_tracking.py:302`) và `run_experiment.py:210` đều dùng nó để hash YAML ⇒ backend `np` chết | `src/run_tracking.py` — resolver **duy nhất, biết backend**; `load_tracker_cfg` nạp từ **đúng** đường dẫn resolver trả về | `test_runner_backend_np.py` #1, #2 — chạy **CLI thật** và **`stage_track()` thật** trong tiến trình chặn `torch`+`ultralytics` | CLI `rc=0`, `torch=False`, `ultralytics=False`; `stage_track` chạy 1 video / 4 dòng |
| **P1-B** | `_Cfg` kế thừa `dict` ⇒ `vars(cfg)` = `{}`; dùng ở `run_tracking.py:311`, `run_experiment.py:213` và **`:223` (`input_fingerprint`)** ⇒ đổi YAML không làm mất hiệu lực tái dùng | `ul_backend.as_mapping()` + `run_tracking.cfg_to_dict()` thay cả 3 chỗ; **thêm `tracker_yaml_sha256_12` vào `input_fingerprint`** | `test_runner_backend_np.py` #3, #4a-c — đúng kịch bản Codex đo | `tracker_cfg` đầy đủ; đổi YAML tại **cùng đường dẫn** ⇒ chạy lại, **4 dòng → 0 dòng**, fingerprint đổi; không đổi gì ⇒ tái dùng |
| **P2-3** | `assoc_log.py` import thẳng Ultralytics **và** `issubclass` so với lớp nền Ultralytics (backend np kế thừa lớp **vendor** ⇒ luôn `False`) | lấy **cả** `TrackState` và `BYTETracker` từ `ul_backend` | `test_runner_backend_np.py` #6 | `assoc_log=True` chạy trên np, ghi **3 file parquet**, output tracking **không đổi** |
| **P2-4** | Đường dẫn tương đối có thư mục nhưng không tồn tại → bỏ phần thư mục → âm thầm nạp cấu hình **mặc định** | `ul_backend.resolve_tracker_yaml_np()`: **chỉ tên trần** trong `VENDORED_YAML` mới tra bản vendor | (tái hiện trong `handoff/logs/review5/repro_truoc_khi_sua.log`) | đường dẫn có thư mục không tồn tại ⇒ `FileNotFoundError` |
| **P2-5** + **phát hiện thêm** | `core.autocrlf=true`, không có `.gitattributes` ⇒ 2/5 hash vendor lệch byte Git/ZIP. **Nặng hơn**: `fingerprint()` hash byte đĩa ⇒ cùng commit cho hash **khác nhau** giữa Windows (CRLF) và Nano (LF) | **Quy ước hash duy nhất**: `sha256` của nội dung **chuẩn hóa LF** (`ul_backend.HASH_CONVENTION` + `sha256_lf`), áp dụng cho `PROVENANCE.json`, `fingerprint()` và test. Thêm `.gitattributes` (`third_party/** -text`), chuẩn hóa byte vendor về LF | `test_bytetrack_np_parity.py` — kiểm **cả hai** trường hash trong PROVENANCE, và hash **giống nhau** với bản LF lẫn CRLF | 5/5 hash khớp; `sha256_lf` cho cùng kết quả trên CRLF và LF; `git check-attr text third_party/...` → `unset` |
| **P2-6** | 5 câu vượt bằng chứng + lịch đo Nano mâu thuẫn | `handoff/MOC2_FINAL_SUMMARY.md`, `scripts/nano/README.md` | — | xem mục 3 dưới |
| **BDD** | Quy trình smoke **thiếu bước xuất GT**; `split_name: BDD-val` dùng chung ⇒ sẽ ghi đè `seqmaps/BDD-val.txt` | `bdd100k_coco.py --export-gt --benchmark BDDSMOKE` ⇒ `split_name = BDDSMOKE-val` | smoke chạy thật từ thư mục sạch | GT + seqmap riêng; `BDD-val` không bị đụng |

**Lưu ý về `.gitattributes`**: lần đầu tôi đặt `* text=auto` **sau** `third_party/** -text` — git lấy dòng
khớp **cuối cùng** nên quy tắc chung ghi đè quy tắc vendor (`check-attr` trả `text: auto`). Đã đảo thứ tự
và kiểm lại: `third_party/...` → `unset`, `src/...` → `auto`.

## 2. Backend NumPy — lệnh chạy và dependencies

```bash
BYTETRACK_BACKEND=np python src/run_tracking.py --cache-id <id> --videos <v> \
    --split-name <split> --tracker-name <ten> --tracker-yaml bytetrack.yaml --elapsed-mode steps
BYTETRACK_BACKEND=np python src/run_experiment.py --exp <yaml>
```

**Dependencies thực tế**: `numpy`, `scipy`, `lap` (bản PC: 2.4.6 / 1.17.1 / 0.5.13). **Không** cần `torch`,
**không** cần `ultralytics`. Phạm vi: **CV + EKF–CTRV** (chưa mở UKF/PF/SORT).

**Provenance**: `run_fingerprint.code_fingerprint()` mang `backend` (tên backend, phiên bản `ultralytics`
**hoặc** sha256-LF của từng file vendor, phiên bản `numpy`/`scipy`/`lap`) ⇒ đổi backend làm fingerprint đổi
⇒ runner **phải chạy lại**.

⚠️ Việc đưa `tracker_yaml_sha256` vào `input_fingerprint` làm **mọi** thí nghiệm có fingerprint mới ở lần
chạy tới. Đây là **sửa lỗi**, không phải đổi cấu hình nghiên cứu — **kết quả UA-DETRAC lịch sử giữ nguyên**,
không chạy lại A/B/C.

**Audit run NumPy**: `grep '"backend": "np"' data/processed/trackers/*/*/run_meta.json` → **0 kết quả**
⇒ **chưa có run nào** được sinh bằng backend NumPy ⇒ không có run nào cần audit hay chạy lại.

## 3. Các câu đã hạ mức (mục 6 của review)

| Trước | Sau |
|:--|:--|
| "giữ ID qua che khuất tăng có ý nghĩa — **kể cả ở tầng che hoàn toàn**" | có ý nghĩa ở `preserved` **chung**; ở tầng `full` báo cáo **11/182 → 22/182 là thay đổi quan sát được** — bảng hiện có **chưa** cung cấp CI/phép kiểm riêng cho tầng đó |
| "chênh lệch **không do seed**" | "**lớn hơn biến thiên giữa ba seed đã thử**"; không loại trừ mọi ảnh hưởng của seed |
| số tốc độ "chỉ dùng để **so tương đối giữa các bộ lọc**" | **số đo mô tả của chính các lượt đã chạy**, không đủ để xếp hạng; và đo từ cache chỉ loại chi phí **detector**, `update_ms` **vẫn gồm** association + vòng đời |
| "`detector_wall` **dao động** 2,2 ms ⇒ nguyên nhân đảo thứ tự" | **khác nhau giữa bốn lượt** (mỗi cấu hình **một** lượt) ⇒ là **giả thuyết cần kiểm bằng đo lặp**, chưa kiểm |
| "nhiễu gấp ~9 lần", "trên Nano **nặng hơn**" | bỏ; thành **rủi ro cần kiểm** — chưa có phép đo nào trên thiết bị |
| lịch đo Nano: mục 2 ghi A/B/A/B, mục 4 ghi AB/BA | **một** lịch: 3 cặp A→B + 3 cặp B→A; giữ **mọi** lượt trong dữ liệu thô; loại **theo cặp** (bỏ một nửa làm lệch hiệu số ghép cặp) |

## 4. Nhãn BDD100K — COCO-video, không phải native

Nguồn trên máy: `bdd_box_track_val_cocofmt.json` (117 MB) — 200 video / 39.973 ảnh / 465.512 annotation,
khớp chính xác số ảnh đã tải.

| | |
|:--|:--|
| **Có** | `instance_id` (track ID), `category_id` (8 lớp đánh giá), `bbox` (xywh), `iscrowd` (22.690), `ignore` (4.065), `video/frame/file_name` |
| **Không có** | `attributes.occluded`, `attributes.truncated`, `timestamp` |
| **Hệ quả** | `occlusion_ratio`/`visibility` = **NaN** (không bịa), `dt_source = "nominal"` (5 fps), **không chạy** phân tầng che khuất |

**Điểm dễ sai nhất — ID theo video**: đo trên nhãn thật, **327/392 `instance_id` xuất hiện ở nhiều video**.
Lấy thẳng làm `track_id` sẽ **gộp track của các video khác nhau**. Adapter đánh số lại theo
`(video, instance_id)`; `test_bdd100k_coco.py` phép kiểm 3 chứng minh điều đó.

`visibility = 1.0` trong `gt.txt` là **PLACEHOLDER** do định dạng MOTChallenge bắt buộc cột `vis` là một số
(để NaN thì TrackEval báo *"Cannot convert gt data to float"*). Nó có nghĩa **"không biết"**, **không** phải
"đo được là nhìn thấy 100 %".

## 5. Smoke BDD — đã chạy thật, là smoke pipeline

Manifest ghi **trước** khi xem bất kỳ metric nào: `results/phase2/manifests/bdd100k_smoke_val.json`
(tiêu chí cố định: 3 video đầu theo thứ tự bảng chữ cái).

```
nhãn → GT (BDDSMOKE-val) → cache (GPU) → tracking → evaluator → tổng hợp
```

- cache `yolov8n-coco_conf0.05_smoke_bdd`: 3 video × **202 frame**, `weights_sha256` = `f59b3d83…` (đúng
  checkpoint đã nạp), đọc strict OK.
- số frame: **ảnh = cache = `seqLength` = 202** cho cả 3 video.
- kết quả (**smoke, KHÔNG phải kết quả thí nghiệm, KHÔNG dùng để chọn tham số**):
  `cv` HOTA 0.2409 / IDF1 0.2893 / IDSW 106 · `ekf` HOTA 0.2498 / IDF1 0.3123 / IDSW 71.

**Giới hạn**: smoke này **không** kiểm chứng đường nhãn **native** (`box_track_20`) — nguồn thiếu
`occluded`/`timestamp`. Con số HOTA thấp là điều dự kiến với detector COCO chưa huấn luyện lại trên cảnh
lái xe BDD ở 5 fps; **không** tuning theo nó.

## 6. Còn thiếu / chưa xác minh

- **Nhãn `box_track_20` native** (MOT 2020 Labels, ~50 MB): cần để có `occluded`/`timestamp` và để chạy
  phân tầng che khuất. Split **`train`** cũng chưa có ⇒ **chưa được tuning trên BDD**.
- **Jetson Nano chưa kết nối**. Parity trên PC **không** chứng minh tương thích hay tốc độ Nano. Chưa kiểm:
  bản vendor cần **Python ≥ 3.7** (JetPack 4.6 có 3.6); `scipy`/`lap` cho aarch64; đường detector trên Nano
  chưa nối vào backend mới.
- 53/56 sequence cache `legacy` và 113 thư mục run `LEGACY` vẫn không xác minh được bằng hash.
