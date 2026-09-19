# REVIEW_BRIEF — xử lý REQUEST CHANGES vòng 5

<!-- BASE_TARGET -->

Base giữ **`a8ecaf6`** (chưa chuyển sang `ed4fda2`). Chi tiết: **`handoff/BACKEND_NP_NOTES.md`**.
Tái hiện trước khi sửa: `handoff/logs/review5/repro_truoc_khi_sua.log` — **mọi finding đều đúng**.

## Finding → sửa → kiểm thử → kết quả

| # | Sửa ở đâu | Kết quả sau sửa (test tái hiện) |
|:--|:--|:--|
| **P1-A** resolver kéo Ultralytics | `resolve_tracker_yaml()` thành resolver **duy nhất, biết backend**; `load_tracker_cfg` nạp từ **đúng** file resolver trả về | `test_runner_backend_np` #1–2, qua **CLI thật** + **`stage_track()` thật** trong tiến trình **chặn** torch+ultralytics: CLI `rc=0`, `torch=False`, `ultralytics=False` |
| **P1-B** cấu hình NumPy → `{}` | `as_mapping()`/`cfg_to_dict()` thay **cả 3** chỗ `vars(cfg)`; **thêm `tracker_yaml_sha256` vào `input_fingerprint`** | #3–4 (kịch bản Codex đo): `tracker_cfg` đầy đủ; sửa YAML **cùng đường dẫn** ⇒ chạy lại, **4 dòng → 0 dòng**; không đổi ⇒ tái dùng |
| **P2-3** `assoc_log` | lấy **cả** `TrackState` và `BYTETracker` từ `ul_backend`, kể cả trong `issubclass` | #6: chạy trên np, ghi **3 parquet**, output tracking **không đổi** |
| **P2-4** fallback YAML | chỉ **tên trần** trong `VENDORED_YAML` mới tra bản vendor | đường dẫn có thư mục không tồn tại ⇒ `FileNotFoundError` |
| **P2-5** + **phát hiện thêm** | **Một** quy ước: `sha256` nội dung **chuẩn hóa LF** cho PROVENANCE + `fingerprint()` + test; thêm `.gitattributes` | `test_bytetrack_np_parity` kiểm **cả hai** trường hash: 5/5 khớp. *Thêm*: `fingerprint()` cũ hash byte đĩa ⇒ cùng commit cho hash **khác nhau** Windows/Nano — nay ổn định |
| **BDD** thiếu bước GT, `split_name` dùng chung | `bdd100k_coco.py --export-gt --benchmark BDDSMOKE` | GT + seqmap **riêng**; `BDD-val` không bị đụng |

## Nhãn BDD và smoke

Nguồn là **COCO-video**, không phải `box_track_20` native ⇒ adapter **riêng**
(`src/mot_datasets/bdd100k_coco.py`). Thiếu `occluded`/`truncated`/`timestamp` ⇒ ghi **NaN + MISSING**,
`dt_source = "nominal"`, **không** chạy phân tầng che khuất. Đo trên nhãn thật: **327/392 `instance_id`
xuất hiện ở nhiều video** ⇒ `track_id` đánh số theo **(video, instance_id)**. `visibility = 1.0` trong
`gt.txt` là **placeholder** do định dạng bắt buộc — nghĩa là "không biết".

Smoke chạy thật từ thư mục sạch, manifest ghi **trước** khi xem metric, 3 video × 202 frame,
ảnh = cache = `seqLength`: `cv` HOTA 0.2409 · `ekf` HOTA 0.2498. Đây là **smoke pipeline**, **không** phải
kết quả thí nghiệm, **không** dùng chọn tham số, và **không** kiểm chứng đường nhãn native.

## Diễn giải đã hạ mức

Tầng `full`: **11/182 → 22/182 là thay đổi quan sát được** (chưa có CI riêng cho tầng). Seed: "lớn hơn biến
thiên giữa **ba seed đã thử**". Tốc độ: **số đo mô tả**, không xếp hạng; đo từ cache chỉ loại chi phí
**detector** — `update_ms` **vẫn gồm** association + vòng đời. Bỏ "nhiễu gấp ~9 lần" và "trên Nano nặng
hơn" → **rủi ro cần kiểm**. Nano: **một** lịch cân bằng **AB/BA** (3 cặp mỗi chiều), giữ **mọi** lượt,
loại **theo cặp**.

## Test thực tế

`run_all_tests.py --out-dir handoff/logs/review5` → **25 PASS, 0 FAIL, 0 SKIPPED, 0 NOT_RUN** (exit 0).
Gồm `test_runner_backend_np` **9/9** (mới), `test_bdd100k_coco` **8/8** (mới), `test_bytetrack_np_parity`
**10/10** (cache **thật**, bit-identical).

`grep '"backend": "np"'` trên mọi `run_meta.json` → **0 kết quả** ⇒ chưa có run nào sinh bằng backend
NumPy ⇒ **không có gì cần audit hay chạy lại**. Việc thêm `tracker_yaml_sha256` vào fingerprint làm mọi thí
nghiệm có fingerprint mới ở lần chạy tới — đó là **sửa lỗi**, không chạy lại A/B/C.

## Chưa xác minh

Nhãn **native** `box_track_20` và split `train` (⇒ chưa tuning trên BDD). **Nano chưa kết nối** — parity
trên PC **không** chứng minh tương thích/tốc độ Nano; bản vendor cần **Python ≥ 3.7** còn JetPack 4.6 có
**3.6**; `scipy`/`lap` aarch64 và đường detector trên Nano chưa thử.
