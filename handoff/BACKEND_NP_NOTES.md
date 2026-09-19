# Phụ lục — backend NumPy, nhãn BDD COCO-video, audit provenance

Chi tiết cho `handoff/REVIEW_BRIEF.md`. Review nguồn gần nhất: `handoff/CODEX_REVIEW_d42f223.md`
(**REQUEST CHANGES**, 4 finding). Base review giữ **`a8ecaf6`**.

## 1. Vòng 6 — finding → bằng chứng trước sửa → thay đổi → kiểm chứng sau sửa

Tái hiện đầy đủ: `handoff/logs/review6/repro_truoc_khi_sua.log`.

| # | Bằng chứng **trước** sửa | Thay đổi | Kiểm chứng **sau** sửa |
|:--|:--|:--|:--|
| **F1** evaluator | `dataset_cls = "mot_crowd" if split.startswith("BDD-")` ⇒ `BDDSMOKE-val` → **`mot`**, mất crowd-ignore | `run_trackeval.dataset_class_for(dataset, split)` chọn theo **loại dataset** (`DATASET_EVALUATOR`); dataset lạ ⇒ **lỗi rõ ràng**; tên thư mục output không còn quyết định chính sách | `test_evaluator_choice` **5/5**, dùng đúng fixture Codex: `mot` FP=**2** (Dets 3) · `mot_crowd` FP=**1** (Dets 2) · TP=1 cả hai |
| **F2** resume CLI | CLI ghi `run_meta.json` mới **trước** vòng lặp rồi `[SKIP]` chỉ vì file tồn tại ⇒ 4 dòng cũ + metadata 0.95 | Tách `src/run_state.py` **dùng chung** cho `stage_track` và CLI: chỉ tái dùng khi có **bằng chứng hoàn tất** (`video_meta`) **và** fingerprint khớp | `test_runner_backend_np` **14/14** qua **CLI thật**, tiến trình chặn torch/ultralytics: 0.25→4 dòng; sửa cùng YAML→0.95 + `--skip-existing` ⇒ **chạy lại, 0 dòng**; không đổi ⇒ tái dùng; thiếu `video_meta` ⇒ chạy lại |
| **F3** audit backend | `run_meta.json` **không** có tên backend (chỉ nằm trong đối tượng đã hash) ⇒ `grep '"backend": "np"'` trả 0 **kể cả với run NumPy thật** | `tracker_backend` (chuỗi) + snapshot `code` lưu **tường minh** ở cả hai đường chạy; `audit_run_provenance.backend_of()` chỉ kết luận từ bằng chứng tường minh | Audit thực tế: **136/136 run = `UNKNOWN/LEGACY`**. Kết luận cũ "chưa có run NumPy" **đã rút lại** |
| **F4** timeline GT | `seq_len = df[df.video==v].frame.max()` (chỉ bảng annotation) ⇒ video có frame cuối không nhãn bị cắt ngắn | `frame_timeline()` + `seq_lengths_from_images()`: độ dài = **chỉ số frame cuối của timeline ảnh**; CLI cảnh báo khi annotation cuối < frame ảnh cuối và khi có frame bị bỏ qua | `test_bdd100k_coco` **11/11**: frame cuối không nhãn ⇒ `seqLength`=5 (không phải 2); video **không** annotation ⇒ timeline đúng, schema đủ cột; timeline có khoảng bỏ qua ⇒ độ dài 5 (không phải 3 ảnh) |

**F4 — phần chưa kiểm được**: video thật `b23b2649-8349d2a1` (ảnh tới frame 202, annotation cuối
frame 94). File nhãn `bdd_box_track_val_cocofmt.json` **không còn trên máy** (đã tìm cả `C:\` và `D:\`).
Phép kiểm trên fixture bao đúng ba tình huống, nhưng **chưa** xác nhận trên video thật đó.

## 2. Kết quả smoke sau khi sửa evaluator

Giữ nguyên cache và tracking cũ — **không** chạy lại detector hay tracking. Bản trước khi sửa lưu tại
`results/phase2/bdd_smoke_val_PRE_F1/` (lịch sử bị thay thế).

| tracker | HOTA trước (`mot`) | HOTA sau (`mot_crowd`) | FP trước | FP sau | IDSW |
|:--|--:|--:|--:|--:|--:|
| cv | 0.2409002 | **0.2409274** | 124 | **123** | 106 |
| ekf | 0.2497727 | **0.2498010** | 111 | **110** | 71 |

Trên **ba video này**, mỗi tracker chỉ có **đúng một** detection rơi vào vùng bỏ qua, nên ảnh hưởng số
học nhỏ. Cơ chế thì nay đã đúng — fixture cho thấy nó có thể đổi FP 2→1. Đây vẫn là **smoke pipeline**,
**không** phải kết quả thí nghiệm và **không** dùng để chọn tham số.

### Metadata smoke (để tái hiện)

| | |
|:--|:--|
| evaluator | `eval_meta.json` → `{"trackeval": "1.3.0", "dataset_class": "mot_crowd"}` |
| backend tracker | **`UNKNOWN`** — tracking chạy **trước** khi có `tracker_backend`; không tạo metadata hồi tố |
| cache | `yolov8n-coco_conf0.05_smoke_bdd`, `weights_sha256` `f59b3d83…`, 3 video × 202 frame |
| manifest | `results/phase2/manifests/bdd100k_smoke_val.json` sha256 `799b1a082768bb46` |
| seqmap | `data/processed/seqmaps/BDDSMOKE-val.txt` sha256 `a362e97b4a525c60` |
| GT | `b1c66a42-6f7d68ca` `716d6fda7f77a6c9` · `b1c81faa-3df17267` `12551f19e4c7b452` · `b1c81faa-c80764c5` `a3fba28a2561aee4` |
| nhãn nguồn | `bdd_box_track_val_cocofmt.json` — **không còn trên máy**; Codex ghi sha256 `074ff795…` |

```bash
# tai hien (can file nhan nguon)
python -m src.mot_datasets.bdd100k_coco --labels <.../bdd_box_track_val_cocofmt.json> \
    --videos b1c66a42-6f7d68ca b1c81faa-3df17267 b1c81faa-c80764c5 --export-gt --benchmark BDDSMOKE
python src/det_cache.py --dataset bdd100k_coco --labels-json <...> --part val \
    --manifest results/phase2/manifests/bdd100k_smoke_val.json --cache-id yolov8n-coco_conf0.05_smoke_bdd
python src/run_experiment.py --exp configs/phase2/experiments/bdd_smoke_val.yaml
```

## 3. Audit backend — kết quả và giới hạn

`python src/audit_run_provenance.py --pattern "p2-*" --splits DETRAC-all DETRAC-test BDDFX-val BDDSMOKE-val`

| Nhóm | Số run |
|:--|--:|
| `NUMPY (xác nhận)` | 0 |
| `ULTRALYTICS (xác nhận)` | 0 |
| **`UNKNOWN/LEGACY (thiếu bằng chứng)`** | **136** |
| Có bằng chứng mâu thuẫn / nguy cơ tái dùng sai | 0 |

**Kết luận cũ đã rút lại**: câu "grep không thấy `np` ⇒ chưa có run NumPy" trong bản vòng 5 **không có
cơ sở** — backend khi đó chỉ nằm trong đối tượng đã hash. Toàn bộ 136 run hiện có là **UNKNOWN**: không
suy ra chúng dùng Ultralytics, và cũng không kết luận được là chưa có run NumPy.

**Không tạo provenance hồi tố.** Từ vòng này, run **mới** ghi `tracker_backend` tường minh.
`cache_meta.backend` là backend **detector** của cache — **khác** backend **tracker**.

**Chưa có bằng chứng cụ thể nào** buộc chạy lại run lịch sử: 0 run ở nhóm mâu thuẫn/nguy cơ, và các kết
quả `_fix1` vẫn được xác minh bằng sha256 đối chiếu `eval_meta.json` (vòng 2). Giữ nguyên UA-DETRAC
A/B/C và GĐ1.

## 4. Backend NumPy — lệnh chạy và dependencies

```bash
BYTETRACK_BACKEND=np python src/run_tracking.py --cache-id <id> --videos <v> \
    --split-name <split> --tracker-name <ten> --tracker-yaml bytetrack.yaml --elapsed-mode steps
BYTETRACK_BACKEND=np python src/run_experiment.py --exp <yaml>
```

`numpy`, `scipy`, `lap` (bản PC: 2.4.6 / 1.17.1 / 0.5.13). Không cần `torch`, không cần `ultralytics`.
Phạm vi: **CV + EKF–CTRV**. Quy ước hash mã vendor: **sha256 nội dung chuẩn hóa LF**
(`ul_backend.HASH_CONVENTION`).

⚠️ Quy ước LF hiện **chỉ** áp cho module **vendor**. `run_fingerprint.code_fingerprint()` vẫn hash byte
thô của module `src/`, và hash YAML cũng là byte thô — **không** mở rộng thành "toàn bộ fingerprint
giống nhau giữa Windows và Linux".

## 5. Nhãn BDD100K — COCO-video, không phải native

| | |
|:--|:--|
| **Có** | `instance_id`, `category_id` (8 lớp), `bbox` (xywh), `iscrowd`, `ignore`, `video/frame/file_name` |
| **Không có** | `attributes.occluded`, `attributes.truncated`, `timestamp` |
| **Hệ quả** | `occlusion_ratio`/`visibility` = **NaN**, `dt_source = "nominal"`, **không** chạy phân tầng che khuất |

**ID theo video**: 327/392 `instance_id` xuất hiện ở nhiều video ⇒ `track_id` đánh số theo
`(video, instance_id)`. `visibility = 1.0` trong `gt.txt` là **placeholder** do định dạng bắt buộc —
nghĩa là "không biết".

## 6. Còn thiếu / chưa xác minh

- **File nhãn COCO-video** không còn trên máy ⇒ chưa kiểm F4 trên video thật `b23b2649-8349d2a1`,
  và chưa tái hiện lại được smoke từ đầu.
- **Nhãn `box_track_20` native** và split `train` ⇒ chưa được tuning trên BDD.
- **Jetson Nano chưa kết nối**. Bản vendor cần **Python ≥ 3.7** (JetPack 4.6 có 3.6); `scipy`/`lap`
  aarch64 và đường detector trên Nano chưa thử.
- 136/136 run hiện có **không xác định được backend** (xem mục 3).
