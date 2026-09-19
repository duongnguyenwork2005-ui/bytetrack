# REVIEW_BRIEF — xử lý REQUEST CHANGES vòng 6

<!-- BASE_TARGET -->

Base giữ **`a8ecaf6`**. Chi tiết: **`handoff/BACKEND_NP_NOTES.md`**.
Tái hiện trước khi sửa: `handoff/logs/review6/repro_truoc_khi_sua.log` — **cả 4 finding đều đúng**.

## Finding → bằng chứng trước sửa → thay đổi → kiểm chứng sau sửa

| # | Trước sửa | Thay đổi | Sau sửa |
|:--|:--|:--|:--|
| **F1** evaluator | `startswith("BDD-")` ⇒ `BDDSMOKE-val` chọn **`mot`**, mất crowd-ignore | `dataset_class_for(dataset, split)` chọn theo **loại dataset**; dataset lạ ⇒ **lỗi rõ ràng**; tên thư mục output không còn quyết định chính sách | `test_evaluator_choice` **5/5** với đúng fixture của bạn: `mot` FP=**2** · `mot_crowd` FP=**1** · TP=1 cả hai |
| **F2** resume CLI | CLI ghi metadata mới **trước** vòng lặp rồi `[SKIP]` chỉ vì file tồn tại ⇒ 4 dòng cũ + metadata 0.95 | Tách `src/run_state.py` **dùng chung** cho `stage_track` và CLI: chỉ tái dùng khi có **bằng chứng hoàn tất** + fingerprint khớp | `test_runner_backend_np` **14/14** qua **CLI thật** (chặn torch/ultralytics): sửa cùng YAML → **chạy lại, 4 dòng → 0**; không đổi ⇒ tái dùng; thiếu `video_meta` ⇒ chạy lại |
| **F3** audit backend | `run_meta.json` không có tên backend ⇒ grep trả 0 **kể cả với run NumPy thật** | `tracker_backend` + `code` lưu **tường minh** ở cả hai đường chạy; `backend_of()` chỉ kết luận từ bằng chứng tường minh | Audit: **136/136 run = `UNKNOWN/LEGACY`**. Kết luận cũ "chưa có run NumPy" **đã rút lại** |
| **F4** timeline GT | `seqLength` lấy từ frame annotation cuối ⇒ cắt ngắn video có frame cuối không nhãn | `frame_timeline()` + `seq_lengths_from_images()`: độ dài = **chỉ số frame cuối của timeline ảnh**; CLI trả **mã lỗi 2** khi thiếu ảnh; `--no-images` ghi rõ chỉ kiểm nhãn | `test_bdd100k_coco` **11/11**: frame cuối không nhãn ⇒ 5 (không phải 2); video **không** annotation ⇒ timeline đúng; timeline có khoảng bỏ qua ⇒ 5 (không phải 3 ảnh) |

## Smoke sau khi sửa evaluator

Giữ nguyên cache và tracking — **không** chạy lại detector/tracking. Bản cũ lưu ở
`results/phase2/bdd_smoke_val_PRE_F1/`.

| tracker | HOTA trước | HOTA sau | FP trước → sau |
|:--|--:|--:|:--|
| cv | 0.2409002 | **0.2409274** | 124 → **123** |
| ekf | 0.2497727 | **0.2498010** | 111 → **110** |

Trên ba video này mỗi tracker chỉ có **đúng một** detection rơi vào vùng bỏ qua ⇒ ảnh hưởng số học nhỏ;
cơ chế nay đã đúng. Vẫn là **smoke pipeline**, không phải kết quả thí nghiệm. `eval_meta.json` ghi
`dataset_class: mot_crowd`; backend tracking của smoke là **`UNKNOWN`** (chạy trước khi có
`tracker_backend`) — **không** tạo metadata hồi tố. Hash GT/seqmap/manifest + lệnh tái hiện: phụ lục §2.

## Audit backend

| Nhóm | Run |
|:--|--:|
| NUMPY (xác nhận) / ULTRALYTICS (xác nhận) | 0 / 0 |
| **UNKNOWN/LEGACY (thiếu bằng chứng)** | **136** |
| Mâu thuẫn / nguy cơ tái dùng sai | 0 |

**Không có bằng chứng cụ thể** buộc chạy lại run lịch sử. Giữ nguyên UA-DETRAC A/B/C và GĐ1, không mở
tuning.

## Test thực tế (tự chạy trên bản sửa cuối)

`run_all_tests.py --out-dir handoff/logs/review6` → **26 PASS, 0 FAIL, 0 SKIPPED, 0 NOT_RUN** (exit 0).
Mới/đổi: `test_evaluator_choice` **5/5**, `test_runner_backend_np` **14/14**, `test_bdd100k_coco`
**11/11**.

## Còn bị chặn

- **File nhãn `bdd_box_track_val_cocofmt.json` không còn trên máy** (đã tìm `C:\` và `D:\`) ⇒ **chưa
  kiểm F4 trên video thật `b23b2649-8349d2a1`** (ảnh tới 202, annotation cuối 94). Fixture bao đúng ba
  tình huống, nhưng đó không thay được phép kiểm trên video thật. Gửi lại file là đóng được.
- Nhãn `box_track_20` **native** và split `train` ⇒ chưa tuning trên BDD.
- **Nano chưa kết nối**; bản vendor cần Python ≥ 3.7 còn JetPack 4.6 có 3.6.
- Quy ước hash LF **chỉ** áp cho module vendor — `code_fingerprint()` và hash YAML vẫn là byte thô.
