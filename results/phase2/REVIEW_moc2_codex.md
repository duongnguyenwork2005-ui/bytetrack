# Xử lý review độc lập (Codex) của gói bàn giao mốc 2

Nguồn: review trên `handoff/phase2_review_latest.zip` (snapshot commit `502f011`). File này ghi **trạng thái
từng phát hiện**, cách **tái hiện trên mã hiện tại**, bản sửa và bằng chứng. Kết quả/log/LOCK của mốc 2 **giữ
nguyên**; mọi lần chạy sau khi sửa nằm trong thí nghiệm có hậu tố `_fix1` (thư mục và tên tracker mới).

## 0. Snapshot được review vs commit bàn giao

| Mục | Giá trị |
|:--|:--|
| Snapshot Codex đọc (trong `versions.json`, `CODEX_HANDOFF.md` của ZIP) | **`502f011`** |
| Commit ghi trong tin nhắn bàn giao | **`28ca79b`** |
| Nguyên nhân | `src/make_handoff.py` được chạy **trước** commit cuối: ZIP dựng lúc 10:45 tại HEAD `502f011`, sau đó commit `28ca79b` mới chứa chính các artifact của ZIP (`handoff/INDEX.md`, `versions.json`, `patches/`) và 2 sửa tham chiếu treo trong `CODEX_HANDOFF.md`. |
| Nội dung khác nhau giữa hai commit | `git diff --stat 502f011..28ca79b` = **7 file, toàn bộ thuộc `handoff/` + `CODEX_HANDOFF.md`** — **không có** thay đổi nào trong `src/`, `configs/`, `results/`. Vì vậy mọi nhận xét của Codex về mã và kết quả **áp dụng nguyên vẹn** cho `28ca79b`. |
| Bản sửa | `make_handoff.py` nay **từ chối đóng gói** khi file sắp đóng gói còn khác HEAD, và ghi `git_head` + danh sách dirty vào `versions.json`/`INDEX.md`; quy trình: commit → dựng ZIP. |

## 1. Bảng phát hiện → trạng thái

| # | Phát hiện | Mức | Tái hiện trên mã hiện tại | Trạng thái | Bằng chứng |
|:--|:--|:--|:--|:--|:--|
| 1 | Baseline "SORT nguyên bản" xuất track sớm hơn upstream 1 frame (`hits=1`, `hit_streak=1` thay vì 0) | P1 | `src/sort_tracker.py` `SORTTrack.__init__`; `test_2_min_hits` khẳng định xuất ở frame 12 | **ĐÃ SỬA** + chạy lại Nhóm B | mục 2 dưới; `src/test_sort_upstream_parity.py`; `results/phase2/expB_detrac_{dev,test}_fix1/` |
| 2 | Runner tái sử dụng metric cũ cho tracking mới; `--no-skip` không tới bước đánh giá; `run_meta` bị ghi đè provenance; lỗi subprocess không thành exit code | P1 | `run_experiment.stage_eval/stage_track/main` | **ĐÃ SỬA** | mục 3; `src/test_run_experiment_guard.py` |
| 3 | Cache detection có thể ghi thêm với `conf/classes/max_det/versions` khác mà metadata giữ nguyên | P2 | `DetectorSpec.STRICT_FIELDS` thiếu 4 trường; `<video>.json` không có fingerprint | **ĐÃ SỬA** (chính sách + fingerprint per-sequence) | mục 4; `src/test_det_cache.py` |
| 4 | ZIP chứa `.gitignore` (file người dùng) trái mô tả; commit trong ZIP lệch tin nhắn | P2 | `src/make_handoff.py` | **ĐÃ SỬA** | mục 0 và 5 |
| 5 | Diễn giải: Nhóm B "chỉ đổi biểu diễn trạng thái"; PF "mất đa dạng trước quan sát"; Nhóm C thiếu cặp `ekf-maha − ekf`; C3 "mọi CI chứa 0"; `has_full`; tổng hợp PF theo seed | — | đọc `PHASE2_REPORT.md` | **ĐÃ SỬA** trong báo cáo | `PHASE2_REPORT.md` mục 7.2, 13.1, 14.2, 15.1, 15.2, 16.2, 16.3, 16.4, 18 |

## 2. Phạm vi bị ảnh hưởng bởi lỗi SORT (đã kiểm, không suy đoán)

`SORTTrack` chỉ được dùng bởi `src/sort_tracker.py` → `SORT_TRACKERS` = `sort_orig`, `sort_cv`, `sort_ekf_ctrv`.
Đối chiếu `tracker_type` của mọi thí nghiệm:

| Thí nghiệm | tracker_type dùng | Bị ảnh hưởng? |
|:--|:--|:--|
| `groupB_detrac_dev`, `expB_detrac_test` | `sort_orig`, `sort_cv`, `sort_ekf_ctrv` (5 nhánh) + `bytetrack_ekf_ctrv` (bt-ekf) | **Có** (5 nhánh SORT). `bt-ekf` **không**. |
| `expA_detrac_dev`, `expA_detrac_test`, `expC*`, `expC1/2/3`, `groupA_*` | chỉ `bytetrack*` | Không |
| Giai đoạn 1 (`baseline_track.py`, runA/runB/testA/testB, buffer_sweep, lost_assoc) | `bytetrack*` qua `model.track` | Không |

→ Chỉ chạy lại **các nhánh SORT của Nhóm B** (dev + test). `bt-ekf` được chạy lại trong thí nghiệm `_fix1`
và **đối chiếu bit-identical** với bản cũ để chứng minh không đổi.

## 3. Kết quả kiểm tra cache hiện có

`src/check_cache_consistency.py` đối chiếu `meta.json` với từng `<video>.json` của mọi cache trong
`data/interim/det_cache/`. Kết quả (ghi trong `results/phase2/cache_consistency.csv`): xem file đó — cache cũ
được sinh trước khi có fingerprint per-sequence nên `spec_sha256` trống; kiểm tra gián tiếp bằng `created`,
`part`, `regions_hash` và `meta.json` cho thấy **một spec duy nhất** cho mỗi cache-id → **không tái tạo cache**.
Từ nay mọi sequence mới đều mang `spec_sha256` và bị kiểm khi đọc.
