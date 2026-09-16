## 0. Đợt sửa sau review độc lập (Codex) — đọc trước

Review của Codex trên snapshot `502f011` nêu 4 lỗi (2×P1, 2×P2) + 6 điểm diễn giải. **Cả 4 lỗi đã được tái
hiện trên mã hiện tại và đã sửa**; phần Nhóm B bị ảnh hưởng **đã chạy lại** trong thư mục mới. Bảng trạng thái
đầy đủ: `results/phase2/REVIEW_moc2_codex.md`; chi tiết + số liệu trước/sau: `PHASE2_REPORT.md` **mục 18**.

| # | Phát hiện | Trạng thái | Bằng chứng |
|:--|:--|:--|:--|
| P1 | Baseline "SORT nguyên bản" xuất track sớm hơn upstream 1 frame (`hits`/`hit_streak` khởi tạo 1 thay vì 0) | **Đã sửa + chạy lại Nhóm B** | `src/sort_tracker.py`; `src/test_sort_upstream_parity.py` **8/8** so toàn chuỗi với mã upstream đã ghim (`third_party/sort_upstream`, commit `bce9f0d`), lệch toạ độ tối đa **5,8·10⁻⁵ px**, ánh xạ ID đồng nhất; kết quả mới: `results/phase2/expB_detrac_{dev,test}_fix1/` |
| P1 | Runner có thể chấm tracking mới bằng metric cũ; `--no-skip` không tới bước đánh giá; `run_meta` bị ghi đè provenance; lỗi subprocess không thành exit code | **Đã sửa** | `src/run_fingerprint.py` + `src/run_experiment.py`; `src/test_run_experiment_guard.py` **7/7** |
| P2 | Cache có thể ghi thêm với `conf`/`classes`/`max_det`/phiên bản khác mà metadata giữ nguyên | **Đã sửa** | `src/det_cache.py` (`STRICT_FIELDS`, `spec_sha256` từng sequence, `write_meta(mode=)`); `src/test_det_cache.py` **47/47**; kiểm cache hiện có: `results/phase2/cache_consistency.csv` — **0 mismatch**, không tái tạo |
| P2 | ZIP chứa `.gitignore`; commit trong ZIP lệch tin nhắn (ZIP dựng trước commit) | **Đã sửa** | `src/make_handoff.py`: `USER_FILES`/`USER_PREFIXES`, **từ chối dựng khi còn file khác HEAD**, `handoff/FILE_HASHES.csv`, `handoff/external/bytetrack.yaml` |
| — | 6 điểm diễn giải (Nhóm B, PF, C2, C3, `has_full`, tổng hợp PF theo seed) | **Đã sửa trong báo cáo** | `PHASE2_REPORT.md` mục 7.2, 13.1, 14.2, 15.1, 15.2, 16.2, 16.3, 16.4, 18.6 |

**Kết quả Nhóm B sau khi sửa** (cấu hình y hệt bản đã khóa, chỉ đổi `name`; không tuning):

| cặp (yếu tố được kiểm soát) — TEST 40 video, c25 | trước khi sửa | **sau khi sửa** |
|:--|:--|:--|
| bộ đệm `max_age` 1 → 30 | +0.0280 [+0.0207, +0.0350] | **+0.0274 [+0.0201, +0.0345]** |
| gói biểu diễn + thiết lập bộ lọc (KF 7-D → XYAH) | −0.0133 [−0.0185, −0.0077] | **−0.0132 [−0.0185, −0.0077]** |
| motion model CV → EKF–CTRV | +0.0020 [−0.0001, +0.0042] | **+0.0020 [−0.0001, +0.0042]** |
| chỉ đổi bộ lọc ở `max_age` 1 | −0.0140 [−0.0167, −0.0116] | **−0.0140 [−0.0167, −0.0116]** |
| vòng đời SORT → ByteTrack | +0.0364 [+0.0298, +0.0428] | **+0.0363 [+0.0296, +0.0428]** |

IDSW giảm 9–12 % ở mọi nhánh SORT (track xuất muộn hơn 1 frame ⇒ ít track "non"); **mọi kết luận định tính
không đổi**. `bt-ekf` (ByteTrack) **bit-identical** bản cũ: 12/12 video dev, 40/40 test → bản sửa không đụng
ByteTrack, nên **không** cần chạy lại Nhóm A, Nhóm C hay Giai đoạn 1 (đã kiểm: chỉ Nhóm B dùng `SORTTrack`).
