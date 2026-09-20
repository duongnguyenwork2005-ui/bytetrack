# REVIEW_BRIEF — vòng 10: sửa PC-F1…F4 của review `8abb368`

<!-- BASE_TARGET -->

Base delta **giữ `b623465`**; review `8abb368` còn REQUEST CHANGES, **chưa** đánh dấu PASS. Nano F1/F3
tiếp tục hoãn. Không tuning/ablation/train; metric, GT, lock UA-DETRAC/BDD **không đổi** (đã kiểm `git diff`).
Chi tiết: **`handoff/REVIEW10_CHI_TIET.md`**; hướng dẫn + thời gian mới: **`handoff/HE_THONG_PC.md`**.

## Tái hiện → sửa → kiểm chứng

Cả 4 tái hiện bằng fixture (detector/`imread` giả lập, tracker thật, `main()` thật) đúng như Codex mô tả
(`handoff/logs/review10/repro_truoc_khi_sua.log`); sau sửa không còn tái hiện được (`repro_sau_khi_sua.log`).

| # | Trước | Sửa | Sau |
|:--|:--|:--|:--|
| **F1** | bộ đọc 20 ms → `decode_ms` 0,003 | timer quanh đúng `imread`/`read`; tách `mask`; `frame_total` từ trước đọc đến sau tracker + vẽ/ghi; `overhead` ghi riêng; render/write/show theo frame; cùng tập frame sau warm-up; wall tách khởi tạo/vòng lặp/ghi/ngoài-frame | `decode_ms` **20,46**; Σstage+overhead == `frame_total` (3,6·10⁻¹⁵); dữ liệu thật: Σ`frame_total` = `wall_vong_lap` − 12…26 ms/1470 frame |
| **F2** | live nhận cấu hình mà replay từ chối | `check_compatible()` **trước** khi chạy → exit 2; chạm trần → exit 3 không ghi (y hệt `VideoDetections`); `--allow-truncated` có ghi meta + lệnh replay | (a) 0/0 parity ĐẠT **bằng đúng lệnh trong `run_meta`**; (b) exit 2; (c) exit 3, với flag → 0/0 ĐẠT; (d) exit 2 |
| **F3** | quy nguyên nhân cho detector; đảo dòng đổi kết luận | bỏ quy nguyên nhân; song ánh từ cặp **không mơ hồ** + ràng buộc toàn sequence; không giải được → **`mo_ho`** (không PASS); không Hungarian | `test_nano_parity` **20/20**: đảo dòng → y hệt; fixture Codex → `mo_ho` cả hai; chỉ đổi conf → *chưa xác định*; đổi liên kết thật vẫn FAIL |
| **F4** | `1,2,10` → `1,10,2` | sắp theo **số** cuối tên (cùng tiền tố); mơ hồ → lỗi có hướng dẫn `--frames-manifest`; ghi `frames_manifest.json` | `test_track_video_file` **23/23** (6 ca F4) |

## Thời gian đo lại (một lượt, `MVI_39031`, sau 25 frame warm-up, cùng tập 1445 frame)

| ms/frame | decode | detect | tracker | vẽ+ghi | overhead | **frame_total** | fps stage / vòng lặp |
|:--|--:|--:|--:|--:|--:|--:|--:|
| CV | 3,74 | 17,50 | 1,03 | — | 0,36 | **23,10** | 43,3 / 40,3 |
| CV `--save-video` | 3,56 | 17,80 | 1,02 | 5,73 | 0,60 | **29,19** | 34,3 / 32,2 |
| EKF | 3,63 | 18,69 | 1,28 | — | 0,42 | **24,54** | 40,7 / 38,4 |
| EKF `--save-video` | 3,90 | 20,50 | 1,38 | 6,35 | 0,63 | **33,33** | 30,0 / 28,6 |

`fps stage` = 1000/`frame_total`; `fps vòng lặp` = 1470/`wall_vong_lap` (kể cả warm-up). Khởi tạo/ghi tách
riêng. Nguồn `results/track_video/v2_MVI_39031_*/pipeline/*.csv`; bản cũ giữ, không dùng.

## Tài liệu / provenance

Ngưỡng **hiệu lực** là `--conf-min` (0,05 không tự bảo đảm vòng ghép thấp; c10 cần 0,1); ví dụ replay khớp
classes; chế độ dataset chỉ mask `ignored_regions`, không đưa GT vào association; `run_meta.entry_point` =
sha256 `track_video_file.py` + trạng thái `src/` (không mở rộng `CODE_FILES`, không hồi tố). Log lỗi 19:39:
stdout thật lưu `handoff/logs/review9/lich_su_that_bai/` + README tách *đã xác minh* / *giả thuyết* (3 commit
19:41–19:42 trong cửa sổ chạy); log chi tiết đã mất, không tái tạo.

## Kiểm thử

`test_nano_parity` 20/20, `test_track_video_file` 23/23 (tầng fixture + YOLO thật/GPU: live→cache→replay
**bit-identical**), `test_run_tracking_replay` 10/10, `test_bytetrack_np_parity` 10/10. Full suite trên code
chốt: `handoff/logs/review10/` — không commit/sửa source trong lúc chạy.
