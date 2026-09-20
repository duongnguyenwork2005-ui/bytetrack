# Phụ lục vòng 10 — xử lý review `8abb368` (PC-F1…F4)

Tóm tắt: `handoff/REVIEW_BRIEF.md`. Hướng dẫn + bảng thời gian mới: `handoff/HE_THONG_PC.md`.

## 0. Trạng thái trước khi sửa

`handoff/logs/review10/trang_thai_truoc_khi_sua.log`: branch `phase2-mixed-traffic`, HEAD **đúng
`8abb368`** (0 commit sau). Working tree chỉ có 2 file riêng (`.gitignore`, `src/demo/rerun_with_state_log.py`)
— không đụng. Python 3.11.9, numpy 2.4.6, cv2 5.0.0, ultralytics 8.4.141, torch 2.6.0+cu124.

## 1. Tái hiện trước sửa → sửa → kiểm chứng sau sửa

Fixture: detector và `cv2.imread` giả lập, **tracker thật**, gọi `main()` thật của cả hai entry point
(`handoff/logs/review10/repro_truoc_khi_sua.log`, exit 0 = tái hiện được cả 4; `repro_sau_khi_sua.log`,
exit 1 = không còn tái hiện được).

| # | Tái hiện trước sửa | Thay đổi | Sau sửa |
|:--|:--|:--|:--|
| **PC-F1** timing | bộ đọc 20 ms/frame → `decode_ms` **0,0028**; "pipeline" 3,96 ms vs wall 27,7 ms/frame | `NguonFrame.__iter__` đo quanh đúng `imread`/`cap.read` và yield `decode_ms`; adapter đặt `t_bat_dau` **trước** `next(it)`; tách `mask_ms`; `frame_total_ms` đo tới sau `tracker.update` + `on_frame`; `overhead_ms` = phần dư, ghi riêng; render/write/show theo frame vào CSV qua dict `ghi_ve`; tổng kết trên **cùng tập frame** sau warm-up cho mọi cột; `wall_khoi_tao` / `wall_vong_lap` / `wall_ghi` / `wall_ngoai_frame` tách riêng; `make_tracker` làm nóng trước vòng lặp (lần đầu 403 ms là import) | `decode_ms` = **20,46**; Σstage + overhead == `frame_total` (sai số 3,6·10⁻¹⁵); trên dữ liệu thật Σ`frame_total` = `wall_vong_lap` − 12…26 ms / 1470 frame |
| **PC-F2** replay | `--det-conf 0.5 --conf-min 0.1`: live exit 0, replay ValueError; `--det-max-det 1`: live exit 0 với 3 frame `truncated`, replay ValueError; consumer `max_det` 10 > producer 5: live exit 0 | `det_cache.check_compatible()` chạy **trước** khi xử lý → exit 2; frame chạm trần → `TruncatedFrame`, exit 3, **không ghi kết quả** (y hệt `VideoDetections`); `--allow-truncated` chấp nhận, ghi `loc.allow_truncated`, `replay_trang_thai`, và **lệnh replay có `--allow-truncated`**; lệnh replay dùng đúng classes/ngưỡng/seed/`--set` hiệu lực | (a) exit 0/0 parity ĐẠT bằng **đúng lệnh trong `run_meta`**; (b) exit **2**; (c) exit **3**; (c′) `--allow-truncated` → 0/0, parity ĐẠT; (d) exit **2** |
| **PC-F3** parity | chỉ đổi conf trong file kết quả → công cụ in "khác biệt ở detector"; hai track bbox/conf giống hệt: đảo 2 dòng frame 2 → `phan_hoach_khop` True→False | Bỏ mọi quy nguyên nhân; ứng viên = **hình học** (Chebyshev ≤ 1e-3), conf là đại lượng riêng; song ánh từ cặp **không mơ hồ** + lan truyền **ràng buộc toàn sequence**; box không giải được → trạng thái **`mo_ho`** (không PASS, không FAIL); không dùng Hungarian; thông báo ghi *CHƯA XÁC ĐỊNH* khi chỉ bbox/conf lệch | `test_nano_parity` **20/20**: đảo thứ tự dòng (cả ca đổi liên kết thật) → kết luận y hệt; fixture Codex → `mo_ho` ở cả hai biến thể; chỉ đổi conf → không nhắc detector; neo ở frame khác giải được mơ hồ; ca đổi liên kết thật vẫn FAIL |
| **PC-F4** thứ tự ảnh | `1.jpg 2.jpg 10.jpg` → đọc `1, 10, 2` | `sap_xep_anh()`: cùng tiền tố → sắp theo **giá trị số** cuối tên; tiền tố khác / không số / trùng số → `ValueError` có hướng dẫn `--frames-manifest`; manifest JSON tường minh; run ghi `frames_manifest.json` (quy ước + file/frame/time_s) | `1,2,10` và `001,002,010` và `img_7,img_08,img_10` đúng; 3 ca mơ hồ → lỗi; manifest được tôn trọng |

## 2. Tài liệu / provenance (mục 6 của review)

| Điểm | Xử lý |
|:--|:--|
| "0,05 đủ bảo đảm vòng ghép thấp" | Sửa README + HE_THONG_PC + docstring: ngưỡng **hiệu lực** là `--conf-min`; với 0,25 mọi detection < 0,25 bị lọc trước tracker; c10 cần `--conf-min 0.1`; không đổi cấu hình khóa |
| Ví dụ replay lệch classes | Ví dụ live nay ghi tường minh `--classes 2 3 5 7` khớp lệnh replay; lệnh replay lấy từ `run_meta` và test F2a chạy **đúng** lệnh đó |
| GT trong chế độ dataset | Ghi rõ: file/thư mục không đọc GT; `--dataset` có thể đọc `ignored_regions` để mask theo quy ước benchmark, không đưa GT vào association; `run_meta.nguon.doc_gt` |
| Provenance entry point | `run_meta.entry_point` = sha256 LF của `track_video_file.py` + `working_tree_src` (file `src/` khác HEAD lúc chạy). Không mở rộng `CODE_FILES`; không bổ sung hồi tố cho 4 run cũ |
| Log lỗi đã xóa | Stdout thật của runner lượt 19:39 còn ở `/tmp` → lưu `handoff/logs/review9/lich_su_that_bai/` kèm README phân định: *đã xác minh* (chạy riêng 9/9, lượt sạch 30/30) vs *giả thuyết có chứng cứ gián tiếp* (3 commit `b897213`/`5e28f37`/`ef3316c` lúc 19:41–19:42 nằm trong cửa sổ chạy 19:39–19:44). Log chi tiết theo test **đã mất**, ghi nhận thiếu, không tái tạo |

## 3. Đo lại thời gian (mục 6 của `HE_THONG_PC.md`)

4 cấu hình × `MVI_39031` (1470 frame), output mới ở `results/track_video/v2_MVI_39031_*`, bản cũ giữ nguyên.
Log: `handoff/logs/review10/do_lai_4_cau_hinh.log` (gồm bảng đối chiếu Σ`frame_total` vs `wall_vong_lap`).
Mỗi cấu hình **một lượt**; không diễn giải thống kê. Chênh detector giữa các lượt (17,5–20,5 ms) chỉ là **quan sát**, chưa đủ bằng chứng quy nguyên nhân.

## 4. Còn mở

- Nano F1/F3 (review `8bab154`) và tài liệu `NANO_STATUS.md` mục 5: **hoãn** theo phạm vi.
- Camera trực tiếp chưa smoke; video fps biến thiên chưa xử lý.
- Review `8abb368` **chưa** PASS cho đến khi Codex xác nhận; base delta giữ `b623465`.
