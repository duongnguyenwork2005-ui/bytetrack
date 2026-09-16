# REVIEW_BRIEF — mốc 2, review vòng 3

<!-- BASE_TARGET -->

Base là gói delta bạn xác nhận đã đọc (`DELTA_REVIEW_9a1c404.md`, 422.749 byte / 71 file). Chỉ trình bày
thay đổi của **vòng này**; lịch sử đầy đủ ở `PHASE2_REPORT.md` §18–19.

## Xử lý từng phát hiện

| # | Phát hiện | Cách xử lý | File | Test — kết quả thực tế |
|:--|:--|:--|:--|:--|
| 1 | Bản vá cache chưa được tích hợp | Áp `cache_fix_review3.patch` (`--check --ignore-space-change` exit 0, **không xung đột**). `fill_env` hash **chính file checkpoint** đọc được; `write_meta` từ chối khi `weights_sha256` thiếu/không phải 64 hex, và từ chối append khi `spec_sha256` cũ khác cách tính hiện tại **kể cả khi `GEN_FIELDS` bằng nhau**; `mkdir` xuống **sau** nhánh "cache đã tồn tại" ⇒ từ chối thì không tạo gì; CLI nạp YOLO → `model.ckpt_path` đã resolve → hash → mới ghi metadata. Cache legacy **vẫn đọc được**, không gán fingerprint hồi tố; GT synthetic không cần checkpoint | `src/det_cache.py`, `src/test_cache_provenance.py`, `src/test_det_cache.py`, `src/run_all_tests.py` | `test_cache_provenance` **12/12**, `test_det_cache` **51/51**; danh sách nay **22 suite** |
| 2 | **P1** — file riêng bị `git rm --cached` lọt nguyên nội dung vào patch | Loại trừ áp dụng **triệt để** cho thêm/sửa/**xóa**/đổi tên, **trước** khi sinh payload lẫn patch. `make_patch()` quét **từng đường dẫn bên trong diff** (`diff --git`, `---`/`+++`, `rename from/to`); có đường dẫn bị loại ⇒ **dừng hẳn**. Kiểm tra cuối quét cả nội dung patch, không chỉ tên thành viên ZIP | `src/make_review_delta.py` | Trên code cũ: **marker nội dung riêng CÓ trong gói**. Sau sửa: không có diff lẫn marker; file trên ổ đĩa còn nguyên |
| 3 | **P2** — `git apply` exit 0 nhưng chưa tái tạo đúng target | Tách **tập thay đổi logic** (sinh patch) khỏi **tập payload vật lý** (nén vào ZIP). Dedup **chỉ** tác động payload, **chỉ** với file không phải code/config. `--no-renames` ⇒ đổi tên thành xóa + thêm, cả hai tên đều qua loại trừ và đều vào patch | `src/make_review_delta.py` | Trên code cũ: đổi tên → **thừa `src/old_util.py`**; hai file trùng nội dung → **thiếu `src/two.py`**. Sau sửa: cây code sau khi áp patch **khớp toàn bộ** target |
| 4 | Khôi phục trạng thái Git của file riêng bị commit nhầm ở `67445dc` | 6 file **vốn untracked** → `git rm --cached` đúng từng tên; `rerun_with_state_log.py` **vốn tracked** → đưa **index** về `23f63ed` bằng `update-index --cacheinfo`, **không đụng working tree**. Commit mới, không sửa lịch sử | commit `534146c` | sha256 file làm việc trước/sau: **0/7 file đổi nội dung**; `git diff 23f63ed HEAD -- src/demo .gitignore` **rỗng** |

Ba phép kiểm hồi quy ở mục 2–3 đều được **xác minh FAIL trên code trước khi sửa**. Hai phép P2 so **toàn bộ**
đường dẫn code/test/config sau khi áp patch với cây target, gồm cả file phải biến mất — không kết luận từ
exit code.

## Kiểm chứng đã chạy

| Lệnh | Kết quả thực tế |
|:--|:--|
| `--only test_cache_provenance test_det_cache test_review_delta` → `handoff/logs/review_delta_fix` | **3 PASS, 0 FAIL, 0 SKIPPED, 19 NOT_RUN**, exit 0 |
| `run_all_tests.py` → `handoff/logs/review_delta_fix_full` | **22 PASS, 0 FAIL, 0 SKIPPED, 0 NOT_RUN**, exit 0 |
| `src/smoke_cache_provenance_yolo.py` — YOLO **thật**, GPU thật, MVI_20052, 30 frame, 1648 box, cache-id mới trong thư mục **tạm** | **5/5 PASS**: `weights_sha256` == sha256 của chính `model.ckpt_path` (`f59b3d83…`); ghi → đọc strict `ok`; hash sai → chặn; thiếu hash → từ chối ghi, **không tạo thư mục**; `spec_sha256` cũ → chặn append, `meta.json` giữ nguyên |

Điểm để mở trong README bản vá ("chưa chạy inference thật") **đã đóng** trên máy dự án.

## Số liệu thực nghiệm

**Không số liệu nào thay đổi.** Không chạy lại A/B/C hay Giai đoạn 1, không tạo lại cache thí nghiệm, không
mở tuning. Smoke YOLO dùng cache-id mới trong thư mục tạm rồi xóa.

## Còn tồn tại

- 53/56 sequence cache `legacy` và 113 thư mục run `LEGACY`: sinh trước khi có fingerprint ⇒ không xác minh
  được bằng hash. Không tạo bằng chứng hồi tố.
- `video_meta/` chỉ có ở run sinh sau bản sửa (0/129 run bàn giao).
- BDD100K và Jetson Nano vẫn chưa có trên máy.

## Cần bạn kiểm kỹ

1. `skip_reason()` + `_paths_trong_diff()`: còn dạng header diff nào mang đường dẫn mà chưa bắt không.
2. Ranh giới dedup (`CODE_PREFIXES` / `DOC_EXACT`) đã đủ rộng chưa.
3. `ckpt_path`: smoke chạy với checkpoint **local**; đường alias/tự tải về chưa kiểm.

## Thứ tự đọc

`REVIEW_BRIEF.md` → `patch/delta_<base>..<target>.diff` → `logs/` + test liên quan → `data/` khi cần số liệu →
`DEPENDENCIES.md` (file phụ thuộc **không** đóng gói, kèm vị trí trong snapshot base và sha256). Thiếu file
nào, báo lại — sẽ bổ sung đúng file đó ở gói kế tiếp.
