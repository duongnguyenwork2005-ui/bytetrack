# REVIEW_BRIEF — mốc 2, review vòng 2

**Base** (snapshot bạn đã đọc): `4889209` — gói đầy đủ `phase2_review_latest.zip`, `git_head` `4889209`,
tương ứng `PHASE2_REVIEW_FIX1.md`. **Target**: `35dd59e`, nhánh `phase2-mixed-traffic` (chỉ commit cục bộ).
Chỉ trình bày thay đổi của **vòng này**; lịch sử đầy đủ ở `PHASE2_REPORT.md` §18–19.

## Xử lý từng phát hiện

| # | Phát hiện / yêu cầu | Cách xử lý | File | Test — kết quả thực tế |
|:--|:--|:--|:--|:--|
| 1 | **P1** — run bị ngắt hoặc thiếu metadata có thể trộn đầu ra hai cấu hình | Trạng thái **theo từng video** (`video_meta/<video>.json`, ghi nguyên tử sau khi video xong); `can_reuse_video()` đòi sidecar + `complete` + fingerprint khớp, thiếu/hỏng ⇒ chạy lại; `status="running"` ghi **trước** mọi thao tác ghi đè | `src/run_experiment.py` | `test_run_resume_guard.py` **11/11 PASS** — qua runner thật, lỗi giả lập ở video 2 |
| 2 | **P2** — cache cho ghi khi đổi phiên bản thư viện rồi từ chối đọc | **Một** tập `GEN_FIELDS` cho **cả** đường ghi (khác ⇒ từ chối, đĩa không đổi) **và** đường đọc (`spec_sha256`); loại đường dẫn `weights`, giữ `weights_sha256`; bỏ backfill hồi tố | `src/det_cache.py` | `test_det_cache.py` **51/51 PASS** |
| 3 | Audit `_fix1`, chỉ chạy lại khi có bằng chứng | Đối chiếu sha256 **từng file tracking hiện tại** với `eval_meta.json` của lần tính metric | `src/audit_run_provenance.py` → `results/phase2/run_provenance_audit.csv` | **12/12 khớp hash, 0 resume, 0 nghi vấn** ⇒ **giữ nguyên, không chạy lại**; 113 thư mục cũ = `LEGACY` |
| 4 | Đồng bộ diễn giải (chi tiết §19) | IDSW **2,8–9,8 % tùy nhánh** thay "9–12 %"; cache **3 `ok` / 53 `legacy` / 0 `mismatch`** kèm giới hạn; C3 → "không cấu hình nào **cải thiện** HOTA có ý nghĩa"; nhãn Nhóm B = đổi **gói** biểu diễn **+ thiết lập bộ lọc**; `has_full` = đoạn che ≥ 0,90; bỏ HEAD cũ `7610b56` | `PHASE2_REPORT.md`, `CODEX_HANDOFF.md`, `handoff/_review_section.md`, `_test_section.md` | — |
| 5 | Đưa `eval_meta.json` / `occl_meta.json` vào gói | `collect()` gom thêm cả `video_meta/`; ghi chú nêu rõ **16/127** thư mục trackeval có `eval_meta.json`, **0/129** có `video_meta/` | `src/make_handoff.py` | `run_all_tests.py` **20/20 PASS, 0 FAIL, 0 SKIPPED** |

## Số liệu thực nghiệm

**Không số liệu thực nghiệm nào thay đổi.** Đối chiếu: `paired_pairs.csv` của `expB_detrac_{dev,test}_fix1`
chỉ khác ở cột `note` (nhãn); chạy lại `aggregate_phase2.py` cho đúng giá trị cũ; 12/12 run `_fix1` có sha256
khớp `eval_meta.json`. Không chạy lại tracking, không tuning/ablation trên test.

Hai con số **trình bày** được sửa: IDSW "9–12 % ở mọi nhánh" → **2,8–9,8 % tùy nhánh**; cặp
`sort-cv30 − sort-orig30` là −0.013273 (trước) và −0.013250 (sau), **cả hai làm tròn thành −0.0133** — bản
trước ghi nhầm −0.0132.

## Còn tồn tại

- **53/56 sequence cache `legacy`** và **113 thư mục run `LEGACY`**: sinh trước khi có fingerprint ⇒ không
  xác minh được bằng hash; 0 dấu hiệu nghi vấn nhưng cũng **không** chứng minh được. Không tạo bằng chứng hồi tố.
- `video_meta/` chỉ có ở run sinh sau bản sửa (0/129 run bàn giao).
- BDD100K và Jetson Nano vẫn chưa có trên máy.

## Cần bạn kiểm kỹ

1. `GEN_FIELDS` có **thiếu** trường nào ảnh hưởng detection không; loại đường dẫn `weights` có đúng không.
2. Có đường nào tái sử dụng đầu ra mà **không** đi qua `can_reuse_video()` không.
3. Tiêu chí `OK` / `LEGACY` / `NGHI VAN` của audit có đủ mạnh để kết luận "không chạy lại `_fix1`" không.

## Thứ tự đọc

`REVIEW_BRIEF.md` → `patch/delta_4889209..35dd59e.diff` → `logs/` + test liên quan → `data/` khi cần số liệu →
`DEPENDENCIES.md` (file phụ thuộc **không** đóng gói, kèm vị trí trong snapshot base và sha256). Nếu thiếu file
nào, báo lại — sẽ bổ sung đúng file đó ở gói delta kế tiếp.
