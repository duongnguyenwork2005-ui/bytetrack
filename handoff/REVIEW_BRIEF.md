# REVIEW_BRIEF — vòng 7: xử lý REQUEST CHANGES của review `ca93c1e`

<!-- BASE_TARGET -->

Trạng thái review `ca93c1e`: **REQUEST CHANGES**. Base delta chuyển sang `ca93c1e` chỉ để ghi nhận
**snapshot đã được đọc**, không phải dấu hiệu chấp thuận. HEAD lúc bắt đầu đúng bằng `ca93c1e`.
Chi tiết, bảng số đầy đủ và phần còn bị chặn: **`handoff/REVIEW7_CHI_TIET.md`**.

## Finding → tái hiện → sửa → bằng chứng

| # | Tái hiện (fixture tạm, `stage_eval()` thật) | Sửa | Sau sửa |
|:--|:--|:--|:--|
| **P1** GT ngoài fingerprint | Đổi `gt.txt` cùng đường dẫn → `TAI SU DUNG`, giữ metric cũ (đúng phải là 2 / 0,8 / 0,6832349). Đổi `seqLength`, **xóa** `gt.txt` cũng không lệch | Hash `gt.txt` + `seqinfo.ini` + `seqLength` riêng + seqmap khi thật sự đọc; `occl_fingerprint` thêm GT và file segment; `check_eval_inputs()` chạy **trước** khi xét tái dùng; `run_state.py` vào `CODE_FILES` | `test_eval_guard` **11/11**: đổi GT → `ly do: gt`, ra đúng số; xóa GT → lỗi, exit ≠ 0; **GT rỗng ≠ GT thiếu** |
| **P2** không theo manifest | 1 detection thật → báo **5**; 1 s/1 frame → **21 s/2 frame**; thiếu `seconds` cộng như **0** | Đọc manifest; đếm qua `open_cache(...).frame(i)` (áp `max_det` **từng frame**); thiếu số đo → **từ chối** đưa ra tổng | `test_report_stats` **8/8** |
| **P3** test không gọi runner | Phép kiểm 5 chỉ tính lại biểu thức, kiểm `rex is not None` | Gọi `stage_eval()` thật, đọc `eval_meta.json` | `test_evaluator_choice` **7/7**, có **độ nhạy**: ép `mot` → FP 1→2 |

## Đánh giá lại từ tracking hiện có (không chạy lại detector/tracker)

`--stage eval` → `--stage aggregate`, exit 0; 12 run chạy lại, lý do `tracking, gt, code`.

| Đối chiếu | Kết quả |
|:--|:--|
| 2.400 file tracking · 12 `run_meta.json` | **0 lệch** (provenance giữ nguyên) |
| 56 giá trị metric · 100 dòng bootstrap (Δ + CI) | lệch tối đa **0,000e+00** |
| `gt_per_video` vs bản khóa | **200/200 khớp** |

⇒ P1 **có thật** nhưng **không** ảnh hưởng số liệu BDD đã bàn giao. Bản trước giữ nguyên ở
`results/phase2/_PRE_P1_ca93c1e/`.

Bảng detection và thời gian detector cũng **không đổi**; lý do đã kiểm (cache đúng 200 video
manifest, frame đông nhất 25/59 detection — xa `max_det` 300, 0 sidecar thiếu `seconds`) nằm ở
phụ lục mục 2.

## Diễn giải đã thu hẹp (sửa ở **nguồn sinh**, không sửa file đã sinh)

Bỏ "cận trên" và bảo đảm "5 lớp sẽ nhanh hơn" → **ước tính cộng hai pha đo riêng**; thay "gần như
không đổi FPS" bằng khoảng tính từ số liệu **35,6 → 41,8 FPS, chênh 17,4 %**. SD của PF ghi **theo
từng metric**, kết luận giới hạn trong ba seed đã thử. Hậu quả lỗi timeline: TrackEval **ném lỗi**
với timestep ngoài `seqLength` — **rút lại** câu "mọi FP sau frame 94 biến mất". Quy mô: **6 cấu
hình/seed × 2 điều kiện = 12 run**. Dùng "CI 95 % của phép so sánh đã định trước không chứa 0", nêu
rõ **chưa** hiệu chỉnh đa so sánh. **Đính chính DETRAC**: test **c10** đã có CI HOTA **không chứa 0**;
chỉ **c25** chứa 0 — đã bỏ giải thích "CI hẹp vì có 200 video".

## Trạng thái

**Test: 28/28 PASS, 0 FAIL** (`handoff/logs/review7/`). **Nano: chưa kết nối**, chưa có số đo.
Còn bị chặn: Thí nghiệm 2 (chỉ có split `val`), chỉ số che khuất (nguồn thiếu `occluded`), metric
theo lớp (GT gộp 5 lớp xe).
