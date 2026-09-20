# Phụ lục vòng 7 — chi tiết xử lý review `ca93c1e`

Bản tóm tắt: `handoff/REVIEW_BRIEF.md`. Tài liệu này giữ phần chi tiết đã chuyển khỏi brief.

## 0. Trạng thái trước khi sửa

`handoff/logs/review7/trang_thai_truoc_khi_sua.log` — branch `phase2-mixed-traffic`, HEAD
**đúng bằng `ca93c1e`** (0 commit sau). Working tree chỉ có 2 file **riêng của người dùng**
(`.gitignore`, `src/demo/rerun_with_state_log.py`) — **không** stage, **không** commit, **không**
đóng gói. Môi trường: Python 3.11.9, numpy 2.4.6, scipy 1.17.1, ultralytics 8.4.141,
torch 2.6.0+cu124, lap 0.5.13, pandas 2.3.3.

## 1. P1 — ground truth không nằm trong dấu vân tay

**Vị trí**: `src/run_fingerprint.py`, `src/run_experiment.py:stage_eval`.

**Tái hiện** (`handoff/logs/review7/repro_F1_truoc_khi_sua.log`, fixture trong thư mục tạm,
`stage_eval()` thật + TrackEval thật, **không** mock metric):

| Bước | GT_dets | IDF1 | HOTA |
|:--|--:|--:|--:|
| Chạy lần đầu (GT có vùng crowd) | 1 | 0,6666667 | 0,7071068 |
| Đổi dòng crowd → GT hợp lệ, chạy bình thường | **1 — metric cũ** | **0,6666667** | **0,7071068** |
| Cùng GT mới, ép `force=True` | **2** | **0,8** | **0,6832349** |

Khớp đúng bảng trong báo cáo review. Ba biến thể khác cũng tái hiện được:

| Biến thể | Trước khi sửa |
|:--|:--|
| Đổi `seqLength` 1 → 50 | trường lệch = `[]` (fingerprint **không** nhận ra) |
| Xóa hẳn `gt.txt` | trường lệch = `[]` (**không** phân biệt GT thiếu) |
| `run_state.py` trong `CODE_FILES` | **không có** |

**Sửa**:

- `gt_fingerprint(split, videos, seqmap_used)` — hash nội dung `gt.txt`, nội dung `seqinfo.ini`, và
  `seqLength` đã phân tích **tách riêng** (để `compare()` chỉ ra đúng "đổi độ dài timeline", không
  chỉ nói "seqinfo đổi"). Seqmap chỉ vào dấu vân tay **khi đường chạy thật sự đọc nó** — TrackEval
  bỏ qua seqmap khi được truyền `SEQ_INFO`.
- `occl_fingerprint()` thêm `gt` và `annotation_source` (các file `*_occlusion_segments.csv` /
  `track_gaps.csv` mà `occlusion_eval.py` thực sự đọc, phân giải đường dẫn y hệt nó).
- `check_eval_inputs()` chạy **trước** vòng lặp và trước mọi quyết định tái dùng; thiếu
  `gt.txt`/`seqinfo.ini` → `ThieuDauVaoDanhGia` (một `FileNotFoundError`) → exit ≠ 0.
- `run_state.py` thêm vào `CODE_FILES`.

**Quy ước hash** — một quy tắc duy nhất, công bố ở `run_fingerprint.HASH_CONVENTION` và docstring:

- văn bản (`.txt`/`.ini`/`.csv`): sha256 của nội dung **chuẩn hóa xuống dòng về LF**;
- nhị phân (`.npz`): sha256 **byte nguyên bản**;
- file thiếu: sentinel `MISSING` — **khác** hash thật của chuỗi rỗng, nên **GT rỗng hợp lệ không bao
  giờ bị lẫn với GT bị xóa**.

Chọn LF cho văn bản có lợi ích đo được: dấu vân tay GT **khớp trực tiếp** quy ước của bản khóa
(`src/make_lock.py`) — đã đối chiếu **200/200 video**. Đổi quy ước này làm trường `tracking` của 12
run cũ lệch, nên cả 12 được đánh giá lại (mục 3); metric **không đổi**.

> **Phạm vi ổn định giữa máy** (review `b623465` lưu ý đúng): chỉ các đuôi trong `TEXT_SUFFIXES`
> được chuẩn hóa. `code_fingerprint()` hash module `.py` theo **byte thô**, mà cây nguồn hiện có
> xuống dòng **lẫn lộn** — đã kiểm: `src/run_fingerprint.py` **281 CRLF / 0 LF**,
> `src/bytetrack_np.py` **0 CRLF / 125 LF** (do `core.autocrlf=true` + `* text=auto`). Vì vậy
> trường `code` **có thể lệch giữa Windows và Jetson Nano dù cùng commit**, và **không** được
> phát biểu "dấu vân tay không đổi giữa hai máy" một cách chung chung. Đây là việc phải xử lý
> **trước khi so provenance giữa hai máy** — xem `handoff/NANO_STATUS.md` mục "Điều kiện còn
> thiếu". Chưa sửa trong đợt này vì đổi `TEXT_SUFFIXES` sẽ làm 12 `eval_meta.json` vừa được review
> thành cũ, trong khi chưa chạy gì trên thiết bị.

**Bằng chứng sau sửa**: `src/test_eval_guard.py` — **11/11 PASS**, mọi phép kiểm đi qua
`stage_eval()` thật.

## 2. P2 — thống kê detection / thời gian không theo manifest

**Tái hiện** (`handoff/logs/review7/repro_P2_truoc_khi_sua.log`):

| Thiết lập | Đầu vào tracker thật | Công cụ báo cáo (trước sửa) |
|:--|--:|--:|
| 1 video trong manifest, 1 frame 3 detection, `max_det=1` | 1 | **3** |
| Thêm 1 video ngoài manifest có 2 detection | 1 | **5** |
| Sidecar: video chọn 1 s, video ngoài manifest 20 s | 1 s / 1 frame | **21 s / 2 frame** |
| Sidecar thiếu hẳn trường `seconds` | — | cộng như **0 s** |

**Sửa**: `videos_of(exp)` đọc manifest (không có manifest → lỗi rõ ràng, **không** âm thầm quét thư
mục); `_det_counts()` đi qua `det_cache.open_cache(...).frame(i)` — đúng đường đọc của runner, áp đủ
`conf_min`, `classes`, **`max_det` theo từng frame**, `class_map`; `_detector_seconds()` trả về
`du_lieu_du` và danh sách `thieu_seconds` / `thieu_sidecar`, và `toan_pipeline()` **từ chối** đưa ra
tổng khi thiếu số đo.

**Bằng chứng sau sửa**: `src/test_report_stats.py` — **8/8 PASS**.

**Kiểm lại trên cache BDD thật**: hai bảng **không đổi**.

| | trước | sau |
|:--|--:|--:|
| car `c25` / `c10` | 193.945 / 385.617 | 193.945 / 385.617 |
| motorcycle `c25` / `c10` | 351 / 1.083 | 351 / 1.083 |
| detector | 22,9 ms/frame, 15,3 phút, 39.973 frame | y hệt |

Lý do đã kiểm chứng, không phải suy đoán:

- cache chứa **đúng 200** video của manifest — 0 npz thừa, 0 video thiếu npz ⇒ nhánh "video ngoài
  manifest" **không bao giờ kích hoạt** trên dữ liệu này;
- số detection **tối đa trong một frame** sau khi lọc 5 lớp: **25** (`conf>0,25`, tại
  `b200e97a-bf074435` frame 2) và **59** (`conf>0,10`, tại `b1e9ee0e-67e26f2e` frame 46) — đều xa
  `max_det` 300 ⇒ phép cắt `[:max_det]` **không bao giờ kích hoạt**;
- **0** sidecar thiếu `seconds`.

## 3. Đánh giá lại từ tracking hiện có

Lệnh đã xác minh bằng `--help` trước khi chạy: `--stage eval` chỉ gọi `stage_eval`, **không** chạm
`stage_track` (`src/run_experiment.py:main`). Bản trước lưu nguyên vẹn ở
`results/phase2/_PRE_P1_ca93c1e/` (69 file, giữ metadata gốc).

Cả 12 run chạy lại, lý do in ra: `tracking, gt, code` — `tracking` vì đổi quy ước hash văn bản,
`gt` là trường mới, `code` vì `run_experiment.py` đổi và `run_state.py` được thêm vào danh sách.

| Đối chiếu | Kết quả |
|:--|:--|
| 2.400 file tracking (12 run × 200 video) | **0 lệch** (hash **chuẩn hóa LF**, không phải so byte thô) |
| 12 `run_meta.json` | **0 lệch** |
| 56 giá trị metric | lệch tối đa **0,000e+00** |
| 100 dòng `paired_pairs.csv` (Δ, CI thấp, CI cao) | lệch tối đa **0,000e+00** |
| `gt_per_video` vs `lock_bdd_transfer.json` | **200/200 khớp** |
| 12 `eval_meta.json` có trường `gt` | 12/12, `n_missing_gt=0`, `seqmap_used=False` |

Kết luận: lỗi P1 **có thật** (đã tái hiện) nhưng **không** ảnh hưởng số liệu BDD đã bàn giao.
Không thêm dấu vân tay hồi tố cho bản đánh giá cũ — bản cũ được **lưu nguyên**, bản mới là kết quả
của một lần đánh giá mới.

## 4. Diễn giải đã thu hẹp

Sửa ở **nguồn sinh** (`src/report_bdd_transfer.py`) nên lần sinh sau không đưa lỗi trở lại.

| Mục | Trước | Sau |
|:--|:--|:--|
| Tốc độ | "cận trên"; "5 lớp + `max_det` 300 **sẽ nhanh hơn**"; "gần như không đổi FPS" | "**ước tính cộng hai pha đo riêng**"; chỉ nói sẽ làm **ít việc hơn ở NMS/hậu xử lý** nhưng **chưa đo**; in khoảng FPS tính từ số liệu: **35,6 (`pf500`) → 41,8 (`cv`), chênh 17,4 %** (c25) |
| PF seed | "SD 0,00005–0,00095" | SD ghi **theo từng metric**: HOTA 0,0000454/0,0009525 · IDF1 0,0010953/0,0009475 · AssA 0,0007535/**0,0029546** (c25/c10); kết luận giới hạn trong **ba seed đã thử**, N=500, implementation này |
| Timeline | "mọi FP sau frame 94 **biến mất**", "làm đẹp số liệu" | TrackEval 1.3.0 (`mot_challenge_2d_box.py:222–224`) **ném `TrackEvalException`** với timestep ngoài `seqLength`; phát biểu: độ dài 94 **sai**, cần giữ đến 202, evaluator **có thể từ chối dữ liệu**. Câu cũ **đã rút lại** |
| Quy mô | "12 lượt × 2 điều kiện" | **6 cấu hình/seed × 2 điều kiện = 12 run**, mỗi run 200 video; pilot là 12 run **riêng** trên 3 video |
| Mức kết luận | "lợi thế là thật" | "**CI 95 % của phép so sánh đã định trước không chứa 0**"; nêu rõ **chưa** hiệu chỉnh đa so sánh; phạm vi là protocol **gộp phương tiện** trên nhãn **COCO-video chuyển đổi**, không phải benchmark native, không phải bằng chứng phục hồi che khuất |
| Ngoại lệ PF | `cap_so_sanh.md` nói mọi CI cùng đại lượng với bảng chính | Ghi rõ: bảng chính trung bình **metric** theo seed, bootstrap trung bình **thống kê đủ** rồi tái tổ hợp ⇒ Δ của `pf500` **không** bằng hiệu hai ô bảng chính |

### Đính chính so sánh với DETRAC

Brief vòng trước viết "DETRAC p = 0,22; BDD CI hẹp vì có 200 video". Số thật (`expA_detrac_test`,
cặp `ekf − cv`, bootstrap ghép cặp theo video):

| Điều kiện | ΔHOTA [CI 95 %] | CI chứa 0 |
|:--|:--|:--|
| DETRAC test **c25** | +0,00117 [−0,00029; +0,00266] | **có** |
| DETRAC test **c10** | +0,00331 [+0,00113; +0,00569] | **không** |

⇒ DETRAC **c10 đã** phân biệt được; chỉ **c25** là không. Đã bỏ cách giải thích nhân quả "CI hẹp chỉ
vì cỡ mẫu" — phân bố video, dữ liệu và độ lớn hiệu ứng đều khác giữa hai dataset. Báo Δ và CI
**riêng cho từng dataset × điều kiện**.

### Đính chính trong bản khóa

`configs/phase2/LOCK_bdd_transfer.md` dòng "Quy mô" trước đây viết
*"12 (6 tracker-run × 2 điều kiện) × 200 video"* — dễ đọc nhầm thành 2.400 run. Đã sửa **câu chữ**
kèm ghi chú đính chính ngay trong file. **Ma trận không đổi** và bản máy đọc
`results/phase2/lock/lock_bdd_transfer.json` **không bị sửa** (đã kiểm bằng `git diff`).

## 5. Phần còn bị chặn

| Việc | Trạng thái |
|:--|:--|
| Jetson Nano | **chưa kết nối**, chưa có số đo nào. `scripts/nano/` có script nhưng chưa chạy lượt nào. Không suy ra tình trạng kết nối chỉ từ việc thiếu SSH config / thư mục kết quả |
| Thí nghiệm 2 (tuning trên BDD train-dev) | **chưa đủ điều kiện**: máy chỉ có split `val` |
| Chỉ số phục hồi sau che khuất trên BDD | **không làm được**: nguồn COCO-video thiếu `occluded` |
| Metric theo từng lớp | **không báo**: GT gộp 5 lớp xe thành một |
