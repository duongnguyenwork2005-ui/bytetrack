## 0. Trạng thái sau HAI vòng review độc lập — đọc trước

| Vòng | Phát hiện | Trạng thái | Bằng chứng |
|:--|:--|:--|:--|
| 1 (P1) | Baseline "SORT nguyên bản" xuất track sớm hơn upstream 1 frame | **Đã sửa + chạy lại Nhóm B** | `src/test_sort_upstream_parity.py` **8/8** so toàn chuỗi với mã upstream ghim (`third_party/sort_upstream`, commit `bce9f0d`), lệch ≤ **5,8·10⁻⁵ px**; kết quả `results/phase2/expB_detrac_{dev,test}_fix1/` |
| 1 (P1) | Runner chấm tracking mới bằng metric cũ | **Đã sửa** | `src/run_fingerprint.py`; `src/test_run_experiment_guard.py` **7/7** |
| 1 (P2) | Cache ghi thêm với `conf`/`classes`/`max_det` khác | **Đã sửa** | `src/det_cache.py`; `src/test_det_cache.py` **51/51** |
| 1 (P2) | ZIP chứa `.gitignore`, commit lệch | **Đã sửa** | `src/make_handoff.py` (`USER_FILES`, từ chối dựng khi dirty, `FILE_HASHES.csv`) |
| **2 (P1)** | **Resume sau khi bị gián đoạn / thiếu metadata có thể trộn cấu hình** | **Đã tái hiện + sửa** | Trạng thái **theo từng video** (`video_meta/<video>.json`, ghi nguyên tử) + `run_meta.status=running` trước khi ghi đè; `src/test_run_resume_guard.py` **10/10** qua đường chạy runner thật với lỗi giả lập |
| **2 (P2)** | **Cache cho ghi khi đổi phiên bản thư viện rồi từ chối đọc** | **Đã tái hiện + sửa** | `GEN_FIELDS` là **một** định nghĩa dùng cho cả đường ghi và đường đọc (gồm `versions`, `device`; bỏ đường dẫn `weights`); `src/test_det_cache.py` **51/51** |
| 2 | Diễn giải: IDSW "9–12 %", "0 mismatch", nhãn cặp KF 7-D → XYAH, C3, số Nhóm B cũ trong handoff | **Đã sửa** | `PHASE2_REPORT.md` mục 18.2, 18.4, 19; mục 5 của file này |

**Audit nguồn gốc `_fix1`** (`src/audit_run_provenance.py` → `results/phase2/run_provenance_audit.csv`):
**12/12** run Nhóm B `_fix1` có `run_meta` hợp lệ, **0 lần resume**, mỗi run là **một lượt chạy liên tục**
(14–123 s), `run_meta` ghi **sau** file tracking cuối, và **sha256 của từng file tracking hiện tại KHỚP
`eval_meta.json`** của lần tính metric ⇒ **giữ nguyên kết quả `_fix1`, không chạy lại**. 113 thư mục còn lại
thuộc bản runner cũ (**LEGACY** — không xác minh được bằng fingerprint), **0 nghi vấn**.

**Kết quả Nhóm B sau khi sửa** (cấu hình y hệt bản đã khóa, chỉ đổi `name`; không tuning) — **đây là bảng dùng
cho luận văn**, số cũ chỉ còn giá trị lịch sử:

| cặp (yếu tố được kiểm soát) — TEST 40 video, c25 | trước khi sửa | **sau khi sửa (`_fix1`)** |
|:--|:--|:--|
| bộ đệm `max_age` 1 → 30 | +0.0280 [+0.0207, +0.0350] | **+0.0274 [+0.0201, +0.0345]** |
| **gói** biểu diễn trạng thái **+ thiết lập bộ lọc** (KF 7-D + Q/R/P₀ của SORT → XYAH + Q/R/P₀ của ByteTrack) | −0.0133 [−0.0185, −0.0077] | **−0.0132 [−0.0185, −0.0077]** (thô −0.0132726 → −0.0132500) |
| motion model CV → EKF–CTRV (cùng XYAH, cùng vòng đời) | +0.0020 [−0.0001, +0.0042] | **+0.0020 [−0.0001, +0.0042]**, ΔIDSW **−108 [−186, −28]** |
| chỉ đổi bộ lọc ở `max_age` 1 | −0.0140 [−0.0167, −0.0116] | **−0.0140 [−0.0167, −0.0116]** |
| vòng đời SORT → ByteTrack (cùng EKF) | +0.0364 [+0.0298, +0.0428] | **+0.0363 [+0.0296, +0.0428]** |

IDSW giảm **2,8–9,8 % tùy nhánh** (orig30 2,79 %; cv30 4,41 %; ekf30 5,03 %; orig1 8,85 %; ekf1 9,79 %) —
**không** phải "9–12 % ở mọi nhánh" như bản trước viết. Mọi kết luận định tính **không đổi**. `bt-ekf`
(ByteTrack) **bit-identical** bản cũ: 12/12 video dev, 40/40 test ⇒ bản sửa SORT không đụng ByteTrack, nên
**không** cần chạy lại Nhóm A, Nhóm C hay Giai đoạn 1 (chỉ Nhóm B dùng `SORTTrack`).

Các hiệu số trên chỉ có nghĩa **trong đúng cặp được kiểm soát**; chúng **không** tạo thành một phép phân rã
cộng hay một thứ hạng áp dụng cho mọi dataset.
