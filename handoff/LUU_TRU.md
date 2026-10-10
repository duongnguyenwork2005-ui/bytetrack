# LUU_TRU — chỉ mục tài liệu lịch sử, vùng lưu trữ và cách khôi phục (từ 2026-10-10)

Trạng thái hiện hành **không** nằm ở đây: xem `handoff/PLAN_DA_CHOT.md` mục 0. Lộ trình đọc: `AGENTS.md`.

## 1. File nào thay file nào

| Nội dung | Nguồn hiện hành | Bản cũ / lịch sử (giữ nguyên vị trí, chỉ mở khi cần) |
|:--|:--|:--|
| Trạng thái, việc dở, bước tiếp | `handoff/PLAN_DA_CHOT.md` mục 0 | `handoff/REVIEW_BRIEF.md` (brief của vòng giao gần nhất), bảng vòng trong `CODEX_HANDOFF.md` mục 4, mục "Trạng thái các giai đoạn" trong `README.md` |
| Phạm vi, thứ tự mốc, quy tắc | `handoff/PLAN_DA_CHOT.md` (sau mục 0) | — |
| Chi tiết từng vòng | vòng mới nhất `handoff/REVIEW46_CHI_TIET.md`, R47 ở `handoff/logs/review47/` | `handoff/REVIEW7..45_CHI_TIET.md` |
| Kết luận review Codex | review mới nhất của vòng đang xét | `handoff/CODEX_REVIEW_<commit>.md` (36 file) — bằng chứng nghiệm thu, không xóa |
| Kết quả tracking mốc 2 | `CODEX_HANDOFF.md` mục 5 + 8, `PHASE2_REPORT.md` | `handoff/MOC2_FINAL_SUMMARY.md` |
| Kết quả giai đoạn 1 | `BAO_CAO_KHOA_LUAN.md` | `FIX_REPORT.md`, `TEST_EVALUATION_REPORT.md` (test trong `src/` còn dẫn tới) |
| Nano / PC launcher (mốc 1, 6 — đang hoãn) | `handoff/NANO_STATUS.md` | `handoff/NANO_LOCAL_UI.md`, `handoff/LAUNCHER_PC.md`, `handoff/HE_THONG_PC.md` |
| Detector 5 lớp | `handoff/DETECTOR_TRAIN_PLAN.md`, `handoff/BDD100K_INVENTORY.md` | `handoff/BDD_TRANSFER_NOTES.md`, `handoff/BACKEND_NP_NOTES.md` |
| Chỉ mục gói đầy đủ (sinh tự động, 16/09) | — | `handoff/INDEX.md`, `handoff/FILE_HASHES.csv`, `handoff/versions.json` (do `src/make_handoff.py` sinh lại) |

Các file lịch sử **không** bị di chuyển: hơn 160 file (gồm test trong `src/`) dẫn chiếu chúng theo đường dẫn, và
chúng là bằng chứng review. Chúng được tách khỏi lộ trình đọc mặc định bằng `AGENTS.md`, không bằng cách dời chỗ.

## 2. Vùng lưu trữ trong repo (bị git ignore)

- `handoff/archive/`: gói delta / đầy đủ / Nano theo commit, ZIP bằng chứng review (`CODEX_REVIEW_*_EVIDENCE.zip`),
  `kiem_goi_cuoi_v44`, `kiem_goi_cuoi_v46`.
- `handoff/*_latest.zip`: gói giao gần nhất (sinh lại bằng `src/make_review_delta.py`, `src/make_handoff.py`,
  `src/make_nano_transfer.py`).

## 3. Dọn ngày 2026-10-10 — đã chuyển ra ngoài repo

Gốc lưu trữ: `D:\LABOTARY\_luu_tru\pre_cleanup_20261010-1429\`
- `sao_luu_zip/<tên>.zip` + `<tên>.manifest.csv` (cột: đường dẫn gốc, đường dẫn trong zip, size, sha256).
- `da_chuyen/<đường dẫn tương đối từ D:\LABOTARY>`: chính các file đã chuyển.

| Đường dẫn cũ | Lý do | Bản khác còn lại |
|:--|:--|:--|
| `CODEX_REVIEW_c4d64a1_EVIDENCE/` (gốc repo, chưa track) | trùng 43/43 file theo sha256 | `handoff/archive/CODEX_REVIEW_c4d64a1_EVIDENCE.zip`; nhánh `backup/pre-cleanup-20261010-1429` |
| `handoff/nano_transfer/` (chưa track, 563 file, chứa bản sao mã cũ) | trùng 563/563 file theo sha256 | `handoff/nano_transfer_latest.zip`; sinh lại: `python src/make_nano_transfer.py build` |
| `handoff/patches/moc2_e353b84..7341812.diff` (chưa track, 18 MB) | sinh lại được, trùng từng byte | `git -c core.autocrlf=false diff e353b84..7341812 -- . ":(exclude)results" ":(exclude)handoff/logs"`, rồi đổi LF → CRLF |

Đã xóa: 22 thư mục `__pycache__` (bytecode, git ignore, mọi công cụ đóng gói / fingerprint đều bỏ qua).

## 4. Khôi phục

- File của repo tại thời điểm trước khi dọn: nhánh `backup/pre-cleanup-20261010-1429` (commit `e664e93`, cha `e4a0311`),
  gồm 8 file đang sửa và 452 file chưa track, ghi **nguyên byte**. Khôi phục một file mà không đụng file khác:
  `git cat-file blob backup/pre-cleanup-20261010-1429:<đường_dẫn> > <đường_dẫn>`.
- Mục đã chuyển ra ngoài: chuyển ngược thư mục/file từ `da_chuyen\` về đúng đường dẫn cũ, hoặc giải nén zip tương ứng
  vào `D:\LABOTARY\`; kiểm lại bằng cột sha256 của manifest.
- Không cần `git reset --hard`, `git clean` hay force-push cho bất kỳ bước nào ở trên.
