# R63 — đính chính provenance, sai lệch quy trình và cách trình bày (01/11/2026, theo review GPT)

Nguồn: `D:/LABOTARY/automation/runs_local/hoi_codex_R63_chot_c1_20261101/answer.json` (`sua_truoc_khi_chot`).
Raw trên Nano và trong repo **giữ nguyên**; file này là lớp đính chính.

## 1. ONNX ghi nhầm ở các lượt C1 (nhánh B)

Runner `chay_mot_luot_r63.py` (bản 31/10) truyền `--onnx-sha256` từ đường cố định `models/export/yolov8n_640_op12.onnx`.
Kiểm `run_meta.json` trên Nano (01/11): 9 lượt sau ghi `onnx_sha256 = a66a5e03…` (ONNX v16) nhưng engine = `a6912393…`:

| Lượt (`~/nano_run/r63_20261030/…`) | onnx ghi | engine ghi | ONNX đúng |
|:--|:--|:--|:--|
| `g3_c1_so_cap_01/cap{1,2,3,4}_{cv,ekf}_B` (8 lượt) | a66a5e03 | a6912393 | **5f5b337d…** |
| `g3_v17c_kiem_384_cv` | a66a5e03 | a6912393 | **5f5b337d…** |

Nối đúng: engine `a6912393…` ← `g3_c1_384x640_toc_do_01/engine/build_dieu_kien.txt`:
`onnx_sha256=5f5b337d2978bd7b…` (ONNX `ung_vien_onnx/384x640/yolov8n_384x640_op12.onnx`, manifest cùng thư mục). Các lượt
nhánh A (`a66a5e03…` + engine `9fae6b43…`) ghi đúng. Số đo không bị ảnh hưởng (chỉ trường metadata).
Sửa công cụ cho lượt sau: runner có `--onnx` (31/10); `g3_so_cap.sh` (01/11) bắt buộc tham số ONNX ứng viên và truyền
`--onnx` cho nhánh B, ghi sha256 ONNX vào `dieu_kien.txt`.

## 2. Build vượt cổng swap — không phải "đạt cổng"

Build C1 (`g3_c1_384x640_toc_do_01`, swap +98 MB), C2 (+95 MB), C3 (+85 MB) **vi phạm cổng** swap tăng > 50 MB (toàn
bộ trong zram). Đã tiếp tục đo bằng engine đó (C1: quyết định thủ công 31/10; C2/C3: cơ chế tự miễn trong
`g3_duong_toc_do.sh` bản 01/11 sáng). Các build này **không** được ghi là đạt cổng. Số đo C1 giữ làm bằng chứng
có sai lệch đã công khai vì mọi lượt đo riêng (trtexec, bench, so cặp) có telemetry hợp lệ; không suy ra engine hỏng.
Đã **bỏ cơ chế tự miễn** trong `g3_duong_toc_do.sh` (vi phạm bất kỳ ⇒ dừng).

## 3. C2–C4 chạy ngoài trình tự

Kế hoạch (`01_ke_hoach_r63.md`, G3/G4 xen kẽ) chỉ cho xuống mức nhỏ hơn sau khi mức trước đạt chất lượng dev. C3/C4 (và
C2) được đo tốc độ trước khi có chất lượng dev ⇒ ghi là **khảo sát tốc độ ngoài trình tự**, không phải ứng viên đã qua
cổng. GPT (01/11) quyết định không mở C2/C3 trong vòng này.

## 4. Cách trình bày số tốc độ

- 16/16 lượt so cặp **vận hành hợp lệ**, nhưng chỉ **5/8 cặp cùng tầng decode** (CV cặp 1–3, EKF cặp 1–2); khoảng
  **+42–44 %** lấy từ 5 cặp này. Ba cặp lệch tầng (cặp 3 EKF, 4 CV, 4 EKF) giữ trong bảng, không quy toàn bộ chênh lệch
  cho tối ưu. Trung vị tỉ lệ 8 cặp (1,43) chỉ là số phụ.
- **Không** ghép trung vị G1 (8,81 / 9,02 FPS, lượt riêng) với C1 13,24 FPS để tính tăng tốc; chỉ dùng tỉ lệ trong cặp.
- Baseline là **mã v16 + engine build lại trên board hiện tại** (`9fae6b43…`), không phải engine v16 nguyên bản
  (`c32690cd…`, build board R62; chênh ~0,5 ms, `g1/02_hai_engine_trtexec_bench.md`).

## 5. Mô tả pixel

43,75 % là tỉ lệ **hàng đệm của đầu vào vuông 640×640** (280/640 hàng) với video 16:9. C1 384×640 giảm **40 % tổng pixel
đầu vào** (409.600 → 245.760) và vẫn còn 24 hàng đệm (12 + 12); vùng nội dung giữ nguyên 360×640 (cùng hệ số co 0,667).
Không viết "C1 loại bỏ 43,75 % tổng pixel", và không suy "nội dung giống nhau ⇒ detection không đổi" (detection đổi,
xem G4).

## 6. Parity Nano↔PC

Nhánh A chạy lại phần PC khi GPU rảnh (`g4/nano_pc_A_lan2.json`, log `g4/08_nano_pc_A_lan2.log`: 0 cảnh báo NMS) — kết
quả trùng lần đầu. Tiêu chí đạt ở **conf > 0,25** (C1 99,73/99,66 %, IoU 0,988; baseline 99,61/99,61 %, IoU 0,985). Ở
conf > 0,05 **chưa đạt 99 % hai chiều** (C1 99,0/98,9 %; baseline 98,8/99,0 %) ⇒ không khẳng định tương đương toàn bộ
detection hay toàn bộ đầu vào tầng điểm thấp của ByteTrack.
