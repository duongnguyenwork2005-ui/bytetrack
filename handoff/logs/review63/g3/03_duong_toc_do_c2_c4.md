# R63 G3 — đường tốc độ C2–C4 (Nano, 01/11/2026, boot `81484e55-f21f-4ac4-92da-2ab477b97f91`)

Boot nguội ở nhà (IP 192.168.1.5), thẻ `SDR104` xung thực ~100 MHz; G0 `g0_suc_khoe_08_nha_0111` sạch (35 °C, swap 0,
MemAvailable 2,43 GB, không dấu hiệu panic, không Chromium). Công cụ `cong_cu_v4` trên Nano khớp hash bản đã commit
`b359515`. Chuỗi `cong_cu_v4/g3_duong_toc_do.sh g3_duong` (log `g3_duong_toc_do_01.log`): mỗi mức build (cùng cờ v16)
→ `g3_toc_do_ung_vien.sh` (trtexec A–E, bench 300 ảnh ứng viên, bench baseline `9fae6b43…` ngay sau).

> **Đính chính 01/11 (review GPT, `g3/04_dinh_chinh_provenance.md`):** C2–C4 được đo **ngoài trình tự** G3/G4 xen kẽ
> (chưa có chất lượng dev) ⇒ khảo sát tốc độ, không phải ứng viên qua cổng; GPT quyết định không mở C2/C3 trong vòng
> này. Cơ chế tự miễn cổng swap đã bị bỏ khỏi `cong_cu/g3_duong_toc_do.sh`.

**Sai lệch ghi nhận:** build C2 và C3 vi phạm cổng `swap tang 95/85 MB` (zram) — **vi phạm cổng**, đã tiếp tục do
cơ chế tự miễn (nay bỏ); build C4 không vi phạm. Mọi lượt đo (trtexec, bench) không vi phạm cổng; parity
đồng bộ/event 300/300 mọi lượt.

| Mức | engine | trtexec GPU A (ms) | bench GPU exec (ms) | letterbox+pre | post | infer host | detector/frame* | n_det TB (conf ≥ 0,05) |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| baseline 640×640 | `9fae6b43` | 67,94 | 68,19–68,28 | 13,4 | 7,3–7,4 | 73,6–73,7 | 94,4 | 10,18 |
| C1 384×640 (31/10, boot khác) | `a6912393` | 41,59 | 41,86 | 9,45 | 5,98 | 45,41 | 60,8 | — |
| C2 352×576 | `48d731a9` | 36,99 | 37,15 | 7,44 | 5,41 | 40,25 | 53,1 | 8,37 |
| C3 288×512 | `e9c37e82` | 26,89 | 27,16 | 6,59 | 4,36 | 29,53 | 40,5 | 7,38 |
| C4 256×448 | `e20136ff` | 20,82 | 20,99 | 5,31 | 3,66 | 24,02 | 33,0 | 6,20 |

\* letterbox+pre + infer + post (đồng bộ). Decode ảnh ~11,3 ms không đổi.

- GPU exec giảm gần theo số pixel: C2 −45 %, C3 −60 %, C4 −69 % so với 640×640.
- *Dự toán theo giả định (chưa đo pipeline):* lấy khoảng frame baseline tầng decode nhanh ~107,7 ms và trừ phần
  detector tiết kiệm được (kiểm lại với C1: dự toán 13,5 vs đo 13,24 FPS) ⇒ C2 ≈ 15, C3 ≈ 18,5, C4 ≈ 21,5 FPS.
- **Khác C1:** C2–C4 **thu nhỏ ảnh** (hệ số 0,60 / 0,533 / 0,467 thay vì 0,667) — số detection trung bình giảm rõ
  (10,2 → 8,4 / 7,4 / 6,2) ⇒ rủi ro chất lượng (xe nhỏ, xe xa) cao hơn C1; phải qua G4 trước khi so cặp pipeline.
- PC tái hiện được đúng tensor từng mức bằng `det_cache --imgsz 576|512|448` (rect, stride 32 ⇒ 352×576 / 288×512 /
  256×448 cho cả 960×540 và 1280×720; suy ra từ công thức LetterBox, cần kiểm pixel như `test_v17_c.py` trước khi dùng).
