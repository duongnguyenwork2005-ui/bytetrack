# R63 G6 — kết luận vòng tối ưu Nano (02/11/2026)

| Ứng viên | Nano (300 JPG, hiển thị bật, so cặp cùng boot) | Chất lượng dev (c25) | Báo cáo cuối (c25) | Quyết định |
|:--|:--|:--|:--|:--|
| v16 (letterbox vuông 640×640, engine `9fae6b43…`) | CV 9,27 / EKF 9,23 FPS (trung vị cặp) | đối chứng | đối chứng | bản đang dùng (launcher HDMI) |
| **C1 384×640** (v17-C, engine `a6912393…`) | CV 13,24 / EKF 13,20 FPS; **+42–44 %** ở 5/8 cặp cùng tầng decode; p90 trễ −27 % | ΔHOTA +0,02…+0,08; ΔmAP50 −0,29 (dev12) / +0,01 (BDD dev); recall −0,35…−1,01 | test40 ΔHOTA +0,09/+0,20, ΔmAP50 −0,17; BDD val ΔHOTA −0,02, ΔmAP50 −0,01; recall −0,18…−0,81 | **đề xuất v17 = C1** |
| C2–C4 (352×576 / 288×512 / 256×448) | chỉ GPU exec 37,2 / 27,2 / 21,0 ms (khảo sát ngoài trình tự) | chưa đánh giá | — | không mở trong vòng này (GPT Q1) |

Bằng chứng: tốc độ `g3/01`, `g3/02` (+ đính chính `g3/04`); dev `g4/10_ket_qua_g4.md`; báo cáo cuối
`bao_cao_cuoi/10_bao_cao_cuoi.md`; tương đương Nano↔PC ở conf > 0,25 (`g4/nano_pc_*.json`).

**Triển khai:** launcher HDMI **giữ v16** theo quyết định người dùng 01/11 ("chưa chuyển"). Gói C1 sẵn trên Nano:
`~/nano_run/r63_20261030/v17c/nano_transfer` + engine `g3_c1_384x640_toc_do_01/engine/…384x640…engine`. Chuyển launcher
cần quyết định riêng, kèm đường quay về v16 và kiểm HDMI/launcher.

**Phạm vi kết luận:**
- Mức tăng FPS đo trên **Jetson Nano này** (L4T 32.7.5, TRT 8.2.1, MAXN, thẻ SD giới hạn 100 MHz, 300 JPG có hiển thị).
  Cơ chế (bớt 40 % pixel đầu vào, vùng nội dung giữ nguyên 360×640) không phụ thuộc thiết bị, nhưng **tỉ lệ tăng trên
  Jetson khác phải đo lại**: engine TensorRT build riêng cho từng thiết bị; tăng tốc chỉ lớn khi engine là nút thắt.
- Chất lượng: tracking và AP nằm trong ngưỡng suy giảm định trước ở dev và hai tập báo cáo; recall detection giảm nhẹ
  nhất quán; tầng xe nhỏ dev12 thiếu GT; BDD val không có nhãn che khuất; CI AP test40/BDD val chưa tính. Không viết
  "hoàn toàn không ảnh hưởng".
- Câu báo cáo thầy: *"Mô hình v16 bê nguyên sang Jetson Nano chạy khoảng 9 FPS. Đưa ảnh vào mạng ở 384×640 (giữ nguyên
  vùng ảnh, bỏ phần đệm) làm FPS tăng khoảng 43 % (≈ 13 FPS). Trên UA-DETRAC test và BDD val, HOTA/IDF1 thay đổi không
  quá ±0,5 điểm, mAP giảm ≤ 0,2 điểm, recall detection giảm dưới 1 điểm ở các nhóm xe nhỏ/che khuất."*
