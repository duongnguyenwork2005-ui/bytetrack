# R63 G3-C1 — gói v17-C trên Nano + so cặp pipeline 384×640 vs baseline (31/10/2026)

Boot `89a0f445-6290-427d-9305-08ed1adaa30a` (reboot ấm, thẻ HS 50 MHz), Chromium đóng, MAXN, một tác vụ Nano một lúc.

## Gói v17-C trên Nano

`r63_20261030/v17c/nano_transfer` = bản sao 562 file đúng MANIFEST v16 (kiểm hash) + thay đúng 2 file từ
`review63/v17_c/src/` (`nano_detector.py` `4d363fdc…`, `nano_detect_stream.py` `e3f3ab43…`, khớp PC; manifest báo
FAILED đúng 2 file này). Kiểm (runner `chay_mot_luot_r63.py`, CV, hiển thị bật, 300 frame):
- v17-C + engine 640 baseline `9fae6b43…`: MOT **`9df71094…` trùng byte** baseline v16 G1, letterbox
  `auto=False 640x640` ⇒ đường 640 không đổi.
- v17-C + engine C1 `a6912393…`: exit 0, parity live/replay trùng byte, letterbox `auto=False 384x640`, 13,22 FPS.

## So cặp (`cong_cu_v4/g3_so_cap.sh`, `c1_so_cap_01/`)

A = bản sao v16 + engine `9fae6b43…`; B = v17-C + engine `a6912393…`. 4 cặp CV + 4 cặp EKF, thứ tự trong cặp đảo
xen kẽ (cap1 cv AB, ekf BA; cap2 cv BA, ekf AB; …), nghỉ 20 s, mỗi lượt qua `chay_co_giam_sat.sh`.
**16/16 lượt hợp lệ**: exit 0, không vi phạm cổng, parity live/replay trùng byte; MOT ổn định theo nhánh (A: CV
`9df71094`, EKF `adfd863c`; B: CV `62c4596e`, EKF `92f6c3eb`), mỗi lượt đủ 300 frame.

| Cặp | tracker | FPS A | FPS B | B/A | decode A/B (ms) | exec A/B | trễ p90 A/B |
|:--|:--|--:|--:|--:|:--|:--|:--|
| 1 | cv | 9,252 | 13,279 | 1,435 | 11,5/11,4 | 69,0/43,9 | 119,3/86,3 |
| 1 | ekf | 9,225 | 13,186 | 1,429 | 11,5/11,5 | 69,0/44,0 | 121,6/88,4 |
| 2 | cv | 9,292 | 13,233 | 1,424 | 11,4/11,5 | 69,0/44,1 | 118,6/86,7 |
| 2 | ekf | 9,264 | 13,321 | 1,438 | 11,4/11,4 | 69,0/43,9 | 120,4/88,6 |
| 3 | cv | 9,281 | 13,252 | 1,428 | 11,5/11,5 | 69,0/44,0 | 118,6/86,1 |
| 3 | ekf | 8,828 | 13,211 | 1,496* | 16,5/11,5 | 68,8/44,0 | 125,4/88,5 |
| 4 | cv | 8,893 | 13,188 | 1,483* | 16,5/11,5 | 68,9/44,0 | 124,9/87,2 |
| 4 | ekf | 9,238 | 12,842 | 1,390* | 11,5/13,1 | 69,0/44,0 | 120,5/89,5 |

\* cặp lệch tầng decode giữa A và B (theo kế hoạch: báo riêng, không quy chênh cho tối ưu).

- **Trung vị tỉ lệ FPS theo cặp: CV 1,432 (min 1,424); EKF 1,434 (min 1,390)**. Chỉ các cặp cùng tầng decode:
  CV 1,424–1,435, EKF 1,429–1,438 ⇒ **≈ +43 %**. FPS trung vị: CV 9,27 → 13,24; EKF 9,23 → 13,20.
- Trễ p90 giảm ~27 % (tỉ lệ trung vị 0,725 CV / 0,732 EKF). Exec engine trong pipeline 69,0 → 44,0 ms.
- **Cổng tốc độ G3 ĐẠT** (≥ +5 % cả CV/EKF, p90 không tăng, đủ frame, live/replay trùng byte). Cổng chất lượng G4
  **chưa đo** ⇒ chưa được kết luận "không giảm chất lượng"; C1 chưa được đề xuất triển khai.
- **Lỗi provenance (phát hiện sau khi chạy):** runner `chay_mot_luot_r63.py` (bản lúc đó) truyền `--onnx-sha256` từ
  đường cố định `models/export/yolov8n_640_op12.onnx` ⇒ mọi lượt B (và `g3_v17c_kiem_384_cv`) ghi `onnx_sha256`
  = `a66a5e03…` (ONNX v16) thay vì `5f5b337d…` (ONNX C1). `weights_sha256`/`engine_sha256` = `a6912393…` ghi đúng, và
  `g3_c1_384x640_toc_do_01/engine/build_dieu_kien.txt` nối engine đó với ONNX `5f5b337d…` ⇒ số đo không bị ảnh hưởng;
  chỉ trường metadata sai. Đã thêm `--onnx` vào runner cho các lượt sau.
- Phạm vi: FPS xử lý 300 JPG MVI_39031 có hiển thị HDMI trên Nano 4 GB/JetPack 4.6/TRT 8.2/FP16, thẻ HS 50 MHz;
  không khái quát thành FPS MP4 hay thiết bị khác.
