# R63 G1(c) — baseline pipeline v16, engine build trên board hiện tại (31/10/2026)

Boot `89a0f445-6290-427d-9305-08ed1adaa30a` (thẻ `new high speed SDXC`, HS 50 MHz), engine `9fae6b43…`, bản sao
v16 `r63_20261030/baseline/nano_transfer` (562/562), runner `cong_cu_v4/chay_mot_luot_r63.py` (= R62 `520d373f` +
`--engine`, ghi hash engine + chế độ mmc1), `cong_cu_v4/g1_c_pipeline.sh`: 300 JPG MVI_39031, hiển thị HDMI bật,
thứ tự cv ekf ekf cv cv ekf ekf cv, nghỉ 20 s, mỗi lượt qua `chay_co_giam_sat.sh` (900 s). Bằng chứng nhỏ:
`c_pipeline_01/` (result, run_meta, before, command, giam_sat mỗi lượt).

**8/8 lượt hợp lệ:** exit 0, không vi phạm cổng (CPU/GPU max 49/46,5 °C, swap +0), parity live/replay trùng byte,
MOT trùng byte giữa các lượt cùng tracker (CV `9df71094…`, EKF `adfd863c…`; khác engine cũ `15ab38b4…` như dự kiến).

| Lượt | tracker | `fps_xu_ly` | decode | pre | exec | post | `t_update` | khoảng frame | trễ |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| 01 | cv | 8,786 | 16,57 | 14,04 | 68,87 | 7,48 | 6,05 | 113,54 | 121,6 |
| 02 | ekf | 9,241 | 11,51 | 13,66 | 68,99 | 7,49 | 6,60 | 107,82 | 116,8 |
| 03 | ekf | 8,794 | 16,58 | 13,89 | 68,97 | 7,40 | 6,63 | 113,38 | 122,4 |
| 04 | cv | 9,256 | 11,40 | 13,73 | 68,89 | 7,39 | 5,98 | 107,72 | 115,5 |
| 05 | cv | 8,760 | 16,66 | 14,11 | 68,92 | 7,32 | 5,87 | 113,74 | 121,6 |
| 06 | ekf | 9,253 | 11,40 | 13,76 | 69,01 | 7,36 | 6,69 | 107,74 | 116,4 |
| 07 | ekf | 8,783 | 16,64 | 14,06 | 68,95 | 7,34 | 6,55 | 113,58 | 122,2 |
| 08 | cv | 8,836 | 16,64 | 13,90 | 68,95 | 7,28 | 5,96 | 113,24 | 121,3 |

(ms, trung vị sau warmup 25.) **Trung vị FPS: CV 8,81; EKF 9,02** (khoảng 8,76–9,26).

## Nhận xét (đo được / suy luận)

- **Hai tầng decode** (như R62): lượt 02/04/06 decode ~11,4 ms → ~9,25 FPS; các lượt còn lại ~16,6 ms → ~8,78 FPS.
  Exec, pre, post không đổi giữa hai tầng. Nguyên nhân chưa xác định (suy luận: trạng thái xung CPU/cache). Theo
  kế hoạch: so cặp phải báo tầng decode; lệch tầng giữa hai nhánh thì chưa quy chênh FPS cho tối ưu.
- Phân rã ~113,5 ms/frame: exec engine 68,9 (≈ 61 %) + decode 16,6 + pre 14,0 + post 7,4 + tracker 6,0. Engine là
  nút thắt ⇒ G3 có cơ sở. Phép tính *giả định* (chưa đo): nếu exec 384×640 giảm theo pixel (×0,6) còn ~41 ms thì
  ~86 ms/frame ≈ 11,6 FPS; phải đo thật ở G3.
- Đây là cột "trước" (v16 bê nguyên, engine build trên board này) của bảng báo cáo; ≈ 9 FPS khớp v21–v23 và R62.
