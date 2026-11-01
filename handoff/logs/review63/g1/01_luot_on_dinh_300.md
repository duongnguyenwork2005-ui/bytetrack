# R63 G1 — lượt ổn định 300 frame đầu tiên (31/10/2026, boot `bd590ce3-6bdb-4411-afec-751509f557c6`)

Mục đích (theo review GPT DUYET_CO_SUA mục 3): chạy một lượt pipeline 300 frame có telemetry và hậu kiểm kernel
**trước** chuỗi benchmark/build. Không phải số đo so sánh.

- Cấu hình: Nano IP trường 10.11.61.127, rootfs thẻ acer; DTB `_sd100`; lần boot nguội này thẻ vào
  `SDR104`, `ios` clock 204000000 nhưng **actual clock 99840000 Hz**, 1,8 V (người dùng đọc bằng sudo).
- Lệnh (Nano, cwd bản sao v16 `r63_20261030/baseline/nano_transfer`):
  `bash ../../cong_cu_v4/chay_co_giam_sat.sh ../../g1_on_dinh_300_cv_engine_cu_01 900 -- <env nano>/python
  ~/nano_run/r62_20261028/chay_mot_luot.py --tracker cv --display on --frames 300 --out ../../g1_on_dinh_300_cv_engine_cu_01/luot`
  (runner R62 `520d373f…`, khớp repo; engine cũ `c32690cd…`).
- Kết quả: **máy không sập**, boot_id không đổi, CPU/GPU max 58,5/55,5 °C. Nhưng **lượt KHÔNG HỢP LỆ**:
  `nano_live` exit 3 (`con thoat ma -15 sau EOS`), swap tăng 181 MB (vi phạm cổng), `fps_xu_ly` 4,88, khởi tạo 106 s,
  decode trung vị 17,3 ms nhưng trung bình 61 ms, p90 204 ms.
- Nguyên nhân (đo được): trên desktop Nano đang mở **Chromium** (3 tiến trình, ~0,7 GB RSS, ~90 % CPU),
  `gnome-software`, evolution; nhiệt nhàn rỗi trước lượt CPU 52,5 °C (> cổng 50 °C), MemAvailable 1,83 GB; swap
  gồm zram 4×494 MB + `/swapfile` 4 GB trên thẻ SD. Đây đúng là trường hợp Q6 (người dùng tự đóng ứng dụng nền).
- Hành động: không chạy lại cho tới khi người dùng đóng Chromium/gnome-software/update-manager/jtop, rồi G0 lại.
  Không dùng số 4,88 FPS cho bất kỳ kết luận nào.
- Ghi chú boot: lượt 01 chạy trong boot `45a44b6e-663e-4d96-8795-269e606412eb` (11:34); boot `bd590ce3…` lúc 11:26 kết
  thúc do **người dùng khởi động lại** (người dùng xác nhận), pstore rỗng.

## Lượt 02 (sau khi người dùng đóng Chromium và reboot ấm, boot `89a0f445-6290-427d-9305-08ed1adaa30a`)

- Thẻ `new high speed SDXC` (HS 50 MHz — reboot ấm). G0 `g0_suc_khoe_07_truong_sau_reboot`: CPU/GPU 38/37,5 °C,
  MemAvailable 2,81 GB, swap 0, không dấu hiệu panic, không Chromium; lúc bắt đầu `packagekitd` ~47 % CPU,
  `gnome-software` ~22 % (dịch vụ khởi động, Q6 giữ nếu nhàn rỗi — lần này chưa nhàn rỗi hẳn).
- `g1_on_dinh_300_cv_engine_cu_02`: **ĐẠT** — exit 0, không vi phạm cổng (CPU/GPU max 41,5/39,5 °C, swap +1 MB),
  `fps_xu_ly` **8,68**, khởi tạo 83,6 s, parity live/replay trùng byte; `MVI_39031.txt` `15ab38b4…` **trùng byte**
  với cả 4 lượt CV baseline300 R62 và lượt 01. Trung vị sau warmup (ms): decode 16,56, pre 14,01, h2d 2,81, exec 69,3,
  d2h 1,68, post 7,35, `t_update` 5,82, khoảng giữa frame 113,87. Đây là lượt ổn định, **không** phải số so sánh.
