# R63 G0 — tiền kiểm Nano chỉ đọc (30/10/2026, 16:54 giờ Nano)

Không reboot, không sudo (sudo -n bị từ chối như dự kiến), không đổi nvpmodel/clocks, không cài gói. Chỉ tạo
`~/nano_run/r63_20261030/` (công cụ + kết quả). Bằng chứng gốc: `g0_suc_khoe_01/` (lần đầu) và `g0_suc_khoe_02/`
(sau khi sửa bộ tóm tắt).

- **Truy cập:** Nano đổi IP do DHCP: `jetson-desktop` ở **192.168.1.5** (wlan0); 192.168.1.4 giờ là thiết bị khác
  (MAC ngẫu nhiên, cổng 22 từ chối). Alias `nano` trong `~/.ssh/config` của PC đã đổi sang 192.168.1.5
  (bản sao cũ `config.bak_r63`). Host key đã có trong known_hosts và khớp.
- **Phần mềm:** L4T R32.7.5, TensorRT 8.2.1-1+cuda10.2 — như R62. nvpmodel **MAXN**; không đọc được
  `jetson_clocks --show` (cần sudo). CPU governor schedutil 1,224–1,479 GHz; GPU nvhost_podgov 76,8–921,6 MHz.
- **Tài nguyên lúc nhàn rỗi:** CPU 33,5 °C, GPU 33,5 °C, PMIC 50 °C; RAM 1,55/3,96 GB, MemAvailable 2,29 GB;
  swap 0 (swapfile 4 GB + zram); thẻ 58 G còn 26 G.
- **Boot:** boot_id `3cf3e6a1-e3a1-4ce6-a0cc-fbe9e548a445`, uptime 27 phút lúc kiểm (khác boot R62
  `f6f8c913…`). Desktop `:0` đang đăng nhập (đo có hiển thị được).
- **Kernel panic:** boot hiện tại không có dòng panic/softlockup/hung_task/RCU stall/I2C timeout; pstore **rỗng**.
  Journal **không lưu qua boot** (chỉ có một boot) ⇒ không xác định được lần khởi động lại trước là tắt bình thường
  hay panic. Lỗi probe imx219 (không gắn camera CSI) và `Host read timeout` lúc 1,6 s khởi động là thông báo boot.
- **Toàn vẹn:** engine `~/nano_run/engine/yolov8n_fp16.engine` SHA256 `c32690cd…` khớp; v16
  `MANIFEST_SHA256.txt` 562/562 đạt; 1.470 JPG `MVI_39031` khớp hash PC.
- **Tiến trình nền:** `update-manager`, `gnome-software`, `packagekitd`, 3 tiến trình `jtop`, `nvpmodel_indicator` —
  CPU thấp lúc kiểm nhưng có thể chiếm CPU/IO bất chợt khi đo. Không tự tắt (thay đổi hệ thống); người dùng quyết định.

**Đánh giá cổng ổn định (tiêu chí đề xuất của kế hoạch mục 4):** đạt về nhiệt, RAM, swap, không panic trong boot
hiện tại, không có tiến trình đo khác. Hai lưu ý: máy mới boot 27 phút; không có log boot trước. Đề xuất: trước lượt
dài 1.470 chạy lượt 300 ảnh và kiểm lại log kernel sau mỗi lượt.

**Lỗi công cụ đã sửa:** lần chạy đầu `tom_tat.json` báo `hung_task`, `i2c.*time` — báo động giả vì bộ tóm tắt khớp
cả dòng lệnh grep được ghi trong file. Đã bỏ dòng `=====`/`$ ...` khỏi phần tìm; lần 02 cho `dau_hieu_panic: []`.
