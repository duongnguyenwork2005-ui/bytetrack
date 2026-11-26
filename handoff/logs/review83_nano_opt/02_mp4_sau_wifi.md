# R83 tiếp tục: MP4 và Wi-Fi khi rút LAN — 05/11/2026

Nguồn chung: `~/nano_run/v16/nano_transfer/data/smoke/MVI_39031_f0001-0300.mp4`, MPEG-4 960×540, 25 fps, 300 frame. So C1 (`v17c`) và LUT+giảm copy (`v21combo`) với cùng engine C1 384×640, CV/c25, không hiển thị. Lượt cũ 04/11 bị người dùng yêu cầu dừng không dùng. Hai cặp mới hoàn tất 05/11: A→B và B→A, output riêng `~/nano_run/r63_20261030/r83_mp4_{base,combo}_20261105{,b}`.

- A→B: C1 14,267 FPS, combo 16,512 FPS (+15,7%); decode median 3,46/3,47 ms; p90 trễ 78,24/73,60 ms.
- B→A: combo 16,471 FPS, C1 15,203 FPS (+8,3%); decode median 3,44/3,48 ms; p90 trễ 73,39/78,18 ms.
- Cả bốn lượt đủ 300 frame. Tệp MOT và lớp C1/combo trùng byte trong từng cặp; lỗi ext4 `mmcblk1p1/errors_count=0`.

Sau khi người dùng rút LAN giữa hai cặp, `eth0:unavailable`, `wlan0:connected`, IP Wi-Fi `192.168.1.5`; SSH qua alias `nano` vẫn được. Cặp B→A hoàn tất qua Wi-Fi. Không suy phần trăm này cho mọi video/codec; bằng chứng mạnh hơn cho JPG+HDMI là ma trận 16 lượt R83, tăng median +5,39% CV/+5,12% EKF.

USB Wi-Fi đã sửa trong R84. OpenCV hệ thống trên Nano báo `GStreamer: NO` dù plugin `nvv4l2decoder` có mặt. Với MP4 này, decode hiện ~3,5 ms/frame trong khi infer ~45 ms/frame; tích hợp decoder phần cứng cần đường đọc mới và chưa được kiểm, không cộng lợi ích giả định. C2–C4 đã có timing engine lịch sử nhưng chưa qua cổng chất lượng xe nhỏ/che khuất; pruning/distillation cần model huấn luyện lại. Launcher vẫn `v21combo`.
