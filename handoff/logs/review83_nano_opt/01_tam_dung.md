# R83 — trạng thái khi người dùng yêu cầu tạm dừng (04/11/2026)

Đã dừng loạt MP4 giữa lượt baseline bằng TERM sau khi người dùng nhắn “tạm dừng”. Kiểm `ps` không còn `run_mp4_screen.sh`, `nano_live.py` hoặc `nano_detect_stream.py` của phép thử; không khởi động lượt combo MP4. Thư mục MP4 có thể chứa đầu ra dở và **không dùng để kết luận**. Bộ đếm lỗi thẻ `/sys/fs/ext4/mmcblk1p1/errors_count=0`.

Kết quả đã hoàn tất trước khi dừng:

- R82 copy đơn: đầu ra trùng, chưa chứng minh tăng FPS toàn luồng; không chọn riêng.
- R83 prefetch queue 2/1: CV 15,25/16,05 FPS (300 JPG, tắt hiển thị), nhưng p90 trễ ~265/201 ms so C1 ~83–92 ms; không triển khai chế độ này.
- LUT: tensor PC trùng byte; 300 JPG Nano và MOT/replay đạt. Cùng ảnh RAM, C1 13,67/13,78 FPS, LUT 14,64, LUT+copy 14,74; p90 tương ứng C1 ~83 ms, combo ~78 ms.
- LUT+copy SD/HDMI: 4 cặp mỗi bộ lọc, 16/16 lượt đủ 300 frame, MOT/lớp A/B trùng byte, không vi phạm giám sát. Trung vị hiệu FPS theo cặp CV +5,389%, EKF +5,122%; trung vị đổi p90 CV −4,160%, EKF −3,852%. Bằng chứng Nano `~/nano_run/r63_20261030/r83_c1_combo_hdmi_20261104/summary.json`; script kiểm `analyze_hdmi.py`.
- Cặp CV đầu lệch decode ~8,24 ms có lợi cho combo; ba cặp CV còn lại decode lệch ≤0,081 ms, hiệu FPS +5,419%, +5,360%, +4,974%. Không quy toàn bộ chênh cặp đầu cho mã.

Launcher desktop/menu hiện trỏ tới `~/nano_run/r63_20261030/v21combo/nano_transfer`; `07_giao_dien.sh --kiem-tra` đạt, engine và ONNX C1 đúng. Sao lưu launcher C1 và v16 ở cạnh tệp gốc với đuôi `.c1_backup_20261104` và `.v16_backup_20261104`. Gói C1 không sửa.

Đã kiểm tra plugin `nvv4l2decoder` có trên Nano; OpenCV hệ thống 3.2.0 được build với `GStreamer: NO`. MP4 300 frame MPEG-4 có trong gói v16 nhưng phép đo toàn luồng bị dừng, nên chưa có kết luận tăng tốc giải mã phần cứng. C2–C4 chỉ có số GPU lịch sử, chưa qua cổng chất lượng; pruning/distillation chưa có model để đo.

Khi được yêu cầu tiếp tục: hoàn tất benchmark MP4 bằng cùng tệp nguồn, rồi cân nhắc adapter giải mã phần cứng bên ngoài OpenCV; nếu thử C2–C4 thì khóa cổng chất lượng xe nhỏ/che khuất trước khi đo triển khai. Không tự chạy tiếp lúc đang tạm dừng.
