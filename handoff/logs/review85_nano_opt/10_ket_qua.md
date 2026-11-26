# R85 — kết quả tối ưu Jetson Nano (06/11/2026)

Phạm vi: kiểm các cách khả thi khi giữ nguyên YOLOv8n COCO, engine C1 384×640, tracker và cổng chất lượng/tốc độ của dự án. Không đổi cấu hình luận văn. Nano original/MAXN, 300 JPG MVI_39031, HDMI, CV/EKF; cùng nguồn và engine trong mỗi cặp. Cổng triển khai: FPS trung vị cặp tăng ≥5% ở **cả CV và EKF**, p90 trễ tăng ≤5%, MOT/lớp trùng byte và không vi phạm giám sát. Kế hoạch khóa trước: `00_protocol.md`.

## Kết quả

- **C2 352×576:** engine GPU lịch sử 41,86→37,15 ms so C1, nhưng dev12 HOTA c25 giảm 0,824 điểm CV và 1,142 điểm EKF; recall che nặng giảm 2,66 điểm trên 15.269 GT (cổng tối đa 2). Xe nhỏ giảm 5,17 điểm nhưng chỉ 58 GT. Loại; không mở C3/C4, BDD dev hay Nano pipeline C2. `01_c2_quality.md`.
- **NMS ma trận:** 300 ảnh thật PC cho decode trùng byte; 1.200 lần NMS trên Nano dùng đầu vào từ ảnh thật cho chỉ số trùng. 16/16 lượt HDMI, 4 cặp CV + 4 cặp EKF, đủ 300 frame/lượt, live/replay và MOT/lớp A/B trùng byte, mọi guard sạch (CPU tối đa 50,5 °C; GPU 47,5 °C; swap tăng tối đa 5 MB). Hậu xử lý giảm trung vị khoảng 2,27 ms CV và 2,16 ms EKF. FPS trung vị cặp **+3,65% CV, +2,30% EKF**; p90 trễ −1,64%/−1,23%. Chỉ các cặp lệch decode <1 ms: CV +3,72% (3 cặp), EKF +2,30% (2 cặp). Các cặp lệch decode 9–10 ms giữ trong hồ sơ, không quy FPS cho mã. **Không đạt +5%, không triển khai.** `02_nms_probe.md`, `15_nms_hdmi_summary.json`.
- **CUDA pinned host memory:** microbenchmark tensor input 2,95 MB cho H2D pageable 1,42–3,32 ms, pinned ~0,55 ms; output 1,69 MB cho D2H pageable 0,92–0,93 ms, pinned 0,25–0,26 ms. Nhưng pipeline trực tiếp đọc output pinned làm NMS chậm hơn. Sau khi chép output về RAM thường, hai cặp gần cùng tầng decode cho FPS **−2,85% output-only**, **−2,41% input+output** so v21; infer tăng ~1,7 ms. Cả bốn lượt 300 frame, MOT/lớp trùng byte, guard sạch. **Loại.** `16_cuda_copy_probe.json`, `17_pin_summary.json`.
- **Tiền xử lý CPU:** LUT hiện tại trung vị 1,44 ms/ảnh 384×640; bốn cách bỏ copy uint8, `np.take` tensor/kênh và gán kênh mất 6,11–8,64 ms, dù tensor trùng byte. **Giữ LUT.** `19_preprocess_probe.json`.
- **Giải mã phần cứng MP4:** GStreamer `nvv4l2decoder` chạy được. Sau chuyển ra ảnh BGR NumPy, hai lượt cân bằng 300 frame mất 5,48/5,45 ms/frame; OpenCV phần mềm hiện tại 3,97/3,97 ms/frame. Pixel của hai đường khác nhau ở frame 1/150/300, nên chuyển decoder còn cần đánh giá lại detection. **Loại cho MP4 này.** `18_decode_pairs.log`, `18_decode_guard.json`. Lượt OpenCV nguội 8,67 ms/frame trước đó không dùng so sánh.
- **Các phép đo đã có từ R83/R63:** LUT+giảm copy đã triển khai, tăng median FPS +5,39% CV/+5,12% EKF trong 16 lượt HDMI (R83). Prefetch queue1 đạt 16,05 FPS không hiển thị nhưng p90 ~201 ms so ~83–92 ms của C1, loại cho thời gian thực. TensorRT CUDA graph C1: GPU compute 41,59→41,38 ms, thay đổi <1%, không phải nút thắt. Engine C1 FP16/MAXN đã dùng.

## Quyết định và giới hạn

Không có ứng viên mới đạt đồng thời cổng chất lượng, tốc độ và độ trễ; hai launcher desktop/menu tiếp tục trỏ tới `v21combo/nano_transfer` với engine C1. Cuối lượt swap 0 MB, lỗi ext4 `mmcblk1p1/errors_count=0`, không còn tiến trình thử nghiệm.

Pruning/distillation chưa được huấn luyện/thử: đây là thay đổi mô hình detector, cần tập huấn luyện và quy trình đánh giá mới để kiểm chất lượng xe máy nhỏ/che khuất trước khi so tốc độ. Không có số FPS hay lợi ích chất lượng được gán cho hướng này. C3/C4 chỉ có thời gian engine lịch sử, không có kết quả chất lượng được phép dùng do C2 đã trượt cổng.

Trong đêm Nano đổi địa chỉ Wi-Fi `.5`→`.3`; khóa máy chủ SSH của `.3` khớp chính xác khóa đã lưu cho `.5`. Alias `nano` giờ dùng `jetson-desktop.local` (IPv4, `HostKeyAlias` khóa cũ); bản sao cấu hình cũ ở `C:/Users/Mikuno/.ssh/config.pre_r85_20261106`. Toàn bộ 16 lượt NMS hoàn tất trên cùng boot `de5b2061…`; Nano hiện ở boot khác, pstore rỗng và journal không giữ boot trước nên không gán nguyên nhân cho lần khởi động lại.
