# R86 — đo tốc độ Nano trên nhiều video BDD và UA-DETRAC

Yêu cầu người dùng 06/11/2026: kiểm mức tăng tốc của bản Nano đang dùng so baseline đầu tiên trên hai tập video, không suy từ một video.

- Baseline A: gói v16 nguyên bản trong `r63_20261030/baseline/nano_transfer`, engine 640×640 build trên board hiện tại (`9fae6b43…`). Bản B: `v21combo/nano_transfer`, engine C1 384×640 (`a6912393…`). Không thay mã tracking, metric, cấu hình luận văn hay launcher.
- Nguồn BDD: 40/200 video val, lấy video thứ 3 rồi cách 5 video trong danh sách manifest đã khóa, không chọn theo FPS/kết quả. Nguồn UA: toàn bộ 40 video test trong seqmap đã khóa. Mỗi video lấy 150 frame liên tiếp ở giữa theo `seqLength`, giữ nguyên JPEG nguồn; không nội suy, mã hóa lại hay đổi độ phân giải. Kiểm đủ ảnh và hash archive. Đây là **mẫu thời gian**, không phải toàn bộ frame của hai tập.
- Mỗi video chạy A/B × CV/EKF, cùng JPEG, FPS 5 BDD hoặc 25 UA, lớp [1,2,3,5,7] BDD hoặc [2,3,5,7] UA, c25, buffer 1,2 s, và tô xám ignored regions UA. Không bật HDMI; mọi phép so dùng cùng chế độ. Không ghi detection cache. Warmup 25 frame; báo cả `fps_xu_ly`, `fps_on_dinh`, p90 latency, median decode/pre/infer/post/update và thời gian toàn lượt gồm khởi tạo. Chỉ số chính là FPS xử lý; thời gian toàn lượt là phụ vì 150 frame/video làm chi phí khởi tạo đáng kể.
- Đảo thứ tự A/B theo vị trí video; một video hoàn tất 4 lượt trước khi sang video tiếp theo. Theo dõi boot, swap, lỗi thẻ, nhiệt độ, full frame và exit code. Dừng nếu guard vi phạm hoặc máy reboot. Không dùng FPS chưa hoàn tất.
- Thống kê riêng từng tập và tracker: trung vị chênh FPS ghép cặp theo video, tỷ lệ video B nhanh hơn, bootstrap video CI95 (5.000 vòng, seed 0). Báo thêm phân bố 10–90%, tổng frame và cảnh có decode lệch lớn. Không dùng bootstrap frame như thể độc lập. Không nhân hai tỷ lệ từ những đợt khác nhau.
- Chạy pilot 2 video/tập trước; nếu ổn mới chạy phần còn lại. Không triển khai ứng viên mới sau phép đo này.
