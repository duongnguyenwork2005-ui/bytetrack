# R96 — PF500 trên Nano, pilot riêng

Ngày 10/11/2026. Theo yêu cầu người dùng, đo riêng PF500 vì R86 chỉ đo CV/EKF. Dùng gói v21combo+C1 và đúng ảnh mẫu BDD/UA R86 đã kiểm hash. Mỗi tập lấy video đầu tiên trong manifest R86, 150 JPEG liên tiếp, BDD 5 fps/steps/lớp [1,2,3,5,7], UA 25 fps/skip/lớp [2,3,5,7] và ignored regions. So CV và PF500 mặc định cùng detector, thứ tự BDD CV→PF, UA PF→CV. Cùng buffer 1,2 s, warmup 25, tắt hiển thị, lưu riêng ngoài launcher. Đây là pilot tốc độ, không chấm chất lượng hay nhận là benchmark nhiều video.

Trước/sau mỗi lượt: kiểm boot ID, `errors_count`, swap tăng không quá 100 MB; bỏ lượt nếu frame thiếu/hỏng, reboot, lỗi thẻ hoặc nhiệt ≥75 °C. Báo FPS ổn định, latency p90, thời gian inference và tracker; không suy kết quả R86 cho PF. Không đổi cấu hình mặc định, engine hay launcher. Nếu pilot ổn mới quyết định mở đo nhiều video có cân bằng thứ tự.
