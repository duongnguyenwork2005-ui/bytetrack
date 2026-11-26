# Đối chiếu khởi tạo vận tốc trước khi đọc điểm dev

Với vận tốc Gaussian Cartesian, quan sát vị trí liên tiếp đã cập nhật vận tốc qua tương quan vị trí–vận tốc. Việc đặt lại vận tốc từ hai detection là một quyết định bổ sung, cần đối chiếu với học vận tốc hoàn toàn trong bộ lọc. Thêm biến thể automatic, giữ cùng số hạt, Q/R, seed, ghép và các kiểm tra/cổng của giao thức chính. Khác biệt duy nhất là khi tracker hỏi khởi tạo chuyển động, giữ posterior đã cập nhật và đánh dấu khởi tạo hoàn tất. Trạng thái vận tốc không cần hướng xác định để theo dõi xe đứng yên.

Thêm biến thể khi lượt dev đầu đang chạy, trước khi đọc các metric dev của vòng này. Chọn trong các bản đạt cổng trên dev theo IDSW thấp nhất để mở phép kiểm chuyển miền, không dò tham số.
