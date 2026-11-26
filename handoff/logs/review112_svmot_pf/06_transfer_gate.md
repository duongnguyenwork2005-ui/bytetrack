# Cổng chuyển miền SVMOT của R112

Ghi khi replay SVMOT đang chạy, trước khi mở điểm gộp SVMOT của ứng viên. Tập gồm đủ 9 chuỗi đã quan sát trong lịch sử dự án, hai cảnh. Cùng detector COCO c25 và cùng cấu hình PF500 K1/K2 motion2 làm đối chứng, chạy lại đối chứng đủ ba seed trong lượt này.

Để coi ứng viên có bằng chứng đủ cho cấu hình thử mới, yêu cầu IDSW trung bình không tăng, HOTA trung bình giảm không quá 0,5 điểm, FN trung bình tăng không quá 0,5%, và không seed nào tăng IDSW so với seed đối chứng cùng chỉ số. Đây là cổng sàng lọc kỹ thuật trên hai cảnh đã xem, không phải chứng minh tổng quát hoặc kiểm định độc lập.

METEOR là pilot clip mới chọn trước metric, dùng để báo kết quả mô tả và kiểm adapter camera xe. Không chọn lại clip hoặc tham số theo điểm pilot, không gọi một clip là benchmark toàn dataset.
