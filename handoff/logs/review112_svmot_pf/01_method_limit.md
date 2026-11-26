# Diễn giải và giới hạn phương pháp R112

RBPF lấy mẫu hướng và tốc độ quay, còn bảy tọa độ khác dùng Gaussian điều kiện. Vì nhiễu quá trình gốc phụ thuộc chiều cao, ứng viên tính Q tại chiều cao trung bình của từng Gaussian điều kiện. Đây là phép xấp xỉ bổ sung so với PF gốc tính Q tại từng mẫu chiều cao. Nhánh tuyến tính được kiểm chính xác bằng bộ lọc Kalman tham chiếu khi các góc đã biết và dùng cùng quy tắc Q đóng băng. Không suy ra từ kiểm thử này rằng hai mô hình hậu nghiệm đầy đủ tương đương.

Ứng viên cũng bỏ roughening K2, giữ nhiễu góc của mô hình, thay lấy mẫu vận tốc bằng cập nhật hiệp phương sai có điều kiện. Phép thử so hai triển khai hoàn chỉnh. Chưa tách tác động từng thành phần và chưa xác lập hiệu quả Nano.

Chẩn đoán SVMOT ban đầu dùng nhầm định dạng hộp detection xyxy thành xywh khi tính độ phủ. Các số IDSW/FN và phân loại lịch sử không dùng phép tính này, đã khớp evaluator. Đã lưu các tệp đầu với hậu tố .invalid-coverage-backup, sửa đổi định dạng hộp và chạy lại toàn bộ chẩn đoán. Chỉ sử dụng các tệp không có hậu tố này sau khi lượt sửa hoàn tất. Giữ log lỗi KeyError ban đầu do đường seqinfo train khác root mặc định và log chạy trước sửa định dạng để truy vết.

Lượt Student-t đầu gặp lỗi lưu metadata vì writer dùng đường rbpf.py dưới thư mục robust. Đã chuyển các kết quả chưa hoàn tất vào robust/failed_save_attempt, giữ log dev.log và inputs_before_save_fix.json, sửa lời gọi writer rồi tạo manifest mới và chạy lại đủ 36 đơn vị. Kết quả hợp lệ dùng dev_corrected.log, evaluation.log, dev_gate.json và integrity.log trong robust, không ghép kết quả từ lượt lỗi.

Student-t cố định ν=4 giảm IDSW dev từ 697 xuống 617 nhưng FN tăng 6,9339%, trượt cổng đã khóa. Không chạy chuyển miền SVMOT cho biến thể này và không dò thêm ν. RBPF Gaussian đạt dev nhưng trượt cổng IDSW SVMOT. Chỉ tích hợp Gaussian thành tên nghiên cứu riêng, không lựa chọn làm mặc định toàn miền.

Các phép kiểm tích hợp đầu dùng sai giả định registry NumPy đã công khai PF cũ và truyền đối số frame_rate không có trong constructor phiên bản hiện tại. Sửa harness dùng registry cục bộ cho hồi quy PF cũ và gọi make_tracker chuẩn. Giữ log lỗi, chỉ dùng registered_np_final.log và registered_ultralytics_corrected.log làm bằng chứng nghiệm thu.

Việc chặn chiều cao dương được giữ tại trung bình Gaussian điều kiện. Cùng với Q đóng băng theo chiều cao trung bình, đây vẫn là suy luận xấp xỉ. Phép kiểm Kalman ở vùng chiều cao dương không xác lập tương đương toàn mô hình phi tuyến, đặc biệt khi phân phối gần biên chiều cao bằng không.
