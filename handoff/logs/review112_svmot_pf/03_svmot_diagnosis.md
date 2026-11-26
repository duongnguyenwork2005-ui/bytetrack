# Chẩn đoán SVMOT trên 30 chuỗi

Đã phát lại quy tắc ghép CLEAR của evaluator trên MOT đã có và đối chiếu đúng IDSW/FN từng clip cho KF, EKF, UKF, tổng cộng 90 clip × bộ lọc. Không chạy detector hoặc thay tracker trong chẩn đoán này. Dữ liệu theo khung ở svmot_events.csv, theo xe ở svmot_per_id.csv và tổng hợp ở svmot_diagnosis.json.

## Toàn video và riêng thời điểm che khuất trả lời hai câu hỏi

Trên 21 chuỗi bổ sung, IDSW toàn video KF/EKF/UKF là 1.189/1.162/1.162. Nhóm visibility ≤0,5 ở đúng khung đổi ID là 149/152/156. Phần visibility cao hơn tương ứng 1.040/1.010/1.006. Như vậy mức tăng 3 và 7 trong nhóm đang che cùng xuất hiện với mức giảm 30 và 34 ở phần còn lại. Kết quả toàn video và kết quả của nhóm hiện tại có thể khác chiều mà vẫn nhất quán số học.

Nhóm đang che gồm 428 xe tính theo clip ở 21 chuỗi. Khi so IDSW của từng xe trong nhóm này với KF, EKF tốt hơn ở 10 xe, bằng ở 404 xe và kém hơn ở 14 xe. UKF tốt hơn ở 7 xe, bằng ở 404 xe và kém hơn ở 17 xe. Chênh tổng nhỏ là tổng của nhiều thay đổi trái chiều. Ví dụ seq_02_evening đóng góp chênh UKF +7, còn seq_07_morning đóng góp −5 và seq_21_evening −3. Chưa có căn cứ gọi đây là lỗi lập trình chung của EKF hoặc UKF.

## Nhãn visibility tại một khung chưa mô tả hết một lần tái xuất hiện

Nếu chẩn đoán bổ sung tính sự kiện đổi ID khi GT có visibility ≤0,5 trong một giây gần nhất, IDSW ở 21 chuỗi là 474/454/455. Nhóm ở đúng khung che là 149/152/156. Sự thay đổi nhóm cho thấy phụ thuộc vào thời điểm đo so với sự kiện che. Phép đo một giây là hậu kiểm được chọn sau khi đã biết điểm; không thay bảng benchmark cũ và không dùng để khẳng định ưu thế mới.

Trong nhóm đang che, các IDSW sau ít nhất một khung không được ghép GT là 94/98/100; khi được ghép liên tục là 55/54/56. Đây là lịch sử ghép của bộ chấm, chưa xác định trực tiếp trạng thái Lost nội bộ hoặc nguyên nhân của từng phép gán.

## Độ phủ phát hiện giới hạn phần bộ lọc có thể sửa

Với cùng cache YOLOv8n COCO c25, 12.051 hộp GT khả kiến thấp ở test9 chỉ có 5.908 hộp có ít nhất một detection IoU ≥0,5, tương đương 49,03%. Ở train21, tỷ lệ tương ứng là 10.310/22.221 = 46,40%. Trên nhóm khả kiến cao, độ phủ là 87,90% ở test9 và 77,88% ở train21.

Với KF ở test9, 6.116 trong 6.515 hộp bị bỏ sót trong nhóm khả kiến thấp không có detection đạt điều kiện nói trên, tương đương 93,88%. Trên train21 là 11.863/12.637 = 93,88%. Số FN còn lại có ít nhất một detection đạt IoU 0,5 là 399 và 774. Các bộ lọc khác có tỷ lệ gần tương tự. Độ phủ này không ràng buộc ghép một-một, chỉ là cận trên khả năng cung cấp quan sát; không phải AP hoặc recall detector chính thức.

## Hướng xử lý có căn cứ

Giữ đồng thời điểm toàn video và điểm của sự kiện có GT trước/sau che, đọc IDSW cùng FN và IDF1. Có thể bổ sung phân nhóm tái xuất hiện theo thời gian thực, khóa định nghĩa trước lượt dữ liệu mới. Phía detector cần tăng độ phủ xe che khuất và xe nhỏ trước khi kỳ vọng chỉ thay bộ lọc khôi phục phần lớn mẫu mất. Với PF, kiểm riêng cách suy luận động học khi có phép đo và sai số dự đoán sau gap, đồng thời giữ cùng cache để xác định ảnh hưởng của bộ lọc.
