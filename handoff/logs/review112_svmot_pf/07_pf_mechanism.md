# Vì sao PF hiện tại chưa mạnh

## Những điểm đã kiểm được

PF gốc lấy mẫu toàn bộ chín tọa độ CTRV. Quan sát chỉ chứa tâm và kích thước hộp cx, cy, a, h. K1 cập nhật tối ưu bốn tọa độ quan sát theo từng hạt nhưng không lưu một Gaussian có điều kiện cho vận tốc, kích thước và toàn bộ tương quan của chúng qua nhiều khung. Khi có đo, thông tin sửa vận tốc chủ yếu truyền qua trọng số và sự phân tán của các hạt. Đây là khác biệt thực trong mã, không phải kết luận rằng công thức K1 bị viết sai.

Với 500 hạt, sai số lấy mẫu có thể ảnh hưởng tâm dự đoán và vận tốc. Bộ ghép ByteTrack quyết định rời rạc theo hộp dự đoán, vì vậy một thay đổi nhỏ cũng có thể dẫn tới ghép khác ID. Nhiễu bổ sung K2 giúp duy trì đa dạng nhưng cũng có thể tăng độ phân tán. Tăng khả năng biểu diễn phân phối không bảo đảm kết quả tốt hơn KF khi dữ liệu quan sát gần tuyến tính và Gaussian.

R112 thay cách suy luận bằng RBPF. Chỉ lấy mẫu hướng và tốc độ quay, còn bảy tọa độ dùng trung bình và hiệp phương sai Gaussian riêng cho từng hạt. Quan sát sửa được vận tốc thông qua tương quan vị trí–vận tốc tích lũy trong bước dự đoán. Phần này có kiểm đối chiếu Kalman khi cố định góc, covariance Joseph, đối xứng/PSD, góc qua biên ±π, khoảng mất đo và lặp cùng seed.

Trên 12 video train dev UA-DETRAC, cùng cache và ba seed, IDSW trung bình giảm 697 xuống 469, HOTA tăng 57,0057% lên 58,9635%. Các đối chứng KF/EKF/UKF lần lượt có IDSW 489/472/474. Kết quả cho thấy có thể cải thiện triển khai PF ở cùng số hạt, không cần thay detector hoặc nới metric. Nó chưa tách riêng tác động giảm chiều lấy mẫu, giữ tương quan và bỏ roughening.

## Những điều chưa chứng minh

Trên SVMOT test9 đã xem trước đây, RBPF Gaussian tăng HOTA 50,2753% lên 51,0145% nhưng IDSW tăng 434 lên 440. Vì vậy chưa có căn cứ dùng một cấu hình thay cho mọi miền. SVMOT có nhiều quan sát thiếu hoặc méo do che khuất, trong khi phép cập nhật Gaussian vẫn giả định quan sát theo mô hình đã đặt. Biến thể Student-t bậc tự do 4 được thử riêng để kiểm hướng chống ngoại lai, kết quả và cổng lựa chọn nằm trong robust/.

Chưa có bằng chứng xác định một lỗi lập trình đơn lẻ giải thích toàn bộ bất lợi của PF. Chưa có phép tách nhân tố hoàn chỉnh, dữ liệu test độc lập mới hoặc phép đo RBPF trên Jetson Nano. Không quy mọi IDSW cho bộ lọc vì detector và association đều có thể quyết định giới hạn kết quả.

## Nguồn phương pháp

- Doucet và cộng sự, Rao-Blackwellised Particle Filtering for Dynamic Bayesian Networks, https://arxiv.org/abs/1301.3853.
- Roth và cộng sự, Robust Bayesian Filtering and Smoothing Using Student's t Distribution, https://arxiv.org/abs/1703.02428. Biến thể R112 dùng xấp xỉ riêng được ghi trong giao thức, không tự nhận là bản sao đầy đủ thuật toán của bài báo.
