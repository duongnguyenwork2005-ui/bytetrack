# RoundaboutHD – Đổi ID trong đoạn xe đổi hướng

Phân tích hậu nghiệm trên 36 đầu ra ByteTrack của R64. Nguồn dữ liệu gồm bốn camera của cùng một vòng xuyến, 3.000 khung mỗi camera, 15 fps. Phát lại phép ghép CLEAR TrackEval ở IoU 0,5 trên toàn timeline trước khi gắn nhóm chuyển động cho sự kiện đổi ID. Tổng IDSW và FN khớp bảng R64 ở 144/144 cặp camera × lượt. Một khung-xe là một xe GT xuất hiện trong một khung.

## Quy tắc lấy đoạn

Lấy điểm giữa cạnh đáy hộp GT trong các dãy khung liên tiếp của cùng ID. Tại mỗi khung t, dùng trung vị ba cụm 5 khung quanh t−13, t và t+13. Tính góc giữa hai vectơ dịch chuyển, yêu cầu mỗi vectơ dài ít nhất max(8 px, 0,25 chiều cao hộp GT). Gộp ít nhất 8 khung liên tiếp đạt cùng dấu và góc ≥25° thành đoạn đổi hướng. Đối chứng gần thẳng dùng |góc| ≤8° và cùng yêu cầu độ dài tối thiểu. Ngưỡng ≥35° kiểm độ nhạy. Các góc đo trên ảnh, chịu ảnh hưởng của phối cảnh và hình dạng hộp GT. Số đếm ở frame xảy ra IDSW theo CLEAR, còn FN được đếm trên cùng các khung-xe của nhóm.

## Kết quả

- Gộp bốn camera, nhóm ≥25° có 318 đoạn và 6.601 khung-xe. YOLOv8n mixed5 c25 cho IDSW theo thứ tự KF/CV, EKF, UKF, PF500 là 28, 32, 29, 96,33. FN lần lượt là 1.780, 1.782, 1.779, 2.080,33. PF500 là trung bình ba seed.
- Ở cấu hình trên, nhóm gần thẳng có 834 đoạn, 36.002 khung-xe và IDSW 269, 269, 276, 609. Nhóm ≥35° có 239 đoạn, 4.225 khung-xe, IDSW 18, 18, 18, 76,67.
- Ở cả sáu cấu hình detector/ngưỡng, chưa có cấu hình mà EKF và UKF đều có IDSW thấp hơn KF trên nhóm ≥25° gộp bốn camera. YOLOv8n COCO c10 có 21, 20, 21 IDSW cho KF/EKF/UKF.
- Camera 4, YOLOv8n mixed5 c25, có 90 đoạn ≥25° và 1.820 khung-xe. IDSW là 8, 7, 6. FN là 151, 149, 148. Khi tăng ngưỡng góc lên ≥35°, nhóm này còn 59 đoạn và 1.038 khung-xe, IDSW là 5, 4, 4 và FN là 73, 74, 73.
- Camera 3, YOLOv8n COCO c25, có 38 đoạn ≥25° và 887 khung-xe. IDSW là 4, 3, 3. FN là 220, 219, 219. Ở ngưỡng góc ≥35°, 36 đoạn và 696 khung-xe, IDSW là 2, 1, 1 và FN là 165, 164, 164.

## Ví dụ trên video

Ba ví dụ đầu trong ảnh kiểm trực quan. Khung bắt đầu từ 1 theo R64. Đổi ID và FN theo thứ tự KF/CV, EKF, UKF.

- Camera 3, xe GT 95, khung 1573–1601 (104,8–106,7 giây). Xe ôm vòng xuyến. YOLOv8n COCO c25 và c10 đều cho IDSW 1, 0, 0 và FN 0, 0, 0. Góc ảnh cực đại 89,9°.
- Camera 4, xe GT 69, khung 1626–1661 (108,3–110,7 giây). YOLOv8n mixed5 c25 cho IDSW 1, 0, 0 và FN 12, 10, 10. Góc ảnh cực đại 61,0°.
- Camera 4, xe GT 66, khung 1676–1692 (111,7–112,7 giây). Xe tải ôm vòng xuyến. YOLOv8n mixed5 c25 cho IDSW 1, 0, 0 và FN 10, 10, 9. Góc ảnh cực đại 58,3°.

Đây là các trường hợp minh họa được tìm sau khi xem kết quả. Các camera cùng một nút giao và số sự kiện ít. Phép phân tầng cho phép quan sát IDSW khi quỹ đạo đổi hướng trên ảnh. Muốn tách ảnh hưởng của mô hình chuyển động khỏi các yếu tố khác cần đối chiếu sai số hộp dự đoán trước cập nhật trên tập rẽ độc lập.

Mã kiểm tái lập trong audit_turns.py, số liệu từng camera và từng đoạn trong audit.json. Nguồn R64 trong handoff/logs/external_eval/RoundaboutHD.md. Nguồn tác giả https://arxiv.org/abs/2507.08729.
