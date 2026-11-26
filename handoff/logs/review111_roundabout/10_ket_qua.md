# RoundaboutHD, kết quả trên 24.000 khung còn lại

Đã suy luận đủ 24.000 khung mới trên bốn camera của cùng vòng xuyến, khung 3.001–9.000 mỗi camera. Dùng chung YOLOv8n COCO c25 cho KF, EKF, UKF và PF500 K1/K2 motion2. Phát lại liền mạch từ khung 1 để giữ lịch sử ghép ở ranh giới. Các dãy số ghi theo thứ tự **KF, EKF, UKF, PF500**. Điểm PF là trung bình ba seed. [Giao thức](00_protocol.md)

## Theo nhóm chuyển động trong phần mới

- **Đổi hướng ≥25°, 13.509 khung-xe:** IDSW **34 / 28 / 28 / 40,33**, FN **5.275 / 5.264 / 5.263 / 5.308,67**. EKF và UKF cùng giảm **17,65% IDSW** so với KF. Trên camera 1–3, mỗi camera giảm **8 → 6**; trên camera 4 cùng có **10**. Ba seed PF lần lượt có **40 / 45 / 36** IDSW.
- **Đổi hướng ≥35°, 8.659 khung-xe:** IDSW **29 / 24 / 24 / 33**, FN **2.543 / 2.539 / 2.539 / 2.573,33**.
- **Gần thẳng ≤8°, 78.372 khung-xe:** IDSW **426 / 423 / 418 / 658,67**.

Góc được tính từ nhãn hộp trên ảnh. Số đổi ID trong các nhóm lấy theo vị trí xảy ra sự kiện sau khi phát lại phép ghép CLEAR trên toàn timeline. Kết quả mô tả bốn góc nhìn của một vòng xuyến.

Mẫu ≥25° gồm **652 đoạn**, phân theo camera 1–4 là **186 / 222 / 68 / 176**. So theo từng đoạn, KF, EKF và UKF bằng nhau về IDSW ở **645 đoạn**. Trong 7 đoạn còn lại, hai bộ lọc phi tuyến cùng giảm IDSW ở 6 đoạn và cùng tăng ở 1 đoạn. Chẳng hạn camera 1, xe GT 371, khung 6.298–6.324, KF/EKF/UKF có 1/0/0 IDSW. Camera 2, xe GT 226, khung 4.665–4.675, ba bộ lọc có 0/1/1 IDSW. [Danh sách đoạn, quỹ đạo và khung xảy ra IDSW](turn_episodes.json)

## Điểm gộp trên toàn cửa sổ mới

- **IDSW:** **716 / 726 / 725 / 1.020,33**.
- **HOTA:** **40,46% / 40,37% / 40,19% / 37,33%**.
- **FN:** **80.557 / 80.496 / 80.551 / 82.077,67**.

Toàn cửa sổ có 157.222 hộp GT. TrackEval khi chấm riêng cửa sổ mới bắt đầu lịch sử ghép tại khung 3.001. Nếu giữ cả lịch sử CLEAR từ phần đầu, IDSW trong cửa sổ mới là **721 / 731 / 730 / 1.025,33**.

## Kiểm chứng

Phát hiện đủ 40 phần cache × 600 khung. YOLOv8n COCO batch 4 khớp cache R64 về lớp, số hộp và dung sai hộp/điểm ở 64/64 khung đối chiếu. Hoàn tất 24 lượt tracking, 216.000 khung phát lại. KF/EKF/UKF khớp đầu ra R64 ở 12/12 lượt camera × bộ lọc trong 3.000 khung đầu, cùng frame và ID, dung sai hộp 0,001 pixel. Tổng IDSW và FN của phép ghép CLEAR liền mạch đối chiếu đúng TrackEval trên đủ 24 lượt. [Kết quả chi tiết theo camera và seed](evaluation/scores.json)
