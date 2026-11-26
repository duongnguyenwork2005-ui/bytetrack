# R123 — PF100 giữ chất lượng, giảm 60,19% chi phí cập nhật PC

## Thay đổi và giới hạn

Giữ Cartesian coordinated-turn RBPF của R118: vị trí/vận tốc Cartesian cập nhật Gaussian có điều kiện, aspect/height Gaussian dùng chung, chỉ lấy mẫu omega. [Marginalization trong PF](https://arxiv.org/abs/1301.3853) là cơ sở giảm số biến cần lấy mẫu; mô hình Q ở đây vẫn là xấp xỉ theo chiều cao trung bình như R118. R123 **chỉ đổi số hạt 500→100**, không sửa Q/R, init hai quan sát hoặc association. Ít hạt không bảo đảm chính xác hơn; phép thử kiểm xem có giữ chất lượng với chi phí thấp hơn không.

Giao thức lưu trước điểm ở `00_protocol.md`. Cổng dev IDSW ≤102% nền, HOTA giảm không quá 0,25 điểm, FN tăng không quá 0,5%. Đạt mới chuyển SVMOT, giữ nguyên 100 hạt, không dò thêm N. Hai tập đã được dự án dùng phân tích từ trước, không gọi là test mù hoặc bằng chứng ưu thế tổng quát. Không đổi backend mặc định, launcher, detector hoặc khóa luận.

## Chất lượng UA-DETRAC dev12

PF500 Cartesian R118 → PF100, trung bình ba seed: IDSW **464,67→458 (−1,43%)**, HOTA **58,995→59,020%**, IDF1 **71,883→71,932%**, FN **37.450,67→37.429,33**, FP **12.097→12.087**. Seed IDSW100: **456/461/457**. Bootstrap ghép cặp 12 video, 10.000 mẫu cho chênh IDSW **[−12,−1]**, vẫn là dev đã dùng nhiều vòng.

Chạy lại KF/EKF/UKF đủ 36 lượt trên đúng cache/GT/buffer/dt: IDSW **489/472/474**, HOTA **59,000/59,252/58,786%**, FN **37.564/37.448/37.458**. Mọi metric trùng R70 đã lưu. PF100 ít IDSW theo điểm ước lượng, HOTA gần KF, cao hơn UKF và thấp hơn EKF. CI chênh IDSW PF100−KF **[−64,67,+2,67]**, so EKF **[−40,+12,33]**, so UKF **[−40,+11,33]**; đều còn chứa 0. Không khẳng định thắng chắc chắn các bộ lọc trên tổng thể.

So PF cổ điển K1/K2 motion2 của R112 cùng dev có 697 IDSW, mức mới 458 giảm 34,29%. Phần lớn cải thiện này đã có từ RBPF R112; **không quy toàn bộ 34,29% cho việc giảm hạt**. Bản R112 RBPF500 có 469 IDSW.

## Chuyển miền SVMOT test9

Giữ nguyên tham số, 27 lượt mới, so kết quả Cartesian500 R118 cùng cache đã khóa: IDSW **435,33→435**, HOTA **50,772→50,902%**, IDF1 **53,306→53,507%**, FN **14.615→14.613,33**, FP **22.259,33→22.245,67**. Seed100 IDSW **432/439/434**. Chênh IDSW gần 0, không mô tả thành cải thiện lớn.

Native visibility ≤0,5 có 12.051 hộp, IDSW **132,33→132**, FN **6.478→6.478**; CLEAR toàn 27 clip×seed được tái ghép và khớp. Đây là duy trì chất lượng nhóm che, chưa phải bằng chứng giải quyết che khuất. Cổng chất lượng hai tập đạt.

So với **PF cổ điển** K1/K2 motion2 cùng cache trong R112, SVMOT cũ có IDSW 434 và HOTA 50,275%; bản mới 435 và 50,902%. Nghĩa là HOTA tăng nhưng IDSW không thấp hơn bản cổ điển trên tập này. Không dùng kết quả dev UA để tuyên bố mọi tập đều giảm IDSW.

## Tốc độ: chỉ PC, không phải FPS Nano

Sau khi mọi job chất lượng/GPU kết thúc, đo tuần tự cùng process, hai prefix 512 của MVI_20052/20062, seed 0, bỏ 32 khung đầu, ba lần đảo thứ tự, 12 lượt. N500/N100 có thời gian trung bình **30,810/12,264 ms/update**, giảm **60,19%**. Cả sáu cặp đều giảm, khoảng **49,66–68,45%**. Thời gian tuyệt đối còn biến thiên theo lần và cảnh; đây chỉ là hai prefix, chưa phải toàn bộ video hoặc 80 video Nano.

Giữ cùng detector và association, chỉ giảm lượng tính toán của phần particle. Không suy ra FPS toàn hệ thống từ nghịch đảo thời gian tracker, không cộng 60,19% với 19,67% của R118 ở phiên đo khác. **Chưa đo Nano** và chưa kiểm PF100 trên HELMET/RoundaboutHD/BDD.

## Kiểm chứng và sử dụng

Smoke500 khớp R118 6/6, PF100 cùng seed lặp 6/6. Hậu kiểm **99 lượt chất lượng** (36 dev, 27 SVMOT, 36 đối chứng), 9 nhóm raw metric, 105 hash đóng băng, MOT/marker/tổng hợp, 12 lượt timing và lặp MOT đạt. `final_verification.json`, `serial_timing.json`, `reference_summary.json`, `occlusion_summary.json`.

Đây là ứng viên nghiên cứu riêng. YAML100 ở `rbpf100.yaml`; runner dùng context `installed()` từ `review118_pf_shape/cartesian.py`, khôi phục hook sau khi chạy. Chưa đặt mặc định sản phẩm và chưa đưa monkeypatch nghiên cứu vào launcher. R124 đã kiểm đủ KITTI20: PF100 có 247,67 IDSW so KF 280 nhưng HOTA 49,487% so 49,764% và FN tăng 0,65%; không tốt hơn toàn diện. Xem `../review124_pf_kitti/10_ket_qua.md`. Bước tiếp là kiểm Nano trước khi nói FPS và cân nhắc backend riêng, không tự sửa luận văn từ kết quả này.
