# R166 — phân nhóm lại toàn bộ kết quả BDD200/UA40 đã có

Người dùng21/11 muốn đánh giá các kịch bản phi tuyến/che khuất. Đây chỉ là
diagnostic từ kết quả R148/R154/R142 đã hoàn tất, không chạy tracker mới,
không mở scenario transfer khi Gaussian full prerequisite chưa đạt, không
chọn video hoặc đổi threshold dựa theo thắng/thua. Không sửa luận văn/default.

Dùng toàn bộ200 BDD và40 UA trong manifest full đã khóa. Báo KF, EKF/UKF
commonR R148, EKF/UKF noise-floor R154 và PF R142 ba seed, không chọn Gaussian
yếu hơn làm reference. Không chạy PF R165 trên full từ kết quả dev.

Phân nhóm GT dùng nguyên labels_for_camera R99 và trace CLEAR đã kiểm ở
R113: rẽ sustained >=25 độ trên ảnh, near-straight<=8 độ, đủ chuyển động,
support hai giây và smoothing bằng median. Không suy từ đó quỹ đạo thế giới
phi tuyến, nhất là camera gắn xe BDD. Chỉ với UA, native visibility<=.5 và
vừa che trong một giây; giữ nhóm recovery và turn25×low. BDD thiếu cờ che
gốc nên không dùng visibility=1 như bằng chứng không che. BDD báo cả car,
motorcycle và tất cả vehicle theo vehicle_types.csv đã có; UA vehicle tổng.

Mỗi output trace phải khớp IDSW/FN của raw CLEAR đã lưu, kiểm GT ID mapping,
hash MOT/GT/raw/source trước/sau. Báo GT/FN/IDSW và delta cùng FN; không nói
ít IDSW là tốt khi bỏ sót nhiều. Giữ toàn bộ nhóm cả thắng lẫn thua, không
thay metric chính/HOTA hoặc suy HOTA nhóm bằng cộng trừ. Bootstrap paired
theo video 5000 lần,seed166, CI mô tả hậu nghiệm, không là test mù hay đã
hiệu chỉnh đa kiểm định. PF mean của3seed, không lựa seed tốt.

Tối đa3 logical Python job. Không đo FPS PC khi recovery detector đang chạy.
