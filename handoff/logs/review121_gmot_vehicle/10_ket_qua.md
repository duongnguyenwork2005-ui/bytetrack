# R121 — GMOT ô tô: pilot, không phải benchmark đã xác minh FPS

Release v0.1 của [tác giả](https://spritea.github.io/GMOT40/download.html), nhãn ID thủ công, source CRC và SHA ảnh đã kiểm. Bốn chuỗi car tải đầy đủ; duyệt ảnh trước khi chấm, loại car-3 đua xe. Ba cảnh car-0 vòng xuyến đêm, car-1 đường đêm, car-2 đường tuyết có **1.027 ảnh / 22.301 hộp / 145 ID**. Nhãn không có motorcycle hay cờ che khuất. ZIP gốc không có timestamp/video/seqinfo. [Bài báo](https://arxiv.org/abs/2011.11858) chỉ công bố toàn tập 24–30 FPS.

Do đó chỉ chạy **18 lượt thăm dò** ở hai giả định đầu mút 24/30 Hz, không nhận 30 FPS là cadence thật. Dùng COCO640 c25 và KF/EKF/UKF; gộp detection car/bus/truck để theo taxonomy phương tiện bốn bánh của pilot. Đây không phải one-shot GMOT protocol của tác giả. Mỗi lượt CLEAR IDSW/FN tái ghép khớp, 18 MOT và hash nguồn/cache/GT đạt, xem `verification.json`.

Hai giả định cùng có IDSW **22 / 20 / 21**, HOTA **16,533 / 16,490 / 16,421%**, FN **20.073 / 20.070 / 20.072**. Không có khác biệt tổng ở hai FPS trong phạm vi ngắn này; không suy ra FPS đã đúng.

Riêng car-0, KF/EKF/UKF có IDSW **11 / 12 / 12**, HOTA **3,858 / 3,728 / 3,733%**. Nhóm rẽ ở 24 Hz có 1.369 hộp và ở 30 Hz 1.521 hộp, cả ba bỏ sót **toàn bộ**, IDSW=0 không thể dùng chứng minh tracking tốt. Car-2 không có track nào, HOTA=0 và FN=3.492 cho cả ba. Car-1 có IDSW 11/8/9 nhưng hầu như không có rẽ theo tiêu chí GT, không coi đó là thắng ở quỹ đạo cong.

**Dừng kết luận về bộ lọc trên GMOT.** R122 thử tăng chi tiết detector trước khi cân nhắc chạy tiếp. Giữ mọi điểm không thuận, không dùng điểm tổng 22/20/21 để quảng bá lợi thế trên xe rẽ. Dữ liệu đã được dùng để chẩn đoán, không còn là test mù.
