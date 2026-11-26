# Hậu kiểm21/11 — phân bố thay đổi IDSW theo đoạn

`tracking_uncertainty.json` tái kiểm cộng gộp IDSW/FN/FP của40 video mỗi tập
khớp TrackEval. Bootstrap ghép cặp5000 lần,seed167, lấy40video/tập; tổng80
giữ phân tầng40+40 và dùng cùng draw cho mọi mode. Phân tích hậu kiểm này
không đổi gate, chọn model, ngưỡng hoặc metric. Quality chấm đủ150khung;
FPS dùng125khung sau25khung nóng.

Precise so original trên BDD: tổng **+7 IDSW**,11đoạn tăng/9giảm/20bằng,
max tăng8 và max giảm6. CI95 của tổng delta là **[−19;+35]**.
UA: tổng **0**,5tăng/7giảm/28bằng, max tăng4 và max giảm2;
CI95 **[−12;+14]**.

Không gọi là giữ nguyên mọi ID hoặc chứng minh non-inferiority. Các khoảng
đều chứa0, chưa xác nhận khác biệt ổn định theo cách resampling này. Cũng
không suy tổng0 nghĩa là đầu ra từng video trùng byte. Matched-control và
fast được báo đủ trong JSON, không loại các mode hoặc clip thua.

Giới hạn: roster phát triển đã xem nhiều vòng, không test mù; nhiều so sánh
hậu kiểm chưa hiệu chỉnh family-wise; video có thể chia sẻ góc quay/ngữ
cảnh, chưa bootstrap theo camera. Khoảng này không phải bảo đảm tổngIDSW
trên tậpvideo hoặc tuyến đường mới. Không thay HOTA aggregate bằng trung
bình HOTA từngvideo. Cổng AP7190/rawparity/FPS80 vẫn giữ như báo cáo chính.
