# HELMET — phạm vi box và khả năng ghép theo IoU

Đã xem ảnh cố định ở cả 12 nhóm (frame gần 50 nhất có GT, hòa chọn frame sớm). Các box nguồn thường bao xe máy và cả người ngồi. Ở cảnh Naypyitaw và một số xe tại Mandalay, box motorcycle của YOLO11n chỉ bao một phần theo chiều cao hoặc bỏ sót đối tượng. Có xe gần biên khung hình, xe khác che ngang và xe ở rất xa. Không diễn giải một lần detector bỏ sót thành sự kiện che khuất đã gán nhãn. Xem `overview_gt.jpg`, `overview_yolo11n_mixed5.jpg` và các overlay theo từng detector.

Trên 3.535 box/1.200 frame smoke, recall ghép một-một tại IoU ≥ 0,5 với ngưỡng conf >0,05 lần lượt là 70,66% (YOLOv8n COCO), 50,98% (YOLOv8n mixed5) và 44,61% (YOLO11n mixed5). Tại conf >0,25, các tỷ lệ lần lượt là 60,03%, 27,69%, 23,96%. Số TP và từng clip ở `recall.json`. Cả ba detector có box đủ khớp để so sánh các bộ lọc theo giao thức dự án.

Khi review kỹ thuật smoke đủ, cho phép chạy test36 với GT xe máy nguồn, cùng ba detector, hai ngưỡng, bốn bộ lọc và clip đã khóa. Chênh lệch điểm tracking sẽ bao gồm cả ảnh hưởng độ phủ detector và sự khác biệt phạm vi box. Phân tích chỉ so bộ lọc dùng chung detector/ngưỡng. Không thay nhãn, cấu hình hoặc chọn lại clip sau smoke. Smoke là 12 trong số 36 clip test, nên chỉ dùng để kiểm kỹ thuật.
