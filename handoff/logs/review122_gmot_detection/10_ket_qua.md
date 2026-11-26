# R122 — detector chưa đủ quan sát xe rẽ GMOT

Giữ weights YOLOv8n COCO, thử hai phương án đã ghi trước: full1280rect và chia ảnh thành tám ô 640×640, stride512 với ô bám biên, merge NMS theo lớp IoU0,7. Đủ 228 ảnh car-0 mỗi phương án; c25/max300 như tracker. Không đo/tuyên bố FPS Nano từ thử nghiệm này.

Trong nhóm rẽ ≥25° theo GT, giả định cadence24/30 Hz:

- Full640 gốc: candidate IoU≥0,5 ở c25 là **1/1.369 (0,073%) / 2/1.521 (0,131%)**.
- Full1280: **47/1.369 (3,433%) / 47/1.521 (3,090%)**.
- Sliced640: **43/1.369 (3,141%) / 53/1.521 (3,485%)**.

Ở c05, full1280 đạt **11,91 / 12,62%**, sliced640 **13,66 / 14,20%**. Đây là cận trên candidate không ghép một-một, **không phải AP hoặc recall detector chuẩn**. Không dùng số đó thay kết quả detection model trên tập đã khóa.

Cả hai trượt cổng25% c25 đã ghi trước, vì vậy **không mở tracking** trên các cache mới, không dò thêm conf/kích thước. Hai cách có tăng candidate nhưng chưa giải quyết được chuyển miền đêm/xe rất nhỏ đủ để phân biệt bộ lọc. Chưa kiểm từng false positive/GT để kết luận mọi thiếu ghép đều chỉ do kích thước; các cách này không được chọn để triển khai Nano.

`coverage.json` giữ đầy đủ từng giả định/ngưỡng, `verification.json` PASS 7 hash nguồn và 3 cache, `selected=null`. Hai cache trên F giữ nguyên để chẩn đoán về sau; chưa thay benchmark hay detector mặc định.
