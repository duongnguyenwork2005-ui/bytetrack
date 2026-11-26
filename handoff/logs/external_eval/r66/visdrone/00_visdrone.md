# R66 VisDrone2019-MOT — smoke detector XONG (qua cổng khả thi); FPS CHƯA XÁC MINH ⇒ chưa chạy full (02/11/2026)

> **Cập nhật R68 03/11/2026:** Đã chạy lại toàn bộ smoke sau khi sửa phép ignore theo [dropObjects.m chính thức](https://raw.githubusercontent.com/VisDrone/VisDrone2018-MOT-toolkit/master/eval/dropObjects.m). File `smoke_det/ket_qua_v2.json` có toàn bộ detector fields và kết quả cổng **trùng** bản R66 bên dưới; không còn đánh dấu recall là tạm thời. Helper `src/r66_ngoai/visdrone_ignore.py` qua 5 fixture, gồm 100 ca đối chiếu đếm pixel. Vẫn chưa có FPS/stride hiệu dụng, vì vậy chưa mở tracking full. Xem mục 0 `handoff/PLAN_DA_CHOT.md` cho trạng thái hiện hành.

**Giao thức:** GPT Q3 (`D:/LABOTARY/automation/runs_local/hoi_codex_R66_dataset_moi_pf_20261102/answer.json`).
**Mã:** `src/r66_ngoai/visdrone_smoke_det.py`. **Kết quả:** `smoke_det/ket_qua.json`; log `../visdrone_smoke_det.log`.

## Mẫu và quy tắc (khóa trước khi chạy)

- **Mẫu:** VisDrone2019-MOT-val, 7 chuỗi. Mỗi chuỗi lấy 10 frame đầu, 10 frame giữa, 10 frame cuối, tổng **210
  frame**.
- **Detector:** 3 detector R54, cùng spec cache R64 (imgsz 640 rect, conf 0,05, iou 0,7, max_det 3000, FP32). mixed5
  được ánh xạ sang ID COCO.
- **Lớp:**
  - GT hợp lệ: score = 1 và category thuộc vehicle4 {4 car, 5 van, 6 truck, 9 bus}.
  - Dự đoán: COCO {2, 5, 7}.
- **Vùng bỏ qua:** hợp các box category 0 (ignored regions) và 11 (others) theo từng frame. Box GT và box dự đoán có
  ≥ 50 % diện tích nằm trong vùng này thì bị loại (quy tắc `dropObjects`).
- **Cách tính recall:** ghép một-một IoU ≥ 0,5.

## Kết quả

Tổng **3.216 GT hợp lệ**. Không frame nào có số box dự đoán chạm trần 3000.

| Detector | Recall conf > 0,05 | > 0,10 | > 0,25 |
|:--|--:|--:|--:|
| yolov8n_coco | 52,4 % | 43,7 % | 31,9 % |
| yolov8n_mixed5 | 41,4 % | 38,5 % | 32,7 % |
| yolo11n_mixed5 | 44,5 % | 41,4 % | 35,2 % |

Chi tiết (kích thước box GT/TP, tỉ lệ frame có TP) ở `ket_qua.json`.

**Cổng khả thi: ĐẠT.** Có ≥ 100 GT, và cả 3 detector đều có recall ≥ 1 % ở conf > 0,05. Như vậy VisDrone **không**
giống UTUAV: detector bắt được khoảng 1/3 đến 1/2 số xe.

## Xác minh FPS / stride (giới hạn 30 phút): CHƯA XÁC MINH

**Không tìm thấy câu nào về FPS gốc hay bước lấy mẫu** trong các nguồn đã đọc:

| Nguồn | Kết quả |
|:--|:--|
| arXiv 2001.06303 (TPAMI 2021), bản ar5iv đọc nguyên văn mục VID và MOT | Chỉ có số clip và số frame; "fps" duy nhất là tốc độ detector |
| README dataset và toolkit MOT trên GitHub | Không ghi FPS |
| arXiv 1804.07437 | PDF không đọc được |
| aiskyeye.com | Không ghi FPS |
| Bài kết quả VisDrone-MOT2019 (ICCVW) | **HTTP 403** |

## Quyết định áp dụng (Q3)

**Bổ sung nguồn tác giả, 02/11:** [FAQ chính thức của VisDrone](https://aiskyeye.com/faq/) có câu trả lời của Visdrone Team ngày 06/06/2020: tốc độ video gốc là **24 fps**, nhưng khi gán nhãn họ **trích một phần frame**. FAQ không ghi quy luật lấy mẫu, FPS hiệu dụng, hoặc quy luật riêng của bản MOT2019 val/test-dev. Vì vậy con số 24 không đủ để tính `time_s` và buffer 1,2 giây trên chuỗi ảnh phát hành; cổng FPS/stride vẫn **CHƯA ĐẠT**.

- "Chưa xác minh được thì chỉ đóng smoke detection, chưa chạy full dù recall tốt."
- ⇒ **Không chạy tracking VisDrone** trong lượt này. Không dùng FPS mặc định hay seqinfo của bên thứ ba.
- Mở lại khi có bằng chứng FPS/stride đúng phiên bản dữ liệu: bài VisDrone-MOT2019, xác nhận của tác giả, hoặc metadata
  video gốc. Khi đó chạy val rồi test-dev, mỗi split đủ 36 run, theo đúng Q3.
