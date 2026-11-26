# R85 C2 352×576 — loại ở cổng chất lượng dev12

Ngày 05/11/2026. Cùng `models/yolov8n.pt` SHA256 `f59b3d83…`, conf cache 0,05, NMS 0,7, ảnh UA-DETRAC train dev12 và tracker R63. C2 `imgsz=576, rect=True` cho tensor 352×576; letterbox được kiểm trùng pixel với `nano_detector` trên ảnh 960×540 và BDD 1280×720. Cache C2 đủ 12/12 chuỗi, không unreadable/truncated. Đối chứng là C1 `r63_yolov8n-coco_c0.05_dev12_rect640`, cùng checkpoint và GT.

| Chỉ tiêu dev12 c25 | C1 | C2 | Thay đổi | Cổng |
|---|---:|---:|---:|---|
| CV HOTA | 59,000% | 58,176% | −0,824 điểm; CI95 video [−1,340; −0,013] | Trượt (≥−0,5) |
| CV IDF1 | 71,768% | 71,337% | −0,431 điểm | Đạt theo điểm |
| CV IDSW | 489 | 462 | −27 | Đạt |
| EKF HOTA | 59,252% | 58,110% | −1,142 điểm; CI95 video [−1,984; −0,087] | Trượt |
| EKF IDF1 | 72,448% | 71,247% | −1,202 điểm | Trượt |
| EKF IDSW | 472 | 436 | −36 | Đạt |
| Recall xe nhỏ (<24 px gốc) | 5,17% | 0% | −5,17 điểm; 58 GT/4 video | Trượt, tầng ít GT |
| Recall che nặng (visibility<0,5) | 22,30% | 19,64% | −2,66 điểm; 15.269 GT | Trượt (≥−2,0) |

IDSW giảm nhưng HOTA, IDF1 EKF và recall che nặng trượt cổng. Không mở BDD dev139/C3/C4, không đo hay triển khai pipeline C2. Lượt AP C2 được dừng ở bước tiền kiểm dataset sau khi cổng khác đã trượt; không có AP C2. Tên `09_ap_dev.log` là lỗi gọi split `dev` (hợp đồng dùng `val`), `09_ap_dev_retry.log` là lượt tiền kiểm đã dừng, không dùng làm kết quả AP.

Bằng chứng: `01_cache_dev12.log`, `recall_dev12_c2_vs_c1/ket_qua.json`, `11_c2_tracking_dev12.json`, TrackEval `results/trackeval/DETRAC-all/r85-c2-dev12-c25-{cv,ekf}/` và baseline `r63g4-rect640-c25-{cv,ekf}/`.
