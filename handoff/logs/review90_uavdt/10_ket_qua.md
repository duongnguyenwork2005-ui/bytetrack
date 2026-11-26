# R90 — UAVDT: theo vết phương tiện thay cho MOT17

Ngày 06/11/2026. **Phạm vi chỉ là xe.** R89 MOT17 là tập người đi bộ và đã bị loại khỏi mọi so sánh hoặc kết luận khóa luận. R90 dùng UAVDT để thử thêm cảnh UAV có che khuất và đổi hướng, với car/truck/bus; **không có xe máy**. Kết quả xe máy hiện vẫn phải dựa vào UTUAV/HELMET và các tập xe liên quan.

## Nguồn và giao thức

Tải ba gói do [trang tác giả UAVDT](https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5) liên kết: `UAV-benchmark-M.zip`, `UAV-benchmark-MOTD_v1.0.zip`, `M_attr.zip`. Bản lưu ở `F:/LABOTARY_data/goc_tai_ve/UAVDT-R90/`. CRC của 40.786/1.094/54 entry đạt, SHA256/byte/URL cụ thể ở `download_manifest.json`. [Bài ECCV 2018 của tác giả](https://openaccess.thecvf.com/content_ECCV_2018/papers/Dawei_Du_The_Unmanned_Aerial_ECCV_2018_paper.pdf) mô tả xe, thuộc tính che khuất và MOT. Trang ghi 30 fps, 1080×540; **ảnh thật trong ZIP là 1024×540**. Tám chuỗi chọn có ảnh đánh số liên tục 1..N; chưa có xác nhận độc lập về stride trích ảnh, nên mốc 30 fps/36 bước buffer là danh nghĩa.

Chọn trước khi xem điểm detector/tracker tám chuỗi train: M1306, M1005, M1202, M0206 (che khuất cao); M0204, M0501 (tỷ lệ đổi hướng 10 frame cao); M0301, M1008 (đối chứng che khuất thấp). Quy tắc và số liệu nguồn nằm trong `selection.json`, `label_audit.json`; tổng **4.367 frame, 59.472 box xe, 227 ID cộng theo chuỗi**. GT MOT gộp car/truck/bus thành lớp số 1 theo quy ước dự án; TrackEval in tên nội bộ `pedestrian` cho lớp số 1 nhưng **mọi box được chấm đều là phương tiện**. Các box `_gt_ignore` thành lớp distractor 8, không tính là xe. Quy tắc bỏ ignore của TrackEval khác mã MATLAB chính thức (MATLAB bỏ prediction nằm trọn trong box ignore); vì vậy đây là **benchmark chuyển miền nội bộ, không phải điểm leaderboard**.

Smoke 10 frame rải đều mỗi chuỗi, ba checkpoint R54, IoU ghép ≥0,5 và conf>0,05. Recall YOLOv8n COCO **510/979 = 52,09%, 7/8 chuỗi ≥15%**; YOLOv8n mixed5 **25,54%**, YOLO11n mixed5 **32,18%**. Cổng khóa trước (recall gộp ≥20%, ít nhất 4 chuỗi ≥15%) chọn COCO. Tạo **một** cache detection cho mọi bộ lọc, không mask ảnh trước detector; kiểm toàn bộ 4.367 ảnh không lỗi và không chạm `max_det=3000`. Cache gồm 116.071 box thuộc mọi lớp COCO; tracker chỉ đọc class car/bus/truck 2/5/7 ở `conf>0,25` hoặc `>0,10`, `max_det=300`, FPS danh nghĩa 30, buffer 1,2 s = 36 frame, elapsed `steps`. Cache meta giữ cờ mặc định `mask_ignored_regions=true`, nhưng đường mã R90 gọi trực tiếp `detect_frame` trên ảnh gốc nên **thực tế không có mask**; xem `src/r90_uavdt/cache.py`.

## Cận trên do detector

Trên toàn 59.472 GT xe, recall detection một-một IoU≥0,5 là **35.278 = 59,32%** ở conf>0,05, **30.495 = 51,28%** ở conf>0,10 và **21.147 = 35,56%** ở conf>0,25. Trong **3.224 box che khuất nặng**, recall tương ứng **862 = 26,74%**, **736 = 22,83%**, **486 = 15,07%**. M0501 (sương mù, độ cao lớn theo thuộc tính nguồn) chỉ có **177/4.583 = 3,86%** ở conf>0,05; ở c25 CV không tạo track nào. Không thể coi HOTA/IDSW thấp ở cảnh này là bằng chứng riêng về chuyển động của bộ lọc.

## Kết quả theo cùng cache

Mỗi dòng sau là **HOTA / IDF1 / MOTA (%) ; IDSW** của COMBINED 8 chuỗi. PF là trung bình số liệu COMBINED của ba seed 0/1/2, không phải một tracker gộp seed.

- **c25:** CV 28,58 / 36,63 / 27,38 ; 327. EKF 28,18 / 35,94 / 27,32 ; 345. UKF 27,78 / 34,76 / 27,17 ; 365. PF mặc định 24,53 / 29,61 / 25,60 ; 502,67. PF K1/K2 motion100 26,40 / 32,48 / 26,54 ; 423,33.
- **c10:** CV 31,48 / 40,00 / 30,01 ; 268. EKF 31,53 / 40,12 / 29,79 ; 302. UKF 30,84 / 38,91 / 29,85 ; 306. PF mặc định 27,14 / 33,05 / 28,03 ; 414,33. PF K1/K2 motion100 28,87 / 35,55 / 29,19 ; 337,67.

PF K1/K2 giảm IDSW so PF mặc định **79,33 (15,8%) c25** và **76,67 (18,5%) c10**, tăng HOTA khoảng 1,87/1,73 điểm. Tuy nhiên PF K1/K2 vẫn hơn CV **96,33 IDSW c25** và **69,67 IDSW c10**. Ở M1306 che khuất cao, c25 CV 140 so PF K1/K2 trung bình 204 IDSW; c10 134 so 192. Một số chuỗi nhỏ PF gần hoặc thấp hơn CV nhưng không đổi kết quả gộp. Không chọn PF thay CV hoặc điều chỉnh YAML luận văn/Nano từ phép thử này.

## Hậu kiểm và giới hạn

`final_review.json` đạt **18/18 run**, **144/144 đơn vị video/seed**, đủ tám video cho từng evaluator; `run_meta` cùng cache/FPS/classes/buffer; tệp MOT hữu hạn, không trùng `(frame, ID)`; GT chuyển đổi khớp SHA; tái cộng HOTA/IDF1/MOTA/IDSW từ raw fields lệch nhiều nhất `5,55e-17`. Số đầy đủ theo seed/video ở `summary.json`, `matrix_progress.json`, `results/trackeval/UAVDT-R90train8/`.

Giới hạn: chỉ tám chuỗi **train có GT công khai**, được chọn theo thuộc tính nguồn để khảo sát tình huống chứ không đại diện ngẫu nhiên toàn UAVDT; chỉ một detector sau cổng smoke; M0501 và M0301 bị detector hạn chế; quy tắc ignore nội bộ khác chính thức; stride nguồn chưa được tác giả xác nhận tường minh. Không dùng phép thử này làm kết quả test ẩn, xếp hạng chính thức hay bằng chứng PF đã ngang CV.
