# R125 — PF100 mới trên BDD: IDSW giảm 10,84% so PF cổ điển

Ngày 19/11/2026, theo yêu cầu thử PF mới trên BDD. Đã hoàn tất **200 video BDD100K COCO-video validation / 39.973 ảnh**, 1.800 lượt clip mới và 24 lượt đo tuần tự PC. PF100 cải thiện rõ so PF cổ điển đã dùng trước đây; về IDSW đã tiến gần EKF/UKF theo điểm ước lượng. HOTA vẫn thấp hơn các đối chứng xác định và chi phí còn cao hơn PF cổ điển trên hai mẫu PC.

## Phạm vi và đối chứng

Giữ nguyên Cartesian PF100 R123, lấy mẫu omega và Gaussian có điều kiện cho các tọa độ còn lại. Không tinh chỉnh bộ lọc trên BDD. Chạy thêm Cartesian500 cùng mô hình, chỉ khác số hạt, mỗi loại ba seed 0/1/2; chạy lại KF/EKF/UKF mỗi loại một lần. Tổng số clip mới: 600 PF100 + 600 Cartesian500 + 600 KF/EKF/UKF. Đây là **200 video khác nhau**, không coi các seed là video độc lập.

PF cổ điển500 K1/K2 motion roughening2 dùng lại 600 lượt R74 đã lưu, cùng đầu vào. Baseline này là biến thể cũ đã có K1/K2, không phải PF bootstrap chưa cải thiện. Cartesian500 là đối chứng riêng để tách tác động số hạt; không gọi hai loại PF500 là cùng một mô hình.

Tất cả dùng **cùng cache YOLOv8n COCO của R61/R74**, conf tracking>.25, COCO vehicle classes [1,2,3,5,7], max300, FPS5, buffer1,2giây=6khung, elapsed steps. Chấm `mot_crowd` để giữ quy tắc vùng bỏ qua của BDD. Không chạy lại detector hoặc đo AP ở lượt này. Đây không phải 40 đoạn BDD với detector YOLO11n TensorRT trên Nano, không so trực tiếp các tổng IDSW khác phạm vi đó.

## Điểm tổng hợp

Mỗi dòng PF là trung bình **metric COMBINED của ba seed**, tính gộp đúng raw fields của TrackEval trước khi lấy trung bình giữa seed:

- **PF100 mới:** HOTA **33,4975%**, IDF1 **38,2511%**, IDSW **5.609,33**, FN **261.378,67**, FP **17.949,33**.
- **Cartesian500 cùng mô hình:** HOTA **33,5027%**, IDF1 **38,2567%**, IDSW **5.599,33**, FN **261.367,67**, FP **17.956**.
- **PF cổ điển500 K1/K2 motion2:** HOTA **32,9508%**, IDF1 **37,0857%**, IDSW **6.291,33**, FN **263.146,67**, FP **18.710**.
- **KF:** HOTA **33,6769%**, IDF1 **38,2511%**, IDSW **6.345**, FN **260.696**, FP **17.892**.
- **EKF:** HOTA **33,8265%**, IDF1 **38,7100%**, IDSW **5.548**, FN **260.982**, FP **17.682**.
- **UKF:** HOTA **33,7688%**, IDF1 **38,6275%**, IDSW **5.655**, FN **260.895**, FP **17.649**.

PF100 theo seed có IDSW **5.621 / 5.588 / 5.619**, HOTA **33,5044 / 33,5028 / 33,4854%**. Ba seed PF cổ điển tương ứng có 6.314 / 6.250 / 6.310 IDSW; cả ba cặp đều giảm với ứng viên mới.

## Diễn giải cùng khoảng tin cậy

Bootstrap ghép cặp theo **200 video**, 5.000 mẫu dùng chung cho mọi bộ lọc. Trong mỗi mẫu tính lại COMBINED riêng từng seed, rồi lấy trung bình. BDD validation đã được dự án phân tích trước đây; đây là kiểm chuyển miền của ứng viên được khóa trên các tập khác, không gọi là test mù.

**So PF cổ điển:** IDSW giảm **682 lần trung bình (10,84%)**, CI95% chênh PF100−cũ **[−855,03,−517,32]**. HOTA tăng **0,5467 điểm**, CI **[+0,4076,+0,6807] điểm**; IDF1 tăng **1,1654 điểm**, CI **[+0,9560,+1,3788]**. FN giảm **1.768 lượt hộp**, CI **[−2.032,03,−1.506,99]**, FP giảm 760,67. Mức giảm IDSW ở đây đi cùng cải thiện HOTA và độ bao phủ, không chỉ do bỏ nhiều mục tiêu hơn.

**So Cartesian500:** PF100 tăng **10 IDSW (0,179%)**, CI **[−21,67,+41,33]**; HOTA giảm **0,0052 điểm**, CI **[−0,0379,+0,0354]**; FN tăng **11**, CI **[−24,+47]**. Khác biệt quan sát rất nhỏ trong lượt này, nhưng CI chứa 0 không tự chứng minh hai bộ lọc tương đương chính xác hay 100 hạt tối ưu mọi tình huống.

**So KF:** PF100 giảm IDSW **735,67 (11,59%)**, CI **[−981,33,−514,65]**. Tuy nhiên HOTA giảm **0,1794 điểm**, CI **[−0,3223,−0,0401]**, FN tăng **682,67**, CI **[+445,99,+941,34]**; IDF1 gần bằng. Không nói PF100 tốt hơn toàn diện chỉ vì ít IDSW hơn.

**So EKF/UKF:** chênh IDSW lần lượt **+61,33 / −45,67**, CI **[−55,67,+176,68] / [−165,34,+76,33]**, chưa thấy khác biệt rõ theo phép kiểm này. HOTA PF100 thấp hơn **0,3290 / 0,2712 điểm** và FN cao hơn **396,67 / 483,67**; các CI tương ứng không chứa 0. PF100 đã tiến gần về IDSW, chưa vượt EKF/UKF về chất lượng tổng thể.

Không phân tầng che khuất từ cờ visibility mặc định của bản COCO-video và không suy diễn kết quả tổng thành ưu thế riêng ở quỹ đạo cong. Mọi điểm, kể cả điểm bất lợi, đều giữ trong `summary.json` và `bootstrap.json`.

## Chi phí cập nhật: PC, không phải FPS Nano

Sau khi mọi job chất lượng kết thúc, đo tuần tự cùng process trên hai video chọn **trước khi xem điểm** theo workload detection phân vị 50/90: `b1f4491b-846d8cb2` và `b1fc95c9-cb2882c7`. Mỗi đoạn lấy 200 ảnh, bỏ 16 khung đầu, ba lần đảo thứ tự, tổng 24 lượt và 184 khung đo/lượt. Cùng cache và seed 0 cho PF, không đổi detector.

Trung bình `tracker.update`: **KF 1,046 ms; PF cổ điển500 6,239 ms; Cartesian500 15,505 ms; PF100 6,911 ms**.

Giảm 500→100 trong cùng mô hình Cartesian giúp giảm **55,43% thời gian update**, cả sáu cặp đều nhanh hơn, khoảng 47,37–64,32%. So PF cổ điển, thời gian trung bình PF100 **tăng 10,78%**; 5/6 cặp chậm hơn, từng cặp dao động từ −9,51% đến +52,74%. Vì chỉ có hai đoạn và clock PC còn dao động, đây là mô tả chi phí trên mẫu, không phải mức chậm cố định cho mọi video.

**Kết luận về chi phí:** PF100 tiết kiệm đáng kể so Cartesian500, giữ gần các điểm chất lượng trên BDD; so PF cổ điển thì đang đổi thêm CPU lấy tracking tốt hơn. Không gọi mức giảm 55,43% là tăng FPS Nano, không tính FPS toàn hệ thống từ nghịch đảo thời gian tracker và không cộng với phần trăm của R123 ở phiên đo khác.

## Kiểm chứng và trạng thái

Preflight 1.053 hash đầu vào/mã nguồn/evaluator/GT/cache/seqinfo đạt. PF cổ điển phát lại trên hai video × ba seed khớp MOT R74 ở 6/6 lượt; PF100 cùng seed lặp khớp 6/6. Sau full: đủ **1.800 marker/MOT/CLS/timing**, số khung và n_steps kiểm đầy đủ, 9 nhóm raw tái hợp khớp COMBINED. KF/EKF khớp toàn bộ scalar fields ở 400 video×bộ lọc với R61 và sai khác tổng bằng 0. Timing 24 lượt lặp MOT khớp và prefix khớp lượt full, đầu vào/mã nguồn kiểm hash lại sau đo.

Mã và bằng chứng: `run.py`, `analyze.py`, `measure.py`, `inputs.json`, `smoke.json`, `summary.json`, `bootstrap.json`, `verification.json`, `serial_timing.json`, `final_verification.json`. PF100 vẫn là ứng viên nghiên cứu qua context `installed()` của R118; YAML một mình chưa chọn backend Cartesian trong launcher. Không đổi mặc định, source thuật toán, detector hoặc khóa luận. Không còn tiến trình chạy của lượt này.
