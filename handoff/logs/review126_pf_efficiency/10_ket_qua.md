# R126 — giảm chi phí PF100 và giữ nguyên chất lượng

Ngày 19/11/2026. Yêu cầu: tiếp tục cải thiện PF vì chi phí cao trong khi chất lượng chưa vượt các bộ lọc khác.

## Kết quả chọn giữ

Lõi C/float64 cho Cartesian PF100 giảm **35,09% thời gian cập nhật tracker** so bản PF100 Python/NumPy trên mẫu đo PC. Cả sáu cặp video/lần đều nhanh hơn, mức giảm 27,45–41,83%. Không thay mô hình chuyển động, số hạt, Q/R, khởi tạo vận tốc, RNG, resampling hoặc association.

Đã chạy lại **200 video BDD100K / 39.973 khung mỗi seed / ba seed**, đủ **600 lượt video**. Toàn bộ trường raw TrackEval của từng video khớp R125. Frame và ID giữ nguyên; sai khác hộp tối đa 5,01e−11 px do số học/lưu CSV. HOTA, IDSW, IDF1, FN và FP giữ nguyên. Đây là cải thiện cài đặt có kiểm hồi quy, không phải kết quả tình cờ tốt lên do sai khác làm tròn.

## Cách làm

Profile cho thấy các phép toán Gaussian và tổng hợp covariance chiếm phần lớn thời gian PF100; nhiều phép toán ma trận nhỏ tạo overhead gọi NumPy, cấp phát và sao chép.

- A1: cache biểu diễn hạt, tái sử dụng chỉ số/identity, truy cập block liên tục, giải chuyên biệt 2×2 có fallback. Giữ số học nhưng chỉ giảm 8,73% thời gian, chưa đạt cổng 20%, vẫn chậm hơn PF cổ điển. Giữ kết quả âm, không chọn A1 làm kết quả cuối.
- A2: chuyển predict Gaussian 4D, cập nhật vị trí 2D, Joseph covariance, hai block shape và within-covariance sang C/float64. Tính trọng số, mixture mean, NumPy RNG và resampling vẫn giữ. Dùng `/fp:strict`, không fast-math; build bằng MSVC BuildTools đã có sẵn, không cài package hay thay môi trường chung.
- Cập nhật C ghi vào scratch buffer; khi phát hiện ma trận kém điều kiện, trở về đường NumPy với input chưa bị sửa. Cache hạt công khai vẫn trả bản sao để giữ quyền sở hữu dữ liệu.

Không loại bỏ covariance hoặc bỏ bước xử lý để lấy tốc độ.

## Chất lượng BDD đủ 200 video

Cùng cache YOLOv8n COCO R61/R74, năm lớp phương tiện, ngưỡng c25, FPS 5, buffer 6 khung, elapsed steps; chấm `mot_crowd` để xử lý đúng crowd ignore. Không dùng số này thay cho chất lượng detector mixed5 trên Nano.

- Native PF100: HOTA **33,4975%**, IDF1 **38,2511%**, IDSW **5.609,33**, FN **261.378,67**; trung bình ba seed, khớp PF100 R125.
- PF cổ điển500 K1/K2 motion2 R74: HOTA **32,9508%**, IDF1 **37,0857%**, IDSW **6.291,33**, FN **263.146,67**. Native giữ mức cải thiện của R125: IDSW giảm **10,84%**, HOTA tăng **0,547 điểm phần trăm**, FN giảm **1.768**. Lõi C không tạo ra mức cải thiện chất lượng này; nó giữ chất lượng PF100 đã có và giảm chi phí.
- KF/EKF/UKF cùng đầu vào: IDSW **6.345 / 5.548 / 5.655**, HOTA **33,6769 / 33,8265 / 33,7688%**. PF ít IDSW hơn KF, nhưng HOTA vẫn thấp hơn cả ba và FN cao hơn. Chênh IDSW PF so EKF/UKF chưa chắc chắn theo bootstrap R125. Chưa có cơ sở gọi PF tốt nhất toàn diện.

BDD val và UA dev đã được phân tích nhiều vòng; không gọi đây là test mù hoặc bằng chứng tổng quát cho mọi camera.

## Đo tốc độ PC

Mẫu đã khóa trước bằng workload trung vị/p90 của detection: `b1f4491b-846d8cb2` và `b1fc95c9-cb2882c7`. Mỗi prefix 200 khung, bỏ 16 khung làm nóng, ba lần với đảo thứ tự ở lần thứ hai. Các lượt đo chạy tuần tự, không đồng thời với quality jobs.

18 lượt so các bản PF, thời gian `tracker.update` trung bình:

- PF cổ điển500: **5,341 ms**.
- Cartesian PF100 Python/NumPy: **6,510 ms**.
- Native Cartesian PF100: **4,225 ms**.

Native giảm thời gian **35,09%** so PF100 gốc và **20,90%** so PF cổ điển. Cả sáu cặp đều nhanh hơn PF cổ điển, khoảng giảm 8,94–27,70%. Hai cổng tốc độ định trước đều đạt.

Đo bổ sung 24 lượt cho bốn bộ lọc, cùng hai prefix và ba lần: KF **1,028 ms**, EKF **1,200 ms**, UKF **2,804 ms**, native PF100 **4,521 ms**. PF vẫn tốn CPU hơn cả ba. Số tuyệt đối khác nhẹ giữa hai lượt đo; dùng các so sánh trong cùng lượt, không trộn 4,521 ms với baseline 6,510 ms để tính phần trăm mới.

Đây là thời gian tracker trên hai mẫu PC, **không phải FPS toàn hệ thống hay FPS Jetson Nano**, cũng không phải đo tốc độ đủ 200 video. Chưa build và đo lõi C trên ARM/Nano.

## Kiểm chứng và phạm vi

- Kiểm số học 720 bước predict/update trên ba seed: max state error 2,84e−13; covariance PSD, chuỗi RNG, copy ownership và khôi phục hook đạt.
- Kiểm riêng API đã biên dịch: 1.500 ma trận SPD trên ba thang 1e−8/1/1e8, Cholesky/Kalman/Joseph khớp NumPy; fallback ở hạt thứ ba không sửa input; PSD đạt.
- Smoke BDD 12/12, full 600/600, đối chiếu raw metric từng video, marker/MOT và 1.671 hash đầu vào đạt.
- 18 timing PF và 24 timing bốn bộ lọc có kiểm lặp MOT; các prefix của 24 lượt sau khớp R125.
- Hậu kiểm tổng hợp R126/R127 đạt trong `native_validation/closeout_verification.json`.

Đã sửa ba lỗi harness có giữ snapshot/manifest/log ban đầu: confidence CSV bị đối chiếu như số chưa làm tròn; import nhầm module `measure` trùng tên; đường tham chiếu KF/EKF/UKF thêm seed trong khi R125 lưu không có seed. Không đổi thuật toán, tolerance theo kết quả hoặc cấu hình để xử lý các lỗi này.

Mã thử cô lập tại `native_pf.py` và `native/kernel.c`; `native_job.py` cài backend bằng context trong process. Chưa đưa vào launcher hoặc đổi mặc định `src/`; YAML100 đơn lẻ không tự chọn backend Cartesian/native. Không sửa khóa luận.

## Bằng chứng và chạy lại

- `native_validation/summary.json`: chất lượng, so sánh thời gian và cả sáu cặp.
- `native_validation/verification.json`: 600 raw-field parity, khung/ID/MOT và hash.
- `native_validation/serial_timing.json`, `all_filters_timing.json`: toàn bộ phép đo.
- `native_validation/numerical.json`, `c_api_checks.json`: số học và C API.
- `native_validation/final_verification.json`, `closeout_verification.json`: integrity cuối.
- `native/build.json`: source/library/compiler hash và lệnh build.

Python của repo: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`.
Chạy `handoff/logs/review126_pf_efficiency/native_job.py verify` để kiểm lại chất lượng,
và `handoff/logs/review126_pf_efficiency/native_validation/closeout.py` để kiểm hồ sơ cuối.

Kết luận: đã có bản PF vừa tốt hơn PF cổ điển về chất lượng BDD, vừa rẻ hơn trên mẫu PC. So với KF/EKF/UKF, đây vẫn là một đánh đổi cần chứng minh theo kịch bản; chưa nên thay mặc định chỉ vì PF là bộ lọc phức tạp hơn.
