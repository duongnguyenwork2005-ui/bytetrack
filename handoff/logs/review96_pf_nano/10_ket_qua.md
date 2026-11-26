# R96 — IDSW theo thời điểm ghép và pilot PF500 trên Jetson Nano

Ngày 10/11/2026. Theo yêu cầu người dùng, tiếp tục hai ưu tiên đã nêu: tìm vị trí phát sinh IDSW của PF và đo PF500 riêng trên Nano. Không chỉnh thuật toán tracker, detector, YAML khóa hoặc launcher.

## IDSW BDD100K

`idsw_events.py` dùng đúng ma trận điểm, ngưỡng IoU 0,5, ưu tiên giữ ID từ frame trước và quy tắc đếm IDSW của `trackeval.metrics.clear.CLEAR` phiên bản đang cài. Nó lấy dữ liệu đã qua bước tiền xử lý chính thức của evaluator `mot_crowd`, nên cùng quy tắc bỏ vùng crowd. Đối chiếu từng video với `per_video_metrics.csv`: 200 video × hai cấu hình × ba seed đều khớp số IDSW nguyên. Mỗi lỗi được chia bằng lịch sử GT đã ghép: "xuất hiện lại" nếu GT này chưa được ghép ở ít nhất một frame liền trước; "liên tục" nếu có ghép ở frame ngay trước. Các trường hợp GT vắng mặt cũng có thể tạo gap; đây là nhãn quan sát, không cho biết trạng thái Lost của tracker.

| BDD val200, trung bình 3 seed | PF nền R74 | PF + association R95 | Chênh |
|---|---:|---:|---:|
| IDSW tổng | 6.291,33 | 6.162,33 | −129,00 |
| IDSW tại lần ghép lại | 3.871,67 | 3.686,00 | −185,67 |
| IDSW khi GT được ghép liên tục | 2.419,67 | 2.476,33 | +56,67 |
| Trong dòng liên tục: ≤3 frame sau một lần ghép lại | 442,00 | 464,33 | +22,33 |

Số theo từng seed (nền→R95): IDSW tổng 6.314→6.179, 6.250→6.064, 6.310→6.244. Nhóm "xuất hiện lại" giảm 208, 179, 170; nhóm "liên tục" đổi +73, −7, +104. Xu hướng giảm tại lần ghép lại có ở cả ba seed; nhóm liên tục biến động. Tách nhóm không chứng minh R95 trực tiếp gây từng lỗi hay rằng phục hồi động học sau che khuất sẽ giảm chúng. R95 vẫn trượt cổng BDD đã khóa ở R95; không đổi quyết định đó.

UA dev12: chỉ còn tệp MOT candidate R95 đầy đủ trong vùng hoạt động; ba seed R95 tái hiện đúng IDSW 656/646/638, trong đó lỗi tại lần ghép lại 480/484/485. Baseline R72 dev12 đã chuyển lưu trữ, nên không có phép so phân nhóm UA full. Hai video smoke có baseline và candidate, seed0 68→72 IDSW; không ngoại suy hai video này sang toàn UA.

Đầu ra có từng event: `idsw_bdd_s{0,1,2}_200.json`, `idsw_ua_s{0,1,2}_12.json`; mã đọc lại `idsw_events.py`. Các ID trong event là ID compact sau tiền xử lý của evaluator, chỉ dùng để truy vết trong cùng video.

## Pilot Nano

Giao thức ở `00_protocol.md`. SSH tới `10.11.62.64`, thiết bị vẫn là Jetson Nano, boot nhất quán, swap xuất phát 0 MB và `errors_count=0`. Dùng 150 JPEG liên tiếp của video đầu tiên BDD và UA trong manifest R86, detector gói v21combo+C1 engine 384×640 FP16, tracker CV/PF500 cấu hình mặc định, buffer 1,2 s, tắt hiển thị. BDD CV→PF; UA PF→CV. Đầu ra và giám sát lưu ở Nano `~/nano_run/r96_pf_pilot_fix_20261110/`, số chuyển về PC `nano_pilot.json`.

Backend NumPy đang dùng trên Nano chỉ đăng ký CV/EKF. Lượt thử đầu: CV BDD hoàn tất, PF dừng trước xử lý với lỗi registry; log giữ riêng ở `~/nano_run/r96_pf_pilot_20261110/`. Sau đó dùng `nano_live_pf_experiment.py` để thêm `PFByteTracker` vào registry **chỉ trong tiến trình thí nghiệm**. `run_pilot.py` chạy lại toàn bộ cặp ở thư mục mới. 4/4 lượt hoàn tất 150/150 frame, không ảnh hỏng, swap tăng tối đa trong ngưỡng 100 MB, nhiệt <75°C, boot không đổi và lỗi thẻ vẫn 0.

| Video | Tracker | FPS ổn định | Decode median ms | Infer median ms | Update tracker median ms | Trễ p90 ms |
|---|---|---:|---:|---:|---:|---:|
| BDD b1c81faa-c80764c5 | CV | 12,592 | 20,391 | 44,712 | 4,733 | 88,411 |
| BDD b1c81faa-c80764c5 | PF500 | 13,454 | 19,067 | 44,634 | 10,501 | 98,376 |
| UA MVI_39031 | PF500 | 13,563 | 18,972 | 44,696 | 6,049 | 91,044 |
| UA MVI_39031 | CV | 15,073 | 11,674 | 44,674 | 3,117 | 73,847 |

PF500 có update median cao hơn CV 5,768 ms ở BDD và 2,932 ms ở UA. Infer gần như bằng nhau; decode biến động đáng kể, đặc biệt lượt UA PF/CV 18,972/11,674 ms, nên FPS toàn luồng trên một video/tập không đủ để xếp tốc độ hai tracker. BDD PF FPS nhỉnh hơn dù update chậm hơn, xác nhận nhiễu IO cần được kiểm soát. Đo Nano này chỉ là tốc độ: chưa kiểm parity đầu ra PF backend NumPy với bản PC/Ultralytics và chưa chấm chất lượng PF trên Nano. Muốn kết luận tốc độ cần nhiều video, cặp đảo thứ tự và cùng tầng decode; muốn tối ưu PF thì đo nóng trên các đoạn có nhiều track, rồi giữ mọi kiểm chất lượng IDSW/HOTA trước khi giảm số hạt.
