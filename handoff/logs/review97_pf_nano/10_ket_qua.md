# R97 — kiểm PF trên Nano và đo CV/PF theo nhiều video

Ngày 10/11/2026. Giao thức ở `00_protocol.md`. Tất cả lượt đo dùng Jetson Nano nguyên bản, gói v21combo+C1, engine TensorRT FP16 384×640 và JPEG mẫu R86. Không sửa launcher, gói đang dùng, detector hoặc cấu hình khóa. PF được đăng ký vào backend NumPy chỉ trong tiến trình wrapper thí nghiệm R96.

## Tương đương đầu ra PF

`parity_run.py` tạo 90 frame detection xác định cho ba phương tiện, gồm xe rẽ, confidence thay đổi và hai khoảng mất detection. Cùng YAML PF500 seed97 và buffer30. Tệp YAML PC/Nano có cùng SHA256 `ff43d4934d6ec72744ada9bb2a87dea7b10b3d0ac8764e8c1ec5a20ef8fa880b`. PC Ultralytics ↔ PC NumPy: 255 hàng ở 90 frame trùng tuyệt đối. PC NumPy ↔ Nano NumPy: cũng trùng tuyệt đối ID, lớp, hộp và score; sai số lớn nhất 0. Cổng khóa trước là cùng ID/lớp/thứ tự, box ≤0,001 px, score ≤1e-6. `compare_parity.py` kiểm hai cặp; các tệp `parity_pc_ultralytics.json`, `parity_pc_np.json`, `parity_nano_np.json` giữ đầu ra.

Mã `tracker_ctrv.py` và `pf_ctrv.py` trong gói Nano có hash khác mã PC hiện tại; các opt-in mới ở PC không được triển khai lên Nano. Vì vậy phép kiểm xác nhận fixture này, không thay thế kiểm parity trên mọi detection thực tế. Điều kiện đủ để mở **đo tốc độ** là cùng hành vi trong fixture, không phải chứng minh chất lượng tracking trên tập dữ liệu.

## Tốc độ nhiều video

`run_multi.py` chọn trước chỉ số 0,5,...,35 của mỗi nhóm 40 video trong `sample_manifest.json` R86: 8 BDD, 8 UA. Mỗi video có 150 JPEG liên tiếp, chạy CV và PF500 trên cùng ảnh. BDD FPS5/steps/lớp `[1,2,3,5,7]`; UA FPS25/skip/lớp `[2,3,5,7]` và ignored regions. Cặp video đảo thứ tự CV→PF/PF→CV. Warmup25, tắt hiển thị. 32/32 lượt hoàn tất đủ 150 frame, không unreadable/truncated; cùng boot, swap tối đa 69,9 MB, `errors_count=0`, nhiệt qua guard <75°C. Không còn tiến trình đo sau kết thúc. Từng lượt và metadata nguyên dạng ở Nano `~/nano_run/r97_pf_multi_20261110/`; tổng hợp PC `nano_multi.json`.

`analyze.py` tính hiệu PF so CV **theo từng video**, báo trung vị 8 cặp mỗi tập và bootstrap lấy mẫu video 5.000 lần, seed9710. BDD: FPS steady chênh trung vị **−0,10%**, khoảng 95% **[−7,42%; +7,48%]**, PF nhanh hơn 4/8 video; update tracker PF−CV **+8,04 ms**, khoảng **[+2,00; +19,21] ms**. UA: FPS **−9,53%**, khoảng **[−10,52%; +6,50%]**, PF nhanh hơn 2/8; update **+34,80 ms**, khoảng **[+9,83; +79,38] ms**. Update PF cao hơn CV ở 16/16 cặp. Chênh infer median ghép cặp BDD +0,006 ms, UA +0,143 ms; trễ p90 PF−CV trung vị BDD +20,24 ms, UA +45,89 ms. Số từng video và khoảng bootstrap ở `multi_summary.json`.

Chi phí PF tăng rõ theo cảnh: chênh update từng video BDD +0,90…+27,40 ms; UA +2,73…+92,12 ms. Decode khác nhau đáng kể giữa một số lượt (chênh −13,00…+8,17 ms) và làm FPS cặp biến động; khoảng FPS đều chứa 0, chỉ có 8 video mỗi tập. Kết luận chắc ở mức phép đo này là **bước update PF tốn thời gian hơn CV**, nhất là cảnh UA nhiều track. Chưa đủ để khẳng định hiệu ứng FPS toàn luồng tổng quát hoặc quy kết do riêng thuật toán PF trong mọi cảnh. Đây là JPEG tắt hiển thị, không phải MP4/camera; không đánh giá HOTA/IDSW và không áp hệ số R86 CV/EKF cho PF.

Hướng tối ưu tiếp theo có căn cứ là profile update theo số track/particle ở các video UA có chênh lớn, xác định vòng lặp tốn thời gian, rồi thử giảm chi phí tính toán trong nhánh PF trên bản cô lập. Mọi ứng viên giảm số hạt, đổi proposal hoặc cách ghép cần qua kiểm chất lượng IDSW/HOTA trên dev và tập chuyển miền trước khi dùng trên Nano.
