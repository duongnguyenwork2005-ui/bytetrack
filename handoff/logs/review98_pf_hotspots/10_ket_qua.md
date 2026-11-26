# R98 — profile PF trên Nano, tối ưu số học nhỏ và kiểm RideSafe-400

Ngày 10/11/2026. Theo yêu cầu tiếp tục tối ưu, vòng này dùng hai video UA đã có trong mẫu R97, nên không chọn hậu nghiệm video theo kết quả của R98. Giao thức và guard ở `00_protocol.md`. Không thay đổi detector, tracker configuration khóa hoặc launcher Nano. Bộ lọc PF trong gói Nano chỉ được mở bằng wrapper thí nghiệm R96/R98.

## Vì sao PF chậm khi nhiều xe

`profile_nano.py` chạy lại C1 trên MVI_39031 và MVI_40855 để lưu detection cache riêng; sau đó dùng cProfile chỉ quanh `tracker.update` khi phát lại chính cache đó. Cả hai video đủ 150 frame. MVI_39031 có trung vị 1 detection, 1 tracked, 0 lost và update khoảng 6,17 ms/frame. MVI_40855 có 24 detection, 24 tracked, 9 lost và update khoảng 110,91 ms/frame. Trong 150 frame của cảnh đông, profiler ghi `update` tổng 16,57 s; `estimate_state` cộng dồn 5,06 s (8.027 lần gọi), `_logsumexp` 2,46 s (14.613 lần), `f_batch` 1,95 s. Cumulative times lồng nhau, không cộng các số này thành tổng độc lập. Dữ liệu từng frame/hàm ở `profile_MVI_39031.json` và `profile_MVI_40855.json`. Kết quả định vị việc cần tối ưu vào nhánh PF khi nhiều track, không phải detector.

Thử hai công thức hiệp phương sai cân bằng trọng số trên Nano (`microbench_cov.py`): đổi thứ tự phép nhân ma trận không tăng tốc đáng kể; `einsum` khoảng 1,10–1,14 ms/lần so công thức hiện tại 0,52–0,56 ms/lần. Không áp dụng hai cách này.

## Tối ưu `_logsumexp`

Ứng viên thay `np.max`/`np.sum`/`np.isfinite` bằng phương thức mảng `a.max()`/`(...).sum()` và `math.isfinite`. Công thức, nhánh giá trị không hữu hạn và kiểu trả về giữ nguyên. Microbenchmark trên Nano (`microbench_logsumexp.py`, N=500) giảm khoảng 99,2→67,1 µs/lần với trọng số đều và 129,7→98,5 µs/lần với trọng số lệch; cả `-inf` cho cùng kết quả. Đây chỉ là thời gian của hàm nhỏ, không phải FPS.

Phát lại detection thật theo A/B/B/A (`bench_lse_nano.py`): cả 8 lượt ở hai video đều tạo MOT trùng SHA theo từng video. Trên cảnh đông MVI_40855, median update mỗi lượt nền 103,69/102,33 ms, ứng viên 97,57/98,50 ms; tổng 3.312 hàng MOT mỗi lượt trùng byte. Trên cảnh thưa MVI_39031, median update nền 5,17/5,20 ms, ứng viên 5,07/5,17 ms; 159 hàng MOT trùng byte. Dữ liệu `lse_abba.json`.

Đo toàn luồng C1 live A/B/B/A trên cùng hai video (`live_abba.py`), 8/8 lượt đủ 150 frame, MOT trùng byte theo video, boot/lỗi thẻ/swap/nhiệt đều qua guard. Ở MVI_40855, decode median mỗi lượt 10,60–10,76 ms và infer 45,69–45,77 ms. Trung vị hai lượt nền/ứng viên: FPS steady **8,687→9,107 (+4,8%)**, update **113,58→108,18 ms (−4,8%)**, p90 trễ 632,24→592,65 ms. Ở MVI_39031, decode nền 19,83/11,65 ms vs ứng viên 11,57/11,55 ms; vì nhiễu I/O lớn, không quy chênh FPS cho mã. Dữ liệu `live_abba.json`. Tất cả là mẫu 150 JPEG/video, không phải FPS MP4/camera hay hiệu quả trên mọi cảnh.

Đã áp dụng ba dòng tối ưu trong `src/pf_ctrv.py` của PC. `src/test_pf_ctrv.py` 26/26, `src/test_pf_rb_observed.py` đạt. Fixture R97 90 frame/255 hàng so trước/sau trùng tuyệt đối trên PC NumPy và PC Ultralytics (`parity_pc_*_fast.json`). Nano launcher/gói v21combo không đổi; ứng viên Nano chỉ chạy qua `fast_pf_live.py`. Vì dữ liệu thí nghiệm còn hai video, không tuyên bố tăng tốc tổng quát hay đổi lựa chọn bộ lọc của luận văn.

## RideSafe-400: nhãn có ID nhưng lấy mỗi 5 frame

Từ [nguồn tải của tác giả](https://huggingface.co/datasets/DeepBug/RideSafe-400/tree/main), tải riêng `videoset1_xml_annots_with_rider_motor_poly.zip` về `F:/LABOTARY_data/goc_tai_ve/RideSafe-400-R98/`: 137.111.722 byte, SHA256 `bf19000cc9d738b1295cfa40e3c17e097ba6c0d98e2e277369c6e28d43b6f308`, CRC toàn ZIP đạt. Không giải nén 1,188 GB XML; `audit_ridesafe_labels.py` đọc luồng 100/100 XML và ghi `ridesafe_videoset1_audit.json`.

**100/100 XML ghi `frame_filter=step=5`**; box motorcycle hiển thị nằm ở frame gốc chia hết cho 5. Có 131.028 box motorcycle hiển thị trên 26.156 mốc frame cộng theo video, 6.856 track CVAT có box hiển thị, 6.532 track có ít nhất hai box hiển thị. Trong cùng track, 119.789 bước nhãn cách đúng 5 frame gốc, 4.383 bước cách khác; không có box không dương hoặc trùng `(frame, CVAT track id)` ở các box hiển thị. Thuộc tính `motor_track_id` thiếu hoặc mặc định `-1` ở 2.336 box, và có 246 lượt trùng `(frame, motor_track_id)`; cần dùng/audit ID track CVAT thay vì mặc định lấy thuộc tính đó. Cờ `occluded=1` chỉ xuất hiện ở 2/143.982 box motorcycle kể cả ngoài hình; không dùng để phân tầng che khuất.

Nhãn stride 5 cho camera gắn xe có ích để kiểm/phát triển detector xe máy và thăm dò MOT nhịp thấp, nhưng không là tập chính phù hợp cho phân tích IDSW khi thiếu detection vài frame liên tiếp ở nhịp 25–30 fps. FPS từng video, chia train/val/test và mức phủ nhãn các lớp xe khác chưa được kiểm với video thực. Vì vậy chưa tải 37,8 GB video hoặc chạy tracking; tiếp tục ưu tiên HELMET/SVMOT và cảnh camera ven đường/xe khác có nhãn dày cho kết luận PF.
