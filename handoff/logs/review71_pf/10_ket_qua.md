# R71 — kiểm chứng roughening có chọn lọc (03/11/2026)

Người dùng chấp thuận thử tách nhiễu K2 cho hộp bao. Lượt này dùng lại UA-DETRAC train dev12, detection cache YOLOv8n COCO, confidence 0,25, FPS 25 và ByteTrack association của R70. Giao thức khóa trước ở `00_protocol.json`; 78/78 hash nguồn trong `input_lock.json` khớp sau chạy. Đây là tập phát triển đã dùng để lựa chọn PF.

## Thay đổi và đối chứng

Tham số `pf_roughening_box_scale` nhân độ lệch chuẩn roughening của `cx,cy,a,h`; năm thành phần còn lại giữ hệ số K2 1,0. Ba cấu hình hộp 1,0 (R70 gốc), 0,5 và 0,0 đều có K1 optimal proposal, 500 hạt, R theo chiều cao quan sát và seed 0/1/2. Giá trị mặc định là 1,0. Mọi cấu hình dùng chung GT và detection cache.

3/3 kiểm thử mới và 8/8 hồi quy R70 đạt. Smoke 18/18, full 108/108 lượt video; điểm COMBINED tính lại từ trường TrackEval thô sai khác tối đa 5,56e-17. Đối chứng 1,0 cho 36/36 tệp MOT trùng byte bản K1K2 R70. Không có khác biệt nguồn đầu vào hoặc thay đổi đường chấm điểm được phát hiện.

## Kết quả

HOTA COMBINED trung bình ba seed: hộp 1,0 **56,600%**; hộp 0,5 **56,282%**; hộp 0,0 **56,346%**. Khoảng mô tả 95% lấy mẫu ghép cặp 12 video 5.000 lần:

- 0,5 − 1,0: HOTA **−0,317 điểm phần trăm** [−0,888; +0,023]; DetA −0,800 [−1,354; −0,393]; MOTA −1,840 [−3,988; −0,574]; IDSW trung bình +6,7 [−17,3; +30,3].
- 0,0 − 1,0: HOTA **−0,254 điểm phần trăm** [−1,048; +0,251]; DetA −1,123 [−1,889; −0,554]; MOTA −2,959 [−6,125; −1,067]; IDSW trung bình −8,7 [−30,7; +13,0].

AssA tăng 0,486 và 1,073 điểm phần trăm, nhưng cả hai khoảng ghép cặp đều qua 0. Điểm HOTA cũng chưa cho bằng chứng chắc chắn rằng hai biến thể mới tốt hơn hoặc kém hơn; DetA và MOTA giảm trên dev này.

Chẩn đoán trước association, chỉ trên các dự đoán có liên kết được với GT: ở khoảng 3–8 frame kể từ lần xuất MOT trước, trung vị sai số tâm/chiều cao GT là 0,492 → 0,447 → 0,407 khi giảm mức nhiễu hộp. Tỷ lệ mẫu đủ điều kiện để chấm đồng thời giảm 29,3% → 28,1% → 27,4%. Mẫu được chấm khác nhau giữa cấu hình, nên trung vị giảm không chứng minh khôi phục ID hoặc tracking tốt hơn. Khoảng frame có thể gồm bỏ sót detector, frame skip hoặc đứt track; không đồng nghĩa che khuất vật lý.

Phát lại tuần tự cùng đoạn MVI_20052 (32 frame khởi động, 128 frame đo × bốn lượt, một luồng) cho trung vị `tracker.update` lần lượt **10,11 / 10,18 / 10,40 ms/frame**. Phép đo không gồm detector, đọc cache và ghi MOT. Không có lợi thế tốc độ thực tế cho hai biến thể này.

## Quyết định

Giữ roughening hộp ở 1,0, giá trị mặc định hiện hành. Giảm nhiễu hộp không khắc phục khoảng cách PF–CV/EKF trên dev12, trong khi giảm DetA/MOTA. Không thay đổi cấu hình benchmark hoặc kết quả luận văn đã khóa. Nếu nghiên cứu tiếp, cần khảo sát số tổ tiên particle thực sự sau resampling và hướng xử lý đa dạng chuyển động; chỉ số `n_unique` hiện đếm tọa độ sau jitter nên không đo được số tổ tiên.

Bằng chứng: `results/phase2/kc48_r71_pf_dev/analysis/summary.csv`, `paired_intervals.csv`, `validation.json`, `prediction_summary.csv`; `handoff/logs/review71_pf/serial_speed.json`. Mã và cấu hình: `src/pf_ctrv.py`, `src/tracker_ctrv.py`, `configs/phase2/trackers/r71/`, `configs/phase2/experiments/kc48_r71_pf_dev.yaml`.
