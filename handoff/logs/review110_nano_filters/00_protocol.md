# So sánh IDSW của các bộ lọc với detector Nano R109

- 40 video BDD100K val và 40 video UA-DETRAC test, đúng các đoạn 150 JPEG của R109. Lưu detection từ bản Nano tối ưu bằng `--save-cache`, giữ lệnh R109 ngoài đường dẫn tạm và tùy chọn lưu cache. Kiểm từng ảnh, engine và hash đầu ra KF với R109.
- Phát lại cache trên PC bằng NumPy, giữ nguyên ngưỡng, lọc/ánh xạ lớp, khoảng mất track 1,2 giây, FPS nguồn và cách tính bước thời gian theo R109. Kiểm kết quả KF trên PC so với KF R109 ở cấp frame, ID, lớp, độ tin cậy và hộp (sai số tối đa 0,001 pixel). So KF, EKF-CTRV, UKF-CTRV và PF500 K1/K2 motion2. PF dùng seed 0, 1, 2.
- Báo IDSW, FN, IDF1 và HOTA theo mỗi tập. IDSW của PF báo từng seed và trung bình. Hai bộ dữ liệu giữ riêng. Lượt này đo chất lượng tracking, FPS Nano vẫn thuộc lần đo R109 với KF.
