# R127 — chỉ phân tầng nhiễu omega, khóa trước điểm

Tiếp yêu cầu cải thiện chất lượng/chi phí PF. Giai đoạn số học R126 không đổi mô hình; dùng lõi nativefloat64 R126-A2 đã kiểm numerical, và lượt này thử đúng một thay đổi thuật toán với100hạt: khi predict, thay100nhiễu Gaussian độc lập cho omega bằng100uniform phân tầng (mỗi ô1/N có một điểm ngẫu nhiên), inverse-normal CDF và hoán vị ngẫu nhiên trước gán cho hạt. Mỗi hạt vẫn có marginal Gaussian đúng; các nhiễu giữa hạt có tương quan. Không thay resampling, Q/R, init, số hạt hoặc association. Không gọi đây là SQMC đầy đủ và không tuyên bố bảo đảm giảm phương sai mọi đại lượng. Smoke lõi native trên UA phải khớp R123, nếu không dừng trước điểm ứng viên.

Tham khảo nguồn gốc ý tưởng giảm sai số Monte Carlo bằng phân tầng/QMC: Gerber & Chopin, Sequential Quasi-Monte Carlo, https://arxiv.org/abs/1402.4039; Douc/Cappé/Moulines, Comparison of Resampling Schemes for Particle Filtering, https://arxiv.org/abs/cs/0507025. Công trình sau so resampling với multinomial, không chứng minh thay nhiễu predict ở đây tốt hơn systematic baseline. Đây là giả thuyết cần đo.

Dev UA train12, cùng cache và seed0/1/2 R123,36lượt. Numerical kiểm finite, mean/variance nhiễu, lặp seed và khôi phục hook. Smoke Fast100 IID phải khớp R123 trước điểm ứng viên. Cổng mở BDD: HOTA trung bình không thấp hơn R123, IDSW không cao hơn458 và FN không cao hơn37.429,33; ít nhất một chỉ số cải thiện. Nếu trượt dừng, không đổi N/strata/cách hoán vị để cứu kết quả, không chạy BDD. Nếu đạt, giữ nguyên source chạy đủBDD200×3seed và so R125, dùng mot_crowd/cùng detection/FPS/buffer. Các tập đã được dự án xem nhiều vòng, không gọi test mù.

Đo tốc độ tuần tự sau chất lượng. Mọi kết quả, kể cả trượt, giữ log. Không đổi tracker mặc định, detector hay khóa luận.
