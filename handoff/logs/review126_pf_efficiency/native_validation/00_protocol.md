# R126-A2 — lõi C/float64, trước kết quả tracking/tốc độ

R126-A1 vector hóa/cache giữ số học nhưng chỉ giảm8,7%update trên mẫu, trượt cổng20% và còn chậm hơnPF cổ điển. Giữ nguyên mọi file/kết quảA1, không chọn.

A2 dùng MSVC BuildTools2022 đã có trênPC, không cài package hoặc đổi môi trường chung. Ckernel làm predict Gaussian4D, update vị trí2D bằng solve chuyên biệt và Joseph covariance, hai khối shapeGaussian, tổng hợp within-covariance. Giữ NumPy RNG, thứ tự draw/resampling, chuyển độngF, tính trọng số/logsumexp và mixturemean nhưA1/R118. float64, /fp:strict, không fast-math. Không thay số hạt100, Q/R/init hoặc association. Đây là tối ưu cài đặt, không phải thuật toán tracking mới.

Numerical720bước trên3seed, near-singular fallback, PSD, RNG và ownership qua bộ kiểmR126; sai lệchstate~3e-13. Khóa sourceC/Python/DLL/buildmeta trước smoke/timing. Smoke12lượt soR125; đo tuần tự18lượt parent100/native100/PF cổ điển trên cùng hai mẫu đãkhóa; cổng20%/khôngchậm hơnPF cổ điển giữ nguyên. Nếu đạt, chấmBDD200×3seed và đối chiếu mọi rawfield/ID/MOT vớiR125. Nếu có khác biệt, báo đúng, không chọn chỉ vì điểm tình cờ tăng. Chưa đoARM/Nano, không suy FPSNano từ PC.
