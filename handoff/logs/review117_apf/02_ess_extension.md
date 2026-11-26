# APF với ESS

Hai ứng viên đầu đủ dev và APF+K2 đủ FastTrack. IDSW FastTrack tăng 1.277→1.282,67 nên kết quả chưa chuyển miền.

Thử một biến thể cuối trên dev12, ba seed. Giữ nguyên ngưỡng ESS/N=0,5 hiện hành. Nếu posterior có ESS cao hơn ngưỡng, thực hiện K1 hiện có như nền và không resample thêm. Nếu thấp hơn ngưỡng, dùng APF+K2 đã thử. Giữ các tham số khác. Mục tiêu là thử ảnh hưởng của lịch resampling mà không dò ngưỡng ESS hay mức nhiễu. Nếu ba seed đều giảm IDSW, HOTA không giảm quá 0,5 điểm phần trăm, FN tăng tối đa 0,5%, thử chuyển miền với tham số cố định. Cổng mục tiêu 10% vẫn báo riêng như hai ứng viên đầu.
