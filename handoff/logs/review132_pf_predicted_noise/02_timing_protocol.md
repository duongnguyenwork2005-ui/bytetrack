# Timing PC sau xác nhận chất lượng

Khóa trước đo. Chỉ đo sau quality jobs đãkếtthúc; không lấy thời gian concurrent
làm benchmark. Hai prefix200 đãchọn theo workload củaR125: b1f4491b-846d8cb2,
b1fc95c9-cb2882c7. Cùngcache/5classes/c25/FPS5/buffer6/steps. Bỏ16khungđầu,
3lần đảo thứtựlần2, sáumode KF/EKF/UKF/PFcổđiển500/native100/R132100,36lượt.

Đọc t_update_ms, báo mean vàp90 từnglượt, tổnghợp sáucặp. R132/native/controls
prefix phải khớp MOTfull củaR132/R125. PFcổđiển soMOTR74nếufilecòncục bộ;
tấtcảmode lặp seed0/mảngphảitrùng. Khóa source/hash/config trước/sau đo.
Đây là trackerCPUtime của2clipPC, không FPSNano hoặc FPS toànpipeline hay
đo hiệu năng full200/40video. Không chỉnhfilter sautiming đểcứu kếtquả.
