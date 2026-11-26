# R137 — đo chi phí sau khi đạt chất lượng hai tập

Chỉ chạy sau khi mọi job chất lượng/bootstrap đã kết thúc và cả BDD, UA test40
đạt cổng khóa trước. Đo tuần tự trên PC, hai prefix200 theo workload50/90 R125,
bỏ16 frame đầu, seed0, ba lần đo; lần thứ hai đảo thứ tự bảy bộ lọc.
KF/EKF/UKF/PF cổ500/Native100/R132 parent100/R137 candidate100:42 lượt.
Cùng cache/classes/c25/FPS5/buffer6/elapsed steps. Kiểm output lặp và full
prefix parity; khóa hash trước/sau. Báo mean/p90; không suy thành FPS Nano.

R137 chỉ dời lúc gắn angular innovation sau selection, không giảm số hạt hay đổi model để cứu timing.
