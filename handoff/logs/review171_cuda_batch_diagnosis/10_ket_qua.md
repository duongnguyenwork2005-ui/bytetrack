# R171 — batch đơn không tái hiện lỗi; chưa xác định nguyên nhân

Đủ bốn fresh process control/joint50 × cuDNN/nativeBN, mỗi process64 KD
forward/backward, tổng256, finite loss/gradient và CUDA synchronize đạt.
Grad_fn đầu tiên xác nhận CudnnBatchNormBackward và NativeBatchNormBackward
thực, không chỉ đổi một flag rồi giả định backend đã đổi. Không optimizer
step, không save checkpoint hoặc mở AP. `verification.json` đã kiểm lại hash.

Batch CPU R169 có4×3×384×640uint8,31 GT, class0–4, bbox[0,1], không NaN.
Nguồn checkpoint controlR169epoch8 và jointR164epoch6 là trạng thái đã lưu,
không phải toàn bộ rawmodel/optimizer tại chính batch lỗi; vì vậy đây chưa
phải tái hiện đầy đủ trạng thái huấn luyện lúc xảy ra lỗi.

R169 dừng sau777batch epoch10 với CUDNN_STATUS_EXECUTION_FAILED tại
cudnn_batch_norm_backward, có nvlddmkm13/153 cùng thời điểm. Event ghi
Graphics SM Warp Out Of Range Address. Những dấu hiệu này không xác định
được nguyên nhân cuối là driver, hardware, kernel hay tương tác khác.
Không kết luận nativeBN đã sửa được lỗi training dài hạn từ64 batch.

Không thay driver, registry/TDR, clock, hoặc đóng ứng dụng người dùng. Giữ
R164/R169 và snapshot lỗi. SecondaryR172 dùng mốc7 epoch liên tục chung
của R164 để đo chất lượng/độ trễ của pruning, không gọi primary10 đã xong
và không chọn giữa các tail control/resume theo AP.

Tham chiếu quyền bật/tắt backend là [tài liệu PyTorch](https://docs.pytorch.org/docs/stable/backends.html).
Đã dùng API thực của môi trường2.6.0+cu124, không nâng thư viện theo tài
liệu stable mới. Không viện một GitHub issue không trùng lỗi làm bằng chứng
đã biết root cause. Nano R167 độc lập và hoàn tất đo/chấm bình thường.
