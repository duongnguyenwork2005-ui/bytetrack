# R169 — paired recovery10epoch chưa hoàn thành

Hostsession86330 thoát1 lúc17:04 ngày21/11/2026, giờ Việt Nam. Control hoàn
tất hai epoch resumed7/8 (zero-based), dừng trong epoch9 sau777batch đã
ghi progress, CUDNN_STATUS_EXECUTION_FAILED tại cudnn_batch_norm_backward.
CUDA_LAUNCH_BLOCKING=1 đã bật cho cả arm. Joint50 resumed chưa chạy.
`control_failed_cpu_batch.pt`2.953.176byte và traceback được giữ nguyên.

Windows có nvlddmkm13/153 cùng thời điểm; EventData ghi Graphics SM Warp
Out Of Range Address. Chưa biết root cause. Không quy lỗi cho operation
cuối, nhiệt độ hoặc driver chỉ từ những event này. Không có bằng chứng OOM
trong traceback này. Không thay driver/registry/TDR hoặc đóng app người dùng.

Hostsession28359 tự dừng do phát hiện failedbatch trước exports. Không có
R169 runtime build/capture/FPS hoặc actualAP, không có complete10 được
nghiệm thu. Các helper runtime chỉ được chuẩn bị/compile. Những checkpoint
control resumed7/8 là bằng chứng đã tính toán, không ghép với student cũ để
gọi đây là paired10. Control R164 đã từng hoàn tất10, nhưng tail đó đã bị
loại khỏi kế hoạch R169 trước điểm, không lựa chọn giữa hai control theo AP.

Chi phí tính toán thực gồm cả tail control cũ, phần epoch student R164 lỗi,
hai epoch resumed và phần epoch10 control R169 lỗi. Không coi các batch
bị bỏ này là chưa từng tính toán. Lịch sử và source/contract cả hai lượt
giữ nguyên, không overwrite và không tự restart lặp lại cùng kế hoạch.

R171 fresh-process256 KD forward/backward trên batch đã lưu đạt ở cuDNN
và nativeBN; không tái hiện lỗi batch đơn, không chứng minh đã sửa lỗi dài
hạn. R172 là phân tích phụ riêng, dùng cố định checkpoint6 của cả hai arm
R164 (7epoch đầu liên tục), không dùng bất kỳ tail resumed nào, không
huấn luyện mới hoặc chọn epoch theo AP. Primary10 vẫn chưa hoàn thành.

Nano R167 độc lập đã đủ320lượt,48.000inference và hậu kiểm thành công.
R169 thất bại không được giấu bằng kết quả tăng FPS của nhánh CUDA khác.
