# R171 — tái hiện batch lỗi, chưa khởi động lại huấn luyện

R164 illegalmemoryaccess và R169 cudnn_batch_norm_backward executionfailed
đều trùng event nvlddmkm13/153. Blocking1 chưa ngăn lỗi. Không kết luận
driver, nhiệt độ hay BatchNorm là nguyên nhân chỉ từ operation cuối.
Giữ nguyên hai study thất bại và batch CPU R169, không ghi đè hoặc chọn
checkpoint/AP có lợi hơn. Queue runtime R169 đã dừng do failed batch.

Fresh process, cùng batch CPU đã lưu, model control checkpoint8 R169 và
joint50 checkpoint6 R164. Mỗi model chạy 64 KD forward/backward, không
optimizer step, không save weights, không AP/epoch. Kiểm finite input/loss/
gradient và CUDA synchronize. Bốn process tuần tự: control cuDNN/nativeBN,
joint50 cuDNN/nativeBN; source ghi rõ backend và grad_fn thực. NativeBN chỉ
vô hiệu cuDNN trong forward BatchNorm để autograd tạo native backward;
convolution vẫn dùng backend cũ. Đây là diagnostic, không phải phép so
chất lượng hoặc cách âm thầm tiếp tục thí nghiệm.

Không đổi driver, registry, TDR, clock hoặc đóng ứng dụng của người dùng.
Nếu diagnostic đều pass, chỉ loại được một số lỗi tái hiện trên batch đơn;
không chứng minh đã sửa lỗi dài hạn. Một kế hoạch paired recovery mới sẽ
cần khóa trước điểm nếu đổi backend/budget, không sửa R169 đang lưu vết.
Nano R167 tiếp tục riêng; không mở công việc GPU trên Nano. Tối đa3 job.
