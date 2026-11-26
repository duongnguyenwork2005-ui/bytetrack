# Chất lượng detector đã pruning trên cùng đường tiền hậu xử lý

Nhánh pruning có mAP50 **43,87884% ở PyTorchFP32** và **43,86807% ở
TensorRTFP16 trên Nano**, cùng7190 ảnh và hình học384×640. Mức giảm
so matched dense control đã xuất hiện ở FP32; chuyển sang FP16 chỉ đi
kèm chênh khoảng−0,01077 điểm AP50 trong phép so sánh này.

## Kết quả đủ7190 ảnh cho mỗi model

- Anchor: PCFP32 AP50/AP50–95 **47,05447/31,97891%**;
  NanoFP16 **47,14673/31,98483%**.
- Control đã huấn luyện cùng ngân sách: PCFP32 **46,13403/31,12986%**;
  NanoFP16 **46,17151/31,12498%**.
- Pruning+kèm distillation: PCFP32 **43,87884/29,31515%**;
  NanoFP16 **43,86807/29,28391%**.

Pruned−control là−2,25519 điểm AP50 ởFP32 và−2,30343 điểm ởNanoFP16.
So anchor, tương ứng−3,17562 và−3,27865 điểm. Vì vậy, phép kiểm hiện tại
không ủng hộ cách giải thích rằng toàn bộ giảmAP là do FP16.

Standard validation trongR145 cho pruned48,0300%mAP50. Phép đó dùng
hình học/đường postprocess khác; không trừ trực tiếp48,03−43,87 rồi gán
toàn bộ chênh lệch cho precision. Audit này dùng evaluatorR151 đã khóa,
argmax/NMS/letterbox384×640, cùng GT/matching/AP với đường Nano.

## Phạm vi bằng chứng

Control và pruned chấm mới đủ7190 ảnh/model; anchor tái dùng kết quả
FP32R151 đã lưu. Checkpoint đúng SHA củaR145, source/dataset/model hash
được kiểm trước/sau. comparison.json kiểm lại các binding và đối chiếu
với ba bộ quality_verification.json của actual Nano engines.

PC PyTorch/OpenCV và Nano TensorRT/OpenCV còn khác backend và phiên
bản OpenCV. Đây là đối chiếu đường thực thi, chưa là ablation chỉ đổi
FP32→FP16 trên cùng backend. Raw parity FAIL của pruned vẫn giữ nguyên;
AP gần nhau không sửa lại kết luận của kiểm raw trước đó.

Không có số FPS mới trong audit này. Tiếp theo là đo nhánh pruning đã
có trên Nano theo cùng pipeline và80 đoạn nếu qua pilot tốc độ. Chưa
huấn luyện/cắt thêm model trước khi xác định lợi ích tốc độ thực tế.
Các nhánh thử nằm riêng, chưa đổi model sử dụng, launcher hay khóa luận.

Nguồn: comparison.json, comparison.log, inputs.json,
quality/control/summary.json, quality/channel25_KD/summary.json;
R150cache anchor/control/channel25_KD và R151quality/baseline.
