# Khôi phục training ngày20/11

Khi tiếp tục sau khi người dùng bậtNano, không còn processPython của
lượt19/11. Logcontrol dừng giữaepoch4; last.pt còn epoch=2 (đã xong3),
optimizer/EMA còn, history có3dòng. Không coi epoch4dở là đã hoàn tất.

Dùng resume chính thức Ultralytics với PrunedTrainer và cùng wrapper
KD R104, teacherFP32/hệsố/data/seed145/5epoch tổng giữ nguyên. Bảo lưu
best/last/history/args trướcresume và hashcheckpoint trong bản ghi riêng.
Train tiếp epoch4–5, không thêm5epoch, không sửa train.py/inputs.json hay
giao thức đã khóa. Ghi rõ resume chuẩn không đảm bảo RNG/trajectory bằng
một lượt chạy không gián đoạn; tham số train và ngân sách epoch vẫn như cũ.
Đối chứng và pruned vẫn cùng5200ảnh/5epoch/KD, nhưng chỉ control bị ngắt,
đây là hạn chế cần nêu khi so sánh nhỏ về AP. Không gọi kết quả test mù.

Sau đó chạy nhánh channel25_KD mới bằng train.py đã khóa, chấmdev7190,
xuấtONNX và đo Nano. Không thay mô hình đang triển khai trước chất lượng
và tốc độ thực tế. Nano đãSSH192.168.1.5, swap0 ở lúc kiểm tra; chưa có
FPS mới chỉ từ việc nối được thiết bị.
