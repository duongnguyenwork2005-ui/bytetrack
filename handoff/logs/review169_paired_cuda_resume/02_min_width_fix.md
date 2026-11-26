# Sửa kiểm tra kênh tối thiểu trước export và trước điểm GPU ứng viên

build_cpu.py tạo head50 và đạt oracle, sau đó dừng đúng assertion(8,8)
ở body50: một Bottleneck vốn chỉ có8kênh. Rule00 giữ tối thiểu8 nên
trường hợp này phải là no-op, không phải lỗi cấu trúc.

Giữ nguyên source/log/inputs.json và head50_init.pt của lần1. Bảnv2 tạo
inputs_v2.json,model/tmp subfolderv2, giữ toàn bộ tolerance và chọnchannel;
nếuwidth đã bằngminimum thì ghi no-op và khôngcắt. Không chỉnhmask/oracle
để épPASS, không thaycấu trúchead50 hay rule chọnGPU/training. Lần1 chưa
export/build/đoGPU hoặctrain ứng viên. Không dùngpartial làm kết quả.
