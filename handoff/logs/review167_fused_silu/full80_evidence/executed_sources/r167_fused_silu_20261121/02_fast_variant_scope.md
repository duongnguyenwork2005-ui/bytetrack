# Hai kernel trước mọi điểm engine/timing/AP

Trước khi v2 compile, thêm đúng một biến thể fast: __expf thay expf trong
sigmoid float32. Namespace/plugin type riêng, không sửa hoặc thay thế precise.
Đây là approximate floating-point exponential, không gọi bit-exact hoặc QAT.
Cùng oracle F32/HALF, format/rounding,10raw,7190AP,guards,caps và GPU-only8%.
Khóa hai source trước điểm. Cả hai ứng viên đạt GPUgate thì đều chấmactualAP;
không chọn theo seed/class/video thắng, không tune coefficients sau điểm.
Không candidate nào đạt thì đóng negative screen. Max2actualAPcandidate.
Giữ original deployed và matched rebuild để phân biệt fusion với build drift.
