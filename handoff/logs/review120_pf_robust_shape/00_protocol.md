# R120 — giảm tác động hộp kích thước bất thường, khóa trước điểm

Tiếp yêu cầu tự cải thiện PF ngày 19/11. Thử đúng một biến thể của Cartesian RBPF R118, 500 hạt, ba seed, UA train dev12 đã dùng phát triển nhiều lần. Không chọn tham số theo điểm test và không sửa mặc định.

Chỉ tăng R của hai phép đo aspect/height theo innovation chuẩn hóa riêng từng khối Gaussian. Với d² = residual²/(P00+R), hệ số max(1,(4+d²)/5). Đây là phép làm mềm xấp xỉ lấy cảm hứng Student-t ν=4, không phải posterior Student-t chính xác. Giữ likelihood vị trí, R tâm, chuyển động, init hai quan sát, Q và ghép ByteTrack nguyên R118. Khác R112 Student-t trên cả bốn tọa độ, không giảm ảnh hưởng tâm đo.

Kiểm CV/Gaussian bằng hệ số1, giảm gain khi hộp đo lệch lớn, PSD, khôi phục hook và lặp seed; đối chứng R118 hai prefix×3seed khớp MOT. Dev đủ12×3; so R118 và R112 đã lưu cùng đầu vào, tái tính raw fields, hash trước/sau. Cổng mở transfer: IDSW trung bình thấp hơn R118, ít nhất2/3seed giảm, FN không tăng quá0,5%, HOTA không giảm quá0,5 điểm phần trăm. Nếu trượt, dừng biến thể, không dò ν khác. Nếu đạt, giữ nguyên mã/ν cho SVMOT test9; không thay mặc định chỉ từ dev.

Không dùng thời gian các job đồng thời để tuyên bố tăng FPS. Không viết khóa luận.
