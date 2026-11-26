# Bổ sung sau profile baseline và trước mọi điểm ứng viên

Profile anchor41,66426ms, mappingmodel.index99,9935%. Detect23 chiếm
12,48624ms; C3k2 block2/4 là hai targetdepth theo rule00. Tổng các layer
Sigmoid12,69307ms vàact/Mul4,73794ms; đây là attribution tênlayer, không
coi là toàn bộ thời gian có thể xóa. Chưa export/build/chấm ứng viên.

Thêm đúng một phương pháp giản lược toán tử: hswish, thay mọi nn.SiLU
bằng nn.Hardswish, giữweights/ConvMAC/params/5lớp/3scale. Đây là thay
phi tuyến, không gọi là pruning params hoặc phép biến đổi giữ nguyên hàm.
Hàm tham chiếu x*clamp(x+3,0,6)/6; rawshape/finite và CPUORT/PyTorchparity
phải đạt. Không hứa AP hoặc tốc độ. Tổng5ứng viên; cùngscreen>=8%GPU,
selectionrule vàbudgettốiđa2student+matchedcontrol của00. Với hswish,
ConvMACgiảm0 nên có thể là cấu trúc ítcắt nhất nếuđạt tốcđộ.

Nguồn: [MobileNetV3 phần5.2](https://arxiv.org/html/1905.02244v5).
Không sao chép các con số speedup của MobileNet trên điện thoại sangNano.

Làm rõ chọn epoch trước training: lưu10checkpoint save_period1, chọn
epoch bằng AP50-95 trên **300monitor exactargmax/NMS384x640** (cùngđường
Nano); tie chọnAP50 rồi epoch sớm. Không lấy best mặcđịnh multi-label
để so cùng lúc với metric argmax. Chỉ mở7190dev sau khi chọn xong epoch.
Control vàstudent dùng cùng phép chọn và sốcheckpoint, ghi đủ10điểm.

Các guard/rawtolerance/qualitycap/final80rule của00 giữ nguyên. Protocol00,
bổsung01, profile và tất cả nguồn sẽ khóa hash trước tạo ứng viên.
