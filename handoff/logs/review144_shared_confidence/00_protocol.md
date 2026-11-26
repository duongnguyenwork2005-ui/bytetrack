# R144 — cùng nhiễu đo theo confidence cho bốn bộ lọc

Yêu cầu19/11: trước hết BDD/UA, EKF và UKF phải ít IDSW hơn KF; PF phải
<=cả EKF lẫn UKF mới được mở dataset kịch bản. Các mốc cũ giữ nguyên.
UA test40 fixed-R hiện có KF2498/EKF2525/UKF2543 nên chưa đạt điều kiện mới.

Thử trước một thay đổi chung, đã có cơ sở và công thức trong repo:
truyền confidence của detection đang matched vào update của KF/EKF/UKF,
giống R142 đã làm cho PF. R(c)=max(1-c,0.05)R0. Không thay Q, khởi tạo,
association, detector, ngưỡng, lifecycle hoặc metric; không làm yếu KF để
đạt thứ hạng. Cả bốn nhận cùng cơ chế NSA, nên so sánh công bằng hơn R142.
Đồng thời giữ đầy đủ fixed-R controls để tách tác động noise adaptation.
Confidence phân lớp chưa phải localization uncertainty đã hiệu chỉnh.

Adapter nghiên cứu riêng chỉ bọc update/re_activate trong score context;
mọi track vẫn chia sẻ cùng filter instance, giữ nguyên multi_predict và
logic baseSTrack. Context phải restore cả khi có lỗi/nesting; không dùng
score cũ. Không sửa src/default/Nano/khóa luận.

Trình tự: numerical + hash + smoke -> UA train dev12 -> BDD val200 ->
UA test40 -> nếu **cả hai tập đầy đủ** đạt mới mở các dataset kịch bản.
Không mở kịch bản vì chỉ thấy một seed/video có lợi. Dùng cùng cache,
FPS/buffer/classes/c25/elapsed-step/evaluator như R142. PF R142 giữ nguyên,
dùng lại36dev+600BDD+120UA sau xác minh hash và smoke prefix/repeat.

Cổng trong từng tập: IDSW(EKF)<IDSW(KF), IDSW(UKF)<IDSW(KF),
meanIDSW(PF)<=min(IDSW(EKF),IDSW(UKF)), ít nhất2/3seedPF<=min đó.
Mỗi filter giữ HOTA>=fixed-R counterpart-0.001 và FN<=1.002 counterpart;
PF so với R142. Không giảm IDSW bằng cách bỏ thêm nhiều đối tượng.
Nếu UAdev không đạt thì đóng R144, không dò floor/exponent/gain để cứu.
Các tập đã quan sát nhiều vòng, không gọi test mù hay ưu thế phổ quát.

Khóa code/config/data/GT/seqinfo/evaluator/backend thực nạp. Numerical
direct-score parity/PSD/zero-confidence/context exceptions; smoke hai
video đầu/cuối mỗi tập:3deterministic controls routing-off khớp historical,
routing-on lặp; PF3seed khớpR142. Full lưu MOT/CLS/timing/marker, tái hợp
COMBINED/raw khớp. Tối đa3 job bền vững, kể cả job training GPU nếu có.
Không benchmark thời gian khi chạy quality/training đồng thời.

Nguồn: [StrongSORT](https://arxiv.org/abs/2202.13514) và công thức NSA trong
`third_party/ultralytics_trackers/utils/kalman_filter.py`, `src/ekf_ctrv.py`.

Nano hiện tắt theo người dùng; mô hình YOLO chuẩn bị/đánh giá trên PC trong
nhánh riêng. Không gán FPS PC hay FLOPs thành FPS Nano.
