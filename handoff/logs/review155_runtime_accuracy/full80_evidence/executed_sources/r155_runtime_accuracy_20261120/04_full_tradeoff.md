# Đủ điểm tốc độ và chất lượng trên80clip

Bổ sung khi pilot đang chạy, trước khi đọc bất kỳ điểm pilot nào để chọn
mô hình full. Không thay APgate, metric, data hoặc số đo đã có.

Giữ rule01: chọn candidate có meanFPS pilot cao nhất trong nhóm đạt AP
và nhanh hơn original; nếu không có, chọn candidate pilot nhanh nhất làm
diagnostic, không promote khiqualityFAIL. Full80 luôn có original ghépcặp.

Để không bỏ đường trade-off khi một engine rebuild đạt AP nhưng chỉ nhỉnh
FPS do nhiễu pilot, bổ sung tối đa **một candidate qualityFAIL** có pilot
meanFPS cao nhất và nhanh hơn original, nếu khác candidate đã chọn. Như
vậy full tốiđa3mode×80clip, giữcùngmanifest vàrotationorder. Tie xử lý
theo tênmodel; ghi toànbộ pilotmeans và selection.json trước full.

Đo diagnostic không đổi statusFAIL hoặc cho phép triểnkhai. Báo FPS
ghépcặp theo tập/toàn80, AP5lớp và HOTA/IDSW/FN/FP cửa sổ150ảnh.
R145pruning và R160352x640 vẫn là batch riêng, chưa có quyết định promote.
