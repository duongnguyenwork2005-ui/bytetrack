# R124 — kiểm PF100 trên camera gắn xe KITTI

Sau R123 qua dev UA và chuyển SVMOT, kiểm thêm toàn bộ20 chuỗi Car hợp lệ KITTI của R119, không chọn chuỗi theo điểm PF. Giữ Cartesian PF100, ba seed, không chỉnh tham số sau khi mở kết quả; không thêm gate lựa chọn theo test. Cùng cache COCO c25 và evaluator KITTI,10Hz,buffer1giây,elapsed steps như ba bộ lọc R119. Không thử PF500 mới trong vòng này nên không suy ra hiệu ứng riêng N100→500 trên KITTI.

Khóa ảnh/cache/nhãn/mã và kết quả KF/EKF/UKF tham chiếu trước điểm. Mỗi seed có workspace evaluator riêng để tránh race file seqmap, mỗi chuỗi khởi động tracker lại. Chấm đủ60 lượt PF mới, tái ghép CLEAR IDSW/FN cho nhóm all/turn25/nativeocclusion2/giao nhóm, ba seed và raw metrics, báo mọi chuỗi. Kiểm toàn bộ hash nguồn/GT/cache/MOT, không dùng timing chạy đồng thời để báo tốc độ. Không chỉnh mặc định, detector hoặc khóa luận.
