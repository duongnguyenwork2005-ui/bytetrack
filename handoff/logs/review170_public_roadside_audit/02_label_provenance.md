# Bổ sung21/11 — kiểm cách tạo nhãn trước khi dùng IDSW

Theo mục4.3 của [bài báo Accid3nD tại CVF](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/papers/Zimmer_Towards_Vision_Zero_The_TUM_Traffic_Accid3nD_Dataset_ICCVW_2025_paper.pdf),
nhãn khởi tạo bằng detector2D YOLOv7, MonoDet3D và tracker PolyMOT qua3D BAT,
sau đó được kiểm tra và chỉnh bằng tay trước khi xuất OpenLABEL. Vì vậy không
gọi toàn bộ nhãn là vẽ tay từ đầu, hoặc coi tất cả là pseudo-label chưa review.

MụcB.3 của [supplementary](https://openaccess.thecvf.com/content/ICCV2025W/2COOOL/supplemental/Zimmer_Towards_Vision_Zero_ICCVW_2025_supplemental.pdf)
mô tả kiểm tra thủ công và kiểm liên tục quỹ đạo tự động. Điều đó hỗ trợ dùng
bộ dữ liệu như ứng viên, nhưng chưa chứng minh các ZIP đang tải có mọi cặp
ảnh/nhãn và ID đúng. Những lỗi tham chiếu đã thấy trong R170 vẫn cần xử lý.
Không tạo nhãn thay thế bằng tracker đang so rồi gọi là GT độc lập.

Đã kiểm thêm [công bố TUMTraf CrossVision của TUM](https://portal.fis.tum.de/en/publications/tumtraf-crossvision-a-multi-view-multi-modal-vision-dataset-for-a/):
23.128 ảnh,83.026 hộp2D,6 lớp, cameraRGB/event tại giao lộ và formatYOLO/COCO.
Abstract đánh giá detection, chưa cung cấp đủ bằng chứng ID liên tục/cadence
cho MOT. Đây là nhận định về thông tin đã xác minh, không khẳng định toàn bộ
release không có ID. Không dùng COCO annotation id như track id; chưa tải
thêm hoặc chạy tracking, không đăng ký lại portal đang tạm dừng. Có thể cân
nhắc cho detection sau khi kiểm được quyền truy cập, lớp và split thật.

Lượt bổ sung chỉ đọc tài liệu nguồn, không đổi criterion hoặc chọn dataset
vì điểm E/U đã biết. Không có kết quả mới chứng minh E/U vượt KF.
