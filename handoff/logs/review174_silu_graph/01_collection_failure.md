# Lỗi đóng gói sau khi screen đã hoàn tất

screen.py đã ghi R174 GPU SCREEN COMPLETE và complete.json12lượt.
Lệnh tar đầu ghi archive ngay trong thư mục đang duyệt bằng dấu chấm:
`tar -czf evidence.tar.gz --exclude=evidence.tar.gz .`
trả exit1 với `tar: .: file changed as we read it`. Hostrun.ps1 dừng trước
fetch, không phải lỗi inference/capture. Giữ archivev1 và toàn bộ log.

collect_v2.ps1 chỉ đóng gói danh sách tệp bằng chứng cố định, không duyệt
thư mục gốc hoặc đưa archive vào chính nó; không chạy lại benchmark hoặc
đổi source/config/engine/threshold. Kiểm cả12log/hash/guard trước kết luận.
