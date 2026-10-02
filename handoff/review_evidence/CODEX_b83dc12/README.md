# Bằng chứng review b83dc12

Đọc CODEX_REVIEW_b83dc12.md trước. Không có dataset, ảnh hay checkpoint.

- check_package.py kiểm đúng ZIP vòng 38 và 118 đường dẫn thay đổi. Cần Git và Python chuẩn.
- check_round38.py đối chiếu log/receipt và số học; không train, không gọi Torch. Cần Python chuẩn.
- tree/ là 118 đường dẫn trong delta, không phải toàn bộ repo.
- prior/ gồm file split nguyên byte và dự toán từ những vòng reviewer đã đọc trước.
- base_preimages/ là sáu file đã bị sửa, được đối chiếu với bản lưu review trước; nguồn đối chiếu ghi trong evidence/base_independent.json.
- evidence/ chứa kết quả reviewer; delta_metadata/ là manifest/phụ thuộc của ZIP tác giả.
- tree/handoff/review_evidence/CODEX_0c0a07a/ giữ nguyên 91 file của bằng chứng reviewer cũ; chúng có SHA256SUMS riêng.

Lệnh (chạy tại gốc gói đã giải nén):

    python check_package.py --zip /path/review_delta_latest.zip --out evidence_lap_lai/package.json
    python check_round38.py --tree tree --prior prior --out evidence_lap_lai/receipts.json

Không cần quyền truy cập máy Windows để lặp lại các kiểm này. Tuy nhiên chúng không thay thế kiểm ảnh/nhãn thật trên Windows, cũng không xác minh toàn cây Git.

SHA256SUMS.txt khóa mọi file khác trong gói. Chạy các script trong thư mục bằng chứng riêng; không chép đè dataset/repo.
