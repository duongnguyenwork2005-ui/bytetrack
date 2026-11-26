# Danh mục ZIP lớn hơn một response metadata

V1 `list_zip.py train1.zip` đọc đúng 98 byte EOCD/ZIP64 từ ba Range, nhưng
dừng do central directory lớn hơn 2 MiB. Giữ nguyên source và
`train1.zip.index.json` STOPPED. Không có ảnh/nhãn nào được tải ở bước này.

V2 chia các lần đọc danh mục thành response tối đa 2 MiB như protocol,
vẫn tổng tải tối đa 20 MiB và tối đa 20 request/archive. Mọi response phải
là206, Content-Range chính xác và ETag giống metadata. Nếu không đạt thì
dừng, không tải cả archive. Sửa transport cho metadata, không đổi tiêu chí
chọn dữ liệu, không có điểm tracking/detection để lựa chọn.
