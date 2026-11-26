# Sửa assert định dạng trong bộ kiểm chứng PC

Lượt score_full đầu dừng ở assert10cột. Nguồn project không đổi khai báo
TRK_COLS đúng7cột [frame,id,left,top,width,height,conf], cũng là định dạng
của MOT R109 đã nghiệm thu; TrackEval đọc trực tiếp7cột. Đã đối chiếu
dòng nguyênbyte và khai báo trong nguồn thực thi đã lưu.

Giữ score_full_v1.py và score_full_v1.log. Bản kiểm chứng dùng đúng7cột,
vẫn yêu cầu finite/positive-size/unique-frame-ID và toàn bộ hash/timestamp/
configuration/evaluator checks. Không sửa/đệm lại bất kỳ MOT hoặc đổi
metric, không chạy lại Nano để né lỗi bộ kiểm chứng.
