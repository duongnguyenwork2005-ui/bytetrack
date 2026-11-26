# Sửa bridge tên trường, giữ nguyên metric và output gốc

V1 dừng ở video đầu sau khi khóa inputs.json: raw_fields.json dùng CLR_FN,
không có key FN. IDSW và trace đã dùng field đúng; chưa có summary nhóm.
Giữ run.py/run.log/inputs.json. V2 chỉ đọc expected['CLR_FN'], thêm hash của
ghi chú này, dùng inputs_v2/partial_v2/summary_v2 và clean_gt_v2 riêng.
Không sửa tracker, raw fields, cấu hình, CLEAR trace, GT hoặc nhóm kịch bản.
