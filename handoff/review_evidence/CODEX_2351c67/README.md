# Bằng chứng review vòng 35 — 2351c67

Đọc CODEX_REVIEW_2351c67.md trước. Hai điểm R35-F1/F2 là giới hạn diễn giải; không phải yêu cầu mở thêm training.

## Kiểm split, log smoke và dự toán

Cần Python 3 và numpy; chạy từ thư mục đã giải nén gói bằng chứng:

    python check_round35.py --tree tree --accepted-label-manifest inputs/accepted_native_labels.tsv --out rerun_evidence
    python tree/handoff/logs/review35/test_do_tai_nguyen_ngoai.py
    python tree/handoff/logs/review35/15_du_toan_training.py --ra rerun_evidence/du_toan_script.json

Các lệnh trên không train, không cần ảnh/weights và không SSH Nano. Script kiểm tính nhất quán của bằng chứng bàn giao; không xác minh dataset vật lý trên Windows. Test công cụ log trên Linux dùng UTF-8, không tái hiện nguyên môi trường cp1252 của Windows.

## Kiểm delta gốc

Đặt review_delta_latest.zip đúng vòng 35 cạnh README (378.096 byte, SHA256 trong báo cáo), cần thêm Git:

    python check_package.py --zip review_delta_latest.zip --base-preimages base_preimages --out rerun_package

base_preimages chỉ có 6 file cần áp patch; không phải checkout đầy đủ. Không sửa Git config của dự án.

## Nội dung

- evidence/: kết quả độc lập của reviewer.
- tree/: bản sao log/script/split trong delta, giữ nguyên byte.
- inputs/: manifest nhãn đã được kiểm ở vòng trước; chỉ đường dẫn và hash, không chứa nhãn dataset.
- SHA256SUMS.txt: hash byte thô của mọi file, trừ chính danh sách này.

Không kèm ảnh, weights, video hoặc dataset. Không chạy các script chẩn đoán GPU trong tree/ nếu chưa có đầu vào/phạm vi được duyệt.
