# Bằng chứng review `a03f6e1`

Đọc `CODEX_REVIEW_a03f6e1.md` trước. Đây là review chưa hoàn tất vì thiếu các mô-đun evaluator không đổi trong gói delta hiện có.

## Chạy lại

Python 3 + numpy đủ cho các script reviewer (script arithmetic/package chỉ dùng thư viện chuẩn và Git cho package).

```bash
python audit_results.py input --out results_audit_recheck.json
python REPRO_REVIEW_a03f6e1.py input --out repro_recheck.json
python check_package.py --zip /duong/dan/review_delta_latest.zip --out work/evidence/package_recheck.json
```

- `audit_results.py`: 180/180; chỉ đối chiếu số học/metadata, KHÔNG tính AP từ detection/GT thật.
- `REPRO_REVIEW_a03f6e1.py`: exit 1, bug_count=1 trên bản bàn giao. Dùng entry point thật với dependency doubles công khai; chỉ kiểm cấu hình ghi ra, không kiểm AP/GT/YOLO.
- `check_package.py`: cần ZIP người dùng gốc, SHA đã khóa trong script. Chỉ xác minh đường dẫn thay đổi, không tái dựng toàn repo độc lập.
- `test_attempt.log`: test dự án bị chặn khi import `bdd_yolo_export`. Không đọc dòng 0/1 FAIL thành một lỗi thuật toán đã tái hiện.

`input/` giữ nguyên byte các đầu vào nhỏ của audit và hai file mã mới; không phải repository chạy đầy đủ. Các dependency thiếu được liệt kê trong báo cáo. Không gửi `.pt`, ảnh, nhãn dataset hay NPZ.

`results_audit_first_attempt.json` được giữ để minh bạch lỗi của script reviewer: giả định `ctrl_c` boolean trong khi schema là danh sách sự kiện. Bản cuối kiểm đúng `[]`, không sửa nguồn. Một lượt trung gian còn dùng phép so với số 0 cũng không đúng; sau khi đọc schema đã sửa dứt điểm.

`SHA256SUMS.txt` băm mọi file trong ZIP trừ chính danh sách hash.
