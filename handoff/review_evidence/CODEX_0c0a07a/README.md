# Bằng chứng review 0c0a07a

Kết luận và phạm vi ở CODEX_REVIEW_0c0a07a.md.

- tree/: mã, cấu hình pilot, log và receipt được lấy nguyên byte từ delta. Không phải toàn repo.
- base_preimages/: 9 file trước sửa. 7 file tài liệu còn có bản từ review trước; 2 file mã được dựng bằng patch ngược và kiểm hash khai báo.
- evidence/: kết quả độc lập. 468 điều kiện kiểm receipt không phải 468 lượt test huấn luyện.
- check_round37.py: tính lại từ CSV/JSON, chạy lại summarizer. Không tự tính AP từ ảnh.
- check_runner_isolated.py: 11 ca cấu hình/callback; dùng hàm mã gốc qua AST, fixture thay clock và CUDA availability.
- check_package.py: cần ZIP delta gốc 967546 B; kiểm 132 đường dẫn thay đổi, không kiểm toàn Git tree.

Chạy từ thư mục này (Python 3.12, numpy, pyyaml; Git cho kiểm gói):

```text
python check_package.py --zip <delta-goc.zip> --out evidence/package_rerun.json
python check_round37.py --tree tree --out evidence/independent_rerun.json
python check_runner_isolated.py --tree tree --base base_preimages --out evidence/runner_rerun.json
```

Không cần GPU/PyTorch/Ultralytics để chạy ba script trên. Không training, inference, SSH hoặc dữ liệu lớn.
Mã training trong tree chỉ để đọc; bằng chứng không gồm weights/dataset hay toàn bộ dependency của dự án.
