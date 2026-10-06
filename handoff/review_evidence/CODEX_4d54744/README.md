# Bằng chứng review vòng 42 — `4d54744`

Kết luận và phạm vi ở `CODEX_REVIEW_4d54744.md`. `training_audit.json` có 128 phép đối chiếu hồ sơ, không phải suite training.

Đầu vào cần có: gói delta **741.857 byte**, SHA256
`f144a834607324b6a94197612d7851ec0e28137bdf85f4a2b8efc43a0d2b92df`.
Gói đầu vào không lặp lại trong ZIP bằng chứng để tránh nhân đôi dung lượng.

Từ thư mục đã giải nén gói bằng chứng, chạy bằng Python 3.8+ và Git:

```text
python check_package.py --zip /duong/dan/review_delta_latest.zip --out replay/evidence/package.json
python audit_training.py --tree replay/tree --base replay/base_preimages --out replay/evidence
```

Trên Windows, thay đường dẫn ZIP bằng đường dẫn thật có quote nếu có dấu cách. Hai lệnh chỉ ghi vào thư mục `replay/` mới bên cạnh, không sử dụng repo làm việc. `check_package.py` kiểm hash ZIP trước rồi tái tạo đường dẫn thay đổi. Nó không có toàn bộ Git base nên không chứng minh toàn cây target. Không dùng Python `-O` vì checker dùng assert.

Kết quả kỳ vọng:

- `package.json`: 107 payload + 6 alias; roundtrip 113/113, 8 preimage hash khớp; pass true.
- `training_audit.json`: `checks_passed = checks_total = 128`; 30 epoch hoàn tất, resume sau 7 và 11; best epoch 15 và 13; tổng runner 58,9848 h. Năm ô làm tròn khác và hai giá trị minimum RAM được liệt kê riêng.
- `epochs_reconciled.csv`: bảng 30 epoch tính lại từ nhật ký, đối chiếu với CSV và JSON runner.

`source_excerpts.md` chép các đoạn nguyên văn có số dòng từ file đầu vào; dùng để tìm nhanh ba mục đính chính. Nó không thay thế hash/nguồn đầy đủ. `input_inventory.csv` khóa các file đầu vào của lần review này theo byte thực được giải nén/tái tạo; không chứa hash tự suy ra cho checkpoint thiếu.

Không chạy lại train/inference, không torch/Ultralytics, không nạp `.pt`, không SSH Nano. Các hash `.pt`, cờ nạp lại, thống kê tài nguyên tổng hợp là khai báo trong bằng chứng tác giả. CSV RAM thô và dataset không có trong đầu vào.

`SHA256SUMS.txt` bao phủ mọi file khác trong ZIP bằng chứng. Ví dụ kiểm tại thư mục giải nén: `sha256sum -c SHA256SUMS.txt`.
