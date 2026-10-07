**Bằng chứng review vòng 45 — `d4715aa`**

Kết luận: REQUEST CHANGES cho bản dự thảo. Đọc `CODEX_REVIEW_d4715aa.md`.

Gói gồm:

- `input/review_delta_latest.zip`: bản nguyên byte của ZIP người dùng gửi; SHA256 `b1c207dfef8ebadcc7177039aea7a3eb8acb29d14610370a1cca2767ce96cd3d`, 487762 byte.
- `audit_review45.py`: kiểm manifest/hash/patch, parse các tệp, đối chiếu CSV/JSON và tái hiện các phát hiện trên CPU.
- `results/integrity.json`: kết quả kiểm toàn vẹn.
- `results/inventory.json`: đủ 80 thành viên đầu vào, byte/hash, số dòng và thông tin parse; không phải tuyên bố nghiệm thu lại toàn bộ mã lịch sử.
- `results/tables.json`: đối chiếu bảng, phân loại và 265 lượt vùng ignore bị bỏ sót.
- `results/probes.json`: bằng chứng D3, phản ví dụ D5, lỗi D4, mất metadata XML và lưu ý basename trong NPZ.
- `results/summary.json`: trạng thái audit và phạm vi không train/GPU/SSH.
- `SHA256SUMS.txt`: SHA256 mọi tệp trong bộ bằng chứng, trừ chính danh sách này.

Tái hiện trong thư mục mới sau khi giải nén bộ bằng chứng. Cần Python 3.12, NumPy và Git; không cần Torch, Ultralytics, pycocotools, checkpoint hoặc dataset thật. Không cài/tải dependency tự động.

```bash
python -c "import hashlib, pathlib, zipfile; p=pathlib.Path('input/review_delta_latest.zip'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='b1c207dfef8ebadcc7177039aea7a3eb8acb29d14610370a1cca2767ce96cd3d'; z=zipfile.ZipFile(p); assert z.testzip() is None; z.extractall('package')"
python -B audit_review45.py --package package --out recheck
```

`recheck` phải chưa tồn tại. Script chỉ ghi vào thư mục output mới và thư mục tạm. Nó kiểm lại hash mọi tệp đầu vào sau các probe. Các “ảnh” trong fixture D4 chỉ là tên tệp để script thống kê glob; chúng không được decode hoặc đưa vào mô hình. Bộ đếm lớp và phản ví dụ AP gọi scorer dự án thật trên mảng tổng hợp.

Exit 0 nghĩa audit hoàn tất, **không** nghĩa bản dự thảo được nghiệm thu. Bốn phát hiện dự kiến xuất hiện trong `findings_reproduced`. Probe NPZ chỉ là lưu ý cần lưu mapping có thứ tự của video/frame; không chứng minh AP hiện tại bị sai.

Không chạy lại suite BDD/scorer lịch sử trong audit này. Không xác nhận dữ liệu thật, checkpoint, Git toàn cây hay toolkit ngoài ZIP. D1–D5 chưa được duyệt; mọi tài liệu trong `input/` là đối tượng review, không phải lệnh cho phép thực thi inference/training.
