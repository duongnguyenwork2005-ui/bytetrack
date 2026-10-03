# Bằng chứng Codex vòng 40

Đọc `CODEX_REVIEW_421dd53.md` trước. Kết luận PASS WITH LIMITATIONS cho sửa R39-F1. Các lỗi CSV được xác nhận riêng, chưa được duyệt sửa.

- `check_journal_recovery.py --repo tree --out result.json`: 54/54 trên target, thư viện chuẩn, không torch/training; checkpoint marker chỉ để kiểm sao chép. Có một phép giả lập lỗi os.replace.
- `observe_csv_risks.py --repo tree --out csv.json`: dùng bước sao lưu và writer batch thật trong thư mục tạm, xác nhận ba tình huống thuộc hai CSV. Exit 0 chỉ có nghĩa script quan sát hoàn tất, không phải CSV đã sửa.
- `REPRO_REVIEW_1454b3d.py --repo tree --out old_repro.json`: cố ý bỏ qua bước chuẩn bị mới, vẫn exit 1 / bug_count 3; không dùng riêng script này để phủ nhận bản sửa.
- `tree/handoff/logs/review40/06b_repro_qua_buoc_chuan_bi.py --out corrected_path.json`: đối chứng của tác giả đã được reviewer chạy lại, bug_count 0.
- `check_package.py --zip <delta gốc> --out <thư mục mới>/evidence/package.json`: cần Git và ZIP gốc 2.177.781 byte, không nằm trong gói này. Tạo tree/base_preimages cạnh evidence; dùng thư mục mới. Chỉ kiểm phần cây thay đổi.

`evidence/` chứa kết quả reviewer. `tree/handoff/logs/review40/` là log tác giả, giữ nguyên byte. `tree/src/` có mã liên quan, không đủ để chạy toàn bộ suite training. Reviewer không chạy torch/Ultralytics, GPU, dataset đầy đủ hoặc SSH Nano.

Mỗi file trừ SHA256SUMS.txt có hash trong SHA256SUMS.txt. Không chứa checkpoint thật, ảnh, dataset, weights, NPZ hoặc gói môi trường.
