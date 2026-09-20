# Lượt suite 2026-09-20 19:39 — 29 PASS / 1 FAIL (`test_run_experiment_guard`)

**Có thật**: `tests_20260920-1939_stdout_runner.log` là stdout gốc của `run_all_tests.py` cho lượt này
(bảng tổng kết + dòng FAIL). **Đã mất**: file log chi tiết theo test `tests_20260920-1939.log` — tôi
đã xóa nó trước khi chạy lại lượt sạch, và không khôi phục được. Không tái tạo.

**Nguyên nhân — mức bằng chứng**:

- *Đã xác minh*: chạy riêng `test_run_experiment_guard` sau đó → 9/9; lượt suite sạch 19:48 (không
  commit song song) → 30/30. Phép kiểm FAIL là "chạy lại không đổi gì → TÁI SỬ DỤNG", với bước
  occlusion in `CHAY LAI, ly do: code`.
- *Giả thuyết được chứng cứ gián tiếp ủng hộ, chưa chứng minh trực tiếp từ log đã mất*: trong cửa sổ
  chạy 19:39–19:44 có commit `5e28f37` và `ef3316c` (xem mốc thời gian git ở trên); `code`
  fingerprint chứa `git rev-parse HEAD`, nên HEAD đổi giữa hai lần tính trong cùng một test sẽ cho
  đúng thông báo `ly_do: code`. Không có cơ chế nào khác thay đổi module trong lúc đó.

Bài học đã áp dụng: không commit/sửa source trong lúc suite chạy; giữ log thất bại ở thư mục này.
