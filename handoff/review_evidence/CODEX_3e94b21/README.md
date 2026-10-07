# Bằng chứng review `3e94b21` — PASS WITH LIMITATIONS

Đọc `CODEX_REVIEW_3e94b21.md`. Bộ bằng chứng kiểm mã/fixture và tính toàn vẹn gói; không chạy inference dữ liệu thật hoặc training.

## Nội dung

- `package_check.json`: ZIP 600666 byte, hash khóa, 116 đường dẫn patch + 17 support.
- `base_comparison.json`: 6 preimage trùng snapshot base reviewer đã có.
- `round43_preservation.json`: so 61 file vòng 43; 57 không đổi.
- `eol_checks.json`: tự chuyển LF sang CRLF để khớp hai hash của mã đã chạy.
- `bdd_current.log`: 22/22; `bdd_old_entry.log`: 20/22, FAIL B5b/B5c đúng dự kiến.
- `evaluator_groups_E.log` / `scorer_tests.json`: 13/13 + SKIP E6 thiếu weights; pycocotools thật.
- `repro_current.json`: script metadata của reviewer trước, bug_count=0.
- `module_origins.json`: đường nạp các mô-đun được theo dõi.
- `input/`: 17 support + hai file mới từ gói, nguyên byte. Đây là cây cho test fixture, không phải toàn repo.
- `old/src/danh_gia_detector_bdd_val.py`: entry point `a03f6e1`, nguyên byte để làm đối chứng.
- `REPRO_REVIEW_a03f6e1.py`: script cũ nguyên byte; phạm vi doubles được ghi trong docstring.

`changed_from_round43.json` lúc đầu đã đếm cả pycache do reviewer tạo. Không dùng bản đó để kết luận; kết luận sử dụng `round43_preservation.json`, chỉ xét đường dẫn do MANIFEST của gói vòng 43 khai báo.

## Chạy lại

Cần Git, Python 3.12, numpy, pandas, Pillow, OpenCV và pycocotools. Phiên bản reviewer trong `environment.json`. Không cần torch, Ultralytics, checkpoint hay dataset thật cho phạm vi harness dưới đây.

Ví dụ tạo thư mục dependency riêng chứa các gói cần thêm:

```bash
python -m pip install --target deps --no-deps opencv-python-headless==4.12.0.88 pycocotools==2.0.11
python run_review_tests.py --tree input --deps deps --old-entry old/src/danh_gia_detector_bdd_val.py --repro REPRO_REVIEW_a03f6e1.py --out recheck
```

Nếu môi trường chưa có numpy/pandas/Pillow/PyYAML thì cài các dependency đó trước. Harness đặt PYTHONPATH chỉ vào thư mục dependency được chỉ định; test tự đưa source của cây fixture vào đường import. Nó không cài hoặc tải model, không chạy nhóm training.

Kiểm ZIP người dùng ban đầu (không nhúng ZIP đó vào bằng chứng):

```bash
python check_package.py --zip /path/review_delta_latest.zip --out work/evidence/package_recheck.json
```

Lệnh cuối kiểm phạm vi file thay đổi và support, không xác nhận toàn bộ Git target độc lập. Cây tạm đối chứng cũ được xóa sau test. Đường dẫn tuyệt đối trong log chỉ là nơi reviewer đã chạy; không cần tái tạo chúng.

`SHA256SUMS.txt` băm tất cả file trong bằng chứng trừ chính nó.
