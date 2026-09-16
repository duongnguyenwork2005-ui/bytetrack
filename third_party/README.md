# third_party

- `sort_upstream/` — bản **đã ghim** của `sort.py` (abewley/sort, commit `bce9f0d`) + `LICENSE` (GPL-3.0),
  chỉ dùng làm **tham chiếu kiểm thử** trong `src/test_sort_upstream_parity.py`. Không import vào pipeline.
- `_testdeps/` — **không commit** (xem `.gitignore` của thư mục này): nơi cài riêng `filterpy==1.4.5` để chạy
  mã upstream mà không đụng vào môi trường `khoaluan-mot`:

      python -m pip install --no-deps --target third_party/_testdeps filterpy==1.4.5

  Thiếu `filterpy` thì `test_sort_upstream_parity.py` báo **SKIP** kèm hướng dẫn, không FAIL.
