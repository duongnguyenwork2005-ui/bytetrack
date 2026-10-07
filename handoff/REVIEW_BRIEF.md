# REVIEW_BRIEF — vòng 46: xử lý review `d4715aa` (R45-F1…F4); UA-DETRAC chưa triển khai

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW46_CHI_TIET.md`**. Base **giữ `3e94b21`**. D1–D5 **chưa duyệt**. Không GPU, không khóa
giao thức.

## Tái hiện trước sửa

Chạy nguyên byte `audit_review45.py` trên gói `d4715aa`. Lần 1 FAIL do môi trường (cp1252 trên Windows). Lần 2 với
`-X utf8`: exit 0, tái hiện **4/4**, đầu ra trùng kết quả reviewer.

## Sửa bằng mã (công cụ audit; bản vòng 45 giữ nguyên làm mã cũ)

- **R45-F4**: `doc_xml()` dùng `node is not None`; thiếu thật thì ghi rõ, không để trống. Trên XML thật cả 40/40 video có
  đủ camera / thời tiết (ô trống ở vòng 45 là do parser).
- **R45-F3**: cộng `ignored_region` cấp sequence cho 173 ảnh khi giữ: **+265 lượt vùng áp dụng trên ảnh** (không phải
  vùng hình học). Bốn phương án giữ thành 275.989 / 259.074 / 237.470 / 220.555.
- **Test** `test_cong_cu_detrac.py`: công cụ cũ **2/8** (FAIL đúng lỗi), công cụ sửa **8/8**.

## Sửa diễn giải

- **R45-F1**: scorer **vẫn** đếm TP / FP thô và precision cho lớp không có GT; chỉ AP / recall / F1 = null. Đã sửa
  REVIEW45, dự thảo, REVIEW43, và thêm vào phụ lục mục 10 của kế hoạch vòng 43. Hiển thị N/A cần bước chuẩn hoá báo cáo,
  **phải triển khai sau khi D3 được duyệt**.
- **R45-F2**: rút kết luận "recall / AP tăng". Phản ví dụ AP50 0,505 → 0; chiều trên dữ liệu thật chưa biết. Toolkit
  chỉ theo phần mã đã đọc (bản trích + sha256 trong gói); `AP_DET_EVAL.exe` chưa xác minh. 40.835 chỉ là GT XML đạt
  ngưỡng trong thống kê này.
- **Làm rõ 1–3**: fingerprint → adapter → khóa; adapter lưu danh sách `video/frame` có thứ tự; kiểm w/h mọi ảnh.

## Còn phụ thuộc người dùng

Chọn D1–D5. Chuẩn hoá D3, adapter, bảng hash ảnh làm sau khi duyệt.

**Nên xem**: `REVIEW46_CHI_TIET.md` mục 3–4, `logs/review46/01`–`03`, `05`, `test_cong_cu_detrac.py`, diff của
`logs/review45/06_du_thao_giao_thuc_detrac.md`.
