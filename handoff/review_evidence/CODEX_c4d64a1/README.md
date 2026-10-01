# Bằng chứng review c4d64a1

Đọc CODEX_REVIEW_c4d64a1.md trước. Gói không có nhãn/dataset hoặc ảnh.

- REPRO_REVIEW_c4d64a1.py: stdlib, chạy CLI thật trong thư mục tạm. Trên c4d64a1: exit 1, bug_count=5, đối chứng sạch đạt. Bốn quan sát COCO không tính vào bug_count.
- audit_real_labels.py: đối chiếu lại 1.600 JSON native và COCO val. Chạy:
  python audit_real_labels.py --nhan <box_track_20_chua_train_va_val> --coco <bdd_box_track_val_cocofmt.json> --out <thu_muc_moi>
- evidence/all_native_hashes.json: 1.600 file đọc độc lập từ HF revision ghim, toàn bộ match_manifest=true.
- evidence/real_labels_audit.json: thống kê đọc nhãn thật; class_mismatches.csv (183 hàng), ignore_exceptions.csv (50 hàng).
- evidence/coco_tracks_with_changes.json: 53 track COCO có thay đổi lớp/ignore; chưa xác định lịch sử chuyển đổi.
- evidence/package_verification.json: roundtrip 204 đường dẫn thay đổi; không tuyên bố tái tạo đủ 3.584 file không đổi.
- evidence/supplied_label_tests.log và supplied_range_tests.log: 3/3 và 7/7; chỉ sửa đường dẫn trong bản sao test khi chạy trên Linux.
- reviewed_source/: hai script đúng nội dung target, để so hash nếu cần.

Không nhập nhãn, sửa code dự án, train, gửi thư hay SSH Nano trong review.
