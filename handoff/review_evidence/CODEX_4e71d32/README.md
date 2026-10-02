# Review 4e71d32 — bằng chứng độc lập

Kết luận và giới hạn: CODEX_REVIEW_4e71d32.md.

- evidence/test_runs.json: script Codex exit 0; regression target 30/30, old 3/30.
- evidence/repro/summary.json: bug_count=0, đối chứng sạch đạt; bốn ca COCO quan sát có exit 2/3.
- evidence/package_verification.json: 301 payload +27 alias, patch 328 paths; không tự xác nhận 3.708 file toàn cây.
- evidence/real_target.json và native_hashes.json: 1.600 native JSON tải từ revision ghim, hash khớp; parser schema không lỗi; so COCO thật có 183 lệch lớp, 50 ngoại lệ ignore. Không kiểm ảnh thực trên PC.
- Hai CSV real_target_* trùng byte với review trước.

Chạy lại fixture (không cần dataset), từ gốc gói:

    python tree/handoff/review_evidence/CODEX_c4d64a1/REPRO_REVIEW_c4d64a1.py --repo tree --out my_repro
    python tree/handoff/logs/review33/test_kiem_nhan_ung_vien.py --script tree/handoff/logs/review32/kiem_nhan_ung_vien.py

Harness đọc nhãn thật (sẽ đọc HTTP Range khoảng 151 MB, không tải ảnh):

    python check_real_target.py --snapshot tree --coco <bdd_box_track_val_cocofmt.json> --out my_real_check

Cần Python có pip (dùng requests bundled trong pip). Giữ đúng COCO sha256 074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d. Harness không ghi nhãn vào repository, chỉ dùng JSON tạm từng file. Entry point của harness được thêm đối số đường dẫn sau lần chạy để tiện tái hiện; thuật toán/parser target giữ nguyên.

Trong gói không có nhãn, ảnh, video, weights hoặc archive nguồn.
