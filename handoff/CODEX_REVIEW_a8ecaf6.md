# Review độc lập `9a1c404` → `a8ecaf6`

**Kết luận: PASS WITH LIMITATIONS.** Không tìm thấy lỗi chặn hoặc lỗi chức năng còn lại trong phạm vi sửa. Không cần sửa code hay chạy lại benchmark/A/B/C.

## Đã tự kiểm chứng

- HEAD/manifest cùng trỏ tới `a8ecaf6`; base là `9a1c404`. ZIP hợp lệ; cả 15 payload khớp SHA256 manifest và blob target.
- Luồng CLI YOLO nạp model, resolve `model.ckpt_path`, gọi `fill_env()` để hash chính checkpoint rồi mới gọi `write_meta()`. Hash thiếu/sai, fingerprint cũ không tương thích đều bị chặn trước khi đổi cache; cache legacy vẫn đọc được và không được backfill; cùng hash ở đường dẫn khác hợp lệ; `gt-synthetic` không cần checkpoint. `test_cache_provenance` có trong registry 22 suite và bao phủ đủ các hành vi này.
- Chạy độc lập từ snapshot target trong thư mục tạm: `test_cache_provenance` **12/12 PASS** và `test_review_delta` **13/13 PASS** (exit 0; Python 3.14.5). Repo tạm bổ sung cho file riêng ở trạng thái thêm/sửa/xóa/đổi tên/`git rm --cached` không để lọt marker/đường dẫn; file `rm --cached` còn trên đĩa.
- Áp patch thật lên snapshot base: `git apply --check` và `git apply` đều exit 0. So toàn bộ 150 đường dẫn `src/`/`configs/` không thuộc chính sách file riêng: không thừa, thiếu hoặc khác nội dung so với target. Việc đổi tên và hai file code trùng nội dung cũng được test bằng so cây, không chỉ exit code.
- Khôi phục file riêng: `67445dc^ == 23f63ed`; sáu file vắng tại `23f63ed`, có tại `67445dc`, rồi vắng tại `534146c`/target. `rerun_with_state_log.py` ở target có đúng blob `23f63ed`; diff được yêu cầu rỗng. Hash index và tám file working-tree liên quan không đổi trong review này.

## Chỉ đối chiếu từ log

- Smoke YOLO thật: MVI_20052, 30 frame, 1.648 box, cache tạm, strict read `ok`, **5/5 PASS**, exit 0. Hash checkpoint local trong log `f59b3d83…` khớp hash tôi tính lại cho `models/yolov8n.pt`; tôi không chạy lại inference GPU.
- Registry, lệnh và exit code trong log full suite xác nhận **22 PASS, 0 FAIL/SKIPPED/NOT_RUN**. Log ở `534146c`; tới target không đổi cache, packager, test hay registry.

## Giới hạn

- Alias/tự tải checkpoint YOLO chưa được kiểm bằng model thật; bằng chứng thật chỉ dùng checkpoint local. Đây không chặn đường sử dụng hiện tại.
- 53/56 sequence cache và 113 run cũ vẫn là legacy; không có fingerprint hồi tố. BDD100K và Jetson Nano chưa có môi trường kiểm chứng.
- `verify_delta_vong3.log` ghi target trung gian `5380221`/16 đường dẫn vì chính log được commit sau đó; manifest cuối là `a8ecaf6`/17. Kiểm tra độc lập trên gói cuối đã bù khoảng tự tham chiếu này.
- Khẳng định “0/7 hash trước/sau” của thao tác khôi phục chỉ còn trong brief/commit message, không có artifact riêng chứa đủ hai bảng hash. Lịch sử Git xác nhận kết quả index/commit cần đạt, nhưng không tự chứng minh toàn bộ trạng thái working tree tại đúng thời điểm thao tác.

Không cần chạy lại full suite hay smoke GPU: hai test trọng tâm đã PASS độc lập và gói cuối tái tạo đúng cây thuộc phạm vi review.
