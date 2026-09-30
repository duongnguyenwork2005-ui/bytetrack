# REVIEW_BRIEF — vòng 29: sửa R28-F1 (cache ảnh `.npy`, review `07615ba`)

<!-- BASE_TARGET -->

Review `handoff/CODEX_REVIEW_07615ba.md`: REQUEST CHANGES, một finding; R27-F1/F2/F3 đã hết lỗi. **Base giữ `239b9c6`**.
**Trạng thái: đã sửa, chờ Codex nghiệm thu.** Chi tiết: **`handoff/REVIEW29_CHI_TIET.md`**. Không training dài, không thêm
dataset hay model, không đụng Nano.

## File review

- Chỉ có `CODEX_REVIEW_07615ba.md` trên máy.
- **Không có** `CODEX_REVIEW_07615ba_EVIDENCE.zip` / `REPRO_REVIEW_07615ba.py`, nên **script Codex = NOT_RUN**. Lệnh chạy
  khi có file nằm trong phụ lục.
- Thay thế bằng script tái dựng `logs/review29/tai_hien_r28_f1.py`, dùng exporter, chốt runner, callback và
  `YOLODataset` thật.
- Ba file review vòng 28 (chép từ đính kèm) nay so với bản gốc: **trùng từng byte**.

## Nguyên nhân

Ultralytics 8.4.141 `BaseDataset.load_image()` nạp `<ảnh>.npy` cùng tên **thay** JPEG nếu file tồn tại, **kể cả khi
`cache=False`**. Hợp đồng vòng 28 chỉ liệt kê ảnh theo `IMG_FORMATS`; callback chỉ so đường dẫn, lớp và bbox. Đây là cache
**ảnh**, khác cache **nhãn** `.cache` đã sửa ở vòng 28.

## Sửa

**Chính sách**: `cache: false`, không dùng cache ảnh; pixel nạp phải là JPEG đã khóa sha256.

- **`kiem_hop_dong_dataset()`** (dùng chung cho exporter, runner và evaluator):
  - `.npy` cạnh chính ảnh train/dev mà trainer sẽ nạp là lỗi, thông báo nêu split, đường dẫn, lý do;
  - không cấm `.npy` nơi khác, không tự xóa;
  - danh tính dataset ghi `cache_anh`.
- **Runner**:
  - từ chối `cache` khác `false` (exit 2);
  - callback trước vòng train kiểm dataset **thật**: `ds.cache` phải là `None`, không có `ds.npy_files`. Phép này chặn
    `.npy` xuất hiện sau hợp đồng.
- **Test**: thiếu weights thì A7/E6/F được ghi **SKIP**, không còn đếm PASS.

Không đổi cấu hình đã khóa, công thức AP, ngưỡng, split hay chốt vòng 28.

## Kiểm chứng (tự chạy)

- **Script tái dựng, trên `07615ba`**: exit 1, `bug_count=2`.
  - JPEG TB 90,40 / 89,69 (trùng số Codex) nhưng pixel nạp = 0; hash JPEG đúng;
  - chốt runner và callback đều cho qua.
- **Script tái dựng, tại HEAD `83482c2`**: **exit 0, `bug_count=0`**; đối chứng sạch đạt.
- **`test_detector_5lop` tại HEAD: 92/92, 0 SKIP.** Nhóm J mới:
  - J0: dataset sạch, `YOLODataset` **thật** đọc pixel trùng JPEG ở 125/125 ảnh;
  - J1/J2: `.npy` cạnh ảnh train/dev — ultralytics thật nạp ảnh đen, còn hợp đồng và chốt runner từ chối;
  - J3/J4: entry point runner với **spy** tại ranh giới model — exit 2, không nạp model, không train, không output;
  - J5a/J5b: callback với `YOLODataset` thật chặn `.npy` xuất hiện sau hợp đồng;
  - J6: `cache: disk` bị từ chối.

  Regression cũ (YAML train→dev, nhãn thiếu/rỗng, hash, cache nhãn, COCO) vẫn đạt.
- **Cùng test trên mã chưa sửa**: 84/92. 8 FAIL: J1–J6 và J5a/J5b là lỗi thật; J0b là khóa mới. Đối chứng J0 và nhóm A–I
  đều PASS.
- **`run_all_tests --only test_detector_5lop`**: PASS; các suite khác NOT_RUN vì không bị ảnh hưởng.
- **Runner thật, không spy, fixture GPU**:
  - `.npy` cạnh ảnh train/dev → exit 2, không tạo thư mục;
  - dataset sạch → smoke YOLOv8n 1 epoch **ĐẠT**; callback ghi 719/90 khớp, `cache_anh None`, `npy 0`.

## Giới hạn

- Script Codex chưa chạy.
- Không băm lại pixel mỗi lần nạp: bảo đảm dựa trên hash JPEG + không `.npy` + `cache=None`.
- Evaluator cũng bị chặn khi có `.npy` (chính sách thống nhất).
- Chưa có BDD MOT train; epoch và ngân sách chưa duyệt.
