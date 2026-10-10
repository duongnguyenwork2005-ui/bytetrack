# Mốc B (R47B) — adapter đánh giá detector trên UA-DETRAC test

**Người làm**: Claude, ngày 2026-10-10 (khoảng 03:00–03:30). Phiên Claude điều phối R47 đã dừng để chờ Codex (hết hạn mức tới 05:50). Người dùng yêu cầu Claude làm tiếp, nên Claude làm mốc này trực tiếp, không qua vòng review của orchestrator.
**Codex review**: chạy sau, song song với suy luận (xem `03_ke_hoach_khoa_danh_gia_detrac_test.md`, mục 6).

## File

| File | sha256 |
|:--|:--|
| `src/danh_gia_detector_detrac_test.py` (mới) | `6540832e8cae42d1b129f3c4aaed65422196ffdb502f0db19d183b640709100b` |
| `src/test_danh_gia_detector_detrac_test.py` (mới) | `a9e8fc0e48c30c8876249a215b3fd7c9fe3931d1775cb705d7ebd7a52b8113c8` |

Evaluator `src/eval_detector_5lop.py` (`30a1fd5a…`) và adapter BDD `src/danh_gia_detector_bdd_val.py` không đổi; adapter mới chỉ import từ hai file này.

## Ánh xạ quyết định → mã → test

| Quy tắc | Mã (`danh_gia_detector_detrac_test.py`) | Test |
|:--|:--|:--|
| car, bus là GT | `LOAI_GT`, `kiem_va_doc_gt` | `ca_d1_d5_d3_d4_qua_entry_point` (n_gt car 3, bus 1) |
| D1: van là vùng bỏ qua cho mọi lớp | `LOAI_BO_QUA`, nhánh `elif b is None / else: ign.append` | car trùng box van không thành FP; van không vào n_gt car |
| D2: others là vùng bỏ qua | như D1 | truck trùng others không thành FP; truck không có GT |
| `ignored_region` cấp sequence | `ign_seq` áp cho mọi ảnh của video | GT car trong vùng vẫn TP; car chưa ghép trong vùng bị bỏ qua |
| D3: chỉ car, bus; `mAP (car, bus)`; N/A | `chuan_hoa_d3`, `ghi_ket_qua` | JSON và cả 3 CSV; `ca_d3_lop_ngoai_tap_co_gt_la_loi` |
| D4: loại ảnh không có `<frame>`, lưu danh sách | `d4_moi` so với file D4 | ảnh `img00003` không được đưa đi dự đoán và được liệt kê |
| D5: quy tắc bỏ qua của evaluator | dùng nguyên `danh_gia` / `ghep_anh_lop` | như dòng `ignored_region` |
| Box: cắt khung, bỏ < 1 px | `_hop_le_box` của exporter | đếm `box_gt_nho_hon_1px_sau_cat` |
| Kiểm trước dự đoán | `kiem_va_doc_gt`, kiểm sha256 checkpoint trong `main` | sai sha256, ảnh đổi nội dung, XML đổi, sai kích thước: exit 2, không dự đoán, không tạo thư mục |
| Danh sách ảnh có thứ tự | `danh_sach_anh.json` + `--cham-lai` | chấm lại cho cùng số liệu; danh sách bị đảo thì exit 2 |

## Lệnh và kết quả

- `python src/test_danh_gia_detector_detrac_test.py` → **10/10 PASS**, exit 0. Fixture tự tạo gồm 2 video, 5 ảnh 960×540; fingerprint của fixture dựng bằng chính script mốc A; dự đoán giả lập, không nạp model.
- `python -u src/danh_gia_detector_detrac_test.py --chi-kiem` trên dữ liệu thật → exit 0, 64 s:
  - 56.167 ảnh đánh giá, 173 ảnh loại theo D4, 40 video;
  - GT: car 548.555, bus 71.785;
  - vùng bỏ qua: `ignored_region` 220.290 lượt theo ảnh, van 38.519, others 16.915.

## Giới hạn

- Codex review sau, song song với suy luận, không làm trước khi chạy.
- Mutation test riêng cho adapter chưa làm.
- Không phải mọi ca test đều đi qua entry point thật (đính chính sau review Codex lần 2, xem mục dưới):
  - đi qua `main(argv)`: các ca D1–D5, D3, D4, biên cắt khung, D5 biên IoA, `--cham-lai`, kiểm trước dự đoán;
  - đọc lại cả JSON lẫn CSV: chỉ ca D1–D5/D3/D4;
  - gọi hàm trực tiếp: ca `vehicle_type` lạ (`doc_gt_xml`) và ca lớp ngoài tập có GT (`chuan_hoa_d3`).

## Sửa sau khi khóa kế hoạch (ghi thêm 2026-10-10)

Ba lượt đánh giá chạy bằng bản `6540832e…` ở trên. Các bản sửa dưới đây chỉ đụng đường `--cham-lai` và đều được đưa vào repo **sau** khi lượt cuối kết thúc (07:27:41). Kết quả và thời điểm: `04_ket_qua_danh_gia_detrac_test.md`, phụ lục của `03_ke_hoach_khoa_danh_gia_detrac_test.md`.

| Bản | Adapter sha256 | Test sha256 | Test | Nội dung |
|:--|:--|:--|:--|:--|
| v1 (đã khóa, dùng cho 3 lượt) | `6540832e…` | `a9e8fc0e…` | 10/10 | như trên |
| v2 (Codex review lần 2 approve) | `971f50a9a22f912e4741f48b203867f831b8f30ec18d6807331673bcb0da2a1d` | `a1632335…` → `2945cd709a05c45a22a660a462cff5a09d8f1b2528e71a8d0c5eab17a55370d8` | 14/14 | sửa 2 issue P2 của Codex review lần 1 (xem bên dưới) |
| v3 | `bb4b497b4f89a4464777144a35d66cbd109d03d3c485c4176cde0b67e358ccb4` | `2945cd70…`, không đổi | 14/14 | `doc_du_doan` đọc mỗi mảng NPZ một lần |

**v2** sửa hai issue P2 về `--cham-lai`:
- Ghi `nguon_du_doan_<tên>.json` ngay sau NPZ, **trước** khi chấm. File gồm sha256 NPZ, danh tính dữ liệu và quy tắc dự đoán.
- `--cham-lai` từ chối (exit 2) khi dữ liệu, cấu hình dự đoán hoặc NPZ khác lúc dự đoán.
- Bước chấm lỗi vẫn chấm lại được mà không cần dự đoán lại.
- Lượt chạy bằng bản cũ (chỉ có `ket_qua.json`) vẫn chấm lại được.

Test thêm cho v2:
- biên cắt khung (box một phần ngoài khung, box < 1 px);
- D5 với IoA 0,50 bị bỏ qua và 0,49 là FP;
- phục hồi khi bước chấm lỗi;
- dữ liệu đổi;
- NPZ đổi;
- lượt chạy bằng bản cũ.

Sau khi Codex approve, Claude chỉ sửa docstring của test (`a1632335…` → `2945cd70…`) để thu hẹp mô tả coverage.

Codex ghi các giới hạn sau:
- test cắt khung chưa assert tọa độ sau cắt;
- test D5 chỉ kiểm tổng TP/FP ở IoU 0,5;
- nhánh lượt chạy bằng bản cũ chỉ kiểm đường thành công;
- chưa có test từ chối cấu hình dự đoán khác.

**v3** sửa lỗi hiệu năng phát hiện khi chấm lại trên dữ liệu thật:
- `doc_du_doan` gọi `z["xyxy"]` (và `conf`, `cls`) bên trong vòng lặp theo ảnh.
- Mỗi lần gọi, `NpzFile` giải nén lại toàn bộ mảng: 1,23 s cho 13,5 triệu box. Trên 56.167 ảnh, việc này mất khoảng 24 giờ.
- Lỗi đã có từ v1, nhưng không ảnh hưởng lượt đánh giá thường (chấm ngay trong bộ nhớ). Fixture 5 ảnh không lộ ra lỗi này.
- Cách sửa: đọc mỗi mảng một lần, trong `with np.load(...)`, rồi cắt theo ảnh. Kết quả không đổi.
- Chưa có test hiệu năng. Bằng chứng là lượt chấm lại 3 model trên dữ liệu thật, ghi ở báo cáo 04.
