# Review vòng 34 — `ccd801f`

Ngày review: 2026-10-02. Kết luận: **PASS WITH LIMITATIONS** cho bước nhập nhãn train theo phương án C. **Không có finding chặn mới.**

Bằng chứng đủ để chuyển sang khóa split theo video, xuất dataset và smoke trên dữ liệu thật theo kế hoạch đã chốt. Kết luận này không xác thực snapshot nhãn là bản phát hành chính thức, và không phải phê duyệt huấn luyện dài.

## Đối tượng và phạm vi

- Base: `4e71d3217aed3b0757136b4c388abd3e3ed4da4a`.
- Target: `ccd801fb7c517edd3c61b412d015a8ae4d7faba6`.
- Gói: `review_delta_latest.zip`, **805.621 byte**.
- SHA256: `4880e54fc98c0de6749d81294c2e93c5bf716718b07cac2e3131b58e147dfd39`.
- Phạm vi: script nhập nhãn, test, manifest, log kiểm nguồn, tài liệu và bằng chứng nghiệm thu được mang sang. Không có thay đổi tại `src/` hoặc `configs/` trong delta.

Reviewer đọc và chạy kiểm chứng trong môi trường Linux riêng. Reviewer không truy cập ổ D: của máy người dùng và không có 278.079 ảnh train tại đây.

## Kiểm chứng độc lập đã chạy

| Phép kiểm | Kết quả |
|:--|:--|
| ZIP, kích thước, SHA256, CRC | Khớp; `testzip` sạch; 68 thành viên |
| Payload và phần trùng nội dung | **62/62 payload**, **15/15 alias** khớp số byte và SHA256 |
| Preimage của các file sửa | **5/5** khớp snapshot `4e71d32` đã lưu từ review trước |
| Áp patch nguyên byte | `git apply` exit 0; **77/77 đường dẫn thay đổi** khớp hash target |
| Bằng chứng review `4e71d32` được lưu vào repo | `SHA256SUMS.txt` khớp **56/56** sau khi áp patch |
| Khai báo NPZ không bàn giao | Đủ **13**, nội dung giữ nguyên từ base |
| Test nhập nhãn | Chạy nguyên `test_nhap_nhan_train.py`: **9/9 PASS**, exit 0 |
| Manifest nhập so với manifest nhãn đã nghiệm thu | **1.400/1.400** khớp tên file, số byte và SHA256 |
| Manifest nhập so với hash native do reviewer tính độc lập ở vòng trước | **1.400/1.400** khớp, 0 khác biệt |

Phép áp patch chỉ xác minh 77 đường dẫn trong delta, **không** xác nhận độc lập toàn cây 3.780 file Git mà Claude báo đã kiểm trên máy Windows.

## Nhập nhãn và provenance

Script `handoff/logs/review34/nhap_nhan_train.py` kiểm nguồn theo manifest trước khi ghi, chặn file nguồn thiếu/thừa/sai hash, chặn file đích khác hash hoặc ngoài danh sách, và giữ nguyên file đích đã trùng hash. File mới được chép qua file tạm, kiểm hash rồi đổi tên. Sau nhập, script đối chiếu đích, nguồn train và hash COCO val. Test đã chạy bao phủ nhập sạch, dry-run, chạy lại, xung đột đích, file lạ, nguồn thiếu/sai và COCO val sai hash.

Manifest nhập có đúng **1.400 tên JSON train riêng biệt**, không có native val hoặc COCO phụ. Tất cả đường đích và đường nguồn cách ly đều đúng phạm vi đã chốt. Tổng dung lượng theo manifest là **1.069.093.707 byte**.

- Manifest nguồn vòng 32: SHA256 `a2e416f26d2ba44a50c7f3ec8578deec39f364cd2381c36dc12fe45a905fcf39`.
- Manifest nhập vòng 34: SHA256 `c5a548bc3d7b3545d1d6b7a9800c926c5418d667f68a21b8da8f25fe286aa6b3`.
- Nguồn vẫn ghi đúng `jaffe03195/bdd100k_sot`, revision `694197099692fe1254e93215e73e68ece8b1f41f`.

Các log bàn giao nhất quán: đích chưa tồn tại trước nhập; ghi mới 1.400 file; sau nhập 1.400 file khớp manifest; bản cách ly 1.602 file còn khớp; COCO val giữ SHA256 `074ff79555483296cf7ccadddeeceeec7a83452c900506dc46588ed3a3e65d5d`.

Log `05_kiem_nguon_chinh.log` ghi lệnh thật trả **exit 0**, 1.400 video có nhãn và ảnh, 278.079 frame, 0 ảnh thiếu. Đây là **bằng chứng do máy Windows bàn giao**, không phải lệnh reviewer chạy lại trên dataset thật. Đối chiếu hash manifest xác nhận danh sách nhập tương ứng đúng snapshot đã kiểm trước đó; không thay thế việc băm lại thư mục đích vật lý trên Windows.

## Giới hạn giữ nguyên

1. Nhãn là snapshot bên thứ ba đã ghim revision/hash; chưa đối chiếu được archive gốc bằng MD5 chính thức. Phần train chưa có tham chiếu độc lập theo từng annotation để xác nhận chất lượng nhãn.
2. Native val và COCO val của dự án vẫn khác **183 nhãn lớp** và **50 ngoại lệ ignore**, là hai phép kiểm riêng. Nguyên nhân còn UNKNOWN. Review này không chọn bên nào đúng và không thay COCO val dùng cho kết quả lịch sử.
3. Chưa khóa split, xuất dataset, chạy smoke thật hoặc huấn luyện. Exit 0 của kiểm nguồn chỉ xác nhận đủ đầu vào theo hợp đồng kiểm nguồn.
4. Reviewer không chạy lại tracking, Nano hoặc full suite vì delta không đổi các đường mã đó. Không thao tác trên repository Windows, không commit, push, merge hay đổi base review của người dùng.

## Bước tiếp theo trong kế hoạch

Chuyển sang **khóa train-fit/dev theo video → xuất dataset với manifest v2 → smoke 1 epoch cho YOLOv8n và YOLO11n trên cùng tập con dữ liệu thật**. Giữ giới hạn smoke đã chốt, chọn tập trước khi xem AP, giữ BDD val và UA-DETRAC test ngoài train/dev, giữ kiểm `cache: false` và chặn `.npy` cạnh ảnh.

Sau smoke, báo số ảnh thực nạp, thời gian, RAM/VRAM và ước lượng ngân sách huấn luyện local. **Số epoch và ngân sách training dài vẫn chờ người dùng duyệt.** Không mở thêm model, dataset, tuning hay công việc Nano ngoài kế hoạch.

## Bằng chứng tái hiện

Gói `CODEX_REVIEW_ccd801f_EVIDENCE.zip` chứa script kiểm độc lập, log 9/9, kết quả JSON, 5 preimage cần cho patch và hướng dẫn chạy lại với delta gốc. Gói không chứa dataset, weights hoặc video. `SHA256SUMS.txt` xác định nội dung các file bằng chứng.
