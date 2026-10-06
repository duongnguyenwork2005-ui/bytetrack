# REVIEW_BRIEF — vòng 42: training dài YOLOv8n → YOLO11n (15 epoch mỗi model, dataset đầy đủ)

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW42_CHI_TIET.md`**. Kế hoạch đăng ký trước: `handoff/logs/review42/02_ke_hoach_luot_dai.md`.

**Phạm vi** (người dùng duyệt): training dài trên PC, tuần tự YOLOv8n rồi YOLO11n; tổng 15 epoch mỗi model kể cả khi
resume; dataset đầy đủ bước 5, split đã khóa; giữ nguyên tham số đã duyệt. Chỉ training + validation dev; **dừng trước
đánh giá test**.

## Review `6c8e37a` — đã ghi nhận

PASS WITH LIMITATIONS, đóng R40-N1 / R40-N2. Bằng chứng khớp 43/43. Base → **`6c8e37a`**, giữ 13 khai báo.
**R41-N1** chỉ sửa câu chữ: docstring `sua_duoi_csv` (AST sau khi bỏ docstring trùng bản cũ), mục 3 phụ lục vòng 41,
đính chính ở phụ lục kế hoạch.

## Chuẩn bị

- Dry-run cả hai model đạt: hợp đồng dataset 0 lỗi; đã băm 56.046 ảnh và nhãn; định danh trùng bản xuất vòng 38;
  checkpoint khởi tạo đúng hash.
- `train_bdd5.yaml` giữ nguyên byte, vì cấu hình pilot ghim sha256 của nó.
- Khởi chạy **độc lập với phiên Claude** qua WMI `Win32_Process.Create` → `chay_giam_sat.py` → runner. Tiến trình cha là
  `WmiPrvSE.exe`, không có Job Object, môi trường người dùng chuẩn (đã thăm dò trước).

## Kết quả — ĐÃ CHẠY

| | YOLOv8n | YOLO11n |
|:--|:--|:--|
| Epoch | **15/15**, 3 lần chạy (resume 2 lần) | **15/15**, 1 lần |
| Điều kiện thành công | 12/12 | 12/12 |
| Exit lần cuối | 0, `[DAT]` | 0, `[DAT]` |
| `best.pt` | epoch 15, `5f66776d…` | epoch 13, `d7d1c7e8…` |
| Dev mAP50-95 / mAP50 (`final_eval`) | 0,280 / 0,435 | 0,272 / 0,416 |
| Thời gian runner | 28,8 h | 30,2 h |
| VRAM tối đa | 2.933 MiB | 3.123 MiB |

**Sự cố YOLOv8n** (có log):
- Người dùng yêu cầu tạm dừng. Dừng bằng Ctrl+C, exit 130; epoch 8 đang dở bị bỏ.
- Rút sạc làm máy ngủ 2 lần. Lượt không bị ngắt.
- Loss **NaN** giữa epoch 12. Người dùng chọn dừng; checkpoint epoch 11 đã được kiểm là sạch.

Mỗi lần resume: dry-run `--tiep-tuc` đạt, kiểm khôi phục 13/13, đang cắm sạc. Lần 2 không còn NaN. **Nguyên nhân NaN
chưa xác định.**

## Giới hạn

- Chỉ **dev nội bộ**; `best.pt` chọn theo chính dev này nên điểm thiên lạc quan.
- Mỗi model một lượt. Chênh lệch 0,280 / 0,272 **chưa đủ để kết luận**.
- YOLOv8n qua 2 lần resume, không trùng từng bit với lượt liên tục.
- Không đóng gói checkpoint và log lớn; có bảng sha256 (`ket_qua/tep_lon_khong_ban_giao.csv`).

## Trạng thái

Training dài đã xong cả hai model. **Chưa nghiệm thu.** Đánh giá test (BDD val / UA-DETRAC), so sánh bộ lọc, tracking
và tối ưu Nano **chờ duyệt**.

**Nên xem**: `logs/review42/02_*`, `15_*`, `19_*`, `11_*`, `ket_qua/`.
