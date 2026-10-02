# REVIEW_BRIEF — vòng 36: đính chính R35-F1 và R35-F2 (chỉ diễn giải)

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW36_CHI_TIET.md`**.

**Phạm vi**: ghi nhận nghiệm thu `2351c67`, đính chính hai điểm diễn giải, rồi **dừng**.

- Không train, inference, pilot, xuất dataset hay chạy lại smoke.
- Split, dataset smoke, checkpoint, ngưỡng đánh giá và số liệu **giữ nguyên**.
- `src/` không đổi; không Nano.

## Ghi nhận nghiệm thu — ĐÃ CHẠY

- Codex `2351c67`: **PASS WITH LIMITATIONS** cho split, tập con và smoke. Bằng chứng 69/69 khớp.
- Gói 378.096 B, `76e7d2ad…` khớp, đã lưu archive. **Base → `2351c67`**; 13 khai báo NPZ giữ nguyên.
- Reviewer không lặp lại training GPU và không xác minh toàn cây Git.

## R35-F1 — quan sát và nguyên nhân — ĐÃ SỬA

Đối chiếu lại log và JSON có sẵn (`logs/review36/04_*`, không nạp model):

- `13_*` so **COCO 80 lớp** với **checkpoint smoke 5 lớp**. Model 5 lớp sau khởi tạo, trước bước học, **chưa được đo**.
- Warmup thực chạy không có log; nhận định "không warmup khi 1 epoch" chỉ dựa trên đọc mã nguồn.

**Câu đính chính**: checkpoint smoke có confidence thấp hơn checkpoint COCO 80 lớp; **nguyên nhân chưa xác định**. Warmup và
trạng thái khởi tạo model 5 lớp là các yếu tố cần phân biệt.

- Conf 1e-6 cho car AP50 ≈ 0,233: chỉ cho thấy có dự đoán **car** khớp GT ở điểm thấp trên 84 ảnh dev smoke. Đã rút câu
  "không phải lỗi đường ống".
- Đánh giá chính ở conf 0,001 giữ nguyên. Dev smoke không có motorcycle hay bicycle.

## R35-F2 — dự toán là kịch bản — ĐÃ SỬA

- Sửa **tại nguồn** `15_du_toan_training.py`, chỉ chuỗi mô tả. Sinh lại vào `logs/review36/02_*`.
- **162 giá trị số và cấu trúc giống hệt** JSON gốc và JSON reviewer (`03_*`; bộ đối chiếu tự kiểm 7 ca).
- Tài liệu: "cao", "cận trên", `≤` → **kịch bản dự toán theo giả định**; thực tế có thể vượt.

| Bước / epoch | Kịch bản | Kịch bản chậm hơn |
|:--|--:|--:|
| 5 / 15 | 56,64 h | 73,95 h |
| 10 / 15 | 29,10 h | 37,99 h |

## Kiểm không đổi — ĐẠT

`logs/review36/05b_*`:

- split `b8a2848e…`, dataset smoke, kết quả runner và 4 checkpoint khớp ghi nhận;
- log và số đo gốc vòng 35 không đổi.

`05a_*` báo sai vì phép kiểm bắt cả file riêng của người dùng; giữ lại, có ghi chú. Full suite NOT_RUN vì `src/` không đổi.

## Chờ người dùng duyệt

Pilot · xuất dataset đầy đủ · số epoch · bước frame · ngân sách training local.

**Nên xem**: `REVIEW35_CHI_TIET.md` mục 7, 9, 10 (đã đính chính); `logs/review36/03_*`, `04_*`, `05b_*`.
