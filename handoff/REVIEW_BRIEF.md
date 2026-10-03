# REVIEW_BRIEF — vòng 40: sửa R39-F1 (nhật ký resume sau dòng cuối ghi dở)

<!-- BASE_TARGET -->

Chi tiết: **`handoff/REVIEW40_CHI_TIET.md`**. Kế hoạch đăng ký trước: `handoff/logs/review40/02_ke_hoach_kiem_r39f1.md`.

**Phạm vi** (người dùng duyệt): chỉ sửa R39-F1 của review `1454b3d`, kiểm hồi quy ngắn trên fixture CPU, bàn giao, dừng.
Không train dài, không GPU, không dataset đầy đủ. **Base giữ `b83dc12`**; vòng 39 chưa được nghiệm thu.

## Review `1454b3d` — đã ghi nhận

REQUEST CHANGES, chỉ vì R39-F1. Bằng chứng khớp 97/97. Gói đã review được lưu archive, chưa nghiệm thu.

## Tái hiện trước khi sửa

- Script của Codex chạy nguyên trên `1454b3d`: **exit 1, `bug_count=3`**; ca sạch đạt.
- Đường gọi thật: preflight bỏ qua dòng cuối hỏng → bước sao lưu không đụng nhật ký → callback `bat_dau` nối vào phần ghi
  dở → dòng hỏng thành dòng giữa, lần resume sau bị chặn.

## Sửa — chỉ ở bước chuẩn bị ghi của resume thật

- `phan_tich_duoi_nhat_ky` (chỉ đọc):
  - đuôi ghi dở → cắt;
  - JSON cuối hợp lệ nhưng thiếu newline → giữ sự kiện, chỉ thêm newline;
  - dòng hỏng ở giữa → từ chối.
- `sua_duoi_nhat_ky` (trong bước sao lưu, trước lần ghi nối đầu tiên): sha256 phải trùng preflight; luôn sao lưu nguyên
  byte vào `khoi_phuc/truoc_tiep_tuc_N/`; sửa bằng file tạm + fsync + `os.replace`; đọc lại, kiểm sự kiện hợp lệ trùng
  từng sự kiện; biên nhận `xu_ly_nhat_ky.json`, cũng ghi vào sự kiện `bat_dau`.
- `--dry-run --tiep-tuc` chỉ in kế hoạch, không ghi gì. `ghi_nhat_ky` / `doc_nhat_ky` không đổi.

## Kiểm — ĐÃ CHẠY

| Phép | Kết quả |
|:--|:--|
| `test_tiep_tuc_huan_luyen`, mã mới | lần 2: **102/102**. Lần 1: 98/99, do test U8 cũ gọi bước sao lưu khi chưa có nhật ký; đã sửa phần thiết lập |
| Nhóm N trên **mã cũ** `1454b3d` | **12/30**, đúng dự đoán: A/D hỏng ở đúng 3 biến thể ("dòng 3 hỏng ở GIỮA"); ca sạch đạt |
| `test_detector_5lop` / `test_pilot_chan_doan` | 92/92 / 34/34 |
| Script Codex sau sửa | vẫn exit 1, `bug_count=3`, vì nó gọi thẳng hai hàm cấp thấp (cố ý không đổi) |
| Đối chứng: cùng 4 ca, đi qua bước chuẩn bị ghi | `bug_count=0` |

Nhóm N chạy thật qua `train_detector.py`. Ở mỗi biến thể: dry-run không đổi file; sau resume nhật ký đọc lại được, sự kiện
cũ còn đủ; bản sao trùng từng byte; checkpoint cũ được bảo toàn; preflight và resume tiếp tới hết đều chạy. Dòng hỏng ở
giữa vẫn bị từ chối, không gọi `train()`.

## Đề xuất ngoài phạm vi — chờ duyệt

- `results.csv`: đuôi byte NUL làm resume dừng bằng `ValueError` thô; đuôi số dở dang thành dòng rác.
- `theo_batch.csv`: dòng ghi dở cộng lần ghi nối sau làm dòng bị dính.
- Phòng thủ thêm: guard trong `ghi_nhat_ky`; khóa thư mục lượt.

## Trạng thái

R39-F1 đã sửa và kiểm; chờ Codex review. Training dài, số epoch, ngân sách và thời điểm **vẫn chưa duyệt**.

**Nên xem**: `logs/review40/01_*`, `03_*_lan2.log`, `04_*`, `06*`.
