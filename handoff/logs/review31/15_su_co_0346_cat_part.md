# Sự cố 2026-10-01 03:46 — `tai_mot20_train.py` cắt mất dữ liệu tốt của phần 2 khi `curl` bị giết

## Diễn biến

- **00:22**: tạm dừng theo yêu cầu người dùng. Log `10_tai/tam_dung_20261001-002220.json`: `.part` phần 2 = 2.744.422.360 B.
- **03:16**: chạy lại 4 tiến trình dưới dạng tác vụ nền của công cụ. Mỗi tiến trình mở một lượt `curl` dài.
  - 03:17 đo được phần 2 = 2,84 GB và đang tăng 0,68 MB/s.
- **03:46:06**: công cụ dừng cả 4 tác vụ nền vì chạm giới hạn thời gian (khoảng 30 phút).
- Kích thước sau đó, đo bằng `os.stat` (mtime của cả 4 file = 03:46:06):

  | Phần | Sau 03:46 | So với lúc tạm dừng |
  |:--|--:|--:|
  | 1 | 4.070.802.904 B | +1.259.315.200 B |
  | 2 | **2.744.422.360 B** | **+0 B**, bị cắt về đúng kích thước đầu lượt |
  | 3 | 4.428.378.276 B | +1.299.804.160 B |
  | 4 | 4.657.056.676 B | +1.252.864.000 B |

  Ước mất khoảng 1,2 GB dữ liệu đã tải của phần 2. Theo hai phần kế bên, phần 2 hẳn cũng đã tăng cỡ 1,2–1,3 GB.
- Không có sự kiện `luot_tai` nào được ghi cho lượt này: Python bị giết ngay sau khi cắt, trước khi kịp ghi log.

## Nguyên nhân

Lỗi của bộ bảo vệ “không để trang HTML lẫn vào `.part`”, trong commit `32c6e49`:

1. `curl` bị giết giữa chừng thì không kịp in `-w "%{http_code}|%{content_type}|…"`, nên `content_type` là chuỗi rỗng.
2. Điều kiện cũ là `sau > co and ctype != "application/zip"`. Chuỗi rỗng bị coi là “không phải zip”, nên script cắt
   `.part` về `co`, tức kích thước lúc bắt đầu lượt.
3. Phần 1, 3, 4 thoát được chỉ vì tiến trình Python bị giết trước khi chạy tới dòng cắt (tranh chấp thời gian).

Dữ liệu còn lại vẫn nhất quán, vì `.part` là tiền tố đúng thứ tự của file trên máy chủ. md5 cuối cùng vẫn kiểm toàn bộ.

## Sửa

Commit sau `32c6e49`, file `tai_mot20_train.py`:

- `curl` ghi header phản hồi ra `<ten>.part.hdr` bằng `-D`. Content-Type lấy theo thứ tự: `-w`, rồi header của phản hồi HTTP
  cuối cùng.
- Chỉ cắt khi **biết chắc** Content-Type khác `application/zip`. Không xác định được thì giữ byte, ghi cờ
  `content_type_khong_ro` vào log; md5, magic ZIP và testzip ở bước xác minh vẫn chặn file hỏng.
- Xóa `.hdr` khi phần tải xong.

## Kiểm thử hồi quy

Bộ test `09_thu_cuc_bo_script_tai/thu_cuc_bo.py` có thêm hai ca:

- **Ca 12**: máy chủ giả lập gửi 1 MB rồi treo 25 s. Test giết đúng tiến trình `curl` con, Python vẫn sống, rồi kiểm:
  - `.part` không bị cắt;
  - Content-Type lấy được từ file header;
  - lượt sau tải tiếp bằng 206, kết thúc DAT với sha256 đúng;
  - file `.hdr` đã được dọn.
- **Ca 13**: phản hồi `text/html` thật vẫn bị cắt về 0 và báo LOI.

| Mã | Kết quả | Log |
|:--|:--|:--|
| Bản cũ `32c6e49` | **22/25**: FAIL 3 ý của ca 12, đúng hiện tượng “Content-Type '' … da cat .part ve 0” | `ket_qua_ban_cu_32c6e49_22_25.txt` |
| Bản sửa | **25/25 PASS** | `ket_qua_ban_sua_25_25.txt` |

## Giới hạn thời gian của tác vụ nền

Công cụ dừng tác vụ nền sau khoảng 30 phút nếu không đặt `timeout`. Các lần chạy lại sau sự cố dùng `timeout` dài hơn;
cách chạy và thời hạn thực tế được ghi trong log `10_tai/`.
