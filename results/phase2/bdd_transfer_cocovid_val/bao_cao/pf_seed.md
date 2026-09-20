# PF — biến thiên giữa ba seed và cách tổng hợp

Cách tổng hợp **đã khóa trước** (`results/phase2/lock/lock_bdd_transfer.json`):

- **Bảng chính**: trung bình **metric** theo seed.
- **Bootstrap**: trung bình **thống kê đủ** theo seed rồi tái tổ hợp.
- Hai đại lượng này **khác nhau** với metric phi tuyến (xem `PHASE2_REPORT.md` mục 15.2).

> SD dưới đây là **theo từng metric** — một con số SD duy nhất **không** mô tả được cả ba. Khi trích dẫn phải nói rõ SD của metric nào.

## `c25`

| tracker | n_seeds | HOTA | SD HOTA | SD IDF1 | SD AssA |
|:--|--:|--:|--:|--:|--:|
| pf500 | 3 | 0.2966 | 0.00005 | 0.00110 | 0.00075 |

## `c10`

| tracker | n_seeds | HOTA | SD HOTA | SD IDF1 | SD AssA |
|:--|--:|--:|--:|--:|--:|
| pf500 | 3 | 0.3012 | 0.00095 | 0.00095 | 0.00295 |

## Kết luận đúng phạm vi

**Trong ba seed đã thử**, chênh lệch HOTA giữa `pf500` và `cv` lớn hơn nhiều lần biến thiên giữa các seed. Phát biểu này giới hạn ở **implementation PF này**, **N = 500**, **ba seed 0/1/2** và **cấu hình đã khóa**; nó **không** loại trừ hoàn toàn ảnh hưởng của seed, và **không** khái quát cho mọi bộ lọc hạt.

