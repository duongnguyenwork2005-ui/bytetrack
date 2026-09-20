# PF — biến thiên giữa ba seed và cách tổng hợp

Cách tổng hợp **đã khóa trước** (`results/phase2/lock/lock_bdd_transfer.json`):

- **Bảng chính**: trung bình **metric** theo seed.
- **Bootstrap**: trung bình **thống kê đủ** theo seed rồi tái tổ hợp.
- Hai đại lượng này **khác nhau** với metric phi tuyến (xem `PHASE2_REPORT.md` mục 15.2).

## `c25`

| tracker | n_seeds | HOTA | SD HOTA | SD IDF1 | SD AssA |
|:--|--:|--:|--:|--:|--:|
| pf500 | 3 | 0.2966 | 0.00005 | 0.00110 | 0.00075 |

## `c10`

| tracker | n_seeds | HOTA | SD HOTA | SD IDF1 | SD AssA |
|:--|--:|--:|--:|--:|--:|
| pf500 | 3 | 0.3012 | 0.00095 | 0.00095 | 0.00295 |

