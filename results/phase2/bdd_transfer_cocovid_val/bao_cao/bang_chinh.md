# Bảng chính — chuyển cấu hình UA-DETRAC → BDD100K val

Thí nghiệm `bdd_transfer_cocovid_val`, split `BDDCOCOVID-val`, nguồn nhãn **COCO-video chuyển đổi** (không phải `box_track_20` native).

> Đây là **kết quả thực nghiệm** trên toàn bộ val. Phân tầng che khuất **không** được báo cáo: nguồn nhãn thiếu `occluded`.

## Điều kiện `c25` (conf > 0.25)

| tracker | HOTA | DetA | AssA | IDF1 | FP | FN | IDSW |
|:--|--:|--:|--:|--:|--:|--:|--:|
| cv | 0.3368 | 0.2474 | 0.4754 | 0.3825 | 17,892 | 260,696 | 6,345 |
| ekf | 0.3383 | 0.2473 | 0.4792 | 0.3871 | 17,682 | 260,982 | 5,548 |
| ukf | 0.3377 | 0.2475 | 0.4773 | 0.3863 | 17,649 | 260,895 | 5,655 |
| pf500 | 0.2966 | 0.2266 | 0.4112 | 0.3211 | 24,448 | 271,451 | 8,040 |

## Điều kiện `c10` (conf > 0.1)

| tracker | HOTA | DetA | AssA | IDF1 | FP | FN | IDSW |
|:--|--:|--:|--:|--:|--:|--:|--:|
| cv | 0.3410 | 0.2547 | 0.4744 | 0.3885 | 24,361 | 255,224 | 6,262 |
| ekf | 0.3435 | 0.2547 | 0.4806 | 0.3945 | 24,039 | 255,459 | 5,437 |
| ukf | 0.3421 | 0.2548 | 0.4767 | 0.3927 | 24,125 | 255,336 | 5,558 |
| pf500 | 0.3012 | 0.2337 | 0.4116 | 0.3278 | 30,523 | 266,536 | 7,932 |

