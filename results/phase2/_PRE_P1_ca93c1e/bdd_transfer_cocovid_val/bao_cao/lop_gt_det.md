# Lớp được đánh giá, ánh xạ COCO–BDD, số lượng GT/detection

## Lớp được đánh giá

Mục tiêu đánh giá là **`vehicle`** = bicycle, bus, car, motorcycle, truck → gộp thành **một lớp** (class 1) trong GT MOTChallenge, liên tục với UA-DETRAC.

- `train` (tàu hỏa) → **vùng bỏ qua**.
- `pedestrian`, `rider` → **không** đánh giá (detector COCO không có lớp `rider`); chỉ thống kê.

> Vì GT gộp 5 lớp xe thành một lớp, **metric riêng từng lớp chưa được báo cáo**: đường tách lớp (`split_tracker_output_by_class`) và chính sách ignore theo lớp **chưa được kiểm chứng** trên nguồn này.

## Ánh xạ COCO → BDD (detector yolov8n pretrained COCO)

| COCO id | lớp BDD | thuộc mục tiêu `vehicle` |
|--:|:--|:--|
| 1 | bicycle | **có** |
| 2 | car | **có** |
| 3 | motorcycle | **có** |
| 5 | bus | **có** |
| 7 | truck | **có** |

`person` (COCO id 0) **không** được ánh xạ: COCO gộp `pedestrian` và `rider`.

## Số lượng GT theo lớp (toàn bộ val)

| lớp | số box GT | tỉ lệ | vai trò |
|:--|--:|--:|:--|
| car | 361,515 | 77.66 % | mục tiêu |
| pedestrian | 58,907 | 12.65 % | chỉ thống kê |
| truck | 27,824 | 5.98 % | mục tiêu |
| bus | 9,091 | 1.95 % | mục tiêu |
| bicycle | 4,406 | 0.95 % | mục tiêu |
| rider | 2,546 | 0.55 % | chỉ thống kê |
| motorcycle | 915 | 0.20 % | mục tiêu |
| train | 308 | 0.07 % | vùng bỏ qua |

> **`motorcycle`: 915 box (0.20 %)** — rất hiếm trong BDD val. Với đề tài giao thông hỗn hợp (xe máy chiếm đa số), **BDD val không phải là tập đại diện**; mọi kết luận về xe máy từ nguồn này đều dựa trên mẫu rất nhỏ.

## Số detection theo lớp (từ detection cache, sau khi lọc)

Số detection đưa vào tracker, đếm từ **chính detection cache** sau khi lọc theo `classes` và ngưỡng confidence của từng điều kiện (trước association):

| lớp | COCO id | det `c25` | det `c10` | GT | det/GT `c25` |
|:--|--:|--:|--:|--:|--:|
| bicycle | 1 | 318 | 898 | 4,406 | 0.07 |
| car | 2 | 193,945 | 385,617 | 361,515 | 0.54 |
| motorcycle | 3 | 351 | 1,083 | 915 | 0.38 |
| bus | 5 | 4,963 | 8,226 | 9,091 | 0.55 |
| truck | 7 | 14,186 | 22,645 | 27,824 | 0.51 |

> `det/GT` **không** là độ chính xác: detection chưa được ghép với GT và hai bên dùng **định nghĩa lớp khác nhau** (COCO vs BDD). Nó chỉ cho thấy detector **đề xuất nhiều hay ít hơn** GT ở mỗi lớp.

> **Xe máy** — số đề xuất so với GT đổi chiều theo ngưỡng: `c25` 351/915 = 0.38; `c10` 1,083/915 = 1.18. Vì GT gộp 5 lớp xe thành một, **nhầm lớp giữa các lớp xe không bị phạt**, nên những con số này **không** nói được detector phát hiện xe máy tốt hay kém; muốn biết phải ghép theo lớp, mà đường đó chưa được kiểm chứng trên nguồn này.

