<!-- sinh tu dong boi tong_hop_danh_gia.py tu ket_qua.json; khong sua tay -->

**AP theo lớp** — BDD100K MOT val (COCO-video), 39.973 ảnh, 200 video. AP nội suy 101 điểm; AP50–95 = trung bình 10 ngưỡng IoU 0,50:0,05:0,95. Chênh lệch Δ = model − baseline `yolov8n_coco`, tính bằng **điểm AP** (×100).

| Lớp | Số GT | Số ảnh có lớp | Model | AP50 | AP50–95 | ΔAP50 | ΔAP50–95 |
|:--|--:|--:|:--|--:|--:|--:|--:|
| car | 341.436 | 38.964 | `yolov8n_coco` | 0,558 | 0,337 | — | — |
| car | 341.436 | 38.964 | `yolov8n_ft` | 0,779 | 0,518 | +22,1 | +18,1 |
| car | 341.436 | 38.964 | `yolo11n_ft` | 0,776 | 0,514 | +21,8 | +17,8 |
| truck | 27.343 | 15.862 | `yolov8n_coco` | 0,183 | 0,135 | — | — |
| truck | 27.343 | 15.862 | `yolov8n_ft` | 0,377 | 0,268 | +19,4 | +13,3 |
| truck | 27.343 | 15.862 | `yolo11n_ft` | 0,369 | 0,263 | +18,6 | +12,9 |
| bus | 9.087 | 7.018 | `yolov8n_coco` | 0,249 | 0,201 | — | — |
| bus | 9.087 | 7.018 | `yolov8n_ft` | 0,395 | 0,298 | +14,7 | +9,7 |
| bus | 9.087 | 7.018 | `yolo11n_ft` | 0,366 | 0,280 | +11,7 | +7,9 |
| motorcycle | 902 | 774 | `yolov8n_coco` | 0,117 | 0,058 | — | — |
| motorcycle | 902 | 774 | `yolov8n_ft` | 0,178 | 0,080 | +6,1 | +2,2 |
| motorcycle | 902 | 774 | `yolo11n_ft` | 0,173 | 0,080 | +5,5 | +2,2 |
| bicycle | 4.135 | 2.738 | `yolov8n_coco` | 0,113 | 0,055 | — | — |
| bicycle | 4.135 | 2.738 | `yolov8n_ft` | 0,188 | 0,090 | +7,5 | +3,5 |
| bicycle | 4.135 | 2.738 | `yolo11n_ft` | 0,196 | 0,096 | +8,3 | +4,1 |

**mAP (macro, 5 lớp: car, truck, bus, motorcycle, bicycle — mọi lớp đều có GT)**

| Model | mAP50 | mAP50–95 | ΔmAP50 | ΔmAP50–95 | P macro @0,25 | R macro @0,25 |
|:--|--:|--:|--:|--:|--:|--:|
| `yolov8n_coco` | 0,244 | 0,157 | — | — | 0,572 | 0,219 |
| `yolov8n_ft` | 0,384 | 0,251 | +14,0 | +9,4 | 0,613 | 0,347 |
| `yolo11n_ft` | 0,376 | 0,247 | +13,2 | +9,0 | 0,613 | 0,338 |

**Điểm vận hành** conf ≥ 0,25, IoU 0,5 (ngưỡng tracker c25). Detection không ghép mà IoA với vùng bỏ qua ≥ ngưỡng thì không tính FP.

| Lớp | Model | TP | FP | FN | Precision | Recall | F1 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| car | `yolov8n_coco` | 150.469 | 41.977 | 190.967 | 0,782 | 0,441 | 0,564 |
| car | `yolov8n_ft` | 239.996 | 51.967 | 101.440 | 0,822 | 0,703 | 0,758 |
| car | `yolo11n_ft` | 239.438 | 56.797 | 101.998 | 0,808 | 0,701 | 0,751 |
| truck | `yolov8n_coco` | 5.450 | 8.444 | 21.893 | 0,392 | 0,199 | 0,264 |
| truck | `yolov8n_ft` | 8.724 | 3.369 | 18.619 | 0,721 | 0,319 | 0,442 |
| truck | `yolo11n_ft` | 8.271 | 3.247 | 19.072 | 0,718 | 0,302 | 0,426 |
| bus | `yolov8n_coco` | 2.434 | 2.473 | 6.653 | 0,496 | 0,268 | 0,348 |
| bus | `yolov8n_ft` | 3.297 | 1.995 | 5.790 | 0,623 | 0,363 | 0,459 |
| bus | `yolo11n_ft` | 3.149 | 2.311 | 5.938 | 0,577 | 0,347 | 0,433 |
| motorcycle | `yolov8n_coco` | 114 | 176 | 788 | 0,393 | 0,126 | 0,191 |
| motorcycle | `yolov8n_ft` | 189 | 294 | 713 | 0,391 | 0,210 | 0,273 |
| motorcycle | `yolo11n_ft` | 167 | 224 | 735 | 0,427 | 0,185 | 0,258 |
| bicycle | `yolov8n_coco` | 252 | 65 | 3.883 | 0,795 | 0,061 | 0,113 |
| bicycle | `yolov8n_ft` | 573 | 555 | 3.562 | 0,508 | 0,139 | 0,218 |
| bicycle | `yolo11n_ft` | 629 | 547 | 3.506 | 0,535 | 0,152 | 0,237 |

**Chẩn đoán mô tả** (không đổi metric)

| Model | Ảnh đã dự đoán | Ảnh không có GT 5 lớp | Detection conf ≥ 0,25 trên các ảnh đó |
|:--|--:|--:|--:|
| `yolov8n_coco` | 39.973/39.973 | 778 | 118 |
| `yolov8n_ft` | 39.973/39.973 | 778 | 121 |
| `yolo11n_ft` | 39.973/39.973 | 778 | 99 |

**Chẩn đoán bổ sung** (đếm từ `du_doan_<ten>.npz` lúc tái tính; KHÔNG có trong kế hoạch khóa, chỉ mô tả, không đổi metric): số detection sau NMS mỗi ảnh (conf ≥ 0,001, trần `max_det` 300).

| Model | Tổng detection | Trung bình / ảnh | Tối đa / ảnh | Số ảnh chạm trần 300 |
|:--|--:|--:|--:|--:|
| `yolov8n_coco` | 4.920.095 | 123,09 | 300 | 330 |
| `yolov8n_ft` | 4.624.928 | 115,7 | 300 | 155 |
| `yolo11n_ft` | 4.661.420 | 116,61 | 300 | 163 |

**Lượt chạy** (bộ giám sát `chay_giam_sat.py`; RSS = working set của tiến trình đánh giá; RAM máy trống = `psutil.virtual_memory().available` theo bộ giám sát, mẫu 1 s)

| Model | Bắt đầu | Kết thúc | Thời gian (s) | Exit | RSS tối đa (MiB) | RAM máy trống thấp nhất (MiB) |
|:--|:--|:--|--:|--:|--:|--:|
| `yolov8n_coco` | 2026-10-06 20:27:33 | 2026-10-06 20:46:36 | 1143,1 | 0 | 941,5 | 50,0 |
| `yolov8n_ft` | 2026-10-06 20:47:16 | 2026-10-06 21:07:22 | 1206,4 | 0 | 971,0 | 130,0 |
| `yolo11n_ft` | 2026-10-06 21:08:08 | 2026-10-06 21:27:49 | 1181,4 | 0 | 1126,1 | 122,0 |
