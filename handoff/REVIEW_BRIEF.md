# REVIEW_BRIEF — vòng 7: chuyển cấu hình UA-DETRAC → BDD100K val

<!-- BASE_TARGET -->

Base giữ **`a8ecaf6`** (chưa có xác nhận chuyển). Phụ lục: **`handoff/BDD_TRANSFER_NOTES.md`**.
Bản khóa: `configs/phase2/LOCK_bdd_transfer.md` + `results/phase2/lock/lock_bdd_transfer.json`
— ghi lúc **03:49:13**, commit `d1f6d62`, **trước** khi chạy ma trận và **trước** mọi lần xem kết quả.

## Đã làm

1. **Nhãn**: tự tìm ở `D:\LABOTARY\01-bdd_box_track_val_cocofmt.json`, đối chiếu **đúng** 117.094.032
   byte và sha256 `074ff795…`, đặt vào `data/raw/bdd100k/labels_cocovid/`.
2. **F4 đóng trên video thật** `b23b2649-8349d2a1`: 202 ảnh, annotation dừng ở frame **94**,
   `seqinfo.ini` ghi `seqLength=**202**`. Bản cũ sẽ ghi 94 và **xoá khỏi phép đo** mọi FP ở frame 95–202.
3. **Kiểm kê toàn bộ val**: 200 video / 39.973 ảnh / 465.512 annotation; 0 ảnh thiếu, 0 sai tên file,
   0 khoảng frame bỏ qua, 0 bbox lỗi. **Không loại video nào.** Bẫy đã xử lý: chỉ có 392 `instance_id`
   cho cả val và **327** dùng chung nhiều video ⇒ ID khoá theo `(video, instance_id)`.
4. **Cấu hình riêng** ở cả bốn chỗ (`bdd_transfer_cocovid_val` / `BDDCOCOVID-val` / manifest riêng /
   cache riêng), backend `ultralytics`, `track_buffer` 1,2 s. **Một bước BDD = 0,2 s vs DETRAC = 0,04 s
   (gấp 5)** ⇒ đây là giữ nguyên **con số theo bước**, *không* phải giữ nguyên nhiễu theo đơn vị vật lý.
5. **Pilot 3 video → toàn bộ 200 video**, 12 lượt × 2 điều kiện, exit 0.

## Kết quả (toàn bộ 200 video)

| c25 | HOTA | AssA | IDF1 | IDSW | | c10 | HOTA | AssA | IDF1 | IDSW |
|:--|--:|--:|--:|--:|--|:--|--:|--:|--:|--:|
| cv | 0.3368 | 0.4754 | 0.3825 | 6.345 | | cv | 0.3410 | 0.4744 | 0.3885 | 6.262 |
| ekf | **0.3383** | 0.4792 | 0.3871 | **5.548** | | ekf | **0.3435** | 0.4806 | 0.3945 | **5.437** |
| ukf | 0.3377 | 0.4773 | 0.3863 | 5.655 | | ukf | 0.3421 | 0.4767 | 0.3927 | 5.558 |
| pf500 | 0.2966 | 0.4112 | 0.3211 | 8.040 | | pf500 | 0.3012 | 0.4116 | 0.3278 | 7.932 |

Bootstrap ghép cặp theo video, 5000 lần, seed 0:

- **EKF − CV**: HOTA **+0,0015** [0,0005; 0,0026] (c25) và **+0,0025** [0,0012; 0,0039] (c10);
  IDSW **−797** [−1.055; −558] và **−825** [−1.066; −601]. CI **không chứa 0** ở cả 4 metric, cả 2 điều kiện.
- **UKF − CV**: HOTA CI **chứa 0** ở cả hai điều kiện (IDF1 và IDSW thì không).
- **PF500** kém rõ rệt: HOTA **−0,040**, IDSW **+1.7k**. SD giữa 3 seed chỉ **0,00005–0,00095**
  ⇒ khoảng cách này **không** do seed.

**Đọc đúng mức**: lợi thế HOTA của EKF là **thật nhưng rất nhỏ** (~0,5–0,7 % tương đối). Hiệu ứng rõ
nằm ở **IDSW: −13 %**. Trên DETRAC cùng so sánh này có p = 0,22; ở đây CI hẹp lại vì mẫu là 200 video,
**không** phải vì hiệu ứng lớn hơn.

## Giới hạn (đã khoá trước, không phải biện hộ sau)

- **Không** báo chỉ số che khuất: nguồn COCO-video thiếu `occluded`/`truncated`/`timestamp`.
  `visibility = 1.0` trong `gt.txt` là **placeholder bắt buộc của định dạng**, nghĩa là "không biết".
- **Không** báo metric theo từng lớp: GT gộp 5 lớp xe thành một.
- **`motorcycle` chỉ 915/465.512 box (0,20 %)** ⇒ với đề tài giao thông hỗn hợp, **BDD val không đại diện**.
- 3 sequence đã xem ở smoke **vẫn nằm trong** tập đánh giá; không mô tả val là tập hoàn toàn chưa xem.
- Tốc độ: tracker-từ-cache (cv 0,99 → pf500 5,15 ms/frame) **tách riêng** khỏi toàn pipeline; số toàn
  pipeline là **tổng hai pha đo riêng**, detector ~22,9 ms/frame áp đảo. **Không** suy ra được cho Nano.

## Trạng thái

- **Test: 26/26 PASS, 0 FAIL** (`handoff/logs/review7/tests_20260920-0428.log`).
- **Jetson Nano: chưa kết nối** — đã kiểm: không có thư mục kết quả, không có mục `nano` trong SSH config.
  `scripts/nano/` có script nhưng **chưa chạy lượt nào**. Mọi số tốc độ trong repo là PC (GTX 1650).
