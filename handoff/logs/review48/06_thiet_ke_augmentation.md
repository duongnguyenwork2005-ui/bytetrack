# R48D / M5 — Thiết kế kiểm chứng dữ liệu và augmentation (chỉ thiết kế, KHÔNG huấn luyện)

Viết ngày 2026-10-11, HEAD `eaf2e87` (cộng thay đổi chưa commit của R48C/R48D). Trong lượt này **không** huấn luyện,
không suy luận detector, không ghi gì vào `data/`. Mọi số dưới đây lấy từ audit R48B (`03_audit_du_lieu.md`) và R47
(`handoff/logs/review47/04_ket_qua_danh_gia_detrac_test.md`). Phần thực thi **bị chặn**: cần GPU nhiều giờ, vượt giới hạn
khoảng 10 phút mỗi lệnh foreground, và cần người dùng duyệt trước (mục 8).

Nhãn: [QS] quan sát đã có (kèm nguồn), [GT] giả thuyết chưa kiểm, [TK] quyết định thiết kế đề xuất.

## 1. Augmentation đang bật (theo R48B, `03` mục 3.2; `results/phase2/kc48_audit/augmentation_hieu_luc.csv`)

| Biến đổi | Giá trị hiệu lực | Chạy khi train detect? |
|:--|:--|:--|
| Mosaic | 1,0; `close_mosaic` 2 | có (tắt 2 epoch cuối) |
| RandomPerspective | `translate` 0,1; `scale` 0,5; `degrees` = `shear` = `perspective` = 0 | có (dịch, co giãn) |
| RandomHSV | 0,015 / 0,7 / 0,4 | có |
| `fliplr` | 0,5 | có |
| CopyPaste / MixUp / CutMix | 0 | không |
| Albumentations | mặc định | **không** (gói không cài trong env) |
| `auto_augment`, `erasing` | có trong args | **không** (chỉ dùng cho classify) |

Cấu hình: `configs/phase2/detector/train_bdd5.yaml`. Hai checkpoint `yolov8n_ft`, `yolo11n_ft`: `imgsz` 640, `batch` 8,
`nbs` 64 (accumulate 8), 6.311 batch / epoch, 15 epoch, SGD `lr0` 0,01, `seed` 0, `deterministic`; `ema_updates` cuối là
12.988 / 12.987 (`03` mục 3.1).

## 2. Điểm yếu thực tế làm căn cứ chọn thay đổi

- [QS] Trên UA-DETRAC test (R47, D1–D5), mAP50 (car, bus): baseline COCO `yolov8n_coco` 0,870 > `yolov8n_ft` 0,586 /
  `yolo11n_ft` 0,564. Recall bus của hai checkpoint fine-tune khoảng 0,34. Nguyên nhân **chưa xác định**: mức đóng góp
  của nhầm lớp bus → truck / car chưa đo (`PLAN_DA_CHOT.md` mục 0, "Điểm chưa rõ").
- [QS] Lệch cỡ và cắt khung giữa hai miền (`03` mục 5.1, 5.2): kt_rel p50 của bus là 0,05–0,07 ở BDD, 0,18–0,20 ở DETRAC.
  Bus DETRAC bị cắt khung (`trunc` > 0) 25,9–48,6 %, BDD 12,3–16,2 %. Car DETRAC cũng lớn hơn (kt_rel p50 0,080–0,100 so
  với 0,048–0,053).
- [QS] Lớp hiếm ở BDD: motorcycle có mặt ở 230 video train-fit / **24** video dev / 29 video val; bicycle ở 397 / 44 / 63.
- [QS] Ranh giới car / truck / van mơ hồ trên mẫu trực quan, chưa đo tác động (`03` mục 10 ý 8).
- [QS] Dev BDD không độc lập theo lần ghi hình với train-fit (70/139 video dev chung tiền tố tên, `03` mục 10 ý 1).

## 3. Một thay đổi đề xuất [TK]

**Đổi `scale` 0,5 → 0,9** (giữ mọi tham số khác). Trong RandomPerspective của ultralytics, hệ số co giãn rút đều trong
[1 − scale, 1 + scale]. Đổi như vậy nới khoảng phóng to từ ×1,5 lên ×1,9, nên ảnh huấn luyện có thêm bus / car **lớn** và
**bị cắt khung**, gần phân bố DETRAC hơn. Đây là quan sát cỡ ở mục 2. [GT] Thay đổi này có thể cải thiện recall bus khi
chuyển miền. Chưa kiểm: có thể hiệu ứng bằng 0 nếu recall bus thấp thực ra do nhầm lớp hoặc do quy tắc đánh giá.

Không chọn:
- MixUp / CutMix: không nhắm điểm yếu đã quan sát.
- CopyPaste: cần mask phân đoạn, nhãn BDD MOT chỉ có box.
- Albumentations: phải cài gói, việc này bị cấm trong lượt.
- `degrees` / `perspective`: góc nhìn camera giám sát có thể là một yếu tố, nhưng chưa có số đo góc nhìn nên chưa có căn
  cứ.

Ràng buộc với thay đổi:
- Chỉ áp cho ảnh **train**.
- Không nhân bản ảnh rồi gọi là tăng số cảnh độc lập.
- Bbox phải được biến đổi cùng ảnh. Ultralytics làm việc này trong `RandomPerspective`; R48D không kiểm lại bằng ảnh.
- Kiểm trực quan: 2 lô `train_batch*.jpg` cho mỗi cấu hình, xem trước khi train hết epoch. Ảnh lưu ở
  `results/phase2/kc48_audit/overlay/`, vì repo public.
- Không thử trên chuỗi tracking; nếu sau này thử thì phải dùng cùng biến đổi cho mọi frame của một chuỗi.

## 4. Cổng loại lỗi đánh giá TRƯỚC khi diễn giải là domain shift / thiếu dữ liệu [TK]

M5 yêu cầu không diễn giải domain shift là thiếu dữ liệu trước khi loại lỗi đánh giá. Các bước trước khi huấn luyện:
1. Ma trận nhầm lớp của 3 checkpoint hiện có trên **UA-DETRAC dev12**: phần Train, không phải tập test. Mục đích: đo tỉ lệ
   bus GT được dự đoán là truck / car. Theo D1–D5, van / others là vùng bỏ qua; DETRAC không có truck GT.
2. Đếm bus GT ở dev12 nằm ≥ 50 % trong `ignored_region`. Theo D5 các box này vẫn được chấm (`03` mục 10 ý 4).
3. Nếu phần lớn bus bị bỏ sót là do nhầm sang truck, thay đổi augmentation ở mục 3 không nhắm đúng nguyên nhân. Khi đó
   dừng, báo người dùng và không huấn luyện.

Hai bước 1–2 cần suy luận detector, mà lượt này cấm suy luận detector mới ⇒ **chưa chạy**.

## 5. Tập con lồng nhau theo video (đường cong lượng dữ liệu) [TK]

- Đơn vị: video của train-fit `bdd5_train_v1` (1.261 video, 50.487 ảnh `buoc_frame` 5). Dev 139 video giữ cố định.
- Phân tầng theo cặp cờ (có motorcycle, có bicycle) của từng video, lấy từ nhãn gốc, 4 tầng. Trong mỗi tầng: hoán vị bằng
  `numpy.random.default_rng(48)` trên danh sách tên video đã sắp xếp. Tập p % = **tiền tố** ⌈p · n_tầng⌉ của từng tầng,
  với p ∈ {25, 50, 100}. Vì là tiền tố nên 25 % ⊂ 50 % ⊂ 100 % (lồng nhau), và giữ tỉ lệ video có lớp hiếm. Kỳ vọng ở
  25 %: khoảng 58 video có motorcycle, 99 video có bicycle.
- Ghi manifest: danh sách video và ảnh, sha256 từng manifest, số box từng lớp, số video có từng lớp. Kiểm: tập p % chứa
  trọn tập nhỏ hơn; mọi video thuộc train-fit; không video nào thuộc dev hay val.
- Hạn chế: video chung tiền tố tên có thể rơi vào các tập con khác nhau. Điều này không ảnh hưởng tính lồng nhau, nhưng
  dev vẫn không độc lập (mục 2).

## 6. Optimizer steps, epoch, compute và giao thức so sánh [TK]

- Hiện trạng: 50.487 ảnh / batch 8 ⇒ 6.311 batch / epoch. Với accumulate 8, sau warmup có khoảng 6.311 / 8 ≈ 789 bước
  optimizer / epoch; 15 epoch ≈ 11.8 nghìn bước. Log ghi `ema_updates` 12.988. [GT] Phần chênh có thể do warmup:
  ultralytics tăng dần accumulate từ 1 lên 8 trong các iteration warmup. Chưa kiểm trong mã; phải kiểm trước khi dùng con
  số này.
- Hai chế độ khi so lượng dữ liệu:
  - (A) **Cùng số bước optimizer**: tập 25 % chạy nhiều epoch hơn, compute bằng nhau. Đây là so sánh chính, vì khi đó khác
    biệt không bị lẫn với ngân sách tối ưu.
  - (B) Cùng 15 epoch: compute tỉ lệ với số ảnh. Chỉ dùng tham khảo; không quy khác biệt cho lượng dữ liệu.
- Augmentation (`scale` 0,5 so với 0,9) so trên 100 % ở chế độ (A). Cùng checkpoint khởi tạo (`yolov8n.pt` COCO), cùng
  seed, chỉ đổi một yếu tố. Một detector trước (`yolov8n`), theo M5.
- Ghi cho mỗi run: số ảnh, số video, epoch, số bước optimizer (đếm từ log), giờ GPU, sha256 trọng số `best.pt`, tiêu
  chí chọn checkpoint (fitness trên dev, như hiện tại).
- Đánh giá: AP50 / AP50-95 từng lớp trên dev BDD và UA-DETRAC dev12 để chọn. Tập đánh giá cuối (BDD val, UA-DETRAC test)
  chỉ chạy **một lần** sau khi chốt, theo protocol hiện hành: giữ D1–D5, quy tắc ignore / mapping
  (`src/eval_detector_5lop.py`, chỉ dùng, không sửa). Báo AP từng lớp cho motorcycle / bicycle kèm số video có lớp; CI
  bootstrap theo video.

## 7. Giới hạn diễn giải

- Dev BDD không độc lập với train-fit, nên chỉ số dev có thể lạc quan.
- UA-DETRAC chỉ có car / bus trong 5 lớp, nên so sánh miền chỉ làm được ở hai lớp này.
- Cải thiện trên UA-DETRAC sau khi đổi `scale` không chứng minh cơ chế "cỡ lớn / cắt khung" nếu chưa phân tầng recall
  theo cỡ bus và theo cờ cắt khung.
- Nếu đường 25 / 50 / 100 % còn dốc, chỉ nói "chưa bão hòa trong miền BDD". Không suy ra rằng thêm dữ liệu BDD sẽ sửa được
  khoảng cách sang UA-DETRAC.
- Nhãn BDD train chưa xác thực với archive gốc (`03` mục 4).

## 8. Phần bị chặn / chưa chạy

| Việc | Lý do | Cần gì |
|:--|:--|:--|
| Ma trận nhầm lớp và kiểm ignore trên dev12 (mục 4) | lượt này cấm suy luận detector | người dùng duyệt suy luận trên dev12 |
| Sinh manifest tập con 25 / 50 / 100 % (mục 5) | ghi vào `data/` nằm ngoài `write_paths`; thiết kế chưa duyệt | duyệt nơi ghi (đề xuất `results/phase2/kc48_m5/`) |
| Huấn luyện (mục 6) | GPU GTX 1650: 15 epoch trước đây phải tiếp tục 2 lần (`03` mục 3.1); vượt giới hạn 10 phút / lệnh và giới hạn tác vụ nền | người dùng duyệt ngân sách GPU và cách chạy dài |
| Kiểm trực quan ảnh augment | phụ thuộc lần huấn luyện | như trên |

Không có số nào trong file này là kết quả của M5. M5 hiện chỉ dừng ở thiết kế.
