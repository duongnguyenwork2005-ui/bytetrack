# Pilot UTUAV B/C cho ByteTrack KF/EKF/UKF/PF — 01/11/2026

## Kết luận

Đã hoàn tất pilot thăm dò trên hai đoạn UTUAV validation. **Không có bằng chứng EKF, UKF hoặc PF tốt hơn KF/CV** trong các điều kiện đã thử. Đây không phải benchmark cuối của luận văn: detector YOLO11n mixed5 hiện tại không nhận được xe UAV đủ tốt để đánh giá tracking thực tế, còn phần so bộ lọc bên dưới dùng detection **tổng hợp từ GT** và FPS 30 giả định.

## Dữ liệu và phạm vi

- Hai archive người dùng tải về tại `D:/LABOTARY/`: `UTUAV_B_C_3and6class.tar.gz` (18.383.495.213 B) và `UTUAV_MOT_VOC_3_6classes.tar.gz` (147.813.351 B). Cả hai đã đọc hết luồng và qua CRC gzip (`utuav_full_check.log`). SVMOT chưa có tệp hoàn chỉnh tại thời điểm pilot.
- Gói ảnh có 11.898 JPG. Validation B có 738 frame liên tiếp `26341–27078`, validation C có 751 frame liên tiếp `20271–21021` (`images_inventory.json`). Gói MOT/XML có ID trong toàn bộ 26.200 XML, parse 0 lỗi; trường `occluded` của 4.604.082 object đều `0` (`labels_inventory.json`). Con số object này gồm cả các bản nhãn 3 lớp và 6 lớp, không phải số vật thể độc lập.
- Chọn **trước khi chạy tracker**: `B_straight_26401_26520` làm đối chứng đi thẳng (0 track rẽ theo quy tắc 10 frame/30°); `C_turn_20901_21020` là đoạn rẽ (31 track rẽ, gồm 12 xe máy). Mỗi đoạn 120 frame liên tiếp (`window_candidates.json`, `pilot_selection.json`). Chỉ giải nén 240 ảnh, tổng 358.940.942 B; ảnh và XML/GT ở `data/extracted/utuav_pilot/` và `data/processed/UTUAV-pilot/`.
- Dùng nhãn **6 lớp**: `motorbike`, `car`, `bus` là đối tượng động cần chấm; ba lớp `_stat` thành MOT distractor class 8 để TrackEval bỏ qua dự đoán khớp xe đứng yên. TrackEval class 1 là nhóm phương tiện động theo quy ước dự án. B có 4.794 GT động, gồm 1.013 xe máy; C có 22.043 GT động, gồm 5.720 xe máy (`extract_log.json`). Chấm thêm riêng xe máy bằng cùng kết quả tracker (`UTUAV-pilot-moto`).
- Nguồn UTUAV mô tả nhãn 3/6 lớp, `track_id`, `occlusion` placeholder và split SequentialGap300 tại [trang chính thức](https://videodatasets.org/UTUAV/). Trang đó không nêu FPS; pilot dùng **30 FPS giả định**, nên tham số thời gian và kết quả chưa khóa.

## Kiểm detector đã chọn: YOLO11n mixed5

Checkpoint: `models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt`. Ngưỡng sinh detection `0,05`, ngưỡng vào tracker `>0,25`, class model `0 car / 1 truck / 2 bus / 3 motorcycle`, ảnh đầy đủ 3840×2160, `imgsz=640` theo cấu hình dự án. Recall dưới đây là **cận trên không xét class**: ghép box detection với mọi GT động ở IoU ≥0,5, nên nếu cả cận trên bằng 0 thì class-aware recall cũng bằng 0.

- C, 10 frame, `imgsz=640`: 1.795 GT động; 130 detection thô, 14 qua ngưỡng tracker; **0 GT khớp** ở cả hai ngưỡng (`smoke_c10.log`, cache `utuav_pilot_smoke_c10`). Tất cả 130 detection thô được model gọi là car, không có motorcycle.
- C, 3 frame, `imgsz=1280`: 538 GT động; 23 detection thô, 1 qua ngưỡng; **0 GT khớp** (`smoke_1280_c3.log`). Không có motorcycle.
- B, 3 frame, `imgsz=640`: 123 GT động; 27 detection thô, 3 qua ngưỡng; **0 GT khớp** (`smoke_b3.log`).
- Thử chia một ảnh C thành 16 ô cố định 960×540, YOLO cùng checkpoint: 70 detection thô, 8 qua ngưỡng, không có motorcycle (`tile_sanity_c1.log`). Đây chỉ là kiểm tra nguyên nhân, không tạo cache/tracking tiled.

Do detector tạo nhiều box quá lớn so với GT xe nhỏ và không tìm thấy xe máy trên các frame thử, **đã dừng benchmark YOLO dài**. Không suy từ smoke này rằng YOLO11n mixed5 luôn kém trên mọi frame UTUAV; nó đủ để cho thấy pilot hiện tại không phải phép đo bộ lọc có ý nghĩa.

## Chẩn đoán riêng: detection tổng hợp từ GT

Hai kịch bản dùng chính bộ sinh `gt-synthetic` đã có trong repo, chung cache cho mọi bộ lọc; không phải kết quả YOLO. Mỗi bbox có nhiễu Gaussian 3% theo kích thước, bỏ sót độc lập 15%, seed 0, confidence 0,3–0,95. Cấu hình ByteTrack KF/CV, EKF/CTRV, UKF/CTRV, PF/CTRV 100 hạt giữ các ngưỡng ghép cặp mặc định như YAML hiện có; PF chạy seed 0, 1, 2. TrackEval chấm HOTA, IDF1, IDSW. Không tune theo hai đoạn.

**Kịch bản A — bỏ sót độc lập:** cache `utuav_pilot_gt_j03_d15_s0` (`gt_cache_log.json`). Tổng hợp B+C:

- KF/CV: HOTA **77,151%**, IDF1 **90,624%**, 20 IDSW.
- EKF: HOTA 77,051%, IDF1 90,427%, 24 IDSW.
- UKF: HOTA 77,155%, IDF1 90,534%, 22 IDSW. HOTA hơn KF chỉ **0,004 điểm phần trăm**, trong khi IDF1 và IDSW kém; không coi là lợi thế có ý nghĩa.
- PF100 ba seed: HOTA 69,838–70,518%, IDF1 81,230–82,433%, 194–199 IDSW.
- Riêng xe máy: KF HOTA 76,942% / IDF1 91,125% / 8 IDSW; EKF 76,643% / 90,697% / 9; UKF 76,692% / 90,746% / 8. PF thấp hơn rõ (`results/trackeval/UTUAV-pilot-moto/`).

**Kịch bản B — thêm mất detection liên tiếp:** chính sách ghi **trước lượt chạy** trong `pilot_gap_policy.json`. Chọn một nửa các track động dài ≥20 frame theo hash `sequence:track_id`, bỏ 8 vị trí frame liên tiếp quanh frame giữa track, GT vẫn giữ nguyên. Cache `utuav_pilot_gt_j03_d15_gap8_v2_s0` lấy **đúng cache A rồi chỉ bỏ box**: 164 detection ở B, 688 ở C (tương ứng 183/807 vị trí GT trước bỏ sót ngẫu nhiên). Mọi box còn lại là tập con trùng giá trị với cache A; đã kiểm bằng `replay_source_ids` và `np.array_equal` (`gt_gap_cache_v2_log.json`). Tổng hợp B+C:

- KF/CV: HOTA **74,358%**, IDF1 **88,676%**, 21 IDSW.
- EKF: HOTA 74,216%, IDF1 88,412%, 26 IDSW.
- UKF: HOTA 74,243%, IDF1 88,335%, 26 IDSW.
- PF100 ba seed: HOTA 67,021–67,604%, IDF1 78,527–79,828%, 204–216 IDSW.
- Riêng xe máy: KF HOTA **74,517%** / IDF1 **89,304%** / 9 IDSW; EKF 74,073% / 88,619% / 11; UKF 73,966% / 88,316% / 11. PF HOTA 55,863–59,214% (`results/trackeval/UTUAV-pilot-gapv2-moto/`).

Kịch bản B là **che khuất giả lập qua detection**, không phải phép phân tầng che khuất thật từ UTUAV. Kết quả không chứng minh KF luôn tốt hơn CTRV; nó cho thấy các đoạn và cấu hình pilot này chưa làm nổi bật lợi thế bộ lọc khác KF.

**Bản gap v1 không dùng để kết luận:** cache `utuav_pilot_gt_j03_d15_gap8_s0` được tạo bằng cách bỏ GT *trước* bộ sinh ngẫu nhiên, làm dịch chuỗi nhiễu ở các box còn lại. Các output `utuav_gap_*`, `UTUAV-pilot-gap-moto` chỉ giữ làm dấu vết; mọi số kịch bản B ở trên đều từ cache `gap8_v2` và output `utuav_gapv2_*`.

## Tái lập và việc tiếp theo

- Script kiểm archive: `src/external_mot_preflight.py`; inventory: `inventory_utuav.py`, `select_utuav_windows.py`; chọn đoạn: `pilot_selection.json`; giải nén/GT: `prepare_utuav_pilot.py`; cache chẩn đoán: `make_gt_diagnostic_cache.py`; tách xe máy: `derive_moto_eval.py`; kiểm detector: `assess_detector_smoke.py`.
- Kết quả TrackEval chi tiết nằm dưới `results/trackeval/UTUAV-pilot*`; stdout của từng lượt `diag_*.log`, `gapv2_*.log`, `eval_*.log` trong thư mục này. Dữ liệu ảnh và archive không đưa vào Git.
- Khi SVMOT tải xong: kiểm ZIP/GT/occlusion labels và ưu tiên pilot dùng detection phát hành kèm bộ dữ liệu, nếu có, để tránh giới hạn chuyển miền của YOLO11n mixed5. Nếu muốn benchmark UTUAV thực tế, cần chốt detector phù hợp ảnh UAV và FPS nguồn trước; không thay cấu hình tracking/metric của luận văn chỉ để tạo kết quả dương.
