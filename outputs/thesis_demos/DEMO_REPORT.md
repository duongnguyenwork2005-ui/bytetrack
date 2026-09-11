# Báo cáo dựng hai video demo khóa luận

Ngày dựng: 2026-09-11 · Nhánh: `fix/tracker-evaluation-correctness` · Môi trường: conda `khoaluan-mot`
(Python 3.11.9, ultralytics 8.4.141, torch 2.6.0+cu124, GTX 1650, ffmpeg 7.1 qua `imageio-ffmpeg`).

**Nguyên tắc xuyên suốt:** mọi box, ID, quỹ đạo và con số trên video đều lấy từ **prediction đã lưu**
của ba tracker (`data/processed/trackers/DETRAC-all/runA-{cv,ekf_ctrv,ukf_ctrv}/data/<video>.txt`,
lượt A, `yolov8n.pt`, `conf=0,25`, ByteTrack mặc định) và từ **ground-truth UA-DETRAC**. Không tạo dữ
liệu giả, không sửa/ghi đè prediction nguồn hay script thí nghiệm. Toàn bộ kết quả nằm trong
`outputs/thesis_demos/`.

> **Ghi chú về Git và phạm vi minh họa**
> - File nhị phân dựng ra (`*.mp4`, `*/*.png`, `*/rerun/*`) **không nằm trong lịch sử Git** —
>   chúng tái tạo được đầy đủ bằng các lệnh ở §10, nên không cần lưu trong repo. CSV, `metadata.json`
>   và báo cáo này vẫn được theo dõi để việc tái lập có căn cứ đối chiếu.
> - Demo 1 và Demo 2 là **minh họa hai trường hợp cụ thể được chọn thủ công từ tập train** (xem §3
>   về cách chọn và kiểm tra ổn định), **không phải bằng chứng tổng quát** rằng EKF + CTRV hay
>   UKF + CTRV tốt hơn CV. Số liệu tổng quát về hiệu năng theo nhóm nằm ở `FIX_REPORT.md` và
>   `TEST_EVALUATION_REPORT.md`, không phải ở hai clip này (xem thêm §7–§8).
> - Muốn xem video: chạy lại lệnh tái tạo ở §10 (cần GPU, ~5 phút), hoặc dùng bản đã lưu ngoài
>   GitHub (máy local tạo báo cáo này, hoặc bản chia sẻ riêng nếu có).

---

## 1. Kết quả tóm tắt

| | Demo 1 — `demo_01_turning_occlusion.mp4` | Demo 2 — `demo_02_heavy_occlusion.mp4` |
|---|---|---|
| Video nguồn / GT target | `MVI_20065` (train), GT id 20 | `MVI_39781` (train), GT id 48 |
| Clip (frame nguồn) | 250 – 345 (96 frame, 3,84 s) | 1550 – 1626 (77 frame, 3,08 s) |
| Sự kiện che khuất (GT) | 239 – 419, che tối đa 37 % | 1590 – 1619, che **100 %** (0 % nhìn thấy) trong 1598 – 1613 (16 frame = 0,64 s) |
| Đoạn tracker thực sự mất dấu | 274 – 312 (39 frame = 1,56 s) | 1599 – 1618 (20 frame = 0,80 s) |
| Chuyển động của xe | rẽ nhẹ ~21° (đo trên GT giữa đầu và cuối sự kiện), đi chậm ~0,66 px/frame | đi thẳng |
| KF + CV | **145 → 218** (đổi ID) | **643 → 643** (giữ ID) |
| EKF + CTRV | **152 → 243** (đổi ID) | **646 → 734** (đổi ID) |
| UKF + CTRV | **152 → 152** (giữ ID) | **647 → 735** (đổi ID) |
| Đầu ra | 1920×1080, H.264 (libx264, crf 18), 12 fps, 138 frame, **11,5 s** | 1920×1080, H.264, 12 fps, 119 frame, **9,9 s** |

Hai clip cho hai kết cục **trái chiều**: ở Demo 1 chỉ UKF + CTRV giữ được ID; ở Demo 2 chỉ KF + CV
giữ được ID. Cả hai được giữ nguyên như dữ liệu thật cho thấy, kể cả khi Demo 2 ngược với kỳ vọng
ban đầu của đề tài.

---

## 2. Dữ liệu nguồn và cách dùng

| Thành phần trên video | Nguồn | Ghi chú |
|---|---|---|
| Box + ID của mọi xe (nét liền, mỏng) | file prediction đã lưu của từng tracker | đọc trực tiếp, không chỉnh sửa |
| Box + quỹ đạo GT của target (xanh lá, nét liền) | `data/processed/DETRAC-all/<video>/gt/gt.txt` | cột `vis` dùng để hiện "GT nhìn thấy x %" |
| Quỹ đạo dự đoán của tracker (nét đứt, màu track) và box dự đoán lúc mất dấu | **chạy lại** tracker bằng `src/demo/rerun_with_state_log.py` | xem §6 — file đã lưu **không** chứa box của track ở trạng thái LOST |
| Ghép target ↔ ID tracker mỗi frame | Hungarian trên toàn GT của frame, IoU ≥ 0,5 (`occlusion_eval.match_frame`) | tránh gán nhầm box của xe bên cạnh |
| Đánh dấu đổi ID (box đỏ, "ID cũ → ID mới" trong 15 frame) | thay đổi của ID ghép được giữa hai frame liên tiếp | |
| Số liệu trong caption | đo trên chính clip (§5, §7) | |

Kết cục "giữ / đổi ID" của từng sự kiện lấy từ bảng đã chấm ở giai đoạn sửa lỗi:
`results/runA/event_outcomes_DETRAC-all.csv` (train) và `results/test_eval/A/event_outcomes_DETRAC-test.csv` (test).

---

## 3. Quét candidate

Script: `src/demo/find_demo_candidates.py` (chỉ đọc dữ liệu có sẵn, không chạy lại tracking).

- Sự kiện đủ điều kiện (`eligible`, có đủ 3 tracker): **2 327** (train 1 003, test 1 324).
- Trong đó ba tracker cho kết cục **khác nhau**: **91** (3,9 %). Phân bố mã kết cục CV/EKF/UKF
  (P = giữ, S = đổi, L = mất, N = chưa ghép trước sự kiện): PPP 1078 · SSS 510 · NNN 455 · LLL 193 ·
  SPP 26 · PSS 24 · PPS 11 · SSP 9 · SPS 8 · PSP 8 · khác 5.

Hai bước cho mỗi demo:

1. **Xếp hạng theo tiêu chí hình học** (bảng dưới).
2. **Kiểm tra độ ổn định** — bắt buộc, vì `switched` trong bảng kết cục chỉ so ID ở frame ngay trước / ngay sau
   sự kiện. Với mỗi candidate, đọc lại prediction đã lưu, ghép target từng frame trong cửa sổ
   `[start − 25, end + 25]` và đếm số lần đổi ID thật của từng tracker. Candidate **sạch** = đúng 0 lần
   (nếu `preserved`) hoặc đúng 1 lần (nếu `switched`) ở **cả ba** tracker.

### 3.1 Demo 1 — xe rẽ + che khuất (bảng xếp hạng, 15 candidate)

Tiêu chí: `turn_deg` = đổi hướng GT giữa 10 frame đầu và 10 frame cuối sự kiện (NaN nếu xe dịch < 5 px
trong cửa sổ → hướng không xác định); `curv` = path/net − 1 trong đoạn che; lọc `net_disp ≥ 40 px`;
điểm cộng nếu CTRV giữ ID mà CV không. Cột `nchg` = số lần đổi ID thật (CV / EKF / UKF).

| # | Nguồn | Video | GT | Sự kiện | turn_deg | curv | net px | px/frame | che max | Kết cục | nchg | Sạch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | test | MVI_40793 | 52 | 954–1017 | 16,3 | 0,022 | 298 | 4,66 | 0,26 | NPP | 2/2/2 | ✗ (chưa ghép trước sự kiện; đổi 2 lần) |
| 2 | test | MVI_40903 | 41 | 590–679 | 5,7 | 0,012 | 231 | 2,57 | 0,46 | SPP | 6/2/2 | ✗ (nhấp nháy) |
| **3** | **train** | **MVI_20065** | **20** | **239–419** | **21,6** | **0,192** | **119** | **0,66** | **0,37** | **SSP** | **1/1/0** | **✓ — CHỌN** |
| 4 | test | MVI_40792 | 20 | 974–1028 | 4,7 | 0,007 | 183 | 3,32 | 0,71 | SPP | 1/0/0 | ✓ nhưng đi thẳng |
| 5 | test | MVI_40762 | 55 | 1460–1511 | 3,5 | 0,001 | 529 | 10,2 | 0,29 | SPP | 4/4/4 | ✗ |
| 6 | train | MVI_40162 | 40 | 990–1018 | −6,5 | 0,017 | 108 | 3,72 | 0,15 | SPS | 4/0/3 | ✗ |
| 7 | train | MVI_40241 | 325 | 2255–2291 | −0,9 | 0,000 | 374 | 10,1 | 0,62 | SPP | 2/1/1 | ✗ |
| 8 | train | MVI_40201 | 47 | 283–308 | 0,4 | 0,001 | 276 | 10,6 | 0,37 | SPP | 1/0/0 | ✓ nhưng đi thẳng |
| 9 | train | MVI_40152 | 15 | 306–323 | −3,7 | 0,002 | 86 | 4,76 | 0,48 | SPP | 1/0/0 | ✓ nhưng đi thẳng |
| 10 | test | MVI_40761 | 7 | 75–94 | 4,7 | 0,006 | 57 | 2,86 | 0,65 | SPP | 4/3/3 | ✗ |
| 11 | train | MVI_40991 | 24 | 1449–1499 | 18,3 | 0,058 | 80 | 1,57 | 0,48 | SSP | 1/1/0 | ✓ — dự phòng |
| 12 | test | MVI_40854 | 7 | 93–136 | NaN | 0,057 | 42 | 0,95 | 0,17 | SPP | 12/8/8 | ✗ (135↔189 nhấp nháy ở cả 3) |
| 13 | test | MVI_40853 | 55 | 990–1403 | NaN | 0,270 | 63 | 0,15 | 0,36 | SPP | 9/10/10 | ✗ |
| 14 | test | MVI_39311 | 41 | 1271–1445 | NaN | 0,619 | 57 | 0,32 | 0,48 | SSP | 1/1/0 | ✓ nhưng xe gần như đứng yên |
| 15 | train | MVI_39801 | 37 | 701–709 | 0,0 | 0,000 | 107 | 11,9 | 0,23 | SPS | 2/0/2 | ✗ |

Candidate **sạch + CTRV giữ ID + |góc rẽ| ≥ 10° + tốc độ ≥ 0,3 px/frame**: chỉ còn **2** —
MVI_20065 t20 (chọn) và MVI_40991 t24 (dự phòng, góc rẽ 18°, đoạn mất dấu ngắn hơn).

**Vì sao chọn MVI_20065 t20:** là candidate duy nhất vừa sạch, vừa có góc rẽ đo được rõ ràng (21,6°),
vừa có net displacement đủ lớn (119 px) để gọi là "xe đang đi". Nó xếp thứ 3 theo điểm tổng vì hai
candidate trên nó đều rớt ở bước kiểm tra ổn định.

**Vì sao loại các candidate nổi bật khác:**
- MVI_40854 t7 (từng được chọn ở bản nháp đầu): cả ba tracker đều nhấp nháy ID qua lại
  (135↔189, 140↔193, 140↔197) dưới mức che chỉ 17 %; "switched" của CV chỉ là một frame lẻ →
  không phải khác biệt thật.
- MVI_39311 t41: `curv` 0,62 cao nhất, nhưng xe nhích ~76 px trong 240 frame (< 0,35 px/frame),
  hướng từng đoạn nhảy lung tung → "độ cong" là nhiễu annotation của xe gần đứng yên, không phải rẽ.
- MVI_40792 / MVI_40201 / MVI_40152: sạch và CTRV giữ ID, nhưng xe đi thẳng (|turn| < 5°) nên không
  minh họa được "rẽ".

### 3.2 Demo 2 — che khuất ≥ 90 % trong 0,5–1,5 s (bảng xếp hạng, 12 candidate)

Tiêu chí: đoạn che gần hoàn toàn (`full_len`) từ 12 đến 37 frame, `max_occ ≥ 0,90`, ưu tiên
kết cục khác nhau, xe che là phương tiện.

| # | Nguồn | Video | GT | Sự kiện | full (frame) | che max | Xe che | net px | Kết cục | nchg | Sạch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **train** | **MVI_39781** | **48** | **1590–1619** | **16 (1598–1613)** | **1,00** | xe + khác | **65** | **PSS** | **0/1/1** | **✓ — CHỌN** |
| 2 | train | MVI_39781 | 30 | 933–1007 | 16 | 1,00 | xe | 169 | NNN | 0/0/0 | ✗ (chưa ghép trước sự kiện) |
| 3 | train | MVI_39781 | 37 | 1075–1127 | 15 | 1,00 | xe | 176 | PPP | 0/0/0 | ✓ nhưng cả ba giống nhau |
| 4 | train | MVI_40212 | 136 | 1011–1056 | 21 | 1,00 | xe | 748 | SSS | 1/1/1 | ✓ nhưng cả ba giống nhau |
| 5 | test | MVI_40701 | 9 | 189–234 | 19 | 1,00 | xe | 188 | SSS | 2/2/2 | ✗ |
| 6 | test | MVI_40701 | 30 | 496–549 | 32 | 1,00 | xe | 171 | SSS | 4/3/3 | ✗ |
| 7 | train | MVI_63561 | 177 | 1206–1255 | 18 | 1,00 | xe | 814 | SSS | 1/2/2 | ✗ |
| 8 | test | MVI_39311 | 43 | 928–1033 | 32 | 1,00 | xe | 179 | NNN | 3/3/3 | ✗ |
| 9 | test | MVI_39501 | 16 | 239–440 | 35 | 1,00 | xe | 394 | NNN | 0/0/0 | ✗ |
| 10 | test | MVI_40701 | 12 | 207–271 | 17 | 1,00 | xe | 287 | SSS | 1/1/2 | ✗ |
| 11 | test | MVI_39401 | 92 | 962–1016 | 27 | 1,00 | xe | 286 | PPP | 0/0/0 | ✓ nhưng cả ba giống nhau |
| 12 | test | MVI_40853 | 30 | 325–372 | 24 | 1,00 | xe | 158 | NNN | 0/0/0 | ✗ |

**Vì sao chọn MVI_39781 t48:** có **124** sự kiện che ≥ 90 % dài 0,5–1,5 s (train + test). Trong đó
**đúng 1** sự kiện cho kết cục khác nhau giữa các tracker — chính là sự kiện này — và nó sạch. Kết cục
là **CV giữ ID, hai CTRV đổi ID** (PSS) — ngược với kỳ vọng "CTRV tốt hơn khi che nặng". Theo yêu cầu,
clip được dựng đúng như dữ liệu, không thay bằng clip "đẹp" hơn. Số sự kiện che nặng mà CTRV giữ được
ID còn CV mất: **0**. 123 sự kiện còn lại cả ba tracker cho cùng kết cục (PPP / SSS / NNN / LLL).

---

## 4. Nội dung từng clip

**Bố cục chung** (1920×1080): thanh tiêu đề (tên demo, video, frame nguồn, thời điểm, trạng thái che
khuất + % GT nhìn thấy) · ba panel 640×720 bằng nhau **KF + CV | EKF + CTRV | UKF + CTRV**, cùng
một vùng crop 400×450 phóng 1,6× từ cùng một frame nguồn · dòng trạng thái mỗi panel (ID đang bám,
số lần đổi ID, số frame mất dấu liên tục) · chú thích màu cố định · caption kết luận giữ 3,5 s cuối.
Nét liền xanh lá = quỹ đạo & box GT; nét đứt màu track = quỹ đạo dự đoán; box đứt = box dự đoán khi
mất dấu; viền vàng = đang bị che; viền cam = che ≥ 90 %; box đỏ + "DOI ID: cũ -> mới" = đổi ID.
Phát ở 12 fps (chậm 2,08× so với 25 fps gốc) để nhìn kịp.

### Demo 1 — `MVI_20065`, GT 20, frame 250–345

- Xe đi chậm (0,66 px/frame), rẽ nhẹ ~21° trên toàn sự kiện GT 239–419; trong đoạn mất dấu 274–312
  xe đi được 27 px đường (net 21 px, curv 0,29).
- Che một phần bởi xe bên cạnh; GT nhìn thấy thấp nhất 62,6 %.
- Cả ba tracker bám target đến frame 273 rồi mất dấu từ 274 (không có detection đủ IoU).
- **UKF + CTRV (ID 152)**: ghép lại được tại frame **302** và **304** (mỗi lần với một detection
  điểm thấp 0,29 / 0,28), nhờ đó bộ đếm `track_buffer` được đặt lại; lại mất 305–311; ghép ổn định
  từ **312**. Giữ ID 152 suốt clip.
- **KF + CV (145)** và **EKF + CTRV (152)**: không ghép được ở 302/304; hết 30 frame buffer nên bị
  xoá sau frame 304; khi detection ổn định trở lại (312–313) ByteTrack tạo track mới **218** (CV) và
  **243** (EKF) → đổi ID tại frame 313.
- Keyframe: `before_occlusion.png` = frame 273 (frame cuối còn bám), `during_occlusion.png` = 293,
  `reappearance.png` = 313.

### Demo 2 — `MVI_39781`, GT 48, frame 1550–1626

- Xe đi thẳng, bị che **hoàn toàn** (GT nhìn thấy 0 %) trong 1598–1613 (16 frame = 0,64 s) sau xe
  khác; sự kiện GT 1590–1619; GT kết thúc ở 1626 (xe rời cảnh) nên clip dừng ở đó.
- Cả ba tracker bám đến 1598, mất dấu 20 frame 1599–1618, box dự đoán trôi dần (sai số tâm trung bình
  12–13 px, tối đa 16 px ở cả ba).
- Detection của target xuất hiện lại ở 1617 (điểm 0,26) và 1618 (0,36) nhưng **không tracker nào**
  ghép được → ByteTrack tạo track mới chưa xác nhận ở 1618 trong cả ba lượt.
- Tại **1619** (detection điểm 0,52): **KF + CV ghép lại được ID 643**; EKF và UKF không ghép được,
  detection rơi vào track mới → ID **734** / **735** được kích hoạt tại 1619.
- Keyframe: `before_occlusion.png` = 1598, `during_occlusion.png` = 1606 (che 100 %),
  `reappearance.png` = 1619.

---

## 5. Số liệu đo trên chính clip (từ `metadata.json`)

### Demo 1

| | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|
| ID trước / sau sự kiện | 145 / 218 | 152 / 243 | 152 / 152 |
| Đổi ID trong clip | 1 (frame 313) | 1 (frame 313) | 0 |
| Frame target mất dấu trong clip | 57 | 57 | 62 |
| Sai số tâm box dự đoán so với GT khi mất dấu (px): trung bình / tối đa / cuối | 4,00 / 11,6 / 5,7 | 4,68 / 12,1 / 5,5 | 4,04 / 11,6 / 5,9 |
| Recall trên sự kiện GT 239–419 | 0,619 | 0,619 | 0,635 |
| Chord effect (độ lệch trung bình khỏi dây cung: GT / dự đoán, px) | −2,84 / −2,76 | −2,84 / −4,60 | −2,59 / +0,46 |

### Demo 2

| | KF + CV | EKF + CTRV | UKF + CTRV |
|---|---|---|---|
| ID trước / sau sự kiện | 643 / 643 | 646 / 734 | 647 / 735 |
| Đổi ID trong clip | 0 | 1 (frame 1619) | 1 (frame 1619) |
| Frame target mất dấu trong clip | 20 | 20 | 20 |
| Sai số tâm box dự đoán khi mất dấu (px): trung bình / tối đa / cuối | 12,58 / 16,0 / 13,6 | 12,94 / 16,3 / 14,4 | 12,15 / 16,0 / 11,9 |
| Frame đầu tiên target được ghép lại (bất kỳ ID) | 1619 | 1619 | 1619 |
| Chord effect (GT / dự đoán, px) | +1,51 / −3,20 | +1,51 / −4,50 | +1,51 / −4,45 |

**Chord effect:** không chú thích trên video nào. Ở Demo 1, dự đoán của UKF nằm gần dây cung hơn GT
2,1 px (CV: 0,1 px) — dưới ngưỡng 3 px mà script đặt ra vì tâm box GT vẽ tay của UA-DETRAC nhiễu
cỡ 1–2 px; ở Demo 2 không tracker nào co vào phía trong. Cơ chế chú thích đã cài trong
`render_thesis_demo.py` (`chord_effect()`, nhãn `[Chord effect]` trên dòng trạng thái) và chỉ bật khi
vượt ngưỡng.

---

## 6. Bằng chứng clip phản ánh đúng prediction thực tế

1. **Box và ID** vẽ trên video đọc trực tiếp từ file prediction đã lưu; script không sửa file nào trong
   `data/`, `results/`, `src/` (ngoài thêm mới trong `src/demo/`).
2. **Quỹ đạo / box dự đoán khi mất dấu** không có trong file đã lưu (ByteTrack chỉ xuất track ở trạng
   thái Tracked). Chúng lấy từ việc **chạy lại** tracker từ frame 1 đến frame cuối clip bằng đúng
   `yolov8n.pt`, đúng file cấu hình tracker, `conf=0,25`, đúng cách tô xám `ignored_region` và
   `persist=True` như `baseline_track.py`, rồi đọc trạng thái Kalman/EKF/UKF trong runtime
   (`rerun_with_state_log.py`). Vì tracking trên GPU không bit-reproducible, script **bắt buộc đối
   chiếu** box xuất ra của lần chạy lại với prediction đã lưu trong cửa sổ clip (cùng frame, cùng
   ID, IoU ≥ 0,9) và tự dừng nếu khớp < 99 %. Kết quả **hai lần chạy lại độc lập** (lần đầu và lần
   `--no-cache` cuối cùng) đều khớp **100 %**:

   | Demo | Tracker | Box khớp cùng ID / box đã lưu | Frame cùng tập ID |
   |---|---|---|---|
   | 1 | CV | 1430 / 1430 (100 %) | 96 / 96 |
   | 1 | EKF | 1412 / 1412 (100 %) | 96 / 96 |
   | 1 | UKF | 1416 / 1416 (100 %) | 96 / 96 |
   | 2 | CV | 435 / 435 (100 %) | 77 / 77 |
   | 2 | EKF | 436 / 436 (100 %) | 77 / 77 |
   | 2 | UKF | 433 / 433 (100 %) | 77 / 77 |

3. **Kết cục trên video trùng với bảng đã chấm** `event_outcomes_*.csv` (Demo 1: SSP với
   145→218, 152→243, 152; Demo 2: PSS với 643, 646→734, 647→735) và với chuỗi ID theo frame đọc lại
   từ file đã lưu (§3, cột `chain_*` trong `candidates_demo{1,2}.csv`).
4. **Cơ chế ghép cặp được đo tường minh** (`match_cost_at_reacquisition.py`): ByteTrack mặc định
   ghép khi `IoU(box dự đoán, detection) × score > 0,2` (`match_thresh 0,8`, `fuse_score True`).
   Box dự đoán lấy từ log ngay sau `multi_predict`, trước khi ghép (`rerun_pred_*.csv`); detection
   tính lại bằng cùng model trên cùng frame đã tô xám.

   *Demo 1 — frame quyết định:*

   | Frame | GT thấy | det: IoU với GT / score | CV: IoU × s | EKF: IoU × s | UKF: IoU × s |
   |---|---|---|---|---|---|
   | 302 | 0,73 | 0,757 / 0,292 | 0,587 × → **0,172** ✗ | 0,440 × → **0,129** ✗ | 0,834 × → **0,244** ✓ |
   | 304 | 0,69 | 0,828 / 0,275 | 0,577 × → **0,159** ✗ | 0,416 × → **0,114** ✗ | 0,854 × → **0,235** ✓ |
   | 305 | 0,67 | 0,118 / 0,462 | track đã bị xoá | track đã bị xoá | 0,163 × → 0,076 ✗ |
   | 312 | 0,67 | 0,794 / 0,342 | — | — | 0,804 × → **0,275** ✓ |

   → UKF giữ ID vì box dự đoán của nó chồng lên detection tốt hơn hẳn (IoU 0,83 so với 0,59 của CV
   và 0,44 của EKF) đúng lúc detection điểm thấp xuất hiện. Lưu ý **EKF + CTRV kém hơn cả CV** ở đây,
   nên không thể quy khác biệt cho riêng "mô hình CTRV".

   *Demo 2 — frame quyết định:*

   | Frame | GT thấy | det: IoU với GT / score | CV: IoU × s | EKF: IoU × s | UKF: IoU × s |
   |---|---|---|---|---|---|
   | 1617 | 0,56 | 0,707 / 0,257 | 0,491 × → 0,126 ✗ | 0,419 × → 0,108 ✗ | 0,442 × → 0,114 ✗ |
   | 1618 | 0,66 | 0,684 / 0,361 | 0,498 × → 0,180 ✗ | 0,406 × → 0,147 ✗ | 0,433 × → 0,157 ✗ |
   | **1619** | 0,75 | 0,694 / 0,524 | 0,432 × → **0,227** ✓ | 0,379 × → **0,199** ✗ | 0,380 × → **0,199** ✗ |
   | 1620 | 1,00 | 0,807 / 0,472 | 0,881 × → 0,416 ✓ | 0,359 × → 0,170 (track mới 734: 0,420) | 0,387 × → 0,183 (track mới 735: 0,423) |

   → Chênh lệch chỉ **0,001 dưới ngưỡng** với EKF/UKF. Kết cục PSS của clip này là một trường hợp
   **sát ngưỡng**, không phải bằng chứng CV "tốt hơn" khi che nặng.

---

## 7. Vì sao KHÔNG đưa các số mô phỏng / thống kê nhóm vào video

- **217,41 px · 14,51 px · 26,61 px** là kết quả **mô phỏng** (quỹ đạo tổng hợp), không đo trên
  clip nào trong hai video. Sai số dự đoán đo được trên clip thật nhỏ hơn rất nhiều (Demo 1: trung bình
  4,0–4,7 px; Demo 2: 12–13 px) vì xe đi chậm và đoạn mất dấu ngắn. Đưa số mô phỏng lên video sẽ gán
  cho clip một kết quả nó không có → **không dùng**.
- **6,41 % · 11,54–12,82 % · 7,69 %** là thống kê **tổng hợp của nhóm che khuất** ở bản đánh giá
  **trước khi sửa lỗi** (đếm trùng full/partial, Hungarian-rồi-mới-ngưỡng…, xem `FIX_REPORT.md`).
  Sau khi sửa, tỷ lệ giữ ID trên nhóm sự kiện đủ điều kiện (lượt A, train) là CV 48,3 % / EKF 47,5 % /
  UKF 47,6 % và trên tập test là 48,2 % / 48,9 % / 48,7 % — ba tracker cách nhau ~1 điểm phần trăm.
  Dù là bản nào, đó cũng là **số của quần thể**, không phải metric của một clip → **không dùng để nói
  về clip**. Caption của cả hai video ghi rõ "Kết quả của riêng clip này".

---

## 8. Những điểm chưa thể xác minh / hạn chế

1. **Không có clip nào mà EKF + CTRV thắng khi rẽ.** Kiểm tra trên **toàn bộ** sự kiện (không chỉ
   top 15): với |góc rẽ| ≥ 10°, net ≥ 40 px, CTRV giữ ID và CV không, chỉ có **3** sự kiện —
   MVI_20065 t20 và MVI_40991 t24 (cả hai sạch, **UKF** giữ, EKF đổi) và MVI_40793 t52 (test, EKF và
   UKF giữ nhưng không sạch: CV chưa ghép được trước sự kiện, cả ba tracker đổi ID 2 lần trong cửa sổ).
   Demo 1 nói đúng điều đó: EKF cũng đổi ID.
2. **Không có clip che nặng ≥ 90 % nào mà CTRV thắng.** Sự kiện duy nhất khác kết cục là PSS
   (CV thắng) với biên **0,001** dưới ngưỡng ghép — về bản chất là ngẫu nhiên. Demo 2 vì thế chỉ minh
   họa "che nặng trông như thế nào và ba tracker hành xử ra sao", không minh họa ưu thế của mô hình nào.
3. **Demo 1 là xe đi chậm (0,66 px/frame) rẽ nhẹ (21°).** Trong 39 frame mất dấu xe chỉ đi 27 px,
   nên khác biệt giữa ba bộ lọc là vài pixel. Ưu thế của UKF dựa vào hai detection điểm thấp
   (0,29 và 0,28) ở frame 302/304 với biên 0,04 trên ngưỡng. Không tìm được ca xe rẽ nhanh, mất dấu
   dài, kết cục sạch và khác nhau trong dữ liệu hiện có.
4. **Chưa tách được nguyên nhân** vì sao UKF dự đoán tốt hơn CV mà EKF (cùng mô hình CTRV) lại kém
   hơn CV ở Demo 1: có thể do cách lan truyền hiệp phương sai (sigma point vs Jacobian) hoặc tham số
   nhiễu, một clip không đủ để kết luận.
5. **Detection trong bảng chi phí ghép (§6.4) được tính lại** bằng `model.predict()`; GPU không
   bit-reproducible nên `score`/toạ độ có thể lệch cỡ 10⁻³ so với lượt tracking gốc. Kết luận
   ghép/không ghép ở mọi frame trong bảng **trùng** với những gì prediction đã lưu thể hiện, nhưng con
   số 0,199 so với 0,200 ở Demo 2 nằm trong biên độ đó.
6. **Cả hai clip thuộc tập train** (`DETRAC-all`, lượt A). Trong tập test không có candidate rẽ sạch
   với xe đang chuyển động (ứng viên test tốt nhất MVI_39311 t41 là xe đứng yên), và sự kiện che nặng
   khác kết cục duy nhất nằm ở train.
7. **Chord effect không đo được ở độ phân giải này**: khác biệt 2 px với tâm box GT vẽ tay (nhiễu 1–2 px).
8. **Góc rẽ 21°** đo trên GT giữa 10 frame đầu và 10 frame cuối sự kiện, trong hệ toạ độ ảnh (không
   phải góc lái thật của xe).
9. Quỹ đạo dự đoán (nét đứt) đến từ lần chạy lại chứ không từ file đã lưu — đã đối chiếu 100 % (§6.2)
   nhưng vẫn là dữ liệu tái tạo.

---

## 9. Danh sách file đã tạo

```
outputs/thesis_demos/
├── DEMO_REPORT.md                                  (file này)
├── demo_01_turning_occlusion.mp4                   1920x1080 H.264 12 fps, 138 frame, 11,5 s, 7,4 MB
├── demo_01_turning_occlusion/
│   ├── before_occlusion.png                        frame 273
│   ├── during_occlusion.png                        frame 293
│   ├── reappearance.png                            frame 313
│   ├── metadata.json                               nguồn, cửa sổ, GT target, mức che, metric từng tracker,
│   │                                               đường dẫn prediction, kết quả đối chiếu, chord effect,
│   │                                               bảng chi phí ghép
│   ├── match_cost_at_reacquisition.csv             frame 298–313
│   └── rerun/                                      cache chạy lại (không commit; tái tạo bằng --no-cache)
│       ├── rerun_boxes_runA-{cv,ekf_ctrv,ukf_ctrv}.csv    box xuất ra (để đối chiếu)
│       ├── rerun_states_runA-*.csv                        trạng thái sau frame (tracked + lost)
│       └── rerun_pred_runA-*.csv                          box dự đoán trước ghép cặp
├── demo_02_heavy_occlusion.mp4                     1920x1080 H.264 12 fps, 119 frame, 9,9 s, 4,3 MB
├── demo_02_heavy_occlusion/
│   ├── before_occlusion.png                        frame 1598
│   ├── during_occlusion.png                        frame 1606
│   ├── reappearance.png                            frame 1619
│   ├── metadata.json
│   ├── match_cost_at_reacquisition.csv             frame 1612–1621
│   └── rerun/                                      (như trên)
├── candidates_demo1.csv                            15 candidate + kiểm tra ổn định
├── candidates_demo2.csv                            12 candidate + kiểm tra ổn định
└── all_eligible_events_3trackers.csv               2 327 sự kiện, kết cục 3 tracker, độ cong, mức che

src/demo/
├── find_demo_candidates.py                         quét + xếp hạng + kiểm tra ổn định
├── rerun_with_state_log.py                         chạy lại tracker, log trạng thái LOST + box dự đoán, đối chiếu
├── render_thesis_demo.py                           dựng video 3 panel, keyframe, metadata (cấu hình 2 demo ở DEMOS)
└── match_cost_at_reacquisition.py                  đo IoU x score tại frame tái ghép
```

Không file nào trong `data/`, `results/`, `configs/` hay các script thí nghiệm cũ bị sửa.

## 10. Lệnh tái tạo

```bash
conda activate khoaluan-mot
cd D:/LABOTARY/bytetrack

# 1. Quét candidate (chỉ đọc CSV + prediction đã lưu, ~1 phút)
python src/demo/find_demo_candidates.py

# 2. Dựng cả hai video từ đầu: chạy lại 6 lượt tracker (~5 phút trên GTX 1650), đối chiếu, render, H.264
python src/demo/render_thesis_demo.py --demo all --no-cache

# 3. Bảng chi phí ghép tại frame tái ghép (YOLO trên vài frame, < 1 phút)
python src/demo/match_cost_at_reacquisition.py --demo 1 --frames 298-313
python src/demo/match_cost_at_reacquisition.py --demo 2 --frames 1612-1621

# 4. Render lại (dùng cache chạy lại) để nhúng bảng chi phí ghép vào metadata.json
python src/demo/render_thesis_demo.py --demo all

# Từng video riêng: --demo 1 hoặc --demo 2
```

Lưu ý: bước 2 phụ thuộc GPU; nếu lần chạy lại không khớp ≥ 99 % với prediction đã lưu, script dừng
và không xuất video (chưa từng xảy ra trong 2 lần chạy trên máy này).
