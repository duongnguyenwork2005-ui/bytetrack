# R65 — chẩn đoán PF500 thấp trên RoundaboutHD prefix3000 (02/11/2026, lượt 2)

**Quyết định và phạm vi** (khóa trước khi chạy; GPT 02/11 Q2:
`D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/answer.json`; `00_giao_thuc.json`):
- Dữ liệu: imagesc001–004, frame 1–3000.
- Detector, ngưỡng: chỉ yolov8n_mixed5, c25.
- Bộ lọc: CV, EKF, UKF và PF500 seed 0/1/2, tổng 24 đơn vị camera × bộ lọc/seed.
- Đầu vào: cache, GT SCT đã sửa và cấu hình giữ nguyên R64 (15 fps, buffer 18 bước, elapsed steps). PF giữ N = 500,
  ESS 0,5N, ước lượng mean, không roughening.
- **Không tune, không thay benchmark R64, không ablation.** Đây chỉ là lượt tìm bằng chứng, không phải điều kiện nghiệm
  thu R64.

**Hai lượt chạy:**
- **Lượt 1** (`r65{off,on}-*`, `*_luot1.json`, `10_ket_luan_luot1_nhap.md`): GPT review giữa chừng (bị ngắt do hết hạn
  mức, `runs_local/hoi_codex_R65_review_20261102/stderr.log`) chỉ ra 6 lỗi phương pháp:
  1. Observer bọc `PFCTRV.propagate`, nhưng PF dự đoán qua `PFSTrack.multi_predict` (vector hóa). Bộ đếm lượt 2 xác
     nhận `pf_propagate` = 0.
  2. Nối track–GT bằng Hungarian trước, lọc IoU sau.
  3. Nhãn "tranh chấp" chỉ xét detection gần GT nhất, bỏ qua ngưỡng chi phí; 88–89 % ca "tranh chấp" của PF thực ra
     không có detection nào khả thi.
  4. Chưa có phân mảnh theo evaluator.
  5. Chưa tách số frame thiếu quan sát.
  6. So độ trải hạt **sau** update với dịch chuyển GT, trong khi cái cần so là độ trải trước association.
- **Lượt 2** (`r65b{off,on}-*`, báo cáo này): đã sửa cả 6. Mọi số dưới đây là của lượt 2. Kết luận lượt 1 về
  "tranh chấp gấp 2,3 lần" **bị thay**.
- **Review GPT lần 2** (`runs_local/hoi_codex_R65_review2_20261102/answer.json`, CẦN SỬA TRƯỚC COMMIT) xác nhận 5/6 sửa
  đúng và số khớp `thong_ke.json`. Các sửa tiếp theo **không chạy lại tracking**:
  - nghĩa thời gian tính offline (mục 6.1);
  - finding mục 7 viết lại;
  - giảm khẳng định nhân quả (mục 8);
  - siết cổng và fixture (mục 1);
  - provenance hậu kiểm;
  - số mục 6 ghi theo khoảng ba seed.

  Log: `04_phan_tich_sua.log`, `05_parity_cong_siet.log`.

Mã: `src/r65_pf_roundabout/` (`quan_sat.py`, `chan_doan.py`, `phan_tich.py`, `test_quan_sat.py` 5/5). Kết quả:
`results/phase2/r65_pf_roundabout_diag/` (`phan_tich/thong_ke.json`, `thong_ke.csv`, `phan_tich/mau_*.csv.gz`, `quan_sat/`).
Log: `02_luot2_chay_cham_parity.log`, `03_phan_tich.log`, `track_logs/`, `ngan_sach.jsonl`. Manifest: `manifest_sha256.json`.

## 1. Cổng "log không đổi hành vi" — ĐẠT 24/24 (`parity.json`)

Mỗi đơn vị chạy hai nhánh, cả hai cùng observer chỉ đọc `quan_sat.py`:
- `off`: không log.
- `on`: bật `--assoc-log --pf-diag`.

Observer băm nối chuỗi tại ba ranh giới:
- **Sau dự đoán, trước association** (bọc `tracker.multi_predict`): mean, covariance; riêng PF thêm particles, logw,
  trạng thái RNG.
- **Sau mỗi `update` của tracker:** mọi track Tracked/Lost.
- **Từng bước PF:** `update_particles`, `reseed_motion`, `sample_initial`.

Kết quả so khớp:
- MOT `data/<v>.txt` và `data_cls/<v>.csv`: **trùng byte off = on = R64 gốc**.
- `per_video_metrics` và `combined_metrics`: off = on = R64.
- Ba chuỗi hash, bộ đếm, `pf_update.csv` và `pf_prior.csv`: off = on.

Sự kiện PF mà dữ liệu thật đi qua (tổng 3 seed):

| Sự kiện | Số lần |
|:--|--:|
| resample | 152.485 |
| reseed chuyển động | 5.048 |
| update từ Lost | 6.148 |
| suy biến | **0** |

Suy biến được phủ bằng fixture `test_quan_sat.py`.

**Cổng siết sau review GPT lần 2** (`05_parity_cong_siet.log`, `parity.json`; bản cổng cũ `parity_luot2_cong_cu.json`).
Không chạy lại tracking; cổng kiểm thêm:
- **Hoàn tất từng run:** `run_meta` có status complete, và mỗi video có `video_meta` hoàn tất đúng fingerprint
  (`run_state`).
- **Đủ đơn vị:** đúng 24 đơn vị mỗi nhánh.
- **Nội dung trace:** mỗi trace đủ 3.000 frame và hook sau dự đoán thực sự chạy.
- **Riêng PF:** số dòng `pf_prior` bằng `du_doan_track`, số dòng `pf_update` bằng `pf_update_particles`, và mọi giá trị
  hữu hạn.
- **Độ phủ sự kiện:** resample, reseed, update từ Lost và refind trên dữ liệu thật đều > 0.
- **Fixture là điều kiện đạt:** `test_quan_sat.py` 5/5. Kết quả kèm sha256 của test và `quan_sat.py` được ghi trong
  `parity.json`. Năm test gồm:
  - suy biến + resample: so particles/logw/mean/cov/RNG giữa diag bật/tắt và observer bật/tắt;
  - Lost/refind: so output **và** hash trạng thái từng frame, chụp độc lập với observer;
  - trace ghi ra file đi qua cổng;
  - test âm cho cổng trace: bắt được trace thiếu frame, hook không chạy, thiếu chuỗi hash, lệch số dòng, giá trị không
    hữu hạn;
  - test âm cho cổng hoàn tất (thêm sau review GPT lần 3): `kiem_hoan_tat_run`/`xong` từ chối run_meta đang chạy,
    thiếu fingerprint (kể cả khi hai phía cùng thiếu), fingerprint rỗng, `video_meta` chưa hoàn tất, fingerprint lệch,
    trace thiếu hoặc hỏng.

  `xong()` nay kiểm nội dung hoàn tất thay vì chỉ kiểm file tồn tại.

**Provenance:**
- **Trước chạy:** `manifest_sha256.json`, giữ nguyên. `kiem_manifest` nay kiểm thêm `gt_log` và phiên bản
  thư viện/backend. Harness đã sửa sau lượt r65b được đối chiếu với snapshot mã lúc chạy
  (`results/phase2/r65_pf_roundabout_diag/ma_luot2_snapshot/`, `SHA256SUMS.txt`).
- **Hậu kiểm, có thời điểm:** `manifest_hau_kiem.json` gồm mã phân tích/test/harness hiện tại, backend ultralytics
  8.4.141 thực nạp, evaluator TrackEval thực nạp, artifact kết luận và dung lượng cả hai lượt.

## 2. Chuyển động GT (SCT, frame liên tiếp, n = 75.316)

| | trung vị | p90 | p95 |
|:--|--:|--:|--:|
| dịch tâm / chiều cao box | 0,097 | 0,269 | 0,328 |
| \|h_t / h_{t−1} − 1\| | 0,013 | 0,037 | 0,050 |
| chiều cao box (px) | 72,7 | 239,9 | 345,3 |

## 3. Sai số dự đoán trước association

**Cách nối track với GT:**
1. Ghép một-một giữa output MOT và GT ở **quan sát trước** (frame output gần nhất < t); cặp IoU < 0,5 bị loại
   **trước** Hungarian.
2. Theo cùng GT đó sang frame t.
3. Tính e = ‖c_pred − c_GT‖ / h_GT.

`gap` = t − frame quan sát trước, nên số frame **thiếu quan sát** = gap − 1.

**Độ phủ mẫu:**

| Bộ lọc | hợp lệ | không nối được GT | thiếu GT ở t |
|:--|--:|--:|--:|
| CV / EKF / UKF | 54.136–54.161 | 22.015–22.144 | 1.351–1.367 |
| PF s0–s2 | 53.690–53.917 | 30.272–31.096 | 1.577–1.644 |

PF có nhiều mẫu "không nối được" hơn, nên các thống kê dưới đây có thể **chệch có lợi cho PF**.

**Sai số e (trung vị / p90) theo số frame thiếu quan sát:**

| Bộ lọc | thiếu 0 (gap 1) | thiếu 1–2 | thiếu 3–8 | thiếu ≥ 9 | IoU(pred, GT) trung vị | IoU < 0,5 |
|:--|:--|:--|:--|:--|--:|--:|
| CV | 0,063 / 0,151 | 0,122 / 0,362 | 0,274 / 1,30 | 0,80 / 3,24 | 0,783 | 11,5 % |
| EKF | 0,060 / 0,144 | 0,113 / 0,315 | 0,266 / 1,01 | 0,86 / 2,57 | 0,787 | 11,5 % |
| UKF | 0,060 / 0,145 | 0,114 / 0,320 | 0,269 / 1,05 | 0,86 / 2,70 | 0,786 | 11,7 % |
| PF s0 | 0,098 / 0,257 | 0,174 / 0,568 | 0,533 / 1,74 | 1,65 / 4,46 | 0,686 | 23,0 % |
| PF s1 | 0,098 / 0,256 | 0,175 / 0,564 | 0,529 / 1,76 | 1,58 / 4,52 | 0,686 | 23,8 % |
| PF s2 | 0,097 / 0,258 | 0,179 / 0,570 | 0,524 / 1,73 | 1,58 / 4,26 | 0,686 | 23,4 % |

Sai số kích thước (trung vị):
- |h/h_GT − 1|: CV 0,066; PF 0,079–0,080.
- |w/w_GT − 1|: CV 0,103; PF 0,131–0,134.

**Tập chung:** 35.465 bộ (video, GT, frame), gap = 1, hợp lệ ở cả 6 run.

| Bộ lọc | e trung vị / p90 | IoU trung vị |
|:--|:--|--:|
| CV | 0,063 / 0,148 | 0,810 |
| EKF / UKF | 0,060 / 0,141–0,142 | 0,813–0,814 |
| PF s0–s2 | 0,095–0,096 / 0,247–0,251 | 0,739–0,742 |

## 4. Kết quả association của mẫu hợp lệ (đã sửa phân loại)

Với track không được ghép trong khi GT còn ở frame t, phân loại theo **mọi** detection phù hợp (IoU det–GT ≥ 0,5):
- **khả thi:** tầng 1 (det high, track Tracked + Lost) có chi phí fused 1 − IoU·conf ≤ 0,8; tầng 2 (det low, chỉ
  Tracked) có 1 − IoU ≤ 0,5;
- **"chi phí không đạt":** có det phù hợp nhưng không det nào khả thi;
- **"tranh chấp":** mọi det phù hợp và khả thi đều bị track khác lấy;
- **"khả thi còn trống":** còn det khả thi chưa ai lấy. Không xuất hiện.

**Tỉ lệ trên toàn bộ mẫu hợp lệ:**

| Bộ lọc | ghép đúng GT | không có det phù hợp | chi phí không đạt | tranh chấp | ghép det vật khác |
|:--|--:|--:|--:|--:|--:|
| CV | 77,16 % | 17,56 % | 3,68 % | 0,89 % | 0,70 % |
| EKF | 77,19 % | 17,71 % | 3,57 % | 0,84 % | 0,68 % |
| UKF | 77,09 % | 17,64 % | 3,73 % | 0,86 % | 0,68 % |
| PF s0–s2 | 71,17–71,44 % | 18,51–18,67 % | **8,63–8,83 %** | 0,68–0,75 % | 0,66–0,68 % |

**"Chi phí không đạt" theo số frame thiếu quan sát (số ca):**

| Bộ lọc | thiếu 0 | thiếu 1–2 | thiếu 3–8 | thiếu ≥ 9 |
|:--|--:|--:|--:|--:|
| CV | 173 | 213 | 467 | 1.142 |
| PF s0–s2 | 356–371 | 482–509 | 1.359–1.431 | 2.397–2.527 |

Trong các ca "chi phí không đạt", IoU giữa box dự đoán và det phù hợp tốt nhất có trung vị CV 0,225 (p90 0,64) và PF
0,058–0,066 (p90 0,52–0,54).

- **Đã thay so với lượt 1:** "tranh chấp" thật nhỏ và **gần bằng nhau** giữa các bộ lọc (PF 0,68–0,75 %, CV 0,89 %).
  Câu "PF tranh chấp gấp 2,3 lần" của lượt 1 là sai.
- **Phần chênh của PF nằm ở "chi phí không đạt":** khoảng 2,4 lần CV. Phần lớn rơi vào track đã thiếu ≥ 3 frame quan
  sát (Lost).
- Tập chung gap = 1 chênh rất nhỏ: chi phí không đạt CV 0,34 % so với PF 0,85–0,88 %; ghép đúng 95,3 % so với
  94,8 %.

## 5. Phân mảnh và sự kiện

| Bộ lọc | Frag | MT / PT / ML | IDSW | track/GT (nối IoU) TB | đổi track liên tiếp |
|:--|--:|:--|--:|--:|--:|
| CV | 1.944 | 79 / 232 / 41 | 471 | 2,34 | 579 |
| EKF | 1.966 | 76 / 235 / 41 | 477 | 2,34 | 573 |
| UKF | 1.945 | 74 / 237 / 41 | 494 | 2,37 | 591 |
| PF s0–s2 | 2.262–2.323 | 46–48 / 258–260 / 45–46 | 1.016–1.073 | 3,91–4,05 | 1.075–1.120 |

Frag, MT, PT, ML lấy từ CLEAR của TrackEval (evaluator `mot` đã khóa) trên 352 GT ID.

## 6. Thống kê nội bộ PF (3 seed gần trùng)

### 6.1 Khoảng cách update (sửa theo review GPT lần 2, tính offline, không đổi trace)

`gap` thô trong `pf_update` = frame hiện tại − `t.frame_id`. Giá trị này đọc **sau** khi `STrack.update` đã gán
`frame_id` hiện tại, nên mọi update Tracked ghi 0 (146.340 dòng trên 3 seed). Ghi giữ nguyên ở `gap_raw_x_state`.

**Khoảng cách đúng** = `n_lost` của `assoc_log` tracks (`frame_id − end_frame`, đo sau dự đoán, trước association),
nối theo (video, frame, track_id):
- Kiểm một-một (`validate="one_to_one"`) và đủ dòng: 0 dòng không nối được.
- Số frame thiếu quan sát = max(khoảng cách − 1, 0).

| State lúc update | khoảng cách | số update / seed (s0 · s1 · s2) | frame thiếu: trung vị (p90) |
|:--|:--|:--|:--|
| Tracked (1) | 1 | 48.824 · 48.747 · 48.769 | 0 (0) |
| Lost (2) | 2–3 | 1.407 · 1.382 · 1.392 | 2 (6), tính chung cả nhóm Lost |
| Lost (2) | 4–9 | 537 · 531 · 553 | |
| Lost (2) | 10–19 | 111 · 117 · 118 | |
| Removed (3) | 20 | 10 · 7 · 4 | 19 (19), xem mục 7 |

State do observer đọc khớp `assoc_log`, trừ 1.646 / 1.704 / 1.673 update (seed 0/1/2) của track **chưa xác nhận**. Ở các update này observer đọc
state = Tracked (1), còn `assoc_log` ghi mã 0 cho track chưa xác nhận; khoảng cách vẫn là 1.

`n_lost` trong `pf_diag` cũng **không** phải số frame thiếu quan sát.

### 6.2 ESS và resample

Chỉ tính dòng update không suy biến. Dòng `init_motion` (1.654–1.714) đếm riêng.
- **ESS/N trước resample:** trung vị 0,188, p90 0,394.
  - ESS/N < 0,5: 99,98–99,99 % số update.
  - ESS/N < 0,2: 52,6–52,8 %.
  - ⇒ resample ở gần như mọi update.
- **Số tổ tiên khác nhau / N** (lấy từ `last_resample_idx`): trung vị 0,356.

### 6.3 Độ trải đám hạt trước association (`pf_prior`) so với sai số thực

Đo ngay sau `multi_predict`, trên cùng (track, frame) với mẫu hợp lệ: 53.690–53.917 mẫu, 0 mẫu thiếu prior. Mỗi ô ghi
**khoảng của ba seed** (min–max).

| Số frame thiếu | spread_xy/h trung vị | spread_xy/h p90 | spread_v/h trung vị | e_px / spread_xy trung vị | p90 | e ≤ 1·spread | e ≤ 2·spread |
|:--|--:|--:|--:|--:|--:|--:|--:|
| 0 | 0,0950–0,0951 | 0,1169–0,1186 | 0,0169–0,0170 | 0,959–0,969 | 2,67–2,71 | 51,6–52,2 % | 82,1–82,3 % |
| 1–2 | 0,141–0,142 | 0,259–0,268 | 0,0195–0,0199 | 1,02–1,06 | 3,51–3,70 | 47,6–48,7 % | 75,6–76,3 % |
| 3–8 | 0,248–0,251 | 0,628–0,657 | 0,0226–0,0229 | 1,50–1,63 | 6,64–6,73 | 33,2–34,3 % | 57,3–59,0 % |
| ≥ 9 | 0,506–0,518 | 1,644–1,671 | 0,0291 | 2,06–2,32 | 8,32–8,80 | 22,1–24,2 % | 45,9–48,8 % |

- spread_xy là độ lệch RMS bán kính 2D của đám hạt (px); spread_v là độ lệch chuẩn vận tốc.
- Mẫu số: spread/h dùng chiều cao box **dự đoán**; e dùng chiều cao **GT** (e_px = e·h_GT).
- "Số frame thiếu" ở bảng này theo `gap` của mục 3 (t − frame output trước − 1).

## 7. Finding phụ (không đặc thù PF, không sửa)

**Mã liên quan:** backend tracker **thực nạp** là ultralytics 8.4.141 cài trong env
(`site-packages/ultralytics/trackers/utils/stracks.py::merge_track_pools`; hash ở `manifest_hau_kiem.json`). Bản vendor
`third_party/ultralytics_trackers/utils/stracks.py` có hàm giống hệt. Hàm này trừ `removed_stracks` khỏi pool Lost
**trước** khi nối danh sách removed của frame hiện tại. Hệ quả:

1. **Track vừa bị đánh dấu Removed vẫn được dự đoán thêm một lần.** Track bị đánh dấu Removed ở frame t
   (`frame_id − end_frame > 18`) còn nằm trong pool Lost và vẫn được dự đoán, đưa vào association ở frame t+1
   (`assoc_log` state 3, n_lost = 20; khoảng 1.039–1.676 dòng/run).
2. **Nếu không được ghép ở t+1:** track bị trừ khỏi pool ở cuối frame đó, tức dư đúng một lần association.
3. **Nếu được ghép ở t+1** (CV 8, EKF 6, UKF 5, PF 10/7/4 lần mỗi run; tổng 40):
   - Track được tái kích hoạt nhưng **vẫn nằm trong `removed_stracks`**.
   - Lần mất quan sát kế tiếp, nó bị trừ khỏi pool Lost ngay, không có buffer.
   - Kiểm trên `assoc_log` của cả 6 run: đủ 40/40 track hồi sinh đều không còn trong pool sau lần mất kế tiếp. Ví dụ:
     CV imagesc002 track 407 hồi sinh ở frame 1100, bám tới 1108, mất ở 1109 rồi không xuất hiện nữa.

Hiện tượng có ở mọi bộ lọc với số ca cùng bậc. **Chưa có bằng chứng định lượng về mức đóng góp vào khoảng thua của
PF.** Ghi lại để mô tả đúng buffer thực tế. Theo giao thức: không sửa, không chạy thay.

## 8. Kết luận — chuỗi bằng chứng quan sát được, chưa xác lập nguyên nhân

1. **Box dự đoán của PF500 lệch GT nhiều hơn CV/EKF/UKF ngay ở frame liền kề.** Trên tập chung, sai số tâm trung vị
   0,095 h so với 0,060–0,063 h, IoU 0,74 so với 0,81. Khi track thiếu quan sát, khoảng cách tăng: thiếu ≥ 3 frame thì
   sai số của PF khoảng gấp đôi CV. Kích thước box cũng lệch hơn.
2. **Trong mẫu nối được GT, chênh lệch association quan sát được tập trung ở nhóm "có detection phù hợp nhưng chi phí
   không đạt ngưỡng"**: 8,6–8,8 % so với 3,6–3,7 %, chủ yếu ở track đã thiếu quan sát. "Tranh chấp" và "không có
   detection phù hợp" gần bằng nhau giữa các bộ lọc.
   - Cache dùng chung chỉ chứng minh **đầu vào giống nhau**. Nó không loại trừ tương tác giữa detection thiếu/sai với
     từng bộ lọc.
   - Chưa tách được đóng góp nhân quả của detection, dự đoán và assignment.

   Đi kèm:
   - Frag tăng khoảng 16–19 %;
   - MT giảm từ 74–79 xuống 46–48;
   - IDSW tăng khoảng 2,2 lần.
3. **Đặc điểm nội bộ đi kèm:**
   - ESS/N trước resample thấp (trung vị 0,19), resample gần như mọi update, còn khoảng 36 % tổ tiên.
   - Khi track được quan sát liên tục, đám hạt trước association (khoảng 0,095 h) xấp xỉ sai số thực: trung vị
     e/spread khoảng 0,97, 82 % mẫu có e ≤ 2·spread.
   - Khi track thiếu ≥ 3 frame, sai số tăng nhanh hơn độ trải: trung vị e/spread 1,5–2,3, chỉ 46–59 % mẫu có
     e ≤ 2·spread.
   - Độ trải vận tốc rất nhỏ (spread_v trung vị 0,017–0,029 h/bước).
4. **Chưa chứng minh nguyên nhân.** Các đặc điểm ở mục 3 là **tương quan đi kèm**. Lượt này không tách được vai trò
   của:
   - nhiễu quá trình (vị trí/vận tốc/heading) và nhiễu đo;
   - số hạt;
   - cách lấy ước lượng: `update_particles` tính mean/cov **trước** resample, còn `multi_predict` kế tiếp ước lượng lại
     từ đám hạt đã truyền;
   - quy tắc vh = 0 khi Lost;
   - FPS/độ phân giải.

   Theo GPT (review lần 2), bằng chứng hiện tại chưa đủ để chọn một can thiệp phân biệt được nguyên nhân, nên **R65 dừng
   tại đây và không đề xuất ablation**. Kết quả R64 giữ nguyên.

**Giới hạn:**
- Một detector, một ngưỡng, 4 camera của cùng một nút giao.
- Mẫu chỉ có trên các track nối được GT, nên **có thể** chệch có lợi cho PF; chiều chệch chưa được chứng minh.
- Mẫu số khác nhau: e/h dùng chiều cao **GT**; spread/h dùng chiều cao box **dự đoán**.
- Chi phí fused được tính lại offline theo công thức của tracker.
- Thời gian chạy trong log không phải benchmark (CPU, có observer/hash, chạy song song với việc chép file sang ổ rời).
