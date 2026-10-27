# Thực thi tracking test40 (21/10/2026) — cache → đo đoạn dày → full 36 run → chấm → hậu kiểm

**Trạng thái:** đã chạy xong. Codex nghiệm thu lần 1 (`D:/LABOTARY/automation/nghiem_thu_test40/lan_012`) revise (P2 hash
stdout, P3 diễn giải thời gian — đã sửa trong file này), **lần 2 (`lan_013`, 23/10 00:24) approve** với giới hạn review tĩnh.
Chưa commit. Số liệu dưới đây chỉ **mô tả**; không bootstrap, không kiểm định, không suy diễn nhân quả.

## 1. Căn cứ và cách chạy

- **Quyết định người dùng (chat 21/10 ~15:50):** sau R56, bỏ qua plan thực thi / planner, chạy cache + đo đoạn dày +
  full bằng bộ điều phối đã khóa **ngoài orchestrator**; **Codex nghiệm thu một lần ở cuối**. Plan R57 do planner đề
  xuất (`plan_0006`, chỉ cache một detector) đã TUCHOI theo quyết định này.
- Khóa thực thi: **R56_LOCK** `results/phase2/r55_tracking/chuan_bi/khoa_20261021_155510_440500_pid8768/`, `khoa.json`
  `ce85628e34bc8507f9577c51a06d381a2d8e5d0f35cdf099c18330429d35247b` (384 file; nối khóa R55 cha và khóa R54).
- Trình tự đúng `handoff/logs/review56/02_khoa_va_ban_giao.md` mục 5–6: cache đủ 40 video + xác nhận cho từng detector
  → đo UKF/PF500 trên MVI_40864 (toàn video, frame 1–1515) bằng `tracking --video` → tracking full 36 run → chấm 36 run →
  kiểm tái lập → hậu kiểm.
- Bộ chạy tuần tự **ngoài repo** `D:/LABOTARY/automation/thuc_thi_r56/chay_tuan_tu.py`, khởi chạy bằng WMI (độc lập
  phiên Claude), 127 lệnh `src/r55_tracking/dieu_phoi_test40.py`, mỗi lệnh foreground trong tiến trình bộ chạy, chờ exit;
  exit ≠ 0 thì dừng (không xảy ra). Lưu stdout/stderr từng lệnh: `…/thuc_thi_r56/stdout/NNN.txt`, nhật ký
  `nhat_ky.log`, trạng thái `trang_thai_chuoi.json`. **Đính chính (Codex nghiệm thu `lan_012`, P2):** `stdout_sha256`
  trong `trang_thai_chuoi.json` được băm trên nội dung xuống dòng LF, còn file ghi trên đĩa là CRLF (Windows), nên **không
  khớp byte trên đĩa** (0/127 khớp trực tiếp, 127/127 khớp sau khi chuẩn hóa CRLF → LF). Bằng chứng cũ giữ nguyên;
  manifest hash theo byte thực tế: `…/thuc_thi_r56/stdout_sha256_thuc_te.json` (SHA256
  `41124dbfecc5060df90bcd99e021d6f4a0a10ed91844c44a5c8141b8b5a374e2`, sinh bởi `lap_manifest_hash_thuc_te.py`, chỉ đọc).
  Bộ chạy về sau phải ghi bytes và băm đúng byte đã ghi.
- `--duyet nguoi_dung_chat_20261021_1550__chay_cache_do_doan_day_full_ngoai_orchestrator__codex_nghiem_thu_cuoi`
  (ghi vào `phien.json` và từng `ket_qua.json` của công việc).
- **Phiên thực thi:** `results/phase2/r55_tracking/thuc_thi/phien_20261021_161832_735800_pid11640/`
  (`cache/`, `xac_nhan_cache/`, `tracking/`, `cham/`, `nhat_ky/{lap_tracker,cham_lai,hau_kiem}/`).
- Thời gian: 16:18:29 → 22:01:14 (≈ 5,7 h); **127/127 lệnh exit 0**. GPU GTX 1650, cắm sạc, không ngủ.

## 2. Cache và xác nhận

| Detector | `cache` (40 video, 56.340 ảnh) | `xac-nhan-cache` |
|:--|--:|:--|
| yolov8n_coco | 1.004,8 s | exit 0 |
| yolov8n_mixed5 | 1.029,7 s | exit 0 |
| yolo11n_mixed5 | 1.403,8 s | exit 0 |

Mỗi detector có đúng một manifest cache công bố; mọi tracker dùng chung manifest đó (không suy luận lại).

## 3. Đo đoạn dày MVI_40864 (frame 1–1515, ≈ 29,5 dòng GT/ảnh; P1980cb21)

24 lệnh `tracking --run <run> --video MVI_40864` (UKF + PF500 s0/s1/s2 × c25/c10 × 3 detector), exit 0 cả 24.
Thời gian **mỗi lệnh** (gồm khởi động Python, kiểm khóa / hash cache mọi video, nạp thư viện, tracking 1.515 frame, ghi
MOT, kiểm module sau đơn vị): UKF 28,7–31,3 s; PF500 38,6–46,2 s. Chưa tách riêng thời gian tracker thuần (QJTF — chi phí
kiểm module mỗi đơn vị — chưa đo). Đơn vị MVI_40864 hoàn tất được đường full tái sử dụng.

## 4. Tracking full và chấm

- 36 run × 40 video = **1.440 đơn vị tracking**; mỗi run một lệnh. Thời gian trung bình mỗi run (40 video, gồm chi phí
  khung): CV 151 s, EKF 153 s, UKF 319 s, PF500 508 s; tổng tracking ≈ 12.884 s (3,58 h). Run đầu tiên
  (`r54-yolov8n_coco-c25-cv`) mất 265,1 s (`stdout/033.txt`), cao hơn các run CV sau; nguyên nhân **chưa đo** (giả
  thuyết chưa kiểm chứng: chi phí lần đầu như nạp đĩa / bộ nhớ đệm).
- 36 lần `cham` (TrackEval `MotChallenge2DBox`, HOTA/CLEAR/Identity, đủ 40 video, combined do evaluator tính): tổng
  2.121 s. Lớp vehicle gộp, ignored_region tô xám, timeline 56.340 ảnh gồm 173 ảnh D4 (chấm như frame không GT).

## 5. Kết quả (combined, 40 video)

Nguồn: `results/phase2/r56_tong_hop/phien_20261021_161832_735800_pid11640/tong_hop_36_run.csv` (sinh bởi
`src/r56_tong_hop/tong_hop_test40.py`, chỉ đọc `pedestrian_summary.txt` của lần chấm hoàn tất; `nguon.json` ghi sha256).

**HOTA**

| Detector | conf | CV | EKF | UKF | PF500 s0 / s1 / s2 (trung bình) |
|:--|:--|--:|--:|--:|:--|
| yolov8n_coco | c25 | 58,235 | 58,353 | 58,244 | 55,170 / 55,171 / 54,900 (55,080) |
| yolov8n_coco | c10 | 57,969 | 58,303 | 58,154 | 54,892 / 55,353 / 54,963 (55,069) |
| yolov8n_mixed5 | c25 | 61,643 | 61,711 | 61,630 | 58,422 / 58,558 / 58,656 (58,545) |
| yolov8n_mixed5 | c10 | 61,243 | 61,426 | 61,293 | 58,328 / 58,294 / 58,207 (58,276) |
| yolo11n_mixed5 | c25 | 62,063 | 62,239 | 62,181 | 58,957 / 58,754 / 58,723 (58,811) |
| yolo11n_mixed5 | c10 | 61,531 | 61,794 | 61,729 | 58,600 / 58,574 / 58,622 (58,599) |

**MOTA / IDF1 / IDSW** (c25)

| Detector | CV | EKF | UKF | PF500 s0 |
|:--|:--|:--|:--|:--|
| yolov8n_coco | 59,23 / 70,12 / 2.498 | 59,26 / 70,36 / 2.523 | 59,24 / 70,18 / 2.543 | 55,25 / 65,88 / 3.545 |
| yolov8n_mixed5 | 66,10 / 73,86 / 2.591 | 66,15 / 74,03 / 2.547 | 66,11 / 73,71 / 2.601 | 60,95 / 69,41 / 3.356 |
| yolo11n_mixed5 | 66,51 / 74,24 / 2.547 | 66,54 / 74,57 / 2.458 | 66,54 / 74,45 / 2.469 | 61,47 / 70,20 / 3.356 |

Bảng đủ 36 run (HOTA, DetA, AssA, LocA, MOTA, MOTP, IDF1, IDSW, Frag, FP, FN, IDs) trong CSV; PF500 trung bình / min /
max qua 3 seed trong `pf500_trung_binh.csv`.

**Mô tả (không phải kết luận nhân quả, không có khoảng tin cậy):**
- Hai detector mixed5 cho HOTA cao hơn baseline COCO khoảng 3–4 điểm ở mọi bộ lọc.
- CV / EKF / UKF chênh nhau ≤ ~0,35 HOTA trong cùng detector và điều kiện; EKF cao nhất ở cả 6 cặp (detector, conf)
  nhưng chênh lệch nhỏ, chưa có ước lượng biến thiên.
- PF500 (trung bình 3 seed) thấp hơn ba bộ lọc Kalman khoảng 2,9–3,4 HOTA ở mọi detector; IDSW cao hơn (~3.300–3.550
  so với ~2.450–2.600 ở c25); khác biệt giữa 3 seed PF ≤ ~0,46 HOTA.
- c10 so với c25 (cùng detector, bộ lọc, seed; 18 cặp): HOTA từ −0,53 đến +0,18 (đa số thấp hơn), MOTA thấp hơn
  0,65–1,45, Frag giảm 1.660–4.412, IDSW giảm 238–473.

## 6. Kiểm tái lập và hậu kiểm

- **`lap-tracker`** PF500 c25 s0/s1/s2 trên MVI_40864, 3 detector: **9/9 `trung_byte` = true** (MOT chạy lại từ cùng cache
  và seed trùng byte; `seed_thuc_dung` đúng; không module ngoài khóa). `nhat_ky/lap_tracker/*/ket_qua.json`.
- **`cham-lai`** một run c25 mỗi bộ lọc (CV, EKF, UKF, PF500 s0) × 3 detector: **12/12** `file_trung_byte`,
  `combined_trung`, `lan_goc_con_nguyen` = true. `nhat_ky/cham_lai/*/ket_qua.json`.
- **`hau-kiem`** exit 0: đếm cache 120, tracking 1.440, chấm 36; `chua_xong` [], `loi` [], không lần dở / marker hỏng,
  `du_36_run` true, không gọi detector. `nhat_ky/hau_kiem/*/ket_qua.json`.
- `kiem-khoa` đầu và cuối: exit 0 (R56_LOCK 384 file, khóa R54 368 file, khóa cha đúng hash).

## 7. Giới hạn và việc còn lại

- **Chưa Codex nghiệm thu.** Báo cáo này và bảng tổng hợp do phiên điều phối (Claude) viết, chưa có review độc lập.
- Không có ước lượng biến thiên (bootstrap theo video, nhiều seed cho bộ lọc tất định); thứ tự giữa CV/EKF/UKF chưa thể
  xem là khác biệt có ý nghĩa.
- Tái lập suy luận detector giữa tiến trình không kiểm (theo nguyên tắc cache bất biến); `lap-tracker` / `cham-lai` chỉ
  kiểm tracker và evaluator từ cùng đầu vào.
- 173 ảnh D4 giữ trong timeline, chấm như frame không GT; không diễn giải là âm tính đã xác nhận.
- Thời gian ở mục 3–4 là thời gian lệnh gồm chi phí khung (khởi động, kiểm khóa / hash, kiểm module), không phải FPS
  tracker; QJTF chưa đo.
- Ngoài phạm vi: SORT trên test40, BDD tracking, tối ưu Nano, phân tích nguyên nhân PF thấp hơn.
