# Kế hoạch đã chốt — từ vòng 27 (2026-09-29)

Nguồn: yêu cầu của người dùng sau khi Codex nghiệm thu `239b9c6` (**PASS WITH LIMITATIONS**, `handoff/CODEX_REVIEW_239b9c6.md`).
File này là phạm vi làm việc cho các vòng sau. Thay đổi phạm vi phải do người dùng quyết định.

## 0. TRẠNG THÁI HIỆN TẠI — tài liệu trạng thái chuẩn (cập nhật 2026-10-12)

> Mục này là **nguồn trạng thái duy nhất**. Brief, handoff, REVIEW*_CHI_TIET là lịch sử theo vòng; khi chúng
> mâu thuẫn với mục này thì mục này đúng, trừ khi có commit/bằng chứng mới hơn (khi đó sửa mục này).
> Mỗi vòng xong việc: cập nhật mục 0 trước, rồi mới viết brief/handoff. Lộ trình đọc: `AGENTS.md`.
> Chỉ mục tài liệu cũ và vùng lưu trữ: `handoff/LUU_TRU.md`.

**Mục tiêu hiện tại.** Chuẩn bị báo cáo ngày **17/10/2026**. Thứ tự ưu tiên do người dùng đặt (10/10/2026):
1. Làm vững **lý thuyết** KF / EKF / PF, **SIS / SIR** (lấy mẫu trọng số tuần tự, tái chọn mẫu) — gắn với mã
   `src/ekf_ctrv.py`, `src/ukf_ctrv.py`, `src/pf_ctrv.py` và số liệu đã đo.
2. **Audit dữ liệu** (UA-DETRAC, BDD100K): nguồn, split, nhãn, fingerprint.
3. **Kiểm chứng PF** (suy biến trọng số, số hạt, thiết kế nhiễu).
4. **Hoãn** tối ưu Jetson Nano (mốc 6) cho tới khi người dùng mở lại.

Kế hoạch chi tiết cho báo cáo 17/10: `handoff/KE_HOACH_BAO_CAO_20261017.md` (bản giao việc người dùng duyệt
2026-10-10, M0–M6; giữ nguyên, không sửa). Plan orchestrator chuyển đổi từ bản này: lượt R48 (mục "Việc đang làm");
tiếp nhận + bảng kế thừa (M0): `handoff/logs/review48/00_tiep_nhan_va_ke_thua.md`.

**Quyết định đã chốt** (nguyên văn ở file dẫn):
- Phạm vi + thứ tự mốc 1–6, quy tắc "hỏi trước", dev nội bộ, hai tập đánh giá cuối: các mục dưới đây của file này.
- Lớp detector 5 lớp `0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle` (`src/detector_classes.py`).
- Nhãn BDD MOT train: phương án C (snapshot HF đã ghim, vòng 34) — `handoff/REVIEW34_CHI_TIET.md`.
- Split `bdd5_train_v1` (train-fit 1.261 / dev 139 video, vòng 35) — `handoff/REVIEW35_CHI_TIET.md`.
- **D1–D5** giao thức đánh giá detector trên UA-DETRAC test: người dùng duyệt 2026-10-09 —
  `handoff/REVIEW46_CHI_TIET.md` mục 11.4; kế hoạch khóa trước suy luận:
  `handoff/logs/review47/03_ke_hoach_khoa_danh_gia_detrac_test.md` (sha256 `7028a291…`).
- Moc 2 tracking: tiền đăng ký + khóa cấu hình `configs/phase2/PREREG_moc2.md`, `configs/phase2/LOCK_moc2.md`.

**Kết quả đã xác minh** (đọc file dẫn, không chép lại số khác):
- Tracking UA-DETRAC test 40 video (mốc 2): EKF–CTRV hơn KF–CV có ý nghĩa ở `c10`, CI chứa 0 ở `c25`; PF–CTRV
  kém rõ rệt so với EKF; thứ tự ảnh hưởng Nhóm B `_fix1` — `CODEX_HANDOFF.md` mục 5 và mục 8, `PHASE2_REPORT.md`.
- PF: ESS/N ~1 % ở các lần cập nhật sau mất dấu dài; khoảng cách với EKF thu hẹp khi N 100 → 300 → 500 nhưng
  không hết — `CODEX_HANDOFF.md` mục 5 ý 7; `src/pf_diag_report.py`; `handoff/logs/pf_diag_report.log`.
- BDD transfer (COCO-video val): `results/phase2/bdd_transfer_cocovid_val/bao_cao/` (`bang_chinh.md`, `pf_seed.md`).
- Detector trên BDD val (vòng 43): `handoff/REVIEW43_CHI_TIET.md`; số liệu `results/phase2/detector_eval/bdd_val_cocovid/`.
- Fingerprint UA-DETRAC test (R47 bước 1: 40 video, 56.340 ảnh, D4 loại 173): `handoff/logs/review47/01_fingerprint_detrac_test.md`.
- Detector trên UA-DETRAC test (R47 bước 4): mAP50 (car, bus) baseline `yolov8n_coco` 0,870 > `yolov8n_ft` 0,586 /
  `yolo11n_ft` 0,564; Codex nghiệm thu **approve** sau một vòng sửa diễn giải —
  `handoff/logs/review47/04_ket_qua_danh_gia_detrac_test.md`; số liệu `results/phase2/detector_eval/detrac_test/`
  (commit `482baaf`, không gồm 3 NPZ dự đoán — chỉ trên máy, sha256 ở mục 7 báo cáo 04). Tài liệu R46 (gồm mục 11.4
  D1–D5): commit `43cd0de`.
- **R48A / M1** (2026-10-10; Claude tự kiểm; Codex cổng A: vòng 1 **revise** F1–F4, vòng 2 **approve** —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_002.json`; chưa commit): lý thuyết KF–CV (XYAH 8-D),
  EKF–CTRV, PF–CTRV, SIS / SIR, ESS, log-sum-exp có phương trình, đơn vị bước, góc, ω ≈ 0 và trạng thái nguồn —
  `handoff/logs/review48/01_ly_thuyet_kf_ekf_pf.md` (ByteTrack: đã đối chiếu toàn văn tr. 2–7; Arulampalam 2002: chỉ
  metadata, mọi ý ghi "chưa đối chiếu toàn văn"). Bảng khái niệm → `file:hàm:dòng` tại `eaf2e87`, giả thuyết H1–H7 —
  `handoff/logs/review48/02_doi_chieu_code_pf.md`. **Sở hữu trạng thái**: mỗi track PF có mảng particle, log-trọng số và
  rng riêng; chỉ `PFCTRV` (cấu hình + bộ đếm chẩn đoán) dùng chung — đọc mã + `src/kc48/kiem_so_huu_hat.py` exit 0
  (`results/phase2/kc48_pf_code/so_huu_hat.json`); "cạnh tranh hạt toàn cục" không áp dụng cho cấu trúc mã hiện tại.
  `src/test_pf_ctrv.py` 26/26 PASS. Không sửa mã tracker. Vòng sửa 1 (chỉ tài liệu, theo
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_001.json`): F1 σ_θ, σ_ω là độ lệch
  chuẩn gia số mỗi predict, đơn vị [rad], [rad/bước] (bảng đơn vị từng thành phần Q ở `01` mục 3; chú thích mã
  `src/config.py:304-305` ghi [rad/frame], [rad/frame²], chưa sửa); F2 tách mô hình lý tưởng khỏi 6 heuristic triển
  khai, phép chặn h ≥ 10⁻³ dồn khối xác suất vào biên chứ không phải Gauss cắt cụt; F3 `pf_estimate: map` là particle
  trọng số lớn nhất, không phải MAP (không cấu hình nào tại HEAD dùng `map`); F4 ghi nhận lệch phạm vi ngay dưới.
  Danh sách chỗ sửa: `01` mục 10, `02` mục 7.
- **Lệch phạm vi R48A** (F4): lệnh `src/test_pf_ctrv.py` (test 11 sẵn có) đã xóa và ghi lại
  `data/interim/det_cache/fixture_pf_seed_test/`, ghi `data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1}/`, ngoài
  `write_paths` của R48A. Log thực thi:
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\calls\0001_coder\stdout.log` dòng 162 (lệnh), 165
  (kết quả). Không xác minh được trạng thái trước lệnh (đường dẫn bị git bỏ qua, không có hash trước); không xóa, khôi
  phục hay chạy lại. Chi tiết: `02` mục 6.1.
- **R48B / M2** (audit 2026-10-10; Claude tự kiểm. Codex cổng B vòng 1 **revise** F1–F4 —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_003.json`. Sửa hậu audit 2026-10-11,
  Claude tự kiểm. Vòng 2 **revise** —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_005.json`: F1–F4 đã đóng, thêm F5
  (ghi ngoài phạm vi, chỉ yêu cầu ghi nhận; xem "Lệch phạm vi R48B" ngay dưới). Đã ghi nhận F5; vòng 3 **approve** —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_007.json`; chưa commit). Báo cáo
  `handoff/logs/review48/03_audit_du_lieu.md` (sửa hậu audit ở mục 13, F5 ở mục 13.3), số liệu
  `results/phase2/kc48_audit/*.csv`, script chỉ đọc `src/kc48/audit_du_lieu.py`. Tiêu chí
  (quỹ đạo, mẫu trực quan, gần trùng, xe nhỏ cạnh xe lớn) công bố trước khi tính:
  `results/phase2/kc48_audit/tieu_chi_cong_bo.json`, sha256 `d7543fff…`, không đổi khi sửa.
  - Mốc slide lần 2 khớp khi đếm lại từ 2–3 nguồn: BDD train-fit 1.261 / 50.487, dev nội bộ 139 / 5.559, BDD val
    200 / 39.973 (`moc_doi_chieu.csv`). Tập chỉ có vài chục video: UA-DETRAC dev 12, test 40, train 60.
  - Toàn bộ 458.183 ảnh đọc được: sha256, header, SOI / EOI 0 lỗi. Giải mã 6.783 ảnh mẫu: 0 lỗi. 320/320 mẫu YOLO khớp
    nhãn gốc. Với ảnh tô xám, mẫu này chỉ kiểm siêu dữ liệu; **chưa kiểm pixel**. 1.400/1.400 file nhãn BDD train khớp
    manifest nhập vòng 34.
  - Trùng sha256 chỉ trong cùng video: 3 nhóm frame đứng hình, gồm 2 nhóm ở BDD train-fit và 1 nhóm ở BDD val. Có 3 cặp
    gần trùng, đều xác nhận là cùng cảnh; 1 cặp xuyên train-fit / dev BDD. 70/139 video dev có chung tiền tố tên với
    train-fit. Giữa BDD train / val và DETRAC train / test **chưa phát hiện** chồng lấn trong phạm vi đã kiểm (sha256
    toàn bộ ảnh, dHash 4 frame mỗi video, tiền tố tên). Video là đơn vị chia split, không chắc là đơn vị độc lập thống kê.
  - **F1, lỗi mã quỹ đạo (đã sửa)**: trước đây track `nghi_nhieu` (max Δψ > 150°) vẫn nằm trong tập hợp lệ, và p90 phân
    loại độc lập. Nay track này bị loại khỏi tập hợp lệ của cả max lẫn p90; đã chạy lại 4 phần `nhan` và `tong_hop`.
    Kiểm bằng `src/kc48/kiem_loc_quy_dao.py`: trước sửa exit 1 (4.687 track vi phạm); sau sửa exit 0, 21/21
    (`results/phase2/kc48_audit/kiem_loc_quy_dao.json`). So hash trước / sau: chỉ đổi đầu ra quỹ đạo, nhật ký chạy và
    một ô ví dụ của `chat_luong.csv` (`03` mục 13). BDD sau sửa: 37,5–42,0 % track hợp lệ đổi hướng trên ảnh ≥ 30° theo
    max, 28,2–32,2 % theo p90; bản trước ghi khoảng 30 %. DETRAC đổi ≤ 15 track mỗi tập.
  - Augmentation thực chạy: mosaic 1,0 (tắt 2 epoch cuối), translate 0,1, scale 0,5, HSV, fliplr 0,5. Albumentations
    không chạy vì không cài gói. `auto_augment` / `erasing` chỉ dùng cho classify.
  - UA-DETRAC dev12: 32 track cua (8 theo p90), 14 track bus, 88 sự kiện xe nhỏ cạnh xe lớn. Dev12 không có track
    `nghi_nhieu`, nên sửa F1 không đổi các số này.
- **Lệch phạm vi R48B** (F5): lời gọi sửa hậu audit đã ghi hai file memory của Claude Code, ngoài repo và ngoài
  `write_paths` của R48B: tạo `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\ket-luan-tu-nguon-day-du.md`
  (`Write`) và chèn 1 dòng trỏ vào `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\MEMORY.md` (`Edit`).
  Log thực thi: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\calls\0010_coder\stdout.log` dòng
  842–843 (tạo), 844–845 (sửa). Không xác minh độc lập được trạng thái trước sửa: vùng này ngoài git, orchestrator
  không chụp hash, chỉ có tự khai của công cụ ở dòng 843, 845. Không đổi file nào trong repo. Không sửa, xóa hay khôi
  phục memory. Chi tiết: `03` mục 13.3.
- **R48C / M3 + tính đúng M4** (2026-10-11; Claude tự kiểm; chưa commit). Lượt cũ: Codex trả **BLOCKED** vì test 11
  chưa chạy. Người dùng quyết định ngày 11/10: cho test 11 ghi fixture và gộp việc truy nguyên lệch EKF vào R48C. Bản
  sửa theo quyết định này được Codex cổng C **approve**, không có issue —
  `handoff/logs/review48/nghiem_thu_r48cde/0002_review_answer.json` (bản sao nguyên văn, nguồn + sha256 ở `MANIFEST.md`
  cùng thư mục). Báo cáo `handoff/logs/review48/05_dung_dan_pf_va_smoke.md` và `05b_truy_nguyen_lech_ekf.md`.
  Giới hạn Codex ghi trong biên bản: review tĩnh, không chạy lại test hay smoke (kết quả thực thi kiểm từ log và
  artifact); ngoài fixture, snapshot chỉ kiểm kích thước và `mtime_ns`, không hash toàn bộ nội dung; đối chiếu EKF chỉ
  bao điều kiện A trên 2/12 video, tái lập seed qua runner chỉ kiểm fixture tổng hợp 40 frame — không suy rộng sang bộ
  lọc, video hay chỉ số khác.
  - Giao thức kiểm chứng PF `handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md` viết **trước** mọi so sánh bộ lọc.
    Khóa bằng sha256 `7063c532…`: chụp 04:29:39, chụp lại 06:51:50 trước smoke và 15:12:53, cả ba trùng nhau
    (`results/phase2/kc48_pf_code/giao_thuc_sha256.json`). Phụ lục P1 rút ngoại lệ "trừ test 11" ở mục 10. Sau P1,
    file có sha256 `479497fc…`; lần chụp 4 kiểm 22.510 byte đầu vẫn là `7063c532…`, tức mục 1–12 không đổi.
  - **Test 11 đã chạy đủ**: `python src/test_pf_ctrv.py` exit 0, 26/26, gồm "chạy lại seed 0: file txt
    bit-identical". `results/phase2/kc48_pf_code/fixture_{truoc,sau}.json` cho thấy chỉ 3 thư mục fixture được phép
    đổi; 24.164 file khác trong `data/interim` + `data/processed`: đổi 0, thêm 0, mất 0. `data/synth.txt` của cả hai
    seed trùng sha256 với bản có từ trước.
  - **EKF tái tạo trùng byte mốc 2** (`05b`; `results/phase2/kc48_ekf_tai_lap/so_sanh_dong.json`): file tracking EKF
    điều kiện A của smoke (`eaf2e87`) và của run mốc 2 (`1c7ce8c`) trùng từng byte trên `MVI_20052` và `MVI_20062`;
    0 frame lệch. Nhận định "khác nhau" trước đây là **sai**. Nó đến từ quy ước hash: commit `935a445` đổi
    `run_fingerprint.sha256_file` sang hash `.txt` sau chuẩn hóa CRLF → LF, nên hai `eval_meta.json` khác đời ghi hash
    khác nhau. Mới kiểm 2/12 video, chỉ EKF, điều kiện A.
  - Giao thức chốt: mô phỏng biết liên kết gồm 8 kịch bản, trong đó có 4 kịch bản lệch mô hình; pipeline dev12; K1
    proposal tối ưu (mới, tùy chọn); K2 roughening (đã có trong mã, không phải đóng góp mới); không đổi association;
    bootstrap theo video / seed nhiễu.
  - Cache dev trùng khít mốc 2 (`results/phase2/kc48_pf_code/cache_dev.json`). Bộ mô phỏng
    `src/kc48/mo_phong_quy_dao.py --tu-kiem` đạt 10/10.
  - Test, tất cả exit 0, chạy lại ngày 11/10 sau 15:12: `src/kc48/test_pf_dung_dan.py` 29/29; `src/test_pf_ctrv.py`
    đủ 11 test 26/26; EKF 14/14, UKF 12/12, lifecycle 20/20, init 17/17.
  - **Không sửa mã tracker**: không có test hỏng do mã. Lần chạy đầu của test mới hỏng 2 phép kiểm, cả hai do thiết kế
    test (`05` mục 3).
  - Smoke `kc48_pf_smoke` (2 video dev; ekf, pf100 × 2 seed, pf300, pf500) qua wrapper `src/kc48/chay_thi_nghiem_kc48.py`
    (đổi gốc `PROCESSED_DIR` vào `results/phase2/kc48_pf_smoke/_processed/`): track và eval đều exit 0.
  - Thời gian: ms / frame `ekf` 1,62, `pf100` 3,48, `pf300` 4,78, `pf500` 5,90 (`results/phase2/kc48_pf_smoke/thoi_gian.json`).
    Kịch bản dự toán cho P1–P3 của R48D: 37–55 phút, trong ngân sách 120 phút. Chỉ số của smoke không dùng cho kết luận.
- **R48D / M4 + M5 (thiết kế) + M6** (2026-10-11; Claude tự kiểm; cổng Codex D vòng 1 **revise** F1–F3 —
  `handoff/logs/review48/nghiem_thu_r48cde/0006_review_answer.json`; vòng 2 **approve** —
  `handoff/logs/review48/nghiem_thu_r48cde/0009_review_answer.json`; chưa commit). Giới hạn Codex ghi ở vòng 2: review
  tĩnh, không chạy lại test, tracking hay mô phỏng; bằng chứng không ghi đè kết quả cũ vẫn dựa trên mtime, hash sao lưu
  chỉ xác nhận các bản sao R48D đã liệt kê; mẫu số ESS / resampling khi bám là `n_upd_bam` − `n_reinit_bam`, thống kê
  tổ tiên chỉ dùng các lần systematic resampling (câu tóm tắt trong `07` mục 11 đã được R49A làm rõ); test runner seed 11
  chưa chạy lại sau K1, đề xuất này vẫn chờ người dùng. Issue nhỏ không chặn: đường dẫn JSON so sánh dev ở `07` mục 10 thiếu
  `phan_tich/` — R49A đã sửa thành `results/phase2/kc48_pf_dev/phan_tich/so_sanh_truoc_sau_vong_sua.json`. Tổng hợp
  `handoff/logs/review48/07_tong_hop_bao_cao_17_10.md`, đề cương slide `08_de_cuong_slide.md`, thiết kế M5
  `06_thiet_ke_augmentation.md` (chỉ thiết kế, không huấn luyện). Giao thức `04` chụp lại trước R48D lúc 15:31:12:
  22.510 byte đầu vẫn `7063c532…` (`results/phase2/kc48_mo_phong/giao_thuc_sha256_r48d.jsonl`); lệch ghi ở phụ lục P2.
  - Tùy chọn mới: K1 `pf_optimal_proposal` (proposal tối ưu, mặc định tắt) trong `src/pf_ctrv.py`, `src/tracker_ctrv.py`;
    K2 dùng khóa `pf_roughening` có sẵn (không phải đóng góp mới). `src/kc48/test_k1_k2.py` 16/16, exit 0: mặc định
    bit-identical với mã `eaf2e87`. File tracking `ekf`, `pf100-s0/s1` dev trùng sha256 với smoke R48C trên 2 video
    (`results/phase2/kc48_pf_dev/phan_tich/mac_dinh_vs_smoke.json`). Test cũ exit 0; test 11 **không chạy lại** sau K1
    (ghi ngoài `write_paths` của R48D).
  - Mô phỏng (`results/phase2/kc48_mo_phong/`, 30 seed nhiễu ghép cặp × 3 seed PF): `pf100` kém `ekf` nhất quán (F, e_r),
    cả nhóm lệch mô hình. K1 giảm nhất quán F, e_r và triệu chứng nghèo hạt khi đang bám, nhưng không sửa sụp ESS sau mất
    đo dài. K2 chỉ giảm e_r. Không biến thể PF nào đạt EKF.
  - Pipeline dev12 (26 run, `results/phase2/kc48_pf_dev/`, `results/trackeval/DETRAC-all/p2-kc48_pf_dev-*`; bootstrap
    theo video): K1, K2, K1+K2 cải thiện `pf100` có CI loại 0. Mọi biến thể PF, kể cả `pf500-k1`, vẫn kém EKF (HOTA
    −0,038 [−0,065; −0,021]). EKF đã chỉnh không khác EKF mặc định. Tầng xe nhỏ cạnh xe lớn (42 sự kiện / 7 video):
    K1 không phân biệt được với EKF (CI chạm 0), cỡ mẫu rất nhỏ. Không chạy trên tập test / BDD val.
  - Không ghi đè kết quả cũ: `src/kc48/kiem_khong_ghi_de.py` exit 0 (bằng chứng mtime,
    `results/phase2/kc48_pf_dev/kiem_khong_ghi_de.json`).
  - **Vòng sửa sau cổng D vòng 1** (Codex **revise** F1–F3,
    `D:\LABOTARY\automation\runs\bytetrack_r48_tiep_effort_high_20261011_1507\reports\review_003.json`, bản sao
    `nghiem_thu_r48cde/0006_review_answer.json`; Claude tự kiểm 2026-10-11; cổng D vòng 2 **approve**, `0009`: F1, F2,
    F3 đã đóng). Không chạy lại tracking. Bản trước sửa sao nguyên byte ở
    `results/phase2/{kc48_pf_dev,kc48_mo_phong}/**/_ban_cu_truoc_vong_sua/` (có `manifest.json` sha256). Chi tiết `07` mục 11,
    phụ lục `04` P3.
    - F1: phân tầng ghép bằng `occlusion_eval.match_frame` (ràng buộc IoU ≥ 0,5 trước Hungarian). Ca MVI_20052 / frame 445:
      GT 38 → 290 (trước: 350). 167/30.810 dòng track-run đổi; tầng lệch tối đa 0,0005; không tầng nào đổi trạng thái CI.
    - F2: tái khởi tạo tách khỏi ESS và systematic resampling; n_deg theo từng quỹ đạo, có mẫu số. Chạy lại có chọn lọc
      12.235 quỹ đạo PF có tái khởi tạo ngoài t_r + 1.838 quỹ đạo đối chứng, cùng cấu hình/seed (169 s, 6 tiến trình).
      Đầu ra bộ lọc trùng khít bản cũ (`results/phase2/kc48_mo_phong/chan_doan_v2/so_sanh_ban_cu.json`, 80/80 đơn vị).
      ESS/N tại t_r của `pf100` ô `cua_deu` L 20 h0 24: 0,011 trên 47/90 quỹ đạo hợp lệ (43 tái khởi tạo); trước ghi 0,006.
      C3 (K2) ESS/N tại t_r: 10 ô CI > 0 trước sửa → 0 ô sau sửa. Các kết luận C1–C3 về F, e_r không đổi.
    - F3: COMBINED tính riêng từng seed rồi trung bình, trong mỗi mẫu bootstrap (`src/kc48/boot_seed.py`). Bảng chính lệch
      tối đa: HOTA 2,8·10⁻⁵, AssA 8,7·10⁻⁵. Không đối chiếu nào đổi trạng thái CI.
    - Test: `src/kc48/test_sua_r48d.py` 29/29 exit 0; `test_k1_k2.py` 16/16, `test_pf_dung_dan.py` 29/29,
      `kiem_khong_ghi_de.py` exit 0.
- **R48E / kiểm pixel ảnh tô xám** (2026-10-11; mốc người dùng duyệt qua email 11/10; Claude tự kiểm; chưa commit).
  Nguồn ghi mục này: biên bản Codex — vòng 1 **revise** F1 (diễn giải vượt bằng chứng về nguyên nhân 3 ảnh không đạt) và
  F2 (mô tả quá rộng cơ chế khóa tiêu chí) — `handoff/logs/review48/nghiem_thu_r48cde/0013_review_answer.json`; vòng 2
  **approve**, F1 và F2 đã đóng, không issue mới — `handoff/logs/review48/nghiem_thu_r48cde/0015_review_answer.json`.
  Báo cáo `handoff/logs/review48/03_audit_du_lieu.md` mục 14; số liệu `results/phase2/kc48_audit/pixel_to_xam_*`
  (`pixel_to_xam_tung_anh.csv`, `pixel_to_xam_tom_tat.json`, `pixel_to_xam_chan_doan_K.{csv,json}`,
  `pixel_to_xam_nhat_ky.jsonl`), tiêu chí công bố trước `results/phase2/kc48_audit/tieu_chi_pixel_to_xam.json` (sha256
  `4f62d165…`), script `src/kc48/kiem_pixel_to_xam.py`.
  - Mẫu: đúng 160 ảnh `to_xam` trong mẫu 320 ảnh của `03` mục 6.3 (4 tầng × 40 ảnh). Kết quả theo ảnh: **131 đạt / 26
    chưa kết luận / 3 không đạt**; theo tiêu chí T1 134 / 0 / 26, T2 160 / 0 / 0, T3 83 / 3 / 74 (đạt / không đạt / không
    áp dụng) — Codex tự đếm lại CSV khớp JSON (0013, 0015).
  - 3 ảnh không đạt (đều ở T3, MAE > 2,0): sai số **phù hợp với** mất mát do nén lại JPEG; **chưa chứng minh** nguyên nhân,
    chưa loại trừ lỗi xử lý vùng. Ngưỡng không sửa sau khi xem; 3 ảnh vẫn ghi `khong_dat`.
  - Cơ chế khóa tiêu chí: `--phan tinh` chỉ tự kiểm chuỗi dung sai xám và bán kính lõi so với file tiêu chí; ngưỡng
    T1 / T2 / T3 là hằng số trong mã, không đối chiếu tự động (khớp file tiêu chí khi đối chiếu thủ công).
  - Giới hạn Codex ghi: review tĩnh, không chạy lại phép đo, hậu kiểm JPEG hay test; không có snapshot toàn bộ `data/`
    trước–sau để chứng minh độc lập dữ liệu không đổi (chỉ kiểm đường ghi trong mã và lệnh đã lưu); kết luận đo **chỉ áp
    dụng cho mẫu 160 ảnh**, không suy ra toàn bộ 18.600 ảnh tô xám.
  - Đề xuất **P9a6c2078** ("Ghi R48E vào mục 0 PLAN_DA_CHOT", tùy chọn, hỏi lúc 2026-10-11 21:36:28 +07): theo
    `handoff/logs/review48/nghiem_thu_r48cde/P9a6c2078_ho_so.json`, `quyet_dinh_da_ghi` rỗng, tức **chưa có quyết định
    của người dùng — chưa duyệt**. Mục R48E này được ghi trong bước đồng bộ R49A (người dùng duyệt 12/10/2026) với nguồn
    là biên bản approve `0015`; lần duyệt R49A **không** phải là duyệt thực hiện P9a6c2078, và trạng thái P9a6c2078 giữ
    nguyên theo hồ sơ.
- **Kiểm chẩn đoán PF mốc 2: tái khởi tạo có trộn vào ESS không** (bước P9a8ee8bf, người dùng duyệt qua email; 2026-10-11;
  chỉ đọc, không sửa kết quả cũ; chưa commit) — `handoff/logs/review48/05c_kiem_pf_diag_moc2.md`; số liệu
  `results/phase2/kc48_pf_code/ess_moc2_tach_tai_khoi_tao/` (`ket_qua.json`, `theo_{run,tracker,n_lost}{,_test}.csv`).
  Về công thức, `src/pf_diag_report.py` **có** trộn dòng suy biến vào ESS / tỉ lệ resample. Trên số liệu mốc 2 thì
  **không** bị trộn: dev 18 run, **0/1.664.036** update là tái khởi tạo; tính lại chỉ trên update hợp lệ trùng số cũ,
  lệch tối đa **1,1·10⁻¹⁶** (sai số làm tròn). `handoff/logs/pf_diag_report.log` và các trích dẫn ESS ở `01`, `02`, `04`
  không phải sửa. Test UA-DETRAC (6 run, 0/3.400.029 tái khởi tạo) là số mới, chỉ để báo cáo, không dùng chọn tham số.

**Việc đang làm:** lượt orchestrator **R48** thực hiện bản giao việc 17/10 theo 4 mốc, mỗi mốc một cổng Codex:
`R48A_LY_THUYET_DOI_CHIEU_PF` (M1) → `R48B_AUDIT_DU_LIEU` (M2) → `R48C_GIAO_THUC_VA_DUNG_DAN_PF` (M3 + test tính
đúng) → `R48D_THI_NGHIEM_VA_TONG_HOP` (M4, M5 chỉ thiết kế, M6), thêm mốc `R48E_KIEM_PIXEL_TO_XAM` (người dùng duyệt qua
email 11/10). Cả năm mốc đã được Codex approve (xem tiến độ dưới). Báo cáo của lượt: `handoff/logs/review48/`.
Hồ sơ lượt: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_*` (R48A, R48B) và
`D:\LABOTARY\automation\runs\bytetrack_r48_tiep_effort_high_20261011_1507` (R48C, R48D, R48E). Biên bản Codex của R48C–E
sao nguyên văn ở `handoff/logs/review48/nghiem_thu_r48cde/` (`MANIFEST.md`: nguồn + sha256). Tiến độ:
- **R48A** approve ở vòng 2.
- **R48B** approve ở vòng 3 (`review_007.json`).
- **R48C** approve (`0002_review_answer.json`), sau BLOCKED ở lượt cũ (test 11) và quyết định người dùng 11/10.
- **R48D** approve ở vòng 2 (`0009_review_answer.json`), sau revise F1–F3 (`0006_review_answer.json`).
- **R48E** approve ở vòng 2 (`0015_review_answer.json`), sau revise F1–F2 (`0013_review_answer.json`).
- **R49A** (2026-10-12, người dùng duyệt 12/10): chỉ đồng bộ hồ sơ — mục 0 này và đường dẫn / mẫu số trong
  `handoff/logs/review48/07_tong_hop_bao_cao_17_10.md`; không chạy lại gì, không đổi số liệu, không mở mốc mới.
  Chờ cổng nghiệm thu Codex.

**Chưa chốt / ngoài R48**
- Kết quả detector BDD val vòng 43 (`results/phase2/detector_eval/bdd_val_cocovid/`) vẫn chưa commit.
- Lượt R46 và R47 đã dừng; repo khác mốc lưu của cả hai — không `start`/`recover` lại, hồ sơ giữ nguyên.

**Bước tiếp theo:** cổng nghiệm thu Codex cho R49A; xong thì dừng. Không có mốc nào khác đang được duyệt. Việc tồn đọng,
mỗi việc cần người dùng quyết định trước khi làm:
- Đề xuất P9a6c2078 (ghi R48E vào mục 0): chưa có quyết định trong hồ sơ (xem mục R48E ở trên).
- Chạy lại test 11 (runner seed) trên mã có K1: đề xuất chờ người dùng (Codex `0009`); test này ghi fixture ngoài
  `write_paths` của R48D nên chưa chạy.
- Commit các sản phẩm R48A–R48E và R49A: chưa commit; người dùng quyết định.
- Giữ hay gỡ hai thay đổi memory ngoài phạm vi của R48B (F5): người dùng quyết định.
- Kiểm cách evaluator xử lý ảnh không có `<frame>`: đề xuất riêng đang chờ người dùng (`04` mục 2).
- Chạy K1 trên tập đánh giá cuối, thực thi thiết kế M5 `06_thiet_ke_augmentation.md`, split BDD mới, kiểm pixel ngoài
  mẫu 160 ảnh: chưa chạy, là việc mới.

Mọi so sánh chỉ dùng các run tạo tại cùng commit, theo giao thức. So dấu vân tay tracking qua mốc `935a445` phải tính
lại theo cùng quy ước trên chính file, không so `eval_meta.json` khác đời (`05b` mục 6). Không tự mở việc ngoài bản giao
việc.

**Điểm chưa rõ**
- Nguyên nhân recall bus thấp (~0,34) của hai checkpoint fine-tune trên UA-DETRAC test; mức đóng góp của nhầm lớp
  bus → truck / car chưa xác định (`04_ket_qua_danh_gia_detrac_test.md` mục 4).
- Khoảng cách PF – EKF: R48D kiểm được một phần H4–H7 trên mô phỏng (`07` mục 5.3). H2 (association) và H3 (likelihood
  trên sai số detector thật) chưa kiểm. Vì sao DetA của PF thấp hơn EKF trên dev12 chưa rõ. Q riêng cho PF chưa chạy.
- Test 11 (runner seed) chưa chạy lại trên mã có K1. Bằng chứng thay thế: `test_k1_k2.py` test 1 và
  `mac_dinh_vs_smoke.json` (`04` P2).
- Toàn văn Arulampalam et al. 2002 không có trên máy: các ý lấy từ bài này ghi "chưa đối chiếu toàn văn".
- Tái tạo mốc 2 tại HEAD mới kiểm cho EKF điều kiện A trên 2/12 video dev (trùng byte). 10 video còn lại, CV / UKF,
  các điều kiện khác và số liệu TrackEval của mốc 2: chưa kiểm (`05b` mục 6).
- Phép chạy lại seed qua runner mới kiểm trên fixture tổng hợp 40 frame (`run_tracking.py`), chưa kiểm qua
  `run_experiment.py` trên video dev.
- Bản fixture có trước có mtime khoảng 2026-10-10 18:29–18:30 +07, khớp khung thời gian lệnh `src/test_pf_ctrv.py` của
  R48A (lệch phạm vi F4 ở trên). Đây là giả thuyết, chưa đối chiếu với log (`05` mục 4).
- Chú thích đơn vị trong mã (`src/config.py:304-305`, `src/time_units.py:33-34`, `src/ekf_ctrv.py:299-300`) ghi
  σ_θ [rad/bước], σ_ω [rad/bước²], khác cách đọc theo gia số trong `01` mục 3. Giá trị số không đổi; chưa sửa vì ngoài
  phạm vi R48A.
- Nguồn gốc cấu hình 53/56 sequence cache `legacy` và 113 thư mục run `LEGACY` không xác minh được bằng hash.
- Nhãn BDD MOT train chưa xác thực bằng archive gốc (chỉ snapshot bên thứ ba đã ghim). R48B xác nhận nhãn trên máy
  đúng là bản đã nhập và đã khóa; không xác nhận trùng nhãn gốc. COCO-video val có 17 track đổi lớp, nguyên nhân chưa
  kiểm.
- Dev nội bộ BDD không độc lập theo lần ghi hình với train-fit (70/139 video dev chung tiền tố tên; 1 cặp cùng cảnh).
  Mức lạc quan của chỉ số dev chưa đo. Split mới cần người dùng quyết định (`03_audit_du_lieu.md` mục 10).
- Tính độc lập giữa các video chưa xác định được: ngay trong một tập đã có video cùng cảnh (DETRAC test `MVI_40773` /
  `MVI_40775`). Kiểm chồng lấn giữa tập huấn luyện và tập đánh giá chỉ bao 4 frame mẫu mỗi video (`03` mục 6.4).
- Pixel ảnh tô xám của bản xuất YOLO: R48E mới kiểm trên mẫu 160 ảnh (131 đạt / 26 chưa kết luận / 3 không đạt, `03`
  mục 14). Toàn bộ 18.600 ảnh tô xám chưa kiểm. Nguyên nhân 3 ảnh không đạt T3 chưa chứng minh (phù hợp với nén lại JPEG,
  chưa loại trừ lỗi xử lý vùng); bản cv2 lúc export chưa xác minh (`03` mục 14.4).
- Evaluator tracking xử lý 572 ảnh dev12 không có `<frame>` (và 173 ảnh test, 1.706 ảnh train60) thế nào: chưa kiểm.
  Giao thức 04 mục 2 ghi đây là hạn chế: mọi biến thể chấm trên cùng GT, nhưng mức ảnh hưởng lên hiệu số chưa biết.
- Ranh giới car / truck / van mơ hồ trên mẫu trực quan, chưa đo tác động lên nhầm lớp.
- Giữ hay gỡ hai thay đổi memory ngoài phạm vi của R48B (F5, `03` mục 13.3): người dùng quyết định; agent không tự hoàn
  tác.

## Thứ tự các mốc

| # | Mốc | Trạng thái (vòng 27) |
|:--|:--|:--|
| 1 | **Giao diện Nano** (HDMI + chuột + bàn phím, launcher, cửa sổ xem trực tiếp) | ✅ **Đã nghiệm thu với giới hạn** (review `239b9c6`): thao tác HDMI/zenity được tự động hóa, reviewer không bấm tay; nhánh SIGKILL chưa kiểm với zenity treo thật trên Nano; không camera; không vận hành nhiều giờ. Gói đang dùng: `~/nano_run/v16` |
| 2 | **Chuẩn bị BDD100K train** và **fine-tune YOLOv8n, YOLO11n** (5 lớp). **YOLOv8n COCO hiện tại** (`models/yolov8n.pt`, sha256 `f59b3d83…`) giữ làm **baseline** | 🔄 Vòng 27: mã chuẩn bị dữ liệu / huấn luyện / đánh giá + smoke đường chạy trên fixture. Vòng 28: sửa R27-F1/F2/F3 của review `4937c85` (hợp đồng dataset dùng chung, khóa nội dung, ghép hòa IoU như COCO); Codex `07615ba` xác nhận 3 finding đã hết nhưng mở R28-F1 (cache ảnh `.npy`). Vòng 29: sửa R28-F1 — Codex `bcb92f8` **PASS WITH LIMITATIONS**, đóng R28-F1; base review → `bcb92f8`. Vòng 30: kiểm lại dữ liệu — BDD100K MOT train **vẫn chưa có**, nên khóa split / xuất / smoke thật **bị chặn**. Vòng 31: **ảnh** MOT train tải + xác minh md5 + giải nén đủ 7/7 phần (1.400 video, 278.079 ảnh); **nhãn** MOT 2020 `box_track_20/train` **không lấy được** từ nguồn chính thức nào (ETH NXDOMAIN, nút Berkeley bị gỡ / 404); `kiem-nguon` exit 2, chỉ còn thiếu nhãn. Vòng 32–33: ứng viên nhãn Scalabel từ zip đóng gói lại trên HF (`jaffe03195/bdd100k_sot` @ `694197099692…`) được cách ly và kiểm chứng. Box/ID của native val khớp COCO val của dự án, nhưng lớp/ignore khác (183/50, UNKNOWN). Codex `4e71d32` **PASS WITH LIMITATIONS**; base review → `4e71d32`. Vòng 34: người dùng **chọn phương án C** — dùng có điều kiện snapshot bên thứ ba đã ghim revision và hash. Đã nhập nguyên byte 1.400 JSON train; hash 1.400/1.400 khớp; `kiem-nguon` **exit 0** (1.400 video, 278.079 frame). Chưa xác thực bằng archive gốc. Codex `ccd801f` **PASS WITH LIMITATIONS**; base review → `ccd801f`. Vòng 35: **khóa split** `bdd5_train_v1` (train-fit 1.261 / dev 139 video); xuất tập con smoke 334 / 84 ảnh; **smoke 1 epoch trên dữ liệu thật đạt cho YOLOv8n và YOLO11n** (lần 1 hỏng do công cụ đo, đã sửa và có regression); dự toán local bước 5 / 15 epoch: kịch bản **~56,6 h GPU** (kịch bản chậm hơn ~74 h; dự toán theo giả định, không phải bảo đảm). Codex `2351c67` **PASS WITH LIMITATIONS** (split + smoke); base review → `2351c67`. Vòng 36: chỉ đính chính diễn giải R35-F1 (nguyên nhân confidence thấp ở checkpoint smoke **chưa xác định**) và R35-F2 (dự toán là kịch bản, không phải cận trên); số liệu không đổi. Vòng 37 (người dùng duyệt): **pilot chẩn đoán 2 model × 2 epoch** trên dataset smoke, cấu hình khóa trước (chỉ đổi `epochs`), **ĐẠT** cả hai. Model 5 lớp trước bước học đã có điểm lớp gần 0 (nhánh phân lớp `cv3` khởi tạo mới khi 80 → 5 lớp); các bước học làm điểm và AP tăng; warmup thực chạy khớp mã. Runner sửa lỗi `final_eval` ghi đè `val_s` epoch cuối. Codex `0c0a07a` **PASS WITH LIMITATIONS** (đóng R35-F1/F2); base review → `0c0a07a`. Vòng 38 (người dùng duyệt bước 5): **đã xuất dataset đầy đủ** `data/yolo_bdd5_v1` — 50.487 ảnh train-fit / 5.559 ảnh dev (khớp số đếm), 8,03 GB; hợp đồng v2 0 lỗi, kiểm chéo ĐẠT (danh sách frame dùng lại `chon_frame()` của exporter — đính chính vòng 39, R38-N2), trùng nội dung train/dev và BDD val 0 (UA-DETRAC test chỉ so tên video); dry-run hai model nhận đúng dataset. Codex `b83dc12` **PASS WITH LIMITATIONS**; base review → `b83dc12`. Vòng 39 (người dùng duyệt): **resume cho runner huấn luyện** (tiếp tục sau epoch đã hoàn tất, bằng resume của ultralytics với đường dẫn checkpoint; kiểm trước khi train; bảo toàn checkpoint; không trùng từng bit với lượt liên tục) và **bộ giám sát RAM bám PID** do công cụ tạo; kiểm trên fixture CPU và dataset smoke GPU, không train trên dataset đầy đủ. Codex `1454b3d` **REQUEST CHANGES** (R39-F1: nhật ký resume bị nối sau dòng cuối ghi dở); vòng 40 sửa riêng R39-F1 (sao lưu nguyên byte nhật ký và xử lý phần đuôi trước lần ghi tiếp đầu tiên), base giữ `b83dc12`. Codex `421dd53` **PASS WITH LIMITATIONS** (đóng R39-F1 trong phạm vi sửa nhật ký), base → `421dd53`; vòng 41 sửa R40-N1/N2 (sao lưu nguyên byte và xử lý hàng cuối ghi dở của `results.csv` / `theo_batch.csv` trước lần ghi tiếp đầu tiên). Codex `6c8e37a` **PASS WITH LIMITATIONS** (đóng R40-N1/N2), base → `6c8e37a`. **Vòng 42 — người dùng duyệt training dài** (2026-10-03): YOLOv8n rồi YOLO11n, tuần tự, **tổng 15 epoch mỗi model** (kể cả khi resume), dataset đầy đủ bước 5, giữ tham số đã duyệt; ngân sách dự toán 56,64 h (kịch bản chậm hơn 73,95 h; không bảo đảm). Chỉ training + validation dev; kế hoạch `handoff/logs/review42/02_ke_hoach_luot_dai.md`. **Đã xong** (2026-10-06): YOLOv8n 15/15 (resume 2 lần), YOLO11n 15/15; dev mAP50-95 tốt nhất 0,280 / 0,272 (dev nội bộ, không phải test; chưa đủ để kết luận model nào tốt hơn). Codex `4d54744` **PASS WITH LIMITATIONS** (hồ sơ training); base review → `4d54744`. **Vòng 43** (người dùng duyệt đánh giá detector, 2026-10-06): kế hoạch khóa trước khi chạy; **BDD val đã chạy** cho 3 checkpoint (3 lượt exit 0, 39.973/39.973 ảnh; mAP50–95 trên 5 lớp: baseline 0,157, YOLOv8n fine-tune 0,251, YOLO11n fine-tune 0,247 — một lượt mỗi model, chỉ mô tả, chưa đủ để chọn model). **UA-DETRAC test chưa chạy**: chờ người dùng chốt ánh xạ lớp (van / others / lớp báo AP / 173 ảnh không có frame / ignored_region). So sánh bộ lọc, tracking, tối ưu Nano chưa làm. Codex `a03f6e1` **CHƯA NGHIỆM THU ĐẦY ĐỦ** (R43-B1: reviewer thiếu mã phụ thuộc để chạy test và kiểm đường chấm, là giới hạn bàn giao, chưa phải lỗi thuật toán; R43-N1 P2: metadata không phản ánh `--conf-van-hanh` hiệu lực ở nhánh CLI không mặc định, không làm sai ba kết quả BDD); base giữ `4d54744`. **Vòng 44** (người dùng duyệt; chỉ xử lý review `a03f6e1`): gửi kèm 17 file support nguyên byte, không đổi nội dung (R43-B1); metadata, scorer và chẩn đoán dùng cùng ngưỡng vận hành hiệu lực theo lượt (R43-N1). Ba lượt BDD vòng 43 dùng ngưỡng 0,25 nên lỗi cũ không kích hoạt, không chạy lại; kết quả và checkpoint giữ nguyên. Codex `3e94b21` **PASS WITH LIMITATIONS** (đóng R43-B1 + R43-N1 trong phạm vi mã / fixture / hồ sơ: test BDD 22/22, nhóm E 13/13 + 1 SKIP trên Linux / Python 3.12; không chạy lại inference trên dữ liệu thật, không kiểm trực tiếp checkpoint, không nghiệm thu UA-DETRAC); base review → `3e94b21`. **Vòng 45** (người dùng duyệt; chỉ đọc, không GPU): chuẩn bị quyết định D1–D5 cho đánh giá detector trên UA-DETRAC test — bảng D1–D5 và dự thảo giao thức (`handoff/logs/review45/06_du_thao_giao_thuc_detrac.md`), **chưa khóa, chưa chạy**; không sửa mã, dataset, checkpoint. Codex `d4715aa` **REQUEST CHANGES** (R45-F1 P2: D3 mô tả lớp N/A "không FP" nhưng scorer giữ nguyên vẫn đếm FP; R45-F2 P2: D5 khẳng định chiều thay đổi recall / AP và gộp quy tắc toolkit vượt quá bằng chứng; R45-F3 P3: phương án giữ 173 ảnh bỏ sót 265 lượt vùng ignore; R45-F4 P3: `doc_xml()` làm mất thuộc tính `sequence_attribute`); base giữ `3e94b21`. **Vòng 46** (người dùng duyệt; chỉ xử lý review `d4715aa`, base giữ `3e94b21`): sửa R45-F3 / R45-F4 trong công cụ phân tích (`test_cong_cu_detrac`: bản sửa 8/8, công cụ vòng 45 2/8); sửa diễn giải R45-F1 / R45-F2 và làm rõ 1–3 (tài liệu); không sửa `src/`, dataset, checkpoint, không GPU. Codex `e4a0311` **PASS WITH LIMITATIONS** (đóng R45-F1…F4 trong phạm vi công cụ phân tích và hồ sơ dự thảo; R45-F1 đóng ở mức đặc tả / hồ sơ, adapter D3 chưa viết nên chưa nghiệm thu; reviewer không có ảnh, XML gốc, checkpoint, không chạy toolkit; hai ghi chú không chặn: R46-N1 precision của lớp không GT cần điều kiện sau xử lý ignore, R46-N2 kích thước 960×540 mới kiểm ảnh đầu / cuối mỗi video). Kế hoạch R46 (2026-10-09; chỉ tài liệu, chưa được review): ghi nhận `e4a0311`, đính chính câu chữ R46-N1 / R46-N2, bảng quyết định D1–D5 cho người dùng (`handoff/REVIEW46_CHI_TIET.md` mục 9–11). Review `e4a0311` và bằng chứng chưa lưu vào repo, gói chưa archive; base review **vẫn `3e94b21`**. **D1–D5 CHƯA DUYỆT**, không phương án nào được chọn. **Tiếp theo** (theo review `e4a0311`): người dùng chốt D1–D5; chỉ sau đó mới triển khai / kiểm adapter, fingerprint toàn bộ đầu vào và khóa kế hoạch trước ba lượt đánh giá PC, mỗi bước cần người dùng duyệt. UA-DETRAC test chưa chạy; so sánh bộ lọc, tracking, tối ưu Nano chưa làm |
| 3 | **Đánh giá detector riêng** (AP theo lớp, mAP50, mAP50–95, precision/recall tại ngưỡng vận hành), tạo **detection cache tái lập** | Evaluator 5 lớp đã có (vòng 27). Vòng 35: chạy trên dev **smoke** thật (84 ảnh) chỉ để kiểm pipeline; chưa đánh giá trên dev đầy đủ. **Vòng 43**: đã đánh giá detector trên **BDD val** (COCO-video, 39.973 ảnh) cho 3 checkpoint theo kế hoạch khóa — AP theo lớp, mAP50 / mAP50–95, P/R tại conf 0,25 (`handoff/REVIEW43_CHI_TIET.md`); UA-DETRAC test chờ quyết định ánh xạ lớp. Dự đoán đánh giá (NPZ, conf ≥ 0,001) đã lưu nhưng **chưa** phải detection cache cho tracking; cache chưa tạo. **Vòng 44**: Codex `3e94b21` đóng R43-B1 / R43-N1 (metadata của entry point BDD val dùng ngưỡng vận hành hiệu lực theo lượt); số liệu BDD val vòng 43 giữ nguyên, không chạy lại. **Vòng 45–46**: bảng D1–D5 và dự thảo giao thức cho UA-DETRAC test (chỉ đọc, chưa khóa); Codex `d4715aa` **REQUEST CHANGES**, sửa ở vòng 46; Codex `e4a0311` **PASS WITH LIMITATIONS**. **D1–D5 CHƯA DUYỆT**; adapter UA-DETRAC chưa viết, UA-DETRAC test chưa chạy, chưa có kết quả detector UA-DETRAC; detection cache vẫn chưa tạo |
| 4 | So sánh bộ lọc **CV / EKF / UKF / PF**, đối chứng **SORT / ByteTrack** bằng các phép so sánh có kiểm soát | Chưa bắt đầu trong plan mới |
| 5 | **Đánh giá cuối** trên **BDD100K val** và **UA-DETRAC test** | Chưa |
| 6 | Cuối cùng mới **tối ưu tốc độ hệ thống Nano**, so với cấu hình baseline đã ghi nhận | Chưa |

Lớp đã khóa: `0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle` (`src/detector_classes.py`).

## Quy tắc phạm vi

- **Hỏi trước**: mọi đề xuất thêm model, dataset, thuật toán, ablation hoặc tính năng ngoài danh sách trên phải hỏi người
  dùng trước khi thực hiện. Ví dụ đang **chờ quyết định**: dùng gói BDD100K *detection* 100K (ảnh tĩnh, nhãn `det_20`) thay
  hoặc cùng gói MOT train. Gói này khác nguồn với MOT, nên chưa dùng.
- **Chọn bằng dev nội bộ**: checkpoint, số epoch và mọi tham số chỉ chọn bằng **dev nội bộ**. Dev này tách **theo video** từ
  split `train` chính thức của BDD100K MOT.
- **Hai tập đánh giá cuối không tham gia lựa chọn**: BDD100K val (bản COCO-video đã định danh) và UA-DETRAC test.
  - Không dùng hai tập này làm train hay dev.
  - Cả hai đã được xem trong các vòng nghiên cứu trước (thí nghiệm transfer BDD, mốc 2 UA-DETRAC), nên **không gọi là tập
    test hoàn toàn chưa từng quan sát**.
- **Phạm vi vòng 27**: chỉ chuẩn bị và smoke. Chưa chạy training dài, benchmark tracking đầy đủ hay tối ưu Nano.
- **Giữ nguyên**: hệ thống Nano v16, kết quả và lock cũ, file riêng của người dùng. Dataset, weights, video và cache lớn nằm
  ngoài gói delta.
