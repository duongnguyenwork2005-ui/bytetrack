# Kế hoạch đã chốt — từ vòng 27 (2026-09-29)

Nguồn: yêu cầu của người dùng sau khi Codex nghiệm thu `239b9c6` (**PASS WITH LIMITATIONS**, `handoff/CODEX_REVIEW_239b9c6.md`).
File này là phạm vi làm việc cho các vòng sau. Thay đổi phạm vi phải do người dùng quyết định.

## 0. TRẠNG THÁI HIỆN TẠI — tài liệu trạng thái chuẩn (cập nhật 2026-10-10)

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

Kế hoạch chi tiết cho báo cáo 17/10: **chưa có file trong repo** (chưa rõ; người dùng sẽ duyệt kế hoạch mới).

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
  `handoff/logs/review47/04_ket_qua_danh_gia_detrac_test.md`; số liệu `results/phase2/detector_eval/detrac_test/`.

**Việc đang dở / chưa chốt**
- Kết quả R47 bước 1–4 (`handoff/logs/review47/`, `src/fingerprint_detrac_test.py`, `src/danh_gia_detector_detrac_test.py`,
  `src/test_danh_gia_detector_detrac_test.py`, `results/phase2/detector_eval/`) và các sửa tài liệu vòng 46 **chưa
  commit** — người dùng quyết định. Bản sao nguyên byte: nhánh `backup/pre-cleanup-20261010-1429` (không gồm NPZ).
- Lượt orchestrator R46 và R47 đều **đã dừng** (R47: dừng tay 02:48 ngày 10/10 khi chờ hạn mức); repo đã khác mốc
  lưu của cả hai, nên không `start`/`recover` lại hai lượt này nếu chưa kiểm — lập kế hoạch mới thay vì chạy tiếp.

**Bước tiếp theo:** người dùng duyệt kế hoạch báo cáo 17/10 (theo thứ tự ưu tiên trên). Không tự mở việc mới.

**Điểm chưa rõ**
- Nguyên nhân recall bus thấp (~0,34) của hai checkpoint fine-tune trên UA-DETRAC test; mức đóng góp của nhầm lớp
  bus → truck / car chưa xác định (`04_ket_qua_danh_gia_detrac_test.md` mục 4).
- PF với thiết kế nhiễu khác (roughening / proposal / Q riêng) chưa chạy; số hạt chưa giải thích hết khoảng cách.
- Nguồn gốc cấu hình 53/56 sequence cache `legacy` và 113 thư mục run `LEGACY` không xác minh được bằng hash.
- Nhãn BDD MOT train chưa xác thực bằng archive gốc (chỉ snapshot bên thứ ba đã ghim).

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
