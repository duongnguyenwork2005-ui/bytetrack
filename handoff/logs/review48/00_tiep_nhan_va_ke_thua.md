# R48 — M0: tiếp nhận trạng thái và bảng kế thừa (2026-10-10, phiên thiết lập, trước khi tạo lượt)

Bản giao việc gốc: `handoff/KE_HOACH_BAO_CAO_20261017.md` (sao nguyên byte từ bản người dùng giao, sha256
`0d5deb4acac62d2b…`). File này là đầu ra M0. Lượt orchestrator R48 thực hiện M1–M6 theo plan JSON đã chuyển đổi.

## 1. Trạng thái lúc tiếp nhận

- Repo `D:\LABOTARY\bytetrack`, nhánh `phase2-mixed-traffic`, remote `origin` (GitHub, **public**). HEAD trước commit
  thiết lập R48: `482baaf` (R47 bước 1–4), trước đó `43cd0de` (tài liệu R46), `44a531c` (dọn hệ thống).
- Working tree: chỉ còn `.gitignore` và `src/demo/rerun_with_state_log.py` sửa dở (file riêng người dùng, được bảo vệ)
  cùng các file chưa track cũ (video, NPZ, kết quả cũ chưa commit) — không thuộc R48.
- Lượt cũ: R46 (`bytetrack_r46_3moc_20261009_025215`) và R47 (`bytetrack_r47_fingerprint_adapter_20261010_021706`)
  đều đã dừng, không tiến trình sống, khóa `D:\LABOTARY\.bytetrack.orchestrator.lock` không bị giữ. Hồ sơ hai lượt giữ
  nguyên; không `recover`/`start` lại (repo đã khác mốc lưu của cả hai).
- Orchestrator 1.7.0 (`D:\LABOTARY\automation\auto_review`), chính sách `claude_first`: Claude làm chính và tự kiểm;
  Codex review bắt buộc khi mỗi mốc xong (cổng nghiệm thu), không review từng lời gọi.

## 2. Cấu hình và dữ liệu (đã có, không đổi trong R48)

| Mục | Giá trị / vị trí |
|:--|:--|
| Cache detection cho tracking | `yolov8n-coco_conf0.05_all` (mốc 2; fingerprint `src/run_fingerprint.py`) |
| Tracker PF | `src/pf_ctrv.py` (`PFCTRV`, kế thừa `EKFTrackerCTRV`), `src/tracker_ctrv.py`; cấu hình `configs/phase2/trackers/bytetrack_pf_ctrv_n{100,300,500}.yaml` |
| Dev / test UA-DETRAC | `results/phase2/manifests/detrac_dev.json`, `detrac_test.json` (40 video) |
| BDD train/dev detector | split `configs/phase2/detector/splits/bdd5_train_v1.json`; nhãn `data/raw/bdd100k/labels/box_track_20/train` (phương án C); xuất `data/yolo_bdd5_v1` |
| BDD val (COCO-video) | `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` |
| Augmentation đang cấu hình | `configs/phase2/detector/train_bdd5.yaml` (hsv, translate 0,1, scale 0,5, fliplr 0,5, mosaic 1,0; mixup / copy_paste 0) |
| Nguồn lý thuyết đặt sẵn | `tai_lieu_nguon/README.md` (toàn văn ByteTrack arXiv; metadata Crossref; **không** có toàn văn Arulampalam 2002) |

## 3. Bảng kế thừa kết quả cũ (dùng lại, không chạy lại)

| Kết quả | Trạng thái | Nguồn |
|:--|:--|:--|
| Tracking mốc 2 trên 40 video test (CV/EKF/UKF/PF, Nhóm B `_fix1`) | đã nghiệm thu | `CODEX_HANDOFF.md` mục 5, 8; `PHASE2_REPORT.md` |
| Chẩn đoán PF (ESS/N ~1 % sau mất dấu dài; N 100→500 thu hẹp nhưng không hết khoảng cách) | đã đo | `CODEX_HANDOFF.md` mục 5 ý 7; `handoff/logs/pf_diag_report.log` |
| BDD transfer COCO-video val | đã nghiệm thu | `results/phase2/bdd_transfer_cocovid_val/bao_cao/` |
| Detector 3 checkpoint trên BDD val (vòng 43) | đã nghiệm thu | `handoff/REVIEW43_CHI_TIET.md` |
| Fingerprint + đánh giá 3 detector trên UA-DETRAC test (R47) | Codex approve; commit `482baaf` | `handoff/logs/review47/01…04` |
| Huấn luyện YOLOv8n / YOLO11n 15 epoch BDD5 | xong | `handoff/DETECTOR_TRAIN_PLAN.md`, `results/phase2/detector_train/` |

Mốc đối chiếu của slide lần 2 (BDD train 1.261 video / 50.487 ảnh; dev 139 / 5.559; BDD val 200 / 39.973) **chưa**
được audit lại trong R48 — đó là việc của M2.

## 4. Ngân sách và hạn mức (đo lúc 16:09 ngày 10/10)

- Plan R48: 4 mốc (= 4 cổng Codex), `max_rounds` 10, `max_turns` 80, timeout mỗi lời gọi 3.600 s; lệnh Bash của Claude
  chạy foreground (giới hạn của CLI ~10 phút/lệnh) nên việc dài phải chia đoạn, tiếp tục được.
- Codex (`CODEX_HOME` `.codex-cli-test-20261008-121617`): telemetry mới nhất 08:05 — tuần **85 %** (reset 16/10 02:22);
  cửa sổ 5 giờ đã reset lúc 11:36, mức hiện tại chưa đo. Ngưỡng hỏi dự phòng tuần 80 % → lần gọi Codex đầu sẽ hỏi người
  dùng qua Gmail (cơ chế có sẵn).
- Claude: không đọc được hạn mức hiện tại ngoài lời gọi; số gần nhất (≈02:00) tuần 35 %, reset 16/10 05:00. Lời gọi đầu
  của lượt sẽ ghi telemetry. Hết hạn mức: orchestrator chờ `resetsAt`, không chuyển việc sang GPT.

## 5. Ánh xạ bản giao việc → mốc của plan

| Bản giao việc | Mốc plan R48 | Cổng Codex |
|:--|:--|:--|
| M0 | file này + commit thiết lập (phiên thiết lập, không phải agent của lượt) | — |
| M1 | `R48A_LY_THUYET_DOI_CHIEU_PF` | có |
| M2 (+ kiểm cấu hình augmentation của M5) | `R48B_AUDIT_DU_LIEU` | có |
| M3 + phần test tính đúng của M4 | `R48C_GIAO_THUC_VA_DUNG_DAN_PF` | có — cổng "giao thức + correctness trước thí nghiệm lớn" |
| M4 (thí nghiệm, ablation) + M5 (chỉ thiết kế) + M6 | `R48D_THI_NGHIEM_VA_TONG_HOP` | có |

M5 chỉ thiết kế trong lượt này: huấn luyện detector cần GPU nhiều giờ, vượt timeout 3.600 s/lời gọi và ngân sách vòng;
phần bị chặn được ghi rõ, không làm chậm lý thuyết/báo cáo.
