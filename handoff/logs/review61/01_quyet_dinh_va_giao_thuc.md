# R61 — Quyết định mốc 5 và giao thức BDD (R61_PREP)

Ngày 23/10/2026. Nguồn quyết định: ủy quyền chat người dùng 23/10/2026, ghi tại
`D:/LABOTARY/automation/runs_local/hoi_codex_R60_need_decision_20261023/answer.json`; plan R61 người dùng duyệt.
Hồ sơ R60 (`handoff/logs/review60/01_ho_so_quyet_dinh_moc5.md`) **giữ nguyên**, không điền phiếu ở đó.

## 1. Quyết định M5-Q1…Q8

| Câu | Lựa chọn |
|---|---|
| M5-Q1 | A — chạy mới `yolov8n_coco`, `yolov8n_mixed5`, `yolo11n_mixed5` trên BDD val; checkpoint/hash đúng R54 |
| M5-Q2 | A — CV, EKF, UKF, PF500 seed 0/1/2 |
| M5-Q3 | A — giữ c25 và c10 (không diễn giải c10−c25 là tác động riêng của association điểm thấp) |
| M5-Q4 | Giữ quy ước BDD cũ: 5 fps; buffer 1,2 s = 6 bước; `elapsed_mode=steps`; lớp [1,2,3,5,7]; `mot_crowd`; GT/seqmap hiện có; `assoc_log=false`; nhiễu theo bước; ảnh không annotation giữ trong timeline, chấm theo chính sách cũ |
| M5-Q5 | BDD COCO cũ chỉ lịch sử; tạo mới cache cho cả 3 detector; UA-DETRAC dùng nguyên R53/R56/R57/R59 đã approve |
| M5-Q6 | B — COMBINED + 6 hiệu số/nhóm + bootstrap ghép cặp theo video 5.000 mẫu, `default_rng(0)`, CI phân vị 95 %; HOTA chính, IDF1/IDSW bổ sung; 108 CI |
| M5-Q7 | A — wrapper mới `src/r61_bdd/`, gọi helper hiện có, không sửa helper |
| M5-Q8 | Báo cáo cuối hai phần riêng: UA-DETRAC test và BDD100K val |

## 2. Giao thức (định danh tường minh: `configs/phase2/r61_bdd/giao_thuc.json`)

- **Ma trận (kế hoạch, chưa chạy):** 3 detector × c25/c10 × {cv, ekf, ukf, pf500-s0/s1/s2} = 36 run; 200 video/run;
  39.973 frame/run; 600 đơn vị cache detector-video; 7.200 đơn vị tracking run-video. Tên `r61-<det>-<dk>-<bộ lọc>[-s<seed>]`.
- **Nguồn:** manifest `results/phase2/manifests/bdd100k_transfer_cocovid_val.json` (sha256 `2071c15b…c3ba`); nhãn
  COCO-video `data/raw/bdd100k/labels_cocovid/bdd_box_track_val_cocofmt.json` (`074ff795…5d5d`, không phải box_track_20
  native); GT/seqinfo `data/processed/BDDCOCOVID-val/`, seqmap `data/processed/seqmaps/BDDCOCOVID-val.txt` (`59003207…44e6`).
  Đọc frame bằng `BDD100KCocoDataset.frames` (không gọi `gt_dataframe`/`build_interim`).
- **Cache mới:** backend ultralytics, imgsz 640, conf 0,05, IoU 0,7, max_det 3000, classes null, half=false (FP32), batch 1,
  `mask_ignored_regions=true` với danh sách vùng rỗng của BDD (không tô); mixed5 ánh xạ qua `detector_classes` sang COCO,
  lưu `cls_goc`; unreadable/truncated ⇒ chặn. Device dự kiến `"0"`; **device thực, phiên bản thư viện, thời gian: chưa đo**
  (ghi ở cổng A).
- **Nạp:** conf > ngưỡng (ngặt) → lớp [1,2,3,5,7] → `[:300]` (`det_cache.filter_dets`), `allow_truncated=false`.
- **Tracker:** `run_tracking.load_tracker_cfg(yaml, {track_buffer: 6})` + `track_video(..., dt=0,2, "steps", seed)`;
  association ByteTrack gốc trong YAML; `assoc_log=false`; `pf_diag=false` (đường BDD hiện có không khai báo ⇒ mặc định
  false của `run_experiment`); PF500: `pf_num_particles=500`, `optimal_proposal=false` (mặc định), `roughening=0.0`.
  Không tuning.
- **Chấm:** `MotChallengeCrowdIgnore` (`dataset_class_for("bdd100k_coco") == "mot_crowd"`), HOTA/CLEAR/Identity,
  ngưỡng 0,5, `pedestrian` = lớp 1; mọi thư mục GT/tracker/output/error log đặt tường minh trong phiên; chỉ chấm full khi
  đủ 200 đơn vị/run (bộ điều phối B: đủ 7.200 trước khi chấm run nào).
- **Thống kê:** bảng chính COMBINED_SEQ; PF500 = trung bình metric COMBINED 3 seed (+từng seed, min/max). Bootstrap: một
  ma trận chỉ số 5.000 × 200 (`default_rng(0)`, mỗi mẫu `integers(0,200,size=200)` — cùng dãy với
  `boot_stats.paired_bootstrap`) dùng cho mọi run/seed; mỗi mẫu `combine_stacked` từng run, PF trung bình metric giữa seed
  rồi lấy hiệu số; phân vị 2,5/97,5 `method=linear`. 6 cặp × 3 metric × 6 nhóm = 108 CI; HOTA/IDF1 pp, IDSW count.
  Dung sai tái tổ hợp (khai báo trước fixture): HOTA/IDF1 1e-9 (thang [0,1]), IDSW 0. Cấm: trung bình metric video, gộp
  thống kê đủ giữa seed, bootstrap seed, p-value, CI so detector, hiệu chỉnh đa so sánh.
- **Định trước (theo vị trí manifest, không dựa điểm):** smoke A = vị trí 50, 150 + `b23b2649-8349d2a1` →
  `b1e1a7b8-a7426a97`, `b22f385b-5d7e5202`, `b23b2649-8349d2a1`; tái lập B = vị trí 100 `b204a5c1-05981158` cho cả 36 run;
  chấm lại B: `{yolov8n_coco, yolov8n_mixed5, yolo11n_mixed5} × {c25-ekf, c10-pf500-s2}`.
- **Vận hành:** tuần tự một detector/job; tiếp tục theo detector-video (cache) và run-video (tracking); đoạn foreground
  ngân sách mềm 420 s, dừng giữa đơn vị; exit 0 xong / 10 chưa xong / 2 lỗi hợp đồng / 1 lỗi bất ngờ.

## 3. Giới hạn đã chấp nhận

Tập BDD val đã từng được quan sát (BDD transfer cũ; 3 sequence smoke), không gọi là test chưa xem. Nguồn COCO-video, vehicle
gộp có bicycle, khác fps/nhiễu/ignore với UA-DETRAC; 615 ảnh không annotation (BDD) và 173 ảnh D4 (DETRAC) chưa rõ nguyên
nhân, chấm như frame không có GT, không suy ra âm tính thật; ít motorcycle/bicycle; không đánh giá che khuất BDD. Không so
trực tiếp HOTA giữa hai tập.
