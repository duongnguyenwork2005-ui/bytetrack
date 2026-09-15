# Ca tieu bieu — nguyen nhan mat ID (cv, dieu kien c25, dev UA-DETRAC 12 video)

Sinh boi `src/representative_cases.py --dir results/phase2/failure_analysis/expA_detrac_dev --tracker cv --cond c25 --n 2` (chon xac dinh: uu tien tang FULL, su kien dai nhat). So lieu day du: `event_causes_c25_cv.csv`; log tung frame quanh su kien: `event_window_log.csv`.

Dinh nghia ket qua (mau so = 184 su kien du dieu kien theo GT): `preserved` = ID ngay truoc == ID ngay sau; `switched` = khac; `lost_after` = co ID truoc, khong ghep lai duoc trong cua so sau (GT van con); `no_match_before` = tracker chua tung ghep GT truoc su kien (khong danh gia duoc phan giu ID).

## prediction_drift — 20 ca that bai
- **MVI_40992 / GT track 23 / event 0** — che khuat frame 1123–1168 (46 frame, FULL), ket qua `switched`, ID truoc 164 → sau 179
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1169 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.938, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 164 tai frame do: trang thai Lost, da mat 28 buoc, IoU(du doan, det) = 0.064, lech tam 73 px, cost fused = 0.953 (det conf 0.736), det duoc ghep boi track 179 o vong 1; track bi xoa tai frame 1172
  - nguon ID sau: new_track_from_det (frame sinh 1164)
  - nhan: `prediction_drift` → nguyen nhan chinh **prediction_drift**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 23 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40992 / GT track 21 / event 1** — che khuat frame 1025–1068 (44 frame, FULL), ket qua `switched`, ID truoc 131 → sau 143
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1069 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.942, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 131 tai frame do: trang thai Lost, da mat 31 buoc, IoU(du doan, det) = 0.150, lech tam 52 px, cost fused = 0.866 (det conf 0.891), det duoc ghep boi track 143 o vong 1; track bi xoa tai frame 1069
  - nguon ID sau: new_track_from_det (frame sinh 1053)
  - nhan: `prediction_drift` → nguyen nhan chinh **prediction_drift**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 21 & event_id == 1` (cot kind = track/det_high/det_low/gt)

## score_fusion_reject — 12 ca that bai
- **MVI_63544 / GT track 8 / event 3** — che khuat frame 159–220 (62 frame, FULL), ket qua `lost_after`, ID truoc 97 → sau -
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 221 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.838, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 97 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.476, lech tam 5 px, cost fused = 0.879 (det conf 0.253), det duoc ghep boi track -1 o vong 0; track bi xoa tai frame 251
  - nguon ID sau: nan (frame sinh -)
  - nhan: `score_fusion_reject` → nguyen nhan chinh **score_fusion_reject**
  - truy log: `event_window_log.csv` loc `video == "MVI_63544" & gt_track == 8 & event_id == 3` (cot kind = track/det_high/det_low/gt)
- **MVI_40992 / GT track 6 / event 0** — che khuat frame 224–275 (52 frame, FULL), ket qua `switched`, ID truoc 20 → sau 26
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 276 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.962, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 20 tai frame do: trang thai Lost, da mat 21 buoc, IoU(du doan, det) = 0.239, lech tam 62 px, cost fused = 0.830 (det conf 0.711), det duoc ghep boi track 26 o vong 1; track bi xoa tai frame 416
  - nguon ID sau: new_track_from_det (frame sinh 260)
  - nhan: `score_fusion_reject` → nguyen nhan chinh **score_fusion_reject**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 6 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## buffer_expired — 12 ca that bai
- **MVI_40992 / GT track 37 / event 0** — che khuat frame 1808–1898 (91 frame, FULL), ket qua `switched`, ID truoc 297 → sau 328
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1899 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.949, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 297 tai frame do: trang thai removed, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 1878
  - nguon ID sau: new_track_from_det (frame sinh 1870)
  - nhan: `buffer_expired` → nguyen nhan chinh **buffer_expired**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 37 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40992 / GT track 36 / event 1** — che khuat frame 1769–1835 (67 frame, FULL), ket qua `switched`, ID truoc 277 → sau 300
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1836 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.924, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 277 tai frame do: trang thai removed, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 1819
  - nguon ID sau: new_track_from_det (frame sinh 1797)
  - nhan: `buffer_expired` → nguyen nhan chinh **buffer_expired**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 36 & event_id == 1` (cot kind = track/det_high/det_low/gt)

## detector_miss — 9 ca that bai
- **MVI_20062 / GT track 25 / event 0** — che khuat frame 713–795 (83 frame, partial), ket qua `lost_after`, ID truoc 359 → sau -
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = - (cach het su kien - frame); IoU tot nhat sau su kien: high 0.124, low 0.000, cache(0,05) 0.283 @conf 0.053
  - track 359 tai frame do: trang thai -, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 794
  - nguon ID sau: nan (frame sinh -)
  - nhan: `detector_miss` → nguyen nhan chinh **detector_miss**
  - truy log: `event_window_log.csv` loc `video == "MVI_20062" & gt_track == 25 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40752 / GT track 146 / event 0** — che khuat frame 1523–1591 (69 frame, partial), ket qua `lost_after`, ID truoc 1385 → sau -
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = - (cach het su kien - frame); IoU tot nhat sau su kien: high 0.412, low 0.000, cache(0,05) 0.412 @conf 0.729
  - track 1385 tai frame do: trang thai -, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 1607
  - nguon ID sau: nan (frame sinh -)
  - nhan: `detector_miss` → nguyen nhan chinh **detector_miss**
  - truy log: `event_window_log.csv` loc `video == "MVI_40752" & gt_track == 146 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## competition — 3 ca that bai
- **MVI_40992 / GT track 17 / event 0** — che khuat frame 838–889 (52 frame, FULL), ket qua `switched`, ID truoc 108 → sau 113
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 890 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.968, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 108 tai frame do: trang thai Lost, da mat 22 buoc, IoU(du doan, det) = 0.338, lech tam 47 px, cost fused = 0.706 (det conf 0.870), det duoc ghep boi track 113 o vong 1; track bi xoa tai frame 899
  - nguon ID sau: new_track_from_det (frame sinh 877)
  - nhan: `competition` → nguyen nhan chinh **competition**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 17 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40871 / GT track 2 / event 0** — che khuat frame 129–147 (19 frame, partial), ket qua `switched`, ID truoc 9 → sau 91
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 148 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.827, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 9 tai frame do: trang thai Lost, da mat 4 buoc, IoU(du doan, det) = 0.326, lech tam 48 px, cost fused = 0.770 (det conf 0.706), det duoc ghep boi track 91 o vong 1; track bi xoa tai frame 175
  - nguon ID sau: new_track_from_det (frame sinh 143)
  - nhan: `competition` → nguyen nhan chinh **competition**
  - truy log: `event_window_log.csv` loc `video == "MVI_40871" & gt_track == 2 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## matched_but_eval_mismatch — 1 ca that bai
- **MVI_40752 / GT track 113 / event 1** — che khuat frame 1204–1226 (23 frame, partial), ket qua `switched`, ID truoc 1048 → sau 1096
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1227 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.796, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 1048 tai frame do: trang thai Lost, da mat 3 buoc, IoU(du doan, det) = 0.652, lech tam 6 px, cost fused = 0.728 (det conf 0.418), det duoc ghep boi track 1048 o vong 1; track bi xoa tai frame 1258
  - nguon ID sau: new_track_from_det (frame sinh 1228)
  - nhan: `matched_but_eval_mismatch` → nguyen nhan chinh **matched_but_eval_mismatch**
  - truy log: `event_window_log.csv` loc `video == "MVI_40752" & gt_track == 113 & event_id == 1` (cot kind = track/det_high/det_low/gt)

## hijacked_during — 1 ca that bai
- **MVI_40752 / GT track 10 / event 1** — che khuat frame 122–128 (7 frame, partial), ket qua `switched`, ID truoc 74 → sau 106
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 129 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.926, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 74 tai frame do: trang thai Lost, da mat 4 buoc, IoU(du doan, det) = 0.000, lech tam 106 px, cost fused = 1.000 (det conf 0.759), det duoc ghep boi track 106 o vong 1; track bi xoa tai frame 156
  - nguon ID sau: existing_other_track (frame sinh 121)
  - nhan: `hijacked_during|prediction_drift` → nguyen nhan chinh **hijacked_during**
  - truy log: `event_window_log.csv` loc `video == "MVI_40752" & gt_track == 10 & event_id == 1` (cot kind = track/det_high/det_low/gt)

## Ca THANH CONG tieu bieu (preserved) — 93 ca
- **MVI_63544 / GT track 10 / event 1** — che khuat frame 159–265 (107 frame, FULL), ket qua `preserved`, ID truoc 79 → sau 79
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 266 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.883, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 79 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.488, lech tam 4 px, cost fused = 0.790 (det conf 0.429), det duoc ghep boi track 79 o vong 1; track bi xoa tai frame 297
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_63544" & gt_track == 10 & event_id == 1` (cot kind = track/det_high/det_low/gt)
- **MVI_40213 / GT track 111 / event 0** — che khuat frame 776–806 (31 frame, FULL), ket qua `preserved`, ID truoc 515 → sau 515
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 807 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.949, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 515 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.860, lech tam 3 px, cost fused = 0.573 (det conf 0.496), det duoc ghep boi track 515 o vong 1; track bi xoa tai frame 860
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_40213" & gt_track == 111 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40752 / GT track 69 / event 2** — che khuat frame 720–729 (10 frame, FULL), ket qua `preserved`, ID truoc 681 → sau 681
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 730 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.867, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 681 tai frame do: trang thai Lost, da mat 4 buoc, IoU(du doan, det) = 0.586, lech tam 13 px, cost fused = 0.585 (det conf 0.708), det duoc ghep boi track 681 o vong 1; track bi xoa tai frame 770
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_40752" & gt_track == 69 & event_id == 2` (cot kind = track/det_high/det_low/gt)
- **MVI_40871 / GT track 31 / event 1** — che khuat frame 901–1099 (199 frame, partial), ket qua `preserved`, ID truoc 27 → sau 27
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1100 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.804, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 27 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.988, lech tam 0 px, cost fused = 0.309 (det conf 0.700), det duoc ghep boi track 27 o vong 1; track bi xoa tai frame -
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_40871" & gt_track == 31 & event_id == 1` (cot kind = track/det_high/det_low/gt)
