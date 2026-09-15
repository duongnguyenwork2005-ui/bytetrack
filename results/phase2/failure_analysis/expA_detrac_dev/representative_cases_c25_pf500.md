# Ca tieu bieu — nguyen nhan mat ID (pf500, dieu kien c25, dev UA-DETRAC 12 video)

Sinh boi `src/representative_cases.py --dir results/phase2/failure_analysis/expA_detrac_dev --tracker pf500 --cond c25 --n 2` (chon xac dinh: uu tien tang FULL, su kien dai nhat). So lieu day du: `event_causes_c25_pf500.csv`; log tung frame quanh su kien: `event_window_log.csv`.

Dinh nghia ket qua (mau so = 184 su kien du dieu kien theo GT): `preserved` = ID ngay truoc == ID ngay sau; `switched` = khac; `lost_after` = co ID truoc, khong ghep lai duoc trong cua so sau (GT van con); `no_match_before` = tracker chua tung ghep GT truoc su kien (khong danh gia duoc phan giu ID).

## prediction_drift — 37 ca that bai
- **MVI_40992 / GT track 6 / event 0** — che khuat frame 224–275 (52 frame, FULL), ket qua `switched`, ID truoc 25 → sau 30
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 276 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.962, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 25 tai frame do: trang thai Lost, da mat 21 buoc, IoU(du doan, det) = 0.000, lech tam 244 px, cost fused = 1.000 (det conf 0.711), det duoc ghep boi track 30 o vong 1; track bi xoa tai frame 286
  - nguon ID sau: new_track_from_det (frame sinh 260)
  - nhan: `prediction_drift` → nguyen nhan chinh **prediction_drift**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 6 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40992 / GT track 2 / event 0** — che khuat frame 15–54 (40 frame, FULL), ket qua `switched`, ID truoc 1 → sau 9
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 55 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.961, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 1 tai frame do: trang thai Lost, da mat 31 buoc, IoU(du doan, det) = 0.000, lech tam 200 px, cost fused = 1.000 (det conf 0.418), det duoc ghep boi track 9 o vong 1; track bi xoa tai frame 55
  - nguon ID sau: new_track_from_det (frame sinh 41)
  - nhan: `prediction_drift` → nguyen nhan chinh **prediction_drift**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 2 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## buffer_expired — 27 ca that bai
- **MVI_63544 / GT track 10 / event 1** — che khuat frame 159–265 (107 frame, FULL), ket qua `switched`, ID truoc 84 → sau 190
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 266 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.883, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 84 tai frame do: trang thai removed, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 232
  - nguon ID sau: new_track_from_det (frame sinh 264)
  - nhan: `buffer_expired` → nguyen nhan chinh **buffer_expired**
  - truy log: `event_window_log.csv` loc `video == "MVI_63544" & gt_track == 10 & event_id == 1` (cot kind = track/det_high/det_low/gt)
- **MVI_40992 / GT track 37 / event 0** — che khuat frame 1808–1898 (91 frame, FULL), ket qua `switched`, ID truoc 369 → sau 400
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1899 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.949, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 369 tai frame do: trang thai removed, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 1848
  - nguon ID sau: new_track_from_det (frame sinh 1870)
  - nhan: `buffer_expired` → nguyen nhan chinh **buffer_expired**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 37 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## detector_miss — 9 ca that bai
- **MVI_20062 / GT track 25 / event 0** — che khuat frame 713–795 (83 frame, partial), ket qua `lost_after`, ID truoc 369 → sau -
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = - (cach het su kien - frame); IoU tot nhat sau su kien: high 0.124, low 0.000, cache(0,05) 0.283 @conf 0.053
  - track 369 tai frame do: trang thai -, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 794
  - nguon ID sau: nan (frame sinh -)
  - nhan: `detector_miss` → nguyen nhan chinh **detector_miss**
  - truy log: `event_window_log.csv` loc `video == "MVI_20062" & gt_track == 25 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40752 / GT track 146 / event 0** — che khuat frame 1523–1591 (69 frame, partial), ket qua `lost_after`, ID truoc 1528 → sau -
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = - (cach het su kien - frame); IoU tot nhat sau su kien: high 0.412, low 0.000, cache(0,05) 0.412 @conf 0.729
  - track 1528 tai frame do: trang thai -, da mat - buoc, IoU(du doan, det) = -, lech tam - px, cost fused = - (det conf -), det duoc ghep boi track - o vong -; track bi xoa tai frame 1564
  - nguon ID sau: nan (frame sinh -)
  - nhan: `detector_miss` → nguyen nhan chinh **detector_miss**
  - truy log: `event_window_log.csv` loc `video == "MVI_40752" & gt_track == 146 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## score_fusion_reject — 6 ca that bai
- **MVI_63544 / GT track 8 / event 3** — che khuat frame 159–220 (62 frame, FULL), ket qua `lost_after`, ID truoc 104 → sau -
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 221 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.838, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 104 tai frame do: trang thai Lost, da mat 2 buoc, IoU(du doan, det) = 0.294, lech tam 9 px, cost fused = 0.925 (det conf 0.253), det duoc ghep boi track -1 o vong 0; track bi xoa tai frame 250
  - nguon ID sau: nan (frame sinh -)
  - nhan: `score_fusion_reject` → nguyen nhan chinh **score_fusion_reject**
  - truy log: `event_window_log.csv` loc `video == "MVI_63544" & gt_track == 8 & event_id == 3` (cot kind = track/det_high/det_low/gt)
- **MVI_40992 / GT track 23 / event 1** — che khuat frame 1187–1218 (32 frame, partial), ket qua `lost_after`, ID truoc 233 → sau -
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1221 (cach het su kien 2 frame); IoU tot nhat sau su kien: high 0.602, low 0.000, cache(0,05) 0.619 @conf 0.174
  - track 233 tai frame do: trang thai Lost, da mat 25 buoc, IoU(du doan, det) = 0.499, lech tam 10 px, cost fused = 0.863 (det conf 0.276), det duoc ghep boi track -1 o vong 0; track bi xoa tai frame 1227
  - nguon ID sau: nan (frame sinh -)
  - nhan: `score_fusion_reject` → nguyen nhan chinh **score_fusion_reject**
  - truy log: `event_window_log.csv` loc `video == "MVI_40992" & gt_track == 23 & event_id == 1` (cot kind = track/det_high/det_low/gt)

## competition — 4 ca that bai
- **MVI_40871 / GT track 4 / event 2** — che khuat frame 285–312 (28 frame, partial), ket qua `switched`, ID truoc 89 → sau 250
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 313 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.853, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 89 tai frame do: trang thai Lost, da mat 9 buoc, IoU(du doan, det) = 0.602, lech tam 8 px, cost fused = 0.567 (det conf 0.718), det duoc ghep boi track 250 o vong 1; track bi xoa tai frame 335
  - nguon ID sau: existing_other_track (frame sinh 279)
  - nhan: `competition` → nguyen nhan chinh **competition**
  - truy log: `event_window_log.csv` loc `video == "MVI_40871" & gt_track == 4 & event_id == 2` (cot kind = track/det_high/det_low/gt)
- **MVI_40871 / GT track 16 / event 1** — che khuat frame 95–117 (23 frame, partial), ket qua `switched`, ID truoc 55 → sau 11
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 118 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.943, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 55 tai frame do: trang thai Lost, da mat 6 buoc, IoU(du doan, det) = 0.423, lech tam 33 px, cost fused = 0.724 (det conf 0.651), det duoc ghep boi track 11 o vong 1; track bi xoa tai frame 143
  - nguon ID sau: existing_other_track (frame sinh 1)
  - nhan: `competition` → nguyen nhan chinh **competition**
  - truy log: `event_window_log.csv` loc `video == "MVI_40871" & gt_track == 16 & event_id == 1` (cot kind = track/det_high/det_low/gt)

## matched_but_eval_mismatch — 2 ca that bai
- **MVI_63554 / GT track 58 / event 0** — che khuat frame 426–446 (21 frame, partial), ket qua `switched`, ID truoc 336 → sau 405
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 447 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.920, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 336 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.383, lech tam 122 px, cost fused = 0.696 (det conf 0.793), det duoc ghep boi track 336 o vong 1; track bi xoa tai frame 484
  - nguon ID sau: new_track_from_det (frame sinh 454)
  - nhan: `matched_but_eval_mismatch` → nguyen nhan chinh **matched_but_eval_mismatch**
  - truy log: `event_window_log.csv` loc `video == "MVI_63554" & gt_track == 58 & event_id == 0` (cot kind = track/det_high/det_low/gt)
- **MVI_40752 / GT track 63 / event 0** — che khuat frame 580–583 (4 frame, partial), ket qua `switched`, ID truoc 678 → sau 712
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 584 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.971, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 678 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.287, lech tam 49 px, cost fused = 0.770 (det conf 0.800), det duoc ghep boi track 678 o vong 1; track bi xoa tai frame 618
  - nguon ID sau: new_track_from_det (frame sinh 588)
  - nhan: `matched_but_eval_mismatch` → nguyen nhan chinh **matched_but_eval_mismatch**
  - truy log: `event_window_log.csv` loc `video == "MVI_40752" & gt_track == 63 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## hijacked_during — 1 ca that bai
- **MVI_40871 / GT track 3 / event 0** — che khuat frame 97–125 (29 frame, partial), ket qua `switched`, ID truoc 11 → sau 64
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 126 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.917, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 11 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.108, lech tam 55 px, cost fused = 0.917 (det conf 0.774), det duoc ghep boi track 64 o vong 1; track bi xoa tai frame 190
  - nguon ID sau: new_track_from_det (frame sinh 117)
  - nhan: `hijacked_during|prediction_drift` → nguyen nhan chinh **hijacked_during**
  - truy log: `event_window_log.csv` loc `video == "MVI_40871" & gt_track == 3 & event_id == 0` (cot kind = track/det_high/det_low/gt)

## Ca THANH CONG tieu bieu (preserved) — 64 ca
- **MVI_40752 / GT track 69 / event 2** — che khuat frame 720–729 (10 frame, FULL), ket qua `preserved`, ID truoc 768 → sau 768
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 730 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.867, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 768 tai frame do: trang thai Lost, da mat 4 buoc, IoU(du doan, det) = 0.493, lech tam 16 px, cost fused = 0.651 (det conf 0.708), det duoc ghep boi track 768 o vong 1; track bi xoa tai frame 770
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_40752" & gt_track == 69 & event_id == 2` (cot kind = track/det_high/det_low/gt)
- **MVI_40871 / GT track 31 / event 1** — che khuat frame 901–1099 (199 frame, partial), ket qua `preserved`, ID truoc 28 → sau 28
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 1100 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.804, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 28 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.982, lech tam 0 px, cost fused = 0.313 (det conf 0.700), det duoc ghep boi track 28 o vong 1; track bi xoa tai frame -
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_40871" & gt_track == 31 & event_id == 1` (cot kind = track/det_high/det_low/gt)
- **MVI_20052 / GT track 43 / event 4** — che khuat frame 437–539 (103 frame, partial), ket qua `preserved`, ID truoc 4 → sau 4
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 540 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.833, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 4 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.930, lech tam 1 px, cost fused = 0.494 (det conf 0.544), det duoc ghep boi track 4 o vong 1; track bi xoa tai frame -
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_20052" & gt_track == 43 & event_id == 4` (cot kind = track/det_high/det_low/gt)
- **MVI_40871 / GT track 32 / event 2** — che khuat frame 248–328 (81 frame, partial), ket qua `preserved`, ID truoc 77 → sau 77
  - detector: frame dau tien thay lai GT (IoU ≥ 0,5, tier high) = 329 (cach het su kien 0 frame); IoU tot nhat sau su kien: high 0.836, low 0.000, cache(0,05) 0.000 @conf 0.000
  - track 77 tai frame do: trang thai Tracked, da mat 1 buoc, IoU(du doan, det) = 0.979, lech tam 1 px, cost fused = 0.478 (det conf 0.533), det duoc ghep boi track 77 o vong 1; track bi xoa tai frame -
  - nguon ID sau: nan (frame sinh -)
  - nhan: `success` → nguyen nhan chinh **success**
  - truy log: `event_window_log.csv` loc `video == "MVI_40871" & gt_track == 32 & event_id == 2` (cot kind = track/det_high/det_low/gt)
