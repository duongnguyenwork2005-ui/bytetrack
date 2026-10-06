# Trích dẫn nguồn cho các ghi chú review

Số dòng 1-based; chỉ đọc byte từ ZIP được review.

## handoff/REVIEW42_CHI_TIET.md

SHA256 `a4d0301a6f61e02e4ad6d5aef146ec24c4aee25adc7ef8d932790d5485b435c2`

```text
71:   8 s.
72: 
73: ## 4. YOLOv8n — **HOÀN TẤT 15/15 epoch** (3 lần chạy)
74: 
75: ### 4.1 Diễn biến
76: 
77: | Lần | Thư mục giám sát | Chạy | Epoch lưu | Kết thúc |
78: |:--|:--|:--|:--|:--|
```

```text
115: 
116: **Dev nội bộ** (validation của Ultralytics trên 5.559 ảnh dev; **không phải test**), mAP50-95 / mAP50:
117: 
118: | Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
119: |:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
120: | mAP50-95 | 0,143 | 0,177 | 0,220 | 0,224 | 0,240 | 0,251 | 0,252 | 0,253 | 0,266 | 0,266 | 0,275 | 0,275 | 0,278 | 0,278 | **0,280** |
121: | mAP50 | 0,233 | 0,282 | 0,343 | 0,351 | 0,375 | 0,388 | 0,394 | 0,393 | 0,412 | 0,413 | 0,428 | 0,427 | 0,431 | 0,432 | 0,435 |
122: 
```

```text
153: 
154: **Dev nội bộ** (không phải test), mAP50-95 / mAP50:
155: 
156: | Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 |
157: |:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|:--|
158: | mAP50-95 | 0,140 | 0,181 | 0,200 | 0,207 | 0,222 | 0,238 | 0,242 | 0,243 | 0,252 | 0,256 | 0,266 | 0,268 | **0,272** | 0,269 | 0,270 |
159: | mAP50 | 0,230 | 0,280 | 0,314 | 0,323 | 0,346 | 0,367 | 0,374 | 0,375 | 0,388 | 0,395 | 0,413 | 0,411 | 0,416 | 0,410 | 0,411 |
160: 
```

```text
174: **Checkpoint và tài nguyên**:
175: - `weights/best.pt` sha256 `d7d1c7e86d03c69d913febe79e4cbb6c2e1bcf9f17f034724af4de2ccfba7923`.
176: - `weights/last.pt` sha256 `3dbd9fb00208e99471513b45ee7f9ff6149633f0f14597d67ecbc52db49c26f4`.
177: - Mỗi file 5.445.075 B; sha256 tính lại trùng giá trị runner ghi. 15 bản sao `khoi_phuc` khớp nhật ký.
178: - Thời gian runner 108.820 s (30,2 h). Một epoch: 7.180–7.226 s train + 52–56 s val.
179: - VRAM tối đa 3.123 MiB.
180: - RSS runner + tiến trình con tối đa 4.200 MiB, trung vị 1.074 MiB.
181: - RAM máy trống thấp nhất 81 MiB, trung vị 951 MiB; 107.295 mẫu, 0 mẫu không đọc được.
182: 
183: **Thời gian cả hai model**: thời gian runner tổng 59,0 h. Con số này nằm giữa kịch bản 56,64 h và 73,95 h, và gồm khoảng
184: 3 h epoch dở bị bỏ của YOLOv8n. Thời gian đồng hồ từ lúc khởi chạy đầu tới lúc kết thúc là khoảng 68,5 h, vì có các
185: khoảng tạm dừng.
```

```text
209:   - Mỗi model chỉ một lượt (một seed). Chưa kiểm định thống kê.
210:   - Chênh lệch giữa hai model trên dev (0,280 / 0,272 mAP50-95) **chưa đủ để kết luận** model nào tốt hơn.
211: - **YOLOv8n qua 2 lần resume**: không trùng từng bit với lượt liên tục (RNG, thứ tự dữ liệu, gradient tích lũy dở;
212:   giới hạn đã công khai từ vòng 39). Phần epoch 8 (95 %) và epoch 12 (73 %) bị bỏ rồi chạy lại.
213: - **NaN**: nguyên nhân chưa xác định. Chưa kiểm cơ chế tự phục hồi NaN của Ultralytics, vì đã dừng trước khi epoch kết
214:   thúc. Nếu cơ chế đó chạy, runner sẽ coi lượt là lỗi (exit 3, `hoan_tat`), nên resume bị chặn.
215: - **Thời gian epoch** chịu ảnh hưởng của tải khác trên máy và các lần ngủ; không dùng làm số đo tốc độ chuẩn.
216: - **Ảnh chụp theo dõi** của trợ lý trống hai lần khi lượt YOLO11n đang chạy:
217:   - 10-05 23:49 → 10-06 01:08 (79 phút);
218:   - 10-06 11:53 → 14:22 (149 phút).
219: 
220:   Với YOLOv8n, ảnh chụp chỉ trống trong lúc lượt đã dừng. Log của bộ giám sát (lấy mẫu mỗi giây) liên tục ở cả hai
221:   model.
222: - **Kết quả runner theo từng lần**: file JSON của lần cuối chỉ chứa các epoch của lần đó. Nhật ký lượt chứa đủ 15 epoch.
```

## handoff/logs/review42/11_dung_luot_nan_yolov8n.log

SHA256 `c4b512962a721ccd711a710e7621c049b34eb0004d998a81e9e09a3ac5840214`

```text
1: # 2026-10-04 23:29:10 +0700 loss epoch 12 (lan 1) thanh NaN tu batch 4597/6311 (~23:20:40, +31321 s); nguoi dung hoi 'co tam dung duoc khong' -> chon 'Dung ngay, resume sau'. Lenh: python handoff/logs/review42/dung_luot.py --giam-sat results/phase2/detector_train/giam_sat/yolov8n_tiep_tuc_1 --cho-s 120
2: # 7 dong log quanh diem NaN:
3: [+  31313.2s] [K      12/15      2.11G      1.059     0.6719      0.948        171        640: 72% ━━━━━━━━╸─── 4594/6311 2.7s/it 1:21:11<1:17:02
4: [+  31316.0s] [K      12/15      2.11G      1.059     0.6719      0.948        102        640: 72% ━━━━━━━━╸─── 4595/6311 2.7s/it 1:21:13<1:16:56
5: [+  31318.7s] [K      12/15      2.11G      1.059      0.672      0.948        136        640: 72% ━━━━━━━━╸─── 4596/6311 2.7s/it 1:21:16<1:17:37
6: [+  31321.4s] [K      12/15      2.11G        nan        nan        nan        117        640: 72% ━━━━━━━━╸─── 4597/6311 2.7s/it 1:21:19<1:17:14
7: [+  31324.1s] [K      12/15      2.11G        nan        nan        nan         96        640: 72% ━━━━━━━━╸─── 4598/6311 2.7s/it 1:21:21<1:17:00
8: [+  31326.8s] [K      12/15      2.11G        nan        nan        nan        141        640: 72% ━━━━━━━━╸─── 4599/6311 2.7s/it 1:21:24<1:17:01
9: [+  31329.5s] [K      12/15      2.11G        nan        nan        nan         76        640: 72% ━━━━━━━━╸─── 4600/6311 2.7s/it 1:21:27<1:16:58
10: # dong cuoi truoc khi dung:
11: [+  31830.4s] [K      12/15      2.11G        nan        nan        nan        125        640: 75% ━━━━━━━━━─── 4786/6311 2.7s/it 1:29:48<1:08:32
12: 2026-10-04T23:29:11+07:00 truoc khi dung: runner 4388 song=True, cong cu 12784 song=True
13: 2026-10-04T23:29:11+07:00 CTRL_C_EVENT gui=True (loi 187) vao console cua cong cu 12784
14: 2026-10-04T23:29:19+07:00 runner da thoat sau 8 s sau Ctrl+C
15: 2026-10-04T23:29:19+07:00 cach dung: ctrl_c; runner song=False, cong cu song=False; trang_thai.json: ket_thuc exit_code=130
16: # exit=0
17: # --- kiem sau khi dung (23:30, CHI DOC) ---
18: # runner: [BI NGAT] KeyboardInterrupt, ket qua do dang results/phase2/detector_train/train_bdd5_v1_yolov8n_day_du__tiep_tuc_1.json; bo giam sat exit 130, ghi ctrl_c (t+31831,4 s).
19: # nhat ky: ket_thuc lan 1 bi_ngat exit 130, epoch_da_luu_lan_nay [8, 9, 10, 11]; epoch da luu 1..11; 11 ban sao khoi_phuc khop sha256.
20: # weights/last.pt = epoch 11 (sha khop nhat ky); best.pt = epoch 11 (3ebfa4b6...), best_fitness 0,27515.
21: # checkpoint epoch 11 nap tren CPU: 0/355 tensor EMA va 0/183 momentum KHONG huu han -> nguon resume sach.
22: # theo_batch.csv 69.421 hang (11 x 6311), 0 hang loss_huu_han False (hang epoch 12 chua ghi vi epoch chua xong); results.csv epoch 1..11.
23: # khong con tien trinh train_detector.
24: # toc do epoch 12: ~1,0 it/s toi batch 4400 (23:12); 4400 -> 4591 cham ~2,6 s/it; NaN tu batch 4597 (23:20:40).
25: # RAM trong 23:07-23:21 on dinh 590-690 MiB (ram_mau_tho.csv); log System 14:30-23:22: khong co nvlddmkm / TDR 4101 / WHEA.
26: # Nguyen nhan NaN CHUA XAC DINH (chi co quan sat: cham truoc NaN; khong thay loi driver / RAM can luc do).
```

## handoff/logs/review42/ket_qua/yolov8n/luot/results.csv

SHA256 `7ad73dc752e6e0df584e27c631998032695349aefe00fa5faf42ad7cf367b922`

```text
1: epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
2: 1,6015.89,1.32592,1.14774,1.06571,0.30773,0.25307,0.233,0.14303,1.26245,0.99706,1.04319,0.00999842,0.00999842,0.0100143
3: 2,12028.1,1.28202,0.95929,1.04198,0.38704,0.30673,0.2817,0.17648,1.17891,0.85313,1.00135,0.00934,0.00934,0.00934
4: 3,17986,1.21334,0.86446,1.01422,0.45586,0.36482,0.34335,0.22021,1.13601,0.78597,0.98034,0.00868,0.00868,0.00868
5: 4,23933.9,1.17697,0.81806,0.99732,0.47792,0.37926,0.35046,0.22438,1.11348,0.76074,0.9724,0.00802,0.00802,0.00802
6: 5,29879.4,1.15225,0.78587,0.98805,0.48572,0.38447,0.37536,0.24005,1.09501,0.72756,0.96384,0.00736,0.00736,0.00736
7: 6,35815.6,1.13335,0.76315,0.97957,0.51429,0.3817,0.38774,0.25143,1.07996,0.7062,0.95579,0.0067,0.0067,0.0067
8: 7,41750.8,1.12128,0.74739,0.97348,0.49621,0.41097,0.39345,0.25177,1.06563,0.69383,0.94504,0.00604,0.00604,0.00604
9: 8,5922.21,1.10447,0.72887,0.96734,0.48946,0.41181,0.39312,0.25317,1.06179,0.68283,0.94735,0.00538,0.00538,0.00538
10: 9,12503,1.09493,0.71615,0.96316,0.5456,0.40521,0.41216,0.26617,1.04942,0.66867,0.94333,0.00472,0.00472,0.00472
11: 10,18922.9,1.08156,0.70023,0.95747,0.52327,0.41098,0.41303,0.26582,1.04678,0.66039,0.93877,0.00406,0.00406,0.00406
12: 11,26210.9,1.069,0.68669,0.95202,0.55098,0.43288,0.42801,0.27515,1.04423,0.65755,0.93725,0.0034,0.0034,0.0034
13: 12,6083.57,1.04983,0.66492,0.94526,0.57314,0.41967,0.42666,0.27471,1.03971,0.64967,0.93643,0.00274,0.00274,0.00274
14: 13,12167.1,1.04174,0.6544,0.94121,0.55741,0.43255,0.43085,0.27843,1.03866,0.64443,0.93439,0.00208,0.00208,0.00208
15: 14,18241.7,1.04504,0.63606,0.93861,0.58621,0.41813,0.43241,0.27807,1.03726,0.64778,0.93314,0.00142,0.00142,0.00142
16: 15,24302.4,1.03048,0.61855,0.93309,0.60361,0.42027,0.43506,0.28015,1.03731,0.64572,0.93164,0.00076,0.00076,0.00076
```

## handoff/logs/review42/ket_qua/yolo11n/luot/results.csv

SHA256 `30a37b1c96cb96b4908195d2c8df45977ee21c64c425db5b054a93ae5172e96f`

```text
1: epoch,time,train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2
2: 1,7258.77,1.34556,1.18549,1.09075,0.35006,0.21527,0.22947,0.13996,1.29674,1.02564,1.0607,0.00999842,0.00999842,0.0100143
3: 2,14499.2,1.3022,0.99311,1.04958,0.40128,0.28362,0.28003,0.18057,1.19876,0.87381,1.00428,0.00934,0.00934,0.00934
4: 3,21746.6,1.22809,0.89137,1.01614,0.4226,0.32509,0.3135,0.19964,1.14785,0.80838,0.98244,0.00868,0.00868,0.00868
5: 4,28990.6,1.18925,0.84221,0.99842,0.44813,0.32654,0.3231,0.20659,1.11515,0.7684,0.96695,0.00802,0.00802,0.00802
6: 5,36254.7,1.16342,0.80778,0.98905,0.47563,0.34883,0.3457,0.2215,1.10953,0.75114,0.96364,0.00736,0.00736,0.00736
7: 6,43488.6,1.14511,0.78591,0.97998,0.50167,0.3586,0.36742,0.23821,1.08634,0.71951,0.95017,0.0067,0.0067,0.0067
8: 7,50768.9,1.13133,0.76717,0.97381,0.49732,0.3802,0.37348,0.24204,1.07146,0.70898,0.948,0.00604,0.00604,0.00604
9: 8,58021.9,1.11556,0.75019,0.96806,0.4865,0.3772,0.37493,0.24258,1.0646,0.6978,0.94444,0.00538,0.00538,0.00538
10: 9,65254.3,1.10642,0.73805,0.96436,0.52694,0.37844,0.38805,0.25195,1.05644,0.683,0.93879,0.00472,0.00472,0.00472
11: 10,72505.6,1.09385,0.72255,0.95878,0.51737,0.38778,0.39455,0.2564,1.0483,0.67475,0.9367,0.00406,0.00406,0.00406
12: 11,79746.4,1.08236,0.70852,0.9539,0.55387,0.38274,0.41316,0.26612,1.04405,0.66493,0.93318,0.0034,0.0034,0.0034
13: 12,86985.4,1.07012,0.6937,0.94894,0.57155,0.37607,0.41111,0.26786,1.0394,0.66223,0.93018,0.00274,0.00274,0.00274
14: 13,94238.2,1.05998,0.68214,0.94615,0.57543,0.38303,0.41628,0.27225,1.03508,0.65511,0.92973,0.00208,0.00208,0.00208
15: 14,101494,1.05091,0.65108,0.93832,0.54227,0.39815,0.41041,0.26874,1.03862,0.6562,0.93059,0.00142,0.00142,0.00142
16: 15,108733,1.03719,0.63496,0.93329,0.53691,0.40441,0.4114,0.27014,1.03618,0.65412,0.92852,0.00076,0.00076,0.00076
```

## handoff/logs/review42/ket_qua/yolo11n/giam_sat/yolo11n_lan0/tom_tat.json

SHA256 `0513bb55e0ed29e1a19dc1e0b5ed5ae73296a6f18fd16586d327e979815d77e3`

```text
1: {
2:  "lenh": [
3:   "C:\\Users\\Mikuno\\anaconda3\\envs\\khoaluan-mot\\python.exe",
4:   "-u",
5:   "src\\train_detector.py",
6:   "--model",
7:   "yolo11n",
8:   "--che-do",
9:   "day-du",
10:   "--xac-nhan-train-dai"
11:  ],
12:  "pid": 4216,
13:  "create_time": 1791167961.843848,
14:  "exit_code": 0,
15:  "exit_code_hex": "0x00000000",
16:  "bat_dau": "2026-10-05T09:39:21+07:00",
17:  "ket_thuc": "2026-10-06T15:55:26+07:00",
18:  "thoi_gian_s": 108964.7,
19:  "so_dong_log": 100351,
20:  "ctrl_c": [],
21:  "loi_in_man_hinh": 0,
22:  "chu_ky_s": 1.0,
23:  "ly_do_dung_lay_mau": "dung theo yeu cau (tien trinh chinh da thoat)",
24:  "ram": {
25:   "so_mau": 107295,
26:   "so_mau_khong_doc_duoc": 0,
27:   "tu_t": 1791167961.871,
28:   "den_t": 1791276925.741,
29:   "khoang_do_s": 108963.9,
30:   "pid_trong_csv": [
31:    4216
32:   ],
33:   "rss_chinh_mib": {
34:    "n": 107295,
35:    "min": 7.9,
36:    "trung_vi": 552.0,
37:    "max": 1458.6
38:   },
39:   "rss_tong_mib": {
40:    "n": 107295,
41:    "min": 7.9,
42:    "trung_vi": 1073.5,
43:    "max": 4200.3
44:   },
45:   "so_con": {
46:    "n": 107295,
47:    "min": 0.0,
48:    "trung_vi": 6.0,
49:    "max": 18.0
50:   },
51:   "ram_he_thong_con_trong_mib": {
52:    "n": 107295,
53:    "min": 81.0,
54:    "trung_vi": 951.0,
55:    "max": 2419.0
56:   },
57:   "so_mau_co_ghi_chu": 0,
58:   "ghi_chu_cot": "RSS = working set Windows; rss_tong = chinh + moi tien trinh con (dem trung trang chia se -> co the cao hon bo nho thuc dung); ram_he_thong_con_trong = psutil.virtual_memory().available (toan may); KHONG gom VRAM"
59:  }
60: }
```

## handoff/logs/review42/ket_qua/yolo11n/runner/train_bdd5_v1_yolo11n_day_du.json

SHA256 `cd39ee9789fa4ccbd94686209775d86e96d81825e9f652cce7a60afe18e8c8a5`

```text
1174:   "ghi_chu": "ultralytics final_eval: validate best.pt (FP16 tu dia) sau vong lap; gom ca luu checkpoint va don dep cuoi epoch"
1175:  },
1176:  "thoi_gian_s": 108820.3,
1177:  "vram_nvidia_smi": {
1178:   "so_mau": 95509,
1179:   "max_mib": 3123,
1180:   "chu_ky_s": 1.0,
1181:   "pham_vi": "nvidia-smi memory.used cua ca GPU (gom desktop + tien trinh khac)",
1182:   "ram_con_trong_min_mib": 6,
1183:   "gpu_util_trung_vi_pct": 96,
1184:   "gpu_util_ghi_chu": "utilization.gpu cua nvidia-smi, lay mau suot luot (gom khoi dong + val)"
1185:  },
1186:  "loi": null,
1187:  "save_dir": "D:\\LABOTARY\\bytetrack\\models\\finetune_phase2\\train_bdd5_v1_yolo11n_day_du",
```

## handoff/review_evidence/CODEX_6c8e37a/tree/src/train_detector.py

SHA256 `250302fb045a3b21dea575067700cdfbd7796b125e4ecfd2e528c91df0731544`

```text
356:     def run(self):
357:         try:
358:             import psutil
359:         except ImportError:
360:             psutil = None
361:         while not self.dung.is_set():
362:             t, mem, util, ram = time.time(), None, None, None
363:             try:
364:                 r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
365:                                     "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
366:                 mem, util = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
367:                 self.mau.append(mem)
368:                 self.gpu.append(util)
369:             except (OSError, ValueError, IndexError, subprocess.SubprocessError):
370:                 pass
371:             if psutil is not None:
372:                 ram = psutil.virtual_memory().available / 2 ** 20
373:                 self.ram.append(ram)
374:             self.tho.append((round(t, 3), mem, util, round(ram) if ram is not None else None))
375:             self.dung.wait(self.chu_ky)
376: 
377:     def ghi_tho(self, path: Path) -> None:
378:         with open(path, "w", newline="", encoding="utf-8") as f:
379:             w = csv.writer(f)
380:             w.writerow(["t_unix", "nvidia_smi_memory_used_mib_ca_gpu", "gpu_util_pct", "ram_he_thong_con_trong_mib"])
381:             w.writerows(self.tho)
382: 
383:     def ket_qua(self) -> dict:
384:         return {"so_mau": len(self.mau), "max_mib": max(self.mau) if self.mau else None,
385:                 "chu_ky_s": self.chu_ky, "pham_vi": "nvidia-smi memory.used cua ca GPU (gom desktop + tien trinh khac)",
386:                 "ram_con_trong_min_mib": round(min(self.ram)) if self.ram else None,
387:                 "gpu_util_trung_vi_pct": sorted(self.gpu)[len(self.gpu) // 2] if self.gpu else None,
388:                 "gpu_util_ghi_chu": "utilization.gpu cua nvidia-smi, lay mau suot luot (gom khoi dong + val)"}
```
