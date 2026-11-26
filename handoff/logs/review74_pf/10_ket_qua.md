# R74 — PF trên BDD và UA-DETRAC (03/11/2026)

## Phạm vi và giao thức

Theo yêu cầu người dùng, vòng này chỉ xem **BDD100K COCO-video val** và **UA-DETRAC train dev12**. UA-DETRAC dùng kết quả R72 đã hoàn tất; không chạy lại detector hay tracker trên tập đó. BDD chạy mới trên 200 video, 39.973 frame của split `BDDCOCOVID-val`, dùng cache YOLOv8n COCO R61. Cả hai mức PF500-K1K2 roughening chuyển động 1,0/2,0 dùng cùng cache, confidence >0,25, các lớp COCO [1,2,3,5,7], max-det 300, FPS 5, buffer 6 frame (1,2 s), `elapsed_mode=steps`, ByteTrack association giữ nguyên và seed 0/1/2. BDD được chấm bằng `mot_crowd` để xử lý GT ignore/crowd đúng quy tắc R61. Giao thức khóa trước ở `00_protocol.json`.

## Cổng kiểm

- Manifest, checkpoint, cache meta và từng NPZ/JSON cache, GT, seqinfo của 200 video khớp SHA256 marker R61 trước và sau chạy; tổng frame đúng 39.973. Chỉ đọc cache R61 mới, không dùng cache BDD cũ.
- Smoke 3 video × 2 mức × 3 seed = 18/18 lượt chạy và chấm; smoke không dùng chọn mức.
- Full 200 video × 2 mức × 3 seed = 1.200/1.200 lượt tracking, sáu run có đủ file MOT, marker hoàn tất, seed và input fingerprint khớp run metadata. Sáu run được chấm đủ 200 video. TrackEval COMBINED khớp khi tái tính từ raw fields (sai số tối đa 8,33e-17). CV/EKF R61 tham chiếu cũng qua kiểm hash artifact và tái tổ hợp điểm.
- Bộ chấm chạy song song với tracking đã đến lượt seed 2 của mức 2,0 trước khi lượt ấy hoàn tất; lượt chấm đó bị từ chối vì thiếu file. Sau khi tracking đủ 200 video, chấm lại thành công. Không sử dụng điểm của lượt chấm lỗi.

## Kết quả

HOTA là COMBINED theo toàn bộ video, sau đó lấy trung bình ba seed PF. Chênh lệch là mức 2,0 trừ mức 1,0 trong cùng tập.

| Tập | CV | EKF | PF 1,0 | PF 2,0 | Chênh PF 2,0 − 1,0 |
|---|---:|---:|---:|---:|---:|
| BDD val, HOTA | 33,677% | 33,827% | 32,643% | 32,951% | **+0,307 điểm** |
| UA-DETRAC dev12, HOTA (R72/R70) | 59,000% | 59,252% | 56,600% | 57,006% | **+0,406 điểm** |

Trên **BDD**, cả ba seed tăng HOTA: +0,267 / +0,290 / +0,365 điểm phần trăm. Trung bình theo seed trên từng video: 149 video tăng, 50 giảm, 1 bằng nhau. Bootstrap ghép cặp 200 video, 5.000 mẫu, cùng chỉ số lấy mẫu cho mọi seed, tái tổ hợp COMBINED từng seed rồi lấy trung bình: chênh HOTA **+0,307 điểm**, CI phần vị 95% **[+0,209; +0,404]**. IDF1 tăng 36,660% → 37,086% (+0,426 điểm; CI [+0,283; +0,569]); IDSW trung bình giảm **6.670 → 6.291,3** (−378,7; CI [−473,0; −285,7]). DetA tăng 0,067 điểm, AssA tăng 0,666 điểm. PF 2,0 vẫn thấp hơn CV **0,726 điểm** và EKF **0,876 điểm HOTA**.

Trên **UA-DETRAC dev12** (R72), chênh HOTA +0,406 điểm có CI 95% **[−0,051; +1,052]**, đi qua 0. IDSW trung bình giảm 65,3. Đây là kết quả trên tập dev đã dùng để phát triển PF, không phải xác nhận độc lập.

## Diễn giải

Theo đúng giao thức BDD này, mức roughening chuyển động 2,0 cải thiện PF so với 1,0 và giảm IDSW; ước lượng và khoảng bootstrap của BDD cùng dấu. UA-DETRAC cũng tăng theo điểm ước lượng nhưng khoảng HOTA đi qua 0. Không gộp HOTA của hai tập vì GT, evaluator và mật độ cảnh khác nhau. BDD val đã được dự án quan sát từ R61, còn UA là tập dev; khoảng bootstrap BDD mô tả biến thiên khi lấy lại video trong tập hiện có, giả định video độc lập chưa được chứng minh và không thay cho kiểm chứng mù trên tập mới. Các kết quả ngoài hai tập ở R73 cũng không nhất quán, nên không suy ra 2,0 là lựa chọn tốt cho mọi cảnh rẽ hoặc che khuất.

**Giữ mặc định roughening 1,0 và cấu hình luận văn đã khóa.** Mức 2,0 là ứng viên PF có lợi trên BDD theo giao thức này, nhưng PF vẫn thấp hơn CV/EKF trên cả hai tập. Không thay thuật toán, association, metric hay cấu hình benchmark cũ.

## Tệp tái kiểm

- `results/phase2/r74_pf_bdd/preflight.json`, `per_seed.csv`, `per_video.csv`, `summary_bdd.csv`, `paired_intervals_bdd.csv`, `two_dataset_comparison.json`.
- Điểm và raw fields từng run: `results/trackeval/BDDCOCOVID-val/r74-bdd-full-motion*-s*/`.
- Tracking từng run: `data/processed/trackers/BDDCOCOVID-val/r74-bdd-full-motion*-s*/`.
- Lệnh phát lại/chấm/hậu kiểm: `src/r74_pf_bdd.py`, `src/r74_pf_bdd_eval.py`, `src/r74_pf_bdd_analyze.py`.
- Nguồn UA-DETRAC: `results/phase2/kc48_r72_pf_dev/analysis/summary.csv`, `paired_intervals.csv`, `validation.json`; CV/EKF UA từ R70.
