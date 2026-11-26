# R77 — giới hạn của tăng roughening chuyển động PF (04/11/2026)

Để đưa IDSW của PF gần CV/EKF, R77 thử hai mức roughening chuyển động **6,0 và 8,0** đã khóa trước, so với mức 2,0 của R72. PF500-K1K2, ba seed, YOLOv8n COCO c25, cùng cache, GT, evaluator và ByteTrack association trên UA-DETRAC train dev12. Mức >2 chỉ mở qua `pf_allow_high_motion_roughening: true`; mặc định và cấu hình cũ giữ nguyên. Giao thức `00_protocol.json` quy định sau hai mức này dừng tìm kiếm scalar và chỉ mở BDD nếu IDSW UA giảm ít nhất 10% mà HOTA không giảm hơn 0,5 điểm.

Smoke 2 video × 2 mức × 3 seed đạt 12/12 lượt video. Full 12 video × 2 mức × 3 seed đạt **72/72** lượt tracking và **6/6** run chấm. Hậu kiểm marker, fingerprint mã trong các run mới, raw-field COMBINED và dấu vân tay cache/GT/evaluator so với R72 đạt. Kiểm thử giới hạn opt-in và hồi quy R72 đạt 4/4.

| Mức roughening | IDSW TB 3 seed | HOTA | IDF1 |
|---:|---:|---:|---:|
| 2,0 (R72) | 697,0 | 57,006% | 68,361% |
| 6,0 | 636,7 | 57,138% | 68,777% |
| 8,0 | 649,7 | 57,203% | 68,661% |

So với mức 2,0, mức 6,0 giảm **60,3 IDSW (8,7%)**, CI bootstrap ghép cặp 12 video, 5.000 mẫu: **[−139,3; +2,7]**. Mức 8,0 giảm **47,3 (6,8%)**, CI **[−131,7; +20,3]**. Cả hai CI đi qua 0. HOTA mức 6,0 tăng 0,133 điểm (CI [−0,236; +0,534]); mức 8,0 tăng 0,197 điểm (CI [−0,041; +0,575]); cả hai CI cũng đi qua 0. Theo seed, IDSW 2,0 là 689/702/700; mức 6,0 là 646/605/659; mức 8,0 là 650/653/646.

**Quyết định:** cả hai mức đều trượt cổng giảm IDSW ≥10%, nên không chạy BDD R77. Mức 6,0 có IDSW thấp nhất trong lượt này nhưng vẫn cao hơn EKF UA (472) khoảng 165 lần đổi ID, và chưa có kiểm BDD. Không chọn nó làm bản PF mới hoặc thay mặc định. Dừng tìm kiếm mức roughening scalar theo giao thức; kết quả này là tìm kiếm thăm dò trên tập dev đã quan sát, không chứng minh hiệu quả tổng quát.

Bằng chứng: `results/phase2/kc48_r77_pf_dev/analysis/{per_seed,summary,paired_intervals}.csv`, `results/phase2/kc48_r77_pf_dev/eval.log`, `results/phase2/kc48_r77_pf_smoke/full_track.log`, mã `src/r77_pf/analyze.py`, cấu hình `configs/phase2/trackers/r77/`.
