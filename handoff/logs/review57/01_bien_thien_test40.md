# R57A — Ước lượng biến thiên test40: bootstrap ghép cặp theo video (23/10/2026)

Trạng thái: **ĐÃ SỬA 3 FINDING của Codex review R57A vòng 1 (verdict revise), CHỜ Codex review lại.** Chưa được Codex
approve. Chưa commit. Phiên hiện hành: **`results/phase2/r57_bien_thien/phien_20261023_011207_108057_pid8500`** (giao
thức phiên bản 2). Phiên cũ `phien_20261023_004544_759728_pid25976` (giao thức phiên bản 1) giữ nguyên. Thay đổi và lý
do: mục 8. Số liệu mục 4–5 không đổi, vì các file kết quả trùng byte giữa hai phiên.
Phạm vi: chỉ phân tích hậu nghiệm từ metric đã lưu của phiên thực thi đã nghiệm thu
(`results/phase2/r55_tracking/thuc_thi/phien_20261021_161832_735800_pid11640/`, Codex nghiệm thu `lan_013` approve).
Không chạy tracking, evaluator hay GPU, không tạo cache. Không so detector, không chọn conf, bộ lọc hoặc seed.

## 1. Giao thức (cố định trước khi tính CI, sau khi đã xem điểm test)

`configs/phase2/r57_bien_thien/giao_thuc.json`, phiên bản 2, SHA256
`987986a58c6b175d246229853524d5afb56b3905061805ffdef0802393f2b4ed` (bản bất biến trong phiên
`.../phien_20261023_011207_108057_pid8500/giao_thuc.json`, cùng hash). Phiên bản 1 (`b01eee60…288438b9e`) vẫn còn
nguyên trong phiên cũ. Phiên bản 2 chỉ đổi phần `dung_sai.bang_r56`, thêm `thay_doi` và sửa câu `vai_tro` của bảng R56.
Bootstrap, metric, hiệu số, nhóm và tiêu chí tái tổ hợp giữ như phiên bản 1.

- Giao thức được cố định **sau khi** đã xem điểm combined test40 (R56). Đây **không phải** tiền đăng ký trước thí nghiệm.
- Đầu vào: 36 lần chấm nêu đích danh trong `results/phase2/r56_tong_hop/phien_20261021_161832_735800_pid11640/nguon.json`.
  Mỗi lần chấm có `raw_fields.json`, `per_video_metrics.csv`, `combined_metrics.csv` và marker `HOAN_TAT.json` ở gốc thư
  mục chấm. Khóa R56 `khoa_20261021_155510_440500_pid8768` (`ce85628e…9d35247b`), giao thức R54 `eb1f8f30…715431ad`,
  hợp đồng R55 `488238d4…4239cf95d`, manifest `detrac_test.json` `ff2b4b9c…3077f47c`.
- 40 video theo đúng thứ tự `videos` của manifest đã khóa. 3 detector × {c25, c10} × {CV, EKF, UKF, PF500 seed 0/1/2}
  = 36 run.
- Metric: **HOTA là chỉ số chính**, IDF1 và IDSW là bổ sung. HOTA/IDF1 tính trên thang 0–1 và trình bày ×100 thành
  **điểm phần trăm (pp)**. IDSW là **số đếm**, bằng tổng trên các video của mẫu; với PF500 là trung bình 3 seed nên có
  thể lẻ.
- Hiệu số có hướng a − b trong từng nhóm detector × điều kiện: EKF−CV, UKF−CV, PF500−CV, UKF−EKF, PF500−EKF,
  PF500−UKF. Tổng cộng 6 nhóm × 6 hiệu số × 3 metric = **108 dòng** cho mỗi thiết kế.
- Bootstrap: B = 5000. Mỗi thiết kế dùng một generator riêng `numpy.random.default_rng(0)`; mẫu i rút
  `rng.integers(0, K, size=K)`, cùng dãy số với `boot_stats.paired_bootstrap` / `boot_seed.paired_bootstrap_seed`.
  CI là phân vị 2,5 / 97,5 với `method='linear'`.
- **Thiết kế video**: rút 40 video có hoàn lại (K = 40).
- **Thiết kế nhóm (kiểm độ nhạy)**: 39 nhóm khai báo tường minh trong giao thức, gồm `N22 = {MVI_40773, MVI_40775}` và
  38 video đơn. Rút 39 nhóm có hoàn lại; mỗi lần một nhóm xuất hiện thì đưa toàn bộ video của nhóm vào mẫu, không cắt
  hay đệm về 40. Số video mỗi mẫu là 39–44 (trung bình 39,998). Nhóm không suy ra từ tên video hay metric. Nguồn của
  cặp: `handoff/logs/review48/03_audit_du_lieu.md` mục 6.4 (cùng camera, cùng cảnh đêm).

## 2. Công thức

- Tái tổ hợp: `boot_stats._stack` + `boot_stats.combine_stacked` (công thức COMBINED_SEQ của TrackEval). HOTA lấy trung
  bình 19 alpha của sqrt(DetA·AssA), với DetA = ΣTP/(ΣTP+ΣFN+ΣFP) và AssA có trọng số HOTA_TP. IDF1 =
  ΣIDTP/(ΣIDTP + 0,5ΣIDFN + 0,5ΣIDFP). IDSW = ΣIDSW. Video lặp lại trong mẫu được cộng đúng số lần lặp. **Không** dùng
  `boot_stats.average_seeds`.
- Trong mỗi mẫu, mỗi run có COMBINED riêng. Với PF500, tính COMBINED riêng của từng seed 0/1/2 rồi lấy trung bình bằng
  `boot_seed._tb`; sau đó mới tính hiệu số. Mỗi thiết kế có **một** bộ chỉ số mẫu, dùng chung cho mọi run và mọi seed
  của cả 6 nhóm (ghép cặp). Ba seed PF được giữ cố định, không lấy mẫu lại seed, và không coi 120 tổ hợp video × seed là
  quan sát độc lập.
- Điểm ước lượng là COMBINED trên toàn bộ 40 video. Thiết kế nhóm cho cùng điểm ước lượng vì mỗi video xuất hiện đúng
  một lần.

## 3. Lệnh, exit code và bằng chứng

Ba lệnh chạy foreground, tuần tự, mỗi lệnh là một lời gọi riêng:

| # | Lệnh | Exit | Bằng chứng |
|:--|:--|:--|:--|
Lần chạy sau khi sửa (vòng 2), ba lệnh foreground, tuần tự, mỗi lệnh một lời gọi riêng:

| # | Lệnh | Exit | Bằng chứng |
|:--|:--|:--|:--|
| 1 | `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r57_bien_thien/test_phan_tich_test40.py` | **0**, 15/15 đạt | `results/phase2/r57_bien_thien/tests/lan_20261023_011135_199544_pid12284/ket_qua_test.json` |
| 2 | `… src/r57_bien_thien/phan_tich_test40.py --buoc chuan-bi --giao-thuc configs/phase2/r57_bien_thien/giao_thuc.json` | **0**; in `PHIEN phien_20261023_011207_108057_pid8500` | `results/phase2/r57_bien_thien/phien_20261023_011207_108057_pid8500/{giao_thuc.json, nguon.json, nhom_video.csv}` |
| 3 | `… src/r57_bien_thien/phan_tich_test40.py --buoc phan-tich --phien phien_20261023_011207_108057_pid8500` | **0** | cùng phiên: `chi_so_video.csv`, `chi_so_nhom.csv`, `diem_uoc_luong.csv`, `doi_chieu_video.csv`, `doi_chieu_nhom.csv`, `kiem_chung.json`, `SHA256SUMS` |

Lần chạy đầu (vòng 1, giữ nguyên): fixture `tests/lan_20261023_004525_846673_pid25692` (exit 0, 13/13), phiên
`phien_20261023_004544_759728_pid25976` (chuan-bi và phan-tich đều exit 0).

SHA256 của mã, theo đúng nơi được ghi:

| File | SHA256 | Ghi trong |
|:--|:--|:--|
| `phan_tich_test40.py` | `53e4933e3355448090452d1ec247344a50946f5d9df8d3a982def2656a73ee82` | `ket_qua_test.json`, `nguon.json`, `kiem_chung.json` |
| `boot_stats.py` | `e9245f6729659f8f8488bab3b54c136925a935a9d65ecb263dc176e5d13c7b72` | `ket_qua_test.json`, `nguon.json`, `kiem_chung.json` |
| `boot_seed.py` | `f415a49ec45433c4dca1860ca015eaa5c94f32af78aacfa2eef48b79e9bead84` | `ket_qua_test.json`, `nguon.json`, `kiem_chung.json` |
| `test_phan_tich_test40.py` | `93f35f0fbbc6220cbdda3bcab6b05d3e67232448038119ab5593de86f7444b27` | **chỉ** `ket_qua_test.json` |
| `giao_thuc.json` | `987986a58c6b175d246229853524d5afb56b3905061805ffdef0802393f2b4ed` | `ket_qua_test.json`; `kiem_chung.json` (`giao_thuc_sha256`); `SHA256SUMS` |

Ba hash mã đầu khớp nhau ở cả ba nơi. Mã không đổi giữa fixture, chuan-bi và phan-tich: hash `phan_tich_test40.py` trong
`ket_qua_test.json` trùng với hash trong `nguon.json` và `kiem_chung.json`. File fixture `test_phan_tich_test40.py` không
được phiên phân tích ghi lại. (Báo cáo vòng 1 ghi nhầm rằng hash này có ở cả ba nơi; xem mục 8, P3.)

SHA256 đầu ra (`SHA256SUMS` trong phiên mới). Cột cuối so với phiên cũ:

| File | SHA256 | So với phiên cũ |
|:--|:--|:--|
| `doi_chieu_video.csv` | `fbfd0dbddb705339a7f7c931a7c2ed70fea47b86360feb8ec56df6e20c0a55c7` | trùng |
| `doi_chieu_nhom.csv` | `ec543e8b95ddb7b13cf6c8db5ec98f2b179ce44542b6f22c7c0f6469249ea07f` | trùng |
| `diem_uoc_luong.csv` | `ab52a46ea724e2d25d7ba888708b100f51963a75487f6543d32032f6a20e21af` | trùng |
| `chi_so_video.csv` | `98525365bcedbb9abd730db332118cdd7df378f60acdafcf179a0aaf5aa4b834` | trùng |
| `chi_so_nhom.csv` | `c01d21e92c9772450f875355dcdb152699a9e683da1276b1c8fb1be918cbd802` | trùng |
| `nhom_video.csv` | `22dc2a062efb141f810ea9b39922601308cd7ee57f133084eb16da261f2535a4` | trùng |
| `giao_thuc.json` | `987986a58c6b175d246229853524d5afb56b3905061805ffdef0802393f2b4ed` | khác (phiên bản 2) |
| `nguon.json` | `1e7f019b55c28fff1781ae9fa5a5ec0e08aa9f524493464a1fec67601611b596` | khác (hash mã/giao thức, thời điểm) |
| `kiem_chung.json` | `fe850f62d89cbe299d9de576ebcc06f82c1543882624055c8e7266920ce82be8` | khác (thêm `sai_so_max`, dung sai mới) |

### 3.1 Fixture (15 ca, CPU, dữ liệu giả 6 video / 36 run / B = 200)

1. COMBINED khác trung bình theo video; khớp phép tính độc lập bằng vòng lặp Python (≤ 1e-12).
2. PF lấy trung bình COMBINED của từng seed đúng thứ tự phi tuyến. Kết quả khác `combine(average_seeds)` ở HOTA/IDF1 và
   trùng ở IDSW, vì IDSW là tổng tuyến tính.
3. Video lấy trùng được cộng đúng số lần lặp.
4. Giao thức thật: 39 nhóm, chỉ N22 có nhiều video. Trên 2000 mẫu, cặp luôn đi cùng nhau đúng số lần; độ dài mẫu thay
   đổi (không cắt/đệm).
5. Ghép cặp: IDSW video j được đặt bằng 10^j để mã hóa tập video. Mọi run/seed trong cùng mẫu nhận đúng cùng mã.
6. Đầu-cuối trên fixture: chuan-bi + phan-tich; 108/108 dòng mỗi thiết kế; nhánh video trùng
   `boot_seed.paired_bootstrap_seed`; nhánh nhóm khớp phép tính độc lập (sai số tối đa 4,4e-16); CI nhóm tính lại từ mẫu;
   tái tính từ chỉ số trên đĩa cho kết quả trùng; `SHA256SUMS` đúng; chạy phan-tich lần hai bị chặn ghi đè.
   Bảng R56 giả theo đúng quy tắc của `src/r56_tong_hop/tong_hop_test40.py`: summary làm tròn 3 chữ số, IDSW nguyên,
   PF `round(trung bình 3 seed, 4)`. Sai số lớn nhất ghi trong `kiem_chung` phải nằm trong dung sai khai báo.
7–9. Các trường hợp bị chặn: sai hash sau marker, raw thiếu video, NaN trong raw.
10. **(mới)** Hỏng schema nhưng hash nhất quán. Dữ liệu bị làm hỏng *trước* khi ghi và băm, nên marker và hash ghi lúc
    chuan-bi đều khớp; lỗi chỉ có thể đến từ bước kiểm schema. 14 ca đều bị chặn ở chuan-bi với đúng thông báo, không
    tạo phiên, và không ca nào báo "sai hash". Bảng R56 `tong_hop`: NaN, dòng trùng run, header trùng `HOTA`, dòng thừa
    cột, seed không khớp tên run. Bảng `pf500`: NaN ở `IDSW_tb`, dòng trùng nhóm, header trùng, `so_seed` = 2.
    `combined_metrics.csv`: header trùng có cùng giá trị (bộ đọc `csv.DictReader` cũ không phát hiện được), dòng thừa
    cột. `per_video_metrics.csv`: dòng trùng video, dòng thừa cột, NaN.
11. **(mới)** Ranh giới làm tròn của bảng R56, với dung sai đọc từ giao thức thật:
    - `tong_hop`: lệch HOTA ±0,0005 qua, 0,0006 bị chặn; IDF1 lệch 0,0006 bị chặn; IDSW lệch ±1 bị chặn.
    - `pf500`: lệch HOTA/IDF1 0,00055 qua, 0,0006 bị chặn. IDSW 3320,3333 so với 9961/3 qua (lệch 3,3e-5). IDSW 3320,3334
      (lệch 6,7e-5) và 3320,3332 bị chặn; mã cũ dùng `max(tb, 0,0005)` nên đã cho ca 3320,3334 qua.
    - Mô phỏng 20.000 lần quy tắc làm tròn R56: tỉ lệ lệch/dung sai lớn nhất là 0,901 (HOTA) và 0,667 (IDSW), tức luôn
      ≤ 1.
    - Đầu-cuối: bảng PF lệch +0,0001 IDSW (hash nhất quán). chuan-bi qua; phan-tich bị chặn với lỗi "lech bang R56" và
      không ghi đầu ra.
12–13. Nguồn thiếu run, run trùng (khóa JSON trùng); lệch liên kết cache.
14. Giao thức trong `configs` bị đổi sau chuan-bi thì phan-tich bị chặn và không ghi đầu ra.
15. Các đường dẫn bị chặn: ghi ngoài vùng, `..`, đọc bằng đường tuyệt đối hoặc `..`, tên phiên lạ, ghi đè, và
    **junction thật** (`_winapi.CreateJunction`).

### 3.2 Kiểm trên đầu vào thật (`kiem_chung.json`)

- **chuan-bi**: đọc đúng 36 lần chấm trong `nguon.json`; summary khớp hash nguồn và marker; marker khớp khóa, giao thức
  R54, hợp đồng R55, run và tập 40 video; 3 file metric khớp `file_metric` và `file` của marker. `raw_fields.json` có
  đúng 40 video và 19 alpha, mọi giá trị hữu hạn; `per_video` / `combined` đúng schema; combined CSV trùng
  `marker.combined`. Mỗi detector có đúng 1 manifest cache, khác nhau giữa các detector. PF có seed 0/1/2 ở cả 6 nhóm.
  Tổng cộng **188 file đầu vào** được băm.
- **phan-tich**: hash 188 file trước và sau phân tích đều 0 lệch; mã và giao thức trùng bản lúc chuan-bi.
- Tái tổ hợp 40 video so với `combined_metrics.csv` của 36 run: **sai số tối đa 0,0** cho HOTA, IDF1 và IDSW (dung sai
  1e-9 / 1e-9 / 0). So với `per_video_metrics.csv` (từng video) sai số cũng là 0,0.
- Bảng R56 (`tong_hop_36_run.csv`, `pf500_trung_binh.csv`) chỉ dùng để đối chiếu theo độ làm tròn và **không** là
  nguồn của bootstrap. Cả hai bảng được đọc bằng bộ đọc CSV chặt và kiểm schema, khóa duy nhất, tập run/nhóm, số hữu hạn
  ngay ở bước chuan-bi. Dung sai khai báo riêng cho từng bảng trong `giao_thuc.json` → `dung_sai.bang_r56`, cộng
  `eps_dau_phay_dong` = 1e-9. Mã dùng đúng các giá trị này, không nâng ngầm:

  | Bảng | HOTA (pp) | IDF1 (pp) | IDSW | Lý do | Sai số lớn nhất thực tế |
  |:--|:--|:--|:--|:--|:--|
  | `tong_hop_36_run` | 0,0005 | 0,0005 | 0 (chính xác) | summary 3 chữ số; IDSW nguyên | 0,000484 / 0,000499 / 0 |
  | `pf500_trung_binh` | 0,00055 | 0,00055 | 0,00005 | R56: `round(tb 3 giá trị summary, 4)` → 0,0005 + 0,00005; IDSW: trung bình số nguyên chính xác, chỉ làm tròn 4 chữ số | 0,0000909 / 0,000198 / 0,0000333 |

  Bảng PF ghi IDSW trung bình với bốn chữ số, nên **không** khớp chính xác: 4/6 nhóm lệch khoảng 3,33e-5, đúng như
  Codex chỉ ra. Báo cáo vòng 1 viết "IDSW khớp chính xác" là sai với bảng PF, và mã vòng 1 đã cho qua nhờ ngầm nâng
  dung sai lên 0,0005. Kết quả đối chiếu với cả hai bảng: đạt, `kiem_chung.json` → `bang_r56`. Tiêu chí tái tổ hợp so với
  `combined_metrics.csv` (1e-9 / 1e-9 / IDSW chính xác) không đổi.
- Nhánh video trùng `boot_seed.paired_bootstrap_seed` ở **108/108** dòng, so sánh chính xác (`==`) mảng mẫu, hiệu số
  và hai đầu CI.
- Chỉ số mẫu được ghi ra CSV trong bộ nhớ rồi đọc lại; đọc lại trùng và tái tính cho kết quả trùng (`==`). Mọi điều kiện
  này được kiểm trước khi ghi bất kỳ file đầu ra nào.

## 4. Chênh lệch và CI 95 % (HOTA, pp; thiết kế video | thiết kế nhóm)

Định dạng: chênh [CI thấp; CI cao]. Điểm ước lượng giống nhau ở hai thiết kế.

| Nhóm | EKF−CV | UKF−CV | PF500−CV | UKF−EKF | PF500−EKF | PF500−UKF |
|:--|:--|:--|:--|:--|:--|:--|
| yolov8n_coco c25 | +0,117 [−0,029; +0,266] \| [−0,027; +0,266] | +0,009 [−0,122; +0,141] \| [−0,123; +0,136] | −3,155 [−3,917; −2,512] \| [−3,938; −2,510] | −0,109 [−0,193; −0,019] \| [−0,192; −0,021] | −3,272 [−4,001; −2,658] \| [−4,000; −2,643] | −3,164 [−3,908; −2,532] \| [−3,926; −2,538] |
| yolov8n_coco c10 | +0,335 [+0,113; +0,576] \| [+0,111; +0,576] | +0,185 [−0,017; +0,413] \| [−0,012; +0,418] | −2,899 [−3,692; −2,181] \| [−3,711; −2,175] | −0,149 [−0,406; +0,070] \| [−0,408; +0,066] | −3,234 [−3,911; −2,623] \| [−3,943; −2,623] | −3,084 [−3,897; −2,345] \| [−3,907; −2,344] |
| yolov8n_mixed5 c25 | +0,068 [−0,134; +0,247] \| [−0,136; +0,247] | −0,014 [−0,243; +0,199] \| [−0,247; +0,195] | −3,098 [−3,977; −2,380] \| [−3,973; −2,369] | −0,081 [−0,231; +0,070] \| [−0,230; +0,074] | −3,166 [−3,969; −2,538] \| [−3,954; −2,521] | −3,084 [−3,874; −2,449] \| [−3,865; −2,436] |
| yolov8n_mixed5 c10 | +0,183 [−0,004; +0,390] \| [−0,014; +0,392] | +0,050 [−0,135; +0,237] \| [−0,140; +0,232] | −2,967 [−3,777; −2,284] \| [−3,750; −2,278] | −0,133 [−0,353; +0,072] \| [−0,353; +0,075] | −3,150 [−3,887; −2,552] \| [−3,874; −2,542] | −3,017 [−3,835; −2,307] \| [−3,814; −2,298] |
| yolo11n_mixed5 c25 | +0,176 [+0,007; +0,375] \| [+0,007; +0,371] | +0,119 [−0,101; +0,344] \| [−0,093; +0,336] | −3,252 [−4,046; −2,607] \| [−4,021; −2,601] | −0,058 [−0,225; +0,087] \| [−0,219; +0,083] | −3,428 [−4,203; −2,792] \| [−4,192; −2,795] | −3,370 [−4,138; −2,733] \| [−4,128; −2,736] |
| yolo11n_mixed5 c10 | +0,263 [−0,018; +0,576] \| [−0,020; +0,579] | +0,197 [−0,040; +0,470] \| [−0,048; +0,474] | −2,933 [−3,863; −2,101] \| [−3,864; −2,110] | −0,066 [−0,273; +0,105] \| [−0,263; +0,113] | −3,196 [−4,066; −2,446] \| [−4,066; −2,455] | −3,130 [−3,970; −2,413] \| [−3,950; −2,425] |

IDF1 (pp) và IDSW (số đếm) ở đủ 108 dòng mỗi thiết kế nằm trong `doi_chieu_video.csv` và `doi_chieu_nhom.csv`. Các cột
`chenh_hien_thi`, `ci_*_hien_thi` đã nhân 100 cho HOTA/IDF1. Tóm tắt thiết kế video:

- **PF500 so với CV/EKF/UKF** (18 dòng mỗi metric): HOTA từ −2,90 đến −3,43 pp; IDF1 từ −3,72 đến −4,62 pp; IDSW từ
  +719 đến +1.084. CI không chứa 0 ở cả 54 dòng (18 × 3 metric), ở cả hai thiết kế.
- **Giữa CV/EKF/UKF** (18 dòng mỗi metric): |chênh HOTA| ≤ 0,34 pp. CI không chứa 0 ở:
  - HOTA, 3/18 dòng: EKF−CV ở yolov8n_coco-c10 (+0,33) và yolo11n_mixed5-c25 (+0,18, cận dưới +0,007); UKF−EKF ở
    yolov8n_coco-c25 (−0,11).
  - IDF1, 2/18 dòng: EKF−CV ở yolov8n_coco-c10 +0,54 [+0,19; +0,91]; UKF−EKF ở yolov8n_mixed5-c25 −0,32 [−0,67; −0,008].
  - IDSW, 2/18 dòng: EKF−CV ở yolo11n_mixed5-c25 −89 [−180; −11]; UKF−EKF ở yolov8n_coco-c10 +42 [+6; +81].

  Các dòng còn lại có CI chứa 0. **CI chứa 0 không có nghĩa là tương đương**: dữ liệu này không xác định được chiều của
  chênh lệch ở mức đã nêu. Có 108 CI riêng lẻ chưa hiệu chỉnh đa so sánh, nên vài CI không chứa 0 có thể xuất hiện chỉ do
  ngẫu nhiên. Không tuyên bố bộ lọc nào thắng toàn cục, và không suy ra nhân quả.

## 5. Độ nhạy với cặp cùng cảnh

- Hai thiết kế **không khác phân loại CI** (âm / chứa 0 / dương) ở dòng nào: **0/108**
  (`kiem_chung.json` → `khac_dien_giai_hai_thiet_ke: []`).
- Tỉ lệ độ rộng CI (nhóm / video) là 0,954–1,060, trung bình 1,001. Như vậy gộp cặp `MVI_40773`/`MVI_40775` gần như không
  đổi độ rộng CI trên dữ liệu này.
- Giới hạn: thiết kế nhóm chỉ kiểm **một** cặp cùng cảnh đã biết. Kiểm chồng lấn ở R48 chỉ dùng 4 frame mẫu mỗi video,
  nên có thể còn cặp cùng cảnh hoặc cùng camera chưa phát hiện; 39 nhóm **không** phải 39 cảnh độc lập đã xác nhận.
  Kết quả này chỉ cho thấy CI không nhạy với cặp đã biết, chưa chứng minh các video độc lập.

## 6. Giới hạn

- CI 95 % bootstrap phân vị là khoảng riêng lẻ, mang tính thăm dò, chưa hiệu chỉnh đa so sánh. Không báo p-value.
- Phân tích giả định các video (hoặc các nhóm) độc lập và có thể hoán đổi. Tập 40 video không phải 40 bối cảnh độc lập
  (xem mục 5). Bootstrap bỏ qua biến thiên giữa các frame/track bên trong một video vì lấy mẫu theo toàn bộ video.
- PF: chỉ có 3 seed cố định và không lấy mẫu lại seed. CI phản ánh biến thiên giữa các video, không phản ánh biến thiên
  giữa các seed.
- Hai tập test đã được xem ở các vòng trước, và giao thức được cố định sau khi xem điểm.
- Chỉ băm các file nhỏ mà nguồn trỏ tới (marker, metric, summary, khóa/giao thức/hợp đồng/manifest, bảng R56, `phien.json`).
  Không băm lại ảnh, video, NPZ, MOT hay cache. **Đây không phải nghiệm thu lại toàn bộ R56.**
- Liên kết cache chỉ kiểm tính nhất quán `manifest_cache_sha256` trong marker (1 giá trị mỗi detector, khác nhau giữa các
  detector); không mở manifest cache.
- `src/__pycache__/boot_stats.cpython-311.pyc` có trên đĩa. Cả hai script đặt `sys.dont_write_bytecode = True` trước
  import, nên nhiều khả năng file này có từ trước; chưa kiểm mtime vì không có lệnh đọc mtime trong quyền được cấp.

## 7. Thứ tự việc sau

1. Codex review lại R57A: mã, fixture `lan_20261023_011135_199544_pid12284`, phiên
   `phien_20261023_011207_108057_pid8500` và báo cáo này. Ngân sách vòng do orchestrator quản lý. Mọi sửa mã/giao thức
   sau khi đã có kết quả sẽ tạo phiên mới và giữ nguyên phiên cũ.
2. Hoàn tất R57A thì **dừng**. SORT trên test40 và QJTF chưa làm, cần người dùng duyệt riêng; agent không tự chuyển
   sang. Hai đề xuất hiệu chỉnh đa so sánh cho 108 CI và kiểm cùng cảnh mở rộng đã bị người dùng **từ chối** qua email
   (Pf01dab37, P33464a9d), nên không làm.

## 8. Thay đổi sau Codex review R57A vòng 1 (verdict revise)

Nguồn finding: `D:/LABOTARY/automation/runs_local/bytetrack_r57_bien_thien_20261023_0029/calls/0002_review/answer.json`.

- **[P2] Đầu vào R56 không hợp lệ bị bỏ lọt.**
  - Vấn đề: dict comprehension gộp dòng trùng mà không báo; `float('NaN')` khiến phép so sai số trả False nên vẫn
    "đạt"; `csv.DictReader` không chặn header trùng hay dòng thừa cột.
  - Sửa: `doc_csv(p, cot, ten)` dùng `csv.reader(strict=True)` và chặn header trùng hoặc rỗng, thiếu cột bắt buộc, mọi
    dòng có số cột khác header (kể cả dòng trống). Thêm `_theo_khoa` để chặn khóa trùng. `doc_bang_r56` kiểm cột,
    run/nhóm duy nhất, đúng tập 36 run / 6 nhóm, cặp detector/dieu_kien/bo_loc/seed khớp tên run, `so_seed` = 3, mọi giá
    trị là số hữu hạn (`_so`), IDSW nguyên. Hàm này được gọi ở **chuan-bi** (chặn trước khi tạo phiên) và lại ở
    phan-tich. `combined_metrics.csv` và `per_video_metrics.csv` cũng đọc bằng bộ đọc chặt. Phép so dùng
    `not d <= dung_sai`, nên NaN không thể "đạt" kể cả khi lọt qua.
  - Fixture: ca 10 và 11 ở mục 3.1.
- **[P2] Dung sai IDSW của bảng PF bị nâng ngầm.**
  - Vấn đề: mã vòng 1 dùng `max(tb[m], 0.0005)`.
  - Sửa: bỏ `max`. Giao thức phiên bản 2 khai báo dung sai riêng `tong_hop_36_run` và `pf500_trung_binh`, kèm lý do
    lấy từ quy tắc làm tròn của `src/r56_tong_hop/tong_hop_test40.py` dòng 73, và `eps_dau_phay_dong`. `kiem_giao_thuc`
    kiểm các trường này. Riêng IDSW của `tong_hop` và tiêu chí tái tổ hợp so với `combined_metrics.csv` (≤ 1e-9,
    IDSW = 0) bắt buộc phải chính xác. `kiem_chung.json` ghi `dung_sai_ap_dung` và `sai_so_max` thực tế.
- **[P3] Mô tả nơi ghi hash mã sai.** Mục 3 nay là bảng hash theo đúng nơi ghi: `test_phan_tich_test40.py` chỉ có trong
  `ket_qua_test.json`.
- **Ảnh hưởng đến kết quả:** không có.
  - Bootstrap, chỉ số mẫu và công thức không đổi. `chi_so_*.csv`, `diem_uoc_luong.csv`, `doi_chieu_*.csv`,
    `nhom_video.csv` của phiên mới trùng byte với phiên cũ (mục 3).
  - Trên đầu vào thật, bảng R56 vốn không có dòng trùng, NaN hay sai schema: chuan-bi mới qua với cùng 188 file.
  - Phiên cũ giữ nguyên; phiên mới là phiên hiện hành.
- Không chạy tracking, evaluator hay GPU; không tạo cache; không đổi đầu vào.
