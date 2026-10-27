# R59B — Kết quả đối chứng SORT–ByteTrack test40 (chỉ c25), 23/10/2026

Trạng thái: **chạy xong, tự kiểm (Tier 1) đạt; đã sửa hồ sơ theo review B vòng 1 (blocked), chờ Codex review lại toàn
bộ R59B.** Mục 0 `handoff/PLAN_DA_CHOT.md` đã cập nhật trước file này. Không sửa mã, fixture, giao thức, YAML, khóa, kết
quả hay bằng chứng A/B; không commit.

**Sửa hồ sơ sau review B vòng 1** (`D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_004.json`,
verdict blocked; người dùng duyệt bước P9e199330, phương án (a), không làm c10; chỉ sửa file này và mục 0):
- **P1** — vị trí bằng chứng kiểm khóa cuối: xem §2a (lệnh 16 trong §2 trỏ tới đó).
- **P2** — mô tả tầng điểm thấp: sửa ở §5 (tầng điểm thấp ByteTrack rỗng ở c25, hiệu số không đo đóng góp của nó).

## 1. Cổng A và ràng buộc

| Mục | Giá trị |
|---|---|
| Review A thực tế của lượt | `D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_003.json`, verdict **approve**, SHA256 `1762b88a8c9a9c865a70ed013ce0a439388a83588c5a4952f7a6227821e9f8d6` (vòng 1 `review_002.json` = revise, đã sửa trong A) |
| Gói A | `results/phase2/r59_sort_test40/chuan_bi/goi_a_20261023_135911_306800_pid9372`, `goi_a.json` SHA256 `d0ddcde2…5b740` |
| Khóa A | `results/phase2/r59_sort_test40/chuan_bi/khoa_20261023_135704_309300_pid25512`, `khoa.json` SHA256 `b18fc376…4057d`, 1.557 file |
| Phiên B | `results/phase2/r59_sort_test40/thuc_thi/phien_20261023_160539_122500_pid12864` |
| `phien.json` | bản sao `review_a.json` + hash; khóa, gói A, giao thức `06e998d4…33c4`, mã wrapper `ff805a96…9d49` |

Mọi lệnh B gọi `Phien.xac_minh`: kiểm khóa (khớp hash lúc mở phiên), giao thức/mã không đổi, bản sao review không đổi,
từng file trong `goi_a.json#file` còn nguyên. `git status` cuối lượt: không file `src/`/`configs/` đã theo dõi nào bị
sửa; `src/r59_sort_test40/` và `configs/phase2/r59_sort_test40/` không bị sửa sau approve (hash mã/giao thức được wrapper
kiểm ở mọi bước; `kiem-khoa` cuối DAT — bằng chứng đã chuyển ra vùng automation, §2a).

## 2. Lệnh đã chạy (foreground, tuần tự, Python `khoaluan-mot`, entry point `src/r59_sort_test40/dieu_phoi_sort_test40.py`)

| # | Lệnh `--buoc …` | Exit | Kết quả |
|---|---|---|---|
| 1 | `mo-phien --khoa khoa_20261023_135704_309300_pid25512 --review-a …/review_003.json` | 0 | `PHIEN_ID=phien_20261023_160539_122500_pid12864` |
| 2 | `full --phien <ID> --gioi-han-giay 420` | 10 | 155 đơn vị mới, dừng có kiểm soát giữa đơn vị (420,5 s) |
| 3 | `full --phien <ID> --gioi-han-giay 420` (cùng lệnh, cùng phiên) | 0 | 85 mới, 155 tái sử dụng (đã kiểm marker/hash), 223,8 s |
| 4–9 | `cham --phien <ID> --run <run>` × 6, theo thứ tự giao thức | 0 ×6 | HOTA SORT 0,54521 / 0,54725 / 0,59851 / 0,60042 / 0,60683 / 0,60756 |
| 10–11 | `lap-tracker --phien <ID> --run r59-yolov8n_coco-c25-sort-{cv30,ekf30} --video MVI_39031` | 0 ×2 | MOT trùng byte |
| 12–13 | `cham-lai --phien <ID> --run r59-yolov8n_coco-c25-sort-{cv30,ekf30}` | 0 ×2 | 3 file metric trùng byte, combined trùng |
| 14 | `hau-kiem --phien <ID>` | 0 | đạt: 240 đơn vị, 6 chấm, 2 lập, 2 chấm lại |
| 15 | `tong-hop --phien <ID>` | 0 | 6 cặp, đạt |
| 16 | `kiem-khoa --khoa khoa_20261023_135704_309300_pid25512` | 0 | 1.557 file, DAT — `kiem_khoa_20261023_162725_744900_pid2452`, nay ở vùng automation (§2a) |

**Sai khác so với văn bản plan (ghi rõ):** plan ghi `--gioi-han-giay 2400`; tôi dùng **420** vì công cụ Bash của lượt
agent chỉ cho lệnh foreground tối đa 600 s, chạy 2400 s sẽ bị giết giữa đơn vị (dừng không kiểm soát). Giới hạn chỉ là
trần dừng giữa đơn vị nên giá trị chặt hơn không đổi phạm vi, thuật toán hay đầu ra; phần thiếu được làm bằng cùng lệnh,
cùng phiên đúng như plan. Đoạn 1 dừng ở run thứ 4 (35/40 video), đoạn 2 hoàn tất; `lan_thu_khac` rỗng ở cả hai
`tom_tat.json` (không có lần dở/lỗi); không có `LOI.json` nào trong phiên.

Nhật ký (`cli_log.json`: argv, stdout, stderr, exit, thời điểm) mỗi lệnh ở `<phiên>/nhat_ky/{mo_phien_*,full,cham,lap_tracker,cham_lai,hau_kiem}/`.
`tom_tat.json` của hai đoạn full: `nhat_ky/full/lan_20261023_160605_129855500_0001_pid20372`, `nhat_ky/full/lan_20261023_161327_985572800_0001_pid18068`.

## 2a. Bằng chứng kiểm khóa cuối — vị trí hiện tại và lịch sử di chuyển (P1)

- **Ghi ban đầu:** lệnh 16 (`kiem-khoa`, exit 0, 2026-10-23 16:27:25–16:27:41) ghi
  `results/phase2/r59_sort_test40/chuan_bi/kiem_khoa_20261023_162725_744900_pid2452/` (`cli_log.json`, `ket_qua.json`,
  `_cau_hinh_thu_vien/`).
- **Di chuyển:** `chuan_bi/` không thuộc write_paths B (`thuc_thi/`, `tong_hop/`, `review59/03`, PLAN), nên lúc
  2026-10-23 16:36:59 phiên điều phối đã **chuyển nguyên vẹn** thư mục này (trước recover, không xóa, không sửa nội dung)
  sang vị trí hiện tại:
  `D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/file_ngoai_pham_vi_B_20261023/kiem_khoa_20261023_162725_744900_pid2452/`.
  Cùng thư mục cha có `GHI_CHU.txt` (lý do, thời điểm) và `SHA256SUMS_truoc_khi_chuyen.txt`. Thư mục không còn ở
  `chuan_bi/` (ở đó chỉ còn ba phiên kiểm khóa của A: `kiem_khoa_20261023_130659_710200_pid23824`,
  `kiem_khoa_20261023_131058_138900_pid26200`, `kiem_khoa_20261023_135723_981300_pid27468`).
- **Hash trước khi chuyển** (theo `SHA256SUMS_truoc_khi_chuyen.txt`, do phiên điều phối ghi):

  | File (tương đối trong thư mục kiểm khóa) | SHA256 |
  |---|---|
  | `cli_log.json` | `a5181c26448ff9b8d6a8359a4ef253de8b5f1dbc2983d243982dec8b104a8870` |
  | `ket_qua.json` | `7a2851c6e26229b4bf402c865c1c00794fb2c40fb22718cae8b7e28b3225b5c3` |
  | `_cau_hinh_thu_vien/ultralytics/Ultralytics/settings.json` | `4a0445a951396483f1e0d4d61b0bf673b8d254889c274f4f83a5b3786c512494` |

- **Nội dung đã đọc lại ở vị trí mới:** `ket_qua.json` — `khoa` = `chuan_bi/khoa_20261023_135704_309300_pid25512`,
  `khoa_sha256` `b18fc37632fe94e0b1f5310df1c685f374ab786531800b69f6dced30c3d4057d` (= khóa A), `so_file` 1557, `lech: []`,
  `dat: true`, `luc` 16:27:29; `cli_log.json` — argv `--buoc kiem-khoa --khoa khoa_20261023_135704_309300_pid25512`,
  `exit: 0`, stderr rỗng, stdout `… 1557 file, khoa.json b18fc376…4057d, DAT`, Python `khoaluan-mot`.
- **Giới hạn kiểm của lượt sửa hồ sơ:** lượt này không có lệnh băm trong quyền Bash nên **không tự băm lại** ba file ở vị
  trí mới; đối chiếu hash với `SHA256SUMS_truoc_khi_chuyen.txt` để reviewer kiểm độc lập.
- **Hạn chế đã biết của mã R59 (không sửa mã đã khóa):** `buoc_kiem_khoa`
  (`src/r59_sort_test40/dieu_phoi_sort_test40.py:991`, `W.tao_phien(ctx.vung("chuan_bi"), …)`) luôn ghi phiên kiểm khóa
  vào `chuan_bi/`, kể cả khi gọi trong B, nên mỗi lần kiểm khóa ở B sẽ ghi ra ngoài write_paths B. Hạn chế này chưa sửa vì
  mã đã khóa ở A.

## 3. Bằng chứng nghiệm thu

- **240 đơn vị chính** (6 run × 40 video theo manifest khóa), mỗi đơn vị có `HOAN_TAT.json` gắn ràng buộc (khóa, run,
  detector, bộ lọc, `conf_min`/lớp/`max_det`, hash cấu hình hiệu lực, hash manifest + NPZ cache, fps=25, `elapsed_mode=skip`,
  `assoc_log=false`, `pf_diag`), hash MOT, `info` tracker; kiểm `kiem_info_sort` (số frame = cache) và module sau đơn vị.
- **Timeline:** `hau_kiem.json` — mỗi run 56.340 ảnh; mỗi video `n_buoc_timing` = `seq_info` khóa. D4 (ảnh không có frame
  GT) nằm trong timeline này; 173 D4/detector đã đếm ở A, B không xóa/không ép D4 (đủ 56.340 ảnh mỗi run). B không đếm
  lại D4 riêng.
- **Cache bất biến:** băm manifest/NPZ detector trước/sau mỗi run trong mỗi đoạn full (`tom_tat.json#cache`, 4 + 6 mục
  đều `trung: true`); trước/sau mỗi lap-tracker (163 file, `479493cc…`); `mo-phien`, `hau-kiem` và `kiem-khoa` cuối
  (bằng chứng ở §2a) so cache với khóa: khớp.
- **Chấm:** 6 lần chấm ở `<phiên>/cham/<run>/lan_*` (đường dẫn trong `tong_hop/…/nguon.json`), mỗi lần có
  `per_video_metrics.csv`, `combined_metrics.csv`, `raw_fields.json`, cấu hình evaluator, bản sao MOT 40 video kiểm hash
  marker, GT kiểm trước/sau; `combined_metrics.csv` = combined evaluator; metric hữu hạn.
- **K8 tái lập:** MVI_39031 (video đầu manifest), frame 1–1470:
  CV `6b4449e8…b201` = gốc; EKF `0993c8fd…fa06` = gốc (`lap_tracker/<run>/MVI_39031/lan_*/ket_qua_lap.json`).
  Chấm lại cùng bộ MOT 40 video: ba file metric trùng byte, combined trùng, lần gốc còn hợp lệ
  (`cham_lai/<run>/lan_*/ket_qua_cham_lai.json`). Các lần này nằm riêng, không tính vào 240 đơn vị.
- **Đối chiếu ByteTrack R56:** combined từ đúng 6 lần chấm R56 định danh trong khóa (hash `combined_metrics.csv` = khóa),
  so bảng `results/phase2/r56_tong_hop/phien_20261021_161832_735800_pid11640/tong_hop_36_run.csv` (hash = giao thức):
  **sai số thực tế 0** mọi metric × 6 run (dung sai khai báo trước: tỷ lệ 0,0005 pp, IDSW/FP 0).
- Không tạo `pf_diag/` hay `assoc_log/` trong phiên (SORT không dùng; `pf_diag=true` truyền như R56 không có hiệu lực).

## 4. Bảng đối chiếu 6 cặp

Nguồn: `results/phase2/r59_sort_test40/tong_hop/phien_20261023_160539_122500_pid12864/doi_chieu_6_cap.csv`
(SHA256 `670494662f685e69cd862bd48567dbf1861f7efc3948a31e57698eb754a1138e`; `nguon.json` `4839345d…6075`;
`kiem_chung.json` `6dde8c4e…1d41`; `SHA256SUMS` cùng thư mục). COMBINED từ evaluator, không trung bình điểm video.
Tỷ lệ = giá trị evaluator × 100 (%), làm tròn 3 chữ số; hiệu = ByteTrack − SORT bằng **điểm phần trăm (pp)**, tính từ
giá trị chưa làm tròn; IDSW/FP là số đếm.

| Detector | Bộ lọc | HOTA BT / SORT / Δpp | AssA Δpp | DetA Δpp | IDF1 Δpp | IDSW BT / SORT / Δ | MOTA Δpp | FP BT / SORT / Δ |
|---|---|---|---|---|---|---|---|---|
| yolov8n_coco | CV | 58,235 / 54,521 / **+3,714** | +7,107 | +0,753 | +5,519 | 2.498 / 4.318 / −1.820 | +0,746 | 73.478 / 62.846 / +10.632 |
| yolov8n_coco | EKF | 58,353 / 54,725 / **+3,628** | +6,930 | +0,745 | +5,375 | 2.523 / 4.210 / −1.687 | +0,772 | 73.138 / 62.949 / +10.189 |
| yolov8n_mixed5 | CV | 61,643 / 59,851 / **+1,792** | +3,478 | +0,213 | +2,802 | 2.591 / 3.019 / −428 | −0,200 | 58.551 / 46.146 / +12.405 |
| yolov8n_mixed5 | EKF | 61,711 / 60,042 / **+1,669** | +3,253 | +0,197 | +2,739 | 2.547 / 2.909 / −362 | −0,219 | 58.265 / 46.044 / +12.221 |
| yolo11n_mixed5 | CV | 62,063 / 60,683 / **+1,380** | +2,921 | −0,060 | +2,409 | 2.547 / 2.887 / −340 | −0,571 | 64.854 / 52.241 / +12.613 |
| yolo11n_mixed5 | EKF | 62,239 / 60,756 / **+1,483** | +3,092 | −0,017 | +2,615 | 2.458 / 2.784 / −326 | −0,628 | 64.628 / 52.003 / +12.625 |

AssA/DetA/IDF1/MOTA hai giá trị gốc có trong CSV. Quan sát mô tả (chỉ điểm): trên cả 6 cặp HOTA, AssA, IDF1 của
ByteTrack cao hơn SORT và IDSW thấp hơn; DetA gần bằng nhau (−0,06…+0,75 pp); ByteTrack xuất nhiều FP hơn (+10.189…
+12.625) và MOTA thấp hơn SORT ở 4/6 cặp mixed5. Hiệu số lớn nhất ở detector COCO.

## 5. Giới hạn diễn giải

- **Chỉ c25** (SORT `det_thresh=0,25`; detection đầu vào conf > 0,25 như ByteTrack R56 c25). Với đầu vào conf > 0,25
  và `track_high_thresh` 0,25 (`track_low_thresh` 0,1), **tầng điểm thấp của ByteTrack rỗng** (`bt_thap = 0`): mọi
  detection vào tracker đều thuộc tầng cao. Xác nhận bởi giao thức đã khóa (`configs/phase2/r59_sort_test40/giao_thuc.json`,
  mục giới hạn: "vong ghep diem thap cua ByteTrack rong"), fixture K3 (`src/r59_sort_test40/test_dieu_phoi_sort_test40.py:115`,
  sau runner `bt_thap = 0`) và K2 dữ liệu thật (`chuan_bi_20261023_135554_118700_pid21600/k2_parity.csv`: 120/120 tổ hợp
  detector–video `n_det_bt_thap = 0`; K2 báo lỗi nếu tầng thấp khác rỗng ở bất kỳ frame nào). Vì vậy hiệu số ByteTrack−SORT
  ở c25 **không** đo đóng góp của association điểm thấp; không có c10.
- Khác biệt còn lại giữa hai tracker ở c25 là **vòng đời, association và khởi tạo track** (ByteTrack: ghép tầng cao với
  track đang theo + lost, vòng ghép track chưa xác nhận, `new_track_thresh`, `track_buffer`; SORT: một vòng IoU 0,3,
  `min_hits=3`, `max_age=30`), cùng bộ lọc CV/EKF; không tách được thành phần nào trong số này gây hiệu số.
- **Chỉ điểm và hiệu số** trên 40 video test, một lần chạy tất định mỗi run; **không CI**, không kiểm định hay suy luận
  thống kê; không kết luận tổng quát ngoài test40/UA-DETRAC.
- **Không đo tốc độ:** thời gian ở mục 2 là thời gian lệnh gồm kiểm khóa/hash/module, không phải FPS.
- Logging khác nhau: SORT `assoc_log=false`; ByteTrack R56 `assoc_log=true` (giữ nguyên nguồn, không chạy lại ByteTrack).
  Theo thiết kế R58/R59A logging không đổi quyết định tracker; B không kiểm lại điều này.
- Số ByteTrack là tái dùng nguồn R56 có điều kiện (mục 4.3 đạt ở A, chấm lại nguồn trùng byte ở A); B không chạy tracker ByteTrack.

## 6. Bước tiếp

Gửi Codex review lại toàn bộ R59B (gói A + review A + phiên B `thuc_thi/phien_20261023_160539_122500_pid12864` +
`tong_hop/phien_20261023_160539_122500_pid12864` + khóa + bằng chứng kiểm khóa cuối ở §2a + bản sửa hồ sơ P1/P2). Ngân
sách sửa-review đã dùng 5/5. Sau nghiệm thu dừng ở
kết quả đối chứng mốc 4; không tự mở mốc tiếp theo.
