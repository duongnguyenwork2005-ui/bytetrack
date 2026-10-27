# R61_PREP — Kết quả gói chuẩn bị (chờ Codex review)

Mục 0 PLAN đã cập nhật trước file này. Approve gói này **không** mở cổng A hay B.

## 1. Lệnh đã chạy (Python `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`, cwd repo)

| Lệnh | Exit | Artifact |
|---|---|---|
| `git status --short`, `git rev-parse HEAD` (`117d62da…4bbb`), `--abbrev-ref HEAD` (`phase2-mixed-traffic`), `git diff --stat` | 0 | (trong chup_dau) |
| `chuan_bi_bdd.py --buoc chup-dau --phien …/chuan_bi_cpu/lan_01` (lần 1) | 1 | không ghi gì: kiểm vùng chặn mkdir thư mục cha chưa có; đã sửa `tao_thu_muc` cho phép mkdir tổ tiên của vùng (vẫn kiểm reparse) |
| `chuan_bi_bdd.py --buoc chup-dau --phien …/lan_01` | 0 | `lan_01/chup_dau/` (PLAN nguyên byte, git status/diff, hash 43 nguồn nhỏ, liệt kê vùng nguy cơ) |
| `chuan_bi_bdd.py --buoc kiem-ke --phien …/lan_01 --giao-thuc configs/phase2/r61_bdd/giao_thuc.json` | 0 | `lan_01/kiem_ke/{kiem_ke_theo_video.csv, kiem_ke.json, nguon.json}` |
| `test_chuan_bi_bdd.py --phien results/phase2/r61_bdd/tests/lan_01 …` | 1 | `CUDA_VISIBLE_DEVICES=""` để `torch.cuda.is_available()` True ⇒ `DetectorSpec.fill_env` lỗi; lỗi bất ngờ được ném lại, không nuốt (chỉ truy vấn tên thiết bị, không tính toán GPU) |
| `test_chuan_bi_bdd.py --phien …/tests/lan_02 …` | 1 | lỗi ghi `bang_chinh.csv` (cột seed/min/max PF); sửa `ghi_csv` |
| `test_chuan_bi_bdd.py --phien …/tests/lan_03 …` | 139 | 1 kiểm không đạt (`cham` đã chấm các run đủ trước khi từ chối run thiếu) rồi segfault native ở bước kế; sửa `cham` từ chối khi còn thiếu bất kỳ đơn vị nào |
| `test_chuan_bi_bdd.py --phien …/tests/lan_04 …` | **0** | `lan_04/ket_qua_test.json` (69/69 đạt), `cli_log.json`, `bang_chung/`, phiên giả `A/phien_A_01`, `B/phien_B_01` |
| `chuan_bi_bdd.py --buoc chup-cuoi --phien …/lan_01` | xem mục 6 | `lan_01/chup_cuoi/so_sanh.json` |

Lưu ý quy trình: lần chạy lan_02 tôi gọi qua `| tail -60` (vi phạm quy tắc "không nối |" của lượt; lệnh Python vẫn đúng mẫu,
không ảnh hưởng artifact). Segfault lan_03 chưa rõ nguyên nhân (lan_02 và lan_04 đi qua cùng bước); ghi là giới hạn.

## 2. Kiểm kê (tiêu chí 200 video / 39.973 ảnh / 615)

`kiem_ke.json`: 200 dòng video, 39.973 ảnh, 465.512 annotation, **615 ảnh không annotation** (khớp `val_report.json`),
27 video có; vị trí: 19 trước annotation đầu, 488 giữa, 108 sau annotation cuối; 0 video không có annotation nào.
GT hiện có: 778 frame không có lớp 1 (= 615 ảnh không annotation + 163 ảnh chỉ có lớp không chấm/ignore), 134 frame chỉ
có lớp 13; 382.903 dòng lớp 1, 22.998 dòng lớp 13. 200/200 video đạt: seqLength = frame cuối, frame đầu 1, không khoảng
thiếu, frameRate 5, số dòng lớp 1/13 = annotation vehicle/ignore, ảnh không annotation ⊆ frame GT vehicle trống, tập frame
có xe khớp nguồn, có trong seqmap. Ảnh thiếu trên đĩa = 0 là **trích báo cáo cũ**, lượt này không mở ảnh. Không có sai khác
cần giải quyết.

## 3. Fixture CPU lan_04 (69/69, exit 0)

Dữ liệu tổng hợp riêng (3 video × 10 frame + 1 video không ảnh, nhãn COCO-video, GT xuất bằng `parse_coco` +
`export_motchallenge` thật); detector mô phỏng (`ModelGia`, CUDA ẩn), mọi đường khác gọi hàm thật. Nhóm kiểm:
- Ánh xạ/cls_goc (lop5→COCO, coco80 giữ nguyên, thiếu lớp/sai loại bị chặn); conf ngặt: float32(0,10)/(0,25) bị loại, ngay
  trên giữ, ngay dưới loại (qua `VideoDetections` của cache thật); lọc conf→lớp→[:300].
- Cache: truncated/unreadable chặn khi tạo (không ghi npz) và khi nạp; legacy (thiếu spec_sha256), nhãn/checkpoint
  provenance sai, meta cấu hình sinh sai, npz đổi byte ⇒ chặn.
- Timeline BDD liên tục 1..N, time_s=(i−1)/5; steps ≡ skip với n_steps={1}; track_buffer 6.
- `mot_crowd`: box trùng vùng lớp 13 không tính FP (FP 1 so với 2 ở `mot` thường); frame GT trống giữ trong timeline, box ở đó là FP.
- PF500 3 seed: seed thực dùng 0/1/2, 500 particle, optimal_proposal false, roughening 0.
- Thống kê: TB COMBINED theo seed ≠ COMBINED raw gộp ≠ TB metric video; chỉ số cùng dãy `boot_stats`; CI khớp
  `boot_stats.paired_bootstrap` (≤1e-9); tái tính từ `.npy` lưu trùng; đơn vị pp/count; 108 CI đúng cấu trúc/dấu.
- Tái tổ hợp vs COMBINED evaluator (36 run fixture): sai số tối đa HOTA 0,0, IDF1 0,0, IDSW 0 (dung sai 1e-9/1e-9/0).
- Luồng A/B (phiên giả trong tests): ngắt/tiếp tục cache (10,10,0) và tracking (10,10,0) không ghi đè (sha + mtime);
  thiếu đơn vị ⇒ `cham` exit 2, không run nào được chấm; review chưa approve / chưa mở phiên ⇒ 2; sửa khóa/gói A/giao thức/
  cache A/đơn vị hoàn tất ⇒ 2; lap-tracker 36 đơn vị trùng byte, chấm lại 6 run trùng byte; hậu kiểm đếm đủ; gói A bất biến
  khi B chạy; kiem-khoa B ghi trong B, không ghi A.
- Vùng ghi: thoát vùng và junction (cả thư mục cha) bị chặn trước ghi; không pyc mới trong `src/`, settings AppData,
  `data/interim`, `data/processed` không đổi; không file `.dang_ghi`; cấu hình thư viện trong `_cau_hinh_thu_vien/`.

**Chưa kiểm ở CPU (chờ A/B):** suy luận thật/GPU, tốc độ, device/phiên bản thực, khóa với ảnh/checkpoint thật, chấm trên
200 video thật, bootstrap 5.000 × 200 trên dữ liệu thật (fixture dùng 300 mẫu × 3 video), segfault lan_03.

## 4. Hợp đồng lệnh và vùng ghi cho plan cổng A

Chi tiết từng lệnh: `configs/phase2/r61_bdd/hop_dong_lenh.json`. Cổng A (một phiên `results/phase2/r61_bdd/chuan_bi/<phien_A>/`):
`dieu_phoi_bdd.py --cong A --phien <phien_A>` với `--buoc tien-kiem` → `cache --detector <det> --device 0 [--so-doan N]`
(lần lượt 3 detector) → `kiem-cache` → `smoke` → `khoa` → `kiem-khoa` → `dong-goi-a` → Codex review A, dừng.
Ghi: `<phien_A>/{tien_kiem,cache/<det>,kiem_cache,smoke,khoa,kiem_khoa,_tam,_cau_hinh_thu_vien}/`, `goi_a.json`,
`SHA256SUMS`, `cli_log.json`. Quyền cần cấp riêng: GPU, Bash cho `dieu_phoi_bdd.py`. Cổng B ghi duy nhất
`results/phase2/r61_bdd/thuc_thi/<phien_B>/`; `mo-phien` cần review A approve (có `goi_a_sha256`) và plan B duyệt
(`duyet`, `goi_a_sha256`, `khoa_sha256`).

## 5. Dự toán (kịch bản, không phải cam kết)

R56/03: cache 3.438,3 s, tracking 12.884 s, chấm 2.121 s trên 56.340 frame/run; × 39.973/56.340 ≈ 3,64 h. BDD cũ `toc_do.md`:
detector COCO 22,9 ms/frame. Kịch bản 4–8 h chưa gồm bootstrap 5.000 × 6 nhóm, hậu kiểm băm 600 NPZ, tái lập, máy bận,
chi phí khởi tạo 200 video. Cổng A phải đo: giây/frame suy luận mỗi detector, giây tracking mỗi bộ lọc trên smoke, giây
chấm/run, giây băm/kiểm cache, dung lượng cache, bộ nhớ đỉnh.

## 6. Đối chiếu cuối

`chup-cuoi` so PLAN ngoài mục 0 (byte), 43 nguồn bảo vệ, settings AppData, pyc, vùng nguy cơ và `git status` mới so với
write_paths; kết quả trong `lan_01/chup_cuoi/so_sanh.json` (exit ghi trong báo cáo JSON của lượt).

## 7. Vòng sửa R61-F1…F7 (24/10/2026) — BLOCKED

Nguồn finding: `D:/LABOTARY/automation/runs_local/bytetrack_r61_bdd_prep_20261023_2215/reports/review_001.json`
(verdict revise). Bằng chứng lan_01…lan_04 và `chuan_bi_cpu/lan_01/` giữ nguyên.

| Finding | Sửa (chỉ `src/r61_bdd/`, `configs/phase2/r61_bdd/`) | Fixture hồi quy (tên kiểm) |
|---|---|---|
| F1 | `nap_phien_b`: `kiem_goi_a` (băm toàn bộ gói A + file thừa cache/smoke) + `kiem_toan_ven_b` (so khóa với ảnh hiện tại) trước mọi thao tác B; `mo-phien` cũng so khóa | `F1_doi_{gt,seqinfo,yaml,nhan,manifest}_sau_mo_phien_bi_chan`, `F1_doi_artifact_smoke_goi_A_bi_chan` (exit 2, B không đổi byte/mtime) |
| F2 | `nap_tien_kiem`: bắt buộc `tien_kiem.json` cùng giao thức, `kiem_nguon` bấm lại = tiền kiểm, TRƯỚC write_meta/nạp detector; provenance ghi hash thực tế; marker cache ghi tien_kiem/gt/seqinfo sha; `khoa` kiểm lại | `F2_cache_khong_tien_kiem_bi_chan`, `F2_cache_{nhan,manifest,gt}_doi_sau_tien_kiem_bi_chan` (đếm lần nạp detector), `F2_cache_chay_khi_nguon_khop_tien_kiem` |
| F3 | `cham_run`: tạo trước `_trackeval/<run>`, kiểm summary/detailed/log lỗi/stdout/marker/per_video/combined/raw_fields trước helper và lại ngay trước ghi pandas/save_raw_fields; stdout lưu cả nhánh lỗi | `cham_run_junction_{trackeval_summary_dir,log_loi,raw_fields,per_video_pandas}_bi_chan_truoc_ghi`, `cham_run_nhanh_loi_log_trong_phien_khong_nuot` |
| F4 | `kiem_marker(artifact, danh_tinh)` kiểm schema, artifact rỗng, tập artifact, danh tính; `da_xong` → marker hỏng = exit 2; quét trước ở cache/tracking/lap/cham/cham-lai/smoke; `chay_don_vi`/`cham_run` từ chối marker đã có | `F4_marker_{rong,artifact_rong,thieu_artifact,sai_danh_tinh}_bi_chan`, `F4_mot_lap_bi_doi_khong_ghi_de`, `F4_metric_cham(_lai)_bi_doi_khong_ghi_de`, `F4_marker_cham_rong_bi_chan` |
| F5 | `ma_nguon` = đóng import tĩnh (ast, kể cả import trong hàm) từ `src/r61_bdd`; `ma_thu_vien` = mọi .py ultralytics/trackeval; `git_dirty_sha256` = byte file tracked dirty; bỏ `ma_da_nap` | `ma_nguon_dong_import_tinh_du`, `ma_nguon_doc_lap_lich_su_import`, `ma_nguon_ban_sao_fixture_trung_byte`, `F5_doi_ma_{thong_ke,boot_stats}_bi_chan`, `F5_khoa_co_ma_nguon_thong_ke_va_git_dirty` |
| F6 | bỏ `--so-mau`; `Ctx`: giao thức không phải fixture mà so_mau ≠ 5000 → 2; `doc_chi_so` kiểm int64/shape/miền/trùng default_rng(0); `kiem_chung` ghi hash chỉ số/CSV/đầu vào; hậu-kiểm tính lại 108 CI và so byte CSV | `so_mau_san_xuat_khac_5000_bi_chan`, `F6_chi_so_{so_mau_299,chi_so_am,doi_noi_dung,dtype_int32}_{tong_hop,hau_kiem}_bi_chan`, `F6_sua_{ci_108.csv,bang_chinh.csv,kiem_chung.json}_...`, `F6_hau_kiem_dat_lai_sau_khoi_phuc` |
| F7 | giao thức: `yaml_sha256` từng bộ lọc, `tham_so_bo_loc_du_kien` (CV/EKF/UKF/PF500 + `src/config.py` sha256); `kiem_tham_so` đối chiếu instance thật ở tiền-kiểm/khóa/mỗi đơn vị | `tham_so_bo_loc_tien_kiem_khop_giao_thuc`, `tham_so_bo_loc_don_vi_pf_khop_giao_thuc` |

Phụ: `chuan_bi_bdd.py chup-cuoi --phien-dau` để đối chiếu baseline gốc lan_01 từ phiên mới (chưa chạy).

**Kết quả chạy** (`test_chuan_bi_bdd.py --phien results/phase2/r61_bdd/tests/lan_0X --giao-thuc configs/phase2/r61_bdd/giao_thuc.json`):

| Phiên | Exit | Điểm dừng | Ghi chú |
|---|---|---|---|
| lan_05 | 139 (segfault) | sau `F5_doi_ma_boot_stats` (cham-lai exit 2 đã ghi log) | mọi kiểm in ra trước đó DAT |
| lan_06 | 139 | sau `F1_doi_nhan` | mọi kiểm in ra trước đó DAT |
| lan_07 | 139 | sau `F1_doi_artifact_smoke` | `faulthandler.log`: access violation trong `pathlib._iterate_directories` |
| lan_08 | 139 | trong `sua_cache_A_tracking_B` | đã đổi rglob → `os.walk`; `faulthandler.log`: access violation trong `hashlib` đọc file (`sha256_file`) |

Chưa có `ket_qua_test.json` cho lan_05…08; các ca F2/F3/F4/F6 và kiểm cuối (pyc/settings/interim) **chưa chạy tới**.
Vị trí crash đổi giữa các lần, kể cả trong đọc file thuần. **Nguyên nhân CHƯA RÕ** (đính chính: bản trước viết "không
phải logic hợp đồng" — bỏ kết luận này vì chưa xác minh). Việc điều tra (P44564acd) đã duyệt nhưng thuộc mốc sau.
Tiêu chí "fixture exit 0" **chưa đạt** ở vòng này → không báo done.

Đính chính mục 7 dòng "Phụ": `chup-cuoi --phien-dau` **đã chạy** ở `chuan_bi_cpu/lan_02` (baseline gốc lan_01):
`so_sanh.json` dat=false, `cli_log.json` exit 2, lỗi duy nhất "vùng nguy cơ đổi ngoài write_paths: .git".

## 8. Vòng P056e6f3d (24/10/2026) — tách lệnh A/B thành tiến trình con

**(1) Ghi nhận crash lan_05…08.** 4/4 phiên exit 139 (log công cụ; reviewer review_002 đọc lại). `faulthandler.log`
lan_07: access violation trong `pathlib._iterate_directories`; lan_08: access violation trong `hop_dong_bdd.sha256_file`
(`hashlib`/đọc file) ← `kiem_goi_a` ← `nap_phien_b` ← `dieu_phoi_bdd.main` chạy **trong tiến trình test** qua
`chay_va_ghi_log` (`thu_sua` sửa cache A rồi `lap-tracker`). Event log do **phiên điều phối** đọc 24/10 ~03:00 (không phải
lượt này): Application 4 giờ không có APPCRASH python.exe; CodeIntegrity/Operational không có sự kiện; Smart App Control
tắt từ 23/10 ~12:00. Lượt này không có quyền đọc event log nên không đọc bổ sung. Thiếu sự kiện **không** dùng để suy
nguyên nhân; nguyên nhân giữ **CHƯA RÕ**.

**(2) Sửa duy nhất `src/r61_bdd/test_chuan_bi_bdd.py`.** `chay_con`: mỗi lệnh A/B giả lập (luồng A/B và F2) chạy bằng
`subprocess` riêng với `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe`, cwd repo, env thêm
`PYTHONFAULTHANDLER=1`, `PYTHONDONTWRITEBYTECODE=1`, `CUDA_VISIBLE_DEVICES=-1`, `PYTHONIOENCODING=utf-8`, timeout 1.800 s.
Lệnh không cần detector chạy thẳng `src/r61_bdd/dieu_phoi_bdd.py <argv>`; lệnh `cache` chạy
`src/r61_bdd/test_chuan_bi_bdd.py --con-dieu-phoi <argv>` (`main_con`): gán `D.NAP_MODEL` = detector mô phỏng `ModelGia`
(dieu_phoi chỉ chấp nhận với giao thức fixture), in một dòng `[NAP_DETECTOR_MO_PHONG]` mỗi lần nạp (F2 đếm từ stdout con),
rồi chạy đúng trình tự khối `__main__` của `dieu_phoi_bdd.py` (Ctx → vung_phien → tao_thu_muc → chuan_bi_moi_truong →
chay_va_ghi_log). Ngay sau mỗi tiến trình con, kể cả exit bất thường/hết giờ, ghi `<phien>/lenh_con/NNN_<cong>_<buoc>.json`
(argv đầy đủ, Python, cwd, exit + hex, giây, sha256 script, số lần nạp detector, stdout, stderr); tiến trình con tự ghi
`cli_log.json` trong phiên A/B giả của nó như trước. Giữ nguyên mọi ca/assert F1–F7, kiểm âm và fixture tổng hợp; các kiểm
đơn vị không phải lệnh A/B (lọc/ánh xạ, cache hỏng, crowd, đường ghi evaluator, thống kê) vẫn trong tiến trình test.
Thêm kiểm `moi_lenh_con_exit_0_2_10_khong_het_gio` (mọi lệnh con có log, không exit bất thường) và
`ma_r61_khong_doi_trong_luc_test`; `ket_qua_test.json` thêm argv/Python, sha256 mã `src/r61_bdd/*.py` và đóng import tĩnh,
tóm tắt lệnh con. Ba chỗ `rglob` còn lại trong test đổi sang `H.liet_ke_file` (os.walk). Không sửa helper, mã sản xuất
(`hop_dong_bdd`/`dieu_phoi_bdd`/`thong_ke_bdd`/`chuan_bi_bdd`), giao thức, thuật toán/metric.

**(3) Ba phiên mới** (`test_chuan_bi_bdd.py --phien results/phase2/r61_bdd/tests/lan_{09,10,11} --giao-thuc
configs/phase2/r61_bdd/giao_thuc.json`, chạy nền qua công cụ Bash, đúng mẫu lệnh, không chuyển hướng/nối lệnh). lan_09
khởi chạy ~03:15; lan_10 và lan_11 khởi chạy ~03:20 **song song** với lan_09 (để kịp ngân sách turn; mỗi phiên vẫn tuần
tự bên trong, thư mục riêng). (Đính chính 24/10, vòng Pa6af1bda: đoạn trước đây ghi "đang chạy / kết quả chưa đọc được"
là trạng thái giữa chừng, đã lỗi thời; kết quả cuối ở đoạn "Kết quả cuối 3 phiên" dưới đây.)

**(4) Đối chiếu cuối.** `chuan_bi_bdd.py --buoc chup-cuoi --phien …/chuan_bi_cpu/lan_03 --phien-dau …/chuan_bi_cpu/lan_01`:
**exit 2, dat=false**, lỗi duy nhất "vùng nguy cơ đổi ngoài write_paths: .git" (mục `.git` trong liệt kê không đệ quy của
thư mục gốc: size −1, chỉ mtime đổi). Điều tra chỉ đọc: lan_01 chup_dau `.git` mtime 1792769476779966700 (≈23/10 22:31
+07), lan_02 chup_cuoi 1792781793444081600 (≈24/10 01:56 +07); HEAD `117d62da…` và nhánh không đổi; PLAN ngoài mục 0, 43
nguồn bảo vệ, settings AppData, pyc không bị báo lỗi. Theo thứ tự mtime (Glob), các mục mới nhất ở gốc `.git` là `index`,
`COMMIT_EDITMSG`, `logs/HEAD`, `config` (config có khóa `vscode-merge-base`/`github-pr-base-branch` do VS Code ghi); không
còn file `.lock`. Ứng viên: làm mới index do `git status` (agent, điều phối, reviewer đều chạy) hoặc tiện ích VS Code —
**chưa xác minh**, không kết luận. Không đặt lại baseline, không sửa `.git`, không nới so sánh → **không đạt**.

**Cập nhật sau khi lan_09 kết thúc** (đính chính 24/10, vòng Pa6af1bda — trước ghi "59 lệnh con" và "055…059", sai):
lan_09 **exit 2**, `ket_qua_test.json` 119/120 đạt, **65 lệnh con** có log (exit 0/10/2 = 18/5/42; không lệnh con nào
exit bất thường, không crash). Kiểm không đạt: `F6_hau_kiem_dat_lai_sau_khoi_phuc`. Lệnh con **050–059** (tong-hop/
hau-kiem của F6) có stderr `[LOI HOP DONG] trang thai hien tai khac khoa: ['git_dirty_sha256: handoff/PLAN_DA_CHOT.md
doi/thieu/thua']`. Nguyên nhân: agent sửa mục 0 PLAN (file tracked dirty được khóa giả bấm byte, đúng thiết kế F5) trong
khi fixture đang chạy → lỗi vận hành của lượt, không phải crash. Hệ quả: các ca F6 âm 050–058 trả 2 vì lý do khác (sai lý
do) nên không dùng làm bằng chứng F6. lan_10/lan_11 bị ảnh hưởng rộng hơn: lệnh con **031–033 và 038–059** (gồm
`sua_don_vi_hoan_tat`, F1 gt/seqinfo, F5, F4, F6) báo cùng lỗi `git_dirty_sha256` → không hợp lệ cho nghiệm thu. Bản mã
lúc đó chỉ kiểm exit 2 nên không phân biệt được (review_003 P1; đã sửa ở mục 9).

**Kết quả cuối 3 phiên:** lan_09, lan_10, lan_11 đều **exit 2, 119/120**, mỗi phiên **65 lệnh con** (18/5/42); kiểm
không đạt duy nhất ở mỗi phiên là `F6_hau_kiem_dat_lai_sau_khoi_phuc` (`ket_qua_test.json` của từng phiên). Không phiên
nào crash. Tần suất: 0/3 exit 0, 0/3 crash. Tiêu chí "3 phiên exit 0" **chưa đạt**.

**(5)** Mục 0 PLAN cập nhật trước mục này. Trạng thái vòng P056e6f3d: **BLOCKED** (đối chiếu `.git` không đạt; 0/3 phiên
exit 0). Không mở cổng A/B, không sửa helper/mã sản xuất/giao thức. Vòng tiếp: mục 9.

## 9. Vòng Pa6af1bda (24/10/2026, vòng sửa-review 4/4) — BLOCKED (1/3 phiên crash)

Quyết định Pa6af1bda (Codex chọn thay người dùng theo ủy quyền) thay bước R61_PREP; giữ mọi tiêu chí gói chuẩn bị và
finding review_003.

**(1) Sửa duy nhất `src/r61_bdd/test_chuan_bi_bdd.py`** (không sửa `chuan_bi_bdd.py`, helper, mã sản xuất, giao thức,
cấu hình, thuật toán/metric; giữ tiến trình con và đủ F1–F7):
- `ly_do(m)` lấy các dòng `[LOI HOP DONG] …` trong stderr lệnh con; `dung_ly_do(m, *can, ds=…)`: kiểm âm chỉ đạt khi
  exit 2 **và** đúng một dòng lỗi hợp đồng chứa mọi chuỗi nguyên nhân dự kiến, không chứa `git_dirty_sha256`; với thông
  điệp danh sách (`trang thai hien tai khac khoa: […]`, `hau-kiem: […]`) danh sách parse bằng `ast.literal_eval` phải
  **bằng từng mục** danh sách kỳ vọng (thừa mục, ví dụ lỗi khóa PLAN, → không đạt).
- Áp dụng cho mọi kiểm âm qua tiến trình con: `A_cong_sai`, `B_mo_phien_review_chua_approve`, `B_chua_mo_phien`,
  `B_thieu_don_vi_khong_cham_full`, 6 ca `thu_sua` (khóa/gói/giao thức/cache A ×2/đơn vị hoàn tất), F1 ×6, F5 ×2,
  F4 ×8 (đường marker cụ thể + nguyên nhân: schema/artifact rỗng, tập artifact, danh tính, `bam lech <file>`), F6 ×12
  (tong-hop: thông điệp chỉ số bootstrap cụ thể; hau-kiem: danh sách `['thong ke: …']` / `['ci_108.csv khac tinh lai…']`
  / `['kiem_chung so_mau lech']` **và** artifact `B/hau_kiem.json` vừa ghi có `dat=false`, `loi` bằng danh sách đó), F2 ×4.
  Kiểm khôi phục `F6_hau_kiem_dat_lai_sau_khoi_phuc` thêm điều kiện không có dòng lỗi và `hau_kiem.json` `dat=true`, `loi=[]`.
- Kiểm toàn phiên mới `khong_loi_khoa_git_dirty_ngoai_du_kien`: không lệnh con nào có `git_dirty_sha256` trong stderr.
  `ket_qua_test.json` → `lenh_con[]` có thêm `ly_do`, `loi_git_dirty`; chi tiết từng kiểm âm ghi `ly_do`, chuỗi kỳ vọng,
  danh sách parse được.
- Chuỗi kỳ vọng lấy từ thông điệp thực của `dieu_phoi_bdd.py` (đã đối chiếu với log lan_09 lệnh 026–049, phần chưa bị
  ảnh hưởng PLAN, và mã nguồn `buoc_hau_kiem`/`doc_chi_so`/`kiem_lai_tong_hop`).

**(2) Ba phiên mới, tuần tự, foreground, cùng bản mã cuối, không sửa file tracked nào trong lúc chạy** (PLAN và báo cáo
này chỉ sửa sau khi lan_14 kết thúc). Lệnh: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe
src/r61_bdd/test_chuan_bi_bdd.py --phien results/phase2/r61_bdd/tests/lan_1N --giao-thuc configs/phase2/r61_bdd/giao_thuc.json`.

| Phiên | Bắt đầu | Exit | Kiểm | Lệnh con (exit 0/10/2) | `git_dirty` ở lệnh con | Ghi chú |
|---|---|---|---|---|---|---|
| lan_12 | 04:47:40 | **0** | 121/121 | 65 (19/5/41) | 0 | 362,8 s; `ket_qua_test.json`, `cli_log.json` đủ |
| lan_13 | ~04:53 (lệnh con 001 lúc 04:53:54, 065 lúc 04:59:25) | **139** (segfault) | — | 65 (19/5/41), đủ 001–065 | 0 | tiến trình cha crash ở bước bấm hash cuối phiên |
| lan_14 | 05:00:10 | **0** | 121/121 | 65 (19/5/41) | 0 | 343,7 s; `ket_qua_test.json`, `cli_log.json` đủ |

- lan_12 và lan_14: mọi kiểm âm F1–F7 đạt **đúng nguyên nhân** (chi tiết `ly_do` trong `ket_qua_test.json`), kiểm khôi phục
  F6 đạt, `goi_A_nguyen_byte_sau_moi_thu_sua`, `khong_pyc_settings_interim_ngoai_pham_vi`, `khong_file_tam_do_dang`,
  `moi_lenh_con_exit_0_2_10_khong_het_gio`, `khong_loi_khoa_git_dirty_ngoai_du_kien`, `ma_r61_khong_doi_trong_luc_test`
  đều đạt; 1.534 file phát sinh trong mỗi phiên. Stderr cấp phiên có traceback TrackEval "fxv-b.txt cannot be read" — là
  ca dự kiến của `cham_run_nhanh_loi_log_trong_phien_khong_nuot` (in-process), giống lan_09–11.
- lan_13: cả 65 lệnh con đã chạy xong với cùng phân bố exit như lan_12/14, không lệnh con nào báo `git_dirty_sha256`.
  Tiến trình cha (test) chết native: `lan_13/faulthandler.log` "Windows fatal exception: access violation" tại
  `hop_dong_bdd.py:61 sha256_file` (`f.read`) ← `bang_hash_thu_muc` ← `test_chuan_bi_bdd.py` dòng 889 trong `main`
  (ghi `bang_chung/hash_phien.json`, sau khi mọi kiểm đã in). Vì crash, không có `ket_qua_test.json`/`cli_log.json` cấp
  phiên và stdout cha (đệm) bị mất; còn `lenh_con/*.json`, `bang_chung/luong_ab.json`, `bang_chung/bootstrap_mau.json`.
  Không tính lan_13 là đạt.
- Tần suất vòng này: **2/3 exit 0, 1/3 crash**. Tổng các phiên subprocess hóa (lan_09–14): crash 1/6 (lan_13). Điểm crash
  lan_13 trùng hàm với lan_08 (`sha256_file` đọc file), lan_07 ở `pathlib`; đây chỉ là quan sát vị trí, **nguyên nhân crash
  CHƯA RÕ**. Lượt này không đọc event log (không có quyền); thông tin event log trước đây (mục 8) là do **phiên điều phối**
  đọc, không phải lượt này; thiếu sự kiện không dùng để suy nguyên nhân.
- Tiêu chí "ba phiên mới cùng bản mã cuối đều exit 0" **chưa đạt**. Không chạy thêm phiên thay lan_13 (bước chỉ cấp
  lan_12–14; P44564acd sau này chỉ tổng hợp bằng chứng, không chạy fixture trùng lặp).

**(3) Tài liệu.** Mục 0 PLAN cập nhật trước mục này (thêm vòng Pa6af1bda; đính chính "055–059" → 050–059 cho lan_09 và
031–033, 038–059 cho lan_10/11; 65 lệnh con/phiên). Mục 8 của báo cáo này đã sửa đoạn lỗi thời ("đang chạy/chưa đọc",
"59 lệnh con").

**(4) Đối chiếu cuối.** Sau mọi cập nhật tài liệu chạy `chuan_bi_bdd.py --buoc chup-cuoi --phien
results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/lan_04 --phien-dau results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/lan_01`
(baseline gốc, không thay). Kết quả (exit, `dat`, danh sách lỗi) nằm ở `chuan_bi_cpu/lan_04/cli_log.json` và
`chuan_bi_cpu/lan_04/chup_cuoi/so_sanh.json`, và được nêu trong báo cáo JSON của lượt gửi orchestrator (không ghi vào
đây để tài liệu không đổi sau snapshot). Theo ủy quyền Pa6af1bda: nếu lỗi duy nhất là mtime của chính thư mục `.git`
thì đó là **ngoại lệ được duyệt**, công cụ vẫn trả `dat=false`/exit 2 và không được gọi là exit 0; nguyên nhân thay
đổi `.git` (git status, VS Code…) **chưa xác minh**.

**(5) Trạng thái: BLOCKED** — lan_13 crash, chưa đủ 3/3 exit 0; ngân sách sửa-review 4/4 đã dùng → dừng chờ người dùng.
Không mở cổng A/B, không chạy dữ liệu thật.

Kết quả `chup-cuoi` lan_04 (đọc lại ở vòng P3bf6db84, `chuan_bi_cpu/lan_04/chup_cuoi/so_sanh.json`): `dat=false`, lỗi
duy nhất "vung nguy co doi ngoai write_paths: .git"; HEAD `117d62da…4bbb`, nhánh `phase2-mixed-traffic`; PLAN trước/sau
mục 0 cùng sha256 với baseline lan_01 (`a50c3bc4…`/`aba1aebb…`).

## 10. Vòng P3bf6db84 (24/10/2026) — quyết định (a) của người dùng: chấp nhận giới hạn crash

Mục 0 PLAN cập nhật trước mục này. Vòng này **chỉ** sửa tài liệu (mục 0 PLAN, báo cáo này) và chạy `chup-cuoi` vào phiên
mới; **không** chạy thêm phiên fixture, không sửa `src/`, helper, giao thức, thuật toán, metric hay bằng chứng cũ.

**(1) Quyết định.** Người dùng chọn phương án (a) trong chat 24/10/2026 ~08:15: chấp nhận `tests/lan_12` + `tests/lan_14`
(exit 0, 121/121, mỗi phiên 65 lệnh con 19/5/41) cùng bằng chứng `tests/lan_13` (đủ 65 lệnh con 001–065 hoàn tất, phân
bố exit 19/5/41 như lan_12/14, không lệnh con nào báo `git_dirty_sha256`; tiến trình cha crash access violation ở bước
băm toàn phiên `bang_hash_thu_muc`, `lan_13/faulthandler.log`) như một **GIỚI HẠN đã được người dùng chấp nhận**. Đây
không phải "3/3 phiên exit 0"; tiêu chí đó được thay bằng quyết định của người dùng, không phải do agent nới.

**(2) Danh sách đầy đủ các phiên crash native (exit 139).**

| Phiên | Chế độ | Vị trí (faulthandler) | Ghi chú |
|---|---|---|---|
| `tests/lan_03` | in-process (vòng đầu) | không có `faulthandler.log` | sau 1 kiểm không đạt (`cham`, đã sửa trước lan_04) |
| `tests/lan_05` | in-process (vòng F1…F7) | không có `faulthandler.log` | không có `ket_qua_test.json`/`cli_log.json` cấp phiên |
| `tests/lan_06` | in-process (vòng F1…F7) | không có `faulthandler.log` | như trên |
| `tests/lan_07` | in-process (vòng F1…F7) | `pathlib.py:384/398 _iterate_directories` | như trên |
| `tests/lan_08` | in-process (vòng F1…F7) | `hop_dong_bdd.py:61 sha256_file` (`hashlib`/đọc file) ← `dieu_phoi_bdd.kiem_goi_a` ← `nap_phien_b` | như trên |
| `tests/lan_13` | subprocess hóa (tiến trình cha) | `hop_dong_bdd.py:61 sha256_file` (`f.read`) ← `bang_hash_thu_muc` ← `test_chuan_bi_bdd.py:889 main` | 65/65 lệnh con xong trước crash |

Vị trí khác nhau (pathlib duyệt thư mục, hashlib/đọc file). Trùng hàm `sha256_file` ở lan_08 và lan_13 chỉ là quan sát vị
trí. **Nguyên nhân CHƯA RÕ**; không loại trừ lỗi logic hợp đồng, lỗi thư viện native hay lỗi phần cứng.

**(3) Môi trường và log Windows** (nguồn: **phiên điều phối** đọc, không phải lượt agent này): Smart App Control tắt từ
23/10 ~12:00; Application log không có APPCRASH `python.exe`; CodeIntegrity/Operational không có sự kiện. Windows Memory
Diagnostic được lên lịch 24/10 05:30, máy khởi động lại 06:03 nhưng **không có bản ghi kết quả**
(MemoryDiagnostics-Results) → **chưa loại trừ được lỗi phần cứng/RAM**. Thiếu sự kiện không dùng để suy nguyên nhân.

**(4) Rủi ro cho cổng A/B (chạy dài).** Crash native ngẫu nhiên (tần suất quan sát: 1/6 phiên subprocess hóa lan_09–14;
4/4 phiên in-process lan_05–08; 1/4 vòng đầu lan_01–04) có thể làm dừng tiến trình giữa chừng, kể cả sau khi đơn vị đã
tính xong nhưng trước khi ghi marker/hash. Yêu cầu đối với plan A/B:
- Tiếp tục theo đơn vị (cache detector-video, tracking run-video, chấm run) như `hop_dong_lenh.json`; đơn vị chỉ hoàn
  tất khi có marker hợp lệ và hash artifact khớp; đơn vị dở/thiếu marker **không** được coi là hoàn tất và không được
  chấm full (đã có kiểm fixture `B_thieu_don_vi_khong_cham_full`, tiếp tục không ghi đè đơn vị hoàn tất).
- Mã thoát: 0 hoàn tất, 10 chưa xong có thể tiếp tục, 2 lỗi hợp đồng; crash native cho mã khác (quan sát 139) và bộ
  điều phối/người vận hành phải phát hiện được, không coi là 0/10.
- **Plan cổng A và plan cổng B phải nêu rõ cách xử lý crash**: phát hiện (exit code, `faulthandler.log`, marker thiếu),
  chạy lại đơn vị dở, giới hạn số lần thử lại, khi nào dừng hỏi người dùng, và cách ghi tần suất crash vào báo cáo.
  Bước hash toàn phiên/kiểm khóa cuối cũng có thể crash → phải chạy lại được mà không đổi kết quả đã ghi.

**(5) Đối chiếu cuối.** Sau mọi cập nhật tài liệu chạy `chuan_bi_bdd.py --buoc chup-cuoi --phien
results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/lan_05 --phien-dau results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/lan_01`
(baseline gốc, không thay). Kết quả nằm ở `chuan_bi_cpu/lan_05/cli_log.json` và `chuan_bi_cpu/lan_05/chup_cuoi/so_sanh.json`,
được nêu trong báo cáo JSON của lượt (không ghi vào đây để tài liệu không đổi sau snapshot). Áp đúng ngoại lệ đã duyệt
(Pa6af1bda/QNR7): chỉ chấp nhận lỗi mtime của chính thư mục `.git`; công cụ vẫn trả `dat=false`/exit 2, không gọi là
exit 0; nguyên nhân thay đổi `.git` **chưa xác minh**. Mọi kiểm khác (HEAD/nhánh, PLAN ngoài mục 0, nguồn bảo vệ, pyc,
settings) phải đạt.

**(6) Trạng thái:** chờ Codex review gói chuẩn bị với giới hạn crash nêu trên. Approve **không** mở cổng A hay B; mỗi
cổng cần plan riêng được người dùng duyệt.

## 11. Tổng hợp crash native (vòng P2e430e0b, 24/10/2026)

Mục 0 PLAN cập nhật trước mục này. Kết quả mục 10: R61_PREP **approve** (`D:/LABOTARY/automation/runs_local/
bytetrack_r61_bdd_prep_20261023_2215/reports/review_005.json`, verdict `approve`) với giới hạn crash người dùng đã chấp
nhận; `chup-cuoi` `chuan_bi_cpu/lan_05`: `dat=false`/exit 2, lỗi duy nhất mtime thư mục `.git` (ngoại lệ đã duyệt).
Bước P44564acd thu hẹp thành **chỉ tổng hợp**: vòng này không chạy phiên fixture nào, không sửa mã/helper/giao thức/bằng
chứng cũ, không đọc event log. Nguồn: `results/phase2/r61_bdd/tests/lan_*/` (`cli_log.json`, `ket_qua_test.json`,
`faulthandler.log`, `lenh_con/`), `chuan_bi_cpu/lan_0*/cli_log.json` và các mục 1, 7–10 ở trên.

**(1) Phiên crash native (exit 139).** Tiến trình chết trước khi ghi `cli_log.json`/`ket_qua_test.json` cấp phiên, nên
exit 139 lấy từ kết quả công cụ Bash của lượt chạy (ghi ở mục 1/7/9, đã qua review), không có artifact exit trong phiên.

| Phiên | Chế độ | Tiến trình chết | Điểm dừng / vị trí (`faulthandler.log`) |
|---|---|---|---|
| `lan_03` | in-process | tiến trình test (duy nhất) | sau 1 kiểm không đạt (`cham`, đã sửa trước lan_04); không có `faulthandler.log` |
| `lan_05` | in-process | tiến trình test (duy nhất) | sau `F5_doi_ma_boot_stats`; không có `faulthandler.log` |
| `lan_06` | in-process | tiến trình test (duy nhất) | sau `F1_doi_nhan`; không có `faulthandler.log` |
| `lan_07` | in-process | tiến trình test (duy nhất) | sau `F1_doi_artifact_smoke`; access violation `pathlib.py:384` ← `:398 _iterate_directories` (log chỉ có hai khung pathlib) |
| `lan_08` | in-process | tiến trình test (duy nhất; `dieu_phoi_bdd.main` chạy trong cùng tiến trình) | trong `sua_cache_A_tracking_B`; access violation `hop_dong_bdd.py:61 sha256_file` ← `dieu_phoi_bdd.py:743–744 kiem_goi_a` ← `:796 nap_phien_b` ← `:1125 main` ← `test_chuan_bi_bdd.py:536 thu_sua` ← `:544 luong_ab` |
| `lan_13` | subprocess hóa | **tiến trình cha** (test); 65/65 lệnh con đã xong (`lenh_con/001…065`, exit 0/10/2 = 19/5/41, đếm lại vòng này) | access violation `hop_dong_bdd.py:61 sha256_file` ← `:324 bang_hash_thu_muc` ← `test_chuan_bi_bdd.py:889 main` (băm toàn phiên cuối `main`) |

Không có tiến trình con nào crash: mọi lệnh con của lan_09–14 có log `lenh_con/*.json` với exit 0/2/10. Vị trí khác nhau
(duyệt thư mục pathlib, đọc file trong `sha256_file`); trùng hàm ở lan_08/lan_13 chỉ là quan sát vị trí.

**(2) Phiên không crash.**

| Phiên | Exit (`cli_log.json`) | Kiểm (`ket_qua_test.json`) | Giây | Ghi chú |
|---|---|---|---|---|
| `lan_01` | 1 | — | 8,8 | lỗi mã (`CUDA_VISIBLE_DEVICES=""`), đã sửa; không phải crash |
| `lan_02` | 1 | — | 27,4 | lỗi ghi CSV, đã sửa; không phải crash |
| `lan_04` | **0** | **69/69** | 32,9 | in-process, bản mã vòng đầu |
| `lan_09` | 2 | 119/120 | 569,1 | 65 lệnh con 18/5/42; `F6_hau_kiem_dat_lai_sau_khoi_phuc` không đạt do PLAN sửa giữa lúc chạy (mục 8) — **không đạt** |
| `lan_10` | 2 | 119/120 | 571,5 | như lan_09 — **không đạt** |
| `lan_11` | 2 | 119/120 | 570,4 | như lan_09 — **không đạt** |
| `lan_12` | **0** | **121/121** | 362,8 | 65 lệnh con 19/5/41 |
| `lan_14` | **0** | **121/121** | 343,7 | 65 lệnh con 19/5/41 |

`faulthandler.log` của lan_09–12 và lan_14 tồn tại nhưng rỗng. Lệnh `chuan_bi_bdd.py` trong `chuan_bi_cpu/` (7 lệnh có
log: chup-dau, kiem-ke, chup-cuoi ở lan_01; chup-cuoi lan_02–05) không crash (exit 0/0/0, rồi 2 ×4 vì mtime `.git`).

**(3) Tần suất (trên số phiên fixture đã chạy).** Tổng **6/14** phiên crash (lan_03, 05, 06, 07, 08, 13); exit 0 ở 3/14
(lan_04, lan_12, lan_14). Theo chế độ: in-process lan_01–08 **5/8**; subprocess hóa lan_09–14 **1/6** (tiến trình
cha), 0/390 lệnh con. Theo vòng: lan_01–04 1/4; lan_05–08 4/4; lan_09–11 0/3; lan_12–14 1/3. Giới hạn so sánh: thời
lượng và khối lượng mỗi phiên khác nhau (lan_04 32,9 s với 69 kiểm; lan_12/14 ~350 s với 121 kiểm; lan_01/02 dừng sớm
vì lỗi mã), nên tỉ lệ theo phiên không phải tỉ lệ theo giờ chạy hay theo thao tác; mẫu nhỏ, không suy ra tỉ lệ cho
cổng A/B.

**(4) Log Windows — nguồn: phiên điều phối đọc, không phải agent.** Application log (cửa sổ 4 giờ, đọc 24/10 ~03:00) không
có APPCRASH `python.exe`; CodeIntegrity/Operational không có sự kiện; Smart App Control tắt từ 23/10 ~12:00. Windows
Memory Diagnostic được lên lịch/chạy 24/10 05:30 (máy khởi động lại 06:03) nhưng **không có bản ghi kết quả**
(MemoryDiagnostics-Results) → chưa loại trừ lỗi phần cứng/RAM. Lưu ý lan_13 (~04:53–04:59) xảy ra **trước** lần chạy
kiểm RAM; sau đó chưa chạy phiên fixture nào. Thiếu sự kiện không dùng để suy nguyên nhân.

**(5) Nguyên nhân: CHƯA RÕ.** Không loại trừ lỗi logic hợp đồng, lỗi thư viện native/Python, lỗi hệ điều hành hay phần
cứng/RAM.

**(6) Khuyến nghị cho plan cổng A/B — CHỈ ĐỀ XUẤT, chưa được duyệt; con số là gợi ý để người dùng quyết.** (Các đề xuất
tách riêng trước đây Pf79bdfa0 "chính sách thử lại" và Pa040a0ba "lấy/chạy lại kiểm RAM" đã bị từ chối như bước riêng;
ở đây chỉ ghi lại để plan A/B cân nhắc.)
- *Phát hiện crash:* coi mọi exit ngoài {0, 2, 10} là bất thường (quan sát qua Git Bash là 139; mã thô có thể khác tùy
  cách gọi tiến trình), cùng với `faulthandler.log` không rỗng, marker đơn vị thiếu/hỏng sau khi tiến trình kết thúc,
  hoặc `cli_log.json` thiếu bản ghi của lệnh. Bật `PYTHONFAULTHANDLER` và ghi `faulthandler.log` trong phiên.
- *Chạy lại đơn vị dở:* dùng cơ chế tiếp tục hiện có (đơn vị chỉ hoàn tất khi marker hợp lệ + hash artifact khớp; không
  ghi đè đơn vị hoàn tất); plan A phải kiểm cách cơ chế này xử lý artifact dở của đơn vị bị crash trước khi chạy thật;
  giữ bằng chứng lần crash (log, faulthandler) không ghi đè.
- *Số lần thử lại tối đa:* gợi ý 2 lần thử lại cho cùng một đơn vị (tổng 3 lần).
- *Ngưỡng dừng hỏi người dùng:* gợi ý dừng khi cùng một đơn vị crash hết số lần thử, khi tổng số crash trong một phiên
  cổng vượt 3, hoặc khi bước khóa/kiểm khóa/hậu kiểm/băm toàn phiên crash 2 lần; khi dừng giữ nguyên phiên, không tự đổi
  giao thức hay bỏ đơn vị.
- *Báo cáo:* ghi bảng crash (thời điểm, lệnh, đơn vị, exit, sha256 `faulthandler.log`, lần thử) và tần suất crash trên
  tổng lệnh/đơn vị vào báo cáo cổng A và B.
- *Kiểm RAM:* đề nghị người dùng chạy lại Windows Memory Diagnostic (`mdsched`) và lưu bản ghi MemoryDiagnostics-Results
  **trước cổng B**; nếu vẫn không có bản ghi, ghi là giới hạn trong hồ sơ B.

**(7) Trạng thái:** chờ Codex review mục này và mục 0 PLAN. Approve **không** mở cổng A/B. Theo quyết định người dùng
23/10: xong mốc 5 thì tạm dừng; mốc 6 (Nano) chưa làm.
