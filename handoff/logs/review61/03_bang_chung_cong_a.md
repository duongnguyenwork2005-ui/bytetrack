# R61 cổng A — bằng chứng A0 (lớp vận hành + fixture), 24/10/2026

Nguồn quyền: plan cổng A (A0→A1→A2) người dùng duyệt 24/10/2026; bước hiện tại **A0**. Chưa chạy GPU, cache, smoke,
khóa hay gói thật; phiên A thật `results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/` **chưa tồn tại**. Mục 0 PLAN đã
cập nhật trước báo cáo này (review_006 approve + chính sách crash mới).

## 1. Phạm vi đã sửa

| File | Thay đổi |
|---|---|
| `src/r61_bdd/van_hanh_cong_a.py` | MỚI — runner vận hành cổng A |
| `src/r61_bdd/test_van_hanh_cong_a.py` | MỚI — fixture CPU, tiếp tục được cùng phiên |
| `src/r61_bdd/dieu_phoi_bdd.py` | chỉ `DIEM_VAO_MA` thêm `r61_bdd/van_hanh_cong_a.py` → `ma_nguon` của khóa bao mã runner và mọi file nó import tĩnh |
| `configs/phase2/r61_bdd/hop_dong_lenh.json` | thêm `entry_point` runner/fixture, mục `van_hanh_cong_a` (mã thoát, chính sách crash, đơn vị, log, khóa, gói), `thu_tu.A_van_hanh`; các mục cũ giữ nguyên |
| `handoff/PLAN_DA_CHOT.md` | chỉ mục 0 |
| `results/phase2/r61_bdd/tests/cong_a_20261024/lan_01/` | phiên fixture |

Không sửa helper (`hop_dong_bdd`, `chuan_bi_bdd`, `thong_ke_bdd`, `test_chuan_bi_bdd`, `det_cache`, `run_tracking`…),
`giao_thuc.json`, thuật toán, metric, kết quả PREP, baseline lan_01. Fixture tự kiểm `ma_r61`, `hop_dong_lenh`,
`giao_thuc_that` không đổi trong lúc chạy.

## 2. Thiết kế runner (`van_hanh_cong_a.py`)

Lệnh: `python src/r61_bdd/van_hanh_cong_a.py --cong A --phien <A> --so-doan 1 --ngan-sach-giay 420 --buoc <buoc>
[--detector <det> --device 0] [--gioi-han-don-vi N]`. Bước: `tien-kiem`, `cache` (×3 detector), `kiem-cache`, `smoke`,
`khoa`, `kiem-khoa`, `dong-goi-a`, `kiem-goi-a`.

- **Chặn trước mọi ghi**: chỉ `--cong A`; giao thức thật → phiên phải đúng `cong_a_20261024_01` (không mở phiên mới để
  reset bộ đếm) và cấm `--tiem-loi`; fixture → phiên dưới `tests/cong_a_20261024/`; `so_doan × ngan_sach ≤ 420`;
  bước khác `tien-kiem` cần `van_hanh/phien.json`. Phiên đã có: phải đúng lượt `R61_cong_A_20261024`, giao thức, và
  snapshot mã (đóng import tĩnh `src/`) + `hop_dong_lenh` + giao thức trùng lúc mở phiên, lệch → exit 2.
- **Cha / con**: cha chỉ thư viện chuẩn, không băm dữ liệu lớn; mỗi đoạn chạy MỘT tiến trình con tuần tự. Trước khi chạy
  ghi `van_hanh/lan_chay/<stt>_<buoc>[_<det>]/goi.json` (bước, detector, argv cha/con, snapshot mã, đầu vào: PLAN tách
  mục 0, sha tiền kiểm/khóa/gói, số lần đã bắt đầu của từng đơn vị, bộ đếm crash trước). Con ghi stdout/stderr thẳng ra
  file của lần đó, `faulthandler.log` riêng, nhật ký đơn vị `tien_do.jsonl` (flush + fsync, `bat_dau` có `lan_thu` trước
  khi làm đơn vị). Cha ghi `ket_thuc.json`: exit thô, `exit_hex`, phân loại, đơn vị đang dở, bộ đếm sau, quyết định.
  Cha có faulthandler riêng `van_hanh/cha/`.
- **Phân loại**: 0 hoàn tất, 10 chưa xong, 2 lỗi hợp đồng, 1 + `Traceback` = `loi_python`, 1 không traceback,
  hết giờ cứng (ngân sách + 150 s, không phải exit 10), mã khác = `crash_native`; cha không ghi được `ket_thuc.json` →
  lần gọi sau ghi bổ sung `cha_gian_doan` (tính là crash, không xác định đơn vị).
- **Chính sách crash** (đếm lại từ mọi `ket_thuc.json` mỗi lần gọi, không có biến đếm riêng): crash dưới ngưỡng → exit
  **30** (chạy lại đúng lệnh = thử lại đơn vị); DỪNG (exit **40**, ghi `van_hanh/dung.json`, mọi lần gọi sau trả 40 không
  chạy con) khi đơn vị crash lần 3, tổng crash phiên 4, cùng bước `khoa`/`kiem-khoa`/`dong-goi-a` crash lần 2, crash
  không xác định đơn vị, `loi_python`, exit 1 không traceback, hết giờ, cha gián đoạn. Exit 2 không thử lại tự động.
  (Mã 30/40 chọn để không trùng exit 3 của CRT `abort`/`raise(SIGSEGV)`.)
- **Đơn vị**: cache `cache:<det>:<video>`; kiểm cache `kiem-cache:<det>:<video>` (kết quả từng đơn vị trong
  `kiem_cache/don_vi/`); smoke `smoke:tracking:<run>:<video>` + `smoke:cham:<run>`; khóa/kiểm khóa/đóng gói/kiểm gói
  `<buoc>:bam:<k>` (danh mục cố định lúc bắt đầu, mỗi đoạn một đơn vị) + đơn vị tổng hợp cuối kiểm đủ mọi đoạn, đúng tập
  file từng đoạn và stat lại mọi file (đổi sau khi băm → lỗi). Đơn vị `<buoc>:chuan_bi` (kiểm điều kiện) không tính vào
  giới hạn. Mỗi đoạn làm ≥ 1 đơn vị rồi mới dừng giữa đơn vị khi hết ngân sách mềm / đạt `--gioi-han-don-vi` (exit 10).
- **Marker / artifact dở**: quét mọi marker trước mọi ghi, marker hỏng/hash lệch/danh tính lệch → exit 2. Đơn vị chưa
  có marker mà có file → CHUYỂN nguyên byte vào `van_hanh/artifact_do/<đơn vị>/lan_<stt>/` (`sao_luu.json` sha256
  trước/sau) rồi mới cho helper ghi lại.
- **Helper**: con gọi đúng `dieu_phoi_bdd` (`nap_tien_kiem`, `buoc_tien_kiem`, `spec_cho`, `nap_model`,
  `tao_cache_video`, `xong_cache`, `kiem_cache_video`, `chay_don_vi`, `xong_don_vi`, `cham_run`, `xong_cham`,
  `anh_khoa`, `so_khoa`) và `det_cache.write_meta`; không gọi `chup-cuoi` cũ.
- **Tiền kiểm A**: `tien_kiem_a/chup/` chụp PLAN nguyên byte, git status/diff, 43 nguồn bảo vệ, vùng nguy cơ, pyc,
  settings (cùng nội dung `chuan_bi_bdd.chup`); `so_sanh.json` đối chiếu TRỰC TIẾP `chuan_bi_cpu/lan_01/chup_dau` với
  chênh lệch thô; ngoại lệ DUY NHẤT: mục `.git` trong vùng `.` chỉ đổi mtime (ghi riêng `ngoai_le`); thay đổi khác phân
  loại `pham_vi_a` / `pham_vi_prep` / `to_tien_pham_vi`, còn lại là lỗi; HEAD/nhánh/git status vẫn kiểm. Sau đó
  `buoc_tien_kiem` + `moi_truong.json` (thiết bị CUDA — thật thì bắt buộc có, RAM, đĩa, thư viện, `ma_nguon`).
  Mọi bước sau kiểm PLAN ngoài mục 0 trùng baseline TRƯỚC nạp detector/ghi.
- **Khóa**: `khoa/khoa.json` = `anh_khoa` (cache từ các đoạn băm, đối chiếu sha kiểm-cache; `ma_nguon` gồm runner) +
  `phu_luc_van_hanh` (PLAN `tach_plan` nguyên byte, sha `hop_dong_lenh`, giao thức, `r61_bdd/*.py`, `tien_kiem_a`,
  `kiem_cache`, `smoke`); `ma_nguon` phải trùng snapshot tiền kiểm; `khoa.xong.json`; khóa hợp lệ không ghi đè.
- **Kiểm khóa**: vòng `kiem_khoa/vong_NN/` (trước đóng gói) hoặc `van_hanh/kiem_khoa_sau_goi/vong_NN/` (sau khi có
  `dong_goi/danh_muc.json`, để không ghi vào payload đã chốt).
- **Gói**: payload = `tien_kiem`, `tien_kiem_a`, `cache`, `kiem_cache`, `smoke`, `khoa`, `kiem_khoa`, `van_hanh/phien.json`,
  `van_hanh/artifact_do`, `van_hanh/lan_chay` có stt < lần `dong-goi-a` đầu, + `goi/nhat_ky_crash_chot.json` (bản chốt
  sổ crash). Loại trừ ghi trong `goi_a.json.loai_tru` (nhật ký đang mở, vùng làm việc, `_tam`). Ghi `goi_a.json` →
  `SHA256SUMS` → marker `goi_a.xong.json`; gói chỉ hợp lệ khi marker hợp lệ. `kiem-goi-a` chỉ đọc payload, băm lại theo
  đoạn, kiểm thừa trong gốc payload, `SHA256SUMS`, `khoa_sha256`; log ở `van_hanh/kiem_goi_a/`.
- **Tiêm lỗi**: chỉ `--tiem-loi` + giao thức fixture dưới `tests/cong_a_20261024/` (cha và con đều kiểm). Điểm:
  `khoi_dong`, `truoc_ghi`, `sau_ghi` (artifact đã ghi, marker chưa), `sau_marker`, `giua_goi`, `cha_sau_con`; native =
  `faulthandler._read_null` (access violation thật), python = `RuntimeError`.

## 3. Fixture A0

Lệnh (đúng plan): `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r61_bdd/test_van_hanh_cong_a.py --phien
results/phase2/r61_bdd/tests/cong_a_20261024/lan_01 --so-doan 1 --ngan-sach-giay 420` → **exit 0**, bắt đầu
2026-10-24T19:16:28, 191,5 s, **65/65 kiểm đạt**, 17/17 kịch bản trong một đoạn (không cần lần tiếp tục). 52 lệnh runner
(log từng lệnh `lan_01/lenh/NNN_*.json`): exit 0 = 18, 10 = 3, 2 = 12, 30 = 8, 40 = 10, `3221225477` (0xC0000005) = 1
(cha crash tiêm). Exit con của mọi crash tiêm là `0xc0000005`, `faulthandler.log` có "Windows fatal exception".
Dữ liệu tổng hợp 3 video × 10 frame, detector mô phỏng `ModelGia`, `CUDA_VISIBLE_DEVICES=-1`.
Kết quả: `lan_01/ket_qua_test.json`, `lan_01/kich_ban/*.json`, `lan_01/cli_log.json`.

| Tiêu chí plan | Kiểm (đều DAT) | Phiên con |
|---|---|---|
| Gián đoạn trước ghi artifact | `gian_doan_truoc_ghi_artifact` (exit 30, không có `fxv-a.*`, đủ goi/stdout/stderr/faulthandler), `crash_truoc_ghi_khong_de_artifact` | S_nguong_phien, S_nguong_don_vi |
| Gián đoạn sau ghi artifact | `gian_doan_sau_ghi_artifact_truoc_marker`, `M_crash_sau_ghi` | S_nguong_phien, M |
| Gián đoạn sau marker | `gian_doan_sau_marker_don_vi_hop_le` (marker hợp lệ, crash tính cho đơn vị) | S_nguong_phien |
| Tiếp tục bỏ qua đơn vị xong, không đổi byte | `tiep_tuc_bo_qua_don_vi_hoan_tat_khong_doi_byte`, `M_don_vi_hoan_tat_khong_doi_byte` (sha + mtime_ns) | S_nguong_phien, M |
| Giữ log/faulthandler và artifact dở | `artifact_do_giu_nguyen_sau_crash`; `M_artifact_do_sao_luu_nguyen_byte_roi_ghi_lai` (sao_luu sha == bản dở, lần thử 2) | S_nguong_phien, M |
| Bộ đếm không reset khi khởi động lại | `bo_dem_khong_reset_qua_loi_goi` (lần thử 1,2,3; bộ đếm trước 0,1,2 qua 3 lần gọi riêng) | S_nguong_don_vi |
| Ngưỡng 3 lần / 4 crash / 2 crash bước khóa | `nguong_don_vi_crash_lan_3_dung` (30,30,40; lý do đúng duy nhất "don vi … crash lan 3"), `nguong_tong_crash_phien_4_dung` (lý do duy nhất "tong crash phien 4"), `nguong_buoc_khoa_crash_lan_2_dung` (30,40; lý do duy nhất "buoc khoa crash lan 2", tổng 2); sau ngưỡng mọi lần gọi trả 40 không tạo lần chạy | S_nguong_* |
| Marker hỏng / metadata / hash lệch dừng không retry | `marker_hash_npz_lech_dung_khong_retry`, `metadata_provenance_lech_dung`, `marker_danh_tinh_lech_dung`, `marker_hong_doc_loi_dung`: exit 2, đúng một dòng `[LOI HOP DONG]` chứa nguyên nhân, đúng 1 lần chạy, không đơn vị `chay` nào, bộ đếm crash không đổi | M |
| Exit 1 traceback không gán crash native | `exit1_traceback_khong_gan_crash_native` (phân loại `loi_python`, đơn vị `tien-kiem`, tổng crash 0, dừng 40) | S_loi_python |
| Crash không xác định đơn vị thì dừng | `crash_khong_xac_dinh_don_vi_thi_dung`; cha crash: `cha_crash_giu_trang_thai` + `cha_gian_doan_phat_hien_va_dung` | S_crash_khoi_dong, S_cha_crash |
| Lỗi PLAN ngoài mục 0 chặn trước inference | `plan_ngoai_muc0_chan_truoc_inference` (exit 2, không tạo `cache/yolov8n_mixed5`, không dòng nạp detector) | M |
| Ngoại lệ `.git` không lan | 8 kiểm hàm thuần `git_*`, `plan_ngoai_muc0_doi_la_loi`, `thay_doi_trong_pham_vi_A_khong_loi`, `git_status_ngoai_pham_vi_la_loi` | — |
| Smoke chấm tiếp tục đủ run | `smoke_cham_tiep_tuc_du_run` (exit 10,10,0; 36 tracking; chấm 4→14→36; file chấm cũ không đổi), `smoke_tham_so_seed_timeline_khop` | M |
| Gián đoạn đóng gói không hợp thức hóa gói thiếu | `dong_goi_gian_doan_de_goi_thieu`, `goi_thieu_khong_duoc_hop_thuc_hoa` (kiem-goi-a exit 2), `dong_goi_tiep_tuc_sao_luu_goi_do`, `goi_payload_dung_pham_vi`, `kiem_goi_a_dat_chi_doc_payload`, `goi_hop_le_khong_ghi_de` | M |
| Kiểm khóa phát hiện thay PLAN sau khóa | `kiem_khoa_phat_hien_PLAN_doi_sau_khoa` (sửa trong mục 0 bản PLAN fixture → exit 2 "phu_luc plan: sha256_toan_bo"), `sau_khoi_phuc_kiem_khoa_va_goi_dat` | M |
| Khóa bao runner + hợp đồng; chia đoạn băm | `khoa_bao_ma_runner_va_hop_dong`, `khoa_chia_doan_tiep_tuc_khong_bam_lai` (10 rồi 0, chỉ băm đoạn 1–3), `khoa_hop_le_khong_ghi_de` | M |
| Chỉ cổng A thật / tiêm lỗi chỉ fixture | `that_phien_khac_bi_chan`, `that_tiem_loi_bi_chan_khong_tao_phien`, `cong_B_bi_chan`, `nhieu_doan_420_giay_bi_chan`, `buoc_truoc_tien_kiem_bi_chan` | — |
| Không ghi ngoài phạm vi, không pyc/settings | `khong_ghi_ngoai_pham_vi_pyc_settings` (settings AppData, pyc trong `src/`, `data/interim`, `data/processed`, trackers, `chuan_bi/`, mã r61, hợp đồng, giao thức thật trùng trước/sau; phiên A thật không tồn tại), `khong_file_tam_do_dang`, `khong_lenh_het_gio` | — |

Sổ crash phiên M (luồng đầy đủ): 2 crash tiêm (cache `sau_ghi`, `dong-goi-a` `giua_goi`), `goi/nhat_ky_crash_chot.json`
ghi tổng 2; không có `dung.json`. `git status --short` sau lượt trùng lúc bắt đầu (chỉ các thư mục untracked sẵn có).

## 4. Phát hiện cần quyết định trước A1 (thông tin, không phải tiêu chí A0)

Kịch bản `so_sanh_git` chạy thêm đối chiếu CHỈ ĐỌC trạng thái thật với baseline lan_01, ghi trong phiên test
`lan_01/thong_tin/doi_chieu_that_lan_01.json`: **dat = false**. Ngoài ngoại lệ mtime `.git` có 3 lỗi ngoài phạm vi:

- `pyc moi: src/__pycache__/baseline_track.cpython-311.pyc`;
- mtime thư mục `src/__pycache__` đổi (baseline `1792319780088286200` → `1792815375687713000` ns);
- mtime thư mục `results/track_video` đổi (`1789967131660364700` → `1792834519649407700` ns).

Các thay đổi này có trước khi fixture chạy (ảnh chụp trước/sau fixture trùng nhau); nguồn gốc **CHƯA XÁC MINH**
(không đọc event log/lịch sử tiến trình). Theo plan, `tien-kiem` A1 chỉ đạt khi ngoài ngoại lệ `.git` không có chênh lệch
nào khác → nếu giữ nguyên, A1 sẽ dừng exit 2 ở tiền kiểm. Agent **không** xóa pyc hay sửa runner để miễn kiểm (ngoài
write_paths / đổi tiêu chí). Người dùng cần quyết định (ví dụ: cách xử lý pyc và hai mtime, hoặc chấp nhận thêm ngoại lệ
có ghi rõ) trước khi mở A1.

## 5. Giới hạn

- Fixture dùng dữ liệu tổng hợp nhỏ và detector mô phỏng; không đo thời gian/bộ nhớ thật (A1 đo). Crash tiêm là access
  violation tổng hợp tại điểm định sẵn, không tái hiện crash ngẫu nhiên của PREP (nguyên nhân vẫn CHƯA RÕ).
- Chưa có ca kiểm tự động cho: hết giờ cứng, exit 1 không traceback, thiếu log (mã phân loại có, chưa tiêm), crash giữa
  lúc chuyển artifact dở, crash trong đơn vị tổng hợp băm. Fixture chạy 1 lần (lan_01), không lặp để đo tần suất crash.
- Chia đoạn băm phát hiện file đổi sau khi băm bằng stat (size, mtime_ns); thay đổi giữ nguyên cả size lẫn mtime không bị
  phát hiện ở bước tổng hợp đó (kiểm khóa / kiểm gói sau sẽ băm lại).
- Khóa/kiểm khóa/đóng gói tính lại `anh_khoa` trong một đơn vị (nguồn, thư viện, Git) — không chia đoạn phần này.
- Kiểm tra tĩnh cú pháp chỉ qua `--help`/chạy fixture; không có linter. Tự kiểm (Tier 1) của agent; Codex chưa review.

## 6. Bước tiếp

Codex review A0 (runner, fixture, hợp đồng, sửa `dieu_phoi_bdd.py`, báo cáo này). A1 chỉ mở sau review chấp nhận và
quyết định của người dùng ở mục 4. Không có phiên A/B thật; chưa commit.

## 7. Vòng sửa A0-F1…F10 (24/10/2026, sau Codex review_001 `revise`)

Nguồn finding: `D:/LABOTARY/automation/runs_local/r61_bdd_cong_a_20261024_20261024_182327/reports/review_001.json`.
Mục 1–6 ở trên là bằng chứng vòng đầu (mã cũ runner `5569b7b1…`, fixture `69638495…`), giữ nguyên làm lịch sử, **không**
chứng nhận mã mới. Mục 0 PLAN cập nhật trước mục này. Vòng này chỉ sửa `van_hanh_cong_a.py`, `test_van_hanh_cong_a.py`,
`hop_dong_lenh.json`, mục 0 PLAN và báo cáo này; `dieu_phoi_bdd.py`, helper `hop_dong_bdd.py`/`chuan_bi_bdd.py`,
`giao_thuc.json`, baseline lan_01 và bằng chứng cũ **không sửa**. Ba chênh lệch baseline (mục 4) **không xử lý, không miễn**.

### 7.1 Lệnh và kết quả (tự kiểm Tier 1)

| Lệnh | Exit | Kết quả |
|---|---|---|
| `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r61_bdd/test_van_hanh_cong_a.py --phien khong_hop_le --so-doan 1 --ngan-sach-giay 420` | 2 | kiểm import/cú pháp; từ chối `--phien` ngoài vùng fixture, không ghi |
| `… test_van_hanh_cong_a.py --phien results/phase2/r61_bdd/tests/cong_a_20261024/lan_01 --so-doan 1 --ngan-sach-giay 420` | 0 | mở `lan_01/vong_02/`; **104/104 kiểm đạt**, 20 kịch bản, 63 lệnh runner, một lời gọi |
| cùng lệnh, gọi lại | 0 | chọn lại `vong_02` (cùng snapshot), không chạy lại kịch bản, đọc `ket_qua_test.json` đã có |

`vong_02/ket_qua_test.json`: exit lệnh runner 0/10/2/30/40/`0xC0000005` = 18/3/19/8/13/2; `kich_ban_sai_snapshot = []`;
snapshot vòng = snapshot hiện tại = `840cb7203941a8b539855055c5eb937a42bdd49dead4dd5f1170d9f5ce38cebd`. Hash mã đã chạy
(trong `vong_02/vong.json`): `van_hanh_cong_a.py` `b1ef34093b64a2cf8b33bd556dec1dc045fd4b8731725f9e397702d28fbd9d46`,
`test_van_hanh_cong_a.py` `0dbf2b01c07f91084f9077c15be2c17b6377bf74dfaae5e1a628a56aaa62f500`, `dieu_phoi_bdd.py`
`62ff2d82…`, `hop_dong_bdd.py` `4ac3c015…`, `chuan_bi_bdd.py` `3f6a8633…`, `hop_dong_lenh.json`
`250a72a10686537f275a1f681c98ea79caec1ccb680be699b87ee931a7be7334`, `giao_thuc.json` thật `82ffe7ae…`. Không có tiến trình
nào còn chạy sau hai lời gọi (fixture chạy foreground; con trong kịch bản F4 đã xác nhận chết).

### 7.2 Ánh xạ finding → sửa → bằng chứng mới (`vong_02/`)

| Finding | Sửa (runner trừ khi ghi khác) | Bằng chứng (kiểm / lệnh trong `vong_02/lenh/`) |
|---|---|---|
| F1 smoke trước khóa | `can_smoke(ctx, A)`: marker tổng hợp + MỌI marker tracking (36 run × video) và chấm (36 run) qua `D.xong_don_vi`/`D.xong_cham` (băm artifact + danh tính), số đếm đúng kỳ vọng, băm marker trùng `marker_sha256` mới ghi trong `ket_qua_smoke.json`; gọi ở `khoa` và `dong-goi-a` | `khoa_chan_smoke_thieu_marker_tracking` (lệnh 049: xóa 1 marker tracking → exit 2 "smoke thieu 1 marker …", không tạo `khoa/`); `khoa_chan_smoke_cham_hash_lech` (050: sửa `per_video_metrics.csv` → exit 2 "marker …cham.xong.json hong … bam lech per_video_metrics.csv") |
| F2 manifest/artifact phục hồi | gói dở (`goi/`, `goi_a.json`, `SHA256SUMS`, `.dang_ghi`) + vùng băm `dong_goi/` cũ sao lưu vào `artifact_do/dong-goi-a__goi_do/` TRƯỚC khi lập danh mục mới → bản sao lưu nằm trong payload băm; `kiem-goi-a` tìm file thừa trên toàn phạm vi `payload()` + `goi/` | `dong_goi_tiep_tuc_sao_luu_goi_do`, `manifest_bao_du_artifact_phuc_hoi` (mọi file `artifact_do` ⊆ `goi_a.json.file`, gồm bản sao `goi_a.json`, `sao_luu.json`, `goi/nhat_ky_crash_chot.json`); `kiem_goi_phat_hien_file_thua_trong_artifact_do` (058: file cài vào `artifact_do` → exit 2 "thua: …"); `kiem_goi_a_dat_chi_doc_payload` (059) |
| F3 thiếu log | `quyet_dinh`: `thieu_log` khác rỗng → DỪNG (40) bất kể exit/phân loại; ghi kèm phân loại thô | `thieu_log_exit0_khong_nhan_hoan_tat` (017: con exit 0, `phan_loai=hoan_tat`, `thieu_log=[faulthandler.log]` → 40, lý do "thieu log…", crash 0); `thieu_log_crash_dung_khong_thu_lai` (018: crash native có đơn vị nhưng thiếu log → 40, không 30) |
| F4 cha crash khi con chạy | con gắn Job Object `KILL_ON_JOB_CLOSE` ngay sau `Popen` (không gắn được → giết con, DỪNG); `con.json` (pid, thời điểm tạo, kết quả gắn job) ghi trước khi chờ; lần sau `trang_thai_con` đối chiếu pid + thời điểm tạo, ghi `con_trang_thai` rồi DỪNG | `cha_crash_khi_con_chay_con_bi_giet` (015: cha access violation sau 4 s trong lúc con ngủ `[CHO]`; con được xác nhận đã chết, không có `ket_qua_con.json`); `tiep_tuc_xac_nhan_con_da_chet_roi_dung` (016: `cha_gian_doan`, `con_song=false`, lý do "con da ket thuc…", 40) |
| F5 `--con`/`--phien` sai | `kiem_lan_con` xác thực đường `<phien>/van_hanh/lan_chay/<stt>_<buoc>`, phiên A thật/fixture, reparse, `goi.json`+`phien.json` cùng lượt/phiên/giao thức, lần chạy chưa dùng — TRƯỚC khi mở `faulthandler.log`; cha kiểm `phien.json` trước mọi ghi, `phien.json` là file đầu tiên của phiên mới; `goi_cha.jsonl` chỉ ghi khi `PHIEN_CHAP_NHAN` | `con_duong_ngoai_vung_khong_ghi` (007), `con_goi_sai_phien_khong_ghi` (008: cây thư mục trước = sau), `phien_sai_luot_khong_ghi_goi_cha` (009: không `goi_cha.jsonl`, `_tam`, `cha/`) |
| F6 `.dang_ghi` | `ghi_bao_toan`: kết quả kiểm/băm (đơn vị + tổng hợp kiem-cache, danh mục/đoạn băm, kiem-khoa, kiem-goi-a) sao lưu bản cũ + `.dang_ghi` rồi ghi không đè; `kem_dang_ghi` cho artifact tracking/khóa/smoke; `sao_luu_do` không ghi đè lần sao lưu trước (`lan_<stt>_<k>`) | `bao_toan_dang_ghi_va_ket_qua_cu_khi_ghi_lai` (045: `.dang_ghi` đơn vị, `ket_qua.json` cũ chưa marker, `ket_qua.json.dang_ghi` → sao lưu đúng sha256 trước/sau); `bao_toan_dang_ghi_doan_bam` (052) |
| F7 nội dung/index Git | `chup_a` ghi thêm `git_diff_cached.patch`; `so_sanh_chup` so băm từng khối `diff --git` (ngoài phạm vi → lỗi) và index (`git diff --cached`; mọi đổi → lỗi; baseline lan_01 thiếu `--cached` → suy ra rỗng khi không có dòng staged, ngược lại lỗi) | `git_ngoai_le_khong_mien_noi_dung_diff`, `git_diff_doi_trong_pham_vi_khong_loi`, `git_ngoai_le_khong_mien_index`, `baseline_thieu_index_co_staged_la_loi` (hàm thuần); `M_tien_kiem_doi_chieu_baseline` (034, tiền kiểm fixture thật qua git) |
| F8 snapshot thực chạy | fixture chạy theo vòng `vong_NN/` với `vong.json` = snapshot (đóng import tĩnh src + mọi `r61_bdd/*.py` + hợp đồng + giao thức + Python); mã khác → vòng mới; mỗi kịch bản ghi `snapshot_sha256` trước/sau; tổng hợp chỉ đạt khi mọi kịch bản cùng snapshot = snapshot hiện tại; không ghi đè `ket_qua_test.json` của vòng | `vong_02/vong.json`; 20 kiểm `<kich_ban>_ma_khong_doi_trong_kich_ban`; `kich_ban_sai_snapshot = []`; gọi lại chọn đúng `vong_02` |
| F9 một đoạn | `--so-doan` ≠ 1 → exit 2; `0 < --ngan-sach-giay ≤ 420`; vòng lặp nhiều đoạn bỏ; CLI fixture cũng bắt `--so-doan 1` | `hai_doan_210_giay_bi_chan` (005: `--so-doan 2 --ngan-sach-giay 210` → exit 2, không tạo phiên); `nhieu_doan_420_giay_bi_chan` (004) |
| F10 mục 0 | đính chính: Pbec787ae và P5d658ee9 là **reject**; ghi chú cũ "CHẤP NHẬN ngoại lệ" là sai, đã thay; không có ngoại lệ nào ngoài mtime `.git` | `handoff/PLAN_DA_CHOT.md` mục 0 (dòng "3 chênh lệch … CHƯA có quyết định xử lý") |

Hồi quy chức năng cũ vẫn đạt trong `vong_02` (ngưỡng 3 lần/4 crash/2 crash bước khóa, marker hỏng/metadata/hash lệch
dừng không retry, exit 1 traceback ≠ crash native, crash không xác định đơn vị, PLAN ngoài mục 0 chặn trước inference,
smoke chấm tiếp tục đủ 36 run, gói dở không được hợp thức hóa, kiểm khóa phát hiện PLAN đổi sau khóa, không ghi ngoài
phạm vi/pyc/settings). Sổ crash phiên M vẫn 2 (`goi/nhat_ky_crash_chot.json`).

Đối chiếu chỉ đọc trạng thái thật với lan_01 (`vong_02/thong_tin/doi_chieu_that_lan_01.json`): `dat=false`, đúng 3 lỗi của
mục 4 (pyc + mtime `src/__pycache__` + mtime `results/track_video`), ngoại lệ chỉ `.git`; phần đối chiếu mới về nội dung
`git diff`/index không thêm lỗi. **Không** xử lý, miễn hay đổi tiêu chí cho 3 lỗi này.

### 7.3 Giới hạn của vòng sửa

- F4: còn khoảng hở rất ngắn giữa `Popen` và `AssignProcessToJobObject` (con chưa thuộc job; con không sinh tiến trình
  cháu trong khoảng đó). Nhánh "không gắn được job" và nhánh `con_song = null` (không mở được tiến trình) chưa tiêm lỗi;
  cả hai dẫn tới DỪNG. Kiểm "con đã chết" dựa trên `OpenProcess`/`GetProcessTimes`.
- F3: fixture chỉ tiêm thiếu `faulthandler.log`; `stdout.txt`/`stderr.txt` do cha tạo trước khi chạy con nên không tiêm
  được thiếu hai file này mà không sửa cha.
- F7: baseline lan_01 không có `git_diff_cached.patch`; index baseline được suy ra rỗng từ `git_status_short.txt` (không có
  dòng staged). Tách khối diff theo dòng `diff --git`, đường lấy sau ` b/` (tên có ký tự đặc biệt chưa kiểm).
- F2: sau gián đoạn đóng gói, lần tiếp tục băm lại TOÀN BỘ payload (chi phí A2 tăng tương ứng kích thước cache).
- Vẫn chưa có ca tiêm cho hết giờ cứng, exit 1 không traceback (đề xuất P5d658ee9 bị reject, không thêm).
- Fixture dữ liệu tổng hợp, detector mô phỏng, chạy một vòng; không đo tần suất crash. Tự kiểm của agent; Codex chưa
  review lại.

### 7.4 Bước tiếp

Codex review lại A0 (mã runner/fixture/hợp đồng ở hash mục 7.1, `vong_02/`, mục 0 PLAN, mục 7 này). A1 chỉ mở sau review
chấp nhận **và** quyết định của người dùng về 3 chênh lệch baseline. Chưa commit.
*(Đã làm: review_002 `revise` → mục 8.)*

## 8. Vòng sửa A0-F2a/F2b/F4/F6 (25/10/2026, sau Codex review_002 `revise`)

Nguồn finding: `D:/LABOTARY/automation/runs_local/r61_bdd_cong_a_20261024_20261024_182327/reports/review_002.json`
(F1, F3, F5, F7–F10 đã đóng; F2, F4, F6 còn nhánh). Mục 0 PLAN cập nhật trước mục này. Chỉ sửa `van_hanh_cong_a.py`,
`test_van_hanh_cong_a.py`, `hop_dong_lenh.json`, mục 0 PLAN, báo cáo này. `dieu_phoi_bdd.py` (`62ff2d82…`), helper
`hop_dong_bdd.py` (`4ac3c015…`)/`chuan_bi_bdd.py` (`3f6a8633…`), `giao_thuc.json` thật (`82ffe7ae…`), baseline lan_01,
`vong_02` và bằng chứng cũ **không sửa**. Ba chênh lệch baseline (mục 4) **không xử lý, không miễn**.

### 8.1 Lệnh và kết quả (tự kiểm Tier 1)

| Lệnh | Exit | Kết quả |
|---|---|---|
| `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r61_bdd/test_van_hanh_cong_a.py --phien results/phase2/r61_bdd/tests/cong_a_20261024/lan_01 --so-doan 1 --ngan-sach-giay 420` | 10 | mở `lan_01/vong_03/` (snapshot mới); 21 kịch bản (00–20) đạt, dừng giữa hai kịch bản theo ngân sách |
| cùng lệnh, gọi lại | 0 | tiếp tục cùng `vong_03`, chạy kịch bản 21–23; tổng **122/122 kiểm đạt**, 24 kịch bản, 82 lệnh runner |

`vong_03/ket_qua_test.json`: exit lệnh runner 0/10/2/30/40/`3221225477` (0xC0000005, cha crash tiêm) = 31/3/19/12/14/3;
`kich_ban_sai_snapshot = []`; snapshot vòng = snapshot hiện tại = `d6c355a2b843535188c513d55f343c690cb34f02894d8a0ca53313977046348e`.
Hash mã đã chạy: `van_hanh_cong_a.py` `65393cbb53b3fe2f088e7aa7f123037f85a69175893f693d499309d4b0b18f28`,
`test_van_hanh_cong_a.py` `e433d8553201940165c0d6fd8788d7f9786acd32aae9f7df1e2dd013db5c37bb`, `hop_dong_lenh.json`
`94e29294406ee89468ba84eed26cd3a32b57d4311d538c088e99d51173595a08`. Cả hai lời gọi chạy foreground và đã trả về; không có
tác vụ nền. `git status --short` sau lượt trùng lúc bắt đầu.

### 8.2 Ánh xạ finding → sửa → bằng chứng mới (`vong_03/`)

Phiên fixture mới `vong_03/N/` (crash mới không cộng dồn vào M — M đã có 3 crash, thêm nữa sẽ chạm ngưỡng 4). Mọi gián
đoạn dưới đây là crash thật (`faulthandler._read_null`) qua điểm tiêm fixture `dang_ghi`: nửa đầu bytes đã vào
`<file>.dang_ghi` (fsync), chưa `os.replace` — đúng trạng thái `H.ghi_bytes` để lại nếu chết giữa chừng.

| Finding | Sửa | Bằng chứng (kiểm / lệnh `vong_03/lenh/NNN_*`) |
|---|---|---|
| F2a gián đoạn ghi đoạn băm `dong-goi-a` | sau bước gói dở, `nk.goc_do = dong_goi/phuc_hoi/`: sao lưu danh mục/đoạn băm/marker dở vào đây (ngoài `payload()`), danh mục cố định không đổi, chỉ băm đoạn còn thiếu; bước ghi gói băm từng file `dong_goi/phuc_hoi/` vào `goi_a.json.file` + `phuc_hoi_dong_goi`; `kiem-goi-a` tìm thừa cả vùng này | `dong_goi_crash_giua_ghi_doan_bam_de_dang_ghi` (075: exit 30, đơn vị `dong-goi-a:bam:0001`, có `0001.json.dang_ghi`, chưa có `goi_a.json`); `dong_goi_tiep_tuc_sau_gian_doan_doan_bam_hoan_tat` (076: exit 0, marker gói hợp lệ, `danh_muc.json` và `doan/0000.json` không đổi byte/mtime, đoạn băm đầu tiên của lần tiếp là `0001` — không băm lại `0000`, bản sao `.dang_ghi` đúng sha256, `van_hanh/artifact_do` không đổi); `manifest_bao_du_phuc_hoi_dong_goi` (mọi file `dong_goi/phuc_hoi/` ∈ `goi_a.json.file` với sha đúng, `file` = danh mục + phục hồi + nhật ký crash chốt, `payload()` = danh mục); `kiem_goi_dat_sau_phuc_hoi_doan_bam` (077: exit 0, `so_thua` 0) |
| F2b phục hồi kiểm sau đóng gói đổi payload | `kiem-goi-a` và `kiem-khoa` đặt `nk.goc_do = <vòng>/phuc_hoi/` (sau gói: `van_hanh/kiem_goi_a/vong_NN/`, `van_hanh/kiem_khoa_sau_goi/vong_NN/` — đều trong `loai_tru`) | `kiem_goi_crash_giua_ghi_doan_bam_de_dang_ghi` (078: exit 30, `kiem-goi-a:bam:0000`); `kiem_goi_tiep_tuc_giu_nguyen_payload` (079: exit 0, dat, `so_thua` 0, bản sao `.dang_ghi` trong `vong_02/phuc_hoi/` đúng sha, **toàn bộ** payload ∪ `goi/` ∪ `dong_goi/phuc_hoi/` ∪ `goi_a.json.file`: danh mục + sha256 từng file + (sha, mtime) của `goi_a.json`/`SHA256SUMS`/`goi_a.xong.json` + danh mục `artifact_do` trùng trước gián đoạn); `kiem_khoa_sau_goi_crash_giua_ghi_marker` (080) + `kiem_khoa_sau_goi_phuc_hoi_marker_giu_nguyen_payload` (081, cùng phép so payload); `N_kiem_goi_cuoi_dat_khong_thua_khong_dung` (082: exit 0, payload vẫn trùng, tổng crash N = 3, không `dung.json`) |
| F4 cha crash trước khi con được gắn job | cha tự `AssignProcessToJobObject` (kill-on-close) TRƯỚC khi tạo con, kết quả ghi `goi.json.job_cha`; không gắn được → không tạo con, DỪNG; con tạo `CREATE_SUSPENDED` (kế thừa job ngay lúc `CreateProcess`); cha kiểm `IsProcessInJob`, ghi `con.json`, rồi `NtResumeProcess`; lỗi ở bất kỳ bước nào → giết con, DỪNG. Lần gọi sau thiếu `con.json` ghi `file_con` + lý do và DỪNG | `cha_crash_truoc_gan_job_con_khong_song_sot` (017: cha access violation ngay sau tạo con, trước `con.json`; pid + thời điểm tạo con lấy từ stderr tiêm của cha; con xác nhận đã chết qua `OpenProcess`/`GetProcessTimes`; không `con.json`, không `faulthandler.log`/`tien_do.jsonl`, `stdout.txt` rỗng → con chưa chạy lệnh nào); `tiep_tuc_sau_cha_crash_truoc_con_json_dung` (018: exit 40, `cha_gian_doan`, `con_trang_thai.ly_do = "thieu con.json"`, lý do có "TAM DUNG", không tạo lần chạy mới, crash 1). Hồi quy: `cha_crash_khi_con_chay_con_bi_giet` (015), `tiep_tuc_xac_nhan_con_da_chet_roi_dung` (016), `cha_crash_giu_trang_thai` (013) vẫn đạt |
| F6 mất `.dang_ghi` của marker | `ghi_marker_bt`: mọi marker do lớp vận hành ghi (tien-kiem, kiem-cache, smoke, khoa, kiem-khoa, goi_a, kiem-goi-a) sao lưu `<marker>.dang_ghi` (nguyên byte, sha trước/sau) trước khi gọi `H.ghi_marker`; gói dở tính cả `goi_a.xong.json.dang_ghi`; marker cache/tracking/chấm do helper ghi đã nằm trong danh sách sao lưu đơn vị (`<v>.*`, `file_don_vi_do`, cả thư mục chấm) | `crash_giua_ghi_marker_khoa_de_dang_ghi` (054: phiên M, exit 30 tại `khoa:ghi`, có `khoa.xong.json.dang_ghi`); `bao_toan_dang_ghi_marker_khoa` (055: exit 0, bản sao marker dở đúng sha256 cả trong `sao_luu.json` lẫn file đích, `khoa.json` cũ cũng sao lưu đúng sha, lần thử `khoa:ghi` = 2, marker mới hợp lệ); sau gói: `kiem_khoa_sau_goi_phuc_hoi_marker_giu_nguyen_payload` (081: bản sao `xong.json.dang_ghi` + `ket_qua.json` cũ đúng sha trong `<vòng>/phuc_hoi/`, payload không đổi — không gây hai lỗi payload của F2a/F2b) |

Hồi quy chức năng cũ vẫn đạt trong `vong_03` (cùng các kiểm của mục 7.2; phiên M nay có 3 crash: cache `sau_ghi`, khoa
marker, `dong-goi-a` `giua_goi`; `goi/nhat_ky_crash_chot.json` tổng 3, chưa chạm ngưỡng). Đối chiếu chỉ đọc trạng thái
thật với lan_01 (`vong_03/thong_tin/doi_chieu_that_lan_01.json`): `dat=false`, đúng 3 lỗi mục 4, ngoại lệ chỉ `.git`.

### 8.3 Giới hạn

- F4: "con chưa chạy lệnh nào" chứng minh gián tiếp (không có file con tạo, stdout rỗng, pid đã chết); nhánh
  `IsProcessInJob`/`NtResumeProcess` thất bại và `AssignProcessToJobObject(cha)` thất bại chưa tiêm lỗi (đề xuất
  P48178d7c bị reject) — mã dẫn tới giết con/không tạo con và DỪNG. Cha giờ thuộc job riêng: mọi tiến trình con cha tạo
  đều bị giết khi cha kết thúc (cha chỉ tạo đúng một con).
- F2a: gián đoạn ở bước ghi gói (`goi/`, `goi_a.json`, `SHA256SUMS`) vẫn đi đường gói dở → băm lại toàn bộ payload (mục
  7.3); gián đoạn ở đoạn băm chỉ băm phần còn thiếu.
- Điểm tiêm `dang_ghi` chỉ đặt ở `ghi_bao_toan`/`ghi_marker_bt` của lớp vận hành (helper giữ nguyên byte), mô phỏng trạng
  thái file của `H.ghi_bytes` chứ không crash bên trong helper.
- Các giới hạn khác của mục 7.3 giữ nguyên (thiếu `stdout`/`stderr`, hết giờ cứng, exit 1 không traceback chưa tiêm).
  Fixture dữ liệu tổng hợp, detector mô phỏng, một vòng; không đo tần suất crash. Tự kiểm của agent; Codex chưa review.

### 8.4 Bước tiếp

Codex review lại A0 (hash mục 8.1, `vong_03/`, mục 0 PLAN, mục 8). A1 chỉ mở sau review chấp nhận **và** quyết định của
người dùng về 3 chênh lệch baseline. Chưa commit.
*(Đã làm: review_003 `approve` A0 → mục 9.)*

## 9. A1 — BLOCKED trước lệnh đầu (25/10/2026)

A0 được Codex approve: `D:/LABOTARY/automation/runs_local/r61_bdd_cong_a_20261024_20261024_182327/reports/review_003.json`
(verdict `approve`, `issues` rỗng; limitations ghi rõ "A1 vẫn blocked bởi ba chênh lệch baseline chưa có quyết định xử lý;
không miễn kiểm"). Mục 0 PLAN cập nhật trước mục này.

**Không chạy lệnh nào của A1** (không `tien-kiem`, không cache/GPU, không smoke); phiên A thật
`results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/` **vẫn chưa tồn tại**. Lý do:

1. Ba chênh lệch so với baseline lan_01 (mục 4) vẫn còn: pyc `src/__pycache__/baseline_track.cpython-311.pyc` còn tồn tại
   (Glob lúc lượt này), mtime thư mục `src/__pycache__` và `results/track_video` (đối chiếu chỉ đọc gần nhất
   `tests/cong_a_20261024/lan_01/vong_03/thong_tin/doi_chieu_that_lan_01.json`: `dat=false`, đúng 3 lỗi này). Quyết định
   người dùng hiện hành: Pbec787ae và P5d658ee9 là **reject**, không có ngoại lệ nào ngoài mtime `.git`.
2. Tiêu chí A1 chỉ cho đạt tiền kiểm khi ngoài ngoại lệ `.git` không có chênh lệch khác → `tien-kiem` chắc chắn exit 2.
3. Runner ghi `van_hanh/phien.json` (snapshot mã/hợp đồng/giao thức) là file **đầu tiên** của phiên A cố định
   (`src/r61_bdd/van_hanh_cong_a.py:1438–1440`) trước đối chiếu baseline; mọi lần gọi sau so snapshot đó và trả exit 2 nếu
   mã/hợp đồng đổi (`:1478–1481`). Hai mtime thư mục không thể trở về giá trị baseline (xóa pyc cũng đổi mtime), nên mọi
   hướng xử lý đều cần ngoại lệ được duyệt và nhiều khả năng sửa runner/hợp đồng → phiên A đã ghim sẽ không dùng tiếp
   được, trong khi plan cấm tự thay phiên. Chạy `tien-kiem` bây giờ chỉ tạo ra một phiên A hỏng, không thêm thông tin.

Kiểm đã chạy lượt này (chỉ đọc): `git status --short -- results/phase2/r61_bdd src/r61_bdd configs/phase2/r61_bdd handoff`
(exit 0, trùng trạng thái đầu lượt); Glob `src/__pycache__/*.pyc` (có `baseline_track.cpython-311.pyc`); Glob phiên A
(không có file). Chưa chạy: toàn bộ lệnh A1; không đo thời gian/bộ nhớ thật.

**Cần người dùng quyết định** cách xử lý 3 chênh lệch (ví dụ duyệt ngoại lệ ghi rõ cho đúng 3 mục, kèm việc có cho phép
sửa runner/hợp đồng tương ứng và kiểm lại A0 trước khi mở phiên A hay không). Agent không xóa pyc, không sửa mtime, không
miễn kiểm. Chưa commit.

## 10. Bước gộp Pe27e304a (ngoại lệ 3 mục + A1) — DỪNG HỎI giữa cache (25/10/2026)

Mục 0 PLAN cập nhật trước mục này.

### 10.1 Quyết định ngoại lệ (Pe27e304a, người dùng duyệt qua email, thay Pbec787ae)

Duyệt ngoại lệ cho **đúng 3 mục** so với baseline lan_01: (1) pyc mới `src/__pycache__/baseline_track.cpython-311.pyc`
(tạo 24/10 11:16:15 khi người dùng chạy lệnh live không `-B`); (2) mtime thư mục `src/__pycache__`; (3) mtime thư mục
`results/track_video` (16:35, phiên điều phối chuyển output live ra ngoài repo). Không xóa/sửa file, không đổi mtime,
không thay baseline. Danh sách ngoại lệ (đường dẫn + loại chênh lệch) nằm trong `configs/phase2/r61_bdd/hop_dong_lenh.json`;
runner `van_hanh_cong_a.py` áp đúng danh sách đó, ngoài ra ngoại lệ duy nhất vẫn là mtime thư mục `.git`.

**Đính chính (review_004 issue 3):** mục 9 (ý 3) và mục 0 PLAN trước đây viết mtime "không thể trở về giá trị baseline"
và chạy `tien-kiem` "chỉ tạo ra một phiên A hỏng". Hai câu này diễn đạt quá mức: mã chỉ chứng minh phiên ghim snapshot
mã/hợp đồng/giao thức ở `phien.json` và chặn (exit 2) khi các thứ đó đổi; không chứng minh mọi hướng xử lý đều cần sửa mã.
Kết luận blocked lúc đó không đổi; câu diễn đạt được rút lại.

### 10.2 Fixture lan_02 (kiểm lại A0 với mã mới)

`results/phase2/r61_bdd/tests/cong_a_20261024/lan_02/vong_02/ket_qua_test.json`: `dat: true`, `so_kich_ban: 25`, gồm ca
ngoại lệ duyệt (`kich_ban/02_ngoai_le_duyet.json`, `thong_tin/doi_chieu_that_ngoai_le_duyet.json`) và đầy đủ hồi quy A0
(lỗi Python exit 1, crash không đơn vị, cha crash các thời điểm, thiếu log, ngưỡng đơn vị/phiên/khóa, chuỗi đầy đủ M/N
qua khóa → kiểm khóa → đóng gói → kiểm gói). Snapshot mã fixture (`vong_02/vong.json`): runner `bbd55797…`,
hợp đồng `7612ca26…`, giao thức thật `82ffe7ae…`. Lệnh chạy và exit của fixture thuộc lời gọi Claude trước (bị ngắt
trước khi trả báo cáo) — lời gọi này chỉ đọc kết quả, không chạy lại fixture.

### 10.3 A1 thật — tiến độ và điểm dừng

Phiên `results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/` (`van_hanh/phien.json` tạo 02:48:30, snapshot runner
`bbd55797…`, hợp đồng `7612ca26…`, giao thức `82ffe7ae…` — trùng snapshot fixture lan_02). Sổ lần chạy
(`van_hanh/so_crash.json`):

| stt | bước | exit | phân loại | giây |
|---|---|---|---|---|
| 0001 | tien-kiem | 0 | hoan_tat | 14,2 |
| 0002–0004 | cache yolov8n_coco | 10, 10, 0 | chua_xong, chua_xong, hoan_tat | 424,2 / 421,6 / 120,6 |
| 0005–0007 | cache yolov8n_mixed5 | 10, 10, 0 | chua_xong, chua_xong, hoan_tat | 425,3 / 423,1 / 112,6 |
| 0008–0009 | cache yolo11n_mixed5 | 10, 10 | chua_xong | 425,0 / 426,1 |
| 0010 | cache yolo11n_mixed5 | — (không có) | **cha_gian_doan** | — |

- Tiền kiểm `tien_kiem_a/so_sanh.json`: `dat: true`; `ngoai_le` = mtime thư mục `.git`; `ngoai_le_duyet` ghi riêng đúng 3
  mục kèm lỗi thô (`pyc moi: …`, `vung nguy co doi ngoai pham vi: …`) và thông tin file pyc (sha256 `9aec6dab…`, 19.092 byte).
- Marker cache: yolov8n_mixed5 200/200 (Glob `*.xong.json`); yolov8n_coco bước hoàn tất exit 0 (chưa đếm lại riêng lượt
  này); yolo11n_mixed5 **182/200**. Chưa chạy `kiem-cache` nên chưa xác nhận 39.973 frame/detector, schema, provenance.
- **Điểm dừng:** lời gọi Claude trước bị orchestrator ngắt khi lần 0010 đang chạy đơn vị
  `cache:yolo11n_mixed5:b24d283f-33783d1b` (dòng cuối `tien_do.jsonl` là `bat_dau`, chưa có artifact của đơn vị này trong
  `cache/yolo11n_mixed5/`). Lời gọi này chạy đúng lệnh tiếp tục
  `… van_hanh_cong_a.py --cong A --phien results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01 --so-doan 1 --ngan-sach-giay 420 --buoc cache --detector yolo11n_mixed5 --device 0`
  → **exit 40**, stderr `[DUNG HOI] phien A da dung: ['cha gian doan o lan chay 10', 'con da ket thuc (xac nhan qua pid + thoi diem tao)']`.
  Runner ghi bổ sung `lan_chay/0010_cache_yolo11n_mixed5/ket_thuc.json` (`phan_loai: cha_gian_doan`, con pid 7400 đã kết
  thúc) và `van_hanh/dung.json` (04:30:10). Bộ đếm: tổng 1/4, bước cache 1, đơn vị không xác định (cha gián đoạn không gắn
  đơn vị). Không có lần chạy mới nào được tạo; không còn tiến trình chạy.
- Đây là hành vi A0 đã duyệt ("cha gián đoạn → dừng hỏi"), không phải crash native của inference: nguyên nhân là lời gọi
  agent bị ngắt (giới hạn thời gian lời gọi), không có traceback/faulthandler của con (faulthandler cha các lần đều 0 byte).
  Agent không xóa `dung.json`, không mở phiên mới, không sửa runner.

### 10.4 Chưa làm / giới hạn

Chưa chạy: 18 video còn lại của yolo11n_mixed5, `kiem-cache`, smoke 36 run × 3 video và chấm, đo thời gian/tài nguyên
tổng hợp (mới có thời gian từng đơn vị trong `tien_do.jsonl`, ví dụ yolo11n_mixed5 ~5,0–5,4 s/video). Chưa khóa/đóng gói.
Chưa commit.

### 10.5 Cần người dùng quyết định

Cách tiếp tục phiên A sau `dung.json` do cha bị ngắt từ bên ngoài (crash đếm 1/4, chưa vượt ngưỡng). Agent không tự gỡ
khóa dừng. Lưu ý vận hành: mỗi lời gọi Claude có giới hạn thời gian ~1 giờ trong khi phần còn lại của A1 (18 video cache,
kiểm cache, smoke và chấm) cần nhiều đoạn 420 s; nếu tiếp tục, nên giới hạn số đoạn mỗi lời gọi để lời gọi không bị ngắt
giữa đoạn.

*(Đã có quyết định: Pd8810cda — xem mục 11.)*

## 11. A1 tiếp tục sau gỡ dừng Pd8810cda — cache, kiem-cache, smoke XONG (25/10/2026)

### 11.1 Bằng chứng gỡ dừng (do phiên điều phối thực hiện, agent chỉ đọc)

- `van_hanh/dung.json` không còn; `van_hanh/dung_da_go/dung_lan_0010.json` có nội dung `stt: 10`, `luc: 2026-10-25T04:30:10`,
  `ly_do` = ["cha gian doan o lan chay 10", "con da ket thuc (xac nhan qua pid + thoi diem tao)"] — khớp dòng
  `van_hanh/goi_cha.jsonl` (exit 40, 04:30:10, pid 21680) và stderr ở mục 10.3.
- `van_hanh/dung_da_go/quyet_dinh_go_dung.json`: `quyet_dinh: go_dung_tiep_tuc_cung_phien_A`, `luc: 2026-10-25T04:59:00`,
  `sha256_truoc` = `sha256_sau` = `adad5985752806b5630ab9d753be15a7ce74c75e3096d15a321d69bb7fc671a1`, `so_byte: 246`,
  `giu_nguyen` = [`van_hanh/so_crash.json`, `van_hanh/lan_chay/0010*/ket_thuc.json`], điều kiện "chỉ gỡ dừng lần 0010;
  mọi dừng mới vẫn phải hỏi".
- **Giới hạn:** agent không có công cụ băm trong quyền lời gọi này, nên **không tự tính lại** sha256 của
  `dung_lan_0010.json`; hai giá trị sha256 là số phiên điều phối ghi. Phần agent kiểm được là nội dung và mốc thời gian
  khớp với log runner.
- Bộ đếm giữ nguyên trước lệnh đầu: `so_crash.json` tổng 1, bước cache 1, `don_vi` rỗng; lần 0010 `phan_loai: cha_gian_doan`.
  Không mở phiên mới, không đổi tên đơn vị.

### 11.2 Lệnh đã chạy (3/5 lệnh cho phép, tuần tự, foreground)

Tiền tố chung: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r61_bdd/van_hanh_cong_a.py --cong A --phien results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01 --so-doan 1 --ngan-sach-giay 420`

| stt | hậu tố | exit cha | exit con / phân loại | giây con / cha | đơn vị mới / bỏ qua |
|---|---|---|---|---|---|
| 0011 | `--buoc cache --detector yolo11n_mixed5 --device 0` | 0 | 0 / hoan_tat | 115,1 / 116,8 | 18 / 182 |
| 0012 | `--buoc kiem-cache` | 0 | 0 / hoan_tat | 37,1 / 37,9 | 601 (600 đơn vị + tổng hợp) |
| 0013 | `--buoc smoke` | 0 | 0 / hoan_tat | 84,8 / 85,8 | 145 (108 tracking + 36 chấm + tổng hợp) |

stdout của runner: `[VAN HANH] lan 11 cache: exit con 0 (hoan_tat, don vi None) -> exit 0`; tương tự lần 12, 13. Không
có `dung.json` mới. Lần 0011: đơn vị đang dở ở lần 0010 (`b24d283f-33783d1b`) chạy lại với `lan_thu: 2`; không có
artifact dở để sao lưu (`van_hanh/artifact_do/` không có file) — khớp mục 10.3 (chưa có artifact của đơn vị này).

### 11.3 Kết quả kiểm

- **Cache — `kiem_cache/ket_qua.json`** (`xong.json` ghi sha256 `fddffe57…61be`): `dat: true`, `loi: []`, `so_don_vi`
  600 = `ky_vong` 600. Mỗi detector 200 video, `so_frame` 39.973. Phạm vi kiểm từng video (`dieu_phoi_bdd.py:455–503`):
  marker `xong`, khóa/dtype npz, `frame_idx`/`time_s` trùng timeline nhãn, `unreadable`/`truncated` = 0 và JSON
  `n_unreadable`/`n_truncated` = 0, `orig_shape`, `cls == ánh xạ(cls_goc)`, provenance, lớp mixed5 trong tập COCO cho phép,
  `sequence_spec_status == ok` (cache legacy/thiếu spec không đạt nhánh này), `gen_spec`/`spec_sha256` = giao thức.
  Cache mới hoàn toàn trong A (`A/cache/<detector>/`), không dùng cache cũ.
- **Smoke — `smoke/ket_qua_smoke.json`**: `dat: true`, `loi: []`, `so_tracking` 108 = kỳ vọng 108, `so_cham` 36 = kỳ vọng
  36, video đúng b1e1a7b8-a7426a97, b22f385b-5d7e5202, b23b2649-8349d2a1; `marker_sha256` liệt kê 144 marker. Kiểm trong
  `tong_hop_smoke` (`van_hanh_cong_a.py:756–790`): `n_frames` của marker và kiem-cache = seqLength (203/202/202), seed
  truyền/thực dùng = seed run (PF500 s0/s1/s2), `tham_so_bo_loc` = `tham_so_bo_loc_du_kien` của giao thức, evaluator và
  `so_video` của chấm đúng, npz cache của từng detector trên 3 video smoke trùng sha256 kiem-cache sau smoke.
- **Cache không đổi qua smoke — phạm vi:** smoke chỉ đối chiếu lại npz của 3 video smoke × 3 detector. Đối chiếu đủ 600
  cache (npz/json/meta/marker) với kiem-cache là `doi_chieu_kiem_cache` ở bước `khoa` (A2), **chưa chạy**.
- Không có tracking đầy đủ 200 video hay bootstrap. Điểm smoke không dùng để chọn cấu hình.

### 11.4 Thời gian và tài nguyên

| hạng mục | giá trị | nguồn |
|---|---|---|
| suy luận/frame yolov8n_coco | 922,29 s / 39.973 = 23,07 ms | `kiem_cache/ket_qua.json` `giay_suy_luan` |
| suy luận/frame yolov8n_mixed5 | 916,45 s / 39.973 = 22,93 ms | như trên |
| suy luận/frame yolo11n_mixed5 | 1.068,97 s / 39.973 = 26,74 ms | như trên |
| dung lượng cache (npz+json) | 19.221.142 + 14.891.776 + 15.115.283 = 49.228.201 byte | như trên |
| tracking ms/frame (smoke, 3 video) | CV 0,921; EKF 1,088; UKF 2,470; PF500 3,637 | `thoi_gian_tracking_theo_bo_loc` |
| chấm/run (3 video) | 0,41–0,54 s | `0013_smoke/tien_do.jsonl` |
| kiem-cache 600 đơn vị | 37,1 s (con) | `0012/ket_qua_con.json` |
| đỉnh working set con | cache 0011: 1.416.241.152 byte; kiem-cache 539.418.624; smoke 694.054.912 | `ket_qua_con.json` |
| đỉnh CUDA cấp phát (torch) | cache 0011: 65.148.928 byte; bước CPU: null | `0011/ket_qua_con.json` |

Không đo được: thời lượng con của lần 0010 (cha bị ngắt, không có `giay`), nên tổng thời gian tường của cache
yolo11n_mixed5 không đầy đủ; VRAM toàn tiến trình (chỉ có đỉnh cấp phát của torch); thời gian băm riêng (gộp trong
kiem-cache).

### 11.5 Sổ crash

`van_hanh/so_crash.json` sau lần 0013: 13 lần chạy; tổng crash **1/4** (chỉ lần 0010 `cha_gian_doan`, nguyên nhân do
orchestrator ngắt lời gọi — không phải crash native); 0 crash native / 13 lần chạy; bước cache 1; không đơn vị nào có
crash; không có `cha_gian_doan` mới. Faulthandler cha các lần 0001–0010 đều 0 byte (sha256 `e3b0c442…`).

### 11.6 Tự kiểm ngoài runner

- `git status --short`: danh sách giống đầu lượt (tracked đổi chỉ `handoff/PLAN_DA_CHOT.md` và file review48 có từ trước);
  không có mục mới ngoài phạm vi.
- Glob `src/**/__pycache__/*.pyc`: file mới nhất vẫn là `src/__pycache__/baseline_track.cpython-311.pyc` (ngoại lệ đã
  duyệt); không thấy pyc mới từ lượt này.
- Không chạy lại fixture (không thay mã trong lượt này).

### 11.7 Chưa làm / bước tiếp

A2 chưa chạy: `khoa`, `kiem-khoa`, `dong-goi-a`, `kiem-goi-a`; chưa có `goi_a_sha256`/`khoa_sha256`. Bước tiếp: Codex
review A1; chỉ sau khi chấp nhận mới sang A2 (cùng phiên A, đóng băng PLAN và file tracked). Chưa commit. Không mở cổng B.

## 12. A2 — khóa, kiểm khóa, đóng gói, kiểm gói (25/10/2026)

### 12.1 Trước khóa (ghi trước lệnh `khoa`)

- A1 được Codex **approve**: `D:/LABOTARY/automation/runs_local/r61_bdd_cong_a_20261024_20261024_182327/reports/review_007.json`
  (verdict `approve`, issues rỗng). Giới hạn reviewer nêu: review tĩnh (không chạy lại inference/fixture/evaluator); ghi
  nhận Pd8810cda vào PLAN diễn ra sau ba lệnh runner (trước báo cáo) — chưa đúng thứ tự "ghi nhận trước khi tiếp tục";
  approve chỉ cho A1.
- Trạng thái trước khóa: 13 lần chạy (0001–0013) đều có `ket_thuc.json`; không có `van_hanh/dung.json`; không còn tiến
  trình chạy. `git status --short --untracked-files=no`: chỉ `handoff/PLAN_DA_CHOT.md` và
  `handoff/logs/review48/09_thiet_ke_phan_tich_deta_pf_ekf.md` (có từ trước lượt).
- PLAN mục 0 đã cập nhật **trước** báo cáo này: dòng "A2 — trạng thái bàn giao CÓ ĐIỀU KIỆN" ghi đường dẫn gói dự kiến
  và điều kiện bàn giao, không ghi trước kết quả bước chưa chạy. Từ đây PLAN và mọi file tracked/mã/cấu hình đóng băng.
  Báo cáo này không tracked (`handoff/logs/review61/` untracked) nên `git_ban()` (`dieu_phoi_bdd.py:296–307`, chỉ file
  tracked) không băm nó — bổ sung sau khóa không làm lệch khóa.
- Phạm vi khóa theo mã (không sửa mã trong A2): `con_khoa` (`van_hanh_cong_a.py:864–892`) = băm theo đoạn toàn bộ
  `A/cache` → `doi_chieu_kiem_cache` (đối chiếu đủ 600 đơn vị với kiem-cache) → `D.anh_khoa` (giao thức, tiền kiểm,
  nguồn, cấu hình hiệu lực, thư viện, `ma_nguon` + `ma_thu_vien` byte thô, HEAD, `git_status_tracked_sha256`,
  `git_dirty_sha256` — có PLAN —, cache) + kiểm `ma_nguon` = snapshot tiền kiểm + `phu_luc_van_hanh` (PLAN nguyên byte
  tách mục 0, `hop_dong_lenh.json`, giao thức, mọi `src/r61_bdd/*.py`, bảng hash `tien_kiem_a`, `kiem_cache`, `smoke`).
- Ranh giới gói theo mã (`payload()` dòng 929–939, `LOAI_TRU_GOI` dòng 65–75): payload = `tien_kiem/`, `tien_kiem_a/`,
  `cache/`, `kiem_cache/`, `smoke/`, `khoa/`, `kiem_khoa/`, `van_hanh/phien.json`, `van_hanh/artifact_do/`, các lần chạy
  có stt < lần `dong-goi-a`; cộng `dong_goi/phuc_hoi/` và `goi/nhat_ky_crash_chot.json`. Loại trừ có ghi lý do: nhật ký
  điều phối đang mở (`van_hanh/so_crash.json`, `dung.json`, `goi_cha.jsonl`, `van_hanh/cha/` — bản chốt là
  `goi/nhat_ky_crash_chot.json`, trong đó `log_cha` băm từng file log cha), lần chạy từ `dong-goi-a` trở đi, vùng kiểm sau
  gói, `_tam/`, `_cau_hinh_thu_vien/`, chính gói.

### 12.2 Lệnh đã chạy (4 lệnh, tuần tự, foreground, mỗi lệnh một đoạn)

Tiền tố chung: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r61_bdd/van_hanh_cong_a.py --cong A --phien results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01 --so-doan 1 --ngan-sach-giay 420`

| stt | hậu tố | exit cha | exit con / phân loại | giây con | stdout con |
|---|---|---|---|---|---|
| 0014 | `--buoc khoa` | 0 | 0 / hoan_tat | 37,3 | `khoa: 1803 file cache, 34 file ma src, 278 file thu vien` |
| 0015 | `--buoc kiem-khoa` | 0 | 0 / hoan_tat | 8,9 | (kết quả trong `kiem_khoa/vong_01/ket_qua.json`) |
| 0016 | `--buoc dong-goi-a` | 0 | 0 / hoan_tat | 14,3 | `goi A: 3286 file; goi_a_sha256 5c9db320…; khoa_sha256 f3a138d7…` |
| 0017 | `--buoc kiem-goi-a` | 0 | 0 / hoan_tat | 3,9 | (kết quả trong `van_hanh/kiem_goi_a/vong_01/ket_qua.json`) |

stdout cha mỗi lệnh: `[VAN HANH] lan N <buoc>: exit con 0 (hoan_tat, don vi None) -> exit 0`. Mỗi bước xong trong đúng
một đoạn (không exit 10), nên không có đoạn tiếp tục; mọi đoạn băm (`khoa/bam/`, `kiem_khoa/vong_01/bam/doan/0000–0028`,
`dong_goi/doan/0000–0051`) thuộc cùng một lần chạy và bao đủ danh mục. Job Object `dat: true`, con trong job, faulthandler
con 0 byte ở cả 4 lần. Không có `dung.json`.

### 12.3 Hash định danh (cho Codex review và plan B sau này)

| đối tượng | sha256 |
|---|---|
| **`goi_a.json`** (`goi_a_sha256`) | `5c9db320e4965798f50652b823b7a20439410e259752dbf72ebfe990d30fe225` |
| **`khoa/khoa.json`** (`khoa_sha256`) | `f3a138d7497727a205abfa78237f0ec9a6029b89a13bdd018928865351794d4f` |
| `SHA256SUMS` | `fdac9e6f27a6def70939fdf16015aea7c214cfc70392f3bf5656b42b8533fa25` |
| `kiem_khoa/vong_01/ket_qua.json` | `e35a742321eb791cee643bed99b2c6c85bcb3b7a3e7d83ce305a6c84420c06f1` |
| `configs/phase2/r61_bdd/giao_thuc.json` | `82ffe7aea4daa418bb7bddd323fe5a76b2413c9521c1acf23e9e061c36181097` |
| `configs/phase2/r61_bdd/hop_dong_lenh.json` | `7612ca26f8cc46a40f1822cfc7644058734d7e42a8445995fada641986bf5b5c` |
| `src/r61_bdd/van_hanh_cong_a.py` | `bbd557978cace81ad941c72d67b509121e4ce5b30ccd92b2fd45365e1c895cf1` |
| `src/r61_bdd/test_van_hanh_cong_a.py` | `a2077f0ea7e7836fb9b63ec9199d3a939612f8d45f3d727f08f2e3a2ec73a0b1` |
| `src/config.py` | `37e4f56247b7be0896c29d74026b863435b2eec5f53c11fc1a8cd073da70d88b` |
| `tien_kiem_sha256` (trong khóa) | `702fb2407d021285337d41a20841cfefb1fcae0043af23f2af82af81931a2de9` |
| PLAN nguyên byte lúc khóa (`git_dirty_sha256` và `phu_luc_van_hanh.plan.sha256_toan_bo`) | `2b2474aabd3215deb3973d5fa342457d09db422df5b537cbe238d3fe6d93e12a` (194.250 byte, mục 0 = byte 322–176.790) |
| HEAD | `117d62da027e8ce3e4c9f455d576a924b4d82bbb` |

Cách lấy: `goi_a_sha256`/`SHA256SUMS` từ `goi_a.xong.json` + `kiem_goi_a/vong_01/ket_qua.json`; `khoa_sha256` từ
`khoa/khoa.xong.json`, `kiem_khoa/vong_01/ket_qua.json` và `goi_a.json` (ba nơi trùng nhau); các hash còn lại đọc
trong `khoa.json`/`goi_a.json`. Agent không có công cụ băm riêng — các số này do runner tính.

### 12.4 Kết quả kiểm

- **Khóa** (`khoa.json`): khóa đủ trường `KHOA_KHOA` — `nguon` (manifest, nhãn, seqmap, 3 checkpoint R54
  `f59b3d83…`/`542b1381…`/`9a5e34a1…`, GT từng video), `cau_hinh_hieu_luc` (cfg tracker, YAML resolve + `yaml_sha256`,
  tham số bộ lọc, elapsed_mode `steps`, assoc_log false), `thu_vien`, `ma_nguon` 34 file src (có `config.py`, mã r61),
  `ma_thu_vien` 278 file, HEAD, `git_status_tracked_sha256`, `git_dirty_sha256` (có PLAN), `cache` 1.803 file
  (600 × npz/json/marker + 3 `meta.json`). `doi_chieu_kiem_cache` chạy trong bước khoa (exit 0 ⇒ 600 đơn vị khớp
  kiem-cache — đây là phần "cache không đổi qua smoke trên đủ 600 cache" còn treo ở mục 11.3). Kiểm `ma_nguon` = snapshot
  tiền kiểm đạt. Phụ lục vận hành: PLAN (`sha256_truoc_muc0` `a50c3bc4…`, `sha256_sau_muc0` `aba1aebb…` — **trùng
  `chup_dau/chup.json` của baseline lan_01**), hợp đồng, giao thức, 8 file `src/r61_bdd/*.py`, bảng hash `tien_kiem_a`,
  `kiem_cache`, `smoke`.
- **Kiểm khóa** `kiem_khoa/vong_01/ket_qua.json`: `dat: true`, `loi: []`, `so_file_cache` 1.803, `sau_dong_goi: false`;
  băm lại toàn bộ cache theo 29 đoạn, `so_khoa` + `so_phu_luc` không lệch, `kiem_plan` (PLAN ngoài mục 0 = baseline) đạt.
- **Gói** `goi_a.json`: 3.286 file = 1.803 cache + 686 `smoke/` + 677 trong `tien_kiem/`, `tien_kiem_a/`, `kiem_cache/`,
  `khoa/`, `kiem_khoa/`, `goi/nhat_ky_crash_chot.json` + 120 trong `van_hanh/` (`phien.json` + 15 lần chạy 0001–0015, đủ
  15 `ket_thuc.json`); `stt_dong_goi_dau` 16; `phuc_hoi_dong_goi` rỗng (không có gói dở); `artifact_do/` không có file. Đếm
  trong danh mục: 600 marker `cache/*/*.xong.json`; 145 file tên chứa `xong` dưới `smoke/` (`ket_qua_smoke.json` liệt kê
  144 marker = 108 tracking + 36 chấm; file còn lại theo tên là marker tổng hợp smoke — suy theo tên, chưa liệt kê từng
  file). Không có file nào của `van_hanh/so_crash.json`, `goi_cha.jsonl`, `cha/`, `_tam/`, `_cau_hinh_thu_vien/` hay
  lần chạy ≥ 0016 trong danh mục (đúng `LOAI_TRU_GOI`). `dung_da_go/` cũng không có trong danh mục nhưng **không** phải
  do `LOAI_TRU_GOI` — nó nằm ngoài `payload()` và không được kiểm thừa (thiếu sót A2-F1, xem mục 12.6 và 12.8).
- **Kiểm gói** `van_hanh/kiem_goi_a/vong_01/ket_qua.json`: `dat: true`, `loi: []`, `so_file` 3.286, `so_lech` 0,
  `so_thua` 0 (tìm thừa trên toàn payload + `goi/` + `dong_goi/phuc_hoi/`), SHA256SUMS trùng danh mục, `khoa.json` trùng
  `khoa_sha256`. Kiểm gói chỉ đọc payload; log và đoạn băm của nó nằm trong `van_hanh/kiem_goi_a/` (ngoài payload).
- **Snapshot mã thống nhất:** runner kiểm snapshot ghim trong `van_hanh/phien.json` ở mỗi lần chạy (lệch → exit 2); cả 17
  lần chạy không lệch; mã r61 trong khóa trùng snapshot fixture lan_02 (runner `bbd55797…`, hợp đồng `7612ca26…`, giao
  thức `82ffe7ae…`, mục 10.2).
- **Sau khóa:** `git status --short --untracked-files=no` sau lần 0017 giống trước khóa (chỉ PLAN và file review48 có từ
  trước); agent không sửa PLAN hay file tracked/mã/cấu hình sau mục 12.1. Glob `src/**/__pycache__/*.pyc`: file mới nhất
  vẫn là `baseline_track.cpython-311.pyc` (ngoại lệ đã duyệt).

### 12.5 Sổ crash (bản chốt `goi/nhat_ky_crash_chot.json`, `chot_tai_stt` 16; bản đang mở `van_hanh/so_crash.json`)

- Mẫu số: **17 lần chạy con** (0001–0017) và **18 lời gọi cha** (18 file `van_hanh/cha/*.faulthandler.log`, đều 0 byte
  `e3b0c442…`; lời gọi cha thứ 18 là lời gọi 04:30:10 tự ghi phân loại cho lần 0010 rồi exit 40, không tạo lần chạy mới).
- Crash theo chính sách: **1/17 lần chạy** (lần 0010 `cha_gian_doan` — cha bị orchestrator ngắt khi quá timeout lời gọi
  Claude, con pid 7400 đã kết thúc; gỡ dừng theo Pd8810cda). Crash native: **0/17**. Bước khóa: `khoa` 0/1,
  `kiem-khoa` 0/1, `dong-goi-a` 0/1 (ngưỡng 2). Không đơn vị nào có crash (ngưỡng 3); tổng phiên 1/4.
- Log thiếu: lần 0010 không có `exit`/`giay` của con (cha bị ngắt; `ket_thuc.json` do lời gọi cha pid 21680 ghi bổ sung
  lúc 04:30:10: `exit`/`het_gio`/`ket_qua_con` = null, `con_trang_thai.con_song: false` (`loi_win` 87), `file_con` có
  `faulthandler.log` + `tien_do.jsonl`, `don_vi` null; không có các trường `nhat_ky_doc_duoc`/`thieu_log`/
  `faulthandler_byte` như lần chạy thường). Bản chốt ghi lần 0016 là `dang_mo: true` (đúng thiết kế: chốt khi lần đóng gói còn mở); kết thúc của 0016 và
  0017 chỉ có trong `van_hanh/so_crash.json` và `lan_chay/0016*/0017*/ket_thuc.json` (ngoài gói, theo `LOAI_TRU_GOI`).
- Không tuyên bố đã biết hết nguyên nhân crash: 6/14 phiên fixture PREP exit 139 (báo cáo 02 mục 11) vẫn CHƯA RÕ nguyên
  nhân; phiên A này chỉ cho thấy không có crash native trong 17 lần chạy, không chứng minh nguyên nhân đó đã hết.

### 12.6 Ngoại lệ và giới hạn

- **Ngoại lệ baseline:** đúng mtime thư mục `.git` + 3 mục duyệt Pe27e304a (pyc `src/__pycache__/baseline_track.cpython-311.pyc`,
  mtime `src/__pycache__`, mtime `results/track_video`) — ghi riêng kèm kết quả thô trong `tien_kiem_a/so_sanh.json`
  (mục 10). Ngoại lệ không miễn kiểm nội dung Git/index/HEAD/PLAN.
- **615 ảnh không annotation** của BDD COCO-video val (778 frame không có lớp 1 = 615 + 163 chỉ có lớp không chấm/ignore;
  báo cáo 02 mục 2): giữ nguyên theo giao thức, không loại frame/đổi GT. Smoke chỉ 3 video; chưa có tracking 200 video
  hay bootstrap — cổng A không cho kết quả HOTA nào để kết luận.
- **Bằng chứng gỡ dừng nằm ngoài gói (review_008 A2-F1 — thiếu sót của gói A, không chỉ là ý tưởng):**
  `van_hanh/dung_da_go/` (2 file) không thuộc `payload()` (`van_hanh_cong_a.py:929–939`), không trong `LOAI_TRU_GOI`
  và không bị kiểm thừa (`:1033` chỉ tìm thừa trong payload + `goi/` + `dong_goi/phuc_hoi/`); gói A **không** băm hai
  file này và `kiem-goi-a` không phát hiện được nếu chúng bị đổi/mất. *Đính chính (A2-F2):* bản trước viết "nội dung
  quyết định được phản ánh gián tiếp qua `ket_thuc.json` của lần 0010" là **sai**: `lan_chay/0010_cache_yolo11n_mixed5/ket_thuc.json`
  chỉ ghi việc **dừng** lúc 04:30:10 (`phan_loai: cha_gian_doan`, `quyet_dinh` = `exit_cha: 40` + hai `ly_do_dung`),
  không chứa quyết định **gỡ dừng** lúc 04:59; quyết định gỡ dừng chỉ nằm ở `van_hanh/dung_da_go/quyet_dinh_go_dung.json`
  (ngoài gói). Không sửa mã/khóa/gói sau khóa để bổ sung (sửa sẽ làm lệch snapshot đã ghim); xử lý bằng phụ lục mục 12.8
  theo quyết định Pab7c2804.
- **PLAN sau khóa:** *Đính chính (A2-F2):* bản trước viết "`kiem-khoa` (06:57) là lần cuối runner băm PLAN" là **sai**.
  Sau `kiem-khoa`, bước `dong-goi-a` (lần 0016) còn **đối chiếu PLAN với khóa**: `van_hanh_cong_a.py:993` gọi
  `so_phu_luc(khoa["phu_luc_van_hanh"], phu_luc(ctx, A))`, trong đó `phu_luc()` (`:545–553`) tách PLAN nguyên byte
  (`sha256_toan_bo`, trước/sau mục 0) — lệch sẽ ném `goi A: khoa lech` và không ghi gói. Lời gọi 0017 (`kiem-goi-a`) không
  đối chiếu PLAN với khóa, nhưng **còn ghi hash PLAN** trong snapshot đầu vào (`dau_vao()` `:1352–1357`):
  `van_hanh/lan_chay/0017_kiem-goi-a/goi.json` → `dau_vao.plan.sha256_toan_bo` = `2b2474aa…e93e12a` (194.250 byte, mục 0 =
  byte 322–176.790), trùng khóa; file này nằm ngoài gói (lần ≥ 0016). Review_008 đã tự băm PLAN hiện tại = `2b2474aa…`,
  khớp khóa. Agent không sửa PLAN sau khóa.
- **Giới hạn reviewer/agent:** không chạy lại fixture trong A2 (mã không đổi kể từ fixture lan_02 đã approve); không
  chạy lại inference/evaluator; review_007 là review tĩnh. Kết quả A2 không được ghi vào PLAN mục 0 (đóng băng sau khóa)
  — mục 0 chỉ có trạng thái "có điều kiện" ghi trước khóa.

### 12.7 Trạng thái và bước tiếp

Bốn bước A2 exit 0; kiểm khóa và kiểm gói `dat: true`. Gói A **chưa** được coi là bàn giao cho tới khi Codex review cuối
approve với `goi_a_sha256 = 5c9db320e4965798f50652b823b7a20439410e259752dbf72ebfe990d30fe225` và
`khoa_sha256 = f3a138d7497727a205abfa78237f0ec9a6029b89a13bdd018928865351794d4f`. Sau review thì DỪNG: không phiên B,
approve A không cấp quyền B. Chưa commit/stage. Không còn tiến trình chạy.

*Cập nhật 25/10 (Pab7c2804, sau review_008 `blocked`):* review_008 tự băm đủ 3.286 file gói, khóa, PLAN và xác nhận
hash định danh ở mục 12.3; nêu A2-F1 (bằng chứng gỡ dừng ngoài gói) và A2-F2 (hai câu sai ở mục 12.6). Xử lý chỉ bằng
báo cáo này: đính chính mục 12.6 và phụ lục 12.8. Không sửa PLAN, mã, cấu hình, khóa, gói hay bằng chứng; không chạy
runner; khóa/gói A giữ nguyên hash ở trên.

### 12.8 Phụ lục bằng chứng ngoài gói A

**Gắn với:** `goi_a_sha256 = 5c9db320e4965798f50652b823b7a20439410e259752dbf72ebfe990d30fe225`,
`khoa_sha256 = f3a138d7497727a205abfa78237f0ec9a6029b89a13bdd018928865351794d4f`, phiên
`results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/`.

Hai file do phiên điều phối tạo 25/10 ~05:10 (quyết định Pd8810cda, `luc` trong file 04:59:00), nằm **ngoài** payload
gói A (đường dẫn tương đối theo phiên A):

| file | số byte | sha256 | nguồn giá trị |
|---|---|---|---|
| `van_hanh/dung_da_go/dung_lan_0010.json` | 246 | `adad5985752806b5630ab9d753be15a7ce74c75e3096d15a321d69bb7fc671a1` | phiên điều phối (`quyet_dinh_go_dung.json`: `sha256_truoc` = `sha256_sau`, `so_byte` 246); review_007 tự băm: 246 byte, `adad5985…` — **trùng**; review_008 tự băm `adad5985…` — **trùng** |
| `van_hanh/dung_da_go/quyet_dinh_go_dung.json` | 1123 | `f346a6fe27138e9eb5b8bd2d3b19c255837daf25706c90fbac0e33af61f4683b` | phiên điều phối; review_008 tự băm `f346a6fe…` — **trùng** (review_007 không băm file này; review_008 không ghi số byte) |

- Nội dung agent đọc được (khớp log runner): `dung_lan_0010.json` = `stt: 10`, `luc: 2026-10-25T04:30:10`, hai `ly_do`
  ("cha gian doan o lan chay 10", "con da ket thuc (xac nhan qua pid + thoi diem tao)"), `huong_dan` dừng hỏi;
  `quyet_dinh_go_dung.json` = `quyet_dinh: go_dung_tiep_tuc_cung_phien_A`, `luc: 2026-10-25T04:59:00`, `tep_goc`
  `van_hanh/dung.json` → `tep_moi` `van_hanh/dung_da_go/dung_lan_0010.json`, `giu_nguyen` [`van_hanh/so_crash.json`,
  `van_hanh/lan_chay/0010*/ket_thuc.json`], điều kiện chỉ gỡ dừng lần 0010.
- **Giới hạn kiểm của agent:** quyền lời gọi này không có công cụ băm/đếm byte, nên agent **không tự tính** sha256 hay
  số byte; số byte 1123 của `quyet_dinh_go_dung.json` chỉ có từ phiên điều phối (chưa reviewer nào ghi lại). Reviewer có
  thể tự băm hai đường dẫn trên để đối chiếu.
- **Giới hạn của gói A:** runner A (`payload()` `:929–939`, kiểm thừa `:1033`) không bao `van_hanh/dung_da_go/`; vì vậy
  hai hash trên **không** được `goi_a.json`/`SHA256SUMS`/`kiem-goi-a` bảo vệ — chúng chỉ được neo bởi phụ lục báo cáo
  (untracked) này và hai review. Bằng chứng ổn định này bị bỏ khỏi kiểm hash của gói A, không phải được loại trừ có lý do.

**YÊU CẦU cho plan cổng B** (ghi để plan B sau này đưa vào; không phải quyền mở B):

1. Khi mở phiên B và trong hậu kiểm B: **băm lại** hai file trên và so với bảng này; lệch, thiếu hoặc có thêm file trong
   `van_hanh/dung_da_go/` → dừng, không retry.
2. **Đưa hai file vào manifest B** (đường dẫn + số byte + sha256), kèm `goi_a_sha256`/`khoa_sha256` ở trên.
3. **Kiểm thừa toàn bộ `van_hanh/`** của phiên A (không chỉ payload A): mọi file dưới `van_hanh/` phải nằm trong manifest
   hoặc trong một **danh sách loại trừ tường minh** (đường dẫn/mẫu + lý do, ví dụ nhật ký điều phối đang mở); không có
   ngoại lệ ngầm.
4. **Băm lại `handoff/PLAN_DA_CHOT.md`** (nguyên byte và tách mục 0) và đối chiếu với khóa A: `sha256_toan_bo`
   `2b2474aabd3215deb3973d5fa342457d09db422df5b537cbe238d3fe6d93e12a` (194.250 byte), `sha256_truoc_muc0` `a50c3bc4…`,
   `sha256_sau_muc0` `aba1aebb…`; ghi kết quả thô. Nếu PLAN đã được sửa hợp lệ sau A (ví dụ cập nhật mục 0 cho B), plan B
   phải nêu rõ phép so nào áp dụng (ngoài mục 0 vs nguyên byte) trước khi chạy, không tự chuẩn hóa hay miễn kiểm.
