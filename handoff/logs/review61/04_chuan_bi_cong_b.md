# R61 cổng B — chuẩn bị lớp vận hành + fixture CPU (B0, 26/10/2026)

Trạng thái mới nhất (27/10, Pbeca24e7 sau B2): **gói chuẩn bị B hoàn tất phía coder; đối chiếu Pbeca24e7 chỉ đọc xong,
chờ Codex review rồi dừng.** B0 approve (review_004), bước gộp P48bd9f29 approve (review_007), B2 approve (review_008):
fixture `lan_02/vong_02` dat=true, exit 0, 244/244 kiểm, snapshot thực thi `dd36a0d2…a7c239c`. **B thật CHƯA chạy, không
có phê duyệt B, không có phiên `thuc_thi/`.** Mục Pbeca24e7 ngay dưới là đối chiếu cuối; B2 là bản chốt; mục P48bd9f29 là
bằng chứng chi tiết; phần sau nữa là lịch sử. Lệnh B1 hiện hành dùng `--phien …/lan_02` (mục 6 còn ghi lan_01 là lịch sử).

## Pbeca24e7 (27/10/2026) — đối chiếu chỉ đọc bằng chứng lan_02 (theo QUVZ/P48bd9f29 mục 6)

Phạm vi theo quyết định QUVZ: phần trùng của Pbeca24e7 sau B2 **chỉ đọc/đối chiếu** bằng chứng lan_02 đã nghiệm thu; không
sửa mã, không chạy lại fixture, không mở lan_03. Lời gọi này chỉ sửa báo cáo 04 (mục này + dòng trạng thái); **0 lần gọi
fixture**, `test_van_hanh_cong_b.py` và `lan_02/` không bị ghi.

Đối chiếu từng yêu cầu của Pbeca24e7 với bằng chứng đã có (đều đã được review_007/008 tự băm/đếm):

| yêu cầu Pbeca24e7 | bằng chứng | ở đâu |
|---|---|---|
| lệnh B1 mới `--phien …/lan_02 --ngan-sach-giay 420 --gioi-han-giay 540`, quyền Bash giữ nguyên | 3 lần gọi đúng lệnh, exit 10/10/0 | Q.1, B2.3; review_007/008 |
| quyết định chuyển phiên ghi rõ | `lan_02/chuyen_phien/quyet_dinh.json` sha256 `f63fbac4…4b38c79` (review_007 tự băm, trùng giá trị fixture ghi) | Q.5, B2.2 |
| manifest lan_01 trước/sau nguyên byte | `manifest_lan_01_truoc.json` + 3 `manifest_lan_01_sau_*` 274 file, `sha256_tep 550e3e28…`; review_007 tự băm 274 file khớp | Q.5 |
| cây tiến trình cũ (pid12248) đã kết thúc trước ma trận, PID + thời điểm tạo + journal | `kiem_cay_cu_20261027_052355_312_pid19660.json` dat=true, 16 PID | Q.5 |
| kế thừa sự cố bằng tham chiếu/hash, không reset/đếm trùng | sổ lan_02: 3 bản ghi kế thừa = 1 sự cố native, 1/4 ngưỡng | Q.5, B2.5 |
| lỗi mới vẫn chặn theo chính sách | ngoại lệ chỉ 3 lý do dừng ghim trong mã; `crash_ngoai_du_kien_vong=[]` | Q.5 |
| snapshot cuối đủ ca 00–14, dat=true, ca âm khớp exit + nguyên nhân | `vong_02/ket_qua_test.json` | Q.3, B2.4 |
| bootstrap/CSV/hậu kiểm, deadline/thu hồi cây, pyc trước/sau | ca 02, 09, 14; 3 cặp pyc 6.453 mục 0/0/0 | Q.3, Q.6 |
| cha fixture không nạp ultralytics/torch | 164 điểm kiểm, `cha_khong_nap_ultralytics_torch=true` | Q.4 |
| đính chính ca 00 = 6 kiểm, ca 01 = 7 kiểm, tổng 13 (không phải tổng ma trận 244) | đã ghi | Q.3, B2.4 |
| event log do phiên điều phối cung cấp | **đầu vào lời gọi này không kèm kết quả event log nào**; không truy vấn (P05759f93 bị từ chối), không suy diễn; nguyên nhân crash pid12248 vẫn CHƯA RÕ | Q.5 |

Kiểm lại trong lời gọi này để xác nhận snapshot thực thi **không đổi sau review_007/008** (chỉ đọc, kết quả thô):

1. `git status --short --untracked-files=all -- <23 file mã ngoài r61_bdd trong vong.json snapshot.ma> src/r61_bdd
   configs/phase2/r61_bdd handoff/PLAN_DA_CHOT.md results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01` → exit 0. 23 file
   ngoài `r61_bdd` (`assoc_log.py` … `ul_backend.py`, `mot_datasets/*`): **không có dòng nào** ⇒ khớp HEAD `117d62d`.
   `src/r61_bdd`: đúng 15 file `??` như B0/B2 (không file mới, không `__pycache__`); `configs/phase2/r61_bdd`: đúng 3 file
   `??`; ` M handoff/PLAN_DA_CHOT.md` có từ trước lượt (review_007/008 xác nhận PLAN khớp hash ghim `2b2474aa…`).
2. Glob theo mtime (tăng dần) trên 15 file `src/r61_bdd/**/*.py`, 3 file `configs/phase2/r61_bdd/*.json`, PLAN,
   `vong_02/ket_qua_test.json` và `khoi_dong/…pid18140.ket_thuc.json`: mọi file mã/hợp đồng/giao thức/PLAN đều **cũ hơn**
   `ket_qua_test.json` (2 file kết quả nằm cuối). Tương tự các file cấp đầu của A (`goi_a*.json`, `SHA256SUMS`, `khoa/*`,
   `van_hanh/*`, 2 phụ lục `dung_da_go/*`) đều cũ hơn `ket_qua_test.json`; `van_hanh/` A không có `dung.json`.
3. `lan_02/khoi_dong/`: đúng 9 file = 3 journal (pid19660/7216/18140) mỗi cái có `bat_dau`/`ket_thuc`/faulthandler ⇒ không
   có lần gọi fixture nào sau review_007. `…pid18140.ket_thuc.json`: `exit 0`, `giay 309.935`, `vong_02`, snapshot
   `dd36a0d2…a7c239c`. `vong_02/ket_qua_test.json`: `dat=true`, `so_kiem=244`, `so_dat=244`, `thieu_kiem_bat_buoc=[]`,
   `crash_ngoai_du_kien_vong=[]`, `cha_khong_nap_ultralytics_torch=true`, `so_lenh_con=142`, snapshot `dd36a0d2…`.
4. Không có `lan_02/dung_fixture.json` (chỉ có `dung_fixture.json` trong các phiên giả `vong_02/so_gia/*`,
   `vong_02/tham_do/ngat_khoi_dong/phien_kd/` của ca 11–13, là sản phẩm có chủ ý của ma trận); không có `lan_03/`, không có
   `results/phase2/r61_bdd/thuc_thi/`, không có `src/r61_bdd/**/*.pyc`.
5. Glob `lan_01/**` + `manifest_lan_01_truoc.json` = 275 ⇒ lan_01 có 274 file, đúng số trong manifest trước.

Kết luận đối chiếu: không thấy dấu hiệu snapshot thực thi, hợp đồng, đầu vào hay khóa liên quan đổi sau nghiệm thu; không
có lỗi mới, không thiếu/mâu thuẫn bằng chứng ⇒ theo QUVZ, Pbeca24e7 được coi hoàn tất bằng bằng chứng lan_02 đã nghiệm thu,
**chờ Codex xác nhận**. Giới hạn: lời gọi này **không băm SHA256** (quyền không có công cụ băm ngoài fixture, và chạy fixture
bị cấm ở bước này); bằng chứng "không đổi" ở đây là git status (mạnh cho 23 file đã commit) + thứ tự mtime + đếm file (yếu
hơn băm cho 15 file `r61_bdd`, 3 hợp đồng/giao thức, A, lan_01). Phép băm lại 39 file + 3 hợp đồng so `dd36a0d2…`, PLAN,
gói/khóa A và 274 file lan_01 cần Codex thực hiện khi review. Ngân sách: orchestrator báo 9/10 vòng trước lời gọi này;
lời gọi này = vòng 10/10 — **hết ngân sách**, sau review không còn vòng sửa. Không mở B thật hoặc Nano.

## B2 (27/10/2026) — chốt báo cáo gói chuẩn bị B (chỉ đọc bằng chứng, không chạy fixture)

Lời gọi này chỉ sửa báo cáo 04; không chạy fixture, không sửa mã/hợp đồng/PLAN/A, không tạo phê duyệt hay phiên B thật.

### B2.1 Trạng thái và chuỗi nghiệm thu (review trong `D:/LABOTARY/automation/runs_local/r61_bdd_prep_b_20261026_20261026_132958/reports/`)

| bước | review | verdict | nội dung |
|---|---|---|---|
| B0 (4 vòng sửa) | 001, 002, 003 / 004 | revise ×3 / **approve** | thiết kế + 6 file; chưa chạy Python (mục 0, 0b, 0c, 9) |
| B1 lan_01 | 005 | blocked | crash native fixture pid12248 → `dung_fixture.json` lan_01 (mục "B1 lời gọi Claude 1") |
| P43284b78 | 006 | blocked | mã chuyển bước bên thứ ba sang con; thiếu bằng chứng chạy (chưa tới lượt chạy lan_02) |
| P48bd9f29 (B1 + Q73M + Q9MG) | 007 | **approve** | lan_02 đạt trên snapshot cuối; Codex tự băm 39 file nguồn + 3 hợp đồng/giao thức khớp `dd36a0d2…`, PLAN/gói/khóa A/review_009/2 phụ lục khớp ghim, 274 file lan_01 khớp manifest, 3 cặp pyc 6.453 mục không đổi |
| B2 | (chờ) | — | báo cáo này |

Fixture ≠ công việc thật: mọi số dưới đây là fixture CPU trên dữ liệu/giao thức tổng hợp (`lan_02/vong_02`); 7.200 tracking,
36 chấm, tái lập, chấm lại, bootstrap 5000×200, 108 CI thật **chưa chạy**.

### B2.2 Digest và artifact chứng nhận

- Snapshot thực thi cuối: `lan_02/vong_02/vong.json` `snapshot.sha256 = dd36a0d2c15d0a38f432ec0c177633c757922800d7a6a22a94b8712c4a7c239c`
  (7 file chính ở Q.2 + mọi `r61_bdd/*.py` + đóng import tĩnh `DIEM_VAO_MA`). `vong.json` **không** chứa báo cáo 04 (Grep
  `04_chuan_bi|handoff` = 0) ⇒ sửa báo cáo ở B2 không đổi snapshot thực thi; digest báo cáo 04 tại review_007 ghi trong
  review đó, khác bản B2 này là đúng dự kiến.
- Kết luận: `lan_02/vong_02/ket_qua_test.json` (= `vong_02/ket_qua/lan_02.json`); chuyển phiên `lan_02/chuyen_phien/quyet_dinh.json`
  (7.800 B, sha256 `f63fbac4…4b38c79` do chính fixture ghi trong `ket_qua_test.json.chuyen_phien_lan_01.quyet_dinh`).
  B2 **không** tự tính/ghim thêm digest của `ket_qua_test.json`/`quyet_dinh.json` (đề xuất P4a7fafa2 bị từ chối); đối chiếu
  Pbeca24e7 dùng snapshot thực thi + review_007.
- Không có trong lan_02: `dung_fixture.json` (Glob rỗng); mọi 3 journal `khoi_dong/*.bat_dau.json` có `ket_thuc.json` tương ứng.
  Không có `results/phase2/r61_bdd/thuc_thi/**`, không có `tests/cong_b_20261026/lan_03/**`, không có `src/r61_bdd/**/*.pyc` (Glob, B2).

### B2.3 Lệnh và các lần gọi fixture ngoài (toàn bộ lượt chuẩn bị B)

Mọi lần đúng mẫu `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r61_bdd/cong_b/test_van_hanh_cong_b.py …`;
cha ghi `orig_argv` có `-B`, `flags.dont_write_bytecode=1`, `sys.dont_write_bytecode=true`, `PYTHONDONTWRITEBYTECODE=1`
(`ket_qua_test.json.bytecode_fixture`).

| lời gọi coder | phiên | lần | exit | giây | ghi chú |
|---|---|---|---|---|---|
| B1 lời gọi 1 | lan_01 | 1 | 139 (0xC0000005) | ~48 tới crash | crash native pid12248, nguyên nhân CHƯA RÕ |
| B1 lời gọi 1 | lan_01 | 2 | 40 | 5,5 | dừng hỏi đúng thiết kế |
| P43284b78 | `khong_ton_tai_kiem_doi_so` | 3 lần kiểm đối số | 1 (`--gioi-han-giay 1`, trần cứng), 2, 2 | — | Tier 1, không ghi file, không chạy ma trận |
| P48bd9f29 | lan_02 | 1 (`vong_01`) | 10 | 431,6 | lỗi lập lịch fixture → sửa → snapshot mới |
| P48bd9f29 | lan_02 | 2 (`vong_02`) | 10 | 399,3 | ca 00–06 |
| P48bd9f29 | lan_02 | 3 (`vong_02`) | **0** | 309,9 | ca 07–14, 244/244 |
| B2 | — | 0 | — | — | không chạy fixture |

Tổng thời gian ma trận lan_02 = 1.140,9 s (3/5 suất, < 2.700 s/lời gọi); không lần nào chạm trần cứng 532 s.

### B2.4 Ma trận `vong_02`: lệnh con, exit và thời gian theo ca (cộng từ `ket_qua_test.json.lenh[].giay`)

| ca | lệnh con | giây | ghi chú loại đơn vị |
|---|---|---|---|
| 00 thiet_lap | 7 | 38,1 | `--viec` 6 (the_gioi 5,3; A_tien_kiem 12,1; 3 cache 4,4–4,6; A_goi 6,7) + `--in-ma-b` 0,6 |
| 01 bao_ve_lan_goi | 4 | 9,3 | tham dò 0,3–0,4; trần cứng 8,35 |
| 02 thanh_cong | 24 | 172,5 | runner: tien-kiem 6,9; mo-phien 7,0; tracking 7,0/12,95; cham 7,2/10,1; lap 9,3/6,9; cham-lai 7,3/6,9; tong-hop 7,2/7,6; kiem-khoa 7,3; hau-kiem 7,6–9,1; dừng sớm 0,6–1,8; `--viec doi_chieu_bootstrap` 5,4 |
| 03 kiem_lai_buoc_xong | 8 | 23,5 | dừng ở kiểm đầu vào 1,6–1,7; đạt 6,7–6,8 |
| 04 am_dau_vao | 41 | 45,5 | dừng ở đối số/hợp đồng/giao thức 0,18–0,24; ở phê duyệt 0,50–0,56; ở kiểm đầu vào 1,5–1,7; ảnh khóa 5,8 |
| 05 phuc_hoi_journal | 0 | — | trong tiến trình |
| 06 ma_b_doi | 2 | 8,4 | |
| 07 crash_ghi | 13 | 90,6 | 6,4–7,9 mỗi lệnh |
| 08 cha_gian_doan | 7 | 20,7 | |
| 09 job_pid_log | 7 | 76,3 | timeout 59,7 |
| 10 nguong | 18 | 97,7 | 6,1–6,8; giữ chặn 0,53–0,55 |
| 11 khoi_dong_ngat | 3 | 9,1 | fixture con 4,4–4,6; tham dò 0,13 |
| 12 so_crash | 3 | 13,2 | fixture con 4,3–4,5 |
| 13 so_crash_bo_sung | 5 | 21,9 | fixture con 4,3–4,5 |
| 14 ket_thuc | 0 | — | |
| **tổng** | **142** | **626,9** | = `giay_lenh_con`; phần còn lại của 709,2 s (2 lần gọi `vong_02`) là chụp pyc, kiểm trong tiến trình, ghi bằng chứng |

Số kiểm theo ca nằm ở `vong_02/ca/NN_<ca>.json` (ca 00: 6, ca 01: 7 — đã đính chính; tổng ma trận 244). Thời gian đơn
vị runner ở đây là của dữ liệu tổng hợp rất nhỏ, gần như toàn bộ là khởi động + kiểm đủ đầu vào/ảnh khóa (~6–7 s/lệnh);
**không dùng để dự toán B thật**.

### B2.5 Phân loại exit/crash (`vong_02`) và sổ crash

`exit_lenh`: 0×33, 2×57 (ca âm, khớp exit + nguyên nhân), 40×25 (dừng/giữ chặn có chủ ý), 10×10 (`cr1/cr2_tiep_*` 4,
`tc_*` giới hạn đơn vị/thiếu dự phòng 5, `bv_thieu_du_phong` 1), 30×10 (runner báo con crash native do tiêm: `cr*` 4, `ng*` 6),
0xC0000005×5 (tiêm trực tiếp: `bv_ngat_giua_popen_gan`, `cha1_khi_con_chay`, `cha2_sau_tao_con`, `jp_pid_cha_sau_con`,
`kd_ngat_khoi_dong`), 1×2 (`bv_loi_python_giua_popen_gan`, `bv_tran_cung_trong_tien_trinh`). Mọi bất thường đều là lỗi tiêm
đã xác nhận theo loại (`tiem`, `so_loi_tiem=50`); `crash_ngoai_du_kien_vong=[]`.
Sổ `lan_02/so_crash_ngoai_du_kien.json`: `muc` chỉ 3 bản ghi kế thừa lan_01 = **1 sự cố** `khoi_dong:20261027_030727_163_pid12248`
(native, tính 1/4 vào ngưỡng phiên lan_02, không reset, không đếm trùng). Crash ngoài dự kiến của cả lượt chuẩn bị B: đúng
1 (pid12248); nguyên nhân CHƯA RÕ; phiên điều phối không cung cấp kết quả event log cho lượt này, coder không truy vấn
(P05759f93 bị từ chối).

### B2.6 Resume, thu hồi cây, bytecode, đường ghi (tóm tắt; chi tiết Q.3–Q.6)

- Resume: exit 10 ở lần 1, 2 → lần sau tiếp tục cùng lệnh, checkpoint `lenh/*.json`, `ca/*.json`, `thiet_lap/*.xong.json`;
  ca 02/07 chứng minh không chạy lại đơn vị hoàn tất, artifact dở sao lưu trước phục hồi, marker hỏng exit 2.
- Thu hồi cây: mọi 142 lệnh `trong_job_goc=true`, `cay.active=0` (review_007 tự đối chiếu); hạn con ≤ hạn chung − 15 (ca 14).
- Cha không nạp ultralytics/torch: 164 điểm kiểm (`ben_thu_ba_cha_diem_kiem`), cả 2 lần gọi `vong_02` chỉ `_distutils_hack`
  nạp sẵn lúc khởi động; 7 `viec/*.json` cho thấy bên thứ ba chỉ nạp ở con.
- Bytecode: 3 lần gọi × chụp trước/sau, 6.453 pyc trong phạm vi, thêm/mất/đổi 0/0/0; không xóa pyc. Đường ghi: kiểm
  lexical/realpath/reparse trước lần ghi đầu (ca 04, 05) đạt; ghi mới chỉ dưới `lan_02/`; lan_01 nguyên byte (274 file).

### B2.7 Giao diện phê duyệt B thật (chưa tồn tại; B2 không tạo, không ghi `duyet=true`)

Theo `hop_dong_cong_b.json` `phe_duyet_b` (mục 4): file JSON do **orchestrator** phát hành **sau khi người dùng duyệt
kế hoạch thực thi B riêng**, nằm ngoài `results/phase2/r61_bdd/tests/`, trường bắt buộc: `loai="phe_duyet_cong_b"`,
`duyet=true`, `goi_a_sha256=5c9db320…0fe225`, `khoa_sha256=f3a138d7…794d4f`, `review_a_sha256=107fa671…0ea25`,
`hop_dong_cong_b_sha256` (sha256 file hợp đồng B lúc phát hành), `ma_b_sha256` (in bằng `van_hanh_cong_b.py --in-ma-b` —
manifest mã B của runner, **khác** snapshot fixture `dd36a0d2…`), `phien_b = results/phase2/r61_bdd/thuc_thi/<ten>` (trùng
`--phien`), `plan_thuc_thi_sha256` (64 hex), `nguoi_duyet`, `luc`. Thiếu/sai → exit 2 trước mọi ghi.

### B2.8 Lộ trình để planner lập kế hoạch thực thi B riêng (đề xuất thứ tự, chưa phải quyết định)

Điều kiện đầu: người dùng duyệt kế hoạch thực thi; mã/hợp đồng B và giao thức vẫn khớp snapshot đã nghiệm thu
(`dd36a0d2…`; lệch → phải nghiệm thu lại bằng fixture trước); orchestrator phát hành phê duyệt (B2.7). Mọi bước dùng
`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r61_bdd/cong_b/van_hanh_cong_b.py --cong B --phien <thuc_thi/…>
--buoc <b> --so-doan 1 --ngan-sach-giay ≤420 --phe-duyet <file>`, exit 10 → tiếp tục cùng lệnh; quyền Bash cần cấp đúng mẫu này.

1. `tien-kiem` — tiền kiểm A mở rộng 11 nhóm (PLAN nguyên byte + mục 0/trước/sau; review_009; gói/SHA256SUMS/marker; 2 phụ lục
   246/1123 B; `van_hanh` A với loại trừ 18/16/55/1/1 — **lần đầu kiểm trên A thật**, sai số đếm → exit 2, dừng hỏi;
   payload; ảnh khóa hiện tại). `dung.json` luôn chặn.
2. `mo-phien` — `phien.json` + `manifest_b.json`, ghim mã B + phê duyệt.
3. `tracking` — 7.200 đơn vị run-video (36 run × 200 video, 39.973 frame/run) từ cache detector của gói A (kiểm cache
   trước mỗi đơn vị); BDD COCO-video val, 5 fps, buffer 6, `steps`, `mot_crowd`, lớp [1,2,3,5,7], `assoc_log=false`.
4. `cham` — 36 run, chỉ khi đủ 7.200 đơn vị hợp lệ.
5. `lap-tracker` — tái lập 36 run trên video vị trí 100 `b204a5c1-05981158`, `data/<video>.txt` trùng byte.
6. `cham-lai` — 6 run định trước (`yolov8n_coco`/`yolov8n_mixed5`/`yolo11n_mixed5` × {`c25-ekf`, `c10-pf500-s2`}),
   `per_video_metrics.csv`, `combined_metrics.csv`, `raw_fields.json` trùng byte.
7. `tong-hop` — chỉ số `default_rng(0)` 5000×200 lưu một lần, khối 250 mẫu (20 khối), PF trung bình metric COMBINED 3 seed,
   108 CI (6 cặp × 3 metric × 6 nhóm) phân vị 2,5/97,5 `linear` trên toàn mẫu.
8. `kiem-khoa` (crash thứ 2 của bước này → dừng hỏi).
9. `hau-kiem` — kiểm lại A/khóa/PLAN/phụ lục/mã B, tái tính mọi khối, so byte CSV/`kiem_chung`.
10. Báo cáo cuối mốc 5 **tách riêng BDD (B) và UA-DETRAC (R54–R59 test40)**, không gộp.
11. **Cập nhật `handoff/PLAN_DA_CHOT.md` mục 0 cần bước chuyển khóa riêng được người dùng duyệt, SAU khi B và báo cáo cuối
    xong**: PLAN nguyên byte được ghim trong khóa A và hợp đồng B; sửa PLAN trước đó làm mọi lệnh B exit 2 `[plan]`.

Chính sách vận hành giữ nguyên hợp đồng: thử lại ≤ 2/đơn vị dở; dừng hỏi ở crash thứ 3/đơn vị, 4/phiên, 2/kiem-khoa;
exit 2 không thử lại; traceback/timeout/thiếu log/cha gián đoạn → 40 dừng hỏi. Planner cần tự lập ngân sách thời gian: **chưa
có số đo thông lượng B thật** (fixture không đại diện); lượt chạy dài phải theo lát ≤ 420 s và tiếp tục qua nhiều lời gọi.
Rủi ro đã biết: crash native chưa rõ nguyên nhân (A: 6/14 phiên chuẩn bị; B: pid12248) có thể chạm ngưỡng 4/phiên.

### B2.9 Tự kiểm B2 và ngân sách

- Đã chạy (chỉ đọc): `git status --short --untracked-files=all src/r61_bdd configs/phase2/r61_bdd handoff/PLAN_DA_CHOT.md`
  → exit 0: ` M handoff/PLAN_DA_CHOT.md` (có từ trước lượt, review_007 xác nhận PLAN khớp hash ghim), còn lại `??` 15 file
  `.py`/`.json` như B0, không `__pycache__`. Glob: không `*.pyc` dưới `src/r61_bdd/`, không `thuc_thi/`, không `lan_03/`,
  không `lan_02/dung_fixture.json`. Grep `vong.json`: không chứa báo cáo 04.
- Không chạy: fixture (B2 cấm), băm SHA256 (quyền không có công cụ băm; các digest dẫn từ artifact fixture/review_007).
- Ngân sách: orchestrator báo 8/10 vòng đã dùng trước lời gọi này; B2 = 1 ⇒ 9/10, còn 1 cho dự phòng/đối chiếu Pbeca24e7
  (chỉ đọc, không chạy lại). Suất fixture ngoài đã dùng: B1 lan_01 2, kiểm đối số P43284b78 (không ma trận), lan_02 3, B2 0.

Lịch sử: **B0 vòng sửa 4 (theo Codex review_003 của lượt `r61_bdd_prep_b_20261026_20261026_132958`, verdict revise,
B0-F8b-1 / B0-F8b-2 / B0-F9) xong phần mã/thiết kế; CHƯA chạy fixture** (B0 cấm chạy; B1 chạy). Vòng sửa 2 (review_001,
F1–F9) ở mục 0; vòng sửa 3 (review_002) ở mục 0b; vòng sửa 4 ở mục 0c. Không có phê duyệt B thật, không có phiên B thật,
không dữ liệu thật. PLAN, gói A, khóa A, mã/hợp đồng A, `DIEM_VAO_MA` không bị sửa; hợp đồng B (`hop_dong_cong_b.json`),
`van_hanh_cong_b.py` và `thong_ke_tiep_tuc.py` không đổi ở vòng 4 (chỉ `kiem_dau_vao.py`, fixture và báo cáo này).

## P48bd9f29 / QUVZ (27/10/2026) — bước gộp B1 + Q73M + Q9MG: fixture lan_02 ĐẠT trên snapshot cuối

Bước gộp do người dùng duyệt (P48bd9f29). Một lời gọi coder; sửa duy nhất `src/r61_bdd/cong_b/test_van_hanh_cong_b.py`,
báo cáo này và ghi trong `results/phase2/r61_bdd/tests/cong_b_20261026/lan_02/`. lan_01 chỉ đọc (manifest trước/sau khớp
toàn bộ). Không sửa PLAN, A, mã/hợp đồng A, `DIEM_VAO_MA`, phép so khóa A, hợp đồng B, runner B, `kiem_dau_vao.py`,
`thong_ke_tiep_tuc.py`. Không tạo phê duyệt/phiên B thật, không chạy B thật/Nano.

**Đây là kết quả fixture CPU trên dữ liệu tổng hợp, không phải công việc B thật** (7200 tracking / 36 chấm / bootstrap
5000×200 … chưa chạy).

### Q.1 Lệnh và các lần gọi fixture ngoài (tuần tự, cùng lệnh, cùng lan_02)

`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r61_bdd/cong_b/test_van_hanh_cong_b.py --phien results/phase2/r61_bdd/tests/cong_b_20261026/lan_02 --ngan-sach-giay 420 --gioi-han-giay 540`

| lần | journal khởi động (`lan_02/khoi_dong/`) | vòng / snapshot | exit | giây (journal) | ghi chú |
|---|---|---|---|---|---|
| 1 | `20261027_052355_312_pid19660` | `vong_01` / `2d46ca9d…a6f7df` | 10 | 431,6 | chuyển phiên + kiểm cây cũ ĐẠT; ca 00–06 đạt; ca 07 dừng do hết ngân sách mềm, **1 kiểm KHÔNG ĐẠT** (dưới) |
| 2 | `20261027_053225_750_pid7216` | `vong_02` / `dd36a0d2…a7c239c` | 10 | 399,3 | snapshot mới (sửa dưới) → vòng mới, ma trận từ đầu; ca 00–06 đạt; dừng sạch trước `cr2__mo_phien` |
| 3 | `20261027_053913_590_pid18140` | `vong_02` | **0** | 309,9 | ca 07–14 đạt; `KET QUA KIEM: 244/244 dat; 142 lenh con; … -> DAT` |

Tổng thời gian fixture ngoài ≈ 1.140,8 s (< 2.700 s); 3/5 suất gọi. Không lần nào chạm trần cứng 532 s; không có `lan_03`.

**Sửa giữa lần 1 và 2 (lỗi lập lịch của fixture, không phải runner):** ở `vong_01`, `cr2_tiep_1` được khởi chạy khi ngân
sách mềm còn 2 s (`goi.json` `ngan_sach: 2.0`) → runner hết `han_mem` sau `tracking:chuan_bi`, exit 10 mà chưa làm đơn vị
nào (`tien_do.jsonl` 0004) → kiểm `cr_dang_ghi_duoc_sao_luu_roi_lam_lai` không đạt (lệnh đã checkpoint nên vòng đó không
thể đạt). Sửa `lenh()`: không khởi chạy lệnh runner khi ngân sách mềm còn < `MEM_TOI_THIEU_RUNNER = 30` s → `HetGio`
(exit 10, lần sau tiếp tục). Mã đổi ⇒ `vong_02` chạy lại đủ ma trận; `vong_01` giữ nguyên làm bằng chứng (không có
`ket_qua_test.json`). Không nới kiểm nào. Kết luận chỉ lấy từ `vong_02` (snapshot cuối).

### Q.2 Snapshot cuối (`lan_02/vong_02/vong.json`, `snapshot.sha256 = dd36a0d2c15d0a38f432ec0c177633c757922800d7a6a22a94b8712c4a7c239c`)

| file | sha256 |
|---|---|
| `src/r61_bdd/cong_b/test_van_hanh_cong_b.py` | `89baa7af585627b4cd0d7859544b43873c5390b163317ee9f850d126b3d8c5bb` |
| `src/r61_bdd/cong_b/van_hanh_cong_b.py` | `52a2e50047898df1559fc3aa52f5128d868836f80c71b7911ad42142d170a026` |
| `src/r61_bdd/cong_b/kiem_dau_vao.py` | `4f1c1cd942cd93690b239d1d5087840f999063cb5ce43307c07bb43d56e87b73` |
| `src/r61_bdd/cong_b/thong_ke_tiep_tuc.py` | `76f4691c8c25dac7994f27a112fe9b13b9bf21f68996cb2a051d539a15577c9f` |
| `configs/phase2/r61_bdd/hop_dong_cong_b.json` | `bf2ebe97af6a7d70164dbc6597fdd79cdb0a086d0d34096dc26b6c6b313c05ed` |
| `configs/phase2/r61_bdd/hop_dong_lenh.json` | `7612ca26f8cc46a40f1822cfc7644058734d7e42a8445995fada641986bf5b5c` |
| `configs/phase2/r61_bdd/giao_thuc.json` | `82ffe7aea4daa418bb7bddd323fe5a76b2413c9521c1acf23e9e061c36181097` |

(Cùng `vong.json` còn băm mọi file `r61_bdd/*.py` và đóng import tĩnh `DIEM_VAO_MA`.) Mỗi ca kiểm `<ca>_ma_khong_doi_trong_ca`
(15/15 đạt). PLAN nguyên byte: runner ghi `sha256_toan_bo 2b2474aa…e93e12a`, trước/sau mục 0 `a50c3bc4…`/`aba1aebb…`
(`goi.json` các lần chạy); ca 14 `PLAN_A_ma_cu_nguyen_byte_khong_tao_thuc_thi` đạt. Ghi chú: digest của chính
`ket_qua_test.json` không tính được trong lời gọi này (quyền không có công cụ băm ngoài fixture); B2 không ghim thêm
(P4a7fafa2 bị từ chối) — xem B2.2.

### Q.3 Kết luận `lan_02/vong_02/ket_qua_test.json` (= `ket_qua/lan_02.json`, `lich_su` 1 mục dat=true)

`dat=true`; 15/15 ca (00–14, không ca lạ); 244/244 kiểm đạt; `thieu_kiem_bat_buoc=[]`; `pyc_moi_lan {lan_01: true, lan_02:
true}`; `crash_ngoai_du_kien_vong=[]`; `bytecode_loi=[]`; `so_loi_tiem=50` (lỗi tiêm đã xác nhận theo loại);
`so_lenh_con=142`, exit `{0:33, 1:2, 2:57, 10:10, 30:10, 40:25, 0xC0000005:5}`; `giay_lenh_con=626,9`. Ca âm khớp exit và
nguyên nhân (`kiem_lenh` đòi cả exit lẫn nhóm/chuỗi). Ma trận và nội dung ca không đổi so với mục 6 (bảng dưới); dấu vết
từng ca: `vong_02/ca/NN_<ca>.json`, log lệnh `vong_02/lenh/<tên>.{json,stdout.txt,stderr.txt}`.

**Đính chính:** ca 00 `thiet_lap` có **6** kiểm, ca 01 `bao_ve_lan_goi` có **7** kiểm (tổng hai ca 13; mục "B1 lời gọi Claude 1"
cũ ghi "9/9" cho ca 01 là sai). 13 không phải tổng ma trận — tổng ma trận là 244.

Nhóm bằng chứng đạt (đều trong `vong_02`): đường thành công + bootstrap chia khối = nguyên khối (dòng/mẫu/thứ tự, 108 CI,
byte CSV, `kiem_lai_tong_hop=[]`) + tái tính hậu kiểm phát hiện đầu ra bị sửa (ca 02); exit 10/resume không chạy lại đơn vị
hoàn tất (ca 02, 07); crash quanh ghi/marker + sao lưu artifact dở + `.dang_ghi` + marker hỏng exit 2 (ca 07); cha gián đoạn
(ca 08); Job Object/resume/PID/traceback/thiếu log/timeout 59,7 s (ca 09); ngưỡng 3/đơn vị, 4/phiên, 2/kiem-khoa + giữ chặn
(ca 10); khởi động bị ngắt (ca 11); sổ crash F8b/F8c (ca 12); F8b-1/F8b-2 (ca 13); kết thúc: mọi lệnh có exit, cây thu hồi
(`cay.active=0`), trong job gốc từ lúc tạo, `[python, -B, …]`, hạn con ≤ hạn chung − 15 (ca 14); thu hồi cây tham dò (ca 01).

### Q.4 Q73M — cha fixture không nạp ultralytics/torch (bằng chứng chạy, không phải phân tích tĩnh)

`ben_thu_ba()` đọc `sys.modules` THỰC TẾ của chính tiến trình (không xóa module), gắn `pid`, `thoi_diem_tao` (FILETIME của
tiến trình), `lan`, `snapshot`, `diem`. Điểm ghi: mỗi lệnh con (`lenh/<tên>.json`), mỗi ca (`ca/*.json`), mỗi bước thiết lập
(`thiet_lap/*.xong.json`), đầu lần gọi (`lan_goi/NN.bat_dau.json`), kết thúc lần gọi (`lan_goi/NN.ket_thuc.json`),
kết thúc khởi động (`khoi_dong/<id>.ket_thuc.json`), `tong_ket`. `tong_ket` đòi mọi điểm sạch (thiếu = không đạt).

Kết quả `vong_02`: `cha_khong_nap_ultralytics_torch=true`; `ben_thu_ba_cha_diem_kiem.so_diem=164` (lệnh 142, ca 15, thiết lập
7), `khong_sach_hoac_thieu=[]`. Từng lần gọi: lan 1 pid 7216 (tạo `134375275456375705`), lan 2 pid 18140 (tạo
`134375279534792151`): `da_nap = luc_khoi_dong = ["_distutils_hack"]` (gói `.pth` nạp sẵn lúc khởi động), `them_sau_khoi_dong=[]`,
`cam_co_mat=[]`, `ngoai_cho_phep=[]` — cha **không nạp cả numpy**. Lần 1 của lan_02 (pid 19660, `vong_01`): journal kết thúc
cũng `cam_co_mat=[]`. Các con `--viec`: `vong_02/viec/` 7 file (`the_gioi`, `A_tien_kiem`, `A_cache__{yolov8n_coco,
yolov8n_mixed5,yolo11n_mixed5}`, `A_goi`, `doi_chieu_bootstrap`) đều exit 0, có log lệnh `lenh/viec_*_00.*`; 6 file đầu có
`ben_thu_ba_sau.cam_co_mat` khác rỗng (con nạp ultralytics/torch như dự kiến), `doi_chieu_bootstrap` chỉ pandas/numpy.

### Q.5 Q9MG — chuyển phiên lan_01 → lan_02, kiểm cây cũ, kế thừa sự cố

Cơ chế (`CHUYEN_PHIEN`, `chuyen_phien()`; chỉ áp cho đúng `--phien …/lan_02`, chạy TRƯỚC rà sổ/chọn vòng/mọi lệnh con):
1. Lần đầu: `chuyen_phien/manifest_lan_01_truoc.json` (274 file, đường tương đối/số byte/sha256; `sha256_tep 550e3e28…ab2c1c0`;
   file 38.063 B `4e4f0f38…aa075cb`) → `kiem_cay_cu_<id>.json` → đối chiếu `dung_fixture.json`/sổ của lan_01 với danh sách
   lý do bao phủ ghim trong mã → nhận diện cùng sự cố → `quyet_dinh.json` (ghi một lần; 7.800 B, `f63fbac4…4b38c79`).
2. Mọi lần gọi: kiểm lại mọi tham chiếu (số byte + sha256: sổ, `dung_fixture.json`, 5 file `khoi_dong/`, `lan_goi/01.bat_dau`,
   `tc_mo_phien.bat_dau`, `con.json`/`goi.json` của 0002, log faulthandler pid12248, manifest trước, kiểm cây) và manifest
   lan_01 hiện tại == trước; lệch → `dung_fixture.json` của lan_02 + 40 (giữ chặn). Cuối mỗi lần gọi:
   `manifest_lan_01_sau_<id>.json`; `tong_ket` đòi mọi bản sau == trước.

**Kiểm cây cũ** (`kiem_cay_cu_20261027_052355_312_pid19660.json`, 10.023 B, `6df0c688…a9f044e6`, `dat=true`): Toolhelp32 liệt kê
280 tiến trình; 16 PID ghi trong lan_01 (fixture 12248 qua journal khởi động + lan_goi, fixture 8652, 6 lệnh con có FILETIME,
2 con runner 16896/15728 có FILETIME từ `con.json`, 3 cháu tham dò, 2 cha runner 14340/23416 từ tên log) — cả 16: không tiến
trình nào mang PID (OpenProcess lỗi 87); với PID có FILETIME phép so là PID + thời điểm tạo, PID chỉ có mốc journal thì
"tiến trình hiện tại tạo sau mốc + 2 s = PID tái dùng". Quét cả hệ thống: không tiến trình sống nào có ppid thuộc 16 PID hoặc
là `python.exe` tạo trong cửa sổ lan_01 (`lien_quan_con_song=[]`). Đối chiếu journal: `khoi_dong/…pid12248` có `bat_dau`,
**thiếu** `ket_thuc`; `vong_01/lan_goi/01` thiếu `ket_thuc`; log pid12248 có "Windows fatal exception"; journal 8652 có
`ket_thuc` (exit 40, `vong=null`). Lệnh `tc_mo_phien` (đang mở, không ghi PID) được phủ bởi log cha runner pid23416 +
`con.json` 0002 + quét ppid.

**Lý do dừng được ngoại lệ** (đúng 3 dòng `lan_01/dung_fixture.json`, ghim trong mã, so nguyên chuỗi):
`khoi_dong/20261027_030727_163_pid12248: fixture_gian_doan`; `vong_01/lan_goi/01: fixture_gian_doan`;
`lan_chay_mo:…/lan_01/vong_01/phien/thanh_cong/van_hanh/lan_chay/0002_mo-phien: lan_chay_runner_dang_mo_cha_gian_doan`.
**Cùng sự cố** (`lien_ket_su_co.dat=true`, 8 điều kiện: lan_goi 01 thuộc khởi động pid12248; vòng chỉ một lần gọi; lệnh đang mở
duy nhất là `tc_mo_phien`, bắt đầu sau lan_goi 01; 0002 thiếu `ket_thuc`, con tạo sau lệnh, argv trỏ 0002; lần gọi sau không
chọn vòng) → cả 3 bản ghi `su_co = khoi_dong:20261027_030727_163_pid12248`. **Bộ đếm kế thừa:** 3 bản ghi, 1 sự cố, xếp
native (log "Windows fatal exception") ⇒ tính 1 vào ngưỡng 4 crash native ngoài dự kiến của lan_02 (không reset, không đếm
trùng); 3 lỗi tiêm của lan_01 chỉ ghi nhận số. Sổ lan_02 chứa 3 mục `ke_thua:lan_01:*` (`dung_ngay=false` chỉ khi khóa thuộc
danh sách ghim VÀ đã xác minh trong lần gọi đó — `_dung_ngay`). Mọi lỗi mới của lan_02 theo chính sách hiện hành (không lỗi
mới nào xảy ra: `crash_ngoai_du_kien_vong=[]`). `ket_qua_test.json.chuyen_phien_lan_01.dat=true`, `lech=[]`, 3 manifest sau
(pid19660/7216/18140) đều 274 file, `sha256_tep 550e3e28…` = trước.

Giới hạn: `quyet_dinh.json` tự nó chỉ ràng buộc qua sha256 ghi trong các `lan_goi`/`ket_qua` sau, không ghim ngoài; phần ngoại
lệ được giới hạn bởi danh sách khóa ghim trong mã. Kiểm cây cũ là ảnh chụp tại 05:24 (một lần, trước ma trận).

**Event log:** phiên điều phối KHÔNG cung cấp kết quả event log Windows trong đầu vào lời gọi này; lời gọi này không truy vấn
event log (đề xuất P05759f93 bị từ chối) và không suy diễn. Nguyên nhân crash native của pid12248 vẫn CHƯA RÕ.

### Q.6 Bytecode và đường ghi

Mỗi lần gọi chụp `.pyc` trước import và sau thu hồi cây (`vong_02/pyc/lan_0{1,2}_{truoc,sau,so_sanh}.json`; `vong_01/pyc/lan_01_*`):
6.453 pyc trong phạm vi, thêm/mất/đổi 0/0/0 cả trong lần gọi và so với đầu vòng (3/3 lần gọi). Không xóa pyc nào.
`git status --short --untracked-files=all src/r61_bdd …` sau chạy: không có `__pycache__`. Cha: `orig_argv` có `-B`,
`flags.dont_write_bytecode=1`, `sys.dont_write_bytecode=true`, `PYTHONDONTWRITEBYTECODE=1`; ca 14 `moi_lenh_python_tuyet_doi_co_B`,
`chau_tham_do_python_tuyet_doi_co_B`, ca 02 `tc_moi_cha_con_dung_interpreter_va_B` đạt. Đường ghi: kiểm lexical/realpath/reparse
trước lần ghi đầu (ca 04 traversal/junction, ca 05) đạt; ghi mới chỉ dưới `lan_02/`.

### Q.7 Thời gian theo loại đơn vị (`vong_02`, từ log lệnh)

Bước `--viec` 4,4–12,1 s (7 lệnh, ≈ 43 s); lệnh runner nặng (kiểm đủ đầu vào + ảnh khóa) 6–13 s; runner dừng ở kiểm đầu vào
0,2–1,8 s; fixture con (ca 11–13) 4,3–4,6 s; tham dò 0,3 s / trần cứng 8,35 s; timeout runner 59,7 s; cha gián đoạn 0,5–7 s.
Tổng thời gian lệnh con 626,9 s; phần còn lại là chụp pyc (2 lần/lời gọi), kiểm trong tiến trình và ghi bằng chứng.

### Q.8 Tự kiểm và ngân sách

Đã chạy: 3 lần lệnh fixture trên (exit 10, 10, 0); `git status --short --untracked-files=all src/r61_bdd handoff/PLAN_DA_CHOT.md
configs/phase2/r61_bdd` (exit 0: ` M handoff/PLAN_DA_CHOT.md` có từ đầu lượt, không sửa; còn lại `??`, không `__pycache__`).
Rà tĩnh trước chạy: đường vào `chuyen_phien` bắt mọi ngoại lệ (trước quyết định → 40 không ghi chặn; sau → chặn), `ra_soat`
chỉ thêm mục kế thừa cho đúng phiên lan_02, các phiên giả của ca 11–13 không bị ảnh hưởng. Chưa chạy: B thật, Nano, mọi
việc ngoài fixture. Ngân sách: orchestrator báo 7/10 vòng đã dùng (gồm lời gọi này); còn 3 cho Codex review bước gộp → B2
(1) → đối chiếu Pbeca24e7 (chỉ đọc, không chạy lại).

## P43284b78 (27/10/2026) — bước cần thư viện ngoài chuyển sang lệnh con; cha fixture chỉ thư viện chuẩn + numpy

Quyết định người dùng P43284b78 (approve). Chỉ sửa `src/r61_bdd/cong_b/test_van_hanh_cong_b.py` và báo cáo này; không
đổi ma trận ca (15 ca 00–14, cùng tên, cùng danh mục kiểm bắt buộc `yeu_cau`), không đổi lệnh B1, hợp đồng B, runner B,
`kiem_dau_vao.py`, `thong_ke_tiep_tuc.py`, mã/khóa A, PLAN. Lý do: ở lan_01 cha fixture đã nạp ultralytics trong tiến trình
(ca 00) trước khi crash native ở ca 02 (mục B1 dưới) — đây là biện pháp cô lập, KHÔNG phải chẩn đoán nguyên nhân crash.

| thay đổi (hàm) | nội dung |
|---|---|
| chế độ mới `--viec` (`viec`) | `[python, -B, test_van_hanh_cong_b.py, --viec <ten> <vong> <han> <mem> <lenh>]`: một bước; kiểm đối số (tên trong `VIEC` hoặc `A_cache:<det>` có trong giao thức fixture, vòng `vong_NN` có `vong.json` dưới `tests/cong_b_20261026/<phien>/`, tên lệnh ràng buộc tên bước, `han` chưa qua), `VB.kiem_bytecode()` (-B), faulthandler → stderr của lệnh + trần cứng `han − 2 s`, `H.chuan_bi_moi_truong(vong)` TRƯỚC import bên thứ ba, rồi chạy đúng hàm cũ (`tl_the_gioi`, `tl_A_tien_kiem`, `tl_A_cache`, `tl_A_goi`, `vc_doi_chieu`). Kết quả `<vong>/viec/<lenh>.json` (mở không ghi đè) kèm `ben_thu_ba_truoc/sau` của con. Exit 0 xong; 10 `HetGio` (cache tạm dừng); 2 đối số/`LoiHopDong`; traceback 1. Không ghi `khoi_dong/`, không chụp pyc, không tạo con |
| `viec_con` (cha) | gọi qua `lenh(..., script=THIS)` sẵn có ⇒ job gốc + job lệnh, `IsProcessInJob` trước resume, hạn chung `T.HAN` (không cấp lại ngân sách; con nhận `han`/`mem` của lần gọi), `can_tg` trước khi khởi chạy, `ap_so_sau_lenh` ngay sau lệnh (exit 1/native/timeout của con = `muc` dừng ngay theo chính sách hiện hành). Mỗi lần thử = lệnh mới `viec_<ten>_NN` (NN = số kết quả đã có) ⇒ lệnh đã xong không chạy lại; bị ngắt giữa lệnh ⇒ `bi_gian_doan` như mọi lệnh |
| `ca_thiet_lap` | `the_gioi`, `A_tien_kiem`, `A_cache_<det>` (tiếp tục theo video như cũ), `A_goi` (gồm `D.anh_khoa` + `VA.phu_luc`) chạy qua `viec_con`; checkpoint `buoc_tl` giữ nguyên; `hop_dong` (chỉ thư viện chuẩn) giữ trong tiến trình |
| `ca_thanh_cong` | đối chiếu bootstrap chia khối/nguyên khối (`D.doc_cham` nạp pandas) chuyển thành `vc_doi_chieu` chạy qua `viec_con`, vẫn bọc `nho(..., can=150)`; cùng 4 kiểm, cùng dữ liệu trả về |
| bằng chứng `sys.modules` (`ben_thu_ba`) | chỉ thư viện chuẩn: tên gốc module nạp từ site-packages (`purelib`), phần nạp thêm sau khởi động (trừ gói `.pth` nạp sẵn lúc khởi động, ghi `luc_khoi_dong`), `cam_co_mat` (`ultralytics`, `torch` — kiểm theo tên trong `sys.modules`, kể cả module con), `ngoai_cho_phep` (ngoài `numpy`). Ghi vào `lan_goi/NN.ket_thuc.json` và `khoi_dong/<id>.ket_thuc.json` (`ben_thu_ba_cha`) |
| `tong_ket` | thêm `cha_khong_nap_ultralytics_torch` + `ben_thu_ba_cha_moi_lan` (lần này: trực tiếp; lần trước trong vòng: journal `ket_thuc`; thiếu = không đạt); `dat` chỉ true khi mọi lần gọi của vòng không có `cam_co_mat` và `ngoai_cho_phep` rỗng. Đây là điều kiện kết luận, không phải ca/kiểm mới |
| `gan_vong` | gom gán `T.FX/T.A/T.HD/T.GT` dùng chung cho `chay()` và `viec()` (cùng giá trị như trước) |

Tự kiểm lời gọi này (Tier 1, không ghi gì): (1) `python -B …test_van_hanh_cong_b.py --phien …/cong_b_20261026/khong_ton_tai_kiem_doi_so --ngan-sach-giay 0 --gioi-han-giay 540`
→ exit 2 (thông báo đối số; import module + `chup_pyc` chạy không lỗi cú pháp/import); (2) `… --viec the_gioi …/khong_ton_tai_kiem_doi_so/vong_01 1 1 viec_the_gioi_00`
→ exit 2 "viec: doi so sai" (đường vào chế độ con, kiểm đối số trước mọi ghi). Sau đó `git status` thư mục `khong_ton_tai_kiem_doi_so`
rỗng, không có `.pyc` dưới `src/r61_bdd/`. Lần thử đầu của (1) với `--gioi-han-giay 1` dừng bởi trần cứng 1 s
(exit 1, "Timeout (" trong `chup_pyc`) — đúng thiết kế F1, không ghi file. **Chưa chạy ma trận**: bước này không có
quyền ghi `results/…/lan_02/` và `lan_01` bị chặn (exit 40); chứng minh cha không nạp ultralytics/torch trên thực tế
(`ket_qua_test.json.cha_khong_nap_ultralytics_torch`, `ben_thu_ba_cha` từng lần gọi, `viec/*.json` cho thấy con nạp)
thuộc bước Pbeca24e7 (lan_02). Giới hạn: thêm ~6 tiến trình con (mỗi con import ultralytics/torch, ước vài–10 s mỗi con —
chưa đo) vào thời gian ca 00/02; tracker/evaluator vẫn chạy trong runner B như trước.

## B1 lời gọi Claude 1 (27/10/2026) — BLOCKED: fixture crash native, đã dừng hỏi

Hai lần gọi fixture ngoài, đúng lệnh mục 6 (`-B`), tuần tự; không sửa mã ở lời gọi này.

| lần | khởi động (journal) | exit | giây | kết quả |
|---|---|---|---|---|
| 1 | `lan_01/khoi_dong/20261027_030727_163_pid12248` | **139** (Git Bash "Segmentation fault" = 0xC0000005) | ~48 tới crash | vòng `vong_01` tạo; ca 00 `thiet_lap` 6/6 kiểm đạt, ca 01 `bao_ve_lan_goi` 9/9 đạt; ca 02 `thanh_cong`: `tc_tien_kiem` exit 0 (7,7 s), `tc_mo_phien` đang chạy thì tiến trình fixture chết |
| 2 | `…/20261027_030955_292_pid8652` | **40** | 5,5 | rà sổ trước chọn vòng → `dung_fixture.json` (3 mục `dung_ngay`): journal khởi động lần 1 thiếu `ket_thuc` (`fixture_gian_doan`), `vong_01/lan_goi/01` thiếu `ket_thuc`, lần chạy runner `phien/thanh_cong/.../0002_mo-phien` mở (`lan_chay_runner_dang_mo_cha_gian_doan`); không chọn vòng, không chạy ca; journal khởi động lần 2 có `ket_thuc` exit 40 |

Bằng chứng crash: `vong_01/faulthandler_lan_01_pid12248.log` = "Windows fatal exception: access violation", chỉ một luồng
Python được in dạng `Thread 0x000047d8` (KHÔNG phải `Current thread`) đang ở `subprocess.wait` ← `khoi_chay_trong_job:340`
← `lenh:402` ← `ca_thanh_cong:926`. Quan sát: luồng gây lỗi không phải luồng Python (luồng native trong tiến trình
fixture); nguyên nhân CHƯA RÕ. Trong ca 00 fixture đã nạp ultralytics trong tiến trình (stdout "Creating new Ultralytics
Settings" vào `vong_01/_cau_hinh_thu_vien/`) — chỉ là bối cảnh, không phải chẩn đoán. Cùng dạng crash native tiến trình
test dài hạn chưa rõ nguyên nhân ở chuẩn bị A (02 mục (1): 6/14 phiên). Runner con `0002_mo-phien`: `con.json` có,
`faulthandler.log`/`stderr.txt` rỗng, `tien_do.jsonl` chỉ có `bat_dau` — bị kết thúc theo job gốc KILL_ON_JOB_CLOSE (suy
từ thiết kế; lời gọi này không có quyền liệt kê tiến trình để xác nhận độc lập). 6 lệnh con đã xong đều `cay.active = 0`,
`trong_job_goc = true`; exit tham dò: ngắt giữa Popen/Assign `0xC0000005`, lỗi Python 1, trần cứng 1 (8,2 s), thiếu dự
phòng 10.

Chưa có: `ket_qua_test.json`, so sánh pyc (`pyc/lan_01_sau.json` không có vì lần 1 chết; lần 2 dừng trước khi ghi pyc),
ca 02–14. Theo thiết kế (mục 6, rủi ro (7)) và ràng buộc "trạng thái an toàn không xác định phải dừng hỏi", mọi lần gọi
đúng lệnh B1 sau này sẽ exit 40 cho tới khi người dùng quyết định; lệnh B1 cố định `--phien …/lan_01` nên không thể tự
chạy lại trên phiên mới.

## 0c. Sửa theo review_003 (F8b-1, F8b-2, F9)

| finding | sửa (file:hàm) | ca hồi quy mới trong fixture |
|---|---|---|
| F8b-1 [P1] cha runner crash / runner tự dừng vì native vẫn được tiếp tục | fixture `bat_thuong_lenh`: chỉ **exit 30** (runner báo con crash native, cây đã thu hồi) là crash con được đếm ngưỡng; tiến trình trực tiếp (cha runner / fixture con / thăm dò) tự crash native (`native and exit != 30`) = cha gián đoạn → `dung_ngay=true` nếu không có tiêm `cha_*` giải thích (exit 30 chỉ là `tiem` khi tiêm giải thích ra `crash_native@…`, không còn "có `--tiem-loi` bất kỳ"). **exit 40 không còn bỏ qua**: `giai_thich_dung_40` — fixture con: phải có `dung_fixture.json` của phiên con; runner B: `van_hanh/dung.json` phải trỏ `stt` một lần chạy đã có `ket_thuc` với `quyet_dinh.exit_cha = 40` và mọi bất thường của lần chạy đó được tiêm của CHÍNH nó giải thích; không khớp → `muc`, dừng ngay. `_xet_phien`: lần chạy **thiếu `ket_thuc`** (rà sổ chỉ chạy khi không còn runner sống) → khóa riêng `lan_chay_mo:` → `tiem` chỉ khi `goi.tiem_loi` của chính lần chạy có điểm `cha_*`, còn lại `muc` dừng ngay; lần chạy đóng với `ko == ["crash_native"]` chỉ được tiếp tục khi có `don_vi` **và** `quyet_dinh.exit_cha != 40` (runner dừng vì không xác định đơn vị / chạm ngưỡng → giữ dừng). Kiểu sai của `goi`/`ket_thuc`/kết quả lệnh → `muc` "doc_loi", dừng | ca 13 `so_crash_bo_sung` (tổng hợp dưới `<vòng>/so_gia/`): 5 ca âm — cha runner native không tiêm (lệnh + lần chạy mở); lần chạy mở với lệnh exit 0; exit 40 không có `dung.json`; exit 40 + `dung.json` → crash không đơn vị (quyết định 40); exit 40 + `dung.json` → ngưỡng do crash không tiêm — mỗi ca: `DungFixture` ngay, mọi `muc` `dung_ngay`, `su_co_native` rỗng (không đếm vào ngưỡng 4), `vong_02` snapshot khác rồi `cap_nhat_so` vẫn dừng, `dung_fixture.json` nguyên byte; 2 ca đối chứng có tiêm khớp (exit 40 ngưỡng có tiêm; cha native có tiêm `cha_khi_con_chay`) → không dừng, chỉ `tiem`; lần gọi fixture con thật trên phiên cha-native → 40, không tạo `vong_03`. Các ca thật 08/09/10 (cha gián đoạn, Job/PID/log/timeout, ngưỡng) nay đi qua `giai_thich_dung_40` với `dung.json` thật — nếu giải thích sai, fixture dừng ngay (lộ ở B1) |
| F8b-2 [P1] sổ JSON hợp lệ sai cấu trúc → AttributeError, ca ghi không đạt rồi chạy tiếp | `loi_so`: sổ phải là object; `muc`/`tiem` là danh sách; mỗi phần tử object có `khoa`/`loai` chuỗi, `su_co`/`vong` chuỗi hoặc null; `muc` có `native`/`dung_ngay`/`an_toan_xac_dinh` bool. `ra_soat` kiểm (cùng reparse / không phải file) TRƯỚC mọi dùng sổ cũ → `LoiHopDong`, không ghi sổ. `LOI_SO` (LoiHopDong, OSError, ValueError, TypeError, KeyError, AttributeError, IndexError) bắt ở `ap_so_sau_lenh` (→ `DungFixture`), khởi động `chay()` (→ 40 trước chọn vòng) và kết thúc lần gọi (→ 40, vẫn chụp pyc + ghi `lan_goi/NN.ket_thuc.json`); cả ba gọi `dung_so_hong` ghi `dung_fixture.json` MỘT lần, **không sửa/tạo lại sổ** | ca 13: sau lệnh, 4 biến thể (`muc:[null]`, `muc:{}`, `tiem:["x"]`, `muc` thiếu bool) → `DungFixture`, sổ nguyên byte, `dung_fixture.json` ghi lý do "khong dung duoc"; lần gọi fixture con thật sau đó → 40, sổ + `dung_fixture.json` nguyên byte, không tạo `vong_02`. Khởi động (fixture con thật): `muc:[null]` → 40 "sai cau truc", sổ nguyên byte, có `dung_fixture.json`, không tạo vòng; lần gọi sau vẫn 40, nguyên byte; `muc:{}` → 40. Kiểm `DungFixture` không phải `Exception` (không bị `except Exception` của `main()` nuốt) |
| F9 [P2] `tracker.fps=0` → ZeroDivisionError exit 1 | `kiem_dau_vao.nap_ctx`: sau schema, `tracker.fps` phải không phải bool, hữu hạn, > 0 → `LoiDauVao("giao_thuc")`; thêm `ArithmeticError` vào nhóm lỗi chuyển thành `LoiDauVao` (không sửa helper A) | ca 04 thêm `am_json_giao_thuc_fps_0` (giao thức fixture `fps=0`, dựng ở thiết lập vòng): exit 2, nhóm `giao_thuc`, "phai la so huu han > 0", không tạo thư mục phiên |

Giới hạn vòng 4: phép giải thích exit 40 dựa vào `dung.json` hiện tại của phiên (một file, ghi một lần); lệnh "giữ chặn"
không tạo lần chạy mới nên được giải thích bằng chính lần chạy đã gây dừng. Nhánh kết thúc lần gọi bắt `LOI_SO` bao cả
`tong_ket` → lỗi lập trình trong `tong_ket` cũng thành dừng hỏi (bảo thủ có chủ ý; trước đây là exit 1 không `ket_thuc`,
lần sau cũng dừng).

## 0b. Sửa theo review_002 (F8a/F8b/F8c, F9 phần còn lại)

| finding | sửa (file:hàm) | ca hồi quy mới trong fixture |
|---|---|---|
| F8a [P1] crash trước journal mất lịch sử | fixture, mức module, **chỉ thư viện chuẩn**: `_doi_so_som` (cùng điều kiện đối số với `chay()`) → `khoi_dong_som`: `_kiem_duong_som` (lexical dưới `tests/cong_b_20261026/`, realpath = abspath, không reparse ở mọi thành phần) → `_tao_som` (mkdir từng cấp, kiểm lại sau mkdir) → `<phien>/khoi_dong/<id>.bat_dau.json` (mở `"x"`, fsync; ghi `ghi_luc_epoch`, `module_du_an_da_nap` = [] ) + `<id>.faulthandler.log` (trần cứng ghi vào đây tới khi `bao_ve_lan_goi` đặt lại) — TRƯỚC `chup_pyc()` và mọi import mã dự án. `ket_thuc_khoi_dong` ghi `<id>.ket_thuc.json` ở mọi đường `return` của `chay()`. `chay()` từ chối (2) nếu không có journal. `lan_goi/NN.bat_dau.json` ghi `khoi_dong`=id. `ra_soat` (toàn bộ) quét `khoi_dong/` ở mức PHIÊN (không theo vòng) → thiếu `ket_thuc` (trừ lần gọi hiện tại) = `fixture_gian_doan`/`tran_cung_watchdog` (tìm `Timeout (` trong log khởi động + log vòng cùng PID), an toàn chưa xác định → dừng | ca 11 `khoi_dong_ngat`: journal của lần gọi này ghi trước `chup_pyc` (so `ghi_luc_epoch` ≤ `t0` chụp pyc, không module dự án nào đã nạp); tham dò `ngat_khoi_dong` (journal phiên con rồi `faulthandler._read_null()` ở mức module, trước chụp pyc/import) → exit native, cây thu hồi, journal thiếu `ket_thuc`, không có vòng; `ra_soat` phát hiện + `ly_do_dung`; lần gọi fixture con kế tiếp (đúng entrypoint, phiên con) → 40 "DUNG HOI"; tạo `vong_01/vong.json` snapshot khác rồi gọi lại → vẫn 40, `dung_fixture.json` nguyên byte, không tạo `vong_02`/`lan_goi`; sổ giữ đúng 1 sự cố khởi động |
| F8b [P1] chưa giữ dừng với lỗi ngoài dự kiến; áp ngưỡng chỉ sau `main()` | `giai_thich_tiem` + `bat_thuong_lan_chay` + `bat_thuong_lenh`: lỗi tiêm **đã xác nhận theo loại** (điểm/kiểu tiêm giải thích đúng phân loại: native@đơn vị → `crash_native`, python → `loi_python`, chờ → `het_gio`, cha_* → `cha_gian_doan`, bo_faulthandler → thiếu log, job/resume → Job, pid_khong_xac_dinh/thiếu con.json → PID; tham dò: exit đúng kỳ vọng của loại) → `tiem`; còn lại → `muc`. Mọi `muc` không phải crash native đã thu hồi (timeout, traceback, exit 1, thiếu log, Job/resume, PID, cha gián đoạn, fixture bị ngắt, cây chưa thu hồi, sổ/ket qua hỏng) có `dung_ngay=true` → `ly_do_dung` dừng ngay. `lenh()` gọi `ap_so_sau_lenh` NGAY sau khi ghi kết quả mỗi lệnh (cả nhánh `bi_gian_doan`): `ra_soat(chi=(lệnh, phiên B))` → ghi sổ → `dung_fixture.json` → ném `DungFixture` (BaseException, không bị `except Exception` của ca nuốt; `chay()` → 40). Sổ hỏng khi đang chạy → `DungFixture`; lúc khởi động → 40 | ca 12 `so_crash` (bằng chứng tổng hợp đúng cấu trúc `lenh()`/`mot_doan()` dưới `<vòng>/so_gia/`, ngoài phạm vi sổ chính): timeout / traceback / thiếu log ngoài dự kiến → `ap_so_sau_lenh` ném `DungFixture` + `dung_fixture.json`; rồi `vong_02` snapshot khác + lần gọi fixture con → 40, `dung_fixture.json` nguyên byte, không tạo `vong_03`; tiêm khớp loại → `tiem`, không dừng; tiêm lệch loại (khai python, ra het_gio) → ngoài dự kiến, dừng; **bằng chứng thật** lần chạy `loi_python` của ca 09 → `tiem`; bản sao bỏ `tiem_loi` → ngoài dự kiến, dừng |
| F8c [P2] một crash native bị đếm hai lần | mỗi bản ghi có `su_co`: lệnh có `phien` + `lan_chay_stt` (lệnh ghi thêm) → `lan_chay:<phiên>/<stt>`, trùng định danh với bản ghi `ket_thuc` lần chạy; khởi động/lan_goi cùng id → `khoi_dong:<id>`. Giữ cả hai nguồn bằng chứng (`khoa` riêng), `su_co_native` đếm tập `su_co` duy nhất; ngưỡng 4 áp trên số sự cố | ca 12: 1/2/4 sự cố (mỗi sự cố 2 nguồn) → bằng chứng 2/4/8, sự cố 1/2/4, dừng chỉ ở 4, ra soát 3 lần không tăng; phiên 2 sự cố + `vong_02` (snapshot khác) thêm 2 sự cố → 4, dừng, không reset vòng cũ |
| F9 [P2] còn lỗi Python | `kiem_dau_vao.nap_ctx`: đọc giao thức qua `doc_json_dv` + `SCHEMA_GIAO_THUC` + kiểm `bo_loc`, đọc manifest nó trỏ tới (`videos` = danh sách chuỗi), rồi mới `D.Ctx` + `runs()` + `detectors()`; lỗi còn lại (`ValueError/TypeError/KeyError/AttributeError/OSError`, `LoiHopDong` của Ctx) → `LoiDauVao("giao_thuc")` giữ nguyên nhân thô. `van_hanh_cong_b.main_cha` và `kiem_lan_con` dùng `KD.nap_ctx` (không sửa helper A). `kiem_lan_con`: `SCHEMA_GOI_CON` (stt int, han_mem số, tiem_loi danh sách object, buoc trong danh mục, gioi_han_don_vi int/null) TRƯỚC mọi `int()`/`float()` → `LoiDauVao("lan_chay")` exit 2 | ca 04 thêm: giao thức cắt cụt → 2 `[giao_thuc]` "khong phai JSON hop le"; manifest trỏ tới sai kiểu → 2 "videos phai la danh sach chuoi"; `tracker.fps` sai kiểu → 2 "tracker.fps sai kieu"; ba ca đều không tạo thư mục phiên; `--con` với `goi.json` `stt=null` → 2 `[lan_chay]` "stt sai kieu", thư mục lần chạy chỉ còn `goi.json` |

Khoảng hở còn lại của F8a (giới hạn đã biết): từ khi tiến trình bắt đầu tới khi journal ghi xong chỉ gồm đọc `sys.argv`,
kiểm đường và `mkdir` bằng thư viện chuẩn (chưa chụp pyc, chưa import mã dự án, chưa tạo con, chưa ghi gì khác); crash đúng
trong khoảng này không để lại journal nhưng cũng chưa có tác dụng phụ nào cần thu hồi.

## 0. Sửa theo review B0 (F1–F9)

| finding | sửa (file:hàm) | ca hồi quy mới trong fixture |
|---|---|---|
| F1 [P1] deadline chưa bao toàn lời gọi | fixture: **trần cứng** `faulthandler.dump_traceback_later(exit=True)` đặt ngay đầu module (trước chụp pyc/import mã dự án) ở `bắt đầu + --gioi-han-giay − 8 s`, đặt lại vào log của lần gọi (`bao_ve_lan_goi`); **hạn mềm** `can_tg` trước mọi việc trong tiến trình: mỗi ca (15 s), mỗi bước thiết lập (`buoc_tl`), tái tính bootstrap (`nho(..., can=150)`), ca kết thúc (60 s); dự phòng kết thúc `max(45, 2×thời gian chụp pyc + 25)` s; cache A tổng hợp nhận ngân sách = thời gian còn lại − 30 s (bỏ `1e9`), dừng sạch theo video rồi tiếp tục (`tam_dung`) | `bao_ve_lan_goi`: tham dò `tran_cung_trong_tien_trinh` (việc trong tiến trình vượt trần 8 s → exit 1, log "Timeout (", cháu chết, cây về 0 không cần fixture terminate); `thieu_du_phong` (không đủ dự phòng → exit 10, không bắt đầu việc) |
| F2 [P1] khoảng hở Popen → Assign | fixture tự gán **job gốc** KILL_ON_JOB_CLOSE (`VA.gan_job_cha`) trước khi tạo con: con thuộc job gốc ngay từ CreateProcess; `khoi_chay_trong_job` kiểm `IsProcessInJob(job gốc)` trước khi resume (không → giết, không chạy), lỗi Python ở bất kỳ đâu → giết con + `TerminateJobObject` rồi ném lại | tham dò `ngat_giua_popen_gan` (crash native đúng giữa Popen và Assign → cháu chết theo job gốc, không cần terminate); `loi_python_giua_popen_gan` (RuntimeError → dọn dẹp); kết thúc: `moi_lenh_con_trong_job_goc_tu_luc_tao` |
| F3 [P1] phục hồi ghi theo journal không kiểm | `phuc_hoi`: journal là đầu vào không tin — JSON/schema, tên journal phải = `sha256(duong)[:16]`, đường tuyệt đối; ngay trước MỌI move/restore kiểm lexical + realpath + reparse cả nguồn và đích; đích chỉ trong `fx/` hoặc `phien/` (`vung_sua`); `sua_tam` kiểm trước lần ghi đầu | `phuc_hoi_journal`: traversal, junction ở cha, journal không ràng buộc tên, `sua_tam` ngoài vùng — đều chặn và không tạo/di chuyển file; phục hồi hợp lệ đạt |
| F4 [P1] bỏ qua bằng marker cũ | runner: `tien-kiem` đã xong vẫn `kiem_tat_ca` hiện tại + so manifest với kết quả lưu (ghi `kiem_lai_lan_NNNN.json`), `mo-phien` đã xong gọi `nap_b` (kiểm đủ đầu vào + manifest B + phien.json) trước khi `bo_qua`; artifact hoàn tất giữ nguyên | `kiem_lai_buoc_xong`: cả 2 bước × (PLAN mục 0 đổi → 2 [plan], `dung.json` của A → 2 [van_hanh], phụ lục đổi → 2 [phu_luc], khôi phục → 0 có sự kiện kiểm lại); artifact hoàn tất nguyên byte |
| F5 [P1] hậu kiểm tin bảng tái lập/chấm lại | runner: `kiem_lap`/`kiem_cham_lai` kiểm lại marker + băm từng file + so byte từng run; `_kiem_bang` so bảng đã ghi với bảng tính lại; dùng ở nhánh bỏ qua của `lap-tracker`/`cham-lai` và trong `hau-kiem` | `thanh_cong`: sửa `lap/.../data/<video>.txt` và `cham_lai/<run>/raw_fields.json` khi bảng khai `trung_byte=true` giữ nguyên → `hau-kiem` 2 và `lap-tracker`/`cham-lai` 2 đúng run; bảng không đổi |
| F6 [P2] traversal không thoát vùng | `phien/../../../../ngoai_vung_traversal_b` (chuẩn hóa ra `tests/`, ngoài `cong_b_20261026/`) | kiểm trước đường chuẩn hóa thật sự ngoài vùng; exit 2 "phai nam duoi"; đường không được tạo |
| F7 [P1] tổng hợp có thể báo đạt khi thiếu kiểm / pyc sau hỏng | mỗi ca khai `yeu_cau(...)` trước mọi nhánh; kiểm bắt buộc không chạy = KHÔNG ĐẠT; mỗi lần gọi khi ma trận đủ **tính lại** kết luận (`tong_ket`): đủ ca, đủ kiểm bắt buộc, mọi kiểm đạt, so sánh pyc của MỌI lần gọi trong vòng đạt, không crash ngoài dự kiến trong vòng; `ket_qua/lan_NN.json` giữ lịch sử, `ket_qua_test.json` = kết luận lần gọi cuối | (cơ chế; `tong_ket` áp lên mọi lần gọi) |
| F8 [P1] sổ crash ngoài dự kiến chưa đủ (vòng 3 bổ sung F8a/b/c, mục 0b) | `ra_soat` rà MỌI vòng ở đầu (trước chọn vòng) và cuối mỗi lần gọi: lần gọi fixture thiếu `ket_thuc` (phân loại `tran_cung_watchdog`/`fixture_gian_doan`), lệnh con bất thường không do tiêm, lần chạy runner crash không do tiêm; lỗi tiêm ghi riêng `tiem`; chỉ thêm, không xóa; an toàn chưa xác định hoặc ≥ 4 crash native ngoài dự kiến → `dung_fixture.json` → mọi lần gọi sau exit 40 | kết thúc: `khong_crash_ngoai_du_kien_trong_vong` |
| F9 [P2] JSON hỏng → lỗi Python | `kiem_dau_vao.doc_json_dv`/`doc_json_hd` + schema hợp đồng (`SCHEMA_HOP_DONG`, phụ lục, loại trừ, regex), `goi_a.json` (`SCHEMA_GOI`, file = 64 hex), review, phê duyệt, khóa, kiem_khoa, kiem_goi_a, `tach_plan`; runner: phien.json/manifest/goi.json/`--tiem-loi` → LoiDauVao/LoiHopDong (exit 2), giữ nguyên nhân thô | `am_dau_vao`: goi_a.json cắt cụt / sai kiểu, kiem_goi_a cắt cụt, review cắt cụt (digest ghim đúng), phê duyệt cắt cụt (không ghi), hợp đồng cắt cụt / sai kiểu trường |

## 1. File của B0 (đúng 6 file trong write_paths)

| file | vai trò |
|---|---|
| `configs/phase2/r61_bdd/hop_dong_cong_b.json` | hợp đồng B: ghim A/khóa/PLAN/review/phụ lục, quy tắc `van_hanh` A, giao diện phê duyệt B, deadline, crash, bytecode, chia khối bootstrap |
| `src/r61_bdd/cong_b/kiem_dau_vao.py` | kiểm đầu vào B (thư viện chuẩn ở mức module) |
| `src/r61_bdd/cong_b/van_hanh_cong_b.py` | runner cha/con cổng B |
| `src/r61_bdd/cong_b/thong_ke_tiep_tuc.py` | bootstrap/tái tính chia khối chỉ số |
| `src/r61_bdd/cong_b/test_van_hanh_cong_b.py` | fixture CPU (entrypoint đúng lệnh B1) |
| `handoff/logs/review61/04_chuan_bi_cong_b.md` | báo cáo này |

Không thêm `__init__.py`: `cong_b/` import bằng `sys.path` (thư mục `r61_bdd` và `cong_b`).

## 2. Kiểm đầu vào (`kiem_dau_vao.py`)

Giá trị ghim (hợp đồng thật): `goi_a_sha256 5c9db320…0fe225`, `khoa_sha256 f3a138d7…794d4f`, `SHA256SUMS fdac9e6f…fa25`,
`kiem_khoa/vong_01/ket_qua.json e35a7423…06f1`, `stt_dong_goi_dau 16`, 3.286 file; PLAN `2b2474aa…e93e12a`, 194.250 byte,
mục 0 = byte 322–176.790, trước `a50c3bc4…12efd9`, sau `aba1aebb…ac1723` (đủ 64 hex lấy từ `khoa.json`
`phu_luc_van_hanh.plan`); review_009 `107fa671…0ea25`, verdict `approve`; hai phụ lục 246 B `adad5985…` và 1123 B `f346a6fe…`.

Nhóm kiểm theo thứ tự (lỗi báo theo nhóm đầu tiên không đạt, thông điệp `[nhom] [...]`, kết quả thô mọi nhóm trả về):
1. `duong` — không reparse ở đường tới phiên A và mọi mục bên trong (os.walk, vì `liet_ke_file` bỏ qua reparse im lặng).
2. `plan` — `chuan_bi_bdd.tach_plan` trên byte PLAN: so đồng thời `sha256_toan_bo`, trước/sau mục 0, `byte_muc0`, `so_byte`;
   thêm `sha256_muc0` vào bản ghi thô. Phép so `nguyen_byte` là phép duy nhất được chấp nhận (khác → lỗi). Không chuẩn hóa.
3. `review` — đọc nguyên bản, sha256 = digest ghim, `verdict` = approve. Không dùng/không đòi `goi_a_sha256` trong review.
4. `phe_duyet` — xem mục 4.
5. `goi` (`goi_a.json` sha256 + `khoa_sha256`/`phien_A`/`stt_dong_goi_dau`/`so_file`/`luot`), 6. `sha256sums` (byte = danh mục
   và sha256 ghim), 7. `marker_goi` (`goi_a.xong.json` qua `hop_dong_bdd.kiem_marker`, danh tính có `khoa_sha256`).
8. `phu_luc` — `van_hanh/dung_da_go/` đúng hai file, so số byte và sha256; thiếu/đổi/thừa → chặn.
9. `van_hanh` — mọi file dưới `<A>/van_hanh/` phải thuộc danh mục gói, phụ lục, hoặc **đúng một** mục loại trừ (regex
   trên đường tương đối, `so_file` đúng, điều kiện, lý do). `dung.json` luôn chặn. Loại trừ thật: `so_crash.json` (1),
   `goi_cha.jsonl` (1), `cha/<ts>_pid<n>.faulthandler.log` (18, mỗi file 0 byte), `lan_chay/0016_dong-goi-a|0017_kiem-goi-a/<8 tên>`
   (16), `kiem_goi_a/vong_01/{ket_qua,xong,bam/danh_muc,bam/doan/NNNN}.json` (55, điều kiện `ket_qua.json` dat=true và ràng
   buộc đúng gói/khóa). Số đếm lấy từ liệt kê `git status --untracked-files=all` của phiên A trong lượt này và báo cáo 03 mục
   12.5; nếu sai, tiền kiểm B thật trả exit 2 (an toàn). Mọi file (kể cả loại trừ) được băm vào manifest B.
10. `payload` — băm lại MỌI file của `goi_a.json`; file thừa trên `van_hanh_cong_a.payload(A, stt_dau)` ∪ `goi/` ∪
    `dong_goi/phuc_hoi/`.
11. `khoa` (nặng, chỉ khi 1–10 đạt) — `khoa.json` sha256 + `khoa.xong.json` + `kiem_khoa/vong_01` + ẢNH KHÓA HIỆN TẠI:
    `dieu_phoi_bdd.so_khoa(khoa, anh_khoa(ctx, A, cache_từ_gói))` + `van_hanh_cong_a.so_phu_luc(phu_luc_van_hanh, phu_luc(ctx, A))`.
    Gọi nguyên hàm A, không đổi phép so khóa A.

`ma_b(hd)`: sha256 3 file runner `cong_b` + đóng import tĩnh `DIEM_VAO_MA` (chỉ đọc) + hợp đồng B, hợp đồng lệnh, giao thức.
Tương thích khóa A: `cong_b/` không nằm trong `dong_import_tinh(DIEM_VAO_MA)` và `phu_luc().ma_r61` chỉ `os.listdir(r61_bdd)`
`*.py` (không đệ quy) → thêm thư mục `cong_b` không đổi `ma_nguon` hay `ma_r61` của khóa A. Fixture kiểm điều này trên khóa
A thật (chỉ đọc).

## 3. Runner (`van_hanh_cong_b.py`)

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r61_bdd/cong_b/van_hanh_cong_b.py --cong B --phien <phien_B>
    --buoc <tien-kiem|mo-phien|tracking|cham|lap-tracker|cham-lai|tong-hop|kiem-khoa|hau-kiem> --so-doan 1
    --ngan-sach-giay <=420 --phe-duyet <file> [--han-chot <epoch>] [--gioi-han-don-vi N] [--hop-dong ..] [--giao-thuc ..]
... van_hanh_cong_b.py --in-ma-b [--hop-dong ..]   # chỉ đọc: in ma_b_sha256 + hop_dong_cong_b_sha256 cho file phê duyệt
```

- **Cha** (thư viện chuẩn + mã r61 mức module thư viện chuẩn): từ chối nếu interpreter ≠ đường tuyệt đối đã chỉ định hoặc
  thiếu `-B` trong `sys.orig_argv` (không tin riêng biến môi trường). Trước lần ghi đầu: `--cong B`, `--so-doan 1`,
  `0 < ngân sách ≤ 420`, hợp đồng (`nap_hop_dong`) và giao thức (`nap_ctx`: JSON + schema + manifest, lỗi → exit 2
  `[giao_thuc]`) cùng chế độ, phiên lexical + realpath + không reparse (`H.kiem_duong_ghi`),
  vùng (thật: `thuc_thi/`, fixture: `tests/cong_b_20261026/`), cấm tiêm lỗi ở chế độ thật, phê duyệt hợp lệ, phiên.json
  (lượt/phiên/giao thức/hợp đồng; manifest mã B + phê duyệt so với lúc tạo phiên). `van_hanh/phien.json` là file đầu tiên
  của phiên, chỉ tạo ở `tien-kiem`.
- Cha phục hồi lần chạy thiếu `ket_thuc.json` (cha gián đoạn): đối chiếu PID + thời điểm tạo (`trang_thai_con`), ghi
  `cha_gian_doan`, `dung.json`, exit 40. `dung.json` chặn mọi lời gọi sau (40).
- Tạo con: `[python, -B, van_hanh_cong_b.py, --con, <lan>]`, `PYTHONDONTWRITEBYTECODE=1`, CREATE_SUSPENDED, cha đã tự gán
  Job Object KILL_ON_JOB_CLOSE (`van_hanh_cong_a.gan_job_cha`), xác nhận con trong job, ghi `con.json` (pid, thời điểm tạo,
  argv, `timeout_giay`) rồi mới `NtResumeProcess`. Log từng lần: `goi.json` (argv cha/con, cờ bytecode cha, hạn, lần thử
  trước, đếm crash trước, tiêm lỗi), `stdout.txt`, `stderr.txt`, `faulthandler.log` (con), `tien_do.jsonl` (fsync),
  `ket_qua_con.json` (cờ bytecode con), `ket_thuc.json`; faulthandler cha trong `van_hanh/cha/`.
- **Deadline**: `han_cung = min(bat_dau_cha + ngan_sach + 150, --han-chot − 15)`; còn < 30 s → không tạo lần chạy, exit 10.
  Con nhận `han_mem = min(bat_dau_cha + ngan_sach, han_cung − du_phong_don_vi)` qua `goi.json` (không cấp lại ngân sách);
  hết `han_mem` hoặc đạt `--gioi-han-don-vi` → dừng giữa đơn vị, exit 10. Cha chờ con tới `han_cung` → quá → kill, `het_gio` → 40.
- **Con**: kiểm `-B`, kiểm lần chạy trước mọi ghi (`goi.json` qua `doc_json_hd` + `SCHEMA_GOI_CON` trước mọi ép kiểu,
  giao thức qua `nap_ctx`), faulthandler, `chuan_bi_moi_truong` (TEMP/cấu hình thư viện trong phiên)
  trước import bên thứ ba; mọi bước sau `mo-phien` gọi `nap_b`: `kiem_dau_vao.kiem_tat_ca` (đủ 11 nhóm, không dùng kết quả cũ)
  + so manifest B + phần còn lại của `dieu_phoi_bdd.nap_phien_b` (sha256 goi/khoa/review/phê duyệt/giao thức) — không tính
  ảnh khóa hai lần.
- Bước/đơn vị: `tien-kiem` (ghi kết quả thô kể cả khi không đạt) → `mo-phien` (kiểm lại, so manifest tiền kiểm, ghi
  `phien.json` tương thích `dieu_phoi_bdd` + `manifest_b.json` + marker) → `tracking:<run>:<video>` (`D.chay_don_vi`, kiểm
  cache A từ gói trước mỗi đơn vị) → `cham:<run>` (chỉ khi đủ mọi đơn vị tracking; `D.cham_run`) → `lap:<run>:<video>` +
  `lap:so_sanh` → `cham-lai:<run>` + `cham-lai:so_sanh` → `tong-hop:chi_so` / `tong-hop:khoi:<k>` / `tong-hop:ghi` →
  `kiem-khoa` (`D.buoc_kiem_khoa`) → `hau-kiem:khoi:<k>` / `hau-kiem:ghi`.
- Artifact dở (chưa marker hợp lệ, kể cả `.dang_ghi`, kể cả thư mục `_trackeval/<run>`) chuyển nguyên byte vào
  `van_hanh/artifact_do/<đơn vị>/lan_<stt>/` (`van_hanh_cong_a.sao_luu_do`, sha256 trước/sau) trước khi ghi lại; kết quả
  kiểm/so sánh ghi qua `ghi_bao_toan` (sao lưu bản cũ, không ghi đè). Marker hỏng → exit 2, không thử lại.
- **Crash**: ngoài {0,1,2,10} = `crash_native` (tính); `cha_gian_doan` tính; dừng ở crash thứ 3/đơn vị, thứ 4/phiên, thứ 2
  của `kiem-khoa`; exit 1 (traceback hoặc không), hết giờ, thiếu log, Job/resume không đạt, PID không xác định, crash
  không xác định đơn vị → 40. Bộ đếm tính lại từ mọi `ket_thuc.json` (không reset). Tiêm lỗi chỉ fixture: con
  `khoi_dong|truoc_ghi|sau_ghi|dang_ghi|sau_marker|bo_faulthandler` × `native|python|cho`; cha
  `cha_sau_tao_con|cha_khi_con_chay|cha_sau_con|job_loi|resume_loi|pid_khong_xac_dinh`; tiêm ghi trong `goi.json`/`ket_thuc.json`.

## 4. Phê duyệt B thật (giao diện, chưa có file nào)

File JSON do **orchestrator** cung cấp sau khi người dùng duyệt kế hoạch thực thi B (plan chuẩn bị này không mở B):
`loai=phe_duyet_cong_b`, `duyet=true`, `goi_a_sha256`, `khoa_sha256`, `review_a_sha256`, `hop_dong_cong_b_sha256`,
`ma_b_sha256` (từ `--in-ma-b`), `phien_b` (= `--phien`, dưới `thuc_thi/`), `plan_thuc_thi_sha256` (64 hex), `nguoi_duyet`, `luc`.
Không nằm dưới `tests/`. Thiếu/sai → exit 2 trước mọi ghi. Agent/runner không tự tạo file này.

## 5. Bootstrap chia khối (`thong_ke_tiep_tuc.py`)

Chỉ số `default_rng(0)` (so_mau × so_video, int64) lưu một lần (`D.doc_chi_so` kiểm dtype/shape/miền/trùng rng khi tái
dùng). Khối k = hàng `[k·kt, …)` (kt 250 thật, 64 fixture); mẫu khối = `(metric a − metric b)·HE_SO` dùng đúng
`thong_ke_bdd.xep/metric_bo_loc` (PF trung bình metric COMBINED 3 seed), lưu float64 `(G, 6, 3, n)` + marker (danh tính:
khối, hàng, sha256 chỉ số, sha256 đầu vào chấm). CI = `np.percentile([2.5, 97.5], method="linear")` trên **toàn mẫu ghép**,
không gộp CI từng khối; dòng/khóa/thứ tự như `thong_ke_bdd.bang_ci`; CSV qua `D.csv_bytes`; `kiem_chung.json` cùng khóa
với `dieu_phoi_bdd` (+ `chia_khoi`). Hậu kiểm: mỗi lần hậu kiểm hoàn tất đóng một vòng `hau_kiem_b/vong_NN/`; lần sau mở
vòng mới và tái tính MỌI khối (không dùng kết quả kiểm cũ); trong vòng chưa đóng thì tiếp tục theo khối. So byte khối với
`tong_hop/khoi`, CSV với dữ liệu tính lại, `kiem_chung` với giá trị tính lại; marker tổng hợp hỏng vẫn tái tính để chỉ rõ
đầu ra bị sửa. `doi_chieu_nguyen_khoi` (fixture) so với `bang_ci` + `bootstrap_nhom` nguyên khối.

## 6. Fixture B1 (`test_van_hanh_cong_b.py`)

Lệnh (đúng plan, khớp quyền Bash): `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -B src/r61_bdd/cong_b/test_van_hanh_cong_b.py --phien results/phase2/r61_bdd/tests/cong_b_20261026/lan_01 --ngan-sach-giay 420 --gioi-han-giay 540`.

- Đầu module, trước chụp pyc và mọi import mã dự án/bên thứ ba: `sys.dont_write_bytecode=True`, `PYTHONDONTWRITEBYTECODE=1`,
  **trần cứng** `faulthandler.dump_traceback_later(exit=True)` tại `bắt đầu + --gioi-han-giay − 8 s` (532 s), rồi chụp pyc.
  Từ chối chạy nếu thiếu `-B`/sai interpreter/`--phien` ngoài `tests/cong_b_20261026/`/ngân sách ∉ (0, 420]/giới hạn > 540.
- Trước khi tạo con đầu tiên: fixture tự gán **job gốc** KILL_ON_JOB_CLOSE (`VA.gan_job_cha`) và đặt lại trần cứng vào
  `vong_NN/faulthandler_lan_KK_pid*.log`; không có job gốc → không khởi chạy lệnh con nào. Mọi lệnh con
  `[python, -B, van_hanh_cong_b.py, …, --han-chot <hạn chung>, --ngan-sach-giay <còn lại ≤ 420>]` (tham dò:
  `[python, -B, test_van_hanh_cong_b.py, --tham-do, …]`) tạo CREATE_SUSPENDED → đã thuộc job gốc từ CreateProcess → kiểm
  `IsProcessInJob` + gán job lệnh riêng (lồng) → resume; quá hạn → `TerminateJobObject`; sau mỗi lệnh chờ `ActiveProcesses = 0`
  (ghi `cay`, `trong_job_goc`, PID + thời điểm tạo). Lỗi Python ở bất kỳ đâu trong `khoi_chay_trong_job` → giết con +
  `TerminateJobObject` rồi ném lại; fixture chết (kể cả trần cứng/crash native) → kernel đóng job gốc → cả cây bị giết.
- Hạn: `T.TRAN` = bắt đầu + giới hạn − 8; `T.HAN` = bắt đầu + giới hạn − dự phòng kết thúc (`max(45, 2 × giây chụp pyc + 25)`);
  `T.MEM` = bắt đầu + ngân sách mềm. `can_tg` trước mọi việc: mỗi ca (15 s), mỗi lệnh con (75 s; timeout 80 s), mỗi bước
  thiết lập, tái tính bootstrap (150 s), ca kết thúc (60 s); không đủ → exit 10 trước khi bắt đầu. Cache A tổng hợp nhận
  ngân sách = `min(T.MEM, T.HAN − 30) − now`, dừng sạch theo video (`tam_dung`) rồi tiếp tục lần gọi sau.
- Checkpoint: `vong_NN/lenh/<tên>.json` (có → dùng lại, không chạy lại; `.bat_dau.json` mà thiếu kết quả → "bi_gian_doan",
  không chạy lại, kết thúc không đạt); `vong_NN/ca/<NN>_<ca>.json` (kèm danh mục kiểm bắt buộc `can`); quan sát/tái tính
  ghi bền vững ngay lần đầu (`nho/`); bước thiết lập `thiet_lap/<bước>.{trang_thai,xong}.json` ('dang_chay' còn lại = bị ngắt
  giữa bước → không làm lại). Sửa tạm thế giới (`sua_tam`, chỉ `fx/` và `phien/`) có bản gốc bền vững (`khoi_phuc/`), phục
  hồi ở đầu lời gọi sau; journal là đầu vào không tin (F3); file thêm tạm chuyển ra `bo/` (không xóa). Mã đổi → vòng mới.
- Journal khởi động (F8a) `lan_01/khoi_dong/<id>.{bat_dau,ket_thuc}.json` + `<id>.faulthandler.log`: ghi bằng thư viện
  chuẩn ngay sau kiểm đối số/đường, trước chụp pyc/import; đóng ở mọi đường `return` của `chay()`.
- Sổ crash ngoài dự kiến `lan_01/so_crash_ngoai_du_kien.json` (F8/F8b/F8c): rà MỌI vòng + journal khởi động ở đầu (trước
  `chon_vong`) và cuối mỗi lần gọi, và rà bằng chứng của từng lệnh NGAY sau lệnh đó (`ap_so_sau_lenh`, trước lệnh kế
  tiếp); chỉ thêm; lỗi tiêm chỉ vào `tiem` khi điểm/kiểu tiêm giải thích đúng phân loại; đếm crash native theo `su_co`.
  `dung_fixture.json` (mọi lỗi ngoài dự kiến không phải crash native đã thu hồi, hoặc ≥ 4 sự cố crash native ngoài dự kiến)
  → mọi lần gọi sau exit 40, kiểm trước chọn vòng nên đổi snapshot không gỡ được.
  Hệ quả có chủ ý: chạm trần cứng / bị ngắt một lần ở bất kỳ giai đoạn nào cũng làm B1 dừng hỏi; một timeout/traceback/
  thiếu log ngoài dự kiến của runner cũng dừng ngay (không chạy tiếp các ca sau); bản sửa tạm dở (nếu có) khi đó nằm trong
  `khoi_phuc/` của vòng, không được tự phục hồi trước khi người dùng quyết định.
- Kết luận (F7): chỉ khi mọi ca đã có kết quả; tính lại mỗi lần gọi từ `ca/*.json` + `pyc/lan_*_so_sanh.json` + sổ crash:
  đủ ca, không ca lạ, đủ kiểm bắt buộc, mọi kiểm đạt, pyc MỌI lần gọi của vòng đạt, không crash ngoài dự kiến trong vòng,
  cờ bytecode đạt. `ket_qua/lan_KK.json` giữ lịch sử, `ket_qua_test.json` = kết luận lần gọi cuối (+ `lich_su`). Exit
  fixture: 0 đạt; 10 còn việc; 2 không đạt/đối số sai; 40 dừng hỏi.
- A tổng hợp dựng bằng helper hiện có (`tao_the_gioi`, `giao_thuc_fixture`, `ModelGia`, `D.buoc_tien_kiem`, `D.buoc_cache`,
  `D.anh_khoa`, `VA.phu_luc`, `VA.chup_a`, `VA.payload`), cùng cấu trúc gói A (goi_a.json/SHA256SUMS/marker, khoa + marker,
  kiem_khoa/vong_01, van_hanh với phien/lan_chay/cha/goi_cha/so_crash/dung_da_go/kiem_goi_a, goi/nhat_ky_crash_chot);
  smoke tối giản (B không đọc smoke). Không chạy runner/fixture A cũ.

Ma trận theo thứ tự `CA` trong mã (đếm tĩnh ≈ 135 lệnh con sau vòng 4 = 129 của vòng 3 + 1 ca `fps=0` + 5 lần gọi fixture
con của ca 13; nhóm "chạy tới xong" có thể thêm vài lệnh exit 10 tùy thời
gian; số thực tế chốt ở B1 từ `ket_qua_test.json.so_lenh_con`). Tách theo nhóm dự toán của plan ở cột cuối.

| # | ca | lệnh con | nội dung (ca hồi quy F1–F9 in đậm) | nhóm plan |
|---|---|---|---|---|
| 00 | thiet_lap | 1 (+ dựng trong tiến trình) | thế giới, A tổng hợp, hợp đồng/review fixture + **bản cắt cụt/sai kiểu (F9)**; `--in-ma-b`; tương thích khóa A thật (chỉ đọc); cờ `-B` | thiết lập |
| 01 | bao_ve_lan_goi | 4 tham dò | **job gốc đạt; trần cứng ≤ bắt đầu + 532 s; ngắt native giữa Popen và Assign → cháu chết theo job gốc, cây về 0 không terminate (F2); RuntimeError giữa hai bước → dọn dẹp; việc trong tiến trình vượt trần 8 s → exit 1, "Timeout (", cháu chết (F1); thiếu dự phòng → exit 10, không bắt đầu việc (F1)** | Job/timeout, cha gián đoạn/thu hồi |
| 02 | thanh_cong | ≈ 23–26 | tien-kiem, mo-phien, tracking (gh 5 → 10, tiếp → 0, không ghi/bắt đầu lại đơn vị xong), cham (gh 2 → 10, tiếp), lap-tracker, cham-lai, tong-hop (gh 2 → 10), đối chiếu chia khối = nguyên khối (dòng, mẫu, thứ tự, byte CSV 108, `D.kiem_lai_tong_hop` = []), kiem-khoa, hau-kiem (gh 2 → 10); thiếu dự phòng runner → 10 không tạo lần chạy; hậu kiểm: sửa ci_108.csv / khối 0000 / phụ lục / file lạ van_hanh → 2 đúng nguyên nhân; **sửa `lap/.../data/<video>.txt` và `cham_lai/<run>/raw_fields.json` khi bảng khai `trung_byte=true` giữ nguyên → hau-kiem 2 + lap-tracker/cham-lai 2 đúng run, bảng nguyên byte (F5)**; khôi phục → 0; `-B` mọi cha/con | thành công, exit 10/resume, hậu kiểm |
| 03 | kiem_lai_buoc_xong | 8 | **tien-kiem và mo-phien đã xong: PLAN mục 0 đổi → 2 [plan]; `dung.json` của A → 2 [van_hanh]; phụ lục đổi → 2 [phu_luc]; khôi phục → 0 kèm sự kiện kiểm lại + `bo_qua`; artifact hoàn tất nguyên byte (F4)** | âm đầu vào, hậu kiểm phụ lục |
| 04 | am_dau_vao | 41 | **giao thức cắt cụt / manifest trỏ tới sai kiểu / `tracker.fps` sai kiểu / `tracker.fps=0` (F9 vòng 4) → 2 `[giao_thuc]`, không tạo phiên; `--con` `goi.json` `stt=null` → 2 `[lan_chay]`, không ghi (F9 vòng 3)**; thiếu/đổi/thừa phụ lục, file lạ van_hanh, dung.json, PLAN mục 0/trước/sau, payload đổi/thừa, provenance cache, goi_a.json, SHA256SUMS, marker gói, khóa, YAML (ảnh khóa), review digest/verdict, **goi_a.json cắt cụt / sai kiểu gốc, kiem_goi_a cắt cụt, review cắt cụt (digest đúng), hợp đồng cắt cụt / sai kiểu trường, phê duyệt cắt cụt (F9)**, phê duyệt (false/gói/phiên/mã B, không ghi), B thật thiếu phê duyệt / có tiêm lỗi (không tạo thư mục), `--so-doan 2`, ngân sách 421, `--cong A`, **traversal thật sự thoát vùng (đường chuẩn hóa ở `tests/`, ngoài `cong_b_20261026/`) → 2 "phai nam duoi", không tạo đường (F6)**, junction | âm đầu vào/quyền/đường |
| 05 | phuc_hoi_journal | 0 (trong tiến trình) | **journal traversal / junction ở cha / tên không ràng buộc đường → chặn, không ghi/di chuyển; `sua_tam` ngoài vùng sửa → chặn trước lần ghi đầu; phục hồi hợp lệ đạt (F3)** | crash/phục hồi, đường ghi |
| 06 | ma_b_doi | 2 | hợp đồng B đổi sau tạo phiên → 2 "cau_hinh hop_dong_cong_b" | âm đầu vào |
| 07 | crash_ghi | 13 | truoc_ghi → 30 → thử lại; sau_ghi → 30 → sao lưu artifact dở rồi làm lại; dang_ghi marker → 30 → sao lưu `.dang_ghi`; sau_marker → 30 → không chạy lại (bo_qua); marker đơn vị hỏng → 2; bộ đếm giữ qua lời gọi | crash quanh ghi/marker |
| 08 | cha_gian_doan | 7 | cha crash khi con chạy (cây thu hồi, con chết theo PID + thời điểm tạo) → lần sau 40 → giữ chặn; cha crash sau tạo con → 40 "thieu con.json" | cha gián đoạn/thu hồi |
| 09 | job_pid_log | 7 | job_loi, resume_loi, loi_python, thiếu faulthandler, PID không xác định, timeout (han-chot ngắn; `timeout_giay` ≤ hạn − 15, không cấp lại ngân sách) → 40 | Job/resume/PID/traceback/log/timeout |
| 10 | nguong | 18 | đơn vị 30/30/40 + giữ chặn; phiên 30/30/30/40 + giữ chặn; kiem-khoa 30/40 + giữ chặn (gồm 6 lệnh mở phiên) | ngưỡng crash |
| 11 | khoi_dong_ngat | 3 (1 tham dò + 2 fixture con) | **F8a: journal lần gọi này trước chụp pyc/import; tham dò ngắt native ngay sau journal (trước chụp pyc/import) → cây thu hồi, journal thiếu `ket_thuc`, không vòng; `ra_soat` dừng; lần gọi sau → 40; đổi snapshot → vẫn 40, không tạo vòng; sổ giữ 1 sự cố** | cha gián đoạn/thu hồi, ngưỡng crash |
| 12 | so_crash | 3 fixture con (+ trong tiến trình) | **F8b: timeout/traceback/thiếu log ngoài dự kiến → dừng ngay sau lệnh (`DungFixture`); lần gọi + snapshot sau vẫn 40; tiêm khớp loại → không dừng; tiêm lệch loại → dừng; bằng chứng thật `loi_python` có tiêm → `tiem`, bỏ tiêm → dừng. F8c: 1/2/4 sự cố → đếm 1/2/4 (bằng chứng 2/4/8), chỉ 4 dừng, ra soát lặp không tăng; qua snapshot cộng dồn tới 4 → dừng** | ngưỡng crash, Job/traceback/log/timeout |
| 13 | so_crash_bo_sung | 5 fixture con (+ trong tiến trình) | **F8b-1 (vòng 4): cha runner native không tiêm; lần chạy thiếu `ket_thuc`; exit 40 không `dung.json`; exit 40 do crash không đơn vị / ngưỡng không tiêm → dừng ngay, không đếm native, giữ chặn qua snapshot; đối chứng có tiêm → không dừng; lần gọi fixture con thật sau cha-native → 40. F8b-2: sổ sai cấu trúc sau lệnh (4 biến thể) → `DungFixture`, sổ nguyên byte, lần gọi thật sau → 40; ở khởi động (`muc:[null]` 2 lần, `muc:{}` 1 lần) → 40, không tạo vòng, sổ nguyên byte; `DungFixture` không phải `Exception`** | ngưỡng crash, cha gián đoạn/thu hồi |
| 14 | ket_thuc | 0 | PLAN/A thật/mã cũ nguyên byte, không tạo `thuc_thi/`, không `.dang_ghi` lạc, mọi lệnh có exit + cây thu hồi, **mọi lệnh con thuộc job gốc từ lúc tạo (F2)**, mọi lệnh/cháu tham dò `[python, -B, …]`, hạn con ≤ hạn chung − 15, **không crash ngoài dự kiến trong vòng (F8)** | kết thúc |

Mỗi ca khai `yeu_cau(...)` trước mọi nhánh (F7): kiểm bắt buộc không chạy (điều kiện trước hỏng, lỗi giữa ca) ghi KHÔNG
ĐẠT; ngoại lệ trong ca ghi `<ca>_khong_loi_bat_ngo` không đạt kèm traceback; mỗi ca thêm kiểm `<ca>_ma_khong_doi_trong_ca`.
Kiểm âm đạt chỉ khi exit **và** nhóm/chuỗi nguyên nhân khớp; ca chưa chạy không tính đạt.

## 7. Phạm vi kiểm bytecode

Chụp `{đường: [size, mtime_ns, sha256]}` mọi `.pyc` trong: `src/` (đệ quy), site-packages `ultralytics, trackeval, numpy,
pandas, scipy, cv2, lap, torch, yaml, matplotlib, PIL`, stdlib (`Lib/` trừ `site-packages`). Chụp trước import mã dự án/bên
thứ ba và sau khi mọi lệnh đã thu hồi cây con, mỗi lời gọi (`vong_NN/pyc/lan_KK_{truoc,sau,so_sanh}.json`); so trong lời
gọi và so với lần chụp đầu vòng: thêm/mất/đổi → không đạt. Không xóa pyc nào (kể cả
`src/__pycache__/baseline_track.cpython-311.pyc` đã có). Không quét dữ liệu lớn. Giới hạn: tiến trình khác của người dùng
(IDE, Python không `-B`) ghi pyc trong phạm vi sẽ làm kiểm không đạt (báo thật, không lọc).

## 8. Dự toán (kịch bản theo giả định, không phải số đo)

≈ 135 lệnh con (+ vài lệnh exit 10 trong nhóm "chạy tới xong"); ~75 lệnh nặng (kiểm đủ đầu vào + ảnh khóa, giả định
~8–12 s), ~45 lệnh nhẹ (dừng ở kiểm đối số/phê duyệt/hợp đồng/giao thức, ~1–4 s), 5 tham dò (~2–15 s, trần cứng 8 s),
10 lần gọi fixture con của ca 11–13 (mỗi lần = khởi động + chụp pyc + import, dừng ở rà sổ; giả định ~10–40 s → thêm
~50–200 s so với vòng 3), TrackEval
chấm nhỏ, ca timeout ~75 s, cha gián đoạn ~5–10 s mỗi ca, dựng A ~40 s, chụp pyc 2 lần/lời gọi (chưa đo; quét `.pyc` của
`src/`, 11 gói site-packages và stdlib); rà sổ sau mỗi lệnh chỉ đọc bằng chứng của lệnh đó (nhẹ) → kịch bản ≈
1.350–1.950 s ≈ 4–5 lát 420 s. Ngoại suy plan theo số lệnh 135/65 × 362,8 × 4 ≈ 3.010 s (≈ 8 lát) là kịch bản xấu: vượt
5 lần gọi fixture/lời gọi Claude (≤ 2.700 s) thì lời gọi Claude thứ 2 của B1 tiếp tục đúng lệnh từ checkpoint (trong dự
toán 2 lời gọi của B1); không bỏ ca để giữ số lệnh. Đây là kịch bản theo giả định, không phải số đo hay bảo đảm.

## 9. Tự kiểm B0 và giới hạn

- Đã chạy (vòng sửa 4, 27/10): `git rev-parse HEAD` → exit 0, `117d62da027e8ce3e4c9f455d576a924b4d82bbb`;
  `git status --short src/r61_bdd configs/phase2/r61_bdd handoff/PLAN_DA_CHOT.md handoff/logs/review61
  results/phase2/r61_bdd/tests` → exit 0, ` M handoff/PLAN_DA_CHOT.md` (có từ đầu lượt, không sửa) và `??` cho 4 thư mục
  (untracked → git không chứng minh nội dung bất biến); Glob: `src/r61_bdd/cong_b/` chỉ 4 file `.py` (không `__pycache__`,
  không `__init__.py`), `results/phase2/r61_bdd/tests/cong_b_20261026*` không tồn tại. Vòng 4 chỉ sửa `kiem_dau_vao.py`
  (`nap_ctx`: kiểm `tracker.fps`, bắt `ArithmeticError`), `test_van_hanh_cong_b.py` (`bat_thuong_lenh`,
  `giai_thich_dung_40`, `bat_thuong_lan_chay`, `_xet_lenh`, `_xet_phien`, `loi_so`, `dung_so_hong`, `LOI_SO`,
  `ap_so_sau_lenh`, nhánh `LOI_SO` ở khởi động/kết thúc `chay()`, ca 04 `fps=0`, ca 13 `so_crash_bo_sung`) và báo cáo này.
  **Chưa chạy Python** (B0 cấm chạy fixture; quyền không có Python tổng quát/`py_compile`) → chưa có bằng chứng cú
  pháp/thực thi cho mã vòng 4.
- Rà tĩnh vòng 4: (a) tên trường mà fixture đọc khớp runner — `ket_thuc.quyet_dinh.exit_cha`, `don_vi`, `phan_loai`,
  `con_trang_thai`, `job.dat`, `thieu_log` (`van_hanh_cong_b.quyet_dinh`/`mot_doan`/`phuc_hoi_cha_gian_doan`); `dung.json`
  có `stt` của lần chạy gây dừng (`dung()`), và `phuc_hoi_cha_gian_doan` ghi bổ sung `ket_thuc` vào CHÍNH lần chạy bị gián
  đoạn (có `goi.tiem_loi` của nó) nên ca thật 08 vẫn được giải thích bằng tiêm `cha_*`; (b) duyệt tay 7 ca F8b-1 qua
  `bat_thuong_lenh` → `_xet_phien` → `ly_do_dung`/`su_co_native`: 5 ca âm cho `muc` `dung_ngay=true`, `su_co_native=[]`;
  2 ca đối chứng chỉ ghi `tiem`; (c) 4 biến thể sổ F8b-2 đều trả lỗi trong `loi_so` (phần tử null / `muc` là object / phần
  tử chuỗi / bool sai) trước mọi `x.get`; thông điệp `LoiHopDong` chứa "sai cau truc" mà ca khởi động chờ; (d) ca `fps=0`
  không chạm nhánh "sai kieu" (schema đạt) nên chờ đúng chuỗi "phai la so huu han > 0". Đây là rà tĩnh, không phải bằng
  chứng chạy; nếu sai, B1 lộ (ca không đạt hoặc dừng hỏi), không báo đạt giả.
- Đã chạy (vòng sửa 3): `git status --short src/r61_bdd configs/phase2/r61_bdd handoff/PLAN_DA_CHOT.md handoff/logs/review61
  results/phase2/r61_bdd/tests` → exit 0, ` M handoff/PLAN_DA_CHOT.md` (đã có từ đầu lượt, vòng này không sửa) và `??` cho
  `configs/phase2/r61_bdd/`, `handoff/logs/review61/`, `results/phase2/r61_bdd/tests/`, `src/r61_bdd/` (untracked → git không
  chứng minh nội dung bất biến); Glob: `src/r61_bdd/cong_b/` chỉ 4 file `.py`, `results/phase2/r61_bdd/tests/cong_b_20261026*`
  không tồn tại. Vòng 3 chỉ sửa `kiem_dau_vao.py` (thêm `SCHEMA_GIAO_THUC`, `nap_ctx`), `van_hanh_cong_b.py` (`SCHEMA_GOI_CON`,
  `kiem_lan_con`, `main_cha` dùng `nap_ctx`), `test_van_hanh_cong_b.py` và báo cáo này. **Chưa chạy Python** (B0 cấm chạy
  fixture; quyền không có Python tổng quát/`py_compile`) → chưa có bằng chứng cú pháp/thực thi cho mã vòng 3.
- Rà tĩnh vòng 3: chuỗi nguyên nhân mà ca mới chờ khớp thông điệp thật — `[giao_thuc]` + "khong phai JSON hop le" /
  "videos phai la danh sach chuoi" / "tracker.fps sai kieu" (`nap_ctx` → `doc_json_dv` / `loi_schema`), `[lan_chay]` + "stt sai
  kieu" (`SCHEMA_GOI_CON`), "DUNG HOI" (`cap_nhat_so` in `[DUNG HOI] fixture da dung (...)`), "Windows fatal exception"
  (faulthandler Windows khi `_read_null`); đối chiếu phân loại tiêm với từng ca tiêm hiện có (crash_ghi, cha_gian_doan,
  job_pid_log gồm job_loi/resume_loi/bo_faulthandler/pid_khong_xac_dinh, nguong, tham dò) để không ca tiêm hợp lệ nào thành
  "ngoài dự kiến": resume_loi/job_loi giải thích mọi bất thường của lần chạy đó; pid_khong_xac_dinh giải thích qua
  `con_trang_thai.tiem_loi_fixture`; cha_sau_tao_con giải thích PID qua thiếu `con.json`. Đây là rà tĩnh, không phải bằng
  chứng chạy; nếu phân loại sai, B1 sẽ dừng hỏi (an toàn) thay vì báo đạt.
- Đã chạy (vòng sửa 2): `git status --short src/r61_bdd configs/phase2/r61_bdd handoff results/phase2/r61_bdd/tests` (chỉ
  đọc; các file B là untracked nên git không chứng minh nội dung file khác bất biến); Glob xác nhận `src/r61_bdd/cong_b/`
  chỉ có 4 file `.py` (không `__pycache__`, không `__init__.py`) và `results/phase2/r61_bdd/tests/cong_b_20261026/` CHƯA tồn
  tại. **Chưa chạy** bất kỳ Python nào (B0 cấm chạy fixture; quyền không có Python tổng quát / `py_compile`), nên chưa có
  bằng chứng cú pháp/thực thi — lỗi cú pháp hoặc logic nếu có chỉ lộ ở B1.
- Rà tĩnh vòng này: đối chiếu từng chuỗi nguyên nhân mà ca âm/ca hồi quy chờ (`phai nam duoi`, `ngoai vung`, `khong phai JSON
  hop le`, `sai kieu goc`, `tai lap:`, `cham lai:`, `lap-tracker kiem lai`, `cham-lai kiem lai`, `thieu con.json`, `KHONG
  xac nhan duoc con`, `THIEU THOI GIAN`, `thua van_hanh`, `chan: van_hanh/dung.json`, `sha256 review`, `cau_hinh …`) với
  thông điệp thật trong `kiem_dau_vao.py`, `van_hanh_cong_b.py`, `hop_dong_bdd.kiem_duong_ghi`; đối chiếu sự kiện nhật ký
  F4 (`tien-kiem:kiem_lai`, `mo-phien:chuan_bi`, `bo_qua`) và các trường hợp đồng fixture dùng (`phien_ban`, `loai_tru`,
  `pham_vi_pyc`, `du_phong_don_vi`). Không phát hiện lệch; đây là rà tĩnh, không phải bằng chứng chạy.
- Không sửa file ngoài 6 write_paths; không sửa PLAN, helper/hợp đồng/giao thức A, `DIEM_VAO_MA`, phép so khóa A.
- Rủi ro đã biết: (1) gán job lồng nhau (job fixture → job cha B → con) dựa vào hỗ trợ nested job của Windows 8+; nếu
  không đạt, runner trả 40 "Job Object" — B1 sẽ lộ. (2) Số `so_file` loại trừ của phiên A thật (18/16/55/1/1) chỉ kiểm
  được khi chạy tiền kiểm B thật. (3) Tiền kiểm B thật đọc review ngoài repo (`D:/LABOTARY/automation/…`). (4) Ca âm
  `am_that_*` gọi runner với hợp đồng/giao thức thật nhưng dừng ở kiểm phê duyệt/tiêm lỗi trước mọi ghi (chỉ đọc).
  (5) Nguyên nhân crash native trước đây vẫn CHƯA RÕ.
  (6) Tham dò `ngat_giua_popen_gan`/trần cứng dựa vào job lồng ba tầng (job lệnh của fixture → job gốc của tham dò →
  cháu); nếu Windows từ chối gán, tham dò báo `dat=false` → kiểm không đạt (báo thật). (7) Một lần chạm trần cứng của
  chính fixture → `dung_fixture.json` → B1 dừng hỏi (bảo thủ có chủ ý, mục 6).
  (8) Mọi lỗi ngoài dự kiến không phải crash native (kể cả một exit 1 do Smart App Control chặn DLL) làm B1 dừng hỏi ngay
  — bảo thủ có chủ ý theo F8b. (9) Ca 11–13 gọi chính entrypoint fixture trên phiên con dưới `<vòng>/tham_do/…` và
  `<vòng>/so_gia/…` (10 lần, mỗi lần chụp pyc + import rồi dừng ở rà sổ); bằng chứng F8b/F8c phần lớn là tổng hợp đúng cấu
  trúc `lenh()`/`mot_doan()`, chỉ một bằng chứng lấy từ lần chạy thật của ca 09. (10) Vòng 4 làm fixture chặt hơn: exit 40
  của runner chỉ "bình thường" khi `dung.json` trỏ lần chạy có bất thường được tiêm của chính nó giải thích; nếu một ca
  thật hợp lệ (08/09/10) có bất thường mà phân loại tiêm chưa phủ, B1 sẽ dừng hỏi ở ca đó (an toàn, nhưng tốn một lời gọi
  sửa).
- Ngân sách: theo orchestrator đã dùng 4/10 vòng sửa-review trước lời gọi này (lời gọi trước dừng vì hết hạn mức tài khoản
  giữa chừng; bộ đếm do orchestrator quản lý). Lời gọi này hoàn tất vòng sửa 4 của B0 (B0 dự toán 2 lời gọi, đã vượt; phần
  vượt lấy từ "sửa theo review 2" và dự phòng). Nếu orchestrator tính lời gọi này là vòng 5, còn 5/10: B1 (2), retest (2),
  B2 (1) — **không còn dự phòng**; thêm một vòng sửa B0 nữa sẽ phải bớt retest B1.

## 10. Bước tiếp

(Cập nhật B2.) B0 → B1/bước gộp → B2 đã xong phía coder; còn Codex review B2, sau đó đối chiếu Pbeca24e7 chỉ đọc (không
chạy lại, không mở lan_03) rồi DỪNG. B thật cần kế hoạch thực thi riêng do người dùng duyệt và file phê duyệt riêng (B2.7,
B2.8); cập nhật PLAN cần bước chuyển khóa riêng sau B và báo cáo cuối.
