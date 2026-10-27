# R50A — Hồ sơ test11 và trạng thái tám đề xuất

Ngày 12/10/2026; cập nhật tiếp nối 13/10/2026. Codex triển khai tạm theo quyết định người dùng;
**R50A chờ review lại ở một phiên Codex riêng, chưa approve**. Mục 0 `handoff/PLAN_DA_CHOT.md` đã được cập nhật
trước khi sửa báo cáo này trong phiên tiếp nối.
Phiên này chỉ sửa ba tài liệu được phép; không chạy lại test11, kiểm kê, tracking hay phân tích DetA.

## 1. P64dfb0d8 — kết quả kế thừa, không phải phép chạy mới

Đã đọc trực tiếp `results/phase2/kc48_pf_code/test11_k1/ket_qua.json`: `dat: true`, `exit: 0`,
`n_phep_kiem` dự kiến 5 / đã chạy 5 / PASS 5, `loi: null`. Đây là **5/5 PASS** của lần chạy đã lưu,
không tuyên bố một lần test mới trong phiên sửa tài liệu.

Lệnh lịch sử trong JSON:
`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/chay_test11_k1.py`.
Wrapper chỉ import và gọi `src/test_pf_ctrv.py::test_11_runner_seeds`, không gọi main / test 1–10.
Bắt đầu `2026-10-12T04:01:02+0700`; test 6,72 giây, toàn wrapper 24,77 giây.
Hai lệnh runner con có `returncode: 0`: seed 0 và 1, rồi chạy lại seed 0 trên fixture `synth` 40 frame.

Năm phép kiểm ghi PASS:

1. `run_tracking PF --seeds 0 1` chạy xong.
2. Tạo hai thư mục `-s0`, `-s1`.
3. Chạy lại seed 0: file TXT bit-identical.
4. Seed 1 khác seed 0.
5. `run_meta` ghi seed.

### 1.1 HEAD, mã thực chạy và cấu hình K1

HEAD lúc chạy: `a7a69615f7d97f3558e5b78cd5bcfa32cdf14551`, nhánh `phase2-mixed-traffic`.
JSON ghi `src/pf_ctrv.py`, `src/tracker_ctrv.py` khi đó `khac_head`, wrapper `chua_theo_doi`:
K1 có trong **cây làm việc**, không được quy toàn bộ mã thực chạy về blob của a7a6961.
Sau đó snapshot `117d62da027e8ce3e4c9f455d576a924b4d82bbb` được tạo lúc 07:30:35 +07 và đã push theo bàn giao
người dùng. `git log -2` xác nhận hai commit / thời điểm trong repo; việc push kế thừa từ bàn giao, không kiểm mạng.
Không đổi `commit_head` trong bằng chứng cũ thành 117d62d.

SHA256 byte cây làm việc tại lúc test (trường `ma.*.sha256_cay_lam_viec`), đã đối chiếu lại bằng `Get-FileHash`
trong phiên tiếp nối ngày 13/10; cả sáu khớp:

- `src/pf_ctrv.py`: `de141de5aa698735a681ca80f0f342d4e2b49b2b8159ae67ee940a9e449a8f18`.
- `src/tracker_ctrv.py`: `b22b4688f50058e8be78e43d05cf0406812e375ddf3e37a12078eef02d31c275`.
- `src/test_pf_ctrv.py`: `cc9a66b865cf590e0ad7757315755b06130cd3b4b37dd1ef8a36f822c27b0085`.
- `src/run_tracking.py`: `5bd131fe958bdf04ed59f704c85919aec43049df1922cb552dced98324910629`.
- `src/det_cache.py`: `d253c2b8063fa15f8c13846eb26555e9f0413a6454c388b7685096081d0c032e`.
- `src/kc48/chay_test11_k1.py`: `67d4777bdef556c64f9988adfa4eb95e6ae396cd482caf9dcd93b9bece51c0e3`.

YAML nạp thực: `configs/phase2/trackers/bytetrack_pf_ctrv_n100.yaml`, SHA256
`9e2e4f88acea015877e9e08bcbca525fdc55c6231530fb378c7d94a135191054`, cũng khớp byte hiện tại.
N=100, `pf_estimate: mean`, `pf_roughening: 0.0`, seed 0/1; không có khóa `pf_optimal_proposal`.
**K1 code-present nhưng disabled (mặc định tắt)**. Test này kiểm runner mặc định trên mã đã có K1,
không chứng minh runner K1 bật tái lập seed. Không ngoại suy fixture sang dev12, `run_experiment.py` hay bộ lọc khác.
Hash byte trên đĩa có thể khác hash blob Git do CRLF/LF; JSON có riêng hai loại hash, không tráo chúng.

### 1.2 Fixture và giới hạn kiểm phạm vi

Nguồn: `fixture_truoc.json`, `fixture_sau.json` cùng thư mục `test11_k1/`, và các trường
`so_fixture_truoc_sau`, `doi_chieu_fixture_r48c` trong `ket_qua.json`.

- `data/interim/det_cache/fixture_pf_seed_test/`: đổi nội dung `synth.json`; `meta.json`, `synth.npz` chỉ đổi mtime.
- `data/processed/trackers/DETRAC-sample/p2-pftest-s0/` và `p2-pftest-s1/`: mỗi thư mục đổi nội dung
  `run_meta.json`, `summary.csv`, `timing/synth.csv`, `video_meta/synth.json`; `data/synth.txt` và
  `data_cls/synth.csv` chỉ đổi mtime. Không thêm / mất file trong ba thư mục theo hồ sơ.
- Ngoài fixture: 24.164 file trước / sau, không thêm / mất / đổi theo **kích thước và mtime_ns** trong
  `data/interim`, `data/processed`. Không có hash nội dung toàn vùng; không tuyên bố chứng minh mọi byte đều giữ nguyên.
- Đối chiếu với R48C trong JSON: TXT seed 0 giữ SHA256
  `dcafead683354e5071bfa1a83cc05d6b1510873c4b1aa53d03621c8267c50fde`; TXT seed 1 giữ
  `0ac1d30645f03690f048171dfe0c3944b1dd51d8decb1eb95b233506748a8fe8`.
  Hai CSV lớp, `meta.json`, `synth.npz` cũng trùng theo hồ sơ; metadata thời gian có thể đổi.
  Phiên này đọc kết quả so đã lưu, không quét lại fixture / 24.164 file.

SHA256 các artifact đã tính lại ngày 13/10 bằng `Get-FileHash` để định danh bản đọc:

- `ket_qua.json`: `75ae733d6d387e38a0a385711006543e52449ae51036cf331c38786bc96b2b9b`.
- `fixture_truoc.json`: `1e907b38c3f21a1c259777f7a3c4e2e2b05e2544f1a6339bf06d1be8e0b0e916`.
- `fixture_sau.json`: `34fa9bd10eae0b0088c3ec31d086d809ac710281a94c3063cc01c5380f8e9193`.

## 2. P487415eb — thiết kế DetA, chưa thực thi

`09_thiet_ke_phan_tich_deta_pf_ekf.md` bao gồm (a) đầu vào / so sánh được, (b) dự đoán box / association hai tầng /
xác nhận / vòng đời / likelihood, (c) M1–M8 có định nghĩa, mẫu số, tổng hợp video và giới hạn diễn giải,
(d) rung của sai số so GT tách chuyển động thật, (e) khả năng và thiếu hụt log, (f) chi phí / pha dữ liệu cũ và pha mới.
CI chứa 0 không bác bỏ giả thuyết. MOT không chứng minh đường gán, prior, likelihood hoặc sự kiện nội bộ.
Log mốc 2 không tự thay cho kc48 chỉ vì MOT trùng. Phần dư so GT không phải NIS hiệu chuẩn hoặc cận chắc chắn.
Pha A/B/C/D đều chờ duyệt thực thi; không tính M1–M8, không kiểm A0, không instrumentation trong phiên này.

Kiểm kê kế thừa: `results/phase2/kc48_pf_dev/thiet_ke_deta/kiem_ke.json`, thời điểm
`2026-10-12T04:06:00+0700`, trường `giay: 5.61`, SHA256 byte đã đọc
`2ee1d28f3fa70c99e6bbd91c49f12117cbde78e590e1bb55e7f0f2171ce55030`.
Lệnh lịch sử ghi trong JSON: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/kc48/kiem_ke_dau_vao_deta.py`.
Không chạy lại; JSON không có trường exit nên không tự gán exit 0 cho lệnh kiểm kê.
Đã đọc chọn trường `runs_csv`, `video_dev12`, `truong_can` và số dòng / số video log:
26 run kc48, dev12; kc48 thiếu assoc_log/pf_diag, mốc 2 có log base, cả hai thiếu likelihood/NIS đầy đủ cho EKF/PF.
Kiểm kê chỉ là danh mục, không phải kết quả phân tích hay kiểm chứng tính tương đương.

Provenance cũ: commit `935a445` đổi quy ước hash TXT CRLF → LF. Khi `src/audit_run_provenance.py` chưa hỗ trợ
hash trước mốc đó, cảnh báo “không khớp” **không chứng minh run cũ không hợp lệ**. Không so hash metadata khác đời
như thể cùng quy ước; không sửa mã audit / bằng chứng trong mốc này.

## 3. Tám đề xuất và trạng thái nghiệm thu

1. **P64dfb0d8 — đã chạy**: test11 5/5 PASS, exit 0 theo artifact; giới hạn K1 tắt như mục 1.
2. **P487415eb — chỉ thiết kế**, chờ người dùng duyệt thực thi phân tích; chưa mở pha A/B/C/D.
3. **P2542b6c1 — đã commit a7a6961** ở phiên thiết lập; sau đó snapshot 117d62d đã push theo người dùng.
   Việc lưu Git không phải nghiệm thu R50A; phiên này không stage/commit/push.
4. **P9a6c2078 — đóng: đã hoàn thành qua R49 / đề xuất trùng**, không phải bác bỏ nội dung nghiên cứu.
   Đối chiếu mục R48E trong mục 0: có nguồn approve `0015`, 131 đạt / 26 chưa kết luận / 3 không đạt,
   giới hạn mẫu 160 ảnh, nguyên nhân ba ảnh chưa chứng minh và giới hạn khóa tiêu chí.
   R49A `nghiem_thu_r49/0004_review_answer.json` approve phần đồng bộ, issues rỗng; MANIFEST ghi SHA256
   `789b5b44ec670ee9607cba10f32c6bd6470f71dd0d97b22f401d47d526b912cb`, đã đối chiếu byte hiện tại khớp.
   Hồ sơ P9a6c2078 gốc chưa quyết định là lịch sử trước quyết định có điều kiện R50; không sửa hồ sơ gốc.
5. **Pee184526 — HOÃN**.
6. **Pf350787b — HOÃN**.
7. **P3c58df9b — HOÃN**.
8. **Paa38fde5 — HOÃN**.

**R49A đã approve; R50A chờ review lại, chưa approve.** Giới hạn biên bản R49A: review tĩnh, không chạy lại thực nghiệm;
reviewer đã bổ sung diff bản đầu mốc cho R49A. Việc đó không thay cho kiểm diff / cổng review R50A.

## 4. Bàn giao, backend tạm và quota

Đã đọc trực tiếp bằng shell chỉ đọc:
`D:/LABOTARY/automation/runs/bytetrack_r50_test11_thiet_ke_deta_20261012_0356/handovers/handover_0001.md`.
Tự tính SHA256 ngày 13/10 bằng `Get-FileHash`, khớp
`dee2e42fb368aff6e6b752bd759f98e5c4f7f0fee27f0674d7fcb9b00c8d0210`.
Handover ghi Claude mất kết nối trước báo cáo JSON (CLI đếm 83 turn); test11 / kiểm kê / bản nháp 09 đã có.
Không coi phiên Claude đó hoàn tất. Việc hết quota sau đó và stop chính thức là thông tin bàn giao vận hành.

Người dùng **đã hủy pause, cho phép tiếp tục ngày 13/10**. Run Codex cũ
`D:/LABOTARY/automation/runs_local/bytetrack_r50_codex_20261012_0738` giữ nguyên; recover bị chặn đúng bởi báo cáo
người dùng tạo ngoài write_paths. Checkpoint `D:/LABOTARY/automation/checkpoints/resume_r50_r51_20261013_1806/before`
giữ nguyên, cùng checkpoint chuyển backend cũ `D:/LABOTARY/automation/checkpoints/r50_codex_switch_20261012_073357/`.
Run hiện tại: `D:/LABOTARY/automation/runs_local/bytetrack_r50_resume_20261013_1806`, bước `R50A_CLOSE_WITH_EVIDENCE`.
Call 0001 của run mới lỗi setup sandbox theo bàn giao; không đồng nhất với handover Claude gốc hay coi exit 0 là nghiệm thu.
Coder không sửa sandbox, state, plan snapshot, control hoặc logs điều phối.

Backend thực **Codex/Codex**, `gpt-6-astra/high`, triển khai và review ở **hai phiên riêng**; không phải kiểm chứng
chéo hai mô hình. Claude giữ `claude-opus-5-5/high` theo kế hoạch, chưa xác nhận reset; chỉ quay lại tại ranh giới an toàn.
Auto-plan tắt, Jetson Nano hoãn. Ngân sách trước lời gọi đã dùng 2/3 vòng sửa-review; đây là vòng sửa còn được phép,
không tăng ngân sách. Sau hai vòng liên tiếp không tiến triển, orchestrator dừng hỏi người dùng.
Chỉ sau R50 complete + approve, bộ nối tất định mới khởi tạo run R51 riêng
`D:/LABOTARY/automation/runs_local/bytetrack_r51_controlled_20261013_1806`; phiên này không mở R51.

Quota đọc trực tiếp từ file, không phải coder gọi API:

- `D:/LABOTARY/automation/checkpoints/resume_r50_r51_20261013_1806/quota_preflight.json`:
  `observed_at: 2026-10-13T18:10:07+07:00`, nguồn `account/rateLimits/read`, 5 giờ **28%**, tuần **20%**.
- `D:/LABOTARY/automation/runs_local/bytetrack_r50_resume_20261013_1806/quota_latest.json` trước call:
  `observed_at: 2026-10-13T23:06:34+07:00`, cùng nguồn, 5 giờ **4%**, tuần **32%**,
  `auth_type: chatgpt`, `ordinary_usage_allowed: true`, `planType: plus`.
- Đây là snapshot orchestrator đã thu thập, không bảo đảm quota tại thời điểm lời gọi kế tiếp. Số 34% / 46%
  thiếu giờ ngày 12/10 và 84% / 54% lúc 07:43:16 +07 trong review cũ chỉ là lịch sử.
- Orchestrator phải đọc quota trực tiếp và kiểm ngân sách trước lời gọi mới; 5 giờ ≥90% hoặc tuần ≥80% thì
  checkpoint/chờ theo công cụ. Chờ reset +5 phút vẫn phải đo lại. Chỉ dùng gói thuê bao, không API key hay extra usage.

## 5. Tự kiểm trực tiếp F1–F6 và bàn giao reviewer

Nguồn finding đã đọc nguyên văn, khớp context người dùng:
`D:/LABOTARY/automation/runs_local/bytetrack_r50_codex_20261012_0738/reports/review_001.json` (`verdict: revise`).
Các dòng trong finding chỉ bản cũ; bảng dưới dẫn mục/nhãn ổn định của bản sửa. Đây là **tự kiểm của coder**,
không tự nhận reviewer đã đóng finding. Phần sửa kế thừa cũng được đọc lại, không chỉ kiểm diff init mới.

| Finding | Vị trí sửa / nội dung hiện tại | Bằng chứng tự kiểm trực tiếp ngày 13/10 |
|:--|:--|:--|
| F1 — hồ sơ / trạng thái | PLAN mục 0: R49A approve, test11 đã chạy, R50A chờ review; 10 mục 1–4: đủ tám đề xuất, provenance, K1 tắt, backend, quota, pause và checkpoint | Đọc `ket_qua.json` (5/5 PASS, exit 0 kế thừa), YAML và mặc định `PFCTRV.__init__`; sáu hash mã + YAML khớp; R49A approve / issues rỗng và hash khớp MANIFEST; đọc handover, quota và ba snapshot |
| F2 — thời điểm n_lost | 09 mục 1.2, M7a/M7b: chọn state Tracked + is_activated + đầu ra cùng ID ở frame trước; không lọc assoc_log.n_lost=0 | `src/assoc_log.py:78,120–124` ghi sau predict / trước gán: frame tăng một, end_frame ở frame trước ⇒ 1; `PFTrackView.update` tại `src/pf_ctrv.py:390–395` và `STrack.update` đặt frame hiện tại trước gọi bộ lọc ⇒ pf_diag có thể là 0; hai thời điểm khác nhau |
| F3 — CI / chọn mẫu M4 | 09 mục 2 (HD1, HD2, HD6), quy ước mục 3, M4/M5/M7a, mục 5 | Đọc lại các nhánh diễn giải: CI chứa/chạm 0 hoặc hiệu gần 0 là chưa phân biệt được; M4 chỉ cặp cùng ghép, không bác bỏ lỗi ở cặp bị loại; không có biên tương đương. Kiểm bằng đọc và `rg`, không chạy bootstrap |
| F4 — MOT và nội bộ | 09 A0, đoạn sau A0, quy ước mục 3, mục 6.2 và pha B/C/D mục 7 | A0 chỉ kiểm đầu ra; MOT trùng không chứng minh prior/association/trọng số. `kiem_ke.json` ghi kc48 thiếu assoc_log/pf_diag; log B chỉ giải thích chính run mốc 2 khi thiếu bằng chứng thực thi tương đương. Chưa chạy A0; instrumentation vẫn chờ duyệt |
| F5 — suy vòng đời từ MOT | 09 mục 0.1, HD4, M3, M6/M6e và mục 4 | M3/M6 chỉ proxy khoảng trống/ID/đoạn, có thể do đổi ID hoặc ghép GT khác. Sự kiện new/lost/removed lấy từ M6e có log. Đối chiếu schema kiểm kê và mã assoc_log; không suy sự kiện từ bảy cột MOT |
| F6 — nhiễu GT / phần dư | 09 M8a, mục 4, mục 6.4 | Sai phân bậc hai gồm gia tốc thật và nhiễu nhãn, không phải sàn nhiễu; phần dư detector−GT có thể cộng/triệt tiêu sai số, không là cận trên hay NIS. Đọc mã R của PF/EKF để xác nhận h_det/h_prior và confidence; phép CV cửa sổ chỉ là kiểm độ nhạy dự kiến |

### 5.1 Bản đầu mốc và cách đối chiếu

Ba gốc snapshot đã đọc manifest và file trực tiếp:

1. **S_Claude** = `D:/LABOTARY/automation/runs/bytetrack_r50_test11_thiet_ke_deta_20261012_0356/milestones/01_R50A_TEST11_THIET_KE_DETA_TRANG_THAI_f9a2e5cc`.
   PLAN được lưu; manifest ghi 09 và 10 `action: absent`.
2. **S_Codex_cu** = `D:/LABOTARY/automation/runs_local/bytetrack_r50_codex_20261012_0738/milestones/01_R50A_TEST11_THIET_KE_DETA_TRANG_THAI_aab373e9`.
   PLAN và 09 được lưu; manifest ghi 10 `action: absent`.
3. **S_tiep_noi** = `D:/LABOTARY/automation/runs_local/bytetrack_r50_resume_20261013_1806/milestones/01_R50A_CLOSE_WITH_EVIDENCE_86087d1c`.
   Có cả ba tài liệu. Trước sửa lần này, `git diff --no-index --numstat` từng file đều exit 0.

Với file có bản lưu: `git diff --no-index -- <S>/files/<path> <path>`; exit 1 là có khác biệt, không phải lỗi kiểm.
Với file `absent`, dùng `git diff --no-index -- /dev/null <path>` để đọc toàn bộ file mới, đồng thời xác nhận manifest
và đường snapshot thực sự không có file. Không gọi diff với HEAD là diff bản đầu mốc.
Đối chiếu byte ngoài mục 0 PLAN bằng Python: giữ nguyên phần trước `## 0.` và từ `## Thứ tự các mốc` đến hết
so với cả ba snapshot. Các phần R48/R49 đã nghiệm thu được giữ nguyên trong lần sửa này.

### 5.2 Lệnh và giới hạn tự kiểm

- `Get-Content -Encoding UTF8` và Python `-X utf8` đọc JSON chọn trường: test11, kiểm kê, R49A/MANIFEST,
  review F1–F6, handover, quota và manifest snapshot; không chạy wrapper test hay script kiểm kê.
- `Get-FileHash -Algorithm SHA256` cho sáu file mã test11, YAML, R49A, bốn artifact mục 1–2 và handover: exit 0;
  sáu phép so với `ma.*.sha256_cay_lam_viec` đều True. Hash artifact định danh bản đọc, không thay kiểm toàn bộ fixture.
- `git rev-parse HEAD`, `git log -2 --format="%H %ci %s"`: exit 0, xác nhận 117d62d / a7a6961 và thời điểm mục 1.1.
- `git diff --no-index -- <S>/files/<path> <path>` (hoặc `/dev/null` khi absent), chạy qua Python `subprocess.run`
  cho ba tài liệu × ba snapshot: cả **9 diff exit 1** do có thay đổi; đã đọc diff, gồm toàn bộ file mới ở snapshot cũ.
  Sáu file snapshot có lưu đều khớp SHA256 manifest; ba trường absent khớp thực tế. Python kiểm các assert này exit 0.
- Python `-X utf8 -` so byte ngoài mục 0 PLAN với cả ba snapshot: **prefix và suffix đều trùng**; phần R48/R49
  đã nghiệm thu so với snapshot tiếp nối cũng trùng. Kiểm tám ID đề xuất trong mục 3 đúng thứ tự và sáu hàng F1–F6:
  exit 0. Lần đầu tìm mốc tiêu đề Unicode qua pipe PowerShell lỗi `ValueError` / exit 1; sửa phép tìm thành ranh giới
  heading ASCII `^## `, chạy lại exit 0, không thay nội dung hay tiêu chí kiểm.
- `git diff --check`: exit 0, không lỗi whitespace. `git status --short`: exit 0; PLAN/09 modified, 10 untracked,
  **còn báo cáo tiến độ `handoff/BAO_CAO_TIEN_DO_HIEN_TAI_20261013_1238.md` untracked từ đầu phiên**.
  Đã so nguyên byte báo cáo đó với bản đọc đầu phiên lưu trong bộ nhớ: không đổi. Không nhận rằng toàn cây chỉ có
  ba đường thay đổi so với HEAD. Chỉ ba write_paths được ghi trong phiên này; file 10 đã kiểm riêng vì diff Git
  mặc định không bao gồm untracked. `git diff --stat` exit 0 chỉ liệt kê hai file tracked là đúng hành vi Git.
- Hai lần tìm mã thử đường `src/pf_diag.py` / `src/ctrv_base.py` không tồn tại (`rg` lỗi đường dẫn; lệnh gộp không
  phản ánh riêng exit đó). Đã đọc đúng `src/pf_ctrv.py`, `src/ekf_ctrv.py`, `src/tracker_ctrv.py`, `src/assoc_log.py`
  và `ultralytics/trackers/byte_tracker.py`; không dùng lần đọc lỗi làm bằng chứng. Lần đọc đầu thiếu `-Encoding UTF8`
  bị lỗi hiển thị dấu; đã đọc lại các đoạn cần dùng với UTF-8. Diff dài bị cắt được đọc lại theo phần.

Không chạy test mới, M1–M8, K/A0, kiểm kê, tracking/eval/GPU hoặc phân tích DetA. Kết quả test11 là bằng chứng kế thừa
đã đọc trực tiếp, không phải PASS mới. Không kiểm lại toàn bộ fixture / đầu ra thí nghiệm; không xác minh push bằng mạng.
Không sửa mã / cấu hình / bằng chứng, không Git mutation, không agent con, không tiến trình nền.
Reviewer Codex riêng phải kiểm cả sáu finding và toàn bộ phần kế thừa trên ba tài liệu, các snapshot và bằng chứng;
R49A approve không thay cổng R50A. Dừng ở R50A; thực thi P487415eb cần người dùng duyệt riêng.
