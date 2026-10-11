# R48C (M3 + phần tính đúng của M4) — Test tính đúng PF và smoke pipeline

Ngày 2026-10-11, HEAD `eaf2e87`, nhánh `phase2-mixed-traffic`. Mọi lệnh chạy bằng
`C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe` (viết tắt `python`), foreground, từ gốc repo. Nhãn: [QS] quan sát
có file / lệnh dẫn, [GT] giả thuyết chưa kiểm, [QĐ] quyết định đã ghi trong giao thức.

## 1. sha256 của giao thức và thứ tự thời gian

Giao thức: `handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md`, **sha256
`7063c532de0c1ec84aced98fb21f7a93645be01b843a1eb3f160bbf09b50e2e4`**, 22.510 byte. Bản chụp nằm ở
`results/phase2/kc48_pf_code/giao_thuc_sha256.json` (lệnh `python src/kc48/ghi_sha256.py --ten giao_thuc_sha256 ...`,
script chỉ thêm mục mới, không ghi đè). Từ thời điểm này mục 1–12 của 04 không sửa; lệch giao thức chỉ ghi ở phụ lục.

**Cập nhật theo bản R48C người dùng chốt 11/10/2026.** Phụ lục P1 của 04 rút ngoại lệ "trừ test 11" ở mục 10 (mục 4
dưới đây). Sau khi thêm P1, file 04 có **sha256 `479497fc0e3e86150f39fac2da60bacec285c7455d3f355f660e7cb29f0938ef`**,
24.127 byte, theo lần chụp 4. Cũng ở lần chụp 4, script kiểm **22.510 byte đầu** (mục 1–12 đã khóa) vẫn có sha256
`7063c532…` (`tien_to.khop: true`, lệnh có `--tien-to`), nên phần đã khóa không đổi. Truy nguyên ở mục 6 không làm lệch
giao thức, nên không có phụ lục P2 (ghi chú của lần chụp 3 có nhắc P2 trước khi xác định điều này).

| Thời điểm (+07:00) | Sự kiện | Bằng chứng |
|:--|:--|:--|
| 04:29:04 | Ghi xong 04 (mtime) | `giao_thuc_sha256.json`, lần chụp 1, trường `mtime` |
| 04:29:39 | **Chụp sha256 lần 1** | `giao_thuc_sha256.json`, lần chụp 1 |
| 04:29:40 | Kiểm cache dev: trùng khít tham chiếu mốc 2 | `results/phase2/kc48_pf_code/cache_dev.json` (`trung_khit: true`) |
| sau đó | Tự kiểm bộ mô phỏng; test tính đúng lần 1 (exit 1) và lần 2 (29/29) | mục 2, mục 3 |
| 06:51:50 | **Chụp sha256 lần 2**, trước smoke: trùng lần 1 | `giao_thuc_sha256.json`, lần chụp 2 |
| 06:52:08–06:52:38 | Smoke bước track, exit 0 | `results/phase2/kc48_pf_smoke/nhat_ky_wrapper.jsonl` dòng 1 |
| 06:53:08–06:53:15 | Smoke bước eval, exit 0 | `nhat_ky_wrapper.jsonl` dòng 2 |
| 06:53:26 | Tổng hợp thời gian | `results/phase2/kc48_pf_smoke/thoi_gian.json` |
| 15:10:30 | Truy nguyên EKF so với mốc 2 (chỉ đọc) | `results/phase2/kc48_ekf_tai_lap/so_sanh_dong.json`; `05b_truy_nguyen_lech_ekf.md` |
| 15:11:48 | Chụp fixture trước test 11 | `results/phase2/kc48_pf_code/fixture_truoc.json` |
| 15:11:48–15:12:12 | `python src/test_pf_ctrv.py` đủ 11 test, exit 0 | mục 4 |
| 15:12:12 | Chụp fixture sau, so | `results/phase2/kc48_pf_code/fixture_sau.json` (`chi_fixture_thay_doi: true`) |
| 15:12:53 | **Chụp sha256 lần 3**: trùng lần 1 | `giao_thuc_sha256.json`, lần chụp 3 |
| 15:13:28 | **Chụp sha256 lần 4**, sau phụ lục P1: tiền tố 22.510 B trùng lần 1 | `giao_thuc_sha256.json`, lần chụp 4 |

Trước lần chụp 1 không có lần chạy so sánh bộ lọc nào. Kiểm cache, tự kiểm bộ mô phỏng và test tính đúng không so sánh
bộ lọc. Thí nghiệm mô phỏng so sánh EKF / PF và pipeline dev **chưa chạy**, vì thuộc R48D.

## 2. Test: lệnh, exit, kết quả

| Lệnh | Exit | Kết quả |
|:--|:--|:--|
| `python src/kc48/test_pf_dung_dan.py` | 0 | 11 test, **29/29** phép kiểm PASS |
| `python src/test_pf_ctrv.py` (file gốc, **đủ 11 test gồm `test_11_runner_seeds`**) | 0 | **26/26** phép kiểm PASS (mục 4) |
| `python src/kc48/chay_test_pf_cu.py` (lời gọi trước; test 1–10, khi test 11 còn chờ quyết định) | 0 | 21/21 PASS; nay được thay bằng dòng trên |
| `python src/test_ekf_ctrv.py` | 0 | 14/14 PASS |
| `python src/test_ukf_ctrv.py` | 0 | 12/12 PASS |
| `python src/test_tracker_lifecycle.py` | 0 | 20/20 PASS |
| `python src/test_init_regression.py` | 0 | 17/17 PASS |
| `python src/kc48/mo_phong_quy_dao.py --tu-kiem` | 0 | 10/10 đạt, `results/phase2/kc48_pf_code/mo_phong_tu_kiem.json` (`dat_tat_ca: true`) |
| `python src/kc48/kiem_cache_dev.py` (lời gọi trước) | 0 | `cache_dev.json`: meta / npz / weights sha256 của 12 video dev trùng `eval_meta.json` tham chiếu, trên đĩa và tại HEAD |

Trong lời gọi chốt 11/10, chiều 2026-10-11 (sau 15:12), các lệnh sau được chạy lại và cho đúng số và exit như bảng:
`test_pf_ctrv.py` (lần đầu chạy đủ), `test_pf_dung_dan.py`, `test_ekf_ctrv.py`, `test_ukf_ctrv.py`,
`test_tracker_lifecycle.py`, `test_init_regression.py`. Dòng cuối lấy exit từ bàn giao của orchestrator cho lời gọi trước. Bàn giao đó chỉ ghi một lệnh lỗi, là lần 1 của
test tính đúng (mục 3). Lệnh này không chạy lại trong lời gọi này, để giữ dấu thời gian 04:29:40.

### 2.1 Tính chất (3) của nhiệm vụ → test trong `src/kc48/test_pf_dung_dan.py`

Các giá trị kỳ vọng lấy từ nguồn **độc lập** với cài đặt: lý thuyết, nghiệm đóng, tính chất toán học hoặc phản thực.
Test không chép lại phép tính của mã.

| Tính chất | Test | Kỳ vọng độc lập |
|:--|:--|:--|
| Trọng số hữu hạn, tổng = 1 | 5, 9 (cả 1) | 60 lần kiểm (track × frame) trên pipeline `PFByteTracker` có mất đo và resample |
| Đúng hậu nghiệm (một bước, mô hình tuyến tính–Gauss) | 1 | Hậu nghiệm Kalman tính tay từng tọa độ; N = 40.000, sai lệch trung bình < 5·sd/√ESS, phương sai lệch < 10 % |
| ESS | 2 | ESS = 1/Σw² trên trọng số biết trước (đều → N, một điểm → 1, hai khối → 2), bất biến khi cộng hằng vào logw |
| Resampling hợp lệ | 3, 4 | 200 bộ trọng số: chỉ số trong [0, N), tổng đếm N, mỗi đếm ∈ {⌊Nwᵢ⌋, ⌈Nwᵢ⌉}, trọng số 0 → đếm 0; trung bình đếm qua 20.000 u ≈ Nwᵢ; dưới ngưỡng ESS thì hàng mới là bản sao hàng cũ, logw đều |
| Độc lập trạng thái giữa track | 10 | Phản thực: đổi / bỏ detection của B thì particle + logw của A **bit-identical** ở mọi frame; thêm track C thì A, B bit-identical; không cặp nào chung bộ nhớ (`np.shares_memory`) hay chung đối tượng rng |
| dt | 6 | Tính chất nửa nhóm của CTRV hằng: f(dt = 2) = f(dt = 1)∘f(dt = 1), sai số < 1e-9; khóa yaml `ctrv_dt` tới được bộ lọc |
| Chuẩn hóa góc | 7 | 200 bước ω = 0,3 qua biên ±π: θ ∈ [−π, π); hướng (cos, sin) và vị trí khớp nghiệm đóng cung tròn |
| Cập nhật khi không có đo | 8, 9 | `predict(lost = True)` đặt v_h = 0 trước khi dự đoán; logw giữ từng bit; trên pipeline track LOST giữ logw qua 3 frame mất đo, particle vẫn dự đoán, ghép lại cùng ID thì logw đổi |
| Ổn định số khi log-likelihood cực trị | 5 | loglik ≈ −1,25·10⁹ (exp trực tiếp = 0): logw sau update hữu hạn, tổng 1; particle gần z nhất nhận trọng số lớn nhất; logw có −inf vẫn hợp lệ |
| Tái lập theo seed | 11 | Chuỗi 40 bước qua 31 lần resample và 2 lần nhánh suy biến: cùng (seed, video, track_id) → bit-identical; khác track_id / video → khác |

## 3. Lỗi và sửa lỗi

**Không sửa `src/pf_ctrv.py` hay `src/tracker_ctrv.py`.** `git status` không ghi thay đổi nào ở hai file này. Không có
test nào hỏng do mã tracker, nên không có điều kiện "hỏng trước sửa, đạt sau sửa" để sửa mã. Kết quả cũ không bị ảnh
hưởng từ phía mã.

Lần chạy đầu của `test_pf_dung_dan.py` (lời gọi trước) **exit 1, 26/28**: có 2 phép kiểm hỏng. Cả hai là lỗi
**thiết kế test**, không phải lỗi mã. Nguồn: log thực thi
`D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\calls\0015_coder\stdout.log`; dòng 635 chứa bản test
lần 1, dòng 641 chứa đầu ra lần 1, dòng 698 chứa lần chạy lại sau sửa test, 29/29.

1. Test 9, phép kiểm "B ghép lại ở frame 13 cùng ID 2, logw đổi". Với `pf_resample_ess_ratio 0.5`, gần như mọi update
   đều resample, và sau resample logw là −log N. Vì vậy logw trước và sau khi ghép lại đều đều, nên kỳ vọng "logw đổi"
   sai một cách tầm thường. Đã sửa test: phần kiểm LOST và ghép lại chạy với ratio 0, để logw trước mất đo không đều
   (ptp = 4050,77). Đã thêm kiểm trạng thái Lost / Tracked.
2. Test 11 đòi `n_degenerate == 1`. Chuỗi thật có 2 lần suy biến: lần nhảy tới (5000, 5000) và lần quay về. Ở lần 1,
   bản thân phép so bit-identical giữa hai lần chạy không hỏng; cả hai lần chạy đều cho `n_resample 31, n_degenerate 2`.
   Đã sửa thành `n_degenerate >= 1`, đồng thời đòi bộ đếm của hai lần chạy bằng nhau.

## 4. Test 11 của `src/test_pf_ctrv.py`: đã chạy đủ (quyết định người dùng 11/10/2026)

**Lịch sử.** Ở lời gọi trước, `test_11_runner_seeds` chưa chạy. Test này xóa rồi ghi lại
`data/interim/det_cache/fixture_pf_seed_test/` và ghi `data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1}/`, khi đó
ngoài `write_paths`. Giao thức 04 mục 10 có ngoại lệ "trừ test 11" do agent tự thêm. Review cuối của Codex trả BLOCKED
vì lý do này.

**Quyết định.** Ngày 11/10/2026, người dùng cho test ghi 3 thư mục fixture nói trên. Ngoại lệ được rút bằng phụ lục P1
của 04.

**Cách chạy và bằng chứng** (lệnh nào cũng exit 0):

| Bước | Lệnh | Kết quả |
|:--|:--|:--|
| 1 | `python src/kc48/chup_fixture.py --ghi truoc` | `results/phase2/kc48_pf_code/fixture_truoc.json` (15:11:48): sha256 + byte + mtime_ns của 3 + 6 + 6 file trong 3 thư mục fixture (cả 3 đã có từ trước, mtime khoảng 2026-10-10 18:29–18:30 +07. [GT] Khớp khung thời gian lệnh `src/test_pf_ctrv.py` của R48A (lệch phạm vi F4 trong `handoff/PLAN_DA_CHOT.md` mục 0), chưa đối chiếu với log); byte + mtime_ns của **24.164** file còn lại trong `data/interim`, `data/processed` |
| 2 | `python src/test_pf_ctrv.py` | 11 test, **26/26** PASS. Test 11: `run_tracking PF --seeds 0 1 chay xong`; `tao 2 thu muc -s0 va -s1`; **`chay lai seed 0: file txt bit-identical`**; `seed 1 khac seed 0`; `run_meta ghi seed` |
| 3 | `python src/kc48/chup_fixture.py --ghi sau` | `fixture_sau.json` (15:12:12) + mục `so_voi_truoc`: ngoài fixture **đổi 0, thêm 0, mất 0** trên 24.164 file → **`chi_fixture_thay_doi: true`** |

Chi tiết trong 3 thư mục fixture (`so_voi_truoc.fixture`):

| Thư mục | Ghi lại (mtime đổi) | sha256 giữ nguyên | sha256 đổi |
|:--|--:|:--|:--|
| `det_cache/fixture_pf_seed_test` | 3/3 | `meta.json`, `synth.npz` | `synth.json` |
| `DETRAC-sample/p2-pftest-s0` | 6/6 | `data/synth.txt`, `data_cls/synth.csv` | `run_meta.json`, `summary.csv`, `timing/synth.csv`, `video_meta/synth.json` |
| `DETRAC-sample/p2-pftest-s1` | 6/6 | `data/synth.txt`, `data_cls/synth.csv` | như s0 |

[QS] File tracking `data/synth.txt` của cả hai seed trùng sha256 với bản có từ trước:
- s0: `dcafead6…`, 8.608 B;
- s1: `0ac1d306…`, 8.675 B.

Các file đổi sha chỉ chứa thời điểm, thời gian chạy hoặc dấu vân tay. [GT] Nhiều khả năng do dấu thời gian, chưa so từng
trường.

Lần chạy lại seed 0 trong test 11 thật sự ghi đè file: lệnh không có `--skip-existing`, nên `run_state.plan_videos` đưa
mọi video vào danh sách phải chạy (`src/run_state.py:74`). Vì vậy phép so bit-identical không phải so một file với
chính nó.

Phần thay thế tạm của lời gọi trước vẫn giữ làm bằng chứng phụ:
- Test 11 cấp bộ lọc của `test_pf_dung_dan.py`.
- Hai thư mục seed của smoke `p2-kc48_pf_smoke-A-pf100-s{0,1}`.

## 5. Smoke pipeline `kc48_pf_smoke`

### 5.1 Cấu hình và cách chạy

- `configs/phase2/experiments/kc48_pf_smoke.yaml` khai báo như `expB_detrac_dev_fix1.yaml`: cache
  `yolov8n-coco_conf0.05_all`, điều kiện A `conf_min 0.25`, `classes [2, 3, 5, 7]`, `max_det 300`, `fps 25`,
  `elapsed_mode skip`.
- Video smoke: 2 video đầu theo thứ tự manifest dev, `MVI_20052` và `MVI_20062`, tổng 1.494 frame. Quy tắc cố định
  trước khi chạy, không chọn theo kết quả.
- Biến thể: `ekf` (`configs/bytetrack_ekf_ctrv.yaml`); `pf100` seed 0, 1; `pf300` seed 0; `pf500` seed 0. Các biến thể
  PF dùng yaml PF hiện tại `bytetrack_pf_ctrv_n{100,300,500}.yaml`. Không có `occlusion`.
- Runner: `src/kc48/chay_thi_nghiem_kc48.py` gọi đúng `run_experiment.main` (`--stage track`, rồi `--stage eval`), như
  giao thức 04 mục 5.1. Lý do: thư mục tracker của runner, `data/processed/trackers/`, nằm ngoài `write_paths`.
  Wrapper chỉ chuyển `config.PROCESSED_DIR` sang `results/phase2/kc48_pf_smoke/_processed/` (có `.gitignore` `*`).
  Mọi đường khác giữ như runner gốc. Wrapper từ chối tên không bắt đầu bằng `kc48_` và từ chối thí nghiệm có
  `occlusion`, vì `occlusion_eval` chạy tiến trình con đọc đường cố định.
- GT: sao `gt/gt.txt` + `seqinfo.ini` của 2 video, tổng 4 file, sha256 bản sao trùng bản gốc
  (`results/phase2/kc48_pf_smoke/gt_sao_kiem.json`). `eval_meta.json` tham chiếu của mốc 2 **không có** trường dấu vân
  tay GT (định dạng cũ), nên không đối chiếu được GT với mốc 2; các ô tương ứng ghi `null`. Mọi run smoke chấm trên cùng
  GT: `gt_per_video` `6cd1c901…`, `03ef1add…` trong cả 5 `eval_meta.json`.

### 5.2 Lệnh và exit

| Lệnh | Exit | Ghi chú |
|:--|:--|:--|
| `python src/kc48/chay_thi_nghiem_kc48.py --exp configs/phase2/experiments/kc48_pf_smoke.yaml --stage track` | 0 | 5 lần chạy × 2 video, 29,9 s; thư mục `p2-kc48_pf_smoke-*` trong `data/processed/trackers`: trước 0, sau 0 |
| `... --stage eval` | 0 | TrackEval 5 run, 6,2 s; `[occl] bỏ qua` (không khai báo); trước 0, sau 0 |
| `... --stage thoi-gian` | 0 | `thoi_gian.csv`, `thoi_gian.json` |

Đầu ra:
- `results/phase2/kc48_pf_smoke/`: `experiment.yaml`, `videos.json`, `runs.csv`, `gt_sao_kiem.json`,
  `nhat_ky_wrapper.jsonl`, `thoi_gian.*`, `_processed/` (bị git bỏ qua).
- `results/trackeval/DETRAC-all/p2-kc48_pf_smoke-A-{ekf,pf100-s0,pf100-s1,pf300-s0,pf500-s0}/`.

Theo giao thức 04 mục 10, chỉ số HOTA / IDF1 của smoke **không** dùng cho kết luận nào. Báo cáo này không chép và không
so các chỉ số đó.

### 5.3 Thời gian từng biến thể

Nguồn: `results/phase2/kc48_pf_smoke/thoi_gian.csv` (từ `summary.csv` của runner, giây tường mỗi video). Đo một tiến
trình, một lần, cùng máy, không có tiến trình đo khác chạy song song.

| Biến thể | Lần chạy | Giây track (2 video) | ms / frame (toàn bộ) | predict / assoc / update ms / frame |
|:--|--:|--:|--:|:--|
| `ekf` | 1 | 2,42 | 1,620 | 0,350 / 1,091 / 1,441 |
| `pf100` | 2 (s0, s1) | 5,19 / 5,21 | 3,474 / 3,487 | 1,063–1,072 / 2,240–2,244 / 3,303–3,316 |
| `pf300` | 1 | 7,14 | 4,779 | 1,866 / 2,730 / 4,596 |
| `pf500` | 1 | 8,82 | 5,904 | 2,642 / 3,079 / 5,721 |

Eval: 6,18 s cho 5 run, tức khoảng 1,24 s mỗi run (gồm nạp TrackEval).

### 5.4 Dự toán cho R48D (kịch bản theo giả định, không phải cận trên)

Giả định:
- Thời gian track tỉ lệ tuyến tính với số frame. Dev12 có 16.919 frame (tổng `seqLength`), gấp 11,3 lần smoke.
- Eval mỗi run khoảng 14,0 s trên dev12, theo cùng tỉ lệ.
- Biến thể K (K1, K2, K1K2) có thời gian track bằng c lần PF cùng N, với c ∈ [1, 2]. Hệ số c **chưa đo**: K1 thêm phép
  rút mẫu lại tọa độ đo ở mỗi update, K2 thêm nhiễu sau resample.

| Nhóm (giao thức 04 mục 9) | Nội dung | Kịch bản c = 1 | Kịch bản c = 2 |
|:--|:--|--:|--:|
| P1 | `ekf` + `pf100` × 3 seed + `pf100-k1` × 3 seed | 479 s (8,0 phút) | 656 s (10,9 phút) |
| P2 | `pf100-k2` × 3 + `pf100-k1k2` × 3 + 1 cấu hình EKF đã chỉnh | 479 s (8,0 phút) | 832 s (13,9 phút) |
| P3 | `pf300`, `pf500` × 3 + biến thể K tốt hơn ở N = 300, 500 × 3 | 1.253 s (20,9 phút) | 1.795 s (29,9 phút) |
| Tổng | | ≈ 37 phút | ≈ 55 phút |

Mỗi run dev12, chưa nhân seed: `ekf` 27,4 + 14,0 s; `pf100` 58,9 + 14,0 s; `pf300` 80,9 + 14,0 s; `pf500` 99,9 + 14,0 s
(`thoi_gian.json`). Trong cả hai kịch bản, toàn bộ ma trận P1–P3 nằm trong ngân sách khoảng 120 phút của giao thức.

Chưa tính:
- Bước sự kiện che khuất, tầng xe nhỏ, giữ ID và bootstrap: smoke chưa chạy các bước này.
- Thời gian mô phỏng: ngân sách riêng khoảng 90 phút, R48D sẽ đo.

Một lệnh P1 có thể vượt 10 phút. R48D phải chia lệnh theo cấu hình hoặc dựa vào cơ chế tiếp tục theo từng video của
runner.

## 6. Quan sát phụ: EKF tại `eaf2e87` so với run mốc 2 — **ĐÍNH CHÍNH**

**Đính chính (lời gọi chốt 11/10/2026; chi tiết ở `05b_truy_nguyen_lech_ekf.md`).** Hai file tracking **trùng từng
byte** trên cả 2 video: sha256 thô `42befd96…` và `2d4a45b8…` ở cả hai bên; 0 frame lệch, 0 track lệch. Hai giá trị
khác nhau dưới đây không phản ánh nội dung tracking. Đó là hai **quy ước hash** của `src/run_fingerprint.sha256_file`:
mốc 2 (`1c7ce8c`) hash byte thô, còn từ `935a445` file `.txt` được hash sau khi chuẩn hóa CRLF → LF. Vì vậy câu
"sha256[:16] file tracking EKF khác nhau" và giả thuyết [GT] ở cuối mục này là **sai**. Đoạn gốc giữ nguyên bên dưới
để truy vết.

Bản gốc (lời gọi trước):

[QS] Cùng yaml (`tracker_yaml_sha256_12` `2c35f2da40c5`), cùng `track_buffer_frames 30`, `elapsed_mode skip`, điều kiện
A và cache trùng khít (mục 2). Dù vậy, sha256[:16] file tracking EKF khác nhau:

| Video | Mốc 2 | Smoke |
|:--|:--|:--|
| `MVI_20052` | `42befd96…` | `4bb3dda8…` |
| `MVI_20062` | `2d4a45b8…` | `cf97601f…` |

Nguồn:
- Mốc 2: `results/trackeval/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/eval_meta.json`; run tạo ở commit `1c7ce8c`,
  ngày 2026-09-16 (`data/processed/trackers/DETRAC-all/p2-expB_detrac_dev_fix1-A-bt-ekf/run_meta.json`).
- Smoke: `results/trackeval/DETRAC-all/p2-kc48_pf_smoke-A-ekf/eval_meta.json`.

`git log --oneline 1c7ce8c..HEAD` trên mã tracking (`src/ekf_ctrv.py`, `src/tracker_ctrv.py`, `src/run_tracking.py`,
`src/config.py`, `src/det_cache.py`, `src/time_units.py`, `configs/bytetrack_ekf_ctrv.yaml`) liệt kê 10 commit. Trong đó
có `11603cb` (backend ByteTrack không phụ thuộc torch), `9dcd2d6` (timeline GT) và `5e28f37` (thống nhất đường chạy).

[GT] Nguyên nhân chưa truy. Có thể là thay đổi mã giữa hai commit, hoặc phiên bản thư viện. Hệ quả cho R48D: mọi so
sánh chỉ dùng các run tạo tại **cùng commit**; không trộn với số của mốc 2. Bảng kế thừa trong
`00_tiep_nhan_va_ke_thua.md` vẫn là kết quả của mã thời mốc 2.

## 7. Kết quả cũ không bị đổi

- `git status --short` trên `results/trackeval`, `results/phase2/manifests`, `results/phase2/kc48_pf_smoke`,
  `results/phase2/kc48_pf_code`, `data/processed`, `configs`, `src/pf_ctrv.py`, `src/tracker_ctrv.py`: không có file đã
  track nào bị sửa.
- Mục mới chưa track của R48C: `configs/phase2/experiments/kc48_pf_smoke.yaml`, `results/phase2/kc48_pf_smoke/`,
  `results/phase2/kc48_pf_code/`, 5 thư mục `results/trackeval/DETRAC-all/p2-kc48_pf_smoke-A-*`.
- Các thư mục kết quả cũ chưa track (`p2-expA_*`, `p2-expC*`, `DETRAC-test/*`) đã có từ trước. Runner của R48C chỉ ghi
  `p2-kc48_pf_smoke-*`.
- `data/processed/trackers/DETRAC-all/`: wrapper kiểm trước / sau mỗi bước, không có thư mục `p2-kc48_pf_smoke-*`.
- Lời gọi trước không ghi file nào trong `data/`. Lời gọi chốt 11/10 chỉ ghi 3 thư mục fixture được người dùng cho
  phép (mục 4). `fixture_sau.json` cho thấy 24.164 file còn lại trong `data/interim`, `data/processed` không đổi byte,
  không đổi mtime, không thêm, không mất.
- Truy nguyên 05b chỉ đọc file của run mốc 2. `so_sanh_ekf_moc2.py` chỉ ghi `results/phase2/kc48_ekf_tai_lap/`.

## 8. File của R48C

- Tài liệu:
  - `04_giao_thuc_kiem_chung_pf.md`: khóa; sha256 ở mục 1; có phụ lục P1.
  - `05_dung_dan_pf_va_smoke.md`: file này.
  - `05b_truy_nguyen_lech_ekf.md`.
- Mã, tất cả chỉ đọc dữ liệu gốc:
  - `src/kc48/mo_phong_quy_dao.py`: bộ sinh quỹ đạo GT + phép đo, `--tu-kiem`.
  - `src/kc48/test_pf_dung_dan.py`.
  - `src/kc48/chay_test_pf_cu.py`.
  - `src/kc48/kiem_cache_dev.py`.
  - `src/kc48/ghi_sha256.py`.
  - `src/kc48/chay_thi_nghiem_kc48.py`.
  - `src/kc48/so_sanh_ekf_moc2.py`.
  - `src/kc48/chup_fixture.py`.
  - `src/kc48/ghi_sha256.py` có thêm `--tien-to`.
- Cấu hình: `configs/phase2/experiments/kc48_pf_smoke.yaml`.
- Kết quả:
  - `results/phase2/kc48_pf_code/{giao_thuc_sha256,cache_dev,mo_phong_tu_kiem,fixture_truoc,fixture_sau}.json`.
  - `results/phase2/kc48_ekf_tai_lap/{so_sanh_dong.json, frame_MVI_20052.csv, frame_MVI_20062.csv}`.
  - `results/phase2/kc48_pf_smoke/`.
  - `results/trackeval/DETRAC-all/p2-kc48_pf_smoke-A-*`.

## 9. Giới hạn

- Thời gian đo một lần trên 2 video. Ngoại suy tuyến tính bỏ qua khác biệt mật độ giữa các video. Hệ số chi phí của K1 /
  K2 chưa đo.
- Test 11 đã chạy đủ (mục 4). Phép chạy lại seed qua runner được kiểm trên fixture tổng hợp 40 frame bằng
  `run_tracking.py`, chưa kiểm qua `run_experiment.py` trên video dev.
- Tái tạo EKF trùng byte với mốc 2 mới kiểm 2/12 video dev, điều kiện A, chỉ EKF (05b mục 6).
- Cách evaluator xử lý 572 ảnh dev12 không có `<frame>` vẫn chưa kiểm (giao thức 04 mục 2; đề xuất riêng đang chờ).
- Không đối chiếu được GT với mốc 2 qua `eval_meta.json` (định dạng cũ). Bản sao GT chỉ được chứng minh trùng bản hiện
  có trong `data/processed`.
