# Kế hoạch đã chốt — từ vòng 27 (2026-09-29)

Nguồn: yêu cầu của người dùng sau khi Codex nghiệm thu `239b9c6` (**PASS WITH LIMITATIONS**, `handoff/CODEX_REVIEW_239b9c6.md`).
File này là phạm vi làm việc cho các vòng sau. Thay đổi phạm vi phải do người dùng quyết định.

## 0. TRẠNG THÁI HIỆN TẠI — tài liệu trạng thái chuẩn (cập nhật 2026-10-29)

> Mục này là **nguồn trạng thái duy nhất**. Brief, handoff, REVIEW*_CHI_TIET là lịch sử theo vòng; khi chúng
> mâu thuẫn với mục này thì mục này đúng, trừ khi có commit/bằng chứng mới hơn (khi đó sửa mục này).
> Mỗi vòng xong việc: cập nhật mục 0 trước, rồi mới viết brief/handoff. Lộ trình đọc: `AGENTS.md`.
> Chỉ mục tài liệu cũ và vùng lưu trữ: `handoff/LUU_TRU.md`.

**Lượt hiện tại 29/10/2026: đang thực hiện MỐC 6 — tối ưu tốc độ hệ thống Jetson Nano, mở ngày 28/10.**
Đã đo baseline ngắn và kiểm ứng viên chuyển tensor; đang hoàn tất baseline dài và phép so cặp.
**Chưa nghiệm thu tăng tốc, chưa sửa gói v16, engine, tracker hay cấu hình thí nghiệm**. Baseline lịch sử: pipeline
300 JPG không hiển thị khoảng 9,1–9,4 FPS (`handoff/REVIEW21_CHI_TIET.md`, `REVIEW22_CHI_TIET.md`), 1.470 JPG
vòng 23 CV 9,22/EKF 9,05 FPS nhưng decode lệch mức nên không xếp hạng bộ lọc (`handoff/REVIEW23_CHI_TIET.md`);
vòng 25 bật hiển thị giảm khoảng 2% trên 300 ảnh (`handoff/REVIEW25_CHI_TIET.md`). Gói triển khai theo hồ sơ v16 ở
`~/nano_run/v16` (`handoff/REVIEW26_CHI_TIET.md`). Kế hoạch và tiêu chí mốc 6: `handoff/logs/review62/01_ke_hoach_moc6.md`.
**Tiền kiểm R62 — cập nhật sau đổi Jetson:** người dùng xác nhận đã đổi Jetson và dùng lại thẻ SD cũ.
SSH key hoạt động; IP mới `192.168.1.4` (alias `nano` đã cập nhật), root `/dev/mmcblk1p1`, thẻ 58,2 GiB UUID
`e63fc90f-5700-4a13-a49d-43dc08fead24`, còn khoảng 27,8 GB; v16, engine 12.626.138 byte, bộ ảnh và Python
`~/miniforge3/envs/nano/bin/python` đều tồn tại. TensorRT 8.2.1.9, CUDA 10.2, OpenCV 3.2.0. Phát hiện trước
đó boot eMMC/không nhận SD đã được trạng thái này thay thế. SSH đã nối lại sau lần khởi động lại;
engine SHA256 `c32690cd…` khớp lịch sử, manifest v16 đạt 562/562, Python env 3.8.20, HDMI hoạt động,
ext4 errors 0 (`handoff/logs/review62/08_tien_kiem_v16.json`). Đã tạo bản sao v16 562 file khớp manifest ở
`~/nano_run/r62_20261028/baseline/nano_transfer`; **baseline 300 ảnh hoàn tất 8/8 lượt**, đủ frame, live/replay
MOT+lớp trùng byte, detection cả 8 lượt giống nhau. FPS 8,59–9,41; decode vẫn hai mức nên không dùng để xếp hạng
CV/EKF. Lượt đầu hiển thị 299/300 (tracking đủ), các lượt bật còn lại không bỏ frame xem trước. Hồ sơ
`handoff/logs/review62/10_baseline300_tong_hop.json`, `12_kiem_detection_baseline300.json`; raw sao lưu ở
`baseline300_evidence.tar.gz` cùng thư mục. 1.470 JPG khớp hash (`13_kiem_1470.log`). Lượt baseline dài đầu
`baseline1470/01_cv_on` bị gián đoạn do Nano khởi động lại: chỉ còn ba file rỗng, chưa có kết quả; giữ nguyên
để ghi nhận, không tính là lượt đạt. Log `15_gian_doan_baseline_dai.json` ghi ext4 phục hồi 14 orphan inode,
errors_count 0. Người dùng xác nhận máy tự treo/tắt rồi yêu cầu tiếp tục. Pstore cho thấy RCU/softlockup,
I2C timeout và `Kernel panic - not syncing: softlockup: hung tasks` (`22_dmesg_ramoops.txt`, `23_console_ramoops.txt`);
chưa xác định căn nguyên. Lượt chẩn đoán headless `chan_doan300/lan02` đã hoàn tất 300/300, 8,889 FPS,
cùng boot `f6f8c913-61cf-4b45-af91-ead44ca14f6b`; replay/MOT/lớp đã trùng byte (`26_chan_doan300_parity.json`).
Baseline dài mới `baseline1470_lan02` **4/4 hoàn tất** trên cùng boot, mỗi lượt 1470/1470 frame,
live/replay byte đạt, runner exit 0: CV 8,275/8,297 FPS, EKF 8,294/8,288 FPS. Lượt lỗi cũ giữ nguyên.
kiểm `baseline1470_lan02.runner.log`, `exit_code` và từng `result.json` trước tiếp tục. Lần chẩn đoán `lan01` lỗi
thiếu biến `BYTETRACK_BACKEND=np` trước khi chạy, không phải crash Nano; giữ log 24/25. Nếu tái diễn panic,
dừng phép đo GPU để xử lý nguyên nhân, không lặp mù. **Ứng viên đã sửa trên PC:** chuyển tensor
bằng bảng tra float32 qua OpenCV, nhánh khác uint8 giữ hành vi cũ; quyết định `17_quyet_dinh_ung_vien.md`.
Test detector PC đạt 7/7, cú pháp Python 3.6 đạt; mã production kiểm CPU trên Nano đạt 11 trường hợp trùng byte
với đường cũ (`18_test_nano_detector_pc_lan02.log`, `19_test_production_tensor_nano.json`). Microbenchmark
chuyển tensor giảm 9,78 xuống 2,40 ms; **chưa có kết luận tăng tốc toàn pipeline**, chưa đổi launcher v16.
Gói ứng viên thử nghiệm trên PC đã dựng từ ZIP v16, kiểm 562 hash, chỉ `src/nano_detector.py` và
`PACKAGE_INFO.json` đổi nội dung, cập nhật manifest; ZIP SHA256 `e1e155b0e37316e341343e502e48d329ae363f073cf57dfc83c90941e80d6bd1`
(`27_goi_ung_vien.json`). Đã chuyển vào `/home/jetson/nano_run/r62_20261028/candidate/nano_transfer` và
kiểm lại ZIP, 562 manifest hash và đúng hai file nội dung khác v16 (`candidate_verified.json`).
Đã bắt đầu runner 16 lượt paired300 (4 cặp CV, 4 cặp EKF, bật hiển thị), PID 8984, cùng boot; lịch
luân phiên v16/ứng viên, mỗi lượt thư mục mới và nghỉ 20 s. Đọc `paired300.runner.log`, `paired300/index.json`,
`paired300/runner_exit.json` trước khi tiếp tục; không chạy trùng nếu còn sống. Cặp CV thứ nhất đạt
parity detection/MOT/lớp trùng byte, FPS v16 9,3034 → ứng viên 9,6565 (+3,79%); đây là một cặp,
**chưa kết luận đạt tiêu chí trung vị 4 cặp và chưa đo xong EKF**.
Phải đo lại baseline v16 trên Jetson hiện tại để so tối ưu; không quy chênh lệch với FPS máy cũ cho sửa mã.
Chỉ ghi bản sao và kết quả R62; chưa sửa boot/mount/cài gói hoặc mã thuật toán.
Gói v16 `handoff/nano_transfer_latest.zip` còn nguyên SHA256 `302f6af73d97aea009066416aaaa202db81311e60121676a75dacbe204e9cbf1`;
engine đã sao lưu về `handoff/archive/nano_engine_r62_c32690cd.engine`, hash nguyên bản khớp. Bằng chứng
và bước tiếp: `handoff/logs/review62/05_tien_kiem_va_khoi_phuc.md`.
**Tiếp tục tự động (người dùng cho phép 29/10, thay yêu cầu không automation trước đây):** đã tạo lịch mỗi giờ
`ti-p-t-c-m-c-6-jetson-nano`, ACTIVE, gắn chat `01a17bcd-4100-79f1-91aa-ba26408cd61f`.
Lịch đọc điểm bàn giao `D:/LABOTARY/automation/checkpoints/nano_moc6_20261029/TIEP_TUC.md`, kiểm tiến trình/kết quả
trước khi chạy tiếp, không mở orchestrator CLI riêng. Đây là kiểm theo lịch, không phải sự kiện reset quota;
cần Windows bật và ứng dụng đang chạy. Không dùng reset credits/mua credits. Tắt lịch khi mốc 6 hoàn tất.

**(Lịch sử tại 27/10/2026) MỐC 5 HOÀN TẤT PHẦN THỰC THI VÀ BÁO CÁO; mốc 6 (tối ưu Nano) CHƯA BẮT ĐẦU ở thời điểm đó.**
R61 cổng A đã đóng gói và được review cuối `review_009.json` approve: gói A SHA256
`5c9db320e4965798f50652b823b7a20439410e259752dbf72ebfe990d30fe225`, khóa A SHA256
`f3a138d7497727a205abfa78237f0ec9a6029b89a13bdd018928865351794d4f` (báo cáo
`handoff/logs/review61/03_bang_chung_cong_a.md` mục 12). Gói chuẩn bị B được `review_009.json` của lượt
`r61_bdd_prep_b_20261026_20261026_132958` approve (`handoff/logs/review61/04_chuan_bi_cong_b.md`). Cổng B thật
đã chạy trong phiên `results/phase2/r61_bdd/thuc_thi/cong_b_20261027_01/`: 600/600 cache,
7.200/7.200 tracking, 36/36 chấm, 36/36 lặp tracker và 6/6 chấm lại trùng byte, 108 CI từ 5.000 × 200 mẫu;
`kiem-khoa` và `hau_kiem.json` đều `dat=true`, 0 crash mới. Báo cáo cuối hai tập tách riêng:
`handoff/logs/review61/06_ket_qua_thuc_thi_cong_b.md`; UA-DETRAC test40 dùng bằng chứng R53/R56/R57/R59
đã nghiệm thu, không chạy lại ở R61.

**Chuyển khóa sau B:** trước khi sửa mục 0, SHA256 toàn file PLAN là
`2b2474aabd3215deb3973d5fa342457d09db422df5b537cbe238d3fe6d93e12a`, đúng ảnh PLAN trong khóa A và
tiền/hậu kiểm B. Người dùng đã yêu cầu 27/10 hoàn tất phần đóng mốc 5 để ngày 28/10 có thể bắt đầu mốc 6. PLAN
được cập nhật **sau** khi B `hau-kiem` exit 0; đây là trạng thái kế tiếp, không phải thay đổi dữ liệu đã khóa của
phiên A/B. Mọi phép kiểm lịch sử B cần dùng ảnh PLAN cũ (có trong commit GitHub `4f8131b`), không chạy lại runner B
trên PLAN mới. Đối chiếu thủ công sau B độc lập với kết luận runner: băm lại gói/khóa A, phê duyệt và hai CSV tổng hợp;
kiểm 32 exit, 36/36 + 6/6 cờ trùng byte, 108/108 hiệu số và cờ CI từ số CSV, tất cả khớp. Đây không phải một
lượt review của tác nhân thứ hai hoặc review tự động. Bằng chứng đóng mốc 5: `handoff/logs/review61/07_dong_moc5.md`.

**(Lịch sử 27/10) Bước tiếp:** đóng hồ sơ và đẩy snapshot mốc 5 lên GitHub (đã xong, commit `09d3dd5`);
sau đó tạm dừng theo quyết định người dùng 23/10. Người dùng đã mở lại mốc 6 ngày 28/10; trạng thái hiện hành ở trên.

**(Lịch sử tại 25/10; đã được trạng thái 27/10 phía trên thay thế) Lượt R61 cổng A (plan A0→A1→A2 người dùng duyệt 24/10/2026) — A0 APPROVE
(`D:/LABOTARY/automation/runs_local/r61_bdd_cong_a_20261024_20261024_182327/reports/review_003.json`, sau review_001/002
`revise`); A1 (bước gộp Pe27e304a, tiếp tục theo Pd8810cda) **APPROVE** (`.../reports/review_007.json`: tự đếm
600 marker cache/39.973 frame mỗi detector, 108 tracking + 36 chấm, tự băm 1.803 file cache khớp kiem-cache, cache
không đổi qua smoke; giới hạn: review tĩnh, approve chỉ A1); **A2 ĐANG BẮT ĐẦU (25/10) — khoa/kiem-khoa/dong-goi-a/
kiem-goi-a CHƯA chạy tại thời điểm ghi dòng này; chưa có `khoa_sha256`/`goi_a_sha256`.** Trước đó: R61_PREP APPROVE
(`review_005.json`), bước tổng hợp crash APPROVE (`review_006.json`). Approve A không mở cổng B.**
- **A2 — trạng thái bàn giao CÓ ĐIỀU KIỆN (ghi trước khóa cuối; từ đây PLAN và mọi file tracked/mã/cấu hình ĐÓNG BĂNG,
  PLAN vẫn nằm trong `git_dirty_sha256` và phụ lục khóa):** dữ liệu đầu vào của khóa = cache/smoke A1 đã xác minh ở dòng
  dưới (600 marker cache, 108 tracking + 36 chấm, 13 lần chạy, crash 1/4 chỉ `cha_gian_doan` lần 0010). Cùng phiên A,
  mỗi lời gọi `--so-doan 1 --ngan-sach-giay 420`, thứ tự `--buoc khoa` → `kiem-khoa` → `dong-goi-a` → `kiem-goi-a`. Đường
  dẫn dự kiến (trong `results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/`): khóa `khoa/khoa.json` + `khoa/khoa.xong.json`;
  kiểm khóa `kiem_khoa/vong_NN/ket_qua.json`; gói `goi_a.json`, `SHA256SUMS`, `goi_a.xong.json`, bản chốt crash
  `goi/nhat_ky_crash_chot.json`; kiểm gói (chỉ đọc payload) `van_hanh/kiem_goi_a/vong_NN/ket_qua.json`. A chỉ được coi là
  bàn giao khi cả bốn bước exit 0, kiểm khóa và kiểm gói `dat: true`, VÀ Codex review cuối approve với `goi_a_sha256` +
  `khoa_sha256` cụ thể. Kết quả thực tế sau khóa chỉ ghi vào báo cáo `handoff/logs/review61/03_bang_chung_cong_a.md`
  mục 12 (không tracked) và bằng chứng trong A — **mục 0 này không được sửa sau khóa**, nên không phản ánh kết quả A2;
  đọc báo cáo 03 mục 12.
- **Gỡ dừng lần 0010 (Pd8810cda, 25/10, phiên điều phối theo ủy quyền người dùng 24/10):** phiên điều phối CHUYỂN
  `van_hanh/dung.json` nguyên byte sang `van_hanh/dung_da_go/dung_lan_0010.json`; `van_hanh/dung_da_go/quyet_dinh_go_dung.json`
  (luc 04:59) ghi sha256 trước = sau = `adad5985752806b5630ab9d753be15a7ce74c75e3096d15a321d69bb7fc671a1`, 246 byte, lý do
  "orchestrator ngắt lời gọi Claude 0012 khi quá timeout 3600 s; con pid 7400 đã kết thúc; không phải crash native",
  điều kiện "mọi dừng mới vẫn phải hỏi". Agent đã đọc: nội dung `dung_lan_0010.json` (stt 10, luc 04:30:10, 2 lý do) khớp
  dòng `goi_cha.jsonl` exit 40 lúc 04:30:10; agent KHÔNG tự băm lại được (không có công cụ băm trong quyền) — sha256 là số
  do phiên điều phối ghi. `so_crash.json` giữ tổng 1/4 (bước cache 1, đơn vị không xác định), `0010/ket_thuc.json` giữ
  `cha_gian_doan`; không reset bộ đếm, không mở phiên mới.
- **A1 kết quả (25/10, cùng phiên A, `--so-doan 1 --ngan-sach-giay 420`, 3 lệnh runner, không dừng mới, không crash
  mới):** lần 0011 `cache yolo11n_mixed5` exit 0 (18 đơn vị mới + 182 bỏ qua, 115,1 s; không có artifact dở cần sao lưu —
  `van_hanh/artifact_do/` trống); lần 0012 `kiem-cache` exit 0 (`kiem_cache/ket_qua.json` `dat: true`, 600/600, mỗi
  detector 200 video, 39.973 frame; kiểm khóa/dtype npz, timeline frame_idx/time_s, unreadable/truncated = 0, ánh xạ
  cls_goc, provenance, spec, gen_spec); lần 0013 `smoke` exit 0 (`smoke/ket_qua_smoke.json` `dat: true`, 108 tracking +
  36 chấm trên 3 video, n_frames = seqLength 203/202/202, seed/tham số bộ lọc khớp, npz của 3 video smoke khớp hash
  kiem-cache). `so_crash.json`: 13 lần chạy, tổng crash 1/4 (chỉ lần 0010 `cha_gian_doan`), 0 crash native. Thời gian suy
  luận/frame: yolov8n_coco 23,07 ms, yolov8n_mixed5 22,93 ms, yolo11n_mixed5 26,74 ms; cache 49.228.201 byte (npz+json).
  Đối chiếu toàn bộ 600 cache với kiem-cache sẽ làm ở bước `khoa` (A2). Báo cáo `handoff/logs/review61/03_bang_chung_cong_a.md`
  mục 11. Không còn tiến trình chạy.
- **(Lịch sử, đã gỡ theo Pd8810cda) A1 dừng hỏi (25/10 04:30, phiên `results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/`):** lời gọi Claude trước bị
  orchestrator ngắt khi lần chạy `0010_cache_yolo11n_mixed5` (cha khởi động 03:36) đang làm đơn vị
  `cache:yolo11n_mixed5:b24d283f-33783d1b` → không có `ket_thuc.json`. Lời gọi kế tiếp (đúng lệnh cache yolo11n_mixed5)
  để runner tự ghi: `phan_loai = cha_gian_doan`, con pid 7400 xác nhận đã kết thúc (pid + thời điểm tạo), tính 1 crash
  (`van_hanh/so_crash.json`: tổng 1/4, bước cache 1, đơn vị: không xác định) → theo thiết kế A0 đã duyệt "cha gián đoạn
  → dừng hỏi", exit 40, ghi `van_hanh/dung.json`; mọi lời gọi sau trả 40. Agent KHÔNG xóa `dung.json`, không mở phiên
  mới. **Cần người dùng quyết định** cách tiếp tục phiên A. Tiến độ đã có (chưa kiểm `kiem-cache`): `tien-kiem` exit 0;
  cache yolov8n_coco hoàn tất (lần 0002–0004, exit 10,10,0), yolov8n_mixed5 hoàn tất 200 marker (0005–0007),
  yolo11n_mixed5 182/200 marker (0008–0010); chưa `kiem-cache`, chưa smoke. Không còn tiến trình chạy.
- **Ngoại lệ 3 mục ĐÃ DUYỆT (Pe27e304a, 24/10, thay Pbec787ae):** danh sách đọc từ `configs/phase2/r61_bdd/hop_dong_lenh.json`
  (đường dẫn + loại chênh lệch), runner áp đúng danh sách; fixture mới `results/phase2/r61_bdd/tests/cong_a_20261024/lan_02/vong_02/ket_qua_test.json`
  `dat: true`, 25 kịch bản (có ca chống lan ngoại lệ). Tiền kiểm A thật `tien_kiem_a/so_sanh.json` `dat: true`, ngoại lệ
  đúng `.git` (mtime) + 3 mục duyệt (pyc `src/__pycache__/baseline_track.cpython-311.pyc` sha256 `9aec6dab…`, mtime thư mục
  `src/__pycache__`, mtime thư mục `results/track_video`), ghi riêng kèm lỗi thô. Hai dòng dưới là lịch sử trước duyệt.
- **(Lịch sử) A1 blocked (25/10, lời gọi A1 đầu tiên):** 3 chênh lệch baseline dưới đây vẫn còn (pyc
  `src/__pycache__/baseline_track.cpython-311.pyc` còn tồn tại; `git status` không đổi) và không có ngoại lệ được duyệt.
  `van_hanh_cong_a.py:1438–1440` ghi `van_hanh/phien.json` (ghim snapshot mã/hợp đồng/giao thức) là file ĐẦU TIÊN của phiên
  A cố định `cong_a_20261024_01`, trước đối chiếu baseline → chạy `tien-kiem` lúc này chắc chắn exit 2 và để lại phiên A
  đã ghim; hai mtime thư mục không thể trở về giá trị baseline, nên mọi cách xử lý cần ngoại lệ/sửa mã → snapshot lệch
  (`van_hanh_cong_a.py:1478–1481`, exit 2) mà plan cấm thay phiên. Vì vậy agent KHÔNG chạy lệnh nào của A1. Cần người
  dùng quyết định cách xử lý 3 chênh lệch (agent không xóa/miễn kiểm). Báo cáo: `handoff/logs/review61/03_bang_chung_cong_a.md` mục 9.
- **(Lịch sử, đã thay bởi Pe27e304a) 3 chênh lệch so với baseline lan_01 — CHƯA có quyết định xử lý (đính chính A0-F10, 24/10):** đề xuất Pbec787ae
  ("người dùng chọn cách xử lý trước A1") bị người dùng **reject**; đề xuất P5d658ee9 (thêm ca tiêm lỗi) cũng **reject**.
  Ghi chú trước đây ở chỗ này viết Pbec787ae là "CHẤP NHẬN là ngoại lệ ghi rõ cho tiền kiểm A1" là **SAI**, trái quyết
  định hiện hành — đã thay bằng dòng này; KHÔNG được dùng làm quyền miễn kiểm hoặc mở A1. Hiện **không có ngoại lệ nào
  ngoài mtime thư mục `.git`**; 3 mục (1) `src/__pycache__/baseline_track.cpython-311.pyc`, (2) mtime thư mục
  `src/__pycache__`, (3) mtime thư mục `results/track_video` vẫn là lỗi chặn `tien-kiem` A1 (runner sẽ exit 2); agent không
  xóa/sửa/miễn kiểm. Thông tin nguồn gốc (do phiên điều phối ghi, chỉ để tham khảo): (1)(2) 24/10 11:16:15 khi người dùng
  chạy thử `src/track_video_file.py` không `-B`; (3) 16:35 khi phiên điều phối chuyển output live ra
  `D:/LABOTARY/automation/runs_local/file_ngoai_repo_20261024/`.
- **A0 (24/10, plan cổng A người dùng duyệt; plan do Codex đề xuất sau lượt `bytetrack_r61_bdd_prep_20261023_2215`):**
  - **review_006 approve:** `D:/LABOTARY/automation/runs_local/bytetrack_r61_bdd_prep_20261023_2215/reports/review_006.json`
    verdict `approve` — đóng bước tổng hợp crash (báo cáo 02 mục 11: 6/14 phiên fixture exit 139, nguyên nhân CHƯA RÕ).
  - **Chính sách crash cổng A (ghi chú người dùng trong plan đã duyệt — thay cho khuyến nghị mục 11, vốn chỉ là đề
    xuất):** tối đa 2 thử lại mỗi đơn vị dở; DỪNG HỎI khi một đơn vị crash lần 3, tổng crash phiên đạt 4, hoặc cùng bước
    `khoa`/`kiem-khoa`/`dong-goi-a` crash lần 2; không reset bộ đếm bằng phiên/đoạn mới hay đổi tên đơn vị. Marker hỏng,
    hash/spec/provenance lệch luôn dừng (exit 2, không thử lại); exit 1 có traceback, hết giờ, thiếu log không tự coi là
    crash native. Phiên A thật cố định `results/phase2/r61_bdd/chuan_bi/cong_a_20261024_01/`.
  - **Đã làm A0 (CPU, không GPU/dữ liệu thật):** `src/r61_bdd/van_hanh_cong_a.py` (runner chỉ cổng A: cha thư viện chuẩn
    → con tuần tự gọi helper theo đơn vị; `goi.json` trước mỗi lần chạy; stdout/stderr/faulthandler riêng; nhật ký đơn
    vị fsync; bộ đếm crash tính lại từ mọi `ket_thuc.json`; exit 0/10/2/30 crash dưới ngưỡng/40 dừng hỏi; artifact dở
    chuyển nguyên byte vào `van_hanh/artifact_do/`; tiền kiểm A chụp PLAN + so trực tiếp baseline lan_01, ngoại lệ duy
    nhất mtime thư mục `.git`; khóa = `anh_khoa` + phụ lục vận hành (PLAN nguyên byte, hợp đồng, mã r61); băm/kiểm theo
    đoạn tiếp tục được; gói có marker `goi_a.xong.json`), `test_van_hanh_cong_a.py`; sửa `dieu_phoi_bdd.py` chỉ
    `DIEM_VAO_MA` (+ runner → khóa bao mã runner); `hop_dong_lenh.json` thêm mục `van_hanh_cong_a`. Helper, `giao_thuc.json`
    giữ nguyên byte.
  - **Fixture vòng đầu (lịch sử, mã cũ runner `5569b7b1…`)** ở gốc `results/phase2/r61_bdd/tests/cong_a_20261024/lan_01/`:
    exit 0, 65/65, 17 kịch bản, 52 lệnh. Codex review_001 (lượt `r61_bdd_cong_a_20261024_20261024_182327`) trả `revise`
    với A0-F1…F10 → kết quả này **không** chứng nhận mã hiện tại.
  - **Sửa A0-F1…F9 (24/10, chỉ trong write_paths A0; helper `hop_dong_bdd`/`chuan_bi_bdd`, `giao_thuc.json`,
    `dieu_phoi_bdd.py` vòng này KHÔNG sửa):** F1 khóa/đóng gói kiểm đủ marker+hash tracking/chấm smoke; F2 gói dở được sao
    lưu cùng vùng băm cũ TRƯỚC khi lập danh mục mới (bản sao lưu nằm trong manifest), kiểm gói tìm file thừa trên toàn
    phạm vi payload; F3 thiếu log → dừng bất kể exit; F4 con gắn Job Object kill-on-close, `con.json` (pid + thời điểm
    tạo), lần sau xác nhận con đã chết rồi dừng; F5 `--con`/`--phien` xác thực đầy đủ trước mọi ghi, `goi_cha.jsonl` chỉ
    ghi cho phiên đã chấp nhận; F6 kết quả kiểm/băm sao lưu bản cũ + `.dang_ghi` trước khi ghi, không ghi đè; F7 đối chiếu
    nội dung `git diff` từng file và index (`git diff --cached`); F8 fixture chạy theo vòng `vong_NN/` gắn snapshot mã thực
    chạy; F9 mỗi lời gọi đúng một đoạn (`--so-doan` ≠ 1 → exit 2). `hop_dong_lenh.json` thêm điều khoản tương ứng.
  - **Fixture mã mới** `results/phase2/r61_bdd/tests/cong_a_20261024/lan_01/vong_02/`: **exit 0, 104/104 kiểm đạt**, 20 kịch
    bản, 63 lệnh runner (exit 0/10/2/30/40 = 18/3/19/8/13, 2 lệnh cha crash tiêm `0xC0000005`), một lời gọi; snapshot vòng
    `840cb720…` = snapshot lúc kết thúc (runner `b1ef3409…`, fixture `0dbf2b01…`, `hop_dong_lenh.json` `250a72a1…`). Gọi lại
    đúng lệnh → chọn lại `vong_02`, không chạy lại kịch bản. Crash tiêm là access violation thật, chỉ giao thức fixture.
  - **Phát hiện cho A1 (thông tin, không thuộc tiêu chí A0):** đối chiếu chỉ đọc trạng thái THẬT hiện tại với baseline
    lan_01 (`lan_01/thong_tin/doi_chieu_that_lan_01.json`) **không đạt**: ngoài ngoại lệ `.git` còn 3 lỗi ngoài phạm vi —
    `pyc moi: src/__pycache__/baseline_track.cpython-311.pyc`, mtime thư mục `src/__pycache__` và `results/track_video`
    đổi. Đã có trước khi fixture chạy (trạng thái trước/sau fixture trùng). Vòng `vong_02` cho cùng 3 lỗi đó và không thêm
    lỗi nội dung `git diff`/index (`vong_02/thong_tin/doi_chieu_that_lan_01.json`). Nếu giữ nguyên, `tien-kiem` A1 sẽ exit
    2 → **cần người dùng quyết định** trước A1 (xem dòng đính chính A0-F10 ở trên); agent không xóa/sửa/miễn kiểm.
  - **Vòng sửa A0-F2a/F2b/F4/F6 (25/10, theo review_002 `revise`; chỉ `van_hanh_cong_a.py`, `test_van_hanh_cong_a.py`,
    `hop_dong_lenh.json`, mục 0 này, báo cáo 03; helper/`giao_thuc.json`/`dieu_phoi_bdd.py`/baseline/bằng chứng cũ KHÔNG
    sửa):** F2a dong-goi-a sau bước gói dở đặt sao lưu phục hồi (danh mục/đoạn băm/marker dở) vào `dong_goi/phuc_hoi/`
    — ngoài `payload()`, danh mục cố định không đổi, `goi_a.json` băm riêng từng file phục hồi (`phuc_hoi_dong_goi`), kiểm
    gói tìm thừa cả vùng này; F2b kiem-goi-a và kiem-khoa sao lưu phục hồi vào `<vòng>/phuc_hoi/` (sau gói → ngoài payload);
    F4 cha tự gắn Job Object kill-on-close TRƯỚC khi tạo con (con kế thừa job ngay lúc tạo), con tạo `CREATE_SUSPENDED`,
    cha xác nhận `IsProcessInJob` + ghi `con.json` rồi mới `NtResumeProcess`; F6 mọi marker do lớp vận hành ghi sao lưu
    `<marker>.dang_ghi` trước khi helper mở lại `wb`; gói dở tính cả `goi_a.xong.json.dang_ghi`. Điểm tiêm mới chỉ fixture:
    `dang_ghi` (nửa bytes vào `.dang_ghi` rồi access violation), `cha_sau_tao_con`.
  - **Fixture mã mới** `results/phase2/r61_bdd/tests/cong_a_20261024/lan_01/vong_03/`: **122/122 kiểm đạt**, 24 kịch bản,
    82 lệnh runner (exit 0/10/2/30/40 = 31/3/19/12/14, 3 lệnh cha crash tiêm `0xC0000005`); lời gọi 1 exit 10 sau 21 kịch
    bản (đúng thiết kế), lời gọi 2 cùng lệnh tiếp tục → exit 0; `kich_ban_sai_snapshot = []`; snapshot vòng = hiện tại
    `d6c355a2…` (runner `65393cbb…`, fixture `e433d855…`, `hop_dong_lenh.json` `94e29294…`). Kết quả `vong_02` là lịch sử,
    không chứng nhận mã này. Ba chênh lệch baseline vẫn nguyên, không xử lý/miễn.
  - **Bước tiếp (cập nhật 25/10 trước khóa A2):** ~~Codex review A1~~ đã approve (`review_007.json`). Đang chạy A2 theo
    dòng "A2 — trạng thái bàn giao có điều kiện" đầu mục 0; sau A2 Codex review gói A rồi DỪNG (không mở B). Lịch sử:
    chỉ sau review A1 chấp nhận mới chạy A2 (`khoa` → `kiem-khoa` → `dong-goi-a` → `kiem-goi-a`, cùng phiên A); A2 đóng băng PLAN và
    file tracked. *(Trước đó: ~~Codex review lại A0~~ đã approve `review_003.json`; A1 đã mở theo Pe27e304a.)* Chưa commit.
- **Vòng P2e430e0b (24/10; người dùng: "tiếp tục chạy automation để hoàn thiện mốc 5"):**
  - **R61_PREP approve:** `D:/LABOTARY/automation/runs_local/bytetrack_r61_bdd_prep_20261023_2215/reports/review_005.json`
    verdict `approve`, nghiệm thu gói chuẩn bị với giới hạn crash (quyết định (a) P3bf6db84). `chup-cuoi`
    `chuan_bi_cpu/lan_05`: `dat=false`/exit 2, lỗi duy nhất mtime thư mục `.git` (ngoại lệ đã duyệt, không gọi là exit 0).
  - **Bước 2 (P44564acd) thu hẹp:** chỉ tổng hợp bằng chứng có sẵn trong `results/phase2/r61_bdd/tests/` và
    `chuan_bi_cpu/` vào mục mới "Tổng hợp crash native" (báo cáo 02 mục 11); **không** chạy thêm phiên fixture, không sửa
    mã/helper/giao thức/bằng chứng cũ, không đọc event log (nguồn event log vẫn là phiên điều phối). Tổng: 14 phiên
    fixture `tests/lan_01…14`, **6 crash native exit 139** (lan_03, 05, 06, 07, 08, 13 → 6/14); exit 0 ở lan_04 (69/69),
    lan_12 và lan_14 (121/121); lan_09–11 exit 2 (119/120, không crash); lan_01/02 exit 1 (lỗi mã đã sửa, không crash).
    Nguyên nhân crash **CHƯA RÕ**. Khuyến nghị xử lý crash cho plan cổng A/B và đề nghị chạy lại `mdsched` trước cổng B
    **chỉ là đề xuất**, chưa được duyệt.
  - **Quyết định người dùng 23/10:** xong mốc 5 thì **tạm dừng**; **mốc 6 (Nano) chưa làm**, không lập/chạy plan mốc 6
    khi chưa được người dùng bảo.
  - **Báo cáo của phiên điều phối cho người dùng:** `BAO_CAO_CAP_NHAT_CHO_SLIDE_20261024_1020.md` (kiểm 24/10 10:15–10:20)
    — do phiên điều phối viết, agent **không sửa**. Bước giao nêu đường `handoff/BAO_CAO_CAP_NHAT_CHO_SLIDE_20261024_1020.md`;
    lúc vòng này đọc, file **không có** ở `handoff/`, bản hiện có ở
    `D:/LABOTARY/automation/runs_local/file_ngoai_repo_20261024/BAO_CAO_CAP_NHAT_CHO_SLIDE_20261024_1020.md`.
- **Vòng P3bf6db84 (24/10, quyết định (a) của người dùng trong chat ~08:15; dòng tiêu đề trước: "vòng P3bf6db84 … chỉ cập
  nhật tài liệu + `chup-cuoi` mới, chờ Codex review gói chuẩn bị (báo cáo 02 mục 10)" — đã approve ở review_005):** chấp nhận lan_12 + lan_14 (121/121,
  exit 0) và bằng chứng lan_13 (đủ 65 lệnh con hoàn tất, nhất quán 19/5/41, không `git_dirty_sha256`; tiến trình cha
  crash access violation ở bước băm toàn phiên `bang_hash_thu_muc`, `lan_13/faulthandler.log`) là **giới hạn được người
  dùng chấp nhận**, không phải "3/3 exit 0". Không chạy thêm phiên fixture, không sửa `src/`, helper, giao thức, thuật
  toán, metric hay bằng chứng cũ.
  - **Các phiên crash native (exit 139) đã biết:** fixture `tests/lan_03` (vòng đầu, in-process; sau 1 kiểm không đạt,
    không có faulthandler); `lan_05`, `lan_06`, `lan_07`, `lan_08` (vòng F1…F7, in-process; lan_05/06 không có
    `faulthandler.log` nên không ghi được vị trí; lan_07 tại `pathlib._iterate_directories`, lan_08 tại `hashlib` đọc
    file trong `sha256_file`); `lan_13` (subprocess hóa; tiến
    trình cha tại `sha256_file` → `f.read` ← `bang_hash_thu_muc`). Vị trí khác nhau (pathlib, hashlib, đọc file).
    **Nguyên nhân CHƯA RÕ.**
  - **Môi trường/log (nguồn: phiên điều phối, không phải agent):** Smart App Control tắt từ 23/10 ~12:00; Application
    log không có APPCRASH python.exe; CodeIntegrity/Operational không có sự kiện. Windows Memory Diagnostic được lên
    lịch 24/10 05:30, máy khởi động lại 06:03 nhưng **không có bản ghi kết quả** (MemoryDiagnostics-Results) → **chưa
    loại trừ được lỗi phần cứng/RAM**. Thiếu sự kiện không dùng để suy nguyên nhân.
  - **Rủi ro cho cổng A/B (chạy dài):** crash native ngẫu nhiên có thể làm dừng tiến trình giữa chừng. Thiết kế tiếp tục
    theo đơn vị (cache detector-video, tracking run-video) và mã thoát (0/10/2; crash → mã khác, ví dụ 139) phải phát
    hiện được; đơn vị dở **không** được coi là hoàn tất (chỉ marker hợp lệ + hash artifact mới tính). **Plan cổng A và
    plan cổng B phải nêu rõ cách xử lý crash** (phát hiện, chạy lại đơn vị dở, giới hạn số lần, khi nào dừng hỏi).
  - `chup-cuoi` sau tài liệu: phiên mới `chuan_bi_cpu/lan_05`, `--phien-dau` baseline gốc `chuan_bi_cpu/lan_01`; áp
    ngoại lệ mtime thư mục `.git` đã duyệt (Pa6af1bda); kết quả nêu trong báo cáo JSON của lượt.
- **Vòng Pa6af1bda (24/10, BLOCKED — lịch sử; dòng tiêu đề trước: "BLOCKED ở vòng Pa6af1bda … chờ người dùng quyết
  định"):** chỉ sửa `src/r61_bdd/test_chuan_bi_bdd.py`: mọi kiểm âm qua tiến trình con dùng
  `dung_ly_do` — exit 2 **và** đúng một dòng `[LOI HOP DONG]` trong stderr chứa nguyên nhân dự kiến; thông điệp dạng
  danh sách ("trang thai hien tai khac khoa: […]", "hau-kiem: […]") phải bằng **từng mục** danh sách kỳ vọng (thừa mục
  `git_dirty_sha256` PLAN → không đạt); F6 hau-kiem còn đối chiếu artifact `hau_kiem.json` (dat/loi); kiểm toàn phiên mới
  `khong_loi_khoa_git_dirty_ngoai_du_kien`. Không sửa file tracked trong lúc ba phiên chạy (PLAN chỉ sửa sau lan_14).
  Kết quả: **lan_12** exit 0, 121/121, 65 lệnh con (exit 0/10/2 = 19/5/41), 362,8 s; **lan_14** exit 0, 121/121, 65 lệnh
  con 19/5/41, 343,7 s; **lan_13** exit 139: đủ 65 lệnh con (19/5/41, không lệnh con nào báo `git_dirty_sha256`), tiến
  trình cha crash khi bấm hash toàn phiên (`faulthandler.log`: access violation trong `sha256_file` → `f.read`, gọi từ
  `bang_hash_thu_muc` ở cuối `main`), không có `ket_qua_test.json`/`cli_log.json` cấp phiên. Tần suất phiên này: 2/3
  exit 0, 1/3 crash. **Nguyên nhân crash CHƯA RÕ** (lan_08 và lan_13 cùng điểm `sha256_file`, chưa phải bằng chứng nguyên
  nhân); không đọc event log ở vòng này. `chup-cuoi` sau tài liệu: xem báo cáo 02 mục 9.
- **Vòng sửa R61-F1…F7 (24/10, review_001 verdict revise)**: đã sửa trong `src/r61_bdd/` + `configs/phase2/r61_bdd/`
  (không sửa helper/protected, không mở A/B): F1 mỗi thao tác B bấm lại toàn bộ gói A + so ảnh khóa hiện tại;
  F2 cache bắt buộc tiền-kiểm và kiểm lại nguồn thực tế trước khi nạp detector; F3 kiểm mọi đường ghi evaluator trước
  helper; F4 marker kiểm schema/tập artifact/danh tính, hỏng → exit 2 trước mọi ghi; F5 khóa mã bằng đóng import tĩnh
  (ast) + mã ultralytics/trackeval + byte file dirty; F6 bỏ `--so-mau`, kiểm chỉ số bootstrap, hậu-kiểm tính lại;
  F7 giao thức ghi tham số bộ lọc dự kiến + yaml_sha256, đối chiếu instance thật. Fixture lan_05/06/07/08: **4/4 exit
  139**, không có `ket_qua_test.json`/`cli_log.json` cấp phiên; mọi kiểm in ra trước điểm crash đều DAT nhưng F2/F3/F4/F6
  và kiểm cuối chưa chạy tới; crash ở vị trí khác nhau (faulthandler lan_07: `pathlib._iterate_directories`; lan_08:
  `hashlib` đọc file trong `sha256_file`). **Nguyên nhân CHƯA RÕ**; không loại trừ lỗi logic hợp đồng khi chưa xác minh
  (đính chính câu "không phải hợp đồng" trước đây). Event log do phiên điều phối đọc 24/10 ~03:00: Application 4 giờ
  không có APPCRASH python.exe; CodeIntegrity/Operational không có sự kiện; Smart App Control tắt từ 23/10 ~12:00 —
  việc thiếu sự kiện không dùng để suy nguyên nhân. `chup-cuoi` đã chạy ở `chuan_bi_cpu/lan_02` (đối chiếu baseline
  gốc lan_01): **dat=false, exit 2**, lỗi duy nhất "vùng nguy cơ đổi ngoài write_paths: .git" (chỉ mtime thư mục `.git`).
  Bằng chứng lan_01…lan_08 giữ nguyên.
- **Vòng P056e6f3d (24/10, BLOCKED):** chỉ sửa `src/r61_bdd/test_chuan_bi_bdd.py` — mỗi lệnh A/B giả lập chạy trong
  tiến trình con riêng (`dieu_phoi_bdd.py`; lệnh `cache` qua `test_chuan_bi_bdd.py --con-dieu-phoi` gán detector mô
  phỏng), log từng lệnh con `<phien>/lenh_con/`. Ba phiên mới lan_09/lan_10/lan_11 khởi chạy (lan_10/11 song song với
  lan_09); kết quả xem báo cáo 02 mục 8. `chup-cuoi` mới `chuan_bi_cpu/lan_03` (baseline gốc lan_01): **dat=false,
  exit 2**, vẫn chỉ lỗi mtime thư mục `.git` (PLAN ngoài mục 0, 43 nguồn bảo vệ, pyc, settings không bị báo lỗi).
  Nguyên nhân thay đổi `.git` chưa xác minh; không đặt lại baseline, không nới so sánh → đối chiếu cuối **không đạt**,
  cần người dùng quyết định. **lan_09 exit 2 (119/120)**: kiểm `F6_hau_kiem_dat_lai_sau_khoi_phuc` không đạt vì
  lệnh con 050–059 (đính chính 24/10 vòng Pa6af1bda: trước ghi "055–059"; lan_10/lan_11 là 031–033 và 038–059, gồm cả
  ca F5/F4; mỗi phiên lan_09–11 có 65 lệnh con, exit 0/10/2 = 18/5/42) báo `git_dirty_sha256: handoff/PLAN_DA_CHOT.md` khác khóa — do agent **sửa mục 0 PLAN trong lúc
  fixture chạy** (khóa giả bấm byte file tracked dirty). Lỗi vận hành của lượt, không phải crash; các ca F6 âm của lan_09
  vì vậy exit 2 sai lý do, không dùng làm bằng chứng. lan_10/lan_11 bị cùng ảnh hưởng (PLAN sửa sau khi khóa) → không hợp
  lệ để nghiệm thu. Kết quả cuối: lan_10 và lan_11 cũng **exit 2, 119/120**, cùng kiểm không đạt
  `F6_hau_kiem_dat_lai_sau_khoi_phuc`; cả 3 phiên không crash. Tần suất: 0/3 exit 0, 0/3 crash. Cần 3 phiên mới chạy
  khi không sửa file tracked.
- **Quyết định M5-Q1…Q8 đã chốt** (ủy quyền chat người dùng 23/10; nguồn
  `D:/LABOTARY/automation/runs_local/hoi_codex_R60_need_decision_20261023/answer.json`; plan R61 người dùng duyệt):
  Q1 ba detector `yolov8n_coco`/`yolov8n_mixed5`/`yolo11n_mixed5` (checkpoint/hash R54); Q2 CV/EKF/UKF/PF500 s0/1/2;
  Q3 c25+c10; Q4 giữ quy ước BDD cũ (5 fps, buffer 6 bước, steps, lớp [1,2,3,5,7], mot_crowd, GT/seqmap hiện có,
  assoc_log=false, ảnh không annotation giữ trong timeline); Q5 BDD COCO cũ chỉ lịch sử, tạo mới 3 cache, UA-DETRAC dùng
  nguyên R53/R56/R57/R59; Q6 COMBINED + 6 hiệu số/nhóm + bootstrap 5.000 × 200, default_rng(0), 108 CI; Q7 wrapper mới
  `src/r61_bdd/`; Q8 báo cáo cuối tách UA-DETRAC và BDD. Hồ sơ: `handoff/logs/review61/01_quyet_dinh_va_giao_thuc.md`
  (hồ sơ R60 giữ nguyên).
- **Đã làm (chỉ CPU, không GPU/cache thật/khóa thật/tracking-chấm thật):** giao thức `configs/phase2/r61_bdd/giao_thuc.json`
  (36 run/200 video/39.973 frame, 600 cache, 7.200 đơn vị — chỉ khai báo), `hop_dong_lenh.json`; mã `src/r61_bdd/`
  (`hop_dong_bdd`, `chuan_bi_bdd`, `dieu_phoi_bdd`, `thong_ke_bdd`, `test_chuan_bi_bdd`). Kiểm kê
  `results/phase2/r61_bdd/chuan_bi/chuan_bi_cpu/lan_01/kiem_ke/` exit 0: 200 video, 39.973 ảnh, **615 ảnh không annotation
  khớp** (27 video; 19 trước annotation đầu, 488 giữa, 108 sau annotation cuối), 778 frame GT vehicle trống, 200/200 video
  đạt kiểm GT/seqinfo/seqmap. Fixture `results/phase2/r61_bdd/tests/lan_04/` exit 0, 69/69 kiểm đạt; lan_01 (exit 1,
  `CUDA_VISIBLE_DEVICES=""`), lan_02 (exit 1, lỗi ghi CSV), lan_03 (exit 139 segfault sau 1 kiểm không đạt) giữ làm bằng chứng.
- **Bước tiếp (24/10, vòng P2e430e0b):** Codex review mục 11 "Tổng hợp crash native" của
  `handoff/logs/review61/02_ket_qua_chuan_bi.md` và cập nhật mục 0 này; approve **không** mở cổng A/B. Cổng A (cache
  GPU/smoke/khóa) và cổng B (thực thi) mỗi cổng cần plan người dùng duyệt riêng, mỗi plan phải có cách xử lý crash native.
  Chưa commit. *(Bước tiếp trước, đã làm — approve `review_005.json`: Codex review gói chuẩn bị, mục 10 nêu giới hạn crash.)*

**Cập nhật của phiên điều phối (23/10/2026 tối, sau khi lượt R60 complete):**
- **R60A COMPLETE:** Codex review vòng 2 **approve** (lượt `r60_ho_so_moc5_20261023_183448`, lời gọi `0004_review`).
  Approve chỉ nghiệm thu hồ sơ tài liệu, **không** mở bước thực thi; phiếu M5-Q1…Q8 trong hồ sơ vẫn để trống.
- **Planner sau R60 (`calls/0005_planner`) trả `need_decision`**, 8 câu M5-Q1…Q8 (phạm vi detector BDD; bộ lọc; c25/c10;
  giữ quy ước BDD cũ; tái dùng kết quả/cache cũ và bằng chứng UA-DETRAC; thống kê BDD; wrapper BDD mới; báo cáo cuối tách
  hai tập). Chưa có plan mốc 5 nào được duyệt; chưa chạy gì.
- **Bảng "Thứ tự các mốc" đã sửa** theo yêu cầu người dùng (chat 23/10): dòng 2 ✅ (R52 mixed5), dòng 4 ✅ (R56/R57/R58/R59),
  dòng 5 🔄 (R60A). Ghi chú "Đính chính bảng (không sửa bảng)" bên dưới là lịch sử trước khi sửa.

**Lượt trước: R60A — hồ sơ quyết định chuyển sang mốc 5 (chỉ đọc tĩnh + tài liệu), 23/10/2026 — COMPLETE (xem trên).**
*(Nội dung dưới ghi khi R60A đang chạy, giữ làm lịch sử.)* Hồ sơ: `handoff/logs/review60/01_ho_so_quyet_dinh_moc5.md`.
- Plan R60 do Codex (GPT) đề xuất sau lượt `bytetrack_r59_sort_test40_20261023_1053` (lời gọi `0012_planner`, 23/10 18:32),
  có hiệu lực vì người dùng đã duyệt; chỉ một bước R60A. Không chạy Python/fixture/tracker/evaluator/bootstrap/kiem-khoa,
  không sửa mã/cấu hình/kết quả; Bash chỉ `git status/diff/rev-parse` (HEAD `117d62da027e8ce3e4c9f455d576a924b4d82bbb`).
- **R59 đóng:** review cuối R59B `D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_005.json`
  verdict **approve**, `issues` rỗng: P1 (bằng chứng kiểm khóa cuối đã chuyển, reviewer băm độc lập 3 file khớp
  `SHA256SUMS_truoc_khi_chuyen.txt`, `dat=true`, 1557 file, `lech=[]`) và P2 (tầng điểm thấp rỗng ở c25) **đã đóng**;
  nghiệm thu đúng phiên `phien_20261023_160539_122500_pid12864`. Giới hạn reviewer: review tĩnh, không chạy lại
  fixture/tracker/evaluator/kiem-khoa; không băm lại 120 NPZ cache hay đếm lại D4; hạn chế `buoc_kiem_khoa` luôn ghi vào
  `chuan_bi/` vẫn tồn tại (không sửa mã đã khóa). **Không còn việc review lại R59 đang chờ.**
- **Đính chính bảng "Thứ tự các mốc" (không sửa bảng):** dòng 4 còn ghi "Chưa: SORT trên test40; ước lượng biến thiên trên
  test40" là trạng thái lịch sử 23/10 trước R57/R59. Hiện: ước lượng biến thiên R57A đã approve (`review_002.json` lượt
  `bytetrack_r57_bien_thien_20261023_0029`); SORT c25 test40 R59B đã approve (`review_005.json` ở trên). SORT c10 không chạy
  (quyết định Q-A R59). Dòng 5 "Chưa" vẫn đúng: chưa có đánh giá cuối mốc 5 nào theo giao thức mới.
- **Phạm vi hồ sơ R60A:** kiểm kê bằng chứng R53/R56/R57/R59 + BDD transfer cũ (khóa 2026-09-20, commit `d1f6d62`, chỉ
  `yolov8n_coco`); đối chiếu BDD cũ với giao thức test40 (checkpoint, ánh xạ lớp, lọc/cắt, cache, fps/dt/buffer, backend,
  evaluator/ignore, PF); phiếu quyết định M5-Q1…Q8 để trống; trình tự sau quyết định. Tracking BDD với detector mixed5
  **chưa thực hiện**; NPZ đánh giá detector R53 **không** dùng làm cache tracking (chưa có bằng chứng hợp đồng tương thích).
- **Review R60A lần 1** (`D:/LABOTARY/automation/runs_local/r60_ho_so_moc5_20261023_183448/reports/review_001.json`):
  verdict **revise**, 2 lỗi P2, đã sửa trong hồ sơ (vòng sửa 2): **R60-F1** — bỏ kết luận "BDD không có tương đương D4"
  suy từ "0 ảnh thiếu"; tách *ảnh thiếu trên đĩa* (`n_missing_images_on_disk` 0) với *ảnh không có annotation*
  (`n_images_without_annotation` 615, `results/phase2/bdd_cocovid_check/val_report.json`) và ví dụ video ảnh tới frame
  202 nhưng annotation dừng ở 94 (`src/mot_datasets/bdd100k_coco.py:155–157`); nguyên nhân và cách chấm các ảnh đó
  **chưa xác minh**, không suy ra âm tính thật, không đổi chính sách đánh giá. **R60-F2** — `open_cache(strict=True)` không
  còn là điều kiện đủ để tái dùng cache: trạng thái `legacy` (thiếu `spec_sha256`) vẫn qua, `check_compatible` chỉ kiểm
  conf/lớp/max_det (`src/det_cache.py:213–230`, :491–530); thêm điều kiện dự kiến về danh tính checkpoint/dataset, cấu hình
  sinh cache, provenance từng sequence, timeline, toàn vẹn bằng chứng, tách phần khóa cũ thực có và phần chưa có bằng
  chứng; thiếu provenance ⇒ dừng phương án phụ thuộc, hỏi người dùng. Không đổi phiếu quyết định nào (vẫn trống).
- **Bước tiếp:** Codex review lại hồ sơ R60A; approve chỉ nghiệm thu tài liệu, **không** mở bước thực thi. Sau đó người dùng
  điền phiếu quyết định; lượt chuẩn bị mã/fixture/giao thức, khóa/smoke và lượt đầy đủ đều cần duyệt riêng. Chưa commit.

**Lượt trước: R59B — thực thi đối chứng SORT–ByteTrack test40 (chỉ c25), 23/10/2026 — COMPLETE: Codex review cuối
approve (`review_005.json`, xem R60A phía trên).** *(Nội dung dưới ghi khi R59B đang sửa hồ sơ, giữ làm lịch sử.)* Báo cáo:
`handoff/logs/review59/03_ket_qua_sort_test40.md`.
- Review B vòng 1 (`D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_004.json`,
  verdict **blocked**): P1 thiếu bằng chứng kiểm khóa cuối trong repo; P2 mô tả tầng điểm thấp chưa chính xác. Người dùng
  duyệt bước P9e199330 (phương án (a), **không** làm c10): chỉ sửa hồ sơ (mục 0 + báo cáo 03), không chạy thí nghiệm, không
  sửa mã/giao thức/khóa/kết quả/bằng chứng cũ.
- **P1 — vị trí bằng chứng kiểm khóa cuối (đã chuyển):** lệnh `kiem-khoa --khoa khoa_20261023_135704_309300_pid25512`
  (exit 0, 16:27:25–16:27:41) ghi `results/phase2/r59_sort_test40/chuan_bi/kiem_khoa_20261023_162725_744900_pid2452/`;
  `chuan_bi/` không thuộc write_paths B nên lúc 2026-10-23 16:36:59 phiên điều phối đã **chuyển nguyên vẹn** thư mục này
  (trước recover, không xóa) sang
  `D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/file_ngoai_pham_vi_B_20261023/kiem_khoa_20261023_162725_744900_pid2452/`
  kèm `SHA256SUMS_truoc_khi_chuyen.txt` (`cli_log.json`
  `a5181c26448ff9b8d6a8359a4ef253de8b5f1dbc2983d243982dec8b104a8870`, `ket_qua.json`
  `7a2851c6e26229b4bf402c865c1c00794fb2c40fb22718cae8b7e28b3225b5c3`,
  `_cau_hinh_thu_vien/ultralytics/Ultralytics/settings.json`
  `4a0445a951396483f1e0d4d61b0bf673b8d254889c274f4f83a5b3786c512494`) và `GHI_CHU.txt`. Lượt sửa hồ sơ chỉ đọc lại nội
  dung, **không** tự băm lại được (lệnh băm không nằm trong quyền Bash của lượt); hash do phiên điều phối ghi. Nội dung `ket_qua.json`:
  `dat: true`, `so_file: 1557`, `lech: []`, `khoa_sha256` `b18fc376…4057d` (= khóa A). Thư mục không còn ở `chuan_bi/`.
  **Hạn chế đã biết (không sửa mã đã khóa):** `buoc_kiem_khoa` (`src/r59_sort_test40/dieu_phoi_sort_test40.py:991`) luôn
  ghi phiên kiểm khóa vào `chuan_bi/`, kể cả khi gọi trong B.
- **P2 — tầng điểm thấp:** với đầu vào conf > 0,25 và `track_high_thresh` 0,25, tầng điểm thấp của ByteTrack **rỗng**
  (`bt_thap = 0`: giao thức `configs/phase2/r59_sort_test40/giao_thuc.json`; fixture
  `src/r59_sort_test40/test_dieu_phoi_sort_test40.py:115`; K2 `chuan_bi_20261023_135554_118700_pid21600/k2_parity.csv`
  120/120 dòng `n_det_bt_thap = 0`). Hiệu số ByteTrack−SORT ở c25 **không** đo đóng góp association điểm thấp; khác biệt
  còn lại là vòng đời, association và khởi tạo track.
- Cổng A: Codex review A vòng 2 **approve** (`D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_003.json`,
  SHA256 `1762b88a…f8d6`) đúng gói `goi_a_20261023_135911_306800_pid9372` + khóa `khoa_20261023_135704_309300_pid25512`.
- Phiên B **`results/phase2/r59_sort_test40/thuc_thi/phien_20261023_160539_122500_pid12864`** (mo-phien ràng buộc review
  A/gói/khóa/mã/giao thức). Full 2 đoạn cùng phiên (`--gioi-han-giay 420` thay 2400 vì lệnh Bash foreground bị giới hạn
  600 s; dừng có kiểm soát giữa đơn vị: 155 rồi 85 đơn vị mới, 0 lần dở/lỗi, cache trước = sau mỗi run). Chấm 6 run,
  lap-tracker MVI_39031 2/2 MOT trùng byte, cham-lai 2/2 ba file metric trùng byte, hau-kiem đạt (240 đơn vị, 6 chấm,
  56.340 ảnh/run), tong-hop đạt (ByteTrack khớp bảng R56 sai số 0 mọi metric), kiem-khoa cuối DAT (bằng chứng nay ở
  vùng automation, xem P1 phía trên). Tất cả exit 0 (trừ đoạn full 1 exit 10 = chưa xong theo thiết kế).
- Kết quả (`results/phase2/r59_sort_test40/tong_hop/phien_20261023_160539_122500_pid12864/doi_chieu_6_cap.csv`, chỉ điểm,
  không CI): ByteTrack − SORT HOTA +3,71/+3,63 pp (yolov8n_coco CV/EKF), +1,79/+1,67 (yolov8n_mixed5), +1,38/+1,48
  (yolo11n_mixed5); chủ yếu từ AssA (+2,9…+7,1 pp), DetA ≈ 0 (−0,06…+0,75), IDSW ByteTrack ít hơn 326–1.820.
- *(Bước tiếp lúc đó, đã làm — approve `review_005.json`:)* Codex review lại toàn bộ R59B (phiên `thuc_thi/phien_20261023_160539_122500_pid12864`, `tong_hop/`,
  bằng chứng kiểm khóa cuối đã chuyển, bản sửa hồ sơ). Ngân sách sửa-review đã dùng 5/5. Sau nghiệm thu dừng ở kết quả
  đối chứng mốc 4; không tự mở mốc tiếp. Chưa commit.

*(R59A, giữ làm lịch sử — đã approve ở review A vòng 2 như trên:)* Báo cáo cổng A: `handoff/logs/review59/02_bang_chung_cong_a.md`.
- Review A vòng 1 (`D:/LABOTARY/automation/runs_local/bytetrack_r59_sort_test40_20261023_1053/reports/review_002.json`,
  revise) trên gói `goi_a_20261023_130840_575200_pid5080` + khóa `khoa_20261023_130642_719700_pid18116`. Đã sửa trong
  write_paths (không sửa helper): (1) fixture kiểm junction/symlink `tests/` + `<lan>` **trước** lần ghi đầu
  (`tao_lan_tests` = `R55.kiem_doc_lap` rồi mới `W.tao_phien`), có test junction thật; (2) wrapper `kiem_goi_a`: mở
  phiên và mọi `Phien.xac_minh` (full/tiếp tục/chấm/…) kiểm **từng file** trong `goi_a.json#file` (tồn tại, không qua
  junction, khớp SHA256) chứ không chỉ marker ngoài; fixture có ca lệch hash smoke / thiếu file chấm nguồn / marker gói
  hỏng (mo-phien exit 2, không tạo phiên) và đổi gói sau mo-phien (full exit 2, không ghi tracking); (3) K3 qua
  `SORT.update` thật cả 2 bộ lọc (==/trên/dưới 0,25) + test cắt `[:300]` (480 det → 300, SORT nhận đủ 300).
- Kết quả chạy lại sau sửa (foreground, tuần tự, đều exit 0; ID truyền tường minh): fixture **15/15**
  `tests/lan_20261023_135030_327000_pid26008`; `chuan-bi` `chuan_bi/chuan_bi_20261023_135554_118700_pid21600` (0 lệch,
  6 nguồn đối chứng đạt, K2 **120/120**, 56.340 ảnh + 173 D4 mỗi detector, `config.py` byte thô = khóa R56, LF = R51);
  khóa **`chuan_bi/khoa_20261023_135704_309300_pid25512`** (1.557 file, `khoa.json` SHA256
  `b18fc37632fe94e0b1f5310df1c685f374ab786531800b69f6dced30c3d4057d`); `kiem-khoa` DAT; smoke K7
  `smoke/smoke_20261023_135748_242700_pid12692`; chấm nguồn `kiem_duong_cham/cham_nguon_20261023_135811_750100_pid4508`
  (3 file metric trùng byte R56); gói A **`chuan_bi/goi_a_20261023_135911_306800_pid9372`** (463 file, 4.3 + K1–K7 đạt,
  `goi_a.json` SHA256 `d0ddcde25126e316883c93dc206f0113a48d9ab8f011ab6b35b58e331935b740`).
- *(Bước tiếp lúc đó, đã làm — approve, xem R59B phía trên:)* Codex review A lại trên gói `goi_a_20261023_135911_306800_pid9372` + khóa
  `khoa_20261023_135704_309300_pid25512`. Chỉ khi approve mới chạy B (`--buoc mo-phien --khoa
  khoa_20261023_135704_309300_pid25512 --review-a <review>`). Khóa/gói cũ dưới đây là lịch sử, **không** dùng cho B. Chưa commit.

*(Lần chạy lại trước, giữ làm lịch sử — bị review A vòng 1 trả revise:)*
- Kết quả lần chạy lại (foreground, tuần tự, đều exit 0): fixture **13/13** `results/phase2/r59_sort_test40/tests/lan_20261023_125956_551000_pid25124`;
  `chuan-bi` `chuan_bi/chuan_bi_20261023_130527_496800_pid26216` (0 lệch; 6 nguồn đối chứng đạt; K2 **120/120**;
  56.340 ảnh + 173 D4 mỗi detector; `config.py` byte thô `37e4f562…` = khóa R56, LF `0e3f8b75…` = R51); khóa
  **`chuan_bi/khoa_20261023_130642_719700_pid18116`** (1.557 file, `khoa.json` SHA256
  `51de77f85ccfb2cf3f8192cd88ad8d779e412568395c9171bd66bedb4dd2a69c`); `kiem-khoa` DAT; smoke K7
  `smoke/smoke_20261023_130721_179800_pid3660` (2 SORT × 2 video × 128 frame, cache bất biến); chấm nguồn
  `kiem_duong_cham/cham_nguon_20261023_130742_612500_pid11204` (3 file metric trùng byte R56); gói A
  **`chuan_bi/goi_a_20261023_130840_575200_pid5080`** (462 file, 4.3 + K1–K7 đạt, kiểm khóa cuối DAT).
- Sửa trong lần này (trước chuan-bi; mã/fixture đã khóa là bản sau sửa): import `run_tracking` sau `dat_moi_truong`
  ở smoke/full/lap-tracker; in ID trên dòng riêng (cham-nguon/dong-goi-a/mo-phien); kỳ vọng K4 sai trong fixture
  (SORT xuất từ lần ghép thứ 4 khi không có ngoại lệ đầu video — đúng `sort_tracker.py`/SORT gốc). Hai lần fixture
  lỗi giữa chừng (7/13, 8/13) giữ nguyên trong `tests/`.
- *(Bước tiếp lúc đó, đã làm:)* Codex review A trên gói/khóa này → revise (xem phía trên).

*(Ghi ở lần chạy đầu, giữ làm lịch sử — nguyên nhân môi trường, đã được người dùng gỡ:)* R59A BLOCKED (môi trường):
Smart App Control chặn `torch\lib\shm.dll` (WinError 4551) nên không nạp được torch/ultralytics; fixture CPU chưa chạy
được, chưa chạy chuan-bi/khoa/smoke/cham-nguon/dong-goi-a, chưa có khóa R59, chưa review A, chưa full.
Plan R59 (Codex đề xuất sau lượt `bytetrack_r58_sort_design_20261023_053436`, lời gọi `thu_cong_20261023_r58`) do người
dùng duyệt; quyết định Q-A..Q-F (chỉ c25; `det_thresh 0,25`; SORT `assoc_log=false`; tái dùng ByteTrack R56 có điều
kiện mục 4.3; chỉ điểm, không CI; kiểm hash `config.py` + bộ lọc từ instance) ghi ở
`handoff/logs/review59/01_quyet_dinh_va_giao_thuc.md`. R58A chỉ là thiết kế (approve chỉ nghiệm thu hồ sơ).
- Đã viết (chưa kiểm runtime): giao thức `configs/phase2/r59_sort_test40/giao_thuc.json` (6 run đúng thứ tự, ánh xạ
  `r54-<detector>-c25-cv/ekf`, quy tắc metric/đơn vị/làm tròn/dung sai khóa trước: tỷ lệ 0,0005 pp theo bảng R56,
  IDSW/FP 0); wrapper `src/r59_sort_test40/dieu_phoi_sort_test40.py` (bước A: chuan-bi, khoa, kiem-khoa, smoke,
  cham-nguon, dong-goi-a; bước B: mo-phien, full, cham, lap-tracker, cham-lai, hau-kiem, tong-hop; chỉ gọi helper
  R54/R55, không sửa helper); fixture `src/r59_sort_test40/test_dieu_phoi_sort_test40.py` (K1/K3/K4 trên instance thật
  + luồng A/B trên dữ liệu tổng hợp).
- Lệnh đã chạy: fixture 2 lần → exit 1 cả hai, `OSError: [WinError 4551] ... torch\lib\shm.dll` ngay khi import
  ultralytics (trước mọi test). Để lại 2 thư mục rỗng `results/phase2/r59_sort_test40/tests/lan_*` (chỉ có
  `_cau_hinh_thu_vien` rỗng, không có `ket_qua_test.json`).
- *(Bước tiếp lúc đó, đã làm ở lần chạy lại phía trên:)* người dùng gỡ chặn Smart App Control, chạy lại R59A từ (4).
- Đính chính trạng thái R57: R57A **đã approve** (`D:/LABOTARY/automation/runs_local/bytetrack_r57_bien_thien_20261023_0029/reports/review_002.json`);
  câu "còn thiếu ước lượng biến thiên" ở bảng "Thứ tự các mốc" là trạng thái cũ (không sửa lịch sử, không mở lại R57).

**Lượt trước: R58A — hồ sơ thiết kế đối chứng SORT–ByteTrack test40 (chỉ đối chiếu tĩnh, DỰ THẢO), 23/10/2026 —
COMPLETE: Codex review vòng 2 approve (06:12, `D:/LABOTARY/automation/runs_local/bytetrack_r58_sort_design_20261023_053436/calls/0004_review`).
Planner `0005_planner` trả need_decision 6 câu Q-A..Q-F (phạm vi c25 / +c10, det_thresh c10, assoc_log SORT, tái dùng
ByteTrack R56, CI, hash config.py); theo ủy quyền người dùng (23/10) Codex trả lời, chờ hạn mức Codex 5 giờ reset 10:22.**
*(Ghi lúc đang chạy R58A, giữ làm lịch sử:)* Plan R58 (Codex đề xuất sau lượt `bytetrack_r57_bien_thien_20261023_0029`, lời gọi `0005_planner`)
do người dùng duyệt; duyệt R58 **chỉ** cho soạn hồ sơ, **không** chốt c25/c10, `det_thresh`, ma trận chạy hay cấp quyền
chạy. Hồ sơ: `handoff/logs/review58/01_thiet_ke_sort_test40.md` (DỰ THẢO, CHƯA KHÓA, CHƯA CẤP QUYỀN CHẠY).
- Không chạy Python, test, tracker, evaluator, kiểm khóa; không sửa mã/cấu hình/kết quả. Chỉ đọc mã, YAML và metadata nhỏ.
- Điểm chính chờ người dùng: phương án c25 (6 run SORT mới) hay c25 + c10 (12 run); nếu c10 thì `det_thresh` của SORT
  (YAML khóa hiện có 0,25 ⇒ theo đọc mã SORT c10 nhận đúng tập detection như c25 — "lọc kép", chưa kiểm runtime);
  `assoc_log` cho SORT (theo đọc mã mixin không tương thích `SORTTrack`); hash `config.py` R51 (LF `0e3f8b75…`) và khóa
  R56 (`37e4f562…`) khác chuỗi, chưa xác minh cùng quy ước băm. Phiếu quyết định: hồ sơ mục 9.
- Review 1 R58A (Codex, `review_001.json` lượt `bytetrack_r58_sort_design_20261023_053436`): **revise**, 4 điểm hồ sơ
  (P2 K4 tách detection rỗng với D4; P2 diễn giải phương án c10 `det_thresh 0,1`; P3 điều kiện xóa/xuất `max_age`;
  P3 tái lập detector). Đã sửa trong hồ sơ, chỉ tài liệu. Người dùng **từ chối** hai ý tưởng vòng 1 (P74d51e88 sửa
  mixin `assoc_log`; P3b9dd663 fixture kiểm lọc kép c10); không đưa lại.
- **Bước tiếp:** Codex review cuối R58A; xong thì **dừng**. Lượt sau (cần duyệt riêng, theo thứ tự): chốt thiết kế → chuẩn
  bị mã/fixture/khóa → duyệt chạy thật. Chưa commit.

**Lượt trước: R57A — ước lượng biến thiên test40 (bootstrap ghép cặp, chỉ phân tích hậu nghiệm), 23/10/2026 —
Codex APPROVE ở review lại (`D:/LABOTARY/automation/runs_local/bytetrack_r57_bien_thien_20261023_0029/reports/review_002.json`,
verdict approve, không issue mở).** Vòng 1 revise 3 finding (schema CSV, dung sai, nơi ghi hash) đã đóng. Giới hạn
reviewer: review tĩnh, không chạy lại fixture/bootstrap/thí nghiệm; chỉ kiểm file nhỏ manifest trỏ tới, không nghiệm thu
lại toàn bộ R56; CI riêng lẻ thăm dò, chưa hiệu chỉnh đa so sánh; kiểm cùng cảnh chỉ bao cặp đã biết. Reviewer ghi nhận
đuôi hash rút gọn giao thức ở mục này bị ghi nhầm (đã sửa dưới đây; lỗi trình bày, không ảnh hưởng nghiệm thu). Hai đề
xuất R57 (hiệu chỉnh đa so sánh, mở rộng kiểm cùng cảnh) **đã bị người dùng từ chối**. Plan R57 (Codex đề xuất sau lượt
`bytetrack_r56_video_prepare_20261021_151440`, lời gọi `thu_cong_20261023`) do người dùng duyệt; khác plan R57 cũ
`plan_0006` đã bị từ chối. Báo cáo (lịch sử, giữ nguyên): `handoff/logs/review57/01_bien_thien_test40.md` (mục 8 = các sửa đổi).
- **Phiên hiện hành: `results/phase2/r57_bien_thien/phien_20261023_011207_108057_pid8500`** (giao thức phiên bản 2,
  SHA256 `987986a58c6b175d246229853524d5afb56b3905061805ffdef0802393f2b4ed`; mã `phan_tich_test40.py` `53e4933e…6a73ee82`, fixture `test_phan_tich_test40.py`
  `93f35f0f…f7444b27`). Fixture **exit 0, 15/15 đạt** (`tests/lan_20261023_011135_199544_pid12284/ket_qua_test.json`);
  `chuan-bi` exit 0, `phan-tich` exit 0. Sửa: (P2) đọc CSV chặt (header trùng/rỗng, dòng thừa/thiếu cột, khóa trùng,
  số không hữu hạn → chặn; bảng R56 kiểm cả ở chuan-bi); (P2) dung sai bảng R56 khai báo riêng từng bảng trong giao thức
  (tong_hop 0,0005/0,0005/0; pf500 0,00055/0,00055/0,00005 theo quy tắc làm tròn R56), bỏ `max(tb, 0,0005)` ngầm; sai số
  lớn nhất thật ghi vào `kiem_chung.json`; tiêu chí tái tổ hợp vs `combined_metrics.csv` (1e-9/1e-9/0) giữ nguyên;
  (P3) sửa mô tả nơi ghi hash mã. Chỉ số mẫu, điểm, `doi_chieu_video.csv`, `doi_chieu_nhom.csv` **trùng byte** phiên cũ.
  Phiên cũ `phien_20261023_004544_759728_pid25976` (giao thức phiên bản 1) giữ nguyên, không còn là phiên hiện hành.
  Các gạch đầu dòng dưới mô tả lần làm đầu (phiên cũ); số liệu kết quả vẫn đúng cho phiên mới.
- Giao thức `configs/phase2/r57_bien_thien/giao_thuc.json` (phiên bản 1: SHA256 `b01eee60…288438b9e`) cố định **sau khi đã xem điểm
  test40 R56** (không phải tiền đăng ký), trước khi tính CI: 36 run × 40 video (thứ tự manifest khóa), B = 5000,
  `default_rng(0)` riêng mỗi thiết kế, phân vị 2,5/97,5 `linear`; HOTA chính, IDF1 + IDSW bổ sung; 6 nhóm detector ×
  c25/c10 × 6 hiệu số EKF−CV, UKF−CV, PF500−CV, UKF−EKF, PF500−EKF, PF500−UKF. Thiết kế video (40) và thiết kế độ nhạy
  39 nhóm (cặp cùng cảnh `MVI_40773`/`MVI_40775` + 38 video đơn, khai báo tường minh). PF500 = trung bình COMBINED 3 seed
  trong từng mẫu, không bootstrap seed. Không so detector, không chọn conf.
- Mã `src/r57_bien_thien/phan_tich_test40.py` (`8cfe90f1…cdf6b8ee`), fixture `test_phan_tich_test40.py`
  (`55b2fa51…d64e3d8`). Lệnh foreground, tuần tự: (1) fixture → **exit 0, 13/13 đạt**
  (`results/phase2/r57_bien_thien/tests/lan_20261023_004525_846673_pid25692/ket_qua_test.json`); (2) `--buoc chuan-bi`
  → **exit 0**, phiên `phien_20261023_004544_759728_pid25976`; (3) `--buoc phan-tich --phien …` → **exit 0**.
- Kiểm trên đầu vào thật (`results/phase2/r57_bien_thien/phien_20261023_004544_759728_pid25976/kiem_chung.json`): 36 run
  × 40 video, PF seed 0/1/2 ở cả 6 nhóm, 188 file đầu vào khớp marker/nguồn, hash trước/sau 0 lệch; tái tổ hợp 40 video
  khớp `combined_metrics.csv` và `per_video_metrics.csv` của 36 run với sai số 0,0 (HOTA/IDF1/IDSW); bảng R56 khớp theo
  làm tròn; nhánh video trùng `boot_seed.paired_bootstrap_seed` 108/108 (==); tái tính từ chỉ số đã lưu trùng; 108 dòng CI
  mỗi thiết kế; 2 thiết kế không khác phân loại CI (âm / chứa 0 / dương) ở dòng nào (0/108), tỉ lệ độ rộng CI nhóm/video
  0,954–1,060.
- **Kết quả mô tả (CI 95 % riêng lẻ, thăm dò, chưa hiệu chỉnh đa so sánh):** PF500 thấp hơn cả CV/EKF/UKF ở cả 6 nhóm:
  HOTA −2,90 đến −3,43 pp, IDF1 −3,72 đến −4,62 pp, IDSW +719 đến +1.084; CI cả 54 dòng PF (×3 metric) không chứa 0.
  Giữa CV/EKF/UKF: chênh HOTA |≤ 0,34| pp; CI HOTA không chứa 0 ở 3/18 dòng (EKF−CV yolov8n_coco-c10 +0,33
  [+0,11; +0,58], yolo11n_mixed5-c25 +0,18 [+0,007; +0,38]; UKF−EKF yolov8n_coco-c25 −0,11 [−0,19; −0,02]); IDF1 2/18,
  IDSW 2/18. CI chứa 0 không có nghĩa tương đương; không tuyên bố thắng toàn cục hay nhân quả.
- Giới hạn: giả định video (hoặc nhóm) độc lập; kiểm cùng cảnh chỉ dựa 4 frame mẫu/video (R48 `03` mục 6.4), có thể còn
  cặp chưa phát hiện; 108 CI mỗi thiết kế chưa hiệu chỉnh; chỉ băm file metric nhỏ, không phải nghiệm thu lại toàn bộ R56.
- **Chưa làm (lúc R57A):** QJTF; SORT trên test40. R57A đã approve; lượt kế tiếp là R58A (ở trên). Chưa commit.

**Lượt trước: THỰC THI tracking test40 (21/10/2026 16:18–22:01) — XONG 127/127 lệnh exit 0, Codex NGHIỆM THU
APPROVE lần 2 (23/10 00:24, `D:/LABOTARY/automation/nghiem_thu_test40/lan_013/answer.json`, review tĩnh có giới hạn).**
Báo cáo: `handoff/logs/review56/03_thuc_thi_tracking_test40.md`.
- **Quyết định người dùng (chat 21/10 ~15:50):** sau R56 bỏ plan thực thi / planner; chạy cache + đo đoạn dày + full bằng
  bộ điều phối đã khóa **ngoài orchestrator**; **Codex nghiệm thu một lần ở cuối**. Plan R57 (`plan_0006`) TUCHOI theo
  quyết định này. Người dùng duyệt dùng phần dự phòng hạn mức tuần Codex (21/10 15:45).
- Khóa R56_LOCK `khoa_20261021_155510_440500_pid8768` (`ce85628e…9d35247b`); `--duyet
  nguoi_dung_chat_20261021_1550__chay_cache_do_doan_day_full_ngoai_orchestrator__codex_nghiem_thu_cuoi`. Bộ chạy tuần tự
  ngoài repo `D:/LABOTARY/automation/thuc_thi_r56/` (WMI; stdout từng lệnh; `stdout_sha256` ghi trên nội dung LF, không
  khớp byte CRLF trên đĩa — manifest hash byte thực tế `stdout_sha256_thuc_te.json` `41124dbf…5a374e2`, 127/127 khớp
  sau chuẩn hóa LF; Codex nghiệm thu `lan_012` P2). Phiên thực thi
  `results/phase2/r55_tracking/thuc_thi/phien_20261021_161832_735800_pid11640/`.
- Đã làm theo đúng thứ tự bàn giao R56: cache + xác nhận 3 detector (1.005 / 1.030 / 1.404 s) → đo UKF/PF500 trên MVI_40864
  toàn video (24 lệnh: UKF ~29–31 s, PF500 ~39–46 s mỗi lệnh, gồm chi phí khung) → tracking full 36 run × 40 video (1.440
  đơn vị, ≈ 3,58 h) → chấm 36 run → `lap-tracker` 9/9 trùng byte, `cham-lai` 12/12 trùng byte → `hau-kiem` exit 0 (120 / 1.440 /
  36, không lỗi, không lần dở) → `kiem-khoa` cuối exit 0.
- **Kết quả mô tả (HOTA combined, 40 video):** mixed5 cao hơn COCO ~3–4 điểm; CV/EKF/UKF chênh ≤ 0,34 trong cùng
  detector/điều kiện (EKF cao nhất ở cả 6 cặp, chênh nhỏ); PF500 thấp hơn Kalman ~2,9–3,4, IDSW cao hơn; cao nhất
  yolo11n_mixed5-c25-ekf 62,239. Bảng: `results/phase2/r56_tong_hop/phien_20261021_161832_735800_pid11640/`
  (`src/r56_tong_hop/tong_hop_test40.py`, chỉ đọc). Không bootstrap / khoảng tin cậy; chưa phải kết luận.
- **Codex nghiệm thu lần 1 (23/10 00:19, `D:/LABOTARY/automation/nghiem_thu_test40/lan_012/answer.json`): revise**, chỉ
  hồ sơ: P2 hash stdout LF/CRLF (đã bổ sung manifest byte thực tế, đính chính báo cáo 03 mục 1 và mục này), P3 diễn giải
  thời gian run CV đầu (đã sửa thành giả thuyết chưa kiểm chứng). Codex xác nhận đủ 36×40, chấm đủ, tái lập khớp, không lỗi
  làm sai metric, không cần chạy lại. **Lần 2 (23/10 00:24, `lan_013`): approve**, P2/P3 đã xử lý, không issue mở.
  Giới hạn reviewer: review tĩnh; chỉ băm mẫu đầu ra cache/tracking; tái lập chỉ từ cùng cache/MOT (không suy luận
  detector giữa tiến trình); chưa có ước lượng biến thiên; không mở rộng sang SORT test40, BDD tracking, QJTF, Nano.
- **Chưa làm:** QJTF; SORT trên test40; ước lượng biến thiên. Việc tiếp theo: người dùng giao Codex chọn (23/10). Chưa commit.

**Lượt trước: R56 — complete (R56A `review_001`, R56B `review_002` approve, 21/10 16:13).** Bổ sung chọn một video cho
tracking trong bộ điều phối R55 + khóa mới liên kết khóa R55 cha; chỉ chuẩn bị. Nội dung dưới đây ghi khi R56 đang chạy.
- **R55 complete:** lượt `bytetrack_r55_prepare_20261021_051232`, Codex `review_003.json` approve R55A, `review_004.json`
  approve R55B (`D:/LABOTARY/automation/runs_local/bytetrack_r55_prepare_20261021_051232/reports/`). Không review nào cấp
  quyền chạy thật. Khóa R55 cha: `results/phase2/r55_tracking/chuan_bi/khoa_20261021_121500_644100_pid22868/`, `khoa.json`
  `5a07efb8…4cf6e42b`, 376 file (kế thừa 368 file khóa R54 `e2d1b0d5…a418e16b4f`). Là bằng chứng lịch sử bất biến; sau R56A
  khóa cha **không còn khớp** mã hiện tại ở 3 file được thay (mã điều phối, test, hợp đồng R55).
- **Người dùng chọn phương án A (plan R56, Codex đề xuất ở lời gọi `0017_planner` 21/10 15:12, người dùng duyệt):** bổ sung
  `tracking --run … --video <v>` (toàn bộ timeline video đó), fixture CPU, khóa mới. Đoạn đo cố định **MVI_40864 frame
  1–1515 = toàn video** (căn cứ CSV tiền kiểm R55: 1.515 ảnh, 44.689 dòng GT → ≈ 29,50 dòng GT/ảnh; **chưa đo**). Giao
  thức giữ nguyên: 3 detector, c25/c10, CV/EKF/UKF/PF500 seed 0/1/2, 36 run × 40 video, 56.340 ảnh / 173 D4.
  P1980cb21 giữ (đo đoạn dày thuộc plan thực thi); P6fb1cad7/P9857b26c không làm lại; QJTF chỉ ghi nhận cho plan sau.
- **R56A (21/10) — Codex approve** (`D:/LABOTARY/automation/runs_local/bytetrack_r56_video_prepare_20261021_151440/reports/review_001.json`,
  review tĩnh, không cấp quyền chạy thật): kiểm khóa cha trước sửa `--buoc kiem-khoa --khoa khoa_20261021_121500_644100_pid22868`
  (không `--luu`) → **exit 0, 376 file, khóa R54 368 file, 0 lệch**. Sửa đúng 3 file: `src/r55_tracking/dieu_phoi_test40.py`,
  `src/r55_tracking/test_dieu_phoi_test40.py`, `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json`; tài liệu mới
  `handoff/logs/review56/01_bo_sung_chon_video.md`. Thiết kế R55 (`review55/01`, `02`) giữ nguyên byte; helper R54 không đổi.
  - Thay đổi: `tracking --video <v>` chỉ lọc danh sách đơn vị (toàn timeline video), video lạ / `--video` ở công việc
    khác tracking/lap-tracker → exit 2; vẫn kiểm manifest + hash cache **mọi** video trước/sau tracking; ràng buộc tracking
    không chứa phạm vi lọc (đường đầy đủ tái sử dụng); nhật ký `pham_vi` + `don_vi` (frame thực xử lý, mới/tái sử dụng).
    Khóa `R56_LOCK`: hợp đồng `khoa_cha_r55` (đường dẫn + SHA256 khóa cha, `duoc_thay` = đúng 3 file); `--buoc khoa` kiểm
    nguyên vẹn khóa cha, băm mọi file cha, chỉ 3 file được khác; khóa mới gồm mọi file cha + khóa cha + tài liệu bổ sung,
    ghi `khac_biet_khoa_cha`; kiểm khóa thực thi băm mọi file theo khóa mới, không ngoại lệ, + kiểm liên kết; đường thật chỉ
    nhận `R56_LOCK`. Hợp đồng `doan_day.doan_co_dinh` MVI_40864 1–1515; `tien-kiem` đường thật đối chiếu với bảng.
  - Fixture foreground `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py`
    **exit 0, 28/28 đạt** (397,6 s; 59 lệnh CLI tiến trình con thật):
    `results/phase2/r55_tracking/tests/lan_20261021_152952_196200_pid9756/` (`ket_qua_test.json`, `cli_log.json`). 24 ca
    R55 giữ nguyên + J1 (khóa R56/khóa cha), J2 (`--video` qua CLI), J3 (đoạn cố định), Z2 (không thực thi thật). SHA256
    ghi trong kết quả = bản gửi review (không sửa sau lần chạy): `dieu_phoi_test40.py`
    `257f4f028e5a39aec4ea61a8387efbab518631d05021942ce8a3d7d4a973961e`, `test_dieu_phoi_test40.py`
    `87689fbcce7f4274d8dc152ac88ed1fbbc65fffebc9046d8d8f8ba31ef4e57a6`, hợp đồng
    `488238d4fc669ffac3ce98800b7031b08e8233504a84751afe3c29e4239cf95d`; thiết kế R55 `9687aae7…2419a598` (= bản trong khóa
    cha, nguyên byte). Kiểm liên kết khóa R55 cha THẬT (chỉ đọc) với mã hiện tại: 376 file, 373 giữ, đúng 3 file thay, 0 lệch.
  - Chi tiết, giới hạn: `handoff/logs/review56/01_bo_sung_chon_video.md`.
- **R56B (21/10 15:54–15:55) — ĐÃ LÀM, CHỜ Codex review.** Chỉ 3 lệnh chuẩn bị, foreground, tuần tự, tên phiên tường minh
  lấy từ dòng in của bước trước (không chọn "mới nhất"); mã/test/hợp đồng/thiết kế giữ nguyên bản R56A đã approve:
  1. `--buoc tien-kiem` → exit 0 (`dat`, 0 lệch, 19,0 s): `results/phase2/r55_tracking/chuan_bi/tien_kiem_20261021_155443_527300_pid26076/`
     (`tien_kiem.json` `aec04ded1df31154b7aa74ef0f525ab311ef8763f81b51542f5d147031b143ba`). 40 video, **56.340 ảnh**
     (stat + size đúng 56.340), **173 D4**, 675.774 dòng GT, 163 vùng; khóa R54 368 file 0 lệch; checkpoint/names CPU, cấu
     hình hiệu lực, thư viện, backend trùng khóa R54. `bang_tien_kiem.csv` `c026e173…87d2c793` và `mau_anh.json`
     `a902a88d…76844dc` trùng byte bản R55. Đoạn cố định đối chiếu bảng: MVI_40864 1–1515 toàn video, 44.689 dòng GT, 0 lệch.
  2. `--buoc khoa --tien-kiem tien_kiem_20261021_155443_527300_pid26076 --tests lan_20261021_152952_196200_pid9756` → exit 0:
     **khóa R56 `results/phase2/r55_tracking/chuan_bi/khoa_20261021_155510_440500_pid8768/`, `R56_LOCK`, 384 file, `khoa.json`
     `ce85628e34bc8507f9577c51a06d381a2d8e5d0f35cdf099c18330429d35247b`**. Liên kết khóa cha R55 `5a07efb8…4cf6e42b` (376
     file: giữ 373, thay đúng 3 file được phép, mới 8 = kết quả fixture R56A, 3 file tiền kiểm, `review56/01`, 3 file khóa
     cha), khóa R54 `e2d1b0d5…a418e16b4f` (368), giao thức R54 `eb1f8f30…715431ad`, hợp đồng `488238d4…4239cf95d`, 36 run.
  3. `--buoc kiem-khoa --khoa khoa_20261021_155510_440500_pid8768 --luu chuan_bi` → exit 0 (384 + 368 file, khóa cha đúng
     hash, 0 lệch): `results/phase2/r55_tracking/chuan_bi/kiem_khoa_20261021_155524_561000_pid12384/ket_qua.json`.
  - Khóa cha R55 và mọi bằng chứng cũ giữ nguyên byte (khóa cha chỉ là lịch sử; **không** khớp mã hiện tại ở 3 file thay).
    PLAN và `review56/02` không nằm trong khóa R56. Không tạo `results/phase2/r55_tracking/thuc_thi/`, không cache/MOT/metric
    thật, không đo đoạn dày, không GPU/`setup_model`. Bàn giao: `handoff/logs/review56/02_khoa_va_ban_giao.md`.
- **Việc chưa làm (chuyển plan thực thi, cần người dùng duyệt riêng `--duyet`):** tạo/xác nhận cache đủ 40 video mỗi
  detector theo khóa R56 → đo PF500/UKF trên đoạn cố định MVI_40864 1–1515 (chưa đo; P1980cb21) → chỉ sau đó tới full
  36 run × 40 video. QJTF (đo chi phí kiểm module mỗi tác vụ) chỉ là ý tưởng cho plan thực thi, chưa triển khai.
- **Bước tiếp:** Codex review R56B. Hoàn tất R56 thì **dừng**; R56 không cấp quyền chạy thật.
- *(Lịch sử R55, giữ nguyên dưới đây.)* R55 — CHỈ CHUẨN BỊ bộ điều phối test40. R55A: Codex `review_003.json` approve;
  R55B 3 bước chuẩn bị trên đầu vào thật exit 0 cả ba → Codex `review_004.json` approve.
- **R55B (21/10 12:14–12:15), chỉ 3 lệnh chuẩn bị, foreground, theo thứ tự, tên phiên tường minh lấy từ kết quả:**
  1. `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/dieu_phoi_test40.py --buoc tien-kiem` →
     **exit 0, `dat`, 0 lệch, 20,5 s**: `results/phase2/r55_tracking/chuan_bi/tien_kiem_20261021_121436_943200_pid22720/`
     (`tien_kiem.json` `a88426a1…ee07b946a0`, `bang_tien_kiem.csv` `c026e173…7d2c793` 40 video, `mau_anh.json`
     `a902a88d…6844dc`). Khóa R54 băm lại 368 file 0 lệch (`khoa.json` `e2d1b0d5…a418e16b4f`); 3 checkpoint SHA = giao thức
     = khóa R53, names đọc CPU (coco80 / lop5); tổng 56.340 ảnh (stat + size đúng 56.340), 173 D4, 56.167 frame có GT,
     675.774 dòng GT, 163 vùng ignored; cấu hình hiệu lực, thư viện (ultralytics 8.4.141, torch 2.6.0+cu124, trackeval
     1.3.0, numpy 2.4.6…), backend `ultralytics` trùng khóa R54; dung lượng: kịch bản ~3,13 GB cần / 44,5 GB trống
     (kích thước file smoke R54 × 220,08, chỉ stat, không đọc NPZ). **Ảnh: kế thừa stat/size theo bảng hash R47 + băm lại
     256 ảnh mẫu cố định R54 (MVI_39031/39051 frame 1–128); KHÔNG băm lại toàn bộ 56.340 ảnh.**
  2. `… --buoc khoa --tien-kiem tien_kiem_20261021_121436_943200_pid22720 --tests lan_20261021_115928_508900_pid21264` →
     **exit 0**: khóa `results/phase2/r55_tracking/chuan_bi/khoa_20261021_121500_644100_pid22868/`, `khoa.json` SHA256
     **`5a07efb8485466ef71366b94c6ae10051df22dacf606e8b6558eef204cf6e42b`**, 376 file = 368 file khóa R54 (băm lại, trùng)
     + 8 file R55 (`dieu_phoi_test40.py` `b9e7f199…3f23c047`, test `9f8a1bcd…6859af05`, hợp đồng `626dd24d…7df12c76`,
     thiết kế `9687aae7…2419a598`, `ket_qua_test.json` 24/24 `f7972339…ecab938e7a3`, 3 file tiền kiểm). Nối tới giao thức
     R54 `eb1f8f30…715431ad` và khóa R54; 36 run; module src/Ultralytics/TrackEval nạp khi import mã thực thi và RECORD
     thư viện đều đã có trong khóa R54 (không thêm file mới ngoài 8 file trên).
  3. `… --buoc kiem-khoa --khoa khoa_20261021_121500_644100_pid22868 --luu chuan_bi` → **exit 0**, 376 file + khóa R54
     368 file, 0 lệch: `results/phase2/r55_tracking/chuan_bi/kiem_khoa_20261021_121515_485600_pid17620/ket_qua.json`.
  - Không tạo `results/phase2/r55_tracking/thuc_thi/`; không cache/suy luận/tracking/chấm/GPU/`setup_model`; không sửa
    mã/cấu hình/thiết kế R55A (thiết kế đã nằm trong khóa); không sửa khóa cũ. Giới hạn import lười: khóa không khởi tạo
    predictor; module Ultralytics nạp lười trong `setup_model`/`stream_inference` chỉ có trong khóa nhờ khóa R54 (rà
    tĩnh + smoke R54B); đường thực thi kiểm module trước và sau từng đơn vị (ngoài khóa → không marker, exit 4).
  - Bàn giao: `handoff/logs/review55/02_khoa_va_ban_giao.md`. Sau khi viết, chạy lại `kiem-khoa` chỉ đọc (không
    `--luu`) → exit 0, 0 lệch (mục 0/bàn giao không thuộc khóa).
Plan do Codex đề xuất sau lượt `bytetrack_r54_test40_prep_20261020_145514` (lời gọi `0016_planner`, 21/10 05:08), người
dùng duyệt. R55 không cấp quyền cache/suy luận/tracking/chấm/đo thời gian trên dữ liệu thật; mọi lần chạy thật sau R55
cần plan thực thi riêng do người dùng duyệt. Không commit.
- **R54 đã đóng phần chuẩn bị (review thực tế, `D:/LABOTARY/automation/runs_local/bytetrack_r54_test40_prep_20261020_145514/reports/`):**
  `review_003.json` approve R54A (khóa `khoa_20261020_211635_887200_pid10816`, `khoa.json` `e2d1b0d5…a418e16b4f`);
  `review_005.json` approve smoke foreground R54B; `review_006.json` approve tài liệu Paad1df4f. **Không review nào duyệt full.**
- **Lịch sử quyết định chuyển tiếp:** P1980cb21 hoãn phép đo thời gian PF500/UKF trên đoạn dày khỏi R54, yêu cầu đưa vào
  plan lượt dài; Pc5fc63da/P482c0392 từ chối thêm bước vào R54 nhưng **không hủy** nội dung chuyển tiếp; P7e7ee3c8 từ chối
  riêng kiểm `setup_model` trên CPU trước smoke — **không** được hiểu là cho phép chạy kiểm đó ở R55 (R55 không làm).
- **R55A vòng sửa 3 — F1 còn mở + F6 (21/10 ~11:50–12:05)** theo
  `D:/LABOTARY/automation/runs_local/bytetrack_r55_prepare_20261021_051232/reports/review_002.json` (review tĩnh, verdict
  `revise`; F2–F5 đóng). Chỉ sửa write_paths R55A (`dieu_phoi_test40.py`, test, hợp đồng, mục 0, thiết kế); helper
  R54/protected không đổi; không tiêm module (P6fb1cad7 từ chối); không marker nguyên tử (P9857b26c từ chối).
  - F1 [P1]: lần xác nhận có `manifest_cache.json` hoặc marker hỏng = **có thể đã công bố** (marker hỏng sau công bố không
    phân biệt được với ngắt giữa lúc ghi marker). Khi đó lệnh `cache` **đóng băng**: chỉ tái sử dụng lần hợp lệ, video cần
    lần mới → exit 2 trước khi nạp model (`goi_detector=false`). Lần cache có marker đọc được nhưng sai hash → `cache` và
    `xac-nhan-cache` chặn cả detector (không suy luận lại để thay). Xác nhận lại phải trỏ **đúng** lần cache + SHA256
    npz/json/meta/marker của mọi bản manifest trước đọc được; khác → chặn. Các ca ngắt trước công bố C3/C4 vẫn đạt.
  - F6 [P2]: ràng buộc chấm (`rang_buoc_cham`) thêm `manifest_cache_sha256` + `mot_lan_ds` (danh sách lần tracking) cạnh
    `mot_sha256_ds`/GT/đánh giá; `lan_cham_da_chon` (chấm lại, hậu kiểm) so **toàn bộ** ràng buộc qua `chon_lan`. Manifest
    khác → lần chấm cũ không được tái sử dụng/chọn dù MOT trùng byte.
  - Fixture foreground `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py`
    **exit 0, 24/24 đạt** (295,1 s; 36 lệnh CLI tiến trình con thật):
    `results/phase2/r55_tracking/tests/lan_20261021_115928_508900_pid21264/` (`ket_qua_test.json`, `cli_log.json`). SHA256
    ghi trong kết quả = bản cuối (không sửa mã/test/hợp đồng sau lần chạy): `dieu_phoi_test40.py`
    `b9e7f19941296bee8ff236fd67ee2687313b8d6c68765d52dc0b7b013f23c047`, `test_dieu_phoi_test40.py`
    `9f8a1bcd09e7140a1ea511aea72f0cd3a585485d1d0f99c461bd0a7f6859af05`, hợp đồng
    `626dd24dd9d753dc7c370cbc7cc3a4809f3447b0e5a3e1ab1789db9d7df12c76`; thiết kế `bbdeaa8c…10bdd1d78` là bản **trước** khi
    thêm mục 13–14 — khóa R55B sẽ băm bản hiện hành.
  - Ca mới: I1 (manifest đã công bố, marker xác nhận bị cắt **sau** công bố + NPZ cache sai hash → `cache` exit 2 "sai hash",
    `goi_detector=false`, không lần mới; `xac-nhan-cache` exit 2; tracking exit 2; biến thể marker cache hỏng → `cache`
    exit 2 "đóng băng"; ảnh chụp byte cache/xác nhận không đổi). I2 (sau khi khôi phục NPZ, marker xác nhận vẫn hỏng: `cache`
    chỉ tái sử dụng; xác nhận lại → manifest SHA mới trỏ đúng lần cache cũ; tracking 1 run → 2 lần mới, MOT trùng byte lần
    cũ; `cham` → lần **mới**, không tái sử dụng lần cũ (cùng `mot_sha256_ds`, khác `manifest_cache_sha256`); `lan_cham_da_chon`/
    `cham-lai` chọn lần mới; hậu kiểm exit 3 liệt kê đúng 11 run cùng detector chưa tracking lại, đếm 25 chấm, liệt kê
    lần xác nhận marker hỏng, lần đó giữ nguyên byte). 22 ca cũ giữ nguyên và đạt.
  - Giới hạn: review/thử trên fixture 2 video × 20 ảnh; marker hỏng là cắt giả lập; chính sách "có thể đã công bố" có thể
    chặn cả trường hợp thực ra chỉ bị ngắt giữa lúc ghi marker nếu sau đó cache bị đổi — khi đó cần người dùng quyết định,
    không tự sửa. Lần thử `lan_20261021_061607_226300_pid7220` (vòng 2) giữ nguyên làm lịch sử, bị thay.
- *(Lịch sử vòng 2)* **R55A vòng sửa F1–F5 (21/10 05:55–06:21)** theo `D:/LABOTARY/automation/runs_local/bytetrack_r55_prepare_20261021_051232/reports/review_001.json`
  (review tĩnh, verdict `revise`): F1 [P1] xác nhận cache công bố bằng `HOAN_TAT.json` ghi cuối (sau `manifest_cache.json`
  + `SHA256SUMS`), lần chưa công bố giữ nguyên và xác nhận lại vào thư mục mới không gọi detector, bản công bố lệch hash
  → chặn; F2 marker JSON bị cắt = lần không hợp lệ (`hong`), xác nhận cache/chấm lại/hậu kiểm tìm tiếp; F3 kiểm
  `n_per_frame` shape (số ảnh,), không âm, tổng = N (+ shape các mảng khác); F4 `kiem_module_sau` (hiện diện **và**
  SHA256) trước marker cache/xác nhận/tracking/chấm, trước kết luận `lap-tracker`/`cham-lai`, trong hậu kiểm; F5
  `kiem_doc_lap`: đích ghi phải trong gốc vùng ghi lexical và không qua junction/symlink, kiểm trước khi tạo phiên/cấu
  hình thư viện. Chỉ sửa write_paths R55A; helper R54/protected không đổi.
  - Fixture foreground `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r55_tracking/test_dieu_phoi_test40.py`
    **exit 0, 22/22 đạt** (263,1 s; 36 lệnh CLI tiến trình con thật):
    `results/phase2/r55_tracking/tests/lan_20261021_061607_226300_pid7220/` (`ket_qua_test.json`, `cli_log.json`). SHA256
    ghi trong kết quả = bản cuối (không sửa mã/test/hợp đồng sau lần chạy): `dieu_phoi_test40.py`
    `16ab6ffce00d80e3188c60d8d16d10bb9de6904f86e40fb430ca807e58c6fe8a`, `test_dieu_phoi_test40.py`
    `7c0313b28008623fea107f96c16214f56fc2a125381453f98a85fe13e06bd2b1`, hợp đồng
    `0c8285713cd9f510e40f3af3dc459e7d2d222a19c2e1c6ce0d82cb6b93459b8c`; tài liệu thiết kế `89bdce01…4322467` là bản
    **trước** khi thêm mục 12 (bằng chứng) — khóa R55B sẽ băm bản hiện hành.
  - Ca mới/đổi: C3/C4 (marker cache cắt; ngắt xác nhận trước `SHA256SUMS`/trước marker/marker cắt → tracking bị chặn,
    xác nhận lại `goi_detector=false`; `n_per_frame` âm/thiếu phần tử giữ tổng qua CLI → xác nhận và tracking exit 2),
    C5 (2 biến thể `n_per_frame`), D1 (manifest công bố sai hash → tracking và xác nhận lại exit 2), F2/G2 (marker chấm
    cắt đứng trước lần hợp lệ, chấm lại chọn đúng lần hợp lệ), G3 (liệt kê lần marker hỏng/xác nhận chưa công bố, giữ
    nguyên byte), G4 (kiểm module sau ở mức hàm: khớp khóa; khóa trong bộ nhớ sai SHA → phát hiện; tập file = hàm R54),
    H2 (junction `chuan_bi`/`thuc_thi`/phiên/thư mục cha lần thử → exit 2, đích junction không bị ghi; liên kết do ca
    tạo được gỡ sau ca).
  - Giới hạn: **không** có ca tiêm module giả ngoài khóa trong một đơn vị (người dùng từ chối P6fb1cad7) — F4 chỉ có
    kiểm mức hàm + đường dương tính qua mọi công việc; marker cắt là giả lập (nửa đầu byte rồi `os._exit(9)`); kiểm
    module sau mỗi đơn vị nhớ kết quả `resolve` theo `__file__` nhưng vẫn băm lại từng file — chi phí trên đường thật
    chưa đo; marker công bố bị hỏng do can thiệp ngoài sau khi ghi xong được coi như chưa công bố (xác nhận lại sinh
    manifest SHA mới, không tái sử dụng lần tracking/chấm cũ) — *vòng 3 đính chính:* khẳng định "không tái sử dụng lần
    chấm cũ" sai ở vòng 2 (Codex F6), và nhánh `cache` khi đó còn thay được cache (F1 còn mở); đã sửa ở vòng 3 ở trên.
  - Lần chạy giữ nguyên, không dùng: `lan_20261021_060404_793900_pid9884` (17/22, exit 1, 631 s — mã điều phối bị sửa
    trong lúc fixture chạy, khóa fixture phát hiện SHA đổi). Phiên vòng 1 bên dưới là lịch sử, bị thay.
- *(Lịch sử vòng 1)* **R55A (21/10 05:30–05:45):** tạo `src/r55_tracking/dieu_phoi_test40.py` (SHA256 `74b9cb6d…77fd6d68`),
  `src/r55_tracking/test_dieu_phoi_test40.py` (`bdba2f7b…ab65666c`), `configs/phase2/r55_tracking/hop_dong_dieu_phoi.json`
  (`bed7ba74…83672f93`; trỏ hash giao thức R54 `eb1f8f30…715431ad` + khóa R54, 36 run, thứ tự cache → xác nhận cache →
  tracking → chấm → hậu kiểm), thiết kế `handoff/logs/review55/01_thiet_ke_dieu_phoi.md`. Bộ điều phối chỉ gọi hàm R54 đã
  review (`tao_cache`/`mo_cache`/`chay_tracker`/`cham`/`tien_kiem`) → hàm hiện có; phần mới: hợp đồng/khóa, marker
  `HOAN_TAT.json`, chọn lần thử, manifest cache bất biến, phiên, nhật ký. Đường chuẩn bị (`tien-kiem`/`khoa --tien-kiem
  --tests` tên tường minh/`kiem-khoa`) tách khỏi `thuc-thi --cong-viec …` (không mặc định chạy tất cả; đường thật cần `--duyet`).
  - Fixture foreground `…/python.exe src/r55_tracking/test_dieu_phoi_test40.py` **exit 0, 20/20 đạt** (194,2 s; 27 lệnh CLI
    tiến trình con thật): `results/phase2/r55_tracking/tests/lan_20261021_054108_288700_pid25844/` (`ket_qua_test.json`,
    `cli_log.json`), hash mã/test/hợp đồng ghi trong kết quả = bản cuối. Các lần trước giữ nguyên làm lịch sử:
    `lan_20261021_053138_647000_pid19852` (3/20, lỗi tên biến trong `khoa`), `lan_20261021_053236_257100_pid22868`
    (3/20, assert đường dẫn Windows trong test), `lan_20261021_053307_766200_pid17024` (20/20, 423,7 s, **trước** khi thêm
    kiểm môi trường cho `lap-tracker`/`cham-lai` và tăng tốc kiểm module — không dùng).
  - Ca đã kiểm (dữ liệu tổng hợp 2 video × 20 ảnh, detector giả lập, tracker/TrackEval thật trên CPU, không YOLO/
    `setup_model`/GPU): thiếu video cache → xác nhận và tracking bị chặn; NPZ/meta/JSON sai hash sau xác nhận → chặn
    trước tracking; cache sau manifest → từ chối, không gọi detector; ngắt trước/sau marker (cache, tracking) → lần dở giữ
    nguyên byte, tiếp tục chỉ làm phần thiếu; đổi backend/khóa → exit 2; đổi seed/cấu hình/backend/khóa/NPZ trong ràng
    buộc → không tái sử dụng; chấm khi thiếu video → từ chối; chấm lỗi giả lập → lần lỗi giữ nguyên, lần mới đạt; frame
    không GT/D4 qua tracker + evaluator; lặp tracker cùng cache/seed trùng byte; chấm lại cùng MOT trùng byte, MOT bị sửa →
    chặn; hậu kiểm 36/36; đường dẫn thoát vùng, trùng tên → từ chối; snapshot không ghi ngoài vùng.
  - Không suy luận/tracking/chấm dữ liệu thật, không cache test40, không GPU, không Git ghi; `git status` chỉ thêm
    `configs/phase2/r55_tracking/`, `handoff/logs/review55/`, `results/phase2/r55_tracking/`, `src/r55_tracking/` + mục 0 này.
  - Giới hạn: fixture nhỏ; khóa fixture kế thừa file khóa R54 thật (chỉ đọc) và dùng kết quả test giả trong vùng fixture;
    nhiều ca gọi `main()` trong tiến trình; module Ultralytics nạp lười chỉ có trong khóa R55 nhờ kế thừa khóa R54 (kiểm
    lại trước/sau từng đơn vị khi chạy thật).
- **Việc còn thiếu, BẮT BUỘC chuyển sang plan thực thi sau R55:** phép đo thời gian PF500/UKF trên đoạn dày **chưa làm** —
  plan thực thi phải chọn trước đoạn dày cố định (video/frame, căn cứ mật độ không dựa trên metric), dùng cache cùng khóa đã
  đủ 40 video của detector tương ứng, chạy foreground, ghi thời gian từng run, detector/cấu hình/seed, phạm vi đo và giả
  định ngoại suy; duyệt trước khi chạy; không xếp hạng/chọn tham số/đổi giao thức từ phép đo. Dự toán từ smoke (~6.774 s
  tổng: suy luận ~3.519 s, tracking ~3.247 s, chấm ≥ 8 s; quy đổi ×220,08) chỉ **tạm thời**, không thay phép đo đó.
- Quan sát smoke giữ nguyên: hai phiên smoke `yolov8n_*` khác giá trị cache dù cùng số box (MOTP lệch ~3e-8 ở ví dụ),
  `yolo11n_mixed5` trùng byte; chưa biết nguyên nhân, không suy rộng. Kiểm biên độ box/conf bằng đọc NPZ là tùy chọn ngoài
  R55, cần duyệt riêng. 173 ảnh D4 vẫn mở.
- **Bước tiếp:** Codex review R55B (khóa `khoa_20261021_121500_644100_pid22868`, `5a07efb8…4cf6e42b`). Hoàn tất R55 thì
  **dừng**: chạy thật test40 (cache → xác nhận → tracking → chấm → hậu kiểm, kèm phép đo đoạn dày ở trên) cần plan thực
  thi riêng do người dùng duyệt (`--duyet <mã plan>`); R55 không cấp quyền. Nếu review yêu cầu sửa mã/cấu hình R55A → quay
  lại R55A, tạo bằng chứng + khóa phiên mới, giữ phiên cũ. Ngân sách: theo orchestrator (4/4 vòng sửa–review đã dùng khi
  giao R55B; hết vòng → dừng hỏi người dùng).

*(Lịch sử — R54)* **R54 — tracking test40, bước R54B (`R54B_SMOKE`): smoke đã CHẠY LẠI FOREGROUND (21/10, quyết định
người dùng P1980cb21) XONG; sau đó được approve (review_005/006 ở trên); CHƯA full.** R54A đã được Codex approve.
Plan do Codex đề xuất sau lượt `bytetrack_r53c_report_full_20261020_1346` (lời gọi `0004_planner`, 20/10 14:52), người dùng
duyệt. Hai mốc: R54A (khóa, Codex review riêng) → R54B (smoke 2 video × 128 frame, chỉ sau khi R54A approve). Lượt dài
(36 run × 40 video) **cần người dùng duyệt riêng**; wrapper hiện tại không có bước full. Không commit.
- **R53C: Codex approve** (`D:/LABOTARY/automation/runs_local/bytetrack_r53c_report_full_20261020_1346/reports/review_001.json`,
  verdict `approve`, không issue). Giới hạn reviewer ghi: review tĩnh, không chạy lại đánh giá/scorer/test/`kiem-khoa`/GPU,
  không băm lại NPZ/checkpoint/toàn bộ ảnh; approve chỉ cho R53C, không phê duyệt tracking hay thao tác Git.
- **Ba quyết định người dùng (20/10 ~14:10) cho tracking test40:** (1) cả `yolov8n_coco`, `yolov8n_mixed5`,
  `yolo11n_mixed5` cùng giao thức; (2) PF500 seed 0/1/2, không PF100/PF300, không K1/K2; (3) vehicle gộp, tô xám
  `ignored_region`, c25/c10, evaluator/timeline hiện hành, không chuyển R0/R1 dev12 sang test40. Phạm vi: chỉ nhánh so bộ
  lọc ByteTrack tương ứng expA (không SORT, expB/expC, ablation, occlusion_eval/bootstrap).
- **R54A: Codex approve** (`D:/LABOTARY/automation/runs_local/bytetrack_r54_test40_prep_20261020_145514/reports/review_003.json`,
  verdict `approve`, không issue) cho khóa `khoa_20261020_211635_887200_pid10816`. Giới hạn reviewer ghi: review tĩnh,
  không chạy lại fixture/tiền kiểm/CLI/evaluator/smoke; predictor mới được khởi tạo (setup_model/forward/GPU chưa kiểm
  bằng chạy thật — R54B dưới đây là lần chạy thật đầu tiên); không băm lại toàn bộ ảnh test40.
- **R54B — Codex review lần 1 `blocked`** (`D:/LABOTARY/automation/runs_local/bytetrack_r54_test40_prep_20261020_145514/reports/review_004.json`):
  không phát hiện lỗi kỹ thuật trong đầu ra đã đối chiếu, một issue [P2] — **sai lệch lần 1: lệnh `--buoc smoke` chạy
  nền** (`run_in_background=true`, `calls/0008_coder/stdout.log` dòng 78/81) trái yêu cầu foreground; tài liệu 02 cũ chỉ
  ghi "cơ chế chờ tiến trình", chưa nêu rõ sai lệch. Ngân sách hết 4/4 vòng → hỏi người dùng.
- **Quyết định người dùng P1980cb21 (21/10 ~00:05):** KHÔNG làm bước đo thời gian tracking PF500/UKF trên đoạn dày
  (**ghi chú để đưa vào plan lượt dài sau**); thay vào đó chạy lại smoke R54B foreground, phiên mới, giữ nguyên phiên cũ
  và mọi bằng chứng cũ, không sửa khóa/mã/cấu hình. Cấp thêm 2 vòng (Pce0a4720; đã dùng 5/6).
- **R54B chạy lại FOREGROUND (21/10 00:20–00:24, Claude tự kiểm, chờ Codex review):** mỗi lệnh chạy foreground bằng
  công cụ Bash (timeout 600 s, không `run_in_background`), chờ exit trong cùng lời gọi; một lần chạy, không lỗi, không
  thử lại. Khóa `khoa_20261020_211635_887200_pid10816` (`khoa.json` SHA256 `e2d1b0d5…16b4f`).
  - `--buoc kiem-khoa --luu smoke` **exit 0** (368 file, 0 lệch; `smoke/kiem_khoa_20261021_002050_065400_pid26920/`).
  - `--buoc smoke` **exit 0**, phiên mới `results/phase2/r54_tracking/smoke/lan_20261021_002054_245100_pid16684/`
    (77,9 s; `dat` true, `loi` rỗng): kiểm trước suy luận đạt (217 module, backend `ultralytics`, predictor chưa
    `setup_model`, CUDA chưa khởi tạo); 3 cache × 256 ảnh (128 frame/video, ánh xạ như lần 1); **36 run, 72 lượt video
    ngắn**, 128 bước/video, `n_steps` {1}; chấm 36/36 hữu hạn; **9/9 lặp PF trùng byte**; **12/12 chấm lại trùng byte**;
    kiem-khoa sau smoke trong bước đạt.
  - `--buoc hau-kiem --smoke lan_20261021_002054_245100_pid16684` **exit 0, 0 lỗi** (`smoke/hau_kiem_20261021_002228_530700_pid18648/`);
    `--buoc kiem-khoa --luu smoke` cuối **exit 0** (368 file, 0 lệch; `smoke/kiem_khoa_20261021_002230_529700_pid27380/`).
  - Kiểm thêm: `--buoc hau-kiem` trên **phiên cũ** `lan_20261020_221756_712200_pid17992` exit 0, 0 lỗi
    (`smoke/hau_kiem_20261021_002344_205300_pid10296/`) → phiên cũ còn nguyên theo hash đã ghi. `git status --porcelain`
    sau chạy giống đầu lượt.
  - **So hai phiên (tính hoạt động/tái lập, không xếp hạng):** hash ảnh tự băm và `spec_sha256` trùng; số box cache
    trùng cả 6 video. `yolo11n_mixed5`: NPZ cache, MOT 12 run và 3 file metric **trùng byte** giữa hai phiên. Hai cache
    `yolov8n_*` **khác byte** NPZ (cùng số box) → MOT 24 run khác hash; `combined_metrics.csv` khác ở mức số thực nhỏ
    (ví dụ `yolov8n_coco-c25-cv`: HOTA/IDF1/FP/FN/IDSW trùng, MOTP lệch ~3e-8), `raw_fields.json` trùng ở 18/24 run, khác
    ở 6 run PF. Quan sát: suy luận GPU giữa hai tiến trình không bit-identical với `yolov8n`; **chưa kiểm nguyên nhân**,
    không thuộc tiêu chí R54B (tái lập yêu cầu là lặp tracker từ cùng cache và chấm lại từ cùng MOT). Chi tiết: tài liệu 02 mục 10.
- *(Lịch sử — lần 1, chạy nền)* **R54B smoke (20/10 22:17–22:20):** một lần chạy, không lỗi, không lần thử lại; lệnh
  smoke chạy nền (sai lệch nêu trên). Chi tiết và bàn giao: `handoff/logs/review54/02_smoke_va_ban_giao.md`.
  - `--buoc kiem-khoa --khoa khoa_20261020_211635_887200_pid10816 --luu smoke` **exit 0** (368 file, 0 lệch;
    `results/phase2/r54_tracking/smoke/kiem_khoa_20261020_221751_170500_pid24028/ket_qua.json`).
  - `--buoc smoke` **exit 0**, phiên `results/phase2/r54_tracking/smoke/lan_20261020_221756_712200_pid17992/` (86 s): kiểm
    trước suy luận đạt (217 module, backend `ultralytics`, predictor chưa `setup_model`, CUDA chưa khởi tạo); 3 cache
    mới × 256 ảnh (128 frame/video, 0 unreadable, 0 truncated; mixed5 ánh xạ `{0:2,1:7,2:5,3:3,4:1}`, coco giữ ID);
    **36 run, 72 lượt video ngắn**, mỗi video 128 bước, `n_steps` = {1}, track_buffer 30, PF N 500 / K1 tắt / K2 0,0 /
    seed thực dùng = seed run; chấm TrackEval 36/36, metric hữu hạn; **9/9 phép lặp PF500 c25 `MVI_39031` trùng byte MOT**;
    **12/12 phép chấm lại trùng byte 3 file metric**; hash cache trước/sau tracking khớp; module nạp nằm trong khóa;
    kiem-khoa sau smoke đạt.
  - `--buoc hau-kiem` **exit 0, 0 lỗi** (`smoke/hau_kiem_20261020_221931_428200_pid25720/hau_kiem.json`); `--buoc kiem-khoa --luu smoke` lần cuối
    **exit 0** (368 file, 0 lệch; `smoke/kiem_khoa_20261020_221934_613600_pid3640/ket_qua.json`).
  - `git status --porcelain` trong lúc/sau smoke giống hệt đầu lượt; không có thư mục `runs/` của Ultralytics trong repo.
    Giới hạn: `git status` không thấy thay đổi bên trong thư mục chưa track hay đường dẫn bị `.gitignore`.
  - Chỉ kết luận đường chạy hoạt động và tái lập trên mẫu smoke; **không xếp hạng detector/bộ lọc, không chọn tham số**.
    Dự toán lượt dài (~7.500 s theo quy đổi tuyến tính) là kịch bản theo giả định, không bảo đảm — xem tài liệu 02 mục 6.
  - Không tracking đầy đủ, không cache test40 đầy đủ, không bootstrap, không Git ghi.
- *(Lịch sử)* **R54A vòng sửa 2 (20/10 21:06–21:17):** Codex review lần 2
  (`D:/LABOTARY/automation/runs_local/bytetrack_r54_test40_prep_20261020_145514/reports/review_002.json`) verdict
  **revise**: đóng F1/F3/F4/F5; **F2 [P1] còn mở** — khóa thiếu `ultralytics/utils/events.py`, file mà
  `BasePredictor.__init__` nạp vô điều kiện (`add_integration_callbacks`) khi `Model.predict` tạo predictor → smoke thật
  sẽ exit 4 ở kiểm module sau smoke. Sửa chỉ trong `src/r54_tracking/` + tài liệu 01 (mục 6, 8, 9, 10.0, 11): hàm
  `khoi_tao_predictor` khởi tạo `DetectionPredictor` thật của 3 checkpoint như `Model.predict` (không `setup_model`/
  predict/warmup/GPU; `project` trong phiên, không tạo thư mục; analytics tắt; CUDA chưa khởi tạo) ở cả `khoa` và kiểm
  trước suy luận của `smoke`; `khoa.json` phiên bản 3 ghi `predictor_khoi_tao`. Giao thức JSON, thư viện, mã lịch sử
  không sửa.
  - Fixture **23/23, exit 0** (`results/phase2/r54_tracking/tests/lan_20261020_211038_548300_pid20336/`, 276,5 s): ca mới
    I6 — khóa CLI chứa `events.py`; `module_moi` của predictor đầu = đúng `events.py`; bản sao khóa **thiếu** `events.py`
    → kiem-khoa 0 nhưng smoke exit 2 trước suy luận; **sai hash** → kiem-khoa 2, smoke 2 không tạo phiên; hàm thật trong
    tiến trình: `project` ngoài vùng → lỗi, trong `tests/` → không tạo thư mục, không module Ultralytics ngoài khóa. Lần
    trước `lan_20261020_210620_308100_pid10128` 22/23 (lỗi trong chính ca I6: `gt["detector"][0]` trên dict).
  - Tiền kiểm **đạt, 0 lệch, exit 0** (`results/phase2/r54_tracking/chuan_bi/tien_kiem_20261020_211554_265400_pid2664/`):
    56.340 ảnh / 173 D4 / 56.167 frame có GT / 675.774 dòng GT / 163 vùng; 256 ảnh mẫu tự băm lại.
  - **Khóa hiện hành:** `results/phase2/r54_tracking/chuan_bi/khoa_20261020_211635_887200_pid10816/` (368 file, có
    `events.py`; `khoa.json` `e2d1b0d52e1ca45fcb98de46e273f4dc16e48a3bcc7a9be1cfc7a9a418e16b4f`, tạo 21:17:03 +0700);
    `--buoc kiem-khoa --luu chuan_bi` **exit 0** (368 file, 0 lệch;
    `chuan_bi/kiem_khoa_20261020_211706_048500_pid5468/ket_qua.json`); `--buoc kiem-khoa` (chỉ đọc) exit 0.
  - Khóa/tiền kiểm/fixture vòng sửa 1 giữ nguyên, **không dùng cho R54B**. Giới hạn mới: predictor chỉ được khởi tạo;
    import lười trong `setup_model`/`stream_inference` chỉ rà tĩnh (kiểm module sau smoke vẫn bắt, exit 4).
  - Không suy luận, không tracking thật, không GPU, không Git ghi.
- *(Lịch sử)* **R54A vòng sửa 1 (20/10 20:48–20:54):** Codex review R54A lần 1
  (`D:/LABOTARY/automation/runs_local/bytetrack_r54_test40_prep_20261020_145514/reports/review_001.json`) verdict
  **revise**, 5 lỗi: F1 [P1] `time_s` của khung mẫu tính `(i-1)/FPS` lệch 1 ulp so `DetracDataset.frames` ở 13/128 frame
  → smoke thật sẽ dừng trước suy luận; F2 [P1] khóa bỏ sót mã Ultralytics thực thi detector/ByteTrack, backend kế thừa
  env, kiểm module chỉ sau smoke; F3 [P2] `kiem-khoa` luôn ghi `chuan_bi/` (ngoài write_paths R54B); F4 [P2] `khoa` không
  đối chiếu bằng chứng tiền kiểm; F5 [P2] `hau_kiem` không băm lại meta/metric/MOT lặp. Đã sửa cả 5, chỉ trong
  `src/r54_tracking/`, giao thức R54 (thêm mục mô tả `khoa_va_kiem_tra`, không đổi tham số) và tài liệu 01 (mục 6, 8, 9,
  10.1, 11). Mã/cấu hình lịch sử không sửa.
  - Fixture **22/22, exit 0** (`results/phase2/r54_tracking/tests/lan_20261020_205046_281500_pid22480/`, 107 s): thêm hồi
    quy H1 đủ 2 × 128 frame (cách cũ lệch 13/128 mỗi video), I2–I3 (F2/F3: khóa có 157 file ultralytics + 36 trackeval +
    RECORD; backend np / sha nguồn / module ngoài khóa / cấu hình / thư viện lệch → smoke exit 2 trước suy luận; chế độ
    fixture dừng exit 5, không suy luận), I4 (F4), D1 (F5: 8 ca sửa/xóa đều bị `hau_kiem` bắt). Lần 22/22 trước đó
    `lan_20261020_204818_235000_pid18552` chạy trước khi thêm mục mô tả vào giao thức.
  - Tiền kiểm mới **đạt, 0 lệch, exit 0** (`results/phase2/r54_tracking/chuan_bi/tien_kiem_20261020_205236_960200_pid336/`):
    cùng tổng 56.340 ảnh / 173 D4 / 56.167 frame có GT / 675.774 dòng GT / 163 vùng; 256/256 ảnh mẫu tự băm lại khớp; ghi
    thêm `dau_vao_sha256` (131 file) và `ma_da_nap_sha256` (157 file).
  - Khóa vòng sửa 1 (đã thay): `results/phase2/r54_tracking/chuan_bi/khoa_20261020_205404_464000_pid5196/` (367 file, `khoa.json`
    `4e778ed1cef5b7bc837f8aff1d0fe060d59154901442983ce3a910ab4bab577f`, tạo 20:54:12 +0700, đối chiếu tiền kiểm 0 lệch);
    `--buoc kiem-khoa --luu chuan_bi` **exit 0** (367 file, 0 lệch;
    `chuan_bi/kiem_khoa_20261020_205415_917200_pid15584/ket_qua.json`).
  - Khóa vòng đầu `khoa_20261020_151705_972100_pid21744` và tiền kiểm `tien_kiem_20261020_151626_732400_pid22580` giữ
    nguyên làm bằng chứng nhưng **không dùng cho R54B**.
  - Không suy luận, không tracking thật, không GPU, không Git ghi. Giới hạn: torch/numpy/OpenCV… chỉ khóa bằng phiên bản +
    SHA256 `RECORD`; module ultralytics nạp lười ngoài danh sách chỉ bị bắt ở kiểm sau smoke (smoke exit 4, không âm
    thầm chạy); ca "nguồn Ultralytics đổi" mô phỏng bằng bản sao khóa, không sửa thư viện thật.
- *(Lịch sử)* **R54A vòng đầu (20/10 15:14–15:17):**
  - Giao thức `configs/phase2/r54_tracking/giao_thuc_test40.json` + tài liệu
    `handoff/logs/review54/01_khoa_giao_thuc_test40.md`: ma trận **khai báo** 3 detector × {c25, c10} × (CV, EKF, UKF,
    PF500-s0/s1/s2) = 36 run × 40 video; cache imgsz 640 / conf 0,05 / IoU 0,7 / max_det 3000 / FP32 / NMS theo lớp / không
    augment / mọi lớp gốc / tô xám 114; mixed5 đổi ID theo `model.names` sang COCO sau suy luận (box/conf/thứ tự giữ nguyên);
    load [2,3,5,7], conf > 0,25 / > 0,10, max_det 300; bicycle không thuộc tập lớp tracking; track_buffer 30 bước @ 25 fps,
    `elapsed_mode skip`; TrackEval `MotChallenge2DBox` HOTA/CLEAR/Identity. Khác D3/D4/D5 detector: không loại 173 ảnh,
    không biến van/others thành ignore, không áp bộ chấm AP/IoA.
  - Mã mới: `src/r54_tracking/chuan_bi_test40.py` (bước tien-kiem / khoa / kiem-khoa / smoke / hau-kiem) và
    `test_chuan_bi_test40.py`. Mã/cấu hình lịch sử không sửa.
  - Fixture **19/19, exit 0** (`results/phase2/r54_tracking/tests/lan_20261020_151541_604300_pid25164/`); lần trước 18/19
    (`lan_20261020_151435_592300_pid19428`, wrapper đọc sai tên khóa `hash_anh`/`loai_d4` của khóa R53 → đã sửa).
  - Tiền kiểm **đạt, 0 lệch, exit 0** (`results/phase2/r54_tracking/chuan_bi/tien_kiem_20261020_151626_732400_pid22580/`):
    56.340 ảnh (stat + size theo bảng hash kế thừa; 256 ảnh mẫu tự băm lại khớp), `seqLength` cộng 56.340, D4 173, frame có
    GT 56.167 = số `<frame>` XML, 675.774 dòng GT = số box XML, 163 vùng bỏ qua; 3 checkpoint khớp khóa R53.
  - Mẫu smoke chốt trước suy luận: `MVI_39031`, `MVI_39051`, frame 1–128 (256 ảnh, cùng cho ba detector).
  - Khóa `results/phase2/r54_tracking/chuan_bi/khoa_20261020_151705_972100_pid21744/` (169 file, `khoa.json`
    `662f6651e5384db56e51aa77fc00e36d3dd2ea6bcf63ccff73dfc0cd5ee1387e`); `--buoc kiem-khoa` **exit 0** (169 file, 0 lệch).
  - Không suy luận, không tracking thật, không GPU, không Git ghi.
- *(Lịch sử)* **Lượt trước: R53C (`R53C_REPORT_FULL`), đồng bộ mục 0 và báo cáo sáu lượt đánh giá detector đầy đủ** (lượt do
Codex chọn 20/10 13:45 theo ủy quyền "cho codex chọn đi" của người dùng; thứ tự A → B → C, đây là B). Một mốc duy nhất,
kết thúc tại GPT review. Chỉ ba detector: `yolov8n_coco` (baseline `models/yolov8n.pt`), `yolov8n_mixed5`,
`yolo11n_mixed5`. Không chạy lại đánh giá, không tracking, không commit.
- **R53C (20/10): mục 0 cập nhật trước, rồi viết `handoff/logs/review53/04_ket_qua_danh_gia_day_du.md`** (bảng AP
  từng lớp, mAP, chênh lệch mô tả so baseline, bằng chứng cho từng tiêu chí mục 10 của tài liệu 02, thời gian, giới
  hạn). Mọi số trong 04 đọc từ `ket_qua.json` / `theo_lop.csv` / `tong_hop.csv` trong `results/phase2/r53_detector/full/`.
  Chờ GPT review.
- **Đánh giá đầy đủ 6/6 xong (20/10 11:30–13:04), chấm lại 6/6 xong (13:04–13:18), chạy ngoài orchestrator.**
  - Bộ chạy tuần tự `D:/LABOTARY/automation/danh_gia_r53_full/chay_tuan_tu.py`; nhật ký `nhat_ky.log`,
    `trang_thai_chuoi.json` (`"ly_do": "xong 6/6, tat ca exit 0"`), `cham_lai.log`. Mỗi lượt: `kiem-khoa` exit 0 (25 file
    khóa gốc + chuỗi PL1–PL3) → WMI ReturnValue=0 → `chay_giam_sat` → exit 0, `ctrl_c` rỗng. Thứ tự DETRAC (coco → v8n
    mixed5 → 11n mixed5) rồi BDD (cùng thứ tự); BDD có bằng chứng NPZ `full/_hau_kiem/bdd_val__<tên>__npz.txt`.
  - Đầu ra `results/phase2/r53_detector/full/{detrac_test,bdd_val}/<tên>/`, `<tên>__cham_lai/`, `_giam_sat/`,
    `_hau_kiem/`. DETRAC chấm lại bằng `--cham-lai` (3× exit 0); BDD bằng `--buoc cham-lai-bdd` (3× exit 0,
    `goi_suy_luan` 0, `dau_vao_khong_doi` true, `doi_chieu_ket_qua_goc.truong_khac` rỗng).
  - Kết quả (chi tiết và chênh lệch ở báo cáo 04; so sánh chỉ mô tả, không nhân quả, không bootstrap): UA-DETRAC test
    theo D3, mAP50 (car, bus) baseline 0,8701 / v8n mixed5 0,8785 / 11n mixed5 0,8884; BDD val năm lớp, mAP50 0,2439 /
    0,3896 / 0,3787.
- **Nghiệm thu A của GPT (sau đánh giá đầy đủ) = approve**
  (`D:/LABOTARY/automation/runs_local/hoi_codex_sau_danh_gia_full_20261020/answer.json`): đạt mục 10 trên bằng chứng
  hiện có; tự đối chiếu SHA256 khóa gốc 25 / PL1 10 / PL2 13 / PL3 16 mục, hash checkpoint, danh tính dữ liệu, sáu NPZ;
  DETRAC đúng D3, BDD đủ năm lớp; chín CSV DETRAC và ba CSV BDD chấm lại trùng byte với gốc.
  - **Giới hạn kiểm tra của A:** phiên review chỉ đọc và đối chiếu; không chạy lại scorer, không băm lại toàn bộ ảnh.
    Approve không phê duyệt tracking hay commit. Phần thiếu khi đó là tài liệu (mục 0, báo cáo 04) → lượt R53C này.
- **R53B: Codex approve ở review vòng 5** (`reports/review_005.json` của lượt
  `bytetrack_r53b_lock_smoke_20261020_0356`): F2 [P2] đóng (cổng nguồn so cwd + đường dẫn adapter), test 120/120,
  SHA256 khóa gốc/PL1/PL2/PL3 khớp. Giới hạn ghi trong review: review tĩnh, không chạy lại test / `kiem-khoa` / GPU;
  cổng nguồn không chứng minh mã chưa từng bị sửa rồi khôi phục trong lượt chạy. Lịch sử các vòng ở dưới.
- **R53B (khóa giao thức, tiền kiểm, smoke): đã làm 20/10 04:07–04:23. Review Codex vòng 1 = revise (F1–F3). Đã sửa
  20/10 05:13–05:18 bằng Phụ lục 1. Review vòng 2 (`review_002.json`) = blocked: F1, F3 đóng; F2 còn mở (chưa có
  lệnh chấm lại BDD từ NPZ trên dữ liệu thật). Vòng 3 (20/10, ngân sách 3/3): không đổi mã, khóa hay tài liệu 02;
  `--buoc kiem-khoa` chạy lại exit 0; kết thúc **blocked**, chờ người dùng quyết F2.** Lượt orchestrator
  mới chỉ gồm R53B, người dùng duyệt. Lượt cũ `..._20261019_1845` dừng vì file khóa lịch bị xóa.
  - **Vòng 5 (ngân sách bổ sung, 5/5 — vòng cuối): sửa F2 [P2] của review vòng 4 bằng Phụ lục 3, làm 20/10 10:40–10:48,
    review vòng 5 = approve (`review_005.json`).** Review vòng 4 (`review_004.json`) = **blocked**: chuỗi khóa 3 mức, ba lệnh bàn giao, fixture phục
    hồi đều được reviewer tự đối chiếu khớp; còn F2 [P2]: `kiem_nguon_bdd` chỉ so tên file adapter, bản cùng tên ở checkout
    khác vẫn qua. Sửa (chỉ trong hai file mã R53B):
    - Cổng nguồn `cham-lai-bdd`: cwd lấy từ `full/_giam_sat/bdd_val__<tên>/trang_thai.json` (do `chay_giam_sat` ghi;
      `tom_tat.json` không có cwd), `lenh` hai file phải trùng; adapter trong lệnh resolve theo cwd đó phải bằng
      `src/danh_gia_detector_bdd_val.py` của repo → khác (kể cả bản sao nguyên byte) exit 2. Không còn ngầm lấy cwd = repo.
    - `--buoc phu-luc-3` (gộp với phụ lục 2 thành `buoc_phu_luc_noi(ch, k)`): khóa mức 3 nối tiếp phụ lục 2.
    - Mã: wrapper `535d53cc…56cf` → `dfa2665d…63dc`; test `8d705a88…3022` → `2c59a6e4…0a12`. Adapter, scorer không đổi.
    - Test 120/120, exit 0 (`tests_eval/lan_20261020_104602_061598_pid15436/`; nhóm X12 = 6 ca nguồn qua CLI, Q1–Q8 =
      phụ lục 3). Lần chạy trước (`lan_20261020_104531_789341_pid16144`, cùng wrapper, test cũ hơn): 119/120 vì ca X12
      "tương đối + cwd khác" trùng `--lan 9` với X0 nên bị từ chối ghi đè (exit 1); chỉ đổi số lần chạy và nhãn trong test.
    - `kiem-khoa` trước phụ lục 3: exit 2 (`chuan_bi_danh_gia.py: sha256 khac phu_luc_2`), đúng kỳ vọng.
    - Tài liệu 02 nối `## Phụ lục 3` (tiền tố = bản phụ lục 2 `b18a7636…3276`). `--buoc phu-luc-3` exit 0 lúc 10:47:08 →
      `chuan_bi/phu_luc_3/` (`SHA256SUMS` 16 file `ab1a397c…bfc8`, `khoa_phu_luc_3.json` `c8ae8221…2640`, bản phụ lục 2
      của hai file mã lấy từ snapshot review `0009_review` trong `ban_khoa_truoc/`). `kiem-khoa` exit 0: 25 file khóa gốc +
      chuỗi `phu_luc_1, phu_luc_2, phu_luc_3`. Khóa gốc, PL1, PL2, bằng chứng smoke giữ nguyên. Không commit.
    - Không chạy GPU, WMI, full, tracking, ba lệnh chấm lại thật; dữ liệu thật chỉ được băm qua cổng khóa.
  - **Vòng 4 (ngân sách bổ sung, 4/4): bước hẹp Pdaa0fd9d — Phụ lục 2, chấm lại BDD từ NPZ, đã làm 20/10 06:43–06:47;
    review vòng 4 = blocked (F2 [P2], sửa ở vòng 5).** Người dùng (Codex chọn thay theo ủy quyền khi ngủ) duyệt Pdaa0fd9d; P1517bbbd (bảng so sánh +
    DETRAC), P56df4991 (+2 vòng), Pd9248370, P479a911c vẫn bị từ chối. Thêm vào wrapper:
    - `--buoc cham-lai-bdd --model <tên>`: chỉ đọc `full/bdd_val/<tên>/du_doan_<tên>.npz`; kiểm khóa 3 mức, hash NPZ =
      bằng chứng `full/_hau_kiem/bdd_val__<tên>__npz.txt`, lệnh trong `full/_giam_sat/bdd_val__<tên>/tom_tat.json` = lệnh
      bàn giao, cổng dữ liệu đầy đủ `bv.kiem_va_doc_gt`, tên + thứ tự ảnh NPZ (`dg.doc_du_doan`); chấm bằng
      `eval_detector_5lop.danh_gia` + `bv.chan_doan` với `bv.quy_tac_hieu_luc(0.25)`; hàm dự đoán bị chặn; ghi
      `ket_qua.json`/`theo_lop.csv` vào `<tên>__cham_lai/` mới (đã có → exit 1). Không so detector, không kiểm DETRAC.
    - `--buoc phu-luc-2`: khóa phụ lục 2 nối tiếp phụ lục 1; `kiem-khoa`/`smoke`/`cham-lai-bdd` kiểm chuỗi 3 mức.
    - Mã: wrapper `6fc0c72a…94e4` → `535d53cc…56cf`; test `4e1f2235…2fbf` → `8d705a88…3022`. Adapter, scorer không đổi.
    - Test 103/103, exit 0 (`tests_eval/lan_20261020_064506_842269_pid20924/`; nhóm P, X mới; lần chạy trước trên mã
      gần cuối: 101/102 do điều kiện test P3 sai, rồi 103/103 trước khi đổi dự đoán giả lập bám GT).
    - Tài liệu 02 nối thêm `## Phụ lục 2` (tiền tố = bản phụ lục 1 `5e27b4b5…ddb5`): ba lệnh `cham-lai-bdd` cho ba detector
      **chỉ bàn giao, chưa chạy**; sáu dòng PowerShell ghi bằng chứng NPZ (`-Width 500 -NoClobber`).
    - `--buoc phu-luc-2` exit 0 lúc 06:46:55 → `chuan_bi/phu_luc_2/` (`SHA256SUMS` 13 file `195c919a…95ab`,
      `khoa_phu_luc_2.json` `2936c897…2277`, bản phụ lục 1 của hai file mã trong `ban_khoa_truoc/`). `kiem-khoa` exit 0:
      25 file khóa gốc + chuỗi `phu_luc_1, phu_luc_2`. Khóa gốc, phụ lục 1, bằng chứng smoke giữ nguyên. Không commit.
    - Không chạy GPU, WMI, full, tracking; dữ liệu thật chỉ được băm qua cổng khóa (`phu-luc-2`, `kiem-khoa`).
  - **Phụ lục 1 (sau khóa).** Khóa gốc và bằng chứng smoke giữ nguyên. Chỉ ba file đổi:
    - tài liệu 02: chỉ nối thêm sau mốc `## Phụ lục 1`, tiền tố byte = bản khóa `7d1f42e3…e2f5`;
    - wrapper: `ffef2f84…f794` → `6fc0c72a…94e4`;
    - test: `806980c6…c2f3` → `4e1f2235…2fbf`.
    Bản gốc hai file mã được chép (đã kiểm hash) vào `chuan_bi/phu_luc_1/ban_khoa_goc/`. Bước `--buoc phu-luc` exit 0
    lúc 05:17:49 tạo `chuan_bi/phu_luc_1/` (`SHA256SUMS` 10 file `2acb9db4…78cd`, `khoa_phu_luc_1.json` `82259f4e…d60e`).
    `--buoc kiem-khoa` kiểm cả chuỗi: exit 2 trước phụ lục, exit 0 sau phụ lục.
  - **F1:** fixture gọi `main()` và parser của chính hai adapter, với tham số bàn giao, cổng sha256, từ chối ghi đè và
    cổng ảnh. **F2:** fixture phục hồi khi NPZ đã lưu nhưng bước chấm lỗi. UA-DETRAC dùng `--cham-lai`. BDD có quy trình
    ghi bằng chứng NPZ; chấm lại bằng hàm cũ trùng kết quả trên fixture. **F3:** 6 dòng `cmd.exe` cho WMI đặt 7 biến
    (tắt bytecode, UTF-8, cấu hình ultralytics/matplotlib trong `full/_cau_hinh_thu_vien`, offline) và tạo thư mục
    cấu hình trước python. Fixture chạy đúng chuỗi cmd → chay_giam_sat → tiến trình con, với môi trường sạch.
  - Test `test_chuan_bi_danh_gia.py` 74/74, exit 0, trên mã cuối (`tests_eval/lan_20261020_051609_986864_pid5104/`).
    Không chạy GPU, không chạy full; ngoài fixture chỉ chạy `kiem-khoa`, `lenh-day-du`, `phu-luc`.
  - **Cần người dùng quyết:**
    - (1) **F2, điều kiện để nghiệm thu R53B** — *đã có quyết định: phương án (a), Pdaa0fd9d, làm ở vòng 4, sửa F2 [P2]
      ở vòng 5 (xem trên); chờ review Codex vòng 5. Đoạn sau là lịch sử vòng 3.* Chưa có công cụ chấm lại BDD từ NPZ trên dữ liệu thật, vì P1517bbbd bị
      từ chối. Không có CLI cũ nào làm được việc này: adapter BDD không có `--cham-lai`, còn `eval_detector_5lop.py`
      chỉ chấm sau khi suy luận. Wrapper cũng không được thêm thao tác ngoài kế hoạch. Người dùng chọn một trong ba:
      (a) duyệt riêng một bước chấm lại BDD từ NPZ (chỉ đọc NPZ, gọi `bv.kiem_va_doc_gt` + `eval_detector_5lop.danh_gia`,
      ra thư mục mới); (b) bỏ tiêu chí 7 mục 10 cho BDD: nếu bước chấm lỗi thì chạy lại adapter vào `__lan2`
      (~18–20 phút GPU/model); (c) hướng khác.
    - (2) ~~`Ultralytics/settings.json` ở gốc repo~~: kiểm lại 20/10 vòng 3 thì file và thư mục không còn (Glob trống,
      `git status` không thấy). Agent không xóa; chưa rõ ai đã xóa. Câu "chờ xóa" ở tài liệu 02 PL1.5 đã cũ, nhưng được
      giữ nguyên để khóa phụ lục 1 không đổi.
  - Mã mới: `src/r53_detector/chuan_bi_danh_gia.py` (sha256 `ffef2f84…f794`). Mã này gọi nguyên các hàm kiểm dữ liệu,
    dự đoán, scorer và D3 của adapter cũ; adapter không bị sửa.
  - Test fixture `test_chuan_bi_danh_gia.py` (`806980c6…c2f3`): 38/38, exit 0. Chạy lại sau khóa vẫn 38/38.
  - Tiền kiểm bằng cổng adapter, băm lại nội dung mọi ảnh; cả hai tập exit 0 và khớp hồ sơ R47/R43:
    - UA-DETRAC test: 40 video, 56.340 ảnh, loại D4 173, còn 56.167 ảnh;
    - BDD val: 200 video, 39.973 ảnh.
  - Mẫu smoke 64 ảnh/tập: UA-DETRAC có car + bus; BDD có đủ năm lớp.
  - Khóa lúc 2026-10-20T04:14:32, **trước** smoke: `results/phase2/r53_detector/chuan_bi/` gồm `khoa_danh_gia.json`
    (`75578d2b…9041`), `SHA256SUMS` (25 file, `e1442ea6…250d`) và `thoi_diem_khoa.json`. Chưa commit.
  - Kế hoạch khóa: `handoff/logs/review53/02_ke_hoach_khoa_danh_gia.md`. File này có sáu lệnh đầy đủ kèm hash, quy trình
    chạy riêng, dự toán và tiêu chí hậu kiểm.
  - Sáu smoke GPU tuần tự, đều exit 0 và đạt mọi kiểm: đúng checkpoint, cùng mẫu, chấm lại từ NPZ trùng, D3 đạt,
    BDD đủ năm lớp. Đầu ra: `results/phase2/r53_detector/smoke/`. AP smoke **không** dùng để xếp hạng.
  - Báo cáo: `handoff/logs/review53/03_kiem_adapter_smoke_va_ban_giao.md`.
  - *(Lịch sử lúc smoke; nay đánh giá đầy đủ 6/6 đã chạy, xem đầu mục.)* Đánh giá đầy đủ (6 lệnh) chưa chạy lúc đó;
    **tracking vẫn CHƯA chạy.** Lượt dài chạy riêng ngoài orchestrator, sau khi R53B được approve và người dùng cho phép.
- **R53A (bàn giao train): Codex approve** (`reports/review_002.json` của lượt
  `bytetrack_r53_mixed5_handoff_eval_prep_20261019_1845`). Vòng 1 có F1, F2, đã sửa 19/10 23:34. Số liệu bàn giao
  không đổi (mọi kết luận bên dưới giữ nguyên).
  - F1: test cũ `rmtree` fixture của lần chạy trước. Nay mỗi lần chạy test ghi vào thư mục mới
    `tests_handoff/lan_<thoi diem>_pid<pid>/`, không xóa, không ghi đè. Test cuối băm lại mọi file có trước lần chạy.
    Bố cục cũ (`tests_handoff/fixture/`, `dau_ra_*/`, `ket_qua_test.json` 12/12) giữ nguyên.
  - F2: `--out` chỉ nhận `ban_giao/` hoặc `tests_handoff/` (kiểm sau resolve, trước tổng hợp và ghi; `ghi_dau_ra` cũng
    kiểm). Đầu ra cũ được chép sang `ban_giao/_ban_truoc/<thoi diem>/` trước khi ghi lại, có kiểm byte.
  - Mã sau sửa: `ban_giao_train.py` sha256 `8cc69c87…e56b`, test `a818b937…9686`. Test 15/15, exit 0, chạy hai lần
    liên tiếp; lần sau kiểm nguyên 816 file có trước. Lần chạy đầu sau sửa đạt 14/15 do trùng tên thư mục mô phỏng;
    thư mục đó được giữ lại làm bằng chứng. Hai lệnh bàn giao chạy lại đều `[DAT]`, exit 0. So với bản trước, chỉ
    `tao_luc` và hash mã thay đổi; bốn CSV gộp giống từng byte.
  - Mã `src/r53_detector/ban_giao_train.py` chỉ
  đọc bằng chứng train: không train, không inference, không nạp model bằng torch. Metadata checkpoint được đọc bằng bộ
  unpickle hạn chế. Test `src/r53_detector/test_ban_giao_train.py` (vòng 1: 12/12; sau sửa: 15/15), exit 0, trên fixture
  tổng hợp. Fixture gồm
  thiếu epoch, trùng epoch sau resume, start_epoch lệch, hash lệch, kết thúc lỗi, thiếu `ket_thuc`, thiếu tài nguyên,
  results.csv lệch, best lệch, thiếu dòng `[DAT]` và unpickle không gọi hàm lạ. Lệnh `--model yolov8n` và
  `--model yolo11n` đều exit 0, `[DAT]`, 0 lỗi, 0 cảnh báo, 0 mục thiếu dữ liệu.
  Đầu ra nằm ở `results/phase2/r53_detector/ban_giao/`: `ban_giao_<model>.json`, `dev_tung_epoch.csv`,
  `checkpoint_sha256.csv`, `tai_nguyen.csv`, `nguon_sha256.csv`. Báo cáo:
  `handoff/logs/review53/01_ban_giao_train_mixed5.md`.
  - **Mục 8:** cả hai model đạt đủ 5 điều kiện của kế hoạch 02.
  - **Epoch:** mỗi model đúng 15 epoch, không trùng. Quỹ đạo YOLOv8n là lan 0 e1–7, lan 1 e8–10, lan 2 e11, lan 3 e12–15
    (3 lần resume, đều sau Ctrl+C có chủ đích, exit 130). YOLO11n chạy 1 lần e1–15, exit 0.
  - **Best:** YOLOv8n e15 (mAP50-95 0,33452), YOLO11n e13 (0,34021). Bằng chứng: fitness max, sự kiện ghi best cuối,
    và `train_metrics` cùng thời điểm lưu của best.pt chỉ khớp epoch đó.
  - **Hash checkpoint cuối** (đã bỏ optimizer) khớp file thật và JSON runner. YOLOv8n: best `542b1381…0694`, last
    `35b2ff82…8598`. YOLO11n: best `9a5e34a1…386fb`, last `db330f8f…3f57`. Bản lưu trong epoch
    (`khoi_phuc/last_eNNN.pt`, còn optimizer) khớp nhật ký 30/30. Nguồn resume khớp 3/3.
  - **Tài nguyên toàn lượt** (lấy max/min, không cộng peak). YOLOv8n: RSS PID 1.576 MiB, RSS gồm tiến trình con 4.309
    MiB, RAM máy còn trống thấp nhất 6 MiB, VRAM nvidia-smi 2.815 MiB. YOLO11n: lần lượt 2.361 / 3.992 / 6 / 3.020 MiB.
    YOLO11n mất mẫu 10.692 s lúc 18/10 19:08–22:06 (máy ngủ).
  - **Validation cuối** `danh_gia_cuoi_ultralytics` được tách khỏi val từng epoch. Khoảng sau epoch cuối (118 / 139 s)
    gồm cả lưu checkpoint và dọn dẹp, không phải thời gian thuần validation.
  - Chưa làm: review Codex R53B vòng 5 (sau Phụ lục 3), đánh giá đầy đủ UA-DETRAC test / BDD val, ba lệnh chấm lại
    BDD, tracking.

**Lượt trước: R52 huấn luyện lại detector trên dữ liệu trộn BDD100K + UA-DETRAC (quyết định người dùng
16/10/2026).** Hai mốc R52A → R52B, mỗi mốc qua review Codex riêng; **lượt R52 complete 16/10 16:36** (cả hai mốc
approve). Lượt dài 2 × 15 epoch chạy NGOÀI orchestrator (tiến trình độc lập qua WMI), xem gạch đầu dòng cuối khối này.
Quyết định: ánh xạ UA-DETRAC `van→car`, `others→truck` (giả định); xe hai bánh trên UA-DETRAC không có nhãn, chấp nhận
nhiễu và đo sau bằng AP motorcycle/bicycle trên BDD val. UA-DETRAC train = 48 video (60 trừ dev12), dev = dev12;
40 video test không bao giờ vào huấn luyện. Khởi tạo COCO, 15 epoch, siêu tham số y hệt `train_bdd5.yaml`, frame
UA-DETRAC bước 10.
- **R52A (dataset trộn):** **approve** (Codex, `calls/0005_review/validated.json` của run
  `bytetrack_r52_mixed5_train_prep_20261016_0621`; 0002/0003 cũng approve nhưng bị guard bỏ vì extension PyDev của VS Code
  xóa `.pyc` trong `src/r51_controlled/__pycache__/` — đã tắt PyDev, phục hồi `.pyc` đúng byte). Báo cáo:
  `handoff/logs/review52/01_dataset_mixed5.md`. Dataset `data/yolo_mixed5_v1/` (Git bỏ qua). Split
  `configs/phase2/detector/splits/mixed5_v1.json` (sha256 `0acc018b…664b`) gồm BDD nguyên `bdd5_train_v1` cộng
  UA-DETRAC 48 / 12, tổng train-fit 1.309 / dev 151 video. Cấu hình `configs/phase2/detector/mixed5_v1.yaml`; mã
  `src/mixed5/`. `bx.kiem_hop_dong_dataset` trả `loi: 0` (`--buoc kiem` exit 0; manifest `f2df97da…7c13`,
  data.yaml `d7828268…6285`). Train 57.043 ảnh (UA-DETRAC 6.556 = 11,49 %), dev 7.190 (UA-DETRAC 1.631). Phần BDD
  là hardlink, trùng byte với `bdd5_v1` (0 dòng lệch). Rò rỉ với UA-DETRAC test / BDD val theo sha256, video và ảnh
  nguồn: 0. Trùng train/dev: 0. UA-DETRAC tính lại từ XML: 0 lệch. Test `src/mixed5/test_mixed5.py` 14/14, exit 0.
  Số liệu: `results/phase2/mixed5/kiem_hop_dong.json`, `thong_ke.md`.
- **R52B (cấu hình, smoke, kế hoạch lượt dài):** **approve** (Codex, `calls/0007_review/validated.json`). Ý tưởng
  hoãn: `Pd003a2ec` (bản tham số hóa kiem_thanh_cong / tom_tat_luot cho mixed5).
  - Kế hoạch + bằng chứng: `handoff/logs/review52/02_ke_hoach_luot_dai.md`.
  - `configs/phase2/detector/train_mixed5.yaml` (sha256 `2dcb31ef…bdd00`) chỉ khác `train_bdd5.yaml` ở `ten`,
    `dataset.day_du`, `dataset.smoke`. Bằng chứng: `src/mixed5/so_cau_hinh_train.py` exit 0 (51/54 khóa trùng cả kiểu).
  - Tập smoke `data/yolo_mixed5_v1_smoke/` = BDD smoke của `bdd5_train.yaml` (16 / 4 video) + UA-DETRAC 3 train / 1 dev
    (seed 0), toàn hardlink. 621 / 295 ảnh. Hợp đồng `loi: 0`, manifest `92d4cd2f…289d`, rò rỉ 0. Kiểm lại sau smoke
    vẫn `loi: 0`.
  - `kiem-cau-hinh` trên dataset đầy đủ: exit 0 cho cả hai model; 7.131 batch/epoch.
  - Smoke 1 epoch GPU: exit 0 cho cả hai model, `[DAT]`, loss hữu hạn, last/best nạp lại đúng 5 lớp, đánh giá dev chạy
    được. s/batch 0,955 (v8n) / 1,141 (11n); toàn lượt 124,7 / 114,6 s; nvidia-smi max 2.442 / 2.663 MiB; RAM trống thấp
    nhất 119 / 49 MiB. Kết quả: `results/phase2/mixed5/smoke/`.
  - Dự toán 15 epoch, là **kịch bản theo giả định**: v8n ~28,5–28,9 h, 11n ~34,3 h, tổng tuần tự ~63 h (chậm hơn ~82 h);
    chưa tính dừng sớm.
  - Test `src/mixed5/test_mixed5.py`: 20/20, exit 0.
  - Lượt dài: người dùng cho phép 16/10. Khởi chạy theo kế hoạch 02 mục 6, bằng
    `handoff/logs/review52/khoi_chay_wmi_mixed5.ps1`.
- **YOLOv8n: HOÀN TẤT** 18/10 04:30:05 (`ket_thuc` `hoan_tat`, `kiem_sau_train_dat` true, exit 0, đủ 15 epoch;
  bộ tự giám sát xác nhận đạt mục 8 lúc 04:31). mAP50 / mAP50-95 dev: e12 0,483 / 0,329; e13 0,488 / 0,332;
  e14 0,489 / 0,333; e15 0,493 / 0,335 (best = e15). Bão hòa: e15 hơn e12 +0,005 mAP50-95 (< 0,01) → đã chững lại,
  phần nhích e14–15 có thể do tắt mosaic. Checkpoint `models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/`.
- **YOLO11n: HOÀN TẤT** 19/10 16:52:37 (`ket_thuc` `hoan_tat`, `kiem_sau_train_dat` true, exit 0, đủ 15 epoch; bộ tự
  giám sát 16:53:33 xác nhận đạt mục 8 và ghi "DUNG: HOAN TAT ca hai model"). mAP50 / mAP50-95 dev: e12 0,495 / 0,339;
  e13 0,496 / **0,340 (best, `best.pt` = e13)**; e14 0,494 / 0,337; e15 0,498 / 0,338. Bão hòa: best ở e13 (không phải
  e14–15), e15 thấp hơn e12 −0,001 mAP50-95 → **đã chững lại** (e14–15 tắt mosaic). So YOLOv8n (best e15 0,4925 /
  0,3345): YOLO11n best cao hơn +0,004 mAP50 / +0,006 mAP50-95 trên val dev (chưa phải đánh giá R53). Checkpoint
  `models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/`. Lịch sử khởi chạy: bộ tự giám sát tự khởi chạy sau `kiem-cau-hinh` exit 0: `yolo11n_lan0`, PID công cụ
  22328, runner 20264, `bat_dau` `moi` 04:37:18 18/10. Dự toán 34,3 h (chậm 44,7 h) → ~15h 19/10 (chậm ~01h 20/10).
  Thực tế ~2,21 h/epoch; 5/15 epoch lưu lúc 15:41 18/10, mAP50 / mAP50-95 dev e5 0,430 / 0,297 (vẫn tăng);
  e6 0,443 / 0,305. 18/10 mất điện → máy **ngủ** 19:08–22:07 (Kernel-Power 42/107, không reboot); runner, bộ tự
  giám sát (pythonw PID 20764) còn sống, train chạy tiếp epoch 7 sau khi thức (GPU 95 %). Dự kiến xong ~17–18h 19/10.
- Lượt dài YOLOv8n chạy tiếp lần 3 từ 17/10 20:59 (người dùng "tiếp tục đi"): dry-run `--tiep-tuc` exit 0
  (từ epoch 12/15, `last_e011.pt`); `-GiamSat yolov8n_tiep_tuc_3`, PID công cụ 3868, runner 7144; bộ tự giám sát
  `--bat-dau-tu yolov8n_tiep_tuc_3` (PID 20764). Dự kiến xong ~04:35 18/10.
- Tạm dừng lần 3 theo yêu cầu người dùng (17/10 19:14 "tạm dừng"; epoch 11 đang val nên dừng
  ngay sau khi lưu): tắt bộ tự giám sát (PID 9728), epoch 11 lưu 19:15:06 (mAP50 / mAP50-95 dev 0,488 / 0,331, best
  mới), Ctrl+C 19:16:22, runner thoát sau 4 s; `ket_thuc` `bi_ngat` exit 130 (lan 2). **Chạy tiếp**: dry-run
  `--tiep-tuc` → `-TiepTuc -GiamSat yolov8n_tiep_tuc_3` → `tu_giam_sat_mixed5.py --bat-dau-tu yolov8n_tiep_tuc_3`.
- Chạy tiếp lần 2 từ 17/10 17:18 (người dùng "tiếp tục đi"; máy ngủ 14:03–17:15 do lệnh
  ngủ "Application API" nên dry-run treo rồi xong 17:18, exit 0, từ epoch 11/15 `last_e010.pt`).
  `-GiamSat yolov8n_tiep_tuc_2`, PID công cụ 2872, runner 24540; `bat_dau` `tiep_tuc` lan 2 17:22:12, start_epoch
  10; bộ tự giám sát `--bat-dau-tu yolov8n_tiep_tuc_2` (PID 9728). Dự kiến xong ~02:45 18/10.
- Tạm dừng lần 2 theo yêu cầu người dùng (17/10 12:42 "tạm dừng", chọn "dừng sau epoch 10"):
  tắt bộ tự giám sát (PID 14172), epoch 10 lưu 14:01:22, Ctrl+C `dung_luot.py` 14:02:34, runner thoát sau 4 s;
  `ket_thuc` `bi_ngat` exit 130 (lan 1, epoch 8–10). mAP50 / mAP50-95 dev: e8 0,471 / 0,321; e9 0,480 / 0,327
  (best); e10 0,474 / 0,321. **Chạy tiếp**: dry-run `--tiep-tuc` → `-TiepTuc -GiamSat yolov8n_tiep_tuc_2` →
  `tu_giam_sat_mixed5.py --bat-dau-tu yolov8n_tiep_tuc_2`.
- Chạy tiếp lần 1 từ 17/10 08:16 (người dùng "tiếp tục đi"): dry-run `--tiep-tuc` exit 0 (từ
  epoch 8/15, `last_e007.pt`); `-TiepTuc -GiamSat yolov8n_tiep_tuc_1`, PID công cụ 2544, runner 25852; `bat_dau`
  `tiep_tuc` lan 1 08:20:43, start_epoch 7; epoch 8 chạy ~1,1 it/s. Bộ tự giám sát khởi chạy lại
  `--bat-dau-tu yolov8n_tiep_tuc_1` (PID 14172).
- Tạm dừng trước đó theo yêu cầu người dùng (17/10 "tạm dừng", chọn "dừng sau epoch 7"): đã tắt bộ tự
  giám sát (06:4x), epoch 7 lưu 07:39:08 (mAP50 dev 0,457), Ctrl+C có kiểm soát bằng
  `handoff/logs/review42/dung_luot.py` 07:40:19, runner thoát sau 4 s; `ket_thuc` `bi_ngat`, exit 130, epoch đã lưu
  1–7 (`khoi_phuc/last_e001..007.pt`). mAP50 dev theo epoch: 0,301 / 0,383 / 0,376 / 0,425 / 0,441 / 0,446 / 0,457
  (~6.700 s/epoch). **Chạy tiếp**: dry-run `--tiep-tuc` rồi `khoi_chay_wmi_mixed5.ps1 -Model yolov8n -TiepTuc
  -GiamSat yolov8n_tiep_tuc_1` (kế hoạch 02 mục 7); bộ tự giám sát dừng ở `bi_ngat` nên phải khởi chạy lại bằng
  `--bat-dau-tu yolov8n_tiep_tuc_1`. Lịch sử bên dưới.
- Lượt dài YOLOv8n (lịch sử): chạy từ 16/10 18:32 (`-Model yolov8n -GiamSat yolov8n_lan0`). Tiền kiểm: cắm sạc, High
  performance không ngủ, không chờ reboot, `D:` trống 44 GB, RAM trống ~0,7 GB; `kiem-cau-hinh` exit 0. PID công cụ
  4480, runner 25588; `bat_dau` 18:37:20 (`che_do_luot` moi, 15 epoch); epoch 1 lưu 20:29 (6.700 s gồm val; mAP50
  dev 0,301). Người dùng cho phép tự động hóa (16/10): `D:/LABOTARY/automation/tu_giam_sat_mixed5.py` (ngoài repo,
  WMI, nhật ký `D:/LABOTARY/automation/giam_sat_mixed5/nhat_ky.log`) tự resume khi tiến trình chết không ghi
  `ket_thuc` (dry-run `--tiep-tuc` phải exit 0, tối đa 4 lần/model) và tự khởi chạy YOLO11n khi YOLOv8n đạt mục 8
  (kèm `kiem-cau-hinh`); dừng khi `loi`, `bi_ngat` (Ctrl+C), `D:` < 15 GB hoặc 2 lần ngắt liền không thêm epoch.
  Giám sát `results/phase2/mixed5/detector_train/giam_sat/yolov8n_lan0/`; theo dõi chỉ đọc
  `results/phase2/mixed5/detector_train/theo_doi_yolov8n.log`. YOLO11n **chưa chạy**: chỉ sau khi YOLOv8n đạt mục 8
  của kế hoạch 02. Bị ngắt → resume theo mục 7 (dry-run `--tiep-tuc` trước).

**Lượt trước: R51 so sánh có kiểm soát (2026-10-16).** Trạng thái dưới đây được ghi khi lượt R51 đang chạy. Lượt R52
không kiểm hay đổi trạng thái R51. Run
`D:/LABOTARY/automation/runs_local/bytetrack_r51_controlled_20261013_1806`, ba mốc R51A → R51B → R51C; mỗi mốc
qua reviewer riêng. Backend do người dùng yêu cầu ngày 16/10: Claude `claude-opus-5-5` / `high` triển khai,
Codex `gpt-6-astra` / `high` review ở phiên riêng. Đây không phải kiểm chứng chéo kết quả khoa học giữa hai mô hình.
Mục tiêu: trên UA-DETRAC dev12, trả lời hai câu hỏi độc lập. (1) Cùng ByteTrack, CV / EKF / UKF / PF baseline và
PF K1 khác nhau thế nào. (2) Cùng bộ lọc và detection, SORT khác ByteTrack thế nào. Chỉ chạy phần còn thiếu.
- **R51A (giao thức + tái dùng):** đã hoàn tất triển khai và tự kiểm. Review orchestrator `0002_review` bị cắt (Codex
  hết hạn mức 06:17). Review thủ công cùng lệnh trên bản đóng băng (16/10 11:09–11:13,
  `.../bytetrack_r51_controlled_20261013_1806/review_thu_cong_R51A/answer.json`): **revise**, chưa approve.
  R51A-F1: cổng provenance trả True khi eval_meta thiếu hash một video (`audit_protocol.py:353`, `:422`), test chưa có
  fixture âm tính. R51A-F2: bảng kiểm kê thiếu thông số bộ lọc hiệu lực (N/K1/roughening, seed, backend, provenance
  `config.py`) và chưa nói rõ 0,04 s danh nghĩa vs dt bộ lọc = 1 bước. R51 dừng sau Mốc A theo quyết định người dùng
  16/10.
  **18/10: đã sửa F1/F2** (người dùng chọn "sửa 2 lỗi R51A", làm thủ công ngoài orchestrator vì repo đã khác mốc R51;
  chỉ trong `write_paths` của R51A, không sửa mã/cấu hình nguồn, không thay tham số, chưa chạy B). F1: `kiem_provenance`
  + `hash_dau_ra_khop` (thiếu bằng chứng = không đạt, đủ tiêu chí 1–6 mục 4.1 kể cả cache tính lại trên đĩa, backend,
  `config.py`), 1 fixture dương tính + 25 âm tính; trên dữ liệu thật đúng 16 run kc48 đạt, 10 run cũ không đạt, khớp ma
  trận. F2: cột `bl_*` thông số bộ lọc hiệu lực đọc từ instance, seed hiệu lực PF = seed runner, `config.py` sha256
  `0e3f8b75…` giống ở commit chạy / HEAD / cây làm việc; giao thức mục 2, 4.1, 5, 9, 11 cập nhật. Kết quả mục 3 và quyết
  định tái dùng không đổi (file frame_* / so_khop / ma_tran trùng byte bản đóng băng; cột cũ của kiem_ke_run không đổi
  giá trị). `audit_protocol.py` exit 0; `test_protocol.py` **19/19 PASS**, exit 0 (`-B -X utf8`).
  **Review vòng 2: APPROVE** (18/10 05:52–05:55, Codex `gpt-6-astra` high, cùng lệnh/schema 0002_review, sandbox chỉ
  đọc; người dùng giao "cần quyết định gì thì hỏi chat gpt"): `.../review_thu_cong_R51A_vong2/answer.json`, issues
  rỗng, F1 và F2 đã đóng, 17/17 hash bản đóng băng khớp. Giới hạn reviewer: chạy `test_protocol.py` được 16/19 vì sandbox
  chặn thư mục tạm của 3 fixture evaluator (không đổi từ vòng 1), không chạy `audit_protocol.py`. Giao thức:
  `handoff/logs/review51/01_giao_thuc.md`; bằng chứng `results/phase2/r51_controlled/protocol/`; script chỉ đọc
  `src/r51_controlled/audit_protocol.py`; test `src/r51_controlled/test_protocol.py`.
  - 572 ảnh dev12 không có `<frame>`: 8 đoạn, đều nằm giữa video. XML không có frame rỗng target (0/16.347). Ở cả 8
    đoạn, track GT ngay trước đều kết thúc tại biên, track GT ngay sau đều bắt đầu tại biên, 0 track xuyên qua. 3 ảnh
    xem trực tiếp có làn đường gán nhãn trống. 0,36–0,41 % box tracker rơi vào các frame này. Bằng chứng **phù hợp với**
    frame âm tính theo quy ước nhãn, chưa loại trừ sót nhãn (mới xem 3/572 ảnh). Evaluator (`MotChallenge2DBox`) tính
    mọi box ở đó là FP, và các box này **giảm cả AssA**, không chỉ DetA (fixture). Quy tắc đề xuất: R0 giữ evaluator
    hiện hành làm chính; R1 bỏ box tracker ở 572 frame làm độ nhạy; chấm đồng nhất mọi run trong namespace mới.
  - Tái dùng: 16 run `kc48_pf_dev` (ekf, pf100, pf100-k1; phụ lục k2 / pf500 / pf500-k1; seed 0/1/2) đạt đủ tiêu chí:
    module mã trùng HEAD `117d62d`, YAML, cache, GT, hash đầu ra. Đầu ra ekf / pf100 còn trùng sha256 12/12 với các
    run mốc 2 ở `a64cb75` / `1c7ce8c`. Chạy mới ở B: bt-cv, bt-ukf, sort-cv30, sort-ekf30 (module cũ lệch HEAD). Thêm 2
    run neo (bt-ekf, bt-pf100-k1 s0) phải trùng 12/12; lệch thì dừng nhánh.
  - Ở điều kiện A, `conf_min` 0,25 trùng `track_high_thresh`, nên vòng ghép điểm thấp của ByteTrack luôn rỗng. Q2 không
    đo được lợi ích của việc ghép detection điểm thấp.
- **R51B (thực thi giao thức): đã chạy 18/10 17:31–17:46; review Codex vòng 1 (thủ công, cùng lệnh 0002_review,
  `.../review_thu_cong_R51B/answer.json`) = revise, 1 issue; đã sửa; review vòng 2 **APPROVE**, issues rỗng
  (`.../review_thu_cong_R51B_vong2/answer.json`; giới hạn reviewer: tracking/TrackEval review tĩnh, chạy lại
  `test_buoc_b.py` 16/16)** (người dùng duyệt 18/10
  "chạy R51B đi"; làm thủ công ngoài orchestrator như R51A vòng 2; chỉ trong `write_paths` của R51B). Báo cáo:
  `handoff/logs/review51/02_thuc_thi.md`. Runner `src/r51_controlled/chay_buoc_b.py` (gọi đúng
  `run_experiment.stage_track` và đường chấm của `stage_eval`, chỉ đổi gốc ghi), cấu hình
  `configs/phase2/r51_controlled/r51_b.yaml`, test `src/r51_controlled/test_buoc_b.py` **16/16 PASS**.
  - R51B-F1 (sai lệch trình tự): lần chạy đầu chấm R0 rồi R1 liền nhau, phép kiểm "R0 tái tạo số cũ ≤ 1e-12" chỉ chạy
    sau cả hai (giao thức mục 8 yêu cầu kiểm R0 trước R1); số không sai nhưng trình tự sai. Sửa: `trinh_tu_cham`
    chấm R0 cho mọi run → cổng `cong_r0` (thiếu file, thiếu cột, không hữu hạn, lệch tập video, sai số > 1e-12 đều là
    lỗi) → chỉ khi đạt mới chấm R1; R0 thất bại cũng dừng trước R1. Fixture không chạy thí nghiệm thật cho ca đạt và
    9 ca lỗi + thứ tự gọi. Chạy lại `cham` bằng tái dùng dấu vân tay (không chạy lại tracking/TrackEval): nhật ký ghi
    20 R0 → cổng R0 16/16 → 20 R1; đầu ra metric không đổi.
  - Chạy mới bt-cv, bt-ukf, sort-cv30, sort-ekf30 + 2 run neo (CPU, 252 s). Run neo bt-ekf và bt-pf100-k1-s0 trùng
    sha256 **12/12** với kc48 → tái dùng 16 run kc48 hợp lệ. Chấm R0 + R1 cho 20 run (40 lần TrackEval, 557 s). R0 của
    16 run tái dùng tái tạo đúng số cũ (chênh 0). R1: FP giảm đúng số box ở 572 frame, TP/FN/IDSW giữ nguyên (20/20).
    Run mới đạt cổng `kiem_provenance` của bước A (4/4). Không ghi vào `data/processed/trackers`, `results/trackeval`
    hay `kc48_pf_dev`.
  - COMBINED điểm R0, **chưa CI**. HOTA: cv 0,5900; ekf 0,5925; ukf 0,5879; pf100 (TB 3 seed) 0,4933; pf100-k1
    0,5377; sort-cv30 0,5677; sort-ekf30 0,5652. Hiệu số HOTA: ekf−cv +0,0025; ukf−ekf −0,0047; pf100−ekf −0,0992;
    pf100k1−pf100 +0,0444; bt-cv−sort-cv30 +0,0223; bt-ekf−sort-ekf30 +0,0273. R1 đổi mọi hiệu số ≤ 0,00034, không đổi
    dấu. Đầu ra mới còn trùng sha256 12/12 với run cũ không tái dùng cùng tên (chỉ là đối chiếu thông tin).
  - Đầu ra: `results/phase2/r51_controlled/{runs,eval,tests}/` (`runs/` 380 MB, git bỏ qua).
- **R51C (tổng hợp, bootstrap CI, phân tầng): đã chạy 18/10 22:15–22:35; review Codex vòng 2 APPROVE 19/10**
  (`.../review_thu_cong_R51C_vong2/answer.json`; 1 ghi chú không chặn về làm tròn, đã sửa trong báo cáo) → **R51
  HOÀN TẤT, DỪNG** (không mở H2/H3 hay thí nghiệm mới). (Người dùng duyệt 18/10
  "Tiếp tục đi" sau câu hỏi có làm R51C không; thủ công ngoài orchestrator như A/B; chỉ trong `write_paths` của R51C).
  Báo cáo `handoff/logs/review51/03_tong_hop_va_slide.md`; script `src/r51_controlled/summarize.py` (chỉ đọc
  `eval/`, `runs/_processed/`, kết quả R48B/R48D; ghi `results/phase2/r51_controlled/summary/`); test
  `src/r51_controlled/test_summary.py` **16/16 PASS**. Chưa công bố số cho tới khi reviewer approve.
  - Review Codex vòng 1 (thủ công, `.../review_thu_cong_R51C/answer.json`) = revise, R51C-F1: tầng cua p90
    (8 track / 4 video) vẫn có tỷ lệ trong `phan_tang_bien_the.csv`. Đã sửa: cổng mẫu áp cho mọi bảng xuất, tầng thiếu
    mẫu chỉ số đếm, `ci_loai_0` để trống khi không có CI; thêm test hồi quy; số báo cáo không đổi. Giới hạn reviewer
    ghi nhận: `02_thuc_thi.md` (B) sửa "10 → 9 ca lỗi" lúc 18:04 18/10 sau B approve (theo limitations của B vòng 2).
    Review vòng 2 APPROVE.
  - Bootstrap ghép cặp theo video (n 12, B 5.000, `default_rng(0)`, một dãy chỉ số chung cho mọi đối chứng / seed;
    PF COMBINED tung seed rồi trung bình trong mỗi mẫu), CI 95 % phân vị; trùng `boot_seed.paired_bootstrap_seed`
    ≤ 1e-12. R0 chính, R1 độ nhạy: 0/40 ô (8 đối chứng × 5 metric) đổi kết luận CI.
  - R0, ΔHOTA [CI]: ekf−cv +0,0025 [−0,0018; +0,0060]; ukf−cv −0,0021 [−0,0085; +0,0010]; ukf−ekf −0,0047
    [−0,0085; −0,0008]; pf100−ekf −0,0992 [−0,1781; −0,0498]; pf100k1−pf100 +0,0444 [+0,0184; +0,0867];
    pf100k1−ekf −0,0548 [−0,0925; −0,0310]; bt-cv−sort-cv30 +0,0223 [+0,0192; +0,0274]; bt-ekf−sort-ekf30 +0,0273
    [+0,0199; +0,0327]. Tức: CV/EKF/UKF gần như ngang (UKF thấp hơn EKF rất nhỏ); PF kém rõ, K1 giúp nhưng vẫn dưới
    EKF; ByteTrack hơn SORT-kiểu cùng bộ lọc.
  - Phân tầng (định nghĩa cũ, R1 không đổi): che khuất 184 sự kiện / 12 video, che hoàn toàn 33 / 6, car nhỏ cạnh xe
    lớn 42 / 7, cua 32 track / 5 video (p90 8 / 4: chỉ số đếm). Cùng hướng bảng chính; che hoàn toàn mọi CI chứa 0;
    không kết luận "PF tốt khi cua". Run tái dùng tái tạo đúng số R48D.

**Chuyển backend tạm (2026-10-12, người dùng duyệt; lịch sử R50).** Claude báo hết tuần lúc 04:35:01 +07;
reset dự kiến 05:00 16/10, chưa xác nhận reset. Lượt `bytetrack_r50_test11_thiet_ke_deta_20261012_0356`
đã stop chính thức, `quota_wait`, `active: null` theo bàn giao; không start/recover hay sửa lượt cũ. Checkpoint:
`D:/LABOTARY/automation/checkpoints/r50_codex_switch_20261012_073357/`.
Người dùng ngày 13/10 **đã hủy pause và cho phép tiếp tục** đúng mốc R50. Run Codex cũ
`D:/LABOTARY/automation/runs_local/bytetrack_r50_codex_20261012_0738` giữ nguyên; recover bị chặn bởi báo cáo người dùng
tạo ngoài write_paths, không phải bằng chứng hoàn tất. Checkpoint tiếp nối:
`D:/LABOTARY/automation/checkpoints/resume_r50_r51_20261013_1806/before`.
Run hiện tại `D:/LABOTARY/automation/runs_local/bytetrack_r50_resume_20261013_1806`, bước `R50A_CLOSE_WITH_EVIDENCE`:
chỉ hoàn thiện ba tài liệu R50. Backend thực **Codex/Codex**, `gpt-6-astra` / `high`, triển khai và review bằng hai phiên
riêng; không phải kiểm chứng chéo hai mô hình. Không kết luận phiên Claude cũ đã hoàn tất.
Test 11 có 5/5 PASS, exit 0 theo `results/phase2/kc48_pf_code/test11_k1/ket_qua.json`, bắt đầu
2026-10-12 04:01:02 +07; 6 hash mã và hash YAML được đối chiếu lại bằng `Get-FileHash` ngày 13/10, đều khớp artifact.
HEAD thực chạy `a7a6961`, mã K1 khi đó nằm trong cây làm việc; snapshot sau đó `117d62d` đã push theo bàn giao người dùng.
K1 **code-present nhưng disabled**: YAML thiếu khóa `pf_optimal_proposal`, mặc định tắt; không phải test runner K1 bật.
Không chạy lại test. Chi tiết seed, fixture, hash và giới hạn kiểm: `handoff/logs/review48/10_test11_k1_va_trang_thai_de_xuat.md`.
Quota: đã đọc trực tiếp file `quota_latest.json` của run hiện tại: quan sát **2026-10-13T23:06:34+07:00**,
nguồn `account/rateLimits/read`, 5 giờ **4%**, tuần **32%**, `auth_type: chatgpt`, `ordinary_usage_allowed: true`.
Preflight `D:/LABOTARY/automation/checkpoints/resume_r50_r51_20261013_1806/quota_preflight.json`:
**2026-10-13T18:10:07+07:00**, cùng nguồn, 5 giờ 28%, tuần 20%. Đây là hai snapshot do orchestrator thu thập;
coder đọc file, không tự gọi API quota. Số 34% / 46% thiếu giờ của ngày 12/10 chỉ là lịch sử.
Orchestrator kiểm quota trực tiếp và ngân sách trước mỗi lời gọi mới; chặn khi 5 giờ ≥ 90% hoặc tuần ≥ 80%;
thiếu số quan sát thì dừng kiểm thủ công. Theo bàn giao ngày 12/10, đã có cơ chế tự chờ tới reset +5 phút rồi kiểm lại
quota trước khi tiếp tục đúng mốc dở; thời gian trôi qua không tự chứng minh quota đã reset. Không trả phí ngoài gói.
Ngân sách bàn giao: đã dùng 2/3 vòng sửa-review; đây là vòng sửa còn được phép, không tự tăng. Sau 2 vòng liên tiếp
không tiến triển, orchestrator dừng hỏi người dùng. Auto-plan tắt. Claude giữ `claude-opus-5-5` / `high` để quay lại.
P487415eb vẫn chỉ thiết kế. Chỉ quay lại Claude-first khi reset được xác nhận tại ranh giới mốc an toàn;
Jetson Nano tiếp tục hoãn. Trạng thái thực thi/nghiệm thu phải đối chiếu state/log của lượt tiếp nối.

**Đồng bộ GitHub (2026-10-12).** Người dùng yêu cầu commit và push toàn bộ dự án hiện tại lên
`origin/phase2-mixed-traffic`, sau đó chọn chỉ mã nguồn, cấu hình, báo cáo và số liệu; bỏ video/cache lớn.
Lượt đồng bộ này lưu các thay đổi hiện có (gồm mã K1, công cụ KC48, demo, kết quả detector và hồ sơ R49/R50)
theo trạng thái trên máy. Các ghi chú "chưa commit" bên dưới mô tả thời điểm nghiệm thu trước lượt đồng bộ này.
Commit tài liệu `a7a6961` đã có; snapshot `117d62d` ngày 12/10 đã push theo người dùng. Phiên này chỉ đọc Git.
Việc lưu Git không phải nghiệm thu mới hay phê duyệt thực thi các đề xuất còn chờ; không chạy lại thí nghiệm,
không sửa thuật toán. Dataset, weights, video/cache lớn và các file đã bị Git bỏ qua tiếp tục nằm trên máy.

**Mục tiêu hiện tại.** Chuẩn bị báo cáo ngày **17/10/2026**. Thứ tự ưu tiên do người dùng đặt (10/10/2026):
1. Làm vững **lý thuyết** KF / EKF / PF, **SIS / SIR** (lấy mẫu trọng số tuần tự, tái chọn mẫu) — gắn với mã
   `src/ekf_ctrv.py`, `src/ukf_ctrv.py`, `src/pf_ctrv.py` và số liệu đã đo.
2. **Audit dữ liệu** (UA-DETRAC, BDD100K): nguồn, split, nhãn, fingerprint.
3. **Kiểm chứng PF** (suy biến trọng số, số hạt, thiết kế nhiễu).
4. **Hoãn** tối ưu Jetson Nano (mốc 6) cho tới khi người dùng mở lại.

Kế hoạch chi tiết cho báo cáo 17/10: `handoff/KE_HOACH_BAO_CAO_20261017.md` (bản giao việc người dùng duyệt
2026-10-10, M0–M6; giữ nguyên, không sửa). Plan orchestrator chuyển đổi từ bản này: lượt R48 (mục "Việc đang làm");
tiếp nhận + bảng kế thừa (M0): `handoff/logs/review48/00_tiep_nhan_va_ke_thua.md`.

**Quyết định đã chốt** (nguyên văn ở file dẫn):
- Phạm vi + thứ tự mốc 1–6, quy tắc "hỏi trước", dev nội bộ, hai tập đánh giá cuối: các mục dưới đây của file này.
- Lớp detector 5 lớp `0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle` (`src/detector_classes.py`).
- Nhãn BDD MOT train: phương án C (snapshot HF đã ghim, vòng 34) — `handoff/REVIEW34_CHI_TIET.md`.
- Split `bdd5_train_v1` (train-fit 1.261 / dev 139 video, vòng 35) — `handoff/REVIEW35_CHI_TIET.md`.
- **D1–D5** giao thức đánh giá detector trên UA-DETRAC test: người dùng duyệt 2026-10-09 —
  `handoff/REVIEW46_CHI_TIET.md` mục 11.4; kế hoạch khóa trước suy luận:
  `handoff/logs/review47/03_ke_hoach_khoa_danh_gia_detrac_test.md` (sha256 `7028a291…`).
- Moc 2 tracking: tiền đăng ký + khóa cấu hình `configs/phase2/PREREG_moc2.md`, `configs/phase2/LOCK_moc2.md`.

**Kết quả đã xác minh** (đọc file dẫn, không chép lại số khác):
- Tracking UA-DETRAC test 40 video (mốc 2): EKF–CTRV hơn KF–CV có ý nghĩa ở `c10`, CI chứa 0 ở `c25`; PF–CTRV
  kém rõ rệt so với EKF; thứ tự ảnh hưởng Nhóm B `_fix1` — `CODEX_HANDOFF.md` mục 5 và mục 8, `PHASE2_REPORT.md`.
- PF: ESS/N ~1 % ở các lần cập nhật sau mất dấu dài; khoảng cách với EKF thu hẹp khi N 100 → 300 → 500 nhưng
  không hết — `CODEX_HANDOFF.md` mục 5 ý 7; `src/pf_diag_report.py`; `handoff/logs/pf_diag_report.log`.
- BDD transfer (COCO-video val): `results/phase2/bdd_transfer_cocovid_val/bao_cao/` (`bang_chinh.md`, `pf_seed.md`).
- Detector trên BDD val (vòng 43): `handoff/REVIEW43_CHI_TIET.md`; số liệu `results/phase2/detector_eval/bdd_val_cocovid/`.
- Fingerprint UA-DETRAC test (R47 bước 1: 40 video, 56.340 ảnh, D4 loại 173): `handoff/logs/review47/01_fingerprint_detrac_test.md`.
- Detector trên UA-DETRAC test (R47 bước 4): mAP50 (car, bus) baseline `yolov8n_coco` 0,870 > `yolov8n_ft` 0,586 /
  `yolo11n_ft` 0,564; Codex nghiệm thu **approve** sau một vòng sửa diễn giải —
  `handoff/logs/review47/04_ket_qua_danh_gia_detrac_test.md`; số liệu `results/phase2/detector_eval/detrac_test/`
  (commit `482baaf`, không gồm 3 NPZ dự đoán — chỉ trên máy, sha256 ở mục 7 báo cáo 04). Tài liệu R46 (gồm mục 11.4
  D1–D5): commit `43cd0de`.
- **R48A / M1** (2026-10-10; Claude tự kiểm; Codex cổng A: vòng 1 **revise** F1–F4, vòng 2 **approve** —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_002.json`; chưa commit): lý thuyết KF–CV (XYAH 8-D),
  EKF–CTRV, PF–CTRV, SIS / SIR, ESS, log-sum-exp có phương trình, đơn vị bước, góc, ω ≈ 0 và trạng thái nguồn —
  `handoff/logs/review48/01_ly_thuyet_kf_ekf_pf.md` (ByteTrack: đã đối chiếu toàn văn tr. 2–7; Arulampalam 2002: chỉ
  metadata, mọi ý ghi "chưa đối chiếu toàn văn"). Bảng khái niệm → `file:hàm:dòng` tại `eaf2e87`, giả thuyết H1–H7 —
  `handoff/logs/review48/02_doi_chieu_code_pf.md`. **Sở hữu trạng thái**: mỗi track PF có mảng particle, log-trọng số và
  rng riêng; chỉ `PFCTRV` (cấu hình + bộ đếm chẩn đoán) dùng chung — đọc mã + `src/kc48/kiem_so_huu_hat.py` exit 0
  (`results/phase2/kc48_pf_code/so_huu_hat.json`); "cạnh tranh hạt toàn cục" không áp dụng cho cấu trúc mã hiện tại.
  `src/test_pf_ctrv.py` 26/26 PASS. Không sửa mã tracker. Vòng sửa 1 (chỉ tài liệu, theo
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_001.json`): F1 σ_θ, σ_ω là độ lệch
  chuẩn gia số mỗi predict, đơn vị [rad], [rad/bước] (bảng đơn vị từng thành phần Q ở `01` mục 3; chú thích mã
  `src/config.py:304-305` ghi [rad/frame], [rad/frame²], chưa sửa); F2 tách mô hình lý tưởng khỏi 6 heuristic triển
  khai, phép chặn h ≥ 10⁻³ dồn khối xác suất vào biên chứ không phải Gauss cắt cụt; F3 `pf_estimate: map` là particle
  trọng số lớn nhất, không phải MAP (không cấu hình nào tại HEAD dùng `map`); F4 ghi nhận lệch phạm vi ngay dưới.
  Danh sách chỗ sửa: `01` mục 10, `02` mục 7.
- **Lệch phạm vi R48A** (F4): lệnh `src/test_pf_ctrv.py` (test 11 sẵn có) đã xóa và ghi lại
  `data/interim/det_cache/fixture_pf_seed_test/`, ghi `data/processed/trackers/DETRAC-sample/p2-pftest-s{0,1}/`, ngoài
  `write_paths` của R48A. Log thực thi:
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\calls\0001_coder\stdout.log` dòng 162 (lệnh), 165
  (kết quả). Không xác minh được trạng thái trước lệnh (đường dẫn bị git bỏ qua, không có hash trước); không xóa, khôi
  phục hay chạy lại. Chi tiết: `02` mục 6.1.
- **R48B / M2** (audit 2026-10-10; Claude tự kiểm. Codex cổng B vòng 1 **revise** F1–F4 —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_003.json`. Sửa hậu audit 2026-10-11,
  Claude tự kiểm. Vòng 2 **revise** —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_005.json`: F1–F4 đã đóng, thêm F5
  (ghi ngoài phạm vi, chỉ yêu cầu ghi nhận; xem "Lệch phạm vi R48B" ngay dưới). Đã ghi nhận F5; vòng 3 **approve** —
  `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\reports\review_007.json`; chưa commit). Báo cáo
  `handoff/logs/review48/03_audit_du_lieu.md` (sửa hậu audit ở mục 13, F5 ở mục 13.3), số liệu
  `results/phase2/kc48_audit/*.csv`, script chỉ đọc `src/kc48/audit_du_lieu.py`. Tiêu chí
  (quỹ đạo, mẫu trực quan, gần trùng, xe nhỏ cạnh xe lớn) công bố trước khi tính:
  `results/phase2/kc48_audit/tieu_chi_cong_bo.json`, sha256 `d7543fff…`, không đổi khi sửa.
  - Mốc slide lần 2 khớp khi đếm lại từ 2–3 nguồn: BDD train-fit 1.261 / 50.487, dev nội bộ 139 / 5.559, BDD val
    200 / 39.973 (`moc_doi_chieu.csv`). Tập chỉ có vài chục video: UA-DETRAC dev 12, test 40, train 60.
  - Toàn bộ 458.183 ảnh đọc được: sha256, header, SOI / EOI 0 lỗi. Giải mã 6.783 ảnh mẫu: 0 lỗi. 320/320 mẫu YOLO khớp
    nhãn gốc. Với ảnh tô xám, mẫu này chỉ kiểm siêu dữ liệu; **chưa kiểm pixel**. 1.400/1.400 file nhãn BDD train khớp
    manifest nhập vòng 34.
  - Trùng sha256 chỉ trong cùng video: 3 nhóm frame đứng hình, gồm 2 nhóm ở BDD train-fit và 1 nhóm ở BDD val. Có 3 cặp
    gần trùng, đều xác nhận là cùng cảnh; 1 cặp xuyên train-fit / dev BDD. 70/139 video dev có chung tiền tố tên với
    train-fit. Giữa BDD train / val và DETRAC train / test **chưa phát hiện** chồng lấn trong phạm vi đã kiểm (sha256
    toàn bộ ảnh, dHash 4 frame mỗi video, tiền tố tên). Video là đơn vị chia split, không chắc là đơn vị độc lập thống kê.
  - **F1, lỗi mã quỹ đạo (đã sửa)**: trước đây track `nghi_nhieu` (max Δψ > 150°) vẫn nằm trong tập hợp lệ, và p90 phân
    loại độc lập. Nay track này bị loại khỏi tập hợp lệ của cả max lẫn p90; đã chạy lại 4 phần `nhan` và `tong_hop`.
    Kiểm bằng `src/kc48/kiem_loc_quy_dao.py`: trước sửa exit 1 (4.687 track vi phạm); sau sửa exit 0, 21/21
    (`results/phase2/kc48_audit/kiem_loc_quy_dao.json`). So hash trước / sau: chỉ đổi đầu ra quỹ đạo, nhật ký chạy và
    một ô ví dụ của `chat_luong.csv` (`03` mục 13). BDD sau sửa: 37,5–42,0 % track hợp lệ đổi hướng trên ảnh ≥ 30° theo
    max, 28,2–32,2 % theo p90; bản trước ghi khoảng 30 %. DETRAC đổi ≤ 15 track mỗi tập.
  - Augmentation thực chạy: mosaic 1,0 (tắt 2 epoch cuối), translate 0,1, scale 0,5, HSV, fliplr 0,5. Albumentations
    không chạy vì không cài gói. `auto_augment` / `erasing` chỉ dùng cho classify.
  - UA-DETRAC dev12: 32 track cua (8 theo p90), 14 track bus, 88 sự kiện xe nhỏ cạnh xe lớn. Dev12 không có track
    `nghi_nhieu`, nên sửa F1 không đổi các số này.
- **Lệch phạm vi R48B** (F5): lời gọi sửa hậu audit đã ghi hai file memory của Claude Code, ngoài repo và ngoài
  `write_paths` của R48B: tạo `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\ket-luan-tu-nguon-day-du.md`
  (`Write`) và chèn 1 dòng trỏ vào `C:\Users\Mikuno\.claude\projects\D--LABOTARY-bytetrack\memory\MEMORY.md` (`Edit`).
  Log thực thi: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_1825\calls\0010_coder\stdout.log` dòng
  842–843 (tạo), 844–845 (sửa). Không xác minh độc lập được trạng thái trước sửa: vùng này ngoài git, orchestrator
  không chụp hash, chỉ có tự khai của công cụ ở dòng 843, 845. Không đổi file nào trong repo. Không sửa, xóa hay khôi
  phục memory. Chi tiết: `03` mục 13.3.
- **R48C / M3 + tính đúng M4** (2026-10-11; Claude tự kiểm; chưa commit). Lượt cũ: Codex trả **BLOCKED** vì test 11
  chưa chạy. Người dùng quyết định ngày 11/10: cho test 11 ghi fixture và gộp việc truy nguyên lệch EKF vào R48C. Bản
  sửa theo quyết định này được Codex cổng C **approve**, không có issue —
  `handoff/logs/review48/nghiem_thu_r48cde/0002_review_answer.json` (bản sao nguyên văn, nguồn + sha256 ở `MANIFEST.md`
  cùng thư mục). Báo cáo `handoff/logs/review48/05_dung_dan_pf_va_smoke.md` và `05b_truy_nguyen_lech_ekf.md`.
  Giới hạn Codex ghi trong biên bản: review tĩnh, không chạy lại test hay smoke (kết quả thực thi kiểm từ log và
  artifact); ngoài fixture, snapshot chỉ kiểm kích thước và `mtime_ns`, không hash toàn bộ nội dung; đối chiếu EKF chỉ
  bao điều kiện A trên 2/12 video, tái lập seed qua runner chỉ kiểm fixture tổng hợp 40 frame — không suy rộng sang bộ
  lọc, video hay chỉ số khác.
  - Giao thức kiểm chứng PF `handoff/logs/review48/04_giao_thuc_kiem_chung_pf.md` viết **trước** mọi so sánh bộ lọc.
    Khóa bằng sha256 `7063c532…`: chụp 04:29:39, chụp lại 06:51:50 trước smoke và 15:12:53, cả ba trùng nhau
    (`results/phase2/kc48_pf_code/giao_thuc_sha256.json`). Phụ lục P1 rút ngoại lệ "trừ test 11" ở mục 10. Sau P1,
    file có sha256 `479497fc…`; lần chụp 4 kiểm 22.510 byte đầu vẫn là `7063c532…`, tức mục 1–12 không đổi.
  - **Test 11 đã chạy đủ**: `python src/test_pf_ctrv.py` exit 0, 26/26, gồm "chạy lại seed 0: file txt
    bit-identical". `results/phase2/kc48_pf_code/fixture_{truoc,sau}.json` cho thấy chỉ 3 thư mục fixture được phép
    đổi; 24.164 file khác trong `data/interim` + `data/processed`: đổi 0, thêm 0, mất 0. `data/synth.txt` của cả hai
    seed trùng sha256 với bản có từ trước.
  - **EKF tái tạo trùng byte mốc 2** (`05b`; `results/phase2/kc48_ekf_tai_lap/so_sanh_dong.json`): file tracking EKF
    điều kiện A của smoke (`eaf2e87`) và của run mốc 2 (`1c7ce8c`) trùng từng byte trên `MVI_20052` và `MVI_20062`;
    0 frame lệch. Nhận định "khác nhau" trước đây là **sai**. Nó đến từ quy ước hash: commit `935a445` đổi
    `run_fingerprint.sha256_file` sang hash `.txt` sau chuẩn hóa CRLF → LF, nên hai `eval_meta.json` khác đời ghi hash
    khác nhau. Mới kiểm 2/12 video, chỉ EKF, điều kiện A.
  - Giao thức chốt: mô phỏng biết liên kết gồm 8 kịch bản, trong đó có 4 kịch bản lệch mô hình; pipeline dev12; K1
    proposal tối ưu (mới, tùy chọn); K2 roughening (đã có trong mã, không phải đóng góp mới); không đổi association;
    bootstrap theo video / seed nhiễu.
  - Cache dev trùng khít mốc 2 (`results/phase2/kc48_pf_code/cache_dev.json`). Bộ mô phỏng
    `src/kc48/mo_phong_quy_dao.py --tu-kiem` đạt 10/10.
  - Test, tất cả exit 0, chạy lại ngày 11/10 sau 15:12: `src/kc48/test_pf_dung_dan.py` 29/29; `src/test_pf_ctrv.py`
    đủ 11 test 26/26; EKF 14/14, UKF 12/12, lifecycle 20/20, init 17/17.
  - **Không sửa mã tracker**: không có test hỏng do mã. Lần chạy đầu của test mới hỏng 2 phép kiểm, cả hai do thiết kế
    test (`05` mục 3).
  - Smoke `kc48_pf_smoke` (2 video dev; ekf, pf100 × 2 seed, pf300, pf500) qua wrapper `src/kc48/chay_thi_nghiem_kc48.py`
    (đổi gốc `PROCESSED_DIR` vào `results/phase2/kc48_pf_smoke/_processed/`): track và eval đều exit 0.
  - Thời gian: ms / frame `ekf` 1,62, `pf100` 3,48, `pf300` 4,78, `pf500` 5,90 (`results/phase2/kc48_pf_smoke/thoi_gian.json`).
    Kịch bản dự toán cho P1–P3 của R48D: 37–55 phút, trong ngân sách 120 phút. Chỉ số của smoke không dùng cho kết luận.
- **R48D / M4 + M5 (thiết kế) + M6** (2026-10-11; Claude tự kiểm; cổng Codex D vòng 1 **revise** F1–F3 —
  `handoff/logs/review48/nghiem_thu_r48cde/0006_review_answer.json`; vòng 2 **approve** —
  `handoff/logs/review48/nghiem_thu_r48cde/0009_review_answer.json`; chưa commit). Giới hạn Codex ghi ở vòng 2: review
  tĩnh, không chạy lại test, tracking hay mô phỏng; bằng chứng không ghi đè kết quả cũ vẫn dựa trên mtime, hash sao lưu
  chỉ xác nhận các bản sao R48D đã liệt kê; mẫu số ESS / resampling khi bám là `n_upd_bam` − `n_reinit_bam`, thống kê
  tổ tiên chỉ dùng các lần systematic resampling (câu tóm tắt trong `07` mục 11 đã được R49A làm rõ); test runner seed 11
  chưa chạy lại sau K1, đề xuất này vẫn chờ người dùng. Issue nhỏ không chặn: đường dẫn JSON so sánh dev ở `07` mục 10 thiếu
  `phan_tich/` — R49A đã sửa thành `results/phase2/kc48_pf_dev/phan_tich/so_sanh_truoc_sau_vong_sua.json`. Tổng hợp
  `handoff/logs/review48/07_tong_hop_bao_cao_17_10.md`, đề cương slide `08_de_cuong_slide.md`, thiết kế M5
  `06_thiet_ke_augmentation.md` (chỉ thiết kế, không huấn luyện). Giao thức `04` chụp lại trước R48D lúc 15:31:12:
  22.510 byte đầu vẫn `7063c532…` (`results/phase2/kc48_mo_phong/giao_thuc_sha256_r48d.jsonl`); lệch ghi ở phụ lục P2.
  - Tùy chọn mới: K1 `pf_optimal_proposal` (proposal tối ưu, mặc định tắt) trong `src/pf_ctrv.py`, `src/tracker_ctrv.py`;
    K2 dùng khóa `pf_roughening` có sẵn (không phải đóng góp mới). `src/kc48/test_k1_k2.py` 16/16, exit 0: mặc định
    bit-identical với mã `eaf2e87`. File tracking `ekf`, `pf100-s0/s1` dev trùng sha256 với smoke R48C trên 2 video
    (`results/phase2/kc48_pf_dev/phan_tich/mac_dinh_vs_smoke.json`). Test cũ exit 0; test 11 **không chạy lại** sau K1
    (ghi ngoài `write_paths` của R48D).
  - Mô phỏng (`results/phase2/kc48_mo_phong/`, 30 seed nhiễu ghép cặp × 3 seed PF): `pf100` kém `ekf` nhất quán (F, e_r),
    cả nhóm lệch mô hình. K1 giảm nhất quán F, e_r và triệu chứng nghèo hạt khi đang bám, nhưng không sửa sụp ESS sau mất
    đo dài. K2 chỉ giảm e_r. Không biến thể PF nào đạt EKF.
  - Pipeline dev12 (26 run, `results/phase2/kc48_pf_dev/`, `results/trackeval/DETRAC-all/p2-kc48_pf_dev-*`; bootstrap
    theo video): K1, K2, K1+K2 cải thiện `pf100` có CI loại 0. Mọi biến thể PF, kể cả `pf500-k1`, vẫn kém EKF (HOTA
    −0,038 [−0,065; −0,021]). EKF đã chỉnh không khác EKF mặc định. Tầng xe nhỏ cạnh xe lớn (42 sự kiện / 7 video):
    K1 không phân biệt được với EKF (CI chạm 0), cỡ mẫu rất nhỏ. Không chạy trên tập test / BDD val.
  - Không ghi đè kết quả cũ: `src/kc48/kiem_khong_ghi_de.py` exit 0 (bằng chứng mtime,
    `results/phase2/kc48_pf_dev/kiem_khong_ghi_de.json`).
  - **Vòng sửa sau cổng D vòng 1** (Codex **revise** F1–F3,
    `D:\LABOTARY\automation\runs\bytetrack_r48_tiep_effort_high_20261011_1507\reports\review_003.json`, bản sao
    `nghiem_thu_r48cde/0006_review_answer.json`; Claude tự kiểm 2026-10-11; cổng D vòng 2 **approve**, `0009`: F1, F2,
    F3 đã đóng). Không chạy lại tracking. Bản trước sửa sao nguyên byte ở
    `results/phase2/{kc48_pf_dev,kc48_mo_phong}/**/_ban_cu_truoc_vong_sua/` (có `manifest.json` sha256). Chi tiết `07` mục 11,
    phụ lục `04` P3.
    - F1: phân tầng ghép bằng `occlusion_eval.match_frame` (ràng buộc IoU ≥ 0,5 trước Hungarian). Ca MVI_20052 / frame 445:
      GT 38 → 290 (trước: 350). 167/30.810 dòng track-run đổi; tầng lệch tối đa 0,0005; không tầng nào đổi trạng thái CI.
    - F2: tái khởi tạo tách khỏi ESS và systematic resampling; n_deg theo từng quỹ đạo, có mẫu số. Chạy lại có chọn lọc
      12.235 quỹ đạo PF có tái khởi tạo ngoài t_r + 1.838 quỹ đạo đối chứng, cùng cấu hình/seed (169 s, 6 tiến trình).
      Đầu ra bộ lọc trùng khít bản cũ (`results/phase2/kc48_mo_phong/chan_doan_v2/so_sanh_ban_cu.json`, 80/80 đơn vị).
      ESS/N tại t_r của `pf100` ô `cua_deu` L 20 h0 24: 0,011 trên 47/90 quỹ đạo hợp lệ (43 tái khởi tạo); trước ghi 0,006.
      C3 (K2) ESS/N tại t_r: 10 ô CI > 0 trước sửa → 0 ô sau sửa. Các kết luận C1–C3 về F, e_r không đổi.
    - F3: COMBINED tính riêng từng seed rồi trung bình, trong mỗi mẫu bootstrap (`src/kc48/boot_seed.py`). Bảng chính lệch
      tối đa: HOTA 2,8·10⁻⁵, AssA 8,7·10⁻⁵. Không đối chiếu nào đổi trạng thái CI.
    - Test: `src/kc48/test_sua_r48d.py` 29/29 exit 0; `test_k1_k2.py` 16/16, `test_pf_dung_dan.py` 29/29,
      `kiem_khong_ghi_de.py` exit 0.
- **R48E / kiểm pixel ảnh tô xám** (2026-10-11; mốc người dùng duyệt qua email 11/10; Claude tự kiểm; chưa commit).
  Nguồn ghi mục này: biên bản Codex — vòng 1 **revise** F1 (diễn giải vượt bằng chứng về nguyên nhân 3 ảnh không đạt) và
  F2 (mô tả quá rộng cơ chế khóa tiêu chí) — `handoff/logs/review48/nghiem_thu_r48cde/0013_review_answer.json`; vòng 2
  **approve**, F1 và F2 đã đóng, không issue mới — `handoff/logs/review48/nghiem_thu_r48cde/0015_review_answer.json`.
  Báo cáo `handoff/logs/review48/03_audit_du_lieu.md` mục 14; số liệu `results/phase2/kc48_audit/pixel_to_xam_*`
  (`pixel_to_xam_tung_anh.csv`, `pixel_to_xam_tom_tat.json`, `pixel_to_xam_chan_doan_K.{csv,json}`,
  `pixel_to_xam_nhat_ky.jsonl`), tiêu chí công bố trước `results/phase2/kc48_audit/tieu_chi_pixel_to_xam.json` (sha256
  `4f62d165…`), script `src/kc48/kiem_pixel_to_xam.py`.
  - Mẫu: đúng 160 ảnh `to_xam` trong mẫu 320 ảnh của `03` mục 6.3 (4 tầng × 40 ảnh). Kết quả theo ảnh: **131 đạt / 26
    chưa kết luận / 3 không đạt**; theo tiêu chí T1 134 / 0 / 26, T2 160 / 0 / 0, T3 83 / 3 / 74 (đạt / không đạt / không
    áp dụng) — Codex tự đếm lại CSV khớp JSON (0013, 0015).
  - 3 ảnh không đạt (đều ở T3, MAE > 2,0): sai số **phù hợp với** mất mát do nén lại JPEG; **chưa chứng minh** nguyên nhân,
    chưa loại trừ lỗi xử lý vùng. Ngưỡng không sửa sau khi xem; 3 ảnh vẫn ghi `khong_dat`.
  - Cơ chế khóa tiêu chí: `--phan tinh` chỉ tự kiểm chuỗi dung sai xám và bán kính lõi so với file tiêu chí; ngưỡng
    T1 / T2 / T3 là hằng số trong mã, không đối chiếu tự động (khớp file tiêu chí khi đối chiếu thủ công).
  - Giới hạn Codex ghi: review tĩnh, không chạy lại phép đo, hậu kiểm JPEG hay test; không có snapshot toàn bộ `data/`
    trước–sau để chứng minh độc lập dữ liệu không đổi (chỉ kiểm đường ghi trong mã và lệnh đã lưu); kết luận đo **chỉ áp
    dụng cho mẫu 160 ảnh**, không suy ra toàn bộ 18.600 ảnh tô xám.
  - **P9a6c2078 — đóng: đã hoàn thành qua R49 / đề xuất trùng**, theo quyết định có điều kiện của người dùng ở R50
    ngày 12/10. Đối chiếu: mục R48E trên đã ghi biên bản `0015`, kết quả 131/26/3, giới hạn mẫu 160 ảnh và giới hạn
    nguyên nhân; `handoff/logs/review48/nghiem_thu_r49/0004_review_answer.json` approve phần đồng bộ mục 0.
    Hồ sơ gốc `nghiem_thu_r48cde/P9a6c2078_ho_so.json` ghi chưa quyết định ở thời điểm cũ, được giữ nguyên.
    Đóng đề xuất trùng không phải bác bỏ nội dung nghiên cứu hay duyệt thêm thí nghiệm.
- **Kiểm chẩn đoán PF mốc 2: tái khởi tạo có trộn vào ESS không** (bước P9a8ee8bf, người dùng duyệt qua email; 2026-10-11;
  chỉ đọc, không sửa kết quả cũ; chưa commit) — `handoff/logs/review48/05c_kiem_pf_diag_moc2.md`; số liệu
  `results/phase2/kc48_pf_code/ess_moc2_tach_tai_khoi_tao/` (`ket_qua.json`, `theo_{run,tracker,n_lost}{,_test}.csv`).
  Về công thức, `src/pf_diag_report.py` **có** trộn dòng suy biến vào ESS / tỉ lệ resample. Trên số liệu mốc 2 thì
  **không** bị trộn: dev 18 run, **0/1.664.036** update là tái khởi tạo; tính lại chỉ trên update hợp lệ trùng số cũ,
  lệch tối đa **1,1·10⁻¹⁶** (sai số làm tròn). `handoff/logs/pf_diag_report.log` và các trích dẫn ESS ở `01`, `02`, `04`
  không phải sửa. Test UA-DETRAC (6 run, 0/3.400.029 tái khởi tạo) là số mới, chỉ để báo cáo, không dùng chọn tham số.

**Lịch sử R48 đã hoàn thành:** bản giao việc 17/10 theo 4 mốc, mỗi mốc một cổng Codex:
`R48A_LY_THUYET_DOI_CHIEU_PF` (M1) → `R48B_AUDIT_DU_LIEU` (M2) → `R48C_GIAO_THUC_VA_DUNG_DAN_PF` (M3 + test tính
đúng) → `R48D_THI_NGHIEM_VA_TONG_HOP` (M4, M5 chỉ thiết kế, M6), thêm mốc `R48E_KIEM_PIXEL_TO_XAM` (người dùng duyệt qua
email 11/10). Cả năm mốc đã được Codex approve (xem tiến độ dưới). Báo cáo của lượt: `handoff/logs/review48/`.
Hồ sơ lượt: `D:\LABOTARY\automation\runs\bytetrack_r48_bao_cao_20261017_*` (R48A, R48B) và
`D:\LABOTARY\automation\runs\bytetrack_r48_tiep_effort_high_20261011_1507` (R48C, R48D, R48E). Biên bản Codex của R48C–E
sao nguyên văn ở `handoff/logs/review48/nghiem_thu_r48cde/` (`MANIFEST.md`: nguồn + sha256). Tiến độ:
- **R48A** approve ở vòng 2.
- **R48B** approve ở vòng 3 (`review_007.json`).
- **R48C** approve (`0002_review_answer.json`), sau BLOCKED ở lượt cũ (test 11) và quyết định người dùng 11/10.
- **R48D** approve ở vòng 2 (`0009_review_answer.json`), sau revise F1–F3 (`0006_review_answer.json`).
- **R48E** approve ở vòng 2 (`0015_review_answer.json`), sau revise F1–F2 (`0013_review_answer.json`).
- **R49A** (2026-10-12, người dùng duyệt 12/10): chỉ đồng bộ hồ sơ — mục 0 này và đường dẫn / mẫu số trong
  `handoff/logs/review48/07_tong_hop_bao_cao_17_10.md`; không chạy lại gì, không đổi số liệu, không mở mốc mới.
  **Đã approve**, không issue: `handoff/logs/review48/nghiem_thu_r49/0004_review_answer.json`; SHA256
  `789b5b44ec670ee9607cba10f32c6bd6470f71dd0d97b22f401d47d526b912cb` khớp `MANIFEST.md` cùng thư mục.
  Review tĩnh, không chạy lại thực nghiệm; không dùng approve R49A thay review R50A.
- **R50A đã approve** (phiên Codex riêng, không chặn):
  `D:/LABOTARY/automation/runs_local/bytetrack_r50_resume_20261013_1806/reports/review_002.json`
  (`verdict: approve`; F1–F6 đã đóng). Biên bản ghi một issue không chặn: `09_thiet_ke_phan_tich_deta_pf_ekf.md:64`
  ghi "9 YAML", thực tế là 10. File này nằm trong vùng protected của R51 nên chưa sửa. Giới hạn reviewer: review
  tĩnh, không chạy lại test11, kiểm kê hay tracking. Nội dung triển khai, giữ nguyên như sau:
  Test11 và kiểm kê đã có;
  phiên tiếp nối chỉ sửa mục 0 này, `09_thiet_ke_phan_tich_deta_pf_ekf.md`, `10_test11_k1_va_trang_thai_de_xuat.md`.
  Đã đọc trực tiếp handover lượt Claude cũ và tự tính SHA256
  `dee2e42fb368aff6e6b752bd759f98e5c4f7f0fee27f0674d7fcb9b00c8d0210`; đường dẫn và kiểm tra trong `10` mục 4–5.
  Claude trước đó mất kết nối rồi hết quota, chưa trả báo cáo JSON hoàn tất. Không suy rằng mốc đã được nghiệm thu.
  Đầu vào từng bị thiếu nay đã đọc được: nguyên văn `reports/review_001.json` của run Codex cũ (F1–F6), quota kèm giờ,
  bản đầu mốc Claude gốc, Codex cũ và run tiếp nối. Đã đối chiếu bằng chứng test11/R49 và sửa hồ sơ theo F1–F6;
  bảng finding → vị trí sửa → tự kiểm ở `10` mục 5. Diff ba tài liệu đã đối chiếu cả ba snapshot; phần ngoài mục 0
  trùng byte, báo cáo tiến độ 13/10 giữ nguyên. Đây là trạng thái triển khai, **chưa phải reviewer đóng finding**.

**Chưa chốt / ngoài R48**
- Các ghi chú chưa commit ở lịch sử được đọc theo thời điểm trước snapshot `117d62d`, không phải trạng thái Git hiện tại.
- Lượt R46 và R47 đã dừng; repo khác mốc lưu của cả hai — không `start`/`recover` lại, hồ sơ giữ nguyên.

**Tám đề xuất theo quyết định R50 ngày 12/10 (không mở thêm mốc):**
- **P64dfb0d8:** đã chạy test11, 5/5 PASS / exit 0 từ bằng chứng kế thừa; mã có K1 nhưng cấu hình tắt.
- **P487415eb:** chỉ thiết kế ở `09`, chờ duyệt thực thi; chưa tính chỉ số DetA, chưa instrumentation hay tracking mới.
- **P2542b6c1:** đã commit tài liệu `a7a6961`, sau đó snapshot `117d62d` đã push theo người dùng; không thao tác Git ghi.
- **P9a6c2078:** đóng vì đã hoàn thành qua R49 / đề xuất trùng, căn cứ đối chiếu mục R48E và biên bản R49A ở trên.
- **Pee184526:** HOÃN.
- **Pf350787b:** HOÃN.
- **P3c58df9b:** HOÃN.
- **Paa38fde5:** HOÃN.

*(Lịch sử — đã làm qua R57A/R58A/R59; bước tiếp hiện hành ở R60A đầu mục 0)* **Bước tiếp theo (23/10):** lượt thực thi test40 đã được Codex nghiệm thu approve (`lan_013`). Việc tiếp theo (ước lượng
biến thiên / SORT trên test40 / BDD tracking / tối ưu Nano): người dùng giao **Codex chọn** (chat 23/10 ~00:15), thực hiện
qua quy trình plan → Claude làm → Codex review. Không commit.

*(Lịch sử)* **Bước tiếp theo (22/10):** Codex nghiệm thu một lần toàn bộ lượt thực thi test40 khi có hạn mức Codex.

*(Lịch sử)* **Bước tiếp theo (21/10, R54B chạy lại foreground):** chờ Codex review smoke R54B phiên foreground
`smoke/lan_20261021_002054_245100_pid16684` (khóa `khoa_20261020_211635_887200_pid10816`, tài liệu
`handoff/logs/review54/02_smoke_va_ban_giao.md` mục 10). Sau đó **xin người dùng duyệt riêng lượt dài** test40 (36 run ×
40 video); wrapper hiện tại chỉ cho phép smoke, chưa có bộ điều phối full — lượt dài cần mã điều phối mới, tiền kiểm và
review riêng (tài liệu 02 mục 7–8). Ghi chú cho plan lượt dài (quyết định P1980cb21): đo thời gian tracking PF500/UKF
trên đoạn dày chưa làm, đưa vào plan lượt dài; cân nhắc tái lập suy luận `yolov8n` giữa tiến trình (mục 10). Không commit.

*(Lịch sử)* **Bước tiếp theo (20/10, R54B lần 1):** chờ Codex review smoke R54B (phiên `smoke/lan_20261020_221756_712200_pid17992`)
— đã review `blocked` (review_004), xử lý bằng lần chạy lại foreground ở trên.

*(Lịch sử)* **Bước tiếp theo (20/10, R54A vòng sửa 2):** chờ Codex review lại R54A (khóa `khoa_20261020_211635_887200_pid10816`).
Chỉ sau approve mới làm R54B: `--buoc kiem-khoa` (mặc định chỉ đọc; `--luu smoke` nếu cần lưu bằng chứng) → `--buoc smoke`
→ `--buoc hau-kiem` → `--buoc kiem-khoa` (GPU, 3 cache × 256 ảnh, 72 lượt video ngắn), rồi tài liệu
`handoff/logs/review54/02_smoke_va_ban_giao.md`. Sửa mã/cấu hình sau approve → quay lại bước chuẩn bị, khóa phiên mới.
Lượt dài test40 cần người dùng duyệt riêng.

*(Lịch sử)* **Bước tiếp theo (20/10, R53C):** chờ GPT review báo cáo 04 và mục 0 này. Sau khi R53C được approve: lượt C — GPT soạn
kế hoạch riêng cho tracking ByteTrack + KF/EKF/UKF/PF trên 40 video test UA-DETRAC (test40) với detector mixed5,
**trình người dùng duyệt trước khi triển khai**; chưa chạy tracking cho tới khi người dùng duyệt riêng. Không chọn
detector theo AP test, không tuning trên test40. Stage/commit: hoãn, người dùng quyết định riêng.

*(Lịch sử)* **Bước tiếp theo (19/10 17:13):** R51 hoàn tất (A, B, C đều approve), đã dừng. Cả hai detector mixed5 đã train xong
và đã chững lại (YOLOv8n best e15 0,3345; YOLO11n best e13 0,340 mAP50-95 dev). Đã báo người dùng; KHÔNG train thêm.
Chờ người dùng duyệt: bàn giao mục 10 của `handoff/logs/review52/02_ke_hoach_luot_dai.md`,
rồi lập kế hoạch đánh giá detector R53 (cần người dùng duyệt).
(Lịch sử: R50A đã approve, xem mục tiến độ R50A ở trên; run R51 đã được khởi tạo sau đó.)
P487415eb vẫn chỉ thiết kế, thực thi cần duyệt riêng. Các việc ngoài mốc vẫn cần người dùng quyết định:
- Giữ hay gỡ hai thay đổi memory ngoài phạm vi của R48B (F5): người dùng quyết định.
- Cách evaluator xử lý ảnh không có `<frame>` trên dev12: R51A đã kiểm trong phạm vi được duyệt (`01_giao_thuc.md`
  mục 3), đang chờ review. Test40 (173 ảnh) và train60 (1.706 ảnh) chưa kiểm.
- Chạy K1 trên tập đánh giá cuối, thực thi thiết kế M5 `06_thiet_ke_augmentation.md`, split BDD mới, kiểm pixel ngoài
  mẫu 160 ảnh: chưa chạy, là việc mới.

Mọi so sánh chỉ dùng các run tạo tại cùng commit, theo giao thức. So dấu vân tay tracking qua mốc `935a445` phải tính
lại theo cùng quy ước trên chính file, không so `eval_meta.json` khác đời (`05b` mục 6). Không tự mở việc ngoài bản giao
việc. Khi `src/audit_run_provenance.py` chưa hỗ trợ hash trước `935a445`, cảnh báo "không khớp" không chứng minh
run cũ không hợp lệ; không sửa bằng chứng hoặc mã audit trong mốc này.

**Điểm chưa rõ**
- Nguyên nhân recall bus thấp (~0,34) của hai checkpoint fine-tune trên UA-DETRAC test; mức đóng góp của nhầm lớp
  bus → truck / car chưa xác định (`04_ket_qua_danh_gia_detrac_test.md` mục 4).
- Khoảng cách PF – EKF: R48D kiểm được một phần H4–H7 trên mô phỏng (`07` mục 5.3). H2 (association) và H3 (likelihood
  trên sai số detector thật) chưa kiểm. Vì sao DetA của PF thấp hơn EKF trên dev12 chưa rõ. Q riêng cho PF chưa chạy.
- Test11 trên mã có K1 đã chạy (R50A, 5/5 PASS); **runner với K1 bật chưa được test11 này kiểm chứng**.
- Toàn văn Arulampalam et al. 2002 không có trên máy: các ý lấy từ bài này ghi "chưa đối chiếu toàn văn".
- Tái tạo mốc 2 tại HEAD mới kiểm cho EKF điều kiện A trên 2/12 video dev (trùng byte). 10 video còn lại, CV / UKF,
  các điều kiện khác và số liệu TrackEval của mốc 2: chưa kiểm (`05b` mục 6).
- Phép chạy lại seed qua runner mới kiểm trên fixture tổng hợp 40 frame (`run_tracking.py`), chưa kiểm qua
  `run_experiment.py` trên video dev.
- Bản fixture có trước có mtime khoảng 2026-10-10 18:29–18:30 +07, khớp khung thời gian lệnh `src/test_pf_ctrv.py` của
  R48A (lệch phạm vi F4 ở trên). Đây là giả thuyết, chưa đối chiếu với log (`05` mục 4).
- Chú thích đơn vị trong mã (`src/config.py:304-305`, `src/time_units.py:33-34`, `src/ekf_ctrv.py:299-300`) ghi
  σ_θ [rad/bước], σ_ω [rad/bước²], khác cách đọc theo gia số trong `01` mục 3. Giá trị số không đổi; chưa sửa vì ngoài
  phạm vi R48A.
- Nguồn gốc cấu hình 53/56 sequence cache `legacy` và 113 thư mục run `LEGACY` không xác minh được bằng hash.
- Nhãn BDD MOT train chưa xác thực bằng archive gốc (chỉ snapshot bên thứ ba đã ghim). R48B xác nhận nhãn trên máy
  đúng là bản đã nhập và đã khóa; không xác nhận trùng nhãn gốc. COCO-video val có 17 track đổi lớp, nguyên nhân chưa
  kiểm.
- Dev nội bộ BDD không độc lập theo lần ghi hình với train-fit (70/139 video dev chung tiền tố tên; 1 cặp cùng cảnh).
  Mức lạc quan của chỉ số dev chưa đo. Split mới cần người dùng quyết định (`03_audit_du_lieu.md` mục 10).
- Tính độc lập giữa các video chưa xác định được: ngay trong một tập đã có video cùng cảnh (DETRAC test `MVI_40773` /
  `MVI_40775`). Kiểm chồng lấn giữa tập huấn luyện và tập đánh giá chỉ bao 4 frame mẫu mỗi video (`03` mục 6.4).
- Pixel ảnh tô xám của bản xuất YOLO: R48E mới kiểm trên mẫu 160 ảnh (131 đạt / 26 chưa kết luận / 3 không đạt, `03`
  mục 14). Toàn bộ 18.600 ảnh tô xám chưa kiểm. Nguyên nhân 3 ảnh không đạt T3 chưa chứng minh (phù hợp với nén lại JPEG,
  chưa loại trừ lỗi xử lý vùng); bản cv2 lúc export chưa xác minh (`03` mục 14.4).
- 572 ảnh dev12 không có `<frame>`: R51A (chờ review) đã kiểm. `MotChallenge2DBox` chấm chúng như frame âm tính, mọi
  box tracker ở đó là FP và làm giảm cả AssA. Bằng chứng phù hợp với âm tính theo quy ước nhãn; chưa loại trừ sót nhãn.
  Ảnh hưởng của chúng lên hiệu số HOTA / AssA / IDF1 chỉ biết sau khi chấm R1 ở R51B. 173 ảnh D4 test40 (R54A, đã
  approve): **đã kiểm cấu trúc GT** — tiền kiểm `results/phase2/r54_tracking/chuan_bi/tien_kiem_20261020_211554_265400_pid2664/`
  (`tien_kiem.json`, `bang_tien_kiem.csv`) cho thấy 173 ảnh nằm ở 6 video (`MVI_39051` 69, `MVI_39211` 94, `MVI_39271` 4,
  `MVI_40763` 4, `MVI_40743` 1, `MVI_40793` 1), không ảnh D4 nào có dòng GT, 0 frame không GT ngoài D4, số `<frame>` XML
  = số ảnh − D4 ở cả 40 video (chỉ đối chiếu số đếm với fingerprint R47, không đọc lại XML); **đã kiểm hành vi evaluator
  trên dữ liệu giả** — fixture D1 `results/phase2/r54_tracking/tests/lan_20261020_211038_548300_pid20336/ket_qua_test.json`:
  frame không GT (kể cả frame kiểu D4) vẫn qua tracker (MOT có dòng ở các frame đó) và được `MotChallenge2DBox` chấm
  (FP > 0, HOTA trong [0, 1]); fixture E7 chặn ảnh D4 có dòng GT. Smoke R54B (frame 1–128) không chứa ảnh D4 nào.
  **Còn mở:** 173 ảnh này có sót nhãn hay là âm tính thật — chưa kiểm, không coi là nhãn âm tính chắc chắn; ảnh hưởng
  lên metric test40 chưa đo. 1.706 ảnh train60: chưa kiểm.
- Ranh giới car / truck / van mơ hồ trên mẫu trực quan, chưa đo tác động lên nhầm lớp.
- Giữ hay gỡ hai thay đổi memory ngoài phạm vi của R48B (F5, `03` mục 13.3): người dùng quyết định; agent không tự hoàn
  tác.

## Thứ tự các mốc

| # | Mốc | Trạng thái (từ vòng 27; dòng 3 cập nhật 22/10/2026, dòng 2, 4, 5 cập nhật 23/10/2026; người dùng cho phép sửa bảng này) |
|:--|:--|:--|
| 1 | **Giao diện Nano** (HDMI + chuột + bàn phím, launcher, cửa sổ xem trực tiếp) | ✅ **Đã nghiệm thu với giới hạn** (review `239b9c6`): thao tác HDMI/zenity được tự động hóa, reviewer không bấm tay; nhánh SIGKILL chưa kiểm với zenity treo thật trên Nano; không camera; không vận hành nhiều giờ. Gói đang dùng: `~/nano_run/v16` |
| 2 | **Chuẩn bị BDD100K train** và **fine-tune YOLOv8n, YOLO11n** (5 lớp). **YOLOv8n COCO hiện tại** (`models/yolov8n.pt`, sha256 `f59b3d83…`) giữ làm **baseline** | **Cập nhật 23/10: ✅ Đã xong.** Theo quyết định người dùng 16/10, detector được train lại trên dữ liệu trộn **BDD100K + UA-DETRAC** (mixed5; UA-DETRAC train 48 video, 40 video test không vào train). **R52** (approve): dataset `data/yolo_mixed5_v1` (`handoff/logs/review52/01_dataset_mixed5.md`), cấu hình + smoke + kế hoạch lượt dài (`handoff/logs/review52/02_ke_hoach_luot_dai.md`). Lượt dài 15 epoch mỗi model, khởi tạo COCO: **YOLOv8n mixed5** xong 18/10 (e15 mAP50 dev 0,493 / mAP50–95 0,335), **YOLO11n mixed5** xong 19/10 (best e13 0,496 / 0,340); checkpoint `models/finetune_mixed5/` (dev nội bộ, không phải test). Detector BDD-only bên dưới không dùng tiếp vì mAP50 (car, bus) UA-DETRAC test 0,586 / 0,564 < baseline 0,870 (R47). Bàn giao + đánh giá đầy đủ: dòng 3 (R53). *(Lịch sử từ vòng 27, giữ nguyên:)* 🔄 Vòng 27: mã chuẩn bị dữ liệu / huấn luyện / đánh giá + smoke đường chạy trên fixture. Vòng 28: sửa R27-F1/F2/F3 của review `4937c85` (hợp đồng dataset dùng chung, khóa nội dung, ghép hòa IoU như COCO); Codex `07615ba` xác nhận 3 finding đã hết nhưng mở R28-F1 (cache ảnh `.npy`). Vòng 29: sửa R28-F1 — Codex `bcb92f8` **PASS WITH LIMITATIONS**, đóng R28-F1; base review → `bcb92f8`. Vòng 30: kiểm lại dữ liệu — BDD100K MOT train **vẫn chưa có**, nên khóa split / xuất / smoke thật **bị chặn**. Vòng 31: **ảnh** MOT train tải + xác minh md5 + giải nén đủ 7/7 phần (1.400 video, 278.079 ảnh); **nhãn** MOT 2020 `box_track_20/train` **không lấy được** từ nguồn chính thức nào (ETH NXDOMAIN, nút Berkeley bị gỡ / 404); `kiem-nguon` exit 2, chỉ còn thiếu nhãn. Vòng 32–33: ứng viên nhãn Scalabel từ zip đóng gói lại trên HF (`jaffe03195/bdd100k_sot` @ `694197099692…`) được cách ly và kiểm chứng. Box/ID của native val khớp COCO val của dự án, nhưng lớp/ignore khác (183/50, UNKNOWN). Codex `4e71d32` **PASS WITH LIMITATIONS**; base review → `4e71d32`. Vòng 34: người dùng **chọn phương án C** — dùng có điều kiện snapshot bên thứ ba đã ghim revision và hash. Đã nhập nguyên byte 1.400 JSON train; hash 1.400/1.400 khớp; `kiem-nguon` **exit 0** (1.400 video, 278.079 frame). Chưa xác thực bằng archive gốc. Codex `ccd801f` **PASS WITH LIMITATIONS**; base review → `ccd801f`. Vòng 35: **khóa split** `bdd5_train_v1` (train-fit 1.261 / dev 139 video); xuất tập con smoke 334 / 84 ảnh; **smoke 1 epoch trên dữ liệu thật đạt cho YOLOv8n và YOLO11n** (lần 1 hỏng do công cụ đo, đã sửa và có regression); dự toán local bước 5 / 15 epoch: kịch bản **~56,6 h GPU** (kịch bản chậm hơn ~74 h; dự toán theo giả định, không phải bảo đảm). Codex `2351c67` **PASS WITH LIMITATIONS** (split + smoke); base review → `2351c67`. Vòng 36: chỉ đính chính diễn giải R35-F1 (nguyên nhân confidence thấp ở checkpoint smoke **chưa xác định**) và R35-F2 (dự toán là kịch bản, không phải cận trên); số liệu không đổi. Vòng 37 (người dùng duyệt): **pilot chẩn đoán 2 model × 2 epoch** trên dataset smoke, cấu hình khóa trước (chỉ đổi `epochs`), **ĐẠT** cả hai. Model 5 lớp trước bước học đã có điểm lớp gần 0 (nhánh phân lớp `cv3` khởi tạo mới khi 80 → 5 lớp); các bước học làm điểm và AP tăng; warmup thực chạy khớp mã. Runner sửa lỗi `final_eval` ghi đè `val_s` epoch cuối. Codex `0c0a07a` **PASS WITH LIMITATIONS** (đóng R35-F1/F2); base review → `0c0a07a`. Vòng 38 (người dùng duyệt bước 5): **đã xuất dataset đầy đủ** `data/yolo_bdd5_v1` — 50.487 ảnh train-fit / 5.559 ảnh dev (khớp số đếm), 8,03 GB; hợp đồng v2 0 lỗi, kiểm chéo ĐẠT (danh sách frame dùng lại `chon_frame()` của exporter — đính chính vòng 39, R38-N2), trùng nội dung train/dev và BDD val 0 (UA-DETRAC test chỉ so tên video); dry-run hai model nhận đúng dataset. Codex `b83dc12` **PASS WITH LIMITATIONS**; base review → `b83dc12`. Vòng 39 (người dùng duyệt): **resume cho runner huấn luyện** (tiếp tục sau epoch đã hoàn tất, bằng resume của ultralytics với đường dẫn checkpoint; kiểm trước khi train; bảo toàn checkpoint; không trùng từng bit với lượt liên tục) và **bộ giám sát RAM bám PID** do công cụ tạo; kiểm trên fixture CPU và dataset smoke GPU, không train trên dataset đầy đủ. Codex `1454b3d` **REQUEST CHANGES** (R39-F1: nhật ký resume bị nối sau dòng cuối ghi dở); vòng 40 sửa riêng R39-F1 (sao lưu nguyên byte nhật ký và xử lý phần đuôi trước lần ghi tiếp đầu tiên), base giữ `b83dc12`. Codex `421dd53` **PASS WITH LIMITATIONS** (đóng R39-F1 trong phạm vi sửa nhật ký), base → `421dd53`; vòng 41 sửa R40-N1/N2 (sao lưu nguyên byte và xử lý hàng cuối ghi dở của `results.csv` / `theo_batch.csv` trước lần ghi tiếp đầu tiên). Codex `6c8e37a` **PASS WITH LIMITATIONS** (đóng R40-N1/N2), base → `6c8e37a`. **Vòng 42 — người dùng duyệt training dài** (2026-10-03): YOLOv8n rồi YOLO11n, tuần tự, **tổng 15 epoch mỗi model** (kể cả khi resume), dataset đầy đủ bước 5, giữ tham số đã duyệt; ngân sách dự toán 56,64 h (kịch bản chậm hơn 73,95 h; không bảo đảm). Chỉ training + validation dev; kế hoạch `handoff/logs/review42/02_ke_hoach_luot_dai.md`. **Đã xong** (2026-10-06): YOLOv8n 15/15 (resume 2 lần), YOLO11n 15/15; dev mAP50-95 tốt nhất 0,280 / 0,272 (dev nội bộ, không phải test; chưa đủ để kết luận model nào tốt hơn). Codex `4d54744` **PASS WITH LIMITATIONS** (hồ sơ training); base review → `4d54744`. **Vòng 43** (người dùng duyệt đánh giá detector, 2026-10-06): kế hoạch khóa trước khi chạy; **BDD val đã chạy** cho 3 checkpoint (3 lượt exit 0, 39.973/39.973 ảnh; mAP50–95 trên 5 lớp: baseline 0,157, YOLOv8n fine-tune 0,251, YOLO11n fine-tune 0,247 — một lượt mỗi model, chỉ mô tả, chưa đủ để chọn model). **UA-DETRAC test chưa chạy**: chờ người dùng chốt ánh xạ lớp (van / others / lớp báo AP / 173 ảnh không có frame / ignored_region). So sánh bộ lọc, tracking, tối ưu Nano chưa làm. Codex `a03f6e1` **CHƯA NGHIỆM THU ĐẦY ĐỦ** (R43-B1: reviewer thiếu mã phụ thuộc để chạy test và kiểm đường chấm, là giới hạn bàn giao, chưa phải lỗi thuật toán; R43-N1 P2: metadata không phản ánh `--conf-van-hanh` hiệu lực ở nhánh CLI không mặc định, không làm sai ba kết quả BDD); base giữ `4d54744`. **Vòng 44** (người dùng duyệt; chỉ xử lý review `a03f6e1`): gửi kèm 17 file support nguyên byte, không đổi nội dung (R43-B1); metadata, scorer và chẩn đoán dùng cùng ngưỡng vận hành hiệu lực theo lượt (R43-N1). Ba lượt BDD vòng 43 dùng ngưỡng 0,25 nên lỗi cũ không kích hoạt, không chạy lại; kết quả và checkpoint giữ nguyên. Codex `3e94b21` **PASS WITH LIMITATIONS** (đóng R43-B1 + R43-N1 trong phạm vi mã / fixture / hồ sơ: test BDD 22/22, nhóm E 13/13 + 1 SKIP trên Linux / Python 3.12; không chạy lại inference trên dữ liệu thật, không kiểm trực tiếp checkpoint, không nghiệm thu UA-DETRAC); base review → `3e94b21`. **Vòng 45** (người dùng duyệt; chỉ đọc, không GPU): chuẩn bị quyết định D1–D5 cho đánh giá detector trên UA-DETRAC test — bảng D1–D5 và dự thảo giao thức (`handoff/logs/review45/06_du_thao_giao_thuc_detrac.md`), **chưa khóa, chưa chạy**; không sửa mã, dataset, checkpoint. Codex `d4715aa` **REQUEST CHANGES** (R45-F1 P2: D3 mô tả lớp N/A "không FP" nhưng scorer giữ nguyên vẫn đếm FP; R45-F2 P2: D5 khẳng định chiều thay đổi recall / AP và gộp quy tắc toolkit vượt quá bằng chứng; R45-F3 P3: phương án giữ 173 ảnh bỏ sót 265 lượt vùng ignore; R45-F4 P3: `doc_xml()` làm mất thuộc tính `sequence_attribute`); base giữ `3e94b21`. **Vòng 46** (người dùng duyệt; chỉ xử lý review `d4715aa`, base giữ `3e94b21`): sửa R45-F3 / R45-F4 trong công cụ phân tích (`test_cong_cu_detrac`: bản sửa 8/8, công cụ vòng 45 2/8); sửa diễn giải R45-F1 / R45-F2 và làm rõ 1–3 (tài liệu); không sửa `src/`, dataset, checkpoint, không GPU. Codex `e4a0311` **PASS WITH LIMITATIONS** (đóng R45-F1…F4 trong phạm vi công cụ phân tích và hồ sơ dự thảo; R45-F1 đóng ở mức đặc tả / hồ sơ, adapter D3 chưa viết nên chưa nghiệm thu; reviewer không có ảnh, XML gốc, checkpoint, không chạy toolkit; hai ghi chú không chặn: R46-N1 precision của lớp không GT cần điều kiện sau xử lý ignore, R46-N2 kích thước 960×540 mới kiểm ảnh đầu / cuối mỗi video). Kế hoạch R46 (2026-10-09; chỉ tài liệu, chưa được review): ghi nhận `e4a0311`, đính chính câu chữ R46-N1 / R46-N2, bảng quyết định D1–D5 cho người dùng (`handoff/REVIEW46_CHI_TIET.md` mục 9–11). Review `e4a0311` và bằng chứng chưa lưu vào repo, gói chưa archive; base review **vẫn `3e94b21`**. **D1–D5 CHƯA DUYỆT**, không phương án nào được chọn. **Tiếp theo** (theo review `e4a0311`): người dùng chốt D1–D5; chỉ sau đó mới triển khai / kiểm adapter, fingerprint toàn bộ đầu vào và khóa kế hoạch trước ba lượt đánh giá PC, mỗi bước cần người dùng duyệt. UA-DETRAC test chưa chạy; so sánh bộ lọc, tracking, tối ưu Nano chưa làm |
| 3 | **Đánh giá detector riêng** (AP theo lớp, mAP50, mAP50–95, precision/recall tại ngưỡng vận hành), tạo **detection cache tái lập** | Evaluator 5 lớp đã có (vòng 27). Vòng 35: chạy trên dev **smoke** thật (84 ảnh) chỉ để kiểm pipeline; chưa đánh giá trên dev đầy đủ. **Vòng 43**: đã đánh giá detector trên **BDD val** (COCO-video, 39.973 ảnh) cho 3 checkpoint theo kế hoạch khóa — AP theo lớp, mAP50 / mAP50–95, P/R tại conf 0,25 (`handoff/REVIEW43_CHI_TIET.md`); UA-DETRAC test chờ quyết định ánh xạ lớp. Dự đoán đánh giá (NPZ, conf ≥ 0,001) đã lưu nhưng **chưa** phải detection cache cho tracking; cache chưa tạo. **Vòng 44**: Codex `3e94b21` đóng R43-B1 / R43-N1 (metadata của entry point BDD val dùng ngưỡng vận hành hiệu lực theo lượt); số liệu BDD val vòng 43 giữ nguyên, không chạy lại. **Vòng 45–46**: bảng D1–D5 và dự thảo giao thức cho UA-DETRAC test (chỉ đọc, chưa khóa); Codex `d4715aa` **REQUEST CHANGES**, sửa ở vòng 46; Codex `e4a0311` **PASS WITH LIMITATIONS**. **D1–D5 CHƯA DUYỆT**; adapter UA-DETRAC chưa viết, UA-DETRAC test chưa chạy, chưa có kết quả detector UA-DETRAC; detection cache vẫn chưa tạo. **Cập nhật 22/10:** ✅ **Đã xong.** D1–D5 người dùng duyệt 09/10; R47 đánh giá 3 detector (COCO + BDD-only) trên UA-DETRAC test. R52 train lại detector trên dữ liệu trộn BDD100K + UA-DETRAC (mixed5). **R53** đánh giá đầy đủ yolov8n_coco / yolov8n_mixed5 / yolo11n_mixed5 theo giao thức khóa: UA-DETRAC test mAP50 (car, bus) 0,870 / 0,879 / 0,888; BDD val mAP50 5 lớp 0,244 / 0,390 / 0,379; chấm lại từ NPZ trùng; GPT nghiệm thu approve (`handoff/logs/review53/04_ket_qua_danh_gia_day_du.md`). **Detection cache cho tracking** test40 (3 detector × 40 video) đã tạo và xác nhận trong lượt thực thi 21/10 (dòng 4) |
| 4 | So sánh bộ lọc **CV / EKF / UKF / PF**, đối chứng **SORT / ByteTrack** bằng các phép so sánh có kiểm soát | **Cập nhật 23/10 (tối): ✅ Đã xong** trên UA-DETRAC test40 — ByteTrack (R56), ước lượng biến thiên (R57), đối chứng SORT (R58 thiết kế, R59 thực thi), đều Codex approve; chi tiết (d)–(e) dưới. *(Dòng cập nhật 23/10 sáng, giữ làm lịch sử: "🔄 ByteTrack trên test40 xong, Codex nghiệm thu approve (`nghiem_thu_test40/lan_013`); còn SORT trên test40 và ước lượng biến thiên.")* (a) **R51** (dev12, detector YOLOv8n-COCO, đã approve): so sánh có kiểm soát ByteTrack vs SORT và CV/EKF/UKF/PF100/PF100-K1 — HOTA ekf 0,5925 / cv 0,5900 / ukf 0,5879 / pf100 0,4933 / pf100-k1 0,5377; ByteTrack > SORT ~0,02–0,03; CI ekf−cv, ukf−cv chứa 0 (`handoff/logs/review51/03_tong_hop_va_slide.md`). (b) **R54–R56** (đã approve): khóa giao thức test40 (3 detector × c25/c10 × CV/EKF/UKF/PF500 s0–s2 = 36 run × 40 video), smoke, bộ điều phối tiếp tục theo video, khóa R56. (c) **Thực thi 21/10** (ngoài orchestrator theo quyết định người dùng): 36/36 run chấm xong, tái lập 9/9 + 12/12 trùng byte, hậu kiểm đạt; HOTA mixed5 cao hơn COCO ~3–4, CV/EKF/UKF chênh ≤ 0,34, PF500 thấp hơn ~2,9–3,4 (`handoff/logs/review56/03_thuc_thi_tracking_test40.md`). (d) **R57A** (approve, `review_002.json` lượt `bytetrack_r57_bien_thien_20261023_0029`): bootstrap ghép cặp theo video trên test40, chỉ phân tích hậu nghiệm — PF500 thấp hơn CV/EKF/UKF HOTA −2,90…−3,43 pp, CI không chứa 0 ở cả 6 nhóm detector × ngưỡng (54/54 dòng HOTA/IDF1/IDSW); giữa CV/EKF/UKF |chênh HOTA| ≤ 0,34 pp, HOTA EKF−CV CI không chứa 0 ở 2/6 nhóm (yolov8n_coco-c10, yolo11n_mixed5-c25); 108 CI riêng lẻ chưa hiệu chỉnh so sánh bội (`handoff/logs/review57/01_bien_thien_test40.md`). (e) **R58A** (approve) hồ sơ thiết kế; **R59B** (approve, `review_005.json` lượt `bytetrack_r59_sort_test40_20261023_1053`): SORT trên test40 **chỉ c25** (c10 không chạy theo quyết định Q-A), 3 detector — ByteTrack hơn SORT HOTA +1,38…+3,71, chủ yếu nhờ AssA / IDF1 / IDSW; ở c25 tầng association điểm thấp của ByteTrack rỗng (`handoff/logs/review59/03_ket_qua_sort_test40.md`). Kết quả mô tả trên một tập test, không suy rộng ngoài phạm vi đã ghi |
| 5 | **Đánh giá cuối** trên **BDD100K val** và **UA-DETRAC test** | **✅ Hoàn tất thực thi và báo cáo 27/10/2026.** R61 cổng A + B đạt; BDD val: 36 run × 200 video, 7.200 tracking, 36 chấm, 108 CI, hậu kiểm đạt. UA-DETRAC test40 dùng R53/R56/R57/R59 đã nghiệm thu. Báo cáo tách hai tập: `handoff/logs/review61/06_ket_qua_thuc_thi_cong_b.md`; đóng mốc và chuyển khóa PLAN: `handoff/logs/review61/07_dong_moc5.md`. Chưa có review bởi tác nhân thứ hai cho kết quả B. |
| 6 | Cuối cùng mới **tối ưu tốc độ hệ thống Nano**, so với cấu hình baseline đã ghi nhận | **🔄 Đang làm 28/10/2026:** người dùng đổi Jetson, dùng lại thẻ cũ; manifest v16 562/562 và engine hash đạt, đang đo baseline mới từ bản sao v16 (`handoff/logs/review62/05_tien_kiem_va_khoi_phuc.md`). Chưa có kết quả tối ưu. |

Lớp đã khóa: `0 car, 1 truck, 2 bus, 3 motorcycle, 4 bicycle` (`src/detector_classes.py`).

## Quy tắc phạm vi

- **Hỏi trước**: mọi đề xuất thêm model, dataset, thuật toán, ablation hoặc tính năng ngoài danh sách trên phải hỏi người
  dùng trước khi thực hiện. Ví dụ đang **chờ quyết định**: dùng gói BDD100K *detection* 100K (ảnh tĩnh, nhãn `det_20`) thay
  hoặc cùng gói MOT train. Gói này khác nguồn với MOT, nên chưa dùng.
- **Chọn bằng dev nội bộ**: checkpoint, số epoch và mọi tham số chỉ chọn bằng **dev nội bộ**. Dev này tách **theo video** từ
  split `train` chính thức của BDD100K MOT.
- **Hai tập đánh giá cuối không tham gia lựa chọn**: BDD100K val (bản COCO-video đã định danh) và UA-DETRAC test.
  - Không dùng hai tập này làm train hay dev.
  - Cả hai đã được xem trong các vòng nghiên cứu trước (thí nghiệm transfer BDD, mốc 2 UA-DETRAC), nên **không gọi là tập
    test hoàn toàn chưa từng quan sát**.
- **Phạm vi vòng 27**: chỉ chuẩn bị và smoke. Chưa chạy training dài, benchmark tracking đầy đủ hay tối ưu Nano.
- **Giữ nguyên**: hệ thống Nano v16, kết quả và lock cũ, file riêng của người dùng. Dataset, weights, video và cache lớn nằm
  ngoài gói delta.
