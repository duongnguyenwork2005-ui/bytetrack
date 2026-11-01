# R63 — Kế hoạch vòng tối ưu mới cho YOLO + ByteTrack trên Jetson Nano

**Trạng thái:** GPT duyệt **DUYET_CO_SUA** 30/10 (`D:/LABOTARY/automation/runs_local/hoi_codex_R63_duyet_ke_hoach_
20261030/answer.json`); người dùng giao GPT quyết định. Các sửa bắt buộc đã đưa vào mục 2.9–2.11, 4 và 7 (31/10).

Ngày lập: 30/10/2026. Người dùng yêu cầu mở vòng tối ưu mới sau R62 (mốc 6 đóng 29/10 với kết quả âm, giữ v16).
Theo `handoff/PLAN_DA_CHOT.md` ("Hỏi trước", và kết luận R62 "một đợt tối ưu khác cần kế hoạch mới"), **chưa có
phép đo Nano, build engine, sửa `src/` hay chạy đánh giá GPU dài nào cho tới khi người dùng duyệt kế hoạch này.**
Phần đã làm là chuẩn bị độc lập trên PC (mục 2) và công cụ (`cong_cu/`, `chat_luong/`).

## 1. Mục tiêu và ranh giới

Mục tiêu: tăng **FPS xử lý toàn luồng** của chương trình trên Nano (đọc ảnh → TensorRT FP16 → ByteTrack CV/EKF →
hiển thị HDMI) so với **bản sao v16 đã kiểm hash**, mà không giảm chất lượng tracking quá ngưỡng đã duyệt, đặc biệt
với xe nhỏ và xe bị che khuất.

Giữ cố định trong so sánh ban đầu: checkpoint `models/yolov8n.pt` (COCO, `f59b3d83…`, đúng model đang chạy trên
Nano), FP16, batch 1, ngưỡng c25 (và c10 khi đánh giá), lớp, YAML tracker, buffer 1,2 s, `elapsed_mode`, metric.
Mọi chỉnh ngưỡng tracker (nếu có) là **thí nghiệm tách riêng**, không gộp vào kết luận tốc độ. Không INT8 (ưu tiên
thấp trên Nano theo yêu cầu). Không huấn luyện hoặc chọn model/kích thước dựa trên UA-DETRAC test hay BDD val.
Gói v16, engine `c32690cd…`, launcher và kết quả R62 chỉ đọc; mọi bản mới là `v17*` trong thư mục mới.
Không kết luận FPS Nano từ số PC.

## 2. Hiện trạng đã kiểm trên PC (30/10/2026)

1. **Repo:** HEAD `304ce7a`; `PLAN_DA_CHOT.md` (đoạn khảo sát sau mốc 6) và `review62/32_khao_sat_toc_do_video.md`
   chưa commit. ZIP v16 `handoff/nano_transfer_latest.zip` SHA256 `302f6af7…` khớp. Trong mã Nano dùng lúc chạy,
   chỉ `src/nano_detector.py` của HEAD khác v16 (bản LUT R62) ⇒ **baseline phải là bản sao v16**, không phải HEAD.
2. **Engine v16:** `models/export/yolov8n_640_op12.onnx` (`a66a5e03…`, opset 12, tĩnh 1×3×640×640 → 1×84×8400),
   build `trtexec --fp16 --workspace=1024` (`scripts/nano/01_build_engine.sh`). Xuất lại 640×640 hôm nay cho đồ thị
   **tương đương** (262 node, 127 initializer trùng từng byte, chỉ khác metadata ngày) ⇒ đường xuất ONNX tái lập được.
3. **Letterbox Nano vuông 640×640** (`src/nano_detector.py:195-202`, `nano_detect_stream.py:243,311` truyền `--imgsz`
   int) cho video 960×540: ảnh co còn 640×360, **280/640 hàng (43,75 % pixel) là viền đệm 114**. Đường PC
   (`det_cache.py:254`, `model.predict(imgsz=640)`) đã kiểm thực nghiệm cho tensor **(1,3,384,640)**; `imgsz` 512/448
   cho 288×512/256×448. ⇒ Mọi số chất lượng PC của yolov8n COCO (R53–R61) tương ứng đầu vào **384×640**, không phải
   640×640 vuông của Nano. R16 từng so Nano↔PC: 579/581 detection ghép, IoU 0,988.
4. **69 ms là gì:** `exec_ms` = thời gian host quanh `execute_v2` đồng bộ (`nano_detector.py:294-302`), gồm GPU thực
   thi + phí khởi chạy/đồng bộ; chưa tách. trtexec lịch sử 68,1 ms dùng `--useCudaGraph --noDataTransfers` (khác đường
   pipeline). ⇒ Cần đo lại cùng engine bằng trtexec (nhiều chế độ) và CUDA event trong pipeline (G1).
5. **ONNX ứng viên** đã xuất (chưa build/đo): 384×640, 352×576, 288×512, 256×448; đầu ra 5040/4158/3024/2352 anchor;
   mọi op có trong bảng TRT 8.2. Manifest ở `ung_vien_onnx/*/manifest.json`. Kích thước file **không** phải bằng chứng FPS.
6. **Engine chữ nhật cần sửa mã nhỏ** (sau duyệt): `nano_detect_stream.py`/`nano_detector.py` phải lấy (H,W) letterbox
   từ shape binding của engine thay vì `--imgsz` int; nếu không sẽ letterbox 640×640 vào engine 384×640 (sai).
   `bench_engine_pipeline.py` đã tự lấy shape từ engine nên đo được ngay.
7. **Công cụ chất lượng:** `det_cache.py --imgsz W` tạo cache ở đúng hình dạng ứng viên. Chưa có tham số để PC chạy
   letterbox **vuông** (tái hiện v16) — cần thêm cờ `--rect/--no-rect` mặc định giữ nguyên. Đánh giá AP
   (`eval_detector_5lop.py`, `danh_gia_detector_*`) cố định `imgsz 640` trong `QUY_TAC` — cần cờ kích thước (mặc định
   giữ nguyên, ghi vào metadata). Chưa có phân tích theo kích thước xe ⇒ đã viết `chat_luong/phan_tang_kich_thuoc.py`
   (recall/precision theo kích thước × che khuất, so cặp bootstrap theo video; test 3/3, smoke 1 video dev thật exit 0).
8. **BDD:** chưa có split tracking cho BDD dev nội bộ (139 video); BDD val chỉ dùng báo cáo. Lựa chọn trên dev hiện
   chỉ khả thi bằng UA-DETRAC dev12 (tracking + che khuất + kích thước) và AP detection trên dev nội bộ mixed5
   (BDD 139 video + dev12) — xem câu hỏi Q3.
9. **Nano (cập nhật sau G0, 30/10 16:54):** IP mới **192.168.1.5** (alias `nano` đã đổi). G0 chỉ đọc đã chạy:
   L4T 32.7.5, TRT 8.2.1, MAXN, nhàn rỗi 33,5 °C, MemAvailable 2,29 GB, swap 0; không panic trong boot hiện tại,
   pstore rỗng, journal không lưu boot trước; engine `c32690cd…`, v16 562/562 và 1.470 ảnh khớp hash. Chi tiết
   `g0/02_ket_qua_g0.md`. Chưa chạy phép đo tải nào.
10. **Sự cố panic 30/10 đã xử lý (31/10)** — `su_co_20261030_1750_dd/01_bien_ban.md`: nguyên nhân là thẻ SD acer ở
   SDR104 204 MHz; DTB sao chép giới hạn khe SD (`sdhci@700b0400` `max-clk-limit`) 100 MHz, rootfs vẫn trên thẻ acer.
   Chế độ thẻ phụ thuộc kiểu boot: boot nguội → SDR104, xung thực 99,84 MHz, 1,8 V; reboot ấm → HS 50 MHz, 3,3 V. Mỗi
   lượt đo ghi dòng `mmc1: new …` của boot đó; so cặp luôn trong **cùng boot**. Dự phòng: LABEL `goc204`. IP trường
   10.11.61.127 (alias `nano`), ở nhà 192.168.1.5 (DHCP, kiểm lại mỗi phiên). Runner `cong_cu_v4` (tegrastats xả từng
   dòng + `sync`). Engine `c32690cd…` được build trên board R62 trước ⇒ TRT cảnh báo khác thiết bị.
11. **Mục tiêu báo cáo (người dùng, 31/10):** v16 "bê nguyên" sang Nano ≈ 9 FPS → sau tối ưu FPS tăng, chất lượng
   không kém hơn ngưỡng. Bảng trước/sau đo **lại cả hai** cùng board/boot/video/hiển thị, FPS đầu-cuối như v21–v23
   (`fps_xu_ly`), kèm GFLOPs (không phụ thuộc thiết bị). Số tăng tốc chỉ khẳng định cho Nano 4 GB/JetPack 4.6/TRT
   8.2/FP16; giảm pixel đệm là tối ưu không phụ thuộc thiết bị (mức lợi trên thiết bị khác phải đo).

## 3. Giả thuyết

- H1: phần lớn ~69 ms là GPU thực thi engine (khi đó tối ưu mã quanh engine không thể vượt nhiều %).
- H2: engine 384×640 giảm thời gian GPU gần tỉ lệ số pixel (60 %) mà **không giảm độ phân giải ảnh** (cùng hệ số
  co 0,667 như hiện tại) ⇒ chất lượng gần như giữ nguyên. Đây là ứng viên đầu tiên vì rủi ro chất lượng thấp nhất.
- H3: các mức nhỏ hơn (352×576, 288×512, 256×448) nhanh hơn nữa nhưng làm mất xe nhỏ/che khuất — phải đo.
- H4: hai ứng viên mã R62 (bỏ copy float32 1×84×N; tái dùng cột lớp đã lọc) cộng LUT có thể vượt 5 % mà đầu ra trùng byte.
Không giả định mức tăng nào trước khi đo.

## 4. Trình tự (mỗi giai đoạn có cổng; dừng khi trượt)

**G0 — Tiền kiểm Nano chỉ đọc (~10 phút).** `cong_cu/kiem_suc_khoe_nano.sh`: JetPack/L4T, TensorRT, nvpmodel, xung,
nhiệt, RAM/swap, đĩa, log kernel boot hiện tại/trước, pstore; kiểm hash engine/v16 manifest/ảnh như R62. Không reboot,
không đổi nvpmodel/jetson_clocks, không cài gói. *Cổng ổn định:* không panic/softlockup/hung_task mới, pstore không có
bản ghi mới so với R62, nhiệt nhàn rỗi **CPU và GPU** < 50 °C (PMIC ghi riêng, thường 50 °C), MemAvailable > 1,5 GB,
swap < 100 MB, không tiến trình đo khác; người dùng đã đóng trình duyệt/update-manager/gnome-software (cửa sổ)/jtop
(Q6; packagekitd, nvpmodel_indicator giữ nếu nhàn rỗi, ghi lại). Trượt ⇒ chỉ báo cáo, không đo tải. G0 chỉ chứng
minh trạng thái hiện tại; thiếu log boot trước không chứng minh đã hết nguyên nhân panic.

**G0b — Lượt ổn định 300 frame có telemetry (trước mọi benchmark/build/lượt dài).** `chay_mot_luot.py` (R62) CV,
hiển thị bật, qua `cong_cu_v4/chay_co_giam_sat.sh` (timeout 900 s, hậu kiểm kernel/boot_id/swap). Đạt: exit 0, không
vi phạm cổng, parity live/replay trùng byte. **Đã chạy 31/10** (`g1/01_luot_on_dinh_300.md`): lượt 01 không hợp lệ
(Chromium mở, swap +181 MB); lượt 02 sau reboot ĐẠT (8,68 FPS, MOT trùng byte 4 lượt CV R62).

**G1 — Baseline v16 + phân rã thời gian (~1–1,5 giờ).** Trên bản sao v16 mới (kiểm 562 hash):
(a0) **build lại engine baseline** từ đúng ONNX v16 `a66a5e03…` trên board hiện tại (`build_engine_nano.sh`, cùng
cờ v16) — ứng viên G3 đều build trên board này nên baseline phải cùng cách build; giữ engine `c32690cd…` làm đối
chiếu R62: trtexec + bench cho **cả hai engine** cùng boot, parity detection hai engine trên 300 frame (báo tỉ lệ ghép,
IoU; không kỳ vọng trùng byte). Engine mới là "trước" của bảng báo cáo; gói v16 không đổi;
(a) `do_trtexec.sh` trên engine v16 (A–E); (b) `bench_engine_pipeline.py` 4 pass xen kẽ đồng bộ/CUDA event;
(c) pipeline thật 300 JPG `MVI_39031_f0001-0300`, CV và EKF, bật hiển thị, 4 lượt mỗi tracker (runner R62
`chay_mot_luot.py`), live/replay trùng byte. Báo: decode, tiền xử lý, H2D, GPU exec (event), phí gọi/enqueue/chờ,
D2H, copy float32, decode_nms, gửi, `tracker.update`, hiển thị, startup; FPS, trễ p50/p90, nhiệt, xung, RAM/swap.
Nếu người dùng có video MP4 đích (Q5), thêm một baseline MP4 cùng cách ghi. *Kết luận bắt buộc:* 69 ms là GPU
thực thi hay có phí gọi đáng kể (so `gpu_exec_ms` event với `exec_ms` host và GPU Compute của trtexec).

**G2 — Ứng viên mã, không đổi model (~1 giờ Nano).** Gói `v17-ma` = v16 + LUT + bỏ copy float32 dư + tái dùng cột
lớp. Trên PC: test detector + tương đương byte với v16 trên tensor/detection. Trên Nano: kiểm 11 ca CPU như R62, rồi
so cặp **4 cặp CV + 4 cặp EKF** (thứ tự cân bằng, nghỉ 20 s, cùng boot) như R62. *Đạt:* trung vị FPS theo cặp
≥ +5 % ở cả CV và EKF, detection/MOT/lớp **trùng byte** với v16, không drop frame, p90 trễ không tăng > 5 %, không swap/
panic mới. Không đạt ⇒ ghi kết quả âm, vẫn có thể ghép vào G3.

**G3 — Giảm kích thước đầu vào (chỉ khi G1 xác nhận engine là nút thắt).** Thứ tự: **C1 384×640** → (nếu cần thêm tốc
độ và C1 đạt chất lượng) C2 352×576 → C3 288×512 → C4 256×448. **Lịch G3/G4 xen kẽ theo từng ứng viên** (chốt trước
G1): build + tốc độ Nano → chất lượng dev G4 → chỉ khi đạt mới xuống mức nhỏ hơn. Mã G2 chỉ ghép vào G3 nếu parity
đạt (dù tốc độ G2 chưa đạt). Mọi cặp báo cùng decode/telemetry; nếu lệch tầng decode giữa hai nhánh như R62 thì chưa
quy tăng FPS cho tối ưu. Mỗi mức: build engine FP16 trên Nano bằng
`build_engine_nano.sh` (cùng cờ v16), trtexec + bench (nhanh, ~15 phút/mức) để vẽ đường tốc độ; chỉ mức nằm trên biên
tốc độ/chất lượng mới đánh giá đầy đủ. Gói `v17-C*` cần sửa nhỏ ở mục 2.6 (test PC + parity). So cặp Nano như G2
(4+4 cặp, cùng video, cùng hiển thị). Đầu ra **không** kỳ vọng trùng byte với v16 ⇒ chất lượng đo ở G4.

**G4 — Chất lượng (PC, ~2–3 giờ GPU mỗi ứng viên theo kịch bản giả định).**
- *Đối chứng:* PC tái hiện v16 = letterbox **vuông 640×640** (cờ mới, mục 2.7) + cùng `yolov8n.pt`; ứng viên = cùng
  trọng số ở (H,W) ứng viên. Báo thêm số 384×640 hiện có của R53–R61 làm tham chiếu.
- *Lựa chọn (dev):* UA-DETRAC dev12 — cache mới mỗi kích thước, ByteTrack CV/EKF cấu hình khóa, c25 và c10: HOTA,
  DetA, AssA, IDF1, IDSW (TrackEval); `occlusion_eval` theo tầng che; `phan_tang_kich_thuoc.py` (xe nhỏ × che khuất,
  so cặp bootstrap 5.000 theo video). AP50/AP50-95/recall theo lớp trên dev nội bộ mixed5 (BDD 139 + dev12).
- *Tương đương Nano↔PC:* detection TensorRT FP16 của lượt Nano (cache live) so PC cùng kích thước trên 300 frame
  (tỉ lệ ghép, IoU) như R16.
- *Báo cáo cuối (sau khi đã chốt, không đổi quyết định):* UA-DETRAC test40 và BDD val — mAP, HOTA, IDF1, IDSW,
  phân tầng che khuất (UA-DETRAC) và kích thước; CI bootstrap riêng lẻ, chưa hiệu chỉnh so sánh bội.
- *Ngưỡng chất lượng ĐÃ CHỐT (Q2, GPT 30/10)* so với đối chứng vuông 640×640, ở c25, cho cả CV và EKF, trên **từng
  miền dev** (UA-DETRAC dev12; BDD dev tracking theo Q3): ΔHOTA ≥ −0,5 điểm (thang 100), ΔIDF1 ≥ −0,5 điểm, IDSW tăng
  ≤ 5 % (baseline IDSW = 0 ⇒ ứng viên phải = 0), **ΔmAP50 và ΔmAP50–95 ≥ −0,5 điểm**, recall xe nhỏ (sqrt diện tích
  < 24 px gốc) giảm ≤ 2 điểm %, recall che nặng giảm ≤ 2 điểm % (UA-DETRAC: visibility < 0,5; BDD: theo nhãn
  `occluded` sẵn có, không quy thành visibility). Báo CI95 bootstrap theo video và số mẫu từng tầng; tầng thiếu GT
  **không** được coi là đạt. c10 là phân tích độ nhạy. Tốc độ: FPS trung vị cặp ≥ +5 % cả CV/EKF, p90 trễ tăng ≤ 5 %,
  đủ frame, live/replay trùng byte. Trượt bất kỳ ⇒ ứng viên không triển khai và **không thử mức nhỏ hơn**.
- *Tương đương Nano↔PC (chốt trước khi xem kết quả):* tỉ lệ detection ghép (IoU ≥ 0,5) ≥ 99 % hai chiều và IoU trung
  vị ≥ 0,98 trên 300 frame (mức R16: 579/581, IoU 0,988); không đạt ⇒ chất lượng PC không đại diện ứng viên Nano.

**G5 — Pruning có cấu trúc + fine-tune, hoặc distillation sang model hẹp hơn.** Chỉ cân nhắc sau khi có kết quả G3/G4,
và cần **duyệt riêng** (huấn luyện; chọn trên dev; tốn nhiều giờ GPU). Không nằm trong phê duyệt lần này.

**G6 — Kết luận và triển khai.** Bảng: ứng viên, FPS/trễ Nano (trung vị cặp, khoảng), thay đổi chất lượng (dev + báo
cáo), quyết định. Ứng viên **nhanh nhất qua cả hai cổng** được đề xuất làm `v17`; chuyển launcher chỉ khi người dùng
đồng ý, kèm đường quay về v16 và kiểm HDMI/launcher. Cập nhật PLAN mục 0 trước khi viết bàn giao.

## 5. Điều kiện dừng chung

Kernel panic/softlockup tái diễn ⇒ dừng mọi phép đo GPU, giữ pstore/log, báo người dùng (không lặp mù). Nhiệt > 80 °C,
swap tăng trong lượt, thiếu/drop frame, parity live/replay hỏng ⇒ lượt không hợp lệ, giữ nguyên bằng chứng, không chọn
lượt đẹp. Hết ngân sách thời gian đã duyệt ⇒ báo cáo phần đã xong.

## 6. Tái lập

Mỗi lượt một thư mục mới (không ghi đè); lưu lệnh đầy đủ, `run_meta`/`moi_truong.json`, boot_id, hash gói/ONNX/engine/
ảnh/cache, phiên bản (L4T, TensorRT, OpenCV, numpy, ultralytics, torch), CSV thô, tegrastats, nhiệt sau lượt, log
trtexec/build. Bằng chứng nhỏ vào `handoff/logs/review63/`; raw lớn đóng tar kèm SHA256, đối chiếu PC/Nano; ONNX/engine
không commit (manifest + lệnh tái tạo). Không bỏ lượt hỏng.

## 7. Câu hỏi đã được trả lời (GPT 30/10, người dùng giao quyết định)

Q1 duyệt G0–G4, G6 và ba sửa mã (G5 duyệt riêng; G6 chỉ đề xuất v17, chưa chuyển launcher). Q2 như mục G4. Q3 dựng
tracking BDD dev từ đúng 139 video dev đã chốt, xong **trước lựa chọn G4**, không chặn G1; không chia lại, không dùng
BDD val/UA-DETRAC test. Q4 chạy trong chat, từng cổng, một tác vụ Nano một lúc. Q5 không bắt buộc MP4; so chính là
JPG + HDMI, không khái quát thành FPS MP4. Q6 người dùng tự đóng ứng dụng nền. Sửa mã `src/` (H,W từ binding engine;
rect/H,W trong `det_cache`/AP) làm **trước G3/G4**, không cần cho G1. Bản gốc câu hỏi giữ dưới đây.

### Câu hỏi gốc (30/10)

- Q1: Duyệt G0–G4 và G6 (G5 duyệt riêng)? Duyệt các sửa mã nhỏ sau duyệt: lấy (H,W) letterbox từ engine; cờ
  `--rect/--no-rect` cho `det_cache.py`; cờ kích thước cho đánh giá AP (mặc định giữ nguyên hành vi cũ).
- Q2: Ngưỡng chất lượng ở G4 và ngưỡng tốc độ +5 % (giữ như R62)?
- Q3: Lựa chọn trên dev chỉ bằng UA-DETRAC dev12 + AP mixed5 dev, hay dựng thêm split tracking BDD dev (139 video,
  nhãn box_track) trước — thêm việc, cần duyệt?
- Q4: Cách chạy: trong chat như R62, hay qua orchestrator `D:/LABOTARY/automation/auto_review` (plan JSON + review GPT/Codex)?
- Q5: Có video MP4 đích (camera/đầu ra thật) để đo thêm ngoài 300/1.470 JPG không? Có cần xuất MP4 bbox/ID trong phép đo?
- Truy cập Nano: user `jetson`, khóa `~/.ssh/id_ed25519_nano`, phiên desktop `:0` cho lượt có hiển thị; IP theo
  mạng (31/10: trường 10.11.61.127, nhà 192.168.1.5) — xem mục 2.10.
