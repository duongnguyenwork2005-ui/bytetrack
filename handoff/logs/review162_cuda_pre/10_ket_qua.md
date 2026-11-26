# Tiền xử lý CUDA tăng8,80%FPS và giữ nguyên chất lượng

Tiền xử lý CUDA giữ nguyên chất lượng detection trên toàn bộ7190 ảnh đã
đo: toàn bộ candidate arrays trùng bit với bản CPU; mAP50 và mAP50–95
vẫn **47,146727% và31,984832%**. Full80 đoạn cho meanFPS
**18,81983→20,47623 (+8,80136%)**, nhanh hơn80/80; MOT/CLS và các metric
tracking giữ nguyên. Host timestamp/source/hash/TrackEval audit đã đạt.

## Phần thay đổi

Giữ engine YOLO11n 50/50 FP16 đầu vào384×640 và toàn bộ trọng số.
Thay CPU resize/letterbox, đổiBGR→RGB, HWC→CHW và chuẩn hóa bằng một
kernel CUDA. Upload ảnh BGR8bit gốc vào buffer GPU tái sử dụng; kernel
ghi tensorfloat32 trực tiếp vào input binding của TensorRT. Dùng cùng
bảng LUT256float32 như bản CPU. Chỉ số và hệ số nội suy được tính một
lần cho mỗi cặp kích thước và cache lại.

Nội suy integer11bit/rounding theo OpenCV3.2.0 thực tế trên Nano;
trường hợp giảm đúng một nửa khớp nhánh averaging của OpenCV.
Nguồn đối chiếu: [OpenCV3.2.0 imgwarp.cpp](https://github.com/opencv/opencv/blob/3.2.0/modules/imgproc/src/imgwarp.cpp).
Biên dịch CUDA10.2 cho sm53, không fast-math. Không đổi hình học,
padding114, JPEG decoder, CPU mask, NMS, ngưỡng, bộ lọc hoặc chính sách
suy luận mỗi frame. ByteTrack vẫn dùng cùng mã và cấu hình đối chứng.

## Kiểm tra đã chạy

- Tensor44/44 trùng bit:24 fixture ngẫu nhiên/biên/halving/oddsize/
  ảnh nhỏ và20 ảnh thật. Max error0; không chỉ đạt ngưỡng1/255.
- Raw YOLO20/20 ảnh thật trùng bit giữa CPUpre và CUDApre trên cùng
  engine/context. Không nới ngưỡng box6px/score0,02.
- Adapter40/40 cặp đầu ra sau NMS trùng bit, với20 ảnh thật và20
  fixture có một vùng che đen. Cùng confidence0,25, NMS0,7,maxdet3000.
- Capture đủ7190 ảnh/72chunk: data, offsets, indices, orig_shapes vàhw
  đều trùng bit với cache CPU gốc. NMS/scorer/GT/hash/khớp lớp được giữ
  nguyên; AP từng lớp, precision và recall cũng bằng nhau.
- Gói stream cô lập chỉ thay nano_detect_stream.py; thêm gpu_pre.py,
  gpu_detector.py và fused_pre.so. nano_detector.py giữ nguyên. Config,
  dữ liệu, model và vendor dùng cùng referent. Kiểm entry/hash đạt.
- Ba cấu hình timing CUDA/pruning/height352 đều qua kiểm nguồn, engine,
  ONNX, AP, package và config trước chạy, ở extra_staging_verification.

AP50 năm lớp lần lượt car84,4577%; truck44,0422%; bus51,4063%;
motorcycle34,4174%; bicycle21,4100%. Đây là dev nội bộ đã quan sát,
không phải một tập test mù mới. Full80 cửa sổ JPEG có mask thực tế cũng
được kiểm: MOT/CLS trùng80/80 và TrackEval xác nhận các metric bằng nhau.

## Tốc độ đã đo đủ80 đoạn

Chuỗi đã chạy xong trong
`../review155_runtime_accuracy/run_extra_chain.ps1`:

1. Microbenchmark luân phiên cùng engine/context,100 mẫu/mode, có YOLO
   chạy trong vòng đo: CPUpre+upload7,19135→2,37751ms; GPUYOLO vẫn~41,9ms.
2. Pilot4 cố định BDD/UA:18,87704→20,44192FPS; dương nên mở full80.
3. Full160 lượt/24.000 inference: BDD19,60334→20,88571FPS (+6,54162%),
   UA18,03632→20,06674 (+11,25744%), chung18,81983→20,47623 (+8,80136%).
   Nhanh hơn40/40 từng tập và80/80 chung.
4. Theo rule đã khóa, chương trình tiếp tục pruning/height352; cả hai
   có chất lượng trượt cổng nên giữ để mô tả trade-off, không promote.

Aggregate FPS theo tổng thời gian18,77945→20,45778; meanFPS trên là
trung bình các đoạn. Median paired gain8,60347%, CI95%[8,39662;9,00098]
theo5000bootstrap/seed155, mẫu chung luôn40BDD+40UA. CI dành cho median
paired gain, không gọi là CI của meanFPS gain. Một lượt/model/đoạn.
Tất cả80 đoạn≥15FPS cả hai bản, ứng viên thấp nhất18,64787FPS.
Median p90 latency theo đoạn121,67594→111,39042ms.

BDD: HOTA43,71884%,IDF1 52,96834%,IDSW1534,FN28539,FP2194 ở cả hai.
UA: HOTA66,57320%,IDF1 78,37333%,IDSW245,FN17590,FP9144 ở cả hai.
Đối chứng mỗi đoạn còn trùng byte R109. Cùng150JPEG/đoạn,warm25,KF,
displayoff,full inference; đây là cửa sổ150 khung trên80 video nguồn,
không gọi là full-video/camera/HDMI FPS.

CPUpre ở bản CUDA ghi0 vì phần việc đã chuyển vào GPU. Toàn bộ CUDApre
và uploaduint8 vẫn nằm trong infer_ms; microbenchmark tách riêng chúng.
Không cộng median từng bước thành thời gian tổng khi decode có overlap.

Lần chạy đầu trả exit75 trước cuda_micro vì swap163,332MiB; sau helper
sudo lần2, swap0 và chuỗi chạy xong. Không có lượt FPS nào được lấy từ
thời điểm trượt cổng đó. Full80 bắt đầu9,24609MiB/kết thúc9,13281MiB,
SDerror0,cùngboot51280481…. Archive17.386.212byte SHA256
`e37697f881f86950b3eeaeab5088943259af34078f05c728199f2bda57ea72af`.

## Bằng chứng và giới hạn

`oracle.json`, `adapter_oracle.json`, `candidate_parity.json`,
`capture_preflight.json`, `pipeline_preflight.json`, `speed_preflight.json`,
`package_verification.json`, `compiled.sha256`,
`capture_evidence/cache/gpu_pre/{quality.json,quality_verification.json}`,
`full80_evidence/runs_absolute/full/score_verified.json` và
`../review155_runtime_accuracy/extra_final_summary.json`.

Lần sao chép package đầu materialize symlink; verifier chặn trước timing.
Giữ package_stage.log và bản sao lỗi; stage_pipeline_v2.py giữ referent,
kiểm lại gói đạt. Không sửa kernel hoặc capture để vượt kiểm tra này.
Chi tiết ở01_package_copy_fix.md. Launcher, cấu hình mặc định và khóa
luận giữ nguyên; chưa triển khai CUDA vào bản sử dụng.
