# R173 — tách sai khác tham chiếu FP32 khỏi engine đang triển khai

R167 rawparity dùng fixture ORT FP32 đã kiểm PyTorch FP32. Nó không so trực
tiếp với engine FP16 cũ. Matched rebuild cũng FAIL cùng ngưỡng6px/.02.
R173 chạy lại đúng10 fixture qua bốn engine đã khóa, không build/tune/train,
không đo FPS, không sửa cổng hoặc rawFAIL của R167.

Trên tất cả50.400 anchor của10 ảnh, original FP16 so ORT FP32 đã có max box
24,25211 pixel và max score0,016283, chỉ4/10 fixture trong cả hai ngưỡng.
Matched control17,03397px/0,013046,4/10. Vì vậy không quy toàn bộ độ lệch
raw với FP32 cho riêng SiLU. Pixel ở đây là đơn vị tọa độ output384×640.

So trực tiếp engine FP16 đang triển khai:

- Precise max box11,75px, score0,024414;8/10 fixture trong6px/.02.
- Fast max box21,5px, score0,023438;4/10.
- Matched control20,5px/0,017578;4/10.

Tất cả vẫn FAIL trên toàn bộ anchor, không bit-exact. Các anchor nền điểm
rất thấp góp phần lớn vào max box. Ở bucket đối xứng mà original **hoặc**
candidate có score>=0,25, precise maxbox0,5px và fast0,625px; không đổi
argmaxclass trên các anchor đó. Maxscore vẫn0,024414/0,023438, chỉ9/10
fixture đạt cả ngưỡng. Bucket này là phân tích hậu kiểm, không được dùng
thay cổng allanchor hoặc coi là hộp detection đã qua NMS.

Kết quả giải thích vì sao maxboxraw lớn có thể đi cùng AP gần giữ nguyên,
nhưng chưa chứng minh mọi sai khác vô hại. ActualAP7190 và full80 vẫn là
kiểm chất lượng chính: precise mAP50−0,09834 điểm%, BDD KF IDSW+7,
UA IDSW bằng baseline. Không dùng10fixture để ngoại suy toàn bộ detection.

`verification.json` đã tái tính từng max/count từ40 outputFP16 và10 output
ORTFP32 được lưu, kiểm engineSHA so R167finalsummary và fixtureSHA/expected
so bản cục bộ. ArchiveSHA
9bf719e2d62b8c57e13f6265a911a73922a397d1effd1e05697b2066bb49fb5e.
Nativejob đã xong trước R172build, sameboot/SD0/health đạt. Không đo tốc độ,
thay launcher/default hoặc khóa luận.

Đọc IEngineInspector bổ sung, không tạo execution context/inference hoặc
engine mới: original/control348layer, precise/fast280. Original chỉ lưu tên
layer; không suy format/precision của nó từ control. Control có77 actSigmoid
với tensor vào/raFP16. Fused có77 PluginV2 và một PointWise classSigmoid cuối.
ElementWise94→17, Reformat62→68; convolution tactics cũng đổi. Do đó không
quy toàn bộ31,45%FPS cho chỉ giảm launch hoặc chỉ giảm copy/reformat.
`layer_inspection_verified.json` kiểm SHA source/engine/layerJSON; số layer
không phải số kernel launch đo trực tiếp hoặc một ablation nguyên nhân.
