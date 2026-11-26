# R162 — resize/letterbox/RGB/CHW/LUT trên CUDA

Khóa trước triểnkhai/đo. R155full80 đãđạt18,804->22,354FPS khi giảm
input320x576, nhưng AP50-1,1624điểm/HOTAgiảm/UAIDSW245->290 nên
chưa khuyến nghị. Profile hiện: CPUpre median-of-clip-medians4,61ms và
H2D1,75ms, GPUYOLO41,95ms. Thử giữ nguyên engine384x640/weights,
chỉ chuyển preprocess sangGPU, upload uint8BGR thay tensorFP32.

Một CUDAkernel: resize tuyến tính8bit tương thíchOpenCV3.2.0 hiệntrên
Nano, pad114, BGR->RGB, HWC->CHW, chuẩn hóa bằng đúng LUT256float32
đãcó. GiữCPUmask/decoder/NMS/ByteTrackKF/5lớp/thresholds/counters/FPS/
buffer/displayoff/150JPEG/warm25/nguồn. Không crop/skipinference/training
hoặc đổi độphângiải. CUDART10.2/nvccsm53, khôngfast-math, khôngclocks.

Precompute interpolation indices vàhệ số11bit cho mỗi source/targetshape,
cache các buffer; đọc thuật toánOpenCV3.2 từnguồn chínhthức:
https://github.com/opencv/opencv/blob/3.2.0/modules/imgproc/src/imgwarp.cpp
MãCUDA mới tựtriểnkhai phépbilinear/separableinteger rounding; khônglấy
sample JPEGkhác hoặc bỏvùngGT để tăngtốc. Nguồn/máyphiênbản khóa.

Trước pipeline: fixturesngẫu nhiên/biên/halving/oddsizes/RGB/padding,
tensorGPU vsactualCPUOpenCVNano, maxerror<=1 mứcuint8 tức1/255;
ghi tỷlệbitexact vàmaxerror, không gọi bitexact nếu cókhác. Kiểm20ảnh
thật cốđịnh dev: rawboxes/score finite/range, soCPU path ngưỡng6px/.02
nhưtrước, statusFAILgiữnguyên. Microbenchmark luânphiên cùngengine và
ảnh, NN chạy trongvòng đểgiữmứcGPUload, báo preprocess/H2D/NNriêng;
không gọi sốmicrobench làend-to-endFPS.

Nếu oraclepixels vượt1LSB: chưađoAP/pipeline, đóng hoặc sửa bugtoánhọc
với version/logrõ, khôngnới tolerance. Nếu hợp lệ, auditactualAP đủ7190
ảnh quaGPUpre, cùng GT/NMS/matchingR155. Cổngkhuyếnnghị.005APmacro/
.02class soactualoriginal, rawFAILkhôngđổi thànhPASS bằngAP. Đo pilot4,
nếu meanFPS nhanhhơn original thì full80ghépcặp/phân tích timestamp và
HOTA/IDSW/FN/FP nhưR155; báo qualityFAIL nếu có, khôngpromote.

Thử nghiệm côlập packagecủaR108, sourcebaseline byteparityR109; không
sửalauncher/default/khóa luận. Khôngbuild/capture/timing đồngthời trên
Nano; chờR160capture xong, cần swapstart<100MB. Tốiđa3job bềnvững.
