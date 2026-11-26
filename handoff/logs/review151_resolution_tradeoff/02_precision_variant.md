# Nhánh FP32 cho khốiattention, trước đoFPS/điểmNano

Engineoriginal vàanchor_rebuilt đã qua10ảnhFP16parity. Engine352x608
trượtngưỡng đãkhóa ởảnh00: maxscoreabsdiff0,0200820565>.02, boxactive
max0,233673px<6. Giữ nguyên cổng vàlog thấtbại; không ghiPASS choengine
này vàkhông mởpilot từ nó. Engine320x576 cũngtrượt ởảnh04:
score0,0239692330>.02,box0,385315px<6. Giữ cảhai log.

Thử cùng một nhánh xửlý precision cho cảhai shape: toàn khối C2PSA/model.10
FP32, phầncònlại choTensorRTtự chọnFP16 nhưcũ. Giảthuyết attention và
FFN nhạy sai số tíchlũy, cần phépđo để xácđịnh, chưa khẳngđịnh nguyênnhân.
Không fine-tune hoặc chọn đoạnvideo/ảnh theođiểm để giảm sai số.

CùngONNX/SHA/input/workspace512MiB;BuilderFlagFP16 và
OBEY_PRECISION_CONSTRAINTS. Chỉsetprecision/outputFP32 cho các compute
layers trongprefix/model.10, giữtensorinteger/shapeops đúngkiểu. Ghi
toànbộlayerđãép,shaONNX/code/engine/flags trướcparity. Không épFP32toàn
mạng rồi gọi đó là engineFP16nhanh; report tên mixedprecisionrõràng.

Tênengine/config mới riêng,r352x608_psa32/r320x576_psa32, khôngghiđèengine/configcũ.
Vẫncùng10ảnh/cổng6px/.02score, thấtbại thì dừngnhánh, khôngdò tiếpblock
hoặc nới tolerance. Nếuđạt mớibenchGPU vàpilot; so vớioriginal và
anchor_rebuilt cùngboot. APFP32devshape352x608 đãchấm từtrước vẫn là
devFP32, khôngtự trở thànhAPengineFP16; trackingNano đoGT riêng.
