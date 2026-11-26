# R155 — audit độ chính xác detection thật của TensorRT trên Nano

Ngày20/11, trước khi xem AP TensorRT. R151 F32 CPU đã chấmoriginal384x640
trên7190dev: mAP50=.4705447,50-95=.3197891. Cácengine352/320 FP16 và
PSA32 trượt ngưỡngmaxscoreparity.02 ởmột sốanchor; boxactive sai<.4px.
R151 vẫn trượt cổng đó, không gọi biếnthểPSA32 đãsửa được vấnđề, không
ghiđèconfig/log hoặc tuyên bố pureFP16bitparity.

R155 là audit độchínhxác engine thực, cần cho yêu cầu so AP/FPS khi xửlý
môhình. Maxsai lệch ởmộtanchor không tươngđương mứcgiảm AP. Khóa tấtcả
engine đãbuild trước điểmAPmới: originalR109,anchor_rebuilt,352/320FP16,
352/320PSA32. Khôngtune block/precision/tolerance theo điểmAPTensorRT.
R145control/pruned saukhi xong là mộtbatch riêng cùng dữliệu/giao thức,
giữ mọi kếtquảâm. Không gọi cácnhánhR151 cũ đạt cổngsốhọc của chúng.

Dữliệu đúng7190mixed5dev từmanifestR151, khôngđổiảnh/nhãn/split/ignore.
GT nămclasscar/truck/bus/motorcycle/bicycle; ảnh đãgraymask theoexporter.
GiữJPEGbyte vàhash,đóngtar, Nano đọcJPEG trực tiếp từtar để không nhân
bản7190file trênSD. Kiểmimdecode/imread pixelparity10ảnh. KhôngdùngGT
đểchọnmôhìnhhoặc sửaweights, chỉ tính AP và eligibility đãđịnh trước.

Nano: đúngsourceletterbox/LUT vàTRTEngine cudart của góiR109. KhóaSHA
engine/ONNX/code/archive/dataset trước capture; static1x3xHxW,raw1x9xN,
finite vàscore[0,1], lớp argmax5class. Thu candidateconf>.001 trướcNMS
(xyxy/conf/class float32), offsets/index/shape theochunk100ảnh, SHA và
marker. Không làmNMS hoặc đoFPS trongcapture; không gọi tốcđộdump là
FPSpipeline. Đủ7190uniqueindices, khôngdropframe. Córesume từngchunk
hoàn tất saukiểmhash, giữtmpchưa xong, khôngghiđèbằngchứng.

PC: đúngNMSgreedy/matrix của snapshotR109, IoU.7,maxdet300; scale/clip
theoimagegốc. Matching/AP dùngDetectionValidator/DetMetrics hiện tại,
không đổi metrictracking. P/Rc25rematch sauconfidencefilter, IoU.5.
AP50/50-95/perclass +GT/pred/TPc25, không gọi P/RbestF1 làP/Rc25.
Fixturebridge raw->candidates->NMS phải khớpdecode_nms gốc trước chấm.

Cổngứngviên theo **APengine thật** sooriginalFP16: mAP50/50-95 không
giảm>.005, AP50mỗi5class không giảm>.02. Báo cảrawparityR151fail và
APengine; runtimeaudit không biếnrawfail thànhPASS. Chỉứngviên qua cổng
AP này mới xét phépđo tracking/FPS riêngR155 (không tự mởR151pilot đã
khóa). Pilot4đoạn chọntrước indices0,10 mỗiBDD/UA; ứngviênpilotmeanFPS
cao nhất quaAP mở80đoạn cũR109, nguyênKF/threshold/ignore/FPS/ảnh,
khôngdropinference. ĐoGPUriêng vàend-to-end riêng, khôngbuild cùngđo.

NếuchẳngứngviênquaAP hoặc khôngnhanhhơn thì báokếtquảâm, khôngnới
cổng/chọnclass/đổiGT đểthắng. Khôngdeploy/đổilauncher haykhóaluận.
Đây là develigibility trên dữliệu đãthấy nhiềuvòng, không blindtest.
GTmotorcycle chỉ226,bicycle407box; cần báo sốmẫu, không khái quát rộng.
R155 chỉdetector/NanoBDD-UA, không mởtrackingdatasetskịchbản khi cổng
Gaussian/PFBDD-UA của người dùng vẫn chưađạt.
