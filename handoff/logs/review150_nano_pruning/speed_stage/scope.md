# Khóa trước điểm FPS pruning

R150 đo đủ actualAP7190 và GPU/4clip pilot cho original/anchor/control/
channel25_KD; status rawparityFAIL của pruned giữ nguyên. Tốc độ/quality
chỉ được đo diagnostic nếu APcap trượt, không promote hoặc nới cổng.

Nếu meanFPS pilot pruned>original, đo đủ80clip **original/control/pruned**
để tách tác động kiến trúc với matched dense control. Anchor rebuilt
giữ GPU/pilot (cùngweights, kiểm build), không chạy lại80 nếu không cần.
Nếu pruned không nhanh hơn original, kết thúc ở pilot như một screen âm;
không gọi có kết quả80 hoặc suy FPStừ giảmparams/byte/MAC.

AP recommendation vẫn cap.005macro/.02class sooriginal vàmatchedcontrol.
Không đổi sourceR145, detector/GT/classes/warmup/150JPEG/manifest/ByteTrack
KF/NMS/counters/metrics/guard, không skipinference. R160 dùng rule riêng
đãkhóa trong00_protocol.md. Model/config/selector/hash khóa trước timing.
