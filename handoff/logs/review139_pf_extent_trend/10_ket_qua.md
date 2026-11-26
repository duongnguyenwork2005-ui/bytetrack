# R139 — damped extent trend khi Lost

Hoàn tất36lượtUAdev19/11/2026; **loại vìtrượtHOTA vàFN**.

TrênR132Native100/Rprediction, thayGaussianh-vh lúcLost bằngdampedtrend có
phi=exp(-dt/tau),tau=buffercũ30frameUA/6frameBDD (1,2s), cùngQ. Giữđộnglực
extent đãhọc vàtắtdần thayclampvhngay. Khôngđổilifetime/association hoặcnoise.
Đây làgiảđịnhextent mới, khôngablationthuầnthuậttoánPF.

Numerical500Fsemigroup maxerror4,44e−16;480notLost và360tau0steps khớp parent;
72Lostoracle/PSD/RNG/ownership, giớihạnzero/infinitehorizonđạt. Controls6/repeat6đạt.

UAtrain dev12×3seed0/1/2,36lượt:

- IDSW **466**,seed **460/466/472**;mean<472,2seed<472 nênquaIDSW.
- HOTA **58,7818%**,giảm **0,238điểm** soR12359,0198%;trượt ngưỡng0,1điểm.
- IDF1 **71,3206%**,FN **37.561**,FP **12.113,67**.
- FNtăng **131,67** (khoảng **0,352%**) soR12337.429,33;trượt ngưỡng0,2%.

Rawtáihợp/MOT/CLS/timing/marker/sourcehashđạt ởverification.json. KhôngmởBDD,
UAtest40 hoặcđotiming; khôngtunehorizon/phi/seed/N đểcứukếtquả. Các tậpđãquansát,
báoselectionbias; khôngchọnseed460thay chomeanba seed. Giữsrc/default/Nano/
khóaluận. R140thửomega-ordered stratifiedresampling với môhìnhR132giữnguyên.
