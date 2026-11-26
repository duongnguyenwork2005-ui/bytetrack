# R141 — covariancefloor khi khởi tạo vận tốc từ hai observations

Hoàn tất36UAdev19/11/2026; **loại do trượtIDSW/two-seeds**.

TrênR132Native100/Rprediction, giữpointvelocitymean/initgating/position/RNG,
chỉfloorvelocitycovariance từVobs=(Rprev+Rcur)/T² vàCnew=Vobs+PSD_part(C0-Vobs).
Mụcđíchtránhcovariancenganghướngquánhỏkhiđộdịchchuyểnthấp. Giảđịnhnoisehai
observationsđộclập; đây làconservativefloor, khôngexactcorrelated-measurement
posterior. HookRcurconfidenceNone, Rprevcontext ghiđúngheight/confidenceđãdùng.

Numerical1000SPDfloorcases đạt, mineigenconstraint−3,16e−15 dofloatingpoint;
483disabledsteps khớpParent/RNG; low-speedfloor/high-speednoop, meanunchanged,
PSD/ownership/noisecontext/gapT^-2 đạt (transversevar24,5/6,125/1,53125).
Controls6/repeat6 trênvideođạt.

UAtrain dev12×3seed0/1/2,36lượt:

- IDSW **473,33**,seed **481/473/466**,chỉ1seed<472;trượtmeanvàtwo-seeds.
- HOTA **59,3281%**,IDF1 **72,6173%**,FN **37.440**,FP **12.119**;guardrailsđạt.

Raw/MOT/CLS/timing/marker/sourcehashđạtởverification.json. KhôngmởBDD/test40
hoặcđotiming, khôngtunefloorweight/gain/N/seedđểcứu. Khôngchọnseed466thaymean.
Khôngđổi src/default/Nano/khóaluận. Các tập đãquansátnhiềuvòng, báoselectionbias.

AuditNPbackend sau đó xácnhậnSTrack.update/re_activate khôngtruyềnscorevào
filterupdate, nênRprevconfidence trongreplaythựcNone dùhàmnoisehỗtrợscore.
Đây làfixedR của benchmarkcũ, khôngtựtuyênbốkếtquảcũsai. R142thửadaptertruyền
currentscore đểkíchhoạtR(c) cósẵn; R142khôngghépcovariancefloorcủaR141.
