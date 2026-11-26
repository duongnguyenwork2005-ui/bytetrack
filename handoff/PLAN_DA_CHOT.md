# Kế hoạch đã chốt — từ vòng 27 (2026-09-29)

Nguồn: yêu cầu của người dùng sau khi Codex nghiệm thu `239b9c6` (**PASS WITH LIMITATIONS**, `handoff/CODEX_REVIEW_239b9c6.md`).
File này là phạm vi làm việc cho các vòng sau. Thay đổi phạm vi phải do người dùng quyết định.

## 0. TRẠNG THÁI HIỆN TẠI — tài liệu trạng thái chuẩn (cập nhật 2026-11-26)

**Sao lưu GitHub26/11 — source đã ở remote; payload đang tải, chưa đóng mốc.** Commit `55916950112bca9ffd4327e554c4fa9a2faba346` và tag `backup-20261126-source` trên origin/phase2-mixed-traffic đã đối chiếu SHA đạt. Snapshot20940tệp source/config/LaTeX/PDF/slide/báo cáo, stagedbytes khớp workingtree trướccommit; manifest ở handoff/backups/20261126/git_snapshot_manifest.tsv thuộc tag nguồn này. Lần push lớn đầu HTTP408, giữlog; hồi phục bằng29nhóm qua privateindex và nhánh codex/backup-upload-20261126, khôngreset/force/rewrite workingbranch. Payload đã đóng góiPASS160391tệp/19.672.354.408byte logic,118157object duy nhất, gồm145model bịignore bổ sung;30ZIP tổng9.406.322.112byte. Manifest/centraldirectory mapping/restore byte-roundtrip/Unicode/longpath/duplicate/empty/refuseoverwrite đạt. Chia48MiB để tránh timeout; ReleaseNHÁP459160982 đã tạo, hai manifest mapping đã upload và SHAserver đạt. Worker hiện tại upload_release_parallel.py4luồng đang tải các gói còn lại (session55164); không coi status nhánh phụ/metadata subset là allcomplete. Engine/ONNX/plugin Nano15tệp đã capture/hash trước–sau và kiểm từng member tar, archive89.978.432byte SHA258d3d27ca0001d5358f4343f2d60169abff774f3e3520177c7e56cff83bbb65,3asset của nó cũng đã upload/hashserverPASS; chỉ backup file được whitelist, không toàn bộFSNano. State/log/ZIP đầy đủ ở C:/Users/Mikuno/AppData/Local/Temp/bytetrack-backup-20261126/; release_state.json,parallel_upload.log,chunk_push_state.json,payload/package-complete.json và native_release_state.json là điểmresume. Chờ all201asset payload đạt rồi táikiểm remote/cậpnhật closeout/helper/tag cuối. Dataset data/,tmp/,results/track_video/,handoff/archive/,các tệp ignore khác/ổngoài không thuộc snapshot, ngoại lệmodel local và whitelistNano như trên. Không sửa nội dung quyển, thuật toán hoặc mở benchmark/training trong lượt backup.

**Sao lưu GitHub26/11 — đang thực hiện theo yêu cầu người dùng.** Sao lưu source/config/sourceLaTeX/báo cáo/các summary vào nhánh phase2-mixed-traffic, giữ byte nguyên gốc bằng attributes cho các vùng mới; không sửa thuật toán hoặc nội dung khóa luận. Kiểm kê ngoài data/tmp/video/archive nặng:181172tệp/17,28GiB; chọn20934tệp/330,76MiB choGit,160246tệp/16,96GiB bằnggói sao lưu Release dạngNHÁP vìrepo côngkhai. Khôngpháthiệnprivatekey/GitHubtoken/AWSkey/UTUAVpassword/Authorizationliteral trongtệp chọnGit. Các dataset gốc/xuấtYOLO/tmp và tệp bịignore khác không thuộc snapshot; model trongmodels được bổsung riêng vào private-release. SourceLaTeX ở khoa_luan_prism/ phải có trong commit. Đang tạo snapshotcommit/tag và sao lưu payload; chỉ ghi hoàn tất sau khi kiểm remoteSHA và asset. Không đổi lịch sử/forcepush. Lưuýthầyđãchốt: cùngtrackerByteTrack, KF tuyến tính làbaseline đểsoEKF/UKF/PF; mục lục đãsửa trongchat, chưađổiquyển.

**Khóa luận26/11 — chỉ đọc hai bản mẫu và đề xuất mục lục để người dùng duyệt.** Yêu cầu mới thay tác vụ đang ưu tiên: đọc `C:/Users/Mikuno/Downloads/22022164_LeQuocViet_DATN.pdf` và `C:/Users/Mikuno/Downloads/ĐATN_22022161_NguyenTrongNam_last.pdf`, gửi mục lục trước; chưa chỉnh sửa quyển/sourceLaTeX. Đã trích xuất bố cục/mục lục và đối chiếu phần mở đầu/kết luận từng chương: Việt5chương, Nam6chương. Đề xuất sáu chương: tổng quan; cơ sở lý thuyết; thiết kế hệ thống và các bộ lọc; triển khai/tối ưu Nano; thực nghiệm/đánh giá; kết luận/hướng phát triển. Hai kịch bản chính phi tuyến và che khuất; xe máy là nhóm đối tượng phân tích trong cả hai. Chưa chốt dataset kịch bản mới hoặc hứa mọi bộ lọc phi tuyến đều thắng. Chờ người dùng duyệt/chỉnh mục lục trước khi sửa quyển. Không chạy benchmark/training mới trong lượt đọc mẫu này; nguồnPDF và filequyển giữ nguyên.

**R174 hoàn tất26/11 — CUDA Graph trên SiLU giảm0,43010% thời gianGPU, không mở pipeline.** Theo yêu cầu tiếp tục, SSH10.11.50.200 đạt, bootdccc433e-ae31-4ca5-9e13-0d89f0f00195. Protocol `logs/review174_silu_graph/00_protocol.md` khóa trước điểm: bốnmode original/graph vàpreciseSiLU/graph,cùngengine/pluginSHA R167,ba lần/mode,12lượt trtexec,warm500/duration6/iterations100. `verification.json` táiparse12log/hash/Enabled/no-fallback/guard/sourcePASS; medianGPUtime trungbình original41,63740→graph41,33893ms(−0,71682%),precise30,28003→30,14980ms(−0,43010%). Enqueueoriginal15,81697→1,26703ms,precise11,09810→1,50134ms; không cộng enqueue vớiGPUtime hoặc gọi đó là FPS. Repeatspreadtốiđa0,20437%,cùngboot/clockpolicy,swap0,nhiệttốiđa50°C,SD0. Cảhaicặp trượtcổngGPUgiảm>=5% đãkhóa, khôngtriểnkhai runtime/khôngđoFPS/AP/trackinggraph. KhôngNsighttrace; captureevidence làlog+nguồn samples đi kèm+enqueuegiảm, khôngchỉcờEnabled. Lỗitarv1 `.: file changed as we read it` xảyraSAU12lượtcomplete,giữbằngchứng; collect_v2 chỉđónggói danh sáchfile,khôngrerunbenchmark. Archivev2 33.909byte SHA a34605cd2daea5e8e87da5b02707531027e2663d276cae8d77d15fa1a3a4d477. Cuốilượt Nanoavailable2909MiB/swap0/SD0,khôngjobGPU tự mở cònchạy. Đãcậpnhậttrạngtháitrướcbáocáo10_ket_qua.md. R172report vàbiểuđồ8điểm FPS80/AP đãxong,sourcevalues/staticchecksPASS,khôngkiểmrenderUI. R16726,90604FPS vẫn làứngviên đãđo tốt hơn vềđánhđổi, khôngđổidefault/launcher/khóaluận/cổngPF. Training10epochvẫnchưahoàn tất; khôngtựrestart đểcứuđiểm.

**ĐÃ TIẾP TỤC theo yêu cầu người dùng 26/11; R172 hoàn tất và bị loại về chất lượng.** SSH10.11.50.200 đạt, boot mới dccc433e-ae31-4ca5-9e13-0d89f0f00195, swap0, RAM available2895MiB, SDerrors0, không benchmark cũ còn chạy. Không thay boot lịch sử trong evidence bằng boot hiện tại. R172 archive23.620.003byte SHA a62cae8257d3a76a094fafa07e1f723fa595d94d467fed17ad0b7095d7d892ed; scorer/summarize PASS240lượt/36.000inference/80đoạn. MeanFPS original/control/joint50 20,47635/20,25659/22,26731; joint +8,74648% sooriginal,80/80nhanhhơn; BDD+8,86498%,UA+8,62303%. IDSW BDD1534/1657/1643,UA245/305/401; HOTA joint42,33337%/63,63511%, giảm1,38547/2,93808điểm sooriginal. AP50joint40,90797% (giảm6,23876điểm), macro/perclasscapsFAIL, rawcontrol/jointFAIL giữ nguyên; common7secondary, primary10epoch chưa hoàn tất. Nguồn final_summary.json và full80_evidence/runs_absolute/full/score_verified.json trong logs/review172_common7_pruning. Đã cập nhật trạng thái trước báo cáo10_ket_qua.md. Bước tiếp: cập nhật biểu đồ bằng FPS80 đã xác minh, kiểm CUDA Graph với engine SiLU R167 hiện tại theo protocol mới trước điểm; đây là kiểm runtime, không thêm detector/precision/weights. Không đổi default/launcher/khóa luận hoặc cổng PF. Các đoạn trạng thái đang chạy phía dưới là lịch sử21/11/lúc thu26/11, không phải job hiện tại.

**Tiếp tục 26/11 — đã kết nối lại Nano, R172 full80 đã hoàn tất trên thiết bị; đang thu và chấm trên PC.** SSH trực tiếp `10.11.50.200` qua khóa/HostKeyAlias cũ đã đạt sau timeout ban đầu; boot hiện tại `e737167e-437b-4b97-ae6f-5f7819add129`, swap0, RAM available2944MiB, SDerrors0. Đây là boot sau lượt đo, không thay boot ghi trong evidence bằng boot hiện tại. Native `speed/runs_absolute/full/complete.json` PASS240lượt; collection đã có archive23.620.003byte SHA256 `a62cae8257d3a76a094fafa07e1f723fa595d94d467fed17ad0b7095d7d892ed`. Không chạy lại FPS hoặc suy kết quả từ pilot. Script mới `logs/review172_common7_pruning/collect_after_reconnect_20261126.ps1` chỉ xác minh archive, thu/extract và gọi nguyên R155 scorer; giữ source/config/model/metric đã khóa. Hostsession33414 đang tải; chưa có score_verified/final_summary R172 tại thời điểm ghi. Không có Native timing/training khác. Sau scorerPASS chạy summarize.py, cập nhật trạng thái trước báo cáo. R16726,90604FPS/mAP5047,04839% vẫn là kết quả full80 đã hậu kiểm; R172 detection40,90797% vẫn FAIL chất lượng, primary10epoch vẫn chưa hoàn tất. Không đổi mặc định/launcher/khóa luận hoặc cổng PF.

**R172 actualTensorRT7190 đã xong; guard swap đã phục hồi, bắt đầu timing.** Sau Desktopupdate, host98205 đã thu/checkarchive và chấm AP, thoát0 với `logs/review172_common7_pruning/timing_pending.json` WAITING_FOR_TIMING_GUARD(swap310,625MiB>=100). NativePID16750 đã xong đủcontrol/joint50 mỗi7190ảnh, khôngcònNativeGPUjob. ActualAP50/AP50–95 original47,14673/31,98483, control7 **46,47791/31,22281**, joint7 **40,90797/26,67362%**. Student−6,23876điểmAP50 sooriginal,−5,56995socontrol; mọi macro/perclasscap trượt, giữFAIL. `actual_quality_comparison.json` hậukiểmhashPASS; params−22,05576%,ConvMAC−24,09253%,ONNX10.500.936→8.218.728byte(−21,73338%). Đây làcommon7secondary, primary10vẫnchưahoàn tất; chưaFPS/trackingR172. Speedstageđãupload. Ngày21/11 lúc20:35user tiếp tục: swapthực~309,68MiB, sudo-nfail(passwordrequired). Đãchuẩnbị/review/bash-n scriptlocalrestore_swap_if_needed.sh, remote~/restore_swap_r172.sh SHA9ad5b040452470623f6a03257cb20b55354d5d46e7745f325089b1592ed116c1, chỉchạysauNativeidle/complete vàRAMheadroom>usedswap+1GiB,giữ/bậtlại /swapfile và4zramcùngpriority; khôngđổiclock/network/config. **Userđãchạyhelper, đãkiểmđủ5swapdevices/prioritytrởlại, swap0,available2863636KiB, sameboot/SD0**; logvàSHAkhớp, `timing_guard_restored.json` PASS. Hostsession25565 đangchạynguyên `finish_timing.ps1`: engine6repeatđãxong, pilot4meanoriginal20,48754/control20,32991/joint50 22,32907FPS, positive→đãmởfull80(3model,240lượt/36000inference dựkiến). ChưaFPS80/IDSW; khôngdùngpilotthayfull. HiệnmộtNanoGPUjobnày, khôngjobNativekhác. Không nới guard; sauđóchấm và `summarize.py`. TrongkhiđợicóthểhậukiểmPC, khôngrestarttraining/Nativecapture.

**R169 đã dừng do lỗi CUDA tái diễn; chưa có kết quả pruning hoàn chỉnh.** R164joint50 dừng epoch8(zero-based7), sau291batch, CUDAillegalmemoryaccess phát hiện ởmake_anchors; Windowsnvlddmkm13/153 trùng15:46, chưa xác định rootcause. Giữlog/source/runs/contract cũ, hostsession94810 thoát1, khôngcoilàtrain10đãxong. Epoch6 của control/joint50 cóEMA/optimizerfinite; freshCUDAhealthvàdiagnostic32KDforward/backwardmỗimodel, CUDA_LAUNCH_BLOCKING1 PASS (`logs/review169_paired_cuda_resume/diagnostic.json`). Khôngđổidriver/registry/TDR/clock.

ProtocolR169khóa trước điểm: cảhaiarm restart từepoch6EMA/optimizer, cùngseed164/blocking1, chỉ3epoch7/8/9 đểtổng10accepted. CheckpointUltralyticskhônggiữrawtrainingmodel/RNG đầyđủ nên **khônggọi bitexact continuation**; restartđối xứng đểtránhdùngcontrolkhôngrestart. Controlcũ7..9/selectionđãgiữnhưngloạikhỏipairedprimary. Chọnmonitortrên7checkpointcũ+3mớimỗiarm,cùngR164rule; saucảhaiselectionmới7190dev. TệpR169riêng, hash tấtcảcheckpointcopy/source/data/EMA/optimizerverified; resume-copychỉđổidata/project/name/save_dir,vẫn5200train/300monitor/SGD/lr/rect384640/batch4/KDR145. SourcecachelabelR169riêng, crashawareCPUbatchsnapshotnếu lỗitáisinh. **Hostsession86330 đã thoát1 lúc17:04 21/11**: control epoch10(zero9), sau777batch, CUDNN_STATUS_EXECUTION_FAILED tại cudnn_batch_norm_backward dù blocking1; Windowsnvlddmkm13/153 cùnglúc, chưa chứng minh rootcause. Giữ control_failed_cpu_batch.pt (2.953.176byte), traceback và windows_events_recurrence.json; epoch7/8 đã hoàn tất nhưng study10epoch chưa xong. Joint50 tail chưa chạy. Hostsession28359 đã tự thoát1 vì failedbatch, không mở Nano R169. Không chạy lại recovery cũ; bước kế là fresh-process diagnostic batch đã lưu và backend BN, không sửa driver/TDR/clock. Runtimeharness R169 đã chuẩn bị/compile riêng (pipeline_source_derivation.json), chưa chạy Nano; hostsession28359 đã dừng do failedbatch, không có R169 Nano runtime đang đợi hoặc chạy. Không chạy Nano R169 trùng R167 fulltiming. Có timing_pending guard nếu swap>=100MiB; không tự nới cổng. Modelđầu r169_paired_resume/ sẽcónếuexportđạt. Khôngsửanguồn/default/launcher/khóaluận/metricđãkhóa; tốiđa3job; các jobR164/R169 đã dừng, hiệnR172Nano duy nhất.

**Kết quả R167 đã hậu kiểm:** đủ320lượt/48000inference,40BDD+40UA,150JPEG/đoạn,warm25,KF,displayoff; hostsession61179 thoát0, `logs/review167_fused_silu/final_summary.json` và full80 score_verified PASS source/hash/TrackEval/sameboot. MeanFPS original/control/precise/fast **20,46866/20,27988/26,90604/26,81362**; precise +31,44997%, fast+30,99841%, cảhai80/80nhanhhơn vàmin24,09608/24,05143FPS. BDD precise20,87494→27,59523(+32,19312%); UA20,06238→26,21686(+30,67673%). Medianpairedprecise31,81125%,CI95[31,60529;31,91575]. AP50/AP50–95 original47,14673/31,98483, precise47,04839/31,95699, fast47,13934/31,94924%; cảhai macro/perclasscaps đạt sooriginal+matchedcontrol. KFIDs BDD1534→1541precise/1539fast; UA245→245/248. HOTA BDD43,71884→43,80858/43,70396; UA66,57320→66,54156/66,59263%. **Không gọi tracking byteexact**; rawparity vẫn FAIL: precise maxbox13,78397px/maxscore0,021777, fast17,04010px/0,013953 (giữ6px/.02). PipelineGPUexecmedian41,91736→30,47261ms(precise), infer44,63364→33,16931ms; decode chồnglấp, khôngcộngmedian. Samebootf52d349d…, SD0, swapcuối90,51953MiB; archive31.167.855byteSHA34d38a1330dabbbe02ece5692b6991652e3041e7f94e13a4c7ea8e9257a2ae64. Hậukiểmposthoccount theo40video: preciseBDD+7IDs gồm11tăng/9giảm/20bằng,bootstrap5000seed167 CI95sumdelta[-19;35]; UA0 gồm5tăng/7giảm/28bằng,CI[-12;14], chưachứngminhkhácbiệtổnđịnh/khôngphảichứngminhnoninferiority. `tracking_uncertainty.json`, giữHOTAaggregateTrackEval, quality150frame/FPS125steady. Đây làfusion77activation/giữweights,khôngphảipruninghoặcINT8mới. Gói thử, khôngđổidefault/launcher/khóaluận.

**R173 hoàn tất hậu kiểm tham chiếu raw10fixture.** Engine/fixture/SO nguồnR167 giữSHA, Nativejob đãxongtrướcR172build, host`logs/review173_deployed_raw_audit/verification.json` táitính từNPZ PASS; archiveSHA9bf719e2d62b8c57e13f6265a911a73922a397d1effd1e05697b2066bb49fb5e. RawR167 thựcsoORTFP32 đãkiểmPyTorch, **khôngphảisoengineFP16deployed**. OriginalFP16 cũngFAILallanchor với24,25211px/.016283; control17,03397px/.013046. So trực tiếp deployed, preciseall11,75px/.024414, fast21,5px/.023438; **vẫnFAIL**, khôngđổicổng6px/.02 hoặcR167kếtluận. Trongbucketmà1trong2score>=.25 trên10fixtures, maxboxprecise.5px/fast.625px(networkunits), khôngđổiargmaxclass, nhưngmaxscore.024414/.023438 làm9/10trongngưỡng; khônggọibitexact. Lệchboxallanchor lớn phầnlớnởbackground khôngđủkếtluậnsilu gâytoànbộlệch; actualAP7190/full80mớilàphépđochấtlượngchính. Đây làposthocdiagnostic, khôngFPSmới/khôngpromote. StaticIEngineInspector bổsung(cùngengineSHA,khônginference) đạt: original/control348layer, precise/fast280; control77actSigmoid cóinput/outputFP16, fused77PluginV2+1classSigmoid. Reformat62→68 tăng, convtacticsđổi, khôngquytoànbộ31,45%chochỉgiảmkernellaunch/formatcopy; `layer_inspection_verified.json`. OriginalchỉLAYER_NAMESmetadata nênkhôngsuyprecisionorig từcontrol.

**R171 hoàn tất 256 KD forward/backward chẩn đoán, không tái hiện lỗi batch đơn.** CapturedbatchR169 hợp lệfinite/imguint8/31GT/5class/bbox[0,1]. Bốn freshprocess control/joint50×cuDNN/nativeBN mỗi64PASS, grad_fn chứng minh CudnnBatchNormBackward/NativeBatchNormBackward thực; khôngoptimizerstep/khôngsavecheckpoint. `logs/review171_cuda_batch_diagnosis/verification.json`. KhôngkếtluậnđãfixcuDNN hoặcdriver; Windowsrecurrence cóGraphics SM Warp Out Of Range Address, chưachứngminhrootcause. Khôngupdate driver/TDR/clock/đóngapp. R172 secondary đangchuẩnbịđánhgiácheckpoint6R164củacảhaiarm(7epochliêntụchoàntấtchung), khôngtrainmới/chọnepochtheoAP/khôngdùngR169resumed hoặccontroltail10. Primary10vẫnFAIL; protocolkhóa `logs/review172_common7_pruning/00_protocol.md`, Hostsession38973 đã thoát0: prepare/3dev7190/export/10ORTfixturemỗimodel PASS, 00..6haiarm cốđịnh, khôngtrainmới. PCFP32AP50/AP50–95 anchor47,05447/31,97889, control7 46,46731/31,24541, joint7 40,98639/26,73237%; student mất6,06808điểmAP50 soanchor, khôngdùngđiểmPC thayactualNano; runtimeharnessR172 riêng đãcompile/pathderive, CodexDesktopupdate lúc17:54 làm mất localhostsession99297; nativePID16750/buildchild16845 vẫn chạy. Host mới98205 (`resume_host_after_update.ps1`, host_resume_derivation.json) chỉđợicomplete cũ rồi nối lại nguyên tailcollection/AP/timing, khôngrestartNative/sửasource/gate. ControlbuildPASS326,75s, đangjoint50build; sauđócapture/actualAP/pilot/full theo00. Stage32.890.880byteSHA2d729cef82a6acb677746a3b63c767b391e7c70e4c6f96018585a4b7f6159c02khớp. ActualAP đã hoàn tất và trượt caps, FPS/tracking chưa đo; xem guard swap ở đầu mục0; cùngR162CUDApre và guard<100MiBswap, khônggộpSiLU vào pruning.

**R170 — hoàn tất kiểm nguồn dữ liệu đường công khai, chưa mở tracking.** Link tác giả Accid3nD public Nextcloud đọc được, không dùng tài khoản TUMTraf-V2X đang tạm dừng. Gói 5ZIP tổng46.587.516.755byte; Range/ETag/index kiểm đủ5, chỉtải41.835.824byte danh mục, không tải46,6GB. 111700JPG/111656JSON, 45nhóm sequence/camera,95345cặp cùngfilename. `logs/review170_public_roadside_audit/inventory_verified.json` PASS metadata; không gọi đây là PASS converter/benchmark. Mẫu ảnh1920×1200+OpenLABEL CRC/SHA đạt, 20object(16CAR/4TRUCK), UUID/bbox/full_mask/timestamp/image_file_names có, không có thuộc tính che trong mẫu; chưa suy sốclass toàn bộ hay semantic amodal. Đọc thêm một JSON đầu mỗi8nhóm không khớp: bốn mẫu image_file_names trỏ đúng ảnh với camera alias, bốn mẫu r09_s05 vẫn trỏ ảnh không có, không tự ghép gần nhất/offset345s. Filename cadence có gap, không giả định released25Hz đều; chưa sửa tracker/metric để bù. V1 dừng CD>2MiB giữfailedindex, v2 chiaRange<=2MiB/response,<=20MiB/ZIP,CRC/sample8transport đạt; không nới qualitygate. R170 không GPU/job nền còn chạy, không có điểm mới chứng minh E/U thắng; đốichiếupaper§4.3 vàsuppB.3 xácnhậnnhãnsemi-autoYOLOv7/MonoDet3D→PolyMOT+manualinspection/correction, khônggọi100%manual hoặc100%pseudo-unreviewed. CrossVisionauthorabstract23.128ảnh/83.026bbox6classcôngbốdetectionYOLO/COCO, chưaxácminhID/cadence choMOT; khôngnhầmCOCOannotationidvớitrackid, khôngmởportalđăngkýđangtạmdừng; cổng GaussianfullUA/PF giữ nguyên. Bước sau nếu ưu tiên: xác minh mapping toànnhóm và liên tụcID/timestamp trước converter, khóa nhóm bằngGT trước điểm.

**R165 hoàn tất dev36 — resample–move không đạt mục tiêu IDSW.** Parent R153 N100, giữ Q/R/init/ESS/association; một independent MH move sau resampling cho incomingomega, target Gaussian likelihood thật×prior/q, ancestor/eta đúng. Numerical gồm off bitparity, 560 SciPy likelihood, 80 detailed-balance, 4000 optimal-linear, 50.000 CT invariant draws, native ancestry/PSD/shape; broad synthetic 1658accept/342reject, 2000 posterior oracle expm/Joseph max2,985e-13 PASS; source/cache/GT và smoke offprefix/onrepeat đạt. Đủ36 marker/raw/hash ở `logs/review165_pf_resample_move/dev/{summary,verification}.json`, hostsession87886 thoát0. PFmean IDSW **436,66667** so parent437,66667, E435/U430/KF449; HOTA **61,227764%**,FN35068,66667. HOTA/FN đạt nhưng mean/two-seeds<=430 trượt; giảm1IDmean, bootstrap paired12video CI95[-3,3333;+1,3333], chưa chứng minh lợi thế ổn định. Seed446/431/433; parent439/435/439. `final_verification.json` kiểm lại source/36MOT/CLS/timing/marker/raw PASS, gate vẫnFalse; 916600MHproposal,916182accepted(99,9544%). Không tune số MHsteps/scale/N/seed/ESS, không promote/default/full/scenario; prerequisite GaussianfullUA chưa đạt. Đây là dev đã sử dụng nhiều vòng. Phương pháp tham khảo Gilks/Berzuini2001, không phải mã tác giả.

**R166 hoàn tất hậu kiểm full BDD200/UA40 — 1920 output, không tracker mới.** `logs/review166_full_scenario_audit/{summary_v2.json,partial_v2.json,inputs_v2.json}`; nguyên GT turn25/straight8 R99 và CLEARtrace R113, mỗi output khớp rawIDSW/CLR_FN, hash source/MOT/GT trước/sau đạt. V1 dừng tên key FN (raw dùngCLR_FN), giữrun.py/run.log/inputs; 01_raw_schema khóa bridgev2, không đổi metric haygroup. KF/Ecommon/Ucommon/Efloor/Ufloor/PF142mean3seed: **BDD toàn200 IDs5319/4854/5028/4947/5075/4788,33**, nhưng nhóm rẽ25 **268/277/277/278/275/278,33** trên15649GT; FN11946/11981/11975/11979/11970/11970,67. BDDmoto rẽ chỉ39GT,36FN vàIDSW0mọimode, không đủđánhgiá. **UA40 toàn IDs2319/2359/2372/2374/2360/2362,67**; rẽ25 có2432GT, IDs11/9/9/10/9/9, FN800/800/806/800/804/799,67; pairedCI E/Ucommon−KF[-5;0], mẫu ít vàchứa0. Visibility<=.5 có69504GT, IDs313/321/331/328/343/345; FN57166/57176/57188/57203/57171/57178,67. Recent1sec IDs672/682/685/693/694/707,33; không ủnghộhaiGaussian cùngthắngchekhuấtfull. Fullcohort lớn hơn khônglặplại hếtlợi thế old80windowsR113; khônggiấu nhómthua/khôngđổiprerequisite. Detector đây làcacheYOLOv8nCOCOc25 cũ, không trộn vớiNanoYOLO11n80đoạn. Hostsession36238 thoát0; không sửa khóa luận/default.

**R167 hoàn tất full80 — CUDA SiLU đạt26,90604FPS (+31,44997%), có sai khác tracking nhỏ.** 77Sigmoid×input thànhplugin, unchangedTensorProto/shape384x640/5class/3scale; preciseexpf/fast__expf khóa trước điểm ở02. V1namespace compilefail giữlog;v2namednamespacevàcảhaiF32/HALF/oddtail1048577oraclePASS. Source/helper/libraryhash hostauditPASS (`screen_verification.json`). GPUthuậnđảo original/control/precise/fast **41,62130/42,08935/30,28195/30,39035ms**, time giảm27,2441%/26,9837% vsoriginal; repeatmax0,139%. **Rawboxparity cảhaiFAIL(giữ6px/.02 khôngnới)**, vẫn actualAPdiagnosticđúngscope. Đủ3×7190 actualCUDApreR162, source/candidateindex/NMS/APmatchhashPASS; mAP50 control/precise/fast **47,032587/47,048392/47,139336%**, AP50-95 **31,961086/31,956989/31,949238%** sooriginal47,14673/31,98483%; cảhai macro/perclassAPcaps đạt sooriginal+control. Khônggọibitexact/khôngsửaCUDAmathđểcứuAP. Native~/nano_run/r167_fused_silu_20261121, `actual_evidence/`, hostsession75237thoát0. Packagecloneđãkiểm byte/symlinkreferents, chỉthêmloader và2SO, khôngmaterializedata/modelslinks. speed_stage/boundconfigthêmcollector/envsourcegiữS1, hostsession61179 đã hoàn tất full80/hostscore và finalsummary; xem số liệu đã xác minh ở đầu mục0. Pilot4 giữ riêng, không dùng thay full80. Cùngpackagechoallmodel, originalMOT/CLS trùngR109 cả80; khôngsuytừ1pilotvideo. Khôngđổidefault/launcher/khóaluận.

**Điều kiện Nano lúc bắt đầu lượt sau người dùng tiếp tục (sốswap/avail lịch sử, không phải hiện tại):** SSH192.168.1.5, boot **f52d349d-e4ce-4413-abee-d9f0f02dac2e**, swap0,avail2790MiB,SD0; bootđãđổisoGPU screen51280481…, khôngtrộntimingkhácboot. CPUgovschedutil,GPUmin76,8/max921,6MHz; protocolR162khôngđổiclocks, finalfreshpairedcùngsetting/boot vàguardcũ. sudo-nkhôngcóquyền, nhưngswap0đạtcổng<100 nên **khôngcần hỏilạihelper chỉvìtrạngtháicũ125MiB**. R167/R173 đãxong; R172 PCevaluation/exportđãxong, hiệnmộtNativeGPUjobR172build/capture. KhôngmởNativejobkhác. Saucontrolbuildswap98,64453MiB/avail2842,36MiB; kiểm lại guard100sauactualAP, không dùng sốswap0lúcđầu thayhiệntại. Rootmodel/ONNX/.so còn nguyên sau khởi động lại.

**R168 LHS PF hoàn tất dev36, không đạt.** N100guidedR153, lấy randompermutation+jitter/strata qua inverseNormal cho predict/guidedinnovations, khôngđổi marginals/Q/R/init/ESS/association, khôngchồngMH. Numerical offRNG/init/Lost/zero,400stratacheck,4000batchnormalmarginal/importanceevidence/PSD/hookrestore vàsmokePASS; source/raw/36markerđạt. SeedIDSW443/442/439,mean **441,33333** soR153437,66667,E435/U430; HOTA61,234132%,FN35068,66667. HOTA/FNđạt nhưngmean/twoseedstrượt. Hostsession92424thoát0; khôngtune/khôngfull/scenario. `logs/review168_pf_lhs/dev/{summary,verification}.json`. NumericalsamplemeanvariancegiảmkhôngđồngnghĩaIDSWgiảm; khônggọiđâylàSQMCfull. Finalaudit/reportđãxong: `final_verification.json` source/36MOT/CLS/timing/marker/rawPASS; deltaIDs+3,6667,pairedCI95[-2;+11], giữgateFalse; khôngđổimặcdịnh.

**R164 đang chạy — đã đo năm cách giản lược, chọn joint50 để phục hồi chất lượng.** Người dùng 21/11 giao tiếp cải thiện bộ lọc/dataset/Nano, nhấn mạnh thử giảm mô hình. Giao thức `logs/review164_latency_pruning/00_protocol.md`, bổ sung 01 trước điểm (Hard-Swish và chọn epoch monitor argmax), 02 sửa no-op minimum8, 03 làm rõ final cùng CUDApre/matched control. Profile anchor 41,66426 ms, Detect23 12,48624 ms; target depth2 là block2/4. Cả5 oracle v2, ONNX checker và 10 ORT/PyTorch mỗi model đạt. Params/ConvMAC anchor2590815/1894456320; head50 2346847/1589422080, body50 2263359/1743068160, joint50 2019391/1438033920, depth2 2574031/1792220160, hswish2590815/1894456320. Đây là ConvMAC, khác THOP cũ. Giữ v1 failed minimum8 và preflight backward v1 thiếu re-enable grad; v2 kiểm finite KD backward/checkpoint roundtrip đủ6 model đã PASS, không tính epoch/AP.

**GPU screen hoàn tất và host tái kiểm PASS** ở `screen_verification.json`: mean hai median original/head50/body50/joint50/depth2/hswish **41,64530/38,39430/40,95520/37,88980/38,95900/54,76080 ms**; giảm thời gian **7,8064/1,6571/9,0178/6,4504/−31,4933%**, repeat lệch tối đa0,487%. Chỉ joint50 đạt cổng >=8% đã khóa, không hạ cổng để train head50. Raw parity head50/hswish FAIL giữ nguyên; body50/joint50/depth2 PASS. Tất cả5 build thành công, đủ12 phép trtexec thuận/đảo, cùng boot51280481…, SD0. Native archive đã fetch/extract `screen_evidence/`, hostsession39169 thoát0. GPU-only guard <512MiB swap đạt, nhưng swap sau screen134,24MiB và sau thoát125MiB **chưa đạt guard pipeline <100**; không dùng screen thay FPS hay nới guard final.

**R164 recovery đãdừnglỗi CUDA; chỉcontrol10epoch/selectionđãxong, joint50mới7epochhoàn tất**, hostsession94810 chạy `recover.ps1`: verify screen→khóa training_inputs→control+joint50 mỗi10epoch/5200train/300monitor/rectTrue/batch4/KD R145→chọn epoch bằng exactargmax/NMS384x640 AP50-95 (tieAP50/earlier), rồi mới dev7190 và export. Control chọnzero-basedepoch9 trênmonitor300: AP50/AP50-95 45,32348/30,96609%; đây khôngphảiAPfull7190. Đủ10controlcheckpoint/sourcehash PASS. Joint50chưa chọn/dev; chưaFPSmới. Batchshapes đãghi đúng4×3×384×640. Profile bổ sung hswish tổng55,00021ms, tênlayerHardSigmoid25,94681ms so Sigmoidbaseline12,69307ms; layercountcùng348, chỉattribution/không khẳngđịnhcausal hoặcFPS, `hswish_profile_analysis.json`. Không gắn local model hook vào checkpoint; ghi batch shape bằng Trainer.preprocess_batch; cache nhãn chuyển dưới R164, không đọc `.npy` cạnh dữ liệu gốc. Nguồn train/monitor đã compile/import và preflight; `continue_runtime.ps1`, stage/capture/speed helpers mới chỉ chuẩn bị/compile, **chưa chạy**, phải kiểm guard sau build/capture trước timing. Final AP dùng actual TensorRT7190; final80 dùng cùng CUDApreR162 cho original/control/student, so baselineCUDA20,48 chứ không chỉ CPU18,82; quality trượt vẫn là trượt dù đo trade-off. Nếu swap vẫn>=100 khi đến final timing, cần root helper đã biết; tiếp tục PC/quality trước, chưa hỏi lại sudo. `capabilities.json` xác nhận SM5.3/TRT8.2.1.9/CUDA10.2, fastFP16 True/fastINT8 False; không có đường INT8 inference được NVIDIA hỗ trợ, không hứa FPS INT8. Không đổi default/launcher/khóa luận/metric. Dataset chưa có phép chạy mới; PF mới R165 chỉ đang dev, prerequisite Gaussian fullUA vẫn trượt; không tự mở scenario mới. Tối đa3 logical Python job, hiện pairedresumeR169PC + NanoR167finaltiming; R165/R166/R168 đãxong, không mởNanoGPUjob khác.

**Hoàn tất 21/11 — CUDA là phương án tốt nhất trong ba nhánh mới.** Cả R162 CUDA, R150 pruning và R160 height352 đã chạy/chấm đủ80 đoạn mỗi nhánh; tổng560 lượt/84.000 inference, 40BDD+40UA, 150JPEG/đoạn, warm25, ByteTrack KF, displayoff, không bỏ suy luận. Hostwrapper đã thoát0, `extra_chain_state.json` ALL_SPEED_TRIALS_COMPLETE; source/ảnh/config/hash/timestamp/schema MOT/TrackEval đều PASS kiểm tính toàn vẹn. Tổng hợp đã tái kiểm mọi hash ở `logs/review155_runtime_accuracy/extra_final_summary.json`; **chỉ CUDA** vừa nhanh hơn cả hai tập vừa giữ AP/raw parity và MOT/CLS chính xác80/80. CUDA meanFPS **18,81983→20,47623 (+8,80136%)**; BDD **19,60334→20,88571 (+6,54162%)**, UA **18,03632→20,06674 (+11,25744%)**, nhanh hơn80/80. Detection7190 ảnh vẫn AP50/AP50-95 **47,14673/31,98483%**; HOTA/IDSW/FN/FP/IDF1 giữ nguyên. Pruning chỉ **18,82208→18,93809FPS (+0,61634%)** và mất3,27865 điểm AP50, IDSW BDD1534→1644/UA245→336: không chọn. Height352 chỉ **18,81030→19,03562FPS (+1,19786%)**, BDD−1,44507%/UA+4,06912%, AP50−0,95665điểm và IDSW1534→1577/245→339: không chọn. PASS audit không biến qualityFAIL thànhPASS. Không còn job tự mở; Nano cùngboot51280481… SDerror0, swapcuối11,328MiB. Không đổi launcher/default/khóa luận hoặc bộ lọc. Báo cáo tổng hợp `logs/review155_runtime_accuracy/11_tong_hop_toi_uu_20261121.md`; ba thư mục R162/R150/R160 có full80_evidence/runs_absolute/full/score_verified.json. Các negative-pilot helper chỉ compile, không cần thực thi vì cả3pilot dương. Native clock+08, người dùng+07; tên experiment20261120 giữ theo ngày bắt đầu.

**R155 — full80 hoàn tất và đã xác minh tốc độ/chất lượng.** Nano boot `51280481-0986-4322-a37c-824de9e072af`, IP192.168.1.5. Cùng 80 đoạn (40 BDD + 40 UA), 150 JPEG/đoạn, warmup25, displayoff, KF và full inference: mean FPS original→320x576 **18,80397→22,35410 (+18,8797%)**, nhanh hơn **80/80**. Riêng BDD **19,60813→22,84748 (+16,5204%)**, UA **17,99980→21,86073 (+21,4498%)**. Aggregate FPS 18,76222→22,32938; paired-median gain19,4080%, stratified bootstrap95%[19,0008;19,4979]. 160 lượt/24.000 inference đều đủ, original MOT/CLS trùng byte R109 cả80; swap0→49,49MiB, lỗi SD0. Chấm lại timestamp, frame/config/source/GT/hash và TrackEval đạt ở `logs/review155_runtime_accuracy/full80_evidence/runs_absolute/full/score_verified.json` (PASS kiểm tính toàn vẹn, không phải PASS chất lượng). BDD HOTA **43,7188→42,7495%**, IDSW **1534→1518**, FN **28539→29304**; UA HOTA **66,5732→65,8941%**, IDSW **245→290**, FN **17590→18571**. Actual mAP50 **47,1467→45,9844%**. Vì giảm chất lượng, **không khuyến nghị/promote 320x576**, không đổi launcher/default/khóa luận. Archive17.323.656 byte SHA256 `86cf4727aa9b06731b73631c6c7036a6125c0aa8b92590f6cedef65d75fe2af1`. Giữ hai lỗi harness ban đầu và lỗi auditor giả định10 cột MOT; bản sửa chỉ verifier theo schema7 cột thật, không đổi đầu ra/metric. `score_full_v1.py/log`, `score_full_v2.log`, notes02–05.

**R162 — hoàn tất CUDA full80 và host audit.** Gộp resize8bit theo OpenCV3.2.0, letterbox114, BGR→RGB/HWC→CHW và LUT256float32 vào một kernel CUDA10.2/sm53; giữ engine384x640/weights/NMS/mask/KF. Oracle44tensor/20raw/40adapter đạt trùng bit; toàn bộ72chunk/7190 ảnh candidate arrays/AP trùng CPU. Full160 lượt/24.000 inference trên80 đoạn: meanFPS18,81983→20,47623 (+8,80136%), 80/80 nhanh hơn; aggregate18,77945→20,45778, mincandidate18,64787FPS. Medianpairedgain8,60347%, CI95[8,39662;9,00098] (5000 bootstrap, seed155, strata40+40). Median p90 theođoạn121,67594→111,39042ms. MOT/CLS trùng80/80; BDD HOTA43,71884%, IDSW1534,FN28539; UA HOTA66,57320%, IDSW245,FN17590, cả hai giữ nguyên. Micro CPUpre+upload7,19135→2,37751ms, GPUYOLO vẫn~41,9ms. Trong pipeline CUDApre nằm trong infer_ms, không gọi pre_ms0 là toàn bộ tiền xử lý bằng0. Source/image/config/timing/hash/TrackEval audit PASS; swapstart9,24609→9,13281MiB, SD0, cùngboot. Archive17.386.212byte SHA e37697f881f86950b3eeaeab5088943259af34078f05c728199f2bda57ea72af. `logs/review162_cuda_pre/{candidate_parity.json,speed_preflight.json,full80_evidence/runs_absolute/full/score_verified.json,10_ket_qua.md}`. Giữ lỗi stage symlink và failedcopy/log; không sửa kernel/capture để vượt kiểm. Gói thử cô lập, không đổi launcher/default/khóa luận.

**R163 — audit FP32 runtime geometry hoàn tất.** Tái dùng evaluatorR151 bất biến, đủ7190 ảnh cho control vàpruned, source/model/dataset hash trước/sau đạt. AP50 PCFP32 anchor/control/pruned **47,05447/46,13403/43,87884%**; actual NanoFP16 **47,14673/46,17151/43,86807%**. Pruned−matchedcontrol **−2,25519 điểm ởFP32** và **−2,30343 điểm ởNanoFP16**; Nano−PC của riêngpruned **−0,01077 điểm AP50**, nên suy giảm lớn đã hiện diện trên đườngFP32 cùnggeometry/argmax. Không quy toàn bộ giảm chất lượng choFP16, cũng không gọi đây là ablation precision thuần vì OpenCV/backend khác. StandardvalR145 còn khác geometry/postprocess nên không trộn AP48,03 với43,87 thành cùngphép đo. `logs/review163_pruning_precision_audit/{comparison.json,comparison.log,inputs.json,quality/*/summary.json}`, báo cáo10_ket_qua.md. Đây là kiểm chất lượng, không có FPS Nano mới. Không train thêm/pruning thêm trước khi đo lợi ích tốc độ của nhánh đã có.

**R161 — diagnosticIDSWdev đủ240outputs, khôngđổi tracker/metric.** TraceCLEAR từR96 khớpIDSW từngvideo12×(5Gaussian+5PF×3seed), frozenraw/MOT/GT/source đạt. KF449=299aftergap+150continuous; E154435=277+158; U154430=279+151; PF R142440,67=288,67+152; guidedR153437,67=284,33+153,33. Guidedgiảm4,33sauGT-unmatchedgap nhưng tăng1,33continuous soR142; auxiliaryR159438=284,33+153,67 (khônggiảmaftergap). Guidedwithin3frameaftergap17, gap9–30117, >3023. `logs/review161_pf_diagnosis/{summary.json,partial.json,inputs.json,run.log}`. Khoảng65%PFIDSWsaugapGTmatching, khôngđồngnghĩachekhuấtvật lý/Lostinternal/quan hệnhânquả; khôngchỉđịnhtuneQ từđây. R107 đãthửpostgapvelocityreset trênPF500 vàtrượt; khônglặp tựđộng khiýnghĩathiếu. KhôngmởfullPF/kịchbản.

**R155 GPU timing hoàn tất12 lượt đảo thứ tự.** Mean của2 median trtexec mỗi engine: original **41,6112ms**, anchorrebuilt **42,05495ms**, 352x608 **37,9076ms**, 320x576 **33,17795ms**, 352PSA32 **39,14365ms**, 320PSA32 **34,29075ms**. Đây là GPU compute, tách khỏi FPS full80 đã xác minh phía trên. PSA32 chậm hơn pure cùngshape và không cứu APgate. `logs/review155_runtime_accuracy/runs/engine/{rows.json,complete.json}` và12log/environment đã fetch/hash. Profile full80 original→320 median-of-clip-medians: CPUpre4,6112→5,5303ms, H2D1,7479→1,3785ms, GPUexec41,9524→33,4967ms, infer44,1368→35,2681ms; trackerupdate10,8763→10,2568ms. Decode chồng lấp xử lý nên không cộng các median thành latency.

**R155 — audit actual TensorRT AP đủ6×7190ảnh hoàn tất; chuẩn bị timing.** Original mAP50 **47,1467%**, mAP50-95 **31,9848%**, AP50 car/truck/bus/motorcycle/bicycle **84,4577/44,0422/51,4063/34,4174/21,4100%**. Anchorrebuilt **47,0326/31,9611%** đạt APgate. 352x608 **46,4037/30,6639%**;320x576 **45,9844/30,9973%**;352PSA32 **46,3962/30,6586%**;320PSA32 **45,8568/30,9297%** đều trượt APgate. Đây là actual engine+pre/post Nano trên dev nội bộ, không test mù/FPS; GTmoto226/bike407. `logs/review155_runtime_accuracy/{actual_AP_summary.json,timing_preflight.json}` và6cache quality/verification. Tar/hash/indices/NMS/AP/P-Rrematch đạt, giữ raw-score-parityFAIL của4shape nhỏ. `01_speed_tradeoff_scope.md` khóa trước AP/FPS cho phép đo ứng viên trượt để báo trade-off, không promote/nới cổng. GPU/pilot/full80 đã hoàn tất; xem kết quả full80 ở đầu mục0. Giữ các nguồn/config/log cũ; R150/R160/R162 là thí nghiệm riêng. Không đổi launcher/default/khóa luận.

**R160 — hoàn tất height352 full80; loại phương án.** Cùngweights, shape352x640; actual AP50/AP50-95 **46,19008/31,76984%**, AP50−0,95665điểm, bicycle−2,14273điểm sooriginal, qualityFAIL; rawFAIL score.020578891>.02 giữ nguyên. GPU41,64825→39,2912ms; đủ160lượt/80đoạn, meanFPS **18,81030→19,03562 (+1,19786%)**. BDD **19,58920→19,30612 (−1,44507%)**, UA **18,03141→18,76513 (+4,06912%)**, nhanh hơn41/80. BDD CPUpre3,75252→7,41993ms trong khi GPUexec41,97943→39,70926ms: giảm GPU chưa bù tăng CPUpre; nguyên nhân nhánh resize AREA/LINEAR là giả thuyết dựa hình học, chưa ablation riêng. UApre5,47981→6,09110ms, GPU41,94007→39,68179ms. BDD HOTA43,71884→43,66353%, IDSW1534→1577,FN28539→28411; UA HOTA66,57320→66,73017%, IDSW245→339,FN17590→17466 (HOTA tăng nhẹ không xóa việc IDSW tăng). Host source/hash/timestamp/TrackEval audit PASS, SD0, swap11,3125→11,328125MiB. Archive16.358.359byte SHA c25a283fac926a521e651769315f06fab958f65ad870b544f52b1a504e0315c2. `logs/review160_height352/full80_evidence/runs_absolute/full/score_verified.json` và10_ket_qua.md. Không promote qualityFAIL/không đổi launcher/default/khóa luận.

**R159 — auxiliary guidedRBPF đủ36dev, loại.** First-stage predictive Gaussian marginalize incomingomega bằngJ, chọnancestor khiESS<.5N, second-stage giữ-log(m_selected)+logprior/q+exactGaussianLL, cachebody/mu/eta đúngancestor; không đổiassociation/Q/R/N100/init/shape. SeedIDSW**442/437/435**,mean**438**, HOTA**61,2272%**,FN35080,33; so guidedparent437,67 vàE435/U430 thìIDSW/two-seeds trượt, HOTA/FNđạt. Numerical480offparity/700Gaussiancomponents/7000constant-weightdraws/80CTPSD/8ancestry/80zero-var/hookrestore, smoke và36marker/raw/hash đạt. `logs/review159_guided_auxiliary/dev/{summary.json,verification.json}`; tham khảoPitt/Shephard1999, khácAPFfullstate500R117. Không tuneESS/lookahead/N đểcứu, khôngfull/kịchbản.

**Phân tích bấtđịnhPFdev sauR159:** ghépcặp12video/5000bootstrap, exactmean3seed. GuidedR153−parentR142=-3ID, CI95[-8,33;+2]; guided−E144/U144=-1,33 vớiCI[-14,67;+13,33]/[-17;+11,67]; guided−E154/U154=+2,67/+7,67 vớiCI[-8,67;+14,33]/[-10,33;+24,67]. R156/R157/R159−guided=+4/+2,67/+.33, CIđềuqua0. `logs/review159_guided_auxiliary/{dev_comparison.json,compare_dev.log}`. Điểm guided nhỏ, chưa chứng minh thắngchắcchắn; CIhậu nghiệm/nhiềuphépso chưahiệu chỉnh vàkhôngkhửselectionbias, khôngmởlại cổngtrượt. GiữguidedR153 lànhánhdevPF tốt nhất trongcácnhánh mới xétmeanIDSW, khôngpromotedefault/toànmiền.

**R158 — Cartesian CT+chord init đủ24dev, loại sođốichứng mạnh.** Khởi tạo haiquan sát/covvelocity giốngNativePF trênR149: E/U IDSW**443/445**, tốt hơnR149461/476 và<KF449, nhưng kémGaussianpolar cũ439/439 lẫnR154435/430. HOTA**61,1803/61,2592%**, FN35090/35087; EKF còntrượt HOTAguardsoR154 (.1040điểmgiảm>.1). Không chọncặp này đểPF440,67 dễthắng. Numerical320offsteps/162chordoracles/320PSD/no-motion/ownership/hookrestore, smoke4control/4repeat/6PFprefix và24marker/raw/sourcehash đạt. `logs/review158_cartesian_chord/dev/{summary.json,verification.json}`. Chỉđổiinit, giữCTCartesian/Q/R/sigma/Joseph; khôngfull/kịchbản, khôngđổisrc/default/Nano/khóaluận.

**R157 — init covariance transport đủ36dev, loại.** Parent guidedR153, khôngtrộnR156; congruence D=diag(I,Ltarget*inverse(Lold)), Pnew=D*Pold*D' giữmarginals vàwhitenedcross. SeedIDSW**442/438/441**, mean**440,33**, kémparent437,67 và>E435/U430; khôngseed<=430. HOTA**61,2197%**, FN35070,33, HOTA/FNguardđạt. Numerical480offsteps/6init/500SPDoracle/63enabledinit/no-motion/hookrestore, smoke và36marker/raw/MOT/CLS/timing/sourcehash đạt. `logs/review157_pf_init_transport/dev/{summary.json,verification.json}`; 01_terminology.md đínhchính từ tiếngViệt tronggiao thức là “tương hợp”, khôngsửa fileđãkhóa/code/sốhọc. Khôngfull/kịchbản, khôngtune đểcứu.

**R156 — đủ36PFdev, loại.** Giữ Lost shape covariance trên guidedR153: seedIDSW**442/441/442**, mean**441,67**, kém parent437,67 và>E435/U430; khôngseed<=430. HOTA**61,2042%**, FN35080, HOTA/FNguard đạt nhưngIDSW/twoseeds trượt. Numerical600offparity/300shapeoracle/40Lostbody/160trackedparity, preflight/smoke và36marker/raw/MOT/CLS/timing/sourcehash đạt. `logs/review156_guided_shape/dev/{summary.json,verification.json}`. Không tune/dò damping đểcứu, không full/kịchbản, không sửa src/default/launcher/khóa luận.

**R154 — audit full480Gaussian hoàn tất, không đạt prerequisite cảhai tập.** GiữCTRVcực/init/omega/R, chỉ speed-aware Qtheta floor; numerical và dev24 E435/U430 đạt trước khi khóa full. BDD200 IDSW KF/E/U **5319/4947/5075**; HOTA% **35,4212/35,4356/35,4876**, E/U<KF nhưng IDSW kém đối chứng commonR cũ4854/5028, strictcomponent trượt. UA40 **2319/2374/2360**, HOTA% **59,0257/59,1688/59,2188**; E/U>KF nên prerequisite trượt dù HOTA/FN guard đạt. PF R142 dùng lại, không PFfull mới. `logs/review154_velocity_noise/full_audit_summary.json`, cảhai `bdd/test40` summary/verification; full summary xác minh480marker và raw/MOT/CLS/timing. Không mở kịch bản hoặc chọn Gaussian yếu để PF dễ thắng. Dev tốt hơn không chuyển thành ưu thế full.

**R153 — guided incoming-turn proposal đạt dev36, chỉ giữ component.** Đúng temporal modelR142: body[t] dùngomega[t-1], không đảo noise trướctransition. Dùngz đang ghép để lậpGaussianproposal cho incomingangularinnovation, cộnglogp/q, giữeta[t] độc lập, ancestry vàno-predict/init/Lost xửlý. N100/Q/R/init/ESS/associationkhôngđổi. Numerical519offbitparity,6init,100linearGaussianconstant-weightcases,CTquadrature20.000draw,80ancestry,zeroVariance100steps/PSD/RNG đạt; smokeparent/onrepeat/off6prefix đạt. DevseedIDSW**439/435/439**, mean**437,67**, <=E/U439 cả3seed; HOTA**61,2211%**, FN35071 soR142440,67/61,2376%/35084,33. Giảm3IDmean nhỏ, khôngclaim chắcchắn. `logs/review153_guided_turn/dev/{summary.json,verification.json}`. Chưa chạyBDD/UAfull vì cổngGaussianfull R148 còntrượt; R154 đangkiểm prerequisite, khôngmởscenario từdev.

**R152 — PF500+confidenceR hoàn tất36dev, loại.** IDSW445/437/440, mean440,67 bằng PF100R142440,67; EKF/UKF439/439. HOTA61,2308%, FN35071,33; HOTA/FN đạt nhưng mean/two-seeds trượt (chỉ1seed<=439). Không tăngN nữa hoặc mởfull/kịchbản; chưa có timing hợp lệ để nói chi phí/tốc độ. Numerical600directparity/RNG/PSD,160CVsteps/countguard500, smokecontrols/parent100/repeat500 và36marker/source/raw/MOT/CLS/timing đạt. `logs/review152_pf500_confidence/{dev/summary.json,dev/verification.json,10_ket_qua.md}`. R153 kế tiếp đã qua dev, xem trạng thái riêng phía trên; kết quả R152 này vẫn trượt.

**R151 — chấm đủ7190dev và build hoàn tất, trial gốc không đạt.** CùnganchorR101, CPUF32 qua Nano-path: AP50 original47,0545%,352x60846,3634%,320x57645,9760%; giảm0,6911/1,0785điểm, trượt cổng. AP50moto34,0008/34,9468/37,1399%, bike21,3553/21,2605/19,0927%; GTmoto226/bike407 nên không khái quát từ chênh nhỏ. BaONNX parity10ảnh đạt; original/anchor_rebuilt TRTrawparity đạt, pure352/320 trượt maxscore .020082/.023969>.02. C2PSA FP32 treatment cho cảhai vẫn trượt .024721/.025352, không gọi đã sửa. Giữ mọi log/cổng; R155 audit APengine thật và trade-off riêng, không hồi tố R151 thànhPASS. `logs/review151_resolution_tradeoff/{quality.json,quality_summary.log,nano_models_psa32.json}`. Không FPS mới, không đổi launcher/default.

**R150 — hoàn tất pruning full80; loại phương án.** Anchor/control/channel25_KD actual AP50 **47,14673/46,17151/43,86807%**, AP50-95 **31,98483/31,12498/29,28391%**. Pruned mất3,27865 điểm AP50 sooriginal và2,30343 so matchedcontrol, giữ rawFAIL .035185>.02. Params−10,62%, MAC−10,73% nhưng GPU41,54245→41,34075ms chỉ−0,4855%; đủ240 lượt/80×3model: meanFPS original/control/pruned **18,822079/18,652057/18,938087**, pruned+0,61634% vsoriginal (+1,5335% vscontrol), nhanh hơn74/80. BDD HOTA43,71884→43,30411%, IDSW1534→1644, FN28539→28457; UA HOTA66,57320→64,77225%, IDSW245→336, FN17590→19654. Matchedcontrol giữ trong evidence, không quy thayđổi do kiến trúc đơn thuần. Host audit source/config/frames/hash/timestamp/TrackEval PASS, qualityFAIL vẫn giữ; swap9,13281→11,3125MiB, SD0. Archive24.129.917byte SHA a099241467a28c15427f6f4da95b4bed3a9f5c9175e9e892c964ddb5cffb6d92. `logs/review150_nano_pruning/full80_evidence/runs_absolute/full/score_verified.json` và10_ket_qua.md. Capture pruned từng ngắt5000/resume2190 có journal/hash; đủ7190, không so partial. Không chọn/không train thêm nhánh này chỉ để cứu +0,6%FPS; không đổi launcher/default/khóa luận.

**R149 — CT Cartesian EKF/UKF hoàn tất24dev và loại.** KF449/EKFcart461/UKFcart476/PF440,67; Gaussian cực commonR R144 E/U439/439. HOTA Ecart61,0282%/Ucart61,2211%, FN35222/35197; trượtIDSW cảhai, EKF trượtHOTA, cảhai trượtFN. Không chọn Gaussian kém hơn để PF dễ thắng, không mở full/kịchbản, không tune noise/init đểcứu. Numerical96Jacobian/ODE/rotation,160CVsteps,100sigma-update,320PSDsteps, smoke4control/4repeat/6PFprefix và24marker/source/raw/MOT/CLS/timing đạt. `logs/review149_cartesian_gaussian/{dev/summary.json,dev/verification.json}`. Đây là thay tọa độ/khởi tạo/Q, không tương đương mô hình cực; không sửa src/default/Nano/khóa luận.

**R148 — audit đầy đủ commonR hoàn tất720reference mới, cổng cảhai tập không đạt.** CùngR(c)=max(1-c,.05)R0 cho cảbốn, PF R142 giữ nguyên/dùnglại720lượt đãkhóa. BDD200/39973ảnh: IDSW KF/EKF/UKF/PF **5319/4854/5028/4788,33**, HOTA% **35,4212/35,5073/35,5337/35,4798**, đủcổng. UA40/56340ảnh: IDSW **2319/2359/2372/2362,67**, HOTA% **59,0257/59,2155/59,1480/59,2696**; EKF/UKF>KF vàPF>EKF, trượt. TấtcảGaussianHOTA/FNguardsofixed-Rđạt. Bootstrapghépcặp5000tập, exactmeanCOMBINEDPFseeds: UA CIIDSW EKF−KF[-19,96],UKF−KF[-8,113],khôngkhẳngđịnhchắcchắnmôhìnhnàotốthơn; BDDchênh EKF−KF=-465,UKF−KF=-291. Smokeoff/on/PFprefix/raw/MOT/CLS/timing/720marker/3529frozenrecords đạt; `logs/review148_common_references/{final_summary.json,final_verification.json}`. Khôngmởkịchbản, khôngpromoteR144/R146/R147đãtrượtdev, khônggọiđâylàcải tiếnPFmới. KhôngcònworkerR148; R145trainingGPUvẫntiếp tục.

**R147 — jointmoment correction+commonR hoàn tất36PFdev, loại.** PF449/438/443, mean**443,33**, HOTA**61,2316%**, FN35.080,67; KF449/EKF439/UKF439R144. MeanPF>439 vàchỉ1seed<=439, trượtmean/two-seeds, HOTA/FNđạt. Không mởBDD/test40/kịchbản hoặcđổiepsilon/strengthđểcứu. Numerical200jointmixtures/720scoredcontrolsteps/PSD/RNG/enabledcorrection, smokecontrols/parentparity/repeat và36MOT/CLS/timing/marker/raw/sourcehashđạt. `logs/review147_pf_shared_moments/{dev/summary.json,dev/verification.json,10_ket_qua.md}`. Không dùngorderingR146, không sửa src/default/Nano/khóaluận.

**R146 — orderedstratified+commonR hoàn tất36UAdev, loại.** PF449/441/440, mean**443,33**, HOTA**61,2324%**, FN35.078,33. KF449/EKF439/UKF439R144: E/UKF<KF đạt nhưngPFmean/two-seeds trượt, HOTA/FN đạt. KhôngmởBDD/test40/kịchbản vàkhôngđổiESS/N/seedđểcứu. Numerical600scored-control/RNG/PSD và100CDFcases đạt; smokeoldPFparent6/repeat6, controllersroutingcontrols6/repeat6 đạt; fullraw/MOT/CLS/timing/marker/hash36 đạt. `logs/review146_pf_shared_ordered/{dev/summary.json,dev/verification.json,10_ket_qua.md}`. NguồnvàcodeR140giữ nguyên; đây là phép thử tổhợpvớiRconfidence, khôngquy kếtquảtrước sanglượt mới.

**R145 — control/pruned đều xong5/5epoch, chấm7190dev và xuất3ONNX đạt.** Control resume từ checkpoint3/5 theo02_resume_protocol, giữ5epoch tổng; không cam kết RNG như chạy liên tục, pruned không ngắt. Khóa5200train (5000mixed5+200RideSafe),300monitor,11014hash và train/dev không trùng byte. Anchor/control/channel25_KD AP50 **49,5584/49,2549/48,0300%**, AP50-95 **34,0917/33,3561/32,0067%**; pruning mất1,5285điểm AP50 soanchor và1,2250 so matchedcontrol, trượt cổng .5điểm, không khuyến nghị. Đây là standardval640, không cùng protocol APengineR155. Params2590815->2315551 (-10,62%) chưa phải FPS. BaONNXstatic384x640/opset12 checker+10ảnh/model CPUORT/PyTorchparity đạt, `logs/review145_yolo_recovery/{quality.json,export_verification.json}`. R150 đã build/capture/chấm actual AP đủ3engine; xem mục R150 phía trên. R145anchorgraph trùngbyte R151 nhưng vẫn build lại cùngworkspace512, không bỏ đo trade-off vì chất lượng trượt. Không sửa src/default/launcher/khóa luận.

**R144 — common confidence-R, hoàn tấtUAdev36lượt, chưa đạtPF.** KF/EKF/UKF/PF IDSW **449/439/439/440,67**; HOTA% **60,7845/61,2760/61,1966/61,2376**; HOTA/FN đều qua cổng so fixedR, nhưng meanPF>439 nên không mởBDD/test40/kịch bản trong nhánh này. PFseed448/436/438,2seed đạt nhưng không đủ vì meantrượt. Numerical360direct-score/PSD/contextrestore, smoke6controls/6repeat/6PFprefixparity đạt; MOT/CLS/timing/hash/raw36lượt xác minh. Backend thực dùngUltralytics đã cài (đườngSTrack tronginputs), có cùng côngthứcNSA nhưvendor. Không nới floor/gain/đổiKF đểcứu. `logs/review144_shared_confidence/{dev/summary.json,dev/verification.json,10_ket_qua.md}`. Nhánh tiếp R146 chỉđổi resampling PF; không thayfixedR mốccũ.

**R143 — kiểm chuyển PF R142 trên các tập kịch bản, hoàn tất765 lượt; kết quả mô tả theo từng tập.** Giữ nguyên bộ lọc/detection/GT/cấu hình R142, mới chạy KF/EKF/UKF fixed-R, PF100 R132 fixed-R và PF100 R142 confidence-R×3 seed.

RoundaboutHD — vòng xuyến: 4 chuỗi/12000 ảnh; IDSW KF/EKF/UKF/R132/R142 **471.00/477.00/494.00/477.00/477.00**; HOTA% **42.8318/42.4829/42.4429/42.7536/44.3868**. Cổng mô tả {"IDSW_below_all": false, "HOTA_vs_parent": true, "FN_vs_parent": true}.

SVMOT — che khuất: 9 chuỗi/8250 ảnh; IDSW KF/EKF/UKF/R132/R142 **435.00/407.00/418.00/436.33/418.33**; HOTA% **48.6604/49.6491/49.2095/50.8886/50.2462**. Cổng mô tả {"IDSW_below_all": false, "HOTA_vs_parent": false, "FN_vs_parent": true}.

HELMET — xe máy, c25: 36 chuỗi/3600 ảnh; IDSW KF/EKF/UKF/R132/R142 **87.00/55.00/54.00/48.33/51.00**; HOTA% **16.8045/17.2513/17.2719/17.2142/18.5916**. Cổng mô tả {"IDSW_below_all": true, "HOTA_vs_parent": true, "FN_vs_parent": true}.

HELMET — xe máy, c10: 36 chuỗi/3600 ảnh; IDSW KF/EKF/UKF/R132/R142 **78.00/54.00/54.00/43.00/44.67**; HOTA% **18.5478/19.0936/19.6446/19.0445/20.4700**. Cổng mô tả {"IDSW_below_all": true, "HOTA_vs_parent": true, "FN_vs_parent": true}.

Riêng nhóm rẽ ảnh>=25° (6601hộp), IDSW KF/EKF/UKF/R142 **28.00/32.00/29.00/23.33**; nhóm>=35° (4225hộp) **18.00/18.00/18.00/15.67**. Nhóm SVMOTvisibility<=0,5 (12051hộp), IDSW KF/EKF/UKF/R142 **125.00/115.00/125.00/126.33**. PF mới tốt hơn về điểm IDSW ở nhóm rẽ, chưa tốt hơn KF/EKF/UKF ở nhóm che; không khái quát từ toàn tập. HELMET CIIDSW soEKF/UKF còn chứa hoặc chạm0; PF R132 có IDSW thấp hơn R142 ở cả hai ngưỡng dù HOTA thấp hơn. SVMOT R142 giảm0,642điểm HOTA soR132, không chọn mặc định toàn miền.

Whole-sequence CLEAR khớp117 lượt trước phân tầng R99turn25/35/straight8 và SVMOTvisibility<=0,5; HELMETbootstrap5000 theo12nhóm cảnh mỗi điều kiện. Raw/COMBINED/historicaldeterministic/source/cache/GT/MOT/CLS/timing/765marker đạt. `logs/review143_pf_scenarios/{final_summary.json,final_verification.json,10_ket_qua.md,scenario_overview.png,scenario_groups.png}`. Không suy CI độc lập từ4camera cùng nút giao hay9clip/2scene; không gọi test mù hoặc purePFsuperiority. HELMET làtest36, Roundabout chỉprefix3000; không gọi full toàn bộ dataset. Không đổi src/default/Nano/khóa luận; không còn job chất lượng chạy.

**R142 — PF + nhiễu đo theo confidence, hoàn tất; đạt mục tiêu IDSW quan sát trên cả BDD và UA test40.** Giữ nền R132 Native Cartesian RBPF100, R theo chiều cao dự đoán, init IID, mean, systematic resampling và Lost shape clamp; không dùng covariance floor R141. Adapter riêng truyền confidence của detection đang ghép vào công thức có sẵn `R(c)=max(1-c,0,05)R0`; không đổi detector/cache/Q/association/threshold/lifetime. Numerical360 bước khớp direct-score parent, kiểm routing current/explicit/None/off, PSD/RNG/context restoration và smoke đối chứng/lặp đạt. UA dev12×3: IDSW **440,67** (448/436/438), HOTA **61,2376%**.

R142 BDD200×3 đủ600 lượt, 39.973 khung/seed: **IDSW4.788,33** (4785/4785/4795), HOTA **35,4798%**, FN258.585,33, FP15.603,67; thấp hơn KF6345/EKF5548/UKF5655 ở cả ba seed. UA test40×3 đủ120 lượt, 56.340 khung/seed: **IDSW2.362,67** (2347/2364/2377), HOTA **59,2696%**, FN196.218,67, FP70.503,33; thấp hơn KF2498/EKF2525/UKF2543 ở cả ba seed. Cả hai tập đạt HOTA/FN guardrails, MOT/CLS/timing/marker/hash/raw tái hợp. Bootstrap ghép cặp theo video5000 mẫu/tập: CI IDSW so EKF trên BDD **[-939,68;-585]**, so KF trên UA **[-212,67;-58,33]**. Phân tích cùng PF R132 fixed-R: BDD IDSW−676,33, HOTA+1,774điểm, CIIDSW[-825;-528,32]; UA IDSW−157,33, HOTA+0,810điểm, CIIDSW[-219,34;-95,67]. CIs là mô tả có điều kiện sau nhiều vòng quan sát, không loại trừ selection bias.

Đo tuần tự42 lượt trên hai prefixBDD200, ba lần đảo thứ tự, PC tracker.update: KF/EKF/UKF/PF cổ500/Native100/R132/R142 **1,055/1,281/3,173/5,634/4,900/4,893/4,776 ms/update**. Lặp output/prefix parity đạt; PF vẫn tốn CPU hơn KF/EKF/UKF, đây chưa phải FPS Nano. `logs/review142_pf_confidence/{final_summary.json,final_verification.json}`: integrityPASS, **goal_observed_met:true**,756 lượt chất lượng mới (36dev+600BDD+120UA),42timing,1366 hash đầu vào/nguồn; dùng lại240 controlsUA và720 lượt parent ablation (120UA+600BDD). Đã cập nhật trạng thái chuẩn trước báo cáo `10_ket_qua.md`; không còn job chạy. Dừng chuỗi tìm ứng viên theo mục tiêu đã đạt, không tiếp tục dò tham số. **Phạm vi kết luận: PF+confidence-conditioned R so với baseline fixed-R**, chưa chứng minh PF thuần tốt hơn nếu mọi bộ lọc dùng cùng R(c); confidence phân lớp chưa được hiệu chỉnh thành độ bất định định vị. Không đổi src/default/Nano/khóa luận. Bước tiếp nếu được giao: ablation cùng R(c) cho KF/EKF/UKF và đo PF mới trên Nano; không tự coi đó là kết quả đã có.

**R141 — measurement-derived motion covariancefloor, hoàn tất36UAdev, loại.** Giữpointvelocitymean, floorCvv từ(Rprev+Rcur)/T² dưới giảđịnhđộclập, khôngexactBayes/confidencehiệntạichưađượchookforward. Numerical1000PSDfloor/483disabledsteps/mean/RNG/noisecontext/ownership/gapT^-2đạt;controls6/repeat6đạt. UAdevmeanIDSW**473,33**(481/473/466),HOTA**59,3281%**,FN37.440,FP12.119. IDSWvàtwo-seedstrượt (<472), HOTA/FNđạt. KhôngmởBDD/test40/timing, khôngnớicổng/tunefloorweightđểcứu. `logs/review141_pf_motion_uncertainty/{dev_summary.json,verification.json,00_protocol.md}`. Khôngđổi src/default/Nano/khóaluận. AuditcallerxácnhậnconfidenceargthựctếNone trongNPbackend;R142thửroutingnoiseadaptationriêng, khôngtrộnfloor.

**R140 — ordered stratified resampling, hoàn tất756lượt, chưađạt cảhai tập.** GiữQ/R/N100/ESS0,5, chỉstratifiedCDFsauomegaorder, dùngNuniformdrawthay1systematic. Numerical500selection/CDF/permutation/expectedoffspring/zero/uniform/PSD và600disabledsteps/43resamplekhớpparent/RNGđạt. UAdev36: IDSW462,67(461/472/455),HOTA59,3042%,FN37.438,67. BDD600: **IDSW5446,67**(5417/5442/5481),<KF6345/EKF5548/UKF5655 cả3seed;HOTA33,7137%,FN261.066,33;CIsoEKF[-195,34;-7,32]. UAtest40đủ120: **IDSW2518,67**(2516/2516/2524), vẫn>KF2498 cả3seed;HOTA58,4392%,FN199.589;CIsoKF[-43,67;+80,67]. HOTA/FNđạt nhưngIDSW/two-seedstrượt. Raw/MOT/CLS/timing/marker/hash/boot5000mỗitậpđạt; final_verification integrityPASS, `goal_observed_met:false`; khôngtiming. Dùnglại240controls+120parentablationR132. Khôngtuneordering/ESS/N/seed, khôngđổi src/default/Nano/khóaluận. `logs/review140_pf_ordered_resample/{final_summary.json,final_verification.json}`. Dữliệuđãquansátnhiềuvòng, khônggọitestmù hayưu thếphổquát.

**R139 — damped Lostextent, hoàn tất36lượtUAdev, loại.** tau=buffercũ30frameUA/6frameBDD, cùngQ/R/association. Numerical500Fsemigroup/480notLostparity/360tau0parity/72Lostoracle/PSD/RNG/ownershipđạt; controls6/repeat6đạt. UAdevmeanIDSW**466**(460/466/472),HOTA**58,7818%**,FN**37.561**,FP12.113,67. IDSW/2seedđạt nhưngHOTA giảm0,238điểmsoR123 (ngưỡng0,1),FN+0,352% (ngưỡng0,2), trượtcảhai guardrails. KhôngmởBDD/test40/timing, khôngdòhorizon/phiđểcứu. `logs/review139_pf_extent_trend/{dev_summary.json,verification.json,00_protocol.md}`. Khôngđổi src/default/Nano/khóaluận.

**R138 — giữ covarianceh-vh khiLost, hoàn tất756lượt, chưađạt cảhai tập.** NềnR132, chỉresetmeanvh=0/giữGaussianuncertainty, khôngđổiQ/R/association. Numerical600notLostbitparity/36Lostoracle2x2maxerror0/PSD/RNG/ownershipđạt. UAdev36: IDSW465(468/469/458),HOTA59,2875%,FN37.460,33. BDD600: **IDSW5424,33**(5400/5421/5452),<KF6345/EKF5548/UKF5655 cả3seed;HOTA33,7810%,FN261.100,33;CIsoEKF[-221,34;-25,65] nhưnglàtậpđãquansátnhiềuvòng. UAtest40 đủ120: **IDSW2539,33**(2544/2534/2540), vẫn>KF2498 cả3seed;HOTA58,4578%,FN199.645;CIsoKF[-17;+99]. HOTA/FNđạt nhưngIDSW/two-seedstrượt. Raw/MOT/CLS/timing/hash/marker/boot5000mỗitậpđạt; final_verification integrityPASS, `goal_observed_met:false`; khôngtiming. Dùnglại240controls+120parentablationR132. Khôngtune covariance/seed/N, khôngđổi src/default/khóaluận. `logs/review138_pf_lost_shape/{final_summary.json,final_verification.json}`.

**R137 — deferred angular innovation, hoàn tất756lượt mới, chưađạt cảhai tập.** Native100Rprediction, cùngQ vàsố/thứtựRNGdraw, chỉgắneta_slot sauselectionđể khôngnhânbảnnoiseđộclập. Numerical360noresample/120noobservation/480q0steps với31resample khớpparent/RNG; forced6ancestor đúngcôngthức/PSD/ownership. UAdev36: IDSW467,33(458/476/468),HOTA59,2899%,FN37.447,33. BDD600: **IDSW5456**(5426/5456/5486),<KF6345/EKF5548/UKF5655 cả3seed;HOTA33,7078%,FN261.069,67. CIIDSWsoEKF[-181,68;-1,33] nhưng dữliệuđãchọnnhiềuvòng, khônggọixácnhậntestmù. UA test40 đủ120lượt: **IDSW2525**(2512/2539/2524), vẫn>KF2498 cả3seed;HOTA58,4336%,FN199.579,67;CIsoKF[-32;+82,67]. HOTA/FNđạt nhưng IDSW/two-seedstrượt. Raw/MOT/CLS/timing/marker/hash/boot5000mỗitậpđạt; final_verification integrityPASS, `goal_observed_met:false`; khôngtiming. Dùnglại240controls+120parentablationR132. KhôngdòQ/seed/Nđểcứu, khôngđổi src/default/khóaluận. `logs/review137_pf_deferred_noise/{final_summary.json,final_verification.json}`.

**R136 — median marginal Gaussian-mixture, hoàn tất36 lượt UAdev, dừng tại cổng.** Numerical500CDF độc lập/360 bước kernel/RNGparity/PSD/ownership/symmetry/outlier/Dirac đạt;maxCDFerror1,01e−12. Controls6/repeat6 đạt. UAdevmeanIDSW**470**(476/478/456),HOTA**59,2875%**,FN37.436,67,FP12.144,33. Mean<472 nhưng chỉ1seed<472, trượttwo-seeds; không mởBDD/test40/timing. MedianCDF root làm tăng chi phí tính toán, chưa có benchmarktiming, không khẳng định tốc độ. Không dòtolerance/axis/blending/seed/N đểcứu; không đổi src/default/khóa luận. `logs/review136_pf_mixture_median/{dev_summary.json,verification.json,00_protocol.md}`.

**R135 — joint moment correction sau resampling, hoàn tất756 lượt, chưa đạt mục tiêu.** Numerical500 mixture/720 bước parent parity/counters/RNG/PSD đạt; maxmean/cov error2,84e−14/8,81e−13. UAdev36: IDSW465(460/473/462),HOTA59,3055%,FN37.453,67. BDD600: **IDSW5463**(5440/5460/5489),<KF6345/EKF5548/UKF5655 cả3seed;HOTA33,6938%,FN261.056,33. UA test40 đủ120 lượt: **IDSW2521,67**(2527/2521/2517), vẫn>KF2498 cả3seed;HOTA58,4233%,FN199.580,67. HOTA/FN đạt nhưng IDSW/two-seeds trượt. BDD CIIDSW soEKF[-179,01;+8,33], UA soKF[-33,67;+78,67], đều chứa0. Raw/MOT/CLS/timing/marker/hash/boot5000 mỗi tập đạt; final_verification integrityPASS, `goal_observed_met:false`; không timing. Dùng lại240controls+120parent ablationR132. Smoke đầu UA khởi động trước preflight xong nên dừng ngay tại assert inputs, chưa xử lý video; chạy lại saupreflight đạt, logs giữ cảhai. Không đổi code/tham số để cứu, không đổi src/default/khóa luận. `logs/review135_pf_moment_resample/{final_summary.json,final_verification.json}`.

**R134 — quantile init omega, hoàn tất756 lượt mới, chưa đạt mục tiêu.** Numerical240 bước parity/80 bước zero variance/PSD/RNG/ownership và các controls/repeat đạt. UAdev36: IDSW467,33(467/473/462),HOTA59,3157%,FN37.440,33. BDD600: **IDSW5464,67**(5471/5454/5469), thấp hơn KF6345/EKF5548/UKF5655 cả3seed;HOTA33,7476%,FN261.073,33. UA test40 đủ120 lượt: **IDSW2515,33**(2515/2515/2516), vẫn>KF2498, không seed nào dưới KF;HOTA58,4290%,FN199.562,33. HOTA/FN đạt nhưng IDSW/2seed trượt. Dùng lại240 lượt controls và120 parent ablation R132 sau khóa hash. Raw/MOT/CLS/timing/sourcehash/boot5000 mỗi tập đạt; final_verification integrityPASS, `goal_observed_met:false`. Không chạy timing vì trượt chất lượng, không dò grid/sigma/N để cứu. `logs/review134_pf_quantile_init/{final_summary.json,final_verification.json}`. Giữ nguyên mặc định/src/khóa luận.

**R133 — likelihood hỗn hợp ba Gaussian, hoàn tất756 lượt mới và chưa đạt mục tiêu.** Giữ covariance R đo tổng,100 hạt omega, Gauss–Laguerre3 node/Gamma shape2 cố định; posterior moment collapse, không phải Student-t chính xác. Numerical100 mixture cases/240 one-component steps/PSD/controls/repeat đạt. UAdev36: IDSW461,33(462/468/454),HOTA59,3166%,FN37.440,33. BDD600: IDSW**5527**(5532/5524/5525), thấp hơn KF6345/EKF5548/UKF5655 ở cả3 seed; HOTA33,6401%,FN261.131,67. UA test40 đủ120 lượt ứng viên: **IDSW2518,33**(2516/2523/2516), vẫn cao hơn KF2498; HOTA58,4597%,FN199.573,33, guardrails đạt nhưng cổng IDSW/2 seed trượt. Dùng lại240 lượt controls R132 và120 lượt parent ablation đã khóa hash. Codex update ngắt ở13/40 mỗi seed; resume_check xác minh39 marker/1346 hash, sau đó hoàn tất bằng nguồn khóa. Raw/MOT/CLS/timing/sourcehash và bootstrap5000 mỗi tập đạt; final_verification PASS về integrity, `goal_observed_met:false`. Không chạy timing vì chưa đạt chất lượng hai tập; không dò scales/nu/seed để cứu. `logs/review133_pf_scale_mixture/{final_summary.json,final_verification.json}`.

**R132 — R theoheightprediction, hoàn tất996lượtchấtlượng và36timingPC; chưađạtIDSWcảhaitập.** NativeCartesianPF100Cfloat64, chỉđổiheightdùngtínhRchoGaussianupdate/likelihood, giữhệsố/confidence/Q/init/resampling/association. Numerical240equal-height/PSD/restore/controlsđạt. UAdev36lượtmeanIDSW466,67(460/483/457),HOTA59,2893%,FN37.453,33,quadgatesđạt. BDD200×3seed600lượt: **IDSW5464,67**(5447/5453/5494),<KF6345/EKF5548/UKF5655,cả3seed<5548;HOTA33,7055%,FN261.064,67. CIIDSWsoEKF[-177,33;+9,67]cònchứa0,HOTAthấphơnEKF0,121điểm. SoPF100R125IDSW−144,67/HOTA+0,208điểm/FN−314. Raw/MOT/marker/classes/timing/hash/CIđạt.

R132UA test40 **40video/56.340khung**, chạyđủ9nhóm/360lượt mới,cùngCOCOc25: KF/EKF/UKF/native100/R132IDSW **2498/2525/2543/2509,33/2520**. R132seed2504/2514/2542, khôngseednào<KF2498, trượtIDSW/two-seeds; HOTA58,4594%soNative58,4341%,FN199.583so199.600,67, guardrailsđạt. CIIDSWsoKF[-41,34;+83,67]chứa0, khônggọiR132thắngcảhaitập. Đotuầntự36lượt2prefixBDDPC: KF/EKF/UKF/PFcổ500/Native100/R132 **0,960/1,126/2,745/5,050/4,123/4,090ms/update**;R132gầnNative, chưaFPSNano. 996unitquality/36timing/boot5000mỗitập/hashesđạt, `logs/review132_pf_predicted_noise/{final_summary.json,final_verification.json,10_ket_qua.md}`; `goal_observed_met:false`. Khôngđổimặcđịnh/YAML/src/thesis, hiệu lựcRpredictionchỉresearchcontextcókèmnguồn.

**R131 — OUturnprior hiệu chỉnh từBDDtrain139, hoàn tất và loại.** Fit145.744cặpomega,139video devcủatrainnộibộ, khôngtrùngBDDval. lambda9,30738s^-1,S10,9742rad²/s²; tại25FPSrho0,68915/initstd0,132509/qstd0,096019rad/frame; tại5FPSrho0,15544/initstd0,66255/qstd0,65449. Khôngchỉnhfit sauđiểm. OUstationary/semigroup/cadence/nativekernel240step/6control/6repeat đạt. UAdev36/36: IDSW479 (472/480/485),HOTA58,9136%,FN37.455,33; trượtIDSW/two-seeds/HOTA, khôngmởBDD. `logs/review131_pf_turn_prior/{turn_prior.json,calibration_inputs.json,dev_summary.json,verification.json,10_ket_qua.md}`. Gaussianfit từbboxcóthểbịjitter ảnhhưởng, chưachứngminhnhânquả; khôngđổiwindows/bounds/variance đểcứu.

**R130 — speed noise theo hướng + heading moment, hoàn tất và không chọn.** Numerical1000tensorcases/trace/PSD/limit,240zero-noisesteps,200nonzeroPSDsteps,6nativecontrol/6repeat đạt. UAdev36/36: IDSW472,33 (475/466/476),HOTA58,9676%,FN37.424,33. TrượtIDSW<472 và2/3seed thắng; HOTA/FNguardrails đạt. Không mởBDD, không dò tensor/scale; `logs/review130_pf_polar_noise/{dev_summary.json,verification.json,10_ket_qua.md}`. Không đổi mặcđịnh/khóa luận.

**R129 — initializer mềm theo covariance, hoàn tất và loại.** Numerical750fusioncase,PSD/no-motion-no-draw/hook đạt, nativeUA6control/6repeat khớp. UAdev12×3seed đủ36: IDSW465,67 (462/464/471),HOTA58,9988%,FN37.529,33. IDSW vẫn<min472 nhưngFN tăng100 (0,267%) soR123, trượt cổng0,2% đã khóa. Không mởBDD, không dògain/nới cổng, không trộn R128 để cứu. Mã/điểm/kiểm ở `logs/review129_pf_soft_init/`; báo cáo `10_ket_qua.md`.

**R128 — nhiễu hướng bằng moment matching, hoàn tất636lượt và chưa đạt mục tiêuBDD.** Dùng `_std_theta` hiện có, không tăng hạt/dòsigma. Numerical/quadrature/PSD/zero-noise control đạt, sai kháccov<=8,09e−13; UAdev36/36lượt: IDSW461 (462/466/455), HOTA59,0201%,FN37.438, đạtcổng. BDD200×3seed đủ600lượt: **IDSW5590** (5609/5560/5601), HOTA33,4991%,FN261.393,33; giảm19,33IDSW soR126 nhưng vẫn cao hơnEKF5548; cả3seed chưa<5548, không chọn. 1123hashBDD,36UAmarker/600BDDmarker, raw tái hợp vàbootstrap5000mẫu đạt. Không mởUAtest40 vìBDDtrượt; kế hoạch vàcode xác nhận đã chuẩn bị ở `logs/review128_pf_heading/{01_test40_confirmation.md,test40_job.py}` nhưngchưachạy. Báo cáo `logs/review128_pf_heading/10_ket_qua.md`. Không dò sigma để cứu; tiếp cơ chếkhác R129.

**R126 — lõi C/float64 cho PF100, hoàn tất kiểm chất lượng và tốc độ PC (19/11).** Giữ mô hình Cartesian100hạt R123, Q/R/init/RNG/resampling/association; chuyển predict/Gaussian update/Joseph covariance/within-moment sang C float64, MSVC có sẵn, /fp:strict. Numerical720bước/3seed đạt, PSD/RNG/copy ownership giữ nguyên, maxstateerror2,84e−13. Kiểm riêng C update1.500ma trận SPD và fallback không sửa input đạt. Smoke12/12 và **600/600 lượt BDD200×3seed** khớp toàn bộ raw metric từng video vớiR125; frame/ID giữ nguyên, sai khác tọa độ tối đa5,01e−11px. 1.671hash đầu vào đúng. HOTA33,4975%, IDSW5.609,33 giữ nguyên.

R126 đo tuần tự18lượt BDD (hai prefix200, bỏ16khung, ba lần đảo thứ tự): PF cổ điển/PF100 gốc/native100 **5,341/6,510/4,225ms/update**. Native giảm thời gian **35,09%** soPF100 (cả6cặp,27,45–41,83%) và **20,90%** soPF cổ điển (cả6cặp,8,94–27,70%). Kết hợp chất lượng đủ200video, native tốt hơn PF cổ điển ở IDSW−10,84%, HOTA+0,547điểm; tốc độ chỉ đo mẫuPC, **chưa là FPSNano/toàn pipeline**. Đo bổ sung24lượt cùng cache cho KF/EKF/UKF/nativePF100: **1,028/1,200/2,804/4,521ms/update**; PF vẫn tốnCPU hơn cả ba. Trên BDD, HOTA PF vẫn thấp hơnKF/EKF/UKF0,179/0,329/0,271điểm và FN cao hơn, không gọi thắng toàn diện. A1 cache/solve2 NumPy chỉ giảm8,7% và còn chậm hơnPF cổ điển, không chọn. Các lỗi harness (confidenceCSV làm tròn, importmeasure trùng tên, tên thư mục tham chiếu KF không có seed) đã sửa có lưu snapshot/log, không đổi filter hay tolerance. Mã chỉ trong `logs/review126_pf_efficiency/`, backend nghiên cứu cần context cài đặt; chưa đổi `src/`, launcher hay khóa luận. Hậu kiểm `native_validation/{verification.json,final_verification.json,closeout_verification.json,summary.json,all_filters_timing.json}`. Báo cáo `logs/review126_pf_efficiency/10_ket_qua.md`.

**R127 — phân tầng nhiễu omega, hoàn tất và loại.** Khóa một ứng viên100uniform phân tầng, inverseGaussianCDF và hoán vị; không đổi Q/R/init/resampling/sốhạt. Không gọiSQMC đầy đủ. Native smoke UA6/6 khớpR123, lặp seed6/6 đạt. UAdev12×3seed, **36/36 lượt**: IID/ứng viên có IDSW **458/465,67**, HOTA **59,0198/58,9950%**, FN **37.429,33/37.439,33**. Trượt cả ba cổng không suy giảm, **dừng và không mởBDD, không dò N/strata để cứu kết quả**. 70hash nguồn,36MOT/marker và tái hợp raw ba seed đạt. `logs/review127_pf_stratified/{dev_summary.json,verification.json,10_ket_qua.md}`. Không còn job chạy. Bước tiếp hợp lý: kiểm/đo lõi native trênARM/Nano và phân tích FN/HOTA của PF; chưa có bằng chứng để thayKF/EKF/UKF bằngPF mặc định.

**R125 — PF mới trên BDD, hoàn tất 1.800 lượt mới và 24 lượt timing PC (19/11).** Đủ **200 video / 39.973 ảnh**, cache YOLOv8n COCO R61/R74 c25, năm lớp phương tiện, FPS5, buffer6khung, elapsed steps, mot_crowd. Giữ nguyên Cartesian PF100 R123, không dò tham số. PF100 / PF cổ điển500 K1/K2 motion2 có HOTA **33,4975/32,9508%**, IDF1 **38,2511/37,0857%**, IDSW **5.609,33/6.291,33 (−10,84%)**, FN **261.378,67/263.146,67 (−1.768)**. Bootstrap ghép cặp200video,5.000mẫu, chênh IDSW **[−855,03,−517,32]**, HOTA **[+0,408,+0,681] điểm**. PF cổ điển dùng lại 600 lượt R74 đã đối chiếu; PF100 chạy mới đủ ba seed 5.621/5.588/5.619.

R125 chạy mới KF/EKF/UKF cùng đầu vào: IDSW **6.345/5.548/5.655**, HOTA **33,6769/33,8265/33,7688%**. PF100 thấp hơn HOTA cả ba (0,179–0,329 điểm), FN cao hơn 397–683 hộp; IDSW ít hơn KF theo CI, chênh so EKF/UKF còn chứa0. Cartesian500 cùng mô hình có **5.599,33 IDSW /33,5027% HOTA /261.367,67 FN**; giảm500→100 làm IDSW+10 (CI[−21,67,+41,33]), HOTA−0,0052điểm (CI[−0,0379,+0,0354]), FN+11. Chênh quan sát nhỏ, không chứng minh tương đương chính xác hay100hạt tối ưu mọi cảnh.

R125 đo tuần tự PC hai video đã khóa bằng workload50/90,200ảnh/video,bỏ16khung,ba lần đảo thứ tự: KF/PF cổ điển500/Cartesian500/PF100 **1,046/6,239/15,505/6,911 ms/update**. PF100 giảm **55,43%** so Cartesian500, cả6cặp giảm47,37–64,32%; so PF cổ điển trung bình **tăng10,78%**,5/6cặp chậm hơn và dao động lớn. Đây là đổi chi phí lấy chất lượng so PF cổ điển; **không phải FPSNano/toàn hệ thống**. 1.800 marker/MOT/timing/CLS,9nhómraw,1.053hash,400đối chiếu scalar KF/EKF với R61,24timing và lặp MOT/prefix đạt. Không còn job chạy, không sửa mặc định hay khóa luận. [Báo cáo R125](logs/review125_pf_bdd/10_ket_qua.md) · `summary.json`, `bootstrap.json`, `final_verification.json`, `serial_timing.json`.

**R123 — Cartesian PF100, hoàn tất 99 lượt chất lượng và 12 lượt đo tuần tự PC.** Giữ mô hình R118, chỉ giảm N500→100 sau khi marginalize còn một chiều omega. UA dev12 ba seed IDSW **464,67→458**, HOTA **58,995→59,020%**, FN **37.450,67→37.429,33**; bootstrap chênh IDSW theo video **[−12,−1]**, vẫn là dev đã phân tích nhiều vòng. Chuyển nguyên cấu hình SVMOT test9, ba seed: IDSW **435,33→435**, HOTA **50,772→50,902%**, FN **14.615→14.613,33**; nhóm che 12.051 hộp IDSW **132,33→132**, FN **6.478→6.478**. Cổng chất lượng hai tập đạt. Chạy lại KF/EKF/UKF cùng UA dev12 đủ 36 lượt, toàn metric khớp R70: IDSW **489/472/474**, HOTA **59,000/59,252/58,786%**; PF100 ít IDSW theo điểm ước lượng, CI chênh với từng bộ lọc đều còn chứa 0.

R123 đo tuần tự PC hai prefix512 × ba lần đảo thứ tự, bỏ32 khung đầu: N500/N100 **30,81/12,26 ms/update (−60,19%)**, cả sáu cặp giảm, khoảng **49,66–68,45%**. Đây là chi phí tracker ở hai prefix, **chưa đo Nano hay FPS toàn hệ thống**; không cộng phần trăm với R118 ở lượt đo khác. Hậu kiểm **99/99** lượt chất lượng, **9** nhóm raw tái hợp, **105** hash đóng băng, **12** lượt timing và 27 CLEAR nhóm SVMOT đạt. Ứng viên ở `logs/review123_pf100/`, chưa đổi `src/`, mặc định hay khóa luận. `final_verification.json`, `dev_summary.json`, `transfer_summary.json`, `serial_timing.json`.

**R124 — PF100 trên toàn KITTI Car, hoàn tất 60 lượt.** Đủ 20 chuỗi R119 / 7.863 ảnh / 24.070 hộp, ba seed, cùng cache/GT/cấu hình chấm, mỗi seed dùng workspace evaluator riêng tránh race seqmap. KF/EKF/UKF/PF100 có IDSW **280/260/260/247,67**, HOTA **49,764/49,680/49,717/49,487%**, FN **9.416/9.422/9.416/9.477,33**. PF100 giảm IDSW so KF 11,55% nhưng HOTA giảm 0,277 điểm và FN tăng 0,65%, chưa tốt hơn toàn diện. Bootstrap chênh IDSW PF100−KF **[−66,67,−4]**, so EKF **[−25,−0,33]**, so UKF **[−25,0]**. Nhóm rẽ 399 hộp IDSW **8/8/8/7,67**, FN **96/106/104/106,33**; nhóm che nặng 4.259 hộp IDSW **43/42/42/43,33**, FN **3.384/3.415/3.411/3.413,67**. Không nhận cải thiện chắc chắn hai kịch bản từ các chênh nhỏ này. 60/60 cache/GT đối chiếu nền, 60 CLEAR trace và 80 hash nguồn/MOT đạt. Chưa có PF500 cùng mô hình trên KITTI nên không quy khác biệt ở đây cho riêng số hạt. [Báo cáo R124](logs/review124_pf_kitti/10_ket_qua.md).

**Bước tiếp sau lượt đêm 19/11:** PF100 là ứng viên giảm chi phí đáng giữ, chưa là mặc định toàn cảnh. Cần đo lại trên Nano trước khi nói FPS và cân nhắc backend riêng sau kiểm hồi quy; ưu tiên chẩn đoán FN/HOTA khi camera gắn xe, không dò tiếp tham số trên KITTI. Dữ liệu mới chưa xác nhận ưu thế EKF/UKF riêng quỹ đạo cong; giữ RoundaboutHD, UA che và HELMET làm các mốc đã kiểm, tiếp tục V2X/TUMTraf khi nguồn tải có lại. Không chạy nền còn sót, không sửa khóa luận trong lượt này.

**R119 — KITTI Tracking toàn bộ ô tô, hoàn tất 60 lượt.** Theo yêu cầu tự nghiên cứu khi người dùng nghỉ (19/11), mở rộng ba chuỗi R115 thành **20 chuỗi có Car hợp lệ / 7.863 ảnh / 24.070 hộp**, 0017 không có Car nên không chấm người. Cùng COCO c25, native KITTI evaluator, FPS 10, KF/EKF/UKF có IDSW **280/260/260** (hai bản phi tuyến giảm 7,14%), HOTA **49,764/49,680/49,717%**, FN **9.416/9.422/9.416**. Phần mới 17 chuỗi IDSW 169/151/152; bootstrap chênh IDSW toàn tập EKF/UKF **[−52,+5]/[−47,+3]**, đều chứa 0. Nhóm rẽ ≥25° trên ảnh 399 hộp có IDSW **8/8/8**, FN 96/106/104; nhóm native che nặng 4.259 hộp IDSW 43/42/42, FN 3384/3415/3411. Không xác nhận thắng riêng quỹ đạo cong hay che khuất từ kết quả này. Đối chứng R115 9/9 lượt khớp toàn bản ghi/MOT; 60 MOT, 20 cache, **35** nguồn và 7.863 hash ảnh đạt. `logs/review119_kitti_full/{summary.json,verification.json,image_verification.json}`.

**R120 — robust R riêng kích thước PF, hoàn tất và loại.** Trên Cartesian RBPF R118, tăng R aspect/height theo innovation với ν=4 cố định, giữ R tâm/likelihood vị trí/association. Numerical/smoke đạt (nền 6/6, lặp 6/6); dev12 × ba seed đủ 36 lượt, so R118 IDSW **464,67→619**, HOTA **58,995→55,610%**, FN **37.450,67→44.626,67**. Trượt cả cổng, dừng đúng giao thức, không dò ν hoặc chạy transfer. 59 hash nguồn, 36 MOT và tái hợp raw ba seed đạt. [Báo cáo R120](logs/review120_pf_robust_shape/10_ket_qua.md).

**R121/R122 — GMOT ô tô, hoàn tất pilot và cổng detector.** Tải chọn lọc bốn chuỗi car từ release v0.1 tác giả, CRC nguồn và SHA ảnh đạt; loại car-3 trường đua sau duyệt ảnh trước điểm. Ba cảnh đường bộ có 1.027 ảnh / 22.301 hộp / 145 ID. FPS từng chuỗi chưa có trong ZIP, bài báo chỉ 24–30 nên chỉ thăm dò hai đầu mút, 18 lượt KF/EKF/UKF, chưa coi là benchmark cadence đã xác minh. COCO640 c25 toàn ba cảnh IDSW **22/20/21**, HOTA **16,533/16,490/16,421%** ở cả hai giả định. Cảnh car-0 vòng xuyến đêm bỏ sót toàn nhóm rẽ, car-2 tuyết không có track. Thử thêm full1280 và sliced640 cố định, độ phủ candidate c25 nhóm rẽ car-0 chỉ **3,1–3,5%** so 640 **0,07–0,13%**, trượt cổng 25% đã ghi, không mở tracking từ hai cache mới; 7 hash nguồn và 3 cache đạt. Không suy diễn 0 IDSW là tốt. DSEC-MOT liên kết nhãn hiện duyệt được thư mục nhưng cả train/test trống; MEVDT 403; TRAF đã lấy được listing tác giả, hai ZIP chỉ chứa ảnh, chưa có GT bbox/ID/cadence đủ chấm. [KITTI và nguồn](logs/review119_kitti_full/10_ket_qua.md) · [GMOT](logs/review121_gmot_vehicle/10_ket_qua.md) · [Detector](logs/review122_gmot_detection/10_ket_qua.md).

**R118 — RBPF vận tốc Cartesian, hoàn tất 126 lượt.** Thay biểu diễn tốc độ/hướng bằng Gaussian vị trí/vx/vy, chỉ lấy mẫu omega. Hai khối kích thước Gaussian dùng chung; Q vận tốc Cartesian đẳng hướng theo hệ số hiện hành, có đổi mô hình nhiễu. Số học CV/Kalman, quỹ đạo tròn, liên tục omega=0, PSD và smoke nền R112 6/6 đạt. Dev12 UA, 500 hạt, ba seed, 72 lượt cho hai cách khởi tạo. RBPF cũ / Cartesian giữ init hai quan sát / Cartesian tự học có IDSW **469 / 464,67 / 514**, HOTA **58,963 / 58,995 / 58,981%**. Chọn bản giữ init theo tiêu chí dev đã ghi. Bootstrap chênh IDSW dev **[−30,+18,33]**, mức giảm nhỏ.

R118 chuyển SVMOT test9, chạy lại cả RBPF nền và ứng viên, **54 lượt**, MOT nền khớp kết quả R112 ở 27/27 lượt. IDSW **440→435,33 (−1,06%)**, HOTA **51,014→50,772% (−0,242 điểm)**, FN **14.633→14.615**, FP **22.383,33→22.259,33**. Nhóm 12.051 hộp visibility≤0,5 IDSW **136→132,33**, FN **6.479→6.478**. Kiểm ghép CLEAR toàn 54 lượt khớp IDSW/FN. Đo tuần tự PC hai prefix512 × ba lần, update RBPF cũ/Cartesian **42,20/33,90 ms (−19,67%)**; từng cặp đều nhanh hơn nhưng thời gian tuyệt đối dao động lớn qua các lần. Chưa đo Nano. Hậu kiểm **126/126** lượt và **56** hash đầu vào/nguồn đạt. Mã ứng viên riêng `handoff/logs/review118_pf_shape/cartesian.py`, chưa đổi backend mặc định hay khóa luận. [Báo cáo](logs/review118_pf_shape/10_ket_qua.md) · [Dev](logs/review118_pf_shape/dev_summary.json) · [SVMOT](logs/review118_pf_shape/transfer_summary.json) · [Nhóm che](logs/review118_pf_shape/occlusion_summary.json) · [Hậu kiểm](logs/review118_pf_shape/verification.json).

**R117 — ba biến thể Auxiliary PF, hoàn tất 180 lượt.** Người dùng yêu cầu tham khảo kỹ thuật PF và thực nghiệm cải thiện. Thử sinh mẫu hậu nghiệm độc lập từ tổ tiên chọn theo likelihood quan sát, giữ 500 hạt và ba seed. UA train dev12, 16.919 khung/seed, **108 lượt** cho APF thuần, APF+K2 và APF+K2 chỉ khi ESS/N<0,5. So PF500 K1K2 motion2 R72 IDSW **697**, ba ứng viên lần lượt **769,33 / 675,67 / 677,67**, HOTA **56,31 / 57,28 / 57,37%** so nền **57,01%**. Hai bản K2 giảm IDSW **3,06% / 2,77%**, FN giảm **1,22% / 0,86%**. Bootstrap theo video chênh IDSW **[−42,33,−3,0] / [−43,67,+3,33]**. Mục tiêu 10% không đạt. Sau dev, mở kiểm chuyển miền thăm dò có ghi quyết định, giữ tham số cho đủ FastTrack12, **72 lượt**, 19.480 khung/seed. PF nền IDSW **1.277**, APF+K2 **1.282,67**, bản ESS **1.285,33**; HOTA **31,11→31,54 /31,47%**. APF+K2 giữ đúng ID **8/8/10** trong 82 sự kiện che so PF nền **9/11/12**, số thiếu ghép đầu/cuối cũng khác. Chưa có cải thiện IDSW nhất quán để thay PF hiện hành.

R117 đo tuần tự PC, hai prefix 512 khung, bỏ 32 khung nóng, đảo thứ tự ba lần. Update PF nền / APF+K2 / ESS **10,53/12,99/11,97 ms/frame**; các ứng viên tốn thêm CPU, chưa đo Nano. Trên cùng dev, RBPF R112 tham chiếu **469 IDSW, HOTA 58,96%** vẫn tốt hơn ba ứng viên. Kiểm mixture posterior theo nghiệm Gaussian, lặp seed, khôi phục hook, smoke nền 6/6 khớp MOT tham chiếu, hậu kiểm **180/180** lượt và **54** hash nguồn/đầu vào đạt. Mã thử cô lập ở `handoff/logs/review117_apf/`, không sửa `src/`, tracker mặc định hay khóa luận. [Báo cáo R117](logs/review117_apf/10_ket_qua.md) · [Dev](logs/review117_apf/dev_summary.json) · [Dev ESS](logs/review117_apf/dev_summary_ess.json) · [FastTrack](logs/review117_apf/transfer_summary.json) · [FastTrack ESS](logs/review117_apf/transfer_ess_summary.json) · [Hậu kiểm](logs/review117_apf/verification.json).

**R116 — FastTrack đủ 12 video, hoàn tất.** Tải thêm 9 ZIP (10,996 GB) lên SSD D, dùng lại 3 ZIP ở F, đủ 12 SHA nguồn (14,756 GB) ở revision `7e16fad8`. Đủ **19.480 ảnh / 545.151 hộp lớp ô tô / 907 ID theo video**, 54 lượt mới và 18 lượt R115 đã đối chiếu, cộng thêm 12 lượt kiểm nhạy FPS, cùng YOLOv8n COCO c25 và KF/EKF/UKF/PF500 K1K2 motion2 (PF ba seed). Toàn 12 video IDSW **1.099/1.075/1.103/1.277**, HOTA **31,22/31,88/32,03/31,11%**; chín video mới IDSW **781/778/801/904,33**. Bootstrap ghép theo video, CI chênh IDSW EKF/UKF so KF **[−81,+26]/[−47,+48]**. Trong 2.327 hộp rẽ ≥25° IDSW **8/8/15/12,67**. Hai chuỗi đêm có `seqinfo` 15 FPS và MP4 tái dựng 30 FPS, dùng 15 làm chính; chạy lại 30 không đổi nhận định tổng.

**R116 — giữ ID sau che khuất.** Duyệt 97 ứng viên ở chín video mới giữ 67 lượt xe buýt/xe tải che ô tô; gộp R115 thành **82 lượt, 21 ID xe gây che theo video**. Trên **27 lượt tất cả cấu hình đều ghép được ở hai đầu**, KF/EKF/UKF giữ đúng ID **4/9/11**, PF theo seed **9/11/12**. Riêng 67 lượt mới, trên 19 lượt ghép đủ, **4/5/6**, PF **4/6/6**. Thêm **48 lượt che bởi vật cố định** trên một góc `task_day_left_turn` (19 trụ cầu đi bộ, 29 cột dải phân cách). Chỉ **1/48** có ghép đủ cả hai đầu, cả bốn đều đổi ID; độ phủ ứng viên Car IoU ≥0,5 ở hai mốc **3/48** ở c25, **21/48** khi chỉ kiểm cache c05. Chưa đủ ghép để phân biệt bộ lọc qua vật cố định. Không sửa cấu hình tracking hay khóa luận. [Báo cáo R116](logs/review116_fasttrack_full/10_ket_qua.md) · [Điểm](logs/review116_fasttrack_full/scores/scores.json) · [Xe che xe](logs/review116_fasttrack_full/scores/occlusion.json) · [Vật cản cố định](logs/review116_fasttrack_full/scores/fixed_occlusion.json).

**R115 — thử FastTrack và KITTI Tracking, hoàn tất.** FastTrack nguồn tác giả revision `7e16fad8`, ba ZIP khớp SHA/Nhật ký tải, 5.375 ảnh 30 FPS; KITTI dùng lại ảnh trái cục bộ, ba chuỗi train chọn bằng số hộp Car che nặng, 1.657 ảnh 10 FPS. Chấm ô tô cùng cache YOLOv8n COCO c25 theo từng chuỗi, **36/36 lượt** KF/EKF/UKF/PF500 K1K2 motion2 (PF ba seed), đối chiếu hash 6 cache, 6 GT, 36 MOT và các tổng CLEAR. FastTrack toàn ba chuỗi IDSW **318/297/302/372,7**, HOTA **30,35/32,29/32,17/31,48%**. Nhóm rẽ trên ảnh ≥25° có 366 hộp, IDSW **0/0/0/0,33**. Ba chuỗi FastTrack có visibility=1 ở mọi nhãn nên duyệt ảnh để xác nhận 15 lượt ô tô bị 7 xe buýt/xe tải che ở hai góc quay. KF/EKF/UKF/PF giữ đúng ID **0/4/5/5,33 trên 15**, cả bốn ghép được đúng hai đầu ở cùng 8/15 lượt. Bảy lượt còn thiếu ghép ở một đầu; mẫu kiểm chứng nhỏ, không phải thống kê che khuất toàn bộ FastTrack. Độ phủ ứng viên detection toàn bộ ô tô chỉ **29,51%** ở IoU≥0,5. KITTI nhóm 2.205 hộp che nặng IDSW **20/16/17/19,33**, FN **1.752/1.767/1.764/1.762,33**. KF/EKF/UKF đều giữ đúng ID **15/71** sự kiện KITTI. Chưa xác nhận lợi thế qua vật cản cố định hay đủ mẫu xe máy FastTrack. TSBOW yêu cầu quyền tải và công bố cho detection; FE-DETRAC không mở được nguồn. Không sửa tracker hay khóa luận. [Báo cáo R115](logs/review115_new_datasets/10_ket_qua.md), [điểm](logs/review115_new_datasets/scores/scores.json), [kiểm từng lượt che](logs/review115_new_datasets/scores/fasttrack_verified_occlusion.json).

**R114 — chốt kết quả theo kịch bản và đo bốn bộ lọc trên Jetson Nano (18/11).** R109 TensorRT FP16 384×640 YOLO11n 50/50 giữ cố định, JPEG tmpfs, tắt hiển thị, 150 ảnh/đoạn, bỏ 25 khung làm nóng. Đo tám đoạn mỗi BDD/UA chọn cách đều từ 40 đoạn khóa, cùng ảnh cho KF/EKF/UKF/RBPF500 seed 0, smoke 8/8 riêng, đo chính **64/64 lượt, 9.600 khung**. Cache detection khớp SHA R110 ở 64/64, KF MOT/lớp trùng byte R109 16/16. Kiểm lại FPS/p90 từ timestamp và MOT so PC R110/R113 đạt 64/64 ở ngưỡng tọa độ 0,001 px, khung/ID/điểm khớp. **FPS trung bình KF/EKF/UKF/RBPF = 18,80/18,83/18,01/11,24**, ≥15 FPS **16/16, 16/16, 14/16, 6/16**. P90 trung vị theo đoạn **121,1/121,9/143,1/833,3 ms**. BDD FPS **19,60/19,61/19,62/13,80**, UA **18,01/18,04/16,39/8,68**. MVI_40855 GPU ~42 ms ở mọi lượt, update tracker trung vị **25,43/34,84/110,33/470,25 ms**, RBPF **2,10 FPS** và p90 **3,03 s**. Cùng boot, lỗi SD 0, swap **5,99→14,69 MiB**. Thử trong gói riêng, chưa thay launcher. Đường Nano `~/nano_run/r114_scenario_nano_20261118/`; hồ sơ `handoff/logs/review114_scenario_nano/measure/verified_summary.json` và `measure_evidence.tar.gz` đã đối chiếu SHA giữa Nano và PC.

**R114 — khóa luận và V2X.** Đã cập nhật nguồn `khoa_luan_prism/` và bản PDF A4 43 trang, thêm UA-DETRAC che khuất, đối chiếu BDD/METEOR, chẩn đoán cutoff SVMOT, chất lượng RBPF R113 và tốc độ R114. Ba mốc kịch bản là RoundaboutHD đổi hướng, UA-DETRAC che khuất, HELMET xe máy. Bootstrap ghép cặp 10.000 mẫu theo 40 đoạn cho chênh IDSW nhóm che UA **[−8, 2] EKF, [−15, 3] UKF**, cả hai chứa 0. Biên dịch XeLaTeX + BibTeX, kiểm bản in trang dữ liệu, tốc độ và tóm tắt, phông nhúng và zip 22 tệp. Bản giao `outputs/khoa_luan_Nguyen_Huu_Thai_Duong_R114_20261118.pdf`, `outputs/khoa_luan_Nguyen_Huu_Thai_Duong_LaTeX_R114_20261118.zip`. Hai gói mẫu V2X-Seq-SPD, metadata train/val và ảnh camera cố định tiếp tục trả quota; portal tác giả không kết nối được. Chưa có điểm V2X. Giao thức camera cố định và công cụ kiểm nhãn đã có, thử fixture theo schema công bố, chưa kiểm trên gói thật. Báo cáo `handoff/logs/review114_scenario_nano/10_ket_qua.md`.

**R113 — khảo sát xe rẽ và che khuất trong giao thông hỗn hợp (18/11), hoàn tất.** Theo yêu cầu người dùng, giữ HELMET làm mốc đã chốt. **UA-DETRAC test40** hậu kiểm 120 phép đối chiếu clip×bộ lọc. Riêng 7.486 hộp visibility ≤0,5, KF/EKF/UKF có IDSW **53/50/48**, FN **4.941/4.940/4.939**. Nhóm có che trong một giây gần nhất IDSW **86/83/77**, FN **5.793/5.792/5.791**. Toàn 40 đoạn IDSW **245/250/248**, nên kết quả thuận chỉ áp dụng nhóm che. Nhóm vừa rẽ vừa che chỉ có 34 hộp. **BDD40** hậu kiểm 120 phép đối chiếu. Nhóm rẽ ≥25° trên ảnh có 2.205 hộp xe, IDSW **64/61/59**, FN **1.201/1.206/1.207**. Riêng 1.896 hộp ô tô rẽ có IDSW **58/56/54**. Có 147 hộp xe máy toàn đoạn, không có đoạn xe máy đạt tiêu chí rẽ. COCO-video cục bộ thiếu cờ che gốc. `handoff/logs/review113_vehicle_scenarios/{ua_occlusion,bdd_turns}.json`.

**SVMOT:** audit c25/c10/c05 đủ 30 chuỗi, 60 hash GT/cache khớp. Độ phủ ứng viên che khuất test9 **49,03/67,42/73,60%**, train21 **46,40/69,38/77,68%**. Đối chiếu 54 clip×ngưỡng×bộ lọc test9 vào kết quả R64. Hạ c25→c10 cho KF làm IDSW toàn tập **435→311**, FN **14.665→12.924**, FP **21.964→29.836**. Riêng nhóm đang che c10, IDSW KF/EKF/UKF **105/107/106**, FN **5.685/5.635/5.597**. C05 chỉ audit độ phủ. `svmot_thresholds.json`.

**METEOR:** audit 9 gói nhãn (8 mới), 8 đạt điều kiện nhịp nhãn, một chỉ có 150 XML với ánh xạ khung chưa xác định. Cả 89.017 hộp ô tô/xe hai bánh từ tám gói hợp lệ có cờ occluded=0. Chọn theo nhãn trước điểm, tải và kiểm CRC/SHA hai video 30 fps có cả hai lớp, 3.600 frame. YOLOv8n COCO c25 + KF/EKF/UKF chấm 12 lượt clip×lớp×bộ lọc. Ô tô rẽ 425 hộp, IDSW **0/0/0**, FN **294/294/294**. Xe máy/scooter rẽ 670 hộp, IDSW **2/1/1**, FN **658/658/658**, độ phủ detection chỉ 18/670. Có tín hiệu trên xe máy nhưng ít quan sát, chưa đủ xác nhận ưu thế trên cả hai lớp. `meteor_selection.json`, `meteor_downloads.json`, `meteor_scores/scores.json`. V2X-Seq-SPD là ứng viên tiếp theo có nhãn 2D, ID và occluded_state, cả gói mẫu cũ, mẫu chính thức thay thế, gói ảnh cố định và metadata train/val đều trả quota. TUMTraf tiếp tục chờ truy cập, FE-DETRAC chưa xác minh lớp xe máy/GT tải, D2-City trang tác giả trả 404. `v2x_access.json`, `source_probe.log`.

**RBPF detector Nano:** phát lại PC đủ 80 đoạn × ba seed, 240 lượt, 36.000 khung. So PF500 K1/K2 motion2 cùng cache, BDD40 IDSW **1.569,67→1.450,33 (−7,60%)**, HOTA **42,81→43,60%**. UA40 IDSW **276,33→246,33 (−10,86%)**, HOTA **65,94→66,69%**. KF IDSW tương ứng **1.534/245**. Đo chất lượng từ cache, chưa đo tốc độ RBPF Nano. `nano80/scores.json`. Kiểm cộng gộp theo clip và hash đạt, xem `verification.log`. Báo cáo `handoff/logs/review113_vehicle_scenarios/10_ket_qua.md`.

**R112 — chẩn đoán SVMOT, phát triển PF và tìm dữ liệu giao thông mới (18/11), hoàn tất lượt được giao.** Người dùng yêu cầu tìm nguyên nhân SVMOT, sửa PF và tìm dữ liệu đúng cảnh phi tuyến, che khuất, xe máy từ camera đường/gắn xe. Đã đối chiếu phát lại ghép CLEAR với IDSW/FN từng chuỗi cho KF/EKF/UKF trên đủ 30 SVMOT, 90 phép kiểm. Train21 toàn video IDSW **1.189/1.162/1.162**, riêng khung visibility ≤0,5 **149/152/156**. Nhóm hậu kiểm có che trong một giây trước đó **474/454/455**, chỉ chẩn đoán phụ thuộc thời điểm, không thay benchmark. Cùng detector COCO c25, độ phủ có detection IoU≥0,5 ở GT khả kiến thấp **49,03% test9, 46,40% train21**; khoảng **93,88% FN** của KF trong nhóm này không có detection đạt IoU 0,5. Đây là cận trên candidate không ghép một-một, không phải AP. Chưa tìm được lỗi lập trình đơn lẻ giải thích toàn bộ kết quả EKF/UKF. `handoff/logs/review112_svmot_pf/03_svmot_diagnosis.md`, `svmot_diagnosis.json`.

R112 tạo RBPF lấy mẫu hướng/tốc độ quay, bảy tọa độ Gaussian điều kiện riêng từng hạt, Joseph covariance, ESS resampling; Q xấp xỉ tại chiều cao trung bình, bỏ K1/K2. UA train dev12 cùng cache/ghép/500 hạt, ba seed, **36/36 lượt**. PF500 K1/K2 motion2 → RBPF HOTA **57,0057→58,9635%**, IDF1 **68,3610→71,7956%**, IDSW **697→469 (−32,71%)**, FN **39.395,67→37.431,33 (−4,99%)**. IDSW seed **480/467/460**, 11/12 clip giảm, một tăng. Cổng dev đạt, dữ liệu đã xem nhiều vòng nên chỉ là phát triển. Chuyển miền SVMOT test9 phát lại đủ nền và ứng viên, **54/54 lượt**, HOTA **50,2753→51,0145%**, IDSW **434→440**, thấp visibility **125,67→136**. **Trượt cổng IDSW**, chưa chọn làm mặc định. Biến thể Student-t ν=4 cố định chạy đủ **36/36 dev**, IDSW **697→617** nhưng FN **+6,93%**, trượt cổng và dừng, không dò thêm ν hoặc chạy transfer. Bằng chứng `dev_gate.json`, `transfer_svmot/verified_summary.json`, `verification.json`, `robust/{dev_gate,verification}.json`. Tái hợp độc lập 24 nhóm raw metric, kiểm 95/99 hash trước tích hợp và 127 hash chẩn đoán. Các lỗi harness/định dạng ban đầu được giữ log và ghi riêng trong `01_method_limit.md`.

R112 tích hợp **Gaussian RBPF dưới tên nghiên cứu riêng** `bytetrack_rbpf_ctrv`, mã `src/rbpf_ctrv.py`, YAML `configs/phase2/trackers/r112/rbpf500.yaml`, registry và seed trong `run_tracking.py`, thêm module vào fingerprint. Không đổi mặc định, association hoặc launcher. Numerical Kalman/PSD/góc/gap/lặp đạt; trên cả backend NumPy và Ultralytics, 12 prefix gồm PF cũ và mới khớp snapshot, 6 CLI đầy đủ × 694 frame khớp MOT ứng viên, kiểm chặn tùy chọn không hỗ trợ đạt. `registered_{np,ultralytics}.json`, `postintegration_provenance.json`. Hash runner thay sau nghiệm thu thí nghiệm được ghi riêng; manifest cũ không có run_fingerprint, không tuyên bố hash module đó đã được khóa trước. Chưa đo tốc độ RBPF Nano.

R112 đã tải thật **METEOR một clip** từ nguồn tác giả, dùng HTTP Range trên ZIP chia phần để không tải cả 93 GB, CRC và SHA256 đạt. Ghim revision kho dữ liệu. Video F:/LABOTARY_data/goc_tai_ve/METEOR-R112/REC_2020_07_12_02_55_27_F.MP4, 1.800 frame 30 fps, 2.023 hộp MotorBike/Scooter, 10 ID, bốn đoạn rẽ ≥25° theo tiêu chí ảnh, không có nhãn che cho xe máy trong clip này. Pilot cùng cache COCO c25, KF/EKF/UKF IDSW **16/23/19**, PF/RBPF mean **17,67/17,67**; HOTA **18,17/17,51/17,43/16,80/19,47%**. Độ phủ candidate 31,19%, một clip, chưa cho thấy EKF/UKF vượt KF. Đã kiểm nguồn V2X-Seq-SPD có 2D box/ID/occlusion/Motorcyclist, mẫu Google Drive bị quota nên chưa tải đủ. CityFlow-V2, KITTI Tracking, TUMTraf R2/R4 có nguồn chính thức và điều kiện nhãn/quyền truy cập trong `02_datasets.md`. GRAM Urban1 audit đủ 23.435 XML, 59.693 hộp/243 ID, 7.766 hộp occluded; ít đoạn rẽ, link ảnh yêu cầu đăng nhập. **Bước tiếp** mở rộng METEOR theo mẫu và sự kiện khóa trước điểm, tăng chất lượng detection xe máy/che khuất, kiểm V2X khi hết quota hoặc TUMTraf khi được cấp quyền. Không chọn dataset vì đã biết bộ lọc thắng. Báo cáo tổng hợp `handoff/logs/review112_svmot_pf/10_ket_qua.md`.

**Tổng hợp trực quan ba kịch bản (18/11).** Đối chiếu R111 `evaluation/scores.json`, R106 `scenario_summary.csv`, R108 `svmot30/verified_summary.json` và `helmet182/verified_combined.json`; tạo biểu đồ sáu ô ở `outputs/ket_qua_ba_kich_ban_20261118.{png,pdf}` bằng `tmp/plot_three_scenarios_20261118.py`. Ba phép so chính là RoundaboutHD nhóm rẽ ≥25° trong 24.000 khung mở rộng, SVMOT 9 test nhóm visibility ≤0,5, HELMET 182 clip test. Ba ô kiểm tra thêm gồm toàn cửa sổ RoundaboutHD, nhóm che khuất SVMOT 21 chuỗi train và HOTA HELMET. Các ô đều ghi phạm vi, PF lấy trung bình ba seed theo đúng biến thể của từng benchmark; chỉ vẽ ba bộ lọc đã đối chiếu nhóm che khuất của SVMOT test9. Đã render, chỉnh khoảng cách và kiểm hình. Không chạy lại tracking trong lượt tổng hợp này.

**Quyển LaTeX theo hai mẫu PDF người dùng cung cấp (18/11), bản thảo để duyệt.** Đã cập nhật `khoa_luan_prism/` thành sáu chương, phụ lục và 15 tài liệu tham khảo, biên dịch XeLaTeX + BibTeX thành PDF A4 39 trang, Times New Roman. Tên đề tài chính thức “Nghiên cứu đánh giá hiệu năng các bộ lọc ước lượng trạng thái phi tuyến trong bài toán theo dõi phương tiện giao thông trên nền tảng nhúng Jetson Nano”. Người dùng xác nhận Nguyễn Hữu Thái Dương, 23021789, Công nghệ kỹ thuật điện tử - viễn thông, hướng dẫn Tiến sĩ Hoàng Gia Hưng. Bổ sung R108–R111, AP năm lớp detector 50/50, HELMET test182, SVMOT 9 test và 21 train riêng, RoundaboutHD phần mở rộng và hình hai trường hợp trái chiều, PF công khai và Nano 80 đoạn. Bản giao `outputs/khoa_luan_Nguyen_Huu_Thai_Duong_20261118.pdf`, gói nguồn `outputs/khoa_luan_Nguyen_Huu_Thai_Duong_LaTeX_20261118.zip` có 22 tệp, CRC/đối chiếu từng tệp đạt. Kiểm log không thiếu ký tự, tham chiếu hoặc citation, không tràn dòng; kiểm phông nhúng, metadata, đủ thông tin cá nhân và render toàn quyển. Bản PDF tại `khoa_luan_prism/main.pdf` cũng được cập nhật. Giữ năm bìa 2026 và loại quyển theo mẫu, người dùng cần duyệt nội dung/lời cảm ơn, hoàn thiện ngày và chữ ký cam đoan trước khi nộp. Không chạy thêm thí nghiệm trong lượt viết quyển này.

**R111 — RoundaboutHD phần còn lại (17/11), hoàn tất.** Đã chạy đủ 24.000 khung mới (3.001–9.000 × 4 camera) với YOLOv8n COCO c25; batch 4 khớp cache R64 ở 64/64 khung kiểm trước. Ghép cache 1–3.000 để mỗi tracker xử lý liên tục 9.000 khung. KF/EKF/UKF và PF500 K1/K2 motion2 seed 0/1/2 đủ 24 lượt, 216.000 khung replay. KF/EKF/UKF khớp đầu ra R64 trên phần đầu ở 12/12 lượt camera × filter. Nhóm rẽ ≥25° ở phần mới, 13.509 khung-xe: IDSW **34/28/28/40,33** (PF ba seed **40/45/36**), FN **5.275/5.264/5.263/5.308,67**. EKF và UKF cùng giảm IDSW **17,65%**, giảm ở camera 1–3, bằng KF ở camera 4. Ở ≥35° IDSW **29/24/24/33**. Toàn cửa sổ mới IDSW **716/726/725/1.020,33**, HOTA **40,46/40,37/40,19/37,33%**. Bốn camera cùng một vòng xuyến. Số theo nhóm dùng lịch sử ghép liền mạch; IDSW gộp cửa sổ được chấm riêng. Cache, kiểm GT và báo cáo trong `handoff/logs/review111_roundabout/`, `evaluation/scores.json`. Không đổi launcher.

Tổng hợp theo đoạn R111: **652 đoạn rẽ ≥25°**, khớp cả **13.509 khung-xe** và toàn bộ tổng IDSW/FN theo camera và seed. KF/EKF/UKF cùng IDSW ở 645 đoạn; cả EKF và UKF giảm ở 6 đoạn, cùng tăng ở 1 đoạn. Dữ liệu có camera, ID xe GT, khung và mốc giây, đường đi trên ảnh, IDSW, FN và danh sách khung đổi ID. `handoff/logs/review111_roundabout/turn_episodes.json` sinh từ `summarize_turns.py` bằng phát lại ghép CLEAR trên kết quả đã lưu.

**R110 — IDSW của bốn bộ lọc với detector Nano R109 (17/11), hoàn tất.** Giữ đúng 80 đoạn, Nano lưu lại detection gốc và xác nhận MOT/lớp KF trùng byte R109 ở 80/80 đoạn. Phát lại PC NumPy cùng cutoff/lọc lớp/ánh xạ/FPS/elapsed/buffer cho KF, EKF-CTRV, UKF-CTRV và PF500 K1/K2 motion2 (seed 0/1/2). Đủ 480 lượt, 72.000 khung replay; KF khớp hộp/lớp/điểm với R109 cả 80 đoạn và mọi metric theo clip khớp. IDSW KF/EKF/UKF/PF trung bình: BDD **1.534/1.354/1.426/1.569,67** (PF từng seed **1.594/1.562/1.553**); UA **245/250/248/276,33** (PF **287/275/267**). HOTA tương ứng BDD **43,72/43,87/43,74/42,81%**, UA **66,57/66,59/66,61/65,94%**. EKF giảm IDSW 11,73% so KF trên BDD, còn UA ít IDSW nhất là KF. So chất lượng tracking, FPS R109 chỉ áp dụng cho KF. Bằng chứng `handoff/logs/review110_nano_filters/replay_full/scores.json`, `10_ket_qua.md`; Nano `~/nano_run/r110_filters_20261117/`.

**R109 — đo lại bản Nano tối ưu trên đủ 80 video (17/11), hoàn tất.** Đúng 40 BDD100K val và 40 UA-DETRAC test từ manifest R86, mỗi video 150 JPEG trên tmpfs. Chạy v21combo và YOLO11n 50/50 + lọc trước NMS + NMS ma trận + giải mã JPEG gối đầu, TensorRT FP16 384×640, ByteTrack KF c25, tắt hiển thị, đảo thứ tự A/B. **160/160 lượt, 24.000 khung suy luận**, cùng boot trong lượt đo, lỗi thẻ 0, swap 83,30→127,50 MiB. Mỗi cặp đã đối chiếu hash ảnh, tính lại FPS/p90 từ timestamp và chấm lại TrackEval. FPS trung bình chung **13,12→18,63 (+41,99%)**, BDD **13,01→19,43 (+49,35%)**, UA **13,23→17,82 (+34,76%)**. Đạt ≥15 FPS **1/80→80/80**, thấp nhất bản tối ưu **16,66 FPS**. Trung vị p90 theo clip **90,08→123,88 ms**. HOTA BDD **33,51→43,72%**, IDSW **971→1.534**, FN **37.403→28.539**; HOTA UA **62,81→66,57%**, IDSW **280→245**, FN **22.287→17.590**. Đây là so toàn cấu hình với hai checkpoint khác nhau, đo trên đoạn JPEG và tắt hiển thị. Chưa chuyển launcher. Bằng chứng Nano `~/nano_run/r109_full80_20261117/`, cục bộ `handoff/logs/review109_nano80/results/score_verified.json`, `10_ket_qua.md`.

**R108 — công việc theo nhận xét GVHD (17/11), đang thực hiện:**

- Detection hiện tại đối chiếu R53/R63, bản gửi GVHD `handoff/logs/review108_followup/01_ket_qua_detection_gui_thay.md`. BDD val 39.973 ảnh/5 lớp: YOLOv8n COCO và YOLO11n mixed5 lần lượt mAP50 **24,39%/37,87%**, mAP50–95 **15,71%/24,94%**. UA-DETRAC test40 car/bus **87,01%/88,84%** và **59,97%/62,22%**. HELMET test182, YOLO11n mixed5 c10 IoU 0,50: precision/recall xe máy **45,17%/42,38%** từ 54.529 hộp GT. AP đo từ checkpoint trên PC. Slide dùng YOLOv8n COCO c25 cho SVMOT, YOLO11n mixed5 c10 cho HELMET và v21combo cho Nano; checkpoint Nano thử 50/50 khác checkpoint mixed5 trên slide.
- PF công khai `pfilter==0.2.5` (MIT) đã xong 12 clip UA train dev × 3 seed, 36/36. Với cùng cache, CTRV và ByteTrack, PF500 K1/K2 motion2 → SIR ngoài: HOTA **57,01→52,68%**, IDF1 **68,36→60,54%**, IDSW **697→1.359,33 (+95,03%)**, FN **39.395,67→42.983,67 (+9,11%)**. Mọi seed tăng IDSW. Giữ PF hiện tại. Phép thử so sánh hai cách cập nhật PF, không tách riêng ảnh hưởng K1, K2 và cơ chế xử lý suy biến. `pf_external/metrics.json`.
- Có đầy đủ gói gốc SVMOT (MD5 khớp Zenodo) và 7 phần HELMET trên F. R64 đã chấm đủ test SVMOT chính thức 9 chuỗi/8.250 frame. Đã chấm bổ sung 21 chuỗi/18.650 frame thuộc train chính thức, 173.242 hộp GT/849 ID tính theo clip, MOT 126/126 lượt và replay IDSW/FN theo visibility so từng chuỗi. HOTA KF/EKF/UKF/PF500 trung bình seed **51,29/52,32/52,37/51,10%**, IDSW **1.189/1.162/1.162/1.673**. Nhóm visibility ≤0,5 (22.221 hộp) có IDSW **149/152/156/171,33**, FN **12.637/12.630/12.626/12.820,33**. Kết quả nhóm che khuất không lặp lợi thế EKF từ test chính thức; 21 chuỗi là mẫu mô tả riêng. `svmot30/verified_summary.json`. HELMET hoàn tất 182 clip test/18.200 frame, 54.529 hộp GT, 1.788 ID tính theo clip. IDSW KF/EKF/UKF/PF trung bình ba seed **717/444/452/630**; HOTA **20,81/21,66/21,82/18,71%**. Riêng 146 clip ngoài mẫu cũ, IDSW **639/390/398/554,33**, HOTA **21,22/22,13/22,21/19,12%**. Tái hợp TrackEval trực tiếp từ raw đủ 18 tổng gộp; ba tracker KF/EKF/UKF trên 36 clip cũ khớp tất cả sáu chỉ số R66 được đối chiếu, PF có lệch nhỏ theo seed. `helmet182/verified_combined.json`, `helmet182/reference36_delta_vs_r66.json`.
- Kịch bản đổi hướng: kiểm GT đủ 36.000 khung/4 camera RoundaboutHD, **20.123 khung-xe qua tiêu chí rẽ 25°**, trong đó **13.509** ở đoạn sau mốc 3.000 từng camera. Số này là độ phủ GT. Tập mới VNTraffic đã tải và kiểm MD5/CRC; 501 khung có nhãn, 85 ID, 16.032 box sau bỏ bốn dòng trùng hình học mâu thuẫn. Chỉ có 24 khung-xe rẽ đạt cổng; IDSW ở nhóm rẽ 0/0/0 và FN 15/15/15 (KF/EKF/UKF). Dùng VNTraffic làm pilot, chọn RoundaboutHD để mở rộng đo đổi hướng. `roundabout_gt/coverage.json`, `vntraffic/metrics.json`.
- Jetson đã đạt mục tiêu throughput trên **16/16 đoạn JPEG**. Smoke 12/12, vòng mở rộng 64/64 (8 BDD + 8 UA × v21combo/50-50/early/matrix), vòng giải mã JPEG gối đầu 32/32. YOLO đủ 150 khung/lượt, tắt hiển thị, cùng JPEG trên tmpfs, ByteTrack KF c25. v21combo → NMS ma trận trung vị FPS **12,92→14,85 BDD**, **13,53→15,52 UA**, gain ghép cặp **+10,72%/+14,41%**. Thêm giải mã gối đầu một JPEG đạt **19,44/18,00 FPS trung vị**, **19,45/17,86 FPS trung bình**, nhanh hơn ma trận **31,55%/16,54% theo cặp**. P90 của gối đầu so ma trận **79,40→115,46 ms BDD**, **76,72→125,45 ms UA**. Các tối ưu hậu xử lý/giải mã trùng MOT/lớp 16/16 so cùng checkpoint. Ứng viên 50/50 vs v21combo trên các đoạn này HOTA **34,49→48,01% BDD, 64,63→68,47% UA**, FN **5.379→3.636 BDD, 3.751→2.584 UA**, IDSW **115→119 BDD, 35→68 UA**. Đã tính lại timestamp và TrackEval, Nano không còn job thử, lỗi thẻ 0. AP toàn tập của 50/50 trên PC qua adapter R63: BDD100K val đã xong (39.973 ảnh, tensor 384×640), mAP50 **37,89%**, mAP50–95 **24,74%**, AP50 xe máy **21,14%**. UA-DETRAC test40 đang chạy. Log `detection_interp50/`. Chưa đo camera/HDMI cấu hình này, chưa chuyển launcher. `nano_evidence/{performance_verified.json,expanded/score_verified.json,decode_overlap/score_verified.json}`.

**R107 — PF sau ghép lại, hoàn tất tự kiểm, ứng viên bị loại; chờ review Codex độc lập (12/11):** Người dùng giao trực tiếp R107; không tự xác nhận review R106 từ lời bàn giao (mục R106 bên dưới là trạng thái tại bàn giao vòng đó). Một plugin cô lập `handoff/logs/review107_filters/pf_postgap.py`: sau gap, chờ hai quan sát liên tiếp rồi dịch tâm vận tốc Cartesian của hạt, giữ logw/hộp hạt/omega/RNG; không sửa tracker/config luận văn. Cổng khóa trước trong `00_protocol.md`: IDSW giảm ≥10%, HOTA không giảm >0,5 điểm, FN không tăng >0,5%, không seed tăng IDSW, số video cải thiện > số xấu đi. Đã chạy CPU cùng cache COCO c25, PF500 K1K2 motion2, seed0/1/2 trên UA-DETRAC **train dev12: 36/36 đơn vị, 50.757 frame, 3.398 lần kích hoạt**. Mean-seed COMBINED nền→ứng viên: HOTA **57,0057→56,7771%**, IDF1 **68,3610→67,9307%**, IDSW **697→673,67 (−3,35%)**, FN **39.395,67→39.545 (+0,38%)**, FP **13.485→13.463,67**. IDSW từng seed **689/702/700→687/660/674**; 6 video giảm/6 tăng, bootstrap video delta95 **[−56;+5]**. Trượt cổng IDSW và số video; **không chọn, không mở RoundaboutHD chuyển miền, không BDD val/UA test, không dò thêm tham số**. Tự kiểm thực: 4 unit đạt sau hai lỗi fixture đã giữ log; 10 hàm hồi quy/21 assertion đạt (không gọi CLI test11 ghi ra data), disabled smoke 6/6 prefix128 khớp mảng MOT R95, ứng viên seed0 lặp đúng hai clip. Chấm TrackEval đủ ba seed, tái hợp trực tiếp sáu raw COMBINED nền/ứng viên bằng HOTA/Identity của TrackEval, kiểm 36 MOT và **91 hash** đầu vào/nguồn/config/HEAD/index không đổi sau chạy. `tier1_checks.json`, `verification.json`, `integrity_verified.json`, `diagnosis_verified.json`, `dev_gate.json` ghi bằng chứng. Giới hạn: baseline dùng raw R72 và parity prefix (không replay full); dev đã xem nhiều lần; ứng viên tính lại mean/cov sau resampling nên hộp báo ra ở frame kích hoạt cũng có thể đổi, **không là ablation thuần vận tốc** (`01_method_limit.md`). Ngoại lệ phạm vi ghi: unit đầu khiến Ultralytics tự tạo `Ultralytics/settings.json` ngoài write_paths do thư mục settings thử chưa tồn tại; đã giữ bằng chứng/hash, tạo sẵn thư mục settings trong R107 cho các lượt sau, không sửa/xóa tệp ngoài phạm vi đó. Không GPU/train/SSH/Git mutation, không agent con; mọi job tự mở đã kết thúc. Đã cập nhật mục 0 trước `10_ket_qua.md`; reviewer phiên riêng do orchestrator, **chưa có review R107 độc lập**.

**R106 — hậu kiểm dữ liệu xe hoàn tất tự kiểm, chờ review Codex độc lập (12/11):** Thực hiện theo bước R106 người dùng giao trực tiếp; không tự xác nhận mốc review R105 từ lời bàn giao. Giao thức `handoff/logs/review106_vehicle_datasets/00_protocol.md` khóa nhóm GT trước runner mở điểm R106, nhưng đã biết kết quả lịch sử nên chỉ là hậu kiểm mô tả. Tái dùng MOT/cache COCO c25 và cấu hình KF/EKF/UKF đã khóa trên **52 clip, 25.451 frame, 1.093 ID tính theo clip/camera, 162.440 hộp GT** của SVMOT, RoundaboutHD, HELMET, Sherbrooke và hai RideSafe pilot R100. Không chạy lại tracker/inference. Kiểm trực tiếp **156/156 clip×filter IDSW/FN**, 12 COMBINED raw-fields và tổ hợp HOTA/Identity trực tiếp TrackEval; chấm lại sáu MOT RideSafe khớp số lưu và ghép thành ba COMBINED (không lấy trung bình HOTA clip). Kiểm hash đầu vào, checkpoint và 12 lineage run/cache; không rehash ảnh/NPZ nguồn. Ở COCO c25, nhóm đổi hướng ảnh ≥25°: SVMOT 240 khung-xe có IDSW **2/2/3**, RoundaboutHD 6.601 có **22/22/22**, HELMET 43 có **0/0/0**, Sherbrooke không có đoạn đạt cổng; RideSafe 123 có **3/3/2** nhưng là góc ảnh camera xe ở 5 fps. Nhóm SVMOT visibility ≤0,5 có 12.051 hộp, IDSW **125/115/125**, FN **6.515/6.517/6.497**. Không kết luận ưu thế chung của bộ lọc phi tuyến. Nguồn tác giả chụp lại có HTTP/hash manifest. Tải công khai nhãn/devkit **GRAM-RTM 24.732.734 byte**, không chạy mã trong gói; Urban1 prefix 250 frame GT-only có 752 hộp/8 ID, không trùng frame/ID, không đoạn rẽ đạt cổng, chưa tracking vì thiếu ảnh/cache cùng detector. Tổng tải mới tài liệu/gói nhãn **25.121.837 byte (<25 MiB)**; hai URL Urban Tracker trả 404, KITTI tracking yêu cầu login ở link tải nên chưa tải/chấm. `tier1_checks.json`, `tier1_integrity.log`, `verification.json`, `integrity_verified.json`, các CSV và manifest trong thư mục R106 ghi lệnh/exit/bằng chứng; có ghi cả lỗi verifier ban đầu đã sửa và HEAD GRAM 403 trước GET 200. Không GPU/train/SSH/Git mutation, mọi job tự mở đã kết thúc. Đã cập nhật mục 0 trước báo cáo `10_ket_qua.md`; reviewer phiên riêng do orchestrator thực hiện, **chưa có review R106 độc lập; chưa mở R107**.

**R105 — đo/chấm Nano xong, chờ review Codex độc lập (12/11):** Bridge của phiên chính đã cung cấp **8/8 smoke và 30/30 lượt thực**, 150 JPEG/lượt, đối chứng trực tiếp v21combo, YOLO11n full/stride2/motion/guarded theo giao thức đã chốt. Bằng chứng nguyên byte và nguồn thực thi ở `handoff/logs/review105_nano_fps/host_evidence/`; không còn bị chặn phần chấm bởi SSH cũ (`01_blocker.json` giữ làm lịch sử). Tự kiểm PC trực tiếp **20/20 test**, hash archive/10 file nguồn duy nhất/driver/engine/ONNX, cấu hình KF/c25, **5.700 dòng frame**, tính lại FPS/p90 từ timestamp; **16/16 hash MOT full/v21combo khớp R102**, hai clip BDD guarded 5 fps đều 150 inference và MOT/lớp trùng full. Lệnh/exit ở `tier1_checks_current.json`, chứng cứ đối chiếu ở `verification.json`, từng lượt ở `per_run_verified.csv`, từng ID mới ở `first_match_verified.csv`, điểm TrackEval ở `host_evidence/full/assessment.json`. FPS ghép cặp YOLO11n full so v21combo: BDD **+6,00%**, UA **+6,73%**; HOTA BDD **29,05→45,64%**, UA **64,02→68,66%**. UA stride2 **+87,98% so v21combo/+76,04% so full**, FPS trung vị **25,50**, HOTA full→stride2 **68,66→68,41%**, IDSW **49→23**, FN **1.810→1.911**, khớp lần đầu tăng trung vị **80 ms** ở 14 ID mới cùng được khớp. Motion/guarded chỉ **+17,23/+7,67% so full**, HOTA **68,43/68,52%**, IDSW **26/28**, FN **1.872/1.862**; scheduler tốn **4,68–10,65 ms/frame**, MVI_39031 suy luận đủ 150 frame và chậm gần 7% so full. Giới hạn: mẫu R102 cũ, JPEG/tmpfs không hiển thị, một lượt/mode/clip; nguồn Nano là schema sớm (runner SHA `94be03b3…`), có snapshot/hash các file trọng yếu và sáu hash nguồn LF mỗi run nhưng thiếu hash toàn bộ package. Bundle local mới hơn chưa được chạy trên Nano; khác biệt ở `executed_to_current.diff`. Nhiệt tối đa 50°C, boot/EXT4 không đổi; có 4 trang swap-in, không tăng swap-out. `host_exit.json` ghi exit 0 và danh sách job R105 rỗng; coder không mở SSH/job Nano mới. Việc tiếp trong R105: reviewer phiên Codex khác kiểm code, frame, cặp FPS, hash và chất lượng; **chưa có review độc lập**, chưa có phép đo NMS/decode bổ sung qua bridge. R106 is now assigned separately by the user (see above); R107 has not started; thesis configuration remains unchanged. Đã cập nhật mục 0 trước khi viết lại `10_ket_qua.md` và brief.

**R104 — nén YOLO11n năm lớp, hoàn tất (12/11):** Chọn checkpoint FP16 **10,71→5,47 MB (−48,94%)** cho dự án. Mixed5 dev 7.190 ảnh mAP50 **49,56→49,49%**, trên hai clip RideSafe c25 recall xe máy **32,28→32,20%**, recall xe nhỏ **26,94→26,94%**, HOTA KF **12,02→12,06%**, IDSW **19→18**. ONNX chỉ lượng tử trọng số INT8 **10,57→2,90 MB (−72,53%)**, nạp ONNX Runtime CPU và chấm trực tiếp dev300, full dev dùng đúng 88 tensor đã giải lượng tử đạt **49,45% mAP50**. RideSafe INT8 recall **31,41%**, HOTA **10,45%**, IDSW **23**, phù hợp khi ưu tiên dung lượng và chấp nhận chênh lệch trên mẫu. INT8 tĩnh, INT4, 4/8 bit và QAT trọng số ngắn đều mất chất lượng trên dev300. Pruning rời rạc 30% đạt **49,59% mAP50**, kích thước checkpoint FP16 gần như giữ nguyên, RideSafe recall **30,12%**. Cắt kênh mức 25% sau ba epoch **46,01% mAP50**, cắt mức 50% + distillation một epoch **36,75%**. Phân rã hạng thấp **49,16%**, engine **8,09→7,87 MB**, GPU **41,62→42,62 ms**, không chọn. Nano TensorRT 8.2.1.9 báo `fast_fp16=True`, `fast_int8=False`. Engine dựng lại từ checkpoint FP16 gần như giữ nguyên dung lượng/tốc độ, chấm đủ **16/16 lượt** trên bốn đoạn BDD và bốn đoạn UA, HOTA **45,64→45,26% / 68,66→68,71%**, IDSW **45→44 / 49→40**. Đóng gói engine R102 gốc `.xz` **8,09→5,73 MB**, giải nén khớp SHA256. Launcher và cấu hình khóa giữ nguyên. Tệp `models/r104_yolo_compression/`, báo cáo và số liệu `handoff/logs/review104_yolo_compression/`.

**R103 — các nhánh tăng tốc Nano, hoàn tất và dừng theo yêu cầu (11/11):** Đối chứng YOLO11n 50/50 FP16 384×640/KF của R102, cùng clip và cutoff. Hoàn tất 40/40 lượt, 150 JPEG/lượt, MOT đối chứng khớp R102 16/16, lỗi thẻ 0. Chạy YOLO cách khung (75/150 inference, đọc đủ khung, dự đoán KF ở khung xen) tăng FPS ghép cặp BDD +61,52%, UA +75,53%. BDD 5 fps: HOTA 45,64→34,31%, IDSW 45→105. UA 25 fps: HOTA 68,66→68,41%, IDSW 49→23, FN 1.810→1.911, độ trễ khớp GT lần đầu tăng trung vị 80 ms trong 14 ID mới được cả hai nhánh khớp. Crop UA 256×640: GPU 41,61→29,07 ms, FPS +26,97%, HOTA 68,66→65,13%, 255/9.279 hộp GT hoàn toàn ngoài crop. Lọc conf 0,25 trước NMS: +2,29% BDD/+2,04% UA, 4/4 MOT/lớp trùng byte. Student YOLO11n bỏ attention C2PSA và bottleneck C3k2 cuối backbone, −12,8% tham số, fine-tune ba epoch: GPU 41,64→40,24 ms, FPS +1,17%/+1,93%, HOTA BDD 45,64→44,18%, UA 68,66→65,44%; mixed5 dev mAP50 49,56→45,02%, RideSafe c25 recall 32,28→42,29% trên hai clip. Các ứng viên nằm trong gói thử riêng, chưa đổi launcher. Báo cáo `handoff/logs/review103_nano_optimization/10_ket_qua.md`.

**R102 — RideSafe giữ riêng, nguyên nhân IDSW PF và YOLO11n trên Nano (11/11):** Hai clip dev 2022-10-10 và hai test 2022-10-17 khác ngày train/R100. Cutoff cache 0,10/0,15 giảm nhẹ FN dev nhưng IDSW giữ COCO/50-50 = 6/9, nên khóa 0,25 cho test. Trên 600 khung/1.388 box xe máy test, COCO→50/50 recall 22,84→32,28%, precision 88,80→75,17%, KF HOTA trung bình 7,56→12,02%, IDSW 10→19, FN 1.297→1.181. Truy vết RoundaboutHD khớp TrackEval 24/24 camera×run; PF motion1→motion2 có 143→110 IDSW qua ba seed, sau khoảng mất match 122→85. Replay có log khớp đủ 12.290 dòng MOT camera 2 seed0: 7/13 IDSW có track cũ Lost ở khung xét, 4/13 có detection IoU≥0,5 với GT nhưng chi phí ghép với dự đoán ID cũ vượt 0,8. Nano build YOLO11n 50/50 FP16 384×640 SHA `13f8cd0f…`; GPU 41,36→41,59 ms; 16/16 lượt, FPS ổn định tăng ghép cặp trung vị BDD +6,38% và UA +6,63%, hậu xử lý giảm. Chấm TrackEval trên cùng đoạn: BDD HOTA 29,05→45,64%, IDSW 55→45; UA 64,02→68,66%, IDSW 23→49. Chỉ 4 clip × 150 khung mỗi tập; engine thử riêng, chưa đổi launcher. Mẫu DAIR thử lại vẫn bị quota. Bằng chứng `handoff/logs/review102_ridesafe_pf_nano/10_ket_qua.md`.

**R101 — detector xe máy, quỹ đạo rẽ và Nano (11/11):** Hai clip RideSafe giữ riêng, 5.166 hộp, YOLOv8n COCO c25 recall xe nhỏ 5,19% (163/3.140). Fine-tune YOLO11n mixed5 hai nhánh 800 ảnh, dùng chung 600; thêm 200 ảnh RideSafe hai ngày khác, 711 GT xe máy và pseudo-label bốn lớp. Recall RideSafe c25 gốc/đối chứng/ứng viên 6,50/13,92/41,97%, nhưng ứng viên giảm mixed5 dev 49,65→47,96 mAP50, trượt cổng. Nội suy 50/50 phục hồi mixed5 dev 7.190 ảnh: 49,56 mAP50, lớp giảm nhiều nhất bus −0,58 điểm, recall RideSafe 24,37% (xe nhỏ 9,84%), precision 53,94%; qua cổng thăm dò. KF mean HOTA hai clip 5,72→17,45%, FN 5.033→4.354 và IDSW 15→53 so YOLO11n gốc; PF ứng viên 49/49/58 IDSW. Checkpoint cô lập `models/r101_ridesafe_interp50/yolo11n_mixed5_interp50.pt`; chưa triển khai. DAIR-V2X-Seq SPD hai bản mẫu và archive camera bị Google Drive giới hạn, chưa chấm. Phát lại RoundaboutHD R73 khớp TrackEval 24/24 camera×run; 6.601 khung-xe đoạn rẽ, IDSW KF/EKF/UKF 28/32/29, PF K1K2 motion1→motion2 47,67→36,67 trung bình ba seed. Nano build FP16 384×640 cắt hai bottleneck C2f chưa fine-tune, GPU 41,34→39,81 ms (−3,70%), 16/16 lượt BDD/UA đủ frame, ứng viên không sinh track trên tám clip nên loại. Chi tiết `handoff/logs/review101_detector_dair_nano/10_ket_qua.md`.

**R100 — profile TensorRT, thử tăng tốc PF và pilot RideSafe (11/11):** SSH alias `nano` hoạt động; engine C1 FP16 384×640 hash `a6912393…`, GPU median 41,35 ms, tầng lớn nhất 1,58 ms, 9 reformat copy node cộng 1,45 ms. Tái sử dụng trọng số đồng đều PF500 so nền R98: replay 8/8 đủ 150 frame, MOT/lớp trùng, cảnh đông update 93,95→90,82 ms. Live 12/12 đủ frame/MOT trùng; SD nhiễu giải mã, kiểm lại cảnh đông trên cùng tmpfs: 9,43→9,58 FPS (+1,54%), update 103,74→102,79 ms (−0,91%), không đổi launcher/checkpoint/tracker khóa. RideSafe-400 tải chọn lọc hai cặp video/XML khác ngày, giải mã 1.500 frame/clip ở 25 fps, stride 5 (5 fps), tổng 600 khung gán nhãn và 5.166 hộp xe máy, 57/214 ID theo clip. Recall ghép một-một IoU≥0,5 c05 COCO/v8 mixed5/v11 mixed5 38,54%/13,32%/21,02%; COCO c25 20,79%. Cùng cache COCO c25, IDSW cộng hai clip KF/EKF/UKF 20/26/23, PF500 K1K2 motion2 19/19/18 theo seed, R95 22/22/20. Pilot hai clip, nhiều bỏ sót và không có box gắn cờ occluded; không chọn cấu hình từ số này. DAIR-V2X-Seq SPD là ứng viên cho camera nút giao; nhãn camera trong hệ tọa độ LiDAR, chưa kiểm mẫu/chấm MOT 2D. MEVDT DOI dữ liệu trả 403 từ PC. Dữ liệu/đo/giới hạn `handoff/logs/review100_nano_pf_dataset/10_ket_qua.md`.

**Slide báo cáo lần 4, bản tổng hợp dữ liệu (11/11):** Từ bản 11 slide có biểu đồ SVMOT/HELMET, chèn đúng một slide ở vị trí 5, sau tốc độ Nano. Bảng chỉnh sửa được ghi 11 mục theo phạm vi mẫu, góc quay và nội dung đánh giá, gồm 9 mục đã chấm tracking và 2 phép thử chỉ chạy detector (VisDrone, TrafficMOT). Bản 12 slide `outputs/bao_cao_tien_do_lan_4_tong_hop_dataset.pptx`. Mở và xuất ảnh bằng PowerPoint đủ 12 slide; 11 slide nguồn khớp ngoài số trang, video và workbook giữ nguyên byte. Số liệu lấy từ kết quả đã xác minh.

**Slide báo cáo lần 4, bản thêm thống kê dataset (11/11):** Dùng đúng tệp PowerPoint 9 slide người dùng đã chỉnh trong Downloads. Thêm hai slide với biểu đồ cột chỉnh sửa được, đánh số lại thành 11 slide. SVMOT test9: 168 ID theo chuỗi, 70.220 hộp nhãn, phân bố 6 loại xe từ classes.txt và GT gốc. HELMET test36: 287 ID theo clip, 9.371 hộp xe máy, phân bố theo số người trên xe từ nhãn R66. Ghi quy mô bộ đầy đủ riêng. Hình của 9 slide gốc khớp ngoài số trang, MP4 và workbook gốc trùng byte. Bản bàn giao `outputs/bao_cao_tien_do_lan_4_them_thong_ke_dataset.pptx`.

**Slide báo cáo lần 4, bản video tracking (11/11):** Cùng hai đoạn ngẫu nhiên SVMOT seq_23_morning 344–583 và HELMET NyaungU_rural_82 1–100, chạy lại KF/EKF/UKF từ đầu chuỗi trên cache R64/R66 theo cấu hình khóa. Cả 6 đầu ra khớp MOT đã chấm trong sai số 1e-8. Bỏ ảnh tĩnh, giữ một video đồng bộ 4 ô (gốc, KF/CV, EKF, UKF) trên mỗi slide minh họa. Bản mới `outputs/bao_cao_tien_do_khoa_luan_lan_4_video_tracking.pptx`, video riêng `outputs/svmot_video_tracking.mp4` và `outputs/helmet_video_tracking.mp4`.

**Slide báo cáo lần 4, bản video ngẫu nhiên (11/11):** Theo yêu cầu sau đó, thay hai demo chọn theo IDSW bằng hai clip gốc lấy ngẫu nhiên trong các chuỗi đã kiểm. SVMOT seq_23_morning khung 344–583 (8 giây), HELMET NyaungU_rural_82 khung 1–100 (10 giây). Bản 14 slide ở `outputs/bao_cao_tien_do_khoa_luan_lan_4_video_ngau_nhien.pptx`, hai MP4 ở `outputs/svmot_clip_ngau_nhien.mp4` và `outputs/helmet_clip_ngau_nhien.mp4`. Chỉ đổi hai slide ảnh/video, không có kết quả thí nghiệm mới.

**Slide báo cáo lần 4 (11/11):** Theo yêu cầu, bỏ RoundaboutHD khỏi phần dữ liệu và giữ SVMOT cùng HELMET. Bản 14 slide `outputs/bao_cao_tien_do_khoa_luan_lan_4_svmot_helmet.pptx` giữ phần Jetson Nano/PF, thêm ảnh gốc và video đối chiếu ba bộ lọc cho mỗi tập. Theo CLEAR, SVMOT seq_25_morning GT 7 khung 130–195 có IDSW 2/0/0, FN 0/0/0, HELMET Naypyitaw_1_3 GT 3 khung 49–70 có IDSW 2/0/0, FN 4/3/3 (KF/EKF/UKF). Hai đoạn chọn sau khi xem kết quả để minh họa. Video MP4 tách riêng ở `outputs/demo_svmot_giu_id.mp4` và `outputs/demo_helmet_giu_id.mp4`. Số liệu lấy từ đầu ra đã chấm R64/R66.

**TUMTraf-V2X R4_S4 — tạm dừng theo yêu cầu (11/11):** Cổng chính thức cấp link sau đăng ký; người dùng cho biết tài khoản đang chờ duyệt và yêu cầu tạm dừng. Chưa có ảnh/nhãn gốc R4_S4, chưa chấm TUMTraf. Runner thăm dò `handoff/logs/tumtraf_v2x_pilot/run_pilot.py` đã chạy đủ trên fixture 76 ảnh RoundaboutHD với tọa độ mặt đất giả lập chỉ để kiểm đường chạy; kết quả fixture `tmp/tumtraf_pilot_fixture_20261111/run/result.json`, không dùng làm số liệu TUMTraf. Kiểm không còn tiến trình fixture.

**SVMOT — hậu kiểm quỹ đạo đổi hướng (11/11):** Phát lại CLEAR từ MOT R64 của KF/EKF/UKF trên 9 chuỗi test × 6 detector/ngưỡng; IDSW/FN khớp 162/162 đơn vị. Quy tắc góc ảnh R99 quy đổi theo 30 fps cho 12 đoạn ≥25°/240 khung-xe, 7 đoạn ≥35°/125 khung-xe. Ở YOLOv8n COCO c25, toàn chuỗi IDSW 435/407/418 nhưng nhóm ≥25° là 2/2/3 và FN 64/65/65. Không cấu hình nào trong sáu cấu hình cho cả EKF và UKF ít IDSW hơn KF trên nhóm đổi hướng. Hai cảnh chủ yếu xe chạy dọc đường, góc ảnh nhạy với hộp gần biên và rung khi xe dừng. Không chạy lại tracker. Bằng chứng `handoff/logs/svmot_turns/`. Ứng viên chưa kiểm tiếp là TUMTraf-V2X R4_S4 (drive_22 theo phụ lục tác giả), camera đường và nhiều xe quay đầu, nhãn 3D/track ID, cần xác nhận box 2D và vùng nhìn trước khi mở thí nghiệm.

**RoundaboutHD — phân tầng đổi hướng từ GT (11/11, hậu kiểm mô tả):** Cùng 36 đầu ra R64, phát lại CLEAR và đối chiếu IDSW/FN đúng 144/144 camera×run. Điểm giữa đáy hộp GT trong cửa sổ 2 giây, ≥8 khung liên tiếp đổi hướng ảnh ≥25° (có cổng chuyển động) cho 318 đoạn/6.601 khung-xe trên bốn camera. Với YOLOv8n mixed5 c25, IDSW KF/EKF/UKF/PF500 ba seed là 28/32/29/96,33; nhóm gần thẳng ≤8° là 269/269/276/609,0. Cả EKF và UKF không cùng giảm trên nhóm rẽ gộp ở bất kỳ cấu hình nào trong sáu cấu hình. Camera 4 ở mixed5 c25, 90 đoạn/1.820 khung-xe, IDSW 8/7/6 và FN 151/149/148; camera 3 ở YOLOv8n COCO c25, 38 đoạn/887 khung-xe, IDSW 4/3/3 và FN 220/219/219. Camera 3 GT 95, frame 1573–1601: IDSW 1/0/0, FN 0/0/0 ở cả hai ngưỡng COCO, xem trực tiếp ảnh xác nhận xe ôm vòng xuyến. Đây là phân tích hậu nghiệm, góc trên ảnh, bốn camera cùng một nút giao; ví dụ minh họa không đại diện cho cả tập. Không chạy lại tracker. Mã, JSON và ảnh trong `handoff/logs/review99_roundabout_turns/`.

**R98 — profile/tối ưu PF Nano và audit nhãn RideSafe (10/11, hoàn tất vòng này):** Theo yêu cầu tiếp tục mọi hướng khả thi, chọn hai mẫu UA đã có trong R97: MVI_39031 (1 tracked/0 lost median) và MVI_40855 (24 tracked/9 lost). Lưu detection cache riêng, cProfile **chỉ** `tracker.update` trên Nano: median khoảng 6,17 vs 110,91 ms/frame, chứng tỏ chi phí tăng mạnh theo số track. Ở cảnh đông, `estimate_state` cộng dồn 5,06 s/150 frame và `_logsumexp` 2,46 s; `f_batch` 1,95 s. Microbenchmark covariance `einsum` chậm khoảng ×2, đổi thứ tự nhân ma trận gần như bằng nhau ⇒ không chọn. Thử thay dispatch `_logsumexp` bằng `ndarray.max/sum` và `math.isfinite`, giữ cùng công thức: microbenchmark giảm ~30 µs/lần; replay Nano A/B/B/A trên cùng cache 150 frame/video cho MOT trùng byte cả 8 lượt, MVI_40855 update median hai lượt nền 103,69/102,33 ms vs ứng viên 97,57/98,50 ms (giảm ~5%); MVI_39031 chênh nhỏ. Live A/B/B/A cùng engine/ảnh ở MVI_40855 MOT trùng SHA 4/4, decode/infer gần nhau, median FPS steady nền 8,687→ứng viên 9,107 (+4,8%), update 113,58→108,18 ms (−4,8%); MVI_39031 MOT cũng trùng byte nhưng decode dao động lớn, không quy FPS thay đổi cho mã. PC `src/pf_ctrv.py` đã áp dụng tối ưu nhỏ, PF unit 26/26 và `test_pf_rb_observed.py` đạt; fixture R97 cũ/mới trùng tuyệt đối 255/255 hàng cả PC NumPy và Ultralytics. **Nano launcher/gói v21combo giữ nguyên**, ứng viên Nano chạy qua wrapper thí nghiệm; chỉ hai video nên chưa tuyên bố lợi ích tổng quát hay triển khai PF vào launcher. Nano cuối lượt `errors_count=0`, swap 81 MB, không còn job. Bằng chứng/giao thức `handoff/logs/review98_pf_hotspots/` và dữ liệu gốc Nano `~/nano_run/r98_{pf_hotspots,lse_live}_20261110/`.

**RideSafe-400 trong R98:** Đã tải **chỉ gói XML videoset1** từ [nguồn tác giả](https://huggingface.co/datasets/DeepBug/RideSafe-400/tree/main) về `F:/LABOTARY_data/goc_tai_ve/RideSafe-400-R98/`, 137.111.722 byte, SHA256 `bf19000c…`, CRC ZIP đạt. Audit luồng cả 100/100 XML: **100/100 `frame_filter=step=5`**, 131.028 motorcycle box hiển thị, 6.856 CVAT track có box hiển thị, 6.532 track có ≥2 box hiển thị; 119.789 bước trong-track đúng 5 frame gốc, 4.383 bước khác. Không có box motorcycle không dương hoặc trùng `(frame,CVAT track id)` trong các box hiển thị; thuộc tính `motor_track_id` thiếu/mặc định ở 2.336 box và trùng `(frame,motor_track_id)` ở 246 lượt, nên dùng CVAT track id sau audit chứ không mặc nhiên dùng thuộc tính. Cờ `occluded=1` chỉ 2/143.982 motorcycle box (gồm outside), không dùng làm phân tầng che khuất. RideSafe có giá trị cho góc nhìn camera gắn xe và huấn luyện/kiểm xe máy, nhưng nhãn stride5 không phù hợp làm tập chính cho IDSW ở chuỗi 25–30 fps; **chưa tải video 37,8 GB/chưa chạy tracking**, chưa biết FPS từng video và split cần đối chiếu. Audit ở `handoff/logs/review98_pf_hotspots/ridesafe_videoset1_audit.json`. Ưu tiên HELMET/SVMOT và tập camera ven đường/xe khác có nhãn dày cho thí nghiệm PF.

**R97 — parity PF và tốc độ nhiều video trên Nano (10/11, XONG):** Theo yêu cầu người dùng, trước hết so PF500 seed97 trên fixture 90 frame/3 xe, có rẽ và mất detection, cùng YAML hash `ff43d493…`: PC Ultralytics ↔ PC NumPy và PC NumPy ↔ Nano NumPy đều **255 hàng đầu ra/90 frame trùng tuyệt đối** (ID/lớp/box/score), qua cổng 0,001 px/1e-6; mã PC hiện tại và gói Nano cũ có hash nguồn khác, nên đây là parity fixture, không khẳng định mọi chuỗi thực tế đều tương đương. Backend Nano chỉ được thêm PF trong wrapper thí nghiệm, launcher/gói cài không sửa. Sau đó đo 8/40 video BDD + 8/40 UA đã chọn trước từ manifest R86 (chỉ số 0,5,…,35), mỗi video 150 JPEG × CV/PF500, thứ tự đảo xen kẽ: **32/32 lượt hoàn tất**, 1 boot, swap tối đa 69,9 MB, `errors_count=0`, guard nhiệt <75°C, không ảnh hỏng. So PF/CV theo median chênh từng video: BDD FPS steady **−0,10%** (CI bootstrap video [−7,42%; +7,48%], PF nhanh hơn 4/8), tracker update **+8,04 ms** (CI [+2,00; +19,21]); UA FPS **−9,53%** (CI [−10,52%; +6,50%], PF nhanh hơn 2/8), update **+34,80 ms** (CI [+9,83; +79,38]). Infer median ghép cặp BDD +0,006 ms; UA +0,143 ms. Decode biến thiên mạnh theo lượt, CI FPS chứa 0 cả hai tập: **chưa kết luận chắc PF chậm hơn toàn luồng** từ 8 video, nhưng update PF chậm hơn CV ở đủ 16/16 cặp. Trễ p90 chênh trung vị BDD +20,24 ms, UA +45,89 ms. Đây là FPS 150 JPG/video tắt hiển thị trên Nano, không phải MP4/camera hay chất lượng tracking; không hồi tố R86 (chỉ CV/EKF). Giao thức, parity, 32 meta tổng hợp và thống kê ở `handoff/logs/review97_pf_nano/`; log gốc Nano `~/nano_run/r97_pf_multi_20261110/`. Bước tiếp nếu tối ưu PF: profile theo mật độ track trên những video nhiều xe, thử giảm chi phí/particle có ràng buộc chất lượng IDSW/HOTA độc lập trước khi chọn; giữ launcher C1 v21combo.

**R96 — phân rã IDSW và pilot PF500 trên Nano (10/11, hoàn tất mức chẩn đoán):** Theo yêu cầu "làm đi, tôi ssh nano rồi", mã `handoff/logs/review96_pf_nano/idsw_events.py` phát lại đúng matching/IDSW của CLEAR TrackEval trên GT/MOT đã chấm; 200/200 BDD video × baseline/R95 × 3 seed đều khớp IDSW per-video, không đổi metric hay output. BDD trung bình seed: baseline→R95 tổng IDSW 6.291,33→6.162,33; lỗi ở lần đầu ghép lại sau ≥1 frame không có GT match **3.871,67→3.686,00 (−185,67)**; lỗi trong chuỗi GT match liên tiếp **2.419,67→2.476,33 (+56,67)**, cộng đúng −129. Trong số sau, IDSW ≤3 frame sau một lần tái xuất hiện 442,00→464,33 (+22,33). Nhãn nhóm chỉ từ lịch sử GT↔track theo TrackEval, **không phải trạng thái Lost nội bộ hay chứng cứ nhân quả**; do đó chưa chọn observation-centric reupdate là sửa lỗi chính. UA candidate dev12 ba seed có IDSW 656/646/638, đều tái hiện metric; MOT baseline R72 dev12 đã lưu trữ ngoài thư mục hoạt động, nên không so phân nhóm baseline/candidate UA full. Pilot smoke UA 2 video seed0 so được nhưng không đại diện. Báo cáo ở `handoff/logs/review96_pf_nano/10_ket_qua.md`.

**R96 Nano:** SSH tới Nano IP mới `10.11.62.64`, kiểm boot/swap/lỗi thẻ; chạy 150 JPG đầu manifest R86 cho mỗi BDD và UA trên gói v21combo+C1. Backend NumPy của launcher chỉ đăng ký CV/EKF, nên dùng wrapper **riêng thí nghiệm** để thêm PFByteTracker vào registry của tiến trình pilot, không sửa gói/launcher đang dùng. Lượt đầu BDD CV hoàn tất, PF bị từ chối ở kiểm YAML trước xử lý; lưu log riêng, sửa wrapper và chạy lại cặp ở thư mục mới. Pilot sau sửa 4/4 lượt hoàn tất đủ 150 frame, không ảnh hỏng, lỗi thẻ 0, swap trong ngưỡng; detector infer median ~44,6–44,7 ms ở cả bốn lượt. BDD CV/PF500: tracker update median **4,733/10,501 ms**, FPS steady **12,592/13,454**; UA PF500/CV: update **6,049/3,117 ms**, FPS steady **13,563/15,073**. FPS BDD đảo chiều do decode/IO biến thiên (BDD CV/PF median decode 20,39/19,07 ms; UA PF/CV 18,97/11,67 ms); **không kết luận PF nhanh hơn CV** từ pilot một video/tập. Đây là PF500 YAML mặc định trên backend NumPy thí nghiệm, chưa có parity output với bản PC/Ultralytics; đo tốc độ, không chấm HOTA/IDSW. Số liệu `handoff/logs/review96_pf_nano/nano_pilot.json`, Nano `~/nano_run/r96_pf_pilot_fix_20261110/`. Bước kế tiếp nếu đo tốc độ kết luận: kiểm parity PF NumPy với PC, chạy nhiều video đảo thứ tự chung nguồn/boot rồi phân tích update latency; nếu cải tiến IDSW thì log ghép cặp nội bộ trên dev riêng trước khi sửa recovery.

**Khảo sát hướng tiếp sau R95 (10/11, nghiên cứu; chưa mở thí nghiệm mới):** Theo yêu cầu tìm hướng cải thiện bộ lọc, dữ liệu và Nano, đã đối chiếu mã hiện hành, các lượt R87/R91–R95 và nguồn tác giả. Ưu tiên PF là nối **IDSW thật từ evaluator** với lịch sử ghép lại để phân biệt: (a) không nhận lại được/nhận sai xe; (b) nhận lại đúng rồi dự đoán lệch và mất ID tiếp. Chỉ nhóm (b) trực tiếp phù hợp với phục hồi trạng thái chuyển động từ quan sát thật sau che khuất, lấy cảm hứng [OC-SORT](https://arxiv.org/abs/2203.14360); hiện `_motion_initialized` chặn khởi tạo lại và chưa có lịch sử quan sát tin cậy. Khi thử phải cập nhật đồng bộ particle cloud/weights/K1 prior, đúng thời gian gap, tránh dùng measurement hai lần và tránh coi nội suy là nhiều quan sát độc lập; không sửa output quá khứ. Các run c25 loại detection trước ngưỡng high=0,25 nên tầng ghép điểm thấp rỗng: đây là cấu hình đã biết, không phải lỗi mới; kiểm tương tác c10/c25 với PF phải có cùng đối chứng và dev, không mặc định hạ ngưỡng sẽ tốt. PF đa chế độ Markov có posterior và [RBPF đầy đủ](https://arxiv.org/abs/1301.3853) vẫn là hướng chưa thử; R92/R94 là chế độ ngẫu nhiên khi Lost, R87 chỉ đổi ước lượng hộp đầu ra. Chưa chọn hướng nào bằng kết quả test đã xem.

**Dữ liệu/detector/Nano trong khảo sát này:** [RideSafe-400](https://huggingface.co/datasets/DeepBug/RideSafe-400/tree/main) là ứng viên ưu tiên cho xe máy từ camera gắn xe, được [repo tác giả](https://github.com/dash-cop/DashCop) dẫn trực tiếp; gói công khai khoảng 37,8 GB, nhãn CVAT có track motorcycle và `motor_track_id`. Chưa tải/audit nhãn thực tế; phải xác minh cadence, interpolation/outside, phạm vi gán nhãn và split theo nguồn video trước khi chuyển MOT. README ghi 300/50/50 video nhưng phần trăm không khớp, không dùng phần trăm đó làm manifest. HELMET/SVMOT/RoundaboutHD tiếp tục phù hợp các kịch bản xe máy/che khuất/đường cong ven đường; đoạn mới cùng cảnh không tự trở thành holdout độc lập. Detector cần thêm xe máy đúng góc nhìn và cân bằng train/dev đủ năm lớp; pilot R91 hai epoch thất bại không chứng minh mọi cách cân bằng đều thất bại. Nano cần profile **PF500 riêng** vì R86 chỉ đo CV/EKF; hướng tốc độ lớn tiếp theo là student nhỏ hơn hoặc pruning kênh thực sự, kèm distillation và đo TensorRT FP16 ở cùng độ phân giải. Distillation giữ nguyên kiến trúc không tự giảm thời gian inference; Nano Maxwell không có tăng tốc INT8 như các nền tảng mới ([ma trận NVIDIA](https://nvidia-ai-iot.github.io/torch2trt/v0.3.0/usage/reduced_precision.html)). Đây là kết quả khảo sát và thứ tự đề xuất, chưa có số đo cải thiện mới; mặc định tracker, checkpoint, launcher và khóa thí nghiệm giữ nguyên.

**R95 — PF + cứu ghép track Lost bằng hạt (10/11, đã chấm hai miền, không chọn):** Theo yêu cầu cải thiện IDSW, thử một thay đổi association opt-in `pf_assoc_particle_rescue`, chỉ cứu cặp track vừa Lost và detection vốn không qua ngưỡng ByteTrack khi phân bố hạt tương đồng cao, duy nhất và không tranh cặp đã ghép được. Đây là **PF + association**, không phải ablation thuần bộ lọc. Giao thức/cổng khóa trước ở `handoff/logs/review95_pf_reacquire/00_protocol.md`. Mặc định tắt: 6/6 MOT UA smoke trùng byte R72/R92-base; unit R95, 26/26 PF và R92 đạt. UA-DETRAC dev12 × 3 seed: HOTA 57,006→57,251%, IDF1 68,361→69,012%, IDSW 697→646,67 (−7,2%), giảm ở 10/12 video; **qua cổng UA ≥5%**. BDD COCO-video val200 × 3 seed, đủ 39.973 frame/seed: HOTA 32,951→33,187%, IDF1 37,086→37,533%, IDSW 6.291,33→6.162,33 (−2,1%), giảm 120 video/tăng 58/bằng 22, FP 18.710→19.452; **trượt cổng BDD giảm ≥3%** (cần ≤6.102,59). TrackEval raw fields tái cộng đúng; MOT hữu hạn, không trùng frame/ID, đủ 600 video-seed. Timing PC phát lại khác lượt: cập nhật tracker trung bình 6,176→7,090 ms/frame (+14,8%), chưa phải Nano benchmark. Hai tập đã dùng nhiều lượt nên chỉ là kết quả thăm dò. Không chuyển R95 vào YAML luận văn, Nano hay checkpoint; không mở tiếp HELMET/SVMOT trong giao thức R95 sau khi trượt cổng BDD. Bằng chứng `handoff/logs/review95_pf_reacquire/`.

**Chỉnh phạm vi theo người dùng 06/11:** Camera chính của đề tài là **camera giao thông ven đường** hoặc **camera gắn trên xe đang di chuyển**. UTUAV-B/C là UAV nhìn thẳng từ cao 100/120 m; UAVDT/UTUAV/VisDrone chỉ là kiểm chuyển miền phụ, không dùng chọn bộ lọc PF hay đặt hướng huấn luyện detector chính. Bộ thử chính theo góc nhìn: UA-DETRAC, HELMET, SVMOT, RoundaboutHD (ven đường); BDD100K MOT (camera xe). **R92** PF chuyển trạng thái thẳng/CTRV xác suất 0,5 khi Lost: unit đạt, mặc định tái tạo MOT R72 trùng byte 6/6 smoke. UA dev12 × 3 seed cùng cache: HOTA 57,006→57,440%, IDF1 68,361→69,356%, IDSW 697,0→642,33 (−7,8%), FP 13.485→13.482,33; IDSW giảm ở 9/12 video, CI bootstrap video delta [−103,7;−14,0]. **Trượt cổng R92 giảm ≥10%**, chưa chọn; bằng chứng `handoff/logs/review92_pf_modes/`. **R93** cùng ứng viên trên BDD val200 (camera xe), 3 seed/39.973 frame mỗi seed: HOTA 32,951→32,965%, IDF1 37,086→37,127%, **IDSW 6.291,33→6.339,33 (tăng 48)**, FP 18.710→19.194,67; IDSW giảm 82 video, tăng 96, bằng 22. Trượt cổng giảm IDSW ≥5%; không chọn. Bằng chứng `handoff/logs/review93_pf_bdd_transfer/`. **R94** thử giữ vận tốc góc ẩn khi hạt PF tạm đi thẳng lúc Lost: mặc định R92 tái tạo trùng byte 6/6 smoke, unit đạt. UA dev12 × 3 seed: HOTA 57,006→57,239%, IDF1 68,361→68,904%, IDSW 697→680; BDD val200 × 3 seed: HOTA 32,951→32,935%, IDF1 37,086→37,080%, IDSW 6.291,33→6.296,33. **Trượt cổng cả hai miền**, không chọn; bằng chứng `handoff/logs/review94_pf_preserve_turn/`. Mặc định PF, YAML luận văn, checkpoint và Nano giữ nguyên. Bước PF sau nên tập trung bất định ghép cặp/khởi tạo lại ID, thiết kế trên dữ liệu phát triển riêng và kiểm trên cảnh ven đường/xe chưa dùng để chọn, tránh tiếp tục dò xác suất trên UA/BDD đã quan sát.

**R91 — PF IDSW, detector xe máy, chọn tập xe theo kịch bản (06/11, kết thúc phép thử này):** Giao thức và cổng khóa trước ở `handoff/logs/review91_pf_assoc/00_protocol.md`, kết quả `10_ket_qua.md`. CV/PF K1K2 seed0 với assoc log trên UAVDT M1306/M0204 có MOT trùng byte R90. M1306: trong các trường hợp GT còn detection và track cũ Lost, PF có 73/803 hộp dự đoán IoU<0,2 so CV 8/655 (proxy điều kiện, không phải TrackEval IDSW). Ứng viên **opt-in** chi phí expected-IoU hạt cho Lost cải thiện M1306 IDSW 193→187, HOTA 35,48→37,13%, nhưng hại M0204 IDSW 22→24, HOTA 43,90→42,91%; **loại theo cổng từng cảnh**, không mở full/holdout. Default PF sau mã opt-in còn trùng byte. Detector YOLO11n mixed5 huấn luyện phụ 2 epoch trên 5.000 ảnh train ưu tiên xe máy: AP50 motorcycle toàn dev 0,3785→0,3622, mAP50 0,4965→0,4931; **không qua cổng, giữ checkpoint gốc**, không dùng test HELMET để chọn. UTUAV sáu frame 4K: cắt ô 960 và 640 px đều 0/148 motorcycle GT khớp đúng lớp ở IoU0,5; ô 640 có 28 lượt suy luận/ảnh, không giải quyết lỗi chuyển miền. GT HELMET test36 cho proxy rẽ ≥30° chỉ 1/36 clip, proxy hộp chồng lấn 27/36; dùng HELMET cho xe máy/giao nhau, RoundaboutHD cho đường cong, SVMOT cho che khuất có thuộc tính, UAVDT cho che khuất car/truck/bus; UTUAV đợi detector phù hợp. Không thay YAML luận văn, detector khóa hoặc Nano; PF **chưa** ngang CV về IDSW trên UAVDT. R91 là thăm dò trên tập đã quan sát, không tuyên bố tổng quát.

**Triển khai Nano C1 (04/11, theo yêu cầu người dùng qua SSH):** đã giữ nguyên gói v16 và chuyển hai launcher `~/Desktop/nano-tracking.desktop`, `~/.local/share/applications/nano-tracking.desktop` sang `~/nano_run/r63_20261030/v17c/nano_transfer`. Cấu hình riêng gói C1 trỏ engine 384×640 hash `a6912393…` và ONNX tương ứng; `07_giao_dien.sh --kiem-tra` không lỗi/cảnh báo. Smoke CV 300 JPG, không hiển thị, xử lý đủ frame, live/replay MOT trùng byte; FPS 11,56 trong điều kiện smoke này (không so trực tiếp với phép đo HDMI 13,2 FPS của G3); `/sys/fs/ext4/mmcblk1p1/errors_count=0`. Bằng chứng trên Nano: `~/nano_run/r63_20261030/deploy_c1_smoke_20261104/{result.json,live/run_meta.json}`. Hai launcher v16 được sao lưu cạnh tệp gốc với đuôi `.v16_backup_20261104`; cấu hình C1 cũ lưu `configs/nano_ui.json.pre_deploy_20261104`.

**R82 — thử bỏ copy đầu ra TensorRT và ma trận điểm lớp (04/11):** ứng viên cô lập ở `handoff/logs/review82_nano_opt/v18/src/nano_detector.py` và trên Nano `~/nano_run/r63_20261030/v18copy/nano_transfer`; launcher vẫn C1. Decode/NMS PC trùng 18/18 trường hợp, Nano CV 300 frame đủ và MOT/lớp trùng byte C1. So FPS ban đầu bị nhiễu mạnh bởi vị trí ảnh trên thẻ: decode 11,57 vs 19,78 ms. Sau khi hai bản dùng cùng thư mục ảnh, chuỗi CV tắt hiển thị B(ứng viên)→A(C1)→B→A cho thấy B 13,73/12,69 FPS và A 12,44/13,64 FPS; riêng B 13,73 dùng bản sao ảnh khác nên không hợp lệ để tính hiệu mã. Ở ba lượt chung nguồn, decode dao động 11,54–19,78 ms; ứng viên chỉ giảm median `infer_ms` từ ~45,45 xuống ~44,89 ms, chưa xác nhận mức tăng FPS toàn luồng ≥5%. **Không chọn/không triển khai ứng viên R82; giữ C1.** Bằng chứng `handoff/logs/review82_nano_opt/01_ket_qua.md` và run trên Nano `v18copy_{smoke,shared_cv}_20261104`, `v18copy_pair{,2}_c1_cv_20261104`.

**R83 — tạm dừng theo yêu cầu người dùng (04/11):** đã thử prefetch JPG queue 2/1: CV 15,25/16,05 FPS nhưng p90 trễ ~265/201 ms so C1 ~83–92 ms ⇒ không dùng cho launcher thời gian thực. LUT và LUT+giảm copy cho tensor/MOT/lớp tương đương; phép so trên cùng ảnh RAM: C1 13,67/13,78, LUT 14,64, combo 14,74 FPS. Ma trận SD+HDMI 4 cặp CV + 4 cặp EKF (16/16 lượt, 300 frame/lượt, parity MOT/lớp trùng byte, `vi_pham_cong=[]`) cho combo trung vị tăng **5,39% CV, 5,12% EKF**, p90 giảm ~4% cả hai; lưu ý cặp CV đầu có decode lệch 8,24 ms nhưng 3 cặp còn lại gần cùng decode. **Đã chuyển launcher desktop/menu sang `~/nano_run/r63_20261030/v21combo/nano_transfer`**, engine C1 giữ nguyên, sao lưu launcher C1 `.c1_backup_20261104` và v16 `.v16_backup_20261104`; `07_giao_dien.sh --kiem-tra` đạt. Bằng chứng `handoff/logs/review83_nano_opt/`, Nano `r83_c1_combo_hdmi_20261104/summary.json`. Thử MP4 bắt đầu sau đó nhưng người dùng yêu cầu tạm dừng: đã gửi TERM, xác minh không còn tiến trình thử, launcher v21combo, bộ đếm lỗi thẻ 0; **chưa có kết luận MP4, C2–C4 hoặc pruning**. Không chạy thêm đến khi người dùng yêu cầu tiếp tục.

**R84 — USB Wi-Fi Nano (05/11): ĐÃ SỬA.** Người dùng yêu cầu sửa trước khi tiếp tục thử nghiệm. SSH alias cũ `nano` (Wi-Fi 192.168.1.5) không tới được; xác định LAN 192.168.1.4 bằng khóa SSH. USB adapter `2357:011e`/driver `rtl8821au` được nhận, `wlan0` không bị rfkill, thấy SSID VIETTEL_5G; NetworkManager báo `no-secrets` do hồ sơ thiếu PSK, không phải lỗi nhận USB. Người dùng qua terminal Nano chạy `sudo nmcli --ask ...` để kết nối, rồi script `~/restore_wifi_20261105.sh` (nguồn `handoff/logs/review84_nano_wifi/restore_wifi.sh`) để lưu PSK ẩn vào hồ sơ và ngắt/nối lại; LAN vẫn hoạt động. Nhật ký xác nhận keyfile cập nhật, lần nối lại dùng `secrets exist, no new secrets needed`; Wi-Fi IP 192.168.1.5, SSH qua alias `nano` thành công, ping từ `wlan0` tới router/Internet đều 3/3, route LAN metric100/Wi-Fi metric600. Không lưu mật khẩu trong repo/log. Người dùng đã yêu cầu tiếp tục sau khi sửa Wi-Fi; có thể mở lại công việc R83.

**R83 tiếp tục — MP4 và Wi-Fi không LAN (05/11):** dùng cùng MP4 MPEG-4 300 frame trong gói v16, C1 và v21combo cùng engine/nguồn/c25/CV, hai thứ tự A→B rồi B→A. FPS C1/combo lần1 14,267/16,512 (+15,7%); lần2 15,203/16,471 (+8,3%); decode trung vị mỗi bản/lượt 3,44–3,48 ms, p90 C1/combo 78,24/73,60 và 78,18/73,39 ms. Mỗi lượt đủ 300 frame, MOT và lớp A/B trùng byte. Lượt B→A chạy sau khi người dùng rút LAN; Wi-Fi `wlan0` vẫn kết nối, SSH qua 192.168.1.5 thành công, `eth0` unavailable, lỗi thẻ 0. OpenCV hệ thống build `GStreamer: NO`; plugin `nvv4l2decoder` có nhưng chưa tích hợp/đo, decoder hiện chỉ ~3,5 ms/frame nên không gán lợi ích giả định. Bằng chứng `handoff/logs/review83_nano_opt/02_mp4_sau_wifi.md`, Nano `r83_mp4_{base,combo}_20261105{,b}/run_meta.json`. Launcher tiếp tục v21combo. C2–C4 và pruning chưa có kết luận chất lượng/tốc độ toàn luồng.

**R85 — tối ưu Nano bổ sung, kết thúc các phương án giữ nguyên mô hình (06/11):** Giao thức và bằng chứng ở handoff/logs/review85_nano_opt/. C2 352×576 bị loại trên dev12: recall che nặng giảm 2,66 điểm (15.269 GT), HOTA CV/EKF giảm 0,824/1,142 điểm so C1; không mở C3/C4. NMS ma trận giữ nguyên C1: 300 ảnh thật PC và 1.200 ca NMS Nano trùng đầu ra; 16/16 lượt Nano 300 JPG HDMI CV/EKF đủ frame, MOT/lớp trùng byte, không vi phạm giám sát; trung vị FPS cặp +3,65% CV/+2,30% EKF, p90 giảm 1,64%/1,23%, không đạt cổng +5% cả hai nên không triển khai. Truyền CUDA pinned có microbenchmark copy nhanh hơn, nhưng pipeline pin output trực tiếp tăng post; sau chép về RAM thường, hai cặp cùng tầng decode cho FPS −2,9% (pin output) và −2,4% (pin input+output), MOT/lớp trùng byte, loại. Bốn cách tiền xử lý khác trùng tensor nhưng chậm 6,1–8,6 ms so LUT hiện hành 1,44 ms, loại. Trên MP4 300 frame, hai lượt cân bằng OpenCV BGR ~3,97 ms/frame so GStreamer phần cứng appsink ~5,45–5,48 ms/frame, pixel khác nhau; loại. CUDA graph C1 lịch sử chỉ giảm GPU compute 41,59→41,38 ms; prefetch lịch sử tăng FPS nhưng p90 trễ >200 ms nên không dùng thời gian thực. Pruning/distillation chưa thử: cần quy trình huấn luyện và kiểm chất lượng mới, không suy đoán lợi ích. Launcher desktop/menu vẫn v21combo, engine C1; swap Nano 0 MB, lỗi thẻ 0. Sau DHCP Nano đổi IP .5→.3, khóa SSH đã đối chiếu; alias nano dùng jetson-desktop.local với HostKeyAlias khóa cũ. Kết quả chi tiết ở handoff/logs/review85_nano_opt/10_ket_qua.md.

**R86 — đo tốc độ Nano nhiều video BDD/UA (06/11, XONG):** So trực tiếp baseline v16 640×640 engine `9fae6b43…` với launcher đang dùng v21combo+C1 384×640 engine `a6912393…` trên 40/200 video BDD COCO val chọn hệ thống và toàn bộ 40 video UA-DETRAC test; mỗi video 150 JPEG liên tiếp ở giữa, 2 tracker CV/EKF, A/B đảo thứ tự, tắt HDMI. 320/320 lượt xử lý đủ 150 frame, không ảnh hỏng, cùng boot, swap tối đa 11 MB, nhiệt tối đa 50 °C, lỗi thẻ 0. Trung vị tăng FPS ghép cặp theo video: BDD **+51,96% CV, +47,74% EKF**; UA **+52,38% CV, +47,74% EKF**. CI95 bootstrap video lần lượt BDD CV [49,91;53,69], EKF [45,62;49,06]; UA CV [51,53;54,27], EKF [45,85;49,66]. B nhanh hơn 40/40 video ở mỗi tập/tracker. Cặp decode lệch <1 ms vẫn tăng trung vị 48,29–51,43% (n=17–20/nhóm). Đây là FPS xử lý trên **mẫu frame**, không phải toàn bộ frame hay FPS video MP4; thời gian toàn lượt 150 frame gồm khởi tạo chỉ cải thiện ~27–29%. Không đổi launcher. Giao thức, audit và số liệu ở `handoff/logs/review86_nano_multivideo/`.

**R87 — PF ước lượng điều kiện Rao–Blackwell (06/11, KHÔNG CHỌN):** Thử opt-in `pf_rb_observed_estimate` (mặc định false) để tính mean/cov hộp quan sát từ phân bố Gaussian điều kiện của proposal K1 thay vì mẫu hạt, giữ nguyên detector, association, proposal và resampling. Unit analytic/parity đạt; smoke baseline trùng byte 3/3 seed với R73. Full SVMOT 9 chuỗi × 3 seed đã chấm, cùng cache/GT R73: HOTA trung bình 50,2893% → 50,2985%, IDF1 52,5980% → 52,5991%, IDSW **436,33 → 436,33**. Không đạt cổng giảm ≥10% IDSW đã khóa trước; không mở RoundaboutHD, không đổi YAML luận văn hay launcher. SVMOT đã quan sát trước nên đây là kết quả thăm dò. Giao thức và số liệu ở `handoff/logs/review87_pf_rb/`.

**R88 — Urban Tracker René/St-Marc (06/11, thăm dò XONG):** Hai ZIP ảnh/nhãn từ R69 được kiểm nhãn SQLite và ghép 1.000 frame/cảnh; không có motorcycle. Smoke 50 frame/cảnh cho thấy YOLOv8n COCO không nhận bicycle đáng kể (René 0/67, St-Marc 2/22), nên ma trận chỉ chấm car; loại khác làm distractor. René dùng ROI mask chính thức lọc tâm detection; St-Marc không có mask. Cùng cache COCO, c25/c10, CV/EKF/UKF/PF500 mặc định 3 seed và PF K1K2 motion100 3 seed: **36/36 lượt đã chạy/chấm và hậu kiểm đạt**. René c10: CV/EKF/PF mặc định/PF motion100 IDSW 16/16/26,67/21,67; HOTA 19,31/19,11/16,88/17,77%. St-Marc c25: IDSW 7/13/9/8,67, HOTA 36,32/36,05/34,68/35,93%. PF cải tiến hơn PF mặc định nhưng không chứng minh ngang CV; chỉ 2 cảnh, detector René c25 trước tracking chỉ khớp 1.448/6.600 car GT, St-Marc có 7 ID car và FP nhiều, MOTA âm, kết quả **không phải benchmark mù/chính thức**. Giao thức, sửa trước ma trận, audit và bảng tại `handoff/logs/review88_urbantracker/`; không đổi cấu hình luận văn/Nano.

**R89 — MOT17 (06/11, LOẠI KHỎI PHẠM VI):** Đã tải và chạy thí nghiệm người đi bộ do chọn dataset sai đối tượng. Người dùng xác nhận 06/11 rằng khóa luận **chỉ theo dõi xe cộ**. Toàn bộ số liệu MOT17 ở `handoff/logs/review89_mot17/` là hồ sơ lịch sử của phép thử ngoài phạm vi; không dùng để chọn/công bố cấu hình PF, không đưa vào bảng kết quả hoặc lập luận luận văn. Không tiếp tục test tập người. Hai ZIP còn ở `F:/LABOTARY_data/goc_tai_ve/MOT17-R89/` để tránh xóa dữ liệu ngoài yêu cầu; ưu tiên tập có GT theo vết phương tiện, đặc biệt xe máy/rẽ/che khuất.

**R90 — UAVDT xe thay cho MOT17 (06/11, thăm dò XONG):** Đã tải ba gói chính thức Benchmark-M/MOTD/M_attr về `F:/LABOTARY_data/goc_tai_ve/UAVDT-R90/`, đủ byte, CRC cả ba ZIP và SHA256 lưu trong `handoff/logs/review90_uavdt/download_manifest.json`. Chỉ dùng nhãn train car/truck/bus, **không có người hoặc xe máy**; chọn trước điểm detector/tracker 8 chuỗi theo tỷ lệ che khuất, đổi hướng và đối chứng, 4.367 frame/59.472 GT xe/227 ID cộng theo chuỗi. Trang tác giả ghi 30 fps và 1080×540, nhưng ảnh thực tế 1024×540; toàn bộ ảnh chọn đủ frame 1..N, không suy diễn kích thước từ trang. Smoke 10 frame/chuỗi của ba checkpoint R54 chọn YOLOv8n COCO theo cổng định trước: 510/979 = 52,09% recall, 7/8 chuỗi ≥15%; full recall IoU≥0,5 ở conf>0,05 là 35.278/59.472 = 59,32%, che nặng 26,74%. Cache toàn bộ ảnh không lỗi/chạm cap. Ma trận cùng cache c25/c10 CV/EKF/UKF/PF500 mặc định và PF500 K1/K2 motion100 ba seed hoàn tất **18/18 run, 144/144 đơn vị video**, hậu kiểm tái cộng raw metric sai khác tối đa 5,6e-17. c25 CV/EKF/UKF/PF mặc định/PF K1K2: HOTA **28,58/28,18/27,78/24,53/26,40%**, IDSW **327/345/365/502,67/423,33**; c10: HOTA **31,48/31,53/30,84/27,14/28,87%**, IDSW **268/302/306/414,33/337,67**. PF K1K2 giảm IDSW ~15,8%/18,5% so PF mặc định nhưng vẫn cao hơn CV 96,33/69,67 IDSW. **Chỉ là phép thử chuyển miền nội bộ trên train GT**, không phải official/hidden test; M0501 gần như không có detection, GT ignore qua distractor TrackEval khác quy tắc chứa trọn box của MATLAB tác giả, FPS 30 danh nghĩa từ trang nguồn nhưng stride ảnh không được xác nhận độc lập. Không dùng R90 để chọn PF ngang CV, không thay đổi YAML luận văn/launcher Nano. Giao thức, mã, audit, chi tiết: `handoff/logs/review90_uavdt/` và `src/r90_uavdt/`.

> Mục này là **nguồn trạng thái duy nhất**. Brief, handoff, REVIEW*_CHI_TIET là lịch sử theo vòng; khi chúng
> mâu thuẫn với mục này thì mục này đúng, trừ khi có commit/bằng chứng mới hơn (khi đó sửa mục này).
> Mỗi vòng xong việc: cập nhật mục 0 trước, rồi mới viết brief/handoff. Lộ trình đọc: `AGENTS.md`.
> Chỉ mục tài liệu cũ và vùng lưu trữ: `handoff/LUU_TRU.md`.

**R81 (04/11) — thử nhiễu quá trình đối dấu cho PF, không đạt cổng BDD.** Theo yêu cầu tiếp tục kéo IDSW PF gần bộ lọc khác, thêm opt-in `pf_antithetic_process_noise` (mặc định false), ghép cặp draw Gaussian ±z khi predict; mỗi hạt vẫn có marginal N(0,Q), giảm dao động trung bình Monte Carlo. Giao thức `handoff/logs/review81_pf/00_protocol.json`; kiểm mới 4/4, hồi quy PF 26/26, UA smoke 12/12 lượt video, full dev12 **36/36** lượt video và ba seed đã chấm. Tệp MOT R72 smoke không còn lưu nên không thể kiểm trùng byte như kế hoạch; đã ghi amendment trước full và kiểm **raw TrackEval fields trùng hoàn toàn 3/3 seed × 2 video** giữa cấu hình mặc định R81 với R72. UA PF motion2 nền / đối dấu: HOTA **57,006% / 56,972%**, IDF1 **68,361% / 68,262%**, IDSW **697,0 / 675,7** (giảm 21,3 = 3,1%; CI bootstrap video [−45,7; +0,7]). Không đạt cổng giảm ≥10% IDSW, **không mở BDD, không chọn R81, mặc định và cấu hình luận văn giữ nguyên**. Bằng chứng `results/phase2/kc48_r81_pf_dev/analysis/`, giao thức và `handoff/logs/review81_pf/10_ket_qua.md`.

**Chẩn đoán sau R81 (04/11):** tính lại từ per-video TrackEval R72/R74 so EKF R70/R61, UA PF hơn EKF 225 IDSW, bốn video MVI_40213/63554/63561/40752 đóng góp 187,7 (~83,4%); PF cao hơn EKF ở 11/12 video. BDD PF hơn EKF 743,3 IDSW, PF cao hơn ở 141/200 video; 20 video gap lớn nhất đóng góp 55,0% chênh lệch ròng. Chỉ là phân bố lỗi, chưa xác định nhân quả. Bằng chứng `results/phase2/r81_pf_diagnosis/{summary.json,ua_per_video.csv,bdd_per_video.csv}`. Không tiếp tục dò scalar roughening hay cửa sổ Lost trên hai tập đã dùng nhiều lần; phương án đổi cấu trúc PF phải có giao thức và đánh giá độc lập trước khi tuyên bố lợi ích tổng quát.

**R80 (04/11) — dừng hướng chỉnh cửa sổ ghép Lost, chưa đưa PF vào cổng IDSW ngang EKF.** Giao thức `handoff/logs/review80_pf/00_protocol.json`: sau R79 max_lost=1 không tác động, thử mức nhỏ nhất có tác động `lost_recover=iou`, IoU≥0,05, `max_lost=2` ở BDD 5 fps; UA 25 fps dùng lại R78 `max_lost=3`, không đổi detector/cache/GT/evaluator. BDD smoke 9/9, full **600/600** lượt video và ba seed đã chấm; marker, cấu hình, raw-field COMBINED và preflight nguồn khớp. BDD PF nền motion2 / IoU max3 R78 / IoU max2 R80: HOTA **32,951% / 33,030% / 33,079%**; IDF1 **37,086% / 37,226% / 37,348%**; IDSW **6.291,3 / 6.712,3 / 6.253,3**. So với nền, max2 chỉ giảm IDSW 38 (bootstrap ghép cặp 200 video CI [−192,0; +129,3] đi qua 0), MOTA giảm 0,325 điểm. Max2 vẫn hơn EKF BDD 5.548 IDSW khoảng 12,7%, **không đạt cổng ≤6.102,8** đã khóa; UA R78 max3 525,7 vẫn hơn EKF UA 472 khoảng 11,4%. **Không chọn R80, không đổi mặc định/cấu hình luận văn, dừng dò độ dài cửa sổ Lost trên BDD val.** BDD val đã được quan sát nhiều lượt nên các kết quả này là thăm dò, không phải test độc lập. Bằng chứng `results/phase2/r80_pf_bdd/analysis/`, báo cáo `handoff/logs/review80_pf/10_ket_qua.md`.

**R79 (04/11) — cửa sổ Lost theo thời gian trên BDD là phép thử âm/không tác động.** Giao thức `handoff/logs/review79_pf/00_protocol.json`: tái dùng R78 UA IoU ≤3 frame ở 25 fps; trên BDD 5 fps giảm `lost_recover_max_lost` xuống 1 frame, giữ mọi thứ khác. BDD smoke 9/9 và full 600/600 lượt video, chấm đủ ba seed; raw COMBINED/marker/cache/GT qua hậu kiểm. Toàn bộ **600/600 tệp MOT trùng byte PF R74 motion2**; HOTA 32,951%, IDF1 37,086%, IDSW 6.291,3 y hệt. `max_lost=1` không tạo nhánh ghép có tác dụng ở BDD theo cơ chế hiện tại; giả thuyết thời gian hóa bằng cách làm tròn 0,12 s lên 1 frame **không cải thiện PF**. Không chọn, không đổi mặc định/cấu hình luận văn. Bằng chứng `results/phase2/r79_pf_bdd/analysis/`, `results/trackeval/BDDCOCOVID-val/r79-bdd-full-iou012s-s{0,1,2}/`.

**R78 (04/11) — PF Lost-recovery, không chọn ứng viên do không chuyển sang BDD.** Người dùng cho phép thử cơ chế ghép lại track Lost nhằm kéo IDSW PF gần các bộ lọc khác; giao thức khóa trước ở `handoff/logs/review78_pf/00_protocol.json`. Giữ PF500-K1K2 motion2, detector/cache/GT/evaluator R72/R74, ba seed. Unit `test_lost_recovery.py` 11/11; UA smoke 18/18 và full 108/108 lượt video; raw COMBINED, fingerprint, marker khớp. UA dev12: PF nền HOTA **57,006%**, IDF1 **68,361%**, IDSW **697**; `lost_no_fuse` **57,386%/69,907%/1.643,7** (loại); IoU Lost ≤3 frame **56,675%/68,030%/525,7** (IDSW −24,6%, HOTA −0,331 điểm, IDF1 −0,332 điểm; qua cổng đã khóa); Mahalanobis ≤3 frame **57,171%/68,983%/639** (không qua cổng IDSW). IoU được mở BDD val 200, smoke 9/9 và full **600/600** lượt video, raw COMBINED và marker khớp. BDD PF nền HOTA **32,951%**, IDF1 **37,086%**, IDSW **6.291,3**; IoU **33,030%/37,226%/6.712,3**: IDSW **tăng 421** (CI bootstrap video [+198; +661]), MOTA giảm 0,649 điểm. **Loại ứng viên IoU R78, không đổi mặc định/cấu hình luận văn.** Nguyên nhân có thể là cửa sổ 3 frame dài 0,12 s trên UA 25 fps nhưng 0,6 s trên BDD 5 fps; chỉ là giả thuyết, chưa xác minh. Bằng chứng `results/phase2/kc48_r78_pf_dev/analysis/`, `results/phase2/r78_pf_bdd/analysis/` và `results/trackeval/{DETRAC-all,BDDCOCOVID-val}/`.

**R77 (04/11) — dừng tìm kiếm mức roughening chuyển động; không đạt cổng BDD.** Theo yêu cầu tiếp tục giảm IDSW PF, khóa trước mức motion6/motion8 và quy tắc dừng sau hai mức (`handoff/logs/review77_pf/00_protocol.json`). Dùng opt-in `pf_allow_high_motion_roughening`, mặc định false nên cấu hình cũ giữ nguyên. Smoke 12/12, full UA-DETRAC dev12 72/72 lượt video, chấm 6/6 run; raw-field COMBINED, marker/fingerprint, cache/GT/evaluator cùng R72 đều kiểm khớp. PF500-K1K2 motion2/motion6/motion8 ba seed: IDSW **697,0 / 636,7 / 649,7**; HOTA **57,006% / 57,138% / 57,203%**. Motion6 giảm IDSW 60,3 (8,7%), CI bootstrap video [−139,3; +2,7]; motion8 giảm 47,3 (6,8%), CI [−131,7; +20,3]. Cả hai mức chưa đạt cổng định trước giảm IDSW ≥10% so với motion2, dù HOTA không giảm >0,5 điểm. **Không mở BDD R77, không chọn mức mới, giữ mặc định và cấu hình luận văn đã khóa.** Theo quy tắc dừng, không thử thêm scalar roughening; muốn tiếp tục hướng khác cần giao thức mới và quyết định phạm vi riêng. Bằng chứng `results/phase2/kc48_r77_pf_dev/analysis/`, báo cáo `handoff/logs/review77_pf/10_ket_qua.md`.

**R76 (04/11) — thử roughening chuyển động 3,0/4,0 trên UA dev12, CHƯA ĐẠT cổng BDD.** Theo yêu cầu tiếp tục giảm IDSW PF, mở riêng opt-in `pf_allow_high_motion_roughening` (mặc định false), giữ cận cũ 2,0 khi tắt. Giao thức `handoff/logs/review76_pf/00_protocol.json`; unit mới 2/2, hồi quy R72 2/2 và R75 2/2; smoke 12/12, full UA 72/72 lượt video, chấm sáu run và raw-field COMBINED khớp. PF500-K1K2 motion2 cũ / motion3 / motion4, ba seed trên UA-DETRAC dev12: IDSW **697,0 / 665,7 / 644,3**, HOTA **57,006% / 57,230% / 57,153%**. Chênh IDSW motion4−motion2 −52,7 (−7,6%), bootstrap video CI [−115,3; +0,3] chạm 0; motion3−motion2 −31,3 [−67,7; +0,3]. Cổng mở BDD khóa trước đòi IDSW giảm ≥10% và HOTA không giảm >0,5 điểm; **không mức nào đạt**, nên không chạy BDD R76 và không chọn cấu hình mới. Vẫn cách EKF 472 IDSW đáng kể, và thử nhiều mức trên dev có nguy cơ chọn theo tập. Bằng chứng `results/phase2/kc48_r76_pf_dev/analysis/`; báo cáo `handoff/logs/review76_pf/10_ket_qua.md`.

**R75 (04/11) — thử giữ tâm đám hạt sau resampling, LOẠI ứng viên.** Theo yêu cầu giảm IDSW PF để gần CV/EKF hơn, thêm tùy chọn `pf_recenter_resample` mặc định false (không đổi cấu hình cũ), bảo toàn trung bình có trọng số của hạt sau resample/roughening trước lần predict tiếp. Giao thức `handoff/logs/review75_pf/00_protocol.json`; unit mới 2/2, hồi quy R72 2/2, PF gốc 26/26 assertion. UA-DETRAC dev12, PF500-K1K2 motion2, 3 seed: IDSW **697,0 → 685,3** (−11,7; bootstrap ghép cặp 12 video CI [−40,0; +16,0]); HOTA 57,006% → 57,132% (CI chênh qua 0). BDD val 200 video, cùng cache/GT R74, 3 seed: IDSW **6.291,3 → 6.299,3** (+8,0; CI [−56,0; +71,7]); HOTA 32,951% → 32,915% (CI chênh qua 0). Smoke và full đều đủ; UA 36/36 lượt video, BDD 600/600, COMBINED tái tính từ raw fields khớp. Thay đổi không nhất quán và không đưa PF đến gần EKF; **không chọn** tùy chọn mới, giữ mặc định false và cấu hình luận văn đã khóa. Bằng chứng `results/phase2/kc48_r75_pf_dev/analysis/`, `results/phase2/r75_pf_bdd/analysis/`; báo cáo `handoff/logs/review75_pf/10_ket_qua.md`.

**R74 (03/11) — giới hạn kiểm PF ở BDD và UA-DETRAC theo yêu cầu người dùng, XONG.** UA-DETRAC train dev12 dùng lại R72, không chạy tập thứ ba. BDD COCO-video val dùng cache R61 YOLOv8n COCO, 200 video/39.973 frame, FPS 5, c25, lớp [1,2,3,5,7], buffer 6 frame, elapsed steps, `mot_crowd`; hai mức roughening chuyển động PF500-K1K2 1,0/2,0 × ba seed. Giao thức `handoff/logs/review74_pf/00_protocol.json`; cache/GT/seqinfo cả 200 video kiểm SHA trước và sau chạy; smoke 18/18 lượt video, full tracking **1.200/1.200** lượt video, chấm đủ 6/6 run, marker và COMBINED tái tính từ raw fields khớp. HOTA BDD trung bình ba seed **32,643% → 32,951%** (+0,307 điểm phần trăm, bootstrap ghép cặp 200 video, 5.000 mẫu, CI 95% [+0,209; +0,404]); ba seed đều tăng, 149 video tăng, 50 giảm, 1 bằng nhau. IDSW TB **6.670 → 6.291,3** (−378,7). CV/EKF tham chiếu R61 cùng cache/GT đạt **33,677%/33,827%**, nên PF 2,0 vẫn thấp hơn cả hai. UA R72: **56,600% → 57,006%** (+0,406 điểm, CI [−0,051; +1,052] qua 0). Hai tập cho chênh lệch dương theo điểm ước lượng, nhưng không gộp điểm qua tập, không gọi là test mù; BDD val đã được dự án quan sát từ R61, UA là dev dùng chọn tham số và các tập khác R73 không nhất quán. Giữ mặc định 1,0 và cấu hình luận văn đã khóa; 2,0 là ứng viên có lợi trên BDD theo giao thức này, chưa tuyên bố ưu thế tổng quát. Bằng chứng `results/phase2/r74_pf_bdd/` và báo cáo `handoff/logs/review74_pf/10_ket_qua.md`; mã phát lại/chấm/hậu kiểm `src/r74_pf_bdd*.py`.

**R73 (03/11) — kiểm ứng viên R72 trên tập ngoài XONG, không đổi mặc định 1,0.** Theo “thế làm nốt đi”, khóa trước so roughening chuyển động 1,0 và 2,0 (PF500-K1K2, ba seed), cùng detector và c25 của R64: SVMOTSYS-test 9 chuỗi, YOLOv8n COCO; RoundaboutHD-prefix3000 4 camera, YOLOv8n mixed5. GT/cache qua cổng hash và nội dung R64; smoke mỗi tập 6/6 run, full 12 run PF = 78/78 lượt video; CV/EKF tham chiếu bốn run R64 cùng cache/GT. Hậu kiểm 16/16 run, trường TrackEval thô tái tính COMBINED sai khác tối đa 5,56e-17. **SVMOT:** HOTA TB 3 seed 1,0 = 50,289%; 2,0 = 50,275% (−0,014 điểm), CV 48,660%, EKF 49,649%; mức 2,0 không cải thiện. **RoundaboutHD:** 1,0 = 39,746%; 2,0 = 40,658% (+0,912 điểm), IDSW 628,7→588,7, cả 4 camera có chênh HOTA dương; vẫn thấp hơn CV 42,832% và EKF 42,483%. Kết quả không nhất quán giữa hai tập; đây cũng không phải benchmark mù vì dự án đã phân tích cả hai ở R64/R65. Giữ roughening 1,0 làm mặc định, không thay cấu hình luận văn đã khóa; 2,0 có thể mô tả là ứng viên theo cảnh Roundabout, không tuyên bố ưu thế tổng quát. Giao thức `handoff/logs/review73_pf/00_protocol.json`, nguồn `source_*_full.json`, bảng và hậu kiểm `results/phase2/r73_external_pf/`, báo cáo `handoff/logs/review73_pf/10_ket_qua.md`.

**R72 (03/11) — đối chứng tăng đa dạng chuyển động XONG, ứng viên 2,0 chưa khóa.** Theo “thế cải thiện tiếp đi”, thêm cột chẩn đoán `n_ancestors` (số chỉ số tổ tiên khác nhau ở mỗi lần resampling) và tùy chọn `pf_roughening_motion_scale` mặc định 1,0. Thử mức chuyển động 0,5/1,0/2,0, hộp giữ 1,0, cùng dev12/cache/association của R71, ba seed. Giao thức `handoff/logs/review72_pf/00_protocol.json`, 78/78 hash nguồn đúng; kiểm thử mới 2/2 và hồi quy R71 3/3; smoke 18/18 và full 108/108 đơn vị đã chấm, COMBINED tái tính sai khác tối đa 5,56e-17. Bản 1,0 trùng byte với R71 36/36 tệp MOT. HOTA trung bình seed: 0,5 **56,423%**; 1,0 **56,600%**; 2,0 **57,006%**. So 2,0−1,0: +0,406 điểm phần trăm HOTA, CI video [−0,051; +1,052] qua 0; DetA +0,184 điểm [0,025; 0,364], IDSW trung bình −65,3 [−116,0; −19,7]. Mức 0,5 giảm HOTA 0,177 điểm và tăng IDSW 51,7. Chẩn đoán tổ tiên: trung vị mỗi seed tính gộp các lần resampling khoảng 237/241/246 trên 500 hạt cho 0,5/1,0/2,0; `n_unique` sau jitter vẫn 500. Tốc độ `tracker.update` phát lại tuần tự một đoạn ~11,25/11,28/11,07 ms/frame, không cho thấy tăng chi phí rõ. 2,0 là ứng viên thăm dò trên dev đã dùng để chọn mô hình, chưa chứng minh HOTA tốt hơn chắc chắn và chưa thay cấu hình luận văn đã khóa. Bằng chứng `results/phase2/kc48_r72_pf_dev/analysis/`, `handoff/logs/review72_pf/serial_speed.json`, `handoff/logs/review72_pf/10_ket_qua.md`.

**R71 (03/11) — kiểm chứng K2 có chọn lọc XONG, không chọn biến thể mới.** Theo “ok làm đi”, thêm `pf_roughening_box_scale` (mặc định 1,0 giữ K2 cũ) cho `cx,cy,a,h`; nhiễu năm thành phần chuyển động giữ 1,0. Ba mức hộp khóa trước 1,0/0,5/0,0 × ba seed PF500-K1K2 trên cùng UA-DETRAC dev12/cache/association R70. Giao thức `handoff/logs/review71_pf/00_protocol.json`, 78/78 hash nguồn đúng; kiểm thử mới 3/3 và hồi quy R70 8/8; smoke 18/18 và full 108/108 đơn vị đã chấm. Đối chứng 1,0 trùng byte R70 đủ 36/36 tệp MOT; COMBINED tái tính sai khác tối đa 5,56e-17. HOTA trung bình seed lần lượt **56,600% / 56,282% / 56,346%**. So với 1,0, mức 0,5 giảm 0,317 điểm phần trăm HOTA (CI video [−0,888; +0,023]), mức 0,0 giảm 0,254 (CI [−1,048; +0,251]); cả hai khoảng đi qua 0. DetA và MOTA giảm với CI không qua 0, IDSW chưa đổi rõ. Trung vị sai số dự đoán có điều kiện ở gap 3–8 frame giảm 0,492→0,447→0,407 chiều cao GT nhưng tỷ lệ mẫu chấm được cũng giảm 29,3%→28,1%→27,4%; không suy ra cải thiện tracking. Đo phát lại CPU một đoạn: 10,11/10,18/10,40 ms/frame. Kết luận giữ K2 1,0, không đổi cấu hình luận văn đã khóa; đây là tập dev, không phải test độc lập. Bằng chứng `results/phase2/kc48_r71_pf_dev/analysis/`, `handoff/logs/review71_pf/serial_speed.json`, `handoff/logs/review71_pf/10_ket_qua.md`.

**R70 (03/11) — đối chứng cải thiện PF đã xong trên dev12 UA-DETRAC.** Theo yêu cầu “làm đi”, dùng lại train dev12, cache YOLOv8n COCO điều kiện A, FPS 25, elapsed skip, K2=1,0 đã chọn trước đây; giao thức khóa trước chạy ở `handoff/logs/review70_pf/00_protocol.json`. Đã chạy/chấm đủ 21 cấu hình × 12 video = 252 đơn vị: PF500 → K1 optimal proposal → K1K2 roughening, mỗi loại dùng R theo chiều cao quan sát hoặc chiều cao dự đoán, ba seed; CV/EKF/UKF đối chứng. Mã mới chỉ thêm tùy chọn `pf_measurement_height` (mặc định `observation`), 8/8 kiểm thử đạt, smoke 42/42 đơn vị đạt và chạy lại hai video cùng seed trùng byte. Dev full kiểm 252 marker, COMBINED tính lại sai khác tối đa 5,56e-17; 84/84 tệp MOT đối chứng R48 (EKF, PF500 và PF500-K1 ba seed) trùng byte, 70/70 hash đầu vào không đổi. Kết quả HOTA trung bình seed: PF500 52,692%; K1 55,476%; K1K2 56,600%; K1K2-R dự đoán 56,776%; CV 59,000%; EKF 59,252%. Bootstrap ghép cặp theo 12 video (5.000 mẫu) cho K1−PF500 +2,784 điểm phần trăm [1,124; 5,514], K1K2−K1 +1,124 [0,504; 2,142], đổi R ở K1K2 +0,177 [−0,214; 0,547]; R dự đoán chưa có bằng chứng cải thiện chắc chắn. PF tốt nhất vẫn thấp hơn CV 2,224 và EKF 2,476 điểm phần trăm. Đo tốc độ phát lại một đoạn cho thấy PF vẫn tốn CPU hơn EKF; chẩn đoán lỗi dự đoán theo khoảng mất quan sát ghi rõ tỷ lệ mẫu được chấm và thiên lệch chọn mẫu. Bằng chứng: `results/phase2/kc48_r70_pf_dev/analysis/`, `handoff/logs/review70_pf/serial_speed.json`, `handoff/logs/review70_pf/10_ket_qua.md`. Đây là tập dev đã dùng để lựa chọn, không phải benchmark cuối/khẳng định trên các tập ngoài. Giữ R mặc định theo quan sát; K1/K2 là ứng viên đã có lợi ích trên dev, chưa thay cấu hình luận văn đã khóa. Các phương án association, resample-move, RBPF chỉ là hướng nghiên cứu nếu người dùng duyệt phạm vi mới.

**R69 (03/11) — tạm dừng test dataset theo yêu cầu.** Đã rà KF/CV, EKF/UKF/PF-CTRV và đường ByteTrack bằng mã thực nạp, fixture tổng hợp và bằng chứng đã lưu. Các bộ kiểm cấu hình, EKF, UKF, PF lần lượt đạt 44/44, 14/14, 12/12, 29/29 và 21/21 assertion (nhóm cuối chọn 10 test, không gọi fixture CLI số 11). Khi khởi tạo chuyển động, PFTrackView đặt lại logw đều cả khi chỉ lấy mẫu lại vận tốc/hướng. Ca kiểm có ESS 400/500 dẫn đến tâm x đổi từ 98 sang 100 px dù cx của mọi hạt giữ nguyên. Trong 30 seed của cảnh đứng rồi đi với PF500 mặc định, trọng số đã đều ở cả 30 lần nên không quan sát tác động này. Đọc lại đúng 12 log pf_diag R65 đã lưu cho 4 camera × 3 seed, cả 5.048 lần init đều nhận trọng số đã đều (4.854 lần từ activate và 194 lần từ update có reset). Loại được thao tác đặt đều này khỏi nguyên nhân hụt điểm R65; ưu tiên kiểm dự đoán trước ghép sau mất quan sát. Đã đối chiếu trọng số K1 với nghiệm Gaussian và kiểm các đường KF/EKF/UKF/PF qua sáu ảnh mất đo. Mô phỏng hộp nhiễu + sáu khung mất đo, cùng một chuỗi nhiễu cho mọi lượt và 30 seed PF: khi cua, K1 giảm sai số tâm trước ghép trung vị từ 10,05 xuống 5,53 px và khôi phục ID 30/30 so với 24/30 của bootstrap. Đây là mô phỏng, không cập nhật kết quả HOTA. Chi tiết và cách tái hiện ở `handoff/logs/review69_filters/`. Tracker và benchmark đã khóa giữ nguyên.

R69 đã tải bổ sung hai cảnh Urban Tracker vào `F:/LABOTARY_data/goc_tai_ve/UrbanTracker-R69/`. St-Marc có ZIP 2.000 ảnh và bộ nhãn, mọi ZIP qua CRC và mọi SQLite qua integrity_check. René-Lévesque có video gốc MOV, nhãn và ROI mask; giải mã đủ 8.501/8.501 frame 1280×720, metadata FPS 30000/1001, trong khi trang nguồn ghi tròn 30 fps. Đã lưu SHA256 cục bộ và URL trong `handoff/logs/review69_filters/datasets.json`. ZIP ảnh René đã tải đủ 1.569.006.177 byte, 8.501 JPG, CRC đạt; không còn tệp .partial trong thư mục. Trước chuyển đổi cần kiểm ánh xạ số frame, ROI và thống kê lớp; số ID trong SQLite khác mô tả web. Chưa tạo giao thức hay chạy tracker trên hai cảnh này.

**R68 — phép kiểm còn lại trên VisDrone và TrafficMOT (03/11/2026), detector smoke XONG; tracking full CHƯA MỞ.** Theo yêu cầu mới của người dùng \"chạy test nốt các bộ dataset kia\", ổ F đã có lại. VisDrone2019-MOT val chạy lại đúng 210 ảnh (7 chuỗi × đầu/giữa/cuối 10 ảnh) bằng helper ignore hình học của toolkit; 5 fixture đạt. Kết quả `smoke_det/ket_qua_v2.json` trùng toàn bộ detector fields và cổng với R66: 3.216 GT vehicle4, recall ở conf >0,05 là **52,394% / 41,387% / 44,465%** theo YOLOv8n COCO / YOLOv8n mixed5 / YOLO11n mixed5. Cổng khả thi đạt; không còn coi kết quả smoke cũ là tạm thời. Tuy nhiên FAQ tác giả chỉ xác nhận video gốc 24 fps và ảnh được lấy mẫu, chưa rõ khoảng thời gian giữa ảnh phát hành; theo Q3 **không chạy tracking full** hoặc tự dùng FPS 24.

TrafficMOT được mở lại **chỉ cho detector smoke thăm dò trên train27**, không đổi quyết định Q2 về tracking benchmark. Mẫu khóa đầu/giữa/cuối mỗi clip (`frame0/30/58`) gồm 81 ảnh, 931 GT của bốn lớp ánh xạ trực tiếp Motor_Bike/Bus/Bike/Truck; giữ ba checkpoint R54, cache spec R64, matching class-aware IoU≥0,5. Ở conf >0,05, recall gộp của YOLOv8n COCO / YOLOv8n mixed5 / YOLO11n mixed5 lần lượt **203/931 = 21,805% / 37/931 = 3,974% / 33/931 = 3,545%**; riêng xe máy **175/823 / 14/823 / 10/823**. Kết quả `handoff/logs/external_eval/r66/trafficmot/smoke_det_train27.json` là kiểm khả năng detector trên train, **không phải TrafficMOT test**. 51 clip test cùng GT chưa có trong gói, FPS/stride nguồn chưa xác minh, và lớp ngoài tập con chưa có giao thức chấm; **không chạy tracking full** hay xếp hạng bộ lọc từ smoke này. Mã `src/r66_ngoai/trafficmot_smoke_det.py`.

**R67 — UTUAV v2 B/C validation, Q3 ĐÓNG (03/11/2026).** Thư trả lời của tác giả Jorge Ernesto Espinosa Oviedo do người dùng cung cấp xác nhận B/C quay ở **30000/1001 fps**, ảnh phát hành liên tiếp **stride 1**, trong mỗi block Sequential_Gap300 không có bỏ frame; gap 300 giữa các block khoảng 10 s. Bước thời gian `1001/30000 s`, buffer 1,2 s = **36 frame**. Bằng chứng là nội dung thư trong chat (chưa có file email gốc/hash), ghi ở `handoff/logs/external_eval/utuav/00_nguon_fps.md`; bài báo *Drones* 2026, 10(1), 15 chỉ mô tả chia tập, không ghi tường minh FPS/stride. Giữ pilot R64 30 fps làm lịch sử, không hồi tố.

Giao thức Q3 đã chạy đúng B validation **26341–27078 (738 frame)** và C validation **20271–21021 (751 frame)**, ảnh gốc 4K trên F, GT 6 lớp XML từ gói nhãn trên D: động motorbike/car/bus → MOT class 1, `_stat` → distractor class 8, evaluator `mot`; tracker 3 checkpoint R54 × CV/EKF/UKF/PF500 s0/1/2 × c25/c10, FPS chính xác. 1.489 JPG đã xác minh/giải nén chọn lọc; toàn bộ XML có **261.303 object**, trong đó **154.761 GT động**. Audit phát hiện **45 box nguồn vượt mép ảnh tối đa 4 px** (B 3, C 42), không có ID trùng/box không dương/lớp lạ; giữ nguyên tọa độ, không cắt hay chữa nhãn. `occluded` toàn 0, không phân tầng che khuất. Smoke 128 frame đầu mỗi chuỗi đạt **36/36 run**, PF cùng seed lặp trùng byte MOT/score; fixture oracle/distractor/output rỗng và review kỹ thuật PASS. Full đạt **36/36 run, 72/72 đơn vị chuỗi**, hậu kiểm 6 cache và toàn bộ raw-field COMBINED khớp sai số tối đa `9,89e-17`: `handoff/logs/external_eval/utuav/full/final_review.json`. Dữ liệu và công cụ ở `src/r67_utuav/`; giao thức `handoff/logs/external_eval/utuav/01_giao_thuc.json`.

**Giới hạn kết luận R67:** ở conf detection `>0,05`, cận trên recall class-agnostic IoU≥0,5 cao nhất trong ba detector là YOLOv8n COCO: **520/29.756 = 1,75% B**, **1.107/125.005 = 0,89% C** (`full/detector_upper_bound_recall.json`). HOTA COMBINED full mọi cấu hình ≤0,46%; chênh lệch bộ lọc gần 0 và không đủ chứng minh lợi thế EKF/UKF/PF khi detector chuyển miền kém. Báo kết quả mô tả theo từng B/C và COMBINED, PF ba seed, không CI (hai cảnh), không so điểm tuyệt đối với tập khác. Báo cáo `handoff/logs/external_eval/utuav/10_ket_qua.md`.

**R66 kết thúc phạm vi trước R67 (03/11).** Sherbrooke full và HELMET test36 full đã đạt đủ 36/36 mỗi tập. VisDrone xong smoke detector, chưa đủ căn cứ FPS hiệu dụng để mở tracking. TrafficMOT hoãn theo Q2. Q6 là đề xuất cải tiến PF riêng, cần quyết định riêng. Giao thức ở `D:/LABOTARY/automation/runs_local/hoi_codex_R66_dataset_moi_pf_20261102/answer.json`.

**VisDrone smoke R66 là lịch sử đã được kiểm lại ở R68.** Bản R66 dùng hình học ignore xấp xỉ; R68 chấm lại bằng helper theo `dropObjects.m` và thu được các detector fields trùng hệt. Cổng FPS/stride hiệu dụng vẫn chưa đạt.

**Bản thảo khóa luận (03/11):** có chỉnh sửa nguồn TeX để thêm kết quả HELMET/UTUAV đã xác minh và sửa trích dẫn; người dùng yêu cầu chưa viết tiếp và giữ các chỉnh sửa. PDF hiện hành chưa biên dịch lại, **không** coi là bản cập nhật; không tiếp tục viết/biên dịch khi chưa có yêu cầu mới.

**HELMET test36.** Đã kiểm size/SHA/CRC 7 gói ảnh, giải nén đúng 36 clip đã khóa, 12 nhóm cảnh, 3.600 frame, 9.371 box xe máy, 287 ID theo clip. Smoke 432/432 đơn vị và lặp PF cùng seed trùng MOT/điểm. Full 1.296/1.296 đơn vị có marker và điểm tái tổ hợp khớp. Trong 6 cặp detector/ngưỡng, EKF cao hơn KF 0,18–1,16 điểm HOTA, UKF cao hơn 0,25–1,12 điểm (hiệu chưa làm tròn tối đa 1,124); PF500 trung bình ba seed thấp hơn KF 1,35–3,93 điểm. Với YOLO11n mixed5 c10, KF/EKF/UKF/PF500 có HOTA 18,55/19,09/19,64/16,38 và KF/EKF/UKF có 78/54/54 IDSW. Box nguồn thường bao cả người ngồi. Kết quả mô tả theo giao thức test36, chưa phân tầng thao tác rẽ hay che khuất. Hậu kiểm và báo cáo ở `handoff/logs/external_eval/r66/helmet/full/final_review.json`, `handoff/logs/external_eval/r66/helmet/10_ket_qua.md`.

**Sherbrooke.** Full v2 hoàn tất **36/36** trên 1.001 frame. PF500 trung bình ba seed thấp hơn KF/CV **6,25–7,73 điểm HOTA** ở sáu tổ hợp detector/ngưỡng. EKF và UKF cách KF không quá 0,14 điểm. Đây là một camera và 15 ID của phạm vi cars. Bằng chứng ở `handoff/logs/external_eval/r66/sherbrooke/v2/full/{coverage.json,scores.csv,filter_summary.csv}`. Sai lệch quy trình khi tiếp quản: đã mở sớm 26 lượt có marker cũ, rồi dừng và lưu riêng theo `v2/noncanonical_manifest.json`. Review B1–B7 ở `hoi_codex_R66_review_smoke_sherbrooke_20261102/answer.json`. Chấm lại 36 smoke cũ + 2 lượt lặp, sửa AssA khi TP=0, ROI và provenance, sinh lại cache, lặp hai seed, 19 fixture đạt. Chỉ dùng ma trận sau quyết định `v2/review_smoke.json`.

**Soạn khóa luận (02/11/2026):** Đã viết bản thảo tiếng Việt 5 chương từ khung Prism, kèm phụ lục tra cứu số liệu, 3 hình, 9 bảng và 11 tài liệu tham khảo. PDF đã biên dịch và kiểm dàn trang: `outputs/khoa-luan-ByteTrack.pdf` (33 trang). Gói nguồn chỉnh sửa: `outputs/khoa-luan-ByteTrack-Prism.zip`, tệp chính `khoa_luan_prism/main.tex`. Số liệu dùng hồ sơ R51–R65 đã xác minh; không chạy lại thí nghiệm. Tên đề tài và thông tin trường theo khung còn tạm; người học cần điền thông tin bìa, cam đoan, cảm ơn và đối chiếu quy cách khoa trước khi nộp.

**R65 — chẩn đoán PF500 thấp trên RoundaboutHD (02/11/2026) — ĐÓNG, kết luận mô tả, nguyên nhân chưa xác định.**
- **Nguồn quyết định:** Q2 của GPT (`D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/answer.json`).
- **Phạm vi khóa trước:** imagesc001–004 frame 1–3000; yolov8n_mixed5 c25; CV/EKF/UKF + PF500 s0/1/2 (24 đơn vị);
  cache, GT và cấu hình R64. Không tune, không ablation, R64 giữ nguyên.
- **Mã và bằng chứng:**
  - mã `src/r65_pf_roundabout/` (test 5/5);
  - báo cáo `handoff/logs/review65_pf_roundabout/10_ket_luan.md`;
  - số liệu `results/phase2/r65_pf_roundabout_diag/`.
- **Hai lượt:**
  - Lượt 1 (`r65{off,on}`) bị GPT review chỉ ra 6 lỗi phương pháp.
  - Lượt 2 (`r65b{off,on}`) đã sửa. Review GPT lần 2 (`runs_local/hoi_codex_R65_review2_20261102/answer.json`) kết
    luận CẦN SỬA TRƯỚC COMMIT. Đã sửa offline, không chạy lại tracking: nghĩa thời gian, finding Removed, giảm khẳng
    định nhân quả, siết cổng/fixture, provenance hậu kiểm `manifest_hau_kiem.json`.
- **Cổng "log không đổi hành vi" đạt 24/24:** MOT trùng byte off = on = R64; hash trạng thái sau dự đoán, sau update và
  từng bước PF trùng off = on.
- **Kết quả (mô tả, trong mẫu nối được GT):**
  - **Dự đoán:** box dự đoán PF lệch GT hơn CV ngay ở frame liền kề (sai số tâm trung vị 0,095 h so với 0,063 h;
    IoU 0,74 so với 0,81).
  - **Association:** chênh lệch tập trung ở nhóm "có det phù hợp nhưng chi phí không đạt" (8,6–8,8 % so với 3,7 %),
    chủ yếu ở track Lost. Tranh chấp và thiếu det tương đương giữa các bộ lọc.
  - **Phân mảnh:** Frag +16–19 %, MT 79 → 46–48, IDSW khoảng ×2,2.
  - **Nội bộ PF:** ESS/N trung vị 0,19, resample gần mọi update. Khi track thiếu ≥ 3 frame quan sát, sai số tăng nhanh
    hơn độ trải hạt.
  - **Nguyên nhân:** chưa xác định. Theo GPT, **dừng R65, không đề xuất ablation**.
- **Finding phụ (mọi bộ lọc, không sửa):** `merge_track_pools` của ultralytics 8.4.141 (thực nạp) và bản vendor để
  track vừa Removed được dự đoán thêm một lần (state 3, n_lost 20). Track được ghép lại lúc đó vẫn nằm trong
  `removed_stracks`, nên bị loại ngay ở lần mất kế tiếp (40/40 ca).
- **Dung lượng:** 363 MB cả hai lượt (trần 1 GiB).

**Ngoài lề 02/11 (yêu cầu người dùng):**
- **UTUAV (Q3):** không xác minh được FPS/stride nguồn ⇒ giữ pilot (`handoff/logs/external_eval/utuav/00_nguon_fps.md`,
  có nháp thư hỏi tác giả để người dùng tự gửi).
- **Video so sánh trước/sau tối ưu Nano:** `results/track_video/r63_so_sanh_nano/` (không commit). Script
  `handoff/logs/review63/cong_cu/dung_video_so_sanh.py` phát lại cache Nano, MOT trùng hash Nano.
- **Ổ rời:** 4 archive tải về (khoảng 50 GB) đã chuyển sang ổ rời F: với kiểm sha256. Quy tắc "F: chỉ chứa thứ tải
  lại được" ghi ở `handoff/LUU_TRU.md` mục 5.
- **Git:** người dùng yêu cầu push (02/11); R63–R65 và phần ngoài lề đã lên origin `phase2-mixed-traffic`.

**Pilot UTUAV ngày 01/11/2026 — hồ sơ lịch sử** (theo yêu cầu người dùng); tại thời điểm pilot SVMOT chưa tải xong.
Trạng thái SVMOT hiện hành xem R64 bên dưới. Hai archive UTUAV tại
`D:/LABOTARY/` đã đọc hết luồng, CRC gzip đạt; gói ảnh có 11.898 JPG, gói GT có XML `track_id` cho B/C. Đã giải nén
chọn lọc hai đoạn validation 120 frame liên tiếp: B đi thẳng `26401–26520`, C có xe máy rẽ `20901–21020`; giữ nguyên
tracker/config/metric, dùng YOLO11n mixed5 theo lựa chọn người dùng. **Pilot YOLO11n mixed5 không ghép được GT động ở
IoU ≥ 0,5 trên các frame đã thử; chưa đủ kết luận cho toàn UTUAV hoặc ba detector:** C 10 frame 640 có 0/1.795 GT động
khớp IoU 0,5 ngay cả ở conf 0,05; C 3 frame 1280 và B 3 frame 640 cũng 0 khớp (smoke, không phải benchmark). Không chạy
benchmark YOLO dài. Chẩn đoán **riêng, không phải kết quả YOLO** dùng GT-synthetic 3% jitter/15% dropout cùng cache,
và kịch bản thêm gap 8 frame định trước (chỉ dùng `gap8_v2`; gap v1 không dùng kết luận): KF/CV, EKF và UKF gần nhau —
kịch bản bỏ sót độc lập có UKF HOTA cao hơn CV 0,004 điểm phần trăm, kịch bản `gap8_v2` có CV nhỉnh hơn; PF100 thấp
hơn ở cả ba seed; lớp xe máy riêng cùng chiều. `occluded` trong UTUAV đều 0, không dùng để phân tầng che khuất. FPS 30 chỉ là giả định cho pilot, chưa xác minh
nguồn. Bằng chứng/giao thức/giới hạn: `handoff/logs/external_mot_prep/02_pilot_utuav.md`; chưa khóa benchmark mới.

**R64 — đánh giá hệ thống trên tập ngoài (01–02/11/2026) — SVMOT + RoundaboutHD XONG 36/36 mỗi tập; UTUAV chỉ pilot.**
Người dùng (01/11 đêm): "hệ thống đã hoàn chỉnh từ huấn luyện tới đánh giá, giờ mang đi test trên các dataset khác",
phạm vi "đủ như R54/R61" (3 detector × CV/EKF/UKF/PF500 s0/1/2 × c25/c10); Claude chạy + viết báo cáo, GPT quyết giao
thức và review. Giao thức GPT (`D:/LABOTARY/automation/runs_local/hoi_codex_R64_dataset_moi_20261101/answer.json`):
**SVMOT** 9 chuỗi test chính thức (8.250 frame), ảnh toàn khung (không tô roi_mask — giao thức full-image dự án),
`mot`, 30 fps, buffer 36 bước; **RoundaboutHD** frame 1–3000 mỗi camera imagesc001–004 (split `RoundaboutHD-prefix3000`),
GT `SCT_GT.txt` (xe đỗ ngoài GT chịu FP — đánh giá theo phạm vi SCT), 15 fps, buffer 18 bước; **UTUAV** chỉ trích pilot
(FPS nguồn chưa xác minh). Thống kê chỉ mô tả, **không CI** (2 scene / 4 camera một nút giao). Review smoke
(`.../hoi_codex_R64_review_smoke_20261102/answer.json`): B1 — lỗi nguồn RoundaboutHD trùng (frame, id) ⇒ **bỏ 16 dòng
cam4 (ID 105, frame 2536–2543)**, GT cũ (distractor lớp 8, bị bác vì TrackEval kiểm ID trước lọc) giữ ở
`data/processed/_r64_bang_chung/`; B2 — viết lại cổng hoàn tất/nội dung/provenance/hậu kiểm/ngân sách/PARTIAL; sai lệch
Q4 (mở cache RoundaboutHD trước khi xong SVMOT) ghi và dừng. Commit **`f588456`** (mã + báo cáo), sửa review cuối
**`92fd797`**. **Review cuối**
(`.../hoi_codex_R64_review_cuoi_20261102/answer.json`): **DAT_CO_SUA** — GPT tái tính 72 COMBINED + 468 dòng chuỗi/camera
+ 72 dòng scene khớp (≤ 2,2e-16), hash GT/mã khớp; đã sửa: số làm tròn/câu chữ PF ở SVMOT.md, khoảng COMBINED và giả
thuyết PF ở RoundaboutHD.md, UTUAV B evaluation 975 frame liên tục, `kiem-run` thay kiểm `ls` lỗi trong `chay.sh`, cổng
fingerprint + legacy cần biên bản gắn hash marker + provenance bắt buộc, tiền kiểm ngân sách ghi `ngan_sach.jsonl` + dự
phòng 30 phút, bằng chứng lặp PF `external_eval/smoke/04_lap_pf_cham_lai.json`; test 15/15; kiểm lại cổng mới trên 39 đơn
vị cache + 48 cấu hình run (24 mỗi tập; một cấu hình PF gồm ba seed ⇒ tổng 72 run đã chấm) ĐẠT
(`external_eval/10_kiem_lai_cong_moi.log`). Lượt đêm **thiếu nhật ký ngân sách** (ghi nhận, không hồi tố).
**Kết quả (mô tả):** SVMOT — yolov8n_coco cao nhất (c25 HOTA CV 48,66 / EKF 49,65 / UKF 49,21 / PF500 50,70); mixed5
DetA thấp ~10 điểm, MOTA rất thấp; ở c25, theo COMBINED (PF500 lấy trung bình ba seed), CTRV hơn CV 0,3–2,0 điểm
HOTA — không áp khoảng này cho c10 (có chênh EKF−CV chỉ ~0,15 điểm); lợi thế PF500 tập trung ở scene_06; PF500 IDSW
cao hơn EKF/UKF ở cả 6 nhóm. RoundaboutHD — mixed5 hơn coco 4–6 điểm HOTA (coco nhiều FP, một phần xe đỗ ngoài SCT_GT); CV/EKF/UKF chênh
≤ 0,6 (COMBINED); **PF500 thấp hơn rõ và nhất quán** (COMBINED TB 3 seed: HOTA −11,56…−12,35, IDSW ×2,22…×2,91; thấp
hơn CV ở đủ 72 tổ hợp camera×detector×ngưỡng×seed), nguyên nhân chưa xác định. Chiều PF500 đảo giữa hai tập ⇒ không
khái quát kết luận bộ lọc. Báo cáo: `handoff/logs/external_eval/{SVMOT,RoundaboutHD,UTUAV}.md`. Mã:
`src/r64_ngoai/`. Ảnh giải nén ngoài repo: `F:/LABOTARY_data/svmot` (ổ rời), `C:/LABOTARY_data/RoundaboutHD`.
Giới hạn giữ nguyên: SVMOT full-image là giao thức dự án; SCT có xe đỗ ngoài GT chịu FP; bỏ đúng 16 dòng cam4; không
CI; thiếu nhật ký ngân sách lịch sử; chưa xác định nguyên nhân PF.
**Quyết định việc tiếp (GPT 02/11, `D:/LABOTARY/automation/runs_local/hoi_codex_R64_viec_tiep_20261102/answer.json`,
người dùng: "hỏi chat gpt đi"):** thứ tự Q1 → Q2 → Q3 → Q4. Q1 sửa + commit mục 0 (lượt này). **Q2 LÀM NGAY (đã làm = R65, ĐÓNG, xem đầu mục 0):** chẩn
đoán PF500 RoundaboutHD (R65) — phạm vi khóa trước: imagesc001–004 frame 1–3000, yolov8n_mixed5 c25, CV/EKF/UKF + PF500
s0/1/2 (24 đơn vị), cache/GT/cấu hình R64, không tune/không thay benchmark; cổng parity log tắt/bật trùng byte; ngân
sách ≤ 2 giờ; đầu ra `results/phase2/r65_pf_roundabout_diag/`, `handoff/logs/review65_pf_roundabout/`. **Q3 (đã làm: FPS chưa xác minh ⇒ giữ pilot) LÀM NGAY
có điều kiện:** tìm bằng chứng FPS nguồn UTUAV (≤ 30 phút, nguồn tác giả/chính thức, phải biết stride); chưa xác minh
⇒ giữ pilot, không chạy ma trận; xác minh được ⇒ adapter + smoke 128 frame + ma trận 36 run × B/C validation
(`UTUAVSYS-validation`). **Q4 LÀM SAU:** CI AP test40/BDD val không bắt buộc cho chương kết quả; ghi "AP là ước lượng
điểm; chưa tính CI". Tài liệu pilot `handoff/logs/external_mot_prep/` và khảo sát `review62/32` còn **untracked** —
bằng chứng đang ở máy local, chưa coi là đã bảo toàn trong Git.

**R63 — vòng tối ưu Nano (30/10–02/11/2026): C1 đã chốt trên dev làm ứng viên; báo cáo cuối và G6 xong; launcher vẫn
giữ v16** (commit `77662de`, `bd72eb4`, `c162d98`). Tốc độ: baseline là mã v16 với engine build lại `9fae6b43…`, đo
trên Nano hiện tại; mức +42–44 % thuộc 5/8 cặp cùng tầng decode. Người dùng yêu cầu mở vòng mới. Đã làm phần
chuẩn bị không cần Nano: kế hoạch `handoff/logs/review63/01_ke_hoach_r63.md` (baseline, đo, tiêu chí đạt/dừng, tái lập,
câu hỏi Q1–Q5); công cụ `review63/cong_cu/` (kiểm sức khỏe chỉ đọc, trtexec, bench engine với CUDA event, xuất ONNX,
build engine; test parser 5/5) và `review63/chat_luong/phan_tang_kich_thuoc.py` (test 3/3). Phát hiện: Nano letterbox
vuông 640×640 (43,75 % pixel là đệm với video 960×540), còn đánh giá PC R53–R61 chạy ở 384×640; xuất lại ONNX 640 tương
đương đồ thị v16; ở bước chuẩn bị ban đầu, các ONNX ứng viên 384×640/352×576/288×512/256×448 mới xuất (op trong bảng
TRT 8.2), chưa build/đo; kết quả đo sau đó xem G1/G3 bên dưới.
Nano đổi IP sang **192.168.1.5** (alias `nano` đã cập nhật). G0 tiền kiểm chỉ đọc (16:54) đạt: L4T 32.7.5, TRT 8.2.1,
MAXN, nhàn rỗi 33,5 °C, swap 0, không panic trong boot hiện tại, pstore rỗng (journal không giữ boot trước); engine
`c32690cd…`, v16 562/562, 1.470 ảnh khớp hash (`review63/g0/02_ket_qua_g0.md`). Tại G0 ngày 30/10 chưa đo tải. v16
vẫn là bản dùng; chưa sửa `src/`. Người dùng giao GPT duyệt: **DUYET_CO_SUA** (`D:/LABOTARY/automation/
runs_local/hoi_codex_R63_duyet_ke_hoach_20261030/answer.json`); đã sửa công cụ theo review (ghép tối đa số cặp, runner
giám sát `chay_co_giam_sat.sh`, kiểm parity/shape/hash, tóm tắt G0 không báo giả). **17:10 smoke 5 ảnh trên bản sao v16
làm Nano KERNEL PANIC** (hard lockup), cùng chữ ký R62: `mmc1 Data CRC error` → `xusb HC died` → I2C 0x3c/max77620
timeout → RCU stall → panic. **Khi đó (30/10) đã dừng mọi phép đo GPU** (đã mở lại sau xử lý thẻ SD, xem dưới); nghi
nguồn/thẻ SD/phần cứng (suy luận). Biên bản:
`review63/su_co_20261030_1710/01_bien_ban_su_co.md`. **Chẩn đoán 17:50:** nguồn là jack DC 5 V 4 A (nhàn rỗi 5,128 V);
người dùng chạy `dd` CHỈ ĐỌC thẻ SD, không GPU ⇒ máy **panic cùng chuỗi** (`mmc1 Data CRC` → `xusb HC died` → RCU
stall → hard lockup), công suất lúc sập ~2,3 W ⇒ nghi đường thẻ SD (thẻ/khe/SDR104); lỗi xuất hiện cả khi đọc thẻ
không dùng GPU (`review63/su_co_20261030_1750_dd/01_bien_ban.md`). **Phân định 30–31/10 (cùng biên bản):** boot rootfs từ eMMC
(`extlinux.conf` → `root=/dev/mmcblk0p1`, bản gốc `extlinux.conf.r63_goc_the_sd`), cùng board/khe/nguồn/Wi-Fi USB:
thẻ hộp đọc toàn thẻ 7,1 h **không lỗi mmc1** (SDR104 204 MHz, nhưng chỉ 2,4 MB/s); thẻ acer **sập sau ~5 phút** ⇒
**bằng chứng nghiêng về đường thẻ acer ở chế độ SDR104 204 MHz**. Hệ eMMC là JetPack cài dở (TRT `iU`, cuDNN `rU`, không CUDA runtime, apt hỏng, OpenCV
4.1.1 ≠ 3.2.0 của v16) nên không dùng. **Người dùng chọn B1 (31/10):** mua thẻ mới, đọc ảnh thẻ acer trên PC, ghi
sang thẻ mới để giữ nguyên môi trường v16/R62; rồi kiểm v16/engine/env, `dd` toàn thẻ mới, G0, lượt ngắn có giám sát
trước mọi phép đo R63. **Cập nhật cùng ngày:** trước khi mua, thử hạ xung khe SD xuống 100 MHz (debugfs, tạm thời)
⇒ thẻ acer đọc toàn thẻ **sạch** (21,2 MB/s, 0 lỗi mmc1) ⇒ lỗi không tái hiện trong các lượt đọc ở xung thấp đã ghi
nhận; chưa xác lập cơ chế hỏng phần cứng; đề xuất giữ thẻ acer + cố định
100 MHz bằng DTB sao chép (`sdhci@700b0400` `max-clk-limit`), chờ người dùng duyệt; B1 tạm hoãn. **Người dùng duyệt
và đã áp dụng:** `/boot/dtb/..._sd100.dtb` + `extlinux.conf` `42333ea6…` (root thẻ acer, LABEL `goc204` dự phòng);
thẻ chạy HS 50 MHz 3,3 V (11,5–14 MB/s; đọc 15 GB sạch). Boot `5d3d12a9…` từ thẻ: v16 562/562, engine `c32690cd…`,
G0 sạch, **smoke 5 ảnh (lượt từng panic) đạt**, runner `cong_cu_v4` (tegrastats xả từng dòng + sync). Mọi số đo R63
từ đây ghi kèm cấu hình SD HS 50 MHz. Engine `c32690cd…` build trên board R62 trước (TRT cảnh báo khác thiết bị).
**31/10 chiều (trường, IP 10.11.61.127):** kế hoạch đã sửa theo DUYET_CO_SUA (`01_ke_hoach_r63.md`); người dùng
đồng ý build lại engine baseline. G0b lượt ổn định 300 frame ĐẠT (lượt đầu không hợp lệ do Chromium mở —
`g1/01_luot_on_dinh_300.md`). G1 xong (`g1/02_hai_engine_trtexec_bench.md`, `g1/03_baseline_pipeline.md`): engine
mới `9fae6b43…` (ONNX v16) nhanh hơn engine cũ ~0,5 ms, detection khớp 99,8 %; **~69 ms là GPU thực thi** (event
68,11 vs host 68,16 ms) ⇒ engine là nút thắt (~61 % của ~113,5 ms/frame); baseline pipeline 8/8 hợp lệ, **FPS trung vị
CV 8,81 / EKF 9,02**, hai tầng decode (11,4 vs 16,6 ms). **G3-C1 384×640:** gói v17-C (v16 + lấy (H,W) từ engine;
`review63/v17_c/`, test PC 6/6 + 5/5; đường 640 trên Nano MOT trùng byte v16; letterbox 384×640 trùng pixel đường PC
`predict(imgsz=640)` R53–R61). Engine C1 `a6912393…`: GPU exec 68,11 → 41,86 ms (−38,5 %) (`g3/01_…`). **So cặp
pipeline 16/16 hợp lệ: FPS +43 % (CV 9,27 → 13,24; EKF 9,23 → 13,20; cặp cùng tầng decode +42–44 %), trễ p90 −27 %**
(`g3/02_c1_so_cap_pipeline.md`; trình bày đúng: 5/8 cặp cùng tầng decode, xem `g3/04`). Cổng tốc độ ĐẠT.
**G4 dev (31/10, `review63/g4/10_ket_qua_g4.md`):** `det_cache --no-rect`, `eval_detector_5lop --hinh-dang` (hash cũ
giữ, test đạt); split tracking `BDDDEV-train` 139 video (Q3). C1 vs vuông 640, c25: **tracking đạt mọi cổng** (dev12
ΔHOTA +0,08, ΔIDF1 +0,31..+0,36; BDD dev +0,02 / +0,03..+0,07; IDSW dev12 +0,6..+1,5 %, BDD −1,3..−2,1 %); recall
detection giảm nhẹ nhất quán nhưng trong ngưỡng (BDD xe nhỏ −0,35 [−0,46; −0,25], dev12 che nặng −1,01 [−1,77; −0,40];
dev12 không đủ xe nhỏ); AP ước lượng điểm đạt (dev12 ΔmAP50 −0,29, Δ50–95 −0,19; BDD +0,01 / +0,03);
Nano↔PC 300 frame ĐẠT ở conf > 0,25 (C1 99,7 %, IoU 0,988). **Snapshot commit `b359515`** (31/10) chứa G0–G4 tới đây.
**Sau `b359515` (01/11, review GPT `runs_local/hoi_codex_R63_chot_c1_20261101/answer.json`):** Q1 dừng ở C1, không mở
C2/C3/G2; Q2 cổng đọc theo ước lượng điểm kèm CI hợp lệ ⇒ C1 đạt mọi cổng dev ⇒ **chốt C1** (giới hạn: tầng xe nhỏ dev12
thiếu GT, không tính đạt). Đã sửa theo `sua_truoc_khi_chot`: CI AP bản 1 SAI (cụm theo ảnh) → bản 2 theo video 5.000
vòng: dev12 ΔmAP50 −0,29 [−1,02; +0,24], Δ50–95 −0,19 [−0,76; +0,08]; BDD dev +0,01 [−0,18; +0,26], +0,03 [−0,07;
+0,11]; Nano↔PC A chạy lại sạch (trùng), conf > 0,05 chưa đạt 99 %; `--hinh-dang` ở hai wrapper báo cáo cuối,
`test_v17_c` 7/7; đính chính provenance ONNX/swap/trình tự C2–C4/cách trình bày tốc độ (`g3/04_dinh_chinh_provenance.md`);
bỏ tự miễn cổng ở `g3_duong_toc_do.sh`. Đường tốc độ C2–C4 (khảo sát ngoài trình tự): GPU exec 37,2 / 27,2 / 21,0 ms.
**Báo cáo cuối XONG (02/11, `review63/bao_cao_cuoi/10_bao_cao_cuoi.md`, CHỈ báo cáo):** C1 − 640×640 ở c25 — test40
ΔHOTA +0,09 (CV) / +0,20 (EKF), IDSW −7 %, ΔmAP50 (car, bus) −0,17, recall xe nhỏ −0,60, che nặng −0,81; BDD val
ΔHOTA −0,02 / −0,02, IDSW −2 %, ΔmAP50 −0,01, recall xe nhỏ −0,18 (BDD val không có nhãn che). Nhánh C1 khớp các lượt
lịch sử ở độ chính xác báo cáo (tracking test40 chênh ≤ 6e-9, BDD val trùng tuyệt đối; AP test40 chênh ~1e-8 so 10/10;
AP BDD val = R43). GPT review 02/11: DAT_CO_SUA (đã sửa làm tròn/câu chữ/g6; `hoi_codex_R64_review_smoke_20261102`). CI AP test40/BDD val chưa tính. **G6 (`review63/g6/01_ket_luan_r63.md`):**
đề xuất v17 = C1; launcher HDMI **giữ v16** (người dùng 01/11); FPS tăng chỉ đúng cho Nano này, thiết bị khác đo lại.

**Khảo sát sau mốc 6 — 29/10/2026:** theo yêu cầu người dùng, đã rà mã từ đầu vào đến kết quả một video và
tính lại số công đoạn từ raw R62. Chưa chạy benchmark MP4 mới hoặc sửa mã thực thi. Hai ứng viên cụ thể là bỏ
bản sao float32 dư sau TensorRT và tái dùng ma trận điểm lớp đã lọc; thời gian tiết kiệm chưa được đo. Giải mã
MP4, khởi tạo và xuất video có khung/ID cần được đo riêng theo đầu ra người dùng cần. Backend NumPy trên Nano
hiện chỉ đăng ký CV/EKF; UKF/PF/SORT cần tích hợp và kiểm tương đương trước khi có benchmark Nano cho chúng.
Chi tiết, số đo và thứ tự đề xuất: `handoff/logs/review62/32_khao_sat_toc_do_video.md`. Đây là khảo sát để lập kế
hoạch tiếp, không thay kết luận/ngưỡng nghiệm thu R62; launcher vẫn dùng v16.

**Lượt đo 29/10/2026: MỐC 6 ĐÃ ĐÓNG VỚI KẾT QUẢ ÂM.** Đã đo baseline, kiểm ứng viên
chuyển tensor và hoàn tất phép so cặp. Ứng viên không đạt ngưỡng tăng FPS đã chốt nên giữ gói v16 làm bản dùng;
không sửa engine, tracker, metric hay cấu hình thí nghiệm. Baseline lịch sử: pipeline
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
Đã hoàn tất runner 16 lượt paired300 (4 cặp CV, 4 cặp EKF, bật hiển thị), exit 0, cùng boot;
mọi cặp có detection/MOT/lớp trùng byte và từng lượt đủ 300 frame, không rơi frame hiển thị.
Trung vị FPS tăng **3,917% CV** và **4,277% EKF**, đều dưới ngưỡng chốt 5%; p90 trễ giảm lần lượt
3,568% và 3,877%, swap peak cùng 11 MB, swap không tăng trong từng lượt. Báo cáo số thô
`handoff/logs/review62/28_so_cap_300_lan01.json`. **Ứng viên chưa đạt tiêu chí tăng tốc, không triển khai launcher**.
Không suy ra throttling từ xung DVFS; chưa có đánh giá riêng về throttling, nhưng ngưỡng FPS đã đủ để kết luận
chưa đạt. Ứng viên dài `candidate1470_lan01` CV/EKF đều 1470/1470 frame, live/replay byte đạt,
runner exit 0, cùng boot; FPS CV 8,587, EKF 9,576 nhưng decode khác tầng nên không dùng số dài để kết luận tăng tốc.
Detection/MOT/lớp của hai lượt dài trùng byte với baseline dài cùng tracker (`29_so_dai_1470_lan01.json`).
Đã kiểm shortcut desktop vẫn trỏ `/home/jetson/nano_run/v16/nano_transfer`; launcher v16 `--kiem-tra`
exit 0, detector Python/TensorRT/OpenCV/NumPy OK. Không chuyển launcher sang gói ứng viên.
Kết luận/bằng chứng cuối R62: `handoff/logs/review62/30_ket_luan_moc6.md`,
`28_so_cap_300_lan01.json`, `29_so_dai_1470_lan01.json`; raw nhỏ đã sao lưu tar SHA256
`81ccef2b53934b1578feca4c338bbde73fdf490ef9b5e7a2eb7731f0e284e637`.
Không còn việc đo trong phạm vi mốc 6. Báo cáo và bằng chứng cuối đã đẩy lên GitHub nhánh
`phase2-mixed-traffic` tại commit `b7a50ab19c52af38e937695b5fe43916bac6f2a0` (đã đối chiếu remote).
Phải đo lại baseline v16 trên Jetson hiện tại để so tối ưu; không quy chênh lệch với FPS máy cũ cho sửa mã.
Chỉ ghi bản sao và kết quả R62; chưa sửa boot/mount/cài gói hoặc mã thuật toán.
Gói v16 `handoff/nano_transfer_latest.zip` còn nguyên SHA256 `302f6af73d97aea009066416aaaa202db81311e60121676a75dacbe204e9cbf1`;
engine đã sao lưu về `handoff/archive/nano_engine_r62_c32690cd.engine`, hash nguyên bản khớp. Bằng chứng
và bước tiếp: `handoff/logs/review62/05_tien_kiem_va_khoi_phuc.md`.
**Tiếp tục tự động:** lịch gắn chat `ti-p-t-c-m-c-6-jetson-nano` đã giúp nối việc khi hạn mức cho phép;
đã **xóa** lịch bằng automation_update sau khi đóng mốc 6, không còn lượt chạy nền. Không mở orchestrator CLI riêng,
không dùng reset credits/mua credits.

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
