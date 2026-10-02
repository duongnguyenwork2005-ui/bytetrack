# Phu thuoc KHONG nam trong goi delta

Cac file duoi day can de hieu / chay lai nhung **khong** duoc dong goi trong vong nay.
Cot *vi tri trong snapshot goc* la duong dan tuong doi tu goc repository — cung la duong dan
trong goi day du `handoff/archive/review_delta_0c0a07a.zip` (snapshot base `0c0a07a`).

**Khong gia dinh nguoi review con giu ban goc.** Neu thieu file nao trong bang nay, bao lai —
se bo sung dung nhung file do trong goi delta ke tiep, khong gui lai ca repository.

| file | vi tri trong snapshot goc | sha256 tai base | sha256 tai target | ly do khong gom |
|:--|:--|:--|:--|:--|
| `CODEX_REVIEW_0c0a07a.md` | `handoff/review_evidence/CODEX_0c0a07a/CODEX_REVIEW_0c0a07a.md` | `-` | `b3330e631a02` | trung noi dung voi files/handoff/CODEX_REVIEW_0c0a07a.md (cung sha256); thay doi van nam trong patch |
| `package_check_reproducible.log` | `handoff/review_evidence/CODEX_0c0a07a/evidence/package_check_reproducible.log` | `-` | `3482774abb81` | trung noi dung voi files/handoff/review_evidence/CODEX_0c0a07a/evidence/package_check_reproducible.json (cung sha256); thay doi van nam trong patch |
| `07_tom_tat_pilot.json` | `handoff/review_evidence/CODEX_0c0a07a/tree/handoff/logs/review37/07_tom_tat_pilot.json` | `-` | `634e690d6e2e` | trung noi dung voi files/handoff/review_evidence/CODEX_0c0a07a/evidence/summary_recomputed.json (cung sha256); thay doi van nam trong patch |

## File thuoc snapshot base nhung CHUA TUNG duoc ban giao (khai bao giu nguyen qua cac lan doi base)

Cac file nay nam trong cay base nhung da bi loai co chu y o goi truoc; nguoi review **khong** co chung.
Khong dung chung de tai dung ca repository. sha256 tinh tai base luc khai bao.

| file | bytes | sha256 | khai bao tu base | ly do |
|:--|--:|:--|:--|:--|
| `handoff/logs/review16/nano_out/nano_trt_MVI_39031_f0001-0300/MVI_39031.npz` | 13,469 | `d18dc4ef4d23a6754b363991df3231a594b6fa8b0a70f3313c80510051e7a412` | `45e086a` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review21/nano_live_v7/data/interim/det_cache/live_trt_cv_20260928120942/MVI_39031.npz` | 37,622 | `4a8e1bd7f4c79a17b7f2415edb4a2e6c6396550294220f3d08b72f1a27550df4` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review21/nano_live_v7/data/interim/det_cache/live_trt_dung_20260928120942/MVI_39031.npz` | 6,198 | `8e979a90944534c3fc56fb8c8e6548fecfbe418299cb19b201e16e25ce0219fa` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review21/nano_live_v7/data/interim/det_cache/live_trt_ekf_20260928120942/MVI_39031.npz` | 37,622 | `4a8e1bd7f4c79a17b7f2415edb4a2e6c6396550294220f3d08b72f1a27550df4` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review21/nano_live_v7/data/interim/det_cache/live_trt_regions_20260928120942/MVI_39031.npz` | 13,508 | `96b00d4275f5f3acb2550838310ac2510ca603fa43465551f8515248494f2a3c` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review22/nano_live_v8/data/interim/det_cache/live_trt_cv_20260928133338/MVI_39031.npz` | 37,622 | `4a8e1bd7f4c79a17b7f2415edb4a2e6c6396550294220f3d08b72f1a27550df4` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review22/nano_live_v8/data/interim/det_cache/live_trt_dung_20260928133338/MVI_39031.npz` | 6,198 | `8e979a90944534c3fc56fb8c8e6548fecfbe418299cb19b201e16e25ce0219fa` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review22/nano_live_v8/data/interim/det_cache/live_trt_ekf_20260928133338/MVI_39031.npz` | 37,622 | `4a8e1bd7f4c79a17b7f2415edb4a2e6c6396550294220f3d08b72f1a27550df4` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review22/nano_live_v8/data/interim/det_cache/live_trt_regions_20260928133338/MVI_39031.npz` | 13,508 | `96b00d4275f5f3acb2550838310ac2510ca603fa43465551f8515248494f2a3c` | `c4b39fe` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review23/nano_dai_v23/results_nano/dai_v23/luot01_cv_cache/MVI_39031.npz` | 222,099 | `8eb7ab80c4e07b0ff692d322dae240e1ab7a6231e569338bce7a12dd4b931ffb` | `f86732b` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review23/nano_dai_v23/results_nano/dai_v23/luot02_ekf_cache/MVI_39031.npz` | 222,099 | `8eb7ab80c4e07b0ff692d322dae240e1ab7a6231e569338bce7a12dd4b931ffb` | `f86732b` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review23/nano_dai_v23/results_nano/dai_v23/luot03_ekf_cache/MVI_39031.npz` | 222,099 | `8eb7ab80c4e07b0ff692d322dae240e1ab7a6231e569338bce7a12dd4b931ffb` | `f86732b` | du lieu nhi phan hoac patch toan bo lich su |
| `handoff/logs/review23/nano_dai_v23/results_nano/dai_v23/luot04_cv_cache/MVI_39031.npz` | 222,099 | `8eb7ab80c4e07b0ff692d322dae240e1ab7a6231e569338bce7a12dd4b931ffb` | `f86732b` | du lieu nhi phan hoac patch toan bo lich su |

## Dau vao lon giu tren may (khong bao gio dong goi)

Bang thong ke + hash trong `data/large_inputs_hashes.csv` cua goi nay. Du lieu goc (anh UA-DETRAC, weights, cache `.npz`) nam tren may tac gia; xem `handoff/large_inputs.md` trong goi day du de biet cach nhan dien va lenh dung chung.

Target commit: `b83dc12`. Moi hash trong goi tinh bang sha256.
