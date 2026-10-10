# Vòng 47 — Mốc A: fingerprint dữ liệu UA-DETRAC test

**Phạm vi**: kế hoạch R47 bước `R47A_FINGERPRINT_DETRAC_TEST` (người dùng duyệt 2026-10-10), theo dự thảo
`handoff/logs/review45/06_du_thao_giao_thuc_detrac.md` mục 7 và D4 đã chốt (`handoff/REVIEW46_CHI_TIET.md` mục 11.4).
Chỉ dựng fingerprint. **Chưa** viết adapter (mốc B), chưa khóa kế hoạch, chưa dự đoán; không GPU, không SSH, không commit.
Base: `e4a0311954278c85c6732f0a53fd2d3cee42ed1b` (nhánh `phase2-mixed-traffic`), cây làm việc có thay đổi chưa commit
(không thuộc bước này).

## 1. File đã viết

| File | Vai trò |
|:--|:--|
| `src/fingerprint_detrac_test.py` (mới, sha256 `37b62da99b77880a27aa234a887158e51498b6822921b430c1366d271f621255`) | script chỉ đọc dữ liệu; dựng 3 file đầu ra; chế độ chạy thử (`--video`), `--khong-ghi`, `--kiem-lai` |
| `data/interim/hash_tap_danh_gia/detrac_test_sha256.csv` | bảng hash nội dung **mọi** ảnh: `rel,size,mtime_ns,sha256,w,h` |
| `data/interim/hash_tap_danh_gia/detrac_test_loai_D4.csv` | 173 ảnh loại theo D4: `rel,video,frame` |
| `data/interim/hash_tap_danh_gia/detrac_test_fingerprint.json` | tóm tắt máy đọc được (khối `van_tay` tất định, `dau_ra`, `doi_chieu`, `lan_chay`) |
| `handoff/logs/review47/01_tu_kiem_d4_sai.txt` | đầu vào cho ca tự kiểm exit 3 (danh sách D4 sai cố ý, 3 dòng); không phải dữ liệu |
| `handoff/logs/review47/01_fingerprint_detrac_test.md` | báo cáo này |

Thư mục `data/interim/hash_tap_danh_gia/` có `.gitignore` = `*`, nên git **không** theo dõi 3 file đầu ra. Chúng được ghim
bằng các hash ở mục 4.

## 2. Script làm gì (đọc mã: `src/fingerprint_detrac_test.py`)

- **Chỉ đọc**: manifest, XML, ảnh đều mở `'rb'`. `--out` nằm trong `data/extracted`, `data/raw` hay `results` → từ chối,
  exit 2. Tên file ra cố định nên không thể ghi đè `bdd100k_mot_val_sha256.csv`. Chỉ dùng thư viện chuẩn + PIL
  (12.3.0); không pandas / pyarrow.
- **Mỗi ảnh đọc một lần**: `size`, sha256 **toàn bộ nội dung** và w/h đều lấy từ cùng một chuỗi byte. w/h đọc từ header
  bằng `PIL.Image.open` (không giải mã pixel), cùng thư viện với `eval_detector_5lop.kich_thuoc_anh`. Không có bước bỏ qua
  theo mtime: script không đọc bảng cũ khi dựng bảng. `mtime_ns` chỉ được ghi lại để giữ định dạng bảng BDD.
- **Thứ tự dòng CSV** = video theo thứ tự trong manifest, rồi frame tăng dần. Đây **chính là** danh sách ảnh
  `video/frame` có thứ tự.
- **Kiểm cứng** (sai → exit 2, **không ghi** file nào):
  - manifest hợp lệ, không trùng video;
  - tập `*.xml` trong `DETRAC-Test-Annotations-XML` == tập video của manifest;
  - mỗi video có thư mục ảnh; mọi mục trong thư mục là file `img%05d.jpg`, đánh số liên tục từ 1;
  - mọi `<frame num>` có ảnh; PIL đọc được header mọi ảnh;
  - tổng số ảnh == `--so-anh-ky-vong` (nếu khai báo).
- **Lệch kỳ vọng** (vẫn ghi, exit 3, liệt kê từng chỗ lệch trong JSON):
  - w/h ≠ 960×540; định dạng không phải JPEG; `<sequence name>` ≠ tên file;
  - danh sách D4 ≠ `anh_khong_frame.csv` vòng 45;
  - 12 trường của `theo_video.csv` vòng 45 khác (gồm `xml_sha256`).
- `--kiem-lai`: tính lại toàn bộ rồi so với 3 file đã ghi, không ghi gì; khác → exit 4. Script so:
  - các cột `rel,size,sha256,w,h` và thứ tự từng dòng (`mtime_ns` chỉ đếm, không tính là lệch);
  - file D4 theo từng byte;
  - khối `van_tay`;
  - sha256 của file trên đĩa với giá trị đã ghi trong JSON.

## 3. Lần chạy chính thức

```
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/fingerprint_detrac_test.py --so-anh-ky-vong 56340
```

- Exit **0**. Công cụ Bash không báo mã khác 0, và script in `"exit": 0`.
- Bắt đầu `2026-10-10T02:36:34+07:00`, kết thúc `02:37:25+07:00`, **51,1 s** (foreground).
- Script sha256 `37b62da9…` (bản cuối, ghi ở `lan_chay.script_sha256`). Python 3.11.9, PIL 12.3.0.

| Đại lượng | Giá trị | So với vòng 45 |
|:--|--:|:--|
| Video | 40 | = manifest; tập XML == manifest |
| Ảnh trên đĩa (= số dòng CSV) | **56.340** | khớp |
| `<frame>` trong XML | **56.167** | khớp |
| Ảnh loại theo D4 (không có `<frame>`) | **173** | khớp 173/173, cùng thứ tự |
| Ảnh sau D4 (tập đánh giá) | 56.167 | — |
| Ảnh lệch 960×540 (kiểm **mọi** ảnh) | **0** (960×540: 56.340) | vòng 45 chỉ đọc ảnh đầu / cuối |
| Định dạng | JPEG: 56.340 | — |
| Tổng byte ảnh | 4.234.039.839 | — |
| Box trong XML gốc | 675.774 (car 548.555, bus 71.785, van 38.519, others 16.915) | khớp |
| `ignored_region` | 163 | khớp |
| `frame` không có ảnh | 0 | khớp |

Số box là số đếm thô trên XML gốc: chưa cắt khung, chưa áp D1–D5. Đây không phải metric.

D4 theo video, đoạn frame:

| Video | Số ảnh | Đoạn |
|:--|--:|:--|
| MVI_39051 | 69 | 992–1060 |
| MVI_39211 | 94 | 477–570 |
| MVI_39271 | 4 | 77–80 |
| MVI_40743 | 1 | 1630 |
| MVI_40763 | 4 | 1742–1745 |
| MVI_40793 | 1 | 1960 |

Đối chiếu vòng 45 (`doi_chieu` trong JSON):
- `anh_khong_frame.csv` (sha256 `ea0a107b37a8452a07b753f68bd0b11be1a9a414616774bb76a7818581a959e4`):
  - chung 173, cùng thứ tự;
  - không có dòng chỉ có ở một bên;
  - không có cột `anh` sai đường dẫn, không có dòng trùng → `khop: true`.
- `theo_video.csv` (sha256 `d070dbe0ce67f469831fcfbd7187d29bdf7149877971e7935fc611a979dad9f8`): 40 video × 12 trường, 0 lệch,
  `khop: true`. 12 trường gồm `xml_sha256`, số frame, `num_min` / `num_max`, số ảnh, `anh_min` / `anh_max`, số ảnh và đoạn
  không có frame, `frame_khong_anh`, số box, số vùng bỏ qua. Vì vậy sha256 của 40 XML **trùng** giá trị đo ở vòng 45.

## 4. Hash

**Manifest** `results/phase2/manifests/detrac_test.json`:
`ff2b4b9c658aced54567247c16b5a0fc43ae1e9836af70e7ea656fce3077f47c` (931 byte, file trên đĩa).

**File đầu ra**

| File | sha256 |
|:--|:--|
| `detrac_test_sha256.csv` (56.340 dòng + header) | `3cea34881d90031a5ed2161a774154e87691228da9b25b1ac0b07dbe8717b166` |
| `detrac_test_loai_D4.csv` (173 dòng + header) | `4d7312530505651b92da414b2dcf82c4ec74486dc408fa37dc9f927d85723551` |
| `detrac_test_fingerprint.json` | `dd0f2a0e9dcaf11f26fe397ff4c4eb7ef8a8ecfd248d4682aa652b3af156343a` |

sha256 của JSON lấy từ hai nguồn trùng nhau: `tom_tat_da_ghi_sha256` lúc ghi và `kiem_lai.sha256_file_tren_dia` lúc tái
kiểm. JSON chứa thời điểm chạy, nên **mỗi lần ghi lại sẽ đổi sha256 của JSON**. CSV chứa `mtime_ns`, nên sha256 file CSV
đổi nếu mtime ảnh đổi, kể cả khi nội dung giữ nguyên. Để khóa nội dung, nên dựa vào các hash tổng hợp dưới đây.

**Hash tổng hợp** (khối `van_tay.tong_hop`; cách tính ghi trong `cach_tinh`)

| Trường | Cách tính | sha256 |
|:--|:--|:--|
| `noi_dung_anh_sha256_tong_hop` | `sha256_ds((rel, sha256))`, giống hệt `danh_gia_detector_bdd_val.sha256_ds` | `4e247a808e038a2509ce5a741e3895c62d55436b0d55f3f511ba759fb76d8f6c` |
| `danh_sach_anh_sha256` | `sha256_ds((rel, ''))`, không phụ thuộc thứ tự (như adapter BDD) | `3707f75585ce2bfcfc03e511c552c60d275ae37b8783e9bb32574ec438423f70` |
| `danh_sach_anh_co_thu_tu_sha256` | sha256 các dòng `'<rel>\n'` **đúng thứ tự CSV** | `fc1f8edc1d4f138c3a1088af35ce84577fe686ab75080026aed9f24acff6851c` |
| `loai_D4.danh_sach_co_thu_tu_sha256` | như trên, trên 173 ảnh D4 | `67d3b935cef6297fbec31003b665bc8511e0026242002f5e716b012b08474a4e` |
| `tap_danh_gia_sau_D4.danh_sach_anh_co_thu_tu_sha256` | 56.167 ảnh = dòng CSV bỏ D4, giữ thứ tự | `46204978758016cb16627b2035d34a40c9b31c5eb477db6dd442afab3eba92d7` |
| `tap_danh_gia_sau_D4.noi_dung_anh_sha256_tong_hop` | `sha256_ds` trên 56.167 ảnh đó | `e4fff621ae775ad7d571d67e23c5f75db100434b4905c1e405fea24dec696cb6` |

`sha256_ds` được viết lại trong script (4 dòng, cùng mã) thay vì import adapter BDD. Lý do: import adapter BDD kéo theo
`bdd_yolo_export` / `eval_detector_5lop` / numpy.

**40 XML** (`data/extracted/DETRAC-Test-Annotations-XML/<video>.xml`; cũng nằm ở `van_tay.xml[*].sha256`)

| Video | sha256 | Video | sha256 |
|:--|:--|:--|:--|
| MVI_39031 | `95104c003013c9d9ec5c3a9ee033275a4ea1c6054cfa84cbf7dfd761afe0f0e6` | MVI_40763 | `2953c4fa771beb29bb163daf6e7d57bd9e364f5eaf6d1819e63fd9cba0398dea` |
| MVI_39051 | `c379907766c26646c08eb9716e28df055a64d7ab881aa985019ae9c0f5ccc863` | MVI_40771 | `8640fec3a2d73a0a1986b93064a5741f6a02d5495fb8c52c44f493ad5a1822ea` |
| MVI_39211 | `86e17975608eaa2d260d078ad84a30f601759f6c5572ebca2c1fc4a417c46b55` | MVI_40772 | `5eb7f2f1b261b5dad226b2aae59e984a45fad86ec4088791c51ee21156945b9f` |
| MVI_39271 | `c3259c9f1cd23369e82fe53aee3dec735d5e0d72ecefa1f815bd09fc0ae674a7` | MVI_40773 | `10454921ec353499ef85ff0b83f8e67b9a692809d0d5be8eaeec4db45dc5f175` |
| MVI_39311 | `a5c4052c3ae287ca580a8539e489cb595f98deac855927cf4e37827c5fdf2b65` | MVI_40774 | `dc676601b1abd60dced781ee4ec85f730ee73c9d12c9f0b11d2b36580cbf133f` |
| MVI_39361 | `4e07b6e5eabd98531a1ca9bf24c2d58e9354d74de73bd835403eeb83fcf99dd1` | MVI_40775 | `db6bb521552b2d665f5f0f910cdcdff46df78ecb643343418caa6c40af1719e5` |
| MVI_39371 | `bd435287f69bb1eca0b9365df088122c41e1bcdb9add7099a667fc227417f49e` | MVI_40792 | `141479733d4b252b0d8cf522a6e028300a7cbe8324e6905bab0e87bf23332d77` |
| MVI_39401 | `ff98077a0e0c1788a6b298fb99a68ab81c9da57dc9001c086309e19f553cac17` | MVI_40793 | `4722cf82de9c036745a7b4ab3e727a8a068a4f1a1e3d9238a44158c97b2a5242` |
| MVI_39501 | `d2e519ea3e6990b7eef5ff7fbba50803c5914a0b23d6d113baa028c1a2f0e6e0` | MVI_40851 | `4f75d5479fb7186d3f12fda7535277720b5f9a872ac942159f7cc37836c76abf` |
| MVI_39511 | `e82b1e8d02b1aa12a5fcf2937c68506d3148987200a6e2238baa3a71ebe40dd0` | MVI_40852 | `6b00c762fb46d9638913c66fcfd2250be01e69f786c3d5b18a01ef866e131d69` |
| MVI_40701 | `d2f317e05ff09ebc5a16232c4da78755f5d41c85955bcd57c416caa39e3a0d70` | MVI_40853 | `5aa16a5fc49be193fa91622f4131e7052f4f78e0cc0cb8d58a3a7e2dea24bf11` |
| MVI_40711 | `3fa1b81df0f5f9f997a58225830c64fcd39180d57b626e206c27b375ac6ec8e8` | MVI_40854 | `0ed968559e70193deef314770409fcd298fad3863369711bcdaa39ef1f7db2ea` |
| MVI_40712 | `58fbb87ed33db0af1ba034f3eca54951c507b6cbc4ea8dc3724bd1f4a52fc960` | MVI_40855 | `577351bec22e368dda12d00a424d2f1ef6393f8dbc6bd3708b2601bbb8a5e599` |
| MVI_40714 | `5d1baaa2d8019d9092e32b568a97c1bb6982018001d862ae218f6621a0c27c4a` | MVI_40863 | `6de301580ebf482c7af4c922c6445edcd9a04469064f0d70ef0f87f44df3596b` |
| MVI_40742 | `0364d4428215e98a59264a540d2b8f7218a3b0c488bc271d093fa53d439a0610` | MVI_40864 | `74de32395e1e23975932173e18b5b1cb2936bdc5d2da0e51a5df66e8972bffbb` |
| MVI_40743 | `c8a8ee2411df9feaa1482bb3f9b39e9be52a0e7184a77e321a3b4a7c0a3e11c6` | MVI_40891 | `a19547a2e65f977de244f8917db379bfb1f9b9218b94b1cd6359dbf00865a394` |
| MVI_40761 | `4271288834157dd274872015a03eae52d8d5311842e249a289ef0644af0e3928` | MVI_40892 | `927b7b34f944e621e2b91f5ec7a777a5baece055ccc9007482de7dd99c5bb60e` |
| MVI_40762 | `0dfc15f8536bc4d3000ba36cda833854364851a390bd19c237fbbf5d01a8a224` | MVI_40901 | `3d102d914f35c0c6b7df978393456d22be1cce52d8507f313eb2add9e5861251` |
| | | MVI_40902 | `9bcf41176876996386769a2873e391449335a8f7250a5d950b093bd316b1b5b9` |
| | | MVI_40903 | `d3f724f9f1a79cad7d3f04f65546aa38dbd8c0d0839c215e018968eca4f28e7a` |
| | | MVI_40904 | `8f0491025240215f5045107b9cb664b0e9c9fdf09253182dffab4017412bc72b` |
| | | MVI_40905 | `7021fa5320b46034d5b928571b69ccdb3401c57b3337ae7f2eab162c2b26b27c` |

## 5. Tự kiểm (Tier 1) — trên bản script cuối `37b62da9…`

Mọi lệnh có dạng `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/fingerprint_detrac_test.py <tham số>`, chạy
foreground. Exit khác 0 do công cụ Bash báo (`Exit code N`); exit 0 thì công cụ không báo gì và script in `"exit": 0`.

| # | Tham số | Kỳ vọng | Kết quả |
|--:|:--|:--|:--|
| 0 | `-m py_compile src/fingerprint_detrac_test.py` (lệnh riêng) | không lỗi | không lỗi |
| 1 | `--so-anh-ky-vong 56340` (chính thức, mục 3) | 0, ghi 3 file | **0**, 51,1 s |
| 2 | sửa cố ý 1 ký tự sha256 dòng 2 của CSV (`14adc898…` → `04adc898…`, công cụ Edit), rồi `--so-anh-ky-vong 56340 --kiem-lai` | 4 | **4**: "sha256 file bang hash khac gia tri ghi trong JSON"; "1 dong khac noi dung / thu tu, vd ['MVI_39031/img00001.jpg']"; sha256 file khi đó `1e448f36…` |
| 3 | hoàn nguyên đúng ký tự đó, rồi `--so-anh-ky-vong 56340 --kiem-lai` | 0 | **0**, 51,8 s: `khop: true`, 56.340 dòng, 0 dòng khác mtime, sha256 3 file trên đĩa = `3cea3488…` / `4d731253…` / `dd0f2a0e…` (hoàn nguyên đúng từng byte) |
| 4 | `--video MVI_99999` | 2 | **2**: "--video ngoai manifest" |
| 5 | `--out results/phase2/fingerprint_thu_khong_duoc_tao` | 2, không tạo gì | **2**: "nam trong cay chi doc results"; Glob `results/phase2/fingerprint_thu*` không có file |
| 6 | `--manifest results/phase2/manifests/detrac_dev.json --khong-ghi` | 2 | **2**: "tap XML ... (40) khac manifest (12)" |
| 7 | `--video MVI_39051 --doi-chieu-theo-video handoff/logs/review45/phan_tich/anh_khong_frame.csv` | 2 | **2**: "file doi chieu thieu cot [...]" |
| 8 | `--video MVI_39051 --doi-chieu-d4 handoff/logs/review47/01_tu_kiem_d4_sai.txt` | 3, liệt kê từng chỗ lệch, không ghi | **3**: `chi_co_trong_file` = [img00991], `chi_co_trong_D4_moi` = 67 ảnh img00994–img01060, `cot_anh_khac_duong_dan` = [img00999], `da_ghi: false` |
| 9 | `--so-anh-ky-vong 56341 --khong-ghi` | 2 | **2**: "tong so anh tren dia 56340 khac --so-anh-ky-vong 56341" |

**Bản script trước đó.** Trước bản cuối có ba bản khác. Thay đổi giữa các bản chỉ gồm:
- kiểm cột của file đối chiếu (thiếu cột → exit 2 thay vì `KeyError`);
- các trường in ra: `doi_chieu_lech_chi_tiet`, `sha256_file_tren_dia`, `tom_tat_da_ghi_sha256`.

Các lần chạy cũ:
- chạy thử `--video MVI_39031 MVI_39051` (exit 0, 3,8 s);
- 3 lần ghi đủ (exit 0; 56,0 / 49,8 / 50,5 s), 3 lần `--kiem-lai` (0 / 4 sau khi sửa cố ý / 0);
- các ca 4–9 lặp lại trên bản thứ ba, cùng kết quả.

Mọi lần ghi đủ, ở mọi bản, cho cùng sha256 CSV, D4 và hash tổng hợp. Chỉ sha256 của JSON đổi, do thời điểm chạy và
`script_sha256`.

**Không thay đổi file được bảo vệ**:
- `git status --short -- src handoff/logs/review45 handoff/logs/review47 results/phase2/manifests data` chỉ thêm
  `src/fingerprint_detrac_test.py` và `handoff/logs/review47/` so với trạng thái đầu phiên. `06_du_thao_giao_thuc_detrac.md`
  và `src/demo/*` đã ở trạng thái M/?? từ trước.
- Manifest không đổi.
- `data/extracted` bị gitignore nên git không chứng minh được gì ở đó. Bảo đảm chỉ đến từ mã: mọi lần mở file đều `'rb'`,
  và `--out` bị chặn.

## 6. Cách tái kiểm

1. Tái kiểm toàn bộ, không ghi, khoảng 1 phút:
   ```
   C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/fingerprint_detrac_test.py --so-anh-ky-vong 56340 --kiem-lai
   ```
   Lệnh băm lại mọi ảnh, đọc lại w/h, XML và manifest, rồi so với 3 file đã ghi (mục 2). Exit 0 = khớp; 4 = file lệch dữ
   liệu; 2 / 3 như mục 2. Mục `kiem_lai.sha256_file_tren_dia` in sha256 của cả 3 file, dùng để so với mục 4.
2. Chạy thử một phần, không ghi: `--video <tên video> ...`.
3. Ngoài script: so sha256 3 file với mục 4 bằng công cụ bất kỳ (ví dụ `certutil -hashfile <file> SHA256`). **Chưa chạy**
   ở vòng này, vì không nằm trong các lệnh được phép.
4. Mốc B (adapter) sẽ băm lại ảnh và đối chiếu với `detrac_test_sha256.csv` trước khi dự đoán. Script có sẵn
   `doc_bang_hash`, `sha256_ds`, `sha256_thu_tu` để import.

## 7. Giới hạn

- Phép kiểm w/h đọc header bằng PIL, không giải mã pixel. Thân JPEG hỏng sẽ không bị phát hiện qua w/h, nhưng nội dung byte
  đã được ghim bằng sha256.
- CSV ghi `\r\n` như bảng BDD (cùng `csv.DictWriter` với `newline=""`; đối chiếu bằng đọc mã `sha256_tap_bdd`). **Chưa** so
  từng byte với file BDD.
- 3 file đầu ra không được git theo dõi (`.gitignore` = `*`). Nếu bị xóa hoặc sửa, cần dựng lại bằng lệnh ở mục 3 rồi
  kiểm theo mục 6. Nội dung dựng lại sẽ trùng nếu dữ liệu không đổi; riêng JSON sẽ khác do thời điểm chạy.
- Ở ca 2–3, CSV chính thức bị sửa tạm 1 ký tự rồi hoàn nguyên. Ca 3 xác nhận file trở lại đúng sha256 đã ghi trong JSON.
- Chưa có adapter, chưa khóa kế hoạch, chưa dự đoán: đó là mốc B và các bước sau.
