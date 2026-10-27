# R52A — Dataset trộn 5 lớp `data/yolo_mixed5_v1` (BDD100K + UA-DETRAC)

Ngày 2026-10-16. Người triển khai: Claude `claude-opus-5-5` / effort high. Chưa qua review Codex.
Căn cứ: quyết định người dùng ngày 16/10/2026 (plan R52 đã duyệt, bước `R52A_DATASET_MIXED5`). Không huấn luyện,
không suy luận detector, không tracking. Không sửa mã hay cấu hình cũ. Không ghi vào `data/yolo_bdd5_v1`,
`data/extracted`, `data/raw`. Không có thao tác Git nào làm thay đổi kho.

## 1. Sản phẩm

| Đường dẫn | Nội dung |
|---|---|
| `configs/phase2/detector/mixed5_v1.yaml` | Cấu hình quy tắc (sha256 `4e22333e…125b`) |
| `configs/phase2/detector/splits/mixed5_v1.json` | Split đã khóa (sha256 `0acc018b…664b`, `sha256_danh_sach` `c1b38e64…0597`) |
| `src/mixed5/mixed5_lib.py` | Logic: đọc XML, chọn frame, ánh xạ lớp, kiểm rò rỉ, xuất, ghi manifest |
| `src/mixed5/xuat_mixed5.py` | CLI theo bước (`chia`, `ke-hoach`, `xuat-bdd`, `xuat-detrac`, `manifest`, `kiem`, `thong-ke`, `overlay`) |
| `src/mixed5/test_mixed5.py` | 14 test |
| `data/yolo_mixed5_v1/` | Dataset (Git bỏ qua, `.gitignore` = `*` trừ chính nó) |
| `results/phase2/mixed5/kiem_hop_dong.json` | Kết quả hợp đồng và các kiểm bổ sung |
| `results/phase2/mixed5/thong_ke.{json,md}` | Thống kê |
| `results/phase2/mixed5/overlay/` | 5 ảnh overlay (Git bỏ qua) |
| `results/phase2/mixed5/xuat_tam/` | Trạng thái từng bước (kế hoạch, từng video UA-DETRAC), phục vụ chạy tiếp |

Mã mới tự thêm `src/` vào `sys.path` và gọi lại `bdd_yolo_export` (`xuat_mot_anh`, `_hop_le_box`,
`kiem_hop_dong_dataset`, `sha256_file`). Không sửa module đó.

## 2. Quy tắc

**Split** (`mixed5_v1.json`):
- BDD: lấy nguyên `bdd5_train_v1.json` (train-fit 1.261 / dev 139 video), không chia lại. Split mới ghi tham chiếu sha256
  file `b8a2848e…6cdb` và `sha256_danh_sach` `a59a8d06…a723`; hai giá trị này trùng `tom_tat.json` của `bdd5_v1`.
  Mỗi lần đọc split, `doc_split_mixed5` kiểm lại: file BDD còn đúng sha256 và `train_fit/dev = BDD ∪ UA-DETRAC`.
- UA-DETRAC: dev = đúng 12 video của `results/phase2/manifests/detrac_dev.json` (sha256 `95c7bc70…dd30`). Train là
  48 video còn lại trong 60 file XML của `data/extracted/DETRAC-Train-Annotations-XML`; vân tay 60 XML là
  `a3cd9c51…eff42`.
- Tổng: train-fit 1.309 video (1.261 + 48), dev 151 video (139 + 12).

**BDD (dùng nguyên)**: ảnh, nhãn và ignore lấy từ `data/yolo_bdd5_v1` theo đúng đường dẫn tương đối. Cả 130.692 file
(56.046 ảnh + 56.046 nhãn + 18.600 ignore) đều là hardlink cùng ổ, không file nào phải sao chép. Mỗi dòng manifest BDD
giữ nguyên giá trị mọi cột hợp đồng của manifest `bdd5_v1` (kể cả `frame_idx` 0-based, bước 5). Hai cột thêm là
`nguon=bdd` và `dat_file=hardlink`.

**UA-DETRAC**:
- Frame: thuộc tính `num` của `<frame>` trong XML, bằng số trong `imgNNNNN.jpg` và bắt đầu từ 1. Chỉ giữ frame có
  `num % 10 == 0` **và** có `<frame>` trong XML. Ảnh không có `<frame>` bị loại. Cột `frame_idx` của dòng UA-DETRAC
  ghi đúng `num`. Tên ảnh xuất có dạng `<video>_imgNNNNN.jpg` để không trùng tên giữa các video.
- Lớp: `car→car`, `van→car`, `bus→bus`, `others→truck`. `van→car` và `others→truck` là **giả định** theo quyết định
  của người dùng. Gặp `vehicle_type` khác thì báo lỗi, không lặng lẽ bỏ qua. Trên tập đã xuất không gặp trường hợp nào.
- Tô xám: `ignored_region` của video áp cho mọi frame của video đó. Xuất bằng **chính** `bx.xuat_mot_anh`: vùng được
  đưa vào dưới vai trò `to_xam` (lớp `other vehicle`, `ign=True`), tô màu (114,114,114) và trừ pixel nằm trong box có
  nhãn. Ảnh có tô xám được ghi JPEG q95; tọa độ vùng ghi vào `ignore/<split>/<ảnh>.txt` (`x1 y1 x2 y2` px).
  `n_vung_to_xam` trong manifest khai báo đúng số vùng, đúng kiểu ignore của hợp đồng. Video không có `ignored_region`
  thì ảnh được sao chép nguyên byte và không có file ignore.
- Ảnh âm tính: frame có `<frame>` nhưng sau lọc còn 0 box hợp lệ vẫn được giữ, với file nhãn rỗng. Đây là quy tắc
  "frame đã gán nhãn" của BDD. Trên dữ liệu thực tế có **0 ảnh như vậy**, vì XML không có frame rỗng.
- Box: cắt vào khung; box còn < 1 px sau khi cắt thì bỏ (`_hop_le_box` của BDD). Box XML không hữu hạn hoặc có w, h ≤ 0
  thì bỏ và đếm riêng.

## 3. Lệnh đã chạy (theo thứ tự)

Python: `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -X utf8`.

| # | Lệnh | Exit | Kết quả chính |
|---|---|---|---|
| 1 | `src/mixed5/xuat_mixed5.py --buoc chia` | 0 | Split: train-fit 1.309 / dev 151; UA-DETRAC 48 / 12 |
| 2 | `… --buoc ke-hoach` | 0 | 8.187 frame UA-DETRAC (train 6.556 / dev 1.631); rò rỉ 0 (mục 5); JPEG thiếu EOI 0/0; ổ trống 45,0 GB; 46,9 s |
| 3 | `… --buoc xuat-bdd` | 0 | 130.692 hardlink, 0 sao chép; 229 s |
| 4 | `… --buoc xuat-detrac --gioi-han-giay 500` | 0 | 60/60 video trong một lượt; 137,5 s |
| 5 | `… --buoc manifest` | 0 | 64.233 dòng, sha256 `anh.csv` `f2df97da…7c13`; rò rỉ lần 2 bằng 0 |
| 6 | `… --buoc kiem` (chạy nền trong phiên) | 0 | Hợp đồng `loi: 0`; các kiểm bổ sung 0 lệch (mục 4); 181,6 s |
| 7 | `… --buoc thong-ke` | 0 | Mục 6 |
| 8 | `… --buoc overlay` | 0 | 5 ảnh |
| 9 | `src/mixed5/test_mixed5.py` | 0 | 14/14 OK |

Bước `ke-hoach` chọn frame và băm ảnh nguồn, rồi kiểm rò rỉ **trước khi ghi bất kỳ ảnh nào**. Nó cũng đếm JPEG thiếu
EOI (`FF D9`). Theo `ultralytics/data/utils.py:234-238`, gặp JPEG thiếu EOI thì ultralytics ghi đè file. Nếu file đó
là hardlink, ghi đè sẽ làm đổi luôn file trong `bdd5_v1`. Đếm được 0 ảnh BDD và 0 ảnh UA-DETRAC thiếu EOI, nên điều kiện
ghi đè không xảy ra với dataset này.

## 4. Hợp đồng dataset

Lệnh `C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -X utf8 src/mixed5/xuat_mixed5.py --buoc kiem`
(chạy nền trong phiên, đã kết thúc) **exit 0**. Kết quả đầy đủ: `results/phase2/mixed5/kiem_hop_dong.json`.

`bx.kiem_hop_dong_dataset(data/yolo_mixed5_v1/data.yaml)` (hàm gốc, không sửa) trả về **`loi: 0`**, mất 170 s.
Danh tính dataset:

| Trường | Giá trị |
|---|---|
| `hop_dong` / manifest | `bdd5-dataset-v2` / phiên bản 2, `ten` `mixed5_v1`, `loai` `day_du`, `tap_con` null |
| `sha256_data_yaml` | `d7828268b4691d10e1a334bfb6aa05c82d0a533e53e9fd022cbfcb7412f16285` |
| `sha256_tom_tat` | `005390d6faace8b6debb7c3117c813ae225ace156863f82287a86fd51c140420` |
| `manifest_sha256` (`anh.csv`) | `f2df97dafd4c451776f8c832cfd244096a2934d20ff20d1ae19a53826fbd7c13` |
| `sha256_video_json` | `1eac1db1805d4a34e6e8bfd137a763be932c8a69cf33954f753179aa29ca6a41` |
| split đã khóa | sha256 `0acc018ba0fc5506b52d20dbffb1b68df5cabd1971dfb26815dcbe642c8f664b`, `sha256_danh_sach` `c1b38e648f40ca2e90b7c675a50ea2bcaaf5a9bc24ceab4554416a8d6ebb0597`, bản gốc `khop` |
| train / val resolve | `data/yolo_mixed5_v1/images/train` / `images/dev` |
| ảnh | train 57.043, dev 7.190; video 1.309 / 151 |
| nhãn rỗng hợp lệ | train 686, dev 95 (đều là BDD) |
| ảnh có file ignore | train 22.853, dev 3.014 |
| file đã băm | 64.233 ảnh, 64.233 nhãn, 25.867 ignore; `.npy` cạnh ảnh: 0 |

Các kiểm bổ sung trong cùng lệnh:
- **BDD trùng byte với `bdd5_v1`**: 56.046 dòng BDD của manifest mới có giá trị mọi cột hợp đồng (gồm `sha256_ra`,
  `sha256_nhan`, `sha256_ignore`) bằng đúng dòng tương ứng của manifest `bdd5_v1` (sha256 `1ee6174f…6069`, được kiểm
  trước khi đọc). Lệch: 0. Hợp đồng vừa băm lại từng file theo manifest mới, nên mỗi file BDD trong dataset trộn trùng
  byte với sha256 đã khóa của `bdd5_v1`. Cả 56.046 ảnh đều `os.path.samefile` với file nguồn (hardlink).
- **UA-DETRAC tính lại từ XML**, độc lập với bước xuất: 60 video, 8.187 frame. Tập frame bằng đúng
  `{num có <frame>, num % 10 == 0}`: 0 frame thiếu hoặc thừa. 0 frame sai split. Số box theo lớp tính lại bằng ánh xạ +
  cắt khung: lệch 0 frame. Số vùng ignore hợp lệ: lệch 0 frame.
- Rò rỉ lần 3 trên manifest cuối: xem mục 5.

## 5. Rò rỉ và trùng lặp

Bảng sha256 tập đánh giá: `data/interim/hash_tap_danh_gia/detrac_test_sha256.csv` (56.340 ảnh, 40 video) và
`bdd100k_mot_val_sha256.csv` (39.973 ảnh, 200 video). Tên video test UA-DETRAC lấy cả từ
`DETRAC-Test-Annotations-XML` (40 XML). Đã kiểm ba lần: trước khi ghi (`ke-hoach`), khi ghi manifest, và trên manifest
cuối (`kiem`).

| Tập cấm | sha256 ảnh nguồn trùng | sha256 ảnh xuất trùng | video trùng | đường dẫn ảnh nguồn trùng |
|---|---:|---:|---:|---:|
| UA-DETRAC test (40 video) | 0 | 0 | 0 | 0 |
| BDD100K MOT val (200 video) | 0 | 0 | 0 | 0 |

Train/dev nội bộ: 0 video trùng (dev12 không nằm trong train); 0 trùng sha256 ảnh nguồn; 0 trùng sha256 ảnh xuất;
0 trùng tên ảnh xuất. Trong cùng một split cũng có 0 sha256 ảnh nguồn lặp. Test `test_split_mixed5_v1` xác nhận 40 XML
test không giao với video nào của split.

## 6. Thống kê (box theo lớp × nguồn × split)

Nguồn: `results/phase2/mixed5/thong_ke.md` (sinh từ `manifest/tom_tat.json`).

| nguồn | split | video | ảnh | âm tính | ảnh tô xám | car | truck | bus | motorcycle | bicycle | tổng box |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bdd | train | 1261 | 50487 | 686 | 17040 | 383387 | 27358 | 10708 | 2235 | 4747 | 428435 |
| bdd | dev | 139 | 5559 | 95 | 1560 | 42211 | 2971 | 984 | 226 | 407 | 46799 |
| detrac | train | 48 | 6556 | 0 | 5813 | 44477 | 281 | 2869 | 0 | 0 | 47627 |
| detrac | dev | 12 | 1631 | 0 | 1454 | 10993 | 83 | 485 | 0 | 0 | 11561 |
| tổng | train | 1309 | 57043 | 686 | 22853 | 427864 | 27639 | 13577 | 2235 | 4747 | 476062 |
| tổng | dev | 151 | 7190 | 95 | 3014 | 53204 | 3054 | 1469 | 226 | 407 | 58360 |

| nguồn | split | ảnh có car | ảnh có truck | ảnh có bus | ảnh có motorcycle | ảnh có bicycle |
|---|---|---:|---:|---:|---:|---:|
| bdd | train | 49530 | 16882 | 8160 | 1822 | 3598 |
| bdd | dev | 5407 | 1957 | 780 | 205 | 334 |
| detrac | train | 6473 | 280 | 2078 | 0 | 0 |
| detrac | dev | 1631 | 79 | 477 | 0 | 0 |
| tổng | train | 56003 | 17162 | 10238 | 1822 | 3598 |
| tổng | dev | 7038 | 2036 | 1257 | 205 | 334 |

- Tỉ lệ ảnh UA-DETRAC trong train là **11,49 %** (6.556 / 57.043); trong dev là 22,68 % (1.631 / 7.190).
- Phần BDD trùng đúng `tom_tat.json` của `bdd5_v1` (50.487 / 5.559 ảnh, box từng lớp bằng nhau).
- Frame UA-DETRAC (60 video): 82.085 frame XML / 83.791 ảnh nguồn. Có 1.706 ảnh không có `<frame>`, trong đó 177 ảnh
  thỏa bước 10 nhưng bị loại theo quy tắc. 73.898 frame XML không thỏa bước. Chọn 8.187 frame. Tổng 244 ignored_region.
- Nhãn UA-DETRAC: 4.507 box bị cắt vào khung; 0 box < 1 px; 0 box XML hỏng; 0 vùng ignore rỗng.
  7.267 / 8.187 ảnh có tô xám.
- Dung lượng: BDD 7,46 GB, toàn bộ là hardlink nên **0 GB mới** trên đĩa. UA-DETRAC 0,833 GB mới (23.641 file).
  Ổ D còn 44,0 GB sau khi xuất.
- Không đếm riêng `van` / `others` trước khi gộp. Số box `truck` UA-DETRAC (281 train, 83 dev) đều đến từ `others`.

## 7. Overlay

`results/phase2/mixed5/overlay/`, Git bỏ qua. Gồm 3 ảnh UA-DETRAC train (có tô xám; có truck; không tô xám), 1 ảnh
dev12 có tô xám, và 1 ảnh BDD train có tô xám. Đã xem trực tiếp 2 ảnh. Ở `MVI_40241_img00730`, một xe dáng van mang
nhãn car, xe tải thùng mang nhãn truck, và các ignored_region đã xám. Box có nhãn không bị tô. Đây là nhận xét bằng
mắt; chưa đối chiếu `vehicle_type` của từng box. Phép đối chiếu toàn bộ với XML nằm ở mục 4.

## 8. Test (`src/mixed5/test_mixed5.py`, 14/14 OK, exit 0)

- Ánh xạ lớp: đúng 4 khóa theo quyết định; `vehicle_type` lạ hoặc thiếu thì báo lỗi; box XML hỏng được đếm.
- Frame và bước: `num % 10`, frame không có `<frame>` bị loại, frame XML thiếu ảnh thì báo lỗi, tên ảnh không trùng.
- Xuất UA-DETRAC (fixture): pixel vùng ignore xám, pixel trong box có nhãn giữ nguyên, nhãn bị cắt khung đúng
  tọa độ, box < 1 px bị bỏ, frame âm tính có file nhãn rỗng, video không có vùng ignore được sao chép nguyên byte.
- Rò rỉ: 7 kiểu lỗi (sha256 test, sha256 val, video test, ảnh nguồn test, train/dev cùng sha256, cùng tên ảnh, ảnh dev
  thuộc video train) cộng trường hợp dev12 lọt vào train đều bị chặn; trường hợp sạch thì qua.
- Split: quy tắc XML − dev bằng train; sửa tay danh sách thì bị phát hiện; dev ngoài tập XML hoặc sai số lượng thì
  báo lỗi.
- Hợp đồng: dataset mini (BDD hardlink + UA-DETRAC) đi qua `bx.kiem_hop_dong_dataset`; sửa một file nhãn thì hợp đồng
  báo lỗi.
- Split thật (chỉ đọc): 60 XML, dev12 khớp, 48 train, phần BDD 1.261 / 139 khớp `bdd5_train_v1`, 40 XML test rời.

## 9. Giới hạn

- `van→car`, `others→truck` là giả định. `others` gồm nhiều loại xe; một phần có thể không phải truck theo định nghĩa
  BDD.
- UA-DETRAC không gán nhãn xe máy hay xe đạp. Có 8.187 ảnh có thể chứa xe hai bánh chưa gán nhãn, tức âm tính giả cho
  lớp 3–4. Thiệt hại sẽ đo bằng AP motorcycle/bicycle trên BDD val ở bước đánh giá sau (theo quyết định người dùng),
  chưa đo ở bước này.
- Nhãn UA-DETRAC có thể sót xe ở mép khung hoặc ở xa. Overlay `MVI_40241_img00730` có một xe ở mép phải dưới không có
  box. Bước này không sửa nhãn gốc.
- Dev nội bộ gồm 22,7 % ảnh UA-DETRAC, cao hơn tỉ lệ trong train (11,5 %), nên mAP dev chọn checkpoint không cùng phân
  bố với train.
- Hardlink: ảnh và nhãn BDD trong `yolo_mixed5_v1` dùng chung inode với `yolo_bdd5_v1`. Ghi đè tại chỗ bất kỳ file
  nào ở một bên sẽ đổi luôn bên kia. Đã kiểm 0 JPEG thiếu EOI (điều kiện ultralytics ghi lại ảnh). Hợp đồng của mỗi
  dataset vẫn bắt được thay đổi nội dung. Quy tắc 7 trong docstring `bdd_yolo_export.py` ghi "không hard link" cho
  bước xuất BDD gốc; ở đây làm theo plan R52 đã duyệt ("ưu tiên hardlink").
- Cột `frame_idx` mang hai quy ước: BDD là 0-based, UA-DETRAC là `num` 1-based. Quy ước được ghi trong `tom_tat.json`.
- `git_head` trong manifest là `117d62d`. Mã `src/mixed5/` chưa được commit (chỉ có trong cây làm việc).
