# Manifest tải ảnh BDD100K MOT 2020 train (vòng 31)

Nguồn `http://dl.yf.io/bdd100k/mot20/`. Lưu ở `data/raw/downloads/bdd100k_mot20/`. Mỗi file đã kiểm:

- HTTP 200, Content-Type `application/zip`;
- kích thước bằng Content-Length;
- 4 byte đầu `PK\x03\x04`;
- md5 khớp file `.md5` của máy chủ;
- `zipfile.testzip()` OK.

| File | Byte | md5 (= máy chủ) | sha256 | Ảnh | Video |
|:--|--:|:--|:--|--:|--:|
| `images20-track-train-1.zip` | 5.172.007.826 | `d9e1e26b987d6a5c014179748199a052` | `345abb981fc55ce7d6368f2e6753b8b6747caa4b49a8abd7851a87aeb2fa1acc` | 39.418 | 200 |
| `images20-track-train-2.zip` | 4.745.249.485 | `dd914ca2e38ca64790f2bac408e0fc88` | `6870a08c0bca721a53562febcda1ecfecb896ddd759558c29727f580b9dbcb39` | 39.990 | 200 |
| `images20-track-train-3.zip` | 4.733.781.107 | `ed5016797347c61b979fddf37dee56ba` | `a83571d3c8fe049ba6960147dcb53ea1bbd7e048b9ed82635eedc6a6a04a47a3` | 39.657 | 200 |
| `images20-track-train-4.zip` | 4.991.884.651 | `c0682738fe7ce99256a621ea7ab7d4c3` | `fa91bbb451b7a12b3fdfc270fcecd4a36d63e877c1a6cc797003916abf341a9f` | 39.354 | 200 |
| `images20-track-train-5.zip` | 4.875.311.401 | `f719896aecbf82fb01d19d03b861d318` | `ee00bbb00508e1de1fcfda0ff1637d397ba207bb9d4a213a5b1ccc422a8a0822` | 39.743 | 200 |
| `images20-track-train-6.zip` | 4.621.294.568 | `a0f809cbcb4ddbb9de12af83722005d7` | `96d0e2b1aa2d6bc3fad829d2c00731b44c913ee7909e5d6eeeff448a33e4f0e7` | 39.459 | 200 |
| `images20-track-train-7.zip` | 4.763.456.582 | `e9a735938ed852ea7621071269520d70` | `30a49e77585b5e51905f0b70009a22dd6f499fb6849d72829c91384cdbd37fea` | 40.458 | 200 |
| **Tổng 7** | **33.902.985.620** | | | **278.079** | **1.400** |

URL từng file: `http://dl.yf.io/bdd100k/mot20/<File>`; md5 máy chủ: `<URL>.md5`. Chi tiết trong `20_manifest_tai_anh_train.json`.
