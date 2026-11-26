# ke hoach fixture
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_detrac_test.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/detrac_test/yolo11n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolov8n_coco=models/yolov8n.pt --sha256 yolov8n_coco=f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolov8n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolov8n_day_du/weights/best.pt --sha256 yolov8n_mixed5=542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694 --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe -u src/danh_gia_detector_bdd_val.py --model yolo11n_mixed5=models/finetune_mixed5/train_mixed5_v1_yolo11n_day_du/weights/best.pt --sha256 yolo11n_mixed5=9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb --batch 16 --device 0 --conf-van-hanh 0.25 --gioi-han-anh 0 --out results/phase2/r53_detector/full/bdd_val/yolo11n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap detrac_test --model yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap detrac_test --model yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap detrac_test --model yolo11n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap bdd_val --model yolov8n_coco
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap bdd_val --model yolov8n_mixed5
C:/Users/Mikuno/anaconda3/envs/khoaluan-mot/python.exe src/r53_detector/chuan_bi_danh_gia.py --buoc smoke --tap bdd_val --model yolo11n_mixed5
f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36
542b1381c33a673531bccdbca769f26408a16e9ba6ed89e49238a0f16b9b0694
9a5e34a12c1d97cd1b0e49325d2eadbd443a965633b464461980df91fea386fb
bb4b497b4f89a4464777144a35d66cbd109d03d3c485c4176cde0b67e358ccb4
de2c06f4db2eb99c15b44d873cc63b6b6b446bc632000701671dedd65677020a
30a1fd5a63641f1cebc22e4e7ef7e4600a48a45ed6f7d533bf319af0fa156f7f
7bba3439987377cb2f161eb31e2cf0afafae3e814f165279087b7a3780699c3e
37b62da99b77880a27aa234a887158e51498b6822921b430c1366d271f621255
1ca964b0fc81701ac746335fa20ac6e5231322a4cdd11ad47753a2b25fc31493
37e4f56247b7be0896c29d74026b863435b2eec5f53c11fc1a8cd073da70d88b
3ab1a14891d0b7ecc4d9c2f78f2f6678d9738378daada08d4f51c005e2095e3f
806980c6c3401618cda90e2be9aec0169ede0fa24896cc10cd7bd14b101ec2f3
