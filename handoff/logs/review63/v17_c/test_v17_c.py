"""R63 v17-C -- kiem tren PC phan sua engine chu nhat cua `nano_detector.py` (ban v16 + v17_c.diff).

Chay tu goc repo (can models/yolov8n.pt, data/extracted/DETRAC-Images/MVI_39031, torch + ultralytics):
    python handoff/logs/review63/v17_c/test_v17_c.py
Nap `v17_c/src/nano_detector.py` (khong phai src/ cua HEAD, vi HEAD co LUT R62).

Cac phep kiem:
  1. imgsz_hieu_luc / mo_ta_letterbox: engine vuong 640 -> giu 640 (int) va chuoi v16; 384x640 -> (384, 640).
  2. letterbox((384,640)) PIXEL-IDENTICAL voi ultralytics LetterBox((384,640), auto=False) tren 3 frame 960x540;
     vien dem 12 hang tren + 12 hang duoi.
  3. letterbox((384,640)) == LetterBox((640,640), auto=True, stride=32) -- tensor ma duong PC `predict(imgsz=640)`
     dung cho R53-R61 (rect) => ung vien C1 tren Nano nhan cung anh vao nhu danh gia PC.
  4. NanoDetector(TorchPtInfer FP32, (384,640)) tren 5 frame == ultralytics model.predict(imgsz=640, conf=0.05,
     iou=0.7, max_det=3000) FP32: cung so box, bbox |d| <= 1e-3, conf |d| <= 1e-5, cls bang nhau.
  5. TRTEngine.__call__ tu choi tensor sai shape (640x640 vao engine 384x640) truoc khi memcpy.
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SMOKE = ROOT / "data" / "extracted" / "DETRAC-Images" / "MVI_39031"
sys.path.insert(0, str(ROOT / "src"))                       # eval_detector_5lop (test 7); nano_detector van nap tu v17_c/src
spec = importlib.util.spec_from_file_location("nano_detector", Path(__file__).resolve().parent / "src" / "nano_detector.py")
nd = importlib.util.module_from_spec(spec)
sys.modules["nano_detector"] = nd
spec.loader.exec_module(nd)

KQ = []


def kiem(ten, ok, chi_tiet=""):
    KQ.append(bool(ok))
    print("  [%s] %s %s" % ("PASS" if ok else "FAIL", ten, chi_tiet))


def main():
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    kiem("1. imgsz_hieu_luc/mo_ta_letterbox",
         nd.imgsz_hieu_luc(640, (640, 640)) == 640 and nd.imgsz_hieu_luc(640, (384, 640)) == (384, 640)
         and nd.mo_ta_letterbox(640) == "auto=False 640x640 (ONNX static)"
         and nd.mo_ta_letterbox((384, 640)) == "auto=False 384x640 (ONNX static)")

    frames = [cv2.imread(str(SMOKE / ("img%05d.jpg" % i))) for i in (1, 150, 300)]
    ok2 = ok3 = True
    for img in frames:
        lb, r, pad = nd.letterbox(img, (384, 640))
        ref = LetterBox((384, 640), auto=False)(image=img)
        ref_rect = LetterBox((640, 640), auto=True, stride=32)(image=img)
        ok2 &= lb.shape == (384, 640, 3) and np.array_equal(lb, ref) and pad == (0, 12)
        ok2 &= bool((lb[:12] == 114).all() and (lb[-12:] == 114).all())
        ok3 &= ref_rect.shape == lb.shape and np.array_equal(lb, ref_rect)
    kiem("2. letterbox((384,640)) pixel-identical LetterBox auto=False, dem 12+12 hang", ok2)
    kiem("3. letterbox((384,640)) == LetterBox((640,640), auto=True, stride=32) (duong PC rect)", ok3)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    infer = nd.TorchPtInfer(str(ROOT / "models" / "yolov8n.pt"), device=dev)
    det = nd.NanoDetector(infer, nd.imgsz_hieu_luc(640, (384, 640)), 0.05, 0.7, 3000, None, False)
    model = YOLO(str(ROOT / "models" / "yolov8n.pt"))
    ok4 = True
    for i in (1, 2, 3, 150, 300):
        img = cv2.imread(str(SMOKE / ("img%05d.jpg" % i)))
        a, _ = det.detect(img)
        res = model.predict(img, imgsz=640, conf=0.05, iou=0.7, max_det=3000, device=dev, half=False, verbose=False)[0]
        b = np.concatenate([res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()[:, None],
                            res.boxes.cls.cpu().numpy()[:, None]], 1).astype(np.float32)
        same = len(a) == len(b)
        if same and len(a):
            db, dc = np.abs(a[:, :4] - b[:, :4]).max(), np.abs(a[:, 4] - b[:, 4]).max()
            same = db <= 1e-3 and dc <= 1e-5 and np.array_equal(a[:, 5], b[:, 5])
        else:
            db = dc = 0.0
        ok4 &= bool(same)
        print("         frame %d: n=%d/%d bbox|d|=%.2e conf|d|=%.2e" % (i, len(a), len(b), db, dc))
    kiem("4. NanoDetector (384,640) == ultralytics predict(imgsz=640) FP32 (rect)", ok4)

    # 6-7 (bo sung 01/11 theo review GPT): anh BDD 1280x720 va tensor hieu luc cua duong AP (eval_detector_5lop)
    bdd_dir = ROOT / "data" / "raw" / "bdd100k" / "images" / "track" / "train" / "0000f77c-cb820c98"
    bdd = cv2.imread(str(sorted(bdd_dir.glob("*.jpg"))[0]))
    lb, _, pad = nd.letterbox(bdd, (384, 640))
    ok6 = (bdd.shape[:2] == (720, 1280) and lb.shape == (384, 640, 3) and pad == (0, 12)
           and np.array_equal(lb, LetterBox((384, 640), auto=False)(image=bdd))
           and np.array_equal(lb, LetterBox((640, 640), auto=True, stride=32)(image=bdd))
           and np.array_equal(nd.letterbox(bdd, 640)[0], LetterBox((640, 640), auto=False)(image=bdd)))
    kiem("6. BDD 1280x720: letterbox (384,640) == LetterBox auto=False == rect stride 32; 640 vuong == auto=False", ok6)

    import eval_detector_5lop as ev
    from ultralytics.data import augment as aug
    hinh = []
    goc_call = aug.LetterBox.__call__

    def ghi(self, labels=None, image=None):
        r = goc_call(self, labels, image)
        hinh.append((r if image is not None else r["img"]).shape)
        return r
    aug.LetterBox.__call__ = ghi
    ok7, mo_ta = True, {}
    tron = [sorted(bdd_dir.glob("*.jpg"))[0], SMOKE / "img00001.jpg"]          # batch TRON 1280x720 + 960x540
    try:
        for hd, ky_vong in (("384x640", {(384, 640, 3)}), ("640x640", {(640, 640, 3)}), ("mac_dinh", {(640, 640, 3)})):
            qt = json.loads(json.dumps(ev.QUY_TAC))
            assert ev.ap_dung_hinh_dang(qt, hd) is None
            hinh.clear()
            ev.du_doan_ultralytics(str(ROOT / "models" / "yolov8n.pt"), tron, qt["du_doan"], batch=2, device=dev)
            mo_ta[hd] = sorted(set(hinh))
            ok7 &= set(hinh) == ky_vong
    finally:
        aug.LetterBox.__call__ = goc_call
    kiem("7. tensor hieu luc du_doan_ultralytics, batch tron BDD+UA-DETRAC: 384x640 / 640x640 co dinh; mac_dinh batch "
         "tron -> vuong 640 (hanh vi cu)", ok7)
    print("         ", mo_ta)

    eng = nd.TRTEngine.__new__(nd.TRTEngine)
    eng.in_shape, eng.in_dtype = (1, 3, 384, 640), np.dtype(np.float32)
    try:
        eng(np.zeros((1, 3, 640, 640), np.float32))
        ok5 = False
    except ValueError:
        ok5 = True
    kiem("5. TRTEngine tu choi tensor 640x640 vao engine 384x640", ok5)
    print("KET QUA: %d/%d PASS" % (sum(KQ), len(KQ)))
    return 0 if all(KQ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
